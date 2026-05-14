"""
Order execution and position management.

Optimizations vs original:
  - Module-level instrument cache (single fetch per exchange per session).
  - NFO instruments pre-trimmed to the underlying root + expiry pre-parsed.
  - Atomic state writes; reconcile no longer clears state on a positions() failure.
  - Idempotent SL-M placement (looks up existing SL-Ms before placing a new one).
  - Debounced trailing-SL modifications (only modifies when the trigger meaningfully moves).
  - Parallelised independent LTP fetches via asyncio.gather.
  - Defensive LTP via infra.safe_ltp; tick-size-aware rounding.
  - Retry-with-backoff for transient kite NetworkException on order placement.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import time
from typing import Optional

import aiohttp
import pandas as pd
import pandas_ta_classic as ta
from kiteconnect import KiteConnect, exceptions

from infra import (
    append_iv_snapshot,
    atomic_write_json,
    compute_ivr,
    get_instruments,
    read_json,
    retry_call,
    safe_ltp,
    state_path,
    tick_round,
)
from option_chain import (
    build_chain_snapshot,
    fetch_chain_quote,
    find_atm_row,
    passes_liquidity,
    realized_vol,
    select_by_delta,
)
from rag_service import RAGService

ACTIVE_TRADE_FILE = state_path("active_trade.json")

# Order statuses we should stop polling on.
TERMINAL_STATUSES = {"COMPLETE", "REJECTED", "CANCELLED"}

# A trail-modify is only sent if the new trigger moves at least this much
# (in absolute price terms) AND at least this fraction of the previous trigger.
# Prevents rate-limit pressure from sub-tick churn.
TRAIL_MIN_MOVE_TICKS = 2          # ≥ 0.10 on a 0.05 tick instrument
TRAIL_MIN_MOVE_PERCENT = 0.5      # ≥ 0.5% of current trigger


# ---------------------------------------------------------------------------
# Per-thread KiteConnect cache + worker functions
# ---------------------------------------------------------------------------

import threading

_KITE_LOCAL = threading.local()


def _kite_worker(api_key: str, access_token: str) -> KiteConnect:
    """
    Returns a thread-local KiteConnect instance, re-using the same client across
    calls in the same worker thread instead of constructing one per order.
    """
    cached = getattr(_KITE_LOCAL, "client", None)
    cached_token = getattr(_KITE_LOCAL, "token", None)
    if cached is not None and cached_token == access_token:
        return cached
    client = KiteConnect(api_key=api_key)
    client.set_access_token(access_token)
    _KITE_LOCAL.client = client
    _KITE_LOCAL.token = access_token
    return client


_RETRYABLE_NETWORK = (exceptions.NetworkException,)


def _execute_order_sync(api_key: str, access_token: str, order_params: dict) -> Optional[str]:
    """Place an order from a worker thread. Retries network blips, fails fast on input errors."""
    try:
        kite_worker = _kite_worker(api_key, access_token)
        logging.info(f"WORKER: place_order {order_params}")
        order_id = retry_call(
            kite_worker.place_order,
            attempts=3, base_delay=0.5,
            retryable_exceptions=_RETRYABLE_NETWORK,
            **order_params,
        )
        logging.info(f"WORKER: place_order OK order_id={order_id}")
        return order_id
    except exceptions.InputException as e:
        logging.error(f"WORKER: InputException placing order: {e}")
    except exceptions.TokenException as e:
        # Re-raise so the orchestrator sees auth failure and can stop.
        logging.error(f"WORKER: TokenException placing order: {e}")
        raise
    except exceptions.PermissionException as e:
        # Re-raise: this is a Kite-app config problem (IP whitelist, API
        # permissions). Retrying every signal cycle won't help and burns API
        # quota; halt and let the operator fix the developer-console setting.
        logging.error(f"WORKER: PermissionException placing order: {e}")
        raise
    except Exception as e:
        logging.error(f"WORKER: Unexpected error placing order: {e}", exc_info=True)
    return None


def _modify_order_sync(api_key, access_token, variety, order_id, **kwargs) -> bool:
    try:
        kite_worker = _kite_worker(api_key, access_token)
        retry_call(
            kite_worker.modify_order,
            attempts=2, base_delay=0.3,
            retryable_exceptions=_RETRYABLE_NETWORK,
            variety=variety, order_id=order_id, **kwargs,
        )
        logging.info(f"WORKER: modify_order OK order_id={order_id} kwargs={kwargs}")
        return True
    except Exception as e:
        logging.warning(f"WORKER: modify_order failed for {order_id}: {e}")
        return False


def _cancel_order_sync(api_key, access_token, variety, order_id) -> bool:
    try:
        kite_worker = _kite_worker(api_key, access_token)
        retry_call(
            kite_worker.cancel_order,
            attempts=2, base_delay=0.3,
            retryable_exceptions=_RETRYABLE_NETWORK,
            variety=variety, order_id=order_id,
        )
        logging.info(f"WORKER: cancel_order OK order_id={order_id}")
        return True
    except Exception as e:
        logging.warning(f"WORKER: cancel_order failed for {order_id}: {e}")
        return False


def _order_history_sync(api_key, access_token, order_id) -> list:
    try:
        kite_worker = _kite_worker(api_key, access_token)
        return kite_worker.order_history(order_id) or []
    except Exception as e:
        logging.warning(f"WORKER: order_history failed for {order_id}: {e}")
        return []


def _orders_sync(api_key, access_token) -> list:
    """Fetches all of today's orders (used for SL-M idempotency)."""
    try:
        kite_worker = _kite_worker(api_key, access_token)
        return kite_worker.orders() or []
    except Exception as e:
        logging.warning(f"WORKER: orders() failed: {e}")
        return []


async def _wait_for_fill(api_key: str, access_token: str, order_id: str,
                         timeout_seconds: int = 30, poll_interval: float = 0.5):
    """
    Polls order_history until terminal status or timeout. Returns
    (status, average_price, filled_quantity). Faster initial poll than the
    legacy 1s — fills usually arrive in well under a second.
    """
    deadline = time.monotonic() + timeout_seconds
    last_status = "UNKNOWN"
    while time.monotonic() < deadline:
        history = await asyncio.to_thread(_order_history_sync, api_key, access_token, order_id)
        if history:
            last = history[-1]
            last_status = last.get("status", "UNKNOWN")
            if last_status in TERMINAL_STATUSES:
                completed = [
                    h for h in history
                    if h.get("status") == "COMPLETE" and h.get("average_price", 0) > 0
                ]
                avg_price = (
                    completed[-1]["average_price"] if completed
                    else last.get("average_price", 0) or 0
                )
                filled_qty = last.get("filled_quantity", 0) or 0
                return last_status, float(avg_price or 0), int(filled_qty or 0)
        await asyncio.sleep(poll_interval)
    return last_status, 0.0, 0


async def _order_status(api_key, access_token, order_id) -> Optional[str]:
    history = await asyncio.to_thread(_order_history_sync, api_key, access_token, order_id)
    if not history:
        return None
    return history[-1].get("status")


# ---------------------------------------------------------------------------
# OrderExecutionAgent
# ---------------------------------------------------------------------------

class OrderExecutionAgent:
    """Sizes, places, and confirms entry orders + their broker-side stop-loss."""

    def __init__(self, kite: KiteConnect, config: dict):
        self.kite = kite
        self.config = config
        self.flags = config["trading_flags"]
        self.strike_steps = config.get("strike_steps", {})

        # Pre-compute underlying root and trim the NFO list to this underlying only,
        # adding a python-date column so we don't reparse on every sizing call.
        self._root = self.flags["underlying_instrument"].split(" ")[0].upper()
        full_nfo = get_instruments(self.kite, "NFO")
        df = full_nfo[full_nfo["name"] == self._root].copy()
        if df.empty:
            logging.warning(
                f"NFO instrument list has no rows for root '{self._root}'. "
                f"Falling back to full NFO list (memory cost ~30K rows)."
            )
            df = full_nfo.copy()
        df["expiry_date"] = pd.to_datetime(df["expiry"]).dt.date
        # Keep just the columns we use (memory + cache friendliness).
        keep_cols = {"tradingsymbol", "name", "strike", "expiry_date",
                     "instrument_type", "lot_size", "tick_size", "instrument_token"}
        df = df[[c for c in keep_cols if c in df.columns]].reset_index(drop=True)
        self.nfo_instruments = df

        self.underlying_token = self._lookup_underlying_token(self.flags["underlying_instrument"])

        # Lazy session-scoped cache for daily bars (used only by realized-vol gate).
        self._daily_bars_cache = None
        self._daily_bars_cached_at_date = None

    # ---------- helpers ----------

    def _lookup_underlying_token(self, name: str) -> int:
        nse = get_instruments(self.kite, "NSE")
        match = nse[nse["tradingsymbol"] == name]
        if match.empty:
            raise ConnectionError(f"Underlying {name!r} not found on NSE.")
        return int(match.iloc[0]["instrument_token"])

    def _strike_step(self) -> int:
        return int(self.strike_steps.get(self._root, 50))

    def is_weekly_expiry_today(self) -> bool:
        """
        True if today is an expiry date for any of this underlying's contracts.
        Detects from the actual instruments list (robust to NSE changing the
        weekly-expiry weekday), not by hardcoding Tue/Thu.
        """
        today = datetime.date.today()
        try:
            return bool((self.nfo_instruments["expiry_date"] == today).any())
        except Exception:
            return False

    def expiry_risk_factor(self) -> float:
        """
        Returns the risk-reduction factor in effect on expiry day, or 1.0 otherwise.
        Reads from config.expiry_day_overrides (defaults: enabled, factor 0.5).
        """
        cfg = (self.config.get("expiry_day_overrides") or {})
        if not cfg.get("enable", True):
            return 1.0
        if not self.is_weekly_expiry_today():
            return 1.0
        return float(cfg.get("risk_reduction_factor", 0.5))

    def _tick_size_for(self, symbol: str) -> float:
        """Use the broker-reported tick size if present; default to 0.05 for NFO."""
        row = self.nfo_instruments[self.nfo_instruments["tradingsymbol"] == symbol]
        if row.empty or "tick_size" not in row.columns:
            return 0.05
        ts = float(row.iloc[0]["tick_size"]) or 0.05
        return ts if ts > 0 else 0.05

    def _limit_price(self, ltp: float, side: str, tick_size: float) -> float:
        slip = float(self.flags.get("limit_order_slippage_percent", 0.5)) / 100.0
        price = ltp * (1.0 + slip) if side == "BUY" else ltp * (1.0 - slip)
        return tick_round(price, tick_size)

    # ---------- entry ----------

    async def place_trade(self, direction, force_mode: bool = False):
        """Places a LIMIT entry, waits for fill, returns trade dict (no SL-M attached here).
        `force_mode=True` propagates to chain analysis so IVR / IV-RV gates are bypassed."""
        symbol, qty, lot_size = await self._get_trade_details(direction, force_mode=force_mode)
        if not symbol or not qty:
            return None

        ltp = safe_ltp(self.kite, f"NFO:{symbol}")
        if ltp is None:
            logging.error(f"Could not fetch LTP for entry pricing on {symbol}.")
            return None

        tick = self._tick_size_for(symbol)
        limit_price = self._limit_price(ltp, "BUY", tick)
        order_params = {
            "variety": self.flags["order_variety"],
            "exchange": self.kite.EXCHANGE_NFO,
            "tradingsymbol": symbol,
            "transaction_type": self.kite.TRANSACTION_TYPE_BUY,
            "quantity": qty,
            "product": self.flags["product_type"],
            "order_type": self.kite.ORDER_TYPE_LIMIT,
            "price": limit_price,
        }
        logging.info(f"ASYNC: placing LIMIT entry {order_params}")

        api_key = self.config["zerodha"]["api_key"]
        access_token = self.config["zerodha"]["access_token"]

        order_id = await asyncio.to_thread(_execute_order_sync, api_key, access_token, order_params)
        if not order_id:
            return None

        timeout = int(self.flags.get("order_fill_timeout_seconds", 30))
        status, avg_price, filled_qty = await _wait_for_fill(
            api_key, access_token, order_id, timeout
        )
        if status != "COMPLETE" or avg_price <= 0:
            logging.error(
                f"Entry order did not fill cleanly. status={status} avg={avg_price} filled={filled_qty}"
            )
            await asyncio.to_thread(
                _cancel_order_sync, api_key, access_token, self.flags["order_variety"], order_id
            )
            return None

        return {
            "order_id": order_id,
            "symbol": symbol,
            "quantity": filled_qty or qty,
            "lot_size": lot_size,
            "tick_size": tick,
            "entry_price": avg_price,
            "type": direction,
            "entry_time": datetime.datetime.now().isoformat(),
        }

    async def find_existing_sl_order(self, symbol: str) -> Optional[str]:
        """
        Returns the order_id of an open SL/SL-M sell order for `symbol`, if one exists.
        Used for idempotent SL-M placement on resume.
        """
        api_key = self.config["zerodha"]["api_key"]
        access_token = self.config["zerodha"]["access_token"]
        orders = await asyncio.to_thread(_orders_sync, api_key, access_token)
        for o in orders:
            if (o.get("tradingsymbol") == symbol
                    and o.get("transaction_type") == "SELL"
                    and o.get("order_type") in ("SL-M", "SL")
                    and o.get("status") in ("OPEN", "TRIGGER PENDING")):
                return o.get("order_id")
        return None

    async def place_stop_loss(self, symbol: str, qty: int, trigger_price: float,
                              tick_size: float = 0.05):
        existing = await self.find_existing_sl_order(symbol)
        if existing:
            logging.info(f"SL-M already present for {symbol} (order_id={existing}); reusing.")
            return existing

        sl_params = {
            "variety": self.flags["order_variety"],
            "exchange": self.kite.EXCHANGE_NFO,
            "tradingsymbol": symbol,
            "transaction_type": self.kite.TRANSACTION_TYPE_SELL,
            "quantity": qty,
            "product": self.flags["product_type"],
            "order_type": self.kite.ORDER_TYPE_SLM,
            "trigger_price": tick_round(trigger_price, tick_size),
        }
        logging.info(f"ASYNC: placing SL-M {sl_params}")
        api_key = self.config["zerodha"]["api_key"]
        access_token = self.config["zerodha"]["access_token"]
        return await asyncio.to_thread(_execute_order_sync, api_key, access_token, sl_params)

    async def get_paper_trade_details(self, direction, force_mode: bool = False):
        symbol, qty, lot_size = await self._get_trade_details(direction, force_mode=force_mode)
        if not symbol or not qty:
            return None
        ltp = safe_ltp(self.kite, f"NFO:{symbol}")
        if ltp is None:
            logging.error(f"Paper: failed to get LTP for {symbol}.")
            return None
        logging.info(f"[Paper] {direction} {symbol} qty={qty} @ {ltp}")
        return {
            "order_id": f"PAPER_{int(datetime.datetime.now().timestamp())}",
            "symbol": symbol,
            "quantity": qty,
            "lot_size": lot_size,
            "tick_size": self._tick_size_for(symbol),
            "entry_price": ltp,
            "type": direction,
            "entry_time": datetime.datetime.now().isoformat(),
        }

    # ---------- sizing ----------

    async def _fetch_daily_bars(self) -> Optional[pd.DataFrame]:
        """Cached for the session — used by the IV/RV gate."""
        today = datetime.date.today()
        if (self._daily_bars_cache is not None
                and self._daily_bars_cached_at_date == today):
            return self._daily_bars_cache
        rv_lookback = int(self.config.get("option_filters", {}).get("rv_lookback_days", 20))
        days_back = max(60, rv_lookback * 3)
        try:
            hist = await asyncio.to_thread(
                self.kite.historical_data, self.underlying_token,
                today - datetime.timedelta(days=days_back), today, "day",
            )
            df = pd.DataFrame(hist)
            if df.empty:
                self._daily_bars_cache = df
                self._daily_bars_cached_at_date = today
                return df
            df["date"] = pd.to_datetime(df["date"]).dt.date
            self._daily_bars_cache = df
            self._daily_bars_cached_at_date = today
            return df
        except Exception as e:
            logging.warning(f"Daily-bars fetch failed: {e}")
            return None

    def _candidate_symbols(self, atm_strike: float, option_type: str,
                            expiry_date, span: int = 5) -> list:
        """Returns up to (2*span+1) tradingsymbols around ATM for one option_type."""
        step = self._strike_step()
        targets = {atm_strike + i * step for i in range(-span, span + 1)}
        df = self.nfo_instruments[
            (self.nfo_instruments["strike"].isin(targets))
            & (self.nfo_instruments["instrument_type"] == option_type)
            & (self.nfo_instruments["expiry_date"] == expiry_date)
        ]
        return df["tradingsymbol"].tolist()

    async def _run_chain_analysis(self, spot: float, atm_strike: float,
                                   option_type: str, expiry_date,
                                   force_mode: bool = False):
        """
        Builds a chain snapshot, runs IV-Rank and IV/RV gates, then picks a strike
        by delta band (with offset fallback) and a liquidity check.

        Returns (symbol, lot_size, ref_price) on success, or None to abort.

        `force_mode=True` bypasses the IVR and IV/RV gates with a warning log.
        Liquidity remains enforced — bypassing it would mean trading strikes
        with 0 OI and 50% spreads, which is bad regardless of force mode.
        """
        flt = self.config.get("option_filters", {}) or {}
        rate = float(flt.get("risk_free_rate", 0.07))
        span = int(flt.get("chain_strikes_each_side", 5))

        symbols = self._candidate_symbols(atm_strike, option_type, expiry_date, span)
        if not symbols:
            logging.warning("Chain analysis: no candidate strikes around ATM. Aborting.")
            return None

        quote_payload = await asyncio.to_thread(fetch_chain_quote, self.kite, symbols)
        if not quote_payload:
            logging.warning("Chain analysis: empty quote response. Aborting.")
            return None

        today = datetime.date.today()
        dte_days = max(1, (expiry_date - today).days)
        T_years = dte_days / 365.0

        chain = build_chain_snapshot(quote_payload, self.nfo_instruments,
                                      spot, T_years, rate)
        if chain.empty:
            logging.warning("Chain analysis: snapshot DataFrame empty. Aborting.")
            return None

        # ---------- ATM IV record + IVR gate ----------
        atm_row = find_atm_row(chain, option_type, atm_strike)
        if atm_row is None or atm_row.get("iv") is None:
            # Fall back to nearest available strike for the IV reading.
            with_iv = chain[(chain["instrument_type"] == option_type) & chain["iv"].notna()].copy()
            if not with_iv.empty:
                with_iv["dist"] = (with_iv["strike"] - atm_strike).abs()
                atm_row = with_iv.sort_values("dist").iloc[0]
        atm_iv = float(atm_row["iv"]) if (atm_row is not None and atm_row.get("iv") is not None) else None

        if atm_iv:
            try:
                append_iv_snapshot(self.flags["underlying_instrument"],
                                    today.isoformat(), atm_iv, spot, atm_strike)
            except Exception as e:
                logging.debug(f"append_iv_snapshot failed (non-fatal): {e}")

            ivr_max = float(flt.get("ivr_max_for_long", 60.0))
            lookback = int(flt.get("ivr_lookback_days", 60))
            min_samples = int(flt.get("ivr_min_samples", 10))
            ivr, samples = compute_ivr(self.flags["underlying_instrument"],
                                        atm_iv, lookback, min_samples)
            if ivr is not None:
                if ivr > ivr_max:
                    if force_mode:
                        logging.warning(
                            f"FORCE-MODE: IVR gate BYPASSED. IVR={ivr:.1f} > max "
                            f"{ivr_max:.0f}. Would normally skip; proceeding under force."
                        )
                    else:
                        logging.warning(
                            f"IVR gate: today's ATM IV {atm_iv:.3f} = {ivr:.1f}IVR "
                            f"(>{ivr_max:.0f}, samples={samples}). Skipping entry."
                        )
                        return None
                else:
                    logging.info(f"IVR check: {ivr:.1f} <= {ivr_max:.0f} (samples={samples}). OK.")
            else:
                logging.info(f"IVR gate bypassed: insufficient history ({samples} samples).")

            # ---------- IV/RV gate ----------
            iv_rv_max = float(flt.get("iv_rv_max_ratio", 0))
            if iv_rv_max > 0:
                bars = await self._fetch_daily_bars()
                if bars is not None and not bars.empty:
                    closes = bars.sort_values("date")["close"].reset_index(drop=True)
                    rv = realized_vol(closes, int(flt.get("rv_lookback_days", 20)))
                    if rv and rv > 0:
                        ratio = atm_iv / rv
                        if ratio > iv_rv_max:
                            if force_mode:
                                logging.warning(
                                    f"FORCE-MODE: IV/RV gate BYPASSED. ratio={ratio:.2f} > "
                                    f"max {iv_rv_max:.2f}. Proceeding under force."
                                )
                            else:
                                logging.warning(
                                    f"IV/RV gate: IV {atm_iv:.3f} / RV {rv:.3f} = {ratio:.2f} "
                                    f"> max {iv_rv_max:.2f}. Skipping (options too expensive)."
                                )
                                return None
                        else:
                            logging.info(f"IV/RV check: {ratio:.2f} <= {iv_rv_max:.2f}. OK.")

        # ---------- Strike selection: delta-targeted with offset fallback ----------
        chosen = None
        if flt.get("use_delta_targeting", True):
            chosen = select_by_delta(
                chain, option_type,
                float(flt.get("target_delta_low", 0.40)),
                float(flt.get("target_delta_high", 0.55)),
            )
            if chosen is not None:
                logging.info(
                    f"Delta-targeted pick: {chosen['tradingsymbol']} "
                    f"strike={chosen['strike']} delta={chosen['delta']:.2f}"
                )
        if chosen is None:
            # Offset fallback (existing strike_offset_steps behaviour).
            step = self._strike_step()
            offset = int(self.flags.get("strike_offset_steps", 0))
            if offset:
                fallback_strike = (atm_strike - offset * step) if option_type == "CE" \
                    else (atm_strike + offset * step)
            else:
                fallback_strike = atm_strike
            row = chain[
                (chain["instrument_type"] == option_type)
                & (chain["strike"] == fallback_strike)
            ]
            if row.empty:
                row = chain[
                    (chain["instrument_type"] == option_type)
                    & (chain["strike"] == atm_strike)
                ]
            if row.empty:
                logging.warning("Chain analysis: no offset/ATM strike in snapshot. Aborting.")
                return None
            chosen = row.iloc[0]
            logging.info(
                f"Offset fallback pick: {chosen['tradingsymbol']} "
                f"strike={chosen['strike']} delta={chosen.get('delta')}"
            )

        # ---------- Liquidity filter on the chosen strike ----------
        ok, reason = passes_liquidity(
            chosen,
            max_spread_pct=float(flt.get("max_spread_percent", 2.0)),
            min_oi=int(flt.get("min_open_interest", 100000)),
            max_age_seconds=float(flt.get("max_quote_age_seconds", 5)),
        )
        if not ok:
            logging.warning(f"Liquidity gate: {chosen['tradingsymbol']} rejected — {reason}.")
            return None

        # Find lot_size from instruments df.
        meta = self.nfo_instruments[
            self.nfo_instruments["tradingsymbol"] == chosen["tradingsymbol"]
        ]
        if meta.empty:
            logging.warning(f"Chain analysis: lot_size not found for {chosen['tradingsymbol']}.")
            return None
        lot_size = int(meta.iloc[0]["lot_size"])

        ref_price = float(chosen["mid"]) if chosen["mid"] > 0 else float(chosen["last"])
        if ref_price <= 0:
            logging.warning(f"Chain analysis: zero reference price for {chosen['tradingsymbol']}.")
            return None

        return chosen["tradingsymbol"], lot_size, ref_price

    async def _get_trade_details(self, direction, force_mode: bool = False):
        try:
            # Fetch underlying LTP and margins concurrently — independent calls.
            underlying_key = str(self.underlying_token)
            ltp_task = asyncio.to_thread(self.kite.ltp, underlying_key)
            margins_task = asyncio.to_thread(self.kite.margins)
            ltp_data, margins = await asyncio.gather(ltp_task, margins_task)

            ltp = (ltp_data or {}).get(underlying_key, {}).get("last_price")
            if ltp is None or ltp <= 0:
                logging.error(f"Underlying LTP unavailable: {ltp_data!r}")
                return None, 0, 0

            step = self._strike_step()
            atm_strike = round(ltp / step) * step
            option_type = "CE" if direction == "BUY" else "PE"

            today = datetime.date.today()
            min_dte = int(self.flags.get("min_days_to_expiry", 0))
            valid_expiries = sorted({
                d for d in self.nfo_instruments["expiry_date"].unique()
                if (d - today).days >= min_dte
            })
            if not valid_expiries:
                logging.warning(f"No expiries with DTE >= {min_dte}. Aborting sizing.")
                return None, 0, 0
            expiry_date = valid_expiries[0]

            symbol = None
            lot_size = 0
            ref_price = 0.0

            # ---------- Chain-analysis pathway ----------
            option_filters = self.config.get("option_filters", {}) or {}
            if option_filters.get("enable", False):
                result = await self._run_chain_analysis(
                    spot=float(ltp),
                    atm_strike=float(atm_strike),
                    option_type=option_type,
                    expiry_date=expiry_date,
                    force_mode=force_mode,
                )
                if result is None:
                    # An enabled chain pipeline that refuses == skip the trade.
                    return None, 0, 0
                symbol, lot_size, ref_price = result

            # ---------- Legacy pathway (when option_filters disabled) ----------
            if symbol is None:
                offset_steps = int(self.flags.get("strike_offset_steps", 0))
                if offset_steps:
                    target_strike = (atm_strike - offset_steps * step) if option_type == "CE" \
                        else (atm_strike + offset_steps * step)
                else:
                    target_strike = atm_strike

                target = self.nfo_instruments[
                    (self.nfo_instruments["strike"] == target_strike)
                    & (self.nfo_instruments["instrument_type"] == option_type)
                    & (self.nfo_instruments["expiry_date"] == expiry_date)
                ]
                if target.empty:
                    logging.warning(
                        f"No option for {self._root} {target_strike}{option_type} expiry "
                        f"{expiry_date}; falling back to ATM {atm_strike}."
                    )
                    target = self.nfo_instruments[
                        (self.nfo_instruments["strike"] == atm_strike)
                        & (self.nfo_instruments["instrument_type"] == option_type)
                        & (self.nfo_instruments["expiry_date"] == expiry_date)
                    ]
                    if target.empty:
                        logging.warning(f"No fallback ATM either; aborting sizing.")
                        return None, 0, 0
                symbol = target.iloc[0]["tradingsymbol"]
                lot_size = int(target.iloc[0]["lot_size"])
                ref_price = safe_ltp(self.kite, f"NFO:{symbol}") or 0
                if ref_price <= 0:
                    logging.warning(f"Option LTP unavailable for {symbol}; skipping.")
                    return None, 0, 0

            # ---------- Sizing (shared by both pathways) ----------
            equity = (margins or {}).get("equity", {}).get("available", {})
            capital = (
                equity.get("live_balance")
                or equity.get("cash")
                or equity.get("net")
                or 0
            )
            if not capital or capital <= 0:
                logging.error(f"Could not determine available capital from margins: {equity}")
                return None, 0, 0

            risk_pct = float(self.flags["risk_per_trade_percent"])
            expiry_factor = self.expiry_risk_factor()
            if expiry_factor < 1.0:
                logging.info(
                    f"Expiry-day risk override: scaling risk_pct by {expiry_factor} "
                    f"({risk_pct:.2f}% -> {risk_pct * expiry_factor:.2f}%)"
                )
                risk_pct *= expiry_factor
            risk_amount = capital * (risk_pct / 100.0)

            sl_pct = float(self.flags.get("stop_loss_percent", 25.0)) / 100.0
            min_sl_pts = float(self.flags.get("min_stop_loss_points", 2.0))
            risk_per_share = max(ref_price * sl_pct, min_sl_pts)

            lots_by_risk = int(risk_amount / max(risk_per_share * lot_size, 1e-6))
            num_lots = max(1, lots_by_risk)
            quantity = num_lots * lot_size

            max_qty_by_capital = int(capital / max(ref_price, 1e-6))
            if quantity > max_qty_by_capital:
                logging.warning(f"Capping qty {quantity} -> {max_qty_by_capital} (capital cap).")
                quantity = max(lot_size, (max_qty_by_capital // lot_size) * lot_size)

            logging.info(
                f"Sizing: symbol={symbol} lot={lot_size} qty={quantity} "
                f"risk_amt={risk_amount:.0f} ref_price={ref_price:.2f} "
                f"risk_per_share={risk_per_share:.2f}"
            )
            return symbol, quantity, lot_size
        except Exception as e:
            logging.error(f"Error in _get_trade_details: {e}", exc_info=True)
            return None, 0, 0


# ---------------------------------------------------------------------------
# PositionManagementAgent
# ---------------------------------------------------------------------------

class PositionManagementAgent:
    """Monitors active trades, manages broker-side SL-M, and coordinates exits."""

    def __init__(self, kite: KiteConnect, config: dict, rag_service: RAGService):
        self.kite = kite
        self.config = config
        self.rag_service = rag_service
        self.active_trade = None
        self.cpr_pivots = {}
        self.tsl_config = self.config.get("trailing_stop_loss", {})
        self.flags = self.config["trading_flags"]
        self.api_key = self.config["zerodha"]["api_key"]

    @property
    def access_token(self):
        # Read fresh each call so token rotations are picked up.
        return self.config["zerodha"]["access_token"]

    # ---------- persistence ----------

    def _save_state(self):
        if not self.active_trade:
            return
        try:
            atomic_write_json(ACTIVE_TRADE_FILE, self.active_trade)
        except Exception as e:
            logging.warning(f"Could not persist active trade: {e}")

    def _clear_state(self):
        try:
            import os
            if os.path.exists(ACTIVE_TRADE_FILE):
                # Keep a `.bak` for forensic recovery in case clearing was a mistake.
                os.replace(ACTIVE_TRADE_FILE, ACTIVE_TRADE_FILE + ".bak")
        except Exception as e:
            logging.warning(f"Could not clear active trade file: {e}")

    def load_state(self) -> Optional[dict]:
        return read_json(ACTIVE_TRADE_FILE, default=None)

    async def reconcile_open_position(self) -> bool:
        """
        On startup, if a saved active_trade exists, verify the position is still open
        with the broker. Returns True if a position was successfully resumed.
        Critically, on a positions() API failure we retain the state file rather
        than blindly clearing it — losing recovery info is worse than retrying.
        """
        saved = self.load_state()
        if not saved:
            return False
        symbol = saved.get("symbol")
        try:
            positions = await asyncio.to_thread(self.kite.positions)
        except Exception as e:
            logging.warning(
                f"RECONCILE: positions() failed ({e}); KEEPING saved state for {symbol}. "
                f"Will retry next session."
            )
            return False

        net = positions.get("net", []) if isinstance(positions, dict) else []
        match = next(
            (p for p in net
             if p.get("tradingsymbol") == symbol and (p.get("quantity") or 0) > 0),
            None,
        )
        if match:
            self.active_trade = saved
            logging.info(f"RECONCILE: resumed open position {symbol} qty={match.get('quantity')}")
            return True
        logging.info(f"RECONCILE: persisted trade {symbol} not in open positions; clearing state.")
        self._clear_state()
        return False

    # ---------- lifecycle ----------

    def start_trade(self, trade_details):
        if not trade_details:
            return
        self.active_trade = trade_details
        self.tsl_config = self.config.get("trailing_stop_loss", {})
        sl_price, _ = self._calculate_initial_sl()
        self.active_trade["initial_stop_loss"] = sl_price
        self.active_trade["trailing_stop_loss"] = sl_price
        self.active_trade["high_water_mark"] = self.active_trade.get("entry_price", 0)
        self.active_trade.setdefault("sl_order_id", None)
        logging.info(
            f"Managing {self.active_trade['symbol']} entry={self.active_trade['entry_price']:.2f} "
            f"hard_SL={sl_price:.2f}"
        )
        self._save_state()

    async def attach_broker_stop_loss(self, order_agent: OrderExecutionAgent):
        """Place a broker-side SL-M for the active trade. Idempotent: re-uses an existing SL-M."""
        if not self.active_trade:
            return None
        sl_price = self.active_trade["initial_stop_loss"]
        tick = float(self.active_trade.get("tick_size", 0.05))
        order_id = await order_agent.place_stop_loss(
            self.active_trade["symbol"], self.active_trade["quantity"], sl_price, tick
        )
        self.active_trade["sl_order_id"] = order_id
        if order_id:
            logging.info(f"SL-M attached order_id={order_id} trigger={sl_price:.2f}")
        else:
            logging.error("Failed to attach broker SL-M; software SL is the only protection.")
        self._save_state()
        return order_id

    # ---------- monitoring ----------

    async def manage(self, is_paper_trade=False, underlying_hist_df=None,
                     sentiment_agent=None, gemini_api_key=None):
        if not self.active_trade:
            return None
        symbol = self.active_trade["symbol"]

        # 1. Was the broker SL-M already filled? That's our exit.
        sl_id = self.active_trade.get("sl_order_id")
        if not is_paper_trade and sl_id:
            status = await _order_status(self.api_key, self.access_token, sl_id)
            if status == "COMPLETE":
                logging.info(f"Broker SL-M filled for {symbol}. Recording exit.")
                return await self._finalize_exit_via_sl(
                    sl_id, underlying_hist_df, sentiment_agent, gemini_api_key
                )
            if status == "REJECTED":
                logging.error(f"Broker SL-M for {symbol} REJECTED. Falling back to software SL.")
                self.active_trade["sl_order_id"] = None
                self._save_state()

        # 2. Pull current premium for trailing/software-SL/indicator checks.
        current_price = safe_ltp(self.kite, f"NFO:{symbol}")
        if current_price is None:
            logging.warning(f"Could not fetch LTP for {symbol}; staying ACTIVE.")
            return "ACTIVE"

        # 3. Update trailing stop and (if live) modify the SL-M trigger upward.
        new_trail = self._update_premium_trailing_stop(current_price)
        if not is_paper_trade and self.active_trade.get("sl_order_id") and new_trail:
            await self._maybe_modify_broker_sl(new_trail)

        # 4. Software backstop: if no broker SL or it's stale, enforce in code.
        trail = self.active_trade.get("trailing_stop_loss")
        hard = self.active_trade["initial_stop_loss"]
        if current_price <= hard or (trail and current_price <= trail):
            logging.info(f"Software SL hit for {symbol} @ {current_price:.2f}.")
            return await self.exit_trade(
                is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
            )

        # 5. Indicator-based exit (PSAR / MA on the underlying).
        if self.tsl_config.get("use_indicator_exit") and underlying_hist_df is not None:
            if self._check_indicator_exit(underlying_hist_df):
                logging.info(f"Indicator exit triggered for {symbol}.")
                return await self.exit_trade(
                    is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
                )

        return "ACTIVE"

    # ---------- trailing / indicator exits ----------

    def _update_premium_trailing_stop(self, current_price):
        prev_trail = self.active_trade.get(
            "trailing_stop_loss", self.active_trade.get("initial_stop_loss", 0)
        )
        self.active_trade["high_water_mark"] = max(
            self.active_trade.get("high_water_mark", 0), current_price
        )
        trail_type = self.tsl_config.get("type", "NONE")
        if trail_type != "PERCENTAGE":
            return None
        pct = float(self.tsl_config.get("percentage", 15.0))
        candidate = self.active_trade["high_water_mark"] * (1 - pct / 100.0)
        new_trail = max(prev_trail or 0, candidate)
        if new_trail > (prev_trail or 0):
            self.active_trade["trailing_stop_loss"] = new_trail
            self._save_state()
            return new_trail
        return None

    def _check_indicator_exit(self, df):
        kind = self.tsl_config.get("indicator_exit_type", "NONE")
        if df is None or df.empty:
            return False
        last = df.iloc[-1]
        price = last["close"]
        side = self.active_trade["type"]

        if kind == "MA":
            period = int(self.tsl_config.get("ma_period", 9))
            col = f"ema_{period}"
            if col not in df.columns:
                df[col] = ta.ema(df["close"], length=period)
            ma = df.iloc[-1].get(col)
            if pd.isna(ma):
                return False
            if side == "BUY" and price < ma:
                return True
            if side == "SELL" and price > ma:
                return True
            return False

        if kind == "PSAR":
            step = float(self.tsl_config.get("psar_step", 0.02))
            max_af = float(self.tsl_config.get("psar_max", 0.2))
            if "psar_long" not in df.columns or "psar_short" not in df.columns:
                psar = ta.psar(df["high"], df["low"], df["close"], af=step, max_af=max_af)
                if psar is not None and not psar.empty:
                    long_col = next(
                        (c for c in psar.columns if c.startswith("PSARl_")), None
                    )
                    short_col = next(
                        (c for c in psar.columns if c.startswith("PSARs_")), None
                    )
                    if long_col:
                        df["psar_long"] = psar[long_col]
                    if short_col:
                        df["psar_short"] = psar[short_col]
            if side == "BUY":
                short_val = df.iloc[-1].get("psar_short")
                if short_val is not None and not pd.isna(short_val) and price < short_val:
                    return True
            else:
                long_val = df.iloc[-1].get("psar_long")
                if long_val is not None and not pd.isna(long_val) and price > long_val:
                    return True
            return False

        return False

    async def _maybe_modify_broker_sl(self, new_trigger: float):
        """Debounced wrapper around order modify — skip if the move is sub-tick noise."""
        order_id = self.active_trade.get("sl_order_id")
        if not order_id:
            return
        last_sent = self.active_trade.get("sl_trigger_sent", 0) or 0
        tick = float(self.active_trade.get("tick_size", 0.05))
        abs_move = new_trigger - last_sent
        rel_move = (abs_move / last_sent * 100.0) if last_sent > 0 else 100.0
        if abs_move < TRAIL_MIN_MOVE_TICKS * tick or rel_move < TRAIL_MIN_MOVE_PERCENT:
            return  # Too small to bother modifying.

        ok = await asyncio.to_thread(
            _modify_order_sync,
            self.api_key, self.access_token,
            self.flags["order_variety"], order_id,
            trigger_price=tick_round(new_trigger, tick),
            order_type=self.kite.ORDER_TYPE_SLM,
        )
        if ok:
            self.active_trade["sl_trigger_sent"] = new_trigger
            self._save_state()
            logging.info(f"SL-M trigger trailed up to {new_trigger:.2f}")

    # ---------- losing-trade post-mortem ----------

    async def analyze_losing_trade(self, trade_details, underlying_df, sentiment_agent, gemini_api_key):
        logging.info(f"Analyzing losing trade for {trade_details['Symbol']}...")
        try:
            entry_time = pd.to_datetime(trade_details["Timestamp"]) - datetime.timedelta(minutes=10)
            exit_time = pd.to_datetime(trade_details["Timestamp"])
            if underlying_df is not None and not underlying_df.empty:
                # Kite returns IST-aware timestamps; the trade_details Timestamp is naive.
                # Normalise both to naive before comparing to avoid pandas' refusal to
                # compare across tz-aware vs tz-naive types.
                df_for_window = underlying_df
                if getattr(df_for_window.index, "tz", None) is not None:
                    df_for_window = df_for_window.copy()
                    df_for_window.index = df_for_window.index.tz_localize(None)
                window = df_for_window[
                    (df_for_window.index >= entry_time) & (df_for_window.index <= exit_time)
                ]
                cols = [c for c in ["open", "high", "low", "close", "volume", "rsi"]
                        if c in window.columns]
                snapshot = window[cols].to_string() if not window.empty else "N/A"
            else:
                snapshot = "N/A"
            news_sentiment = sentiment_agent.get_market_sentiment() if sentiment_agent else "N/A"
            rag_context = self.rag_service.retrieve_context_for_loss_analysis(trade_details)
            prompt = (
                f"Analyze this losing options trade.\n\nTrade: {trade_details}\n\n"
                f"Underlying snapshot:\n{snapshot}\n\nNews sentiment at exit: {news_sentiment}\n\n"
                f"Historical context:\n{rag_context}\n\n"
                f"Give a 3-sentence rationale for the loss and one specific lesson."
            )
            api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={gemini_api_key}"
            )
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logging.error(f"Failed to analyze losing trade: {e}")
            return "Analysis failed due to an internal error."

    # ---------- exit paths ----------

    async def _finalize_exit_via_sl(self, sl_order_id, underlying_df, sentiment_agent, gemini_api_key):
        history = await asyncio.to_thread(
            _order_history_sync, self.api_key, self.access_token, sl_order_id
        )
        completed = [
            h for h in history
            if h.get("status") == "COMPLETE" and h.get("average_price", 0) > 0
        ]
        exit_price = float(completed[-1]["average_price"]) if completed else 0.0
        return await self._book_completed_trade(
            exit_price, underlying_df, sentiment_agent, gemini_api_key,
            exit_order_id=sl_order_id, exit_reason="SL_M_TRIGGERED",
        )

    async def exit_trade(self, is_paper_trade=False, underlying_df=None,
                         sentiment_agent=None, gemini_api_key=None):
        if not self.active_trade:
            return None
        trade = self.active_trade
        symbol = trade["symbol"]

        current_ltp = safe_ltp(self.kite, f"NFO:{symbol}")
        if current_ltp is None:
            current_ltp = trade.get("entry_price", 0)
        exit_price = current_ltp
        exit_order_id = None
        exit_reason = "PAPER" if is_paper_trade else "INDICATOR_OR_SOFTWARE_SL"

        if not is_paper_trade:
            sl_id = trade.get("sl_order_id")
            if sl_id:
                await asyncio.to_thread(
                    _cancel_order_sync, self.api_key, self.access_token,
                    self.flags["order_variety"], sl_id,
                )
            tick = float(trade.get("tick_size", 0.05))
            slip = float(self.flags.get("limit_order_slippage_percent", 0.5)) / 100.0
            limit_price = tick_round(current_ltp * (1 - slip), tick)
            exit_params = {
                "variety": self.flags["order_variety"],
                "exchange": self.kite.EXCHANGE_NFO,
                "tradingsymbol": symbol,
                "transaction_type": self.kite.TRANSACTION_TYPE_SELL,
                "quantity": trade["quantity"],
                "product": self.flags["product_type"],
                "order_type": self.kite.ORDER_TYPE_LIMIT,
                "price": limit_price,
            }
            logging.info(f"ASYNC: placing LIMIT exit {exit_params}")
            exit_order_id = await asyncio.to_thread(
                _execute_order_sync, self.api_key, self.access_token, exit_params
            )
            if exit_order_id:
                timeout = int(self.flags.get("order_fill_timeout_seconds", 30))
                status, avg, _ = await _wait_for_fill(
                    self.api_key, self.access_token, exit_order_id, timeout
                )
                if status == "COMPLETE" and avg > 0:
                    exit_price = avg
                else:
                    logging.error(
                        f"Exit LIMIT did not fill (status={status}); cancelling and falling back to MARKET."
                    )
                    await asyncio.to_thread(
                        _cancel_order_sync, self.api_key, self.access_token,
                        self.flags["order_variety"], exit_order_id,
                    )
                    market_params = dict(exit_params)
                    market_params.pop("price", None)
                    market_params["order_type"] = self.kite.ORDER_TYPE_MARKET
                    fallback_id = await asyncio.to_thread(
                        _execute_order_sync, self.api_key, self.access_token, market_params
                    )
                    if fallback_id:
                        status2, avg2, _ = await _wait_for_fill(
                            self.api_key, self.access_token, fallback_id, timeout
                        )
                        exit_order_id = fallback_id
                        if status2 == "COMPLETE" and avg2 > 0:
                            exit_price = avg2

        return await self._book_completed_trade(
            exit_price, underlying_df, sentiment_agent, gemini_api_key,
            exit_order_id=exit_order_id, exit_reason=exit_reason,
        )

    async def _book_completed_trade(self, exit_price, underlying_df, sentiment_agent,
                                    gemini_api_key, exit_order_id=None, exit_reason="UNKNOWN"):
        trade = self.active_trade
        pnl = (exit_price - trade["entry_price"]) * trade["quantity"] if exit_price > 0 else 0.0
        completed = {
            "Timestamp": datetime.datetime.now(),
            "OrderID": trade.get("order_id"),
            "ExitOrderID": exit_order_id,
            "ExitReason": exit_reason,
            "Symbol": trade["symbol"],
            "TradeType": trade["type"],
            "EntryPrice": trade["entry_price"],
            "ExitPrice": exit_price,
            "Quantity": trade["quantity"],
            "ProfitLoss": pnl,
            "Status": "CLOSED",
            "Strategy": trade.get("Strategy", "N/A"),
        }

        if pnl < 0 and self.flags.get("enable_gemini_loss_analysis") and gemini_api_key:
            try:
                completed["Rationale"] = await self.analyze_losing_trade(
                    completed, underlying_df, sentiment_agent, gemini_api_key
                )
            except Exception as e:
                logging.warning(f"Loss-analysis skipped: {e}")

        # Clear in-memory state, but only AFTER the dict is returned/logged by the caller.
        self.active_trade = None
        self._clear_state()
        return completed

    # ---------- sizing math ----------

    def _calculate_initial_sl(self):
        entry_price = self.active_trade.get("entry_price", 0)
        if entry_price == 0:
            return 0, 0
        sl_pct = float(self.flags.get("stop_loss_percent", 25.0))
        min_pts = float(self.flags.get("min_stop_loss_points", 2.0))
        risk_per_share = max(entry_price * (sl_pct / 100.0), min_pts)
        return entry_price - risk_per_share, risk_per_share

    def _calculate_target_price(self, risk_per_share):
        entry_price = self.active_trade.get("entry_price", 0)
        if entry_price == 0:
            return 0
        rr = float(self.flags.get("risk_reward_ratio", 2.0))
        return entry_price + (risk_per_share * rr)
