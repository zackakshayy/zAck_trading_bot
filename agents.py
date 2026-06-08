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
    compute_iv_percentile,
    estimate_options_cost,
    get_instruments,
    read_json,
    retry_call,
    risk_pct_for_capital,
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


async def _place_entry_with_retry(
    api_key: str,
    access_token: str,
    base_params: dict,
    ltp_key: str,         # "NFO:NIFTY24000C" — passed back to safe_ltp on retry
    kite,                 # live kite instance for LTP refresh
    max_retries: int = 2,
    base_slip_pct: float = 0.005,
    slip_multiplier: float = 2.0,
    timeout_per_attempt: int = 15,
) -> tuple:
    """
    Places a LIMIT BUY entry with automatic price-widening on non-fill:

      Attempt 1 : LIMIT at ltp * (1 + base_slip_pct)
      Attempt 2 : refresh LTP, LIMIT at ltp * (1 + base_slip_pct * slip_multiplier)
      …
      Final     : MARKET order after all retries are exhausted.

    Returns (status, avg_fill_price, filled_qty, order_id).
    On all failures returns ("FAILED", 0.0, 0, None).

    Rationale: widening before falling back to MARKET gives the exchange one
    more chance to fill at a known price, reducing adverse selection vs a blind
    MARKET order during thin tape (common on NIFTY options around signal time).
    """
    variety    = base_params.get("variety", "regular")
    total_tries = max_retries + 1  # initial attempt + retries

    for attempt in range(total_tries):
        # Refresh LTP on every attempt so the widened price tracks reality.
        current_ltp = safe_ltp(kite, ltp_key)
        if current_ltp is None or current_ltp <= 0:
            logging.warning(
                f"[FillRetry] LTP unavailable for {ltp_key} on attempt {attempt + 1}."
            )
            break

        slip = base_slip_pct * (slip_multiplier ** attempt)
        params = dict(base_params)
        params["price"] = tick_round(current_ltp * (1.0 + slip), 0.05)
        params["order_type"] = "LIMIT"

        logging.info(
            f"[FillRetry] attempt {attempt + 1}/{total_tries}: "
            f"LIMIT @ {params['price']:.2f}  (slip={slip * 100:.2f}%)"
        )
        order_id = await asyncio.to_thread(
            _execute_order_sync, api_key, access_token, params
        )
        if not order_id:
            continue

        status, avg, qty = await _wait_for_fill(
            api_key, access_token, order_id, timeout_per_attempt
        )
        if status == "COMPLETE" and avg > 0:
            logging.info(
                f"[FillRetry] Filled on attempt {attempt + 1}: "
                f"avg={avg:.2f} qty={qty}"
            )
            return status, avg, qty, order_id

        # Not filled — cancel before retrying.
        logging.warning(
            f"[FillRetry] attempt {attempt + 1} not filled "
            f"(status={status}); cancelling."
        )
        await asyncio.to_thread(
            _cancel_order_sync, api_key, access_token, variety, order_id
        )

        # 6C — Partial-fill guard: after a cancel the exchange may have
        # partially filled the order before cancellation was processed.
        # If so, stop the retry loop immediately to avoid oversizing.
        post_history = await asyncio.to_thread(
            _order_history_sync, api_key, access_token, order_id
        )
        partial_qty = 0
        partial_avg = 0.0
        if post_history:
            last_record = post_history[-1]
            partial_qty = int(last_record.get("filled_quantity") or 0)
            partial_avg = float(last_record.get("average_price") or 0.0)
        if partial_qty > 0:
            logging.warning(
                f"[FillRetry] Partial fill detected after cancel: "
                f"qty={partial_qty} avg={partial_avg:.2f}. "
                f"Stopping retry to avoid oversizing."
            )
            return "PARTIAL", partial_avg, partial_qty, order_id

    # All LIMIT attempts failed — last resort: MARKET.
    logging.warning(
        f"[FillRetry] All {total_tries} LIMIT attempts exhausted for "
        f"{base_params.get('tradingsymbol')}; placing MARKET order."
    )
    mkt_ltp = safe_ltp(kite, ltp_key)
    if mkt_ltp and mkt_ltp > 0:
        mkt_params = dict(base_params)
        mkt_params.pop("price", None)
        mkt_params["order_type"] = "MARKET"
        mkt_id = await asyncio.to_thread(
            _execute_order_sync, api_key, access_token, mkt_params
        )
        if mkt_id:
            s, avg, qty = await _wait_for_fill(
                api_key, access_token, mkt_id, 30
            )
            if s == "COMPLETE" and avg > 0:
                logging.info(f"[FillRetry] MARKET fallback filled: avg={avg:.2f}")
                return s, avg, qty, mkt_id

    return "FAILED", 0.0, 0, None


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

        # Signal-data token — separate from underlying_token. Indices (NIFTY 50,
        # BANKNIFTY, etc.) don't trade, so kite.historical_data on the index
        # token returns volume=0 on every bar, which breaks every volume-using
        # strategy. We resolve the nearest-expiry FUTURES token instead and use
        # THAT for signal-bar fetches. Falls back to the index if no futures
        # are listed for this underlying.
        self.signal_data_token = self._lookup_nearest_futures_token() or self.underlying_token

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

    def _lookup_nearest_futures_token(self) -> Optional[int]:
        """
        Returns the instrument_token of the nearest-expiry futures contract
        for this underlying — used as the signal-data source because indices
        (NIFTY 50, BANKNIFTY) have zero trading volume on `kite.historical_data`.

        Returns None if no futures are listed (caller falls back to the
        underlying/index token).
        """
        if self.nfo_instruments is None or self.nfo_instruments.empty:
            return None
        today = datetime.date.today()
        try:
            mask = (
                (self.nfo_instruments["instrument_type"] == "FUT")
                & (self.nfo_instruments["expiry_date"] >= today)
            )
            futures = self.nfo_instruments[mask]
            if futures.empty:
                logging.warning(
                    f"No futures listed for {self._root} — falling back to "
                    f"index token for signal data (volume will be zero)."
                )
                return None
            futures = futures.sort_values("expiry_date")
            nearest = futures.iloc[0]
            token = int(nearest["instrument_token"])
            logging.info(
                f"Signal-data source: {self._root} futures "
                f"(symbol={nearest.get('tradingsymbol', '?')}, "
                f"expiry={nearest['expiry_date']}, token={token}). "
                f"Index has zero volume; futures provide real volume bars."
            )
            return token
        except Exception as e:
            logging.warning(f"Futures-token lookup failed: {e}. Falling back to index.")
            return None

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

        Kept for backward compatibility. New code should call dte_risk_factor()
        which supersedes this with a continuous DTE-based scale.
        """
        cfg = (self.config.get("expiry_day_overrides") or {})
        if not cfg.get("enable", True):
            return 1.0
        if not self.is_weekly_expiry_today():
            return 1.0
        return float(cfg.get("risk_reduction_factor", 0.5))

    def dte_risk_factor(self, expiry_date: datetime.date) -> float:
        """
        Continuous DTE-based risk scaling.

        Rationale
        ---------
        Options lose value non-linearly as expiry approaches. Theta and gamma
        risks accelerate dramatically in the final 2 DTE. Buying options with
        0-1 DTE remaining is structurally disadvantaged unless the move starts
        immediately — so we reduce position size rather than skip the trade.

        Scale (overridable via config.dte_sizing):
          0 DTE  →  0.50  (expiry-day — theta crush + binary gamma risk)
          1 DTE  →  0.70  (next-day expiry — overnight gap risk)
          2 DTE  →  0.85  (two sessions to expiry — moderate theta drag)
          3-4 DTE→  0.95  (short but workable window)
          5+ DTE →  1.00  (normal sizing)

        Reads overrides from config.dte_sizing:
          factor_0, factor_1, factor_2, factor_3_4, factor_5_plus
        """
        today = datetime.date.today()
        dte = max(0, (expiry_date - today).days)
        cfg = (self.config.get("dte_sizing") or {})

        if dte == 0:
            factor = float(cfg.get("factor_0", 0.50))
        elif dte == 1:
            factor = float(cfg.get("factor_1", 0.70))
        elif dte == 2:
            factor = float(cfg.get("factor_2", 0.85))
        elif dte <= 4:
            factor = float(cfg.get("factor_3_4", 0.95))
        else:
            factor = float(cfg.get("factor_5_plus", 1.00))

        if factor < 1.0:
            logging.info(
                f"[DTE sizing] expiry={expiry_date} DTE={dte} "
                f"→ risk factor={factor:.2f}"
            )
        return factor

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

    # ---------- debit spread helpers ----------

    def _spread_enabled(self) -> bool:
        # HARD buy-only guarantee. A debit spread's short leg is a SOLD
        # (written) option — the only place this bot would ever sell-to-open.
        # When trading_flags.buy_only is true (the default) we never write an
        # option, so spreads are disabled regardless of debit_spread.enable.
        # The bot then only ever BUYS options (long CE for bullish, long PE for
        # bearish), with risk capped at the premium paid.
        if (self.config.get("trading_flags") or {}).get("buy_only", True):
            return False
        return bool((self.config.get("debit_spread") or {}).get("enable", False))

    def _select_spread_short_leg(
        self,
        long_symbol: str,
        direction: str,
        expiry_date,
    ) -> tuple:
        """
        Given the long-leg symbol, return (short_symbol, short_ltp) for a
        debit-spread entry, or (None, None) if the leg is unavailable.

        For a call debit spread (BUY CE): short leg is OTM — strike is HIGHER.
        For a put debit spread (SELL PE):  short leg is OTM — strike is LOWER.

        Reads config.debit_spread.spread_width_steps (default 2) to determine
        how many strike steps OTM the short leg is.
        """
        cfg = (self.config.get("debit_spread") or {})
        width_steps = int(cfg.get("spread_width_steps", 2))
        step = self._strike_step()

        # Look up the long leg to get its strike and option type.
        row = self.nfo_instruments[self.nfo_instruments["tradingsymbol"] == long_symbol]
        if row.empty:
            logging.warning(f"[Spread] Long leg {long_symbol} not found in instruments.")
            return None, None

        long_strike    = float(row.iloc[0]["strike"])
        option_type    = str(row.iloc[0]["instrument_type"])   # CE or PE

        # BUY CE spread → short CE is further OTM (higher strike).
        # SELL PE spread → short PE is further OTM (lower strike).
        if option_type == "CE":
            short_strike = long_strike + width_steps * step
        else:
            short_strike = long_strike - width_steps * step

        short_leg = self.nfo_instruments[
            (self.nfo_instruments["strike"]           == short_strike)
            & (self.nfo_instruments["instrument_type"] == option_type)
            & (self.nfo_instruments["expiry_date"]     == expiry_date)
        ]
        if short_leg.empty:
            logging.warning(
                f"[Spread] Short leg {option_type} strike={short_strike:.0f} "
                f"expiry={expiry_date} not found — falling back to naked entry."
            )
            return None, None

        short_symbol = short_leg.iloc[0]["tradingsymbol"]
        short_ltp    = safe_ltp(self.kite, f"NFO:{short_symbol}")
        if short_ltp is None or short_ltp <= 0:
            logging.warning(
                f"[Spread] LTP unavailable for short leg {short_symbol} "
                f"— falling back to naked entry."
            )
            return None, None

        return short_symbol, float(short_ltp)

    # ---------- duplicate-entry guard (6A) ----------

    async def _has_open_position(self, symbol: str) -> bool:
        """
        Returns True if the broker reports a non-zero net quantity for `symbol`.
        Called before every entry to prevent double positions caused by:
          • bot restart while a position is open but reconcile_open_position
            missed the file (e.g. state dir was wiped), OR
          • a stale reconcile that failed silently.

        Non-fatal on API failure: returns False and allows entry, so a transient
        network glitch never silently blocks a valid trade.
        """
        try:
            positions = await asyncio.to_thread(self.kite.positions)
            net = positions.get("net", []) if isinstance(positions, dict) else []
            for p in net:
                if p.get("tradingsymbol") == symbol and int(p.get("quantity") or 0) != 0:
                    logging.warning(
                        f"[DupGuard] Open position already exists for {symbol} "
                        f"(qty={p.get('quantity')}). Skipping new entry."
                    )
                    return True
            return False
        except Exception as e:
            logging.warning(
                f"[DupGuard] Could not check positions for {symbol}: {e}. "
                f"Allowing entry (non-fatal)."
            )
            return False

    # ---------- entry ----------

    async def place_trade(self, direction, force_mode: bool = False):
        """
        Places a LIMIT entry and returns a trade dict.

        If config.debit_spread.enable is True, attempts a 1x1 debit spread:
          BUY direction → buy ATM/ITM CE (long leg) + sell OTM CE (short leg).
          SELL direction → buy ATM/ITM PE + sell OTM PE.

        entry_price in the returned dict is the NET DEBIT (long fill − short fill),
        so all downstream P&L math (exit − entry) × qty remains unchanged.

        Falls back to a naked long option if the short leg is unavailable.
        `force_mode=True` propagates to chain analysis so IVR / IV-RV gates are bypassed.
        """
        symbol, qty, lot_size = await self._get_trade_details(direction, force_mode=force_mode)
        if not symbol or not qty:
            return None

        # 6A — Duplicate-entry guard: abort if broker already shows an open position.
        if await self._has_open_position(symbol):
            return None

        ltp = safe_ltp(self.kite, f"NFO:{symbol}")
        if ltp is None:
            logging.error(f"Could not fetch LTP for entry pricing on {symbol}.")
            return None

        tick       = self._tick_size_for(symbol)
        api_key    = self.config["zerodha"]["api_key"]
        access_tok = self.config["zerodha"]["access_token"]

        # ── Long leg: place with automatic price-widening retries ─────────────
        max_retries  = int(self.flags.get("max_fill_retries", 2))
        slip_mult    = float(self.flags.get("fill_retry_slippage_mult", 2.0))
        base_slip    = float(self.flags.get("limit_order_slippage_percent", 0.5)) / 100.0
        # Divide the overall timeout evenly across attempts.
        timeout_total = int(self.flags.get("order_fill_timeout_seconds", 30))
        per_attempt   = max(5, timeout_total // (max_retries + 1))

        base_long_params = {
            "variety":          self.flags["order_variety"],
            "exchange":         self.kite.EXCHANGE_NFO,
            "tradingsymbol":    symbol,
            "transaction_type": self.kite.TRANSACTION_TYPE_BUY,
            "quantity":         qty,
            "product":          self.flags["product_type"],
            # price and order_type set by _place_entry_with_retry
        }
        # BUY-ONLY hard guarantee: an entry must NEVER sell-to-open an option.
        # This assertion makes the invariant impossible to break via future edits.
        assert base_long_params["transaction_type"] == self.kite.TRANSACTION_TYPE_BUY, \
            "buy_only invariant violated: entry transaction_type must be BUY"
        long_status, long_fill, long_filled_qty, long_id = await _place_entry_with_retry(
            api_key, access_tok,
            base_params=base_long_params,
            ltp_key=f"NFO:{symbol}",
            kite=self.kite,
            max_retries=max_retries,
            base_slip_pct=base_slip,
            slip_multiplier=slip_mult,
            timeout_per_attempt=per_attempt,
        )
        if long_status != "COMPLETE" or long_fill <= 0:
            logging.error(
                f"Long-leg entry failed after all retries: "
                f"status={long_status} fill={long_fill}. Aborting."
            )
            return None

        # ── Short leg (debit spread, optional) ───────────────────────────────
        short_symbol = short_fill = short_id = None
        if self._spread_enabled():
            # Derive expiry from the long-leg instrument record.
            long_row = self.nfo_instruments[
                self.nfo_instruments["tradingsymbol"] == symbol
            ]
            expiry_date = long_row.iloc[0]["expiry_date"] if not long_row.empty else None
            if expiry_date is not None:
                short_symbol, short_ltp = self._select_spread_short_leg(
                    symbol, direction, expiry_date
                )
                if short_symbol and short_ltp:
                    short_tick  = self._tick_size_for(short_symbol)
                    short_limit = self._limit_price(short_ltp, "SELL", short_tick)
                    short_params = {
                        "variety":          self.flags["order_variety"],
                        "exchange":         self.kite.EXCHANGE_NFO,
                        "tradingsymbol":    short_symbol,
                        "transaction_type": self.kite.TRANSACTION_TYPE_SELL,
                        "quantity":         qty,
                        "product":          self.flags["product_type"],
                        "order_type":       self.kite.ORDER_TYPE_LIMIT,
                        "price":            short_limit,
                    }
                    logging.info(
                        f"[Spread] Placing LIMIT short-leg {short_params}"
                    )
                    short_id = await asyncio.to_thread(
                        _execute_order_sync, api_key, access_tok, short_params
                    )
                    if short_id:
                        s_status, s_fill, _ = await _wait_for_fill(
                            api_key, access_tok, short_id, timeout
                        )
                        if s_status == "COMPLETE" and s_fill > 0:
                            short_fill = s_fill
                            logging.info(
                                f"[Spread] Short leg filled: {short_symbol} @ {short_fill:.2f}"
                            )
                        else:
                            logging.warning(
                                f"[Spread] Short leg did not fill cleanly "
                                f"(status={s_status}); running as naked long."
                            )
                            await asyncio.to_thread(
                                _cancel_order_sync, api_key, access_tok,
                                self.flags["order_variety"], short_id,
                            )
                            short_symbol = short_fill = short_id = None

        # entry_price = net debit for a spread, or just the long fill for naked.
        entry_price = (
            long_fill - short_fill
            if (short_fill is not None and short_fill > 0)
            else long_fill
        )
        is_spread = short_symbol is not None and short_fill is not None

        trade_dict: dict = {
            "order_id":    long_id,
            "symbol":      symbol,
            "quantity":    long_filled_qty or qty,
            "lot_size":    lot_size,
            "tick_size":   tick,
            "entry_price": entry_price,
            "type":        direction,
            "entry_time":  datetime.datetime.now().isoformat(),
            "is_spread":   is_spread,
        }
        if is_spread:
            trade_dict.update({
                "spread_short_symbol":      short_symbol,
                "spread_short_entry_price": short_fill,
                "spread_short_order_id":    short_id,
            })
            logging.info(
                f"[Spread] Debit spread entered — long={symbol} @ {long_fill:.2f}, "
                f"short={short_symbol} @ {short_fill:.2f}, "
                f"net_debit={entry_price:.2f}"
            )
        return trade_dict

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

        # Debit spread paper trade.
        short_symbol = short_ltp = None
        if self._spread_enabled():
            long_row = self.nfo_instruments[self.nfo_instruments["tradingsymbol"] == symbol]
            expiry_date = long_row.iloc[0]["expiry_date"] if not long_row.empty else None
            if expiry_date is not None:
                short_symbol, short_ltp = self._select_spread_short_leg(
                    symbol, direction, expiry_date
                )

        is_spread   = short_symbol is not None and short_ltp is not None
        entry_price = (ltp - short_ltp) if is_spread else ltp
        # The order placed is always a BUY of a long option. `direction` is the
        # market view: BUY → long CALL (bullish), SELL → long PUT (bearish).
        # Label the log by the actual action to avoid "selling options" confusion.
        _opt_type = "CE" if symbol.endswith("CE") else ("PE" if symbol.endswith("PE") else "OPT")
        _view = "bullish" if direction == "BUY" else "bearish"
        logging.info(
            f"[Paper] BUY {qty} {symbol} @ {ltp:.2f}  (long {_opt_type} — {_view} view)"
            + (f" | spread short={short_symbol} @ {short_ltp:.2f} net_debit={entry_price:.2f}"
               if is_spread else "")
        )
        trade_dict = {
            "order_id":    f"PAPER_{int(datetime.datetime.now().timestamp())}",
            "symbol":      symbol,
            "quantity":    qty,
            "lot_size":    lot_size,
            "tick_size":   self._tick_size_for(symbol),
            "entry_price": entry_price,
            "type":        direction,
            "entry_time":  datetime.datetime.now().isoformat(),
            "is_spread":   is_spread,
        }
        if is_spread:
            trade_dict.update({
                "spread_short_symbol":      short_symbol,
                "spread_short_entry_price": short_ltp,
                "spread_short_order_id":    f"PAPER_SHORT_{int(datetime.datetime.now().timestamp())}",
            })
        return trade_dict

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

            # ---------- IV-percentile gate (playbook: don't buy expensive IV) ----------
            self._last_iv_percentile = None
            ivp_cfg = (self.config.get("iv_percentile") or {})
            avoid_above = float(ivp_cfg.get("avoid_above", 0) or 0)
            buy_below   = float(ivp_cfg.get("buy_below", 0) or 0)
            if avoid_above > 0:
                ivp, ivp_n = compute_iv_percentile(
                    self.flags["underlying_instrument"], atm_iv, lookback, min_samples
                )
                if ivp is not None:
                    self._last_iv_percentile = round(ivp, 1)
                    if ivp > avoid_above:
                        if force_mode:
                            logging.warning(
                                f"FORCE-MODE: IV-percentile gate BYPASSED "
                                f"({ivp:.0f} > {avoid_above:.0f})."
                            )
                        else:
                            logging.warning(
                                f"IV-percentile gate: {ivp:.0f} > {avoid_above:.0f} — options "
                                f"historically EXPENSIVE; skipping buy (samples={ivp_n})."
                            )
                            return None
                    elif buy_below > 0 and ivp > buy_below:
                        logging.info(
                            f"IV-percentile {ivp:.0f}: above preferred buy zone "
                            f"(<{buy_below:.0f}) but under the {avoid_above:.0f} avoid line — "
                            f"proceeding with caution."
                        )
                    else:
                        logging.info(f"IV-percentile check: {ivp:.0f} (buy zone). OK.")
                else:
                    logging.info(f"IV-percentile gate bypassed: insufficient history ({ivp_n}).")

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

            # Professional DTE sweet spot: 5-10 calendar days.
            # Enough time value to survive one adverse bar; enough gamma to
            # profit from a 0.5% underlying move. Fall back to nearest valid
            # expiry if no contract falls in the window (e.g. on expiry week).
            preferred_dte = [d for d in valid_expiries if 5 <= (d - today).days <= 10]
            expiry_date = preferred_dte[0] if preferred_dte else valid_expiries[0]
            logging.info(
                f"Expiry selected: {expiry_date} "
                f"({(expiry_date - today).days} DTE"
                + (" — preferred 5-10 DTE window" if preferred_dte else " — fallback to nearest")
                + ")"
            )

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

            # Base risk percentage: TIERED by live capital (playbook alignment).
            #   <₹1L → 10% | ₹1L–₹3L → 5% | >₹3L → 2%  (configurable: risk_tiers)
            # This supersedes the static risk_per_trade_percent and the
            # AGGRESSIVE/MODERATE mode override for the per-trade risk figure.
            if (self.config.get("risk_tiers")):
                risk_pct = risk_pct_for_capital(capital, self.config)
                logging.info(
                    f"[RiskTier] capital ₹{float(capital):,.0f} → risk {risk_pct:.1f}% per trade."
                )
            else:
                risk_pct = float(
                    self.flags.get("_effective_risk_pct")
                    or self.flags["risk_per_trade_percent"]
                )
            # Continuous DTE scaling supersedes the old binary expiry_risk_factor.
            dte_factor = self.dte_risk_factor(expiry_date)
            if dte_factor < 1.0:
                logging.info(
                    f"DTE risk scaling: risk_pct {risk_pct:.2f}% "
                    f"→ {risk_pct * dte_factor:.2f}% (DTE factor={dte_factor:.2f})"
                )
                risk_pct *= dte_factor

            # Professional size multiplier: progressive loss reduction × time-of-day factor.
            # Written to config by the orchestrator before calling place_trade().
            pro_multiplier = float(self.config.get('_effective_risk_pct_multiplier', 1.0) or 1.0)
            if pro_multiplier != 1.0:
                logging.info(
                    f"[ProSize] Applying size multiplier {pro_multiplier:.2f} "
                    f"to risk_pct ({risk_pct:.2f}% -> {risk_pct * pro_multiplier:.2f}%)"
                )
                risk_pct *= pro_multiplier
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

            # ── Net-edge gate: skip trades that can't clear costs ──────────────
            # Estimate round-trip cost and the BEST-CASE gross profit (the R:R
            # target). If even hitting the full target wouldn't net at least
            # `min_net_profit_inr` after costs, the trade isn't worth taking —
            # the brokerage + STT + GST would eat the move. Bypassed in force
            # mode and when transaction_costs.enable is false.
            tc_cfg = (self.config.get("transaction_costs") or {})
            min_net = float(tc_cfg.get("min_net_profit_inr", 0) or 0)
            if not force_mode and tc_cfg.get("enable", True) and min_net > 0:
                rr = float(self.flags.get("risk_reward_ratio", 2.0))
                target_px      = ref_price + risk_per_share * rr
                expected_gross = risk_per_share * rr * quantity
                est = estimate_options_cost(
                    ref_price * quantity, target_px * quantity, 2, self.config
                )
                expected_net = expected_gross - est["total"]
                if expected_net < min_net:
                    logging.warning(
                        f"Net-edge gate: {symbol} skipped — best-case net "
                        f"₹{expected_net:,.0f} (gross ₹{expected_gross:,.0f} − costs "
                        f"₹{est['total']:,.0f}) < min ₹{min_net:,.0f}. "
                        f"Position too small to beat brokerage/STT/GST."
                    )
                    return None, 0, 0

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
            # For spreads, log that we've resumed and note the short leg as well.
            short_sym = saved.get("spread_short_symbol")
            logging.info(
                f"RECONCILE: resumed open position {symbol} qty={match.get('quantity')}"
                + (f" [spread short={short_sym}]" if short_sym else "")
            )
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

        # Partial-exit state — enabled only when at least 2 lots are held so we can
        # actually split the position. With 1 lot there is nothing to split.
        pe_cfg = self.config.get('partial_exits') or {}
        lot_size = int(self.active_trade.get('lot_size', 1) or 1)
        qty = int(self.active_trade.get('quantity', 0) or 0)
        pe_eligible = pe_cfg.get('enable', False) and qty >= lot_size * 2
        self.active_trade['_pe_enabled']       = pe_eligible
        self.active_trade['_pe_original_qty']  = qty
        self.active_trade['_pe_t1_hit']        = False
        self.active_trade['_pe_t2_hit']        = False
        self.active_trade['_pe_realized_pnl']  = 0.0

        # Snapshot the underlying spot price at entry for the give-up rule
        # (detects IV crush when spot moves in favour but premium stays flat).
        try:
            underlying_name = self.flags.get('underlying_instrument', 'NIFTY 50')
            nse = get_instruments(self.kite, 'NSE')
            match = nse[nse['tradingsymbol'] == underlying_name]
            if not match.empty:
                token = str(int(match.iloc[0]['instrument_token']))
                data = self.kite.ltp(token)
                spot = float((data or {}).get(token, {}).get('last_price', 0))
                self.active_trade['_entry_spot'] = spot if spot > 0 else 0
        except Exception:
            self.active_trade.setdefault('_entry_spot', 0)

        logging.info(
            f"Managing {self.active_trade['symbol']} entry={self.active_trade['entry_price']:.2f} "
            f"hard_SL={sl_price:.2f} partial_exits={'ON' if pe_eligible else 'OFF'} "
            f"entry_spot={self.active_trade.get('_entry_spot', 0):.2f}"
        )
        self._save_state()

    async def attach_broker_stop_loss(self, order_agent: OrderExecutionAgent):
        """
        Place a broker-side SL-M for the active trade. Idempotent: re-uses an existing SL-M.

        For debit spreads the short leg already hard-caps the maximum loss to the net
        debit paid, so a separate SL-M on the long leg would race the spread logic and
        leave an orphaned short position. We therefore skip the broker SL-M for spreads
        and rely solely on the software trailing-stop and indicator exits.
        """
        if not self.active_trade:
            return None

        if self.active_trade.get("is_spread"):
            logging.info(
                "[Spread] Skipping broker SL-M — short leg already caps max loss to "
                f"net_debit={self.active_trade['entry_price']:.2f}. "
                "Software SL and indicator exits are active."
            )
            self.active_trade["sl_order_id"] = None
            self._save_state()
            return None

        sl_price = self.active_trade["initial_stop_loss"]
        tick = float(self.active_trade.get("tick_size", 0.05))
        symbol = self.active_trade["symbol"]
        qty    = self.active_trade["quantity"]

        # 6B — Retry loop: up to 3 attempts. On each REJECTED response we
        # tighten the trigger by 0.5 % so the next attempt is further away
        # from the current market price and less likely to be rejected as
        # "trigger too close to LTP" by Kite.
        _SLM_MAX_ATTEMPTS  = 3
        _SLM_TIGHTEN_PCT   = 0.005        # tighten trigger by 0.5% per retry
        _SLM_STATUS_WAIT_S = 1.0          # seconds to wait before status check

        order_id   = None
        used_trigger = sl_price

        for attempt in range(_SLM_MAX_ATTEMPTS):
            order_id = await order_agent.place_stop_loss(symbol, qty, used_trigger, tick)
            if not order_id:
                logging.warning(
                    f"[SLM-Retry] attempt {attempt + 1}/{_SLM_MAX_ATTEMPTS}: "
                    f"place_stop_loss returned None."
                )
                used_trigger = tick_round(used_trigger * (1.0 - _SLM_TIGHTEN_PCT), tick)
                continue

            # Give the exchange a moment to process before checking status.
            await asyncio.sleep(_SLM_STATUS_WAIT_S)
            sl_status = await _order_status(self.api_key, self.access_token, order_id)
            if sl_status not in ("REJECTED",):
                # TRIGGER PENDING or any non-rejected status → accepted.
                logging.info(
                    f"[SLM-Retry] SL-M accepted on attempt {attempt + 1}: "
                    f"order_id={order_id} trigger={used_trigger:.2f} status={sl_status}"
                )
                break
            # Rejected — tighten trigger and loop.
            logging.warning(
                f"[SLM-Retry] attempt {attempt + 1}/{_SLM_MAX_ATTEMPTS}: "
                f"SL-M REJECTED (trigger={used_trigger:.2f}). "
                f"Tightening by {_SLM_TIGHTEN_PCT * 100:.1f}% and retrying."
            )
            order_id = None
            used_trigger = tick_round(used_trigger * (1.0 - _SLM_TIGHTEN_PCT), tick)
        else:
            # Loop exhausted without a successful placement.
            order_id = None

        self.active_trade["sl_order_id"] = order_id
        # _slm_absent flag: the IN_POSITION loop watches this to re-attach and
        # to tighten the software-SL poll interval.
        self.active_trade["_slm_absent"] = (order_id is None)

        if order_id:
            logging.info(
                f"SL-M attached order_id={order_id} trigger={used_trigger:.2f}"
            )
        else:
            logging.error(
                f"[SLM-Retry] All {_SLM_MAX_ATTEMPTS} SL-M attempts failed for "
                f"{symbol}. Software SL is the only protection; polling faster."
            )
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
                logging.error(
                    f"Broker SL-M for {symbol} REJECTED mid-session. "
                    f"Setting _slm_absent=True so the IN_POSITION loop can "
                    f"re-attach and poll faster."
                )
                self.active_trade["sl_order_id"] = None
                self.active_trade["_slm_absent"] = True
                self._save_state()

        # 2. Pull current premium for trailing/software-SL/indicator checks.
        #    For a debit spread, current_price = long LTP − short LTP (net spread value).
        current_price = safe_ltp(self.kite, f"NFO:{symbol}")
        if current_price is None:
            logging.warning(f"Could not fetch LTP for {symbol}; staying ACTIVE.")
            return "ACTIVE"

        if self.active_trade.get("is_spread"):
            short_sym   = self.active_trade.get("spread_short_symbol")
            short_price = safe_ltp(self.kite, f"NFO:{short_sym}") if short_sym else None
            if short_price is not None:
                current_price = max(0.0, float(current_price) - float(short_price))
            # If short LTP is unavailable, fall back to long LTP only (conservative).

        # 3. Hard time exit — normally 14:00 (theta + spread widen in the last
        #    75 min). EXCEPTION (hold-the-winner): a high-conviction trending
        #    WINNER tagged hold_to_close at entry is allowed to ride to ~15:15 to
        #    capture the afternoon trend continuation, instead of being cut early.
        hold_to_close = bool(self.active_trade.get('hold_to_close'))
        in_profit = float(current_price) > float(self.active_trade.get('entry_price', 0) or 0)
        _hard_close_time = datetime.time(14, 0)
        if hold_to_close and in_profit:
            try:
                _hc = str((self.config.get('hold_to_close') or {}).get('exit_time', '15:15'))
                hh, mm = [int(x) for x in _hc.split(':')]
                _hard_close_time = datetime.time(hh, mm)
            except Exception:
                _hard_close_time = datetime.time(15, 15)
        if datetime.datetime.now().time() >= _hard_close_time:
            logging.info(
                f"Hard time exit: {datetime.datetime.now().strftime('%H:%M')} >= "
                f"{_hard_close_time.strftime('%H:%M')} — closing {symbol}"
                + (" (held the conviction winner to near-close)." if hold_to_close and in_profit
                   else " to avoid theta/spread damage.")
            )
            return await self.exit_trade(
                is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
            )

        # 4. Partial exits (T1 / T2 premium targets) — before the SL check so that
        #    a winning trade books partial profits rather than waiting for a reversal.
        if self.active_trade.get('_pe_enabled'):
            pe_result = await self._check_partial_exits(current_price, is_paper_trade)
            if pe_result == 'FULLY_EXITED':
                return await self._book_completed_trade(
                    current_price, underlying_hist_df, sentiment_agent, gemini_api_key,
                    exit_order_id=None, exit_reason='PARTIAL_EXITS_COMPLETE',
                )

        # 5. Give-up rule: underlying moved in our favour but the premium
        #    didn't respond — IV crush has already started. Exit before it
        #    accelerates. Threshold: ≥0.3% underlying move, <10% premium gain.
        if underlying_hist_df is not None and not underlying_hist_df.empty:
            entry_spot = float(self.active_trade.get('_entry_spot', 0) or 0)
            if entry_spot > 0:
                current_spot = float(underlying_hist_df.iloc[-1]['close'])
                spot_move_pct = (current_spot - entry_spot) / entry_spot * 100.0
                side = self.active_trade['type']
                favorable_move = spot_move_pct if side == 'BUY' else -spot_move_pct
                if favorable_move >= 0.3:
                    entry_px = float(self.active_trade['entry_price'])
                    expected_min_gain = entry_px * 0.10
                    actual_gain = current_price - entry_px
                    if actual_gain < expected_min_gain:
                        logging.warning(
                            f"[GiveUp] Underlying moved {favorable_move:.2f}% in favour "
                            f"but premium only gained {actual_gain:.2f} "
                            f"(expected ≥{expected_min_gain:.2f}). "
                            f"IV crush in progress — exiting."
                        )
                        return await self.exit_trade(
                            is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
                        )

        # 6. Tighten trail after 13:30 to protect intraday gains from theta drain.
        #    EXCEPTION (hold-the-winner): a high-conviction trending WINNER keeps
        #    its WIDE trail so it can ride the trend to near-close; the profit-
        #    protector indicator exit still locks the gain on a structure break.
        _late_tighten_time = datetime.time(13, 30)
        if (datetime.datetime.now().time() >= _late_tighten_time
                and not (hold_to_close and in_profit)):
            current_trail_pct = float(self.tsl_config.get("percentage", 15.0))
            if current_trail_pct > 5.0:
                self.tsl_config = dict(self.tsl_config)
                self.tsl_config["percentage"] = 5.0
                logging.info(
                    "[TrailTighten] 13:30 reached — tightening trail to 5% "
                    "to lock intraday gains before theta accelerates."
                )

        # 7. Update trailing stop (with profit-level tightening) and modify SL-M.
        new_trail = self._update_premium_trailing_stop(current_price)
        if not is_paper_trade and self.active_trade.get("sl_order_id") and new_trail:
            await self._maybe_modify_broker_sl(new_trail)

        # 8. Software backstop: if no broker SL or it's stale, enforce in code.
        trail = self.active_trade.get("trailing_stop_loss")
        hard = self.active_trade["initial_stop_loss"]
        if current_price <= hard or (trail and current_price <= trail):
            logging.info(f"Software SL hit for {symbol} @ {current_price:.2f}.")
            return await self.exit_trade(
                is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
            )

        # 9. Indicator-based exit (PSAR / MA on the underlying).
        if self.tsl_config.get("use_indicator_exit") and underlying_hist_df is not None:
            if self._check_indicator_exit(underlying_hist_df):
                # Net-profit guard: don't book a *profit-taking* indicator exit
                # whose gain wouldn't even clear transaction costs + margin —
                # that just converts a sub-cost scalp into a guaranteed net loss.
                # Loss-protective exits (price at/below entry) are NEVER blocked;
                # the trailing stop (step 7/8) still caps downside if we hold.
                if self._exit_clears_costs(current_price):
                    logging.info(f"Indicator exit triggered for {symbol}.")
                    return await self.exit_trade(
                        is_paper_trade, underlying_hist_df, sentiment_agent, gemini_api_key
                    )
                else:
                    logging.info(
                        f"[ExitGuard] Indicator exit held for {symbol} @ "
                        f"{current_price:.2f} — profit-protector mode: it won't cut a "
                        f"loss on a wobble. The hard SL / trailing stop / give-up "
                        f"rule / 14:00 exit handle losers."
                    )

        return "ACTIVE"

    def _exit_clears_costs(self, current_price: float) -> bool:
        """
        Gate for the INDICATOR exit. Returns True to allow the exit, False to hold.

        Two protections:
          • Profit-protector mode (trailing_stop_loss.indicator_exit_profit_only,
            default True): the indicator exit must NOT cut a LOSS on an underlying
            wobble — that is the death-by-cuts trap (premium moves 2-3 pts, costs
            ~₹100, net loss). When the position is NOT in profit, suppress the
            indicator exit and let the hard SL / trailing stop / give-up rule /
            14:00 time exit handle the loser. Set false for the legacy behaviour
            (indicator cuts losers too).
          • Sub-cost guard (transaction_costs.guard_profit_exits): when in profit,
            only book the exit if the NET gain (after costs) ≥ min_net_profit_inr.
        """
        trade = self.active_trade or {}
        entry = float(trade.get("entry_price", 0) or 0)
        qty   = int(trade.get("quantity", 0) or 0)
        if entry <= 0 or qty <= 0 or current_price is None:
            return True
        gross = (float(current_price) - entry) * qty

        # ── Loss / breakeven: suppress the indicator exit (profit-protector mode) ──
        if gross <= 0:
            profit_only = bool((self.tsl_config or {}).get("indicator_exit_profit_only", True))
            return not profit_only   # profit_only → hold (False); legacy → cut (True)

        # ── In profit: only book it if the NET clears the floor ──
        tc_cfg = (self.config.get("transaction_costs") or {})
        if not tc_cfg.get("enable", True) or not tc_cfg.get("guard_profit_exits", True):
            return True
        est = estimate_options_cost(entry * qty, float(current_price) * qty, 2, self.config)
        net = gross - est["total"]
        min_net = float(tc_cfg.get("min_net_profit_inr", 0) or 0)
        return net >= min_net

    # ---------- trailing / indicator exits ----------

    def _dynamic_trail_pct(self, current_price: float) -> float:
        """
        Tightens the trailing-stop % as profit grows — protects larger gains
        more aggressively without killing a trade too early.

        Profit bands (% gain on entry premium):
          < T1 threshold  → base trail %   (loose, give the trade room)
          T1 → T2         → 8%             (moderate — first partial already booked)
          > T2            → 5%             (tight — runner is free money)

        In AGGRESSIVE mode the base trail is wider (default 20% vs 15%) so the
        trade gets more room before being stopped out — matching the higher-risk
        profile of that mode.
        """
        pe_cfg = self.config.get('partial_exits') or {}
        flags  = self.config.get('trading_flags', {})

        # Use T1/T2 gain targets from the active mode (aggressive overrides static config).
        t1_gain = float(flags.get('_agg_t1_gain_pct', pe_cfg.get('t1_gain_pct', 30))) / 100.0
        t2_gain = float(flags.get('_agg_t2_gain_pct', pe_cfg.get('t2_gain_pct', 60))) / 100.0

        # Base trail priority: scalp mode (tightest) → aggressive mode (widest) → static config.
        if flags.get('_scalp_mode'):
            base_pct = float(flags.get('_scalp_trail_pct', 10.0))
        else:
            base_pct = float(flags.get('_agg_trail_pct') or self.tsl_config.get('percentage', 15.0))

        entry = float(self.active_trade.get('entry_price', 0) or 0)
        if entry <= 0:
            return base_pct
        gain_pct = (current_price - entry) / entry

        if gain_pct >= t2_gain:
            return 5.0
        elif gain_pct >= t1_gain:
            return 8.0
        return base_pct

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
        # Use dynamic (profit-level-based) trail % instead of fixed %.
        pct = self._dynamic_trail_pct(current_price)
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

        # ── Minimum hold: give the trade room to breathe through entry noise. ──
        # Without this, a single bar closing on the wrong side of a fast EMA/PSAR
        # (common in low-vol chop) exits the trade within minutes for a few-point
        # loss. The hard stop-loss and software trailing stop still protect
        # against genuine adverse moves during this window.
        min_hold = float(self.tsl_config.get("indicator_exit_min_hold_minutes", 10) or 0)
        if min_hold > 0:
            et = (self.active_trade or {}).get("entry_time")
            if et:
                try:
                    entry_dt = datetime.datetime.fromisoformat(et)
                    held_min = (datetime.datetime.now() - entry_dt).total_seconds() / 60.0
                    if held_min < min_hold:
                        return False
                except Exception:
                    pass

        last = df.iloc[-1]
        price = last["close"]
        side = self.active_trade["type"]

        # ── Volatility-aware buffer: the underlying must be BEYOND the indicator
        # by at least atr_mult × ATR before we act, so noise wiggles around the
        # line don't chop us out. The buffer scales with volatility — bigger ATR
        # demands a bigger breach. Falls back to 0.1% of price if ATR is missing. ──
        buf = 0.0
        atr_mult = float(self.tsl_config.get("indicator_exit_atr_buffer_mult", 0.25) or 0)
        if atr_mult > 0:
            atr_val = last.get("atr")
            if atr_val is not None and not pd.isna(atr_val) and float(atr_val) > 0:
                buf = atr_mult * float(atr_val)
            else:
                buf = 0.001 * float(price)

        if kind == "MA":
            period = int(self.tsl_config.get("ma_period", 9))
            col = f"ema_{period}"
            if col not in df.columns:
                df[col] = ta.ema(df["close"], length=period)
            ma = df.iloc[-1].get(col)
            if pd.isna(ma):
                return False
            if side == "BUY" and price < (ma - buf):
                return True
            if side == "SELL" and price > (ma + buf):
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
                if short_val is not None and not pd.isna(short_val) and price < (short_val - buf):
                    return True
            else:
                long_val = df.iloc[-1].get("psar_long")
                if long_val is not None and not pd.isna(long_val) and price > (long_val + buf):
                    return True
            return False

        return False

    async def _exit_partial_quantity(self, qty_to_exit: int, reason: str,
                                      is_paper_trade: bool, current_price: float) -> float:
        """Exit `qty_to_exit` of an active position. Returns actual exit price.

        On first partial: cancels the full-qty broker SL-M (prevents a double-fill
        against the already-sold lots) and switches to software SL management only.
        """
        trade = self.active_trade
        symbol = trade['symbol']
        exit_price = current_price

        if not is_paper_trade:
            # Cancel broker SL-M before the first partial so it doesn't fire on
            # lots we've already sold.
            sl_id = trade.get('sl_order_id')
            if sl_id:
                await asyncio.to_thread(
                    _cancel_order_sync, self.api_key, self.access_token,
                    self.flags['order_variety'], sl_id,
                )
                trade['sl_order_id'] = None
                logging.info(f"Broker SL-M {sl_id} cancelled before partial exit.")

            tick = float(trade.get('tick_size', 0.05))
            slip = float(self.flags.get('limit_order_slippage_percent', 0.5)) / 100.0
            limit_px = tick_round(current_price * (1 - slip), tick)
            params = {
                'variety': self.flags['order_variety'],
                'exchange': self.kite.EXCHANGE_NFO,
                'tradingsymbol': symbol,
                'transaction_type': self.kite.TRANSACTION_TYPE_SELL,
                'quantity': qty_to_exit,
                'product': self.flags['product_type'],
                'order_type': self.kite.ORDER_TYPE_LIMIT,
                'price': limit_px,
            }
            oid = await asyncio.to_thread(
                _execute_order_sync, self.api_key, self.access_token, params
            )
            if oid:
                timeout = int(self.flags.get('order_fill_timeout_seconds', 30))
                status, avg, _ = await _wait_for_fill(
                    self.api_key, self.access_token, oid, timeout
                )
                if status == 'COMPLETE' and avg > 0:
                    exit_price = avg

        partial_pnl = (exit_price - trade['entry_price']) * qty_to_exit
        trade['_pe_realized_pnl'] = float(trade.get('_pe_realized_pnl', 0.0)) + partial_pnl
        trade['quantity'] = int(trade['quantity']) - qty_to_exit
        logging.info(
            f"PARTIAL EXIT [{reason}]: {qty_to_exit} lots @ {exit_price:.2f} "
            f"partial_pnl={partial_pnl:+.2f} | remaining={trade['quantity']} lots"
        )
        self._save_state()
        return exit_price

    async def _check_partial_exits(self, current_price: float,
                                    is_paper_trade: bool) -> Optional[str]:
        """Fire T1 / T2 partial exits when premium targets are hit.

        T1 (+t1_gain_pct%): exit t1_exit_pct% of original position; move SL to breakeven.
        T2 (+t2_gain_pct%): exit t2_exit_pct% of original; trail remainder aggressively.

        Returns 'FULLY_EXITED' if no lots remain after partial exits, else None.
        Partial exits are SKIPPED if not enough lots to round to a lot boundary.
        """
        trade = self.active_trade
        if not trade.get('_pe_enabled'):
            return None

        pe_cfg = self.config.get('partial_exits') or {}
        flags  = self.config.get('trading_flags', {})
        entry       = float(trade['entry_price'])
        orig_qty    = int(trade['_pe_original_qty'])
        lot_size    = int(trade.get('lot_size', 1) or 1)
        remaining   = int(trade.get('quantity', 0))

        # Target priority: scalp mode (tightest) → aggressive mode (widest) → static config.
        if flags.get('_scalp_mode'):
            t1_pct  = float(flags.get('_scalp_t1_gain_pct', 15)) / 100.0
            t2_pct  = float(flags.get('_scalp_t2_gain_pct', 25)) / 100.0
        else:
            # In AGGRESSIVE mode, let winners run further before booking partials.
            t1_pct  = float(flags.get('_agg_t1_gain_pct', pe_cfg.get('t1_gain_pct', 30))) / 100.0
            t2_pct  = float(flags.get('_agg_t2_gain_pct', pe_cfg.get('t2_gain_pct', 60))) / 100.0
        t1_frac     = float(pe_cfg.get('t1_exit_pct', 40)) / 100.0
        t2_frac     = float(pe_cfg.get('t2_exit_pct', 40)) / 100.0

        # T1 — first partial profit booking
        if not trade.get('_pe_t1_hit') and current_price >= entry * (1 + t1_pct):
            raw = orig_qty * t1_frac
            qty_exit = max(lot_size, int(raw // lot_size) * lot_size)
            qty_exit = min(qty_exit, remaining)
            if qty_exit >= lot_size:
                await self._exit_partial_quantity(qty_exit, 'T1_TARGET', is_paper_trade, current_price)
                trade['_pe_t1_hit'] = True
                # Slide SL to breakeven — protect the trade after first win.
                be = entry
                trade['trailing_stop_loss'] = max(float(trade.get('trailing_stop_loss', 0)), be)
                trade['initial_stop_loss']  = max(float(trade['initial_stop_loss']), be)
                logging.info(
                    f"T1 target hit @ {current_price:.2f} (+{t1_pct*100:.0f}%). "
                    f"SL moved to breakeven {be:.2f}."
                )
                self._save_state()

        # T2 — second partial profit booking (only after T1 confirmed)
        remaining = int(trade.get('quantity', 0))
        if trade.get('_pe_t1_hit') and not trade.get('_pe_t2_hit') and current_price >= entry * (1 + t2_pct):
            raw = orig_qty * t2_frac
            qty_exit = max(lot_size, int(raw // lot_size) * lot_size)
            qty_exit = min(qty_exit, remaining)
            if qty_exit >= lot_size:
                await self._exit_partial_quantity(qty_exit, 'T2_TARGET', is_paper_trade, current_price)
                trade['_pe_t2_hit'] = True
                logging.info(
                    f"T2 target hit @ {current_price:.2f} (+{t2_pct*100:.0f}%). "
                    f"Trailing remainder aggressively."
                )
                self._save_state()

        if int(trade.get('quantity', 0)) <= 0:
            return 'FULLY_EXITED'
        return None

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
            # gemini-1.5-flash is retired (404). Use the current model, overridable
            # via config.google_api.model (defaults match the rest of the bot).
            model = ((self.config.get("google_api") or {}).get("model")
                     or "gemini-2.0-flash")
            api_url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent?key={gemini_api_key}"
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

    async def _close_one_leg(
        self,
        symbol: str,
        qty: int,
        transaction_type,   # kite.TRANSACTION_TYPE_SELL / BUY
        current_ltp: float,
        timeout: int,
        side_label: str = "",
    ) -> tuple:
        """
        Place a LIMIT order to close one option leg; fall back to MARKET on
        non-fill.  Returns (fill_price, order_id).
        """
        # _tick_size_for lives in OrderExecutionAgent (needs nfo_instruments).
        # PositionManagementAgent stores tick_size in active_trade at entry;
        # use that for the main leg, and 0.05 (standard NFO minimum) as a
        # safe fallback for spread short legs.
        _stored_trade = self.active_trade or {}
        tick = float(_stored_trade.get("tick_size") or 0.05)
        slip  = float(self.flags.get("limit_order_slippage_percent", 0.5)) / 100.0

        is_sell = transaction_type == self.kite.TRANSACTION_TYPE_SELL
        limit_price = tick_round(
            current_ltp * (1 - slip) if is_sell else current_ltp * (1 + slip), tick
        )
        params = {
            "variety":          self.flags["order_variety"],
            "exchange":         self.kite.EXCHANGE_NFO,
            "tradingsymbol":    symbol,
            "transaction_type": transaction_type,
            "quantity":         qty,
            "product":          self.flags["product_type"],
            "order_type":       self.kite.ORDER_TYPE_LIMIT,
            "price":            limit_price,
        }
        logging.info(f"ASYNC: placing LIMIT exit{side_label} {params}")
        order_id = await asyncio.to_thread(
            _execute_order_sync, self.api_key, self.access_token, params
        )
        if not order_id:
            return current_ltp, None

        status, avg, _ = await _wait_for_fill(self.api_key, self.access_token, order_id, timeout)
        if status == "COMPLETE" and avg > 0:
            return float(avg), order_id

        logging.warning(
            f"Exit LIMIT{side_label} did not fill (status={status}); falling back to MARKET."
        )
        await asyncio.to_thread(
            _cancel_order_sync, self.api_key, self.access_token,
            self.flags["order_variety"], order_id,
        )
        mkt_params = dict(params)
        mkt_params.pop("price", None)
        mkt_params["order_type"] = self.kite.ORDER_TYPE_MARKET
        mkt_id = await asyncio.to_thread(
            _execute_order_sync, self.api_key, self.access_token, mkt_params
        )
        if mkt_id:
            s2, avg2, _ = await _wait_for_fill(self.api_key, self.access_token, mkt_id, timeout)
            if s2 == "COMPLETE" and avg2 > 0:
                return float(avg2), mkt_id
        return current_ltp, order_id  # best-effort fallback

    async def exit_trade(self, is_paper_trade=False, underlying_df=None,
                         sentiment_agent=None, gemini_api_key=None):
        if not self.active_trade:
            return None
        trade      = self.active_trade
        symbol     = trade["symbol"]
        is_spread  = trade.get("is_spread", False)
        qty        = trade["quantity"]
        timeout    = int(self.flags.get("order_fill_timeout_seconds", 30))
        exit_reason = "PAPER" if is_paper_trade else "INDICATOR_OR_SOFTWARE_SL"

        long_ltp = safe_ltp(self.kite, f"NFO:{symbol}") or trade.get("entry_price", 0)
        exit_price = long_ltp
        exit_order_id = None

        if not is_paper_trade:
            # Cancel any existing SL-M on the long leg first.
            sl_id = trade.get("sl_order_id")
            if sl_id:
                await asyncio.to_thread(
                    _cancel_order_sync, self.api_key, self.access_token,
                    self.flags["order_variety"], sl_id,
                )

            # ── Close long leg (SELL) ─────────────────────────────────────────
            long_exit, exit_order_id = await self._close_one_leg(
                symbol, qty, self.kite.TRANSACTION_TYPE_SELL, long_ltp, timeout,
                side_label=" (long leg)",
            )
            exit_price = long_exit

            # ── Close short leg if this is a spread (BUY back the short) ─────
            if is_spread:
                short_sym = trade.get("spread_short_symbol")
                if short_sym:
                    short_ltp  = safe_ltp(self.kite, f"NFO:{short_sym}") or 0.0
                    short_exit, _ = await self._close_one_leg(
                        short_sym, qty, self.kite.TRANSACTION_TYPE_BUY,
                        float(short_ltp), timeout, side_label=" (short leg)",
                    )
                    # exit_price = net credit received = long_exit − short_exit
                    exit_price = long_exit - short_exit
                    logging.info(
                        f"[Spread] Exit: long={long_exit:.2f} short_buyback={short_exit:.2f} "
                        f"net_credit={exit_price:.2f}"
                    )

        elif is_paper_trade and is_spread:
            # Paper spread: net credit = long_ltp − short_ltp
            short_sym   = trade.get("spread_short_symbol")
            short_price = safe_ltp(self.kite, f"NFO:{short_sym}") if short_sym else None
            if short_price:
                exit_price = max(0.0, long_ltp - float(short_price))

        return await self._book_completed_trade(
            exit_price, underlying_df, sentiment_agent, gemini_api_key,
            exit_order_id=exit_order_id, exit_reason=exit_reason,
        )

    async def _book_completed_trade(self, exit_price, underlying_df, sentiment_agent,
                                    gemini_api_key, exit_order_id=None, exit_reason="UNKNOWN"):
        trade = self.active_trade
        # Remaining-lots P&L + any partial-exit P&L already banked at T1/T2.
        remaining_pnl = (exit_price - trade["entry_price"]) * trade["quantity"] if exit_price > 0 else 0.0
        gross_pnl = remaining_pnl + float(trade.get('_pe_realized_pnl', 0.0))

        # ── Transaction costs → NET P&L ────────────────────────────────────
        # Brokerage + STT + exchange/SEBI charges + stamp duty + GST. A small
        # gross win is often a NET loss once these are deducted, so everything
        # downstream (realized-P&L tracking, loss limits, reports) must use NET.
        qty       = int(trade.get("quantity", 0) or 0)
        entry_px  = float(trade.get("entry_price", 0) or 0)
        buy_value  = abs(entry_px) * qty
        sell_value = abs(float(exit_price)) * qty if exit_price > 0 else 0.0
        # Order count: 1 entry + 1 final exit, +1 for each partial that fired,
        # +2 for a debit spread's extra short leg (entry + exit).
        num_orders = 2
        if trade.get("_pe_t1_hit"):  num_orders += 1
        if trade.get("_pe_t2_hit"):  num_orders += 1
        if trade.get("is_spread"):   num_orders += 2
        tc_cfg = (self.config.get("transaction_costs") or {})
        if tc_cfg.get("enable", True):
            cost_breakdown = estimate_options_cost(buy_value, sell_value, num_orders, self.config)
            costs = cost_breakdown["total"]
        else:
            cost_breakdown, costs = {}, 0.0
        net_pnl = gross_pnl - costs

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
            "ProfitLoss": net_pnl,            # NET of all costs — used everywhere downstream
            "GrossProfitLoss": gross_pnl,     # before costs (for transparency / reports)
            "Costs": costs,
            "CostBreakdown": cost_breakdown,
            "Status": "CLOSED",
            "Strategy": trade.get("Strategy", "N/A"),
            # Extra context for the loss post-mortem (loss_analyzer.build_loss_report).
            # Carried on the dict; reporting.log_trade ignores unknown keys.
            "entry_time": trade.get("entry_time"),
            "high_water_mark": trade.get("high_water_mark"),
            "initial_stop_loss": trade.get("initial_stop_loss"),
            "lot_size": trade.get("lot_size"),
        }
        if costs:
            logging.info(
                f"[Costs] {trade['symbol']}: gross ₹{gross_pnl:,.2f} − costs "
                f"₹{costs:,.2f} (brokerage ₹{cost_breakdown.get('brokerage',0):.0f}, "
                f"STT ₹{cost_breakdown.get('stt',0):.0f}, "
                f"txn ₹{cost_breakdown.get('exchange_txn',0):.0f}, "
                f"GST ₹{cost_breakdown.get('gst',0):.0f}) = NET ₹{net_pnl:,.2f}"
            )

        if net_pnl < 0 and self.flags.get("enable_gemini_loss_analysis") and gemini_api_key:
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
