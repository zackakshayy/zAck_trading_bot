"""
Shared infrastructure: state I/O, retries, instrument caching, defensive accessors.

Pulled out of agents.py / trading_bot.py so both can share without circular imports.
Designed for low-overhead use inside the trading hot loop.
"""
from __future__ import annotations

import json
import logging
import os
import random
import tempfile
import threading
import time
from typing import Any, Callable, Iterable

import pandas as pd

# ---------------------------------------------------------------------------
# State directory + atomic JSON I/O
# ---------------------------------------------------------------------------

STATE_DIR = "state"
os.makedirs(STATE_DIR, exist_ok=True)


def state_path(filename: str) -> str:
    return os.path.join(STATE_DIR, filename)


def atomic_write_json(path: str, payload: Any) -> None:
    """
    Write JSON atomically: tempfile in same dir + os.replace. A crash mid-write
    cannot leave a half-written or 0-byte file at `path`.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=directory, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, default=str, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logging.warning(f"read_json: corrupt or unreadable {path} ({e}); returning default.")
        return default


# ---------------------------------------------------------------------------
# Retry-with-backoff for kite API calls
# ---------------------------------------------------------------------------

def retry_call(func: Callable, *args,
               attempts: int = 3,
               base_delay: float = 0.5,
               retryable_exceptions: Iterable[type] = (),
               **kwargs):
    """
    Synchronous retry wrapper. Only retries on `retryable_exceptions`; everything
    else propagates immediately so we don't mask logic errors. Backoff is
    base_delay * 2**i with up to 25% jitter.
    """
    last_exc = None
    retryable_exceptions = tuple(retryable_exceptions)
    for i in range(attempts):
        try:
            return func(*args, **kwargs)
        except retryable_exceptions as e:
            last_exc = e
            if i == attempts - 1:
                break
            delay = base_delay * (2 ** i)
            delay *= 1 + random.uniform(0, 0.25)
            logging.warning(
                f"retry_call: {func.__name__} attempt {i+1}/{attempts} failed ({e}); "
                f"retrying in {delay:.2f}s"
            )
            time.sleep(delay)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Instrument cache (per-exchange, session-scoped, thread-safe)
# ---------------------------------------------------------------------------

_INSTRUMENT_LOCK = threading.Lock()
_INSTRUMENT_CACHE: dict[str, pd.DataFrame] = {}
_INSTRUMENT_FETCHED_AT: dict[str, float] = {}
_INSTRUMENT_TTL_SECONDS = 6 * 60 * 60  # 6 hours – safe for an intraday session


def get_instruments(kite, exchange: str, force_refresh: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame of instruments for an exchange, fetched at most once
    per TTL window. Replaces ad-hoc `kite.instruments(exchange)` calls scattered
    across agents/market_context, each of which downloads ~30K rows.
    """
    now = time.time()
    with _INSTRUMENT_LOCK:
        cached = _INSTRUMENT_CACHE.get(exchange)
        fetched_at = _INSTRUMENT_FETCHED_AT.get(exchange, 0)
        if cached is not None and not force_refresh and (now - fetched_at) < _INSTRUMENT_TTL_SECONDS:
            return cached
    df = pd.DataFrame(kite.instruments(exchange))
    with _INSTRUMENT_LOCK:
        _INSTRUMENT_CACHE[exchange] = df
        _INSTRUMENT_FETCHED_AT[exchange] = now
    logging.info(f"get_instruments: cached {len(df)} rows for {exchange}.")
    return df


def get_instrument_token(kite, tradingsymbol: str, exchange: str) -> int:
    """Looks up a single token from the cached instrument list."""
    df = get_instruments(kite, exchange)
    match = df[df["tradingsymbol"] == tradingsymbol]
    if match.empty:
        raise KeyError(f"Instrument {tradingsymbol!r} not found on {exchange}.")
    return int(match.iloc[0]["instrument_token"])


# ---------------------------------------------------------------------------
# Defensive LTP / dict access
# ---------------------------------------------------------------------------

def safe_ltp(kite, key: str) -> float | None:
    """Returns last_price for `key` (e.g. 'NFO:NIFTY24DEC24500CE') or None on any failure."""
    try:
        data = kite.ltp(key) or {}
        entry = data.get(key) or {}
        price = entry.get("last_price")
        return float(price) if price else None
    except Exception as e:
        logging.debug(f"safe_ltp({key}) failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Tick-size-aware price rounding
# ---------------------------------------------------------------------------

def tick_round(price: float, tick_size: float = 0.05) -> float:
    """Round `price` to the nearest valid tick. NFO options are 0.05; equity 0.05; commodity varies."""
    if tick_size <= 0:
        return round(price, 2)
    return round(round(price / tick_size) * tick_size, 2)


# ---------------------------------------------------------------------------
# Transaction-cost model (Indian NFO options, Zerodha reference rates)
# ---------------------------------------------------------------------------
# A ₹100 gross "profit" on an options scalp can be a net LOSS once brokerage,
# STT, exchange + SEBI charges, stamp duty and GST are deducted. These helpers
# let the bot reason in NET terms — both when deciding to enter and when
# booking the result. All rates are fractions (0.001 = 0.10%) and reflect
# Zerodha's equity-options charges as of FY2024-25; override any of them under
# config['transaction_costs'].
DEFAULT_COST_RATES: dict = {
    "brokerage_per_order": 20.0,      # flat ₹20 per executed order (options)
    "stt_sell_pct":        0.001,     # 0.10% STT — charged on the SELL-side premium only
    "exchange_txn_pct":    0.0003503, # NSE option txn charge on premium turnover (both sides)
    "sebi_pct":            0.000001,  # SEBI ₹10 per crore = 0.0001%
    "stamp_duty_buy_pct":  0.00003,   # 0.003% stamp duty — BUY side only
    "gst_pct":             0.18,      # 18% GST on (brokerage + exchange txn + SEBI)
}


def estimate_options_cost(buy_value: float, sell_value: float,
                          num_orders: int = 2, config: dict | None = None) -> dict:
    """
    Estimate the all-in round-trip transaction cost (INR) for an intraday NFO
    options trade. `buy_value` / `sell_value` are premium turnovers
    (price × quantity) for the buy and sell sides respectively.

    Returns a breakdown dict including 'total'. Pure function — cheap enough to
    call from the hot loop and from pre-trade sizing.
    """
    rates = dict(DEFAULT_COST_RATES)
    user = ((config or {}).get("transaction_costs") or {})
    for k in rates:
        if user.get(k) is not None:
            try:
                rates[k] = float(user[k])
            except (TypeError, ValueError):
                pass

    buy_value  = max(0.0, float(buy_value or 0))
    sell_value = max(0.0, float(sell_value or 0))
    turnover   = buy_value + sell_value

    brokerage = rates["brokerage_per_order"] * max(1, int(num_orders))
    stt       = sell_value * rates["stt_sell_pct"]
    exch_txn  = turnover   * rates["exchange_txn_pct"]
    sebi      = turnover   * rates["sebi_pct"]
    stamp     = buy_value  * rates["stamp_duty_buy_pct"]
    gst       = (brokerage + exch_txn + sebi) * rates["gst_pct"]
    total     = brokerage + stt + exch_txn + sebi + stamp + gst

    return {
        "brokerage":    round(brokerage, 2),
        "stt":          round(stt, 2),
        "exchange_txn": round(exch_txn, 2),
        "sebi":         round(sebi, 2),
        "stamp_duty":   round(stamp, 2),
        "gst":          round(gst, 2),
        "total":        round(total, 2),
    }


# ---------------------------------------------------------------------------
# NSE trading-day calendar
# ---------------------------------------------------------------------------

# Static list of full-day NSE market holidays with their names.
# Sourced from https://www.nseindia.com/resources/exchange-communication-holidays
# Update NSE_HOLIDAY_NAMES every December for the coming year.
# NSE_HOLIDAYS is derived automatically — do NOT edit it separately.
NSE_HOLIDAY_NAMES: dict = {
    # ── 2025 ──────────────────────────────────────────────────────────────
    "2025-02-26": "Mahashivratri",
    "2025-03-14": "Holi",
    "2025-03-31": "Id-Ul-Fitr (Eid)",
    "2025-04-10": "Shri Ram Navami",
    "2025-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2025-04-18": "Good Friday",
    "2025-05-01": "Maharashtra Day",
    "2025-08-15": "Independence Day",
    "2025-08-27": "Ganesh Chaturthi",
    "2025-10-02": "Gandhi Jayanti",
    "2025-10-21": "Diwali – Lakshmi Puja",
    "2025-10-22": "Diwali – Balipratipada",
    "2025-11-05": "Guru Nanak Jayanti",
    "2025-12-25": "Christmas",
    # ── 2026 ──────────────────────────────────────────────────────────────
    # Verify exact dates at NSE each December before the new year begins.
    "2026-01-26": "Republic Day",
    "2026-03-03": "Mahashivratri",
    "2026-03-19": "Holi",
    "2026-04-03": "Good Friday",
    "2026-04-10": "Shri Ram Navami",
    "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
    "2026-05-01": "Maharashtra Day",
    "2026-05-27": "Id-Ul-Adha (Bakri Eid)",
    "2026-05-28": "Buddha Purnima",
    "2026-06-27": "Id-Ul-Adha (Bakri Eid)",
    "2026-07-17": "Muharram",
    "2026-08-15": "Independence Day",
    "2026-10-02": "Gandhi Jayanti",
    "2026-10-20": "Dussehra (Vijaya Dashami)",
    "2026-11-09": "Diwali – Lakshmi Puja",
    "2026-11-24": "Guru Nanak Jayanti",
    "2026-12-25": "Christmas",
}

# Derived automatically — single source of truth is NSE_HOLIDAY_NAMES above.
NSE_HOLIDAYS: set = set(NSE_HOLIDAY_NAMES.keys())


def is_nse_holiday(d) -> bool:
    """Accepts datetime.date or datetime.datetime."""
    if hasattr(d, "date"):
        d = d.date()
    return d.strftime("%Y-%m-%d") in NSE_HOLIDAYS


# ---------------------------------------------------------------------------
# Daily P&L persistence
# ---------------------------------------------------------------------------

DAILY_PNL_FILE = state_path("daily_pnl.json")


def load_daily_pnl(date_str: str) -> float:
    """Reads realized P&L for `date_str` (YYYY-MM-DD); returns 0.0 if missing."""
    data = read_json(DAILY_PNL_FILE, default={})
    if not isinstance(data, dict):
        return 0.0
    return float(data.get(date_str, 0.0))


def save_daily_pnl(date_str: str, pnl: float) -> None:
    data = read_json(DAILY_PNL_FILE, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    data[date_str] = float(pnl)
    atomic_write_json(DAILY_PNL_FILE, data)


# ---------------------------------------------------------------------------
# Weekly P&L persistence
# ---------------------------------------------------------------------------

WEEKLY_PNL_FILE = state_path("weekly_pnl.json")


def load_weekly_pnl(week_str: str) -> float:
    """Reads realized P&L for `week_str` (e.g. '2025-W03'); returns 0.0 if missing."""
    data = read_json(WEEKLY_PNL_FILE, default={})
    if not isinstance(data, dict):
        return 0.0
    return float(data.get(week_str, 0.0))


def save_weekly_pnl(week_str: str, pnl: float) -> None:
    data = read_json(WEEKLY_PNL_FILE, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    data[week_str] = float(pnl)
    atomic_write_json(WEEKLY_PNL_FILE, data)


# ---------------------------------------------------------------------------
# ATM IV history (per underlying, dated) + IV-Rank computation
# ---------------------------------------------------------------------------

IV_HISTORY_FILE = state_path("iv_history.json")
IV_HISTORY_MAX_DAYS = 250  # cap file size; ~1 trading year


def load_iv_history(underlying: str) -> list:
    data = read_json(IV_HISTORY_FILE, default={}) or {}
    if not isinstance(data, dict):
        return []
    raw = data.get(underlying, [])
    return raw if isinstance(raw, list) else []


def append_iv_snapshot(underlying: str, date_str: str, iv: float,
                        spot: float, atm_strike: float) -> None:
    """At most one entry per (underlying, date_str) — last write wins."""
    data = read_json(IV_HISTORY_FILE, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    history = data.get(underlying, []) or []
    if not isinstance(history, list):
        history = []
    history = [h for h in history if isinstance(h, dict) and h.get("date") != date_str]
    history.append({
        "date": date_str,
        "iv": float(iv),
        "spot": float(spot),
        "atm_strike": float(atm_strike),
    })
    history.sort(key=lambda h: h.get("date", ""))
    if len(history) > IV_HISTORY_MAX_DAYS:
        history = history[-IV_HISTORY_MAX_DAYS:]
    data[underlying] = history
    atomic_write_json(IV_HISTORY_FILE, data)


def compute_ivr(underlying: str, current_iv: float,
                lookback_days: int = 60, min_samples: int = 10):
    """
    Returns (IVR_percent, samples_used) where IVR is current vs (min,max) of the
    last `lookback_days` samples. Returns (None, n) when sample count is below
    `min_samples` or when min == max (no spread).
    """
    history = load_iv_history(underlying)
    if not history:
        return None, 0
    sample = history[-lookback_days:]
    ivs = [float(h["iv"]) for h in sample if isinstance(h, dict) and "iv" in h]
    if len(ivs) < min_samples:
        return None, len(ivs)
    iv_min, iv_max = min(ivs), max(ivs)
    if iv_max <= iv_min:
        return None, len(ivs)
    ivr = (current_iv - iv_min) / (iv_max - iv_min) * 100.0
    return max(0.0, min(100.0, ivr)), len(ivs)


# ---------------------------------------------------------------------------
# Tiered per-trade risk by live capital (playbook alignment, Phase A)
# ---------------------------------------------------------------------------
# The professional playbook sizes risk to capital. This maps the LIVE account
# balance to a per-trade risk %. Defaults match the operator's spec:
#   capital < ₹1,00,000          → 10%
#   ₹1,00,000 ≤ capital ≤ ₹3,00,000 → 5%
#   capital > ₹3,00,000          → 2%
# Override under config['risk_tiers'] as a list of {max_capital, risk_pct};
# a null/absent max_capital is the catch-all top tier.
_DEFAULT_RISK_TIERS = [
    {"max_capital": 100000, "risk_pct": 10.0},
    {"max_capital": 300000, "risk_pct": 5.0},
    {"max_capital": None,   "risk_pct": 2.0},
]


def risk_pct_for_capital(capital, config=None) -> float:
    """Return the per-trade risk % for the given live capital, from config tiers."""
    tiers = ((config or {}).get("risk_tiers")) or _DEFAULT_RISK_TIERS
    cap = float(capital or 0)
    for t in tiers:
        try:
            mx = t.get("max_capital")
            # Strict "<" so "under ₹1,00,000" → 10% and exactly ₹1,00,000 → 5%.
            # At the upper boundary this errs toward the lower (safer) risk tier.
            if mx is None or cap < float(mx):
                return float(t.get("risk_pct", 2.0))
        except (TypeError, ValueError):
            continue
    try:
        return float(tiers[-1].get("risk_pct", 2.0))
    except Exception:
        return 2.0


# ---------------------------------------------------------------------------
# Weekly trading-day governor (max distinct trading days per ISO week)
# ---------------------------------------------------------------------------
# "Cash is a position." Tracks which calendar dates the bot actually TRADED in
# each ISO week so a frequency cap can force the bot to sit out once it has
# used its allotment of trading days that week.
WEEKLY_TRADE_DAYS_FILE = state_path("weekly_trade_days.json")


def load_week_trade_days(week_str: str) -> list:
    """Returns the list of date strings (YYYY-MM-DD) traded in `week_str`."""
    data = read_json(WEEKLY_TRADE_DAYS_FILE, default={}) or {}
    if not isinstance(data, dict):
        return []
    raw = data.get(week_str, [])
    return list(raw) if isinstance(raw, list) else []


def add_week_trade_day(week_str: str, date_str: str) -> None:
    """Record `date_str` as a traded day in `week_str` (idempotent)."""
    data = read_json(WEEKLY_TRADE_DAYS_FILE, default={}) or {}
    if not isinstance(data, dict):
        data = {}
    days = data.get(week_str, [])
    if not isinstance(days, list):
        days = []
    if date_str not in days:
        days.append(date_str)
        days.sort()
        data[week_str] = days
        # Keep the file small — retain only the most recent 12 weeks.
        if len(data) > 12:
            for old in sorted(data.keys())[:-12]:
                data.pop(old, None)
        atomic_write_json(WEEKLY_TRADE_DAYS_FILE, data)
