"""
Option-chain analytics — Black-Scholes pricing, Greeks, IV solver,
realized-vol computation, chain-snapshot fetcher, and strike-selection helpers.

Zero new dependencies: math.erf (stdlib) is used instead of scipy.norm,
keeping the install footprint identical to before.

Used by agents.OrderExecutionAgent to:
  - Refuse trades on illiquid strikes (spread / OI / quote-age).
  - Refuse to buy options when IV-Rank is too high.
  - Refuse to buy when implied >> realized vol.
  - Pick a strike by delta band (default 0.40-0.55) instead of fixed offset.
"""
from __future__ import annotations

import datetime
import logging
import math
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Black-Scholes price + Greeks (math.erf based, no scipy needed)
# ---------------------------------------------------------------------------

_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def black_scholes_price(spot: float, strike: float, T: float, rate: float,
                         sigma: float, option_type: str) -> float:
    """European Black-Scholes price. Falls back to intrinsic on degenerate inputs."""
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        if option_type == "CE":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    if option_type == "CE":
        return spot * _norm_cdf(d1) - strike * math.exp(-rate * T) * _norm_cdf(d2)
    return strike * math.exp(-rate * T) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def greeks(spot: float, strike: float, T: float, rate: float,
            sigma: float, option_type: str) -> dict:
    """
    Delta / gamma / theta / vega for a European option.
    Conventions: theta is per *calendar day*, vega is per *1% IV* change.
    """
    zero = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    if T <= 0 or sigma <= 0 or spot <= 0 or strike <= 0:
        return zero
    sqrt_T = math.sqrt(T)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    pdf_d1 = _norm_pdf(d1)
    if option_type == "CE":
        delta = _norm_cdf(d1)
        theta_annual = (-spot * pdf_d1 * sigma / (2.0 * sqrt_T)
                        - rate * strike * math.exp(-rate * T) * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (-spot * pdf_d1 * sigma / (2.0 * sqrt_T)
                        + rate * strike * math.exp(-rate * T) * _norm_cdf(-d2))
    gamma = pdf_d1 / (spot * sigma * sqrt_T)
    vega_per_1sigma = spot * pdf_d1 * sqrt_T
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_annual / 365.0,
        "vega": vega_per_1sigma / 100.0,
    }


def implied_vol(market_price: float, spot: float, strike: float, T: float,
                 rate: float, option_type: str,
                 max_iter: int = 50, tol: float = 1e-4) -> Optional[float]:
    """
    Newton-Raphson IV solver. Returns annualised sigma (e.g. 0.18 = 18% IV) or None
    if it doesn't converge / the input is unworkable (price below intrinsic, etc.).
    """
    if market_price <= 0 or T <= 0 or spot <= 0 or strike <= 0:
        return None
    intrinsic = max(0.0, spot - strike) if option_type == "CE" else max(0.0, strike - spot)
    if market_price < intrinsic - 0.05:
        return None  # quote below intrinsic — stale or arbitrage; refuse to compute IV

    sigma = 0.30  # 30% — reasonable midpoint for Nifty
    for _ in range(max_iter):
        price = black_scholes_price(spot, strike, T, rate, sigma, option_type)
        diff = price - market_price
        if abs(diff) < tol:
            return max(0.001, min(5.0, sigma))
        # vega per 1.0 sigma change
        vega = spot * _norm_pdf(
            (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        ) * math.sqrt(T)
        if vega < 1e-8:
            break
        sigma -= diff / vega
        sigma = max(0.001, min(5.0, sigma))
    return None


# ---------------------------------------------------------------------------
# Realized volatility (close-to-close, annualised)
# ---------------------------------------------------------------------------

def realized_vol(closes: pd.Series, lookback_days: int = 20,
                  periods_per_year: int = 252) -> Optional[float]:
    """Annualised stdev of log returns over the last `lookback_days` daily closes."""
    if closes is None or len(closes) < lookback_days + 1:
        return None
    log_returns = np.log(closes / closes.shift(1)).dropna().tail(lookback_days)
    if len(log_returns) < 2:
        return None
    return float(log_returns.std() * math.sqrt(periods_per_year))


# ---------------------------------------------------------------------------
# Chain snapshot via kite.quote()
# ---------------------------------------------------------------------------

def fetch_chain_quote(kite, symbols: list) -> dict:
    """Wrap kite.quote() defensively. Returns {} on any failure."""
    if not symbols:
        return {}
    keys = [s if str(s).startswith("NFO:") else f"NFO:{s}" for s in symbols]
    try:
        return kite.quote(keys) or {}
    except Exception as e:
        logging.warning(f"fetch_chain_quote failed for {len(keys)} symbols: {e}")
        return {}


def quote_age_seconds(payload: dict) -> Optional[float]:
    """Seconds since `last_trade_time`, naive about timezone — returns None on parse failure."""
    if not payload:
        return None
    ltt = payload.get("last_trade_time")
    if not ltt:
        return None
    try:
        if isinstance(ltt, str):
            ltt_dt = datetime.datetime.fromisoformat(ltt.replace("Z", "+00:00"))
        elif isinstance(ltt, datetime.datetime):
            ltt_dt = ltt
        else:
            return None
        if ltt_dt.tzinfo is not None:
            ltt_dt = ltt_dt.replace(tzinfo=None)
        return (datetime.datetime.now() - ltt_dt).total_seconds()
    except Exception:
        return None


def build_chain_snapshot(quote_payload: dict, instruments_df: pd.DataFrame,
                          spot: float, T_years: float, rate: float) -> pd.DataFrame:
    """
    Fold a kite.quote() payload + instruments metadata into a single DataFrame.
    Adds bid/ask/spread/oi/age + computed IV + Greeks for each row.
    """
    rows = []
    for sym, payload in (quote_payload or {}).items():
        clean_sym = sym.replace("NFO:", "")
        match = instruments_df[instruments_df["tradingsymbol"] == clean_sym]
        if match.empty:
            continue
        meta = match.iloc[0]
        strike = float(meta["strike"])
        opt_type = meta["instrument_type"]

        last = float(payload.get("last_price") or 0)
        depth = payload.get("depth") or {}
        buy = depth.get("buy") or []
        sell = depth.get("sell") or []
        bid = float((buy[0] or {}).get("price") or 0) if buy else 0.0
        ask = float((sell[0] or {}).get("price") or 0) if sell else 0.0
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
        spread = (ask - bid) if (ask > 0 and bid > 0) else 0.0
        spread_pct = (spread / mid * 100.0) if mid > 0 else float("inf")
        oi = int(payload.get("oi") or 0)
        age = quote_age_seconds(payload)

        ref_price = mid if mid > 0 else last
        iv = (implied_vol(ref_price, spot, strike, T_years, rate, opt_type)
              if ref_price > 0 else None)
        if iv is not None:
            g = greeks(spot, strike, T_years, rate, iv, opt_type)
        else:
            g = {"delta": None, "gamma": None, "theta": None, "vega": None}

        rows.append({
            "tradingsymbol": clean_sym,
            "strike": strike,
            "instrument_type": opt_type,
            "last": last,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_pct": spread_pct,
            "oi": oi,
            "age_seconds": age,
            "iv": iv,
            "delta": g["delta"],
            "gamma": g["gamma"],
            "theta": g["theta"],
            "vega": g["vega"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strike selection + liquidity filter
# ---------------------------------------------------------------------------

def select_by_delta(chain_df: pd.DataFrame, option_type: str,
                     delta_low: float, delta_high: float) -> Optional[pd.Series]:
    """
    Pick the row whose |delta| is closest to the midpoint of [delta_low, delta_high].
    Prefers rows already inside the band; otherwise the closest one outside.
    Returns the chosen row (Series) or None when the chain has no Greeks.
    """
    if chain_df is None or chain_df.empty:
        return None
    df = chain_df[chain_df["instrument_type"] == option_type].copy()
    df = df[df["delta"].notna()]
    if df.empty:
        return None
    df["abs_delta"] = df["delta"].abs()
    midpoint = (delta_low + delta_high) / 2.0
    df["dist"] = (df["abs_delta"] - midpoint).abs()
    in_band = df[(df["abs_delta"] >= delta_low) & (df["abs_delta"] <= delta_high)]
    pool = in_band if not in_band.empty else df
    return pool.sort_values("dist").iloc[0]


def passes_liquidity(row: pd.Series, max_spread_pct: float, min_oi: int,
                      max_age_seconds: float) -> tuple:
    """Returns (ok: bool, reason: Optional[str]) — reason set only when ok=False."""
    spread = float(row.get("spread_pct") or float("inf"))
    if spread > max_spread_pct:
        return False, f"spread {spread:.2f}% > {max_spread_pct:.2f}%"
    oi = int(row.get("oi") or 0)
    if oi < min_oi:
        return False, f"OI {oi} < {min_oi}"
    age = row.get("age_seconds")
    if age is not None and age > max_age_seconds:
        return False, f"quote age {age:.1f}s > {max_age_seconds}s"
    return True, None


def find_atm_row(chain_df: pd.DataFrame, option_type: str,
                  atm_strike: float) -> Optional[pd.Series]:
    """Returns the chain row at exactly `atm_strike` for `option_type`, or None."""
    if chain_df is None or chain_df.empty:
        return None
    match = chain_df[
        (chain_df["instrument_type"] == option_type)
        & (chain_df["strike"] == atm_strike)
    ]
    if match.empty:
        return None
    return match.iloc[0]
