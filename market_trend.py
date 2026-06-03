"""
Pre-market price-action / global-cues trend analyzer.

Produces a single directional verdict for the day — **CE** (buy calls / bullish),
**PE** (buy puts / bearish), or **NEUTRAL** — from signals that are available
*before* the Indian market opens:

    • US markets        – overnight S&P/Nasdaq/Dow futures + prior cash close
    • Asian markets     – Nikkei / Hang Seng (live during Indian pre-open)
    • GIFT Nifty        – implied gap vs prior NIFTY close (best-effort)
    • Open gap          – NIFTY LTP / pre-open vs prior close (via Kite)
    • Price action      – last few NIFTY daily candles (structure / momentum)

Each component is best-effort: a missing or failed source simply drops out of the
weighted average rather than breaking the verdict. All weights, scales and the
verdict threshold are configurable under config['market_trend'].

This module is broker-agnostic for the global feeds (yfinance) and uses the Kite
session only for NIFTY gap + price action. It performs blocking network I/O, so
call get_trend() via asyncio.to_thread or rely on the async wrapper provided.
"""
from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Optional

import pandas as pd

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except Exception:  # pragma: no cover - yfinance optional
    _YF_AVAILABLE = False


# Default config — every value overridable under config['market_trend'].
_DEFAULTS = {
    "enable": False,
    # yfinance tickers per bucket. Empty list disables that bucket.
    "us_futures_tickers": ["ES=F", "NQ=F", "YM=F"],   # S&P / Nasdaq / Dow futures
    "us_index_tickers":   ["^GSPC", "^IXIC", "^DJI"],  # S&P500 / Nasdaq / Dow cash
    "asia_tickers":       ["^N225", "^HSI"],           # Nikkei / Hang Seng
    "gift_nifty_ticker":  "",        # e.g. a yfinance symbol if you have one; "" = skip yfinance path
    "nifty_yf_ticker":    "^NSEI",   # NIFTY 50 prior close (yfinance fallback for gap)
    # Component weights (relative; normalised over the components that resolve).
    "weights": {
        "gift_nifty":   0.30,
        "gap":          0.25,
        "us_futures":   0.18,
        "us_indices":   0.10,
        "asia":         0.10,
        "price_action": 0.07,
    },
    # % move that maps to a full ±1.0 component score (smaller = more sensitive).
    "scales": {
        "gift_nifty":   0.50,
        "gap":          0.50,
        "us_futures":   0.75,
        "us_indices":   0.75,
        "asia":         0.90,
        "price_action": 1.00,
    },
    # |composite| below this is NEUTRAL; above maps to Bullish/Bearish.
    "neutral_band": 0.15,
    # |composite| at/above this maps to the "Very" variant.
    "strong_band": 0.50,
}


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


class MarketTrendAnalyzer:
    """Computes a pre-market directional verdict (CE / PE / NEUTRAL)."""

    def __init__(self, kite, config: dict):
        self.kite = kite
        self.config = config or {}
        cfg = (self.config.get("market_trend") or {})
        # Shallow-merge user config over defaults (nested dicts merged one level).
        self.cfg = dict(_DEFAULTS)
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                merged = dict(self.cfg[k]); merged.update(v); self.cfg[k] = merged
            else:
                self.cfg[k] = v

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    async def get_trend(self) -> dict:
        """Async wrapper — runs the blocking analysis off the event loop."""
        return await asyncio.to_thread(self._analyze)

    def _analyze(self) -> dict:
        """
        Returns a verdict dict:
          {
            "verdict": "CE" | "PE" | "NEUTRAL",
            "sentiment": "Very Bullish"|"Bullish"|"Neutral"|"Bearish"|"Very Bearish",
            "composite": float in [-1, 1],
            "components": { name: {"pct": float|None, "score": float, "weight": float, "detail": str} },
            "summary": str,            # one-line human summary
          }
        """
        components: dict = {}

        # — Global feeds (yfinance) —
        if _YF_AVAILABLE:
            self._add_basket(components, "us_futures", self.cfg["us_futures_tickers"])
            self._add_basket(components, "us_indices", self.cfg["us_index_tickers"])
            self._add_basket(components, "asia",       self.cfg["asia_tickers"])
        else:
            logging.warning("[MarketTrend] yfinance unavailable — global cues skipped.")

        # — GIFT Nifty (best-effort) —
        gift_pct = self._gift_nifty_gap_pct()
        if gift_pct is not None:
            components["gift_nifty"] = self._mk(gift_pct, "gift_nifty",
                                                f"GIFT-implied gap {gift_pct:+.2f}%")

        # — Open gap from Kite (NIFTY LTP / pre-open vs prior close) —
        gap_pct = self._kite_gap_pct()
        if gap_pct is not None:
            components["gap"] = self._mk(gap_pct, "gap", f"NIFTY gap {gap_pct:+.2f}%")

        # — Price action (last few NIFTY daily candles) —
        pa_pct = self._price_action_pct()
        if pa_pct is not None:
            components["price_action"] = self._mk(pa_pct, "price_action",
                                                  f"3-day momentum {pa_pct:+.2f}%")

        # — Composite (weighted average over resolved components) —
        total_w = sum(c["weight"] for c in components.values())
        composite = (
            sum(c["score"] * c["weight"] for c in components.values()) / total_w
            if total_w > 0 else 0.0
        )
        composite = _clip(composite)

        sentiment, verdict = self._verdict(composite, bool(components))
        summary = self._summary(verdict, sentiment, composite, components)
        return {
            "verdict": verdict,
            "sentiment": sentiment,
            "composite": round(composite, 3),
            "components": components,
            "summary": summary,
        }

    # ------------------------------------------------------------------ #
    # Component helpers
    # ------------------------------------------------------------------ #
    def _mk(self, pct: float, name: str, detail: str) -> dict:
        scale  = float(self.cfg["scales"].get(name, 0.75)) or 0.75
        weight = float(self.cfg["weights"].get(name, 0.0))
        return {"pct": pct, "score": _clip(pct / scale), "weight": weight, "detail": detail}

    def _add_basket(self, components: dict, name: str, tickers: list) -> None:
        """Average the % change across a basket of yfinance tickers."""
        if not tickers or float(self.cfg["weights"].get(name, 0)) <= 0:
            return
        pcts, bits = [], []
        for t in tickers:
            p = self._yf_pct_change(t)
            if p is not None:
                pcts.append(p)
                bits.append(f"{t} {p:+.2f}%")
        if pcts:
            avg = sum(pcts) / len(pcts)
            components[name] = self._mk(avg, name, ", ".join(bits))

    @staticmethod
    def _yf_pct_change(ticker: str) -> Optional[float]:
        """Latest % change for a yfinance ticker (last price vs previous close)."""
        if not _YF_AVAILABLE:
            return None
        try:
            tk = yf.Ticker(ticker)
            # fast_info is cheap and works for futures + indices.
            fi = getattr(tk, "fast_info", None)
            last = prev = None
            if fi:
                last = fi.get("last_price") if hasattr(fi, "get") else getattr(fi, "last_price", None)
                prev = fi.get("previous_close") if hasattr(fi, "get") else getattr(fi, "previous_close", None)
            if not last or not prev:
                hist = tk.history(period="5d", interval="1d")
                if hist is None or hist.empty or len(hist) < 2:
                    return None
                last = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2])
            if not prev:
                return None
            return (float(last) - float(prev)) / float(prev) * 100.0
        except Exception as e:
            logging.debug(f"[MarketTrend] yf {ticker} failed: {e}")
            return None

    def _prior_nifty_close(self) -> Optional[float]:
        """Prior trading-day NIFTY 50 close via Kite (falls back to yfinance)."""
        try:
            from infra import get_instrument_token
            token = get_instrument_token(self.kite, "NIFTY 50", "NSE")
            today = datetime.date.today()
            hist = self.kite.historical_data(
                token, today - datetime.timedelta(days=10), today, "day"
            )
            df = pd.DataFrame(hist or [])
            if not df.empty:
                df["d"] = pd.to_datetime(df["date"]).dt.date
                prior = df[df["d"] < today]
                if not prior.empty:
                    return float(prior.iloc[-1]["close"])
        except Exception as e:
            logging.debug(f"[MarketTrend] Kite prior close failed: {e}")
        # Fallback: yfinance ^NSEI previous close.
        if _YF_AVAILABLE:
            try:
                hist = yf.Ticker(self.cfg["nifty_yf_ticker"]).history(period="5d", interval="1d")
                if hist is not None and not hist.empty:
                    return float(hist["Close"].iloc[-1])
            except Exception:
                pass
        return None

    def _kite_gap_pct(self) -> Optional[float]:
        """Open gap %: current NIFTY LTP (or pre-open) vs prior close."""
        try:
            from infra import get_instrument_token, safe_ltp
            token = get_instrument_token(self.kite, "NIFTY 50", "NSE")
            ltp = safe_ltp(self.kite, str(token))
            if ltp is None:
                data = self.kite.ltp(str(token)) or {}
                ltp = (data.get(str(token)) or {}).get("last_price")
            prior = self._prior_nifty_close()
            if ltp and prior:
                return (float(ltp) - prior) / prior * 100.0
        except Exception as e:
            logging.debug(f"[MarketTrend] Kite gap failed: {e}")
        return None

    def _gift_nifty_gap_pct(self) -> Optional[float]:
        """
        GIFT-Nifty implied gap vs prior NIFTY close. Best-effort: GIFT Nifty has
        no reliable free feed, so this uses a configured yfinance ticker if one
        is provided, else returns None (the verdict still works from the other
        cues). Set market_trend.gift_nifty_ticker to a working symbol to enable.
        """
        sym = (self.cfg.get("gift_nifty_ticker") or "").strip()
        if not sym or not _YF_AVAILABLE:
            return None
        try:
            tk = yf.Ticker(sym)
            fi = getattr(tk, "fast_info", None)
            last = fi.get("last_price") if (fi and hasattr(fi, "get")) else getattr(fi, "last_price", None)
            if not last:
                hist = tk.history(period="2d", interval="1d")
                last = float(hist["Close"].iloc[-1]) if (hist is not None and not hist.empty) else None
            prior = self._prior_nifty_close()
            if last and prior:
                return (float(last) - prior) / prior * 100.0
        except Exception as e:
            logging.debug(f"[MarketTrend] GIFT Nifty failed: {e}")
        return None

    def _price_action_pct(self) -> Optional[float]:
        """
        Simple price-action read: % change of the last completed NIFTY daily
        close vs the close 3 sessions earlier (short-term structure/momentum).
        """
        try:
            from infra import get_instrument_token
            token = get_instrument_token(self.kite, "NIFTY 50", "NSE")
            today = datetime.date.today()
            hist = self.kite.historical_data(
                token, today - datetime.timedelta(days=12), today, "day"
            )
            df = pd.DataFrame(hist or [])
            if df.empty:
                return None
            df["d"] = pd.to_datetime(df["date"]).dt.date
            closes = df[df["d"] < today]["close"].astype(float).tolist()
            if len(closes) < 4:
                return None
            return (closes[-1] - closes[-4]) / closes[-4] * 100.0
        except Exception as e:
            logging.debug(f"[MarketTrend] price action failed: {e}")
            return None

    # ------------------------------------------------------------------ #
    # Verdict + formatting
    # ------------------------------------------------------------------ #
    def _verdict(self, composite: float, have_data: bool) -> tuple:
        if not have_data:
            return "Neutral", "NEUTRAL"
        nb = float(self.cfg["neutral_band"])
        sb = float(self.cfg["strong_band"])
        if composite >= sb:
            return "Very Bullish", "CE"
        if composite >= nb:
            return "Bullish", "CE"
        if composite <= -sb:
            return "Very Bearish", "PE"
        if composite <= -nb:
            return "Bearish", "PE"
        return "Neutral", "NEUTRAL"

    @staticmethod
    def _summary(verdict: str, sentiment: str, composite: float, components: dict) -> str:
        if not components:
            return "No pre-market data available — verdict NEUTRAL (no trade bias)."
        action = {
            "CE": "BUY CALLS (CE) — bullish bias",
            "PE": "BUY PUTS (PE) — bearish bias",
            "NEUTRAL": "NO directional bias (Neutral)",
        }[verdict]
        return f"{action}  [score {composite:+.2f} → {sentiment}]"

    def format_breakdown(self, trend: dict) -> list:
        """Returns a list of display lines for the terminal banner."""
        lines = [
            "PRE-MARKET TREND  (price action + global cues)",
            f"  Verdict : {trend['summary']}",
        ]
        comps = trend.get("components") or {}
        if comps:
            lines.append("  Drivers :")
            for name, c in comps.items():
                arrow = "▲" if c["score"] > 0.05 else ("▼" if c["score"] < -0.05 else "•")
                lines.append(
                    f"     {arrow} {name:<13} score {c['score']:+.2f} (w{c['weight']:.2f})  {c['detail']}"
                )
        else:
            lines.append("  Drivers : none available (offline / no data).")
        return lines
