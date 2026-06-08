"""
Market-regime classifier (playbook Part 3 — "the mandatory first step every day").

Maps the morning's context into ONE of five regimes and the buy-only strategy
family that fits it — or stands the bot down when there is no clean regime:

  1. TRENDING     → directional ATM momentum buys
  2. RANGE        → wait-for-breakout only (no naked buys inside a range)
  3. PRE_EVENT    → sit out (buy-only can't run the doc's straddle/spread)
  4. POST_EVENT   → wait-and-trade the confirmed direction after the open chaos
  5. EXPIRY       → expiry gamma scalp
  UNCLEAR         → no clean regime → no trade

Pure logic (no I/O): the orchestrator gathers the inputs (day-quality from its
existing TA, VIX, expiry, event calendar, direction) and this returns a
RegimeResult. Strategy families reuse the bot's existing strategy names so the
deterministic selector simply picks within the regime-appropriate set.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# Canonical strategy universe (must match strategy_factory / langgraph_agent).
ALL_STRATEGIES = (
    "Gemini_Default", "Supertrend_MACD", "Volatility_Cluster_Reversal",
    "Volume_Spread_Analysis", "EMA_Cross_RSI", "Momentum_VWAP_RSI",
    "Breakout_Prev_Day_HL", "Opening_Range_Breakout", "BB_Squeeze_Breakout",
    "MA_Crossover", "RSI_Divergence", "Reversal_Detector",
    "VWAP_Reversion", "NR7_Compression", "Expiry_Momentum_Scalp",
)

# Regime-appropriate, BUY-ONLY strategy families.
# Trend-following set for normal/high vol (VIX ≥ low_vix_threshold).
_DIRECTIONAL = ("EMA_Cross_RSI", "Supertrend_MACD", "Momentum_VWAP_RSI",
                "Breakout_Prev_Day_HL", "Gemini_Default")
# Low-vol "grind" trending: VWAP-anchored momentum / mean-reversion fire more
# readily than CPR breakouts when the market isn't moving much. This keeps the
# cascade's natural VIX_LOW pick (VWAP_Reversion) INSIDE the regime family
# instead of falling through to the last-resort Gemini_Default.
_DIRECTIONAL_LOWVOL = ("Momentum_VWAP_RSI", "VWAP_Reversion", "EMA_Cross_RSI",
                       "Gemini_Default")
_POST_EVENT  = ("EMA_Cross_RSI", "Supertrend_MACD", "Momentum_VWAP_RSI",
                "Breakout_Prev_Day_HL")
_BREAKOUT    = ("Breakout_Prev_Day_HL", "Opening_Range_Breakout",
                "NR7_Compression", "BB_Squeeze_Breakout")
_EXPIRY      = ("Expiry_Momentum_Scalp",)


@dataclass
class RegimeResult:
    regime: str                       # TRENDING | RANGE | PRE_EVENT | POST_EVENT | EXPIRY | UNCLEAR
    direction: str = "NEUTRAL"        # BULLISH | BEARISH | NEUTRAL
    clean: bool = True                # False only for UNCLEAR
    sit_out: bool = False             # True → stand down for the day
    reason: str = ""
    allowed_strategies: Tuple[str, ...] = field(default_factory=tuple)
    entry_not_before: Optional[str] = None   # "HH:MM" (post-event wait gate)

    def excluded(self) -> set:
        """Strategies to EXCLUDE from the selector to keep it inside this regime."""
        if not self.allowed_strategies:
            return set(ALL_STRATEGIES)
        return set(ALL_STRATEGIES) - set(self.allowed_strategies)


def _direction(hint: Optional[str]) -> str:
    h = (hint or "").strip().lower()
    if h in ("bullish", "very bullish", "ce", "buy", "up"):
        return "BULLISH"
    if h in ("bearish", "very bearish", "pe", "sell", "down"):
        return "BEARISH"
    return "NEUTRAL"


class RegimeClassifier:
    def __init__(self, config: dict):
        self.cfg = ((config or {}).get("regime") or {})

    def classify(self, *, day_quality: str, vix: float = 0.0,
                 is_expiry_day: bool = False, dte: Optional[int] = None,
                 event_today: Optional[str] = None,
                 days_to_next_event: Optional[int] = None,
                 direction_hint: Optional[str] = None,
                 gap_pct: Optional[float] = None) -> RegimeResult:

        require_clean   = bool(self.cfg.get("require_clean_regime", True))
        pre_event_days  = int(self.cfg.get("pre_event_days", 3))
        sit_out_pre     = bool(self.cfg.get("sit_out_pre_event", True))
        expiry_dte      = int(self.cfg.get("expiry_dte_threshold", 1))
        post_event_wait = str(self.cfg.get("post_event_wait_until", "10:30"))
        direction       = _direction(direction_hint)
        vix_s = f"VIX {vix:.1f}" if vix else "VIX n/a"

        # 1) EXPIRY — overrides everything (0–1 DTE gamma environment).
        if is_expiry_day or (dte is not None and dte <= expiry_dte):
            return RegimeResult(
                regime="EXPIRY", direction=direction, clean=True, sit_out=False,
                allowed_strategies=_EXPIRY,
                reason=f"Expiry pressure (DTE≈{0 if is_expiry_day else dte}) — gamma scalp only ({vix_s}).",
            )

        # 2) POST_EVENT — today is an event day → wait, then trade the confirmed move.
        if event_today:
            return RegimeResult(
                regime="POST_EVENT", direction=direction, clean=True, sit_out=False,
                allowed_strategies=_POST_EVENT, entry_not_before=post_event_wait,
                reason=(f"Event day ({event_today}) — wait-and-trade after {post_event_wait}; "
                        f"avoid the opening chaos / IV crush ({vix_s})."),
            )

        # 3) PRE_EVENT — a known event is near. Buy-only can't run the straddle, so sit out.
        if days_to_next_event is not None and 0 < days_to_next_event <= pre_event_days:
            return RegimeResult(
                regime="PRE_EVENT", direction=direction, clean=True,
                sit_out=sit_out_pre,
                allowed_strategies=() if sit_out_pre else _DIRECTIONAL,
                reason=(f"Pre-event accumulation — major event in {days_to_next_event} day(s). "
                        + ("Buy-only has no straddle here → sitting out." if sit_out_pre
                           else f"Directional only, {vix_s}.")),
            )

        # 4) TRENDING — directional momentum buying. The family is VIX-aware:
        #    low-vol grinds favour VWAP-anchored plays over CPR/trend breakouts.
        if day_quality == "TRENDING":
            low_vix = float(self.cfg.get("low_vix_threshold", 16))
            if vix and 0 < vix < low_vix:
                fam, tag = _DIRECTIONAL_LOWVOL, "low-vol grind: VWAP-anchored"
            else:
                fam, tag = _DIRECTIONAL, "directional ATM buys"
            return RegimeResult(
                regime="TRENDING", direction=direction, clean=True, sit_out=False,
                allowed_strategies=fam,
                reason=f"Trending market ({direction.title()}, {vix_s}) — {tag}.",
            )

        # 5) RANGE — wait for a breakout; no naked buys inside the range.
        if day_quality == "RANGE":
            return RegimeResult(
                regime="RANGE", direction=direction, clean=True, sit_out=False,
                allowed_strategies=_BREAKOUT,
                reason=f"Range-bound ({vix_s}) — breakout-watch only; no naked buys inside the range.",
            )

        # 6) UNCLEAR — choppy / undefined → no clean regime.
        return RegimeResult(
            regime="UNCLEAR", direction=direction, clean=False,
            sit_out=require_clean, allowed_strategies=(),
            reason=(f"No clean regime (day_quality={day_quality or 'UNKNOWN'}, {vix_s}) — "
                    + ("standing down; forcing trades into chop bleeds theta + costs."
                       if require_clean else "proceeding with caution (require_clean_regime off).")),
        )
