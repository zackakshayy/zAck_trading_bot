import logging
import json
import asyncio
import aiohttp
from typing import Optional, Tuple

import pandas as pd

from rag_service import RAGService


# ---------------------------------------------------------------------------
# DETERMINISTIC STRATEGY SELECTOR (5-layer cascade)
# ---------------------------------------------------------------------------
# No LLM required. Same shape a discretionary trader would use:
#   1. Hard pins (event days, expiry day)
#   2. Open-gap override (gap-and-go days have a dedicated strategy)
#   3. Indicator overrides (on-screen patterns force their own strategy)
#   4. Regime table (3-D lookup over VIX x IV x Sentiment-family)
#   5. Last-resort default
#
# pick_strategy_deterministic() returns (strategy_name, reason). All inputs
# beyond market_conditions + sentiment are optional — missing data simply
# skips that layer rather than failing.
# ---------------------------------------------------------------------------


def _detect_indicator_override(df, excluded: Optional[set] = None) -> Optional[Tuple[str, str]]:
    """
    Builds the full list of indicator-pattern matches in priority order, then
    returns the first one whose strategy is not in `excluded`. This way, if
    NR7_Compression has been cooled down for the day but BB compression also
    fires, BB_Squeeze_Breakout gets picked instead — same Layer 3, different
    pattern. First non-excluded match wins.
    """
    if df is None or len(df) < 8:
        return None

    excluded = excluded or set()
    last = df.iloc[-1]
    candidates: list = []

    # 1. Bollinger-band compression: bandwidth materially below its own MA.
    if "bb_bandwidth" in df.columns and "bb_bandwidth_ma" in df.columns:
        bw, bw_ma = last.get("bb_bandwidth"), last.get("bb_bandwidth_ma")
        if pd.notna(bw) and pd.notna(bw_ma) and bw_ma > 0 and bw < 0.7 * bw_ma:
            candidates.append((
                "BB_Squeeze_Breakout",
                f"BB-bandwidth compression (bw={bw:.3f} < 0.7 x MA={bw_ma:.3f})",
            ))

    # 2. NR7: the narrowest of the last 8 bars is within the last 3 (fresh).
    try:
        ranges_8 = (df["high"] - df["low"]).iloc[-8:]
        if not ranges_8.isna().any():
            min_idx_in_8 = int(ranges_8.values.argmin())
            if min_idx_in_8 >= 5:
                candidates.append((
                    "NR7_Compression",
                    f"Fresh NR7 compression (narrowest bar in last 3, "
                    f"range={float(ranges_8.min()):.2f})",
                ))
    except Exception:
        pass

    # 3. RSI extreme on the underlying (textbook reversal zone).
    rsi = last.get("rsi")
    if pd.notna(rsi):
        rsi_f = float(rsi)
        if rsi_f > 78 or rsi_f < 22:
            candidates.append((
                "Reversal_Detector",
                f"RSI extreme ({rsi_f:.1f}) — outside (22, 78)",
            ))

    # 4. Volume spike vs 20-bar MA (smart-money footprint).
    vol = last.get("volume")
    vol_ma = last.get("volume_ma")
    if pd.notna(vol) and pd.notna(vol_ma) and vol_ma > 0 and float(vol) > 2.0 * float(vol_ma):
        candidates.append((
            "Volume_Spread_Analysis",
            f"Volume spike (vol={float(vol):.0f} > 2x MA={float(vol_ma):.0f})",
        ))

    # 5. Supertrend flipped on the last completed bar.
    if "supertrend_direction" in df.columns and len(df) >= 2:
        last_st = df.iloc[-1].get("supertrend_direction")
        prev_st = df.iloc[-2].get("supertrend_direction")
        if (pd.notna(last_st) and pd.notna(prev_st)
                and last_st != 0 and prev_st != 0 and last_st != prev_st):
            candidates.append((
                "Supertrend_MACD",
                f"Supertrend flipped ({int(prev_st)} -> {int(last_st)})",
            ))

    # First non-excluded match wins.
    for strat, reason in candidates:
        if strat not in excluded:
            return (strat, reason)
    return None


def _regime_table_pick(market_conditions: set, sentiment: str) -> Optional[str]:
    """
    3-D table lookup. Sentiment-family bucketing:
      Bull = {Bullish, Very Bullish}
      Bear = {Bearish, Very Bearish}
      Neutral handled upstream (no trade today).
    Returns the strategy name, or None if conditions don't match any cell.
    """
    is_bull = sentiment in ("Bullish", "Very Bullish")
    is_bear = sentiment in ("Bearish", "Very Bearish")
    if not (is_bull or is_bear):
        return None

    if "VIX_HIGH" in market_conditions:
        # All VIX_HIGH cells route to the same strategy — high vol = vol-cluster reversal.
        return "Volatility_Cluster_Reversal"

    iv_high = "IV_HIGH" in market_conditions

    if "VIX_MEDIUM" in market_conditions:
        if not iv_high:  # IV_LOW or unspecified
            return "VWAP_Reversion"
        return "EMA_Cross_RSI" if is_bull else "Supertrend_MACD"

    if "VIX_LOW" in market_conditions:
        if not iv_high:
            return "BB_Squeeze_Breakout" if is_bull else "NR7_Compression"
        return "Volume_Spread_Analysis" if is_bull else "RSI_Divergence"

    return None


def pick_strategy_deterministic(
    market_conditions: set,
    sentiment: str,
    is_expiry_day: bool = False,
    open_gap_pct: Optional[float] = None,
    underlying_bars=None,
    config: Optional[dict] = None,
    exclude_strategies: Optional[set] = None,
) -> Optional[Tuple[str, str]]:
    """
    Returns (strategy_name, reason) — first non-excluded match across the 5 layers.
    Returns None if every layer's candidate is in `exclude_strategies` (orchestrator
    should halt with "exhausted" message). `exclude_strategies` is typically the
    "cooled" set: strategies that were picked earlier today but produced zero
    non-HOLD signals during their evaluation window.
    """
    cfg = (config or {}).get("strategy_selector", {}) or {}
    gap_threshold = float(cfg.get("gap_override_pct", 0.8))
    excluded = set(exclude_strategies or [])

    def _take(name: str, reason: str) -> Optional[Tuple[str, str]]:
        return (name, reason) if name not in excluded else None

    # ----- Layer 1: hard pins -----
    if "EVENT_FED_MEETING" in market_conditions:
        pick = _take("Opening_Range_Breakout", "Hard pin: FOMC meeting day")
        if pick: return pick
    if "EVENT_RBI_POLICY" in market_conditions:
        pick = _take("Opening_Range_Breakout", "Hard pin: RBI policy day")
        if pick: return pick
    if is_expiry_day:
        pick = _take("RSI_Divergence",
                     "Hard pin: weekly expiry day (mean-reversion bias on gamma whipsaw)")
        if pick: return pick

    # ----- Layer 2: open-gap override -----
    if open_gap_pct is not None and abs(open_gap_pct) >= gap_threshold:
        pick = _take("Breakout_Prev_Day_HL",
                     f"Open-gap override: {open_gap_pct:+.2f}% (|gap| >= {gap_threshold}%)")
        if pick: return pick

    # ----- Layer 3: indicator overrides (filter happens INSIDE the detector) -----
    ind_pick = _detect_indicator_override(underlying_bars, excluded=excluded)
    if ind_pick:
        return (ind_pick[0], f"Indicator override: {ind_pick[1]}")

    # ----- Layer 4: regime table -----
    regime_pick = _regime_table_pick(market_conditions, sentiment)
    if regime_pick:
        pick = _take(regime_pick,
                     f"Regime table: VIX/IV/Sentiment match -> "
                     f"{sorted(market_conditions)} + {sentiment}")
        if pick: return pick

    # ----- Layer 5: last resort -----
    pick = _take("Gemini_Default", "Last-resort default (no other layer matched)")
    if pick: return pick

    # All layers exhausted by cooldown.
    return None


class LangGraphAgent:
    """AI agent using Google's Gemini API to recommend a strategy from a full suite."""

    def __init__(self, config, rag_service: RAGService):
        self.config = config
        self.rag_service = rag_service
        self.api_key = config.get('google_api', {}).get('api_key', "")
        self.model_name = "gemini-2.0-flash"

    def _deterministic_pick(self, market_conditions, sentiment, is_expiry_day,
                             open_gap_pct, underlying_bars,
                             exclude_strategies: Optional[set] = None) -> Optional[str]:
        """Wrap pick_strategy_deterministic with consistent logging. Returns
        None if every cascade layer is excluded — caller should treat that as
        'strategies exhausted, halt for the day'."""
        result = pick_strategy_deterministic(
            market_conditions=market_conditions,
            sentiment=sentiment,
            is_expiry_day=is_expiry_day,
            open_gap_pct=open_gap_pct,
            underlying_bars=underlying_bars,
            config=self.config,
            exclude_strategies=exclude_strategies,
        )
        if result is None:
            logging.error(
                f"[Selector] All cascade layers excluded by cooldown "
                f"(cooled={sorted(exclude_strategies or [])}). No strategy available."
            )
            return None
        strat, reason = result
        logging.info(f"[Selector] Deterministic pick: {strat} — {reason}")
        return strat

    async def get_recommended_strategy(
        self,
        market_conditions: set,
        sentiment: str = "Neutral",
        is_expiry_day: bool = False,
        open_gap_pct: Optional[float] = None,
        underlying_bars=None,
        exclude_strategies: Optional[set] = None,
        user_prompt: str = None,
        rag_context: str = None,
    ):
        """
        Returns a strategy name. Decision path:
          - If `strategy_selector.use_llm` is False (default) OR no Gemini key
            available: skip LLM entirely, return the deterministic cascade's pick.
          - If `use_llm` is True: try the LLM with the rich prompt below; on any
            failure (rate limit, network, invalid response) fall back to the
            deterministic cascade.
        """
        cfg = (self.config.get("strategy_selector", {}) or {})
        use_llm = bool(cfg.get("use_llm", False))

        if not use_llm or not self.api_key:
            return self._deterministic_pick(
                market_conditions, sentiment, is_expiry_day,
                open_gap_pct, underlying_bars, exclude_strategies,
            )

        logging.info(f"[Gemini Agent] Market Conditions: {market_conditions}. Recommending strategy...")

        prompt_sections = [
            "You are an expert intraday options trading strategist for the Indian NIFTY 50 index.",
            "Your task is to select the single best strategy for today based on the provided data.",
            f"\n**Today's Market Conditions:** {', '.join(market_conditions)}",
        ]
        
        # --- FIX: Conditionally add the RAG context to the prompt ---
        if rag_context:
            logging.info("[Gemini Agent] Using RAG context for strategy selection.")
            prompt_sections.append(f"\n**RAG Context (Historical Performance):**\n{rag_context}")
        else:
            logging.info("[Gemini Agent] Bypassing RAG context for strategy selection.")

        if user_prompt:
            prompt_sections.append(f"\n**User's Preference/Observation:** '{user_prompt}'")

        prompt_sections.append("\n**Available Strategies (and their primary purpose):**")
        prompt_sections.append(
            """
1.  **'Gemini_Default'**: A balanced, multi-indicator strategy (CPR, EMA, RSI Divergence).
2.  **'Supertrend_MACD'**: A strong trend-following strategy.
3.  **'Volatility_Cluster_Reversal'**: A counter-trend strategy for high volatility.
4.  **'Volume_Spread_Analysis'**: Detects smart money activity.
5.  **'EMA_Cross_RSI'**: A classic, fast-acting momentum strategy.
6.  **'Momentum_VWAP_RSI'**: A momentum strategy using VWAP + RSI confirmation.
7.  **'Breakout_Prev_Day_HL'**: A breakout strategy on previous day's high/low.
8.  **'Opening_Range_Breakout'**: A classic ORB strategy.
9.  **'BB_Squeeze_Breakout'**: A volatility breakout strategy.
10. **'MA_Crossover'**: A simple moving average crossover strategy.
11. **'RSI_Divergence'**: A pure reversal strategy on RSI divergence.
12. **'Reversal_Detector'**: A specialized reversal strategy for overextended trends.
13. **'VWAP_Reversion'**: HIGH-FREQUENCY intraday VWAP-reclaim play — fires multiple times per day in a trending session. Best on directional days with normal-to-low vol.
14. **'NR7_Compression'**: Compression-then-expansion breakout — looks for the narrowest range bar of the last 7 and buys/sells the breakout on volume. Best on low-volatility, low-IV days.
"""
        )
        prompt_sections.append("\nBased on all the above information, which single strategy name from the list has the highest probability of success today? Return only the name.")
        
        prompt = "\n".join(prompt_sections)
        
        try:
            api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

            async with aiohttp.ClientSession() as session:
                async with session.post(api_url, json=payload) as response:
                    response.raise_for_status()
                    result = await response.json()

            recommended_strategy = result["candidates"][0]["content"]["parts"][0]["text"].strip().replace("'", "").split('\n')[-1]

            valid_strategies = [
                "Gemini_Default", "Supertrend_MACD", "Volatility_Cluster_Reversal",
                "Volume_Spread_Analysis", "EMA_Cross_RSI", "Momentum_VWAP_RSI",
                "Breakout_Prev_Day_HL", "Opening_Range_Breakout", "BB_Squeeze_Breakout",
                "MA_Crossover", "RSI_Divergence", "Reversal_Detector",
                "VWAP_Reversion", "NR7_Compression",
            ]
            if recommended_strategy not in valid_strategies:
                logging.warning(
                    f"[Gemini Agent] LLM returned unknown strategy: '{recommended_strategy}'. "
                    f"Falling back to deterministic cascade."
                )
                return self._deterministic_pick(
                    market_conditions, sentiment, is_expiry_day,
                    open_gap_pct, underlying_bars, exclude_strategies,
                )

            logging.info(f"[Gemini Agent] AI Recommended Strategy: {recommended_strategy}")
            return recommended_strategy

        except Exception as e:
            logging.error(
                f"[Gemini Agent] Error calling Gemini API: {e}. Using deterministic cascade."
            )
            return self._deterministic_pick(
                market_conditions, sentiment, is_expiry_day,
                open_gap_pct, underlying_bars, exclude_strategies,
            )
