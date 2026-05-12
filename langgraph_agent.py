import logging
import json
import asyncio
import aiohttp
from typing import TypedDict
from rag_service import RAGService


# ---------------------------------------------------------------------------
# Regime-based deterministic fallback
# ---------------------------------------------------------------------------
# Mapped from market_conditions set -> recommended strategy. Used when the LLM
# is unavailable (rate limit, network, invalid response) and we must still pick
# something *defensible* without the LLM doing the regime reasoning for us.
#
# Order matters: the first rule whose condition-set is a subset of today's
# conditions wins. Place stricter / more specific rules first.
REGIME_FALLBACK_RULES = [
    # Macro event days — dedicated strategy regardless of VIX/IV
    ({"EVENT_FED_MEETING"}, "Opening_Range_Breakout"),
    ({"EVENT_RBI_POLICY"},  "Opening_Range_Breakout"),
    # High volatility regimes — favour reversal / vol-cluster plays
    ({"VIX_HIGH", "IV_HIGH"},     "Volatility_Cluster_Reversal"),
    ({"VIX_HIGH"},                "Volatility_Cluster_Reversal"),
    ({"VIX_MEDIUM", "IV_HIGH"},   "Reversal_Detector"),
    # Calm regimes — favour higher-frequency setups that actually fire often.
    # NR7 catches compression-then-expansion breakouts on quiet days.
    # VWAP_Reversion fires multiple times per day on a trending session.
    ({"VIX_LOW", "IV_LOW"},       "NR7_Compression"),
    ({"VIX_LOW"},                 "VWAP_Reversion"),
    ({"VIX_MEDIUM", "IV_LOW"},    "VWAP_Reversion"),
    ({"VIX_MEDIUM"},              "Gemini_Default"),
]


def regime_fallback_strategy(market_conditions: set) -> str | None:
    """First matching rule wins. Returns None if no rule matches."""
    if not market_conditions:
        return None
    for required, strategy in REGIME_FALLBACK_RULES:
        if required.issubset(market_conditions):
            return strategy
    return None


class LangGraphAgent:
    """AI agent using Google's Gemini API to recommend a strategy from a full suite."""

    def __init__(self, config, rag_service: RAGService):
        self.config = config
        self.rag_service = rag_service
        self.api_key = config.get('google_api', {}).get('api_key', "")
        self.model_name = "gemini-2.0-flash"

    def _smart_fallback(self, market_conditions: set) -> str:
        """
        Two-layer fallback when the LLM is unavailable:
          1. If a regime rule matches, use it.
          2. Else pick the strategy with best recent risk-adjusted score
             (computed by RAGService over the recency window).
          3. Else 'Gemini_Default' as last-resort.
        """
        regime_pick = regime_fallback_strategy(market_conditions)
        if regime_pick:
            logging.info(
                f"[Fallback] Regime rule matched conditions={set(market_conditions)} -> {regime_pick}"
            )
            return regime_pick

        try:
            stats = self.rag_service.compute_strategy_stats()
        except Exception as e:
            logging.warning(f"[Fallback] compute_strategy_stats failed: {e}")
            stats = {}

        if stats:
            best = max(stats.items(), key=lambda kv: kv[1]['score'])
            best_name, best_meta = best
            if best_meta['score'] > 0:
                logging.info(
                    f"[Fallback] Best recent score: '{best_name}' "
                    f"(score={best_meta['score']:.2f}, n={best_meta['trades']})"
                )
                return best_name
            logging.info(
                f"[Fallback] No strategy has positive recent score; using Gemini_Default."
            )
        else:
            logging.info(
                "[Fallback] No regime match, no recent stats available; using Gemini_Default."
            )
        return "Gemini_Default"

    async def get_recommended_strategy(self, market_conditions: set, user_prompt: str = None, rag_context: str = None):
        """
        Gets a strategy recommendation from the Gemini API, optionally augmented with
        RAG context. On any failure, falls back to a regime-rule + recent-score chain
        instead of a hardcoded Gemini_Default.
        """
        if not self.api_key:
            logging.error("[Gemini Agent] Google API key not found. Using smart fallback.")
            return self._smart_fallback(market_conditions)

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
                    f"Falling back to regime/score-based pick."
                )
                return self._smart_fallback(market_conditions)

            logging.info(f"[Gemini Agent] AI Recommended Strategy: {recommended_strategy}")
            return recommended_strategy

        except Exception as e:
            logging.error(
                f"[Gemini Agent] Error calling Gemini API: {e}. Using smart fallback."
            )
            return self._smart_fallback(market_conditions)
