"""
Deterministic post-mortem for losing trades.

Every section is computed from the trade record + the underlying bars during
the holding window — no LLM, no network. The bot's Gemini access has been
unreliable, so the analysis must stand on its own. An optional LLM paragraph
can be appended by the caller if it happens to be available.

Public API:
    build_loss_report(trade, underlying_df, market_conditions, sentiment) -> str
        Returns a formatted multi-section plain-text report.
    report_to_html(report_text) -> str
        Wraps the plain-text report in a <pre> block for the email body.
"""
from __future__ import annotations

import datetime
import logging
from typing import Optional

import pandas as pd


def _money(x) -> str:
    try:
        return f"Rs.{float(x):,.2f}"
    except Exception:
        return "Rs.?"


def _parse_dt(value) -> Optional[pd.Timestamp]:
    """Parse a value to a tz-naive pandas Timestamp, or None."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value)
        if getattr(ts, "tz", None) is not None:
            ts = ts.tz_localize(None)
        return ts
    except Exception:
        return None


# If the nearest bar is further than this from the requested time, the bar
# data simply doesn't cover that window — return None rather than a misleading
# value snapped to the closest (possibly hours-away) bar.
_MAX_BAR_GAP_MINUTES = 20


def _underlying_close_at(df: Optional[pd.DataFrame], when) -> Optional[float]:
    """
    Returns the close of the bar nearest to `when`. Handles tz-aware indices
    (Kite returns IST-aware timestamps) by normalising both sides to tz-naive.
    Returns None when the nearest bar is more than `_MAX_BAR_GAP_MINUTES` away
    — i.e., the bar set doesn't actually cover the requested timestamp.
    """
    if df is None or df.empty or "close" not in df.columns:
        return None
    when_ts = _parse_dt(when)
    if when_ts is None:
        return None
    try:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            return float(df["close"].iloc[-1])
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        pos = idx.get_indexer([when_ts], method="nearest")
        if len(pos) == 0 or pos[0] < 0:
            return None
        nearest_ts = idx[pos[0]]
        gap_min = abs((nearest_ts - when_ts).total_seconds()) / 60.0
        if gap_min > _MAX_BAR_GAP_MINUTES:
            logging.debug(
                f"loss_analyzer: nearest bar is {gap_min:.0f} min from "
                f"{when_ts} — bar data doesn't cover this window."
            )
            return None
        return float(df["close"].iloc[pos[0]])
    except Exception as e:
        logging.debug(f"loss_analyzer: underlying-close lookup failed: {e}")
        return None


def _holding_minutes(entry_time, exit_time) -> Optional[float]:
    e, x = _parse_dt(entry_time), _parse_dt(exit_time)
    if e is None or x is None:
        return None
    delta = (x - e).total_seconds() / 60.0
    return delta if delta >= 0 else None


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

_SEP = "=" * 80
_SUB = "-" * 80


def build_loss_report(trade: dict, underlying_df: Optional[pd.DataFrame],
                      market_conditions, sentiment: str) -> str:
    """
    trade keys used (all optional-safe):
      Symbol, TradeType ('BUY'=long CE / 'SELL'=long PE), Strategy,
      EntryPrice, ExitPrice, Quantity, ProfitLoss, ExitReason,
      entry_time, Timestamp (exit time), high_water_mark, initial_stop_loss.
    """
    sym = trade.get("Symbol", "?")
    ttype = (trade.get("TradeType") or "?").upper()
    strategy = trade.get("Strategy", "?")
    entry = float(trade.get("EntryPrice", 0) or 0)
    exit_ = float(trade.get("ExitPrice", 0) or 0)
    qty = int(trade.get("Quantity", 0) or 0)
    pnl = float(trade.get("ProfitLoss", 0) or 0)
    exit_reason = trade.get("ExitReason", "?")
    hwm = trade.get("high_water_mark")
    hard_sl = trade.get("initial_stop_loss")
    entry_time = trade.get("entry_time")
    exit_time = trade.get("Timestamp")

    is_call = ttype == "BUY"             # bot always BUYS options; BUY=CE, SELL=PE
    option_kind = "CALL (bullish bet)" if is_call else "PUT (bearish bet)"

    pnl_per_share = (exit_ - entry) if entry else 0.0
    pnl_pct = (pnl_per_share / entry * 100.0) if entry else 0.0
    held_min = _holding_minutes(entry_time, exit_time)

    L = []
    L.append(_SEP)
    L.append("  TRADE LOSS POST-MORTEM")
    L.append(_SEP)
    L.append(f"  Symbol:        {sym}   ({option_kind})")
    L.append(f"  Strategy:      {strategy}")
    L.append(f"  Entry:         {_money(entry)} x {qty} qty")
    L.append(f"  Exit:          {_money(exit_)} x {qty} qty")
    if held_min is not None:
        L.append(f"  Holding time:  {held_min:.0f} minutes")
    L.append(f"  P&L:           {_money(pnl)}   ({pnl_pct:+.1f}% on premium)")
    L.append(f"  Exit reason:   {_decode_exit_reason(exit_reason)}")
    L.append("")

    # ---- 1. How the trade exited ----
    L.append(_SUB)
    L.append("  1. HOW THE TRADE EXITED")
    L.append(_SUB)
    L.extend(_section_exit(exit_reason, entry, exit_, hwm, hard_sl, held_min))
    L.append("")

    # ---- 2. What the underlying did ----
    L.append(_SUB)
    L.append("  2. WHAT THE UNDERLYING DID")
    L.append(_SUB)
    u_entry = _underlying_close_at(underlying_df, entry_time)
    u_exit = _underlying_close_at(underlying_df, exit_time)
    favorable, u_section = _section_underlying(is_call, u_entry, u_exit)
    L.extend(u_section)
    L.append("")

    # ---- 3. Why the premium fell ----
    L.append(_SUB)
    L.append("  3. WHY THE PREMIUM FELL (direction vs decay)")
    L.append(_SUB)
    L.extend(_section_attribution(favorable, u_entry, u_exit, pnl_per_share, held_min))
    L.append("")

    # ---- 4. Market context ----
    L.append(_SUB)
    L.append("  4. MARKET CONTEXT")
    L.append(_SUB)
    L.extend(_section_context(market_conditions, sentiment, favorable))
    L.append("")

    # ---- 5. What to improve ----
    L.append(_SUB)
    L.append("  5. WHAT TO IMPROVE NEXT TIME")
    L.append(_SUB)
    L.extend(_section_improvements(
        strategy, exit_reason, favorable, held_min, entry, exit_, hwm,
        market_conditions
    ))
    L.append(_SEP)
    return "\n".join(L)


def _decode_exit_reason(reason: str) -> str:
    return {
        "SL_M_TRIGGERED": "Broker SL-M order triggered",
        "INDICATOR_OR_SOFTWARE_SL": "Software stop-loss / indicator exit",
        "PAPER": "Paper-mode exit",
        "UNKNOWN": "Unknown",
    }.get(reason, reason or "Unknown")


def _section_exit(exit_reason, entry, exit_, hwm, hard_sl, held_min) -> list:
    out = []
    # Did the trade ever go green?
    if hwm is not None and entry:
        try:
            hwm = float(hwm)
            peak_pct = (hwm - entry) / entry * 100.0
            if peak_pct > 0.5:
                out.append(
                    f"  Premium peaked at {_money(hwm)} — a brief {peak_pct:+.1f}% gain "
                    f"— before reversing and ending the trade in a loss."
                )
                out.append(
                    "  The trade WAS in profit at one point; the gain was given back. "
                    "A partial-profit rule would have salvaged some of it."
                )
            else:
                out.append(
                    f"  Premium never moved meaningfully into profit "
                    f"(peak {_money(hwm)}, only {peak_pct:+.1f}%). The trade was "
                    f"underwater for essentially its entire life."
                )
        except Exception:
            pass
    if hard_sl is not None:
        try:
            out.append(f"  Hard stop-loss was set at {_money(hard_sl)}.")
        except Exception:
            pass
    # Speed of the loss
    if held_min is not None:
        if held_min <= 15:
            out.append(
                f"  Time-in-trade was only {held_min:.0f} min — a FAST reversal. "
                f"The market turned against the position almost immediately, "
                f"which usually points to poor entry timing."
            )
        elif held_min <= 60:
            out.append(
                f"  Time-in-trade was {held_min:.0f} min — a moderate-duration loss."
            )
        else:
            out.append(
                f"  Time-in-trade was {held_min:.0f} min — a slow bleed. The "
                f"directional thesis never materialised and the position decayed."
            )
    if not out:
        out.append("  (Insufficient data to characterise the exit.)")
    return out


def _section_underlying(is_call: bool, u_entry, u_exit):
    out = []
    if u_entry is None or u_exit is None:
        out.append("  Underlying bar data for the holding window was unavailable,")
        out.append("  so direction attribution could not be computed.")
        return None, out

    move = u_exit - u_entry
    move_pct = (move / u_entry * 100.0) if u_entry else 0.0
    out.append(f"  Underlying at entry:  {u_entry:,.2f}")
    out.append(f"  Underlying at exit:   {u_exit:,.2f}")
    out.append(f"  Underlying move:      {move:+,.2f} points ({move_pct:+.2f}%)")
    out.append("")

    favorable = (move > 0) if is_call else (move < 0)
    if favorable:
        out.append(
            "  The underlying moved IN FAVOUR of the position — yet the trade "
            "still lost. The loss is NOT a directional miss (see section 3)."
        )
    else:
        direction_word = "DOWN" if is_call else "UP"
        bet_word = "bullish" if is_call else "bearish"
        out.append(
            f"  You held a {bet_word} option, but the underlying went "
            f"{direction_word}. The market moved AGAINST the position — this "
            f"is a directional miss and the primary cause of the loss."
        )
    return favorable, out


def _section_attribution(favorable, u_entry, u_exit, pnl_per_share, held_min) -> list:
    out = []
    out.append(f"  Premium lost:  {_money(abs(pnl_per_share))} per share")
    if favorable is None:
        out.append("  Direction vs decay cannot be split without underlying data.")
        return out

    if favorable:
        out.append("")
        out.append("  The underlying moved your way, but the option still lost value.")
        out.append("  That means the loss came from one or both of:")
        out.append("    - THETA (time decay): every minute an option is held, it loses")
        out.append("      a little extrinsic value. Worse near expiry.")
        out.append("    - IV CRUSH: implied volatility dropped, deflating the premium")
        out.append("      even though the direction was right.")
        out.append("  ACTION POINTER: the strike may have been too far OTM (low delta)")
        out.append("  to capture the move. A deeper-ITM strike (delta 0.55-0.65) would")
        out.append("  have responded more to the favourable underlying move.")
    else:
        out.append("")
        out.append("  The underlying moved against the position. The loss is")
        out.append("  DIRECTIONAL — the trade's core thesis was wrong for this window.")
        if held_min is not None and held_min > 60:
            out.append("  Time decay (theta) added a secondary drag over the long hold.")
    return out


def _section_context(market_conditions, sentiment, favorable) -> list:
    out = []
    conds = sorted(market_conditions) if market_conditions else []
    vix = next((c for c in conds if c.startswith("VIX_")), "VIX_UNKNOWN")
    iv = next((c for c in conds if c.startswith("IV_")), "IV_UNKNOWN")
    events = [c for c in conds if c.startswith("EVENT_")]
    out.append(f"  VIX regime:    {vix}")
    out.append(f"  IV regime:     {iv}")
    out.append(f"  Day sentiment: {sentiment}")
    if events:
        out.append(f"  Event flags:   {', '.join(events)}")
    out.append("")
    if vix == "VIX_HIGH":
        out.append(
            "  VIX was HIGH — option premiums were inflated and price action "
            "choppy. High-VIX days are hostile to directional option buyers; "
            "consider sizing down or sitting out."
        )
    if iv == "IV_HIGH":
        out.append(
            "  IV was elevated — you paid a rich premium. If the move didn't "
            "come fast, IV mean-reversion (crush) worked against you."
        )
    if favorable is False:
        out.append(
            "  The intraday move during the hold contradicted the day's "
            "sentiment read. Sentiment/price mismatch is a classic chop signal."
        )
    if not events and vix != "VIX_HIGH" and iv != "IV_HIGH":
        out.append(
            "  Conditions were unremarkable — no event, normal vol. The loss "
            "is best explained by the trade itself, not the regime."
        )
    return out


def _section_improvements(strategy, exit_reason, favorable, held_min,
                          entry, exit_, hwm, market_conditions) -> list:
    """Rule-based, concrete improvement bullets — the actionable payoff."""
    tips = []

    # Fast reversal
    if held_min is not None and held_min <= 15:
        tips.append(
            "Entry timing: the trade reversed within 15 min. Add a confirmation "
            "filter — wait for a second bar to hold past the trigger level "
            "before entering, instead of entering on the first breakout bar."
        )
    # Slow bleed
    if held_min is not None and held_min > 60 and favorable is False:
        tips.append(
            "Time stop: this was a slow bleed with no directional progress. "
            "Add a rule to exit after ~30-40 min if the trade hasn't moved "
            "meaningfully in your favour — don't let theta grind it down."
        )
    # Gave back a gain
    if hwm is not None and entry:
        try:
            peak_pct = (float(hwm) - entry) / entry * 100.0
            if peak_pct > 5:
                tips.append(
                    f"Profit protection: premium reached {peak_pct:+.0f}% before "
                    f"reversing. Book 50% of the position at +{int(peak_pct/2)}% "
                    f"and trail the rest — turn round-trips into partial wins."
                )
        except Exception:
            pass
    # Favourable underlying but still lost -> strike selection
    if favorable is True:
        tips.append(
            "Strike selection: the underlying moved your way but the option "
            "didn't follow. Move to a higher-delta (deeper-ITM) strike so the "
            "premium tracks the underlying instead of bleeding theta."
        )
    # Strategy-specific
    sl = (strategy or "").lower()
    if "nr7" in sl or "breakout" in sl or "squeeze" in sl:
        tips.append(
            f"{strategy} is a breakout strategy — most breakouts fail in "
            f"range-bound conditions. Require stronger volume confirmation "
            f"(>1.5x the 20-bar average, not 1.2x) before taking the entry."
        )
    if "reversal" in sl or "divergence" in sl:
        tips.append(
            f"{strategy} is a counter-trend strategy — it fights momentum. "
            f"Only take it when the trend is genuinely overextended; require a "
            f"clear structure break (not just an indicator reading) to confirm."
        )
    if "vwap" in sl:
        tips.append(
            f"{strategy} depends on a clean VWAP reclaim/loss. A whipsaw across "
            f"VWAP is the trap — require the reclaim bar to also close above the "
            f"prior bar's high (for longs) for a stronger signal."
        )
    # High-VIX context
    conds = set(market_conditions or [])
    if "VIX_HIGH" in conds:
        tips.append(
            "Regime: VIX was high. Either skip new entries above your VIX gate "
            "or halve position size on high-VIX days — premiums are inflated "
            "and whipsaw risk is elevated."
        )
    # Always-relevant baseline
    tips.append(
        "Process: one losing trade is noise, not signal. Log it, look for the "
        "PATTERN across 15-20 trades before changing strategy parameters."
    )

    return [f"  - {t}" for t in tips]


def report_to_html(report_text: str) -> str:
    """Wrap the plain-text report in a monospace HTML block for email."""
    safe = (report_text
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    return (
        "<html><body>"
        "<pre style=\"font-family:'Courier New',monospace;font-size:13px;"
        "line-height:1.4;background:#f7f7f7;padding:16px;border-radius:6px;\">"
        f"{safe}"
        "</pre></body></html>"
    )
