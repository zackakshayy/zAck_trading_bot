"""
Structure map — the day's support / resistance terrain.

Pure functions (no I/O) that turn the data the bot ALREADY fetches — CPR
pivots, previous-day high/low, and option-chain OI walls — into the nearest
levels above and below the current spot. These are the market's own
crowd-sourced support/resistance: the strikes where option writers have the
most open interest (walls), and the prior session's range extremes.

Used for two things the bot previously ignored:
  1. Entry validation — is there ROOM between spot and the nearest adverse
     level for the trade to actually reach a worthwhile target, or is it
     capped from the start (buying a CE two points under a massive call wall)?
  2. Exit targeting — the nearest favourable level IS the natural profit
     target, far better than a fixed premium %.  A CE's target is the nearest
     resistance above; a PE's is the nearest support below.
"""
from __future__ import annotations

from typing import Optional, Tuple

# Signals/directions that mean "bullish / long CE" vs "bearish / long PE".
_BULLISH = {"BUY", "CE", "BULLISH", "LONG"}


def _levels_above(spot, cpr, prev_high, call_walls, margin) -> list:
    """All known levels strictly above spot (+margin), nearest first."""
    out = []
    cpr = cpr or {}
    for key in ("r1", "r2", "r3", "tc", "pivot"):
        v = cpr.get(key)
        if v and float(v) > spot + margin:
            out.append((float(v), f"CPR-{key.upper()}"))
    if prev_high and float(prev_high) > spot + margin:
        out.append((float(prev_high), "PDH"))
    for w in (call_walls or []):
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if wv > spot + margin:
            out.append((wv, "CALL-WALL"))
    return sorted(out, key=lambda t: t[0])  # ascending → nearest above first


def _levels_below(spot, cpr, prev_low, put_walls, margin) -> list:
    """All known levels strictly below spot (−margin), nearest first."""
    out = []
    cpr = cpr or {}
    for key in ("s1", "s2", "s3", "bc", "pivot"):
        v = cpr.get(key)
        if v and float(v) < spot - margin:
            out.append((float(v), f"CPR-{key.upper()}"))
    if prev_low and float(prev_low) < spot - margin:
        out.append((float(prev_low), "PDL"))
    for w in (put_walls or []):
        try:
            wv = float(w)
        except (TypeError, ValueError):
            continue
        if wv < spot - margin:
            out.append((wv, "PUT-WALL"))
    return sorted(out, key=lambda t: -t[0])  # descending → nearest below first


def nearest_target(spot, direction, *, cpr=None, prev_high=None, prev_low=None,
                   call_walls=None, put_walls=None, margin=2.0
                   ) -> Optional[Tuple[float, str]]:
    """
    The favourable profit target for the trade:
      • bullish (CE/BUY)  → nearest RESISTANCE above (where the up-move stalls)
      • bearish (PE/SELL) → nearest SUPPORT below (where the down-move stalls)
    Returns (level, label) or None when no level is known on that side.
    """
    if spot is None:
        return None
    spot = float(spot)
    bullish = str(direction).upper() in _BULLISH
    levels = (_levels_above(spot, cpr, prev_high, call_walls, margin) if bullish
              else _levels_below(spot, cpr, prev_low, put_walls, margin))
    return levels[0] if levels else None


def room_to_target(spot, direction, **kw) -> Optional[float]:
    """Absolute NIFTY-point distance from spot to the nearest favourable level.
    None when no level is known (caller should EXCLUDE the factor, not penalise)."""
    t = nearest_target(spot, direction, **kw)
    if t is None:
        return None
    return abs(t[0] - float(spot))
