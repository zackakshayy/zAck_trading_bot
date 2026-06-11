"""
Pre-market intelligence briefing.

Printed at bot startup (and refreshed automatically at 09:00 if the bot was
started during the pre-market window). Answers, before any trade is taken:

  1. How did the top-5 NIFTY heavyweights move yesterday?
  2. Do any of them have corporate events today / in the next few days
     (earnings, board meetings, dividends) that could move the whole index?
  3. Is a big index move likely today (VIX + event density + gap behaviour)?
  4. Where are NIFTY's nearest support / resistance levels (CPR + prev-day H/L)?
  5. What regime/strategy did the bot pick in response?

Everything degrades gracefully: every section is best-effort, and a failed
fetch prints "[unavailable]" instead of blocking startup. The full payload is
also written to state/premarket_briefing.json for the dashboard.
"""

import asyncio
import datetime
import logging
from typing import Optional

import pandas as pd

from indicators import calculate_cpr
from infra import (
    atomic_write_json,
    get_instrument_token,
    is_nse_holiday,
    state_path,
)

PREMARKET_FILE = state_path("premarket_briefing.json")

# Top NIFTY constituents by index weight. Stable enough to hardcode as a
# default; override via config premarket_briefing.components if weights shift.
DEFAULT_COMPONENTS = ["HDFCBANK", "RELIANCE", "ICICIBANK", "INFY", "TCS"]

_NSE_EVENT_URL = "https://www.nseindia.com/api/event-calendar"
_NSE_HOME_URL = "https://www.nseindia.com"
_NSE_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar",
}


def _prev_trading_day(today: datetime.date) -> datetime.date:
    d = today - datetime.timedelta(days=1)
    while d.weekday() >= 5 or is_nse_holiday(d):
        d -= datetime.timedelta(days=1)
    return d


class PreMarketBriefing:
    """Builds and prints the pre-market intelligence brief."""

    def __init__(self, kite, config: dict):
        self.kite = kite
        self.config = config
        self.cfg = (config.get("premarket_briefing") or {})
        self.components = list(self.cfg.get("components") or DEFAULT_COMPONENTS)
        self.event_lookahead_days = int(self.cfg.get("event_lookahead_days", 3))
        self.last_generated_at: Optional[datetime.datetime] = None

    # ------------------------------------------------------------------ #
    # Section fetchers — each is best-effort and returns a partial result #
    # ------------------------------------------------------------------ #

    async def _daily_bars(self, token: int, days: int = 12) -> pd.DataFrame:
        to_date = datetime.date.today()
        from_date = to_date - datetime.timedelta(days=days)
        data = await asyncio.to_thread(
            self.kite.historical_data, token, from_date, to_date, "day"
        )
        return pd.DataFrame(data)

    async def _component_moves(self) -> list:
        """Yesterday's % move for each top-5 component, vs its own 5-day rhythm."""
        out = []
        for sym in self.components:
            try:
                token = get_instrument_token(self.kite, sym, "NSE")
                df = await self._daily_bars(token)
                if len(df) < 3:
                    raise ValueError("not enough daily bars")
                df["date"] = pd.to_datetime(df["date"]).dt.date
                # Use only completed sessions (exclude any partial bar for today).
                df = df[df["date"] < datetime.date.today()]
                last, prev = df.iloc[-1], df.iloc[-2]
                pct = (last["close"] - prev["close"]) / prev["close"] * 100.0
                # Average absolute daily move over the prior 5 sessions — an
                # outsized move yesterday often carries follow-through risk.
                hist = df["close"].pct_change().abs().tail(6).head(5)
                avg_abs = float(hist.mean() * 100.0) if not hist.empty else 0.0
                out.append({
                    "symbol": sym,
                    "pct_change": round(float(pct), 2),
                    "close": round(float(last["close"]), 2),
                    "avg_abs_move_5d": round(avg_abs, 2),
                    "outsized": bool(avg_abs > 0 and abs(pct) > 1.8 * avg_abs),
                })
            except Exception as e:
                logging.debug(f"[PreMarket] component fetch failed for {sym}: {e}")
                out.append({"symbol": sym, "pct_change": None})
        return out

    async def _fetch_nse_events(self) -> dict:
        """
        Corporate events (earnings / board meetings / dividends) for the top-5
        components from NSE's public event-calendar endpoint. NSE is a
        browser-first site: we warm a session against the homepage to collect
        cookies first. Any failure returns {} — events are nice-to-have.
        """
        def _fetch():
            import requests
            s = requests.Session()
            s.headers.update(_NSE_HEADERS)
            s.get(_NSE_HOME_URL, timeout=8)  # cookie warm-up
            today = datetime.date.today()
            until = today + datetime.timedelta(days=self.event_lookahead_days)
            r = s.get(
                _NSE_EVENT_URL,
                params={
                    "index": "equities",
                    "from_date": today.strftime("%d-%m-%Y"),
                    "to_date": until.strftime("%d-%m-%Y"),
                },
                timeout=10,
            )
            r.raise_for_status()
            return r.json()

        try:
            raw = await asyncio.to_thread(_fetch)
        except Exception as e:
            logging.info(f"[PreMarket] NSE event calendar unavailable ({e}); "
                         f"brief will print without events.")
            return {}

        rows = raw if isinstance(raw, list) else (raw or {}).get("data", [])
        events: dict = {}
        today_str = datetime.date.today().strftime("%d-%b-%Y")
        for row in rows or []:
            try:
                sym = str(row.get("symbol", "")).upper()
                if sym not in self.components:
                    continue
                date_s = str(row.get("date", "") or row.get("bm_date", ""))
                purpose = str(row.get("purpose", "") or row.get("bm_purpose", "")).strip()
                is_today = date_s.startswith(today_str)
                events.setdefault(sym, []).append({
                    "date": date_s,
                    "purpose": purpose[:60],
                    "today": is_today,
                })
            except Exception:
                continue
        return events

    async def _nifty_levels(self) -> dict:
        """CPR + classic pivots from yesterday's NIFTY daily bar, plus prev H/L."""
        try:
            token = get_instrument_token(self.kite, "NIFTY 50", "NSE")
            df = await self._daily_bars(token)
            if df.empty:
                return {}
            df["date"] = pd.to_datetime(df["date"]).dt.date
            completed = df[df["date"] < datetime.date.today()]
            if completed.empty:
                return {}
            prev = completed.tail(1)
            pivots = calculate_cpr(prev)
            last = prev.iloc[-1]
            return {
                "last_close": round(float(last["close"]), 2),
                "prev_high": round(float(last["high"]), 2),
                "prev_low": round(float(last["low"]), 2),
                "pivot": round(pivots["pivot"], 2),
                "cpr_top": round(pivots["tc"], 2),
                "cpr_bottom": round(pivots["bc"], 2),
                "r1": round(pivots["r1"], 2),
                "r2": round(pivots["r2"], 2),
                "s1": round(pivots["s1"], 2),
                "s2": round(pivots["s2"], 2),
            }
        except Exception as e:
            logging.debug(f"[PreMarket] NIFTY level computation failed: {e}")
            return {}

    async def _vix_close(self) -> Optional[float]:
        try:
            token = get_instrument_token(self.kite, "INDIA VIX", "NSE")
            df = await self._daily_bars(token, days=7)
            if df.empty:
                return None
            return round(float(df.iloc[-1]["close"]), 2)
        except Exception as e:
            logging.debug(f"[PreMarket] VIX fetch failed: {e}")
            return None

    # ------------------------------------------------------------------ #

    @staticmethod
    def _big_move_risk(vix: Optional[float], events: dict, moves: list) -> dict:
        """
        Heuristic index-level move risk for today:
          - events TODAY on heavyweights add the most weight
          - elevated VIX adds weight
          - an outsized move yesterday in a heavyweight adds follow-through risk
        """
        score = 0
        reasons = []
        events_today = sum(
            1 for evs in events.values() for e in evs if e.get("today")
        )
        if events_today >= 2:
            score += 2; reasons.append(f"{events_today} heavyweight events today")
        elif events_today == 1:
            score += 1; reasons.append("1 heavyweight event today")
        if vix is not None:
            if vix >= 18:
                score += 2; reasons.append(f"VIX elevated ({vix:.1f})")
            elif vix >= 15:
                score += 1; reasons.append(f"VIX mid-range ({vix:.1f})")
        outsized = [m["symbol"] for m in moves if m.get("outsized")]
        if outsized:
            score += 1
            reasons.append(f"outsized move yesterday: {', '.join(outsized)}")
        level = "HIGH" if score >= 3 else ("MEDIUM" if score >= 1 else "LOW")
        return {"level": level, "score": score, "reasons": reasons}

    # ------------------------------------------------------------------ #

    async def generate(self, strategy_hint: Optional[dict] = None) -> dict:
        """
        Builds the full brief, prints it, persists it for the dashboard, and
        returns the payload. `strategy_hint` carries the bot's own regime /
        strategy choice so the brief ends with what the bot will actually do.
        """
        prev_day = _prev_trading_day(datetime.date.today())
        moves, events, levels, vix = await asyncio.gather(
            self._component_moves(),
            self._fetch_nse_events(),
            self._nifty_levels(),
            self._vix_close(),
        )
        risk = self._big_move_risk(vix, events, moves)
        payload = {
            "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "reference_session": prev_day.isoformat(),
            "components": moves,
            "events": events,
            "vix": vix,
            "big_move_risk": risk,
            "nifty_levels": levels,
            "strategy": strategy_hint or {},
        }
        try:
            atomic_write_json(PREMARKET_FILE, payload)
        except Exception as e:
            logging.warning(f"[PreMarket] could not persist briefing: {e}")

        self._print(payload)
        self.last_generated_at = datetime.datetime.now()
        return payload

    # ------------------------------------------------------------------ #

    @staticmethod
    def _print(p: dict) -> None:
        W = 68
        line = "═" * W

        def row(text=""):
            print(f"║ {text:<{W - 2}}║")

        print("\n╔" + line + "╗")
        row(f"PRE-MARKET BRIEF — {datetime.date.today().strftime('%a %d %b %Y')}"
            f"  (vs session {p['reference_session']})")
        print("╠" + line + "╣")

        row("TOP-5 NIFTY COMPONENTS (yesterday)")
        events = p.get("events") or {}
        for m in p.get("components", []):
            sym = m["symbol"]
            if m.get("pct_change") is None:
                row(f"  {sym:<10} [data unavailable]")
                continue
            arrow = "▲" if m["pct_change"] >= 0 else "▼"
            tag = " ⚡OUTSIZED" if m.get("outsized") else ""
            ev = events.get(sym) or []
            ev_today = next((e for e in ev if e.get("today")), None)
            if ev_today:
                ev_txt = f"[{(ev_today.get('purpose') or 'event')[:28]} TODAY ⚠]"
            elif ev:
                ev_txt = f"[{(ev[0].get('purpose') or 'event')[:24]} {ev[0].get('date', '')[:11]}]"
            else:
                ev_txt = "[no events]"
            row(f"  {sym:<10} {arrow} {m['pct_change']:+5.2f}%{tag}  {ev_txt}")
        if not events:
            row("  (NSE event calendar unavailable — events not shown)")

        print("╠" + line + "╣")
        risk = p.get("big_move_risk") or {}
        vix = p.get("vix")
        vix_txt = f"VIX {vix:.1f}" if vix is not None else "VIX n/a"
        row(f"BIG-MOVE RISK: {risk.get('level', '?'):<6} ({vix_txt})")
        for r in risk.get("reasons", []):
            row(f"  • {r}")

        lv = p.get("nifty_levels") or {}
        if lv:
            print("╠" + line + "╣")
            row(f"NIFTY LEVELS (last close {lv['last_close']:.0f})")
            row(f"  R2 {lv['r2']:>9.0f}    R1      {lv['r1']:>9.0f}")
            row(f"  CPR-TOP {lv['cpr_top']:>9.0f}    CPR-BOT {lv['cpr_bottom']:>9.0f}"
                f"    PIVOT {lv['pivot']:>9.0f}")
            row(f"  S1 {lv['s1']:>9.0f}    S2      {lv['s2']:>9.0f}")
            row(f"  Prev Day H {lv['prev_high']:>9.0f}    Prev Day L {lv['prev_low']:>9.0f}")

        strat = p.get("strategy") or {}
        if strat:
            print("╠" + line + "╣")
            if strat.get("regime"):
                row(f"REGIME: {strat['regime']}"
                    + (f"  |  CONVICTION: {strat['conviction']}" if strat.get("conviction") else ""))
            if strat.get("strategy"):
                row(f"STRATEGY: {strat['strategy']}")
        print("╚" + line + "╝\n")
