"""
PCR Feed — computes NIFTY/BANKNIFTY Put-Call Ratio from Kite's NFO option chain.

PCR = Σ put OI / Σ call OI  over ATM ± strikes_each_side for the nearest expiry.

Standard contrarian interpretation (NSE intraday options context):
  PCR ≥ bullish_threshold (default 1.2)
      → PCR_BULLISH  (heavy put writing / put buying = smart money long-biased;
                      often precedes upside; confirms BUY signals)
  PCR ≤ bearish_threshold (default 0.8)
      → PCR_BEARISH  (retail call-buying / put-selling = exuberance top risk;
                      often precedes downside; confirms SELL signals)
  Otherwise
      → PCR_NEUTRAL  (no directional edge; gate bypassed)

The feed fetches OI once per call and caches the result for `cache_seconds`
(default 300 s ≈ one 5-min bar) to avoid hammering the broker API.
Failures are non-fatal: the gate is simply bypassed when PCR data is unavailable.
"""

import asyncio
import datetime
import logging

import pandas as pd


_UNDERLYING_ROOT_MAP = {
    "NIFTY 50": "NIFTY",
    "NIFTY":    "NIFTY",
    "BANKNIFTY": "BANKNIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "FINNIFTY":  "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "SENSEX":    "SENSEX",
}


class PCRFeed:
    """
    Computes the Put-Call Ratio for the bot's configured underlying and
    nearest weekly expiry.  Designed to be initialised once (after Kite auth)
    and queried on every 5-min bar boundary.

    Public API
    ----------
    await get_pcr(spot_price) → dict with keys:
        pcr        float | None
        tag        'PCR_BULLISH' | 'PCR_BEARISH' | 'PCR_NEUTRAL' |
                   'PCR_DISABLED' | 'PCR_ERROR'
        put_oi     int
        call_oi    int
        strikes    int   (number of unique strikes queried)
        expiry     str   (ISO date of the expiry used)
        atm        float (ATM strike used)
        error      str   (only present when tag == 'PCR_ERROR')
    """

    def __init__(self, kite, config: dict):
        self.kite   = kite
        self.config = config or {}

        cfg = (config or {}).get("pcr_feed", {}) or {}
        self.enabled           = bool(cfg.get("enable", True))
        self.strikes_each_side = int(cfg.get("strikes_each_side", 5))
        self.bullish_threshold = float(cfg.get("bullish_threshold", 1.2))
        self.bearish_threshold = float(cfg.get("bearish_threshold", 0.8))
        self.cache_seconds     = int(cfg.get("cache_seconds", 300))

        underlying = (
            (config.get("trading_flags") or {}).get("underlying_instrument", "NIFTY 50")
        )
        self._root = _UNDERLYING_ROOT_MAP.get(underlying.upper(), "NIFTY")

        # Runtime state
        self._cached_result: dict | None = None
        self._cached_at:     datetime.datetime | None = None
        self._instruments:   pd.DataFrame | None = None  # loaded once per session

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    async def get_pcr(self, spot_price: float) -> dict:
        """
        Return PCR for the nearest expiry, using the cache when it is still valid.
        Never raises — failures return a dict with tag='PCR_ERROR'.
        """
        if not self.enabled:
            return {"pcr": None, "tag": "PCR_DISABLED"}

        if self._is_cache_valid():
            return self._cached_result  # type: ignore[return-value]

        result = await self._compute_pcr(spot_price)
        if result.get("pcr") is not None:
            self._cached_result = result
            self._cached_at     = datetime.datetime.now()
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_cache_valid(self) -> bool:
        if self._cached_result is None or self._cached_at is None:
            return False
        age = (datetime.datetime.now() - self._cached_at).total_seconds()
        return age < self.cache_seconds

    async def _load_instruments(self) -> pd.DataFrame:
        """Load NFO instrument list once per session."""
        if self._instruments is not None:
            return self._instruments
        try:
            raw = await asyncio.to_thread(self.kite.instruments, "NFO")
            self._instruments = pd.DataFrame(raw)
            logging.info(
                f"[PCRFeed] Loaded {len(self._instruments):,} NFO instruments."
            )
        except Exception as exc:
            logging.warning(f"[PCRFeed] Could not load NFO instruments: {exc}")
            self._instruments = pd.DataFrame()
        return self._instruments

    async def _compute_pcr(self, spot_price: float) -> dict:
        try:
            insts = await self._load_instruments()
            if insts is None or insts.empty:
                return {"pcr": None, "tag": "PCR_ERROR", "error": "no instruments"}

            root = self._root

            # ---- filter to options for this underlying ----
            opts = insts[
                (insts["name"] == root)
                & (insts["instrument_type"].isin(["CE", "PE"]))
                & (insts["segment"] == "NFO-OPT")
            ].copy()

            if opts.empty:
                return {
                    "pcr": None, "tag": "PCR_ERROR",
                    "error": f"no option instruments for {root}",
                }

            # ---- nearest expiry ----
            opts["expiry"] = pd.to_datetime(opts["expiry"])
            today = pd.Timestamp(datetime.date.today())
            future_expiries = opts[opts["expiry"] >= today]["expiry"].unique()
            if len(future_expiries) == 0:
                return {"pcr": None, "tag": "PCR_ERROR", "error": "no future expiries"}
            nearest_expiry = sorted(future_expiries)[0]
            expiry_opts = opts[opts["expiry"] == nearest_expiry].copy()

            # ---- strike range around ATM ----
            strike_steps = (self.config or {}).get("strike_steps", {})
            step = float(strike_steps.get(root, 50))
            atm   = round(spot_price / step) * step
            lower = atm - self.strikes_each_side * step
            upper = atm + self.strikes_each_side * step

            range_opts = expiry_opts[
                (expiry_opts["strike"] >= lower)
                & (expiry_opts["strike"] <= upper)
            ]
            if range_opts.empty:
                return {
                    "pcr": None, "tag": "PCR_ERROR",
                    "error": f"no options in range [{lower:.0f}–{upper:.0f}]",
                }

            # ---- fetch OI in batches (Kite limit ≈ 500 per call) ----
            tokens = [str(int(t)) for t in range_opts["instrument_token"].tolist()]
            quotes: dict = {}
            for i in range(0, len(tokens), 400):
                batch = tokens[i : i + 400]
                try:
                    q = await asyncio.to_thread(self.kite.quote, batch)
                    quotes.update(q)
                except Exception as exc:
                    logging.warning(f"[PCRFeed] Quote batch [{i}:{i+400}] failed: {exc}")

            # ---- sum OI by option type ----
            total_put_oi  = 0
            total_call_oi = 0
            for _, row in range_opts.iterrows():
                token_str = str(int(row["instrument_token"]))
                oi = int((quotes.get(token_str) or {}).get("oi") or 0)
                if row["instrument_type"] == "PE":
                    total_put_oi  += oi
                else:
                    total_call_oi += oi

            if total_call_oi == 0:
                return {"pcr": None, "tag": "PCR_ERROR", "error": "zero call OI"}

            pcr = total_put_oi / total_call_oi

            if pcr >= self.bullish_threshold:
                tag = "PCR_BULLISH"
            elif pcr <= self.bearish_threshold:
                tag = "PCR_BEARISH"
            else:
                tag = "PCR_NEUTRAL"

            result = {
                "pcr":     pcr,
                "tag":     tag,
                "put_oi":  total_put_oi,
                "call_oi": total_call_oi,
                "strikes": len(range_opts["strike"].unique()),
                "expiry":  nearest_expiry.strftime("%Y-%m-%d"),
                "atm":     atm,
            }
            logging.info(
                f"[PCRFeed] PCR={pcr:.3f} ({tag})  "
                f"puts={total_put_oi:,}  calls={total_call_oi:,}  "
                f"expiry={result['expiry']}  ATM={atm:.0f}  "
                f"strikes={result['strikes']}"
            )
            return result

        except Exception as exc:
            logging.warning(f"[PCRFeed] Computation failed: {exc}")
            return {"pcr": None, "tag": "PCR_ERROR", "error": str(exc)}
