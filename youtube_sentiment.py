"""
YouTube-driven Nifty sentiment, layered onto the existing news-based pipeline.

Daily flow per configured channel:
  1. Resolve @handle -> channel_id (UC...) via YouTube Data API; result cached
     to disk so we don't burn quota on repeat lookups.
  2. List recent uploads via the uploads-playlist trick (UC... -> UU...),
     filter to videos published in the last `max_video_age_hours`.
  3. For the most recent qualifying video:
     a. Fetch transcript via `youtube-transcript-api` (English / English-IN / Hindi).
     b. Fetch sponsor segments from SponsorBlock's public API.
     c. Drop transcript entries whose timestamp falls inside a sponsor /
        selfpromo / interaction segment.
     d. Truncate to `transcript_max_chars`, send to Gemini with a structured-
        output prompt, parse the verdict JSON.
  4. Cache today's verdicts to `state/youtube_sentiment_<YYYY-MM-DD>.json` so
     reassessment cycles within the same day don't re-fetch / re-cost LLM calls.

Failure modes are designed to degrade gracefully:
  - Transcript missing -> skip the channel for the day.
  - SponsorBlock returns nothing -> instruct Gemini to ignore promo segments
    in the prompt instead (LLM fallback).
  - Gemini call fails / malformed JSON -> skip the channel.
  - Network failure anywhere -> log a warning, return empty list.

Never raises into the orchestrator's hot path.
"""
from __future__ import annotations

import asyncio
import datetime
import json
import logging
from typing import Optional

import aiohttp
import requests

from infra import atomic_write_json, read_json, state_path


# Map verdict labels to numeric polarity. Matches the bucket thresholds the
# SentimentAgent uses to convert the final score back to a label.
# Sentinel returned by _process_channel when YouTube confirms no recent
# video exists for a channel. Distinct from None (Gemini/network error)
# so fetch_today can stop retrying the channel for the rest of the day.
_NO_VIDEO = object()

DIRECTION_SCORE = {
    "Very Bullish": 0.7,
    "Bullish": 0.3,
    "Neutral": 0.0,
    "Bearish": -0.3,
    "Very Bearish": -0.7,
}


def verdict_to_score(verdict: dict) -> float:
    """
    Converts a YouTube verdict dict to a numeric polarity in [-0.7, +0.7],
    scaled by the analyst's stated confidence.
    """
    base = DIRECTION_SCORE.get(verdict.get("direction", "Neutral"), 0.0)
    conf = float(verdict.get("confidence", 0.0) or 0.0)
    return base * max(0.0, min(1.0, conf))


class YouTubeSentimentAgent:
    """Fetches and caches Nifty-directional verdicts from configured YouTubers."""

    CHANNEL_ID_CACHE_FILE = state_path("youtube_channel_ids.json")

    def __init__(self, config: dict, gemini_api_key: str):
        self.config = config
        cfg = config.get("youtube_sentiment", {}) or {}
        self.enabled = bool(cfg.get("enable", False))
        self.max_age_hours = int(cfg.get("max_video_age_hours", 18))
        self.transcript_max_chars = int(cfg.get("transcript_max_chars", 8000))
        self.strip_categories = set(
            cfg.get("strip_categories", ["sponsor", "selfpromo", "interaction"])
        )
        self.llm_strip_fallback = bool(cfg.get("llm_strip_fallback", True))
        self.channels = list(cfg.get("channels", []) or [])
        self.youtube_api_key = (config.get("youtube_api", {}) or {}).get("api_key", "")
        self.gemini_api_key = gemini_api_key
        self.gemini_model = cfg.get("gemini_model", "gemini-2.0-flash")
        self._verdicts: list = []
        self._ready = False
        # Set once the "loaded N cached verdicts" line has been printed, so the
        # per-tick fetch_today() cache-hit path doesn't reprint it every loop.
        self._cache_log_emitted = False
        # Channels that definitively had no recent video (stop fetching them).
        self._no_video_channels: set = set()
        # Channels that errored last time (Gemini 429, network, etc.) — retry.
        self._error_channels: set = set()

    # ---------- public read-only API used by SentimentAgent + orchestrator ----------

    def is_ready(self) -> bool:
        """
        True when YouTube sentiment has been resolved for today and no retries
        are pending. Returns False when there were errors on the last fetch so
        that setup() will call fetch_today() again to retry the failed channels.
        """
        return self._ready and len(self._error_channels) == 0

    def get_verdicts(self) -> list:
        return list(self._verdicts)

    # ---------- top-level: fetch the day's verdicts ----------

    async def fetch_today(self) -> list:
        """Fetches latest verdicts. Cache hit -> instant return. Cache miss -> network."""
        if not self.enabled:
            self._ready = True
            return []
        if not self.youtube_api_key:
            logging.warning("YouTubeSentiment: youtube_api.api_key missing; disabling.")
            self._ready = True
            return []
        if not self.gemini_api_key:
            logging.warning("YouTubeSentiment: google_api.api_key missing; disabling.")
            self._ready = True
            return []

        today = datetime.date.today().isoformat()
        cache_path = state_path(f"youtube_sentiment_{today}.json")
        cached = read_json(cache_path, default=None)
        if isinstance(cached, dict) and isinstance(cached.get("channels"), list):
            self._verdicts = cached["channels"]
            self._ready = True
            if not self._cache_log_emitted:
                logging.info(
                    f"YouTubeSentiment: loaded {len(self._verdicts)} cached verdicts for today."
                )
                self._cache_log_emitted = True
            return self._verdicts

        # On retry runs, only re-process channels that errored last time.
        # Channels that explicitly had no recent video are skipped permanently.
        is_retry = bool(self._error_channels)
        channels_to_process = [
            ch for ch in self.channels
            if ch.get('handle', '') not in self._no_video_channels
            and (not is_retry or ch.get('handle', '') in self._error_channels)
        ]

        if not channels_to_process:
            # All channels either had no video or were already resolved.
            self._error_channels.clear()
            self._ready = True
            logging.info(
                f"YouTubeSentiment: nothing left to fetch "
                f"(no-video channels: {sorted(self._no_video_channels)})."
            )
            return self._verdicts

        # Clear error set — will be repopulated only for channels that fail again.
        self._error_channels.clear()

        verdicts = list(self._verdicts)  # keep already-fetched verdicts from prior calls
        fetch_attempt = "retry" if is_retry else "first run"
        logging.info(
            f"YouTubeSentiment: {fetch_attempt} — processing "
            f"{len(channels_to_process)} channel(s)."
        )

        for idx, ch in enumerate(channels_to_process):
            handle = ch.get('handle', '?')
            # Small inter-channel delay so back-to-back Gemini calls don't
            # immediately hit the free-tier RPM cap (15 req/min on flash).
            if idx > 0:
                await asyncio.sleep(5)
            try:
                v = await self._process_channel(ch)
                if v is _NO_VIDEO:
                    # YouTube API confirmed: no recent upload from this channel.
                    # Stop retrying — it won't have a new video for the rest of the day.
                    self._no_video_channels.add(handle)
                    logging.info(
                        f"YouTubeSentiment: {handle} has no recent video — "
                        f"will not retry this channel today."
                    )
                elif v is not None:
                    # Successful verdict — replace any stale entry and keep it.
                    verdicts = [x for x in verdicts if x.get('channel_handle') != handle]
                    verdicts.append(v)
                else:
                    # None = Gemini/transcript error (retriable).
                    # Mark as errored so next setup() call retries this channel.
                    self._error_channels.add(handle)
                    logging.warning(
                        f"YouTubeSentiment: {handle} returned no verdict "
                        f"(Gemini/transcript issue) — will retry on next setup cycle."
                    )
            except Exception as e:
                # Unexpected exception — also retriable.
                self._error_channels.add(handle)
                logging.warning(
                    f"YouTubeSentiment: channel {handle} failed: "
                    f"{self._sanitize_error(e)} — will retry on next setup cycle."
                )

        self._verdicts = verdicts
        # Only mark ready (no more retries) when no channels are still errored.
        self._ready = len(self._error_channels) == 0
        try:
            atomic_write_json(
                cache_path,
                {
                    "fetched_at": datetime.datetime.now().isoformat(),
                    "max_age_hours": self.max_age_hours,
                    "channels": verdicts,
                },
            )
        except Exception as e:
            logging.warning(f"YouTubeSentiment: cache write failed: {e}")

        logging.info(
            f"YouTubeSentiment: fetched {len(verdicts)} verdicts across "
            f"{len(self.channels)} configured channels."
        )
        return verdicts

    # ---------- per-channel pipeline ----------

    async def _process_channel(self, channel_cfg: dict) -> Optional[dict]:
        handle = channel_cfg.get("handle", "").strip()
        if not handle:
            return None

        channel_id = await self._resolve_channel_id(handle)
        if not channel_id:
            logging.info(f"YouTubeSentiment: could not resolve channel for {handle}.")
            return None

        recent = await self._list_recent_videos(channel_id)
        if not recent:
            logging.info(
                f"YouTubeSentiment: no videos from {handle} in last "
                f"{self.max_age_hours}h — marking channel as no-video for today."
            )
            return _NO_VIDEO  # ← definitive: YouTube confirmed no recent upload

        video = recent[0]  # most-recent qualifying upload
        video_id = video.get("videoId")
        if not video_id:
            return None

        transcript = await asyncio.to_thread(self._fetch_transcript, video_id)
        if not transcript:
            logging.info(
                f"YouTubeSentiment: no transcript available for {video.get('title','?')}."
            )
            return None

        segments = await asyncio.to_thread(self._fetch_sponsor_segments, video_id)
        cleaned_text = self._strip_sponsors(transcript, segments)
        cleaned_text = cleaned_text[: self.transcript_max_chars] if cleaned_text else ""
        if not cleaned_text or len(cleaned_text) < 200:
            logging.info(
                f"YouTubeSentiment: cleaned transcript too short for "
                f"{video.get('title','?')} ({len(cleaned_text)} chars)."
            )
            return None

        no_sponsor_data = not segments
        verdict = await self._extract_verdict(
            cleaned_text, video, channel_cfg, no_sponsor_data
        )
        return verdict

    # ---------- channel-id resolution (cached on disk) ----------

    async def _resolve_channel_id(self, handle: str) -> Optional[str]:
        cache = read_json(self.CHANNEL_ID_CACHE_FILE, default={}) or {}
        if not isinstance(cache, dict):
            cache = {}
        cached_id = cache.get(handle)
        if cached_id:
            return cached_id

        clean_handle = handle.lstrip("@")
        url = (
            "https://www.googleapis.com/youtube/v3/channels"
            f"?part=id&forHandle=@{clean_handle}&key={self.youtube_api_key}"
        )
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
                async with s.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            logging.warning(f"YouTubeSentiment: handle resolve failed for {handle}: {e}")
            return None

        items = data.get("items") or []
        if not items:
            logging.warning(f"YouTubeSentiment: no channel found for handle {handle}.")
            return None
        channel_id = items[0].get("id")
        if not channel_id:
            return None

        cache[handle] = channel_id
        try:
            atomic_write_json(self.CHANNEL_ID_CACHE_FILE, cache)
        except Exception as e:
            logging.debug(f"YouTubeSentiment: channel-id cache write failed: {e}")
        return channel_id

    # ---------- list recent uploads ----------

    async def _list_recent_videos(self, channel_id: str) -> list:
        """Returns list of {videoId, title, publishedAt} for videos within `max_age_hours`."""
        if not channel_id.startswith("UC"):
            return []
        uploads_playlist_id = "UU" + channel_id[2:]
        url = (
            "https://www.googleapis.com/youtube/v3/playlistItems"
            f"?part=snippet&maxResults=5&playlistId={uploads_playlist_id}"
            f"&key={self.youtube_api_key}"
        )
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
                async with s.get(url) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
        except Exception as e:
            logging.warning(
                f"YouTubeSentiment: playlistItems fetch failed for {channel_id}: {e}"
            )
            return []

        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            hours=self.max_age_hours
        )
        recent = []
        for item in data.get("items", []) or []:
            snippet = item.get("snippet") or {}
            pub_str = snippet.get("publishedAt")
            if not pub_str:
                continue
            try:
                pub_dt = datetime.datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if pub_dt >= cutoff:
                recent.append({
                    "videoId": (snippet.get("resourceId") or {}).get("videoId"),
                    "title": snippet.get("title", ""),
                    "publishedAt": pub_str,
                })
        recent.sort(key=lambda v: v.get("publishedAt", ""), reverse=True)
        return recent

    # ---------- transcript fetch ----------

    def _fetch_transcript(self, video_id: str) -> Optional[list]:
        """
        Returns list of {text, start, duration} or None on failure.

        Compatible with BOTH the legacy `youtube-transcript-api` API
        (v0.6.x — classmethod `get_transcript`) and the newer v1.x API
        (instance method `.fetch()` returning a `FetchedTranscript`).
        First inspects the installed class to pick the right path.
        """
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            logging.error(
                "YouTubeSentiment: youtube-transcript-api not installed. "
                "Run: pip install youtube-transcript-api"
            )
            return None

        languages = ["en", "en-IN", "hi"]

        try:
            # Legacy API (v0.6.x) — classmethod still present.
            if hasattr(YouTubeTranscriptApi, "get_transcript"):
                return YouTubeTranscriptApi.get_transcript(
                    video_id, languages=languages
                )
            # New API (v1.x) — instance method returning a FetchedTranscript.
            ytt_api = YouTubeTranscriptApi()
            fetched = ytt_api.fetch(video_id, languages=languages)
            # Prefer the official conversion helper if it's exposed.
            if hasattr(fetched, "to_raw_data"):
                return fetched.to_raw_data()
            # Otherwise iterate snippet attributes ourselves.
            return [
                {"text": getattr(s, "text", ""),
                 "start": float(getattr(s, "start", 0.0) or 0.0),
                 "duration": float(getattr(s, "duration", 0.0) or 0.0)}
                for s in fetched
            ]
        except Exception as e:
            logging.info(
                f"YouTubeSentiment: transcript fetch failed for {video_id}: {e}"
            )
            return None

    # ---------- SponsorBlock segments ----------

    def _fetch_sponsor_segments(self, video_id: str) -> list:
        """
        Returns list of {startSec, endSec, category} for segments whose category is
        in `self.strip_categories`. Returns [] on 404 (no community data) or any error.
        """
        try:
            url = f"https://sponsor.ajay.app/api/skipSegments?videoID={video_id}"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json() or []
        except Exception as e:
            logging.debug(f"YouTubeSentiment: SponsorBlock fetch failed for {video_id}: {e}")
            return []

        out = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            category = entry.get("category")
            if category not in self.strip_categories:
                continue
            seg = entry.get("segment") or [0, 0]
            try:
                out.append({
                    "startSec": float(seg[0]),
                    "endSec": float(seg[1]),
                    "category": category,
                })
            except Exception:
                continue
        return out

    # ---------- sponsor-strip core (pure logic, testable) ----------

    @staticmethod
    def _strip_sponsors(transcript: list, segments: list) -> str:
        """
        Drops transcript entries whose `start` timestamp falls inside any sponsor
        segment, then concatenates remaining text. Pure function — no I/O.
        """
        if not transcript:
            return ""
        if not segments:
            return " ".join((entry.get("text") or "") for entry in transcript)

        # Pre-sort segments by start to allow early exit per entry.
        segs = sorted(segments, key=lambda s: s.get("startSec", 0.0))
        keep = []
        for entry in transcript:
            start = float(entry.get("start", 0.0) or 0.0)
            in_sponsor = False
            for seg in segs:
                if seg["startSec"] <= start <= seg["endSec"]:
                    in_sponsor = True
                    break
                if seg["startSec"] > start:
                    break  # remaining segments are later in the video
            if not in_sponsor:
                keep.append(entry.get("text") or "")
        return " ".join(keep)

    # ---------- helpers ----------

    def _sanitize_error(self, err: Exception) -> str:
        """
        Strip the Gemini API key from aiohttp exception messages before logging.
        aiohttp embeds the full request URL (including ?key=...) in its exception
        repr, which would leak the key to log files / stdout.
        """
        msg = str(err)
        if self.gemini_api_key and self.gemini_api_key in msg:
            msg = msg.replace(self.gemini_api_key, "****")
        return msg

    # ---------- Gemini extraction ----------

    async def _extract_verdict(self, cleaned_text: str, video: dict,
                                channel_cfg: dict, no_sponsor_data: bool) -> Optional[dict]:
        anti_promo_note = ""
        sponsor_strip_method = "SponsorBlock"
        if no_sponsor_data:
            sponsor_strip_method = "LLM-fallback" if self.llm_strip_fallback else "none"
            if self.llm_strip_fallback:
                anti_promo_note = (
                    "\nIMPORTANT: This transcript was NOT pre-cleaned. Discard any "
                    "segments that look like sponsorships, course promotions, "
                    "'subscribe/follow', referral codes, or affiliate links. "
                    "Focus only on the analyst's actual market commentary.\n"
                )

        channel_label = channel_cfg.get("display_name", channel_cfg.get("handle", "?"))

        prompt = f"""You are analysing an Indian-markets YouTube transcript to extract the channel's NIFTY 50 directional view for the NEXT trading day.
{anti_promo_note}
Channel: {channel_label}
Video title: {video.get('title', '')}

Return STRICT JSON ONLY (no markdown fences, no commentary outside JSON):
{{
  "direction": "Very Bullish" | "Bullish" | "Neutral" | "Bearish" | "Very Bearish",
  "confidence": <float 0.0 to 1.0>,
  "key_thesis": "<single-sentence summary of the analyst's directional thesis>",
  "specific_levels": {{
    "nifty_target": <integer or null>,
    "support": <integer or null>,
    "resistance": <integer or null>
  }},
  "excitement_level": <float 0.0 to 1.0>
}}

Rules:
- If the analyst does NOT discuss Nifty 50 / Indian index directional view,
  return direction = "Neutral" with confidence < 0.3.
- `excitement_level`: 1.0 = euphoric clickbait energy ("Nifty to the moon!"),
  0.0 = calm measured analysis. Independent of direction.
- All `specific_levels` integers in Nifty points (e.g. 24500). Use null if
  the analyst did not state a level.

Transcript:
{cleaned_text}
"""

        # Keep the key out of any log: build URL separately, never log it.
        gemini_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.gemini_model}:generateContent?key={self.gemini_api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        }

        # Retry with short backoff on 429 (rate-limit). A 429 from the free-tier
        # DAILY quota won't clear within a minute, so long waits just stall
        # startup for nothing — keep retries brief and skip fast. Configurable
        # via youtube_sentiment.rate_limit_backoff_secs (list).
        # Other HTTP errors (4xx != 429, 5xx) abort immediately.
        _yt_cfg = (self.config.get("youtube_sentiment") or {})
        _BACKOFF_SECS = list(_yt_cfg.get("rate_limit_backoff_secs", [5, 10]))
        _MAX_RETRIES  = len(_BACKOFF_SECS) + 1   # one final attempt after the last wait
        video_title = video.get("title", "?")
        verdict_data = None

        for attempt in range(_MAX_RETRIES):
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=45)
                ) as session:
                    async with session.post(gemini_url, json=payload) as resp:
                        if resp.status == 429:
                            if attempt < len(_BACKOFF_SECS):
                                wait = _BACKOFF_SECS[attempt]
                                logging.warning(
                                    f"YouTubeSentiment: Gemini rate-limited (429) for "
                                    f"'{video_title}' — waiting {wait}s before retry "
                                    f"(attempt {attempt + 1}/{_MAX_RETRIES})."
                                )
                                await asyncio.sleep(wait)
                            else:
                                logging.warning(
                                    f"YouTubeSentiment: Gemini still rate-limited after "
                                    f"{_MAX_RETRIES} attempts for '{video_title}'. "
                                    f"Skipping — news-only sentiment will be used today."
                                )
                            continue
                        resp.raise_for_status()
                        result = await resp.json()

                raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
                verdict_data = json.loads(raw_text)
                break  # success

            except aiohttp.ClientResponseError as e:
                # Sanitize: ClientResponseError includes the URL in its message.
                logging.warning(
                    f"YouTubeSentiment: Gemini HTTP error for '{video_title}': "
                    f"status={e.status} message={e.message!r}"
                )
                return None
            except Exception as e:
                logging.warning(
                    f"YouTubeSentiment: Gemini extraction failed for "
                    f"'{video_title}': {self._sanitize_error(e)}"
                )
                return None

        if verdict_data is None:
            logging.warning(
                f"YouTubeSentiment: Gemini still rate-limited after "
                f"{_MAX_RETRIES} retries for '{video_title}'. Skipping."
            )
            return None

        return {
            "channel_handle": channel_cfg.get("handle", ""),
            "channel_display_name": channel_label,
            "channel_weight": float(channel_cfg.get("weight", 10)),
            "video_id": video.get("videoId"),
            "video_title": video.get("title", ""),
            "published_at": video.get("publishedAt", ""),
            "direction": verdict_data.get("direction", "Neutral"),
            "confidence": float(verdict_data.get("confidence", 0.0) or 0.0),
            "key_thesis": verdict_data.get("key_thesis", "") or "",
            "specific_levels": verdict_data.get("specific_levels", {}) or {},
            "excitement_level": float(verdict_data.get("excitement_level", 0.0) or 0.0),
            "sponsor_strip_method": sponsor_strip_method,
        }
