"""
News-driven sentiment for the Nifty-50 trading bot.

Two layers of relevance:

  1. AT FETCH TIME — query NewsAPI with focused boolean OR over Nifty-50
     constituents + macro/India-specific terms, biased toward Indian financial
     news domains. Casts a reasonably wide net.

  2. AT FILTER TIME — every fetched article is checked against a Nifty/India
     keyword set; articles that mention none of them are dropped. This catches
     the "Bharti Airtel" → "Brittney Griner won game" type false positives that
     NewsAPI's relevance ranking lets through.

The cache (1-hour TTL) stores ONLY the post-filtered articles, so downstream
consumers (`get_market_sentiment`, `get_top_headlines`) work off relevant data
without having to re-filter each call.
"""
import datetime
import json
import logging
import os
import time

from newsapi import NewsApiClient
from textblob import TextBlob


# ---------------------------------------------------------------------------
# Static constants — Nifty 50 universe + relevance keywords + domain bias
# ---------------------------------------------------------------------------

# Heavyweight Nifty 50 names — the ~30 stocks that move the index most
# (top weights from NSE Indices factsheets, kept ASCII-only for query safety).
NIFTY_50_KEY_NAMES = [
    "Reliance Industries", "HDFC Bank", "ICICI Bank", "Infosys", "TCS",
    "Larsen & Toubro", "Bharti Airtel", "ITC", "Kotak Mahindra Bank",
    "Hindustan Unilever", "Axis Bank", "State Bank of India",
    "Bajaj Finance", "Asian Paints", "Maruti Suzuki", "Sun Pharma",
    "Mahindra & Mahindra", "Tata Motors", "Nestle India", "Wipro",
    "UltraTech Cement", "Power Grid", "NTPC", "Tata Steel", "JSW Steel",
    "Adani Enterprises", "Adani Ports", "Coal India", "ONGC",
    "HCL Technologies", "Tech Mahindra", "Cipla", "Bajaj Finserv",
    "Eicher Motors", "Britannia", "Hero MotoCorp", "Bajaj Auto",
    "Grasim Industries", "Tata Consumer", "IndusInd Bank", "SBI Life",
    "HDFC Life",
]

# Macro / index / regulator keywords used in the boolean OR query.
MARKET_TERMS = [
    "Nifty 50", "Nifty50", "Sensex", "BSE India", "NSE India",
    "RBI", "Reserve Bank of India", "Indian stock market",
    "Indian economy", "FII flows", "DII flows", "rupee dollar",
    "FED rate", "repo rate", "Indian budget", "SEBI",
]

# ---------------------------------------------------------------------------
# Hard exclusion: non-financial topic keywords.
# If ANY of these appear in title+description the article is dropped
# immediately — before any anchor check. Prevents sports/entertainment
# headlines from sneaking through via ambiguous weak anchors like "hero",
# "sun", "trade deadline", "market for pitchers", etc.
# ---------------------------------------------------------------------------
NON_FINANCIAL_EXCLUSION_KEYWORDS = sorted([
    # Sports — American
    "baseball", "softball", "home run", "home runs", "innings", "pitcher",
    "batter", "mlb ", "nfl ", "nba ", "nhl ", "mls ",
    "touchdown", "quarterback", "nfl draft", "super bowl",
    "world series", "playoffs", "batting average",
    # Sports — general
    "cricket match", "ipl match", "test match", "odi match",
    "football match", "premier league", "champions league",
    "la liga", "bundesliga", "serie a",
    "tennis tournament", "wimbledon", "us open tennis",
    "golf tournament", "pga tour", "masters golf",
    "olympics", "commonwealth games", "asian games",
    "world cup cricket", "t20 world cup",
    "college softball", "college baseball", "ncaa",
    # Entertainment / celebrity
    "box office", "box-office", "film review", "movie review",
    "box office collection", "bollywood gossip", "celebrity",
    "award show", "grammy", "oscar", "golden globe", "emmy",
    "music video", "album release", "concert tour",
])

# ---------------------------------------------------------------------------
# Source domains that should NEVER contribute to market sentiment.
# These are entertainment, sports, or general lifestyle outlets that
# occasionally produce finance-tagged content but are structurally noisy.
# ---------------------------------------------------------------------------
EXCLUDED_SOURCE_DOMAINS = {
    "yahoo.com/entertainment", "entertainment.yahoo.com",
    "roundtable.io", "sports.yahoo.com",
    "espn.com", "bleacherreport.com", "cbssports.com",
    "nbcsports.com", "foxsports.com", "theathletic.com",
    "sportskeeda.com", "scroll.in/field",
    "people.com", "tmz.com", "eonline.com",
}

# ---------------------------------------------------------------------------
# Post-fetch relevance filter.
# ---------------------------------------------------------------------------
# Safe multi-word fragments for constituents whose first word is a common
# English word (sun, hero, tech, state, power, coal, asian).
# These replace the naive split()[0] fragments for those companies so we
# don't match "sun rises", "hero of the match", "tech startup", etc.
_SAFE_MULTI_WORD = {
    "sun pharma", "hero motocorp", "tech mahindra",
    "state bank", "power grid", "coal india", "asian paints",
    "tata consumer", "tata steel", "tata motors",
    "jsw steel", "sbi life", "hdfc life", "hdfc bank",
    "hcl tech", "bajaj finserv", "bajaj finance", "bajaj auto",
    "eicher motors", "ultratech cement", "grasim industries",
}

# Single-word fragments are only kept for companies where the first word
# is genuinely distinctive (won't match sports/entertainment noise).
_SAFE_SINGLE_WORD_COMPANIES = {
    "reliance", "infosys", "wipro", "cipla", "nestle",
    "britannia", "maruti", "ongc", "ntpc", "kotak",
    "indusind", "icici", "hdfc", "axis", "bharti",
    "adani", "ambani", "mahindra", "bajaj", "tata",
}
_CONSTITUENT_FRAGMENTS = sorted(
    _SAFE_MULTI_WORD | _SAFE_SINGLE_WORD_COMPANIES
)

# "Strong" anchors — substrings that on their own definitively place the article
# in Nifty/India financial context. Articles containing any of these pass the
# filter unconditionally (after exclusion check).
STRONG_ANCHORS = sorted(set([
    "nifty", "sensex", "bse", "nse", "rbi", "sebi",
    "fii flows", "dii flows", "dalal street",
    "indian markets", "indian economy", "indian stock",
    "indian shares", "indian equities", "rupee",
    "ambani", "adani", "repo rate", "monetary policy",
    "stock market india", "share market india",
]))

# "Weak" anchors — company names or India references that need FINANCIAL_CONTEXT
# confirmation (2+ terms required now — raises the bar vs the old 1-term check).
WEAK_ANCHORS = sorted(set([
    "indian", "india", "mumbai", "dalal",
] + _CONSTITUENT_FRAGMENTS))

# Financial-context keywords. Ambiguous words removed:
#   "trade"   → appears as "trade deadline" in sports
#   "market"  → appears as "player market" in sports
#   "loss"    → appears as "team loss" in sports
#   "results" → appears as "game results" in sports
#   "rate"    → too generic
# Kept only unambiguously financial terms.
FINANCIAL_CONTEXT = sorted(set([
    "stock", "stocks", "shares", "share price", "equity", "equities",
    "earnings", "profit", "revenue", "quarterly results",
    "q1", "q2", "q3", "q4",
    "crore", "lakh", "rupee", "rupees", " rs ", "₹",
    "ipo", "broker", "investor", "investment",
    "bourse", "fund", "yield", "interest rate", "policy rate",
    "nse", "bse", "sebi", "exchange", "listed", "valuation",
    "dividend", "buyback", "stake", "shareholding", "portfolio",
]))

# Convenience union for legacy callers.
RELEVANCE_KEYWORDS = sorted(set(STRONG_ANCHORS + WEAK_ANCHORS))

# Indian-financial-news domain bias (NewsAPI 'domains' arg, comma-separated).
# Used as a soft filter — if the domain query returns too few articles we
# fall back to an unrestricted fetch with the same query and post-filter.
INDIAN_FINANCIAL_DOMAINS = ",".join([
    "moneycontrol.com",
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "financialexpress.com",
    "thehindu.com",
    "indianexpress.com",
    "businesstoday.in",
    "cnbctv18.com",
    "ndtv.com",
    "reuters.com",
    "bloomberg.com",
    "bloombergquint.com",
])

# How many heavyweight constituents to put in the OR query (NewsAPI has a
# 500-char query limit; 15 names + 16 macro terms keeps us well under).
_CONSTITUENT_QUERY_DEPTH = 15

# Minimum filtered-article count below which we re-fetch without the domain
# restriction. NewsAPI domains can be patchy on indexing; this is the safety net.
_MIN_FILTERED_FOR_DOMAIN_FETCH = 12


class SentimentAgent:
    """Fetches Nifty-relevant news and computes a recency-weighted sentiment."""

    def __init__(self, config, youtube_agent=None):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config['news_api']['api_key'])
        self.cache_dir = "news_cache"
        os.makedirs(self.cache_dir, exist_ok=True)
        # Optional YouTubeSentimentAgent. When set and ready, get_market_sentiment
        # blends its verdicts with the news-derived score using the
        # `youtube_sentiment.overall_weight_vs_news` config multiplier.
        self.youtube_agent = youtube_agent
        # Change-only log dedup. get_market_sentiment() and the cache read run
        # every loop tick; without this they reprint identical lines constantly.
        self._last_logs: dict = {}

    def _log_changed(self, key: str, message: str):
        """Emit `message` only when it differs from the last one logged for `key`."""
        if self._last_logs.get(key) != message:
            logging.info(message)
            self._last_logs[key] = message

    # ---------- query builders ----------

    def _build_query(self, max_chars: int = 480) -> str:
        """
        Boolean-OR of macro terms + heavyweight constituents, capped to NewsAPI's
        free-tier 500-char query budget. Macro terms go in first (high priority);
        constituents are appended until the budget is consumed.
        """
        terms = [f'"{t}"' for t in MARKET_TERMS]
        candidates = [f'"{c}"' for c in NIFTY_50_KEY_NAMES]
        for cand in candidates:
            tentative = " OR ".join(terms + [cand])
            if len(tentative) > max_chars:
                break
            terms.append(cand)
        return " OR ".join(terms)

    # ---------- relevance filter ----------

    @staticmethod
    def _is_relevant(article: dict) -> bool:
        """
        Four-stage relevance gate:
          0. Hard exclusion — drop if source domain is in EXCLUDED_SOURCE_DOMAINS
             OR if any NON_FINANCIAL_EXCLUSION_KEYWORD appears in the text.
             Eliminates sports/entertainment articles regardless of anchors.
          1. Strong anchor — if a STRONG_ANCHOR matches, keep unconditionally.
          2. Weak anchor + 2 financial-context terms — raises bar vs old 1-term
             check so ambiguous words like "trade", "market", "loss" alone can't
             let a sports article through.
          3. Else drop.
        """
        # Stage 0a: block known non-financial source domains.
        source_url = (article.get('url') or '').lower()
        source_name = (article.get('source', {}).get('name') or '').lower()
        if any(d in source_url or d in source_name for d in EXCLUDED_SOURCE_DOMAINS):
            return False

        text = (
            (article.get('title') or '') + ' '
            + (article.get('description') or '') + ' '
            + (article.get('content') or '')
        ).lower()

        # Stage 0b: hard-exclude non-financial topics (sports, entertainment).
        if any(kw in text for kw in NON_FINANCIAL_EXCLUSION_KEYWORDS):
            return False

        # Stage 1: strong India/market anchor — keep unconditionally.
        if any(kw in text for kw in STRONG_ANCHORS):
            return True

        # Stage 2: weak anchor must be accompanied by 2+ unambiguous financial
        # terms (raised from 1 to avoid "trade deadline" / "market for pitchers"
        # false positives from ambiguous single-word constituent fragments).
        if any(kw in text for kw in WEAK_ANCHORS):
            fin_matches = sum(1 for ctx in FINANCIAL_CONTEXT if ctx in text)
            return fin_matches >= 2

        return False

    def _filter_relevant(self, articles: list) -> list:
        if not articles:
            return []
        return [a for a in articles if self._is_relevant(a)]

    # ---------- raw NewsAPI calls ----------

    def _fetch_from_api(self, query: str, from_date, to_date,
                       domains: str | None = None) -> list:
        kwargs = dict(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=100,
            from_param=from_date.isoformat(),
            to=to_date.isoformat(),
        )
        if domains:
            kwargs['domains'] = domains
        try:
            resp = self.newsapi.get_everything(**kwargs)
        except Exception as e:
            logging.error(f"SentimentAgent: NewsAPI call failed (domains={bool(domains)}): {e}")
            return []
        return resp.get('articles', []) or []

    # ---------- cache + main fetch ----------

    def _get_news_articles(self):
        """
        Returns a dict with key 'articles' containing post-filtered relevant
        articles. Cached to disk for 1 hour to avoid hammering NewsAPI.
        """
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=2)
        cache_path = os.path.join(self.cache_dir, f"news_{today.isoformat()}.json")
        CACHE_EXPIRATION_SECONDS = 3600

        if (os.path.exists(cache_path)
                and (time.time() - os.path.getmtime(cache_path)) < CACHE_EXPIRATION_SECONDS):
            try:
                with open(cache_path, 'r') as f:
                    cached = json.load(f)
                self._log_changed(
                    "cache_load",
                    f"SentimentAgent: loaded {len(cached.get('articles', []))} cached "
                    f"relevant articles (< 60min old)."
                )
                return cached
            except Exception as e:
                logging.warning(f"SentimentAgent: cache read failed ({e}); refetching.")

        query = self._build_query()
        logging.info("SentimentAgent: fetching fresh news (Indian financial domain bias)...")

        # 1st pass: domain-restricted
        articles = self._fetch_from_api(query, from_date, today,
                                         domains=INDIAN_FINANCIAL_DOMAINS)
        relevant = self._filter_relevant(articles)
        domain_count = len(relevant)

        # 2nd pass (fallback) if domain-restricted was thin
        if len(relevant) < _MIN_FILTERED_FOR_DOMAIN_FETCH:
            logging.info(
                f"SentimentAgent: domain-restricted yielded {len(relevant)} relevant "
                f"articles (< {_MIN_FILTERED_FOR_DOMAIN_FETCH}); fetching unrestricted."
            )
            extra = self._fetch_from_api(query, from_date, today, domains=None)
            extra_relevant = self._filter_relevant(extra)
            # Dedupe by URL (NewsAPI articles always have a 'url' field).
            seen = {a.get('url') for a in relevant if a.get('url')}
            for a in extra_relevant:
                u = a.get('url')
                if u and u not in seen:
                    relevant.append(a)
                    seen.add(u)

        logging.info(
            f"SentimentAgent: kept {len(relevant)} relevant articles "
            f"(domain-restricted: {domain_count}, after fallback: {len(relevant) - domain_count})."
        )

        payload = {
            'articles': relevant,
            'totalResults': len(relevant),
            'fetchedAt': datetime.datetime.now().isoformat(),
        }
        try:
            with open(cache_path, 'w') as f:
                json.dump(payload, f)
        except Exception as e:
            logging.warning(f"SentimentAgent: cache write failed: {e}")
        return payload

    # ---------- public API ----------

    def get_top_headlines(self, n: int = 10) -> list:
        """
        Returns up to `n` most recent relevant headlines with their individual
        polarity scores — for showing the operator what is actually driving the
        automated sentiment read before they confirm or override it.

        Each entry: {title, source, published_at, polarity}.
        Polarity in [-1.0, +1.0]: positive = bullish-leaning text.
        """
        articles = self._get_news_articles()
        if not articles or not articles.get('articles'):
            return []
        out = []
        for a in articles['articles']:
            title = a.get('title') or ''
            if not title or title == "[Removed]":
                continue
            description = a.get('description') or ''
            content = f"{title}. {description}".strip()
            try:
                polarity = float(TextBlob(content).sentiment.polarity)
            except Exception:
                polarity = 0.0
            out.append({
                "title": title,
                "source": (a.get('source') or {}).get('name', ''),
                "published_at": a.get('publishedAt', ''),
                "polarity": polarity,
            })
            if len(out) >= n:
                break
        return out

    def _news_weighted_average(self):
        """
        Internal: returns (avg, sample_count) for the news polarity score,
        using the same linear-decay weighting (newest articles count most).
        """
        top = self._get_news_articles()
        if not top or not top.get('articles'):
            return 0.0, 0
        scores = []
        for article in top['articles']:
            title = article.get('title') or ''
            if not title or title == "[Removed]":
                continue
            content = f"{title}. {article.get('description', '')}"
            try:
                scores.append(float(TextBlob(content).sentiment.polarity))
            except Exception:
                continue
        if not scores:
            return 0.0, 0
        n = len(scores)
        weighted_sum = sum(score * (n - i) for i, score in enumerate(scores))
        total_weight = sum(range(1, n + 1))
        return (weighted_sum / total_weight if total_weight else 0.0), n

    def _youtube_weighted_average(self):
        """
        Internal: returns (avg, sample_count) for the YouTube verdict set.
        Score per verdict = direction_score × confidence; weighted by the
        per-channel `weight`. Returns (0.0, 0) if no YouTube agent or no
        cached verdicts yet.
        """
        if not self.youtube_agent or not self.youtube_agent.is_ready():
            return 0.0, 0
        from youtube_sentiment import verdict_to_score  # local import to avoid cycles at import time
        verdicts = self.youtube_agent.get_verdicts()
        if not verdicts:
            return 0.0, 0
        weighted_sum = 0.0
        total_weight = 0.0
        for v in verdicts:
            score = verdict_to_score(v)
            weight = float(v.get('channel_weight', 10) or 10)
            if weight <= 0:
                continue
            weighted_sum += score * weight
            total_weight += weight
        return (weighted_sum / total_weight if total_weight else 0.0), len(verdicts)

    def get_market_sentiment(self):
        """
        Combined weighted sentiment from news + YouTube analyst verdicts.
        News and YouTube each produce their own weighted-average; the two
        averages are then blended using `youtube_sentiment.overall_weight_vs_news`
        (default 2.0 — YouTube collectively gets 2x the weight of news).

        Returns one of: Very Bullish / Bullish / Neutral / Bearish / Very Bearish.
        """
        news_avg, news_n = self._news_weighted_average()
        yt_avg, yt_n = self._youtube_weighted_average()

        if news_n == 0 and yt_n == 0:
            logging.warning("SentimentAgent: no news or YouTube data; defaulting to Neutral.")
            return "Neutral"

        yt_cfg = self.config.get('youtube_sentiment', {}) or {}
        yt_overall_weight = float(yt_cfg.get('overall_weight_vs_news', 2.0))

        if yt_n == 0:
            final_avg = news_avg
            self._log_changed(
                "final_avg",
                f"SentimentAgent: news-only avg = {final_avg:+.3f} (over {news_n} headlines)."
            )
        elif news_n == 0:
            final_avg = yt_avg
            self._log_changed(
                "final_avg",
                f"SentimentAgent: YouTube-only avg = {final_avg:+.3f} (over {yt_n} verdicts)."
            )
        else:
            news_w, yt_w = 1.0, yt_overall_weight
            final_avg = (news_w * news_avg + yt_w * yt_avg) / (news_w + yt_w)
            self._log_changed(
                "final_avg",
                f"SentimentAgent: combined sentiment - "
                f"news avg {news_avg:+.3f} (n={news_n}) | "
                f"yt avg {yt_avg:+.3f} (n={yt_n}, weight={yt_overall_weight}x) | "
                f"final {final_avg:+.3f}"
            )

        if final_avg > 0.4:
            return "Very Bullish"
        if final_avg > 0.05:
            return "Bullish"
        if final_avg < -0.4:
            return "Very Bearish"
        if final_avg < -0.05:
            return "Bearish"
        return "Neutral"
