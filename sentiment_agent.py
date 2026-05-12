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

# Post-fetch relevance filter. An article must contain at least ONE of these
# substrings (case-insensitive) in title+description to be retained. Mix of
# index/regulator names + first-name fragments of major constituents.
_CONSTITUENT_FRAGMENTS = [c.lower().split()[0] for c in NIFTY_50_KEY_NAMES]

# "Strong" anchors — substrings that on their own definitively place the article
# in Nifty/India financial context. Articles containing any of these pass the
# filter unconditionally.
STRONG_ANCHORS = sorted(set([
    "nifty", "sensex", "bse", "nse", "rbi", "sebi",
    "fii flows", "dii flows", "dalal street",
    "indian markets", "indian economy", "indian stock",
    "indian shares", "indian equities", "rupee",
    "ambani", "adani",
]))

# "Weak" anchors — single-word stock surnames that *may* appear in non-financial
# contexts (e.g. "Bharti" as a person's name in a film). When only weak anchors
# match, we additionally require a FINANCIAL_CONTEXT keyword in the same article.
WEAK_ANCHORS = sorted(set([
    "indian", "india", "mumbai", "tata", "bajaj", "mahindra",
] + _CONSTITUENT_FRAGMENTS))

# Financial-context keywords. Disambiguates weak anchors like "Tata" from a
# personal surname to "Tata Motors / Tata Group". Article must contain at least
# one of these alongside a weak anchor to qualify.
FINANCIAL_CONTEXT = sorted(set([
    "stock", "stocks", "shares", "share price", "equity", "equities",
    "market", "markets", "index", "trading", "trader", "trade",
    "earnings", "profit", "loss", "revenue", "results", "quarterly",
    "q1", "q2", "q3", "q4",
    "crore", "lakh", "rupee", "rupees", " rs ", "₹",
    "bse", "nse", "ipo", "broker", "investor", "investment",
    "bourse", "fund", "yield", "rate", "policy",
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
        Strict two-tier relevance:
          1. If a STRONG_ANCHOR matches -> keep.
          2. Else if a WEAK_ANCHOR matches AND a FINANCIAL_CONTEXT term also
             matches -> keep. (Disambiguates "Bharti's latest film" from
             "Bharti Airtel beats Q4 estimates".)
          3. Else drop.
        """
        text = (
            (article.get('title') or '') + ' '
            + (article.get('description') or '') + ' '
            + (article.get('content') or '')
        ).lower()
        if any(kw in text for kw in STRONG_ANCHORS):
            return True
        if any(kw in text for kw in WEAK_ANCHORS):
            return any(ctx in text for ctx in FINANCIAL_CONTEXT)
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
                logging.info(
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
            logging.info(
                f"SentimentAgent: news-only avg = {final_avg:+.3f} (over {news_n} headlines)."
            )
        elif news_n == 0:
            final_avg = yt_avg
            logging.info(
                f"SentimentAgent: YouTube-only avg = {final_avg:+.3f} (over {yt_n} verdicts)."
            )
        else:
            news_w, yt_w = 1.0, yt_overall_weight
            final_avg = (news_w * news_avg + yt_w * yt_avg) / (news_w + yt_w)
            logging.info(
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
