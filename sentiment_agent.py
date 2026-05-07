"""
SentimentAgent — NIFTY-relevant news sentiment.

Earlier version pulled "India" / "Attack" / "FED" as standalone tokens, which
matched plenty of unrelated global news. This version is structured around the
actual NIFTY 50 constituents + Indian-market macro terms, restricts NewsAPI to
a whitelist of Indian financial domains, matches keywords in the *title* (not
the article body), and applies a post-fetch relevance filter as a safety net.

Cache: per-day file with a 1-hour TTL — keeps NewsAPI quota under control
(free tier = 100 reqs/day; we use 3 reqs per cache miss × ~7 misses = ~21/day).
"""
import logging
import datetime
import json
import os
import time

from newsapi import NewsApiClient
from textblob import TextBlob


# ---------------------------------------------------------------------------
# NIFTY 50 relevance corpus
# ---------------------------------------------------------------------------
# Every keyword here is a phrase strict enough that a hit implies Indian-market
# context. Single broad words ("India", "Election") are deliberately avoided.

NIFTY_50_CONSTITUENTS = [
    "Reliance Industries", "HDFC Bank", "ICICI Bank", "Infosys", "TCS",
    "Larsen Toubro", "L&T", "Bharti Airtel", "ITC", "Kotak Mahindra Bank",
    "Hindustan Unilever", "Bajaj Finance", "Maruti Suzuki", "Mahindra Mahindra",
    "Asian Paints", "Sun Pharma", "Tata Motors", "Tata Steel", "Wipro",
    "NTPC", "Power Grid", "UltraTech Cement", "Titan Company", "Nestle India",
    "Adani Enterprises", "Adani Ports", "JSW Steel", "ONGC", "Coal India",
    "Tech Mahindra", "HCL Technologies", "Bajaj Finserv", "State Bank of India",
    "Axis Bank", "Hindalco", "Britannia Industries", "Cipla", "Dr Reddy",
    "Eicher Motors", "Grasim Industries", "Hero MotoCorp", "IndusInd Bank",
    "Trent Limited", "BPCL", "Apollo Hospitals", "Tata Consumer", "Bajaj Auto",
    "Shriram Finance", "LTIMindtree", "HDFC Life", "SBI Life",
]

# Macro / index phrases — strict, Indian-market only.
MACRO_AND_INDEX_TERMS = [
    "Nifty 50", "Nifty50", "Sensex", "BSE Sensex",
    "Indian stock market", "Indian markets", "Indian equities",
    "NSE India", "BSE India",
    "RBI", "Reserve Bank of India", "repo rate", "MPC meeting",
    "Indian economy", "Indian GDP",
    "FII inflow", "FII outflow", "foreign portfolio investor",
    "Union Budget", "Lok Sabha election", "Modi government",
    "FOMC", "Fed rate decision",
]

# Whitelisted Indian financial news domains (NewsAPI `domains` filter).
INDIAN_FINANCIAL_DOMAINS = [
    "economictimes.indiatimes.com",
    "livemint.com",
    "business-standard.com",
    "thehindubusinessline.com",
    "financialexpress.com",
    "moneycontrol.com",
    "ndtvprofit.com",
    "cnbctv18.com",
    "bloombergquint.com",
    "reuters.com",
    "bloomberg.com",
]

# Common short-forms that headlines frequently use ("Reliance" instead of
# "Reliance Industries", "HDFC" instead of "HDFC Bank", etc.). Hand-curated to
# stay specific — we deliberately omit ambiguous tokens like "Tata" alone (too
# many unrelated subsidiaries) and require multi-word context for those.
_NIFTY_STEM_TOKENS = [
    # Conglomerates / specific
    "reliance", "infosys", "wipro", "ntpc", "ongc", "ltimindtree",
    # Banks (each maps unambiguously to NIFTY 50 constituents)
    "hdfc", "icici", "axis bank", "kotak", "indusind", "sbi",
    "state bank", "yes bank",
    # Telecom / single-word stocks
    "airtel", "bharti",
    # Pharma
    "sun pharma", "dr reddy", "cipla", "apollo hospitals",
    # Consumer / paints
    "asian paints", "hindustan unilever", "britannia", "nestle india",
    "itc limited", "titan company", "trent ltd", "trent limited",
    # Autos (avoid bare "Tata")
    "maruti", "mahindra mahindra", "eicher motors", "hero motocorp",
    "tata motors", "tata steel", "tata consumer",
    # Bajaj group (specify which one)
    "bajaj finance", "bajaj finserv", "bajaj auto",
    # Cement / metals
    "ultratech", "grasim", "hindalco", "jsw steel",
    # Tech
    "tech mahindra", "hcl tech",
    # Adani (specify which one)
    "adani enterprises", "adani ports",
    # Energy
    "bpcl", "power grid", "coal india",
    # Other
    "shriram finance", "hdfc life", "sbi life",
]

# Lowercase keyword set used by the post-fetch relevance filter.
_RELEVANCE_KEYWORDS = {s.lower() for s in NIFTY_50_CONSTITUENTS + MACRO_AND_INDEX_TERMS}
_RELEVANCE_KEYWORDS.update(s.lower() for s in _NIFTY_STEM_TOKENS)
_RELEVANCE_KEYWORDS.update({
    "nifty", "sensex", "rbi", "nse", "bse",
    "indian market", "indian stocks", "indian shares", "indian rupee",
})


class SentimentAgent:
    """
    Fetches NIFTY-relevant news from a whitelist of Indian financial domains,
    runs a polarity score over recent headlines, and reports a recency-weighted
    market-sentiment verdict. Caches the merged article list per day with a
    1-hour TTL so NewsAPI quota stays under control.
    """

    CACHE_EXPIRATION_SECONDS = 3600  # 1 hour

    def __init__(self, config):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config['news_api']['api_key'])
        self.cache_dir = "news_cache"
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # NewsAPI plumbing
    # ------------------------------------------------------------------ #

    def _fetch_one_query(self, q: str, from_date, to_date,
                          use_qintitle: bool = True,
                          use_domain_filter: bool = True) -> list:
        """
        One defensive NewsAPI call. Returns the 'articles' list or [] on failure.
        Defaults to title-only matching + domain whitelist for maximum precision.
        """
        try:
            kwargs = dict(
                language="en",
                sort_by="publishedAt",
                page_size=100,
                from_param=from_date.isoformat(),
                to=to_date.isoformat(),
            )
            if use_domain_filter:
                kwargs["domains"] = ",".join(INDIAN_FINANCIAL_DOMAINS)
            if use_qintitle:
                kwargs["qintitle"] = q
            else:
                kwargs["q"] = q
            response = self.newsapi.get_everything(**kwargs)
            return (response or {}).get("articles", []) or []
        except Exception as e:
            logging.warning(
                f"SentimentAgent: news query failed (q='{q[:80]}...'): {e}"
            )
            return []

    @staticmethod
    def _is_relevant(article: dict) -> bool:
        """Drop anything that doesn't mention at least one NIFTY-relevant keyword."""
        title = (article.get('title') or '').lower()
        desc = (article.get('description') or '').lower()
        text = f"{title} {desc}"
        return any(kw in text for kw in _RELEVANCE_KEYWORDS)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _get_news_articles(self):
        """
        Build a NIFTY-relevant news payload by combining three targeted queries,
        deduplicating by URL, and post-filtering for relevance. Result is cached
        per day with a 1-hour TTL.

        Returns the standard newsapi-python payload shape:
            {"status": "ok", "totalResults": int, "articles": [...]}
        so existing consumers (get_market_sentiment, get_top_headlines,
        display_market_closed_info) keep working unchanged.
        """
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=2)
        cache_file_path = os.path.join(self.cache_dir, f"news_{today.isoformat()}.json")

        # Cache hit?
        if (os.path.exists(cache_file_path)
                and (time.time() - os.path.getmtime(cache_file_path)) < self.CACHE_EXPIRATION_SECONDS):
            logging.info(
                f"SentimentAgent: Loading recent news from cache "
                f"(less than {int(self.CACHE_EXPIRATION_SECONDS / 60)} minutes old)."
            )
            try:
                with open(cache_file_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"SentimentAgent: cache read failed ({e}); refetching.")

        logging.info("SentimentAgent: Fetching fresh news (NIFTY-relevant only)...")

        # Three batch queries. Splitting constituents keeps each query's URL
        # under NewsAPI's per-request length cap.
        queries = [
            " OR ".join(f'"{t}"' for t in MACRO_AND_INDEX_TERMS),
            " OR ".join(f'"{c}"' for c in NIFTY_50_CONSTITUENTS[:25]),
            " OR ".join(f'"{c}"' for c in NIFTY_50_CONSTITUENTS[25:]),
        ]

        unique_by_url = {}
        for q in queries:
            for article in self._fetch_one_query(q, from_date, today):
                url = article.get('url')
                if url and url not in unique_by_url:
                    unique_by_url[url] = article

        # Title-only match plus domain restriction is strict; the post-filter
        # is a safety net for anything that still slips through (e.g., an
        # Economic Times piece that mentions "FOMC" but is really about US tech).
        relevant = [a for a in unique_by_url.values() if self._is_relevant(a)]
        relevant.sort(key=lambda a: a.get('publishedAt', ''), reverse=True)

        logging.info(
            f"SentimentAgent: fetched {len(unique_by_url)} unique articles, "
            f"{len(relevant)} kept after relevance filter."
        )

        # If the strict path returned nothing, retry once without the domain
        # whitelist — the user's NewsAPI plan may not index those domains.
        # The relevance filter still keeps things NIFTY-only.
        if not relevant:
            logging.warning(
                "SentimentAgent: 0 relevant articles from whitelisted domains; "
                "retrying without domain filter."
            )
            unique_by_url.clear()
            for q in queries:
                for article in self._fetch_one_query(q, from_date, today,
                                                     use_qintitle=True,
                                                     use_domain_filter=False):
                    url = article.get('url')
                    if url and url not in unique_by_url:
                        unique_by_url[url] = article
            relevant = [a for a in unique_by_url.values() if self._is_relevant(a)]
            relevant.sort(key=lambda a: a.get('publishedAt', ''), reverse=True)
            logging.info(
                f"SentimentAgent: fallback fetch -> {len(unique_by_url)} unique, "
                f"{len(relevant)} relevant."
            )

        payload = {
            "status": "ok",
            "totalResults": len(relevant),
            "articles": relevant,
        }
        try:
            with open(cache_file_path, 'w') as f:
                json.dump(payload, f)
        except Exception as e:
            logging.warning(f"SentimentAgent: cache write failed (non-fatal): {e}")
        return payload

    def get_top_headlines(self, n: int = 10) -> list:
        """
        Returns up to `n` most recent valid headlines with their individual
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

    def get_market_sentiment(self):
        """
        Calculates sentiment using a recency-weighted average of TextBlob
        polarity scores over the (already NIFTY-filtered) headlines.
        """
        top_headlines = self._get_news_articles()
        if not top_headlines or not top_headlines.get('articles'):
            logging.warning("SentimentAgent: No NIFTY-relevant news articles found.")
            return "Neutral"

        sentiment_scores = []
        for article in top_headlines['articles']:
            title = article.get('title', '')
            if not title or title == "[Removed]":
                continue
            content = f"{title}. {article.get('description', '')}"
            sentiment_scores.append(TextBlob(content).sentiment.polarity)

        if not sentiment_scores:
            logging.warning("SentimentAgent: No valid headlines to analyze.")
            return "Neutral"

        # Weights decay linearly from N (newest) to 1 (oldest).
        n = len(sentiment_scores)
        weighted_sum = 0.0
        total_weight = 0
        for i, score in enumerate(sentiment_scores):
            weight = n - i
            weighted_sum += score * weight
            total_weight += weight
        avg_sentiment = weighted_sum / total_weight if total_weight else 0.0

        logging.info(
            f"SentimentAgent: Weighted average sentiment score is {avg_sentiment:.3f} "
            f"over {n} NIFTY-relevant articles."
        )

        if avg_sentiment > 0.4:    return "Very Bullish"
        if avg_sentiment > 0.05:   return "Bullish"
        if avg_sentiment < -0.4:   return "Very Bearish"
        if avg_sentiment < -0.05:  return "Bearish"
        return "Neutral"
