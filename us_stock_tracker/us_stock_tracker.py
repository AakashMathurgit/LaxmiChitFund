"""US Stock Tracker — discovers important US stocks from RSS news feeds.

Parallel to news_stock_tracker/stock_tracker.py (Indian stocks).
Fetches from 8 US financial RSS feeds, extracts tickers, caches news items,
and runs the LCF pipeline (optional).

Usage:
    python us_stock_tracker/us_stock_tracker.py
"""

import time
import re
import os
import sys
import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime

import yaml  # type: ignore

# Add LCF root and src to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_LCF_ROOT, "src")
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Enable debug output
os.environ["LCF_DEBUG"] = "1"

# -------- CONFIG --------
UPDATE_INTERVAL = 600  # 10 minutes
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "stocks.txt")
US_SYMBOL_FILE = os.path.join(_SCRIPT_DIR, "us_symbols.txt")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "analysis_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "analysis_history.jsonl")
NEWS_CACHE_FILE = os.path.join(_LCF_ROOT, "data", "news_cache_us.jsonl")

# US Financial RSS Feeds
RSS_FEEDS = [
    # Seeking Alpha — small caps, analyst opinions, earnings alerts
    {
        "url": "https://seekingalpha.com/market_currents.xml",
        "source": "Seeking Alpha",
    },
    # Yahoo Finance — broad US stock news
    {
        "url": "https://finance.yahoo.com/news/rssindex",
        "source": "Yahoo Finance",
    },
    # MarketWatch — earnings, macro, market commentary
    {
        "url": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "source": "MarketWatch",
    },
    # CNBC — breaking financial news
    {
        "url": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "source": "CNBC",
    },
    # Investing.com — US stocks, economic events
    {
        "url": "https://www.investing.com/rss/news.rss",
        "source": "Investing.com",
    },
    # Nasdaq — corporate announcements, IPOs
    {
        "url": "https://www.nasdaq.com/feed/rssoutbound?category=Stock%20Market%20News",
        "source": "Nasdaq",
    },
    # Benzinga — trading news, small caps, unusual activity
    {
        "url": "https://www.benzinga.com/feed",
        "source": "Benzinga",
    },
    # Motley Fool — retail investor analysis
    {
        "url": "https://www.fool.com/feeds/index.aspx",
        "source": "Motley Fool",
    },
]

HEADERS = {
    "User-Agent": "LCF-USStockTracker/1.0 (research)",
    "Accept": "application/rss+xml, application/xml, text/xml",
}

TOKEN_REGEX = re.compile(r"\b[A-Z]{1,5}\b")  # US tickers are 1-5 uppercase letters

# Words that look like tickers but aren't
TICKER_BLACKLIST = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "AI", "CEO", "CFO", "CTO",
    "IPO", "FPO", "SEC", "FBI", "CIA", "GDP", "CPI", "FED", "NYSE",
    "DOW", "ETF", "IT", "IS", "AT", "ON", "IN", "OR", "AN", "AS",
    "BY", "TO", "UP", "IF", "SO", "NO", "DO", "GO", "BE", "HE",
    "WE", "MY", "TV", "PC", "PR", "HR", "VP", "VP", "ER", "ICU",
    "LLC", "INC", "LTD", "CO", "THE", "FOR", "AND", "BUT", "NOT",
    "ALL", "ARE", "WAS", "HAS", "HAD", "CAN", "MAY", "NEW", "OLD",
    "BIG", "TOP", "LOW", "HIGH", "RSI", "PE", "EPS", "ROE", "ROA",
    "FDA", "CDC", "WHO", "IMF", "ECB", "RBI", "BOJ", "EU", "UN",
    "API", "EV", "ESG", "DEI", "COO", "CMO", "CIO", "CSO", "CRO",
    "OTC", "SPX", "VIX", "DXY", "WTI", "OPEC",
}

HOT_NEWS_KEYWORDS = {
    "merger", "acquisition", "takeover", "buyout",
    "earnings", "revenue", "profit", "loss", "guidance",
    "upgrade", "downgrade", "price target",
    "dividend", "buyback", "stock split",
    "sec filing", "fda approval", "fda rejection",
    "layoffs", "restructuring",
    "bankruptcy", "fraud", "investigation",
    "ipo", "spac", "delisting",
    "activist investor", "hostile bid",
    "insider buying", "insider selling",
    "short squeeze", "options activity",
    "record high", "record low", "52-week",
    "beat estimates", "missed estimates",
    "raised guidance", "lowered guidance",
    "contract win", "partnership", "deal",
}
# ------------------------


def load_us_symbols():
    """Load valid US stock symbols."""
    if not os.path.exists(US_SYMBOL_FILE):
        print(f"[WARN] {US_SYMBOL_FILE} not found. Run generate_us_symbols.py first.")
        return set()
    with open(US_SYMBOL_FILE) as f:
        return set(line.strip() for line in f if line.strip())


def fetch_rss(url):
    """Fetch RSS feed content."""
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.text


def is_significant_news(title, description):
    """Check if news is market-significant using keyword heuristics."""
    text = f"{title} {description}".lower()
    return any(keyword in text for keyword in HOT_NEWS_KEYWORDS)


def classify_event_type(text):
    """Classify the event type from news text."""
    text = text.lower()
    EVENT_MAP = {
        "EARNINGS": ["earnings", "revenue", "profit", "loss", "quarterly", "beat estimates", "missed estimates"],
        "MERGER_ACQUISITION": ["merger", "acquisition", "takeover", "buyout", "hostile bid"],
        "DIVIDEND": ["dividend", "buyback", "stock split"],
        "REGULATORY": ["sec filing", "fda approval", "fda rejection", "investigation", "fraud"],
        "ANALYST": ["upgrade", "downgrade", "price target", "raised guidance", "lowered guidance"],
        "CONTRACT": ["contract win", "partnership", "deal"],
        "CORPORATE": ["ipo", "spac", "delisting", "layoffs", "restructuring"],
        "UNUSUAL_ACTIVITY": ["short squeeze", "options activity", "insider buying", "insider selling"],
    }
    for event_type, keywords in EVENT_MAP.items():
        if any(kw in text for kw in keywords):
            return event_type
    return ""


def extract_tickers(xml_text, valid_symbols, feed_info):
    """Extract tickers and cache-ready news items from RSS XML.

    Returns (tickers_set, news_items_list).
    """
    root = ET.fromstring(xml_text)
    found = set()
    news_items = []

    for item in root.findall(".//item"):
        title = item.findtext("title", "")
        desc = item.findtext("description", "")
        link = item.findtext("link", "")
        pub_date = item.findtext("pubDate", "")
        text = f"{title} {desc}"

        if not is_significant_news(title, desc):
            continue

        matched = set()
        for token in TOKEN_REGEX.findall(text):
            if token in valid_symbols and token not in TICKER_BLACKLIST:
                matched.add(token)
                found.add(token)

        if matched:
            event_type = classify_event_type(text)
            for sym in matched:
                news_items.append({
                    "symbol": sym,
                    "headline": title,
                    "description": desc,
                    "source": feed_info.get("source", ""),
                    "url": link,
                    "pub_date": pub_date,
                    "fetched_at": datetime.now().isoformat(),
                    "significant": True,
                    "event_type": event_type,
                    "matched_symbols": sorted(matched),
                    "market": "US",
                    "feed_url": feed_info.get("url", ""),
                })

    return found, news_items


def update_stocks():
    """Fetch all US RSS feeds, extract tickers, write to cache."""
    valid_symbols = load_us_symbols()
    if not valid_symbols:
        print("  [ERROR] No US symbols loaded!")
        return []

    all_stocks = set()
    all_news_items = []

    for feed_info in RSS_FEEDS:
        url = feed_info["url"]
        source = feed_info["source"]
        try:
            xml = fetch_rss(url)
            tickers, news_items = extract_tickers(xml, valid_symbols, feed_info)
            all_stocks.update(tickers)
            all_news_items.extend(news_items)
            print(f"  [{len(tickers):>2} tickers] {source}: {url[:60]}")
        except Exception as e:
            print(f"  [error] {source}: {str(e)[:60]}")

    with open(OUTPUT_FILE, "w") as f:
        for stock in sorted(all_stocks):
            f.write(stock + "\n")

    # HTML scraper discovery — scrape news sites for additional tickers
    try:
        from src.main.controllers.html_news_scraper import NewsPageScraper, NewsScraperConfig
        config = NewsScraperConfig(market="US", max_items_per_site=20)
        scraper = NewsPageScraper(config)
        scraped_items = scraper.scrape_all()
        scraper_tickers = set()
        scraper_news = []
        for item in scraped_items:
            text = f"{item.headline} {item.description}"
            for token in TOKEN_REGEX.findall(text):
                if token in valid_symbols and token not in TICKER_BLACKLIST:
                    scraper_tickers.add(token)
                    scraper_news.append({
                        "symbol": token,
                        "headline": item.headline,
                        "description": item.description,
                        "source": item.source,
                        "url": item.url,
                        "pub_date": item.pub_date,
                        "fetched_at": datetime.now().isoformat(),
                        "significant": True,
                        "event_type": "",
                        "matched_symbols": [token],
                        "market": "US",
                        "feed_url": item.url,
                    })
        new_from_scraper = scraper_tickers - all_stocks
        all_stocks.update(scraper_tickers)
        all_news_items.extend(scraper_news)
        print(f"  [scraper] {len(scraper_tickers)} tickers ({len(new_from_scraper)} new) from {len(scraped_items)} headlines")
    except Exception as e:
        print(f"  [scraper] failed: {e}")

    # Rewrite stocks.txt with scraper additions
    with open(OUTPUT_FILE, "w") as f:
        for stock in sorted(all_stocks):
            f.write(stock + "\n")

    # NLP enrichment — FinBERT sentiment + NER ticker extraction
    nlp_tickers = set()
    try:
        from src.main.controllers.nlp_processor import NLPProcessor
        nlp = NLPProcessor()

        # Extract tickers via NER from all news items
        for item in all_news_items:
            text = f"{item.get('headline', '')} {item.get('description', '')}"
            found = nlp.extract_tickers(text, valid_symbols)
            for t in found:
                if t not in all_stocks and t not in TICKER_BLACKLIST:
                    nlp_tickers.add(t)
                    all_stocks.add(t)

        # Batch sentiment on all headlines
        headlines = [item.get("headline", "") for item in all_news_items if item.get("headline")]
        if headlines and nlp.finbert_available:
            sentiments = nlp.analyze_sentiment_batch(headlines)
            for i, item in enumerate(all_news_items):
                if i < len(sentiments):
                    item["sentiment_label"] = sentiments[i]["label"]
                    item["sentiment_score"] = sentiments[i]["score"]
            print(f"  [nlp] FinBERT scored {len(sentiments)} headlines | {len(nlp_tickers)} new NER tickers")
        else:
            print(f"  [nlp] NER found {len(nlp_tickers)} new tickers (FinBERT unavailable)")
    except Exception as e:
        print(f"  [nlp] enrichment failed: {e}")

    # Final rewrite with NLP-discovered tickers
    if nlp_tickers:
        with open(OUTPUT_FILE, "w") as f:
            for stock in sorted(all_stocks):
                f.write(stock + "\n")

    # Write to shared news cache (now with sentiment scores)
    try:
        from src.main.controllers.news_cache import NewsCache
        cache = NewsCache(NEWS_CACHE_FILE, ttl_hours=72)
        written = cache.write_raw(all_news_items)
        print(f"Updated {len(all_stocks)} US stocks | Cached {written} news items")
    except Exception as e:
        print(f"Updated {len(all_stocks)} US stocks | Cache write failed: {e}")

    return sorted(all_stocks)


def main():
    print(f"\n{'#' * 60}")
    print(f"# LCF US Stock Tracker + News Feed")
    print(f"# Update interval: {UPDATE_INTERVAL // 60} minutes")
    print(f"# RSS feeds: {len(RSS_FEEDS)}")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    while True:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Fetching US RSS feeds...")

        # Step 1: Discover stocks from RSS news
        detected_symbols = update_stocks()

        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] "
              f"Sleeping for {UPDATE_INTERVAL // 60} minutes...")
        print("-" * 60)
        time.sleep(UPDATE_INTERVAL)


if __name__ == "__main__":
    main()
