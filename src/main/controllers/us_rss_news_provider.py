"""US RSS News Provider for LCF.

Fetches financial news from US market RSS feeds (Seeking Alpha, Yahoo Finance,
MarketWatch, CNBC, Investing.com, Nasdaq, Benzinga, Motley Fool) and provides
them as NEWS and EVENTS data types.

Cache-aware: reads from data/news_cache_us.jsonl first (written by us_stock_tracker),
falls back to live RSS fetch if cache is unavailable.
"""

from __future__ import annotations

import re
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import requests

from .data_provider import DataProvider, DataProviderConfig, DataProviderResult, DataType
from .data_context import NewsItem, EventData, StockDataContext

try:
    from ...utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# Token regex for US tickers (1-5 uppercase characters)
TOKEN_REGEX = re.compile(r"\b[A-Z]{1,5}\b")

# Common words that look like tickers but aren't
TICKER_BLACKLIST = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "AI", "CEO", "CFO", "CTO",
    "IPO", "SEC", "FBI", "GDP", "CPI", "FED", "NYSE", "DOW", "ETF",
    "IT", "IS", "AT", "ON", "IN", "OR", "AN", "AS", "BY", "TO", "UP",
    "IF", "SO", "NO", "DO", "GO", "BE", "HE", "WE", "MY", "TV", "PR",
    "LLC", "INC", "LTD", "CO", "THE", "FOR", "AND", "BUT", "NOT",
    "ALL", "ARE", "WAS", "HAS", "HAD", "CAN", "MAY", "NEW", "OLD",
    "BIG", "TOP", "LOW", "RSI", "PE", "EPS", "ROE", "ROA",
    "FDA", "CDC", "WHO", "IMF", "ECB", "UN",
    "API", "EV", "ESG", "OTC", "SPX", "VIX", "DXY", "WTI", "OPEC",
}

# Keywords for significant US market news
HOT_NEWS_KEYWORDS = {
    "merger", "acquisition", "takeover", "buyout",
    "earnings", "revenue", "profit", "loss", "guidance",
    "upgrade", "downgrade", "price target",
    "dividend", "buyback", "stock split",
    "sec filing", "fda approval", "fda rejection",
    "layoffs", "restructuring", "bankruptcy", "fraud",
    "ipo", "spac", "delisting",
    "insider buying", "insider selling",
    "short squeeze", "options activity",
    "beat estimates", "missed estimates",
    "contract win", "partnership", "deal",
}

# Event type classification
EVENT_KEYWORDS = {
    "EARNINGS": ["earnings", "quarterly", "revenue", "profit", "loss", "beat estimates", "missed estimates"],
    "MERGER_ACQUISITION": ["merger", "acquisition", "takeover", "buyout", "hostile bid"],
    "DIVIDEND": ["dividend", "buyback", "stock split"],
    "REGULATORY": ["sec filing", "fda approval", "fda rejection", "investigation", "fraud", "penalty"],
    "ANALYST": ["upgrade", "downgrade", "price target", "raised guidance", "lowered guidance"],
    "CORPORATE": ["ipo", "spac", "delisting", "layoffs", "restructuring"],
    "CONTRACT": ["contract win", "partnership", "deal", "agreement"],
    "UNUSUAL_ACTIVITY": ["short squeeze", "options activity", "insider buying", "insider selling"],
}

# Company name patterns for matching headlines to tickers
US_COMPANY_PATTERNS = {
    r"apple\b": "AAPL",
    r"microsoft\b": "MSFT",
    r"alphabet|google\b": "GOOGL",
    r"amazon\b": "AMZN",
    r"meta\s+platforms|facebook\b": "META",
    r"nvidia\b": "NVDA",
    r"tesla\b": "TSLA",
    r"netflix\b": "NFLX",
    r"advanced\s+micro|amd\b": "AMD",
    r"intel\b(?!\s+report)": "INTC",
    r"salesforce\b": "CRM",
    r"oracle\b": "ORCL",
    r"adobe\b": "ADBE",
    r"paypal\b": "PYPL",
    r"block\s+inc|square\b": "SQ",
    r"shopify\b": "SHOP",
    r"snowflake\b": "SNOW",
    r"palantir\b": "PLTR",
    r"coinbase\b": "COIN",
    r"uber\b": "UBER",
    r"airbnb\b": "ABNB",
    r"broadcom\b": "AVGO",
    r"qualcomm\b": "QCOM",
    r"jp\s*morgan|jpmorgan\b": "JPM",
    r"goldman\s+sachs": "GS",
    r"morgan\s+stanley": "MS",
    r"bank\s+of\s+america": "BAC",
    r"wells\s+fargo": "WFC",
    r"citigroup\b": "C",
    r"visa\b": "V",
    r"mastercard\b": "MA",
    r"johnson\s+&?\s*johnson|j&j\b": "JNJ",
    r"unitedhealth\b": "UNH",
    r"pfizer\b": "PFE",
    r"abbvie\b": "ABBV",
    r"merck\b": "MRK",
    r"eli\s+lilly|lilly\b": "LLY",
    r"moderna\b": "MRNA",
    r"exxon\s*mobil": "XOM",
    r"chevron\b": "CVX",
    r"walmart\b": "WMT",
    r"costco\b": "COST",
    r"home\s+depot": "HD",
    r"nike\b": "NKE",
    r"starbucks\b": "SBUX",
    r"mcdonald": "MCD",
    r"coca.?cola": "KO",
    r"pepsi(co)?\b": "PEP",
    r"procter\s*&?\s*gamble|p&g\b": "PG",
    r"disney\b": "DIS",
    r"boeing\b": "BA",
    r"caterpillar\b": "CAT",
    r"general\s+electric": "GE",
    r"rivian\b": "RIVN",
    r"lucid\b": "LCID",
    r"nio\b": "NIO",
    r"gamestop\b": "GME",
    r"amc\s+entertainment": "AMC",
    r"sofi\b": "SOFI",
    r"robinhood\b": "HOOD",
    r"super\s+micro": "SMCI",
    r"crowdstrike\b": "CRWD",
    r"datadog\b": "DDOG",
    r"cloudflare\b": "NET",
    r"arm\s+holdings": "ARM",
}


@dataclass
class USRSSNewsConfig(DataProviderConfig):
    """Configuration for US RSS News provider."""

    # RSS feed URLs
    feeds: List[Dict[str, str]] = field(default_factory=lambda: [
        {"url": "https://seekingalpha.com/market_currents.xml", "source": "Seeking Alpha"},
        {"url": "https://finance.yahoo.com/news/rssindex", "source": "Yahoo Finance"},
        {"url": "https://feeds.marketwatch.com/marketwatch/topstories/", "source": "MarketWatch"},
        {"url": "https://www.cnbc.com/id/100003114/device/rss/rss.html", "source": "CNBC"},
        {"url": "https://www.investing.com/rss/news.rss", "source": "Investing.com"},
        {"url": "https://www.nasdaq.com/feed/rssoutbound?category=Stock%20Market%20News", "source": "Nasdaq"},
        {"url": "https://www.benzinga.com/feed", "source": "Benzinga"},
        {"url": "https://www.fool.com/feeds/index.aspx", "source": "Motley Fool"},
    ])

    # Request settings
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "LCF-USStockTracker/1.0 (research)",
        "Accept": "application/rss+xml, application/xml, text/xml",
    })
    timeout_seconds: int = 20

    # Filtering
    only_significant_news: bool = True
    max_news_age_hours: int = 72

    # News cache
    cache_path: Optional[str] = None
    prefer_cache: bool = True

    # Company name patterns for matching
    company_patterns: Dict[str, str] = field(default_factory=lambda: dict(US_COMPANY_PATTERNS))

    # Supported data types
    supported_types: Set[DataType] = field(default_factory=lambda: {
        DataType.NEWS,
        DataType.EVENTS,
    })


class USRSSNewsProvider(DataProvider):
    """Data provider that fetches US stock news from RSS feeds.

    Cache-aware: reads from news_cache_us.jsonl first, falls back to live fetch.
    """

    def __init__(self, config: Optional[USRSSNewsConfig] = None):
        self._config = config or USRSSNewsConfig()
        self._compiled_patterns: Dict[re.Pattern, str] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        for pattern, symbol in self._config.company_patterns.items():
            self._compiled_patterns[re.compile(pattern, re.IGNORECASE)] = symbol

    @property
    def name(self) -> str:
        return "us_rss_news"

    @property
    def supported_types(self) -> Set[DataType]:
        return self._config.supported_types

    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs,
    ) -> DataProviderResult:
        """Fetch US news for given symbols. Tries cache + Google News per-ticker."""
        result = DataProviderResult(provider_name=self.name, symbols=symbols)
        types_to_fetch = data_types or self.supported_types

        # Normalize symbols
        symbol_set = {s.upper() for s in symbols}

        # --- Parallel: cache + Google News per-ticker ---
        news_items: List[NewsItem] = []
        event_items: List[EventData] = []
        sources_used = []

        # 1. Read from cache
        cached = self._read_from_cache(symbol_set)
        if cached and self._config.prefer_cache:
            cache_news, cache_events = cached
            news_items.extend(cache_news)
            event_items.extend(cache_events)
            sources_used.append("cache")

        # 2. Google News per-ticker (always runs to enrich data)
        google_news, google_events = self._fetch_google_news_for_symbols(
            list(symbol_set), types_to_fetch, market="US"
        )
        news_items.extend(google_news)
        event_items.extend(google_events)
        if google_news:
            sources_used.append(f"google_news({len(google_news)})")

        # 3. HTML scraper (scrapes news pages for deeper data)
        scraped_news, scraped_events = self._fetch_scraped_news(
            list(symbol_set), types_to_fetch, market="US"
        )
        news_items.extend(scraped_news)
        event_items.extend(scraped_events)
        if scraped_news:
            sources_used.append(f"scraper({len(scraped_news)})")

        # If we got data from cache + google + scraper, return it
        if news_items:
            if news_items and DataType.NEWS in types_to_fetch:
                result.add_data(DataType.NEWS, news_items)
            if event_items and DataType.EVENTS in types_to_fetch:
                result.add_data(DataType.EVENTS, event_items)
            result.metadata["source"] = "+".join(sources_used)
            result.metadata["items_matched"] = len(news_items)
            logger.info(
                f"USRSSNewsProvider ({'+'.join(sources_used)}): {len(news_items)} news, "
                f"{len(event_items)} events for {symbols}"
            )
            return result

        # 3. Fall back to live RSS fetch (all feeds)
        all_items: List[Dict[str, Any]] = []
        for feed_config in self._config.feeds:
            try:
                items = self._fetch_feed(feed_config["url"], feed_config["source"])
                all_items.extend(items)
            except Exception as e:
                result.add_error(f"Error fetching {feed_config['source']}: {str(e)}")
                logger.warning(f"Failed to fetch {feed_config['url']}: {e}")

        live_news: List[NewsItem] = []
        live_events: List[EventData] = []

        for item in all_items:
            matched_symbols = self._match_symbols(item, symbol_set)
            if not matched_symbols:
                continue
            if self._config.only_significant_news and not self._is_significant(item):
                continue

            for symbol in matched_symbols:
                if DataType.NEWS in types_to_fetch:
                    live_news.append(self._create_news_item(symbol, item))
                if DataType.EVENTS in types_to_fetch:
                    event_type = self._classify_event_type(item)
                    if event_type:
                        live_events.append(self._create_event_item(symbol, item, event_type))

        if live_news:
            result.add_data(DataType.NEWS, live_news)
        if live_events:
            result.add_data(DataType.EVENTS, live_events)

        result.metadata["source"] = "live"
        result.metadata["feeds_processed"] = len(self._config.feeds)
        result.metadata["total_items_fetched"] = len(all_items)
        result.metadata["items_matched"] = len(live_news)

        logger.info(f"USRSSNewsProvider (live): {len(live_news)} news, {len(live_events)} events")
        return result

    # ------------------------------------------------------------------
    # RSS Fetching
    # ------------------------------------------------------------------

    def _fetch_feed(self, url: str, source: str) -> List[Dict[str, Any]]:
        resp = requests.get(url, headers=self._config.headers, timeout=self._config.timeout_seconds)
        resp.raise_for_status()
        items = []
        root = ET.fromstring(resp.text)
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "").strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            if title:
                items.append({
                    "title": title,
                    "description": description,
                    "link": link,
                    "pub_date": pub_date,
                    "source": source,
                    "text": f"{title} {description}",
                })
        return items

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def _match_symbols(self, item: Dict[str, Any], target_symbols: Set[str]) -> Set[str]:
        matched = set()
        text = item.get("text", "")
        for token in TOKEN_REGEX.findall(text):
            if token in target_symbols and token not in TICKER_BLACKLIST:
                matched.add(token)
        text_lower = text.lower()
        for pattern, symbol in self._compiled_patterns.items():
            if symbol in target_symbols and pattern.search(text_lower):
                matched.add(symbol)
        return matched

    def _is_significant(self, item: Dict[str, Any]) -> bool:
        text = item.get("text", "").lower()
        return any(kw in text for kw in HOT_NEWS_KEYWORDS)

    def _classify_event_type(self, item: Dict[str, Any]) -> Optional[str]:
        text = item.get("text", "").lower()
        for event_type, keywords in EVENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return event_type
        return None

    # ------------------------------------------------------------------
    # Item creation
    # ------------------------------------------------------------------

    def _create_news_item(self, symbol: str, item: Dict[str, Any]) -> NewsItem:
        pub_date = item.get("pub_date", "")
        date_str = self._parse_date(pub_date)
        return NewsItem(
            symbol=symbol,
            date=date_str,
            headline=item.get("title"),
            news_text=item.get("description"),
            source=item.get("source"),
            url=item.get("link"),
        )

    def _create_event_item(self, symbol: str, item: Dict[str, Any], event_type: str) -> EventData:
        date_str = self._parse_date(item.get("pub_date", ""))
        return EventData(
            symbol=symbol,
            date=date_str,
            event_type=event_type,
            description=item.get("title", ""),
            impact_score=self._calculate_impact_score(item),
            source=item.get("source"),
        )

    def _parse_date(self, pub_date: str) -> str:
        for fmt in [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%d-%b-%Y %H:%M:%S",
        ]:
            try:
                return datetime.strptime(pub_date, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
        return datetime.now().strftime("%Y-%m-%d")

    def _calculate_impact_score(self, item: Dict[str, Any]) -> float:
        text = item.get("text", "").lower()
        high_impact = {"fraud", "bankruptcy", "acquisition", "merger", "takeover", "fda approval", "fda rejection"}
        medium_impact = {"earnings", "profit", "loss", "upgrade", "downgrade", "dividend", "beat estimates"}
        score = 0.3
        if any(kw in text for kw in high_impact):
            score += 0.4
        if any(kw in text for kw in medium_impact):
            score += 0.2
        return min(score, 1.0)

    # ------------------------------------------------------------------
    # Cache integration
    # ------------------------------------------------------------------

    def _read_from_cache(self, symbol_set: Set[str]):
        """Read from news_cache_us.jsonl. Returns (news, events) or None."""
        if not self._config.cache_path:
            return None
        try:
            from .news_cache import NewsCache
            cache = NewsCache(self._config.cache_path, ttl_hours=self._config.max_news_age_hours)
            cached = cache.read_items(symbols=symbol_set, significant_only=True)
            if not cached:
                return None

            news_items: List[NewsItem] = []
            event_items: List[EventData] = []
            for item in cached:
                news_items.append(NewsItem(
                    symbol=item.symbol,
                    date=item.pub_date[:10] if item.pub_date else datetime.now().strftime("%Y-%m-%d"),
                    headline=item.headline,
                    news_text=item.description,
                    source=item.source,
                    url=item.url,
                ))
                if item.event_type:
                    event_items.append(EventData(
                        symbol=item.symbol,
                        date=item.pub_date[:10] if item.pub_date else datetime.now().strftime("%Y-%m-%d"),
                        event_type=item.event_type,
                        description=item.headline,
                        impact_score=item.impact_score,
                        source=item.source,
                    ))
            return news_items, event_items
        except Exception as e:
            logger.warning(f"US NewsCache read failed: {e}")
            return None

    # ------------------------------------------------------------------
    # StockDataContext enrichment
    # ------------------------------------------------------------------

    def enrich_stock_context(self, stock_ctx: StockDataContext, result: DataProviderResult) -> StockDataContext:
        symbol = stock_ctx.symbol.upper()
        for news_item in result.get_data(DataType.NEWS):
            if news_item.symbol == symbol:
                stock_ctx.news_items.append(news_item)
        for event in result.get_data(DataType.EVENTS):
            if event.symbol == symbol:
                stock_ctx.events.append(event)
        return stock_ctx

    # ------------------------------------------------------------------
    # Google News per-ticker feed
    # ------------------------------------------------------------------

    def _fetch_google_news_for_symbols(
        self,
        symbols: List[str],
        types_to_fetch: Set[DataType],
        market: str = "US",
    ) -> tuple:
        """Fetch Google News per-ticker RSS for each symbol in parallel.

        Returns (news_items, event_items).
        """
        news_items: List[NewsItem] = []
        event_items: List[EventData] = []

        try:
            from .google_news_ticker_feed import GoogleNewsTickerFeed, GoogleNewsFeedConfig
            config = GoogleNewsFeedConfig(market=market, max_items_per_symbol=10)
            fetcher = GoogleNewsTickerFeed(config)
            results = fetcher.fetch_for_symbols(symbols)

            for symbol, items in results.items():
                for item in items:
                    if DataType.NEWS in types_to_fetch:
                        news_items.append(self._create_news_item(symbol, item))
                    if DataType.EVENTS in types_to_fetch:
                        event_type = self._classify_event_type(item)
                        if event_type:
                            event_items.append(self._create_event_item(symbol, item, event_type))

        except Exception as e:
            logger.warning(f"Google News per-ticker fetch failed: {e}")

        return news_items, event_items

    # ------------------------------------------------------------------
    # HTML scraper integration
    # ------------------------------------------------------------------

    def _fetch_scraped_news(
        self,
        symbols: List[str],
        types_to_fetch: Set[DataType],
        market: str = "US",
    ) -> tuple:
        """Scrape HTML news pages and match to symbols.

        Uses both general page scraping AND per-ticker page scraping.
        Returns (news_items, event_items).
        """
        news_items: List[NewsItem] = []
        event_items: List[EventData] = []

        try:
            from .html_news_scraper import NewsPageScraper, NewsScraperConfig
            config = NewsScraperConfig(market=market, max_items_per_site=15)
            scraper = NewsPageScraper(config)

            # 1. General page scrape (front pages, filtered by symbol)
            matched = scraper.scrape_for_symbols(symbols)

            # 2. Per-ticker page scrape (dedicated ticker URLs)
            ticker_matched = scraper.scrape_ticker_pages_batch(symbols)

            # Merge results (per-ticker pages first, then general)
            all_matched: Dict[str, list] = {}
            for sym, items in ticker_matched.items():
                all_matched.setdefault(sym, []).extend(items)
            for sym, items in matched.items():
                all_matched.setdefault(sym, []).extend(items)

            for symbol, scraped_items in all_matched.items():
                for item in scraped_items:
                    rss_item = {
                        "title": item.headline,
                        "description": item.description,
                        "link": item.url,
                        "pub_date": item.pub_date,
                        "source": item.source,
                        "text": f"{item.headline} {item.description}",
                    }
                    if DataType.NEWS in types_to_fetch:
                        news_items.append(self._create_news_item(symbol, rss_item))
                    if DataType.EVENTS in types_to_fetch:
                        event_type = self._classify_event_type(rss_item)
                        if event_type:
                            event_items.append(self._create_event_item(symbol, rss_item, event_type))

        except Exception as e:
            logger.warning(f"HTML scraper fetch failed: {e}")

        return news_items, event_items