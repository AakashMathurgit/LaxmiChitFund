"""RSS News Provider for LCF.

Fetches financial news from multiple RSS feeds (Moneycontrol, Economic Times,
The Hindu, Google News) and provides them as NEWS and EVENTS data types.

This enables the pipeline to incorporate real-time news sentiment
into stock analysis.
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

# Token regex to find potential stock symbols in news text
TOKEN_REGEX = re.compile(r"\b[A-Z]{2,15}\b")

# Keywords indicating potentially market-moving news
HOT_NEWS_KEYWORDS = {
    "merger", "acquisition", "stake sale", "buyback", "dividend",
    "results", "profit", "loss", "guidance", "regulatory", "sebi",
    "downgrade", "upgrade", "fraud", "default", "bankruptcy",
    "order win", "contract", "investigation", "insolvency", "debt",
    "fii", "dii", "earnings", "quarterly", "annual", "bonus",
    "split", "rights issue", "ipo", "fpo", "delisting", "promoter",
    "block deal", "bulk deal", "insider trading", "board meeting",
}

# Event type keywords for classification
EVENT_KEYWORDS = {
    "earnings": ["results", "quarterly", "annual", "profit", "loss", "earnings", "revenue"],
    "merger_acquisition": ["merger", "acquisition", "takeover", "buyout", "stake sale"],
    "dividend": ["dividend", "interim dividend", "final dividend", "special dividend"],
    "corporate_action": ["split", "bonus", "rights issue", "buyback"],
    "regulatory": ["sebi", "regulatory", "investigation", "fraud", "penalty", "compliance"],
    "analyst": ["upgrade", "downgrade", "target", "rating", "recommendation"],
    "management": ["ceo", "cfo", "director", "board", "resignation", "appointment"],
    "contract": ["order win", "contract", "deal", "partnership", "agreement"],
}


@dataclass
class RSSNewsConfig(DataProviderConfig):
    """Configuration for RSS News provider."""
    
    # RSS feed URLs
    feeds: List[Dict[str, str]] = field(default_factory=lambda: [
        # Moneycontrol
        {"url": "https://www.moneycontrol.com/rss/marketreports.xml", "source": "Moneycontrol"},
        {"url": "https://www.moneycontrol.com/rss/results.xml", "source": "Moneycontrol"},
        # The Hindu Business
        {"url": "https://www.thehindu.com/business/markets/feeder/default.rss", "source": "The Hindu"},
        # Economic Times
        {"url": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms", "source": "Economic Times"},
        {"url": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms", "source": "Economic Times"},
        # Google News aggregators
        {"url": "https://news.google.com/rss/search?q=stock+market+india&hl=en-IN&gl=IN&ceid=IN:en", "source": "Google News"},
        {"url": "https://news.google.com/rss/search?q=company+results+India+stocks&hl=en-IN&gl=IN&ceid=IN:en", "source": "Google News"},
    ])
    
    # Request settings
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": "LCF-StockTracker/1.0"
    })
    timeout_seconds: int = 15
    
    # Filtering
    only_significant_news: bool = True  # Filter out non-market-moving news
    max_news_age_hours: int = 72  # Only keep news from last N hours
    
    # News cache — if set, reads from cache file first before live-fetching
    cache_path: Optional[str] = None
    prefer_cache: bool = True  # If True and cache has data, skip live fetch
    
    # Symbol mapping (company name patterns to NSE symbols)
    # This helps match news headlines to stock symbols
    company_patterns: Dict[str, str] = field(default_factory=lambda: {
        r"reliance(\s+industries)?": "RELIANCE",
        r"tata\s+consultancy|tcs\b": "TCS",
        r"infosys": "INFY",
        r"hdfc\s+bank": "HDFCBANK",
        r"icici\s+bank": "ICICIBANK",
        r"state\s+bank|sbi\b": "SBIN",
        r"bharti\s+airtel|airtel": "BHARTIARTL",
        r"hindustan\s+unilever|hul\b": "HINDUNILVR",
        r"itc\s+limited|itc\b": "ITC",
        r"kotak\s+mahindra": "KOTAKBANK",
        r"axis\s+bank": "AXISBANK",
        r"larsen\s+&?\s*toubro|l&t\b": "LT",
        r"bajaj\s+finance": "BAJFINANCE",
        r"asian\s+paints": "ASIANPAINT",
        r"maruti\s+suzuki|maruti": "MARUTI",
        r"sun\s+pharma": "SUNPHARMA",
        r"titan\s+company|titan": "TITAN",
        r"wipro": "WIPRO",
        r"hcl\s+tech": "HCLTECH",
        r"power\s+grid": "POWERGRID",
        r"ntpc": "NTPC",
        r"ongc": "ONGC",
        r"ultra\s+tech|ultratech": "ULTRACEMCO",
        r"tech\s+mahindra": "TECHM",
        r"nestle\s+india|nestle": "NESTLEIND",
        r"adani\s+enterprises": "ADANIENT",
        r"adani\s+ports": "ADANIPORTS",
        r"tata\s+motors": "TATAMOTORS",
        r"tata\s+steel": "TATASTEEL",
        r"coal\s+india": "COALINDIA",
        r"bharat\s+petroleum|bpcl": "BPCL",
        r"indian\s+oil|ioc\b": "IOC",
        r"gail\s+india|gail\b": "GAIL",
        r"hindustan\s+aeronautics|hal\b": "HAL",
        r"mrf\b": "MRF",
        r"bajaj\s+auto": "BAJAJ-AUTO",
        r"hero\s+motocorp": "HEROMOTOCO",
        r"eicher\s+motors": "EICHERMOT",
        r"britannia": "BRITANNIA",
        r"divis\s+lab": "DIVISLAB",
        r"dr\.?\s*reddy": "DRREDDY",
        r"cipla": "CIPLA",
        r"grasim": "GRASIM",
        r"sbi\s+life": "SBILIFE",
        r"hdfc\s+life": "HDFCLIFE",
        r"icici\s+prudential|icici\s+pru": "ICICIPRULI",
        r"indusind\s+bank": "INDUSINDBK",
        r"jsw\s+steel": "JSWSTEEL",
        r"tata\s+consumer": "TATACONSUM",
        r"apollo\s+hospital": "APOLLOHOSP",
    })
    
    # Supported data types
    supported_types: Set[DataType] = field(default_factory=lambda: {
        DataType.NEWS,
        DataType.EVENTS,
    })


class RSSNewsProvider(DataProvider):
    """Data provider that fetches news from RSS feeds.
    
    Fetches real-time financial news from multiple Indian market RSS feeds
    and matches them to stock symbols for sentiment analysis.
    
    Usage:
        provider = RSSNewsProvider()
        
        # Fetch news for specific symbols
        result = provider.fetch(["TCS", "INFY", "RELIANCE"])
        
        # Get news items
        news = result.get_data(DataType.NEWS)
        events = result.get_data(DataType.EVENTS)
    """
    
    def __init__(self, config: Optional[RSSNewsConfig] = None):
        """Initialize RSS News provider.
        
        Args:
            config: Optional RSSNewsConfig. Uses defaults if not provided.
        """
        self._config = config or RSSNewsConfig()
        self._compiled_patterns: Dict[re.Pattern, str] = {}
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for company name matching."""
        for pattern, symbol in self._config.company_patterns.items():
            self._compiled_patterns[re.compile(pattern, re.IGNORECASE)] = symbol
    
    @property
    def name(self) -> str:
        return "rss_news"
    
    @property
    def supported_types(self) -> Set[DataType]:
        return self._config.supported_types
    
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch news from RSS feeds for the given symbols.
        
        Args:
            symbols: List of stock symbols to fetch news for
            data_types: Optional set of data types to fetch. If None, fetches all supported.
            **kwargs: Additional parameters (e.g., start_date, end_date)
        
        Returns:
            DataProviderResult containing NewsItem and EventData objects
        """
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols
        )
        
        types_to_fetch = data_types or self.supported_types
        
        # Normalize symbols to uppercase for matching
        symbol_set = {s.upper().replace(".NS", "").replace(".BO", "") for s in symbols}
        
        # --- Parallel: cache + Google News per-ticker ---
        news_items: List[NewsItem] = []
        event_items: List[EventData] = []
        sources_used = []

        # 1. Read from cache
        cached_news = self._read_from_cache(symbol_set)
        if cached_news and self._config.prefer_cache:
            cache_news, cache_events = cached_news
            news_items.extend(cache_news)
            event_items.extend(cache_events)
            sources_used.append("cache")

        # 2. Google News per-ticker (always runs in parallel to enrich data)
        google_news, google_events = self._fetch_google_news_for_symbols(
            list(symbol_set), types_to_fetch, market="IND"
        )
        news_items.extend(google_news)
        event_items.extend(google_events)
        if google_news:
            sources_used.append(f"google_news({len(google_news)})")

        # 3. HTML scraper (scrapes news pages for deeper data)
        scraped_news, scraped_events = self._fetch_scraped_news(
            list(symbol_set), types_to_fetch, market="IND"
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
                f"RSSNewsProvider ({'+'.join(sources_used)}): {len(news_items)} news, "
                f"{len(event_items)} events for symbols {symbols}"
            )
            return result

        # 3. Fall back to live RSS fetch (all feeds)
        all_items: List[Dict[str, Any]] = []
        
        for feed_config in self._config.feeds:
            try:
                items = self._fetch_feed(feed_config["url"], feed_config["source"])
                all_items.extend(items)
                logger.debug(f"Fetched {len(items)} items from {feed_config['source']}")
            except Exception as e:
                result.add_error(f"Error fetching {feed_config['source']}: {str(e)}")
                logger.warning(f"Failed to fetch {feed_config['url']}: {e}")
        
        # Process items and match to symbols
        news_items: List[NewsItem] = []
        event_items: List[EventData] = []
        
        for item in all_items:
            matched_symbols = self._match_symbols(item, symbol_set)
            
            if not matched_symbols:
                continue
            
            # Check significance if filtering is enabled
            if self._config.only_significant_news:
                if not self._is_significant(item):
                    continue
            
            for symbol in matched_symbols:
                # Create NewsItem
                if DataType.NEWS in types_to_fetch:
                    news_item = self._create_news_item(symbol, item)
                    news_items.append(news_item)
                
                # Create EventData if this looks like an event
                if DataType.EVENTS in types_to_fetch:
                    event_type = self._classify_event_type(item)
                    if event_type:
                        event_item = self._create_event_item(symbol, item, event_type)
                        event_items.append(event_item)
        
        # Add to result
        if news_items:
            result.add_data(DataType.NEWS, news_items)
        if event_items:
            result.add_data(DataType.EVENTS, event_items)
        
        result.metadata["feeds_processed"] = len(self._config.feeds)
        result.metadata["total_items_fetched"] = len(all_items)
        result.metadata["items_matched"] = len(news_items)
        
        logger.info(
            f"RSSNewsProvider: {len(news_items)} news, {len(event_items)} events "
            f"for symbols {symbols}"
        )
        
        return result
    
    def _fetch_feed(self, url: str, source: str) -> List[Dict[str, Any]]:
        """Fetch and parse a single RSS feed.
        
        Args:
            url: RSS feed URL
            source: Source name for attribution
            
        Returns:
            List of parsed news items as dictionaries
        """
        response = requests.get(
            url,
            headers=self._config.headers,
            timeout=self._config.timeout_seconds
        )
        response.raise_for_status()
        
        items = []
        root = ET.fromstring(response.text)
        
        for item in root.findall(".//item"):
            title = item.findtext("title", "").strip()
            description = item.findtext("description", "").strip()
            link = item.findtext("link", "").strip()
            pub_date = item.findtext("pubDate", "").strip()
            
            if not title:
                continue
            
            items.append({
                "title": title,
                "description": description,
                "link": link,
                "pub_date": pub_date,
                "source": source,
                "text": f"{title} {description}",
            })
        
        return items
    
    def _match_symbols(
        self,
        item: Dict[str, Any],
        target_symbols: Set[str]
    ) -> Set[str]:
        """Match news item to stock symbols.
        
        Uses both direct symbol mentions and company name pattern matching.
        
        Args:
            item: News item dictionary
            target_symbols: Set of symbols we're looking for
            
        Returns:
            Set of matched symbols
        """
        matched = set()
        text = item.get("text", "")
        
        # Direct symbol matching (e.g., "TCS", "INFY")
        for token in TOKEN_REGEX.findall(text):
            if token in target_symbols:
                matched.add(token)
        
        # Company name pattern matching
        text_lower = text.lower()
        for pattern, symbol in self._compiled_patterns.items():
            if symbol in target_symbols and pattern.search(text_lower):
                matched.add(symbol)
        
        return matched
    
    def _is_significant(self, item: Dict[str, Any]) -> bool:
        """Check if news item is market-significant.
        
        Args:
            item: News item dictionary
            
        Returns:
            True if the news is potentially market-moving
        """
        text = item.get("text", "").lower()
        return any(keyword in text for keyword in HOT_NEWS_KEYWORDS)
    
    def _classify_event_type(self, item: Dict[str, Any]) -> Optional[str]:
        """Classify the event type based on keywords.
        
        Args:
            item: News item dictionary
            
        Returns:
            Event type string or None if not an event
        """
        text = item.get("text", "").lower()
        
        for event_type, keywords in EVENT_KEYWORDS.items():
            if any(kw in text for kw in keywords):
                return event_type.upper()
        
        return None
    
    def _create_news_item(self, symbol: str, item: Dict[str, Any]) -> NewsItem:
        """Create a NewsItem from parsed RSS data.
        
        Args:
            symbol: Stock symbol
            item: Parsed RSS item
            
        Returns:
            NewsItem dataclass instance
        """
        # Parse date
        pub_date = item.get("pub_date", "")
        try:
            # Try common RSS date formats
            for fmt in [
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%d-%b-%Y %H:%M:%S",
            ]:
                try:
                    dt = datetime.strptime(pub_date, fmt)
                    date_str = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        return NewsItem(
            symbol=symbol,
            date=date_str,
            headline=item.get("title"),
            news_text=item.get("description"),
            source=item.get("source"),
            url=item.get("link"),
        )
    
    def _create_event_item(
        self,
        symbol: str,
        item: Dict[str, Any],
        event_type: str
    ) -> EventData:
        """Create an EventData from parsed RSS data.
        
        Args:
            symbol: Stock symbol
            item: Parsed RSS item
            event_type: Classified event type
            
        Returns:
            EventData dataclass instance
        """
        # Parse date
        pub_date = item.get("pub_date", "")
        try:
            for fmt in [
                "%a, %d %b %Y %H:%M:%S %z",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%Y-%m-%dT%H:%M:%S%z",
            ]:
                try:
                    dt = datetime.strptime(pub_date, fmt)
                    date_str = dt.strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
            else:
                date_str = datetime.now().strftime("%Y-%m-%d")
        except Exception:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        # Calculate impact score based on keywords
        impact_score = self._calculate_impact_score(item)
        
        return EventData(
            symbol=symbol,
            date=date_str,
            event_type=event_type,
            description=item.get("title", ""),
            impact_score=impact_score,
            source=item.get("source"),
        )
    
    def _calculate_impact_score(self, item: Dict[str, Any]) -> float:
        """Calculate impact score based on keyword presence.
        
        Args:
            item: News item dictionary
            
        Returns:
            Impact score between 0.0 and 1.0
        """
        text = item.get("text", "").lower()
        
        # High impact keywords
        high_impact = {"fraud", "bankruptcy", "default", "acquisition", "merger", "takeover"}
        medium_impact = {"results", "profit", "loss", "upgrade", "downgrade", "dividend"}
        
        score = 0.3  # Base score for any matched news
        
        if any(kw in text for kw in high_impact):
            score += 0.4
        if any(kw in text for kw in medium_impact):
            score += 0.2
        
        return min(score, 1.0)
    
    def enrich_stock_context(
        self,
        stock_ctx: StockDataContext,
        result: DataProviderResult
    ) -> StockDataContext:
        """Enrich a StockDataContext with RSS news data.
        
        Args:
            stock_ctx: The StockDataContext to enrich
            result: DataProviderResult from this provider
            
        Returns:
            Enriched StockDataContext
        """
        symbol = stock_ctx.symbol.upper().replace(".NS", "").replace(".BO", "")
        
        # Add news items
        for news_item in result.get_data(DataType.NEWS):
            if news_item.symbol == symbol:
                stock_ctx.news_items.append(news_item)
        
        # Add events
        for event in result.get_data(DataType.EVENTS):
            if event.symbol == symbol:
                stock_ctx.events.append(event)
        
        return stock_ctx

    # ------------------------------------------------------------------
    # Cache integration
    # ------------------------------------------------------------------

    def _read_from_cache(self, symbol_set: Set[str]):
        """Try to read news from the shared NewsCache.

        Returns (news_items, event_items) or None if cache is unavailable.
        """
        if not self._config.cache_path:
            return None

        try:
            from .news_cache import NewsCache
            cache = NewsCache(
                self._config.cache_path,
                ttl_hours=self._config.max_news_age_hours,
            )
            cached = cache.read_items(symbols=symbol_set, significant_only=True)
            if not cached:
                return None

            news_items: List[NewsItem] = []
            event_items: List[EventData] = []

            for item in cached:
                # Create NewsItem
                news_items.append(NewsItem(
                    symbol=item.symbol,
                    date=item.pub_date[:10] if item.pub_date else datetime.now().strftime("%Y-%m-%d"),
                    headline=item.headline,
                    news_text=item.description,
                    source=item.source,
                    url=item.url,
                ))

                # Create EventData if classifiable
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
            logger.warning(f"NewsCache read failed: {e}")
            return None

    # ------------------------------------------------------------------
    # Google News per-ticker feed
    # ------------------------------------------------------------------

    def _fetch_google_news_for_symbols(
        self,
        symbols: List[str],
        types_to_fetch: Set[DataType],
        market: str = "IND",
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
        market: str = "IND",
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