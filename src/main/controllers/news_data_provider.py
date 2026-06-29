"""News Data Provider for LCF.

Implements the DataProvider interface to fetch news articles from various sources:
- Google News RSS feeds
- Financial news APIs
- Custom RSS feeds

Responsibilities:
- Fetch news articles for given stock symbols
- Basic noise filtering (duplicates, irrelevant content)
- Return structured NewsItem-compatible data
"""

from __future__ import annotations

import re
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

from .data_provider import DataProvider, DataProviderResult, DataType, DataProviderConfig

logger = logging.getLogger(__name__)

# Graceful import of requests
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    requests = None


@dataclass
class NewsDataProviderConfig(DataProviderConfig):
    """Configuration for NewsDataProvider.
    
    Attributes:
        sources: List of news sources to use ("google", "rss")
        google_news_url: Google News RSS base URL
        custom_rss_feeds: Dict mapping symbols to custom RSS feed URLs
        lookback_hours: How many hours of news to fetch
        max_articles_per_symbol: Maximum articles per symbol
        filter_noise: Whether to filter noise articles
        noise_keywords: Keywords that indicate noise
        company_name_map: Dict mapping symbols to company names for search
        request_timeout: HTTP request timeout in seconds
    """
    sources: List[str] = field(default_factory=lambda: ["google"])
    google_news_url: str = "https://news.google.com/rss/search"
    custom_rss_feeds: Dict[str, str] = field(default_factory=dict)
    lookback_hours: int = 72
    max_articles_per_symbol: int = 20
    filter_noise: bool = True
    noise_keywords: List[str] = field(default_factory=lambda: [
        "sponsored", "advertisement", "promoted", "affiliate"
    ])
    company_name_map: Dict[str, str] = field(default_factory=dict)
    request_timeout: int = 30
    
    # Default supported types
    supported_types: Set[DataType] = field(default_factory=lambda: {DataType.NEWS})


class NewsDataProvider(DataProvider):
    """Data provider for fetching news articles.
    
    Fetches news from:
    - Google News RSS (default)
    - Custom RSS feeds
    - (future) Financial news APIs
    
    Features:
    - Deduplication based on title hash
    - Basic noise filtering
    - Date filtering (lookback window)
    
    Usage:
        config = NewsDataProviderConfig(
            lookback_hours=48,
            company_name_map={"TCS.NS": "Tata Consultancy Services"}
        )
        provider = NewsDataProvider(config)
        result = provider.fetch(["TCS.NS", "INFY.NS"])
        news = result.get_data(DataType.NEWS)
    """
    
    def __init__(self, config: Optional[NewsDataProviderConfig] = None):
        """Initialize NewsDataProvider.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        
        Raises:
            ImportError: If requests library is not installed.
        """
        if not _REQUESTS_AVAILABLE:
            raise ImportError(
                "requests is not installed. Install with: pip install requests"
            )
        
        self.config = config or NewsDataProviderConfig()
        self._seen_hashes: Set[str] = set()
        logger.info("NewsDataProvider initialized")
    
    @property
    def name(self) -> str:
        """Unique provider name."""
        return "news"
    
    @property
    def supported_types(self) -> Set[DataType]:
        """Data types this provider can fetch."""
        return self.config.supported_types
    
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch news articles for given symbols.
        
        Args:
            symbols: List of stock symbols
            data_types: Set of DataTypes (only NEWS supported)
            **kwargs: Additional parameters
        
        Returns:
            DataProviderResult containing news articles
        """
        symbols = self.validate_symbols(symbols)
        
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols,
            metadata={
                "lookback_hours": self.config.lookback_hours,
                "sources": self.config.sources,
            }
        )
        
        if data_types and DataType.NEWS not in data_types:
            return result
        
        # Clear seen hashes for new fetch
        self._seen_hashes.clear()
        
        all_articles = []
        
        for symbol in symbols:
            try:
                articles = self._fetch_news_for_symbol(symbol)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                result.add_error(f"News error for {symbol}: {str(e)}")
        
        result.add_data(DataType.NEWS, all_articles)
        logger.info(f"NewsDataProvider fetched {len(all_articles)} articles")
        return result
    
    def _fetch_news_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch news for a single symbol from all configured sources."""
        articles = []
        
        # Get search term (company name or symbol)
        search_term = self._get_search_term(symbol)
        
        # Fetch from each source
        if "google" in self.config.sources:
            google_articles = self._fetch_google_news(symbol, search_term)
            articles.extend(google_articles)
        
        # Fetch from custom RSS if configured
        if symbol in self.config.custom_rss_feeds:
            rss_articles = self._fetch_rss(
                symbol, 
                self.config.custom_rss_feeds[symbol]
            )
            articles.extend(rss_articles)
        
        # Filter and deduplicate
        filtered = self._filter_articles(articles)
        
        # Limit per symbol
        return filtered[:self.config.max_articles_per_symbol]
    
    def _get_search_term(self, symbol: str) -> str:
        """Get search term for a symbol (company name if available)."""
        # Check config map
        if symbol in self.config.company_name_map:
            return self.config.company_name_map[symbol]
        
        # Default: use symbol without exchange suffix
        clean_symbol = symbol.split(".")[0]
        return f"{clean_symbol} stock"
    
    def _fetch_google_news(
        self, 
        symbol: str, 
        search_term: str
    ) -> List[Dict[str, Any]]:
        """Fetch news from Google News RSS."""
        articles = []
        
        try:
            # Build Google News RSS URL
            encoded_query = quote_plus(search_term)
            url = f"{self.config.google_news_url}?q={encoded_query}&hl=en&gl=US&ceid=US:en"
            
            response = requests.get(
                url, 
                timeout=self.config.request_timeout,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            
            # Parse RSS XML
            root = ET.fromstring(response.content)
            
            # Find all items (articles)
            for item in root.findall(".//item"):
                article = self._parse_rss_item(item, symbol)
                if article:
                    articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} articles from Google News for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching Google News for {symbol}: {e}")
        
        return articles
    
    def _fetch_rss(self, symbol: str, rss_url: str) -> List[Dict[str, Any]]:
        """Fetch news from a custom RSS feed."""
        articles = []
        
        try:
            response = requests.get(
                rss_url, 
                timeout=self.config.request_timeout,
                headers={"User-Agent": "Mozilla/5.0"}
            )
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            for item in root.findall(".//item"):
                article = self._parse_rss_item(item, symbol)
                if article:
                    articles.append(article)
            
            logger.debug(f"Fetched {len(articles)} articles from RSS for {symbol}")
            
        except Exception as e:
            logger.error(f"Error fetching RSS for {symbol}: {e}")
        
        return articles
    
    def _parse_rss_item(
        self, 
        item: ET.Element, 
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a single RSS item into article dict."""
        try:
            title = item.findtext("title", "").strip()
            if not title:
                return None
            
            # Get published date
            pub_date_str = item.findtext("pubDate", "")
            publish_time = self._parse_date(pub_date_str)
            
            # Check if within lookback window
            if publish_time:
                cutoff = datetime.now() - timedelta(hours=self.config.lookback_hours)
                if publish_time < cutoff:
                    return None
            
            # Extract source from title (Google News format: "Title - Source")
            source = "Unknown"
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                if len(parts) == 2:
                    title = parts[0]
                    source = parts[1]
            
            # Get description/summary
            description = item.findtext("description", "")
            # Clean HTML from description
            description = re.sub(r'<[^>]+>', '', description).strip()
            
            return {
                "symbol": symbol,
                "title": title,
                "summary": description[:500] if description else None,
                "publisher": source,
                "link": item.findtext("link", ""),
                "publish_time": int(publish_time.timestamp()) if publish_time else None,
                "date": publish_time.strftime("%Y-%m-%d") if publish_time else "",
            }
            
        except Exception as e:
            logger.debug(f"Error parsing RSS item: {e}")
            return None
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string from RSS feed."""
        if not date_str:
            return None
        
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",  # RFC 822
            "%a, %d %b %Y %H:%M:%S GMT",
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601
            "%Y-%m-%dT%H:%M:%S%z",
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    def _filter_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and deduplicate articles."""
        filtered = []
        
        for article in articles:
            # Skip if title is empty
            title = article.get("title", "")
            if not title:
                continue
            
            # Deduplicate by title hash
            title_hash = hashlib.md5(title.lower().encode()).hexdigest()
            if title_hash in self._seen_hashes:
                continue
            self._seen_hashes.add(title_hash)
            
            # Filter noise
            if self.config.filter_noise and self._is_noise(article):
                continue
            
            filtered.append(article)
        
        # Sort by date (newest first)
        filtered.sort(
            key=lambda x: x.get("publish_time") or 0, 
            reverse=True
        )
        
        return filtered
    
    def _is_noise(self, article: Dict[str, Any]) -> bool:
        """Check if article is noise/spam."""
        title = article.get("title", "").lower()
        summary = article.get("summary", "").lower() if article.get("summary") else ""
        
        for keyword in self.config.noise_keywords:
            if keyword in title or keyword in summary:
                return True
        
        return False


def is_requests_available() -> bool:
    """Check if requests library is available."""
    return _REQUESTS_AVAILABLE
