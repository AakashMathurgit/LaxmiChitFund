"""HTML News Scraper for LCF.

Scrapes financial news pages from major sites to get deeper article data
beyond what RSS feeds provide. Runs in parallel with RSS/cache reads.

Supported sites:
    - Seeking Alpha (market-news)
    - Benzinga (news)
    - Yahoo Finance (news)
    - MarketWatch (latest-news)
    - Nasdaq (news-and-insights)

Each site has its own parser to handle different HTML structures.
All scrapers share a common interface and return standardized news items.

Usage:
    scraper = NewsPageScraper(market="US")
    items = scraper.scrape_all()                     # All sites
    items = scraper.scrape_site("seeking_alpha")     # Single site
    items = scraper.scrape_for_symbols(["NVDA"])     # Filter to symbols
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin

import requests

try:
    from bs4 import BeautifulSoup
    _BS4_AVAILABLE = True
except ImportError:
    _BS4_AVAILABLE = False
    BeautifulSoup = None

try:
    from ...utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# Token regex for matching tickers in headlines
US_TICKER_RE = re.compile(r"\b[A-Z]{1,5}\b")
IND_TICKER_RE = re.compile(r"\b[A-Z]{2,15}\b")

# Common non-ticker words
TICKER_BLACKLIST = {
    "A", "I", "AM", "PM", "US", "UK", "EU", "AI", "CEO", "CFO", "CTO",
    "IPO", "SEC", "FBI", "GDP", "CPI", "FED", "NYSE", "DOW", "ETF",
    "IT", "IS", "AT", "ON", "IN", "OR", "AN", "AS", "BY", "TO", "UP",
    "IF", "SO", "NO", "DO", "GO", "BE", "HE", "WE", "MY", "TV", "PR",
    "LLC", "INC", "LTD", "CO", "THE", "FOR", "AND", "BUT", "NOT",
    "ALL", "ARE", "WAS", "HAS", "HAD", "CAN", "MAY", "NEW", "OLD",
    "BIG", "TOP", "LOW", "RSI", "PE", "EPS", "ROE", "ROA",
    "FDA", "CDC", "WHO", "IMF", "ECB", "UN", "API", "EV", "ESG",
    "OTC", "SPX", "VIX", "DXY", "WTI", "OPEC", "HIGH",
    "SEBI", "RBI", "NSE", "BSE", "FII", "DII", "NRI",
}


@dataclass
class ScrapedNewsItem:
    """A single news item scraped from an HTML page."""
    headline: str
    description: str = ""
    url: str = ""
    source: str = ""
    pub_date: str = ""
    author: str = ""
    tickers_mentioned: List[str] = field(default_factory=list)
    category: str = ""          # earnings, analysis, market-news, etc.
    scraped_at: str = ""

    def to_cache_dict(self, symbol: str, market: str = "US") -> Dict[str, Any]:
        """Convert to a dict suitable for NewsCache.write_raw()."""
        return {
            "symbol": symbol,
            "headline": self.headline,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "pub_date": self.pub_date,
            "fetched_at": self.scraped_at or datetime.now().isoformat(),
            "significant": True,
            "matched_symbols": self.tickers_mentioned,
            "market": market,
            "feed_url": self.url,
        }


@dataclass
class ScraperSiteConfig:
    """Configuration for a single scraping target."""
    name: str
    url: str
    parser: str           # Name of the parser method to use
    enabled: bool = True


@dataclass
class NewsScraperConfig:
    """Configuration for the HTML news scraper."""
    market: str = "US"
    max_parallel: int = 3
    timeout_seconds: int = 20
    max_items_per_site: int = 20
    headers: Dict[str, str] = field(default_factory=lambda: {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })

    # Sites to scrape
    sites: List[ScraperSiteConfig] = field(default_factory=lambda: [
        ScraperSiteConfig(
            name="seeking_alpha",
            url="https://seekingalpha.com/market-news",
            parser="_parse_seeking_alpha",
        ),
        ScraperSiteConfig(
            name="benzinga",
            url="https://www.benzinga.com/news",
            parser="_parse_benzinga",
        ),
        ScraperSiteConfig(
            name="yahoo_finance",
            url="https://finance.yahoo.com/news/",
            parser="_parse_yahoo_finance",
        ),
        ScraperSiteConfig(
            name="marketwatch",
            url="https://www.marketwatch.com/latest-news",
            parser="_parse_marketwatch",
            enabled=False,  # Returns 403 Forbidden
        ),
        ScraperSiteConfig(
            name="nasdaq",
            url="https://www.nasdaq.com/news-and-insights",
            parser="_parse_nasdaq",
        ),
    ])

    # Indian market sites (used when market="IND")
    ind_sites: List[ScraperSiteConfig] = field(default_factory=lambda: [
        ScraperSiteConfig(
            name="moneycontrol",
            url="https://www.moneycontrol.com/news/business/markets/",
            parser="_parse_moneycontrol",
        ),
        ScraperSiteConfig(
            name="economic_times",
            url="https://economictimes.indiatimes.com/markets/stocks/news",
            parser="_parse_economic_times",
        ),
        ScraperSiteConfig(
            name="livemint",
            url="https://www.livemint.com/market/stock-market-news",
            parser="_parse_livemint",
        ),
    ])


class NewsPageScraper:
    """Scrapes news from financial websites for headline + article data.

    Each site has a dedicated parser that handles its specific HTML structure.
    Falls back to a generic parser if the dedicated one fails.
    """

    def __init__(self, config: Optional[NewsScraperConfig] = None):
        if not _BS4_AVAILABLE:
            raise ImportError("beautifulsoup4 is required. Install: pip install beautifulsoup4")
        self._config = config or NewsScraperConfig()
        self._session = requests.Session()
        self._session.headers.update(self._config.headers)

    @property
    def active_sites(self) -> List[ScraperSiteConfig]:
        """Get enabled sites for the configured market."""
        if self._config.market == "IND":
            return [s for s in self._config.ind_sites if s.enabled]
        return [s for s in self._config.sites if s.enabled]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_all(self) -> List[ScrapedNewsItem]:
        """Scrape all enabled sites in parallel. Returns combined items."""
        sites = self.active_sites
        all_items: List[ScrapedNewsItem] = []

        with ThreadPoolExecutor(max_workers=self._config.max_parallel) as executor:
            futures = {
                executor.submit(self._scrape_site, site): site
                for site in sites
            }
            for future in as_completed(futures):
                site = futures[future]
                try:
                    items = future.result()
                    all_items.extend(items)
                    logger.debug(f"Scraped {len(items)} items from {site.name}")
                except Exception as e:
                    logger.warning(f"Scrape failed for {site.name}: {e}")

        logger.info(f"NewsPageScraper: {len(all_items)} items from {len(sites)} sites")
        return all_items

    def scrape_site(self, site_name: str) -> List[ScrapedNewsItem]:
        """Scrape a single site by name."""
        for site in self.active_sites:
            if site.name == site_name:
                return self._scrape_site(site)
        logger.warning(f"Site '{site_name}' not found")
        return []

    def scrape_for_symbols(
        self,
        symbols: List[str],
        valid_symbols: Optional[Set[str]] = None,
    ) -> Dict[str, List[ScrapedNewsItem]]:
        """Scrape all sites and filter to items mentioning given symbols.

        Uses NLP entity extraction + direct ticker matching + company name matching.
        Returns dict: symbol -> list of matched items.
        """
        all_items = self.scrape_all()
        symbol_set = {s.upper() for s in symbols}
        ticker_re = IND_TICKER_RE if self._config.market == "IND" else US_TICKER_RE

        # Build reverse name->symbol map from google_news_ticker_feed
        name_patterns = self._build_name_patterns(symbol_set)

        # Try NLP processor for NER-based matching
        nlp = None
        try:
            from .nlp_processor import NLPProcessor
            nlp = NLPProcessor()
        except Exception:
            pass

        results: Dict[str, List[ScrapedNewsItem]] = {s: [] for s in symbol_set}

        for item in all_items:
            text = f"{item.headline} {item.description}"
            text_lower = text.lower()
            found_tickers = set()

            # 1. NLP entity extraction → ticker (highest quality)
            if nlp:
                nlp_tickers = nlp.extract_tickers(text, symbol_set)
                found_tickers.update(t for t in nlp_tickers if t in symbol_set)

            # 2. Direct ticker matching
            for token in ticker_re.findall(text):
                if token in symbol_set and token not in TICKER_BLACKLIST:
                    found_tickers.add(token)

            # 3. Company name matching (e.g., "Nvidia" -> NVDA)
            for pattern, symbol in name_patterns:
                if pattern.search(text_lower):
                    found_tickers.add(symbol)

            item.tickers_mentioned = sorted(found_tickers)
            for sym in found_tickers:
                results[sym].append(item)

        matched_count = sum(len(v) for v in results.values())
        logger.info(f"NewsPageScraper: {matched_count} items matched to {len(symbol_set)} symbols")
        return {k: v for k, v in results.items() if v}

    def _build_name_patterns(self, symbol_set: Set[str]) -> List[Tuple[re.Pattern, str]]:
        """Build compiled regex patterns for company name -> ticker matching."""
        patterns = []
        try:
            from .google_news_ticker_feed import US_TICKER_NAMES, IND_TICKER_NAMES
            name_map = IND_TICKER_NAMES if self._config.market == "IND" else US_TICKER_NAMES
            for symbol, name in name_map.items():
                if symbol in symbol_set:
                    # Split name into words and create a pattern matching any of them (>3 chars)
                    words = [w for w in name.split() if len(w) > 3]
                    if words:
                        # Match the first significant word (usually the company name)
                        pattern_str = r"\b" + re.escape(words[0].lower()) + r"\b"
                        patterns.append((re.compile(pattern_str, re.IGNORECASE), symbol))
        except ImportError:
            pass
        return patterns

    # ------------------------------------------------------------------
    # Per-ticker page scraping (targeted)
    # ------------------------------------------------------------------

    def scrape_ticker_pages(self, symbol: str) -> List[ScrapedNewsItem]:
        """Scrape ticker-specific news pages for a single symbol.

        Hits dedicated per-ticker URLs on each site (e.g. seekingalpha.com/symbol/NVDA/news).
        Returns all scraped items tagged with this symbol.
        """
        ticker_urls = self._get_ticker_urls(symbol)
        if not ticker_urls:
            return []

        all_items: List[ScrapedNewsItem] = []
        with ThreadPoolExecutor(max_workers=self._config.max_parallel) as executor:
            futures = {
                executor.submit(self._scrape_ticker_page, url_info, symbol): url_info
                for url_info in ticker_urls
            }
            for future in as_completed(futures):
                url_info = futures[future]
                try:
                    items = future.result()
                    all_items.extend(items)
                except Exception as e:
                    logger.debug(f"Ticker page scrape failed for {url_info['source']}/{symbol}: {e}")

        logger.info(f"NewsPageScraper: {len(all_items)} ticker-specific items for {symbol}")
        return all_items

    def scrape_ticker_pages_batch(self, symbols: List[str]) -> Dict[str, List[ScrapedNewsItem]]:
        """Scrape ticker-specific pages for multiple symbols in parallel.

        Returns dict: symbol -> list of scraped items.
        """
        results: Dict[str, List[ScrapedNewsItem]] = {}
        with ThreadPoolExecutor(max_workers=min(len(symbols), 5)) as executor:
            futures = {
                executor.submit(self.scrape_ticker_pages, sym): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    items = future.result()
                    if items:
                        results[sym] = items
                except Exception as e:
                    logger.debug(f"Ticker batch scrape failed for {sym}: {e}")

        total = sum(len(v) for v in results.values())
        logger.info(f"NewsPageScraper: {total} ticker-page items for {len(results)}/{len(symbols)} symbols")
        return results

    def _get_ticker_urls(self, symbol: str) -> List[Dict[str, str]]:
        """Build per-ticker URLs for each site."""
        sym = symbol.upper()
        if self._config.market == "US":
            return [
                {
                    "url": f"https://seekingalpha.com/symbol/{sym}/news",
                    "source": "Seeking Alpha",
                    "parser": "_parse_seeking_alpha",
                },
                {
                    "url": f"https://finance.yahoo.com/quote/{sym}/news/",
                    "source": "Yahoo Finance",
                    "parser": "_parse_yahoo_finance",
                },
                {
                    "url": f"https://www.benzinga.com/stock/{sym}",
                    "source": "Benzinga",
                    "parser": "_parse_benzinga",
                },
                {
                    "url": f"https://www.nasdaq.com/market-activity/stocks/{sym.lower()}/news-headlines",
                    "source": "Nasdaq",
                    "parser": "_parse_nasdaq",
                },
            ]
        else:  # IND
            return [
                {
                    "url": f"https://www.moneycontrol.com/india/stockpricequote/{sym}",
                    "source": "Moneycontrol",
                    "parser": "_parse_moneycontrol",
                },
                {
                    "url": f"https://economictimes.indiatimes.com/topic/{sym}",
                    "source": "Economic Times",
                    "parser": "_parse_economic_times",
                },
            ]

    def _scrape_ticker_page(
        self, url_info: Dict[str, str], symbol: str
    ) -> List[ScrapedNewsItem]:
        """Scrape a single per-ticker page."""
        url = url_info["url"]
        source = url_info["source"]
        parser_name = url_info["parser"]
        now = datetime.now().isoformat()

        try:
            resp = self._session.get(url, timeout=self._config.timeout_seconds)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            parser_method = getattr(self, parser_name, None)
            site_config = ScraperSiteConfig(name=source, url=url, parser=parser_name)

            if parser_method:
                items = parser_method(soup, site_config)
            else:
                items = self._parse_generic(soup, site_config)

            # Tag all items with the target symbol
            for item in items:
                item.tickers_mentioned = [symbol]
                if not item.scraped_at:
                    item.scraped_at = now

            return items[:self._config.max_items_per_site]

        except Exception as e:
            logger.debug(f"Ticker page {source}/{symbol} failed: {e}")
            return []

    # ------------------------------------------------------------------
    # Internal scraping
    # ------------------------------------------------------------------

    def _scrape_site(self, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Fetch + parse a single site."""
        try:
            resp = self._session.get(site.url, timeout=self._config.timeout_seconds)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # Try dedicated parser first
            parser_method = getattr(self, site.parser, None)
            if parser_method:
                items = parser_method(soup, site)
            else:
                items = self._parse_generic(soup, site)

            # Limit items per site
            return items[: self._config.max_items_per_site]

        except Exception as e:
            logger.warning(f"Failed to scrape {site.name} ({site.url}): {e}")
            return []

    # ------------------------------------------------------------------
    # US Site Parsers
    # ------------------------------------------------------------------

    def _parse_seeking_alpha(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Seeking Alpha market-news page."""
        items = []
        now = datetime.now().isoformat()

        # SA uses article tags or div with data-test-id
        for article in soup.find_all("article", limit=30):
            headline_tag = article.find(["h3", "h4", "a"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            link = ""
            a_tag = article.find("a", href=True)
            if a_tag:
                link = urljoin("https://seekingalpha.com", a_tag["href"])
            desc_tag = article.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            if headline:
                items.append(ScrapedNewsItem(
                    headline=headline, description=desc, url=link,
                    source="Seeking Alpha", scraped_at=now,
                ))

        # Fallback: generic link extraction
        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_benzinga(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Benzinga news page."""
        items = []
        now = datetime.now().isoformat()

        # Benzinga uses .content-feed-list or .story blocks
        for block in soup.find_all(["article", "div"], class_=re.compile(r"story|news-item|content-feed"), limit=30):
            headline_tag = block.find(["h3", "h2", "a"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            link = ""
            a_tag = block.find("a", href=True)
            if a_tag:
                link = urljoin("https://www.benzinga.com", a_tag["href"])
            desc_tag = block.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            time_tag = block.find("time")
            pub_date = time_tag.get("datetime", "") if time_tag else ""
            if headline:
                items.append(ScrapedNewsItem(
                    headline=headline, description=desc, url=link,
                    source="Benzinga", pub_date=pub_date, scraped_at=now,
                ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_yahoo_finance(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Yahoo Finance news page."""
        items = []
        now = datetime.now().isoformat()

        # Yahoo uses li items within stream or news lists
        for li in soup.find_all("li", limit=40):
            h3 = li.find("h3")
            if not h3:
                continue
            headline = h3.get_text(strip=True)
            if len(headline) < 10:
                continue
            link = ""
            a_tag = h3.find("a", href=True) or li.find("a", href=True)
            if a_tag:
                href = a_tag["href"]
                link = urljoin("https://finance.yahoo.com", href)
            p_tag = li.find("p")
            desc = p_tag.get_text(strip=True) if p_tag else ""
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="Yahoo Finance", scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_marketwatch(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse MarketWatch latest-news page."""
        items = []
        now = datetime.now().isoformat()

        # MW uses .article__content or .element--article
        for block in soup.find_all(["div", "article"], class_=re.compile(r"article|element.*article|story"), limit=30):
            headline_tag = block.find(["h3", "h2", "a"], class_=re.compile(r"title|headline"))
            if not headline_tag:
                headline_tag = block.find(["h3", "h2"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            if len(headline) < 10:
                continue
            link = ""
            a_tag = block.find("a", href=True)
            if a_tag:
                link = urljoin("https://www.marketwatch.com", a_tag["href"])
            desc_tag = block.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            time_tag = block.find("time") or block.find("span", class_=re.compile(r"timestamp"))
            pub_date = ""
            if time_tag:
                pub_date = time_tag.get("datetime", "") or time_tag.get_text(strip=True)
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="MarketWatch", pub_date=pub_date, scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_nasdaq(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Nasdaq news-and-insights page."""
        items = []
        now = datetime.now().isoformat()

        for block in soup.find_all(["div", "article"], class_=re.compile(r"quote-news|latest-article|news-headline"), limit=30):
            headline_tag = block.find(["h3", "h2", "a", "p"], class_=re.compile(r"title|headline"))
            if not headline_tag:
                headline_tag = block.find(["h3", "h2"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            if len(headline) < 10:
                continue
            link = ""
            a_tag = block.find("a", href=True)
            if a_tag:
                link = urljoin("https://www.nasdaq.com", a_tag["href"])
            desc_tag = block.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="Nasdaq", scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    # ------------------------------------------------------------------
    # Indian Site Parsers
    # ------------------------------------------------------------------

    def _parse_moneycontrol(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Moneycontrol markets page."""
        items = []
        now = datetime.now().isoformat()

        for li in soup.find_all("li", class_=re.compile(r"clearfix"), limit=30):
            h2 = li.find("h2")
            if not h2:
                continue
            headline = h2.get_text(strip=True)
            link = ""
            a_tag = h2.find("a", href=True) or li.find("a", href=True)
            if a_tag:
                link = a_tag["href"]
            p_tag = li.find("p")
            desc = p_tag.get_text(strip=True) if p_tag else ""
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="Moneycontrol", scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_economic_times(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Economic Times stock news page."""
        items = []
        now = datetime.now().isoformat()

        for block in soup.find_all(["div", "li"], class_=re.compile(r"eachStory|story_list"), limit=30):
            headline_tag = block.find(["h3", "h4", "a"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            if len(headline) < 10:
                continue
            link = ""
            a_tag = block.find("a", href=True)
            if a_tag:
                link = urljoin("https://economictimes.indiatimes.com", a_tag["href"])
            p_tag = block.find("p")
            desc = p_tag.get_text(strip=True) if p_tag else ""
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="Economic Times", scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    def _parse_livemint(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Parse Livemint stock market news page."""
        items = []
        now = datetime.now().isoformat()

        for block in soup.find_all(["div", "article"], class_=re.compile(r"listingNew|headlineSec"), limit=30):
            headline_tag = block.find(["h2", "h3", "a"])
            if not headline_tag:
                continue
            headline = headline_tag.get_text(strip=True)
            if len(headline) < 10:
                continue
            link = ""
            a_tag = block.find("a", href=True)
            if a_tag:
                link = urljoin("https://www.livemint.com", a_tag["href"])
            desc_tag = block.find("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source="Livemint", scraped_at=now,
            ))

        if not items:
            items = self._parse_generic(soup, site)

        return items

    # ------------------------------------------------------------------
    # Generic fallback parser
    # ------------------------------------------------------------------

    def _parse_generic(self, soup: BeautifulSoup, site: ScraperSiteConfig) -> List[ScrapedNewsItem]:
        """Generic parser — extracts headlines from any page using common patterns."""
        items = []
        now = datetime.now().isoformat()
        seen_headlines = set()

        # Strategy: find all heading tags with links
        for tag in soup.find_all(["h2", "h3", "h4"], limit=50):
            headline = tag.get_text(strip=True)
            if len(headline) < 15 or headline in seen_headlines:
                continue
            seen_headlines.add(headline)

            link = ""
            a_tag = tag.find("a", href=True)
            if not a_tag:
                a_tag = tag.find_parent("a", href=True)
            if a_tag:
                href = a_tag["href"]
                if href.startswith("/"):
                    link = urljoin(site.url, href)
                elif href.startswith("http"):
                    link = href

            # Try to find a nearby description paragraph
            desc = ""
            next_p = tag.find_next_sibling("p")
            if next_p:
                desc = next_p.get_text(strip=True)[:300]

            items.append(ScrapedNewsItem(
                headline=headline, description=desc, url=link,
                source=site.name.replace("_", " ").title(), scraped_at=now,
            ))

        return items
