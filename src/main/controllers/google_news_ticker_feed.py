"""Google News per-ticker RSS feed fetcher for LCF.

Fetches aggregated news for a SPECIFIC stock symbol using Google News RSS search.
URL pattern:
    https://news.google.com/rss/search?q={query}+stock&hl={lang}&gl={country}&ceid={ceid}

This runs IN PARALLEL with cache reads to enrich agent data with the latest
ticker-specific headlines from across all news sources.

Supports both Indian and US markets via different query templates.

Usage:
    fetcher = GoogleNewsTickerFeed(market="US")
    items = fetcher.fetch_for_symbol("NVDA")
    items = fetcher.fetch_for_symbols(["NVDA", "TSLA", "AAPL"])
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus

import requests

try:
    from ...utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    logger = logging.getLogger(__name__)


# Reverse map: ticker → company search name
# For tickers where the symbol alone won't match well in Google News
US_TICKER_NAMES: Dict[str, str] = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet Google",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "NVDA": "Nvidia",
    "TSLA": "Tesla",
    "NFLX": "Netflix",
    "AMD": "AMD",
    "INTC": "Intel",
    "CRM": "Salesforce",
    "ORCL": "Oracle",
    "ADBE": "Adobe",
    "PYPL": "PayPal",
    "SQ": "Block Square",
    "SHOP": "Shopify",
    "SNOW": "Snowflake",
    "PLTR": "Palantir",
    "COIN": "Coinbase",
    "UBER": "Uber",
    "ABNB": "Airbnb",
    "AVGO": "Broadcom",
    "QCOM": "Qualcomm",
    "JPM": "JPMorgan",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "WFC": "Wells Fargo",
    "V": "Visa",
    "MA": "Mastercard",
    "JNJ": "Johnson Johnson",
    "UNH": "UnitedHealth",
    "PFE": "Pfizer",
    "ABBV": "AbbVie",
    "MRK": "Merck",
    "LLY": "Eli Lilly",
    "MRNA": "Moderna",
    "XOM": "Exxon Mobil",
    "CVX": "Chevron",
    "WMT": "Walmart",
    "COST": "Costco",
    "HD": "Home Depot",
    "NKE": "Nike",
    "SBUX": "Starbucks",
    "MCD": "McDonalds",
    "KO": "Coca Cola",
    "PEP": "PepsiCo",
    "PG": "Procter Gamble",
    "DIS": "Disney",
    "BA": "Boeing",
    "CAT": "Caterpillar",
    "GE": "General Electric",
    "RIVN": "Rivian",
    "LCID": "Lucid Motors",
    "NIO": "NIO",
    "GME": "GameStop",
    "AMC": "AMC Entertainment",
    "SOFI": "SoFi",
    "HOOD": "Robinhood",
    "SMCI": "Super Micro Computer",
    "CRWD": "CrowdStrike",
    "DDOG": "Datadog",
    "NET": "Cloudflare",
    "ARM": "ARM Holdings",
    "PLUG": "Plug Power",
}

IND_TICKER_NAMES: Dict[str, str] = {
    "RELIANCE": "Reliance Industries",
    "TCS": "TCS Tata Consultancy",
    "INFY": "Infosys",
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank India SBI",
    "BHARTIARTL": "Bharti Airtel",
    "HINDUNILVR": "Hindustan Unilever HUL",
    "ITC": "ITC Limited",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "LT": "Larsen Toubro",
    "BAJFINANCE": "Bajaj Finance",
    "MARUTI": "Maruti Suzuki",
    "SUNPHARMA": "Sun Pharma",
    "TITAN": "Titan Company",
    "WIPRO": "Wipro",
    "HCLTECH": "HCL Tech",
    "NTPC": "NTPC",
    "ADANIENT": "Adani Enterprises",
    "TATAMOTORS": "Tata Motors",
    "TATASTEEL": "Tata Steel",
    "CIPLA": "Cipla",
    "BEL": "Bharat Electronics BEL",
    "BHEL": "Bharat Heavy Electricals BHEL",
    "SUZLON": "Suzlon Energy",
    "EXIDEIND": "Exide Industries",
    "NUCLEUS": "Nucleus Software",
    "PINELABS": "Pine Labs",
    "THYROCARE": "Thyrocare Technologies",
}


@dataclass
class GoogleNewsFeedConfig:
    """Configuration for Google News per-ticker feed."""
    market: str = "US"          # "US" or "IND"
    max_items_per_symbol: int = 10
    timeout_seconds: int = 15
    max_parallel: int = 5       # Max parallel requests
    headers: Dict[str, str] = None

    def __post_init__(self):
        if self.headers is None:
            self.headers = {
                "User-Agent": "LCF-StockTracker/1.0 (research)",
                "Accept": "application/rss+xml, application/xml, text/xml",
            }

    @property
    def lang(self) -> str:
        return "en-US" if self.market == "US" else "en-IN"

    @property
    def country(self) -> str:
        return "US" if self.market == "US" else "IN"

    @property
    def ceid(self) -> str:
        return "US:en" if self.market == "US" else "IN:en"

    @property
    def ticker_names(self) -> Dict[str, str]:
        return US_TICKER_NAMES if self.market == "US" else IND_TICKER_NAMES


class GoogleNewsTickerFeed:
    """Fetches ticker-specific news from Google News RSS.

    For each symbol, constructs a Google News search RSS URL like:
        https://news.google.com/rss/search?q=Nvidia+stock&hl=en-US&gl=US&ceid=US:en

    Designed to run in parallel with cache reads to maximize data freshness.
    """

    def __init__(self, config: Optional[GoogleNewsFeedConfig] = None):
        self._config = config or GoogleNewsFeedConfig()

    def _build_url(self, symbol: str) -> str:
        """Build Google News RSS URL for a ticker."""
        # Use company name if available, otherwise use ticker itself
        name = self._config.ticker_names.get(symbol, symbol)
        query = quote_plus(f"{name} stock")
        return (
            f"https://news.google.com/rss/search?"
            f"q={query}&hl={self._config.lang}"
            f"&gl={self._config.country}&ceid={self._config.ceid}"
        )

    def fetch_for_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch Google News items for a single symbol.

        Returns list of parsed news item dicts ready for NewsItem creation.
        """
        url = self._build_url(symbol)
        try:
            resp = requests.get(
                url,
                headers=self._config.headers,
                timeout=self._config.timeout_seconds,
            )
            resp.raise_for_status()
            return self._parse_feed(resp.text, symbol, url)
        except Exception as e:
            logger.debug(f"Google News fetch failed for {symbol}: {e}")
            return []

    def fetch_for_symbols(self, symbols: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Fetch Google News for multiple symbols in parallel.

        Returns dict mapping symbol -> list of news items.
        """
        results: Dict[str, List[Dict[str, Any]]] = {}

        with ThreadPoolExecutor(max_workers=self._config.max_parallel) as executor:
            futures = {
                executor.submit(self.fetch_for_symbol, sym): sym
                for sym in symbols
            }
            for future in as_completed(futures):
                sym = futures[future]
                try:
                    items = future.result()
                    if items:
                        results[sym] = items
                except Exception as e:
                    logger.debug(f"Google News parallel fetch error for {sym}: {e}")

        total = sum(len(v) for v in results.values())
        logger.info(
            f"GoogleNewsTickerFeed: {total} items for {len(results)}/{len(symbols)} symbols"
        )
        return results

    def _parse_feed(self, xml_text: str, symbol: str, url: str) -> List[Dict[str, Any]]:
        """Parse RSS XML into news item dicts."""
        items = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

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
                "source": "Google News",
                "text": f"{title} {description}",
                "symbol": symbol,
                "feed_url": url,
            })

            if len(items) >= self._config.max_items_per_symbol:
                break

        return items
