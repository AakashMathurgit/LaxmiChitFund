"""Data Context for LCF Pipeline.

DataContext holds all input data collected from various sources,
ready to be passed to agents for processing.

Two main containers:
  - DataContext      : batch / multi-symbol container (legacy, kept for backward compat)
  - StockDataContext : per-stock container (new, used by DataProcessorAgent pattern)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import hashlib
import json


# ---------------------------------------------------------------------------
# Sub-item dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NewsItem:
    """Single news article for sentiment / event analysis."""
    symbol: str
    date: str
    headline: Optional[str] = None
    news_text: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None     # "positive", "negative", "neutral" (FinBERT)
    sentiment_acceleration: Optional[float] = None
    news_volume_spike: Optional[float] = None
    narrative_strength: Optional[float] = None
    narrative_type: Optional[str] = None
    entities: Optional[List[str]] = None      # Extracted entity names (spaCy/regex)


@dataclass
class PriceData:
    """Single OHLCV bar for technical analysis."""
    symbol: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None


@dataclass
class FundamentalData:
    """Fundamental data for a symbol."""
    symbol: str
    # Valuation
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    price_to_book: Optional[float] = None
    # Earnings
    eps: Optional[float] = None
    book_value: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_surprise: Optional[float] = None
    return_on_equity: Optional[float] = None
    # Margins
    profit_margin: Optional[float] = None
    # Debt
    debt_to_equity: Optional[float] = None
    # Dividends
    dividend_yield: Optional[float] = None
    # Size
    market_cap: Optional[float] = None
    # Identity
    sector: Optional[str] = None
    industry: Optional[str] = None
    # Analyst
    analyst_rating: Optional[str] = None
    price_target: Optional[float] = None


@dataclass
class EventData:
    """A single corporate event or catalyst."""
    symbol: str
    date: str
    event_type: str
    description: str
    impact_score: Optional[float] = None
    source: Optional[str] = None
    # Upcoming earnings
    earnings_date: Optional[str] = None
    # Splits / dividends / bonuses from ticker.actions
    recent_corporate_actions: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# IndexData — fetched once per run, shared across all stocks
# ---------------------------------------------------------------------------

@dataclass
class IndexData:
    """Market index data (e.g. NIFTY 50) shared across all StockDataContexts."""
    index_symbol: str
    historical_ohlc: List[PriceData] = field(default_factory=list)
    last_close: Optional[float] = None
    last_volume: Optional[int] = None
    last_trading_date: Optional[str] = None

    def get_historical_closes(self) -> List[float]:
        return [p.close for p in self.historical_ohlc]


# ---------------------------------------------------------------------------
# StockDataContext — one per stock, built by DataProcessorAgent
# ---------------------------------------------------------------------------

@dataclass
class StockDataContext:
    """Per-stock data context passed to all agents.

    Built once by DataProcessorAgent and handed to:
      FundamentalAgent, TechnicalAgent, SentimentAgent, EventAgent.

    Each agent reads only the fields it needs.

    Raw data fields are immutable after creation.
    Computed results go into computed_metrics / flags.
    """

    # --- Identity ---
    symbol: str
    exchange: str = ""
    company_name: str = ""
    sector: str = ""
    industry: str = ""
    isin: Optional[str] = None
    currency: str = "INR"

    # --- Price snapshot (latest bar) ---
    last_close: Optional[float] = None
    previous_close: Optional[float] = None
    last_open: Optional[float] = None
    last_high: Optional[float] = None
    last_low: Optional[float] = None
    last_volume: Optional[int] = None
    last_trading_date: Optional[str] = None

    # --- Historical OHLCV (400-500 bars) ---
    historical_ohlc: List[PriceData] = field(default_factory=list)

    # --- Market index (shared, injected from outside) ---
    index_data: Optional[IndexData] = None

    # --- Fundamentals ---
    fundamentals: Optional[FundamentalData] = None

    # --- News (last 48-72 hours) ---
    news_items: List[NewsItem] = field(default_factory=list)

    # --- Corporate events ---
    event_data: List[EventData] = field(default_factory=list)

    # --- Computed layer (agents / processor may write here) ---
    computed_metrics: Dict[str, Any] = field(default_factory=dict)
    flags: Dict[str, Any] = field(default_factory=dict)

    # --- Metadata ---
    created_at: datetime = field(default_factory=datetime.now)

    # --- Convenience accessors ---

    def get_historical_closes(self) -> List[float]:
        return [p.close for p in self.historical_ohlc]

    def get_historical_volumes(self) -> List[int]:
        return [p.volume for p in self.historical_ohlc]

    def get_historical_highs(self) -> List[float]:
        return [p.high for p in self.historical_ohlc]

    def get_historical_lows(self) -> List[float]:
        return [p.low for p in self.historical_ohlc]

    def get_historical_opens(self) -> List[float]:
        return [p.open for p in self.historical_ohlc]

    def summary(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "company_name": self.company_name,
            "exchange": self.exchange,
            "sector": self.sector,
            "last_close": self.last_close,
            "last_trading_date": self.last_trading_date,
            "historical_bars": len(self.historical_ohlc),
            "news_count": len(self.news_items),
            "has_fundamentals": self.fundamentals is not None,
            "has_index_data": self.index_data is not None,
            "events_count": len(self.event_data),
        }


# ---------------------------------------------------------------------------
# DataContext — legacy batch container (kept for backward compatibility)
# ---------------------------------------------------------------------------

@dataclass
class DataContext:
    """Aggregated data context passed to agents.

    Legacy multi-symbol batch container.
    Prefer StockDataContext for new agent code.
    """
    request_id: str
    symbols: List[str]
    timestamp: datetime = field(default_factory=datetime.now)

    news_items: List[NewsItem] = field(default_factory=list)
    price_data: List[PriceData] = field(default_factory=list)
    fundamental_data: List[FundamentalData] = field(default_factory=list)
    event_data: List[EventData] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_sources: Dict[str, Any] = field(default_factory=dict)

    def get_news_for_symbol(self, symbol: str) -> List[NewsItem]:
        return [n for n in self.news_items if n.symbol.upper() == symbol.upper()]

    def get_prices_for_symbol(self, symbol: str) -> List[PriceData]:
        return [p for p in self.price_data if p.symbol.upper() == symbol.upper()]

    def get_fundamentals_for_symbol(self, symbol: str) -> Optional[FundamentalData]:
        for f in self.fundamental_data:
            if f.symbol.upper() == symbol.upper():
                return f
        return None

    def get_events_for_symbol(self, symbol: str) -> List[EventData]:
        return [e for e in self.event_data if e.symbol.upper() == symbol.upper()]

    def compute_checksum(self) -> str:
        content = json.dumps({
            "request_id": self.request_id,
            "symbols": sorted(self.symbols),
            "news_count": len(self.news_items),
            "price_count": len(self.price_data),
            "fundamental_count": len(self.fundamental_data),
            "event_count": len(self.event_data),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def summary(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "symbols": self.symbols,
            "timestamp": self.timestamp.isoformat(),
            "news_items": len(self.news_items),
            "price_records": len(self.price_data),
            "fundamental_records": len(self.fundamental_data),
            "event_records": len(self.event_data),
            "checksum": self.compute_checksum(),
        }
