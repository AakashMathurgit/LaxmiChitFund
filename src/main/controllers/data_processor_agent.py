"""DataProcessorAgent — Central data orchestration layer.

DataProcessorAgent is responsible for:
1. Managing a list of DataProviders
2. Fetching data from all providers for given symbols
3. Building unified StockDataContext objects
4. Caching shared data (like IndexData) once per run

This is the single entry point for all data fetching in the pipeline.
Agents receive StockDataContext objects and extract only the fields they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import logging

from .data_provider import DataProvider, DataProviderResult, DataType
from .data_context import (
    StockDataContext,
    IndexData,
    PriceData,
    FundamentalData,
    NewsItem,
    EventData,
)

logger = logging.getLogger(__name__)


@dataclass
class DataProcessorAgentConfig:
    """Configuration for DataProcessorAgent.
    
    Attributes:
        index_symbol: Market index symbol to fetch once per run (e.g., "^NSEI" for NIFTY 50)
        historical_days: Number of historical days to fetch (400-500 recommended)
        news_lookback_hours: Hours of news to fetch (48-72 recommended)
        fetch_index: Whether to fetch market index data
        default_exchange: Default exchange for symbols
        default_currency: Default currency
    """
    index_symbol: str = "^NSEI"  # NIFTY 50
    historical_days: int = 500
    news_lookback_hours: int = 72
    fetch_index: bool = True
    default_exchange: str = "NSE"
    default_currency: str = "INR"


class DataProcessorAgent:
    """Central agent for data collection and context building.
    
    DataProcessorAgent orchestrates multiple DataProviders to build
    complete StockDataContext objects for each stock in the analysis.
    
    Key responsibilities:
    1. Manage registered DataProviders
    2. Fetch index data ONCE per run (shared across all stocks)
    3. For each stock: fetch all data and build StockDataContext
    4. Clean, normalize, and validate data
    5. Keep intelligence in agents, not here (no scoring)
    
    Usage:
        agent = DataProcessorAgent()
        agent.register_provider(YahooFinanceProvider(config))
        agent.register_provider(NewsDataProvider(config))
        
        # Build context for multiple stocks
        contexts = agent.build_contexts(["TCS.NS", "INFY.NS", "RELIANCE.NS"])
        
        # Or build for single stock
        context = agent.build_context("MSFT")
    
    Flow:
        DataProcessorAgent
            ├── YahooFinanceProvider  → prices, fundamentals, events
            ├── NewsDataProvider      → news articles
            └── (future providers)
                    ↓
            StockDataContext (per stock)
                    ↓
            [FundamentalAgent, TechnicalAgent, SentimentAgent, EventAgent]
    """
    
    def __init__(self, config: Optional[DataProcessorAgentConfig] = None):
        """Initialize the DataProcessorAgent.
        
        Args:
            config: Configuration options. Uses defaults if not provided.
        """
        self.config = config or DataProcessorAgentConfig()
        self._providers: List[DataProvider] = []
        self._index_cache: Optional[IndexData] = None
        self._run_timestamp: Optional[datetime] = None
        logger.info("DataProcessorAgent initialized")
    
    # -------------------------------------------------------------------------
    # Provider Management
    # -------------------------------------------------------------------------
    
    def register_provider(self, provider: DataProvider) -> None:
        """Register a data provider.
        
        Providers are called in registration order.
        
        Args:
            provider: DataProvider instance to register
        """
        self._providers.append(provider)
        logger.info(f"Registered provider: {provider.name}")
    
    def unregister_provider(self, provider_name: str) -> bool:
        """Unregister a provider by name."""
        for i, p in enumerate(self._providers):
            if p.name == provider_name:
                self._providers.pop(i)
                logger.info(f"Unregistered provider: {provider_name}")
                return True
        return False
    
    def get_provider(self, name: str) -> Optional[DataProvider]:
        """Get a registered provider by name."""
        for p in self._providers:
            if p.name == name:
                return p
        return None
    
    @property
    def providers(self) -> List[DataProvider]:
        """Get list of registered providers."""
        return self._providers.copy()
    
    # -------------------------------------------------------------------------
    # Index Data (Fetched Once Per Run)
    # -------------------------------------------------------------------------
    
    def _fetch_index_data(self) -> Optional[IndexData]:
        """Fetch market index data once per run.
        
        This is cached and shared across all StockDataContext objects.
        """
        if not self.config.fetch_index:
            return None
        
        # Find a provider that supports OHLCV
        for provider in self._providers:
            if DataType.OHLCV in provider.supported_types:
                try:
                    result = provider.fetch(
                        symbols=[self.config.index_symbol],
                        data_types={DataType.OHLCV}
                    )
                    
                    ohlcv_data = result.get_data(DataType.OHLCV)
                    if not ohlcv_data:
                        continue
                    
                    # Convert to PriceData objects
                    historical = []
                    for item in ohlcv_data:
                        historical.append(PriceData(
                            symbol=self.config.index_symbol,
                            date=item.get("date", ""),
                            open=float(item.get("open", 0)),
                            high=float(item.get("high", 0)),
                            low=float(item.get("low", 0)),
                            close=float(item.get("close", 0)),
                            volume=int(item.get("volume", 0)),
                        ))
                    
                    # Get latest values
                    last_bar = historical[-1] if historical else None
                    
                    index_data = IndexData(
                        index_symbol=self.config.index_symbol,
                        historical_ohlc=historical,
                        last_close=last_bar.close if last_bar else None,
                        last_volume=last_bar.volume if last_bar else None,
                        last_trading_date=last_bar.date if last_bar else None,
                    )
                    
                    logger.info(f"Fetched index data: {self.config.index_symbol} ({len(historical)} bars)")
                    return index_data
                    
                except Exception as e:
                    logger.error(f"Error fetching index from {provider.name}: {e}")
        
        logger.warning(f"Could not fetch index data for {self.config.index_symbol}")
        return None
    
    # -------------------------------------------------------------------------
    # Context Building
    # -------------------------------------------------------------------------
    
    def build_contexts(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
    ) -> List[StockDataContext]:
        """Build StockDataContext for multiple symbols.
        
        This is the main entry point for batch processing.
        Index data is fetched once and shared across all contexts.
        
        Args:
            symbols: List of stock symbols to process
            data_types: Optional set of data types to fetch
        
        Returns:
            List of StockDataContext objects, one per symbol
        """
        self._run_timestamp = datetime.now()
        
        # Fetch index data once
        if self.config.fetch_index and self._index_cache is None:
            self._index_cache = self._fetch_index_data()
        
        # Build context for each symbol
        contexts = []
        for symbol in symbols:
            try:
                ctx = self.build_context(symbol, data_types)
                contexts.append(ctx)
            except Exception as e:
                logger.error(f"Error building context for {symbol}: {e}")
                # Create minimal context with error flag
                ctx = StockDataContext(symbol=symbol)
                ctx.flags["error"] = str(e)
                contexts.append(ctx)
        
        logger.info(f"Built {len(contexts)} StockDataContexts")
        return contexts
    
    def build_context(
        self,
        symbol: str,
        data_types: Optional[Set[DataType]] = None,
    ) -> StockDataContext:
        """Build StockDataContext for a single symbol.
        
        Fetches data from all registered providers and assembles
        a complete StockDataContext.
        
        Args:
            symbol: Stock symbol (e.g., "TCS.NS", "MSFT")
            data_types: Optional set of data types to fetch
        
        Returns:
            Complete StockDataContext for the symbol
        """
        logger.info(f"Building context for {symbol}")
        
        # Default data types
        if data_types is None:
            data_types = {
                DataType.OHLCV,
                DataType.FUNDAMENTALS,
                DataType.NEWS,
                DataType.EVENTS,
                DataType.ANALYST,
            }
        
        # Collect data from all providers
        all_data: Dict[DataType, List[Any]] = {dt: [] for dt in data_types}
        
        for provider in self._providers:
            provider_types = data_types & provider.supported_types
            if not provider_types:
                continue
            
            try:
                result = provider.fetch([symbol], provider_types)
                for dt, items in result.data.items():
                    all_data[dt].extend(items)
                    
            except Exception as e:
                logger.error(f"Error fetching from {provider.name} for {symbol}: {e}")
        
        # Build the context
        context = self._assemble_context(symbol, all_data)
        
        # Inject shared index data
        if self._index_cache:
            context.index_data = self._index_cache
        
        logger.debug(f"Context built: {context.summary()}")
        return context
    
    def _assemble_context(
        self,
        symbol: str,
        data: Dict[DataType, List[Any]]
    ) -> StockDataContext:
        """Assemble StockDataContext from raw provider data.
        
        This method:
        1. Creates the context with identity fields
        2. Populates historical OHLCV
        3. Populates snapshot fields (last_close, etc.)
        4. Populates fundamentals
        5. Populates news items
        6. Populates event data
        """
        context = StockDataContext(
            symbol=symbol,
            exchange=self.config.default_exchange,
            currency=self.config.default_currency,
        )
        
        # Process OHLCV data
        ohlcv_items = data.get(DataType.OHLCV, [])
        self._populate_price_data(context, ohlcv_items)
        
        # Process fundamentals
        fundamental_items = data.get(DataType.FUNDAMENTALS, [])
        self._populate_fundamentals(context, fundamental_items)
        
        # Process news
        news_items = data.get(DataType.NEWS, [])
        self._populate_news(context, news_items)
        
        # Process events
        event_items = data.get(DataType.EVENTS, [])
        self._populate_events(context, event_items)
        
        return context
    
    def _populate_price_data(
        self, 
        context: StockDataContext, 
        items: List[Dict[str, Any]]
    ) -> None:
        """Populate historical OHLCV and snapshot fields."""
        if not items:
            return
        
        # Sort by date
        sorted_items = sorted(items, key=lambda x: x.get("date", ""))
        
        # Build historical list
        for item in sorted_items:
            price = PriceData(
                symbol=context.symbol,
                date=item.get("date", ""),
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                adjusted_close=item.get("adj_close") or item.get("adjusted_close"),
            )
            context.historical_ohlc.append(price)
        
        # Populate snapshot fields from latest bar
        if context.historical_ohlc:
            latest = context.historical_ohlc[-1]
            context.last_close = latest.close
            context.last_open = latest.open
            context.last_high = latest.high
            context.last_low = latest.low
            context.last_volume = latest.volume
            context.last_trading_date = latest.date
            
            # Previous close from second-to-last bar
            if len(context.historical_ohlc) > 1:
                context.previous_close = context.historical_ohlc[-2].close
    
    def _populate_fundamentals(
        self, 
        context: StockDataContext, 
        items: List[Dict[str, Any]]
    ) -> None:
        """Populate fundamentals data."""
        if not items:
            return
        
        # Use first item (should only be one per symbol)
        item = items[0]
        
        context.fundamentals = FundamentalData(
            symbol=context.symbol,
            # Valuation
            pe_ratio=item.get("pe_ratio") or item.get("trailing_pe"),
            forward_pe=item.get("forward_pe"),
            price_to_book=item.get("price_to_book"),
            # Earnings
            eps=item.get("eps") or item.get("trailing_eps"),
            book_value=item.get("book_value"),
            revenue_growth_yoy=item.get("revenue_growth") or item.get("revenue_growth_yoy"),
            earnings_surprise=item.get("earnings_surprise") or item.get("earnings_growth"),
            return_on_equity=item.get("return_on_equity"),
            # Margins
            profit_margin=item.get("profit_margin"),
            # Debt
            debt_to_equity=item.get("debt_to_equity"),
            # Dividends
            dividend_yield=item.get("dividend_yield"),
            # Size
            market_cap=item.get("market_cap"),
            # Identity
            sector=item.get("sector"),
            industry=item.get("industry"),
            # Analyst
            analyst_rating=item.get("recommendation_key") or item.get("analyst_rating"),
            price_target=item.get("target_mean_price") or item.get("price_target"),
        )
        
        # Also update context identity fields from fundamentals
        if item.get("company_name"):
            context.company_name = item["company_name"]
        if item.get("sector"):
            context.sector = item["sector"]
        if item.get("industry"):
            context.industry = item["industry"]
    
    def _populate_news(
        self, 
        context: StockDataContext, 
        items: List[Dict[str, Any]]
    ) -> None:
        """Populate news items."""
        for item in items:
            # Handle different date formats
            date_str = item.get("date", "")
            if not date_str and item.get("publish_time"):
                try:
                    date_str = datetime.fromtimestamp(item["publish_time"]).strftime("%Y-%m-%d")
                except Exception:
                    pass
            
            news = NewsItem(
                symbol=context.symbol,
                date=date_str,
                headline=item.get("title") or item.get("headline"),
                news_text=item.get("summary") or item.get("news_text"),
                source=item.get("publisher") or item.get("source"),
                url=item.get("link") or item.get("url"),
            )
            context.news_items.append(news)
    
    def _populate_events(
        self, 
        context: StockDataContext, 
        items: List[Dict[str, Any]]
    ) -> None:
        """Populate event data."""
        for item in items:
            event = EventData(
                symbol=context.symbol,
                date=item.get("date", ""),
                event_type=item.get("event_type", "unknown"),
                description=item.get("description", ""),
                impact_score=item.get("impact_score"),
                source=item.get("source"),
                earnings_date=item.get("earnings_date"),
                recent_corporate_actions=item.get("corporate_actions", []),
            )
            context.event_data.append(event)
    
    # -------------------------------------------------------------------------
    # Run Management
    # -------------------------------------------------------------------------
    
    def start_run(self) -> None:
        """Start a new processing run. Clears cached data."""
        self._index_cache = None
        self._run_timestamp = datetime.now()
        logger.info(f"Started new run at {self._run_timestamp}")
    
    def end_run(self) -> None:
        """End the current run. Clears cached data."""
        self._index_cache = None
        logger.info("Run ended, cache cleared")
    
    @property
    def index_data(self) -> Optional[IndexData]:
        """Get cached index data for current run."""
        return self._index_cache
