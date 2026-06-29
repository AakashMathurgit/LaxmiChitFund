"""Data Processor Controller for LCF.

DataProcessor is the central orchestrator that:
1. Manages registered DataProviders
2. Fetches data from all providers for given symbols
3. Aggregates results and creates a unified DataContext
4. Provides a clean interface for the agent pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
from datetime import datetime
import uuid

from .data_context import (
    DataContext,
    NewsItem,
    PriceData,
    FundamentalData,
    EventData,
)
from .data_provider import (
    DataProvider,
    DataProviderResult,
    DataType,
    CompositeDataProvider,
)
from ...utils.logger import get_logger

if TYPE_CHECKING:
    from .data_context import IndexData, StockDataContext

logger = get_logger(__name__)


@dataclass
class DataProcessorConfig:
    """Configuration for DataProcessor.
    
    Attributes:
        default_data_types: Default data types to fetch if not specified
        aggregate_strategy: How to handle duplicate data from multiple providers
                          ("first" = first provider wins, "merge" = combine all)
        validate_data: Whether to validate fetched data
        cache_results: Whether to cache provider results
    """
    default_data_types: Set[DataType] = field(default_factory=lambda: {
        DataType.OHLCV,
        DataType.FUNDAMENTALS,
        DataType.NEWS,
        DataType.EVENTS,
    })
    aggregate_strategy: str = "first"  # "first" or "merge"
    validate_data: bool = True
    cache_results: bool = False
    cache_ttl_seconds: int = 300  # 5 minutes


class DataProcessor:
    """Central controller for fetching and aggregating financial data.
    
    DataProcessor orchestrates multiple DataProviders to fetch data for given
    stock symbols. It aggregates results from all providers and creates a 
    unified DataContext that can be passed to agents in the pipeline.
    
    Key responsibilities:
    1. Manage registered DataProviders
    2. Coordinate data fetching across providers
    3. Aggregate and deduplicate results
    4. Convert raw data to typed DataContext objects
    5. Validate and enrich the data
    
    Usage:
        # Create processor and register providers
        processor = DataProcessor()
        processor.register_provider(YahooFinanceProvider())
        processor.register_provider(FileDataProvider(config))
        
        # Fetch data and get context
        context = processor.process(["AAPL", "MSFT"])
        
        # Access typed data
        prices = context.get_prices_for_symbol("AAPL")
        news = context.get_news_for_symbol("AAPL")
    
    The processor supports multiple provider registration strategies:
    - First provider wins (default): Data from higher-priority provider used
    - Merge: Combine data from all providers (may have duplicates)
    """
    
    def __init__(self, config: Optional[DataProcessorConfig] = None):
        """Initialize the data processor.
        
        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or DataProcessorConfig()
        self._providers: List[DataProvider] = []
        self._result_cache: Dict[str, DataProviderResult] = {}
        logger.info("DataProcessor initialized")
    
    def register_provider(self, provider: DataProvider) -> None:
        """Register a data provider.
        
        Providers are called in registration order. For "first wins" strategy,
        register higher-priority providers first.
        
        Args:
            provider: DataProvider instance to register
        """
        self._providers.append(provider)
        logger.info(f"Registered provider: {provider.name} (supports: {[t.name for t in provider.supported_types]})")
    
    def unregister_provider(self, provider_name: str) -> bool:
        """Unregister a provider by name.
        
        Returns:
            True if provider was found and removed, False otherwise.
        """
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
    
    @property
    def available_data_types(self) -> Set[DataType]:
        """Get union of all supported data types from registered providers."""
        types: Set[DataType] = set()
        for provider in self._providers:
            types.update(provider.supported_types)
        return types
    
    def process(
        self,
        symbols: List[str],
        request_id: Optional[str] = None,
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataContext:
        """Fetch data from all providers and create DataContext.
        
        This is the main entry point for the data processor. It:
        1. Calls each registered provider to fetch data
        2. Aggregates results based on strategy
        3. Converts raw data to typed objects
        4. Creates and returns a DataContext
        
        Args:
            symbols: List of stock symbols to fetch data for
            request_id: Optional request ID for tracing (auto-generated if not provided)
            data_types: Set of DataTypes to fetch. If None, uses default types.
            **kwargs: Additional parameters passed to providers (e.g., period, interval)
        
        Returns:
            DataContext containing all fetched data with typed objects
        """
        if request_id is None:
            request_id = str(uuid.uuid4())[:8]
        
        # Normalize symbols
        symbols = [s.upper().strip() for s in symbols if s and s.strip()]
        if not symbols:
            logger.warning("No valid symbols provided")
            return DataContext(request_id=request_id, symbols=[], timestamp=datetime.now())
        
        requested_types = data_types or self.config.default_data_types
        
        logger.info(f"Processing data for symbols: {symbols} (request_id: {request_id})")
        logger.debug(f"Requested data types: {[t.name for t in requested_types]}")
        
        # Fetch from all providers
        aggregated_data = self._fetch_from_providers(symbols, requested_types, **kwargs)
        
        # Create and populate context
        context = DataContext(
            request_id=request_id,
            symbols=symbols,
            timestamp=datetime.now(),
        )
        
        # Convert aggregated data to typed objects
        self._populate_context(context, aggregated_data)
        
        # Validate if enabled
        if self.config.validate_data:
            warnings = self.validate_context(context)
            if warnings:
                logger.warning(f"Data validation warnings: {warnings}")
        
        logger.info(f"DataContext created: {context.summary()}")
        return context
    
    def _fetch_from_providers(
        self, 
        symbols: List[str],
        data_types: Set[DataType],
        **kwargs
    ) -> Dict[DataType, List[Any]]:
        """Fetch data from all registered providers and aggregate.
        
        Returns:
            Dictionary mapping DataType to list of raw data items
        """
        aggregated: Dict[DataType, List[Any]] = {dt: [] for dt in data_types}
        seen_types: Set[DataType] = set()  # Track which types we've populated
        
        for provider in self._providers:
            # Determine which types this provider should fetch
            provider_types = data_types & provider.supported_types
            
            # For "first wins", skip types we already have
            if self.config.aggregate_strategy == "first":
                provider_types = provider_types - seen_types
            
            if not provider_types:
                continue
            
            try:
                logger.debug(f"Fetching from {provider.name}: {[t.name for t in provider_types]}")
                result = provider.fetch(symbols, provider_types, **kwargs)
                
                # Store raw sources for debugging
                for data_type, items in result.data.items():
                    if items:
                        if self.config.aggregate_strategy == "first" and data_type in seen_types:
                            continue
                        aggregated[data_type].extend(items)
                        seen_types.add(data_type)
                
                # Log any errors
                for error in result.errors:
                    logger.warning(f"Provider {provider.name} error: {error}")
                    
            except Exception as e:
                logger.error(f"Error fetching from provider {provider.name}: {e}")
        
        return aggregated
    
    def _populate_context(
        self, 
        context: DataContext, 
        aggregated_data: Dict[DataType, List[Any]]
    ) -> None:
        """Convert raw aggregated data to typed objects and populate context."""
        
        # Populate OHLCV data
        for item in aggregated_data.get(DataType.OHLCV, []):
            price = PriceData(
                symbol=item.get("symbol", ""),
                date=item.get("date", ""),
                open=float(item.get("open", 0)),
                high=float(item.get("high", 0)),
                low=float(item.get("low", 0)),
                close=float(item.get("close", 0)),
                volume=int(item.get("volume", 0)),
                adjusted_close=item.get("adjusted_close") or item.get("adj_close"),
            )
            context.price_data.append(price)
        
        # Populate fundamentals data
        for item in aggregated_data.get(DataType.FUNDAMENTALS, []):
            fundamental = FundamentalData(
                symbol=item.get("symbol", ""),
                pe_ratio=item.get("pe_ratio") or item.get("trailing_pe"),
                eps=item.get("eps") or item.get("trailing_eps"),
                revenue_growth_yoy=item.get("revenue_growth_yoy") or item.get("revenue_growth"),
                earnings_surprise=item.get("earnings_surprise"),
                analyst_rating=item.get("analyst_rating") or item.get("recommendation_key"),
                price_target=item.get("price_target") or item.get("target_median_price"),
                sector=item.get("sector"),
                market_cap=item.get("market_cap"),
            )
            context.fundamental_data.append(fundamental)
        
        # Populate news data
        for item in aggregated_data.get(DataType.NEWS, []):
            # Handle different date formats
            date_str = item.get("date", "")
            if not date_str and item.get("publish_time"):
                try:
                    date_str = datetime.fromtimestamp(item["publish_time"]).strftime("%Y-%m-%d")
                except Exception:
                    pass
            
            news_item = NewsItem(
                symbol=item.get("symbol", ""),
                date=date_str,
                headline=item.get("headline") or item.get("title"),
                news_text=item.get("news_text"),
                source=item.get("source") or item.get("publisher"),
                sentiment_score=item.get("sentiment_score"),
                sentiment_acceleration=item.get("sentiment_acceleration"),
                news_volume_spike=item.get("news_volume_spike"),
                narrative_strength=item.get("narrative_strength"),
                narrative_type=item.get("narrative_type"),
            )
            context.news_items.append(news_item)
        
        # Populate events data
        for item in aggregated_data.get(DataType.EVENTS, []):
            event = EventData(
                symbol=item.get("symbol", ""),
                date=item.get("date", ""),
                event_type=item.get("event_type", ""),
                description=item.get("description", ""),
                impact_score=item.get("impact_score"),
                source=item.get("source"),
            )
            context.event_data.append(event)
        
        # Store raw data for debugging/extension
        context.raw_sources = {
            dt.name.lower(): items 
            for dt, items in aggregated_data.items() 
            if items
        }
    
    def validate_context(self, context: DataContext) -> List[str]:
        """Validate the data context for completeness.
        
        Returns:
            List of validation warnings (empty if all good)
        """
        warnings = []
        
        if not context.symbols:
            warnings.append("No symbols specified in context")
        
        # Check for missing data per symbol
        for symbol in context.symbols:
            has_prices = any(p.symbol.upper() == symbol.upper() for p in context.price_data)
            has_news = any(n.symbol.upper() == symbol.upper() for n in context.news_items)
            has_fundamentals = any(f.symbol.upper() == symbol.upper() for f in context.fundamental_data)
            
            if not has_prices and not has_news and not has_fundamentals:
                warnings.append(f"No data found for symbol: {symbol}")
        
        return warnings
    
    def clear_cache(self) -> None:
        """Clear the result cache."""
        self._result_cache.clear()
        logger.debug("Result cache cleared")

    # ------------------------------------------------------------------
    # Per-stock typed builders (new DataProcessorAgent pattern)
    # ------------------------------------------------------------------

    def build_index_context(
        self,
        index_symbol: str = "^NSEI",
        period: str = "2y",
    ) -> Optional[IndexData]:
        """Fetch market index data and return a typed IndexData object.

        Call this ONCE before your stock loop and pass the result into
        each build_stock_context() call.  Avoids redundant index fetches.

        Args:
            index_symbol: Yahoo Finance ticker for the index (default: NIFTY 50)
            period: History length, e.g. "2y"

        Returns:
            IndexData, or None if no provider supports it
        """
        for provider in self._providers:
            if hasattr(provider, "fetch_index_data"):
                try:
                    index_data = provider.fetch_index_data(
                        index_symbol=index_symbol,
                        period=period,
                    )
                    logger.info(
                        f"Index data fetched via {provider.name}: "
                        f"{index_symbol}, bars={len(index_data.historical_ohlc)}"
                    )
                    return index_data
                except Exception as e:
                    logger.error(
                        f"Error fetching index data from {provider.name}: {e}"
                    )
        logger.warning(
            "No registered provider supports fetch_index_data — "
            "index data will be unavailable"
        )
        return None

    def build_stock_context(
        self,
        symbol: str,
        index_data: Optional[IndexData] = None,
        period: str = "2y",
    ) -> StockDataContext:
        """Build a complete StockDataContext by merging data from ALL providers.

        Each provider contributes what it knows. Merge rules:
          - Scalar fields (close, pe_ratio, company_name…): first provider wins,
            later providers only fill in fields still None / empty.
          - List fields (news_items, event_data): union — all items from all
            providers are combined.
          - Nested objects (fundamentals): field-level first-wins merge.
          - historical_ohlc / index_data: first provider wins (don't override
            if already populated).

        Providers contribute via:
          - fetch_stock_context(symbol, index_data, period) → StockDataContext
          - enrich_stock_context(ctx, symbol) → None  (in-place enrichment)

        Args:
            symbol: Stock ticker without exchange suffix (e.g. "TCS")
            index_data: Shared IndexData from build_index_context()
            period: History length passed to providers

        Returns:
            Fully merged StockDataContext ready for all agents
        """
        from .data_context import StockDataContext as _StockDataContext

        base = _StockDataContext(symbol=symbol)

        for provider in self._providers:
            if hasattr(provider, "fetch_stock_context"):
                try:
                    partial = provider.fetch_stock_context(
                        symbol=symbol,
                        index_data=index_data,
                        period=period,
                    )
                    _merge_stock_contexts(base, partial)
                    logger.debug(
                        f"[{symbol}] Merged context from {provider.name}"
                    )
                except Exception as e:
                    logger.error(
                        f"[{symbol}] fetch_stock_context failed on "
                        f"{provider.name}: {e}"
                    )
            elif hasattr(provider, "enrich_stock_context"):
                try:
                    provider.enrich_stock_context(base, symbol)
                    logger.debug(
                        f"[{symbol}] Enriched context via {provider.name}"
                    )
                except Exception as e:
                    logger.error(
                        f"[{symbol}] enrich_stock_context failed on "
                        f"{provider.name}: {e}"
                    )

        logger.info(f"StockDataContext built for {symbol}: {base.summary()}")
        return base


# ---------------------------------------------------------------------------
# Merge helpers for build_stock_context
# ---------------------------------------------------------------------------

def _merge_fundamentals(base: FundamentalData, addition: FundamentalData) -> None:
    """Field-level first-wins merge: fill None fields in base from addition."""
    if base.pe_ratio is None:           base.pe_ratio = addition.pe_ratio
    if base.forward_pe is None:         base.forward_pe = addition.forward_pe
    if base.price_to_book is None:      base.price_to_book = addition.price_to_book
    if base.eps is None:                base.eps = addition.eps
    if base.book_value is None:         base.book_value = addition.book_value
    if base.revenue_growth_yoy is None: base.revenue_growth_yoy = addition.revenue_growth_yoy
    if base.earnings_surprise is None:  base.earnings_surprise = addition.earnings_surprise
    if base.return_on_equity is None:   base.return_on_equity = addition.return_on_equity
    if base.profit_margin is None:      base.profit_margin = addition.profit_margin
    if base.debt_to_equity is None:     base.debt_to_equity = addition.debt_to_equity
    if base.dividend_yield is None:     base.dividend_yield = addition.dividend_yield
    if base.market_cap is None:         base.market_cap = addition.market_cap
    if not base.sector:                 base.sector = addition.sector
    if not base.industry:               base.industry = addition.industry
    if not base.analyst_rating:         base.analyst_rating = addition.analyst_rating
    if base.price_target is None:       base.price_target = addition.price_target


def _merge_stock_contexts(base: "StockDataContext", addition: "StockDataContext") -> None:
    """Merge addition into base in-place.

    Rules:
      - Scalar / optional fields : first-wins (only fill if base value is None / empty)
      - historical_ohlc          : first-wins (don't override non-empty series)
      - index_data               : first-wins
      - fundamentals             : field-level first-wins via _merge_fundamentals
      - news_items, event_data   : union (append all from addition)
    """
    # --- Identity ---
    if not base.exchange:       base.exchange = addition.exchange
    if not base.company_name:   base.company_name = addition.company_name
    if not base.sector:         base.sector = addition.sector
    if not base.industry:       base.industry = addition.industry
    if base.isin is None:       base.isin = addition.isin
    if not base.currency or base.currency == "INR":
        base.currency = addition.currency or base.currency

    # --- Price snapshot ---
    if base.last_close is None:        base.last_close = addition.last_close
    if base.previous_close is None:    base.previous_close = addition.previous_close
    if base.last_open is None:         base.last_open = addition.last_open
    if base.last_high is None:         base.last_high = addition.last_high
    if base.last_low is None:          base.last_low = addition.last_low
    if base.last_volume is None:       base.last_volume = addition.last_volume
    if base.last_trading_date is None: base.last_trading_date = addition.last_trading_date

    # --- Historical OHLC: first-wins ---
    if not base.historical_ohlc:
        base.historical_ohlc = addition.historical_ohlc

    # --- Index data: first-wins ---
    if base.index_data is None:
        base.index_data = addition.index_data

    # --- Fundamentals: field-level first-wins ---
    if base.fundamentals is None:
        base.fundamentals = addition.fundamentals
    elif addition.fundamentals is not None:
        _merge_fundamentals(base.fundamentals, addition.fundamentals)

    # --- News: union (combine from all providers) ---
    base.news_items.extend(addition.news_items)

    # --- Events: union ---
    base.event_data.extend(addition.event_data)


# ---------------------------------------------------------------------------
# Factory functions for easy setup
# ---------------------------------------------------------------------------

def create_processor_with_yahoo_finance(
    exchange_suffix: str = ".NS",
    price_period: str = "1mo",
    price_interval: str = "1d",
) -> DataProcessor:
    """Create a DataProcessor with Yahoo Finance provider pre-configured.
    
    Args:
        exchange_suffix: Exchange suffix for symbols (.NS for NSE, .BO for BSE, "" for US)
        price_period: Default price history period
        price_interval: Default price history interval
    
    Returns:
        DataProcessor with YahooFinanceProvider registered
    """
    from .yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig
    
    processor = DataProcessor()
    
    yf_config = YFinanceConfig(
        default_exchange_suffix=exchange_suffix,
        price_period=price_period,
        price_interval=price_interval,
    )
    processor.register_provider(YahooFinanceProvider(yf_config))
    
    return processor


def create_processor_with_files(
    news_file: Optional[str] = None,
    ohlcv_file: Optional[str] = None,
    fundamentals_file: Optional[str] = None,
    events_file: Optional[str] = None,
    base_dir: Optional[str] = None,
) -> DataProcessor:
    """Create a DataProcessor with file-based provider pre-configured.
    
    Args:
        news_file: Path to news JSON/CSV file
        ohlcv_file: Path to price data file
        fundamentals_file: Path to fundamentals file
        events_file: Path to events file
        base_dir: Base directory for relative paths
    
    Returns:
        DataProcessor with FileDataProvider registered
    """
    from .file_data_provider import FileDataProvider, FileDataProviderConfig
    
    processor = DataProcessor()
    
    file_config = FileDataProviderConfig(
        news_file=news_file,
        ohlcv_file=ohlcv_file,
        fundamentals_file=fundamentals_file,
        events_file=events_file,
        base_dir=base_dir,
    )
    processor.register_provider(FileDataProvider(file_config))
    
    return processor


def create_processor_hybrid(
    news_file: Optional[str] = None,
    base_dir: Optional[str] = None,
    exchange_suffix: str = ".NS",
    enable_yahoo: bool = True,
) -> DataProcessor:
    """Create a DataProcessor with both file and Yahoo Finance providers.
    
    Yahoo Finance provides live prices/fundamentals, files provide news/events.
    
    Args:
        news_file: Path to news JSON file
        base_dir: Base directory for files
        exchange_suffix: Exchange suffix for Yahoo Finance
        enable_yahoo: Whether to enable Yahoo Finance provider
    
    Returns:
        DataProcessor with both providers registered
    """
    from .file_data_provider import FileDataProvider, FileDataProviderConfig
    
    processor = DataProcessor()
    
    # Register file provider first for news
    if news_file:
        file_config = FileDataProviderConfig(
            news_file=news_file,
            base_dir=base_dir,
        )
        processor.register_provider(FileDataProvider(file_config))
    
    # Register Yahoo Finance for prices/fundamentals
    if enable_yahoo:
        try:
            from .yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig
            yf_config = YFinanceConfig(
                default_exchange_suffix=exchange_suffix,
            )
            processor.register_provider(YahooFinanceProvider(yf_config))
        except ImportError:
            logger.warning("yfinance not available, skipping Yahoo Finance provider")
    
    return processor
