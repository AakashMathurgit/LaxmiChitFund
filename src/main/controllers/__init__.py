"""Controllers package for LCF.

Controllers coordinate data flow and orchestration between components.

This package provides:
- DataProvider interface and DataType enum for extensible data sources
- DataProcessor for orchestrating data fetching from multiple providers
- Concrete provider implementations (YahooFinance, File-based)
- DataContext for passing typed data through the agent pipeline
"""

# Data models
from .data_context import (
    DataContext,
    NewsItem,
    PriceData,
    FundamentalData,
    EventData,
)

# Data provider interface and types
from .data_provider import (
    DataType,
    DataProvider,
    DataProviderResult,
    DataProviderConfig,
    CompositeDataProvider,
)

# Data processor
from .data_processor import (
    DataProcessor,
    DataProcessorConfig,
    create_processor_with_yahoo_finance,
    create_processor_with_files,
    create_processor_hybrid,
)

# File data provider
from .file_data_provider import (
    FileDataProvider,
    FileDataProviderConfig,
)

# Yahoo Finance provider (optional - requires yfinance)
try:
    from .yahoo_finance_provider import (
        YahooFinanceProvider,
        YFinanceConfig,
        YahooFinanceDataSource,  # Legacy alias
        is_yfinance_available,
    )
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    YahooFinanceProvider = None
    YFinanceConfig = None
    YahooFinanceDataSource = None
    is_yfinance_available = lambda: False

__all__ = [
    # Data Context and Models
    "DataContext",
    "NewsItem",
    "PriceData",
    "FundamentalData",
    "EventData",
    # DataProvider Interface
    "DataType",
    "DataProvider",
    "DataProviderResult",
    "DataProviderConfig",
    "CompositeDataProvider",
    # Data Processor
    "DataProcessor",
    "DataProcessorConfig",
    "create_processor_with_yahoo_finance",
    "create_processor_with_files",
    "create_processor_hybrid",
    # File Provider
    "FileDataProvider",
    "FileDataProviderConfig",
    # Yahoo Finance Provider
    "YahooFinanceProvider",
    "YFinanceConfig",
    "YahooFinanceDataSource",  # Legacy alias
    "is_yfinance_available",
]
