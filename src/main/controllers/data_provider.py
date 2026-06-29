"""Data Provider interfaces and types for LCF.

This module defines the core abstractions for data providers:
- DataType: Enum representing different types of financial data
- DataProvider: Abstract interface for data source implementations
- DataProviderResult: Container for data fetched by a provider
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Set


class DataType(Enum):
    """Types of financial data that can be fetched from data providers.
    
    Each data type represents a category of financial information:
    - OHLCV: Open, High, Low, Close, Volume price data
    - FUNDAMENTALS: Company fundamentals (PE, EPS, market cap, etc.)
    - NEWS: News articles and headlines
    - EVENTS: Corporate events (earnings, dividends, splits)
    - ANALYST: Analyst ratings and price targets
    - INCOME_STATEMENT: Income statement financial data
    - BALANCE_SHEET: Balance sheet financial data
    - CASH_FLOW: Cash flow statement data
    - INSIDER: Insider trading activity
    - INSTITUTIONAL: Institutional holdings
    - SENTIMENT: Sentiment analysis data
    - TECHNICAL: Technical indicators
    """
    OHLCV = auto()
    FUNDAMENTALS = auto()
    NEWS = auto()
    EVENTS = auto()
    ANALYST = auto()
    INCOME_STATEMENT = auto()
    BALANCE_SHEET = auto()
    CASH_FLOW = auto()
    INSIDER = auto()
    INSTITUTIONAL = auto()
    SENTIMENT = auto()
    TECHNICAL = auto()


@dataclass
class DataProviderResult:
    """Result container for data fetched by a DataProvider.
    
    Contains the fetched data organized by DataType, along with metadata
    about the fetch operation.
    
    Attributes:
        provider_name: Name of the provider that fetched this data
        symbols: List of symbols for which data was requested
        data: Dictionary mapping DataType to list of data items
        timestamp: When the data was fetched
        errors: Any errors encountered during fetch
        metadata: Additional provider-specific metadata
    """
    provider_name: str
    symbols: List[str]
    data: Dict[DataType, List[Any]] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_data(self, data_type: DataType) -> bool:
        """Check if this result contains data of the specified type."""
        return data_type in self.data and len(self.data[data_type]) > 0
    
    def get_data(self, data_type: DataType) -> List[Any]:
        """Get data of the specified type, or empty list if not present."""
        return self.data.get(data_type, [])
    
    def add_data(self, data_type: DataType, items: List[Any]) -> None:
        """Add data items for a specific data type."""
        if data_type not in self.data:
            self.data[data_type] = []
        self.data[data_type].extend(items)
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
    
    @property
    def available_types(self) -> Set[DataType]:
        """Get set of data types that have data in this result."""
        return {dt for dt, items in self.data.items() if items}
    
    @property
    def total_records(self) -> int:
        """Get total number of records across all data types."""
        return sum(len(items) for items in self.data.values())
    
    def summary(self) -> str:
        """Get a summary of this result."""
        type_counts = {dt.name: len(items) for dt, items in self.data.items() if items}
        return (
            f"DataProviderResult(provider={self.provider_name}, "
            f"symbols={self.symbols}, types={type_counts}, "
            f"errors={len(self.errors)})"
        )


@dataclass 
class DataProviderConfig:
    """Base configuration for data providers.
    
    Subclass this for provider-specific configurations.
    
    Attributes:
        enabled: Whether this provider is enabled
        supported_types: Data types this provider can fetch
        priority: Provider priority (higher = preferred)
        timeout_seconds: Timeout for API calls
        retry_count: Number of retries on failure
    """
    enabled: bool = True
    supported_types: Set[DataType] = field(default_factory=lambda: set(DataType))
    priority: int = 0
    timeout_seconds: float = 30.0
    retry_count: int = 3


class DataProvider(ABC):
    """Abstract interface for financial data providers.
    
    A DataProvider fetches financial data from a specific source (API, file, etc.)
    and returns it in a standardized DataProviderResult format.
    
    Implementations should:
    1. Define which DataTypes they support via supported_types property
    2. Implement fetch() to retrieve data for given symbols
    3. Return data organized by DataType in the result
    
    Example implementation:
        class MyDataProvider(DataProvider):
            @property
            def name(self) -> str:
                return "my_provider"
            
            @property
            def supported_types(self) -> Set[DataType]:
                return {DataType.OHLCV, DataType.NEWS}
            
            def fetch(self, symbols, data_types=None) -> DataProviderResult:
                result = DataProviderResult(
                    provider_name=self.name,
                    symbols=symbols
                )
                # Fetch and add data...
                return result
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this data provider."""
        pass
    
    @property
    @abstractmethod
    def supported_types(self) -> Set[DataType]:
        """Set of DataTypes this provider can fetch."""
        pass
    
    @abstractmethod
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch data for the given symbols.
        
        Args:
            symbols: List of stock symbols to fetch data for
            data_types: Optional set of data types to fetch. If None, fetches all supported types.
            **kwargs: Provider-specific parameters
        
        Returns:
            DataProviderResult containing the fetched data organized by DataType
        """
        pass
    
    def supports(self, data_type: DataType) -> bool:
        """Check if this provider supports the given data type."""
        return data_type in self.supported_types
    
    def validate_symbols(self, symbols: List[str]) -> List[str]:
        """Validate and normalize symbol list. Override for custom validation."""
        return [s.upper().strip() for s in symbols if s and s.strip()]


class CompositeDataProvider(DataProvider):
    """A data provider that aggregates results from multiple providers.
    
    This provider delegates to registered sub-providers and merges their results.
    Useful for combining data from multiple sources.
    
    Attributes:
        providers: List of registered data providers
    """
    
    def __init__(self, providers: Optional[List[DataProvider]] = None):
        """Initialize with optional list of providers."""
        self._providers: List[DataProvider] = providers or []
    
    @property
    def name(self) -> str:
        return "composite"
    
    @property
    def supported_types(self) -> Set[DataType]:
        """Union of all registered provider's supported types."""
        types: Set[DataType] = set()
        for provider in self._providers:
            types.update(provider.supported_types)
        return types
    
    def register(self, provider: DataProvider) -> None:
        """Register a data provider."""
        self._providers.append(provider)
    
    def unregister(self, provider_name: str) -> bool:
        """Unregister a provider by name. Returns True if found and removed."""
        for i, p in enumerate(self._providers):
            if p.name == provider_name:
                self._providers.pop(i)
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
    
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch data from all registered providers and merge results.
        
        For each data type, if multiple providers return data, the first
        provider's data is used (based on registration order).
        """
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols
        )
        
        requested_types = data_types or self.supported_types
        
        for provider in self._providers:
            # Filter to types this provider supports and we want
            provider_types = requested_types & provider.supported_types
            if not provider_types:
                continue
            
            try:
                provider_result = provider.fetch(symbols, provider_types, **kwargs)
                
                # Merge data - first provider wins for each type
                for data_type, items in provider_result.data.items():
                    if data_type not in result.data:
                        result.data[data_type] = items
                        result.metadata[f"{data_type.name}_source"] = provider.name
                
                # Collect errors
                result.errors.extend(provider_result.errors)
                
            except Exception as e:
                result.add_error(f"Error from {provider.name}: {str(e)}")
        
        return result
