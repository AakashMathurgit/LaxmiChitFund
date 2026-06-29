"""File-based Data Provider for LCF.

Implements the DataProvider interface to fetch data from JSON/CSV files.
Useful for testing, historical data, and offline analysis.
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
import logging

from .data_provider import DataProvider, DataProviderResult, DataType, DataProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class FileDataProviderConfig(DataProviderConfig):
    """Configuration for file-based data provider."""
    # File paths for different data types
    ohlcv_file: Optional[str] = None
    fundamentals_file: Optional[str] = None
    news_file: Optional[str] = None
    events_file: Optional[str] = None
    analyst_file: Optional[str] = None
    
    # Base directory for relative paths
    base_dir: Optional[str] = None
    
    # File format options
    default_encoding: str = "utf-8"
    json_indent: int = 2
    
    # Data filtering
    filter_by_symbol: bool = True  # Only return data matching requested symbols
    
    # Default supported types based on configured files
    supported_types: Set[DataType] = field(default_factory=lambda: {
        DataType.OHLCV,
        DataType.FUNDAMENTALS,
        DataType.NEWS,
        DataType.EVENTS,
        DataType.ANALYST,
    })


class FileDataProvider(DataProvider):
    """Data provider that reads from JSON/CSV files.
    
    Useful for:
    - Development and testing with mock data
    - Historical analysis with downloaded data
    - Offline operation
    - Combining with live data sources
    
    File format expectations:
    - JSON files should have an "items" array with data objects
    - Each item should have a "symbol" field for filtering
    - CSV files have headers matching field names
    
    Example JSON structure:
        {
            "items": [
                {"symbol": "AAPL", "date": "2024-01-01", "close": 150.0, ...},
                {"symbol": "MSFT", "date": "2024-01-01", "close": 300.0, ...}
            ]
        }
    
    Usage:
        config = FileDataProviderConfig(
            news_file="data/stock_news.json",
            ohlcv_file="data/prices.json"
        )
        provider = FileDataProvider(config)
        result = provider.fetch(["AAPL", "MSFT"])
    """
    
    def __init__(self, config: Optional[FileDataProviderConfig] = None):
        """Initialize file data provider.
        
        Args:
            config: Configuration with file paths. Uses defaults if not provided.
        """
        self.config = config or FileDataProviderConfig()
        self._cache: Dict[str, Any] = {}
        logger.info("FileDataProvider initialized")
    
    @property
    def name(self) -> str:
        """Unique provider name."""
        return "file"
    
    @property
    def supported_types(self) -> Set[DataType]:
        """Data types this provider can fetch based on configured files."""
        types = set()
        if self.config.ohlcv_file:
            types.add(DataType.OHLCV)
        if self.config.fundamentals_file:
            types.add(DataType.FUNDAMENTALS)
        if self.config.news_file:
            types.add(DataType.NEWS)
        if self.config.events_file:
            types.add(DataType.EVENTS)
        if self.config.analyst_file:
            types.add(DataType.ANALYST)
        return types or self.config.supported_types
    
    def _resolve_path(self, file_path: str) -> Path:
        """Resolve file path, applying base_dir if path is relative."""
        path = Path(file_path)
        if not path.is_absolute() and self.config.base_dir:
            path = Path(self.config.base_dir) / path
        return path
    
    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load and parse a JSON file."""
        path = self._resolve_path(file_path)
        
        if str(path) in self._cache:
            return self._cache[str(path)]
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return {"items": []}
        
        try:
            with open(path, "r", encoding=self.config.default_encoding) as f:
                data = json.load(f)
            self._cache[str(path)] = data
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in {path}: {e}")
            return {"items": []}
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            return {"items": []}
    
    def _load_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load and parse a CSV file."""
        path = self._resolve_path(file_path)
        
        cache_key = f"csv:{path}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        if not path.exists():
            logger.warning(f"CSV file not found: {path}")
            return []
        
        try:
            with open(path, "r", encoding=self.config.default_encoding, newline="") as f:
                reader = csv.DictReader(f)
                items = list(reader)
            self._cache[cache_key] = items
            return items
        except Exception as e:
            logger.error(f"Error reading CSV {path}: {e}")
            return []
    
    def _filter_by_symbols(
        self, 
        items: List[Dict[str, Any]], 
        symbols: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter items to only those matching requested symbols."""
        if not self.config.filter_by_symbol:
            return items
        
        symbols_upper = {s.upper() for s in symbols}
        return [
            item for item in items 
            if item.get("symbol", "").upper() in symbols_upper
        ]
    
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch data from configured files.
        
        Args:
            symbols: List of stock symbols to fetch data for
            data_types: Set of DataTypes to fetch. If None, fetches all configured types.
            **kwargs: Additional parameters (ignored for file provider)
        
        Returns:
            DataProviderResult containing fetched data organized by DataType
        """
        symbols = self.validate_symbols(symbols)
        requested_types = data_types or self.supported_types
        
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols,
            metadata={
                "base_dir": self.config.base_dir,
            }
        )
        
        # Fetch each requested data type
        if DataType.OHLCV in requested_types and self.config.ohlcv_file:
            self._fetch_from_file(
                self.config.ohlcv_file, 
                DataType.OHLCV, 
                symbols, 
                result
            )
        
        if DataType.FUNDAMENTALS in requested_types and self.config.fundamentals_file:
            self._fetch_from_file(
                self.config.fundamentals_file, 
                DataType.FUNDAMENTALS, 
                symbols, 
                result
            )
        
        if DataType.NEWS in requested_types and self.config.news_file:
            self._fetch_from_file(
                self.config.news_file, 
                DataType.NEWS, 
                symbols, 
                result
            )
        
        if DataType.EVENTS in requested_types and self.config.events_file:
            self._fetch_from_file(
                self.config.events_file, 
                DataType.EVENTS, 
                symbols, 
                result
            )
        
        if DataType.ANALYST in requested_types and self.config.analyst_file:
            self._fetch_from_file(
                self.config.analyst_file, 
                DataType.ANALYST, 
                symbols, 
                result
            )
        
        logger.info(f"FileDataProvider fetched: {result.summary()}")
        return result
    
    def _fetch_from_file(
        self, 
        file_path: str, 
        data_type: DataType,
        symbols: List[str],
        result: DataProviderResult
    ) -> None:
        """Fetch data from a specific file."""
        try:
            path = self._resolve_path(file_path)
            
            # Determine file format and load
            if str(path).endswith(".csv"):
                items = self._load_csv_file(file_path)
            else:
                data = self._load_json_file(file_path)
                items = data.get("items", data.get("data", []))
                # Handle case where JSON is a list directly
                if isinstance(data, list):
                    items = data
            
            # Filter by symbols
            filtered_items = self._filter_by_symbols(items, symbols)
            
            result.add_data(data_type, filtered_items)
            result.metadata[f"{data_type.name.lower()}_file"] = file_path
            result.metadata[f"{data_type.name.lower()}_count"] = len(filtered_items)
            
            logger.debug(f"Loaded {len(filtered_items)} {data_type.name} records from {file_path}")
            
        except Exception as e:
            logger.error(f"Error fetching {data_type.name} from {file_path}: {e}")
            result.add_error(f"{data_type.name} file error: {str(e)}")
    
    def clear_cache(self) -> None:
        """Clear the file cache."""
        self._cache.clear()
        logger.debug("File cache cleared")
    
    def reload_file(self, file_path: str) -> None:
        """Reload a specific file by removing it from cache."""
        path = self._resolve_path(file_path)
        keys_to_remove = [k for k in self._cache if str(path) in k]
        for key in keys_to_remove:
            del self._cache[key]
        logger.debug(f"Cache invalidated for {file_path}")
