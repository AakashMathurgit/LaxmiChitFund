"""NSE Corporate Data Provider for LCF.

Fetches corporate actions and announcements from local SQLite databases
populated by the NSE RSS ingestion scripts.

Data sources:
- corporate_actions.sqlite: Dividends, splits, bonuses, rights issues, buybacks
- corporate_announcements.sqlite: Board meetings, director changes, filings

This provider enriches the StockDataContext with official NSE corporate events
that may not be available via Yahoo Finance.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .data_provider import DataProvider, DataProviderResult, DataType, DataProviderConfig
from .data_context import EventData, NewsItem

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class NSECorporateConfig(DataProviderConfig):
    """Configuration for NSE Corporate data provider.
    
    Attributes:
        db_base_path: Base path to the normalized NSE data directory
        actions_db: Filename for corporate actions SQLite database
        announcements_db: Filename for corporate announcements SQLite database
        lookback_days_actions: Days of corporate actions to fetch
        lookback_days_announcements: Days of announcements to fetch
        include_upcoming_events: Include events with future ex_dates
        upcoming_days: How many days ahead to look for upcoming events
        company_to_symbol_map: Manual mapping of company names to symbols
    """
    db_base_path: str = os.path.join("data", "normalized", "nse")
    actions_db: str = "corporate_actions.sqlite"
    announcements_db: str = "corporate_announcements.sqlite"
    
    lookback_days_actions: int = 30
    lookback_days_announcements: int = 7
    include_upcoming_events: bool = True
    upcoming_days: int = 14
    
    # Manual company name -> symbol mapping (NSE symbols use .NS suffix)
    company_to_symbol_map: Dict[str, str] = field(default_factory=lambda: {
        "TATA CONSULTANCY SERVICES LIMITED": "TCS",
        "INFOSYS LIMITED": "INFY",
        "RELIANCE INDUSTRIES LIMITED": "RELIANCE",
        "HDFC BANK LIMITED": "HDFCBANK",
        "ICICI BANK LIMITED": "ICICIBANK",
        "STATE BANK OF INDIA": "SBIN",
        "WIPRO LIMITED": "WIPRO",
        "BHARTI AIRTEL LIMITED": "BHARTIARTL",
        "HINDUSTAN UNILEVER LIMITED": "HINDUNILVR",
        "ITC LIMITED": "ITC",
        "LARSEN & TOUBRO LIMITED": "LT",
        "KOTAK MAHINDRA BANK LIMITED": "KOTAKBANK",
        "ASIAN PAINTS LIMITED": "ASIANPAINT",
        "AXIS BANK LIMITED": "AXISBANK",
        "MARUTI SUZUKI INDIA LIMITED": "MARUTI",
        "SUN PHARMACEUTICAL INDUSTRIES LIMITED": "SUNPHARMA",
        "BAJAJ FINANCE LIMITED": "BAJFINANCE",
        "TITAN COMPANY LIMITED": "TITAN",
        "NESTLE INDIA LIMITED": "NESTLEIND",
        "ULTRATECH CEMENT LIMITED": "ULTRACEMCO",
        "TECH MAHINDRA LIMITED": "TECHM",
        "POWER GRID CORPORATION OF INDIA LIMITED": "POWERGRID",
        "NTPC LIMITED": "NTPC",
        "HCL TECHNOLOGIES LIMITED": "HCLTECH",
        "MAHINDRA & MAHINDRA LIMITED": "M&M",
        "TATA STEEL LIMITED": "TATASTEEL",
        "TATA MOTORS LIMITED": "TATAMOTORS",
        "ADANI ENTERPRISES LIMITED": "ADANIENT",
        "BAJAJ FINSERV LIMITED": "BAJAJFINSV",
        "INDUSIND BANK LIMITED": "INDUSINDBK",
        "SBI LIFE INSURANCE COMPANY LIMITED": "SBILIFE",
        "HDFC LIFE INSURANCE COMPANY LIMITED": "HDFCLIFE",
    })
    
    # Default supported types
    supported_types: Set[DataType] = field(default_factory=lambda: {
        DataType.EVENTS,
        DataType.NEWS,
    })


# ---------------------------------------------------------------------------
# Provider Implementation
# ---------------------------------------------------------------------------

class NSECorporateProvider(DataProvider):
    """Data provider for NSE corporate actions and announcements.
    
    Reads from local SQLite databases populated by:
    - scripts/ingest_nse_corp_actions.py
    - scripts/ingest_nse_corp_announcements.py
    
    Returns:
    - EventData for corporate actions (dividends, splits, bonuses)
    - NewsItem for corporate announcements (board meetings, filings)
    
    Usage:
        provider = NSECorporateProvider()
        result = provider.fetch(["TCS", "INFY", "RELIANCE"])
        
        events = result.get_data(DataType.EVENTS)
        news = result.get_data(DataType.NEWS)
    """
    
    def __init__(self, config: Optional[NSECorporateConfig] = None, base_path: Optional[str] = None):
        """Initialize NSE Corporate provider.
        
        Args:
            config: Optional NSECorporateConfig. Uses defaults if not provided.
            base_path: Base path for relative db_base_path resolution.
                      If not provided, uses current working directory.
        """
        self.config = config or NSECorporateConfig()
        self._base_path = base_path or os.getcwd()
        
        # Resolve database paths
        db_base = self.config.db_base_path
        if not os.path.isabs(db_base):
            db_base = os.path.join(self._base_path, db_base)
        
        self._actions_db_path = os.path.join(db_base, self.config.actions_db)
        self._announcements_db_path = os.path.join(db_base, self.config.announcements_db)
        
        # Build reverse lookup: symbol -> company names (for matching)
        self._symbol_to_companies: Dict[str, List[str]] = {}
        for company, symbol in self.config.company_to_symbol_map.items():
            symbol_upper = symbol.upper()
            if symbol_upper not in self._symbol_to_companies:
                self._symbol_to_companies[symbol_upper] = []
            self._symbol_to_companies[symbol_upper].append(company.upper())
        
        logger.info(f"NSECorporateProvider initialized (actions_db={self._actions_db_path})")
    
    @property
    def name(self) -> str:
        """Provider name for identification."""
        return "nse_corporate"
    
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
        """Fetch corporate data for given symbols.
        
        Args:
            symbols: List of stock symbols (e.g., ["TCS", "INFY"])
            data_types: Optional set of DataTypes to fetch.
                       If None, fetches all supported types.
            **kwargs: Additional parameters (ignored for this provider)
        
        Returns:
            DataProviderResult with EventData and NewsItem objects
        """
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols,
        )
        
        requested_types = data_types or self.config.supported_types
        
        # Normalize symbols
        symbols_upper = [self._normalize_symbol(s) for s in symbols]
        
        try:
            # Fetch corporate actions as events
            if DataType.EVENTS in requested_types:
                events = self._fetch_corporate_actions(symbols_upper)
                result.add_data(DataType.EVENTS, events)
                logger.debug(f"Fetched {len(events)} corporate action events")
            
            # Fetch corporate announcements as news
            if DataType.NEWS in requested_types:
                news = self._fetch_announcements(symbols_upper)
                result.add_data(DataType.NEWS, news)
                logger.debug(f"Fetched {len(news)} corporate announcements")
                
        except Exception as e:
            error_msg = f"Error fetching NSE corporate data: {e}"
            logger.error(error_msg)
            result.add_error(error_msg)
        
        return result
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol by removing exchange suffix."""
        symbol = symbol.upper().strip()
        for suffix in [".NS", ".BO", ".BSE", ".NSE"]:
            if symbol.endswith(suffix):
                return symbol[:-len(suffix)]
        return symbol
    
    def _match_company_to_symbol(self, company_name: str, symbols: List[str]) -> Optional[str]:
        """Match company name to one of the requested symbols.
        
        Uses both the config mapping and fuzzy substring matching.
        """
        company_upper = company_name.upper().strip()
        
        for symbol in symbols:
            # Check explicit mapping
            if symbol in self._symbol_to_companies:
                for mapped_company in self._symbol_to_companies[symbol]:
                    if mapped_company == company_upper:
                        return symbol
            
            # Fuzzy match: symbol appears in company name
            if symbol in company_upper.replace(" ", ""):
                return symbol
            
            # Common variations
            if symbol == "TCS" and "TATA CONSULTANCY" in company_upper:
                return symbol
            if symbol == "INFY" and "INFOSYS" in company_upper:
                return symbol
            if symbol == "RELIANCE" and "RELIANCE INDUSTRIES" in company_upper:
                return symbol
        
        return None
    
    def _fetch_corporate_actions(self, symbols: List[str]) -> List[EventData]:
        """Fetch corporate actions from SQLite and convert to EventData."""
        events: List[EventData] = []
        
        if not os.path.exists(self._actions_db_path):
            logger.warning(f"Corporate actions DB not found: {self._actions_db_path}")
            return events
        
        conn = sqlite3.connect(self._actions_db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cur = conn.cursor()
            
            # Date range filter
            today = datetime.now().strftime("%Y-%m-%d")
            lookback = (datetime.now() - timedelta(days=self.config.lookback_days_actions)).strftime("%Y-%m-%d")
            future = (datetime.now() + timedelta(days=self.config.upcoming_days)).strftime("%Y-%m-%d")
            
            # Query all recent/upcoming corporate actions
            query = """
                SELECT * FROM corporate_actions
                WHERE (ex_date >= ? AND ex_date <= ?)
                   OR (record_date >= ? AND record_date <= ?)
                ORDER BY ex_date DESC
            """
            cur.execute(query, (lookback, future, lookback, future))
            
            for row in cur.fetchall():
                company_name = row["company_name"] or ""
                matched_symbol = self._match_company_to_symbol(company_name, symbols)
                
                if matched_symbol is None:
                    continue  # Skip if company doesn't match any requested symbol
                
                # Build description from action details
                action_type = row["action_type"] or "CORPORATE_ACTION"
                description = self._build_action_description(row)
                
                # Calculate impact score based on action type
                impact_score = self._calculate_action_impact(row)
                
                # Create EventData
                event = EventData(
                    symbol=matched_symbol,
                    date=row["ex_date"] or row["record_date"] or today,
                    event_type=action_type,
                    description=description,
                    impact_score=impact_score,
                    source="NSE",
                    earnings_date=None,
                    recent_corporate_actions=[{
                        "action_type": action_type,
                        "action_subtype": row["action_subtype"],
                        "amount_inr": row["amount_inr"],
                        "ratio_from": row["ratio_from"],
                        "ratio_to": row["ratio_to"],
                        "ex_date": row["ex_date"],
                        "record_date": row["record_date"],
                        "face_value": row["face_value"],
                        "company_name": company_name,
                    }],
                )
                events.append(event)
                
        finally:
            conn.close()
        
        return events
    
    def _build_action_description(self, row: sqlite3.Row) -> str:
        """Build human-readable description from action row."""
        action_type = row["action_type"] or "ACTION"
        subtype = row["action_subtype"]
        amount = row["amount_inr"]
        ratio_from = row["ratio_from"]
        ratio_to = row["ratio_to"]
        ex_date = row["ex_date"]
        
        parts = [f"{action_type}"]
        
        if subtype:
            parts[0] = f"{subtype} {action_type}"
        
        if action_type == "DIVIDEND" and amount:
            parts.append(f"Rs {amount:.2f} per share")
        elif action_type in ("BONUS", "RIGHTS") and ratio_from and ratio_to:
            parts.append(f"{ratio_from}:{ratio_to}")
        elif action_type == "SPLIT":
            if row["split_from_fv"] and row["split_to_fv"]:
                parts.append(f"FV Rs {row['split_from_fv']:.0f} to Rs {row['split_to_fv']:.0f}")
        
        if ex_date:
            parts.append(f"Ex-date: {ex_date}")
        
        return " | ".join(parts)
    
    def _calculate_action_impact(self, row: sqlite3.Row) -> float:
        """Calculate impact score (0-1) for corporate action.
        
        Higher scores for more significant events:
        - Stock splits: 0.8 (high impact on price/technical)
        - Large dividends: 0.6-0.7
        - Bonus issues: 0.7
        - Rights issues: 0.6
        - Small dividends: 0.3-0.5
        """
        action_type = row["action_type"]
        amount = row["amount_inr"] or 0
        
        if action_type == "SPLIT":
            return 0.8
        elif action_type == "BONUS":
            return 0.7
        elif action_type == "RIGHTS":
            return 0.6
        elif action_type == "BUYBACK":
            return 0.7
        elif action_type == "DIVIDEND":
            # Scale dividend impact based on amount
            if amount >= 10:
                return 0.7
            elif amount >= 5:
                return 0.5
            elif amount >= 1:
                return 0.4
            else:
                return 0.3
        else:
            return 0.3
    
    def _fetch_announcements(self, symbols: List[str]) -> List[NewsItem]:
        """Fetch corporate announcements from SQLite and convert to NewsItem."""
        news_items: List[NewsItem] = []
        
        if not os.path.exists(self._announcements_db_path):
            logger.warning(f"Announcements DB not found: {self._announcements_db_path}")
            return news_items
        
        conn = sqlite3.connect(self._announcements_db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cur = conn.cursor()
            
            # Date range filter (announcements use fetched_at)
            lookback = (datetime.now() - timedelta(days=self.config.lookback_days_announcements)).isoformat()
            
            # Query recent announcements
            query = """
                SELECT * FROM corporate_announcements
                WHERE fetched_at >= ?
                ORDER BY fetched_at DESC
                LIMIT 500
            """
            cur.execute(query, (lookback,))
            
            for row in cur.fetchall():
                company_name = row["title"] or ""  # title contains company name
                matched_symbol = self._match_company_to_symbol(company_name, symbols)
                
                if matched_symbol is None:
                    continue
                
                summary = row["summary"] or ""
                
                # Extract subject from summary if present
                headline = company_name
                if "|SUBJECT:" in summary:
                    subject_part = summary.split("|SUBJECT:")[-1].strip()
                    headline = f"{company_name}: {subject_part[:100]}"
                
                news_item = NewsItem(
                    symbol=matched_symbol,
                    date=row["fetched_at"][:10] if row["fetched_at"] else datetime.now().strftime("%Y-%m-%d"),
                    headline=headline,
                    news_text=summary,
                    source="NSE",
                    url=row["link"],
                    sentiment_score=None,  # To be computed by SentimentAgent
                )
                news_items.append(news_item)
                
        finally:
            conn.close()
        
        return news_items
    
    # ---------------------------------------------------------------------------
    # StockDataContext integration (for DataProcessorAgent pattern)
    # ---------------------------------------------------------------------------
    
    def enrich_stock_context(
        self,
        ctx: "StockDataContext",
        **kwargs
    ) -> None:
        """Enrich a StockDataContext with NSE corporate data.
        
        Adds:
        - Corporate actions to ctx.event_data
        - Corporate announcements to ctx.news_items
        
        Args:
            ctx: StockDataContext to enrich
        """
        from .data_context import StockDataContext
        
        result = self.fetch([ctx.symbol])
        
        # Add events
        events = result.get_data(DataType.EVENTS)
        for event in events:
            if isinstance(event, EventData):
                ctx.event_data.append(event)
        
        # Add news
        news = result.get_data(DataType.NEWS)
        for item in news:
            if isinstance(item, NewsItem):
                ctx.news_items.append(item)
        
        # Update flags
        ctx.flags["nse_corp_data_enriched"] = True
        ctx.flags["nse_event_count"] = len(events)
        ctx.flags["nse_announcement_count"] = len(news)
        
        logger.debug(f"Enriched {ctx.symbol} with {len(events)} events, {len(news)} announcements")
