"""Yahoo Finance Data Provider for LCF.

Implements the DataProvider interface using the yfinance library to fetch 
real-time and historical stock data.

Install: pip install yfinance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set
from datetime import datetime
import logging

from .data_provider import DataProvider, DataProviderResult, DataType, DataProviderConfig

if TYPE_CHECKING:
    from .data_context import IndexData, StockDataContext

logger = logging.getLogger(__name__)

# Graceful import of yfinance
try:
    import yfinance as yf
    import pandas as pd
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False
    yf = None
    pd = None


@dataclass
class YFinanceConfig(DataProviderConfig):
    """Configuration for Yahoo Finance data provider."""
    # Price history settings
    price_period: str = "1mo"  # 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    price_interval: str = "1d"  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
    
    # Data fetch options
    fetch_info: bool = True
    fetch_history: bool = True
    fetch_fundamentals: bool = True
    fetch_news: bool = True
    fetch_analyst_data: bool = True
    fetch_financials: bool = False  # Income statement, balance sheet
    
    # NSE/BSE suffix for Indian stocks (e.g., "TCS.NS" for NSE)
    default_exchange_suffix: str = ".NS"  # .NS for NSE, .BO for BSE, "" for US stocks
    
    # Rate limiting
    request_delay_ms: int = 100
    
    # Default supported types
    supported_types: Set[DataType] = field(default_factory=lambda: {
        DataType.OHLCV,
        DataType.FUNDAMENTALS,
        DataType.NEWS,
        DataType.ANALYST,
        DataType.EVENTS,
        DataType.INCOME_STATEMENT,
        DataType.BALANCE_SHEET,
    })


class YahooFinanceProvider(DataProvider):
    """Data provider that fetches stock data from Yahoo Finance.
    
    Implements the DataProvider interface to provide:
    - Real-time quotes and info
    - Historical OHLCV price data
    - Fundamental data (PE, EPS, market cap, etc.)
    - Analyst recommendations and price targets
    - Company news
    - Financial statements
    
    Usage:
        provider = YahooFinanceProvider()
        
        # Fetch all supported data types
        result = provider.fetch(["TCS.NS", "INFY.NS"])
        
        # Fetch specific data types
        result = provider.fetch(
            ["MSFT", "AAPL"], 
            data_types={DataType.OHLCV, DataType.FUNDAMENTALS}
        )
        
        # Access data by type
        prices = result.get_data(DataType.OHLCV)
        fundamentals = result.get_data(DataType.FUNDAMENTALS)
    """
    
    def __init__(self, config: Optional[YFinanceConfig] = None):
        """Initialize Yahoo Finance provider.
        
        Args:
            config: Optional YFinanceConfig. Uses defaults if not provided.
        
        Raises:
            ImportError: If yfinance is not installed.
        """
        if not _YFINANCE_AVAILABLE:
            raise ImportError(
                "yfinance is not installed. Install with: pip install yfinance"
            )
        
        self.config = config or YFinanceConfig()
        self._ticker_cache: Dict[str, Any] = {}
        logger.info("YahooFinanceProvider initialized")
    
    @property
    def name(self) -> str:
        """Unique provider name."""
        return "yahoo_finance"
    
    @property
    def supported_types(self) -> Set[DataType]:
        """Data types this provider can fetch."""
        return self.config.supported_types
    
    def _normalize_symbol(self, symbol: str) -> str:
        """Add exchange suffix if not present."""
        if "." not in symbol and self.config.default_exchange_suffix:
            return f"{symbol}{self.config.default_exchange_suffix}"
        return symbol
    
    def _get_ticker(self, symbol: str) -> Any:
        """Get or create a yfinance Ticker object (cached)."""
        normalized = self._normalize_symbol(symbol)
        if normalized not in self._ticker_cache:
            self._ticker_cache[normalized] = yf.Ticker(normalized)
        return self._ticker_cache[normalized]
    
    def fetch(
        self,
        symbols: List[str],
        data_types: Optional[Set[DataType]] = None,
        **kwargs
    ) -> DataProviderResult:
        """Fetch data for given symbols.
        
        Args:
            symbols: List of stock symbols
            data_types: Set of DataTypes to fetch. If None, fetches all supported types.
            **kwargs: Additional parameters (e.g., period, interval for OHLCV)
        
        Returns:
            DataProviderResult containing all fetched data organized by DataType
        """
        symbols = self.validate_symbols(symbols)
        requested_types = data_types or self.supported_types
        
        result = DataProviderResult(
            provider_name=self.name,
            symbols=symbols,
            metadata={
                "exchange_suffix": self.config.default_exchange_suffix,
                "price_period": self.config.price_period,
                "price_interval": self.config.price_interval,
            }
        )
        
        # Fetch each requested data type
        if DataType.OHLCV in requested_types:
            self._fetch_ohlcv(symbols, result, **kwargs)
        
        if DataType.FUNDAMENTALS in requested_types:
            self._fetch_fundamentals(symbols, result)
        
        if DataType.NEWS in requested_types:
            self._fetch_news(symbols, result)
        
        if DataType.ANALYST in requested_types:
            self._fetch_analyst(symbols, result)
        
        if DataType.INCOME_STATEMENT in requested_types:
            self._fetch_income_statement(symbols, result, **kwargs)
        
        if DataType.BALANCE_SHEET in requested_types:
            self._fetch_balance_sheet(symbols, result, **kwargs)
        
        if DataType.EVENTS in requested_types:
            self._fetch_corporate_events(symbols, result)
        
        logger.info(f"YahooFinanceProvider fetched: {result.summary()}")
        return result
    
    def _fetch_ohlcv(
        self, 
        symbols: List[str], 
        result: DataProviderResult,
        period: Optional[str] = None,
        interval: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> None:
        """Fetch OHLCV price data."""
        period = period or self.config.price_period
        interval = interval or self.config.price_interval
        
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                # Fetch history
                if start_date and end_date:
                    hist = ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    hist = ticker.history(period=period, interval=interval)
                
                if hist.empty:
                    logger.warning(f"No price data for {normalized}")
                    continue
                
                # Convert DataFrame to list of dicts
                for date_idx, row in hist.iterrows():
                    items.append({
                        "symbol": symbol,
                        "normalized_symbol": normalized,
                        "date": date_idx.strftime("%Y-%m-%d"),
                        "datetime": date_idx.isoformat(),
                        "open": float(row.get("Open", 0)),
                        "high": float(row.get("High", 0)),
                        "low": float(row.get("Low", 0)),
                        "close": float(row.get("Close", 0)),
                        "volume": int(row.get("Volume", 0)),
                        "dividends": float(row.get("Dividends", 0)),
                        "stock_splits": float(row.get("Stock Splits", 0)),
                    })
                
                logger.debug(f"Fetched {len(hist)} price records for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching prices for {symbol}: {e}")
                result.add_error(f"OHLCV error for {symbol}: {str(e)}")
        
        result.add_data(DataType.OHLCV, items)
        result.metadata["ohlcv_period"] = period
        result.metadata["ohlcv_interval"] = interval
    
    def _fetch_fundamentals(self, symbols: List[str], result: DataProviderResult) -> None:
        """Fetch fundamental data."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                info = ticker.info
                
                if not info:
                    logger.warning(f"No fundamental data for {normalized}")
                    continue
                
                fundamental = {
                    "symbol": symbol,
                    "normalized_symbol": normalized,
                    # Basic info
                    "company_name": info.get("longName") or info.get("shortName"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "currency": info.get("currency"),
                    # Valuation
                    "market_cap": info.get("marketCap"),
                    "enterprise_value": info.get("enterpriseValue"),
                    "pe_ratio": info.get("trailingPE"),
                    "forward_pe": info.get("forwardPE"),
                    "peg_ratio": info.get("pegRatio"),
                    "price_to_book": info.get("priceToBook"),
                    "price_to_sales": info.get("priceToSalesTrailing12Months"),
                    # Earnings
                    "eps": info.get("trailingEps"),
                    "forward_eps": info.get("forwardEps"),
                    "earnings_growth": info.get("earningsGrowth"),
                    "revenue_growth": info.get("revenueGrowth"),
                    # Dividends
                    "dividend_yield": info.get("dividendYield"),
                    "dividend_rate": info.get("dividendRate"),
                    "payout_ratio": info.get("payoutRatio"),
                    # Margins
                    "profit_margin": info.get("profitMargins"),
                    "operating_margin": info.get("operatingMargins"),
                    "gross_margin": info.get("grossMargins"),
                    # Financial health
                    "total_revenue": info.get("totalRevenue"),
                    "total_debt": info.get("totalDebt"),
                    "total_cash": info.get("totalCash"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "current_ratio": info.get("currentRatio"),
                    "quick_ratio": info.get("quickRatio"),
                    # Stock info
                    "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                    "previous_close": info.get("previousClose"),
                    "open_price": info.get("open") or info.get("regularMarketOpen"),
                    "day_high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                    "day_low": info.get("dayLow") or info.get("regularMarketDayLow"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "fifty_day_average": info.get("fiftyDayAverage"),
                    "two_hundred_day_average": info.get("twoHundredDayAverage"),
                    "volume": info.get("volume") or info.get("regularMarketVolume"),
                    "avg_volume": info.get("averageVolume"),
                    "avg_volume_10d": info.get("averageVolume10days"),
                    # Shares
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                    "shares_short": info.get("sharesShort"),
                    "short_ratio": info.get("shortRatio"),
                    # Beta
                    "beta": info.get("beta"),
                }
                
                items.append(fundamental)
                logger.debug(f"Fetched fundamentals for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching fundamentals for {symbol}: {e}")
                result.add_error(f"Fundamentals error for {symbol}: {str(e)}")
        
        result.add_data(DataType.FUNDAMENTALS, items)
    
    def _fetch_analyst(self, symbols: List[str], result: DataProviderResult) -> None:
        """Fetch analyst recommendations and price targets."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                analyst = {
                    "symbol": symbol,
                    "normalized_symbol": normalized,
                }
                
                # Analyst price targets
                try:
                    targets = ticker.analyst_price_targets
                    if targets is not None:
                        analyst["price_target_current"] = targets.get("current")
                        analyst["price_target_low"] = targets.get("low")
                        analyst["price_target_high"] = targets.get("high")
                        analyst["price_target_mean"] = targets.get("mean")
                        analyst["price_target_median"] = targets.get("median")
                except Exception:
                    pass
                
                # Recommendations
                try:
                    recs = ticker.recommendations
                    if recs is not None and not recs.empty:
                        latest = recs.iloc[-1] if len(recs) > 0 else None
                        if latest is not None:
                            analyst["latest_recommendation"] = latest.get("To Grade")
                            analyst["recommendation_firm"] = latest.get("Firm")
                except Exception:
                    pass
                
                # Info-based analyst data
                info = ticker.info
                if info:
                    analyst["analyst_target_price"] = info.get("targetMeanPrice")
                    analyst["analyst_target_high"] = info.get("targetHighPrice")
                    analyst["analyst_target_low"] = info.get("targetLowPrice")
                    analyst["recommendation_key"] = info.get("recommendationKey")
                    analyst["recommendation_mean"] = info.get("recommendationMean")
                    analyst["number_of_analysts"] = info.get("numberOfAnalystOpinions")
                
                items.append(analyst)
                logger.debug(f"Fetched analyst data for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching analyst data for {symbol}: {e}")
                result.add_error(f"Analyst error for {symbol}: {str(e)}")
        
        result.add_data(DataType.ANALYST, items)
    
    def _fetch_news(self, symbols: List[str], result: DataProviderResult) -> None:
        """Fetch recent news."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                news = ticker.news
                if not news:
                    logger.debug(f"No news for {normalized}")
                    continue
                
                for article in news:
                    items.append({
                        "symbol": symbol,
                        "normalized_symbol": normalized,
                        "title": article.get("title"),
                        "publisher": article.get("publisher"),
                        "link": article.get("link"),
                        "publish_time": article.get("providerPublishTime"),
                        "type": article.get("type"),
                        "thumbnail": article.get("thumbnail", {}).get("resolutions", [{}])[0].get("url") if article.get("thumbnail") else None,
                        "related_tickers": article.get("relatedTickers", []),
                    })
                
                logger.debug(f"Fetched {len(news)} news items for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching news for {symbol}: {e}")
                result.add_error(f"News error for {symbol}: {str(e)}")
        
        result.add_data(DataType.NEWS, items)
    
    def _fetch_income_statement(
        self, 
        symbols: List[str], 
        result: DataProviderResult,
        quarterly: bool = False,
        **kwargs
    ) -> None:
        """Fetch income statement data."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                if quarterly:
                    stmt = ticker.quarterly_income_stmt
                else:
                    stmt = ticker.income_stmt
                
                if stmt is not None and not stmt.empty:
                    items.append({
                        "symbol": symbol,
                        "normalized_symbol": normalized,
                        "type": "quarterly" if quarterly else "annual",
                        "data": stmt.to_dict(),
                    })
                    logger.debug(f"Fetched income statement for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching income statement for {symbol}: {e}")
                result.add_error(f"Income statement error for {symbol}: {str(e)}")
        
        result.add_data(DataType.INCOME_STATEMENT, items)
    
    def _fetch_balance_sheet(
        self, 
        symbols: List[str], 
        result: DataProviderResult,
        quarterly: bool = False,
        **kwargs
    ) -> None:
        """Fetch balance sheet data."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                if quarterly:
                    sheet = ticker.quarterly_balance_sheet
                else:
                    sheet = ticker.balance_sheet
                
                if sheet is not None and not sheet.empty:
                    items.append({
                        "symbol": symbol,
                        "normalized_symbol": normalized,
                        "type": "quarterly" if quarterly else "annual",
                        "data": sheet.to_dict(),
                    })
                    logger.debug(f"Fetched balance sheet for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching balance sheet for {symbol}: {e}")
                result.add_error(f"Balance sheet error for {symbol}: {str(e)}")
        
        result.add_data(DataType.BALANCE_SHEET, items)
    
    def _fetch_corporate_events(
        self, 
        symbols: List[str], 
        result: DataProviderResult,
    ) -> None:
        """Fetch corporate events: earnings dates, dividends, splits."""
        items = []
        
        for symbol in symbols:
            try:
                ticker = self._get_ticker(symbol)
                normalized = self._normalize_symbol(symbol)
                
                event_item = {
                    "symbol": symbol,
                    "normalized_symbol": normalized,
                    "earnings_date": None,
                    "ex_dividend_date": None,
                    "dividend_date": None,
                    "corporate_actions": [],
                }
                
                # Fetch earnings date from calendar
                try:
                    calendar = ticker.calendar
                    if calendar is not None and not calendar.empty:
                        dates = calendar.get("Earnings Date")
                        if dates is not None and len(dates) > 0:
                            ed = dates[0]
                            event_item["earnings_date"] = (
                                str(ed.date()) if hasattr(ed, "date") else str(ed)
                            )
                        # Ex-dividend date
                        ex_div = calendar.get("Ex-Dividend Date")
                        if ex_div is not None:
                            event_item["ex_dividend_date"] = (
                                str(ex_div.date()) if hasattr(ex_div, "date") else str(ex_div)
                            )
                        # Dividend date
                        div_date = calendar.get("Dividend Date")
                        if div_date is not None:
                            event_item["dividend_date"] = (
                                str(div_date.date()) if hasattr(div_date, "date") else str(div_date)
                            )
                except Exception as e:
                    logger.debug(f"Error fetching calendar for {normalized}: {e}")
                
                # Fetch corporate actions (dividends, splits)
                try:
                    actions = ticker.actions
                    if actions is not None and not actions.empty:
                        for date_idx, row in actions.iterrows():
                            div = float(row.get("Dividends", 0))
                            split = float(row.get("Stock Splits", 0))
                            if div > 0:
                                event_item["corporate_actions"].append({
                                    "date": date_idx.strftime("%Y-%m-%d"),
                                    "type": "dividend",
                                    "value": div,
                                })
                            if split > 0:
                                event_item["corporate_actions"].append({
                                    "date": date_idx.strftime("%Y-%m-%d"),
                                    "type": "split",
                                    "value": split,
                                })
                except Exception as e:
                    logger.debug(f"Error fetching actions for {normalized}: {e}")
                
                items.append(event_item)
                logger.debug(f"Fetched corporate events for {normalized}")
                
            except Exception as e:
                logger.error(f"Error fetching events for {symbol}: {e}")
                result.add_error(f"Events error for {symbol}: {str(e)}")
        
        result.add_data(DataType.EVENTS, items)

    # ------------------------------------------------------------------
    # Typed builders — return StockDataContext / IndexData directly
    # ------------------------------------------------------------------

    def fetch_index_data(
        self,
        index_symbol: str = "^NSEI",
        period: str = "2y",
        interval: str = "1d",
    ) -> "IndexData":
        """Fetch index historical data and return a typed IndexData object.

        Call this ONCE per run and share the result across all stocks.

        Args:
            index_symbol: Yahoo Finance ticker for the index (default: NIFTY 50)
            period: History length, e.g. "2y", "1y"
            interval: Bar interval, e.g. "1d"

        Returns:
            IndexData with historical_ohlc, last_close, last_volume, last_trading_date
        """
        from .data_context import IndexData, PriceData

        ticker = yf.Ticker(index_symbol)
        hist = ticker.history(period=period, interval=interval)

        ohlc: List[PriceData] = []
        for date_idx, row in hist.iterrows():
            ohlc.append(PriceData(
                symbol=index_symbol,
                date=date_idx.strftime("%Y-%m-%d"),
                open=float(row.get("Open", 0)),
                high=float(row.get("High", 0)),
                low=float(row.get("Low", 0)),
                close=float(row.get("Close", 0)),
                volume=int(row.get("Volume", 0)),
            ))

        last_close = ohlc[-1].close if ohlc else None
        last_volume = ohlc[-1].volume if ohlc else None
        last_trading_date = ohlc[-1].date if ohlc else None

        logger.info(
            f"Fetched index data for {index_symbol}: "
            f"{len(ohlc)} bars, last_close={last_close}"
        )
        return IndexData(
            index_symbol=index_symbol,
            historical_ohlc=ohlc,
            last_close=last_close,
            last_volume=last_volume,
            last_trading_date=last_trading_date,
        )

    def fetch_stock_context(
        self,
        symbol: str,
        index_data: Optional[Any] = None,
        period: str = "2y",
        interval: str = "1d",
    ) -> "StockDataContext":
        """Build a complete StockDataContext for a single symbol.

        Fetches fresh data every call (no caching). Designed to be called
        per-stock inside a run loop after index data is fetched once.

        Args:
            symbol: Stock ticker without exchange suffix (e.g. "TCS")
            index_data: Optional IndexData returned by fetch_index_data()
            period: History length, e.g. "2y"
            interval: Bar interval, e.g. "1d"

        Returns:
            StockDataContext with all fields populated
        """
        from .data_context import (
            StockDataContext, PriceData, FundamentalData,
            NewsItem, EventData,
        )

        normalized = self._normalize_symbol(symbol)
        ticker = yf.Ticker(normalized)

        # --- 1. Historical OHLCV ---
        hist = ticker.history(period=period, interval=interval)
        ohlc: List[PriceData] = []
        for date_idx, row in hist.iterrows():
            ohlc.append(PriceData(
                symbol=symbol,
                date=date_idx.strftime("%Y-%m-%d"),
                open=float(row.get("Open", 0)),
                high=float(row.get("High", 0)),
                low=float(row.get("Low", 0)),
                close=float(row.get("Close", 0)),
                volume=int(row.get("Volume", 0)),
            ))

        # Price snapshot from last two bars
        last_close = last_open = last_high = last_low = previous_close = None
        last_volume: Optional[int] = None
        last_trading_date: Optional[str] = None
        if len(ohlc) >= 1:
            last = ohlc[-1]
            last_close = last.close
            last_open = last.open
            last_high = last.high
            last_low = last.low
            last_volume = last.volume
            last_trading_date = last.date
        if len(ohlc) >= 2:
            previous_close = ohlc[-2].close

        # --- 2. Company info + fundamentals ---
        info: Dict[str, Any] = {}
        try:
            info = ticker.info or {}
        except Exception as e:
            logger.warning(f"Could not fetch .info for {normalized}: {e}")

        fundamentals = FundamentalData(
            symbol=symbol,
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            eps=info.get("trailingEps"),
            book_value=info.get("bookValue"),
            revenue_growth_yoy=info.get("revenueGrowth"),
            return_on_equity=info.get("returnOnEquity"),
            market_cap=info.get("marketCap"),
            price_to_book=info.get("priceToBook"),
            debt_to_equity=info.get("debtToEquity"),
            profit_margin=info.get("profitMargins"),
            dividend_yield=info.get("dividendYield"),
            sector=info.get("sector"),
            industry=info.get("industry"),
            analyst_rating=info.get("recommendationKey"),
            price_target=info.get("targetMeanPrice"),
        )

        # --- 3. News ---
        news_items: List[NewsItem] = []
        try:
            raw_news = ticker.news or []
            for article in raw_news:
                date_str = ""
                if article.get("providerPublishTime"):
                    try:
                        date_str = datetime.fromtimestamp(
                            article["providerPublishTime"]
                        ).strftime("%Y-%m-%d")
                    except Exception:
                        pass
                news_items.append(NewsItem(
                    symbol=symbol,
                    date=date_str,
                    headline=article.get("title"),
                    source=article.get("publisher"),
                    url=article.get("link"),
                ))
        except Exception as e:
            logger.warning(f"Could not fetch news for {normalized}: {e}")

        # --- 4. Corporate events (earnings date + actions) ---
        earnings_date: Optional[str] = None
        recent_corporate_actions: List[Dict[str, Any]] = []
        try:
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                dates = calendar.get("Earnings Date")
                if dates is not None and len(dates) > 0:
                    ed = dates[0]
                    earnings_date = (
                        str(ed.date()) if hasattr(ed, "date") else str(ed)
                    )
        except Exception:
            pass

        try:
            actions = ticker.actions
            if actions is not None and not actions.empty:
                for date_idx, row in actions.iterrows():
                    div = float(row.get("Dividends", 0))
                    split = float(row.get("Stock Splits", 0))
                    if div > 0:
                        recent_corporate_actions.append({
                            "date": date_idx.strftime("%Y-%m-%d"),
                            "type": "dividend",
                            "value": div,
                        })
                    if split > 0:
                        recent_corporate_actions.append({
                            "date": date_idx.strftime("%Y-%m-%d"),
                            "type": "split",
                            "value": split,
                        })
        except Exception:
            pass

        event_data: List[EventData] = []
        if earnings_date or recent_corporate_actions:
            event_data.append(EventData(
                symbol=symbol,
                date=earnings_date or "",
                event_type="corporate",
                description="Corporate events from ticker.calendar / ticker.actions",
                earnings_date=earnings_date,
                recent_corporate_actions=recent_corporate_actions,
            ))

        ctx = StockDataContext(
            symbol=symbol,
            exchange=info.get("exchange", ""),
            company_name=info.get("longName") or info.get("shortName") or "",
            sector=info.get("sector") or "",
            industry=info.get("industry") or "",
            isin=info.get("isin"),
            currency=info.get("currency") or "INR",
            last_close=last_close,
            previous_close=previous_close,
            last_open=last_open,
            last_high=last_high,
            last_low=last_low,
            last_volume=last_volume,
            last_trading_date=last_trading_date,
            historical_ohlc=ohlc,
            index_data=index_data,
            fundamentals=fundamentals,
            news_items=news_items,
            event_data=event_data,
        )
        logger.info(f"Built StockDataContext: {ctx.summary()}")
        return ctx

    def download_bulk(
        self, 
        symbols: List[str], 
        period: str = "1mo",
        interval: str = "1d",
        **kwargs
    ) -> Optional[Any]:
        """Download data for multiple symbols at once (more efficient).
        
        Uses yf.download() for bulk operations.
        
        Returns:
            pandas DataFrame with MultiIndex columns (if multiple symbols)
        """
        try:
            normalized = [self._normalize_symbol(s) for s in symbols]
            data = yf.download(
                normalized, 
                period=period, 
                interval=interval,
                group_by="ticker",
                auto_adjust=True,
                **kwargs
            )
            return data
        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            return None


# Legacy alias for backward compatibility
YahooFinanceDataSource = YahooFinanceProvider


def is_yfinance_available() -> bool:
    """Check if yfinance is available."""
    return _YFINANCE_AVAILABLE
