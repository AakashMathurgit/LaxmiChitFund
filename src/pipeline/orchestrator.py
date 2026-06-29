"""Pipeline orchestrator for LCF.

All four agents are active:
  - TechnicalAgent   (reads ohlc_daily, volume_daily, 52w high/low)
  - FundamentalAgent (reads pe_ratio, revenue_growth, profit_margin, …)
  - SentimentAgent   (reads news_articles list + recent_price_change)
  - EventAgent       (reads earnings_date, gap data, dividend/split info)
  - JudgeAgent       (ML / rule-based final decision)
"""

from typing import Any, Dict, List, Optional
import uuid
import yaml
import os

from ..utils.logger import get_logger
from ..main.agents.adapters.llm_adapter import LLMAdapter
from ..main.agents.technical_agent import TechnicalAgent
from ..main.agents.fundamental_agent import FundamentalAgent
from ..main.agents.sentiment_agent import SentimentAgent
from ..main.agents.event_agent import EventAgent
from ..main.agents.judge_agent import JudgeAgent
from ..main.agents.regime_detector_agent import RegimeDetectorAgent
from ..main.agents.trade_planner_agent import TradePlannerAgent
from ..main.agents.risk_manager_agent import RiskManagerAgent
from ..main.agents.position_management_agent import PositionManagementAgent
from ..main.agents.bull_agent import BullAgent
from ..main.agents.bear_agent import BearAgent
from ..main.agents.debate_agent import DebateAgent
from ..main.agents.future_prediction_agent import FuturePredictionAgent
from ..main.agents.interfaces.agent import AgentContext
from ..main.agents.interfaces.signals import (
    AgentFeatureBundle,
    TechnicalSignal,
    FundamentalSignal,
    SentimentSignal,
    EventSignal,
    RegimeSignal,
    MarketRegime,
    VolatilityState,
)
from ..main.controllers.data_processor import DataProcessor
from ..main.controllers.data_context import StockDataContext
from ..main.controllers.data_processor_agent import DataProcessorAgent, DataProcessorAgentConfig

from ..main.controllers.pattern_store import PatternStore
from ..main.controllers.trade_memory import TradeMemory


# ---------------------------------------------------------------------------
# Neutral default signals (used only when an agent hard-fails)
# ---------------------------------------------------------------------------

def _default_technical_signal() -> TechnicalSignal:
    return TechnicalSignal(
        technical_score=0.5,
        rsi=50.0,
        macd_signal="neutral",
        volatility=0.5,
        breakout_flag=False,
        trend_direction="neutral",
    )


def _default_fundamental_signal() -> FundamentalSignal:
    return FundamentalSignal(
        fundamental_score=0.5,
        valuation_label="fair",
        growth_score=0.5,
        financial_health_score=0.5,
    )


def _default_sentiment_signal() -> SentimentSignal:
    return SentimentSignal(
        sentiment_score=0.5,
        positive_news_count=0,
        negative_news_count=0,
        sentiment_trend="stable",
        news_confidence_score=0.0,
    )


def _default_event_signal() -> EventSignal:
    return EventSignal(
        event_score=0.0,
        earnings_impact_flag=False,
        event_risk_level="low",
        gap_up_flag=False,
        gap_down_flag=False,
    )


class PipelineOrchestrator:
    """Orchestrates all four agents and the ML Judge."""


    def __init__(self, config_path: Optional[str] = None, mode: str = "adaptive"):
        self.logger = get_logger("PipelineOrchestrator")

        # Set config base path for data storage
        if config_path:
            self._config_base_path = os.path.dirname(os.path.abspath(config_path))
        else:
            self._config_base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        self.config = self._load_config(config_path)

        # --- Mode/tuning config ---
        from src.main.config import load_mode_tuning_config, get_mode_config
        self.mode = mode
        self._tuning = load_mode_tuning_config(self.mode)
        self._mode_config = get_mode_config(self.mode)

        # --- LLM adapter (Azure OpenAI) ---
        self.llm = LLMAdapter.from_config(self.config, base_path=self._config_base_path)
        self.logger.info(f"LLM adapter ready: {self.config.get('llm', {}).get('model', 'gpt-4.1')}")

        # --- All agents ---
        self.technical = TechnicalAgent(config=self.config)
        self.fundamental = FundamentalAgent(config=self.config)
        self.sentiment = SentimentAgent(config=self.config)
        self.event = EventAgent(config=self.config)
        self.judge = JudgeAgent(
            config=self.config,
            model_path=self.config.get("judge", {}).get("model_path"),
        )

        # --- DataProcessor (for run_for_symbols flow) ---
        self.data_processor = self._init_data_processor()

        self.logger.info(
            f"Mode params: buy_threshold={self._tuning.judge.buy_threshold}, "
            f"rule_weight={self._tuning.debate.rule_weight}, "
            f"max_positions={self._mode_config.max_positions}, "
            f"hold_days={self._mode_config.hold_days}"
        )

        # --- RAG: PatternStore (ChromaDB, local) ---
        pattern_dir = os.path.join(self._config_base_path, "data", "pattern_store")
        self.pattern_store = PatternStore(persist_dir=pattern_dir)

        # --- Memory: TradeMemory (append-only JSONL) ---
        memory_path = os.path.join(self._config_base_path, "data", "trade_memory.jsonl")
        self.trade_memory = TradeMemory(file_path=memory_path)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _init_data_processor(self) -> DataProcessor:
        """Create a DataProcessor with Yahoo Finance provider configured for India."""
        from ..main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig

        processor = DataProcessor()
        yf_config = YFinanceConfig(
            default_exchange_suffix=self.config.get("data", {}).get("exchange_suffix", ".NS"),
            price_period="2y",
            price_interval="1d",
        )
        processor.register_provider(YahooFinanceProvider(yf_config))
        
        # NSE Corporate provider for official corporate actions/announcements
        try:
            from ..main.controllers.nse_corporate_provider import NSECorporateProvider, NSECorporateConfig
            
            # Resolve base path for NSE data
            nse_base_path = os.path.dirname(self._config_path) if hasattr(self, '_config_path') else None
            nse_config = NSECorporateConfig(
                lookback_days_actions=30,
                lookback_days_announcements=7,
                include_upcoming_events=True,
                upcoming_days=14,
            )
            processor.register_provider(NSECorporateProvider(nse_config, base_path=nse_base_path))
            self.logger.info("NSE Corporate provider registered")
        except Exception as e:
            self.logger.warning(f"NSE Corporate provider not available: {e}")
        
        # RSS News provider for real-time news from multiple feeds
        try:
            from ..main.controllers.rss_news_provider import RSSNewsProvider, RSSNewsConfig
            
            rss_config = RSSNewsConfig(
                only_significant_news=True,
                max_news_age_hours=72,
            )
            processor.register_provider(RSSNewsProvider(rss_config))
            self.logger.info("RSS News provider registered")
        except Exception as e:
            self.logger.warning(f"RSS News provider not available: {e}")
        
        return processor

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        if config_path is None:
            config_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "config.yaml",
            ))
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            self.logger.warning(f"Config not found: {config_path}, using defaults")
            return {}

    # ------------------------------------------------------------------
    # Pipeline execution helpers
    # ------------------------------------------------------------------

    def _build_context(self, input_data: Dict[str, Any]) -> AgentContext:
        return AgentContext(
            run_id=str(uuid.uuid4()),
            rules_version=self.config.get("version", "1.0.0"),
            input_data=input_data,
            config=self.config,
            llm=self.llm,
        )

    @staticmethod
    def _build_input_data(stock_ctx: StockDataContext) -> Dict[str, Any]:
        """Flatten StockDataContext into a flat dict for AgentContext.input_data."""
        symbol = stock_ctx.symbol
        date = stock_ctx.last_trading_date or ""

        # --- Technical: raw OHLCV ---
        ohlc_daily = [
            {"open": p.open, "high": p.high, "low": p.low, "close": p.close}
            for p in stock_ctx.historical_ohlc
        ]
        volume_daily = [p.volume for p in stock_ctx.historical_ohlc]

        # 52-week high/low from most recent 252 bars
        recent_252 = stock_ctx.historical_ohlc[-252:] if stock_ctx.historical_ohlc else []
        week52_high = max((p.high for p in recent_252), default=None) if recent_252 else None
        week52_low = min((p.low for p in recent_252), default=None) if recent_252 else None

        # --- Sentiment: news_articles list ---
        news_articles = [
            {
                "headline": n.headline or "",
                "summary": n.news_text or "",
                "date": n.date,
                "source": n.source or "",
            }
            for n in stock_ctx.news_items
        ]

        # --- recent_price_change ---
        if (
            stock_ctx.last_close
            and stock_ctx.previous_close
            and stock_ctx.previous_close != 0
        ):
            recent_price_change = (
                (stock_ctx.last_close - stock_ctx.previous_close)
                / stock_ctx.previous_close
            )
        else:
            recent_price_change = 0.0

        # --- Event: earnings, dividends, splits, gap ---
        first_event = stock_ctx.event_data[0] if stock_ctx.event_data else None
        earnings_date = first_event.earnings_date if first_event else None

        dividend_info = None
        stock_split_info = None
        if first_event:
            for action in (first_event.recent_corporate_actions or []):
                action_type = str(action.get("type", "")).lower()
                if action_type == "dividend" and dividend_info is None:
                    dividend_info = action
                elif action_type in ("split", "stock_split") and stock_split_info is None:
                    stock_split_info = action

        gap_pct = 0.0
        if (
            stock_ctx.last_open
            and stock_ctx.previous_close
            and stock_ctx.previous_close != 0
        ):
            gap_pct = (stock_ctx.last_open - stock_ctx.previous_close) / stock_ctx.previous_close
        recent_gap_data = {"gap_pct": gap_pct}

        major_news_flag = len(stock_ctx.news_items) > 5

        # --- Fundamentals ---
        f = stock_ctx.fundamentals

        return {
            "symbol": symbol,
            "date": date,
            # Technical inputs
            "ohlc_daily": ohlc_daily,
            "volume_daily": volume_daily,
            "latest_price": stock_ctx.last_close,
            "52_week_high": week52_high,
            "52_week_low": week52_low,
            # Fundamental inputs
            "pe_ratio": f.pe_ratio if f else None,
            "forward_pe": f.forward_pe if f else None,
            "eps": f.eps if f else None,
            "revenue_growth": f.revenue_growth_yoy if f else None,
            "profit_margin": f.profit_margin if f else None,
            "debt_to_equity": f.debt_to_equity if f else None,
            "roe": f.return_on_equity if f else None,
            "market_cap": f.market_cap if f else None,
            "sector": f.sector if f else None,
            # Sentiment inputs
            "news_articles": news_articles,
            "recent_price_change": recent_price_change,
            # Event inputs
            "earnings_date": earnings_date,
            "earnings_results": stock_ctx.computed_metrics.get("earnings_results"),
            "dividend_info": dividend_info,
            "stock_split_info": stock_split_info,
            "recent_gap_data": recent_gap_data,
            "major_news_flag": major_news_flag,
            # Pass-through computed metrics (similarity scores, regime, etc.)
            **stock_ctx.computed_metrics,
        }

    # ------------------------------------------------------------------
    # Primary entry point: StockDataContext → decision
    # ------------------------------------------------------------------

    def analyse_stock_context(self, stock_ctx: StockDataContext) -> Dict[str, Any]:
        """Run the full agent pipeline on a pre-built StockDataContext.

        Parameters
        ----------
        stock_ctx : StockDataContext
            Fully populated context built by DataProcessor.build_stock_context()

        Returns
        -------
        dict with keys: symbol, date, technical, fundamental, sentiment, event, judge_decision
        """
        symbol = stock_ctx.symbol
        date = stock_ctx.last_trading_date or ""
        self.logger.info(f"[{symbol}] Starting analysis (StockDataContext)")

        input_data = self._build_input_data(stock_ctx)
        ctx = self._build_context(input_data)

        # --- Run all four agents with StockDataContext ---
        tech_result = self.technical.run(ctx, stock_context=stock_ctx)
        tech_signal = tech_result.payload.get("raw_signal") or _default_technical_signal()

        fund_result = self.fundamental.run(ctx, stock_context=stock_ctx)
        fund_signal = fund_result.payload.get("raw_signal") or _default_fundamental_signal()

        sent_result = self.sentiment.run(ctx, stock_context=stock_ctx)
        sent_signal = sent_result.payload.get("raw_signal") or _default_sentiment_signal()

        evt_result = self.event.run(ctx, stock_context=stock_ctx)
        evt_signal = evt_result.payload.get("raw_signal") or _default_event_signal()

        # --- Regime placeholder ---
        regime = RegimeSignal(
            market_regime=MarketRegime(
                stock_ctx.flags.get("market_regime", "sideways")
            ),
            volatility_state=VolatilityState(
                stock_ctx.flags.get("volatility_state", "moderate")
            ),
            regime_confidence=stock_ctx.flags.get("regime_confidence", 0.5),
        )

        bundle = AgentFeatureBundle(
            symbol=symbol,
            date=date,
            technical=tech_signal,
            fundamental=fund_signal,
            sentiment=sent_signal,
            event=evt_signal,
            regime=regime,
            similarity_avg_return=stock_ctx.computed_metrics.get("similarity_avg_return", 0.0),
            similarity_positive_rate=stock_ctx.computed_metrics.get("similarity_positive_rate", 0.0),
            similarity_max_drawdown=stock_ctx.computed_metrics.get("similarity_max_drawdown", 0.0),
        )

        judge_result = self.judge.run(ctx, bundle=bundle)

        self.logger.info(
            f"[{symbol}] Decision: {judge_result.payload.get('decision')} "
            f"| P(up5d)={judge_result.payload.get('prob_up_5d')} "
            f"| Confidence={judge_result.payload.get('confidence')}"
        )

        return {
            "symbol": symbol,
            "date": date,
            "technical": tech_result.to_dict(),
            "fundamental": fund_result.to_dict(),
            "sentiment": sent_result.to_dict(),
            "event": evt_result.to_dict(),
            "judge_decision": judge_result.to_dict(),
        }

    # ------------------------------------------------------------------
    # Batch entry point: symbol list → results list
    # ------------------------------------------------------------------

    def run_for_symbols(
        self,
        symbols: List[str],
        index_symbol: str = "^NSEI",
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline for a list of stock symbols.

        Fetches index data ONCE, then builds a StockDataContext per symbol
        and runs analyse_stock_context() for each.

        Parameters
        ----------
        symbols : list of str
            Stock tickers without exchange suffix, e.g. ["TCS", "INFY"]
        index_symbol : str
            Yahoo Finance index ticker to use as market context

        Returns
        -------
        list of result dicts (one per symbol)
        """
        self.logger.info(f"run_for_symbols: {len(symbols)} stocks, index={index_symbol}")

        index_data = self.data_processor.build_index_context(index_symbol=index_symbol)

        results = []
        for symbol in symbols:
            try:
                stock_ctx = self.data_processor.build_stock_context(
                    symbol=symbol,
                    index_data=index_data,
                )
                result = self.analyse_stock_context(stock_ctx)
                results.append(result)
            except Exception as e:
                self.logger.error(f"[{symbol}] Pipeline error: {e}")
                results.append({"symbol": symbol, "error": str(e)})

        self.logger.info("run_for_symbols complete")
        return results

    # ------------------------------------------------------------------
    # Stock Discovery — auto-select symbols
    # ------------------------------------------------------------------

    def discover_symbols(
        self,
        universe: Optional[List[str]] = None,
        max_stocks: int = 10,
        index_symbol: str = "^NSEI",
    ) -> List[Dict[str, Any]]:
        """Auto-discover top stock candidates by screening the universe.

        Fetches quick market snapshots for each stock in the universe and
        scores them by volume spikes, price breakouts, news activity, etc.

        Parameters
        ----------
        universe : list of str, optional
            Stocks to screen. Uses NIFTY 50 default if not provided.
        max_stocks : int
            Maximum number of candidates to return.
        index_symbol : str
            Market index for context.

        Returns
        -------
        list of dicts with 'symbol', 'score', 'reasons' keys
        """
        self.logger.info(f"discover_symbols: screening {len(universe or [])} stocks (max={max_stocks})")

        # Build quick market data for screening
        market_data = {}
        symbols_to_screen = universe or self.discovery_agent._config.default_universe

        for symbol in symbols_to_screen:
            try:
                stock_ctx = self.data_processor.build_stock_context(
                    symbol=symbol,
                    index_data=None,
                )
                if stock_ctx is None:
                    continue

                # Extract screening data from StockDataContext
                bars = stock_ctx.price_data or []
                last_close = stock_ctx.last_close or 0
                volume = bars[-1].volume if bars else 0
                prev_close = stock_ctx.previous_close or last_close

                # 20-day average volume
                recent_vols = [b.volume for b in bars[-20:] if b.volume]
                avg_volume = sum(recent_vols) / len(recent_vols) if recent_vols else 0

                # 52-week high/low from last 252 bars
                recent_252 = bars[-252:] if len(bars) >= 252 else bars
                week52_high = max((b.high for b in recent_252), default=0) if recent_252 else 0
                week52_low = min((b.low for b in recent_252), default=0) if recent_252 else 0

                # Price change %
                price_change_pct = ((last_close - prev_close) / prev_close) if prev_close else 0

                # News count
                news_count = len(stock_ctx.news_items or [])

                market_data[symbol] = {
                    "last_close": last_close,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "52_week_high": week52_high,
                    "52_week_low": week52_low,
                    "price_change_pct": price_change_pct,
                    "news_count": news_count,
                }
            except Exception as e:
                self.logger.debug(f"[{symbol}] Discovery screen skip: {e}")

        # Run discovery agent
        self.discovery_agent._config.max_candidates = max_stocks
        candidates = self.discovery_agent.discover(
            universe=symbols_to_screen,
            market_data=market_data,
        )

        self.logger.info(f"discover_symbols: {len(candidates)} candidates found")
        return [
            {"symbol": c.symbol, "score": c.score, "reasons": c.reasons}
            for c in candidates
        ]

    # ------------------------------------------------------------------
    # Legacy entry point: pre-built dict list
    # ------------------------------------------------------------------

    def analyse_stock(self, stock_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline for a single stock snapshot passed as a plain dict.

        Useful for testing or when StockDataContext is not available.
        Agents will run with whatever fields are present; missing fields
        will produce neutral/default sub-scores.

        Parameters
        ----------
        stock_data : dict
            Must contain at minimum ``symbol`` and ``date``.
            Pass ``news_articles`` for meaningful sentiment output.
            Pass ``ohlc_daily`` for meaningful technical output.
        """
        symbol = stock_data.get("symbol", "UNKNOWN")
        date = stock_data.get("date", "")
        self.logger.info(f"[{symbol}] Starting analysis for {date}")

        ctx = self._build_context(stock_data)

        tech_result = self.technical.run(ctx)
        tech_signal = tech_result.payload.get("raw_signal") or _default_technical_signal()

        fund_result = self.fundamental.run(ctx)
        fund_signal = fund_result.payload.get("raw_signal") or _default_fundamental_signal()

        sent_result = self.sentiment.run(ctx)
        sent_signal = sent_result.payload.get("raw_signal") or _default_sentiment_signal()

        evt_result = self.event.run(ctx)
        evt_signal = evt_result.payload.get("raw_signal") or _default_event_signal()

        regime = RegimeSignal(
            market_regime=MarketRegime(stock_data.get("market_regime", "sideways")),
            volatility_state=VolatilityState(stock_data.get("volatility_state", "moderate")),
            regime_confidence=stock_data.get("regime_confidence", 0.5),
        )

        bundle = AgentFeatureBundle(
            symbol=symbol,
            date=date,
            technical=tech_signal,
            fundamental=fund_signal,
            sentiment=sent_signal,
            event=evt_signal,
            regime=regime,
            similarity_avg_return=stock_data.get("similarity_avg_return", 0.0),
            similarity_positive_rate=stock_data.get("similarity_positive_rate", 0.0),
            similarity_max_drawdown=stock_data.get("similarity_max_drawdown", 0.0),
        )

        judge_result = self.judge.run(ctx, bundle=bundle)

        self.logger.info(
            f"[{symbol}] Decision: {judge_result.payload.get('decision')} "
            f"| P(up5d)={judge_result.payload.get('prob_up_5d')} "
            f"| Confidence={judge_result.payload.get('confidence')}"
        )

        return {
            "symbol": symbol,
            "date": date,
            "technical": tech_result.to_dict(),
            "fundamental": fund_result.to_dict(),
            "sentiment": sent_result.to_dict(),
            "event": evt_result.to_dict(),
            "judge_decision": judge_result.to_dict(),
        }

    def run(self, stock_list: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Run pipeline for a list of stock snapshots.

        If *stock_list* is ``None``, loads from configured data source (Phase 2).
        """
        if stock_list is None:
            self.logger.warning("No stock_list provided — nothing to analyse")
            return []

        self.logger.info(f"Analysing {len(stock_list)} stocks…")
        results = [self.analyse_stock(s) for s in stock_list]
        self.logger.info("Pipeline run complete")
        return results

    # ------------------------------------------------------------------
    # New architecture: DataProcessorAgent entry point
    # ------------------------------------------------------------------

    def run_with_data_processor_agent(
        self,
        symbols: List[str],
        index_symbol: str = "^NSEI",
    ) -> List[Dict[str, Any]]:
        """Run the full pipeline using the new DataProcessorAgent architecture.

        DataProcessorAgent handles:
        - Registering multiple providers (Yahoo Finance, News, etc.)
        - Fetching index data ONCE and caching
        - Building StockDataContext for each symbol

        Parameters
        ----------
        symbols : list of str
            Stock tickers without exchange suffix, e.g. ["TCS", "INFY"]
        index_symbol : str
            Yahoo Finance index ticker to use as market context

        Returns
        -------
        list of result dicts (one per symbol)
        """
        from ..main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig
        from ..main.controllers.news_data_provider import NewsDataProvider, NewsDataProviderConfig

        self.logger.info(f"run_with_data_processor_agent: {len(symbols)} stocks, index={index_symbol}")

        # Create DataProcessorAgent with config
        agent_config = DataProcessorAgentConfig(
            index_symbol=index_symbol,
            historical_days=500,  # ~2 years
            news_lookback_hours=72,
        )
        processor_agent = DataProcessorAgent(agent_config)

        # Register providers
        exchange_suffix = self.config.get("data", {}).get("exchange_suffix", ".NS")
        yf_config = YFinanceConfig(
            default_exchange_suffix=exchange_suffix,
            price_period="2y",
            price_interval="1d",
        )
        processor_agent.register_provider(YahooFinanceProvider(yf_config))

        # Register news provider (optional, may not be available)
        try:
            news_config = NewsDataProviderConfig(
                lookback_hours=72,
                max_articles_per_symbol=20,
            )
            processor_agent.register_provider(NewsDataProvider(news_config))
        except ImportError:
            self.logger.warning("NewsDataProvider not available (requests not installed)")

        # Build contexts for all symbols
        contexts = processor_agent.build_contexts(symbols)

        # Analyse each context
        results = []
        for stock_ctx in contexts.values():
            try:
                result = self.analyse_stock_context(stock_ctx)
                results.append(result)
            except Exception as e:
                self.logger.error(f"[{stock_ctx.symbol}] Pipeline error: {e}")
                results.append({"symbol": stock_ctx.symbol, "error": str(e)})

        self.logger.info("run_with_data_processor_agent complete")
        return results

    # ------------------------------------------------------------------
    # Future Prediction: multi-horizon price forecasts
    # ------------------------------------------------------------------

    def predict_future(self, stock_ctx: StockDataContext) -> Dict[str, Any]:
        """Run all agents, then generate 4-horizon price predictions.

        Runs: Technical → Fundamental → Sentiment → Event → Regime →
              Bundle all signals → FuturePredictionAgent (LLM)

        Returns dict with all agent results + prediction.
        """
        symbol = stock_ctx.symbol
        date = stock_ctx.last_trading_date or ""
        self.logger.info(f"[{symbol}] Starting future prediction")

        input_data = self._build_input_data(stock_ctx)
        ctx = self._build_context(input_data)

        # Run all analysis agents
        tech_result = self.technical.run(ctx, stock_context=stock_ctx)
        tech_signal = tech_result.payload.get("raw_signal") or _default_technical_signal()

        fund_result = self.fundamental.run(ctx, stock_context=stock_ctx)
        fund_signal = fund_result.payload.get("raw_signal") or _default_fundamental_signal()

        sent_result = self.sentiment.run(ctx, stock_context=stock_ctx)
        sent_signal = sent_result.payload.get("raw_signal") or _default_sentiment_signal()

        evt_result = self.event.run(ctx, stock_context=stock_ctx)
        evt_signal = evt_result.payload.get("raw_signal") or _default_event_signal()

        regime = RegimeSignal(
            market_regime=MarketRegime(
                stock_ctx.flags.get("market_regime", "sideways")
            ),
            volatility_state=VolatilityState(
                stock_ctx.flags.get("volatility_state", "moderate")
            ),
            regime_confidence=stock_ctx.flags.get("regime_confidence", 0.5),
        )

        bundle = AgentFeatureBundle(
            symbol=symbol,
            date=date,
            technical=tech_signal,
            fundamental=fund_signal,
            sentiment=sent_signal,
            event=evt_signal,
            regime=regime,
            similarity_avg_return=stock_ctx.computed_metrics.get("similarity_avg_return", 0.0),
            similarity_positive_rate=stock_ctx.computed_metrics.get("similarity_positive_rate", 0.0),
            similarity_max_drawdown=stock_ctx.computed_metrics.get("similarity_max_drawdown", 0.0),
        )

        # Run future prediction agent
        prediction_agent = FuturePredictionAgent(config=self.config)
        prediction = prediction_agent.predict(
            stock_ctx=stock_ctx,
            llm=self.llm,
            bundle=bundle,
            debate_ctx=None,
        )

        self.logger.info(
            f"[{symbol}] Prediction complete: outlook={prediction.overall_outlook}, "
            f"confidence={prediction.overall_confidence:.0%}"
        )

        return {
            "symbol": symbol,
            "date": date,
            "current_price": stock_ctx.last_close,
            "technical": tech_result.to_dict(),
            "fundamental": fund_result.to_dict(),
            "sentiment": sent_result.to_dict(),
            "event": evt_result.to_dict(),
            "regime": {
                "regime": regime.market_regime.value,
                "vol_state": regime.volatility_state.value,
                "confidence": regime.regime_confidence,
            },
            "prediction": prediction.to_dict(),
        }

    def predict_future_for_symbols(
        self,
        symbols: List[str],
        index_symbol: str = "^GSPC",
    ) -> List[Dict[str, Any]]:
        """Run future predictions for a list of symbols.

        Fetches index data once, builds StockDataContext per symbol,
        then runs predict_future() for each.
        """
        self.logger.info(f"predict_future_for_symbols: {len(symbols)} stocks, index={index_symbol}")

        index_data = self.data_processor.build_index_context(index_symbol=index_symbol)

        results = []
        for symbol in symbols:
            try:
                stock_ctx = self.data_processor.build_stock_context(
                    symbol=symbol,
                    index_data=index_data,
                )
                result = self.predict_future(stock_ctx)
                results.append(result)
            except Exception as e:
                self.logger.error(f"[{symbol}] Prediction error: {e}")
                results.append({"symbol": symbol, "error": str(e)})

        self.logger.info("predict_future_for_symbols complete")
        return results
