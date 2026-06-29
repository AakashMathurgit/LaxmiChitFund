"""Test script for the new DataProcessorAgent architecture.

Run this script to verify the full data flow:
1. DataProcessorAgent fetches data from multiple providers
2. StockDataContext is built per symbol with shared IndexData
3. Each agent receives and processes StockDataContext

Usage:
    python scripts/test_data_processor_agent.py
    python scripts/test_data_processor_agent.py --symbols MSFT AAPL GOOG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir / "src"))

from main.controllers.data_processor_agent import DataProcessorAgent, DataProcessorAgentConfig
from main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig
from main.controllers.data_context import StockDataContext
from main.agents.fundamental_agent import FundamentalAgent
from main.agents.technical_agent import TechnicalAgent
from main.agents.sentiment_agent import SentimentAgent
from main.agents.event_agent import EventAgent
from main.agents.interfaces.agent import AgentContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("test_data_processor_agent")


def test_data_processor_agent(symbols: list[str], exchange_suffix: str = ""):
    """Test the DataProcessorAgent with real data."""
    
    logger.info("=" * 60)
    logger.info("Testing DataProcessorAgent Architecture")
    logger.info("=" * 60)
    
    # --- 1. Create DataProcessorAgent ---
    logger.info("\n[1] Creating DataProcessorAgent...")
    config = DataProcessorAgentConfig(
        index_symbol="^GSPC",  # S&P 500 for US stocks
        historical_days=500,
        news_lookback_hours=72,
    )
    processor = DataProcessorAgent(config)
    logger.info(f"    Index symbol: {config.index_symbol}")
    logger.info(f"    Historical days: {config.historical_days}")
    
    # --- 2. Register providers ---
    logger.info("\n[2] Registering data providers...")
    yf_config = YFinanceConfig(
        default_exchange_suffix=exchange_suffix,
        price_period="2y",
        price_interval="1d",
    )
    yahoo_provider = YahooFinanceProvider(yf_config)
    processor.register_provider(yahoo_provider)
    logger.info(f"    Registered: {yahoo_provider.name}")
    
    # Try to add news provider (optional)
    try:
        from main.controllers.news_data_provider import NewsDataProvider, NewsDataProviderConfig
        news_config = NewsDataProviderConfig(
            lookback_hours=72,
            max_articles_per_symbol=10,
        )
        news_provider = NewsDataProvider(news_config)
        processor.register_provider(news_provider)
        logger.info(f"    Registered: {news_provider.name}")
    except ImportError:
        logger.warning("    NewsDataProvider not available (requests not installed)")
    
    # --- 3. Build contexts ---
    logger.info(f"\n[3] Building contexts for {len(symbols)} symbols: {symbols}")
    contexts = processor.build_contexts(symbols)
    
    for symbol, ctx in contexts.items():
        logger.info(f"\n[{symbol}] StockDataContext Summary:")
        logger.info(f"    Company: {ctx.company_name}")
        logger.info(f"    Sector: {ctx.sector}")
        logger.info(f"    Industry: {ctx.industry}")
        logger.info(f"    Last Close: {ctx.last_close}")
        logger.info(f"    Historical bars: {len(ctx.historical_ohlc)}")
        logger.info(f"    News items: {len(ctx.news_items)}")
        logger.info(f"    Events: {len(ctx.event_data)}")
        if ctx.fundamentals:
            logger.info(f"    PE Ratio: {ctx.fundamentals.pe_ratio}")
            logger.info(f"    EPS: {ctx.fundamentals.eps}")
            logger.info(f"    Market Cap: {ctx.fundamentals.market_cap}")
        if ctx.index_data:
            logger.info(f"    Index: {ctx.index_data.index_symbol} @ {ctx.index_data.last_close}")
    
    # --- 4. Test agents with StockDataContext ---
    logger.info("\n[4] Testing agents with StockDataContext...")
    
    # Create agent instances
    fundamental_agent = FundamentalAgent()
    technical_agent = TechnicalAgent()
    sentiment_agent = SentimentAgent()
    event_agent = EventAgent()
    
    for symbol, ctx in contexts.items():
        logger.info(f"\n[{symbol}] Running agents:")
        
        # Create dummy AgentContext (agents will use stock_context instead)
        agent_ctx = AgentContext(
            run_id=f"test-{symbol}",
            rules_version="1.0.0",
            input_data={},
            config={},
            llm=None,
        )
        
        # Run FundamentalAgent
        fund_result = fundamental_agent.run(agent_ctx, stock_context=ctx)
        fund_signal = fund_result.payload.get("signal", {})
        logger.info(f"    Fundamental: score={fund_signal.get('fundamental_score')}, valuation={fund_signal.get('valuation_label')}")
        
        # Run TechnicalAgent
        tech_result = technical_agent.run(agent_ctx, stock_context=ctx)
        tech_signal = tech_result.payload.get("signal", {})
        logger.info(f"    Technical: score={tech_signal.get('technical_score')}, RSI={tech_signal.get('rsi')}, trend={tech_signal.get('trend_direction')}")
        
        # Run SentimentAgent
        sent_result = sentiment_agent.run(agent_ctx, stock_context=ctx)
        sent_signal = sent_result.payload.get("signal", {})
        logger.info(f"    Sentiment: score={sent_signal.get('sentiment_score')}, positive={sent_signal.get('positive_news_count')}, negative={sent_signal.get('negative_news_count')}")
        
        # Run EventAgent
        evt_result = event_agent.run(agent_ctx, stock_context=ctx)
        evt_signal = evt_result.payload.get("signal", {})
        logger.info(f"    Event: score={evt_signal.get('event_score')}, risk={evt_signal.get('event_risk_level')}, earnings_impact={evt_signal.get('earnings_impact_flag')}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Test completed successfully!")
    logger.info("=" * 60)
    return contexts


def test_orchestrator_integration(symbols: list[str]):
    """Test the full PipelineOrchestrator with new architecture."""
    
    logger.info("\n" + "=" * 60)
    logger.info("Testing PipelineOrchestrator Integration")
    logger.info("=" * 60)
    
    try:
        from pipeline.orchestrator import PipelineOrchestrator
        
        orchestrator = PipelineOrchestrator()
        
        logger.info(f"\nRunning run_with_data_processor_agent for: {symbols}")
        results = orchestrator.run_with_data_processor_agent(
            symbols=symbols,
            index_symbol="^GSPC",  # S&P 500
        )
        
        for result in results:
            symbol = result.get("symbol")
            if "error" in result:
                logger.error(f"[{symbol}] Error: {result['error']}")
            else:
                judge = result.get("judge_decision", {}).get("payload", {})
                logger.info(f"[{symbol}] Decision: {judge.get('decision')}, Confidence: {judge.get('confidence')}")
        
        logger.info("\nOrchestrator integration test complete!")
        return results
        
    except Exception as e:
        logger.error(f"Orchestrator integration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Test DataProcessorAgent architecture")
    parser.add_argument(
        "--symbols", 
        nargs="+", 
        default=["MSFT"],
        help="Stock symbols to test (default: MSFT)"
    )
    parser.add_argument(
        "--exchange", 
        default="",
        help="Exchange suffix (.NS for NSE, .BO for BSE, empty for US)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full orchestrator integration test"
    )
    
    args = parser.parse_args()
    
    try:
        # Basic test
        contexts = test_data_processor_agent(args.symbols, args.exchange)
        
        # Full orchestrator test (optional)
        if args.full:
            test_orchestrator_integration(args.symbols)
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
