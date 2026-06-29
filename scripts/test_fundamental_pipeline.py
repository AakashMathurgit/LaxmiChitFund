"""Test script for DataProcessor + FundamentalAgent integration.

This script demonstrates:
1. Fetching MSFT stock data from Yahoo Finance using the new DataProvider architecture
2. Mapping Yahoo Finance data to FundamentalAgent expected format
3. Running the FundamentalAgent with the fetched data

Usage:
    cd LCF
    python -m scripts.test_fundamental_pipeline
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from typing import Dict, Any

from src.main.controllers import (
    DataProcessor,
    DataType,
    DataProcessorConfig,
)
from src.main.controllers.yahoo_finance_provider import (
    YahooFinanceProvider,
    YFinanceConfig,
    is_yfinance_available,
)
from src.main.agents.fundamental_agent import FundamentalAgent
from src.main.agents.interfaces.agent import AgentContext
from src.utils.logger import get_logger

logger = get_logger("test_fundamental_pipeline")


def map_yahoo_to_fundamental_input(yahoo_data: Dict[str, Any]) -> Dict[str, Any]:
    """Map Yahoo Finance fundamentals to FundamentalAgent expected format.
    
    FundamentalAgent expects:
    - earnings_surprise_pct: Earnings surprise percentage (we use earnings_growth as proxy)
    - pe_vs_sector_ratio: P/E vs sector average (we use forward_pe / trailing_pe as proxy)
    - eps_revision_direction: EPS revision direction (-1 to 1)
    - revenue_growth_yoy: Year-over-year revenue growth
    
    Yahoo Finance provides:
    - pe_ratio (trailing P/E)
    - forward_pe
    - eps, forward_eps
    - earnings_growth, revenue_growth
    - recommendation_key, recommendation_mean
    """
    # Get values with defaults
    trailing_pe = yahoo_data.get("pe_ratio") or yahoo_data.get("trailing_pe") or 0
    forward_pe = yahoo_data.get("forward_pe") or trailing_pe or 0
    earnings_growth = yahoo_data.get("earnings_growth") or 0  # decimal (0.15 = 15%)
    revenue_growth = yahoo_data.get("revenue_growth") or yahoo_data.get("revenue_growth_yoy") or 0
    
    # Compute P/E vs "sector" - here we use forward vs trailing as proxy
    # If forward_pe < trailing_pe, analysts expect earnings growth (positive signal)
    pe_ratio = forward_pe / trailing_pe if trailing_pe and trailing_pe > 0 else 1.0
    
    # Map recommendation to EPS revision direction (-1 to 1)
    # recommendation_mean: 1=Strong Buy, 2=Buy, 3=Hold, 4=Sell, 5=Strong Sell
    rec_mean = yahoo_data.get("recommendation_mean") or 3.0
    eps_revision = (3.0 - rec_mean) / 2.0  # 1=Strong Buy -> +1, 5=Strong Sell -> -1
    
    # Convert revenue_growth from decimal to percentage
    revenue_growth_pct = revenue_growth * 100 if isinstance(revenue_growth, float) and revenue_growth < 1 else revenue_growth
    
    # Earnings surprise proxy: use earnings_growth as percentage
    earnings_surprise_pct = earnings_growth * 100 if isinstance(earnings_growth, float) and earnings_growth < 1 else earnings_growth
    
    return {
        "symbol": yahoo_data.get("symbol", ""),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "earnings_surprise_pct": earnings_surprise_pct,
        "pe_vs_sector_ratio": pe_ratio,
        "eps_revision_direction": eps_revision,
        "revenue_growth_yoy": revenue_growth_pct,
        # Pass through raw values for reference
        "raw_pe_ratio": trailing_pe,
        "raw_forward_pe": forward_pe,
        "raw_eps": yahoo_data.get("eps"),
        "raw_market_cap": yahoo_data.get("market_cap"),
        "raw_sector": yahoo_data.get("sector"),
        "raw_recommendation_key": yahoo_data.get("recommendation_key"),
    }


def run_test():
    """Run the test pipeline."""
    print("=" * 60)
    print("LCF - Fundamental Agent Test with Yahoo Finance Data")
    print("=" * 60)
    
    # Check if yfinance is available
    if not is_yfinance_available():
        print("\nERROR: yfinance is not installed.")
        print("Install with: pip install yfinance pandas")
        return
    
    print("\n[1] Setting up DataProcessor with YahooFinanceProvider...")
    
    # Configure Yahoo Finance for US stocks (no suffix)
    yf_config = YFinanceConfig(
        default_exchange_suffix="",  # Empty for US stocks like MSFT
        price_period="1mo",
        price_interval="1d",
    )
    
    # Create provider
    yf_provider = YahooFinanceProvider(yf_config)
    print(f"    Provider: {yf_provider.name}")
    print(f"    Supported types: {[t.name for t in yf_provider.supported_types]}")
    
    # Create DataProcessor and register provider
    processor = DataProcessor()
    processor.register_provider(yf_provider)
    
    # Fetch MSFT data
    print("\n[2] Fetching MSFT data from Yahoo Finance...")
    symbols = ["MSFT"]
    
    context = processor.process(
        symbols=symbols,
        data_types={DataType.FUNDAMENTALS, DataType.OHLCV, DataType.ANALYST}
    )
    
    print(f"\n    DataContext summary: {context.summary()}")
    print(f"    Price records: {len(context.price_data)}")
    print(f"    Fundamental records: {len(context.fundamental_data)}")
    
    # Get fundamental data
    if not context.fundamental_data:
        print("\nERROR: No fundamental data fetched for MSFT")
        return
    
    # Display raw Yahoo Finance data
    yahoo_fundamental = context.raw_sources.get("fundamentals", [{}])[0] if context.raw_sources.get("fundamentals") else {}
    print("\n[3] Raw Yahoo Finance fundamental data:")
    print(f"    Company: {yahoo_fundamental.get('company_name', 'N/A')}")
    print(f"    Sector: {yahoo_fundamental.get('sector', 'N/A')}")
    print(f"    Market Cap: ${yahoo_fundamental.get('market_cap', 0):,.0f}")
    print(f"    P/E Ratio: {yahoo_fundamental.get('pe_ratio', 'N/A')}")
    print(f"    Forward P/E: {yahoo_fundamental.get('forward_pe', 'N/A')}")
    print(f"    EPS: {yahoo_fundamental.get('eps', 'N/A')}")
    print(f"    Revenue Growth: {yahoo_fundamental.get('revenue_growth', 'N/A')}")
    print(f"    Earnings Growth: {yahoo_fundamental.get('earnings_growth', 'N/A')}")
    print(f"    Recommendation: {yahoo_fundamental.get('recommendation_key', 'N/A')}")
    
    # Map to FundamentalAgent format
    print("\n[4] Mapping Yahoo Finance data to FundamentalAgent format...")
    agent_input = map_yahoo_to_fundamental_input(yahoo_fundamental)
    print(f"    Mapped input:")
    print(f"      - earnings_surprise_pct: {agent_input.get('earnings_surprise_pct')}")
    print(f"      - pe_vs_sector_ratio: {agent_input.get('pe_vs_sector_ratio'):.4f}")
    print(f"      - eps_revision_direction: {agent_input.get('eps_revision_direction'):.4f}")
    print(f"      - revenue_growth_yoy: {agent_input.get('revenue_growth_yoy')}")
    
    # Create and run FundamentalAgent
    print("\n[5] Running FundamentalAgent...")
    fundamental_agent = FundamentalAgent()
    
    # Create agent context
    agent_ctx = AgentContext(
        run_id="test-001",
        rules_version="1.0.0",
        input_data=agent_input,
        config={},
        llm=None,  # Not needed for fundamental agent
    )
    
    # Run the agent
    result = fundamental_agent.run(agent_ctx)
    
    print(f"\n[6] FundamentalAgent Results:")
    print(f"    Success: {result.success}")
    
    if result.success:
        signal = result.payload.get("signal", {})
        raw_signal = result.payload.get("raw_signal")
        print(f"\n    Signal Output (feature dict for ML Judge):")
        print(f"      - Earnings Surprise (fund_earnings_surprise): {signal.get('fund_earnings_surprise', 'N/A')}")
        print(f"      - Valuation Risk (fund_valuation_risk): {signal.get('fund_valuation_risk', 'N/A')}")
        print(f"      - Analyst Revision (fund_analyst_rev): {signal.get('fund_analyst_rev', 'N/A')}")
        print(f"      - Growth Momentum (fund_growth_mom): {signal.get('fund_growth_mom', 'N/A')}")
        print(f"      - Confidence (fund_confidence): {signal.get('fund_confidence', 'N/A')}")
        
        if raw_signal:
            print(f"\n    Raw Signal object:")
            print(f"      - earnings_surprise_strength: {raw_signal.earnings_surprise_strength}")
            print(f"      - valuation_risk_score: {raw_signal.valuation_risk_score}")
            print(f"      - analyst_revision_trend: {raw_signal.analyst_revision_trend}")
            print(f"      - growth_momentum: {raw_signal.growth_momentum}")
            print(f"      - fundamental_confidence: {raw_signal.fundamental_confidence}")
    else:
        print(f"    Errors: {[e.to_dict() for e in result.errors]}")
    
    # Also show recent price data
    if context.price_data:
        print("\n[7] Recent MSFT Price Data (last 5 days):")
        for price in context.price_data[-5:]:
            print(f"    {price.date}: Open=${price.open:.2f}, Close=${price.close:.2f}, Vol={price.volume:,}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
