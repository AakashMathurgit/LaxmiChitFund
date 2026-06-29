"""Run LCF Future Prediction for US stocks.

Generates multi-horizon price predictions (1 week, 1 month, 1 quarter, 1 year)
using all available data: technicals, fundamentals, sentiment, news, events,
market regime, and historical pattern matching.

Usage:
    python us_stock_tracker/run_us_prediction.py AMZN
    python us_stock_tracker/run_us_prediction.py AMZN NVDA TSLA --mode adaptive
"""

import os
import sys
import json
import traceback
from datetime import datetime

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

os.environ["LCF_DEBUG"] = "1"

CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")
PREDICTION_RESULTS_FILE = os.path.join(_SCRIPT_DIR, "prediction_results.json")
PREDICTION_HISTORY_FILE = os.path.join(_SCRIPT_DIR, "prediction_history.jsonl")

# US data paths
US_NEWS_CACHE_PATH = os.path.join(_LCF_ROOT, "data", "news_cache_us.jsonl")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LCF Future Price Prediction (US Stocks)")
    parser.add_argument("symbols", nargs="+", type=str, help="Stock symbols to predict (e.g., AMZN NVDA TSLA)")
    parser.add_argument("--mode", type=str, default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"],
                        help="Trading mode for context (default: adaptive)")
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols]
    start_time = datetime.now()

    print(f"\n{'#' * 65}")
    print(f"#  LCF FUTURE PREDICTION")
    print(f"#  Stocks: {', '.join(symbols)}")
    print(f"#  Horizons: 1 Week | 1 Month | 1 Quarter | 1 Year")
    print(f"#  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 65}")

    # Initialize orchestrator
    print(f"\n[1/4] Initializing pipeline...")
    try:
        from src.pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)

        # Patch for US market (no .NS suffix)
        orchestrator.data_processor = _create_us_data_processor(orchestrator)

        print(f"  Orchestrator ready (US market, mode={args.mode})")
    except Exception as e:
        print(f"  [ERROR] Failed to initialize: {e}")
        traceback.print_exc()
        return

    # Run predictions
    print(f"\n[2/4] Fetching data & running predictions for {len(symbols)} stock(s)...")
    try:
        results = orchestrator.predict_future_for_symbols(symbols, index_symbol="^GSPC")
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return

    # Display results
    print(f"\n{'=' * 65}")
    print(f"  FUTURE PRICE PREDICTIONS")
    print(f"{'=' * 65}")

    for r in results:
        symbol = r.get("symbol", "?")
        if "error" in r:
            print(f"\n  {symbol}: ERROR — {r['error']}")
            continue

        pred = r.get("prediction", {})
        current = pred.get("current_price", 0)
        outlook = pred.get("overall_outlook", "?").upper()
        confidence = pred.get("overall_confidence", 0)

        print(f"\n  {'=' * 60}")
        print(f"  {symbol} — Current: ${current:,.2f} | Market Cap: {pred.get('market_cap', 'N/A')} | Sector: {pred.get('sector', 'N/A')}")
        print(f"  Overall Outlook: {outlook} (Confidence: {confidence * 100:.0f}%)")
        print(f"  {'─' * 60}")

        if pred.get("summary"):
            print(f"  Summary: {pred['summary']}")
            print()

        # Agent context
        regime = r.get("regime", {})
        if regime:
            print(f"  Market Regime: {regime.get('regime', '?')} | Volatility: {regime.get('vol_state', '?')}")

        # Print each horizon
        for horizon_key, label in [("one_week", "1 WEEK"), ("one_month", "1 MONTH"), ("one_quarter", "1 QUARTER"), ("one_year", "1 YEAR")]:
            h = pred.get(horizon_key)
            if not h:
                continue

            price = h.get("predicted_price", 0)
            change = h.get("predicted_change_pct", 0)
            conf = h.get("confidence", 0)
            direction = h.get("direction", "?").upper()

            arrow = "▲" if change > 0 else ("▼" if change < 0 else "─")
            color_tag = f"[{direction}]"

            print(f"\n  ┌─ {label} {'─' * (48 - len(label))}")
            print(f"  │ Predicted Price: ${price:,.2f}  ({arrow} {change:+.2f}%)")
            print(f"  │ Direction: {direction}  |  Confidence: {conf * 100:.0f}%")

            drivers = h.get("key_drivers", [])
            if drivers:
                print(f"  │ Key Drivers:")
                for d in drivers[:3]:
                    print(f"  │   ✦ {d}")

            risks = h.get("risks", [])
            if risks:
                print(f"  │ Risks:")
                for risk in risks[:2]:
                    print(f"  │   ⚠ {risk}")

            reasoning = h.get("reasoning", "")
            if reasoning:
                print(f"  │ Reasoning: {reasoning}")

            print(f"  └{'─' * 58}")

    # Save results
    print(f"\n[3/4] Saving results...")
    try:
        output = {
            "timestamp": datetime.now().isoformat(),
            "market": "US",
            "mode": args.mode,
            "type": "future_prediction",
            "symbols": symbols,
            "results": results,
        }
        with open(PREDICTION_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved to {PREDICTION_RESULTS_FILE}")

        with open(PREDICTION_HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(output, default=str) + "\n")
        print(f"  History appended to {PREDICTION_HISTORY_FILE}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[4/4] Done in {elapsed:.1f}s")


def _create_us_data_processor(orchestrator):
    """Create a DataProcessor configured for US market."""
    from src.main.controllers.data_processor import DataProcessor
    from src.main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig

    processor = DataProcessor()
    yf_config = YFinanceConfig(
        default_exchange_suffix="",
        price_period="2y",
        price_interval="1d",
    )
    processor.register_provider(YahooFinanceProvider(yf_config))

    try:
        from src.main.controllers.us_rss_news_provider import USRSSNewsProvider, USRSSNewsConfig
        us_rss_config = USRSSNewsConfig(
            only_significant_news=True,
            max_news_age_hours=72,
            cache_path=US_NEWS_CACHE_PATH if os.path.exists(US_NEWS_CACHE_PATH) else None,
            prefer_cache=True,
        )
        processor.register_provider(USRSSNewsProvider(us_rss_config))
    except Exception:
        pass

    return processor


if __name__ == "__main__":
    main()
