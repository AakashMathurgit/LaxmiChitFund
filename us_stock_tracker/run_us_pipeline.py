"""Run the LCF orchestrator pipeline for US stocks.

Parallel to news_stock_tracker/run_orchestrator_pipeline.py (Indian stocks).
Uses the same PipelineOrchestrator but configured for US market:
  - Exchange suffix: "" (no suffix needed for Yahoo Finance US)
  - Portfolio: data/us_portfolio.json
  - Watchlist: data/us_watchlist.json
  - News cache: data/news_cache_us.jsonl
  - Trade memory: data/us_trade_memory.jsonl

Usage:
    python us_stock_tracker/run_us_pipeline.py --mode adaptive --stocks 10
    python us_stock_tracker/run_us_pipeline.py --source watchlist --show-portfolio
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

# Enable debug
os.environ["LCF_DEBUG"] = "1"

STOCKS_FILE = os.path.join(_SCRIPT_DIR, "stocks.txt")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "analysis_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "analysis_history.jsonl")
CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")

# US-specific data paths
US_PORTFOLIO_PATH = os.path.join(_LCF_ROOT, "data", "us_portfolio.json")
US_WATCHLIST_PATH = os.path.join(_LCF_ROOT, "data", "us_watchlist.json")
US_TRADE_MEMORY_PATH = os.path.join(_LCF_ROOT, "data", "us_trade_memory.jsonl")
US_NEWS_CACHE_PATH = os.path.join(_LCF_ROOT, "data", "news_cache_us.jsonl")
US_PATTERN_STORE_DIR = os.path.join(_LCF_ROOT, "data", "us_pattern_store")


def load_symbols(limit: int = 5) -> list:
    """Load discovered US symbols from stocks.txt."""
    if not os.path.exists(STOCKS_FILE):
        print(f"[WARN] {STOCKS_FILE} not found, using watchlist defaults")
        return ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]
    with open(STOCKS_FILE) as f:
        symbols = [line.strip() for line in f if line.strip()]
    if not symbols:
        return ["NVDA", "TSLA", "AAPL", "MSFT", "AMZN"]
    return symbols[:limit]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LCF US Pipeline Runner")
    parser.add_argument("--mode", type=str, default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"],
                        help="Trading mode (default: adaptive)")
    parser.add_argument("--stocks", type=int, default=10, help="Number of stocks to analyze")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "watchlist", "discovery", "file"],
                        help="Symbol source (default: auto)")
    parser.add_argument("--auto-trade", action="store_true", help="Auto-enter BUY positions")
    parser.add_argument("--show-portfolio", action="store_true", help="Show portfolio summary")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'#' * 60}")
    print(f"#  LCF US Orchestrator Pipeline Runner")
    print(f"#  Mode: {args.mode.upper()}")
    print(f"#  Market: US (NYSE / NASDAQ)")
    print(f"#  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # Load US portfolio manager (pointed to US-specific files)
    from src.main.controllers.portfolio_manager import PortfolioManager
    pm = PortfolioManager(
        portfolio_path=US_PORTFOLIO_PATH,
        watchlist_path=US_WATCHLIST_PATH,
    )
    pm.load()

    # Initialize orchestrator with US configuration
    print(f"\n[1/7] Initializing PipelineOrchestrator (US market, mode={args.mode})...")
    try:
        from src.pipeline.orchestrator import PipelineOrchestrator
        
        # Override config for US market
        orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)
        
        # Patch the data processor for US market:
        # 1. Change exchange suffix to "" (US stocks don't need .NS)
        orchestrator.data_processor = _create_us_data_processor(orchestrator)
        
        print(f"  Orchestrator ready (US market)")
        print(f"  Mode: {orchestrator.mode.upper()} - {orchestrator._mode_config.description}")
        print(f"  PatternStore: {orchestrator.pattern_store.count} records")
        print(f"  TradeMemory: {orchestrator.trade_memory.count} records")
    except Exception as e:
        print(f"  [ERROR] Failed to create orchestrator: {e}")
        traceback.print_exc()
        return

    # Resolve symbols
    print(f"\n[2/7] Resolving US symbols (source={args.source})...")
    symbols = []

    if args.source == "watchlist":
        symbols = pm.get_watchlist_symbols()[:args.stocks]
        print(f"  Watchlist: {symbols}")

    elif args.source == "file":
        symbols = load_symbols(limit=args.stocks)
        print(f"  From stocks.txt: {symbols}")

    else:  # auto — watchlist + portfolio + discovered
        held = [h.symbol for h in pm.get_all_holdings()]
        wl_high = [w.symbol for w in pm.get_watchlist() if w.priority == "high"]
        combined = list(dict.fromkeys(held + wl_high))

        remaining_slots = args.stocks - len(combined)
        if remaining_slots > 0:
            # Fill from stocks.txt (news-discovered)
            discovered = load_symbols(limit=remaining_slots + len(combined))
            for s in discovered:
                if s not in combined and len(combined) < args.stocks:
                    combined.append(s)

        symbols = combined[:args.stocks]
        print(f"  Auto-selected: {symbols}")

    if not symbols:
        print("  [ERROR] No symbols to analyze!")
        return

    print(f"  Final symbols ({len(symbols)}): {symbols}")

    # Run the pipeline
    print(f"\n[3/7] Running pipeline for {len(symbols)} US symbols...")
    try:
        results = orchestrator.run_for_symbols(symbols, index_symbol="^GSPC")  # S&P 500
    except Exception as e:
        print(f"  [ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return

    # Print results
    print(f"\n{'=' * 60}")
    print("US ANALYSIS RESULTS")
    print(f"{'=' * 60}")

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for r in results:
        symbol = r.get("symbol", "?")
        if "error" in r:
            print(f"\n  {symbol}: ERROR — {r['error']}")
            continue

        jd = r.get("judge_decision", {}).get("payload", {})
        decision = jd.get("decision", "?")
        prob_up = jd.get("prob_up_5d", 0)
        confidence = jd.get("confidence", 0)

        indicator = {"BUY": "[BUY] ", "SELL": "[SELL]", "HOLD": "[HOLD]"}.get(decision, "[????]")
        print(f"\n  {indicator} {symbol}")
        print(f"    Decision: {decision} | Prob Up 5d: {prob_up * 100:.1f}% | Confidence: {confidence * 100:.1f}%")

        if decision == "BUY": buy_count += 1
        elif decision == "SELL": sell_count += 1
        else: hold_count += 1

        # Regime
        regime = r.get("regime", {})
        if regime:
            print(f"    Regime: {regime.get('regime', '?')} | Vol: {regime.get('vol_state', '?')}")

        # Debate
        debate = r.get("debate")
        if debate:
            bull = debate.get("bull", {})
            bear = debate.get("bear", {})
            dd = debate.get("debate_decision", {})
            print(f"    Bull: {bull.get('recommendation', '?')} ({bull.get('confidence', 0) * 100:.0f}%)")
            for pt in bull.get("key_points", [])[:2]:
                print(f"      + {pt[:80]}")
            print(f"    Bear: {bear.get('recommendation', '?')} ({bear.get('confidence', 0) * 100:.0f}%)")
            for pt in bear.get("key_points", [])[:2]:
                print(f"      - {pt[:80]}")
            print(f"    Debate Winner: {dd.get('winning_side', '?')} ({dd.get('decision', '?')})")

        # Hybrid decision
        hybrid = r.get("hybrid_decision")
        if hybrid:
            print(f"    Hybrid: {hybrid.get('final_decision', '?')} "
                  f"(confidence={hybrid.get('final_confidence', 0) * 100:.0f}%, "
                  f"agreement={hybrid.get('agreement', '?')})")

        # Trade plan
        tp = r.get("trade_plan")
        if tp:
            print(f"    Trade Plan: {tp.get('entry_type', '?')} @ ${tp.get('entry_price', 0):,.2f}")
            print(f"      SL: ${tp.get('stop_loss_price', 0):,.2f} | "
                  f"Target: ${tp.get('target_price', 0):,.2f} | "
                  f"R:R={tp.get('risk_reward_ratio', 0):.1f}")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"US SUMMARY: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL")
    print(f"{'=' * 60}")

    # Post-run stats
    print(f"\n[4/7] Post-run stats:")
    print(f"  PatternStore: {orchestrator.pattern_store.count} records")
    print(f"  TradeMemory: {orchestrator.trade_memory.count} records")

    # Portfolio processing
    print(f"\n[5/7] Portfolio processing...")
    try:
        actions = pm.process_signals(
            results, auto_enter=args.auto_trade, auto_exit=True, mode=args.mode
        )
        if actions["entered"]:
            print(f"  ENTERED {len(actions['entered'])} positions:")
            for a in actions["entered"]:
                print(f"    BUY {a['symbol']} {a['shares']} shares @ ${a['price']:,.2f}")
        if actions["exited"]:
            print(f"  EXITED {len(actions['exited'])} positions:")
            for a in actions["exited"]:
                print(f"    SOLD {a['symbol']} P&L: ${a.get('pnl', 0):+,.2f}")
        if actions["sl_triggered"]:
            print(f"  STOP LOSS triggered: {len(actions['sl_triggered'])}")
        if actions["target_hit"]:
            print(f"  TARGETS hit: {len(actions['target_hit'])}")
        if actions["updated"]:
            print(f"  Updated prices for: {', '.join(actions['updated'])}")
        if not any(actions[k] for k in ["entered", "exited", "sl_triggered", "target_hit"]):
            print(f"  No portfolio trades (use --auto-trade to auto-enter BUY positions)")
    except Exception as e:
        print(f"  [WARN] Portfolio processing: {e}")

    if args.show_portfolio:
        print(f"\n{pm.portfolio_text_summary()}")

    # Skip WhatsApp for US
    # Pushover push notifications
    print(f"\n[6/7] Pushover notifications...")
    try:
        from src.main.controllers.pushover_controller import PushoverController
        pushover = PushoverController(market="US")
        if pushover._enabled:
            sent = pushover.send_batch_trade_alerts(results, mode=args.mode, market="US")
            print(f"  Sent combined report ({len(results)} stocks) via Pushover" if sent else "  No alerts to send")
        else:
            print(f"  Pushover not configured (add pushover keys to credentials.yaml)")
    except Exception as e:
        print(f"  [WARN] Pushover failed: {e}")

    # Save results
    print(f"\n[7/7] Saving results...")
    try:
        output = {
            "timestamp": datetime.now().isoformat(),
            "market": "US",
            "mode": args.mode,
            "source": args.source,
            "symbols_analyzed": len(symbols),
            "symbols": symbols,
            "buy_count": buy_count,
            "hold_count": hold_count,
            "sell_count": sell_count,
            "results": results,
        }
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"  Saved to {RESULTS_FILE}")

        # Append to history
        with open(HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(output, default=str) + "\n")
        print(f"  History appended to {HISTORY_FILE}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")


def _create_us_data_processor(orchestrator):
    """Create a DataProcessor configured for US market (no .NS suffix, US news cache)."""
    from src.main.controllers.data_processor import DataProcessor
    from src.main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig

    processor = DataProcessor()

    # Yahoo Finance — no exchange suffix for US stocks
    yf_config = YFinanceConfig(
        default_exchange_suffix="",  # US stocks don't need .NS
        price_period="2y",
        price_interval="1d",
    )
    processor.register_provider(YahooFinanceProvider(yf_config))

    # US RSS News provider (cache-aware)
    try:
        from src.main.controllers.us_rss_news_provider import USRSSNewsProvider, USRSSNewsConfig

        us_rss_config = USRSSNewsConfig(
            only_significant_news=True,
            max_news_age_hours=72,
            cache_path=US_NEWS_CACHE_PATH if os.path.exists(US_NEWS_CACHE_PATH) else None,
            prefer_cache=True,
        )
        processor.register_provider(USRSSNewsProvider(us_rss_config))
        cache_status = "with cache" if us_rss_config.cache_path else "live only"
        print(f"  US RSS News provider registered ({cache_status})")
    except Exception as e:
        print(f"  [WARN] US RSS News provider not available: {e}")

    return processor


if __name__ == "__main__":
    main()
