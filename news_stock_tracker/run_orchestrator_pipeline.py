"""Run the full LCF orchestrator pipeline on stocks from news_stock_tracker.

Uses the PipelineOrchestrator with:
  - DebateContext (rich data for Bull/Bear agents)
  - PatternStore (ChromaDB similarity search)
  - TradeMemory (append-only JSONL trade log)
  - Full debate flow: Bull vs Bear -> HybridDecision

Picks top 3 stocks from stocks.txt and runs the complete pipeline.
"""

import os
import sys
import json
import traceback
from datetime import datetime

# Setup paths
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
_SRC_DIR = os.path.join(_LCF_ROOT, "src")
# Add LCF_ROOT so that `from src.pipeline.orchestrator` works with relative imports
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

# Enable debug
os.environ["LCF_DEBUG"] = "1"

STOCKS_FILE = os.path.join(_SCRIPT_DIR, "stocks.txt")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "analysis_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "analysis_history.jsonl")
CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")


def load_symbols(limit: int = 3) -> list:
    """Load symbols from stocks.txt (written by stock_tracker)."""
    if not os.path.exists(STOCKS_FILE):
        print(f"[WARN] {STOCKS_FILE} not found, using defaults")
        return ["TCS", "INFY", "RELIANCE"]
    with open(STOCKS_FILE) as f:
        symbols = [line.strip() for line in f if line.strip()]
    if not symbols:
        return ["TCS", "INFY", "RELIANCE"]
    return symbols[:limit]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LCF Pipeline Runner")
    parser.add_argument("--mode", type=str, default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"],
                        help="Trading mode (default: adaptive)")
    parser.add_argument("--stocks", type=int, default=3, help="Number of stocks to analyze (default: 3)")
    parser.add_argument("--whatsapp", type=str, default="", help="WhatsApp number for alerts (e.g., +919876543210)")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "watchlist", "discovery", "file"],
                        help="Symbol source: auto (watchlist+discovery), watchlist, discovery (screen NIFTY50), file (stocks.txt)")
    parser.add_argument("--auto-trade", action="store_true", help="Auto-enter BUY positions in portfolio")
    parser.add_argument("--show-portfolio", action="store_true", help="Show portfolio summary after run")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"\n{'#'*60}")
    print(f"#  LCF Orchestrator Pipeline Runner")
    print(f"#  Mode: {args.mode.upper()}")
    print(f"#  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}")

    # Load portfolio manager
    from src.main.controllers.portfolio_manager import PortfolioManager
    pm = PortfolioManager()
    pm.load()

    # Import and create orchestrator
    print(f"\n[1/6] Initializing PipelineOrchestrator (mode={args.mode})...")
    try:
        from src.pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)
        print(f"  Orchestrator ready")
        print(f"  Mode: {orchestrator.mode.upper()} - {orchestrator._mode_config.description}")
        print(f"  Max positions: {orchestrator._mode_config.max_positions}")
        print(f"  Hold days: {orchestrator._mode_config.hold_days}")
        print(f"  PatternStore: {'available' if orchestrator.pattern_store.available else 'unavailable'} ({orchestrator.pattern_store.count} records)")
        print(f"  TradeMemory: {orchestrator.trade_memory.count} records")
    except Exception as e:
        print(f"  [ERROR] Failed to create orchestrator: {e}")
        traceback.print_exc()
        return

    # Resolve symbols based on source
    print(f"\n[2/6] Resolving symbols (source={args.source})...")
    symbols = []

    if args.source == "watchlist":
        symbols = pm.get_watchlist_symbols()[:args.stocks]
        print(f"  Watchlist: {symbols}")

    elif args.source == "discovery":
        print(f"  Running StockDiscoveryAgent on NIFTY 50 universe...")
        try:
            candidates = orchestrator.discover_symbols(max_stocks=args.stocks)
            symbols = [c["symbol"] for c in candidates]
            for c in candidates:
                print(f"    {c['symbol']}: score={c['score']:.2f} -- {', '.join(c['reasons'])}")
        except Exception as e:
            print(f"  [WARN] Discovery failed ({e}), falling back to watchlist")
            symbols = pm.get_watchlist_symbols()[:args.stocks]

    elif args.source == "file":
        symbols = load_symbols(limit=args.stocks)
        print(f"  From stocks.txt: {symbols}")

    else:  # auto — combine watchlist + portfolio holdings + discovery
        # 1. Always include current portfolio holdings (need price updates)
        held = [h.symbol for h in pm.get_all_holdings()]
        # 2. Watchlist high-priority stocks
        wl_high = [w.symbol for w in pm.get_watchlist() if w.priority == "high"]
        # 3. Fill remaining slots via discovery
        combined = list(dict.fromkeys(held + wl_high))  # deduplicate, preserve order
        remaining_slots = args.stocks - len(combined)

        if remaining_slots > 0:
            print(f"  Running StockDiscoveryAgent for {remaining_slots} additional stocks...")
            try:
                # Exclude already-selected symbols from discovery
                discovery_universe = [
                    s for s in orchestrator.discovery_agent._config.default_universe
                    if s not in combined
                ]
                candidates = orchestrator.discover_symbols(
                    universe=discovery_universe, max_stocks=remaining_slots,
                )
                for c in candidates:
                    combined.append(c["symbol"])
                    print(f"    Discovered: {c['symbol']} (score={c['score']:.2f} -- {', '.join(c['reasons'])})")
            except Exception as e:
                print(f"    [WARN] Discovery failed ({e}), using watchlist only")
                # Fill from remaining watchlist
                wl_rest = [w.symbol for w in pm.get_watchlist() if w.symbol not in combined]
                combined.extend(wl_rest[:remaining_slots])

        symbols = combined[:args.stocks]
        source_parts = []
        if held:
            source_parts.append(f"{len([s for s in symbols if s in held])} holdings")
        if wl_high:
            source_parts.append(f"{len([s for s in symbols if s in wl_high and s not in held])} watchlist")
        disc_count = len([s for s in symbols if s not in held and s not in wl_high])
        if disc_count:
            source_parts.append(f"{disc_count} discovered")
        print(f"  Auto-selected: {symbols} ({', '.join(source_parts)})")

    if not symbols:
        print("  [ERROR] No symbols to analyze!")
        return

    print(f"  Final symbols ({len(symbols)}): {symbols}")

    # Run the pipeline
    print(f"\n[3/6] Running pipeline for {len(symbols)} symbols...")
    try:
        results = orchestrator.run_for_symbols(symbols, index_symbol="^NSEI")
    except Exception as e:
        print(f"  [ERROR] Pipeline failed: {e}")
        traceback.print_exc()
        return

    # Print results
    print(f"\n{'='*60}")
    print("ANALYSIS RESULTS")
    print(f"{'='*60}")

    buy_count = 0
    sell_count = 0
    hold_count = 0

    for r in results:
        symbol = r.get("symbol", "?")
        if "error" in r:
            print(f"\n  {symbol}: ERROR — {r['error']}")
            continue

        # Judge decision
        jd = r.get("judge_decision", {}).get("payload", {})
        decision = jd.get("decision", "?")
        prob_up = jd.get("prob_up_5d", 0)
        confidence = jd.get("confidence", 0)

        indicator = {"BUY": "[BUY] ", "SELL": "[SELL]", "HOLD": "[HOLD]"}.get(decision, "[????]")
        print(f"\n  {indicator} {symbol}")
        print(f"    Decision: {decision} | Prob Up 5d: {prob_up*100:.1f}% | Confidence: {confidence*100:.1f}%")

        if decision == "BUY": buy_count += 1
        elif decision == "SELL": sell_count += 1
        else: hold_count += 1

        # Regime
        regime = r.get("regime", {})
        if regime:
            print(f"    Regime: {regime.get('regime', '?')} | Vol: {regime.get('vol_state', '?')}")

        # Debate flow
        debate = r.get("debate")
        if debate:
            bull = debate.get("bull", {})
            bear = debate.get("bear", {})
            dd = debate.get("debate_decision", {})
            print(f"    Bull: {bull.get('recommendation', '?')} ({bull.get('confidence', 0)*100:.0f}%)")
            for pt in bull.get("key_points", [])[:2]:
                print(f"      + {pt[:80]}")
            print(f"    Bear: {bear.get('recommendation', '?')} ({bear.get('confidence', 0)*100:.0f}%)")
            for pt in bear.get("key_points", [])[:2]:
                print(f"      - {pt[:80]}")
            print(f"    Debate Winner: {dd.get('winning_side', '?')} ({dd.get('decision', '?')})")

        # Hybrid decision
        hybrid = r.get("hybrid_decision")
        if hybrid:
            print(f"    Hybrid: {hybrid.get('final_decision', '?')} "
                  f"(confidence={hybrid.get('final_confidence', 0)*100:.0f}%, "
                  f"agreement={hybrid.get('agreement', '?')})")

        # Trade plan
        tp = r.get("trade_plan")
        if tp:
            print(f"    Trade Plan: {tp.get('entry_type', '?')} @ Rs.{tp.get('entry_price', 0):,.2f}")
            print(f"      SL: Rs.{tp.get('stop_loss_price', 0):,.2f} | "
                  f"Target: Rs.{tp.get('target_price', 0):,.2f} | "
                  f"R:R={tp.get('risk_reward_ratio', 0):.1f}")

        # Risk assessment
        risk = r.get("risk_assessment")
        if risk:
            print(f"    Risk: {risk.get('overall_risk_level', '?')}")
            if r.get("trade_blocked"):
                print(f"    *** TRADE BLOCKED: {r.get('block_reason', '?')} ***")

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {buy_count} BUY | {hold_count} HOLD | {sell_count} SELL")
    print(f"{'='*60}")

    # Post-run stats
    print(f"\n[4/6] Post-run stats:")
    print(f"  PatternStore: {orchestrator.pattern_store.count} records")
    print(f"  TradeMemory: {orchestrator.trade_memory.count} records")
    mem_stats = orchestrator.trade_memory.get_stats()
    if mem_stats.get("resolved", 0) > 0:
        print(f"  Win rate: {mem_stats.get('win_rate', 0)*100:.0f}%")
        print(f"  Avg return: {mem_stats.get('avg_return', 0)*100:.1f}%")

    # Portfolio processing
    print(f"\n[5/6] Portfolio processing...")
    try:
        actions = pm.process_signals(
            results, auto_enter=args.auto_trade, auto_exit=True, mode=args.mode
        )
        if actions["entered"]:
            print(f"  ENTERED {len(actions['entered'])} positions:")
            for a in actions["entered"]:
                print(f"    BUY {a['symbol']} {a['shares']} shares @ Rs.{a['price']:,.2f}")
        if actions["exited"]:
            print(f"  EXITED {len(actions['exited'])} positions:")
            for a in actions["exited"]:
                print(f"    SOLD {a['symbol']} P&L: Rs.{a.get('pnl', 0):+,.2f}")
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

    # WhatsApp alerts
    if args.whatsapp:
        print(f"\n[6/7] Sending WhatsApp alerts to {args.whatsapp}...")
        try:
            from src.main.controllers.message_controller import MessageController
            messenger = MessageController(phone=args.whatsapp)

            # Send individual alerts for BUY/SELL signals
            alerts_sent = 0
            for r in results:
                if messenger.send_trade_alert(r):
                    alerts_sent += 1

            # Send daily summary
            messenger.send_daily_summary(results, mode=args.mode)
            print(f"  Sent {alerts_sent} trade alerts + 1 daily summary")
        except Exception as e:
            print(f"  [WARN] WhatsApp alerts failed: {e}")
    else:
        print(f"\n[6/7] WhatsApp alerts: skipped (use --whatsapp +91XXXXXXXXXX)")

    # Pushover push notifications (always active if configured)
    print(f"\n[6.5/7] Pushover notifications...")
    try:
        from src.main.controllers.pushover_controller import PushoverController
        pushover = PushoverController(market="IND")
        if pushover._enabled:
            sent = pushover.send_batch_trade_alerts(results, mode=args.mode, market="IND")
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

        # Append to history (one JSON line per cycle, never overwritten)
        with open(HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(output, default=str) + "\n")
        print(f"  History appended to {HISTORY_FILE}")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
