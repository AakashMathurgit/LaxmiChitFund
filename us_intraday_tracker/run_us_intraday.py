"""US intraday fluctuation funnel — one pass.

Flow:
  1. Load the editable universe (5 sectors x 20 tickers + peer map).
  2. Cheap scan of ALL symbols (one batched yfinance call) -> rank top movers.
  3. Fetch news for movers from the shared US news cache.
  4. Deep-analyse ONLY the movers via the existing PipelineOrchestrator
     (intraday bars) -> judge decision per mover.
  5. Cross-ticker correlation: explain each move via own + peer news.
  6. Reconcile judge decision with the correlation signal -> final decision.
  7. Notify (Pushover) on every final BUY/SELL.
  8. Trade only if a broker is available (paper by default), via TradeExecutor
     -> VestedBroker, then PortfolioManager.
  9. Persist results.

Usage:
    python us_intraday_tracker/run_us_intraday.py --mode adaptive --top 8
    python us_intraday_tracker/run_us_intraday.py --scan-only
    python us_intraday_tracker/run_us_intraday.py --no-auto-trade
"""

import os
import sys
import json
import argparse
import traceback
from datetime import datetime

# --- Paths / imports ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "intraday_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "intraday_history.jsonl")
UNIVERSE_FILE = os.path.join(_SCRIPT_DIR, "universe_us.yaml")

# US-specific data paths (match run_us_pipeline.py)
US_PORTFOLIO_PATH = os.path.join(_LCF_ROOT, "data", "us_portfolio.json")
US_WATCHLIST_PATH = os.path.join(_LCF_ROOT, "data", "us_watchlist.json")
US_NEWS_CACHE_PATH = os.path.join(_LCF_ROOT, "data", "news_cache_us.jsonl")


def _load_config():
    import yaml
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _build_scan_provider():
    from src.main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig
    return YahooFinanceProvider(YFinanceConfig(default_exchange_suffix=""))


def _create_intraday_data_processor(deep_period: str, deep_interval: str):
    """US data processor whose price history is intraday (for the deep stage)."""
    from src.main.controllers.data_processor import DataProcessor
    from src.main.controllers.yahoo_finance_provider import YahooFinanceProvider, YFinanceConfig

    processor = DataProcessor()
    yf_config = YFinanceConfig(
        default_exchange_suffix="",
        price_period=deep_period,
        price_interval=deep_interval,
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
    except Exception as e:
        print(f"  [WARN] US RSS News provider not available: {e}")

    return processor


def _reconcile(judge_decision: str, judge_conf: float, cross) -> dict:
    """Combine the judge decision with the cross-ticker signal.

    - BUY on an UNEXPLAINED move -> downgrade to HOLD (move not news-backed).
    - BUY confirmed by a peer-spillover tailwind -> keep BUY, nudge confidence.
    - Otherwise keep the judge decision.
    """
    final = judge_decision
    conf = judge_conf
    note = "judge decision unchanged"

    if judge_decision == "BUY" and cross.explanation_type == "unexplained":
        final = "HOLD"
        note = "downgraded BUY->HOLD: move not explained by any news"
    elif judge_decision == "BUY" and cross.explanation_type == "peer_spillover" \
            and cross.move_direction == "rise":
        conf = min(conf + 0.05, 0.99)
        note = f"BUY confirmed by peer spillover (driver={cross.driver_ticker})"
    elif judge_decision == "SELL" and cross.explanation_type == "unexplained":
        note = "SELL kept but move is unexplained — verify before acting"

    return {"final_decision": final, "final_confidence": round(conf, 4), "note": note}


def main(argv=None):
    parser = argparse.ArgumentParser(description="LCF US Intraday Fluctuation Funnel")
    parser.add_argument("--mode", default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"])
    parser.add_argument("--top", type=int, default=None, help="Number of top movers to deep-analyse")
    parser.add_argument("--min-fluctuation", type=float, default=None,
                        help="Minimum abs %% change to qualify as a mover")
    parser.add_argument("--scan-only", action="store_true", help="Only scan + rank, no deep analysis")
    parser.add_argument("--no-auto-trade", dest="auto_trade", action="store_false",
                        help="Disable auto paper-trading (notify only)")
    parser.set_defaults(auto_trade=True)
    args = parser.parse_args(argv)

    cfg = _load_config()
    intraday_cfg = cfg.get("intraday", {})
    top_n = args.top or intraday_cfg.get("top_movers", 8)
    min_fluc = args.min_fluctuation if args.min_fluctuation is not None \
        else intraday_cfg.get("min_fluctuation_pct", 1.5)
    scan_interval_yf = intraday_cfg.get("scan_interval_yf", "2m")
    deep_interval = intraday_cfg.get("deep_interval_yf", "15m")
    deep_period = intraday_cfg.get("deep_period_yf", "5d")
    vested_cfg = cfg.get("vested", {})
    alpaca_cfg = cfg.get("alpaca", {})

    start = datetime.now()
    print(f"\n{'#' * 60}")
    print(f"#  LCF US Intraday Funnel  | mode={args.mode}  top={top_n}")
    print(f"#  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # --- 1. Universe ---
    from us_intraday_tracker.universe import load_universe
    universe = load_universe(UNIVERSE_FILE)
    symbols = universe.all_symbols()
    print(f"\n[1] Universe: {len(symbols)} symbols across {len(universe.sectors_list())} sectors")

    # --- 2. Scan + rank ---
    from us_intraday_tracker.fluctuation_scanner import FluctuationScanner, ScanConfig
    scanner = FluctuationScanner(
        _build_scan_provider(),
        ScanConfig(intraday_interval=scan_interval_yf, min_fluctuation_pct=min_fluc),
        sector_of=universe.sector_of,
    )
    print(f"[2] Scanning {len(symbols)} symbols (interval={scan_interval_yf})...")
    snapshots = scanner.scan(symbols)
    movers = scanner.rank_movers(snapshots, top_n=top_n)
    print(f"    {len(snapshots)} snapshots, {len(movers)} movers (min |change| >= {min_fluc}%)")
    for m in movers:
        print(f"      {m.symbol:6s} {m.pct_change:+6.2f}%  range={m.intraday_range_pct:5.2f}%  "
              f"volx{m.vol_spike:4.2f}  score={m.fluctuation_score:5.2f}")

    if args.scan_only or not movers:
        if not movers:
            print("    No movers — nothing to analyse.")
        _save({"timestamp": start.isoformat(), "mode": args.mode, "scan_only": True,
               "movers": [m.to_dict() for m in movers]})
        return 0

    mover_symbols = [m.symbol for m in movers]
    mover_by_symbol = {m.symbol: m for m in movers}

    # --- 3-4. Deep analysis (orchestrator, intraday bars) ---
    print(f"\n[3] Deep-analysing {len(mover_symbols)} movers (interval={deep_interval})...")
    from src.pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)
    orchestrator.data_processor = _create_intraday_data_processor(deep_period, deep_interval)
    try:
        results = orchestrator.run_for_symbols(mover_symbols, index_symbol="^GSPC")
    except Exception as e:
        print(f"  [ERROR] Deep analysis failed: {e}")
        traceback.print_exc()
        return 1

    # --- 5. Cross-ticker correlation ---
    print(f"\n[4] Cross-ticker correlation...")
    from src.main.agents.cross_ticker_agent import CrossTickerAgent
    from src.main.controllers.news_cache import NewsCache
    news_cache = NewsCache(US_NEWS_CACHE_PATH) if os.path.exists(US_NEWS_CACHE_PATH) else None
    cross_agent = CrossTickerAgent()

    for r in results:
        symbol = r.get("symbol")
        if not symbol or "error" in r:
            continue
        mover = mover_by_symbol.get(symbol)
        move_pct = mover.pct_change if mover else 0.0

        own_news, peer_news = [], {}
        if news_cache:
            own_news = news_cache.read_items(symbol=symbol)
            for peer in universe.peer_symbols(symbol):
                pn = news_cache.read_items(symbol=peer)
                if pn:
                    peer_news[peer] = pn

        cross = cross_agent.correlate(
            symbol, move_pct, own_news=own_news, peer_news=peer_news, llm=orchestrator.llm
        )
        r["cross_ticker"] = cross.to_dict()

        judge = r.get("judge_decision", {}).get("payload", {})
        decision = judge.get("decision", "HOLD")
        conf = float(judge.get("confidence", 0.0) or 0.0)
        r["intraday_decision"] = _reconcile(decision, conf, cross)

        print(f"      {symbol:6s} judge={decision:4s} -> final={r['intraday_decision']['final_decision']:4s}"
              f"  | {cross.explanation_type}"
              + (f" (driver={cross.driver_ticker})" if cross.driver_ticker else ""))

    # --- 6. Notify ---
    final_actions = [
        r for r in results
        if r.get("intraday_decision", {}).get("final_decision") in ("BUY", "SELL")
    ]
    print(f"\n[5] Notify: {len(final_actions)} actionable (BUY/SELL)")
    if final_actions:
        _notify(final_actions, mover_by_symbol)

    # --- 7. Trade (only if broker available) ---
    print(f"\n[6] Trade (auto_trade={args.auto_trade})...")
    if args.auto_trade and final_actions:
        _trade(final_actions, mover_by_symbol, cfg, args.mode, results)
    elif not args.auto_trade:
        print("    Auto-trade disabled — notify only.")
    else:
        print("    No actionable signals.")

    # --- 8. Persist ---
    buy = sum(1 for r in final_actions if r["intraday_decision"]["final_decision"] == "BUY")
    sell = sum(1 for r in final_actions if r["intraday_decision"]["final_decision"] == "SELL")
    _save({
        "timestamp": start.isoformat(),
        "market": "US",
        "mode": args.mode,
        "movers_analyzed": len(mover_symbols),
        "movers": [m.to_dict() for m in movers],
        "buy_count": buy,
        "sell_count": sell,
        "results": results,
    })

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone in {elapsed:.1f}s — {buy} BUY, {sell} SELL")
    return 0


def _notify(final_actions, mover_by_symbol):
    try:
        from src.runtime import Notifier
        notifier = Notifier(market="US")
        if not notifier.enabled:
            print("    Notifier not configured (set PUSHOVER_* env or credentials.yaml).")
            return
        lines = ["<b>[INTRADAY] Movers — actionable</b>"]
        for r in final_actions:
            sym = r["symbol"]
            d = r["intraday_decision"]
            cross = r.get("cross_ticker", {})
            m = mover_by_symbol.get(sym)
            chg = f"{m.pct_change:+.2f}%" if m else ""
            driver = f" via {cross.get('driver_ticker')}" if cross.get("driver_ticker") else ""
            lines.append(
                f"{d['final_decision']} <b>{sym}</b> {chg} "
                f"(conf {d['final_confidence']:.0%}) — {cross.get('explanation_type', '')}{driver}"
            )
        sent = notifier.send("\n".join(lines), title="[INTRADAY] LCF Movers")
        print(f"    Notify {'sent' if sent else 'skipped'} ({len(final_actions)} signals).")
    except Exception as e:
        print(f"    [WARN] Notify failed: {e}")


def _trade(final_actions, mover_by_symbol, cfg, mode, all_results):
    try:
        from src.runtime import BrokerRouter
        from src.runtime.position_sizer import PositionSizer
        router = BrokerRouter(cfg)
        if not router.available:
            print("    No broker available — notify only, no orders.")
            return
        print(f"    Brokers firing on each signal: {', '.join(router.available)}")

        # Shared position sizer — sizes orders off the live account equity.
        sizer = PositionSizer(cfg)
        fallback_eq = float(cfg.get("position_sizing", {}).get("fallback_equity", 100000))
        equity = router.account_equity(default=fallback_eq)
        bp = router.buying_power(default=equity)
        print(f"    Sizing: method={sizer.method}, equity=${equity:,.0f}")

        for r in final_actions:
            sym = r["symbol"]
            action = r["intraday_decision"]["final_decision"]
            m = mover_by_symbol.get(sym)
            price = m.current if m and m.current else 0.0
            if not price:
                continue
            stop = (r.get("trade_plan") or {}).get("stop_loss_price", 0.0)
            qty = sizer.size(equity, price, stop_loss=stop, strategy="intraday", buying_power=bp)
            if qty <= 0:
                print(f"    [{sym}] sized 0 shares — skipping")
                continue
            execs = router.execute(sym, action, quantity=qty, price=price)
            r["executions"] = execs
            r["order_qty"] = qty
            for name, res in execs.items():
                print(f"    [{name}] {action} {sym} x{qty} @ {price:.2f} -> "
                      f"{res.get('status')} ({res.get('order_id', '')})")

        # Reflect entries/exits in the US portfolio (reuse existing logic).
        try:
            from src.main.controllers.portfolio_manager import PortfolioManager
            pm = PortfolioManager(portfolio_path=US_PORTFOLIO_PATH, watchlist_path=US_WATCHLIST_PATH)
            pm.load()
            pm.process_signals(all_results, auto_enter=True, auto_exit=True, mode=mode)
        except Exception as e:
            print(f"    [WARN] Portfolio update skipped: {e}")
    except Exception as e:
        print(f"    [WARN] Trade step failed: {e}")


def _save(output: dict):
    try:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)
        with open(HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(output, default=str) + "\n")
    except Exception as e:
        print(f"  [WARN] Save failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
