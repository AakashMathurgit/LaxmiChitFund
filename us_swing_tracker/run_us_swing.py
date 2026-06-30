"""US swing funnel — one pass.

Parallel to us_intraday_tracker/run_us_intraday.py, but tuned for a MULTI-DAY
holding horizon instead of immediate intraday profit.

Flow:
  1. Load the SAME 100-stock universe as intraday (us_intraday_tracker/universe_us.yaml).
  2. Cheap scan of all symbols -> rank the most active as swing candidates.
  3. Deep multi-horizon PREDICTION on each candidate (1wk / 1mo) using the
     future-prediction pipeline — driven by fundamentals + news + technicals.
  4. Swing decision: BUY only if the outlook is bullish AND the 1-month
     predicted gain clears a threshold with enough confidence (i.e. "worth
     holding for days/weeks"); SELL if bearish; else HOLD.
  5. Notify (Pushover) on every BUY/SELL — clearly labelled [SWING].
  6. Trade into the SWING Alpaca paper account (Account 2) via BrokerRouter.
  7. Persist results to swing-specific files (never collides with intraday).

The Alpaca account is selected by the ALPACA_KEY_ID / ALPACA_SECRET_KEY env
vars, which the scheduler's USSwingFlow injects with the Account-2 keys before
launching this subprocess.

Usage:
    python us_swing_tracker/run_us_swing.py --mode value --top 20
    python us_swing_tracker/run_us_swing.py --scan-only
    python us_swing_tracker/run_us_swing.py --no-auto-trade
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
RESULTS_FILE = os.path.join(_SCRIPT_DIR, "swing_results.json")
HISTORY_FILE = os.path.join(_SCRIPT_DIR, "swing_history.jsonl")
# Reuse the intraday universe (same 100 stocks) so both strategies share a watchlist.
UNIVERSE_FILE = os.path.join(_LCF_ROOT, "us_intraday_tracker", "universe_us.yaml")

# Swing-specific state (separate from intraday so P&L never mixes).
SWING_PORTFOLIO_PATH = os.path.join(_LCF_ROOT, "data", "swing_portfolio.json")
SWING_WATCHLIST_PATH = os.path.join(_LCF_ROOT, "data", "us_watchlist.json")
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


def _create_swing_data_processor(deep_period: str, deep_interval: str):
    """US data processor whose price history is DAILY + long (for swing horizon)."""
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
            max_news_age_hours=168,   # 1 week of news matters for a swing thesis
            cache_path=US_NEWS_CACHE_PATH if os.path.exists(US_NEWS_CACHE_PATH) else None,
            prefer_cache=True,
        )
        processor.register_provider(USRSSNewsProvider(us_rss_config))
    except Exception as e:
        print(f"  [WARN] US RSS News provider not available: {e}")

    return processor


def _swing_decision(pred: dict, min_conf: float, min_hold_gain_pct: float) -> dict:
    """Turn a multi-horizon prediction into a swing BUY/SELL/HOLD.

    "Worth holding for days/weeks" means: bullish overall outlook, a 1-month
    predicted gain above the threshold, and enough confidence. A weak or
    contradicted signal is downgraded to HOLD so we don't churn the account.
    """
    outlook = (pred.get("overall_outlook") or "neutral").lower()
    conf = float(pred.get("overall_confidence", 0.0) or 0.0)
    base = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}.get(outlook, "HOLD")

    one_week = pred.get("one_week") or {}
    one_month = pred.get("one_month") or {}
    ow_chg = float(one_week.get("predicted_change_pct", 0.0) or 0.0)
    om_chg = float(one_month.get("predicted_change_pct", 0.0) or 0.0)

    final = base
    note = f"outlook={outlook} conf={conf:.0%} 1w={ow_chg:+.1f}% 1m={om_chg:+.1f}%"

    if base == "BUY":
        # Require the multi-day thesis to actually project a worthwhile gain.
        if conf < min_conf or om_chg < min_hold_gain_pct:
            final = "HOLD"
            note += " -> HOLD (gain/confidence below swing threshold)"
        else:
            note += " -> BUY (worth holding)"
    elif base == "SELL":
        # Only act on a confident bearish thesis; otherwise just hold/avoid.
        if conf < min_conf:
            final = "HOLD"
            note += " -> HOLD (bearish but low confidence)"
        else:
            note += " -> SELL (exit/avoid)"

    return {
        "final_decision": final,
        "final_confidence": round(conf, 4),
        "one_week_pct": round(ow_chg, 2),
        "one_month_pct": round(om_chg, 2),
        "note": note,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="LCF US Swing Funnel")
    parser.add_argument("--mode", default="value",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"])
    parser.add_argument("--top", type=int, default=None, help="Number of swing candidates to deep-analyse")
    parser.add_argument("--scan-only", action="store_true", help="Only scan + rank, no prediction")
    parser.add_argument("--no-auto-trade", dest="auto_trade", action="store_false",
                        help="Disable auto paper-trading (notify only)")
    parser.set_defaults(auto_trade=True)
    args = parser.parse_args(argv)

    cfg = _load_config()
    swing_cfg = cfg.get("swing", {})
    top_n = args.top or swing_cfg.get("top_candidates", 20)
    scan_interval_yf = swing_cfg.get("scan_interval_yf", "2m")
    deep_interval = swing_cfg.get("deep_interval_yf", "1d")
    deep_period = swing_cfg.get("deep_period_yf", "1y")
    min_conf = float(swing_cfg.get("min_confidence", 0.60))
    min_hold_gain = float(swing_cfg.get("min_hold_gain_pct", 1.0))
    order_qty = int(swing_cfg.get("order_qty", 1))

    start = datetime.now()
    print(f"\n{'#' * 60}")
    print(f"#  LCF US SWING Funnel  | mode={args.mode}  top={top_n}")
    print(f"#  Horizon: hold 1 week / 1 month  (fundamentals + news + prediction)")
    print(f"#  Started: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}")

    # --- 1. Universe (same 100 stocks as intraday) ---
    from us_intraday_tracker.universe import load_universe
    universe = load_universe(UNIVERSE_FILE)
    symbols = universe.all_symbols()
    print(f"\n[1] Universe: {len(symbols)} symbols across {len(universe.sectors_list())} sectors")

    # --- 2. Scan + rank candidates ---
    from us_intraday_tracker.fluctuation_scanner import FluctuationScanner, ScanConfig
    scanner = FluctuationScanner(
        _build_scan_provider(),
        ScanConfig(intraday_interval=scan_interval_yf, min_fluctuation_pct=0.0),
        sector_of=universe.sector_of,
    )
    print(f"[2] Scanning {len(symbols)} symbols for swing candidates...")
    snapshots = scanner.scan(symbols)
    candidates = scanner.rank_movers(snapshots, top_n=top_n)
    cand_by_symbol = {m.symbol: m for m in candidates}
    print(f"    {len(snapshots)} snapshots -> {len(candidates)} candidates")
    for m in candidates:
        print(f"      {m.symbol:6s} {m.pct_change:+6.2f}%  score={m.fluctuation_score:5.2f}")

    if args.scan_only or not candidates:
        if not candidates:
            print("    No candidates — nothing to analyse.")
        _save({"timestamp": start.isoformat(), "mode": args.mode, "scan_only": True,
               "candidates": [m.to_dict() for m in candidates]})
        return 0

    cand_symbols = [m.symbol for m in candidates]

    # --- 3. Multi-horizon prediction (the swing brain) ---
    print(f"\n[3] Predicting 1wk/1mo outlook for {len(cand_symbols)} candidates "
          f"(interval={deep_interval}, period={deep_period})...")
    from src.pipeline.orchestrator import PipelineOrchestrator
    orchestrator = PipelineOrchestrator(config_path=CONFIG_PATH, mode=args.mode)
    orchestrator.data_processor = _create_swing_data_processor(deep_period, deep_interval)
    try:
        results = orchestrator.predict_future_for_symbols(cand_symbols, index_symbol="^GSPC")
    except Exception as e:
        print(f"  [ERROR] Prediction failed: {e}")
        traceback.print_exc()
        return 1

    # --- 4. Swing decision per candidate ---
    print(f"\n[4] Swing decisions...")
    for r in results:
        symbol = r.get("symbol")
        if not symbol or "error" in r:
            continue
        pred = r.get("prediction", {}) or {}
        r["swing_decision"] = _swing_decision(pred, min_conf, min_hold_gain)
        d = r["swing_decision"]
        print(f"      {symbol:6s} {d['final_decision']:4s} "
              f"(conf {d['final_confidence']:.0%}, 1w {d['one_week_pct']:+.1f}%, 1m {d['one_month_pct']:+.1f}%)")

    # --- 5. Notify ---
    final_actions = [
        r for r in results
        if r.get("swing_decision", {}).get("final_decision") in ("BUY", "SELL")
    ]
    print(f"\n[5] Notify: {len(final_actions)} actionable (BUY/SELL)")
    if final_actions:
        _notify(final_actions, cand_by_symbol)

    # --- 6. Trade into the SWING account ---
    print(f"\n[6] Trade (auto_trade={args.auto_trade})...")
    if args.auto_trade and final_actions:
        _trade(final_actions, cand_by_symbol, cfg, args.mode, order_qty)
    elif not args.auto_trade:
        print("    Auto-trade disabled — notify only.")
    else:
        print("    No actionable signals.")

    # --- 7. Persist ---
    buy = sum(1 for r in final_actions if r["swing_decision"]["final_decision"] == "BUY")
    sell = sum(1 for r in final_actions if r["swing_decision"]["final_decision"] == "SELL")
    _save({
        "timestamp": start.isoformat(),
        "market": "US",
        "strategy": "swing",
        "mode": args.mode,
        "candidates_analyzed": len(cand_symbols),
        "candidates": [m.to_dict() for m in candidates],
        "buy_count": buy,
        "sell_count": sell,
        "results": results,
    })

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\nDone in {elapsed:.1f}s — {buy} BUY, {sell} SELL")
    return 0


def _notify(final_actions, cand_by_symbol):
    try:
        from src.runtime import Notifier
        notifier = Notifier(market="US")
        if not notifier.enabled:
            print("    Notifier not configured (set PUSHOVER_* env or credentials.yaml).")
            return
        lines = ["<b>[SWING] Multi-day signals</b>"]
        for r in final_actions:
            sym = r["symbol"]
            d = r["swing_decision"]
            lines.append(
                f"{d['final_decision']} <b>{sym}</b> "
                f"(conf {d['final_confidence']:.0%}) — hold outlook 1w {d['one_week_pct']:+.1f}%, "
                f"1m {d['one_month_pct']:+.1f}%"
            )
        sent = notifier.send("\n".join(lines), title="[SWING] LCF Swing Signals")
        print(f"    Notify {'sent' if sent else 'skipped'} ({len(final_actions)} signals).")
    except Exception as e:
        print(f"    [WARN] Notify failed: {e}")


def _trade(final_actions, cand_by_symbol, cfg, mode, order_qty):
    try:
        from src.runtime import BrokerRouter
        router = BrokerRouter(cfg)
        if not router.available:
            print("    No broker available — notify only, no orders.")
            return
        print(f"    Brokers firing on each signal: {', '.join(router.available)}")

        for r in final_actions:
            sym = r["symbol"]
            action = r["swing_decision"]["final_decision"]
            m = cand_by_symbol.get(sym)
            price = m.current if m and m.current else 0.0
            if not price:
                continue
            execs = router.execute(sym, action, quantity=order_qty, price=price)
            r["executions"] = execs
            for name, res in execs.items():
                print(f"    [{name}] {action} {sym} x{order_qty} @ {price:.2f} -> "
                      f"{res.get('status')} ({res.get('order_id', '')})")

        # Track swing positions in the dedicated swing portfolio file.
        # PortfolioManager reads judge_decision.payload.decision + trade_plan.current_price,
        # so map our swing_decision into that shape first.
        try:
            from src.main.controllers.portfolio_manager import PortfolioManager
            for r in final_actions:
                sd = r.get("swing_decision", {})
                m = cand_by_symbol.get(r.get("symbol"))
                r["judge_decision"] = {"payload": {
                    "decision": sd.get("final_decision", "HOLD"),
                    "confidence": sd.get("final_confidence", 0.0),
                }}
                r["trade_plan"] = {"current_price": (m.current if m and m.current else 0.0)}
            pm = PortfolioManager(portfolio_path=SWING_PORTFOLIO_PATH, watchlist_path=SWING_WATCHLIST_PATH)
            pm.load()
            pm.process_signals(final_actions, auto_enter=True, auto_exit=True, mode=mode)
        except Exception as e:
            print(f"    [WARN] Swing portfolio update skipped: {e}")
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
