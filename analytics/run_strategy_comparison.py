"""Weekly strategy comparison — intraday (Account 1) vs swing (Account 2).

Pulls real P&L from both Alpaca accounts, has an LLM analyst write insights +
tuning recommendations, pushes a [WEEKLY] summary, and saves the full report.

Usage:
    python analytics/run_strategy_comparison.py
    python analytics/run_strategy_comparison.py --no-notify
"""

import os
import sys
import json
import argparse
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

CONFIG_PATH = os.path.join(_LCF_ROOT, "config.yaml")
REPORT_FILE = os.path.join(_LCF_ROOT, "data", "strategy_comparison.json")
HISTORY_FILE = os.path.join(_LCF_ROOT, "data", "strategy_comparison_history.jsonl")

ANALYST_SYSTEM = (
    "You are a quantitative trading performance analyst. You are given JSON metrics "
    "for two paper-trading strategies that trade the SAME 100-stock universe but with "
    "different horizons: 'intraday' (1-min cadence, immediate profit) and 'swing' "
    "(30-min cadence, multi-day holds chosen on a 1wk/1mo prediction). Both started at "
    "$1,000,000. Compare them objectively and be concrete and actionable. "
    "Return STRICT JSON with keys: "
    "winner (str: 'intraday'|'swing'|'tie'), "
    "headline (str, one sentence), "
    "summary (str, 3-5 sentences on why one is winning), "
    "per_stock_insights (array of short strings about notable symbol-level wins/losses), "
    "recommendations (array of concrete, specific tuning changes to improve profit — "
    "reference thresholds/cadence/universe/sizing where relevant), "
    "risk_flags (array of short strings about risks seen in the data)."
)


def _load_config():
    import yaml
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _llm_analysis(comparison: dict, cfg: dict) -> dict:
    """Ask the LLM analyst for insights. Degrades gracefully if LLM unavailable."""
    try:
        from src.main.agents.adapters.llm_adapter import LLMAdapter
        llm = LLMAdapter.from_config(cfg, base_path=_LCF_ROOT)
        # Trim per-symbol noise to keep the prompt lean.
        slim = {
            name: {k: v for k, v in m.items() if k != "per_symbol"}
            for name, m in comparison.get("metrics", {}).items()
        }
        user = json.dumps(slim, default=str)
        return llm.invoke_json(ANALYST_SYSTEM, user, temperature=0.3, max_tokens=900)
    except Exception as e:
        return {"_llm_error": str(e),
                "winner": comparison.get("winner"),
                "headline": "LLM analysis unavailable — see raw metrics.",
                "summary": "", "per_stock_insights": [], "recommendations": [], "risk_flags": []}


def _format_report(comparison: dict, analysis: dict) -> str:
    m = comparison.get("metrics", {})
    lines = []
    lines.append(f"{'=' * 64}")
    lines.append("  LCF WEEKLY STRATEGY COMPARISON — intraday vs swing")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"{'=' * 64}")
    for name in ("intraday", "swing"):
        if name not in m:
            lines.append(f"\n  [{name}] no data (account keys missing)")
            continue
        a = m[name]
        lines.append(f"\n  [{name.upper()}]  account {a.get('account_number','?')}")
        lines.append(f"    Equity:        ${a['equity']:,.2f}")
        lines.append(f"    Total P&L:     ${a['total_pl']:,.2f}  ({a['total_return_pct']:+.2f}%)")
        lines.append(f"    Realized:      ${a['realized_pl']:,.2f}   Unrealized: ${a['unrealized_pl']:,.2f}")
        lines.append(f"    Trades:        {a['closed_trades']} closed / {a['num_fills']} fills"
                     f"   Win rate: {a['win_rate_pct']}%")
        if a.get("pending_orders"):
            lines.append(f"    Pending:       {a['pending_orders']} orders queued (not yet filled)")
        if a["top_winners"]:
            tw = ", ".join(f"{x['symbol']} ${x['total']:+,.0f}" for x in a["top_winners"][:3])
            lines.append(f"    Top names:     {tw}")
    lines.append(f"\n  WINNER: {str(analysis.get('winner', comparison.get('winner'))).upper()}")
    if analysis.get("headline"):
        lines.append(f"  {analysis['headline']}")
    if analysis.get("summary"):
        lines.append(f"\n  Analysis:\n    {analysis['summary']}")
    for label, key in [("Per-stock insights", "per_stock_insights"),
                       ("Recommendations", "recommendations"),
                       ("Risk flags", "risk_flags")]:
        items = analysis.get(key) or []
        if items:
            lines.append(f"\n  {label}:")
            for it in items:
                lines.append(f"    - {it}")
    lines.append(f"\n{'=' * 64}")
    return "\n".join(lines)


def _notify(comparison: dict, analysis: dict):
    try:
        from src.runtime import Notifier
        notifier = Notifier(market="US")
        if not notifier.enabled:
            print("    Notifier not configured.")
            return
        m = comparison.get("metrics", {})
        lines = ["<b>[WEEKLY] Strategy comparison</b>"]
        for name in ("intraday", "swing"):
            if name in m:
                a = m[name]
                lines.append(f"{name}: ${a['total_pl']:+,.0f} ({a['total_return_pct']:+.2f}%), "
                             f"win {a['win_rate_pct']}%")
        lines.append(f"<b>Winner: {str(analysis.get('winner','?')).upper()}</b>")
        if analysis.get("headline"):
            lines.append(analysis["headline"])
        recs = analysis.get("recommendations") or []
        if recs:
            lines.append("Top fix: " + recs[0])
        notifier.send("\n".join(lines), title="[WEEKLY] LCF Strategy Comparison")
        print("    Pushover sent.")
    except Exception as e:
        print(f"    [WARN] Notify failed: {e}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="LCF weekly strategy comparison")
    parser.add_argument("--no-notify", dest="notify", action="store_false")
    parser.set_defaults(notify=True)
    args = parser.parse_args(argv)

    cfg = _load_config()
    print("Pulling broker truth from both Alpaca accounts...")
    from analytics.strategy_comparator import compare
    comparison = compare()

    if not comparison.get("metrics"):
        print("No account data available (check ALPACA_* / ALPACA_SWING_* keys).")
        return 1

    print("Running LLM performance analyst...")
    analysis = _llm_analysis(comparison, cfg)

    report_text = _format_report(comparison, analysis)
    print("\n" + report_text)

    out = {
        "timestamp": datetime.now().isoformat(),
        "comparison": comparison,
        "analysis": analysis,
        "report_text": report_text,
    }
    try:
        os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        with open(HISTORY_FILE, "a", encoding="utf-8") as hf:
            hf.write(json.dumps(out, default=str) + "\n")
    except Exception as e:
        print(f"[WARN] Save failed: {e}")

    if args.notify:
        _notify(comparison, analysis)

    return 0


if __name__ == "__main__":
    sys.exit(main())
