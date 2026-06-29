"""LCF unified entrypoint — run any flow, run all, or schedule them always-on.

This is what the cloud container runs (`python lcf.py schedule`). It wires the 5
flows to one shared runtime (config, brokers, notifier), records every run to a
combined store, flags cross-flow conflicts, and pings a heartbeat so you know
it's alive.

Usage:
    python lcf.py run us-intraday --top 8
    python lcf.py run all
    python lcf.py schedule                 # always-on (cloud)
    python lcf.py schedule --max-ticks 1   # bounded, for testing
    python lcf.py list
"""

from __future__ import annotations

import os
import sys
import time
import argparse
from datetime import datetime, timezone, timedelta

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.runtime import LCFRuntime
from src.flows import all_flows, get_flow
from src.flows.combined_store import record, detect_conflicts

CONFIG_PATH = os.path.join(_ROOT, "config.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _opts_from_args(args) -> dict:
    opts = {}
    for k in ("mode", "stocks", "top", "report"):
        v = getattr(args, k, None)
        if v is not None:
            opts[k] = v
    if getattr(args, "auto_trade", None) is not None:
        opts["auto_trade"] = args.auto_trade
    return opts


def _us_market_open(now_utc: datetime) -> bool:
    """Rough US regular-hours gate: Mon-Fri, 13:30-20:00 UTC (09:30-16:00 ET)."""
    if now_utc.weekday() >= 5:
        return False
    minutes = now_utc.hour * 60 + now_utc.minute
    return (13 * 60 + 30) <= minutes <= (20 * 60)


def _cadence_seconds(cadence: str, config: dict) -> int:
    """Translate a flow cadence string into a minimum seconds-between-runs."""
    cadence = (cadence or "").strip().lower()
    if cadence.endswith("m") and cadence[:-1].isdigit():
        return int(cadence[:-1]) * 60
    return {
        "daily": 24 * 3600,
        "weekly": 7 * 24 * 3600,
        "monthly": 30 * 24 * 3600,
    }.get(cadence, 24 * 3600)


def _print_result(res) -> None:
    status = "OK" if res.ok else f"FAILED: {res.error}"
    print(f"  [{res.flow}] {status} — {res.summary}")
    for d in res.decisions:
        if d.action not in ("HOLD",):
            print(f"      {d.action:5s} {d.symbol:6s} (conf {d.confidence:.0%}) {d.detail}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_list(rt, args):
    print("Available flows:")
    for f in all_flows():
        gate = " [market-hours]" if f.market_hours_only else ""
        print(f"  {f.name:12s} cadence={f.cadence:8s} horizon={f.horizon}{gate}")


def cmd_run(rt, args):
    opts = _opts_from_args(args)
    if args.flow == "all":
        return _run_all(rt, opts)
    flow = get_flow(args.flow)
    print(f"Running flow: {flow.name}")
    res = flow.run(rt, **opts)
    _print_result(res)
    record([res])
    return 0 if res.ok else 1


def _run_all(rt, opts) -> int:
    print(f"Running ALL flows ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    results = []
    for flow in all_flows():
        try:
            results.append(flow.run(rt, **opts))
        except Exception as e:
            print(f"  [{flow.name}] crashed: {e}")
    print("\nSweep summary:")
    for res in results:
        _print_result(res)

    conflicts = detect_conflicts(results)
    if conflicts:
        print("\n⚠ Cross-flow conflicts:")
        for c in conflicts:
            calls = ", ".join(f"{x['flow']}:{x['action']}" for x in c["calls"])
            print(f"  {c['symbol']}: {calls}")
    record(results)
    _notify_sweep(rt, results, conflicts)
    return 0


def _notify_sweep(rt, results, conflicts) -> None:
    actionable = [
        d for r in results for d in r.decisions
        if d.action not in ("HOLD",)
    ]
    if not actionable and not conflicts:
        return
    lines = []
    for d in actionable[:20]:
        lines.append(f"{d.action} {d.symbol} ({d.horizon}, {d.confidence:.0%})")
    for c in conflicts:
        lines.append("⚠ conflict " + c["symbol"] + ": " +
                     ", ".join(f"{x['flow']}:{x['action']}" for x in c["calls"]))
    rt.notifier("US").send("\n".join(lines), title="LCF Daily Sweep")


def cmd_schedule(rt, args):
    flows = all_flows()
    config = rt.config

    def _interval_for(f) -> int:
        # Intraday cadence can be overridden by config.intraday.scan_interval_min.
        if f.name == "us-intraday":
            mins = config.get("intraday", {}).get("scan_interval_min")
            if mins:
                return int(mins) * 60
        return _cadence_seconds(f.cadence, config)

    intervals = {f.name: _interval_for(f) for f in flows}
    last_run = {f.name: 0.0 for f in flows}
    last_heartbeat = 0.0
    heartbeat_secs = max(args.heartbeat_min, 1) * 60
    tick = max(args.tick_seconds, 5)
    started = time.time()

    print(f"\n{'#' * 60}")
    print(f"#  LCF SCHEDULER — {len(flows)} flows, tick={tick}s, heartbeat={args.heartbeat_min}m")
    for f in flows:
        print(f"#    {f.name:12s} every {intervals[f.name]}s"
              + (" (market-hours)" if f.market_hours_only else ""))
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    ticks = 0
    while True:
        ticks += 1
        now = time.time()
        now_utc = datetime.now(timezone.utc)

        for f in flows:
            due = (now - last_run[f.name]) >= intervals[f.name]
            if not due:
                continue
            if f.market_hours_only and not _us_market_open(now_utc):
                continue
            print(f"[{datetime.now().strftime('%H:%M:%S')}] running {f.name}...")
            try:
                res = f.run(rt)
                _print_result(res)
                record([res])
            except Exception as e:
                print(f"  [{f.name}] crashed: {e}")
            last_run[f.name] = time.time()

        # Heartbeat
        if (now - last_heartbeat) >= heartbeat_secs:
            uptime_h = (now - started) / 3600
            msg = f"LCF alive — tick {ticks}, uptime {uptime_h:.1f}h"
            print(f"[heartbeat] {msg}")
            rt.notifier("US").send(msg, title="LCF Heartbeat")
            last_heartbeat = now

        if args.max_ticks and ticks >= args.max_ticks:
            print(f"[scheduler] reached max-ticks ({args.max_ticks}), exiting.")
            break
        time.sleep(tick)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="LCF unified entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List available flows")

    p_run = sub.add_parser("run", help="Run one flow (or 'all')")
    p_run.add_argument("flow", help="Flow name or 'all'")
    p_run.add_argument("--mode", default=None)
    p_run.add_argument("--stocks", type=int, default=None)
    p_run.add_argument("--top", type=int, default=None)
    p_run.add_argument("--report", default=None)
    p_run.add_argument("--auto-trade", dest="auto_trade", action="store_true", default=None)
    p_run.add_argument("--no-auto-trade", dest="auto_trade", action="store_false")

    p_sched = sub.add_parser("schedule", help="Run all flows always-on, on cadence")
    p_sched.add_argument("--tick-seconds", type=int, default=60)
    p_sched.add_argument("--heartbeat-min", type=int, default=60)
    p_sched.add_argument("--max-ticks", type=int, default=0, help="0 = run forever")

    args = parser.parse_args(argv)
    rt = LCFRuntime.from_config(CONFIG_PATH)

    if args.command == "list":
        return cmd_list(rt, args)
    if args.command == "run":
        return cmd_run(rt, args)
    if args.command == "schedule":
        return cmd_schedule(rt, args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
