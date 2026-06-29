"""Repeatedly run the US intraday funnel at a configurable interval (default 4 min).

Each cycle runs run_us_intraday.py in a subprocess so a hung yfinance/LLM call
cannot wedge the long-lived loop (same pattern as run_us_loop.py).

Usage:
    python us_intraday_tracker/run_us_intraday_loop.py --interval 4
    python us_intraday_tracker/run_us_intraday_loop.py --interval 4 --max-runs 2 --no-auto-trade
"""

import os
import sys
import time
import subprocess
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
FUNNEL_SCRIPT = os.path.join(_SCRIPT_DIR, "run_us_intraday.py")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LCF US Intraday Loop Runner")
    parser.add_argument("--interval", type=int, default=4, help="Minutes between cycles (default: 4)")
    parser.add_argument("--mode", default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"])
    parser.add_argument("--top", type=int, default=None, help="Top movers per cycle")
    parser.add_argument("--no-auto-trade", dest="auto_trade", action="store_false")
    parser.add_argument("--max-runs", type=int, default=0, help="Max cycles (0=infinite)")
    parser.set_defaults(auto_trade=True)
    args = parser.parse_args()

    interval_secs = args.interval * 60
    run_count = 0

    print(f"\n{'#' * 60}")
    print(f"#  LCF US Intraday LOOP  | interval={args.interval}m  mode={args.mode}")
    print(f"#  AutoTrade={args.auto_trade}  MaxRuns={'inf' if args.max_runs == 0 else args.max_runs}")
    print(f"#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    while True:
        run_count += 1
        if args.max_runs > 0 and run_count > args.max_runs:
            print(f"\n[LOOP] Reached max runs ({args.max_runs}). Stopping.")
            break

        start = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"  INTRADAY CYCLE #{run_count} — {start.strftime('%H:%M:%S')}")
        print(f"{'=' * 60}")

        cmd = [sys.executable, FUNNEL_SCRIPT, "--mode", args.mode]
        if args.top:
            cmd += ["--top", str(args.top)]
        if not args.auto_trade:
            cmd.append("--no-auto-trade")

        try:
            result = subprocess.run(cmd, cwd=_LCF_ROOT, timeout=300)
            elapsed = (datetime.now() - start).total_seconds()
            status = "OK" if result.returncode == 0 else f"EXIT {result.returncode}"
            print(f"\n[LOOP] Cycle #{run_count} finished in {elapsed:.0f}s — {status}")
        except subprocess.TimeoutExpired:
            print(f"\n[LOOP] Cycle #{run_count} TIMED OUT after 300s")
        except KeyboardInterrupt:
            print(f"\n[LOOP] Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[LOOP] Cycle #{run_count} ERROR: {e}")

        if args.max_runs > 0 and run_count >= args.max_runs:
            break

        next_str = datetime.fromtimestamp(datetime.now().timestamp() + interval_secs).strftime('%H:%M:%S')
        print(f"[LOOP] Next cycle at {next_str} (sleeping {args.interval}m)...")
        try:
            time.sleep(interval_secs)
        except KeyboardInterrupt:
            print(f"\n[LOOP] Interrupted during sleep. Exiting.")
            break


if __name__ == "__main__":
    main()
