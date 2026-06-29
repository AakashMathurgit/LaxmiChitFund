"""Repeatedly run the LCF US orchestrator pipeline at a configurable interval.

Usage:
    python us_stock_tracker/run_us_loop.py --interval 15 --stocks 10 --mode adaptive
    python us_stock_tracker/run_us_loop.py --interval 30 --auto-trade
"""

import os
import sys
import time
import subprocess
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
PIPELINE_SCRIPT = os.path.join(_SCRIPT_DIR, "run_us_pipeline.py")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="LCF US Pipeline Loop Runner")
    parser.add_argument("--interval", type=int, default=15,
                        help="Minutes between pipeline runs (default: 15)")
    parser.add_argument("--mode", type=str, default="adaptive",
                        choices=["conservative", "aggressive", "momentum", "value", "scalper", "adaptive"],
                        help="Trading mode (default: adaptive)")
    parser.add_argument("--stocks", type=int, default=10,
                        help="Number of stocks per run (default: 10)")
    parser.add_argument("--source", type=str, default="auto",
                        choices=["auto", "watchlist", "discovery", "file"],
                        help="Symbol source (default: auto)")
    parser.add_argument("--auto-trade", action="store_true",
                        help="Auto-enter BUY positions in portfolio")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Max iterations (0=infinite)")
    args = parser.parse_args()

    interval_secs = args.interval * 60
    run_count = 0

    print(f"\n{'#' * 60}")
    print(f"#  LCF US Orchestrator LOOP Runner")
    print(f"#  Market   : US (NYSE / NASDAQ)")
    print(f"#  Mode     : {args.mode.upper()}")
    print(f"#  Interval : {args.interval} minutes")
    print(f"#  Stocks   : {args.stocks}")
    print(f"#  Source   : {args.source}")
    print(f"#  AutoTrade: {args.auto_trade}")
    print(f"#  Max Runs : {'infinite' if args.max_runs == 0 else args.max_runs}")
    print(f"#  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 60}\n")

    while True:
        run_count += 1
        if args.max_runs > 0 and run_count > args.max_runs:
            print(f"\n[LOOP] Reached max runs ({args.max_runs}). Stopping.")
            break

        start = datetime.now()
        print(f"\n{'=' * 60}")
        print(f"  US LOOP ITERATION #{run_count} — {start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 60}")

        cmd = [
            sys.executable,
            PIPELINE_SCRIPT,
            "--mode", args.mode,
            "--stocks", str(args.stocks),
            "--source", args.source,
            "--show-portfolio",
        ]
        if args.auto_trade:
            cmd.append("--auto-trade")

        try:
            result = subprocess.run(cmd, cwd=_LCF_ROOT, timeout=600)
            elapsed = (datetime.now() - start).total_seconds()
            status = "OK" if result.returncode == 0 else f"EXIT CODE {result.returncode}"
            print(f"\n[LOOP] US Run #{run_count} finished in {elapsed:.0f}s — {status}")
        except subprocess.TimeoutExpired:
            print(f"\n[LOOP] US Run #{run_count} TIMED OUT after 600s")
        except KeyboardInterrupt:
            print(f"\n[LOOP] Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n[LOOP] US Run #{run_count} ERROR: {e}")

        if args.max_runs > 0 and run_count >= args.max_runs:
            break

        next_run = datetime.now().timestamp() + interval_secs
        next_run_str = datetime.fromtimestamp(next_run).strftime('%H:%M:%S')
        print(f"[LOOP] Next US run at {next_run_str} (sleeping {args.interval} min)...")
        print(f"{'-' * 60}")

        try:
            time.sleep(interval_secs)
        except KeyboardInterrupt:
            print(f"\n[LOOP] Interrupted during sleep. Exiting.")
            break


if __name__ == "__main__":
    main()
