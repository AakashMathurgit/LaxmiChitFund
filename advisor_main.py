"""Entry point for the long-horizon financial advisor.

Usage:
    python advisor_main.py                    # monthly report
    python advisor_main.py --report quarterly
    python advisor_main.py --profile configs/investor_profile.yaml \
                           --tuning  configs/long_term_investor.yaml
"""

from __future__ import annotations

import argparse
import sys

from src.utils.logger import get_logger
from src.pipeline.advisor_orchestrator import AdvisorOrchestrator


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LCF long-horizon financial advisor")
    p.add_argument("--profile", default="configs/investor_profile.yaml",
                   help="Path to investor profile YAML")
    p.add_argument("--tuning", default="configs/long_term_investor.yaml",
                   help="Path to advisor tuning YAML")
    p.add_argument("--report", default="monthly", choices=["monthly", "quarterly"],
                   help="Reporting cadence")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logger = get_logger("advisor_main")
    args = parse_args(argv)
    logger.info(f"Starting advisor pipeline (cadence={args.report})")
    orch = AdvisorOrchestrator(profile_path=args.profile, tuning_path=args.tuning)
    report = orch.run(cadence=args.report)
    logger.info(report.summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
