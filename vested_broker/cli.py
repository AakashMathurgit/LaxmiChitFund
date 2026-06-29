"""Manual buy/sell CLI for the Vested broker layer.

Lets you trigger an order by hand, independent of the intraday tracker. Wraps
`VestedBroker` in the existing `TradeExecutor` so retry + slippage logic is
shared with the automated path.

Usage:
    python -m vested_broker.cli buy  --symbol NVDA --qty 5 --price 120.50
    python -m vested_broker.cli sell --symbol AAPL --qty 3 --price 210.00 --mode paper
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Make the repo root importable so `src` and `vested_broker` resolve when this
# module is run directly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_LCF_ROOT = os.path.dirname(_SCRIPT_DIR)
if _LCF_ROOT not in sys.path:
    sys.path.insert(0, _LCF_ROOT)

from vested_broker.vested_broker import VestedBroker, VestedConfig
from src.main.controllers.trade_executor import TradeExecutor


def main(argv=None):
    parser = argparse.ArgumentParser(description="Vested manual buy/sell")
    parser.add_argument("action", choices=["buy", "sell"], help="Order side")
    parser.add_argument("--symbol", required=True, help="Ticker, e.g. NVDA")
    parser.add_argument("--qty", type=int, required=True, help="Number of shares")
    parser.add_argument("--price", type=float, required=True, help="Expected price")
    parser.add_argument("--mode", default="paper", choices=["paper", "live"],
                        help="Broker mode (default: paper)")
    parser.add_argument("--order-type", default="MARKET", choices=["MARKET", "LIMIT"],
                        help="Order type (default: MARKET)")
    args = parser.parse_args(argv)

    broker = VestedBroker(VestedConfig(mode=args.mode))
    executor = TradeExecutor(broker)

    result = executor.execute(
        symbol=args.symbol,
        action=args.action.upper(),
        quantity=args.qty,
        expected_price=args.price,
        order_type=args.order_type,
    )

    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.status == "SUCCESS" else 1


if __name__ == "__main__":
    sys.exit(main())
