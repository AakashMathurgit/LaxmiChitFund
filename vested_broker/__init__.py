"""Vested broker package — a buy/sell API layer for US stocks.

Exposes a `VestedBroker` that implements the LCF `BrokerInterface`
(src/main/controllers/trade_executor.py), so it drops straight into
`TradeExecutor` for both manual and automated trading.

Modes:
  - "paper" (default): logs intended orders to a JSONL file and returns
    SUCCESS — safe, no real money. Use for the scheduled tracker.
  - "live": NOT IMPLEMENTED. Vested has no public trading API; this is the
    swap-in point for a real session/REST integration.

Manual usage::

    python -m vested_broker.cli buy --symbol NVDA --qty 5 --price 120.50

Programmatic usage::

    from vested_broker import VestedBroker, VestedConfig
    from src.main.controllers.trade_executor import TradeExecutor

    executor = TradeExecutor(VestedBroker(VestedConfig(mode="paper")))
    result = executor.execute("NVDA", "BUY", quantity=5, expected_price=120.50)
"""

from .vested_broker import VestedBroker, VestedConfig

__all__ = ["VestedBroker", "VestedConfig"]
