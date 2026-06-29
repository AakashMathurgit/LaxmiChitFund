"""Alpaca broker package — real (paper) order execution for US stocks.

Exposes `AlpacaBroker`, which implements the LCF `BrokerInterface`
(src/main/controllers/trade_executor.py) by calling the Alpaca REST API. It
drops straight into `TradeExecutor` alongside `VestedBroker`.

Credentials live in the git-ignored credentials.yaml under an `alpaca:` section.

Usage::

    from alpaca_broker import AlpacaBroker, AlpacaConfig
    broker = AlpacaBroker.from_credentials()   # reads credentials.yaml
    result = broker.place_order("NVDA", "BUY", 1, price=120.0)
"""

from .alpaca_broker import AlpacaBroker, AlpacaConfig

__all__ = ["AlpacaBroker", "AlpacaConfig"]
