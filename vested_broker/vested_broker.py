"""VestedBroker — BrokerInterface implementation for the Vested app.

Implements the abstract `BrokerInterface` from the LCF trade executor so it can
be wrapped by `TradeExecutor` (retry + slippage handling) exactly like
`MockBroker`.

Two modes (see `VestedConfig.mode`):

  - "paper" (DEFAULT): paper trading. Orders are appended to a JSONL ledger
    (`data/vested_paper_orders.jsonl`) and reported as SUCCESS with zero
    slippage. No real money is touched. This is what the scheduled intraday
    tracker uses.

  - "live": NOT IMPLEMENTED. Vested exposes no public trading API, so a real
    integration would have to drive their private session/REST endpoints. The
    methods raise NotImplementedError and mark exactly where that code goes.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

# Import the existing broker contract so VestedBroker is a drop-in for MockBroker.
from src.main.controllers.trade_executor import BrokerInterface, ExecutionResult
from src.utils.logger import get_logger


@dataclass
class VestedConfig:
    """Configuration for the Vested broker adapter."""
    mode: str = "paper"  # "paper" | "live"
    paper_orders_path: str = "data/vested_paper_orders.jsonl"
    # Reserved for a future live integration (loaded from credentials.yaml).
    username: str = ""
    token: str = ""


class VestedBroker(BrokerInterface):
    """Vested brokerage adapter.

    Usage::

        broker = VestedBroker(VestedConfig(mode="paper"))
        executor = TradeExecutor(broker)
        result = executor.execute("NVDA", "BUY", 5, expected_price=120.50)
    """

    def __init__(self, config: Optional[VestedConfig] = None):
        self._config = config or VestedConfig()
        self._logger = get_logger("VestedBroker")
        self._orders: Dict[str, str] = {}

        if self._config.mode == "paper":
            # Ensure the ledger directory exists.
            os.makedirs(os.path.dirname(self._config.paper_orders_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Availability — used by the tracker to decide notify-only vs trade.
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Whether this broker can actually place orders right now.

        Paper mode is always available. Live mode requires credentials AND a
        real implementation (which does not exist yet), so it is unavailable.
        """
        if self._config.mode == "paper":
            return True
        # mode == "live": needs both credentials and a real implementation.
        return False

    # ------------------------------------------------------------------
    # BrokerInterface
    # ------------------------------------------------------------------

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str = "MARKET",
    ) -> ExecutionResult:
        if self._config.mode == "live":
            return self._place_order_live(symbol, action, quantity, price, order_type)
        return self._place_order_paper(symbol, action, quantity, price, order_type)

    def get_order_status(self, order_id: str) -> str:
        return self._orders.get(order_id, "UNKNOWN")

    # ------------------------------------------------------------------
    # Paper trading (STUB — no real money, Vested has no public trading API)
    # ------------------------------------------------------------------

    def _place_order_paper(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str,
    ) -> ExecutionResult:
        order_id = f"PAPER-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now().isoformat()

        # Paper fills at the expected price (zero slippage).
        result = ExecutionResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            expected_price=price,
            executed_price=round(price, 2),
            status="SUCCESS",
            order_id=order_id,
            slippage=0.0,
            slippage_pct=0.0,
            timestamp=timestamp,
        )

        record = result.to_dict()
        record.update({"mode": "paper", "order_type": order_type, "broker": "vested"})
        try:
            with open(self._config.paper_orders_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:
            self._logger.warning(f"Could not write paper order ledger: {e}")

        self._orders[order_id] = "COMPLETE"
        self._logger.info(
            f"[PAPER] {action} {quantity} {symbol} @ {price} -> {order_id} (logged)"
        )
        return result

    # ------------------------------------------------------------------
    # Live trading (NOT IMPLEMENTED — swap-in point for a real integration)
    # ------------------------------------------------------------------

    def _place_order_live(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str,
    ) -> ExecutionResult:
        # IMPLEMENT HERE: authenticate against Vested (self._config.username /
        # self._config.token), submit the order via their session/REST API,
        # poll for the fill, and translate the response into an ExecutionResult.
        raise NotImplementedError(
            "VestedBroker live mode is not implemented — Vested has no public "
            "trading API. Implement the session/REST integration in "
            "_place_order_live, or use mode='paper'."
        )
