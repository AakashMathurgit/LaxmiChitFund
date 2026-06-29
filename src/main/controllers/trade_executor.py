"""Trade Executor ΓÇö broker API interface with retry logic and slippage tracking.

Provides an abstract BrokerInterface and a MockBroker for testing.
Real broker implementations (Zerodha, Upstox) can be plugged in later.

Usage:
    executor = TradeExecutor(MockBroker())
    result = executor.execute("TCS", "BUY", qty=10, expected_price=3920.0)
    if result.status == "SUCCESS":
        portfolio_manager.open_position(...)
"""

from __future__ import annotations

import time
import uuid
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...utils.logger import get_logger


@dataclass
class ExecutionResult:
    """Result of a trade execution attempt."""
    symbol: str
    action: str                    # BUY | SELL
    quantity: int
    expected_price: float
    executed_price: float = 0.0
    status: str = "PENDING"        # SUCCESS | FAILED | PENDING
    order_id: str = ""
    slippage: float = 0.0          # executed - expected
    slippage_pct: float = 0.0
    retry_count: int = 0
    reason: str = ""               # failure reason if FAILED
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "quantity": self.quantity,
            "expected_price": self.expected_price,
            "executed_price": self.executed_price,
            "status": self.status,
            "order_id": self.order_id,
            "slippage": round(self.slippage, 2),
            "slippage_pct": round(self.slippage_pct, 4),
            "retry_count": self.retry_count,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class BrokerInterface(ABC):
    """Abstract broker API. Implement for real broker integration."""

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str = "MARKET",
    ) -> ExecutionResult:
        """Place an order and return execution result."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> str:
        """Check status of an existing order."""
        ...


class MockBroker(BrokerInterface):
    """Mock broker for testing ΓÇö always succeeds with small random slippage."""

    def __init__(self, slippage_pct: float = 0.001, fail_rate: float = 0.0):
        self._slippage_pct = slippage_pct   # ┬▒0.1% default
        self._fail_rate = fail_rate         # 0% fail rate by default
        self._orders: Dict[str, str] = {}

    def place_order(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str = "MARKET",
    ) -> ExecutionResult:
        order_id = f"MOCK-{uuid.uuid4().hex[:8].upper()}"

        # Simulate random failure
        if random.random() < self._fail_rate:
            return ExecutionResult(
                symbol=symbol,
                action=action,
                quantity=quantity,
                expected_price=price,
                status="FAILED",
                reason="mock_random_failure",
                timestamp=datetime.now().isoformat(),
            )

        # Simulate slippage: BUY slightly higher, SELL slightly lower
        direction = 1.0 if action == "BUY" else -1.0
        slip = price * self._slippage_pct * random.uniform(0, 1) * direction
        executed = round(price + slip, 2)

        self._orders[order_id] = "COMPLETE"

        return ExecutionResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            expected_price=price,
            executed_price=executed,
            status="SUCCESS",
            order_id=order_id,
            slippage=round(executed - price, 2),
            slippage_pct=round((executed - price) / price, 4) if price else 0.0,
            timestamp=datetime.now().isoformat(),
        )

    def get_order_status(self, order_id: str) -> str:
        return self._orders.get(order_id, "UNKNOWN")


@dataclass
class ExecutorConfig:
    """Configuration for trade executor retry logic."""
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    max_slippage_pct: float = 0.005   # Reject if slippage > 0.5%


class TradeExecutor:
    """Wraps a BrokerInterface with retry logic, slippage validation, and logging."""

    def __init__(self, broker: BrokerInterface, config: Optional[ExecutorConfig] = None):
        self._broker = broker
        self._config = config or ExecutorConfig()
        self._logger = get_logger("TradeExecutor")
        self._execution_log: List[ExecutionResult] = []

    def execute(
        self,
        symbol: str,
        action: str,
        quantity: int,
        expected_price: float,
        order_type: str = "MARKET",
    ) -> ExecutionResult:
        """Execute a trade with retry logic.

        Returns ExecutionResult with status SUCCESS or FAILED.
        """
        last_result = None

        for attempt in range(self._config.max_retries):
            self._logger.info(
                f"[{symbol}] {action} {quantity} @ {expected_price} "
                f"(attempt {attempt + 1}/{self._config.max_retries})"
            )

            result = self._broker.place_order(
                symbol=symbol,
                action=action,
                quantity=quantity,
                price=expected_price,
                order_type=order_type,
            )
            result.retry_count = attempt + 1

            if result.status == "SUCCESS":
                # Validate slippage
                if abs(result.slippage_pct) > self._config.max_slippage_pct:
                    self._logger.warning(
                        f"[{symbol}] Slippage {result.slippage_pct:.4f} "
                        f"exceeds max {self._config.max_slippage_pct}"
                    )
                    # Still accept ΓÇö just warn

                self._logger.info(
                    f"[{symbol}] {action} SUCCESS @ {result.executed_price} "
                    f"(slippage: {result.slippage:+.2f})"
                )
                self._execution_log.append(result)
                return result

            # Failed ΓÇö retry
            last_result = result
            self._logger.warning(
                f"[{symbol}] {action} FAILED: {result.reason} "
                f"(attempt {attempt + 1})"
            )
            if attempt < self._config.max_retries - 1:
                time.sleep(self._config.retry_delay_seconds)

        # All retries exhausted
        final = last_result or ExecutionResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            expected_price=expected_price,
            status="FAILED",
            reason="max_retries_exhausted",
            retry_count=self._config.max_retries,
            timestamp=datetime.now().isoformat(),
        )
        self._execution_log.append(final)
        return final

    def get_execution_log(self) -> List[ExecutionResult]:
        return list(self._execution_log)

    def get_avg_slippage(self) -> float:
        """Average slippage across successful executions."""
        successful = [r for r in self._execution_log if r.status == "SUCCESS"]
        if not successful:
            return 0.0
        return sum(r.slippage for r in successful) / len(successful)
