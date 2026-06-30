"""BrokerRouter — fans a single BUY/SELL out to every available broker.

Promoted from the inline fan-out in us_intraday_tracker/run_us_intraday.py so
that ALL flows execute through one consistent path. Each available broker
(Vested real-slot/paper + Alpaca paper + future real brokers) is wrapped in a
TradeExecutor (retry + slippage) and fired on every order.

Configured from the `vested:` and `alpaca:` blocks in config.yaml; secrets come
from credentials.yaml via the broker factories.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..utils.logger import get_logger


class BrokerRouter:
    def __init__(self, config: Dict[str, Any]):
        self._config = config or {}
        self._logger = get_logger("BrokerRouter")
        self._executors: List[Tuple[str, Any]] = []
        self._alpaca = None  # kept for account-equity queries (position sizing)
        # When True (default), a SELL is only placed if the stock is actually
        # held — never open/increase a short position. Set
        # `trading.sell_only_if_held: false` in config.yaml to allow shorting.
        self._sell_only_if_held = bool(
            self._config.get("trading", {}).get("sell_only_if_held", True)
        )
        self._build()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build(self) -> None:
        from src.main.controllers.trade_executor import TradeExecutor

        vested_cfg = self._config.get("vested", {})
        alpaca_cfg = self._config.get("alpaca", {})

        # 1. Vested (real-broker slot — paper today, no public live API).
        try:
            from vested_broker.vested_broker import VestedBroker, VestedConfig
            vested = VestedBroker(VestedConfig(
                mode=vested_cfg.get("mode", "paper"),
                paper_orders_path=vested_cfg.get("paper_orders_path", "data/vested_paper_orders.jsonl"),
            ))
            if vested.is_available():
                self._executors.append(("vested", TradeExecutor(vested)))
        except Exception as e:
            self._logger.warning(f"Vested init failed: {e}")

        # 2. Alpaca paper trading (real API).
        if alpaca_cfg.get("enabled", True):
            try:
                from alpaca_broker.alpaca_broker import AlpacaBroker
                alpaca = AlpacaBroker.from_credentials()
                if alpaca.is_available():
                    self._executors.append(("alpaca", TradeExecutor(alpaca)))
                    self._alpaca = alpaca
                else:
                    self._logger.info("Alpaca unavailable (no keys in credentials.yaml).")
            except Exception as e:
                self._logger.warning(f"Alpaca init failed: {e}")

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    @property
    def available(self) -> List[str]:
        return [name for name, _ in self._executors]

    def account_equity(self, default: float = 0.0) -> float:
        """Live equity of the Alpaca account this router trades (for sizing)."""
        if self._alpaca is not None:
            eq = self._alpaca.get_account_equity()
            if eq and eq > 0:
                return eq
        return default

    def buying_power(self, default: float = 0.0) -> float:
        if self._alpaca is not None:
            bp = self._alpaca.get_buying_power()
            if bp and bp > 0:
                return bp
        return default

    def portfolio_context(self):
        """Live :class:`PortfolioContext` for the Alpaca account this router trades.

        Fetched in a single bulk call (all positions + equity). Returns an empty
        context when no Alpaca broker is configured/available.
        """
        from .portfolio_context import PortfolioContext

        if self._alpaca is None:
            return PortfolioContext.empty()
        return PortfolioContext.from_alpaca(self._alpaca)

    def holds(self, symbol: str) -> bool:
        """Whether the live (Alpaca) account currently holds `symbol`.

        Used to gate SELLs when `trading.sell_only_if_held` is enabled. If the
        holdings cannot be determined (no Alpaca broker / query error), this
        returns False so a guarded SELL is skipped rather than opening a short.
        """
        if self._alpaca is None:
            return False
        try:
            return self._alpaca.get_position_qty(symbol) > 0
        except Exception as e:
            self._logger.warning(f"holdings check failed for {symbol}: {e}")
            return False

    def execute(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        order_type: str = "MARKET",
    ) -> Dict[str, Dict[str, Any]]:
        """Place the order on every available broker. Returns {broker: result}."""
        # Guard: only sell what we hold (default on). Blocks SELLs that would
        # otherwise open/increase a short position.
        if self._sell_only_if_held and action.upper() == "SELL" and not self.holds(symbol):
            self._logger.info(
                f"[SKIP] SELL {symbol} — not held (sell_only_if_held=True)."
            )
            skip = {
                "status": "SKIPPED",
                "reason": "not_held_sell_only_if_held",
                "symbol": symbol,
                "action": action,
            }
            return {name: dict(skip) for name, _ in self._executors}

        results: Dict[str, Dict[str, Any]] = {}
        for name, executor in self._executors:
            try:
                res = executor.execute(symbol, action, quantity, price, order_type=order_type)
                results[name] = res.to_dict()
            except Exception as e:
                self._logger.warning(f"[{name}] {action} {symbol} failed: {e}")
                results[name] = {"status": "FAILED", "reason": str(e), "symbol": symbol, "action": action}
        return results
