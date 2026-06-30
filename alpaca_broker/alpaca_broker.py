"""AlpacaBroker — BrokerInterface implementation backed by the Alpaca REST API.

Places real orders against an Alpaca account (paper or live). Implements the
same `BrokerInterface` as MockBroker / VestedBroker so it can be wrapped by
`TradeExecutor` and fanned out alongside the other brokers.

Credentials are read from credentials.yaml (`alpaca:` section) — never from
source. The default endpoint is Alpaca's paper-trading host.

API reference: https://docs.alpaca.markets/reference/postorder
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
import yaml

from src.main.controllers.trade_executor import BrokerInterface, ExecutionResult
from src.utils.logger import get_logger

_CREDENTIALS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "credentials.yaml"
)

_PAPER_ENDPOINT = "https://paper-api.alpaca.markets/v2"


@dataclass
class AlpacaConfig:
    """Configuration for the Alpaca broker adapter."""
    endpoint: str = _PAPER_ENDPOINT
    key_id: str = ""
    secret_key: str = ""
    time_in_force: str = "day"   # day | gtc | ioc | fok
    timeout_secs: float = 10.0


class AlpacaBroker(BrokerInterface):
    """Alpaca brokerage adapter (paper or live, depending on endpoint)."""

    def __init__(self, config: Optional[AlpacaConfig] = None):
        self._config = config or AlpacaConfig()
        self._logger = get_logger("AlpacaBroker")
        self._orders: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_credentials(cls, credentials_path: Optional[str] = None) -> "AlpacaBroker":
        """Build an AlpacaBroker from credentials.yaml (`alpaca:` section)."""
        path = credentials_path or _CREDENTIALS_PATH
        cfg = AlpacaConfig()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    creds = yaml.safe_load(f) or {}
                alpaca = creds.get("alpaca", {}) or {}
                cfg.endpoint = alpaca.get("endpoint", cfg.endpoint)
                cfg.key_id = alpaca.get("key_id", "")
                cfg.secret_key = alpaca.get("secret_key", "")
            except Exception as e:
                get_logger("AlpacaBroker").warning(f"Could not read Alpaca credentials: {e}")

        # Environment variables take precedence (used in cloud / containers).
        cfg.endpoint = os.environ.get("ALPACA_ENDPOINT", cfg.endpoint)
        cfg.key_id = os.environ.get("ALPACA_KEY_ID", cfg.key_id)
        cfg.secret_key = os.environ.get("ALPACA_SECRET_KEY", cfg.secret_key)
        return cls(cfg)

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True if credentials are present (orders can be attempted)."""
        return bool(self._config.key_id and self._config.secret_key)

    @property
    def is_paper(self) -> bool:
        return "paper-api" in (self._config.endpoint or "")

    def _headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self._config.key_id,
            "APCA-API-SECRET-KEY": self._config.secret_key,
            "Content-Type": "application/json",
        }

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
        timestamp = datetime.now().isoformat()

        if not self.is_available():
            return ExecutionResult(
                symbol=symbol, action=action, quantity=quantity, expected_price=price,
                status="FAILED", reason="alpaca_credentials_missing", timestamp=timestamp,
            )

        payload = {
            "symbol": symbol,
            "qty": str(quantity),
            "side": action.lower(),               # buy | sell
            "type": order_type.lower(),           # market | limit
            "time_in_force": self._config.time_in_force,
        }
        if order_type.upper() == "LIMIT":
            payload["limit_price"] = str(price)

        try:
            resp = requests.post(
                f"{self._config.endpoint}/orders",
                json=payload,
                headers=self._headers(),
                timeout=self._config.timeout_secs,
            )
        except Exception as e:
            self._logger.warning(f"[{symbol}] Alpaca request failed: {e}")
            return ExecutionResult(
                symbol=symbol, action=action, quantity=quantity, expected_price=price,
                status="FAILED", reason=f"alpaca_request_error: {e}", timestamp=timestamp,
            )

        if resp.status_code not in (200, 201):
            reason = f"alpaca_http_{resp.status_code}: {resp.text[:160]}"
            self._logger.warning(f"[{symbol}] {reason}")
            return ExecutionResult(
                symbol=symbol, action=action, quantity=quantity, expected_price=price,
                status="FAILED", reason=reason, timestamp=timestamp,
            )

        data = resp.json()
        order_id = data.get("id", "")
        # Market orders usually return "accepted"/"new"/"pending_new" instantly;
        # filled_avg_price is null until the fill lands.
        order_status = data.get("status", "")
        filled_price = data.get("filled_avg_price")
        executed_price = float(filled_price) if filled_price else round(price, 2)

        self._orders[order_id] = order_status
        slippage = round(executed_price - price, 2)
        self._logger.info(
            f"[{'PAPER' if self.is_paper else 'LIVE'}] {action} {quantity} {symbol} "
            f"-> {order_id} ({order_status})"
        )

        return ExecutionResult(
            symbol=symbol,
            action=action,
            quantity=quantity,
            expected_price=price,
            executed_price=executed_price,
            status="SUCCESS",
            order_id=order_id,
            slippage=slippage,
            slippage_pct=round(slippage / price, 4) if price else 0.0,
            reason=order_status,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------
    # Account info (for position sizing)
    # ------------------------------------------------------------------

    def _get_account(self) -> Dict[str, str]:
        if not self.is_available():
            return {}
        try:
            resp = requests.get(
                f"{self._config.endpoint}/account",
                headers=self._headers(),
                timeout=self._config.timeout_secs,
            )
            if resp.status_code == 200:
                return resp.json()
            self._logger.debug(f"account fetch http {resp.status_code}")
        except Exception as e:
            self._logger.debug(f"account fetch failed: {e}")
        return {}

    def get_account_equity(self) -> float:
        info = self._get_account()
        try:
            return float(info.get("equity", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_buying_power(self) -> float:
        info = self._get_account()
        try:
            return float(info.get("buying_power", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_position_qty(self, symbol: str) -> float:
        """Currently held quantity for `symbol` (0.0 if no open position).

        Uses Alpaca's GET /v2/positions/{symbol}, which returns HTTP 404 when no
        position is held. Any error is treated as "unknown" and returns 0.0.
        """
        if not symbol or not self.is_available():
            return 0.0
        try:
            resp = requests.get(
                f"{self._config.endpoint}/positions/{symbol.upper()}",
                headers=self._headers(),
                timeout=self._config.timeout_secs,
            )
            if resp.status_code == 200:
                return float(resp.json().get("qty", 0.0) or 0.0)
            if resp.status_code == 404:
                return 0.0
            self._logger.debug(f"position fetch http {resp.status_code} for {symbol}")
        except Exception as e:
            self._logger.debug(f"position fetch failed for {symbol}: {e}")
        return 0.0

    def get_positions(self) -> List[Dict[str, Any]]:
        """All currently open positions as a list of raw Alpaca position dicts.

        Uses Alpaca's GET /v2/positions (no symbol = every open position) so the
        whole portfolio can be fetched in a single call instead of one request
        per ticker. Returns an empty list on any error or when unavailable.

        Each dict contains (among others): symbol, qty, avg_entry_price,
        current_price, market_value, unrealized_pl, unrealized_plpc.
        """
        if not self.is_available():
            return []
        try:
            resp = requests.get(
                f"{self._config.endpoint}/positions",
                headers=self._headers(),
                timeout=self._config.timeout_secs,
            )
            if resp.status_code == 200:
                data = resp.json()
                return data if isinstance(data, list) else []
            self._logger.debug(f"positions fetch http {resp.status_code}")
        except Exception as e:
            self._logger.debug(f"positions fetch failed: {e}")
        return []

    def get_order_status(self, order_id: str) -> str:
        if not order_id or not self.is_available():
            return "UNKNOWN"
        try:
            resp = requests.get(
                f"{self._config.endpoint}/orders/{order_id}",
                headers=self._headers(),
                timeout=self._config.timeout_secs,
            )
            if resp.status_code == 200:
                return resp.json().get("status", "UNKNOWN")
        except Exception as e:
            self._logger.debug(f"order status error: {e}")
        return self._orders.get(order_id, "UNKNOWN")
