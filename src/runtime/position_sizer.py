"""PositionSizer — single source of truth for how many shares an order buys/sells.

Both the intraday and swing flows call this so their position sizes are computed
identically, making their P&L directly comparable. Configured from the
`position_sizing:` block in config.yaml.

Modes (config `method`):
  - "percent_equity" (default/active): spend a fixed % of account equity per
    position. Simple, fair, comparable. e.g. 10% of $1M -> $100k per name.
  - "risk_based" (available, disabled by default): risk a fixed % of equity per
    trade, sized by the stop-loss distance. The professionally-correct method —
    each trade risks the same dollar amount regardless of price/volatility.
  - "fixed_dollar": a flat dollar amount per position.

Every result is clamped by `max_position_pct` (never over-concentrate in one
name) and, when provided, available buying power.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..utils.logger import get_logger


class PositionSizer:
    def __init__(self, config: Dict[str, Any]):
        ps = (config or {}).get("position_sizing", {}) or {}
        self.method = str(ps.get("method", "percent_equity")).lower()
        self.percent_equity_pct = float(ps.get("percent_equity_pct", 10.0))
        self.risk_per_trade_pct = float(ps.get("risk_per_trade_pct", 1.0))
        self.max_position_pct = float(ps.get("max_position_pct", 20.0))
        self.fixed_dollar = float(ps.get("fixed_dollar", 5000))
        self.default_stop_pct = ps.get("default_stop_pct", {}) or {}
        self._logger = get_logger("PositionSizer")

    def size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float = 0.0,
        strategy: str = "intraday",
        buying_power: Optional[float] = None,
    ) -> int:
        """Return the integer share quantity for an order (>= 0)."""
        if entry_price <= 0 or equity <= 0:
            return 0

        if self.method == "risk_based":
            shares = self._risk_based(equity, entry_price, stop_loss, strategy)
        elif self.method == "fixed_dollar":
            shares = int(self.fixed_dollar / entry_price)
        else:  # percent_equity (default)
            shares = int((equity * self.percent_equity_pct / 100.0) / entry_price)

        # Clamp: never exceed max_position_pct of equity in one name.
        max_by_notional = int((equity * self.max_position_pct / 100.0) / entry_price)
        if max_by_notional > 0:
            shares = min(shares, max_by_notional)

        # Clamp: never exceed available buying power.
        if buying_power is not None and buying_power > 0:
            shares = min(shares, int(buying_power / entry_price))

        return max(shares, 0)

    def _risk_based(self, equity: float, entry: float, stop: float, strategy: str) -> int:
        """shares = (equity * risk%) / (entry - stop). Falls back to a default
        stop distance when the strategy didn't supply one (e.g. the swing
        prediction path)."""
        if not stop or stop <= 0 or stop >= entry:
            stop_pct = float(self.default_stop_pct.get(strategy, 5.0))
            stop = entry * (1.0 - stop_pct / 100.0)
        risk_per_share = entry - stop
        if risk_per_share <= 0:
            return 0
        risk_budget = equity * self.risk_per_trade_pct / 100.0
        return int(math.floor(risk_budget / risk_per_share))
