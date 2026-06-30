"""Portfolio context — a broker-agnostic snapshot of what we currently hold.

A single :class:`PortfolioContext` is fetched once per run and threaded into the
decision/execution path so agents and sizing logic know what is actually held:

* US flows (intraday / swing / US daily) source it from **Alpaca** live positions
  via :meth:`PortfolioContext.from_alpaca`.
* The NSE / India daily flow sources it from the local **PortfolioManager** JSON
  (Alpaca does not hold Indian equities) via
  :meth:`PortfolioContext.from_portfolio_manager`.

Both factories produce the **same shape** so downstream code (SELL sizing, BUY
exposure gating, LLM holdings text) stays source-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class Position:
    """A single open position, normalised across brokers."""

    symbol: str
    qty: float = 0.0
    avg_entry_price: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0  # fraction, e.g. 0.066 == +6.6%

    def summary(self) -> str:
        pnl_pct = self.unrealized_pnl_pct * 100.0
        return (
            f"{self.symbol}: {self.qty:g} sh @ avg ${self.avg_entry_price:,.2f}, "
            f"now ${self.current_price:,.2f}, P&L {pnl_pct:+.1f}% "
            f"(${self.unrealized_pnl:+,.0f})"
        )


@dataclass
class PortfolioContext:
    """Live snapshot of held positions plus account totals."""

    positions: Dict[str, Position] = field(default_factory=dict)
    total_equity: float = 0.0
    cash: float = 0.0
    long_market_value: float = 0.0
    source: str = "unknown"  # "alpaca" | "portfolio_manager" | "empty"

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def has(self, symbol: str) -> bool:
        return self.held_qty(symbol) > 0

    def position(self, symbol: str) -> Optional[Position]:
        return self.positions.get((symbol or "").upper())

    def held_qty(self, symbol: str) -> float:
        pos = self.position(symbol)
        return pos.qty if pos else 0.0

    def exposure_pct(self, symbol: str) -> float:
        """Position market value as a fraction of total equity (0.0 if unknown)."""
        pos = self.position(symbol)
        if not pos or self.total_equity <= 0:
            return 0.0
        return abs(pos.market_value) / self.total_equity

    # ------------------------------------------------------------------
    # LLM text
    # ------------------------------------------------------------------

    def holding_note(self, symbol: str) -> str:
        """One-line holdings note for a single symbol, for LLM prompts."""
        pos = self.position(symbol)
        if not pos or pos.qty <= 0:
            return f"You do NOT currently hold {symbol.upper()} (no open position)."
        return "You currently HOLD " + pos.summary() + "."

    def to_llm_text(self, focus_symbols: Optional[List[str]] = None) -> str:
        """Compact holdings block for injecting into an LLM prompt."""
        if not self.positions:
            return "Current portfolio: no open positions (all cash)."
        lines = ["Current portfolio holdings (live):"]
        for sym in sorted(self.positions):
            lines.append("  - " + self.positions[sym].summary())
        if self.total_equity > 0:
            lines.append(
                f"Total equity ${self.total_equity:,.0f}, cash ${self.cash:,.0f}."
            )
        if focus_symbols:
            for sym in focus_symbols:
                lines.append(self.holding_note(sym))
        return "\n".join(lines)

    def summary_line(self) -> str:
        return (
            f"{len(self.positions)} positions, equity ${self.total_equity:,.0f}, "
            f"cash ${self.cash:,.0f} [{self.source}]"
        )

    # ------------------------------------------------------------------
    # Factories
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls) -> "PortfolioContext":
        return cls(source="empty")

    @classmethod
    def from_alpaca(cls, alpaca: Any) -> "PortfolioContext":
        """Build from an :class:`AlpacaBroker` (US live positions)."""
        if alpaca is None or not getattr(alpaca, "is_available", lambda: False)():
            return cls.empty()
        try:
            raw = alpaca.get_positions()
        except Exception:
            raw = []

        positions: Dict[str, Position] = {}
        long_mv = 0.0
        for item in raw or []:
            sym = (item.get("symbol") or "").upper()
            if not sym:
                continue
            mv = _to_float(item.get("market_value"))
            positions[sym] = Position(
                symbol=sym,
                qty=_to_float(item.get("qty")),
                avg_entry_price=_to_float(item.get("avg_entry_price")),
                current_price=_to_float(item.get("current_price")),
                market_value=mv,
                unrealized_pnl=_to_float(item.get("unrealized_pl")),
                unrealized_pnl_pct=_to_float(item.get("unrealized_plpc")),
            )
            long_mv += mv

        equity = 0.0
        try:
            equity = _to_float(alpaca.get_account_equity())
        except Exception:
            equity = 0.0
        cash = max(equity - long_mv, 0.0)

        return cls(
            positions=positions,
            total_equity=equity,
            cash=cash,
            long_market_value=long_mv,
            source="alpaca",
        )

    @classmethod
    def from_portfolio_manager(cls, pm: Any) -> "PortfolioContext":
        """Build from a :class:`PortfolioManager` (local JSON, e.g. India/NSE)."""
        if pm is None:
            return cls.empty()
        state = getattr(pm, "_state", None)
        if state is None:
            return cls.empty()

        positions: Dict[str, Position] = {}
        long_mv = 0.0
        for h in getattr(state, "holdings", []) or []:
            sym = (getattr(h, "symbol", "") or "").upper()
            if not sym:
                continue
            qty = _to_float(getattr(h, "quantity", 0))
            entry = _to_float(getattr(h, "entry_price", 0.0))
            current = _to_float(getattr(h, "current_price", 0.0)) or entry
            mv = qty * current
            pnl = _to_float(getattr(h, "unrealized_pnl", 0.0))
            cost = qty * entry
            pnl_pct = (pnl / cost) if cost > 0 else 0.0
            positions[sym] = Position(
                symbol=sym,
                qty=qty,
                avg_entry_price=entry,
                current_price=current,
                market_value=mv,
                unrealized_pnl=pnl,
                unrealized_pnl_pct=pnl_pct,
            )
            long_mv += mv

        cash = _to_float(getattr(state, "cash", 0.0))
        return cls(
            positions=positions,
            total_equity=cash + long_mv,
            cash=cash,
            long_market_value=long_mv,
            source="portfolio_manager",
        )
