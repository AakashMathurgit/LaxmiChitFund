"""Performance Tracker ΓÇö computes trading performance metrics.

Reads trade history from PortfolioManager and computes:
win rate, average P&L, max drawdown, Sharpe ratio, slippage stats.

Usage:
    tracker = PerformanceTracker(portfolio_manager)
    metrics = tracker.compute_metrics()
    print(tracker.summary())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math


@dataclass
class PerformanceMetrics:
    """Computed trading performance metrics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0          # 0-1

    total_pnl: float = 0.0
    avg_profit: float = 0.0        # avg of winning trades
    avg_loss: float = 0.0          # avg of losing trades (negative)
    largest_win: float = 0.0
    largest_loss: float = 0.0

    profit_factor: float = 0.0     # gross profit / gross loss
    expectancy: float = 0.0        # avg pnl per trade

    max_drawdown: float = 0.0      # largest peak-to-trough in cumulative P&L
    max_drawdown_pct: float = 0.0

    avg_slippage: float = 0.0
    total_slippage: float = 0.0

    avg_holding_days: float = 0.0

    # Sharpe (simplified, using trade returns)
    sharpe_ratio: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "total_pnl": round(self.total_pnl, 2),
            "avg_profit": round(self.avg_profit, 2),
            "avg_loss": round(self.avg_loss, 2),
            "largest_win": round(self.largest_win, 2),
            "largest_loss": round(self.largest_loss, 2),
            "profit_factor": round(self.profit_factor, 2),
            "expectancy": round(self.expectancy, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "avg_slippage": round(self.avg_slippage, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
        }


class PerformanceTracker:
    """Computes performance metrics from trade history."""

    def __init__(self, portfolio_manager: Any):
        """
        Parameters
        ----------
        portfolio_manager : PortfolioManager
            Must have .get_trade_history() and .get_portfolio_value() methods.
        """
        self._pm = portfolio_manager

    def compute_metrics(self) -> PerformanceMetrics:
        """Compute all performance metrics from trade history."""
        history = self._pm.get_trade_history()
        closed = [t for t in history if t.status == "CLOSED"]

        if not closed:
            return PerformanceMetrics()

        pnls = [t.pnl for t in closed]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        slippages = [t.slippage for t in closed if t.slippage != 0]

        total_trades = len(closed)
        winning = len(winners)
        losing = len(losers)

        gross_profit = sum(winners) if winners else 0.0
        gross_loss = abs(sum(losers)) if losers else 0.0

        # Drawdown from cumulative P&L curve
        cum_pnl = []
        running = 0.0
        for p in pnls:
            running += p
            cum_pnl.append(running)

        max_dd, max_dd_pct = self._compute_drawdown(cum_pnl)

        # Sharpe ratio (simplified: mean / std of trade returns)
        pnl_pcts = [t.pnl_pct for t in closed if t.pnl_pct != 0]
        sharpe = self._compute_sharpe(pnl_pcts)

        # Holding days
        holding_days = []
        for t in closed:
            if t.entry_date and t.exit_date:
                try:
                    from datetime import datetime
                    entry_dt = datetime.strptime(t.entry_date, "%Y-%m-%d")
                    exit_dt = datetime.strptime(t.exit_date, "%Y-%m-%d")
                    holding_days.append((exit_dt - entry_dt).days)
                except ValueError:
                    pass

        return PerformanceMetrics(
            total_trades=total_trades,
            winning_trades=winning,
            losing_trades=losing,
            win_rate=winning / total_trades if total_trades else 0.0,
            total_pnl=sum(pnls),
            avg_profit=sum(winners) / len(winners) if winners else 0.0,
            avg_loss=sum(losers) / len(losers) if losers else 0.0,
            largest_win=max(winners) if winners else 0.0,
            largest_loss=min(losers) if losers else 0.0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else float("inf"),
            expectancy=sum(pnls) / total_trades if total_trades else 0.0,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            avg_slippage=sum(slippages) / len(slippages) if slippages else 0.0,
            total_slippage=sum(slippages),
            avg_holding_days=sum(holding_days) / len(holding_days) if holding_days else 0.0,
            sharpe_ratio=sharpe,
        )

    @staticmethod
    def _compute_drawdown(cum_pnl: List[float]) -> tuple[float, float]:
        """Compute max drawdown from cumulative P&L series."""
        if not cum_pnl:
            return 0.0, 0.0
        peak = cum_pnl[0]
        max_dd = 0.0
        for val in cum_pnl:
            peak = max(peak, val)
            dd = peak - val
            max_dd = max(max_dd, dd)
        # Percentage relative to peak
        max_dd_pct = max_dd / peak if peak > 0 else 0.0
        return max_dd, max_dd_pct

    @staticmethod
    def _compute_sharpe(returns: List[float], risk_free: float = 0.0) -> float:
        """Simplified Sharpe ratio from list of trade return percentages."""
        if len(returns) < 2:
            return 0.0
        mean_ret = sum(returns) / len(returns) - risk_free
        variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0:
            return 0.0
        # Annualize assuming ~50 trades per year
        return round((mean_ret / std) * math.sqrt(50), 2)

    def summary(self) -> str:
        """Human-readable performance summary."""
        m = self.compute_metrics()
        lines = [
            "=" * 50,
            "  TRADING PERFORMANCE SUMMARY",
            "=" * 50,
            f"  Total Trades:     {m.total_trades}",
            f"  Win Rate:         {m.win_rate:.1%}",
            f"  Total P&L:        Rs.{m.total_pnl:,.2f}",
            f"  Avg Profit:       Rs.{m.avg_profit:,.2f}",
            f"  Avg Loss:         Rs.{m.avg_loss:,.2f}",
            f"  Largest Win:      Rs.{m.largest_win:,.2f}",
            f"  Largest Loss:     Rs.{m.largest_loss:,.2f}",
            f"  Profit Factor:    {m.profit_factor:.2f}",
            f"  Max Drawdown:     Rs.{m.max_drawdown:,.2f} ({m.max_drawdown_pct:.1%})",
            f"  Sharpe Ratio:     {m.sharpe_ratio:.2f}",
            f"  Avg Slippage:     Rs.{m.avg_slippage:.2f}",
            f"  Avg Hold (days):  {m.avg_holding_days:.1f}",
            "=" * 50,
        ]
        return "\n".join(lines)
