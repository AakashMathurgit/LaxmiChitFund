"""Backtest/Evaluation Agent — validates signal quality with historical data.

Performs:
- Walk-forward testing of trading signals
- Performance metrics (Sharpe, Sortino, max drawdown, win rate)
- Stability analysis and overfitting detection
- Signal quality scoring

This agent prevents blind trust in untested signals.
"""

from __future__ import annotations

import os
import math
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError

# Debug flag
DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


class SignalQuality(str, Enum):
    """Signal quality classifications."""
    EXCELLENT = "excellent"    # Sharpe > 2, stable
    GOOD = "good"              # Sharpe 1-2
    FAIR = "fair"              # Sharpe 0.5-1
    POOR = "poor"              # Sharpe 0-0.5
    UNRELIABLE = "unreliable"  # Sharpe < 0 or unstable


@dataclass
class TradeRecord:
    """Record of a simulated trade."""
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    direction: str              # "LONG" or "SHORT"
    position_size: float
    pnl_pct: float
    pnl_absolute: float
    holding_days: int
    hit_stop_loss: bool = False
    hit_take_profit: bool = False


@dataclass
class PerformanceMetrics:
    """Comprehensive backtest performance metrics."""
    # Returns
    total_return_pct: float
    annualized_return_pct: float
    avg_trade_return_pct: float
    
    # Risk-adjusted
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float         # Annual return / max drawdown
    
    # Drawdown
    max_drawdown_pct: float
    avg_drawdown_pct: float
    max_drawdown_duration_days: int
    
    # Win/Loss
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float        # Gross profit / gross loss
    
    # Risk
    volatility_annual: float
    var_95: float               # Value at Risk (95%)
    expected_shortfall: float   # Average loss beyond VaR
    
    # Stability
    stability_score: float      # 0-1, consistency across periods
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return_pct": self.total_return_pct,
            "annualized_return_pct": self.annualized_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "total_trades": self.total_trades,
            "profit_factor": self.profit_factor,
            "stability_score": self.stability_score,
        }


@dataclass
class BacktestResult:
    """Complete backtest result."""
    symbol: str
    start_date: str
    end_date: str
    metrics: PerformanceMetrics
    signal_quality: SignalQuality
    quality_score: float        # 0-1 composite score
    trades: List[TradeRecord] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_overfit: bool = False
    overfit_reason: Optional[str] = None


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    # Simulation
    initial_capital: float = 100000.0
    commission_pct: float = 0.001      # 0.1% per trade
    slippage_pct: float = 0.0005       # 0.05% slippage
    
    # Walk-forward
    train_window_days: int = 252       # 1 year training
    test_window_days: int = 63         # 3 months test
    min_trades_for_validity: int = 10  # Minimum trades to be statistically valid
    
    # Thresholds
    min_sharpe_threshold: float = 0.5
    max_drawdown_threshold: float = 0.25
    min_win_rate_threshold: float = 0.40
    
    # Overfit detection
    max_train_test_sharpe_diff: float = 1.0  # If train Sharpe - test Sharpe > this, overfit
    
    # Risk-free rate (annual, for Sharpe)
    risk_free_rate: float = 0.05       # 5% (India)


class BacktestAgent(Agent):
    """Validates trading signals through historical backtesting.

    INPUT DATA REQUIRED:
    --------------------
    - `symbol`: str - Stock symbol to backtest
    - `historical_signals`: List[Dict] with:
        - date: str (ISO format)
        - signal: str ("BUY" | "SELL" | "HOLD")
        - confidence: float
        - price: float
    - `price_history`: List[Dict] with {date, open, high, low, close, volume}
    - Optional:
        - `stop_loss_pct`: float - Default stop loss
        - `take_profit_pct`: float - Default take profit
        - `holding_period_days`: int - Max days to hold (default: 5)

    OUTPUT:
    -------
    - `backtest_result`: BacktestResult with:
        - Performance metrics (Sharpe, Sortino, max DD, win rate)
        - Signal quality classification
        - Overfitting detection
        - Trade-by-trade records
    """

    name = "backtest_agent"

    def __init__(self, config: Optional[BacktestConfig] = None):
        super().__init__(name=self.name)
        self._config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Utility Methods
    # ------------------------------------------------------------------

    def _parse_date(self, date_str: str) -> datetime:
        """Parse ISO date string."""
        return datetime.fromisoformat(date_str.replace("Z", "+00:00").split("T")[0])

    def _compute_returns(self, pnls: List[float]) -> List[float]:
        """Convert PnL percentages to daily returns."""
        return pnls  # Already percentages

    def _compute_sharpe(
        self, 
        returns: List[float], 
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """Compute annualized Sharpe ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        excess_return = mean_return - (risk_free_rate / periods_per_year)
        
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std_dev = math.sqrt(variance) if variance > 0 else 0.0001
        
        if std_dev == 0:
            return 0.0
        
        return (excess_return / std_dev) * math.sqrt(periods_per_year)

    def _compute_sortino(
        self, 
        returns: List[float], 
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252,
    ) -> float:
        """Compute Sortino ratio (downside deviation only)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns)
        excess_return = mean_return - (risk_free_rate / periods_per_year)
        
        downside_returns = [r for r in returns if r < 0]
        if not downside_returns:
            return 3.0  # Cap if no downside
        
        downside_variance = sum(r ** 2 for r in downside_returns) / len(downside_returns)
        downside_std = math.sqrt(downside_variance) if downside_variance > 0 else 0.0001
        
        return (excess_return / downside_std) * math.sqrt(periods_per_year)

    def _compute_max_drawdown(self, equity_curve: List[float]) -> Tuple[float, int]:
        """Compute maximum drawdown and its duration.
        
        Returns (max_dd_pct, max_dd_duration_days).
        """
        if not equity_curve:
            return 0.0, 0
        
        peak = equity_curve[0]
        max_dd = 0.0
        current_dd_start = 0
        max_dd_duration = 0
        
        for i, value in enumerate(equity_curve):
            if value > peak:
                peak = value
                current_dd_start = i
            
            dd = (peak - value) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_duration = i - current_dd_start
        
        return max_dd, max_dd_duration

    def _compute_var(self, returns: List[float], confidence: float = 0.95) -> float:
        """Compute Value at Risk at given confidence level."""
        if not returns:
            return 0.0
        
        sorted_returns = sorted(returns)
        index = int((1 - confidence) * len(sorted_returns))
        return abs(sorted_returns[index]) if index < len(sorted_returns) else 0.0

    # ------------------------------------------------------------------
    # Trade Simulation
    # ------------------------------------------------------------------

    def _simulate_trades(
        self,
        signals: List[Dict],
        prices: Dict[str, Dict],  # {date: {open, high, low, close}}
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        holding_period: int = 5,
    ) -> List[TradeRecord]:
        """Simulate trades based on signals.
        
        Returns list of trade records.
        """
        cfg = self._config
        trades = []
        
        # Sort signals by date
        signals = sorted(signals, key=lambda x: x.get("date", ""))
        
        current_position: Optional[Dict] = None
        
        for signal in signals:
            date = signal.get("date", "")
            sig = signal.get("signal", "HOLD")
            confidence = signal.get("confidence", 0.5)
            
            if date not in prices:
                continue
            
            price_data = prices[date]
            current_price = price_data.get("close", 0)
            
            if current_price <= 0:
                continue
            
            # Check exit conditions for open position
            if current_position:
                days_held = (self._parse_date(date) - self._parse_date(current_position["entry_date"])).days
                entry_price = current_position["entry_price"]
                
                # Calculate current PnL
                if current_position["direction"] == "LONG":
                    pnl_pct = (current_price - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - current_price) / entry_price
                
                # Check stop/target/holding period
                hit_stop = pnl_pct <= -stop_loss_pct
                hit_target = pnl_pct >= take_profit_pct
                expired = days_held >= holding_period
                
                if hit_stop or hit_target or expired:
                    # Close position
                    trades.append(TradeRecord(
                        symbol=signal.get("symbol", "UNKNOWN"),
                        entry_date=current_position["entry_date"],
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=current_price,
                        direction=current_position["direction"],
                        position_size=current_position["size"],
                        pnl_pct=round(pnl_pct - cfg.commission_pct * 2 - cfg.slippage_pct * 2, 6),
                        pnl_absolute=round(pnl_pct * current_position["size"] * cfg.initial_capital, 2),
                        holding_days=days_held,
                        hit_stop_loss=hit_stop,
                        hit_take_profit=hit_target,
                    ))
                    current_position = None
            
            # Open new position on BUY/SELL signals (if not already in position)
            if current_position is None:
                if sig == "BUY" and confidence > 0.5:
                    current_position = {
                        "entry_date": date,
                        "entry_price": current_price * (1 + cfg.slippage_pct),
                        "direction": "LONG",
                        "size": confidence,  # Scale by confidence
                    }
                elif sig == "SELL" and confidence > 0.5:
                    current_position = {
                        "entry_date": date,
                        "entry_price": current_price * (1 - cfg.slippage_pct),
                        "direction": "SHORT",
                        "size": confidence,
                    }
        
        return trades

    # ------------------------------------------------------------------
    # Metrics Computation
    # ------------------------------------------------------------------

    def _compute_metrics(self, trades: List[TradeRecord]) -> PerformanceMetrics:
        """Compute comprehensive performance metrics from trade records."""
        cfg = self._config
        
        if not trades:
            return PerformanceMetrics(
                total_return_pct=0,
                annualized_return_pct=0,
                avg_trade_return_pct=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                calmar_ratio=0,
                max_drawdown_pct=0,
                avg_drawdown_pct=0,
                max_drawdown_duration_days=0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                profit_factor=0,
                volatility_annual=0,
                var_95=0,
                expected_shortfall=0,
                stability_score=0,
            )
        
        # PnL list
        pnls = [t.pnl_pct for t in trades]
        
        # Basic stats
        total_return = sum(pnls)
        avg_return = total_return / len(pnls)
        
        # Time-based annualization
        if trades:
            first_date = self._parse_date(trades[0].entry_date)
            last_date = self._parse_date(trades[-1].exit_date)
            days = max(1, (last_date - first_date).days)
            years = days / 365.0
            annualized_return = ((1 + total_return) ** (1 / years) - 1) if years > 0 else 0
        else:
            annualized_return = 0
        
        # Sharpe / Sortino
        sharpe = self._compute_sharpe(pnls, cfg.risk_free_rate)
        sortino = self._compute_sortino(pnls, cfg.risk_free_rate)
        
        # Equity curve for drawdown
        equity = [cfg.initial_capital]
        for pnl in pnls:
            equity.append(equity[-1] * (1 + pnl))
        
        max_dd, max_dd_duration = self._compute_max_drawdown(equity)
        
        # Calmar
        calmar = annualized_return / max_dd if max_dd > 0 else 0
        
        # Win/Loss
        winning = [t for t in trades if t.pnl_pct > 0]
        losing = [t for t in trades if t.pnl_pct <= 0]
        win_rate = len(winning) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl_pct for t in winning) if winning else 0
        gross_loss = abs(sum(t.pnl_pct for t in losing)) if losing else 0.0001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Volatility
        if len(pnls) > 1:
            mean = sum(pnls) / len(pnls)
            variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
            vol = math.sqrt(variance) * math.sqrt(252)  # Annualized
        else:
            vol = 0
        
        # VaR and Expected Shortfall
        var_95 = self._compute_var(pnls, 0.95)
        worst_5pct = sorted(pnls)[:max(1, len(pnls) // 20)]
        expected_shortfall = abs(sum(worst_5pct) / len(worst_5pct)) if worst_5pct else 0
        
        # Stability: consistency across quarters
        if len(trades) >= 4:
            quarter_size = len(trades) // 4
            quarter_sharpes = []
            for i in range(4):
                q_pnls = pnls[i * quarter_size:(i + 1) * quarter_size]
                if q_pnls:
                    quarter_sharpes.append(self._compute_sharpe(q_pnls, cfg.risk_free_rate))
            
            if quarter_sharpes:
                mean_sharpe = sum(quarter_sharpes) / len(quarter_sharpes)
                sharpe_variance = sum((s - mean_sharpe) ** 2 for s in quarter_sharpes) / len(quarter_sharpes)
                stability = 1.0 / (1.0 + math.sqrt(sharpe_variance))  # Lower variance = higher stability
            else:
                stability = 0.5
        else:
            stability = 0.5  # Not enough data
        
        return PerformanceMetrics(
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(annualized_return, 4),
            avg_trade_return_pct=round(avg_return, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            calmar_ratio=round(calmar, 4),
            max_drawdown_pct=round(max_dd, 4),
            avg_drawdown_pct=round(max_dd / 2, 4),  # Approximation
            max_drawdown_duration_days=max_dd_duration,
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            win_rate=round(win_rate, 4),
            profit_factor=round(profit_factor, 4),
            volatility_annual=round(vol, 4),
            var_95=round(var_95, 4),
            expected_shortfall=round(expected_shortfall, 4),
            stability_score=round(stability, 4),
        )

    # ------------------------------------------------------------------
    # Quality Assessment
    # ------------------------------------------------------------------

    def _assess_signal_quality(self, metrics: PerformanceMetrics) -> Tuple[SignalQuality, float]:
        """Determine signal quality from metrics.
        
        Returns (quality_enum, quality_score).
        """
        cfg = self._config
        
        # Compute composite score (0-1)
        score = 0.0
        
        # Sharpe contribution (40%)
        sharpe_score = min(1.0, max(0, metrics.sharpe_ratio) / 2.0)
        score += sharpe_score * 0.4
        
        # Win rate contribution (20%)
        win_score = min(1.0, metrics.win_rate / 0.6)  # 60% win rate = max
        score += win_score * 0.2
        
        # Drawdown contribution (20%) - inverse
        dd_score = max(0, 1.0 - metrics.max_drawdown_pct / cfg.max_drawdown_threshold)
        score += dd_score * 0.2
        
        # Stability contribution (20%)
        score += metrics.stability_score * 0.2
        
        # Classify
        if score >= 0.8 and metrics.sharpe_ratio >= 2.0:
            quality = SignalQuality.EXCELLENT
        elif score >= 0.6 and metrics.sharpe_ratio >= 1.0:
            quality = SignalQuality.GOOD
        elif score >= 0.4 and metrics.sharpe_ratio >= 0.5:
            quality = SignalQuality.FAIR
        elif score >= 0.2:
            quality = SignalQuality.POOR
        else:
            quality = SignalQuality.UNRELIABLE
        
        return quality, round(score, 4)

    def _detect_overfitting(
        self, 
        train_metrics: PerformanceMetrics, 
        test_metrics: PerformanceMetrics,
    ) -> Tuple[bool, Optional[str]]:
        """Detect if signal is overfit to training data."""
        cfg = self._config
        
        sharpe_diff = train_metrics.sharpe_ratio - test_metrics.sharpe_ratio
        
        if sharpe_diff > cfg.max_train_test_sharpe_diff:
            return True, f"Train Sharpe ({train_metrics.sharpe_ratio:.2f}) >> Test Sharpe ({test_metrics.sharpe_ratio:.2f})"
        
        # Win rate degradation
        if train_metrics.win_rate > 0.6 and test_metrics.win_rate < 0.4:
            return True, f"Win rate degraded from {train_metrics.win_rate:.1%} to {test_metrics.win_rate:.1%}"
        
        # Test period has negative return while train is positive
        if train_metrics.total_return_pct > 0.1 and test_metrics.total_return_pct < -0.05:
            return True, "Positive train return but negative test return"
        
        return False, None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def backtest(
        self,
        symbol: str,
        historical_signals: List[Dict],
        price_history: List[Dict],
        stop_loss_pct: float = 0.03,
        take_profit_pct: float = 0.06,
        holding_period_days: int = 5,
    ) -> BacktestResult:
        """Run backtest on historical signals.
        
        Returns BacktestResult with metrics and quality assessment.
        """
        cfg = self._config
        warnings: List[str] = []
        
        if not historical_signals:
            return BacktestResult(
                symbol=symbol,
                start_date="",
                end_date="",
                metrics=self._compute_metrics([]),
                signal_quality=SignalQuality.UNRELIABLE,
                quality_score=0.0,
                warnings=["No historical signals provided"],
            )
        
        # Build price lookup
        prices: Dict[str, Dict] = {}
        for p in price_history:
            date = p.get("date", "")
            if date:
                prices[date] = p
        
        # Simulate trades
        trades = self._simulate_trades(
            signals=historical_signals,
            prices=prices,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            holding_period=holding_period_days,
        )
        
        if len(trades) < cfg.min_trades_for_validity:
            warnings.append(f"Only {len(trades)} trades, below minimum {cfg.min_trades_for_validity} for validity")
        
        # Compute metrics
        metrics = self._compute_metrics(trades)
        
        # Assess quality
        quality, quality_score = self._assess_signal_quality(metrics)
        
        # Check thresholds
        if metrics.sharpe_ratio < cfg.min_sharpe_threshold:
            warnings.append(f"Sharpe ratio {metrics.sharpe_ratio:.2f} below threshold {cfg.min_sharpe_threshold}")
        
        if metrics.max_drawdown_pct > cfg.max_drawdown_threshold:
            warnings.append(f"Max drawdown {metrics.max_drawdown_pct:.1%} exceeds limit {cfg.max_drawdown_threshold:.1%}")
        
        if metrics.win_rate < cfg.min_win_rate_threshold:
            warnings.append(f"Win rate {metrics.win_rate:.1%} below threshold {cfg.min_win_rate_threshold:.1%}")
        
        # Get date range
        start_date = historical_signals[0].get("date", "") if historical_signals else ""
        end_date = historical_signals[-1].get("date", "") if historical_signals else ""
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            metrics=metrics,
            signal_quality=quality,
            quality_score=quality_score,
            trades=trades,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute backtest.
        
        Expected kwargs:
        - symbol: str
        - historical_signals: List[Dict]
        - price_history: List[Dict]
        - stop_loss_pct: float (optional)
        - take_profit_pct: float (optional)
        - holding_period_days: int (optional)
        """
        started = self._pre_run()
        
        try:
            symbol = kwargs.get("symbol", "UNKNOWN")
            historical_signals = kwargs.get("historical_signals", [])
            price_history = kwargs.get("price_history", [])
            stop_loss_pct = kwargs.get("stop_loss_pct", 0.03)
            take_profit_pct = kwargs.get("take_profit_pct", 0.06)
            holding_period_days = kwargs.get("holding_period_days", 5)
            
            result = self.backtest(
                symbol=symbol,
                historical_signals=historical_signals,
                price_history=price_history,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                holding_period_days=holding_period_days,
            )
            
            payload = {
                "symbol": result.symbol,
                "signal_quality": result.signal_quality.value,
                "quality_score": result.quality_score,
                "metrics": result.metrics.to_dict(),
                "total_trades": len(result.trades),
                "warnings": result.warnings,
                "is_overfit": result.is_overfit,
                "raw_result": result,
            }
            
            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] BacktestAgent | {symbol}")
                print(f"  Signal Quality: {result.signal_quality.value}")
                print(f"  Quality Score: {result.quality_score:.1%}")
                print(f"  Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
                print(f"  Sortino Ratio: {result.metrics.sortino_ratio:.2f}")
                print(f"  Total Return: {result.metrics.total_return_pct:.1%}")
                print(f"  Max Drawdown: {result.metrics.max_drawdown_pct:.1%}")
                print(f"  Win Rate: {result.metrics.win_rate:.1%}")
                print(f"  Total Trades: {result.metrics.total_trades}")
                if result.warnings:
                    print(f"  Warnings: {', '.join(result.warnings[:3])}")
            
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[
                    symbol,
                    result.signal_quality.value,
                    str(result.quality_score),
                ],
            )
            
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="BACKTEST_ERROR", message=str(exc))],
            )
