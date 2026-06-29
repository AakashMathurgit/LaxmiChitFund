"""Risk Manager Agent — portfolio-level risk controls and position sizing.

Converts trading signals into actionable positions with:
- Position sizing based on volatility and confidence
- Portfolio-level constraints (max exposure, correlation limits)
- Drawdown guardrails and "no-trade" conditions
- Stop-loss and take-profit calculations

This agent ensures the system doesn't become a reckless recommender.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    JudgeDecision, 
    MarketRegime, 
    VolatilityState,
    RegimeSignal,
)

# Debug flag
DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


class RiskLevel(str, Enum):
    """Risk level classifications."""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


@dataclass
class PositionRisk:
    """Risk assessment for a single position."""
    symbol: str
    decision: str                    # BUY/SELL/HOLD
    raw_position_size: float         # From JudgeAgent
    adjusted_position_size: float    # After risk adjustments
    stop_loss_pct: float
    take_profit_pct: float
    risk_level: RiskLevel
    risk_score: float               # 0-1, higher = more risky
    warnings: List[str] = field(default_factory=list)
    blocked: bool = False           # True = no-trade condition triggered
    block_reason: Optional[str] = None


@dataclass
class PortfolioRiskAssessment:
    """Portfolio-level risk assessment."""
    total_exposure_pct: float       # Sum of all position sizes
    max_single_position_pct: float  # Largest position
    correlation_risk: float         # 0-1, higher = more correlated
    drawdown_risk: float            # Current drawdown risk level
    regime_risk_multiplier: float   # Adjustment based on market regime
    overall_risk_level: RiskLevel
    positions: List[PositionRisk] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RiskManagerConfig:
    """Configuration for risk management rules."""
    # Position limits
    max_single_position_pct: float = 0.05      # 5% max per position
    max_total_exposure_pct: float = 0.60       # 60% max total exposure
    min_position_size_pct: float = 0.005       # 0.5% minimum (ignore smaller)
    
    # Volatility adjustments
    vol_scale_low: float = 1.2                 # Scale up in low vol
    vol_scale_moderate: float = 1.0            # No adjustment
    vol_scale_high: float = 0.6                # Scale down in high vol
    vol_scale_extreme: float = 0.3             # Major reduction in extreme vol
    
    # Regime adjustments
    regime_scale_bull: float = 1.1
    regime_scale_sideways: float = 1.0
    regime_scale_bear: float = 0.7             # Reduce in bear markets
    regime_scale_high_vol: float = 0.5
    
    # Stop-loss / Take-profit (ATR multipliers)
    atr_stop_multiplier: float = 2.0
    atr_profit_multiplier: float = 3.0
    min_stop_loss_pct: float = 0.02            # 2% minimum stop
    max_stop_loss_pct: float = 0.08            # 8% maximum stop
    
    # Drawdown controls
    max_portfolio_drawdown_pct: float = 0.15   # 15% max drawdown
    drawdown_scale_factor: float = 0.5         # Reduce by 50% if near max DD
    
    # Correlation limits (same sector)
    max_sector_exposure_pct: float = 0.25      # 25% max in one sector
    
    # No-trade conditions
    min_confidence_threshold: float = 0.45     # Below this = no trade
    min_expected_return: float = 0.005         # 0.5% minimum expected return


class RiskManagerAgent(Agent):
    """Portfolio-level risk management and position sizing.

    INPUT DATA REQUIRED:
    --------------------
    - `decisions`: List[JudgeDecision] - Trading decisions from JudgeAgent
    - `regime`: RegimeSignal - Current market regime
    - `portfolio_value`: float - Current portfolio value
    - `current_positions`: Optional[Dict[str, float]] - Existing positions {symbol: value}
    - `current_drawdown_pct`: Optional[float] - Current portfolio drawdown
    - `sector_map`: Optional[Dict[str, str]] - {symbol: sector} for correlation check
    - `atr_values`: Optional[Dict[str, float]] - {symbol: ATR} for stop calculation

    OUTPUT:
    -------
    - `portfolio_assessment`: PortfolioRiskAssessment with:
        - Overall risk level
        - Adjusted positions with risk controls
        - Warnings and blocked trades
    - `adjusted_decisions`: List of decisions with adjusted position sizes
    """

    name = "risk_manager_agent"

    def __init__(self, config: Optional[RiskManagerConfig] = None):
        super().__init__(name=self.name)
        self._config = config or RiskManagerConfig()

    # ------------------------------------------------------------------
    # Risk Scaling
    # ------------------------------------------------------------------

    def _get_volatility_scale(self, vol_state: VolatilityState) -> float:
        """Get position scale factor based on volatility."""
        cfg = self._config
        return {
            VolatilityState.LOW: cfg.vol_scale_low,
            VolatilityState.MODERATE: cfg.vol_scale_moderate,
            VolatilityState.HIGH: cfg.vol_scale_high,
            VolatilityState.EXTREME: cfg.vol_scale_extreme,
        }.get(vol_state, cfg.vol_scale_moderate)

    def _get_regime_scale(self, regime: MarketRegime) -> float:
        """Get position scale factor based on market regime."""
        cfg = self._config
        return {
            MarketRegime.BULL_TREND: cfg.regime_scale_bull,
            MarketRegime.SIDEWAYS: cfg.regime_scale_sideways,
            MarketRegime.BEAR_TREND: cfg.regime_scale_bear,
            MarketRegime.HIGH_VOLATILITY: cfg.regime_scale_high_vol,
        }.get(regime, cfg.regime_scale_sideways)

    def _get_drawdown_scale(self, current_dd: float) -> float:
        """Reduce position sizes as drawdown increases."""
        cfg = self._config
        if current_dd <= 0:
            return 1.0
        
        # Linear reduction as we approach max drawdown
        dd_ratio = current_dd / cfg.max_portfolio_drawdown_pct
        if dd_ratio >= 1.0:
            return 0.0  # Stop trading if at max drawdown
        elif dd_ratio >= 0.5:
            return 1.0 - (dd_ratio * cfg.drawdown_scale_factor)
        return 1.0

    # ------------------------------------------------------------------
    # Stop-Loss / Take-Profit Calculation
    # ------------------------------------------------------------------

    def _calculate_stop_loss(
        self, 
        atr: Optional[float], 
        price: float,
        vol_state: VolatilityState,
    ) -> float:
        """Calculate stop-loss percentage based on ATR or defaults."""
        cfg = self._config
        
        if atr and price > 0:
            # ATR-based stop
            stop_pct = (atr * cfg.atr_stop_multiplier) / price
        else:
            # Default based on volatility
            defaults = {
                VolatilityState.LOW: 0.025,
                VolatilityState.MODERATE: 0.03,
                VolatilityState.HIGH: 0.05,
                VolatilityState.EXTREME: 0.06,
            }
            stop_pct = defaults.get(vol_state, 0.03)
        
        # Clamp to bounds
        return max(cfg.min_stop_loss_pct, min(cfg.max_stop_loss_pct, stop_pct))

    def _calculate_take_profit(
        self, 
        atr: Optional[float], 
        price: float,
        stop_loss_pct: float,
    ) -> float:
        """Calculate take-profit, typically 1.5-2x the stop-loss (reward/risk)."""
        cfg = self._config
        
        if atr and price > 0:
            profit_pct = (atr * cfg.atr_profit_multiplier) / price
        else:
            # Default: 2x stop loss
            profit_pct = stop_loss_pct * 2.0
        
        return max(stop_loss_pct * 1.5, profit_pct)  # At least 1.5:1 R/R

    # ------------------------------------------------------------------
    # Position Risk Assessment
    # ------------------------------------------------------------------

    def _assess_position_risk(
        self,
        decision: JudgeDecision,
        regime: RegimeSignal,
        current_drawdown: float,
        atr: Optional[float],
    ) -> PositionRisk:
        """Assess and adjust risk for a single position."""
        cfg = self._config
        warnings: List[str] = []
        blocked = False
        block_reason = None
        
        # Get scaling factors
        vol_scale = self._get_volatility_scale(regime.volatility_state)
        regime_scale = self._get_regime_scale(regime.market_regime)
        dd_scale = self._get_drawdown_scale(current_drawdown)
        
        # Raw position from judge
        raw_size = decision.position_size_pct
        
        # Apply all adjustments
        total_scale = vol_scale * regime_scale * dd_scale
        adjusted_size = raw_size * total_scale
        
        # Cap at maximum
        if adjusted_size > cfg.max_single_position_pct:
            warnings.append(f"Position capped from {adjusted_size:.1%} to {cfg.max_single_position_pct:.1%}")
            adjusted_size = cfg.max_single_position_pct
        
        # Minimum threshold
        if adjusted_size < cfg.min_position_size_pct and decision.decision == "BUY":
            adjusted_size = 0.0
            warnings.append("Position too small, skipped")
        
        # Calculate stops
        price = getattr(decision, 'price', 0) or 100  # Fallback
        stop_loss = self._calculate_stop_loss(atr, price, regime.volatility_state)
        take_profit = self._calculate_take_profit(atr, price, stop_loss)
        
        # Check no-trade conditions
        if decision.confidence < cfg.min_confidence_threshold:
            blocked = True
            block_reason = f"Confidence {decision.confidence:.1%} below threshold {cfg.min_confidence_threshold:.1%}"
            adjusted_size = 0.0
        
        if decision.expected_return_5d < cfg.min_expected_return:
            if decision.decision == "BUY":
                blocked = True
                block_reason = f"Expected return {decision.expected_return_5d:.2%} below minimum {cfg.min_expected_return:.2%}"
                adjusted_size = 0.0
        
        # Bear market + BUY = extra caution
        if regime.market_regime == MarketRegime.BEAR_TREND and decision.decision == "BUY":
            warnings.append("BUY signal in BEAR market - reduced confidence")
            adjusted_size *= 0.5
        
        # Extreme volatility = halt new positions
        if regime.volatility_state == VolatilityState.EXTREME and decision.decision == "BUY":
            blocked = True
            block_reason = "New positions blocked during EXTREME volatility"
            adjusted_size = 0.0
        
        # Drawdown guardrail
        if current_drawdown >= cfg.max_portfolio_drawdown_pct:
            blocked = True
            block_reason = f"Portfolio at max drawdown ({current_drawdown:.1%})"
            adjusted_size = 0.0
        
        # Calculate risk score (0-1, higher = riskier)
        risk_score = (
            (1 - decision.confidence) * 0.3 +
            (decision.downside_risk_prob) * 0.3 +
            (vol_scale < 1.0) * (1 - vol_scale) * 0.2 +
            (regime_scale < 1.0) * (1 - regime_scale) * 0.2
        )
        
        # Determine risk level
        if risk_score < 0.2:
            risk_level = RiskLevel.VERY_LOW
        elif risk_score < 0.4:
            risk_level = RiskLevel.LOW
        elif risk_score < 0.6:
            risk_level = RiskLevel.MODERATE
        elif risk_score < 0.8:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME
        
        return PositionRisk(
            symbol=decision.symbol,
            decision=decision.decision,
            raw_position_size=raw_size,
            adjusted_position_size=round(adjusted_size, 6),
            stop_loss_pct=round(stop_loss, 4),
            take_profit_pct=round(take_profit, 4),
            risk_level=risk_level,
            risk_score=round(risk_score, 4),
            warnings=warnings,
            blocked=blocked,
            block_reason=block_reason,
        )

    # ------------------------------------------------------------------
    # Portfolio-Level Assessment
    # ------------------------------------------------------------------

    def _assess_portfolio(
        self,
        positions: List[PositionRisk],
        regime: RegimeSignal,
        current_drawdown: float,
        sector_map: Dict[str, str],
    ) -> PortfolioRiskAssessment:
        """Assess overall portfolio risk."""
        cfg = self._config
        warnings: List[str] = []
        
        # Calculate totals
        buy_positions = [p for p in positions if p.decision == "BUY" and not p.blocked]
        total_exposure = sum(p.adjusted_position_size for p in buy_positions)
        max_single = max((p.adjusted_position_size for p in buy_positions), default=0)
        
        # Check total exposure limit
        if total_exposure > cfg.max_total_exposure_pct:
            warnings.append(f"Total exposure {total_exposure:.1%} exceeds limit {cfg.max_total_exposure_pct:.1%}")
            # Scale down proportionally
            scale = cfg.max_total_exposure_pct / total_exposure
            for p in buy_positions:
                p.adjusted_position_size *= scale
                p.warnings.append(f"Scaled down by {(1-scale):.1%} for portfolio limit")
            total_exposure = cfg.max_total_exposure_pct
        
        # Check sector concentration
        sector_exposure: Dict[str, float] = {}
        for p in buy_positions:
            sector = sector_map.get(p.symbol, "unknown")
            sector_exposure[sector] = sector_exposure.get(sector, 0) + p.adjusted_position_size
        
        for sector, exposure in sector_exposure.items():
            if exposure > cfg.max_sector_exposure_pct:
                warnings.append(f"Sector '{sector}' exposure {exposure:.1%} exceeds limit")
        
        # Correlation risk (simplified: count of same-sector positions)
        if sector_exposure:
            max_sector_exposure = max(sector_exposure.values())
            correlation_risk = max_sector_exposure / max(total_exposure, 0.01)
        else:
            correlation_risk = 0.0
        
        # Regime-based risk multiplier
        regime_risk = self._get_regime_scale(regime.market_regime)
        regime_risk_multiplier = 1.0 / max(regime_risk, 0.1)  # Inverse: higher in risky regimes
        
        # Overall risk level
        avg_risk = sum(p.risk_score for p in positions) / len(positions) if positions else 0.5
        portfolio_risk_score = (
            avg_risk * 0.4 +
            (total_exposure / cfg.max_total_exposure_pct) * 0.3 +
            correlation_risk * 0.2 +
            (current_drawdown / cfg.max_portfolio_drawdown_pct) * 0.1
        )
        
        if portfolio_risk_score < 0.2:
            overall_level = RiskLevel.VERY_LOW
        elif portfolio_risk_score < 0.4:
            overall_level = RiskLevel.LOW
        elif portfolio_risk_score < 0.6:
            overall_level = RiskLevel.MODERATE
        elif portfolio_risk_score < 0.8:
            overall_level = RiskLevel.HIGH
        else:
            overall_level = RiskLevel.EXTREME
        
        return PortfolioRiskAssessment(
            total_exposure_pct=round(total_exposure, 4),
            max_single_position_pct=round(max_single, 4),
            correlation_risk=round(correlation_risk, 4),
            drawdown_risk=round(current_drawdown, 4),
            regime_risk_multiplier=round(regime_risk_multiplier, 4),
            overall_risk_level=overall_level,
            positions=positions,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess_risk(
        self,
        decisions: List[JudgeDecision],
        regime: RegimeSignal,
        current_drawdown: float = 0.0,
        atr_values: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
    ) -> PortfolioRiskAssessment:
        """Main risk assessment method.
        
        Returns PortfolioRiskAssessment with adjusted positions.
        """
        atr_values = atr_values or {}
        sector_map = sector_map or {}
        
        # Assess each position
        positions = [
            self._assess_position_risk(
                decision=d,
                regime=regime,
                current_drawdown=current_drawdown,
                atr=atr_values.get(d.symbol),
            )
            for d in decisions
        ]
        
        # Portfolio-level assessment
        return self._assess_portfolio(positions, regime, current_drawdown, sector_map)

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute risk assessment.
        
        Expected kwargs:
        - decisions: List[JudgeDecision]
        - regime: RegimeSignal
        - current_drawdown: float (optional)
        - atr_values: Dict[str, float] (optional)
        - sector_map: Dict[str, str] (optional)
        """
        started = self._pre_run()
        
        try:
            decisions = kwargs.get("decisions", [])
            regime = kwargs.get("regime") or RegimeSignal()
            current_drawdown = kwargs.get("current_drawdown", 0.0)
            atr_values = kwargs.get("atr_values", {})
            sector_map = kwargs.get("sector_map", {})
            
            assessment = self.assess_risk(
                decisions=decisions,
                regime=regime,
                current_drawdown=current_drawdown,
                atr_values=atr_values,
                sector_map=sector_map,
            )
            
            # Build adjusted decisions list
            adjusted_decisions = []
            for pos in assessment.positions:
                adjusted_decisions.append({
                    "symbol": pos.symbol,
                    "decision": pos.decision if not pos.blocked else "NO_TRADE",
                    "position_size_pct": pos.adjusted_position_size,
                    "stop_loss_pct": pos.stop_loss_pct,
                    "take_profit_pct": pos.take_profit_pct,
                    "risk_level": pos.risk_level.value,
                    "blocked": pos.blocked,
                    "block_reason": pos.block_reason,
                    "warnings": pos.warnings,
                })
            
            payload = {
                "assessment": {
                    "total_exposure_pct": assessment.total_exposure_pct,
                    "max_single_position_pct": assessment.max_single_position_pct,
                    "correlation_risk": assessment.correlation_risk,
                    "drawdown_risk": assessment.drawdown_risk,
                    "overall_risk_level": assessment.overall_risk_level.value,
                    "warnings": assessment.warnings,
                },
                "adjusted_decisions": adjusted_decisions,
                "raw_assessment": assessment,
            }
            
            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] RiskManagerAgent")
                print(f"  Overall Risk Level: {assessment.overall_risk_level.value}")
                print(f"  Total Exposure: {assessment.total_exposure_pct:.1%}")
                print(f"  Correlation Risk: {assessment.correlation_risk:.1%}")
                print(f"  Positions Assessed: {len(assessment.positions)}")
                blocked_count = sum(1 for p in assessment.positions if p.blocked)
                print(f"  Blocked Trades: {blocked_count}")
                if assessment.warnings:
                    print(f"  Warnings: {', '.join(assessment.warnings)}")
            
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[
                    assessment.overall_risk_level.value,
                    str(assessment.total_exposure_pct),
                    str(len(assessment.positions)),
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
                errors=[AgentError(code="RISK_ERROR", message=str(exc))],
            )
