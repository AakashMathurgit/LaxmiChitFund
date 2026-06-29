"""Trade Planner Agent — generates complete trade plans with entry/exit levels.

Converts a JudgeDecision into an actionable TradePlan with:
- Entry price (market or limit)
- Stop-loss based on ATR or support levels
- Target price based on risk-reward ratio
- Position sizing based on risk tolerance
- Expected holding period

This is what professional trading systems use.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    JudgeDecision,
    TradePlan,
    EntryType,
    RegimeSignal,
    MarketRegime,
    VolatilityState,
    TechnicalSignal,
)

# Debug flag
DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


@dataclass
class TradePlannerConfig:
    """Configuration for trade planning."""
    # Risk-reward settings
    min_risk_reward_ratio: float = 1.5     # Minimum acceptable R:R
    default_risk_reward: float = 2.0       # Default target R:R
    max_risk_reward: float = 4.0           # Cap unrealistic targets
    
    # Stop-loss settings
    atr_stop_multiplier: float = 1.5       # Stop = entry - (ATR * multiplier)
    min_stop_pct: float = 0.02             # Minimum 2% stop
    max_stop_pct: float = 0.08             # Maximum 8% stop
    
    # Position sizing
    max_risk_per_trade_pct: float = 0.01   # Risk 1% of portfolio per trade
    max_position_size_pct: float = 0.05    # Max 5% in single position
    default_portfolio_value: float = 1000000.0  # ₹10 lakh default
    
    # Entry settings
    limit_offset_pct: float = 0.005        # 0.5% below current for limit orders
    breakout_buffer_pct: float = 0.003     # 0.3% above resistance for breakout
    
    # Timing
    default_holding_days: int = 5          # Swing trading default
    limit_order_validity_days: int = 2     # Limit order expires after
    
    # Trailing stop
    enable_trailing_stop: bool = True
    trailing_stop_activation_pct: float = 0.03  # Activate after 3% profit
    trailing_stop_distance_pct: float = 0.02    # Trail by 2%


class TradePlannerAgent(Agent):
    """Generates complete trade plans from JudgeDecisions.

    INPUT DATA REQUIRED:
    --------------------
    - `decision`: JudgeDecision - The buy/sell/hold decision
    - `current_price`: float - Current market price
    - `technical`: TechnicalSignal - For support/resistance/ATR
    - `regime`: RegimeSignal - Market conditions
    - `portfolio_value`: Optional[float] - For position sizing

    OUTPUT:
    -------
    - `trade_plan`: TradePlan with:
        - Entry type (market/limit) and price
        - Stop-loss price
        - Target price
        - Risk-reward ratio
        - Position sizing
        - Expected holding period
        - Human-readable reasoning
    """

    name = "trade_planner_agent"

    def __init__(self, config: Optional[TradePlannerConfig] = None):
        super().__init__(name=self.name)
        self._config = config or TradePlannerConfig()

    # ------------------------------------------------------------------
    # Support/Resistance Calculation
    # ------------------------------------------------------------------

    def _find_support_resistance(
        self, 
        ohlc: List[Dict],
        lookback: int = 50,
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate support and resistance levels from price history.
        
        Uses pivot point detection algorithm.
        Returns (support, resistance).
        """
        if not ohlc or len(ohlc) < lookback:
            return None, None
        
        recent = ohlc[-lookback:]
        highs = [bar["high"] for bar in recent]
        lows = [bar["low"] for bar in recent]
        closes = [bar["close"] for bar in recent]
        
        current_price = closes[-1]
        
        # Find local maxima (resistance candidates)
        resistance_candidates = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                resistance_candidates.append(highs[i])
        
        # Find local minima (support candidates)
        support_candidates = []
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                support_candidates.append(lows[i])
        
        # Find nearest resistance above current price
        resistances_above = [r for r in resistance_candidates if r > current_price]
        resistance = min(resistances_above) if resistances_above else max(highs)
        
        # Find nearest support below current price
        supports_below = [s for s in support_candidates if s < current_price]
        support = max(supports_below) if supports_below else min(lows)
        
        return support, resistance

    def _compute_atr(self, ohlc: List[Dict], period: int = 14) -> float:
        """Compute Average True Range."""
        if len(ohlc) < period + 1:
            return 0.0
        
        trs = []
        for i in range(1, len(ohlc)):
            high = ohlc[i]["high"]
            low = ohlc[i]["low"]
            prev_close = ohlc[i-1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return sum(trs[-period:]) / period if trs else 0.0

    # ------------------------------------------------------------------
    # Entry Price Logic
    # ------------------------------------------------------------------

    def _determine_entry(
        self,
        decision: JudgeDecision,
        current_price: float,
        resistance: Optional[float],
        support: Optional[float],
        technical: Optional[TechnicalSignal],
        regime: RegimeSignal,
    ) -> tuple[EntryType, float, str]:
        """Determine entry type and price.
        
        Returns (entry_type, entry_price, reasoning).
        """
        cfg = self._config
        
        # Check for breakout signal
        is_breakout = False
        if technical:
            is_breakout = technical.breakout_flag
        
        # High volatility or strong momentum = market order
        high_momentum = regime.volatility_state in (VolatilityState.HIGH, VolatilityState.EXTREME)
        
        if is_breakout or high_momentum:
            # Market order for breakouts
            entry_type = EntryType.MARKET
            entry_price = current_price
            reasoning = "Market order due to breakout/momentum - immediate execution needed"
        else:
            # Limit order for pullback entries
            entry_type = EntryType.LIMIT
            
            if decision.decision == "BUY" and support:
                # Try to enter near support
                pullback_price = current_price * (1 - cfg.limit_offset_pct)
                # Don't go below support
                entry_price = max(pullback_price, support * 1.01)
                reasoning = f"Limit order near support level (₹{support:.2f})"
            else:
                # Default: small discount to current price
                entry_price = current_price * (1 - cfg.limit_offset_pct)
                reasoning = "Limit order with small pullback discount"
        
        return entry_type, round(entry_price, 2), reasoning

    # ------------------------------------------------------------------
    # Stop-Loss Logic
    # ------------------------------------------------------------------

    def _determine_stop_loss(
        self,
        entry_price: float,
        support: Optional[float],
        atr: float,
        regime: RegimeSignal,
    ) -> tuple[float, str]:
        """Determine stop-loss price.
        
        Returns (stop_price, reasoning).
        """
        cfg = self._config
        
        # ATR-based stop
        atr_stop = entry_price - (atr * cfg.atr_stop_multiplier)
        
        # Support-based stop (below support)
        support_stop = (support * 0.98) if support else atr_stop
        
        # Use the tighter of the two (less risk)
        stop_price = max(atr_stop, support_stop)
        
        # Ensure within bounds
        min_stop = entry_price * (1 - cfg.max_stop_pct)
        max_stop = entry_price * (1 - cfg.min_stop_pct)
        
        stop_price = max(min_stop, min(max_stop, stop_price))
        
        # Widen stop in high volatility
        if regime.volatility_state == VolatilityState.HIGH:
            stop_price = stop_price * 0.99  # 1% wider
        elif regime.volatility_state == VolatilityState.EXTREME:
            stop_price = stop_price * 0.98  # 2% wider
        
        stop_pct = (entry_price - stop_price) / entry_price * 100
        
        if support and abs(stop_price - support * 0.98) < atr * 0.5:
            reasoning = f"Stop below support at ₹{support:.2f} ({stop_pct:.1f}% risk)"
        else:
            reasoning = f"ATR-based stop ({cfg.atr_stop_multiplier}x ATR, {stop_pct:.1f}% risk)"
        
        return round(stop_price, 2), reasoning

    # ------------------------------------------------------------------
    # Target Price Logic
    # ------------------------------------------------------------------

    def _determine_target(
        self,
        entry_price: float,
        stop_price: float,
        resistance: Optional[float],
        decision: JudgeDecision,
    ) -> tuple[float, float, str]:
        """Determine target price based on risk-reward.
        
        Returns (target_price, risk_reward_ratio, reasoning).
        """
        cfg = self._config
        
        risk = entry_price - stop_price
        
        # Calculate target for desired R:R
        rr_target = entry_price + (risk * cfg.default_risk_reward)
        
        # Use resistance as target if it gives good R:R
        if resistance and resistance > entry_price:
            resistance_rr = (resistance - entry_price) / risk if risk > 0 else 0
            
            if resistance_rr >= cfg.min_risk_reward_ratio:
                target_price = resistance
                actual_rr = resistance_rr
                reasoning = f"Target at resistance ₹{resistance:.2f} (R:R={actual_rr:.1f})"
            else:
                # Resistance too close, use R:R target
                target_price = rr_target
                actual_rr = cfg.default_risk_reward
                reasoning = f"R:R based target (resistance too close)"
        else:
            target_price = rr_target
            actual_rr = cfg.default_risk_reward
            reasoning = f"R:R={cfg.default_risk_reward} target (no clear resistance)"
        
        # Cap maximum target
        max_target = entry_price * (1 + cfg.max_risk_reward * (risk / entry_price))
        target_price = min(target_price, max_target)
        
        # Adjust based on confidence
        if decision.confidence < 0.6:
            # Lower confidence = conservative target
            target_price = entry_price + (target_price - entry_price) * 0.8
            actual_rr = (target_price - entry_price) / risk if risk > 0 else 0
            reasoning += " (reduced for lower confidence)"
        
        return round(target_price, 2), round(actual_rr, 2), reasoning

    # ------------------------------------------------------------------
    # Position Sizing
    # ------------------------------------------------------------------

    def _calculate_position_size(
        self,
        entry_price: float,
        stop_price: float,
        portfolio_value: float,
        regime: RegimeSignal,
    ) -> tuple[float, int, float]:
        """Calculate position size based on risk.
        
        Returns (position_pct, shares, max_loss).
        """
        cfg = self._config
        
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return 0.0, 0, 0.0
        
        # Maximum amount to risk
        max_risk_amount = portfolio_value * cfg.max_risk_per_trade_pct
        
        # Reduce risk in adverse conditions
        if regime.market_regime == MarketRegime.BEAR_TREND:
            max_risk_amount *= 0.5
        elif regime.volatility_state == VolatilityState.HIGH:
            max_risk_amount *= 0.7
        elif regime.volatility_state == VolatilityState.EXTREME:
            max_risk_amount *= 0.3
        
        # Calculate shares from risk
        shares = int(max_risk_amount / risk_per_share)
        
        # Cap position size
        position_value = shares * entry_price
        max_position_value = portfolio_value * cfg.max_position_size_pct
        
        if position_value > max_position_value:
            shares = int(max_position_value / entry_price)
            position_value = shares * entry_price
        
        position_pct = position_value / portfolio_value
        max_loss = shares * risk_per_share
        
        return round(position_pct, 4), shares, round(max_loss, 2)

    # ------------------------------------------------------------------
    # Main Planning Logic
    # ------------------------------------------------------------------

    def create_trade_plan(
        self,
        decision: JudgeDecision,
        current_price: float,
        ohlc: Optional[List[Dict]] = None,
        technical: Optional[TechnicalSignal] = None,
        regime: Optional[RegimeSignal] = None,
        portfolio_value: Optional[float] = None,
    ) -> TradePlan:
        """Create a complete trade plan from a decision.
        
        Returns TradePlan with all execution details.
        """
        cfg = self._config
        
        regime = regime or RegimeSignal()
        portfolio_value = portfolio_value or cfg.default_portfolio_value
        
        # Calculate support/resistance and ATR
        support, resistance = None, None
        atr = current_price * 0.02  # Default 2% of price
        
        if ohlc:
            support, resistance = self._find_support_resistance(ohlc)
            computed_atr = self._compute_atr(ohlc)
            if computed_atr > 0:
                atr = computed_atr
        
        # Use technical signal levels if available
        if technical:
            if technical.support_level:
                support = technical.support_level
            if technical.resistance_level:
                resistance = technical.resistance_level
        
        # For HOLD/SELL decisions, return minimal plan
        if decision.decision != "BUY":
            return TradePlan(
                symbol=decision.symbol,
                date=decision.date,
                decision=decision.decision,
                confidence=decision.confidence,
                entry_type=EntryType.MARKET,
                entry_price=current_price,
                current_price=current_price,
                stop_loss_price=0,
                target_price=0,
                reasoning=f"{decision.decision} - No trade plan needed",
            )
        
        # Determine entry
        entry_type, entry_price, entry_reasoning = self._determine_entry(
            decision, current_price, resistance, support, technical, regime
        )
        
        # Determine stop-loss
        stop_price, stop_reasoning = self._determine_stop_loss(
            entry_price, support, atr, regime
        )
        
        # Determine target
        target_price, risk_reward, target_reasoning = self._determine_target(
            entry_price, stop_price, resistance, decision
        )
        
        # Calculate position size
        position_pct, shares, max_loss = self._calculate_position_size(
            entry_price, stop_price, portfolio_value, regime
        )
        
        # Calculate risk/reward per share
        risk_per_share = entry_price - stop_price
        reward_per_share = target_price - entry_price
        
        # Build reasoning
        reasoning_parts = []
        if decision.confidence >= 0.7:
            reasoning_parts.append(f"High confidence ({decision.confidence:.0%}) trade")
        else:
            reasoning_parts.append(f"Moderate confidence ({decision.confidence:.0%}) trade")
        
        reasoning_parts.append(entry_reasoning.split(" - ")[0])
        
        if regime.market_regime == MarketRegime.BULL_TREND:
            reasoning_parts.append("favorable market conditions")
        elif regime.market_regime == MarketRegime.BEAR_TREND:
            reasoning_parts.append("cautious in bearish market")
        
        # Trailing stop config
        trailing_stop = None
        if cfg.enable_trailing_stop and risk_reward >= 2.0:
            trailing_stop = cfg.trailing_stop_distance_pct
        
        return TradePlan(
            symbol=decision.symbol,
            date=decision.date,
            decision=decision.decision,
            confidence=decision.confidence,
            entry_type=entry_type,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss_price=stop_price,
            target_price=target_price,
            trailing_stop_pct=trailing_stop,
            risk_reward_ratio=risk_reward,
            risk_per_share=round(risk_per_share, 2),
            reward_per_share=round(reward_per_share, 2),
            position_size_pct=position_pct,
            suggested_shares=shares,
            max_loss_amount=max_loss,
            expected_holding_days=cfg.default_holding_days,
            entry_valid_until=(datetime.now() + timedelta(days=cfg.limit_order_validity_days)).strftime("%Y-%m-%d") if entry_type == EntryType.LIMIT else None,
            support_level=support,
            resistance_level=resistance,
            atr=round(atr, 2),
            reasoning="; ".join(reasoning_parts),
            entry_reasoning=entry_reasoning,
            exit_reasoning=f"Stop: {stop_reasoning}. Target: {target_reasoning}",
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute trade planning.
        
        Expected kwargs:
        - decision: JudgeDecision
        - current_price: float
        - ohlc: Optional[List[Dict]]
        - technical: Optional[TechnicalSignal]
        - regime: Optional[RegimeSignal]
        - portfolio_value: Optional[float]
        """
        started = self._pre_run()
        
        try:
            decision = kwargs.get("decision")
            if not decision:
                raise ValueError("TradePlannerAgent requires 'decision' (JudgeDecision)")
            
            current_price = kwargs.get("current_price", 0)
            if current_price <= 0:
                raise ValueError("TradePlannerAgent requires positive 'current_price'")
            
            ohlc = kwargs.get("ohlc")
            technical = kwargs.get("technical")
            regime = kwargs.get("regime")
            portfolio_value = kwargs.get("portfolio_value")
            
            plan = self.create_trade_plan(
                decision=decision,
                current_price=current_price,
                ohlc=ohlc,
                technical=technical,
                regime=regime,
                portfolio_value=portfolio_value,
            )
            
            payload = {
                "trade_plan": plan.to_dict(),
                "raw_plan": plan,
                "summary": plan.summary(),
            }
            
            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] TradePlannerAgent | {plan.symbol}")
                print(f"  Decision: {plan.decision}")
                print(f"  Entry: {plan.entry_type.value.upper()} @ ₹{plan.entry_price:.2f}")
                print(f"  Stop Loss: ₹{plan.stop_loss_price:.2f} ({(plan.entry_price - plan.stop_loss_price) / plan.entry_price * 100:.1f}%)")
                print(f"  Target: ₹{plan.target_price:.2f} ({(plan.target_price - plan.entry_price) / plan.entry_price * 100:.1f}%)")
                print(f"  Risk:Reward = 1:{plan.risk_reward_ratio:.1f}")
                print(f"  Position: {plan.position_size_pct:.1%} ({plan.suggested_shares} shares)")
                print(f"  Max Loss: ₹{plan.max_loss_amount:.2f}")
                print(f"  Hold: {plan.expected_holding_days} days")
                if plan.support_level:
                    print(f"  Support: ₹{plan.support_level:.2f}")
                if plan.resistance_level:
                    print(f"  Resistance: ₹{plan.resistance_level:.2f}")
            
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[
                    plan.symbol,
                    plan.decision,
                    str(plan.entry_price),
                    str(plan.stop_loss_price),
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
                errors=[AgentError(code="TRADE_PLAN_ERROR", message=str(exc))],
            )
