"""Position Management Agent — manages open positions with trailing stops and exits.

Monitors open trades and recommends:
- Trailing stop adjustments
- Partial profit booking
- Full exits (target/stop/time-based)
- Early exit on regime/sentiment changes

This is what position management systems use professionally.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    OpenPosition,
    PositionUpdate,
    ExitReason,
    RegimeSignal,
    MarketRegime,
    VolatilityState,
    SentimentSignal,
)

# Debug flag
DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


@dataclass
class PositionManagementConfig:
    """Configuration for position management logic."""
    # Trailing stop settings
    enable_trailing_stop: bool = True
    trailing_activation_pct: float = 0.03     # Activate after 3% profit
    trailing_distance_pct: float = 0.02       # Trail by 2%
    min_stop_move_pct: float = 0.005          # Min 0.5% move to update stop
    
    # Partial exit settings
    enable_partial_exits: bool = True
    partial_exit_trigger_pct: float = 0.04    # First partial at 4% profit
    partial_exit_ratio: float = 0.5           # Sell 50% at partial target
    
    # Time-based exits
    max_holding_days: int = 10                # Force review after 10 days
    stagnant_days_threshold: int = 5          # If flat for 5 days, consider exit
    stagnant_pnl_threshold: float = 0.01      # <1% move is "stagnant"
    
    # Early exit triggers
    sentiment_drop_threshold: float = 0.3     # Exit if sentiment drops below
    regime_adverse_exit: bool = True          # Exit on adverse regime change


class PositionManagementAgent(Agent):
    """Manages open positions and generates exit/adjustment signals.

    INPUT DATA REQUIRED:
    --------------------
    - `positions`: List[OpenPosition] - Open trades to manage
    - `current_prices`: Dict[str, float] - Current market prices by symbol
    - `regime`: Optional[RegimeSignal] - Current market regime
    - `sentiments`: Optional[Dict[str, SentimentSignal]] - Per-symbol sentiment

    OUTPUT:
    -------
    - `updates`: List[PositionUpdate] with action recommendations
    """

    name = "position_management_agent"

    def __init__(self, config: Optional[PositionManagementConfig] = None):
        super().__init__(name=self.name)
        self._config = config or PositionManagementConfig()

    # ------------------------------------------------------------------
    # Core Position Update Logic
    # ------------------------------------------------------------------

    def _update_position_state(
        self, 
        position: OpenPosition, 
        current_price: float
    ) -> OpenPosition:
        """Update position with current market price."""
        position.current_price = current_price
        position.unrealized_pnl = (current_price - position.entry_price) * position.shares
        position.unrealized_pnl_pct = (current_price - position.entry_price) / position.entry_price
        
        # Update highest price for trailing stop
        if current_price > position.highest_price:
            position.highest_price = current_price
        
        return position

    def _check_stop_loss_hit(self, position: OpenPosition) -> Optional[PositionUpdate]:
        """Check if stop loss was hit."""
        if position.current_price <= position.current_stop_loss:
            loss_pct = (position.entry_price - position.current_price) / position.entry_price
            return PositionUpdate(
                symbol=position.symbol,
                action="FULL_EXIT",
                exit_price=position.current_price,
                exit_reason=ExitReason.STOP_LOSS_HIT,
                exit_shares=position.shares,
                reasoning=f"Stop loss triggered at ₹{position.current_stop_loss:.2f} ({loss_pct:.1%} loss)"
            )
        return None

    def _check_target_hit(self, position: OpenPosition) -> Optional[PositionUpdate]:
        """Check if target was hit."""
        if position.current_price >= position.target_price:
            gain_pct = position.unrealized_pnl_pct
            return PositionUpdate(
                symbol=position.symbol,
                action="FULL_EXIT",
                exit_price=position.current_price,
                exit_reason=ExitReason.TARGET_HIT,
                exit_shares=position.shares,
                reasoning=f"Target price hit at ₹{position.target_price:.2f} ({gain_pct:.1%} gain)"
            )
        return None

    def _check_partial_exit(self, position: OpenPosition) -> Optional[PositionUpdate]:
        """Check if partial profit should be taken."""
        cfg = self._config
        
        if not cfg.enable_partial_exits:
            return None
        
        # Skip if already partially exited (check via shares or flag)
        if position.partial_exit_at and position.current_price >= position.partial_exit_at:
            # Calculate shares to sell
            shares_to_sell = int(position.shares * cfg.partial_exit_ratio)
            if shares_to_sell > 0:
                gain_pct = position.unrealized_pnl_pct
                return PositionUpdate(
                    symbol=position.symbol,
                    action="PARTIAL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.TARGET_HIT,
                    exit_shares=shares_to_sell,
                    reasoning=f"Partial profit at ₹{position.current_price:.2f} ({gain_pct:.1%} gain) - selling {shares_to_sell} shares"
                )
        
        # Check if we should trigger partial based on profit
        if position.unrealized_pnl_pct >= cfg.partial_exit_trigger_pct:
            if not position.partial_exit_at:  # First time hitting partial threshold
                shares_to_sell = int(position.shares * cfg.partial_exit_ratio)
                if shares_to_sell > 0:
                    return PositionUpdate(
                        symbol=position.symbol,
                        action="PARTIAL_EXIT",
                        exit_price=position.current_price,
                        exit_reason=ExitReason.TARGET_HIT,
                        exit_shares=shares_to_sell,
                        reasoning=f"Taking {cfg.partial_exit_ratio:.0%} profit at {position.unrealized_pnl_pct:.1%} gain"
                    )
        
        return None

    def _check_trailing_stop_adjustment(
        self, 
        position: OpenPosition
    ) -> Optional[PositionUpdate]:
        """Check if trailing stop should be moved up."""
        cfg = self._config
        
        if not cfg.enable_trailing_stop:
            return None
        
        # Only trail if profitable enough
        if position.unrealized_pnl_pct < cfg.trailing_activation_pct:
            return None
        
        # Use position's trailing stop if set, else config default
        trail_pct = position.trailing_stop_pct or cfg.trailing_distance_pct
        
        # Calculate new trailing stop based on highest price
        new_stop = position.highest_price * (1 - trail_pct)
        
        # Only update if meaningfully higher than current stop
        if new_stop > position.current_stop_loss * (1 + cfg.min_stop_move_pct):
            # Don't let trailing stop go above entry (lock in profit)
            if new_stop > position.entry_price:
                return PositionUpdate(
                    symbol=position.symbol,
                    action="ADJUST_STOP",
                    new_stop_loss=round(new_stop, 2),
                    reasoning=f"Trailing stop raised to ₹{new_stop:.2f} (highest: ₹{position.highest_price:.2f})"
                )
        
        return None

    def _check_trailing_stop_hit(self, position: OpenPosition) -> Optional[PositionUpdate]:
        """Check if trailing stop was hit."""
        cfg = self._config
        
        if not cfg.enable_trailing_stop:
            return None
        
        # Check if price dropped from high to trailing trigger
        if position.unrealized_pnl_pct >= cfg.trailing_activation_pct:
            trail_pct = position.trailing_stop_pct or cfg.trailing_distance_pct
            trailing_stop = position.highest_price * (1 - trail_pct)
            
            if position.current_price <= trailing_stop and trailing_stop > position.entry_price:
                gain_pct = (trailing_stop - position.entry_price) / position.entry_price
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=trailing_stop,
                    exit_reason=ExitReason.TRAILING_STOP,
                    exit_shares=position.shares,
                    reasoning=f"Trailing stop triggered at ₹{trailing_stop:.2f} ({gain_pct:.1%} gain locked)"
                )
        
        return None

    def _check_time_exit(self, position: OpenPosition) -> Optional[PositionUpdate]:
        """Check for time-based exit conditions."""
        cfg = self._config
        
        # Max holding period exceeded
        if position.days_held >= cfg.max_holding_days:
            if position.unrealized_pnl_pct > 0:
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.TIME_EXIT,
                    exit_shares=position.shares,
                    reasoning=f"Max holding period ({cfg.max_holding_days}d) exceeded with {position.unrealized_pnl_pct:.1%} profit"
                )
            else:
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.TIME_EXIT,
                    exit_shares=position.shares,
                    reasoning=f"Max holding period ({cfg.max_holding_days}d) exceeded - cutting {position.unrealized_pnl_pct:.1%} loss"
                )
        
        # Stagnant position check
        if position.days_held >= cfg.stagnant_days_threshold:
            if abs(position.unrealized_pnl_pct) < cfg.stagnant_pnl_threshold:
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.TIME_EXIT,
                    exit_shares=position.shares,
                    reasoning=f"Position stagnant for {position.days_held} days ({position.unrealized_pnl_pct:.1%} move)"
                )
        
        return None

    def _check_regime_exit(
        self, 
        position: OpenPosition,
        regime: RegimeSignal,
    ) -> Optional[PositionUpdate]:
        """Check for regime-based exit."""
        cfg = self._config
        
        if not cfg.regime_adverse_exit:
            return None
        
        # Exit on extreme volatility or bear trend
        if regime.volatility_state == VolatilityState.EXTREME:
            return PositionUpdate(
                symbol=position.symbol,
                action="FULL_EXIT",
                exit_price=position.current_price,
                exit_reason=ExitReason.REGIME_CHANGE,
                exit_shares=position.shares,
                reasoning=f"Exiting due to extreme market volatility (P&L: {position.unrealized_pnl_pct:.1%})"
            )
        
        if regime.market_regime == MarketRegime.BEAR_TREND and regime.regime_confidence > 0.7:
            if position.unrealized_pnl_pct > 0:
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.REGIME_CHANGE,
                    exit_shares=position.shares,
                    reasoning=f"Exiting profitable position ({position.unrealized_pnl_pct:.1%}) due to confirmed bear trend"
                )
        
        return None

    def _check_sentiment_exit(
        self,
        position: OpenPosition,
        sentiment: Optional[SentimentSignal],
    ) -> Optional[PositionUpdate]:
        """Check for sentiment-based exit."""
        cfg = self._config
        
        if not sentiment:
            return None
        
        if sentiment.sentiment_score < cfg.sentiment_drop_threshold:
            if sentiment.sentiment_trend == "deteriorating":
                return PositionUpdate(
                    symbol=position.symbol,
                    action="FULL_EXIT",
                    exit_price=position.current_price,
                    exit_reason=ExitReason.SENTIMENT_CHANGE,
                    exit_shares=position.shares,
                    reasoning=f"Sentiment deteriorated to {sentiment.sentiment_score:.2f} with {sentiment.negative_news_count} negative news"
                )
        
        return None

    # ------------------------------------------------------------------
    # Main Analysis Logic
    # ------------------------------------------------------------------

    def analyze_position(
        self,
        position: OpenPosition,
        current_price: float,
        regime: Optional[RegimeSignal] = None,
        sentiment: Optional[SentimentSignal] = None,
    ) -> PositionUpdate:
        """Analyze a single position and return update recommendation.
        
        Checks in priority order:
        1. Stop loss hit
        2. Target hit
        3. Trailing stop hit
        4. Regime change exit
        5. Sentiment change exit
        6. Time-based exit
        7. Partial exit opportunity
        8. Trailing stop adjustment
        9. Hold
        """
        regime = regime or RegimeSignal()
        
        # Update position state
        position = self._update_position_state(position, current_price)
        
        # Priority order checks
        
        # 1. Stop loss
        update = self._check_stop_loss_hit(position)
        if update:
            return update
        
        # 2. Target hit
        update = self._check_target_hit(position)
        if update:
            return update
        
        # 3. Trailing stop hit
        update = self._check_trailing_stop_hit(position)
        if update:
            return update
        
        # 4. Regime change
        update = self._check_regime_exit(position, regime)
        if update:
            return update
        
        # 5. Sentiment deterioration
        update = self._check_sentiment_exit(position, sentiment)
        if update:
            return update
        
        # 6. Time-based
        update = self._check_time_exit(position)
        if update:
            return update
        
        # 7. Partial exit opportunity
        update = self._check_partial_exit(position)
        if update:
            return update
        
        # 8. Trailing stop adjustment
        update = self._check_trailing_stop_adjustment(position)
        if update:
            return update
        
        # 9. Default: HOLD
        return PositionUpdate(
            symbol=position.symbol,
            action="HOLD",
            reasoning=f"Holding at {position.unrealized_pnl_pct:+.1%} (day {position.days_held})"
        )

    def analyze_portfolio(
        self,
        positions: List[OpenPosition],
        current_prices: Dict[str, float],
        regime: Optional[RegimeSignal] = None,
        sentiments: Optional[Dict[str, SentimentSignal]] = None,
    ) -> List[PositionUpdate]:
        """Analyze all open positions and return updates."""
        sentiments = sentiments or {}
        updates = []
        
        for position in positions:
            current_price = current_prices.get(position.symbol, position.current_price)
            sentiment = sentiments.get(position.symbol)
            
            update = self.analyze_position(
                position=position,
                current_price=current_price,
                regime=regime,
                sentiment=sentiment,
            )
            updates.append(update)
        
        return updates

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute position management analysis.
        
        Expected kwargs:
        - positions: List[OpenPosition]
        - current_prices: Dict[str, float]
        - regime: Optional[RegimeSignal]
        - sentiments: Optional[Dict[str, SentimentSignal]]
        """
        started = self._pre_run()
        
        try:
            positions = kwargs.get("positions", [])
            if not positions:
                return self._result(
                    ctx=ctx,
                    success=True,
                    started=started,
                    completed=self._post_run(),
                    payload={"updates": [], "summary": "No open positions to manage"},
                )
            
            current_prices = kwargs.get("current_prices", {})
            regime = kwargs.get("regime")
            sentiments = kwargs.get("sentiments")
            
            updates = self.analyze_portfolio(
                positions=positions,
                current_prices=current_prices,
                regime=regime,
                sentiments=sentiments,
            )
            
            # Categorize updates
            exits = [u for u in updates if u.action in ("FULL_EXIT", "PARTIAL_EXIT")]
            adjustments = [u for u in updates if u.action == "ADJUST_STOP"]
            holds = [u for u in updates if u.action == "HOLD"]
            
            payload = {
                "updates": [
                    {
                        "symbol": u.symbol,
                        "action": u.action,
                        "new_stop_loss": u.new_stop_loss,
                        "exit_price": u.exit_price,
                        "exit_reason": u.exit_reason.value if u.exit_reason else None,
                        "exit_shares": u.exit_shares,
                        "reasoning": u.reasoning,
                    }
                    for u in updates
                ],
                "raw_updates": updates,
                "exits_count": len(exits),
                "adjustments_count": len(adjustments),
                "holds_count": len(holds),
            }
            
            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] PositionManagementAgent | {len(positions)} positions")
                for u in updates:
                    action_icon = {
                        "HOLD": "⏸️",
                        "ADJUST_STOP": "🔼",
                        "PARTIAL_EXIT": "🔸",
                        "FULL_EXIT": "🔴",
                    }.get(u.action, "?")
                    print(f"  {action_icon} {u.symbol}: {u.action} - {u.reasoning}")
            
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[str(len(exits)), str(len(adjustments)), str(len(holds))],
            )
            
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="POSITION_MGMT_ERROR", message=str(exc))],
            )
