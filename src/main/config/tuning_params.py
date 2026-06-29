"""Centralized tunable parameters for all LCF agents.

Every hardcoded weight, threshold, and magic number across the system
is defined here. Agents read from this config at initialization.

To tune: modify values here, or load from tuning_params.yaml override.

Categories:
  1. JUDGE    ΓÇö decision weights and thresholds
  2. TECHNICAL ΓÇö indicator periods and scoring weights
  3. FUNDAMENTAL ΓÇö valuation/growth/health thresholds
  4. SENTIMENT ΓÇö keyword lists and scoring
  5. EVENT    ΓÇö earnings window, gap thresholds
  6. REGIME   ΓÇö trend detection and volatility thresholds
  7. DEBATE   ΓÇö bull/bear scoring and hybrid weights
  8. TRADE    ΓÇö entry/exit/position sizing
  9. RISK     ΓÇö exposure limits and regime adjustments
  10. POSITION ΓÇö trailing stops, partial exits, time exits
"""

from __future__ import annotations

import os
import yaml
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# =========================================================================
# 1. JUDGE AGENT
# =========================================================================

@dataclass
class JudgeParams:
    """ML Meta-Judge weights and decision thresholds."""

    # Feature weights (should sum to ~1.0)
    weights: Dict[str, float] = field(default_factory=lambda: {
        # Technical (total 0.35)
        "tech_score":    0.15,
        "tech_trend":    0.10,
        "tech_macd":     0.05,
        "tech_breakout": 0.05,
        # Fundamental (total 0.22)
        "fund_score":    0.12,
        "fund_growth":   0.05,
        "fund_health":   0.05,
        # Sentiment (total 0.22)
        "sent_score":       0.10,
        "sent_net_ratio":   0.07,
        "sent_trend":       0.05,
        # Event (total 0.11)
        "evt_score":    0.06,
        "evt_gap_up":   0.03,
        "evt_earnings": 0.02,
        # Similarity/RAG (total 0.10)
        "sim_avg_return": 0.06,
        "sim_pos_rate":   0.04,
    })

    # Decision thresholds
    buy_threshold: float = 0.65
    sell_threshold: float = 0.35
    min_expected_return: float = 0.015       # 1.5%
    max_downside_risk: float = 0.25          # 25%
    max_position_size_pct: float = 0.02      # 2% of capital

    # Return estimation
    return_linear_scale: float = 0.10        # (prob - 0.5) * this

    # Default stop/target
    default_stop_loss_pct: float = -0.03
    default_take_profit_pct: float = 0.06


# =========================================================================
# 2. TECHNICAL AGENT
# =========================================================================

@dataclass
class TechnicalParams:
    """Technical indicator periods and scoring weights."""

    # Indicator periods
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    ema_short: int = 20
    ema_long: int = 50
    atr_period: int = 14
    support_resistance_window: int = 20
    week_52_lookback: int = 252

    # Trend detection thresholds
    trend_bullish_threshold: float = 1.001   # EMA20 > EMA50 * this
    trend_bearish_threshold: float = 0.999   # EMA20 < EMA50 * this

    # Breakout detection
    breakout_price_pct: float = 0.97         # within 3% of 52w high
    breakout_volume_spike: float = 1.3       # 30% above 20-day avg

    # RSI scoring range
    rsi_score_low: float = 30.0              # RSI below this = 0
    rsi_score_high: float = 70.0             # RSI above this = 1

    # Composite score weights
    weight_rsi: float = 0.30
    weight_macd: float = 0.25
    weight_trend: float = 0.30
    weight_breakout: float = 0.15


# =========================================================================
# 3. FUNDAMENTAL AGENT
# =========================================================================

@dataclass
class FundamentalParams:
    """Valuation, growth, and health thresholds."""

    # PE valuation buckets
    pe_undervalued: float = 12.0
    pe_overvalued: float = 25.0

    # Growth score mapping
    growth_offset: float = 10.0              # added to growth% * 100
    growth_scale: float = 50.0               # divided by this

    # Health score ΓÇö profit margin range
    margin_max: float = 0.30                 # 30% margin = score 1.0

    # Health score ΓÇö D/E ratio
    de_max: float = 2.0                      # D/E >= 2.0 = score 0.0

    # Health score ΓÇö ROE range
    roe_max: float = 0.30                    # 30% ROE = score 1.0

    # Health composite weights
    health_weight_margin: float = 0.40
    health_weight_de: float = 0.30
    health_weight_roe: float = 0.30

    # Fundamental composite weights
    fund_weight_valuation: float = 0.35
    fund_weight_growth: float = 0.35
    fund_weight_health: float = 0.30

    # Valuation score mapping
    valuation_undervalued: float = 1.0
    valuation_fair: float = 0.6
    valuation_overvalued: float = 0.2


# =========================================================================
# 4. SENTIMENT AGENT
# =========================================================================

@dataclass
class SentimentParams:
    """Keyword lists and sentiment scoring."""

    positive_keywords: List[str] = field(default_factory=lambda: [
        "beat", "record", "growth", "profit", "win", "contract", "upgrade",
        "strong", "outperform", "buy", "bullish", "surge", "rally", "gain",
        "expansion", "acquisition", "partnership", "award", "launch",
        "milestone", "dividend", "buyback", "raise", "guidance",
    ])

    negative_keywords: List[str] = field(default_factory=lambda: [
        "miss", "loss", "decline", "warning", "cut", "downgrade", "sell",
        "bearish", "drop", "fall", "plunge", "risk", "regulatory", "fine",
        "lawsuit", "fraud", "layoff", "recall", "bankruptcy", "weak",
        "probe", "investigation", "penalty", "debt", "default",
    ])

    # Confidence saturation
    confidence_saturation_articles: int = 10  # max articles for full confidence

    # Price change thresholds for trend
    price_change_improving: float = 0.01     # >1% = improving
    price_change_deteriorating: float = -0.01  # <-1% = deteriorating


# =========================================================================
# 5. EVENT AGENT
# =========================================================================

@dataclass
class EventParams:
    """Earnings window, gap thresholds, event scoring."""

    earnings_window_days: int = 7            # +/- days from today
    gap_threshold: float = 0.02              # 2% gap is significant
    gap_max_for_score: float = 0.05          # 5% gap = max score

    # Risk level thresholds
    risk_high_factors: int = 3               # >= 3 risk factors = high
    risk_medium_factors: int = 1             # >= 1 = medium

    # Event score composition
    weight_earnings: float = 0.40
    weight_gap: float = 0.30
    weight_risk: float = 0.30

    # Risk level score mapping
    risk_score_low: float = 0.1
    risk_score_medium: float = 0.5
    risk_score_high: float = 0.9


# =========================================================================
# 6. REGIME DETECTOR
# =========================================================================

@dataclass
class RegimeParams:
    """Market regime detection thresholds."""

    # RSI thresholds for index
    rsi_bull_threshold: float = 55.0
    rsi_bear_threshold: float = 45.0

    # SMA periods for trend
    trend_sma_short: int = 50
    trend_sma_long: int = 200

    # VIX thresholds (India VIX)
    vix_low: float = 12.0
    vix_moderate: float = 18.0
    vix_high: float = 25.0

    # ATR-based volatility fallback (% of price)
    atr_low: float = 1.0
    atr_moderate: float = 1.5
    atr_high: float = 2.5

    # Trend signal thresholds
    min_bull_signals: int = 3
    min_bear_signals: int = 3

    # Confidence calculation
    min_data_points: int = 50
    base_confidence: float = 0.5
    signal_confidence_step: float = 0.1
    max_confidence: float = 0.9
    sideways_confidence: float = 0.6


# =========================================================================
# 7. DEBATE AGENTS
# =========================================================================

@dataclass
class DebateParams:
    """Bull/Bear agent thresholds and hybrid decision weights."""

    # Bull agent
    bull_buy_confidence_threshold: float = 0.5
    bull_tech_score_strong: float = 0.6
    bull_pe_attractive: float = 20.0
    bull_growth_strong: float = 0.10
    bull_roe_high: float = 0.15
    bull_margin_healthy: float = 0.15
    bull_sentiment_positive: float = 0.6
    bull_near_52w_low_pct: float = 0.15      # within 15% of 52w low

    # Bear agent
    bear_sell_confidence_threshold: float = 0.65
    bear_rsi_overbought: float = 70.0
    bear_tech_score_weak: float = 0.4
    bear_volatility_high: float = 0.7
    bear_pe_expensive: float = 40.0
    bear_growth_weak: float = 0.05
    bear_de_high: float = 1.5
    bear_margin_thin: float = 0.05
    bear_sentiment_negative: float = 0.4
    bear_below_52w_high_pct: float = 0.25    # >25% below 52w high
    bear_recent_decline_pct: float = 3.0     # >3% in 5 sessions
    bear_recent_decline_sessions: int = 5

    # Debate evaluator
    debate_bull_win_threshold: float = 0.6   # bull_strength > this = BUY
    debate_bear_win_threshold: float = 0.6   # bear_strength > this = SELL
    debate_bear_sell_confidence: float = 0.7 # bear confidence needed for SELL
    debate_point_weight: float = 0.1         # per key_point bonus

    # Hybrid decision
    rule_weight: float = 0.6
    debate_weight: float = 0.4
    agreement_boost: float = 1.1             # multiply confidence when agreed
    max_confidence: float = 0.95
    disagreement_confidence_scale: float = 0.5  # reduce on strong disagree

    # Weighted average thresholds
    hybrid_buy_threshold: float = 0.65
    hybrid_sell_threshold: float = 0.35


# =========================================================================
# 8. TRADE PLANNER
# =========================================================================

@dataclass
class TradeParams:
    """Entry, exit, and position sizing parameters."""

    # Risk-reward
    min_risk_reward_ratio: float = 1.5
    default_risk_reward: float = 2.0
    max_risk_reward: float = 4.0

    # Stop-loss
    atr_stop_multiplier: float = 1.5
    min_stop_pct: float = 0.02              # 2%
    max_stop_pct: float = 0.08              # 8%
    high_vol_stop_widen: float = 0.01       # 1% wider in HIGH vol
    extreme_vol_stop_widen: float = 0.02    # 2% wider in EXTREME vol
    support_stop_buffer: float = 0.02       # 2% below support

    # Position sizing
    max_risk_per_trade_pct: float = 0.01    # 1% portfolio risk
    max_position_size_pct: float = 0.05     # 5% max position
    default_portfolio_value: float = 1_000_000.0

    # Entry
    limit_offset_pct: float = 0.005         # 0.5% below current
    breakout_buffer_pct: float = 0.003      # 0.3% above resistance

    # Timing
    default_holding_days: int = 5
    limit_order_validity_days: int = 2

    # Trailing stop
    trailing_activation_pct: float = 0.03   # activate after 3% profit
    trailing_distance_pct: float = 0.02     # trail by 2%

    # Target adjustment
    low_confidence_target_reduction: float = 0.20  # reduce 20% if conf < 0.6
    low_confidence_threshold: float = 0.6

    # Support/resistance
    pivot_lookback: int = 50
    pivot_window: int = 4


# =========================================================================
# 9. RISK MANAGER
# =========================================================================

@dataclass
class RiskParams:
    """Portfolio risk limits and regime adjustments."""

    # Position limits
    max_single_position_pct: float = 0.05   # 5%
    max_total_exposure_pct: float = 0.60    # 60%
    min_position_size_pct: float = 0.005    # 0.5%

    # Volatility scaling
    vol_scale_low: float = 1.2
    vol_scale_moderate: float = 1.0
    vol_scale_high: float = 0.6
    vol_scale_extreme: float = 0.3

    # Regime scaling
    regime_scale_bull: float = 1.1
    regime_scale_sideways: float = 1.0
    regime_scale_bear: float = 0.7
    regime_scale_high_vol: float = 0.5

    # Stop-loss defaults by volatility
    stop_low: float = 0.025
    stop_moderate: float = 0.03
    stop_high: float = 0.05
    stop_extreme: float = 0.06

    # ATR-based stop/target
    atr_stop_multiplier: float = 2.0
    atr_profit_multiplier: float = 3.0
    min_stop_loss_pct: float = 0.02
    max_stop_loss_pct: float = 0.08

    # Drawdown
    max_portfolio_drawdown_pct: float = 0.15
    drawdown_scale_factor: float = 0.5

    # Sector concentration
    max_sector_exposure_pct: float = 0.25

    # No-trade conditions
    min_confidence_threshold: float = 0.45
    min_expected_return: float = 0.005

    # Risk score composition
    risk_weight_confidence: float = 0.3
    risk_weight_downside: float = 0.3
    risk_weight_volatility: float = 0.2
    risk_weight_regime: float = 0.2

    # Risk level thresholds
    risk_very_low: float = 0.2
    risk_low: float = 0.4
    risk_moderate: float = 0.6
    risk_high: float = 0.8

    # Portfolio risk composition
    portfolio_risk_weight_position: float = 0.4
    portfolio_risk_weight_exposure: float = 0.3
    portfolio_risk_weight_correlation: float = 0.2
    portfolio_risk_weight_drawdown: float = 0.1


# =========================================================================
# 10. POSITION MANAGEMENT
# =========================================================================

@dataclass
class PositionParams:
    """Trailing stops, partial exits, time-based exits."""

    # Trailing stop
    trailing_activation_pct: float = 0.03
    trailing_distance_pct: float = 0.02
    min_stop_move_pct: float = 0.005

    # Partial exits
    partial_exit_trigger_pct: float = 0.04  # first partial at 4% profit
    partial_exit_ratio: float = 0.5         # sell 50%

    # Time-based
    max_holding_days: int = 10
    stagnant_days_threshold: int = 5
    stagnant_pnl_threshold: float = 0.01    # <1% = stagnant

    # Early exit
    sentiment_drop_threshold: float = 0.3
    regime_adverse_exit: bool = True
    regime_adverse_min_confidence: float = 0.7


# =========================================================================
# MASTER CONFIG ΓÇö combines all sections
# =========================================================================

@dataclass
class TuningConfig:
    """Master configuration combining all tunable parameters."""
    judge: JudgeParams = field(default_factory=JudgeParams)
    technical: TechnicalParams = field(default_factory=TechnicalParams)
    fundamental: FundamentalParams = field(default_factory=FundamentalParams)
    sentiment: SentimentParams = field(default_factory=SentimentParams)
    event: EventParams = field(default_factory=EventParams)
    regime: RegimeParams = field(default_factory=RegimeParams)
    debate: DebateParams = field(default_factory=DebateParams)
    trade: TradeParams = field(default_factory=TradeParams)
    risk: RiskParams = field(default_factory=RiskParams)
    position: PositionParams = field(default_factory=PositionParams)


def load_tuning_config(yaml_path: Optional[str] = None) -> TuningConfig:
    """Load tuning config, optionally overriding from a YAML file.

    If yaml_path is provided, values in the YAML override the defaults.
    Only specified keys are overridden ΓÇö unspecified keys keep defaults.
    """
    config = TuningConfig()

    if yaml_path and os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            overrides = yaml.safe_load(f) or {}

        # Apply overrides to each section
        for section_name, section_obj in [
            ("judge", config.judge),
            ("technical", config.technical),
            ("fundamental", config.fundamental),
            ("sentiment", config.sentiment),
            ("event", config.event),
            ("regime", config.regime),
            ("debate", config.debate),
            ("trade", config.trade),
            ("risk", config.risk),
            ("position", config.position),
        ]:
            section_overrides = overrides.get(section_name, {})
            for key, value in section_overrides.items():
                if hasattr(section_obj, key):
                    setattr(section_obj, key, value)

    return config


# Singleton default instance
DEFAULT_CONFIG = TuningConfig()


# =========================================================================
# TRADING MODES ΓÇö preset parameter profiles
# =========================================================================

from enum import Enum


class TradingMode(str, Enum):
    """Available trading modes with different risk/return profiles."""
    CONSERVATIVE = "conservative"
    AGGRESSIVE = "aggressive"
    MOMENTUM = "momentum"
    VALUE = "value"
    SCALPER = "scalper"
    ADAPTIVE = "adaptive"


@dataclass
class ModeConfig:
    """Runtime config for a trading mode (beyond tuning params)."""
    max_positions: int = 5
    hold_days: int = 5
    description: str = ""


# Mode-specific parameter overrides and runtime config
MODE_PRESETS: Dict[str, Dict[str, Any]] = {

    "conservative": {
        "runtime": {"max_positions": 3, "hold_days": 5,
                     "description": "Protect capital, low risk, sleep well"},
        "judge": {
            "buy_threshold": 0.80,
            "sell_threshold": 0.30,
            "min_expected_return": 0.025,
        },
        "debate": {
            "rule_weight": 0.5,
            "debate_weight": 0.5,
            "bull_buy_confidence_threshold": 0.65,
            "bear_sell_confidence_threshold": 0.50,
            "hybrid_buy_threshold": 0.70,
            "hybrid_sell_threshold": 0.30,
            "agreement_boost": 1.2,
        },
        "risk": {
            "max_single_position_pct": 0.03,
            "max_total_exposure_pct": 0.30,
        },
        "trade": {
            "max_risk_per_trade_pct": 0.005,
            "atr_stop_multiplier": 2.0,
        },
    },

    "aggressive": {
        "runtime": {"max_positions": 10, "hold_days": 7,
                     "description": "Max return, accept higher risk"},
        "judge": {
            "buy_threshold": 0.55,
            "sell_threshold": 0.40,
            "min_expected_return": 0.005,
        },
        "debate": {
            "rule_weight": 0.4,
            "debate_weight": 0.6,
            "bull_buy_confidence_threshold": 0.35,
            "bear_sell_confidence_threshold": 0.70,
            "hybrid_buy_threshold": 0.45,
            "hybrid_sell_threshold": 0.40,
            "agreement_boost": 1.0,
        },
        "risk": {
            "max_single_position_pct": 0.08,
            "max_total_exposure_pct": 0.80,
        },
        "trade": {
            "max_risk_per_trade_pct": 0.02,
        },
    },

    "momentum": {
        "runtime": {"max_positions": 8, "hold_days": 7,
                     "description": "Ride trends, cut losers fast"},
        "judge": {
            "weights": {
                "tech_score": 0.25, "tech_trend": 0.20, "tech_macd": 0.15,
                "tech_breakout": 0.10, "fund_score": 0.03, "fund_growth": 0.02,
                "fund_health": 0.02, "sent_score": 0.08, "sent_net_ratio": 0.05,
                "sent_trend": 0.05, "evt_score": 0.02, "evt_gap_up": 0.01,
                "evt_earnings": 0.01, "sim_avg_return": 0.01, "sim_pos_rate": 0.00,
            },
            "buy_threshold": 0.60,
            "sell_threshold": 0.35,
            "min_expected_return": 0.01,
        },
        "debate": {
            "rule_weight": 0.3,
            "debate_weight": 0.7,
            "bull_tech_score_strong": 0.55,
            "hybrid_buy_threshold": 0.50,
        },
        "trade": {
            "trailing_activation_pct": 0.02,
            "trailing_distance_pct": 0.015,
        },
    },

    "value": {
        "runtime": {"max_positions": 5, "hold_days": 10,
                     "description": "Buy undervalued, hold longer"},
        "judge": {
            "weights": {
                "tech_score": 0.05, "tech_trend": 0.05, "tech_macd": 0.02,
                "tech_breakout": 0.01, "fund_score": 0.25, "fund_growth": 0.18,
                "fund_health": 0.15, "sent_score": 0.08, "sent_net_ratio": 0.05,
                "sent_trend": 0.03, "evt_score": 0.05, "evt_gap_up": 0.02,
                "evt_earnings": 0.03, "sim_avg_return": 0.02, "sim_pos_rate": 0.01,
            },
            "buy_threshold": 0.62,
            "min_expected_return": 0.012,
        },
        "fundamental": {
            "pe_undervalued": 18.0,
            "pe_overvalued": 22.0,
        },
        "debate": {
            "rule_weight": 0.6,
            "debate_weight": 0.4,
            "bull_pe_attractive": 22.0,
            "bear_pe_expensive": 35.0,
            "hybrid_buy_threshold": 0.55,
        },
    },

    "scalper": {
        "runtime": {"max_positions": 10, "hold_days": 2,
                     "description": "Many small wins, very short holds"},
        "judge": {
            "weights": {
                "tech_score": 0.20, "tech_trend": 0.05, "tech_macd": 0.10,
                "tech_breakout": 0.02, "fund_score": 0.08, "fund_growth": 0.05,
                "fund_health": 0.05, "sent_score": 0.15, "sent_net_ratio": 0.10,
                "sent_trend": 0.05, "evt_score": 0.05, "evt_gap_up": 0.05,
                "evt_earnings": 0.02, "sim_avg_return": 0.02, "sim_pos_rate": 0.01,
            },
            "buy_threshold": 0.58,
            "sell_threshold": 0.38,
            "min_expected_return": 0.005,
        },
        "debate": {
            "rule_weight": 0.5,
            "debate_weight": 0.5,
            "hybrid_buy_threshold": 0.50,
            "bull_buy_confidence_threshold": 0.40,
        },
        "trade": {
            "min_stop_pct": 0.01,
            "max_stop_pct": 0.03,
            "default_holding_days": 2,
        },
    },

    "adaptive": {
        "runtime": {"max_positions": 5, "hold_days": 5,
                     "description": "Auto-switch mode based on market regime"},
        # Adaptive uses default params as base; the orchestrator switches
        # mode presets dynamically based on regime detection
    },
}


def get_mode_config(mode: str) -> ModeConfig:
    """Get the runtime config for a trading mode."""
    preset = MODE_PRESETS.get(mode, {})
    runtime = preset.get("runtime", {})
    return ModeConfig(
        max_positions=runtime.get("max_positions", 5),
        hold_days=runtime.get("hold_days", 5),
        description=runtime.get("description", ""),
    )


def load_mode_tuning_config(mode: str, yaml_path: Optional[str] = None) -> TuningConfig:
    """Load TuningConfig with mode-specific overrides applied.

    Priority: defaults -> mode preset -> yaml overrides
    """
    config = load_tuning_config(yaml_path)

    preset = MODE_PRESETS.get(mode, {})
    if not preset:
        return config

    # Apply mode overrides
    section_map = {
        "judge": config.judge,
        "technical": config.technical,
        "fundamental": config.fundamental,
        "sentiment": config.sentiment,
        "event": config.event,
        "regime": config.regime,
        "debate": config.debate,
        "trade": config.trade,
        "risk": config.risk,
        "position": config.position,
    }

    for section_name, section_obj in section_map.items():
        overrides = preset.get(section_name, {})
        for key, value in overrides.items():
            if hasattr(section_obj, key):
                setattr(section_obj, key, value)

    return config
