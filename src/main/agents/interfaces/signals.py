"""Structured signal data models for all trading agents.

Each agent outputs a typed dataclass ΓÇö no free-text opinions.
These are the features that feed the ML Meta-Judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """Discrete catalyst events (kept for backward compatibility)."""
    EARNINGS_BEAT = "earnings_beat"
    EARNINGS_MISS = "earnings_miss"
    GUIDANCE_RAISE = "guidance_raise"
    GUIDANCE_CUT = "guidance_cut"
    MA_ANNOUNCEMENT = "ma_announcement"
    CEO_CHANGE = "ceo_change"
    REGULATORY_NEWS = "regulatory_news"
    BIG_CONTRACT = "big_contract"
    INSIDER_BUY = "insider_buy"
    INSIDER_SELL = "insider_sell"
    BUYBACK = "buyback"
    NONE = "none"


class MarketRegime(str, Enum):
    """Market regime labels from the Regime Detector."""
    BULL_TREND = "bull_trend"
    BEAR_TREND = "bear_trend"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class VolatilityState(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# Agent Signal Outputs
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TechnicalSignal:
    """Structured output from the Technical Agent.

    Computed directly from raw OHLCV bars ΓÇö no pre-computed indicators needed.
    """
    technical_score: float          # 0ΓÇô1 composite score
    rsi: float                      # raw RSI value (0ΓÇô100)
    macd_signal: str                # "buy" | "sell" | "neutral"
    volatility: float               # normalised ATR/price ratio [0, 1]
    breakout_flag: bool             # True if near 52-week high with volume
    trend_direction: str            # "bullish" | "bearish" | "neutral"

    # Optional detail
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None

    def to_feature_dict(self) -> Dict[str, Any]:
        return {
            "tech_score": self.technical_score,
            "tech_rsi": self.rsi / 100.0,           # normalise to [0, 1]
            "tech_macd": (
                1.0 if self.macd_signal == "buy"
                else 0.0 if self.macd_signal == "sell"
                else 0.5
            ),
            "tech_volatility": self.volatility,
            "tech_breakout": 1.0 if self.breakout_flag else 0.0,
            "tech_trend": (
                1.0 if self.trend_direction == "bullish"
                else 0.0 if self.trend_direction == "bearish"
                else 0.5
            ),
        }


@dataclass(slots=True)
class FundamentalSignal:
    """Structured output from the Fundamental Agent.

    Uses real valuation and financial-health metrics from Yahoo Finance.
    """
    fundamental_score: float        # 0ΓÇô1 composite score
    valuation_label: str            # "undervalued" | "fair" | "overvalued"
    growth_score: float             # 0ΓÇô1 revenue growth signal
    financial_health_score: float   # 0ΓÇô1 margin/debt/ROE composite

    # Optional detail
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None

    def to_feature_dict(self) -> Dict[str, Any]:
        valuation_map = {"undervalued": 1.0, "fair": 0.5, "overvalued": 0.0}
        return {
            "fund_score": self.fundamental_score,
            "fund_valuation": valuation_map.get(self.valuation_label, 0.5),
            "fund_growth": self.growth_score,
            "fund_health": self.financial_health_score,
        }


@dataclass(slots=True)
class SentimentSignal:
    """Structured output from the Sentiment Agent.

    Built from news_articles list ΓÇö LLM or keyword rule-based path.
    """
    sentiment_score: float          # 0 = very negative ΓÇª 1 = very positive
    positive_news_count: int
    negative_news_count: int
    sentiment_trend: str            # "improving" | "stable" | "deteriorating"
    news_confidence_score: float    # 0ΓÇô1, scales with article count & recency

    def to_feature_dict(self) -> Dict[str, Any]:
        total = self.positive_news_count + self.negative_news_count
        net_ratio = self.positive_news_count / (total + 1)   # Laplace-smoothed
        trend_map = {"improving": 1.0, "stable": 0.5, "deteriorating": 0.0}
        return {
            "sent_score": self.sentiment_score,
            "sent_net_ratio": round(net_ratio, 4),
            "sent_trend": trend_map.get(self.sentiment_trend, 0.5),
            "sent_confidence": self.news_confidence_score,
        }


@dataclass(slots=True)
class EventSignal:
    """Structured output from the Event Agent.

    Detects earnings proximity, price gaps, dividends/splits, and major news.
    """
    event_score: float              # 0ΓÇô1 composite event impact
    earnings_impact_flag: bool      # earnings event within ┬▒7 days
    event_risk_level: str           # "low" | "medium" | "high"
    gap_up_flag: bool
    gap_down_flag: bool

    # Optional LLM-classified detail
    event_type: Optional[str] = None
    event_description: Optional[str] = None

    def to_feature_dict(self) -> Dict[str, Any]:
        risk_map = {"low": 0.0, "medium": 0.5, "high": 1.0}
        return {
            "evt_score": self.event_score,
            "evt_earnings": 1.0 if self.earnings_impact_flag else 0.0,
            "evt_risk": risk_map.get(self.event_risk_level, 0.0),
            "evt_gap_up": 1.0 if self.gap_up_flag else 0.0,
            "evt_gap_down": 1.0 if self.gap_down_flag else 0.0,
        }


@dataclass(slots=True)
class RegimeSignal:
    """Market regime context (from Regime Detector layer)."""
    market_regime: MarketRegime = MarketRegime.SIDEWAYS
    volatility_state: VolatilityState = VolatilityState.MODERATE
    regime_confidence: float = 0.5

    def to_feature_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.market_regime.value,
            "vol_state": self.volatility_state.value,
            "regime_confidence": self.regime_confidence,
        }


# ---------------------------------------------------------------------------
# Aggregated Feature Bundle (input to ML Judge)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AgentFeatureBundle:
    """Combines all agent signals into a single feature bundle for the ML Judge."""
    symbol: str
    date: str                                       # ISO date string
    technical: TechnicalSignal
    fundamental: FundamentalSignal
    sentiment: SentimentSignal
    event: EventSignal
    regime: RegimeSignal
    similarity_avg_return: float = 0.0              # from RAG layer
    similarity_positive_rate: float = 0.0           # from RAG layer
    similarity_max_drawdown: float = 0.0            # from RAG layer
    extra_features: Dict[str, Any] = field(default_factory=dict)

    def to_flat_features(self) -> Dict[str, Any]:
        """Merge all signals into a flat feature dict for ML model input."""
        features: Dict[str, Any] = {"symbol": self.symbol, "date": self.date}
        features.update(self.technical.to_feature_dict())
        features.update(self.fundamental.to_feature_dict())
        features.update(self.sentiment.to_feature_dict())
        features.update(self.event.to_feature_dict())
        features.update(self.regime.to_feature_dict())
        features["sim_avg_return"] = self.similarity_avg_return
        features["sim_pos_rate"] = self.similarity_positive_rate
        features["sim_max_dd"] = self.similarity_max_drawdown
        features.update(self.extra_features)
        return features


# ---------------------------------------------------------------------------
# Judge Decision Output
# ---------------------------------------------------------------------------

class EntryType(str, Enum):
    """Order entry type."""
    MARKET = "market"
    LIMIT = "limit"


class ExitReason(str, Enum):
    """Reason for exiting a position."""
    TARGET_HIT = "target_hit"
    STOP_LOSS_HIT = "stop_loss_hit"
    TRAILING_STOP = "trailing_stop"
    TIME_EXIT = "time_exit"
    SENTIMENT_CHANGE = "sentiment_change"
    REGIME_CHANGE = "regime_change"
    MANUAL = "manual"


@dataclass(slots=True)
class JudgeDecision:
    """Output from the ML Meta-Judge."""
    symbol: str
    date: str
    decision: str                         # BUY | SELL | HOLD
    prob_up_5d: float                     # probability of +3% in 5 days
    expected_return_5d: float             # point estimate
    downside_risk_prob: float             # probability of -3% in 5 days
    confidence: float
    position_size_pct: float = 0.0        # recommended allocation (0ΓÇô2% of capital)
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    reasoning: Optional[str] = None       # optional LLM-generated explanation
    feature_importances: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class TradePlan:
    """Complete trade plan with entry, exit, and risk parameters.
    
    This is the enhanced output that tells exactly how to execute a trade.
    """
    # Core decision
    symbol: str
    date: str
    decision: str                         # BUY | SELL | HOLD
    confidence: float                     # 0-1 confidence score
    
    # Entry parameters
    entry_type: EntryType                 # MARKET or LIMIT order
    entry_price: float                    # Suggested entry price
    current_price: float                  # Current market price
    
    # Exit parameters
    stop_loss_price: float                # Exit if trade fails
    target_price: float                   # Take profit target
    trailing_stop_pct: Optional[float] = None  # Trailing stop percentage
    
    # Risk metrics
    risk_reward_ratio: float = 0.0        # reward ├╖ risk (>1 is good)
    risk_per_share: float = 0.0           # entry - stop_loss
    reward_per_share: float = 0.0         # target - entry
    
    # Position sizing
    position_size_pct: float = 0.0        # % of portfolio
    suggested_shares: int = 0             # Number of shares for given capital
    max_loss_amount: float = 0.0          # Maximum loss if stop hit
    
    # Timing
    expected_holding_days: int = 5        # Estimated hold period
    entry_valid_until: Optional[str] = None  # Limit order expiry
    
    # Technical context
    support_level: Optional[float] = None
    resistance_level: Optional[float] = None
    atr: Optional[float] = None           # Average True Range
    
    # Reasoning
    reasoning: str = ""                   # Human-readable explanation
    entry_reasoning: str = ""             # Why this entry price
    exit_reasoning: str = ""              # Why this stop/target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "decision": self.decision,
            "confidence": self.confidence,
            "entry_type": self.entry_type.value,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "stop_loss_price": self.stop_loss_price,
            "target_price": self.target_price,
            "trailing_stop_pct": self.trailing_stop_pct,
            "risk_reward_ratio": round(self.risk_reward_ratio, 2),
            "position_size_pct": round(self.position_size_pct, 4),
            "suggested_shares": self.suggested_shares,
            "expected_holding_days": self.expected_holding_days,
            "support_level": self.support_level,
            "resistance_level": self.resistance_level,
            "reasoning": self.reasoning,
        }
    
    def summary(self) -> str:
        """One-line summary for display."""
        return (
            f"{self.decision} {self.symbol} @ Γé╣{self.entry_price:.2f} "
            f"(SL: Γé╣{self.stop_loss_price:.2f}, Target: Γé╣{self.target_price:.2f}, "
            f"R:R={self.risk_reward_ratio:.1f})"
        )


@dataclass(slots=True)
class OpenPosition:
    """Represents an open trading position being managed."""
    symbol: str
    entry_date: str
    entry_price: float
    shares: int
    original_stop_loss: float
    current_stop_loss: float
    target_price: float
    trailing_stop_pct: Optional[float] = None
    
    # Current state
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    days_held: int = 0
    highest_price: float = 0.0            # For trailing stop
    
    # Partial exits
    partial_exit_at: Optional[float] = None  # Price for partial profit
    partial_exit_pct: float = 0.5         # % to sell at partial target


@dataclass(slots=True)
class PositionUpdate:
    """Update recommendation for an open position."""
    symbol: str
    action: str                           # HOLD | ADJUST_STOP | PARTIAL_EXIT | FULL_EXIT
    new_stop_loss: Optional[float] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[ExitReason] = None
    exit_shares: int = 0                  # For partial exit
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Debate Flow Signals
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class DebateArgument:
    """A single argument from the Bull or Bear agent."""
    role: str                             # "bull" | "bear"
    recommendation: str                   # BUY | SELL | HOLD
    confidence: float                     # 0-1
    key_points: List[str] = field(default_factory=list)
    data_citations: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "recommendation": self.recommendation,
            "confidence": self.confidence,
            "key_points": self.key_points,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class CrossTickerSignal:
    """Cross-ticker / indirect correlation result from the CrossTickerAgent.

    Explains an intraday move by relating it to its own news AND to peer news
    (competitors / suppliers / customers). For example a competitor's guidance
    miss can be a tailwind for the mover, or a supplier disruption a headwind.
    """
    symbol: str
    move_direction: str                   # "rise" | "loss" | "flat"
    move_pct: float                       # signed % move that triggered analysis
    news_synced: bool                     # is the move explained by any news?
    explanation_type: str                 # own_news | peer_spillover | sector_move | unexplained
    driver_ticker: Optional[str] = None   # the peer responsible (if peer_spillover)
    driver_headline: str = ""             # the headline driving the conclusion
    confidence: float = 0.0               # 0-1 confidence in the explanation
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "move_direction": self.move_direction,
            "move_pct": round(self.move_pct, 4),
            "news_synced": self.news_synced,
            "explanation_type": self.explanation_type,
            "driver_ticker": self.driver_ticker,
            "driver_headline": self.driver_headline,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class DebateDecision:
    """Output from the Debate Judge ΓÇö evaluates bull vs bear arguments."""
    symbol: str
    date: str
    decision: str                         # BUY | SELL | HOLD
    confidence: float                     # 0-1
    winning_side: str                     # "bull" | "bear" | "neutral"
    bull_strength: float                  # 0-1 how strong was the bull case
    bear_strength: float                  # 0-1 how strong was the bear case
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "winning_side": self.winning_side,
            "bull_strength": round(self.bull_strength, 4),
            "bear_strength": round(self.bear_strength, 4),
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class HybridDecision:
    """Final decision combining rule-based and debate flows."""
    symbol: str
    date: str
    final_decision: str                   # BUY | SELL | HOLD
    final_confidence: float
    rule_decision: str                    # from JudgeAgent
    rule_confidence: float
    debate_decision: str                  # from DebateAgent
    debate_confidence: float
    agreement: bool                       # True if both flows agree
    disagreement_action: Optional[str] = None  # HOLD if strong disagreement
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "final_decision": self.final_decision,
            "final_confidence": round(self.final_confidence, 4),
            "rule_decision": self.rule_decision,
            "rule_confidence": round(self.rule_confidence, 4),
            "debate_decision": self.debate_decision,
            "debate_confidence": round(self.debate_confidence, 4),
            "agreement": self.agreement,
            "disagreement_action": self.disagreement_action,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# Debate Context ΓÇö rich input for Bull/Bear agents
# ---------------------------------------------------------------------------

@dataclass
class DebateContext:
    """Rich context for Bull/Bear debate agents.

    Combines selected raw evidence from StockDataContext with pre-computed
    agent signals so debate agents can both cite real data AND reference
    what analysis agents concluded.

    Kept intentionally slim to control LLM token budgets:
      - recent_ohlc: last 20 bars (not 500)
      - news_headlines: last 10 items
      - fundamentals: full object with real numbers
      - events: upcoming 14 days only
    """

    # --- Identity & price ---
    symbol: str
    date: str
    latest_price: float
    previous_close: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None

    # --- Raw evidence (selected slices) ---
    recent_ohlc: List[Dict[str, Any]] = field(default_factory=list)
    news_headlines: List[Dict[str, str]] = field(default_factory=list)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    upcoming_events: List[Dict[str, Any]] = field(default_factory=list)

    # --- Market context ---
    regime: Optional[RegimeSignal] = None

    # --- What analysis agents concluded ---
    signals: Optional[AgentFeatureBundle] = None

    # --- RAG: similar past setups (from PatternStore) ---
    similar_past_setups: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"symbol": "TCS", "date": "2024-11-15", "return_5d": 0.042, "regime": "bull_trend"}]

    # --- Memory: past trades for this symbol (from TradeMemory) ---
    past_trades_this_symbol: List[Dict[str, Any]] = field(default_factory=list)
    # e.g. [{"date": "2024-09-20", "decision": "BUY", "pnl": "+3.1%", "notes": "target in 4d"}]

    # --- Memory: warnings from past mistakes ---
    mistake_warnings: List[str] = field(default_factory=list)
    # e.g. ["Last BUY in bear_trend hit stop loss (-3.2%)"]

    def format_for_llm(self) -> str:
        """Format into a concise text block for LLM prompts."""
        lines = [f"Stock: {self.symbol} | Price: {self.latest_price}"]

        if self.previous_close and self.latest_price:
            chg = (self.latest_price - self.previous_close) / self.previous_close * 100
            lines.append(f"Day Change: {chg:+.2f}%")
        if self.week_52_high is not None:
            lines.append(f"52-Week High: {self.week_52_high}")
        if self.week_52_low is not None:
            lines.append(f"52-Week Low: {self.week_52_low}")

        # Fundamentals (real numbers)
        if self.fundamentals:
            f = self.fundamentals
            parts = []
            if f.get("pe_ratio") is not None:
                parts.append(f"PE={f['pe_ratio']:.1f}")
            if f.get("forward_pe") is not None:
                parts.append(f"FwdPE={f['forward_pe']:.1f}")
            if f.get("eps") is not None:
                parts.append(f"EPS={f['eps']:.2f}")
            if f.get("revenue_growth_yoy") is not None:
                parts.append(f"RevGrowth={f['revenue_growth_yoy']:.1%}")
            if f.get("profit_margin") is not None:
                parts.append(f"Margin={f['profit_margin']:.1%}")
            if f.get("debt_to_equity") is not None:
                parts.append(f"D/E={f['debt_to_equity']:.2f}")
            if f.get("return_on_equity") is not None:
                parts.append(f"ROE={f['return_on_equity']:.1%}")
            if f.get("sector"):
                parts.append(f"Sector={f['sector']}")
            if parts:
                lines.append("Fundamentals: " + " | ".join(parts))

        # Recent price action (last 5 bars summary)
        if self.recent_ohlc:
            last5 = self.recent_ohlc[-5:]
            closes = [b["close"] for b in last5 if "close" in b]
            if closes:
                lines.append(f"Last 5 closes: {[round(c, 2) for c in closes]}")

        # News headlines
        if self.news_headlines:
            lines.append(f"Recent News ({len(self.news_headlines)} articles):")
            for i, item in enumerate(self.news_headlines[:10], 1):
                src = item.get("source", "")
                date = item.get("date", "")
                tag = f" [{src}]" if src else ""
                tag += f" ({date})" if date else ""
                lines.append(f"  {i}. {item.get('headline', 'N/A')}{tag}")

        # Events
        if self.upcoming_events:
            lines.append("Upcoming Events:")
            for evt in self.upcoming_events[:5]:
                lines.append(f"  - {evt.get('event_type', 'N/A')}: {evt.get('description', 'N/A')}")

        # Market regime
        if self.regime:
            lines.append(
                f"Market Regime: {self.regime.market_regime.value} "
                f"| Volatility: {self.regime.volatility_state.value} "
                f"| Confidence: {self.regime.regime_confidence:.0%}"
            )

        # Agent signal summaries (what the analysis agents concluded)
        if self.signals:
            t = self.signals.technical
            lines.append(
                f"Technical Summary: score={t.technical_score:.2f} RSI={t.rsi:.0f} "
                f"MACD={t.macd_signal} trend={t.trend_direction} "
                f"breakout={'YES' if t.breakout_flag else 'no'}"
            )
            f_sig = self.signals.fundamental
            lines.append(
                f"Fundamental Summary: score={f_sig.fundamental_score:.2f} "
                f"valuation={f_sig.valuation_label} growth={f_sig.growth_score:.2f} "
                f"health={f_sig.financial_health_score:.2f}"
            )
            s = self.signals.sentiment
            lines.append(
                f"Sentiment Summary: score={s.sentiment_score:.2f} "
                f"+news={s.positive_news_count} -news={s.negative_news_count} "
                f"trend={s.sentiment_trend}"
            )
            e = self.signals.event
            lines.append(
                f"Event Summary: score={e.event_score:.2f} "
                f"earnings={'YES' if e.earnings_impact_flag else 'no'} "
                f"risk={e.event_risk_level}"
            )

        # RAG: similar past setups
        if self.similar_past_setups:
            lines.append(f"Similar Historical Setups ({len(self.similar_past_setups)} matches):")
            for i, setup in enumerate(self.similar_past_setups[:5], 1):
                ret = setup.get("return_5d")
                ret_str = f"{ret:+.1%}" if ret is not None else "pending"
                lines.append(
                    f"  {i}. {setup.get('symbol', '?')} ({setup.get('date', '?')}) "
                    f"ΓåÆ {ret_str} [{setup.get('regime', '?')}]"
                )

        # Memory: past trades for this symbol
        if self.past_trades_this_symbol:
            lines.append(f"Past Trades on {self.symbol}:")
            for trade in self.past_trades_this_symbol[:5]:
                lines.append(
                    f"  - {trade.get('decision', '?')} @ Γé╣{trade.get('entry_price', 0):.0f} "
                    f"ΓåÆ {trade.get('pnl', 'pending')} ({trade.get('notes', '')})"
                )

        # Memory: mistake warnings
        if self.mistake_warnings:
            lines.append("Mistake Warnings:")
            for warning in self.mistake_warnings[:3]:
                lines.append(f"  ΓÜá {warning}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Future Prediction Signals
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HorizonPrediction:
    """A single time-horizon price prediction."""
    horizon: str                          # "1_week" | "1_month" | "1_quarter" | "1_year"
    horizon_label: str                    # "1 Week" | "1 Month" | "1 Quarter" | "1 Year"
    predicted_price: float                # Expected price at horizon end
    predicted_change_pct: float           # Expected % change from current price
    confidence: float                     # 0-1 confidence in this prediction
    direction: str                        # "bullish" | "bearish" | "neutral"
    key_drivers: List[str] = field(default_factory=list)   # Top reasons
    risks: List[str] = field(default_factory=list)         # Key risks to this prediction
    reasoning: str = ""                   # Detailed rationale

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon": self.horizon,
            "horizon_label": self.horizon_label,
            "predicted_price": round(self.predicted_price, 2),
            "predicted_change_pct": round(self.predicted_change_pct, 2),
            "confidence": round(self.confidence, 4),
            "direction": self.direction,
            "key_drivers": self.key_drivers,
            "risks": self.risks,
            "reasoning": self.reasoning,
        }


@dataclass(slots=True)
class FuturePrediction:
    """Complete future prediction across all 4 horizons."""
    symbol: str
    date: str
    current_price: float
    market_cap: Optional[str] = None
    sector: Optional[str] = None
    one_week: Optional[HorizonPrediction] = None
    one_month: Optional[HorizonPrediction] = None
    one_quarter: Optional[HorizonPrediction] = None
    one_year: Optional[HorizonPrediction] = None
    overall_outlook: str = ""             # "bullish" | "bearish" | "neutral"
    overall_confidence: float = 0.0
    summary: str = ""                     # 2-3 sentence executive summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "date": self.date,
            "current_price": round(self.current_price, 2),
            "market_cap": self.market_cap,
            "sector": self.sector,
            "one_week": self.one_week.to_dict() if self.one_week else None,
            "one_month": self.one_month.to_dict() if self.one_month else None,
            "one_quarter": self.one_quarter.to_dict() if self.one_quarter else None,
            "one_year": self.one_year.to_dict() if self.one_year else None,
            "overall_outlook": self.overall_outlook,
            "overall_confidence": round(self.overall_confidence, 4),
            "summary": self.summary,
        }
