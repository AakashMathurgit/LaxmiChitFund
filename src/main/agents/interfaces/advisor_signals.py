"""Structured signal dataclasses for the long-horizon advisor pipeline.

Kept separate from `signals.py` so the swing-trading signals remain untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HoldingVerdict(str, Enum):
    ACCUMULATE = "accumulate"
    HOLD = "hold"
    TRIM = "trim"
    EXIT = "exit"
    AVOID = "avoid"
    BUY_NEW = "buy_new"


class AssetClass(str, Enum):
    US_EQUITY = "us_equity"
    IN_EQUITY = "in_equity"
    MF_EQUITY = "mf_equity"
    MF_DEBT = "mf_debt"
    ETF = "etf"
    GOLD = "gold"
    CASH = "cash"


class ConvictionLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


# ---------------------------------------------------------------------------
# Smart-money / Pro-trader signal
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProTraderHolding:
    """One investor's position in a single symbol at a point in time."""
    investor: str
    region: str                       # "US" | "IN" | "MF"
    shares: Optional[float] = None
    market_value_usd: Optional[float] = None
    position_pct_of_portfolio: Optional[float] = None
    qoq_change_pct: Optional[float] = None   # quarter-over-quarter change
    first_seen_quarter: Optional[str] = None  # e.g. "2016Q1"
    last_action: Optional[str] = None         # "ADD" | "TRIM" | "EXIT" | "NEW" | "HOLD"
    rationale_excerpt: Optional[str] = None   # from letter / interview


@dataclass(slots=True)
class SmartMoneySignal:
    """Aggregated smart-money view for one symbol."""
    symbol: str
    smart_money_score: float           # 0..1 (higher = stronger conviction across investors)
    consensus_verdict: str             # "STRONG_HOLD" | "ACCUMULATING" | "DISTRIBUTING" | "MIXED" | "NEUTRAL"
    holders_count: int
    total_position_value_usd: Optional[float] = None
    recent_buyers: List[str] = field(default_factory=list)
    recent_sellers: List[str] = field(default_factory=list)
    holders: List[ProTraderHolding] = field(default_factory=list)
    consensus_thesis: Optional[str] = None
    contrarian_flag: bool = False
    concentration_score: float = 0.0   # how concentrated this position is in their portfolios

    def to_feature_dict(self) -> Dict[str, Any]:
        verdict_map = {
            "STRONG_HOLD": 0.85,
            "ACCUMULATING": 1.0,
            "DISTRIBUTING": 0.15,
            "MIXED": 0.5,
            "NEUTRAL": 0.5,
        }
        return {
            "smart_money_score": self.smart_money_score,
            "smart_money_verdict": verdict_map.get(self.consensus_verdict, 0.5),
            "smart_money_holders": float(self.holders_count),
            "smart_money_concentration": self.concentration_score,
            "smart_money_contrarian": 1.0 if self.contrarian_flag else 0.0,
        }


# ---------------------------------------------------------------------------
# Holding-level recommendation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class HoldingRecommendation:
    symbol: str
    asset_class: AssetClass
    current_weight_pct: float
    target_weight_pct: float
    verdict: HoldingVerdict
    conviction: ConvictionLevel
    horizon_months: int
    thesis: str
    key_risks: List[str] = field(default_factory=list)
    tax_note: Optional[str] = None
    expected_action_amount: Optional[float] = None   # in user's base currency
    smart_money_alignment: Optional[str] = None      # "aligned" | "contrarian" | "n/a"
    smart_money_signal: Optional[SmartMoneySignal] = None


# ---------------------------------------------------------------------------
# Mutual fund signal
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MutualFundSignal:
    scheme_code: str
    scheme_name: str
    category: str                          # e.g. "Flexi Cap", "Mid Cap", "Debt - Short Duration"
    expense_ratio: Optional[float] = None
    aum_cr: Optional[float] = None
    rolling_returns_3y_pct: Optional[float] = None
    rolling_returns_5y_pct: Optional[float] = None
    category_percentile_3y: Optional[float] = None    # 0..100, higher = better
    manager_tenure_years: Optional[float] = None
    sharpe_3y: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    quality_score: float = 0.5             # 0..1 composite
    verdict: HoldingVerdict = HoldingVerdict.HOLD


# ---------------------------------------------------------------------------
# Asset allocation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AllocationTarget:
    asset_class: AssetClass
    target_pct: float
    current_pct: float
    drift_pct: float
    rebalance_action: str                  # "increase" | "decrease" | "hold"


@dataclass(slots=True)
class AssetAllocationPlan:
    targets: List[AllocationTarget]
    total_drift_pct: float
    requires_rebalance: bool
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Final advisor bundle
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AdvisorReport:
    run_id: str
    generated_at: str
    regime_us: Optional[str] = None
    regime_in: Optional[str] = None
    allocation_plan: Optional[AssetAllocationPlan] = None
    holdings: List[HoldingRecommendation] = field(default_factory=list)
    new_ideas: List[HoldingRecommendation] = field(default_factory=list)
    mutual_funds: List[MutualFundSignal] = field(default_factory=list)
    sip_plan: List[Dict[str, Any]] = field(default_factory=list)
    goal_progress: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""
