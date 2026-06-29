"""PortfolioAdvisorAgent — per-holding HOLD/ADD/TRIM/EXIT verdicts.

Combines fundamental, smart-money, regime, and tax inputs into a single
HoldingRecommendation per existing position.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces.advisor_signals import (
    AssetClass,
    ConvictionLevel,
    HoldingRecommendation,
    HoldingVerdict,
    SmartMoneySignal,
)
from ....data.portfolio_loader import Holding


class PortfolioAdvisorAgent:
    name = "portfolio_advisor_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._horizon_months = int(
            self._config.get("horizon_months_default", 12)
        )

    def evaluate_holding(
        self,
        holding: Holding,
        fundamental_score: float,
        smart_money: Optional[SmartMoneySignal],
        regime_label: Optional[str],
        current_weight_pct: float,
        target_weight_pct: float,
        tax_note: Optional[str] = None,
    ) -> HoldingRecommendation:
        sm_score = smart_money.smart_money_score if smart_money else 0.5
        sm_verdict = smart_money.consensus_verdict if smart_money else "NEUTRAL"
        regime_bias = 0.1 if (regime_label or "").startswith("bull") else (
            -0.1 if (regime_label or "").startswith("bear") else 0.0
        )

        composite = (
            0.45 * fundamental_score
            + 0.30 * sm_score
            + 0.15 * (1.0 if sm_verdict == "ACCUMULATING" else 0.5)
            + 0.10 * (0.5 + regime_bias)
        )

        drift = current_weight_pct - target_weight_pct
        if composite >= 0.75 and drift < 2:
            verdict = HoldingVerdict.ACCUMULATE
            conviction = ConvictionLevel.HIGH
        elif composite >= 0.6:
            verdict = HoldingVerdict.HOLD
            conviction = ConvictionLevel.MEDIUM
        elif composite >= 0.4:
            verdict = HoldingVerdict.HOLD if drift <= 0 else HoldingVerdict.TRIM
            conviction = ConvictionLevel.LOW
        else:
            verdict = HoldingVerdict.EXIT if drift > 0 else HoldingVerdict.TRIM
            conviction = ConvictionLevel.MEDIUM

        thesis_bits = [
            f"composite {composite:.2f}",
            f"fundamentals {fundamental_score:.2f}",
            f"smart money {sm_verdict.lower()} ({sm_score:.2f})",
            f"regime {regime_label or 'unknown'}",
        ]

        return HoldingRecommendation(
            symbol=holding.symbol,
            asset_class=AssetClass(holding.asset_class) if holding.asset_class in AssetClass.__members__.values() else AssetClass.IN_EQUITY,
            current_weight_pct=round(current_weight_pct, 2),
            target_weight_pct=round(target_weight_pct, 2),
            verdict=verdict,
            conviction=conviction,
            horizon_months=self._horizon_months,
            thesis=" · ".join(thesis_bits),
            tax_note=tax_note,
            smart_money_alignment=(
                "aligned" if sm_verdict in ("ACCUMULATING", "STRONG_HOLD")
                else "contrarian" if sm_verdict == "DISTRIBUTING"
                else "n/a"
            ),
            smart_money_signal=smart_money,
        )
