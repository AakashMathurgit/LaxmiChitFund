"""SIPPlannerAgent — proposes monthly SIP allocation across MFs/ETFs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces.advisor_signals import MutualFundSignal


class SIPPlannerAgent:
    name = "sip_planner_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def plan(
        self,
        monthly_investable: float,
        fund_signals: List[MutualFundSignal],
        min_sip_inr: float = 500,
    ) -> List[Dict[str, Any]]:
        """Allocate SIP weighted by each fund's quality_score."""
        if monthly_investable <= 0 or not fund_signals:
            return []

        # Filter to funds we'd accumulate or hold.
        candidates = [f for f in fund_signals if f.quality_score >= 0.4]
        if not candidates:
            return []

        total_weight = sum(f.quality_score for f in candidates) or 1.0
        plan: List[Dict[str, Any]] = []
        for f in candidates:
            share = f.quality_score / total_weight
            amount = round(max(min_sip_inr, monthly_investable * share) / 100) * 100
            plan.append({
                "scheme_code": f.scheme_code,
                "scheme_name": f.scheme_name,
                "monthly_sip_inr": amount,
                "rationale": f"quality {f.quality_score:.2f} · 3y {f.rolling_returns_3y_pct}",
            })
        return plan
