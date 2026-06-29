"""GoalTrackerAgent — maps current portfolio value to user goals."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

class GoalTrackerAgent:
    name = "goal_tracker_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}

    def evaluate(
        self,
        goals: List[Dict[str, Any]],
        current_portfolio_value_inr: float,
        assumed_cagr_pct: float = 11.0,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        today = datetime.utcnow().year
        for g in goals or []:
            target = float(g.get("target_amount_inr", 0))
            year = int(g.get("target_year", today + 1))
            years = max(1, year - today)
            growth = (1 + assumed_cagr_pct / 100.0) ** years
            projected = current_portfolio_value_inr * growth
            gap = target - projected
            on_track = projected >= target * 0.95
            monthly_needed = 0.0
            if gap > 0:
                # PMT formula for a future value annuity at monthly rate.
                r_m = (assumed_cagr_pct / 100.0) / 12.0
                n = years * 12
                if r_m > 0:
                    monthly_needed = gap * r_m / ((1 + r_m) ** n - 1)
                else:
                    monthly_needed = gap / n
            out.append({
                "goal": g.get("name", "Goal"),
                "target_inr": target,
                "target_year": year,
                "projected_inr": round(projected, 0),
                "gap_inr": round(gap, 0),
                "on_track": on_track,
                "extra_monthly_sip_inr": round(max(0.0, monthly_needed), 0),
            })
        return out
