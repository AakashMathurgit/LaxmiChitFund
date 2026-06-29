"""AssetAllocatorAgent — top-down target allocation + drift detection."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces.advisor_signals import (
    AllocationTarget,
    AssetAllocationPlan,
    AssetClass,
)


class AssetAllocatorAgent:
    name = "asset_allocator_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._default_targets: Dict[str, float] = (
            self._config.get("default_allocation") or {}
        )
        self._drift_threshold = float(
            self._config.get("rebalance_drift_threshold_pct", 5)
        )

    def build_plan(
        self,
        current_pct_by_class: Dict[str, float],
        target_override: Optional[Dict[str, float]] = None,
    ) -> AssetAllocationPlan:
        targets_pct = target_override or self._default_targets or {}
        out: List[AllocationTarget] = []
        total_drift = 0.0
        for key, target in targets_pct.items():
            current = float(current_pct_by_class.get(key, 0.0))
            drift = current - float(target)
            total_drift += abs(drift)
            if abs(drift) < 1:
                action = "hold"
            elif drift < 0:
                action = "increase"
            else:
                action = "decrease"
            try:
                ac = AssetClass(key)
            except ValueError:
                continue
            out.append(AllocationTarget(
                asset_class=ac,
                target_pct=float(target),
                current_pct=round(current, 2),
                drift_pct=round(drift, 2),
                rebalance_action=action,
            ))

        return AssetAllocationPlan(
            targets=out,
            total_drift_pct=round(total_drift, 2),
            requires_rebalance=any(
                abs(t.drift_pct) >= self._drift_threshold for t in out
            ),
        )
