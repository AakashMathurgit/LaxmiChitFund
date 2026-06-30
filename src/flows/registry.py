"""Flow registry — single source of truth for all runnable flows."""

from __future__ import annotations

from typing import Dict, List

from .base import Flow
from .adapters import (
    IntradayUSFlow,
    USSwingFlow,
    INDailyFlow,
    PredictUSFlow,
    AdvisorFlow,
)

# Note: us-daily (USDailyFlow) was superseded by us-swing — a 100-stock,
# prediction-driven funnel that trades a separate Alpaca account. The class
# still exists in adapters.py but is intentionally not scheduled here.
_FLOWS: Dict[str, Flow] = {
    f.name: f for f in [
        IntradayUSFlow(),
        USSwingFlow(),
        INDailyFlow(),
        PredictUSFlow(),
        AdvisorFlow(),
    ]
}


def all_flows() -> List[Flow]:
    return list(_FLOWS.values())


def get_flow(name: str) -> Flow:
    if name not in _FLOWS:
        raise KeyError(f"Unknown flow '{name}'. Known: {', '.join(_FLOWS)}")
    return _FLOWS[name]
