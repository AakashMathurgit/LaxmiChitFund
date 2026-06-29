"""Combined cross-flow result store + conflict detection.

Every flow run is appended to one store so you can see what all flows concluded
together. After a multi-flow sweep, `detect_conflicts` flags symbols where flows
disagree (e.g. intraday says SELL while the advisor wants to ACCUMULATE) so they
surface in the summary / notification instead of silently contradicting.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List

from .base import FlowResult

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_STORE_PATH = os.path.join(_ROOT, "data", "combined_results.jsonl")

# Normalize every flow's verbs onto a directional axis for comparison.
_BULLISH = {"BUY", "ACCUMULATE", "ADD", "BUY_NEW"}
_BEARISH = {"SELL", "EXIT", "TRIM", "AVOID"}


def _direction(action: str) -> str:
    a = (action or "").upper()
    if a in _BULLISH:
        return "bullish"
    if a in _BEARISH:
        return "bearish"
    return "neutral"


def record(results: List[FlowResult], store_path: str = _STORE_PATH) -> None:
    """Append a sweep's results to the combined store."""
    os.makedirs(os.path.dirname(store_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "flows": [r.to_dict() for r in results],
    }
    with open(store_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def detect_conflicts(results: List[FlowResult]) -> List[Dict[str, Any]]:
    """Find symbols where two flows point in opposite directions."""
    by_symbol: Dict[str, List[Dict[str, str]]] = {}
    for r in results:
        for d in r.decisions:
            if _direction(d.action) == "neutral":
                continue
            by_symbol.setdefault(d.symbol, []).append({
                "flow": r.flow, "action": d.action, "direction": _direction(d.action),
            })

    conflicts = []
    for symbol, calls in by_symbol.items():
        dirs = {c["direction"] for c in calls}
        if "bullish" in dirs and "bearish" in dirs:
            conflicts.append({"symbol": symbol, "calls": calls})
    return conflicts
