"""Common Flow interface + standardized result types.

Every flow returns a FlowResult containing a normalized list of Decisions so the
combined store and cross-flow conflict checks can compare them regardless of
which pipeline produced them.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

_LCF_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class Decision:
    """A normalized per-symbol decision, comparable across flows."""
    symbol: str
    action: str                       # BUY | SELL | HOLD | ACCUMULATE | TRIM | EXIT ...
    confidence: float = 0.0
    horizon: str = ""                 # "intraday" | "swing" | "long_term" | ...
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "horizon": self.horizon,
            "detail": self.detail,
        }


@dataclass
class FlowResult:
    flow: str
    timestamp: str
    ok: bool = True
    decisions: List[Decision] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    artifacts: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow": self.flow,
            "timestamp": self.timestamp,
            "ok": self.ok,
            "decisions": [d.to_dict() for d in self.decisions],
            "summary": self.summary,
            "error": self.error,
            "artifacts": self.artifacts,
        }


class Flow(ABC):
    """Base class for all flows."""

    name: str = "flow"
    horizon: str = ""
    # Cadence is advisory metadata used by the scheduler.
    #   "Nm"  -> every N minutes (market-hours gated)
    #   "daily"   -> once per day after US close
    #   "weekly"  -> once per week
    #   "monthly" -> once per month
    cadence: str = "daily"
    market_hours_only: bool = False

    @abstractmethod
    def run(self, rt: Any, **opts) -> FlowResult:
        """Run the flow using the shared LCFRuntime `rt`."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers shared by subprocess-style adapters
    # ------------------------------------------------------------------

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _run_script(self, rel_path: str, args: List[str], timeout: int = 900,
                    env: Optional[Dict[str, str]] = None) -> int:
        """Run an existing entry script in a subprocess (isolation).

        `env` overrides are merged onto the current environment for that
        subprocess only — used to route a flow at a specific broker account
        (e.g. the swing flow gets the Account-2 Alpaca keys).
        """
        cmd = [sys.executable, os.path.join(_LCF_ROOT, rel_path), *args]
        proc_env = None
        if env:
            proc_env = os.environ.copy()
            proc_env.update({k: v for k, v in env.items() if v is not None})
        proc = subprocess.run(cmd, cwd=_LCF_ROOT, timeout=timeout, env=proc_env)
        return proc.returncode

    @staticmethod
    def _load_json(path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
