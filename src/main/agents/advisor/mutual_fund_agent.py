"""MutualFundAgent — quality assessment for Indian MF schemes via mfapi.in."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..interfaces.advisor_signals import HoldingVerdict, MutualFundSignal

from ....data.mf_data import compute_rolling_return_pct, fetch_scheme_history


class MutualFundAgent:
    name = "mutual_fund_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("mutual_funds", {}) or {}
        self._cache_dir = cfg.get("cache_dir", "data/pro_traders")
        self._rolling_years: List[int] = list(cfg.get("rolling_return_years", [3, 5]))
        self._min_aum_cr = float(cfg.get("min_aum_cr", 500))
        self._max_expense = float(cfg.get("max_expense_ratio_pct", 1.5))

    def evaluate(self, scheme_code: str, category_hint: Optional[str] = None) -> MutualFundSignal:
        data = fetch_scheme_history(scheme_code, self._cache_dir) or {}
        meta = data.get("meta", {}) or {}
        navs = data.get("data", []) or []

        roll_3y = compute_rolling_return_pct(navs, 3)
        roll_5y = compute_rolling_return_pct(navs, 5)

        # Heuristic quality score: returns + (eventually) expense/aum.
        return_score = 0.5
        if roll_3y is not None:
            return_score = max(0.0, min(1.0, (roll_3y - 6.0) / 18.0))  # 6% -> 0, 24% -> 1
        quality = round(return_score, 3)

        if quality >= 0.7:
            verdict = HoldingVerdict.ACCUMULATE
        elif quality >= 0.5:
            verdict = HoldingVerdict.HOLD
        elif quality >= 0.3:
            verdict = HoldingVerdict.TRIM
        else:
            verdict = HoldingVerdict.EXIT

        return MutualFundSignal(
            scheme_code=scheme_code,
            scheme_name=meta.get("scheme_name", scheme_code),
            category=meta.get("scheme_category", category_hint or "Unknown"),
            rolling_returns_3y_pct=roll_3y,
            rolling_returns_5y_pct=roll_5y,
            quality_score=quality,
            verdict=verdict,
        )
