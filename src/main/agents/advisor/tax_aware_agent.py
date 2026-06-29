"""TaxAwareAgent — India-focused tax impact notes before TRIM/EXIT actions."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

class TaxAwareAgent:
    name = "tax_aware_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = (config or {}).get("tax_in", {}) or {}
        self._stcg_pct = float(cfg.get("stcg_equity_pct", 20))
        self._ltcg_pct = float(cfg.get("ltcg_equity_pct", 12.5))
        self._ltcg_exempt_lakh = float(cfg.get("ltcg_exemption_lakh", 1.25))
        self._buffer_days = int(cfg.get("hold_to_ltcg_buffer_days", 30))

    def tax_note_for_equity(
        self,
        purchase_date: Optional[str],
        avg_cost_inr: float,
        current_price_inr: float,
        quantity: float,
    ) -> Optional[str]:
        if not purchase_date:
            return None
        try:
            bought = datetime.fromisoformat(purchase_date)
        except ValueError:
            return None

        held_days = (datetime.utcnow() - bought).days
        gain_inr = (current_price_inr - avg_cost_inr) * quantity
        if gain_inr <= 0:
            return f"loss position (~₹{gain_inr:,.0f}); selling could realize a write-off"

        if held_days < 365:
            days_to_ltcg = 365 - held_days
            base = f"STCG @ {self._stcg_pct:.1f}% on ~₹{gain_inr:,.0f}"
            if days_to_ltcg <= self._buffer_days:
                base += f"; wait {days_to_ltcg}d for LTCG @ {self._ltcg_pct:.1f}%"
            return base

        ltcg_inr = gain_inr
        exempt_inr = self._ltcg_exempt_lakh * 100000
        taxable = max(0.0, ltcg_inr - exempt_inr)
        return (
            f"LTCG @ {self._ltcg_pct:.1f}% on ~₹{taxable:,.0f}"
            f" (₹{exempt_inr:,.0f} exempt this FY)"
        )
