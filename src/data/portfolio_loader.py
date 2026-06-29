"""Unified loader for the user's holdings across US equities, IN equities, and MFs.

Reads the JSON files referenced in investor_profile.yaml and produces a single
in-memory portfolio object the advisor pipeline can reason over.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class Holding:
    symbol: str
    name: str
    asset_class: str            # "us_equity" | "in_equity" | "mf_equity" | "mf_debt" | "etf"
    quantity: float
    avg_cost: float
    currency: str
    purchase_date: Optional[str] = None
    sector: Optional[str] = None
    scheme_code: Optional[str] = None
    category: Optional[str] = None
    sip_amount: Optional[float] = None


@dataclass(slots=True)
class UnifiedPortfolio:
    as_of: str
    base_currency: str
    cash_by_currency: Dict[str, float] = field(default_factory=dict)
    holdings: List[Holding] = field(default_factory=list)

    def total_holdings(self) -> int:
        return len(self.holdings)

    def by_asset_class(self) -> Dict[str, List[Holding]]:
        out: Dict[str, List[Holding]] = {}
        for h in self.holdings:
            out.setdefault(h.asset_class, []).append(h)
        return out


def _read_json(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _classify_mf(category: Optional[str]) -> str:
    if not category:
        return "mf_equity"
    cat = category.lower()
    if "debt" in cat or "liquid" in cat or "bond" in cat or "gilt" in cat:
        return "mf_debt"
    return "mf_equity"


def load_unified_portfolio(profile: Dict[str, Any]) -> UnifiedPortfolio:
    """Load the three holdings files and merge into a UnifiedPortfolio."""
    base_currency = profile.get("investor", {}).get("base_currency", "INR")
    portfolio_cfg = profile.get("portfolio", {}) or {}

    cash: Dict[str, float] = {}
    holdings: List[Holding] = []
    as_of = ""

    in_data = _read_json(portfolio_cfg.get("in_equity_file", ""))
    if in_data:
        as_of = in_data.get("as_of", as_of)
        cash["INR"] = cash.get("INR", 0.0) + float(in_data.get("cash_inr", 0) or 0)
        for h in in_data.get("holdings", []):
            holdings.append(Holding(
                symbol=h.get("symbol", ""),
                name=h.get("name", h.get("symbol", "")),
                asset_class="in_equity",
                quantity=float(h.get("quantity") or 0),
                avg_cost=float(h.get("avg_cost_inr") or 0),
                currency="INR",
                purchase_date=h.get("purchase_date"),
                sector=h.get("sector"),
            ))

    us_data = _read_json(portfolio_cfg.get("us_equity_file", ""))
    if us_data:
        as_of = us_data.get("as_of", as_of)
        cash["USD"] = cash.get("USD", 0.0) + float(us_data.get("cash_usd", 0) or 0)
        for h in us_data.get("holdings", []):
            sector = h.get("sector") or ""
            asset_class = "etf" if sector.lower() == "etf" else "us_equity"
            holdings.append(Holding(
                symbol=h.get("symbol", ""),
                name=h.get("name", h.get("symbol", "")),
                asset_class=asset_class,
                quantity=float(h.get("quantity") or 0),
                avg_cost=float(h.get("avg_cost_usd") or 0),
                currency="USD",
                purchase_date=h.get("purchase_date"),
                sector=sector or None,
            ))

    mf_data = _read_json(portfolio_cfg.get("mutual_funds_file", ""))
    if mf_data:
        as_of = mf_data.get("as_of", as_of)
        for h in mf_data.get("holdings", []):
            category = h.get("category")
            holdings.append(Holding(
                symbol=h.get("scheme_code", ""),
                name=h.get("scheme_name", h.get("scheme_code", "")),
                asset_class=_classify_mf(category),
                quantity=float(h.get("units") or 0),
                avg_cost=float(h.get("avg_nav_inr") or 0),
                currency="INR",
                purchase_date=h.get("purchase_date"),
                scheme_code=h.get("scheme_code"),
                category=category,
                sip_amount=h.get("sip_amount_inr"),
            ))

    return UnifiedPortfolio(
        as_of=as_of or "",
        base_currency=base_currency,
        cash_by_currency=cash,
        holdings=holdings,
    )
