"""Fundamental Agent — valuation, growth, and financial-health signals.

Reads real Yahoo Finance fields from StockDataContext:
  pe_ratio, forward_pe, eps, revenue_growth, profit_margin,
  debt_to_equity, roe, market_cap, sector.

Outputs: valuation_label, growth_score, financial_health_score, fundamental_score.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import FundamentalSignal

if TYPE_CHECKING:
    from ..controllers.data_context import StockDataContext


class FundamentalAgent(Agent):
    """Computes lightweight fundamental signals relevant to 3–10 day swings."""

    name = "fundamental_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    # ------------------------------------------------------------------
    # Sub-scores
    # ------------------------------------------------------------------

    def _valuation_label(self, data: Dict[str, Any]) -> str:
        """Classify PE ratio into undervalued / fair / overvalued."""
        pe = self._safe(data.get("pe_ratio"), -1.0)
        if pe <= 0:
            return "fair"   # negative / missing PE — can't classify cleanly
        if pe < 12:
            return "undervalued"
        if pe <= 25:
            return "fair"
        return "overvalued"

    def _growth_score(self, data: Dict[str, Any]) -> float:
        """Map revenue_growth (fraction, e.g. 0.15 = 15%) to [0, 1].

        Scale: -10% → 0.0, 0% → 0.2, +40% → 1.0.
        """
        rev_growth = self._safe(data.get("revenue_growth"), 0.0)
        score = (rev_growth * 100.0 + 10.0) / 50.0
        return round(self._clamp(score), 4)

    def _financial_health_score(self, data: Dict[str, Any]) -> float:
        """Composite of profit margin, debt/equity, and ROE → [0, 1]."""
        profit_margin = self._safe(data.get("profit_margin"), 0.0)
        debt_to_equity = self._safe(data.get("debt_to_equity"), 1.0)
        roe = self._safe(data.get("roe"), 0.0)

        # Profit margin: 0%–30% maps to 0–1
        margin_score = self._clamp(profit_margin / 0.30)
        # D/E: lower is better; 0 → 1.0, ≥ 2 → 0.0
        de_score = self._clamp(1.0 - debt_to_equity / 2.0)
        # ROE: 0%–30% maps to 0–1
        roe_score = self._clamp(roe / 0.30)

        return round(0.40 * margin_score + 0.30 * de_score + 0.30 * roe_score, 4)

    def _fundamental_score(
        self,
        valuation_label: str,
        growth_score: float,
        health_score: float,
    ) -> float:
        """Weighted composite of valuation, growth, and health."""
        valuation_map = {"undervalued": 1.0, "fair": 0.6, "overvalued": 0.2}
        v = valuation_map.get(valuation_label, 0.5)
        return round(0.35 * v + 0.35 * growth_score + 0.30 * health_score, 4)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data: Dict[str, Any]) -> FundamentalSignal:
        val_label = self._valuation_label(data)
        growth = self._growth_score(data)
        health = self._financial_health_score(data)
        score = self._fundamental_score(val_label, growth, health)

        return FundamentalSignal(
            fundamental_score=score,
            valuation_label=val_label,
            growth_score=growth,
            financial_health_score=health,
            pe_ratio=data.get("pe_ratio"),
            forward_pe=data.get("forward_pe"),
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            # Try to get StockDataContext from kwargs first
            stock_ctx: Optional["StockDataContext"] = kwargs.get("stock_context")
            
            if stock_ctx is not None:
                # Extract fundamental data from StockDataContext
                data = self._extract_from_context(stock_ctx)
            else:
                # Legacy: use input_data dict
                data = dict(ctx.input_data) if ctx.input_data else {}
                data.update(kwargs)

            signal = self.analyse(data)
            payload = {"signal": signal.to_feature_dict(), "raw_signal": signal}

            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[str(v) for v in signal.to_feature_dict().values()],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="FUNDAMENTAL_ERROR", message=str(exc))],
            )

    def _extract_from_context(self, stock_ctx: "StockDataContext") -> Dict[str, Any]:
        """Extract fundamental data fields from StockDataContext."""
        data: Dict[str, Any] = {}
        
        # Add stock identity
        data["symbol"] = stock_ctx.symbol
        data["company_name"] = stock_ctx.company_name
        data["sector"] = stock_ctx.sector
        data["industry"] = stock_ctx.industry
        
        # Extract from fundamentals dataclass
        if stock_ctx.fundamentals:
            f = stock_ctx.fundamentals
            data["pe_ratio"] = f.pe_ratio
            data["forward_pe"] = f.forward_pe
            data["eps"] = f.eps
            data["book_value"] = f.book_value
            data["revenue_growth"] = f.revenue_growth_yoy
            data["roe"] = f.return_on_equity
            data["market_cap"] = f.market_cap
            data["price_to_book"] = f.price_to_book
            data["debt_to_equity"] = f.debt_to_equity
            data["profit_margin"] = f.profit_margin
            data["dividend_yield"] = f.dividend_yield
            data["analyst_rating"] = f.analyst_rating
            data["price_target"] = f.price_target
        
        # Add price context
        data["last_close"] = stock_ctx.last_close
        data["previous_close"] = stock_ctx.previous_close
        
        return data
