"""AdvisorOrchestrator — long-horizon (3-month+) financial advisor pipeline.

Reuses LCF agents where useful (regime, fundamental, sentiment, event,
bull/bear/debate/judge) and layers in advisor-specific agents:
  - ProTraderPortfolioAnalyzer (smart-money signal)
  - PortfolioAdvisorAgent
  - AssetAllocatorAgent
  - MutualFundAgent
  - SIPPlannerAgent
  - TaxAwareAgent
  - GoalTrackerAgent

This module deliberately AVOIDS coupling to the swing-trade
PipelineOrchestrator. It calls the existing agents directly so the
intraday trade planner / position manager stay out of the way.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..utils.logger import get_logger
from ..main.agents.adapters.llm_adapter import LLMAdapter
from ..main.agents.advisor.asset_allocator_agent import AssetAllocatorAgent
from ..main.agents.advisor.goal_tracker_agent import GoalTrackerAgent
from ..main.agents.advisor.mutual_fund_agent import MutualFundAgent
from ..main.agents.advisor.portfolio_advisor_agent import PortfolioAdvisorAgent
from ..main.agents.advisor.pro_trader_portfolio_analyzer import ProTraderPortfolioAnalyzer
from ..main.agents.advisor.sip_planner_agent import SIPPlannerAgent
from ..main.agents.advisor.tax_aware_agent import TaxAwareAgent
from ..main.agents.fundamental_agent import FundamentalAgent
from ..main.agents.regime_detector_agent import RegimeDetectorAgent
from ..main.agents.interfaces.advisor_signals import (
    AdvisorReport,
    MutualFundSignal,
)
from ..data.portfolio_loader import Holding, UnifiedPortfolio, load_unified_portfolio
from ..data import market_data


class AdvisorOrchestrator:
    """Long-horizon financial advisor pipeline."""

    def __init__(
        self,
        profile_path: str = "configs/investor_profile.yaml",
        tuning_path: str = "configs/long_term_investor.yaml",
    ):
        self.logger = get_logger("AdvisorOrchestrator")
        self._base = self._repo_root()
        self.profile = self._load_yaml(self._resolve(profile_path))
        self.tuning = self._load_yaml(self._resolve(tuning_path))
        self._llm = self._build_llm()

        # Agents
        self.pro_traders = ProTraderPortfolioAnalyzer(self.tuning, llm_adapter=self._llm)
        self.portfolio_advisor = PortfolioAdvisorAgent(self.tuning)
        self.allocator = AssetAllocatorAgent(self.tuning)
        self.mf_agent = MutualFundAgent(self.tuning)
        self.sip_planner = SIPPlannerAgent(self.tuning)
        self.tax_agent = TaxAwareAgent(self.tuning)
        self.goal_tracker = GoalTrackerAgent(self.tuning)
        self.fundamental = FundamentalAgent(self.tuning)
        self.regime = RegimeDetectorAgent()

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------

    def run(self, cadence: str = "monthly") -> AdvisorReport:
        run_id = str(uuid.uuid4())
        self.logger.info(f"Advisor run {run_id} (cadence={cadence})")
        portfolio = load_unified_portfolio(self.profile)
        self.logger.info(
            f"Portfolio loaded: {portfolio.total_holdings()} holdings, "
            f"cash={portfolio.cash_by_currency}"
        )

        # Live prices + FX (cost-basis fallback if offline)
        fx_usd_inr = market_data.get_fx("USDINR=X") or 83.0
        prices = self._fetch_prices(portfolio)

        # Market regime (US + IN, computed once per run)
        regime_us = self._detect_regime("^GSPC")
        regime_in = self._detect_regime("^NSEI", vix_symbol="^INDIAVIX")
        self.logger.info(f"Regimes — US: {regime_us}, IN: {regime_in}")

        weights = self._weights_by_asset_class(portfolio, prices, fx_usd_inr)
        target_alloc = self.profile.get("target_allocation") or (
            self.tuning.get("default_allocation") or {}
        )
        plan = self.allocator.build_plan(weights, target_alloc)

        # Per-holding evaluation (equities + ETFs)
        holding_recos = []
        for h in portfolio.holdings:
            if h.asset_class not in ("us_equity", "in_equity", "etf"):
                continue
            region = "US" if h.currency == "USD" else "IN"
            smart_money = self.pro_traders.analyze_symbol(h.symbol, region_hint=region)
            fundamental_score = self._fundamental_score(h.symbol)
            current_weight = self._weight_of_holding(h, portfolio, prices, fx_usd_inr)
            target_weight = self._target_weight_for(h, target_alloc)
            tax_note = self._tax_note(h, prices)
            regime_label = regime_us if region == "US" else regime_in
            reco = self.portfolio_advisor.evaluate_holding(
                holding=h,
                fundamental_score=fundamental_score,
                smart_money=smart_money,
                regime_label=regime_label,
                current_weight_pct=current_weight,
                target_weight_pct=target_weight,
                tax_note=tax_note,
            )
            holding_recos.append(reco)

        # Mutual funds
        mf_signals: List[MutualFundSignal] = []
        for h in portfolio.holdings:
            if h.asset_class in ("mf_equity", "mf_debt") and h.scheme_code:
                mf_signals.append(
                    self.mf_agent.evaluate(h.scheme_code, h.category)
                )

        # SIP plan
        monthly_investable = float(
            self.profile.get("investor", {}).get("monthly_investable_inr", 0)
        )
        sip_plan = self.sip_planner.plan(monthly_investable, mf_signals)

        # Goal tracking
        total_value_inr = self._estimate_total_value_inr(portfolio, prices, fx_usd_inr)
        goal_progress = self.goal_tracker.evaluate(
            self.profile.get("goals", []) or [],
            total_value_inr,
        )

        report = AdvisorReport(
            run_id=run_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            regime_us=regime_us,
            regime_in=regime_in,
            allocation_plan=plan,
            holdings=holding_recos,
            mutual_funds=mf_signals,
            sip_plan=sip_plan,
            goal_progress=goal_progress,
            summary=self._summarize(holding_recos, mf_signals, plan),
        )
        self._write_report(report)
        return report

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _repo_root() -> str:
        # .../LCF/src/pipeline/advisor_orchestrator.py -> .../LCF
        return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def _resolve(self, rel: str) -> str:
        return rel if os.path.isabs(rel) else os.path.join(self._base, rel)

    @staticmethod
    def _load_yaml(path: str) -> Dict[str, Any]:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}

    def _build_llm(self) -> Optional[LLMAdapter]:
        """Construct an LLMAdapter from config.yaml. Returns None on failure.

        Used by ProTraderPortfolioAnalyzer to extract investment rationale
        from cached investor letters.
        """
        cfg_path = self._resolve("config.yaml")
        if not os.path.exists(cfg_path):
            return None
        try:
            cfg = self._load_yaml(cfg_path)
            if not cfg.get("llm"):
                return None
            return LLMAdapter.from_config(cfg, base_path=self._base)
        except Exception as exc:
            self.logger.warning(f"LLM adapter unavailable: {exc}")
            return None

    @staticmethod
    def _weight_of_holding(
        h: Holding,
        portfolio: UnifiedPortfolio,
        prices: Dict[str, float],
        fx_usd_inr: float,
    ) -> float:
        total = AdvisorOrchestrator._portfolio_value_in_inr(portfolio, prices, fx_usd_inr) or 1.0
        value = AdvisorOrchestrator._holding_value_in_inr(h, prices, fx_usd_inr)
        return value / total * 100.0

    @staticmethod
    def _weights_by_asset_class(
        portfolio: UnifiedPortfolio,
        prices: Dict[str, float],
        fx_usd_inr: float,
    ) -> Dict[str, float]:
        total = AdvisorOrchestrator._portfolio_value_in_inr(portfolio, prices, fx_usd_inr) or 1.0
        out: Dict[str, float] = {}
        for h in portfolio.holdings:
            v = AdvisorOrchestrator._holding_value_in_inr(h, prices, fx_usd_inr)
            out[h.asset_class] = out.get(h.asset_class, 0.0) + v / total * 100.0
        return out

    @staticmethod
    def _target_weight_for(h: Holding, target_alloc: Dict[str, float]) -> float:
        return float(target_alloc.get(h.asset_class, 0.0))

    @staticmethod
    def _estimate_total_value_inr(
        portfolio: UnifiedPortfolio,
        prices: Dict[str, float],
        fx_usd_inr: float,
    ) -> float:
        total = AdvisorOrchestrator._portfolio_value_in_inr(portfolio, prices, fx_usd_inr)
        for cur, amt in portfolio.cash_by_currency.items():
            total += amt * fx_usd_inr if cur == "USD" else amt
        return total

    # ---- value helpers (price-aware) ----
    @staticmethod
    def _holding_value_in_inr(h: Holding, prices: Dict[str, float], fx_usd_inr: float) -> float:
        price = prices.get(h.symbol) or h.avg_cost
        value = h.quantity * price
        return value * fx_usd_inr if h.currency == "USD" else value

    @staticmethod
    def _portfolio_value_in_inr(
        portfolio: UnifiedPortfolio,
        prices: Dict[str, float],
        fx_usd_inr: float,
    ) -> float:
        return sum(
            AdvisorOrchestrator._holding_value_in_inr(h, prices, fx_usd_inr)
            for h in portfolio.holdings
        )

    # ---- enrichment helpers ----
    def _fetch_prices(self, portfolio: UnifiedPortfolio) -> Dict[str, float]:
        prices: Dict[str, float] = {}
        for h in portfolio.holdings:
            if h.asset_class in ("us_equity", "in_equity", "etf"):
                p = market_data.get_quote(h.symbol)
                if p is not None:
                    prices[h.symbol] = p
        return prices

    def _detect_regime(self, index_symbol: str, vix_symbol: Optional[str] = None) -> Optional[str]:
        ohlc = market_data.get_index_ohlc(index_symbol, period="1y")
        if not ohlc:
            return None
        vix = market_data.get_vix(vix_symbol) if vix_symbol else None
        try:
            signal, _ = self.regime.detect_regime(ohlc, vix)
            return signal.market_regime.value
        except Exception as exc:
            self.logger.warning(f"Regime detection failed for {index_symbol}: {exc}")
            return None

    def _fundamental_score(self, symbol: str) -> float:
        data = market_data.get_fundamentals(symbol)
        if not data:
            return 0.5
        try:
            signal = self.fundamental.analyse(data)
            return float(signal.fundamental_score)
        except Exception:
            return 0.5

    def _tax_note(self, h: Holding, prices: Dict[str, float]) -> Optional[str]:
        if h.asset_class not in ("in_equity",):  # India tax rules only
            return None
        price = prices.get(h.symbol)
        if price is None:
            return None
        return self.tax_agent.tax_note_for_equity(
            purchase_date=h.purchase_date,
            avg_cost_inr=h.avg_cost,
            current_price_inr=price,
            quantity=h.quantity,
        )

    @staticmethod
    def _summarize(holdings, funds, plan) -> str:
        n_acc = sum(1 for h in holdings if h.verdict.value == "accumulate")
        n_trim = sum(1 for h in holdings if h.verdict.value in ("trim", "exit"))
        return (
            f"{len(holdings)} stocks reviewed: {n_acc} accumulate, {n_trim} trim/exit. "
            f"{len(funds)} MFs scored. Rebalance needed: {plan.requires_rebalance if plan else 'n/a'}."
        )

    def _write_report(self, report: AdvisorReport) -> None:
        out_dir = self._resolve(
            (self.tuning.get("reporting", {}) or {}).get("output_dir", "data/advisor_reports")
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(out_dir, f"advisor_{ts}.json")
        md_path = os.path.join(out_dir, f"advisor_{ts}.md")

        # JSON
        def _enc(o: Any) -> Any:
            if hasattr(o, "__dict__"):
                return o.__dict__
            if hasattr(o, "value"):
                return o.value
            return str(o)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=_enc)

        # Markdown
        lines = [
            f"# Advisor Report — {report.generated_at}",
            "",
            f"**Summary:** {report.summary}",
            "",
            f"**Market Regime** — US: `{report.regime_us or 'n/a'}` · IN: `{report.regime_in or 'n/a'}`",
            "",
        ]
        if report.allocation_plan:
            lines += ["## Asset Allocation", ""]
            lines.append("| Class | Target % | Current % | Drift % | Action |")
            lines.append("|---|---:|---:|---:|---|")
            for t in report.allocation_plan.targets:
                lines.append(
                    f"| {t.asset_class.value} | {t.target_pct:.1f} | "
                    f"{t.current_pct:.1f} | {t.drift_pct:+.1f} | {t.rebalance_action} |"
                )
            lines.append("")
        if report.holdings:
            lines += ["## Holdings", ""]
            lines.append("| Symbol | Verdict | Conviction | Smart Money | Tax | Thesis |")
            lines.append("|---|---|---|---|---|---|")
            for h in report.holdings:
                lines.append(
                    f"| {h.symbol} | {h.verdict.value} | {h.conviction.value} | "
                    f"{h.smart_money_alignment or ''} | {h.tax_note or ''} | {h.thesis} |"
                )
            lines.append("")

            # Smart-money detail: who's doing what across each holding.
            sm_rows: List[str] = []
            for h in report.holdings:
                sig = h.smart_money_signal
                if not sig or not sig.holders:
                    continue
                for holder in sig.holders:
                    change = (
                        f"{holder.qoq_change_pct:+.1f}%"
                        if holder.qoq_change_pct is not None else "—"
                    )
                    shares = (
                        f"{int(holder.shares):,}" if holder.shares else "—"
                    )
                    value = (
                        f"${holder.market_value_usd/1e6:,.1f}M"
                        if holder.market_value_usd else "—"
                    )
                    sm_rows.append(
                        f"| {h.symbol} | {holder.investor} | {holder.last_action or '—'} | "
                        f"{change} | {shares} | {value} | {holder.first_seen_quarter or '—'} |"
                    )
            if sm_rows:
                lines += ["## Smart Money Detail", ""]
                lines.append("| Stock | Investor | Action | QoQ Δ | Shares | Value | As of |")
                lines.append("|---|---|---|---:|---:|---:|---|")
                lines.extend(sm_rows)
                lines.append("")

            # Rationales — short LLM-extracted "why" per holding/investor.
            rationale_rows: List[str] = []
            for h in report.holdings:
                sig = h.smart_money_signal
                if not sig:
                    continue
                seen = set()
                for holder in sig.holders:
                    if not holder.rationale_excerpt:
                        continue
                    key = (h.symbol, holder.investor)
                    if key in seen:
                        continue
                    seen.add(key)
                    excerpt = holder.rationale_excerpt.replace("|", "\\|")
                    rationale_rows.append(
                        f"| {h.symbol} | {holder.investor} | {excerpt} |"
                    )
            if rationale_rows:
                lines += ["## Smart Money Rationales", ""]
                lines.append("| Stock | Investor | Rationale (from letter) |")
                lines.append("|---|---|---|")
                lines.extend(rationale_rows)
                lines.append("")
        if report.mutual_funds:
            lines += ["## Mutual Funds", ""]
            lines.append("| Scheme | Category | 3y % | 5y % | Quality | Verdict |")
            lines.append("|---|---|---:|---:|---:|---|")
            for f in report.mutual_funds:
                lines.append(
                    f"| {f.scheme_name} | {f.category} | "
                    f"{f.rolling_returns_3y_pct or '-'} | {f.rolling_returns_5y_pct or '-'} | "
                    f"{f.quality_score:.2f} | {f.verdict.value} |"
                )
            lines.append("")
        if report.sip_plan:
            lines += ["## SIP Plan (next month)", ""]
            lines.append("| Scheme | Amount (INR) | Rationale |")
            lines.append("|---|---:|---|")
            for s in report.sip_plan:
                lines.append(f"| {s['scheme_name']} | {s['monthly_sip_inr']:.0f} | {s['rationale']} |")
            lines.append("")
        if report.goal_progress:
            lines += ["## Goals", ""]
            lines.append("| Goal | Target Year | Target | Projected | Gap | Extra Monthly SIP |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for g in report.goal_progress:
                lines.append(
                    f"| {g['goal']} | {g['target_year']} | ₹{g['target_inr']:,.0f} | "
                    f"₹{g['projected_inr']:,.0f} | ₹{g['gap_inr']:,.0f} | "
                    f"₹{g['extra_monthly_sip_inr']:,.0f} |"
                )
            lines.append("")

        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        self.logger.info(f"Report written: {md_path}")
        self.logger.info(f"Report JSON: {json_path}")
