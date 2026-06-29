"""ProTraderPortfolioAnalyzer — smart-money signal for a symbol.

Aggregates evidence from:
  - SEC 13F filings (US institutional holdings, quarterly)
  - NSE bulk deals (Indian large transactions, daily)
  - AMFI monthly portfolio disclosures (Indian MF managers)
  - Cached investor letters / news for the "why"

Produces a SmartMoneySignal that plugs into bull/bear/judge agents and
feeds the holding-level recommendation.

Reuses the existing LCF Agent base class (signature compatible with
the rest of the pipeline).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from ..interfaces.agent import Agent, AgentContext, AgentResult, AgentError, TimingInfo
from ..interfaces.advisor_signals import SmartMoneySignal, ProTraderHolding

from ....data.pro_trader_sources import (
    BulkDeal,
    ThirteenFHolding,
    ThirteenFSnapshot,
    fetch_13f_history,
    fetch_13f_latest,
    fetch_letter_text,
    fetch_nse_bulk_deals,
    filter_bulk_deals_by_investor,
    load_letters_index,
    load_watchlist,
    ticker_to_cusip,
)

try:
    from ....data import market_data  # type: ignore
except Exception:  # pragma: no cover
    market_data = None  # type: ignore


class ProTraderPortfolioAnalyzer(Agent):
    """Aggregates smart-money positioning for one symbol."""

    name = "pro_trader_portfolio_analyzer"

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_adapter: Optional[Any] = None):
        super().__init__(name=self.name)
        cfg = config or {}
        sm_cfg = cfg.get("smart_money", {}) if cfg else {}
        self._watchlist_path = sm_cfg.get("watchlist_path", "data/pro_traders/watchlist.yaml")
        self._cache_dir = sm_cfg.get("cache_dir", "data/pro_traders")
        self._refresh = sm_cfg.get("refresh_days", {}) or {}
        self._min_consensus = int(sm_cfg.get("min_consensus_holders", 2))
        self._watchlist = load_watchlist(self._watchlist_path)
        agg = self._watchlist.get("aggregation", {}) or {}
        self._region_weights = (
            agg.get("region_weights")
            or {"US": 1.0, "IN": 1.0, "MF": 0.6}
        )
        self._decay_quarters = float(agg.get("recency_decay_quarters", 4))
        self._contrarian_threshold = float(agg.get("contrarian_threshold_pct", 30))

        # Per-investor reputation weights, indexed by name.
        self._investor_weights: Dict[str, float] = {}
        for group in ("us_investors", "in_investors", "mutual_fund_managers"):
            for inv in self._watchlist.get(group, []) or []:
                name = inv.get("name", "")
                if name:
                    self._investor_weights[name] = float(inv.get("weight", 1.0))

        # LLM rationale extraction (optional).
        self._llm = llm_adapter
        self._letters_index_path = sm_cfg.get(
            "letters_index_path", "data/pro_traders/letters_index.yaml"
        )
        self._letters_index = load_letters_index(self._letters_index_path)
        self._rationale_cache_path = os.path.join(self._cache_dir, "rationale_cache.json")
        self._rationale_cache: Dict[str, str] = self._load_rationale_cache()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def analyze_symbol(self, symbol: str, region_hint: Optional[str] = None) -> SmartMoneySignal:
        """Build a SmartMoneySignal for one symbol.

        region_hint: "US" or "IN" (optional). When set, irrelevant fetchers
        are skipped to keep latency down.
        """
        symbol_u = symbol.upper().strip()
        holders: List[ProTraderHolding] = []

        if region_hint != "IN":
            holders.extend(self._collect_us_holders(symbol_u))
        if region_hint != "US":
            holders.extend(self._collect_in_holders(symbol_u))
        holders.extend(self._collect_mf_holders(symbol_u))

        return self._aggregate(symbol_u, holders)

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Agent-protocol entry point used by orchestrators."""
        started = int(time.time() * 1000)
        run_id = ctx.run_id or str(uuid.uuid4())
        try:
            symbol = (ctx.input_data or {}).get("symbol", "")
            region = (ctx.input_data or {}).get("region")
            if not symbol:
                raise ValueError("input_data.symbol is required")
            signal = self.analyze_symbol(symbol, region)
            payload = {"smart_money_signal": {
                **{k: getattr(signal, k) for k in (
                    "symbol", "smart_money_score", "consensus_verdict",
                    "holders_count", "total_position_value_usd",
                    "recent_buyers", "recent_sellers", "consensus_thesis",
                    "contrarian_flag", "concentration_score",
                )},
                "holders": [asdict(h) for h in signal.holders],
            }}
            completed = int(time.time() * 1000)
            return AgentResult(
                success=True,
                run_id=run_id,
                rules_version=ctx.rules_version,
                timing=TimingInfo(started_epoch_ms=started, completed_epoch_ms=completed),
                payload=payload,
            )
        except Exception as exc:
            completed = int(time.time() * 1000)
            return AgentResult(
                success=False,
                run_id=run_id,
                rules_version=ctx.rules_version,
                timing=TimingInfo(started_epoch_ms=started, completed_epoch_ms=completed),
                errors=[AgentError(code="ANALYZE_FAILED", message=str(exc))],
                payload={},
            )

    # ------------------------------------------------------------------
    # Source collectors
    # ------------------------------------------------------------------

    def _collect_us_holders(self, symbol: str) -> List[ProTraderHolding]:
        """Walk US investors and pull whether each holds this symbol.

        Matches in this priority order:
          1. CUSIP lookup via OpenFIGI (most precise)
          2. Issuer-name fuzzy match against the ticker's company name

        For each match, compares the latest filing to the previous one and
        tags the action as NEW / ADD / TRIM / EXIT / HOLD.
        """
        out: List[ProTraderHolding] = []
        max_age = int(self._refresh.get("sec_13f", 30))
        cusip = ticker_to_cusip(symbol, self._cache_dir)
        company_name = self._company_name_for(symbol)

        for inv in self._watchlist.get("us_investors", []) or []:
            cik = str(inv.get("cik", "")).strip()
            if not cik:
                continue
            history = fetch_13f_history(cik, self._cache_dir, max_age_days=max_age, n=2)
            if not history:
                continue
            latest = history[0]
            previous = history[1] if len(history) > 1 else None

            curr_match = self._match_holding_in_13f(latest, symbol, cusip, company_name)
            prev_match = (
                self._match_holding_in_13f(previous, symbol, cusip, company_name)
                if previous else None
            )

            if curr_match is None and prev_match is None:
                continue

            curr_shares, curr_value = self._total_position(latest, curr_match)
            prev_shares, _ = self._total_position(previous, prev_match) if previous else (0.0, 0.0)
            action, change_pct = self._classify_action(prev_shares, curr_shares)

            display = curr_match or prev_match
            if display is None:
                continue
            investor_name = inv.get("name", latest.investor_name)
            rationale = self._get_rationale(investor_name, symbol, company_name)
            out.append(ProTraderHolding(
                investor=investor_name,
                region="US",
                shares=curr_shares if curr_shares else None,
                market_value_usd=curr_value if curr_value else None,
                qoq_change_pct=change_pct,
                first_seen_quarter=latest.period_of_report,
                last_action=action,
                rationale_excerpt=rationale,
            ))
        return out

    def _collect_in_holders(self, symbol: str) -> List[ProTraderHolding]:
        """Aggregate NSE bulk deals over the lookback window per investor.

        Emits ONE ProTraderHolding per matching investor, with net shares
        (positive = net accumulator, negative = net distributor), latest
        deal date for recency decay, and action derived from net direction.
        """
        out: List[ProTraderHolding] = []
        max_age = int(self._refresh.get("nse_bulk_deals", 1))
        deals = fetch_nse_bulk_deals(self._cache_dir, lookback_days=90, max_age_days=max_age)
        if not deals:
            return out
        sym_root = symbol.replace(".NS", "").replace(".BO", "").upper()
        symbol_deals = [d for d in deals if d.symbol.upper() == sym_root]
        if not symbol_deals:
            return out
        for inv in self._watchlist.get("in_investors", []) or []:
            aliases = inv.get("aliases") or [inv.get("name", "")]
            matched = filter_bulk_deals_by_investor(symbol_deals, aliases)
            if not matched:
                continue
            net_shares = 0.0
            total_inr = 0.0
            latest_date_iso: Optional[str] = None
            for d in matched:
                sign = 1 if d.buy_sell == "BUY" else -1
                net_shares += sign * d.quantity
                total_inr += d.quantity * d.avg_price_inr
                d_iso = self._normalize_in_date(d.date)
                if d_iso and (latest_date_iso is None or d_iso > latest_date_iso):
                    latest_date_iso = d_iso
            if net_shares > 0:
                action = "ADD"
            elif net_shares < 0:
                action = "TRIM"
            else:
                action = "HOLD"
            out.append(ProTraderHolding(
                investor=inv.get("name", ""),
                region="IN",
                shares=net_shares,
                market_value_usd=None,   # INR amount kept as-is in position_pct slot below
                qoq_change_pct=None,
                first_seen_quarter=latest_date_iso,
                last_action=action,
                position_pct_of_portfolio=None,
            ))
        return out

    def _collect_mf_holders(self, symbol: str) -> List[ProTraderHolding]:
        """Stub for AMFI monthly portfolio scans.

        Full implementation requires parsing each AMC's monthly portfolio
        disclosure (PDF/Excel). For now we return [] so the rest of the
        pipeline runs end-to-end.
        """
        return []

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _match_holding_in_13f(
        snap: ThirteenFSnapshot,
        symbol: str,
        cusip: Optional[str] = None,
        company_name: Optional[str] = None,
    ) -> Optional[ThirteenFHolding]:
        # Skip option positions; we care about equity ownership.
        equity = [h for h in snap.holdings if not h.put_call]

        if cusip:
            c8 = cusip[:8].upper()
            for h in equity:
                if (h.cusip or "").upper().startswith(c8):
                    return h

        for h in equity:
            if (h.ticker or "").upper() == symbol:
                return h

        if company_name:
            needle = ProTraderPortfolioAnalyzer._normalize_name(company_name)
            needle_tokens = needle.split()
            best: Optional[ThirteenFHolding] = None
            best_score = 0
            for h in equity:
                cand = ProTraderPortfolioAnalyzer._normalize_name(h.issuer_name)
                if not cand:
                    continue
                cand_tokens = cand.split()
                shared = 0
                for a, b in zip(needle_tokens, cand_tokens):
                    if a == b:
                        shared += 1
                    else:
                        break
                if shared > best_score and shared >= 1:
                    best_score = shared
                    best = h
            return best
        return None

    @staticmethod
    def _normalize_name(name: str) -> str:
        if not name:
            return ""
        n = name.upper()
        # Strip corporate suffixes that vary between SEC filings and Yahoo.
        for suffix in (
            " INCORPORATED", " INC.", " INC", " CORPORATION", " CORP.", " CORP",
            " COMPANY", " CO.", " COMPANIES", " LIMITED", " LTD.", " LTD",
            " HOLDINGS", " GROUP", " PLC", " THE",
        ):
            n = n.replace(suffix, "")
        n = n.replace(",", "").replace(".", "").replace("&", "AND")
        return " ".join(n.split())

    def _company_name_for(self, symbol: str) -> Optional[str]:
        if market_data is None:
            return None
        try:
            data = market_data.get_fundamentals(symbol)
            return data.get("longName") or data.get("shortName") or data.get("symbol")
        except Exception:
            return None

    @staticmethod
    def _total_position(
        snap: Optional[ThirteenFSnapshot],
        match: Optional[ThirteenFHolding],
    ) -> tuple[float, float]:
        """Sum (shares, value_usd) across all infoTable rows for the same CUSIP.

        13F filings often split one position across many <infoTable> rows
        (managed accounts). Without summing we'd massively understate.
        Returns (0, 0) when nothing matches.
        """
        if not snap or not match:
            return 0.0, 0.0
        target_cusip = (match.cusip or "").upper()
        target_name = ProTraderPortfolioAnalyzer._normalize_name(match.issuer_name)
        shares = 0.0
        value = 0.0
        for h in snap.holdings:
            if h.put_call:
                continue
            if target_cusip and (h.cusip or "").upper() == target_cusip:
                shares += float(h.shares or 0)
                value += float(h.value_usd or 0)
            elif not target_cusip and ProTraderPortfolioAnalyzer._normalize_name(h.issuer_name) == target_name:
                shares += float(h.shares or 0)
                value += float(h.value_usd or 0)
        return shares, value

    @staticmethod
    def _classify_action(prev_shares: float, curr_shares: float) -> tuple[str, Optional[float]]:
        """Classify a quarter-over-quarter holding delta.

        Returns (action, change_pct_or_None).
        """
        if curr_shares <= 0 and prev_shares > 0:
            return "EXIT", -100.0
        if prev_shares <= 0 and curr_shares > 0:
            return "NEW", None
        if prev_shares <= 0 and curr_shares <= 0:
            return "HOLD", None
        change_pct = (curr_shares - prev_shares) / prev_shares * 100.0
        if change_pct >= 10.0:
            return "ADD", round(change_pct, 1)
        if change_pct <= -10.0:
            return "TRIM", round(change_pct, 1)
        return "HOLD", round(change_pct, 1)

    def _aggregate(self, symbol: str, holders: List[ProTraderHolding]) -> SmartMoneySignal:
        if not holders:
            return SmartMoneySignal(
                symbol=symbol,
                smart_money_score=0.5,
                consensus_verdict="NEUTRAL",
                holders_count=0,
            )

        recent_buyers: List[str] = []
        recent_sellers: List[str] = []
        weighted_score = 0.0
        weight_total = 0.0

        for h in holders:
            region_w = float(self._region_weights.get(h.region, 1.0))
            investor_w = float(self._investor_weights.get(h.investor, 1.0))
            recency_w = self._recency_weight(h.first_seen_quarter, h.region)
            combined = region_w * investor_w * recency_w
            if combined <= 0:
                continue
            if h.last_action in ("ADD", "NEW"):
                weighted_score += 1.0 * combined
                recent_buyers.append(h.investor)
            elif h.last_action in ("TRIM", "EXIT"):
                weighted_score += 0.0 * combined
                recent_sellers.append(h.investor)
            else:  # HOLD
                weighted_score += 0.7 * combined
            weight_total += combined

        score = (weighted_score / weight_total) if weight_total else 0.5

        buyers, sellers = len(recent_buyers), len(recent_sellers)
        if buyers >= self._min_consensus and buyers > sellers:
            verdict = "ACCUMULATING"
        elif sellers >= self._min_consensus and sellers > buyers:
            verdict = "DISTRIBUTING"
        elif len(holders) >= self._min_consensus and buyers == 0 and sellers == 0:
            verdict = "STRONG_HOLD"
        elif buyers and sellers:
            verdict = "MIXED"
        else:
            verdict = "NEUTRAL"

        contrarian = (
            score >= 0.7
            and sellers
            and (sellers / max(1, buyers + sellers)) * 100 >= self._contrarian_threshold
        )

        total_value = sum(
            (h.market_value_usd or 0.0) for h in holders if h.market_value_usd
        ) or None

        # Pull a representative thesis from the strongest-weighted holder
        # who has one, so the report has at least one "why" up top.
        consensus_thesis: Optional[str] = None
        ranked = sorted(
            holders,
            key=lambda h: (
                float(self._investor_weights.get(h.investor, 1.0))
                * self._recency_weight(h.first_seen_quarter, h.region)
            ),
            reverse=True,
        )
        for h in ranked:
            if h.rationale_excerpt:
                consensus_thesis = f"{h.investor}: {h.rationale_excerpt}"
                break

        return SmartMoneySignal(
            symbol=symbol,
            smart_money_score=round(score, 4),
            consensus_verdict=verdict,
            holders_count=len(holders),
            total_position_value_usd=total_value,
            recent_buyers=recent_buyers,
            recent_sellers=recent_sellers,
            holders=holders,
            consensus_thesis=consensus_thesis,
            contrarian_flag=bool(contrarian),
        )

    # ------------------------------------------------------------------
    # Recency / rationale helpers
    # ------------------------------------------------------------------

    def _recency_weight(self, as_of_date: Optional[str], region: str) -> float:
        """Decay weight in [0.2, 1.0] based on how stale a holding date is.

        US (quarterly 13F): linear decay over `recency_decay_quarters`.
        IN bulk deals: step decay (1.0 @ 0-30d, 0.75 @ 30-60d, 0.5 @ 60-90d).
        MF: same as US.
        """
        if not as_of_date:
            return 0.5
        try:
            dt = datetime.fromisoformat(as_of_date[:10])
        except Exception:
            return 0.5
        days = max(0, (datetime.utcnow() - dt).days)
        if region == "IN":
            if days <= 30:
                return 1.0
            if days <= 60:
                return 0.75
            if days <= 90:
                return 0.5
            return 0.3
        quarters_old = days / 91.31
        decay_q = max(1.0, self._decay_quarters)
        return max(0.2, 1.0 - quarters_old / decay_q)

    @staticmethod
    def _normalize_in_date(s: Optional[str]) -> Optional[str]:
        """NSE serves dates like '01-May-2026' or '01-05-2026'. Return ISO."""
        if not s:
            return None
        for fmt in ("%d-%b-%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).date().isoformat()
            except ValueError:
                continue
        return s

    # ------------------------------------------------------------------
    # Letter rationale (LLM)
    # ------------------------------------------------------------------

    def _load_rationale_cache(self) -> Dict[str, str]:
        try:
            if os.path.exists(self._rationale_cache_path):
                with open(self._rationale_cache_path, "r", encoding="utf-8") as fh:
                    return json.load(fh) or {}
        except Exception:
            pass
        return {}

    def _save_rationale_cache(self) -> None:
        try:
            os.makedirs(os.path.dirname(self._rationale_cache_path), exist_ok=True)
            with open(self._rationale_cache_path, "w", encoding="utf-8") as fh:
                json.dump(self._rationale_cache, fh, indent=2)
        except Exception:
            pass

    def _get_rationale(
        self,
        investor: str,
        symbol: str,
        company_name: Optional[str],
    ) -> Optional[str]:
        """Return short LLM-extracted rationale for (investor, symbol).

        Cached forever (negative caching included). Skipped silently if no
        LLM adapter or no letter URLs are configured for the investor.
        """
        key = f"{investor}|{symbol.upper()}"
        if key in self._rationale_cache:
            cached = self._rationale_cache[key]
            return cached or None
        if self._llm is None:
            return None
        entry = self._letters_index.get(investor) or {}
        urls = entry.get("letters") or []
        if not urls:
            self._rationale_cache[key] = ""
            self._save_rationale_cache()
            return None
        for url in urls[:2]:   # try the newest two letters only
            text = fetch_letter_text(url, self._cache_dir, max_age_days=90)
            if not text or len(text) < 200:
                continue
            rationale = self._extract_rationale_via_llm(
                investor=investor,
                symbol=symbol,
                company_name=company_name,
                letter_text=text[:30000],
            )
            if rationale:
                self._rationale_cache[key] = rationale
                self._save_rationale_cache()
                return rationale
        self._rationale_cache[key] = ""
        self._save_rationale_cache()
        return None

    def _extract_rationale_via_llm(
        self,
        investor: str,
        symbol: str,
        company_name: Optional[str],
        letter_text: str,
    ) -> Optional[str]:
        sys_prompt = (
            "You extract investment rationales from investor letters. "
            "You answer concisely with no preamble or markdown."
        )
        company = company_name or symbol
        user_prompt = (
            f"From this letter by {investor}, extract a 1-2 sentence "
            f"rationale for their position in {symbol} ({company}). "
            f"If the letter does not discuss {company} or {symbol}, "
            "respond with exactly: NONE\n\n"
            f"LETTER:\n{letter_text}\n\n"
            "Rationale (or NONE):"
        )
        try:
            raw = self._llm.invoke(sys_prompt, user_prompt, max_tokens=200) or ""
        except Exception:
            return None
        out = raw.strip().strip('"').strip("'")
        if not out or out.upper().startswith("NONE") or len(out) < 15:
            return None
        # Trim runaway responses to one paragraph.
        out = out.split("\n\n")[0].strip()
        return out[:400]
