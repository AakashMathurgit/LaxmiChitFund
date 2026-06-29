"""Bull Agent — generates the bullish investment case for a stock.

Accepts a DebateContext containing:
  - Raw evidence: recent prices, news headlines, real fundamentals, events
  - Pre-computed signals: what analysis agents concluded
  - Market regime context

LLM path: rich context prompt → JSON argument with data citations
Rule-based path: checks positive signals from both raw data and agent scores
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import DebateArgument, DebateContext

DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")

_BULL_SYSTEM_PROMPT = """\
You are a bullish stock analyst. Given comprehensive market data for a stock,
construct the strongest possible case for BUYING this stock for a 3-10 day
swing trade.

You have access to:
- Real price data, fundamentals (PE, growth, margins), and news headlines
- Pre-computed analysis from technical, fundamental, sentiment, and event agents
- Market regime and volatility context

Focus on:
- Positive technical momentum (RSI, MACD, trend, breakout patterns)
- Strong fundamentals (growth, valuation, financial health) with real numbers
- Positive news sentiment — cite specific headlines as evidence
- Upcoming catalysts (earnings, corporate events)
- Favourable market regime

Output a JSON object:
{
  "recommendation": "BUY",
  "confidence": <0.0 to 1.0>,
  "key_points": ["point 1 — cite specific data", "point 2", "point 3"],
  "reasoning": "<2-3 sentence summary citing real data>"
}

Only the JSON object. No commentary.
"""


class BullAgent(Agent):
    """Generates the bullish case using LLM or rule-based fallback."""

    name = "bull_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _argue_with_llm(self, debate_ctx: DebateContext, llm: Any) -> Dict[str, Any]:
        import json as _json
        data_summary = debate_ctx.format_for_llm()
        response = llm.invoke(
            system_prompt=_BULL_SYSTEM_PROMPT,
            user_prompt=data_summary,
        )
        try:
            return _json.loads(response)
        except (_json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _argue_rule_based(self, debate_ctx: DebateContext) -> Dict[str, Any]:
        points: List[str] = []
        confidence_factors: List[float] = []

        signals = debate_ctx.signals

        # --- Technical bullish signals (from agent + raw data) ---
        if signals:
            t = signals.technical
            if t.technical_score > 0.6:
                points.append(f"Strong technical score ({t.technical_score:.2f})")
                confidence_factors.append(t.technical_score)
            if t.trend_direction == "bullish":
                points.append("Bullish trend (EMA20 > EMA50)")
                confidence_factors.append(0.7)
            if t.breakout_flag:
                points.append("Breakout detected near 52-week high with volume")
                confidence_factors.append(0.8)
            if t.macd_signal == "buy":
                points.append(f"MACD buy signal (RSI={t.rsi:.0f})")
                confidence_factors.append(0.65)

        # 52-week proximity from raw data
        if debate_ctx.week_52_low and debate_ctx.latest_price:
            pct_above_low = (debate_ctx.latest_price - debate_ctx.week_52_low) / debate_ctx.week_52_low
            if pct_above_low < 0.15:
                points.append(
                    f"Near 52-week low (₹{debate_ctx.week_52_low:.0f}), "
                    f"potential reversal opportunity"
                )
                confidence_factors.append(0.6)

        # --- Fundamental bullish signals (real numbers) ---
        f = debate_ctx.fundamentals
        if f:
            pe = f.get("pe_ratio")
            if pe is not None and pe < 20:
                points.append(f"Attractive valuation (PE={pe:.1f})")
                confidence_factors.append(0.7)
            growth = f.get("revenue_growth_yoy")
            if growth is not None and growth > 0.10:
                points.append(f"Strong revenue growth ({growth:.0%} YoY)")
                confidence_factors.append(0.7)
            roe = f.get("return_on_equity")
            if roe is not None and roe > 0.15:
                points.append(f"High return on equity ({roe:.0%})")
                confidence_factors.append(0.65)
            margin = f.get("profit_margin")
            if margin is not None and margin > 0.15:
                points.append(f"Healthy profit margins ({margin:.0%})")
                confidence_factors.append(0.6)

        if signals:
            fs = signals.fundamental
            if fs.fundamental_score > 0.6 and not any("valuation" in p.lower() for p in points):
                points.append(f"Solid fundamental score ({fs.fundamental_score:.2f})")
                confidence_factors.append(fs.fundamental_score)

        # --- Sentiment (cite real headlines) ---
        if signals:
            s = signals.sentiment
            if s.sentiment_score > 0.6:
                label = f"Positive sentiment ({s.sentiment_score:.2f}, "
                label += f"+{s.positive_news_count}/-{s.negative_news_count} articles)"
                # Cite top headline if available
                if debate_ctx.news_headlines:
                    top = debate_ctx.news_headlines[0].get("headline", "")
                    if top:
                        label += f' — e.g. "{top}"'
                points.append(label)
                confidence_factors.append(s.sentiment_score)
            if s.sentiment_trend == "improving":
                points.append("Sentiment trend improving")
                confidence_factors.append(0.6)

        # --- Events / catalysts ---
        if signals and signals.event.earnings_impact_flag:
            points.append("Upcoming earnings catalyst")
            confidence_factors.append(0.55)
        if debate_ctx.upcoming_events:
            for evt in debate_ctx.upcoming_events[:2]:
                etype = evt.get("event_type", "").lower()
                if etype in ("dividend", "buyback", "bonus"):
                    points.append(f"Positive event: {evt.get('description', etype)}")
                    confidence_factors.append(0.6)
                    break

        # --- Regime ---
        if debate_ctx.regime and debate_ctx.regime.market_regime.value == "bull_trend":
            points.append("Broad market in BULL TREND — favourable macro backdrop")
            confidence_factors.append(0.65)

        if not points:
            points.append("No strong bullish signals identified")

        confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors else 0.3
        )

        return {
            "recommendation": "BUY" if confidence > 0.5 else "HOLD",
            "confidence": round(min(confidence, 0.95), 4),
            "key_points": points[:5],
            "reasoning": f"Bull case based on {len(points)} positive signals.",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def argue(self, debate_ctx: DebateContext, llm: Any = None) -> DebateArgument:
        if llm:
            raw = self._argue_with_llm(debate_ctx, llm)
            if not raw:
                raw = self._argue_rule_based(debate_ctx)
        else:
            raw = self._argue_rule_based(debate_ctx)

        return DebateArgument(
            role="bull",
            recommendation=raw.get("recommendation", "HOLD"),
            confidence=float(raw.get("confidence", 0.5)),
            key_points=raw.get("key_points", []),
            reasoning=raw.get("reasoning", ""),
        )

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            debate_ctx: Optional[DebateContext] = kwargs.get("debate_context")
            if debate_ctx is None:
                # Legacy fallback: build minimal DebateContext from input_data
                data = dict(ctx.input_data) if ctx.input_data else {}
                data.update(kwargs)
                debate_ctx = DebateContext(
                    symbol=data.get("symbol", "UNKNOWN"),
                    date=data.get("date", ""),
                    latest_price=data.get("latest_price", 0.0),
                )
            llm = ctx.llm or self.get_llm_adapter()

            argument = self.argue(debate_ctx, llm=llm)
            payload = {"argument": argument.to_dict(), "raw_argument": argument}

            if DEBUG:
                print(f"\n[DEBUG] BullAgent | {debate_ctx.symbol}")
                print(f"  Recommendation: {argument.recommendation}")
                print(f"  Confidence: {argument.confidence:.1%}")
                for p in argument.key_points:
                    print(f"  + {p}")

            completed = self._post_run()
            return self._result(
                ctx=ctx, success=True, started=started, completed=completed,
                payload=payload,
                checksum_parts=[argument.recommendation, str(argument.confidence)],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx, success=False, started=started, completed=completed,
                payload={}, errors=[AgentError(code="BULL_ERROR", message=str(exc))],
            )
