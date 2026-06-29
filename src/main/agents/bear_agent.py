"""Bear Agent — generates the bearish investment case for a stock.

Accepts a DebateContext containing:
  - Raw evidence: recent prices, news headlines, real fundamentals, events
  - Pre-computed signals: what analysis agents concluded
  - Market regime context

LLM path: rich context prompt → JSON argument with data citations
Rule-based path: checks negative signals from both raw data and agent scores
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import DebateArgument, DebateContext

DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")

_BEAR_SYSTEM_PROMPT = """\
You are a bearish stock analyst. Given comprehensive market data for a stock,
construct the strongest possible case for NOT buying (or SELLING) this stock.

You have access to:
- Real price data, fundamentals (PE, growth, margins), and news headlines
- Pre-computed analysis from technical, fundamental, sentiment, and event agents
- Market regime and volatility context

Focus on:
- Overbought technicals (high RSI, exhaustion, bearish divergence)
- Weak or expensive fundamentals (high PE, low growth, high debt) with real numbers
- Negative news sentiment — cite specific headlines as evidence
- Risks (earnings miss, regulatory, sector weakness, corporate events)
- Unfavourable market regime (bear trend, high volatility)
- Price near 52-week high with weak volume (distribution)

Output a JSON object:
{
  "recommendation": "SELL" or "HOLD",
  "confidence": <0.0 to 1.0>,
  "key_points": ["point 1 — cite specific data", "point 2", "point 3"],
  "reasoning": "<2-3 sentence summary citing real data>"
}

Only the JSON object. No commentary.
"""


class BearAgent(Agent):
    """Generates the bearish case using LLM or rule-based fallback."""

    name = "bear_agent"

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
            system_prompt=_BEAR_SYSTEM_PROMPT,
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

        # --- Technical bearish signals (from agent + raw data) ---
        if signals:
            t = signals.technical
            if t.rsi > 70:
                points.append(f"RSI overbought ({t.rsi:.0f})")
                confidence_factors.append(0.7)
            if t.trend_direction == "bearish":
                points.append("Bearish trend (EMA20 < EMA50)")
                confidence_factors.append(0.7)
            if t.technical_score < 0.4:
                points.append(f"Weak technical score ({t.technical_score:.2f})")
                confidence_factors.append(0.6)
            if t.volatility > 0.7:
                points.append(f"High volatility ({t.volatility:.2f}) increases downside risk")
                confidence_factors.append(0.6)
            if t.macd_signal == "sell":
                points.append("MACD sell signal — bearish momentum")
                confidence_factors.append(0.65)

        # 52-week proximity from raw data
        if debate_ctx.week_52_high and debate_ctx.latest_price:
            pct_below_high = (debate_ctx.week_52_high - debate_ctx.latest_price) / debate_ctx.week_52_high
            if pct_below_high > 0.25:
                points.append(
                    f"Trading {pct_below_high:.0%} below 52-week high "
                    f"(₹{debate_ctx.week_52_high:.0f}) — sustained weakness"
                )
                confidence_factors.append(0.6)

        # Recent price decline from raw data
        if debate_ctx.recent_ohlc and len(debate_ctx.recent_ohlc) >= 5:
            closes = [b["close"] for b in debate_ctx.recent_ohlc[-5:] if "close" in b]
            if len(closes) >= 2 and closes[-1] < closes[0]:
                decline = (closes[0] - closes[-1]) / closes[0] * 100
                if decline > 3:
                    points.append(f"Down {decline:.1f}% over last 5 sessions")
                    confidence_factors.append(0.6)

        # --- Fundamental bearish signals (real numbers) ---
        f = debate_ctx.fundamentals
        if f:
            pe = f.get("pe_ratio")
            if pe is not None and pe > 40:
                points.append(f"Expensive valuation (PE={pe:.1f})")
                confidence_factors.append(0.7)
            growth = f.get("revenue_growth_yoy")
            if growth is not None and growth < 0.05:
                label = f"Weak revenue growth ({growth:.0%} YoY)"
                if growth < 0:
                    label = f"Revenue DECLINING ({growth:.0%} YoY)"
                points.append(label)
                confidence_factors.append(0.65)
            de = f.get("debt_to_equity")
            if de is not None and de > 1.5:
                points.append(f"High leverage (D/E={de:.2f})")
                confidence_factors.append(0.6)
            margin = f.get("profit_margin")
            if margin is not None and margin < 0.05:
                points.append(f"Thin profit margins ({margin:.1%})")
                confidence_factors.append(0.6)

        if signals:
            fs = signals.fundamental
            if fs.financial_health_score < 0.4:
                if not any("leverage" in p.lower() or "margin" in p.lower() for p in points):
                    points.append(f"Weak financial health score ({fs.financial_health_score:.2f})")
                    confidence_factors.append(0.6)

        # --- Sentiment (cite real headlines) ---
        if signals:
            s = signals.sentiment
            if s.sentiment_score < 0.4:
                label = f"Negative sentiment ({s.sentiment_score:.2f}, "
                label += f"+{s.positive_news_count}/-{s.negative_news_count} articles)"
                # Cite negative headline if available
                if debate_ctx.news_headlines:
                    for item in debate_ctx.news_headlines:
                        headline = item.get("headline", "")
                        hl = headline.lower()
                        if any(w in hl for w in ("fall", "drop", "loss", "weak", "concern",
                                                  "risk", "decline", "cut", "miss", "warn")):
                            label += f' — e.g. "{headline}"'
                            break
                points.append(label)
                confidence_factors.append(0.65)
            if s.sentiment_trend == "deteriorating":
                points.append("Sentiment deteriorating")
                confidence_factors.append(0.6)

        # --- Event risks ---
        if signals:
            e = signals.event
            if e.event_risk_level == "high":
                points.append("High event risk (earnings/gap/news)")
                confidence_factors.append(0.6)
            if e.gap_down_flag:
                points.append("Recent gap-down detected — potential distribution")
                confidence_factors.append(0.7)

        if debate_ctx.upcoming_events:
            for evt in debate_ctx.upcoming_events[:3]:
                etype = evt.get("event_type", "").lower()
                if etype in ("regulatory", "delisting", "warning"):
                    points.append(f"Risk event: {evt.get('description', etype)}")
                    confidence_factors.append(0.65)
                    break

        # --- Market regime ---
        if debate_ctx.regime:
            regime_val = debate_ctx.regime.market_regime.value
            if regime_val == "bear_trend":
                points.append("Broad market in BEAR TREND — avoid new buys")
                confidence_factors.append(0.75)
            elif regime_val == "high_volatility":
                points.append("Market HIGH VOLATILITY — reduce exposure")
                confidence_factors.append(0.6)

        if not points:
            points.append("No strong bearish signals identified")

        confidence = (
            sum(confidence_factors) / len(confidence_factors)
            if confidence_factors else 0.3
        )

        return {
            "recommendation": "SELL" if confidence > 0.65 else "HOLD",
            "confidence": round(min(confidence, 0.95), 4),
            "key_points": points[:5],
            "reasoning": f"Bear case based on {len(points)} risk factors.",
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
            role="bear",
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
                print(f"\n[DEBUG] BearAgent | {debate_ctx.symbol}")
                print(f"  Recommendation: {argument.recommendation}")
                print(f"  Confidence: {argument.confidence:.1%}")
                for p in argument.key_points:
                    print(f"  - {p}")

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
                payload={}, errors=[AgentError(code="BEAR_ERROR", message=str(exc))],
            )
