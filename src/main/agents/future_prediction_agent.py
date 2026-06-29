"""Future Prediction Agent — generates multi-horizon price predictions.

Collects ALL available data for a stock:
  - Technical indicators (RSI, MACD, trend, breakout, support/resistance)
  - Fundamental data (PE, EPS, revenue growth, margins, market cap)
  - Sentiment signals (news headlines, sentiment score/trend)
  - Event data (earnings, corporate actions, catalysts)
  - Market regime and volatility context
  - Historical price patterns and similar past setups

Then asks GPT-4.1 for structured price predictions across 4 horizons:
  1 Week  |  1 Month  |  1 Quarter  |  1 Year

Each prediction includes expected price, direction, confidence, key drivers, and risks.
"""

from __future__ import annotations

import json as _json
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    FuturePrediction,
    HorizonPrediction,
    AgentFeatureBundle,
    DebateContext,
)

if TYPE_CHECKING:
    from ..controllers.data_context import StockDataContext


_PREDICTION_SYSTEM_PROMPT = """\
You are an expert financial analyst and market forecaster. Given comprehensive data about a stock — including technical indicators, fundamentals, news sentiment, market regime, and historical patterns — generate price predictions for 4 time horizons.

You have access to:
- Current price, 52-week range, historical price action
- Fundamental data: PE ratio, forward PE, EPS, revenue growth, profit margins, debt/equity, ROE, market cap
- Technical indicators: RSI, MACD, trend direction, breakout signals, support/resistance levels
- Recent news headlines with sentiment analysis
- Market regime (bull/bear/sideways) and volatility state
- Upcoming events and catalysts (earnings, dividends, corporate actions)
- Similar historical setups and their outcomes

For each horizon, provide:
- A specific price target with reasoning grounded in the data
- Confidence level based on data quality and predictability at that horizon
- Key drivers that support the prediction
- Risks that could invalidate it

Output ONLY a JSON object with this exact structure:
{
  "one_week": {
    "predicted_price": <float>,
    "predicted_change_pct": <float, e.g. 2.5 for +2.5%>,
    "confidence": <0.0 to 1.0>,
    "direction": "bullish" | "bearish" | "neutral",
    "key_drivers": ["driver 1 with data", "driver 2"],
    "risks": ["risk 1", "risk 2"],
    "reasoning": "1-2 sentences citing specific data points"
  },
  "one_month": { ... same structure ... },
  "one_quarter": { ... same structure ... },
  "one_year": { ... same structure ... },
  "overall_outlook": "bullish" | "bearish" | "neutral",
  "overall_confidence": <0.0 to 1.0>,
  "summary": "2-3 sentence executive summary of the stock's outlook across all horizons"
}

Important guidelines:
- Ground every prediction in the provided data — cite specific numbers
- Short-term predictions should weigh technicals and sentiment more heavily
- Long-term predictions should weigh fundamentals and growth trajectory more
- Be realistic about confidence — longer horizons should have lower confidence
- Account for market regime and volatility in your confidence levels
- If news is strongly positive or negative, adjust short-term predictions accordingly

Only the JSON object. No commentary outside JSON.
"""


class FuturePredictionAgent(Agent):
    """Generates multi-horizon price predictions using LLM analysis."""

    name = "future_prediction_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}

    # ------------------------------------------------------------------
    # Data formatting
    # ------------------------------------------------------------------

    def _format_prediction_context(
        self,
        stock_ctx: "StockDataContext",
        debate_ctx: Optional[DebateContext] = None,
        bundle: Optional[AgentFeatureBundle] = None,
    ) -> str:
        """Build a rich text context for the LLM from all sources."""
        lines = []

        # === Identity & Price ===
        lines.append(f"=== STOCK: {stock_ctx.symbol} ({stock_ctx.company_name or 'N/A'}) ===")
        lines.append(f"Sector: {stock_ctx.sector or 'N/A'} | Industry: {stock_ctx.industry or 'N/A'}")
        lines.append(f"Exchange: {stock_ctx.exchange or 'N/A'} | Currency: {stock_ctx.currency or 'USD'}")
        lines.append(f"Current Price: ${stock_ctx.last_close:,.2f}" if stock_ctx.last_close else "Current Price: N/A")
        if stock_ctx.previous_close and stock_ctx.last_close:
            day_chg = (stock_ctx.last_close - stock_ctx.previous_close) / stock_ctx.previous_close * 100
            lines.append(f"Day Change: {day_chg:+.2f}%")
        lines.append(f"Last Trading Date: {stock_ctx.last_trading_date or 'N/A'}")

        # === 52-Week Range ===
        recent_252 = stock_ctx.historical_ohlc[-252:] if stock_ctx.historical_ohlc else []
        if recent_252:
            w52_high = max(p.high for p in recent_252)
            w52_low = min(p.low for p in recent_252)
            lines.append(f"52-Week High: ${w52_high:,.2f} | 52-Week Low: ${w52_low:,.2f}")
            if stock_ctx.last_close:
                pct_from_high = (stock_ctx.last_close - w52_high) / w52_high * 100
                pct_from_low = (stock_ctx.last_close - w52_low) / w52_low * 100
                lines.append(f"  From 52w High: {pct_from_high:+.1f}% | From 52w Low: {pct_from_low:+.1f}%")

        # === Price History Summary ===
        closes = stock_ctx.get_historical_closes()
        if closes and len(closes) >= 5:
            lines.append(f"\n--- PRICE HISTORY ---")
            lines.append(f"Last 5 closes: {[round(c, 2) for c in closes[-5:]]}")
            if len(closes) >= 20:
                avg_20 = sum(closes[-20:]) / 20
                lines.append(f"20-day avg: ${avg_20:,.2f}")
            if len(closes) >= 50:
                avg_50 = sum(closes[-50:]) / 50
                lines.append(f"50-day avg: ${avg_50:,.2f}")
            if len(closes) >= 200:
                avg_200 = sum(closes[-200:]) / 200
                lines.append(f"200-day avg: ${avg_200:,.2f}")
            # Performance periods
            if len(closes) >= 5:
                chg_1w = (closes[-1] - closes[-5]) / closes[-5] * 100
                lines.append(f"1-week change: {chg_1w:+.2f}%")
            if len(closes) >= 22:
                chg_1m = (closes[-1] - closes[-22]) / closes[-22] * 100
                lines.append(f"1-month change: {chg_1m:+.2f}%")
            if len(closes) >= 66:
                chg_3m = (closes[-1] - closes[-66]) / closes[-66] * 100
                lines.append(f"3-month change: {chg_3m:+.2f}%")
            if len(closes) >= 252:
                chg_1y = (closes[-1] - closes[-252]) / closes[-252] * 100
                lines.append(f"1-year change: {chg_1y:+.2f}%")

        # === Volume ===
        volumes = stock_ctx.get_historical_volumes()
        if volumes and len(volumes) >= 20:
            avg_vol_20 = sum(volumes[-20:]) / 20
            last_vol = volumes[-1] if volumes else 0
            vol_ratio = last_vol / avg_vol_20 if avg_vol_20 else 0
            lines.append(f"Volume: {last_vol:,.0f} (vs 20d avg: {avg_vol_20:,.0f}, ratio: {vol_ratio:.2f}x)")

        # === Fundamentals ===
        f = stock_ctx.fundamentals
        if f:
            lines.append(f"\n--- FUNDAMENTALS ---")
            parts = []
            if f.pe_ratio is not None: parts.append(f"PE Ratio: {f.pe_ratio:.1f}")
            if f.forward_pe is not None: parts.append(f"Forward PE: {f.forward_pe:.1f}")
            if f.eps is not None: parts.append(f"EPS: ${f.eps:.2f}")
            if f.revenue_growth_yoy is not None: parts.append(f"Revenue Growth YoY: {f.revenue_growth_yoy:.1%}")
            if f.profit_margin is not None: parts.append(f"Profit Margin: {f.profit_margin:.1%}")
            if f.debt_to_equity is not None: parts.append(f"Debt/Equity: {f.debt_to_equity:.2f}")
            if f.return_on_equity is not None: parts.append(f"ROE: {f.return_on_equity:.1%}")
            if f.market_cap is not None: parts.append(f"Market Cap: ${f.market_cap:,.0f}")
            if f.dividend_yield is not None: parts.append(f"Dividend Yield: {f.dividend_yield:.2%}")
            if f.analyst_rating is not None: parts.append(f"Analyst Rating: {f.analyst_rating}")
            if f.sector: parts.append(f"Sector: {f.sector}")
            lines.append("\n".join(parts))

        # === Technical Signals (from bundle) ===
        if bundle:
            t = bundle.technical
            lines.append(f"\n--- TECHNICAL ANALYSIS ---")
            lines.append(f"Technical Score: {t.technical_score:.2f}/1.0")
            lines.append(f"RSI(14): {t.rsi:.1f}")
            lines.append(f"MACD Signal: {t.macd_signal}")
            lines.append(f"Trend Direction: {t.trend_direction}")
            lines.append(f"Breakout: {'YES' if t.breakout_flag else 'No'}")
            lines.append(f"Volatility: {t.volatility:.2f}")
            if t.support_level is not None:
                lines.append(f"Support: ${t.support_level:,.2f}")
            if t.resistance_level is not None:
                lines.append(f"Resistance: ${t.resistance_level:,.2f}")

            # Fundamental signals
            fs = bundle.fundamental
            lines.append(f"\n--- FUNDAMENTAL SCORE ---")
            lines.append(f"Fundamental Score: {fs.fundamental_score:.2f}/1.0")
            lines.append(f"Valuation: {fs.valuation_label}")
            lines.append(f"Growth Score: {fs.growth_score:.2f}")
            lines.append(f"Financial Health: {fs.financial_health_score:.2f}")

            # Sentiment signals
            s = bundle.sentiment
            lines.append(f"\n--- SENTIMENT ---")
            lines.append(f"Sentiment Score: {s.sentiment_score:.2f}/1.0")
            lines.append(f"Positive News: {s.positive_news_count} | Negative: {s.negative_news_count}")
            lines.append(f"Sentiment Trend: {s.sentiment_trend}")
            lines.append(f"News Confidence: {s.news_confidence_score:.2f}")

            # Event signals
            e = bundle.event
            lines.append(f"\n--- EVENTS ---")
            lines.append(f"Event Score: {e.event_score:.2f}")
            lines.append(f"Earnings Impact: {'YES' if e.earnings_impact_flag else 'No'}")
            lines.append(f"Event Risk: {e.event_risk_level}")

            # Regime
            r = bundle.regime
            lines.append(f"\n--- MARKET REGIME ---")
            lines.append(f"Regime: {r.market_regime.value}")
            lines.append(f"Volatility State: {r.volatility_state.value}")
            lines.append(f"Regime Confidence: {r.regime_confidence:.0%}")

        # === News Headlines ===
        if stock_ctx.news_items:
            lines.append(f"\n--- RECENT NEWS ({len(stock_ctx.news_items)} articles) ---")
            for i, item in enumerate(stock_ctx.news_items[:15], 1):
                src = item.source or ""
                date = item.date or ""
                sentiment = item.sentiment_label or ""
                tag = f" [{src}]" if src else ""
                tag += f" ({date})" if date else ""
                tag += f" [{sentiment}]" if sentiment else ""
                lines.append(f"  {i}. {item.headline or 'N/A'}{tag}")
                if item.news_text:
                    lines.append(f"     {item.news_text[:200]}")

        # === Events ===
        if stock_ctx.event_data:
            lines.append(f"\n--- UPCOMING EVENTS ---")
            for evt in stock_ctx.event_data[:5]:
                if evt.earnings_date:
                    lines.append(f"  Earnings Date: {evt.earnings_date}")
                for action in (evt.recent_corporate_actions or [])[:5]:
                    lines.append(f"  Corporate Action: {action}")

        # === Similar Past Setups (RAG) ===
        if debate_ctx and debate_ctx.similar_past_setups:
            lines.append(f"\n--- SIMILAR HISTORICAL SETUPS ---")
            for i, setup in enumerate(debate_ctx.similar_past_setups[:5], 1):
                ret = setup.get("return_5d")
                ret_str = f"{ret:+.1%}" if ret is not None else "pending"
                lines.append(
                    f"  {i}. {setup.get('symbol', '?')} ({setup.get('date', '?')}) "
                    f"-> {ret_str} [{setup.get('regime', '?')}]"
                )

        # === Past Trades ===
        if debate_ctx and debate_ctx.past_trades_this_symbol:
            lines.append(f"\n--- PAST TRADES ON {stock_ctx.symbol} ---")
            for trade in debate_ctx.past_trades_this_symbol[:5]:
                lines.append(
                    f"  - {trade.get('decision', '?')} @ ${trade.get('entry_price', 0):.0f} "
                    f"-> {trade.get('pnl', 'pending')} ({trade.get('notes', '')})"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM prediction
    # ------------------------------------------------------------------

    def _predict_with_llm(self, context_text: str, llm: Any) -> Dict[str, Any]:
        """Send context to LLM and get structured predictions."""
        response = llm.invoke(
            system_prompt=_PREDICTION_SYSTEM_PROMPT,
            user_prompt=context_text,
            temperature=0.3,
            max_tokens=2048,
        )
        try:
            return _json.loads(response)
        except (_json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Parse LLM output into dataclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_horizon(raw: Dict[str, Any], horizon_key: str, horizon_label: str, current_price: float) -> HorizonPrediction:
        return HorizonPrediction(
            horizon=horizon_key,
            horizon_label=horizon_label,
            predicted_price=float(raw.get("predicted_price", current_price)),
            predicted_change_pct=float(raw.get("predicted_change_pct", 0.0)),
            confidence=float(raw.get("confidence", 0.3)),
            direction=raw.get("direction", "neutral"),
            key_drivers=raw.get("key_drivers", []),
            risks=raw.get("risks", []),
            reasoning=raw.get("reasoning", ""),
        )

    def _parse_prediction(self, llm_output: Dict[str, Any], symbol: str, date: str, current_price: float, stock_ctx: "StockDataContext") -> FuturePrediction:
        """Convert raw LLM JSON into FuturePrediction dataclass."""
        f = stock_ctx.fundamentals
        market_cap_str = None
        if f and f.market_cap:
            mc = f.market_cap
            if mc >= 1e12:
                market_cap_str = f"${mc/1e12:.2f}T"
            elif mc >= 1e9:
                market_cap_str = f"${mc/1e9:.1f}B"
            elif mc >= 1e6:
                market_cap_str = f"${mc/1e6:.0f}M"

        one_week = self._parse_horizon(llm_output.get("one_week", {}), "1_week", "1 Week", current_price)
        one_month = self._parse_horizon(llm_output.get("one_month", {}), "1_month", "1 Month", current_price)
        one_quarter = self._parse_horizon(llm_output.get("one_quarter", {}), "1_quarter", "1 Quarter", current_price)
        one_year = self._parse_horizon(llm_output.get("one_year", {}), "1_year", "1 Year", current_price)

        return FuturePrediction(
            symbol=symbol,
            date=date,
            current_price=current_price,
            market_cap=market_cap_str,
            sector=stock_ctx.sector,
            one_week=one_week,
            one_month=one_month,
            one_quarter=one_quarter,
            one_year=one_year,
            overall_outlook=llm_output.get("overall_outlook", "neutral"),
            overall_confidence=float(llm_output.get("overall_confidence", 0.3)),
            summary=llm_output.get("summary", ""),
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def predict(
        self,
        stock_ctx: "StockDataContext",
        llm: Any,
        bundle: Optional[AgentFeatureBundle] = None,
        debate_ctx: Optional[DebateContext] = None,
    ) -> FuturePrediction:
        """Generate future predictions for a stock.

        Parameters
        ----------
        stock_ctx : StockDataContext
            Fully populated stock context
        llm : LLMAdapter
            LLM for generating predictions
        bundle : AgentFeatureBundle, optional
            Pre-computed agent signals (technical, fundamental, sentiment, event, regime)
        debate_ctx : DebateContext, optional
            Debate context with similar setups and past trades

        Returns
        -------
        FuturePrediction with 4 horizon predictions
        """
        symbol = stock_ctx.symbol
        date = stock_ctx.last_trading_date or ""
        current_price = stock_ctx.last_close or 0.0

        context_text = self._format_prediction_context(stock_ctx, debate_ctx, bundle)
        llm_output = self._predict_with_llm(context_text, llm)

        if not llm_output:
            # Return empty prediction on LLM failure
            return FuturePrediction(
                symbol=symbol, date=date, current_price=current_price,
                summary="Prediction failed — LLM returned no valid response.",
            )

        return self._parse_prediction(llm_output, symbol, date, current_price, stock_ctx)

    # ------------------------------------------------------------------
    # Agent interface (run method)
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            stock_ctx = kwargs.get("stock_context")
            bundle = kwargs.get("bundle")
            debate_ctx = kwargs.get("debate_context")
            llm = ctx.llm or self.get_llm_adapter()

            prediction = self.predict(stock_ctx, llm, bundle=bundle, debate_ctx=debate_ctx)

            completed = self._post_run()
            return self._result(
                ctx, True, started, completed,
                payload={"prediction": prediction.to_dict(), "raw_prediction": prediction},
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx, False, started, completed,
                payload={},
                errors=[AgentError(code="PREDICTION_ERROR", message=str(exc))],
            )
