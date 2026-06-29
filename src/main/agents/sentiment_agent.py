"""Sentiment Agent — extracts sentiment signals from news_articles list.

Two paths:
  1. LLM (preferred): formats articles into text → asks LLM for structured JSON.
  2. Rule-based (fallback): keyword counting on headlines/summaries.

Input fields:
  - news_articles: list of {headline, summary, date, source}
  - recent_price_change: float (fraction, e.g. 0.015 = +1.5%)

Output: SentimentSignal with sentiment_score, counts, trend, confidence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import SentimentSignal

if TYPE_CHECKING:
    from ..controllers.data_context import StockDataContext


_POSITIVE_WORDS = {
    "beat", "record", "growth", "profit", "win", "contract", "upgrade",
    "strong", "outperform", "buy", "bullish", "surge", "rally", "gain",
    "expansion", "acquisition", "partnership", "award", "launch", "milestone",
    "dividend", "buyback", "raise", "guidance",
}

_NEGATIVE_WORDS = {
    "miss", "loss", "decline", "warning", "cut", "downgrade", "sell",
    "bearish", "drop", "fall", "plunge", "risk", "regulatory", "fine",
    "lawsuit", "fraud", "layoff", "recall", "bankruptcy", "weak", "probe",
    "investigation", "penalty", "debt", "default",
}

_SENTIMENT_SYSTEM_PROMPT = """\
You are a financial sentiment analyst. Given a list of recent news headlines and summaries,
output a JSON object with these exact keys:

{
  "sentiment_score": <0.0 = very negative … 1.0 = very positive>,
  "positive_news_count": <integer count of positive articles>,
  "negative_news_count": <integer count of negative articles>,
  "sentiment_trend": <one of: improving, stable, deteriorating>,
  "news_confidence_score": <0.0 to 1.0 based on article count and recency>
}

Only the JSON object. No commentary.
"""


class SentimentAgent(Agent):
    """Extracts structured sentiment signals from a list of news articles."""

    name = "sentiment_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}
        self._system_prompt = self._config.get("system_prompt", _SENTIMENT_SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _articles_to_text(articles: List[Dict[str, Any]]) -> str:
        parts = []
        for a in articles:
            headline = (a.get("headline") or "").strip()
            summary = (a.get("summary") or "").strip()
            date = (a.get("date") or "")
            line = f"[{date}] {headline}"
            if summary:
                line += f". {summary}"
            parts.append(line)
        return "\n".join(parts)

    @staticmethod
    def _classify_article(article: Dict[str, Any]) -> str:
        """Classify sentiment of a single article.
        
        Priority:
        1. Pre-computed FinBERT sentiment (from news cache)
        2. Live FinBERT analysis
        3. Keyword-based fallback
        """
        # Check for pre-computed sentiment (from NLP processor / cache)
        pre_label = article.get("sentiment_label", "")
        if pre_label in ("positive", "negative", "neutral"):
            return pre_label

        text = (
            (article.get("headline") or "") + " " + (article.get("summary") or "")
        ).strip()

        # Try FinBERT (lazy-loaded singleton)
        try:
            from ..controllers.nlp_processor import NLPProcessor
            nlp = NLPProcessor()
            if nlp.finbert_available and text:
                result = nlp.analyze_sentiment(text)
                return result.get("label", "neutral")
        except Exception:
            pass

        # Fallback: keyword counting
        text_lower = text.lower()
        pos = sum(1 for w in _POSITIVE_WORDS if w in text_lower)
        neg = sum(1 for w in _NEGATIVE_WORDS if w in text_lower)
        if pos > neg:
            return "positive"
        if neg > pos:
            return "negative"
        return "neutral"

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _analyse_with_llm(self, news_text: str, llm: Any) -> Dict[str, Any]:
        import json as _json

        response = llm.invoke(
            system_prompt=self._system_prompt,
            user_prompt=news_text,
        )
        try:
            return _json.loads(response)
        except (_json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _analyse_rule_based(
        self,
        articles: List[Dict[str, Any]],
        recent_price_change: float,
    ) -> Dict[str, Any]:
        if not articles:
            return {
                "sentiment_score": 0.5,
                "positive_news_count": 0,
                "negative_news_count": 0,
                "sentiment_trend": "stable",
                "news_confidence_score": 0.0,
            }

        classifications = [self._classify_article(a) for a in articles]
        total = len(classifications)
        pos_count = classifications.count("positive")
        neg_count = classifications.count("negative")
        neutral_count = total - pos_count - neg_count

        # Sentiment score: positives + half of neutrals / total
        sentiment_score = (pos_count + 0.5 * neutral_count) / total

        # Trend aligned with recent price movement
        if recent_price_change > 0.01:
            trend = "improving"
        elif recent_price_change < -0.01:
            trend = "deteriorating"
        else:
            trend = "stable"

        # Confidence grows with article count (saturates at 10)
        confidence = round(min(total / 10.0, 1.0), 4)

        return {
            "sentiment_score": round(sentiment_score, 4),
            "positive_news_count": pos_count,
            "negative_news_count": neg_count,
            "sentiment_trend": trend,
            "news_confidence_score": confidence,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data: Dict[str, Any], llm: Any = None) -> SentimentSignal:
        articles: List[Dict[str, Any]] = data.get("news_articles") or []
        recent_price_change = float(data.get("recent_price_change") or 0.0)

        if llm and articles:
            news_text = self._articles_to_text(articles)
            raw = self._analyse_with_llm(news_text, llm)
            if not raw:
                raw = self._analyse_rule_based(articles, recent_price_change)
        else:
            raw = self._analyse_rule_based(articles, recent_price_change)

        return SentimentSignal(
            sentiment_score=float(raw.get("sentiment_score", 0.5)),
            positive_news_count=int(raw.get("positive_news_count", 0)),
            negative_news_count=int(raw.get("negative_news_count", 0)),
            sentiment_trend=str(raw.get("sentiment_trend", "stable")),
            news_confidence_score=float(raw.get("news_confidence_score", 0.0)),
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
                # Extract news data from StockDataContext
                data = self._extract_from_context(stock_ctx)
            else:
                # Legacy: use input_data dict
                data = dict(ctx.input_data) if ctx.input_data else {}
                data.update(kwargs)
            
            llm = ctx.llm or self.get_llm_adapter()

            signal = self.analyse(data, llm=llm)
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
                errors=[AgentError(code="SENTIMENT_ERROR", message=str(exc))],
            )

    def _extract_from_context(self, stock_ctx: "StockDataContext") -> Dict[str, Any]:
        """Extract news and sentiment data fields from StockDataContext."""
        data: Dict[str, Any] = {}
        
        # Add stock identity
        data["symbol"] = stock_ctx.symbol
        
        # Convert NewsItem list to article dict list
        news_articles: List[Dict[str, Any]] = []
        for news_item in stock_ctx.news_items:
            news_articles.append({
                "headline": news_item.headline,
                "summary": getattr(news_item, "summary", None),
                "date": news_item.date,
                "source": news_item.source,
                "url": news_item.url,
            })
        
        data["news_articles"] = news_articles
        
        # Calculate recent price change from last_close and previous_close
        recent_price_change = 0.0
        if stock_ctx.last_close and stock_ctx.previous_close and stock_ctx.previous_close > 0:
            recent_price_change = (stock_ctx.last_close - stock_ctx.previous_close) / stock_ctx.previous_close
        
        data["recent_price_change"] = recent_price_change
        
        return data
