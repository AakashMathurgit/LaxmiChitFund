"""Event Agent — detects earnings, gaps, dividends, splits, and major news.

Input fields (from StockDataContext via orchestrator):
  - earnings_date:      str | None  (ISO date of next/last earnings)
  - earnings_results:   dict | None (EPS actual vs expected, etc.)
  - dividend_info:      dict | None (ex-date, amount)
  - stock_split_info:   dict | None (ratio, date)
  - recent_gap_data:    dict | None ({gap_pct: float})
  - major_news_flag:    bool        (True if > 5 news items)
  - news_articles:      list        (for optional LLM event classification)

Output: EventSignal — event_score, flags, risk_level, optional LLM event_type.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import EventSignal

if TYPE_CHECKING:
    from ..controllers.data_context import StockDataContext


_EVENT_SYSTEM_PROMPT = """\
You are a financial event classifier. Given a list of recent news headlines,
identify the most significant catalyst and output a JSON object:

{
  "event_type": <one of: earnings_beat, earnings_miss, guidance_raise, guidance_cut,
                 ma_announcement, ceo_change, regulatory_news, big_contract,
                 insider_buy, insider_sell, buyback, none>,
  "event_description": <one-sentence summary or null>
}

Only the JSON object. No commentary.
"""


class EventAgent(Agent):
    """Detects discrete catalyst events and scores their expected swing impact."""

    name = "event_agent"

    EARNINGS_WINDOW_DAYS = 7    # ± days from today to flag an earnings event
    GAP_THRESHOLD = 0.02        # 2% price gap considered significant

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}
        self._system_prompt = self._config.get("system_prompt", _EVENT_SYSTEM_PROMPT)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _days_to_earnings(earnings_date_str: Optional[str]) -> Optional[int]:
        """Return days between today and earnings date (negative = past)."""
        if not earnings_date_str:
            return None
        try:
            target = datetime.strptime(str(earnings_date_str)[:10], "%Y-%m-%d").date()
            return (target - date.today()).days
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # LLM event classification (optional)
    # ------------------------------------------------------------------

    def _classify_with_llm(
        self,
        articles: List[Dict[str, Any]],
        llm: Any,
    ) -> Dict[str, Any]:
        import json as _json

        headlines = "\n".join(
            a.get("headline", "") for a in articles if a.get("headline")
        )
        if not headlines:
            return {}
        response = llm.invoke(
            system_prompt=self._system_prompt,
            user_prompt=headlines,
        )
        try:
            return _json.loads(response)
        except (_json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data: Dict[str, Any], llm: Any = None) -> EventSignal:
        earnings_date_str: Optional[str] = data.get("earnings_date")
        dividend_info: Optional[Dict] = data.get("dividend_info")
        stock_split_info: Optional[Dict] = data.get("stock_split_info")
        recent_gap_data: Optional[Dict] = data.get("recent_gap_data")
        major_news_flag: bool = bool(data.get("major_news_flag", False))
        articles: List[Dict[str, Any]] = data.get("news_articles") or []

        # --- Earnings proximity ---
        days_to_earn = self._days_to_earnings(earnings_date_str)
        earnings_impact = (
            days_to_earn is not None
            and -1 <= days_to_earn <= self.EARNINGS_WINDOW_DAYS
        )

        # --- Gap detection ---
        gap_pct = float((recent_gap_data or {}).get("gap_pct", 0.0) or 0.0)
        gap_up = gap_pct >= self.GAP_THRESHOLD
        gap_down = gap_pct <= -self.GAP_THRESHOLD

        # --- Risk level ---
        risk_factors = sum([
            earnings_impact,
            bool(gap_up or gap_down),
            major_news_flag,
            bool(stock_split_info),
            bool(dividend_info),
        ])
        if risk_factors >= 3:
            risk_level = "high"
        elif risk_factors >= 1:
            risk_level = "medium"
        else:
            risk_level = "low"

        # --- Optional LLM event classification ---
        event_type = None
        event_desc = None
        if llm and articles:
            llm_result = self._classify_with_llm(articles, llm)
            event_type = llm_result.get("event_type")
            event_desc = llm_result.get("event_description")

        # --- Composite event score ---
        gap_score = min(abs(gap_pct) / 0.05, 1.0)      # 5% gap → full score
        earnings_score = 1.0 if earnings_impact else 0.0
        risk_score = {"low": 0.1, "medium": 0.5, "high": 0.9}[risk_level]

        event_score = round(
            0.40 * earnings_score
            + 0.30 * gap_score
            + 0.30 * risk_score,
            4,
        )

        return EventSignal(
            event_score=event_score,
            earnings_impact_flag=earnings_impact,
            event_risk_level=risk_level,
            gap_up_flag=gap_up,
            gap_down_flag=gap_down,
            event_type=event_type,
            event_description=event_desc,
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
                # Extract event data from StockDataContext
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
                errors=[AgentError(code="EVENT_ERROR", message=str(exc))],
            )

    def _extract_from_context(self, stock_ctx: "StockDataContext") -> Dict[str, Any]:
        """Extract event data fields from StockDataContext."""
        data: Dict[str, Any] = {}
        
        # Add stock identity
        data["symbol"] = stock_ctx.symbol
        
        # Extract from EventData list
        earnings_date = None
        dividend_info = None
        stock_split_info = None
        
        for event in stock_ctx.event_data:
            # Get earnings date
            if event.earnings_date:
                earnings_date = event.earnings_date
            
            # Process corporate actions
            if event.recent_corporate_actions:
                for action in event.recent_corporate_actions:
                    action_type = action.get("type", "")
                    if action_type == "dividend":
                        dividend_info = {
                            "ex_date": action.get("date"),
                            "amount": action.get("value"),
                        }
                    elif action_type == "split":
                        stock_split_info = {
                            "date": action.get("date"),
                            "ratio": action.get("value"),
                        }
        
        data["earnings_date"] = earnings_date
        data["dividend_info"] = dividend_info
        data["stock_split_info"] = stock_split_info
        
        # Calculate gap from recent price bars
        data["recent_gap_data"] = self._calculate_gap(stock_ctx)
        
        # Major news flag: more than 5 articles
        data["major_news_flag"] = len(stock_ctx.news_items) > 5
        
        # Convert NewsItem list to article dict list
        news_articles: List[Dict[str, Any]] = []
        for news_item in stock_ctx.news_items:
            news_articles.append({
                "headline": news_item.headline,
                "summary": getattr(news_item, "summary", None),
                "date": news_item.date,
                "source": news_item.source,
            })
        
        data["news_articles"] = news_articles
        
        return data
    
    def _calculate_gap(self, stock_ctx: "StockDataContext") -> Optional[Dict[str, float]]:
        """Calculate gap percentage from last two trading days."""
        if len(stock_ctx.historical_ohlc) < 2:
            return None
        
        # Get last two bars
        today = stock_ctx.historical_ohlc[-1]
        yesterday = stock_ctx.historical_ohlc[-2]
        
        # Gap = (today's open - yesterday's close) / yesterday's close
        if yesterday.close and yesterday.close > 0:
            gap_pct = (today.open - yesterday.close) / yesterday.close
            return {"gap_pct": gap_pct}
        
        return None
