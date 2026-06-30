"""Cross-Ticker Agent — explains an intraday move via own + peer news.

Given a stock that moved significantly and the recent news for both the stock
and its peers (competitors / suppliers / customers), this agent decides whether
the move is:
  - own_news      : explained by the stock's own headlines
  - peer_spillover: explained by a peer's news (e.g. a competitor's guidance
                    miss is a tailwind for this stock; a supplier disruption is
                    a headwind for its customers)
  - sector_move   : the whole sector moved together (no single driver)
  - unexplained   : the move is not backed by any news narrative

LLM path: rich context prompt -> JSON. Rule-based path: sentiment heuristics.
Mirrors the structure of bull_agent.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import CrossTickerSignal

DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")

_CROSS_TICKER_SYSTEM_PROMPT = """\
You are a cross-asset equity analyst. A US stock just made a notable intraday
move. You are given the stock's own recent news plus recent news for its peers
(competitors, suppliers, customers). Decide what best explains the move.

Reasoning guidance:
- A competitor's BAD news (miss, downgrade, recall, outage) is often a TAILWIND
  for this stock; a competitor's GOOD news is often a HEADWIND.
- A supplier's disruption or price hike is a HEADWIND for its customers.
- A customer's strong demand is a TAILWIND for its suppliers.
- If the whole sector moved together with no single driver, call it sector_move.
- If nothing in the news explains the move, call it unexplained.

Output a JSON object:
{
  "news_synced": <true|false>,
  "explanation_type": "own_news" | "peer_spillover" | "sector_move" | "unexplained",
  "driver_ticker": "<peer ticker responsible, or null>",
  "driver_headline": "<the single headline most responsible, or empty>",
  "confidence": <0.0 to 1.0>,
  "reasoning": "<2-3 sentences citing the specific news>"
}

Only the JSON object. No commentary.
"""


class CrossTickerAgent(Agent):
    """Explains a price move via own news and indirect peer correlation."""

    name = "cross_ticker_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}
        # Move below this absolute % is treated as "flat".
        self._flat_threshold = float(self._config.get("flat_threshold_pct", 0.5))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _direction(move_pct: float, flat_threshold: float) -> str:
        if move_pct >= flat_threshold:
            return "rise"
        if move_pct <= -flat_threshold:
            return "loss"
        return "flat"

    @staticmethod
    def _news_lines(symbol: str, items: List[Any], limit: int = 5) -> List[str]:
        lines = []
        for it in items[:limit]:
            headline = getattr(it, "headline", "") or ""
            sentiment = getattr(it, "sentiment_label", "") or ""
            source = getattr(it, "source", "") or ""
            tag = f" [{sentiment}]" if sentiment else ""
            tag += f" ({source})" if source else ""
            if headline:
                lines.append(f"  - {symbol}: {headline}{tag}")
        return lines

    def _format_for_llm(
        self,
        symbol: str,
        move_pct: float,
        own_news: List[Any],
        peer_news: Dict[str, List[Any]],
        holding_note: str = "",
    ) -> str:
        lines = [
            f"Stock: {symbol}",
            f"Intraday move: {move_pct:+.2f}%",
        ]
        if holding_note:
            lines.append(f"Position: {holding_note}")
        lines += [
            "",
            f"{symbol} own news:",
        ]
        own_lines = self._news_lines(symbol, own_news)
        lines.extend(own_lines or ["  (none)"])

        lines.append("")
        lines.append("Peer news (competitors / suppliers / customers):")
        any_peer = False
        for peer, items in peer_news.items():
            peer_lines = self._news_lines(peer, items)
            if peer_lines:
                any_peer = True
                lines.extend(peer_lines)
        if not any_peer:
            lines.append("  (none)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM path
    # ------------------------------------------------------------------

    def _correlate_with_llm(
        self,
        symbol: str,
        move_pct: float,
        own_news: List[Any],
        peer_news: Dict[str, List[Any]],
        llm: Any,
        holding_note: str = "",
    ) -> Dict[str, Any]:
        user_prompt = self._format_for_llm(
            symbol, move_pct, own_news, peer_news, holding_note
        )
        try:
            return llm.invoke_json(
                system_prompt=_CROSS_TICKER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
            )
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Rule-based path
    # ------------------------------------------------------------------

    def _correlate_rule_based(
        self,
        symbol: str,
        move_pct: float,
        own_news: List[Any],
        peer_news: Dict[str, List[Any]],
    ) -> Dict[str, Any]:
        direction = self._direction(move_pct, self._flat_threshold)
        rose = direction == "rise"

        def sentiment_of(items: List[Any]) -> List[str]:
            return [(getattr(it, "sentiment_label", "") or "").lower() for it in items]

        # 1. Own news explains the move if its sentiment matches the direction.
        own_sent = sentiment_of(own_news)
        if own_news:
            if rose and "positive" in own_sent:
                top = next((it for it in own_news
                            if (getattr(it, "sentiment_label", "") or "").lower() == "positive"), own_news[0])
                return {
                    "news_synced": True, "explanation_type": "own_news",
                    "driver_ticker": symbol,
                    "driver_headline": getattr(top, "headline", ""),
                    "confidence": 0.7,
                    "reasoning": f"{symbol} rose on its own positive news.",
                }
            if not rose and "negative" in own_sent:
                top = next((it for it in own_news
                            if (getattr(it, "sentiment_label", "") or "").lower() == "negative"), own_news[0])
                return {
                    "news_synced": True, "explanation_type": "own_news",
                    "driver_ticker": symbol,
                    "driver_headline": getattr(top, "headline", ""),
                    "confidence": 0.7,
                    "reasoning": f"{symbol} fell on its own negative news.",
                }

        # 2. Peer spillover: a competitor's bad news while we rose (tailwind),
        #    or a competitor's good news while we fell (headwind).
        for peer, items in peer_news.items():
            sents = sentiment_of(items)
            if rose and "negative" in sents:
                top = next((it for it in items
                            if (getattr(it, "sentiment_label", "") or "").lower() == "negative"), items[0])
                return {
                    "news_synced": True, "explanation_type": "peer_spillover",
                    "driver_ticker": peer,
                    "driver_headline": getattr(top, "headline", ""),
                    "confidence": 0.6,
                    "reasoning": f"{symbol} rose as peer {peer} had negative news (spillover tailwind).",
                }
            if not rose and "positive" in sents:
                top = next((it for it in items
                            if (getattr(it, "sentiment_label", "") or "").lower() == "positive"), items[0])
                return {
                    "news_synced": True, "explanation_type": "peer_spillover",
                    "driver_ticker": peer,
                    "driver_headline": getattr(top, "headline", ""),
                    "confidence": 0.6,
                    "reasoning": f"{symbol} fell as peer {peer} had positive news (competitive headwind).",
                }

        # 3. Nothing explains it.
        return {
            "news_synced": False, "explanation_type": "unexplained",
            "driver_ticker": None, "driver_headline": "",
            "confidence": 0.4,
            "reasoning": f"No own or peer news explains {symbol}'s {move_pct:+.2f}% move.",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def correlate(
        self,
        symbol: str,
        move_pct: float,
        own_news: Optional[List[Any]] = None,
        peer_news: Optional[Dict[str, List[Any]]] = None,
        llm: Any = None,
        holding_note: str = "",
    ) -> CrossTickerSignal:
        own_news = own_news or []
        peer_news = peer_news or {}

        raw: Dict[str, Any] = {}
        if llm:
            raw = self._correlate_with_llm(
                symbol, move_pct, own_news, peer_news, llm, holding_note
            )
            if not raw or raw.get("_parse_error"):
                raw = {}
        if not raw:
            raw = self._correlate_rule_based(symbol, move_pct, own_news, peer_news)

        return CrossTickerSignal(
            symbol=symbol,
            move_direction=self._direction(move_pct, self._flat_threshold),
            move_pct=move_pct,
            news_synced=bool(raw.get("news_synced", False)),
            explanation_type=raw.get("explanation_type", "unexplained"),
            driver_ticker=raw.get("driver_ticker") or None,
            driver_headline=raw.get("driver_headline", "") or "",
            confidence=float(raw.get("confidence", 0.4) or 0.4),
            reasoning=raw.get("reasoning", ""),
        )

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            symbol = kwargs.get("symbol") or (ctx.input_data or {}).get("symbol", "UNKNOWN")
            move_pct = float(kwargs.get("move_pct", (ctx.input_data or {}).get("move_pct", 0.0)))
            own_news = kwargs.get("own_news", [])
            peer_news = kwargs.get("peer_news", {})
            llm = ctx.llm or self.get_llm_adapter()

            signal = self.correlate(symbol, move_pct, own_news, peer_news, llm=llm)
            payload = {"signal": signal.to_dict(), "raw_signal": signal}

            if DEBUG:
                print(f"\n[DEBUG] CrossTickerAgent | {symbol} ({move_pct:+.2f}%)")
                print(f"  Explanation: {signal.explanation_type} "
                      f"(driver={signal.driver_ticker}, synced={signal.news_synced})")

            completed = self._post_run()
            return self._result(
                ctx=ctx, success=True, started=started, completed=completed,
                payload=payload,
                checksum_parts=[symbol, signal.explanation_type, str(signal.confidence)],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx, success=False, started=started, completed=completed,
                payload={}, errors=[AgentError(code="CROSS_TICKER_ERROR", message=str(exc))],
            )
