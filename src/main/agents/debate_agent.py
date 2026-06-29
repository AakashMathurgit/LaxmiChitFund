"""Debate Agent — evaluates Bull vs Bear arguments and produces a debate decision.

Also implements the Hybrid Decider logic:
  - Combines rule-based flow (JudgeDecision) with debate flow (DebateDecision)
  - Detects disagreement → forces HOLD when flows strongly conflict
  - Produces a final HybridDecision with weighted confidence

LLM path: sends both arguments to a judge prompt
Rule-based path: compares confidence scores and key point counts
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    DebateArgument,
    DebateContext,
    DebateDecision,
    HybridDecision,
    JudgeDecision,
)

DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")

_DEBATE_JUDGE_PROMPT = """\
You are a professional hedge fund investment committee chair.
Two analysts have presented opposing views on a stock.

BULL CASE:
{bull_argument}

BEAR CASE:
{bear_argument}

STOCK DATA:
{data_summary}

Evaluate both arguments and decide: BUY, SELL, or HOLD.
Consider the strength of evidence, not just opinion.

Output a JSON object:
{{
  "decision": "BUY" or "SELL" or "HOLD",
  "confidence": <0.0 to 1.0>,
  "winning_side": "bull" or "bear" or "neutral",
  "bull_strength": <0.0 to 1.0>,
  "bear_strength": <0.0 to 1.0>,
  "reasoning": "<2-3 sentence summary>"
}}

Only the JSON object. No commentary.
"""


class DebateAgent(Agent):
    """Evaluates bull vs bear arguments and combines with rule-based decision."""

    name = "debate_agent"

    # Hybrid decision weights
    RULE_WEIGHT = 0.6
    DEBATE_WEIGHT = 0.4
    DISAGREEMENT_THRESHOLD = 0.3  # If confidence gap > this, flag disagreement

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}
        rule_w = self._config.get("rule_weight")
        if rule_w is not None:
            self.RULE_WEIGHT = float(rule_w)
            self.DEBATE_WEIGHT = 1.0 - self.RULE_WEIGHT

    # ------------------------------------------------------------------
    # LLM debate evaluation
    # ------------------------------------------------------------------

    def _evaluate_with_llm(
        self,
        bull: DebateArgument,
        bear: DebateArgument,
        data_summary: str,
        llm: Any,
    ) -> Dict[str, Any]:
        import json as _json

        bull_text = f"Recommendation: {bull.recommendation} (confidence: {bull.confidence:.0%})\n"
        bull_text += "\n".join(f"  • {p}" for p in bull.key_points)
        bull_text += f"\n{bull.reasoning}"

        bear_text = f"Recommendation: {bear.recommendation} (confidence: {bear.confidence:.0%})\n"
        bear_text += "\n".join(f"  • {p}" for p in bear.key_points)
        bear_text += f"\n{bear.reasoning}"

        prompt = _DEBATE_JUDGE_PROMPT.format(
            bull_argument=bull_text,
            bear_argument=bear_text,
            data_summary=data_summary,
        )

        response = llm.invoke(
            system_prompt="You are an investment committee chair.",
            user_prompt=prompt,
        )
        try:
            return _json.loads(response)
        except (_json.JSONDecodeError, TypeError):
            return {}

    # ------------------------------------------------------------------
    # Rule-based debate evaluation
    # ------------------------------------------------------------------

    def _evaluate_rule_based(
        self,
        bull: DebateArgument,
        bear: DebateArgument,
    ) -> Dict[str, Any]:
        """Compare bull vs bear by confidence and argument count."""
        bull_score = bull.confidence * (1 + len(bull.key_points) * 0.1)
        bear_score = bear.confidence * (1 + len(bear.key_points) * 0.1)

        # Normalize
        total = bull_score + bear_score
        if total > 0:
            bull_strength = bull_score / total
            bear_strength = bear_score / total
        else:
            bull_strength = bear_strength = 0.5

        if bull_strength > 0.6:
            decision = "BUY"
            winning = "bull"
            confidence = bull.confidence
        elif bear_strength > 0.6:
            decision = "SELL" if bear.confidence > 0.7 else "HOLD"
            winning = "bear"
            confidence = bear.confidence
        else:
            decision = "HOLD"
            winning = "neutral"
            confidence = 0.5

        return {
            "decision": decision,
            "confidence": round(confidence, 4),
            "winning_side": winning,
            "bull_strength": round(bull_strength, 4),
            "bear_strength": round(bear_strength, 4),
            "reasoning": (
                f"Bull ({bull.confidence:.0%}) vs Bear ({bear.confidence:.0%}). "
                f"{winning.title()} case stronger with {len(bull.key_points if winning == 'bull' else bear.key_points)} key points."
            ),
        }

    # ------------------------------------------------------------------
    # Debate evaluation
    # ------------------------------------------------------------------

    def evaluate_debate(
        self,
        bull: DebateArgument,
        bear: DebateArgument,
        symbol: str = "",
        date: str = "",
        data_summary: str = "",
        llm: Any = None,
        debate_context: Optional[DebateContext] = None,
    ) -> DebateDecision:
        """Evaluate bull vs bear arguments and produce a debate decision."""
        # Prefer DebateContext for rich data summary
        effective_summary = data_summary
        if not effective_summary and debate_context is not None:
            effective_summary = debate_context.format_for_llm()
        if not symbol and debate_context is not None:
            symbol = debate_context.symbol
        if not date and debate_context is not None:
            date = debate_context.date

        if llm and effective_summary:
            raw = self._evaluate_with_llm(bull, bear, effective_summary, llm)
            if not raw:
                raw = self._evaluate_rule_based(bull, bear)
        else:
            raw = self._evaluate_rule_based(bull, bear)

        return DebateDecision(
            symbol=symbol,
            date=date,
            decision=raw.get("decision", "HOLD"),
            confidence=float(raw.get("confidence", 0.5)),
            winning_side=raw.get("winning_side", "neutral"),
            bull_strength=float(raw.get("bull_strength", 0.5)),
            bear_strength=float(raw.get("bear_strength", 0.5)),
            reasoning=raw.get("reasoning", ""),
        )

    # ------------------------------------------------------------------
    # Hybrid decision: combine rule-based + debate flows
    # ------------------------------------------------------------------

    def make_hybrid_decision(
        self,
        rule_decision: JudgeDecision,
        debate_decision: DebateDecision,
    ) -> HybridDecision:
        """Combine rule-based and debate-based decisions.

        - If both agree: use combined confidence (boosted)
        - If they disagree strongly: output HOLD (disagreement detected)
        - Otherwise: weighted average
        """
        rule_d = rule_decision.decision
        debate_d = debate_decision.decision

        # Map decisions to numeric for weighted average
        decision_map = {"BUY": 1.0, "HOLD": 0.5, "SELL": 0.0}
        rule_score = decision_map.get(rule_d, 0.5) * rule_decision.confidence
        debate_score = decision_map.get(debate_d, 0.5) * debate_decision.confidence

        weighted = (
            rule_score * self.RULE_WEIGHT
            + debate_score * self.DEBATE_WEIGHT
        )

        agreement = rule_d == debate_d
        disagreement_action = None

        # Strong disagreement: one says BUY, other says SELL
        strong_disagree = (
            (rule_d == "BUY" and debate_d == "SELL")
            or (rule_d == "SELL" and debate_d == "BUY")
        )
        if strong_disagree:
            final_decision = "HOLD"
            final_confidence = min(rule_decision.confidence, debate_decision.confidence) * 0.5
            disagreement_action = "HOLD — strong disagreement between rule and debate flows"
        elif agreement:
            final_decision = rule_d
            # Boost confidence when both agree
            final_confidence = min(
                0.95,
                max(rule_decision.confidence, debate_decision.confidence) * 1.1,
            )
        else:
            # Mild disagreement — go with weighted
            if weighted > 0.65:
                final_decision = "BUY"
            elif weighted < 0.35:
                final_decision = "SELL"
            else:
                final_decision = "HOLD"
            final_confidence = (
                rule_decision.confidence * self.RULE_WEIGHT
                + debate_decision.confidence * self.DEBATE_WEIGHT
            )

        reasoning_parts = [f"Rule: {rule_d} ({rule_decision.confidence:.0%})"]
        reasoning_parts.append(f"Debate: {debate_d} ({debate_decision.confidence:.0%})")
        if disagreement_action:
            reasoning_parts.append(disagreement_action)
        reasoning_parts.append(f"Final: {final_decision} ({final_confidence:.0%})")

        return HybridDecision(
            symbol=rule_decision.symbol,
            date=rule_decision.date,
            final_decision=final_decision,
            final_confidence=round(final_confidence, 4),
            rule_decision=rule_d,
            rule_confidence=rule_decision.confidence,
            debate_decision=debate_d,
            debate_confidence=debate_decision.confidence,
            agreement=agreement,
            disagreement_action=disagreement_action,
            reasoning=" | ".join(reasoning_parts),
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            bull: Optional[DebateArgument] = kwargs.get("bull_argument")
            bear: Optional[DebateArgument] = kwargs.get("bear_argument")
            symbol = kwargs.get("symbol", "")
            date = kwargs.get("date", "")

            if bull is None or bear is None:
                raise ValueError("DebateAgent requires 'bull_argument' and 'bear_argument'")

            llm = ctx.llm or self.get_llm_adapter()
            data_summary = kwargs.get("data_summary", "")

            debate_decision = self.evaluate_debate(
                bull=bull, bear=bear,
                symbol=symbol, date=date,
                data_summary=data_summary, llm=llm,
            )

            payload = {
                "debate_decision": debate_decision.to_dict(),
                "raw_decision": debate_decision,
                "bull_argument": bull.to_dict(),
                "bear_argument": bear.to_dict(),
            }

            if DEBUG:
                print(f"\n[DEBUG] DebateAgent | {symbol}")
                print(f"  Bull: {bull.recommendation} ({bull.confidence:.0%})")
                print(f"  Bear: {bear.recommendation} ({bear.confidence:.0%})")
                print(f"  Debate Decision: {debate_decision.decision} ({debate_decision.confidence:.0%})")
                print(f"  Winner: {debate_decision.winning_side}")

            completed = self._post_run()
            return self._result(
                ctx=ctx, success=True, started=started, completed=completed,
                payload=payload,
                checksum_parts=[debate_decision.decision, str(debate_decision.confidence)],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx, success=False, started=started, completed=completed,
                payload={}, errors=[AgentError(code="DEBATE_ERROR", message=str(exc))],
            )
