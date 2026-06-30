"""ML Meta-Judge — trainable decision brain.

Consumes the flat feature vector from AgentFeatureBundle and produces
a JudgeDecision (BUY / SELL / HOLD with probabilities).

Phase 1: XGBoost classifier + quantile regression.
Phase 2: Online rolling-window retraining + Platt calibration.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import json
import os

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import (
    AgentFeatureBundle,
    JudgeDecision,
    TechnicalSignal,
    FundamentalSignal,
    SentimentSignal,
    EventSignal,
    RegimeSignal,
)


class JudgeAgent(Agent):
    """ML Meta-Judge that aggregates agent signals into a trading decision.

    Supports two modes:
      1. **Rule-based** (default) — weighted formula, no trained model needed.
      2. **ML-based** — loads a trained model (XGBoost/LightGBM) for inference.
    """

    name = "judge_agent"

    # Default weights for the rule-based fallback
    # Keys must match to_feature_dict() output of each signal class.
    _DEFAULT_WEIGHTS: Dict[str, float] = {
        # Technical (total 0.35)
        "tech_score":    0.15,
        "tech_trend":    0.10,
        "tech_macd":     0.05,
        "tech_breakout": 0.05,
        # Fundamental (total 0.22)
        "fund_score":    0.12,
        "fund_growth":   0.05,
        "fund_health":   0.05,
        # Sentiment (total 0.22)
        "sent_score":       0.10,
        "sent_net_ratio":   0.07,
        "sent_trend":       0.05,
        # Event (total 0.11)
        "evt_score":    0.06,
        "evt_gap_up":   0.03,
        "evt_earnings": 0.02,
        # Similarity (total 0.10)
        "sim_avg_return": 0.06,
        "sim_pos_rate":   0.04,
    }

    # Neutral resting value of each weighted feature. The rule-based score is
    # computed as a DEVIATION from these so a "nothing stands out" stock maps to
    # ~0.5 (not biased low). Directional 0-1 scores rest at 0.5; binary flags and
    # magnitude/return features rest at 0.0 (absent => no signal => no nudge).
    _NEUTRAL_VALUES: Dict[str, float] = {
        "tech_score":    0.5,
        "tech_trend":    0.5,
        "tech_macd":     0.5,
        "tech_breakout": 0.0,
        "fund_score":    0.5,
        "fund_growth":   0.5,
        "fund_health":   0.5,
        "sent_score":    0.5,
        "sent_net_ratio": 0.5,
        "sent_trend":    0.5,
        "evt_score":     0.0,
        "evt_gap_up":    0.0,
        "evt_earnings":  0.0,
        "sim_avg_return": 0.0,
        "sim_pos_rate":  0.0,
    }

    # Decision thresholds
    BUY_THRESHOLD = 0.65
    SELL_THRESHOLD = 0.35
    MIN_EXPECTED_RETURN = 0.015       # 1.5 %
    MAX_DOWNSIDE_RISK = 0.35          # symmetric with BUY_THRESHOLD (1 - 0.65)
    MAX_POSITION_SIZE_PCT = 0.02      # 2 % of capital

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        model_path: Optional[str] = None,
    ):
        super().__init__(name=self.name)
        self._config = config or {}
        self._model = None
        self._calibrator = None
        self._weights = dict(self._DEFAULT_WEIGHTS)

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def _load_model(self, path: str) -> None:
        """Load a serialised ML model (pickle / joblib)."""
        try:
            import joblib  # type: ignore
            self._model = joblib.load(path)
        except ImportError:
            import pickle
            with open(path, "rb") as f:
                self._model = pickle.load(f)

    def save_model(self, path: str) -> None:
        """Persist the current model."""
        if self._model is None:
            raise ValueError("No model to save")
        try:
            import joblib  # type: ignore
            joblib.dump(self._model, path)
        except ImportError:
            import pickle
            with open(path, "wb") as f:
                pickle.dump(self._model, f)

    def update_weights(self, new_weights: Dict[str, float]) -> None:
        """Hot-update rule-based weights (from self-reflection loop)."""
        self._weights.update(new_weights)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score_rule_based(self, features: Dict[str, Any]) -> float:
        """Weighted deviation-from-neutral → probability proxy in [0, 1].

        Each feature contributes ``weight * (value - neutral)`` so a stock whose
        signals all sit at their neutral resting points scores ~0.5. Bullish
        features push the score up, bearish features push it down, symmetrically.
        """
        total = 0.0
        weight_sum = 0.0
        for key, w in self._weights.items():
            val = features.get(key)
            if val is not None and isinstance(val, (int, float)):
                neutral = self._NEUTRAL_VALUES.get(key, 0.5)
                total += w * (float(val) - neutral)
                weight_sum += w
        if weight_sum == 0:
            return 0.5
        return max(0.0, min(1.0, 0.5 + total / weight_sum))

    def _score_ml(self, features: Dict[str, Any]) -> float:
        """Use the trained ML model to predict P(up_5d)."""
        import numpy as np  # type: ignore

        # Build ordered feature vector matching training schema
        numeric_keys = sorted(
            k for k, v in features.items()
            if isinstance(v, (int, float))
        )
        X = np.array([[features[k] for k in numeric_keys]])

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            # Assume binary: class 1 = up
            return float(proba[1]) if len(proba) > 1 else float(proba[0])
        else:
            # Regressor — treat output as raw probability
            return float(max(0.0, min(1.0, self._model.predict(X)[0])))

    # ------------------------------------------------------------------
    # Risk sizing
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_position_size(prob: float, risk: float) -> float:
        """Kelly-inspired position sizing capped at MAX_POSITION_SIZE."""
        if prob <= 0.5 or risk >= 1.0:
            return 0.0
        edge = prob - 0.5
        size = edge * 2 * JudgeAgent.MAX_POSITION_SIZE_PCT  # linear scale
        return round(min(size, JudgeAgent.MAX_POSITION_SIZE_PCT), 6)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def judge(self, bundle: AgentFeatureBundle) -> JudgeDecision:
        """Produce a final trading decision from the agent feature bundle."""
        features = bundle.to_flat_features()

        # Score
        if self._model is not None:
            prob_up = self._score_ml(features)
        else:
            prob_up = self._score_rule_based(features)

        prob_down = 1.0 - prob_up  # simplified; Phase 2 uses separate model
        expected_return = (prob_up - 0.5) * 0.10  # rough linear mapping

        # Decision logic
        regime = features.get("regime", "sideways")
        if (
            prob_up >= self.BUY_THRESHOLD
            and expected_return >= self.MIN_EXPECTED_RETURN
            and prob_down <= self.MAX_DOWNSIDE_RISK
            and regime != "bear_trend"
        ):
            decision = "BUY"
        elif prob_down >= self.BUY_THRESHOLD:
            decision = "SELL"
        else:
            decision = "HOLD"

        position_size = self._compute_position_size(prob_up, prob_down)

        return JudgeDecision(
            symbol=bundle.symbol,
            date=bundle.date,
            decision=decision,
            prob_up_5d=round(prob_up, 4),
            expected_return_5d=round(expected_return, 6),
            downside_risk_prob=round(prob_down, 4),
            confidence=round(prob_up if decision == "BUY" else (1 - prob_up), 4),
            position_size_pct=position_size,
            stop_loss_pct=-0.03,    # default; Phase 2 = ATR-based
            take_profit_pct=0.06,   # default; Phase 2 = ATR-based
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            bundle: AgentFeatureBundle = kwargs.get("bundle")  # type: ignore[assignment]
            if bundle is None:
                raise ValueError("JudgeAgent requires 'bundle' (AgentFeatureBundle) in kwargs")

            decision = self.judge(bundle)
            payload = {
                "decision": decision.decision,
                "prob_up_5d": decision.prob_up_5d,
                "expected_return_5d": decision.expected_return_5d,
                "downside_risk_prob": decision.downside_risk_prob,
                "confidence": decision.confidence,
                "position_size_pct": decision.position_size_pct,
                "raw_decision": decision,
            }

            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[
                    decision.symbol,
                    decision.date,
                    decision.decision,
                    str(decision.prob_up_5d),
                ],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="JUDGE_ERROR", message=str(exc))],
            )
