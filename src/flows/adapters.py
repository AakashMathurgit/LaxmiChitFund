"""Concrete flow adapters for the 5 LCF pipelines.

Each adapter runs its existing entry script in an isolated subprocess (so a hung
flow can't wedge the scheduler), then parses that script's result JSON into a
normalized FlowResult/Decision list for the combined store + conflict checks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from .base import Flow, FlowResult, Decision

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _decisions_from_judge(data: Dict[str, Any], horizon: str) -> List[Decision]:
    out: List[Decision] = []
    for r in (data or {}).get("results", []):
        if "error" in r:
            continue
        jd = r.get("judge_decision", {}).get("payload", {})
        decision = jd.get("decision")
        if not decision:
            continue
        out.append(Decision(
            symbol=r.get("symbol", "?"),
            action=decision,
            confidence=float(jd.get("confidence", 0.0) or 0.0),
            horizon=horizon,
        ))
    return out


class IntradayUSFlow(Flow):
    name = "us-intraday"
    horizon = "intraday"
    cadence = "4m"
    market_hours_only = True

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--mode", opts.get("mode", "adaptive")]
        if opts.get("top"):
            args += ["--top", str(opts["top"])]
        if opts.get("auto_trade") is False:
            args.append("--no-auto-trade")
        try:
            rc = self._run_script("us_intraday_tracker/run_us_intraday.py", args, timeout=300)
            data = self._load_json(os.path.join(_ROOT, "us_intraday_tracker", "intraday_results.json")) or {}
            decisions = [
                Decision(
                    symbol=r.get("symbol", "?"),
                    action=r.get("intraday_decision", {}).get("final_decision", "HOLD"),
                    confidence=float(r.get("intraday_decision", {}).get("final_confidence", 0.0) or 0.0),
                    horizon=self.horizon,
                    detail=r.get("cross_ticker", {}).get("explanation_type", ""),
                )
                for r in data.get("results", []) if "error" not in r
            ]
            return FlowResult(self.name, ts, ok=(rc == 0), decisions=decisions,
                              summary=f"{data.get('buy_count', 0)} BUY / {data.get('sell_count', 0)} SELL")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


def _swing_account_env() -> Dict[str, str]:
    """Resolve the SWING (Account-2) Alpaca keys and expose them to the swing
    subprocess as the standard ALPACA_* vars, so AlpacaBroker trades Account 2.

    Source order: ALPACA_SWING_* env (cloud secrets) -> credentials.yaml
    `alpaca_swing:` (local). If neither is set, returns {} and the swing flow
    falls back to the default account (with a warning logged by the broker).
    """
    key = os.environ.get("ALPACA_SWING_KEY_ID", "")
    secret = os.environ.get("ALPACA_SWING_SECRET_KEY", "")
    endpoint = os.environ.get("ALPACA_SWING_ENDPOINT", "")
    if not (key and secret):
        try:
            import yaml
            creds_path = os.path.join(_ROOT, "credentials.yaml")
            if os.path.exists(creds_path):
                with open(creds_path, "r", encoding="utf-8") as f:
                    creds = yaml.safe_load(f) or {}
                sw = creds.get("alpaca_swing", {}) or {}
                key = key or sw.get("key_id", "")
                secret = secret or sw.get("secret_key", "")
                endpoint = endpoint or sw.get("endpoint", "")
        except Exception:
            pass
    env: Dict[str, str] = {}
    if key and secret:
        env["ALPACA_KEY_ID"] = key
        env["ALPACA_SECRET_KEY"] = secret
        if endpoint:
            env["ALPACA_ENDPOINT"] = endpoint
    return env


class USSwingFlow(Flow):
    name = "us-swing"
    horizon = "swing"
    cadence = "30m"
    market_hours_only = True

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--mode", opts.get("mode", "value")]
        if opts.get("top"):
            args += ["--top", str(opts["top"])]
        if opts.get("auto_trade") is False:
            args.append("--no-auto-trade")
        try:
            # Route this subprocess at the SWING Alpaca account (Account 2).
            rc = self._run_script("us_swing_tracker/run_us_swing.py", args,
                                  timeout=1500, env=_swing_account_env())
            data = self._load_json(os.path.join(_ROOT, "us_swing_tracker", "swing_results.json")) or {}
            decisions = [
                Decision(
                    symbol=r.get("symbol", "?"),
                    action=r.get("swing_decision", {}).get("final_decision", "HOLD"),
                    confidence=float(r.get("swing_decision", {}).get("final_confidence", 0.0) or 0.0),
                    horizon=self.horizon,
                    detail=r.get("swing_decision", {}).get("note", ""),
                )
                for r in data.get("results", []) if "error" not in r and "swing_decision" in r
            ]
            return FlowResult(self.name, ts, ok=(rc == 0), decisions=decisions,
                              summary=f"{data.get('buy_count', 0)} BUY / {data.get('sell_count', 0)} SELL")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


class USDailyFlow(Flow):
    name = "us-daily"
    horizon = "swing"
    cadence = "daily"

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--mode", opts.get("mode", "adaptive"), "--stocks", str(opts.get("stocks", 10))]
        if opts.get("auto_trade"):
            args.append("--auto-trade")
        try:
            rc = self._run_script("us_stock_tracker/run_us_pipeline.py", args, timeout=900)
            data = self._load_json(os.path.join(_ROOT, "us_stock_tracker", "analysis_results.json")) or {}
            decisions = _decisions_from_judge(data, self.horizon)
            return FlowResult(self.name, ts, ok=(rc == 0), decisions=decisions,
                              summary=f"{data.get('buy_count', 0)} BUY / {data.get('sell_count', 0)} SELL")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


class INDailyFlow(Flow):
    name = "in-daily"
    horizon = "swing"
    cadence = "daily"

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--mode", opts.get("mode", "adaptive"), "--stocks", str(opts.get("stocks", 10))]
        if opts.get("auto_trade"):
            args.append("--auto-trade")
        try:
            rc = self._run_script("news_stock_tracker/run_orchestrator_pipeline.py", args, timeout=900)
            data = self._load_json(os.path.join(_ROOT, "news_stock_tracker", "analysis_results.json")) or {}
            decisions = _decisions_from_judge(data, self.horizon)
            return FlowResult(self.name, ts, ok=(rc == 0), decisions=decisions,
                              summary=f"{data.get('buy_count', 0)} BUY / {data.get('sell_count', 0)} SELL")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


class PredictUSFlow(Flow):
    name = "us-predict"
    horizon = "forecast"
    cadence = "weekly"

    _OUTLOOK = {"bullish": "BUY", "bearish": "SELL", "neutral": "HOLD"}

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--mode", opts.get("mode", "adaptive")]
        try:
            rc = self._run_script("us_stock_tracker/run_us_prediction.py", args, timeout=900)
            data = self._load_json(os.path.join(_ROOT, "us_stock_tracker", "prediction_results.json")) or {}
            decisions = []
            for r in data.get("results", []):
                if "error" in r:
                    continue
                outlook = (r.get("overall_outlook") or "").lower()
                decisions.append(Decision(
                    symbol=r.get("symbol", "?"),
                    action=self._OUTLOOK.get(outlook, "HOLD"),
                    confidence=float(r.get("overall_confidence", 0.0) or 0.0),
                    horizon=self.horizon,
                    detail=outlook,
                ))
            return FlowResult(self.name, ts, ok=(rc == 0), decisions=decisions,
                              summary=f"{len(decisions)} forecasts")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


class CompareFlow(Flow):
    name = "compare"
    horizon = "report"
    cadence = "weekly"

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        try:
            rc = self._run_script("analytics/run_strategy_comparison.py", [], timeout=300)
            data = self._load_json(os.path.join(_ROOT, "data", "strategy_comparison.json")) or {}
            comp = data.get("comparison", {})
            m = comp.get("metrics", {})
            winner = data.get("analysis", {}).get("winner") or comp.get("winner") or "?"
            parts = [f"{n}:${m[n]['total_pl']:+,.0f}" for n in ("intraday", "swing") if n in m]
            return FlowResult(self.name, ts, ok=(rc == 0),
                              summary=f"winner={winner} | " + " ".join(parts))
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))


class AdvisorFlow(Flow):
    name = "advise"
    horizon = "long_term"
    cadence = "monthly"

    def run(self, rt: Any, **opts) -> FlowResult:
        ts = self._now()
        args = ["--report", opts.get("report", "monthly")]
        try:
            rc = self._run_script("advisor_main.py", args, timeout=1200)
            # Advisor writes a report under data/advisor_reports; decisions are
            # holdings recommendations (not directly comparable to trades), so we
            # report success/summary only here.
            return FlowResult(self.name, ts, ok=(rc == 0),
                              summary="advisor report generated" if rc == 0 else "advisor failed")
        except Exception as e:
            return FlowResult(self.name, ts, ok=False, error=str(e))
