"""Training & Backtesting Strategy for LCF.

This module defines the approach for:
  1. Collecting labeled training data from historical runs
  2. Optimizing the 180+ tunable parameters
  3. Training the ML Judge model
  4. Walk-forward backtesting
  5. Parameter sensitivity analysis

NOT an executable script ΓÇö this is a design document + utility functions
for building the training pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# =========================================================================
# PHASE 1: Data Collection ΓÇö Build Labeled Dataset
# =========================================================================
#
# Goal: For each historical stock-day, generate a feature vector and
#       record the actual 5-day forward return as the label.
#
# Process:
#   1. Pick universe: NIFTY 50 stocks (liquid, good data coverage)
#   2. Date range: last 2 years of trading days (~500 days)
#   3. For each stock-day:
#      a. Build StockDataContext from Yahoo Finance historical data
#      b. Run all 4 analysis agents ΓåÆ TechnicalSignal, FundamentalSignal, etc.
#      c. Flatten to AgentFeatureBundle.to_flat_features() ΓåÆ 20 floats
#      d. Compute actual_return_5d = (close[t+5] - close[t]) / close[t]
#      e. Label: BUY if return > +3%, SELL if return < -3%, HOLD otherwise
#      f. Store: (features, return_5d, label, regime, date, symbol)
#
# Output: labeled_data.parquet with ~25,000 rows (50 stocks * 500 days)
#
# Key: Run agents with SAME parameters as production to ensure consistency.


@dataclass
class TrainingExample:
    """A single labeled training example."""
    symbol: str
    date: str
    features: Dict[str, float]      # from AgentFeatureBundle.to_flat_features()
    actual_return_5d: float          # label: forward return
    label: str                       # BUY / SELL / HOLD
    regime: str                      # market regime at the time
    actual_return_10d: Optional[float] = None
    max_drawdown_5d: Optional[float] = None
    hit_3pct_up: bool = False        # did it go +3% within 5 days?
    hit_3pct_down: bool = False      # did it go -3% within 5 days?


# =========================================================================
# PHASE 2: Parameter Optimization ΓÇö What to Tune and How
# =========================================================================
#
# Three tiers of parameters, each with different optimization strategies:
#
# TIER 1: Judge Weights (15 params) ΓÇö MOST IMPACTFUL
#   Method: Bayesian optimization (Optuna)
#   Objective: Maximize Sharpe ratio on validation set
#   Constraints: Weights must sum to 1.0, each in [0, 0.30]
#   Search space: ~15 continuous variables
#
# TIER 2: Decision Thresholds (10 params) ΓÇö HIGH IMPACT
#   Method: Grid search (small discrete space)
#   Params: buy_threshold, sell_threshold, min_expected_return,
#           debate rule_weight, hybrid thresholds
#   Grid: buy_threshold in [0.55, 0.60, 0.65, 0.70, 0.75]
#         sell_threshold in [0.25, 0.30, 0.35, 0.40, 0.45]
#         rule_weight in [0.4, 0.5, 0.6, 0.7, 0.8]
#
# TIER 3: Agent-Level Params (150+ params) ΓÇö MODERATE IMPACT
#   Method: Sensitivity analysis first, then tune only sensitive ones
#   Approach: Vary each param ┬▒20%, measure decision change rate
#   Only optimize params with >5% decision change rate


# Tier 1: Parameters for Bayesian optimization
TIER1_PARAMS = {
    "judge.weights.tech_score":    (0.05, 0.30),
    "judge.weights.tech_trend":    (0.02, 0.20),
    "judge.weights.tech_macd":     (0.01, 0.15),
    "judge.weights.tech_breakout": (0.01, 0.15),
    "judge.weights.fund_score":    (0.05, 0.25),
    "judge.weights.fund_growth":   (0.01, 0.15),
    "judge.weights.fund_health":   (0.01, 0.15),
    "judge.weights.sent_score":    (0.02, 0.20),
    "judge.weights.sent_net_ratio": (0.01, 0.15),
    "judge.weights.sent_trend":    (0.01, 0.10),
    "judge.weights.evt_score":     (0.01, 0.15),
    "judge.weights.evt_gap_up":    (0.01, 0.10),
    "judge.weights.evt_earnings":  (0.01, 0.10),
    "judge.weights.sim_avg_return": (0.01, 0.15),
    "judge.weights.sim_pos_rate":  (0.01, 0.10),
}

# Tier 2: Parameters for grid search
TIER2_PARAMS = {
    "judge.buy_threshold":      [0.55, 0.60, 0.65, 0.70, 0.75],
    "judge.sell_threshold":     [0.25, 0.30, 0.35, 0.40, 0.45],
    "judge.min_expected_return": [0.005, 0.010, 0.015, 0.020, 0.025],
    "debate.rule_weight":       [0.4, 0.5, 0.6, 0.7, 0.8],
    "debate.agreement_boost":   [1.0, 1.05, 1.10, 1.15, 1.20],
    "risk.regime_scale_bear":   [0.5, 0.6, 0.7, 0.8],
    "risk.vol_scale_high":      [0.4, 0.5, 0.6, 0.7],
    "trade.atr_stop_multiplier": [1.0, 1.5, 2.0, 2.5],
    "trade.default_risk_reward": [1.5, 2.0, 2.5, 3.0],
    "trade.max_risk_per_trade_pct": [0.005, 0.01, 0.015, 0.02],
}


# =========================================================================
# PHASE 3: ML Judge Training
# =========================================================================
#
# Model: XGBoost binary classifier (predict P(return > +3% in 5 days))
#
# Features (20 inputs):
#   tech_score, tech_rsi, tech_macd, tech_volatility, tech_breakout, tech_trend
#   fund_score, fund_valuation, fund_growth, fund_health
#   sent_score, sent_net_ratio, sent_trend, sent_confidence
#   evt_score, evt_earnings, evt_risk, evt_gap_up, evt_gap_down
#   regime_confidence
#
# Label: 1 if actual_return_5d > 0.03, else 0
#
# Training Protocol:
#   1. Walk-forward split (NOT random split ΓÇö time-series data!)
#      - Train: months 1-12
#      - Validate: months 13-15
#      - Test: months 16-18
#      - Then shift forward by 3 months and repeat
#
#   2. Hyperparameters (tune via Optuna):
#      - n_estimators: [100, 500]
#      - max_depth: [3, 8]
#      - learning_rate: [0.01, 0.3]
#      - min_child_weight: [1, 10]
#      - subsample: [0.6, 1.0]
#      - colsample_bytree: [0.6, 1.0]
#
#   3. Calibration: Platt scaling on validation set
#      - Ensures P(up) is well-calibrated (not just ranked correctly)
#
#   4. Output: model.joblib + calibrator.joblib
#      - Load via JudgeAgent(model_path="model.joblib")


@dataclass
class WalkForwardSplit:
    """A single walk-forward train/validate/test split."""
    train_start: str
    train_end: str
    validate_start: str
    validate_end: str
    test_start: str
    test_end: str


def generate_walk_forward_splits(
    start_date: str = "2024-01-01",
    end_date: str = "2026-03-01",
    train_months: int = 12,
    validate_months: int = 3,
    test_months: int = 3,
    step_months: int = 3,
) -> List[WalkForwardSplit]:
    """Generate walk-forward time-series splits.

    Example with defaults:
      Split 1: Train Jan24-Dec24 | Val Jan25-Mar25 | Test Apr25-Jun25
      Split 2: Train Apr24-Mar25 | Val Apr25-Jun25 | Test Jul25-Sep25
      Split 3: Train Jul24-Jun25 | Val Jul25-Sep25 | Test Oct25-Dec25
    """
    from datetime import datetime
    from dateutil.relativedelta import relativedelta

    splits = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    while True:
        train_start = current
        train_end = train_start + relativedelta(months=train_months) - relativedelta(days=1)
        val_start = train_end + relativedelta(days=1)
        val_end = val_start + relativedelta(months=validate_months) - relativedelta(days=1)
        test_start = val_end + relativedelta(days=1)
        test_end = test_start + relativedelta(months=test_months) - relativedelta(days=1)

        if test_end > end:
            break

        splits.append(WalkForwardSplit(
            train_start=train_start.strftime("%Y-%m-%d"),
            train_end=train_end.strftime("%Y-%m-%d"),
            validate_start=val_start.strftime("%Y-%m-%d"),
            validate_end=val_end.strftime("%Y-%m-%d"),
            test_start=test_start.strftime("%Y-%m-%d"),
            test_end=test_end.strftime("%Y-%m-%d"),
        ))

        current += relativedelta(months=step_months)

    return splits


# =========================================================================
# PHASE 4: Backtesting Framework
# =========================================================================
#
# Walk-forward backtest simulating real trading:
#
#   For each test period:
#     1. Load model trained on corresponding train period
#     2. Load TuningConfig optimized on validation period
#     3. For each trading day:
#        a. Run full pipeline (regime ΓåÆ analysis ΓåÆ judge ΓåÆ debate ΓåÆ trade plan)
#        b. Execute trades via MockBroker
#        c. Update portfolio (PortfolioManager)
#        d. Manage open positions (PositionManagementAgent)
#     4. Record daily portfolio value, drawdown, returns
#
#   Metrics:
#     - Total return
#     - Annualized return
#     - Sharpe ratio (risk-free rate = 6% for India)
#     - Max drawdown
#     - Win rate
#     - Profit factor (gross profit / gross loss)
#     - Average R:R realized
#     - Trade count per month
#     - Regime-conditional performance (bull vs bear vs sideways)


@dataclass
class BacktestMetrics:
    """Aggregate metrics from a backtest run."""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_holding_days: float = 0.0
    avg_return_per_trade: float = 0.0

    # Regime breakdown
    return_bull: float = 0.0
    return_bear: float = 0.0
    return_sideways: float = 0.0
    trades_bull: int = 0
    trades_bear: int = 0
    trades_sideways: int = 0

    def summary(self) -> str:
        return (
            f"Return: {self.total_return:+.1%} | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"MaxDD: {self.max_drawdown:.1%} | "
            f"WinRate: {self.win_rate:.0%} | "
            f"Trades: {self.total_trades}"
        )


# =========================================================================
# PHASE 5: Sensitivity Analysis
# =========================================================================
#
# For each of the 180+ parameters:
#   1. Hold all other params at default
#   2. Vary this param at -20%, -10%, default, +10%, +20%
#   3. Run backtest on validation set
#   4. Measure: decision change rate, Sharpe change, return change
#   5. Rank parameters by impact
#
# Only parameters with decision_change_rate > 5% are worth optimizing.
# The rest can stay at defaults.


@dataclass
class SensitivityResult:
    """Result of varying a single parameter."""
    param_name: str
    default_value: float
    tested_values: List[float]
    sharpe_at_each: List[float]
    return_at_each: List[float]
    decision_change_rate: float      # % of decisions that changed

    @property
    def is_sensitive(self) -> bool:
        """True if varying this param changes >5% of decisions."""
        return self.decision_change_rate > 0.05


# =========================================================================
# TRAINING PIPELINE ΓÇö Step by Step
# =========================================================================
#
# Step 1: Data Collection (run once, ~2 hours)
#   python -m scripts.collect_training_data \
#     --symbols NIFTY50 \
#     --start 2024-01-01 \
#     --end 2026-03-01 \
#     --output data/training/labeled_data.parquet
#
# Step 2: Sensitivity Analysis (run once, ~4 hours)
#   python -m scripts.sensitivity_analysis \
#     --data data/training/labeled_data.parquet \
#     --output data/training/sensitivity_report.json
#
# Step 3: Bayesian Optimization of Judge Weights (Tier 1)
#   python -m scripts.optimize_weights \
#     --data data/training/labeled_data.parquet \
#     --n-trials 200 \
#     --output tuning_params_optimized.yaml
#
# Step 4: Grid Search of Decision Thresholds (Tier 2)
#   python -m scripts.grid_search_thresholds \
#     --data data/training/labeled_data.parquet \
#     --output tuning_params_optimized.yaml  (append)
#
# Step 5: Train XGBoost Judge Model
#   python -m scripts.train_judge_model \
#     --data data/training/labeled_data.parquet \
#     --walk-forward \
#     --output models/judge_model.joblib
#
# Step 6: Walk-Forward Backtest
#   python -m scripts.backtest \
#     --data data/training/labeled_data.parquet \
#     --config tuning_params_optimized.yaml \
#     --model models/judge_model.joblib \
#     --output data/backtest/results.json
#
# Step 7: Review & Deploy
#   - Compare backtest metrics vs baseline (default params)
#   - If Sharpe improves > 0.1 and drawdown doesn't worsen > 2%:
#     - Copy tuning_params_optimized.yaml ΓåÆ tuning_params.yaml
#     - Copy judge_model.joblib ΓåÆ config path
#   - Monitor live performance for 2 weeks before full rollout
