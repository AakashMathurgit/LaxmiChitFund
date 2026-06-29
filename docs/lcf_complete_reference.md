# LCF — Complete System Reference

> **LCF: Large-Cap Framework** — A Multi-Agent AI Trading System for Indian Stocks  
> 14 Agents | 5 Controllers | 3 Data Providers | 180+ Tunable Parameters | 6 Trading Modes

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture Layers](#2-architecture-layers)
3. [Agent Base Framework](#3-agent-base-framework)
4. [Analysis Agents (Layer 3)](#4-analysis-agents-layer-3)
   - 4.1 TechnicalAgent
   - 4.2 FundamentalAgent
   - 4.3 SentimentAgent
   - 4.4 EventAgent
5. [Decision Agents (Layer 4)](#5-decision-agents-layer-4)
   - 5.1 JudgeAgent (Rule-Based Path)
   - 5.2 BullAgent (AI Debate)
   - 5.3 BearAgent (AI Debate)
   - 5.4 DebateAgent (Evaluator + Hybrid Combiner)
6. [Execution Agents (Layer 5)](#6-execution-agents-layer-5)
   - 6.1 RegimeDetectorAgent
   - 6.2 TradePlannerAgent
   - 6.3 RiskManagerAgent
   - 6.4 PositionManagementAgent
7. [Pre-Pipeline Agents](#7-pre-pipeline-agents)
   - 7.1 StockDiscoveryAgent
   - 7.2 BacktestAgent
8. [Controllers](#8-controllers)
   - 8.1 DataProcessor (Multi-Provider)
   - 8.2 PatternStore (RAG)
   - 8.3 TradeMemory
   - 8.4 PortfolioManager
   - 8.5 TradeExecutor
   - 8.6 MessageController (WhatsApp)
   - 8.7 PerformanceTracker
9. [Data Models & Signal Structures](#9-data-models--signal-structures)
10. [Data Providers](#10-data-providers)
11. [Pipeline Orchestrator](#11-pipeline-orchestrator)
12. [Trading Modes & Tuning Parameters](#12-trading-modes--tuning-parameters)
13. [Response Structures & Output Formats](#13-response-structures--output-formats)
14. [LLM Integration](#14-llm-integration)
15. [Configuration Files](#15-configuration-files)
16. [File Structure Map](#16-file-structure-map)

---

## 1. System Overview

LCF is a fully autonomous AI trading system that:

- **Auto-discovers** stocks from the NIFTY 50 universe using volume spikes, breakouts, and news movers
- **Analyzes** each stock through **14 specialized agents** running technical, fundamental, sentiment, and event analysis
- **Debates** every decision through a **Bull vs Bear AI debate** with real market data, RAG, and trade memory
- **Combines** three independent decision paths (rule-based, AI debate, XGBoost ML) into a single HybridDecision
- **Manages** a portfolio with auto-entry, auto-exit, stop-loss monitoring, and P&L tracking
- **Learns** from every trade via PatternStore (vector similarity) and TradeMemory (append-only JSONL)
- **Alerts** via WhatsApp with rich trade messages including full analysis breakdown

**Tech Stack:** Python 3.14, XGBoost, GPT-4.1 (Azure OpenAI), NumPy, Yahoo Finance API, NSE Corporate, Google News RSS, pywhatkit

---

## 2. Architecture Layers

The pipeline processes each stock through 5 sequential layers:

| Layer | Name | Components | Latency |
|-------|------|-----------|---------|
| **1** | **DATA** | Yahoo Finance + NSE Corporate + Google News RSS | ~3s |
| **2** | **REGIME** | RegimeDetectorAgent (NIFTY 50 + India VIX) | ~1s (once) |
| **3** | **ANALYSIS** | TechnicalAgent ∥ FundamentalAgent ∥ SentimentAgent ∥ EventAgent | ~1s |
| **4** | **DECISION** | JudgeAgent ∥ BullAgent + BearAgent → DebateAgent → HybridDecision | ~8s |
| **5** | **EXECUTION** | TradePlanner → RiskManager → TradeExecutor → PortfolioManager | ~0.5s |

**Total per stock: ~10 seconds**

### What Runs Once vs Per-Stock

| Scope | Components |
|-------|-----------|
| **Once per run** | Index data fetch (NIFTY 50), VIX fetch, RegimeDetectorAgent, StockDiscoveryAgent |
| **Per stock** | DataProcessor.build_stock_context(), all 4 analysis agents, JudgeAgent, debate flow, trade planner, risk manager |

### Parallel Execution Points

| Stage | Parallel Components |
|-------|-------------------|
| Data fetch | YahooFinance ∥ NSE Corporate ∥ RSS News (per stock) |
| Analysis | TechnicalAgent ∥ FundamentalAgent ∥ SentimentAgent ∥ EventAgent |
| Decision | JudgeAgent (~1ms) ∥ AI Debate (~8s) ∥ XGBoost ML (~1ms) |

---

## 3. Agent Base Framework

All agents inherit from a common `Agent` abstract base class with built-in:

### AgentContext (Runtime Context)
```
AgentContext:
  run_id: str              — UUID for this execution
  rules_version: str       — Config version (e.g., "1.0.0")
  input_data: Dict         — Flat dict of stock data
  config: Dict             — Full config.yaml contents
  llm: LLMAdapter          — Injected GPT-4.1 adapter
  adapters: Dict           — Named adapter registry
  workflow_state: Dict     — LangGraph state
```

### AgentResult (Structured Output)
Every agent returns an `AgentResult` with:
```
AgentResult:
  success: bool                — True if agent executed without errors
  run_id: str                  — Matching run UUID
  rules_version: str           — Config version
  timing: TimingInfo           — { started_epoch_ms, completed_epoch_ms, duration_ms }
  errors: List[AgentError]     — Any errors with code, message, severity
  provenance_id: str           — Audit trail ID
  integrity: IntegrityInfo     — SHA256 checksum of payload
  payload: Dict                — Agent-specific output data
```

### Integrity & Provenance
- Every agent output includes a **SHA256 checksum** computed over the ordered payload values
- **TimingInfo** records exact start/end timestamps in epoch milliseconds
- **AgentError** captures errors with severity levels (INFO, WARNING, ERROR)

### LangGraph Integration
The base `Agent` class supports optional LangGraph workflow execution:
- `supports_langgraph()` — checks if LangGraph is available and enabled
- `build_workflow()` — creates a compiled StateGraph
- `execute_with_langgraph()` — runs the agent through LangGraph

---

## 4. Analysis Agents (Layer 3)

These 4 agents run **in parallel** on each stock and output normalized signals.

---

### 4.1 TechnicalAgent

**Purpose:** Computes technical indicators from raw OHLCV data for swing trading (3-10 day horizon).

**File:** `src/main/agents/technical_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `ohlc_daily` | `List[Dict]` with `{open, high, low, close}` | StockDataContext.historical_ohlc |
| `volume_daily` | `List[float]` | StockDataContext.historical_ohlc[].volume |
| `latest_price` | `float` | StockDataContext.last_close |
| `52_week_high` | `float` | Computed from last 252 bars |

#### Indicators Computed
| Indicator | Method | Parameters |
|-----------|--------|-----------|
| **RSI** | Wilder RSI | Period: 14 bars |
| **MACD** | EMA12 vs EMA26 direction | Fast: 12, Slow: 26 |
| **Trend** | EMA20 vs EMA50 crossover | Short: 20, Long: 50 |
| **Volatility** | ATR / Price (normalized) | ATR period: 14 |
| **Breakout** | Price ≥ 97% of 52w high + volume spike >1.3x | — |
| **Support** | 20-day low from OHLC | Window: 20 bars |
| **Resistance** | 20-day high from OHLC | Window: 20 bars |

#### Output: `TechnicalSignal`
```python
TechnicalSignal:
  technical_score: float     # 0-1 composite (0.30*RSI + 0.25*MACD + 0.30*trend + 0.15*breakout)
  rsi: float                 # Raw RSI (0-100)
  macd_signal: str           # "buy" | "sell" | "neutral"
  volatility: float          # Normalized ATR/price [0, 1]
  breakout_flag: bool        # True if near 52w high with volume spike
  trend_direction: str       # "bullish" | "bearish" | "neutral"
  support_level: float       # 20-day support price
  resistance_level: float    # 20-day resistance price
```

#### Feature Dict (for ML Judge)
```python
{
  "tech_score": 0.65,        # composite
  "tech_rsi": 0.48,          # RSI / 100
  "tech_macd": 1.0,          # 1.0=buy, 0.0=sell, 0.5=neutral
  "tech_volatility": 0.03,   # ATR / price
  "tech_breakout": 0.0,      # 1.0 or 0.0
  "tech_trend": 1.0,         # 1.0=bullish, 0.0=bearish
}
```

---

### 4.2 FundamentalAgent

**Purpose:** Computes valuation, growth, and financial health signals from Yahoo Finance real data.

**File:** `src/main/agents/fundamental_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `pe_ratio` | `float` | Yahoo Finance fundamentals |
| `forward_pe` | `float` | Yahoo Finance |
| `eps` | `float` | Yahoo Finance |
| `revenue_growth` | `float` | Fraction (e.g., 0.15 = 15%) |
| `profit_margin` | `float` | Fraction |
| `debt_to_equity` | `float` | Ratio |
| `roe` | `float` | Return on equity (fraction) |
| `market_cap` | `float` | In INR |
| `sector` | `str` | Company sector |

#### Sub-Scores
| Score | Calculation | Range |
|-------|------------|-------|
| **Valuation** | PE < 12 → "undervalued", ≤ 25 → "fair", > 25 → "overvalued" | 1.0 / 0.6 / 0.2 |
| **Growth** | `(revenue_growth% + 10) / 50` | [0, 1] |
| **Health** | `0.40 * margin_score + 0.30 * DE_score + 0.30 * ROE_score` | [0, 1] |
| **Composite** | `0.35 * valuation + 0.35 * growth + 0.30 * health` | [0, 1] |

#### Output: `FundamentalSignal`
```python
FundamentalSignal:
  fundamental_score: float        # 0-1 composite
  valuation_label: str            # "undervalued" | "fair" | "overvalued"
  growth_score: float             # 0-1
  financial_health_score: float   # 0-1
  pe_ratio: float                 # Raw PE
  forward_pe: float               # Forward PE
```

#### Feature Dict
```python
{
  "fund_score": 0.52,
  "fund_valuation": 0.6,     # 1.0=undervalued, 0.5=fair, 0.0=overvalued
  "fund_growth": 0.44,
  "fund_health": 0.35,
}
```

---

### 4.3 SentimentAgent

**Purpose:** Extracts sentiment signals from news articles using LLM or keyword-based fallback.

**File:** `src/main/agents/sentiment_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `news_articles` | `List[Dict]` with `{headline, summary, date, source}` | RSS News + Yahoo News |
| `recent_price_change` | `float` | Fraction (e.g., 0.015 = +1.5%) |

#### Two Processing Paths

**Path 1: LLM (Preferred)**
- Formats all articles into text
- Sends to GPT-4.1 with system prompt asking for structured JSON
- Prompt: *"You are a financial sentiment analyst..."*

**Path 2: Rule-Based (Fallback)**
- 24 positive keywords: beat, record, growth, profit, surge, rally, dividend...
- 25 negative keywords: miss, loss, decline, warning, downgrade, plunge, fraud...
- Counts positive/negative articles by keyword matching
- Sentiment trend from recent price change: >1% = improving, <-1% = deteriorating
- Confidence scales with article count (saturates at 10)

#### Output: `SentimentSignal`
```python
SentimentSignal:
  sentiment_score: float          # 0 (very negative) to 1 (very positive)
  positive_news_count: int        # Count of positive articles
  negative_news_count: int        # Count of negative articles
  sentiment_trend: str            # "improving" | "stable" | "deteriorating"
  news_confidence_score: float    # 0-1 (scales with article count/recency)
```

#### Feature Dict
```python
{
  "sent_score": 0.65,
  "sent_net_ratio": 0.67,        # positive / (total + 1), Laplace-smoothed
  "sent_trend": 1.0,             # 1.0=improving, 0.5=stable, 0.0=deteriorating
  "sent_confidence": 0.80,
}
```

---

### 4.4 EventAgent

**Purpose:** Detects earnings proximity, price gaps, dividends, splits, and major news catalysts.

**File:** `src/main/agents/event_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `earnings_date` | `str \| None` | NSE Corporate / Yahoo Finance |
| `earnings_results` | `Dict \| None` | EPS actual vs expected |
| `dividend_info` | `Dict \| None` | Ex-date, amount |
| `stock_split_info` | `Dict \| None` | Ratio, date |
| `recent_gap_data` | `Dict` | `{gap_pct: float}` |
| `major_news_flag` | `bool` | True if > 5 news articles |
| `news_articles` | `List[Dict]` | For optional LLM classification |

#### Detection Logic
| Check | Condition | Impact |
|-------|-----------|--------|
| **Earnings proximity** | Earnings within ±7 days of today | `earnings_impact_flag = True` |
| **Gap up** | Opening gap ≥ 2% above previous close | `gap_up_flag = True` |
| **Gap down** | Opening gap ≤ -2% below previous close | `gap_down_flag = True` |
| **Risk level** | Count of: earnings + gap + news + split + dividend | ≥3 = "high", ≥1 = "medium" |

#### Optional LLM Event Classification
Sends headlines to GPT-4.1 to classify the main event type into one of:
`earnings_beat`, `earnings_miss`, `guidance_raise`, `guidance_cut`,
`ma_announcement`, `ceo_change`, `regulatory_news`, `big_contract`,
`insider_buy`, `insider_sell`, `buyback`, `none`

#### Output: `EventSignal`
```python
EventSignal:
  event_score: float              # 0-1 (0.40*earnings + 0.30*gap + 0.30*risk)
  earnings_impact_flag: bool      # Earnings within ±7 days
  event_risk_level: str           # "low" | "medium" | "high"
  gap_up_flag: bool
  gap_down_flag: bool
  event_type: str | None          # LLM-classified event type
  event_description: str | None   # LLM one-sentence summary
```

#### Feature Dict
```python
{
  "evt_score": 0.15,
  "evt_earnings": 0.0,           # 1.0 or 0.0
  "evt_risk": 0.5,               # 0.0=low, 0.5=medium, 1.0=high
  "evt_gap_up": 0.0,
  "evt_gap_down": 0.0,
}
```

---

## 5. Decision Agents (Layer 4)

Three independent decision paths converge into a single **HybridDecision**.

---

### 5.1 JudgeAgent (Path 1: Rule-Based)

**Purpose:** ML Meta-Judge that aggregates all agent signals into a trading decision via weighted formula or trained model.

**File:** `src/main/agents/judge_agent.py`

#### Input: `AgentFeatureBundle`
```python
AgentFeatureBundle:
  symbol: str
  date: str
  technical: TechnicalSignal      # From TechnicalAgent
  fundamental: FundamentalSignal  # From FundamentalAgent
  sentiment: SentimentSignal      # From SentimentAgent
  event: EventSignal              # From EventAgent
  regime: RegimeSignal            # From RegimeDetectorAgent
  similarity_avg_return: float    # From PatternStore (RAG)
  similarity_positive_rate: float # From PatternStore
  similarity_max_drawdown: float  # From PatternStore
```

The bundle is flattened into a **20-dimensional feature vector** via `to_flat_features()`.

#### Two Scoring Modes

**Mode 1: Rule-Based (Default)**
- Weighted linear combination of 16 features
- Default weights (sum ~1.0):

| Feature | Weight | Category |
|---------|--------|----------|
| `tech_score` | 0.15 | Technical |
| `tech_trend` | 0.10 | Technical |
| `tech_macd` | 0.05 | Technical |
| `tech_breakout` | 0.05 | Technical |
| `fund_score` | 0.12 | Fundamental |
| `fund_growth` | 0.05 | Fundamental |
| `fund_health` | 0.05 | Fundamental |
| `sent_score` | 0.10 | Sentiment |
| `sent_net_ratio` | 0.07 | Sentiment |
| `sent_trend` | 0.05 | Sentiment |
| `evt_score` | 0.06 | Event |
| `evt_gap_up` | 0.03 | Event |
| `evt_earnings` | 0.02 | Event |
| `sim_avg_return` | 0.06 | RAG |
| `sim_pos_rate` | 0.04 | RAG |

**Mode 2: ML-Based (XGBoost)**
- Loads pre-trained model from pickle/joblib
- Uses `predict_proba()` for probability of 5-day up move
- Trained on 48,216 labeled rows

#### Decision Logic
```
IF prob_up >= BUY_THRESHOLD (0.65)
   AND expected_return >= MIN_RETURN (1.5%)
   AND prob_down < MAX_DOWNSIDE (25%)
   AND regime != "bear_trend"
THEN → BUY

IF prob_down >= BUY_THRESHOLD (0.65)
THEN → SELL

OTHERWISE → HOLD
```

#### Position Sizing
Kelly-inspired: `size = (prob - 0.5) * 2 * MAX_POSITION_SIZE` (capped at 2%)

#### Output: `JudgeDecision`
```python
JudgeDecision:
  symbol: str
  date: str
  decision: str                   # "BUY" | "SELL" | "HOLD"
  prob_up_5d: float               # Probability of +3% in 5 days
  expected_return_5d: float       # Point estimate (e.g., 0.029)
  downside_risk_prob: float       # Probability of -3% in 5 days
  confidence: float               # 0-1
  position_size_pct: float        # Recommended allocation (0-2%)
  stop_loss_pct: float            # Default -3%
  take_profit_pct: float          # Default +6%
  reasoning: str | None
  feature_importances: Dict       # Feature weight contributions
```

---

### 5.2 BullAgent (Path 2a: AI Debate — Bull Side)

**Purpose:** Generates the strongest possible BUY case for a stock with real evidence.

**File:** `src/main/agents/bull_agent.py`

#### Input: `DebateContext`
```python
DebateContext:
  symbol: str
  date: str
  latest_price: float
  previous_close: float
  week_52_high: float
  week_52_low: float
  recent_ohlc: List[Dict]                # Last 20 bars
  news_headlines: List[Dict]             # Last 10 headlines with source/date
  fundamentals: Dict                     # Real PE, EPS, growth, margins, D/E, ROE
  upcoming_events: List[Dict]            # Next 14 days
  regime: RegimeSignal                   # Market conditions
  signals: AgentFeatureBundle            # Pre-computed agent conclusions
  similar_past_setups: List[Dict]        # From PatternStore (RAG)
  past_trades_this_symbol: List[Dict]    # From TradeMemory
  mistake_warnings: List[str]            # From TradeMemory
```

#### Two Processing Paths

**LLM Path:**
- Formats DebateContext via `format_for_llm()` into text block (~500 tokens)
- System prompt: *"You are a bullish stock analyst. Construct the strongest BUY case..."*
- Output: JSON with `recommendation`, `confidence`, `key_points[]`, `reasoning`

**Rule-Based Path (Fallback):**
Checks for bullish signals across 6 categories:

| Category | Bullish Signals Checked |
|----------|----------------------|
| Technical | Score > 0.6, bullish trend, breakout, MACD buy |
| Price | Near 52-week low (within 15%) — reversal opportunity |
| Fundamental | PE < 20, revenue growth > 10%, ROE > 15%, margins > 15% |
| Sentiment | Score > 0.6, improving trend, positive headlines |
| Events | Upcoming earnings, dividend, buyback |
| Regime | Market in bull_trend |

Each signal adds a confidence factor; final confidence = average of all factors.

#### Output: `DebateArgument`
```python
DebateArgument:
  role: "bull"
  recommendation: str              # "BUY" or "HOLD"
  confidence: float                # 0-1
  key_points: List[str]            # 3-5 evidence-based points
  data_citations: Dict             # Source data references
  reasoning: str                   # 2-3 sentence summary
```

---

### 5.3 BearAgent (Path 2b: AI Debate — Bear Side)

**Purpose:** Generates the strongest possible SELL case for a stock with real evidence.

**File:** `src/main/agents/bear_agent.py`

#### Input: Same `DebateContext` as BullAgent

#### Two Processing Paths

**LLM Path:**
- System prompt: *"You are a bearish stock analyst. Construct the strongest case for NOT buying..."*
- Output: JSON with `recommendation`, `confidence`, `key_points[]`, `reasoning`

**Rule-Based Path (Fallback):**
Checks for bearish signals across 7 categories:

| Category | Bearish Signals Checked |
|----------|----------------------|
| Technical | RSI > 70 (overbought), bearish trend, score < 0.4, volatility > 0.7, MACD sell |
| Price | > 25% below 52-week high, down > 3% in 5 sessions |
| Fundamental | PE > 40, growth < 5%, D/E > 1.5, margins < 5% |
| Sentiment | Score < 0.4, deteriorating trend, negative headlines cited |
| Events | High event risk, gap down |
| Regime | Market in bear_trend or high_volatility |
| Financial Health | Health score < 0.4 |

#### Output: `DebateArgument`
```python
DebateArgument:
  role: "bear"
  recommendation: str              # "SELL" or "HOLD"
  confidence: float                # 0-1
  key_points: List[str]            # 3-5 evidence-based points
  data_citations: Dict
  reasoning: str
```

---

### 5.4 DebateAgent (Path 2c: Evaluator + Hybrid Combiner)

**Purpose:** (1) Evaluates bull vs bear arguments to pick a winner. (2) Combines rule-based and debate decisions into a final HybridDecision.

**File:** `src/main/agents/debate_agent.py`

#### Part 1: Debate Evaluation

**LLM Path:**
- System prompt: *"You are a professional hedge fund investment committee chair..."*
- Receives both bull and bear arguments + stock data summary
- Outputs: winning_side, bull_strength, bear_strength, decision, confidence

**Rule-Based Path:**
- `bull_score = confidence × (1 + key_points_count × 0.1)`
- `bear_score = confidence × (1 + key_points_count × 0.1)`
- Normalized: if bull_strength > 0.6 → BUY, bear_strength > 0.6 → SELL/HOLD, else HOLD

#### Output: `DebateDecision`
```python
DebateDecision:
  symbol: str
  date: str
  decision: str                   # "BUY" | "SELL" | "HOLD"
  confidence: float               # 0-1
  winning_side: str               # "bull" | "bear" | "neutral"
  bull_strength: float            # 0-1
  bear_strength: float            # 0-1
  reasoning: str
```

#### Part 2: Hybrid Decision (Rule + Debate)

Weights (configurable per mode):
- `RULE_WEIGHT` = 0.345 (optimized from 0.600)
- `DEBATE_WEIGHT` = 0.655 (optimized from 0.400)

Decision logic:
```
IF both agree → boost confidence × 1.1 (capped at 95%)
IF strong disagreement (one BUY, other SELL) → HOLD with 50% reduced confidence
IF mild disagreement → weighted average → BUY if > 0.65, SELL if < 0.35
```

#### Output: `HybridDecision`
```python
HybridDecision:
  symbol: str
  date: str
  final_decision: str             # "BUY" | "SELL" | "HOLD"
  final_confidence: float         # 0-1
  rule_decision: str              # From JudgeAgent
  rule_confidence: float
  debate_decision: str            # From DebateAgent
  debate_confidence: float
  agreement: bool                 # True if both flows agree
  disagreement_action: str | None # "HOLD" if strong conflict
  reasoning: str                  # "Rule: SELL (87%) | Debate: SELL (80%) | Final: SELL (95%)"
```

---

## 6. Execution Agents (Layer 5)

---

### 6.1 RegimeDetectorAgent

**Purpose:** Detects overall market conditions (trend + volatility) from NIFTY 50 index data and India VIX. Runs **once per pipeline execution** and results are cached.

**File:** `src/main/agents/regime_detector_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `index_ohlc` | `List[Dict]` | NIFTY 50 daily OHLC via Yahoo Finance (^NSEI) |
| `vix_value` | `float \| None` | India VIX (^INDIAVIX) current value |

#### Trend Detection
Uses 4 signals:
1. Price > SMA50 → bull signal
2. Price > SMA200 → bull signal
3. SMA50 > SMA200 → bull signal
4. RSI > 55 → bull signal

≥3 bull signals → `BULL_TREND` (confidence 0.7-0.9)  
≥3 bear signals → `BEAR_TREND` (confidence 0.7-0.9)  
Otherwise → `SIDEWAYS` (confidence 0.6)

#### Volatility Detection (India VIX)
| VIX Range | State |
|-----------|-------|
| < 12 | LOW |
| 12-18 | MODERATE |
| 18-25 | HIGH |
| ≥ 25 | EXTREME |

If VIX unavailable, falls back to ATR-based detection.

#### Output: `RegimeSignal`
```python
RegimeSignal:
  market_regime: MarketRegime     # BULL_TREND | BEAR_TREND | SIDEWAYS | HIGH_VOLATILITY
  volatility_state: VolatilityState  # LOW | MODERATE | HIGH | EXTREME
  regime_confidence: float        # 0-1
```

---

### 6.2 TradePlannerAgent

**Purpose:** Converts JudgeDecision into a complete, actionable trade plan with entry, exit, and risk parameters.

**File:** `src/main/agents/trade_planner_agent.py`

#### Input
| Field | Type | Source |
|-------|------|--------|
| `decision` | `JudgeDecision` | From JudgeAgent (or HybridDecision-adjusted) |
| `current_price` | `float` | StockDataContext.last_close |
| `ohlc` | `List[Dict]` | Historical OHLC for support/resistance/ATR |
| `technical` | `TechnicalSignal` | For breakout flag |
| `regime` | `RegimeSignal` | Market conditions |

#### Entry Logic
| Condition | Entry Type | Logic |
|-----------|-----------|-------|
| Breakout or high volatility | **MARKET** | Immediate execution needed |
| Normal conditions | **LIMIT** | 0.5% below current price, above support |

#### Stop-Loss Logic
- **ATR-based:** `entry - (ATR × 1.5 multiplier)`
- **Support-based:** `support × 0.98` (2% below support)
- Uses the tighter of the two
- Clamped to bounds: min 2%, max 8% from entry
- **Widened** in high volatility: +1% (HIGH) or +2% (EXTREME)

#### Target Logic
- `target = entry + risk_per_share × risk_reward_ratio`
- Default R:R = 2.0 (configurable 1.5-4.0)
- Reduced 20% if confidence < 0.6

#### Position Sizing
- Max 1% portfolio risk per trade
- Max 5% in single position
- `shares = (portfolio_value × risk_per_trade) / risk_per_share`

#### Output: `TradePlan`
```python
TradePlan:
  symbol: str
  date: str
  decision: str                   # "BUY" | "SELL"
  confidence: float
  entry_type: EntryType           # MARKET | LIMIT
  entry_price: float              # Suggested entry
  current_price: float
  stop_loss_price: float          # Exit if trade fails
  target_price: float             # Take profit target
  trailing_stop_pct: float        # Trailing stop %
  risk_reward_ratio: float        # reward ÷ risk (>1 is good)
  risk_per_share: float           # entry - stop_loss
  reward_per_share: float         # target - entry
  position_size_pct: float        # % of portfolio
  suggested_shares: int           # For given capital
  max_loss_amount: float          # If stop hit
  expected_holding_days: int      # Default: 5
  support_level: float
  resistance_level: float
  atr: float                      # Average True Range
  reasoning: str                  # Human-readable explanation
```

---

### 6.3 RiskManagerAgent

**Purpose:** Portfolio-level risk controls — adjusts position sizes, blocks dangerous trades, enforces exposure limits.

**File:** `src/main/agents/risk_manager_agent.py`

#### Input
| Field | Type |
|-------|------|
| `decisions` | `List[JudgeDecision]` |
| `regime` | `RegimeSignal` |
| `portfolio_value` | `float` (default ₹10L) |
| `current_positions` | `Dict[str, float]` (existing holdings) |
| `current_drawdown_pct` | `float` |
| `sector_map` | `Dict[str, str]` (symbol → sector) |

#### Risk Scaling Factors

**Volatility Scale:**
| State | Factor | Effect |
|-------|--------|--------|
| LOW | 1.2 | Scale UP positions |
| MODERATE | 1.0 | No change |
| HIGH | 0.6 | Scale DOWN 40% |
| EXTREME | 0.3 | Scale DOWN 70% |

**Regime Scale:**
| Regime | Factor |
|--------|--------|
| BULL_TREND | 1.1 |
| SIDEWAYS | 1.0 |
| BEAR_TREND | 0.7 |
| HIGH_VOLATILITY | 0.5 |

**Drawdown Scale:** Linear reduction as portfolio approaches max 15% drawdown. At max → stop trading.

#### No-Trade Conditions (Trade Blocking)
- Confidence < 0.45 → **BLOCKED**
- Expected return < 0.5% → **BLOCKED**
- Total exposure > 60% → **BLOCKED**
- At max drawdown → **BLOCKED**

#### Output: `PortfolioRiskAssessment`
```python
PortfolioRiskAssessment:
  total_exposure_pct: float
  max_single_position_pct: float
  correlation_risk: float
  drawdown_risk: float
  regime_risk_multiplier: float
  overall_risk_level: RiskLevel    # VERY_LOW | LOW | MODERATE | HIGH | EXTREME
  positions: List[PositionRisk]    # Per-position risk details
  warnings: List[str]

PositionRisk:
  symbol: str
  adjusted_position_size: float
  risk_level: RiskLevel
  blocked: bool
  block_reason: str | None
```

---

### 6.4 PositionManagementAgent

**Purpose:** Monitors open positions and recommends trailing stop adjustments, partial exits, and full exits.

**File:** `src/main/agents/position_management_agent.py`

#### Input
| Field | Type |
|-------|------|
| `positions` | `List[OpenPosition]` |
| `current_prices` | `Dict[str, float]` |
| `regime` | `RegimeSignal` |
| `sentiments` | `Dict[str, SentimentSignal]` |

#### Checks Performed (in order)
1. **Stop-loss hit** — current price ≤ stop → FULL_EXIT
2. **Target hit** — current price ≥ target → FULL_EXIT
3. **Trailing stop hit** — current price ≤ trailing stop → FULL_EXIT
4. **Partial profit** — unrealized P&L > 4% → sell 50%
5. **Trailing stop adjustment** — if profit > 3% and highest price increased → raise stop
6. **Time exit** — held > 10 days → review
7. **Stagnant exit** — flat for 5 days with < 1% move → suggest exit
8. **Sentiment exit** — sentiment drops below 0.3 → exit
9. **Regime exit** — adverse regime change → exit

#### Output: `PositionUpdate`
```python
PositionUpdate:
  symbol: str
  action: str                     # "HOLD" | "ADJUST_STOP" | "PARTIAL_EXIT" | "FULL_EXIT"
  new_stop_loss: float | None     # For ADJUST_STOP
  exit_price: float | None        # For exits
  exit_reason: ExitReason         # TARGET_HIT | STOP_LOSS_HIT | TRAILING_STOP | TIME_EXIT | ...
  exit_shares: int                # For partial exit
  reasoning: str                  # Human-readable explanation
```

---

## 7. Pre-Pipeline Agents

---

### 7.1 StockDiscoveryAgent

**Purpose:** Screens the NIFTY 50 universe to find the most actionable stocks for daily analysis. Runs **before** the main pipeline.

**File:** `src/main/agents/stock_discovery_agent.py`

#### Default Universe
30 NIFTY 50 components: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK, LT, AXISBANK, ASIANPAINT, MARUTI, TITAN, SUNPHARMA, BAJFINANCE, WIPRO, HCLTECH, ULTRACEMCO, NESTLEIND, TATAMOTORS, POWERGRID, NTPC, TECHM, INDUSINDBK, ONGC, JSWSTEEL, TATASTEEL, ADANIPORTS

#### Screening Criteria
| Criterion | Condition | Score |
|-----------|-----------|-------|
| Volume spike | Volume > 1.5× 20-day avg | +0.3 |
| Near 52-week high | Price ≥ 95% of 52w high | +0.25 |
| Near 52-week low | Price ≤ 105% of 52w low | +0.15 |
| Big price move | \|Change\| > 2% | +0.20 |
| News driven | > 3 news articles | +0.10 |

#### Output: `List[StockCandidate]`
```python
StockCandidate:
  symbol: str
  score: float                    # 0-1 priority score
  reasons: List[str]              # e.g. ["Volume spike (2.1x avg)", "Near 52-week high"]
```

---

### 7.2 BacktestAgent

**Purpose:** Validates signal quality through historical walk-forward testing. Prevents blind trust in untested signals.

**File:** `src/main/agents/backtest_agent.py`

#### Input
| Field | Type |
|-------|------|
| `symbol` | `str` |
| `historical_signals` | `List[Dict]` with date, signal, confidence, price |
| `price_history` | `List[Dict]` with OHLCV data |
| `stop_loss_pct` | `float` (default 3%) |
| `take_profit_pct` | `float` (default 6%) |
| `holding_period_days` | `int` (default 5) |

#### Metrics Computed
| Metric | Description |
|--------|-------------|
| Sharpe Ratio | Risk-adjusted return (annualized) |
| Sortino Ratio | Downside-risk-adjusted return |
| Calmar Ratio | Annual return / max drawdown |
| Max Drawdown | Worst peak-to-trough decline |
| Win Rate | Winning trades / total trades |
| Profit Factor | Gross profit / gross loss |
| VaR (95%) | Value at Risk |
| Expected Shortfall | Average loss beyond VaR |
| Stability Score | Consistency across periods |

#### Overfitting Detection
If `train_Sharpe - test_Sharpe > 1.0` → flagged as overfit

#### Signal Quality Classification
| Quality | Sharpe Range |
|---------|-------------|
| EXCELLENT | > 2.0, stable |
| GOOD | 1.0-2.0 |
| FAIR | 0.5-1.0 |
| POOR | 0.0-0.5 |
| UNRELIABLE | < 0.0 or unstable |

#### Output: `BacktestResult`
```python
BacktestResult:
  symbol: str
  start_date: str
  end_date: str
  metrics: PerformanceMetrics     # Sharpe, Sortino, win rate, etc.
  signal_quality: SignalQuality   # EXCELLENT | GOOD | FAIR | POOR | UNRELIABLE
  quality_score: float            # 0-1 composite
  trades: List[TradeRecord]       # Detailed trade-by-trade log
  warnings: List[str]
  is_overfit: bool
  overfit_reason: str | None
```

---

## 8. Controllers

---

### 8.1 DataProcessor (Multi-Provider Architecture)

**Purpose:** Manages multiple data providers, fetches stock data, and builds `StockDataContext` objects.

**File:** `src/main/controllers/data_processor.py`

#### Registered Providers
| Provider | Data | Source |
|----------|------|--------|
| `YahooFinanceProvider` | OHLCV (2y), fundamentals (PE, EPS, margins), 52w high/low | `yfinance` API |
| `NSECorporateProvider` | Corporate actions (30d), announcements (7d), upcoming events (14d) | NSE India |
| `RSSNewsProvider` | News headlines + summaries from 10+ feeds (72h) | Google News RSS |

#### Key Methods
- `register_provider(provider)` — Adds a data provider
- `build_stock_context(symbol, index_data)` → `StockDataContext`
- `build_index_context(index_symbol)` → `IndexData`

---

### 8.2 PatternStore (RAG — Vector Similarity Search)

**Purpose:** Stores 20-dimensional feature vectors from every analysis and enables cosine similarity search to find similar historical setups.

**File:** `src/main/controllers/pattern_store.py`

**Implementation:** Pure NumPy — no external vector DB needed.

#### Storage
| File | Contents |
|------|----------|
| `vectors.npz` | NumPy array of feature vectors (20-dim) |
| `records.jsonl` | One PatternRecord JSON per line |

#### Feature Vector Keys (20 dimensions)
```python
["tech_score", "tech_rsi", "tech_macd", "tech_volatility", "tech_breakout", "tech_trend",
 "fund_score", "fund_valuation", "fund_growth", "fund_health",
 "sent_score", "sent_net_ratio", "sent_trend", "sent_confidence",
 "evt_score", "evt_earnings", "evt_risk", "evt_gap_up", "evt_gap_down",
 "regime_confidence"]
```

#### Key Methods
- `record(features, symbol, date, decision, confidence, regime)` — Store a new vector
- `search_similar(features, top_k=10)` → `SimilarityResult`
- `record_outcome(symbol, date, actual_return_5d)` — Fill in what actually happened

#### SimilarityResult
```python
SimilarityResult:
  similar_count: int
  avg_return_5d: float            # Average return of similar setups
  positive_rate: float            # % of similar setups that went up
  max_drawdown: float
  records: List[PatternRecord]    # The actual similar records
```

---

### 8.3 TradeMemory (Append-Only JSONL)

**Purpose:** Records every BUY/SELL decision with full reasoning, tracks outcomes when positions close, and extracts mistake warnings for future debates.

**File:** `src/main/controllers/trade_memory.py`

#### Storage: `data/trade_memory.jsonl` (one JSON per line)

#### TradeMemoryItem
```python
TradeMemoryItem:
  # At decision time:
  timestamp: str
  symbol: str
  date: str
  decision: str                   # BUY / SELL
  confidence: float
  entry_price: float
  regime: str
  key_reasons: List[str]
  debate_agreement: bool

  # After exit:
  exit_price: float
  exit_date: str
  pnl_pct: float
  hit_stop_loss: bool
  hit_target: bool
  days_held: int
  outcome_notes: str
  is_mistake: bool                # True if stop loss hit
```

#### Key Methods
- `record_decision(symbol, date, decision, confidence, entry_price, regime, key_reasons, debate_agreement)`
- `record_outcome(symbol, date, exit_price, pnl_pct, hit_stop_loss, hit_target)`
- `get_history_for_symbol(symbol, limit=5, only_with_outcomes=True)` → `List[TradeMemoryItem]`
- `get_mistakes(regime, limit=3)` → `List[TradeMemoryItem]` (past losing trades in similar regime)
- `get_stats()` → `{win_rate, avg_return, resolved_count}`

---

### 8.4 PortfolioManager

**Purpose:** Persistent portfolio with holdings, cash tracking, trade history, and watchlist management.

**File:** `src/main/controllers/portfolio_manager.py`

#### Storage Files
| File | Contents |
|------|----------|
| `data/portfolio.json` | Holdings, cash, trade history |
| `data/watchlist.json` | Stocks to monitor with priority |

#### Data Models
```python
Holding:
  symbol: str, quantity: int, entry_price: float, entry_date: str,
  stop_loss: float, target_price: float, trailing_stop_pct: float,
  current_price: float, highest_price: float, unrealized_pnl: float

WatchlistItem:
  symbol: str, added_date: str, reason: str,
  priority: str ("high"|"medium"|"low"),
  last_signal: str, last_signal_date: str

PortfolioState:
  cash: float (default ₹10,00,000)
  initial_capital: float
  holdings: List[Holding]
  trade_history: List[TradeRecord]
```

#### Key Methods
- `load()` / `save()` — JSON persistence
- `open_position(symbol, quantity, entry_price, stop_loss, target_price, ...)`
- `close_position(symbol, exit_price, exit_reason)`
- `process_signals(results, auto_enter, auto_exit, mode)` → actions dict
- `get_watchlist_symbols()` / `get_all_holdings()`
- `portfolio_text_summary()` → formatted string

#### Signal Processing Flow
```
For each pipeline result:
  IF BUY and auto_enter:
    - Check max positions not exceeded
    - Check enough cash for 10% position
    - open_position()
  IF SELL and auto_exit:
    - Find matching holding
    - close_position()
  Always:
    - Update current prices for all holdings
    - Check SL/target hits → auto-exit
    - Update watchlist last_signal
```

---

### 8.5 TradeExecutor

**Purpose:** Broker API interface with retry logic and slippage tracking. Uses `MockBroker` for testing; real brokers (Zerodha Kite) can be plugged in.

**File:** `src/main/controllers/trade_executor.py`

#### Broker Interface
```python
BrokerInterface (Abstract):
  place_order(symbol, action, quantity, price, order_type) → ExecutionResult
  get_order_status(order_id) → str
```

#### MockBroker
- Always succeeds with small random slippage (±0.1%)
- Configurable fail rate for testing
- Returns `ExecutionResult` with order_id

#### Retry Logic
- Max 3 retries with 2-second delays
- Slippage validation: warns if > 0.5%

#### ExecutionResult
```python
ExecutionResult:
  symbol: str, action: str, quantity: int
  expected_price: float, executed_price: float
  status: str ("SUCCESS"|"FAILED"|"PENDING")
  order_id: str
  slippage: float, slippage_pct: float
  retry_count: int
  reason: str (if FAILED)
```

---

### 8.6 MessageController (WhatsApp Alerts)

**Purpose:** Sends rich trade alerts via WhatsApp Web automation.

**File:** `src/main/controllers/message_controller.py`

#### Alert Format (for BUY/SELL signals)
```
>>> *BUY SIGNAL* <<<
*Stock: RELIANCE*
Date: 2026-03-09

*Current Price*: Rs.1,424.00
*Confidence*: 87%
*Prob Up (5 days)*: 87%
*Expected Return*: +2.97%
*Downside Risk*: 13%

*--- TRADE PLAN ---*
  Entry: MARKET ORDER @ Rs.1,424.00
  Stop Loss: Rs.1,307.00 (-8.2%)
  Target: Rs.1,473.00 (+3.4%)
  Risk:Reward = 1:2.0
  Hold Period: 7 days

*--- TECHNICALS ---*
  RSI: 48 (Neutral) | MACD: SELL | Trend: Bearish
  Support: Rs.1,307 | Resistance: Rs.1,473

*--- FUNDAMENTALS ---*
  PE: 23.1 (Fwd: 21.8) | Valuation: FAIR
  Growth: 41/100 | Health: 11/100

*--- AI DEBATE ---*
  Bull (55%) vs Bear (80%) — Bear wins
  + Point 1 from bull with evidence
  - Point 1 from bear with evidence

*--- FINAL VERDICT ---*
  Rule: SELL (87%) | AI Debate: SELL (80%)
  Agreement: YES | FINAL: SELL (95%)
```

#### Daily Summary Message
Sent after all stocks analyzed: total BUY/HOLD/SELL counts, top recommendations, portfolio status.

---

### 8.7 PerformanceTracker

**Purpose:** Computes portfolio performance metrics from trade history.

**File:** `src/main/controllers/performance_tracker.py`

Metrics include: total return, win rate, average win/loss, Sharpe ratio, max drawdown.

---

## 9. Data Models & Signal Structures

### StockDataContext (Per-Stock Data Container)
```python
StockDataContext:
  # Identity
  symbol: str, exchange: str, company_name: str
  sector: str, industry: str, isin: str, currency: "INR"

  # Price snapshot
  last_close: float, previous_close: float
  last_open: float, last_high: float, last_low: float
  last_volume: int, last_trading_date: str

  # Historical data (400-500 bars)
  historical_ohlc: List[PriceData]    # PriceData: {symbol, date, open, high, low, close, volume}

  # Fundamentals
  fundamentals: FundamentalData       # {pe_ratio, forward_pe, eps, revenue_growth_yoy, profit_margin,
                                      #  debt_to_equity, dividend_yield, market_cap, sector, industry, ...}

  # News
  news_items: List[NewsItem]          # {symbol, date, headline, news_text, source, url, sentiment_score}

  # Events
  event_data: List[EventData]         # {symbol, date, event_type, description, earnings_date,
                                      #  recent_corporate_actions: [{type, date, details}]}

  # Computed / flags
  computed_metrics: Dict              # similarity scores, regime info, etc.
  flags: Dict                         # market_regime, volatility_state, etc.

  # Price data (alias for historical_ohlc)
  price_data: List[PriceData]
```

### IndexData (Shared per run)
```python
IndexData:
  index_symbol: str                   # "^NSEI"
  historical_ohlc: List[PriceData]    # NIFTY 50 daily bars
  last_close: float
  last_volume: int
  last_trading_date: str
```

### Enumerations
```python
MarketRegime:     BULL_TREND | BEAR_TREND | SIDEWAYS | HIGH_VOLATILITY
VolatilityState:  LOW | MODERATE | HIGH | EXTREME
EntryType:        MARKET | LIMIT
ExitReason:       TARGET_HIT | STOP_LOSS_HIT | TRAILING_STOP | TIME_EXIT |
                  SENTIMENT_CHANGE | REGIME_CHANGE | MANUAL
SignalQuality:    EXCELLENT | GOOD | FAIR | POOR | UNRELIABLE
RiskLevel:        VERY_LOW | LOW | MODERATE | HIGH | EXTREME
TradingMode:      CONSERVATIVE | AGGRESSIVE | MOMENTUM | VALUE | SCALPER | ADAPTIVE
```

---

## 10. Data Providers

### YahooFinanceProvider
| Data Type | Details |
|-----------|---------|
| OHLCV | 2 years daily with `.NS` suffix for NSE |
| Fundamentals | PE, forward PE, EPS, revenue growth, profit margin, D/E, ROE, market cap |
| Market Info | Sector, industry, 52-week high/low |
| Config | `exchange_suffix: ".NS"`, `price_period: "2y"`, `price_interval: "1d"` |

### NSECorporateProvider
| Data Type | Details |
|-----------|---------|
| Corporate Actions | Last 30 days: dividends, splits, bonuses |
| Announcements | Last 7 days: board meetings, results |
| Upcoming Events | Next 14 days: AGMs, earnings dates |

### RSSNewsProvider
| Data Type | Details |
|-----------|---------|
| News | Google News RSS + financial feeds |
| Filtering | `only_significant_news: true`, `max_news_age_hours: 72` |
| Coverage | Headlines + summaries, up to 20 articles per symbol |

---

## 11. Pipeline Orchestrator

**File:** `src/pipeline/orchestrator.py` (1383 lines)

### Class: `PipelineOrchestrator`

Central coordinator that:
1. Initializes all 14 agents with mode-specific configurations
2. Manages trading mode switching (hot-swap at runtime)
3. Runs the complete pipeline from data fetch → decision → execution

### Initialization
```python
orch = PipelineOrchestrator(config_path="config.yaml", mode="momentum")
```
Creates and configures:
- LLM adapter (Azure OpenAI GPT-4.1)
- 4 analysis agents (Technical, Fundamental, Sentiment, Event)
- JudgeAgent with mode-tuned weights
- 3 debate agents (Bull, Bear, Debate) with mode-tuned weights
- RegimeDetector, TradePlanner, RiskManager, PositionManager
- DataProcessor with 3 providers registered
- PatternStore (NumPy) + TradeMemory (JSONL)
- StockDiscoveryAgent

### Key Methods

#### `run_for_symbols(symbols, index_symbol)` → `List[Dict]`
Main entry point used by `run_orchestrator_pipeline.py`:
1. Fetch index data ONCE
2. Detect regime ONCE
3. For each symbol:
   - `build_stock_context()` — fetch all data
   - `analyse_stock_context()` — run full pipeline
4. Return list of result dicts

#### `analyse_stock_context(stock_ctx, regime)` → `Dict`
Full pipeline for a single stock:
1. Run 4 analysis agents → TechnicalSignal, FundamentalSignal, SentimentSignal, EventSignal
2. Build AgentFeatureBundle (20 features)
3. Run JudgeAgent → JudgeDecision
4. Build DebateContext (rich data for LLM agents)
5. Run BullAgent → DebateArgument
6. Run BearAgent → DebateArgument
7. Run DebateAgent.evaluate_debate() → DebateDecision
8. Run DebateAgent.make_hybrid_decision() → HybridDecision
9. If BUY/SELL: Run TradePlanner → TradePlan
10. If BUY/SELL: Run RiskManager → RiskAssessment
11. Record to PatternStore + TradeMemory

#### `discover_symbols(universe, max_stocks)` → `List[Dict]`
Auto-discover stocks by screening the universe.

#### `switch_mode(new_mode)` — Hot-swap trading mode
Updates agent weights, thresholds, and runtime config without restart.

#### `run_full_pipeline(symbols, auto_execute)` → `Dict`
Complete pipeline with auto-execution: analyse → plan → risk → execute → portfolio.

---

## 12. Trading Modes & Tuning Parameters

### 6 Trading Modes

| Mode | Buy Threshold | Max Positions | Hold Days | Strategy |
|------|--------------|---------------|-----------|----------|
| **CONSERVATIVE** | 0.80 | 3 | 5 | Protect capital, low risk |
| **AGGRESSIVE** | 0.55 | 10 | 7 | Max return, higher risk |
| **MOMENTUM** | 0.60 | 8 | 7 | Ride trends, cut losers fast |
| **VALUE** | 0.62 | 5 | 10 | Buy undervalued, hold longer |
| **SCALPER** | 0.58 | 10 | 2 | Many small wins, very short holds |
| **ADAPTIVE** | Dynamic | Dynamic | Dynamic | Auto-switch based on regime |

### Mode-Specific Weight Overrides

**MOMENTUM mode (tech-heavy):**
```
tech_score: 0.25, tech_trend: 0.20, tech_macd: 0.15, tech_breakout: 0.10
fund_score: 0.03, sent_score: 0.08
Rule weight: 0.30, Debate weight: 0.70
```

**VALUE mode (fundamental-heavy):**
```
fund_score: 0.25, fund_growth: 0.18, fund_health: 0.15
tech_score: 0.05
Rule weight: 0.60, Debate weight: 0.40
```

### 10 Tuning Parameter Categories (180+ params)

| Category | Key Parameters | Count |
|----------|---------------|-------|
| **Judge** | Weights (16), buy/sell thresholds, return limits | ~22 |
| **Technical** | RSI/MACD/EMA periods, scoring weights, breakout thresholds | ~16 |
| **Fundamental** | PE buckets, growth ranges, health weights | ~14 |
| **Sentiment** | Keyword lists (49 words), confidence saturation | ~6 |
| **Event** | Earnings window, gap threshold, risk levels | ~10 |
| **Regime** | RSI/SMA thresholds, VIX levels, confidence calc | ~14 |
| **Debate** | Bull/bear thresholds, hybrid weights, agreement boost | ~22 |
| **Trade** | R:R ratios, stop limits, position sizing, trailing stop | ~18 |
| **Risk** | Exposure limits, volatility/regime scaling, drawdown controls | ~28 |
| **Position** | Trailing activation, partial exits, time/stagnant exits | ~10 |

### Configuration Loading Priority
```
Defaults (Python dataclasses) → Mode Preset → tuning_params.yaml overrides
```

---

## 13. Response Structures & Output Formats

### Per-Stock Pipeline Result (as returned by orchestrator)
```json
{
  "symbol": "RELIANCE",
  "date": "2026-03-09",
  "regime": {
    "regime": "bear_trend",
    "vol_state": "high",
    "regime_confidence": 0.8
  },
  "technical": {
    "success": true,
    "run_id": "uuid",
    "timing": {"started_epoch_ms": ..., "completed_epoch_ms": ..., "duration_ms": 12},
    "integrity": {"checksum": "sha256...", "algorithm": "SHA256"},
    "payload": {
      "signal": {
        "tech_score": 0.35, "tech_rsi": 0.48, "tech_macd": 0.0,
        "tech_volatility": 0.03, "tech_breakout": 0.0, "tech_trend": 0.0
      },
      "raw_signal": "<TechnicalSignal object>"
    }
  },
  "fundamental": {
    "success": true,
    "payload": {
      "signal": {
        "fund_score": 0.41, "fund_valuation": 0.6,
        "fund_growth": 0.44, "fund_health": 0.11
      }
    }
  },
  "sentiment": {
    "success": true,
    "payload": {
      "signal": {
        "sent_score": 0.45, "sent_net_ratio": 0.43,
        "sent_trend": 0.0, "sent_confidence": 0.90
      }
    }
  },
  "event": {
    "success": true,
    "payload": {
      "signal": {
        "evt_score": 0.15, "evt_earnings": 0.0,
        "evt_risk": 0.5, "evt_gap_up": 0.0, "evt_gap_down": 0.0
      }
    }
  },
  "judge_decision": {
    "success": true,
    "payload": {
      "decision": "SELL",
      "prob_up_5d": 0.13,
      "expected_return_5d": -0.037,
      "downside_risk_prob": 0.87,
      "confidence": 0.87,
      "position_size_pct": 0.0
    }
  },
  "debate": {
    "bull": {
      "role": "bull",
      "recommendation": "HOLD",
      "confidence": 0.55,
      "key_points": ["Price rebounded +4.8% in 5 sessions", "..."],
      "reasoning": "Moderate bull case..."
    },
    "bear": {
      "role": "bear",
      "recommendation": "SELL",
      "confidence": 0.80,
      "key_points": ["D/E=35.65, thin margin 8.1%", "Bear regime with high vol", "..."],
      "reasoning": "Strong bear case..."
    },
    "debate_decision": {
      "decision": "SELL",
      "confidence": 0.80,
      "winning_side": "bear",
      "bull_strength": 0.41,
      "bear_strength": 0.59
    }
  },
  "hybrid_decision": {
    "final_decision": "SELL",
    "final_confidence": 0.95,
    "rule_decision": "SELL",
    "rule_confidence": 0.87,
    "debate_decision": "SELL",
    "debate_confidence": 0.80,
    "agreement": true,
    "reasoning": "Rule: SELL (87%) | Debate: SELL (80%) | Final: SELL (95%)"
  },
  "trade_plan": {
    "decision": "SELL",
    "entry_type": "market",
    "entry_price": 1424.00,
    "stop_loss_price": 1307.00,
    "target_price": 1218.00,
    "risk_reward_ratio": 1.8,
    "position_size_pct": 0.05,
    "expected_holding_days": 7
  },
  "risk_assessment": {
    "overall_risk_level": "moderate",
    "regime_risk_multiplier": 0.7,
    "warnings": ["Bear regime — reduced position sizes"],
    "positions": [{
      "symbol": "RELIANCE",
      "blocked": false,
      "adjusted_position_size": 0.035,
      "risk_level": "moderate"
    }]
  }
}
```

### analysis_results.json (Final Output)
```json
{
  "timestamp": "2026-03-09T10:30:00",
  "symbols_analyzed": 5,
  "buy_count": 1,
  "hold_count": 3,
  "sell_count": 1,
  "results": [/* array of per-stock results above */]
}
```

### portfolio.json
```json
{
  "cash": 850000.0,
  "initial_capital": 1000000.0,
  "holdings": [
    {
      "symbol": "TCS",
      "quantity": 25,
      "entry_price": 3920.0,
      "entry_date": "2026-03-05",
      "stop_loss": 3802.0,
      "target_price": 4150.0,
      "trailing_stop_pct": 0.02,
      "current_price": 4010.0,
      "highest_price": 4050.0,
      "unrealized_pnl": 2250.0
    }
  ],
  "trade_history": [
    {
      "symbol": "INFY",
      "action": "BUY",
      "quantity": 50,
      "entry_price": 1680.0,
      "exit_price": 1730.0,
      "pnl": 2500.0,
      "pnl_pct": 0.0298,
      "status": "CLOSED",
      "exit_reason": "target_hit"
    }
  ]
}
```

### watchlist.json
```json
{
  "watchlist": [
    {
      "symbol": "RELIANCE",
      "added_date": "2026-01-15",
      "reason": "Core holding",
      "priority": "high",
      "last_signal": "SELL",
      "last_signal_date": "2026-03-09"
    }
  ]
}
```

### trade_memory.jsonl (one line per record)
```json
{"timestamp":"2026-03-09T10:30:00","symbol":"RELIANCE","date":"2026-03-09","decision":"SELL","confidence":0.95,"entry_price":1424.0,"regime":"bear_trend","key_reasons":["D/E=35.65","Bear regime"],"debate_agreement":true,"outcome_recorded":false}
```

---

## 14. LLM Integration

### LLMAdapter
**File:** `src/main/agents/adapters/llm_adapter.py`

Wraps Azure OpenAI GPT-4.1 with a simple interface:

```python
llm = LLMAdapter(
    endpoint="https://...openai.azure.com/openai/v1",
    api_key="...",
    model="gpt-4.1",
    temperature=0.2,
    max_tokens=1024,
)

# Text response
text = llm.invoke(system_prompt="...", user_prompt="...")

# JSON response
data = llm.invoke_json(system_prompt="...", user_prompt="...")
```

### Agents Using LLM
| Agent | LLM Usage | Fallback |
|-------|-----------|----------|
| **SentimentAgent** | Classify news sentiment → JSON | Keyword counting |
| **EventAgent** | Classify event type from headlines → JSON | Rule-based risk scoring |
| **BullAgent** | Generate BUY arguments with citations → JSON | Score-based signal checking |
| **BearAgent** | Generate SELL arguments with citations → JSON | Score-based signal checking |
| **DebateAgent** | Evaluate bull vs bear, pick winner → JSON | Confidence×points scoring |

### Token Budget Control
DebateContext limits data sent to LLM:
- Last 20 OHLC bars (not 500)
- Last 10 news headlines (not all)
- Full fundamentals dict
- Up to 5 upcoming events
- Up to 5 similar past setups (from RAG)
- Up to 5 past trades (from Memory)
- Up to 3 mistake warnings

---

## 15. Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main config: project info, LLM settings, orchestrator stages, logging |
| `credentials.yaml` | **Git-ignored.** Azure OpenAI endpoint + API key |
| `tuning_params.yaml` | Optional YAML overrides for any of the 180+ tuning parameters |
| `data/portfolio.json` | Persistent portfolio state (cash, holdings, trades) |
| `data/watchlist.json` | Stock watchlist with priorities |
| `data/trade_memory.jsonl` | Append-only trade decision log |
| `data/pattern_store/vectors.npz` | RAG feature vectors (NumPy) |
| `data/pattern_store/records.jsonl` | RAG metadata records |

---

## 16. File Structure Map

```
LCF/
├── config.yaml                         # Main configuration
├── credentials.yaml                    # LLM credentials (git-ignored)
├── main.py                             # Entry point
├── requirements.txt                    # Python dependencies
│
├── src/
│   ├── pipeline/
│   │   └── orchestrator.py             # PipelineOrchestrator (1383 lines)
│   │
│   ├── main/
│   │   ├── agents/
│   │   │   ├── interfaces/
│   │   │   │   ├── agent.py            # Agent base class, AgentResult, AgentContext
│   │   │   │   └── signals.py          # All signal dataclasses (615 lines)
│   │   │   ├── adapters/
│   │   │   │   └── llm_adapter.py      # Azure OpenAI GPT-4.1 wrapper
│   │   │   │
│   │   │   ├── technical_agent.py      # RSI, MACD, trend, breakout, volatility
│   │   │   ├── fundamental_agent.py    # PE, growth, health, valuation
│   │   │   ├── sentiment_agent.py      # LLM/keyword news analysis
│   │   │   ├── event_agent.py          # Earnings, gaps, dividends, events
│   │   │   ├── judge_agent.py          # ML Meta-Judge (rule + XGBoost)
│   │   │   ├── regime_detector_agent.py # NIFTY trend + VIX volatility
│   │   │   ├── bull_agent.py           # BUY argument generator (LLM)
│   │   │   ├── bear_agent.py           # SELL argument generator (LLM)
│   │   │   ├── debate_agent.py         # Debate evaluator + hybrid combiner
│   │   │   ├── trade_planner_agent.py  # Entry/SL/target/position sizing
│   │   │   ├── risk_manager_agent.py   # Portfolio risk controls
│   │   │   ├── position_management_agent.py # Trailing stops, exits
│   │   │   ├── stock_discovery_agent.py # NIFTY 50 screening
│   │   │   └── backtest_agent.py       # Walk-forward validation
│   │   │
│   │   ├── controllers/
│   │   │   ├── data_context.py         # StockDataContext, IndexData, data models
│   │   │   ├── data_processor.py       # Multi-provider data aggregation
│   │   │   ├── data_processor_agent.py # Agent-pattern data processor
│   │   │   ├── yahoo_finance_provider.py
│   │   │   ├── nse_corporate_provider.py
│   │   │   ├── rss_news_provider.py
│   │   │   ├── news_data_provider.py
│   │   │   ├── pattern_store.py        # RAG (NumPy cosine similarity)
│   │   │   ├── trade_memory.py         # JSONL trade log
│   │   │   ├── portfolio_manager.py    # Holdings + watchlist
│   │   │   ├── trade_executor.py       # Broker interface + MockBroker
│   │   │   ├── message_controller.py   # WhatsApp alerts
│   │   │   └── performance_tracker.py  # Portfolio metrics
│   │   │
│   │   └── config/
│   │       └── tuning_params.py        # 180+ params, 6 mode presets (712 lines)
│   │
│   └── utils/
│       └── logger.py
│
├── news_stock_tracker/
│   ├── run_orchestrator_pipeline.py    # CLI runner (323 lines)
│   ├── stock_tracker.py               # News-based stock discovery
│   ├── analysis_results.json          # Pipeline output
│   ├── stocks.txt                     # Symbol list
│   └── my_watchlist.txt               # Personal watchlist
│
└── data/
    ├── portfolio.json                  # Persistent portfolio
    ├── trade_memory.jsonl              # Append-only trade log
    ├── pattern_store/                  # RAG vectors + records
    └── ...
```
