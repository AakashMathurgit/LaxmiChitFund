# LCF — Architecture & Flow Diagrams (Mermaid)

---

## 1. End-to-End Pipeline Overview

```mermaid
flowchart TB
    subgraph INPUT["🔌 INPUT & CONFIG"]
        CLI["CLI Arguments<br/>--mode --stocks --source<br/>--auto-trade --whatsapp"]
        CONFIG["config.yaml<br/>tuning_params.yaml<br/>credentials.yaml"]
        PORTFOLIO_FILE["portfolio.json<br/>watchlist + holdings"]
    end

    subgraph INIT["⚙️ INITIALIZATION"]
        ORCH["PipelineOrchestrator<br/>mode: adaptive|momentum|...<br/>180+ tunable params"]
        LLM["LLM Adapter<br/>GPT-4.1 via Azure OpenAI"]
        PM["PortfolioManager<br/>load watchlist + holdings"]
    end

    subgraph SYMBOLS["📋 SYMBOL RESOLUTION"]
        direction TB
        SRC_AUTO["AUTO: holdings + watchlist_high + discovery"]
        SRC_WL["WATCHLIST: from portfolio watchlist"]
        SRC_DISC["DISCOVERY: StockDiscoveryAgent screens NIFTY 50"]
        SRC_FILE["FILE: stocks.txt"]
    end

    subgraph REGIME["🌍 REGIME DETECTION (once per run)"]
        INDEX["Fetch NIFTY 50 Index<br/>^NSEI via Yahoo Finance"]
        VIX["Fetch India VIX<br/>^INDIAVIX"]
        RDA["RegimeDetectorAgent<br/>SMA20/SMA50 crossover + VIX"]
        REGIME_OUT["RegimeSignal<br/>bull_trend | bear_trend | sideways<br/>low | moderate | high volatility"]
    end

    subgraph PER_STOCK["🔁 PER-STOCK PIPELINE (~10s each)"]
        DATA_FETCH["DataProcessor<br/>build_stock_context()"]
        ANALYSIS["Layer 3: Analysis<br/>(4 agents in parallel)"]
        DECISION["Layer 4: Decision<br/>(3 parallel paths)"]
        EXECUTION["Layer 5: Execution"]
    end

    subgraph OUTPUT["📤 OUTPUT"]
        RESULTS_JSON["analysis_results.json"]
        WHATSAPP["WhatsApp Alerts<br/>rich trade messages"]
        PORTFOLIO_UPD["Portfolio Updates<br/>enter/exit/SL/target"]
        CONSOLE["Console Summary<br/>BUY | HOLD | SELL counts"]
    end

    CLI --> ORCH
    CONFIG --> ORCH
    PORTFOLIO_FILE --> PM
    ORCH --> LLM
    ORCH --> PM

    PM --> SYMBOLS
    ORCH --> SYMBOLS

    SYMBOLS --> REGIME
    INDEX --> RDA
    VIX --> RDA
    RDA --> REGIME_OUT

    REGIME_OUT --> PER_STOCK
    SYMBOLS --> PER_STOCK

    PER_STOCK --> OUTPUT
```

---

## 2. Data Ingestion Flow — Multi-Provider Architecture

```mermaid
flowchart LR
    subgraph PROVIDERS["📡 Data Providers (registered in DataProcessor)"]
        YF["YahooFinanceProvider<br/>───────────────────<br/>• OHLCV price history (2y)<br/>• Fundamentals (PE, EPS, margins)<br/>• Market cap, sector, industry<br/>• 52-week high/low"]
        NSE["NSECorporateProvider<br/>───────────────────<br/>• Corporate actions (30 days)<br/>• Announcements (7 days)<br/>• Upcoming events (14 days)<br/>• Dividends, splits, bonuses"]
        RSS["RSSNewsProvider<br/>───────────────────<br/>• Google News RSS<br/>• Multiple financial feeds<br/>• Headlines + summaries<br/>• Last 72 hours"]
    end

    subgraph PROCESSOR["🔧 DataProcessor"]
        DP["DataProcessor<br/>register_provider()<br/>build_stock_context()<br/>build_index_context()"]
    end

    subgraph CONTEXT["📦 StockDataContext (per symbol)"]
        CTX_PRICE["historical_ohlc[]<br/>~500 daily bars"]
        CTX_FUND["fundamentals<br/>PE, EPS, growth, D/E, ROE"]
        CTX_NEWS["news_items[]<br/>headlines + text + source"]
        CTX_EVENT["event_data[]<br/>earnings, dividends, splits"]
        CTX_FLAGS["flags{}<br/>regime, volatility, computed"]
        CTX_META["symbol, last_close,<br/>previous_close, dates"]
    end

    YF --> DP
    NSE --> DP
    RSS --> DP

    DP --> CTX_PRICE
    DP --> CTX_FUND
    DP --> CTX_NEWS
    DP --> CTX_EVENT
    DP --> CTX_FLAGS
    DP --> CTX_META
```

---

## 3. Analysis Layer — 4 Agents in Parallel

```mermaid
flowchart TB
    CTX["StockDataContext"]

    CTX --> TECH & FUND & SENT & EVT

    subgraph PARALLEL_ANALYSIS["⚡ Layer 3: 4 Analysis Agents (Parallel)"]
        TECH["🔧 TechnicalAgent<br/>──────────────<br/>• RSI (overbought/oversold)<br/>• MACD signal (buy/sell/neutral)<br/>• Volatility measure<br/>• Breakout flag<br/>• Trend direction<br/>──────────────<br/>Output: TechnicalSignal"]

        FUND["📊 FundamentalAgent<br/>──────────────<br/>• PE ratio analysis<br/>• Revenue growth score<br/>• Financial health (D/E, ROE)<br/>• Valuation label<br/>──────────────<br/>Output: FundamentalSignal"]

        SENT["📰 SentimentAgent<br/>──────────────<br/>• News headline analysis<br/>• Positive/negative count<br/>• Sentiment trend<br/>• Confidence score<br/>──────────────<br/>Output: SentimentSignal"]

        EVT["📅 EventAgent<br/>──────────────<br/>• Earnings impact<br/>• Dividend info<br/>• Gap up/down detection<br/>• Event risk level<br/>──────────────<br/>Output: EventSignal"]
    end

    TECH --> BUNDLE
    FUND --> BUNDLE
    SENT --> BUNDLE
    EVT --> BUNDLE

    BUNDLE["AgentFeatureBundle<br/>20 normalized features<br/>+ symbol + date + regime"]
```

---

## 4. Triple-Path Decision Engine (Parallel)

```mermaid
flowchart TB
    BUNDLE["AgentFeatureBundle<br/>(20 features from 4 agents)"]
    REGIME["RegimeSignal"]
    RAG["PatternStore (RAG)"]
    MEM["TradeMemory"]

    BUNDLE --> PATH1 & PATH2 & PATH3

    subgraph PARALLEL_DECISION["⚡ Layer 4: 3 Decision Paths (Parallel)"]
        subgraph PATH1["Path 1: Rule-Based (~1ms)"]
            JUDGE["JudgeAgent<br/>──────────────<br/>• Weighted sum of 20 features<br/>• Mode-specific weights (Optuna)<br/>• BUY if prob > 0.766<br/>• SELL if prob < 0.218<br/>──────────────<br/>Output: JudgeDecision"]
        end

        subgraph PATH2["Path 2: AI Debate (~8s)"]
            DC["DebateContext Builder<br/>──────────────<br/>• Last 20 OHLC bars<br/>• Last 10 news headlines<br/>• Real fundamentals (PE, growth)<br/>• 52-week context<br/>• RAG: similar past setups<br/>• Memory: past mistakes"]

            BULL["🟢 BullAgent (LLM)<br/>3-5 BUY arguments<br/>with real evidence"]
            BEAR["🔴 BearAgent (LLM)<br/>3-5 SELL arguments<br/>with real evidence"]
            DEBATE["⚖️ DebateAgent (LLM)<br/>Evaluates arguments<br/>picks winner + confidence"]

            DC --> BULL & BEAR
            BULL --> DEBATE
            BEAR --> DEBATE
        end

        subgraph PATH3["Path 3: XGBoost ML (~1ms)"]
            XGBOOST["XGBoost Classifier<br/>──────────────<br/>• Trained on 48,216 rows<br/>• predict_proba()<br/>• probability of 5-day up move<br/>• Sharpe: 1.73, Win Rate: 62.5%"]
        end
    end

    RAG --> DC
    MEM --> DC
    REGIME --> DC

    JUDGE --> META
    DEBATE --> META
    XGBOOST --> META

    META["Meta-Combiner<br/>──────────────<br/>Rule weight: 0.345<br/>Debate weight: 0.655<br/>If all 3 agree → 95%+ confidence"]

    META --> HYBRID["HybridDecision<br/>final_decision: BUY|SELL|HOLD<br/>final_confidence: 0-100%<br/>agreement: YES|NO"]
```

---

## 5. RAG + Memory Learning Loop

```mermaid
flowchart TB
    subgraph DURING_ANALYSIS["During Analysis"]
        FEATURES["AgentFeatureBundle<br/>20-dim feature vector"]
        PS_SEARCH["PatternStore.search_similar()<br/>NumPy cosine similarity<br/>top_k=10"]
        SIM_RESULT["SimilarityResult<br/>• avg_return_5d<br/>• positive_rate<br/>• max_drawdown<br/>• similar records"]

        TM_SEARCH["TradeMemory.get_history()<br/>Past trades for this symbol<br/>+ mistake warnings"]
        MEM_RESULT["Past Trades<br/>• date, decision, PnL<br/>• outcome notes<br/>Mistake Warnings<br/>• 'Last SELL in bear → -4.2%'"]
    end

    subgraph FEEDS_INTO["Feeds Into Debate"]
        DC2["DebateContext<br/>similar_past_setups[]<br/>past_trades_this_symbol[]<br/>mistake_warnings[]"]
    end

    subgraph AFTER_DECISION["After Decision (Recording)"]
        PS_RECORD["PatternStore.record()<br/>Store feature vector<br/>+ symbol, date, decision, regime"]
        TM_RECORD["TradeMemory.record_decision()<br/>Store BUY/SELL with:<br/>confidence, entry_price,<br/>regime, key_reasons"]
    end

    subgraph AFTER_EXIT["After Position Exit"]
        TM_RESOLVE["TradeMemory.resolve_trade()<br/>Record outcome:<br/>PnL%, hit_stop_loss,<br/>hit_target, outcome_notes"]
    end

    FEATURES --> PS_SEARCH --> SIM_RESULT --> DC2
    FEATURES --> TM_SEARCH --> MEM_RESULT --> DC2

    DC2 -->|"Bull/Bear agents<br/>cite past evidence"| DECISION_OUT["Better Decisions"]

    DECISION_OUT --> PS_RECORD
    DECISION_OUT --> TM_RECORD
    TM_RECORD -->|"position closed"| TM_RESOLVE

    TM_RESOLVE -->|"more data = smarter RAG"| PS_SEARCH
    TM_RESOLVE -->|"more outcomes = better warnings"| TM_SEARCH

    style DECISION_OUT fill:#2d6a4f,color:#fff
```

---

## 6. Execution & Portfolio Management Flow

```mermaid
flowchart TB
    HYBRID["HybridDecision<br/>BUY / SELL / HOLD"]

    HYBRID -->|"BUY or SELL"| TP
    HYBRID -->|"HOLD"| SKIP["Skip Execution<br/>No trade plan needed"]

    subgraph EXECUTION["Layer 5: Execution Pipeline"]
        TP["TradePlannerAgent<br/>──────────────<br/>• entry_price (current)<br/>• stop_loss_price (ATR-based)<br/>• target_price (R:R ratio)<br/>• position_size (% of capital)<br/>• risk_reward_ratio"]

        RM["RiskManagerAgent<br/>──────────────<br/>• Portfolio-level risk check<br/>• Regime risk multiplier<br/>• Max position count check<br/>• Trade blocking logic"]

        TP --> RM

        RM -->|"BLOCKED"| BLOCKED["Trade Blocked<br/>reason logged"]
        RM -->|"APPROVED"| EXEC["TradeExecutor<br/>(MockBroker)<br/>execute BUY/SELL order"]
    end

    subgraph PORTFOLIO["Portfolio Manager"]
        PM2["PortfolioManager<br/>──────────────<br/>• open_position() on BUY<br/>• close_position() on SELL<br/>• Check SL / Target hits<br/>• Update current prices"]
        
        WL["Watchlist<br/>10 stocks with priority<br/>last_signal updated"]
        HOLDINGS["Holdings<br/>symbol, qty, entry_price<br/>SL, target, trailing_stop"]
        PNL["P&L Tracking<br/>realized + unrealized<br/>win rate, avg win/loss"]
    end

    subgraph POSITION_MGMT["Position Monitoring"]
        PMA["PositionManagementAgent<br/>──────────────<br/>• Trailing stop updates<br/>• Time-based exits<br/>• Regime-based adjustments"]
    end

    EXEC --> PM2
    PM2 --> WL & HOLDINGS & PNL
    HOLDINGS --> PMA
    PMA -->|"exit signal"| PM2
```

---

## 7. Output Formats

```mermaid
flowchart LR
    PIPELINE["Pipeline Results"]

    PIPELINE --> OUT1 & OUT2 & OUT3 & OUT4 & OUT5

    subgraph OUTPUTS["📤 Output Channels"]
        OUT1["📄 analysis_results.json<br/>─────────────────────<br/>{<br/>  timestamp, symbols_analyzed,<br/>  buy_count, hold_count, sell_count,<br/>  results: [{<br/>    symbol, date, regime,<br/>    technical, fundamental,<br/>    sentiment, event,<br/>    judge_decision,<br/>    debate: {bull, bear, winner},<br/>    hybrid_decision,<br/>    trade_plan, risk_assessment<br/>  }]<br/>}"]

        OUT2["📱 WhatsApp Alert<br/>─────────────────────<br/>>>> BUY/SELL SIGNAL <<<br/>Stock, Price, Confidence<br/>--- TECHNICALS ---<br/>RSI, MACD, Trend, S/R<br/>--- FUNDAMENTALS ---<br/>PE, Growth, Health<br/>--- AI DEBATE ---<br/>Bull vs Bear summary<br/>--- FINAL VERDICT ---<br/>Rule + Debate agreement"]

        OUT3["💼 Portfolio State<br/>─────────────────────<br/>portfolio.json<br/>• holdings[]: symbol, qty,<br/>  entry, SL, target, PnL<br/>• watchlist[]: symbol,<br/>  priority, last_signal<br/>• cash, total_value"]

        OUT4["🧠 TradeMemory<br/>─────────────────────<br/>trade_memory.jsonl<br/>(append-only)<br/>• decision + reasoning<br/>• entry_price, regime<br/>• outcome when closed<br/>• PnL%, SL/target hit"]

        OUT5["📊 PatternStore<br/>─────────────────────<br/>data/pattern_store/<br/>• 20-dim feature vectors<br/>• symbol, date, decision<br/>• confidence, regime<br/>• For future RAG search"]
    end
```

---

## 8. 6 Trading Modes — Adaptive Switching

```mermaid
flowchart TB
    subgraph MODES["6 Trading Modes"]
        CON["🛡️ Conservative<br/>Buy: 0.80 | Max: 3 | Hold: 10d<br/>Best for: Bear markets"]
        AGG["🚀 Aggressive<br/>Buy: 0.55 | Max: 10 | Hold: 5d<br/>Best for: Bull markets"]
        MOM["📈 Momentum<br/>Buy: 0.60 | Max: 8 | Hold: 7d<br/>Best for: Trending markets"]
        VAL["💎 Value<br/>Buy: 0.70 | Max: 5 | Hold: 15d<br/>Best for: Sideways markets"]
        SCA["⚡ Scalper<br/>Buy: 0.50 | Max: 12 | Hold: 2d<br/>Best for: High volatility"]
        ADA["🔄 Adaptive<br/>Buy: Dynamic | Max: Dynamic<br/>Best for: All markets"]
    end

    subgraph ADAPTIVE_LOGIC["Adaptive Mode Auto-Switch Logic"]
        NIFTY["NIFTY 50<br/>SMA20 vs SMA50"]
        VIXCHECK["India VIX<br/>Volatility Level"]

        NIFTY -->|"SMA20 > SMA50"| BULL_DET["Bull Detected"]
        NIFTY -->|"SMA20 < SMA50"| BEAR_DET["Bear Detected"]
        NIFTY -->|"Flat crossover"| SIDE_DET["Sideways Detected"]
        VIXCHECK -->|"VIX > 20"| HIGH_VOL["High Volatility"]
        VIXCHECK -->|"VIX < 15"| LOW_VOL["Low Volatility"]

        BULL_DET --> SW_AGG["Switch → Aggressive"]
        BEAR_DET --> SW_CON["Switch → Conservative"]
        SIDE_DET --> SW_VAL["Switch → Value"]
        HIGH_VOL --> SW_SCA["Switch → Scalper"]
    end

    ADA --> ADAPTIVE_LOGIC
```

---

## 9. Complete Sequence — Single Stock Analysis

```mermaid
sequenceDiagram
    participant CLI as CLI / Runner
    participant ORCH as Orchestrator
    participant DP as DataProcessor
    participant YF as Yahoo Finance
    participant NSE as NSE Corporate
    participant RSS as RSS News
    participant RD as RegimeDetector
    participant TA as TechnicalAgent
    participant FA as FundamentalAgent
    participant SA as SentimentAgent
    participant EA as EventAgent
    participant JA as JudgeAgent
    participant BULL as BullAgent (LLM)
    participant BEAR as BearAgent (LLM)
    participant DA as DebateAgent (LLM)
    participant PS as PatternStore (RAG)
    participant TM as TradeMemory
    participant TPL as TradePlanner
    participant RM as RiskManager
    participant PM as PortfolioManager
    participant WA as WhatsApp

    CLI->>ORCH: run_for_symbols(["RELIANCE"], mode="momentum")
    
    Note over ORCH: Step 1: Index & Regime (ONCE)
    ORCH->>DP: build_index_context("^NSEI")
    DP->>YF: Fetch NIFTY 50 OHLCV
    YF-->>DP: Index data
    DP-->>ORCH: index_data
    ORCH->>RD: detect_regime(index_ohlc, VIX)
    RD-->>ORCH: RegimeSignal (bear_trend, high_vol)

    Note over ORCH: Step 2: Fetch Stock Data
    ORCH->>DP: build_stock_context("RELIANCE")
    par Data Providers Fetch in Parallel
        DP->>YF: Price + Fundamentals
        DP->>NSE: Corporate Actions
        DP->>RSS: News Headlines
    end
    YF-->>DP: OHLCV + PE + EPS + margins
    NSE-->>DP: Dividends + splits + events
    RSS-->>DP: 10+ news articles
    DP-->>ORCH: StockDataContext

    Note over ORCH: Step 3: 4 Analysis Agents (Parallel)
    par Analysis Agents
        ORCH->>TA: run(stock_context)
        ORCH->>FA: run(stock_context)
        ORCH->>SA: run(stock_context)
        ORCH->>EA: run(stock_context)
    end
    TA-->>ORCH: TechnicalSignal (RSI=48, MACD=SELL)
    FA-->>ORCH: FundamentalSignal (PE=23.1, health=11)
    SA-->>ORCH: SentimentSignal (negative trend)
    EA-->>ORCH: EventSignal (no major events)

    Note over ORCH: Step 4a: Rule-Based Decision
    ORCH->>JA: run(bundle of 20 features)
    JA-->>ORCH: JudgeDecision (SELL, prob=0.13, conf=87%)

    Note over ORCH: Step 4b: RAG + Memory Enrichment
    ORCH->>PS: search_similar(features, top_k=10)
    PS-->>ORCH: 3 similar setups (avg +2.1%, 67% positive)
    ORCH->>TM: get_history("RELIANCE") + get_mistakes()
    TM-->>ORCH: Past trades + warnings

    Note over ORCH: Step 4c: AI Debate
    ORCH->>BULL: argue BUY case (with DebateContext)
    ORCH->>BEAR: argue SELL case (with DebateContext)
    BULL-->>ORCH: BuyArg (55% conf, 3 points)
    BEAR-->>ORCH: SellArg (80% conf, 3 points)
    ORCH->>DA: evaluate(bull_arg, bear_arg)
    DA-->>ORCH: Bear wins → SELL

    Note over ORCH: Step 4d: Hybrid Combiner
    ORCH->>ORCH: combine(rule=SELL 87%, debate=SELL 80%)
    Note over ORCH: Agreement=YES → SELL 95%

    Note over ORCH: Step 5: Execution
    ORCH->>TPL: create_trade_plan(SELL, price=1424)
    TPL-->>ORCH: TradePlan (exit@1424, SL=1307, target=1218)
    ORCH->>RM: assess_risk([SELL decision], regime)
    RM-->>ORCH: Risk=moderate, NOT blocked

    Note over ORCH: Step 6: Portfolio + Record
    ORCH->>PM: process_signals(results, auto_exit=true)
    PM-->>ORCH: Exited RELIANCE position
    ORCH->>PS: record(features, "RELIANCE", "SELL")
    ORCH->>TM: record_decision("RELIANCE", "SELL", conf=95%)

    Note over ORCH: Step 7: Output
    ORCH->>CLI: analysis_results.json
    ORCH->>WA: Rich SELL alert to +91XXXXXXXXXX
```

---

## 10. System Component Map

```mermaid
graph TB
    subgraph AGENTS["14 Agents"]
        A1["TechnicalAgent"]
        A2["FundamentalAgent"]
        A3["SentimentAgent"]
        A4["EventAgent"]
        A5["JudgeAgent"]
        A6["RegimeDetectorAgent"]
        A7["BullAgent 🟢"]
        A8["BearAgent 🔴"]
        A9["DebateAgent ⚖️"]
        A10["TradePlannerAgent"]
        A11["RiskManagerAgent"]
        A12["PositionManagementAgent"]
        A13["StockDiscoveryAgent"]
        A14["BacktestAgent"]
    end

    subgraph CONTROLLERS["5 Controllers"]
        C1["DataProcessor<br/>(multi-provider)"]
        C2["PortfolioManager<br/>(watchlist + holdings)"]
        C3["TradeExecutor<br/>(MockBroker)"]
        C4["PatternStore<br/>(RAG vector search)"]
        C5["TradeMemory<br/>(JSONL learning)"]
    end

    subgraph DATA_PROVIDERS["3 Data Providers"]
        P1["YahooFinanceProvider"]
        P2["NSECorporateProvider"]
        P3["RSSNewsProvider"]
    end

    subgraph EXTERNAL["External Services"]
        E1["Azure OpenAI<br/>GPT-4.1"]
        E2["Yahoo Finance API"]
        E3["NSE India"]
        E4["Google News RSS"]
        E5["WhatsApp Web<br/>(pywhatkit)"]
    end

    P1 --> E2
    P2 --> E3
    P3 --> E4
    A7 & A8 & A9 --> E1
    C3 --> E5

    style AGENTS fill:#1a365d,color:#fff
    style CONTROLLERS fill:#2d6a4f,color:#fff
    style DATA_PROVIDERS fill:#7c3aed,color:#fff
    style EXTERNAL fill:#92400e,color:#fff
```
