# LCF Architecture — Complete Mermaid Diagrams

## 1. End-to-End System Flow

```mermaid
flowchart TB
    subgraph ENTRY["ENTRY POINT"]
        CLI["run_orchestrator_pipeline.py<br/>--mode momentum<br/>--source auto<br/>--stocks 5<br/>--auto-trade<br/>--whatsapp +91XXX<br/>--show-portfolio"]
    end

    subgraph CONFIG["CONFIGURATION LAYER"]
        TP["TuningParams<br/>180+ parameters<br/>10 agent categories"]
        MODES["6 Trading Modes<br/>conservative | aggressive<br/>momentum | value<br/>scalper | adaptive"]
        YAML["tuning_params.yaml<br/>+ tuning_params_optimized.yaml"]
        MODES --> TP
        YAML --> TP
    end

    subgraph PERSISTENCE["PERSISTENT STORAGE"]
        WL[("watchlist.json<br/>10 stocks, priority levels<br/>last_signal tracking")]
        PF[("portfolio.json<br/>Holdings, Cash Rs.10L<br/>Trade History, P&L")]
        PS[("PatternStore<br/>vectors.npz + records.jsonl<br/>NumPy cosine similarity")]
        TM[("trade_memory.jsonl<br/>Append-only decisions<br/>Outcomes + Mistakes")]
        MODELS[("models/<br/>XGBoost per-mode<br/>Optimized weights")]
    end

    subgraph DATA_SOURCES["EXTERNAL DATA SOURCES"]
        YF["Yahoo Finance API<br/>OHLCV, Fundamentals<br/>Balance Sheet, Analyst"]
        NSE["NSE Corporate<br/>Corporate Events<br/>Board Meetings"]
        RSS["Google News RSS<br/>Headlines per stock<br/>12-month history"]
        VIX["India VIX<br/>Volatility Index<br/>Fear gauge"]
    end

    subgraph SYMBOL_RESOLUTION["STEP 1: SYMBOL RESOLUTION"]
        direction LR
        SDA["StockDiscoveryAgent<br/>Screen NIFTY 50 universe<br/>Volume spikes, Breakouts<br/>News movers, 52-week extremes"]
        SR_LOGIC{"Source?"}
        SR_AUTO["AUTO mode<br/>1. Portfolio holdings<br/>2. Watchlist high-priority<br/>3. Discovery fills remaining"]
        SR_WL["WATCHLIST mode<br/>Top N from watchlist"]
        SR_DISC["DISCOVERY mode<br/>Screen full NIFTY 50"]
        SR_FILE["FILE mode<br/>Read stocks.txt"]
        SYMBOLS["Final Symbol List<br/>e.g. RELIANCE, TCS,<br/>HDFCBANK, INFY, ICICIBANK"]
    end

    subgraph ORCHESTRATOR["STEP 2: PIPELINE ORCHESTRATOR"]
        direction TB
        INIT["PipelineOrchestrator.__init__<br/>Load mode config<br/>Initialize 14 agents<br/>Connect PatternStore + TradeMemory"]

        subgraph DATA_LAYER["DATA COLLECTION"]
            DP["DataProcessor<br/>Multi-provider aggregation<br/>Yahoo + NSE + RSS"]
            IDX["build_index_context<br/>NIFTY 50 + VIX"]
            SDC["build_stock_context<br/>StockDataContext per symbol<br/>OHLCV + Fund + News + Events"]
        end

        subgraph REGIME_LAYER["MARKET REGIME DETECTION (once per run)"]
            RDA["RegimeDetectorAgent<br/>SMA20 vs SMA50 crossover<br/>VIX level classification"]
            RS["RegimeSignal<br/>bull_trend | bear_trend | sideways<br/>low | moderate | high volatility<br/>confidence %"]
        end

        subgraph ANALYSIS_LAYER["ANALYSIS AGENTS (per stock, parallel)"]
            TA["TechnicalAgent<br/>RSI (14-day)<br/>MACD signal<br/>Trend direction<br/>Breakout flag<br/>Support & Resistance<br/>Volatility %"]
            FA["FundamentalAgent<br/>PE Ratio & Forward PE<br/>Revenue Growth %<br/>Operating Margin<br/>ROE, D/E Ratio<br/>Valuation label<br/>Financial Health score"]
            SA["SentimentAgent<br/>LLM-powered (GPT-4.1)<br/>News headline analysis<br/>Positive/Negative count<br/>Sentiment trend<br/>Confidence score"]
            EA["EventAgent<br/>Earnings dates<br/>Dividends (ex-date, amount)<br/>Stock splits<br/>Price gaps detection<br/>Event risk level"]
        end

        subgraph DECISION_ENGINE["DECISION ENGINE (per stock)"]
            direction TB

            subgraph RULE_PATH["PATH 1: Rule-Based Scoring"]
                AFB["AgentFeatureBundle<br/>20 normalized features<br/>tech_score, fund_score<br/>sent_score, evt_score<br/>rsi, macd, trend, breakout<br/>valuation, growth, health<br/>+ similarity from RAG"]
                JA["JudgeAgent<br/>Weighted sum of features<br/>Mode-specific weights<br/>buy_threshold, sell_threshold"]
                JD["JudgeDecision<br/>decision: BUY/SELL/HOLD<br/>prob_up_5d: 0-1<br/>expected_return_5d<br/>confidence, position_size"]
            end

            subgraph DEBATE_PATH["PATH 2: AI Debate Flow"]
                DC["DebateContext<br/>Real price data + OHLC bars<br/>Actual PE ratios, headlines<br/>52-week high/low range<br/>Agent signal summaries<br/>Similar past setups (RAG)<br/>Past trade mistakes (Memory)"]
                BULL["BullAgent (LLM)<br/>Argues BUY case<br/>3-5 evidence-based points<br/>Confidence 0-100%<br/>Full reasoning paragraph"]
                BEAR["BearAgent (LLM)<br/>Argues SELL case<br/>3-5 evidence-based points<br/>Confidence 0-100%<br/>Full reasoning paragraph"]
                DA["DebateAgent<br/>Count arguments<br/>Weight by confidence<br/>Determine winner<br/>bull_strength vs bear_strength"]
                DD["DebateDecision<br/>winning_side: bull/bear/neutral<br/>confidence, reasoning"]
            end

            HC["HybridCombiner<br/>final = rule_weight * judge + debate_weight * debate<br/>Check agreement between paths<br/>Boost confidence if both agree"]
            HD["HybridDecision<br/>final_decision: BUY/SELL/HOLD<br/>final_confidence: 0-100%<br/>rule_decision + debate_decision<br/>agreement: true/false"]
        end

        subgraph EXECUTION_LAYER["EXECUTION PLANNING (BUY/SELL only)"]
            TPA["TradePlannerAgent<br/>Entry type: market/limit<br/>Entry price<br/>Stop Loss price & %<br/>Target price & %<br/>Risk:Reward ratio<br/>Position size %<br/>Suggested shares<br/>Hold period (days)<br/>Support & Resistance levels"]
            RMA["RiskManagerAgent<br/>Portfolio exposure check<br/>Regime risk multiplier<br/>Max positions check<br/>Per-stock risk level<br/>Trade blocking logic"]
            PMA["PositionManagementAgent<br/>Trailing stop monitoring<br/>Partial exit rules<br/>Time-based exit"]
        end
    end

    subgraph POST_PIPELINE["STEP 3: POST-PIPELINE ACTIONS"]
        direction TB

        subgraph PORTFOLIO_MGMT["PORTFOLIO MANAGEMENT"]
            PM["PortfolioManager<br/>process_signals()"]
            PM_UPDATE["Update prices for holdings"]
            PM_SL["Check stop losses"]
            PM_TGT["Check targets"]
            PM_EXIT["Auto-exit on SELL signal"]
            PM_ENTER["Auto-enter on BUY signal<br/>(if --auto-trade)"]
            PM_SUMMARY["portfolio_text_summary()<br/>Holdings, P&L, Watchlist"]
        end

        subgraph ALERTS["WHATSAPP ALERTS"]
            MC["MessageController<br/>pywhatkit automation"]
            MSG_TRADE["Trade Alert Message<br/>Stock, Price, Confidence<br/>Trade Plan (SL/Target/R:R)<br/>Technicals (RSI/MACD/Trend)<br/>Fundamentals (PE/Growth)<br/>AI Debate (Bull vs Bear)<br/>Final Verdict + Risk"]
            MSG_DAILY["Daily Summary Message<br/>BUY/SELL/HOLD counts<br/>Per-stock breakdown<br/>Market regime<br/>Confidence stats"]
            MSG_PORT["Portfolio Status<br/>Holdings, Unrealized P&L<br/>Closed trades, Win rate"]
        end

        SAVE["analysis_results.json<br/>Full results per stock"]
    end

    subgraph LEARNING["STEP 4: LEARNING LOOP"]
        REC_PS["PatternStore.record()<br/>Save 20-dim feature vector<br/>For future RAG similarity"]
        REC_TM["TradeMemory.record_decision()<br/>Save decision + reasons<br/>Track for outcome later"]
        FEEDBACK["Future runs use:<br/>- Similar past setups<br/>- Past mistakes<br/>- Outcome history"]
    end

    %% === CONNECTIONS ===

    %% Entry -> Config
    CLI --> SR_LOGIC
    TP --> INIT

    %% Symbol Resolution
    SR_LOGIC -->|auto| SR_AUTO
    SR_LOGIC -->|watchlist| SR_WL
    SR_LOGIC -->|discovery| SR_DISC
    SR_LOGIC -->|file| SR_FILE
    WL --> SR_AUTO
    PF --> SR_AUTO
    SDA --> SR_AUTO
    WL --> SR_WL
    SDA --> SR_DISC
    SR_AUTO --> SYMBOLS
    SR_WL --> SYMBOLS
    SR_DISC --> SYMBOLS
    SR_FILE --> SYMBOLS

    %% Discovery agent data
    YF --> SDA

    %% Orchestrator init
    SYMBOLS --> INIT
    INIT --> DP
    INIT --> RDA

    %% Data sources -> DataProcessor
    YF --> DP
    NSE --> DP
    RSS --> DP
    VIX --> DP

    %% Data flow
    DP --> IDX
    DP --> SDC
    IDX --> RDA
    RDA --> RS

    %% Analysis (parallel per stock)
    SDC --> TA
    SDC --> FA
    SDC --> SA
    SDC --> EA

    %% Signals -> Feature Bundle -> Judge
    TA -->|TechnicalSignal| AFB
    FA -->|FundamentalSignal| AFB
    SA -->|SentimentSignal| AFB
    EA -->|EventSignal| AFB
    AFB --> JA
    JA --> JD

    %% Debate Context enrichment
    RS --> DC
    SDC --> DC
    PS -.->|Similar past setups| DC
    TM -.->|Past trades & mistakes| DC

    %% Debate flow
    DC --> BULL
    DC --> BEAR
    BULL -->|BUY argument| DA
    BEAR -->|SELL argument| DA
    DA --> DD

    %% Hybrid combination
    JD --> HC
    DD --> HC
    HC --> HD

    %% Execution
    HD -->|BUY or SELL| TPA
    TPA --> RMA
    RMA --> PMA

    %% Learning loop
    AFB -.->|Record feature vector| REC_PS
    HD -.->|Record decision| REC_TM
    REC_PS -.-> PS
    REC_TM -.-> TM
    PS -.-> FEEDBACK
    TM -.-> FEEDBACK
    FEEDBACK -.-> DC

    %% Post-pipeline
    HD --> PM
    TPA --> PM
    RMA --> PM
    PM --> PM_UPDATE
    PM --> PM_SL
    PM --> PM_TGT
    PM --> PM_EXIT
    PM --> PM_ENTER
    PM --> PM_SUMMARY
    PM --> PF
    PM --> WL

    HD --> MC
    MC --> MSG_TRADE
    MC --> MSG_DAILY
    PM --> MSG_PORT

    HD --> SAVE

    %% Styling
    classDef agent fill:#4a90d9,stroke:#2c5f8a,color:#fff,stroke-width:2px
    classDef llm_agent fill:#7b68ee,stroke:#4a3aaa,color:#fff,stroke-width:2px
    classDef controller fill:#50c878,stroke:#2d8a4e,color:#fff,stroke-width:2px
    classDef storage fill:#f4a460,stroke:#c47832,color:#000,stroke-width:2px
    classDef config fill:#dda0dd,stroke:#aa6eaa,color:#000,stroke-width:2px
    classDef datasource fill:#87ceeb,stroke:#5a9bb5,color:#000,stroke-width:2px
    classDef output fill:#ff6b6b,stroke:#cc4444,color:#fff,stroke-width:2px
    classDef decision fill:#ffd700,stroke:#cca800,color:#000,stroke-width:2px
    classDef action fill:#98fb98,stroke:#5eb85e,color:#000,stroke-width:2px

    class TA,FA,EA,JA,RDA,TPA,RMA,PMA,SDA agent
    class SA,BULL,BEAR,DA llm_agent
    class DP,PM,MC controller
    class WL,PF,PS,TM,MODELS storage
    class TP,MODES,YAML config
    class YF,NSE,RSS,VIX datasource
    class MSG_TRADE,MSG_DAILY,MSG_PORT,SAVE output
    class JD,DD,HD,HC,AFB,DC,RS decision
    class PM_UPDATE,PM_SL,PM_TGT,PM_EXIT,PM_ENTER,PM_SUMMARY action
    class SR_AUTO,SR_WL,SR_DISC,SR_FILE,SYMBOLS action
```

## 2. Per-Stock Agent Pipeline (Sequence)

```mermaid
sequenceDiagram
    autonumber
    participant CLI as Pipeline Runner
    participant PM as PortfolioManager
    participant SDA as StockDiscoveryAgent
    participant DP as DataProcessor
    participant RDA as RegimeDetector
    participant TA as TechnicalAgent
    participant FA as FundamentalAgent
    participant SA as SentimentAgent
    participant EA as EventAgent
    participant JA as JudgeAgent
    participant PS as PatternStore (RAG)
    participant TM as TradeMemory
    participant Bull as BullAgent (LLM)
    participant Bear as BearAgent (LLM)
    participant DA as DebateAgent
    participant TP as TradePlanner
    participant RM as RiskManager
    participant MC as MessageController

    rect rgb(240, 248, 255)
        Note over CLI,SDA: STEP 1 — Symbol Resolution
        CLI->>PM: Load watchlist + portfolio
        PM-->>CLI: Holdings + Watchlist symbols
        CLI->>SDA: Screen NIFTY 50 for remaining slots
        SDA->>DP: Fetch quick data for 30 stocks
        DP-->>SDA: Volume, Price, 52-week, News count
        SDA-->>CLI: Ranked candidates (score + reasons)
        Note over CLI: Final symbols: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK
    end

    rect rgb(255, 248, 240)
        Note over DP,RDA: STEP 2 — Market Context (ONCE)
        CLI->>DP: build_index_context(NIFTY 50)
        DP-->>RDA: Index OHLCV + VIX data
        RDA->>RDA: SMA20 vs SMA50 + VIX classification
        RDA-->>CLI: RegimeSignal (bear_trend, high vol, 80% conf)
    end

    loop For Each Stock Symbol
        rect rgb(240, 255, 240)
            Note over DP,EA: STEP 3 — Data Collection
            CLI->>DP: build_stock_context(symbol)
            DP->>DP: Yahoo Finance + NSE + RSS News
            DP-->>CLI: StockDataContext (OHLCV, Fund, News, Events)
        end

        rect rgb(248, 240, 255)
            Note over TA,EA: STEP 4 — Analysis Agents (Parallel)
            par Run all 4 agents simultaneously
                CLI->>TA: Analyze technicals
                TA-->>CLI: RSI=48, MACD=sell, Trend=bearish, Score=13/100
            and
                CLI->>FA: Analyze fundamentals
                FA-->>CLI: PE=23.1, Growth=41%, Health=11%, Score=39/100
            and
                CLI->>SA: Analyze sentiment (LLM)
                SA-->>CLI: Sentiment=0.50, Trend=stable, Conf=0%
            and
                CLI->>EA: Analyze events
                EA-->>CLI: Dividend, Gap=-2.1%, Risk=high, Score=40/100
            end
        end

        rect rgb(255, 255, 240)
            Note over JA: STEP 5 — Rule-Based Decision
            CLI->>JA: AgentFeatureBundle (20 features)
            JA->>JA: Weighted scoring with mode thresholds
            JA-->>CLI: SELL (prob_up=12.8%, confidence=87.2%)
        end

        rect rgb(255, 240, 245)
            Note over PS,DA: STEP 6 — AI Debate
            CLI->>PS: Search similar past setups
            PS-->>CLI: 3 similar patterns (avg return, positive rate)
            CLI->>TM: Get past trades for this symbol
            TM-->>CLI: 2 past decisions + 1 mistake warning

            Note over CLI: Build DebateContext with real data
            CLI->>Bull: DebateContext (headlines, PE, 52w range, RAG, memory)
            Bull-->>CLI: BUY 55% — 3 key points + reasoning
            CLI->>Bear: DebateContext (same data)
            Bear-->>CLI: SELL 80% — 4 key points + reasoning

            CLI->>DA: Evaluate Bull vs Bear arguments
            DA->>DA: Count points, weight confidence
            DA-->>CLI: Bear wins (bull_strength=0.39, bear_strength=0.61)
        end

        rect rgb(240, 248, 248)
            Note over DA: STEP 7 — Hybrid Decision
            CLI->>DA: Combine Rule (SELL 87%) + Debate (SELL 80%)
            DA-->>CLI: FINAL: SELL 95% confidence, Agreement=YES
        end

        rect rgb(248, 248, 240)
            Note over TP,RM: STEP 8 — Execution Planning
            CLI->>TP: Create trade plan for SELL
            TP-->>CLI: EXIT @ Rs.1,424, Hold 5 days, Expected -3.7%
            CLI->>RM: Assess portfolio risk
            RM-->>CLI: Risk=very_low, Not blocked
        end

        rect rgb(240, 255, 248)
            Note over PS,TM: STEP 9 — Learning
            CLI->>PS: Record feature vector for future RAG
            CLI->>TM: Record SELL decision + reasons
        end
    end

    rect rgb(255, 245, 238)
        Note over PM,MC: STEP 10 — Post-Pipeline Actions
        CLI->>PM: process_signals(results)
        PM->>PM: Update held positions prices
        PM->>PM: Check stop losses & targets
        PM->>PM: Auto-exit held stocks with SELL signal
        PM-->>CLI: Actions summary

        CLI->>MC: Send WhatsApp alerts
        MC->>MC: Format rich trade alerts
        MC-->>CLI: 5 SELL alerts + 1 daily summary sent
    end
```

## 3. Training & Backtesting Pipeline

```mermaid
flowchart LR
    subgraph FETCH["Data Fetching"]
        F1["fetch_training_data.py<br/>5yr OHLCV: 60,564 rows<br/>49 NIFTY 50 stocks"]
        F2["fetch_historical_news.py<br/>Google News RSS<br/>27,711 articles, 62 stocks"]
        F3["compute_labels.py<br/>Forward returns (5-day)<br/>BUY/SELL/HOLD labels<br/>19.9% BUY, 65.5% HOLD, 14.5% SELL"]
    end

    subgraph TRAIN["Training (3 approaches)"]
        direction TB
        T1["train_and_test.py<br/>XGBoost Classifier<br/>Temporal 80/20 split<br/>No data leakage"]
        T2["train_multi_agent.py<br/>Optuna Bayesian (50 trials)<br/>Rule-based weight optimization<br/>Multi-agent parameter tuning"]
        T3["train_all_modes.py<br/>5 modes x 3 models each<br/>Rule-based + XGBoost + Debate<br/>Per-mode hyperparameters"]
    end

    subgraph ARTIFACTS["Output Artifacts"]
        M1["judge_model.joblib<br/>Sharpe: 1.73 | Win: 62.5%<br/>Avg Return: +2.97%"]
        M2["tuning_params_optimized.yaml<br/>debate_weight: 0.655<br/>buy_threshold: 0.766<br/>sell_threshold: 0.218"]
        M3["models/<mode>_xgboost.joblib<br/>models/<mode>_params.yaml<br/>5 mode-specific models"]
    end

    subgraph BACKTEST["Backtesting"]
        B1["backtest_simulation.py<br/>Rs.100 portfolio<br/>Walk-forward daily"]
        B2["backtest_modes.py<br/>6 modes comparison<br/>Adaptive regime switching<br/>vs NIFTY 50 benchmark"]
    end

    subgraph RESULTS["Results (Mar 2025 - Feb 2026)"]
        R_TABLE["Mode      | Return | Alpha<br/>----------|--------|------<br/>MOMENTUM  | +19.6% | +4.3%<br/>ADAPTIVE  | +17.5% | +2.2%<br/>AGGRESSIVE| +16.5% | +1.3%<br/>NIFTY B&H | +15.3% |   0%<br/>SCALPER   |  +6.7% | -8.6%<br/>VALUE     |  -2.4% |-17.7%"]
        R_METRICS["XGBoost Metrics:<br/>Sharpe: 1.73<br/>Profit Factor: 7.61<br/>Max Drawdown: -3.3%<br/>Alpha vs baseline: +2.59%"]
    end

    F1 --> F3
    F2 --> F3
    F3 --> T1
    F3 --> T2
    F3 --> T3
    T1 --> M1
    T2 --> M2
    T3 --> M3
    M1 --> B1
    M2 --> B1
    M3 --> B2
    B1 --> RESULTS
    B2 --> RESULTS
```

## 4. Component Integration Map

```mermaid
graph LR
    subgraph AGENTS["14 Agents"]
        A1[TechnicalAgent]
        A2[FundamentalAgent]
        A3[SentimentAgent]
        A4[EventAgent]
        A5[JudgeAgent]
        A6[RegimeDetectorAgent]
        A7[BullAgent]
        A8[BearAgent]
        A9[DebateAgent]
        A10[TradePlannerAgent]
        A11[RiskManagerAgent]
        A12[PositionManagementAgent]
        A13[StockDiscoveryAgent]
        A14[LLMAdapter]
    end

    subgraph CONTROLLERS["5 Controllers"]
        C1[DataProcessor]
        C2[PatternStore]
        C3[TradeMemory]
        C4[PortfolioManager]
        C5[MessageController]
    end

    subgraph CONFIG_LAYER["Configuration"]
        CF1[TuningConfig - 180+ params]
        CF2[6 Mode Presets]
        CF3[YAML Overrides]
    end

    subgraph DATA_FILES["Data Files"]
        D1[watchlist.json]
        D2[portfolio.json]
        D3[trade_memory.jsonl]
        D4[pattern_store/]
        D5[training/ models + data]
    end

    ORCH[PipelineOrchestrator<br/>Central coordinator]

    ORCH --> A1 & A2 & A3 & A4 & A5
    ORCH --> A6 & A7 & A8 & A9
    ORCH --> A10 & A11 & A12
    ORCH --> A13
    A3 & A7 & A8 --> A14

    ORCH --> C1 & C2 & C3
    ORCH --> CF1

    RUN[run_orchestrator_pipeline.py] --> ORCH
    RUN --> C4 & C5
    RUN --> A13

    C4 --> D1 & D2
    C3 --> D3
    C2 --> D4
    CF1 --> CF2 & CF3

    style ORCH fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:3px
    style RUN fill:#4caf50,stroke:#2e7d32,color:#fff,stroke-width:3px
```

## 5. Symbol Resolution Decision Tree

```mermaid
flowchart TD
    START["Pipeline Start"] --> SOURCE{"--source flag?"}

    SOURCE -->|auto<br/>DEFAULT| AUTO_START["Collect from 3 sources"]
    SOURCE -->|watchlist| WL_ONLY["Read watchlist.json"]
    SOURCE -->|discovery| DISC_ONLY["Screen NIFTY 50"]
    SOURCE -->|file| FILE_ONLY["Read stocks.txt"]

    AUTO_START --> HOLDINGS["1. Portfolio holdings<br/>(need price updates)"]
    AUTO_START --> WL_HIGH["2. Watchlist high-priority<br/>(RELIANCE, TCS, HDFCBANK)"]
    AUTO_START --> REMAINING{"Slots remaining?"}

    HOLDINGS --> COMBINED["Combined list<br/>(deduplicated)"]
    WL_HIGH --> COMBINED
    COMBINED --> REMAINING

    REMAINING -->|Yes| DISCOVER["3. StockDiscoveryAgent<br/>Screen remaining NIFTY 50<br/>Score by: volume spikes,<br/>breakouts, news, price moves"]
    REMAINING -->|No| FINAL

    DISCOVER --> FINAL["Final Symbol List<br/>Limited to --stocks N"]

    WL_ONLY --> FINAL
    DISC_ONLY --> FINAL
    FILE_ONLY --> FINAL

    FINAL --> PIPELINE["Run Pipeline<br/>Orchestrator.run_for_symbols()"]

    style START fill:#4caf50,color:#fff
    style FINAL fill:#ff9800,color:#fff
    style PIPELINE fill:#2196f3,color:#fff
    style DISCOVER fill:#9c27b0,color:#fff
```

## 6. Triple-Path Decision Engine (Rule-Based + AI Debate + XGBoost in Parallel)

```mermaid
flowchart TB
    subgraph INPUT["INPUT: Per-Stock Signals"]
        SDC["StockDataContext<br/>OHLCV, Fundamentals<br/>News, Events"]
        RS["RegimeSignal<br/>bear_trend | bull_trend<br/>volatility level"]

        subgraph AGENT_SIGNALS["4 Analysis Agent Outputs"]
            TS["TechnicalSignal<br/>RSI=48, MACD=sell<br/>Trend=bearish<br/>Breakout=no<br/>Support=1307, Res=1473<br/>Score: 13/100"]
            FS["FundamentalSignal<br/>PE=23.1, FwdPE=21.8<br/>Growth=41%, Health=11%<br/>Valuation=fair<br/>Score: 39/100"]
            SS["SentimentSignal<br/>Score=0.50, Trend=stable<br/>Positive=0, Negative=0<br/>Confidence=0%"]
            ES["EventSignal<br/>Dividend, Gap=-2.1%<br/>Risk=high<br/>Score: 40/100"]
        end
    end

    subgraph FEATURE_PREP["FEATURE PREPARATION"]
        AFB["AgentFeatureBundle<br/>20 normalized floats<br/>tech_score, tech_rsi, tech_macd<br/>tech_volatility, tech_breakout, tech_trend<br/>fund_score, fund_valuation, fund_growth, fund_health<br/>sent_score, sent_net_ratio, sent_trend, sent_confidence<br/>evt_score, evt_earnings, evt_risk, evt_gap_up, evt_gap_down<br/>+ similarity_avg_return (from RAG)"]
    end

    subgraph TRIPLE_PATH["THREE PARALLEL DECISION PATHS"]
        direction LR

        subgraph PATH1["PATH 1: RULE-BASED<br/>(Weighted Scoring)"]
            direction TB
            JA["JudgeAgent<br/>Weighted sum of 20 features"]
            JA_WEIGHTS["Mode-Specific Weights:<br/>tech: 0.30 | fund: 0.15<br/>sent: 0.10 | evt: 0.10<br/>macd: 0.10 | rsi: 0.08<br/>breakout: 0.08 | trend: 0.09"]
            JA_THRESH["Thresholds:<br/>BUY if prob > 0.60<br/>SELL if prob < 0.35<br/>else HOLD"]
            JD["Rule Decision<br/>SELL 87% confidence<br/>prob_up=12.8%<br/>expected_return=-3.7%"]
            JA --> JA_WEIGHTS --> JA_THRESH --> JD
        end

        subgraph PATH2["PATH 2: AI DEBATE<br/>(LLM Bull vs Bear)"]
            direction TB
            DC["DebateContext<br/>Real headlines, PE ratios<br/>52-week range, raw OHLC<br/>RAG similar setups<br/>Past trade mistakes"]
            BULL["BullAgent (GPT-4.1)<br/>BUY 55%<br/>+ Price rebounded +4.8%<br/>+ PE=23.1 is fair<br/>+ Stable sentiment"]
            BEAR["BearAgent (GPT-4.1)<br/>SELL 80%<br/>- MACD sell, bearish trend<br/>- D/E=35.65, thin margin<br/>- Bear regime, high vol<br/>- No positive catalysts"]
            DA["DebateAgent<br/>Bull strength: 0.39<br/>Bear strength: 0.61<br/>Winner: BEAR"]
            DD["Debate Decision<br/>SELL 80% confidence<br/>Bear wins with 4 points"]
            DC --> BULL & BEAR
            BULL & BEAR --> DA --> DD
        end

        subgraph PATH3["PATH 3: XGBOOST ML<br/>(Trained Classifier)"]
            direction TB
            XGB_IN["Same 20 features<br/>as AgentFeatureBundle"]
            XGB["XGBoost Model<br/>judge_model.joblib<br/>Trained on 48,216 rows<br/>Temporal split, no leakage"]
            XGB_METRICS["Training Metrics:<br/>Sharpe: 1.73<br/>Win Rate: 62.5%<br/>Profit Factor: 7.61"]
            XGB_OUT["ML Decision<br/>predict_proba()<br/>P(BUY)=0.08<br/>SELL with high certainty"]
            XGB_IN --> XGB --> XGB_OUT
            XGB_METRICS -.-> XGB
        end
    end

    subgraph META_COMBINER["META-COMBINER (Consensus Engine)"]
        direction TB
        VOTES["Vote Collection:<br/>Rule-Based: SELL (87%)<br/>AI Debate: SELL (80%)<br/>XGBoost ML: SELL (92%)"]
        AGREEMENT{"All 3 agree?"}
        AGR_YES["STRONG SIGNAL<br/>Boost confidence to 95%+<br/>Increase position size"]
        AGR_PARTIAL["2 of 3 agree<br/>Use majority with<br/>weighted average confidence"]
        AGR_NO["All disagree<br/>Default to HOLD<br/>Flag for manual review"]
        FINAL_DEC["Final HybridDecision<br/>SELL @ 95% confidence<br/>Agreement: 3/3 unanimous<br/>All paths confirmed bearish"]
    end

    subgraph EXECUTION["EXECUTION"]
        TP["TradePlannerAgent<br/>EXIT @ Rs.1,424<br/>Expected drop: -3.7%<br/>Hold: 5 days"]
        RM["RiskManagerAgent<br/>Risk: very_low<br/>Not blocked"]
        PM["PortfolioManager<br/>Auto-exit if holding<br/>Skip entry (SELL signal)"]
    end

    %% Input flow
    TS & FS & SS & ES --> AFB

    %% Feature bundle to all 3 paths
    AFB --> JA
    AFB --> XGB_IN
    SDC --> DC
    RS --> DC

    %% RAG & Memory feed debate
    PS[("PatternStore<br/>RAG")] -.->|Similar setups| DC
    TM[("TradeMemory")] -.->|Past mistakes| DC

    %% All 3 paths feed meta-combiner
    JD --> VOTES
    DD --> VOTES
    XGB_OUT --> VOTES

    VOTES --> AGREEMENT
    AGREEMENT -->|Yes, all 3| AGR_YES
    AGREEMENT -->|2 of 3| AGR_PARTIAL
    AGREEMENT -->|No consensus| AGR_NO
    AGR_YES --> FINAL_DEC
    AGR_PARTIAL --> FINAL_DEC
    AGR_NO --> FINAL_DEC

    %% To execution
    FINAL_DEC --> TP --> RM --> PM

    %% Learning
    FINAL_DEC -.->|Record| TM
    AFB -.->|Record vector| PS

    %% Styling
    classDef rule fill:#4a90d9,stroke:#2c5f8a,color:#fff,stroke-width:2px
    classDef debate fill:#7b68ee,stroke:#4a3aaa,color:#fff,stroke-width:2px
    classDef ml fill:#ff6347,stroke:#cc3322,color:#fff,stroke-width:2px
    classDef signal fill:#87ceeb,stroke:#5a9bb5,color:#000,stroke-width:1px
    classDef decision fill:#ffd700,stroke:#cca800,color:#000,stroke-width:2px
    classDef meta fill:#ff9800,stroke:#e65100,color:#fff,stroke-width:3px
    classDef exec fill:#50c878,stroke:#2d8a4e,color:#fff,stroke-width:2px
    classDef storage fill:#f4a460,stroke:#c47832,color:#000,stroke-width:2px
    classDef agree fill:#98fb98,stroke:#5eb85e,color:#000,stroke-width:2px

    

    class JA,JA_WEIGHTS,JA_THRESH rule
    class DC,BULL,BEAR,DA debate
    class XGB,XGB_IN,XGB_METRICS ml
    class TS,FS,SS,ES signal
    class JD,DD,XGB_OUT,FINAL_DEC decision
    class VOTES,AGREEMENT meta
    class TP,RM,PM exec
    class PS,TM storage
    class AGR_YES,AGR_PARTIAL,AGR_NO agree
```

## 7. Triple-Path Timing (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    participant O as Orchestrator
    participant AFB as FeatureBundle
    participant JA as Rule-Based<br/>(JudgeAgent)
    participant DC as DebateContext
    participant Bull as BullAgent<br/>(GPT-4.1)
    participant Bear as BearAgent<br/>(GPT-4.1)
    participant DA as DebateAgent
    participant XGB as XGBoost<br/>Classifier
    participant HC as HybridCombiner
    participant TP as TradePlanner

    Note over O: Stock: RELIANCE | Mode: MOMENTUM

    O->>AFB: Build 20-dim feature vector from 4 agent signals

    par THREE PATHS RUN IN PARALLEL
        rect rgb(220, 235, 255)
            Note over JA: PATH 1: Rule-Based (~1ms)
            AFB->>JA: 20 features + mode weights
            JA->>JA: Weighted sum scoring
            JA-->>HC: SELL (prob_up=12.8%, conf=87.2%)
        end
    and
        rect rgb(235, 220, 255)
            Note over Bull,Bear: PATH 2: AI Debate (~8 sec)
            O->>DC: Build DebateContext (real data + RAG + memory)
            DC->>Bull: Full context + "argue BUY"
            Bull-->>DA: BUY 55% | 3 key points
            DC->>Bear: Full context + "argue SELL"
            Bear-->>DA: SELL 80% | 4 key points
            DA->>DA: Evaluate arguments
            DA-->>HC: SELL (bear wins, strength=0.61)
        end
    and
        rect rgb(255, 230, 230)
            Note over XGB: PATH 3: XGBoost ML (~1ms)
            AFB->>XGB: Same 20 features
            XGB->>XGB: predict_proba()
            XGB-->>HC: SELL (P(BUY)=0.08)
        end
    end

    Note over HC: All 3 paths complete

    HC->>HC: Check agreement: SELL + SELL + SELL = UNANIMOUS
    HC->>HC: Boost confidence: 95% (all agree)
    HC-->>O: HybridDecision: SELL @ 95% confidence

    O->>TP: Create trade plan
    TP-->>O: EXIT @ Rs.1,424 | Expected -3.7% in 5 days

    Note over O: Total time: ~8 sec<br/>(bottleneck = LLM calls in debate path)
```
