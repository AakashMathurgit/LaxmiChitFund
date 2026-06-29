# LCF - Multi-Agent Framework

A LangGraph-based multi-agent system for orchestrating complex workflows with deterministic execution and provenance tracking.

## Features

- **LangGraph Integration**: Built-in support for LangGraph state machine workflows
- **Adapter Pattern**: Flexible adapter system for LLM and service integrations
- **Deterministic Execution**: Reproducible outputs with integrity checksums
- **Provenance Tracking**: Full audit trail for all agent executions

## Project Structure

```
LCF/
├── main.py              # Entry point
├── config.yaml          # Configuration
├── requirements.txt     # Dependencies
├── configs/             # Agent configurations
├── data/                # Data files and schemas
├── docs/                # Documentation
├── notebooks/           # Jupyter notebooks
├── prompts/             # Prompt templates
├── rules/               # Business rules
├── scripts/             # Utility scripts
└── src/
    ├── main/
    │   └── agents/
    │       ├── interfaces/  # Agent base classes
    │       └── adapters/    # Adapter implementations
    ├── pipeline/            # Orchestration logic
    ├── evaluation/          # Metrics and evaluation
    ├── utils/               # Utilities
    └── tests/               # Test suite
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure your settings in `config.yaml`

3. Run the pipeline:
   ```bash
   python main.py
   ```

## Long-horizon Financial Advisor

Separate pipeline for 3-month+ investing across US stocks, Indian stocks, and mutual funds.
Includes a `ProTraderPortfolioAnalyzer` that mines SEC 13F filings, NSE bulk deals, and AMFI
disclosures to surface what well-known investors are buying/selling and why.

Quick start:

```bash
# 1. Copy templates and fill in your details
cp configs/investor_profile.example.yaml configs/investor_profile.yaml
cp data/advisor/portfolio_in.example.json data/advisor/portfolio_in.json
cp data/advisor/portfolio_us.example.json data/advisor/portfolio_us.json
cp data/advisor/portfolio_mf.example.json data/advisor/portfolio_mf.json

# 2. Run a monthly advisor report
python advisor_main.py

# 3. Find output under data/advisor_reports/advisor_<timestamp>.md
```

Tune behaviour in `configs/long_term_investor.yaml`. Tracked investors live in
`data/pro_traders/watchlist.yaml`.

Architecture:

```
advisor_main.py
  └─ AdvisorOrchestrator
       ├─ ProTraderPortfolioAnalyzer    (SEC 13F + NSE bulk deals + AMFI)
       ├─ PortfolioAdvisorAgent         (per-holding HOLD/ADD/TRIM/EXIT)
       ├─ AssetAllocatorAgent           (top-down targets + drift)
       ├─ MutualFundAgent               (mfapi.in NAV → rolling returns)
       ├─ SIPPlannerAgent               (next-month SIP allocation)
       ├─ TaxAwareAgent                 (India LTCG/STCG impact)
       └─ GoalTrackerAgent              (gap-to-goal + extra SIP needed)
```

Personal data (`investor_profile.yaml`, portfolio JSON, cached reports) is gitignored.

## License

MIT License
