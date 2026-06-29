"""Quick test: Sentiment Agent + Judge only.

Run from project root:
    cd c:\\Users\\mathuraakash\\source\\repos\\LCF\\LCF
    python -m scripts.test_sentiment_judge
"""

import sys
import os
import json

# Add project root to path so imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline.orchestrator import PipelineOrchestrator

# Path to sample data (relative to this script)
DATA_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sample_stock_news.json")


def load_stock_data(path: str = DATA_FILE):
    """Load stock news data from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # ---- LOAD INPUT DATA ----
    stock_data = load_stock_data()
    print(f"Loaded {len(stock_data)} stocks from {os.path.basename(DATA_FILE)}\n")

    # ---- RUN PIPELINE ----
    orchestrator = PipelineOrchestrator()
    results = orchestrator.run(stock_data)

    # ---- PRINT RESULTS ----
    for r in results:
        print(f"\n{'='*60}")
        print(f"  {r['symbol']}  |  {r['date']}")
        print(f"{'='*60}")

        # Sentiment output
        sent = r["sentiment"]
        print(f"  Sentiment success: {sent['success']}")
        if sent.get("payload") and sent["payload"].get("signal"):
            for k, v in sent["payload"]["signal"].items():
                print(f"    {k}: {v}")

        # Judge decision
        judge = r["judge_decision"]
        print(f"\n  Judge success: {judge['success']}")
        if judge.get("payload"):
            p = judge["payload"]
            print(f"    Decision:        {p.get('decision')}")
            print(f"    P(up 5d):        {p.get('prob_up_5d')}")
            print(f"    Expected return: {p.get('expected_return_5d')}")
            print(f"    Downside risk:   {p.get('downside_risk_prob')}")
            print(f"    Confidence:      {p.get('confidence')}")
            print(f"    Position size:   {p.get('position_size_pct')}")

        print(f"  Timing: {judge.get('timing', {}).get('duration_ms')} ms")


if __name__ == "__main__":
    main()
