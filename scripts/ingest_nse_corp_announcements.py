"""
Run NSE Corporate Announcements ingestion.

Run from project root (LCF/LCF):
    python -m scripts.ingest_nse_corp_announcements
"""

import os
import sys

# Ensure project root is in sys.path so `import src...` works
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

print("✅ Script loaded")
print("✅ Project root:", PROJECT_ROOT)

# ✅ This import path MUST match your folder structure:
# src/ingestion/nse/nse_corp_announcements.py
from src.ingestion.nse.nse_corp_announcements import ingest_once  # noqa: E402


def main() -> None:
    print("✅ Entered main()")
    fetched, inserted = ingest_once()
    print("✅ Ingestion finished")
    print(f"Fetched items  : {fetched}")
    print(f"Inserted items: {inserted}")


if __name__ == "__main__":
    print("✅ __main__ block executing")
    main()