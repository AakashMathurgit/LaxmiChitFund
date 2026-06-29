"""Generate a list of valid US stock symbols (NYSE + NASDAQ).

Fetches from a public API and writes to us_symbols.txt.
Run once to bootstrap, then periodically to keep updated.

Usage:
    python us_stock_tracker/generate_us_symbols.py
"""

import os
import json
import requests

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(_SCRIPT_DIR, "us_symbols.txt")

# SEC EDGAR company tickers (free, no API key needed)
SEC_URL = "https://www.sec.gov/files/company_tickers.json"

# Fallback: predefined list of major US tickers
FALLBACK_SYMBOLS = [
    # FAANG+ / Mag 7
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA",
    # Major tech
    "NFLX", "AMD", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "SQ", "SHOP",
    "SNOW", "PLTR", "COIN", "UBER", "ABNB", "DASH", "PINS", "SNAP", "RBLX",
    "U", "DDOG", "NET", "CRWD", "ZS", "MDB", "TEAM",
    # Semis
    "AVGO", "QCOM", "TXN", "MU", "MRVL", "LRCX", "KLAC", "AMAT", "ASML",
    "ARM", "SMCI", "MARA", "RIOT",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "BLK", "SCHW", "AXP", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "AMGN",
    "GILD", "ISRG", "MRNA", "BNTX",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "HAL", "DVN",
    # Consumer
    "WMT", "COST", "HD", "TGT", "NKE", "SBUX", "MCD", "KO", "PEP", "PG",
    "DIS", "CMCSA", "T", "VZ",
    # Industrial
    "BA", "CAT", "GE", "RTX", "HON", "LMT", "DE", "UPS", "FDX",
    # EV / Clean energy
    "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "SEDG", "FSLR", "PLUG",
    "BE", "RUN", "CHPT",
    # Small cap / Meme / Popular
    "GME", "AMC", "SOFI", "HOOD", "IONQ", "JOBY", "SPCE", "DKNG",
    # ETFs (for reference/context)
    "SPY", "QQQ", "IWM", "DIA", "VTI", "ARKK",
]


def fetch_us_symbols():
    """Fetch US stock symbols from SEC EDGAR."""
    try:
        headers = {"User-Agent": "LCF-StockTracker/1.0 (research)"}
        resp = requests.get(SEC_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        symbols = set()
        for entry in data.values():
            ticker = entry.get("ticker", "").strip().upper()
            if ticker and len(ticker) <= 5 and ticker.isalpha():
                symbols.add(ticker)

        print(f"Fetched {len(symbols)} symbols from SEC EDGAR")
        return sorted(symbols)

    except Exception as e:
        print(f"SEC EDGAR fetch failed ({e}), using fallback list")
        return sorted(set(FALLBACK_SYMBOLS))


def main():
    symbols = fetch_us_symbols()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sym in symbols:
            f.write(sym + "\n")

    print(f"Wrote {len(symbols)} symbols to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
