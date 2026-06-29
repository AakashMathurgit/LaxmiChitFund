"""Lightweight market-data helpers for the advisor pipeline.

Wraps yfinance for:
  - Live quotes (current price)
  - Fundamentals dict compatible with FundamentalAgent.analyse()
  - Index OHLC history for RegimeDetectorAgent
  - FX (USD/INR)

All functions cache per-process and fail soft so the pipeline still
produces a report when offline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore


_PROCESS_CACHE: Dict[str, Tuple[float, Any]] = {}
_TTL_SECONDS = 600  # 10 min — advisor runs are infrequent


def _cache_get(key: str) -> Optional[Any]:
    hit = _PROCESS_CACHE.get(key)
    if not hit:
        return None
    ts, val = hit
    if time.time() - ts > _TTL_SECONDS:
        return None
    return val


def _cache_set(key: str, value: Any) -> None:
    _PROCESS_CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------------------
# Quotes / fundamentals
# ---------------------------------------------------------------------------

def get_quote(symbol: str) -> Optional[float]:
    """Latest close in the symbol's native currency. None on failure."""
    if not symbol or yf is None:
        return None
    key = f"quote:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        hist = yf.Ticker(symbol).history(period="5d", auto_adjust=False)
        if hist is None or hist.empty:
            return None
        price = float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None
    _cache_set(key, price)
    return price


def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Return a dict matching FundamentalAgent.analyse() expected keys."""
    if not symbol or yf is None:
        return {}
    key = f"fund:{symbol}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception:
        return {}

    data: Dict[str, Any] = {
        "symbol": symbol,
        "longName": info.get("longName"),
        "shortName": info.get("shortName"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "eps": info.get("trailingEps"),
        "revenue_growth": info.get("revenueGrowth"),
        "profit_margin": info.get("profitMargins"),
        "debt_to_equity": (info.get("debtToEquity") or 0) / 100.0
            if info.get("debtToEquity") else None,
        "roe": info.get("returnOnEquity"),
        "market_cap": info.get("marketCap"),
        "sector": info.get("sector"),
    }
    _cache_set(key, data)
    return data


# ---------------------------------------------------------------------------
# Indexes / regime input
# ---------------------------------------------------------------------------

def get_index_ohlc(symbol: str, period: str = "1y") -> List[Dict[str, float]]:
    """Return list[{open,high,low,close}] for an index symbol."""
    if not symbol or yf is None:
        return []
    key = f"idx:{symbol}:{period}"
    cached = _cache_get(key)
    if cached is not None:
        return cached
    try:
        df = yf.Ticker(symbol).history(period=period, auto_adjust=False)
        if df is None or df.empty:
            return []
        rows = [
            {
                "open": float(r.Open),
                "high": float(r.High),
                "low": float(r.Low),
                "close": float(r.Close),
            }
            for r in df.itertuples()
        ]
    except Exception:
        return []
    _cache_set(key, rows)
    return rows


def get_vix(symbol: str = "^INDIAVIX") -> Optional[float]:
    """Most recent VIX print (India VIX by default)."""
    return get_quote(symbol)


# ---------------------------------------------------------------------------
# FX
# ---------------------------------------------------------------------------

def get_fx(pair: str = "USDINR=X") -> Optional[float]:
    """Most recent FX rate, e.g. USD->INR. None on failure."""
    return get_quote(pair)
