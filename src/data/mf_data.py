"""Indian mutual-fund data client.

Primary source: mfapi.in (free, no key, mirrors AMFI NAV data).
Falls back to cached values when offline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


MFAPI_BASE = "https://api.mfapi.in/mf"


def _cache_path(cache_dir: str, scheme_code: str) -> str:
    return os.path.join(cache_dir, f"mf_{scheme_code}.json")


def _fresh(path: str, max_age_days: int) -> bool:
    if not os.path.exists(path):
        return False
    return (time.time() - os.path.getmtime(path)) < max_age_days * 86400


def fetch_scheme_history(
    scheme_code: str,
    cache_dir: str,
    max_age_days: int = 1,
) -> Optional[Dict[str, Any]]:
    """Return mfapi.in payload: {meta:..., data:[{date, nav}, ...]}."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, scheme_code)
    if _fresh(path, max_age_days):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, json.JSONDecodeError):
            pass

    if requests is None:
        return None

    try:
        resp = requests.get(f"{MFAPI_BASE}/{scheme_code}", timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)
    return data


def compute_rolling_return_pct(navs: List[Dict[str, str]], years: int) -> Optional[float]:
    """Compute annualized return over `years` from mfapi nav series.

    navs[0] is the most recent. Each item: {"date": "DD-MM-YYYY", "nav": "..."}.
    """
    if not navs or years <= 0:
        return None
    try:
        latest_nav = float(navs[0]["nav"])
    except (KeyError, ValueError):
        return None

    target_idx = years * 252  # approx trading days/year
    if target_idx >= len(navs):
        target_idx = len(navs) - 1
    try:
        old_nav = float(navs[target_idx]["nav"])
    except (KeyError, ValueError, IndexError):
        return None

    if old_nav <= 0:
        return None
    total_return = latest_nav / old_nav
    annualized = total_return ** (1.0 / years) - 1.0
    return round(annualized * 100.0, 2)
