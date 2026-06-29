"""Data fetchers for ProTraderPortfolioAnalyzer.

Sources (all free / public):
  - SEC EDGAR 13F-HR filings (US institutional holdings, quarterly).
  - NSE bulk deals & block deals (Indian large transactions, daily).
  - AMFI monthly mutual-fund portfolio disclosures (Indian MF holdings).
  - Investor letters / news (RSS) for rationale extraction.
  - OpenFIGI for ticker → CUSIP mapping.

All fetchers cache aggressively to a configurable cache_dir so we are
polite to upstream APIs.

Network access is wrapped in try/except so callers degrade gracefully
when offline.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore


SEC_BASE = "https://data.sec.gov"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEC_UA = "LCF-Advisor research-script contact@example.com"
NSE_BULK_DEALS = "https://www.nseindia.com/api/historical/cm/bulk-deals"
AMFI_PORTFOLIO_BASE = "https://portal.amfiindia.com/spages/aaa"  # disclosure landing
OPENFIGI_MAP = "https://api.openfigi.com/v3/mapping"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _cache_path(cache_dir: str, key: str) -> str:
    safe = key.replace("/", "_").replace(":", "_")
    return os.path.join(cache_dir, f"{safe}.json")


def _cache_fresh(path: str, max_age_days: int) -> bool:
    if not os.path.exists(path):
        return False
    age_sec = time.time() - os.path.getmtime(path)
    return age_sec < max_age_days * 86400


def _read_cache(path: str) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(path: str, payload: Any) -> None:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# SEC EDGAR 13F
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ThirteenFHolding:
    cusip: str
    issuer_name: str
    ticker: Optional[str]
    shares: float
    value_usd: float
    put_call: Optional[str] = None


@dataclass(slots=True)
class ThirteenFSnapshot:
    cik: str
    investor_name: str
    period_of_report: str               # e.g. "2026-03-31"
    accession_number: str
    holdings: List[ThirteenFHolding] = field(default_factory=list)


def fetch_13f_history(
    cik: str,
    cache_dir: str,
    max_age_days: int = 30,
    n: int = 2,
) -> List[ThirteenFSnapshot]:
    """Fetch the N most-recent 13F-HR snapshots for a CIK, newest first.

    Returns [] when network/parsing fails. Each snapshot's holdings are
    cached individually so subsequent runs are cheap.
    """
    if requests is None:
        return []

    cache_key = f"sec_13f_history_{cik}_n{n}"
    cache_file = _cache_path(cache_dir, cache_key)
    if _cache_fresh(cache_file, max_age_days):
        raw = _read_cache(cache_file)
        if raw:
            return [_snapshot_from_dict(s) for s in raw]

    try:
        cik_padded = str(int(cik)).zfill(10)
        sub_url = f"{SEC_BASE}/submissions/CIK{cik_padded}.json"
        resp = requests.get(sub_url, headers={"User-Agent": SEC_UA}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        cached = _read_cache(cache_file)
        return [_snapshot_from_dict(s) for s in cached] if cached else []

    investor_name = data.get("name", cik)
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accs = recent.get("accessionNumber", [])
    dates = recent.get("reportDate", [])

    indexes = [i for i, f in enumerate(forms) if f in ("13F-HR", "13F-HR/A")][:n]
    if not indexes:
        return []

    snapshots: List[ThirteenFSnapshot] = []
    for idx in indexes:
        accession = accs[idx]
        holdings = _fetch_13f_holdings(cik, accession, cache_dir)
        snapshots.append(ThirteenFSnapshot(
            cik=cik,
            investor_name=investor_name,
            period_of_report=dates[idx],
            accession_number=accession,
            holdings=holdings,
        ))

    payload = [
        {
            "cik": s.cik,
            "investor_name": s.investor_name,
            "period_of_report": s.period_of_report,
            "accession_number": s.accession_number,
            "holdings": [asdict(h) for h in s.holdings],
        }
        for s in snapshots
    ]
    _write_cache(cache_file, payload)
    return snapshots


def fetch_13f_latest(
    cik: str,
    cache_dir: str,
    max_age_days: int = 30,
) -> Optional[ThirteenFSnapshot]:
    """Convenience wrapper returning just the most recent 13F snapshot."""
    history = fetch_13f_history(cik, cache_dir, max_age_days=max_age_days, n=1)
    return history[0] if history else None


def _fetch_13f_holdings(
    cik: str,
    accession_number: str,
    cache_dir: str,
) -> List["ThirteenFHolding"]:
    """Fetch and parse the 13F information table for one filing.

    SEC archive URL pattern:
      https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_no_nodash}/

    The filing index.json lists all files; we look for the *infotable* XML
    (or the second XML if naming differs).
    """
    if requests is None:
        return []

    cik_int = str(int(cik))
    acc_clean = accession_number.replace("-", "")
    base = f"{SEC_ARCHIVES}/{cik_int}/{acc_clean}"
    headers = {"User-Agent": SEC_UA, "Accept": "application/json"}

    try:
        idx = requests.get(f"{base}/index.json", headers=headers, timeout=30)
        idx.raise_for_status()
        items = (idx.json().get("directory", {}) or {}).get("item", []) or []
    except Exception:
        return []

    xml_files = [it.get("name", "") for it in items if it.get("name", "").lower().endswith(".xml")]
    if not xml_files:
        return []

    # Filings typically contain `primary_doc.xml` (cover) and a separate XML
    # that holds the holdings (often a numeric name like `53405.xml`).
    # Prefer infotable/information by name; else any non-cover XML; else the
    # last XML as last resort.
    candidates = (
        [n for n in xml_files if "infotable" in n.lower() or "information" in n.lower()]
        or [n for n in xml_files if n.lower() != "primary_doc.xml"]
        or xml_files
    )

    for name in candidates:
        try:
            xml_resp = requests.get(
                f"{base}/{name}",
                headers={"User-Agent": SEC_UA, "Accept": "application/xml"},
                timeout=30,
            )
            xml_resp.raise_for_status()
            parsed = _parse_13f_infotable(xml_resp.text)
            if parsed:
                return parsed
        except Exception:
            continue
    return []


def _parse_13f_infotable(xml_text: str) -> List["ThirteenFHolding"]:
    """Parse SEC 13F infotable XML into ThirteenFHolding list.

    The schema is namespaced. Each <infoTable> has nameOfIssuer, cusip,
    value (USD), shrsOrPrnAmt/sshPrnamt (shares), putCall (optional).
    Historically `value` was in USD thousands; from 2023 it's full dollars.
    We detect by max value magnitude.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    # Strip namespaces for simple tag access.
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    holdings: List[ThirteenFHolding] = []
    for it in root.iter("infoTable"):
        name = (it.findtext("nameOfIssuer") or "").strip()
        cusip = (it.findtext("cusip") or "").strip()
        value_str = (it.findtext("value") or "0").strip().replace(",", "")
        sh_el = it.find("shrsOrPrnAmt")
        shares_str = (sh_el.findtext("sshPrnamt") if sh_el is not None else "0") or "0"
        shares_str = shares_str.strip().replace(",", "")
        put_call = it.findtext("putCall")
        try:
            value = float(value_str)
            shares = float(shares_str)
        except ValueError:
            continue
        holdings.append(ThirteenFHolding(
            cusip=cusip,
            issuer_name=name,
            ticker=None,
            shares=shares,
            value_usd=value,
            put_call=(put_call.strip() if put_call else None),
        ))

    # Heuristic: if every value is < 1e7, the filing is still in $-thousands.
    if holdings and max(h.value_usd for h in holdings) < 1e7:
        for h in holdings:
            h.value_usd *= 1000.0

    return holdings


# ---------------------------------------------------------------------------
# OpenFIGI: ticker -> CUSIP
# ---------------------------------------------------------------------------

_TICKER_CUSIP_FILE = "openfigi_ticker_cusip.json"


def ticker_to_cusip(ticker: str, cache_dir: str) -> Optional[str]:
    """Map a US ticker to its 9-char CUSIP via OpenFIGI. Cached forever.

    Returns None on failure. Ignores Indian tickers (we don't use CUSIPs
    for NSE/BSE matching).
    """
    if not ticker or requests is None:
        return None
    t = ticker.upper().strip()
    if "." in t and (t.endswith(".NS") or t.endswith(".BO")):
        return None  # India — handled via bulk-deal name match instead

    path = os.path.join(cache_dir, _TICKER_CUSIP_FILE)
    cache: Dict[str, Optional[str]] = _read_cache(path) or {}
    if t in cache:
        return cache[t]

    try:
        resp = requests.post(
            OPENFIGI_MAP,
            json=[{"idType": "TICKER", "idValue": t, "exchCode": "US"}],
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json() or []
        cusip: Optional[str] = None
        if data and isinstance(data, list) and data[0].get("data"):
            for row in data[0]["data"]:
                if row.get("compositeFIGI") or row.get("figi"):
                    # OpenFIGI doesn't always return CUSIP directly; if shareClassFIGI
                    # is present we can do a follow-up. For now, try id-mapping again
                    # by share class ticker.
                    pass
            # Some responses include 'cusip' on the row. Try direct extract.
            cusip = next(
                (row.get("cusip") for row in data[0]["data"] if row.get("cusip")),
                None,
            )
    except Exception:
        cusip = None

    cache[t] = cusip
    _write_cache(path, cache)
    return cusip



def _snapshot_from_dict(raw: Dict[str, Any]) -> ThirteenFSnapshot:
    return ThirteenFSnapshot(
        cik=raw.get("cik", ""),
        investor_name=raw.get("investor_name", ""),
        period_of_report=raw.get("period_of_report", ""),
        accession_number=raw.get("accession_number", ""),
        holdings=[ThirteenFHolding(**h) for h in raw.get("holdings", [])],
    )


# ---------------------------------------------------------------------------
# NSE bulk deals
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BulkDeal:
    date: str
    symbol: str
    client_name: str
    buy_sell: str            # "BUY" | "SELL"
    quantity: float
    avg_price_inr: float


def fetch_nse_bulk_deals(
    cache_dir: str,
    lookback_days: int = 30,
    max_age_days: int = 1,
) -> List[BulkDeal]:
    """Fetch recent NSE bulk deals. Returns [] on failure."""
    if requests is None:
        return []

    cache_key = f"nse_bulk_deals_{lookback_days}d"
    cache_file = _cache_path(cache_dir, cache_key)
    if _cache_fresh(cache_file, max_age_days):
        raw = _read_cache(cache_file)
        if raw:
            return [BulkDeal(**d) for d in raw]

    to_date = datetime.utcnow().date()
    from_date = to_date - timedelta(days=lookback_days)
    params = {
        "from": from_date.strftime("%d-%m-%Y"),
        "to": to_date.strftime("%d-%m-%Y"),
    }
    url = f"{NSE_BULK_DEALS}?{urlencode(params)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (LCF-Advisor)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/",
    }
    try:
        session = requests.Session()
        session.headers.update(headers)
        # NSE requires a homepage hit to seed cookies.
        session.get("https://www.nseindia.com/", timeout=15)
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("data", [])
    except Exception:
        cached = _read_cache(cache_file)
        return [BulkDeal(**d) for d in cached] if cached else []

    deals: List[BulkDeal] = []
    for r in rows:
        try:
            deals.append(BulkDeal(
                date=r.get("BD_DT_DATE") or r.get("date", ""),
                symbol=r.get("BD_SYMBOL") or r.get("symbol", ""),
                client_name=r.get("BD_CLIENT_NAME") or r.get("clientName", ""),
                buy_sell=(r.get("BD_BUY_SELL") or r.get("buySell", "")).upper(),
                quantity=float(r.get("BD_QTY_TRD") or r.get("quantity") or 0),
                avg_price_inr=float(r.get("BD_TP_WATP") or r.get("price") or 0),
            ))
        except (TypeError, ValueError):
            continue

    _write_cache(cache_file, [asdict(d) for d in deals])
    return deals


def filter_bulk_deals_by_investor(
    deals: Iterable[BulkDeal],
    aliases: Iterable[str],
) -> List[BulkDeal]:
    """Case-insensitive substring match on client_name."""
    needles = [a.lower() for a in aliases]
    out: List[BulkDeal] = []
    for d in deals:
        name_l = d.client_name.lower()
        if any(n in name_l for n in needles):
            out.append(d)
    return out


# ---------------------------------------------------------------------------
# AMFI mutual fund portfolio disclosure (placeholder)
# ---------------------------------------------------------------------------

def fetch_amfi_scheme_portfolio(
    scheme_code: str,
    cache_dir: str,
    max_age_days: int = 30,
) -> Optional[Dict[str, Any]]:
    """Fetch latest monthly portfolio disclosure for an AMFI scheme code.

    AMFI publishes monthly portfolio PDFs/Excel; full parsing is non-trivial.
    For now this is a stub that hits mfapi.in for NAV history and returns
    metadata. Real portfolio extraction can be wired in later via
    AMC websites (e.g. PPFAS, HDFC) which publish CSV/Excel.
    """
    if requests is None:
        return None

    cache_key = f"amfi_scheme_{scheme_code}"
    cache_file = _cache_path(cache_dir, cache_key)
    if _cache_fresh(cache_file, max_age_days):
        return _read_cache(cache_file)

    try:
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return _read_cache(cache_file)

    payload = {
        "scheme_code": scheme_code,
        "meta": data.get("meta", {}),
        "latest_nav": (data.get("data") or [{}])[0],
        "holdings": [],  # TODO: parse AMC monthly disclosure
    }
    _write_cache(cache_file, payload)
    return payload


# ---------------------------------------------------------------------------
# Watchlist loader
# ---------------------------------------------------------------------------

def load_watchlist(path: str) -> Dict[str, Any]:
    """Load investor watchlist YAML. Returns {} on failure."""
    try:
        import yaml  # local import to keep this module importable w/o pyyaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Investor-letter fetching (for LLM rationale extraction)
# ---------------------------------------------------------------------------

def load_letters_index(path: str) -> Dict[str, Any]:
    """Load investor-letter URL index YAML. Returns {} on failure.

    Expected shape:
        Warren Buffett (Berkshire Hathaway):
          letters:
            - https://...2024ltr.html
            - https://...2023ltr.html
    """
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _html_to_text(html: str) -> str:
    """Strip HTML to readable text. PDFs (binary) get returned empty."""
    if not html:
        return ""
    # PDFs would start with %PDF — we don't parse those here.
    if html.lstrip().startswith("%PDF"):
        return ""
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"&nbsp;|&amp;|&lt;|&gt;|&quot;|&#39;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_letter_text(
    url: str,
    cache_dir: str,
    max_age_days: int = 90,
) -> Optional[str]:
    """Fetch and cache the plaintext of an investor-letter URL.

    Returns None on failure or for non-HTML responses (e.g. PDFs).
    """
    if not url or requests is None:
        return None

    key = "letter_" + hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    path = _cache_path(cache_dir, key)
    if _cache_fresh(path, max_age_days):
        cached = _read_cache(path)
        return cached.get("text") if cached else None

    try:
        resp = requests.get(
            url,
            headers={"User-Agent": SEC_UA, "Accept": "text/html, text/plain, */*"},
            timeout=60,
        )
        resp.raise_for_status()
        ctype = (resp.headers.get("Content-Type") or "").lower()
        if "pdf" in ctype:
            text = ""
        else:
            text = _html_to_text(resp.text)
    except Exception:
        cached = _read_cache(path)
        return cached.get("text") if cached else None

    _write_cache(path, {
        "url": url,
        "text": text,
        "fetched_at": datetime.utcnow().isoformat(),
    })
    return text or None

