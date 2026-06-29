from __future__ import annotations

import calendar
import csv
import hashlib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import feedparser
import requests


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

# ✅ Verified working RSS endpoint for Corporate Actions [1](https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml)
CORP_ACTIONS_RSS_URL = "https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml"

# NSE CSV API includes SYMBOL + COMPANY NAME + PURPOSE + EX-DATE + RECORD DATE etc. [2](https://www.nseindia.com/api/corporates-corporateActions?index=equities&csv=true)
CORP_ACTIONS_CSV_URL = "https://www.nseindia.com/api/corporates-corporateActions?index=equities&csv=true"

RAW_XML_PATH = os.path.join("data", "ingest", "nse", "corporate_actions.xml")
SQLITE_PATH = os.path.join("data", "normalized", "nse", "corporate_actions.sqlite")
JSONL_PATH = os.path.join("data", "normalized", "nse", "corporate_actions.jsonl")


# ------------------------------------------------------------------
# Agent-ready normalized model
# ------------------------------------------------------------------

@dataclass
class CorporateActionNormalizedEvent:
    """
    Agent-ready corporate action.
    Everything important is extracted into explicit fields.
    Agents do NOT need to parse raw text.
    """
    event_id: str
    event_type: str              # "CORPORATE_ACTION"
    source: str                  # "NSE"

    symbol: Optional[str]
    company_name: str
    series: Optional[str]

    action_type: str             # DIVIDEND / SPLIT / BONUS / RIGHTS / BUYBACK / OTHER
    action_subtype: Optional[str]  # INTERIM / FINAL etc. (when applicable)

    # Numeric/structured details (optional depending on action_type)
    amount_inr: Optional[float]     # dividend amount or premium etc.
    ratio_from: Optional[int]       # e.g., bonus 3:1 => 3
    ratio_to: Optional[int]         # e.g., bonus 3:1 => 1
    split_from_fv: Optional[float]  # split from face value
    split_to_fv: Optional[float]    # split to face value
    face_value: Optional[float]     # FACE VALUE field

    # Dates (critical for trading logic)
    ex_date: Optional[str]          # YYYY-MM-DD
    record_date: Optional[str]      # YYYY-MM-DD
    book_closure_start: Optional[str]
    book_closure_end: Optional[str]

    published_raw: Optional[str]    # RSS “published” string (kept as-is)
    fetched_at: str                 # ISO-8601 UTC
    raw_text: str                   # original DESCRIPTION string for audit
    raw: dict                       # full feedparser entry


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_DATE_DD_MON_YYYY = re.compile(r"(\d{2})-([A-Za-z]{3})-(\d{4})")
_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
}

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def _event_id(*parts: str) -> str:
    base = "|".join(p or "" for p in parts)
    return hashlib.sha256(base.encode("utf-8")).hexdigest()

def _parse_dd_mon_yyyy(s: str) -> Optional[str]:
    """
    Convert '06-Mar-2026' to '2026-03-06'
    Returns None if not parseable.
    """
    m = _DATE_DD_MON_YYYY.search(s.strip())
    if not m:
        return None
    dd = int(m.group(1))
    mon = _MONTHS.get(m.group(2).upper())
    yyyy = int(m.group(3))
    if not mon:
        return None
    return f"{yyyy:04d}-{mon:02d}-{dd:02d}"

def _parse_pipe_kv(raw_text: str) -> Dict[str, str]:
    """
    Parse text like:
    SERIES:EQ |PURPOSE:INTERIM DIVIDEND - RS 2.70 PER SHARE |FACE VALUE:10 |RECORD DATE:06-Mar-2026 |...
    into a dict.
    """
    out: Dict[str, str] = {}
    for part in raw_text.split("|"):
        part = part.strip()
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip().upper()] = v.strip()
    return out

def _classify_purpose(purpose: str) -> Tuple[str, Optional[str]]:
    """
    Normalize action_type + subtype from PURPOSE field.
    """
    p = purpose.upper().strip()

    # Dividend variants
    if "DIVIDEND" in p:
        subtype = "INTERIM" if p.startswith("INTERIM DIVIDEND") else "FINAL" if "FINAL" in p else None
        return "DIVIDEND", subtype

    if p.startswith("FACE VALUE SPLIT") or "SPLIT (SUB-DIVISION)" in p:
        return "SPLIT", None

    if p.startswith("BONUS"):
        return "BONUS", None

    if p.startswith("RIGHTS"):
        return "RIGHTS", None

    if p.startswith("BUYBACK") or "BUY BACK" in p:
        return "BUYBACK", None

    return "OTHER", None

def _extract_amount_inr(purpose: str) -> Optional[float]:
    """
    Extract numeric amount from strings like:
    'INTERIM DIVIDEND - RS 2.70 PER SHARE'
    'INTERIM DIVIDEND - RE 1 PER SHARE'
    'RIGHTS 1:17 @ PREMIUM RS 502/-'
    Returns float or None.
    """
    p = purpose.upper()

    # RE 1  => treat as 1.0
    m_re = re.search(r"\bRE\s*([0-9]+(?:\.[0-9]+)?)\b", p)
    if m_re:
        return float(m_re.group(1))

    m_rs = re.search(r"\bRS\s*([0-9]+(?:\.[0-9]+)?)\b", p)
    if m_rs:
        return float(m_rs.group(1))

    return None

def _extract_ratio(purpose: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Extract ratios like 'BONUS 3:1' or 'RIGHTS 1:17 ...'
    """
    m = re.search(r"(\d+)\s*:\s*(\d+)", purpose)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _extract_split_fv(purpose: str) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract split from/to face values:
    '... FROM RS 10/- PER SHARE TO RS 2/- PER SHARE'
    """
    p = purpose.upper()
    m = re.search(r"FROM\s+RS\s*([0-9]+(?:\.[0-9]+)?)", p)
    n = re.search(r"TO\s+RS\s*([0-9]+(?:\.[0-9]+)?)", p)
    if not (m and n):
        return None, None
    return float(m.group(1)), float(n.group(1))


# ------------------------------------------------------------------
# Symbol map (company_name -> symbol) using NSE CSV endpoint [2](https://www.nseindia.com/api/corporates-corporateActions?index=equities&csv=true)
# ------------------------------------------------------------------

def fetch_symbol_map(timeout_sec: int = 30) -> Dict[str, str]:
    """
    Best-effort: build COMPANY NAME -> SYMBOL mapping from NSE CSV API.
    This is what makes the output agent-ready (symbol is critical).
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,*/*;q=0.8",
        "Referer": "https://www.nseindia.com/companies-listing/corporate-filings-actions",
    }
    resp = requests.get(CORP_ACTIONS_CSV_URL, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()

    text = resp.text
    # CSV begins with header line like: "SYMBOL","COMPANY NAME",...
    reader = csv.DictReader(text.splitlines())
    mapping: Dict[str, str] = {}
    for row in reader:
        company = (row.get("COMPANY NAME") or "").strip()
        symbol = (row.get("SYMBOL") or "").strip()
        if company and symbol:
            mapping[company.upper()] = symbol
    return mapping


# ------------------------------------------------------------------
# Normalize RSS -> agent-ready objects [1](https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml)
# ------------------------------------------------------------------

def parse_and_normalize(xml_text: str, symbol_map: Dict[str, str]) -> List[CorporateActionNormalizedEvent]:
    parsed = feedparser.parse(xml_text)
    fetched_at = _iso_now()

    out: List[CorporateActionNormalizedEvent] = []

    for entry in parsed.entries:
        title = (getattr(entry, "title", "") or "").strip()
        raw_text = (getattr(entry, "summary", "") or "").strip()  # DESCRIPTION-style payload [1](https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml)
        published_raw = getattr(entry, "published", None)

        # Title pattern: "<Company Name> - Ex-Date: 02-Mar-2026" [1](https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml)
        company_name = title.split(" - Ex-Date:", 1)[0].strip()
        ex_date = None
        if " - Ex-Date:" in title:
            ex_date = _parse_dd_mon_yyyy(title.split(" - Ex-Date:", 1)[1].strip())

        kv = _parse_pipe_kv(raw_text)

        series = kv.get("SERIES")
        purpose = kv.get("PURPOSE", "")
        face_value = float(kv["FACE VALUE"]) if kv.get("FACE VALUE", "").replace(".", "").isdigit() else None
        record_date = _parse_dd_mon_yyyy(kv.get("RECORD DATE", "")) if kv.get("RECORD DATE") else None
        bc_start = _parse_dd_mon_yyyy(kv.get("BOOK CLOSURE START DATE", "")) if kv.get("BOOK CLOSURE START DATE") else None
        bc_end = _parse_dd_mon_yyyy(kv.get("BOOK CLOSURE END DATE", "")) if kv.get("BOOK CLOSURE END DATE") else None

        action_type, action_subtype = _classify_purpose(purpose)
        amount_inr = _extract_amount_inr(purpose)
        ratio_from, ratio_to = _extract_ratio(purpose)
        split_from_fv, split_to_fv = _extract_split_fv(purpose) if action_type == "SPLIT" else (None, None)

        symbol = symbol_map.get(company_name.upper())

        # Deterministic ID from stable fields
        event_id = _event_id("NSE", "CORPORATE_ACTION", company_name, action_type, ex_date or "", record_date or "", purpose)

        raw = {k: entry.get(k) for k in entry.keys()}

        out.append(
            CorporateActionNormalizedEvent(
                event_id=event_id,
                event_type="CORPORATE_ACTION",
                source="NSE",
                symbol=symbol,
                company_name=company_name,
                series=series,
                action_type=action_type,
                action_subtype=action_subtype,
                amount_inr=amount_inr,
                ratio_from=ratio_from,
                ratio_to=ratio_to,
                split_from_fv=split_from_fv,
                split_to_fv=split_to_fv,
                face_value=face_value,
                ex_date=ex_date,
                record_date=record_date,
                book_closure_start=bc_start,
                book_closure_end=bc_end,
                published_raw=published_raw,
                fetched_at=fetched_at,
                raw_text=raw_text,
                raw=raw,
            )
        )

    return out


# ------------------------------------------------------------------
# SQLite (agent-ready schema)
# ------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corporate_actions (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,
    source TEXT,
    symbol TEXT,
    company_name TEXT,
    series TEXT,
    action_type TEXT,
    action_subtype TEXT,
    amount_inr REAL,
    ratio_from INTEGER,
    ratio_to INTEGER,
    split_from_fv REAL,
    split_to_fv REAL,
    face_value REAL,
    ex_date TEXT,
    record_date TEXT,
    book_closure_start TEXT,
    book_closure_end TEXT,
    published_raw TEXT,
    fetched_at TEXT,
    raw_text TEXT,
    raw_json TEXT
);
"""


_INSERT_SQL = """
INSERT OR IGNORE INTO corporate_actions (
    event_id, event_type, source, symbol, company_name, series,
    action_type, action_subtype, amount_inr, ratio_from, ratio_to,
    split_from_fv, split_to_fv, face_value, ex_date, record_date,
    book_closure_start, book_closure_end, published_raw, fetched_at,
    raw_text, raw_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def init_sqlite(path: str) -> sqlite3.Connection:
    _ensure_parent(path)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")

    cur = conn.cursor()

    # If table exists, check schema column count
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='corporate_actions'")
    if cur.fetchone():
        cur.execute("PRAGMA table_info(corporate_actions)")
        col_count = len(cur.fetchall())

        expected_cols = 22  # must match _SCHEMA_SQL above
        if col_count != expected_cols:
            print(f"⚠️ Schema mismatch: corporate_actions has {col_count} cols, expected {expected_cols}. Recreating table...")
            cur.execute("DROP TABLE IF EXISTS corporate_actions")
            conn.commit()

    # Create latest schema
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn

def write_sqlite(conn: sqlite3.Connection, events: List[CorporateActionNormalizedEvent]) -> int:
    before = conn.total_changes
    cur = conn.cursor()

    for ev in events:
        cur.execute(
            _INSERT_SQL,
            (
                ev.event_id,
                ev.event_type,
                ev.source,
                ev.symbol,
                ev.company_name,
                ev.series,
                ev.action_type,
                ev.action_subtype,
                ev.amount_inr,
                ev.ratio_from,
                ev.ratio_to,
                ev.split_from_fv,
                ev.split_to_fv,
                ev.face_value,
                ev.ex_date,
                ev.record_date,
                ev.book_closure_start,
                ev.book_closure_end,
                ev.published_raw,
                ev.fetched_at,
                ev.raw_text,
                json.dumps(ev.raw, ensure_ascii=False),
            ),
        )

    conn.commit()
    return max(0, conn.total_changes - before)

def append_jsonl(path: str, events: List[CorporateActionNormalizedEvent]) -> None:
    _ensure_parent(path)
    with open(path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

def ingest_once() -> Tuple[int, int]:
    print("✅ ingest_once() – corporate actions (normalized, agent-ready)")

    headers = {"User-Agent": "Mozilla/5.0"}
    xml = requests.get(CORP_ACTIONS_RSS_URL, headers=headers, timeout=30).text  # [1](https://nsearchives.nseindia.com/content/RSS/Corporate_action.xml)

    _ensure_parent(RAW_XML_PATH)
    with open(RAW_XML_PATH, "w", encoding="utf-8") as f:
        f.write(xml)

    # Build symbol map from NSE CSV endpoint [2](https://www.nseindia.com/api/corporates-corporateActions?index=equities&csv=true)
    symbol_map = fetch_symbol_map()

    events = parse_and_normalize(xml, symbol_map)

    conn = init_sqlite(SQLITE_PATH)
    inserted = write_sqlite(conn, events)
    append_jsonl(JSONL_PATH, events)

    return len(events), inserted
