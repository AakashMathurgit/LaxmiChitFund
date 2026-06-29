"""
NSE Corporate Announcements RSS ingestion.

What this module does:
1) Fetch the Corporate Announcements RSS XML from NSE
2) Save a raw snapshot of the XML under data/ingest/nse/
3) Parse RSS items into a structured internal schema
4) Save structured data into:
   - SQLite (deduped by event_id)
   - JSONL (append-only event log)

This module is import-safe:
- It defines functions/classes only
- It does NOT execute ingestion on import
"""

from __future__ import annotations  # ✅ must be here (top of file)

import calendar
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import feedparser
import requests

# Debug print is OK AFTER the future import
print("✅ Ingestion module imported from:", __file__)

# -------------------------------------------------------------------------
# 0) Constants (RSS endpoint + storage paths)
# -------------------------------------------------------------------------

# NSE Corporate Announcements RSS endpoint. [1](https://microsoftapc.sharepoint.com/teams/UC-SMFCC/_layouts/15/Doc.aspx?sourcedoc=%7BC42CAF04-E24E-460B-81BB-CD383FB1CDCA%7D&file=Gdb_scripts_for_smf_core_analysis.pptx&action=edit&mobileredirect=true&DefaultItemOpen=1)
CORP_ANNOUNCEMENTS_RSS_URL = "https://nsearchives.nseindia.com/content/RSS/Online_announcements.xml"

# Store raw XML snapshot exactly as fetched (ingest layer)
RAW_XML_PATH = os.path.join("data", "ingest", "nse", "corporate_announcements.xml")

# Store normalized outputs (structured layer)
SQLITE_PATH = os.path.join("data", "normalized", "nse", "corporate_announcements.sqlite")
JSONL_PATH = os.path.join("data", "normalized", "nse", "corporate_announcements.jsonl")


# -------------------------------------------------------------------------
# 1) Structured internal model
# -------------------------------------------------------------------------

@dataclass
class CorporateAnnouncementEvent:
    """One normalized corporate announcement event."""
    event_id: str
    source: str
    category: str
    title: str
    link: str
    published_at: Optional[str]
    summary: str
    fetched_at: str
    raw: dict


# -------------------------------------------------------------------------
# 2) Helpers
# -------------------------------------------------------------------------

def _iso_now_utc() -> str:
    """Return current UTC time as ISO-8601."""
    return datetime.now(timezone.utc).isoformat()


def _ensure_parent_dir(path: str) -> None:
    """Ensure the directory for a file path exists."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _entry_published_iso(entry) -> Optional[str]:
    """Convert feedparser published/updated time structs to ISO-8601 UTC."""
    dt_struct = getattr(entry, "published_parsed", None) or getattr(entry, "updated_parsed", None)
    if not dt_struct:
        return None
    try:
        ts = calendar.timegm(dt_struct)  # UTC-safe conversion
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _stable_event_id(title: str, link: str, published_at: Optional[str]) -> str:
    """Stable dedupe key based on title+link+published_at."""
    base = f"{title}|{link}|{published_at or ''}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------------
# 3) Fetch RSS XML
# -------------------------------------------------------------------------

def fetch_rss_xml(url: str = CORP_ANNOUNCEMENTS_RSS_URL, timeout_sec: int = 25) -> str:
    """Fetch RSS XML from NSE."""
    headers = {
        "User-Agent": "Mozilla/5.0 (LCF-LocalRSSIngest/1.0)",
        "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8",
    }
    resp = requests.get(url, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.text


def save_raw_xml(xml_text: str, raw_path: str = RAW_XML_PATH) -> None:
    """Save raw XML snapshot (ingest layer)."""
    _ensure_parent_dir(raw_path)
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(xml_text)


# -------------------------------------------------------------------------
# 4) Parse + normalize
# -------------------------------------------------------------------------

def parse_and_normalize(xml_text: str) -> List[CorporateAnnouncementEvent]:
    """Parse RSS XML and normalize into structured events."""
    parsed = feedparser.parse(xml_text)
    fetched_at = _iso_now_utc()

    events: List[CorporateAnnouncementEvent] = []
    for entry in parsed.entries:
        title = (getattr(entry, "title", "") or "").strip()
        link = (getattr(entry, "link", "") or "").strip()
        summary = (
            (getattr(entry, "summary", "") or "").strip()
            or (getattr(entry, "description", "") or "").strip()
        )

        published_at = _entry_published_iso(entry)
        event_id = _stable_event_id(title, link, published_at)

        raw = {}
        try:
            raw = {k: entry.get(k) for k in entry.keys()}
        except Exception:
            raw = {"_raw": str(entry)}

        events.append(
            CorporateAnnouncementEvent(
                event_id=event_id,
                source="NSE",
                category="CORPORATE_ANNOUNCEMENTS",
                title=title,
                link=link,
                published_at=published_at,
                summary=summary,
                fetched_at=fetched_at,
                raw=raw,
            )
        )
    return events


# -------------------------------------------------------------------------
# 5) Storage: SQLite + JSONL
# -------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS corporate_announcements (
    event_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    title TEXT,
    link TEXT,
    published_at TEXT,
    summary TEXT,
    fetched_at TEXT NOT NULL,
    raw_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_ca_published_at ON corporate_announcements(published_at);
"""

_INSERT_SQL = """
INSERT OR IGNORE INTO corporate_announcements (
    event_id, source, category, title, link, published_at, summary, fetched_at, raw_json
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def init_sqlite(db_path: str = SQLITE_PATH) -> sqlite3.Connection:
    """Create/open SQLite DB and ensure schema exists."""
    _ensure_parent_dir(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    return conn


def write_events_sqlite(conn: sqlite3.Connection, events: List[CorporateAnnouncementEvent]) -> int:
    """Write events into SQLite (deduped)."""
    before = conn.total_changes
    cur = conn.cursor()

    for ev in events:
        cur.execute(
            _INSERT_SQL,
            (
                ev.event_id,
                ev.source,
                ev.category,
                ev.title,
                ev.link,
                ev.published_at,
                ev.summary,
                ev.fetched_at,
                json.dumps(ev.raw, ensure_ascii=False),
            ),
        )

    conn.commit()
    after = conn.total_changes
    return max(0, after - before)


def append_events_jsonl(jsonl_path: str, events: List[CorporateAnnouncementEvent]) -> None:
    """Append events to JSONL file (one JSON object per line)."""
    _ensure_parent_dir(jsonl_path)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(asdict(ev), ensure_ascii=False) + "\n")


# -------------------------------------------------------------------------
# 6) One-call entry used by the script
# -------------------------------------------------------------------------

def ingest_once(
    rss_url: str = CORP_ANNOUNCEMENTS_RSS_URL,
    raw_xml_path: str = RAW_XML_PATH,
    sqlite_path: str = SQLITE_PATH,
    jsonl_path: str = JSONL_PATH,
) -> Tuple[int, int]:
    """End-to-end ingestion run."""
    print("✅ ingest_once() called")

    xml_text = fetch_rss_xml(rss_url)
    save_raw_xml(xml_text, raw_xml_path)

    events = parse_and_normalize(xml_text)

    conn = init_sqlite(sqlite_path)
    inserted = write_events_sqlite(conn, events)
    append_events_jsonl(jsonl_path, events)

    return (len(events), inserted)