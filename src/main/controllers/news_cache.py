"""Shared News Cache for LCF.

Provides a file-backed cache for RSS news items so that:
1. The stock tracker (discovery) saves all fetched news + matched tickers
2. Data providers (processing) read from cache instead of re-fetching feeds

Supports both Indian and US markets via separate cache files.

Cache format: JSONL (one JSON object per news item), with TTL-based expiration.

Usage:
    # Writer side (stock tracker)
    cache = NewsCache("data/news_cache_ind.jsonl", ttl_hours=72)
    cache.write_items([
        {"symbol": "TCS", "headline": "TCS wins $2B deal", ...},
    ])

    # Reader side (data provider / agent)
    cache = NewsCache("data/news_cache_ind.jsonl", ttl_hours=72)
    items = cache.read_items(symbol="TCS")
    items = cache.read_items()  # all items
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set


@dataclass
class CachedNewsItem:
    """A single cached news item."""
    symbol: str
    headline: str
    description: str = ""
    source: str = ""
    url: str = ""
    pub_date: str = ""
    fetched_at: str = ""           # ISO timestamp when item was cached
    significant: bool = True        # LLM/keyword significance classification
    significance_reason: str = ""
    event_type: str = ""            # EARNINGS, MERGER_ACQUISITION, etc.
    matched_symbols: List[str] = field(default_factory=list)  # All tickers matched
    market: str = "IND"             # IND or US
    feed_url: str = ""              # Which feed this came from
    impact_score: float = 0.0       # 0-1 impact score
    # NLP enrichment fields (populated by NLPProcessor)
    sentiment_label: str = ""       # "positive", "negative", "neutral"
    sentiment_score: float = 0.0    # 0.0-1.0 confidence score
    entities: List[str] = field(default_factory=list)  # Extracted entity names

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CachedNewsItem":
        # Handle any missing fields gracefully
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in known_fields}
        return cls(**filtered)


class NewsCache:
    """Thread-safe, file-backed news cache with TTL expiration.

    Each market (IND, US) should use a separate cache file.
    """

    def __init__(
        self,
        cache_path: str,
        ttl_hours: int = 72,
        max_items: int = 10000,
    ):
        """
        Args:
            cache_path: Path to the JSONL cache file.
            ttl_hours: Items older than this are considered expired.
            max_items: Maximum items to retain (oldest trimmed on compaction).
        """
        self._path = cache_path
        self._ttl = timedelta(hours=ttl_hours)
        self._max_items = max_items
        self._lock = threading.Lock()

        # Ensure directory exists
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def write_items(self, items: List[CachedNewsItem]) -> int:
        """Append new items to the cache. Returns count written."""
        if not items:
            return 0

        now = datetime.now().isoformat()
        written = 0
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                for item in items:
                    if not item.fetched_at:
                        item.fetched_at = now
                    f.write(json.dumps(item.to_dict(), default=str) + "\n")
                    written += 1
        return written

    def write_raw(self, raw_items: List[Dict[str, Any]]) -> int:
        """Write raw dicts (auto-wrapped as CachedNewsItem)."""
        items = []
        for d in raw_items:
            try:
                items.append(CachedNewsItem.from_dict(d))
            except Exception:
                continue
        return self.write_items(items)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def read_items(
        self,
        symbol: Optional[str] = None,
        symbols: Optional[Set[str]] = None,
        since_hours: Optional[int] = None,
        significant_only: bool = False,
    ) -> List[CachedNewsItem]:
        """Read cached items, optionally filtered.

        Args:
            symbol: Filter to a single symbol.
            symbols: Filter to a set of symbols.
            since_hours: Override TTL — only return items from last N hours.
            significant_only: Only return items marked as significant.

        Returns:
            List of CachedNewsItem within TTL window.
        """
        if not os.path.exists(self._path):
            return []

        cutoff = datetime.now() - (timedelta(hours=since_hours) if since_hours else self._ttl)
        target_symbols = None
        if symbol:
            target_symbols = {symbol.upper()}
        elif symbols:
            target_symbols = {s.upper() for s in symbols}

        results: List[CachedNewsItem] = []
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        item = CachedNewsItem.from_dict(d)
                    except Exception:
                        continue

                    # TTL filter
                    try:
                        fetched = datetime.fromisoformat(item.fetched_at)
                        if fetched < cutoff:
                            continue
                    except Exception:
                        pass  # Keep items with unparseable dates

                    # Significance filter
                    if significant_only and not item.significant:
                        continue

                    # Symbol filter
                    if target_symbols:
                        item_syms = {item.symbol.upper()} | {s.upper() for s in item.matched_symbols}
                        if not item_syms & target_symbols:
                            continue

                    results.append(item)

        return results

    def read_symbols(self, since_hours: Optional[int] = None) -> Set[str]:
        """Return all unique symbols in cache within TTL window."""
        items = self.read_items(since_hours=since_hours)
        symbols: Set[str] = set()
        for item in items:
            symbols.add(item.symbol.upper())
            for s in item.matched_symbols:
                symbols.add(s.upper())
        return symbols

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def compact(self) -> int:
        """Remove expired items and trim to max_items. Returns items kept."""
        if not os.path.exists(self._path):
            return 0

        cutoff = datetime.now() - self._ttl
        kept: List[str] = []

        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        fetched = datetime.fromisoformat(d.get("fetched_at", ""))
                        if fetched >= cutoff:
                            kept.append(line)
                    except Exception:
                        kept.append(line)  # Keep unparseable items

            # Trim to max
            if len(kept) > self._max_items:
                kept = kept[-self._max_items:]

            with open(self._path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")

        return len(kept)

    def clear(self) -> None:
        """Delete all cached items."""
        with self._lock:
            if os.path.exists(self._path):
                os.remove(self._path)

    @property
    def count(self) -> int:
        """Total items in cache (including expired)."""
        if not os.path.exists(self._path):
            return 0
        with self._lock:
            with open(self._path, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        items = self.read_items()
        symbols = set()
        sources = set()
        for item in items:
            symbols.add(item.symbol)
            sources.add(item.source)
        return {
            "cache_path": self._path,
            "total_on_disk": self.count,
            "active_items": len(items),
            "unique_symbols": len(symbols),
            "sources": sorted(sources),
            "ttl_hours": self._ttl.total_seconds() / 3600,
        }
