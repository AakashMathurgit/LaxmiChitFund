"""Load the editable US intraday universe (sectors + peer relationships).

Usage:
    from us_intraday_tracker.universe import load_universe
    u = load_universe("us_intraday_tracker/universe_us.yaml")
    u.all_symbols()      # -> ["AAPL", "MSFT", ...]
    u.sector_of("NVDA")  # -> "Technology"
    u.peers_of("NVDA")   # -> {"competitors": [...], "suppliers": [...], ...}
    u.peer_symbols("NVDA")  # -> flat list of all peer tickers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import yaml


@dataclass
class Universe:
    sectors: Dict[str, List[str]] = field(default_factory=dict)
    peers: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)

    def all_symbols(self) -> List[str]:
        """All tickers across all sectors, de-duplicated, order-preserving."""
        seen: Dict[str, None] = {}
        for symbols in self.sectors.values():
            for s in symbols:
                seen.setdefault(s, None)
        return list(seen.keys())

    def sectors_list(self) -> List[str]:
        return list(self.sectors.keys())

    def sector_of(self, symbol: str) -> str:
        for sector, symbols in self.sectors.items():
            if symbol in symbols:
                return sector
        return ""

    def peers_of(self, symbol: str) -> Dict[str, List[str]]:
        """Return the relationship map for a symbol (competitors/suppliers/...)."""
        return self.peers.get(symbol, {})

    def peer_symbols(self, symbol: str) -> List[str]:
        """Flat, de-duplicated list of all peer tickers for a symbol."""
        seen: Dict[str, None] = {}
        for group in self.peers_of(symbol).values():
            for s in group or []:
                if s != symbol:
                    seen.setdefault(s, None)
        return list(seen.keys())


def load_universe(path: str) -> Universe:
    """Parse the universe YAML into a Universe object."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sectors = data.get("sectors", {}) or {}
    # Normalize: ensure every value is a list of stripped, non-empty strings.
    norm_sectors: Dict[str, List[str]] = {}
    for sector, symbols in sectors.items():
        norm_sectors[sector] = [str(s).strip() for s in (symbols or []) if str(s).strip()]

    peers = data.get("peers", {}) or {}
    return Universe(sectors=norm_sectors, peers=peers)
