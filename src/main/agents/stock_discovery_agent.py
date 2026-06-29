"""Stock Discovery Agent — selects which stocks to analyze daily.

Uses multiple screening criteria to find actionable stocks from a universe:
- Volume spikes (unusual activity)
- Price breakouts (near 52-week high/low)
- News-driven movers (high news count)
- Sector momentum (Nifty 50 components)

Returns a ranked list of symbols to feed into the analysis pipeline.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


@dataclass
class StockCandidate:
    """A stock flagged for analysis with discovery reason."""
    symbol: str
    score: float             # 0-1 priority score
    reasons: List[str] = field(default_factory=list)


@dataclass
class DiscoveryConfig:
    """Configuration for stock discovery."""
    # Default universe: Nifty 50 components
    default_universe: List[str] = field(default_factory=lambda: [
        "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",

        
        "HINDUNILVR", "ITC", "SBIN", "BHARTIARTL", "KOTAKBANK",
        "LT", "AXISBANK", "ASIANPAINT", "MARUTI", "TITAN",
        "SUNPHARMA", "BAJFINANCE", "WIPRO", "HCLTECH", "ULTRACEMCO",
        "NESTLEIND", "TATAMOTORS", "POWERGRID", "NTPC", "TECHM",
        "INDUSINDBK", "ONGC", "JSWSTEEL", "TATASTEEL", "ADANIPORTS",
    ])
    max_candidates: int = 10
    volume_spike_threshold: float = 1.5     # 50% above 20-day avg
    price_near_high_pct: float = 0.95       # Within 5% of 52-week high
    min_market_cap: float = 10_000_000_000  # ₹1000 Cr minimum


class StockDiscoveryAgent:
    """Selects top stock candidates for daily analysis.

    Not a full Agent (doesn't inherit from Agent base) since it runs
    before the agent pipeline and just returns a symbol list.
    """

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self._config = config or DiscoveryConfig()

    def discover(
        self,
        universe: Optional[List[str]] = None,
        market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[StockCandidate]:
        """Screen the universe and return ranked candidates.

        Parameters
        ----------
        universe : list of str, optional
            Stock symbols to screen. Uses Nifty 50 if not provided.
        market_data : dict, optional
            Pre-fetched data: {symbol: {last_close, volume, avg_volume, ...}}.
            If not provided, returns all universe stocks with equal score.

        Returns
        -------
        list of StockCandidate, sorted by score descending
        """
        symbols = universe or self._config.default_universe

        if not market_data:
            # No data — return all with equal priority
            return [
                StockCandidate(symbol=s, score=0.5, reasons=["default_universe"])
                for s in symbols[:self._config.max_candidates]
            ]

        candidates = []
        for symbol in symbols:
            data = market_data.get(symbol, {})
            if not data:
                continue

            score = 0.0
            reasons = []

            # Volume spike detection
            volume = data.get("volume", 0)
            avg_volume = data.get("avg_volume", 0)
            if avg_volume > 0 and volume > avg_volume * self._config.volume_spike_threshold:
                spike_ratio = volume / avg_volume
                score += min(0.3, spike_ratio * 0.1)
                reasons.append(f"Volume spike ({spike_ratio:.1f}x avg)")

            # Near 52-week high
            last_close = data.get("last_close", 0)
            week52_high = data.get("52_week_high", 0)
            if week52_high > 0 and last_close >= week52_high * self._config.price_near_high_pct:
                score += 0.25
                reasons.append("Near 52-week high")

            # Near 52-week low (potential reversal)
            week52_low = data.get("52_week_low", 0)
            if week52_low > 0 and last_close <= week52_low * 1.05:
                score += 0.15
                reasons.append("Near 52-week low (reversal candidate)")

            # Price change > 2% in either direction
            price_change = data.get("price_change_pct", 0)
            if abs(price_change) > 0.02:
                score += 0.2
                direction = "up" if price_change > 0 else "down"
                reasons.append(f"Big move {direction} ({price_change:.1%})")

            # News-driven
            news_count = data.get("news_count", 0)
            if news_count > 3:
                score += 0.1
                reasons.append(f"{news_count} news articles")

            if reasons:
                candidates.append(StockCandidate(
                    symbol=symbol,
                    score=round(min(score, 1.0), 4),
                    reasons=reasons,
                ))

        # Sort by score descending
        candidates.sort(key=lambda c: c.score, reverse=True)

        result = candidates[:self._config.max_candidates]

        if DEBUG:
            print(f"\n[DEBUG] StockDiscoveryAgent | {len(result)} candidates from {len(symbols)} universe")
            for c in result:
                print(f"  {c.symbol}: {c.score:.2f} — {', '.join(c.reasons)}")

        return result

    def get_symbols(
        self,
        universe: Optional[List[str]] = None,
        market_data: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[str]:
        """Convenience: return just the symbol list."""
        return [c.symbol for c in self.discover(universe, market_data)]
