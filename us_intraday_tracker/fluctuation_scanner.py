"""Fluctuation scanner — the cheap "scan all 100" stage of the intraday funnel.

Uses YahooFinanceProvider.download_bulk() (one batched yf.download call) to pull
intraday bars for the whole universe, then computes a fluctuation score per
symbol and ranks the top movers. Deliberately avoids per-ticker `.info` calls
(the main rate-limit risk) — book value and fundamentals are fetched later, for
the top movers only, by the orchestrator's per-symbol path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ScanConfig:
    """Tuning for the fluctuation scan."""
    intraday_period: str = "1d"
    intraday_interval: str = "2m"
    daily_period: str = "1mo"      # for prev_close + average volume
    # Composite score weights.
    w_change: float = 1.0          # weight on abs(% day change)
    w_range: float = 0.5           # weight on intraday high-low range %
    w_vol: float = 2.0             # weight on (capped) volume spike
    vol_spike_cap: float = 5.0
    min_fluctuation_pct: float = 1.5   # ignore movers below this abs % change


@dataclass
class QuoteSnapshot:
    """A cheap intraday snapshot for one symbol (no fundamentals)."""
    symbol: str
    sector: str = ""
    current: Optional[float] = None
    day_open: Optional[float] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    prev_close: Optional[float] = None
    pct_change: float = 0.0
    volume: int = 0
    avg_volume: float = 0.0
    book_value: Optional[float] = None   # filled later for movers only
    intraday_range_pct: float = 0.0
    vol_spike: float = 1.0
    fluctuation_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "sector": self.sector,
            "current": self.current,
            "day_open": self.day_open,
            "day_high": self.day_high,
            "day_low": self.day_low,
            "prev_close": self.prev_close,
            "pct_change": round(self.pct_change, 3),
            "volume": self.volume,
            "avg_volume": round(self.avg_volume, 1),
            "intraday_range_pct": round(self.intraday_range_pct, 3),
            "vol_spike": round(self.vol_spike, 2),
            "fluctuation_score": round(self.fluctuation_score, 3),
        }


class FluctuationScanner:
    """Scans a universe for the most-fluctuating stocks."""

    def __init__(self, provider: Any, config: Optional[ScanConfig] = None,
                 sector_of=None):
        """
        Args:
            provider: a YahooFinanceProvider (exposes download_bulk()).
            config: ScanConfig.
            sector_of: optional callable symbol -> sector label (e.g.
                Universe.sector_of) used to tag snapshots.
        """
        self._provider = provider
        self._config = config or ScanConfig()
        self._sector_of = sector_of or (lambda s: "")

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def scan(self, symbols: List[str]) -> List[QuoteSnapshot]:
        cfg = self._config

        intraday = self._provider.download_bulk(
            symbols, period=cfg.intraday_period, interval=cfg.intraday_interval,
            progress=False,
        )
        daily = self._provider.download_bulk(
            symbols, period=cfg.daily_period, interval="1d",
            progress=False,
        )

        snapshots: List[QuoteSnapshot] = []
        for symbol in symbols:
            try:
                snap = self._build_snapshot(symbol, intraday, daily)
                if snap is not None:
                    snapshots.append(snap)
            except Exception as e:
                logger.debug(f"[{symbol}] snapshot error: {e}")
        return snapshots

    def _sub_frame(self, data: Any, symbol: str):
        """Extract a single symbol's OHLCV frame from a download_bulk result."""
        if data is None:
            return None
        normalized = self._provider._normalize_symbol(symbol)
        cols = data.columns
        # MultiIndex (ticker, field) when multiple symbols were requested.
        if hasattr(cols, "levels") and cols.nlevels > 1:
            level0 = set(cols.get_level_values(0))
            key = normalized if normalized in level0 else (symbol if symbol in level0 else None)
            if key is None:
                return None
            sub = data[key]
        else:
            # Single-symbol download: flat columns already.
            sub = data
        return sub.dropna(how="all")

    def _build_snapshot(self, symbol: str, intraday: Any, daily: Any) -> Optional[QuoteSnapshot]:
        cfg = self._config
        intra = self._sub_frame(intraday, symbol)
        day = self._sub_frame(daily, symbol)
        if intra is None or len(intra) == 0:
            return None

        current = float(intra["Close"].iloc[-1])
        day_open = float(intra["Open"].iloc[0])
        day_high = float(intra["High"].max())
        day_low = float(intra["Low"].min())
        volume = int(intra["Volume"].sum())

        # prev_close & avg_volume from daily bars.
        prev_close = None
        avg_volume = 0.0
        if day is not None and len(day) >= 1:
            closes = day["Close"].dropna()
            # Second-to-last daily close is "yesterday".
            if len(closes) >= 2:
                prev_close = float(closes.iloc[-2])
            elif len(closes) >= 1:
                prev_close = float(closes.iloc[-1])
            vols = day["Volume"].dropna()
            if len(vols) > 0:
                avg_volume = float(vols.mean())

        if prev_close is None or prev_close == 0:
            prev_close = day_open or current

        pct_change = ((current - prev_close) / prev_close * 100) if prev_close else 0.0
        intraday_range_pct = ((day_high - day_low) / prev_close * 100) if prev_close else 0.0
        vol_spike = (volume / avg_volume) if avg_volume > 0 else 1.0

        score = (
            cfg.w_change * abs(pct_change)
            + cfg.w_range * intraday_range_pct
            + cfg.w_vol * min(vol_spike, cfg.vol_spike_cap)
        )

        return QuoteSnapshot(
            symbol=symbol,
            sector=self._sector_of(symbol),
            current=current,
            day_open=day_open,
            day_high=day_high,
            day_low=day_low,
            prev_close=prev_close,
            pct_change=pct_change,
            volume=volume,
            avg_volume=avg_volume,
            intraday_range_pct=intraday_range_pct,
            vol_spike=vol_spike,
            fluctuation_score=score,
        )

    # ------------------------------------------------------------------
    # Rank
    # ------------------------------------------------------------------

    def rank_movers(self, snapshots: List[QuoteSnapshot], top_n: int = 8) -> List[QuoteSnapshot]:
        """Return the top_n snapshots by fluctuation score, filtered by min move."""
        eligible = [
            s for s in snapshots
            if abs(s.pct_change) >= self._config.min_fluctuation_pct
        ]
        eligible.sort(key=lambda s: s.fluctuation_score, reverse=True)
        return eligible[:top_n]
