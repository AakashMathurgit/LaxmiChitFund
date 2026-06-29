"""Trade Memory ΓÇö append-only JSONL log of past trading decisions + outcomes.

Provides learning-from-experience by tracking what the system decided,
why, and what actually happened. Used by debate agents to cite past
trades and by RiskManager to avoid repeating mistakes.

Storage: simple JSONL file (one JSON object per line).
No external dependencies.

Usage:
    memory = TradeMemory("./data/trade_memory.jsonl")
    memory.record_decision(item)
    memory.record_outcome("TCS", "2024-11-15", exit_price=4050, pnl_pct=0.033)
    mistakes = memory.get_mistakes(regime="bear_trend")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class TradeMemoryItem:
    """A single trade decision record with optional outcome."""

    # Decision fields (recorded at trade time)
    timestamp: str
    symbol: str
    date: str
    decision: str                               # BUY / SELL / HOLD
    confidence: float
    entry_price: float
    regime: str = "sideways"
    key_reasons: List[str] = field(default_factory=list)
    debate_agreement: Optional[bool] = None     # did rule + debate agree?

    # Outcome fields (recorded after exit)
    exit_price: Optional[float] = None
    exit_date: Optional[str] = None
    pnl_pct: Optional[float] = None
    hit_stop_loss: bool = False
    hit_target: bool = False
    days_held: Optional[int] = None
    outcome_notes: str = ""
    outcome_recorded: bool = False

    # Classification
    is_mistake: bool = False                    # True if stop loss hit

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> TradeMemoryItem:
        return TradeMemoryItem(**{k: v for k, v in d.items() if k in TradeMemoryItem.__dataclass_fields__})

    def summary(self) -> str:
        """One-line summary for display in debate context."""
        result = f"{self.decision} {self.symbol} @ Γé╣{self.entry_price:.0f}"
        if self.outcome_recorded and self.pnl_pct is not None:
            result += f" ΓåÆ {self.pnl_pct:+.1%}"
            if self.hit_stop_loss:
                result += " (STOP LOSS)"
            elif self.hit_target:
                result += " (TARGET HIT)"
        result += f" [{self.regime}]"
        return result


class TradeMemory:
    """Append-only JSONL store of past trading decisions and outcomes.

    Thread-safe for single-writer usage (append mode).
    """

    def __init__(self, file_path: str = "./data/trade_memory.jsonl"):
        self._file_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(self._file_path), exist_ok=True)

        # Ensure file exists
        if not os.path.exists(self._file_path):
            with open(self._file_path, "w", encoding="utf-8"):
                pass

        self._cache: Optional[List[TradeMemoryItem]] = None
        logger.info(f"TradeMemory initialized: {self._file_path} ({self.count} records)")

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def _load_all(self) -> List[TradeMemoryItem]:
        """Load all records from JSONL file."""
        if self._cache is not None:
            return self._cache

        items: List[TradeMemoryItem] = []
        try:
            with open(self._file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        items.append(TradeMemoryItem.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except FileNotFoundError:
            pass
        self._cache = items
        return items

    def _append(self, item: TradeMemoryItem) -> None:
        """Append a single record to the JSONL file."""
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item.to_dict(), default=str) + "\n")
        # Invalidate cache
        self._cache = None

    def _rewrite_all(self, items: List[TradeMemoryItem]) -> None:
        """Rewrite the entire file (used for outcome updates)."""
        with open(self._file_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item.to_dict(), default=str) + "\n")
        self._cache = None

    @property
    def count(self) -> int:
        return len(self._load_all())

    # ------------------------------------------------------------------
    # Record decision (at trade time)
    # ------------------------------------------------------------------

    def record_decision(
        self,
        symbol: str,
        date: str,
        decision: str,
        confidence: float,
        entry_price: float,
        regime: str = "sideways",
        key_reasons: Optional[List[str]] = None,
        debate_agreement: Optional[bool] = None,
    ) -> TradeMemoryItem:
        """Record a trading decision. Call at trade execution time."""
        item = TradeMemoryItem(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            date=date,
            decision=decision,
            confidence=confidence,
            entry_price=entry_price,
            regime=regime,
            key_reasons=key_reasons or [],
            debate_agreement=debate_agreement,
        )
        self._append(item)
        logger.info(f"Decision recorded: {item.summary()}")
        return item

    # ------------------------------------------------------------------
    # Record outcome (after exit)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        symbol: str,
        date: str,
        exit_price: float,
        exit_date: Optional[str] = None,
        hit_stop_loss: bool = False,
        hit_target: bool = False,
        days_held: Optional[int] = None,
        outcome_notes: str = "",
    ) -> bool:
        """Update a stored decision with its actual outcome.

        Finds the most recent unresolved trade for symbol+date.
        Returns True if found and updated.
        """
        items = self._load_all()
        updated = False

        for item in reversed(items):
            if (
                item.symbol == symbol
                and item.date == date
                and not item.outcome_recorded
            ):
                item.exit_price = exit_price
                item.exit_date = exit_date or datetime.now().strftime("%Y-%m-%d")
                item.pnl_pct = (
                    (exit_price - item.entry_price) / item.entry_price
                    if item.entry_price > 0 else 0.0
                )
                item.hit_stop_loss = hit_stop_loss
                item.hit_target = hit_target
                item.days_held = days_held
                item.outcome_notes = outcome_notes
                item.outcome_recorded = True
                item.is_mistake = hit_stop_loss
                updated = True
                break

        if updated:
            self._rewrite_all(items)
            logger.info(f"Outcome recorded: {symbol} {date} ΓåÆ exit={exit_price}")
        else:
            logger.warning(f"No unresolved trade found for {symbol} {date}")

        return updated

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_history_for_symbol(
        self,
        symbol: str,
        limit: int = 10,
        only_with_outcomes: bool = False,
    ) -> List[TradeMemoryItem]:
        """Get recent trade history for a specific symbol."""
        items = self._load_all()
        filtered = [
            i for i in items
            if i.symbol == symbol
            and (not only_with_outcomes or i.outcome_recorded)
        ]
        return filtered[-limit:]

    def get_mistakes(
        self,
        regime: Optional[str] = None,
        limit: int = 10,
    ) -> List[TradeMemoryItem]:
        """Get past trades that hit stop loss (mistakes to avoid)."""
        items = self._load_all()
        mistakes = [
            i for i in items
            if i.is_mistake
            and (regime is None or i.regime == regime)
        ]
        return mistakes[-limit:]

    def get_successes(
        self,
        regime: Optional[str] = None,
        limit: int = 10,
    ) -> List[TradeMemoryItem]:
        """Get past trades that hit target (patterns to repeat)."""
        items = self._load_all()
        successes = [
            i for i in items
            if i.hit_target
            and (regime is None or i.regime == regime)
        ]
        return successes[-limit:]

    def get_recent_decisions(self, limit: int = 20) -> List[TradeMemoryItem]:
        """Get the most recent decisions (with or without outcomes)."""
        return self._load_all()[-limit:]

    def get_pending_outcomes(self) -> List[TradeMemoryItem]:
        """Get all decisions that still need outcome recording."""
        return [i for i in self._load_all() if not i.outcome_recorded]

    def get_stats(self) -> Dict[str, Any]:
        """Get aggregate statistics from trade memory."""
        items = self._load_all()
        with_outcomes = [i for i in items if i.outcome_recorded]

        if not with_outcomes:
            return {"total_trades": len(items), "resolved": 0}

        returns = [i.pnl_pct for i in with_outcomes if i.pnl_pct is not None]
        wins = [r for r in returns if r > 0]

        return {
            "total_trades": len(items),
            "resolved": len(with_outcomes),
            "pending": len(items) - len(with_outcomes),
            "win_rate": len(wins) / len(returns) if returns else 0.0,
            "avg_return": sum(returns) / len(returns) if returns else 0.0,
            "best_trade": max(returns) if returns else 0.0,
            "worst_trade": min(returns) if returns else 0.0,
            "total_mistakes": sum(1 for i in with_outcomes if i.is_mistake),
        }
