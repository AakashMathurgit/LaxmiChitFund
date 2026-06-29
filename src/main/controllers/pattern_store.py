"""Pattern Store ΓÇö vector similarity search over historical trading setups.

Pure NumPy implementation ΓÇö no external vector DB needed.
Stores vectors in a .npz file and metadata in a .jsonl file.
Uses cosine similarity for nearest-neighbor search.

For LCF's use case (~20-dimensional vectors, <10K records),
this is faster to initialize and has zero dependency issues.

Usage:
    store = PatternStore("./data/pattern_store")
    store.record(features, symbol="TCS", date="2024-11-15", decision="BUY")
    # ... after 5 days ...
    store.record_outcome("TCS", "2024-11-15", actual_return_5d=0.042)
    # At inference:
    similar = store.search_similar(features, top_k=10)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from ...utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

# Feature keys used for vector embedding (must be stable and ordered)
_VECTOR_KEYS = [
    "tech_score", "tech_rsi", "tech_macd", "tech_volatility",
    "tech_breakout", "tech_trend",
    "fund_score", "fund_valuation", "fund_growth", "fund_health",
    "sent_score", "sent_net_ratio", "sent_trend", "sent_confidence",
    "evt_score", "evt_earnings", "evt_risk", "evt_gap_up", "evt_gap_down",
    "regime_confidence",
]

VECTOR_DIM = len(_VECTOR_KEYS)


@dataclass
class PatternRecord:
    """A past feature bundle snapshot + what actually happened."""
    record_id: str
    timestamp: str
    symbol: str
    date: str
    features: Dict[str, float]
    decision_made: str                          # BUY / SELL / HOLD
    confidence: float = 0.0
    regime: str = "sideways"

    # Outcome fields (filled after 5 trading days)
    actual_return_5d: Optional[float] = None
    hit_stop_loss: bool = False
    hit_target: bool = False
    max_drawdown_5d: Optional[float] = None
    outcome_recorded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> PatternRecord:
        return PatternRecord(**{k: v for k, v in d.items() if k in PatternRecord.__dataclass_fields__})


@dataclass
class SimilarityResult:
    """Result of a similarity search ΓÇö aggregated stats from top-K matches."""
    similar_count: int = 0
    avg_return_5d: float = 0.0
    positive_rate: float = 0.0
    max_drawdown: float = 0.0
    records: List[PatternRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similar_count": self.similar_count,
            "avg_return_5d": round(self.avg_return_5d, 6),
            "positive_rate": round(self.positive_rate, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "top_records": [
                {
                    "symbol": r.symbol,
                    "date": r.date,
                    "decision": r.decision_made,
                    "return_5d": r.actual_return_5d,
                    "regime": r.regime,
                }
                for r in self.records[:5]
            ],
        }


def _features_to_vector(features: Dict[str, Any]) -> np.ndarray:
    """Convert flat feature dict to ordered float vector."""
    return np.array([float(features.get(k, 0.0)) for k in _VECTOR_KEYS], dtype=np.float32)


def _cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of vectors.

    Returns array of similarities in [-1, 1], shape (N,).
    """
    query_norm = np.linalg.norm(query)
    if query_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float32)

    row_norms = np.linalg.norm(matrix, axis=1)
    # Avoid division by zero
    row_norms = np.where(row_norms == 0, 1.0, row_norms)

    return (matrix @ query) / (row_norms * query_norm)


class PatternStore:
    """Local vector store for historical trading pattern similarity search.

    Backed by NumPy (.npz) + JSONL. No external dependencies beyond numpy.
    Vectors are stored in memory and persisted on every write.

    Storage files:
      <persist_dir>/vectors.npz   ΓÇö numpy array of feature vectors
      <persist_dir>/records.jsonl  ΓÇö one PatternRecord JSON per line
    """

    def __init__(self, persist_dir: str = "./data/pattern_store"):
        self._persist_dir = os.path.abspath(persist_dir)
        self._vectors_path = os.path.join(self._persist_dir, "vectors.npz")
        self._records_path = os.path.join(self._persist_dir, "records.jsonl")

        # In-memory state
        self._vectors: np.ndarray = np.empty((0, VECTOR_DIM), dtype=np.float32)
        self._records: List[PatternRecord] = []

        self._init_store()

    def _init_store(self) -> None:
        """Load existing data from disk."""
        os.makedirs(self._persist_dir, exist_ok=True)

        # Load vectors
        if os.path.exists(self._vectors_path):
            try:
                data = np.load(self._vectors_path)
                self._vectors = data["vectors"].astype(np.float32)
            except Exception as e:
                logger.warning(f"Failed to load vectors: {e}")
                self._vectors = np.empty((0, VECTOR_DIM), dtype=np.float32)

        # Load records
        if os.path.exists(self._records_path):
            try:
                with open(self._records_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._records.append(PatternRecord.from_dict(json.loads(line)))
            except Exception as e:
                logger.warning(f"Failed to load records: {e}")
                self._records = []

        # Sanity check: vectors and records must match
        if len(self._vectors) != len(self._records):
            logger.warning(
                f"Vector/record count mismatch ({len(self._vectors)} vs {len(self._records)}), "
                "rebuilding vectors from records"
            )
            self._rebuild_vectors()

        logger.info(f"PatternStore ready: {self.count} records in {self._persist_dir}")

    def _rebuild_vectors(self) -> None:
        """Rebuild vector matrix from records (recovery path)."""
        if not self._records:
            self._vectors = np.empty((0, VECTOR_DIM), dtype=np.float32)
            return
        self._vectors = np.array(
            [_features_to_vector(r.features) for r in self._records],
            dtype=np.float32,
        )

    def _save_vectors(self) -> None:
        """Persist vector matrix to disk."""
        np.savez_compressed(self._vectors_path, vectors=self._vectors)

    def _append_record(self, record: PatternRecord) -> None:
        """Append a single record to the JSONL file."""
        with open(self._records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), default=str) + "\n")

    def _rewrite_records(self) -> None:
        """Rewrite all records to disk (used after outcome updates)."""
        with open(self._records_path, "w", encoding="utf-8") as f:
            for r in self._records:
                f.write(json.dumps(r.to_dict(), default=str) + "\n")

    @property
    def available(self) -> bool:
        return True  # Always available ΓÇö no external deps

    @property
    def count(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Record a pattern (at decision time)
    # ------------------------------------------------------------------

    def record(
        self,
        features: Dict[str, Any],
        symbol: str,
        date: str,
        decision: str,
        confidence: float = 0.0,
        regime: str = "sideways",
    ) -> Optional[str]:
        """Store a feature bundle snapshot at decision time.

        Returns the record_id.
        """
        record_id = f"{symbol}_{date}_{int(time.time())}"
        vector = _features_to_vector(features)

        record = PatternRecord(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            date=date,
            features={k: float(features.get(k, 0.0)) for k in _VECTOR_KEYS},
            decision_made=decision,
            confidence=confidence,
            regime=regime,
        )

        # Append to in-memory state
        self._records.append(record)
        self._vectors = np.vstack([self._vectors, vector.reshape(1, -1)])

        # Persist
        self._append_record(record)
        self._save_vectors()

        logger.debug(f"Pattern recorded: {record_id}")
        return record_id

    # ------------------------------------------------------------------
    # Record outcome (after 5 trading days)
    # ------------------------------------------------------------------

    def record_outcome(
        self,
        symbol: str,
        date: str,
        actual_return_5d: float,
        hit_stop_loss: bool = False,
        hit_target: bool = False,
        max_drawdown_5d: float = 0.0,
    ) -> bool:
        """Update a stored pattern with its actual outcome.

        Finds the most recent matching record by symbol+date.
        Returns True if found and updated.
        """
        updated = False
        for record in reversed(self._records):
            if record.symbol == symbol and record.date == date and not record.outcome_recorded:
                record.actual_return_5d = actual_return_5d
                record.hit_stop_loss = hit_stop_loss
                record.hit_target = hit_target
                record.max_drawdown_5d = max_drawdown_5d
                record.outcome_recorded = True
                updated = True
                break

        if updated:
            self._rewrite_records()
            logger.info(f"Outcome recorded: {symbol} {date} -> return={actual_return_5d:+.2%}")
        else:
            logger.warning(f"No pattern found for {symbol} {date}")

        return updated

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def search_similar(
        self,
        features: Dict[str, Any],
        top_k: int = 10,
        only_with_outcomes: bool = True,
    ) -> SimilarityResult:
        """Find the most similar historical setups to the current feature bundle.

        Uses cosine similarity over the 20-dimensional feature vector.

        Args:
            features: Flat feature dict from AgentFeatureBundle.to_flat_features()
            top_k: Number of similar patterns to retrieve
            only_with_outcomes: If True, only return patterns that have recorded outcomes

        Returns:
            SimilarityResult with aggregated stats and individual records
        """
        if self.count == 0:
            return SimilarityResult()

        query = _features_to_vector(features)

        # Filter indices if only_with_outcomes
        if only_with_outcomes:
            valid_indices = [
                i for i, r in enumerate(self._records)
                if r.outcome_recorded and r.actual_return_5d is not None
            ]
            if not valid_indices:
                return SimilarityResult()
            search_vectors = self._vectors[valid_indices]
            search_records = [self._records[i] for i in valid_indices]
        else:
            search_vectors = self._vectors
            search_records = self._records

        # Cosine similarity
        similarities = _cosine_similarity(query, search_vectors)

        # Top-K indices (highest similarity first)
        k = min(top_k, len(search_records))
        top_indices = np.argsort(similarities)[-k:][::-1]

        matched_records = [search_records[i] for i in top_indices]

        # Filter to those with outcomes for stats
        with_outcomes = [r for r in matched_records if r.outcome_recorded and r.actual_return_5d is not None]

        if not with_outcomes:
            return SimilarityResult(similar_count=len(matched_records), records=matched_records)

        returns = [r.actual_return_5d for r in with_outcomes]
        drawdowns = [r.max_drawdown_5d or 0.0 for r in with_outcomes]

        return SimilarityResult(
            similar_count=len(with_outcomes),
            avg_return_5d=mean(returns),
            positive_rate=sum(1 for r in returns if r > 0) / len(returns),
            max_drawdown=min(drawdowns) if drawdowns else 0.0,
            records=with_outcomes,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_all_pending_outcomes(self) -> List[Dict[str, Any]]:
        """Get all patterns that don't have outcomes recorded yet."""
        return [
            {"record_id": r.record_id, "symbol": r.symbol, "date": r.date,
             "decision": r.decision_made}
            for r in self._records
            if not r.outcome_recorded
        ]
