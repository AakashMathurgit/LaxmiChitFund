"""Regime Detector Agent — detects market conditions from Nifty50/VIX.

Analyzes broad market data to determine:
- Market trend (bull/bear/sideways)
- Volatility state (low/moderate/high/extreme)
- Regime confidence score

This context helps other agents make better decisions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import MarketRegime, VolatilityState, RegimeSignal

# Debug flag
DEBUG = os.environ.get("LCF_DEBUG", "0").lower() in ("1", "true", "yes")


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection thresholds."""
    # Trend detection
    rsi_bull_threshold: float = 55.0      # RSI > this = bullish bias
    rsi_bear_threshold: float = 45.0      # RSI < this = bearish bias
    trend_sma_short: int = 50             # Short-term SMA period
    trend_sma_long: int = 200             # Long-term SMA period
    
    # Volatility detection (India VIX based)
    vix_low_threshold: float = 12.0       # VIX < this = low vol
    vix_moderate_threshold: float = 18.0  # VIX < this = moderate vol
    vix_high_threshold: float = 25.0      # VIX < this = high vol
    # VIX >= high_threshold = extreme
    
    # Confidence
    min_data_points: int = 50             # Minimum bars for reliable signal


class RegimeDetectorAgent(Agent):
    """Detects market regime from index data (Nifty50, India VIX).

    INPUT DATA REQUIRED:
    --------------------
    - `index_ohlc`: List[Dict] with keys {open, high, low, close} for Nifty50
    - `index_volumes`: Optional[List[int]] volume data
    - `vix_value`: Optional[float] current India VIX value
    - `vix_history`: Optional[List[float]] recent VIX values

    OUTPUT:
    -------
    - `regime`: RegimeSignal dataclass with:
        - market_regime: MarketRegime enum (BULL_TREND, BEAR_TREND, SIDEWAYS, HIGH_VOLATILITY)
        - volatility_state: VolatilityState enum (LOW, MODERATE, HIGH, EXTREME)
        - regime_confidence: float 0-1
    - `raw_metrics`: Dict with RSI, SMA values, VIX level
    """

    name = "regime_detector_agent"

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        super().__init__(name=self.name)
        self._config = config or RegimeDetectorConfig()

    # ------------------------------------------------------------------
    # Technical Indicators
    # ------------------------------------------------------------------

    def _compute_rsi(self, closes: List[float], period: int = 14) -> float:
        """Compute RSI from closing prices."""
        if len(closes) < period + 1:
            return 50.0  # neutral default
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        recent_deltas = deltas[-(period):]
        
        gains = [d for d in recent_deltas if d > 0]
        losses = [-d for d in recent_deltas if d < 0]
        
        avg_gain = sum(gains) / period if gains else 0
        avg_loss = sum(losses) / period if losses else 0
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _compute_sma(self, closes: List[float], period: int) -> Optional[float]:
        """Compute Simple Moving Average."""
        if len(closes) < period:
            return None
        return sum(closes[-period:]) / period

    def _compute_atr(self, ohlc: List[Dict], period: int = 14) -> float:
        """Compute Average True Range for volatility."""
        if len(ohlc) < period + 1:
            return 0.0
        
        trs = []
        for i in range(1, len(ohlc)):
            high = ohlc[i]["high"]
            low = ohlc[i]["low"]
            prev_close = ohlc[i-1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(tr)
        
        return sum(trs[-period:]) / period if trs else 0.0

    # ------------------------------------------------------------------
    # Regime Detection Logic
    # ------------------------------------------------------------------

    def _detect_trend(self, closes: List[float]) -> tuple[MarketRegime, float]:
        """Detect market trend from price action.
        
        Returns (regime, confidence).
        """
        cfg = self._config
        
        # Compute indicators
        rsi = self._compute_rsi(closes)
        sma_short = self._compute_sma(closes, cfg.trend_sma_short)
        sma_long = self._compute_sma(closes, cfg.trend_sma_long)
        
        # Default
        regime = MarketRegime.SIDEWAYS
        confidence = 0.5
        
        if sma_short is None or sma_long is None:
            return regime, confidence
        
        current_price = closes[-1]
        
        # Bull trend: price > both SMAs, short SMA > long SMA, RSI > threshold
        bull_signals = 0
        if current_price > sma_short:
            bull_signals += 1
        if current_price > sma_long:
            bull_signals += 1
        if sma_short > sma_long:
            bull_signals += 1
        if rsi > cfg.rsi_bull_threshold:
            bull_signals += 1
        
        # Bear trend: price < both SMAs, short SMA < long SMA, RSI < threshold
        bear_signals = 0
        if current_price < sma_short:
            bear_signals += 1
        if current_price < sma_long:
            bear_signals += 1
        if sma_short < sma_long:
            bear_signals += 1
        if rsi < cfg.rsi_bear_threshold:
            bear_signals += 1
        
        # Determine regime
        if bull_signals >= 3:
            regime = MarketRegime.BULL_TREND
            confidence = min(0.9, 0.5 + bull_signals * 0.1)
        elif bear_signals >= 3:
            regime = MarketRegime.BEAR_TREND
            confidence = min(0.9, 0.5 + bear_signals * 0.1)
        else:
            regime = MarketRegime.SIDEWAYS
            confidence = 0.6
        
        return regime, confidence

    def _detect_volatility(
        self, 
        vix_value: Optional[float], 
        ohlc: List[Dict]
    ) -> tuple[VolatilityState, float]:
        """Detect volatility state from VIX or ATR.
        
        Returns (volatility_state, normalized_volatility).
        """
        cfg = self._config
        
        # Prefer VIX if available
        if vix_value is not None:
            if vix_value < cfg.vix_low_threshold:
                return VolatilityState.LOW, vix_value / 30.0
            elif vix_value < cfg.vix_moderate_threshold:
                return VolatilityState.MODERATE, vix_value / 30.0
            elif vix_value < cfg.vix_high_threshold:
                return VolatilityState.HIGH, vix_value / 30.0
            else:
                return VolatilityState.EXTREME, min(1.0, vix_value / 40.0)
        
        # Fallback to ATR-based detection
        if ohlc:
            atr = self._compute_atr(ohlc)
            avg_price = sum(bar["close"] for bar in ohlc[-20:]) / min(20, len(ohlc))
            atr_pct = (atr / avg_price) * 100 if avg_price else 0
            
            if atr_pct < 1.0:
                return VolatilityState.LOW, atr_pct / 3.0
            elif atr_pct < 1.5:
                return VolatilityState.MODERATE, atr_pct / 3.0
            elif atr_pct < 2.5:
                return VolatilityState.HIGH, atr_pct / 3.0
            else:
                return VolatilityState.EXTREME, min(1.0, atr_pct / 4.0)
        
        return VolatilityState.MODERATE, 0.5

    def detect_regime(
        self,
        index_ohlc: List[Dict],
        vix_value: Optional[float] = None,
    ) -> tuple[RegimeSignal, Dict[str, Any]]:
        """Main regime detection method.
        
        Returns (RegimeSignal, raw_metrics_dict).
        """
        closes = [bar["close"] for bar in index_ohlc] if index_ohlc else []
        
        # Detect components
        trend_regime, trend_confidence = self._detect_trend(closes)
        vol_state, vol_normalized = self._detect_volatility(vix_value, index_ohlc)
        
        # Override to HIGH_VOLATILITY regime if extreme volatility
        final_regime = trend_regime
        if vol_state == VolatilityState.EXTREME:
            final_regime = MarketRegime.HIGH_VOLATILITY
        
        # Compute overall confidence
        data_quality = min(1.0, len(index_ohlc) / self._config.min_data_points) if index_ohlc else 0.3
        overall_confidence = trend_confidence * data_quality
        
        # Build signal
        regime_signal = RegimeSignal(
            market_regime=final_regime,
            volatility_state=vol_state,
            regime_confidence=round(overall_confidence, 4),
        )
        
        # Raw metrics for debugging
        rsi = self._compute_rsi(closes) if closes else 50.0
        sma_50 = self._compute_sma(closes, 50)
        sma_200 = self._compute_sma(closes, 200)
        
        raw_metrics = {
            "rsi_14": round(rsi, 2),
            "sma_50": round(sma_50, 2) if sma_50 else None,
            "sma_200": round(sma_200, 2) if sma_200 else None,
            "vix": vix_value,
            "volatility_normalized": round(vol_normalized, 4),
            "data_points": len(index_ohlc) if index_ohlc else 0,
        }
        
        return regime_signal, raw_metrics

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        """Execute regime detection.
        
        Expected input_data keys:
        - index_ohlc: List[Dict] - Nifty50 OHLC bars
        - vix_value: Optional[float] - Current India VIX
        """
        started = self._pre_run()
        
        try:
            data = ctx.input_data or {}
            
            # Get index data (can be passed directly or in input_data)
            index_ohlc = kwargs.get("index_ohlc") or data.get("index_ohlc", [])
            vix_value = kwargs.get("vix_value") or data.get("vix_value")
            
            regime_signal, raw_metrics = self.detect_regime(index_ohlc, vix_value)
            
            payload = {
                "regime": regime_signal.market_regime.value,
                "volatility_state": regime_signal.volatility_state.value,
                "regime_confidence": regime_signal.regime_confidence,
                "raw_signal": regime_signal,
                "raw_metrics": raw_metrics,
            }
            
            # Debug output
            if DEBUG:
                print(f"\n[DEBUG] RegimeDetectorAgent")
                print(f"  Market Regime: {regime_signal.market_regime.value}")
                print(f"  Volatility: {regime_signal.volatility_state.value}")
                print(f"  Confidence: {regime_signal.regime_confidence:.1%}")
                print(f"  RSI: {raw_metrics['rsi_14']}")
                print(f"  SMA50: {raw_metrics['sma_50']}")
                print(f"  SMA200: {raw_metrics['sma_200']}")
                print(f"  VIX: {raw_metrics['vix']}")
            
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[
                    regime_signal.market_regime.value,
                    regime_signal.volatility_state.value,
                    str(regime_signal.regime_confidence),
                ],
            )
            
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="REGIME_ERROR", message=str(exc))],
            )
