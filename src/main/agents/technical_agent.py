"""Technical Agent — computes indicators from raw OHLCV data.

Reads ohlc_daily (list of {open, high, low, close} dicts) and volume_daily
directly from StockDataContext. No pre-computed indicators required.

Computes: RSI-14, MACD (12/26), EMA trend (20/50), ATR volatility,
          breakout detection, 20-day support/resistance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .interfaces.agent import Agent, AgentContext, AgentResult, AgentError
from .interfaces.signals import TechnicalSignal

if TYPE_CHECKING:
    from ..controllers.data_context import StockDataContext


class TechnicalAgent(Agent):
    """Computes structured technical signals for swing-trading (3–10 day horizon)."""

    name = "technical_agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=self.name)
        self._config = config or {}

    # ------------------------------------------------------------------
    # Low-level math helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(prices: List[float], period: int) -> float:
        """Exponential moving average over *period* bars."""
        if not prices:
            return 0.0
        if len(prices) < period:
            return sum(prices) / len(prices)
        k = 2.0 / (period + 1)
        ema = sum(prices[:period]) / period
        for p in prices[period:]:
            ema = p * k + ema * (1.0 - k)
        return ema

    @staticmethod
    def _rsi(closes: List[float], period: int = 14) -> float:
        """Wilder RSI. Returns 50.0 if not enough data."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        avg_gain = sum(max(d, 0.0) for d in recent) / period
        avg_loss = sum(max(-d, 0.0) for d in recent) / period
        if avg_loss == 0.0:
            return 100.0
        return round(100.0 - 100.0 / (1.0 + avg_gain / avg_loss), 2)

    @classmethod
    def _macd_signal(cls, closes: List[float]) -> str:
        """MACD line direction: 'buy' if EMA12 > EMA26, 'sell' if below, 'neutral' otherwise."""
        if len(closes) < 26:
            return "neutral"
        diff = cls._ema(closes, 12) - cls._ema(closes, 26)
        if diff > 0:
            return "buy"
        if diff < 0:
            return "sell"
        return "neutral"

    @classmethod
    def _trend_direction(cls, closes: List[float]) -> str:
        """EMA20 vs EMA50 crossover. Needs ≥ 50 bars."""
        if len(closes) < 50:
            return "neutral"
        ema20 = cls._ema(closes, 20)
        ema50 = cls._ema(closes, 50)
        if ema20 > ema50 * 1.001:
            return "bullish"
        if ema20 < ema50 * 0.999:
            return "bearish"
        return "neutral"

    @staticmethod
    def _atr(ohlc: List[Dict[str, float]], period: int = 14) -> float:
        """Average True Range over *period* bars."""
        if len(ohlc) < 2:
            return 0.0
        trs = []
        for i in range(1, len(ohlc)):
            h = ohlc[i].get("high", 0.0)
            l = ohlc[i].get("low", 0.0)
            pc = ohlc[i - 1].get("close", 0.0)
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        recent = trs[-period:]
        return sum(recent) / len(recent) if recent else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, data: Dict[str, Any]) -> TechnicalSignal:
        """Compute TechnicalSignal from raw OHLCV data."""
        ohlc: List[Dict[str, float]] = data.get("ohlc_daily") or []
        volumes: List[float] = [float(v) for v in (data.get("volume_daily") or [])]
        latest_price: float = float(data.get("latest_price") or 0.0)
        week52_high: float = float(data.get("52_week_high") or 0.0)

        closes = [bar.get("close", 0.0) for bar in ohlc]

        # --- Core indicators ---
        rsi = self._rsi(closes)
        macd_sig = self._macd_signal(closes)
        trend = self._trend_direction(closes)

        # --- Volatility (ATR / price) ---
        atr = self._atr(ohlc)
        volatility = round(atr / latest_price, 4) if latest_price > 0 else 0.0

        # --- Support / resistance (20-day window) ---
        recent_20 = ohlc[-20:] if len(ohlc) >= 20 else ohlc
        resistance = max((b.get("high", 0.0) for b in recent_20), default=0.0)
        support = min((b.get("low", 0.0) for b in recent_20), default=0.0)

        # --- Breakout: price ≥ 97% of 52-week high + volume spike ---
        vol_avg = sum(volumes[-20:]) / len(volumes[-20:]) if len(volumes) >= 20 else 0.0
        last_vol = volumes[-1] if volumes else 0.0
        vol_spike = last_vol > vol_avg * 1.3 if vol_avg > 0 else False
        breakout = bool(week52_high > 0 and latest_price >= week52_high * 0.97 and vol_spike)

        # --- Composite technical score ---
        rsi_score = min(max((rsi - 30.0) / 40.0, 0.0), 1.0)   # 30–70 maps to 0–1
        macd_score = 1.0 if macd_sig == "buy" else 0.0 if macd_sig == "sell" else 0.5
        trend_score = 1.0 if trend == "bullish" else 0.0 if trend == "bearish" else 0.5
        breakout_score = 1.0 if breakout else 0.0

        technical_score = round(
            0.30 * rsi_score
            + 0.25 * macd_score
            + 0.30 * trend_score
            + 0.15 * breakout_score,
            4,
        )

        return TechnicalSignal(
            technical_score=technical_score,
            rsi=rsi,
            macd_signal=macd_sig,
            volatility=min(volatility, 1.0),
            breakout_flag=breakout,
            trend_direction=trend,
            support_level=round(support, 2) if support else None,
            resistance_level=round(resistance, 2) if resistance else None,
        )

    # ------------------------------------------------------------------
    # Agent.run() contract
    # ------------------------------------------------------------------

    def run(self, ctx: AgentContext, **kwargs) -> AgentResult:
        started = self._pre_run()
        try:
            # Try to get StockDataContext from kwargs first
            stock_ctx: Optional["StockDataContext"] = kwargs.get("stock_context")
            
            if stock_ctx is not None:
                # Extract OHLCV data from StockDataContext
                data = self._extract_from_context(stock_ctx)
            else:
                # Legacy: use input_data dict
                data = dict(ctx.input_data) if ctx.input_data else {}
                data.update(kwargs)

            signal = self.analyse(data)
            payload = {"signal": signal.to_feature_dict(), "raw_signal": signal}

            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=True,
                started=started,
                completed=completed,
                payload=payload,
                checksum_parts=[str(v) for v in signal.to_feature_dict().values()],
            )
        except Exception as exc:
            completed = self._post_run()
            return self._result(
                ctx=ctx,
                success=False,
                started=started,
                completed=completed,
                payload={},
                errors=[AgentError(code="TECHNICAL_ERROR", message=str(exc))],
            )

    def _extract_from_context(self, stock_ctx: "StockDataContext") -> Dict[str, Any]:
        """Extract technical data fields from StockDataContext."""
        data: Dict[str, Any] = {}
        
        # Add stock identity
        data["symbol"] = stock_ctx.symbol
        
        # Convert PriceData list to OHLC dict list
        ohlc_daily: List[Dict[str, float]] = []
        volume_daily: List[float] = []
        
        for price_bar in stock_ctx.historical_ohlc:
            ohlc_daily.append({
                "open": price_bar.open,
                "high": price_bar.high,
                "low": price_bar.low,
                "close": price_bar.close,
            })
            volume_daily.append(float(price_bar.volume))
        
        data["ohlc_daily"] = ohlc_daily
        data["volume_daily"] = volume_daily
        data["latest_price"] = stock_ctx.last_close or 0.0
        
        # Calculate 52-week high from historical data
        if ohlc_daily:
            # Use last 252 days (approx 1 year)
            yearly_data = ohlc_daily[-252:]
            data["52_week_high"] = max(bar.get("high", 0.0) for bar in yearly_data)
        else:
            data["52_week_high"] = 0.0
        
        return data
