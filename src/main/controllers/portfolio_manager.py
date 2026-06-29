"""Portfolio Manager — persistent holdings, cash, trade history & watchlist.

Stores portfolio state in JSON files. Updates ONLY after trade execution
confirmation to prevent data corruption from failed orders.

Files:
    data/portfolio.json  — holdings, cash, trade history
    data/watchlist.json  — stocks to monitor daily with priority

Usage:
    pm = PortfolioManager("data/portfolio.json", "data/watchlist.json")
    pm.load()

    # Watchlist
    pm.add_to_watchlist("RELIANCE", reason="Core holding", priority="high")
    pm.get_watchlist_symbols()

    # Portfolio
    pm.open_position(symbol="TCS", quantity=10, entry_price=3920, ...)
    pm.close_position("TCS", exit_price=4050, exit_reason="Target hit")
    print(pm.portfolio_text_summary())
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
class WatchlistItem:
    """A stock on the personal watchlist."""
    symbol: str
    added_date: str
    reason: str = ""
    priority: str = "medium"   # high, medium, low
    last_signal: str = ""      # last BUY/SELL/HOLD signal
    last_signal_date: str = ""
    notes: str = ""


@dataclass
class Holding:
    """A single open position in the portfolio."""
    symbol: str
    quantity: int
    entry_price: float
    entry_date: str
    stop_loss: float
    target_price: float
    trailing_stop_pct: Optional[float] = None
    current_price: float = 0.0
    highest_price: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass
class TradeRecord:
    """A completed (closed) or failed trade record."""
    symbol: str
    action: str               # BUY or SELL
    quantity: int
    entry_price: float
    exit_price: Optional[float] = None
    entry_date: str = ""
    exit_date: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    status: str = "OPEN"      # OPEN | CLOSED | FAILED
    exit_reason: str = ""
    slippage: float = 0.0
    order_id: Optional[str] = None


@dataclass
class PortfolioState:
    """Full portfolio snapshot."""
    cash: float = 1000000.0   # Default ₹10 lakh
    initial_capital: float = 1000000.0
    holdings: List[Holding] = field(default_factory=list)
    trade_history: List[TradeRecord] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data")
_DEFAULT_PORTFOLIO = os.path.join(_DATA_DIR, "portfolio.json")
_DEFAULT_WATCHLIST = os.path.join(_DATA_DIR, "watchlist.json")


class PortfolioManager:
    """Manages portfolio state and watchlist with JSON file persistence."""

    def __init__(
        self,
        portfolio_path: str = _DEFAULT_PORTFOLIO,
        watchlist_path: str = _DEFAULT_WATCHLIST,
    ):
        self._path = portfolio_path
        self._watchlist_path = watchlist_path
        self._state = PortfolioState(
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
        )
        self._watchlist: List[WatchlistItem] = []

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load portfolio and watchlist from JSON files."""
        # Portfolio
        if not os.path.exists(self._path) or os.path.getsize(self._path) == 0:
            self.save()
        else:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._state = PortfolioState(
                cash=data.get("cash", 1000000.0),
                initial_capital=data.get("initial_capital", 1000000.0),
                holdings=[Holding(**h) for h in data.get("holdings", [])],
                trade_history=[TradeRecord(**t) for t in data.get("trade_history", [])],
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )

        # Watchlist
        if os.path.exists(self._watchlist_path):
            with open(self._watchlist_path, "r", encoding="utf-8") as f:
                wl_data = json.load(f)
            self._watchlist = [WatchlistItem(**w) for w in wl_data.get("watchlist", [])]
        else:
            self._watchlist = []
            self._save_watchlist()

        logger.info(
            f"PortfolioManager loaded: {len(self._state.holdings)} positions, "
            f"Rs.{self._state.cash:,.0f} cash, {len(self._watchlist)} watchlist"
        )

    def save(self) -> None:
        """Persist current state to JSON."""
        self._state.updated_at = datetime.now().isoformat()
        data = {
            "cash": round(self._state.cash, 2),
            "initial_capital": self._state.initial_capital,
            "holdings": [asdict(h) for h in self._state.holdings],
            "trade_history": [asdict(t) for t in self._state.trade_history],
            "created_at": self._state.created_at,
            "updated_at": self._state.updated_at,
        }
        os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_position(self, symbol: str) -> Optional[Holding]:
        """Get open holding for symbol, or None."""
        for h in self._state.holdings:
            if h.symbol.upper() == symbol.upper():
                return h
        return None

    def has_position(self, symbol: str) -> bool:
        return self.get_position(symbol) is not None

    def get_available_cash(self) -> float:
        return self._state.cash

    def get_portfolio_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Total value = cash + sum of holdings at current prices."""
        total = self._state.cash
        for h in self._state.holdings:
            price = (current_prices or {}).get(h.symbol, h.current_price) or h.entry_price
            total += h.quantity * price
        return total

    def get_total_exposure(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """Fraction of portfolio currently invested (0-1)."""
        pv = self.get_portfolio_value(current_prices)
        if pv <= 0:
            return 0.0
        invested = sum(
            h.quantity * ((current_prices or {}).get(h.symbol, h.current_price) or h.entry_price)
            for h in self._state.holdings
        )
        return invested / pv

    def get_all_holdings(self) -> List[Holding]:
        return list(self._state.holdings)

    def get_trade_history(self) -> List[TradeRecord]:
        return list(self._state.trade_history)

    # ------------------------------------------------------------------
    # Trade operations
    # ------------------------------------------------------------------

    def open_position(
        self,
        symbol: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        target_price: float,
        trailing_stop_pct: Optional[float] = None,
        order_id: Optional[str] = None,
        slippage: float = 0.0,
    ) -> bool:
        """Open a new position. Returns False if already holding or insufficient cash."""
        if self.has_position(symbol):
            return False

        cost = quantity * entry_price
        if cost > self._state.cash:
            return False

        self._state.cash -= cost
        self._state.holdings.append(Holding(
            symbol=symbol.upper(),
            quantity=quantity,
            entry_price=entry_price,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            stop_loss=stop_loss,
            target_price=target_price,
            trailing_stop_pct=trailing_stop_pct,
            current_price=entry_price,
            highest_price=entry_price,
        ))
        self._state.trade_history.append(TradeRecord(
            symbol=symbol.upper(),
            action="BUY",
            quantity=quantity,
            entry_price=entry_price,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            status="OPEN",
            slippage=slippage,
            order_id=order_id,
        ))
        self.save()
        return True

    def close_position(
        self,
        symbol: str,
        exit_price: float,
        exit_reason: str = "manual",
        order_id: Optional[str] = None,
        slippage: float = 0.0,
    ) -> Optional[TradeRecord]:
        """Close an existing position. Returns the trade record or None."""
        holding = self.get_position(symbol)
        if holding is None:
            return None

        pnl = (exit_price - holding.entry_price) * holding.quantity
        pnl_pct = (exit_price - holding.entry_price) / holding.entry_price if holding.entry_price else 0.0

        self._state.cash += holding.quantity * exit_price
        self._state.holdings = [
            h for h in self._state.holdings
            if h.symbol.upper() != symbol.upper()
        ]

        record = TradeRecord(
            symbol=symbol.upper(),
            action="SELL",
            quantity=holding.quantity,
            entry_price=holding.entry_price,
            exit_price=exit_price,
            entry_date=holding.entry_date,
            exit_date=datetime.now().strftime("%Y-%m-%d"),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 4),
            status="CLOSED",
            exit_reason=exit_reason,
            slippage=slippage,
            order_id=order_id,
        )
        self._state.trade_history.append(record)
        self.save()
        return record

    def update_holding_price(self, symbol: str, current_price: float) -> None:
        """Update current price and unrealized P&L for a holding."""
        holding = self.get_position(symbol)
        if holding is None:
            return
        holding.current_price = current_price
        holding.highest_price = max(holding.highest_price, current_price)
        holding.unrealized_pnl = round(
            (current_price - holding.entry_price) * holding.quantity, 2
        )

    def record_failed_trade(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float,
        reason: str,
    ) -> None:
        """Log a failed trade attempt without modifying holdings/cash."""
        self._state.trade_history.append(TradeRecord(
            symbol=symbol.upper(),
            action=action,
            quantity=quantity,
            entry_price=price,
            entry_date=datetime.now().strftime("%Y-%m-%d"),
            status="FAILED",
            exit_reason=reason,
        ))
        self.save()

    # ------------------------------------------------------------------
    # P&L
    # ------------------------------------------------------------------

    def get_realized_pnl(self) -> float:
        """Sum of P&L from all closed trades."""
        return sum(
            t.pnl for t in self._state.trade_history
            if t.status == "CLOSED"
        )

    def get_unrealized_pnl(self) -> float:
        """Sum of unrealized P&L from open holdings."""
        return sum(h.unrealized_pnl for h in self._state.holdings)

    def summary(self) -> Dict[str, Any]:
        """Human-readable portfolio summary."""
        closed = [t for t in self._state.trade_history if t.status == "CLOSED"]
        return {
            "cash": round(self._state.cash, 2),
            "open_positions": len(self._state.holdings),
            "holdings": [
                {"symbol": h.symbol, "qty": h.quantity, "entry": h.entry_price, "pnl": h.unrealized_pnl}
                for h in self._state.holdings
            ],
            "total_trades": len(closed),
            "realized_pnl": round(self.get_realized_pnl(), 2),
            "unrealized_pnl": round(self.get_unrealized_pnl(), 2),
        }

    # ------------------------------------------------------------------
    # Watchlist
    # ------------------------------------------------------------------

    def _save_watchlist(self) -> None:
        os.makedirs(os.path.dirname(self._watchlist_path), exist_ok=True)
        data = {"watchlist": [asdict(w) for w in self._watchlist]}
        with open(self._watchlist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def get_watchlist(self) -> List[WatchlistItem]:
        return list(self._watchlist)

    def get_watchlist_symbols(self) -> List[str]:
        return [w.symbol for w in self._watchlist]

    def add_to_watchlist(
        self, symbol: str, reason: str = "", priority: str = "medium", notes: str = ""
    ) -> bool:
        symbol = symbol.upper().strip()
        if any(w.symbol == symbol for w in self._watchlist):
            logger.info(f"[Watchlist] {symbol} already exists")
            return False
        self._watchlist.append(WatchlistItem(
            symbol=symbol,
            added_date=datetime.now().strftime("%Y-%m-%d"),
            reason=reason,
            priority=priority,
            notes=notes,
        ))
        self._save_watchlist()
        logger.info(f"[Watchlist] Added {symbol} ({priority}): {reason}")
        return True

    def remove_from_watchlist(self, symbol: str) -> bool:
        symbol = symbol.upper().strip()
        before = len(self._watchlist)
        self._watchlist = [w for w in self._watchlist if w.symbol != symbol]
        if len(self._watchlist) < before:
            self._save_watchlist()
            logger.info(f"[Watchlist] Removed {symbol}")
            return True
        return False

    def update_watchlist_signal(self, symbol: str, signal: str) -> None:
        symbol = symbol.upper().strip()
        for w in self._watchlist:
            if w.symbol == symbol:
                w.last_signal = signal
                w.last_signal_date = datetime.now().strftime("%Y-%m-%d")
        self._save_watchlist()

    # ------------------------------------------------------------------
    # Signal Processing — auto-trade from pipeline results
    # ------------------------------------------------------------------

    def process_signals(
        self,
        results: List[Dict[str, Any]],
        auto_enter: bool = False,
        auto_exit: bool = True,
        mode: str = "adaptive",
    ) -> Dict[str, Any]:
        """Process pipeline results: update prices, check SL/targets, auto-trade.

        Args:
            results: List of stock analysis results from orchestrator
            auto_enter: If True, open positions on BUY signals
            auto_exit: If True, close positions on SELL signals
            mode: Current trading mode name

        Returns:
            Summary of actions taken
        """
        actions = {"entered": [], "exited": [], "sl_triggered": [], "target_hit": [], "updated": []}

        for r in results:
            symbol = r.get("symbol", "").upper()
            jd = r.get("judge_decision", {})
            payload = jd.get("payload", jd)
            decision = payload.get("decision", "HOLD")
            confidence = payload.get("confidence", 0)
            tp = r.get("trade_plan") or {}
            price = tp.get("current_price") or tp.get("entry_price", 0)

            if not symbol or price <= 0:
                continue

            # Update watchlist signal
            self.update_watchlist_signal(symbol, decision)

            # Update price if we hold it
            holding = self.get_position(symbol)
            if holding:
                self.update_holding_price(symbol, price)
                actions["updated"].append(symbol)

                # Check stop loss
                if holding.stop_loss > 0 and price <= holding.stop_loss:
                    record = self.close_position(symbol, price, exit_reason="Stop loss triggered")
                    if record:
                        actions["sl_triggered"].append(asdict(record))
                        logger.warning(f"[Portfolio] SL triggered: {symbol} @ Rs.{price:,.2f}")
                    continue

                # Check target
                if holding.target_price > 0 and price >= holding.target_price:
                    record = self.close_position(symbol, price, exit_reason="Target reached")
                    if record:
                        actions["target_hit"].append(asdict(record))
                        logger.info(f"[Portfolio] Target hit: {symbol} @ Rs.{price:,.2f}")
                    continue

            # Auto-exit on SELL signal
            if decision == "SELL" and auto_exit and holding:
                record = self.close_position(
                    symbol, price,
                    exit_reason=f"SELL signal (conf={confidence*100:.0f}%, mode={mode})"
                )
                if record:
                    actions["exited"].append(asdict(record))

            # Auto-enter on BUY signal
            if decision == "BUY" and auto_enter and not holding:
                sl = tp.get("stop_loss_price", 0)
                target = tp.get("target_price", 0)
                # Use 10% of cash per position
                invest = self._state.cash * 0.10
                shares = int(invest / price) if price > 0 else 0
                if shares > 0:
                    success = self.open_position(
                        symbol=symbol, quantity=shares, entry_price=price,
                        stop_loss=sl, target_price=target,
                    )
                    if success:
                        actions["entered"].append({
                            "symbol": symbol, "shares": shares,
                            "price": price, "confidence": confidence,
                        })
                        logger.info(f"[Portfolio] Auto-BUY: {symbol} {shares} shares @ Rs.{price:,.2f}")

        self.save()
        return actions

    # ------------------------------------------------------------------
    # Check stop losses / targets against live prices
    # ------------------------------------------------------------------

    def check_stop_losses(self, price_map: Dict[str, float]) -> List[str]:
        triggered = []
        for h in self._state.holdings:
            if h.stop_loss > 0 and price_map.get(h.symbol, float("inf")) <= h.stop_loss:
                triggered.append(h.symbol)
        return triggered

    def check_targets(self, price_map: Dict[str, float]) -> List[str]:
        triggered = []
        for h in self._state.holdings:
            if h.target_price > 0 and price_map.get(h.symbol, 0) >= h.target_price:
                triggered.append(h.symbol)
        return triggered

    # ------------------------------------------------------------------
    # Rich Text Summaries
    # ------------------------------------------------------------------

    def portfolio_text_summary(self) -> str:
        """Full text summary of portfolio + watchlist for terminal display."""
        holdings = self._state.holdings
        closed = [t for t in self._state.trade_history if t.status == "CLOSED"]
        cash = self._state.cash
        initial = self._state.initial_capital

        total_invested = sum(h.quantity * (h.current_price or h.entry_price) for h in holdings)
        total_unrealized = sum(h.unrealized_pnl for h in holdings)
        total_value = cash + total_invested
        total_return = ((total_value - initial) / initial * 100) if initial > 0 else 0
        total_realized = sum(t.pnl for t in closed)

        lines = []
        lines.append("=" * 55)
        lines.append("  PORTFOLIO SUMMARY")
        lines.append("=" * 55)
        lines.append(f"  Initial Capital:  Rs.{initial:>12,.2f}")
        lines.append(f"  Current Cash:     Rs.{cash:>12,.2f}")
        lines.append(f"  Invested Value:   Rs.{total_invested:>12,.2f}")
        lines.append(f"  Total Value:      Rs.{total_value:>12,.2f}")
        lines.append(f"  Total Return:        {total_return:>+10.2f}%")
        lines.append("")

        if holdings:
            lines.append(f"  OPEN POSITIONS ({len(holdings)})")
            lines.append("  " + "-" * 53)
            lines.append(f"  {'Symbol':<10} {'Qty':>5} {'Entry':>9} {'Current':>9} {'P&L':>10} {'%':>7}")
            for h in sorted(holdings, key=lambda x: x.unrealized_pnl, reverse=True):
                cp = h.current_price or h.entry_price
                pnl_pct = ((cp - h.entry_price) / h.entry_price * 100) if h.entry_price else 0
                lines.append(
                    f"  {h.symbol:<10} {h.quantity:>5} "
                    f"{h.entry_price:>9,.1f} {cp:>9,.1f} "
                    f"{h.unrealized_pnl:>+10,.1f} {pnl_pct:>+6.1f}%"
                )
                sl_str = f"SL:{h.stop_loss:,.0f}" if h.stop_loss > 0 else "SL:--"
                tgt_str = f"TGT:{h.target_price:,.0f}" if h.target_price > 0 else "TGT:--"
                entry_dt = h.entry_date
                lines.append(f"           {sl_str} | {tgt_str} | Since: {entry_dt}")
            lines.append(f"  {'':>10} {'':>5} {'':>9} {'TOTAL':>9} {total_unrealized:>+10,.1f}")
        else:
            lines.append("  No open positions")

        lines.append("")
        if closed:
            wins = [t for t in closed if t.pnl > 0]
            losses = [t for t in closed if t.pnl <= 0]
            win_rate = len(wins) / len(closed) * 100 if closed else 0
            lines.append(f"  TRADE HISTORY ({len(closed)} trades | Win Rate: {win_rate:.0f}%)")
            lines.append("  " + "-" * 53)
            recent = sorted(closed, key=lambda t: t.exit_date, reverse=True)[:5]
            for t in recent:
                tag = "WIN " if t.pnl > 0 else "LOSS"
                lines.append(
                    f"  {tag} {t.symbol:<10} {t.pnl:>+10,.1f} ({t.pnl_pct*100:>+.1f}%) "
                    f"{t.exit_reason}"
                )
            if len(closed) > 5:
                lines.append(f"  ... and {len(closed) - 5} more")
            lines.append(f"  Total Realized P&L: Rs.{total_realized:>+,.2f}")

        lines.append("")
        # Watchlist
        if self._watchlist:
            lines.append(f"  WATCHLIST ({len(self._watchlist)} stocks)")
            lines.append("  " + "-" * 53)
            for priority in ["high", "medium", "low"]:
                stocks = [w for w in self._watchlist if w.priority == priority]
                if stocks:
                    for w in stocks:
                        held = " [HOLDING]" if self.has_position(w.symbol) else ""
                        sig = f" [{w.last_signal}]" if w.last_signal else ""
                        lines.append(f"  [{priority[0].upper()}] {w.symbol:<10} {w.reason[:30]}{held}{sig}")

        lines.append("")
        lines.append(f"  Updated: {self._state.updated_at}")
        lines.append("=" * 55)
        return "\n".join(lines)

    def format_whatsapp_portfolio(self) -> str:
        """Format portfolio for WhatsApp message."""
        holdings = self._state.holdings
        closed = [t for t in self._state.trade_history if t.status == "CLOSED"]
        cash = self._state.cash
        initial = self._state.initial_capital

        total_invested = sum(h.quantity * (h.current_price or h.entry_price) for h in holdings)
        total_unrealized = sum(h.unrealized_pnl for h in holdings)
        total_value = cash + total_invested
        total_return = ((total_value - initial) / initial * 100) if initial > 0 else 0
        total_realized = sum(t.pnl for t in closed)

        lines = []
        lines.append("*PORTFOLIO STATUS*")
        lines.append(f"Capital: Rs.{initial:,.0f}")
        lines.append(f"Cash: Rs.{cash:,.0f} | Invested: Rs.{total_invested:,.0f}")
        lines.append(f"Total Value: Rs.{total_value:,.0f} ({total_return:+.1f}%)")
        lines.append("")

        if holdings:
            lines.append(f"*Open Positions ({len(holdings)})*")
            for h in sorted(holdings, key=lambda x: x.unrealized_pnl, reverse=True):
                cp = h.current_price or h.entry_price
                pnl_pct = ((cp - h.entry_price) / h.entry_price * 100) if h.entry_price else 0
                lines.append(
                    f"  {h.symbol}: {h.quantity}x Rs.{h.entry_price:,.0f} -> "
                    f"Rs.{cp:,.0f} ({pnl_pct:+.1f}%)"
                )
                if h.stop_loss > 0 or h.target_price > 0:
                    lines.append(f"    SL: Rs.{h.stop_loss:,.0f} | Target: Rs.{h.target_price:,.0f}")
            lines.append(f"  Unrealized: Rs.{total_unrealized:+,.0f}")
        else:
            lines.append("No open positions")

        if closed:
            wins = len([t for t in closed if t.pnl > 0])
            lines.append(f"\nClosed: {len(closed)} trades | Wins: {wins} | P&L: Rs.{total_realized:+,.0f}")

        # Watchlist summary
        if self._watchlist:
            wl_names = ", ".join(w.symbol for w in self._watchlist[:8])
            lines.append(f"\n*Watchlist ({len(self._watchlist)})*: {wl_names}")

        lines.append(f"\n_Updated: {datetime.now().strftime('%H:%M')}_")
        return "\n".join(lines)
