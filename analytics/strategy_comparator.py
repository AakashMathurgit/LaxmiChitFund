"""StrategyComparator — pulls real broker truth from both Alpaca accounts and
computes comparable performance metrics for intraday vs swing.

Source of truth is Alpaca itself (both accounts started at $1,000,000 paper), so
this never relies on the local JSON sims. Per-symbol realized P&L is computed
FIFO from the fill history; unrealized P&L comes straight from open positions.
"""

from __future__ import annotations

import os
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import requests
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CREDS = os.path.join(_ROOT, "credentials.yaml")
_PAPER = "https://paper-api.alpaca.markets/v2"
INITIAL_CAPITAL = 1_000_000.0


def resolve_accounts() -> Dict[str, Dict[str, str]]:
    """Return {'intraday': {key,secret,endpoint}, 'swing': {...}} from env
    (preferred) or credentials.yaml. Skips an account whose keys are missing."""
    creds = {}
    if os.path.exists(_CREDS):
        try:
            with open(_CREDS, "r", encoding="utf-8") as f:
                creds = yaml.safe_load(f) or {}
        except Exception:
            creds = {}

    a1 = creds.get("alpaca", {}) or {}
    a2 = creds.get("alpaca_swing", {}) or {}
    accounts = {
        "intraday": {
            "key": os.environ.get("ALPACA_KEY_ID", a1.get("key_id", "")),
            "secret": os.environ.get("ALPACA_SECRET_KEY", a1.get("secret_key", "")),
            "endpoint": os.environ.get("ALPACA_ENDPOINT", a1.get("endpoint", _PAPER)),
        },
        "swing": {
            "key": os.environ.get("ALPACA_SWING_KEY_ID", a2.get("key_id", "")),
            "secret": os.environ.get("ALPACA_SWING_SECRET_KEY", a2.get("secret_key", "")),
            "endpoint": os.environ.get("ALPACA_SWING_ENDPOINT", a2.get("endpoint", _PAPER)),
        },
    }
    return {k: v for k, v in accounts.items() if v["key"] and v["secret"]}


def _headers(acct: Dict[str, str]) -> Dict[str, str]:
    return {
        "APCA-API-KEY-ID": acct["key"],
        "APCA-API-SECRET-KEY": acct["secret"],
    }


def _get(acct: Dict[str, str], path: str, params: Optional[dict] = None) -> Any:
    try:
        resp = requests.get(f"{acct['endpoint']}{path}", headers=_headers(acct),
                            params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _all_fills(acct: Dict[str, str]) -> List[Dict[str, Any]]:
    """Page through FILL activities (oldest->newest)."""
    fills: List[Dict[str, Any]] = []
    page_token = None
    for _ in range(20):  # cap pages
        params = {"activity_types": "FILL", "page_size": 100, "direction": "asc"}
        if page_token:
            params["page_token"] = page_token
        batch = _get(acct, "/account/activities", params)
        if not batch:
            break
        fills.extend(batch)
        if len(batch) < 100:
            break
        page_token = batch[-1].get("id")
    return fills


def _fifo_realized(fills: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, List[float]]]:
    """FIFO realized P&L per symbol (handles long and short). Returns
    (realized_by_symbol, closed_trade_pnls_by_symbol)."""
    lots: Dict[str, deque] = defaultdict(deque)   # symbol -> deque of [qty_signed, price]
    realized: Dict[str, float] = defaultdict(float)
    closed: Dict[str, List[float]] = defaultdict(list)

    for f in fills:
        sym = f.get("symbol")
        side = (f.get("side") or "").lower()
        try:
            qty = float(f.get("qty", 0))
            price = float(f.get("price", 0))
        except (TypeError, ValueError):
            continue
        if not sym or qty <= 0 or price <= 0:
            continue
        signed = qty if side == "buy" else -qty
        dq = lots[sym]
        # Close against opposite-direction lots first (FIFO).
        while signed != 0 and dq and (dq[0][0] > 0) != (signed > 0):
            lot_qty, lot_price = dq[0]
            match = min(abs(signed), abs(lot_qty))
            pnl = (price - lot_price) * match if lot_qty > 0 else (lot_price - price) * match
            realized[sym] += pnl
            closed[sym].append(pnl)
            if abs(lot_qty) > match:
                dq[0][0] = lot_qty - match if lot_qty > 0 else lot_qty + match
            else:
                dq.popleft()
            signed = signed - match if signed > 0 else signed + match
        if signed != 0:
            dq.append([signed, price])
    return dict(realized), dict(closed)


def account_metrics(name: str, acct: Dict[str, str]) -> Dict[str, Any]:
    """Compute full performance metrics for one Alpaca account."""
    account = _get(acct, "/account") or {}
    positions = _get(acct, "/positions") or []
    open_orders = _get(acct, "/orders", {"status": "open", "limit": 100}) or []
    fills = _all_fills(acct)
    realized, closed = _fifo_realized(fills)

    try:
        equity = float(account.get("equity", 0) or 0)
    except (TypeError, ValueError):
        equity = 0.0

    unrealized_by_symbol: Dict[str, float] = {}
    open_positions = []
    for p in positions:
        sym = p.get("symbol")
        try:
            upl = float(p.get("unrealized_pl", 0) or 0)
        except (TypeError, ValueError):
            upl = 0.0
        unrealized_by_symbol[sym] = upl
        open_positions.append({
            "symbol": sym,
            "qty": p.get("qty"),
            "avg_entry": p.get("avg_entry_price"),
            "current_price": p.get("current_price"),
            "market_value": p.get("market_value"),
            "unrealized_pl": round(upl, 2),
            "unrealized_plpc": round(float(p.get("unrealized_plpc", 0) or 0) * 100, 2),
        })

    # Per-symbol total P&L = realized + unrealized.
    symbols = set(realized) | set(unrealized_by_symbol)
    per_symbol = []
    for s in symbols:
        r = realized.get(s, 0.0)
        u = unrealized_by_symbol.get(s, 0.0)
        per_symbol.append({"symbol": s, "realized": round(r, 2),
                           "unrealized": round(u, 2), "total": round(r + u, 2)})
    per_symbol.sort(key=lambda x: x["total"], reverse=True)

    all_closed = [pnl for lst in closed.values() for pnl in lst]
    wins = sum(1 for x in all_closed if x > 0)
    losses = sum(1 for x in all_closed if x < 0)
    win_rate = (wins / (wins + losses) * 100) if (wins + losses) else 0.0

    return {
        "strategy": name,
        "account_number": account.get("account_number"),
        "equity": round(equity, 2),
        "total_pl": round(equity - INITIAL_CAPITAL, 2),
        "total_return_pct": round((equity - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100, 3),
        "realized_pl": round(sum(realized.values()), 2),
        "unrealized_pl": round(sum(unrealized_by_symbol.values()), 2),
        "num_fills": len(fills),
        "pending_orders": len(open_orders),
        "closed_trades": len(all_closed),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(win_rate, 1),
        "open_positions": open_positions,
        "top_winners": per_symbol[:5],
        "top_losers": per_symbol[-5:][::-1] if len(per_symbol) > 5 else [],
        "per_symbol": per_symbol,
    }


def compare() -> Dict[str, Any]:
    """Pull both accounts and build a head-to-head comparison."""
    accounts = resolve_accounts()
    metrics = {name: account_metrics(name, acct) for name, acct in accounts.items()}

    winner = None
    if "intraday" in metrics and "swing" in metrics:
        winner = max(metrics, key=lambda k: metrics[k]["total_pl"])

    return {"metrics": metrics, "winner": winner}
