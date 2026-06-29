"""Notifier — one place for phone/app alerts across all flows.

Wraps PushoverController so every flow sends alerts the same way. Degrades
gracefully (logs, returns False) when Pushover keys are absent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..utils.logger import get_logger


class Notifier:
    def __init__(self, market: str = "US"):
        self._market = market
        self._logger = get_logger("Notifier")
        self._pushover = None
        try:
            from src.main.controllers.pushover_controller import PushoverController
            self._pushover = PushoverController(market=market)
        except Exception as e:
            self._logger.warning(f"Pushover init failed: {e}")

    @property
    def enabled(self) -> bool:
        return bool(self._pushover and getattr(self._pushover, "_enabled", False))

    def send(self, message: str, title: str = "LCF Alert") -> bool:
        if not self.enabled:
            self._logger.info(f"Notifier disabled — would send: {title}")
            return False
        try:
            return self._pushover._send(message, title=title)
        except Exception as e:
            self._logger.warning(f"Notifier send failed: {e}")
            return False

    def notify_actions(self, actions: List[Dict[str, Any]], title: str = "LCF Actions") -> bool:
        """Send a concise digest of actionable BUY/SELL items.

        Each action: {symbol, decision, confidence, detail}.
        """
        if not actions:
            return False
        lines = [f"<b>{title}</b>"]
        for a in actions:
            sym = a.get("symbol", "?")
            decision = a.get("decision", "?")
            conf = a.get("confidence", 0.0) or 0.0
            detail = a.get("detail", "")
            lines.append(f"{decision} <b>{sym}</b> (conf {conf:.0%}) {detail}".rstrip())
        return self.send("\n".join(lines), title=title)
