"""LCFRuntime — load config/credentials once, expose shared singletons.

Every flow receives an LCFRuntime and pulls the shared LLM, BrokerRouter,
Notifier, and news caches from it instead of building its own. This is what
lets the 5 flows co-run without duplicating expensive setup or diverging in how
they place trades / send alerts.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import yaml

from ..utils.logger import get_logger
from .broker_router import BrokerRouter
from .notifier import Notifier

_DEFAULT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LCFRuntime:
    def __init__(self, config: Dict[str, Any], base_path: str):
        self.config = config
        self.base_path = base_path
        self.config_path = os.path.join(base_path, "config.yaml")
        self._logger = get_logger("LCFRuntime")
        self._llm = None
        self._broker_router: Optional[BrokerRouter] = None
        self._notifiers: Dict[str, Notifier] = {}
        self._news_caches: Dict[str, Any] = {}

    @classmethod
    def from_config(cls, config_path: Optional[str] = None) -> "LCFRuntime":
        base = os.path.dirname(os.path.abspath(config_path)) if config_path else _DEFAULT_ROOT
        path = config_path or os.path.join(base, "config.yaml")
        config: Dict[str, Any] = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            pass
        return cls(config=config, base_path=base)

    # ------------------------------------------------------------------
    # Shared singletons (lazy)
    # ------------------------------------------------------------------

    @property
    def llm(self):
        if self._llm is None:
            from src.main.agents.adapters.llm_adapter import LLMAdapter
            self._llm = LLMAdapter.from_config(self.config, base_path=self.base_path)
        return self._llm

    @property
    def broker_router(self) -> BrokerRouter:
        if self._broker_router is None:
            self._broker_router = BrokerRouter(self.config)
        return self._broker_router

    def notifier(self, market: str = "US") -> Notifier:
        if market not in self._notifiers:
            self._notifiers[market] = Notifier(market=market)
        return self._notifiers[market]

    def news_cache(self, market: str = "US"):
        if market not in self._news_caches:
            from src.main.controllers.news_cache import NewsCache
            fname = "news_cache_us.jsonl" if market == "US" else "news_cache_ind.jsonl"
            path = os.path.join(self.base_path, "data", fname)
            self._news_caches[market] = NewsCache(path)
        return self._news_caches[market]
