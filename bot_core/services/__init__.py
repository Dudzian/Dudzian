"""Nowoczesne usługi runtime przeniesione z pakietu legacy."""
from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "alerting",
    "atr_monitor",
    "indicators",
    "marketdata",
    "order_router",
    "performance_monitor",
    "persistence",
    "position_sizer",
    "risk_dashboard",
    "risk_guard",
    "risk_manager",
    "stop_tp",
    "strategy_engine",
    "strategy_service",
    "walkforward_service",
    "wfo",
]

_modules = {name: name for name in __all__}


def __getattr__(name: str) -> Any:
    if name in _modules:
        module = import_module(f"bot_core.services.{_modules[name]}")
        globals()[name] = module
        return module
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
