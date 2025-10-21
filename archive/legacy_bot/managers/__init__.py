"""Legacy namespace ensuring ``archive.legacy_bot.managers`` imports keep working."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Dict

_ALIAS_MAP: Dict[str, str] = {
    "ai_manager": "KryptoLowca.ai_manager",
    "config_manager": "KryptoLowca.config_manager",
    "database_manager": "KryptoLowca.database_manager",
    "exchange_adapter": "KryptoLowca.exchange_adapter",
    "exchange_core": "bot_core.exchanges.core",
    "exchange_manager": "KryptoLowca.exchange_manager",
    "live_exchange_ccxt": "bot_core.exchanges.ccxt_adapter",
    "paper_exchange": "KryptoLowca.paper_exchange",
    "report_manager": "KryptoLowca.report_manager",
    "risk_manager_adapter": "KryptoLowca.risk_manager",
    "scanner": "KryptoLowca.scanner",
    "security_manager": "KryptoLowca.security_manager",
}

__all__ = sorted(_ALIAS_MAP)


def __getattr__(name: str) -> ModuleType:  # pragma: no cover - trivial forwarding
    try:
        target = _ALIAS_MAP[name]
    except KeyError as exc:  # pragma: no cover - mirror default behaviour
        raise AttributeError(name) from exc
    module = import_module(target)
    globals()[name] = module
    return module


def __dir__() -> list[str]:  # pragma: no cover - trivial forwarding
    return __all__
