"""Adaptery Binance (lazy imports)."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

from bot_core.optional import missing_module_proxy

_LAZY_SUBMODULES = {"futures", "margin", "spot", "symbols"}
_LAZY_EXPORTS = {
    "BinanceFuturesAdapter": ("bot_core.exchanges.binance.futures", "BinanceFuturesAdapter"),
    "BinanceMarginAdapter": ("bot_core.exchanges.binance.margin", "BinanceMarginAdapter"),
    "BinanceSpotAdapter": ("bot_core.exchanges.binance.spot", "BinanceSpotAdapter"),
}


def _missing_dependency(module: str, symbol: str, error: Exception) -> Any:
    dependency = getattr(error, "name", None) or "dependency"
    message = (
        f"Brak opcjonalnej zależności '{dependency}' wymaganej dla "
        f"{module}:{symbol}. Zainstaluj ją, aby używać tego adaptera."
    )
    return missing_module_proxy(message, cause=error)


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY_SUBMODULES:
        mod: ModuleType = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    if name in _LAZY_EXPORTS:
        module_path, symbol = _LAZY_EXPORTS[name]
        try:
            mod = importlib.import_module(module_path)
            value = getattr(mod, symbol)
        except (ModuleNotFoundError, ImportError) as exc:
            value = _missing_dependency(module_path, symbol, exc)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | _LAZY_SUBMODULES | set(_LAZY_EXPORTS.keys()))


__all__ = list(_LAZY_EXPORTS.keys()) + ["symbols"]
