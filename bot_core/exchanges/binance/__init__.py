"""Adaptery Binance."""

from __future__ import annotations

import importlib

from bot_core.optional import missing_module_proxy

_LAZY_EXPORTS = {
    "BinanceFuturesAdapter": "bot_core.exchanges.binance.futures",
    "BinanceMarginAdapter": "bot_core.exchanges.binance.margin",
    "BinanceSpotAdapter": "bot_core.exchanges.binance.spot",
    "symbols": "bot_core.exchanges.binance.symbols",
}

__all__ = list(_LAZY_EXPORTS.keys())


def __getattr__(name: str):
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    try:
        mod = importlib.import_module(module)
        value = mod if name == "symbols" else getattr(mod, name)
        globals()[name] = value
        return value
    except (ModuleNotFoundError, ImportError) as exc:
        dep = getattr(exc, "name", None) or "dependency"
        message = f"Brak opcjonalnej zależności '{dep}' wymaganej dla {module}:{name}."
        proxy = missing_module_proxy(message, cause=exc)
        globals()[name] = proxy
        return proxy


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
