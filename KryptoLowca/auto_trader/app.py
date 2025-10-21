"""Compatibility shim delegating to :mod:`bot_core.auto_trader.app`."""
from __future__ import annotations

from importlib import import_module
from sys import modules
from typing import Any, Callable

from bot_core.alerts import emit_alert as _core_emit_alert
from bot_core.auto_trader.app import AutoTrader, RiskDecision

__all__ = ["AutoTrader", "RiskDecision", "emit_alert", "_emit_alert"]


def _resolve_emit_alert() -> Callable[..., None]:
    """Zwraca handler alertów preferując ten z pakietu ``KryptoLowca``."""

    package_name = __package__ or "KryptoLowca.auto_trader"
    package = modules.get(package_name)
    if package is None:
        try:
            package = import_module(package_name)
        except ModuleNotFoundError:
            package = None

    handler = getattr(package, "emit_alert", None) if package else None
    if callable(handler):
        return handler
    return _core_emit_alert


def emit_alert(*args: Any, **kwargs: Any) -> None:
    """Public delegat zachowujący historyczne API modułu."""

    handler = _resolve_emit_alert()
    handler(*args, **kwargs)


def _emit_alert(*args: Any, **kwargs: Any) -> None:
    """Alias wykorzystywany przez moduły pomocnicze wewnątrz pakietu."""

    emit_alert(*args, **kwargs)
