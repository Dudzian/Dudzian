"""Compatibility shim delegating to :mod:`bot_core.auto_trader.app`."""
from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from bot_core.alerts import emit_alert as _core_emit_alert
from bot_core.auto_trader.app import AutoTrader, RiskDecision

__all__ = ["AutoTrader", "RiskDecision", "_emit_alert"]


def _resolve_emit_alert() -> Callable[..., None]:
    module = import_module(__package__ or "KryptoLowca.auto_trader")
    handler: Callable[..., None] = getattr(module, "emit_alert", _core_emit_alert)
    return handler


def _emit_alert(*args: Any, **kwargs: Any) -> None:
    """Delegate alert emission to the package-level handler."""

    handler = _resolve_emit_alert()
    handler(*args, **kwargs)
