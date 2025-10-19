"""Compatibility wrapper exposing AutoTrader under the bot_core namespace."""
from __future__ import annotations

from importlib import import_module
import sys

_runtime_module = import_module("bot_core.runtime")
if not hasattr(_runtime_module, "resolve_core_config_path"):
    _runtime_module.resolve_core_config_path = import_module(  # type: ignore[attr-defined]
        "bot_core.runtime.paths"
    ).resolve_core_config_path
if not hasattr(_runtime_module, "PaperTradingAdapter"):
    _runtime_module.PaperTradingAdapter = import_module(  # type: ignore[attr-defined]
        "bot_core.runtime.paper_trading"
    ).PaperTradingAdapter

_LEGACY_MODULE = import_module("KryptoLowca.auto_trader")

sys.modules[__name__] = _LEGACY_MODULE
