"""Compatibility wrapper for the legacy backtest simulation module."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_LEGACY_MODULE: ModuleType = import_module("KryptoLowca.backtest.simulation")

BacktestFill = getattr(_LEGACY_MODULE, "BacktestFill")
MatchingConfig = getattr(_LEGACY_MODULE, "MatchingConfig")
MatchingEngine = getattr(_LEGACY_MODULE, "MatchingEngine")

__all__ = ["BacktestFill", "MatchingConfig", "MatchingEngine"]

def __getattr__(name: str) -> Any:  # pragma: no cover - passthrough for other helpers
    return getattr(_LEGACY_MODULE, name)
