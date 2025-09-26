"""Warstwa bazowa strategii – interfejs oraz rejestr."""

from .engine import BaseStrategy, DataProvider, StrategyContext, StrategyError, StrategyMetadata, StrategySignal
from .registry import StrategyRegistry, registry, strategy

__all__ = [
    "BaseStrategy",
    "DataProvider",
    "StrategyContext",
    "StrategyError",
    "StrategyMetadata",
    "StrategySignal",
    "StrategyRegistry",
    "registry",
    "strategy",
]
