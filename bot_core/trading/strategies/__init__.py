"""Plugin-based trading strategies built around :class:`TradingParameters`."""
from __future__ import annotations

from .plugins import (
    ArbitrageStrategy,
    DayTradingStrategy,
    MeanReversionStrategy,
    StrategyCatalog,
    StrategyPlugin,
    TrendFollowingStrategy,
)

__all__ = [
    "ArbitrageStrategy",
    "DayTradingStrategy",
    "MeanReversionStrategy",
    "StrategyCatalog",
    "StrategyPlugin",
    "TrendFollowingStrategy",
]
