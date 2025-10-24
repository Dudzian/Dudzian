"""Plugin-based trading strategies built around :class:`TradingParameters`."""
from __future__ import annotations

from .plugins import (
    ArbitrageStrategy,
    DayTradingStrategy,
    GridTradingStrategy,
    MeanReversionStrategy,
    OptionsIncomeStrategy,
    ScalpingStrategy,
    StatisticalArbitrageStrategy,
    StrategyCatalog,
    StrategyPlugin,
    TrendFollowingStrategy,
    VolatilityTargetStrategy,
)

__all__ = [
    "ArbitrageStrategy",
    "DayTradingStrategy",
    "GridTradingStrategy",
    "MeanReversionStrategy",
    "OptionsIncomeStrategy",
    "ScalpingStrategy",
    "StatisticalArbitrageStrategy",
    "StrategyCatalog",
    "StrategyPlugin",
    "TrendFollowingStrategy",
    "VolatilityTargetStrategy",
]
