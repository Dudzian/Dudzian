"""Plugin-based trading strategies built around :class:`TradingParameters`."""
from __future__ import annotations

from .plugins import (
    ArbitrageStrategy,
    CrossExchangeHedgeStrategy,
    DayTradingStrategy,
    FuturesSpreadStrategy,
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
    "CrossExchangeHedgeStrategy",
    "DayTradingStrategy",
    "FuturesSpreadStrategy",
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
