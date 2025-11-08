"""Plugin-based trading strategies built around :class:`TradingParameters`."""
from __future__ import annotations

from .plugins import (
    AdaptiveMarketMakingPlugin,
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
    TriangularArbitragePlugin,
    VolatilityTargetStrategy,
)

__all__ = [
    "AdaptiveMarketMakingPlugin",
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
    "TriangularArbitragePlugin",
    "VolatilityTargetStrategy",
]
