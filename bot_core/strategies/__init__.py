"""Silniki strategii oraz optymalizacja.

Pakiet udostępnia komplet strategii budowanych przez katalog OEM.  Wbudowane
silniki obejmują m.in. ``daily_trend_momentum`` (momentum dzienne),
``grid_trading`` (handel siatkowy), ``dollar_cost_averaging`` (akumulacja DCA),
``volatility_target`` (zarządzanie zmiennością), strategie arbitrażowe,
hedgingowe oraz intraday.  Aktualną listę silników można uzyskać poprzez
funkcje :func:`describe_supported_strategies` oraz
:func:`supported_strategy_keys`.
"""

from __future__ import annotations

from functools import lru_cache

from bot_core.strategies.base import (
    MarketSnapshot,
    SignalLeg,
    StrategyEngine,
    StrategySignal,
    WalkForwardOptimizer,
)
from bot_core.strategies.day_trading import DayTradingSettings, DayTradingStrategy
from bot_core.strategies.daily_trend import (
    DailyTrendMomentumSettings,
    DailyTrendMomentumStrategy,
)
from bot_core.strategies.dca import (
    DollarCostAveragingSettings,
    DollarCostAveragingStrategy,
)
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    PresetLicenseState,
    PresetLicenseStatus,
    StrategyCatalog,
    StrategyDefinition,
    StrategyPresetDescriptor,
    StrategyPresetProfile,
    StrategyPresetWizard,
)
from bot_core.strategies.installer import (
    MarketplaceInstallResult,
    MarketplacePresetInstaller,
)
from bot_core.strategies.marketplace import (
    MarketplaceAuthor,
    MarketplaceCatalog,
    MarketplaceCatalogError,
    MarketplacePreset,
    load_catalog,
)
from bot_core.strategies.regime_workflow import (
    ActivationCadenceStats,
    ActivationHistoryStats,
    ActivationTransitionStats,
    ActivationUptimeStats,
    PresetVersionInfo,
    RegimePresetActivation,
    StrategyRegimeWorkflow,
)
from bot_core.strategies.cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from bot_core.strategies.cross_exchange_hedge import (
    CrossExchangeHedgeSettings,
    CrossExchangeHedgeStrategy,
)
from bot_core.strategies.futures_spread import (
    FuturesSpreadSettings,
    FuturesSpreadStrategy,
)
from bot_core.strategies.grid import GridTradingSettings, GridTradingStrategy
from bot_core.strategies.adaptive_market_making import (
    AdaptiveMarketMakingSettings,
    AdaptiveMarketMakingStrategy,
)
from bot_core.strategies.market_making import (
    MarketMakingSettings,
    MarketMakingStrategy,
)
from bot_core.strategies.mean_reversion import MeanReversionSettings, MeanReversionStrategy
from bot_core.strategies.options import OptionsIncomeSettings, OptionsIncomeStrategy
from bot_core.strategies.scalping import ScalpingSettings, ScalpingStrategy
from bot_core.strategies.statistical_arbitrage import (
    StatisticalArbitrageSettings,
    StatisticalArbitrageStrategy,
)
from bot_core.strategies.testing import (
    ParameterTestResult,
    StrategyParameterTestReport,
    StrategyParameterTester,
)
from bot_core.strategies.triangular_arbitrage import (
    TriangularArbitrageSettings,
    TriangularArbitrageStrategy,
)
from bot_core.strategies.public import StrategyDescriptor, list_available_strategies
from bot_core.strategies.volatility_target import (
    VolatilityTargetSettings,
    VolatilityTargetStrategy,
)


@lru_cache(maxsize=1)
def describe_supported_strategies() -> tuple[StrategyDescriptor, ...]:
    """Zwraca listę strategii dostępnych w domyślnym katalogu."""

    return list_available_strategies()


def supported_strategy_keys() -> tuple[str, ...]:
    """Zwraca identyfikatory silników strategii wspieranych w dystrybucji."""

    return tuple(entry.engine for entry in describe_supported_strategies())


__all__ = [
    "MarketSnapshot",
    "SignalLeg",
    "StrategyEngine",
    "StrategySignal",
    "WalkForwardOptimizer",
    "DailyTrendMomentumSettings",
    "DailyTrendMomentumStrategy",
    "DayTradingSettings",
    "DayTradingStrategy",
    "DollarCostAveragingSettings",
    "DollarCostAveragingStrategy",
    "StrategyCatalog",
    "StrategyDefinition",
    "StrategyPresetWizard",
    "StrategyPresetDescriptor",
    "StrategyPresetProfile",
    "PresetLicenseStatus",
    "PresetLicenseState",
    "MarketplaceCatalog",
    "MarketplaceCatalogError",
    "MarketplacePreset",
    "MarketplaceAuthor",
    "MarketplacePresetInstaller",
    "MarketplaceInstallResult",
    "load_catalog",
    "StrategyParameterTester",
    "StrategyParameterTestReport",
    "ParameterTestResult",
    "StrategyRegimeWorkflow",
    "ActivationHistoryStats",
    "ActivationTransitionStats",
    "ActivationCadenceStats",
    "ActivationUptimeStats",
    "PresetVersionInfo",
    "RegimePresetActivation",
    "DEFAULT_STRATEGY_CATALOG",
    "GridTradingSettings",
    "GridTradingStrategy",
    "AdaptiveMarketMakingSettings",
    "AdaptiveMarketMakingStrategy",
    "MarketMakingSettings",
    "MarketMakingStrategy",
    "MeanReversionSettings",
    "MeanReversionStrategy",
    "OptionsIncomeSettings",
    "OptionsIncomeStrategy",
    "ScalpingSettings",
    "ScalpingStrategy",
    "StatisticalArbitrageSettings",
    "StatisticalArbitrageStrategy",
    "TriangularArbitrageSettings",
    "TriangularArbitrageStrategy",
    "VolatilityTargetSettings",
    "VolatilityTargetStrategy",
    "CrossExchangeArbitrageSettings",
    "CrossExchangeArbitrageStrategy",
    "FuturesSpreadSettings",
    "FuturesSpreadStrategy",
    "CrossExchangeHedgeSettings",
    "CrossExchangeHedgeStrategy",
    "StrategyDescriptor",
    "list_available_strategies",
    "describe_supported_strategies",
    "supported_strategy_keys",
]
