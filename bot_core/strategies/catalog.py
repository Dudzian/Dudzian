"""Katalog strategii i wspólne interfejsy fabryk."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Protocol, Sequence

from .base import StrategyEngine
from .cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from .daily_trend import DailyTrendMomentumSettings, DailyTrendMomentumStrategy
from .grid import GridTradingSettings, GridTradingStrategy
from .mean_reversion import MeanReversionSettings, MeanReversionStrategy
from .volatility_target import VolatilityTargetSettings, VolatilityTargetStrategy


class StrategyFactory(Protocol):
    """Fabryka budująca `StrategyEngine` na podstawie parametrów."""

    def __call__(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        ...


@dataclass(slots=True)
class StrategyDefinition:
    """Opis pojedynczej strategii dostępnej w katalogu."""

    name: str
    engine: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    risk_profile: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StrategyEngineSpec:
    """Opis silnika strategii wraz z wymaganą licencją."""

    key: str
    factory: StrategyFactory
    capability: str | None = None
    default_tags: Sequence[str] = field(default_factory=tuple)

    def build(
        self,
        *,
        name: str,
        parameters: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
    ) -> StrategyEngine:
        return self.factory(name=name, parameters=parameters, metadata=metadata)


class StrategyCatalog:
    """Rejestr zarejestrowanych silników strategii."""

    def __init__(self) -> None:
        self._registry: MutableMapping[str, StrategyEngineSpec] = {}

    def register(self, spec: StrategyEngineSpec) -> None:
        key = spec.key.lower()
        self._registry[key] = spec

    def get(self, engine: str) -> StrategyEngineSpec:
        key = engine.lower()
        if key not in self._registry:
            raise KeyError(f"Nie znaleziono silnika strategii: {engine}")
        return self._registry[key]

    def create(self, definition: StrategyDefinition) -> StrategyEngine:
        spec = self.get(definition.engine)
        tags = tuple(dict.fromkeys((*spec.default_tags, *definition.tags)))
        metadata = dict(definition.metadata)
        if tags and "tags" not in metadata:
            metadata["tags"] = tags
        return spec.build(
            name=definition.name,
            parameters=definition.parameters,
            metadata=metadata,
        )


def _build_daily_trend_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = DailyTrendMomentumSettings(
        fast_ma=int(parameters.get("fast_ma", 20)),
        slow_ma=int(parameters.get("slow_ma", 100)),
        breakout_lookback=int(parameters.get("breakout_lookback", 55)),
        momentum_window=int(parameters.get("momentum_window", 20)),
        atr_window=int(parameters.get("atr_window", 14)),
        atr_multiplier=float(parameters.get("atr_multiplier", 2.0)),
        min_trend_strength=float(parameters.get("min_trend_strength", 0.005)),
        min_momentum=float(parameters.get("min_momentum", 0.0)),
    )
    return DailyTrendMomentumStrategy(settings)


def _build_mean_reversion_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = MeanReversionSettings(
        lookback=int(parameters.get("lookback", 96)),
        entry_zscore=float(parameters.get("entry_zscore", 2.0)),
        exit_zscore=float(parameters.get("exit_zscore", 0.5)),
        max_holding_period=int(parameters.get("max_holding_period", 12)),
        volatility_cap=float(parameters.get("volatility_cap", 0.04)),
        min_volume_usd=float(parameters.get("min_volume_usd", 1000.0)),
    )
    return MeanReversionStrategy(settings)


def _build_grid_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = GridTradingSettings(
        grid_size=int(parameters.get("grid_size", 5)),
        grid_spacing=float(parameters.get("grid_spacing", 0.004)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.001)),
        max_inventory=float(parameters.get("max_inventory", 1.0)),
    )
    return GridTradingStrategy(settings)


def _build_volatility_target_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = VolatilityTargetSettings(
        target_volatility=float(parameters.get("target_volatility", 0.1)),
        lookback=int(parameters.get("lookback", 60)),
        rebalance_threshold=float(parameters.get("rebalance_threshold", 0.1)),
        min_allocation=float(parameters.get("min_allocation", 0.1)),
        max_allocation=float(parameters.get("max_allocation", 1.0)),
        floor_volatility=float(parameters.get("floor_volatility", 0.02)),
    )
    return VolatilityTargetStrategy(settings)


def _build_cross_exchange_strategy(
    *, name: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
) -> StrategyEngine:
    settings = CrossExchangeArbitrageSettings(
        primary_exchange=str(parameters.get("primary_exchange", "")),
        secondary_exchange=str(parameters.get("secondary_exchange", "")),
        spread_entry=float(parameters.get("spread_entry", 0.0015)),
        spread_exit=float(parameters.get("spread_exit", 0.0005)),
        max_notional=float(parameters.get("max_notional", 50_000.0)),
        max_open_seconds=int(parameters.get("max_open_seconds", 120)),
    )
    return CrossExchangeArbitrageStrategy(settings)


def build_default_catalog() -> StrategyCatalog:
    catalog = StrategyCatalog()
    catalog.register(
        StrategyEngineSpec(
            key="daily_trend_momentum",
            factory=_build_daily_trend_strategy,
            capability="trend_d1",
            default_tags=("trend", "momentum"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="mean_reversion",
            factory=_build_mean_reversion_strategy,
            capability="mean_reversion",
            default_tags=("mean_reversion", "stat_arbitrage"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="grid_trading",
            factory=_build_grid_strategy,
            capability="grid_trading",
            default_tags=("grid", "market_making"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="volatility_target",
            factory=_build_volatility_target_strategy,
            capability="volatility_target",
            default_tags=("volatility", "risk"),
        )
    )
    catalog.register(
        StrategyEngineSpec(
            key="cross_exchange_arbitrage",
            factory=_build_cross_exchange_strategy,
            capability="cross_exchange",
            default_tags=("arbitrage", "liquidity"),
        )
    )
    return catalog


DEFAULT_STRATEGY_CATALOG = build_default_catalog()


__all__ = [
    "StrategyCatalog",
    "StrategyDefinition",
    "StrategyEngineSpec",
    "StrategyFactory",
    "DEFAULT_STRATEGY_CATALOG",
    "build_default_catalog",
]
