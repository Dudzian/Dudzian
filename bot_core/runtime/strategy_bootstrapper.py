"""Bootstrap warstwy strategii dla runtime pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from bot_core.config.models import CoreConfig, StrategyScheduleConfig
from bot_core.security.guards import get_capability_guard
from bot_core.strategies.base import StrategyEngine
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyCatalog,
    StrategyDefinition,
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StrategyBootstrapResult:
    """Wynik bootstrapu strategii dla scheduler-a."""

    definitions: Mapping[str, StrategyDefinition]
    strategies: Mapping[str, StrategyEngine]


class StrategyBootstrapper:
    """Ładuje, mapuje i waliduje strategie używane przez runtime pipeline."""

    def __init__(self, *, catalog: StrategyCatalog | None = None) -> None:
        self._catalog = catalog or DEFAULT_STRATEGY_CATALOG

    def bootstrap(self, core_config: CoreConfig) -> StrategyBootstrapResult:
        definitions = self.collect_definitions(core_config)
        strategies = self.instantiate(definitions)
        return StrategyBootstrapResult(definitions=definitions, strategies=strategies)

    def collect_definitions(self, core_config: CoreConfig) -> dict[str, StrategyDefinition]:
        """Zbiera definicje strategii z konfiguracji core."""

        definitions: dict[str, StrategyDefinition] = {}

        def _resolve_metadata(
            engine: str,
        ) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str | None]:
            try:
                spec = self._catalog.get(engine)
            except KeyError:
                return (
                    "unspecified",
                    ("unspecified",),
                    ("unspecified",),
                    (),
                    (),
                    None,
                )
            return (
                spec.license_tier,
                spec.risk_classes,
                spec.required_data,
                spec.risk_hooks,
                spec.default_tags,
                spec.capability,
            )

        for name, cfg in getattr(core_config, "strategy_definitions", {}).items():
            (
                license_tier,
                risk_classes,
                required_data,
                risk_hooks,
                default_tags,
                capability,
            ) = _resolve_metadata(cfg.engine)
            metadata = dict(cfg.metadata)
            resolved_capability = getattr(cfg, "capability", None) or capability
            if resolved_capability and "capability" not in metadata:
                metadata["capability"] = resolved_capability
            merged_tags = tuple(dict.fromkeys((*default_tags, *tuple(cfg.tags))))
            if merged_tags and "tags" not in metadata:
                metadata["tags"] = merged_tags
            definitions[name] = StrategyDefinition(
                name=cfg.name,
                engine=cfg.engine,
                license_tier=cfg.license_tier or license_tier,
                risk_classes=tuple(cfg.risk_classes) or risk_classes,
                required_data=tuple(cfg.required_data) or required_data,
                risk_hooks=tuple(getattr(cfg, "risk_hooks", ())) or risk_hooks,
                parameters=dict(cfg.parameters),
                risk_profile=cfg.risk_profile,
                tags=merged_tags,
                metadata=metadata,
            )

        def _fallback(name: str, engine: str, params: Mapping[str, Any]) -> None:
            if name in definitions:
                return
            (
                license_tier,
                risk_classes,
                required_data,
                risk_hooks,
                default_tags,
                capability,
            ) = _resolve_metadata(engine)
            metadata: dict[str, Any] = {}
            if capability:
                metadata["capability"] = capability
            if default_tags:
                metadata["tags"] = default_tags
            definitions[name] = StrategyDefinition(
                name=name,
                engine=engine,
                license_tier=license_tier,
                risk_classes=risk_classes,
                required_data=required_data,
                risk_hooks=risk_hooks,
                parameters=dict(params),
                tags=default_tags,
                metadata=metadata,
            )

        for name, cfg in getattr(core_config, "strategies", {}).items():
            _fallback(
                name,
                "daily_trend_momentum",
                {
                    "fast_ma": cfg.fast_ma,
                    "slow_ma": cfg.slow_ma,
                    "breakout_lookback": cfg.breakout_lookback,
                    "momentum_window": cfg.momentum_window,
                    "atr_window": cfg.atr_window,
                    "atr_multiplier": cfg.atr_multiplier,
                    "min_trend_strength": cfg.min_trend_strength,
                    "min_momentum": cfg.min_momentum,
                },
            )
        for name, cfg in getattr(core_config, "mean_reversion_strategies", {}).items():
            _fallback(
                name,
                "mean_reversion",
                {
                    "lookback": cfg.lookback,
                    "entry_zscore": cfg.entry_zscore,
                    "exit_zscore": cfg.exit_zscore,
                    "max_holding_period": cfg.max_holding_period,
                    "volatility_cap": cfg.volatility_cap,
                    "min_volume_usd": cfg.min_volume_usd,
                },
            )
        for name, cfg in getattr(core_config, "volatility_target_strategies", {}).items():
            _fallback(
                name,
                "volatility_target",
                {
                    "target_volatility": cfg.target_volatility,
                    "lookback": cfg.lookback,
                    "rebalance_threshold": cfg.rebalance_threshold,
                    "min_allocation": cfg.min_allocation,
                    "max_allocation": cfg.max_allocation,
                    "floor_volatility": cfg.floor_volatility,
                },
            )
        for name, cfg in getattr(core_config, "cross_exchange_arbitrage_strategies", {}).items():
            _fallback(
                name,
                "cross_exchange_arbitrage",
                {
                    "primary_exchange": cfg.primary_exchange,
                    "secondary_exchange": cfg.secondary_exchange,
                    "spread_entry": cfg.spread_entry,
                    "spread_exit": cfg.spread_exit,
                    "max_notional": cfg.max_notional,
                    "max_open_seconds": cfg.max_open_seconds,
                },
            )

        for name, cfg in getattr(core_config, "scalping_strategies", {}).items():
            _fallback(
                name,
                "scalping",
                {
                    "min_price_change": cfg.min_price_change,
                    "take_profit": cfg.take_profit,
                    "stop_loss": cfg.stop_loss,
                    "max_hold_bars": cfg.max_hold_bars,
                },
            )

        for name, cfg in getattr(core_config, "options_income_strategies", {}).items():
            _fallback(
                name,
                "options_income",
                {
                    "min_iv": cfg.min_iv,
                    "max_delta": cfg.max_delta,
                    "min_days_to_expiry": cfg.min_days_to_expiry,
                    "roll_threshold_iv": cfg.roll_threshold_iv,
                },
            )

        for name, cfg in getattr(core_config, "statistical_arbitrage_strategies", {}).items():
            _fallback(
                name,
                "statistical_arbitrage",
                {
                    "lookback": cfg.lookback,
                    "spread_entry_z": cfg.spread_entry_z,
                    "spread_exit_z": cfg.spread_exit_z,
                    "max_notional": cfg.max_notional,
                },
            )

        for name, cfg in getattr(core_config, "day_trading_strategies", {}).items():
            _fallback(
                name,
                "day_trading",
                {
                    "momentum_window": cfg.momentum_window,
                    "volatility_window": cfg.volatility_window,
                    "entry_threshold": cfg.entry_threshold,
                    "exit_threshold": cfg.exit_threshold,
                    "take_profit_atr": cfg.take_profit_atr,
                    "stop_loss_atr": cfg.stop_loss_atr,
                    "max_holding_bars": cfg.max_holding_bars,
                    "atr_floor": cfg.atr_floor,
                    "bias_strength": cfg.bias_strength,
                },
            )

        return definitions

    def instantiate(self, definitions: Mapping[str, StrategyDefinition]) -> dict[str, StrategyEngine]:
        registry: dict[str, StrategyEngine] = {}
        guard = get_capability_guard()
        for name, definition in definitions.items():
            try:
                spec = self._catalog.get(definition.engine)
                if guard is not None and spec.capability:
                    guard.require_strategy(
                        spec.capability,
                        message=(
                            f"Strategia '{name}' wymaga aktywnej licencji {spec.capability}."
                        ),
                    )
                registry[name] = self._catalog.create(definition)
            except Exception:
                _LOGGER.exception(
                    "Nie udało się zainicjalizować strategii '%s' (engine=%s)",
                    name,
                    definition.engine,
                )
                raise
        return registry

    @staticmethod
    def validate_schedule_strategies(
        *,
        schedules: Sequence[StrategyScheduleConfig],
        strategies: Mapping[str, StrategyEngine],
    ) -> None:
        for schedule in schedules:
            if schedule.strategy not in strategies:
                raise KeyError(
                    f"Strategia {schedule.strategy} nie została zarejestrowana w konfiguracji"
                )
