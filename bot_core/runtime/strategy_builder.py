"""Budowanie i opis strategii z konfiguracji core."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from bot_core.config.loader import load_core_config
from bot_core.config.models import CoreConfig
from bot_core.security.guards import get_capability_guard
from bot_core.strategies.catalog import DEFAULT_STRATEGY_CATALOG, StrategyCatalog, StrategyDefinition
from bot_core.strategies.base import StrategyEngine


def _collect_strategy_definitions(core_config: CoreConfig) -> dict[str, StrategyDefinition]:
    definitions: dict[str, StrategyDefinition] = {}

    def _resolve_metadata(
        engine: str,
    ) -> tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...], str | None]:
        try:
            spec = DEFAULT_STRATEGY_CATALOG.get(engine)
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
            },
        )
    for name, cfg in getattr(core_config, "volatility_targeting_strategies", {}).items():
        _fallback(
            name,
            "volatility_targeting",
            {
                "target_volatility": cfg.target_volatility,
                "lookback": cfg.lookback,
                "min_leverage": cfg.min_leverage,
                "max_leverage": cfg.max_leverage,
            },
        )
    for name, cfg in getattr(core_config, "cross_exchange_arbitrage_strategies", {}).items():
        _fallback(
            name,
            "cross_exchange_arbitrage",
            {
                "leg_size": cfg.leg_size,
                "max_slippage": cfg.max_slippage,
                "min_spread": cfg.min_spread,
                "max_open_positions": cfg.max_open_positions,
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


def instantiate_strategies(
    core_config: CoreConfig, *, catalog: StrategyCatalog | None = None
) -> dict[str, StrategyEngine]:
    catalog = catalog or DEFAULT_STRATEGY_CATALOG
    registry: dict[str, StrategyEngine] = {}
    guard = get_capability_guard()

    definitions = _collect_strategy_definitions(core_config)

    for name, definition in definitions.items():
        spec = catalog.get(definition.engine)
        if guard is not None and spec.capability:
            guard.require_strategy(
                spec.capability,
                message=(
                    f"Strategia '{name}' wymaga aktywnej licencji "
                    f"{spec.capability}."
                ),
            )
        registry[name] = catalog.create(definition)

    return registry


def describe_strategy_definitions(
    core_config: CoreConfig,
    *,
    catalog: StrategyCatalog | None = None,
) -> Sequence[Mapping[str, object]]:
    resolved_catalog = catalog or DEFAULT_STRATEGY_CATALOG
    definitions = _collect_strategy_definitions(core_config)
    described = resolved_catalog.describe_definitions(definitions, include_metadata=True)
    return list(described)


def describe_multi_strategy_configuration(
    *,
    config_path: str | Path,
    scheduler_name: str | None = None,
    catalog: StrategyCatalog | None = None,
    include_strategy_definitions: bool = True,
    only_scheduler_definitions: bool = False,
) -> Mapping[str, object]:
    resolved_catalog = catalog or DEFAULT_STRATEGY_CATALOG
    core_config = load_core_config(config_path)
    scheduler_configs = getattr(core_config, "multi_strategy_schedulers", {})
    if not scheduler_configs:
        raise ValueError("Konfiguracja nie zawiera sekcji multi_strategy_schedulers.")

    resolved_name = scheduler_name or next(iter(scheduler_configs))
    scheduler_cfg = scheduler_configs.get(resolved_name)
    if scheduler_cfg is None:
        raise KeyError(f"Nie znaleziono scheduler-a '{resolved_name}'.")

    definitions = _collect_strategy_definitions(core_config)
    guard = get_capability_guard()
    schedules: list[dict[str, object]] = []
    blocked_schedules: list[str] = []
    blocked_strategies: list[str] = []
    blocked_strategy_capabilities: dict[str, str] = {}
    blocked_schedule_capabilities: dict[str, str] = {}
    strategy_capabilities: dict[str, str] = {}

    for schedule in scheduler_cfg.schedules:
        strategy_name = schedule.strategy
        definition = definitions.get(strategy_name)
        if definition is None:
            blocked_schedules.append(schedule.name)
            continue
        spec = resolved_catalog.get(definition.engine)
        capability = spec.capability
        if capability:
            strategy_capabilities[strategy_name] = capability
        if guard is not None and capability:
            try:
                guard.require_strategy(capability)
            except Exception:
                blocked_schedules.append(schedule.name)
                blocked_schedule_capabilities[schedule.name] = capability
                blocked_strategy_capabilities[strategy_name] = capability
                blocked_strategies.append(strategy_name)
                continue
        schedules.append(
            {
                "name": schedule.name,
                "strategy": strategy_name,
                "risk_profile": schedule.risk_profile,
                "cadence_seconds": schedule.cadence_seconds,
                "interval": schedule.interval,
                "max_signals": schedule.max_signals,
                "tags": list(definition.tags),
            }
        )

    result: dict[str, object] = {
        "name": scheduler_cfg.name,
        "environment": scheduler_cfg.environment,
        "portfolio": getattr(scheduler_cfg, "portfolio_id", None),
        "schedules": schedules,
        "blocked_schedules": blocked_schedules,
        "blocked_strategies": blocked_strategies,
        "strategy_capabilities": strategy_capabilities,
        "blocked_strategy_capabilities": blocked_strategy_capabilities,
        "blocked_schedule_capabilities": blocked_schedule_capabilities,
    }
    if include_strategy_definitions and not only_scheduler_definitions:
        result["strategies"] = resolved_catalog.describe_definitions(
            definitions,
            include_metadata=True,
        )
    return result

