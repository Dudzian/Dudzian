"""Publiczny interfejs wyboru wariantu PortfolioGovernor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Mapping, Sequence, Union

from bot_core.portfolio.asset_governor import (
    AssetPortfolioGovernor,
    AssetPortfolioGovernorConfig,
    PortfolioAdjustment,
    PortfolioAdvisory,
    PortfolioAssetConfig,
    PortfolioDecision,
    PortfolioDriftTolerance,
    PortfolioRiskBudgetConfig,
    PortfolioSloOverrideConfig,
)
from bot_core.portfolio.decision_log import PortfolioDecisionLog
from bot_core.portfolio.io_service import PortfolioIOService
from bot_core.portfolio.strategy_governor import (
    PortfolioGovernorScoringWeights,
    PortfolioGovernorStrategyConfig,
    PortfolioRebalanceDecision,
    StrategyAllocationDecision,
    StrategyMetricsSnapshot,
    StrategyPortfolioGovernor,
    StrategyPortfolioGovernorConfig,
    coerce_strategy_config,
)

PortfolioGovernorVariant = Union[AssetPortfolioGovernor, StrategyPortfolioGovernor]


@dataclass(slots=True)
class PortfolioGovernorBuildConfig:
    """Deklaracja konfiguracji do budowy governora."""

    config: AssetPortfolioGovernorConfig | StrategyPortfolioGovernorConfig | object
    clock: Callable[[], datetime] | None = None
    decision_log: PortfolioDecisionLog | None = None
    io_service: PortfolioIOService | None = None


def _coerce_asset_config(config: object) -> AssetPortfolioGovernorConfig:
    if isinstance(config, AssetPortfolioGovernorConfig):
        return config

    drift_value = getattr(config, "drift_tolerance", None)
    if isinstance(drift_value, Mapping):
        drift_value = PortfolioDriftTolerance(
            absolute=float(drift_value.get("absolute", 0.01)),
            relative=float(drift_value.get("relative", 0.25)),
        )
    if not isinstance(drift_value, PortfolioDriftTolerance):
        drift_value = PortfolioDriftTolerance()

    risk_budgets_value = getattr(config, "risk_budgets", {})
    if isinstance(risk_budgets_value, Mapping):
        risk_budgets = dict(risk_budgets_value)
    else:
        risk_budgets = {}

    slo_overrides_value = getattr(config, "slo_overrides", ())
    if isinstance(slo_overrides_value, Sequence) and not isinstance(
        slo_overrides_value, (str, bytes)
    ):
        slo_overrides = tuple(slo_overrides_value)
    else:
        slo_overrides = tuple()

    risk_overrides_value = getattr(config, "risk_overrides", ())
    if isinstance(risk_overrides_value, Sequence) and not isinstance(
        risk_overrides_value, (str, bytes)
    ):
        risk_overrides = tuple(str(item) for item in risk_overrides_value)
    else:
        risk_overrides = tuple()

    return AssetPortfolioGovernorConfig(
        name=str(getattr(config, "name", getattr(config, "portfolio_id", "portfolio"))),
        portfolio_id=str(getattr(config, "portfolio_id", getattr(config, "name", "portfolio"))),
        drift_tolerance=drift_value,
        rebalance_cooldown_seconds=int(getattr(config, "rebalance_cooldown_seconds", 900)),
        min_rebalance_value=float(getattr(config, "min_rebalance_value", 0.0)),
        min_rebalance_weight=float(getattr(config, "min_rebalance_weight", 0.0)),
        assets=tuple(getattr(config, "assets", ())),
        risk_budgets=risk_budgets,
        risk_overrides=risk_overrides,
        slo_overrides=slo_overrides,
        market_intel_interval=getattr(config, "market_intel_interval", None),
        market_intel_lookback_bars=int(getattr(config, "market_intel_lookback_bars", 168)),
    )


def build_portfolio_governor(
    config: AssetPortfolioGovernorConfig | StrategyPortfolioGovernorConfig | object,
    *,
    clock: Callable[[], datetime] | None = None,
    decision_log: PortfolioDecisionLog | None = None,
    io_service: PortfolioIOService | None = None,
) -> PortfolioGovernorVariant:
    io_layer = io_service or PortfolioIOService(decision_log=decision_log)
    if hasattr(config, "assets"):
        asset_cfg = _coerce_asset_config(config)
        return AssetPortfolioGovernor(asset_cfg, clock=clock, io_service=io_layer)

    strategy_cfg = coerce_strategy_config(config)
    return StrategyPortfolioGovernor(strategy_cfg, clock=clock, io_service=io_layer)


def PortfolioGovernor(
    config: AssetPortfolioGovernorConfig | StrategyPortfolioGovernorConfig | object,
    *,
    clock: Callable[[], datetime] | None = None,
    decision_log: PortfolioDecisionLog | None = None,
    io_service: PortfolioIOService | None = None,
) -> PortfolioGovernorVariant:
    """Zachowuje kompatybilność starego API jako cienki wrapper fabryki."""

    return build_portfolio_governor(
        config,
        clock=clock,
        decision_log=decision_log,
        io_service=io_service,
    )


__all__ = [
    "AssetPortfolioGovernor",
    "AssetPortfolioGovernorConfig",
    "PortfolioAdjustment",
    "PortfolioAdvisory",
    "PortfolioAssetConfig",
    "PortfolioDecision",
    "PortfolioDecisionLog",
    "PortfolioDriftTolerance",
    "PortfolioRiskBudgetConfig",
    "PortfolioSloOverrideConfig",
    "PortfolioGovernorVariant",
    "PortfolioGovernorBuildConfig",
    "build_portfolio_governor",
    "PortfolioGovernor",
    "PortfolioGovernorScoringWeights",
    "PortfolioIOService",
    "PortfolioGovernorStrategyConfig",
    "PortfolioRebalanceDecision",
    "StrategyAllocationDecision",
    "StrategyMetricsSnapshot",
    "StrategyPortfolioGovernor",
    "StrategyPortfolioGovernorConfig",
]
