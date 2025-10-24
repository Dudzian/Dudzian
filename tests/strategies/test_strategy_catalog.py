"""Testy katalogu strategii Multi-Strategy."""
from __future__ import annotations

from datetime import date

import pytest

from bot_core.config.loader import (
    _load_day_trading_strategies,
    _load_options_income_strategies,
    _load_scalping_strategies,
    _load_statistical_arbitrage_strategies,
    _load_strategy_definitions,
)
from bot_core.config.models import (
    CoreConfig,
    EnvironmentConfig,
    RiskProfileConfig,
    ScalpingStrategyConfig,
    StrategyDefinitionConfig,
)
from bot_core.exchanges.base import Environment
from bot_core.strategies.base import StrategyEngine
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyDefinition,
    StrategyPresetWizard,
)
from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)


def _activate_guard(strategies: dict[str, bool]) -> None:
    capabilities = build_capabilities_from_payload(
        {
            "edition": "pro",
            "environments": ["paper"],
            "exchanges": {},
            "strategies": strategies,
            "runtime": {},
            "modules": {},
            "limits": {},
        },
        effective_date=date(2025, 1, 1),
    )
    install_capability_guard(capabilities)
from bot_core.runtime.pipeline import _collect_strategy_definitions


def test_catalog_builds_trend_strategy() -> None:
    definition = StrategyDefinition(
        name="demo_trend",
        engine="daily_trend_momentum",
        license_tier="standard",
        risk_classes=("directional",),
        required_data=("ohlcv",),
        parameters={
            "fast_ma": 5,
            "slow_ma": 20,
            "breakout_lookback": 10,
            "momentum_window": 10,
            "atr_window": 14,
            "atr_multiplier": 1.5,
        },
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    assert isinstance(engine, StrategyEngine)
    metadata = getattr(engine, "metadata", {})
    assert metadata.get("tags")
    assert metadata.get("license_tier") == "standard"
    assert metadata.get("risk_classes") == ("directional", "momentum")
    assert metadata.get("required_data") == ("ohlcv", "technical_indicators")
    assert metadata.get("capability") == "trend_d1"


def test_strategy_definition_requires_metadata() -> None:
    with pytest.raises(ValueError):
        StrategyDefinition(
            name="invalid",
            engine="daily_trend_momentum",
            license_tier="",
            risk_classes=(),
            required_data=(),
        )


def test_catalog_unknown_engine() -> None:
    definition = StrategyDefinition(
        name="unknown",
        engine="non_existing",
        license_tier="unspecified",
        risk_classes=("unspecified",),
        required_data=("unspecified",),
        parameters={},
    )
    with pytest.raises(KeyError):
        DEFAULT_STRATEGY_CATALOG.create(definition)


def test_catalog_rejects_mismatched_license() -> None:
    definition = StrategyDefinition(
        name="mismatch",
        engine="daily_trend_momentum",
        license_tier="enterprise",
        risk_classes=("directional",),
        required_data=("ohlcv",),
        parameters={},
    )
    with pytest.raises(ValueError):
        DEFAULT_STRATEGY_CATALOG.create(definition)


def test_preset_wizard_propagates_metadata() -> None:
    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    preset = wizard.build_preset(
        "demo",
        [
            {
                "engine": "daily_trend_momentum",
                "name": "trend-entry",
                "risk_classes": ["swing"],
                "required_data": ["custom_indicator"],
            }
        ],
    )
    entry = preset["strategies"][0]
    assert entry["license_tier"] == "standard"
    assert entry["risk_classes"] == ["directional", "momentum", "swing"]
    assert entry["required_data"] == [
        "ohlcv",
        "technical_indicators",
        "custom_indicator",
    ]
    assert entry["metadata"]["license_tier"] == "standard"
    assert entry["capability"] == "trend_d1"
    assert entry["metadata"]["capability"] == "trend_d1"
    assert entry["metadata"]["tags"] == ("trend", "momentum")


def test_preset_wizard_respects_capability_guard() -> None:
    wizard = StrategyPresetWizard(DEFAULT_STRATEGY_CATALOG)
    try:
        _activate_guard({"trend_d1": True, "scalping": False})
        with pytest.raises(LicenseCapabilityError):
            wizard.build_preset(
                "blocked",
                [
                    {
                        "engine": "scalping",
                        "name": "blocked-entry",
                    }
                ],
            )
    finally:
        reset_capability_guard()


def test_catalog_describe_engines_includes_metadata() -> None:
    summary = DEFAULT_STRATEGY_CATALOG.describe_engines()
    assert summary
    first = summary[0]
    assert "license_tier" in first
    assert "risk_classes" in first and first["risk_classes"]
    assert "required_data" in first and first["required_data"]


def test_catalog_describe_engines_filters_by_capability_guard() -> None:
    try:
        _activate_guard(
            {
                "trend_d1": True,
                "day_trading": True,
                "scalping": False,
                "options_income": False,
                "stat_arbitrage": False,
            }
        )
        summary = DEFAULT_STRATEGY_CATALOG.describe_engines()
        engines = {entry["engine"] for entry in summary}
        assert "daily_trend_momentum" in engines
        assert "day_trading" in engines
        assert "scalping" not in engines
        assert "options_income" not in engines
        assert "statistical_arbitrage" not in engines
    finally:
        reset_capability_guard()


def test_catalog_describe_definitions_merges_metadata() -> None:
    definition = StrategyDefinition(
        name="custom_trend",
        engine="daily_trend_momentum",
        license_tier="standard",
        risk_classes=("custom",),
        required_data=("ohlcv", "alt_feed"),
        parameters={"fast_ma": 10},
        metadata={"note": "extra"},
    )
    summary = DEFAULT_STRATEGY_CATALOG.describe_definitions(
        {definition.name: definition}, include_metadata=True
    )
    assert len(summary) == 1
    payload = summary[0]
    assert payload["license_tier"] == "standard"
    assert payload["risk_classes"] == ["directional", "momentum", "custom"]
    assert payload["required_data"] == [
        "ohlcv",
        "technical_indicators",
        "alt_feed",
    ]
    assert payload["capability"] == "trend_d1"
    assert payload["metadata"]["note"] == "extra"


def test_catalog_describe_definitions_skips_blocked_capabilities() -> None:
    definition = StrategyDefinition(
        name="blocked_scalp",
        engine="scalping",
        license_tier="professional",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={},
    )
    try:
        _activate_guard({"trend_d1": True, "scalping": False})
        summary = DEFAULT_STRATEGY_CATALOG.describe_definitions(
            {definition.name: definition}
        )
        assert summary == []
    finally:
        reset_capability_guard()


def test_loader_backfills_metadata_for_legacy_definitions() -> None:
    raw = {
        "strategies": {
            "legacy_trend": {
                "engine": "daily_trend_momentum",
                "parameters": {"fast_ma": 15},
            }
        }
    }
    result = _load_strategy_definitions(raw)
    cfg = result["legacy_trend"]
    assert cfg.license_tier == "standard"
    assert cfg.risk_classes == ("directional", "momentum")
    assert cfg.required_data == ("ohlcv", "technical_indicators")
    assert cfg.capability == "trend_d1"
    assert cfg.tags == ("trend", "momentum")
    assert cfg.metadata["tags"] == ("trend", "momentum")


def test_loader_builds_specialized_strategy_configs() -> None:
    raw = {
        "scalping_strategies": {
            "quick_scalp": {"parameters": {"min_price_change": 0.0009, "max_hold_bars": 3}}
        },
        "options_income_strategies": {
            "theta_vault": {"parameters": {"min_iv": 0.55, "max_delta": 0.3}}
        },
        "statistical_arbitrage_strategies": {
            "pairs_alpha": {"parameters": {"lookback": 45, "spread_exit_z": 0.4}}
        },
        "day_trading_strategies": {
            "intraday_momo": {
                "parameters": {
                    "momentum_window": 4,
                    "entry_threshold": 0.8,
                    "bias_strength": 0.1,
                }
            }
        },
    }

    scalping_cfg = _load_scalping_strategies(raw)["quick_scalp"]
    assert scalping_cfg.min_price_change == pytest.approx(0.0009)
    assert scalping_cfg.max_hold_bars == 3

    options_cfg = _load_options_income_strategies(raw)["theta_vault"]
    assert options_cfg.min_iv == pytest.approx(0.55)
    assert options_cfg.max_delta == pytest.approx(0.3)

    stat_cfg = _load_statistical_arbitrage_strategies(raw)["pairs_alpha"]
    assert stat_cfg.lookback == 45
    assert stat_cfg.spread_exit_z == pytest.approx(0.4)

    day_cfg = _load_day_trading_strategies(raw)["intraday_momo"]
    assert day_cfg.momentum_window == 4
    assert day_cfg.entry_threshold == pytest.approx(0.8)
    assert day_cfg.bias_strength == pytest.approx(0.1)


def test_loader_backfills_specialized_definitions_with_metadata() -> None:
    raw = {
        "scalping_strategies": {"quick_scalp": {}},
        "options_income_strategies": {"theta_vault": {}},
        "statistical_arbitrage_strategies": {"pairs_alpha": {}},
        "day_trading_strategies": {"intraday_momo": {}},
    }

    result = _load_strategy_definitions(raw)

    scalping = result["quick_scalp"]
    assert scalping.engine == "scalping"
    assert scalping.license_tier == "professional"
    assert scalping.risk_classes == ("intraday", "scalping")
    assert scalping.required_data == ("ohlcv", "order_book")
    assert scalping.capability == "scalping"
    assert scalping.tags == ("intraday", "scalping")
    assert scalping.metadata["tags"] == ("intraday", "scalping")

    options = result["theta_vault"]
    assert options.engine == "options_income"
    assert options.license_tier == "enterprise"
    assert options.risk_classes == ("derivatives", "income")
    assert options.required_data == ("options_chain", "greeks", "ohlcv")
    assert options.capability == "options_income"
    assert options.tags == ("options", "income")
    assert options.metadata["tags"] == ("options", "income")

    stat = result["pairs_alpha"]
    assert stat.engine == "statistical_arbitrage"
    assert stat.license_tier == "professional"
    assert stat.risk_classes == ("statistical", "mean_reversion")
    assert stat.required_data == ("ohlcv", "spread_history")
    assert stat.capability == "stat_arbitrage"
    assert stat.tags == ("stat_arbitrage", "pairs_trading")
    assert stat.metadata["tags"] == ("stat_arbitrage", "pairs_trading")

    day = result["intraday_momo"]
    assert day.engine == "day_trading"
    assert day.license_tier == "standard"
    assert day.risk_classes == ("intraday", "momentum")
    assert day.required_data == ("ohlcv", "technical_indicators")
    assert day.capability == "day_trading"
    assert day.tags == ("intraday", "momentum")
    assert day.metadata["tags"] == ("intraday", "momentum")


def test_collect_strategy_definitions_merges_default_tags() -> None:
    env_cfg = EnvironmentConfig(
        name="paper",
        exchange="demo",
        environment=Environment.PAPER,
        keychain_key="paper-key",
        data_cache_path="/tmp/cache",
        risk_profile="balanced",
        alert_channels=("email",),
    )
    risk_cfg = RiskProfileConfig(
        name="balanced",
        max_daily_loss_pct=0.1,
        max_position_pct=0.2,
        target_volatility=0.15,
        max_leverage=2.0,
        stop_loss_atr_multiple=1.5,
        max_open_positions=5,
        hard_drawdown_pct=0.25,
    )
    scalping_cfg = ScalpingStrategyConfig(
        name="scalp-alpha",
        min_price_change=0.0008,
        take_profit=0.0015,
        stop_loss=0.0007,
        max_hold_bars=4,
    )
    custom_definition = StrategyDefinitionConfig(
        name="custom_scalp",
        engine="scalping",
        parameters={},
        license_tier=None,
        risk_classes=(),
        required_data=(),
        capability=None,
        risk_profile=None,
        tags=("custom",),
        metadata={},
    )
    core_config = CoreConfig(
        environments={"paper": env_cfg},
        risk_profiles={"balanced": risk_cfg},
        scalping_strategies={"scalp-alpha": scalping_cfg},
        strategy_definitions={"custom_scalp": custom_definition},
    )

    definitions = _collect_strategy_definitions(core_config)

    fallback = definitions["scalp-alpha"]
    assert fallback.tags == ("intraday", "scalping")
    assert fallback.metadata["tags"] == ("intraday", "scalping")
    assert fallback.metadata["capability"] == "scalping"

    custom = definitions["custom_scalp"]
    assert custom.tags == ("intraday", "scalping", "custom")
    assert custom.metadata["tags"] == ("intraday", "scalping", "custom")
    assert custom.metadata["capability"] == "scalping"


def test_catalog_builds_scalping_strategy() -> None:
    definition = StrategyDefinition(
        name="scalp-alpha",
        engine="scalping",
        license_tier="professional",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={"min_price_change": 0.0008, "take_profit": 0.0015},
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    assert isinstance(engine, StrategyEngine)
    metadata = getattr(engine, "metadata", {})
    assert metadata["capability"] == "scalping"
    assert metadata["required_data"] == ("ohlcv", "order_book")
    assert metadata["risk_classes"] == ("intraday", "scalping")
    assert metadata["tags"] == ("intraday", "scalping")


def test_catalog_create_blocks_capability_without_license() -> None:
    definition = StrategyDefinition(
        name="blocked-scalp",
        engine="scalping",
        license_tier="professional",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={},
    )
    try:
        _activate_guard({"trend_d1": True, "scalping": False})
        with pytest.raises(LicenseCapabilityError):
            DEFAULT_STRATEGY_CATALOG.create(definition)
    finally:
        reset_capability_guard()


def test_catalog_builds_options_strategy() -> None:
    definition = StrategyDefinition(
        name="options-income",
        engine="options_income",
        license_tier="enterprise",
        risk_classes=("income",),
        required_data=("options_chain",),
        parameters={"min_iv": 0.4, "max_delta": 0.3, "min_days_to_expiry": 10},
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    metadata = getattr(engine, "metadata", {})
    assert metadata["capability"] == "options_income"
    assert metadata["required_data"] == ("options_chain", "greeks", "ohlcv")
    assert metadata["risk_classes"] == ("derivatives", "income")


def test_catalog_builds_statistical_arbitrage_strategy() -> None:
    definition = StrategyDefinition(
        name="pairs-core",
        engine="statistical_arbitrage",
        license_tier="professional",
        risk_classes=("statistical",),
        required_data=("ohlcv",),
        parameters={"lookback": 40, "spread_entry_z": 2.5, "spread_exit_z": 0.75},
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    metadata = getattr(engine, "metadata", {})
    assert metadata["capability"] == "stat_arbitrage"
    assert metadata["risk_classes"] == ("statistical", "mean_reversion")


def test_catalog_builds_day_trading_strategy() -> None:
    definition = StrategyDefinition(
        name="day-momentum",
        engine="day_trading",
        license_tier="standard",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={"momentum_window": 4, "volatility_window": 6},
    )
    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    metadata = getattr(engine, "metadata", {})
    assert metadata["capability"] == "day_trading"
    assert metadata["risk_classes"] == ("intraday", "momentum")
    assert metadata["required_data"] == ("ohlcv", "technical_indicators")


def test_catalog_rejects_invalid_scalping_parameter() -> None:
    definition = StrategyDefinition(
        name="invalid-scalper",
        engine="scalping",
        license_tier="professional",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={"min_price_change": 1.5},
    )
    with pytest.raises(ValueError):
        DEFAULT_STRATEGY_CATALOG.create(definition)


def test_catalog_rejects_invalid_day_trading_thresholds() -> None:
    definition = StrategyDefinition(
        name="invalid-day-trader",
        engine="day_trading",
        license_tier="standard",
        risk_classes=("intraday",),
        required_data=("ohlcv",),
        parameters={"entry_threshold": 0.2, "exit_threshold": 0.3},
    )
    with pytest.raises(ValueError):
        DEFAULT_STRATEGY_CATALOG.create(definition)


def test_catalog_rejects_invalid_options_parameters() -> None:
    definition = StrategyDefinition(
        name="invalid-options",
        engine="options_income",
        license_tier="enterprise",
        risk_classes=("income",),
        required_data=("options_chain",),
        parameters={"max_delta": 1.5},
    )
    with pytest.raises(ValueError):
        DEFAULT_STRATEGY_CATALOG.create(definition)
