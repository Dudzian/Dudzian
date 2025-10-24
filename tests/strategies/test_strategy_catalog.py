"""Testy katalogu strategii Multi-Strategy."""
from __future__ import annotations

from datetime import date

import pytest

from bot_core.config.loader import _load_strategy_definitions

from bot_core.security.capabilities import build_capabilities_from_payload
from bot_core.security.guards import (
    LicenseCapabilityError,
    install_capability_guard,
    reset_capability_guard,
)
from bot_core.strategies.base import StrategyEngine
from bot_core.strategies.catalog import (
    DEFAULT_STRATEGY_CATALOG,
    StrategyDefinition,
    StrategyPresetWizard,
)


@pytest.fixture(autouse=True)
def _reset_guard() -> None:
    reset_capability_guard()
    yield
    reset_capability_guard()


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


def test_catalog_respects_license_guard() -> None:
    payload = {
        "edition": "standard",
        "environments": ["paper"],
        "strategies": {"volatility_target": True},
        "runtime": {},
        "modules": {},
        "exchanges": {},
        "limits": {},
    }
    guard = build_capabilities_from_payload(payload, effective_date=date(2025, 1, 1))
    install_capability_guard(guard)

    definition = StrategyDefinition(
        name="volatility", 
        engine="volatility_target",
        license_tier="enterprise",
        risk_classes=("risk_control",),
        required_data=("ohlcv", "realized_volatility"),
        parameters={"target_volatility": 0.15},
    )
    with pytest.raises(LicenseCapabilityError):
        DEFAULT_STRATEGY_CATALOG.create(definition)

    reset_capability_guard()
    upgraded_payload = dict(payload, edition="commercial")
    upgraded_guard = build_capabilities_from_payload(
        upgraded_payload, effective_date=date(2025, 1, 1)
    )
    install_capability_guard(upgraded_guard)

    engine = DEFAULT_STRATEGY_CATALOG.create(definition)
    assert isinstance(engine, StrategyEngine)


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


def test_catalog_describe_engines_includes_metadata() -> None:
    summary = DEFAULT_STRATEGY_CATALOG.describe_engines()
    assert summary
    first = summary[0]
    assert "license_tier" in first
    assert "risk_classes" in first and first["risk_classes"]
    assert "required_data" in first and first["required_data"]


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
