from __future__ import annotations

import logging
from pathlib import Path

import pytest

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from bot_core.runtime.metadata import (
    RiskManagerSettings,
    derive_risk_manager_settings,
    load_risk_manager_settings,
    load_risk_profile_config,
    load_runtime_entrypoint_metadata,
)
from tests.test_runtime_bootstrap import _BASE_CONFIG


@pytest.fixture
def core_config_path(tmp_path: Path) -> Path:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(_BASE_CONFIG, encoding="utf-8")
    return config_path


def test_load_runtime_entrypoint_metadata_returns_payload(core_config_path: Path) -> None:
    metadata = load_runtime_entrypoint_metadata("auto_trader", config_path=core_config_path)
    assert metadata is not None
    assert metadata.environment == "binance_paper"
    assert metadata.risk_profile == "balanced"
    assert metadata.to_dict()["tags"] == []


def test_load_runtime_entrypoint_metadata_handles_missing_entrypoint(
    core_config_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    metadata = load_runtime_entrypoint_metadata("missing", config_path=core_config_path, logger=logging.getLogger(__name__))
    assert metadata is None
    assert any("missing" in record.getMessage() for record in caplog.records)


def test_load_runtime_entrypoint_metadata_uses_default_path(
    core_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "bot_core.runtime.metadata.resolve_core_config_path",
        lambda: core_config_path,
    )
    metadata = load_runtime_entrypoint_metadata("trading_gui")
    assert metadata is not None
    assert metadata.environment == "binance_paper"


def test_load_risk_profile_config_returns_dataclass(core_config_path: Path) -> None:
    name, profile = load_risk_profile_config("auto_trader", config_path=core_config_path)
    assert name == "balanced"
    assert profile is not None
    assert getattr(profile, "max_position_pct", None) == pytest.approx(0.05)
    assert getattr(profile, "max_daily_loss_pct", None) == pytest.approx(0.015)


def test_load_risk_profile_config_falls_back_to_yaml(
    core_config_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    monkeypatch.setattr("bot_core.runtime.metadata._load_core_config", None)
    name, profile = load_risk_profile_config("auto_trader", config_path=core_config_path, logger=logging.getLogger(__name__))
    assert name == "balanced"
    assert isinstance(profile, dict)
    assert profile["max_position_pct"] == pytest.approx(0.05)
    assert any("Typowany loader" in record.getMessage() for record in caplog.records) is False


def test_load_risk_manager_settings_combines_sources(core_config_path: Path) -> None:
    name, profile, settings = load_risk_manager_settings("auto_trader", config_path=core_config_path)
    assert name == "balanced"
    assert profile is not None
    assert isinstance(settings, RiskManagerSettings)
    assert settings.profile_name == "balanced"
    assert settings.max_risk_per_trade == pytest.approx(0.05)
    assert settings.max_daily_loss_pct == pytest.approx(0.015)


def test_load_risk_manager_settings_handles_yaml_fallback(
    core_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot_core.runtime.metadata._load_core_config", None)
    name, profile, settings = load_risk_manager_settings(
        "auto_trader",
        config_path=core_config_path,
        logger=logging.getLogger(__name__),
    )
    assert name == "balanced"
    assert isinstance(profile, dict)
    assert settings.max_risk_per_trade == pytest.approx(0.05)


def test_derive_risk_manager_settings_default_profile() -> None:
    settings = derive_risk_manager_settings(None)
    assert isinstance(settings, RiskManagerSettings)
    assert settings.max_risk_per_trade == pytest.approx(0.02)
    assert settings.max_daily_loss_pct == pytest.approx(0.10)
    assert settings.max_positions == 10
    assert settings.profile_name is None


def test_derive_risk_manager_settings_applies_profile_mapping() -> None:
    profile_payload = {
        "max_position_pct": 0.07,
        "max_daily_loss_pct": 0.12,
        "max_open_positions": 7,
        "hard_drawdown_pct": 0.25,
        "target_volatility": 0.18,
    }
    settings = derive_risk_manager_settings(profile_payload, profile_name="balanced")

    assert settings.profile_name == "balanced"
    assert settings.max_risk_per_trade == pytest.approx(0.07)
    assert settings.max_daily_loss_pct == pytest.approx(0.12)
    assert settings.max_portfolio_risk >= settings.max_risk_per_trade
    assert settings.max_positions == 7
    assert settings.confidence_level is not None
    assert settings.risk_service_kwargs()["max_position_notional_pct"] == pytest.approx(0.07)
