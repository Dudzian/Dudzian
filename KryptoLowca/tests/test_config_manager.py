"""Testy menedżera konfiguracji opartego na nowym module ``KryptoLowca.config_manager``."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import time
from pathlib import Path

import pytest

from bot_core.runtime.metadata import RiskManagerSettings, derive_risk_manager_settings

from KryptoLowca.config_manager import ConfigManager, StrategyConfig, ValidationError


@pytest.fixture()
def config_path(tmp_path: Path) -> Path:
    return tmp_path / "kl_config.yaml"


@pytest.fixture()
def cfg(config_path: Path) -> ConfigManager:
    return ConfigManager(config_path)


def test_save_and_load_roundtrip(cfg: ConfigManager, config_path: Path) -> None:
    strategy = StrategyConfig(preset="SAFE").validate()
    trade = {"symbols": ["BTC/USDT", "ETH/USDT"], "max_open_positions": 2}
    payload = {"strategy": asdict(strategy), "trade": trade, "meta": {}}

    saved = asyncio.run(
        cfg.save_config(payload, actor="tester", preset_id="SAFE", note="baseline")
    )
    assert config_path.exists()
    history = cfg.get_preset_history("SAFE")
    assert history and history[0]["note"] == "baseline"

    loaded = asyncio.run(cfg.load_config())
    assert loaded["strategy"]["preset"] == "SAFE"
    assert loaded["trade"]["max_open_positions"] == trade["max_open_positions"]
    assert loaded["trade"]["symbols"] == trade["symbols"]


def test_strategy_validation_rejects_invalid_notional(cfg: ConfigManager) -> None:
    strategy = StrategyConfig(max_position_notional_pct=1.5)
    with pytest.raises(ValidationError):
        strategy.validate()


def test_load_config_initialises_defaults(cfg: ConfigManager) -> None:
    loaded = asyncio.run(cfg.load_config())
    assert loaded["strategy"]["mode"] == "demo"
    assert loaded["trade"]["max_open_positions"] == 3
    assert loaded["meta"] == {}


def test_live_mode_requires_backtest_confirmation(cfg: ConfigManager) -> None:
    strategy = StrategyConfig(preset="SAFE", mode="live")
    config = {"strategy": asdict(strategy), "trade": {}, "meta": {}}
    with pytest.raises(ValidationError):
        asyncio.run(cfg.save_config(config, actor="tester", preset_id="SAFE"))


def test_record_history_creates_versions(cfg: ConfigManager) -> None:
    strategy = StrategyConfig(preset="SAFE").validate()
    base_payload = {"strategy": asdict(strategy), "trade": {}, "meta": {}}

    asyncio.run(cfg.save_config(base_payload, actor="tester", preset_id="SAFE", note="v1"))
    # świeży backtest dla trybu live
    updated = asdict(strategy.mark_backtest_passed())
    updated["mode"] = "live"
    updated["backtest_passed_at"] = time.time()
    asyncio.run(
        cfg.save_config(
            {"strategy": updated, "trade": {}, "meta": {}},
            actor="tester",
            preset_id="SAFE",
            note="v2",
        )
    )

    history = cfg.get_preset_history("SAFE")
    assert len(history) >= 2
    assert history[0]["note"] == "v2"
    assert history[1]["note"] == "v1"


def test_core_config_derives_risk_manager_settings(core_config) -> None:  # type: ignore[annotation-unchecked]
    profile = core_config.risk_profiles["balanced"]

    settings = derive_risk_manager_settings(profile, profile_name=profile.name)

    assert isinstance(settings, RiskManagerSettings)
    assert settings.max_risk_per_trade == pytest.approx(profile.max_position_pct)
    assert settings.max_daily_loss_pct == pytest.approx(profile.max_daily_loss_pct)
    assert settings.max_positions == profile.max_open_positions
    assert settings.emergency_stop_drawdown == pytest.approx(profile.hard_drawdown_pct)
    assert settings.target_volatility == pytest.approx(profile.target_volatility)
    assert settings.profile_name == profile.name
    assert settings.confidence_level is not None
    assert settings.max_portfolio_risk >= settings.max_risk_per_trade


def test_derive_risk_manager_settings_honours_defaults(core_config) -> None:  # type: ignore[annotation-unchecked]
    defaults = RiskManagerSettings(
        max_risk_per_trade=0.01,
        max_daily_loss_pct=0.03,
        max_portfolio_risk=0.2,
        max_positions=4,
        emergency_stop_drawdown=0.09,
        confidence_level=0.8,
        target_volatility=0.15,
        profile_name="defaults",
    )

    overrides = {"max_position_pct": 0.0, "max_daily_loss_pct": 0.0}

    settings = derive_risk_manager_settings(overrides, defaults=defaults, profile_name="custom")

    assert settings.max_risk_per_trade == pytest.approx(defaults.max_risk_per_trade)
    assert settings.max_daily_loss_pct == pytest.approx(defaults.max_daily_loss_pct)
    assert settings.max_positions == defaults.max_positions
    assert settings.emergency_stop_drawdown == pytest.approx(defaults.emergency_stop_drawdown)
    assert settings.target_volatility == pytest.approx(defaults.target_volatility)
    assert settings.confidence_level is not None
    assert settings.confidence_level >= defaults.confidence_level
    assert settings.confidence_level <= 0.99
    assert settings.profile_name == "custom"


def test_core_config_runtime_entrypoint_maps_environment(core_config) -> None:  # type: ignore[annotation-unchecked]
    entry = core_config.runtime_entrypoints["paper_runtime"]

    assert entry.environment == "paper"
    assert entry.risk_profile == "balanced"
    assert entry.controller == "default_controller"
