from __future__ import annotations

from dataclasses import replace

import pytest

from bot_core.config.models import (
    CoreConfig,
    EnvironmentConfig,
    RiskProfileConfig,
    TelegramChannelSettings,
)
from bot_core.config.validation import (
    ConfigValidationError,
    assert_core_config_valid,
    validate_core_config,
)
from bot_core.exchanges.base import Environment


@pytest.fixture()
def base_config() -> CoreConfig:
    risk = RiskProfileConfig(
        name="balanced",
        max_daily_loss_pct=0.01,
        max_position_pct=0.03,
        target_volatility=0.07,
        max_leverage=2.0,
        stop_loss_atr_multiple=1.0,
        max_open_positions=3,
        hard_drawdown_pct=0.05,
    )
    environment = EnvironmentConfig(
        name="paper",
        exchange="binance_spot",
        environment=Environment.PAPER,
        keychain_key="binance_paper",
        data_cache_path="/tmp/cache",
        risk_profile="balanced",
        alert_channels=("telegram:primary",),
        ip_allowlist=(),
        credential_purpose="trading",
        instrument_universe=None,
        adapter_settings={},
        required_permissions=("read", "trade"),
        forbidden_permissions=("withdraw",),
    )
    telegram = TelegramChannelSettings(
        name="primary",
        chat_id="123",
        token_secret="telegram_primary_token",
        parse_mode="MarkdownV2",
    )
    return CoreConfig(
        environments={"paper": environment},
        risk_profiles={"balanced": risk},
        instrument_universes={},
        strategies={},
        reporting=None,
        sms_providers={},
        telegram_channels={"primary": telegram},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
        runtime_controllers={},
    )


def test_validate_core_config_accepts_valid_configuration(base_config: CoreConfig) -> None:
    result = validate_core_config(base_config)
    assert result.is_valid()
    assert result.errors == []


def test_validate_core_config_detects_missing_risk_profile(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], risk_profile="unknown")
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "profil ryzyka 'unknown'" in result.errors[0]


def test_validate_core_config_detects_unknown_alert_channel(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], alert_channels=("telegram:missing",))
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "kanał alertowy 'telegram:missing'" in result.errors[0]


def test_validate_core_config_detects_overlapping_permissions(base_config: CoreConfig) -> None:
    invalid_env = replace(
        base_config.environments["paper"],
        required_permissions=("read",),
        forbidden_permissions=("read",),
    )
    config = replace(base_config, environments={"paper": invalid_env})

    with pytest.raises(ConfigValidationError):
        assert_core_config_valid(config)


def test_validate_core_config_detects_negative_risk_values(base_config: CoreConfig) -> None:
    broken_risk = replace(base_config.risk_profiles["balanced"], max_daily_loss_pct=-0.1)
    config = replace(base_config, risk_profiles={"balanced": broken_risk})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "max_daily_loss_pct" in result.errors[0]
