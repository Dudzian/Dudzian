from __future__ import annotations

from dataclasses import replace

import pytest

from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    EnvironmentConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
    DailyTrendMomentumStrategyConfig,
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


def _config_with_universe(base_config: CoreConfig) -> CoreConfig:
    instrument = InstrumentConfig(
        name="BTC_USDT",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("core",),
        exchange_symbols={"binance_spot": "BTCUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=30),),
    )
    universe = InstrumentUniverseConfig(
        name="core",
        description="test universe",
        instruments=(instrument,),
    )
    return replace(base_config, instrument_universes={"core": universe})


def test_validate_core_config_detects_empty_instrument_list(base_config: CoreConfig) -> None:
    empty_universe = InstrumentUniverseConfig(name="core", description="desc", instruments=())
    config = replace(base_config, instrument_universes={"core": empty_universe})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "musi zawierać co najmniej jeden instrument" in result.errors[0]


def test_validate_core_config_detects_missing_exchange_symbols(base_config: CoreConfig) -> None:
    universe_config = _config_with_universe(base_config)
    instrument = replace(
        next(iter(universe_config.instrument_universes["core"].instruments)),
        exchange_symbols={},
    )
    broken_universe = replace(
        universe_config.instrument_universes["core"], instruments=(instrument,)
    )
    config = replace(universe_config, instrument_universes={"core": broken_universe})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "powiązanie giełdowe" in result.errors[0]


def test_validate_core_config_detects_invalid_backfill_window(base_config: CoreConfig) -> None:
    universe_config = _config_with_universe(base_config)
    instrument = replace(
        next(iter(universe_config.instrument_universes["core"].instruments)),
        backfill_windows=(InstrumentBackfillWindow(interval="", lookback_days=-1),),
    )
    broken_universe = replace(
        universe_config.instrument_universes["core"], instruments=(instrument,)
    )
    config = replace(universe_config, instrument_universes={"core": broken_universe})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("backfill" in err for err in result.errors)


def test_validate_core_config_detects_universe_without_exchange_mapping(base_config: CoreConfig) -> None:
    instrument = InstrumentConfig(
        name="BTC_USDT",
        base_asset="BTC",
        quote_asset="USDT",
        categories=("core",),
        exchange_symbols={"kraken_spot": "XBTUSDT"},
        backfill_windows=(InstrumentBackfillWindow(interval="1d", lookback_days=30),),
    )
    universe = InstrumentUniverseConfig(
        name="core",
        description="test universe",
        instruments=(instrument,),
    )
    environment = replace(
        base_config.environments["paper"],
        instrument_universe="core",
    )
    config = replace(
        base_config,
        instrument_universes={"core": universe},
        environments={"paper": environment},
    )

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("nie zawiera powiązań" in err for err in result.errors)


def test_validate_core_config_detects_invalid_strategy_settings(base_config: CoreConfig) -> None:
    strategy = DailyTrendMomentumStrategyConfig(
        name="invalid",
        fast_ma=20,
        slow_ma=10,
        breakout_lookback=5,
        momentum_window=3,
        atr_window=7,
        atr_multiplier=2.0,
        min_trend_strength=0.001,
        min_momentum=0.001,
    )
    config = replace(base_config, strategies={"invalid": strategy})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("fast_ma" in err for err in result.errors)


def test_validate_core_config_detects_invalid_runtime_controller(base_config: CoreConfig) -> None:
    controller = ControllerRuntimeConfig(tick_seconds=0.0, interval=" ")
    config = replace(base_config, runtime_controllers={"bad": controller})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("tick_seconds" in err for err in result.errors)
    assert any("interval" in err for err in result.errors)
