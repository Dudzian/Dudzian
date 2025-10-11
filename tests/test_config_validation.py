from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.config.models import (
    ControllerRuntimeConfig,
    CoreConfig,
    DailyTrendMomentumStrategyConfig,
    EnvironmentConfig,
    InstrumentBackfillWindow,
    InstrumentConfig,
    InstrumentUniverseConfig,
    MetricsServiceConfig,
    MetricsServiceTlsConfig,
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
    strategy = DailyTrendMomentumStrategyConfig(
        name="core_daily_trend",
        fast_ma=25,
        slow_ma=100,
        breakout_lookback=55,
        momentum_window=20,
        atr_window=14,
        atr_multiplier=2.0,
        min_trend_strength=0.0,
        min_momentum=0.0,
    )
    controller = ControllerRuntimeConfig(tick_seconds=86400, interval="1d")
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
        default_strategy="core_daily_trend",
        default_controller="daily_trend_core",
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
        strategies={"core_daily_trend": strategy},
        reporting=None,
        sms_providers={},
        telegram_channels={"primary": telegram},
        email_channels={},
        signal_channels={},
        whatsapp_channels={},
        messenger_channels={},
        runtime_controllers={"daily_trend_core": controller},
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


def test_validate_core_config_detects_missing_default_strategy(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], default_strategy=None)
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "default_strategy" in result.errors[0]


def test_validate_core_config_detects_unknown_default_strategy(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], default_strategy="missing")
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "domyślna strategia" in result.errors[0]


def test_validate_core_config_detects_missing_default_controller(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], default_controller=None)
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "default_controller" in result.errors[0]


def test_validate_core_config_detects_unknown_default_controller(base_config: CoreConfig) -> None:
    invalid_env = replace(base_config.environments["paper"], default_controller="missing")
    config = replace(base_config, environments={"paper": invalid_env})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert "domyślny kontroler" in result.errors[0]


def test_validate_core_config_checks_metrics_risk_profiles_file(
    base_config: CoreConfig, tmp_path: Path
) -> None:
    missing_path = tmp_path / "missing_profiles.yaml"
    metrics_config = MetricsServiceConfig(
        enabled=True,
        ui_alerts_risk_profile="balanced",
        ui_alerts_risk_profiles_file=str(missing_path),
    )
    config = replace(base_config, metrics_service=metrics_config)

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("ui_alerts_risk_profiles_file" in err for err in result.errors)

    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text("{}", encoding="utf-8")
    valid_metrics = replace(
        metrics_config,
        ui_alerts_risk_profiles_file=str(profiles_path),
    )
    result_ok = validate_core_config(replace(base_config, metrics_service=valid_metrics))
    assert result_ok.is_valid()


def _metrics_config_base() -> MetricsServiceConfig:
    tls = MetricsServiceTlsConfig()
    tls.enabled = True
    tls.certificate_path = "cert.pem"
    tls.private_key_path = "key.pem"
    tls.client_ca_path = "clients.pem"
    tls.require_client_auth = True

    return MetricsServiceConfig(
        enabled=True,
        host="127.0.0.1",
        port=55060,
        history_size=256,
        auth_token="token",
        log_sink=True,
        jsonl_path="audit/metrics.jsonl",
        jsonl_fsync=False,
        ui_alerts_jsonl_path="audit/ui_alerts.jsonl",
        ui_alerts_risk_profile="balanced",
        tls=tls,
        reduce_motion_alerts=True,
        reduce_motion_mode="enable",
        reduce_motion_category="ui.performance",
        reduce_motion_severity_active="warning",
        reduce_motion_severity_recovered="info",
        overlay_alerts=True,
        overlay_alert_mode="enable",
        overlay_alert_category="ui.performance.overlay",
        overlay_alert_severity_exceeded="warning",
        overlay_alert_severity_recovered="info",
        overlay_alert_severity_critical="critical",
        overlay_alert_critical_threshold=3,
        jank_alerts=True,
        jank_alert_mode="enable",
        jank_alert_category="ui.performance.jank",
        jank_alert_severity_spike="warning",
        jank_alert_severity_critical="critical",
        jank_alert_critical_over_ms=10.0,
    )


def test_validate_core_config_detects_controller_interval_without_backfill(
    base_config: CoreConfig,
) -> None:
    config_with_universe = _config_with_universe(base_config)
    controller = replace(
        config_with_universe.runtime_controllers["daily_trend_core"], interval="1h"
    )
    environment = replace(
        config_with_universe.environments["paper"],
        instrument_universe="core",
    )
    config = replace(
        config_with_universe,
        runtime_controllers={"daily_trend_core": controller},
        environments={"paper": environment},
    )

    result = validate_core_config(config)

    assert not result.is_valid()
    assert "interwału '1h'" in result.errors[0]


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


def test_validate_core_config_detects_invalid_backfill_interval_format(base_config: CoreConfig) -> None:
    universe_config = _config_with_universe(base_config)
    instrument = replace(
        next(iter(universe_config.instrument_universes["core"].instruments)),
        backfill_windows=(InstrumentBackfillWindow(interval="1x", lookback_days=10),),
    )
    broken_universe = replace(
        universe_config.instrument_universes["core"], instruments=(instrument,)
    )
    config = replace(universe_config, instrument_universes={"core": broken_universe})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("niepoprawny format" in err or "nieobsługiwany" in err for err in result.errors)


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


def test_validate_core_config_detects_unknown_runtime_interval(base_config: CoreConfig) -> None:
    controller = ControllerRuntimeConfig(tick_seconds=60.0, interval="daily")
    config = replace(base_config, runtime_controllers={"daily": controller})

    result = validate_core_config(config)
    assert not result.is_valid()
    assert any("niepoprawny format" in err or "nieobsługiwany" in err for err in result.errors)


def test_validate_core_config_warns_on_tick_mismatch(base_config: CoreConfig) -> None:
    controller = ControllerRuntimeConfig(tick_seconds=30.0, interval="1m")
    env = replace(base_config.environments["paper"], default_controller="fast")
    config = replace(
        base_config,
        runtime_controllers={"fast": controller},
        environments={"paper": env},
    )

    result = validate_core_config(config)
    assert result.is_valid()
    assert any("tick_seconds" in warn for warn in result.warnings)


def test_validate_core_config_accepts_valid_metrics_block(base_config: CoreConfig) -> None:
    metrics = _metrics_config_base()
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert result.is_valid()
    assert result.errors == []


def test_validate_core_config_detects_invalid_metrics_mode(base_config: CoreConfig) -> None:
    metrics = _metrics_config_base()
    metrics.reduce_motion_mode = "invalid"
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert not result.is_valid()
    assert any("reduce_motion_mode" in err for err in result.errors)


def test_validate_core_config_detects_unknown_metrics_risk_profile(base_config: CoreConfig) -> None:
    metrics = _metrics_config_base()
    metrics.ui_alerts_risk_profile = "unknown"
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert not result.is_valid()
    assert any("ui_alerts_risk_profile" in err for err in result.errors)


def test_validate_core_config_detects_nonpositive_overlay_threshold(
    base_config: CoreConfig,
) -> None:
    metrics = _metrics_config_base()
    metrics.overlay_alert_critical_threshold = 0
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert not result.is_valid()
    assert any("overlay_alert_critical_threshold" in err for err in result.errors)


def test_validate_core_config_detects_nonpositive_jank_threshold(
    base_config: CoreConfig,
) -> None:
    metrics = _metrics_config_base()
    metrics.jank_alert_critical_over_ms = -1.0
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert not result.is_valid()
    assert any("jank_alert_critical_over_ms" in err for err in result.errors)


def test_validate_core_config_detects_missing_tls_material(base_config: CoreConfig) -> None:
    metrics = _metrics_config_base()
    metrics.tls = MetricsServiceTlsConfig()
    metrics.tls.enabled = True
    metrics.tls.certificate_path = None
    metrics.tls.private_key_path = ""
    metrics.tls.require_client_auth = True
    metrics.tls.client_ca_path = ""
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert not result.is_valid()
    assert any("TLS" in err for err in result.errors)


def test_validate_core_config_warns_on_metrics_mode_flag_conflict(
    base_config: CoreConfig,
) -> None:
    metrics = _metrics_config_base()
    metrics.reduce_motion_alerts = False
    metrics.reduce_motion_mode = "enable"
    config = replace(base_config, metrics_service=metrics)

    result = validate_core_config(config)

    assert result.is_valid()
    assert any("reduce_motion_alerts" in warn for warn in result.warnings)
