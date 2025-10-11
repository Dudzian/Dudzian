"""Testy ładowania konfiguracji."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.config import (
    AlertAuditConfig,
    EmailChannelSettings,
    EnvironmentDataQualityConfig,
    InstrumentConfig,
    MessengerChannelSettings,
    SMSProviderSettings,
    SignalChannelSettings,
    TelegramChannelSettings,
    WhatsAppChannelSettings,
    load_core_config,
)


def test_load_core_config_reads_sms_providers(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        instrument_universes:
          core_multi_exchange:
            description: "Testowe uniwersum"
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                  kraken_spot: XBTUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 3650
        environments: {}
        reporting: {}
        alerts:
          sms_providers:
            orange_local:
              provider: orange_pl
              api_base_url: https://api.orange.pl/sms/v1
              from_number: "+48500100100"
              recipients: ["+48555111222"]
              allow_alphanumeric_sender: true
              sender_id: BOT-ORANGE
              credential_key: orange_sms_credentials
          telegram_channels:
            primary:
              chat_id: "123456789"
              token_secret: telegram_primary_token
              parse_mode: MarkdownV2
          signal_channels:
            workstation:
              service_url: https://signal-gateway.local
              sender_number: "+48500100999"
              recipients: ["+48555111222"]
              credential_secret: signal_cli_token
              verify_tls: true
          whatsapp_channels:
            business:
              phone_number_id: "10987654321"
              recipients: ["48555111222"]
              token_secret: whatsapp_primary_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
          messenger_channels:
            ops:
              page_id: "1357924680"
              recipients: ["2468013579"]
              token_secret: messenger_page_token
              api_base_url: https://graph.facebook.com
              api_version: v16.0
          email_channels:
            ops:
              host: smtp.example.com
              port: 587
              from_address: bot@example.com
              recipients: ["ops@example.com"]
              credential_secret: smtp_ops_credentials
              use_tls: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    # SMS providers
    assert "orange_local" in config.sms_providers
    provider = config.sms_providers["orange_local"]
    assert isinstance(provider, SMSProviderSettings)
    assert provider.provider_key == "orange_pl"
    assert provider.api_base_url == "https://api.orange.pl/sms/v1"
    assert provider.allow_alphanumeric_sender is True
    assert provider.sender_id == "BOT-ORANGE"
    assert provider.credential_key == "orange_sms_credentials"

    # Telegram
    assert "primary" in config.telegram_channels
    telegram = config.telegram_channels["primary"]
    assert isinstance(telegram, TelegramChannelSettings)
    assert telegram.chat_id == "123456789"
    assert telegram.token_secret == "telegram_primary_token"
    assert telegram.parse_mode == "MarkdownV2"

    # Signal
    assert "workstation" in config.signal_channels
    signal = config.signal_channels["workstation"]
    assert isinstance(signal, SignalChannelSettings)
    assert signal.service_url == "https://signal-gateway.local"
    assert signal.sender_number == "+48500100999"
    assert signal.credential_secret == "signal_cli_token"
    assert signal.verify_tls is True

    # WhatsApp
    assert "business" in config.whatsapp_channels
    whatsapp = config.whatsapp_channels["business"]
    assert isinstance(whatsapp, WhatsAppChannelSettings)
    assert whatsapp.phone_number_id == "10987654321"
    assert whatsapp.token_secret == "whatsapp_primary_token"
    assert whatsapp.api_version == "v16.0"

    # Messenger
    assert "ops" in config.messenger_channels
    messenger = config.messenger_channels["ops"]
    assert isinstance(messenger, MessengerChannelSettings)
    assert messenger.page_id == "1357924680"
    assert messenger.token_secret == "messenger_page_token"

    # Email
    email = config.email_channels["ops"]
    assert isinstance(email, EmailChannelSettings)
    assert email.host == "smtp.example.com"
    assert email.port == 587
    assert email.from_address == "bot@example.com"
    assert email.recipients == ("ops@example.com",)
    assert email.credential_secret == "smtp_ops_credentials"
    assert email.use_tls is True


def test_load_core_config_parses_alert_audit(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          balanced:
            max_daily_loss_pct: 0.015
            max_position_pct: 0.05
            target_volatility: 0.11
            max_leverage: 3.0
            stop_loss_atr_multiple: 1.5
            max_open_positions: 5
            hard_drawdown_pct: 0.10
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: binance_paper_key
            credential_purpose: trading
            data_cache_path: ./var/data/binance_paper
            risk_profile: balanced
            alert_channels: []
            alert_audit:
              backend: file
              directory: alerts
              filename_pattern: alerts-%Y%m%d.jsonl
              retention_days: 30
              fsync: true
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    env = config.environments["binance_paper"]
    assert isinstance(env.alert_audit, AlertAuditConfig)
    assert env.alert_audit.backend == "file"
    assert env.alert_audit.directory == "alerts"
    assert env.alert_audit.retention_days == 30
    assert env.alert_audit.fsync is True


def test_load_core_config_loads_strategies(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        strategies:
          core_daily_trend:
            engine: daily_trend_momentum
            parameters:
              fast_ma: 30
              slow_ma: 120
              breakout_lookback: 60
              momentum_window: 25
              atr_window: 15
              atr_multiplier: 1.8
              min_trend_strength: 0.01
              min_momentum: 0.002
        instrument_universes: {}
        environments: {}
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert "core_daily_trend" in config.strategies
    strategy = config.strategies["core_daily_trend"]
    assert strategy.fast_ma == 30
    assert strategy.slow_ma == 120
    assert strategy.breakout_lookback == 60
    assert strategy.momentum_window == 25
    assert strategy.atr_window == 15
    assert abs(strategy.atr_multiplier - 1.8) < 1e-9
    assert abs(strategy.min_trend_strength - 0.01) < 1e-9
    assert abs(strategy.min_momentum - 0.002) < 1e-9


def test_load_core_config_inherits_risk_profile_data_quality(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles:
          balanced:
            max_daily_loss_pct: 0.02
            max_position_pct: 0.05
            target_volatility: 0.1
            max_leverage: 3.0
            stop_loss_atr_multiple: 1.5
            max_open_positions: 5
            hard_drawdown_pct: 0.1
            data_quality:
              max_gap_minutes: 180.0
              min_ok_ratio: 0.85
        instrument_universes:
          core_daily:
            description: Sample
            instruments:
              BTC_USDT:
                base_asset: BTC
                quote_asset: USDT
                categories: [core]
                exchanges:
                  binance_spot: BTCUSDT
                backfill:
                  - interval: 1d
                    lookback_days: 30
        environments:
          binance_paper:
            exchange: binance_spot
            environment: paper
            keychain_key: binance_spot_paper
            data_cache_path: ./var/data/binance
            risk_profile: balanced
            alert_channels: []
            instrument_universe: core_daily
        reporting: {}
        alerts: {}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    env = config.environments["binance_paper"]
    assert isinstance(env.data_quality, EnvironmentDataQualityConfig)
    assert env.data_quality.max_gap_minutes == 180.0
    assert env.data_quality.min_ok_ratio == 0.85


def test_load_core_config_reads_metrics_service(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            host: 0.0.0.0
            port: 55123
            history_size: 256
            auth_token: secret-token
            log_sink: false
            # brak jsonl_path => None
            ui_alerts_risk_profile: Conservative
            reduce_motion_alerts: true
            reduce_motion_mode: enable
            reduce_motion_category: ui.performance.guard
            reduce_motion_severity_active: critical
            reduce_motion_severity_recovered: notice
            overlay_alerts: true
            overlay_alert_mode: jsonl
            overlay_alert_category: ui.performance.overlay
            overlay_alert_severity_exceeded: critical
            overlay_alert_severity_recovered: notice
            overlay_alert_severity_critical: emergency
            overlay_alert_critical_threshold: 3
            jank_alerts: true
            jank_alert_mode: enable
            jank_alert_category: ui.performance.jank
            jank_alert_severity_spike: major
            jank_alert_severity_critical: critical
            jank_alert_critical_over_ms: 7.5
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    assert config.metrics_service is not None
    metrics = config.metrics_service
    assert metrics.enabled is True
    assert metrics.host == "0.0.0.0"
    assert metrics.port == 55123
    assert metrics.history_size == 256
    assert metrics.auth_token == "secret-token"
    assert metrics.log_sink is False
    assert metrics.jsonl_path is None
    assert metrics.ui_alerts_risk_profile == "conservative"

    # Pola reduce/overlay z gałęzi UI
    assert metrics.reduce_motion_alerts is True
    assert metrics.reduce_motion_mode == "enable"
    assert metrics.reduce_motion_category == "ui.performance.guard"
    assert metrics.reduce_motion_severity_active == "critical"
    assert metrics.reduce_motion_severity_recovered == "notice"
    assert metrics.overlay_alerts is True
    assert metrics.overlay_alert_mode == "jsonl"
    assert metrics.overlay_alert_category == "ui.performance.overlay"
    assert metrics.overlay_alert_severity_exceeded == "critical"
    assert metrics.overlay_alert_severity_recovered == "notice"
    assert metrics.overlay_alert_severity_critical == "emergency"
    assert metrics.overlay_alert_critical_threshold == 3
    assert metrics.jank_alerts is True
    assert metrics.jank_alert_mode == "enable"
    assert metrics.jank_alert_category == "ui.performance.jank"
    assert metrics.jank_alert_severity_spike == "major"
    assert metrics.jank_alert_severity_critical == "critical"
    assert metrics.jank_alert_critical_over_ms == pytest.approx(7.5)

    # Metadane ścieżek źródłowych configu (ustawiane przez loader)
    assert Path(config.source_path or "").is_absolute()
    expected_dir = config_path.resolve(strict=False).parent
    assert config.source_directory == str(expected_dir)


def test_load_core_config_normalizes_ui_alert_modes(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            reduce_motion_mode: ENABLE
            overlay_alert_mode: JsonL
            jank_alert_mode: DISABLE
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    metrics = config.metrics_service
    assert metrics is not None
    assert metrics.reduce_motion_mode == "enable"
    assert metrics.overlay_alert_mode == "jsonl"
    assert metrics.jank_alert_mode == "disable"


def test_load_core_config_rejects_unknown_ui_alert_mode(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            reduce_motion_mode: maybe
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_core_config(config_path)


def test_load_core_config_resolves_metrics_paths_relative_to_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "conf"
    config_dir.mkdir()
    config_path = config_dir / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            jsonl_path: logs/metrics/telemetry.jsonl
            ui_alerts_jsonl_path: logs/metrics/ui_alerts.jsonl
            tls:
              enabled: true
              certificate_path: secrets/server.crt
              private_key_path: secrets/server.key
              client_ca_path: secrets/client_ca.pem
              require_client_auth: true
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)

    metrics = config.metrics_service
    assert metrics is not None
    expected_jsonl = (config_dir / "logs/metrics/telemetry.jsonl").resolve(strict=False)
    expected_alerts = (config_dir / "logs/metrics/ui_alerts.jsonl").resolve(strict=False)
    assert metrics.jsonl_path == str(expected_jsonl)
    assert metrics.ui_alerts_jsonl_path == str(expected_alerts)

    assert metrics.tls is not None
    expected_cert = (config_dir / "secrets/server.crt").resolve(strict=False)
    expected_key = (config_dir / "secrets/server.key").resolve(strict=False)
    expected_ca = (config_dir / "secrets/client_ca.pem").resolve(strict=False)
    assert metrics.tls.certificate_path == str(expected_cert)
    assert metrics.tls.private_key_path == str(expected_key)
    assert metrics.tls.client_ca_path == str(expected_ca)


def test_load_core_config_parses_metrics_risk_profiles_file(tmp_path: Path) -> None:
    profiles_path = tmp_path / "telemetry_profiles.yaml"
    profiles_path.write_text("risk_profiles: {}\n", encoding="utf-8")

    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        f"""
        risk_profiles: {{}}
        runtime:
          metrics_service:
            enabled: true
            ui_alerts_risk_profiles_file: {profiles_path.name}
        """,
        encoding="utf-8",
    )

    config = load_core_config(config_path)
    assert config.metrics_service is not None
    resolved_value = Path(config.metrics_service.ui_alerts_risk_profiles_file or "")
    assert resolved_value.resolve(strict=False) == profiles_path.resolve(strict=False)


def test_load_core_config_rejects_invalid_jank_threshold(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        """
        risk_profiles: {}
        environments: {}
        alerts: {}
        runtime:
          metrics_service:
            enabled: true
            jank_alerts: true
            jank_alert_critical_over_ms: not-a-number
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_core_config(config_path)
