from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping, Sequence
from textwrap import dedent

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.alerts import EmailChannel, SMSChannel, TelegramChannel
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeCredentials,
    OrderRequest,
)
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.repository import FileRiskRepository
from bot_core.runtime import BootstrapContext, bootstrap_environment
from bot_core.runtime.metrics_alerts import DEFAULT_UI_ALERTS_JSONL_PATH
from bot_core.security import SecretManager, SecretStorage, SecretStorageError


class _MemorySecretStorage(SecretStorage):
    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    def get_secret(self, key: str) -> str | None:
        return self._store.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._store[key] = value

    def delete_secret(self, key: str) -> None:
        self._store.pop(key, None)


_BASE_CONFIG = dedent(
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
      aggressive:
        max_daily_loss_pct: 0.05
        max_position_pct: 0.20
        target_volatility: 0.35
        max_leverage: 5.0
        stop_loss_atr_multiple: 2.0
        max_open_positions: 12
        hard_drawdown_pct: 0.25
    environments:
      binance_paper:
        exchange: binance_spot
        environment: paper
        keychain_key: binance_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/binance_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary", "email:ops", "sms:orange_local"]
        ip_allowlist: ["127.0.0.1"]
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
        alert_throttle:
          window_seconds: 60
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
      zonda_paper:
        exchange: zonda_spot
        environment: paper
        keychain_key: zonda_paper_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_paper
        risk_profile: balanced
        alert_channels: ["telegram:primary"]
        ip_allowlist: ["127.0.0.1"]
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
        alert_throttle:
          window_seconds: 120
          exclude_severities: [critical]
          exclude_categories: [health]
          max_entries: 32
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123456789"
          token_secret: telegram_token
      email_channels:
        ops:
          host: smtp.example.com
          port: 587
          from_address: bot@example.com
          recipients: ["ops@example.com"]
          credential_secret: smtp_credentials
      sms_providers:
        orange_local:
          provider: orange_pl
          api_base_url: https://api.orange.pl/sms/v1
          from_number: BOT-ORANGE
          recipients: ["+48555111222"]
          allow_alphanumeric_sender: true
          sender_id: BOT-ORANGE
          credential_key: sms_orange
    """
)


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "core.yaml"
    config_path.write_text(_BASE_CONFIG, encoding="utf-8")
    return config_path


def _write_config_custom(
    tmp_path: Path,
    *,
    runtime_metrics: Mapping[str, object] | None = None,
    environment_overrides: Mapping[str, object] | None = None,
) -> Path:
    data = yaml.safe_load(_BASE_CONFIG)
    if environment_overrides:
        data.setdefault("environments", {}).setdefault("binance_paper", {}).update(
            environment_overrides
        )
    if runtime_metrics is not None:
        runtime_section = data.setdefault("runtime", {})
        runtime_section["metrics_service"] = runtime_metrics

    config_path = tmp_path / "core_custom.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return config_path


def _prepare_manager() -> tuple[_MemorySecretStorage, SecretManager]:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")
    credentials_payload = {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "passphrase": None,
        "permissions": ["read", "trade"],
        "environment": Environment.PAPER.value,
    }
    storage.set_secret("tests:binance_paper_key:trading", json.dumps(credentials_payload))
    storage.set_secret("tests:zonda_paper_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")
    manager.store_secret_value(
        "smtp_credentials",
        json.dumps({"username": "bot", "password": "secret"}),
        purpose="alerts:email",
    )
    manager.store_secret_value(
        "sms_orange",
        json.dumps({"account_sid": "AC123", "auth_token": "token"}),
        purpose="alerts:sms",
    )
    return storage, manager


def test_bootstrap_environment_initialises_components(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert isinstance(context, BootstrapContext)
    assert context.environment.name == "binance_paper"
    assert context.risk_profile_name == "balanced"
    assert context.credentials.key_id == "paper-key"
    assert context.adapter.credentials.key_id == "paper-key"

    assert isinstance(context.risk_engine, ThresholdRiskEngine)
    assert isinstance(context.risk_repository, FileRiskRepository)
    result = context.risk_engine.apply_pre_trade_checks(
        OrderRequest(
            symbol="BTCUSDT",
            side="buy",
            quantity=0.2,
            order_type="limit",
            price=100.0,
            # ATR=1.0, stop_loss_atr_multiple=1.5 => wymagany SL = 98.5
            stop_price=100.0 - 1.0 * 1.5,
            atr=1.0,
        ),
        account=AccountSnapshot(
            balances={"USDT": 1000.0},
            total_equity=1000.0,
            available_margin=1000.0,
            maintenance_margin=0.0,
        ),
        profile_name="balanced",
    )
    assert result.allowed is True

    assert set(context.alert_channels.keys()) == {
        "telegram:primary",
        "email:ops",
        "sms:orange_local",
    }
    assert isinstance(context.alert_channels["telegram:primary"], TelegramChannel)
    assert isinstance(context.alert_channels["email:ops"], EmailChannel)
    assert isinstance(context.alert_channels["sms:orange_local"], SMSChannel)
    assert len(context.alert_router.channels) == 3
    assert context.alert_router.audit_log is context.audit_log
    assert context.alert_router.throttle is not None
    assert context.alert_router.throttle.window.total_seconds() == pytest.approx(60.0)
    assert "critical" in context.alert_router.throttle.exclude_severities

    # Health check nie powinien zgłaszać wyjątków dla świeżo zainicjalizowanych kanałów.
    snapshot = context.alert_router.health_snapshot()
    assert snapshot["telegram:primary"]["status"] == "ok"

    assert context.risk_engine.should_liquidate(profile_name="balanced") is False
    assert context.adapter_settings == {}
    risk_state_path = Path("./var/data/binance_paper/risk_state/balanced.json")
    assert risk_state_path.parent.exists()
    assert context.metrics_server is None
    assert context.metrics_ui_alert_sink_active is True
    default_alert_log = DEFAULT_UI_ALERTS_JSONL_PATH.expanduser()
    if not default_alert_log.is_absolute():
        default_alert_log = default_alert_log.resolve(strict=False)
    assert Path(context.metrics_ui_alerts_path or "") == default_alert_log
    assert context.metrics_jsonl_path is None
    assert context.metrics_service_enabled is None
    assert context.metrics_ui_alerts_metadata is not None
    metadata_path = context.metrics_ui_alerts_metadata.get("path")
    assert metadata_path is not None
    assert Path(str(metadata_path)).name == DEFAULT_UI_ALERTS_JSONL_PATH.name
    assert Path(
        context.metrics_ui_alerts_metadata.get("absolute_path")
    ).name == DEFAULT_UI_ALERTS_JSONL_PATH.name
    assert context.metrics_jsonl_metadata is None
    if context.metrics_security_warnings is not None:
        assert isinstance(context.metrics_security_warnings, tuple)
    assert context.metrics_ui_alerts_settings is not None
    assert context.metrics_ui_alerts_settings["reduce_mode"] == "enable"
    assert context.metrics_ui_alerts_settings["overlay_mode"] == "enable"
    assert context.metrics_ui_alerts_settings["jank_mode"] == "disable"
    assert context.metrics_ui_alerts_settings["reduce_motion_alerts"] is True
    assert context.metrics_ui_alerts_settings["overlay_alerts"] is True
    assert context.metrics_ui_alerts_settings["jank_alerts"] is False
    assert context.metrics_ui_alerts_settings["reduce_motion_logging"] is True
    assert context.metrics_ui_alerts_settings["overlay_logging"] is True
    assert context.metrics_ui_alerts_settings["jank_logging"] is False
    assert context.metrics_ui_alerts_settings["jank_category"] == "ui.performance"
    assert context.metrics_ui_alerts_settings["jank_severity_spike"] == "warning"
    assert context.metrics_ui_alerts_settings["jank_severity_critical"] is None
    assert context.metrics_ui_alerts_settings["jank_critical_over_ms"] is None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "inherit"
    assert audit_info["backend"] == "memory"
    assert audit_info["note"] == "inherited_environment_router"


def test_bootstrap_environment_allows_risk_profile_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    storage, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper",
        config_path=config_path,
        secret_manager=manager,
        risk_profile_name="aggressive",
    )

    assert context.risk_profile_name == "aggressive"
    assert context.environment.risk_profile == "aggressive"
    assert context.risk_engine.should_liquidate(profile_name="aggressive") is False


def test_bootstrap_environment_detects_missing_permissions(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_path = _write_config(tmp_path)
    credentials_payload = {
        "key_id": "paper-key",
        "secret": "paper-secret",
        "passphrase": None,
        "permissions": ["read"],
        "environment": Environment.PAPER.value,
    }
    storage.set_secret("tests:binance_paper_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "telegram-secret", purpose="alerts:telegram")
    manager.store_secret_value(
        "smtp_credentials",
        json.dumps({"username": "bot", "password": "secret"}),
        purpose="alerts:email",
    )
    manager.store_secret_value(
        "sms_orange",
        json.dumps({"account_sid": "AC123", "auth_token": "token"}),
        purpose="alerts:sms",
    )

    with pytest.raises(SecretStorageError):
        bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)


def test_bootstrap_environment_supports_zonda(tmp_path: Path) -> None:
    storage = _MemorySecretStorage()
    manager = SecretManager(storage, namespace="tests")

    config_content = """
    risk_profiles:
      conservative:
        max_daily_loss_pct: 0.01
        max_position_pct: 0.03
        target_volatility: 0.07
        max_leverage: 2.0
        stop_loss_atr_multiple: 1.0
        max_open_positions: 3
        hard_drawdown_pct: 0.05
    environments:
      zonda_live:
        exchange: zonda_spot
        environment: live
        keychain_key: zonda_live_key
        credential_purpose: trading
        data_cache_path: ./var/data/zonda_live
        risk_profile: conservative
        alert_channels: ["telegram:primary"]
        ip_allowlist: []
        required_permissions: [read, trade]
        forbidden_permissions: [withdraw]
    reporting: {}
    alerts:
      telegram_channels:
        primary:
          chat_id: "123"
          token_secret: telegram_token
      email_channels: {}
      sms_providers: {}
      signal_channels: {}
      whatsapp_channels: {}
      messenger_channels: {}
    """

    config_path = tmp_path / "core.yaml"
    config_path.write_text(config_content, encoding="utf-8")

    credentials_payload = {
        "key_id": "zonda-key",
        "secret": "zonda-secret",
        "permissions": ["read", "trade"],
        "environment": Environment.LIVE.value,
    }
    storage.set_secret("tests:zonda_live_key:trading", json.dumps(credentials_payload))
    manager.store_secret_value("telegram_token", "tg-token", purpose="alerts:telegram")

    context = bootstrap_environment("zonda_live", config_path=config_path, secret_manager=manager)

    assert context.adapter.name == "zonda_spot"
    assert context.credentials.key_id == "zonda-key"
    assert context.environment.exchange == "zonda_spot"
    assert context.adapter_settings == {}


def test_bootstrap_metrics_ui_alerts_audit_requested_file_without_backend(tmp_path: Path) -> None:
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={"enabled": True, "ui_alerts_audit_backend": "file"},
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "file"
    assert audit_info["backend"] == "memory"
    assert audit_info["note"] == "file_backend_unavailable"


def test_bootstrap_metrics_ui_alerts_audit_inherits_file_backend(tmp_path: Path) -> None:
    alerts_dir = tmp_path / "ui-alert-audit"
    runtime_metrics = {
        "enabled": True,
        "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
    }
    environment_overrides = {
        "alert_audit": {
            "backend": "file",
            "directory": str(alerts_dir),
            "filename_pattern": "alerts-%Y%m%d.jsonl",
            "retention_days": 14,
            "fsync": True,
        }
    }
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics=runtime_metrics,
        environment_overrides=environment_overrides,
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "inherit"
    assert audit_info["backend"] == "file"
    assert audit_info["note"] == "inherited_environment_router"
    assert audit_info["directory"] == str(alerts_dir)
    assert audit_info["pattern"] == "alerts-%Y%m%d.jsonl"
    assert audit_info["retention_days"] == 14
    assert audit_info["fsync"] is True


def test_bootstrap_metrics_ui_alerts_audit_memory_request_on_file_backend(tmp_path: Path) -> None:
    alerts_dir = tmp_path / "ui-alert-audit"
    runtime_metrics = {
        "enabled": True,
        "ui_alerts_audit_backend": "memory",
        "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
    }
    environment_overrides = {
        "alert_audit": {
            "backend": "file",
            "directory": str(alerts_dir),
            "filename_pattern": "alerts-%Y%m%d.jsonl",
            "retention_days": 7,
            "fsync": False,
        }
    }
    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics=runtime_metrics,
        environment_overrides=environment_overrides,
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment("binance_paper", config_path=config_path, secret_manager=manager)

    assert context.metrics_ui_alerts_settings is not None
    audit_info = context.metrics_ui_alerts_settings.get("audit")
    assert audit_info is not None
    assert audit_info["requested"] == "memory"
    assert audit_info["backend"] == "file"
    assert audit_info["note"] == "memory_backend_not_selected"


def test_bootstrap_metrics_risk_profiles_file_metadata(tmp_path: Path) -> None:
    profiles_path = tmp_path / "telemetry_profiles.json"
    profiles_path.write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "balanced": {
                        "metrics_service_overrides": {
                            "ui_alerts_overlay_critical_threshold": 7,
                        },
                        "severity_min": "notice",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "jsonl_path": str((tmp_path / "metrics.jsonl").resolve()),
            "ui_alerts_jsonl_path": str((tmp_path / "ui_alerts.jsonl").resolve()),
            "reduce_motion_alerts": True,
            "overlay_alerts": True,
            "jank_alerts": False,
            "ui_alerts_risk_profile": "balanced",
            "ui_alerts_risk_profiles_file": str(profiles_path),
        },
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    assert context.metrics_ui_alerts_settings is not None
    settings = context.metrics_ui_alerts_settings
    risk_meta = settings.get("risk_profile")
    assert isinstance(risk_meta, dict)
    assert risk_meta.get("name") == "balanced"
    summary_meta = settings.get("risk_profile_summary")
    assert isinstance(summary_meta, dict)
    assert summary_meta.get("name") == "balanced"
    file_meta = settings.get("risk_profiles_file")
    assert isinstance(file_meta, dict)
    assert Path(str(file_meta.get("path"))).resolve() == profiles_path.resolve()
    assert "balanced" in file_meta.get("registered_profiles", [])


def test_bootstrap_metrics_risk_profiles_directory_metadata(tmp_path: Path) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "ops.json").write_text(
        json.dumps(
            {
                "risk_profiles": {
                    "ops_dir": {
                        "metrics_service_overrides": {
                            "ui_alerts_overlay_critical_threshold": 5
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    (profiles_dir / "lab.yaml").write_text(
        "risk_profiles:\n  lab_dir:\n    severity_min: warning\n",
        encoding="utf-8",
    )

    config_path = _write_config_custom(
        tmp_path,
        runtime_metrics={
            "enabled": True,
            "ui_alerts_risk_profile": "balanced",
            "ui_alerts_risk_profiles_file": str(profiles_dir),
        },
    )
    _, manager = _prepare_manager()

    context = bootstrap_environment(
        "binance_paper", config_path=config_path, secret_manager=manager
    )

    settings = context.metrics_ui_alerts_settings
    assert isinstance(settings, dict)
    summary_meta = settings.get("risk_profile_summary")
    assert isinstance(summary_meta, dict)
    assert summary_meta.get("name") == "balanced"
    file_meta = settings.get("risk_profiles_file")
    assert isinstance(file_meta, dict)
    assert file_meta["type"] == "directory"
    assert file_meta["path"] == str(profiles_dir)
    assert "ops_dir" in file_meta.get("registered_profiles", [])
    assert any(entry["path"].endswith("lab.yaml") for entry in file_meta.get("files", []))
