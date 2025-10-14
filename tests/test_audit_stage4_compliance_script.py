from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import os

import yaml

from scripts import audit_stage4_compliance


_DEF_ENVIRONMENT = "demo"
_DEF_PROFILE = "balanced"
_DEF_BUNDLE = "core-oem"


def _write_config(tmp_path: Path, rotation_path: Path) -> Path:
    config_dir = tmp_path
    (config_dir / "secrets/runtime/metrics").mkdir(parents=True, exist_ok=True)
    (config_dir / "secrets/mtls").mkdir(parents=True, exist_ok=True)
    for name in (
        "server.crt",
        "server.key",
        "ca.pem",
        "core-oem-ca.pem",
        "core-oem-server.pem",
        "core-oem-server-key.pem",
        "core-oem-client.pem",
        "core-oem-client-key.pem",
    ):
        (config_dir / f"secrets/runtime/metrics/{name}").write_text("cert", encoding="utf-8")
        (config_dir / f"secrets/mtls/{name}").write_text("cert", encoding="utf-8")

    config = {
        "environments": {
            _DEF_ENVIRONMENT: {
                "name": _DEF_ENVIRONMENT,
                "exchange": "binance",
                "environment": "paper",
                "keychain_key": "demo-key",
                "data_cache_path": "data/demo",
                "risk_profile": _DEF_PROFILE,
                "alert_channels": [],
            }
        },
        "risk_profiles": {
            _DEF_PROFILE: {
                "name": _DEF_PROFILE,
                "max_daily_loss_pct": 0.05,
                "max_position_pct": 0.3,
                "target_volatility": 0.15,
                "max_leverage": 2.0,
                "stop_loss_atr_multiple": 3.0,
                "max_open_positions": 8,
                "hard_drawdown_pct": 0.2,
            }
        },
        "metrics_service": {
            "auth_token_env": "METRICS_TOKEN",
            "tls": {
                "enabled": True,
                "require_client_auth": True,
                "certificate_path": "secrets/runtime/metrics/server.crt",
                "private_key_path": "secrets/runtime/metrics/server.key",
                "client_ca_path": "secrets/runtime/metrics/ca.pem",
            },
        },
        "multi_strategy_schedulers": {
            "core": {
                "telemetry_namespace": "runtime.multi",
                "decision_log_category": "runtime.scheduler.core",
                "health_check_interval": 60,
                "rbac_tokens": [
                    {
                        "token_id": "scheduler",
                        "token_env": "SCHED_TOKEN",
                        "scopes": [
                            "runtime.schedule.read",
                            "runtime.schedule.write",
                        ],
                    }
                ],
                "schedules": {
                    "stub": {
                        "strategy": "core_mean_reversion",
                        "cadence_seconds": 900,
                        "max_drift_seconds": 60,
                        "warmup_bars": 5,
                        "risk_profile": _DEF_PROFILE,
                    }
                },
            }
        },
        "execution": {
            "live_router": {
                "decision_log": {
                    "path": "audit/decision_logs/live.jsonl",
                    "hmac_key_env": "LIVE_ROUTER_HMAC",
                    "key_id": "live-router",
                }
            },
            "mtls": {
                "bundle_directory": "secrets/mtls",
                "ca_certificate": "secrets/mtls/core-oem-ca.pem",
                "server_certificate": "secrets/mtls/core-oem-server.pem",
                "server_key": "secrets/mtls/core-oem-server-key.pem",
                "client_certificate": "secrets/mtls/core-oem-client.pem",
                "client_key": "secrets/mtls/core-oem-client-key.pem",
                "rotation_registry": str(rotation_path),
            },
        },
        "strategies": {},
        "mean_reversion_strategies": {
            "core_mean_reversion": {
                "parameters": {
                    "lookback": 10,
                    "entry_zscore": 2.0,
                    "exit_zscore": 0.5,
                    "max_holding_period": 5,
                    "volatility_cap": 0.02,
                }
            }
        },
        "volatility_target_strategies": {},
        "cross_exchange_arbitrage_strategies": {},
    }

    config_path = config_dir / "core.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def _write_rotation(rotation_path: Path, *, days_ago: float) -> None:
    rotation_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    payload = {
        f"{_DEF_BUNDLE}::tls_ca": timestamp.isoformat().replace("+00:00", "Z"),
        f"{_DEF_BUNDLE}::tls_server": timestamp.isoformat().replace("+00:00", "Z"),
        f"{_DEF_BUNDLE}::tls_client": timestamp.isoformat().replace("+00:00", "Z"),
    }
    rotation_path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_stage4_compliance_pass(monkeypatch, tmp_path):
    rotation_path = tmp_path / "var/security/tls_rotation.json"
    _write_rotation(rotation_path, days_ago=5)
    config_path = _write_config(tmp_path, rotation_path)

    monkeypatch.setenv("METRICS_TOKEN", "x" * 32)
    monkeypatch.setenv("SCHED_TOKEN", "y" * 32)
    monkeypatch.setenv("LIVE_ROUTER_HMAC", "z" * 48)

    exit_code = audit_stage4_compliance.run(
        [
            "--config",
            str(config_path),
            "--check-paths",
            "--mtls-bundle-name",
            _DEF_BUNDLE,
        ]
    )

    assert exit_code == 0


def test_audit_stage4_compliance_reports_missing_env(monkeypatch, tmp_path):
    rotation_path = tmp_path / "var/security/tls_rotation.json"
    _write_rotation(rotation_path, days_ago=5)
    config_path = _write_config(tmp_path, rotation_path)

    monkeypatch.delenv("SCHED_TOKEN", raising=False)
    monkeypatch.setenv("METRICS_TOKEN", "x" * 32)
    monkeypatch.setenv("LIVE_ROUTER_HMAC", "z" * 32)

    exit_code = audit_stage4_compliance.run(
        [
            "--config",
            str(config_path),
            "--mtls-bundle-name",
            _DEF_BUNDLE,
        ]
    )

    assert exit_code == 1


def test_audit_stage4_compliance_warns_about_rotation(monkeypatch, tmp_path):
    rotation_path = tmp_path / "var/security/tls_rotation.json"
    _write_rotation(rotation_path, days_ago=85)
    config_path = _write_config(tmp_path, rotation_path)

    monkeypatch.setenv("METRICS_TOKEN", "x" * 32)
    monkeypatch.setenv("SCHED_TOKEN", "y" * 32)
    monkeypatch.setenv("LIVE_ROUTER_HMAC", "z" * 48)

    exit_code = audit_stage4_compliance.run(
        [
            "--config",
            str(config_path),
            "--mtls-bundle-name",
            _DEF_BUNDLE,
        ]
    )

    assert exit_code == 0
