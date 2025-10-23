from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Sequence

import os

import yaml

from scripts import audit_stage4_compliance


_DEF_ENVIRONMENT = "demo"
_DEF_PROFILE = "balanced"
_DEF_BUNDLE = "core-oem"


def _write_config(
    tmp_path: Path,
    rotation_path: Path,
    *,
    tco_reports: Sequence[str] | None = None,
    stage5_key_registry: str | None = None,
) -> Path:
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

    if tco_reports is None:
        tco_reports = ["var/audit/tco/latest_report.json"]
    if stage5_key_registry is None:
        stage5_key_registry = "var/security/key_rotation.json"

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
        "decision_engine": {
            "orchestrator": {
                "max_cost_bps": 12.0,
                "min_net_edge_bps": 3.0,
                "max_daily_loss_pct": 0.02,
                "max_drawdown_pct": 0.1,
                "max_position_ratio": 0.4,
                "max_open_positions": 6,
                "max_latency_ms": 250.0,
                "max_trade_notional": 25000.0,
            },
            "require_cost_data": True,
            "penalty_cost_bps": 0.0,
            "evaluation_history_limit": 128,
            "tco": {
                "reports": list(tco_reports),
                "warn_report_age_hours": 24.0,
                "max_report_age_hours": 72.0,
            },
        },
        "observability": {
            "slo": {
                "decision_latency": {
                    "metric": "bot_core_decision_latency_ms",
                    "objective": 220.0,
                    "comparator": "<=",
                    "aggregation": "p95",
                },
                "trade_cost": {
                    "metric": "bot_core_trade_cost_bps",
                    "objective": 12.0,
                    "comparator": "<=",
                    "aggregation": "average",
                },
                "fill_rate": {
                    "metric": "bot_core_fill_rate_pct",
                    "objective": 0.9,
                    "comparator": ">=",
                    "aggregation": "average",
                },
            },
            "key_rotation": {
                "registry_path": stage5_key_registry,
                "default_interval_days": 90.0,
                "default_warn_within_days": 14.0,
                "entries": [
                    {
                        "key": "binance-api",
                        "purpose": "trading",
                        "interval_days": 60.0,
                        "warn_within_days": 10.0,
                    },
                    {
                        "key": "core-mtls",
                        "purpose": "cert",
                    },
                ],
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


def _write_stage5_key_registry(path: Path, *, days_ago: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
    payload = {
        "binance-api::trading": timestamp.isoformat().replace("+00:00", "Z"),
        "core-mtls::cert": timestamp.isoformat().replace("+00:00", "Z"),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


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


def test_audit_stage5_compliance_pass(monkeypatch, tmp_path):
    rotation_path = tmp_path / "var/security/tls_rotation.json"
    _write_rotation(rotation_path, days_ago=5)
    key_registry_path = tmp_path / "var/security/key_rotation.json"
    _write_stage5_key_registry(key_registry_path, days_ago=5)
    tco_report_path = tmp_path / "var/audit/tco/latest_report.json"
    tco_report_path.parent.mkdir(parents=True, exist_ok=True)
    tco_report_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    config_path = _write_config(
        tmp_path,
        rotation_path,
        tco_reports=["var/audit/tco/latest_report.json"],
        stage5_key_registry="var/security/key_rotation.json",
    )

    monkeypatch.setenv("METRICS_TOKEN", "x" * 32)
    monkeypatch.setenv("SCHED_TOKEN", "y" * 32)
    monkeypatch.setenv("LIVE_ROUTER_HMAC", "z" * 48)

    exit_code = audit_stage4_compliance.run(
        [
            "--config",
            str(config_path),
            "--profile",
            "stage5",
            "--check-paths",
            "--mtls-bundle-name",
            _DEF_BUNDLE,
        ]
    )

    assert exit_code == 0


def test_audit_stage5_compliance_detects_stale_tco(monkeypatch, tmp_path):
    rotation_path = tmp_path / "var/security/tls_rotation.json"
    _write_rotation(rotation_path, days_ago=5)
    key_registry_path = tmp_path / "var/security/key_rotation.json"
    _write_stage5_key_registry(key_registry_path, days_ago=5)
    tco_report_path = tmp_path / "var/audit/tco/latest_report.json"
    tco_report_path.parent.mkdir(parents=True, exist_ok=True)
    tco_report_path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")

    stale_timestamp = datetime.now(timezone.utc) - timedelta(hours=80)
    os.utime(tco_report_path, (stale_timestamp.timestamp(), stale_timestamp.timestamp()))

    config_path = _write_config(
        tmp_path,
        rotation_path,
        tco_reports=["var/audit/tco/latest_report.json"],
        stage5_key_registry="var/security/key_rotation.json",
    )

    monkeypatch.setenv("METRICS_TOKEN", "x" * 32)
    monkeypatch.setenv("SCHED_TOKEN", "y" * 32)
    monkeypatch.setenv("LIVE_ROUTER_HMAC", "z" * 48)

    exit_code = audit_stage4_compliance.run(
        [
            "--config",
            str(config_path),
            "--profile",
            "stage5",
            "--mtls-bundle-name",
            _DEF_BUNDLE,
        ]
    )

    assert exit_code == 1
