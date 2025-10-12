"""Testy dla skryptu run_paper_smoke_ci.py."""
from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import yaml

from scripts import publish_paper_smoke_artifacts, run_paper_smoke_ci, validate_paper_smoke_summary


class _FakeCompleted:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _format_env(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    return str(value)


def _fake_subprocess_run_factory(
    *,
    tmp_path: Path,
    summary_payload: dict,
    validator_stdout: str | None = None,
    validator_returncode: int = 0,
    verify_returncode: int = 0,
    bundle_calls: list[dict] | None = None,
    manifest_returncode: int = 0,
    manifest_summary: dict | None = None,
    manifest_calls: list[dict] | None = None,
    tls_returncode: int = 0,
    tls_report: dict | None = None,
    tls_calls: list[dict] | None = None,
    token_returncode: int = 0,
    token_report: dict | None = None,
    token_calls: list[dict] | None = None,
):
    """Tworzy atrapę subprocess.run obsługującą run_daily_trend i walidator."""

    telemetry_overrides = {
        "ui_alerts_reduce_mode": "enable",
        "ui_alerts_overlay_mode": "enable",
        "ui_alerts_jank_mode": "enable",
        "ui_alerts_reduce_active_severity": "notice",
        "ui_alerts_overlay_exceeded_severity": "notice",
        "ui_alerts_jank_spike_severity": "notice",
        "ui_alerts_overlay_critical_threshold": 2,
    }
    telemetry_summary = {
        "summary": {
            "total_snapshots": 1,
            "events": {"reduce_motion": {"count": 1}},
        },
        "metadata": {
            "mode": "jsonl",
            "summary_enabled": True,
            "risk_profile_summary": {
                "name": "balanced",
                "recommended_overrides": telemetry_overrides,
            },
        },
    }

    default_manifest_summary = {
        "status_counts": {"ok": 2},
        "total_entries": 2,
        "worst_status": "ok",
        "generated_at": "2025-01-01T00:00:00+00:00",
        "manifest_path": str(tmp_path / "cache" / "ohlcv_manifest.sqlite"),
        "environment": "binance_paper",
        "exchange": "binance_spot",
    }
    manifest_payload = manifest_summary or default_manifest_summary

    default_tls_report = tls_report or {
        "services": {
            "metrics_service": {
                "enabled": True,
                "auth_token_configured": False,
                "tls": {"enabled": False},
                "warnings": ["MetricsService działa bez TLS – rozważ włączenie szyfrowania"],
                "errors": [],
            },
            "risk_service": {
                "enabled": True,
                "auth_token_configured": False,
                "tls": {"enabled": False},
                "warnings": ["RiskService działa bez TLS – rozważ włączenie szyfrowania"],
                "errors": [],
            },
        },
        "warnings": [
            "MetricsService działa bez TLS – rozważ włączenie szyfrowania",
            "RiskService działa bez TLS – rozważ włączenie szyfrowania",
        ],
        "errors": [],
    }

    default_token_report = token_report or {
        "services": [
            {
                "service": "metrics_service",
                "status": "ok",
                "findings": [],
                "coverage": {"metrics.read": ["metrics-reader"]},
                "required_scopes": {"metrics.read": ["metrics.read"]},
            },
            {
                "service": "risk_service",
                "status": "ok",
                "findings": [],
                "coverage": {"risk.read": ["risk-reader"]},
                "required_scopes": {"risk.read": ["risk.read"]},
            },
        ],
        "warnings": [],
        "errors": [],
    }

    def _run(cmd, *_, **kwargs):  # noqa: ANN001
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script == "run_daily_trend.py":
            summary_arg = cmd.index("--paper-smoke-summary-json")
            summary_path = Path(cmd[summary_arg + 1])
            summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")
            return _FakeCompleted(returncode=0)
        if script == "validate_paper_smoke_summary.py":
            assert "--require-publish-success" in cmd
            assert "--require-publish-required" in cmd
            assert "--require-publish-exit-zero" in cmd
            stdout = validator_stdout or json.dumps({"status": "ok"})
            return _FakeCompleted(stdout=stdout, returncode=validator_returncode)
        if script == "watch_metrics_stream.py":
            summary_arg = cmd.index("--summary-output")
            summary_path = Path(cmd[summary_arg + 1])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(telemetry_summary), encoding="utf-8")
            log_arg = cmd.index("--decision-log")
            log_path = Path(cmd[log_arg + 1])
            log_path.parent.mkdir(parents=True, exist_ok=True)
            entries = [
                {"kind": "metadata", "metadata": {"mode": "jsonl", "summary_enabled": True}},
                {
                    "kind": "snapshot",
                    "timestamp": "2025-01-01T00:00:01Z",
                    "event": "reduce_motion",
                    "severity": "notice",
                    "source": "jsonl",
                    "screen": {"index": 0, "name": "Desk"},
                },
            ]
            log_path.write_text(
                "\n".join(json.dumps(entry, ensure_ascii=False) for entry in entries) + "\n",
                encoding="utf-8",
            )
            return _FakeCompleted(returncode=0)
        if script == "telemetry_risk_profiles.py":
            if "bundle" in cmd:
                output_arg = cmd.index("--output-dir")
                output_dir = Path(cmd[output_arg + 1])
                output_dir.mkdir(parents=True, exist_ok=True)
                stage_entries = [cmd[idx + 1] for idx, token in enumerate(cmd) if token == "--stage"]
                stage_map = {"demo": "conservative", "paper": "balanced", "live": "manual"}
                for entry in stage_entries:
                    stage, profile = entry.split("=", 1)
                    stage_map[stage.strip().lower()] = profile.strip().lower()
                if bundle_calls is not None:
                    bundle_calls.append({"cmd": list(cmd), "stage_map": dict(stage_map)})
                env_style = "dotenv"
                if "--env-style" in cmd:
                    env_style = cmd[cmd.index("--env-style") + 1]
                config_format = "yaml"
                if "--config-format" in cmd:
                    config_format = cmd[cmd.index("--config-format") + 1]
                stages_payload = []
                for stage_name, profile_name in stage_map.items():
                    stage_dir = output_dir / stage_name
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    env_path = stage_dir / "metrics.env"
                    env_path.write_text(f"PROFILE={profile_name}\n", encoding="utf-8")
                    config_path = stage_dir / "metrics.yaml"
                    config_path.write_text(
                        json.dumps({"risk_profile": profile_name}, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                    stages_payload.append(
                        {
                            "stage": stage_name,
                            "risk_profile": profile_name,
                            "risk_profile_summary": {"name": profile_name},
                            "paths": {"env": str(env_path), "config": str(config_path)},
                        }
                    )
                manifest = {
                    "output_dir": str(output_dir),
                    "env_style": env_style,
                    "config_format": config_format,
                    "stages": stages_payload,
                    "manifest_path": str(output_dir / "manifest.json"),
                }
                manifest_path = output_dir / "manifest.json"
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
                return _FakeCompleted(stdout=json.dumps(manifest, ensure_ascii=False))

            output_arg = cmd.index("--output")
            output_path = Path(cmd[output_arg + 1])
            fmt_arg = cmd.index("--format")
            fmt = cmd[fmt_arg + 1]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if fmt == "env":
                lines = [
                    f"RUN_TRADING_STUB_METRICS_{key.upper()}={_format_env(value)}"
                    for key, value in telemetry_overrides.items()
                ]
                output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            elif fmt == "yaml":
                payload = {
                    "summary": telemetry_summary["metadata"]["risk_profile_summary"],
                    "metrics_service_overrides": telemetry_overrides,
                    "env_assignments": [
                        f"RUN_TRADING_STUB_METRICS_{key.upper()}={_format_env(value)}"
                        for key, value in telemetry_overrides.items()
                    ],
                }
                output_path.write_text(
                    json.dumps(payload, ensure_ascii=False),
                    encoding="utf-8",
                )
            else:
                raise AssertionError(f"Unexpected format in telemetry renderer: {fmt}")
            return _FakeCompleted(returncode=0)
        if script == "verify_decision_log.py":
            report_arg = cmd.index("--report-output")
            report_path = Path(cmd[report_arg + 1])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "report_version": 1,
                "summary": telemetry_summary["summary"],
                "risk_profile_summary": telemetry_summary["metadata"]["risk_profile_summary"],
                "risk_profile_snippet_validation": [
                    {"type": "env", "status": "ok"},
                    {"type": "yaml", "status": "ok"},
                ],
            }
            report_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return _FakeCompleted(returncode=verify_returncode)
        if script == "export_manifest_metrics.py":
            output_arg = cmd.index("--output")
            metrics_path = Path(cmd[output_arg + 1])
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text("# HELP ohlcv_manifest_gap_minutes ...\n", encoding="utf-8")
            summary_arg = cmd.index("--summary-output")
            summary_path = Path(cmd[summary_arg + 1])
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(manifest_payload, ensure_ascii=False), encoding="utf-8")
            if manifest_calls is not None:
                manifest_calls.append({"cmd": list(cmd)})
            return _FakeCompleted(returncode=manifest_returncode)
        if script == "audit_tls_assets.py":
            if tls_calls is not None:
                tls_calls.append({"cmd": list(cmd)})
            json_arg = cmd.index("--json-output")
            report_path = Path(cmd[json_arg + 1])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(default_tls_report, ensure_ascii=False), encoding="utf-8")
            return _FakeCompleted(stdout=json.dumps(default_tls_report, ensure_ascii=False), returncode=tls_returncode)
        if script == "audit_service_tokens.py":
            if token_calls is not None:
                token_calls.append({"cmd": list(cmd)})
            json_arg = cmd.index("--json-output")
            report_path = Path(cmd[json_arg + 1])
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(default_token_report, ensure_ascii=False), encoding="utf-8")
            return _FakeCompleted(stdout=json.dumps(default_token_report, ensure_ascii=False), returncode=token_returncode)
        raise AssertionError(f"Unexpected command: {cmd}")

    return _run


def _write_core_config(tmp_path: Path, *, reporting: dict) -> Path:
    payload = {
        "risk_profiles": {
            "balanced": {
                "max_daily_loss_pct": 0.02,
                "max_position_pct": 0.05,
                "target_volatility": 0.1,
                "max_leverage": 3.0,
                "stop_loss_atr_multiple": 1.5,
                "max_open_positions": 5,
                "hard_drawdown_pct": 0.1,
            }
        },
        "environments": {
            "binance_paper": {
                "exchange": "binance_spot",
                "environment": "paper",
                "keychain_key": "binance_paper_key",
                "data_cache_path": str(tmp_path / "cache"),
                "risk_profile": "balanced",
                "alert_channels": [],
            }
        },
        "reporting": reporting,
        "runtime": {
            "metrics_service": {
                "enabled": True,
                "ui_alerts_jsonl_path": str(tmp_path / "metrics" / "ui_alerts.jsonl"),
                "ui_alerts_risk_profile": "balanced",
            }
        },
    }
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    metrics_path = tmp_path / "metrics" / "ui_alerts.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps({"kind": "snapshot", "event": "reduce_motion", "severity": "notice"}) + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "cache" / "ohlcv_manifest.sqlite"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("stub", encoding="utf-8")
    return config_path


def test_build_command_creates_directories(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text("core: {}", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    run_daily_trend = tmp_path / "scripts" / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub')", encoding="utf-8")

    command, paths = run_paper_smoke_ci._build_command(
        config_path=config_path,
        environment="binance_paper",
        output_dir=tmp_path / "output",
        operator="Tester",
        auto_publish_required=True,
        extra_run_daily_trend_args=[],
    )

    assert "--paper-smoke-auto-publish-required" in command
    assert paths["summary"].parent.exists()
    assert paths["json_log"].parent.exists()
    assert paths["audit_log"].parent.exists()


def test_build_command_accepts_extra_args(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    config_path.write_text("core: {}", encoding="utf-8")
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub')", encoding="utf-8")

    command, _ = run_paper_smoke_ci._build_command(
        config_path=config_path,
        environment="binance_paper",
        output_dir=tmp_path / "output",
        operator="Tester",
        auto_publish_required=False,
        extra_run_daily_trend_args=["--date-window 2024-01-01:2024-02-01", "--run-once"],
    )

    assert command.count("--date-window") == 1
    assert "2024-01-01:2024-02-01" in command
    assert "--run-once" in command


def test_main_runs_smoke_and_prints_summary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok", "backend": "local"},
            "archive_upload": {"status": "ok", "backend": "s3"},
        },
    }

    fake_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload=summary_payload)
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    result = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert result == 0
    summary_file = tmp_path / "output" / "paper_smoke_summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary.get("validation", {}).get("status") == "ok"
    telemetry = summary.get("telemetry", {})
    assert telemetry.get("risk_profile", {}).get("name") == "balanced"
    assert telemetry.get("decision_log_report", {}).get("status") == "ok"
    telemetry_summary = telemetry.get("summary", {})
    assert telemetry_summary.get("summary", {}).get("total_snapshots") == 1
    snippets = telemetry.get("snippets", {})
    assert snippets.get("env_path")
    assert snippets.get("yaml_path")
    bundle = telemetry.get("bundle", {})
    assert bundle.get("manifest_path")
    manifest = bundle.get("manifest", {})
    assert manifest.get("stages")
    stage_names = {stage["stage"] for stage in manifest["stages"]}
    assert {"demo", "paper", "live"}.issubset(stage_names)
    manifest_section = summary.get("manifest", {})
    assert manifest_section.get("worst_status") == "ok"
    assert manifest_section.get("exit_code") == 0
    assert manifest_section.get("metrics_path")
    tls_section = summary.get("tls_audit", {})
    assert tls_section.get("status") == "warning"
    assert tls_section.get("exit_code") == 0
    assert tls_section.get("report_path")
    assert tls_section.get("warnings")
    token_section = summary.get("token_audit", {})
    assert token_section.get("status") == "ok"
    assert token_section.get("exit_code") == 0
    assert token_section.get("report_path")


def test_main_propagates_decision_log_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {"status": "ok", "required": True, "exit_code": 0},
    }

    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload=summary_payload,
        verify_returncode=5,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "decision_log_failed"
    assert payload["decision_log_exit_code"] == 5

    summary_file = tmp_path / "output" / "paper_smoke_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    telemetry = summary.get("telemetry", {})
    report = telemetry.get("decision_log_report", {})
    assert report.get("status") == "failed"
    assert report.get("exit_code") == 5

    assert exit_code == 5


def test_main_propagates_manifest_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {"status": "ok", "required": True, "exit_code": 0},
    }

    manifest_calls: list[dict] = []
    manifest_summary = {
        "status_counts": {"warning": 1},
        "total_entries": 1,
        "worst_status": "warning",
        "environment": "binance_paper",
        "manifest_path": str(tmp_path / "cache" / "ohlcv_manifest.sqlite"),
    }

    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload=summary_payload,
        verify_returncode=0,
        manifest_returncode=2,
        manifest_summary=manifest_summary,
        manifest_calls=manifest_calls,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 2
    assert manifest_calls, "export_manifest_metrics.py powinien zostać wywołany"
    cmd_tokens = manifest_calls[0]["cmd"]
    assert "--deny-status" in cmd_tokens
    summary = json.loads((tmp_path / "output" / "paper_smoke_summary.json").read_text(encoding="utf-8"))
    manifest_section = summary.get("manifest", {})
    assert manifest_section.get("worst_status") == "warning"
    assert manifest_section.get("exit_code") == 2


def test_manifest_export_uses_signing_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub run')", encoding="utf-8")

    manifest_calls: list[dict] = []
    manifest_summary = {
        "status_counts": {"ok": 1},
        "total_entries": 1,
        "worst_status": "ok",
        "generated_at": "2025-01-01T00:00:00+00:00",
        "manifest_path": str(tmp_path / "cache" / "ohlcv_manifest.sqlite"),
        "environment": "binance_paper",
        "exchange": "binance_spot",
        "summary_signature": {
            "algorithm": "HMAC-SHA256",
            "value": "dGVzdF9kaWdlc3Q=",
            "key_id": "ci-manifest",
        },
    }

    config_path = _write_core_config(
        tmp_path,
        reporting={
            "manifest_metrics": {
                "signing": {
                    "key_env": "EXPORT_MANIFEST_KEY",
                    "key_id": "ci-manifest",
                    "require": True,
                }
            }
        },
    )

    monkeypatch.setenv("EXPORT_MANIFEST_KEY", "manifest-signing-secret-123456")

    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload={
            "status": "ok",
            "environment": "binance_paper",
            "timestamp": "2024-01-01T00:00:00Z",
            "operator": "CI Agent",
            "severity": "info",
            "window": {"start": "2024-01-01", "end": "2024-01-02"},
            "publish": {"status": "ok", "required": True, "exit_code": 0},
            "report": {
                "summary_path": str(tmp_path / "summary.json"),
                "directory": str(tmp_path / "output"),
                "summary_sha256": "deadbeef",
            },
        },
        manifest_summary=manifest_summary,
        manifest_calls=manifest_calls,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)

    env_file = tmp_path / "paper.env"
    markdown_path = tmp_path / "summary.md"

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--env-file",
            str(env_file),
            "--operator",
            "CI Operator",
            "--render-summary-markdown",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    assert manifest_calls, "export_manifest_metrics.py powinien zostać wywołany"
    cmd_tokens = manifest_calls[0]["cmd"]
    assert "--summary-hmac-key-env" in cmd_tokens
    assert "EXPORT_MANIFEST_KEY" in cmd_tokens
    assert "--summary-hmac-key-id" in cmd_tokens
    assert "ci-manifest" in cmd_tokens
    assert "--require-summary-signature" in cmd_tokens

    env_content = env_file.read_text(encoding="utf-8")
    assert "PAPER_SMOKE_MANIFEST_SIGNATURE=dGVzdF9kaWdlc3Q=" in env_content
    assert "PAPER_SMOKE_MANIFEST_SIGNATURE_ALGORITHM=HMAC-SHA256" in env_content
    assert "PAPER_SMOKE_MANIFEST_SIGNATURE_KEY_ID=ci-manifest" in env_content
    assert "PAPER_SMOKE_TLS_AUDIT_PATH=" in env_content
    assert "PAPER_SMOKE_TLS_AUDIT_STATUS=warning" in env_content
    assert "PAPER_SMOKE_TLS_AUDIT_EXIT_CODE=0" in env_content
    assert "PAPER_SMOKE_TOKEN_AUDIT_PATH=" in env_content
    assert "PAPER_SMOKE_TOKEN_AUDIT_STATUS=ok" in env_content
    assert "PAPER_SMOKE_TOKEN_AUDIT_EXIT_CODE=0" in env_content


def test_main_fails_on_tls_audit_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    error_report = {
        "services": {
            "metrics_service": {
                "enabled": True,
                "auth_token_configured": True,
                "tls": {"enabled": True},
                "warnings": [],
                "errors": ["Certyfikat wygasł"],
            }
        },
        "warnings": [],
        "errors": ["Certyfikat wygasł"],
    }

    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload={
            "status": "ok",
            "environment": "binance_paper",
            "timestamp": "2024-01-01T00:00:00Z",
            "operator": "CI Agent",
            "severity": "info",
            "window": {"start": "2024-01-01", "end": "2024-01-02"},
            "publish": {"status": "ok", "required": True, "exit_code": 0},
        },
        tls_returncode=2,
        tls_report=error_report,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["tls_audit_exit_code"] == 2
    assert payload["token_audit_exit_code"] == 0
    assert payload["status"].startswith("tls_audit_failed")


def test_main_fails_on_token_audit_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    error_report = {
        "services": [
            {
                "service": "metrics_service",
                "findings": [
                    {"level": "error", "message": "Missing metrics.read", "details": {"scope": "metrics.read"}}
                ],
            }
        ],
        "warnings": [],
        "errors": ["Brak tokenów"],
    }

    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload={
            "status": "ok",
            "environment": "binance_paper",
            "timestamp": "2024-01-01T00:00:00Z",
            "operator": "CI Agent",
            "severity": "info",
            "window": {"start": "2024-01-01", "end": "2024-01-02"},
            "publish": {"status": "ok", "required": True, "exit_code": 0},
        },
        token_returncode=3,
        token_report=error_report,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 3
    assert payload["token_audit_exit_code"] == 3
    assert payload["status"].startswith("token_audit_failed")


def test_main_propagates_non_zero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = _write_core_config(tmp_path, reporting={})

    def fake_run(cmd, *_, **kwargs):  # noqa: ANN001
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script == "run_daily_trend.py":
            return _FakeCompleted(returncode=6)
        raise AssertionError(cmd)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert exit_code == 6


def test_main_writes_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Operator",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok"},
        },
    }

    fake_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload=summary_payload)
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    env_file = tmp_path / "env" / "paper_smoke.env"
    markdown_path = tmp_path / "output" / "paper_smoke.md"
    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--env-file",
            str(env_file),
            "--operator",
            "CI Operator",
            "--render-summary-markdown",
            str(markdown_path),
            "--render-summary-title",
            "Raport CI",
        ]
    )

    assert exit_code == 0
    content = env_file.read_text(encoding="utf-8")
    lines = dict(line.split("=", 1) for line in content.strip().splitlines())
    assert lines["PAPER_SMOKE_OPERATOR"] == "CI Operator"
    assert lines["PAPER_SMOKE_STATUS"] == "ok"
    assert lines["PAPER_SMOKE_PUBLISH_STATUS"] == "ok"
    assert Path(lines["PAPER_SMOKE_MARKDOWN_PATH"]) == markdown_path.resolve()
    assert lines["PAPER_SMOKE_VALIDATION_STATUS"] == "ok"
    assert lines.get("PAPER_SMOKE_RISK_BUNDLE_DIR")
    assert lines.get("PAPER_SMOKE_RISK_BUNDLE_MANIFEST_PATH")
    assert lines.get("PAPER_SMOKE_MANIFEST_PATH")
    assert lines.get("PAPER_SMOKE_MANIFEST_STATUS") == "ok"
    assert "PAPER_SMOKE_MANIFEST_METRICS_PATH" in lines
    assert lines.get("PAPER_SMOKE_MANIFEST_EXIT_CODE") == "0"
    assert lines.get("PAPER_SMOKE_MANIFEST_STAGE") == "paper"
    assert lines.get("PAPER_SMOKE_MANIFEST_RISK_PROFILE") == "balanced"
    assert lines.get("PAPER_SMOKE_TLS_AUDIT_PATH")
    assert lines.get("PAPER_SMOKE_TLS_AUDIT_STATUS") == "warning"
    assert lines.get("PAPER_SMOKE_TLS_AUDIT_EXIT_CODE") == "0"
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Raport CI")


def test_main_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = _write_core_config(tmp_path, reporting={})

    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "dry_run"
    assert exit_code == 0


def test_main_allows_stage_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "run_daily_trend.py").write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {"status": "ok", "required": True, "exit_code": 0},
    }

    bundle_calls: list[dict] = []
    fake_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path,
        summary_payload=summary_payload,
        bundle_calls=bundle_calls,
    )
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    bundle_dir = tmp_path / "custom_bundle"
    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--risk-profile-stage",
            "demo=balanced",
            "--risk-profile-stage",
            "live=manual",
            "--risk-profile-bundle-dir",
            str(bundle_dir),
            "--risk-profile-bundle-config-format",
            "json",
        ]
    )

    assert exit_code == 0
    assert bundle_calls, "expected telemetry bundler to be invoked"
    call = bundle_calls[-1]
    assert call["stage_map"]["demo"] == "balanced"
    assert call["stage_map"]["paper"] == "balanced"
    assert call["stage_map"]["live"] == "manual"
    summary = json.loads((tmp_path / "output" / "paper_smoke_summary.json").read_text(encoding="utf-8"))
    bundle = summary.get("telemetry", {}).get("bundle", {})
    assert bundle.get("output_dir") == str(bundle_dir)
    assert bundle.get("manifest", {}).get("config_format") == "json"


def test_main_allows_optional_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
    }

    base_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload=summary_payload)

    def fake_run(cmd, *_, **kwargs):  # noqa: ANN001
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script == "run_daily_trend.py":
            assert "--paper-smoke-auto-publish" in cmd
            assert "--paper-smoke-auto-publish-required" not in cmd
            summary_arg = cmd.index("--paper-smoke-summary-json")
            Path(cmd[summary_arg + 1]).write_text(json.dumps(summary_payload), encoding="utf-8")
            return _FakeCompleted(returncode=0)
        if script == "validate_paper_smoke_summary.py":
            assert "--require-publish-success" not in cmd
            assert "--require-publish-required" not in cmd
            assert "--require-publish-exit-zero" not in cmd
            return _FakeCompleted(stdout=json.dumps({"status": "ok"}), returncode=0)
        return base_run(cmd, *_, **kwargs)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--allow-auto-publish-failure",
        ]
    )

    assert exit_code == 0


def test_main_renders_markdown_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")

    config_path = _write_core_config(tmp_path, reporting={})

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok", "backend": "local"},
        },
    }

    fake_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload=summary_payload)
    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", fake_run)
    monkeypatch.chdir(tmp_path)

    markdown_path = tmp_path / "output" / "summary.md"
    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(tmp_path / "output"),
            "--render-summary-markdown",
            str(markdown_path),
            "--render-summary-title",
            "Custom title",
            "--render-summary-max-json-chars",
            "100",
        ]
    )

    assert exit_code == 0
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Custom title")
    assert "Auto-publikacja artefaktów" in markdown


def test_ci_main_end_to_end_local_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reporting_cfg = {
        "paper_smoke_json_sync": {
            "backend": "local",
            "local": {
                "directory": str(tmp_path / "json_sync"),
                "filename_pattern": "{environment}_{date}.jsonl",
                "fsync": False,
            },
        },
        "smoke_archive_upload": {
            "backend": "local",
            "local": {
                "directory": str(tmp_path / "archive_store"),
                "filename_pattern": "{environment}_{hash}.zip",
                "fsync": False,
            },
        },
    }

    config_path = _write_core_config(tmp_path, reporting=reporting_cfg)

    base_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload={"status": "ok"})

    base_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload={"status": "ok"})

    base_run = _fake_subprocess_run_factory(tmp_path=tmp_path, summary_payload={"status": "ok"})

    base_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path, summary_payload={"status": "ok"}
    )

    base_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path, summary_payload={"status": "ok"}
    )

    def _fake_run(cmd, *_, **kwargs):  # noqa: ANN001
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script == "run_daily_trend.py":
            summary_idx = cmd.index("--paper-smoke-summary-json")
            summary_path = Path(cmd[summary_idx + 1])
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            json_log_idx = cmd.index("--paper-smoke-json-log")
            json_log_path = Path(cmd[json_log_idx + 1])
            json_log_path.parent.mkdir(parents=True, exist_ok=True)

            audit_log_idx = cmd.index("--paper-smoke-audit-log")
            audit_log_path = Path(cmd[audit_log_idx + 1])
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            audit_log_path.write_text("audit-entry", encoding="utf-8")

            operator_idx = cmd.index("--paper-smoke-operator")
            operator_name = cmd[operator_idx + 1]

            config_idx = cmd.index("--config")
            config_arg = Path(cmd[config_idx + 1])

            env_idx = cmd.index("--environment")
            environment = cmd[env_idx + 1]

            report_dir = summary_path.parent / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_summary_path = report_dir / "summary.json"
            report_summary_payload = {"status": "ok", "orders": []}
            report_summary_path.write_text(
                json.dumps(report_summary_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            summary_sha = hashlib.sha256(report_summary_path.read_bytes()).hexdigest()

            record_id = "J-20250102T030405-0001"
            json_log_payload = {
                "record_id": record_id,
                "environment": environment,
                "summary_sha256": summary_sha,
                "summary_path": str(report_summary_path),
                "timestamp": "2025-01-02T03:04:05+00:00",
            }
            json_log_path.write_text(
                json.dumps(json_log_payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            structured_summary = {
                "status": "ok",
                "severity": "info",
                "environment": environment,
                "operator": operator_name,
                "timestamp": "2025-01-02T03:04:05+00:00",
                "window": {"start": "2024-01-01", "end": "2024-01-31"},
                "report": {
                    "directory": str(report_dir),
                    "summary_path": str(report_summary_path),
                    "summary_sha256": summary_sha,
                },
                "json_log": {
                    "path": str(json_log_path),
                    "record_id": record_id,
                },
            }
            summary_path.write_text(
                json.dumps(structured_summary, ensure_ascii=False),
                encoding="utf-8",
            )

            publish_args = [
                "--config",
                str(config_arg),
                "--environment",
                environment,
                "--report-dir",
                str(report_dir),
                "--json-log",
                str(json_log_path),
                "--summary-json",
                str(summary_path),
                "--json",
            ]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                publish_exit_code = publish_paper_smoke_artifacts.main(publish_args)
            publish_payload = json.loads(buf.getvalue())

            structured_summary["publish"] = {
                **publish_payload,
                "exit_code": publish_exit_code,
                "required": "--paper-smoke-auto-publish-required" in cmd,
            }
            summary_path.write_text(
                json.dumps(structured_summary, ensure_ascii=False),
                encoding="utf-8",
            )

            return _FakeCompleted(returncode=0)

        if script == "validate_paper_smoke_summary.py":
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exit_code = validate_paper_smoke_summary.main(cmd[2:])
            return _FakeCompleted(stdout=buf.getvalue(), returncode=exit_code)

        return base_run(cmd, *_, **kwargs)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", _fake_run)
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "ci_output"
    env_file = output_dir / "paper.env"

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(output_dir),
            "--operator",
            "CI Operator",
            "--env-file",
            str(env_file),
        ]
    )

    assert exit_code == 0

    summary_path = output_dir / "paper_smoke_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    publish = summary.get("publish", {})
    assert publish.get("status") == "ok"
    assert publish.get("required") is True
    assert publish.get("exit_code") == 0
    assert publish.get("json_sync", {}).get("status") == "ok"
    assert publish.get("archive_upload", {}).get("status") == "ok"
    validation = summary.get("validation", {})
    assert validation.get("status") == "ok"

    synced_files = list((tmp_path / "json_sync").glob("*.jsonl"))
    assert synced_files, "Zsynchronizowany dziennik JSONL powinien istnieć"
    archive_files = list((tmp_path / "archive_store").glob("*.zip"))
    assert archive_files, "Archiwum smoke powinno zostać przesłane do katalogu docelowego"

    env_content = env_file.read_text(encoding="utf-8")
    assert "PAPER_SMOKE_SUMMARY_PATH=" in env_content
    assert "PAPER_SMOKE_PUBLISH_STATUS=ok" in env_content
    assert "PAPER_SMOKE_TELEMETRY_SUMMARY_PATH=" in env_content
    assert "PAPER_SMOKE_DECISION_LOG_PATH=" in env_content
    assert "PAPER_SMOKE_DECISION_LOG_REPORT_PATH=" in env_content
    assert "PAPER_SMOKE_RISK_PROFILE=balanced" in env_content
    assert "PAPER_SMOKE_MANIFEST_STATUS=ok" in env_content
    assert "PAPER_SMOKE_MANIFEST_METRICS_PATH=" in env_content
    assert "PAPER_SMOKE_MANIFEST_PATH=" in env_content
    assert "PAPER_SMOKE_MANIFEST_STAGE=paper" in env_content
    assert "PAPER_SMOKE_TLS_AUDIT_STATUS=warning" in env_content
    assert "PAPER_SMOKE_TLS_AUDIT_EXIT_CODE=0" in env_content


def test_ci_main_end_to_end_s3_backends(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    reporting_cfg = {
        "paper_smoke_json_sync": {
            "backend": "s3",
            "credential_secret": "json_sync_secret",
            "s3": {
                "bucket": "json-audit",
                "object_prefix": "logs",
                "endpoint_url": "https://mock-s3.local",
                "region": "us-east-1",
                "use_ssl": False,
                "extra_args": {"ACL": "bucket-owner-full-control"},
            },
        },
        "smoke_archive_upload": {
            "backend": "s3",
            "credential_secret": "archive_secret",
            "s3": {
                "bucket": "archive-bucket",
                "object_prefix": "smoke",
                "endpoint_url": "https://mock-s3.local",
                "region": "us-east-1",
                "use_ssl": False,
                "extra_args": {"StorageClass": "STANDARD"},
            },
        },
    }

    config_path = _write_core_config(tmp_path, reporting=reporting_cfg)

    secrets = {
        "json_sync_secret": json.dumps(
            {"access_key_id": "AKIAJSONSYNC", "secret_access_key": "JSONSECRET"}
        ),
        "archive_secret": json.dumps(
            {"access_key_id": "AKIAARCHIVE", "secret_access_key": "ARCHSECRET"}
        ),
    }

    class _StubSecretManager:
        def __init__(self, payload: dict[str, str]) -> None:
            self._payload = payload

        def load_secret_value(self, keychain_key: str, *, purpose: str = "generic") -> str:
            return self._payload[keychain_key]

    monkeypatch.setattr(
        publish_paper_smoke_artifacts,
        "_create_secret_manager",
        lambda args: _StubSecretManager(secrets),
    )

    uploads: dict[tuple[str, str], dict[str, object]] = {}
    version_counter = itertools.count(1)

    class _StubS3Client:
        def upload_file(self, filename: str, bucket: str, object_key: str, ExtraArgs=None):  # noqa: N803
            payload = Path(filename).read_bytes()
            args = ExtraArgs or {}
            metadata = dict(args.get("Metadata") or {})
            if "sha256" not in metadata:
                raise AssertionError("Brak metadanych sha256 podczas przesyłania do S3")
            uploads[(bucket, object_key)] = {
                "payload": payload,
                "metadata": metadata,
                "extra_args": args,
                "version": f"ver-{next(version_counter):04d}",
            }

        def head_object(self, Bucket: str, Key: str):  # noqa: N803
            entry = uploads[(Bucket, Key)]
            return {
                "Metadata": dict(entry["metadata"]),
                "VersionId": entry["version"],
                "ResponseMetadata": {"HTTPStatusCode": 200, "RequestId": f"req-{Bucket}-{Key}"},
            }

    stub_client = _StubS3Client()

    class _StubSession:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        def client(self, service_name: str, **kwargs):
            assert service_name == "s3"
            return stub_client

    monkeypatch.setitem(
        sys.modules,
        "boto3",
        types.SimpleNamespace(session=types.SimpleNamespace(Session=_StubSession)),
    )

    base_run = _fake_subprocess_run_factory(
        tmp_path=tmp_path, summary_payload={"status": "ok"}
    )

    def _fake_run(cmd, *_, **kwargs):  # noqa: ANN001
        script = Path(cmd[1]).name if len(cmd) > 1 else ""
        if script == "run_daily_trend.py":
            summary_idx = cmd.index("--paper-smoke-summary-json")
            summary_path = Path(cmd[summary_idx + 1])
            summary_path.parent.mkdir(parents=True, exist_ok=True)

            json_log_idx = cmd.index("--paper-smoke-json-log")
            json_log_path = Path(cmd[json_log_idx + 1])
            json_log_path.parent.mkdir(parents=True, exist_ok=True)

            audit_log_idx = cmd.index("--paper-smoke-audit-log")
            audit_log_path = Path(cmd[audit_log_idx + 1])
            audit_log_path.parent.mkdir(parents=True, exist_ok=True)
            audit_log_path.write_text("audit-entry", encoding="utf-8")

            operator_idx = cmd.index("--paper-smoke-operator")
            operator_name = cmd[operator_idx + 1]

            config_idx = cmd.index("--config")
            config_arg = Path(cmd[config_idx + 1])

            env_idx = cmd.index("--environment")
            environment = cmd[env_idx + 1]

            report_dir = summary_path.parent / "report"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_summary_path = report_dir / "summary.json"
            report_summary_payload = {"status": "ok", "orders": []}
            report_summary_path.write_text(
                json.dumps(report_summary_payload, ensure_ascii=False),
                encoding="utf-8",
            )
            summary_sha = hashlib.sha256(report_summary_path.read_bytes()).hexdigest()

            record_id = "J-20250102T030405-0002"
            json_log_payload = {
                "record_id": record_id,
                "environment": environment,
                "summary_sha256": summary_sha,
                "summary_path": str(report_summary_path),
                "timestamp": "2025-01-02T03:04:05+00:00",
            }
            json_log_path.write_text(
                json.dumps(json_log_payload, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            structured_summary = {
                "status": "ok",
                "severity": "info",
                "environment": environment,
                "operator": operator_name,
                "timestamp": "2025-01-02T03:04:05+00:00",
                "window": {"start": "2024-02-01", "end": "2024-02-29"},
                "report": {
                    "directory": str(report_dir),
                    "summary_path": str(report_summary_path),
                    "summary_sha256": summary_sha,
                },
                "json_log": {
                    "path": str(json_log_path),
                    "record_id": record_id,
                },
            }
            summary_path.write_text(
                json.dumps(structured_summary, ensure_ascii=False),
                encoding="utf-8",
            )

            publish_args = [
                "--config",
                str(config_arg),
                "--environment",
                environment,
                "--report-dir",
                str(report_dir),
                "--json-log",
                str(json_log_path),
                "--summary-json",
                str(summary_path),
                "--json",
            ]

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                publish_exit_code = publish_paper_smoke_artifacts.main(publish_args)
            publish_payload = json.loads(buf.getvalue())

            structured_summary["publish"] = {
                **publish_payload,
                "exit_code": publish_exit_code,
                "required": "--paper-smoke-auto-publish-required" in cmd,
            }
            summary_path.write_text(
                json.dumps(structured_summary, ensure_ascii=False),
                encoding="utf-8",
            )

            return _FakeCompleted(returncode=0)

        if script == "validate_paper_smoke_summary.py":
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                exit_code = validate_paper_smoke_summary.main(cmd[2:])
            return _FakeCompleted(stdout=buf.getvalue(), returncode=exit_code)

        return base_run(cmd, *_, **kwargs)

    monkeypatch.setattr(run_paper_smoke_ci.subprocess, "run", _fake_run)
    monkeypatch.chdir(tmp_path)

    output_dir = tmp_path / "ci_output"
    env_file = output_dir / "paper.env"

    exit_code = run_paper_smoke_ci.main(
        [
            "--config",
            str(config_path),
            "--environment",
            "binance_paper",
            "--output-dir",
            str(output_dir),
            "--operator",
            "CI Operator",
            "--env-file",
            str(env_file),
        ]
    )

    assert exit_code == 0

    summary_path = output_dir / "paper_smoke_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    publish = summary.get("publish", {})
    assert publish.get("status") == "ok"
    assert publish.get("required") is True
    assert publish.get("exit_code") == 0
    assert publish.get("json_sync", {}).get("backend") == "s3"
    assert publish.get("archive_upload", {}).get("backend") == "s3"
    json_sync_meta = publish.get("json_sync", {}).get("metadata", {})
    archive_meta = publish.get("archive_upload", {}).get("metadata", {})
    assert json_sync_meta.get("ack_request_id", "").startswith("req-json-audit")
    assert archive_meta.get("ack_request_id", "").startswith("req-archive-bucket")
    assert json_sync_meta.get("remote_sha256") == json_sync_meta.get("log_sha256")
    assert archive_meta.get("remote_sha256") == archive_meta.get("archive_sha256")

    synced_objects = [key for key in uploads if key[0] == "json-audit"]
    archive_objects = [key for key in uploads if key[0] == "archive-bucket"]
    assert synced_objects, "Powinien powstać obiekt JSONL w magazynie S3"
    assert archive_objects, "Powinno zostać wysłane archiwum do magazynu S3"

    json_log_info = summary.get("json_log", {})
    json_log_path = Path(json_log_info["path"])
    assert json_log_path.exists()
    json_log_hash = hashlib.sha256(json_log_path.read_bytes()).hexdigest()
    uploaded_json_metadata = uploads[synced_objects[0]]["metadata"]
    assert uploaded_json_metadata.get("sha256") == json_log_hash

    env_content = env_file.read_text(encoding="utf-8")
    assert "PAPER_SMOKE_SUMMARY_PATH=" in env_content
    assert "PAPER_SMOKE_PUBLISH_STATUS=ok" in env_content
