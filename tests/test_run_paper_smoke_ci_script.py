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

import pytest
import yaml

from scripts import publish_paper_smoke_artifacts, run_paper_smoke_ci, validate_paper_smoke_summary


class _FakeCompleted:
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _fake_subprocess_run_factory(
    *,
    tmp_path: Path,
    summary_payload: dict,
    validator_stdout: str | None = None,
    validator_returncode: int = 0,
):
    """Tworzy atrapę subprocess.run obsługującą run_daily_trend i walidator."""

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
    }
    config_path = tmp_path / "core.yaml"
    config_path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
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

    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

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


def test_main_propagates_non_zero_exit(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

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
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

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
    markdown = markdown_path.read_text(encoding="utf-8")
    assert markdown.startswith("# Raport CI")


def test_main_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

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


def test_main_allows_optional_publish(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

    summary_payload = {
        "status": "ok",
        "environment": "binance_paper",
        "timestamp": "2024-01-01T00:00:00Z",
        "operator": "CI Agent",
        "severity": "info",
        "window": {"start": "2024-01-01", "end": "2024-01-02"},
    }

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
            "--allow-auto-publish-failure",
        ]
    )

    assert exit_code == 0


def test_main_renders_markdown_when_requested(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    run_daily_trend = scripts_dir / "run_daily_trend.py"
    run_daily_trend.write_text("print('stub run')", encoding="utf-8")

    config_path = tmp_path / "config.yaml"
    config_path.write_text("core: {}", encoding="utf-8")

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

        raise AssertionError(f"Unexpected command: {cmd}")

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

        raise AssertionError(f"Unexpected command: {cmd}")

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
