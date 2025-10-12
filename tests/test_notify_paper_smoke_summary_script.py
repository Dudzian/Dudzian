import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bot_core.alerts import AlertMessage
from scripts import notify_paper_smoke_summary as cli  # noqa: E402 - import po modyfikacji sys.path


@dataclass
class _RecordedAuditLog:
    records: list[Mapping[str, str]]

    def append(self, message: AlertMessage, *, channel: str) -> None:  # noqa: D401 - zgodnie z protokołem
        self.records.append({"category": message.category, "channel": channel})

    def export(self) -> Iterable[Mapping[str, str]]:  # noqa: D401 - zgodnie z protokołem
        return list(self.records)


class _RecorderRouter:
    def __init__(self) -> None:
        self.messages: list[AlertMessage] = []

    def register(self, _channel):  # noqa: ANN001 - interfejs nieużywany w testach
        raise NotImplementedError

    def dispatch(self, message: AlertMessage) -> None:  # noqa: D401 - zgodnie z protokołem
        self.messages.append(message)


class _DummySecretManager:
    def __init__(self) -> None:
        self.loaded = True


def _write_summary(tmp_path: Path, **overrides: object) -> Path:
    manifest_payload = {
        "manifest_path": str(tmp_path / "manifest" / "ohlcv_manifest.sqlite"),
        "metrics_path": str(tmp_path / "manifest" / "metrics.prom"),
        "summary_path": str(tmp_path / "manifest" / "summary.json"),
        "worst_status": "ok",
        "status_counts": {"ok": 5},
        "total_entries": 5,
        "exit_code": 0,
        "stage": "paper",
        "risk_profile": "balanced",
        "deny_status": ["warning", "invalid_metadata"],
        "summary": {"status_counts": {"ok": 5}},
        "summary_signature": {
            "algorithm": "HMAC-SHA256",
            "value": "YWJj",
            "key_id": "ci-manifest",
        },
    }
    payload = {
        "environment": "binance_paper",
        "operator": "CI Agent",
        "severity": "info",
        "timestamp": "2024-02-01T00:00:00Z",
        "window": {"start": "2024-02-01", "end": "2024-02-02"},
        "report": {
            "directory": str(tmp_path / "reports"),
            "summary_path": str(tmp_path / "reports" / "summary.json"),
            "summary_sha256": "deadbeef",
        },
        "precheck": {"status": "ok", "coverage_status": "ok", "risk_status": "ok"},
        "json_log": {
            "path": str(tmp_path / "audit" / "paper.jsonl"),
            "record_id": "2024-02-01T00:00:00Z",
            "sync": {
                "backend": "s3",
                "location": "s3://audit/jsonl",
                "metadata": {"version_id": "v1"},
            },
        },
        "archive": {
            "path": str(tmp_path / "archive.zip"),
            "upload": {
                "backend": "s3",
                "location": "s3://audit/archive.zip",
                "metadata": {"ack_request_id": "req-1"},
            },
        },
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok", "backend": "s3", "location": "s3://audit/jsonl"},
            "archive_upload": {"status": "ok", "backend": "s3", "location": "s3://audit/archive.zip"},
        },
        "manifest": manifest_payload,
        "tls_audit": {
            "report_path": str(tmp_path / "tls" / "audit.json"),
            "status": "warning",
            "exit_code": 0,
            "warnings": ["MetricsService działa bez TLS"],
            "errors": [],
        },
        "token_audit": {
            "report_path": str(tmp_path / "rbac" / "audit.json"),
            "status": "ok",
            "exit_code": 0,
            "warnings": ["Legacy auth token"],
            "errors": [],
        },
    }
    manifest_override = overrides.pop("manifest", None)
    payload.update(overrides)
    if isinstance(manifest_override, Mapping):
        payload["manifest"].update(manifest_override)
    elif manifest_override is None:
        payload["manifest"] = manifest_payload
    else:
        payload["manifest"] = manifest_override
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_main_dispatches_alert(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    summary_path = _write_summary(tmp_path)

    recorded_router = _RecorderRouter()
    recorded_audit = _RecordedAuditLog(records=[])

    def _fake_build_alert_channels(**_kwargs):  # noqa: ANN003
        return {}, recorded_router, recorded_audit

    monkeypatch.setattr(cli, "build_alert_channels", _fake_build_alert_channels)
    monkeypatch.setattr(cli, "_create_secret_manager", lambda args: _DummySecretManager())
    monkeypatch.setattr(cli, "load_core_config", lambda _path: types.SimpleNamespace(environments={"binance_paper": object()}))

    exit_code = cli.main(
        [
            "--config",
            str(tmp_path / "core.yaml"),
            "--environment",
            "binance_paper",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert len(recorded_router.messages) == 1
    message = recorded_router.messages[0]
    assert message.category == "paper_smoke_compliance"
    assert message.severity == "warning"
    assert message.context["summary_sha256"] == "deadbeef"
    assert message.context["paper_smoke_publish_status"] == "ok"
    assert message.context["paper_smoke_manifest_status"] == "ok"
    assert message.context["paper_smoke_manifest_exit_code"] == "0"
    assert message.context["paper_smoke_manifest_status_count_ok"] == "5"
    assert message.context["paper_smoke_manifest_signature"] == "YWJj"
    assert message.context["paper_smoke_manifest_signature_algorithm"] == "HMAC-SHA256"
    assert message.context["paper_smoke_manifest_signature_key_id"] == "ci-manifest"
    assert message.context["paper_smoke_tls_audit_status"] == "warning"
    assert message.context["paper_smoke_tls_audit_exit_code"] == "0"
    assert message.context["paper_smoke_tls_audit_warning_count"] == "1"
    assert message.context["paper_smoke_token_audit_status"] == "ok"
    assert message.context["paper_smoke_token_audit_exit_code"] == "0"
    assert message.context["paper_smoke_token_audit_warning_count"] == "1"


def test_main_dry_run_prints_preview(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = _write_summary(tmp_path, publish={"status": "warning", "required": False})

    exit_code = cli.main(
        [
            "--summary-json",
            str(summary_path),
            "--dry-run",
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["status"] == "dry-run"
    assert payload["severity"] == "warning"
    assert payload["context"]["paper_smoke_publish_status"] == "warning"
    assert payload["context"]["paper_smoke_manifest_status"] == "ok"
    assert payload["context"]["paper_smoke_manifest_signature"] == "YWJj"
    assert payload["context"]["paper_smoke_tls_audit_status"] == "warning"
    assert payload["context"]["paper_smoke_token_audit_status"] == "ok"


def test_manifest_failure_escalates_severity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    summary_path = _write_summary(
        tmp_path,
        manifest={
            "worst_status": "missing_metadata",
            "status_counts": {"missing_metadata": 2},
            "total_entries": 2,
            "exit_code": 2,
        },
    )

    recorded_router = _RecorderRouter()
    recorded_audit = _RecordedAuditLog(records=[])

    def _fake_build_alert_channels(**_kwargs):  # noqa: ANN003
        return {}, recorded_router, recorded_audit

    monkeypatch.setattr(cli, "build_alert_channels", _fake_build_alert_channels)
    monkeypatch.setattr(cli, "_create_secret_manager", lambda args: _DummySecretManager())
    monkeypatch.setattr(
        cli,
        "load_core_config",
        lambda _path: types.SimpleNamespace(environments={"binance_paper": object()}),
    )

    exit_code = cli.main(
        [
            "--config",
            str(tmp_path / "core.yaml"),
            "--environment",
            "binance_paper",
            "--summary-json",
            str(summary_path),
        ]
    )

    assert exit_code == 0
    assert recorded_router.messages[0].severity == "error"
    assert (
        recorded_router.messages[0].context["paper_smoke_manifest_status"]
        == "missing_metadata"
    )


def test_main_returns_error_when_summary_missing(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    exit_code = cli.main(["--summary-json", str(missing_path)])
    assert exit_code == 2

