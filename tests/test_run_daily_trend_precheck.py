"""Testy pomocniczej integracji paper_precheck w run_daily_trend."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.reporting.audit import PaperSmokeJsonSyncResult
from bot_core.reporting.upload import SmokeArchiveUploadResult
from scripts import run_daily_trend  # noqa: E402  - import po modyfikacji sys.path


def test_run_paper_precheck_skip(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=run_daily_trend._LOGGER.name)

    payload, exit_code, metadata = run_daily_trend._run_paper_precheck_for_smoke(
        config_path=Path("config/core.yaml"),
        environment="binance_paper",
        fail_on_warnings=False,
        skip=True,
    )

    assert exit_code == 0
    assert metadata is None
    assert isinstance(payload, dict)
    assert payload["status"] == "skipped"
    assert "Pomijam automatyczny paper_precheck" in caplog.text


def test_run_paper_precheck_success(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.INFO, logger=run_daily_trend._LOGGER.name)

    expected_payload = {
        "status": "ok",
        "coverage_status": "ok",
        "risk_status": "ok",
        "coverage_warnings": [],
        "config": {"warnings": []},
    }

    monkeypatch.setattr(
        run_daily_trend.paper_precheck_cli,
        "run_precheck",
        lambda **kwargs: (expected_payload, 0),
    )

    payload, exit_code, metadata = run_daily_trend._run_paper_precheck_for_smoke(
        config_path=tmp_path / "core.yaml",
        environment="binance_paper",
        fail_on_warnings=False,
        skip=False,
    )

    assert exit_code == 0
    assert payload == expected_payload
    assert metadata is None
    assert "Paper pre-check zakończony statusem ok" in caplog.text
    assert "audit_record" not in payload


def test_run_paper_precheck_failure(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, tmp_path: Path) -> None:
    caplog.set_level(logging.ERROR, logger=run_daily_trend._LOGGER.name)

    failing_payload = {
        "status": "error",
        "coverage_status": "error",
        "risk_status": "warning",
    }

    monkeypatch.setattr(
        run_daily_trend.paper_precheck_cli,
        "run_precheck",
        lambda **kwargs: (failing_payload, 3),
    )

    payload, exit_code, metadata = run_daily_trend._run_paper_precheck_for_smoke(
        config_path=tmp_path / "core.yaml",
        environment="binance_paper",
        fail_on_warnings=False,
        skip=False,
    )

    assert exit_code == 3
    assert payload == failing_payload
    assert metadata is None
    assert "Paper pre-check zakończony niepowodzeniem" in caplog.text


def test_run_paper_precheck_persists_report(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=run_daily_trend._LOGGER.name)

    expected_payload = {
        "status": "warning",
        "coverage_status": "warning",
        "risk_status": "ok",
        "coverage_warnings": ["manifest_missing"],
    }

    def _fake_run_precheck(**kwargs):  # type: ignore[unused-argument]
        return expected_payload, 0

    run_daily_trend.paper_precheck_cli.run_precheck, original = (  # type: ignore[attr-defined]
        _fake_run_precheck,
        run_daily_trend.paper_precheck_cli.run_precheck,
    )
    try:
        payload, exit_code, metadata = run_daily_trend._run_paper_precheck_for_smoke(
            config_path=tmp_path / "core.yaml",
            environment="binance paper ",
            fail_on_warnings=False,
            skip=False,
            audit_dir=tmp_path / "audit",
            audit_clock=lambda: datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        )
    finally:
        run_daily_trend.paper_precheck_cli.run_precheck = original  # type: ignore[attr-defined]

    assert exit_code == 0
    assert payload is expected_payload
    assert metadata is not None
    audit_record = payload.get("audit_record")
    assert isinstance(audit_record, dict)
    saved_path = Path(audit_record["path"])
    assert saved_path.exists()
    assert saved_path.read_text(encoding="utf-8").strip().startswith("{")
    assert audit_record["sha256"]
    assert metadata == audit_record
    assert "paper_precheck zapisany" in caplog.text


def test_append_smoke_audit_entry_appends_row(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=run_daily_trend._LOGGER.name)

    log_template = "\n".join(
        [
            "# Log audytu – Paper trading",
            "",
            "## Sekcja B1 – Smoke testy paper tradingu",
            "| ID | Data (UTC) | Operator | Środowisko | Zakres dat | Raport (`summary.json`) | Hash SHA-256 | Status alertów | Uwagi |",
            "|----|------------|----------|------------|------------|-------------------------|--------------|----------------|-------|",
            "| S-0010 | 2025-09-30T17:19:30Z | CI Agent | binance_paper | 2024-01-01 → 2024-02-15 | `/tmp/smoke/summary.json` | `abc` | WARN | - |",
            "",
            "## Sekcja C – Incydenty",
        ]
    )
    log_path = tmp_path / "paper_trading_log.md"
    log_path.write_text(log_template + "\n", encoding="utf-8")

    precheck_metadata = {
        "path": str(tmp_path / "audit" / "20250102T030405Z_binance_paper.json"),
        "sha256": "0123456789abcdef",
        "created_at": "2025-01-02T03:04:05Z",
    }

    new_id = run_daily_trend._append_smoke_audit_entry(
        log_path=log_path,
        timestamp=datetime(2025, 1, 2, 3, 4, 6, tzinfo=timezone.utc),
        operator="QA Operator",
        environment="binance_paper",
        window={"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
        summary_path=tmp_path / "reports" / "summary.json",
        summary_sha256="deadbeef",
        severity="info",
        precheck_metadata=precheck_metadata,
        precheck_status="ok",
        precheck_coverage_status="ok",
        precheck_risk_status="warning",
    )

    assert new_id == "S-0011"
    content = log_path.read_text(encoding="utf-8")
    assert "S-0011" in content
    lines = content.splitlines()
    inserted_line = next(line for line in lines if "S-0011" in line)
    assert "paper_precheck_report=`" in inserted_line
    assert "paper_precheck_sha256=`0123456789abcdef`" in inserted_line
    assert "paper_precheck_status=ok" in inserted_line
    assert "paper_precheck_risk_status=warning" in inserted_line
    assert inserted_line.endswith("|")
    assert "Dodano wpis S-0011" in caplog.text


def test_append_smoke_json_log_entry_appends_record(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=run_daily_trend._LOGGER.name)

    summary_path = tmp_path / "reports" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("{}", encoding="utf-8")

    json_log_path = tmp_path / "audit" / "paper_trading_log.jsonl"
    precheck_metadata = {
        "path": str(tmp_path / "audit" / "20250102T030405Z_binance_paper.json"),
        "sha256": "0123456789abcdef",
        "created_at": "2025-01-02T03:04:05Z",
    }
    precheck_payload = {
        "status": "ok",
        "coverage_status": "ok",
        "risk_status": "warning",
        "warnings": ["risk_buffer_low"],
    }

    record = run_daily_trend._append_smoke_json_log_entry(
        json_path=json_log_path,
        timestamp=datetime(2025, 1, 2, 3, 4, 6, tzinfo=timezone.utc),
        operator="QA Operator",
        environment="binance_paper",
        window={"start": "2025-01-01T00:00:00Z", "end": "2025-01-02T00:00:00Z"},
        summary_path=summary_path,
        summary_sha256="deadbeefcafebabe",
        severity="info",
        precheck_metadata=precheck_metadata,
        precheck_payload=precheck_payload,
        precheck_status="ok",
        precheck_coverage_status="ok",
        precheck_risk_status="warning",
        markdown_entry_id="S-0011",
    )

    assert record is not None
    assert json_log_path.exists()
    lines = json_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["record_id"] == record["record_id"]
    assert parsed["markdown_entry_id"] == "S-0011"
    assert parsed["precheck_metadata"]["report_sha256"] == "0123456789abcdef"
    assert parsed["precheck_payload"]["risk_status"] == "warning"
    assert any("paper_precheck_status=ok" in note for note in record["notes"])
    assert "Dodano wpis JSON smoke testu" in caplog.text


def test_sync_smoke_json_log_invokes_synchronizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    json_path = tmp_path / "log.jsonl"
    json_path.write_text("{}\n", encoding="utf-8")

    called: dict[str, object] = {}

    class DummyResult:
        backend = "local"
        location = "/audit/dest.jsonl"
        metadata = {"log_sha256": "abc123", "verified": "true"}

    class DummySynchronizer:
        def __init__(self, config, *, secret_manager=None):  # type: ignore[no-untyped-def]
            called["config"] = config
            called["secret_manager"] = secret_manager

        def sync(self, json_log_path, *, environment, record_id, timestamp):  # type: ignore[no-untyped-def]
            called["args"] = (json_log_path, environment, record_id, timestamp)
            return DummyResult()

    monkeypatch.setattr(run_daily_trend, "PaperSmokeJsonSynchronizer", DummySynchronizer)

    config_object = object()
    timestamp = datetime(2025, 1, 2, 3, 4, 6, tzinfo=timezone.utc)
    result = run_daily_trend._sync_smoke_json_log(
        json_sync_cfg=config_object,
        json_log_path=json_path,
        environment="binance_paper",
        record_id="J-0001",
        timestamp=timestamp,
        secret_manager=None,
    )

    assert isinstance(result, DummyResult)
    assert called["config"] is config_object
    assert called["args"][0] == json_path
    assert called["args"][1] == "binance_paper"
    assert called["args"][2] == "J-0001"
    assert called["args"][3] == timestamp


def test_sync_smoke_json_log_handles_missing_config(tmp_path: Path) -> None:
    json_path = tmp_path / "log.jsonl"
    assert run_daily_trend._sync_smoke_json_log(
        json_sync_cfg=None,
        json_log_path=json_path,
        environment="binance_paper",
        record_id="J-0001",
        timestamp=datetime.now(timezone.utc),
        secret_manager=None,
    ) is None


def test_build_smoke_summary_payload_includes_metadata(tmp_path: Path) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    summary_path = report_dir / "summary.json"
    summary_path.write_text("{}", encoding="utf-8")
    json_log_path = tmp_path / "logs" / "paper.jsonl"

    json_record = {
        "record_id": "J-20250102T030405-deadbeef",
        "summary_path": str(summary_path),
        "severity": "INFO",
    }

    json_sync_result = PaperSmokeJsonSyncResult(
        backend="local",
        location="audit/logs/paper.jsonl",
        metadata={"verified": "true", "version_id": "v1"},
    )

    archive_upload_result = SmokeArchiveUploadResult(
        backend="local",
        location="audit/archive/report.zip",
        metadata={"acknowledged": "true", "ack_request_id": "req-123"},
    )

    publish_payload = {
        "status": "ok",
        "exit_code": 0,
        "json_sync": {
            "status": "skipped",
            "backend": "local",
            "metadata": {"note": "already_synced"},
        },
        "archive_upload": {
            "status": "ok",
            "backend": "local",
            "metadata": {"version_id": "v2"},
        },
    }

    payload = run_daily_trend._build_smoke_summary_payload(
        environment="binance_paper",
        timestamp=datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
        operator="CI Agent",
        window={"start": "2024-01-01", "end": "2024-01-31"},
        report_dir=report_dir,
        summary_path=summary_path,
        summary_sha256="abc123",
        severity="info",
        storage_context={"storage_status": "ok"},
        precheck_status="ok",
        precheck_coverage_status="ok",
        precheck_risk_status="warning",
        precheck_payload={"status": "ok"},
        json_log_path=json_log_path,
        json_record=json_record,
        json_sync_result=json_sync_result,
        archive_path=report_dir / "archive.zip",
        archive_upload_result=archive_upload_result,
        publish_result=publish_payload,
    )

    assert payload["report"]["summary_sha256"] == "abc123"
    assert payload["precheck"]["risk_status"] == "warning"
    assert payload["json_log"]["record_id"] == "J-20250102T030405-deadbeef"
    assert payload["json_log"]["sync"]["metadata"]["version_id"] == "v1"
    assert payload["archive"]["upload"]["metadata"]["ack_request_id"] == "req-123"
    assert payload["publish"]["status"] == "ok"
    assert payload["publish"]["exit_code"] == 0
    assert payload["publish"]["json_sync"]["status"] == "skipped"
    assert payload["publish"]["archive_upload"]["metadata"]["version_id"] == "v2"


def test_write_smoke_summary_json_creates_file(tmp_path: Path) -> None:
    output_path = tmp_path / "out" / "smoke_summary.json"
    payload = {"status": "ok", "details": {"value": 1}}

    run_daily_trend._write_smoke_summary_json(output_path, payload)

    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded == payload


def test_auto_publish_smoke_artifacts_invokes_script(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    summary_json = tmp_path / "summary_struct.json"
    summary_json.write_text("{}", encoding="utf-8")
    archive_path = tmp_path / "report.zip"
    archive_path.write_bytes(b"zip")
    json_log_path = tmp_path / "log.jsonl"
    json_log_path.write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "core.yaml"
    config_path.write_text("{}", encoding="utf-8")

    called: dict[str, object] = {}

    class DummyCompleted:
        returncode = 0
        stdout = json.dumps(
            {
                "status": "ok",
                "json_sync": {"status": "skipped", "backend": "local", "metadata": {"note": "cached"}},
                "archive_upload": {
                    "status": "ok",
                    "backend": "local",
                    "metadata": {"ack_request_id": "req-1"},
                },
            }
        )
        stderr = ""

    def fake_run(cmd, *, capture_output, text, check):  # type: ignore[no-untyped-def]
        called["cmd"] = cmd
        assert capture_output and text and not check
        return DummyCompleted()

    monkeypatch.setattr(run_daily_trend.subprocess, "run", fake_run)

    exit_code, payload = run_daily_trend._auto_publish_smoke_artifacts(
        config_path=config_path,
        environment="binance_paper",
        report_dir=report_dir,
        json_log_path=json_log_path,
        summary_json_path=summary_json,
        archive_path=archive_path,
        record_id="J-0001",
        skip_json_sync=True,
        skip_archive_upload=False,
        dry_run=False,
    )

    assert exit_code == 0
    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    cmd = called["cmd"]
    assert "--summary-json" in cmd
    assert "--skip-json-sync" in cmd
    assert "--archive" in cmd


def test_normalize_publish_result_sets_required_and_exit_code() -> None:
    payload = run_daily_trend._normalize_publish_result(
        {"status": "ok"}, exit_code=0, required=True
    )

    assert payload["status"] == "ok"
    assert payload["exit_code"] == 0
    assert payload["required"] is True


def test_normalize_publish_result_handles_none() -> None:
    payload = run_daily_trend._normalize_publish_result(
        None, exit_code=None, required=False
    )

    assert payload["status"] == "unknown"
    assert "exit_code" in payload
    assert payload["required"] is False


def test_is_publish_result_ok_detects_failure() -> None:
    assert run_daily_trend._is_publish_result_ok({"status": "ok", "exit_code": 0})
    assert not run_daily_trend._is_publish_result_ok({"status": "ok", "exit_code": 1})
    assert not run_daily_trend._is_publish_result_ok({"status": "skipped", "exit_code": 0})
