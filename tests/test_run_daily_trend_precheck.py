"""Testy pomocniczej integracji paper_precheck w run_daily_trend."""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
