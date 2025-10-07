"""Testy dla skryptu render_paper_smoke_summary.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import render_paper_smoke_summary


@pytest.fixture()
def sample_summary(tmp_path: Path) -> Path:
    summary = {
        "environment": "binance_paper",
        "timestamp": "2025-05-01T12:00:00+00:00",
        "operator": "CI Agent",
        "severity": "INFO",
        "window": {"start": "2025-04-01T00:00:00+00:00", "end": "2025-05-01T00:00:00+00:00"},
        "report": {
            "directory": "/tmp/report",
            "summary_path": "/tmp/report/summary.json",
            "summary_sha256": "abc123",
        },
        "storage": {"status": "ok", "free_mb": 1024.5},
        "precheck": {
            "status": "ok",
            "coverage_status": "ok",
            "risk_status": "ok",
            "payload": {"message": "all good"},
        },
        "json_log": {
            "path": "/tmp/report/log.jsonl",
            "record_id": "J-20240501",
            "record": {"record_id": "J-20240501", "status": "ok"},
            "sync": {
                "backend": "local",
                "location": "/audit/log.jsonl",
                "metadata": {"acknowledged": "true", "version_id": "1"},
            },
        },
        "archive": {
            "path": "/tmp/report/archive.zip",
            "upload": {
                "backend": "s3",
                "location": "s3://bucket/archive.zip",
                "metadata": {"version_id": "3"},
            },
        },
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "reason": None,
            "raw_stdout": "{" + "x" * 120 + "}",
            "raw_stderr": "",
        },
    }
    path = tmp_path / "paper_smoke_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_render_summary_stdout_contains_sections(sample_summary: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = render_paper_smoke_summary.main(
        ["--summary-json", str(sample_summary), "--max-json-chars", "80"]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Podsumowanie smoke paper trading" in output
    assert "## Paper pre-check" in output
    assert "| Status | ok |" in output
    assert "Wyjście publish_paper_smoke_artifacts (stdout)" in output
    assert "…" in output  # skrócony blok JSON


def test_render_summary_writes_output_file(sample_summary: Path, tmp_path: Path) -> None:
    output_file = tmp_path / "summary.md"
    exit_code = render_paper_smoke_summary.main(
        [
            "--summary-json",
            str(sample_summary),
            "--output",
            str(output_file),
            "--title",
            "Custom tytuł",
        ]
    )
    assert exit_code == 0
    content = output_file.read_text(encoding="utf-8")
    assert content.startswith("# Custom tytuł")
    assert "Archiwum smoke" in content
