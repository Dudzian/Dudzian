"""Testy dla skryptu validate_paper_smoke_summary.py."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validate_paper_smoke_summary


def _write_summary(tmp_path: Path, payload: dict) -> Path:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(payload), encoding="utf-8")
    return summary_path


def test_validation_passes_for_ok_summary(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "binance_paper",
        "operator": "CI",
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-environment",
            "binance_paper",
            "--require-operator",
            "CI",
        ]
    )

    assert exit_code == 0


def test_validation_detects_publish_failure(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "binance_paper",
        "operator": "CI",
        "publish": {"status": "error", "required": True, "exit_code": 1},
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-publish-success",
        ]
    )

    assert exit_code == 1


def test_missing_summary_returns_error(tmp_path: Path) -> None:
    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(tmp_path / "missing.json"),
        ]
    )

    assert exit_code == 2


def test_validation_reports_environment_mismatch(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "kraken_paper",
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-environment",
            "binance_paper",
        ]
    )

    assert exit_code == 1


def test_validation_checks_publish_required_and_exit_code(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "binance_paper",
        "publish": {"status": "ok", "required": False, "exit_code": 5},
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-publish-required",
            "--require-publish-exit-zero",
        ]
    )

    assert exit_code == 1


def test_validation_checks_publish_steps(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "binance_paper",
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok"},
            "archive_upload": {"status": "error"},
        },
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-json-sync-ok",
            "--require-archive-upload-ok",
        ]
    )

    assert exit_code == 1


def test_validation_accepts_publish_requirements(tmp_path: Path) -> None:
    payload = {
        "status": "ok",
        "severity": "info",
        "environment": "binance_paper",
        "publish": {
            "status": "ok",
            "required": True,
            "exit_code": 0,
            "json_sync": {"status": "ok", "backend": "local"},
            "archive_upload": {"status": "ok", "backend": "s3"},
        },
    }
    summary_path = _write_summary(tmp_path, payload)

    exit_code = validate_paper_smoke_summary.main(
        [
            "--summary",
            str(summary_path),
            "--require-publish-success",
            "--require-publish-required",
            "--require-publish-exit-zero",
            "--require-json-sync-ok",
            "--require-archive-upload-ok",
        ]
    )

    assert exit_code == 0
