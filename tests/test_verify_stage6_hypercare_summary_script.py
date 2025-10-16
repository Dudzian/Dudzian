from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature
from scripts import verify_stage6_hypercare_summary


def _write_summary(path: Path, payload: dict[str, object], *, key: bytes | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if key is not None:
        signature = build_hmac_signature(payload, key=key, key_id="stage6")
        signature_path = path.with_suffix(path.suffix + ".sig")
        signature_path.write_text(json.dumps(signature), encoding="utf-8")


def test_cli_verifies_summary_ok(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "type": "stage6_hypercare_summary",
        "generated_at": "2024-01-01T00:00:00Z",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {
            "observability": {"status": "ok"},
            "resilience": {"status": "ok"},
            "portfolio": {"status": "ok"},
        },
    }
    summary_path = tmp_path / "summary.json"
    key_value = "0123456789abcdef0123456789abcdef"
    _write_summary(summary_path, payload, key=key_value.encode("utf-8"))

    exit_code = verify_stage6_hypercare_summary.run(
        [str(summary_path), "--hmac-key", key_value, "--require-signature"]
    )

    stdout, stderr = capsys.readouterr()
    assert exit_code == 0
    assert "overall_status" in stdout
    assert stderr == ""


def test_cli_reports_signature_issue(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = {
        "type": "stage6_hypercare_summary",
        "generated_at": "2024-01-01T00:00:00Z",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {"observability": {"status": "ok"}},
    }
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, payload)

    exit_code = verify_stage6_hypercare_summary.run(
        [str(summary_path), "--require-signature", "--hmac-key", "0123456789abcdef"]
    )

    stdout, stderr = capsys.readouterr()
    assert exit_code == 2
    assert "Wymagany podpis" in stderr
    assert stdout == ""


def test_cli_handles_structural_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary_path = tmp_path / "missing.json"

    exit_code = verify_stage6_hypercare_summary.run([str(summary_path)])

    stdout, stderr = capsys.readouterr()
    assert exit_code == 1
    assert "Błąd" in stderr
    assert stdout == ""
