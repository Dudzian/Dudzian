"""Testy CLI verify_full_hypercare_summary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature
from scripts import verify_full_hypercare_summary


def _write_payload(
    path: Path, payload: dict[str, object], *, key: bytes | None = None, key_id: str = "full"
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if key is not None:
        signature = build_hmac_signature(payload, key=key, key_id=key_id)
        sig_path = path.with_suffix(path.suffix + ".sig")
        sig_path.write_text(json.dumps(signature), encoding="utf-8")
        return sig_path
    return path.with_suffix(path.suffix + ".sig")


def _stage5_payload() -> dict[str, object]:
    return {
        "type": "stage5_hypercare_summary",
        "generated_at": "2024-01-01T00:00:00Z",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "artifacts": {
            "tco": {"status": "ok", "issues": [], "warnings": [], "details": {}},
        },
    }


def _stage6_payload() -> dict[str, object]:
    return {
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


def _build_full_summary(
    stage5_path: Path,
    stage6_path: Path,
    *,
    stage5_sig: Path | None,
    stage6_sig: Path | None,
) -> dict[str, object]:
    return {
        "type": "full_hypercare_summary",
        "generated_at": "2024-01-01T00:00:00Z",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {
            "stage5": {
                "status": "ok",
                "summary_path": stage5_path.as_posix(),
                "signature_path": stage5_sig.as_posix() if stage5_sig else None,
            },
            "stage6": {
                "status": "ok",
                "summary_path": stage6_path.as_posix(),
                "signature_path": stage6_sig.as_posix() if stage6_sig else None,
            },
        },
    }


def test_cli_verifies_full_summary_with_revalidation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stage5_path = tmp_path / "stage5.json"
    stage6_path = tmp_path / "stage6.json"

    stage5_key = b"0123456789abcdef0123456789ab"
    stage6_key = b"abcdef0123456789abcdef012345"
    full_key = b"feedfeed0123456789feedfeed0123"

    stage5_sig = _write_payload(stage5_path, _stage5_payload(), key=stage5_key, key_id="stage5")
    stage6_sig = _write_payload(stage6_path, _stage6_payload(), key=stage6_key, key_id="stage6")

    full_path = tmp_path / "full.json"
    full_payload = _build_full_summary(stage5_path, stage6_path, stage5_sig=stage5_sig, stage6_sig=stage6_sig)
    _write_payload(full_path, full_payload, key=full_key)

    exit_code = verify_full_hypercare_summary.run(
        [
            str(full_path),
            "--hmac-key",
            full_key.decode("utf-8"),
            "--require-signature",
            "--revalidate-stage5",
            "--stage5-hmac-key",
            stage5_key.decode("utf-8"),
            "--stage5-require-signature",
            "--revalidate-stage6",
            "--stage6-hmac-key",
            stage6_key.decode("utf-8"),
            "--stage6-require-signature",
        ]
    )

    stdout, stderr = capsys.readouterr()
    assert exit_code == 0
    data = json.loads(stdout)
    assert data["overall_status"] == "ok"
    assert data["stage5"]["signature_valid"] is True
    assert data["stage6"]["signature_valid"] is True
    assert stderr == ""


def test_cli_reports_missing_signature(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    stage5_path = tmp_path / "stage5.json"
    stage6_path = tmp_path / "stage6.json"
    stage5_sig = _write_payload(stage5_path, _stage5_payload(), key=None)
    stage6_sig = _write_payload(stage6_path, _stage6_payload(), key=None)

    full_path = tmp_path / "full.json"
    payload = _build_full_summary(stage5_path, stage6_path, stage5_sig=stage5_sig, stage6_sig=stage6_sig)
    full_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = verify_full_hypercare_summary.run(
        [
            str(full_path),
            "--require-signature",
            "--hmac-key",
            "0123456789abcdef",
        ]
    )

    stdout, stderr = capsys.readouterr()
    assert exit_code == 2
    assert "Wymagany podpis" in stderr
    assert stdout == ""


def test_cli_handles_missing_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "missing.json"

    exit_code = verify_full_hypercare_summary.run([str(missing)])

    stdout, stderr = capsys.readouterr()
    assert exit_code == 1
    assert "Błąd" in stderr
    assert stdout == ""
