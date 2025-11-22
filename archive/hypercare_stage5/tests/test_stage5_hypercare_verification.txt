"""Testy weryfikacji raportu hypercare Stage5."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.runtime.stage5_hypercare import verify_stage5_hypercare_summary
from bot_core.security.signing import build_hmac_signature


def _write_summary(path: Path, payload: dict[str, object], key: bytes | None = None) -> Path | None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if key is None:
        return None
    signature = build_hmac_signature(payload, key=key, key_id="stage5")
    signature_path = path.with_suffix(path.suffix + ".sig")
    signature_path.write_text(json.dumps(signature), encoding="utf-8")
    return signature_path


def _base_payload() -> dict[str, object]:
    return {
        "type": "stage5_hypercare_summary",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "artifacts": {
            "tco": {"status": "ok", "issues": [], "warnings": [], "details": {}},
            "rotation": {"status": "ok", "issues": [], "warnings": [], "details": {}},
        },
    }


def test_verify_stage5_hypercare_summary_success(tmp_path: Path) -> None:
    payload = _base_payload()
    summary_path = tmp_path / "stage5_summary.json"
    key = b"0123456789abcdef"
    signature_path = _write_summary(summary_path, payload, key)

    result = verify_stage5_hypercare_summary(
        summary_path,
        signing_key=key,
        require_signature=True,
    )

    assert result.signature_valid is True
    assert not result.issues
    assert result.signature_path == signature_path
    assert result.overall_status == "ok"


def test_verify_stage5_hypercare_detects_artifact_failure(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["artifacts"]["tco"] = {"status": "fail", "issues": [], "warnings": [], "details": {}}
    payload["overall_status"] = "ok"
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, payload)

    result = verify_stage5_hypercare_summary(summary_path)

    assert result.issues
    assert any("fail" in issue for issue in result.issues)
    assert any("Status" in warning for warning in result.warnings)


def test_verify_stage5_hypercare_requires_signature(tmp_path: Path) -> None:
    payload = _base_payload()
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, payload)

    result = verify_stage5_hypercare_summary(
        summary_path,
        signing_key=b"0123456789abcdef",
        require_signature=True,
    )

    assert any("Wymagany podpis" in issue for issue in result.issues)
    assert not result.signature_valid


@pytest.mark.parametrize("field", ["issues", "warnings"])
def test_verify_stage5_hypercare_rejects_invalid_collections(
    tmp_path: Path, field: str
) -> None:
    payload = _base_payload()
    payload[field] = "oops"  # type: ignore[assignment]
    summary_path = tmp_path / "summary.json"
    _write_summary(summary_path, payload)

    result = verify_stage5_hypercare_summary(summary_path)

    assert any(field in issue for issue in result.issues)
