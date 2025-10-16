from __future__ import annotations

import json
from pathlib import Path

from bot_core.runtime.full_hypercare import (
    FullHypercareSummaryBuilder,
    FullHypercareSummaryConfig,
    verify_full_hypercare_summary,
)
from bot_core.security.signing import build_hmac_signature


def _write_stage_summary(
    path: Path,
    *,
    payload: dict[str, object],
    key: bytes,
    key_id: str,
) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    signature = build_hmac_signature(payload, key=key, key_id=key_id)
    signature_path = path.with_suffix(path.suffix + ".sig")
    signature_path.write_text(json.dumps(signature, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return signature_path


def test_builder_generates_signed_summary(tmp_path: Path) -> None:
    stage5_payload = {
        "type": "stage5_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "artifacts": {
            "tco": {
                "status": "ok",
                "issues": [],
                "warnings": [],
                "summary_path": "tco.json",
            }
        },
    }
    stage6_payload = {
        "type": "stage6_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {
            "observability": {"status": "ok"},
            "resilience": {"status": "ok"},
            "portfolio": {"status": "ok"},
        },
    }

    stage5_path = tmp_path / "stage5.json"
    stage5_signature = _write_stage_summary(stage5_path, payload=stage5_payload, key=b"stage5", key_id="s5")
    stage6_path = tmp_path / "stage6.json"
    stage6_signature = _write_stage_summary(stage6_path, payload=stage6_payload, key=b"stage6", key_id="s6")

    output_path = tmp_path / "full.json"
    signature_path = tmp_path / "full.sig"

    config = FullHypercareSummaryConfig(
        stage5_summary_path=stage5_path,
        stage6_summary_path=stage6_path,
        stage5_signature_path=stage5_signature,
        stage6_signature_path=stage6_signature,
        stage5_signing_key=b"stage5",
        stage6_signing_key=b"stage6",
        stage5_require_signature=True,
        stage6_require_signature=True,
        output_path=output_path,
        signature_path=signature_path,
        signing_key=b"full",
        signing_key_id="full",
    )

    result = FullHypercareSummaryBuilder(config).run()

    assert result.output_path == output_path
    assert result.signature_path == signature_path
    assert result.payload["overall_status"] == "ok"
    assert result.stage5.signature_valid is True
    assert result.stage6.signature_valid is True
    assert signature_path.exists()


def test_verify_summary_with_component_revalidation(tmp_path: Path) -> None:
    stage5_payload = {
        "type": "stage5_hypercare_summary",
        "overall_status": "warn",
        "issues": ["rotations overdue"],
        "warnings": [],
        "artifacts": {"rotation": {"status": "fail", "issues": ["overdue"], "warnings": []}},
    }
    stage6_payload = {
        "type": "stage6_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": ["observability overrides active"],
        "components": {
            "observability": {"status": "warn"},
            "resilience": {"status": "ok"},
            "portfolio": {"status": "ok"},
        },
    }

    stage5_path = tmp_path / "stage5.json"
    stage5_signature = _write_stage_summary(stage5_path, payload=stage5_payload, key=b"stage5", key_id="s5")
    stage6_path = tmp_path / "stage6.json"
    stage6_signature = _write_stage_summary(stage6_path, payload=stage6_payload, key=b"stage6", key_id="s6")

    summary_config = FullHypercareSummaryConfig(
        stage5_summary_path=stage5_path,
        stage6_summary_path=stage6_path,
        stage5_signature_path=stage5_signature,
        stage6_signature_path=stage6_signature,
        stage5_signing_key=b"stage5",
        stage6_signing_key=b"stage6",
        output_path=tmp_path / "full.json",
        signing_key=b"full",
    )
    builder_result = FullHypercareSummaryBuilder(summary_config).run()

    verification = verify_full_hypercare_summary(
        builder_result.output_path,
        signing_key=b"full",
        require_signature=True,
        revalidate_stage5=True,
        revalidate_stage6=True,
        stage5_signing_key=b"stage5",
        stage6_signing_key=b"stage6",
        stage5_require_signature=True,
        stage6_require_signature=True,
    )

    assert verification.signature_valid is True
    assert verification.component_statuses["stage5"] == "warn"
    assert verification.component_statuses["stage6"] == "ok"
    assert verification.stage5 is not None
    assert verification.stage6 is not None
    assert "rotations overdue" in verification.issues


def test_verify_summary_missing_signature_reports_issue(tmp_path: Path) -> None:
    stage5_payload = {
        "type": "stage5_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "artifacts": {},
    }
    stage6_payload = {
        "type": "stage6_hypercare_summary",
        "overall_status": "ok",
        "issues": [],
        "warnings": [],
        "components": {},
    }

    stage5_path = tmp_path / "stage5.json"
    _write_stage_summary(stage5_path, payload=stage5_payload, key=b"stage5", key_id="s5")
    stage6_path = tmp_path / "stage6.json"
    _write_stage_summary(stage6_path, payload=stage6_payload, key=b"stage6", key_id="s6")

    config = FullHypercareSummaryConfig(
        stage5_summary_path=stage5_path,
        stage6_summary_path=stage6_path,
        output_path=tmp_path / "full.json",
    )
    builder_result = FullHypercareSummaryBuilder(config).run()

    verification = verify_full_hypercare_summary(
        builder_result.output_path,
        require_signature=True,
    )

    assert verification.signature_valid is False
    assert any("Wymagany podpis HMAC" in issue for issue in verification.issues)

