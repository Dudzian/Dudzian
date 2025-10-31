from __future__ import annotations

import base64
import json
import sys
from hashlib import sha256
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Always import the CLI module under test
from scripts import validate_compliance_reports

# -----------------------------------------------------------------------------
# Capability detection for the two alternative implementations
#   • HEAD variant: uses bot_core.compliance.reports.* and positional paths
#   • MAIN variant: uses scripts.verify_signature and flag-per-artifact CLI
# -----------------------------------------------------------------------------
try:
    # HEAD-style API (ComplianceReport/ComplianceControl)
    from bot_core.compliance.reports import (  # type: ignore[attr-defined]
        ComplianceControl,
        ComplianceReport,
    )

    _HAVE_HEAD_API = True
except Exception:  # pragma: no cover - environment without HEAD API
    ComplianceControl = None  # type: ignore[assignment]
    ComplianceReport = None  # type: ignore[assignment]
    _HAVE_HEAD_API = False

try:
    # MAIN-style API (verify_signature helper is imported by the CLI)
    from scripts import verify_signature as _verify_signature_module  # type: ignore[attr-defined]  # noqa: F401

    _HAVE_MAIN_API = True
except Exception:  # pragma: no cover - environment without MAIN API
    _HAVE_MAIN_API = False

# Common helper from core signing for MAIN-style signatures
try:
    from bot_core.security.signing import build_hmac_signature  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - allow environments missing this helper
    build_hmac_signature = None  # type: ignore[assignment]


# =============================================================================
#                               HEAD-variant tests
# =============================================================================
def _signed_report_HEAD(path: Path, key: bytes) -> None:
    """Create a minimal Stage5 compliance report with embedded HMAC signature (HEAD variant)."""
    report = ComplianceReport(  # type: ignore[call-arg]
        report_id="S5-2024-OK",
        generated_at=None,
        controls=[
            ComplianceControl(control_id="stage5.training.log", status="pass"),  # type: ignore[call-arg]
            ComplianceControl(control_id="stage5.tco.report", status="pass"),  # type: ignore[call-arg]
            ComplianceControl(control_id="stage5.oem.dry_run", status="pass"),  # type: ignore[call-arg]
            ComplianceControl(control_id="stage5.key_rotation", status="warn"),  # type: ignore[call-arg]
            ComplianceControl(control_id="stage5.compliance.review", status="pass"),  # type: ignore[call-arg]
        ],
    )
    payload = report.to_payload()  # type: ignore[attr-defined]
    # Build HMAC-SHA256 signature equivalent to script expectations
    import hmac, hashlib  # local import to avoid unused warnings in MAIN-only envs

    digest = hmac.new(
        key,
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        hashlib.sha256,
    ).digest()
    signature = {
        "algorithm": "HMAC-SHA256",
        "value": base64.b64encode(digest).decode("ascii"),
        "key_id": "stage5",
    }
    payload["signature"] = signature
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@pytest.mark.skipif(not _HAVE_HEAD_API, reason="HEAD-style compliance reports API not available")
def test_validate_compliance_reports_script_head_ok(tmp_path: Path) -> None:
    key_b64 = base64.b64encode(b"stage5-key").decode("ascii")
    report_path = tmp_path / "report.json"
    _signed_report_HEAD(report_path, base64.b64decode(key_b64))

    exit_code = validate_compliance_reports.run(  # type: ignore[attr-defined]
        [
            str(report_path),
            "--signing-key",
            key_b64,
            "--require-signature",
        ]
    )
    assert exit_code == 0


@pytest.mark.skipif(not _HAVE_HEAD_API, reason="HEAD-style compliance reports API not available")
def test_validate_compliance_reports_script_head_detects_missing(tmp_path: Path) -> None:
    # Missing controls list -> should fail without defaults
    payload = {
        "report_id": "S5-bad",
        "generated_at": "2024-05-11T12:00:00Z",
        "controls": [],
    }
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = validate_compliance_reports.run([str(bad_path), "--no-default-controls"])  # type: ignore[attr-defined]
    assert exit_code == 1


# =============================================================================
#                               MAIN-variant tests
# =============================================================================
@pytest.fixture()
def signing_key(tmp_path: Path) -> Path:
    key_path = tmp_path / "signing.key"
    key_path.write_bytes(b"k" * 48)
    if sys.platform != "win32":
        key_path.chmod(0o600)
    return key_path


def _write_signature_MAIN(artifact: Path, key_path: Path, *, key_id: str) -> Path:
    """Write detached signature JSON next to an artifact (MAIN variant format)."""
    assert build_hmac_signature is not None, "build_hmac_signature helper is required"
    payload = {
        "artifact": artifact.name,
        "artifact_type": artifact.suffix.lstrip("."),
        "sha256": sha256(artifact.read_bytes()).hexdigest(),
        "generated_at": "2025-01-01T00:00:00Z",
    }
    signature = build_hmac_signature(  # type: ignore[misc]
        payload,
        key=key_path.read_bytes(),
        algorithm="HMAC-SHA256",
        key_id=key_id,
    )
    signature_path = artifact.with_suffix(artifact.suffix + ".sig")
    signature_path.write_text(
        json.dumps({"payload": payload, "signature": signature}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if sys.platform != "win32":
        signature_path.chmod(0o600)
    return signature_path


@pytest.mark.skipif(not _HAVE_MAIN_API, reason="MAIN-style compliance CLI not available")
def test_validate_compliance_reports_success_main(tmp_path: Path, signing_key: Path) -> None:
    # Prepare artefacts
    tco_path = tmp_path / "tco.json"
    tco_path.write_text(json.dumps({"strategies": {"s1": {}}, "total": {}, "metadata": {}}), encoding="utf-8")
    if sys.platform != "win32":
        tco_path.chmod(0o600)
    tco_sig = _write_signature_MAIN(tco_path, signing_key, key_id="tco-key")

    observability_path = tmp_path / "observability.json"
    observability_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    if sys.platform != "win32":
        observability_path.chmod(0o600)
    observability_sig = _write_signature_MAIN(observability_path, signing_key, key_id="obs-key")

    decision_smoke_path = tmp_path / "decision.json"
    decision_smoke_path.write_text(
        json.dumps(
            {"accepted": 1, "rejected": 0, "stress_failures": 0, "evaluations": [{"id": "alpha"}]}
        ),
        encoding="utf-8",
    )
    if sys.platform != "win32":
        decision_smoke_path.chmod(0o600)

    decision_log_summary = tmp_path / "decision_log.json"
    decision_log_summary.write_text(json.dumps({"status": "PASS", "missing_fields": []}), encoding="utf-8")
    if sys.platform != "win32":
        decision_log_summary.chmod(0o600)

    slo_report = tmp_path / "slo.json"
    slo_report.write_text(json.dumps({"generated_at": "2025-01-01T00:00:00Z", "metrics": []}), encoding="utf-8")
    if sys.platform != "win32":
        slo_report.chmod(0o600)

    alerts_report = tmp_path / "alerts.json"
    alerts_report.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    if sys.platform != "win32":
        alerts_report.chmod(0o600)

    rotation_plan = tmp_path / "rotation.json"
    rotation_plan.write_text(json.dumps({"plan": []}), encoding="utf-8")
    if sys.platform != "win32":
        rotation_plan.chmod(0o600)

    compliance_report = tmp_path / "compliance.json"
    compliance_report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    if sys.platform != "win32":
        compliance_report.chmod(0o600)

    summary_path = tmp_path / "summary.json"

    exit_code = validate_compliance_reports.run(  # type: ignore[attr-defined]
        [
            "--tco-json",
            str(tco_path),
            "--tco-signature",
            str(tco_sig),
            "--tco-signing-key",
            str(signing_key),
            "--observability-manifest",
            str(observability_path),
            "--observability-signature",
            str(observability_sig),
            "--observability-signing-key",
            str(signing_key),
            "--decision-smoke",
            str(decision_smoke_path),
            "--decision-log-summary",
            str(decision_log_summary),
            "--slo-report",
            str(slo_report),
            "--alerts-report",
            str(alerts_report),
            "--rotation-report",
            str(rotation_plan),
            "--compliance-report",
            str(compliance_report),
            "--summary-output",
            str(summary_path),
        ]
    )
    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert {entry["check"] for entry in summary} >= {
        "tco_report",
        "observability_bundle",
        "decision_smoke",
        "decision_log",
        "slo_report",
        "alerts_report",
        "rotation_plan",
        "compliance_report",
    }


@pytest.mark.skipif(not _HAVE_MAIN_API, reason="MAIN-style compliance CLI not available")
def test_validate_compliance_reports_missing_field_main(tmp_path: Path) -> None:
    report = tmp_path / "invalid.json"
    report.write_text(json.dumps({"generated_at": "2025"}), encoding="utf-8")
    if sys.platform != "win32":
        report.chmod(0o600)

    exit_code = validate_compliance_reports.run(["--slo-report", str(report)])  # type: ignore[attr-defined]
    assert exit_code == 1
