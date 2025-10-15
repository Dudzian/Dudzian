from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from hashlib import sha256

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.security.signing import build_hmac_signature
from scripts import validate_compliance_reports


@pytest.fixture()
def signing_key(tmp_path: Path) -> Path:
    key_path = tmp_path / "signing.key"
    key_path.write_bytes(b"k" * 48)
    key_path.chmod(0o600)
    return key_path


def _write_signature(artifact: Path, key_path: Path, *, key_id: str) -> Path:
    payload = {
        "artifact": artifact.name,
        "artifact_type": artifact.suffix.lstrip("."),
        "sha256": sha256(artifact.read_bytes()).hexdigest(),
        "generated_at": "2025-01-01T00:00:00Z",
    }
    signature = build_hmac_signature(
        payload,
        key=key_path.read_bytes(),
        algorithm="HMAC-SHA256",
        key_id=key_id,
    )
    signature_path = artifact.with_suffix(artifact.suffix + ".sig")
    signature_path.write_text(
        json.dumps({"payload": payload, "signature": signature}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    signature_path.chmod(0o600)
    return signature_path


def test_validate_compliance_reports_success(tmp_path: Path, signing_key: Path) -> None:
    tco_path = tmp_path / "tco.json"
    tco_path.write_text(
        json.dumps({"strategies": {"s1": {}}, "total": {}, "metadata": {}}),
        encoding="utf-8",
    )
    tco_path.chmod(0o600)
    tco_sig = _write_signature(tco_path, signing_key, key_id="tco-key")

    observability_path = tmp_path / "observability.json"
    observability_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    observability_path.chmod(0o600)
    observability_sig = _write_signature(
        observability_path, signing_key, key_id="obs-key"
    )

    decision_smoke_path = tmp_path / "decision.json"
    decision_smoke_path.write_text(
        json.dumps(
            {
                "accepted": 1,
                "rejected": 0,
                "stress_failures": 0,
                "evaluations": [{"id": "alpha"}],
            }
        ),
        encoding="utf-8",
    )
    decision_smoke_path.chmod(0o600)

    decision_log_summary = tmp_path / "decision_log.json"
    decision_log_summary.write_text(
        json.dumps({"status": "PASS", "missing_fields": []}), encoding="utf-8"
    )
    decision_log_summary.chmod(0o600)

    slo_report = tmp_path / "slo.json"
    slo_report.write_text(
        json.dumps({"generated_at": "2025-01-01T00:00:00Z", "metrics": []}),
        encoding="utf-8",
    )
    slo_report.chmod(0o600)

    alerts_report = tmp_path / "alerts.json"
    alerts_report.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    alerts_report.chmod(0o600)

    rotation_plan = tmp_path / "rotation.json"
    rotation_plan.write_text(json.dumps({"plan": []}), encoding="utf-8")
    rotation_plan.chmod(0o600)

    compliance_report = tmp_path / "compliance.json"
    compliance_report.write_text(json.dumps({"status": "PASS"}), encoding="utf-8")
    compliance_report.chmod(0o600)

    summary_path = tmp_path / "summary.json"

    exit_code = validate_compliance_reports.run(
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


def test_validate_compliance_reports_missing_field(tmp_path: Path) -> None:
    report = tmp_path / "invalid.json"
    report.write_text(json.dumps({"generated_at": "2025"}), encoding="utf-8")
    report.chmod(0o600)

    exit_code = validate_compliance_reports.run(
        ["--slo-report", str(report)]
    )
    assert exit_code == 1
