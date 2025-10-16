from __future__ import annotations

from datetime import datetime, timezone
import base64

from bot_core.compliance.reports import (
    ComplianceControl,
    ComplianceReport,
    validate_compliance_report,
)


def test_validate_compliance_report_success():
    report = ComplianceReport(
        report_id="S5-2024-01",
        generated_at=datetime(2024, 5, 10, 10, 0, tzinfo=timezone.utc),
        controls=[
            ComplianceControl(control_id="stage5.training.log", status="pass"),
            ComplianceControl(control_id="stage5.tco.report", status="warn"),
        ],
        metadata={"cycle": "may"},
    )
    payload = report.to_payload()
    issues, warnings, failed, passed = validate_compliance_report(
        payload,
        expected_controls=["stage5.training.log"],
    )
    assert not issues
    assert "stage5.tco.report" not in failed
    assert "stage5.training.log" in passed


def test_validate_compliance_report_signature(tmp_path):
    key = base64.b64encode(b"compliance_secret").decode("ascii")
    control = ComplianceControl(control_id="stage5.training.log", status="pass")
    report = ComplianceReport(
        report_id="S5-2024-02",
        generated_at=datetime(2024, 5, 10, 12, 0, tzinfo=timezone.utc),
        controls=[control],
    )
    payload = report.to_payload()
    signature = {
        "algorithm": "HMAC-SHA256",
        "value": base64.b64encode(b"dummy").decode("ascii"),
        "key_id": "stage5",
    }
    payload["signature"] = signature

    issues, warnings, failed, passed = validate_compliance_report(
        payload,
        signing_key=base64.b64decode(key),
        require_signature=True,
    )
    assert issues  # niepoprawny podpis

