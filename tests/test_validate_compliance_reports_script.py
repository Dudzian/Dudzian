from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import hmac
import hashlib

from bot_core.compliance.reports import ComplianceControl, ComplianceReport
from scripts import validate_compliance_reports


def _signed_report(path, key: bytes) -> None:
    report = ComplianceReport(
        report_id="S5-2024-OK",
        generated_at=datetime(2024, 5, 11, 9, 0, tzinfo=timezone.utc),
        controls=[
            ComplianceControl(control_id="stage5.training.log", status="pass"),
            ComplianceControl(control_id="stage5.tco.report", status="pass"),
            ComplianceControl(control_id="stage5.oem.dry_run", status="pass"),
            ComplianceControl(control_id="stage5.key_rotation", status="warn"),
            ComplianceControl(control_id="stage5.compliance.review", status="pass"),
        ],
    )
    payload = report.to_payload()
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


def test_validate_compliance_reports_script(tmp_path):
    key = base64.b64encode(b"stage5-key").decode("ascii")
    report_path = tmp_path / "report.json"
    _signed_report(report_path, base64.b64decode(key))

    exit_code = validate_compliance_reports.run(
        [
            str(report_path),
            "--signing-key",
            key,
            "--require-signature",
        ]
    )

    assert exit_code == 0


def test_validate_compliance_reports_detects_missing(tmp_path):
    payload = {
        "report_id": "S5-bad",
        "generated_at": datetime(2024, 5, 11, 12, 0, tzinfo=timezone.utc).isoformat(),
        "controls": [],
    }
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")

    exit_code = validate_compliance_reports.run([str(bad_path), "--no-default-controls"])
    assert exit_code == 1
