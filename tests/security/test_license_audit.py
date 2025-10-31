from __future__ import annotations

import json
from pathlib import Path

from core.security.license_audit import generate_license_audit_report


def _write_status(path: Path) -> None:
    payload = {
        "license_id": "LIC-123",
        "edition": "enterprise",
        "effective_date": "2025-02-01",
        "issued_at": "2025-01-15T12:00:00Z",
        "maintenance": {"until": "2025-12-31", "active": True},
        "trial": {"enabled": False, "expires_at": None, "active": False},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_audit_log(path: Path) -> None:
    base = {
        "event": "license_snapshot",
        "license_id": "LIC-123",
        "edition": "enterprise",
        "bundle_path": "licenses/lic-123.json",
        "payload_sha256": "deadbeef",
    }
    entries = [
        {
            **base,
            "timestamp": "2025-02-01T10:00:00Z",
            "local_hwid_hash": "abc123",
            "activation_count": 1,
            "repeat_activation": False,
        },
        {
            **base,
            "timestamp": "2025-02-10T08:30:00Z",
            "local_hwid_hash": "abc123",
            "activation_count": 2,
            "repeat_activation": True,
        },
        {
            **base,
            "timestamp": "2025-02-11T09:15:00Z",
            "local_hwid_hash": "ff00aa",
            "activation_count": 1,
            "repeat_activation": False,
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_generate_report_with_valid_inputs(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    audit_path = tmp_path / "audit.log"
    _write_status(status_path)
    _write_audit_log(audit_path)

    report = generate_license_audit_report(
        status_path=status_path,
        audit_log_path=audit_path,
        activation_limit=10,
    )

    assert report.summary.total_activations == 3
    assert report.summary.unique_devices == 2
    assert report.summary.license_id == "LIC-123"
    assert report.summary.edition == "enterprise"
    assert report.summary.latest_activation.isoformat().startswith("2025-02-11T09:15:00")
    assert report.status_document is not None
    assert not report.warnings

    markdown = report.to_markdown()
    assert "## Historia aktywacji" in markdown
    assert "ff00aa" in markdown


def test_generate_report_missing_files(tmp_path: Path) -> None:
    report = generate_license_audit_report(
        status_path=tmp_path / "missing-status.json",
        audit_log_path=tmp_path / "missing-audit.log",
    )

    assert report.summary.total_activations == 0
    assert any("Brak pliku statusu" in warning for warning in report.warnings)
    assert any("Brak dziennika" in warning for warning in report.warnings)


def test_activation_limit_applied(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    audit_path = tmp_path / "audit.log"
    _write_status(status_path)
    _write_audit_log(audit_path)

    report = generate_license_audit_report(
        status_path=status_path,
        audit_log_path=audit_path,
        activation_limit=2,
    )

    assert len(report.activations) == 2
    timestamps = [record.timestamp for record in report.activations]
    assert timestamps[0] > timestamps[1]

