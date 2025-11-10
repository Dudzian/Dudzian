from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.security.license import summarize_license_payload, validate_license


def _write_license(path: Path, payload: dict[str, object]) -> None:
    data = {
        "payload": payload,
        "signature": {"algorithm": "HMAC-SHA256", "key_id": "demo", "value": "ignored"},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_validate_license_includes_multi_seat_and_subscription(tmp_path: Path) -> None:
    license_file = tmp_path / "license.json"
    payload = {
        "profile": "enterprise",
        "issuer": "OEM-Labs",
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "license_id": "LIC-2025-001",
        "issued_at": "2025-01-01T00:00:00Z",
        "expires_at": "2025-12-31T23:59:59Z",
        "fingerprint": "OEM-DEVICE-001",
        "seat_policy": {
            "total": 3,
            "assignments": ["OEM-DEVICE-001"],
            "pending": ["OEM-DEVICE-002"],
            "enforcement": "hard",
            "auto_assign": False,
        },
        "subscription": {
            "status": "active",
            "current_period": {
                "start": "2025-01-01T00:00:00Z",
                "end": "2025-06-30T00:00:00Z",
            },
            "grace_period_days": 15,
            "renews_at": "2025-07-01T00:00:00Z",
        },
    }
    _write_license(license_file, payload)

    result = validate_license(
        license_path=license_file,
        license_keys=None,
        fingerprint_path=None,
        fingerprint_keys=None,
        current_time=datetime(2025, 5, 10, tzinfo=timezone.utc),
    )

    assert result.seats_total == 3
    assert result.seats_in_use == 1
    assert result.seat_assignments == ("OEM-DEVICE-001",)
    assert result.seat_pending == ("OEM-DEVICE-002",)
    assert result.subscription_status == "active"
    assert result.subscription_renews_at == "2025-07-01T00:00:00+00:00" or result.subscription_renews_at == "2025-07-01T00:00:00Z"
    assert result.subscription_period_start == "2025-01-01T00:00:00+00:00" or result.subscription_period_start == "2025-01-01T00:00:00Z"


def test_seat_policy_enforcement_reports_missing_assignment(tmp_path: Path) -> None:
    license_file = tmp_path / "license.json"
    payload = {
        "profile": "enterprise",
        "issuer": "OEM-Labs",
        "schema": "core.oem.license",
        "schema_version": "1.0",
        "license_id": "LIC-2025-002",
        "issued_at": "2025-01-01T00:00:00Z",
        "expires_at": "2025-12-31T23:59:59Z",
        "fingerprint": "OEM-DEVICE-003",
        "seat_policy": {
            "total": 1,
            "assignments": [],
            "enforcement": "hard",
            "auto_assign": False,
        },
    }
    _write_license(license_file, payload)

    result = validate_license(
        license_path=license_file,
        license_keys=None,
        fingerprint_path=None,
        fingerprint_keys=None,
        current_time=datetime(2025, 1, 10, tzinfo=timezone.utc),
    )

    assert any(msg.code == "license.seats.fingerprint_not_assigned" for msg in result.errors)


def test_summarize_license_payload_returns_normalized_summary() -> None:
    payload = {
        "status": "active",
        "expires_at": "2025-12-31T23:59:59Z",
        "seat_summary": {
            "total": 2,
            "in_use": 1,
            "assignments": ["OEM-DEVICE-010"],
            "pending": ["OEM-DEVICE-011"],
            "enforcement": "soft",
            "auto_assign": True,
        },
        "subscription_summary": {
            "status": "trial",
            "renews_at": "2025-02-01T00:00:00Z",
            "period_start": "2025-01-01T00:00:00Z",
            "period_end": "2025-01-31T23:59:59Z",
            "grace_expires_at": "2025-02-05T00:00:00Z",
        },
        "validation": {
            "warning_messages": [
                "Liczba przydziałów przekracza pulę",
                "Liczba przydziałów przekracza pulę",
                "Oczekujące zgłoszenia wymagają uwagi",
            ],
            "warning_codes": ["license.seats.overallocated"],
            "error_messages": ["Licencja wygasła"],
            "error_codes": ["license.expired"],
        },
    }

    summary = summarize_license_payload(payload)

    assert summary["status"] == "active"
    assert summary["expiresAt"] == "2025-12-31T23:59:59Z"
    seat_summary = summary["seatSummary"]
    assert seat_summary["total"] == 2
    assert seat_summary["assignments"] == ["OEM-DEVICE-010"]
    assert seat_summary["autoAssign"] is True
    subscription = summary["subscriptionSummary"]
    assert subscription["status"] == "trial"
    assert subscription["renewsAt"].startswith("2025-02-01")
    assert "Liczba przydziałów przekracza pulę" in summary["warningMessages"]
    assert summary["warningCodes"] == ["license.seats.overallocated"]
    assert summary["errorCodes"] == ["license.expired"]
