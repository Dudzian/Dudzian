from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from bot_core.security.clock import ClockService
from bot_core.security.license_service import (
    LicenseRollbackDetectedError,
    LicenseService,
    LicenseStateTamperedError,
)


class StaticHwIdProvider:
    def __init__(self, value: str) -> None:
        self._value = value

    def read(self) -> str:
        return self._value


def _write_license_bundle(path: Path, payload: dict[str, object], signing_key: SigningKey) -> None:
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature_bytes = signing_key.sign(payload_bytes).signature
    bundle = {
        "payload_b64": base64.b64encode(payload_bytes).decode("ascii"),
        "signature_b64": base64.b64encode(signature_bytes).decode("ascii"),
    }
    path.write_text(json.dumps(bundle), encoding="utf-8")


@pytest.fixture()
def license_service(tmp_path: Path) -> tuple[LicenseService, Path, SigningKey]:
    signing_key = SigningKey.generate()
    verify_key_hex = signing_key.verify_key.encode().hex()
    hwid = "TEST-HWID-001"
    state_path = tmp_path / "clock.json"
    status_path = tmp_path / "status.json"
    audit_path = tmp_path / "audit.log"
    binding_secret = tmp_path / "binding.key"
    clock = ClockService(state_path=state_path, today_provider=lambda: date(2025, 7, 2))
    service = LicenseService(
        verify_key_hex=verify_key_hex,
        state_path=state_path,
        status_path=status_path,
        audit_log_path=audit_path,
        clock_service=clock,
        hwid_provider=StaticHwIdProvider(hwid),
        binding_secret_path=binding_secret,
    )
    return service, status_path, signing_key


def _sample_payload(sequence: int, issued_at: str) -> dict[str, object]:
    return {
        "license_id": "DUD-2025-00123",
        "edition": "pro",
        "environments": ["paper"],
        "exchanges": {"binance_spot": True},
        "strategies": {"trend_d1": True},
        "runtime": {"auto_trader": True},
        "modules": {"futures": True},
        "limits": {
            "max_paper_controllers": 1,
            "max_live_controllers": 0,
            "max_concurrent_bots": 2,
            "max_alert_channels": 2,
        },
        "holder": {"name": "QA"},
        "issued_at": issued_at,
        "maintenance_until": "2025-12-31",
        "trial": {"enabled": False, "expires_at": None},
        "seats": 1,
        "hwid": "TEST-HWID-001",
        "sequence": sequence,
    }


def test_license_snapshot_is_signed_and_blocks_rollback(license_service: tuple[LicenseService, Path, SigningKey], tmp_path: Path) -> None:
    service, status_path, signing_key = license_service
    payload = _sample_payload(sequence=5, issued_at="2025-07-01T00:00:00Z")
    bundle_path = tmp_path / "license.json"
    _write_license_bundle(bundle_path, payload, signing_key)

    snapshot = service.load_from_file(bundle_path)
    assert snapshot.capabilities.license_id == payload["license_id"]

    status_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert "signature" in status_document
    assert status_document["signature"]["algorithm"] == "HMAC-SHA384"

    newer_payload = _sample_payload(sequence=6, issued_at="2025-08-01T00:00:00Z")
    newer_path = tmp_path / "license_new.json"
    _write_license_bundle(newer_path, newer_payload, signing_key)
    service.load_from_file(newer_path)

    older_payload = _sample_payload(sequence=3, issued_at="2024-06-01T00:00:00Z")
    older_path = tmp_path / "license_old.json"
    _write_license_bundle(older_path, older_payload, signing_key)
    with pytest.raises(LicenseRollbackDetectedError):
        service.load_from_file(older_path)


def test_tampering_with_status_file_is_detected(license_service: tuple[LicenseService, Path, SigningKey], tmp_path: Path) -> None:
    service, status_path, signing_key = license_service
    payload = _sample_payload(sequence=10, issued_at="2025-07-01T00:00:00Z")
    bundle_path = tmp_path / "tamper.json"
    _write_license_bundle(bundle_path, payload, signing_key)
    service.load_from_file(bundle_path)

    tampered = json.loads(status_path.read_text(encoding="utf-8"))
    tampered["monotonic"]["sequence"] = 999
    status_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(LicenseStateTamperedError):
        service.load_from_file(bundle_path)
