from __future__ import annotations

import base64
import json
from datetime import date
from pathlib import Path

import pytest
from nacl.signing import SigningKey

from bot_core.security.clock import ClockService
from bot_core.security.fingerprint import sign_license_payload
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
def license_service(tmp_path: Path) -> tuple[LicenseService, Path, SigningKey, Path, str]:
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
    return service, status_path, signing_key, binding_secret, hwid


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


def test_license_snapshot_is_signed_and_blocks_rollback(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
) -> None:
    service, status_path, signing_key, _, _ = license_service
    payload = _sample_payload(sequence=5, issued_at="2025-07-01T00:00:00Z")
    bundle_path = tmp_path / "license.json"
    _write_license_bundle(bundle_path, payload, signing_key)

    snapshot = service.load_from_file(bundle_path)
    assert snapshot.capabilities.license_id == payload["license_id"]
    assert snapshot.payload_sha256 == snapshot.payload_sha256.lower()

    status_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_document["payload_sha256"] == snapshot.payload_sha256
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


def test_license_snapshot_requires_hardware_wallet_flag(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
) -> None:
    service, status_path, signing_key, _, _ = license_service
    payload = _sample_payload(sequence=8, issued_at="2025-09-01T00:00:00Z")
    payload["security"] = {"require_hardware_wallet_for_outgoing": True}
    bundle_path = tmp_path / "hw.json"
    _write_license_bundle(bundle_path, payload, signing_key)

    snapshot = service.load_from_file(bundle_path)
    assert snapshot.requires_hardware_wallet is True
    assert snapshot.capabilities.require_hardware_wallet_for_outgoing is True

    status_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert status_document["security"]["requires_hardware_wallet_for_outgoing"] is True


def test_tampering_with_status_file_is_detected(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
) -> None:
    service, status_path, signing_key, _, _ = license_service
    payload = _sample_payload(sequence=10, issued_at="2025-07-01T00:00:00Z")
    bundle_path = tmp_path / "tamper.json"
    _write_license_bundle(bundle_path, payload, signing_key)
    service.load_from_file(bundle_path)

    tampered = json.loads(status_path.read_text(encoding="utf-8"))
    tampered["monotonic"]["sequence"] = 999
    status_path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(LicenseStateTamperedError):
        service.load_from_file(bundle_path)


def test_status_file_write_is_atomic(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, status_path, signing_key, _, _ = license_service

    initial_payload = _sample_payload(sequence=2, issued_at="2025-05-01T00:00:00Z")
    initial_bundle_path = tmp_path / "initial.json"
    _write_license_bundle(initial_bundle_path, initial_payload, signing_key)
    service.load_from_file(initial_bundle_path)

    original_content = status_path.read_text(encoding="utf-8")
    original_document = json.loads(original_content)
    assert original_document["signature"]["algorithm"] == "HMAC-SHA384"

    newer_payload = _sample_payload(sequence=3, issued_at="2025-06-01T00:00:00Z")
    newer_bundle_path = tmp_path / "newer.json"
    _write_license_bundle(newer_bundle_path, newer_payload, signing_key)

    tmp_status_path = status_path.with_suffix(status_path.suffix + ".tmp")
    original_replace = Path.replace

    def failing_replace(self: Path, target: Path) -> Path:  # pragma: no cover - patched for test
        if self == tmp_status_path and target == status_path:
            raise OSError("Simulated interruption")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError):
        service.load_from_file(newer_bundle_path)

    monkeypatch.undo()

    assert status_path.read_text(encoding="utf-8") == original_content
    assert not tmp_status_path.exists()

    service.load_from_file(newer_bundle_path)
    updated_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert updated_document["signature"]["algorithm"] == "HMAC-SHA384"


def test_previous_status_digest_is_case_insensitive(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
) -> None:
    service, status_path, signing_key, binding_secret, hwid = license_service

    initial_payload = _sample_payload(sequence=7, issued_at="2025-05-01T00:00:00Z")
    bundle_path = tmp_path / "initial.json"
    _write_license_bundle(bundle_path, initial_payload, signing_key)
    service.load_from_file(bundle_path)

    status_document = json.loads(status_path.read_text(encoding="utf-8"))
    status_document["payload_sha256"] = status_document["payload_sha256"].upper()
    monotonic = status_document["monotonic"]
    monotonic["payload_sha256"] = monotonic["payload_sha256"].upper()
    status_document["signature"] = dict(
        sign_license_payload(
            monotonic,
            fingerprint=hwid,
            secret_path=binding_secret,
        )
    )
    status_path.write_text(
        json.dumps(status_document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    newer_payload = _sample_payload(sequence=8, issued_at="2025-06-01T00:00:00Z")
    newer_path = tmp_path / "newer.json"
    _write_license_bundle(newer_path, newer_payload, signing_key)

    updated_snapshot = service.load_from_file(newer_path)

    updated_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert updated_document["payload_sha256"] == updated_snapshot.payload_sha256
    assert updated_document["payload_sha256"].islower()


def test_status_file_partial_write_failure_is_rolled_back(
    license_service: tuple[LicenseService, Path, SigningKey, Path, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, status_path, signing_key, _, _ = license_service

    initial_payload = _sample_payload(sequence=4, issued_at="2025-05-15T00:00:00Z")
    initial_bundle_path = tmp_path / "initial.json"
    _write_license_bundle(initial_bundle_path, initial_payload, signing_key)
    service.load_from_file(initial_bundle_path)

    original_content = status_path.read_text(encoding="utf-8")

    newer_payload = _sample_payload(sequence=5, issued_at="2025-06-01T00:00:00Z")
    newer_bundle_path = tmp_path / "newer.json"
    _write_license_bundle(newer_bundle_path, newer_payload, signing_key)

    tmp_status_path = status_path.with_suffix(status_path.suffix + ".tmp")
    original_open = Path.open

    class FailingWriter:
        def __init__(self, handle: object) -> None:
            self._handle = handle

        def __enter__(self) -> "FailingWriter":
            self._handle.__enter__()
            return self

        def __exit__(self, exc_type, exc, tb) -> bool | None:
            return self._handle.__exit__(exc_type, exc, tb)

        def write(self, data: str) -> int:
            # zapisz kawałek, aby w pliku tymczasowym pojawił się częściowy JSON
            self._handle.write(data[:10])
            raise OSError("Simulated write failure")

        def flush(self) -> None:
            self._handle.flush()

        def fileno(self) -> int:
            return self._handle.fileno()

    def failing_open(self: Path, *args, **kwargs):  # type: ignore[override]
        handle = original_open(self, *args, **kwargs)
        if self == tmp_status_path:
            return FailingWriter(handle)
        return handle

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError):
        service.load_from_file(newer_bundle_path)

    monkeypatch.undo()

    assert status_path.read_text(encoding="utf-8") == original_content
    assert not tmp_status_path.exists()

    service.load_from_file(newer_bundle_path)
    updated_document = json.loads(status_path.read_text(encoding="utf-8"))
    assert updated_document["signature"]["algorithm"] == "HMAC-SHA384"
