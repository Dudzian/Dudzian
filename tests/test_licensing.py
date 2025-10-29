from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from core.licensing.controller import (
    HardwareProbe,
    LicenseController,
    LicenseEvaluation,
    LicenseHardwareStatus,
)
from bot_core.security.capabilities import (
    LicenseCapabilities,
    LicenseLimits,
    LicenseMaintenanceWindow,
    LicenseTrialInfo,
)
from bot_core.security.license_service import LicenseSnapshot
from bot_core.security.license_store import LicenseStore, LicenseStoreDecryptionError


@dataclass(slots=True)
class _StubLicenseService:
    snapshot: LicenseSnapshot

    def load_from_file(self, path: Path, *, expected_hwid: str | None = None) -> LicenseSnapshot:
        return self.snapshot


def _make_capabilities(license_id: str, payload: dict[str, object]) -> LicenseCapabilities:
    return LicenseCapabilities(
        edition="enterprise",
        environments=frozenset({"paper"}),
        exchanges={},
        strategies={},
        runtime={},
        modules={},
        limits=LicenseLimits(),
        trial=LicenseTrialInfo(enabled=False, expires_at=None),
        maintenance=LicenseMaintenanceWindow(until=None),
        issued_at=None,
        maintenance_until=None,
        effective_date=date.today(),
        seats=None,
        holder={},
        metadata={},
        license_id=license_id,
        hwid=None,
        raw_payload=payload,
    )


def _make_snapshot(license_id: str, payload: dict[str, object]) -> LicenseSnapshot:
    return LicenseSnapshot(
        bundle_path=Path("dummy.json"),
        payload=payload,
        payload_bytes=json.dumps(payload).encode("utf-8"),
        signature_bytes=b"signature",
        capabilities=_make_capabilities(license_id, payload),
        effective_date=date.today(),
        local_hwid="local-hwid",
    )


def test_license_store_encrypts_and_decrypts(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    store = LicenseStore(path=store_path, fingerprint_override="fp-123")
    payload = {"licenses": {"alpha": {"status": "pending"}}}
    store.save(payload)

    raw = store_path.read_text(encoding="utf-8")
    assert "pending" not in raw
    assert '"version"' in raw

    reopened = LicenseStore(path=store_path, fingerprint_override="fp-123")
    document = reopened.load()
    assert document.data["licenses"]["alpha"]["status"] == "pending"
    assert document.migrated is False


def test_license_store_migration_from_plain_json(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    store_path.write_text(json.dumps({"licenses": {"alpha": {"status": "active"}}}), encoding="utf-8")
    store = LicenseStore(path=store_path, fingerprint_override="fp-123")
    document = store.load()
    assert document.migrated is True
    assert document.data["licenses"]["alpha"]["status"] == "active"
    store.save(document.data)
    raw = store_path.read_text(encoding="utf-8")
    assert '"version"' in raw


def test_license_store_decryption_error_on_other_fingerprint(tmp_path: Path) -> None:
    store_path = tmp_path / "store.json"
    original = LicenseStore(path=store_path, fingerprint_override="fp-123")
    original.save({"licenses": {"alpha": {"status": "active"}}})

    tampered = LicenseStore(path=store_path, fingerprint_override="fp-456")
    with pytest.raises(LicenseStoreDecryptionError):
        tampered.load()


def test_license_controller_updates_store(tmp_path: Path) -> None:
    license_payload = {
        "hardware": {"cpu_id": "cpu-a", "board_id": "board-a", "tpm_id": "tpm-a"},
    }
    snapshot = _make_snapshot("lic-1", license_payload)
    service = _StubLicenseService(snapshot)
    probe = HardwareProbe(
        cpu_reader=lambda: "CPU-A",
        board_reader=lambda: "BOARD-A",
        tpm_reader=lambda: "TPM-A",
    )
    store = LicenseStore(path=tmp_path / "store.json", fingerprint_override="fp-123")
    controller = LicenseController(service, store=store, hardware_probe=probe)

    evaluation = controller.verify_license(tmp_path / "bundle.json")
    assert isinstance(evaluation, LicenseEvaluation)
    assert evaluation.status is LicenseHardwareStatus.OK
    assert evaluation.store_updated is True

    document = store.load()
    assert document.data["licenses"]["lic-1"]["status"] == "ok"
    assert document.data["licenses"]["lic-1"]["hardware"]["cpu_id"] == "cpu-a"


def test_license_controller_warns_on_hardware_change(tmp_path: Path) -> None:
    payload = {"hardware": {"cpu_id": "cpu-a"}}
    snapshot = _make_snapshot("lic-1", payload)
    service = _StubLicenseService(snapshot)
    store_path = tmp_path / "store.json"
    store = LicenseStore(path=store_path, fingerprint_override="fp-123")

    first_probe = HardwareProbe(
        cpu_reader=lambda: "CPU-A",
        board_reader=lambda: "BOARD-A",
        tpm_reader=lambda: "TPM-A",
    )
    controller = LicenseController(service, store=store, hardware_probe=first_probe)
    controller.verify_license(tmp_path / "bundle.json")

    second_probe = HardwareProbe(
        cpu_reader=lambda: "CPU-A",
        board_reader=lambda: "BOARD-B",
        tpm_reader=lambda: "TPM-A",
    )
    controller_changed = LicenseController(service, store=store, hardware_probe=second_probe)
    evaluation = controller_changed.verify_license(tmp_path / "bundle.json")
    assert evaluation.status is LicenseHardwareStatus.WARNING
    assert "hardware_changed" in evaluation.issues


def test_license_controller_blocks_when_store_unavailable(tmp_path: Path) -> None:
    payload = {"hardware": {"cpu_id": "cpu-a"}}
    snapshot = _make_snapshot("lic-1", payload)
    service = _StubLicenseService(snapshot)

    store_path = tmp_path / "store.json"
    initial_store = LicenseStore(path=store_path, fingerprint_override="fp-123")
    initial_store.save({"licenses": {"lic-1": {"status": "ok"}}})

    mismatched_store = LicenseStore(path=store_path, fingerprint_override="fp-456")
    probe = HardwareProbe(
        cpu_reader=lambda: "CPU-A",
        board_reader=lambda: "BOARD-A",
        tpm_reader=lambda: "TPM-A",
    )
    controller = LicenseController(service, store=mismatched_store, hardware_probe=probe)
    evaluation = controller.verify_license(tmp_path / "bundle.json")
    assert evaluation.status is LicenseHardwareStatus.BLOCKED
    assert "license_store_unavailable" in evaluation.issues
    assert evaluation.store_updated is False
