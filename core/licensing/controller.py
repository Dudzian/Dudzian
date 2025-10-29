"""Kontroler licencyjny z twardymi zabezpieczeniami OEM."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from bot_core.security.hwid import HwIdProvider
from bot_core.security.license_service import LicenseService, LicenseSnapshot
from bot_core.security.license_store import (
    LicenseStore,
    LicenseStoreDecryptionError,
    LicenseStoreDocument,
    LicenseStoreError,
    LicenseStoreFingerprintError,
)

LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Modele danych
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HardwareFingerprintSnapshot:
    """Sfotografowany fingerprint sprzętu wraz z komponentami."""

    cpu_id: str | None
    board_id: str | None
    tpm_id: str | None

    def as_dict(self) -> dict[str, str]:
        return {
            key: value
            for key, value in (
                ("cpu_id", self.cpu_id),
                ("board_id", self.board_id),
                ("tpm_id", self.tpm_id),
            )
            if value
        }

    def normalized(self) -> tuple[str, str, str]:
        return tuple(_normalize_component(value) or "" for value in (self.cpu_id, self.board_id, self.tpm_id))


class LicenseHardwareStatus(str, Enum):
    """Status dopasowania licencji do sprzętu."""

    OK = "ok"
    WARNING = "warning"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class LicenseEvaluation:
    """Wynik weryfikacji licencji względem aktualnego sprzętu."""

    license_id: str | None
    status: LicenseHardwareStatus
    issues: tuple[str, ...]
    hardware: HardwareFingerprintSnapshot
    license_hardware: HardwareFingerprintSnapshot | None
    stored_hardware: HardwareFingerprintSnapshot | None
    store_updated: bool
    store_error: str | None = None


# ---------------------------------------------------------------------------
# Narzędzia pomocnicze
# ---------------------------------------------------------------------------


def _read_first_line(path: Path) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            line = handle.readline().strip()
    except FileNotFoundError:
        return None
    return line or None


def _normalize_component(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned.lower()


def _default_cpu_reader() -> str | None:
    override = os.environ.get("OEM_CPU_ID")
    if override:
        return override.strip()
    candidate = _read_first_line(Path("/sys/devices/virtual/dmi/id/product_uuid"))
    if candidate:
        return candidate
    candidate = _read_first_line(Path("/sys/devices/virtual/dmi/id/product_serial"))
    if candidate:
        return candidate
    candidate = _read_first_line(Path("/sys/devices/virtual/dmi/id/board_serial"))
    if candidate:
        return candidate
    return None


def _default_board_reader() -> str | None:
    override = os.environ.get("OEM_BOARD_ID")
    if override:
        return override.strip()
    candidate = _read_first_line(Path("/sys/devices/virtual/dmi/id/board_name"))
    if candidate:
        return candidate
    candidate = _read_first_line(Path("/sys/devices/virtual/dmi/id/board_vendor"))
    if candidate:
        serial = _read_first_line(Path("/sys/devices/virtual/dmi/id/board_serial"))
        if serial:
            return f"{candidate}:{serial}"
    return None


def _default_tpm_reader() -> str | None:
    override = os.environ.get("OEM_TPM_ID")
    if override:
        return override.strip()
    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/device/unique_id"))
    if candidate:
        return candidate
    candidate = _read_first_line(Path("/sys/class/tpm/tpm0/unique_id"))
    if candidate:
        return candidate
    return None


class HardwareProbe:
    """Dostarcza szczegółów fingerprintu sprzętowego."""

    def __init__(
        self,
        *,
        cpu_reader: Callable[[], str | None] | None = None,
        board_reader: Callable[[], str | None] | None = None,
        tpm_reader: Callable[[], str | None] | None = None,
    ) -> None:
        self._cpu_reader = cpu_reader or _default_cpu_reader
        self._board_reader = board_reader or _default_board_reader
        self._tpm_reader = tpm_reader or _default_tpm_reader

    def snapshot(self) -> HardwareFingerprintSnapshot:
        return HardwareFingerprintSnapshot(
            cpu_id=_normalize_component(self._cpu_reader()) or None,
            board_id=_normalize_component(self._board_reader()) or None,
            tpm_id=_normalize_component(self._tpm_reader()) or None,
        )


def _snapshot_from_mapping(data: Mapping[str, Any] | None) -> HardwareFingerprintSnapshot | None:
    if not isinstance(data, Mapping):
        return None
    return HardwareFingerprintSnapshot(
        cpu_id=_normalize_component(data.get("cpu") or data.get("cpu_id")) or None,
        board_id=_normalize_component(data.get("board") or data.get("board_id")) or None,
        tpm_id=_normalize_component(data.get("tpm") or data.get("tpm_id")) or None,
    )


# ---------------------------------------------------------------------------
# Kontroler licencji
# ---------------------------------------------------------------------------


class LicenseController:
    """Zapewnia walidację licencji oraz stan magazynu OEM."""

    def __init__(
        self,
        license_service: LicenseService,
        *,
        store: LicenseStore | None = None,
        hardware_probe: HardwareProbe | None = None,
        hwid_provider: HwIdProvider | None = None,
    ) -> None:
        self._license_service = license_service
        self._store = store or LicenseStore()
        self._hardware_probe = hardware_probe or HardwareProbe()
        self._hwid_provider = hwid_provider or HwIdProvider()

    # ------------------------------------------------------------------
    def verify_license(self, bundle_path: str | Path) -> LicenseEvaluation:
        """Weryfikuje licencję, zapisuje stan magazynu i zwraca status."""

        current_hardware = self._hardware_probe.snapshot()
        bundle_path = Path(bundle_path)

        store_document: LicenseStoreDocument | None = None
        store_error_message: str | None = None
        try:
            store_document = self._store.load()
        except LicenseStoreDecryptionError as exc:
            store_error_message = str(exc)
        except LicenseStoreFingerprintError as exc:
            store_error_message = str(exc)
        except LicenseStoreError as exc:
            store_error_message = str(exc)

        snapshot = self._license_service.load_from_file(bundle_path)
        license_id = snapshot.capabilities.license_id
        license_hardware = _extract_license_hardware(snapshot)

        issues: list[str] = []
        status = LicenseHardwareStatus.OK

        if store_error_message:
            status = LicenseHardwareStatus.BLOCKED
            issues.append("license_store_unavailable")

        stored_hardware = _resolve_stored_hardware(store_document, license_id)
        if stored_hardware and stored_hardware.normalized() != current_hardware.normalized():
            if status is LicenseHardwareStatus.OK:
                status = LicenseHardwareStatus.WARNING
            issues.append("hardware_changed")

        status, issues = _evaluate_license_vs_hardware(
            status,
            issues,
            license_hardware,
            current_hardware,
        )

        store_updated = False
        if store_document and status is not LicenseHardwareStatus.BLOCKED:
            store_updated = self._persist_store(
                store_document,
                license_id=license_id,
                status=status,
                issues=tuple(issues),
                hardware=current_hardware,
            )

        return LicenseEvaluation(
            license_id=license_id,
            status=status,
            issues=tuple(issues),
            hardware=current_hardware,
            license_hardware=license_hardware,
            stored_hardware=stored_hardware,
            store_updated=store_updated,
            store_error=store_error_message,
        )

    # ------------------------------------------------------------------
    def _persist_store(
        self,
        document: LicenseStoreDocument,
        *,
        license_id: str | None,
        status: LicenseHardwareStatus,
        issues: tuple[str, ...],
        hardware: HardwareFingerprintSnapshot,
    ) -> bool:
        data = document.data
        licenses_section = data.setdefault("licenses", {})
        if not isinstance(licenses_section, MutableMapping):
            LOGGER.warning("Sekcja 'licenses' w magazynie ma niepoprawny format – resetowana")
            licenses_section = {}
            data["licenses"] = licenses_section
        key = license_id or "__default__"
        record: dict[str, Any] = dict(licenses_section.get(key) or {})
        record["status"] = status.value
        record["issues"] = list(issues)
        record["hardware"] = hardware.as_dict()
        record["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        licenses_section[key] = record
        try:
            self._store.save(data)
        except LicenseStoreError as exc:
            LOGGER.error("Nie udało się zapisać magazynu licencji: %s", exc)
            return False
        return True


def _extract_license_hardware(snapshot: LicenseSnapshot) -> HardwareFingerprintSnapshot | None:
    payload = snapshot.payload
    if not isinstance(payload, Mapping):
        return None
    hardware = payload.get("hardware")
    if isinstance(hardware, Mapping):
        return HardwareFingerprintSnapshot(
            cpu_id=_normalize_component(hardware.get("cpu") or hardware.get("cpu_id")) or None,
            board_id=_normalize_component(hardware.get("board") or hardware.get("board_id")) or None,
            tpm_id=_normalize_component(hardware.get("tpm") or hardware.get("tpm_id")) or None,
        )
    # Fallback: pojedyncze pola w payload
    return HardwareFingerprintSnapshot(
        cpu_id=_normalize_component(payload.get("cpu") or payload.get("cpu_id")) or None,
        board_id=_normalize_component(payload.get("board") or payload.get("board_id")) or None,
        tpm_id=_normalize_component(payload.get("tpm") or payload.get("tpm_id")) or None,
    )


def _resolve_stored_hardware(
    document: LicenseStoreDocument | None, license_id: str | None
) -> HardwareFingerprintSnapshot | None:
    if document is None:
        return None
    licenses_section = document.data.get("licenses")
    if not isinstance(licenses_section, Mapping):
        return None
    key = license_id or "__default__"
    record = licenses_section.get(key)
    if not isinstance(record, Mapping):
        return None
    return _snapshot_from_mapping(record.get("hardware"))


def _evaluate_license_vs_hardware(
    status: LicenseHardwareStatus,
    issues: list[str],
    license_hardware: HardwareFingerprintSnapshot | None,
    current_hardware: HardwareFingerprintSnapshot,
) -> tuple[LicenseHardwareStatus, list[str]]:
    if license_hardware is None:
        if status is LicenseHardwareStatus.OK:
            status = LicenseHardwareStatus.WARNING
        issues.append("license_hardware_missing")
        return status, issues

    for field_name in ("cpu_id", "board_id", "tpm_id"):
        expected = getattr(license_hardware, field_name)
        actual = getattr(current_hardware, field_name)
        if expected and not actual:
            if status is LicenseHardwareStatus.OK:
                status = LicenseHardwareStatus.WARNING
            issues.append(f"{field_name}_unknown")
        elif expected and actual and expected != actual:
            status = LicenseHardwareStatus.BLOCKED
            issues.append(f"{field_name}_mismatch")
    return status, issues


__all__ = [
    "HardwareFingerprintSnapshot",
    "HardwareProbe",
    "LicenseController",
    "LicenseEvaluation",
    "LicenseHardwareStatus",
]
