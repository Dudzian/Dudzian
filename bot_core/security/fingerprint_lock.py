"""Persistent hardware fingerprint lock used during runtime bootstrap."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.fingerprint import DeviceFingerprintGenerator, FingerprintError


class FingerprintLockError(RuntimeError):
    """Signals that fingerprint lock storage is corrupted or inconsistent."""


@dataclass(slots=True, frozen=True)
class FingerprintLock:
    """Represents persisted hardware fingerprint binding."""

    fingerprint: str
    created_at: datetime
    path: Path
    metadata: Mapping[str, Any] | None = None


def _default_lock_path() -> Path:
    configured = os.environ.get("BOT_CORE_FINGERPRINT_LOCK")
    return Path(configured or "var/security/device_fingerprint.json")


def _canonical_fingerprint(value: str | None) -> str:
    if value is None:
        raise FingerprintLockError("Fingerprint lock nie zawiera wartości fingerprint.")
    normalized = value.strip().upper()
    if not normalized:
        raise FingerprintLockError("Fingerprint w pliku blokady jest pusty.")
    return normalized


def _decode_timestamp(candidate: str | None) -> datetime:
    if not candidate:
        return datetime.fromtimestamp(0, tz=timezone.utc)
    try:
        parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise FingerprintLockError("Nie udało się zinterpretować znacznika czasu blokady.") from exc
    return parsed.astimezone(timezone.utc)


def load_fingerprint_lock(path: str | os.PathLike[str] | None = None) -> FingerprintLock | None:
    """Reads fingerprint lock from disk.

    Returns ``None`` when lock file is missing.
    """

    target = Path(path).expanduser() if path is not None else _default_lock_path()
    if not target.exists():
        return None
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FingerprintLockError(f"Plik blokady fingerprintu {target} ma niepoprawny format JSON.") from exc
    if not isinstance(payload, Mapping):
        raise FingerprintLockError("Plik blokady fingerprintu powinien zawierać obiekt JSON.")

    fingerprint_value = _canonical_fingerprint(str(payload.get("fingerprint") or ""))
    created_at = _decode_timestamp(str(payload.get("created_at") or ""))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), Mapping) else None

    return FingerprintLock(
        fingerprint=fingerprint_value,
        created_at=created_at,
        path=target,
        metadata=metadata,
    )


def _build_payload(fingerprint: str, *, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    normalized = _canonical_fingerprint(fingerprint)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    payload: dict[str, Any] = {
        "fingerprint": normalized,
        "created_at": now.isoformat().replace("+00:00", "Z"),
        "version": 1,
    }
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def write_fingerprint_lock(
    fingerprint: str,
    *,
    path: str | os.PathLike[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FingerprintLock:
    """Stores fingerprint lock and returns resulting descriptor."""

    target = Path(path).expanduser() if path is not None else _default_lock_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _build_payload(fingerprint, metadata=metadata)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, target)
    return FingerprintLock(
        fingerprint=_canonical_fingerprint(fingerprint),
        created_at=datetime.now(timezone.utc),
        path=target,
        metadata=dict(metadata) if metadata else None,
    )


def verify_local_hardware(
    lock: FingerprintLock,
    *,
    generator: DeviceFingerprintGenerator | None = None,
) -> None:
    """Validates that current machine fingerprint matches persisted lock."""

    hardware = generator or DeviceFingerprintGenerator()
    try:
        current = hardware.generate_fingerprint()
    except FingerprintError as exc:
        raise FingerprintLockError("Nie udało się odczytać lokalnego fingerprintu urządzenia.") from exc

    if _canonical_fingerprint(current) != lock.fingerprint:
        raise FingerprintLockError(
            "Fingerprint urządzenia nie zgadza się z blokadą instalacyjną."
        )


__all__ = [
    "FingerprintLock",
    "FingerprintLockError",
    "load_fingerprint_lock",
    "verify_local_hardware",
    "write_fingerprint_lock",
]
