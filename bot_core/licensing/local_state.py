"""Verified-only local package store and orthogonal startup/connectivity state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

from .verification import EnrollmentVerificationError, VerifiedEnrollmentV1


class ActivationStatus(str, Enum):
    ACTIVATED = "ACTIVATED"
    NOT_ACTIVATED = "NOT_ACTIVATED"
    ONLINE_UNAVAILABLE = "ONLINE_UNAVAILABLE"
    OFFLINE_ACTIVATION_REQUIRED = "OFFLINE_ACTIVATION_REQUIRED"
    LICENSE_INVALID = "LICENSE_INVALID"
    DEVICE_MISMATCH = "DEVICE_MISMATCH"
    EXPIRED = "EXPIRED"


@dataclass(frozen=True)
class StartupStateV1:
    activation_status: ActivationStatus
    service_status: str
    may_run: bool


class LocalEnrollmentStore:
    def __init__(self, path: Path, *, verifier: Callable[[bytes], VerifiedEnrollmentV1]) -> None:
        self.path = path
        self._verifier = verifier

    def import_and_verify(self, raw: bytes) -> VerifiedEnrollmentV1:
        """Cryptographically verify at the persistence boundary, then write exact bytes."""
        verified = self._verifier(raw)
        if verified.canonical_bytes != raw:
            raise EnrollmentVerificationError("VERIFIED_SOURCE_MISMATCH")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(raw)
        return verified

    def load(self) -> bytes | None:
        return self.path.read_bytes() if self.path.exists() else None


def _service_status(online_probe: Callable[[], bool]) -> str:
    try:
        return "ONLINE_AVAILABLE" if online_probe() else "ONLINE_SERVICE_UNAVAILABLE"
    except (ConnectionError, TimeoutError):
        return "ONLINE_SERVICE_UNAVAILABLE"


def resolve_startup(
    verify_local: Callable[[], VerifiedEnrollmentV1 | None],
    online_probe: Callable[[], bool],
) -> StartupStateV1:
    try:
        verified = verify_local()
    except EnrollmentVerificationError as exc:
        service = _service_status(online_probe)
        if exc.code == "EXPIRED":
            status = ActivationStatus.EXPIRED
        elif exc.code in {
            "DEVICE_MISMATCH",
            "K_PSA_NAME_MISMATCH",
            "K_PSA_PUBLIC_AREA_MISMATCH",
            "TPM_EVIDENCE_MISMATCH",
        }:
            status = ActivationStatus.DEVICE_MISMATCH
        else:
            status = ActivationStatus.LICENSE_INVALID
        return StartupStateV1(status, service, False)

    service = _service_status(online_probe)
    if verified is not None:
        return StartupStateV1(ActivationStatus.ACTIVATED, service, True)
    status = (
        ActivationStatus.NOT_ACTIVATED
        if service == "ONLINE_AVAILABLE"
        else ActivationStatus.OFFLINE_ACTIVATION_REQUIRED
    )
    return StartupStateV1(status, service, False)
