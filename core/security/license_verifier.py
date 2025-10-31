"""Warstwa pomocnicza do walidacji licencji na potrzeby interfejsu UI."""
from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from bot_core.security.hwid import HwIdProvider, HwIdProviderError
from bot_core.security.license_service import (
    LicenseBundleError,
    LicenseHardwareMismatchError,
    LicenseRollbackDetectedError,
    LicenseService,
    LicenseServiceError,
    LicenseSignatureError,
    LicenseSnapshot,
    LicenseStateTamperedError,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FingerprintResult:
    """Wynik pobrania fingerprintu sprzętu."""

    fingerprint: str | None
    error_code: str | None = None
    details: str | None = None

    @property
    def ok(self) -> bool:
        return self.fingerprint is not None and not self.error_code


@dataclass(slots=True, frozen=True)
class LicenseVerificationOutcome:
    """Znormalizowany wynik weryfikacji licencji."""

    ok: bool
    code: str
    license_id: str | None = None
    fingerprint: str | None = None
    details: str | None = None
    issues: tuple[str, ...] = ()


class LicenseVerifier:
    """Weryfikuje pliki licencyjne i udostępnia wyniki dla UI."""

    def __init__(
        self,
        *,
        license_service_factory: Callable[[], LicenseService] | None = None,
        hwid_provider: HwIdProvider | None = None,
    ) -> None:
        self._license_service_factory = license_service_factory or LicenseService
        self._hwid_provider = hwid_provider or HwIdProvider()
        self._service: LicenseService | None = None

    # ------------------------------------------------------------------
    def read_fingerprint(self) -> FingerprintResult:
        """Zwraca fingerprint urządzenia lub kod błędu."""

        try:
            fingerprint = self._hwid_provider.read()
        except HwIdProviderError as exc:
            LOGGER.warning("Nie udało się odczytać fingerprintu urządzenia: %s", exc)
            return FingerprintResult(None, "fingerprint_unavailable", str(exc))
        except Exception as exc:  # pragma: no cover - zabezpieczenie defensywne
            LOGGER.exception("Nieoczekiwany błąd fingerprintu")
            return FingerprintResult(None, "unexpected_error", str(exc))
        return FingerprintResult(fingerprint)

    # ------------------------------------------------------------------
    def verify_license_file(
        self,
        path: str | Path,
        *,
        fingerprint: str | None = None,
    ) -> LicenseVerificationOutcome:
        """Weryfikuje licencję z pliku."""

        try:
            service = self._ensure_service()
        except LicenseServiceError as exc:
            return LicenseVerificationOutcome(False, "service_unavailable", details=str(exc))
        except Exception as exc:  # pragma: no cover - zabezpieczenie defensywne
            LOGGER.exception("Błąd inicjalizacji usługi licencji")
            return LicenseVerificationOutcome(False, "unexpected_error", details=str(exc))

        bundle_path = Path(path).expanduser()

        try:
            snapshot = service.load_from_file(bundle_path, expected_hwid=fingerprint)
        except FileNotFoundError as exc:
            return LicenseVerificationOutcome(False, "file_not_found", details=str(exc))
        except LicenseHardwareMismatchError as exc:
            return LicenseVerificationOutcome(False, "hardware_mismatch", details=str(exc))
        except LicenseSignatureError as exc:
            return LicenseVerificationOutcome(False, "invalid_signature", details=str(exc))
        except LicenseBundleError as exc:
            return LicenseVerificationOutcome(False, "invalid_bundle", details=str(exc))
        except LicenseRollbackDetectedError as exc:
            return LicenseVerificationOutcome(False, "rollback_detected", details=str(exc))
        except LicenseStateTamperedError as exc:
            return LicenseVerificationOutcome(False, "state_tampered", details=str(exc))
        except LicenseServiceError as exc:
            return LicenseVerificationOutcome(False, "verification_failed", details=str(exc))
        except OSError as exc:
            return LicenseVerificationOutcome(False, "io_error", details=str(exc))
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            LOGGER.exception("Nieoczekiwany wyjątek podczas weryfikacji licencji")
            return LicenseVerificationOutcome(False, "unexpected_error", details=str(exc))

        return self._success_from_snapshot(snapshot)

    # ------------------------------------------------------------------
    def verify_license_text(
        self,
        text: str,
        *,
        fingerprint: str | None = None,
    ) -> LicenseVerificationOutcome:
        """Weryfikuje licencję przekazaną jako tekst JSON."""

        payload = (text or "").strip()
        if not payload:
            return LicenseVerificationOutcome(False, "empty_input")

        try:
            json.loads(payload)
        except json.JSONDecodeError as exc:
            return LicenseVerificationOutcome(False, "invalid_bundle", details=str(exc))

        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as handle:
            handle.write(payload)
            temp_path = Path(handle.name)

        try:
            return self.verify_license_file(temp_path, fingerprint=fingerprint)
        finally:
            try:
                temp_path.unlink()
            except OSError:  # pragma: no cover - nieistotne dla testów
                LOGGER.debug("Nie udało się usunąć pliku tymczasowego %s", temp_path, exc_info=True)

    # ------------------------------------------------------------------
    def _ensure_service(self) -> LicenseService:
        if self._service is None:
            self._service = self._license_service_factory()
        return self._service

    # ------------------------------------------------------------------
    def _success_from_snapshot(self, snapshot: LicenseSnapshot) -> LicenseVerificationOutcome:
        license_id = snapshot.capabilities.license_id
        hwid = str(snapshot.payload.get("hwid") or "").strip() or snapshot.local_hwid
        return LicenseVerificationOutcome(True, "ok", license_id=license_id, fingerprint=hwid)


__all__ = [
    "FingerprintResult",
    "LicenseVerificationOutcome",
    "LicenseVerifier",
]
