"""Kontroler aktywacji licencji wykorzystywany przez kreator UI."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Iterable

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.monitoring.events import EventPublisher, OnboardingCompleted, OnboardingFailed
from core.monitoring.metrics import OnboardingMetricSet
from core.security.license_verifier import (
    FingerprintResult,
    LicenseVerificationOutcome,
    LicenseVerifier,
)
from .logging import get_onboarding_logger

_STATUS_MESSAGE_IDS: dict[str, str] = {
    "pending": "licenseWizard.status.pending",
    "ok": "licenseWizard.status.ok",
    "empty_input": "licenseWizard.error.emptyInput",
    "invalid_bundle": "licenseWizard.error.invalidBundle",
    "invalid_signature": "licenseWizard.error.invalidSignature",
    "hardware_mismatch": "licenseWizard.error.hardwareMismatch",
    "file_not_found": "licenseWizard.error.fileNotFound",
    "rollback_detected": "licenseWizard.error.rollbackDetected",
    "state_tampered": "licenseWizard.error.stateTampered",
    "verification_failed": "licenseWizard.error.generalFailure",
    "service_unavailable": "licenseWizard.error.serviceUnavailable",
    "io_error": "licenseWizard.error.io",
    "unexpected_error": "licenseWizard.error.unexpected",
}

_FINGERPRINT_ERROR_IDS: dict[str, str] = {
    "fingerprint_unavailable": "licenseWizard.error.fingerprintUnavailable",
    "unexpected_error": "licenseWizard.error.unexpected",
}


class LicensingController(QObject):
    """Udostępnia QML logikę weryfikacji licencji OEM."""

    fingerprintChanged = Signal()
    fingerprintErrorMessageIdChanged = Signal()
    statusMessageIdChanged = Signal()
    statusDetailsChanged = Signal()
    licenseIdChanged = Signal()
    licenseAcceptedChanged = Signal()
    warningCodesChanged = Signal()

    def __init__(
        self,
        verifier: LicenseVerifier | None = None,
        parent: QObject | None = None,
        *,
        log_directory: str | Path | None = None,
        event_publisher: EventPublisher | None = None,
        metrics: OnboardingMetricSet | None = None,
    ) -> None:
        super().__init__(parent)
        self._verifier = verifier or LicenseVerifier()
        self._fingerprint = ""
        self._fingerprint_error_id = ""
        self._status_code = "pending"
        self._status_message_id = _STATUS_MESSAGE_IDS["pending"]
        self._status_details = ""
        self._license_id = ""
        self._license_accepted = False
        self._warnings: tuple[str, ...] = ()
        self._event_publisher = event_publisher
        self._metrics = metrics or OnboardingMetricSet()
        directory = log_directory or Path("logs/ui/onboarding")
        self._onboarding_logger = get_onboarding_logger(directory)
        self._started_at = perf_counter()
        self._completed = False

    # ------------------------------------------------------------------
    @Property(str, notify=fingerprintChanged)
    def fingerprint(self) -> str:  # type: ignore[override]
        return self._fingerprint

    @Property(str, notify=fingerprintErrorMessageIdChanged)
    def fingerprintErrorMessageId(self) -> str:  # type: ignore[override]
        return self._fingerprint_error_id

    @Property(str, notify=statusMessageIdChanged)
    def statusMessageId(self) -> str:  # type: ignore[override]
        return self._status_message_id

    @Property(str, notify=statusDetailsChanged)
    def statusDetails(self) -> str:  # type: ignore[override]
        return self._status_details

    @Property(str, notify=licenseIdChanged)
    def licenseId(self) -> str:  # type: ignore[override]
        return self._license_id

    @Property(bool, notify=licenseAcceptedChanged)
    def licenseAccepted(self) -> bool:  # type: ignore[override]
        return self._license_accepted

    @Property('QStringList', notify=warningCodesChanged)
    def warningCodes(self) -> list[str]:  # type: ignore[override]
        return list(self._warnings)

    # ------------------------------------------------------------------
    @Slot()
    def refreshFingerprint(self) -> None:
        """Próbuje odczytać aktualny fingerprint sprzętu."""

        result = self._verifier.read_fingerprint()
        self._apply_fingerprint_result(result)

    @Slot()
    def resetStatus(self) -> None:
        """Resetuje stan komunikatów w kreatorze."""

        self._apply_outcome(LicenseVerificationOutcome(False, "pending"))

    @Slot(str, result=bool)
    def applyLicenseText(self, text: str) -> bool:
        """Weryfikuje licencję przekazaną jako tekst JSON."""

        outcome = self._verifier.verify_license_text(text, fingerprint=self._fingerprint or None)
        self._apply_outcome(outcome)
        return outcome.ok

    @Slot(str, result=bool)
    def applyLicenseFile(self, path: str) -> bool:
        """Weryfikuje licencję wskazaną plikiem."""

        outcome = self._verifier.verify_license_file(path, fingerprint=self._fingerprint or None)
        self._apply_outcome(outcome)
        return outcome.ok

    @Slot(bool, str, str, str)
    def finalizeOnboarding(
        self,
        success: bool,
        strategy_title: str = "",
        exchange_id: str = "",
        onboarding_status_id: str = "",
    ) -> None:
        """Rejestruje zakończenie kreatora onboardingowego."""

        if self._completed:
            return
        self._completed = True

        duration = max(0.0, perf_counter() - self._started_at)
        strategy = strategy_title.strip() or None
        exchange = exchange_id.strip() or None
        onboarding_status = onboarding_status_id.strip() or None
        details = self._status_details or None

        histogram = getattr(self._metrics, "duration_seconds", None)
        if histogram is not None:
            try:
                histogram.observe(duration)
            except Exception:  # pragma: no cover - telemetria opcjonalna
                pass

        if success and self._license_accepted:
            self._onboarding_logger.info(
                "ONBOARDING_COMPLETED license=%s strategy=%s exchange=%s duration=%.3fs status=%s",
                self._license_id or "-",
                strategy or "",
                exchange or "",
                duration,
                self._status_message_id,
            )
            if self._event_publisher is not None:
                self._event_publisher(
                    OnboardingCompleted(
                        duration_seconds=duration,
                        license_id=self._license_id or None,
                        strategy=strategy,
                        exchange=exchange,
                        details=details,
                        onboarding_status_id=onboarding_status,
                    )
                )
        else:
            status_message_id = self._status_message_id or _STATUS_MESSAGE_IDS["pending"]
            self._onboarding_logger.warning(
                "ONBOARDING_FAILED status=%s code=%s strategy=%s exchange=%s duration=%.3fs details=%s",
                status_message_id,
                self._status_code,
                strategy or "",
                exchange or "",
                duration,
                onboarding_status or details or "",
            )
            if self._event_publisher is not None:
                self._event_publisher(
                    OnboardingFailed(
                        duration_seconds=duration,
                        status_code=self._status_code,
                        status_message_id=status_message_id,
                        details=details,
                        strategy=strategy,
                        exchange=exchange,
                        onboarding_status_id=onboarding_status,
                    )
                )

    # ------------------------------------------------------------------
    def _apply_fingerprint_result(self, result: FingerprintResult) -> None:
        if result.fingerprint:
            self._update_fingerprint(result.fingerprint)
            self._set_fingerprint_error_id("")
        else:
            self._update_fingerprint("")
            message_id = _FINGERPRINT_ERROR_IDS.get(
                result.error_code or "fingerprint_unavailable",
                _FINGERPRINT_ERROR_IDS["fingerprint_unavailable"],
            )
            self._set_fingerprint_error_id(message_id)
            if result.details:
                self._set_status_details(result.details)

    def _apply_outcome(self, outcome: LicenseVerificationOutcome) -> None:
        self._status_code = outcome.code
        self._set_status_message_id(_STATUS_MESSAGE_IDS.get(outcome.code, _STATUS_MESSAGE_IDS["unexpected_error"]))
        self._set_status_details(outcome.details or "")
        self._set_license_id(outcome.license_id or "")
        self._set_license_accepted(outcome.ok)
        self._set_warnings(outcome.issues)
        if outcome.fingerprint:
            self._update_fingerprint(outcome.fingerprint)
            self._set_fingerprint_error_id("")

    # ------------------------------------------------------------------
    def _update_fingerprint(self, value: str) -> None:
        normalized = str(value or "").strip()
        if normalized == self._fingerprint:
            return
        self._fingerprint = normalized
        self.fingerprintChanged.emit()

    def _set_fingerprint_error_id(self, message_id: str) -> None:
        if message_id == self._fingerprint_error_id:
            return
        self._fingerprint_error_id = message_id
        self.fingerprintErrorMessageIdChanged.emit()

    def _set_status_message_id(self, message_id: str) -> None:
        if message_id == self._status_message_id:
            return
        self._status_message_id = message_id
        self.statusMessageIdChanged.emit()

    def _set_status_details(self, details: str) -> None:
        if details == self._status_details:
            return
        self._status_details = details
        self.statusDetailsChanged.emit()

    def _set_license_id(self, license_id: str) -> None:
        if license_id == self._license_id:
            return
        self._license_id = license_id
        self.licenseIdChanged.emit()

    def _set_license_accepted(self, accepted: bool) -> None:
        if accepted == self._license_accepted:
            return
        self._license_accepted = accepted
        self.licenseAcceptedChanged.emit()

    def _set_warnings(self, warnings: Iterable[str] | None) -> None:
        normalized = tuple(str(item) for item in (warnings or ()))
        if normalized == self._warnings:
            return
        self._warnings = normalized
        self.warningCodesChanged.emit()


__all__ = ["LicensingController"]
