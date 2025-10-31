import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.monitoring.events import OnboardingCompleted, OnboardingFailed
from core.security.license_verifier import LicenseVerificationOutcome
from ui.backend.licensing_controller import LicensingController


class _StubVerifier:
    def __init__(self, *, succeed: bool) -> None:
        self._succeed = succeed

    def read_fingerprint(self):
        from core.security.license_verifier import FingerprintResult

        return FingerprintResult("HW-XYZ-001")

    def verify_license_text(self, _: str, *, fingerprint: str | None = None) -> LicenseVerificationOutcome:
        if self._succeed:
            return LicenseVerificationOutcome(True, "ok", license_id="demo", fingerprint=fingerprint)
        return LicenseVerificationOutcome(False, "invalid_signature", details="bad signature")

    def verify_license_file(self, path: str, *, fingerprint: str | None = None) -> LicenseVerificationOutcome:
        return self.verify_license_text(path, fingerprint=fingerprint)


class _HistogramRecorder:
    def __init__(self) -> None:
        self.samples: list[float] = []

    def observe(self, value: float) -> None:
        self.samples.append(float(value))


class _MetricsStub:
    def __init__(self) -> None:
        self.duration_seconds = _HistogramRecorder()


@pytest.fixture(scope="module")
def _qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def test_onboarding_logging_success(tmp_path: Path, _qt_app) -> None:
    events: list[object] = []
    metrics = _MetricsStub()
    controller = LicensingController(
        verifier=_StubVerifier(succeed=True),
        log_directory=tmp_path,
        event_publisher=events.append,
        metrics=metrics,
    )

    controller.refreshFingerprint()
    controller.applyLicenseText("VALID")
    controller.finalizeOnboarding(True, "Grid Demo", "binance", "onboarding.strategy.status.ready")

    log_file = tmp_path / "onboarding.log"
    assert log_file.exists(), "Log onboardingowy nie został utworzony"
    contents = log_file.read_text(encoding="utf-8")
    assert "ONBOARDING_COMPLETED" in contents
    assert metrics.duration_seconds.samples, "Metryka czasu onboardingowego nie została zarejestrowana"
    assert events, "Zdarzenia monitorujące nie zostały opublikowane"
    assert isinstance(events[0], OnboardingCompleted)
    assert events[0].license_id == "demo"


def test_onboarding_logging_failure(tmp_path: Path, _qt_app) -> None:
    events: list[object] = []
    metrics = _MetricsStub()
    log_dir = tmp_path / "fail"
    controller = LicensingController(
        verifier=_StubVerifier(succeed=False),
        log_directory=log_dir,
        event_publisher=events.append,
        metrics=metrics,
    )

    controller.refreshFingerprint()
    controller.applyLicenseText("INVALID")
    controller.finalizeOnboarding(False, "", "", "onboarding.strategy.error.load")

    log_file = log_dir / "onboarding.log"
    assert log_file.exists()
    contents = log_file.read_text(encoding="utf-8")
    assert "ONBOARDING_FAILED" in contents
    assert metrics.duration_seconds.samples
    assert events
    assert isinstance(events[0], OnboardingFailed)
    assert events[0].status_code == "invalid_signature"
