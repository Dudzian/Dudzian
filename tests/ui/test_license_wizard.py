import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QObject, Property, Qt, QMetaObject, QUrl, Signal, Slot  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.security.license_verifier import LicenseVerificationOutcome
from ui.backend.licensing_controller import LicensingController


class _FakeVerifier:
    def __init__(self, *, succeed: bool = True) -> None:
        self.succeed = succeed
        self.text_calls: list[tuple[str, str | None]] = []
        self.file_calls: list[tuple[str, str | None]] = []

    def read_fingerprint(self):
        from core.security.license_verifier import FingerprintResult

        return FingerprintResult("HW-ABC-123")

    def verify_license_text(self, text: str, *, fingerprint: str | None = None) -> LicenseVerificationOutcome:
        self.text_calls.append((text, fingerprint))
        if self.succeed and "VALID" in text:
            return LicenseVerificationOutcome(True, "ok", license_id="demo-pro", fingerprint=fingerprint or "HW-ABC-123")
        return LicenseVerificationOutcome(False, "invalid_signature", details="signature mismatch")

    def verify_license_file(self, path: str, *, fingerprint: str | None = None) -> LicenseVerificationOutcome:
        self.file_calls.append((path, fingerprint))
        if self.succeed:
            return LicenseVerificationOutcome(True, "ok", license_id="demo-pro", fingerprint=fingerprint or "HW-ABC-123")
        return LicenseVerificationOutcome(False, "file_not_found", details="missing file")


class _StubOnboardingService(QObject):
    configurationReadyChanged = Signal()
    selectedStrategyChanged = Signal()
    statusMessageIdChanged = Signal()
    statusDetailsChanged = Signal()
    lastSavedExchangeChanged = Signal()
    strategiesChanged = Signal()
    availableExchangesChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._configurationReady = False
        self._selectedStrategyTitle = ""
        self._selectedStrategy = ""
        self._statusMessageId = ""
        self._statusDetails = ""
        self._lastSavedExchange = ""
        self._strategies = [
            {
                "name": "grid",
                "engine": "grid",
                "title": "Grid Demo",
                "licenseTier": "standard",
                "riskClasses": ["grid"],
                "requiredData": ["ohlcv"],
                "tags": ["demo"],
            }
        ]
        self._available_exchanges = ["binance"]

    @Property(bool, notify=configurationReadyChanged)
    def configurationReady(self) -> bool:  # type: ignore[override]
        return self._configurationReady

    @Property(str, notify=selectedStrategyChanged)
    def selectedStrategyTitle(self) -> str:  # type: ignore[override]
        return self._selectedStrategyTitle

    @Property(str, notify=selectedStrategyChanged)
    def selectedStrategy(self) -> str:  # type: ignore[override]
        return self._selectedStrategy

    @Property(str, notify=statusMessageIdChanged)
    def statusMessageId(self) -> str:  # type: ignore[override]
        return self._statusMessageId

    @Property(str, notify=statusDetailsChanged)
    def statusDetails(self) -> str:  # type: ignore[override]
        return self._statusDetails

    @Property(str, notify=lastSavedExchangeChanged)
    def lastSavedExchange(self) -> str:  # type: ignore[override]
        return self._lastSavedExchange

    @Property("QVariantList", notify=strategiesChanged)
    def strategies(self) -> list[dict]:  # type: ignore[override]
        return list(self._strategies)

    @Property("QStringList", notify=availableExchangesChanged)
    def availableExchanges(self) -> list[str]:  # type: ignore[override]
        return list(self._available_exchanges)

    @Slot(result=bool)
    def refreshStrategies(self) -> bool:
        self.strategiesChanged.emit()
        return True

    @Slot(str, result=bool)
    def selectStrategy(self, name: str) -> bool:
        self._selectedStrategy = name
        self.selectedStrategyChanged.emit()
        return True

    @Slot(str, str, str, str, result=bool)
    def importApiKey(self, exchange: str, api_key: str, api_secret: str, passphrase: str = "") -> bool:
        self.set_ready(exchange=exchange)
        return True

    def set_ready(self, *, title: str = "Grid Demo", exchange: str = "binance") -> None:
        self._selectedStrategyTitle = title
        self._selectedStrategy = "grid"
        self._statusMessageId = "onboarding.strategy.credentials.saved"
        self._statusDetails = ""
        self._lastSavedExchange = exchange
        self._configurationReady = True
        self.selectedStrategyChanged.emit()
        self.statusMessageIdChanged.emit()
        self.statusDetailsChanged.emit()
        self.lastSavedExchangeChanged.emit()
        self.configurationReadyChanged.emit()


def _load_wizard(
    controller: LicensingController,
    onboarding_service: _StubOnboardingService | None = None,
) -> tuple[object, QQmlApplicationEngine, QApplication, _StubOnboardingService | None]:
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("licensingController", controller)
    if onboarding_service is not None:
        engine.rootContext().setContextProperty("onboardingService", onboarding_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "onboarding" / "LicenseWizard.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować LicenseWizard.qml"
    root = engine.rootObjects()[0]
    return root, engine, app, onboarding_service  # type: ignore[return-value]


@pytest.mark.timeout(30)
def test_license_wizard_happy_path(tmp_path: Path) -> None:
    controller = LicensingController(verifier=_FakeVerifier())
    onboarding_service = _StubOnboardingService()
    root, engine, app, _ = _load_wizard(controller, onboarding_service)

    assert root.property("currentStep") == 0

    QMetaObject.invokeMethod(root, "goToNextStep", Qt.DirectConnection)
    app.processEvents()
    fingerprint_label = root.findChild(QObject, "licenseWizardFingerprintValue")
    assert fingerprint_label is not None
    assert fingerprint_label.property("text") == "HW-ABC-123"

    QMetaObject.invokeMethod(root, "goToNextStep", Qt.DirectConnection)
    app.processEvents()
    license_input = root.findChild(QObject, "licenseWizardInput")
    apply_button = root.findChild(QObject, "licenseWizardApplyButton")
    assert license_input is not None and apply_button is not None

    license_input.setProperty("text", "VALID LICENSE JSON")
    QMetaObject.invokeMethod(apply_button, "click", Qt.DirectConnection)
    app.processEvents()

    assert controller.licenseAccepted is True
    assert root.property("currentStep") == 3

    onboarding_service.set_ready()
    QMetaObject.invokeMethod(root, "goToNextStep", Qt.DirectConnection)
    app.processEvents()
    assert root.property("currentStep") == 4

    summary_status = root.findChild(QObject, "licenseWizardSummaryStatus")
    summary_license = root.findChild(QObject, "licenseWizardSummaryLicenseId")
    assert summary_status is not None and summary_license is not None
    assert "Licencja aktywowana" in summary_status.property("text")
    assert "demo-pro" in summary_license.property("text")

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_license_wizard_shows_error_on_invalid_payload(tmp_path: Path) -> None:
    controller = LicensingController(verifier=_FakeVerifier(succeed=False))
    root, engine, app, _ = _load_wizard(controller, _StubOnboardingService())

    QMetaObject.invokeMethod(root, "goToNextStep", Qt.DirectConnection)
    QMetaObject.invokeMethod(root, "goToNextStep", Qt.DirectConnection)
    app.processEvents()

    license_input = root.findChild(QObject, "licenseWizardInput")
    apply_button = root.findChild(QObject, "licenseWizardApplyButton")
    status_label = root.findChild(QObject, "licenseWizardStatusLabel")
    assert license_input is not None and apply_button is not None and status_label is not None

    license_input.setProperty("text", "BŁĘDNA LICENCJA")
    QMetaObject.invokeMethod(apply_button, "click", Qt.DirectConnection)
    app.processEvents()

    assert controller.licenseAccepted is False
    assert root.property("currentStep") == 2
    assert "Podpis licencji jest niepoprawny" in status_label.property("text")

    engine.deleteLater()
    app.quit()
