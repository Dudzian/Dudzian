from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.ui._qt import require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

require_pyside6()

from PySide6.QtCore import QObject, Property, Qt, QMetaObject, QUrl, Slot
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtTest import QSignalSpy
except ImportError as exc:  # brak bibliotek systemowych (np. libGL)
    pytest.skip(f"Brak zależności QtTest: {exc}", allow_module_level=True)

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # brak bibliotek systemowych (np. libGL)
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)


class StubLicenseController(QObject):
    def __init__(self, active: bool = True) -> None:
        super().__init__()
        self._active = active

    @Property(bool, constant=False)
    def licenseActive(self) -> bool:  # type: ignore[override]
        return self._active

    @licenseActive.setter  # type: ignore[misc]
    def licenseActive(self, value: bool) -> None:
        self._active = bool(value)

    @Slot()
    def refreshLicenseStatus(self) -> None:
        self._active = True


class StubAppController(QObject):
    def __init__(self) -> None:
        super().__init__()
        self._exchanges = ["BINANCE", "KRAKEN"]
        self._theme = "dark"
        self._layout = "classic"
        self._alerts = True
        self.list_calls: list[str] = []
        self.last_update: tuple[str, str, str, str, str, str] | None = None

    @Slot(result="QVariantList")
    def supportedExchanges(self) -> list[str]:
        return self._exchanges

    @Slot(str, result="QVariantList")
    def listTradableInstruments(self, exchange: str) -> list[dict[str, object]]:
        self.list_calls.append(exchange)
        return [
            {
                "symbol": "BTC/USDT",
                "priceStep": "0.1",
                "minNotional": "50",
                "config": {
                    "exchange": exchange,
                    "symbol": "BTCUSDT",
                    "venueSymbol": "BTCUSDT",
                    "quoteCurrency": "USDT",
                    "baseCurrency": "BTC",
                    "granularityIso8601": "PT1M",
                },
            }
        ]

    @Slot(str, str, str, str, str, str, result=bool)
    def updateInstrument(
        self,
        exchange: str,
        symbol: str,
        venue_symbol: str,
        quote_currency: str,
        base_currency: str,
        granularity: str,
    ) -> bool:
        self.last_update = (exchange, symbol, venue_symbol, quote_currency, base_currency, granularity)
        return True

    @Slot(result="QVariantMap")
    def personalizationSnapshot(self) -> dict[str, object]:
        return {
            "theme": self._theme,
            "layout": self._layout,
            "alert_toasts": self._alerts,
        }

    @Slot(str, result=bool)
    def setUiTheme(self, theme: str) -> bool:
        self._theme = theme
        return True

    @Slot(str, result=bool)
    def setUiLayoutMode(self, mode: str) -> bool:
        self._layout = mode
        return True

    @Slot(bool)
    def setAlertToastsEnabled(self, enabled: bool) -> None:
        self._alerts = bool(enabled)

    @Property(str, constant=False)
    def uiTheme(self) -> str:  # type: ignore[override]
        return self._theme

    @Property(str, constant=False)
    def uiLayoutMode(self) -> str:  # type: ignore[override]
        return self._layout

    @Property(bool, constant=False)
    def alertToastsEnabled(self) -> bool:  # type: ignore[override]
        return self._alerts


@pytest.mark.timeout(30)
def test_setup_wizard_configures_instrument_and_preferences(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    stub_app = StubAppController()
    license_stub = StubLicenseController(active=True)

    context = engine.rootContext()
    context.setContextProperty("appController", stub_app)
    context.setContextProperty("licenseController", license_stub)

    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "SetupWizard.qml"
    engine.load(QUrl.fromLocalFile(str(view_path)))
    assert engine.rootObjects(), "Nie udało się załadować kreatora konfiguracji"

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    # Exchange/instrument data
    exchange_combo = root.findChild(QObject, "setupWizardExchangeCombo")
    assert exchange_combo is not None
    assert exchange_combo.property("count") == 2

    instrument_view = root.findChild(QObject, "setupWizardInstrumentView")
    assert instrument_view is not None
    app.processEvents()
    assert stub_app.list_calls, "Lista instrumentów nie została pobrana"

    root.setProperty("selectedInstrumentIndex", 0)
    QMetaObject.invokeMethod(root, "applySelectedInstrument", Qt.DirectConnection)
    app.processEvents()
    assert stub_app.last_update is not None
    assert stub_app.last_update[0] in {"BINANCE", "KRAKEN"}

    # Personalization controls
    theme_combo = root.findChild(QObject, "setupWizardThemeCombo")
    layout_combo = root.findChild(QObject, "setupWizardLayoutCombo")
    toast_switch = root.findChild(QObject, "setupWizardToastSwitch")
    assert theme_combo is not None and layout_combo is not None and toast_switch is not None

    theme_combo.setProperty("currentIndex", 1)  # Jasny motyw
    layout_combo.setProperty("currentIndex", 2)  # Zaawansowany układ
    toast_switch.setProperty("checked", False)
    app.processEvents()

    assert stub_app.uiTheme == "light"
    assert stub_app.uiLayoutMode == "advanced"
    assert stub_app.alertToastsEnabled is False

    # Final step should emit wizardCompleted
    spy = QSignalSpy(root, b"wizardCompleted()")
    root.setProperty("currentStep", 3)
    QMetaObject.invokeMethod(root, "stepCanAdvance", Qt.DirectConnection)
    QMetaObject.invokeMethod(root, "wizardCompleted", Qt.DirectConnection)
    app.processEvents()
    assert spy.count() >= 1

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
