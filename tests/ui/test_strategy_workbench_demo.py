from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")
pytest.importorskip("playwright.sync_api", reason="Wymagany Playwright do testów end-to-end")

from PySide6.QtCore import QObject, QUrl, QMetaObject, Qt, Q_ARG
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication
from playwright.sync_api import sync_playwright


@pytest.mark.timeout(30)
def test_workbench_demo_mode_roundtrip() -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    workbench_qml = (
        Path(__file__).resolve().parents[2]
        / "ui"
        / "qml"
        / "components"
        / "workbench"
        / "StrategyWorkbench.qml"
    )
    engine.load(QUrl.fromLocalFile(str(workbench_qml)))
    assert engine.rootObjects(), "Nie udało się załadować widoku StrategyWorkbench"

    root_object = engine.rootObjects()[0]
    assert isinstance(root_object, QObject)

    view_model = root_object.findChild(QObject, "strategyWorkbenchViewModel")
    assert view_model is not None, "ViewModel nie został odnaleziony"

    assert QMetaObject.invokeMethod(
        view_model,
        "activateDemoMode",
        Qt.DirectConnection,
        Q_ARG(str, "momentum"),
    ) is True
    assert view_model.property("demoModeActive") is True
    assert view_model.property("demoModeTitle") == "Momentum Pro"

    instrument_details = view_model.property("instrumentDetails")
    assert instrument_details["exchange"] == "BINANCE"
    assert instrument_details["venueSymbol"] == "BTCUSDT"
    assert instrument_details["quoteCurrency"] == "USDT"

    portfolio_summary = view_model.property("portfolioSummary")
    assert portfolio_summary["maxExposureUtilization"] == pytest.approx(0.82, rel=1e-3)

    assert QMetaObject.invokeMethod(
        view_model,
        "startScheduler",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["schedulerRunning"] is True
    assert control_state["lastActionSuccess"] is True

    assert QMetaObject.invokeMethod(
        view_model,
        "triggerRiskRefresh",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["manualRefreshCount"] == 3
    assert control_state["lastActionMessage"] == "Zainicjowano odświeżenie ryzyka"

    assert QMetaObject.invokeMethod(
        view_model,
        "stopScheduler",
        Qt.DirectConnection,
    ) is True

    control_state = view_model.property("controlState")
    assert control_state["schedulerRunning"] is False
    assert control_state["lastActionSuccess"] is True

    with sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - zależne od środowiska CI
            pytest.skip(f"Brak zainstalowanej przeglądarki Playwright: {exc}")
        page = browser.new_page()
        page.set_content("<html><body><span id='mode'></span></body></html>")
        page.eval_on_selector("#mode", "(el, value) => el.textContent = value", view_model.property("demoModeTitle"))
        assert page.text_content("#mode") == "Momentum Pro"
        browser.close()

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
