from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.ui._qt import qml_value_to_python, require_pyside6
from tests.ui._qt_utils import teardown_qml_engine

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

require_pyside6()

from PySide6.QtCore import QObject, QUrl, QMetaObject, Qt, Q_ARG
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtWidgets import QApplication


@pytest.fixture
def workbench_demo() -> Iterator[tuple[QQmlApplicationEngine, QObject]]:
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
    try:
        engine.load(QUrl.fromLocalFile(str(workbench_qml)))
        assert engine.rootObjects(), "Nie udało się załadować widoku StrategyWorkbench"

        root_object = engine.rootObjects()[0]
        assert isinstance(root_object, QObject)

        view_model = root_object.findChild(QObject, "strategyWorkbenchViewModel")
        assert view_model is not None, "ViewModel nie został odnaleziony"

        assert (
            QMetaObject.invokeMethod(
                view_model,
                "activateDemoMode",
                Qt.DirectConnection,
                Q_ARG("QVariant", "momentum"),
            )
            is True
        )
        assert view_model.property("demoModeActive") is True
        assert view_model.property("demoModeTitle") == "Momentum Pro"
        yield engine, view_model
    finally:
        teardown_qml_engine(engine, process_events=app.processEvents)
        app.processEvents()


@pytest.mark.timeout(30)
def test_workbench_demo_mode_roundtrip(
    workbench_demo: tuple[QQmlApplicationEngine, QObject],
) -> None:
    _, view_model = workbench_demo

    instrument_details = qml_value_to_python(view_model.property("instrumentDetails"))
    assert instrument_details["exchange"] == "BINANCE"
    assert instrument_details["venueSymbol"] == "BTCUSDT"
    assert instrument_details["quoteCurrency"] == "USDT"

    portfolio_summary = qml_value_to_python(view_model.property("portfolioSummary"))
    assert portfolio_summary["maxExposureUtilization"] == pytest.approx(0.82, rel=1e-3)

    assert (
        QMetaObject.invokeMethod(
            view_model,
            "startScheduler",
            Qt.DirectConnection,
        )
        is True
    )

    control_state = qml_value_to_python(view_model.property("controlState"))
    assert control_state["schedulerRunning"] is True
    assert control_state["lastActionSuccess"] is True

    assert (
        QMetaObject.invokeMethod(
            view_model,
            "triggerRiskRefresh",
            Qt.DirectConnection,
        )
        is True
    )

    control_state = qml_value_to_python(view_model.property("controlState"))
    assert control_state["manualRefreshCount"] == 3
    assert control_state["lastActionMessage"] == "Zainicjowano odświeżenie ryzyka"

    assert (
        QMetaObject.invokeMethod(
            view_model,
            "stopScheduler",
            Qt.DirectConnection,
        )
        is True
    )

    control_state = qml_value_to_python(view_model.property("controlState"))
    assert control_state["schedulerRunning"] is False
    assert control_state["lastActionSuccess"] is True


@pytest.mark.timeout(30)
def test_workbench_demo_title_in_browser(
    workbench_demo: tuple[QQmlApplicationEngine, QObject],
) -> None:
    playwright_api = pytest.importorskip(
        "playwright.sync_api", reason="Wymagany Playwright do testów end-to-end"
    )
    _, view_model = workbench_demo
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except Exception as exc:  # pragma: no cover - zależne od środowiska CI
            pytest.skip(f"Brak zainstalowanej przeglądarki Playwright: {exc}")
        try:
            page = browser.new_page()
            page.set_content("<html><body><span id='mode'></span></body></html>")
            page.eval_on_selector(
                "#mode",
                "(el, value) => el.textContent = value",
                view_model.property("demoModeTitle"),
            )
            assert page.text_content("#mode") == "Momentum Pro"
        finally:
            browser.close()
