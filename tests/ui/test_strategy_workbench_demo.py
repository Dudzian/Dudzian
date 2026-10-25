from __future__ import annotations

import os
import importlib
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.qml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_OPTIONAL_BROWSER_SKIP = "WORKBENCH_OPTIONAL_BROWSER_SKIP: "
_SNAPSHOT_PROPERTIES = (
    "schedulerEntries",
    "exchangeConnections",
    "aiConfiguration",
    "portfolioSummary",
    "riskSnapshot",
    "runtimeStatus",
    "licenseStatus",
)


def _assert_native_result(result: subprocess.CompletedProcess[str], *, browser: bool) -> None:
    if result.returncode == 77 and browser:
        for line in result.stdout.splitlines():
            if line.startswith(_OPTIONAL_BROWSER_SKIP):
                pytest.skip(line.removeprefix(_OPTIONAL_BROWSER_SKIP))
    assert result.returncode == 0, (
        f"Workbench native QML process failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _run_workbench_case(case: str) -> None:
    env = os.environ.copy()
    # The dedicated native case must execute; missing Qt is a failure in this
    # process, including when the parent suite has not requested QML explicitly.
    env["PYTEST_REQUIRE_QML"] = "1"
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("QT_QUICK_BACKEND", "software")
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-m", "tests.ui.test_strategy_workbench_demo", case],
        cwd=_REPOSITORY_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    # Keep native Qt diagnostics visible in the parent pytest artifacts.
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    _assert_native_result(result, browser=case == "browser")


@contextmanager
def _qt_diagnostics() -> Iterator[None]:
    from tests.ui._qt import require_pyside6

    require_pyside6(modules=("QtCore", "QtQml", "QtWidgets"))
    from PySide6.QtCore import QtMsgType, qInstallMessageHandler

    diagnostics: list[str] = []
    previous_handler = None

    def record(message_type: Any, context: Any, message: str) -> None:
        if message_type in {QtMsgType.QtWarningMsg, QtMsgType.QtCriticalMsg, QtMsgType.QtFatalMsg}:
            diagnostics.append(message)
        if previous_handler is not None:
            previous_handler(message_type, context, message)
        else:
            sys.stderr.write(message + "\n")

    previous_handler = qInstallMessageHandler(record)
    try:
        yield
    finally:
        qInstallMessageHandler(previous_handler)
        active_error = sys.exc_info()[1]
        if active_error is None or isinstance(active_error, pytest.skip.Exception):
            assert not diagnostics, "Native Workbench Qt diagnostics:\n" + "\n".join(diagnostics)


@contextmanager
def _workbench_demo() -> Iterator[tuple[Any, Any]]:
    # Import native dependencies only in the fresh process. Ordinary Python and
    # PostgreSQL shards can collect/deselect this module without loading EGL.
    from tests.ui._qt import require_pyside6

    require_pyside6(modules=("QtCore", "QtQml", "QtWidgets"))
    from PySide6.QtCore import QObject, QUrl
    from PySide6.QtQml import QQmlApplicationEngine
    from PySide6.QtWidgets import QApplication

    from tests.ui._qt_utils import teardown_qml_engine

    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication), (
        "Workbench requires a GUI application, not QCoreApplication"
    )
    # Retain the application and make engine ownership explicit for its complete
    # lifetime. This process has no deferred deletes from preceding QML tests.
    engine = QQmlApplicationEngine(app)
    workbench_qml = (
        _REPOSITORY_ROOT / "ui" / "qml" / "components" / "workbench" / "StrategyWorkbench.qml"
    )
    try:
        engine.load(QUrl.fromLocalFile(str(workbench_qml)))
        assert engine.rootObjects(), "Nie udało się załadować widoku StrategyWorkbench"
        root_object = engine.rootObjects()[0]
        assert isinstance(root_object, QObject)
        view_model = root_object.findChild(QObject, "strategyWorkbenchViewModel")
        assert view_model is not None, "ViewModel nie został odnaleziony"
        app.processEvents()
        yield engine, view_model
    finally:
        teardown_qml_engine(engine, process_events=app.processEvents)
        app.processEvents()


def _invoke(view_model: Any, method: str, *args: str) -> Any:
    from PySide6.QtCore import Q_ARG, Q_RETURN_ARG, QMetaObject, Qt

    # QML functions return QVariant. Capture the actual return value rather than
    # treating successful Qt method dispatch as a successful demo activation.
    return QMetaObject.invokeMethod(
        view_model,
        method,
        Qt.DirectConnection,
        Q_RETURN_ARG("QVariant"),
        *(Q_ARG("QVariant", value) for value in args),
    )


def _snapshot(view_model: Any) -> dict[str, Any]:
    from tests.ui._qt import qml_value_to_python

    properties = (
        "demoModeActive",
        "demoModeId",
        "demoModeTitle",
        "demoModeDescription",
        *_SNAPSHOT_PROPERTIES,
    )
    return {name: qml_value_to_python(view_model.property(name)) for name in properties}


def _assert_demo_snapshot(snapshot: dict[str, Any], preset_id: str) -> None:
    momentum = preset_id == "momentum"
    assert snapshot["demoModeActive"] is True
    assert snapshot["demoModeId"] == preset_id
    assert snapshot["demoModeTitle"] == ("Momentum Pro" if momentum else "Market Neutral")
    assert snapshot["demoModeDescription"]

    scheduler = snapshot["schedulerEntries"]
    assert [item["name"] for item in scheduler] == (
        ["Momentum Alpha", "Momentum Beta"] if momentum else ["Delta Hedge", "Gamma Overlay"]
    )
    assert [item["enabled"] for item in scheduler] == ([True, True] if momentum else [True, False])
    assert [item["scheduleCount"] for item in scheduler] == ([4, 2] if momentum else [6, 1])

    exchanges = snapshot["exchangeConnections"]
    assert [item["exchange"] for item in exchanges] == (
        ["BINANCE", "COINBASE"] if momentum else ["OKX", "DERIBIT"]
    )
    assert exchanges[0]["symbol"] == "BTC/USDT"
    assert exchanges[0]["venueSymbol"] == ("BTCUSDT" if momentum else "BTC-USDT-SWAP")
    assert exchanges[0]["symbol"].split("/")[1] == "USDT"

    ai = snapshot["aiConfiguration"]
    assert ai["policy"] == ("momentum" if momentum else "market_neutral")
    assert ai["risk_profile"] == ("growth" if momentum else "conservative")
    assert ai["decision_window"] == ("PT15S" if momentum else "PT1M")

    portfolio = snapshot["portfolioSummary"]
    assert portfolio["latestValue"] == (129800 if momentum else 203450)
    assert portfolio["maxDrawdown"] == pytest.approx(0.083 if momentum else 0.019)
    assert portfolio["maxLeverage"] == pytest.approx(2.4 if momentum else 1.2)
    assert portfolio["anyBreach"] is False

    risk = snapshot["riskSnapshot"]
    assert risk["profileLabel"] == ("Momentum" if momentum else "Neutral")
    assert risk["currentDrawdown"] == pytest.approx(0.021 if momentum else 0.008)
    assert len(risk["exposures"]) == 3
    if not momentum:
        assert risk["exposures"][1]["code"] == "ETH_DELTA"
        assert risk["exposures"][1]["current"] == pytest.approx(-0.01)

    runtime = snapshot["runtimeStatus"]
    assert runtime["connection"] == (
        "Demo: połączenie symulacyjne" if momentum else "Demo: tryb market neutral"
    )
    assert runtime["riskRefresh"]["enabled"] is True
    assert runtime["riskRefresh"]["intervalSeconds"] == (30 if momentum else 45)
    assert runtime["automationRunning"] is True

    license_status = snapshot["licenseStatus"]
    assert license_status["active"] is True
    assert license_status["licenseId"] == ("DEMO-MOMENTUM" if momentum else "DEMO-HEDGE")
    assert license_status["modules"] == (
        ["momentum", "portfolio", "scheduler"] if momentum else ["hedge", "risk", "scheduler"]
    )


def _assert_roundtrip(engine: Any, view_model: Any) -> None:
    from PySide6.QtCore import QObject, QMetaObject, Qt
    from PySide6.QtQml import QQmlExpression
    from PySide6.QtWidgets import QApplication

    from tests.ui._qt import qml_value_to_python

    initial = _snapshot(view_model)
    assert initial["demoModeActive"] is False
    assert initial["demoModeId"] == ""
    momentum_snapshot = None
    for preset_id in ("momentum", "hedge", "momentum"):
        assert _invoke(view_model, "activateDemoMode", preset_id) is True
        app = QApplication.instance()
        app.processEvents()
        active = _snapshot(view_model)
        _assert_demo_snapshot(active, preset_id)
        list_view = engine.rootObjects()[0].findChild(QObject, "strategyListView")
        assert list_view is not None
        # A standalone Item has no window layout to give this ListView height.
        # Lay it out explicitly so actual delegates, including the last-row
        # separator binding, are exercised instead of merely inspecting data.
        list_view.setProperty("height", 200)
        assert QMetaObject.invokeMethod(list_view, "forceLayout", Qt.DirectConnection) is True
        app.processEvents()
        # Read scalar values inside QML: delegate objects can be replaced when
        # models switch, and their Python wrappers must not outlive those items.
        expression = QQmlExpression(
            engine.rootContext(),
            list_view,
            """(function() {
                var values = [];
                for (var i = 0; i < contentItem.children.length; ++i) {
                    var children = contentItem.children[i].children;
                    for (var j = 0; j < children.length; ++j)
                        if (children[j].objectName === "strategyEntrySeparator")
                            values.push(children[j].opacity);
                }
                return values;
            })()""",
        )
        opacities, undefined = expression.evaluate()
        assert not expression.hasError(), expression.error().toString()
        assert undefined is False
        assert sorted(qml_value_to_python(opacities)) == [0.0, 0.4]
        if preset_id == "momentum":
            if momentum_snapshot is None:
                momentum_snapshot = active
            else:
                assert active == momentum_snapshot
        # A live refresh must not overwrite a currently active demo.
        _invoke(view_model, "refreshFromLive")
        assert _snapshot(view_model) == active
        for invalid_id in ("unknown-preset", ""):
            assert _invoke(view_model, "activateDemoMode", invalid_id) is False
            assert _snapshot(view_model) == active

    _invoke(view_model, "disableDemoMode")
    disabled = _snapshot(view_model)
    assert disabled["demoModeActive"] is False
    assert disabled["demoModeId"] == ""
    assert disabled["demoModeTitle"] == ""
    assert disabled["demoModeDescription"] == ""
    # This view has no injected live controllers. Leaving demo must clear its
    # simulated data, including the active demo license and risk snapshot.
    for name in _SNAPSHOT_PROPERTIES:
        assert disabled[name] == ([] if name in {"schedulerEntries", "exchangeConnections"} else {})
    _invoke(view_model, "disableDemoMode")
    assert _snapshot(view_model) == disabled


def _assert_catalog_status(engine: Any, view_model: Any) -> None:
    from PySide6.QtCore import QObject, Property, Signal, Slot
    from PySide6.QtWidgets import QApplication

    class CatalogStatus(QObject):
        busyChanged = Signal()
        lastErrorChanged = Signal()
        catalogChanged = Signal()
        lastValidationChanged = Signal()
        presetHistoryChanged = Signal()

        def __init__(self) -> None:
            super().__init__()
            self.busy_value = False
            self.error_value = ""
            self.clear_calls = 0

        @Property(bool, notify=busyChanged)
        def busy(self) -> bool:
            return self.busy_value

        @Property(str, notify=lastErrorChanged)
        def lastError(self) -> str:
            return self.error_value

        @Property("QVariantList", notify=catalogChanged)
        def catalogDefinitions(self) -> list[dict[str, str]]:
            return [{"name": "read-only-catalog", "engine": "test"}]

        @Slot()
        def clearLastError(self) -> None:
            self.clear_calls += 1
            self.error_value = ""
            self.lastErrorChanged.emit()

    app = QApplication.instance()
    root = engine.rootObjects()[0]
    scroll = root.findChild(QObject, "workbenchScroll")
    assert scroll is not None
    assert view_model.property("workbenchBusy") is False
    assert view_model.property("workbenchError") == ""

    assert scroll.property("enabled") is False
    controller = CatalogStatus()
    try:
        assert root.setProperty("workbenchControllerRef", controller)
        app.processEvents()
        assert view_model.property("catalogReady") is True
        assert scroll.property("enabled") is True
        controller.busy_value = True
        controller.busyChanged.emit()
        app.processEvents()
        assert view_model.property("workbenchBusy") is True
        assert scroll.property("enabled") is False
        controller.error_value = "Catalog bridge unavailable"
        controller.lastErrorChanged.emit()
        app.processEvents()
        assert view_model.property("workbenchError") == "Catalog bridge unavailable"
        _invoke(view_model, "clearWorkbenchError")
        assert controller.clear_calls == 1
        assert view_model.property("workbenchError") == ""
        controller.busy_value = False
        controller.busyChanged.emit()
        app.processEvents()
        assert view_model.property("workbenchBusy") is False
        assert scroll.property("enabled") is True
    finally:
        root.setProperty("workbenchControllerRef", None)
        app.processEvents()
    assert view_model.property("workbenchBusy") is False
    assert view_model.property("workbenchError") == ""


def _assert_alert_controls(engine: Any) -> None:
    from PySide6.QtCore import QObject, QMetaObject, Qt
    from PySide6.QtWidgets import QApplication

    from tests.ui._qt import qml_value_to_python

    root = engine.rootObjects()[0]
    panel = root.findChild(QObject, "autoModePanel")
    slider = root.findChild(QObject, "autoModeDrawdownSlider")
    issues = root.findChild(QObject, "autoModeLicenseIssues")
    assert panel is not None and slider is not None and issues is not None
    app = QApplication.instance()
    original_draft = qml_value_to_python(panel.property("alertDraft"))
    assert original_draft == {"severity": "warning", "maxDrawdown": 0.07, "notifyOnSwitch": True}
    assert panel.setProperty("alertDraft", {**original_draft, "severity": "critical"})
    app.processEvents()
    slider.setProperty("value", 0.12)
    app.processEvents()
    assert qml_value_to_python(panel.property("alertDraft")) == {
        **original_draft,
        "severity": "critical",
    }
    assert QMetaObject.invokeMethod(slider, "moved", Qt.DirectConnection) is True
    app.processEvents()
    assert qml_value_to_python(panel.property("alertDraft")) == {
        **original_draft,
        "severity": "critical",
        "maxDrawdown": pytest.approx(0.12),
    }
    panel.setProperty("alertDraft", {**original_draft, "maxDrawdown": 0.09})
    app.processEvents()
    assert slider.property("value") == pytest.approx(0.09)
    panel.setProperty("alertDraft", original_draft)

    assert issues.property("visible") is False
    panel.setProperty("activePreset", {"license_issues": ["test-only license issue"]})
    app.processEvents()
    assert issues.property("visible") is True
    assert "test-only license issue" in issues.property("text")
    panel.setProperty("activePreset", {"license_issues": []})
    app.processEvents()
    assert issues.property("visible") is False
    panel.setProperty("activePreset", {})


def _is_missing_browser_executable(error: BaseException) -> bool:
    return str(error).startswith("BrowserType.launch: Executable doesn't exist at ")


def _assert_browser_title(view_model: Any) -> None:
    assert _invoke(view_model, "activateDemoMode", "momentum") is True
    assert view_model.property("demoModeTitle") == "Momentum Pro"
    try:
        playwright_api = importlib.import_module("playwright.sync_api")
    except ModuleNotFoundError as exc:
        if exc.name not in {"playwright", "playwright.sync_api"}:
            raise
        pytest.skip("Wymagany Playwright do testów end-to-end")
    with playwright_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(headless=True)
        except playwright_api.Error as exc:  # pragma: no cover - zależne od środowiska CI
            if not _is_missing_browser_executable(exc):
                raise
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


@pytest.mark.timeout(75)
def test_workbench_demo_mode_roundtrip() -> None:
    _run_workbench_case("roundtrip")


@pytest.mark.timeout(75)
def test_workbench_demo_title_in_browser() -> None:
    _run_workbench_case("browser")


def _main(case: str) -> int:
    assert case in {"roundtrip", "browser"}, f"Unknown Workbench case: {case}"
    try:
        with _qt_diagnostics():
            with _workbench_demo() as (engine, view_model):
                if case == "roundtrip":
                    _assert_roundtrip(engine, view_model)
                    _assert_catalog_status(engine, view_model)
                    _assert_alert_controls(engine)
                else:
                    _assert_browser_title(view_model)
    except pytest.skip.Exception as exc:
        if case != "browser":
            raise
        print(_OPTIONAL_BROWSER_SKIP + str(exc), flush=True)
        return 77
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1]))
