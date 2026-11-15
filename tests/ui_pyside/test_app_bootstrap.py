"""Testy weryfikujące bootstrap PySide6 w trybie offscreen."""

from __future__ import annotations

from functools import partial
from pathlib import Path

import pytest

try:
    from PySide6.QtCore import QObject
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlEngine, QQmlExpression
    from PySide6.QtQuick import QQuickItem
except ImportError as exc:  # pragma: no cover - środowiska bez wsparcia GL
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app import AppOptions, BotPysideApplication
from tests._qt_utils import teardown_qml_engine
from tests.ui_pyside.qml_test_helpers import assert_engine_loaded, collect_engine_warnings


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(autouse=True)
def _isolate_mode_wizard_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ui.pyside_app import qml_bridge

    monkeypatch.setattr(
        qml_bridge,
        "ModeWizardController",
        partial(qml_bridge.ModeWizardController, storage_path=tmp_path / "mode-wizard.json"),
    )


def _teardown_application(app: BotPysideApplication, qt_app: QGuiApplication) -> None:
    """Stop owned workers while their Python/Qt owners and engine are alive."""
    engine = app.engine
    if engine is None:
        return
    bridge = engine.rootContext().contextProperty("grpcBridge")
    service = bridge.runtimeService
    workers = tuple(
        worker for worker in (service._grpc_thread, service._ai_feed_thread) if worker is not None
    )
    try:
        service._stop_grpc_stream()
        for worker in workers:
            # A connection-ready wait has its own bounded deadline, which can exceed
            # the short best-effort join used by the product stop method.
            worker.join(timeout=service._grpc_ready_timeout + 1.0)
    finally:
        teardown_qml_engine(engine, process_events=qt_app.processEvents)
    assert all(not worker.is_alive() for worker in workers), "UI test leaked a runtime worker"
    # The app argument retains its bridge and all Python-owned context objects
    # until the native QML roots/engine have actually been destroyed.


def test_pyside_app_bootstrap_loads_qml(
    tmp_path: Path, request: pytest.FixtureRequest, qt_gui_application: QGuiApplication
) -> None:
    qt_app = qt_gui_application
    options = AppOptions(config_path=Path("ui/config/example.yaml"))
    app = BotPysideApplication(options)
    load_warnings: list[str] = []
    engine = app.load(warning_sink=load_warnings.append)
    request.addfinalizer(lambda: _teardown_application(app, qt_app))
    warnings = load_warnings + collect_engine_warnings(engine)
    warnings = list(dict.fromkeys(warnings))
    assert_engine_loaded(engine, warnings, "QML nie został załadowany")
    assert not warnings, " | ".join(warnings)

    ctx = engine.rootContext()
    grpc_bridge = ctx.contextProperty("grpcBridge")
    runtime_service = grpc_bridge.runtimeService if grpc_bridge else None
    assert runtime_service is not None
    runtime_service.loadRecentDecisions(5)
    decisions = runtime_service.decisions
    assert isinstance(decisions, list)
    mode_controller = ctx.contextProperty("modeWizardController")
    assert mode_controller is not None
    root = engine.rootObjects()[0]
    runtime_panels = []
    for component_name in (
        "strategiesPanelComponent",
        "riskControlsPanelComponent",
        "aiDecisionsPanelComponent",
    ):
        expression = QQmlExpression(
            QQmlEngine.contextForObject(root),
            root,
            component_name + ".createObject(root.contentItem, {width: 1200, height: 800})",
        )
        value, undefined = expression.evaluate()
        assert not expression.hasError(), expression.error().toString()
        assert not undefined
        panel = value if isinstance(value, QObject) else value.toQObject()
        assert panel is not None
        runtime_panels.append(panel)
        assert panel.property("runtimeService") is runtime_service
    # These are live parent bindings, not copies captured at creation time.
    try:
        clear_service = QQmlExpression(
            QQmlEngine.contextForObject(root), root, "root.runtimeService = null"
        )
        clear_service.evaluate()
        assert not clear_service.hasError(), clear_service.error().toString()
        qt_app.processEvents()
        assert all(panel.property("runtimeService") is None for panel in runtime_panels)
    finally:
        root.setProperty("runtimeService", runtime_service)
        qt_app.processEvents()
    assert all(panel.property("runtimeService") is runtime_service for panel in runtime_panels)
    mode_wizard = root.findChild(QObject, "modeWizardPreviewPanel")
    assert mode_wizard is not None
    assert mode_wizard.property("modeWizardController") is mode_controller
    dock_manager = root.findChild(QQuickItem, "previewDockManager")
    assert dock_manager is not None
    assert dock_manager.property("layoutController") is ctx.contextProperty("layoutController")
    dock_manager.setProperty("visible", True)
    for _ in range(3):
        qt_app.processEvents()
    pending = list(dock_manager.childItems())
    panels = []
    while pending:
        item = pending.pop()
        pending.extend(item.childItems())
        if item.metaObject().indexOfProperty("dockManager") >= 0:
            panels.append(item)
    assert panels, "The registered layout must create panel delegates"
    for panel in panels:
        assert panel.property("dockManager") is dock_manager
        assert panel.property("panelId")
    assert not load_warnings, " | ".join(load_warnings)
    dock_manager.setProperty("visible", False)
    strategy_manager = ctx.contextProperty("strategyManagementController")
    assert strategy_manager is not None
    cloud_flag = ctx.contextProperty("cloudRuntimeEnabled")
    assert cloud_flag is False


def test_cloud_flag_context_property(
    request: pytest.FixtureRequest, qt_gui_application: QGuiApplication
) -> None:
    options = AppOptions(
        config_path=Path("ui/config/example.yaml"),
        enable_cloud_runtime=True,
    )
    app = BotPysideApplication(options)
    engine = app.load()
    qt_app = qt_gui_application
    request.addfinalizer(lambda: _teardown_application(app, qt_app))
    ctx = engine.rootContext()
    assert ctx.contextProperty("cloudRuntimeEnabled") is True
    licensing_controller = ctx.contextProperty("licensingController")
    assert licensing_controller is not None
    licensing_controller.resetStatus()

    qt_app.processEvents()


def test_bootstrap_teardown_joins_owned_runtime_workers(
    qt_gui_application: QGuiApplication,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("BOT_CORE_UI_GRPC_READY_TIMEOUT", "1")
    app = BotPysideApplication(AppOptions(config_path=Path("ui/config/example.yaml")))
    engine = app.load()
    service = engine.rootContext().contextProperty("grpcBridge").runtimeService
    try:
        service._start_grpc_stream("127.0.0.1:1", 5)
        workers = tuple(
            worker
            for worker in (service._grpc_thread, service._ai_feed_thread)
            if worker is not None
        )
        assert workers, "The lifecycle regression must exercise actual runtime workers"
    finally:
        _teardown_application(app, qt_gui_application)
    assert all(not worker.is_alive() for worker in workers)
    assert service._grpc_client is None
    assert service._grpc_thread is None
    assert service._ai_feed_thread is None
