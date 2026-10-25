"""Testy weryfikujące bootstrap PySide6 w trybie offscreen."""

from __future__ import annotations

import os
import sys
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


def _ensure_qt_application() -> QGuiApplication:
    try:
        app = QGuiApplication.instance()
        if app is None:  # pragma: no cover - w praktyce tworzy się przy pierwszym bootstrapie
            app = QGuiApplication([])
        return app
    except Exception as exc:  # pragma: no cover - środowiska bez backendu GL/Qt
        qt_qpa_platform = os.getenv("QT_QPA_PLATFORM", "<unset>")
        pytest.skip(
            f"Qt runtime unavailable on {sys.platform} (QT_QPA_PLATFORM={qt_qpa_platform}): {exc}",
            allow_module_level=True,
        )


def test_pyside_app_bootstrap_loads_qml(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    qt_app = _ensure_qt_application()
    options = AppOptions(config_path=Path("ui/config/example.yaml"))
    app = BotPysideApplication(options)
    load_warnings: list[str] = []
    engine = app.load(warning_sink=load_warnings.append)
    request.addfinalizer(lambda: teardown_qml_engine(engine, process_events=qt_app.processEvents))
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


def test_cloud_flag_context_property(request: pytest.FixtureRequest) -> None:
    options = AppOptions(
        config_path=Path("ui/config/example.yaml"),
        enable_cloud_runtime=True,
    )
    app = BotPysideApplication(options)
    engine = app.load()
    qt_app = _ensure_qt_application()
    request.addfinalizer(lambda: teardown_qml_engine(engine, process_events=qt_app.processEvents))
    ctx = engine.rootContext()
    assert ctx.contextProperty("cloudRuntimeEnabled") is True
    licensing_controller = ctx.contextProperty("licensingController")
    assert licensing_controller is not None
    licensing_controller.resetStatus()

    qt_app.processEvents()
