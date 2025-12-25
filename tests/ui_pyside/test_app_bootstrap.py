"""Testy weryfikujące bootstrap PySide6 w trybie offscreen."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
try:
    from PySide6.QtGui import QGuiApplication
except ImportError as exc:  # pragma: no cover - środowiska bez wsparcia GL
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app import AppOptions, BotPysideApplication
from tests.ui_pyside.qml_test_helpers import assert_engine_loaded, collect_engine_warnings


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def _ensure_qt_application() -> QGuiApplication:
    try:
        app = QGuiApplication.instance()
        if app is None:  # pragma: no cover - w praktyce tworzy się przy pierwszym bootstrapie
            app = QGuiApplication([])
        return app
    except Exception as exc:  # pragma: no cover - środowiska bez backendu GL/Qt
        pytest.skip(
            f"Qt runtime unavailable on {sys.platform}: {exc}",
            allow_module_level=True,
        )


def test_pyside_app_bootstrap_loads_qml(tmp_path: Path) -> None:
    options = AppOptions(config_path=Path("ui/config/example.yaml"))
    app = BotPysideApplication(options)
    engine = app.load()
    warnings = collect_engine_warnings(engine)
    assert_engine_loaded(engine, warnings, "QML nie został załadowany")
    ctx = engine.rootContext()
    grpc_bridge = ctx.contextProperty("grpcBridge")
    runtime_service = grpc_bridge.runtimeService if grpc_bridge else None
    assert runtime_service is not None
    runtime_service.loadRecentDecisions(5)
    decisions = runtime_service.decisions
    assert isinstance(decisions, list)
    mode_controller = ctx.contextProperty("modeWizardController")
    assert mode_controller is not None
    strategy_manager = ctx.contextProperty("strategyManagementController")
    assert strategy_manager is not None
    cloud_flag = ctx.contextProperty("cloudRuntimeEnabled")
    assert cloud_flag is False


def test_cloud_flag_context_property() -> None:
    options = AppOptions(
        config_path=Path("ui/config/example.yaml"),
        enable_cloud_runtime=True,
    )
    app = BotPysideApplication(options)
    engine = app.load()
    ctx = engine.rootContext()
    assert ctx.contextProperty("cloudRuntimeEnabled") is True
    licensing_controller = ctx.contextProperty("licensingController")
    assert licensing_controller is not None
    licensing_controller.resetStatus()

    qt_app = _ensure_qt_application()
    qt_app.processEvents()
