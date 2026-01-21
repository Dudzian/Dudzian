"""Testy weryfikujące bootstrap PySide6 w trybie offscreen."""
from __future__ import annotations

from pathlib import Path

import pytest
try:
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover - środowiska bez wsparcia GL
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app import AppOptions, BotPysideApplication
from tests.ui_pyside.qml_test_helpers import assert_engine_loaded, collect_engine_warnings


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def test_pyside_app_bootstrap_loads_qml(tmp_path: Path, qt_app_session: object | None) -> None:
    if qt_app_session is None:
        pytest.skip("Brak QApplication; uruchom test QML w izolowanym procesie.")
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


def test_cloud_flag_context_property(qt_app_session: object | None) -> None:
    if qt_app_session is None:
        pytest.skip("Brak QApplication; uruchom test QML w izolowanym procesie.")
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

    qt_app = QApplication.instance()
    if qt_app is not None:
        qt_app.processEvents()
