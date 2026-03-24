from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

try:
    from PySide6.QtCore import QObject, QUrl, Signal
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
except ImportError as exc:  # pragma: no cover - środowisko bez GL/Qt
    pytest.skip(f"PySide6 unavailable: {exc}", allow_module_level=True)

from ui.pyside_app.config import UiAppConfig
from ui.pyside_app.controllers.wizards import ModeWizardController
from tests.ui_pyside.qml_test_helpers import assert_engine_loaded, collect_engine_warnings


class _FakeRuntimeService(QObject):
    decisionsChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[dict[str, object]] = []

    def push_decisions(self, entries: list[dict[str, object]]) -> None:
        self.decisions = entries
        self.decisionsChanged.emit()


@pytest.fixture(autouse=True)
def _force_offscreen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")


def _ensure_app() -> QGuiApplication:
    try:
        app = QGuiApplication.instance()
        if app is None:
            app = QGuiApplication([])
        return app
    except Exception as exc:  # pragma: no cover - środowisko bez backendu GL/Qt
        qt_qpa_platform = os.getenv("QT_QPA_PLATFORM", "<unset>")
        pytest.skip(
            f"Qt runtime unavailable on {sys.platform} (QT_QPA_PLATFORM={qt_qpa_platform}): {exc}",
            allow_module_level=True,
        )


def _ui_config() -> UiAppConfig:
    return UiAppConfig(
        source_path=Path("ui/config/example.yaml"),
        profile="default",
        payload={"runtime_config_path": "config/runtime.yaml"},
        qml_entrypoint=Path("ui/pyside_app/qml/MainWindow.qml"),
        decision_limit=30,
        theme_palette="dark",
    )


def test_mode_wizard_controller_recommendations(tmp_path: Path) -> None:
    runtime = _FakeRuntimeService()
    storage = tmp_path / "ui_mode_wizard_state.json"
    controller = ModeWizardController(
        runtime,
        _ui_config(),
        storage_path=storage,
        definitions_path=Path("config/ui/mode_wizards"),
    )

    assert len(controller.modes) >= 5

    runtime.push_decisions(
        [
            {"environment": "binance_futures", "decision": {"shouldTrade": True}, "side": "buy"},
            {"environment": "binance_futures", "decision": {"shouldTrade": True}, "side": "sell"},
        ]
    )
    assert controller.recommendedModeId == "futures"

    controller.saveResult("futures", {"leverage": "3x"})
    restored = controller.savedAnswers("futures")
    assert restored["leverage"] == "3x"

    controller2 = ModeWizardController(
        runtime,
        _ui_config(),
        storage_path=storage,
        definitions_path=Path("config/ui/mode_wizards"),
    )
    assert controller2.savedAnswers("futures")["leverage"] == "3x"


def test_mode_wizard_qml_loads_without_reference_and_connections_warnings() -> None:
    _ensure_app()
    engine = QQmlApplicationEngine()
    warnings = collect_engine_warnings(engine)
    qml_path = Path("ui/pyside_app/qml/views/ModeWizard.qml").resolve()
    engine.load(QUrl.fromLocalFile(qml_path.as_posix()))
    assert_engine_loaded(engine, warnings, "ModeWizard view failed to load")

    warning_blob = " | ".join(warnings)
    assert "ReferenceError: presetCandidate is not defined" not in warning_blob
    assert 'Detected function "onRecommendationChanged" in Connections element' not in warning_blob
    assert "Unable to assign [undefined] to QObject*" not in warning_blob
