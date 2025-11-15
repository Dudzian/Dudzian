from __future__ import annotations

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ui.pyside_app.config import UiAppConfig
from ui.pyside_app.controllers.wizards import ModeWizardController


class _FakeRuntimeService(QObject):
    decisionsChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.decisions: list[dict[str, object]] = []

    def push_decisions(self, entries: list[dict[str, object]]) -> None:
        self.decisions = entries
        self.decisionsChanged.emit()


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
