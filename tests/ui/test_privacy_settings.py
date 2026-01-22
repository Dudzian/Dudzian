import json
import os
from pathlib import Path

import pytest

from tests.ui._qt import require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = require_pyside6()

from PySide6.QtCore import QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od systemu
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.telemetry import AnonymousTelemetryCollector
from ui.backend.privacy_settings import PrivacySettingsController


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance() or QApplication([])
    return app


def test_privacy_settings_controller(tmp_path: Path) -> None:
    collector = AnonymousTelemetryCollector(storage_dir=tmp_path)
    controller = PrivacySettingsController(collector=collector)

    assert controller.optInEnabled is False
    controller.setOptIn(True, "HWID-001")
    assert controller.optInEnabled is True

    controller.addPreviewEvent("demo.event", "key", "value", "HWID-001")
    assert controller.queuedEvents == 1
    preview = json.loads(controller.previewJson)
    assert preview[0]["event_type"] == "demo.event"

    export_path = controller.exportTelemetry()
    assert Path(export_path).exists()
    assert controller.queuedEvents == 0


def test_privacy_settings_qml_loads(tmp_path: Path, qapp: QApplication) -> None:
    collector = AnonymousTelemetryCollector(storage_dir=tmp_path)
    controller = PrivacySettingsController(collector=collector)

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("privacySettingsController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "settings" / "PrivacySettings.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))

    assert engine.rootObjects(), "Komponent PrivacySettings powinien się załadować"
