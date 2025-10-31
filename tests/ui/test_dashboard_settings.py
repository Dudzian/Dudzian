import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska CI
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak bibliotek systemowych
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from core.config.ui_settings import UISettingsStore
from ui.backend.dashboard_settings import DashboardSettingsController


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:  # pragma: no cover - środowisko testowe
        app = QApplication([])
    return app


def test_dashboard_settings_controller_persists_changes(tmp_path: Path, qapp: QApplication) -> None:
    store_path = tmp_path / "ui_settings.json"
    store = UISettingsStore(store_path)
    controller = DashboardSettingsController(store=store)

    assert controller.cardOrder == ["io_queue", "guardrails", "retraining", "compliance"]
    assert controller.visibleCardOrder == ["io_queue", "guardrails", "retraining", "compliance"]

    controller.moveCard("retraining", -2)
    controller.setCardVisibility("guardrails", False)
    controller.setRefreshIntervalMs(5500)
    controller.setTheme("dark")

    assert controller.cardOrder[0] == "retraining"
    assert controller.visibleCardOrder == ["retraining", "io_queue", "compliance"]

    payload = json.loads(store_path.read_text(encoding="utf-8"))
    assert payload["dashboard"]["card_order"][0] == "retraining"
    assert payload["dashboard"]["hidden_cards"] == ["guardrails"]
    assert payload["dashboard"]["refresh_interval_ms"] == 5500
    assert payload["dashboard"]["theme"] == "dark"


def test_dashboard_settings_qml_loads(tmp_path: Path, qapp: QApplication) -> None:
    store = UISettingsStore(tmp_path / "ui_settings.json")
    controller = DashboardSettingsController(store=store)

    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("settingsController", controller)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "settings" / "DashboardSettings.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))

    assert engine.rootObjects(), "Komponent DashboardSettings powinien się załadować"
