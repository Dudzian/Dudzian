import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

try:  # pragma: no cover - zależne od środowiska
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
except ImportError as exc:  # pragma: no cover - brak QtWidgets
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from ui.backend.diagnostics_controller import DiagnosticsController


@pytest.fixture()
def project_with_data(tmp_path: Path) -> Path:
    base = tmp_path / "project"
    (base / "logs").mkdir(parents=True)
    (base / "config").mkdir(parents=True)
    (base / "reports").mkdir(parents=True)
    (base / "logs" / "app.log").write_text("entry", encoding="utf-8")
    (base / "config" / "runtime.yml").write_text("key: value", encoding="utf-8")
    (base / "reports" / "summary.txt").write_text("report", encoding="utf-8")
    return base


@pytest.mark.timeout(30)
def test_ticket_dialog_generates_package(tmp_path: Path, project_with_data: Path) -> None:
    controller = DiagnosticsController()
    controller.baseDirectory = str(project_with_data)
    controller.outputDirectory = str(tmp_path / "exports")

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("diagnosticsController", controller)

    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "support" / "TicketDialog.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować TicketDialog.qml"

    dialog = engine.rootObjects()[0]
    dialog.open()
    controller.description = "Przykładowy opis"

    assert controller.generateDiagnostics() is True
    assert controller.lastArchivePath.endswith(".zip")
    assert Path(controller.lastArchivePath).exists()

    engine.deleteLater()
    app.quit()
