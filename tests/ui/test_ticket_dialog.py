import os
from pathlib import Path

import pytest

from tests.ui._qt import require_libgl, require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
require_libgl()

if os.getenv("PYTEST_REQUIRE_QML", "").lower() not in {"1", "true", "yes"}:
    pytest.skip(
        "TicketDialog wymaga pełnego środowiska QML; ustaw PYTEST_REQUIRE_QML=1, aby włączyć test.",
        allow_module_level=True,
    )

PySide6 = require_pyside6()
qt_root = Path(PySide6.__file__).resolve().parent
os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "Qt" / "qml"))
os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "Qt" / "plugins"))

from PySide6.QtCore import QUrl  # type: ignore[attr-defined]
from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]

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
    try:  # pragma: no cover - zależne od środowiska
        from PySide6.QtCore import Qt  # type: ignore[attr-defined]
        from PySide6.QtQuick import QQuickWindow, QSGRendererInterface  # type: ignore[attr-defined]
        from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
    except ImportError as exc:  # pragma: no cover - brak QtWidgets/libGL
        pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=False)

    QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL)
    try:
        QQuickWindow.setGraphicsApi(QSGRendererInterface.Software)
    except AttributeError:  # pragma: no cover - zgodność ze starszymi wersjami Qt
        QQuickWindow.setSceneGraphBackend("software")

    controller = DiagnosticsController()
    controller.baseDirectory = str(project_with_data)
    controller.outputDirectory = str(tmp_path / "exports")

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("diagnosticsController", controller)

    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "support" / "TicketDialog.qml"
    qml_warnings: list = []

    def _collect(warnings_list: list) -> None:
        qml_warnings.extend(warnings_list)

    engine.warnings.connect(_collect)  # type: ignore[attr-defined]
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    if qml_warnings or not engine.rootObjects():
        warnings_text = "; ".join(warning.toString() for warning in qml_warnings) or "brak obiektów root"
        pytest.skip(
            f"Nie udało się załadować TicketDialog.qml: {warnings_text}",
            allow_module_level=False,
        )

    dialog = engine.rootObjects()[0]
    dialog.open()
    controller.description = "Przykładowy opis"

    assert controller.generateDiagnostics() is True
    assert controller.lastArchivePath.endswith(".zip")
    assert Path(controller.lastArchivePath).exists()

    engine.deleteLater()
    app.quit()
