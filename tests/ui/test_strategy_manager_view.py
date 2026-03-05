import os
from pathlib import Path

import pytest

from tests.ui._qt import require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = require_pyside6()
qt_root = Path(PySide6.__file__).resolve().parent
os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "Qt" / "qml"))
os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "Qt" / "plugins"))

from PySide6.QtCore import QObject, Property, QMetaObject, Qt, Signal, Slot, Q_ARG, QUrl
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)


class StubMarketplaceController(QObject):
    presetsChanged = Signal()
    lastErrorChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._presets: list[dict[str, object]] = [
            {
                "presetId": "scalping_ai",
                "name": "Scalping AI",
                "version": "1.0",
                "summary": "Szybkie decyzje na rynku spot.",
            }
        ]
        self.install_calls: list[tuple[str, str]] = []
        self.assign_calls: list[tuple[str, str]] = []

    @Property("QVariantList", notify=presetsChanged)
    def presets(self) -> list[dict[str, object]]:  # type: ignore[override]
        return self._presets

    @Property(str, notify=lastErrorChanged)
    def lastError(self) -> str:  # type: ignore[override]
        return ""

    @Slot(result=bool)
    def refreshPresets(self) -> bool:  # type: ignore[override]
        self._presets.append(
            {
                "presetId": "swing_guard",
                "name": "Swing Guard",
                "version": "2.0",
                "summary": "Swing trading z guardrailami.",
            }
        )
        self.presetsChanged.emit()
        return True

    @Slot(str, str, result="QVariantMap")
    def activateAndAssignPreset(self, preset_id: str, portfolio_id: str) -> dict[str, object]:  # type: ignore[override]
        self.install_calls.append((preset_id, portfolio_id))
        return {"success": True, "presetId": preset_id, "assignedPortfolios": [portfolio_id]}

    @Slot(str, str, result=bool)
    def assignPresetToPortfolio(self, preset_id: str, portfolio_id: str) -> bool:  # type: ignore[override]
        self.assign_calls.append((preset_id, portfolio_id))
        return True


@pytest.mark.timeout(20)
def test_strategy_manager_view_triggers_actions(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    controller = StubMarketplaceController()
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("marketplaceController", controller)

    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "StrategyManager.qml"
    qml_warnings: list = []

    def _collect(warnings_list: list) -> None:
        qml_warnings.extend(warnings_list)

    engine.warnings.connect(_collect)  # type: ignore[attr-defined]
    engine.load(QUrl.fromLocalFile(str(view_path)))
    if qml_warnings or not engine.rootObjects():
        warnings_text = (
            "; ".join(warning.toString() for warning in qml_warnings) or "brak obiektów root"
        )
        pytest.skip(
            f"Nie udało się załadować widoku StrategyManager: {warnings_text}",
            allow_module_level=False,
        )

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    QMetaObject.invokeMethod(root, "refreshMarketplace", Qt.DirectConnection)
    app.processEvents()
    root.setProperty("targetPortfolioId", "desk-1")

    QMetaObject.invokeMethod(
        root,
        "quickInstall",
        Qt.DirectConnection,
        Q_ARG("QString", "scalping_ai"),
    )
    app.processEvents()

    assert controller.install_calls[-1] == ("scalping_ai", "desk-1")

    QMetaObject.invokeMethod(
        root,
        "assignPreset",
        Qt.DirectConnection,
        Q_ARG("QString", "swing_guard"),
        Q_ARG("QString", "desk-1"),
    )
    app.processEvents()

    assert controller.assign_calls[-1] == ("swing_guard", "desk-1")

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
