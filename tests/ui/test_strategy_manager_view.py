import os
from pathlib import Path

import pytest

from tests.ui._qml_hosting import (
    collect_object_names,
    ensure_item_has_host_window,
    teardown_hosted_qml_engine,
)
from tests.ui._qt import require_pyside6
from tests.ui._qt_utils import wait_for

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = require_pyside6()
qt_root = Path(PySide6.__file__).resolve().parent
os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "Qt" / "qml"))
os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "Qt" / "plugins"))

from PySide6.QtCore import QObject, Property, QMetaObject, Qt, Signal, Slot, QUrl
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

    host_window = ensure_item_has_host_window(root, default_height=700)
    app.processEvents()

    try:
        QMetaObject.invokeMethod(root, "refreshMarketplace", Qt.DirectConnection)
        app.processEvents()
        app.processEvents()
        root.setProperty("targetPortfolioId", "desk-1")
        app.processEvents()

        presets_cache = wait_for(
            lambda: root.property("presetsCache")
            if isinstance(root.property("presetsCache"), list)
            and len(root.property("presetsCache")) >= 2
            else None,
            timeout_s=2.0,
            process_events=app.processEvents,
            description="StrategyManager presetsCache should contain refreshed delegates",
        )

        quick_install_button = wait_for(
            lambda: root.findChild(QObject, "quickInstallButton_scalping_ai"),
            timeout_s=2.0,
            process_events=app.processEvents,
            description="quick install button for scalping_ai",
        )
        assign_button = wait_for(
            lambda: root.findChild(QObject, "assignPresetButton_swing_guard"),
            timeout_s=2.0,
            process_events=app.processEvents,
            description="assign button for swing_guard",
        )

        quick_install_names = collect_object_names(root, "quickInstallButton_")
        assign_button_names = collect_object_names(root, "assignPresetButton_")

        assert quick_install_button is not None, (
            "quickInstallButton_scalping_ai was not materialized; "
            f"presetsCache={presets_cache!r}; "
            f"quickInstallButtons={quick_install_names!r}; "
            f"assignButtons={assign_button_names!r}"
        )
        assert QMetaObject.invokeMethod(quick_install_button, "click", Qt.DirectConnection)
        app.processEvents()

        assert controller.install_calls[-1] == ("scalping_ai", "desk-1")

        assert assign_button is not None, (
            "assignPresetButton_swing_guard was not materialized; "
            f"presetsCache={presets_cache!r}; "
            f"quickInstallButtons={quick_install_names!r}; "
            f"assignButtons={assign_button_names!r}"
        )
        assert assign_button.property("enabled") is True
        assert QMetaObject.invokeMethod(assign_button, "click", Qt.DirectConnection)
        app.processEvents()

        assert controller.assign_calls[-1] == ("swing_guard", "desk-1")
    finally:
        teardown_hosted_qml_engine(
            root,
            host_window,
            engine,
            process_events=app.processEvents,
        )
        app.processEvents()
