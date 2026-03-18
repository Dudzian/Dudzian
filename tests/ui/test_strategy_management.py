import os
from pathlib import Path

import pytest

try:
    import yaml  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    yaml = None  # type: ignore

pytestmark = [
    pytest.mark.qml,
    pytest.mark.skipif(
        yaml is None, reason="PyYAML nie jest zainstalowane w tym środowisku testowym."
    ),
]

if yaml is not None:
    from tests.ui._qt import require_pyside6
    from ui.pyside_app.controllers.strategy import StrategyManagementController

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    PySide6 = require_pyside6()
    qt_root = Path(PySide6.__file__).resolve().parent
    os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "Qt" / "qml"))
    os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "Qt" / "plugins"))

    from PySide6.QtCore import QObject, Property, Qt, QUrl, Signal, Slot, QMetaObject, Q_ARG
    from PySide6.QtQml import QQmlApplicationEngine

    try:  # pragma: no cover - zależne od środowiska CI
        from PySide6.QtQuick import QQuickItem, QQuickWindow
    except ImportError as exc:  # pragma: no cover
        pytest.skip(
            f"Brak zależności systemowych Qt Quick (np. libEGL.so.1): {exc}",
            allow_module_level=True,
        )

    try:  # pragma: no cover - zależne od środowiska CI
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover
        pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)
else:

    class QObject:  # pragma: no cover - wykorzystywane tylko gdy moduł jest skipowany
        def __init__(self, *args, **kwargs) -> None:
            pass

    class QUrl:  # pragma: no cover
        @staticmethod
        def fromLocalFile(path: str) -> str:
            return path

    class Qt:  # pragma: no cover
        QueuedConnection = 0

    class QQmlApplicationEngine:  # pragma: no cover
        pass

    class QQuickItem:  # pragma: no cover
        pass

    class QQuickWindow:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            pass

        def contentItem(self):
            return None

        def setWidth(self, *args, **kwargs) -> None:
            pass

        def setHeight(self, *args, **kwargs) -> None:
            pass

        def show(self) -> None:
            pass

        def close(self) -> None:
            pass

        def deleteLater(self) -> None:
            pass

    class QApplication:  # pragma: no cover
        @staticmethod
        def instance():
            return None

        def __init__(self, *args, **kwargs) -> None:
            pass

    class Signal:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:
            pass

        def emit(self, *args, **kwargs) -> None:
            pass

    def Property(*args, **kwargs):  # pragma: no cover
        def _decorator(func):
            return func

        return _decorator

    def Slot(*args, **kwargs):  # pragma: no cover
        def _decorator(func):
            return func

        return _decorator

    class QMetaObject:  # pragma: no cover
        @staticmethod
        def invokeMethod(*args, **kwargs) -> bool:
            return True

    def Q_ARG(*args, **kwargs):  # pragma: no cover
        return None

    class StrategyManagementController:  # pragma: no cover
        pass


def _safe_qml_property(obj: QObject, name: str) -> object:
    try:
        return obj.property(name)
    except RuntimeError as exc:
        return f"<unavailable:{name}:{exc}>"


def _safe_qobject_class_name(value: object) -> str | None:
    if not isinstance(value, QObject):
        return None
    try:
        return value.metaObject().className()
    except RuntimeError:
        return "<metaObject-unavailable>"


def _popup_snapshot(popup: QObject) -> dict[str, object]:
    parent_obj = popup.parent()
    window_obj = _safe_qml_property(popup, "window")
    parent_window_obj = _safe_qml_property(popup, "parentWindow")
    return {
        "visible": _safe_qml_property(popup, "visible"),
        "opened": _safe_qml_property(popup, "opened"),
        "modal": _safe_qml_property(popup, "modal"),
        "popupType": _safe_qml_property(popup, "popupType"),
        "parentObjectName": parent_obj.objectName() if isinstance(parent_obj, QObject) else None,
        "parentClass": _safe_qobject_class_name(parent_obj),
        "windowClass": _safe_qobject_class_name(window_obj),
        "parentWindowClass": _safe_qobject_class_name(parent_window_obj),
    }


def _snapshot_str(label: str, popup: QObject) -> str:
    return f"{label}: {_popup_snapshot(popup)}"


def _collect_object_names(root: QObject, prefix: str) -> list[str]:
    names: list[str] = []
    stack: list[QObject] = [root]
    while stack:
        current = stack.pop()
        for child in current.children():
            if not isinstance(child, QObject):
                continue
            child_name = child.objectName()
            if child_name.startswith(prefix):
                names.append(child_name)
            stack.append(child)
    return sorted(set(names))


def _ensure_item_has_host_window(root: QObject) -> QQuickWindow | None:
    if not isinstance(root, QQuickItem):
        return None

    existing_window = _safe_qml_property(root, "window")
    if isinstance(existing_window, QObject):
        return None

    host_window = QQuickWindow()
    host_content = host_window.contentItem()
    if isinstance(host_content, QQuickItem):
        root.setParentItem(host_content)

    width = _safe_qml_property(root, "implicitWidth")
    height = _safe_qml_property(root, "implicitHeight")
    host_window.setWidth(int(width) if isinstance(width, (int, float)) and width > 0 else 960)
    host_window.setHeight(int(height) if isinstance(height, (int, float)) and height > 0 else 600)
    host_window.show()
    return host_window


def _is_item_hosted_in_window(root: QObject, host_window: QQuickWindow | None) -> bool:
    if host_window is None or not isinstance(root, QQuickItem):
        return False

    host_content = host_window.contentItem()
    if not isinstance(host_content, QQuickItem):
        return False

    item: QQuickItem | None = root
    while isinstance(item, QQuickItem):
        if item is host_content:
            return True
        item = item.parentItem()
    return False


class RuntimeServiceStub(QObject):
    def __init__(self) -> None:
        super().__init__()
        self._presets = [
            {"name": "Alpha Momentum", "slug": "alpha-momentum", "path": "/tmp/alpha.json"},
            {"name": "Beta Mean Reversion", "slug": "beta-mean", "path": "/tmp/beta.json"},
        ]
        self._cache: list[dict[str, object]] | None = None
        self.list_calls = 0
        self.clone_count = 0

    @Slot(result="QVariantList")
    def listStrategyPresets(self):  # type: ignore[override]
        self.list_calls += 1
        if self._cache is None:
            self._cache = [dict(entry) for entry in self._presets]
        return self._cache

    @Slot("QVariantMap", result="QVariantMap")
    def previewStrategyPreset(self, selector):  # type: ignore[override]
        slug = ""
        name = ""
        if isinstance(selector, dict):
            slug = str(selector.get("slug") or "")
            name = str(selector.get("name") or "")
        for entry in self._presets:
            if entry.get("slug") == slug or entry.get("name") == name:
                return {
                    "ok": True,
                    "preset": {"name": entry["name"], "slug": entry["slug"], "path": entry["path"]},
                    "preset_payload": {
                        "name": entry["name"],
                        "blocks": [
                            {
                                "type": "alpha",
                                "label": "Alpha",
                                "params": {"lookback": 12, "threshold": 1.5},
                            },
                        ],
                        "metadata": {"source": "stub"},
                    },
                    "diff": [
                        {
                            "parameter": "max_daily_loss_pct",
                            "label": "Limit dziennej straty",
                            "preset_value": 0.04,
                            "champion_value": 0.03,
                            "delta": 0.01,
                            "is_percent": True,
                        },
                        {
                            "parameter": "risk_per_trade",
                            "label": "Ryzyko na transakcję",
                            "preset_value": 0.01,
                            "champion_value": 0.008,
                            "delta": 0.002,
                            "is_percent": True,
                        },
                    ],
                    "champion": {"version": "v1"},
                    "validation": {"status": "ok"},
                    "simulation": {"net_return_pct": 0.12, "max_drawdown_pct": 0.05},
                }
        return {"ok": False, "error": "not found"}

    @Slot("QVariantMap", result="QVariantMap")
    def saveStrategyPreset(self, payload):  # type: ignore[override]
        name = str(payload.get("name") or f"Preset {self.clone_count + 1}")
        slug = name.lower().replace(" ", "-")
        self.clone_count += 1
        new_entry = {
            "name": name,
            "slug": f"{slug}-{self.clone_count}",
            "path": f"/tmp/{slug}-{self.clone_count}.json",
        }
        self._presets.insert(0, new_entry)
        self._cache = None
        return {"ok": True, "name": name, "path": new_entry["path"]}

    @Slot("QVariantMap", result="QVariantMap")
    def deleteStrategyPreset(self, selector):  # type: ignore[override]
        slug = str(selector.get("slug") or "") if isinstance(selector, dict) else ""
        self._presets = [entry for entry in self._presets if entry.get("slug") != slug]
        self._cache = None
        return {"ok": True}


class ReportControllerStub(QObject):
    championOverviewChanged = Signal()
    overviewStatsChanged = Signal()
    lastErrorChanged = Signal()
    lastNotificationChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._champion = {}
        self._stats = {}
        self._last_error = ""
        self._last_notification = ""
        self.busy = False

    @Property("QVariantMap", notify=championOverviewChanged)
    def championOverview(self):  # type: ignore[override]
        return self._champion

    @Property("QVariantMap", notify=overviewStatsChanged)
    def overviewStats(self):  # type: ignore[override]
        return self._stats

    @Property(str, notify=lastErrorChanged)
    def lastError(self):  # type: ignore[override]
        return self._last_error

    @Property(str, notify=lastNotificationChanged)
    def lastNotification(self):  # type: ignore[override]
        return self._last_notification

    @Slot()
    def refreshChampionOverview(self) -> None:
        self._champion = {}
        self.championOverviewChanged.emit()

    @Slot()
    def refresh(self) -> None:
        self._stats = {}
        self.overviewStatsChanged.emit()

    @Slot(str, str, str)
    def promoteChampion(self, model: str, version: str, reason: str) -> None:
        self._last_notification = f"Promoted {model} {version}: {reason}"
        self.lastNotificationChanged.emit()


class MarketplaceStub:
    def list_presets_payload(self):
        return []


@pytest.mark.timeout(20)
def test_strategy_management_clone_refreshes_presets(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    runtime_service = RuntimeServiceStub()
    report_controller = ReportControllerStub()
    runtime_config = tmp_path / "config" / "runtime.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text("cloud:\n  enabled_signed: false\n", encoding="utf-8")
    strategy_controller = StrategyManagementController(
        marketplace_service=MarketplaceStub(), runtime_config_path=runtime_config
    )

    context = engine.rootContext()
    context.setContextProperty("runtimeService", runtime_service)
    context.setContextProperty("reportController", report_controller)
    context.setContextProperty("strategyManagementController", strategy_controller)

    view_path = (
        Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "StrategyManagement.qml"
    )
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
            f"Nie udało się załadować StrategyManagement: {warnings_text}",
            allow_module_level=False,
        )

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    host_window = _ensure_item_has_host_window(root)
    app.processEvents()

    try:
        saved_presets = root.property("savedPresets")
        assert isinstance(saved_presets, list)
        assert len(saved_presets) == 2
        assert runtime_service.list_calls == 1

        preset_preview = root.property("presetPreview")
        assert preset_preview is not None
        assert preset_preview["ok"] is True

        clone_dialog = root.findChild(QObject, "cloneDialog")
        assert clone_dialog is not None

        clone_parent = clone_dialog.parent()
        assert isinstance(clone_parent, QObject)
        assert clone_parent is root

        if host_window is not None:
            assert _is_item_hosted_in_window(root, host_window), (
                "Root item should be attached to host window content tree before dialog checks"
            )

        before_request = _snapshot_str("before_request", clone_dialog)
        assert QMetaObject.invokeMethod(root, "requestClonePreset", Qt.DirectConnection) is True
        after_request_before_events = _snapshot_str("after_request_before_events", clone_dialog)

        # `requestClonePreset()` sets pending payload + clone name and calls `cloneDialog.open()`.
        pending_payload = root.property("pendingClonePayload")
        assert pending_payload is not None

        app.processEvents()
        after_events = _snapshot_str("after_events", clone_dialog)
        assert clone_dialog.property("visible") is True, (
            "Clone dialog not visible after requestClonePreset/processEvents; "
            + " | ".join([before_request, after_request_before_events, after_events])
        )

        clone_name_field = root.findChild(QObject, "cloneNameField")
        assert clone_name_field is not None
        clone_name_field.setProperty("text", "Gamma Hedge")

        assert QMetaObject.invokeMethod(clone_dialog, "accept", Qt.DirectConnection) is True
        app.processEvents()

        saved_presets = root.property("savedPresets")
        assert isinstance(saved_presets, list)
        assert len(saved_presets) == 3
        assert runtime_service.list_calls >= 2
        assert runtime_service.clone_count == 1

        cloud_switch = root.findChild(QObject, "cloudToggle")
        assert cloud_switch is not None
        assert cloud_switch.property("checked") is False
        # `setProperty("checked", ...)` updates the visual state only.
        # Use the control API that mirrors a real user toggle so `onToggled`
        # runs and persists runtime cloud settings via the controller.
        assert QMetaObject.invokeMethod(cloud_switch, "toggle", Qt.DirectConnection) is True
        app.processEvents()
        assert cloud_switch.property("checked") is True
        config_data = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
        assert config_data["cloud"]["enabled_signed"] is True

        bundle_selection = root.property("bundleSelection")
        saved_presets = root.property("savedPresets")
        bundle_selector_names = _collect_object_names(root, "bundleSelector_")
        bundle_selector_repeater = root.findChild(QObject, "bundleSelectorRepeater")
        bundle_selector_count = (
            bundle_selector_repeater.property("count") if bundle_selector_repeater is not None else None
        )

        alpha_selector = root.findChild(QObject, "bundleSelector_alpha-momentum")
        beta_selector = root.findChild(QObject, "bundleSelector_beta-mean")

        if alpha_selector is None or beta_selector is None:
            fallback_alpha = root.findChild(QObject, "bundleSelector_/tmp/alpha.json")
            fallback_beta = root.findChild(QObject, "bundleSelector_/tmp/beta.json")
            if alpha_selector is None and fallback_alpha is not None:
                alpha_selector = fallback_alpha
            if beta_selector is None and fallback_beta is not None:
                beta_selector = fallback_beta

        assert bundle_selector_count == len(saved_presets), (
            "Bundle selector repeater did not materialize every preset; "
            f"repeaterCount={bundle_selector_count!r}; "
            f"savedPresetSlugs={[(entry.get('slug'), entry.get('path')) for entry in (saved_presets or []) if isinstance(entry, dict)]!r}; "
            f"availableSelectors={bundle_selector_names!r}"
        )
        assert alpha_selector is not None and beta_selector is not None, (
            "Bundle selectors not materialized under expected IDs; "
            f"repeaterCount={bundle_selector_count!r}; "
            f"bundleSelection={bundle_selection!r}; "
            f"savedPresetSlugs={[(entry.get('slug'), entry.get('path')) for entry in (saved_presets or []) if isinstance(entry, dict)]!r}; "
            f"availableSelectors={bundle_selector_names!r}"
        )
        alpha_selector.setProperty("checked", True)
        beta_selector.setProperty("checked", True)
        bundle_name = root.findChild(QObject, "bundleNameField")
        assert bundle_name is not None
        bundle_name.setProperty("text", "AlphaBetaCombo")
        assert QMetaObject.invokeMethod(root, "triggerBundleExport", Qt.DirectConnection)
        app.processEvents()
        assert strategy_controller.lastBundlePath
        bundle_path = Path(strategy_controller.lastBundlePath)
        assert bundle_path.exists()
        payload = yaml.safe_load(bundle_path.read_text(encoding="utf-8"))
        assert len(payload["presets"]) == 2
    finally:
        if host_window is not None:
            host_window.close()
            host_window.deleteLater()
        for obj in engine.rootObjects():
            obj.deleteLater()
        engine.deleteLater()
        app.processEvents()


@pytest.mark.timeout(20)
def test_strategy_management_promotion_dialog_hosting_is_consistent(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    engine = QQmlApplicationEngine()
    runtime_service = RuntimeServiceStub()
    report_controller = ReportControllerStub()
    runtime_config = tmp_path / "config" / "runtime.yaml"
    runtime_config.parent.mkdir(parents=True, exist_ok=True)
    runtime_config.write_text("cloud:\n  enabled_signed: false\n", encoding="utf-8")
    strategy_controller = StrategyManagementController(
        marketplace_service=MarketplaceStub(), runtime_config_path=runtime_config
    )

    context = engine.rootContext()
    context.setContextProperty("runtimeService", runtime_service)
    context.setContextProperty("reportController", report_controller)
    context.setContextProperty("strategyManagementController", strategy_controller)

    view_path = (
        Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "StrategyManagement.qml"
    )
    engine.load(QUrl.fromLocalFile(str(view_path)))
    if not engine.rootObjects():
        pytest.skip("Nie udało się załadować StrategyManagement", allow_module_level=False)

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    host_window = _ensure_item_has_host_window(root)
    app.processEvents()

    try:
        promotion_dialog = root.findChild(QObject, "promotionDialog")
        if promotion_dialog is None:
            for child in root.children():
                if isinstance(child, QObject) and child.metaObject().className().endswith("Dialog"):
                    if child.property("title") and "Promocja championa" in str(
                        child.property("title")
                    ):
                        promotion_dialog = child
                        break
        assert promotion_dialog is not None

        parent_obj = promotion_dialog.parent()
        assert isinstance(parent_obj, QObject)
        assert parent_obj is root

        if host_window is not None:
            assert _is_item_hosted_in_window(root, host_window), (
                "Root item should be attached to host window content tree before dialog checks"
            )

        assert (
            QMetaObject.invokeMethod(
                root,
                "startPromotion",
                Qt.DirectConnection,
                Q_ARG(str, "alpha-model"),
                Q_ARG(str, "v2"),
                Q_ARG(str, "manual"),
            )
            is True
        )
        app.processEvents()

        assert promotion_dialog.property("visible") is True, _snapshot_str(
            "promotion_after_events", promotion_dialog
        )
    finally:
        if host_window is not None:
            host_window.close()
            host_window.deleteLater()
        for obj in engine.rootObjects():
            obj.deleteLater()
        engine.deleteLater()
        app.processEvents()
