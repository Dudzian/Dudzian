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

    app.processEvents()

    saved_presets = root.property("savedPresets")
    assert isinstance(saved_presets, list)
    assert len(saved_presets) == 2
    assert runtime_service.list_calls == 1

    preset_preview = root.property("presetPreview")
    assert preset_preview is not None
    assert preset_preview["ok"] is True

    clone_dialog = root.findChild(QObject, "cloneDialog")
    assert clone_dialog is not None

    assert QMetaObject.invokeMethod(root, "requestClonePreset", Qt.DirectConnection) is True
    app.processEvents()
    assert clone_dialog.property("visible") is True

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
    cloud_switch.setProperty("checked", True)
    app.processEvents()
    config_data = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    assert config_data["cloud"]["enabled_signed"] is True

    alpha_selector = root.findChild(QObject, "bundleSelector_alpha-momentum")
    beta_selector = root.findChild(QObject, "bundleSelector_beta-mean")
    assert alpha_selector is not None and beta_selector is not None
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

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
