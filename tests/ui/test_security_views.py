from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Iterator, Tuple

import pytest

from tests.utils.libgl import ensure_libgl_available

try:
    ensure_libgl_available()
except RuntimeError as exc:  # pragma: no cover - brak możliwości instalacji libGL
    pytest.skip(
        f"Pomijam testy UI: nie udało się zapewnić libGL w środowisku ({exc}).",
        allow_module_level=True,
    )

PySide6 = pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

qt_root = Path(PySide6.__file__).resolve().parent
os.environ.setdefault("QML2_IMPORT_PATH", str(qt_root / "Qt" / "qml"))
os.environ.setdefault("QT_PLUGIN_PATH", str(qt_root / "Qt" / "plugins"))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

try:
    from PySide6.QtCore import (
        Property,
        QMetaObject,
        QObject,
        QUrl,
        Signal,
        Slot,
    )
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtQml import QQmlApplicationEngine
except ImportError as exc:  # pragma: no cover - środowisko bez bibliotek systemowych Qt
    pytest.skip(
        f"Pomijam testy UI: środowisko nie udostępnia bibliotek Qt ({exc}).",
        allow_module_level=True,
    )

RootLoader = Callable[[str], Tuple[QQmlApplicationEngine, QObject]]


class _LicenseControllerStub(QObject):
    def __init__(self) -> None:
        super().__init__()
        self.setProperty("licenseActive", True)
        self.setProperty("licenseFingerprint", "stub-fingerprint")
        self.setProperty("licenseEdition", "enterprise")
        self.setProperty("licenseLicenseId", "LIC-001")
        self.setProperty("licenseMaintenanceUntil", "2099-12-31")
        self.setProperty("statusMessage", "")
        self.setProperty("statusIsError", False)
        self.setProperty("provisioningInProgress", False)
        self._expected = "stub-expected"
        self.setProperty("expectedFingerprint", self._expected)

    @Slot(str)
    def loadLicenseUrl(self, url: str) -> None:  # pragma: no cover - tylko do QML
        self.setProperty("lastLoadedUrl", url)

    @Slot(str)
    def saveExpectedFingerprint(self, value: str) -> None:  # pragma: no cover - tylko do QML
        self._expected = value
        self.setProperty("expectedFingerprint", value)

    @Slot(str)
    def overrideExpectedFingerprint(self, value: str) -> None:  # pragma: no cover - tylko do QML
        self._expected = value
        self.setProperty("expectedFingerprint", value)

    @Slot("QVariant")
    def autoProvision(self, payload) -> None:  # pragma: no cover - tylko do QML
        self.setProperty("lastAutoProvisionPayload", payload)


class _ActivationControllerStub(QObject):
    fingerprintChanged = Signal()
    licensesChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._fingerprint = {
            "fingerprint": "stub-fingerprint",
            "payload": {"fingerprint": "stub-fingerprint"},
        }
        self._licenses = [
            {
                "fingerprint": "stub-fingerprint",
                "mode": "online",
                "licenseId": "LIC-001",
                "issuedAt": "2024-01-01T00:00:00Z",
                "maintenance_until": "2099-12-31",
                "recorded_at": "2024-01-01T00:00:00Z",
            }
        ]

    @Property("QVariant", notify=fingerprintChanged)
    def fingerprint(self):  # type: ignore[override]
        return self._fingerprint

    @Property("QVariant", notify=licensesChanged)
    def licenses(self):  # type: ignore[override]
        return self._licenses

    @Slot()
    def refresh(self) -> None:  # pragma: no cover - tylko do QML
        return None

    @Slot(str)
    def exportFingerprint(self, destination: str) -> None:  # pragma: no cover - tylko do QML
        self.setProperty("lastExportDestination", destination)

    @Slot()
    def reloadRegistry(self) -> None:  # pragma: no cover - tylko do QML
        return None


class _SecurityControllerStub(QObject):
    licenseInfoChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._license_info = {
            "status": "aktywny",
            "fingerprint": "stub-fingerprint",
            "local_fingerprint": "stub-fingerprint",
            "edition": "enterprise",
            "license_id": "LIC-001",
            "maintenance_until": "2099-12-31",
            "modules": ["core"],
            "environments": ["prod"],
            "warnings": [],
            "errors": [],
        }

    @Property("QVariant", notify=licenseInfoChanged)
    def licenseInfo(self):  # type: ignore[override]
        return self._license_info


class _AppControllerStub(QObject):
    licenseRefreshScheduleChanged = Signal()
    securityCacheChanged = Signal()
    fingerprintRefreshScheduleChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._security_stub = _SecurityControllerStub()
        self._license_schedule = {
            "active": True,
            "intervalSeconds": 600,
            "lastRefreshAt": "2024-01-01T00:00:00Z",
            "nextRefreshDueAt": "2024-01-01T01:00:00Z",
            "nextRefreshInSeconds": 3600,
        }
        self._fingerprint_schedule = {
            "active": True,
            "intervalSeconds": 86400,
            "lastRequestAt": "2024-01-01T00:00:00Z",
            "lastCompletedAt": "2024-01-01T00:05:00Z",
            "nextRefreshDueAt": "2024-01-02T00:00:00Z",
            "nextRefreshInSeconds": 7200,
            "lastError": "",
        }
        self._security_cache = {
            "lastError": "",
            "fingerprintRefresh": self._fingerprint_schedule,
            "oemLicense": self._security_stub.licenseInfo,
        }

    @Property("QVariant", notify=licenseRefreshScheduleChanged)
    def licenseRefreshSchedule(self):  # type: ignore[override]
        return self._license_schedule

    @Property("QVariant", notify=fingerprintRefreshScheduleChanged)
    def fingerprintRefreshSchedule(self):  # type: ignore[override]
        return self._fingerprint_schedule

    @Property("QVariant", notify=securityCacheChanged)
    def securityCache(self):  # type: ignore[override]
        return self._security_cache

    @Slot(bool, result=bool)
    def setLicenseRefreshEnabled(self, enabled: bool) -> bool:  # pragma: no cover
        self._license_schedule["active"] = bool(enabled)
        self.licenseRefreshScheduleChanged.emit()
        return True

    @Slot(int)
    def setLicenseRefreshIntervalSeconds(self, interval: int) -> None:  # pragma: no cover
        self._license_schedule["intervalSeconds"] = int(interval)
        self.licenseRefreshScheduleChanged.emit()

    @Slot(result=bool)
    def triggerLicenseRefreshNow(self) -> bool:  # pragma: no cover
        self._license_schedule["lastRefreshAt"] = "2024-01-01T00:10:00Z"
        self._license_schedule["nextRefreshInSeconds"] = 0
        self.licenseRefreshScheduleChanged.emit()
        return True

    @Slot(bool, result=bool)
    def setFingerprintRefreshEnabled(self, enabled: bool) -> bool:  # pragma: no cover
        self._fingerprint_schedule["active"] = bool(enabled)
        self.fingerprintRefreshScheduleChanged.emit()
        return True

    @Slot(int)
    def setFingerprintRefreshIntervalSeconds(self, interval: int) -> None:  # pragma: no cover
        self._fingerprint_schedule["intervalSeconds"] = int(interval)
        self.fingerprintRefreshScheduleChanged.emit()

    @Slot(result=bool)
    def triggerFingerprintRefreshNow(self) -> bool:  # pragma: no cover
        self._fingerprint_schedule["lastRequestAt"] = "2024-01-01T00:10:00Z"
        self._fingerprint_schedule["nextRefreshInSeconds"] = 0
        self.fingerprintRefreshScheduleChanged.emit()
        self.securityCacheChanged.emit()
        return True


@pytest.fixture()
def qt_app() -> Iterator[QGuiApplication]:
    app = QGuiApplication.instance() or QGuiApplication([])
    yield app
    app.processEvents()


@pytest.fixture()
def load_security_view(qt_app: QGuiApplication, tmp_path: Path) -> Iterator[RootLoader]:
    os.environ["QT_QUICK_LOCAL_STORAGE_PATH"] = str(tmp_path)
    base_dir = Path(__file__).resolve().parents[2] / "ui" / "qml" / "components" / "security"

    def _loader(relative_name: str) -> Tuple[QQmlApplicationEngine, QObject]:
        engine = QQmlApplicationEngine()
        engine.setOfflineStoragePath(str(tmp_path))
        qml_path = base_dir / relative_name

        stubs = {
            "licenseController": _LicenseControllerStub(),
            "activationController": _ActivationControllerStub(),
            "appController": _AppControllerStub(),
            "securityController": _SecurityControllerStub(),
        }
        context = engine.rootContext()
        for name, obj in stubs.items():
            context.setContextProperty(name, obj)
        context.setContextProperty("schedule", stubs["appController"].fingerprintRefreshSchedule)
        context.setContextProperty("summary", stubs["securityController"].licenseInfo)
        engine._test_stubs = stubs  # type: ignore[attr-defined]

        qml_warnings: list = []

        def _collect(warnings_list: list) -> None:
            qml_warnings.extend(warnings_list)

        engine.warnings.connect(_collect)  # type: ignore[attr-defined]
        engine.load(QUrl.fromLocalFile(str(qml_path)))
        roots = engine.rootObjects()
        assert roots, f"Nie udało się załadować widoku {relative_name}"
        return engine, roots[0]

    try:
        yield _loader
    finally:
        if "QT_QUICK_LOCAL_STORAGE_PATH" in os.environ:
            del os.environ["QT_QUICK_LOCAL_STORAGE_PATH"]


def _cleanup_engine(engine: QQmlApplicationEngine) -> None:
    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()


def test_license_activation_view_exposes_audit_controls(
    load_security_view: RootLoader, qt_app: QGuiApplication
) -> None:
    engine, root = load_security_view("LicenseActivationView.qml")
    try:
        meta = root.metaObject()
        assert meta.indexOfProperty("auditTotal") != -1
        assert meta.indexOfMethod("refreshAudit()") != -1
        assert isinstance(root.property("scheduleAutoEnabled"), bool)
        assert meta.indexOfMethod("formatRemaining(QVariant)") != -1
    finally:
        _cleanup_engine(engine)


def test_hwid_management_view_exposes_schedule_api(
    load_security_view: RootLoader, qt_app: QGuiApplication
) -> None:
    engine, root = load_security_view("HwidManagementView.qml")
    try:
        meta = root.metaObject()
        assert meta.indexOfMethod("fingerprintSchedule()") != -1
        assert meta.indexOfMethod("syncScheduleFromController()") != -1
        assert isinstance(root.property("scheduleAutoEnabled"), bool)
    finally:
        _cleanup_engine(engine)


def test_license_history_view_initializes_model(
    load_security_view: RootLoader, qt_app: QGuiApplication
) -> None:
    engine, root = load_security_view("LicenseHistoryView.qml")
    try:
        meta = root.metaObject()
        assert meta.indexOfMethod("licenseSummary()") != -1
    finally:
        _cleanup_engine(engine)
