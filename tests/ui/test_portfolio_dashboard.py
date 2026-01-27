from __future__ import annotations

import contextlib
import gc
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import pytest

from bot_core.database.manager import DatabaseManager
from tests.ui._qt import qml_value_to_python, require_pyside6

logger = logging.getLogger(__name__)

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

require_pyside6()

import PySide6
from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, Property,
                            Qt, QUrl, Slot, QMetaObject)
from PySide6.QtQml import QQmlApplicationEngine, QQmlComponent

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # brak bibliotek systemowych (np. libGL)
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)


class StubRiskHistoryModel(QObject):
    def __init__(self, entries: list[dict[str, object]]) -> None:
        super().__init__()
        self._entries = entries

    @Property(int, constant=False)
    def count(self) -> int:  # type: ignore[override]
        return len(self._entries)

    @Slot(int, result="QVariantMap")
    def get(self, index: int) -> dict[str, object]:
        if 0 <= index < len(self._entries):
            return self._entries[index]
        return {}


class StubRiskModel(QObject):
    def __init__(self, exposures: list[dict[str, object]]) -> None:
        super().__init__()
        self._exposures = exposures

    @Property(int, constant=False)
    def count(self) -> int:  # type: ignore[override]
        return len(self._exposures)

    @Slot(int, result="QVariantMap")
    def get(self, index: int) -> dict[str, object]:
        if 0 <= index < len(self._exposures):
            return self._exposures[index]
        return {}


class StubAlertsModel(QAbstractListModel):
    TITLE_ROLE = Qt.UserRole + 1
    DESCRIPTION_ROLE = Qt.UserRole + 2
    SEVERITY_ROLE = Qt.UserRole + 3
    TIMESTAMP_ROLE = Qt.UserRole + 4

    def __init__(self, alerts: list[dict[str, object]]) -> None:
        super().__init__()
        self._alerts = alerts

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._alerts)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._alerts)):
            return None
        alert = self._alerts[index.row()]
        if role == self.TITLE_ROLE:
            return alert.get("title")
        if role == self.DESCRIPTION_ROLE:
            return alert.get("description")
        if role == self.SEVERITY_ROLE:
            return alert.get("severity", 0)
        if role == self.TIMESTAMP_ROLE:
            return alert.get("timestamp")
        return None

    def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
        return {
            self.TITLE_ROLE: b"title",
            self.DESCRIPTION_ROLE: b"description",
            self.SEVERITY_ROLE: b"severity",
            self.TIMESTAMP_ROLE: b"timestamp",
        }

    @Property(int, constant=False)
    def unacknowledgedCount(self) -> int:  # type: ignore[override]
        return len(self._alerts)

    @Slot()
    def acknowledgeAll(self) -> None:
        if not self._alerts:
            return
        self.beginResetModel()
        self._alerts = []
        self.endResetModel()


@pytest.mark.timeout(30)
def test_portfolio_dashboard_builds_exposures_and_history(tmp_path: Path) -> None:
    app = QApplication.instance() or QApplication([])

    risk_history = StubRiskHistoryModel(
        [
            {"timestamp": datetime(2024, 1, 1, 12, 0, 0), "portfolioValue": 100000.0},
            {"timestamp": datetime(2024, 1, 1, 13, 0, 0), "portfolioValue": 103500.0},
        ]
    )
    risk_model = StubRiskModel(
        [
            {"code": "exchange:BINANCE", "currentValue": 12000.0, "maxValue": 25000.0, "thresholdValue": 18000.0},
            {"code": "strategy:theta_income_balanced", "currentValue": 8000.0, "maxValue": 15000.0, "thresholdValue": 0.0},
        ]
    )
    alerts_model = StubAlertsModel(
        [
            {
                "title": "Przekroczony limit",
                "description": "Ekspozycja na BINANCE przekroczyła 18k",
                "severity": 2,
                "timestamp": datetime(2024, 1, 1, 13, 5, 0),
            },
        ]
    )

    engine = QQmlApplicationEngine()
    context = engine.rootContext()
    context.setContextProperty("riskHistoryModel", risk_history)
    context.setContextProperty("riskModel", risk_model)
    context.setContextProperty("alertsModel", alerts_model)
    context.setContextProperty("uiTestMode", True)

    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "PortfolioDashboard.qml"
    qml_warnings = []

    def collect_warnings(warnings: list[object]) -> None:
        qml_warnings.extend(warnings)

    def wait_for_aiosqlite_threads(timeout: float = 5.0, poll_interval: float = 0.1) -> None:
        deadline = time.monotonic() + timeout
        while True:
            threads = [t for t in threading.enumerate() if t.is_alive() and "aiosqlite" in t.name]
            if not threads:
                return
            if time.monotonic() >= deadline:
                thread_details = [
                    f"{t.name} (ident={t.ident}, daemon={t.daemon})" for t in threads
                ]
                logger.error("Lingering aiosqlite threads: %s", "; ".join(thread_details))
                pytest.fail(
                    "Pozostały aktywne wątki aiosqlite po sprzątaniu: "
                    + "; ".join(thread_details)
                )
            time.sleep(poll_interval)

    engine.warnings.connect(collect_warnings)
    roots_snapshot: list[QObject] = []
    try:
        engine.load(QUrl.fromLocalFile(str(view_path)))
        if not engine.rootObjects():
            warning_lines = []
            for warning in qml_warnings:
                try:
                    line = warning.line()
                    column = warning.column()
                    location = f" (line {line}, column {column})" if line or column else ""
                except Exception:
                    location = ""
                warning_lines.append(f"- {warning}{location}")
            try:
                engine_warnings = engine.warnings()
            except Exception:
                engine_warnings = []
            if engine_warnings:
                warning_lines.extend(f"- {warning}" for warning in engine_warnings)
            warnings_message = (
                "\n".join(warning_lines) if warning_lines else "Brak zarejestrowanych ostrzeżeń QML."
            )

            if sys.platform == "win32":
                component_error_details = (
                    "(skipped on win32: QQmlComponent diagnostics can trigger access violation)"
                )
                component_error_string = (
                    "(skipped on win32: QQmlComponent diagnostics can trigger access violation)"
                )
            else:
                component_error_details = "(unavailable)"
                component_error_string = "(unavailable)"
                try:
                    component = QQmlComponent(engine)
                    component.loadUrl(QUrl.fromLocalFile(str(view_path)))
                    component_errors = []
                    try:
                        is_error = False
                        is_error_method = getattr(component, "isError", None)
                        if callable(is_error_method):
                            try:
                                is_error = bool(is_error_method())
                            except Exception:
                                is_error = False
                        status_method = getattr(component, "status", None)
                        if not is_error and callable(status_method):
                            try:
                                is_error = status_method() == getattr(QQmlComponent, "Error", None)
                            except Exception:
                                is_error = False
                        if is_error:
                            errors_method = getattr(component, "errors", None)
                            if callable(errors_method):
                                try:
                                    component_errors = errors_method()
                                except Exception:
                                    component_errors = []
                    except Exception:
                        component_errors = []
                    component_error_details = "\n".join(
                        getattr(error, "toString", lambda: str(error))() for error in component_errors
                    ) or "(none)"
                    error_string_method = getattr(component, "errorString", None)
                    if callable(error_string_method):
                        try:
                            component_error_string = error_string_method() or "(none)"
                        except Exception:
                            component_error_string = "(errorString failed)"
                    else:
                        component_error_string = "(unavailable)"
                except Exception as exc:
                    component_error_details = f"(QQmlComponent diagnostics failed: {exc})"
                    component_error_string = "(QQmlComponent diagnostics failed)"

            exists_message = f"Path exists: {os.path.exists(view_path)}"
            env_context_message = (
                "Env context: QT_QPA_PLATFORM="
                f"{os.environ.get('QT_QPA_PLATFORM', '(unset)')}, "
                "QT_PLUGIN_PATH="
                f"{os.environ.get('QT_PLUGIN_PATH', '(unset)')}, "
                "QML2_IMPORT_PATH="
                f"{os.environ.get('QML2_IMPORT_PATH', '(unset)')}, "
                "QML_IMPORT_TRACE="
                f"{os.environ.get('QML_IMPORT_TRACE', '(unset)')}, "
                "QT_DEBUG_PLUGINS="
                f"{os.environ.get('QT_DEBUG_PLUGINS', '(unset)')}"
            )
            python_message = f"Python: {sys.executable}"
            pyside_version = getattr(PySide6, "__version__", None) or "(unknown)"
            pyside_message = f"PySide6: {pyside_version}"
            pytest.fail(
                "\n".join(
                    [
                        "qml_load_failed/portfolio_dashboard: Nie udało się załadować dashboardu portfela.",
                        f"view_path: {view_path}",
                        exists_message,
                        env_context_message,
                        python_message,
                        pyside_message,
                        "QML warnings:",
                        warnings_message,
                        "=== Component errors ===",
                        component_error_details,
                        "=== Component errorString ===",
                        component_error_string,
                    ]
                )
            )

        root = engine.rootObjects()[0]
        assert isinstance(root, QObject)

        # Wymuś przebudowanie danych
        for method in ("rebuildHistoryPoints", "rebuildExposureTables"):
            QMetaObject.invokeMethod(root, method, Qt.DirectConnection)
        app.processEvents()

        history_points = qml_value_to_python(root.property("historyPoints"))
        assert isinstance(history_points, list)
        assert len(history_points) == 2
        assert history_points[0]["value"] == 100000.0

        exchange_items = qml_value_to_python(root.property("exchangeExposureItems"))
        strategy_items = qml_value_to_python(root.property("strategyExposureItems"))
        assert isinstance(exchange_items, list) and isinstance(strategy_items, list)
        assert exchange_items[0]["name"].upper() == "BINANCE"
        assert strategy_items[0]["name"].startswith("theta_income")

        alerts_view = root.findChild(QObject, "alertsListView")
        assert alerts_view is not None
        assert alerts_view.property("count") == 1

        # Potwierdzenie alertów powinno wyczyścić model
        alerts_model.acknowledgeAll()
        app.processEvents()
        assert alerts_model.rowCount() == 0
    finally:
        # Snapshot przed teardownem Qt
        try:
            roots_snapshot = list(engine.rootObjects())
        except Exception:
            roots_snapshot = []

        # Odłącz callback, żeby Qt nie wołał Pythona podczas niszczenia engine
        with contextlib.suppress(Exception):
            engine.warnings.disconnect(collect_warnings)

        for obj in roots_snapshot:
            obj.deleteLater()
        engine.deleteLater()
        for _ in range(3):
            app.processEvents()
        DatabaseManager.close_all_active(blocking=True, timeout=5.0)
        wait_for_aiosqlite_threads(timeout=5.0)
        gc.collect()
