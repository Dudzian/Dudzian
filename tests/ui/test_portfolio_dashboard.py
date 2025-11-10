from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, Property,
                            Qt, QUrl, Slot, QMetaObject)
from PySide6.QtQml import QQmlApplicationEngine

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

    view_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "PortfolioDashboard.qml"
    engine.load(QUrl.fromLocalFile(str(view_path)))
    assert engine.rootObjects(), "Nie udało się załadować dashboardu portfela"

    root = engine.rootObjects()[0]
    assert isinstance(root, QObject)

    # Wymuś przebudowanie danych
    for method in ("rebuildHistoryPoints", "rebuildExposureTables"):
        QMetaObject.invokeMethod(root, method, Qt.DirectConnection)
    app.processEvents()

    history_points = root.property("historyPoints")
    assert isinstance(history_points, list)
    assert len(history_points) == 2
    assert history_points[0]["value"] == 100000.0

    exchange_items = root.property("exchangeExposureItems")
    strategy_items = root.property("strategyExposureItems")
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

    for obj in engine.rootObjects():
        obj.deleteLater()
    engine.deleteLater()
    app.processEvents()
