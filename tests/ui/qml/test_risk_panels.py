from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6", reason="Wymagany PySide6 do testów UI")

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, Property,
                            Qt, QUrl, Signal, Slot)
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.simulation import build_profile


class EngineLimitsModel(QAbstractListModel):
    KEY_ROLE = Qt.UserRole + 1
    LABEL_ROLE = Qt.UserRole + 2
    VALUE_ROLE = Qt.UserRole + 3
    MIN_ROLE = Qt.UserRole + 4
    MAX_ROLE = Qt.UserRole + 5
    STEP_ROLE = Qt.UserRole + 6
    PERCENT_ROLE = Qt.UserRole + 7

    def __init__(self, limits: dict[str, float]):
        super().__init__()
        self._entries: list[dict[str, object]] = []
        definitions = {
            "max_positions": ("Liczba pozycji", 0.0, 200.0, 1.0, False),
            "max_leverage": ("Maksymalna dźwignia", 0.0, 50.0, 0.1, False),
            "max_position_pct": ("Limit ekspozycji", 0.0, 1.0, 0.01, True),
            "daily_loss_limit": ("Limit dzienny", 0.0, 1.0, 0.01, True),
            "drawdown_limit": ("Limit obsunięcia", 0.0, 1.0, 0.01, True),
            "target_volatility": ("Docelowa zmienność", 0.0, 1.0, 0.01, True),
            "stop_loss_atr_multiple": ("Stop loss (ATR)", 0.0, 10.0, 0.1, False),
        }
        for key, meta in definitions.items():
            label, min_value, max_value, step, is_percent = meta
            value = limits.get(key, 0.0)
            self._entries.append(
                {
                    "key": key,
                    "label": label,
                    "value": float(value),
                    "minimum": min_value,
                    "maximum": max_value,
                    "step": step,
                    "isPercent": is_percent,
                }
            )
        for key, value in limits.items():
            if any(entry["key"] == key for entry in self._entries):
                continue
            self._entries.append(
                {
                    "key": key,
                    "label": key,
                    "value": float(value),
                    "minimum": 0.0,
                    "maximum": max(1.0, float(value) * 5.0),
                    "step": 0.1,
                    "isPercent": False,
                }
            )
        self._limits = dict(limits)

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        entry = self._entries[index.row()]
        if role == self.KEY_ROLE:
            return entry["key"]
        if role == self.LABEL_ROLE:
            return entry["label"]
        if role == self.VALUE_ROLE:
            return entry["value"]
        if role == self.MIN_ROLE:
            return entry["minimum"]
        if role == self.MAX_ROLE:
            return entry["maximum"]
        if role == self.STEP_ROLE:
            return entry["step"]
        if role == self.PERCENT_ROLE:
            return entry["isPercent"]
        return None

    def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
        return {
            self.KEY_ROLE: b"key",
            self.LABEL_ROLE: b"label",
            self.VALUE_ROLE: b"value",
            self.MIN_ROLE: b"minimum",
            self.MAX_ROLE: b"maximum",
            self.STEP_ROLE: b"step",
            self.PERCENT_ROLE: b"isPercent",
        }

    @Property("QVariantMap", constant=True)
    def limits(self) -> dict[str, float]:  # type: ignore[override]
        return dict(self._limits)

    @Slot(str, float, result=bool)
    def setLimitValue(self, key: str, value: float) -> bool:
        for idx, entry in enumerate(self._entries):
            if entry["key"] == key:
                entry["value"] = float(value)
                self._limits[key] = float(value)
                self.dataChanged.emit(self.index(idx, 0), self.index(idx, 0), [self.VALUE_ROLE])
                return True
        return False


class EngineCostModel(QAbstractListModel):
    KEY_ROLE = Qt.UserRole + 1
    LABEL_ROLE = Qt.UserRole + 2
    VALUE_ROLE = Qt.UserRole + 3
    FORMATTED_ROLE = Qt.UserRole + 4

    def __init__(self, summary: dict[str, object]):
        super().__init__()
        self._entries: list[dict[str, object]] = []
        labels = {
            "dailyRealizedPnl": "Zrealizowany wynik (dzień)",
            "grossNotional": "Wartość brutto pozycji",
            "activePositions": "Aktywne pozycje",
            "dailyLossPct": "Strata dzienna",
            "drawdownPct": "Obsunięcie kapitału",
            "averageCostBps": "Średni koszt (bps)",
            "totalCostBps": "Łączny koszt (bps)",
        }
        for key, label in labels.items():
            value = summary.get(key)
            if value is None:
                continue
            formatted = value
            if key.endswith("Pct"):
                formatted = f"{float(value) * 100:.2f} %"
            elif isinstance(value, float):
                formatted = f"{float(value):.2f}"
            self._entries.append({"key": key, "label": label, "value": value, "formatted": formatted})

    def rowCount(self, parent: QModelIndex | None = None) -> int:  # type: ignore[override]
        return len(self._entries)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> object:  # type: ignore[override]
        if not index.isValid() or not (0 <= index.row() < len(self._entries)):
            return None
        entry = self._entries[index.row()]
        if role == self.KEY_ROLE:
            return entry["key"]
        if role == self.LABEL_ROLE:
            return entry["label"]
        if role == self.VALUE_ROLE:
            return entry["value"]
        if role == self.FORMATTED_ROLE:
            return entry["formatted"]
        return None

    def roleNames(self) -> dict[int, bytes]:  # type: ignore[override]
        return {
            self.KEY_ROLE: b"key",
            self.LABEL_ROLE: b"label",
            self.VALUE_ROLE: b"value",
            self.FORMATTED_ROLE: b"formatted",
        }


class ControllerStub(QObject):
    killSwitchChanged = Signal()

    def __init__(self, engaged: bool) -> None:
        super().__init__()
        self._kill_switch = engaged

    @Property(bool, notify=killSwitchChanged)
    def riskKillSwitchEngaged(self) -> bool:  # type: ignore[override]
        return self._kill_switch

    @Slot(bool, result=bool)
    def setRiskKillSwitchEngaged(self, engaged: bool) -> bool:
        if self._kill_switch == engaged:
            return True
        self._kill_switch = engaged
        self.killSwitchChanged.emit()
        return True


@pytest.mark.timeout(30)
def test_risk_controls_panel_handles_engine_snapshot(tmp_path):
    app = QApplication.instance() or QApplication([])

    engine = ThresholdRiskEngine()
    profile = build_profile("balanced")
    engine.register_profile(profile)

    now = datetime.utcnow()
    engine.on_mark_to_market(
        profile_name=profile.name,
        equity=200_000.0,
        positions={},
        realized_pnl=0.0,
        timestamp=now,
    )
    engine.on_mark_to_market(
        profile_name=profile.name,
        equity=120_000.0,
        positions={"BTCUSDT": {"side": "long", "notional": 75_000.0}},
        realized_pnl=-12_500.0,
        timestamp=now,
    )

    snapshot = engine.snapshot_state(profile.name)
    limits_model = EngineLimitsModel(snapshot.get("limits", {}))
    statistics = snapshot.get("statistics", {})
    assert isinstance(statistics, dict)
    assert statistics, "ThresholdRiskEngine powinien zwrócić sekcję statistics"

    cost_summary = dict(statistics)
    cost_summary.update(snapshot.get("cost_breakdown", {}))
    cost_model = EngineCostModel(cost_summary)
    controller = ControllerStub(bool(snapshot.get("force_liquidation", False)))

    engine_qml = QQmlApplicationEngine()
    context = engine_qml.rootContext()
    context.setContextProperty("limitsModel", limits_model)
    context.setContextProperty("costModel", cost_model)
    context.setContextProperty("appController", controller)

    source_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "views" / "RiskControls.qml"
    temp_view = tmp_path / "RiskControls.qml"
    temp_view.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")

    engine_qml.load(QUrl.fromLocalFile(str(temp_view)))
    assert engine_qml.rootObjects(), "Nie udało się załadować panelu ryzyka"

    root = engine_qml.rootObjects()[0]
    limits_view = root.findChild(QObject, "limitsListView")
    assert limits_view is not None
    assert limits_view.property("count") > 0

    kill_switch = root.findChild(QObject, "killSwitchToggle")
    assert kill_switch is not None
    assert kill_switch.property("checked") == controller.riskKillSwitchEngaged

    limits_model.setLimitValue("max_positions", 15.0)
    assert pytest.approx(limits_model.limits()["max_positions"], rel=1e-6) == 15.0

    controller.setRiskKillSwitchEngaged(not controller.riskKillSwitchEngaged)
    app.processEvents()
    assert kill_switch.property("checked") == controller.riskKillSwitchEngaged

    for obj in engine_qml.rootObjects():
        obj.deleteLater()
    engine_qml.deleteLater()
    app.processEvents()

