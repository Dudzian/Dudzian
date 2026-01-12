from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import pytest

from tests.ui._qt import require_pyside6

pytestmark = pytest.mark.qml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

require_pyside6()

from PySide6.QtCore import (QAbstractListModel, QModelIndex, QObject, Property,
                            Qt, QUrl, Signal, Slot)
from PySide6.QtQml import QQmlApplicationEngine

try:  # pragma: no cover - zależy od środowiska CI
    from PySide6.QtWidgets import QApplication
except ImportError as exc:  # pragma: no cover
    pytest.skip(f"Brak zależności QtWidgets: {exc}", allow_module_level=True)

from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.simulation import build_profile


def _repo_root_candidates(start: Path) -> list[Path]:
    resolved = start.resolve()
    return [resolved, *resolved.parents]


def find_repo_root(start: Path) -> Path:
    candidates = _repo_root_candidates(start)
    marker_root: Path | None = None
    for candidate in candidates:
        if (candidate / "ui" / "qml" / "views").is_dir():
            return candidate
        if marker_root is None and (candidate / "pyproject.toml").is_file():
            marker_root = candidate
        if marker_root is None and (candidate / ".git").exists():
            marker_root = candidate
    if marker_root is not None:
        return marker_root
    checked = ", ".join(str(path) for path in candidates)
    raise AssertionError(
        "Nie udało się odnaleźć katalogu repozytorium. "
        f"Sprawdzone lokalizacje: {checked}"
    )


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
    riskKillSwitchChanged = Signal()

    def __init__(self, engaged: bool) -> None:
        super().__init__()
        self._kill_switch = engaged

    @Property(bool, notify=riskKillSwitchChanged)
    def riskKillSwitchEngaged(self) -> bool:  # type: ignore[override]
        return self._kill_switch

    @Slot(bool, result=bool)
    def setRiskKillSwitchEngaged(self, engaged: bool) -> bool:
        if self._kill_switch == engaged:
            return True
        self._kill_switch = engaged
        self.riskKillSwitchChanged.emit()
        return True


def _get_limits_dict(model: object) -> dict[str, float]:
    """
    EngineLimitsModel exposes limits either as:
      - a method: limits() -> dict
      - a dict-like attribute: limits
    Keep the test compatible across variants.
    """
    limits_attr = getattr(model, "limits", None)
    if callable(limits_attr):
        return limits_attr()
    if isinstance(limits_attr, Mapping):
        return dict(limits_attr)
    raise AssertionError(f"Nieobsługiwany typ model.limits: {type(limits_attr)!r}")


@pytest.mark.timeout(30)
def test_risk_controls_panel_handles_engine_snapshot():
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

    repo_root = find_repo_root(Path(__file__))
    source_path = repo_root / "ui" / "qml" / "views" / "RiskControls.qml"
    if not source_path.is_file():
        checked = ", ".join(str(path) for path in _repo_root_candidates(Path(__file__)))
        raise AssertionError(
            "Nie znaleziono pliku RiskControls.qml. "
            f"Sprawdzona ścieżka: {source_path}. "
            f"Kandydaci na root: {checked}"
        )
    engine_qml.addImportPath(str(repo_root / "ui" / "qml"))
    engine_qml.addImportPath(str(repo_root / "ui"))
    qml_warnings: list[str] = []
    engine_qml.warnings.connect(
        lambda warns: qml_warnings.extend(
            f"{warning.url().toString()}:{warning.line()}:{warning.column()} {warning.description()}"
            for warning in warns
        )
    )
    clear_cache = getattr(engine_qml, "clearComponentCache", None)
    if callable(clear_cache):
        clear_cache()

    source_url = QUrl.fromLocalFile(str(source_path))
    engine_qml.load(source_url)
    missing_property_warning = next(
        (warning for warning in qml_warnings if "Cannot assign to non-existent property" in warning),
        None,
    )
    if missing_property_warning:
        raise AssertionError(
            "Wykryto ostrzeżenie QML o nieistniejącej właściwości:\n"
            f"{missing_property_warning}"
        )
    if not engine_qml.rootObjects():
        warnings_summary = "\n".join(qml_warnings) if qml_warnings else "(brak warnings)"
        errors_provider = getattr(engine_qml, "errors", None)
        if callable(errors_provider):
            qml_errors = [
                f"{error.url().toString()}:{error.line()}:{error.column()} {error.description()}"
                for error in errors_provider()
            ]
        else:
            qml_errors = []
        errors_summary = "\n".join(qml_errors) if qml_errors else "(brak errors)"
        raise AssertionError(
            "Nie udało się załadować panelu ryzyka.\n"
            f"Źródło: {source_path}\n"
            f"QML source URL: {source_url.toString()}\n"
            f"Importy QML: {engine_qml.importPathList()}\n"
            f"Ostrzeżenia QML:\n{warnings_summary}\n"
            f"Błędy QML:\n{errors_summary}"
        )

    root = engine_qml.rootObjects()[0]
    root.setProperty("limitsModel", limits_model)
    root.setProperty("costModel", cost_model)
    root.setProperty("appController", controller)
    app.processEvents()
    limits_view = root.findChild(QObject, "limitsListView")
    assert limits_view is not None
    assert limits_view.property("count") > 0

    kill_switch = root.findChild(QObject, "killSwitchToggle")
    assert kill_switch is not None
    assert kill_switch.property("checked") == controller.riskKillSwitchEngaged

    limits_model.setLimitValue("max_positions", 15.0)
    limits_after = _get_limits_dict(limits_model)
    assert "max_positions" in limits_after
    assert pytest.approx(limits_after["max_positions"], rel=1e-6) == 15.0

    controller.setRiskKillSwitchEngaged(not controller.riskKillSwitchEngaged)
    app.processEvents()
    assert kill_switch.property("checked") == controller.riskKillSwitchEngaged

    for obj in engine_qml.rootObjects():
        obj.deleteLater()
    engine_qml.deleteLater()
    app.processEvents()
