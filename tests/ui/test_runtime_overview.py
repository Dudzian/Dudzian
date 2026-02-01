from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
import base64
from pathlib import Path
from typing import Any, Mapping, Iterator, NoReturn
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_QT_IMPORT_ERROR: Exception | None = None
try:  # pragma: no cover - zależne od środowiska
    from PySide6.QtCore import (  # type: ignore[attr-defined]
        QObject,
        Property,
        QUrl,
        Qt,
        QMetaObject,
        Q_ARG,
        Signal,
        QByteArray,
    )
    from PySide6.QtQml import QQmlApplicationEngine  # type: ignore[attr-defined]
    from PySide6.QtGui import QImage  # type: ignore[attr-defined]
    from PySide6.QtWidgets import QApplication  # type: ignore[attr-defined]
    _QT_READY = True
except ImportError as exc:  # pragma: no cover - brak PySide6 lub zależności systemowych
    _QT_READY = False
    _QT_IMPORT_ERROR = exc

    class QObject:  # type: ignore[no-redef]
        pass

    class Property:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def __call__(self, func: Any) -> Any:
            return func

    class QUrl:  # type: ignore[no-redef]
        pass

    class Qt:  # type: ignore[no-redef]
        pass

    class QMetaObject:  # type: ignore[no-redef]
        pass

    def Q_ARG(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        return None

    class Signal:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    class QByteArray:  # type: ignore[no-redef]
        pass

    class QQmlApplicationEngine:  # type: ignore[no-redef]
        pass

    class QApplication:  # type: ignore[no-redef]
        pass

    QImage = None  # type: ignore[assignment]

_QIMAGE_TYPE: type | None = QImage if _QT_READY else None

pytestmark = [pytest.mark.qml]
if not _QT_READY:
    qt_qpa_platform = os.getenv("QT_QPA_PLATFORM", "<unset>")
    pytestmark.append(
        pytest.mark.skip(
            reason=(
                "Brak zależności Qt/PySide6"
                f" na {sys.platform} (QT_QPA_PLATFORM={qt_qpa_platform}): {_QT_IMPORT_ERROR}"
            )
        )
    )

from core.monitoring.metrics_api import (
    ComplianceTelemetry,
    GuardrailOverview,
    IOQueueTelemetry,
    RetrainingTelemetry,
    RuntimeTelemetrySnapshot,
)
from tests.ui._qt_utils import qt_wait

if _QT_READY:
    from bot_core.observability.metrics import MetricsRegistry
    from bot_core.observability.ui_metrics import (
        FeedHealthMetricsExporter,
        RiskJournalMetricsExporter,
    )
    from ui.backend import runtime_service as runtime_service_module
    from ui.backend.runtime_service import RuntimeService
    from ui.backend.telemetry_provider import TelemetryProvider
    from ui.backend.qml_bridge import to_plain_value
else:  # pragma: no cover - brak PySide6 w środowisku collect-only
    MetricsRegistry = None  # type: ignore[assignment]
    FeedHealthMetricsExporter = None  # type: ignore[assignment]
    RiskJournalMetricsExporter = None  # type: ignore[assignment]
    runtime_service_module = None  # type: ignore[assignment]
    RuntimeService = None  # type: ignore[assignment]
    TelemetryProvider = None  # type: ignore[assignment]
    def to_plain_value(value: Any) -> Any:  # type: ignore[override]
        return value


@pytest.fixture(autouse=True)
def qt_runtime_sanity() -> Iterator[None]:
    if not _QT_READY:
        yield
        return
    created_here = QApplication.instance() is None
    try:
        app = QApplication.instance() or QApplication([])
    except Exception as exc:  # pragma: no cover - awaria pluginu platformy
        qt_qpa_platform = os.getenv("QT_QPA_PLATFORM", "<unset>")
        pytest.skip(
            f"Qt runtime niedostępny na {sys.platform} "
            f"(QT_QPA_PLATFORM={qt_qpa_platform}): {exc}"
        )
    yield
    if created_here and app is not None:
        app.processEvents()
        app.quit()
        app.processEvents()


@pytest.fixture(autouse=True)
def qml_prop(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Zapewnia, że property() z QML zwraca plain Python w ramach tego modułu."""

    original_property = QObject.property

    def _plain_property(self: QObject, name: str) -> object:
        value = original_property(self, name)
        if isinstance(value, QObject):
            return value
        if _QIMAGE_TYPE is not None and isinstance(value, _QIMAGE_TYPE):
            return value
        return to_plain_value(value)

    monkeypatch.setattr(QObject, "property", _plain_property, raising=True)
    yield


@pytest.fixture
def decision_feed_degradation_samples() -> list[dict[str, float | None]]:
    return [
        {"status": "connected", "p95_ms": 2100.0, "downtimeMs": 0.0, "nextRetrySeconds": None},
        {"status": "retrying", "p95_ms": 2950.0, "downtimeMs": 480.0, "nextRetrySeconds": 1.5},
    ]


class _StubTelemetryProvider(QObject):
    """Minimalny provider emitujący zmiany dla testów live."""

    errorMessageChanged = Signal()
    telemetryUpdated = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._last_updated = ""
        self._error_message = ""

    @Property(str, notify=errorMessageChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    def refreshTelemetry(self) -> bool:
        self._last_updated = datetime.now(timezone.utc).isoformat()
        self.telemetryUpdated.emit()
        return True

    def push_error(self, message: str) -> None:
        self._error_message = message
        self.errorMessageChanged.emit()


class _StubRuntimeService(QObject):
    """Uproszczony serwis runtime do symulacji sygnałów live."""

    decisionsChanged = Signal()
    errorMessageChanged = Signal()
    riskMetricsChanged = Signal()
    riskTimelineChanged = Signal()
    operatorActionChanged = Signal()
    longPollMetricsChanged = Signal()
    cycleMetricsChanged = Signal()
    feedTransportSnapshotChanged = Signal()
    feedHealthChanged = Signal()
    feedSlaReportChanged = Signal()
    feedAlertHistoryChanged = Signal()
    feedAlertChannelsChanged = Signal()
    aiRegimeBreakdownChanged = Signal()
    adaptiveStrategySummaryChanged = Signal()
    regimeActivationSummaryChanged = Signal()

    def __init__(self) -> None:
        super().__init__()
        self._decisions: list[dict[str, Any]] = []
        self._error_message = ""
        self._risk_metrics: dict[str, Any] = {}
        self._risk_timeline: list[dict[str, Any]] = []
        self._last_operator_action: dict[str, Any] = {}
        self._longpoll_metrics: list[dict[str, Any]] = []
        self._cycle_metrics: dict[str, float] = {}
        self._feed_transport_snapshot: dict[str, Any] = {
            "status": "initializing",
            "mode": "demo",
            "label": "",
            "reconnects": 0,
        }
        self._feed_health: dict[str, Any] = {
            "status": "initializing",
            "reconnects": 0,
            "downtimeMs": 0.0,
            "lastError": "",
        }
        self._feed_sla_report: dict[str, Any] = {}
        self._feed_alert_history: list[dict[str, Any]] = []
        self._feed_alert_channels: list[dict[str, Any]] = []
        self._ai_regime_breakdown: list[dict[str, Any]] = []
        self._adaptive_summary = ""
        self._regime_summary = ""

    @Property("QVariantList", notify=decisionsChanged)
    def decisions(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._decisions)

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    @Property("QVariantMap", notify=riskMetricsChanged)
    def riskMetrics(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._risk_metrics)

    @Property("QVariantList", notify=riskTimelineChanged)
    def riskTimeline(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._risk_timeline)

    @Property("QVariantMap", notify=operatorActionChanged)
    def lastOperatorAction(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._last_operator_action)

    @Property("QVariantList", notify=longPollMetricsChanged)
    def longPollMetrics(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return [dict(entry) for entry in self._longpoll_metrics]

    @Property("QVariantMap", notify=cycleMetricsChanged)
    def cycleMetrics(self) -> dict[str, Any]:  # type: ignore[override]
        return {key: float(value) for key, value in self._cycle_metrics.items()}

    @Property("QVariantMap", notify=feedTransportSnapshotChanged)
    def feedTransportSnapshot(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._feed_transport_snapshot)

    @Property("QVariantMap", notify=feedHealthChanged)
    def feedHealth(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._feed_health)

    @Property("QVariantMap", notify=feedSlaReportChanged)
    def feedSlaReport(self) -> dict[str, Any]:  # type: ignore[override]
        return dict(self._feed_sla_report)

    @Property("QVariantList", notify=feedAlertHistoryChanged)
    def feedAlertHistory(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._feed_alert_history)

    @Property("QVariantList", notify=feedAlertChannelsChanged)
    def feedAlertChannels(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return list(self._feed_alert_channels)

    @Property("QVariantList", notify=aiRegimeBreakdownChanged)
    def aiRegimeBreakdown(self) -> list[dict[str, Any]]:  # type: ignore[override]
        return [dict(entry) for entry in self._ai_regime_breakdown]

    @Property(str, notify=adaptiveStrategySummaryChanged)
    def adaptiveStrategySummary(self) -> str:  # type: ignore[override]
        return self._adaptive_summary

    @Property(str, notify=regimeActivationSummaryChanged)
    def regimeActivationSummary(self) -> str:  # type: ignore[override]
        return self._regime_summary

    def loadRecentDecisions(self, limit: int = 0) -> list[dict[str, Any]]:
        if limit > 0:
            return list(self._decisions[:limit])
        return list(self._decisions)

    def push_decisions(self, payload: list[dict[str, Any]]) -> None:
        self._decisions = list(payload)
        self.decisionsChanged.emit()

    def push_error(self, message: str) -> None:
        self._error_message = message
        self.errorMessageChanged.emit()

    def push_risk_update(
        self,
        metrics: dict[str, Any],
        timeline: list[dict[str, Any]],
        action: dict[str, Any] | None = None,
    ) -> None:
        self._risk_metrics = dict(metrics)
        self._risk_timeline = list(timeline)
        self.riskMetricsChanged.emit()
        self.riskTimelineChanged.emit()
        if action is not None:
            self._last_operator_action = dict(action)
            self.operatorActionChanged.emit()

    def push_longpoll_metrics(self, payload: list[dict[str, Any]]) -> None:
        self._longpoll_metrics = [dict(entry) for entry in payload]
        self.longPollMetricsChanged.emit()

    def push_cycle_metrics(self, payload: dict[str, float]) -> None:
        self._cycle_metrics = {str(key): float(value) for key, value in payload.items()}
        self.cycleMetricsChanged.emit()

    def push_feed_transport(self, payload: dict[str, Any]) -> None:
        self._feed_transport_snapshot = dict(payload)
        self.feedTransportSnapshotChanged.emit()

    def push_feed_health(self, payload: dict[str, Any]) -> None:
        self._feed_health = dict(payload)
        self.feedHealthChanged.emit()

    def push_feed_sla_report(self, payload: dict[str, Any]) -> None:
        self._feed_sla_report = dict(payload)
        self.feedSlaReportChanged.emit()

    def push_feed_alerts(self, history: list[dict[str, Any]], channels: list[dict[str, Any]]) -> None:
        self._feed_alert_history = [dict(entry) for entry in history]
        self._feed_alert_channels = [dict(entry) for entry in channels]
        self.feedAlertHistoryChanged.emit()
        self.feedAlertChannelsChanged.emit()

    def push_ai_regimes(self, payload: list[dict[str, Any]]) -> None:
        self._ai_regime_breakdown = [dict(entry) for entry in payload]
        self.aiRegimeBreakdownChanged.emit()

    def set_adaptive_summary(self, summary: str, activation: str) -> None:
        if self._adaptive_summary != summary:
            self._adaptive_summary = summary
            self.adaptiveStrategySummaryChanged.emit()
        if self._regime_summary != activation:
            self._regime_summary = activation
            self.regimeActivationSummaryChanged.emit()


def _sample_snapshot() -> RuntimeTelemetrySnapshot:
    generated = datetime(2025, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    io_entries = (
        IOQueueTelemetry(
            environment="prod",
            queue="binance.spot",
            timeout_total=2.0,
            timeout_avg_seconds=1.25,
            rate_limit_wait_total=4.0,
            rate_limit_wait_avg_seconds=0.8,
            severity="warning",
        ),
    )
    guardrail = GuardrailOverview(
        total_queues=1,
        normal_queues=0,
        info_queues=0,
        warning_queues=1,
        error_queues=0,
        total_timeouts=2.0,
        total_rate_limit_waits=4.0,
    )
    retraining_entries = (
        RetrainingTelemetry(
            status="completed",
            runs=3,
            average_duration_seconds=42.5,
            average_drift_score=0.12,
        ),
    )
    compliance = ComplianceTelemetry(
        total_violations=1.0,
        by_severity={"warning": 1.0},
        by_rule={"KYC_MISSING_FIELDS": 1.0},
    )
    return RuntimeTelemetrySnapshot(
        generated_at=generated,
        io_queues=io_entries,
        guardrail_overview=guardrail,
        retraining=retraining_entries,
        compliance=compliance,
    )


def _sample_decisions() -> list[dict[str, str]]:
    return [
        {
            "event": "order_submitted",
            "timestamp": "2025-01-01T12:00:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "balanced",
            "strategy": "mean_reversion",
            "schedule": "auto",
            "symbol": "BTC/USDT",
            "side": "buy",
            "status": "submitted",
            "decision_state": "trade",
            "decision_signal": "long",
            "decision_should_trade": "true",
            "decision_model": "xgb-v5",
            "decision_confidence": "0.9120",
            "ai_probability": "0.8450",
            "market_regime": "bull",
            "market_regime_risk_level": "elevated",
            "market_regime_confidence": "0.7800",
            "strategy_recommendation": "momentum_v2",
        }
    ]


def _sample_risk_decisions() -> list[dict[str, object]]:
    return [
        {
            "event": "risk_blocked",
            "timestamp": "2025-01-02T09:15:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "momentum_v2",
            "status": "risk_block",
            "decision_state": "halt",
            "decision_should_trade": "false",
            "risk_flags": ["latency_spike"],
            "stress_failures": ["latency_spike"],
            "stress_overrides": [{"reason": "latency_spike"}],
        },
        {
            "event": "risk_update",
            "timestamp": "2025-01-02T10:00:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "mean_reversion",
            "status": "ok",
            "decision_state": "monitor",
            "decision_should_trade": "false",
            "risk_flags": ["drawdown_watch"],
        },
        {
            "event": "risk_freeze",
            "timestamp": "2025-01-02T11:30:00+00:00",
            "environment": "prod",
            "portfolio": "alpha",
            "risk_profile": "dynamic",
            "strategy": "momentum_v2",
            "status": "risk_freeze",
            "risk_action": "manual_freeze",
            "decision_state": "halt",
            "decision_should_trade": "false",
            "risk_flags": ["latency_spike"],
        },
    ]


@pytest.mark.timeout(30)
def test_runtime_overview_renders_snapshot(tmp_path: Path) -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)
    runtime_service = RuntimeService(decision_loader=lambda limit: [])
    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    engine.rootContext().setContextProperty("ciSnapshot", True)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]
    # Utrzymujemy deterministyczną kolejność kart w snapshotach, ale nie zerujemy
    # dashboardSettingsController, bo QML może od niego zależeć przy budowie listy kart.
    default_order = root.property("defaultCardOrder")

    class _StubDashboardSettingsController(QObject):
        def __init__(self, visible_order: object, parent: QObject | None = None) -> None:
            super().__init__(parent)
            self._visible_order = visible_order

        @Property("QVariant", constant=True)
        def visibleCardOrder(self) -> object:
            return self._visible_order

    stub_controller = _StubDashboardSettingsController(default_order, parent=root)
    root.setProperty("dashboardSettingsController", stub_controller)
    root.setProperty("complianceController", None)
    root.setProperty("reportController", None)

    # Dajemy QML chwilę na przepięcie bindingów i ewentualne zbudowanie listy kart.
    app.processEvents()
    qt_wait(50)

    ok = provider.refreshTelemetry()
    assert ok is True
    app.processEvents()
    assert provider.lastUpdated, "TelemetryProvider.lastUpdated pozostał pusty po refreshTelemetry()."

    summary = provider.complianceSummary
    assert summary["totalViolations"] == 1.0

    last_updated = root.findChild(QObject, "runtimeOverviewLastUpdated")
    assert last_updated is not None

    def _last_updated_text() -> str:
        return str(last_updated.property("text"))

    def _wait_for_last_updated(timeout_s: float = 2.0) -> str:
        start = time.monotonic()
        while time.monotonic() - start < timeout_s:
            app.processEvents()
            current = _last_updated_text()
            if "n/d" not in current:
                return current
            qt_wait(50)
        return _last_updated_text()

    label_text = _wait_for_last_updated()
    assert "n/d" not in label_text, (
        "Label runtimeOverviewLastUpdated nie zaktualizował się po refreshTelemetry. "
        f"text={label_text!r}, provider.lastUpdated={provider.lastUpdated!r}"
    )
    assert re.search(r"\d{4}", label_text), (
        "Label runtimeOverviewLastUpdated nie zawiera roku. "
        f"text={label_text!r}, provider.lastUpdated={provider.lastUpdated!r}"
    )
    year_match = re.search(r"\d{4}", str(provider.lastUpdated))
    if year_match:
        year = year_match.group(0)
        assert year in label_text, (
            "Label runtimeOverviewLastUpdated nie zawiera roku z provider.lastUpdated. "
            f"text={label_text!r}, provider.lastUpdated={provider.lastUpdated!r}"
        )

    # QQuickLoader status codes: Null=0, Ready=1, Loading=2, Error=3 (QtQuick.Loader / Loader.*)
    LOADER_READY = 1
    LOADER_ERROR = 3
    LOADER_STATUS_NAME = {0: "Null", 1: "Ready", 2: "Loading", 3: "Error"}

    def _normalize_error_string(loader: QObject) -> str:
        error_attr = getattr(loader, "errorString", None)
        if callable(error_attr):
            return str(error_attr())
        if error_attr is not None:
            return str(error_attr)
        return ""

    def _safe_prop(obj: QObject, name: str) -> object:
        try:
            return obj.property(name)
        except RuntimeError as exc:
            return f"<RuntimeError: {exc}>"
        except Exception as exc:  # pragma: no cover
            return f"<{type(exc).__name__}: {exc}>"

    def _safe_int(value: object, default: int = -1) -> int:
        if value is None:
            return default
        if isinstance(value, str) and value.startswith("<"):
            return default
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return default

    def _normalize_source_component(loader: QObject) -> str:
        source_component = _safe_prop(loader, "sourceComponent")
        if isinstance(source_component, str) and source_component.startswith("<"):
            return ""
        return "" if source_component is None else str(source_component)

    def _class_name(obj: QObject) -> str:
        try:
            return str(obj.metaObject().className())
        except Exception:
            return "<unknown>"

    def _fail_loader_error(loader: QObject, message_prefix: str) -> NoReturn:
        active = _safe_prop(loader, "active")
        source_component = _normalize_source_component(loader)
        error_string = _normalize_error_string(loader)
        card_id = str(_safe_prop(loader, "cardId") or "")
        status = _safe_int(_safe_prop(loader, "status"), default=-1)
        status_name = LOADER_STATUS_NAME.get(status, "Unknown")
        pytest.fail(
            f"{message_prefix} "
            f"objectName={loader.objectName()}, cardId={card_id}, status={status}, "
            f"statusName={status_name}, active={active}, sourceComponent={source_component}, "
            f"errorString={error_string}"
        )

    def _collect_card_loaders() -> list[QObject]:
        loaders: list[QObject] = []
        for child in root.findChildren(QObject):
            object_name = str(child.objectName())
            if object_name.startswith("runtimeOverviewCardLoader_"):
                loaders.append(child)
                continue
            # Najpierw filtrujemy po cardId, żeby nie dotykać "status" na obcych typach
            # (np. QQuickFontLoader ma status o typie bez konwertera na Windows).
            card_id = _safe_prop(child, "cardId")
            if isinstance(card_id, str) and card_id.startswith("<"):
                continue
            if card_id is None:
                continue
            status = _safe_prop(child, "status")
            if isinstance(status, str) and status.startswith("<"):
                continue
            if status is None:
                continue
            status_i = _safe_int(status, default=-1)
            source_component = _safe_prop(child, "sourceComponent")
            if isinstance(source_component, str) and source_component.startswith("<"):
                source_component = None
            active = _safe_prop(child, "active")
            if isinstance(active, str) and active.startswith("<"):
                active = None
            item = _safe_prop(child, "item")
            if isinstance(item, str) and item.startswith("<"):
                item = None
            if source_component is None and active is None and item is None:
                if status_i != LOADER_ERROR:
                    continue
            loaders.append(child)
        return loaders

    def _find_guardrail_loader(aliases: set[str], deadline: float) -> QObject | None:
        cached_loaders: list[QObject] | None = None
        last_refresh = 0.0
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            app.processEvents()
            if cached_loaders is None or now - last_refresh >= 0.2:
                cached_loaders = _collect_card_loaders()
                last_refresh = now
            loaders = cached_loaders if cached_loaders is not None else []
            for loader in loaders:
                card_id = str(_safe_prop(loader, "cardId") or "")
                if card_id in aliases:
                    status = _safe_int(_safe_prop(loader, "status"), default=-1)
                    if status == LOADER_ERROR:
                        _fail_loader_error(
                            loader,
                            "Loader guardrail (alias cardId) zakończył się błędem podczas lookup.",
                        )
                    return loader
            qt_wait(50)
        return None

    def _wait_for_loader_item(loader: QObject, deadline: float) -> QObject | None:
        while time.monotonic() < deadline:
            app.processEvents()
            status = _safe_int(_safe_prop(loader, "status"), default=-1)
            if status == LOADER_ERROR:
                _fail_loader_error(
                    loader,
                    "Loader guardrail (alias cardId) zakończył się błędem podczas oczekiwania.",
                )
            if status == LOADER_READY:
                item = loader.property("item")
                if item is not None:
                    return item
            qt_wait(50)
        return loader.property("item")

    guardrail_aliases = {"guardrails", "guardrail", "guardrails_card", "guardrail_card"}
    guardrail_timeout_s = 6.0
    guardrail_deadline = time.monotonic() + guardrail_timeout_s
    guardrail_loader = _find_guardrail_loader(guardrail_aliases, deadline=guardrail_deadline)
    if guardrail_loader is None:
        # Szerokie diagnostyki: co w ogóle mamy w drzewie QML?
        all_children = root.findChildren(QObject)
        sample = []
        for child in all_children[:250]:  # limit, żeby nie zalać logów
            object_name = str(child.objectName())
            card_id = _safe_prop(child, "cardId")
            card_id_ok = not (isinstance(card_id, str) and card_id.startswith("<"))
            if (card_id is not None and card_id_ok) or ("Loader" in object_name) or ("runtimeOverview" in object_name) or ("card" in object_name.lower()):
                sample.append(
                    {
                        "class": _class_name(child),
                        "objectName": object_name,
                        "cardId": card_id if card_id_ok else None,
                        "status": _safe_prop(child, "status"),
                        "item": _safe_prop(child, "item"),
                        "active": _safe_prop(child, "active"),
                        "sourceComponent": _safe_prop(child, "sourceComponent"),
                    }
                )
        available_loaders = []
        for child in _collect_card_loaders():
            status = _safe_int(_safe_prop(child, "status"), default=-1)
            available_loaders.append(
                {
                    "objectName": str(child.objectName()),
                    "cardId": str(_safe_prop(child, "cardId") or ""),
                    "status": status,
                    "statusName": LOADER_STATUS_NAME.get(status, "Unknown"),
                    "active": _safe_prop(child, "active"),
                    "sourceComponent": _normalize_source_component(child),
                    "errorString": _normalize_error_string(child),
                }
            )
        default_order = root.property("defaultCardOrder")
        controller = root.property("dashboardSettingsController")
        visible_order = None
        if controller is not None:
            visible_order = controller.property("visibleCardOrder")
        pytest.fail(
            "Nie znaleziono guardrail loadera (alias cardId). "
            f"Szukano aliasów={sorted(guardrail_aliases)}, dostępne loadery={available_loaders}, "
            f"defaultCardOrder={default_order}, visibleCardOrder={visible_order}, "
            f"qmlChildrenCount={len(all_children)}, sampleChildren={sample}"
        )
    guardrail_card = _wait_for_loader_item(guardrail_loader, deadline=guardrail_deadline)
    if guardrail_card is None:
        _fail_loader_error(
            guardrail_loader,
            "Loader guardrail (alias cardId) nie osiągnął statusu Ready lub nie zwrócił item.",
        )
    assert guardrail_card.property("objectName") == "runtimeOverviewGuardrailCard"
    manual_button = root.findChild(QObject, "manualRefreshButton")
    assert manual_button is not None and manual_button.property("enabled") is True

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_ai_card_populates_decisions() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, str]]:
        return _sample_decisions()

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    provider.refreshTelemetry()
    runtime_service.loadRecentDecisions(5)
    app.processEvents()

    decisions = root.property("aiDecisions")
    assert isinstance(decisions, list)
    assert len(decisions) == 1
    first = decisions[0]
    assert first["decision"]["state"] == "trade"
    assert first["marketRegime"]["regime"] == "bull"

    card = root.findChild(QObject, "runtimeOverviewAiCard")
    assert card is not None
    error_banner = root.findChild(QObject, "runtimeOverviewAiErrorBanner")
    assert error_banner is not None
    assert error_banner.property("visible") is False

    empty_label = root.findChild(QObject, "runtimeOverviewAiEmptyLabel")
    assert empty_label is not None
    assert empty_label.property("visible") is False

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_ai_card_handles_errors() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, str]]:
        raise RuntimeError("journal offline")

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service.loadRecentDecisions(3)
    app.processEvents()

    error_text = root.property("aiDecisionError")
    assert error_text == "journal offline"
    error_banner = root.findChild(QObject, "runtimeOverviewAiErrorBanner")
    assert error_banner is not None
    assert error_banner.property("visible") is True


@pytest.mark.timeout(30)
def test_runtime_overview_risk_panel_filters_and_actions() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)

    def _loader(limit: int) -> list[dict[str, object]]:
        return _sample_risk_decisions()

    runtime_service = RuntimeService(decision_loader=_loader)

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service.loadRecentDecisions(10)
    app.processEvents()

    metrics = root.property("riskMetrics")
    assert isinstance(metrics, dict)
    assert metrics.get("blockCount") == 1
    assert "latency_spike" in metrics.get("uniqueRiskFlags", [])
    assert metrics.get("lastBlock", {}).get("timestamp") == "2025-01-02T09:15:00+00:00"
    assert metrics.get("lastFreeze", {}).get("timestamp") == "2025-01-02T11:30:00+00:00"
    assert metrics.get("lastStressOverride", {}).get("timestamp") == "2025-01-02T09:15:00+00:00"
    summaries = metrics.get("strategySummaries")
    assert isinstance(summaries, list)
    assert summaries, "Oczekiwano podsumowań strategii"
    leader = summaries[0]
    assert leader.get("strategy") == "momentum_v2"
    assert leader.get("blockCount") == 1
    assert leader.get("freezeCount") == 1
    assert leader.get("stressOverrideCount") == 1
    assert leader.get("severity") == "block"

    risk_panel = root.findChild(QObject, "riskJournalPanel")
    assert risk_panel is not None

    timeline = risk_panel.property("filteredTimeline")
    assert isinstance(timeline, list)
    assert len(timeline) == 3

    risk_panel.setProperty("riskFilter", "drawdown_watch")
    app.processEvents()
    filtered = risk_panel.property("filteredTimeline")
    assert len(filtered) == 1

    risk_panel.setProperty("riskFilter", "")
    risk_panel.setProperty("strategyFilter", "momentum_v2")
    app.processEvents()
    filtered_strategy = risk_panel.property("filteredTimeline")
    assert len(filtered_strategy) == 2

    risk_panel.setProperty("riskFilter", "drawdown_watch")
    app.processEvents()
    filtered_combo = risk_panel.property("filteredTimeline")
    assert len(filtered_combo) == 0

    risk_panel.setProperty("riskFilter", "latency_spike")
    app.processEvents()
    filtered_lat = risk_panel.property("filteredTimeline")
    assert len(filtered_lat) == 2

    assert risk_panel.property("strategyChipCount") >= 1
    assert risk_panel.property("riskFlagChipCount") >= 2
    assert risk_panel.property("stressFailureChipCount") >= 1

    top_strategy_chip = risk_panel.findChild(QObject, "riskJournalStrategyChip_0")
    assert top_strategy_chip is not None
    strategy_text = top_strategy_chip.property("chipText")
    assert isinstance(strategy_text, str)
    assert "momentum_v2" in strategy_text
    assert "2" in strategy_text

    top_flag_chip = risk_panel.findChild(QObject, "riskJournalFlagChip_0")
    assert top_flag_chip is not None
    chip_text = top_flag_chip.property("chipText")
    assert isinstance(chip_text, str)
    assert "latency_spike" in chip_text
    assert "2" in chip_text

    top_stress_chip = risk_panel.findChild(QObject, "riskJournalStressChip_0")
    assert top_stress_chip is not None
    stress_text = top_stress_chip.property("chipText")
    assert isinstance(stress_text, str)
    assert "latency_spike" in stress_text
    assert ")" in stress_text

    summary_flow = risk_panel.findChild(QObject, "riskJournalStrategySummaries")
    assert summary_flow is not None
    top_summary_card = risk_panel.findChild(QObject, "riskJournalStrategySummary_0")
    assert top_summary_card is not None
    assert top_summary_card.property("summaryStrategy") == "momentum_v2"
    assert top_summary_card.property("summaryBlockCount") == 1
    assert top_summary_card.property("summarySeverity") == "block"

    entry = filtered_lat[0]
    QMetaObject.invokeMethod(
        risk_panel,
        "openDrilldown",
        Qt.DirectConnection,
        Q_ARG("QVariant", entry),
    )
    app.processEvents()
    assert risk_panel.property("selectedEntry") is not None

    QMetaObject.invokeMethod(
        risk_panel,
        "triggerOperatorAction",
        Qt.DirectConnection,
        Q_ARG("QString", "requestFreeze"),
    )
    app.processEvents()

    last_action = runtime_service.lastOperatorAction
    assert last_action.get("action") == "freeze"
    assert last_action.get("entry", {}).get("event") == entry.get("record", {}).get("event")

    assert risk_panel.property("selectedEntry") is None

    last_action_label = risk_panel.findChild(QObject, "riskJournalLastOperatorAction")
    assert last_action_label is not None
    text = last_action_label.property("text")
    assert isinstance(text, str)
    assert "zamrożenie" in text
    assert "risk_blocked" in text

    last_block_label = risk_panel.findChild(QObject, "riskJournalLastBlock")
    assert last_block_label is not None
    assert "Ostatnia blokada" in last_block_label.property("text")
    assert "2025-01-02T09:15:00+00:00" in last_block_label.property("text")

    last_freeze_label = risk_panel.findChild(QObject, "riskJournalLastFreeze")
    assert last_freeze_label is not None
    freeze_text = last_freeze_label.property("text")
    assert "Ostatnia blokada strategiczna" in freeze_text
    assert "2025-01-02T11:30:00+00:00" in freeze_text
    assert "manual_freeze" in freeze_text

    last_override_label = risk_panel.findChild(QObject, "riskJournalLastOverride")
    assert last_override_label is not None
    override_text = last_override_label.property("text")
    assert "Ostatni stress override" in override_text
    assert "2025-01-02T09:15:00+00:00" in override_text

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_cards_react_to_live_signals() -> None:
    provider = _StubTelemetryProvider()
    runtime_service = _StubRuntimeService()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    provider.refreshTelemetry()
    runtime_service.push_decisions(
        [
            {
                "event": "order_submitted",
                "timestamp": "2025-03-01T12:00:00+00:00",
                "portfolio": "alpha",
                "environment": "prod",
                "strategy": "adaptive_alpha",
                "riskProfile": "balanced",
                "marketRegime": {"regime": "bull"},
                "decision": {"state": "trade", "model": "xgb-v6"},
                "ai": {"confidence": 0.91},
            }
        ]
    )
    app.processEvents()

    decisions = root.property("aiDecisions")
    assert isinstance(decisions, list)
    assert decisions and decisions[0]["event"] == "order_submitted"
    assert decisions[0]["decision"]["model"] == "xgb-v6"

    runtime_service.push_error("feed degraded")
    app.processEvents()

    assert root.property("aiDecisionError") == "feed degraded"
    ai_error_banner = root.findChild(QObject, "runtimeOverviewAiErrorBanner")
    assert ai_error_banner is not None and ai_error_banner.property("visible") is True

    runtime_service.push_feed_transport(
        {
            "status": "connected",
            "mode": "grpc",
            "label": "grpc://localhost:5100",
            "reconnects": 2,
            "latencyP95": 2800.0,
            "lastError": "timeout spike",
        }
    )
    runtime_service.push_feed_health(
        {
            "status": "connected",
            "reconnects": 2,
            "downtimeMs": 500.0,
            "lastError": "timeout spike",
        }
    )
    runtime_service.push_feed_sla_report(
        {
            "p95_ms": 2800.0,
            "p50_ms": 1200.0,
            "latency_state": "warning",
            "latency_warning_ms": 2500.0,
            "reconnects": 2,
            "reconnects_warning": 3,
            "reconnects_state": "ok",
            "downtime_seconds": 0.5,
            "downtime_state": "ok",
            "downtime_warning_seconds": 30.0,
            "sla_state": "warning",
            "nextRetrySeconds": 1.5,
        }
    )
    runtime_service.push_feed_alerts(
        [
            {
                "metric": "latency",
                "label": "Latencja p95",
                "severity": "warning",
                "formattedValue": "2800 ms",
                "timestamp": "2025-03-01T12:00:00Z",
            },
            {
                "metric": "reconnects",
                "label": "Reconnecty",
                "severity": "info",
                "formattedValue": "2",
                "timestamp": "2025-03-01T12:00:00Z",
            },
        ],
        [{"name": "cloud-escalation", "status": "ok"}],
    )
    app.processEvents()

    sla_card = root.findChild(QObject, "runtimeOverviewFeedSlaCard")
    assert sla_card is not None
    sla_state_label = root.findChild(QObject, "runtimeOverviewSlaStateLabel")
    assert sla_state_label is not None
    assert "connected" in sla_state_label.property("text").lower()
    sla_latency_label = root.findChild(QObject, "runtimeOverviewSlaLatency")
    assert sla_latency_label is not None and "2800" in sla_latency_label.property("text")
    sla_last_error = root.findChild(QObject, "runtimeOverviewSlaLastError")
    assert sla_last_error is not None and sla_last_error.property("visible") is True
    sla_retry = root.findChild(QObject, "runtimeOverviewSlaRetry")
    assert sla_retry is not None and "1.5" in sla_retry.property("text")
    alert_list = root.findChild(QObject, "runtimeOverviewSlaAlertList")
    assert alert_list is not None and alert_list.property("count") == 2
    escalation_label = root.findChild(QObject, "runtimeOverviewSlaEscalationStatus")
    assert escalation_label is not None
    assert "cloud-escalation" in escalation_label.property("text")

    metrics = {
        "blockCount": 1,
        "uniqueRiskFlags": ["latency_spike"],
        "uniqueStressFailures": ["latency_spike"],
        "strategySummaries": [
            {
                "strategy": "adaptive_alpha",
                "blockCount": 1,
                "freezeCount": 0,
                "stressOverrideCount": 1,
                "severity": "block",
                "lastTimestamp": "2025-03-01T12:00:00+00:00",
                "lastEvent": "risk_blocked",
                "lastRiskFlags": ["latency_spike"],
            }
        ],
        "lastBlock": {"timestamp": "2025-03-01T12:00:00+00:00", "strategy": "adaptive_alpha"},
    }
    timeline = [
        {
            "event": "risk_blocked",
            "timestamp": "2025-03-01T12:00:00+00:00",
            "strategy": "adaptive_alpha",
            "riskFlags": ["latency_spike"],
            "stressFailures": ["latency_spike"],
        }
    ]
    runtime_service.push_risk_update(
        metrics,
        timeline,
        {"action": "freeze", "entry": timeline[0], "timestamp": "2025-03-01T12:05:00+00:00"},
    )
    app.processEvents()

    risk_metrics = root.property("riskMetrics")
    assert risk_metrics.get("blockCount") == 1
    assert "latency_spike" in risk_metrics.get("uniqueRiskFlags", [])

    risk_timeline = root.property("riskTimeline")
    assert risk_timeline and risk_timeline[0]["event"] == "risk_blocked"

    operator_action = root.property("lastOperatorAction")
    assert operator_action.get("action") == "freeze"
    assert operator_action.get("timestamp") == "2025-03-01T12:05:00+00:00"

    provider.push_error("telemetry degraded")
    app.processEvents()

    banner = root.findChild(QObject, "runtimeOverviewErrorBanner")
    assert banner is not None and banner.property("visible") is True

    runtime_service.push_cycle_metrics(
        {
            "cycles_total": 42.0,
            "strategy_switch_total": 6.0,
            "guardrail_blocks_total": 2.0,
            "cycle_latency_p50_ms": 1250.0,
            "cycle_latency_p95_ms": 2450.0,
        }
    )
    app.processEvents()

    cycle_group = root.findChild(QObject, "runtimeOverviewCycleMetricsGroup")
    assert cycle_group is not None
    cycle_count = cycle_group.findChild(QObject, "runtimeOverviewCycleCount")
    assert cycle_count is not None and "42" in cycle_count.property("text")
    strategy_switches = cycle_group.findChild(QObject, "runtimeOverviewStrategySwitches")
    assert strategy_switches is not None and "6" in strategy_switches.property("text")
    guardrail_blocks = cycle_group.findChild(QObject, "runtimeOverviewGuardrailBlocks")
    assert guardrail_blocks is not None and "2" in guardrail_blocks.property("text")
    guardrail_alert = cycle_group.findChild(QObject, "runtimeOverviewGuardrailAlert")
    assert guardrail_alert is not None and guardrail_alert.property("visible") is True
    latency_label = cycle_group.findChild(QObject, "runtimeOverviewCycleLatency")
    assert latency_label is not None
    latency_text = latency_label.property("text")
    assert "2450" in latency_text

    runtime_service.push_longpoll_metrics(
        [
            {
                "labels": {"adapter": "binance", "scope": "spot", "environment": "paper"},
                "requestLatency": {"p50": 0.150, "p95": 0.480},
                "httpErrors": {"total": 2},
                "reconnects": {"attempts": 3, "failure": 1},
            }
        ]
    )
    app.processEvents()

    longpoll_entries = root.findChildren(QObject, "runtimeOverviewLongPollEntry")
    assert longpoll_entries, "Brak widocznych wpisów long-pollowych"
    header = longpoll_entries[0].findChild(QObject, "runtimeOverviewLongPollHeader")
    assert header is not None and "binance" in header.property("text")
    latency_label = longpoll_entries[0].findChild(QObject, "runtimeOverviewLongPollLatency")
    assert latency_label is not None
    assert "0.480" in latency_label.property("text")
    reconnect_label = longpoll_entries[0].findChild(QObject, "runtimeOverviewLongPollReconnects")
    assert reconnect_label is not None
    assert "próby 3" in reconnect_label.property("text")

    engine.deleteLater()
    app.quit()


def test_telemetry_provider_reports_errors() -> None:
    calls: list[int] = []

    def _failing_loader() -> RuntimeTelemetrySnapshot:
        calls.append(1)
        raise RuntimeError("registry unreachable")

    provider = TelemetryProvider(snapshot_loader=_failing_loader)
    result = provider.refreshTelemetry()
    assert result is False
    assert provider.errorMessage == "registry unreachable"
    assert len(calls) == 1


def test_runtime_service_attaches_to_live_decision_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_file = tmp_path / "audit" / "decision_logs" / "live_execution.jsonl"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps(_sample_decisions()[0]) + "\n", encoding="utf-8")

    config_path = tmp_path / "config" / "core.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    dummy_config = object()

    monkeypatch.setenv("BOT_CORE_UI_CORE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(runtime_service_module, "load_core_config", lambda path: dummy_config)
    monkeypatch.setattr(runtime_service_module, "resolve_decision_log_config", lambda _cfg: (log_file, {}))

    service = RuntimeService()
    assert service.attachToLiveDecisionLog("alpha") is True

    decisions = service.loadRecentDecisions(5)
    assert len(decisions) == 1
    assert decisions[0]["portfolio"] == "alpha"
    assert service.errorMessage == ""
    assert Path(service.activeDecisionLogPath) == log_file


def test_runtime_service_attach_reports_missing_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config" / "core.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("{}", encoding="utf-8")

    dummy_config = object()
    missing = tmp_path / "audit" / "decision_logs" / "missing.jsonl"

    monkeypatch.setenv("BOT_CORE_UI_CORE_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(runtime_service_module, "load_core_config", lambda path: dummy_config)
    monkeypatch.setattr(runtime_service_module, "resolve_decision_log_config", lambda _cfg: (missing, {}))

    service = RuntimeService()
    assert service.attachToLiveDecisionLog("stage6") is False
    assert "nie istnieje" in service.errorMessage.lower()


def test_runtime_service_feed_health_exports_alerts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_WARNING_MS", "1.0")
    monkeypatch.setenv("BOT_CORE_UI_FEED_LATENCY_P95_CRITICAL_MS", "2.0")

    events: list[str] = []

    class _Sink:
        def emit_feed_health_event(self, **payload: object) -> None:
            events.append(str(payload.get("severity")))

    exporter = FeedHealthMetricsExporter(registry=MetricsRegistry())
    sink = _Sink()
    service = RuntimeService(feed_alert_sink=sink, feed_metrics_exporter=exporter)

    samples = service._latency_samples_for("grpc")
    samples.clear()
    samples.append(5.0)
    service._update_feed_health(status="connected", reconnects=0, last_error="")
    critical_report = service.feedSlaReport
    assert critical_report["latency_state"] == "critical"
    assert critical_report["sla_state"] == "critical"

    samples.clear()
    samples.append(0.2)
    service._update_feed_health(status="connected", reconnects=0, last_error="")

    assert events[:2] == ["critical", "info"]
    recovery_report = service.feedSlaReport
    assert recovery_report["latency_state"] == "ok"
    assert recovery_report["sla_state"] == "ok"
    assert service.feedAlertHistory
    assert service.feedAlertHistory[0]["metric"] == "latency"

    feed_snapshot = service.feedHealth
    assert "p95LatencyMs" in feed_snapshot
    assert feed_snapshot["p95LatencyMs"] >= 0.0

    dashboard = exporter.dashboard()
    assert dashboard
    demo_entry = next(entry for entry in dashboard if entry["adapter"] == "demo")
    assert demo_entry["latency_p95_ms"] is not None
    assert demo_entry["status"] == "connected"

    registry = exporter._registry  # type: ignore[attr-defined]
    sla_latency_gauge = registry.get("bot_ui_feed_sla_latency_p95_ms")
    assert sla_latency_gauge.value(labels={"adapter": "demo", "transport": "grpc"}) > 0.0
    sla_reconnects = registry.get("bot_ui_feed_sla_reconnects_total")
    assert sla_reconnects.value(labels={"adapter": "demo", "transport": "grpc"}) >= 0.0


def test_runtime_service_records_escalation_channels(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOT_CORE_UI_FEED_RECONNECT_WARNING", "1")
    monkeypatch.setenv("BOT_CORE_UI_FEED_DOWNTIME_WARNING_SECONDS", "1.0")

    from bot_core.alerts import DefaultAlertRouter, InMemoryAlertAuditLog
    from bot_core.alerts.base import AlertChannel
    from bot_core.runtime.metrics_alerts import UiTelemetryAlertSink

    class _Channel(AlertChannel):
        def __init__(self, name: str) -> None:
            self.name = name
            self.messages: list[object] = []

        def send(self, message) -> None:  # pragma: no cover - prosta implementacja
            self.messages.append(message)

        def health_check(self) -> dict[str, str]:  # pragma: no cover - prosta implementacja
            return {"status": "ok"}

    router = DefaultAlertRouter(audit_log=InMemoryAlertAuditLog())
    router.register(_Channel("cloud-escalation"))
    sink = UiTelemetryAlertSink(router)
    service = RuntimeService(feed_alert_sink=sink)

    samples = service._latency_samples_for("grpc")
    samples.clear()
    samples.append(6000.0)
    service._feed_downtime_total = 2.0
    service._update_feed_health(status="connected", reconnects=4, last_error="timeout")

    history = service.feedAlertHistory
    assert history
    assert any(entry.get("metric") == "reconnects" for entry in history)
    assert any(entry.get("metric") == "downtime" for entry in history)

    channels = service.feedAlertChannels
    assert channels
    assert channels[0]["name"] == "cloud-escalation"


def test_risk_journal_metrics_exporter_records_state(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[dict[str, object]] = []

    class _Sink:
        def emit_feed_health_event(self, **kwargs: object) -> None:  # pragma: no cover - prosta implementacja
            captured.append(kwargs)

    service = RuntimeService(feed_alert_sink=_Sink())

    diagnostics = {"incompleteEntries": 2, "incompleteSamples": ["a", "b"]}
    service._maybe_emit_risk_journal_alert(diagnostics)

    exporter = service._risk_journal_metrics_exporter
    registry = exporter._registry  # type: ignore[attr-defined]
    state_metric = registry.get("bot_ui_risk_journal_state")
    assert state_metric.value(labels={"channel": "risk_journal", "environment": "default"}) == 1.0
    assert (
        registry.get("bot_ui_risk_journal_incomplete_entries_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 2.0
    )
    assert (
        registry.get("bot_ui_risk_journal_incomplete_samples_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 2.0
    )
    assert captured, "Brak zdarzenia alertu telemetrii"
    event = captured[-1]
    payload = event.get("payload") if isinstance(event, Mapping) else {}
    assert payload.get("environment") == "default"
    assert payload.get("incompleteEntries") == 2
    assert payload.get("incomplete_entries") == 2
    assert payload.get("incompleteSamples") == 2
    assert payload.get("incomplete_samples") == 2
    assert payload.get("riskFlagCounts") == {}


def test_risk_journal_metrics_exporter_normalizes_snake_case() -> None:
    service = RuntimeService(feed_alert_sink=None)

    diagnostics = {"incomplete_entries": 3, "incomplete_samples": ["x", "y"]}
    service._maybe_emit_risk_journal_alert(diagnostics)

    exporter = service._risk_journal_metrics_exporter
    registry = exporter._registry  # type: ignore[attr-defined]
    assert (
        registry.get("bot_ui_risk_journal_incomplete_entries_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 3.0
    )
    assert (
        registry.get("bot_ui_risk_journal_incomplete_samples_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 2.0
    )


def test_risk_journal_metrics_exporter_tracks_risk_flag_counts() -> None:
    registry = MetricsRegistry()
    service = RuntimeService(feed_alert_sink=None)
    service._risk_journal_metrics_exporter = RiskJournalMetricsExporter(
        registry=registry
    )

    diagnostics = {
        "incompleteEntries": 1,
        "incompleteSamples": [],
        "riskFlagCounts": {"missing_action": 3, "stress_override": 2},
    }
    service._maybe_emit_risk_journal_alert(diagnostics)

    assert (
        registry.get("bot_ui_risk_journal_risk_flag_entries_total").value(
            labels={
                "channel": "risk_journal",
                "environment": "default",
                "riskFlag": "missing_action",
            }
        )
        == 3.0
    )
    assert (
        registry.get("bot_ui_risk_journal_risk_flag_entries_total").value(
            labels={
                "channel": "risk_journal",
                "environment": "default",
                "riskFlag": "stress_override",
            }
        )
        == 2.0
    )


def test_risk_journal_metrics_exporter_accepts_numeric_samples() -> None:
    registry = MetricsRegistry()
    service = RuntimeService(feed_alert_sink=None)
    service._risk_journal_metrics_exporter = RiskJournalMetricsExporter(
        registry=registry
    )

    diagnostics = {"incomplete_entries": 1, "incomplete_samples": 5}
    service._maybe_emit_risk_journal_alert(diagnostics)

    assert (
        registry.get("bot_ui_risk_journal_incomplete_entries_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 1.0
    )
    assert (
        registry.get("bot_ui_risk_journal_incomplete_samples_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 5.0
    )


def test_risk_journal_metrics_exporter_prefers_explicit_sample_count() -> None:
    registry = MetricsRegistry()
    service = RuntimeService(feed_alert_sink=None)
    service._risk_journal_metrics_exporter = RiskJournalMetricsExporter(
        registry=registry
    )

    diagnostics = {
        "incompleteEntries": 4,
        "incompleteSamples": [{"timestamp": "t1", "event": "missing"}],
        "incompleteSamplesCount": 4,
    }
    service._maybe_emit_risk_journal_alert(diagnostics)

    assert (
        registry.get("bot_ui_risk_journal_incomplete_samples_total").value(
            labels={"channel": "risk_journal", "environment": "default"}
        )
        == 4.0
    )


@pytest.mark.timeout(30)
def test_runtime_overview_strategy_ai_panel_tracks_transport() -> None:
    provider = _StubTelemetryProvider()
    runtime_service = _StubRuntimeService()

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service.push_ai_regimes(
        [
            {
                "regime": "trend",
                "bestStrategy": "trend_following",
                "meanReward": 1.2,
                "plays": 5,
            }
        ]
    )
    runtime_service.push_feed_transport(
        {
            "status": "connected",
            "mode": "grpc",
            "label": "grpc://localhost:50051",
            "reconnects": 1,
            "latencyP95": 245.0,
            "lastError": "",
        }
    )
    runtime_service.set_adaptive_summary("trend: trend_following", "trend -> trend_following")
    app.processEvents()

    panel = root.findChild(QObject, "runtimeOverviewStrategyAiPanel")
    assert panel is not None
    transport_label = root.findChild(QObject, "strategyAiTransportLabel")
    assert transport_label is not None
    assert "connected" in transport_label.property("text").lower()

    activation_label = root.findChild(QObject, "strategyAiActivationSummary")
    assert activation_label is not None
    assert activation_label.property("visible") is True

    regime_model = root.property("aiRegimeBreakdown")
    assert isinstance(regime_model, list)
    assert regime_model[0]["bestStrategy"] == "trend_following"

    engine.deleteLater()
    app.quit()


@pytest.mark.timeout(30)
def test_runtime_overview_feed_sla_exposes_anti_flap_counters() -> None:
    provider = TelemetryProvider(snapshot_loader=_sample_snapshot)
    runtime_service = RuntimeService(decision_loader=lambda limit: [])

    app = QApplication.instance() or QApplication([])
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("telemetryProvider", provider)
    engine.rootContext().setContextProperty("runtimeService", runtime_service)
    qml_path = Path(__file__).resolve().parents[2] / "ui" / "qml" / "dashboard" / "RuntimeOverview.qml"
    engine.load(QUrl.fromLocalFile(str(qml_path)))
    assert engine.rootObjects(), "Nie udało się załadować RuntimeOverview.qml"
    root = engine.rootObjects()[0]

    runtime_service._feed_sla_report = {
        "sla_state": "warning",
        "p95_ms": 1250.0,
        "latency_warning_ms": 800.0,
        "latency_critical_ms": 1500.0,
        "consecutive_degraded_periods": 3,
        "consecutive_healthy_periods": 0,
    }
    runtime_service.feedSlaReportChanged.emit()
    app.processEvents()

    report = root.property("feedSlaReport")
    assert isinstance(report, dict)
    assert report.get("consecutive_degraded_periods") == 3
    assert report.get("consecutive_healthy_periods") == 0
    sla_card = root.findChild(QObject, "runtimeOverviewFeedSlaCard")
    assert sla_card is not None
    assert sla_card.property("severity") == "warning"


def test_runtime_overview_reference_screenshot_exists() -> None:
    reference_path = Path(__file__).resolve().parent / "screenshots" / "runtime_overview_reference.json"
    assert reference_path.exists(), "Brak referencyjnego zrzutu RuntimeOverview"

    payload = json.loads(reference_path.read_text(encoding="utf-8"))
    encoded_chunks = payload.get("data", [])
    assert encoded_chunks, "Brak danych bazowych zrzutu"
    raw_bytes = base64.b64decode("".join(encoded_chunks))

    image = QImage()
    assert image.loadFromData(QByteArray(raw_bytes), payload.get("format", "png")), "Nie udało się odczytać zrzutu"
    assert not image.isNull(), "Referencyjny zrzut ekranu jest uszkodzony"
    assert image.width() >= payload.get("width", 32)
    assert image.height() >= payload.get("height", 32)
