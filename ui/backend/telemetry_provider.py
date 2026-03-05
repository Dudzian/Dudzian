"""Dostawca danych telemetrycznych dla panelu RuntimeOverview."""

from __future__ import annotations


from datetime import datetime, timezone
from typing import Callable, Iterable, Mapping

from PySide6.QtCore import QObject, Property, QTimer, Signal, Slot

from core.monitoring.metrics_api import (
    ComplianceTelemetry,
    GuardrailOverview,
    IOQueueTelemetry,
    RetrainingTelemetry,
    RuntimeTelemetrySnapshot,
    load_runtime_snapshot,
)
from ui.backend.qml_bridge import to_plain_dict, to_plain_list, to_plain_text

SnapshotLoader = Callable[[], RuntimeTelemetrySnapshot]


def _format_timestamp(timestamp: datetime | str | None) -> str:
    if timestamp is None:
        return ""
    if isinstance(timestamp, str):
        stripped = timestamp.strip()
        return stripped
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone().isoformat(timespec="seconds")


def _queue_payload(queue: IOQueueTelemetry) -> dict[str, object]:
    return {
        "environment": queue.environment,
        "queue": queue.queue,
        "timeoutTotal": float(queue.timeout_total),
        "timeoutAvgSeconds": queue.timeout_avg_seconds,
        "rateLimitWaitTotal": float(queue.rate_limit_wait_total),
        "rateLimitWaitAvgSeconds": queue.rate_limit_wait_avg_seconds,
        "severity": queue.severity,
    }


def _guardrail_payload(overview: GuardrailOverview) -> dict[str, object]:
    return {
        "totalQueues": int(overview.total_queues),
        "normalQueues": int(overview.normal_queues),
        "infoQueues": int(overview.info_queues),
        "warningQueues": int(overview.warning_queues),
        "errorQueues": int(overview.error_queues),
        "totalTimeouts": float(overview.total_timeouts),
        "totalRateLimitWaits": float(overview.total_rate_limit_waits),
    }


def _retraining_payload(entries: Iterable[RetrainingTelemetry]) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for entry in entries:
        payload.append(
            {
                "status": entry.status,
                "runs": int(entry.runs),
                "averageDurationSeconds": entry.average_duration_seconds,
                "averageDriftScore": entry.average_drift_score,
            }
        )
    return payload


class TelemetryProvider(QObject):
    """Udostępnia QML aktualne metryki runtime."""

    ioQueuesChanged = Signal()
    guardrailSummaryChanged = Signal()
    retrainingChanged = Signal()
    complianceSummaryChanged = Signal()
    lastUpdatedChanged = Signal()
    errorMessageChanged = Signal()

    def __init__(
        self,
        snapshot_loader: SnapshotLoader | None = None,
        parent: QObject | None = None,
        *,
        auto_refresh_interval_ms: int = 0,
    ) -> None:
        super().__init__(parent)
        self._snapshot_loader = snapshot_loader or load_runtime_snapshot
        self._io_queues: list[dict[str, object]] = []
        self._guardrail_summary: dict[str, object] = _guardrail_payload(
            GuardrailOverview(0, 0, 0, 0, 0, 0.0, 0.0)
        )
        self._retraining: list[dict[str, object]] = []
        self._compliance_summary: dict[str, object] = self._default_compliance_payload()
        self._last_updated: str = ""
        self._error_message: str = ""
        self._timer: QTimer | None = None
        if auto_refresh_interval_ms > 0:
            timer = QTimer(self)
            timer.setInterval(max(500, auto_refresh_interval_ms))
            timer.setSingleShot(False)
            timer.timeout.connect(self.refreshTelemetry)
            timer.start()
            self._timer = timer

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=ioQueuesChanged)
    def ioQueues(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._io_queues)

    @Property("QVariantMap", notify=guardrailSummaryChanged)
    def guardrailSummary(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._guardrail_summary)

    @Property("QVariantList", notify=retrainingChanged)
    def retraining(self) -> list[dict[str, object]]:  # type: ignore[override]
        return to_plain_list(self._retraining)

    @Property("QVariantMap", notify=complianceSummaryChanged)
    def complianceSummary(self) -> dict[str, object]:  # type: ignore[override]
        return to_plain_dict(self._compliance_summary)

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return to_plain_text(self._last_updated)

    @lastUpdated.setter
    def lastUpdated(self, value: datetime | str | None) -> None:
        formatted = _format_timestamp(value)
        if formatted != self._last_updated:
            self._last_updated = formatted
            self.lastUpdatedChanged.emit()

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return to_plain_text(self._error_message)

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshTelemetry(self) -> bool:
        """Pobiera aktualne metryki z rejestru i aktualizuje stan QML."""

        try:
            snapshot = self._snapshot_loader()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiska
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return False

        self._error_message = ""
        self.errorMessageChanged.emit()

        self._io_queues = [_queue_payload(item) for item in snapshot.io_queues]
        self.ioQueuesChanged.emit()

        self._guardrail_summary = _guardrail_payload(snapshot.guardrail_overview)
        self.guardrailSummaryChanged.emit()

        self._retraining = _retraining_payload(snapshot.retraining)
        self.retrainingChanged.emit()

        self._compliance_summary = self._compliance_payload(snapshot.compliance)
        self.complianceSummaryChanged.emit()

        self.lastUpdated = snapshot.generated_at
        return True

    @Slot("QVariantMap")
    def updateComplianceSummary(self, payload: Mapping[str, object] | None = None) -> None:
        """Umożliwia kontrolerom UI aktualizację danych zgodności."""

        summary = to_plain_dict(payload)
        self._compliance_summary = summary or self._default_compliance_payload()
        self.complianceSummaryChanged.emit()

    def _default_compliance_payload(self) -> dict[str, object]:
        return {
            "totalViolations": 0.0,
            "bySeverity": {},
            "byRule": {},
        }

    def _compliance_payload(self, telemetry: ComplianceTelemetry) -> dict[str, object]:
        return telemetry.to_dict()


__all__ = ["TelemetryProvider"]
