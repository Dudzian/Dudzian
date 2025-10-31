"""Kontroler odpowiedzialny za mapowanie alertów guardrail na runbooki."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping

from PySide6.QtCore import QObject, Property, Signal, Slot

from core.reporting.guardrails_reporter import (
    GuardrailLogRecord,
    GuardrailQueueSummary,
    GuardrailReport,
    GuardrailReportEndpoint,
)


def _now_local_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class RunbookController(QObject):
    """Udostępnia QML listę alertów oraz przypisane runbooki."""

    alertsChanged = Signal()
    lastUpdatedChanged = Signal()
    errorMessageChanged = Signal()

    def __init__(
        self,
        *,
        report_endpoint: GuardrailReportEndpoint | None = None,
        runbook_directory: str | Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._endpoint = report_endpoint or GuardrailReportEndpoint()
        self._alerts: list[dict[str, object]] = []
        self._last_updated = ""
        self._error_message = ""
        self._runbooks = self._load_runbooks(runbook_directory)

    # ------------------------------------------------------------------
    @Property("QVariantList", notify=alertsChanged)
    def alerts(self) -> list[dict[str, object]]:  # type: ignore[override]
        return list(self._alerts)

    @Property(str, notify=lastUpdatedChanged)
    def lastUpdated(self) -> str:  # type: ignore[override]
        return self._last_updated

    @Property(str, notify=errorMessageChanged)
    def errorMessage(self) -> str:  # type: ignore[override]
        return self._error_message

    # ------------------------------------------------------------------
    @Slot(result=bool)
    def refreshAlerts(self) -> bool:
        """Aktualizuje listę alertów na podstawie bieżącego raportu guardrail."""

        try:
            report = self._endpoint.build_report()
        except Exception as exc:  # pragma: no cover - diagnostyka środowiska
            self._error_message = str(exc)
            self.errorMessageChanged.emit()
            return False

        self._error_message = ""
        self.errorMessageChanged.emit()

        self._alerts = self._map_alerts(report)
        self.alertsChanged.emit()

        self._last_updated = _now_local_iso()
        self.lastUpdatedChanged.emit()
        return True

    @Slot(str, result=bool)
    def openRunbook(self, path: str) -> bool:
        """Zwraca ``True`` gdy ścieżka do runbooka istnieje."""

        if not path:
            return False
        resolved = Path(path).expanduser()
        return resolved.exists()

    # ------------------------------------------------------------------
    def _map_alerts(self, report: GuardrailReport) -> list[dict[str, object]]:
        alerts: list[dict[str, object]] = []
        used_ids: set[str] = set()

        for summary in report.summaries:
            entry = self._summary_to_alert(summary)
            if entry is None:
                continue
            if entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        for record in report.logs:
            entry = self._log_to_alert(record)
            if entry is None or entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        for index, recommendation in enumerate(report.recommendations):
            entry = self._recommendation_to_alert(recommendation, index)
            if entry is None or entry["id"] in used_ids:
                continue
            used_ids.add(entry["id"])
            alerts.append(entry)

        return alerts

    # ------------------------------------------------------------------
    def _summary_to_alert(self, summary: GuardrailQueueSummary) -> dict[str, object] | None:
        severity = summary.severity()
        if severity == "normal":
            return None
        message = (
            "Kolejka {queue} w środowisku {env} zgłosiła {timeouts:.0f} timeoutów i "
            "{waits:.0f} oczekiwań na limit.".format(
                queue=summary.queue,
                env=summary.environment,
                timeouts=summary.timeout_total,
                waits=summary.rate_limit_wait_total,
            )
        )
        runbook = self._select_runbook(message, severity)
        return {
            "id": f"summary:{summary.environment}:{summary.queue}",
            "source": "summary",
            "severity": severity,
            "message": message,
            "environment": summary.environment,
            "queue": summary.queue,
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "timestamp": "",
        }

    def _log_to_alert(self, record: GuardrailLogRecord) -> dict[str, object] | None:
        severity = "error" if record.level == "ERROR" else "warning"
        message = record.message or record.event
        runbook = self._select_runbook(message, severity)
        return {
            "id": f"log:{record.timestamp.isoformat()}:{record.event}",
            "source": "log",
            "severity": severity,
            "message": message,
            "environment": record.metadata.get("environment", "") if record.metadata else "",
            "queue": record.metadata.get("queue", "") if record.metadata else "",
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "timestamp": record.timestamp.isoformat(),
        }

    def _recommendation_to_alert(self, recommendation: str, index: int) -> dict[str, object] | None:
        message = recommendation.strip()
        if not message:
            return None
        runbook = self._select_runbook(message, "info")
        return {
            "id": f"recommendation:{index}",
            "source": "recommendation",
            "severity": "info",
            "message": message,
            "environment": "",
            "queue": "",
            "runbookTitle": runbook.get("title", ""),
            "runbookPath": runbook.get("path", ""),
            "timestamp": "",
        }

    # ------------------------------------------------------------------
    def _select_runbook(self, text: str, severity: str) -> Mapping[str, str]:
        normalized = text.lower()
        for keywords, identifier in _KEYWORD_MAP:
            if any(keyword in normalized for keyword in keywords):
                return self._runbooks.get(identifier, {})
        if severity == "error":
            return self._runbooks.get("strategy_incident_playbook", {})
        if severity == "warning":
            return self._runbooks.get("autotrade_threshold_calibration", {})
        return next(iter(self._runbooks.values()), {})

    # ------------------------------------------------------------------
    def _load_runbooks(self, directory: str | Path | None) -> MutableMapping[str, Mapping[str, str]]:
        search_dir = Path(directory).expanduser() if directory else Path(__file__).resolve().parents[2] / "docs" / "runbooks" / "operations"
        results: MutableMapping[str, Mapping[str, str]] = {}
        if not search_dir.exists():
            return results
        for path in sorted(search_dir.glob("*.md")):
            title = self._extract_title(path)
            results[path.stem] = {"title": title, "path": str(path)}
        return results

    @staticmethod
    def _extract_title(path: Path) -> str:
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    return stripped.lstrip("# ")
        except OSError:  # pragma: no cover - błędy IO
            return path.stem
        return path.stem.replace("_", " ")


_KEYWORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("timeout", "awarie", "guardrail"), "strategy_incident_playbook"),
    (("limit", "rate", "oczekiw"), "autotrade_threshold_calibration"),
    (("licenc", "fingerprint", "oem"), "oem_license_provisioning"),
)


__all__ = ["RunbookController"]
