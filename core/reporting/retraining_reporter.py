"""Generowanie raportów z przebiegu cyklu retreningu."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from core.monitoring.events import (
    DataDriftDetected,
    MissingDataDetected,
    MonitoringEvent,
    RetrainingCycleCompleted,
    RetrainingDelayInjected,
)
from core.runtime.retraining_scheduler import RetrainingRunOutcome
from core.ml.training_pipeline import TrainingPipelineResult

ISO_FORMAT = "%Y%m%dT%H%M%S"


@dataclass(slots=True)
class RetrainingEventRow:
    """Znormalizowane zdarzenie monitorujące wykorzystane w raporcie."""

    name: str
    severity: str
    details: Mapping[str, object]


class RetrainingReport:
    """Podsumowanie pojedynczego uruchomienia retreningu."""

    def __init__(
        self,
        *,
        generated_at: datetime,
        started_at: datetime | None,
        finished_at: datetime | None,
        status: str,
        backend: str | None,
        kpi: Mapping[str, object],
        fallback_chain: Sequence[Mapping[str, object]],
        events: Sequence[RetrainingEventRow],
        alerts: Sequence[str],
        errors: Sequence[str],
        dataset_metadata: Mapping[str, object],
    ) -> None:
        self.generated_at = generated_at.astimezone(timezone.utc)
        self.started_at = started_at.astimezone(timezone.utc) if started_at else None
        self.finished_at = finished_at.astimezone(timezone.utc) if finished_at else None
        self.status = str(status)
        self.backend = backend
        self.kpi = dict(kpi)
        self.fallback_chain = tuple(dict(item) for item in fallback_chain)
        self.events = tuple(events)
        self.alerts = tuple(alerts)
        self.errors = tuple(errors)
        self.dataset_metadata = dict(dataset_metadata)

    @classmethod
    def from_execution(
        cls,
        *,
        started_at: datetime | None,
        finished_at: datetime | None,
        outcome: RetrainingRunOutcome,
        training_result: TrainingPipelineResult | None,
        events: Iterable[MonitoringEvent] = (),
        dataset_metadata: Mapping[str, object] | None = None,
    ) -> "RetrainingReport":
        generated_at = datetime.now(timezone.utc)
        backend = training_result.backend if training_result else None
        fallback_chain: Sequence[Mapping[str, object]]
        fallback_chain = training_result.fallback_chain if training_result else ()

        validation_log_path: str | None = None
        if training_result and training_result.validation_log_path:
            validation_log_path = str(training_result.validation_log_path)

        duration_seconds: float | None = None
        if started_at and finished_at:
            duration_seconds = max(0.0, (finished_at - started_at).total_seconds())

        kpi: MutableMapping[str, object] = {
            "status": outcome.status,
            "delay_seconds": outcome.delay_seconds,
            "duration_seconds": duration_seconds,
            "drift_score": outcome.drift_score,
            "backend": backend,
            "fallback_count": len(fallback_chain),
        }

        if validation_log_path:
            kpi["validation_log_path"] = validation_log_path

        meta = dict(dataset_metadata or {})
        if "row_count" in meta:
            kpi["dataset_rows"] = meta.get("row_count")
        if "feature_names" in meta:
            kpi["feature_count"] = len(meta.get("feature_names", ()))

        normalized_events = [_normalize_event(event) for event in events]
        alerts = _derive_alerts(normalized_events, fallback_chain, outcome)

        errors: list[str] = []
        if outcome.status != "completed" and outcome.reason:
            errors.append(f"Retraining zakończony statusem {outcome.status}: {outcome.reason}")

        return cls(
            generated_at=generated_at,
            started_at=started_at,
            finished_at=finished_at,
            status=outcome.status,
            backend=backend,
            kpi=kpi,
            fallback_chain=fallback_chain,
            events=tuple(normalized_events),
            alerts=tuple(alerts),
            errors=tuple(errors),
            dataset_metadata=meta,
        )

    def to_markdown(self) -> str:
        started = self.started_at.isoformat() if self.started_at else "n/d"
        finished = self.finished_at.isoformat() if self.finished_at else "n/d"
        duration = self.kpi.get("duration_seconds")
        duration_text = f"{duration:.2f}" if isinstance(duration, (int, float)) else "n/d"
        drift = self.kpi.get("drift_score")
        drift_text = f"{drift:.4f}" if isinstance(drift, (int, float)) else "n/d"
        delay = self.kpi.get("delay_seconds")
        delay_text = f"{delay:.2f}" if isinstance(delay, (int, float)) else "0.00"

        lines = [
            "# Raport cyklu retreningu",
            "",
            "## Podsumowanie KPI",
            "",
            "| KPI | Wartość |",
            "| --- | --- |",
            f"| Status | {self.status} |",
            f"| Backend | {self.backend or 'n/d'} |",
            f"| Czas rozpoczęcia | {started} |",
            f"| Czas zakończenia | {finished} |",
            f"| Czas trwania [s] | {duration_text} |",
            f"| Liczba fallbacków | {self.kpi.get('fallback_count', 0)} |",
            f"| Opóźnienie [s] | {delay_text} |",
            f"| Dryf danych | {drift_text} |",
        ]
        if "dataset_rows" in self.kpi:
            lines.append(f"| Liczba próbek datasetu | {self.kpi['dataset_rows']} |")
        if "feature_count" in self.kpi:
            lines.append(f"| Liczba cech | {self.kpi['feature_count']} |")
        lines.append("")

        lines.extend(self._fallback_section())
        lines.extend(self._alerts_section())
        lines.extend(self._events_section())
        lines.extend(self._dataset_section())
        return "\n".join(lines).strip() + "\n"

    def _fallback_section(self) -> list[str]:
        lines = ["## Łańcuch fallbacków", ""]
        if not self.fallback_chain:
            lines.append("Brak aktywowanych fallbacków backendów.")
            lines.append("")
            return lines
        lines.append("| Backend | Komunikat | Sugestia instalacji |")
        lines.append("| --- | --- | --- |")
        for entry in self.fallback_chain:
            lines.append(
                "| {backend} | {message} | {hint} |".format(
                    backend=entry.get("backend", "n/d"),
                    message=entry.get("message", ""),
                    hint=entry.get("install_hint", "n/d"),
                )
            )
        lines.append("")
        return lines

    def _alerts_section(self) -> list[str]:
        lines = ["## Alerty i ostrzeżenia", ""]
        if not self.alerts and not self.errors:
            lines.append("Brak alertów związanych z cyklem retreningu.")
            lines.append("")
            return lines
        for item in self.alerts:
            lines.append(f"- ⚠️ {item}")
        for item in self.errors:
            lines.append(f"- ❗ {item}")
        lines.append("")
        return lines

    def _events_section(self) -> list[str]:
        lines = ["## Zarejestrowane zdarzenia", ""]
        if not self.events:
            lines.append("Brak zdarzeń monitorujących.")
            lines.append("")
            return lines
        lines.append("| Zdarzenie | Poziom | Szczegóły |")
        lines.append("| --- | --- | --- |")
        for row in self.events:
            detail_str = ", ".join(f"{key}={value}" for key, value in row.details.items()) or "n/d"
            lines.append(f"| {row.name} | {row.severity} | {detail_str} |")
        lines.append("")
        return lines

    def _dataset_section(self) -> list[str]:
        if not self.dataset_metadata:
            return []
        lines = ["## Metadane datasetu", ""]
        for key, value in sorted(self.dataset_metadata.items()):
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        return lines

    def write_markdown(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"retraining_{self.generated_at.strftime(ISO_FORMAT)}.md"
        path = destination / filename
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def write_json(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"retraining_{self.generated_at.strftime(ISO_FORMAT)}.json"
        path = destination / filename
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "generated_at": self.generated_at.isoformat(),
            "status": self.status,
            "backend": self.backend,
            "kpi": dict(self.kpi),
            "fallback_chain": [dict(item) for item in self.fallback_chain],
            "alerts": list(self.alerts),
            "errors": list(self.errors),
            "dataset_metadata": dict(self.dataset_metadata),
        }
        if self.started_at:
            payload["started_at"] = self.started_at.isoformat()
        if self.finished_at:
            payload["finished_at"] = self.finished_at.isoformat()
        payload["events"] = [
            {"name": row.name, "severity": row.severity, "details": dict(row.details)} for row in self.events
        ]
        return payload


def _normalize_event(event: MonitoringEvent) -> RetrainingEventRow:
    if isinstance(event, MissingDataDetected):
        return RetrainingEventRow(
            name="MissingDataDetected",
            severity="error",
            details={
                "source": event.source,
                "missing_batches": event.missing_batches,
            },
        )
    if isinstance(event, DataDriftDetected):
        return RetrainingEventRow(
            name="DataDriftDetected",
            severity="warning",
            details={
                "source": event.source,
                "drift_score": round(float(event.drift_score), 6),
                "drift_threshold": round(float(event.drift_threshold), 6),
            },
        )
    if isinstance(event, RetrainingDelayInjected):
        return RetrainingEventRow(
            name="RetrainingDelayInjected",
            severity="info",
            details={
                "reason": event.reason,
                "delay_seconds": round(float(event.delay_seconds), 6),
            },
        )
    if isinstance(event, RetrainingCycleCompleted):
        details = {
            "source": event.source,
            "status": event.status,
            "duration_seconds": round(float(event.duration_seconds), 6),
        }
        if event.drift_score is not None:
            details["drift_score"] = round(float(event.drift_score), 6)
        if event.metadata:
            details.update(dict(event.metadata))
        return RetrainingEventRow(
            name="RetrainingCycleCompleted",
            severity="info" if event.status == "completed" else "warning",
            details=details,
        )
    return RetrainingEventRow(
        name=event.__class__.__name__,
        severity="info",
        details={},
    )


def _derive_alerts(
    events: Sequence[RetrainingEventRow],
    fallback_chain: Sequence[Mapping[str, object]],
    outcome: RetrainingRunOutcome,
) -> list[str]:
    alerts: list[str] = []
    for row in events:
        if row.name == "MissingDataDetected":
            alerts.append(
                "Brak danych treningowych (missing_batches={}).".format(row.details.get("missing_batches", "n/d"))
            )
        elif row.name == "DataDriftDetected":
            alerts.append(
                "Wykryto dryf danych (score={score}, threshold={threshold}).".format(
                    score=row.details.get("drift_score", "n/d"),
                    threshold=row.details.get("drift_threshold", "n/d"),
                )
            )
        elif row.name == "RetrainingDelayInjected":
            alerts.append(
                "Wstrzymano retraining przez {delay}s (powód: {reason}).".format(
                    delay=row.details.get("delay_seconds", "n/d"),
                    reason=row.details.get("reason", "unknown"),
                )
            )
    if fallback_chain:
        alerts.append(
            "Aktywowano fallback backendów: {}".format(
                ", ".join(entry.get("backend", "n/d") for entry in fallback_chain)
            )
        )
    if outcome.drift_score is not None and outcome.drift_score > 0:
        alerts.append(f"Dryf danych podczas retrainingu: {outcome.drift_score:.4f}")
    return alerts


__all__ = ["RetrainingReport", "RetrainingEventRow"]
