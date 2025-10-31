"""Raportowanie scenariuszy przejścia demo → paper."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

ISO_FORMAT = "%Y%m%dT%H%M%S"


@dataclass(slots=True)
class StepStatus:
    """Pojedynczy krok scenariusza E2E."""

    name: str
    status: str
    details: Mapping[str, object]


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


class DemoPaperReport:
    """Reprezentuje zsyntetyzowany raport z przebiegu scenariusza demo → paper."""

    def __init__(
        self,
        *,
        mode: str,
        started_at: datetime | None,
        finished_at: datetime | None,
        kpi: Mapping[str, object],
        steps: Sequence[StepStatus],
        warnings: Sequence[str],
        errors: Sequence[str],
        payload: Mapping[str, object],
        decision_events: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self.mode = str(mode)
        self.started_at = started_at
        self.finished_at = finished_at
        self.kpi = dict(kpi)
        self.steps = tuple(steps)
        self.warnings = tuple(warnings)
        self.errors = tuple(errors)
        self.payload = dict(payload)
        self.decision_events = tuple(decision_events)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object],
        *,
        decision_events: Iterable[Mapping[str, object]] = (),
    ) -> "DemoPaperReport":
        mode = str(payload.get("mode", "unknown"))
        started_at = _parse_timestamp(payload.get("started_at"))
        finished_at = _parse_timestamp(payload.get("finished_at"))
        errors = tuple(str(item) for item in payload.get("errors", ()))
        warnings = tuple(str(item) for item in payload.get("warnings", ()))
        events = tuple(dict(event) for event in decision_events)

        duration_seconds: float | None = None
        if started_at and finished_at:
            duration_seconds = max(0.0, (finished_at - started_at).total_seconds())

        kpi = {
            "mode": mode,
            "duration_seconds": duration_seconds,
            "warning_count": len(warnings),
            "error_count": len(errors),
            "decision_events": len(events),
        }

        validation_status = "success"
        if payload.get("status") == "validation_failed":
            validation_status = "failed"
        validation_details = {
            "entrypoint": payload.get("entrypoint"),
            "environment": payload.get("validation", {}).get("environment"),
            "expected_environment": payload.get("validation", {}).get("expected_environment"),
            "symbols": payload.get("validation", {}).get("symbols"),
        }

        checkpoint_required = mode in {"paper", "live"}
        checkpoint_data = payload.get("checkpoint")
        if checkpoint_required and not checkpoint_data:
            checkpoint_status = "failed"
        elif checkpoint_required:
            checkpoint_status = "success"
        else:
            checkpoint_status = "skipped"
        checkpoint_details = {
            "required": checkpoint_required,
            "available": bool(checkpoint_data),
        }

        runtime_status = "success" if payload.get("status") == "success" else "failed"
        runtime_details = {
            "metrics_endpoint": payload.get("metrics_endpoint"),
            "decision_events": len(events),
        }

        steps = (
            StepStatus("Walidacja konfiguracji", validation_status, validation_details),
            StepStatus("Checkpoint demo", checkpoint_status, checkpoint_details),
            StepStatus("Uruchomienie runtime", runtime_status, runtime_details),
        )

        return cls(
            mode=mode,
            started_at=started_at,
            finished_at=finished_at,
            kpi=kpi,
            steps=steps,
            warnings=warnings,
            errors=errors,
            payload=payload,
            decision_events=events,
        )

    def to_markdown(self) -> str:
        started = self.started_at.isoformat() if self.started_at else "n/d"
        finished = self.finished_at.isoformat() if self.finished_at else "n/d"
        duration = self.kpi.get("duration_seconds")
        duration_text = f"{duration:.2f}" if isinstance(duration, (int, float)) else "n/d"

        lines = [
            f"# Raport scenariusza demo → paper ({self.mode})",
            "",
            "## Podsumowanie KPI",
            "",
            "| KPI | Wartość |",
            "| --- | --- |",
            f"| Data rozpoczęcia | {started} |",
            f"| Data zakończenia | {finished} |",
            f"| Czas wykonania [s] | {duration_text} |",
            f"| Liczba ostrzeżeń | {self.kpi.get('warning_count', 'n/d')} |",
            f"| Liczba błędów | {self.kpi.get('error_count', 'n/d')} |",
            f"| Liczba zdarzeń tradingowych | {self.kpi.get('decision_events', 'n/d')} |",
            "",
            "## Status kroków",
            "",
            "| Krok | Status | Szczegóły |",
            "| --- | --- | --- |",
        ]

        for step in self.steps:
            details = ", ".join(f"{key}: {value}" for key, value in step.details.items() if value not in (None, "")) or "brak"
            lines.append(f"| {step.name} | {step.status} | {details} |")

        lines.extend(["", "## Ostrzeżenia", ""])
        if self.warnings:
            for warning in self.warnings:
                lines.append(f"- {warning}")
        else:
            lines.append("Brak ostrzeżeń.")

        lines.extend(["", "## Błędy", ""])
        if self.errors:
            for error in self.errors:
                lines.append(f"- {error}")
        else:
            lines.append("Brak błędów.")

        return "\n".join(lines).strip() + "\n"

    def write_markdown(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        timestamp_source = self.finished_at or self.started_at or datetime.now(timezone.utc)
        filename = f"demo_paper_{self.mode}_{timestamp_source.strftime(ISO_FORMAT)}.md"
        path = destination / filename
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_dict(self) -> Mapping[str, object]:
        return {
            "mode": self.mode,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "kpi": dict(self.kpi),
            "steps": [
                {"name": step.name, "status": step.status, "details": dict(step.details)}
                for step in self.steps
            ],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "decision_events": list(self.decision_events),
        }


__all__ = ["DemoPaperReport", "StepStatus"]
