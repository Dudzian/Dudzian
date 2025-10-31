"""Generowanie raportów guardrail'i kolejki I/O."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

ISO_FORMAT = "%Y%m%dT%H%M%S"
_PROM_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][\w:]*)" r"(?P<labels>\{[^}]*\})?" r"\s+" r"(?P<value>-?[0-9.eE+-]+)\s*$"
)
_LABEL_RE = re.compile(r"(?P<key>[a-zA-Z_][\w]*)\s*=\s*\"(?P<value>(?:\\.|[^\\\"])*)\"")


@dataclass(slots=True)
class GuardrailQueueSummary:
    """Zsyntetyzowane metryki guardrail'i dla pojedynczej kolejki."""

    environment: str
    queue: str
    rate_limit_wait_total: float = 0.0
    rate_limit_wait_avg_seconds: float | None = None
    timeout_total: float = 0.0
    timeout_avg_seconds: float | None = None

    def severity(self) -> str:
        if self.timeout_total > 0:
            return "error"
        if (self.rate_limit_wait_avg_seconds or 0.0) >= 1.0:
            return "warning"
        if self.rate_limit_wait_total > 0:
            return "info"
        return "normal"


@dataclass(slots=True)
class GuardrailLogRecord:
    """Pojedynczy wpis z pliku ``events.log`` guardrail'i."""

    timestamp: datetime
    level: str
    message: str
    event: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GuardrailReport:
    """Raport zbiorczy guardrail'i kolejki I/O."""

    generated_at: datetime
    summaries: Sequence[GuardrailQueueSummary]
    logs: Sequence[GuardrailLogRecord]
    recommendations: Sequence[str]

    def to_markdown(self) -> str:
        timestamp = self.generated_at.astimezone(timezone.utc).isoformat()
        lines = [
            "# Raport guardrail'i kolejki I/O",
            "",
            f"Wygenerowano: {timestamp}",
            "",
        ]
        lines.extend(self._summaries_section())
        lines.extend(self._logs_section())
        lines.extend(self._recommendations_section())
        return "\n".join(lines).strip() + "\n"

    def _summaries_section(self) -> list[str]:
        if not self.summaries:
            return ["## Podsumowanie metryk", "", "Brak zarejestrowanych zdarzeń guardrail.", ""]
        rows = [
            "## Podsumowanie metryk",
            "",
            "| Środowisko | Kolejka | Łączna liczba timeoutów | Średni czas timeoutu [s] | Łączna liczba oczekiwań | Średni czas oczekiwania [s] | Poziom |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for summary in self.summaries:
            timeout_avg = (
                f"{summary.timeout_avg_seconds:.3f}" if summary.timeout_avg_seconds is not None else "n/d"
            )
            wait_avg = (
                f"{summary.rate_limit_wait_avg_seconds:.3f}"
                if summary.rate_limit_wait_avg_seconds is not None
                else "n/d"
            )
            rows.append(
                "| {env} | {queue} | {timeouts:.0f} | {timeout_avg} | {waits:.0f} | {wait_avg} | {severity} |".format(
                    env=summary.environment,
                    queue=summary.queue,
                    timeouts=summary.timeout_total,
                    timeout_avg=timeout_avg,
                    waits=summary.rate_limit_wait_total,
                    wait_avg=wait_avg,
                    severity=summary.severity(),
                )
            )
        rows.append("")
        return rows

    def _logs_section(self) -> list[str]:
        rows = ["## Ostatnie zdarzenia guardrail", ""]
        if not self.logs:
            rows.append("Brak ostrzeżeń w pliku events.log.")
            rows.append("")
            return rows
        for record in self.logs:
            metadata = ", ".join(f"{key}={value}" for key, value in record.metadata.items())
            rows.append(
                f"* {record.timestamp.astimezone(timezone.utc).isoformat()} [{record.level}] {record.event}: {metadata or record.message}"
            )
        rows.append("")
        return rows

    def _recommendations_section(self) -> list[str]:
        rows = ["## Rekomendacje", ""]
        if not self.recommendations:
            rows.append("Brak dodatkowych rekomendacji – konfiguracja guardrail'i wygląda stabilnie.")
        else:
            for item in self.recommendations:
                rows.append(f"- {item}")
        rows.append("")
        return rows

    def write_markdown(self, directory: str | Path) -> Path:
        destination = Path(directory).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        filename = f"guardrails_{self.generated_at.strftime(ISO_FORMAT)}.md"
        path = destination / filename
        path.write_text(self.to_markdown(), encoding="utf-8")
        return path

    def to_dict(self) -> Mapping[str, object]:
        return {
            "generated_at": self.generated_at.astimezone(timezone.utc).isoformat(),
            "summaries": [
                {
                    "environment": summary.environment,
                    "queue": summary.queue,
                    "rate_limit_wait_total": summary.rate_limit_wait_total,
                    "rate_limit_wait_avg_seconds": summary.rate_limit_wait_avg_seconds,
                    "timeout_total": summary.timeout_total,
                    "timeout_avg_seconds": summary.timeout_avg_seconds,
                    "severity": summary.severity(),
                }
                for summary in self.summaries
            ],
            "recommendations": list(self.recommendations),
            "log_records": [
                {
                    "timestamp": record.timestamp.astimezone(timezone.utc).isoformat(),
                    "level": record.level,
                    "event": record.event,
                    "message": record.message,
                    "metadata": dict(record.metadata),
                }
                for record in self.logs
            ],
        }

    @classmethod
    def from_sources(
        cls,
        *,
        registry: MetricsRegistry | None = None,
        log_directory: str | Path | None = None,
        environment_hint: str | None = None,
        max_log_records: int = 20,
    ) -> "GuardrailReport":
        registry = registry or get_global_metrics_registry()
        summaries = _collect_queue_summaries(registry, environment_hint=environment_hint)
        logs = tuple(
            _read_log_records(log_directory, limit=max_log_records) if log_directory else ()
        )
        recommendations = _build_recommendations(summaries, logs)
        return cls(
            generated_at=datetime.now(timezone.utc),
            summaries=summaries,
            logs=logs,
            recommendations=recommendations,
        )


def _collect_queue_summaries(
    registry: MetricsRegistry, *, environment_hint: str | None = None
) -> tuple[GuardrailQueueSummary, ...]:
    raw = registry.render_prometheus()
    aggregates: MutableMapping[tuple[str, str], dict[str, float]] = {}
    for name, labels, value in _iter_metric_samples(raw):
        if name not in {
            "exchange_io_timeout_total",
            "exchange_io_timeout_duration_seconds_sum",
            "exchange_io_timeout_duration_seconds_count",
            "exchange_io_rate_limit_wait_total",
            "exchange_io_rate_limit_wait_seconds_sum",
            "exchange_io_rate_limit_wait_seconds_count",
        }:
            continue
        environment = labels.get("environment", "unknown")
        queue = labels.get("queue", labels.get("strategy", "default"))
        if environment_hint and environment.lower() != environment_hint.lower():
            # Pozwalamy raportowi obejmować wiele środowisk tylko, gdy brak wskazówki.
            continue
        bucket = aggregates.setdefault((environment, queue), {})
        bucket[name] = bucket.get(name, 0.0) + value
    summaries: list[GuardrailQueueSummary] = []
    for (environment, queue), metrics in sorted(aggregates.items()):
        timeout_total = metrics.get("exchange_io_timeout_total", 0.0)
        timeout_avg = _safe_ratio(
            metrics.get("exchange_io_timeout_duration_seconds_sum"),
            metrics.get("exchange_io_timeout_duration_seconds_count"),
        )
        wait_total = metrics.get("exchange_io_rate_limit_wait_total", 0.0)
        wait_avg = _safe_ratio(
            metrics.get("exchange_io_rate_limit_wait_seconds_sum"),
            metrics.get("exchange_io_rate_limit_wait_seconds_count"),
        )
        summaries.append(
            GuardrailQueueSummary(
                environment=environment,
                queue=queue,
                rate_limit_wait_total=wait_total,
                rate_limit_wait_avg_seconds=wait_avg,
                timeout_total=timeout_total,
                timeout_avg_seconds=timeout_avg,
            )
        )
    return tuple(summaries)


def _iter_metric_samples(text: str) -> Iterable[tuple[str, Mapping[str, str], float]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels = _parse_labels(match.group("labels"))
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue
        yield name, labels, value


def _parse_labels(segment: str | None) -> Mapping[str, str]:
    if not segment:
        return {}
    content = segment.strip()[1:-1]
    labels: dict[str, str] = {}
    for match in _LABEL_RE.finditer(content):
        value = match.group("value").encode("utf-8").decode("unicode_escape")
        labels[match.group("key")] = value
    return labels


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return None
    return numerator / denominator


def _read_log_records(directory: str | Path | None, *, limit: int = 20) -> tuple[GuardrailLogRecord, ...]:
    path = Path(directory).expanduser() / "events.log"
    if not path.exists():
        return ()
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = lines[-limit:] if limit > 0 else lines
    records: list[GuardrailLogRecord] = []
    for line in selected:
        record = _parse_log_line(line)
        if record is not None:
            records.append(record)
    return tuple(records)


def _parse_log_line(line: str) -> GuardrailLogRecord | None:
    stripped = line.strip()
    if not stripped:
        return None
    parts = stripped.split(" ", 2)
    if len(parts) < 3:
        return None
    timestamp_text, level, message = parts
    try:
        timestamp = datetime.strptime(timestamp_text, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        return None
    tokens = message.split()
    if not tokens:
        event = "unknown"
    else:
        event = tokens[0]
    metadata: dict[str, object] = {}
    for token in tokens[1:]:
        if "=" not in token:
            continue
        key, raw_value = token.split("=", 1)
        if not key:
            continue
        value = raw_value.rstrip(",")
        if value.endswith("s"):
            value = value[:-1]
        value = value.strip()
        try:
            if "." in value:
                metadata[key] = float(value)
            else:
                metadata[key] = int(value)
        except ValueError:
            metadata[key] = value
    return GuardrailLogRecord(
        timestamp=timestamp.astimezone(timezone.utc),
        level=level.upper(),
        message=message,
        event=event,
        metadata=metadata,
    )


def _build_recommendations(
    summaries: Sequence[GuardrailQueueSummary],
    logs: Sequence[GuardrailLogRecord],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    for summary in summaries:
        if summary.timeout_total > 0:
            recommendations.append(
                "Zwiększ limity czasowe lub zbadaj stabilność połączenia dla kolejki "
                f"{summary.queue} (środowisko {summary.environment})."
            )
        if summary.rate_limit_wait_avg_seconds and summary.rate_limit_wait_avg_seconds >= 1.0:
            recommendations.append(
                "Rozważ zwiększenie współbieżności lub obniżenie burst dla kolejki "
                f"{summary.queue}, ponieważ średni czas oczekiwania wynosi "
                f"{summary.rate_limit_wait_avg_seconds:.2f} s."
            )
    if any(record.level == "ERROR" for record in logs):
        recommendations.append(
            "Sprawdź wpisy błędów w logs/guardrails/events.log – wystąpiły błędy guardrail'i."
        )
    if not recommendations:
        if summaries:
            recommendations.append("Guardrails nie sygnalizują problemów – utrzymuj obecną konfigurację.")
        else:
            recommendations.append(
                "Brak zdarzeń guardrail – potwierdź, że kolejka I/O jest poprawnie zainicjalizowana."
            )
    return tuple(dict.fromkeys(recommendations))


__all__ = [
    "GuardrailReport",
    "GuardrailQueueSummary",
    "GuardrailLogRecord",
]
