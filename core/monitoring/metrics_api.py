"""Pomocnicze API do ekstrakcji metryk runtime na potrzeby UI."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry

_PROM_LINE_PREFIX = ("#",)


@dataclass(slots=True)
class IOQueueTelemetry:
    """Zsyntetyzowane metryki dla pojedynczej kolejki I/O."""

    environment: str
    queue: str
    timeout_total: float
    timeout_avg_seconds: float | None
    rate_limit_wait_total: float
    rate_limit_wait_avg_seconds: float | None
    severity: str


@dataclass(slots=True)
class GuardrailOverview:
    """Zestawienie zbiorcze poziomów guardrail'i."""

    total_queues: int
    normal_queues: int
    info_queues: int
    warning_queues: int
    error_queues: int
    total_timeouts: float
    total_rate_limit_waits: float


@dataclass(slots=True)
class RetrainingTelemetry:
    """Metryki cykli retrainingu pogrupowane według statusów."""

    status: str
    runs: int
    average_duration_seconds: float | None
    average_drift_score: float | None


@dataclass(slots=True)
class RuntimeTelemetrySnapshot:
    """Kompletny zrzut danych telemetrycznych dla panelu runtime."""

    generated_at: datetime
    io_queues: tuple[IOQueueTelemetry, ...]
    guardrail_overview: GuardrailOverview
    retraining: tuple[RetrainingTelemetry, ...]


def load_runtime_snapshot(
    *, registry: MetricsRegistry | None = None
) -> RuntimeTelemetrySnapshot:
    """Buduje zrzut telemetryczny z rejestru metryk."""

    registry = registry or get_global_metrics_registry()
    metrics_text = registry.render_prometheus()
    queue_rows = _collect_queue_metrics(metrics_text)
    guardrail = _summarize_guardrails(queue_rows)
    retraining_rows = _collect_retraining_metrics(metrics_text)
    return RuntimeTelemetrySnapshot(
        generated_at=datetime.now(timezone.utc),
        io_queues=queue_rows,
        guardrail_overview=guardrail,
        retraining=retraining_rows,
    )


def _collect_queue_metrics(text: str) -> tuple[IOQueueTelemetry, ...]:
    buckets: dict[tuple[str, str], dict[str, float]] = {}
    for name, labels, value in _iter_metric_samples(text):
        if not name.startswith("exchange_io_"):
            continue
        environment = labels.get("environment", "unknown")
        queue = labels.get("queue", "default")
        entry = buckets.setdefault((environment, queue), {})
        entry[name] = value

    results: list[IOQueueTelemetry] = []
    for (environment, queue), data in sorted(buckets.items()):
        timeout_total = data.get("exchange_io_timeout_total", 0.0)
        timeout_avg = _safe_ratio(
            data.get("exchange_io_timeout_duration_seconds_sum"),
            data.get("exchange_io_timeout_duration_seconds_count"),
        )
        rate_limit_total = data.get("exchange_io_rate_limit_wait_total", 0.0)
        rate_limit_avg = _safe_ratio(
            data.get("exchange_io_rate_limit_wait_seconds_sum"),
            data.get("exchange_io_rate_limit_wait_seconds_count"),
        )
        severity = _determine_severity(
            timeout_total=timeout_total,
            rate_limit_avg=rate_limit_avg,
            rate_limit_total=rate_limit_total,
        )
        results.append(
            IOQueueTelemetry(
                environment=environment,
                queue=queue,
                timeout_total=timeout_total,
                timeout_avg_seconds=timeout_avg,
                rate_limit_wait_total=rate_limit_total,
                rate_limit_wait_avg_seconds=rate_limit_avg,
                severity=severity,
            )
        )
    return tuple(results)


def _summarize_guardrails(entries: Iterable[IOQueueTelemetry]) -> GuardrailOverview:
    normal = info = warning = error = 0
    total_timeouts = 0.0
    total_waits = 0.0
    for entry in entries:
        total_timeouts += entry.timeout_total
        total_waits += entry.rate_limit_wait_total
        if entry.severity == "error":
            error += 1
        elif entry.severity == "warning":
            warning += 1
        elif entry.severity == "info":
            info += 1
        else:
            normal += 1
    total = normal + info + warning + error
    return GuardrailOverview(
        total_queues=total,
        normal_queues=normal,
        info_queues=info,
        warning_queues=warning,
        error_queues=error,
        total_timeouts=total_timeouts,
        total_rate_limit_waits=total_waits,
    )


def _collect_retraining_metrics(text: str) -> tuple[RetrainingTelemetry, ...]:
    aggregates: dict[str, dict[str, float]] = {}
    for name, labels, value in _iter_metric_samples(text):
        if not name.startswith("retraining_"):
            continue
        status = labels.get("status", "unknown")
        entry = aggregates.setdefault(status, {})
        entry[name] = value

    rows: list[RetrainingTelemetry] = []
    for status, data in sorted(aggregates.items()):
        runs = int(data.get("retraining_duration_seconds_count", 0.0))
        avg_duration = _safe_ratio(
            data.get("retraining_duration_seconds_sum"),
            data.get("retraining_duration_seconds_count"),
        )
        avg_drift = _safe_ratio(
            data.get("retraining_drift_score_sum"),
            data.get("retraining_drift_score_count"),
        )
        rows.append(
            RetrainingTelemetry(
                status=status,
                runs=runs,
                average_duration_seconds=avg_duration,
                average_drift_score=avg_drift,
            )
        )
    return tuple(rows)


def _iter_metric_samples(text: str) -> Iterable[tuple[str, Mapping[str, str], float]]:
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(_PROM_LINE_PREFIX):
            continue
        name, labels_text, value_text = _split_metric_line(line)
        if name is None:
            continue
        labels = _parse_labels(labels_text)
        try:
            value = float(value_text)
        except (TypeError, ValueError):
            continue
        yield name, labels, value


def _split_metric_line(line: str) -> tuple[str | None, str | None, str | None]:
    if "{" in line:
        name_part, rest = line.split("{", 1)
        labels_part, value_part = rest.split("}", 1)
        return name_part.strip(), labels_part.strip(), value_part.strip()
    parts = line.split()
    if len(parts) < 2:
        return None, None, None
    return parts[0].strip(), None, parts[1].strip()


def _parse_labels(segment: str | None) -> Mapping[str, str]:
    if not segment:
        return {}
    labels: dict[str, str] = {}
    cursor = 0
    while cursor < len(segment):
        eq_index = segment.find("=", cursor)
        if eq_index == -1:
            break
        key = segment[cursor:eq_index].strip()
        if not key:
            break
        if eq_index + 1 >= len(segment) or segment[eq_index + 1] != '"':
            break
        cursor = eq_index + 2
        value_chars: list[str] = []
        while cursor < len(segment):
            char = segment[cursor]
            if char == '"' and (cursor == eq_index + 2 or segment[cursor - 1] != "\\"):
                cursor += 1
                break
            value_chars.append(char)
            cursor += 1
        labels[key] = bytes("".join(value_chars), "utf-8").decode("unicode_escape")
        if cursor < len(segment) and segment[cursor] == ',':
            cursor += 1
    return labels


def _safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _determine_severity(
    *, timeout_total: float, rate_limit_avg: float | None, rate_limit_total: float
) -> str:
    if timeout_total > 0:
        return "error"
    if rate_limit_avg is not None and rate_limit_avg >= 1.0:
        return "warning"
    if rate_limit_total > 0:
        return "info"
    return "normal"


__all__ = [
    "GuardrailOverview",
    "IOQueueTelemetry",
    "RetrainingTelemetry",
    "RuntimeTelemetrySnapshot",
    "load_runtime_snapshot",
]
