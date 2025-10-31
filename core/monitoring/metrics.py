"""Metryki obserwowalności dla guardrail'i kolejki I/O."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from bot_core.observability.metrics import (
    CounterMetric,
    HistogramMetric,
    MetricsRegistry,
    get_global_metrics_registry,
)


@dataclass(slots=True)
class AsyncIOMetricSet:
    """Zestaw metryk monitorujących kolejkę I/O oraz limity adapterów."""

    registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    _timeout_total: CounterMetric = field(init=False, repr=False)
    _timeout_duration: HistogramMetric = field(init=False, repr=False)
    _rate_limit_wait_total: CounterMetric = field(init=False, repr=False)
    _rate_limit_wait_seconds: HistogramMetric = field(init=False, repr=False)
    timeout_buckets: Sequence[float] = (1.0, 2.0, 5.0, 10.0, 30.0)
    wait_buckets: Sequence[float] = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)

    def __post_init__(self) -> None:
        self._timeout_total = self.registry.counter(
            "exchange_io_timeout_total",
            "Łączna liczba timeoutów wykrytych przez guardrails kolejki I/O.",
        )
        self._timeout_duration = self.registry.histogram(
            "exchange_io_timeout_duration_seconds",
            "Rozkład czasu trwania operacji zakończonych timeoutem.",
            buckets=tuple(self.timeout_buckets),
        )
        self._rate_limit_wait_total = self.registry.counter(
            "exchange_io_rate_limit_wait_total",
            "Liczba zdarzeń oczekiwania na limiter kolejki I/O.",
        )
        self._rate_limit_wait_seconds = self.registry.histogram(
            "exchange_io_rate_limit_wait_seconds",
            "Rozkład czasu oczekiwania w kolejce I/O po osiągnięciu limitów.",
            buckets=tuple(self.wait_buckets),
        )

    @property
    def timeout_total(self) -> CounterMetric:
        return self._timeout_total

    @property
    def timeout_duration(self) -> HistogramMetric:
        return self._timeout_duration

    @property
    def rate_limit_wait_total(self) -> CounterMetric:
        return self._rate_limit_wait_total

    @property
    def rate_limit_wait_seconds(self) -> HistogramMetric:
        return self._rate_limit_wait_seconds


__all__ = ["AsyncIOMetricSet"]
