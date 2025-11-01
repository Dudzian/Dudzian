"""Metryki obserwowalności dla guardrail'i kolejki I/O, retrainingu i onboardingu."""
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


@dataclass(slots=True)
class RetrainingMetricSet:
    """Metryki obserwowalności dla cykli retrainingu."""

    registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    _duration_seconds: HistogramMetric = field(init=False, repr=False)
    _drift_score: HistogramMetric = field(init=False, repr=False)
    duration_buckets: Sequence[float] = (10.0, 30.0, 60.0, 120.0, 300.0, 900.0)
    drift_buckets: Sequence[float] = (0.01, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5)

    def __post_init__(self) -> None:
        self._duration_seconds = self.registry.histogram(
            "retraining_duration_seconds",
            "Rozkład czasu trwania pojedynczego cyklu retrainingu.",
            buckets=tuple(self.duration_buckets),
        )
        self._drift_score = self.registry.histogram(
            "retraining_drift_score",
            "Rozkład wartości metryki dryfu danych podczas retrainingu.",
            buckets=tuple(self.drift_buckets),
        )

    @property
    def duration_seconds(self) -> HistogramMetric:
        return self._duration_seconds

    @property
    def drift_score(self) -> HistogramMetric:
        return self._drift_score


@dataclass(slots=True)
class OnboardingMetricSet:
    """Metryki obserwowalności dla kreatora onboardingowego."""

    registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    _duration_seconds: HistogramMetric = field(init=False, repr=False)
    duration_buckets: Sequence[float] = (5.0, 15.0, 30.0, 60.0, 120.0, 300.0)

    def __post_init__(self) -> None:
        self._duration_seconds = self.registry.histogram(
            "onboarding_duration_seconds",
            "Rozkład czasu przejścia kreatora onboardingowego.",
            buckets=tuple(self.duration_buckets),
        )

    @property
    def duration_seconds(self) -> HistogramMetric:
        return self._duration_seconds


@dataclass(slots=True)
class ComplianceMetricSet:
    """Metryki obserwowalności dla alertów zgodności."""

    registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    _violations_total: CounterMetric = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._violations_total = self.registry.counter(
            "compliance_violation_total",
            "Liczba naruszeń zgodności wykrytych przez audyt.",
        )

    @property
    def violations_total(self) -> CounterMetric:
        return self._violations_total


__all__ = [
    "AsyncIOMetricSet",
    "ComplianceMetricSet",
    "OnboardingMetricSet",
    "RetrainingMetricSet",
]
