"""Kontrola budżetów zasobów podczas uruchomień sandboxa AI."""

from __future__ import annotations

import importlib
import time
from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Protocol

from bot_core.alerts import AlertSeverity, emit_alert
from bot_core.observability.metrics import MetricsRegistry, get_global_metrics_registry


def _load_optional_module(name: str):  # type: ignore[no-untyped-def]
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    module = importlib.util.module_from_spec(spec)
    loader = spec.loader
    if loader is None:
        return None
    loader.exec_module(module)
    return module


_psutil = _load_optional_module("psutil")


@dataclass(slots=True)
class SandboxResourceSample:
    """Pojedyncza próbka wykorzystania zasobów."""

    cpu_percent: float | None = None
    gpu_percent: float | None = None
    elapsed_seconds: float = 0.0


class ResourceSampler(Protocol):
    """Kontrakt pobierania bieżących metryk zasobów."""

    def __call__(self) -> SandboxResourceSample:
        ...


@dataclass(slots=True)
class SandboxBudgetConfig:
    wall_time_seconds: float | None = None
    cpu_utilization_percent: float | None = None
    gpu_utilization_percent: float | None = None


@dataclass(slots=True)
class SandboxAlertConfig:
    source: str = "ai.sandbox"
    severity: AlertSeverity = AlertSeverity.WARNING
    enabled: bool = True


@dataclass(slots=True)
class SandboxMetricsConfig:
    cpu_counter: tuple[str, str]
    gpu_counter: tuple[str, str]
    wall_time_histogram: tuple[str, str]
    wall_time_buckets: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
    events_counter: tuple[str, str] | None = None
    decisions_counter: tuple[str, str] | None = None
    risk_limit_utilization_gauge: tuple[str, str] | None = None
    risk_limit_breach_counter: tuple[str, str] | None = None


@dataclass(slots=True)
class SandboxCostGuard:
    """Monitoruje zużycie CPU/GPU oraz czas wykonywania scenariusza."""

    budgets: SandboxBudgetConfig
    metrics: SandboxMetricsConfig
    alerts: SandboxAlertConfig
    metrics_registry: MetricsRegistry = field(default_factory=get_global_metrics_registry)
    sampler: ResourceSampler | None = None
    metric_labels: Mapping[str, str] | None = None
    _start_monotonic: float | None = field(init=False, repr=False, default=None)
    _cpu_counter: object = field(init=False, repr=False)
    _gpu_counter: object = field(init=False, repr=False)
    _wall_histogram: object = field(init=False, repr=False)
    _statistics: MutableMapping[str, float] = field(init=False, repr=False, default_factory=dict)
    _alert_emitted: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        self._cpu_counter = self.metrics_registry.counter(
            self.metrics.cpu_counter[0], self.metrics.cpu_counter[1]
        )
        self._gpu_counter = self.metrics_registry.counter(
            self.metrics.gpu_counter[0], self.metrics.gpu_counter[1]
        )
        self._wall_histogram = self.metrics_registry.histogram(
            self.metrics.wall_time_histogram[0],
            self.metrics.wall_time_histogram[1],
            buckets=self.metrics.wall_time_buckets,
        )
        self._statistics = {
            "max_cpu_percent": 0.0,
            "max_gpu_percent": 0.0,
            "wall_time_seconds": 0.0,
        }

    def __enter__(self) -> "SandboxCostGuard":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.finish()
        return None

    def start(self) -> None:
        self._start_monotonic = time.monotonic()
        self._statistics.update({"max_cpu_percent": 0.0, "max_gpu_percent": 0.0, "wall_time_seconds": 0.0})

    def finish(self) -> None:
        elapsed = self._elapsed_seconds()
        if elapsed is not None:
            self._statistics["wall_time_seconds"] = elapsed
            self._wall_histogram.observe(elapsed, labels=self.metric_labels)

    def sample(self) -> SandboxResourceSample:
        if self.sampler is not None:
            sample = self.sampler()
        else:
            sample = self._sample_process_resources()
        elapsed = self._elapsed_seconds()
        if elapsed is not None:
            sample.elapsed_seconds = elapsed
        return sample

    def update(self, *, sample: SandboxResourceSample | None = None, context: Mapping[str, object] | None = None) -> None:
        reading = sample or self.sample()
        labels = self.metric_labels
        if reading.cpu_percent is not None:
            self._statistics["max_cpu_percent"] = max(self._statistics["max_cpu_percent"], float(reading.cpu_percent))
            self._cpu_counter.inc(max(reading.cpu_percent, 0.0), labels=labels)
        if reading.gpu_percent is not None:
            self._statistics["max_gpu_percent"] = max(self._statistics["max_gpu_percent"], float(reading.gpu_percent))
            self._gpu_counter.inc(max(reading.gpu_percent, 0.0), labels=labels)
        self._statistics["wall_time_seconds"] = max(
            self._statistics.get("wall_time_seconds", 0.0), float(reading.elapsed_seconds)
        )
        self._enforce_limits(reading, context=context)

    def statistics(self) -> Mapping[str, float]:
        return dict(self._statistics)

    def _elapsed_seconds(self) -> float | None:
        if self._start_monotonic is None:
            return None
        return max(time.monotonic() - self._start_monotonic, 0.0)

    def _sample_process_resources(self) -> SandboxResourceSample:
        cpu = None
        gpu = None
        if _psutil is not None:
            process = _psutil.Process()  # type: ignore[attr-defined]
            try:
                cpu = float(process.cpu_percent(interval=None))
            except Exception:
                cpu = None
        return SandboxResourceSample(cpu_percent=cpu, gpu_percent=gpu, elapsed_seconds=0.0)

    def _enforce_limits(self, sample: SandboxResourceSample, *, context: Mapping[str, object] | None = None) -> None:
        limit = self.budgets.wall_time_seconds
        if limit is not None and sample.elapsed_seconds > limit:
            self._raise_budget_exceeded(
                "wall_time",
                sample.elapsed_seconds,
                limit,
                context=context,
            )
        limit = self.budgets.cpu_utilization_percent
        if limit is not None and sample.cpu_percent is not None and sample.cpu_percent > limit:
            self._raise_budget_exceeded("cpu", sample.cpu_percent, limit, context=context)
        limit = self.budgets.gpu_utilization_percent
        if limit is not None and sample.gpu_percent is not None and sample.gpu_percent > limit:
            self._raise_budget_exceeded("gpu", sample.gpu_percent, limit, context=context)

    def _raise_budget_exceeded(
        self,
        name: str,
        observed: float,
        limit: float,
        *,
        context: Mapping[str, object] | None = None,
    ) -> None:
        if self.alerts.enabled and not self._alert_emitted:
            payload = {
                "limit": name,
                "observed": f"{observed:.4f}",
                "threshold": f"{limit:.4f}",
            }
            if context:
                payload.update({str(key): str(value) for key, value in context.items()})
            emit_alert(
                f"Sandbox {name} budget exceeded ({observed:.2f} > {limit:.2f})",
                severity=self.alerts.severity,
                source=self.alerts.source,
                context=payload,
            )
            self._alert_emitted = True
        raise SandboxBudgetExceeded(name=name, observed=observed, limit=limit)


@dataclass(slots=True)
class SandboxBudgetExceeded(RuntimeError):
    name: str
    observed: float
    limit: float

    def __str__(self) -> str:
        return f"Sandbox budget {self.name} exceeded: {self.observed:.4f} > {self.limit:.4f}"


__all__ = [
    "SandboxBudgetConfig",
    "SandboxBudgetExceeded",
    "SandboxCostGuard",
    "SandboxMetricsConfig",
    "SandboxResourceSample",
]
