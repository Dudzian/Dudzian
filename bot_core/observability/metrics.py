"""Prosta rejestracja metryk kompatybilna z formatem Prometheusa."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

LabelTuple = Tuple[Tuple[str, str], ...]


def _normalize_labels(labels: Mapping[str, str] | None) -> LabelTuple:
    if not labels:
        return ()
    normalized = tuple(sorted((str(k), str(v)) for k, v in labels.items()))
    return normalized


def _format_labels(labels: LabelTuple) -> str:
    if not labels:
        return ""
    parts = [f'{key}="{value}"' for key, value in labels]
    return "{" + ",".join(parts) + "}"


@dataclass(slots=True)
class Metric:
    """Bazowy typ metryki przechowujący opis i nazwę."""

    name: str
    description: str

    def render(self) -> Iterable[str]:  # pragma: no cover - implementowane w klasach potomnych
        raise NotImplementedError


@dataclass(slots=True)
class CounterMetric(Metric):
    """Licznik tylko rosnący."""

    _values: MutableMapping[LabelTuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        if amount < 0:
            raise ValueError("Liczniki można zwiększać wyłącznie o wartości nieujemne.")
        key = _normalize_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, *, labels: Mapping[str, str] | None = None) -> float:
        key = _normalize_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> Iterable[str]:
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} counter"]
        with self._lock:
            for labels, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_format_labels(labels)} {value}")
        return lines


@dataclass(slots=True)
class GaugeMetric(Metric):
    """Metryka typu gauge pozwalająca ustawiać i zwiększać wartości."""

    _values: MutableMapping[LabelTuple, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def set(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + amount

    def dec(self, amount: float = 1.0, *, labels: Mapping[str, str] | None = None) -> None:
        self.inc(-amount, labels=labels)

    def value(self, *, labels: Mapping[str, str] | None = None) -> float:
        key = _normalize_labels(labels)
        with self._lock:
            return self._values.get(key, 0.0)

    def render(self) -> Iterable[str]:
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} gauge"]
        with self._lock:
            for labels, value in sorted(self._values.items()):
                lines.append(f"{self.name}{_format_labels(labels)} {value}")
        return lines


@dataclass(slots=True)
class HistogramState:
    counts: MutableMapping[float, int] = field(default_factory=dict)
    sum: float = 0.0
    count: int = 0


@dataclass(slots=True)
class HistogramMetric(Metric):
    """Metryka histogramu zgodna z formatem Prometheusa."""

    buckets: Tuple[float, ...]
    _values: MutableMapping[LabelTuple, HistogramState] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        if not self.buckets:
            raise ValueError("Histogram wymaga co najmniej jednego koszyka.")
        ordered = []
        last = -math.inf
        for boundary in self.buckets:
            if boundary <= last:
                raise ValueError("Granice histogramu muszą być rosnące.")
            ordered.append(float(boundary))
            last = boundary
        self.buckets = tuple(ordered)

    def observe(self, value: float, *, labels: Mapping[str, str] | None = None) -> None:
        key = _normalize_labels(labels)
        with self._lock:
            state = self._values.get(key)
            if state is None:
                counts = {boundary: 0 for boundary in self.buckets}
                counts[math.inf] = 0
                state = HistogramState(counts=counts)
                self._values[key] = state
            state.count += 1
            state.sum += value
            for boundary in self.buckets:
                if value <= boundary:
                    state.counts[boundary] += 1
            state.counts[math.inf] += 1

    def snapshot(self, *, labels: Mapping[str, str] | None = None) -> HistogramState:
        key = _normalize_labels(labels)
        with self._lock:
            state = self._values.get(key)
            if state is None:
                counts = {boundary: 0 for boundary in self.buckets}
                counts[math.inf] = 0
                return HistogramState(counts=counts)
            # Tworzymy kopię, by uniknąć modyfikacji przez odbiorcę.
            counts_copy = dict(state.counts)
            return HistogramState(counts=counts_copy, sum=state.sum, count=state.count)

    def render(self) -> Iterable[str]:
        lines = [f"# HELP {self.name} {self.description}", f"# TYPE {self.name} histogram"]
        with self._lock:
            for labels, state in sorted(self._values.items()):
                cumulative = 0
                for boundary in self.buckets:
                    cumulative += state.counts.get(boundary, 0)
                    bucket_labels = dict(labels)
                    bucket_labels["le"] = str(boundary)
                    lines.append(f"{self.name}_bucket{_format_labels(_normalize_labels(bucket_labels))} {cumulative}")
                cumulative += state.counts.get(math.inf, 0)
                inf_labels = dict(labels)
                inf_labels["le"] = "+Inf"
                lines.append(f"{self.name}_bucket{_format_labels(_normalize_labels(inf_labels))} {cumulative}")
                lines.append(f"{self.name}_sum{_format_labels(labels)} {state.sum}")
                lines.append(f"{self.name}_count{_format_labels(labels)} {state.count}")
        return lines


class MetricsRegistry:
    """Rejestr przechowujący metryki i umożliwiający eksport."""

    def __init__(self) -> None:
        self._metrics: Dict[str, Metric] = {}
        self._lock = threading.Lock()

    def counter(self, name: str, description: str) -> CounterMetric:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = CounterMetric(name=name, description=description)
                self._metrics[name] = metric
            elif not isinstance(metric, CounterMetric):
                raise TypeError(f"Metryka {name} istnieje, ale nie jest licznikiem.")
            return metric

    def gauge(self, name: str, description: str) -> GaugeMetric:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = GaugeMetric(name=name, description=description)
                self._metrics[name] = metric
            elif not isinstance(metric, GaugeMetric):
                raise TypeError(f"Metryka {name} istnieje, ale nie jest gauge.")
            return metric

    def histogram(self, name: str, description: str, buckets: Sequence[float]) -> HistogramMetric:
        with self._lock:
            metric = self._metrics.get(name)
            if metric is None:
                metric = HistogramMetric(name=name, description=description, buckets=tuple(buckets))
                self._metrics[name] = metric
            elif not isinstance(metric, HistogramMetric):
                raise TypeError(f"Metryka {name} istnieje, ale nie jest histogramem.")
            return metric

    def render_prometheus(self) -> str:
        lines: list[str] = []
        with self._lock:
            for metric in self._metrics.values():
                lines.extend(metric.render())
        return "\n".join(lines) + "\n"

    def get(self, name: str) -> Metric:
        with self._lock:
            try:
                return self._metrics[name]
            except KeyError as exc:  # pragma: no cover - defensywne dla błędnej konfiguracji
                raise KeyError(f"Metryka {name} nie istnieje w rejestrze") from exc


_GLOBAL_REGISTRY: MetricsRegistry | None = None
_REGISTRY_LOCK = threading.Lock()


def get_global_metrics_registry() -> MetricsRegistry:
    """Zwraca singleton rejestru metryk dla całej aplikacji."""

    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = MetricsRegistry()
    return _GLOBAL_REGISTRY


__all__ = [
    "MetricsRegistry",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "get_global_metrics_registry",
]
