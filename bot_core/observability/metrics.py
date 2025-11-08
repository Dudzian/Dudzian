"""Prosta rejestracja metryk kompatybilna z formatem Prometheusa."""
from __future__ import annotations

import logging
import math
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

from bot_core.alerts.dispatcher import AlertSeverity, emit_alert

_LOGGER = logging.getLogger(__name__)

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


def _labels_to_dict(labels: LabelTuple) -> dict[str, str]:
    return {key: value for key, value in labels}


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

    def samples(self) -> list[tuple[LabelTuple, float]]:
        with self._lock:
            return [(tuple(labels), value) for labels, value in self._values.items()]

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
    minimum: float | None = None
    maximum: float | None = None
    sum_of_squares: float = 0.0


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
            state.sum_of_squares += value * value
            if state.minimum is None or value < state.minimum:
                state.minimum = value
            if state.maximum is None or value > state.maximum:
                state.maximum = value
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
            return HistogramState(
                counts=counts_copy,
                sum=state.sum,
                count=state.count,
                minimum=state.minimum,
                maximum=state.maximum,
                sum_of_squares=state.sum_of_squares,
            )

    def samples(self) -> list[tuple[LabelTuple, HistogramState]]:
        with self._lock:
            results: list[tuple[LabelTuple, HistogramState]] = []
            for labels, state in self._values.items():
                results.append(
                    (
                        tuple(labels),
                        HistogramState(
                            counts=dict(state.counts),
                            sum=state.sum,
                            count=state.count,
                            minimum=state.minimum,
                            maximum=state.maximum,
                            sum_of_squares=state.sum_of_squares,
                        ),
                    )
                )
            return results

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


def _quantile_from_histogram(state: HistogramState, quantile: float) -> float | None:
    if state.count == 0:
        return None
    if not 0.0 < quantile <= 1.0:
        raise ValueError("Quantyl musi należeć do zakresu (0, 1].")
    target = max(1, math.ceil(state.count * quantile))
    cumulative = 0
    for boundary in sorted(state.counts, key=lambda value: (math.inf if math.isinf(value) else value)):
        if math.isinf(boundary):
            return None
        cumulative += state.counts.get(boundary, 0)
        if cumulative >= target:
            return float(boundary)
    return None


def summarize_live_execution_metrics(
    registry: "MetricsRegistry",
    *,
    exchange: str | None = None,
    symbol: str | None = None,
    portfolio: str | None = None,
    route: str | None = None,
) -> dict[str, float | int | None]:
    """Zwraca zagregowane metryki egzekucji live na potrzeby regresji."""

    labels = {
        k: v
        for k, v in (
            ("exchange", exchange),
            ("symbol", symbol),
            ("portfolio", portfolio),
            ("route", route),
        )
        if v
    }
    normalized_labels = {key: str(value) for key, value in labels.items()}

    summary: dict[str, float | int | None] = {
        "fill_ratio_avg": None,
        "fill_ratio_count": 0,
        "fill_ratio_sum": 0.0,
        "fill_ratio_p50": None,
        "fill_ratio_p95": None,
        "fill_ratio_min": None,
        "fill_ratio_max": None,
        "fill_ratio_stddev": None,
        "latency_avg": None,
        "latency_p95": None,
        "latency_p50": None,
        "latency_p99": None,
        "latency_min": None,
        "latency_max": None,
        "latency_count": 0,
        "latency_sum": 0.0,
        "latency_stddev": None,
        "errors_total": 0,
        "orders_attempts_total": 0,
        "orders_attempts_success": 0,
        "orders_attempts_error": 0,
        "orders_attempts_api_error": 0,
        "orders_attempts_auth_error": 0,
        "orders_attempts_exception": 0,
        "orders_attempts_success_rate": None,
        "orders_attempts_error_rate": None,
        "orders_attempts_api_error_rate": None,
        "orders_attempts_auth_error_rate": None,
        "orders_attempts_exception_rate": None,
        "orders_routed_total": 0,
        "orders_success_total": 0,
        "orders_failed_total": 0,
        "orders_total": 0,
        "orders_fallback_total": 0,
        "orders_success_rate": None,
        "orders_failure_rate": None,
        "orders_fallback_rate": None,
    }

    def _counter_total(
        metric: CounterMetric,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> float:
        filters = dict(normalized_labels)
        if extra:
            filters.update({k: str(v) for k, v in extra.items()})

        total = 0.0
        for labels_tuple, value in metric.samples():
            label_map = _labels_to_dict(labels_tuple)
            match = True
            for key, expected in filters.items():
                actual = label_map.get(key)
                if actual is None or actual != expected:
                    match = False
                    break
            if match:
                total += value
        return total

    def _histogram_total(
        metric: HistogramMetric,
        *,
        extra: Mapping[str, str] | None = None,
    ) -> HistogramState:
        filters = dict(normalized_labels)
        if extra:
            filters.update({k: str(v) for k, v in extra.items()})

        aggregated: HistogramState | None = None
        for labels_tuple, state in metric.samples():
            label_map = _labels_to_dict(labels_tuple)
            match = True
            for key, expected in filters.items():
                actual = label_map.get(key)
                if actual is None or actual != expected:
                    match = False
                    break
            if not match:
                continue

            if aggregated is None:
                counts = {boundary: 0 for boundary in metric.buckets}
                counts[math.inf] = 0
                aggregated = HistogramState(counts=counts)

            aggregated.count += state.count
            aggregated.sum += state.sum
            aggregated.sum_of_squares += state.sum_of_squares
            for boundary, value in state.counts.items():
                aggregated.counts[boundary] = aggregated.counts.get(boundary, 0) + value
            if state.minimum is not None:
                if aggregated.minimum is None or state.minimum < aggregated.minimum:
                    aggregated.minimum = state.minimum
            if state.maximum is not None:
                if aggregated.maximum is None or state.maximum > aggregated.maximum:
                    aggregated.maximum = state.maximum

        if aggregated is None:
            counts = {boundary: 0 for boundary in metric.buckets}
            counts[math.inf] = 0
            return HistogramState(counts=counts)

        return aggregated

    try:
        fill_metric = registry.get("live_orders_fill_ratio")
    except KeyError:
        fill_metric = None
    if isinstance(fill_metric, HistogramMetric):
        fill_state = _histogram_total(fill_metric)
        if fill_state.count:
            summary["fill_ratio_sum"] = fill_state.sum
            summary["fill_ratio_count"] = fill_state.count
            summary["fill_ratio_avg"] = fill_state.sum / fill_state.count
            summary["fill_ratio_p50"] = _quantile_from_histogram(fill_state, 0.5)
            summary["fill_ratio_p95"] = _quantile_from_histogram(fill_state, 0.95)
            summary["fill_ratio_min"] = fill_state.minimum
            summary["fill_ratio_max"] = fill_state.maximum
            variance = (fill_state.sum_of_squares / fill_state.count) - (
                summary["fill_ratio_avg"] ** 2
            )
            if variance < 0:
                variance = 0.0
            summary["fill_ratio_stddev"] = math.sqrt(variance)

    try:
        latency_metric = registry.get("live_execution_latency_seconds")
    except KeyError:
        latency_metric = None
    if isinstance(latency_metric, HistogramMetric):
        latency_state = _histogram_total(latency_metric, extra={"result": "success"})
        if latency_state.count:
            summary["latency_count"] = latency_state.count
            summary["latency_sum"] = latency_state.sum
            summary["latency_avg"] = latency_state.sum / latency_state.count
            summary["latency_p50"] = _quantile_from_histogram(latency_state, 0.5)
            summary["latency_p95"] = _quantile_from_histogram(latency_state, 0.95)
            summary["latency_p99"] = _quantile_from_histogram(latency_state, 0.99)
            summary["latency_min"] = latency_state.minimum
            summary["latency_max"] = latency_state.maximum
            variance = (latency_state.sum_of_squares / latency_state.count) - (
                summary["latency_avg"] ** 2
            )
            if variance < 0:
                variance = 0.0
            summary["latency_stddev"] = math.sqrt(variance)

    try:
        errors_metric = registry.get("live_orders_errors_total")
    except KeyError:
        errors_metric = None
    if isinstance(errors_metric, CounterMetric):
        summary["errors_total"] = int(round(_counter_total(errors_metric)))

    try:
        attempts_metric = registry.get("live_orders_attempts_total")
    except KeyError:
        attempts_metric = None
    if isinstance(attempts_metric, CounterMetric):
        attempts_by_result = {
            "success": "orders_attempts_success",
            "error": "orders_attempts_error",
            "api_error": "orders_attempts_api_error",
            "auth_error": "orders_attempts_auth_error",
            "exception": "orders_attempts_exception",
        }
        attempts_total = 0.0
        for result, field in attempts_by_result.items():
            count = _counter_total(attempts_metric, extra={"result": result})
            summary[field] = int(round(count))
            attempts_total += count
        summary["orders_attempts_total"] = int(round(attempts_total))
        if attempts_total > 0:
            summary["orders_attempts_success_rate"] = summary["orders_attempts_success"] / attempts_total
            summary["orders_attempts_error_rate"] = summary["orders_attempts_error"] / attempts_total
            summary["orders_attempts_api_error_rate"] = summary["orders_attempts_api_error"] / attempts_total
            summary["orders_attempts_auth_error_rate"] = summary["orders_attempts_auth_error"] / attempts_total
            summary["orders_attempts_exception_rate"] = summary["orders_attempts_exception"] / attempts_total

    try:
        routed_metric = registry.get("live_orders_total")
    except KeyError:
        routed_metric = None
    if isinstance(routed_metric, CounterMetric):
        summary["orders_routed_total"] = int(round(_counter_total(routed_metric)))

    try:
        success_metric = registry.get("live_orders_success_total")
    except KeyError:
        success_metric = None
    if isinstance(success_metric, CounterMetric):
        summary["orders_success_total"] = int(round(_counter_total(success_metric)))

    try:
        failed_metric = registry.get("live_orders_failed_total")
    except KeyError:
        failed_metric = None
    if isinstance(failed_metric, CounterMetric):
        summary["orders_failed_total"] = int(round(_counter_total(failed_metric)))

    try:
        fallback_metric = registry.get("live_orders_fallback_total")
    except KeyError:
        fallback_metric = None
    if isinstance(fallback_metric, CounterMetric):
        summary["orders_fallback_total"] = int(round(_counter_total(fallback_metric)))

    orders_total = summary["orders_success_total"] + summary["orders_failed_total"]
    summary["orders_total"] = orders_total
    if orders_total > 0:
        summary["orders_success_rate"] = summary["orders_success_total"] / orders_total
        summary["orders_failure_rate"] = summary["orders_failed_total"] / orders_total
    if summary["orders_success_total"] > 0:
        summary["orders_fallback_rate"] = summary["orders_fallback_total"] / summary["orders_success_total"]

    return summary


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


@dataclass(slots=True)
class ExchangeMetricSet:
    """Zestaw metryk dotyczących adapterów giełdowych."""

    registry: MetricsRegistry
    _requests: CounterMetric = field(init=False)
    _latency: HistogramMetric = field(init=False)
    _health: GaugeMetric = field(init=False)
    _errors: CounterMetric = field(init=False)
    _rate_limits: CounterMetric = field(init=False)

    def __post_init__(self) -> None:
        self._requests = self.registry.counter(
            "bot_exchange_requests_total",
            "Liczba operacji wykonanych przez adapter giełdowy",
        )
        self._latency = self.registry.histogram(
            "bot_exchange_latency_seconds",
            "Rozkład opóźnień żądań do giełd",
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
        self._health = self.registry.gauge(
            "bot_exchange_health_status",
            "Bieżący status zdrowia adaptera giełdowego (1 = zdrowy, 0 = problemy)",
        )
        self._errors = self.registry.counter(
            "bot_exchange_errors_total",
            "Liczba błędów zwróconych przez adapter giełdowy",
        )
        self._rate_limits = self.registry.counter(
            "bot_exchange_rate_limited_total",
            "Liczba zdarzeń rate limit/retry dla adaptera",
        )

    def observe_log(
        self,
        *,
        exchange: str,
        outcome: str,
        severity: str,
        latency_seconds: float | None,
        rate_limited: bool,
    ) -> None:
        labels = {"exchange": exchange.lower(), "status": outcome, "severity": severity}
        self._requests.inc(labels=labels)
        if latency_seconds is not None and latency_seconds >= 0:
            self._latency.observe(latency_seconds, labels={"exchange": exchange.lower()})
        if severity in {"error", "critical"}:
            self._errors.inc(labels={"exchange": exchange.lower(), "severity": severity})
        if rate_limited:
            self._rate_limits.inc(labels={"exchange": exchange.lower()})

    def report_health(self, *, exchange: str, healthy: bool) -> None:
        self._health.set(1.0 if healthy else 0.0, labels={"exchange": exchange.lower()})


@dataclass(slots=True)
class StrategyMetricSet:
    """Zestaw metryk monitorujących strategie tradingowe."""

    registry: MetricsRegistry
    _decisions: CounterMetric = field(init=False)
    _latency: HistogramMetric = field(init=False)
    _warnings: CounterMetric = field(init=False)

    def __post_init__(self) -> None:
        self._decisions = self.registry.counter(
            "bot_strategy_decisions_total",
            "Liczba decyzji strategii (wykonane/odrzucone/błędy)",
        )
        self._latency = self.registry.histogram(
            "bot_strategy_decision_latency_seconds",
            "Opóźnienie podejmowania decyzji przez strategie",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
        )
        self._warnings = self.registry.counter(
            "bot_strategy_alerts_total",
            "Liczba ostrzeżeń generowanych przez strategie",
        )

    def observe_log(
        self,
        *,
        strategy: str,
        outcome: str,
        severity: str,
        latency_seconds: float | None,
    ) -> None:
        labels = {"strategy": strategy, "outcome": outcome, "severity": severity}
        self._decisions.inc(labels=labels)
        if latency_seconds is not None and latency_seconds >= 0:
            self._latency.observe(latency_seconds, labels={"strategy": strategy})
        if severity in {"warning", "error", "critical"}:
            self._warnings.inc(labels={"strategy": strategy, "severity": severity})


@dataclass(slots=True)
class SecurityMetricSet:
    """Metryki monitorujące moduły bezpieczeństwa i licencjonowanie."""

    registry: MetricsRegistry
    _events: CounterMetric = field(init=False)
    _failures: CounterMetric = field(init=False)

    def __post_init__(self) -> None:
        self._events = self.registry.counter(
            "bot_security_events_total",
            "Liczba zdarzeń bezpieczeństwa (walidacje, rotacje, alerty)",
        )
        self._failures = self.registry.counter(
            "bot_security_failures_total",
            "Liczba błędów modułów bezpieczeństwa",
        )

    def observe_log(self, *, source: str, event: str, severity: str) -> None:
        labels = {"source": source, "event": event, "severity": severity}
        self._events.inc(labels=labels)
        if severity in {"error", "critical"}:
            self._failures.inc(labels={"source": source, "event": event})


@dataclass(slots=True)
class PandasWarningMetricSet:
    """Metryki monitorujące ostrzeżenia generowane przez pandas."""

    registry: MetricsRegistry
    _warnings: CounterMetric = field(init=False)

    def __post_init__(self) -> None:
        self._warnings = self.registry.counter(
            "bot_pandas_warnings_total",
            "Liczba ostrzeżeń generowanych przez pandas w krytycznych komponentach",
        )

    def observe(self, *, component: str, category: str) -> None:
        labels = {"component": component, "category": category}
        self._warnings.inc(labels=labels)


_exchange_sets: Dict[str, ExchangeMetricSet] = {}
_strategy_sets: Dict[str, StrategyMetricSet] = {}
_security_sets: Dict[str, SecurityMetricSet] = {}
_pandas_metrics: Optional[PandasWarningMetricSet] = None
_metrics_lock = threading.Lock()
_exchange_error_streak: dict[str, int] = defaultdict(int)
_pandas_warning_streak: dict[str, int] = defaultdict(int)
_pandas_warning_alerted: set[str] = set()
_pandas_warning_examples: dict[str, str] = {}
_PANDAS_WARNING_ALERT_THRESHOLD = 3


def _resolve_severity(levelno: int) -> str:
    if levelno >= logging.CRITICAL:
        return "critical"
    if levelno >= logging.ERROR:
        return "error"
    if levelno >= logging.WARNING:
        return "warning"
    if levelno >= logging.INFO:
        return "info"
    return "debug"


def _dispatch_metric_alert(
    *,
    message: str,
    severity: AlertSeverity,
    source: str,
    context: Mapping[str, str] | None = None,
) -> None:
    try:
        emit_alert(
            message,
            severity=severity,
            source=source,
            context=dict(context or {}),
        )
    except Exception:  # pragma: no cover - alert nie powinien zatrzymać procesu metryk
        _LOGGER.debug("Nie udało się wysłać alertu metrycznego", exc_info=True)


def _get_exchange_metrics(exchange: str, registry: MetricsRegistry | None = None) -> ExchangeMetricSet:
    with _metrics_lock:
        normalized = exchange.lower()
        if normalized not in _exchange_sets:
            _exchange_sets[normalized] = ExchangeMetricSet(registry or get_global_metrics_registry())
        return _exchange_sets[normalized]


def _get_strategy_metrics(strategy: str, registry: MetricsRegistry | None = None) -> StrategyMetricSet:
    with _metrics_lock:
        if strategy not in _strategy_sets:
            _strategy_sets[strategy] = StrategyMetricSet(registry or get_global_metrics_registry())
        return _strategy_sets[strategy]


def _get_security_metrics(source: str, registry: MetricsRegistry | None = None) -> SecurityMetricSet:
    with _metrics_lock:
        if source not in _security_sets:
            _security_sets[source] = SecurityMetricSet(registry or get_global_metrics_registry())
        return _security_sets[source]


def _get_pandas_warning_metrics(registry: MetricsRegistry | None = None) -> PandasWarningMetricSet:
    global _pandas_metrics
    with _metrics_lock:
        if _pandas_metrics is None:
            _pandas_metrics = PandasWarningMetricSet(registry or get_global_metrics_registry())
        return _pandas_metrics


def observe_exchange_log_record(
    record: logging.LogRecord,
    *,
    registry: MetricsRegistry | None = None,
) -> None:
    exchange_name = getattr(record, "exchange", None)
    if not exchange_name:
        for token in record.name.split("."):
            lowered = token.lower()
            if lowered in {
                "binance",
                "coinbase",
                "kraken",
                "okx",
                "bitget",
                "bybit",
                "bitfinex",
                "kucoin",
                "mexc",
                "zonda",
            }:
                exchange_name = lowered
                break
    if not exchange_name:
        return

    severity = _resolve_severity(record.levelno)
    outcome = getattr(record, "event_status", None) or ("error" if severity in {"error", "critical"} else "ok")
    latency_seconds: float | None = None
    if hasattr(record, "latency_seconds"):
        try:
            latency_seconds = float(record.latency_seconds)
        except Exception:
            latency_seconds = None
    elif hasattr(record, "latency_ms"):
        try:
            latency_seconds = float(record.latency_ms) / 1000.0
        except Exception:
            latency_seconds = None

    rate_limited = bool(getattr(record, "rate_limited", False))
    if not rate_limited:
        message = record.getMessage().lower()
        if "rate limit" in message or "retry-after" in message:
            rate_limited = True

    metrics = _get_exchange_metrics(str(exchange_name), registry)
    metrics.observe_log(
        exchange=str(exchange_name),
        outcome=str(outcome),
        severity=severity,
        latency_seconds=latency_seconds,
        rate_limited=rate_limited,
    )

    if severity in {"error", "critical"}:
        key = str(exchange_name)
        _exchange_error_streak[key] += 1
        if _exchange_error_streak[key] >= 5:
            _dispatch_metric_alert(
                message=f"Seria błędów adaptera giełdowego: {exchange_name}",
                severity=AlertSeverity.ERROR if severity == "error" else AlertSeverity.CRITICAL,
                source=f"exchange:{exchange_name}",
                context={
                    "error_count": str(_exchange_error_streak[key]),
                    "logger": record.name,
                },
            )
    else:
        _exchange_error_streak.pop(str(exchange_name), None)


def observe_strategy_log_record(
    record: logging.LogRecord,
    *,
    registry: MetricsRegistry | None = None,
) -> None:
    strategy_name = getattr(record, "strategy", None)
    if not strategy_name and "strategies" in record.name:
        parts = record.name.split(".")
        try:
            idx = parts.index("strategies")
        except ValueError:
            idx = -1
        if idx >= 0 and len(parts) > idx + 1:
            strategy_name = parts[idx + 1]
    if not strategy_name:
        return

    severity = _resolve_severity(record.levelno)
    outcome = getattr(record, "event_status", None) or (
        "rejected" if severity in {"warning", "error", "critical"} else "executed"
    )
    latency_seconds: float | None = None
    if hasattr(record, "decision_latency_seconds"):
        try:
            latency_seconds = float(record.decision_latency_seconds)
        except Exception:
            latency_seconds = None
    metrics = _get_strategy_metrics(str(strategy_name), registry)
    metrics.observe_log(
        strategy=str(strategy_name),
        outcome=str(outcome),
        severity=severity,
        latency_seconds=latency_seconds,
    )

    if severity == "critical":
        _dispatch_metric_alert(
            message=f"Strategia {strategy_name} zgłosiła krytyczny incydent",
            severity=AlertSeverity.CRITICAL,
            source=f"strategy:{strategy_name}",
            context={"logger": record.name},
        )


def observe_security_log_record(
    record: logging.LogRecord,
    *,
    registry: MetricsRegistry | None = None,
) -> None:
    source = getattr(record, "security_source", None)
    if not source and record.name.startswith("bot_core.security"):
        source = record.name.split(".")[2] if len(record.name.split(".")) > 2 else "core"
    if not source:
        return

    severity = _resolve_severity(record.levelno)
    event = getattr(record, "security_event", None) or ("failure" if severity in {"error", "critical"} else "info")
    metrics = _get_security_metrics(str(source), registry)
    metrics.observe_log(source=str(source), event=str(event), severity=severity)

    if severity in {"error", "critical"}:
        _dispatch_metric_alert(
            message=f"Incydent bezpieczeństwa: {source}",
            severity=AlertSeverity.CRITICAL if severity == "critical" else AlertSeverity.ERROR,
            source=f"security:{source}",
            context={"logger": record.name, "event": str(event)},
        )


def observe_pandas_warning(
    *,
    component: str,
    category: str,
    message: str | None = None,
    registry: MetricsRegistry | None = None,
) -> None:
    """Zarejestruj ostrzeżenie pandas i potencjalnie wyślij alert."""

    metrics = _get_pandas_warning_metrics(registry)
    metrics.observe(component=component, category=category)

    key = f"{component}:{category}"
    _pandas_warning_streak[key] += 1
    if message:
        _pandas_warning_examples[key] = message

    if _pandas_warning_streak[key] >= _PANDAS_WARNING_ALERT_THRESHOLD and key not in _pandas_warning_alerted:
        alert_context: dict[str, str] = {
            "category": category,
            "count": str(_pandas_warning_streak[key]),
        }
        if example := _pandas_warning_examples.get(key):
            alert_context["example"] = example[:200]

        _dispatch_metric_alert(
            message=f"Ostrzeżenia pandas w komponencie {component}",
            severity=AlertSeverity.WARNING,
            source=f"pandas_warning:{component}",
            context=alert_context,
        )
        _pandas_warning_alerted.add(key)


def get_global_metrics_registry() -> MetricsRegistry:
    """Zwraca singleton rejestru metryk dla całej aplikacji."""

    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _GLOBAL_REGISTRY is None:
                _GLOBAL_REGISTRY = MetricsRegistry()
    return _GLOBAL_REGISTRY


def reset_pandas_warning_tracking() -> None:
    """Wyzeruj stan monitorowania ostrzeżeń pandas (tylko do testów)."""

    global _pandas_metrics

    with _metrics_lock:
        _pandas_metrics = None

    _pandas_warning_streak.clear()
    _pandas_warning_alerted.clear()
    _pandas_warning_examples.clear()


__all__ = [
    "MetricsRegistry",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "get_global_metrics_registry",
    "observe_pandas_warning",
    "reset_pandas_warning_tracking",
]
