import math

import pytest


from bot_core.observability.metrics import (  # type: ignore[import-not-found]
    observe_pandas_warning,
    reset_pandas_warning_tracking,
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
)
import bot_core.observability.metrics as metrics_module  # type: ignore[import-not-found]


def test_counter_tracks_values_per_labels() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("orders_total", "Liczba zleceń")
    counter.inc(labels={"symbol": "BTCUSDT"})
    counter.inc(2.0, labels={"symbol": "BTCUSDT"})
    counter.inc(labels={"symbol": "ETHUSDT"})

    assert counter.value(labels={"symbol": "BTCUSDT"}) == pytest.approx(3.0)
    assert counter.value(labels={"symbol": "ETHUSDT"}) == pytest.approx(1.0)


def test_gauge_can_set_and_increment() -> None:
    registry = MetricsRegistry()
    gauge = registry.gauge("open_positions", "Liczba otwartych pozycji")
    gauge.set(2.0, labels={"profile": "balanced"})
    gauge.inc(labels={"profile": "balanced"})
    gauge.dec(0.5, labels={"profile": "balanced"})

    assert gauge.value(labels={"profile": "balanced"}) == pytest.approx(2.5)


def test_histogram_records_buckets_and_snapshot() -> None:
    registry = MetricsRegistry()
    histogram = registry.histogram(
        "latency_seconds",
        "Latency prób",
        buckets=(0.1, 0.5, 1.0),
    )

    histogram.observe(0.05, labels={"status": "ok"})
    histogram.observe(0.3, labels={"status": "ok"})
    histogram.observe(0.9, labels={"status": "ok"})

    snapshot = histogram.snapshot(labels={"status": "ok"})
    assert snapshot.count == 3
    assert snapshot.counts[0.1] == 1
    assert snapshot.counts[0.5] == 2
    assert snapshot.counts[1.0] == 3
    assert snapshot.counts[math.inf] == 3
    assert snapshot.sum == pytest.approx(1.25)


def test_render_prometheus_format_contains_headers() -> None:
    registry = MetricsRegistry()
    registry.counter("test_metric", "Opis").inc()
    output = registry.render_prometheus()
    assert "# HELP test_metric Opis" in output
    assert "# TYPE test_metric counter" in output


def test_registry_get_returns_metric_instance() -> None:
    registry = MetricsRegistry()
    counter = registry.counter("executions_total", "Licznik")
    retrieved = registry.get("executions_total")
    assert isinstance(retrieved, CounterMetric)
    assert retrieved is counter


def test_observe_pandas_warning_records_metrics_labels() -> None:
    registry = MetricsRegistry()
    reset_pandas_warning_tracking()
    try:
        observe_pandas_warning(
            component="engine", category="PerformanceWarning", registry=registry
        )
        counter = registry.get("bot_pandas_warnings_total")
        assert isinstance(counter, CounterMetric)
        assert counter.value(
            labels={"component": "engine", "category": "PerformanceWarning"}
        ) == pytest.approx(1.0)
    finally:
        reset_pandas_warning_tracking()


def test_observe_pandas_warning_emits_alert_after_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    reset_pandas_warning_tracking()
    alerts: list[dict[str, object]] = []

    def fake_dispatch(**kwargs: object) -> None:
        alerts.append(kwargs)  # pragma: no cover - prosty rejestrator

    monkeypatch.setattr(metrics_module, "_dispatch_metric_alert", fake_dispatch)

    try:
        for _ in range(metrics_module._PANDAS_WARNING_ALERT_THRESHOLD):
            observe_pandas_warning(
                component="engine",
                category="PerformanceWarning",
                message="vectorized fallback",
                registry=registry,
            )

        assert len(alerts) == 1
        payload = alerts[0]
        assert payload["source"] == "pandas_warning:engine"
        assert payload["message"] == "Ostrzeżenia pandas w komponencie engine"
        context = payload["context"]
        assert isinstance(context, dict)
        assert context["count"] == str(metrics_module._PANDAS_WARNING_ALERT_THRESHOLD)
        assert context["example"] == "vectorized fallback"

        counter = registry.get("bot_pandas_warnings_total")
        assert isinstance(counter, CounterMetric)
        assert counter.value(
            labels={"component": "engine", "category": "PerformanceWarning"}
        ) == pytest.approx(float(metrics_module._PANDAS_WARNING_ALERT_THRESHOLD))
    finally:
        reset_pandas_warning_tracking()


