from pathlib import Path
import sys

import math

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.observability.metrics import (  # type: ignore[import-not-found]
    CounterMetric,
    GaugeMetric,
    HistogramMetric,
    MetricsRegistry,
)


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


