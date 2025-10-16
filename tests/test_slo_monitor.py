from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.observability import (
    SLOCompositeDefinition,
    SLODefinition,
    SLOMeasurement,
    SLOMonitor,
    write_slo_results_csv,
)


def test_slo_monitor_evaluates_statuses() -> None:
    definitions = [
        SLODefinition(
            name="availability",
            indicator="router_availability_pct",
            target=99.0,
            comparison=">=",
            warning_threshold=99.5,
        ),
        SLODefinition(
            name="latency",
            indicator="router_latency_ms",
            target=250.0,
            comparison="<=",
            warning_threshold=200.0,
        ),
        SLODefinition(
            name="fill_rate",
            indicator="order_fill_pct",
            target=97.0,
            comparison=">=",
        ),
    ]

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    measurements = {
        "router_availability_pct": SLOMeasurement(
            indicator="router_availability_pct",
            value=98.4,
            window_start=start,
            window_end=end,
            sample_size=86400,
        ),
        "router_latency_ms": SLOMeasurement(
            indicator="router_latency_ms",
            value=240.0,
            window_start=start,
            window_end=end,
            sample_size=5000,
        ),
    }

    monitor = SLOMonitor(definitions)
    statuses = monitor.evaluate(measurements)

    availability = statuses["availability"]
    assert availability.status == "breach"
    assert availability.severity == "critical"
    assert pytest.approx(availability.error_budget_pct, rel=1e-3) == (99.0 - 98.4) / 99.0

    latency = statuses["latency"]
    assert latency.status == "warning"
    assert latency.severity == "warning"
    assert latency.warning_threshold == pytest.approx(200.0)

    fills = statuses["fill_rate"]
    assert fills.status == "unknown"
    assert fills.reason == "brak danych"

    summary = monitor.summary(statuses)
    assert summary["status_counts"]["breach"] == 1
    assert summary["status_counts"]["warning"] == 1
    assert summary["status_counts"]["unknown"] == 1
    assert summary["status_counts"]["ok"] == 0


def test_slo_monitor_composites() -> None:
    definitions = [
        SLODefinition(
            name="availability",
            indicator="router_availability_pct",
            target=99.0,
            comparison=">=",
        ),
        SLODefinition(
            name="latency",
            indicator="router_latency_ms",
            target=250.0,
            comparison="<=",
        ),
        SLODefinition(
            name="fill_rate",
            indicator="order_fill_pct",
            target=97.0,
            comparison=">=",
        ),
    ]
    composites = [
        SLOCompositeDefinition(
            name="core_stack",
            objectives=("availability", "latency", "fill_rate"),
            max_breaches=1,
            max_warnings=1,
            min_ok_ratio=0.66,
            severity="critical",
        ),
        SLOCompositeDefinition(
            name="market_health",
            objectives=("fill_rate",),
            max_breaches=0,
            severity="error",
        ),
    ]

    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)
    measurements = {
        "router_availability_pct": SLOMeasurement(
            indicator="router_availability_pct",
            value=98.4,
            window_start=start,
            window_end=end,
            sample_size=86400,
        ),
        "router_latency_ms": SLOMeasurement(
            indicator="router_latency_ms",
            value=210.0,
            window_start=start,
            window_end=end,
            sample_size=5000,
        ),
        "order_fill_pct": SLOMeasurement(
            indicator="order_fill_pct",
            value=97.2,
            window_start=start,
            window_end=end,
            sample_size=2000,
        ),
    }

    monitor = SLOMonitor(definitions, composites=composites)
    statuses = monitor.evaluate(measurements)
    composite_statuses = monitor.evaluate_composites(statuses)

    assert composite_statuses["core_stack"].status == "warning"
    assert "tolerowane breach" in (composite_statuses["core_stack"].reason or "")
    assert composite_statuses["market_health"].status == "ok"

    summary = monitor.summary(statuses, composite_statuses)
    assert summary["composites"]["status_counts"]["warning"] == 1
    assert summary["composites"]["status_counts"]["ok"] == 1


def test_write_slo_results_csv(tmp_path: Path) -> None:
    definitions = [
        SLODefinition(
            name="availability",
            indicator="router_availability_pct",
            target=99.0,
            comparison=">=",
        ),
        SLODefinition(
            name="latency",
            indicator="router_latency_ms",
            target=250.0,
            comparison="<=",
            warning_threshold=220.0,
        ),
    ]
    composites = [
        SLOCompositeDefinition(
            name="core_stack",
            objectives=("availability", "latency"),
            max_breaches=1,
            severity="critical",
        )
    ]
    start = datetime(2024, 2, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 2, tzinfo=timezone.utc)
    measurements = {
        "router_availability_pct": SLOMeasurement(
            indicator="router_availability_pct",
            value=99.5,
            window_start=start,
            window_end=end,
            sample_size=86400,
        ),
        "router_latency_ms": SLOMeasurement(
            indicator="router_latency_ms",
            value=260.0,
            window_start=start,
            window_end=end,
            sample_size=5000,
        ),
    }

    monitor = SLOMonitor(definitions, composites=composites)
    statuses = monitor.evaluate(measurements)
    composite_statuses = monitor.evaluate_composites(statuses)

    output = tmp_path / "slo_report.csv"
    write_slo_results_csv(statuses, output, composites=composite_statuses)

    rows = output.read_text(encoding="utf-8").strip().splitlines()
    assert rows[0].split(",")[0] == "type"
    assert any("slo,availability" in row for row in rows[1:])
    assert any("composite,core_stack" in row for row in rows[1:])
