from __future__ import annotations

import json

import pytest
from pathlib import Path

from bot_core.config.models import (
    ResilienceConfig,
    ResilienceDrillConfig,
    ResilienceDrillThresholdsConfig,
)
from bot_core.resilience.failover import (
    FailoverDrillMetrics,
    FailoverDrillReport,
    FailoverDrillResult,
    ResilienceFailoverDrill,
)


def _write_dataset(tmp_path: Path, *, max_latency: float, error_rate: float, duration: float, failed: int) -> Path:
    dataset_path = tmp_path / "dataset.json"
    payload = {
        "max_latency_ms": max_latency,
        "error_rate": error_rate,
        "failover_duration_seconds": duration,
        "orders_redirected": 100,
        "orders_failed": failed,
        "notes": ["unit-test"],
    }
    dataset_path.write_text(json.dumps(payload), encoding="utf-8")
    return dataset_path


def test_resilience_failover_pass(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path, max_latency=200.0, error_rate=0.03, duration=60.0, failed=2)
    config = ResilienceConfig(
        enabled=True,
        drills=(
            ResilienceDrillConfig(
                name="pass",
                primary="binance",
                fallbacks=("kraken",),
                dataset_path=str(dataset_path),
                thresholds=ResilienceDrillThresholdsConfig(
                    max_latency_ms=220.0,
                    max_error_rate=0.08,
                    max_failover_duration_seconds=90.0,
                    max_orders_failed=5,
                ),
            ),
        ),
    )

    report = ResilienceFailoverDrill(config).run()
    assert not report.has_failures()
    mapping = report.to_mapping()
    assert mapping["failure_count"] == 0
    assert mapping["drills"][0]["status"] == "passed"


def test_resilience_failover_fail(tmp_path: Path) -> None:
    dataset_path = _write_dataset(tmp_path, max_latency=250.0, error_rate=0.12, duration=150.0, failed=12)
    config = ResilienceConfig(
        enabled=True,
        require_success=True,
        drills=(
            ResilienceDrillConfig(
                name="fail",
                primary="binance",
                fallbacks=("kraken",),
                dataset_path=str(dataset_path),
                thresholds=ResilienceDrillThresholdsConfig(
                    max_latency_ms=200.0,
                    max_error_rate=0.05,
                    max_failover_duration_seconds=90.0,
                    max_orders_failed=5,
                ),
            ),
        ),
    )

    report = ResilienceFailoverDrill(config).run()
    assert report.has_failures()
    result = report.drills[0]
    assert result.has_failures()
    assert "max_latency_ms" in result.failures[0]


def test_resilience_failover_extract_metrics_normalizes_inputs(tmp_path: Path) -> None:
    dataset = {
        "max_latency_ms": {"p95": 180.0, "p99": 240.0},
        "error_rate": {"avg": 0.03, "p99": 0.07},
        "failover_duration_seconds": {"median": "invalid"},
        "orders_redirected": "42",
        "orders_failed": "-3",
        "notes": ["  ok  ", "", 7, None, {"ignored": True}, "second"],
    }

    drill = ResilienceFailoverDrill(ResilienceConfig(enabled=True, drills=()))
    metrics = drill._extract_metrics(dataset)

    assert metrics.max_latency_ms == 240.0
    assert metrics.error_rate == 0.07
    assert metrics.failover_duration_seconds == 0.0
    assert metrics.orders_redirected == 42
    # orders_failed is clamped to be non-negative
    assert metrics.orders_failed == 0
    assert metrics.notes == ("ok", "7", "second")


def test_resilience_failover_extract_metrics_accepts_string_notes() -> None:
    dataset = {
        "max_latency_ms": 120.0,
        "error_rate": 0.01,
        "failover_duration_seconds": 30.0,
        "orders_redirected": 5,
        "orders_failed": 1,
        "notes": "  single note  ",
    }

    drill = ResilienceFailoverDrill(ResilienceConfig(enabled=True, drills=()))
    metrics = drill._extract_metrics(dataset)

    assert metrics.notes == ("single note",)


def test_resilience_failover_load_dataset_requires_mapping(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.json"
    dataset_path.write_text("[1, 2, 3]", encoding="utf-8")

    drill = ResilienceFailoverDrill(ResilienceConfig(enabled=True, drills=()))

    with pytest.raises(ValueError):
        drill._load_dataset(dataset_path)


def test_failover_report_serialization_and_signature(tmp_path: Path) -> None:
    metrics = FailoverDrillMetrics(
        max_latency_ms=200.0,
        error_rate=0.05,
        failover_duration_seconds=120.0,
        orders_redirected=15,
        orders_failed=3,
        notes=("primary outage",),
    )
    thresholds = ResilienceDrillThresholdsConfig(
        max_latency_ms=150.0,
        max_error_rate=0.02,
        max_failover_duration_seconds=90.0,
        max_orders_failed=1,
    )
    result = FailoverDrillResult(
        name="dc-outage",
        primary="primary-dc",
        fallbacks=("secondary-dc",),
        status="failed",
        metrics=metrics,
        thresholds=thresholds,
        failures=("max_latency_ms=200.00>150.00",),
        description="Monthly failover",
        dataset_path="var/resilience/datasets/dc-outage.json",
    )
    report = FailoverDrillReport(generated_at="2024-01-01T00:00:00+00:00", drills=(result,))

    json_path = tmp_path / "report.json"
    signature_path = tmp_path / "report.sig"

    written_json = report.write_json(json_path)
    assert written_json == json_path
    report_payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert report_payload["failure_count"] == 1
    assert report_payload["drills"][0]["description"] == "Monthly failover"
    assert report_payload["drills"][0]["dataset_path"] == "var/resilience/datasets/dc-outage.json"

    expected_signature = report.build_signature(key=b"secret-key", key_id="stage6")
    written_signature = report.write_signature(signature_path, key=b"secret-key", key_id="stage6")
    assert written_signature == signature_path
    assert json.loads(signature_path.read_text(encoding="utf-8")) == expected_signature
