from __future__ import annotations

import json
from pathlib import Path

from bot_core.config.models import (
    ResilienceConfig,
    ResilienceDrillConfig,
    ResilienceDrillThresholdsConfig,
)
from bot_core.resilience import ResilienceFailoverDrill


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
