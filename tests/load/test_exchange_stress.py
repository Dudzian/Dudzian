import asyncio
import json
from pathlib import Path

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from scripts.loadtests.exchange_stress import (
    ExchangeScenario,
    ExchangeStressConfig,
    QueueDefaults,
    load_config,
    main as cli_main,
    run_exchange_stress,
)


def test_run_exchange_stress_collects_success_metrics() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=2, burst=4),
        scenarios=[
            ExchangeScenario(
                name="binance_spot",
                request_count=5,
                base_latency_ms=1.0,
                jitter_ms=0.0,
                error_rate=0.0,
                throttle_rate=0.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=123))
    metrics = result.metrics["binance_spot"]
    assert metrics.total_requests == 5
    assert metrics.successes == 5
    assert metrics.errors == 0
    assert metrics.throttled == 0
    summary = metrics.latency_summary()
    assert summary["max_ms"] >= summary["min_ms"]
    assert summary["p95_ms"] >= summary["avg_ms"]
    queue_summary = metrics.queue_wait_summary()
    assert queue_summary["max_ms"] >= queue_summary["min_ms"]
    assert queue_summary["p95_ms"] >= queue_summary["avg_ms"]


def test_run_exchange_stress_counts_throttling() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="kraken_spot",
                request_count=3,
                base_latency_ms=0.5,
                jitter_ms=0.0,
                error_rate=0.0,
                throttle_rate=1.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=5))
    metrics = result.metrics["kraken_spot"]
    assert metrics.total_requests == 3
    assert metrics.successes == 0
    assert metrics.errors == 0
    assert metrics.throttled == 3


def test_load_config_applies_defaults(tmp_path: Path) -> None:
    config_path = tmp_path / "scenario.yml"
    config_path.write_text(
        """
        defaults:
          max_concurrency: 6
          burst: 12
          request_count: 4
          base_latency_ms: 10.0
          jitter_ms: 1.0
        exchanges:
          - name: test_exchange
        """,
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.queue_defaults.max_concurrency == 6
    assert config.queue_defaults.burst == 12
    scenario = config.scenarios[0]
    assert scenario.request_count == 4
    assert scenario.base_latency_ms == 10.0
    assert scenario.jitter_ms == 1.0
    assert scenario.error_rate == 0.0
    assert scenario.throttle_rate == 0.0


def test_cli_main_writes_output(tmp_path: Path) -> None:
    config_path = tmp_path / "scenario.yml"
    config_path.write_text(
        """
        defaults:
          request_count: 2
          base_latency_ms: 0.2
          jitter_ms: 0.0
        exchanges:
          - name: cli_exchange
        """,
        encoding="utf-8",
    )
    output_path = tmp_path / "result.json"
    exit_code = cli_main([
        "--config",
        str(config_path),
        "--output",
        str(output_path),
        "--seed",
        "7",
    ])
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["exchanges"]["cli_exchange"]["total_requests"] == 2
