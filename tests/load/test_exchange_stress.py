import asyncio
import json
import random
import zlib
from pathlib import Path
from typing import Callable

import httpx
import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from scripts.loadtests.exchange_stress import (
    ExchangeScenario,
    ExchangeStressConfig,
    QueueDefaults,
    _execute_request,
    load_config,
    main as cli_main,
    run_exchange_stress,
)


def _find_seed_for_retry_transition(
    *,
    exchange: str,
    request_index: int,
    attempt0_predicate: Callable[[float], bool],
    attempt1_predicate: Callable[[float], bool],
    base_latency_ms: float,
    jitter_ms: float,
    limit: int = 200_000,
) -> int:
    for seed in range(limit):
        scenario_seed = (seed + zlib.crc32(exchange.encode("utf-8"))) & 0xFFFFFFFF
        rng0 = random.Random((scenario_seed + request_index * 1009 + 0 * 9176) & 0xFFFFFFFF)
        _ = max(0.0, rng0.gauss(base_latency_ms, jitter_ms))
        roll0 = rng0.random()
        rng1 = random.Random((scenario_seed + request_index * 1009 + 1 * 9176) & 0xFFFFFFFF)
        _ = max(0.0, rng1.gauss(base_latency_ms, jitter_ms))
        roll1 = rng1.random()
        if attempt0_predicate(roll0) and attempt1_predicate(roll1):
            return seed
    raise AssertionError("Nie znaleziono seeda spełniającego warunki retry")


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
    assert metrics.retry_attempts == 0
    assert metrics.transient_5xx == 0
    assert metrics.throttle_events == 0
    assert metrics.transient_transport_errors == 0
    assert metrics.recovered_after_retry == 0
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
    assert metrics.retry_attempts == 6
    assert metrics.throttle_events == 9
    assert metrics.transient_5xx == 0
    assert metrics.transient_transport_errors == 0
    assert metrics.recovered_after_retry == 0


def test_run_exchange_stress_retries_and_recovers_from_transient_500() -> None:
    seed = _find_seed_for_retry_transition(
        exchange="kraken_spot",
        request_index=0,
        attempt0_predicate=lambda roll: roll < 0.6,
        attempt1_predicate=lambda roll: roll >= 0.6,
        base_latency_ms=0.5,
        jitter_ms=0.0,
    )
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="kraken_spot",
                request_count=1,
                base_latency_ms=0.5,
                jitter_ms=0.0,
                error_rate=0.6,
                throttle_rate=0.0,
                max_retries=3,
                backoff_base_ms=0.1,
                backoff_jitter_ms=0.0,
                timeout_ms=100.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=seed))
    metrics = result.metrics["kraken_spot"]
    assert metrics.total_requests == 1
    assert metrics.successes == 1
    assert metrics.errors == 0
    assert metrics.throttled == 0
    assert metrics.retry_attempts >= 1
    assert metrics.transient_5xx >= 1
    assert metrics.throttle_events == 0
    assert metrics.transient_transport_errors == 0
    assert metrics.recovered_after_retry == 1


def test_run_exchange_stress_retries_and_recovers_from_transient_429() -> None:
    seed = _find_seed_for_retry_transition(
        exchange="binance_spot",
        request_index=0,
        attempt0_predicate=lambda roll: roll < 0.6,
        attempt1_predicate=lambda roll: roll >= 0.6,
        base_latency_ms=0.5,
        jitter_ms=0.0,
    )
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="binance_spot",
                request_count=1,
                base_latency_ms=0.5,
                jitter_ms=0.0,
                error_rate=0.0,
                throttle_rate=0.6,
                max_retries=3,
                backoff_base_ms=0.1,
                backoff_jitter_ms=0.0,
                timeout_ms=100.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=seed))
    metrics = result.metrics["binance_spot"]
    assert metrics.total_requests == 1
    assert metrics.successes == 1
    assert metrics.errors == 0
    assert metrics.throttled == 0
    assert metrics.retry_attempts >= 1
    assert metrics.throttle_events >= 1
    assert metrics.transient_5xx == 0
    assert metrics.transient_transport_errors == 0
    assert metrics.recovered_after_retry == 1


def test_run_exchange_stress_reports_hard_failure_after_retry_budget() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="kraken_spot",
                request_count=1,
                base_latency_ms=0.5,
                jitter_ms=0.0,
                error_rate=1.0,
                throttle_rate=0.0,
                max_retries=2,
                backoff_base_ms=0.1,
                backoff_jitter_ms=0.0,
                timeout_ms=100.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=1))
    metrics = result.metrics["kraken_spot"]
    assert metrics.total_requests == 1
    assert metrics.successes == 0
    assert metrics.errors == 1
    assert metrics.throttled == 0
    assert metrics.retry_attempts == 2
    assert metrics.transient_5xx == 3
    assert metrics.throttle_events == 0
    assert metrics.transient_transport_errors == 0
    assert metrics.recovered_after_retry == 0


def test_latency_ms_includes_retry_backoff_end_to_end() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="kraken_spot",
                request_count=1,
                base_latency_ms=0.2,
                jitter_ms=0.0,
                error_rate=1.0,
                throttle_rate=0.0,
                max_retries=1,
                backoff_base_ms=40.0,
                backoff_jitter_ms=0.0,
                timeout_ms=100.0,
            )
        ],
    )
    result = asyncio.run(run_exchange_stress(config, seed=11))
    latency = result.metrics["kraken_spot"].latency_summary()["avg_ms"]
    assert latency >= 35.0


def test_execute_request_counts_transient_transport_errors_on_timeout() -> None:
    class _TimeoutThenSuccessClient:
        def __init__(self) -> None:
            self.calls = 0

        async def get(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                req = httpx.Request("GET", "https://exchange-stress.local/kraken_spot")
                raise httpx.TimeoutException("timeout", request=req)
            return httpx.Response(200, json={"status": "ok"})

    scenario = ExchangeScenario(
        name="kraken_spot",
        request_count=1,
        base_latency_ms=0.1,
        jitter_ms=0.0,
        error_rate=0.0,
        throttle_rate=0.0,
        max_retries=2,
        backoff_base_ms=0.1,
        backoff_jitter_ms=0.0,
        timeout_ms=100.0,
    )
    outcome = asyncio.run(
        _execute_request(
            _TimeoutThenSuccessClient(),
            scenario,
            0,
            enqueued_at=0.0,
            seed=123,
        )
    )
    assert outcome.status_code == 200
    assert outcome.retry_attempts == 1
    assert outcome.transient_transport_errors == 1
    assert outcome.recovered_after_retry is True


def test_run_exchange_stress_remains_deterministic_with_seed() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=3, burst=6),
        scenarios=[
            ExchangeScenario(
                name="binance_spot",
                request_count=12,
                base_latency_ms=0.2,
                jitter_ms=0.0,
                error_rate=0.0,
                throttle_rate=0.25,
                max_retries=2,
                backoff_base_ms=0.1,
                backoff_jitter_ms=0.0,
                timeout_ms=100.0,
            )
        ],
    )
    first = asyncio.run(run_exchange_stress(config, seed=1234))
    second = asyncio.run(run_exchange_stress(config, seed=1234))
    assert first.metrics["binance_spot"].as_dict() == second.metrics["binance_spot"].as_dict()


def test_report_schema_remains_backward_compatible_and_extended() -> None:
    config = ExchangeStressConfig(
        queue_defaults=QueueDefaults(max_concurrency=1, burst=2),
        scenarios=[
            ExchangeScenario(
                name="coinbase_spot",
                request_count=1,
                base_latency_ms=0.2,
                jitter_ms=0.0,
                error_rate=0.0,
                throttle_rate=0.0,
            )
        ],
    )
    payload = asyncio.run(run_exchange_stress(config, seed=42)).as_dict()["exchanges"]["coinbase_spot"]
    for field in ("total_requests", "successes", "errors", "throttled", "latency_ms", "queue_wait_ms"):
        assert field in payload
    for field in ("retry_attempts", "transient_5xx", "throttle_events", "recovered_after_retry"):
        assert field in payload
    assert "transient_transport_errors" in payload


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
    assert scenario.max_retries == 2
    assert scenario.backoff_base_ms == 20.0
    assert scenario.backoff_jitter_ms == 5.0
    assert scenario.timeout_ms == 500.0


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
    exit_code = cli_main(
        [
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--seed",
            "7",
        ]
    )
    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["exchanges"]["cli_exchange"]["total_requests"] == 2
