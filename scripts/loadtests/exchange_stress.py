"""Symulator obciążeniowy integracji giełd z wykorzystaniem AsyncIOTaskQueue."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
import statistics
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bot_core.runtime.scheduler import AsyncIOTaskQueue

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class QueueDefaults:
    """Domyślne limity kolejki I/O."""

    max_concurrency: int = 8
    burst: int = 16


@dataclass(slots=True)
class ExchangeScenario:
    """Konfiguracja scenariusza obciążenia dla pojedynczej giełdy."""

    name: str
    request_count: int
    base_latency_ms: float
    jitter_ms: float
    error_rate: float
    throttle_rate: float
    max_concurrency: int | None = None
    burst: int | None = None
    max_retries: int = 2
    backoff_base_ms: float = 20.0
    backoff_jitter_ms: float = 5.0
    timeout_ms: float = 500.0


@dataclass(slots=True)
class ExchangeMetrics:
    """Metryki uzyskane w trakcie testu obciążeniowego.

    Uwaga: `latency_ms` reprezentuje czas end-to-end pojedynczego żądania widziany
    przez runner (łącznie z retry oraz backoff), a nie wyłącznie pojedynczą latencję
    usługi dla jednego podejścia.
    """

    total_requests: int
    successes: int
    errors: int
    throttled: int
    retry_attempts: int = 0
    transient_5xx: int = 0
    throttle_events: int = 0
    transient_transport_errors: int = 0
    recovered_after_retry: int = 0
    latencies_ms: list[float] = field(default_factory=list)
    queue_wait_ms: list[float] = field(default_factory=list)

    def latency_summary(self) -> Mapping[str, float]:
        if not self.latencies_ms:
            return {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(self.latencies_ms)
        minimum = ordered[0]
        maximum = ordered[-1]
        average = statistics.fmean(ordered)
        index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
        percentile = ordered[index]
        return {
            "min_ms": round(minimum, 3),
            "max_ms": round(maximum, 3),
            "avg_ms": round(average, 3),
            "p95_ms": round(percentile, 3),
        }

    def queue_wait_summary(self) -> Mapping[str, float]:
        if not self.queue_wait_ms:
            return {"min_ms": 0.0, "max_ms": 0.0, "avg_ms": 0.0, "p95_ms": 0.0}
        ordered = sorted(self.queue_wait_ms)
        minimum = ordered[0]
        maximum = ordered[-1]
        average = statistics.fmean(ordered)
        index = max(0, min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1))
        percentile = ordered[index]
        return {
            "min_ms": round(minimum, 3),
            "max_ms": round(maximum, 3),
            "avg_ms": round(average, 3),
            "p95_ms": round(percentile, 3),
        }

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successes": self.successes,
            "errors": self.errors,
            "throttled": self.throttled,
            "retry_attempts": self.retry_attempts,
            "transient_5xx": self.transient_5xx,
            "throttle_events": self.throttle_events,
            "transient_transport_errors": self.transient_transport_errors,
            "recovered_after_retry": self.recovered_after_retry,
            "latency_ms": self.latency_summary(),
            "queue_wait_ms": self.queue_wait_summary(),
        }


@dataclass(slots=True)
class ExchangeStressConfig:
    """Pełna konfiguracja testu obciążeniowego."""

    queue_defaults: QueueDefaults
    scenarios: list[ExchangeScenario]


@dataclass(slots=True)
class RequestOutcome:
    status_code: int | None
    latency_ms: float
    queue_wait_ms: float
    retry_attempts: int = 0
    transient_5xx: int = 0
    throttle_events: int = 0
    transient_transport_errors: int = 0
    recovered_after_retry: bool = False
    error: str | None = None


@dataclass(slots=True)
class ExchangeStressResult:
    started_at: float
    finished_at: float
    metrics: Mapping[str, ExchangeMetrics]

    def duration_seconds(self) -> float:
        return max(self.finished_at - self.started_at, 0.0)

    def as_dict(self) -> Mapping[str, Any]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds(), 3),
            "exchanges": {name: metric.as_dict() for name, metric in self.metrics.items()},
        }


def _normalize_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(number, 0.0)


def _normalize_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(number, 0)


def load_config(path: Path) -> ExchangeStressConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = raw.get("defaults", {})
    queue_defaults = QueueDefaults(
        max_concurrency=_normalize_int(defaults.get("max_concurrency"), 8) or 8,
        burst=_normalize_int(defaults.get("burst"), 16) or 16,
    )
    default_request_count = _normalize_int(defaults.get("request_count"), 25) or 1
    default_latency = _normalize_float(defaults.get("base_latency_ms"), 25.0)
    default_jitter = _normalize_float(defaults.get("jitter_ms"), 5.0)
    default_error_rate = _normalize_float(defaults.get("error_rate"), 0.0)
    default_throttle_rate = _normalize_float(defaults.get("throttle_rate"), 0.0)
    default_max_retries = _normalize_int(defaults.get("max_retries"), 2)
    default_backoff_base_ms = _normalize_float(defaults.get("backoff_base_ms"), 20.0)
    default_backoff_jitter_ms = _normalize_float(defaults.get("backoff_jitter_ms"), 5.0)
    default_timeout_ms = _normalize_float(defaults.get("timeout_ms"), 500.0)

    scenarios_data = raw.get("exchanges", [])
    if not isinstance(scenarios_data, Iterable):
        raise ValueError("Sekcja 'exchanges' musi zawierać listę scenariuszy")

    scenarios: list[ExchangeScenario] = []
    for item in scenarios_data:
        if not isinstance(item, dict):
            raise ValueError("Element scenariusza musi być słownikiem")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("Każdy scenariusz musi mieć nazwę")
        scenario = ExchangeScenario(
            name=name,
            request_count=_normalize_int(item.get("request_count"), default_request_count)
            or default_request_count,
            base_latency_ms=_normalize_float(item.get("base_latency_ms"), default_latency),
            jitter_ms=_normalize_float(item.get("jitter_ms"), default_jitter),
            error_rate=_normalize_float(item.get("error_rate"), default_error_rate),
            throttle_rate=_normalize_float(item.get("throttle_rate"), default_throttle_rate),
            max_concurrency=(_normalize_int(item.get("max_concurrency"), 0) or None),
            burst=(_normalize_int(item.get("burst"), 0) or None),
            max_retries=_normalize_int(item.get("max_retries"), default_max_retries),
            backoff_base_ms=_normalize_float(
                item.get("backoff_base_ms"), default_backoff_base_ms
            ),
            backoff_jitter_ms=_normalize_float(
                item.get("backoff_jitter_ms"), default_backoff_jitter_ms
            ),
            timeout_ms=_normalize_float(item.get("timeout_ms"), default_timeout_ms),
        )
        scenarios.append(scenario)

    if not scenarios:
        raise ValueError("Konfiguracja musi zawierać co najmniej jeden scenariusz giełdy")

    return ExchangeStressConfig(queue_defaults=queue_defaults, scenarios=scenarios)


def build_mock_transport(
    scenarios: Iterable[ExchangeScenario],
    *,
    seed: int | None,
) -> httpx.MockTransport:
    scenario_map = {scenario.name: scenario for scenario in scenarios}
    base_seed = seed or 0
    scenario_seeds = {
        scenario.name: (base_seed + zlib.crc32(scenario.name.encode("utf-8"))) & 0xFFFFFFFF
        for scenario in scenarios
    }

    async def handler(request: httpx.Request) -> httpx.Response:
        name = request.url.path.lstrip("/") or request.headers.get("X-Exchange", "")
        scenario = scenario_map.get(name)
        if scenario is None:
            return httpx.Response(404, json={"error": "unknown_exchange", "exchange": name})
        scenario_seed = scenario_seeds[name]
        request_index = _normalize_int(request.headers.get("X-Request-Index"), 0)
        retry_attempt = _normalize_int(request.headers.get("X-Retry-Attempt"), 0)
        request_seed = (scenario_seed + request_index * 1009 + retry_attempt * 9176) & 0xFFFFFFFF
        rng = random.Random(request_seed)
        latency = max(0.0, rng.gauss(scenario.base_latency_ms, scenario.jitter_ms))
        await asyncio.sleep(latency / 1000.0)
        roll = rng.random()
        throttle_threshold = min(max(scenario.throttle_rate, 0.0), 1.0)
        error_probability = min(max(scenario.error_rate, 0.0), 1.0)
        error_threshold = min(throttle_threshold + error_probability, 1.0)
        if roll < throttle_threshold:
            return httpx.Response(429, json={"error": "throttled", "exchange": name})
        if roll < error_threshold:
            return httpx.Response(500, json={"error": "internal_error", "exchange": name})
        return httpx.Response(200, json={"status": "ok", "exchange": name})

    return httpx.MockTransport(handler)


async def _execute_request(
    client: httpx.AsyncClient,
    scenario: ExchangeScenario,
    index: int,
    enqueued_at: float,
    seed: int,
) -> RequestOutcome:
    start = time.perf_counter()
    queue_wait_ms = max((start - enqueued_at) * 1000.0, 0.0)
    rng_seed = (seed + zlib.crc32(f"{scenario.name}:{index}".encode("utf-8"))) & 0xFFFFFFFF
    rng = random.Random(rng_seed)
    status_code: int | None = None
    error: str | None = None
    attempts = 0
    transient_5xx = 0
    throttle_events = 0
    transient_transport_errors = 0
    retried = False
    for attempt in range(max(0, scenario.max_retries) + 1):
        attempts = attempt + 1
        try:
            response = await client.get(
                f"/{scenario.name}",
                headers={"X-Request-Index": str(index), "X-Retry-Attempt": str(attempt)},
                timeout=max(scenario.timeout_ms / 1000.0, 0.001),
            )
            status_code = response.status_code
            error = None
            if status_code == 429:
                throttle_events += 1
            if status_code in (500, 502, 503, 504):
                transient_5xx += 1
            if status_code not in (429, 500, 502, 503, 504):
                break
        except (httpx.TimeoutException, httpx.NetworkError, asyncio.TimeoutError) as exc:
            status_code = None
            error = str(exc)
            transient_transport_errors += 1

        if attempt >= max(0, scenario.max_retries):
            break
        retried = True
        jitter = rng.uniform(0.0, max(scenario.backoff_jitter_ms, 0.0))
        delay_ms = max(scenario.backoff_base_ms, 0.0) * (2**attempt) + jitter
        await asyncio.sleep(delay_ms / 1000.0)

    latency_ms = (time.perf_counter() - start) * 1000.0
    return RequestOutcome(
        status_code=status_code,
        latency_ms=latency_ms,
        queue_wait_ms=queue_wait_ms,
        retry_attempts=max(attempts - 1, 0),
        transient_5xx=transient_5xx,
        throttle_events=throttle_events,
        transient_transport_errors=transient_transport_errors,
        recovered_after_retry=bool(retried and status_code == 200),
        error=error,
    )


async def _run_scenario(
    queue: AsyncIOTaskQueue,
    client: httpx.AsyncClient,
    scenario: ExchangeScenario,
    seed: int,
) -> ExchangeMetrics:
    tasks = []
    for index in range(scenario.request_count):
        enqueued_at = time.perf_counter()
        tasks.append(
            queue.submit(
                scenario.name,
                lambda index=index, enqueued_at=enqueued_at: _execute_request(
                    client,
                    scenario,
                    index,
                    enqueued_at=enqueued_at,
                    seed=seed,
                ),
            )
        )
    outcomes = await asyncio.gather(*tasks)
    metrics = ExchangeMetrics(total_requests=len(outcomes), successes=0, errors=0, throttled=0)
    for outcome in outcomes:
        metrics.latencies_ms.append(outcome.latency_ms)
        metrics.queue_wait_ms.append(outcome.queue_wait_ms)
        metrics.retry_attempts += outcome.retry_attempts
        metrics.transient_5xx += outcome.transient_5xx
        metrics.throttle_events += outcome.throttle_events
        metrics.transient_transport_errors += outcome.transient_transport_errors
        metrics.recovered_after_retry += int(outcome.recovered_after_retry)
        if outcome.status_code == 200:
            metrics.successes += 1
        elif outcome.status_code == 429:
            metrics.throttled += 1
        else:
            metrics.errors += 1
        if outcome.error is not None:
            _LOGGER.debug("Błąd transportu podczas testu %s: %s", scenario.name, outcome.error)
    return metrics


async def run_exchange_stress(
    config: ExchangeStressConfig,
    *,
    seed: int | None = None,
) -> ExchangeStressResult:
    base_seed = seed or 0
    queue = AsyncIOTaskQueue(
        default_max_concurrency=config.queue_defaults.max_concurrency,
        default_burst=config.queue_defaults.burst,
    )
    for scenario in config.scenarios:
        if scenario.max_concurrency is not None or scenario.burst is not None:
            queue.configure_exchange(
                scenario.name,
                max_concurrency=scenario.max_concurrency,
                burst=scenario.burst,
            )

    transport = build_mock_transport(config.scenarios, seed=seed)
    started_at = time.time()
    async with httpx.AsyncClient(
        transport=transport, base_url="https://exchange-stress.local"
    ) as client:
        metrics: dict[str, ExchangeMetrics] = {}
        for scenario in config.scenarios:
            scenario_seed = (base_seed + zlib.crc32(scenario.name.encode("utf-8"))) & 0xFFFFFFFF
            metrics[scenario.name] = await _run_scenario(queue, client, scenario, scenario_seed)
    finished_at = time.time()
    return ExchangeStressResult(started_at=started_at, finished_at=finished_at, metrics=metrics)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/loadtests/exchange_stress.yml",
        help="Ścieżka do pliku konfiguracyjnego scenariusza",
    )
    parser.add_argument(
        "--output",
        help="Opcjonalna ścieżka do pliku JSON z wynikami (domyślnie logs/loadtests/exchange_stress-<timestamp>.json)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Opcjonalny seed RNG dla deterministycznych wyników",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Poziom logowania (domyślnie INFO)",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str) -> None:
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric, format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")


def _resolve_output_path(arg: str | None) -> Path:
    if arg:
        return Path(arg).expanduser()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    return Path("logs") / "loadtests" / f"exchange_stress-{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _configure_logging(args.log_level)
    config_path = Path(args.config).expanduser()
    try:
        config = load_config(config_path)
    except Exception as exc:  # pragma: no cover - walidacja konfiguracji CLI
        _LOGGER.error("Nie udało się wczytać konfiguracji %s: %s", config_path, exc)
        return 2
    output_path = _resolve_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        result = asyncio.run(run_exchange_stress(config, seed=args.seed))
    except KeyboardInterrupt:  # pragma: no cover - interakcja użytkownika
        _LOGGER.warning("Przerwano test obciążeniowy przez użytkownika")
        return 130

    payload = result.as_dict()
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _LOGGER.info("Zapisano wynik testu do %s", output_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - punkt wejścia CLI
    raise SystemExit(main())
