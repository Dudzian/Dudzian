"""Utilities for pinning strategy workers and throttling runtime feeds."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class StrategyAffinity:
    """Mapping between strategy names and CPU cores."""

    strategy: str
    cores: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DataFeedThrottlePolicy:
    """Throttle configuration for a runtime data feed."""

    feed: str
    rate_per_second: float
    burst: int = 1


class DataFeedThrottler:
    """Token bucket rate limiter supporting sync and async code paths."""

    def __init__(self, rate_per_second: float, *, burst: int = 1) -> None:
        self._rate = max(float(rate_per_second), 0.0)
        self._capacity = max(int(burst), 1)
        self._tokens = float(self._capacity)
        self._lock = threading.Lock()
        self._timestamp = time.perf_counter()

    def _reserve(self) -> float:
        with self._lock:
            now = time.perf_counter()
            elapsed = now - self._timestamp
            self._timestamp = now
            if self._rate > 0.0:
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            else:
                self._tokens = float(self._capacity)
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            if self._rate <= 0.0:
                return 0.0
            missing = 1.0 - self._tokens
            wait_time = missing / self._rate
            self._tokens = 0.0
            return max(wait_time, 0.0)

    def blocking_acquire(self) -> None:
        while True:
            wait_time = self._reserve()
            if wait_time <= 0.0:
                return
            time.sleep(wait_time)

    async def acquire(self) -> None:
        while True:
            wait_time = self._reserve()
            if wait_time <= 0.0:
                return
            await asyncio.sleep(wait_time)


class RuntimeResourceManager:
    """Applies resource policies to multi-strategy runtime instances."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        self._logger = logger or LOGGER
        self._throttlers: dict[str, DataFeedThrottler] = {}

    # ------------------------------------------------------------------
    # CPU affinity management
    # ------------------------------------------------------------------
    def pin_strategies(self, runtime: object, affinities: Iterable[StrategyAffinity]) -> None:
        plan = [entry for entry in affinities if entry.strategy]
        if not plan:
            return
        workers = self._resolve_strategy_workers(runtime)
        if not workers:
            cores = sorted({core for entry in plan for core in entry.cores})
            self._logger.debug(
                "Runtime does not expose strategy workers; applying process-level affinity",
                extra={"cores": cores},
            )
            self._set_affinity(os.getpid(), cores)
            return
        for entry in plan:
            worker = workers.get(entry.strategy)
            if worker is None:
                self._logger.debug(
                    "Strategy worker not found; skipping affinity pin",
                    extra={"strategy": entry.strategy},
                )
                continue
            pid = self._resolve_worker_pid(worker)
            if pid is None:
                self._logger.debug(
                    "Could not resolve worker PID; skipping affinity pin",
                    extra={"strategy": entry.strategy},
                )
                continue
            self._set_affinity(pid, entry.cores)

    def _resolve_strategy_workers(self, runtime: object) -> Mapping[str, object]:
        candidate_attrs = (
            "strategy_workers",
            "strategy_threads",
            "strategy_processes",
            "strategy_runners",
        )
        for attr in candidate_attrs:
            value = getattr(runtime, attr, None)
            if isinstance(value, Mapping):
                return value
        scheduler = getattr(runtime, "scheduler", None)
        if scheduler is not None and scheduler is not runtime:
            value = getattr(scheduler, "strategy_workers", None)
            if isinstance(value, Mapping):
                return value
        return {}

    def _resolve_worker_pid(self, worker: object) -> int | None:
        for attr in ("pid", "process_id", "ident", "native_id"):
            value = getattr(worker, attr, None)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    def _set_affinity(self, target_pid: int, cores: Sequence[int]) -> None:
        unique = sorted({int(core) for core in cores if core >= 0})
        if not unique:
            return
        try:
            os.sched_setaffinity(target_pid, set(unique))  # type: ignore[attr-defined]
            self._logger.info(
                "Pinned worker to CPU cores", extra={"pid": target_pid, "cores": unique}
            )
        except AttributeError:  # pragma: no cover - platform dependent
            self._logger.warning(
                "CPU affinity not supported on this platform", extra={"pid": target_pid}
            )
        except PermissionError:  # pragma: no cover - requires elevated permissions
            self._logger.warning(
                "Insufficient permissions to set CPU affinity", extra={"pid": target_pid}
            )
        except OSError as exc:  # pragma: no cover - defensive guard
            self._logger.warning(
                "Failed to set CPU affinity", extra={"pid": target_pid, "error": str(exc)}
            )

    # ------------------------------------------------------------------
    # Feed throttling management
    # ------------------------------------------------------------------
    def apply_feed_throttling(
        self,
        runtime: object,
        policies: Iterable[DataFeedThrottlePolicy],
    ) -> None:
        policies = list(policies)
        if not policies:
            return
        for policy in policies:
            feed = self._resolve_feed(runtime, policy.feed)
            if feed is None:
                self._logger.debug(
                    "Runtime feed not found; skipping throttle",
                    extra={"feed": policy.feed},
                )
                continue
            throttler = DataFeedThrottler(policy.rate_per_second, burst=policy.burst)
            self._wrap_feed(feed, throttler)
            self._throttlers[policy.feed] = throttler
            self._logger.info(
                "Enabled feed throttling",
                extra={
                    "feed": policy.feed,
                    "rate_per_second": policy.rate_per_second,
                    "burst": policy.burst,
                },
            )

    def _resolve_feed(self, runtime: object, name: str) -> object | None:
        candidates = (
            getattr(runtime, "data_feeds", None),
            getattr(runtime, "feeds", None),
            getattr(runtime, "market_feeds", None),
        )
        for candidate in candidates:
            if isinstance(candidate, Mapping) and name in candidate:
                return candidate[name]
        resolver = getattr(runtime, "get_feed", None)
        if callable(resolver):
            try:
                return resolver(name)
            except Exception:  # pragma: no cover - defensive guard
                self._logger.debug("Feed resolver failed", exc_info=True)
        return None

    def _wrap_feed(self, feed: object, throttler: DataFeedThrottler) -> None:
        for attribute in ("fetch", "read", "pull"):
            handler = getattr(feed, attribute, None)
            if handler is None or not callable(handler):
                continue
            if asyncio.iscoroutinefunction(handler):
                async def async_wrapper(*args, _h=handler, **kwargs):
                    await throttler.acquire()
                    return await _h(*args, **kwargs)

                setattr(feed, attribute, async_wrapper)
            else:
                def sync_wrapper(*args, _h=handler, **kwargs):
                    throttler.blocking_acquire()
                    return _h(*args, **kwargs)

                setattr(feed, attribute, sync_wrapper)
            return


def parse_strategy_affinity_specs(raw: Sequence[str] | None) -> list[StrategyAffinity]:
    specs: list[StrategyAffinity] = []
    if not raw:
        return specs
    for entry in raw:
        if not entry:
            continue
        strategy, _, cores_raw = entry.partition("=")
        strategy = strategy.strip()
        if not strategy or not cores_raw:
            continue
        try:
            cores = tuple(sorted({int(core.strip()) for core in cores_raw.split(",") if core.strip()}))
        except ValueError:
            continue
        if not cores:
            continue
        specs.append(StrategyAffinity(strategy=strategy, cores=cores))
    return specs


def parse_feed_throttle_specs(raw: Sequence[str] | None) -> list[DataFeedThrottlePolicy]:
    specs: list[DataFeedThrottlePolicy] = []
    if not raw:
        return specs
    for entry in raw:
        if not entry:
            continue
        feed, _, payload = entry.partition("=")
        feed = feed.strip()
        if not feed or not payload:
            continue
        rate_part, _, burst_part = payload.partition(":")
        try:
            rate = float(rate_part)
        except ValueError:
            continue
        burst = 1
        if burst_part:
            try:
                burst = int(burst_part)
            except ValueError:
                burst = 1
        specs.append(DataFeedThrottlePolicy(feed=feed, rate_per_second=rate, burst=max(burst, 1)))
    return specs
