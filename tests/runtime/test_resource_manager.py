"""Testy dla modułu zarządzania zasobami runtime."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from scripts.runtime import (
    DataFeedThrottlePolicy,
    RuntimeResourceManager,
    StrategyAffinity,
    parse_feed_throttle_specs,
    parse_strategy_affinity_specs,
)
import scripts.runtime.resource_manager as resource_manager


@dataclass
class _DummyWorker:
    pid: int


class _DummyRuntime:
    def __init__(self) -> None:
        self.data_feeds: dict[str, Any] = {}
        self.strategy_workers: dict[str, Any] = {}


def test_parse_strategy_affinity_specs_normalises_and_filters() -> None:
    specs = parse_strategy_affinity_specs([
        "alpha=0,1, 2",
        "beta=",
        "gamma=3,3",
        "",
    ])
    assert specs == [
        StrategyAffinity(strategy="alpha", cores=(0, 1, 2)),
        StrategyAffinity(strategy="gamma", cores=(3,)),
    ]


def test_parse_feed_throttle_specs_supports_burst() -> None:
    specs = parse_feed_throttle_specs([
        "ticks=5:3",
        "orders=2",
        "",
    ])
    assert specs == [
        DataFeedThrottlePolicy(feed="ticks", rate_per_second=5.0, burst=3),
        DataFeedThrottlePolicy(feed="orders", rate_per_second=2.0, burst=1),
    ]


def test_pin_strategies_uses_sched_setaffinity(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _DummyRuntime()
    runtime.strategy_workers["alpha"] = _DummyWorker(pid=3210)

    calls: list[tuple[int, set[int]]] = []

    def fake_sched_setaffinity(pid: int, cores: set[int]) -> None:
        calls.append((pid, cores))

    monkeypatch.setattr(resource_manager.os, "sched_setaffinity", fake_sched_setaffinity)

    manager = RuntimeResourceManager()
    manager.pin_strategies(
        runtime,
        [
            StrategyAffinity(strategy="alpha", cores=(2, 3, 2)),
            StrategyAffinity(strategy="missing", cores=(4,)),
        ],
    )

    assert calls == [(3210, {2, 3})]


def test_apply_feed_throttling_wraps_sync_method(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = _DummyRuntime()

    class DummyFeed:
        def __init__(self) -> None:
            self.calls = 0

        def fetch(self) -> str:
            self.calls += 1
            return "ok"

    feed = DummyFeed()
    runtime.data_feeds["orders"] = feed

    blocking_calls: list[str] = []

    def fake_blocking_acquire(self: resource_manager.DataFeedThrottler) -> None:  # noqa: ANN001
        blocking_calls.append("acquire")

    monkeypatch.setattr(
        resource_manager.DataFeedThrottler,
        "blocking_acquire",
        fake_blocking_acquire,
    )

    manager = RuntimeResourceManager()
    manager.apply_feed_throttling(
        runtime,
        [DataFeedThrottlePolicy(feed="orders", rate_per_second=100.0, burst=2)],
    )

    assert runtime.data_feeds["orders"].fetch() == "ok"
    assert feed.calls == 1
    assert blocking_calls == ["acquire"]
