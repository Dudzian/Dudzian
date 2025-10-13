"""Narzędzia do testów obciążeniowych scheduler-a multi-strategy."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Sequence

from bot_core.runtime.journal import InMemoryTradingDecisionJournal
from bot_core.runtime.multi_strategy_scheduler import MultiStrategyScheduler
from bot_core.runtime.resource_monitor import (
    ResourceBudgets,
    ResourceSample,
    evaluate_resource_sample,
)
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal

__all__ = [
    "LoadTestSettings",
    "LoadTestResult",
    "execute_scheduler_load_test",
]


@dataclass(slots=True)
class LoadTestSettings:
    iterations: int = 25
    schedules: int = 3
    signals_per_snapshot: int = 4
    simulated_latency_ms: float = 2.0
    jitter_ms: float = 0.5
    cpu_budget_percent: float = 70.0
    memory_budget_mb: float = 3072.0


@dataclass(slots=True)
class LoadTestResult:
    schedules: int
    iterations: int
    avg_latency_ms: float
    max_latency_ms: float
    jitter_ms: float
    signals_emitted: int
    resource_status: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schedules": self.schedules,
            "iterations": self.iterations,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "jitter_ms": self.jitter_ms,
            "signals_emitted": self.signals_emitted,
            "resource_status": self.resource_status,
        }


class _LoadTestStrategy(StrategyEngine):
    def __init__(self, *, latency_ms: float, jitter_ms: float, signals: int) -> None:
        self._latency_ms = latency_ms
        self._jitter_ms = jitter_ms
        self._signals = max(1, signals)
        self._counter = 0

    def warm_up(self, history: Sequence[MarketSnapshot]) -> None:
        return None

    def on_data(self, snapshot: MarketSnapshot) -> Sequence[StrategySignal]:
        # Symulujemy niewielką złożoność CPU
        spins = int(self._latency_ms * 1000)
        accumulator = 0
        for _ in range(spins):
            accumulator += 1
        self._counter += 1
        base_signal = StrategySignal(
            symbol=snapshot.symbol,
            side="buy",
            confidence=0.8,
            metadata={
                "primary_exchange": "binance",
                "secondary_exchange": "kraken",
                "instrument_type": "spot",
                "data_feed": "load_test",
                "risk_budget_bucket": "balanced",
            },
        )
        return [base_signal for _ in range(self._signals)]


class _StaticFeed:
    def __init__(self, symbols: Iterable[str]) -> None:
        self._symbols = list(symbols)

    def load_history(self, strategy_name: str, bars: int) -> Sequence[MarketSnapshot]:
        return []

    def fetch_latest(self, strategy_name: str) -> Sequence[MarketSnapshot]:
        return [
            MarketSnapshot(
                symbol=symbol,
                timestamp=0,
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.5,
                volume=1_000.0,
            )
            for symbol in self._symbols
        ]


class _CountingSink:
    def __init__(self) -> None:
        self.submissions: int = 0

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: object,
        signals: Sequence[StrategySignal],
    ) -> None:
        self.submissions += len(signals)

    def reset(self) -> None:
        self.submissions = 0


async def _run_iteration(
    scheduler: MultiStrategyScheduler,
    *,
    sink: _CountingSink,
    timestamps: Sequence[float],
) -> list[float]:
    durations: list[float] = []
    for ts in timestamps:
        start = perf_counter()
        await scheduler.run_once()
        durations.append((perf_counter() - start) * 1000.0)
        sink.reset()
    return durations


def execute_scheduler_load_test(settings: LoadTestSettings) -> LoadTestResult:
    journal = InMemoryTradingDecisionJournal()
    scheduler = MultiStrategyScheduler(
        environment="load_test",
        portfolio="paper",
        decision_journal=journal,
    )
    sink = _CountingSink()
    feed = _StaticFeed(["BTCUSDT", "ETHUSDT"])

    for idx in range(settings.schedules):
        strategy = _LoadTestStrategy(
            latency_ms=settings.simulated_latency_ms,
            jitter_ms=settings.jitter_ms,
            signals=settings.signals_per_snapshot,
        )
        scheduler.register_schedule(
            name=f"schedule_{idx}",
            strategy_name=f"strategy_{idx}",
            strategy=strategy,
            feed=feed,
            sink=sink,
            cadence_seconds=5,
            max_drift_seconds=1,
            warmup_bars=0,
            risk_profile="balanced",
            max_signals=settings.signals_per_snapshot,
        )

    timestamps = [0.0 for _ in range(settings.iterations)]
    durations = asyncio.run(_run_iteration(scheduler, sink=sink, timestamps=timestamps))

    avg_latency = sum(durations) / len(durations) if durations else 0.0
    max_latency = max(durations) if durations else 0.0
    jitter = max_latency - (min(durations) if durations else 0.0)
    total_signals = settings.iterations * settings.schedules * settings.signals_per_snapshot

    budgets = ResourceBudgets(
        cpu_percent=settings.cpu_budget_percent,
        memory_mb=settings.memory_budget_mb,
        io_read_mb_s=120.0,
        io_write_mb_s=80.0,
    )
    sample = ResourceSample(
        cpu_percent=settings.simulated_latency_ms * settings.schedules,
        memory_mb=settings.memory_budget_mb * 0.4,
        io_read_mb_s=40.0,
        io_write_mb_s=20.0,
    )
    evaluation = evaluate_resource_sample(budgets, sample)

    return LoadTestResult(
        schedules=settings.schedules,
        iterations=settings.iterations,
        avg_latency_ms=avg_latency,
        max_latency_ms=max_latency,
        jitter_ms=jitter,
        signals_emitted=total_signals,
        resource_status=evaluation.status,
    )
