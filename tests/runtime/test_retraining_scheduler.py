from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Sequence

import pytest

from core.monitoring.events import (
    DataDriftDetected,
    MissingDataDetected,
    RetrainingCycleCompleted,
    RetrainingDelayInjected,
)
from core.runtime.retraining_scheduler import ChaosSettings, RetrainingScheduler


def test_missing_data_scenario_skips_training():
    captured: list[object] = []

    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=30),
        chaos=ChaosSettings(
            enabled=True,
            missing_data_frequency=1.0,
            missing_data_intensity=3,
        ),
        event_publisher=captured.append,
        random_source=random_fixed([0.0, 0.5, 0.5]),
    )

    calls: list[int] = []

    async def _train() -> str:  # pragma: no cover - nie powinien zostać wywołany
        calls.append(1)
        return "unexpected"

    outcome = run_in_loop(scheduler.run_once(_train))

    assert outcome.status == "skipped"
    assert outcome.reason == "missing_data"
    assert outcome.result is None
    assert calls == []
    assert any(isinstance(event, MissingDataDetected) for event in captured)


def test_drift_scenario_runs_training_and_emits_event():
    captured: list[object] = []

    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=15),
        chaos=ChaosSettings(
            enabled=True,
            drift_frequency=1.0,
            drift_threshold=0.4,
        ),
        event_publisher=captured.append,
        random_source=random_fixed([0.5, 0.2, 0.7]),
    )

    async def _train() -> str:
        await asyncio.sleep(0)
        return "ok"

    outcome = run_in_loop(scheduler.run_once(_train))

    assert outcome.status == "completed"
    assert outcome.result == "ok"
    drift_events = [event for event in captured if isinstance(event, DataDriftDetected)]
    assert drift_events, "powinien zostać wygenerowany event dryfu"
    assert drift_events[0].drift_threshold == pytest.approx(0.4)
    assert outcome.drift_score is not None and outcome.drift_score >= drift_events[0].drift_threshold
    completed = [event for event in captured if isinstance(event, RetrainingCycleCompleted)]
    assert completed, "powinien zostać wygenerowany raport zakończenia retrainingu"
    assert completed[0].status == "completed"
    assert completed[0].duration_seconds >= 0.0


def test_delay_scenario_waits_before_training():
    captured: list[object] = []
    fake_now = datetime(2024, 5, 1, 12, 0, tzinfo=timezone.utc)

    async def _train() -> str:
        return "ok"

    scheduler = RetrainingScheduler(
        interval=timedelta(minutes=10),
        clock=lambda: fake_now,
        chaos=ChaosSettings(
            enabled=True,
            delay_frequency=1.0,
            delay_min_seconds=0.01,
            delay_max_seconds=0.02,
        ),
        event_publisher=captured.append,
        random_source=random_fixed([0.3, 0.1, 0.1]),
    )

    loop = asyncio.new_event_loop()
    try:
        start = loop.time()
        outcome = loop.run_until_complete(scheduler.run_once(_train))
        duration = loop.time() - start
    finally:
        loop.close()

    assert outcome.status == "completed"
    assert outcome.delay_seconds >= 0.01
    assert duration >= 0.01
    assert any(isinstance(event, RetrainingDelayInjected) for event in captured)
    completed = [event for event in captured if isinstance(event, RetrainingCycleCompleted)]
    assert completed, "po zakończeniu retrainingu powinno pojawić się zdarzenie podsumowujące"
    assert completed[0].metadata == {"delay_seconds": pytest.approx(outcome.delay_seconds)}


def random_fixed(values: Sequence[float]):
    class _Random:
        def __init__(self, seq: Sequence[float]):
            self._iterator = iter(seq)

        def random(self) -> float:
            return next(self._iterator)

        def uniform(self, start: float, end: float) -> float:
            return start + (end - start) * next(self._iterator)

    return _Random(values)


def run_in_loop(awaitable):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(awaitable)
    finally:
        loop.close()
