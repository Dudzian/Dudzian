"""Scheduler retrainingu z obsługą scenariuszy chaosowych."""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Mapping

from core.monitoring.events import (
    DataDriftDetected,
    EventPublisher,
    MissingDataDetected,
    RetrainingCycleCompleted,
    RetrainingDelayInjected,
)


@dataclass(slots=True)
class ChaosSettings:
    """Konfiguracja scenariuszy chaosowych dla retrainingu."""

    enabled: bool = False
    missing_data_frequency: float = 0.0
    missing_data_intensity: int = 1
    drift_frequency: float = 0.0
    drift_threshold: float = 0.0
    delay_frequency: float = 0.0
    delay_min_seconds: float = 0.0
    delay_max_seconds: float = 0.0

    @classmethod
    def from_mapping(cls, data: Mapping[str, object] | None) -> "ChaosSettings":
        from collections.abc import Mapping as ABCMapping

        if not data:
            return cls()
        if not isinstance(data, ABCMapping):  # pragma: no cover - ochrona przed złym typem
            raise TypeError("Konfiguracja chaosu musi być mapą klucz→wartość")
        return cls(
            enabled=bool(data.get("enabled", False)),
            missing_data_frequency=float(data.get("missing_data_frequency", 0.0)),
            missing_data_intensity=max(1, int(data.get("missing_data_intensity", 1))),
            drift_frequency=float(data.get("drift_frequency", 0.0)),
            drift_threshold=max(0.0, float(data.get("drift_threshold", 0.0))),
            delay_frequency=float(data.get("delay_frequency", 0.0)),
            delay_min_seconds=max(0.0, float(data.get("delay_min_seconds", 0.0))),
            delay_max_seconds=max(
                float(data.get("delay_min_seconds", 0.0)),
                float(data.get("delay_max_seconds", 0.0)),
            ),
        )

    def should_trigger(self, frequency: float, rng: random.Random) -> bool:
        if not self.enabled:
            return False
        return max(0.0, frequency) > 0.0 and rng.random() < min(1.0, frequency)

    def choose_delay(self, rng: random.Random) -> float:
        if self.delay_max_seconds <= 0:
            return 0.0
        start = min(self.delay_min_seconds, self.delay_max_seconds)
        end = max(self.delay_min_seconds, self.delay_max_seconds)
        return rng.uniform(start, end)


@dataclass(slots=True)
class RetrainingRunOutcome:
    """Opis skutków pojedynczego przebiegu retrainingu."""

    status: str
    result: object | None
    reason: str | None
    delay_seconds: float
    drift_score: float | None
    events: tuple[object, ...]


class RetrainingScheduler:
    """Prosty harmonogram retrainingu z możliwością symulacji chaosu."""

    def __init__(
        self,
        *,
        interval: timedelta,
        clock: Callable[[], datetime] | None = None,
        chaos: ChaosSettings | None = None,
        event_publisher: EventPublisher | None = None,
        logger: logging.Logger | None = None,
        random_source: random.Random | None = None,
    ) -> None:
        if interval <= timedelta(0):  # pragma: no cover - chroni konfigurację
            raise ValueError("Interval retrainingu musi być dodatni")
        self._interval = interval
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._chaos = chaos or ChaosSettings()
        self._publish_event = event_publisher or (lambda event: None)
        self._logger = logger or logging.getLogger(__name__)
        self._rng = random_source or random.Random()
        self._last_run: datetime | None = None
        self._next_run: datetime | None = None

    @property
    def last_run(self) -> datetime | None:
        return self._last_run

    @property
    def next_run(self) -> datetime | None:
        return self._next_run

    def should_run(self, *, now: datetime | None = None) -> bool:
        moment = now or self._clock()
        if self._next_run is None:
            return True
        return moment >= self._next_run

    async def run_once(self, retrain: Callable[[], Awaitable[object]]) -> RetrainingRunOutcome:
        now = self._clock()
        if not self.should_run(now=now):
            return RetrainingRunOutcome(
                status="skipped",
                result=None,
                reason="not_due",
                delay_seconds=0.0,
                drift_score=None,
                events=(),
            )

        events: list[object] = []
        delay_seconds = 0.0
        drift_score: float | None = None

        if self._chaos.should_trigger(self._chaos.delay_frequency, self._rng):
            delay_seconds = self._chaos.choose_delay(self._rng)
            if delay_seconds > 0:
                delay_event = RetrainingDelayInjected(reason="chaos_delay", delay_seconds=delay_seconds)
                events.append(delay_event)
                self._publish_event(delay_event)
                self._logger.warning("Retraining opóźniony o %.3fs z powodu chaosu", delay_seconds)
                await asyncio.sleep(delay_seconds)

        if self._chaos.should_trigger(self._chaos.missing_data_frequency, self._rng):
            missing_event = MissingDataDetected(
                source="retraining_pipeline",
                missing_batches=self._chaos.missing_data_intensity,
            )
            events.append(missing_event)
            self._publish_event(missing_event)
            self._logger.error(
                "Chaos: brak danych (%s paczek), retraining zostaje pominięty", missing_event.missing_batches
            )
            self._last_run = now
            self._next_run = now + self._interval
            return RetrainingRunOutcome(
                status="skipped",
                result=None,
                reason="missing_data",
                delay_seconds=delay_seconds,
                drift_score=None,
                events=tuple(events),
            )

        if self._chaos.should_trigger(self._chaos.drift_frequency, self._rng):
            base = max(self._chaos.drift_threshold, 0.0)
            jitter = self._rng.uniform(0.05, 0.25)
            drift_score = base * (1.0 + jitter) if base > 0 else jitter
            drift_event = DataDriftDetected(
                source="retraining_pipeline",
                drift_score=drift_score,
                drift_threshold=self._chaos.drift_threshold,
            )
            events.append(drift_event)
            self._publish_event(drift_event)
            self._logger.warning(
                "Chaos: wykryto dryf danych score=%.4f threshold=%.4f", drift_score, self._chaos.drift_threshold
            )

        started_monotonic = self._loop_time()
        result = await retrain()
        duration_seconds = self._loop_time() - started_monotonic
        completed_event = RetrainingCycleCompleted(
            source="retraining_scheduler",
            status="completed",
            duration_seconds=max(0.0, duration_seconds),
            drift_score=drift_score,
            metadata={"delay_seconds": delay_seconds} if delay_seconds > 0 else None,
        )
        events.append(completed_event)
        self._publish_event(completed_event)
        self._last_run = now
        self._next_run = now + self._interval
        return RetrainingRunOutcome(
            status="completed",
            result=result,
            reason=None,
            delay_seconds=delay_seconds,
            drift_score=drift_score,
            events=tuple(events),
        )

    @staticmethod
    def _loop_time() -> float:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return time.monotonic()
        return loop.time()


__all__ = [
    "ChaosSettings",
    "RetrainingRunOutcome",
    "RetrainingScheduler",
]
