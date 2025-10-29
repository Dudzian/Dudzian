from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, timezone

from bot_core.runtime.scheduler import (
    CyclicTaskScheduler,
    ScheduleWindow,
    ScheduledTask,
)


class FakeClock:
    def __init__(self, start: datetime) -> None:
        self.current = start

    def now(self) -> datetime:
        return self.current

    def advance(self, delta: timedelta) -> None:
        self.current += delta


async def _async_test_priority(clock: FakeClock) -> None:
    scheduler = CyclicTaskScheduler(clock=clock.now)

    executed: list[str] = []

    async def make_callback(name: str):
        async def _callback(_timestamp: datetime) -> dict[str, object]:
            executed.append(name)
            return {"name": name}

        return _callback

    scheduler.register(
        ScheduledTask(
            name="low-priority",
            priority=0,
            cooldown=timedelta(seconds=5),
            callback=await make_callback("low"),
        )
    )
    scheduler.register(
        ScheduledTask(
            name="high-priority",
            priority=10,
            cooldown=timedelta(seconds=0),
            callback=await make_callback("high"),
        )
    )

    results = await scheduler.run_pending()
    assert [result.name for result in results] == ["high-priority", "low-priority"]
    assert executed == ["high", "low"]

    executed.clear()
    clock.advance(timedelta(seconds=4))
    results = await scheduler.run_pending()
    assert [result.name for result in results] == ["high-priority"]

    clock.advance(timedelta(seconds=2))
    results = await scheduler.run_pending()
    assert [result.name for result in results] == ["high-priority", "low-priority"]


def test_scheduler_respects_priority_and_cooldown() -> None:
    clock = FakeClock(datetime(2024, 5, 17, 10, tzinfo=timezone.utc))
    asyncio.run(_async_test_priority(clock))


async def _async_test_windows(clock: FakeClock, triggered: list[datetime]) -> None:
    scheduler = CyclicTaskScheduler(clock=clock.now)

    def callback(moment: datetime) -> dict[str, object]:
        triggered.append(moment)
        return {"timestamp": moment.isoformat()}

    scheduler.register(
        ScheduledTask(
            name="windowed",
            priority=1,
            cooldown=timedelta(minutes=15),
            window=ScheduleWindow(start=time(9, 0), end=time(10, 0)),
            callback=callback,
        )
    )

    await scheduler.run_pending()
    clock.advance(timedelta(hours=1))
    await scheduler.run_pending()
    clock.advance(timedelta(minutes=5))
    await scheduler.run_pending()
    clock.advance(timedelta(minutes=20))
    await scheduler.run_pending()
    clock.advance(timedelta(hours=2))
    await scheduler.run_pending()


def test_scheduler_respects_windows() -> None:
    start_time = datetime(2024, 5, 17, 8, tzinfo=timezone.utc)
    clock = FakeClock(start_time)
    triggered: list[datetime] = []
    asyncio.run(_async_test_windows(clock, triggered))
    assert len(triggered) == 2
