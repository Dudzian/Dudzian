import asyncio

import tests._pathbootstrap  # noqa: F401  # pylint: disable=unused-import

from datetime import datetime, timezone

import pytest

from bot_core.runtime.scheduler import AsyncIOTaskQueue, CyclicTaskScheduler, ScheduledTask


def test_async_io_task_queue_limits_concurrency() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=2, default_burst=4)
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def job() -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.05)
            async with lock:
                active -= 1

        await asyncio.gather(*(queue.submit("binance", job) for _ in range(6)))
        assert max_active <= 2

    asyncio.run(runner())


def test_async_io_task_queue_respects_burst_limits() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=1, default_burst=1)
        started: list[str] = []
        release = asyncio.Event()

        async def job(name: str) -> str:
            started.append(name)
            await release.wait()
            return name

        first = asyncio.create_task(queue.submit("kraken", lambda: job("first")))
        await asyncio.sleep(0.05)
        assert started == ["first"]

        second = asyncio.create_task(queue.submit("kraken", lambda: job("second")))
        await asyncio.sleep(0.05)
        assert started == ["first"]

        release.set()
        results = await asyncio.gather(first, second)
        assert results == ["first", "second"]
        assert started == ["first", "second"]

    asyncio.run(runner())


def test_async_io_task_queue_supports_per_exchange_configuration() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=3, default_burst=3)
        queue.configure_exchange("nowa_gielda_spot", max_concurrency=1, burst=2)
        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def job() -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1

        await asyncio.gather(*(queue.submit("nowa_gielda_spot", job) for _ in range(5)))
        assert max_active <= 1

    asyncio.run(runner())


def test_cyclic_scheduler_maps_regular_exception_to_failed_result() -> None:
    async def runner() -> None:
        scheduler = CyclicTaskScheduler(clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))

        async def job(_started_at: datetime) -> None:
            raise RuntimeError("boom")

        scheduler.register(ScheduledTask(name="failing", callback=job))
        results = await scheduler.run_pending()

        assert len(results) == 1
        result = results[0]
        assert result.name == "failing"
        assert result.success is False
        assert isinstance(result.error, RuntimeError)

    asyncio.run(runner())


def test_cyclic_scheduler_propagates_cancelled_error() -> None:
    async def runner() -> None:
        scheduler = CyclicTaskScheduler(clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc))

        async def job(_started_at: datetime) -> None:
            raise asyncio.CancelledError()

        scheduler.register(ScheduledTask(name="cancelled", callback=job))

        with pytest.raises(asyncio.CancelledError):
            await scheduler.run_pending()

    asyncio.run(runner())
