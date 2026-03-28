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


def test_async_io_task_queue_rejects_reconfigure_during_inflight_work() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=1, default_burst=2)
        started: list[str] = []
        release = asyncio.Event()

        async def job(name: str) -> str:
            started.append(name)
            await release.wait()
            return name

        first = asyncio.create_task(queue.submit("binance", lambda: job("first")))
        await asyncio.sleep(0.05)
        assert started == ["first"]

        second = asyncio.create_task(queue.submit("binance", lambda: job("second")))
        await asyncio.sleep(0.05)

        with pytest.raises(RuntimeError, match="in-flight"):
            queue.configure_exchange("binance", max_concurrency=2, burst=3)

        release.set()
        results = await asyncio.gather(first, second)
        assert results == ["first", "second"]
        assert started == ["first", "second"]

    asyncio.run(runner())


def test_async_io_task_queue_rejected_reconfigure_does_not_mutate_state() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=3, default_burst=4)
        queue.configure_exchange("binance", max_concurrency=1, burst=2)
        original_state = queue._queues["binance"]
        started: list[str] = []
        release = asyncio.Event()

        async def job(name: str) -> str:
            started.append(name)
            await release.wait()
            return name

        first = asyncio.create_task(queue.submit("binance", lambda: job("first")))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(queue.submit("binance", lambda: job("second")))
        await asyncio.sleep(0.05)

        with pytest.raises(RuntimeError, match="in-flight"):
            queue.configure_exchange("binance", max_concurrency=2, burst=3)

        same_state = queue._queues["binance"]
        assert same_state is original_state
        assert same_state.limits.max_concurrency == 1
        assert same_state.limits.burst == 2
        assert same_state.pending == 2
        assert started == ["first"]

        release.set()
        results = await asyncio.gather(first, second)
        assert results == ["first", "second"]
        assert started == ["first", "second"]
        assert queue._queues["binance"] is original_state
        assert queue._queues["binance"].pending == 0

    asyncio.run(runner())


def test_async_io_task_queue_reconfigure_after_drain_applies_new_limits() -> None:
    async def runner() -> None:
        queue = AsyncIOTaskQueue(default_max_concurrency=3, default_burst=4)
        queue.configure_exchange("binance", max_concurrency=1, burst=2)
        original_state = queue._queues["binance"]
        release = asyncio.Event()

        async def blocking_job() -> str:
            await release.wait()
            return "done"

        first = asyncio.create_task(queue.submit("binance", blocking_job))
        second = asyncio.create_task(queue.submit("binance", blocking_job))
        await asyncio.sleep(0.05)
        with pytest.raises(RuntimeError, match="in-flight"):
            queue.configure_exchange("binance", max_concurrency=2, burst=3)
        release.set()
        await asyncio.gather(first, second)

        queue.configure_exchange("binance", max_concurrency=2, burst=3)
        reconfigured_state = queue._queues["binance"]
        assert reconfigured_state is not original_state
        assert reconfigured_state.pending == 0
        assert reconfigured_state.limits.max_concurrency == 2
        assert reconfigured_state.limits.burst == 3

        active = 0
        max_active = 0
        lock = asyncio.Lock()

        async def concurrency_job() -> None:
            nonlocal active, max_active
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1

        await asyncio.gather(*(queue.submit("binance", concurrency_job) for _ in range(5)))
        assert max_active == 2

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
