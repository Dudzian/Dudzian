import asyncio
import pytest

from bot_core.exchanges.streaming import StreamBatch
from bot_core.runtime.pipeline import consume_stream, consume_stream_async


class _ClosableStream:
    def __init__(self, batches: list[StreamBatch]):
        self._iterator = iter(batches)
        self.close_calls = 0

    def __iter__(self) -> "_ClosableStream":  # noqa: D401
        return self

    def __next__(self) -> StreamBatch:  # noqa: D401
        return next(self._iterator)

    def close(self) -> None:  # noqa: D401
        self.close_calls += 1


class _DualCloseSyncIterator:
    def __init__(self, stream: "_DualCloseSyncStream") -> None:
        self._iterator = iter(stream._batches)
        self.close = stream.close

    def __iter__(self) -> "_DualCloseSyncIterator":
        return self

    def __next__(self) -> StreamBatch:
        return next(self._iterator)


class _DualCloseSyncStream:
    def __init__(self, batches: list[StreamBatch]):
        self._batches = batches
        self.close_calls = 0

    def __iter__(self) -> _DualCloseSyncIterator:
        return _DualCloseSyncIterator(self)

    def close(self) -> None:
        self.close_calls += 1


class _DualCloseAsyncIterator:
    def __init__(self, stream: "_DualCloseAsyncStream") -> None:
        self._iterator = iter(stream._batches)
        self.aclose = stream.aclose

    def __aiter__(self) -> "_DualCloseAsyncIterator":
        return self

    async def __anext__(self) -> StreamBatch:
        try:
            return next(self._iterator)
        except StopIteration as exc:  # pragma: no cover - standard async iterator contract
            raise StopAsyncIteration from exc


class _DualCloseAsyncStream:
    def __init__(self, batches: list[StreamBatch]):
        self._batches = batches
        self.aclose_calls = 0

    def __aiter__(self) -> _DualCloseAsyncIterator:
        return _DualCloseAsyncIterator(self)

    async def aclose(self) -> None:
        self.aclose_calls += 1


def test_consume_stream_processes_events_and_heartbeats() -> None:
    batches = [
        StreamBatch(channel="ticker", events=({"price": 101.0},), received_at=0.0),
        StreamBatch(channel="ticker", events=(), received_at=5.0, heartbeat=True),
    ]

    processed: list[StreamBatch] = []
    heartbeats: list[float] = []

    def stream():
        for batch in batches:
            yield batch

    consume_stream(
        stream(),
        handle_batch=processed.append,
        on_heartbeat=heartbeats.append,
        heartbeat_interval=1.0,
        idle_timeout=30.0,
        stop_condition=lambda: len(processed) >= 1 and len(heartbeats) >= 1,
        clock=lambda: 0.0,
    )

    assert processed and processed[0].events[0]["price"] == 101.0
    assert heartbeats and pytest.approx(heartbeats[0], abs=1e-6) == 5.0


def test_consume_stream_raises_on_missing_data() -> None:
    def stream():
        yield StreamBatch(channel="orders", events=(), received_at=0.0, heartbeat=True)
        yield StreamBatch(channel="orders", events=(), received_at=120.0, heartbeat=True)

    with pytest.raises(TimeoutError):
        consume_stream(
            stream(),
            handle_batch=lambda batch: None,
            on_heartbeat=lambda ts: None,
            heartbeat_interval=10.0,
            idle_timeout=60.0,
            clock=lambda: 0.0,
        )


def test_consume_stream_closes_stream_after_stop_condition() -> None:
    batches = [StreamBatch(channel="ticker", events=({"price": 10.0},), received_at=0.0)]
    stream = _ClosableStream(batches)

    processed: list[StreamBatch] = []

    consume_stream(
        stream,
        handle_batch=lambda batch: processed.append(batch),
        stop_condition=lambda: bool(processed),
        heartbeat_interval=1.0,
        idle_timeout=30.0,
        clock=lambda: 0.0,
    )

    assert processed and processed[0].events[0]["price"] == 10.0
    assert stream.close_calls == 1


def test_consume_stream_closes_stream_on_exception() -> None:
    batches = [StreamBatch(channel="orders", events=({"id": 1},), received_at=0.0)]
    stream = _ClosableStream(batches)

    def _raise(_: StreamBatch) -> None:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        consume_stream(
            stream,
            handle_batch=_raise,
            heartbeat_interval=1.0,
            idle_timeout=30.0,
            clock=lambda: 0.0,
        )

    assert stream.close_calls == 1


def test_consume_stream_deduplicates_stream_and_iterator_close() -> None:
    stream = _DualCloseSyncStream(
        [StreamBatch(channel="ticker", events=({"price": 10.0},), received_at=0.0)]
    )

    consume_stream(
        stream,
        handle_batch=lambda batch: None,
        heartbeat_interval=1.0,
        idle_timeout=30.0,
        clock=lambda: 0.0,
    )

    assert stream.close_calls == 1


def test_consume_stream_async_deduplicates_stream_and_iterator_aclose() -> None:
    stream = _DualCloseAsyncStream(
        [StreamBatch(channel="ticker", events=({"price": 10.0},), received_at=0.0)]
    )

    async def _run() -> None:
        await consume_stream_async(
            stream,
            handle_batch=lambda batch: None,
            heartbeat_interval=1.0,
            idle_timeout=30.0,
            clock=lambda: 0.0,
        )

    asyncio.run(_run())

    assert stream.aclose_calls == 1
