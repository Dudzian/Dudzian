import pytest

from bot_core.exchanges.streaming import StreamBatch
from bot_core.runtime.pipeline import consume_stream


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
