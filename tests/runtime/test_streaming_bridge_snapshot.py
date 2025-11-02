from __future__ import annotations

import asyncio
import json

import pytest

import bot_core.runtime.streaming_bridge as streaming_bridge
from bot_core.observability.metrics import MetricsRegistry
from bot_core.runtime.streaming_bridge import (
    load_snapshot_from_file,
    write_snapshot_to_file,
)


def test_write_snapshot_to_file_sorts_and_deduplicates(tmp_path) -> None:
    output = tmp_path / "snapshot.json"
    events = [
        {
            "timestamp_ms": 2000,
            "open": 2.0,
            "high": 3.0,
            "low": 1.0,
            "close": 2.5,
            "volume": 5.0,
            "channel": "ohlcv",
        },
        {
            "timestamp": "1970-01-01T00:00:01Z",
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": 1.2,
            "volume": 3.0,
            "channel": "ohlcv",
        },
        {
            "timestamp_ms": 2000,
            "open": 3.0,
            "high": 3.5,
            "low": 2.5,
            "close": 3.2,
            "volume": 6.0,
            "channel": "ohlcv",
        },
    ]

    write_snapshot_to_file(events, str(output))

    with open(output, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert [entry["timestamp_ms"] for entry in payload] == [1000, 2000]
    assert payload[-1]["close"] == 3.2

    normalized = load_snapshot_from_file(str(output))
    assert [entry["timestamp_ms"] for entry in normalized] == [1000, 2000]
    assert normalized[-1]["close"] == 3.2


def test_capture_stream_snapshot_uses_metrics_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    registry = MetricsRegistry()
    captured: dict[str, MetricsRegistry | None] = {}

    class _StubStream:
        def __init__(self, *_, **kwargs) -> None:
            captured["metrics"] = kwargs.get("metrics_registry")

        def start(self):  # noqa: D401 - interfejs fluent
            captured["started"] = True
            return self

        def wait_prefill(self, *, min_batches: int = 1, timeout: float | None = None) -> bool:
            captured["wait_prefill"] = {"min_batches": min_batches, "timeout": timeout}
            return True

        def __iter__(self):  # pragma: no cover - brak paczek
            return self

        def __next__(self):  # pragma: no cover - brak paczek
            raise StopIteration

        def close(self) -> None:  # pragma: no cover - brak efektów ubocznych
            pass

    monkeypatch.setattr(streaming_bridge, "LocalLongPollStream", _StubStream)

    result = streaming_bridge.capture_stream_snapshot(
        base_url="http://127.0.0.1:8080",
        path="/demo",
        channels=("ticker",),
        adapter="demo",
        scope="public",
        environment="paper",
        limit=0,
        metrics_registry=registry,
    )

    assert result == []
    assert captured["metrics"] is registry
    assert captured.get("started") is True
    assert captured.get("wait_prefill") == {"min_batches": 1, "timeout": 1.0}


def test_capture_stream_snapshot_async_prefills_and_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = MetricsRegistry()
    captured: dict[str, object] = {}

    class _StubStream:
        def __init__(self, *_, **kwargs) -> None:
            captured["metrics"] = kwargs.get("metrics_registry")

        def start(self):  # noqa: D401 - interfejs fluent
            captured["started"] = True
            return self

        async def wait_prefill_async(
            self, *, min_batches: int = 1, timeout: float | None = None
        ) -> bool:
            captured["wait_prefill_async"] = {
                "min_batches": min_batches,
                "timeout": timeout,
            }
            return True

        def __aiter__(self):
            captured["iterated"] = True
            self._delivered = False
            return self

        async def __anext__(self):
            if getattr(self, "_delivered", False):
                raise StopAsyncIteration
            self._delivered = True
            return streaming_bridge.StreamBatch(
                channel="ticker",
                events=({"timestamp_ms": 1000, "open": 1.0},),
                received_at=0.0,
            )

        async def aclose(self) -> None:
            captured["closed"] = captured.get("closed", 0) + 1

    monkeypatch.setattr(streaming_bridge, "LocalLongPollStream", _StubStream)

    result = asyncio.run(
        streaming_bridge.capture_stream_snapshot_async(
            base_url="http://127.0.0.1:8080",
            path="/demo",
            channels=("ticker",),
            adapter="demo",
            scope="public",
            environment="paper",
            limit=1,
            metrics_registry=registry,
        )
    )

    assert result == [
        {
            "timestamp_ms": 1000,
            "open": 1.0,
            "channel": "ticker",
        }
    ]
    assert captured["metrics"] is registry
    assert captured.get("started") is True
    assert captured.get("iterated") is True
    assert captured.get("wait_prefill_async") == {"min_batches": 1, "timeout": 1.0}
    assert captured.get("closed") == 2


def test_capture_stream_snapshot_async_closes_on_prefill_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, int] = {"closed": 0}

    class _FailingStream:
        def __init__(self, *_, **__):
            pass

        def start(self):
            return self

        async def wait_prefill_async(self, *, min_batches: int = 1, timeout: float | None = None) -> bool:
            raise RuntimeError("boom")

        def __aiter__(self):  # pragma: no cover - nie powinniśmy iterować
            raise AssertionError("Iteracja nie powinna się zdarzyć")

        async def aclose(self) -> None:
            captured["closed"] += 1

    monkeypatch.setattr(streaming_bridge, "LocalLongPollStream", _FailingStream)

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(
            streaming_bridge.capture_stream_snapshot_async(
                base_url="http://127.0.0.1:8080",
                path="/demo",
                channels=("ticker",),
                adapter="demo",
                scope="public",
                environment="paper",
                limit=1,
            )
        )

    assert captured["closed"] == 1
