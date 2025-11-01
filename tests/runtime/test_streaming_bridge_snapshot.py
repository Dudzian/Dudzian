from __future__ import annotations

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
