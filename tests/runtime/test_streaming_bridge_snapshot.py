from __future__ import annotations

import json

import pandas as pd

from bot_core.exchanges.streaming import StreamBatch
from bot_core.runtime.streaming_bridge import (
    load_snapshot_from_file,
    normalize_snapshot_events,
    stream_batches_to_frame,
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
    assert payload[-1]["sequence"] == 1
    assert payload[-1]["close"] == 3.2

    normalized = load_snapshot_from_file(str(output))
    assert [entry["timestamp_ms"] for entry in normalized] == [1000, 2000]
    assert [entry["sequence"] for entry in normalized] == [0, 1]
    assert normalized[-1]["close"] == 3.2


def test_normalize_snapshot_events_idempotent() -> None:
    source = [
        {"timestamp_ms": 1000, "open": 1.0, "high": 1.5, "low": 0.5, "close": 1.2, "volume": 1.0},
        {"timestamp_ms": 2000, "open": 2.0, "high": 2.5, "low": 1.5, "close": 2.2, "volume": 1.0},
    ]

    normalized = normalize_snapshot_events(source)
    normalized_again = normalize_snapshot_events(normalized)

    assert normalized_again == normalized


def test_snapshot_roundtrip_matches_history_frame(tmp_path) -> None:
    events = [
        {
            "timestamp_ms": 1000,
            "open": 1.0,
            "high": 1.5,
            "low": 0.5,
            "close": 1.2,
            "volume": 10.0,
            "channel": "ohlcv",
        },
        {
            "timestamp_ms": 2000,
            "open": 1.2,
            "high": 1.8,
            "low": 0.8,
            "close": 1.6,
            "volume": 12.0,
            "channel": "ohlcv",
        },
    ]

    output = tmp_path / "snapshot.json"
    write_snapshot_to_file(events, str(output))

    normalized = load_snapshot_from_file(str(output))
    batches = [
        StreamBatch(channel="ohlcv", cursor="cursor-1", events=normalized, received_at=0.0)
    ]

    frame = stream_batches_to_frame(batches)
    expected = pd.DataFrame(events)
    expected.index = pd.to_datetime(expected["timestamp_ms"], unit="ms", utc=True)
    expected.index.name = "timestamp"

    pd.testing.assert_frame_equal(
        frame[["open", "high", "low", "close", "volume"]],
        expected[["open", "high", "low", "close", "volume"]],
    )
