"""Narzędzia pomocnicze do mapowania historii na feed strumieniowy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import pandas as pd

from bot_core.exchanges.streaming import LocalLongPollStream, StreamBatch


def _normalize_timestamp(value: Any) -> int:
    if value is None:
        raise ValueError("Wymagany jest znacznik czasu w ms")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, datetime):
        return int(value.astimezone(timezone.utc).timestamp() * 1000)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Pusty znacznik czasu")
        try:
            if text.isdigit():
                return int(text)
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return int(parsed.astimezone(timezone.utc).timestamp() * 1000)
        except ValueError as exc:  # pragma: no cover - diagnostyka danych
            raise ValueError(f"Niepoprawny znacznik czasu: {value!r}") from exc
    raise TypeError(f"Nieobsługiwany typ znacznika czasu: {type(value)!r}")


def _normalize_event(event: Mapping[str, Any], *, channel: str | None = None) -> MutableMapping[str, Any]:
    payload: MutableMapping[str, Any] = {str(key): value for key, value in event.items()}
    if channel is not None:
        payload.setdefault("channel", channel)
    timestamp = payload.get("timestamp_ms")
    if timestamp is None:
        timestamp = payload.get("timestamp")
    payload["timestamp_ms"] = _normalize_timestamp(timestamp)
    return payload


def history_to_stream_batches(
    history: Sequence[Mapping[str, Any]],
    *,
    channel: str,
    batch_size: int = 250,
    cursor_prefix: str = "cursor",
) -> list[dict[str, Any]]:
    """Konwertuje historię OHLCV na sekwencję paczek streamu long-pollowego."""

    batches: list[dict[str, Any]] = []
    buffer: list[Mapping[str, Any]] = []
    cursor_counter = 0
    for entry in history:
        buffer.append(_normalize_event(entry, channel=channel))
        if len(buffer) >= max(1, batch_size):
            cursor_counter += 1
            batches.append(
                {
                    "channel": channel,
                    "events": [dict(item) for item in buffer],
                    "cursor": f"{cursor_prefix}-{cursor_counter}",
                }
            )
            buffer = []
    if buffer:
        cursor_counter += 1
        batches.append(
            {
                "channel": channel,
                "events": [dict(item) for item in buffer],
                "cursor": f"{cursor_prefix}-{cursor_counter}",
            }
        )
    return batches


def stream_batches_to_frame(batches: Iterable[StreamBatch]) -> pd.DataFrame:
    """Buduje ramkę danych OHLCV na podstawie paczek strumienia."""

    rows: list[MutableMapping[str, Any]] = []
    for batch in batches:
        for event in batch.events:
            rows.append(_normalize_event(event, channel=batch.channel))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"]).set_index(
            pd.Index([], name="timestamp")
        )
    frame = pd.DataFrame(rows)
    frame.sort_values("timestamp_ms", inplace=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True)
    frame.set_index("timestamp", inplace=True)
    columns = [column for column in ("open", "high", "low", "close", "volume") if column in frame]
    return frame[columns]


def capture_stream_snapshot(
    *,
    base_url: str,
    path: str,
    channels: Sequence[str],
    adapter: str,
    scope: str,
    environment: str,
    limit: int = 500,
    poll_interval: float = 0.25,
    timeout: float = 10.0,
) -> list[MutableMapping[str, Any]]:
    """Pobiera snapshot danych strumieniowych korzystając z LocalLongPollStream."""

    stream = LocalLongPollStream(
        base_url=base_url,
        path=path,
        channels=channels,
        adapter=adapter,
        scope=scope,
        environment=environment,
        poll_interval=poll_interval,
        timeout=timeout,
        max_retries=3,
        backoff_base=0.0,
        backoff_cap=0.0,
        jitter=(0.0, 0.0),
    )
    events: list[MutableMapping[str, Any]] = []
    try:
        for batch in stream:
            for event in batch.events:
                events.append(_normalize_event(event, channel=batch.channel))
                if limit and len(events) >= limit:
                    stream.close()
                    return events
    finally:
        stream.close()
    return events


def write_snapshot_to_file(events: Sequence[Mapping[str, Any]], path: str) -> None:
    import json

    normalized: list[MutableMapping[str, Any]] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        normalized.append(_normalize_event(event))

    normalized.sort(key=lambda item: (item.get("timestamp_ms", 0), item.get("channel")))

    deduplicated: list[MutableMapping[str, Any]] = []
    last_key: tuple[int, str | None] | None = None
    for item in normalized:
        channel = item.get("channel")
        key = (int(item["timestamp_ms"]), channel if isinstance(channel, str) else None)
        if deduplicated and last_key == key:
            deduplicated[-1] = item
        else:
            deduplicated.append(item)
            last_key = key

    with open(path, "w", encoding="utf-8") as handle:
        json.dump([dict(event) for event in deduplicated], handle, ensure_ascii=False, indent=2)


def load_snapshot_from_file(path: str) -> list[MutableMapping[str, Any]]:
    import json

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return [
        _normalize_event(entry)
        for entry in payload
        if isinstance(entry, Mapping)
    ]


__all__ = [
    "capture_stream_snapshot",
    "history_to_stream_batches",
    "load_snapshot_from_file",
    "stream_batches_to_frame",
    "write_snapshot_to_file",
]

