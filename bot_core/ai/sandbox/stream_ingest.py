"""Obsługa odtwarzania strumieni OHLCV/ksiąg z trading stubu."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from heapq import heappop, heappush
from itertools import count
from pathlib import Path
from typing import Iterator, Mapping, MutableMapping, Sequence

import yaml

_ROOT = Path(__file__).resolve().parents[3]
_DATA_ROOT = _ROOT / "data" / "trading_stub"
_DATASET_DIR = _DATA_ROOT / "datasets"


def _normalize_dataset_path(dataset: str | Path | None) -> Path:
    if dataset is None:
        return _DATASET_DIR / "multi_asset_performance.yaml"
    candidate = Path(dataset)
    if candidate.exists():
        return candidate
    if candidate.suffix:
        resolved = _ROOT / candidate if not candidate.is_absolute() else candidate
        if resolved.exists():
            return resolved
        raise FileNotFoundError(resolved)
    # treat as dataset name without extension
    target = _DATASET_DIR / f"{candidate.name}.yaml"
    if target.exists():
        return target
    # fallback to potential relative path without extension
    alt = (_ROOT / candidate).with_suffix(".yaml")
    if alt.exists():
        return alt
    raise FileNotFoundError(target)


def _parse_timestamp(payload: Mapping[str, object]) -> datetime:
    for key in ("timestamp", "time", "open_time", "ts"):
        raw = payload.get(key)
        if isinstance(raw, datetime):
            if raw.tzinfo is None:
                return raw.replace(tzinfo=timezone.utc)
            return raw.astimezone(timezone.utc)
        if isinstance(raw, str) and raw:
            value = raw.strip()
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            else:
                parsed = parsed.astimezone(timezone.utc)
            return parsed
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class InstrumentDescriptor:
    """Opis instrumentu dostępny w stubie danych."""

    exchange: str
    symbol: str
    venue_symbol: str | None = None
    quote_currency: str | None = None
    base_currency: str | None = None


@dataclass(slots=True)
class SandboxStreamEvent:
    """Pojedynczy event strumienia sandboxowego."""

    instrument: InstrumentDescriptor
    payload: Mapping[str, object]
    event_type: str
    timestamp: datetime
    sequence: int


class TradingStubStreamIngestor:
    """Ładuje i odtwarza strumienie zdarzeń z katalogu trading stub."""

    def __init__(self, dataset: str | Path | None = None) -> None:
        self._dataset_path = _normalize_dataset_path(dataset)
        with self._dataset_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"Dataset {self._dataset_path} musi być mapowaniem")
        self._raw_dataset: Mapping[str, object] = raw
        self._streams: list[tuple[InstrumentDescriptor, Mapping[str, object], str | None]] = []
        self._risk_streams: list[tuple[InstrumentDescriptor, Sequence[Mapping[str, object]]]] = []
        self._prepare_streams()

    @property
    def dataset_path(self) -> Path:
        return self._dataset_path

    def iter_events(
        self,
        *,
        instruments: Sequence[str] | None = None,
        event_types: Sequence[str] | None = None,
    ) -> Iterator[SandboxStreamEvent]:
        """Zwraca zdarzenia strumienia w porządku chronologicznym."""

        instrument_filter = {symbol.lower() for symbol in instruments} if instruments else None
        type_filter = {etype for etype in event_types} if event_types else None
        sequence = 0
        for event in self._iter_merged_events(instrument_filter=instrument_filter, type_filter=type_filter):
            yield SandboxStreamEvent(
                instrument=event.instrument,
                payload=event.payload,
                event_type=event.event_type,
                timestamp=event.timestamp,
                sequence=sequence,
            )
            sequence += 1

    def _prepare_streams(self) -> None:
        market_data = self._raw_dataset.get("market_data")
        if not isinstance(market_data, Sequence):
            return
        streams: list[tuple[InstrumentDescriptor, Mapping[str, object], str | None]] = []
        for entry in market_data:
            if not isinstance(entry, Mapping):
                continue
            instrument_meta = entry.get("instrument") if isinstance(entry.get("instrument"), Mapping) else {}
            instrument = InstrumentDescriptor(
                exchange=str(instrument_meta.get("exchange", "UNKNOWN")),
                symbol=str(instrument_meta.get("symbol", instrument_meta.get("venue_symbol", "UNKNOWN"))),
                venue_symbol=instrument_meta.get("venue_symbol"),
                quote_currency=instrument_meta.get("quote_currency"),
                base_currency=instrument_meta.get("base_currency"),
            )
            stream = entry.get("stream")
            if isinstance(stream, Mapping):
                streams.append((instrument, stream, None))
            book_stream = entry.get("book_stream")
            if isinstance(book_stream, Mapping):
                streams.append((instrument, book_stream, "order_book"))
        self._streams = streams
        self._risk_streams = []
        risk_states = self._raw_dataset.get("risk_states")
        if isinstance(risk_states, Sequence):
            for entry in risk_states:
                if not isinstance(entry, Mapping):
                    continue
                instrument_meta = entry.get("instrument") if isinstance(entry.get("instrument"), Mapping) else {}
                if instrument_meta:
                    instrument = InstrumentDescriptor(
                        exchange=str(instrument_meta.get("exchange", "UNKNOWN")),
                        symbol=str(
                            instrument_meta.get(
                                "symbol",
                                instrument_meta.get("venue_symbol", instrument_meta.get("exchange", "UNKNOWN")),
                            )
                        ),
                        venue_symbol=instrument_meta.get("venue_symbol"),
                        quote_currency=instrument_meta.get("quote_currency"),
                        base_currency=instrument_meta.get("base_currency"),
                    )
                else:
                    instrument = InstrumentDescriptor(
                        exchange=str(entry.get("exchange", "GLOBAL")),
                        symbol=str(entry.get("symbol", "GLOBAL")),
                    )
                states = entry.get("states")
                if not isinstance(states, Sequence):
                    continue
                normalized_states = [state for state in states if isinstance(state, Mapping)]
                if normalized_states:
                    self._risk_streams.append((instrument, normalized_states))

    def _iter_merged_events(
        self,
        *,
        instrument_filter: set[str] | None,
        type_filter: set[str] | None,
    ) -> Iterator[SandboxStreamEvent]:
        heap: list[tuple[datetime, int, int, SandboxStreamEvent, Iterator[SandboxStreamEvent]]] = []
        order_counter = count()
        for instrument, stream, prefix in self._streams:
            if instrument_filter and instrument.symbol.lower() not in instrument_filter:
                continue
            generator = self._iter_stream_events(
                instrument,
                stream,
                prefix=prefix,
                type_filter=type_filter,
            )
            try:
                first = next(generator)
            except StopIteration:
                continue
            heappush(
                heap,
                (
                    first.timestamp,
                    first.sequence,
                    next(order_counter),
                    first,
                    generator,
                ),
            )
        for instrument, states in self._risk_streams:
            if instrument_filter and instrument.symbol.lower() not in instrument_filter:
                continue
            generator = self._iter_risk_state_events(
                instrument,
                states,
                type_filter=type_filter,
            )
            try:
                first = next(generator)
            except StopIteration:
                continue
            heappush(
                heap,
                (
                    first.timestamp,
                    first.sequence,
                    next(order_counter),
                    first,
                    generator,
                ),
            )
        while heap:
            _, _, _, event, generator = heappop(heap)
            yield event
            try:
                following = next(generator)
            except StopIteration:
                continue
            heappush(
                heap,
                (
                    following.timestamp,
                    following.sequence,
                    next(order_counter),
                    following,
                    generator,
                ),
            )

    def _iter_stream_events(
        self,
        instrument: InstrumentDescriptor,
        stream: Mapping[str, object],
        *,
        prefix: str | None,
        type_filter: set[str] | None,
    ) -> Iterator[SandboxStreamEvent]:
        local_sequence = 0
        event_prefix = f"{prefix}_" if prefix else ""
        for key in ("snapshot", "snapshots"):
            snapshot_entries = stream.get(key)
            if isinstance(snapshot_entries, Sequence):
                event_type = f"{event_prefix}snapshot"
                for payload in snapshot_entries:
                    if not isinstance(payload, Mapping):
                        continue
                    if type_filter and event_type not in type_filter:
                        continue
                    yield SandboxStreamEvent(
                        instrument=instrument,
                        payload=dict(payload),
                        event_type=event_type,
                        timestamp=_parse_timestamp(payload),
                        sequence=local_sequence,
                    )
                    local_sequence += 1
                break
        increments = stream.get("increments")
        if isinstance(increments, Sequence):
            event_type = f"{event_prefix}increment"
            for payload in increments:
                if not isinstance(payload, Mapping):
                    continue
                if type_filter and event_type not in type_filter:
                    continue
                yield SandboxStreamEvent(
                    instrument=instrument,
                    payload=dict(payload),
                    event_type=event_type,
                    timestamp=_parse_timestamp(payload),
                    sequence=local_sequence,
                )
                local_sequence += 1

    def _iter_risk_state_events(
        self,
        instrument: InstrumentDescriptor,
        states: Sequence[Mapping[str, object]],
        *,
        type_filter: set[str] | None,
    ) -> Iterator[SandboxStreamEvent]:
        event_type = "risk_state"
        if type_filter and event_type not in type_filter:
            return
        local_sequence = 0
        for payload in states:
            if not isinstance(payload, Mapping):
                continue
            yield SandboxStreamEvent(
                instrument=instrument,
                payload=dict(payload),
                event_type=event_type,
                timestamp=_parse_timestamp(payload),
                sequence=local_sequence,
            )
            local_sequence += 1

    def summary(self) -> Mapping[str, object]:
        """Zwraca podsumowanie datasetu zliczające zdarzenia."""

        counts: MutableMapping[str, int] = {}
        total_events = 0
        for event in self.iter_events():
            total_events += 1
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return {
            "dataset": str(self._dataset_path),
            "events": total_events,
            "event_types": dict(counts),
        }


__all__ = [
    "InstrumentDescriptor",
    "SandboxStreamEvent",
    "TradingStubStreamIngestor",
]
