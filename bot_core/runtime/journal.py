"""Dziennik decyzji tradingowych – wspiera audyt i compliance."""
from __future__ import annotations

import json
import os
import threading
import math
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Optional, Protocol


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _format_float(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:.10f}".rstrip("0").rstrip(".") if value else "0"


@dataclass(slots=True)
class TradingDecisionEvent:
    """Pojedyncze zdarzenie zapisane w dzienniku decyzji."""

    event_type: str
    timestamp: datetime
    environment: str
    portfolio: str
    risk_profile: str
    symbol: Optional[str] = None
    side: Optional[str] = None
    quantity: Optional[float] = None
    price: Optional[float] = None
    status: Optional[str] = None
    schedule: Optional[str] = None
    strategy: Optional[str] = None
    schedule_run_id: Optional[str] = None
    strategy_instance_id: Optional[str] = None
    signal_id: Optional[str] = None
    primary_exchange: Optional[str] = None
    secondary_exchange: Optional[str] = None
    base_asset: Optional[str] = None
    quote_asset: Optional[str] = None
    instrument_type: Optional[str] = None
    data_feed: Optional[str] = None
    risk_budget_bucket: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    telemetry_namespace: Optional[str] = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> Mapping[str, str]:
        payload: MutableMapping[str, str] = {
            "event": self.event_type,
            "timestamp": _ensure_utc(self.timestamp).isoformat(),
            "environment": self.environment,
            "portfolio": self.portfolio,
            "risk_profile": self.risk_profile,
        }
        if self.symbol:
            payload["symbol"] = self.symbol
        if self.side:
            payload["side"] = self.side
        quantity = _format_float(self.quantity)
        if quantity is not None:
            payload["quantity"] = quantity
        price = _format_float(self.price)
        if price is not None:
            payload["price"] = price
        if self.status:
            payload["status"] = self.status
        if self.schedule:
            payload["schedule"] = self.schedule
        if self.strategy:
            payload["strategy"] = self.strategy
        if self.schedule_run_id:
            payload["schedule_run_id"] = self.schedule_run_id
        if self.strategy_instance_id:
            payload["strategy_instance_id"] = self.strategy_instance_id
        if self.signal_id:
            payload["signal_id"] = self.signal_id
        if self.primary_exchange:
            payload["primary_exchange"] = self.primary_exchange
        if self.secondary_exchange:
            payload["secondary_exchange"] = self.secondary_exchange
        if self.base_asset:
            payload["base_asset"] = self.base_asset
        if self.quote_asset:
            payload["quote_asset"] = self.quote_asset
        if self.instrument_type:
            payload["instrument_type"] = self.instrument_type
        if self.data_feed:
            payload["data_feed"] = self.data_feed
        if self.risk_budget_bucket:
            payload["risk_budget_bucket"] = self.risk_budget_bucket
        confidence = _format_float(self.confidence)
        if confidence is not None:
            payload["confidence"] = confidence
        latency = _format_float(self.latency_ms)
        if latency is not None:
            payload["latency_ms"] = latency
        if self.telemetry_namespace:
            payload["telemetry_namespace"] = self.telemetry_namespace
        for key, value in self.metadata.items():
            payload[str(key)] = str(value)
        return payload


class TradingDecisionJournal(Protocol):
    """Minimalny kontrakt dziennika decyzji."""

    def record(self, event: TradingDecisionEvent) -> None:
        ...

    def export(self) -> Iterable[Mapping[str, str]]:
        ...


@dataclass(slots=True)
class InMemoryTradingDecisionJournal(TradingDecisionJournal):
    """Lekki dziennik wykorzystywany w testach i dev."""

    _events: list[TradingDecisionEvent] = field(default_factory=list)

    def record(self, event: TradingDecisionEvent) -> None:
        self._events.append(event)

    def export(self) -> Iterable[Mapping[str, str]]:
        return tuple(event.as_dict() for event in self._events)


@dataclass(slots=True)
class JsonlTradingDecisionJournal(TradingDecisionJournal):
    """Dziennik zapisujący zdarzenia do plików JSONL z retencją."""

    directory: str | Path
    filename_pattern: str = "decisions-%Y%m%d.jsonl"
    retention_days: Optional[int] = 730
    fsync: bool = False
    encoding: str = "utf-8"
    newline: str = "\n"
    _path: Path = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.directory)
        self._path.mkdir(parents=True, exist_ok=True)
        _ensure_utc(datetime.now(timezone.utc)).strftime(self.filename_pattern)
        self._lock = threading.Lock()

    def record(self, event: TradingDecisionEvent) -> None:
        record = json.dumps(event.as_dict(), ensure_ascii=False, separators=(",", ":"))
        target = self._target_file(event.timestamp)
        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding=self.encoding) as handle:
                handle.write(record)
                handle.write(self.newline)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            self._purge_old_files(current_date=_ensure_utc(event.timestamp))

    def export(self) -> Iterable[Mapping[str, str]]:
        events: list[Mapping[str, str]] = []
        for file_path in sorted(self._path.glob("*")):
            if not file_path.is_file():
                continue
            try:
                with file_path.open("r", encoding=self.encoding) as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                continue
        return tuple(events)

    def _target_file(self, timestamp: datetime) -> Path:
        utc_time = _ensure_utc(timestamp)
        name = utc_time.strftime(self.filename_pattern)
        return self._path / name

    def _purge_old_files(self, *, current_date: datetime) -> None:
        if not self.retention_days or self.retention_days <= 0:
            return
        cutoff = current_date.date() - timedelta(days=self.retention_days - 1)
        for file_path in self._path.glob("*"):
            if not file_path.is_file():
                continue
            try:
                file_date = datetime.strptime(file_path.name, self.filename_pattern).date()
            except ValueError:
                continue
            if file_date < cutoff:
                try:
                    file_path.unlink()
                except OSError:
                    continue


def log_decision_event(
    journal: TradingDecisionJournal | None,
    *,
    event: str,
    environment: str,
    portfolio: str,
    risk_profile: str,
    timestamp: datetime | None = None,
    symbol: str | None = None,
    side: str | None = None,
    quantity: float | None = None,
    price: float | None = None,
    status: str | None = None,
    schedule: str | None = None,
    strategy: str | None = None,
    metadata: Mapping[str, object] | None = None,
    latency_ms: float | None = None,
    confidence: float | None = None,
) -> None:
    """Zapisuje standardowe zdarzenie decyzji do dziennika."""

    if journal is None:
        return

    meta: MutableMapping[str, str] = {}
    if metadata:
        meta.update({str(key): str(value) for key, value in metadata.items()})

    event_obj = TradingDecisionEvent(
        event_type=event,
        timestamp=_ensure_utc(timestamp or datetime.now(timezone.utc)),
        environment=environment,
        portfolio=portfolio,
        risk_profile=risk_profile,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        status=status,
        schedule=schedule,
        strategy=strategy,
        latency_ms=latency_ms,
        confidence=confidence,
        metadata=meta,
    )
    journal.record(event_obj)


def _parse_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return _ensure_utc(value)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        return _ensure_utc(parsed)
    return None


def _percentile(values: Iterable[float], fraction: float) -> float:
    sequence = sorted(values)
    if not sequence:
        return 0.0
    if fraction <= 0:
        return sequence[0]
    if fraction >= 1:
        return sequence[-1]
    index = fraction * (len(sequence) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sequence[lower]
    weight = index - lower
    return sequence[lower] + (sequence[upper] - sequence[lower]) * weight


def aggregate_decision_statistics(
    journal_or_records: TradingDecisionJournal | Iterable[Mapping[str, object]],
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> Mapping[str, object]:
    """Agreguje podstawowe statystyki decyzji w zadanym oknie czasowym."""

    if hasattr(journal_or_records, "export"):
        records = journal_or_records.export()
    else:
        records = journal_or_records

    start_bound = _ensure_utc(start) if start is not None else None
    end_bound = _ensure_utc(end) if end is not None else None

    total = 0
    by_status: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    latencies: list[float] = []
    confidences: list[float] = []

    for record in records:
        if not isinstance(record, Mapping):
            continue
        timestamp = _parse_timestamp(record.get("timestamp"))
        if start_bound is not None:
            if timestamp is None or timestamp < start_bound:
                continue
        if end_bound is not None:
            if timestamp is None or timestamp >= end_bound:
                continue

        total += 1

        status = str(record.get("status") or "").strip()
        if status:
            by_status[status.lower()] += 1

        symbol = record.get("symbol")
        if symbol:
            by_symbol[str(symbol)] += 1

        latency_raw = record.get("latency_ms")
        try:
            latency_value = float(latency_raw)
        except (TypeError, ValueError):
            latency_value = None
        if latency_value is not None and math.isfinite(latency_value):
            latencies.append(latency_value)

        confidence_raw = record.get("confidence")
        try:
            confidence_value = float(confidence_raw)
        except (TypeError, ValueError):
            confidence_value = None
        if confidence_value is not None and math.isfinite(confidence_value):
            confidences.append(confidence_value)

    summary: dict[str, object] = {
        "total": total,
        "by_status": dict(by_status),
        "by_symbol": dict(by_symbol),
    }

    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)
        summary["p95_latency_ms"] = _percentile(latencies, 0.95)

    if confidences:
        summary["avg_confidence"] = sum(confidences) / len(confidences)

    return summary


__all__ = [
    "TradingDecisionEvent",
    "TradingDecisionJournal",
    "InMemoryTradingDecisionJournal",
    "JsonlTradingDecisionJournal",
    "log_decision_event",
    "aggregate_decision_statistics",
]

