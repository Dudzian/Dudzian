"""Dziennik decyzji tradingowych – wspiera audyt i compliance."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Optional, Protocol


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


__all__ = [
    "TradingDecisionEvent",
    "TradingDecisionJournal",
    "InMemoryTradingDecisionJournal",
    "JsonlTradingDecisionJournal",
]

