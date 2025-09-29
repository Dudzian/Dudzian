"""Audyt jakości danych OHLCV."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Protocol


@dataclass(slots=True)
class GapAuditRecord:
    """Pojedynczy wpis audytowy opisujący stan danych dla symbolu/interwału."""

    timestamp: datetime
    environment: str
    exchange: str
    symbol: str
    interval: str
    status: str
    gap_minutes: float | None
    row_count: int | None
    last_timestamp: str | None
    warnings_in_window: int | None = None
    incident_minutes: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "environment": self.environment,
            "exchange": self.exchange,
            "symbol": self.symbol,
            "interval": self.interval,
            "status": self.status,
            "gap_minutes": None if self.gap_minutes is None else round(self.gap_minutes, 3),
            "row_count": self.row_count,
            "last_timestamp": self.last_timestamp,
            "warnings_in_window": self.warnings_in_window,
            "incident_minutes": None
            if self.incident_minutes is None
            else round(self.incident_minutes, 3),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "GapAuditRecord":
        """Buduje wpis audytowy z danych JSON (np. z pliku JSONL)."""
        raw_timestamp = payload.get("timestamp")
        if not isinstance(raw_timestamp, str):
            raise ValueError("Pole 'timestamp' musi być tekstem w formacie ISO 8601")

        timestamp = datetime.fromisoformat(raw_timestamp)
        # Wyrównanie do UTC, jeśli brak strefy czasowej
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        def _maybe_float(key: str) -> float | None:
            value = payload.get(key)
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ValueError(f"Pole '{key}' musi być liczbą zmiennoprzecinkową lub null") from None

        def _maybe_int(key: str) -> int | None:
            value = payload.get(key)
            if value is None:
                return None
            try:
                return int(value)
            except (TypeError, ValueError):
                raise ValueError(f"Pole '{key}' musi być liczbą całkowitą lub null") from None

        environment = str(payload.get("environment", ""))
        exchange = str(payload.get("exchange", ""))
        symbol = str(payload.get("symbol", ""))
        interval = str(payload.get("interval", ""))
        status = str(payload.get("status", ""))

        last_timestamp_raw = payload.get("last_timestamp")
        last_timestamp = None if last_timestamp_raw is None else str(last_timestamp_raw)

        warnings_in_window = payload.get("warnings_in_window")
        warnings_value = None
        if warnings_in_window is not None:
            try:
                warnings_value = int(warnings_in_window)
            except (TypeError, ValueError) as exc:  # pragma: no cover - walidacja wejścia
                raise ValueError("Pole 'warnings_in_window' musi być liczbą całkowitą") from exc

        return cls(
            timestamp=timestamp,
            environment=environment,
            exchange=exchange,
            symbol=symbol,
            interval=interval,
            status=status,
            gap_minutes=_maybe_float("gap_minutes"),
            row_count=_maybe_int("row_count"),
            last_timestamp=last_timestamp,
            warnings_in_window=warnings_value,
            incident_minutes=_maybe_float("incident_minutes"),
        )

    @classmethod
    def from_json(cls, line: str) -> "GapAuditRecord":
        """Buduje wpis audytowy na podstawie pojedynczego wiersza JSONL."""
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - walidacja wejścia
            raise ValueError("Niepoprawny wiersz JSONL") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Wiersz JSONL musi być obiektem JSON")
        return cls.from_dict(payload)


class GapAuditLogger(Protocol):
    """Interfejs loggera przyjmującego wpisy audytowe luk danych."""

    def log(self, record: GapAuditRecord) -> None:
        ...  # pragma: no cover - protokół typów


class JSONLGapAuditLogger:
    """Logger zapisujący wpisy audytowe w pliku JSONL (append-only)."""

    def __init__(self, path: str | Path, *, fsync: bool = False) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fsync = fsync

    def log(self, record: GapAuditRecord) -> None:
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(payload + "\n")
            if self._fsync:
                handle.flush()
                os.fsync(handle.fileno())


__all__ = ["GapAuditRecord", "GapAuditLogger", "JSONLGapAuditLogger"]
