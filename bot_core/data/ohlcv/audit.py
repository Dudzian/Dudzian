"""Audyt jakości danych OHLCV."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol


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

