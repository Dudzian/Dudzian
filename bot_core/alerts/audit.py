"""Repozytoria audytu wykorzystywane przez system alertów."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping

from bot_core.alerts.base import AlertAuditLog, AlertMessage


@dataclass(slots=True)
class AlertAuditEntry:
    """Pojedynczy zapis audytowy odpowiadający wysłanemu komunikatowi."""

    channel: str
    message: AlertMessage
    created_at: datetime

    def as_dict(self) -> Mapping[str, str]:
        """Eksportuje wpis w formacie przyjaznym serializacji."""

        payload: dict[str, str] = {
            "channel": self.channel,
            "category": self.message.category,
            "title": self.message.title,
            "severity": self.message.severity,
            "timestamp": self.message.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
        }
        payload.update({f"ctx_{k}": v for k, v in self.message.context.items()})
        payload["body"] = self.message.body
        return payload


class InMemoryAlertAuditLog(AlertAuditLog):
    """Prosta implementacja audytu na potrzeby środowisk deweloperskich."""

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: List[AlertAuditEntry] = []

    def append(self, message: AlertMessage, *, channel: str) -> None:
        entry = AlertAuditEntry(channel=channel, message=message, created_at=message.timestamp)
        self._entries.append(entry)

    def export(self) -> Iterable[Mapping[str, str]]:
        return tuple(entry.as_dict() for entry in self._entries)


@dataclass(slots=True)
class FileAlertAuditLog(AlertAuditLog):
    """Zapisuje wpisy audytowe do plików JSONL z rotacją po dniach."""

    directory: str | Path
    filename_pattern: str = "alerts-%Y%m%d.jsonl"
    retention_days: int | None = 730
    fsync: bool = False
    encoding: str = "utf-8"
    newline: str = "\n"
    _path: Path = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._path = Path(self.directory)
        self._path.mkdir(parents=True, exist_ok=True)
        # Walidacja wzorca – strftime zgłosi ValueError, jeśli pattern jest niepoprawny.
        datetime.now(timezone.utc).strftime(self.filename_pattern)
        self._lock = threading.Lock()

    def append(self, message: AlertMessage, *, channel: str) -> None:
        entry = AlertAuditEntry(channel=channel, message=message, created_at=message.timestamp)
        record = entry.as_dict()
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        target = self._target_file(message.timestamp)

        with self._lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding=self.encoding) as handle:
                handle.write(payload)
                handle.write(self.newline)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())
            self._purge_old_files(current_date=message.timestamp.astimezone(timezone.utc))

    def export(self) -> Iterable[Mapping[str, str]]:
        return tuple(self._iter_entries())

    def _iter_entries(self) -> Iterator[Mapping[str, str]]:
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
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        yield {str(key): str(value) for key, value in data.items()}
            except OSError:
                continue

    def _target_file(self, timestamp: datetime) -> Path:
        ts = timestamp.astimezone(timezone.utc)
        name = ts.strftime(self.filename_pattern)
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


__all__ = ["AlertAuditEntry", "InMemoryAlertAuditLog", "FileAlertAuditLog"]

