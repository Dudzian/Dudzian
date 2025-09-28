"""Pomocnicze narzędzia do monitorowania rotacji kluczy API."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator


def _ensure_utc(timestamp: datetime | None) -> datetime:
    """Zwraca znacznik czasu w strefie UTC (domyślnie bieżący moment)."""

    if timestamp is None:
        return datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _serialize_timestamp(timestamp: datetime) -> str:
    value = _ensure_utc(timestamp).replace(microsecond=0)
    return value.isoformat().replace("+00:00", "Z")


def _deserialize_timestamp(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw:
        return None
    candidate = raw.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return _ensure_utc(parsed)


@dataclass(slots=True)
class RotationStatus:
    """Stan rotacji pojedynczego wpisu w rejestrze."""

    key: str
    purpose: str
    interval_days: float
    last_rotated: datetime | None
    days_since_rotation: float | None
    due_in_days: float
    is_due: bool
    is_overdue: bool


class RotationRegistry:
    """Rejestr rotacji kluczy API przechowywany w pliku JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, datetime] = {}
        self._load()

    # ------------------------------------------------------------------
    # API publiczne
    # ------------------------------------------------------------------
    def mark_rotated(
        self,
        key: str,
        purpose: str,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """Aktualizuje wpis rotacji dla wskazanego klucza i celu."""

        record_key = self._record_key(key, purpose)
        self._records[record_key] = _ensure_utc(timestamp)
        self._persist()

    def status(
        self,
        key: str,
        purpose: str,
        *,
        interval_days: float = 90.0,
        now: datetime | None = None,
    ) -> RotationStatus:
        """Zwraca informacje o stanie rotacji dla wskazanego wpisu."""

        record_key = self._record_key(key, purpose)
        current_time = _ensure_utc(now)
        last_rotated = self._records.get(record_key)

        if last_rotated is None:
            return RotationStatus(
                key=key,
                purpose=purpose,
                interval_days=interval_days,
                last_rotated=None,
                days_since_rotation=None,
                due_in_days=0.0,
                is_due=True,
                is_overdue=True,
            )

        delta = current_time - last_rotated
        days_since = delta.total_seconds() / 86_400.0
        due_in = interval_days - days_since
        is_due = days_since >= interval_days
        is_overdue = days_since > interval_days

        return RotationStatus(
            key=key,
            purpose=purpose,
            interval_days=interval_days,
            last_rotated=last_rotated,
            days_since_rotation=days_since,
            due_in_days=due_in,
            is_due=is_due,
            is_overdue=is_overdue,
        )

    def due_within(
        self,
        *,
        interval_days: float = 90.0,
        warn_within_days: float = 14.0,
        now: datetime | None = None,
    ) -> Iterator[RotationStatus]:
        """Iteruje po wpisach wymagających rotacji w najbliższym czasie."""

        reference = _ensure_utc(now)
        for record_key, rotated_at in self._records.items():
            key, purpose = record_key.split("::", maxsplit=1)
            status = self.status(key, purpose, interval_days=interval_days, now=reference)
            if status.is_overdue or status.is_due or status.due_in_days <= warn_within_days:
                yield status

    def entries(self) -> Iterable[tuple[str, str, datetime]]:
        """Zwraca wszystkie wpisy rejestru w formie krotek."""

        for record_key, rotated_at in sorted(self._records.items()):
            key, purpose = record_key.split("::", maxsplit=1)
            yield key, purpose, rotated_at

    # ------------------------------------------------------------------
    # Wewnętrzne narzędzia
    # ------------------------------------------------------------------
    def _record_key(self, key: str, purpose: str) -> str:
        return f"{key.strip().lower()}::{purpose.strip().lower()}"

    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self._records = {}
            return
        except json.JSONDecodeError:
            self._records = {}
            return

        records: Dict[str, datetime] = {}
        for record_key, raw_timestamp in raw.items():
            normalized = _deserialize_timestamp(raw_timestamp)
            if normalized is None:
                continue
            records[str(record_key)] = normalized
        self._records = records

    def _persist(self) -> None:
        payload: Dict[str, str] = {
            record_key: _serialize_timestamp(timestamp)
            for record_key, timestamp in self._records.items()
        }
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        tmp_path.replace(self._path)


__all__ = ["RotationRegistry", "RotationStatus"]

