"""Runtime decision audit log for AutoTrader."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Collection, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class DecisionAuditRecord:
    """Single audit entry describing a decision pipeline stage."""

    timestamp: datetime
    stage: str
    symbol: str
    mode: str
    payload: Mapping[str, object]
    risk_snapshot: Mapping[str, object] | None = None
    portfolio_snapshot: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None

    def to_mapping(self) -> MutableMapping[str, object]:
        data: MutableMapping[str, object] = {
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "stage": self.stage,
            "symbol": self.symbol,
            "mode": self.mode,
            "payload": dict(self.payload),
        }
        if self.risk_snapshot is not None:
            data["risk_snapshot"] = dict(self.risk_snapshot)
        if self.portfolio_snapshot is not None:
            data["portfolio_snapshot"] = dict(self.portfolio_snapshot)
        if self.metadata is not None:
            data["metadata"] = dict(self.metadata)
        return data


class DecisionAuditLog:
    """Thread-safe in-memory audit log with optional retention limits."""

    def __init__(self, *, max_entries: int = 512) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: list[DecisionAuditRecord] = []
        self._lock = Lock()

    def record(
        self,
        stage: str,
        symbol: str,
        *,
        mode: str,
        payload: Mapping[str, object] | None = None,
        risk_snapshot: Mapping[str, object] | None = None,
        portfolio_snapshot: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        timestamp: datetime | None = None,
    ) -> DecisionAuditRecord:
        stamp = timestamp or datetime.now(timezone.utc)
        record = DecisionAuditRecord(
            timestamp=stamp,
            stage=str(stage),
            symbol=str(symbol),
            mode=str(mode),
            payload=dict(payload or {}),
            risk_snapshot=dict(risk_snapshot or {}) or None,
            portfolio_snapshot=dict(portfolio_snapshot or {}) or None,
            metadata=dict(metadata or {}) or None,
        )
        with self._lock:
            self._entries.append(record)
            if len(self._entries) > self._max_entries:
                excess = len(self._entries) - self._max_entries
                del self._entries[0:excess]
        return record

    def tail(self, limit: int = 20) -> Sequence[DecisionAuditRecord]:
        if limit <= 0:
            return ()
        with self._lock:
            return tuple(self._entries[-limit:])

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def to_dicts(self, limit: int = 20) -> Sequence[Mapping[str, object]]:
        return tuple(record.to_mapping() for record in self.tail(limit))

    def query_dicts(
        self,
        *,
        limit: int | None = 20,
        reverse: bool = False,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        normalized_limit: int | None = None
        if limit is not None:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):
                normalized_limit = 0
            if normalized_limit <= 0:
                return ()

        stage_filter = self._normalize_filter(stage)
        symbol_filter = self._normalize_filter(symbol, case_sensitive=True)
        mode_filter = self._normalize_filter(mode)
        since_ts = self._normalize_timestamp(since)
        until_ts = self._normalize_timestamp(until)

        with self._lock:
            entries = list(self._entries)

        filtered: list[DecisionAuditRecord] = []
        for record in entries:
            if stage_filter and record.stage.lower() not in stage_filter:
                continue
            if symbol_filter and record.symbol not in symbol_filter:
                continue
            if mode_filter and record.mode.lower() not in mode_filter:
                continue
            if has_risk_snapshot is True and record.risk_snapshot is None:
                continue
            if has_risk_snapshot is False and record.risk_snapshot is not None:
                continue
            if has_portfolio_snapshot is True and record.portfolio_snapshot is None:
                continue
            if has_portfolio_snapshot is False and record.portfolio_snapshot is not None:
                continue
            if since_ts and record.timestamp < since_ts:
                continue
            if until_ts and record.timestamp > until_ts:
                continue
            filtered.append(record)

        if normalized_limit is not None:
            filtered = filtered[-normalized_limit:]

        if reverse:
            filtered = list(reversed(filtered))

        return tuple(entry.to_mapping() for entry in filtered)

    @staticmethod
    def _normalize_filter(
        raw: str | Collection[str] | None,
        *,
        case_sensitive: bool = False,
    ) -> set[str] | None:
        if raw is None:
            return None
        values: Collection[Any]
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, Collection):
            values = raw
        else:
            return None
        normalized: set[str] = set()
        for item in values:
            if item in (None, ""):
                continue
            text = str(item).strip()
            if not text:
                continue
            if case_sensitive:
                normalized.add(text)
            else:
                normalized.add(text.lower())
        return normalized or None

    @staticmethod
    def _normalize_timestamp(value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            stamp = value
        else:
            text = str(value).strip()
            if not text:
                return None
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                stamp = datetime.fromisoformat(text)
            except ValueError:
                return None
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        return stamp.astimezone(timezone.utc)


__all__ = ["DecisionAuditLog", "DecisionAuditRecord"]
