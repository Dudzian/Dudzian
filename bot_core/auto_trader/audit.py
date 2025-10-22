"""Runtime decision audit log for AutoTrader."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Mapping, MutableMapping, Sequence


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


__all__ = ["DecisionAuditLog", "DecisionAuditRecord"]
