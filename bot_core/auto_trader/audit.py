"""Runtime decision audit log for AutoTrader."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


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

    @staticmethod
    def _normalize_token_filter(
        value: str | Iterable[object] | None,
    ) -> frozenset[str] | None:
        if value is None:
            return None
        if isinstance(value, str):
            token = value.strip()
            if not token:
                return frozenset()
            return frozenset({token})
        if isinstance(value, Iterable):
            tokens: set[str] = set()
            for item in value:
                if item is None:
                    continue
                token = str(item).strip()
                if token:
                    tokens.add(token)
            return frozenset(tokens)
        raise TypeError("filters must be strings or iterables of strings")

    @staticmethod
    def _normalize_time_bound(value: Any) -> datetime | None:
        if value is None:
            return None
        candidate: datetime
        if isinstance(value, datetime):
            candidate = value
        elif isinstance(value, (int, float)):
            candidate = datetime.fromtimestamp(float(value), timezone.utc)
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("time bounds must not be empty strings")
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                candidate = datetime.fromisoformat(text)
            except ValueError as exc:  # pragma: no cover - validation guard
                raise ValueError(f"invalid datetime string: {value!r}") from exc
        else:
            raise TypeError(
                "time bounds must be datetime, ISO datetime strings or UNIX timestamps",
            )
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=timezone.utc)
        else:
            candidate = candidate.astimezone(timezone.utc)
        return candidate

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

    def query(
        self,
        *,
        limit: int | None = 20,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Sequence[DecisionAuditRecord]:
        stage_filter = self._normalize_token_filter(stage)
        symbol_filter = self._normalize_token_filter(symbol)
        mode_filter = self._normalize_token_filter(mode)
        since_bound = self._normalize_time_bound(since)
        until_bound = self._normalize_time_bound(until)

        normalized_limit: int | None
        if limit is None:
            normalized_limit = None
        else:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):  # pragma: no cover - validation guard
                normalized_limit = None
            else:
                if normalized_limit <= 0:
                    return ()

        with self._lock:
            entries = list(self._entries)

        filtered: list[DecisionAuditRecord] = []
        for record in entries:
            if stage_filter is not None and record.stage not in stage_filter:
                continue
            if symbol_filter is not None and record.symbol not in symbol_filter:
                continue
            if mode_filter is not None and record.mode not in mode_filter:
                continue
            if since_bound is not None and record.timestamp < since_bound:
                continue
            if until_bound is not None and record.timestamp > until_bound:
                continue
            if has_risk_snapshot is not None:
                if bool(record.risk_snapshot) != has_risk_snapshot:
                    continue
            if has_portfolio_snapshot is not None:
                if bool(record.portfolio_snapshot) != has_portfolio_snapshot:
                    continue
            filtered.append(record)

        if normalized_limit is not None and len(filtered) > normalized_limit:
            filtered = filtered[-normalized_limit:]

        if reverse:
            filtered = list(reversed(filtered))

        return tuple(filtered)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def to_dicts(self, limit: int = 20) -> Sequence[Mapping[str, object]]:
        return self.query_dicts(limit=limit)

    def query_dicts(
        self,
        *,
        limit: int | None = 20,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Sequence[Mapping[str, object]]:
        records = self.query(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )
        return tuple(record.to_mapping() for record in records)


__all__ = ["DecisionAuditLog", "DecisionAuditRecord"]
