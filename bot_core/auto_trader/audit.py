"""Runtime decision audit log for AutoTrader."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import (
    Any,
    Callable,
    Collection,
    Iterable,
    Mapping,
    MutableMapping,
    Sequence,
)

try:  # pragma: no cover - optional pandas import for type checking
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas is optional at runtime
    pd = None  # type: ignore[assignment]


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
    decision_id: str | None = None

    def to_mapping(self) -> MutableMapping[str, object]:
        data: MutableMapping[str, object] = {
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "stage": self.stage,
            "symbol": self.symbol,
            "mode": self.mode,
            "payload": dict(self.payload),
        }
        if self.decision_id is not None:
            data["decision_id"] = self.decision_id
        if self.risk_snapshot is not None:
            data["risk_snapshot"] = dict(self.risk_snapshot)
        if self.portfolio_snapshot is not None:
            data["portfolio_snapshot"] = dict(self.portfolio_snapshot)
        if self.metadata is not None:
            data["metadata"] = dict(self.metadata)
        return data


class DecisionAuditLog:
    """Thread-safe in-memory audit log with optional retention limits."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        max_age_s: float | int | None = None,
    ) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_age_s: float | None
        if max_age_s is None:
            self._max_age_s = None
        else:
            try:
                candidate = float(max_age_s)
            except (TypeError, ValueError):
                candidate = 0.0
            self._max_age_s = candidate if candidate > 0.0 else None
        self._entries: list[DecisionAuditRecord] = []
        self._lock = Lock()
        self._listeners: set[Callable[[DecisionAuditRecord], None]] = set()

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
        decision_id: str | None = None,
    ) -> DecisionAuditRecord:
        stamp = timestamp or datetime.now(timezone.utc)
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        stamp = stamp.astimezone(timezone.utc)
        payload_dict = dict(payload or {})
        normalized_decision_id = self._normalize_decision_id(
            decision_id,
            payload_dict,
            metadata,
        )
        record = DecisionAuditRecord(
            timestamp=stamp,
            stage=str(stage),
            symbol=str(symbol),
            mode=str(mode),
            payload=payload_dict,
            risk_snapshot=dict(risk_snapshot or {}) or None,
            portfolio_snapshot=dict(portfolio_snapshot or {}) or None,
            metadata=dict(metadata or {}) or None,
            decision_id=normalized_decision_id,
        )
        with self._lock:
            self._entries.append(record)
            self._enforce_retention()
        self._notify_listeners(record)
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
        decision_id: str | Collection[str] | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
    ) -> Sequence[Mapping[str, object]]:
        filtered = self._select_records(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
            since=since,
            until=until,
        )
        return tuple(entry.to_mapping() for entry in filtered)

    def summarize(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        decision_id: str | Collection[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Mapping[str, Any]:
        records = self._select_records(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )

        total = len(records)
        stages = Counter()
        symbols = Counter()
        modes = Counter()
        decision_ids = Counter()
        risk_snapshot_count = 0
        portfolio_snapshot_count = 0

        for record in records:
            stages[record.stage] += 1
            symbols[record.symbol] += 1
            modes[record.mode] += 1
            if record.decision_id:
                decision_ids[record.decision_id] += 1
            if record.risk_snapshot:
                risk_snapshot_count += 1
            if record.portfolio_snapshot:
                portfolio_snapshot_count += 1

        return {
            "count": total,
            "stages": dict(stages),
            "symbols": dict(symbols),
            "modes": dict(modes),
            "decision_ids": dict(decision_ids),
            "unique_decision_ids": len(decision_ids),
            "with_risk_snapshot": risk_snapshot_count,
            "with_portfolio_snapshot": portfolio_snapshot_count,
        }

    def to_dataframe(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        decision_id: str | Collection[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | None = timezone.utc,
    ) -> "pd.DataFrame":
        if pd is None:  # pragma: no cover - optional dependency guard
            raise RuntimeError("pandas is required to export the decision audit log as a DataFrame")

        tzinfo = timezone_hint or timezone.utc
        records = self._select_records(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )

        rows: list[dict[str, Any]] = []
        for record in records:
            rows.append(
                {
                    "timestamp": record.timestamp.astimezone(tzinfo),
                    "stage": record.stage,
                    "symbol": record.symbol,
                    "mode": record.mode,
                    "decision_id": record.decision_id,
                    "payload": dict(record.payload),
                    "risk_snapshot": dict(record.risk_snapshot or {}),
                    "portfolio_snapshot": dict(record.portfolio_snapshot or {}),
                    "metadata": dict(record.metadata or {}),
                }
            )

        if not rows:
            df = pd.DataFrame(
                {
                    "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
                    "stage": pd.Series(dtype="object"),
                    "symbol": pd.Series(dtype="object"),
                    "mode": pd.Series(dtype="object"),
                    "decision_id": pd.Series(dtype="object"),
                    "payload": pd.Series(dtype="object"),
                    "risk_snapshot": pd.Series(dtype="object"),
                    "portfolio_snapshot": pd.Series(dtype="object"),
                    "metadata": pd.Series(dtype="object"),
                }
            )
        else:
            df = pd.DataFrame(rows)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df.attrs["audit_filters"] = {
            "limit": limit,
            "reverse": reverse,
            "stage": stage,
            "symbol": symbol,
            "mode": mode,
            "decision_id": decision_id,
            "since": since,
            "until": until,
            "has_risk_snapshot": has_risk_snapshot,
            "has_portfolio_snapshot": has_portfolio_snapshot,
            "timezone_hint": tzinfo.tzname(None) if hasattr(tzinfo, "tzname") else tzinfo,
        }
        return df

    def group_by_decision(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        decision_id: str | Collection[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | None = timezone.utc,
        include_unidentified: bool = False,
    ) -> Mapping[str | None, Sequence[Mapping[str, object]]]:
        records = self._select_records(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )

        groups: dict[str | None, list[Mapping[str, object]]] = defaultdict(list)
        tzinfo = timezone_hint or timezone.utc
        for record in records:
            key = record.decision_id
            if key is None and not include_unidentified:
                continue
            mapping = record.to_mapping()
            mapping["timestamp"] = record.timestamp.astimezone(tzinfo).isoformat()
            groups[key].append(mapping)
        return {key: tuple(values) for key, values in groups.items()}

    def trace_decision(
        self,
        decision_id: Any,
        *,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | None = timezone.utc,
        include_payload: bool = True,
        include_snapshots: bool = True,
        include_metadata: bool = True,
    ) -> Sequence[Mapping[str, object]]:
        normalized_id = self._normalize_single_token(decision_id)
        if normalized_id is None:
            return ()

        records = self._select_records(
            limit=None,
            reverse=False,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=[normalized_id],
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )

        tzinfo = timezone_hint or timezone.utc
        results: list[Mapping[str, object]] = []
        for record in records:
            mapping = record.to_mapping()
            mapping["timestamp"] = record.timestamp.astimezone(tzinfo).isoformat()
            if not include_payload:
                mapping.pop("payload", None)
            if not include_snapshots:
                mapping.pop("risk_snapshot", None)
                mapping.pop("portfolio_snapshot", None)
            if not include_metadata:
                mapping.pop("metadata", None)
            results.append(mapping)
        return tuple(results)

    def add_listener(self, listener: Callable[[DecisionAuditRecord], None]) -> None:
        self._listeners.add(listener)

    def remove_listener(self, listener: Callable[[DecisionAuditRecord], None]) -> bool:
        try:
            self._listeners.remove(listener)
        except KeyError:
            return False
        return True

    def trim(
        self,
        *,
        before: datetime | str | None = None,
        max_age_s: float | int | None = None,
    ) -> int:
        threshold = self._normalize_timestamp(before)
        if max_age_s not in (None, 0, 0.0):
            try:
                age = float(max_age_s)
            except (TypeError, ValueError):
                age = 0.0
            if age > 0.0:
                now = datetime.now(timezone.utc)
                age_threshold = now - timedelta(seconds=age)
                if threshold is None or age_threshold > threshold:
                    threshold = age_threshold
        if threshold is None:
            return 0

        with self._lock:
            original_len = len(self._entries)
            self._entries = [
                record for record in self._entries if record.timestamp >= threshold
            ]
            removed = original_len - len(self._entries)
        return removed

    def export(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Collection[str] | None = None,
        symbol: str | Collection[str] | None = None,
        mode: str | Collection[str] | None = None,
        decision_id: str | Collection[str] | None = None,
        since: datetime | str | None = None,
        until: datetime | str | None = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | None = timezone.utc,
    ) -> Mapping[str, object]:
        records = self._select_records(
            limit=limit,
            reverse=reverse,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=decision_id,
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )
        tzinfo = timezone_hint or timezone.utc
        entries = []
        for record in records:
            mapping = record.to_mapping()
            mapping["timestamp"] = record.timestamp.astimezone(tzinfo).isoformat()
            entries.append(mapping)

        return {
            "version": 1,
            "entries": entries,
            "retention": {
                "max_entries": self._max_entries,
                "max_age_s": self._max_age_s,
            },
            "filters": {
                "limit": limit,
                "reverse": reverse,
                "stage": self._normalize_token_filter(stage),
                "symbol": self._normalize_token_filter(symbol, case_sensitive=True),
                "mode": self._normalize_token_filter(mode),
                "decision_id": self._normalize_token_filter(decision_id),
                "since": since,
                "until": until,
                "has_risk_snapshot": has_risk_snapshot,
                "has_portfolio_snapshot": has_portfolio_snapshot,
                "timezone_hint": tzinfo.tzname(None) if hasattr(tzinfo, "tzname") else tzinfo,
            },
        }

    def load(
        self,
        payload: Mapping[str, object],
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        entries = payload.get("entries")
        if not isinstance(entries, Sequence):
            return 0

        retention = payload.get("retention", {})
        max_entries = getattr(retention, "get", lambda *_: None)("max_entries")
        if isinstance(max_entries, int) and max_entries > 0:
            self._max_entries = max_entries
        max_age = getattr(retention, "get", lambda *_: None)("max_age_s")
        if max_age in (None, ""):
            self._max_age_s = None
        elif isinstance(max_age, (int, float)) and float(max_age) > 0.0:
            self._max_age_s = float(max_age)

        parsed: list[DecisionAuditRecord] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            timestamp = self._normalize_timestamp(entry.get("timestamp"))
            if timestamp is None:
                continue
            record = DecisionAuditRecord(
                timestamp=timestamp,
                stage=str(entry.get("stage", "")),
                symbol=str(entry.get("symbol", "")),
                mode=str(entry.get("mode", "")),
                payload=dict(entry.get("payload", {})),
                risk_snapshot=dict(entry.get("risk_snapshot", {})) or None,
                portfolio_snapshot=dict(entry.get("portfolio_snapshot", {})) or None,
                metadata=dict(entry.get("metadata", {})) or None,
                decision_id=self._normalize_single_token(
                    entry.get("decision_id")
                    or entry.get("payload", {}).get("decision_id")
                ),
            )
            parsed.append(record)

        if not merge:
            with self._lock:
                self._entries = parsed
                self._enforce_retention()
        else:
            with self._lock:
                self._entries.extend(parsed)
                self._entries.sort(key=lambda item: item.timestamp)
                self._enforce_retention()

        if notify_listeners:
            for record in parsed:
                self._notify_listeners(record)
        return len(parsed)

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

    @staticmethod
    def _normalize_token_filter(
        raw: str | Iterable[object] | None,
        *,
        case_sensitive: bool = False,
    ) -> set[str] | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            values: Iterable[object] = [raw]
        elif isinstance(raw, Iterable):
            values = raw
        else:
            return None
        normalized: set[str] = set()
        for value in values:
            if value in (None, ""):
                continue
            text = str(value).strip()
            if not text:
                continue
            if not case_sensitive:
                text = text.lower()
            normalized.add(text)
        return normalized or None

    @staticmethod
    def _normalize_single_token(value: Any) -> str | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _normalize_decision_id(
        decision_id: str | None,
        payload: Mapping[str, object] | None,
        metadata: Mapping[str, object] | None,
    ) -> str | None:
        if decision_id:
            return str(decision_id)
        for container in (payload, metadata):
            if not isinstance(container, Mapping):
                continue
            candidate = container.get("decision_id")
            normalized = DecisionAuditLog._normalize_single_token(candidate)
            if normalized:
                return normalized
        return None

    def _select_records(
        self,
        *,
        limit: int | None,
        reverse: bool,
        stage: str | Collection[str] | None,
        symbol: str | Collection[str] | None,
        mode: str | Collection[str] | None,
        decision_id: str | Collection[str] | None,
        since: datetime | str | None,
        until: datetime | str | None,
        has_risk_snapshot: bool | None,
        has_portfolio_snapshot: bool | None,
    ) -> list[DecisionAuditRecord]:
        normalized_limit: int | None = None
        if limit is not None:
            try:
                normalized_limit = int(limit)
            except (TypeError, ValueError):
                normalized_limit = 0
            if normalized_limit <= 0:
                return []

        stage_filter = self._normalize_filter(stage)
        symbol_filter = self._normalize_filter(symbol, case_sensitive=True)
        mode_filter = self._normalize_filter(mode)
        decision_filter = self._normalize_token_filter(decision_id, case_sensitive=True)
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
            if decision_filter and (
                record.decision_id is None
                or record.decision_id not in decision_filter
            ):
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

        if normalized_limit is not None and len(filtered) > normalized_limit:
            filtered = filtered[-normalized_limit:]

        if reverse:
            filtered = list(reversed(filtered))

        return filtered

    def _enforce_retention(self) -> None:
        if self._max_age_s:
            threshold = datetime.now(timezone.utc) - timedelta(seconds=self._max_age_s)
            self._entries = [
                record for record in self._entries if record.timestamp >= threshold
            ]
        if len(self._entries) > self._max_entries:
            excess = len(self._entries) - self._max_entries
            del self._entries[0:excess]

    def _notify_listeners(self, record: DecisionAuditRecord) -> None:
        if not self._listeners:
            return
        listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(record)
            except Exception:  # pragma: no cover - listeners should not break logging
                continue


__all__ = ["DecisionAuditLog", "DecisionAuditRecord"]
