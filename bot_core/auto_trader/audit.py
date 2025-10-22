"""Runtime decision audit log for AutoTrader."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, tzinfo
import logging
import math
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency guard
    import json
except ModuleNotFoundError:  # pragma: no cover - fallback for constrained runtimes
    json = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


AuditListener = Callable[["DecisionAuditRecord"], None]


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

    def to_mapping(
        self,
        *,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ) -> MutableMapping[str, object]:
        data: MutableMapping[str, object] = {
            "timestamp": (
                self.timestamp
                if timezone_hint is None
                else self.timestamp.astimezone(timezone_hint)
            ).isoformat(),
            "stage": self.stage,
            "symbol": self.symbol,
            "mode": self.mode,
            "payload": dict(self.payload),
            "decision_id": self.decision_id,
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

    def __init__(self, *, max_entries: int = 512, max_age_s: float | None = None) -> None:
        self._max_entries = max(1, int(max_entries))
        self._max_age_s = self._normalize_age(max_age_s)
        self._entries: list[DecisionAuditRecord] = []
        self._lock = Lock()
        self._listeners: set[AuditListener] = set()

    @staticmethod
    def _normalize_age(value: float | int | None) -> float | None:
        if value is None:
            return None
        try:
            seconds = float(value)
        except (TypeError, ValueError):  # pragma: no cover - validation guard
            raise TypeError("max_age_s must be a positive number") from None
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("max_age_s must be a finite positive number")
        return seconds

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
    def _normalize_decision_id(value: Any | None) -> str | None:
        if value is None:
            return None
        token = str(value).strip()
        return token or None

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
        decision_id: str | None = None,
    ) -> DecisionAuditRecord:
        stamp = timestamp or datetime.now(timezone.utc)
        normalized_decision_id = self._normalize_decision_id(decision_id)
        record = DecisionAuditRecord(
            timestamp=stamp,
            stage=str(stage),
            symbol=str(symbol),
            mode=str(mode),
            payload=dict(payload or {}),
            risk_snapshot=dict(risk_snapshot or {}) or None,
            portfolio_snapshot=dict(portfolio_snapshot or {}) or None,
            metadata=dict(metadata or {}) or None,
            decision_id=normalized_decision_id,
        )
        with self._lock:
            self._entries.append(record)
            self._trim_locked(reference_time=stamp)
            if len(self._entries) > self._max_entries:
                excess = len(self._entries) - self._max_entries
                del self._entries[0:excess]
            listeners = tuple(self._listeners)
        self._notify_listeners(record, listeners)
        return record

    def add_listener(self, listener: AuditListener) -> None:
        if not callable(listener):
            raise TypeError("listener must be callable")
        with self._lock:
            self._listeners.add(listener)

    def remove_listener(self, listener: AuditListener) -> bool:
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)
                return True
        return False

    def _notify_listeners(
        self, record: DecisionAuditRecord, listeners: Sequence[AuditListener]
    ) -> None:
        if not listeners:
            return
        for listener in listeners:
            try:
                listener(record)
            except Exception:  # pragma: no cover - listeners must not break logging
                LOGGER.debug("Decision audit listener failed", exc_info=True)

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
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Sequence[DecisionAuditRecord]:
        stage_filter = self._normalize_token_filter(stage)
        symbol_filter = self._normalize_token_filter(symbol)
        mode_filter = self._normalize_token_filter(mode)
        decision_filter = self._normalize_token_filter(decision_id)
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

        reference_time = datetime.now(timezone.utc)
        with self._lock:
            self._trim_locked(reference_time=reference_time)
            entries = list(self._entries)

        filtered: list[DecisionAuditRecord] = []
        for record in entries:
            if stage_filter is not None and record.stage not in stage_filter:
                continue
            if symbol_filter is not None and record.symbol not in symbol_filter:
                continue
            if mode_filter is not None and record.mode not in mode_filter:
                continue
            if decision_filter is not None:
                record_decision_id = record.decision_id
                if record_decision_id is None or record_decision_id not in decision_filter:
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

    def trim(
        self,
        *,
        before: Any | None = None,
        max_age_s: float | int | None = None,
    ) -> int:
        """Remove entries older than the provided cut-offs."""

        normalized_before: datetime | None = None
        if before is not None:
            normalized_before = self._normalize_time_bound(before)
        normalized_age = self._normalize_age(max_age_s) if max_age_s is not None else None

        reference_time = datetime.now(timezone.utc)
        with self._lock:
            removed = self._trim_locked(
                reference_time=reference_time,
                before=normalized_before,
                max_age_s=normalized_age,
            )
        return removed

    def set_retention(
        self,
        *,
        max_entries: int | None = None,
        max_age_s: float | int | None = None,
    ) -> None:
        """Update retention parameters and apply them immediately."""

        if max_entries is not None:
            self._max_entries = max(1, int(max_entries))
        if max_age_s is not None:
            self._max_age_s = self._normalize_age(max_age_s)
        reference_time = datetime.now(timezone.utc)
        with self._lock:
            self._trim_locked(reference_time=reference_time)
            if len(self._entries) > self._max_entries:
                excess = len(self._entries) - self._max_entries
                del self._entries[0:excess]

    def _trim_locked(
        self,
        *,
        reference_time: datetime | None = None,
        before: datetime | None = None,
        max_age_s: float | None = None,
    ) -> int:
        if not self._entries:
            return 0

        cutoff: datetime | None = None
        if before is not None:
            cutoff = before
        else:
            age = max_age_s if max_age_s is not None else self._max_age_s
            if age is not None:
                ref = reference_time or datetime.now(timezone.utc)
                cutoff = ref - timedelta(seconds=age)

        if cutoff is None:
            return 0

        # Ensure UTC timezone for comparison
        if cutoff.tzinfo is None:
            cutoff = cutoff.replace(tzinfo=timezone.utc)
        else:
            cutoff = cutoff.astimezone(timezone.utc)

        kept: list[DecisionAuditRecord] = []
        removed = 0
        for record in self._entries:
            if record.timestamp < cutoff:
                removed += 1
                continue
            kept.append(record)

        if removed:
            self._entries[:] = kept
        return removed

    def __len__(self) -> int:
        reference_time = datetime.now(timezone.utc)
        with self._lock:
            self._trim_locked(reference_time=reference_time)
            return len(self._entries)

    def __bool__(self) -> bool:  # pragma: no cover - trivial truthiness override
        return True

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
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ) -> Sequence[Mapping[str, object]]:
        records = self.query(
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
        return tuple(record.to_mapping(timezone_hint=timezone_hint) for record in records)

    def group_by_decision(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
        include_unidentified: bool = False,
    ) -> Mapping[str | None, Sequence[Mapping[str, object]]]:
        """Return audit entries grouped by decision identifier.

        Entries are filtered using the same semantics as :meth:`query`.  The
        resulting mapping preserves insertion order both for decision groups and
        for records within each group.  Records without an assigned
        ``decision_id`` are omitted unless ``include_unidentified`` is set to
        ``True``.
        """

        records = self.query(
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

        groups: dict[str | None, list[Mapping[str, object]]] = {}
        for record in records:
            key = record.decision_id
            if key is None and not include_unidentified:
                continue
            if key not in groups:
                groups[key] = []
            groups[key].append(record.to_mapping(timezone_hint=timezone_hint))

        return {group_id: tuple(entries) for group_id, entries in groups.items()}

    def trace_decision(
        self,
        decision_id: Any,
        *,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
        include_payload: bool = True,
        include_snapshots: bool = True,
        include_metadata: bool = True,
    ) -> Sequence[Mapping[str, object]]:
        """Return ordered decision pipeline trace enriched with timing metadata."""

        normalized_id = self._normalize_decision_id(decision_id)
        if normalized_id is None:
            return ()

        records = self.query(
            limit=None,
            stage=stage,
            symbol=symbol,
            mode=mode,
            decision_id=[normalized_id],
            since=since,
            until=until,
            has_risk_snapshot=has_risk_snapshot,
            has_portfolio_snapshot=has_portfolio_snapshot,
        )
        if not records:
            return ()

        first_timestamp = records[0].timestamp
        previous_timestamp = first_timestamp
        timeline: list[Mapping[str, object]] = []
        for index, record in enumerate(records):
            mapping = record.to_mapping(timezone_hint=timezone_hint)
            if not include_payload:
                mapping.pop("payload", None)
            if not include_snapshots:
                mapping.pop("risk_snapshot", None)
                mapping.pop("portfolio_snapshot", None)
            if not include_metadata:
                mapping.pop("metadata", None)

            elapsed_first = (record.timestamp - first_timestamp).total_seconds()
            elapsed_previous = (record.timestamp - previous_timestamp).total_seconds()
            mapping["step_index"] = index
            mapping["elapsed_since_first_s"] = float(elapsed_first)
            mapping["elapsed_since_previous_s"] = float(elapsed_previous if index else 0.0)
            timeline.append(mapping)
            previous_timestamp = record.timestamp

        return tuple(timeline)

    def export(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ) -> Mapping[str, object]:
        """Serialise the audit log to a JSON-friendly mapping."""

        records = self.query_dicts(
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
            timezone_hint=timezone_hint,
        )

        def _serialize_token_filter(
            value: str | Iterable[object] | None,
        ) -> Sequence[str] | None:
            normalized = self._normalize_token_filter(value)
            if normalized is None:
                return None
            return tuple(sorted(normalized))

        filters: dict[str, object | None] = {
            "limit": limit,
            "reverse": reverse,
            "stage": _serialize_token_filter(stage),
            "symbol": _serialize_token_filter(symbol),
            "mode": _serialize_token_filter(mode),
            "decision_id": _serialize_token_filter(decision_id),
            "since": since,
            "until": until,
            "has_risk_snapshot": has_risk_snapshot,
            "has_portfolio_snapshot": has_portfolio_snapshot,
            "timezone_hint": timezone_hint.tzname(None)
            if isinstance(timezone_hint, (timezone, tzinfo))
            else timezone_hint,
        }

        # ``max_age_s`` may be ``None`` when no TTL is configured. Preserve the
        # raw attribute value so that the exported payload round-trips via
        # :meth:`load` without losing retention semantics.
        retention = {
            "max_entries": self._max_entries,
            "max_age_s": self._max_age_s,
        }

        return {
            "version": 1,
            "entries": list(records),
            "retention": retention,
            "filters": filters,
        }

    def dump(
        self,
        destination: str | Path,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
        ensure_ascii: bool = False,
    ) -> None:
        """Persist the audit log to a JSON file.

        The method relies on the lightweight :mod:`json` module to avoid a hard
        dependency on :mod:`pandas`.  The resulting file mirrors the structure
        returned by :meth:`export`.
        """

        if json is None:  # pragma: no cover - defensive guard for exotic envs
            raise RuntimeError("json module is required to dump the decision audit log")

        payload = self.export(
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
            timezone_hint=timezone_hint,
        )
        path = Path(destination)
        with path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, ensure_ascii=ensure_ascii, indent=2)

    @staticmethod
    def _deserialize_record(entry: Mapping[str, object]) -> DecisionAuditRecord:
        try:
            timestamp_raw = entry["timestamp"]
            stage = entry["stage"]
            symbol = entry["symbol"]
            mode = entry["mode"]
        except KeyError as exc:  # pragma: no cover - validation guard
            raise ValueError("audit entry is missing required fields") from exc

        timestamp = DecisionAuditLog._normalize_time_bound(timestamp_raw)
        payload = entry.get("payload") or {}
        risk_snapshot = entry.get("risk_snapshot") or None
        portfolio_snapshot = entry.get("portfolio_snapshot") or None
        metadata = entry.get("metadata") or None
        decision_id_raw = entry.get("decision_id")
        decision_id = DecisionAuditLog._normalize_decision_id(decision_id_raw)

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if risk_snapshot is not None and not isinstance(risk_snapshot, Mapping):
            raise TypeError("risk_snapshot must be a mapping or None")
        if portfolio_snapshot is not None and not isinstance(portfolio_snapshot, Mapping):
            raise TypeError("portfolio_snapshot must be a mapping or None")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping or None")

        return DecisionAuditRecord(
            timestamp=timestamp,
            stage=str(stage),
            symbol=str(symbol),
            mode=str(mode),
            payload=dict(payload),
            risk_snapshot=dict(risk_snapshot) if risk_snapshot is not None else None,
            portfolio_snapshot=(
                dict(portfolio_snapshot) if portfolio_snapshot is not None else None
            ),
            metadata=dict(metadata) if metadata is not None else None,
            decision_id=decision_id,
        )

    def load(
        self,
        payload: Mapping[str, object],
        *,
        merge: bool = False,
        notify_listeners: bool = False,
    ) -> int:
        """Load audit entries from a previously exported payload."""

        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping produced by export()")

        entries_payload = payload.get("entries", [])
        if entries_payload is None:
            entries_payload = []
        if not isinstance(entries_payload, Iterable):
            raise TypeError("entries must be an iterable of mappings")

        retention = payload.get("retention", {})
        if retention is None:
            retention = {}
        if not isinstance(retention, Mapping):
            raise TypeError("retention must be a mapping")

        records: list[DecisionAuditRecord] = []
        for entry in entries_payload:
            if not isinstance(entry, Mapping):
                raise TypeError("each entry must be a mapping")
            record = self._deserialize_record(entry)
            records.append(record)

        with self._lock:
            if not merge:
                self._entries = []

            max_entries_raw = retention.get("max_entries")
            if max_entries_raw is not None:
                self._max_entries = max(1, int(max_entries_raw))

            if "max_age_s" in retention:
                max_age_raw = retention["max_age_s"]
                self._max_age_s = (
                    self._normalize_age(max_age_raw) if max_age_raw is not None else None
                )

            self._entries.extend(records)
            self._entries.sort(key=lambda record: record.timestamp)
            reference_time = (
                self._entries[-1].timestamp if self._entries else datetime.now(timezone.utc)
            )
            self._trim_locked(reference_time=reference_time)
            if len(self._entries) > self._max_entries:
                excess = len(self._entries) - self._max_entries
                del self._entries[0:excess]

            loaded = len(records)
            listeners = tuple(self._listeners) if notify_listeners else ()
        if notify_listeners and listeners:
            for record in records:
                self._notify_listeners(record, listeners)
        return loaded

    def to_dataframe(
        self,
        *,
        limit: int | None = 20,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
        timezone_hint: timezone | tzinfo | None = timezone.utc,
    ):
        """Return filtered audit entries as a ``pandas.DataFrame``.

        The resulting frame stores timestamps in UTC by default. When a
        ``timezone_hint`` is provided the timestamps are converted to the
        requested timezone.  The frame includes an ``audit_filters`` attribute
        mirroring the parameters used to build it, enabling callers to retain
        provenance in analytics pipelines.
        """

        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise RuntimeError(
                "pandas is required to export the decision audit log as a DataFrame",
            ) from exc

        records = self.query_dicts(
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
            timezone_hint=timezone_hint,
        )

        if records:
            frame = pd.DataFrame.from_records(records)
        else:
            frame = pd.DataFrame(
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

        if not frame.empty:
            timestamps = pd.to_datetime(frame["timestamp"], utc=True)
            if timezone_hint is not None:
                timestamps = timestamps.dt.tz_convert(timezone_hint)
            frame["timestamp"] = timestamps

        frame.attrs["audit_filters"] = {
            "limit": limit,
            "reverse": reverse,
            "stage": None if stage is None else self._normalize_token_filter(stage),
            "symbol": None if symbol is None else self._normalize_token_filter(symbol),
            "mode": None if mode is None else self._normalize_token_filter(mode),
            "decision_id": None
            if decision_id is None
            else self._normalize_token_filter(decision_id),
            "since": since,
            "until": until,
            "has_risk_snapshot": has_risk_snapshot,
            "has_portfolio_snapshot": has_portfolio_snapshot,
            "timezone_hint": timezone_hint.tzname(None)
            if isinstance(timezone_hint, (timezone, tzinfo))
            else timezone_hint,
        }

        return frame

    def summarize(
        self,
        *,
        limit: int | None = None,
        reverse: bool = False,
        stage: str | Iterable[object] | None = None,
        symbol: str | Iterable[object] | None = None,
        mode: str | Iterable[object] | None = None,
        decision_id: str | Iterable[object] | None = None,
        since: Any = None,
        until: Any = None,
        has_risk_snapshot: bool | None = None,
        has_portfolio_snapshot: bool | None = None,
    ) -> Mapping[str, object]:
        """Aggregate audit log statistics for UI dashboards and monitoring."""

        records = self.query(
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

        count = len(records)
        stages = Counter(record.stage for record in records)
        symbols = Counter(record.symbol for record in records)
        modes = Counter(record.mode for record in records)
        decision_ids = Counter(
            record.decision_id for record in records if record.decision_id is not None
        )
        with_risk_snapshot = sum(1 for record in records if record.risk_snapshot)
        with_portfolio_snapshot = sum(
            1 for record in records if record.portfolio_snapshot
        )

        summary: dict[str, object] = {
            "count": count,
            "stages": dict(stages),
            "symbols": dict(symbols),
            "modes": dict(modes),
            "decision_ids": dict(decision_ids),
            "unique_decision_ids": len(decision_ids),
            "with_risk_snapshot": with_risk_snapshot,
            "with_portfolio_snapshot": with_portfolio_snapshot,
        }

        if records:
            timestamps = [record.timestamp for record in records]
            first_ts = min(timestamps)
            last_ts = max(timestamps)
            summary["first_timestamp"] = first_ts.astimezone(timezone.utc).isoformat()
            summary["last_timestamp"] = last_ts.astimezone(timezone.utc).isoformat()

        return summary


__all__ = ["DecisionAuditLog", "DecisionAuditRecord"]
