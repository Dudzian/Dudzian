"""Obsługa override alertów opartych o statusy SLO Stage6."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping, MutableMapping, Sequence

from bot_core.observability.slo import SLODefinition, SLOStatus

_SCHEMA = "stage6.observability.alert_overrides"
_SCHEMA_VERSION = "1.0"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _isoformat(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _to_epoch_millis(value: datetime | None) -> int | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def _to_float_mapping(metadata: Mapping[str, object] | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if not metadata:
        return result
    for key, value in metadata.items():
        if isinstance(value, (int, float)):
            result[str(key)] = float(value)
    return result


@dataclass(slots=True)
class AlertOverride:
    """Definicja pojedynczego override'u alertu."""

    alert: str
    status: str
    severity: str
    reason: str | None
    indicator: str | None = None
    source: str = "slo_monitor"
    created_at: datetime = field(default_factory=_utcnow)
    expires_at: datetime | None = None
    requested_by: str | None = None
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "alert": self.alert,
            "status": self.status,
            "severity": self.severity,
            "reason": self.reason,
            "indicator": self.indicator,
            "source": self.source,
            "created_at": _isoformat(self.created_at),
            "expires_at": _isoformat(self.expires_at),
            "requested_by": self.requested_by,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AlertOverride":
        def _parse_dt(value: object) -> datetime | None:
            if value in (None, ""):
                return None
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            try:
                return datetime.fromisoformat(text)
            except ValueError:
                return None

        metadata = _to_float_mapping(payload.get("metadata") if isinstance(payload, Mapping) else None)
        tags_value = payload.get("tags") if isinstance(payload, Mapping) else None
        if isinstance(tags_value, Sequence) and not isinstance(tags_value, (str, bytes)):
            tags = tuple(str(tag) for tag in tags_value)
        else:
            tags = ()
        return cls(
            alert=str(payload.get("alert")),
            status=str(payload.get("status", "unknown")),
            severity=str(payload.get("severity", "warning")),
            reason=(str(payload.get("reason")) if payload.get("reason") not in (None, "") else None),
            indicator=(
                str(payload.get("indicator"))
                if payload.get("indicator") not in (None, "")
                else None
            ),
            source=str(payload.get("source", "slo_monitor")),
            created_at=_parse_dt(payload.get("created_at")) or _utcnow(),
            expires_at=_parse_dt(payload.get("expires_at")),
            requested_by=(
                str(payload.get("requested_by"))
                if payload.get("requested_by") not in (None, "")
                else None
            ),
            tags=tags,
            metadata=metadata,
        )

    def is_active(self, *, reference: datetime | None = None) -> bool:
        reference = reference or _utcnow()
        if self.expires_at is None:
            return True
        return self.expires_at > reference

    def to_annotation(self) -> dict[str, object]:
        tags = ["override", self.status, self.severity]
        if self.indicator:
            tags.append(self.indicator)
        tags.extend(self.tags)
        seen: dict[str, None] = {}
        unique_tags = [tag for tag in tags if not (tag in seen or seen.setdefault(tag, None))]
        return {
            "title": f"{self.alert} {self.status.upper()}",
            "text": self.reason,
            "severity": self.severity,
            "alert": self.alert,
            "indicator": self.indicator,
            "expires_at": _isoformat(self.expires_at),
            "source": self.source,
            "tags": unique_tags,
            "metadata": dict(self.metadata),
        }

    def to_grafana_event(
        self,
        *,
        dashboard_uid: str | None = None,
        panel_id: int | None = None,
    ) -> dict[str, object]:
        annotation = self.to_annotation()
        payload: dict[str, object] = {
            "dashboardUid": dashboard_uid,
            "panelId": panel_id,
            "time": _to_epoch_millis(self.created_at),
            "timeEnd": _to_epoch_millis(self.expires_at),
            "tags": annotation["tags"],
            "text": annotation["text"] or annotation["title"],
            "title": annotation["title"],
            "data": {
                "alert": self.alert,
                "status": self.status,
                "severity": self.severity,
                "source": self.source,
                "indicator": self.indicator,
                "metadata": annotation["metadata"],
            },
        }
        return {key: value for key, value in payload.items() if value not in (None, {})}


class AlertOverrideManager:
    """Zarządza zestawem override'ów i buduje podsumowania."""

    def __init__(self, overrides: Iterable[AlertOverride] | None = None) -> None:
        self._overrides = list(overrides or [])

    def add(self, override: AlertOverride) -> None:
        self._overrides.append(override)

    def upsert(self, override: AlertOverride) -> None:
        for index, existing in enumerate(self._overrides):
            if existing.alert == override.alert and existing.status == override.status:
                self._overrides[index] = override
                return
        self._overrides.append(override)

    def extend(self, overrides: Iterable[AlertOverride]) -> None:
        for override in overrides:
            self.add(override)

    def merge(self, overrides: Iterable[AlertOverride]) -> None:
        for override in overrides:
            self.upsert(override)

    def prune_expired(self, *, reference: datetime | None = None) -> None:
        reference = reference or _utcnow()
        self._overrides = [
            override for override in self._overrides if override.is_active(reference=reference)
        ]

    def active(self, *, reference: datetime | None = None) -> list[AlertOverride]:
        return [override for override in self._overrides if override.is_active(reference=reference)]

    def summary(self, *, reference: datetime | None = None) -> dict[str, object]:
        reference = reference or _utcnow()
        active_overrides = self.active(reference=reference)
        counts_status: MutableMapping[str, int] = {}
        counts_severity: MutableMapping[str, int] = {}
        expires: list[datetime] = []
        for override in active_overrides:
            counts_status[override.status] = counts_status.get(override.status, 0) + 1
            counts_severity[override.severity] = counts_severity.get(override.severity, 0) + 1
            if override.expires_at is not None:
                expires.append(override.expires_at)
        latest_expiry = max(expires).astimezone(timezone.utc) if expires else None
        return {
            "active": len(active_overrides),
            "counts_by_status": dict(counts_status),
            "counts_by_severity": dict(counts_severity),
            "latest_expiry": _isoformat(latest_expiry),
        }

    def to_payload(self, *, reference: datetime | None = None) -> dict[str, object]:
        reference = reference or _utcnow()
        overrides_payload = [override.to_dict() for override in self._overrides]
        annotations = [override.to_annotation() for override in self.active(reference=reference)]
        payload = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "generated_at": _isoformat(reference),
            "overrides": overrides_payload,
            "summary": self.summary(reference=reference),
            "annotations": annotations,
        }
        return payload


class AlertOverrideBuilder:
    """Buduje override'y na podstawie statusów SLO i konfiguracji."""

    def __init__(self, definitions: Mapping[str, SLODefinition] | None = None) -> None:
        self._definitions = dict(definitions or {})

    def build_from_statuses(
        self,
        statuses: Mapping[str, SLOStatus],
        *,
        include_warning: bool = True,
        default_ttl: timedelta = timedelta(minutes=60),
        severity_overrides: Mapping[str, str] | None = None,
        requested_by: str | None = None,
        source: str = "slo_monitor",
        extra_tags: Sequence[str] | None = None,
        reference: datetime | None = None,
    ) -> list[AlertOverride]:
        severity_map = {"breach": "critical", "warning": "warning"}
        if severity_overrides:
            severity_map.update({str(k): str(v) for k, v in severity_overrides.items()})
        reference = reference or _utcnow()
        overrides: list[AlertOverride] = []
        for name, status in statuses.items():
            if status.status not in {"breach", "warning"}:
                continue
            if status.status == "warning" and not include_warning:
                continue
            definition = self._definitions.get(name)
            indicator = status.indicator or (definition.indicator if definition else None)
            tags: list[str] = []
            if extra_tags:
                tags.extend(extra_tags)
            if definition and definition.tags:
                tags.extend(str(tag) for tag in definition.tags)
            severity = severity_map.get(status.status, status.severity or "warning")
            expires_at = reference + default_ttl if default_ttl.total_seconds() > 0 else None
            metadata = _to_float_mapping(status.metadata)
            if status.error_budget_pct is not None:
                metadata.setdefault("error_budget_pct", float(status.error_budget_pct))
            override = AlertOverride(
                alert=name,
                status=status.status,
                severity=severity,
                reason=status.reason,
                indicator=indicator,
                source=source,
                created_at=reference,
                expires_at=expires_at,
                requested_by=requested_by,
                tags=tuple(tags),
                metadata=metadata,
            )
            overrides.append(override)
        return overrides


def load_overrides_document(data: Mapping[str, object]) -> list[AlertOverride]:
    if data.get("schema") != _SCHEMA:
        raise ValueError("Dokument override nie posiada poprawnego schematu Stage6")
    entries = data.get("overrides")
    if not isinstance(entries, Sequence):
        raise ValueError("Sekcja 'overrides' powinna być listą")
    return [AlertOverride.from_dict(entry) for entry in entries if isinstance(entry, Mapping)]


__all__ = [
    "AlertOverride",
    "AlertOverrideManager",
    "AlertOverrideBuilder",
    "load_overrides_document",
]

