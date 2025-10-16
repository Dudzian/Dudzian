"""Raportowanie rotacji kluczy API dla hypercare Stage5."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.security.rotation import RotationStatus
from bot_core.security.signing import build_hmac_signature


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    return _ensure_utc(value).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _round_optional(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


@dataclass(slots=True)
class RotationRecord:
    """Opis pojedynczego wpisu rotacji zapisanego w raporcie."""

    environment: str
    key: str
    purpose: str
    registry_path: Path
    status_before: RotationStatus
    rotated_at: datetime
    interval_days: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        rotated_at = _ensure_utc(self.rotated_at)
        next_due_at = rotated_at + timedelta(days=self.interval_days)

        payload: MutableMapping[str, object] = {
            "environment": str(self.environment),
            "key": str(self.key),
            "purpose": str(self.purpose),
            "registry_path": str(self.registry_path),
            "rotated_at": _format_timestamp(rotated_at),
            "interval_days": float(self.interval_days),
            "next_due_at": _format_timestamp(next_due_at),
            "was_due": bool(self.status_before.is_due),
            "was_overdue": bool(self.status_before.is_overdue),
        }

        if self.status_before.last_rotated is not None:
            payload["previous_rotation"] = _format_timestamp(self.status_before.last_rotated)
        if self.status_before.days_since_rotation is not None:
            payload["days_since_previous"] = _round_optional(self.status_before.days_since_rotation)
        payload["due_in_days_before"] = _round_optional(self.status_before.due_in_days)

        if self.metadata:
            payload["metadata"] = dict(self.metadata)

        return dict(payload)


@dataclass(slots=True)
class RotationSummary:
    """Podsumowanie rotacji przeprowadzonej podczas jednej sesji."""

    operator: str
    executed_at: datetime
    records: Sequence[RotationRecord]
    notes: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        executed_at = _ensure_utc(self.executed_at)
        records_payload = [record.to_payload() for record in self.records]

        overdue_before = sum(1 for record in self.records if record.status_before.is_overdue)
        due_before = sum(1 for record in self.records if record.status_before.is_due)

        payload: MutableMapping[str, object] = {
            "type": "stage5_key_rotation",
            "operator": str(self.operator),
            "executed_at": _format_timestamp(executed_at),
            "records": records_payload,
            "stats": {
                "total": len(records_payload),
                "due_before": due_before,
                "overdue_before": overdue_before,
            },
        }

        if self.notes:
            payload["notes"] = str(self.notes)
        if self.metadata:
            payload["metadata"] = dict(self.metadata)

        return dict(payload)


def build_rotation_summary_entry(
    summary: RotationSummary,
    *,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> dict[str, object]:
    payload = summary.to_payload()
    if signing_key:
        payload["signature"] = build_hmac_signature(payload, key=signing_key, key_id=signing_key_id)
    return payload


def write_rotation_summary(
    summary: RotationSummary,
    *,
    output: str | Path,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> Path:
    entry = build_rotation_summary_entry(
        summary,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    target.write_text(serialized + "\n", encoding="utf-8")
    return target


__all__ = [
    "RotationRecord",
    "RotationSummary",
    "build_rotation_summary_entry",
    "write_rotation_summary",
]

