"""Obsługa rejestrowania szkoleń Stage5 wraz z podpisami HMAC."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.security.signing import build_hmac_signature


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_sequence(value: Sequence[str] | None) -> list[str]:
    if not value:
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_mapping(value: Mapping[str, str] | None) -> dict[str, str]:
    if not value:
        return {}
    return {str(k): str(v) for k, v in value.items() if str(k)}


@dataclass(slots=True)
class TrainingSession:
    """Reprezentuje szkolenie Stage5 wymagające udokumentowania."""

    session_id: str
    title: str
    trainer: str
    participants: Sequence[str]
    topics: Sequence[str]
    occurred_at: datetime
    duration_minutes: int | float
    summary: str
    actions: Mapping[str, str] = field(default_factory=dict)
    materials: Sequence[str] = field(default_factory=list)
    compliance_tags: Sequence[str] = field(default_factory=list)
    metadata: Mapping[str, object] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        occurred_at_dt = self.occurred_at
        if occurred_at_dt.tzinfo is None:
            occurred_at_dt = occurred_at_dt.replace(tzinfo=timezone.utc)
        occurred_at = occurred_at_dt.astimezone(timezone.utc).isoformat()
        payload: MutableMapping[str, object] = {
            "type": "stage5_training",
            "session_id": str(self.session_id),
            "title": str(self.title),
            "trainer": str(self.trainer),
            "participants": _normalize_sequence(self.participants),
            "topics": _normalize_sequence(self.topics),
            "occurred_at": occurred_at,
            "duration_minutes": float(self.duration_minutes),
            "summary": str(self.summary),
        }
        actions = _normalize_mapping(self.actions)
        if actions:
            payload["actions"] = actions
        materials = _normalize_sequence(self.materials)
        if materials:
            payload["materials"] = materials
        tags = sorted({tag.lower() for tag in _normalize_sequence(self.compliance_tags)})
        if tags:
            payload["compliance_tags"] = tags
        extra = self.metadata
        if extra:
            payload["metadata"] = extra
        return dict(payload)


def build_training_log_entry(
    session: TrainingSession,
    *,
    logged_at: datetime | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> dict[str, object]:
    payload = session.to_payload()
    logged_dt = logged_at or _utcnow()
    if logged_dt.tzinfo is None:
        logged_dt = logged_dt.replace(tzinfo=timezone.utc)
    payload.setdefault("logged_at", logged_dt.astimezone(timezone.utc).isoformat())
    if signing_key:
        payload["signature"] = build_hmac_signature(payload, key=signing_key, key_id=signing_key_id)
    return payload


def write_training_log(
    session: TrainingSession,
    *,
    output: str | Path,
    logged_at: datetime | None = None,
    signing_key: bytes | None = None,
    signing_key_id: str | None = None,
) -> Path:
    entry = build_training_log_entry(
        session,
        logged_at=logged_at,
        signing_key=signing_key,
        signing_key_id=signing_key_id,
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(entry, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    target.write_text(serialized + "\n", encoding="utf-8")
    return target


__all__ = ["TrainingSession", "build_training_log_entry", "write_training_log"]
