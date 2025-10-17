"""Helpers for managing UI user profiles backed by JSON storage.

This module complements the runtime security stack by providing lightweight
utilities to read and persist user profile definitions that the desktop UI can
surface.  The data model intentionally mirrors the structures used by the
token
auditors so that the same configuration sources can be reused.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping

LOGGER = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(slots=True)
class UserProfile:
    """Representation of a single UI user profile."""

    user_id: str
    display_name: str
    roles: tuple[str, ...]
    updated_at: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "UserProfile":
        user_id = str(data.get("user_id", "")).strip()
        if not user_id:
            raise ValueError("Profil użytkownika musi zawierać 'user_id'.")
        display_name = str(data.get("display_name", user_id)).strip()
        roles_raw = data.get("roles", ())
        if isinstance(roles_raw, (list, tuple)):
            roles = tuple(sorted({str(role).strip() for role in roles_raw if str(role).strip()}))
        elif isinstance(roles_raw, str):
            roles = tuple(sorted({part.strip() for part in roles_raw.split(",") if part.strip()}))
        else:
            roles = ()
        updated_at = str(data.get("updated_at") or _utcnow_iso())
        return cls(user_id=user_id, display_name=display_name, roles=roles, updated_at=updated_at)

    def to_dict(self) -> MutableMapping[str, object]:
        payload = asdict(self)
        payload["roles"] = list(self.roles)
        return payload


def load_profiles(path: str | Path) -> list[UserProfile]:
    storage = Path(path).expanduser()
    if not storage.exists():
        LOGGER.debug("Brak pliku profili użytkowników %s – zwracam pustą listę.", storage)
        return []
    try:
        content = storage.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - przekazywane do UI/logów
        raise RuntimeError(f"Nie udało się odczytać profili użytkowników z {storage}: {exc}") from exc
    if not content.strip():
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plik {storage} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(data, Iterable):
        raise ValueError(f"Struktura profili w {storage} musi być listą.")
    profiles: list[UserProfile] = []
    for entry in data:
        if not isinstance(entry, Mapping):
            LOGGER.warning("Pominięto wpis profilu o niepoprawnym typie: %r", entry)
            continue
        profiles.append(UserProfile.from_mapping(entry))
    return profiles


def save_profiles(profiles: Iterable[UserProfile], path: str | Path) -> None:
    storage = Path(path).expanduser()
    storage.parent.mkdir(parents=True, exist_ok=True)
    serialized = [profile.to_dict() for profile in profiles]
    payload = json.dumps(serialized, ensure_ascii=False, indent=2)
    storage.write_text(payload + "\n", encoding="utf-8")


def upsert_profile(
    profiles: list[UserProfile],
    *,
    user_id: str,
    display_name: str | None = None,
    roles: Iterable[str] | None = None,
) -> UserProfile:
    normalized_id = str(user_id).strip()
    if not normalized_id:
        raise ValueError("Identyfikator użytkownika nie może być pusty.")
    normalized_roles = tuple(sorted({str(role).strip() for role in roles or () if str(role).strip()}))
    now = _utcnow_iso()
    for index, profile in enumerate(profiles):
        if profile.user_id == normalized_id:
            updated = UserProfile(
                user_id=normalized_id,
                display_name=str(display_name or profile.display_name).strip() or profile.display_name,
                roles=normalized_roles or profile.roles,
                updated_at=now,
            )
            profiles[index] = updated
            return updated
    created = UserProfile(
        user_id=normalized_id,
        display_name=str(display_name or normalized_id).strip() or normalized_id,
        roles=normalized_roles,
        updated_at=now,
    )
    profiles.append(created)
    return created


def log_admin_event(message: str, *, log_path: str | Path) -> None:
    log_file = Path(log_path).expanduser()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": _utcnow_iso(),
        "message": str(message),
    }
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def remove_profile(profiles: list[UserProfile], *, user_id: str) -> UserProfile | None:
    """Removes a profile by identifier and returns the removed entry."""

    normalized_id = str(user_id).strip()
    if not normalized_id:
        raise ValueError("Identyfikator użytkownika nie może być pusty.")
    for index, profile in enumerate(profiles):
        if profile.user_id == normalized_id:
            removed = profiles.pop(index)
            return removed
    return None


__all__ = [
    "UserProfile",
    "load_profiles",
    "save_profiles",
    "upsert_profile",
    "log_admin_event",
    "remove_profile",
]

