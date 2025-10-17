"""Small CLI bridge exposing security artefacts for the Qt UI layer.

The desktop shell is written in C++/Qt.  Rather than embedding a Python
interpreter inside the client we expose a thin command line interface that
returns JSON payloads.  The bridge reuses the security helpers implemented in
``bot_core.security`` so that UI actions stay aligned with the runtime
configuration and RBAC expectations.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from bot_core.security.profiles import (
    load_profiles,
    log_admin_event,
    remove_profile,
    save_profiles,
    upsert_profile,
)

LOGGER = logging.getLogger(__name__)


def _resolve_license_path(path: str | None) -> Path:
    if path:
        return Path(path).expanduser()
    return Path("var/licenses/active/license.json")


def _resolve_profiles_path(path: str | None) -> Path:
    if path:
        return Path(path).expanduser()
    return Path("config/user_profiles.json")


def _read_license_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "status": "inactive",
            "fingerprint": None,
            "valid_from": None,
            "valid_to": None,
            "path": str(path),
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plik licencji {path} zawiera niepoprawny JSON: {exc}") from exc
    except OSError as exc:  # pragma: no cover - propagujemy do UI/logów
        raise RuntimeError(f"Nie udało się odczytać pliku licencji {path}: {exc}") from exc

    fingerprint = None
    validity_from = None
    validity_to = None
    if isinstance(payload, dict):
        fingerprint = payload.get("fingerprint") or payload.get("license_fingerprint")
        validity = payload.get("valid") or payload.get("validity")
        if isinstance(validity, dict):
            validity_from = validity.get("from") or validity.get("start")
            validity_to = validity.get("to") or validity.get("end")
        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            fingerprint = fingerprint or metadata.get("fingerprint")

    status = "active" if fingerprint else "unknown"
    return {
        "status": status,
        "fingerprint": fingerprint,
        "valid_from": validity_from,
        "valid_to": validity_to,
        "path": str(path),
    }


def dump_state(*, license_path: str | None, profiles_path: str | None) -> dict[str, Any]:
    license_file = _resolve_license_path(license_path)
    profiles_file = _resolve_profiles_path(profiles_path)
    profiles = [profile.to_dict() for profile in load_profiles(profiles_file)]
    return {
        "license": _read_license_summary(license_file),
        "profiles": profiles,
    }


def assign_profile(
    *,
    profiles_path: str | None,
    user_id: str,
    display_name: str | None,
    roles: list[str],
    log_path: str | None,
    actor: str | None,
) -> dict[str, Any]:
    storage = _resolve_profiles_path(profiles_path)
    profiles = load_profiles(storage)
    updated = upsert_profile(
        profiles,
        user_id=user_id,
        display_name=display_name,
        roles=roles,
    )
    save_profiles(profiles, storage)
    actor_label = actor or "ui"
    message = f"{actor_label} updated profile {updated.user_id} -> roles={list(updated.roles)}"
    log_destination = Path(log_path).expanduser() if log_path else Path("logs/security_admin.log")
    log_admin_event(message, log_path=log_destination)
    LOGGER.info(message)
    return {
        "status": "ok",
        "profile": updated.to_dict(),
        "log_path": str(log_destination),
    }


def remove_profile_entry(
    *,
    profiles_path: str | None,
    user_id: str,
    log_path: str | None,
    actor: str | None,
) -> dict[str, Any]:
    storage = _resolve_profiles_path(profiles_path)
    profiles = load_profiles(storage)
    removed = remove_profile(profiles, user_id=user_id)
    if removed is None:
        return {
            "status": "not_found",
            "user_id": user_id,
        }
    save_profiles(profiles, storage)
    actor_label = actor or "ui"
    message = f"{actor_label} removed profile {removed.user_id}"
    log_destination = Path(log_path).expanduser() if log_path else Path("logs/security_admin.log")
    log_admin_event(message, log_path=log_destination)
    LOGGER.info(message)
    return {
        "status": "ok",
        "removed": removed.to_dict(),
        "log_path": str(log_destination),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UI security bridge")
    subparsers = parser.add_subparsers(dest="command", required=True)

    dump_parser = subparsers.add_parser("dump", help="Zwraca stan licencji i profili użytkowników")
    dump_parser.add_argument("--license-path", dest="license_path", default=None)
    dump_parser.add_argument("--profiles-path", dest="profiles_path", default=None)

    assign_parser = subparsers.add_parser("assign-profile", help="Aktualizuje profil użytkownika")
    assign_parser.add_argument("--profiles-path", dest="profiles_path", default=None)
    assign_parser.add_argument("--user", dest="user_id", required=True)
    assign_parser.add_argument("--display-name", dest="display_name", default=None)
    assign_parser.add_argument("--role", dest="roles", action="append", default=[])
    assign_parser.add_argument("--log-path", dest="log_path", default=None)
    assign_parser.add_argument("--actor", dest="actor", default=None)

    remove_parser = subparsers.add_parser("remove-profile", help="Usuwa profil użytkownika")
    remove_parser.add_argument("--profiles-path", dest="profiles_path", default=None)
    remove_parser.add_argument("--user", dest="user_id", required=True)
    remove_parser.add_argument("--log-path", dest="log_path", default=None)
    remove_parser.add_argument("--actor", dest="actor", default=None)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "dump":
        state = dump_state(license_path=args.license_path, profiles_path=args.profiles_path)
        print(json.dumps(state, ensure_ascii=False))
        return 0
    if args.command == "assign-profile":
        result = assign_profile(
            profiles_path=args.profiles_path,
            user_id=args.user_id,
            display_name=args.display_name,
            roles=args.roles,
            log_path=args.log_path,
            actor=args.actor,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "remove-profile":
        result = remove_profile_entry(
            profiles_path=args.profiles_path,
            user_id=args.user_id,
            log_path=args.log_path,
            actor=args.actor,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    raise ValueError(f"Nieobsługiwane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

