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
import sys
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.fingerprint import HardwareFingerprintService, build_key_provider, decode_secret
from bot_core.security.license import validate_license
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


def _load_keys_from_file(path: Path) -> dict[str, bytes]:
    if not path.exists():
        raise FileNotFoundError(f"Plik kluczy fingerprint nie istnieje: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plik {path} zawiera niepoprawny JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Plik {path} ma nieoczekiwaną strukturę – oczekiwano obiektu JSON")
    raw_keys = payload.get("keys")
    if not isinstance(raw_keys, dict) or not raw_keys:
        raise ValueError(f"Plik {path} nie zawiera sekcji 'keys' z kluczami HMAC")
    decoded: dict[str, bytes] = {}
    for key_id, raw_value in raw_keys.items():
        if not isinstance(key_id, str):
            raise ValueError("Identyfikatory kluczy muszą być napisami")
        value = str(raw_value)
        decoded[key_id] = decode_secret(value)
    return decoded


def _read_license_summary(
    path: Path,
    *,
    fingerprint_path: Path | None = None,
    license_keys_path: Path | None = None,
    fingerprint_keys_path: Path | None = None,
    revocation_path: Path | None = None,
    revocation_keys_path: Path | None = None,
    revocation_signature_required: bool = False,
) -> dict[str, Any]:
    license_keys: Mapping[str, bytes] | None = None
    fingerprint_keys: Mapping[str, bytes] | None = None
    revocation_keys: Mapping[str, bytes] | None = None
    errors: list[str] = []
    warnings: list[str] = []

    if license_keys_path:
        try:
            license_keys = _load_keys_from_file(license_keys_path)
        except FileNotFoundError:
            errors.append(f"Brak pliku kluczy licencji: {license_keys_path}")
        except ValueError as exc:
            errors.append(
                f"Nie udało się wczytać kluczy licencji ({license_keys_path}): {exc}"
            )

    if fingerprint_keys_path:
        try:
            fingerprint_keys = _load_keys_from_file(fingerprint_keys_path)
        except FileNotFoundError:
            errors.append(f"Brak pliku kluczy fingerprintu: {fingerprint_keys_path}")
        except ValueError as exc:
            errors.append(
                f"Nie udało się wczytać kluczy fingerprintu ({fingerprint_keys_path}): {exc}"
            )

    if revocation_keys_path:
        try:
            revocation_keys = _load_keys_from_file(revocation_keys_path)
        except FileNotFoundError:
            errors.append(f"Brak pliku kluczy listy odwołań: {revocation_keys_path}")
        except ValueError as exc:
            errors.append(
                f"Nie udało się wczytać kluczy listy odwołań ({revocation_keys_path}): {exc}"
            )
    elif revocation_signature_required:
        errors.append(
            "Wymagano podpisanej listy odwołań, ale nie dostarczono pliku kluczy HMAC."
        )

    result = validate_license(
        license_path=path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        revocation_list_path=revocation_path,
        revocation_keys=revocation_keys,
        revocation_signature_required=revocation_signature_required,
    )
    status_map = {"ok": "active", "missing": "inactive"}
    status = status_map.get(result.status, "invalid" if result.errors else "unknown")
    summary = {
        "status": status,
        "fingerprint": result.fingerprint,
        "fingerprint_source": result.fingerprint_source,
        "valid_from": result.issued_at,
        "valid_to": result.expires_at,
        "profile": result.profile,
        "issuer": result.issuer,
        "schema": result.schema,
        "schema_version": result.schema_version,
        "license_id": result.license_id,
        "revocation_status": result.revocation_status,
        "revocation_checked": result.revocation_checked,
        "revocation_list_path": str(result.revocation_list_path) if result.revocation_list_path else None,
        "revocation_generated_at": result.revocation_generated_at,
        "revocation_reason": result.revocation_reason,
        "revocation_revoked_at": result.revocation_revoked_at,
        "path": str(path),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
    }
    if result.license_signature_key:
        summary["license_key_id"] = result.license_signature_key
    if result.fingerprint_signature_key:
        summary["fingerprint_key_id"] = result.fingerprint_signature_key
    if result.revocation_signature_key:
        summary["revocation_key_id"] = result.revocation_signature_key
    if warnings:
        summary["warnings"].extend(warnings)
    if errors:
        summary["errors"].extend(errors)
    return summary


def dump_state(
    *,
    license_path: str | None,
    profiles_path: str | None,
    fingerprint_path: str | None = None,
    license_keys_path: str | None = None,
    fingerprint_keys_path: str | None = None,
    revocation_path: str | None = None,
    revocation_keys_path: str | None = None,
    revocation_signature_required: bool = False,
) -> dict[str, Any]:
    license_file = _resolve_license_path(license_path)
    profiles_file = _resolve_profiles_path(profiles_path)
    fingerprint_file = Path(fingerprint_path).expanduser() if fingerprint_path else None
    license_keys_file = Path(license_keys_path).expanduser() if license_keys_path else None
    fingerprint_keys_file = (
        Path(fingerprint_keys_path).expanduser() if fingerprint_keys_path else None
    )
    revocation_file = Path(revocation_path).expanduser() if revocation_path else None
    revocation_keys_file = Path(revocation_keys_path).expanduser() if revocation_keys_path else None
    profiles = [profile.to_dict() for profile in load_profiles(profiles_file)]
    return {
        "license": _read_license_summary(
            license_file,
            fingerprint_path=fingerprint_file,
            license_keys_path=license_keys_file,
            fingerprint_keys_path=fingerprint_keys_file,
            revocation_path=revocation_file,
            revocation_keys_path=revocation_keys_file,
            revocation_signature_required=revocation_signature_required,
        ),
        "profiles": profiles,
    }


def generate_fingerprint(
    *,
    keys: Mapping[str, bytes],
    rotation_log: str | None,
    purpose: str | None,
    interval_days: float | None,
    dongle_hint: str | None,
) -> dict[str, Any]:
    if not keys:
        raise ValueError("Wymagany jest co najmniej jeden klucz podpisujący fingerprint")
    provider = build_key_provider(
        keys,
        rotation_log or "var/licenses/fingerprint_rotation.json",
        purpose=purpose or "hardware-fingerprint",
        interval_days=interval_days or 90.0,
    )
    service = HardwareFingerprintService(provider)
    record = service.build(dongle_serial=dongle_hint)
    return record.as_dict()


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

    dump_parser = subparsers.add_parser(
        "dump", help="Zwraca stan licencji i profili użytkowników"
    )
    dump_parser.add_argument("--license-path", dest="license_path", default=None)
    dump_parser.add_argument("--profiles-path", dest="profiles_path", default=None)
    dump_parser.add_argument("--fingerprint-path", dest="fingerprint_path", default=None)
    dump_parser.add_argument("--license-keys", dest="license_keys", default=None)
    dump_parser.add_argument("--fingerprint-keys", dest="fingerprint_keys", default=None)
    dump_parser.add_argument("--revocation-path", dest="revocation_path", default=None)
    dump_parser.add_argument("--revocation-keys", dest="revocation_keys", default=None)
    dump_parser.add_argument(
        "--require-signed-revocations",
        dest="revocation_signed",
        action="store_true",
        help="Wymaga podpisanej listy odwołań oraz dostarczonych kluczy HMAC.",
    )

    fingerprint_parser = subparsers.add_parser("fingerprint", help="Generuje podpisany fingerprint hosta")
    fingerprint_parser.add_argument("--keys-file", dest="keys_file", default=None)
    fingerprint_parser.add_argument(
        "--key",
        dest="keys",
        action="append",
        default=[],
        help="Klucz w formacie key_id=wartość (np. hex:abcd).",
    )
    fingerprint_parser.add_argument("--rotation-log", dest="rotation_log", default=None)
    fingerprint_parser.add_argument("--purpose", dest="purpose", default=None)
    fingerprint_parser.add_argument("--interval-days", dest="interval_days", type=float, default=None)
    fingerprint_parser.add_argument("--dongle", dest="dongle", default=None)

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
        state = dump_state(
            license_path=args.license_path,
            profiles_path=args.profiles_path,
            fingerprint_path=args.fingerprint_path,
            license_keys_path=args.license_keys,
            fingerprint_keys_path=args.fingerprint_keys,
            revocation_path=args.revocation_path,
            revocation_keys_path=args.revocation_keys,
            revocation_signature_required=args.revocation_signed,
        )
        print(json.dumps(state, ensure_ascii=False))
        return 0
    if args.command == "fingerprint":
        try:
            keys: dict[str, bytes] = {}
            if args.keys_file:
                keys.update(_load_keys_from_file(Path(args.keys_file)))
            for entry in args.keys:
                if "=" not in entry:
                    raise ValueError("Argument --key musi mieć format key_id=wartość")
                key_id, raw_value = entry.split("=", 1)
                keys[key_id.strip()] = decode_secret(raw_value.strip())
            record = generate_fingerprint(
                keys=keys,
                rotation_log=args.rotation_log,
                purpose=args.purpose,
                interval_days=args.interval_days,
                dongle_hint=args.dongle,
            )
        except Exception as exc:  # pragma: no cover - błędy raportujemy do stderr
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(record, ensure_ascii=False))
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

