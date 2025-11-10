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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from bot_core.security.fingerprint import (
    FingerprintError,
    HardwareFingerprintService,
    LICENSE_SECRET_PATH,
    build_key_provider,
    decode_secret,
    get_local_fingerprint,
    load_license_secret,
    _normalize_binding_fingerprint,
)
from bot_core.security.logs import export_security_bundle, export_signed_audit_log, read_audit_entries
from bot_core.security.license import validate_license
from bot_core.security.license_service import (
    LicenseBundleError,
    LicenseService,
    LicenseServiceError,
    LicenseSignatureError,
    LicenseSnapshot,
)
from bot_core.security.profiles import (
    load_profiles,
    log_admin_event,
    remove_profile,
    save_profiles,
    upsert_profile,
)
from bot_core.security.tpm import TpmValidationError, validate_attestation

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


def ensure_binding_secret(secret_path: str | None = None, expected_fingerprint: str | None = None) -> dict[str, Any]:
    normalized_expected = (
        _normalize_binding_fingerprint(expected_fingerprint)
        if expected_fingerprint is not None and expected_fingerprint.strip()
        else None
    )
    if normalized_expected:
        try:
            local = _normalize_binding_fingerprint(get_local_fingerprint())
        except FingerprintError as exc:
            return {
                "status": "error",
                "error": str(exc),
            }
        except Exception as exc:  # pragma: no cover - zabezpieczenie defensywne
            return {
                "status": "error",
                "error": str(exc),
            }
        if local != normalized_expected:
            return {
                "status": "error",
                "error": "Fingerprint urządzenia nie zgadza się z oczekiwaną licencją.",
                "fingerprint": local,
                "expected": normalized_expected,
            }

    destination = Path(secret_path).expanduser() if secret_path else LICENSE_SECRET_PATH
    try:
        secret = load_license_secret(destination)
    except FingerprintError as exc:
        return {
            "status": "error",
            "error": str(exc),
        }

    return {
        "status": "ok",
        "secret_length": len(secret),
        "path": str(destination),
    }


def _empty_license_summary(path: Path) -> dict[str, Any]:
    return {
        "status": "inactive",
        "fingerprint": None,
        "local_fingerprint": None,
        "fingerprint_source": None,
        "valid_from": None,
        "valid_to": None,
        "profile": None,
        "issuer": None,
        "schema": None,
        "schema_version": None,
        "license_id": None,
        "revocation_status": None,
        "revocation_checked": False,
        "revocation_list_path": None,
        "revocation_generated_at": None,
        "revocation_reason": None,
        "revocation_revoked_at": None,
        "path": str(path),
        "warnings": [],
        "errors": [],
        "edition": None,
        "environments": [],
        "modules": [],
        "runtime": [],
        "exchanges": {},
        "strategies": {},
        "limits": {},
        "maintenance_until": None,
        "maintenance_active": False,
        "trial_active": False,
        "trial_expires_at": None,
        "holder": {},
        "metadata": {},
        "seats": None,
        "effective_date": None,
    }


def _build_license_policy(capabilities: Any) -> dict[str, Any]:
    if isinstance(capabilities, LicenseSnapshot):  # pragma: no cover - kompatybilność
        capabilities = capabilities.capabilities
    today = datetime.now(timezone.utc).date()
    maintenance_until = getattr(capabilities, "maintenance_until", None)
    trial_info = getattr(capabilities, "trial", None)
    trial_until = None
    trial_active = False
    if trial_info is not None:
        trial_until = getattr(trial_info, "expires_at", None)
        trial_active = bool(getattr(trial_info, "enabled", False)) and (
            trial_until is None or today <= trial_until
        )
    expires_on = maintenance_until or trial_until
    days_remaining: int | None = None
    state = "active"
    if expires_on is not None:
        days_remaining = (expires_on - today).days
        if days_remaining < 0:
            state = "expired"
        elif days_remaining <= 7:
            state = "critical"
        elif days_remaining <= 30:
            state = "warning"
    elif trial_active:
        state = "trial"

    return {
        "state": state,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "maintenance_until": maintenance_until.isoformat() if maintenance_until else None,
        "trial_expires_at": trial_until.isoformat() if trial_until else None,
        "days_remaining": days_remaining,
        "trial_active": trial_active,
    }


def _build_capability_summary(snapshot: LicenseSnapshot) -> dict[str, Any]:
    capabilities = snapshot.capabilities
    modules = sorted(name for name, enabled in capabilities.modules.items() if enabled)
    runtime = sorted(name for name, enabled in capabilities.runtime.items() if enabled)
    strategies = {name: bool(enabled) for name, enabled in capabilities.strategies.items()}
    exchanges = {name: bool(enabled) for name, enabled in capabilities.exchanges.items()}
    environments = sorted(capabilities.environments)
    summary = _empty_license_summary(snapshot.bundle_path)
    fingerprint_value = capabilities.hwid or snapshot.local_hwid
    fingerprint_source = "payload" if capabilities.hwid else ("local" if snapshot.local_hwid else None)
    summary.update(
        {
            "status": "active",
            "fingerprint": fingerprint_value,
            "local_fingerprint": snapshot.local_hwid,
            "fingerprint_source": fingerprint_source,
            "valid_from": capabilities.issued_at.isoformat() if capabilities.issued_at else None,
            "valid_to": capabilities.maintenance_until.isoformat()
            if capabilities.maintenance_until
            else None,
            "profile": capabilities.edition,
            "issuer": capabilities.raw_payload.get("issuer"),
            "schema": capabilities.raw_payload.get("schema"),
            "schema_version": capabilities.raw_payload.get("schema_version"),
            "license_id": capabilities.license_id,
            "revocation_status": "skipped",
            "revocation_checked": False,
            "edition": capabilities.edition,
            "environments": environments,
            "modules": modules,
            "runtime": runtime,
            "strategies": strategies,
            "exchanges": exchanges,
            "limits": {
                "max_paper_controllers": capabilities.limits.max_paper_controllers,
                "max_live_controllers": capabilities.limits.max_live_controllers,
                "max_concurrent_bots": capabilities.limits.max_concurrent_bots,
                "max_alert_channels": capabilities.limits.max_alert_channels,
            },
            "maintenance_until": capabilities.maintenance_until.isoformat()
            if capabilities.maintenance_until
            else None,
            "maintenance_active": capabilities.is_maintenance_active(),
            "trial_active": capabilities.is_trial_active(),
            "trial_expires_at": capabilities.trial.expires_at.isoformat()
            if capabilities.trial.expires_at
            else None,
            "holder": dict(capabilities.holder),
            "metadata": dict(capabilities.metadata),
            "seats": capabilities.seats,
            "effective_date": capabilities.effective_date.isoformat(),
        }
    )
    policy = _build_license_policy(capabilities)
    summary["policy"] = policy
    if policy["state"] == "expired":
        summary["errors"].append("Licencja utrzymaniowa wygasła i wymaga odnowienia.")
    elif policy["state"] in {"critical", "warning"}:
        summary["warnings"].append("Licencja wkrótce wygaśnie – zaplanuj odnowienie.")
    return summary


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
    errors: list[str] = []
    warnings: list[str] = []

    try:
        service = LicenseService()
        snapshot = service.load_from_file(path)
    except FileNotFoundError:
        summary = _empty_license_summary(path)
        summary["status"] = "inactive"
        summary["errors"].append(f"Brak pliku licencji: {path}")
        return summary
    except LicenseSignatureError as exc:
        summary = _empty_license_summary(path)
        summary["status"] = "invalid"
        summary["errors"].append(str(exc))
        return summary
    except LicenseBundleError as exc:
        summary = _empty_license_summary(path)
        summary["status"] = "invalid"
        summary["errors"].append(str(exc))
        if warnings:
            summary["warnings"].extend(warnings)
        return summary
    except LicenseServiceError as exc:
        warnings.append(str(exc))
    else:
        summary = _build_capability_summary(snapshot)
        if warnings:
            summary["warnings"].extend(warnings)
        return summary

    license_keys: Mapping[str, bytes] | None = None
    fingerprint_keys: Mapping[str, bytes] | None = None
    revocation_keys: Mapping[str, bytes] | None = None

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
    warning_details = [issue.to_dict() for issue in result.warnings]
    error_details = [issue.to_dict() for issue in result.errors]
    summary = {
        "status": status,
        "fingerprint": result.fingerprint,
        "fingerprint_source": result.fingerprint_source,
        "local_fingerprint": None,
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
        "warnings": [issue.message for issue in result.warnings],
        "errors": [issue.message for issue in result.errors],
        "warning_details": warning_details,
        "error_details": error_details,
        "edition": result.profile,
        "environments": [],
        "modules": [],
        "runtime": [],
        "exchanges": {},
        "strategies": {},
        "limits": {},
        "maintenance_until": None,
        "maintenance_active": False,
        "trial_active": False,
        "trial_expires_at": None,
        "holder": {},
        "metadata": {},
        "seats": None,
        "effective_date": None,
    }
    if result.license_signature_key:
        summary["license_key_id"] = result.license_signature_key
    if result.fingerprint_signature_key:
        summary["fingerprint_key_id"] = result.fingerprint_signature_key
    if result.revocation_signature_key:
        summary["revocation_key_id"] = result.revocation_signature_key
    if warnings:
        summary.setdefault("warnings", []).extend(warnings)
    if errors:
        summary.setdefault("errors", []).extend(errors)
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
    audit_path: str | None = None,
    audit_limit: int = 200,
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
    audit_entries = read_audit_entries(audit_path, limit=audit_limit)
    audit_payload = {
        "path": str(Path(audit_path).expanduser()) if audit_path else None,
        "entries": audit_entries,
        "count": len(audit_entries),
    }

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
        "audit": audit_payload,
    }


def validate_oem_bundle(
    *,
    license_path: str,
    fallback_path: str | None,
    fingerprint_path: str | None = None,
    license_keys_path: str | None = None,
    fingerprint_keys_path: str | None = None,
    revocation_path: str | None = None,
    revocation_keys_path: str | None = None,
    revocation_signature_required: bool = False,
) -> dict[str, Any]:
    primary_file = _resolve_license_path(license_path)
    fallback_file = Path(fallback_path).expanduser() if fallback_path else None
    fingerprint_file = Path(fingerprint_path).expanduser() if fingerprint_path else None
    license_keys_file = Path(license_keys_path).expanduser() if license_keys_path else None
    fingerprint_keys_file = (
        Path(fingerprint_keys_path).expanduser() if fingerprint_keys_path else None
    )
    revocation_file = Path(revocation_path).expanduser() if revocation_path else None
    revocation_keys_file = Path(revocation_keys_path).expanduser() if revocation_keys_path else None

    primary_summary = _read_license_summary(
        primary_file,
        fingerprint_path=fingerprint_file,
        license_keys_path=license_keys_file,
        fingerprint_keys_path=fingerprint_keys_file,
        revocation_path=revocation_file,
        revocation_keys_path=revocation_keys_file,
        revocation_signature_required=revocation_signature_required,
    )

    fallback_summary: dict[str, Any] | None = None
    if fallback_file:
        fallback_summary = _read_license_summary(
            fallback_file,
            fingerprint_path=fingerprint_file,
            license_keys_path=license_keys_file,
            fingerprint_keys_path=fingerprint_keys_file,
            revocation_path=revocation_file,
            revocation_keys_path=revocation_keys_file,
            revocation_signature_required=revocation_signature_required,
        )

    use_fallback = False
    effective = primary_summary
    if primary_summary.get("status") != "active" and fallback_summary:
        if fallback_summary.get("status") == "active":
            effective = fallback_summary
            use_fallback = True

    return {
        "primary": primary_summary,
        "fallback": fallback_summary,
        "using_fallback": use_fallback,
        "effective_source": "fallback" if use_fallback else "primary",
        "effective": effective,
    }


def verify_tpm_evidence(
    *,
    evidence_path: str,
    expected_fingerprint: str | None = None,
    license_path: str | None = None,
    keyring_path: str | None = None,
) -> dict[str, Any]:
    fingerprint_hint = expected_fingerprint
    if fingerprint_hint is None and license_path:
        summary = _read_license_summary(Path(license_path).expanduser())
        fingerprint_hint = summary.get("fingerprint") or summary.get("local_fingerprint")
    try:
        result = validate_attestation(
            evidence_path=evidence_path,
            expected_fingerprint=fingerprint_hint,
            keyring=keyring_path,
        )
    except FileNotFoundError:
        return {
            "status": "missing",
            "errors": [f"Brak pliku dowodu TPM: {evidence_path}"],
            "warnings": [],
            "expected_fingerprint": fingerprint_hint,
        }
    except TpmValidationError as exc:
        return {
            "status": "invalid",
            "errors": [str(exc)],
            "warnings": [],
            "expected_fingerprint": fingerprint_hint,
        }

    payload = result.to_dict()
    payload["expected_fingerprint"] = fingerprint_hint
    payload["evidence_path"] = str(Path(evidence_path).expanduser())
    return payload


def export_audit_bundle(
    *,
    log_path: str | None,
    output_dir: str | None,
    limit: int | None,
    key_source: str | None = None,
    key_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    result = export_signed_audit_log(
        log_path=log_path,
        destination_dir=output_dir,
        limit=limit,
        key_source=key_source,
        key_id=key_id,
        metadata=metadata or {},
    )
    payload = result.to_dict()
    payload["status"] = "ok"
    return payload


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
    dump_parser.add_argument("--audit-path", dest="audit_path", default=None)
    dump_parser.add_argument("--audit-limit", dest="audit_limit", type=int, default=200)

    oem_parser = subparsers.add_parser(
        "oem-validate", help="Waliduje licencję OEM oraz fallback")
    oem_parser.add_argument("--license-path", dest="license_path", required=True)
    oem_parser.add_argument("--fallback-path", dest="fallback_path", default=None)
    oem_parser.add_argument("--fingerprint-path", dest="fingerprint_path", default=None)
    oem_parser.add_argument("--license-keys", dest="license_keys", default=None)
    oem_parser.add_argument("--fingerprint-keys", dest="fingerprint_keys", default=None)
    oem_parser.add_argument("--revocation-path", dest="revocation_path", default=None)
    oem_parser.add_argument("--revocation-keys", dest="revocation_keys", default=None)
    oem_parser.add_argument(
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

    verify_tpm_parser = subparsers.add_parser("verify-tpm", help="Waliduje dowód TPM/secure enclave")
    verify_tpm_parser.add_argument("--evidence-path", dest="evidence_path", required=True)
    verify_tpm_parser.add_argument("--expected-fingerprint", dest="expected_fingerprint", default=None)
    verify_tpm_parser.add_argument("--license-path", dest="license_path", default=None)
    verify_tpm_parser.add_argument("--keyring", dest="keyring", default=None)

    ensure_secret_parser = subparsers.add_parser(
        "ensure-binding-secret", help="Zapewnia wygenerowanie i zabezpieczenie lokalnego sekretu licencyjnego"
    )
    ensure_secret_parser.add_argument("--secret-path", dest="secret_path", default=None)
    ensure_secret_parser.add_argument("--fingerprint", dest="fingerprint", default=None)

    bundle_parser = subparsers.add_parser(
        "export-security-bundle", help="Eksportuje podpisany pakiet logów bezpieczeństwa oraz alertów"
    )
    bundle_parser.add_argument("--audit-path", dest="audit_path", required=True)
    bundle_parser.add_argument("--alerts-path", dest="alerts_path", required=True)
    bundle_parser.add_argument("--output-dir", dest="output_dir", required=True)
    bundle_parser.add_argument(
        "--include-log",
        dest="include_logs",
        action="append",
        default=[],
        help="Dodatkowe pliki logów do dołączenia (można powtórzyć).",
    )
    bundle_parser.add_argument(
        "--metadata",
        dest="metadata",
        default=None,
        help="Opcjonalne metadane w formacie JSON do zapisania w pakiecie.",
    )

    audit_parser = subparsers.add_parser("export-audit", help="Eksportuje podpisany pakiet logów bezpieczeństwa")
    audit_parser.add_argument("--log-path", dest="log_path", default=None)
    audit_parser.add_argument("--output-dir", dest="output_dir", default=None)
    audit_parser.add_argument("--limit", dest="limit", type=int, default=None)
    audit_parser.add_argument("--key", dest="key_source", default=None)
    audit_parser.add_argument("--key-id", dest="key_id", default=None)
    audit_parser.add_argument("--metadata", dest="metadata", default=None, help="Dodatkowe metadane w formacie JSON")

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
            audit_path=args.audit_path,
            audit_limit=args.audit_limit,
        )
        print(json.dumps(state, ensure_ascii=False))
        return 0
    if args.command == "oem-validate":
        summary = validate_oem_bundle(
            license_path=args.license_path,
            fallback_path=args.fallback_path,
            fingerprint_path=args.fingerprint_path,
            license_keys_path=args.license_keys,
            fingerprint_keys_path=args.fingerprint_keys,
            revocation_path=args.revocation_path,
            revocation_keys_path=args.revocation_keys,
            revocation_signature_required=args.revocation_signed,
        )
        print(json.dumps(summary, ensure_ascii=False))
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
    if args.command == "verify-tpm":
        result = verify_tpm_evidence(
            evidence_path=args.evidence_path,
            expected_fingerprint=args.expected_fingerprint,
            license_path=args.license_path,
            keyring_path=args.keyring,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "ensure-binding-secret":
        result = ensure_binding_secret(
            secret_path=args.secret_path,
            expected_fingerprint=args.fingerprint,
        )
        print(json.dumps(result, ensure_ascii=False))
        return 0
    if args.command == "export-security-bundle":
        metadata: Mapping[str, Any] | None = None
        if args.metadata:
            try:
                parsed = json.loads(args.metadata)
            except json.JSONDecodeError as exc:
                print(str(exc), file=sys.stderr)
                return 1
            if not isinstance(parsed, Mapping):
                print("Metadane muszą być obiektem JSON.", file=sys.stderr)
                return 1
            metadata = parsed
        result = export_security_bundle(
            audit_path=args.audit_path,
            alerts_path=args.alerts_path,
            destination_dir=args.output_dir,
            include_logs=args.include_logs,
            metadata=metadata,
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False))
        return 0
    if args.command == "export-audit":
        metadata: Mapping[str, Any] | None = None
        if args.metadata:
            try:
                metadata = json.loads(args.metadata)
            except json.JSONDecodeError as exc:
                print(f"Niepoprawne metadane JSON: {exc}", file=sys.stderr)
                return 2
        try:
            result = export_audit_bundle(
                log_path=args.log_path,
                output_dir=args.output_dir,
                limit=args.limit,
                key_source=args.key_source,
                key_id=args.key_id,
                metadata=metadata,
            )
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        print(json.dumps(result, ensure_ascii=False))
        return 0
    raise ValueError(f"Nieobsługiwane polecenie: {args.command}")


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())

