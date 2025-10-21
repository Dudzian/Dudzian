"""Walidacja licencji OEM oraz podpisanych fingerprintów."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from bot_core.config.models import LicenseValidationConfig
from bot_core.security.capabilities import LicenseCapabilities
from bot_core.security.fingerprint import decode_secret
from bot_core.security.signing import build_hmac_signature

if TYPE_CHECKING:
    from bot_core.security.guards import CapabilityGuard

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LicenseValidationResult:
    """Zbiorcze informacje o stanie licencji po weryfikacji."""

    status: str
    fingerprint: str | None
    license_path: Path
    issued_at: str | None
    expires_at: str | None
    fingerprint_source: str | None
    profile: str | None
    issuer: str | None
    schema: str | None
    schema_version: str | None
    license_id: str | None
    revocation_list_path: Path | None
    revocation_status: str | None
    revocation_reason: str | None
    revocation_revoked_at: str | None
    revocation_generated_at: str | None
    revocation_checked: bool
    revocation_signature_key: str | None
    errors: list[str]
    warnings: list[str]
    payload: Mapping[str, Any] | None
    license_signature_key: str | None
    fingerprint_signature_key: str | None
    capabilities: "LicenseCapabilities | None" = None
    capability_guard: "CapabilityGuard | None" = None

    @property
    def is_valid(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_context(self) -> dict[str, Any]:
        """Buduje kontekst raportowany w alertach."""

        context = {
            "status": self.status,
            "license_path": str(self.license_path),
            "fingerprint": self.fingerprint,
        }
        if self.issued_at:
            context["issued_at"] = self.issued_at
        if self.expires_at:
            context["expires_at"] = self.expires_at
        if self.fingerprint_source:
            context["fingerprint_source"] = self.fingerprint_source
        if self.profile:
            context["profile"] = self.profile
        if self.issuer:
            context["issuer"] = self.issuer
        if self.schema:
            context["schema"] = self.schema
        if self.schema_version:
            context["schema_version"] = self.schema_version
        if self.license_id:
            context["license_id"] = self.license_id
        if self.revocation_list_path:
            context["revocation_list_path"] = str(self.revocation_list_path)
        if self.revocation_status:
            context["revocation_status"] = self.revocation_status
        if self.revocation_reason:
            context["revocation_reason"] = self.revocation_reason
        if self.revocation_revoked_at:
            context["revocation_revoked_at"] = self.revocation_revoked_at
        if self.revocation_generated_at:
            context["revocation_generated_at"] = self.revocation_generated_at
        context["revocation_checked"] = self.revocation_checked
        if self.revocation_signature_key:
            context["revocation_signature_key"] = self.revocation_signature_key
        if self.license_signature_key:
            context["license_key_id"] = self.license_signature_key
        if self.fingerprint_signature_key:
            context["fingerprint_key_id"] = self.fingerprint_signature_key
        if self.errors:
            context["errors"] = list(self.errors)
        if self.warnings:
            context["warnings"] = list(self.warnings)
        return context


class LicenseValidationError(RuntimeError):
    """Sygnalizuje błąd uniemożliwiający uruchomienie runtime bez licencji."""

    def __init__(self, message: str, *, result: LicenseValidationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Plik {path} zawiera niepoprawny JSON: {exc}") from exc


def _extract_fingerprint(candidate: Any) -> str | None:
    if isinstance(candidate, str):
        text = candidate.strip()
        return text or None
    if isinstance(candidate, Mapping):
        value = candidate.get("value")
        if isinstance(value, str):
            text = value.strip()
            return text or None
    return None


def _normalize_text(value: Any) -> str | None:
    if value in (None, ""):
        return None
    return str(value).strip() or None


def _coerce_license_identifier(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("license_id", "id", "value", "license"):
            candidate = _normalize_text(value.get(key))
            if candidate:
                return candidate
        return None
    return _normalize_text(value)


def _normalize_reason(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("reason", "cause", "message", "comment", "details"):
            candidate = _normalize_text(value.get(key))
            if candidate:
                return candidate
        return None
    return _normalize_text(value)


def _coerce_revocation_timestamp(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in (
            "revoked_at",
            "revokedAt",
            "revoked_on",
            "revokedOn",
            "timestamp",
            "time",
            "date",
            "revocation_time",
            "revocation_timestamp",
            "revocation_at",
        ):
            candidate = _normalize_text(value.get(key))
            if candidate:
                return candidate
        for key in ("meta", "metadata", "details", "info"):
            nested = value.get(key)
            if nested is not None:
                candidate = _coerce_revocation_timestamp(nested)
                if candidate:
                    return candidate
        candidate = _normalize_text(value.get("value"))
        if candidate:
            return candidate
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            candidate = _coerce_revocation_timestamp(item)
            if candidate:
                return candidate
    else:
        return _normalize_text(value)
    return None


def _coerce_revocation_entry(
    value: Any,
    *,
    key_hint: str | None = None,
) -> tuple[str | None, str | None, str | None]:
    license_id = _coerce_license_identifier(value)
    reason: str | None = None
    revoked_at: str | None = None

    if isinstance(value, Mapping):
        if not license_id:
            license_id = _coerce_license_identifier(value.get("license"))
        reason = (
            _normalize_reason(value)
            or _normalize_text(value.get("value"))
            or _normalize_text(value.get("reason"))
        )
        revoked_at = _coerce_revocation_timestamp(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        # Nested kolekcja – potraktuj jako listę wpisów
        aggregated_reason: str | None = None
        aggregated_id: str | None = None
        aggregated_revoked_at: str | None = None
        for item in value:
            item_id, item_reason, item_revoked_at = _coerce_revocation_entry(
                item, key_hint=key_hint
            )
            if item_id:
                aggregated_id = item_id
            if item_reason and not aggregated_reason:
                aggregated_reason = item_reason
            if item_revoked_at and not aggregated_revoked_at:
                aggregated_revoked_at = item_revoked_at
        return (
            aggregated_id or _normalize_text(key_hint),
            aggregated_reason,
            aggregated_revoked_at,
        )
    elif isinstance(value, str) and key_hint:
        # Wartość jako opis powodu, identyfikator pochodzi z klucza
        reason = _normalize_text(value)

    if not license_id:
        license_id = _normalize_text(key_hint)

    if not reason and isinstance(value, Mapping):
        reason = _normalize_reason(value)

    if revoked_at is None and isinstance(value, Mapping):
        revoked_at = _coerce_revocation_timestamp(value.get("revoked_at"))

    return license_id, reason, revoked_at


def _extract_revocation_entries(container: Any) -> dict[str, tuple[str | None, str | None]]:
    entries: dict[str, tuple[str | None, str | None]] = {}

    def _register(
        license_id: str | None, reason: str | None, revoked_at: str | None
    ) -> None:
        if not license_id:
            return
        existing_reason: str | None = None
        existing_revoked_at: str | None = None
        if license_id in entries:
            existing_reason, existing_revoked_at = entries[license_id]
        new_reason = existing_reason or reason
        new_revoked_at = existing_revoked_at or revoked_at
        entries[license_id] = (new_reason, new_revoked_at)

    if isinstance(container, Mapping):
        for key, value in container.items():
            key_hint = _normalize_text(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                for item in value:
                    _register(*_coerce_revocation_entry(item, key_hint=key_hint))
            else:
                _register(*_coerce_revocation_entry(value, key_hint=key_hint))
    elif isinstance(container, Sequence) and not isinstance(container, (str, bytes, bytearray)):
        for item in container:
            _register(*_coerce_revocation_entry(item))
    else:
        _register(*_coerce_revocation_entry(container))

    return entries


def _parse_revocation_document(
    document: Any,
    *,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, tuple[str | None, str | None]], str | None]:
    if not isinstance(document, (Mapping, Sequence)) or isinstance(document, (str, bytes, bytearray)):
        errors.append("Lista odwołań ma niepoprawny format JSON.")
        return {}, None

    generated_at: str | None = None
    revoked_entries: dict[str, tuple[str | None, str | None]] = {}

    if isinstance(document, Mapping):
        generated_at = _normalize_text(
            document.get("generated_at") or document.get("updated_at") or document.get("timestamp")
        )
        raw_revoked = document.get("revoked")
        if raw_revoked is None:
            raw_revoked = document.get("revocations")
        if raw_revoked is None:
            raw_revoked = document.get("licenses")
        if raw_revoked is None:
            errors.append("Lista odwołań nie zawiera sekcji 'revoked'.")
        else:
            revoked_entries = _extract_revocation_entries(raw_revoked)
    else:
        revoked_entries = _extract_revocation_entries(document)

    if not revoked_entries and not errors:
        warnings.append("Lista odwołań nie zawiera żadnych identyfikatorów licencji.")

    return revoked_entries, generated_at


def _parse_timestamp(
    value: str | None,
    *,
    label: str,
    errors: list[str],
    warnings: list[str],
) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        errors.append(f"Pole {label} nie jest poprawnym znacznikiem czasu ISO 8601: {text}.")
        return None
    if parsed.tzinfo is None:
        warnings.append(
            f"Pole {label} nie zawiera strefy czasowej – przyjmuję, że odnosi się do UTC."
        )
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def load_hmac_keys_file(path: str | Path) -> dict[str, bytes]:
    file_path = Path(path).expanduser()
    if not file_path.exists():
        raise FileNotFoundError(file_path)
    payload = _load_json(file_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Plik {file_path} powinien zawierać obiekt JSON z sekcją 'keys'.")
    raw_keys = payload.get("keys") if "keys" in payload else payload
    if not isinstance(raw_keys, Mapping):
        raise ValueError(f"Plik {file_path} nie zawiera mapy kluczy HMAC.")
    decoded: dict[str, bytes] = {}
    for key_id, raw_value in raw_keys.items():
        if not isinstance(key_id, str) or not key_id.strip():
            raise ValueError("Identyfikatory kluczy muszą być niepustymi napisami.")
        decoded[key_id.strip()] = decode_secret(str(raw_value))
    return decoded


def _verify_signature(
    payload: Mapping[str, Any] | Sequence[Any],
    signature: Mapping[str, Any],
    *,
    keys: Mapping[str, bytes],
    errors: list[str],
    warnings: list[str],
    label: str,
) -> str | None:
    if not keys:
        warnings.append(f"Pominięto weryfikację podpisu {label} – brak dostępnych kluczy.")
        return None

    key_id_val = signature.get("key_id")
    if not isinstance(key_id_val, str) or not key_id_val.strip():
        errors.append(f"Podpis {label} nie zawiera identyfikatora klucza.")
        return None
    key_id = key_id_val.strip()
    try:
        key_bytes = keys[key_id]
    except KeyError:
        errors.append(f"Brak klucza HMAC '{key_id}' do weryfikacji podpisu {label}.")
        return None

    algorithm = str(signature.get("algorithm") or "HMAC-SHA256")
    expected = build_hmac_signature(payload, key=key_bytes, algorithm=algorithm, key_id=key_id)
    actual = signature.get("value")
    if not isinstance(actual, str) or actual != expected.get("value"):
        errors.append(f"Sygnatura {label} niezgodna z oczekiwanym HMAC.")
        return None
    return key_id


def validate_license(
    *,
    license_path: str | Path,
    license_keys: Mapping[str, bytes] | None,
    fingerprint_path: str | Path | None = None,
    fingerprint_keys: Mapping[str, bytes] | None = None,
    current_time: datetime | None = None,
    allowed_profiles: Sequence[str] | None = None,
    allowed_issuers: Sequence[str] | None = None,
    max_validity_days: float | None = None,
    required_schema: str | None = "core.oem.license",
    allowed_schema_versions: Sequence[str] | None = ("1.0",),
    revocation_list_path: str | Path | None = None,
    revocation_required: bool = False,
    revocation_list_max_age_hours: float | None = None,
    revocation_keys: Mapping[str, bytes] | None = None,
    revocation_signature_required: bool = False,
) -> LicenseValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    license_file = Path(license_path).expanduser()
    revocation_file = Path(revocation_list_path).expanduser() if revocation_list_path else None
    revocation_status: str | None = None
    revocation_generated_at: str | None = None
    revocation_checked = False
    revocation_signature_key: str | None = None
    revocation_reason: str | None = None
    revocation_revoked_at: str | None = None
    now = current_time.astimezone(timezone.utc) if current_time else datetime.now(timezone.utc)
    allowed_profile_set = {
        str(value).strip().lower()
        for value in (allowed_profiles or ())
        if str(value).strip()
    }
    allowed_issuer_set = {
        str(value).strip().lower()
        for value in (allowed_issuers or ())
        if str(value).strip()
    }
    allowed_schema_versions_set = {
        str(value).strip()
        for value in (allowed_schema_versions or ())
        if str(value).strip()
    }

    if not license_file.exists():
        errors.append(f"Brak pliku licencji: {license_file}")
        return LicenseValidationResult(
            status="missing",
            fingerprint=None,
            license_path=license_file,
            issued_at=None,
            expires_at=None,
            fingerprint_source=None,
            profile=None,
            issuer=None,
            schema=None,
            schema_version=None,
            license_id=None,
            revocation_list_path=revocation_file,
            revocation_status=revocation_status,
            revocation_reason=revocation_reason,
            revocation_revoked_at=revocation_revoked_at,
            revocation_generated_at=revocation_generated_at,
            revocation_checked=revocation_checked,
            revocation_signature_key=revocation_signature_key,
            errors=errors,
            warnings=warnings,
            payload=None,
            license_signature_key=None,
            fingerprint_signature_key=None,
        )

    try:
        document = _load_json(license_file)
    except ValueError as exc:
        errors.append(str(exc))
        return LicenseValidationResult(
            status="invalid",
            fingerprint=None,
            license_path=license_file,
            issued_at=None,
            expires_at=None,
            fingerprint_source=None,
            profile=None,
            issuer=None,
            schema=None,
            schema_version=None,
            license_id=None,
            revocation_list_path=revocation_file,
            revocation_status=revocation_status,
            revocation_reason=revocation_reason,
            revocation_revoked_at=revocation_revoked_at,
            revocation_generated_at=revocation_generated_at,
            revocation_checked=revocation_checked,
            revocation_signature_key=revocation_signature_key,
            errors=errors,
            warnings=warnings,
            payload=None,
            license_signature_key=None,
            fingerprint_signature_key=None,
        )

    payload = document.get("payload") if isinstance(document, Mapping) else None
    signature = document.get("signature") if isinstance(document, Mapping) else None
    if not isinstance(payload, Mapping) or not isinstance(signature, Mapping):
        errors.append("Licencja nie zawiera struktur payload/signature.")
        fingerprint_value = _extract_fingerprint(document.get("fingerprint")) if isinstance(document, Mapping) else None
        issued_at = _normalize_text(document.get("issued_at")) if isinstance(document, Mapping) else None
        expires_at = _normalize_text(document.get("expires_at")) if isinstance(document, Mapping) else None
        if isinstance(document, Mapping):
            validity = document.get("valid")
            if isinstance(validity, Mapping):
                issued_at = issued_at or _normalize_text(validity.get("from"))
                expires_at = expires_at or _normalize_text(validity.get("to"))
        return LicenseValidationResult(
            status="invalid",
            fingerprint=fingerprint_value,
            license_path=license_file,
            issued_at=issued_at,
            expires_at=expires_at,
            fingerprint_source=None,
            profile=None,
            issuer=None,
            schema=None,
            schema_version=None,
            license_id=None,
            revocation_list_path=revocation_file,
            revocation_status=revocation_status,
            revocation_reason=revocation_reason,
            revocation_revoked_at=revocation_revoked_at,
            revocation_generated_at=revocation_generated_at,
            revocation_checked=revocation_checked,
            revocation_signature_key=revocation_signature_key,
            errors=errors,
            warnings=warnings,
            payload=payload if isinstance(payload, Mapping) else None,
            license_signature_key=None,
            fingerprint_signature_key=None,
        )

    issued_at = _normalize_text(payload.get("issued_at"))
    expires_at = _normalize_text(payload.get("expires_at"))
    profile = _normalize_text(payload.get("profile"))
    issuer = _normalize_text(payload.get("issuer"))
    schema = _normalize_text(payload.get("schema"))
    schema_version = _normalize_text(payload.get("schema_version"))
    license_id = _normalize_text(payload.get("license_id"))
    fingerprint_value = _extract_fingerprint(payload.get("fingerprint"))
    license_signature_key = None
    fingerprint_signature_key = None
    fingerprint_source: str | None = None

    if profile is None:
        errors.append("Licencja nie zawiera pola profile.")
    elif allowed_profile_set and profile.lower() not in allowed_profile_set:
        errors.append(
            f"Profil licencji '{profile}' nie znajduje się na liście dozwolonych: {sorted(allowed_profile_set)}."
        )

    if issuer is None:
        errors.append("Licencja nie zawiera pola issuer.")
    elif allowed_issuer_set and issuer.lower() not in allowed_issuer_set:
        errors.append(
            f"Wystawca licencji '{issuer}' nie jest dozwolony: {sorted(allowed_issuer_set)}."
        )

    if schema is None:
        errors.append("Licencja nie zawiera pola schema.")
    elif required_schema and schema.lower() != required_schema.strip().lower():
        errors.append(
            f"Licencja ma nieoczekiwany typ schematu '{schema}' – oczekiwano '{required_schema}'."
        )

    if schema_version is None:
        errors.append("Licencja nie zawiera pola schema_version.")
    elif allowed_schema_versions_set and schema_version not in allowed_schema_versions_set:
        errors.append(
            "Licencja ma nieobsługiwaną wersję schematu '%s'. Dozwolone: %s."
            % (schema_version, sorted(allowed_schema_versions_set))
        )

    if license_id is None:
        errors.append("Licencja nie zawiera pola license_id.")

    if issued_at is None:
        errors.append("Licencja nie zawiera pola issued_at.")
    if expires_at is None:
        errors.append("Licencja nie zawiera pola expires_at.")

    if license_keys is None:
        warnings.append("Pominięto weryfikację podpisu licencji – brak konfiguracji kluczy.")
    else:
        key_id = _verify_signature(
            payload,
            signature,
            keys=license_keys,
            errors=errors,
            warnings=warnings,
            label="licencji",
        )
        if key_id:
            license_signature_key = key_id

    fp_payload = payload.get("fingerprint_payload") if isinstance(payload.get("fingerprint_payload"), Mapping) else None
    fp_signature = payload.get("fingerprint_signature") if isinstance(payload.get("fingerprint_signature"), Mapping) else None
    if fp_payload is not None:
        fingerprint_source = "license_payload"
        fp_value = _extract_fingerprint(fp_payload.get("fingerprint"))
        if fp_value and fingerprint_value and fp_value != fingerprint_value:
            errors.append("Fingerprint w licencji nie jest zgodny z podpisanym payloadem.")
        fingerprint_value = fingerprint_value or fp_value
        if fingerprint_keys is None:
            warnings.append("Pominięto weryfikację podpisu fingerprintu – brak konfiguracji kluczy.")
        elif fp_signature is not None:
            key_id = _verify_signature(
                fp_payload,
                fp_signature,
                keys=fingerprint_keys,
                errors=errors,
                warnings=warnings,
                label="fingerprintu",
            )
            if key_id:
                fingerprint_signature_key = key_id
        else:
            errors.append("Licencja nie zawiera podpisu fingerprintu.")

    file_fingerprint_path = None
    if fingerprint_path:
        fingerprint_file = Path(fingerprint_path).expanduser()
        file_fingerprint_path = fingerprint_file
        if not fingerprint_file.exists():
            errors.append(f"Brak pliku fingerprintu: {fingerprint_file}")
        else:
            try:
                fp_document = _load_json(fingerprint_file)
            except ValueError as exc:
                errors.append(str(exc))
            else:
                fp_payload_doc = fp_document.get("payload") if isinstance(fp_document, Mapping) else None
                fp_signature_doc = fp_document.get("signature") if isinstance(fp_document, Mapping) else None
                if isinstance(fp_payload_doc, Mapping):
                    fp_value = _extract_fingerprint(fp_payload_doc.get("fingerprint"))
                    if fp_value and fingerprint_value and fp_value != fingerprint_value:
                        errors.append("Fingerprint w licencji nie zgadza się z plikiem fingerprintu.")
                    fingerprint_value = fingerprint_value or fp_value
                    fingerprint_source = str(fingerprint_file)
                    if fingerprint_keys is None:
                        warnings.append("Pominięto weryfikację podpisu fingerprintu z pliku – brak konfiguracji kluczy.")
                    elif isinstance(fp_signature_doc, Mapping):
                        key_id = _verify_signature(
                            fp_payload_doc,
                            fp_signature_doc,
                            keys=fingerprint_keys,
                            errors=errors,
                            warnings=warnings,
                            label="fingerprintu",
                        )
                        if key_id:
                            fingerprint_signature_key = key_id
                    else:
                        errors.append("Plik fingerprintu nie zawiera sekcji signature.")
                else:
                    errors.append("Plik fingerprintu ma nieoczekiwaną strukturę JSON.")

    revocation_keys = revocation_keys or {}

    if revocation_file is None:
        if revocation_required:
            errors.append(
                "Konfiguracja wymaga listy odwołań licencji, ale nie podano ścieżki."
            )
            revocation_status = "missing"
        else:
            revocation_status = "skipped"
    else:
        revocation_checked = True
        if not revocation_file.exists():
            message = f"Brak pliku listy odwołań: {revocation_file}"
            target = errors if revocation_required else warnings
            target.append(message)
            revocation_status = "missing"
        else:
            try:
                revocation_document = _load_json(revocation_file)
            except ValueError as exc:
                target = errors if revocation_required else warnings
                target.append(str(exc))
                revocation_status = "error"
            else:
                local_errors: list[str] = []
                local_warnings: list[str] = []
                payload_document = revocation_document
                signature_document: Mapping[str, Any] | None = None
                if isinstance(revocation_document, Mapping) and "payload" in revocation_document:
                    payload_candidate = revocation_document.get("payload")
                    signature_candidate = revocation_document.get("signature")
                    if isinstance(payload_candidate, (Mapping, Sequence)) and not isinstance(
                        payload_candidate, (str, bytes, bytearray)
                    ):
                        payload_document = payload_candidate
                        if signature_candidate is not None:
                            if isinstance(signature_candidate, Mapping):
                                signature_document = signature_candidate
                            else:
                                local_errors.append(
                                    "Podpis listy odwołań ma niepoprawny format."
                                )
                    else:
                        local_errors.append(
                            "Sekcja payload listy odwołań ma nieoczekiwaną strukturę."
                        )
                elif revocation_signature_required:
                    local_errors.append(
                        "Konfiguracja wymaga podpisanej listy odwołań, ale dokument nie zawiera sekcji 'payload' i 'signature'."
                    )

                payload_is_json = isinstance(payload_document, Mapping) or (
                    isinstance(payload_document, Sequence)
                    and not isinstance(payload_document, (str, bytes, bytearray))
                )

                if signature_document is not None and not payload_is_json:
                    local_errors.append(
                        "Sekcja payload listy odwołań ma nieobsługiwany format do weryfikacji podpisu."
                    )
                elif signature_document is not None:
                    if not revocation_keys:
                        message = (
                            "Pominięto weryfikację podpisu listy odwołań – brak dostępnych kluczy."
                        )
                        if revocation_signature_required:
                            local_errors.append(message)
                        else:
                            local_warnings.append(message)
                    else:
                        key_id = _verify_signature(
                            payload_document,
                            signature_document,
                            keys=revocation_keys,
                            errors=local_errors,
                            warnings=local_warnings,
                            label="listy odwołań",
                        )
                        if key_id:
                            revocation_signature_key = key_id
                        elif revocation_signature_required:
                            local_errors.append("Podpis listy odwołań nie został zweryfikowany.")
                elif revocation_signature_required:
                    local_errors.append(
                        "Konfiguracja wymaga podpisanej listy odwołań, ale dokument nie zawiera sekcji 'signature'."
                    )

                revocation_payload = payload_document
                revoked_entries, generated_at = _parse_revocation_document(
                    revocation_payload,
                    errors=local_errors,
                    warnings=local_warnings,
                )
                if local_errors:
                    if revocation_signature_required:
                        errors.extend(local_errors)
                    else:
                        target = errors if revocation_required else warnings
                        target.extend(local_errors)
                    if revocation_status is None:
                        revocation_status = "error"
                if local_warnings:
                    warnings.extend(local_warnings)

                generated_dt = None
                if generated_at:
                    timestamp_errors: list[str] = []
                    timestamp_warnings: list[str] = []
                    generated_dt = _parse_timestamp(
                        generated_at,
                        label="generated_at",
                        errors=timestamp_errors,
                        warnings=timestamp_warnings,
                    )
                    if timestamp_errors:
                        target = errors if revocation_required else warnings
                        target.extend(timestamp_errors)
                    if timestamp_warnings:
                        warnings.extend(timestamp_warnings)
                    if generated_dt:
                        revocation_generated_at = generated_dt.isoformat()
                    else:
                        revocation_generated_at = generated_at
                if generated_dt is None and generated_at:
                    revocation_generated_at = revocation_generated_at or generated_at

                if revocation_list_max_age_hours not in (None, float("inf")) and generated_at:
                    try:
                        max_age = timedelta(hours=float(revocation_list_max_age_hours))
                    except (TypeError, ValueError):
                        errors.append(
                            "Konfiguracja revocation_list_max_age_hours musi być liczbą."
                        )
                    else:
                        if max_age <= timedelta(0):
                            errors.append(
                                "Konfiguracja revocation_list_max_age_hours musi być dodatnia."
                            )
                        elif generated_dt and now - generated_dt > max_age:
                            message = (
                                "Lista odwołań licencji jest starsza niż dozwolone %.1f godzin."
                                % (max_age.total_seconds() / 3600.0)
                            )
                            target = errors if revocation_required else warnings
                            target.append(message)
                            if revocation_status != "revoked":
                                revocation_status = "stale"
                elif revocation_list_max_age_hours not in (None, float("inf")) and not generated_at:
                    target = errors if revocation_required else warnings
                    target.append(
                        "Lista odwołań nie zawiera znacznika generated_at – nie można ocenić świeżości."
                    )
                    if revocation_status is None:
                        revocation_status = "unknown"

                if revoked_entries:
                    if license_id and license_id in revoked_entries:
                        entry_reason, entry_revoked_at = revoked_entries[license_id]
                        revocation_reason = entry_reason
                        if entry_revoked_at:
                            revoked_at_errors: list[str] = []
                            revoked_at_warnings: list[str] = []
                            revoked_dt = _parse_timestamp(
                                entry_revoked_at,
                                label="revoked_at",
                                errors=revoked_at_errors,
                                warnings=revoked_at_warnings,
                            )
                            if revoked_at_errors:
                                warnings.extend(revoked_at_errors)
                            if revoked_at_warnings:
                                warnings.extend(revoked_at_warnings)
                            revocation_revoked_at = (
                                revoked_dt.isoformat() if revoked_dt else entry_revoked_at
                            )
                        errors.append(
                            f"Licencja {license_id} znajduje się na liście odwołań."
                        )
                        revocation_status = "revoked"
                    elif license_id:
                        revocation_status = "clear"
                    else:
                        revocation_status = "unknown"
                elif not local_errors and license_id:
                    if revocation_status not in {"stale", "missing", "error", "unknown"}:
                        revocation_status = "clear"

    issued_dt = _parse_timestamp(issued_at, label="issued_at", errors=errors, warnings=warnings)
    expires_dt = _parse_timestamp(expires_at, label="expires_at", errors=errors, warnings=warnings)

    if issued_dt and issued_dt - now > timedelta(minutes=5):
        warnings.append(
            f"Data wystawienia licencji ({issued_dt.isoformat()}) znajduje się w przyszłości."
        )

    if issued_dt and expires_dt and expires_dt <= issued_dt:
        errors.append(
            "Data wygaśnięcia licencji jest wcześniejsza lub równa dacie wystawienia."
        )

    if expires_dt:
        if expires_dt <= now:
            errors.append(f"Licencja wygasła {expires_dt.isoformat()}.")
        else:
            remaining = expires_dt - now
            if remaining <= timedelta(days=30):
                warnings.append(
                    "Licencja wygaśnie w ciągu 30 dni (data: "
                    f"{expires_dt.isoformat()})."
                )

    if issued_dt and expires_dt and max_validity_days not in (None, float("inf")):
        try:
            max_validity = timedelta(days=float(max_validity_days))
        except (TypeError, ValueError):
            errors.append("Konfiguracja max_validity_days musi być liczbą.")
        else:
            if max_validity <= timedelta(0):
                errors.append("Konfiguracja max_validity_days musi być dodatnia.")
            else:
                validity_window = expires_dt - issued_dt
                if validity_window > max_validity:
                    validity_days = validity_window.total_seconds() / 86400.0
                    errors.append(
                        "Okres ważności licencji (" "%.1f dni) przekracza dozwolone %.1f dni." % (
                            validity_days,
                            max_validity.total_seconds() / 86400.0,
                        )
                    )

    status = "ok" if not errors else "invalid"
    return LicenseValidationResult(
        status=status,
        fingerprint=fingerprint_value,
        license_path=license_file,
        issued_at=issued_at,
        expires_at=expires_at,
        fingerprint_source=fingerprint_source or (str(file_fingerprint_path) if file_fingerprint_path else None),
        profile=profile,
        issuer=issuer,
        schema=schema,
        schema_version=schema_version,
        license_id=license_id,
        revocation_list_path=revocation_file,
        revocation_status=revocation_status,
        revocation_reason=revocation_reason,
        revocation_revoked_at=revocation_revoked_at,
        revocation_generated_at=revocation_generated_at,
        revocation_checked=revocation_checked,
        revocation_signature_key=revocation_signature_key,
        errors=errors,
        warnings=warnings,
        payload=payload,
        license_signature_key=license_signature_key,
        fingerprint_signature_key=fingerprint_signature_key,
    )


def validate_license_from_config(config: LicenseValidationConfig) -> LicenseValidationResult:
    """Waliduje licencję na podstawie konfiguracji Core."""

    if not config.license_keys_path:
        raise LicenseValidationError("Konfiguracja license.license_keys_path jest wymagana.")
    if not config.fingerprint_keys_path:
        raise LicenseValidationError("Konfiguracja license.fingerprint_keys_path jest wymagana.")

    try:
        license_keys = load_hmac_keys_file(config.license_keys_path)
    except FileNotFoundError as exc:
        raise LicenseValidationError(
            f"Brak pliku kluczy licencyjnych: {config.license_keys_path}",
        ) from exc
    except ValueError as exc:
        raise LicenseValidationError(
            f"Niepoprawny plik kluczy licencyjnych ({config.license_keys_path}): {exc}",
        ) from exc

    try:
        fingerprint_keys = load_hmac_keys_file(config.fingerprint_keys_path)
    except FileNotFoundError as exc:
        raise LicenseValidationError(
            f"Brak pliku kluczy fingerprintu: {config.fingerprint_keys_path}",
        ) from exc
    except ValueError as exc:
        raise LicenseValidationError(
            f"Niepoprawny plik kluczy fingerprintu ({config.fingerprint_keys_path}): {exc}",
        ) from exc

    revocation_keys: Mapping[str, bytes] | None = None
    if config.revocation_keys_path:
        try:
            revocation_keys = load_hmac_keys_file(config.revocation_keys_path)
        except FileNotFoundError as exc:
            raise LicenseValidationError(
                f"Brak pliku kluczy listy odwołań: {config.revocation_keys_path}",
            ) from exc
        except ValueError as exc:
            raise LicenseValidationError(
                f"Niepoprawny plik kluczy listy odwołań ({config.revocation_keys_path}): {exc}",
            ) from exc
    elif config.revocation_signature_required:
        raise LicenseValidationError(
            "Konfiguracja license.revocation_keys_path jest wymagana do weryfikacji podpisu listy odwołań.",
        )

    result = validate_license(
        license_path=config.license_path,
        license_keys=license_keys,
        fingerprint_path=config.fingerprint_path,
        fingerprint_keys=fingerprint_keys,
        allowed_profiles=config.allowed_profiles,
        allowed_issuers=config.allowed_issuers,
        max_validity_days=config.max_validity_days,
        required_schema=config.required_schema,
        allowed_schema_versions=config.allowed_schema_versions,
        revocation_list_path=config.revocation_list_path,
        revocation_required=config.revocation_required,
        revocation_list_max_age_hours=config.revocation_list_max_age_hours,
        revocation_keys=revocation_keys,
        revocation_signature_required=config.revocation_signature_required,
    )
    if not result.is_valid:
        raise LicenseValidationError("Licencja nie przeszła weryfikacji podpisów.", result=result)
    return result


__all__ = [
    "LicenseValidationResult",
    "LicenseValidationError",
    "validate_license",
    "validate_license_from_config",
    "load_hmac_keys_file",
]
