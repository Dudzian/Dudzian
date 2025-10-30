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
from bot_core.security.messages import ValidationMessage, make_error, make_warning
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
    errors: list[ValidationMessage]
    warnings: list[ValidationMessage]
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
            context["errors"] = [entry.to_dict() for entry in self.errors]
            context["error_messages"] = [entry.message for entry in self.errors]
        if self.warnings:
            context["warnings"] = [entry.to_dict() for entry in self.warnings]
            context["warning_messages"] = [entry.message for entry in self.warnings]
        return context


class LicenseValidationError(RuntimeError):
    """Sygnalizuje błąd uniemożliwiający uruchomienie runtime bez licencji."""

    def __init__(self, message: str, *, result: LicenseValidationResult | None = None) -> None:
        super().__init__(message)
        self.result = result


def _append_error(
    bucket: list[ValidationMessage], code: str, message: str, *, hint: str | None = None
) -> None:
    bucket.append(make_error(code, message, hint=hint))


def _append_warning(
    bucket: list[ValidationMessage], code: str, message: str, *, hint: str | None = None
) -> None:
    bucket.append(make_warning(code, message, hint=hint))


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
    errors: list[ValidationMessage],
    warnings: list[ValidationMessage],
) -> tuple[dict[str, tuple[str | None, str | None]], str | None]:
    if not isinstance(document, (Mapping, Sequence)) or isinstance(document, (str, bytes, bytearray)):
        _append_error(
            errors,
            "license.revocation.invalid_format",
            "Lista odwołań ma niepoprawny format JSON.",
            hint="Zweryfikuj plik revocations.json i upewnij się, że zawiera poprawny obiekt JSON.",
        )
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
            _append_error(
                errors,
                "license.revocation.missing_section",
                "Lista odwołań nie zawiera sekcji 'revoked'.",
                hint="Dodaj sekcję 'revoked' z identyfikatorami licencji do listy odwołań.",
            )
        else:
            revoked_entries = _extract_revocation_entries(raw_revoked)
    else:
        revoked_entries = _extract_revocation_entries(document)

    if not revoked_entries and not errors:
        _append_warning(
            warnings,
            "license.revocation.empty",
            "Lista odwołań nie zawiera żadnych identyfikatorów licencji.",
            hint="Jeśli licencje są aktywne, pozostaw listę pustą; w przeciwnym razie dodaj odwołane identyfikatory.",
        )

    return revoked_entries, generated_at


def _parse_timestamp(
    value: str | None,
    *,
    label: str,
    errors: list[ValidationMessage],
    warnings: list[ValidationMessage],
) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        _append_error(
            errors,
            "license.timestamp.invalid",
            f"Pole {label} nie jest poprawnym znacznikiem czasu ISO 8601: {text}.",
            hint="Użyj formatu ISO 8601, np. 2024-01-01T00:00:00Z.",
        )
        return None
    if parsed.tzinfo is None:
        _append_warning(
            warnings,
            "license.timestamp.no_timezone",
            f"Pole {label} nie zawiera strefy czasowej – przyjmuję, że odnosi się do UTC.",
            hint="Dodaj oznaczenie strefy czasowej, np. sufiks Z dla UTC.",
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
    errors: list[ValidationMessage],
    warnings: list[ValidationMessage],
    label: str,
) -> str | None:
    if not keys:
        _append_warning(
            warnings,
            "license.signature.keys_missing",
            f"Pominięto weryfikację podpisu {label} – brak dostępnych kluczy.",
            hint="Dodaj odpowiednie klucze HMAC do konfiguracji, aby zweryfikować podpis.",
        )
        return None

    key_id_val = signature.get("key_id")
    if not isinstance(key_id_val, str) or not key_id_val.strip():
        _append_error(
            errors,
            "license.signature.missing_key_id",
            f"Podpis {label} nie zawiera identyfikatora klucza.",
            hint="Uzupełnij pole key_id w podpisie, aby wskazać użyty klucz.",
        )
        return None
    key_id = key_id_val.strip()
    try:
        key_bytes = keys[key_id]
    except KeyError:
        _append_error(
            errors,
            "license.signature.key_missing",
            f"Brak klucza HMAC '{key_id}' do weryfikacji podpisu {label}.",
            hint="Dołącz właściwy klucz do magazynu kluczy licencyjnych.",
        )
        return None

    algorithm = str(signature.get("algorithm") or "HMAC-SHA256")
    expected = build_hmac_signature(payload, key=key_bytes, algorithm=algorithm, key_id=key_id)
    actual = signature.get("value")
    if not isinstance(actual, str) or actual != expected.get("value"):
        _append_error(
            errors,
            "license.signature.mismatch",
            f"Sygnatura {label} niezgodna z oczekiwanym HMAC.",
            hint="Upewnij się, że plik licencji pochodzi od producenta i nie został zmodyfikowany.",
        )
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
    errors: list[ValidationMessage] = []
    warnings: list[ValidationMessage] = []
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
        _append_error(
            errors,
            "license.file.missing",
            f"Brak pliku licencji: {license_file}",
            hint="Zweryfikuj konfigurację license_path i skopiuj aktualny pakiet licencyjny.",
        )
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
        _append_error(
            errors,
            "license.file.invalid_json",
            str(exc),
            hint="Upewnij się, że plik licencji nie został uszkodzony ani ręcznie zmodyfikowany.",
        )
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
        _append_error(
            errors,
            "license.structure.missing_sections",
            "Licencja nie zawiera struktur payload/signature.",
            hint="Sprawdź, czy pakiet licencyjny nie został przycięty lub niekompletnie skopiowany.",
        )
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
        _append_error(
            errors,
            "license.field.profile_missing",
            "Licencja nie zawiera pola profile.",
            hint="Upewnij się, że plik licencji zawiera sekcję 'profile'.",
        )
    elif allowed_profile_set and profile.lower() not in allowed_profile_set:
        _append_error(
            errors,
            "license.field.profile_not_allowed",
            f"Profil licencji '{profile}' nie znajduje się na liście dozwolonych: {sorted(allowed_profile_set)}.",
            hint="Wybierz licencję przeznaczoną dla konfiguracji docelowej.",
        )

    if issuer is None:
        _append_error(
            errors,
            "license.field.issuer_missing",
            "Licencja nie zawiera pola issuer.",
            hint="Kontaktuj się z dostawcą licencji w celu uzyskania poprawnego pliku.",
        )
    elif allowed_issuer_set and issuer.lower() not in allowed_issuer_set:
        _append_error(
            errors,
            "license.field.issuer_not_allowed",
            f"Wystawca licencji '{issuer}' nie jest dozwolony: {sorted(allowed_issuer_set)}.",
            hint="Zaimportuj licencję wystawioną przez autoryzowanego partnera.",
        )

    if schema is None:
        _append_error(
            errors,
            "license.field.schema_missing",
            "Licencja nie zawiera pola schema.",
            hint="Upewnij się, że używasz aktualnego formatu pakietu licencyjnego.",
        )
    elif required_schema and schema.lower() != required_schema.strip().lower():
        _append_error(
            errors,
            "license.field.schema_unexpected",
            f"Licencja ma nieoczekiwany typ schematu '{schema}' – oczekiwano '{required_schema}'.",
            hint="Zastosuj licencję zgodną z wymaganą gałęzią oprogramowania.",
        )

    if schema_version is None:
        _append_error(
            errors,
            "license.field.schema_version_missing",
            "Licencja nie zawiera pola schema_version.",
            hint="Zaktualizuj licencję do wspieranego formatu.",
        )
    elif allowed_schema_versions_set and schema_version not in allowed_schema_versions_set:
        _append_error(
            errors,
            "license.field.schema_version_unsupported",
            "Licencja ma nieobsługiwaną wersję schematu '%s'. Dozwolone: %s."
            % (schema_version, sorted(allowed_schema_versions_set)),
            hint="Użyj licencji przygotowanej dla tej wersji oprogramowania.",
        )

    if license_id is None:
        _append_error(
            errors,
            "license.field.id_missing",
            "Licencja nie zawiera pola license_id.",
            hint="Skontaktuj się z wydawcą licencji po poprawny dokument.",
        )

    if issued_at is None:
        _append_error(
            errors,
            "license.field.issued_at_missing",
            "Licencja nie zawiera pola issued_at.",
            hint="Licencja musi zawierać datę wystawienia w formacie ISO 8601.",
        )
    if expires_at is None:
        _append_error(
            errors,
            "license.field.expires_at_missing",
            "Licencja nie zawiera pola expires_at.",
            hint="Skontaktuj się z wydawcą licencji po zaktualizowany dokument.",
        )

    if license_keys is None:
        _append_warning(
            warnings,
            "license.signature.license_keys_missing",
            "Pominięto weryfikację podpisu licencji – brak konfiguracji kluczy.",
            hint="Ustaw ścieżkę do pliku kluczy licencyjnych w konfiguracji OEM.",
        )
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
            _append_error(
                errors,
                "license.fingerprint.payload_mismatch",
                "Fingerprint w licencji nie jest zgodny z podpisanym payloadem.",
                hint="Ponownie wygeneruj pakiet licencyjny wraz z fingerprintem urządzenia.",
            )
        fingerprint_value = fingerprint_value or fp_value
        if fingerprint_keys is None:
            _append_warning(
                warnings,
                "license.fingerprint.keys_missing",
                "Pominięto weryfikację podpisu fingerprintu – brak konfiguracji kluczy.",
                hint="Ustaw ścieżkę do pliku kluczy fingerprintu.",
            )
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
            _append_error(
                errors,
                "license.fingerprint.signature_missing",
                "Licencja nie zawiera podpisu fingerprintu.",
                hint="Dostarcz plik fingerprintu z podpisem HMAC producenta.",
            )

    file_fingerprint_path = None
    if fingerprint_path:
        fingerprint_file = Path(fingerprint_path).expanduser()
        file_fingerprint_path = fingerprint_file
        if not fingerprint_file.exists():
            _append_error(
                errors,
                "license.fingerprint.file_missing",
                f"Brak pliku fingerprintu: {fingerprint_file}",
                hint="Wskaż poprawną ścieżkę fingerprint_path lub ponownie wyeksportuj fingerprint.",
            )
        else:
            try:
                fp_document = _load_json(fingerprint_file)
            except ValueError as exc:
                _append_error(
                    errors,
                    "license.fingerprint.invalid_json",
                    str(exc),
                    hint="Zweryfikuj integralność pliku fingerprintu.",
                )
            else:
                fp_payload_doc = fp_document.get("payload") if isinstance(fp_document, Mapping) else None
                fp_signature_doc = fp_document.get("signature") if isinstance(fp_document, Mapping) else None
                if isinstance(fp_payload_doc, Mapping):
                    fp_value = _extract_fingerprint(fp_payload_doc.get("fingerprint"))
                    if fp_value and fingerprint_value and fp_value != fingerprint_value:
                        _append_error(
                            errors,
                            "license.fingerprint.external_mismatch",
                            "Fingerprint w licencji nie zgadza się z plikiem fingerprintu.",
                            hint="Upewnij się, że używasz pary licencja + fingerprint z tego samego zestawu.",
                        )
                    fingerprint_value = fingerprint_value or fp_value
                    fingerprint_source = str(fingerprint_file)
                    if fingerprint_keys is None:
                        _append_warning(
                            warnings,
                            "license.fingerprint.file_keys_missing",
                            "Pominięto weryfikację podpisu fingerprintu z pliku – brak konfiguracji kluczy.",
                            hint="Dodaj klucze fingerprintu do konfiguracji, aby zweryfikować plik.",
                        )
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
                        _append_error(
                            errors,
                            "license.fingerprint.signature_section_missing",
                            "Plik fingerprintu nie zawiera sekcji signature.",
                            hint="Zweryfikuj, czy plik fingerprintu został wygenerowany poprawnym narzędziem.",
                        )
                else:
                    _append_error(
                        errors,
                        "license.fingerprint.invalid_structure",
                        "Plik fingerprintu ma nieoczekiwaną strukturę JSON.",
                        hint="Otwórz plik fingerprintu i sprawdź, czy zawiera sekcje payload/signature.",
                    )

    revocation_keys = revocation_keys or {}

    if revocation_file is None:
        if revocation_required:
            _append_error(
                errors,
                "license.revocation.required_missing",
                "Konfiguracja wymaga listy odwołań licencji, ale nie podano ścieżki.",
                hint="Ustaw parametr revocation_list_path lub wyłącz wymaganie listy odwołań.",
            )
            revocation_status = "missing"
        else:
            revocation_status = "skipped"
    else:
        revocation_checked = True
        if not revocation_file.exists():
            message = f"Brak pliku listy odwołań: {revocation_file}"
            if revocation_required:
                _append_error(
                    errors,
                    "license.revocation.file_missing",
                    message,
                    hint="Dostarcz aktualny plik listy odwołań lub wyłącz wymóg revocation_required.",
                )
            else:
                _append_warning(
                    warnings,
                    "license.revocation.file_missing",
                    message,
                    hint="Jeżeli korzystasz z listy odwołań, wskaż poprawną ścieżkę.",
                )
            revocation_status = "missing"
        else:
            try:
                revocation_document = _load_json(revocation_file)
            except ValueError as exc:
                if revocation_required:
                    _append_error(
                        errors,
                        "license.revocation.invalid_json",
                        str(exc),
                        hint="Zweryfikuj integralność pliku listy odwołań.",
                    )
                else:
                    _append_warning(
                        warnings,
                        "license.revocation.invalid_json",
                        str(exc),
                        hint="Plik listy odwołań jest ignorowany do czasu poprawienia formatu.",
                    )
                revocation_status = "error"
            else:
                local_errors: list[ValidationMessage] = []
                local_warnings: list[ValidationMessage] = []
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
                                _append_error(
                                    local_errors,
                                    "license.revocation.signature_invalid_format",
                                    "Podpis listy odwołań ma niepoprawny format.",
                                    hint="Sekcja 'signature' powinna być obiektem JSON z polami key_id/value.",
                                )
                    else:
                        _append_error(
                            local_errors,
                            "license.revocation.payload_structure_invalid",
                            "Sekcja payload listy odwołań ma nieoczekiwaną strukturę.",
                            hint="Payload listy odwołań powinien być tablicą lub obiektem JSON.",
                        )
                elif revocation_signature_required:
                    _append_error(
                        local_errors,
                        "license.revocation.signature_required_missing",
                        "Konfiguracja wymaga podpisanej listy odwołań, ale dokument nie zawiera sekcji 'payload' i 'signature'.",
                        hint="Upewnij się, że plik revocations.json zawiera pola payload/signature.",
                    )

                payload_is_json = isinstance(payload_document, Mapping) or (
                    isinstance(payload_document, Sequence)
                    and not isinstance(payload_document, (str, bytes, bytearray))
                )

                if signature_document is not None and not payload_is_json:
                    _append_error(
                        local_errors,
                        "license.revocation.payload_signature_incompatible",
                        "Sekcja payload listy odwołań ma nieobsługiwany format do weryfikacji podpisu.",
                        hint="Payload musi być obiektem JSON lub tablicą do poprawnej walidacji podpisu.",
                    )
                elif signature_document is not None:
                    if not revocation_keys:
                        message = (
                            "Pominięto weryfikację podpisu listy odwołań – brak dostępnych kluczy."
                        )
                        if revocation_signature_required:
                            _append_error(
                                local_errors,
                                "license.revocation.signature_keys_missing",
                                message,
                                hint="Dodaj klucze HMAC listy odwołań do konfiguracji.",
                            )
                        else:
                            _append_warning(
                                local_warnings,
                                "license.revocation.signature_keys_missing",
                                message,
                                hint="Podpis listy odwołań został pominięty z powodu braku kluczy.",
                            )
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
                            _append_error(
                                local_errors,
                                "license.revocation.signature_verification_failed",
                                "Podpis listy odwołań nie został zweryfikowany.",
                                hint="Zweryfikuj klucz podpisujący listę odwołań i integralność pliku.",
                            )
                elif revocation_signature_required:
                    _append_error(
                        local_errors,
                        "license.revocation.signature_section_missing",
                        "Konfiguracja wymaga podpisanej listy odwołań, ale dokument nie zawiera sekcji 'signature'.",
                        hint="Dodaj podpis HMAC do listy odwołań lub wyłącz wymóg podpisu.",
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
                    timestamp_errors: list[ValidationMessage] = []
                    timestamp_warnings: list[ValidationMessage] = []
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
                        _append_error(
                            errors,
                            "license.revocation.max_age_invalid",
                            "Konfiguracja revocation_list_max_age_hours musi być liczbą.",
                            hint="Podaj wartość numeryczną w godzinach, np. 24.",
                        )
                    else:
                        if max_age <= timedelta(0):
                            _append_error(
                                errors,
                                "license.revocation.max_age_non_positive",
                                "Konfiguracja revocation_list_max_age_hours musi być dodatnia.",
                                hint="Ustaw dodatnią liczbę godzin, np. 24.",
                            )
                        elif generated_dt and now - generated_dt > max_age:
                            message = (
                                "Lista odwołań licencji jest starsza niż dozwolone %.1f godzin."
                                % (max_age.total_seconds() / 3600.0)
                            )
                            if revocation_required:
                                _append_error(
                                    errors,
                                    "license.revocation.list_stale",
                                    message,
                                    hint="Pobierz aktualną listę odwołań od producenta.",
                                )
                            else:
                                _append_warning(
                                    warnings,
                                    "license.revocation.list_stale",
                                    message,
                                    hint="Rozważ odświeżenie listy odwołań, aby zachować aktualność.",
                                )
                            if revocation_status != "revoked":
                                revocation_status = "stale"
                elif revocation_list_max_age_hours not in (None, float("inf")) and not generated_at:
                    if revocation_required:
                        _append_error(
                            errors,
                            "license.revocation.generated_at_missing",
                            "Lista odwołań nie zawiera znacznika generated_at – nie można ocenić świeżości.",
                            hint="Dodaj pole generated_at do listy odwołań.",
                        )
                    else:
                        _append_warning(
                            warnings,
                            "license.revocation.generated_at_missing",
                            "Lista odwołań nie zawiera znacznika generated_at – nie można ocenić świeżości.",
                            hint="Dodaj pole generated_at, aby monitorować aktualność listy.",
                        )
                    if revocation_status is None:
                        revocation_status = "unknown"

                if revoked_entries:
                    if license_id and license_id in revoked_entries:
                        entry_reason, entry_revoked_at = revoked_entries[license_id]
                        revocation_reason = entry_reason
                        if entry_revoked_at:
                            revoked_at_errors: list[ValidationMessage] = []
                            revoked_at_warnings: list[ValidationMessage] = []
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
                        _append_error(
                            errors,
                            "license.revocation.entry_match",
                            f"Licencja {license_id} znajduje się na liście odwołań.",
                            hint="Skontaktuj się z dostawcą licencji w celu odnowienia uprawnień.",
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
        _append_warning(
            warnings,
            "license.time.issued_in_future",
            f"Data wystawienia licencji ({issued_dt.isoformat()}) znajduje się w przyszłości.",
            hint="Sprawdź ustawienia zegara systemowego lub poprawność dokumentu licencyjnego.",
        )

    if issued_dt and expires_dt and expires_dt <= issued_dt:
        _append_error(
            errors,
            "license.time.invalid_window",
            "Data wygaśnięcia licencji jest wcześniejsza lub równa dacie wystawienia.",
            hint="Skontaktuj się z wydawcą licencji w celu korekty okresu ważności.",
        )

    if expires_dt:
        if expires_dt <= now:
            _append_error(
                errors,
                "license.time.expired",
                f"Licencja wygasła {expires_dt.isoformat()}.",
                hint="Odnów licencję, aby kontynuować pracę programu.",
            )
        else:
            remaining = expires_dt - now
            if remaining <= timedelta(days=30):
                _append_warning(
                    warnings,
                    "license.time.expiring_soon",
                    "Licencja wygaśnie w ciągu 30 dni (data: "
                    f"{expires_dt.isoformat()}).",
                    hint="Zaplanowanie odnowienia licencji zapewni ciągłość działania.",
                )

    if issued_dt and expires_dt and max_validity_days not in (None, float("inf")):
        try:
            max_validity = timedelta(days=float(max_validity_days))
        except (TypeError, ValueError):
            _append_error(
                errors,
                "license.time.max_validity_invalid",
                "Konfiguracja max_validity_days musi być liczbą.",
                hint="Zdefiniuj limit ważności jako liczbę dni, np. 365.",
            )
        else:
            if max_validity <= timedelta(0):
                _append_error(
                    errors,
                    "license.time.max_validity_non_positive",
                    "Konfiguracja max_validity_days musi być dodatnia.",
                    hint="Wprowadź dodatnią liczbę dni.",
                )
            else:
                validity_window = expires_dt - issued_dt
                if validity_window > max_validity:
                    validity_days = validity_window.total_seconds() / 86400.0
                    _append_error(
                        errors,
                        "license.time.max_validity_exceeded",
                        "Okres ważności licencji (" "%.1f dni) przekracza dozwolone %.1f dni." % (
                            validity_days,
                            max_validity.total_seconds() / 86400.0,
                        ),
                        hint="Zastosuj licencję z krótszym okresem ważności lub zaktualizuj konfigurację limitu.",
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
