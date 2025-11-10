from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, MutableMapping

from bot_core.security.fingerprint import verify_document
from bot_core.security.license import (
    LicenseValidationResult,
    load_hmac_keys_file,
    validate_license,
)
@dataclass(slots=True)
class FingerprintValidationResult:
    status: str
    fingerprint: str | None
    key_id: str | None
    errors: list[str]
    warnings: list[str]
    payload: Mapping[str, object] | None

    @property
    def is_valid(self) -> bool:
        return self.status == "ok" and not self.errors


def _load_document(path: Path) -> Mapping[str, object]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Dokument fingerprintu powinien być obiektem JSON.")
    return data


def validate_fingerprint_document(
    *,
    document_path: str | Path,
    keys_path: str | Path | None = None,
    keys: Mapping[str, bytes] | None = None,
) -> FingerprintValidationResult:
    path = Path(document_path).expanduser()
    if not path.exists():
        return FingerprintValidationResult(
            status="missing",
            fingerprint=None,
            key_id=None,
            errors=[f"Brak pliku fingerprintu: {path}"],
            warnings=[],
            payload=None,
        )

    try:
        document = _load_document(path)
    except (ValueError, json.JSONDecodeError) as exc:
        return FingerprintValidationResult(
            status="invalid",
            fingerprint=None,
            key_id=None,
            errors=[f"Niepoprawny plik fingerprintu: {exc}"],
            warnings=[],
            payload=None,
        )

    fingerprint_value = None
    payload = document.get("payload") if isinstance(document, Mapping) else None
    if isinstance(payload, Mapping):
        fp = payload.get("fingerprint")
        if isinstance(fp, Mapping):
            value = fp.get("value")
            if isinstance(value, str):
                fingerprint_value = value
            elif isinstance(fp.get("id"), str):
                fingerprint_value = fp["id"]
        elif isinstance(fp, str):
            fingerprint_value = fp

    loaded_keys: MutableMapping[str, bytes] = {}
    if keys_path:
        loaded_keys.update(load_hmac_keys_file(keys_path))
    if keys:
        loaded_keys.update(keys)

    errors: list[str] = []
    warnings: list[str] = []
    matched_key: str | None = None
    if loaded_keys:
        for key_id, secret in loaded_keys.items():
            try:
                if verify_document(document, key=secret):
                    matched_key = key_id
                    break
            except Exception as exc:  # pragma: no cover - obrona przed złym formatem
                warnings.append(f"Nie udało się zweryfikować podpisu kluczem {key_id}: {exc}")
    else:
        warnings.append("Nie dostarczono kluczy HMAC – pomijam weryfikację podpisu fingerprintu.")

    status = "ok" if matched_key or not loaded_keys else "unknown"
    if matched_key is None and loaded_keys:
        status = "invalid"
        errors.append("Żaden z kluczy HMAC nie potwierdził podpisu dokumentu fingerprintu.")

    return FingerprintValidationResult(
        status=status,
        fingerprint=fingerprint_value,
        key_id=matched_key,
        errors=errors,
        warnings=warnings,
        payload=payload if isinstance(payload, Mapping) else None,
    )


def validate_license_bundle(
    *,
    license_path: str | Path,
    license_keys_path: str | Path | None = None,
    fingerprint_path: str | Path | None = None,
    fingerprint_keys_path: str | Path | None = None,
) -> LicenseValidationResult:
    license_keys = load_hmac_keys_file(license_keys_path) if license_keys_path else None
    fingerprint_keys = load_hmac_keys_file(fingerprint_keys_path) if fingerprint_keys_path else None
    return validate_license(
        license_path=license_path,
        license_keys=license_keys,
        fingerprint_path=fingerprint_path,
        fingerprint_keys=fingerprint_keys,
    )


__all__ = [
    "FingerprintValidationResult",
    "validate_fingerprint_document",
    "validate_license_bundle",
]
