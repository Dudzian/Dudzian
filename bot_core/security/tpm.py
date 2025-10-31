"""Walidacja dowodów TPM / secure enclave bez uruchamiania procesów pomocniczych."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from nacl.exceptions import BadSignatureError
from nacl.signing import VerifyKey

from bot_core.security.messages import ValidationMessage, make_error, make_warning
from bot_core.security.signing import canonical_json_bytes

TPM_KEYRING_ENV = "BOT_CORE_TPM_PUBLIC_KEYS"


class TpmValidationError(RuntimeError):
    """Sygnalizuje niepoprawny format dowodu TPM."""


@dataclass(slots=True)
class TpmValidationResult:
    """Zbiorczy wynik walidacji dowodu TPM/secure enclave."""

    status: str
    fingerprint: str | None
    sealed_fingerprint: str | None
    nonce: str | None
    attested_at: datetime | None
    expires_at: datetime | None
    secure_enclave: str | None
    signature_verified: bool
    signature_key: str | None
    warnings: list[ValidationMessage]
    errors: list[ValidationMessage]
    payload: Mapping[str, Any]

    @property
    def is_valid(self) -> bool:
        return self.status == "ok" and not self.errors

    def to_dict(self) -> dict[str, Any]:
        warning_payload = [entry.to_dict() for entry in self.warnings]
        error_payload = [entry.to_dict() for entry in self.errors]

        def _serialize_dt(value: datetime | None) -> str | None:
            if value is None:
                return None
            return value.astimezone(timezone.utc).isoformat()

        return {
            "status": self.status,
            "fingerprint": self.fingerprint,
            "sealed_fingerprint": self.sealed_fingerprint,
            "nonce": self.nonce,
            "attested_at": _serialize_dt(self.attested_at),
            "expires_at": _serialize_dt(self.expires_at),
            "secure_enclave": self.secure_enclave,
            "signature_verified": self.signature_verified,
            "signature_key": self.signature_key,
            "warnings": warning_payload,
            "errors": error_payload,
            "warning_messages": [entry.message for entry in self.warnings],
            "error_messages": [entry.message for entry in self.errors],
            "payload": dict(self.payload),
        }


def _load_keyring(path_or_payload: str | None) -> dict[str, VerifyKey]:
    raw = path_or_payload or os.environ.get(TPM_KEYRING_ENV)
    if not raw:
        return {}

    candidate_path = Path(raw)
    if candidate_path.exists():
        text = candidate_path.read_text(encoding="utf-8")
    else:
        text = raw

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - konfiguracja środowiska
        raise TpmValidationError(f"Niepoprawny JSON z kluczami TPM: {exc}") from exc

    if isinstance(document, Mapping) and "keys" in document:
        mapping = document.get("keys")
    else:
        mapping = document

    if not isinstance(mapping, Mapping):
        raise TpmValidationError("Klucze TPM muszą być obiektem mapującym identyfikator na klucz hex")

    result: dict[str, VerifyKey] = {}
    for key_id, value in mapping.items():
        if not isinstance(key_id, str):
            raise TpmValidationError("Identyfikatory kluczy TPM muszą być napisami")
        if not isinstance(value, str):
            raise TpmValidationError(f"Klucz TPM {key_id} musi być zakodowany jako string hex")
        try:
            result[key_id] = VerifyKey(bytes.fromhex(value.strip()))
        except ValueError as exc:  # pragma: no cover - błędna konfiguracja
            raise TpmValidationError(f"Klucz TPM {key_id} nie jest poprawnym ciągiem hex") from exc
    return result


def _decode_payload(document: Mapping[str, Any]) -> tuple[Mapping[str, Any], bytes]:
    if "payload_b64" in document:
        payload_b64 = document["payload_b64"]
        if not isinstance(payload_b64, str):
            raise TpmValidationError("Pole payload_b64 w dowodzie TPM musi być tekstem (base64)")
        try:
            payload_bytes = base64.b64decode(payload_b64.encode("ascii"))
        except Exception as exc:  # pragma: no cover - walidacja wejścia
            raise TpmValidationError("Nie można zdekodować payload_b64 (base64)") from exc
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise TpmValidationError("Dekodowany payload TPM nie zawiera poprawnego JSON") from exc
        if not isinstance(payload, Mapping):
            raise TpmValidationError("Dekodowany payload TPM musi być obiektem JSON")
        return payload, payload_bytes

    payload = document.get("payload")
    if payload is None:
        payload = document
    if not isinstance(payload, Mapping):
        raise TpmValidationError("Dowód TPM musi zawierać obiekt payload")
    payload_bytes = canonical_json_bytes(payload)
    return payload, payload_bytes


def _parse_signature(document: Mapping[str, Any]) -> tuple[str | None, bytes | None]:
    signature = document.get("signature")
    if signature is None:
        return None, None
    if not isinstance(signature, Mapping):
        raise TpmValidationError("Sekcja signature w dowodzie TPM musi być obiektem")
    algorithm = signature.get("algorithm")
    if not isinstance(algorithm, str):
        raise TpmValidationError("Signature.algorithm musi być tekstem")
    if algorithm.lower() != "ed25519":
        raise TpmValidationError(f"Nieobsługiwany algorytm podpisu TPM: {algorithm}")
    value = signature.get("value")
    if not isinstance(value, str):
        raise TpmValidationError("Signature.value musi być napisem (base64)")
    try:
        signature_bytes = base64.b64decode(value.encode("ascii"))
    except Exception as exc:  # pragma: no cover
        raise TpmValidationError("Nie można zdekodować signature.value (base64)") from exc
    key_id = signature.get("key_id")
    if key_id is not None and not isinstance(key_id, str):
        raise TpmValidationError("Signature.key_id musi być tekstem")
    return key_id, signature_bytes


def _parse_datetime(value: Any, *, field: str) -> datetime | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TpmValidationError(f"Pole {field} w dowodzie TPM musi być napisem ISO8601")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise TpmValidationError(f"Pole {field} ma niepoprawny format daty: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def validate_attestation(
    *,
    evidence_path: str | Path,
    expected_fingerprint: str | None,
    keyring: str | None = None,
) -> TpmValidationResult:
    """Waliduje dowód TPM i zwraca informacje przyjazne dla UI."""

    evidence_file = Path(evidence_path).expanduser()
    if not evidence_file.exists():
        raise FileNotFoundError(evidence_file)

    try:
        document = json.loads(evidence_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TpmValidationError("Plik dowodu TPM zawiera niepoprawny JSON") from exc
    if not isinstance(document, Mapping):
        raise TpmValidationError("Plik dowodu TPM musi zawierać obiekt JSON")

    payload, payload_bytes = _decode_payload(document)
    key_id, signature_bytes = _parse_signature(document)
    keyring_map = _load_keyring(keyring)

    fingerprint = None
    sealed_fingerprint = None
    nonce = None
    secure_enclave = None
    attested_at: datetime | None = None
    expires_at: datetime | None = None
    warnings: list[ValidationMessage] = []
    errors: list[ValidationMessage] = []

    if isinstance(payload.get("fingerprint"), str):
        fingerprint = payload["fingerprint"].strip() or None
    if isinstance(payload.get("sealed_fingerprint"), str):
        sealed_fingerprint = payload["sealed_fingerprint"].strip() or None
    if isinstance(payload.get("nonce"), str):
        nonce = payload["nonce"].strip() or None
    enclave_info = payload.get("secure_enclave")
    if isinstance(enclave_info, Mapping):
        secure_enclave = str(enclave_info.get("type") or enclave_info.get("vendor") or "tpm")
    elif isinstance(enclave_info, str):
        secure_enclave = enclave_info.strip() or None
    attested_at = _parse_datetime(payload.get("attested_at"), field="attested_at")
    expires_at = _parse_datetime(payload.get("expires_at"), field="expires_at")

    signature_verified = False
    signature_key: str | None = None
    if signature_bytes is not None:
        if key_id is None and len(keyring_map) == 1:
            key_id = next(iter(keyring_map))
        if key_id is None:
            warnings.append(
                make_warning(
                    "tpm.signature.missing_key_id",
                    "Dowód TPM nie określa key_id podpisu – pominięto weryfikację.",
                    hint="Dodaj pole signature.key_id do pliku dowodu, aby umożliwić weryfikację.",
                )
            )
        else:
            key = keyring_map.get(key_id)
            if key is None:
                warnings.append(
                    make_warning(
                        "tpm.key.unknown",
                        f"Brak klucza TPM o identyfikatorze {key_id} – pominięto weryfikację.",
                        hint="Zainstaluj właściwy publiczny klucz TPM lub zaktualizuj plik keyring.",
                    )
                )
            else:
                try:
                    key.verify(payload_bytes, signature_bytes)
                except BadSignatureError:
                    errors.append(
                        make_error(
                            "tpm.signature.invalid",
                            "Podpis dowodu TPM jest niepoprawny.",
                            hint="Wygeneruj nowy dowód TPM na właściwym urządzeniu i upewnij się, że korzystasz z aktualnego klucza.",
                        )
                    )
                else:
                    signature_verified = True
                    signature_key = key_id
    else:
        warnings.append(
            make_warning(
                "tpm.signature.missing",
                "Dowód TPM nie zawiera podpisu – wynik ma charakter informacyjny.",
                hint="Skonfiguruj proces generowania dowodu tak, aby zawierał podpis Ed25519.",
            )
        )

    status = "ok"
    if expected_fingerprint:
        candidate = sealed_fingerprint or fingerprint
        if not candidate or not candidate.startswith(expected_fingerprint[:16]):
            errors.append(
                make_error(
                    "tpm.fingerprint.mismatch",
                    "Dowód TPM nie jest powiązany z aktualną licencją.",
                    hint="Zweryfikuj przypisanie licencji do tego urządzenia lub wygeneruj nowy fingerprint.",
                )
            )
            status = "mismatch"
    if expires_at and expires_at < datetime.now(timezone.utc):
        warnings.append(
            make_warning(
                "tpm.attestation.expired",
                "Dowód TPM wygasł – przeprowadź ponowną atestację.",
                hint="Uruchom procedurę atestacji TPM, aby odświeżyć dowód.",
            )
        )

    return TpmValidationResult(
        status=status,
        fingerprint=fingerprint,
        sealed_fingerprint=sealed_fingerprint,
        nonce=nonce,
        attested_at=attested_at,
        expires_at=expires_at,
        secure_enclave=secure_enclave,
        signature_verified=signature_verified,
        signature_key=signature_key,
        warnings=warnings,
        errors=errors,
        payload=payload,
    )


__all__ = [
    "TpmValidationResult",
    "TpmValidationError",
    "validate_attestation",
]

