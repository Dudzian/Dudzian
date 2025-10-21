"""Wspólne helpery dla backendów raportowania wykorzystujących AWS S3."""

from __future__ import annotations

import json
from typing import Mapping

from bot_core.security import SecretManager, SecretStorageError


def load_s3_credentials(
    secret_manager: SecretManager | None,
    secret_name: str | None,
) -> Mapping[str, str]:
    """Odczytuje parę kluczy dostępowych z magazynu sekretów.

    Funkcja pilnuje identycznych komunikatów błędów jak historyczne
    implementacje w modułach raportowania, aby regresje były natychmiastowo
    wychwytywane przez istniejące testy.
    """

    if not secret_name:
        raise SecretStorageError(
            "Konfiguracja backendu 's3' wymaga podania credential_secret z kluczami dostępowymi"
        )
    if secret_manager is None:
        raise SecretStorageError("Brak dostępu do SecretManager przy backendzie 's3'")

    raw_value = secret_manager.load_secret_value(secret_name)
    try:
        payload = json.loads(raw_value)
    except json.JSONDecodeError as exc:  # pragma: no cover - walidacja błędu
        raise SecretStorageError("Sekret S3 ma nieprawidłowy format JSON") from exc

    expected_keys = {"access_key_id", "secret_access_key"}
    missing = sorted(key for key in expected_keys if key not in payload)
    if missing:
        raise SecretStorageError(
            "Sekret S3 nie zawiera wymaganych pól: " + ", ".join(missing)
        )

    return {str(key): str(value) for key, value in payload.items()}


__all__ = ["load_s3_credentials"]

