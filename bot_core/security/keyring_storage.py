"""Implementacja magazynu sekretów oparta o bibliotekę ``keyring``."""
from __future__ import annotations

from typing import Optional

from bot_core.security.base import SecretStorage, SecretStorageError


class KeyringSecretStorage(SecretStorage):
    """Odwzorowuje interfejs ``SecretStorage`` na natywne keychainy systemów operacyjnych."""

    def __init__(self, *, service_name: str = "dudzian.trading") -> None:
        try:
            import keyring  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - zależne od środowiska CI
            raise SecretStorageError(
                "Biblioteka 'keyring' nie jest dostępna. Zainstaluj ją, aby korzystać z natywnego "
                "przechowywania sekretów (pip install keyring)."
            ) from exc

        self._keyring = keyring
        self._service_name = service_name

    def get_secret(self, key: str) -> Optional[str]:
        return self._keyring.get_password(self._service_name, key)

    def set_secret(self, key: str, value: str) -> None:
        result = self._keyring.set_password(self._service_name, key, value)
        if result is not None:  # pragma: no cover - większość backendów zwraca None
            raise SecretStorageError(f"Nie udało się zapisać sekretu '{key}'.")

    def delete_secret(self, key: str) -> None:
        try:
            self._keyring.delete_password(self._service_name, key)
        except self._keyring.errors.PasswordDeleteError:
            # Wyrównujemy zachowanie do idempotentnego kasowania – ignorujemy brak wpisu.
            return


__all__ = ["KeyringSecretStorage"]
