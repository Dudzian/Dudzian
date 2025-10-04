"""Abstrakcje przechowywania sekretów wykorzystywane przez adaptery giełdowe."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Sequence

from bot_core.exchanges.base import Environment, ExchangeCredentials


class SecretStorageError(RuntimeError):
    """Błąd zgłaszany, gdy operacja na magazynie sekretów się nie powiodła."""


class SecretStorage(abc.ABC):
    """Abstrakcyjna definicja magazynu sekretów."""

    @abc.abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        """Zwraca sekret zapisany pod wskazanym kluczem lub ``None``."""

    @abc.abstractmethod
    def set_secret(self, key: str, value: str) -> None:
        """Zapisuje lub aktualizuje sekret podanym kluczem."""

    @abc.abstractmethod
    def delete_secret(self, key: str) -> None:
        """Usuwa sekret. Operacja powinna być idempotentna."""


@dataclass(slots=True)
class SecretPayload:
    """Reprezentuje komplet informacji niezbędnych do uwierzytelnienia na giełdzie."""

    key_id: str
    secret: str
    passphrase: Optional[str]
    permissions: Sequence[str]
    environment: Environment

    @classmethod
    def from_exchange_credentials(
        cls, credentials: ExchangeCredentials
    ) -> "SecretPayload":
        return cls(
            key_id=credentials.key_id,
            secret=credentials.secret or "",
            passphrase=credentials.passphrase,
            permissions=tuple(str(permission).lower() for permission in credentials.permissions),
            environment=credentials.environment,
        )


class SecretManager:
    """Zarządza serializacją i walidacją poświadczeń w magazynach sekretów."""

    def __init__(self, storage: SecretStorage, *, namespace: str = "dudzian.trading") -> None:
        self._storage = storage
        self._namespace = namespace.rstrip(":")

    def _format_key(self, keychain_key: str, purpose: str) -> str:
        return f"{self._namespace}:{keychain_key}:{purpose}".lower()

    def load_exchange_credentials(
        self,
        keychain_key: str,
        *,
        expected_environment: Environment,
        purpose: str = "trading",
        required_permissions: Sequence[str] | None = None,
        forbidden_permissions: Sequence[str] | None = None,
    ) -> ExchangeCredentials:
        """Ładuje poświadczenia i weryfikuje zgodność środowiska."""

        storage_key = self._format_key(keychain_key, purpose)
        raw_value = self._storage.get_secret(storage_key)
        if raw_value is None:
            raise SecretStorageError(
                f"Brak sekretu '{storage_key}'. Dodaj klucze przy użyciu narzędzia konfiguracyjnego."
            )

        environment = expected_environment

        try:
            payload = self._deserialize(
                raw_value,
                expected_environment=environment,
            )
        except ValueError as exc:  # pragma: no cover - ochrona przed zepsutymi danymi
            raise SecretStorageError(
                f"Sekret '{storage_key}' ma niepoprawny format – usuń go i zapisz ponownie."
            ) from exc

        if payload.environment != expected_environment:
            raise SecretStorageError(
                "Środowisko zapisane w magazynie sekretów ("
                f"{payload.environment.value}) nie pasuje do oczekiwanego "
                f"({expected_environment.value})."
            )

        stored_permissions = {str(permission).lower() for permission in payload.permissions}

        if required_permissions:
            required = {str(permission).lower() for permission in required_permissions}
            missing = sorted(permission for permission in required if permission not in stored_permissions)
            if missing:
                missing_str = ", ".join(missing)
                raise SecretStorageError(
                    "Klucz API zapisany w magazynie nie posiada wymaganych uprawnień: "
                    f"{missing_str}. Zaktualizuj uprawnienia w panelu giełdy i ponownie zapisz sekret."
                )

        if forbidden_permissions:
            forbidden = {str(permission).lower() for permission in forbidden_permissions}
            present = sorted(permission for permission in stored_permissions if permission in forbidden)
            if present:
                present_str = ", ".join(present)
                raise SecretStorageError(
                    "Klucz API zapisany w magazynie posiada zabronione uprawnienia: "
                    f"{present_str}. Usuń klucz lub odepnij zbędne uprawnienia przed dalszym użyciem."
                )

        return ExchangeCredentials(
            key_id=payload.key_id,
            secret=payload.secret or None,
            passphrase=payload.passphrase,
            environment=payload.environment,
            permissions=payload.permissions,
        )

    def store_exchange_credentials(
        self,
        keychain_key: str,
        credentials: ExchangeCredentials,
        *,
        purpose: str = "trading",
    ) -> None:
        storage_key = self._format_key(keychain_key, purpose)
        serialized = self._serialize(SecretPayload.from_exchange_credentials(credentials))
        self._storage.set_secret(storage_key, serialized)

    def delete_exchange_credentials(self, keychain_key: str, *, purpose: str = "trading") -> None:
        storage_key = self._format_key(keychain_key, purpose)
        self._storage.delete_secret(storage_key)

    def load_secret_value(self, keychain_key: str, *, purpose: str = "generic") -> str:
        """Zwraca arbitralny sekret zapisany w magazynie.

        Funkcja jest wykorzystywana do ładowania tokenów kanałów alertów oraz innych
        poświadczeń, które nie pasują do schematu ``ExchangeCredentials``. Zwracamy
        prosty łańcuch znaków, aby wywołujący mógł sam zdecydować o formacie
        serializacji (np. JSON dla kont SMTP).
        """

        storage_key = self._format_key(keychain_key, purpose)
        raw_value = self._storage.get_secret(storage_key)
        if raw_value is None:
            raise SecretStorageError(
                f"Brak sekretu '{storage_key}'. Dodaj go do natywnego keychaina przed startem systemu."
            )
        return raw_value

    def store_secret_value(self, keychain_key: str, value: str, *, purpose: str = "generic") -> None:
        """Zapisuje arbitralny sekret w magazynie."""

        storage_key = self._format_key(keychain_key, purpose)
        self._storage.set_secret(storage_key, value)

    def delete_secret_value(self, keychain_key: str, *, purpose: str = "generic") -> None:
        """Usuwa sekret zapisany dla wskazanego celu."""

        storage_key = self._format_key(keychain_key, purpose)
        self._storage.delete_secret(storage_key)

    @staticmethod
    def _serialize(payload: SecretPayload) -> str:
        import json

        return json.dumps(
            {
                "key_id": payload.key_id,
                "secret": payload.secret,
                "passphrase": payload.passphrase,
                "permissions": list(payload.permissions),
                "environment": payload.environment.value,
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _deserialize(
        raw_value: str,
        *,
        expected_environment: Environment | None = None,
    ) -> SecretPayload:
        import json

        data = json.loads(raw_value)

        def _first_present(*keys: str) -> str | None:
            for key in keys:
                if key in data and data[key] is not None:
                    value = data[key]
                    if isinstance(value, str):
                        value = value.strip()
                    return value
            return None

        key_id = _first_present("key_id", "keyId", "keyID", "api_key", "apiKey")
        if not key_id:
            raise ValueError("sekret nie zawiera pola 'key_id' ani aliasów 'keyId' / 'api_key'")

        secret_value = _first_present("secret", "api_secret", "apiSecret", "secret_key", "secretKey") or ""
        passphrase = _first_present("passphrase", "api_passphrase", "apiPassphrase")

        permissions = data.get("permissions")
        if not permissions and "scopes" in data:
            permissions = data.get("scopes")

        environment_value = _first_present("environment", "env")

        environment: Environment | None = None
        if environment_value not in (None, ""):
            if isinstance(environment_value, Environment):
                environment = environment_value
            else:
                try:
                    environment = Environment(str(environment_value).lower())
                except ValueError as exc:  # pragma: no cover - walidacja formatu
                    raise ValueError(
                        f"nieobsługiwane środowisko w sekrecie: {environment_value}"
                    ) from exc

        if environment is None:
            if expected_environment is None:
                raise ValueError("sekret nie zawiera pola 'environment'")
            environment = expected_environment

        return SecretPayload(
            key_id=str(key_id),
            secret=str(secret_value),
            passphrase=str(passphrase) if passphrase is not None else None,
            permissions=tuple(
                str(permission).lower() for permission in (permissions or ()) if permission is not None
            ),
            environment=environment,
        )


__all__ = [
    "SecretStorage",
    "SecretStorageError",
    "SecretPayload",
    "SecretManager",
]
