"""Bezpieczny magazyn kluczy API wykorzystywany w kreatorze onboardingu."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from threading import RLock
from typing import Iterable

from bot_core.security.base import SecretStorageError
from bot_core.security.keyring_storage import KeyringSecretStorage


LOGGER = logging.getLogger(__name__)


class SecretStoreError(RuntimeError):
    """Wyjątek zgłaszany w przypadku błędów operacji na magazynie sekretów."""


@dataclass(frozen=True)
class ExchangeCredentials:
    """Dane uwierzytelniające pojedynczej giełdy."""

    exchange: str
    api_key: str
    api_secret: str
    api_passphrase: str | None = None

    def normalized_exchange(self) -> str:
        return self.exchange.strip().lower()


class SecretStore:
    """Magazyn sekretów korzystający z natywnego keychaina i fingerprintu."""

    SECURITY_DETAILS_TOKEN = "onboarding.strategy.credentials.secured"
    _STORAGE_NAMESPACE = "desktop.exchange"

    def __init__(
        self,
        *,
        storage: KeyringSecretStorage | None = None,
        service_name: str | None = None,
        deprecated_path: str | Path | None = None,
        index_path: str | Path | None = None,
        data_dir: str | Path | None = None,
    ) -> None:
        base_dir = Path(data_dir).expanduser() if data_dir is not None else _default_data_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        if storage is not None:
            self._storage = storage
        else:
            derived_index = Path(index_path).expanduser() if index_path is not None else base_dir / "secret_index.json"
            self._storage = KeyringSecretStorage(
                service_name=service_name or "dudzian.trading.desktop",
                index_path=derived_index,
            )

        if deprecated_path:
            self._deprecated_hint_path = Path(deprecated_path).expanduser()
        else:
            self._deprecated_hint_path = _default_deprecated_path()

        self._lock = RLock()
        self._migration_checked = False

    # ------------------------------------------------------------------
    def save_exchange_credentials(self, credentials: ExchangeCredentials) -> None:
        """Zapisuje lub aktualizuje dane API w natywnym keychainie."""

        exchange_id = credentials.normalized_exchange()
        if not exchange_id:
            raise SecretStoreError("Brak identyfikatora giełdy")

        payload = {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "api_key": credentials.api_key.strip(),
            "api_secret": credentials.api_secret.strip(),
            "api_passphrase": (credentials.api_passphrase or "").strip() or None,
        }

        if not payload["api_key"] or not payload["api_secret"]:
            raise SecretStoreError("Klucz API i sekret są wymagane")

        storage_key = self._storage_key(exchange_id)

        with self._lock:
            self._ensure_migrated_locked()
            try:
                self._storage.set_secret(storage_key, json.dumps(payload))
            except SecretStorageError as exc:
                raise SecretStoreError(str(exc)) from exc

    def load_exchange_credentials(self, exchange: str) -> ExchangeCredentials:
        """Zwraca zapisane dane API lub zgłasza błąd, gdy ich brak."""

        exchange_id = self._normalize_exchange(exchange)
        if not exchange_id:
            raise SecretStoreError("Brak identyfikatora giełdy")

        storage_key = self._storage_key(exchange_id)

        with self._lock:
            self._ensure_migrated_locked()
            try:
                raw = self._storage.get_secret(storage_key)
            except SecretStorageError as exc:
                raise SecretStoreError(str(exc)) from exc

        if raw is None:
            raise SecretStoreError(f"Brak zapisanych kluczy API dla giełdy '{exchange_id}'.")

        return self._deserialize_credentials(exchange_id, raw)

    def list_exchanges(self) -> Iterable[str]:
        with self._lock:
            self._ensure_migrated_locked()
            try:
                keys = self._storage.list_registered_keys()
            except SecretStorageError as exc:
                raise SecretStoreError(str(exc)) from exc

        prefix = f"{self._STORAGE_NAMESPACE}:"
        exchanges = [
            key[len(prefix) :]
            for key in keys
            if key.startswith(prefix) and key[len(prefix) :]
        ]
        return tuple(dict.fromkeys(exchanges))

    def rotate_master_key(self) -> None:
        """Wymusza rotację klucza głównego magazynu."""

        with self._lock:
            self._ensure_migrated_locked()
            try:
                self._storage.rotate_master_key()
            except SecretStorageError as exc:
                raise SecretStoreError(str(exc)) from exc

    def security_details_token(self) -> str:
        """Zwraca identyfikator komunikatu opisującego zabezpieczenia magazynu."""

        return self.SECURITY_DETAILS_TOKEN

    # ------------------------------------------------------------------
    def _ensure_migrated_locked(self) -> None:
        if self._migration_checked:
            return
        self._migration_checked = True

        if self._deprecated_hint_path and self._deprecated_hint_path.exists():
            raise SecretStoreError(
                "Wykryto archiwalny magazyn kluczy API w {path}. Migracja automatyczna została usunięta – "
                "uruchom narzędzie z pakietu 'dudzian-migrate' opisane w docs/migrations/2024-stage5-storage-removal.md "
                "i usuń plik przed dalszym korzystaniem z aplikacji."
                .format(path=self._deprecated_hint_path)
            )

    def _storage_key(self, exchange_id: str) -> str:
        return f"{self._STORAGE_NAMESPACE}:{exchange_id}"

    def _extract_exchange_id(self, storage_key: str | None) -> str | None:
        if not storage_key:
            return None
        prefix = f"{self._STORAGE_NAMESPACE}:"
        if storage_key.startswith(prefix) and len(storage_key) > len(prefix):
            return storage_key[len(prefix) :]
        return None

    def _deserialize_credentials(self, exchange_id: str, raw: str) -> ExchangeCredentials:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretStoreError("Zapisane klucze API mają nieprawidłowy format.") from exc

        api_key = str(payload.get("api_key", "")).strip()
        api_secret = str(payload.get("api_secret", "")).strip()
        api_passphrase = payload.get("api_passphrase")
        if api_passphrase is not None:
            api_passphrase = str(api_passphrase).strip() or None

        if not api_key or not api_secret:
            raise SecretStoreError("Zapisane dane API są niekompletne – zapisz je ponownie.")

        return ExchangeCredentials(
            exchange=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            api_passphrase=api_passphrase,
        )

    def _normalize_exchange(self, exchange: str) -> str:
        return str(exchange or "").strip().lower()


def _default_data_dir() -> Path:
    override = os.environ.get("DUDZIAN_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".dudzian"


def _default_deprecated_path() -> Path:
    return (Path.home() / ".kryptolowca" / "api_credentials.json").expanduser()


__all__ = ["ExchangeCredentials", "SecretStore", "SecretStoreError"]
