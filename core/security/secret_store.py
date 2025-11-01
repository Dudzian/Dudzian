"""Bezpieczny magazyn kluczy API wykorzystywany w kreatorze onboardingu."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable

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
        legacy_path: str | Path | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        self._legacy_path = Path(legacy_path) if legacy_path is not None else _default_path()
        self._legacy_path = self._legacy_path.expanduser().resolve()
        self._legacy_path.parent.mkdir(parents=True, exist_ok=True)

        if storage is not None:
            self._storage = storage
        else:
            derived_index = index_path or _default_index_path()
            self._storage = KeyringSecretStorage(
                service_name=service_name or "dudzian.trading.desktop",
                index_path=derived_index,
            )

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

        if not self._legacy_path.exists():
            return

        legacy_payload = self._read_legacy_payload()
        if not legacy_payload:
            self._archive_legacy_file()
            return

        try:
            registered_keys = self._storage.list_registered_keys()
        except SecretStorageError as exc:
            raise SecretStoreError(str(exc)) from exc

        existing = {
            self._extract_exchange_id(key)
            for key in registered_keys
            if self._extract_exchange_id(key)
        }

        for exchange_id, payload in legacy_payload.items():
            if exchange_id in existing:
                LOGGER.info(
                    "Pomijam migrację kluczy API dla '%s' – wpis istnieje już w keychainie.",
                    exchange_id,
                )
                continue
            payload.setdefault("version", 1)
            payload.setdefault("migrated_from", "plaintext_v1")
            try:
                self._storage.set_secret(self._storage_key(exchange_id), json.dumps(payload))
            except SecretStorageError as exc:
                raise SecretStoreError(
                    "Migracja starego magazynu kluczy API nie powiodła się. Usuń plik \"api_credentials.json\" lub "
                    "napraw jego zawartość, a następnie spróbuj ponownie."
                ) from exc

        self._archive_legacy_file()

    def _read_legacy_payload(self) -> Dict[str, Dict[str, str | None]]:
        try:
            raw = self._legacy_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SecretStoreError(f"Nie można odczytać magazynu sekretów: {exc}") from exc

        if not raw.strip():
            return {}

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SecretStoreError("Magazyn sekretów jest uszkodzony") from exc

        if not isinstance(data, dict):
            raise SecretStoreError("Nieprawidłowy format magazynu sekretów")

        result: Dict[str, Dict[str, str | None]] = {}
        for key, value in data.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            normalized = key.strip().lower()
            if not normalized:
                continue
            api_key = str(value.get("api_key", "")).strip()
            api_secret = str(value.get("api_secret", "")).strip()
            api_passphrase = value.get("api_passphrase")
            if api_passphrase is not None:
                api_passphrase = str(api_passphrase).strip() or None
            if not api_key or not api_secret:
                continue
            result[normalized] = {
                "api_key": api_key,
                "api_secret": api_secret,
                "api_passphrase": api_passphrase,
            }
        return result

    def _archive_legacy_file(self) -> None:
        try:
            backup_path = self._legacy_path.with_name(self._legacy_path.stem + ".legacy")
            if backup_path.exists():
                backup_path.unlink()
            self._legacy_path.replace(backup_path)
        except FileNotFoundError:  # pragma: no cover - plik mógł zostać usunięty równolegle
            return
        except OSError as exc:  # pragma: no cover - ostrzegawcze logowanie
            LOGGER.warning("Nie udało się zarchiwizować starego magazynu sekretów: %s", exc)

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


def _default_path() -> Path:
    config_dir = Path.home() / ".kryptolowca"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "api_credentials.json"


def _default_index_path() -> Path:
    config_dir = Path.home() / ".kryptolowca"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "secret_index.json"


__all__ = ["ExchangeCredentials", "SecretStore", "SecretStoreError"]
