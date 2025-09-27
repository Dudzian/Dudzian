"""Magazyn sekretów oparty o zaszyfrowany plik dla środowisk headless."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict

from bot_core.security.base import SecretStorage, SecretStorageError

try:  # pragma: no cover - zależne od środowiska CI
    from cryptography.fernet import Fernet, InvalidToken
except ImportError as exc:  # pragma: no cover - import zależy od opcjonalnej zależności
    _FERNET_IMPORT_ERROR: Exception | None = exc
    Fernet = None  # type: ignore[assignment]
    InvalidToken = Exception  # type: ignore[assignment]
else:
    _FERNET_IMPORT_ERROR = None


def _derive_key(passphrase: str, salt: bytes, *, iterations: int = 390_000) -> bytes:
    """Wyprowadza klucz symetryczny z hasła użytkownika."""

    import hashlib

    key = hashlib.pbkdf2_hmac("sha256", passphrase.encode("utf-8"), salt, iterations, dklen=32)
    return base64.urlsafe_b64encode(key)


class EncryptedFileSecretStorage(SecretStorage):
    """Przechowuje sekrety w pliku zaszyfrowanym kluczem pochodnym od hasła."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        passphrase: str,
        *,
        iterations: int = 390_000,
    ) -> None:
        if _FERNET_IMPORT_ERROR is not None:
            raise SecretStorageError(
                "Biblioteka 'cryptography' jest wymagana do korzystania z szyfrowanego magazynu "
                "plikowego. Zainstaluj ją poleceniem 'pip install cryptography'."
            ) from _FERNET_IMPORT_ERROR

        self._path = Path(path).expanduser().resolve()
        self._passphrase = passphrase
        self._iterations = iterations
        self._data: Dict[str, str] = {}
        self._salt: bytes | None = None
        self._fernet: Fernet | None = None

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load_or_initialize()

    # ------------------------------------------------------------------
    # Implementacja interfejsu ``SecretStorage``
    # ------------------------------------------------------------------
    def get_secret(self, key: str) -> str | None:
        return self._data.get(key)

    def set_secret(self, key: str, value: str) -> None:
        self._data[key] = value
        self._persist()

    def delete_secret(self, key: str) -> None:
        if key in self._data:
            del self._data[key]
            self._persist()

    # ------------------------------------------------------------------
    # Metody pomocnicze
    # ------------------------------------------------------------------
    def _ensure_crypto(self) -> Fernet:
        if self._fernet is None or self._salt is None:
            raise SecretStorageError("Magazyn plikowy nie został poprawnie zainicjalizowany.")
        return self._fernet

    def _load_or_initialize(self) -> None:
        if not self._path.exists():
            self._salt = os.urandom(16)
            key = _derive_key(self._passphrase, self._salt, iterations=self._iterations)
            self._fernet = Fernet(key)
            self._data = {}
            self._persist()
            return

        with self._path.open("r", encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except json.JSONDecodeError as exc:
                raise SecretStorageError(
                    "Plik magazynu sekretów jest uszkodzony. Usuń go ręcznie i zapisz sekrety ponownie."
                ) from exc

        try:
            salt_b64 = payload["salt"]
            ciphertext_b64 = payload["ciphertext"]
        except KeyError as exc:
            raise SecretStorageError(
                "Plik magazynu sekretów ma niepoprawną strukturę. Usuń go i zainicjalizuj ponownie."
            ) from exc

        try:
            self._salt = base64.b64decode(salt_b64)
            ciphertext = base64.b64decode(ciphertext_b64)
        except (TypeError, ValueError) as exc:
            raise SecretStorageError(
                "Nie udało się odczytać danych z pliku magazynu sekretów."
            ) from exc

        key = _derive_key(self._passphrase, self._salt, iterations=self._iterations)
        self._fernet = Fernet(key)

        try:
            decrypted = self._fernet.decrypt(ciphertext)
        except InvalidToken as exc:
            raise SecretStorageError(
                "Nieprawidłowe hasło lub uszkodzone dane magazynu sekretów."
            ) from exc

        try:
            raw_dict = json.loads(decrypted.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SecretStorageError(
                "Nie udało się zdeserializować danych z magazynu sekretów."
            ) from exc

        if not isinstance(raw_dict, dict):
            raise SecretStorageError("Oczekiwano słownika z parami klucz/wartość w magazynie sekretów.")

        self._data = {str(k): str(v) for k, v in raw_dict.items()}

    def _persist(self) -> None:
        fernet = self._ensure_crypto()
        salt = self._salt
        if salt is None:
            raise SecretStorageError("Brak soli kryptograficznej w magazynie sekretów.")

        plaintext = json.dumps(self._data, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ciphertext = fernet.encrypt(plaintext)

        payload = {
            "salt": base64.b64encode(salt).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }

        with NamedTemporaryFile("w", dir=str(self._path.parent), delete=False, encoding="utf-8") as tmp:
            json.dump(payload, tmp, separators=(",", ":"))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, self._path)
        try:
            os.chmod(self._path, 0o600)
        except PermissionError:  # pragma: no cover - na niektórych systemach (np. Windows)
            pass


__all__ = ["EncryptedFileSecretStorage"]
