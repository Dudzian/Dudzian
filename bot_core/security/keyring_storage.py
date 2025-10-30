from __future__ import annotations

import base64
import hashlib
import hmac
import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from bot_core.security.base import SecretStorage, SecretStorageError
from bot_core.security.hwid import HwIdProvider, HwIdProviderError


class KeyringSecretStorage(SecretStorage):
    """Magazyn sekretów wykorzystujący natywny keychain oraz lokalny fingerprint."""

    MASTER_KEY_SLOT = "__master_key_v2"
    MASTER_KEY_BYTES = 32
    SECRET_VERSION = 2
    INDEX_VERSION = 1
    DEFAULT_INDEX_PATH = Path("var/security/secret_index.json")

    def __init__(
        self,
        *,
        service_name: str = "dudzian.trading",
        hwid_provider: HwIdProvider | None = None,
        index_path: str | Path | None = None,
    ) -> None:
        try:
            import keyring  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - zależne od środowiska CI
            raise SecretStorageError(
                "Biblioteka 'keyring' nie jest dostępna. Zainstaluj ją, aby korzystać z natywnego "
                "przechowywania sekretów (pip install keyring)."
            ) from exc

        self._keyring = keyring
        self._backend = self._ensure_native_backend(keyring)
        self._service_name = service_name
        self._hwid_provider = hwid_provider or HwIdProvider()
        self._index_path = Path(index_path).expanduser() if index_path else self.DEFAULT_INDEX_PATH

    # ------------------------------------------------------------------
    # API SecretStorage
    # ------------------------------------------------------------------
    def get_secret(self, key: str) -> Optional[str]:
        encrypted = self._keyring.get_password(self._service_name, key)
        if encrypted is None:
            return None
        master, hwid_digest = self._load_master_key()
        try:
            return self._decrypt_value(encrypted, storage_key=key, master=master, hwid_digest=hwid_digest)
        except SecretStorageError:
            raise
        except Exception as exc:  # pragma: no cover - silne logowanie
            raise SecretStorageError(f"Nie udało się odszyfrować sekretu '{key}'.") from exc

    def set_secret(self, key: str, value: str) -> None:
        master, hwid_digest = self._load_master_key()
        encrypted = self._encrypt_value(value, storage_key=key, master=master, hwid_digest=hwid_digest)
        result = self._keyring.set_password(self._service_name, key, encrypted)
        if result is not None:  # pragma: no cover - większość backendów zwraca None
            raise SecretStorageError(f"Nie udało się zapisać sekretu '{key}'.")
        self._register_key(key, hwid_digest)

    def delete_secret(self, key: str) -> None:
        try:
            self._keyring.delete_password(self._service_name, key)
        except self._keyring.errors.PasswordDeleteError:
            return
        self._unregister_key(key)

    # ------------------------------------------------------------------
    # Rotacja klucza głównego
    # ------------------------------------------------------------------
    def rotate_master_key(self) -> None:
        index = self._load_index()
        stored_keys = list(index.get("keys", {}).keys())
        old_master, hwid_digest = self._load_master_key()
        new_master = os.urandom(self.MASTER_KEY_BYTES)
        self._store_master_key(new_master, hwid_digest)

        for storage_key in stored_keys:
            encrypted = self._keyring.get_password(self._service_name, storage_key)
            if encrypted is None:
                self._unregister_key(storage_key)
                continue
            plaintext = self._decrypt_value(encrypted, storage_key=storage_key, master=old_master, hwid_digest=hwid_digest)
            rotated_value = self._encrypt_value(plaintext, storage_key=storage_key, master=new_master, hwid_digest=hwid_digest)
            result = self._keyring.set_password(self._service_name, storage_key, rotated_value)
            if result is not None:  # pragma: no cover - defensywne
                raise SecretStorageError(f"Nie udało się zapisać sekretu '{storage_key}' podczas rotacji.")
            self._register_key(storage_key, hwid_digest)

    # ------------------------------------------------------------------
    # Obsługa klucza głównego i indeksu
    # ------------------------------------------------------------------
    def _read_hwid_digest(self) -> str:
        try:
            fingerprint = self._hwid_provider.read()
        except HwIdProviderError as exc:
            raise SecretStorageError("Nie udało się pobrać fingerprintu urządzenia.") from exc
        normalized = fingerprint.strip()
        if not normalized:
            raise SecretStorageError("Fingerprint urządzenia jest pusty.")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _load_master_key(self) -> tuple[bytes, str]:
        record = self._keyring.get_password(self._service_name, self.MASTER_KEY_SLOT)
        hwid_digest = self._read_hwid_digest()
        if record is None:
            master = os.urandom(self.MASTER_KEY_BYTES)
            self._store_master_key(master, hwid_digest)
            return master, hwid_digest

        try:
            document = json.loads(record)
        except json.JSONDecodeError as exc:
            raise SecretStorageError("Uszkodzony rekord klucza głównego w keychainie.") from exc

        stored_digest = document.get("hwid_digest")
        if stored_digest and stored_digest != hwid_digest:
            raise SecretStorageError(
                "Magazyn sekretów jest powiązany z innym urządzeniem (fingerprint mismatch)."
            )

        key_b64 = document.get("key_b64")
        if not isinstance(key_b64, str):
            raise SecretStorageError("Rekord klucza głównego nie zawiera danych klucza.")
        try:
            master = base64.b64decode(key_b64.encode("ascii"))
        except Exception as exc:
            raise SecretStorageError("Nie można zdekodować klucza głównego.") from exc
        if len(master) != self.MASTER_KEY_BYTES:
            raise SecretStorageError("Klucz główny magazynu sekretów ma niepoprawną długość.")

        return master, hwid_digest

    def _store_master_key(self, master: bytes, hwid_digest: str) -> None:
        payload = {
            "version": self.SECRET_VERSION,
            "key_b64": base64.b64encode(master).decode("ascii"),
            "hwid_digest": hwid_digest,
            "rotated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        result = self._keyring.set_password(self._service_name, self.MASTER_KEY_SLOT, json.dumps(payload))
        if result is not None:  # pragma: no cover - defensywne
            raise SecretStorageError("Nie udało się zapisać klucza głównego w keychainie.")

    def _load_index(self) -> MutableMapping[str, Any]:
        path = self._index_path
        if not path.exists():
            return {"version": self.INDEX_VERSION, "keys": {}}
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"version": self.INDEX_VERSION, "keys": {}}
        except OSError as exc:
            raise SecretStorageError(f"Nie udało się odczytać indeksu magazynu sekretów: {exc}") from exc

        keys = document.get("keys")
        if not isinstance(keys, Mapping):
            return {"version": self.INDEX_VERSION, "keys": {}}
        canonical: MutableMapping[str, Any] = {}
        for storage_key, metadata in keys.items():
            canonical[str(storage_key)] = metadata if isinstance(metadata, Mapping) else {}
        return {"version": self.INDEX_VERSION, "keys": canonical}

    def _save_index(self, document: Mapping[str, Any]) -> None:
        path = self._index_path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)

    def _register_key(self, storage_key: str, hwid_digest: str) -> None:
        index = self._load_index()
        keys = index.setdefault("keys", {})
        keys[storage_key] = {
            "hwid_digest": hwid_digest,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self._save_index(index)

    def _unregister_key(self, storage_key: str) -> None:
        index = self._load_index()
        keys = index.setdefault("keys", {})
        if storage_key in keys:
            del keys[storage_key]
            self._save_index(index)

    # ------------------------------------------------------------------
    # Szyfrowanie/dekryptowanie wartości
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Wybór natywnego backendu keyring
    # ------------------------------------------------------------------
    def _platform_identifier(self) -> str:
        if sys.platform.startswith("win"):
            return "windows"
        if sys.platform == "darwin":
            return "macos"
        return "linux"

    def _ensure_native_backend(self, keyring_module: Any) -> Any:
        backend = keyring_module.get_keyring()
        platform_id = self._platform_identifier()

        if self._is_native_backend(backend, platform_id):
            return backend

        try:
            native_backend = self._load_native_backend(platform_id)
        except ImportError as exc:  # pragma: no cover - zależne od środowiska
            raise SecretStorageError(
                "Brak natywnego backendu keyring dla tej platformy. Zainstaluj odpowiednie rozszerzenia "
                "(SecretService na Linux, macOS Keychain, Windows Credential Manager)."
            ) from exc

        keyring_module.set_keyring(native_backend)
        return keyring_module.get_keyring()

    def _load_native_backend(self, platform_id: str) -> Any:
        module_name, class_name = {
            "windows": ("keyring.backends.Windows", "WinVaultKeyring"),
            "macos": ("keyring.backends.macOS", "Keyring"),
            "linux": ("keyring.backends.SecretService", "SecretServiceKeyring"),
        }[platform_id]
        module = importlib.import_module(module_name)
        backend_cls = getattr(module, class_name)
        return backend_cls()

    def _is_native_backend(self, backend: Any, platform_id: str) -> bool:
        if backend is None:
            return False
        module_name, class_name = {
            "windows": ("keyring.backends.Windows", "WinVaultKeyring"),
            "macos": ("keyring.backends.macOS", "Keyring"),
            "linux": ("keyring.backends.SecretService", "SecretServiceKeyring"),
        }[platform_id]

        # Prosty przypadek: backend ma oczekiwany moduł i klasę
        if backend.__class__.__module__ == module_name and backend.__class__.__name__ == class_name:
            return True

        # Chained backend – sprawdź pod-backendy
        nested = getattr(backend, "_backends", None) or getattr(backend, "backends", None)
        if nested:
            for candidate in nested:
                if candidate.__class__.__module__ == module_name and candidate.__class__.__name__ == class_name:
                    return True
        return False

    def _derive_data_key(self, master: bytes, hwid_digest: str) -> bytes:
        return hmac.new(master, hwid_digest.encode("ascii"), hashlib.sha256).digest()

    def _encrypt_value(self, value: str, *, storage_key: str, master: bytes, hwid_digest: str) -> str:
        aes_key = self._derive_data_key(master, hwid_digest)
        nonce = os.urandom(12)
        cipher = AESGCM(aes_key)
        ciphertext = cipher.encrypt(nonce, value.encode("utf-8"), storage_key.encode("utf-8"))
        payload = {
            "version": self.SECRET_VERSION,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "hwid_digest": hwid_digest,
        }
        return json.dumps(payload)

    def _decrypt_value(self, document: str, *, storage_key: str, master: bytes, hwid_digest: str) -> str:
        try:
            payload = json.loads(document)
        except json.JSONDecodeError as exc:
            raise SecretStorageError(f"Sekret '{storage_key}' ma niepoprawny format.") from exc

        stored_digest = payload.get("hwid_digest")
        if stored_digest and stored_digest != hwid_digest:
            raise SecretStorageError(
                "Sekret został zapisany na innym urządzeniu – fingerprint nie zgadza się z magazynem."
            )

        nonce_b64 = payload.get("nonce")
        ciphertext_b64 = payload.get("ciphertext")
        if not isinstance(nonce_b64, str) or not isinstance(ciphertext_b64, str):
            raise SecretStorageError(f"Sekret '{storage_key}' nie zawiera danych szyfru.")

        try:
            nonce = base64.b64decode(nonce_b64.encode("ascii"))
            ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
        except Exception as exc:
            raise SecretStorageError(f"Sekret '{storage_key}' ma uszkodzone dane szyfru.") from exc

        aes_key = self._derive_data_key(master, hwid_digest)
        cipher = AESGCM(aes_key)
        try:
            plaintext = cipher.decrypt(nonce, ciphertext, storage_key.encode("utf-8"))
        except Exception as exc:
            raise SecretStorageError(f"Nie udało się odszyfrować sekretu '{storage_key}'.") from exc
        return plaintext.decode("utf-8")


__all__ = ["KeyringSecretStorage"]
