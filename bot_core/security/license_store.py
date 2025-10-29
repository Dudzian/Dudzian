"""Magazyn licencji zabezpieczony fingerprintem sprzętowym."""
from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from bot_core.security.hwid import HwIdProvider, HwIdProviderError


class LicenseStoreError(RuntimeError):
    """Błąd ogólny magazynu licencji."""


class LicenseStoreFingerprintError(LicenseStoreError):
    """Nie udało się ustalić fingerprintu do ochrony magazynu."""


class LicenseStoreDecryptionError(LicenseStoreError):
    """Magazyn licencji nie mógł zostać odszyfrowany (np. zmiana sprzętu)."""


@dataclass(slots=True)
class LicenseStoreDocument:
    """Dokument magazynu licencji wraz z metadanymi."""

    data: dict[str, Any]
    fingerprint_hash: str | None
    migrated: bool


class LicenseStore:
    """Zapewnia szyfrowany magazyn licencji oparty o fingerprint urządzenia."""

    CURRENT_VERSION = 2
    DEFAULT_PATH = Path("secrets/license_store.json")

    def __init__(
        self,
        path: str | os.PathLike[str] | None = None,
        *,
        hwid_provider: HwIdProvider | None = None,
        fingerprint_override: str | None = None,
        fingerprint_reader: Callable[[], str] | None = None,
    ) -> None:
        self._path = Path(path).expanduser() if path else self.DEFAULT_PATH
        if fingerprint_reader is not None:
            self._fingerprint_reader = fingerprint_reader
        elif fingerprint_override is not None:
            override = fingerprint_override.strip()
            if not override:
                raise LicenseStoreFingerprintError(
                    "Przekazany override fingerprintu jest pusty i nie może zostać użyty."
                )
            self._fingerprint_reader = lambda: override
        elif hwid_provider is not None:
            self._fingerprint_reader = hwid_provider.read
        else:
            self._fingerprint_reader = HwIdProvider().read

    # ------------------------------------------------------------------
    def load(self) -> LicenseStoreDocument:
        if not self._path.exists():
            return LicenseStoreDocument(data={"licenses": {}}, fingerprint_hash=None, migrated=False)

        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover - nieoczekiwane błędy IO
            raise LicenseStoreError(f"Nie udało się odczytać magazynu licencji: {exc}") from exc

        try:
            document = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LicenseStoreError("Plik magazynu licencji zawiera niepoprawny JSON.") from exc

        if isinstance(document, Mapping) and document.get("version") == self.CURRENT_VERSION:
            return self._decrypt_document(document)

        if not isinstance(document, Mapping):
            raise LicenseStoreError("Oczekiwano, że magazyn licencji będzie obiektem JSON.")

        cleaned: dict[str, Any] = {"licenses": {}}
        licenses = document.get("licenses")
        if isinstance(licenses, Mapping):
            cleaned["licenses"] = {str(key): value for key, value in licenses.items()}
        return LicenseStoreDocument(data=cleaned, fingerprint_hash=None, migrated=True)

    # ------------------------------------------------------------------
    def save(self, data: Mapping[str, Any]) -> LicenseStoreDocument:
        fingerprint = self._read_fingerprint()
        fingerprint_hash = _hash_fingerprint(fingerprint)
        plaintext = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        nonce = os.urandom(12)
        cipher = AESGCM(_derive_key(fingerprint))
        ciphertext = cipher.encrypt(nonce, plaintext, fingerprint_hash.encode("utf-8"))
        envelope = {
            "version": self.CURRENT_VERSION,
            "fingerprint_hash": fingerprint_hash,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        self._path.write_text(payload, encoding="utf-8")
        return LicenseStoreDocument(data=dict(data), fingerprint_hash=fingerprint_hash, migrated=False)

    # ------------------------------------------------------------------
    def _read_fingerprint(self) -> str:
        try:
            value = self._fingerprint_reader()
        except HwIdProviderError as exc:
            raise LicenseStoreFingerprintError(str(exc)) from exc
        except Exception as exc:  # pragma: no cover - defensywne logowanie
            raise LicenseStoreFingerprintError("Nie udało się pobrać fingerprintu urządzenia.") from exc
        cleaned = str(value).strip()
        if not cleaned:
            raise LicenseStoreFingerprintError("Fingerprint urządzenia jest pusty.")
        return cleaned

    # ------------------------------------------------------------------
    def _decrypt_document(self, document: Mapping[str, Any]) -> LicenseStoreDocument:
        fingerprint = self._read_fingerprint()
        fingerprint_hash = str(document.get("fingerprint_hash") or "").strip()
        nonce_b64 = document.get("nonce")
        ciphertext_b64 = document.get("ciphertext")
        if not isinstance(nonce_b64, str) or not isinstance(ciphertext_b64, str):
            raise LicenseStoreError("Dokument magazynu jest uszkodzony (brak nonce lub ciphertext).")
        try:
            nonce = base64.b64decode(nonce_b64.encode("ascii"))
            ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
        except Exception as exc:
            raise LicenseStoreError("Nie udało się zdekodować magazynu licencji (base64).") from exc
        cipher = AESGCM(_derive_key(fingerprint))
        try:
            plaintext = cipher.decrypt(nonce, ciphertext, fingerprint_hash.encode("utf-8"))
        except Exception as exc:
            raise LicenseStoreDecryptionError(
                "Nie udało się odszyfrować magazynu licencji – możliwa zmiana sprzętu."
            ) from exc
        try:
            payload = json.loads(plaintext.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise LicenseStoreError("Odszyfrowany magazyn licencji zawiera niepoprawny JSON.") from exc
        if not isinstance(payload, Mapping):
            raise LicenseStoreError("Odszyfrowany magazyn licencji ma niepoprawny format.")
        return LicenseStoreDocument(data=dict(payload), fingerprint_hash=fingerprint_hash or None, migrated=False)


def _derive_key(fingerprint: str) -> bytes:
    digest = fingerprint.encode("utf-8")
    return _sha256(digest)


def _hash_fingerprint(fingerprint: str) -> str:
    return _sha256(fingerprint.encode("utf-8")).hex()


def _sha256(payload: bytes) -> bytes:
    import hashlib

    return hashlib.sha256(payload).digest()


__all__ = [
    "LicenseStore",
    "LicenseStoreDocument",
    "LicenseStoreError",
    "LicenseStoreDecryptionError",
    "LicenseStoreFingerprintError",
]
