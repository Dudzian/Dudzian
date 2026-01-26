"""Native/obfuscated helpers for license secret cryptography."""

from __future__ import annotations

import base64
import importlib
import logging
import os
from dataclasses import dataclass
from typing import Callable, Mapping, Protocol

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

LOGGER = logging.getLogger(__name__)


class _CryptoBackend(Protocol):
    """Interface implemented by native (binary) crypto helpers."""

    def current_hwid_digest(self, fingerprint: str) -> str:  # pragma: no cover - interface
        ...

    def derive_encryption_key(self, fingerprint: str, salt: bytes) -> bytes:  # pragma: no cover - interface
        ...

    def encrypt_license_secret(
        self,
        secret: bytes,
        fingerprint: str,
        *,
        file_version: int,
    ) -> Mapping[str, object]:  # pragma: no cover - interface
        ...

    def decrypt_license_secret(
        self,
        document: Mapping[str, object],
        fingerprint: str,
        *,
        file_version: int,
    ) -> bytes:  # pragma: no cover - interface
        ...


@dataclass(slots=True)
class _ModuleBackend:
    """Adapter wrapping dynamically loaded binary module."""

    module: object

    def current_hwid_digest(self, fingerprint: str) -> str:
        return self.module.current_hwid_digest(fingerprint)

    def derive_encryption_key(self, fingerprint: str, salt: bytes) -> bytes:
        return self.module.derive_encryption_key(fingerprint, salt)

    def encrypt_license_secret(
        self,
        secret: bytes,
        fingerprint: str,
        *,
        file_version: int,
    ) -> Mapping[str, object]:
        return self.module.encrypt_license_secret(  # type: ignore[return-value]
            secret,
            fingerprint,
            file_version=file_version,
        )

    def decrypt_license_secret(
        self,
        document: Mapping[str, object],
        fingerprint: str,
        *,
        file_version: int,
    ) -> bytes:
        return self.module.decrypt_license_secret(  # type: ignore[return-value]
            document,
            fingerprint,
            file_version=file_version,
        )


class _PythonCryptoBackend:
    """Pure Python fallback implementation used when binary helper is unavailable."""

    @staticmethod
    def current_hwid_digest(fingerprint: str) -> str:
        import hashlib

        normalized = fingerprint.strip().upper()
        if not normalized:
            raise ValueError("Fingerprint nie może być pusty")
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalized(fingerprint: str) -> str:
        normalized = fingerprint.strip().upper()
        if not normalized:
            raise ValueError("Fingerprint nie może być pusty")
        return normalized

    def derive_encryption_key(self, fingerprint: str, salt: bytes) -> bytes:
        import hashlib
        import hmac

        normalized = self._normalized(fingerprint)
        return hmac.new(normalized.encode("utf-8"), salt, hashlib.sha256).digest()

    def encrypt_license_secret(
        self,
        secret: bytes,
        fingerprint: str,
        *,
        file_version: int,
    ) -> Mapping[str, object]:
        normalized = self._normalized(fingerprint)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        key = self.derive_encryption_key(normalized, salt)
        cipher = AESGCM(key)
        ciphertext = cipher.encrypt(nonce, secret, normalized.encode("utf-8"))
        return {
            "version": file_version,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
            "hwid_digest": self.current_hwid_digest(normalized),
            "length": len(secret),
        }

    def decrypt_license_secret(
        self,
        document: Mapping[str, object],
        fingerprint: str,
        *,
        file_version: int,
    ) -> bytes:
        import hashlib

        version = document.get("version")
        if version != file_version:
            raise ValueError("Nieobsługiwana wersja zaszyfrowanego sekretu")

        try:
            salt_b64 = document["salt"]
            nonce_b64 = document["nonce"]
            ciphertext_b64 = document["ciphertext"]
        except KeyError as exc:  # pragma: no cover - defensive path
            raise ValueError("Dokument sekretu licencji jest uszkodzony") from exc

        salt = base64.b64decode(str(salt_b64).encode("ascii"))
        nonce = base64.b64decode(str(nonce_b64).encode("ascii"))
        ciphertext = base64.b64decode(str(ciphertext_b64).encode("ascii"))

        normalized = self._normalized(fingerprint)
        expected_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        stored_digest = document.get("hwid_digest")
        if isinstance(stored_digest, str) and stored_digest and stored_digest != expected_digest:
            raise ValueError("Sekret licencji zapisano dla innego urządzenia")

        key = self.derive_encryption_key(normalized, salt)
        cipher = AESGCM(key)
        return cipher.decrypt(nonce, ciphertext, normalized.encode("utf-8"))


def _iter_backend_candidates() -> list[str]:
    env_override = os.environ.get("BOT_CORE_SECURITY_NATIVE_MODULE")
    candidates: list[str] = []
    if env_override:
        candidates.extend([entry for entry in env_override.split(":") if entry])
    candidates.append("bot_core.security._native_security")
    return candidates


def _load_backend() -> _CryptoBackend:
    for candidate in _iter_backend_candidates():
        try:
            module = importlib.import_module(candidate)
        except ModuleNotFoundError:
            continue
        missing = [
            attr
            for attr in (
                "current_hwid_digest",
                "derive_encryption_key",
                "encrypt_license_secret",
                "decrypt_license_secret",
            )
            if not hasattr(module, attr)
        ]
        if missing:
            LOGGER.warning(
                "Moduł %s nie implementuje wymaganych metod: %s", candidate, ", ".join(missing)
            )
            continue
        LOGGER.info("Załadowano natywny moduł bezpieczeństwa: %s", candidate)
        return _ModuleBackend(module)

    LOGGER.debug("Używam fallbacku Pythona dla kryptografii licencji")
    return _PythonCryptoBackend()


_BACKEND: _CryptoBackend | None = None


def _backend() -> _CryptoBackend:
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = _load_backend()
    return _BACKEND


def reset_backend(loader: Callable[[], _CryptoBackend] | None = None) -> None:
    """Resetuje backend (używane w testach)."""

    global _BACKEND
    _BACKEND = loader() if loader is not None else None


def current_hwid_digest(fingerprint: str) -> str:
    return _backend().current_hwid_digest(fingerprint)


def derive_encryption_key(fingerprint: str, salt: bytes) -> bytes:
    return _backend().derive_encryption_key(fingerprint, salt)


def encrypt_license_secret(
    secret: bytes,
    fingerprint: str,
    *,
    file_version: int,
) -> Mapping[str, object]:
    return _backend().encrypt_license_secret(secret, fingerprint, file_version=file_version)


def decrypt_license_secret(
    document: Mapping[str, object],
    fingerprint: str,
    *,
    file_version: int,
) -> bytes:
    return _backend().decrypt_license_secret(document, fingerprint, file_version=file_version)


__all__ = [
    "current_hwid_digest",
    "derive_encryption_key",
    "decrypt_license_secret",
    "encrypt_license_secret",
    "reset_backend",
]

