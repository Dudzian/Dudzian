# managers/security_manager.py
# -*- coding: utf-8 -*-
"""
Security Manager — lokalne szyfrowanie (domyślnie) + opcjonalny AWS Secrets Manager.

✅ Domyślnie: PBKDF2-HMAC-SHA256 (480k iteracji) + AES-256-GCM, sól z pliku salt.bin
✅ Zgodność wsteczna: odczyt legacy plików zaszyfrowanych Fernet (PBKDF2 100k, stała sól)
✅ API zgodne z GUI:
    - save_encrypted_keys(keys: Dict[str, Any], password: str) -> None
    - load_encrypted_keys(password: str) -> Dict[str, Any]
✅ Walidacja kluczy API: validate_api_key()

Opcjonalnie:
- backend="aws": zapis/odczyt w AWS Secrets Manager (SecretString), wymaga regionu (parametr lub zmienna środowiskowa)
"""

from __future__ import annotations

import os
import json
import base64
import logging
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from bot_core.alerts import AlertSeverity, emit_alert  # type: ignore

from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.fernet import Fernet, InvalidToken as FernetInvalidToken  # legacy fallback

logger = logging.getLogger(__name__)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

warnings.warn(
    "SecurityManager jest przestarzały – użyj bot_core.security.file_storage.EncryptedFileSecretStorage",
    DeprecationWarning,
    stacklevel=2,
)

audit_logger = logging.getLogger(f"{__name__}.audit")
if not audit_logger.handlers:
    _audit_handler = logging.StreamHandler()
    _audit_handler.setFormatter(
        logging.Formatter('[%(asctime)s] SECURITY-AUDIT %(levelname)s: %(message)s')
    )
    audit_logger.addHandler(_audit_handler)
audit_logger.setLevel(logging.INFO)


class SecurityError(Exception):
    """Błąd operacji bezpieczeństwa (szyfrowanie/odszyfrowywanie)."""
    pass


class SecurityManager:
    """
    Manager szyfrowania kluczy API.

    Parameters
    ----------
    key_file : str | Path
        Ścieżka do pliku z zaszyfrowanymi kluczami (np. 'api_keys.enc').
    salt_file : str | Path | None
        Ścieżka do pliku z solą dla PBKDF2 (np. 'salt.bin'). Jeśli None — użyje domyślnej nazwy obok key_file.
    backend : Literal["local","aws"]
        'local' (domyślnie) używa AES-GCM lokalnie; 'aws' używa AWS Secrets Manager.
    aws_secret_id : str | None
        Identyfikator sekretu w AWS Secrets Manager (wymagany dla backend="aws").
    aws_region_name : str | None
        Region AWS (np. "eu-central-1"). Wymagany dla backend="aws" jeśli nie skonfigurowany w środowisku.
    """

    # parametry KDF/AES
    _KDF_ITERS = 480_000
    _SALT_LEN = 16
    _NONCE_LEN = 12
    _LEGACY_SALT = b"trading_bot_salt_2025"   # dla kompatybilności Fernet (poprzednia wersja)
    _LEGACY_ITERS = 100_000

    def __init__(
        self,
        key_file: str | Path = "api_keys.enc",
        salt_file: Optional[str | Path] = None,
        backend: str = "local",
        aws_secret_id: Optional[str] = None,
        aws_region_name: Optional[str] = None,
    ):
        self.key_file = Path(key_file)
        self.salt_file = Path(salt_file) if salt_file else self.key_file.with_name("salt.bin")
        self.backend = (backend or "local").lower()
        self.aws_secret_id = aws_secret_id
        self.aws_region_name = aws_region_name
        self._lock = threading.RLock()
        self._audit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

        # AWS opcjonalnie
        self._aws_available = False
        self._boto3 = None
        if self.backend == "aws":
            try:
                import boto3  # type: ignore
                self._aws_available = True
                self._boto3 = boto3
            except Exception:
                logger.error("boto3 not available — AWS backend disabled")
                raise SecurityError("AWS backend requested but boto3 is not installed")

        # upewnij się, że folder istnieje (dla plików lokalnych)
        if self.backend == "local":
            try:
                self.key_file.parent.mkdir(parents=True, exist_ok=True)
                self.salt_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to ensure key/salt directories: {e}")

    # --------------------- Walidacje ---------------------

    @staticmethod
    def validate_api_key(key: str) -> bool:
        """Prosta walidacja kluczy API (długość + charset)."""
        if not key or len(key) < 32:
            return False
        for ch in key:
            if ch.isalnum() or ch in "+/=._-":
                continue
            return False
        return True

    # --------------------- KDF/Key utils ---------------------

    def _ensure_salt(self) -> bytes:
        if self.salt_file.exists():
            try:
                data = self.salt_file.read_bytes()
                if len(data) >= self._SALT_LEN:
                    return data[: self._SALT_LEN]
            except Exception as e:
                logger.warning(f"Salt read error, regenerating: {e}")
        # generuj nową sól
        salt = os.urandom(self._SALT_LEN)
        try:
            self.salt_file.write_bytes(salt)
        except Exception as e:
            logger.warning(f"Salt write error (using transient salt): {e}")
        return salt

    @staticmethod
    def _pbkdf2(password: str, salt: bytes, iters: int) -> bytes:
        if not password or not isinstance(password, str):
            raise SecurityError("Password must be a non-empty string")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iters,
        )
        return kdf.derive(password.encode("utf-8"))

    # --------------------- LOCAL (AES-GCM) ---------------------

    def _local_encrypt(self, plaintext: bytes, password: str) -> bytes:
        salt = self._ensure_salt()
        key = self._pbkdf2(password, salt, self._KDF_ITERS)
        aes = AESGCM(key)
        nonce = os.urandom(self._NONCE_LEN)
        ct = aes.encrypt(nonce, plaintext, associated_data=None)
        blob = {
            "version": 2,
            "scheme": "aes-gcm",
            "kdf": "pbkdf2-sha256",
            "iterations": self._KDF_ITERS,
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ct).decode("ascii"),
        }
        return json.dumps(blob, separators=(",", ":")).encode("utf-8")

    def _local_decrypt(self, enc: bytes, password: str) -> bytes:
        # próba: nowszy format (JSON AES-GCM)
        try:
            blob = json.loads(enc.decode("utf-8"))
            if not isinstance(blob, dict) or "ciphertext" not in blob:
                raise ValueError("Invalid blob")
            if str(blob.get("scheme", "")).lower() != "aes-gcm":
                raise ValueError("Unknown scheme")

            salt = base64.b64decode(blob["salt"])
            nonce = base64.b64decode(blob["nonce"])
            ct = base64.b64decode(blob["ciphertext"])
            iters = int(blob.get("iterations", self._KDF_ITERS))
            key = self._pbkdf2(password, salt, iters)
            aes = AESGCM(key)
            return aes.decrypt(nonce, ct, associated_data=None)
        except Exception:
            # ZGODNOŚĆ WSTECZNA: legacy Fernet
            try:
                legacy_key = base64.urlsafe_b64encode(
                    self._pbkdf2(password, self._LEGACY_SALT, self._LEGACY_ITERS)
                )
                f = Fernet(legacy_key)
                return f.decrypt(enc)
            except (FernetInvalidToken, Exception) as ee:
                # dopasowanie do testu: komunikat musi zawierać "Invalid password"
                raise SecurityError("Invalid password") from ee

    # --------------------- AWS Secrets Manager ---------------------

    def _aws_client(self):
        if not (self._aws_available and self._boto3):
            raise SecurityError("AWS backend is not available")
        region = self.aws_region_name or os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if not region:
            raise SecurityError("You must specify a region.")
        return self._boto3.client("secretsmanager", region_name=region)

    # --------------------- Audyt bezpieczeństwa ---------------------

    def register_audit_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Pozwala GUI/bazie na rejestrowanie zdarzeń audytowych (odszyfrowanie kluczy)."""
        with self._lock:
            self._audit_callback = callback

    @staticmethod
    def _mask_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        if not value:
            return value
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}***{value[-2:]}"

    def _mask_keys(self, keys: Dict[str, Any]) -> Dict[str, Any]:
        masked: Dict[str, Any] = {}
        for scope, creds in keys.items():
            if isinstance(creds, dict):
                masked[scope] = {k: self._mask_value(v) for k, v in creds.items()}
            else:
                masked[scope] = self._mask_value(creds)
        return masked

    def _emit_audit_event(self, action: str, **context: Any) -> None:
        status = str(context.pop("status", "success"))
        metadata: Dict[str, Any] = {}
        for key, value in context.items():
            if key in {"secret", "secrets", "payload"}:
                continue
            if key == "keys" and isinstance(value, dict):
                metadata[key] = self._mask_keys(value)
            else:
                metadata[key] = value

        payload: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "status": status,
            "backend": self.backend,
            "metadata": metadata,
            "detail": f"{action} ({self.backend})",
            "actor": "security_manager",
        }
        if self.backend == "local":
            payload["metadata"].setdefault("key_file", str(self.key_file))
        if self.backend == "aws" and self.aws_secret_id:
            payload["metadata"].setdefault("secret_id", self.aws_secret_id)

        audit_logger.info("%s %s", action.upper(), payload["metadata"])

        # emit CRITICAL alert na nieudane odszyfrowanie (wymóg testu)
        if status != "success" and action.startswith("decrypt"):
            emit_alert(
                "Błąd odszyfrowania kluczy API",
                severity=AlertSeverity.CRITICAL,
                source="security",
                context={"action": action, "backend": self.backend},
            )

        callback = self._audit_callback
        if callback:
            try:
                callback(action, payload)
            except Exception:  # pragma: no cover
                logger.exception("Security audit callback zgłosił wyjątek")

    # --------------------- API PUBLICZNE ---------------------

    def save_encrypted_keys(self, keys: Dict[str, Any], password: str) -> None:
        """
        Zapis zaszyfrowanych kluczy:
        - local: do pliku `key_file` jako JSON (AES-GCM)
        - aws: do Secrets Manager (SecretString)

        `keys` może być płaskim dictem (testnet_key/live_key...) lub zagnieżdżonym.
        """
        with self._lock:
            try:
                if not isinstance(keys, dict) or not keys:
                    raise SecurityError("Keys must be a non-empty dictionary")

                payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

                if self.backend == "aws":
                    if not self.aws_secret_id:
                        raise SecurityError("aws_secret_id is required for AWS backend")
                    client = self._aws_client()
                    client.put_secret_value(
                        SecretId=self.aws_secret_id,
                        SecretString=payload.decode("utf-8"),
                    )
                    logger.info("API keys saved to AWS Secrets Manager")
                    self._emit_audit_event(
                        "encrypt_keys",
                        backend="aws",
                        keys=keys,
                        status="success",
                    )
                else:
                    enc = self._local_encrypt(payload, password)
                    self.key_file.write_bytes(enc)
                    logger.info(f"API keys saved locally to {self.key_file}")
                    self._emit_audit_event(
                        "encrypt_keys",
                        backend="local",
                        keys=keys,
                        status="success",
                    )
            except SecurityError:
                raise
            except Exception as e:
                logger.error(f"Failed to save keys: {e}")
                self._emit_audit_event(
                    "encrypt_keys_failed",
                    error=str(e),
                    backend=self.backend,
                    status="error",
                )
                raise SecurityError(f"Failed to save keys: {e}") from e

    def load_encrypted_keys(self, password: str) -> Dict[str, Any]:
        """
        Odczyt zaszyfrowanych kluczy:
        - local: z pliku `key_file` (obsługa AES-GCM i legacy Fernet)
        - aws: z Secrets Manager (SecretString)

        Zwraca dict (taki sam kształt, jaki zapisano).
        """
        with self._lock:
            try:
                if self.backend == "aws":
                    if not self.aws_secret_id:
                        raise SecurityError("aws_secret_id is required for AWS backend")
                    client = self._aws_client()
                    res = client.get_secret_value(SecretId=self.aws_secret_id)
                    raw = res.get("SecretString")
                    if raw is None:
                        raw_b = res.get("SecretBinary")
                        if raw_b is None:
                            raise SecurityError("Secret has no payload")
                        raw = raw_b if isinstance(raw_b, str) else raw_b.decode("utf-8", "ignore")
                    data = json.loads(raw)
                    if not isinstance(data, dict):
                        raise SecurityError("Invalid key format in AWS secret")
                    logger.info("API keys loaded from AWS Secrets Manager")
                    self._emit_audit_event(
                        "decrypt_keys",
                        backend="aws",
                        keys=data,
                        status="success",
                    )
                    return data

                # LOCAL
                if not self.key_file.exists():
                    raise SecurityError(f"Key file {self.key_file} not found")

                enc = self.key_file.read_bytes()
                dec = self._local_decrypt(enc, password)
                data = json.loads(dec.decode("utf-8"))

                if not isinstance(data, dict):
                    raise SecurityError("Invalid key format")
                logger.info("API keys loaded from local file")
                self._emit_audit_event(
                    "decrypt_keys",
                    backend="local",
                    keys=data,
                    status="success",
                )
                return data
            except SecurityError as se:
                # zasygnalizuj porażkę i emuluj alert (emit nastąpi w _emit_audit_event)
                self._emit_audit_event(
                    "decrypt_keys_failed",
                    backend=self.backend,
                    error=str(se),
                    status="error",
                )
                raise
            except Exception as e:
                msg = str(e)
                if "region" in msg.lower():
                    msg = "You must specify a region."
                logger.error(f"Failed to load keys: {msg}")
                self._emit_audit_event(
                    "decrypt_keys_failed",
                    backend=self.backend,
                    error=msg,
                    status="error",
                )
                raise SecurityError(f"Failed to load keys: {msg}") from e
