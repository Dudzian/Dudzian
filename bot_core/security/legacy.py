"""Legacy komponenty bezpieczeństwa wymagane przez narzędzia migracyjne Stage6."""

from __future__ import annotations

import base64
import json
import logging
import os
import threading
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from bot_core.alerts import AlertSeverity, emit_alert  # type: ignore

from cryptography.fernet import Fernet, InvalidToken as FernetInvalidToken  # legacy fallback
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

__all__ = ["SecurityError", "SecurityManager"]

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(_handler)
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
        logging.Formatter("[%(asctime)s] SECURITY-AUDIT %(levelname)s: %(message)s")
    )
    audit_logger.addHandler(_audit_handler)
audit_logger.setLevel(logging.INFO)


class SecurityError(Exception):
    """Błąd operacji bezpieczeństwa (szyfrowanie/odszyfrowywanie)."""

    pass


class SecurityManager:
    """Manager szyfrowania kluczy API wykorzystywany przez narzędzia migracyjne."""

    _KDF_ITERS = 480_000
    _SALT_LEN = 16
    _NONCE_LEN = 12
    _LEGACY_SALT = b"trading_bot_salt_2025"
    _LEGACY_ITERS = 100_000

    def __init__(
        self,
        key_file: str | Path = "api_keys.enc",
        salt_file: Optional[str | Path] = None,
        backend: str = "local",
        aws_secret_id: Optional[str] = None,
        aws_region_name: Optional[str] = None,
    ) -> None:
        self.key_file = Path(key_file)
        self.salt_file = Path(salt_file) if salt_file else self.key_file.with_name("salt.bin")
        self.backend = (backend or "local").lower()
        self.aws_secret_id = aws_secret_id
        self.aws_region_name = aws_region_name
        self._lock = threading.RLock()
        self._audit_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

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

        if self.backend == "local":
            try:
                self.key_file.parent.mkdir(parents=True, exist_ok=True)
                self.salt_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                logger.warning("Failed to ensure key/salt directories: %s", exc)

    # ------------------------------------------------------------------
    @staticmethod
    def validate_api_key(key: str) -> bool:
        if not key or len(key) < 32:
            return False
        for ch in key:
            if ch.isalnum() or ch in "+/=._-":
                continue
            return False
        return True

    # ------------------------------------------------------------------
    def _ensure_salt(self) -> bytes:
        if self.salt_file.exists():
            try:
                data = self.salt_file.read_bytes()
                if len(data) >= self._SALT_LEN:
                    return data[: self._SALT_LEN]
            except Exception as exc:
                logger.warning("Salt read error, regenerating: %s", exc)
        salt = os.urandom(self._SALT_LEN)
        try:
            self.salt_file.write_bytes(salt)
        except Exception as exc:
            logger.warning("Salt write error (using transient salt): %s", exc)
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
        except Exception as exc:
            raise SecurityError("Failed to decrypt secret payload") from exc

    def _legacy_encrypt(self, plaintext: bytes, password: str) -> bytes:
        salt = self._LEGACY_SALT
        key = self._pbkdf2(password, salt, self._LEGACY_ITERS)
        token = Fernet(base64.urlsafe_b64encode(key)).encrypt(plaintext)
        return token

    def _legacy_decrypt(self, enc: bytes, password: str) -> bytes:
        try:
            salt = self._LEGACY_SALT
            key = self._pbkdf2(password, salt, self._LEGACY_ITERS)
            return Fernet(base64.urlsafe_b64encode(key)).decrypt(enc)
        except (SecurityError, FernetInvalidToken) as exc:
            raise SecurityError("Failed to decrypt legacy secret payload") from exc

    def save_encrypted_keys(self, keys: Dict[str, Any], password: str) -> None:
        if not isinstance(keys, dict):
            raise SecurityError("Keys must be provided as a mapping")
        payload = json.dumps(keys, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if self.backend == "aws":
            self._store_remote(payload, password)
        else:
            encrypted = self._local_encrypt(payload, password)
            self.key_file.write_bytes(encrypted)
            audit_logger.info("Saved encrypted keys to %s", self.key_file)

    def load_encrypted_keys(self, password: str) -> Dict[str, Any]:
        if self.backend == "aws":
            payload = self._load_remote(password)
        else:
            payload = self._local_load(password)
        if not isinstance(payload, dict):
            raise SecurityError("SecurityManager zwrócił niepoprawną strukturę sekretów (oczekiwano mapowania klucz→wartość).")
        return payload

    # ------------------------------------------------------------------
    def _local_load(self, password: str) -> Dict[str, Any]:
        try:
            raw = self.key_file.read_bytes()
        except FileNotFoundError:
            return {}
        try:
            try:
                decrypted = self._local_decrypt(raw, password)
            except SecurityError:
                decrypted = self._legacy_decrypt(raw, password)
            payload = json.loads(decrypted.decode("utf-8"))
        except Exception as exc:
            raise SecurityError("Nie udało się odczytać pliku SecurityManager") from exc
        return payload

    # ------------------------------------------------------------------
    def _store_remote(self, payload: bytes, password: str) -> None:
        if not self._aws_available or self._boto3 is None:
            raise SecurityError("AWS backend not configured")
        if not self.aws_secret_id:
            raise SecurityError("AWS secret id is required for remote storage")
        session = self._boto3.session.Session()
        client = session.client("secretsmanager", region_name=self.aws_region_name)
        encrypted = self._legacy_encrypt(payload, password)
        client.put_secret_value(SecretId=self.aws_secret_id, SecretBinary=encrypted)
        audit_logger.info("Saved encrypted keys to AWS Secrets Manager (%s)", self.aws_secret_id)

    def _load_remote(self, password: str) -> Dict[str, Any]:
        if not self._aws_available or self._boto3 is None:
            raise SecurityError("AWS backend not configured")
        if not self.aws_secret_id:
            raise SecurityError("AWS secret id is required for remote storage")
        session = self._boto3.session.Session()
        client = session.client("secretsmanager", region_name=self.aws_region_name)
        response = client.get_secret_value(SecretId=self.aws_secret_id)
        blob = response.get("SecretBinary")
        if blob is None:
            raise SecurityError("Missing SecretBinary in AWS response")
        decrypted = self._legacy_decrypt(blob, password)
        return json.loads(decrypted.decode("utf-8"))

    # ------------------------------------------------------------------
    def set_audit_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._audit_callback = callback

    def _emit_audit_event(self, event: str, payload: Dict[str, Any]) -> None:
        audit_logger.info("%s: %s", event, payload)
        if self._audit_callback:
            try:
                self._audit_callback(event, payload)
            except Exception:  # pragma: no cover - audyt nie może zatrzymać operacji
                logger.debug("Audit callback failed", exc_info=True)

    # ------------------------------------------------------------------
    def record_secret_rotation(self, key: str, *, actor: str | None = None) -> None:
        payload = {"key": key, "actor": actor, "timestamp": datetime.now(timezone.utc).isoformat()}
        self._emit_audit_event("secret_rotation", payload)
        emit_alert(
            "security.secret_rotation",
            AlertSeverity.INFO,
            message="Zarejestrowano rotację sekretu",
            metadata=payload,
        )
