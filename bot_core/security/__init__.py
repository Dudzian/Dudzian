"""Warstwa bezpieczeństwa i obsługi sekretów bota handlowego."""

from bot_core.security.base import (
    SecretManager,
    SecretPayload,
    SecretStorage,
    SecretStorageError,
)
from bot_core.security.factory import create_default_secret_storage
from bot_core.security.file_storage import EncryptedFileSecretStorage
from bot_core.security.fingerprint import (
    DeviceFingerprintGenerator,
    FingerprintDocument,
    FingerprintError,
    build_fingerprint_document,
    get_local_fingerprint,
    verify_document,
)
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes
from bot_core.security.token_audit import (
    TokenAuditReport,
    TokenAuditServiceReport,
    audit_service_token_configs,
    audit_service_tokens,
)
from bot_core.security.tokens import (
    ServiceToken,
    ServiceTokenValidator,
    build_service_token_validator,
    resolve_service_token,
    resolve_service_token_secret,
)

__all__ = [
    "SecretManager",
    "SecretPayload",
    "SecretStorage",
    "SecretStorageError",
    "KeyringSecretStorage",
    "EncryptedFileSecretStorage",
    "create_default_secret_storage",
    "RotationRegistry",
    "RotationStatus",
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
    "canonical_json_bytes",
    "build_hmac_signature",
    "ServiceToken",
    "ServiceTokenValidator",
    "build_service_token_validator",
    "resolve_service_token",
    "resolve_service_token_secret",
    "TokenAuditReport",
    "TokenAuditServiceReport",
    "audit_service_token_configs",
    "audit_service_tokens",
]
