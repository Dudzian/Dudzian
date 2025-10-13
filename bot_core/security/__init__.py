"""Warstwa bezpieczeństwa i obsługi sekretów bota handlowego."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from bot_core.security.base import (
    SecretManager,
    SecretPayload,
    SecretStorage,
    SecretStorageError,
)
from bot_core.security.factory import (
    create_default_fingerprint_service,
    create_default_secret_storage,
)
from bot_core.security.file_storage import EncryptedFileSecretStorage
from bot_core.security.fingerprint import (
    HardwareFingerprintService,
    RotatingHmacKeyProvider,
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

if TYPE_CHECKING:  # pragma: no cover - tylko dla statycznych analizatorów
    from bot_core.security import tls_audit as _tls_audit  # noqa: F401

_TLS_EXPORTS = {
    "verify_certificate_key_pair",
    "audit_tls_entry",
    "audit_tls_assets",
    "audit_mtls_bundle",
}


def __getattr__(name: str):
    if name in _TLS_EXPORTS:
        module = import_module("bot_core.security.tls_audit")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'bot_core.security' has no attribute '{name}'")

__all__ = [
    "SecretManager",
    "SecretPayload",
    "SecretStorage",
    "SecretStorageError",
    "KeyringSecretStorage",
    "EncryptedFileSecretStorage",
    "create_default_secret_storage",
    "create_default_fingerprint_service",
    "RotationRegistry",
    "RotationStatus",
    "canonical_json_bytes",
    "build_hmac_signature",
    "HardwareFingerprintService",
    "RotatingHmacKeyProvider",
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

__all__.extend(sorted(_TLS_EXPORTS))
