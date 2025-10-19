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
    DeviceFingerprintGenerator,
    FingerprintDocument,
    FingerprintError,
    build_fingerprint_document,
    get_local_fingerprint,
    verify_document,
)
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.rotation_report import (
    RotationRecord,
    RotationSummary,
    build_rotation_summary_entry,
    write_rotation_summary,
)
from bot_core.security.signing import build_hmac_signature, canonical_json_bytes
from bot_core.security.profiles import (
    UserProfile,
    load_profiles,
    save_profiles,
    upsert_profile,
    log_admin_event,
)
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
from bot_core.security.update import (
    UpdateArtifact,
    UpdateManifest,
    UpdateVerificationError,
    UpdateVerificationResult,
    verify_update_bundle,
)

# --- Opcjonalne eksporty fingerprint z dwóch gałęzi --------------------------
_fingerprint_v1_exports: list[str] = []
try:
    from bot_core.security.fingerprint import (  # type: ignore
        DeviceFingerprintGenerator,
        FingerprintDocument,
        FingerprintError,
        build_fingerprint_document,
        get_local_fingerprint,
        verify_document,
    )

    _fingerprint_v1_exports = [
        "DeviceFingerprintGenerator",
        "FingerprintDocument",
        "FingerprintError",
        "build_fingerprint_document",
        "get_local_fingerprint",
        "verify_document",
    ]
except Exception:  # pragma: no cover
    pass

_fingerprint_v2_exports: list[str] = []
try:
    from bot_core.security.fingerprint import (  # type: ignore
        HardwareFingerprintService,
        RotatingHmacKeyProvider,
    )

    _fingerprint_v2_exports = [
        "HardwareFingerprintService",
        "RotatingHmacKeyProvider",
    ]
except Exception:  # pragma: no cover
    pass

# --- Leniwe eksporty TLS (zgodne z obiema gałęziami) -------------------------
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
        globals()[name] = value  # cache
        return value
    raise AttributeError(f"module 'bot_core.security' has no attribute '{name}'")


# --- Publiczne API -----------------------------------------------------------
__all__ = [
    # podstawowe
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
    "RotationRecord",
    "RotationSummary",
    "build_rotation_summary_entry",
    "write_rotation_summary",
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
    "canonical_json_bytes",
    "build_hmac_signature",
    "UserProfile",
    "load_profiles",
    "save_profiles",
    "upsert_profile",
    "log_admin_event",
    "remove_profile",
    # tokeny i audyty
    "ServiceToken",
    "ServiceTokenValidator",
    "build_service_token_validator",
    "resolve_service_token",
    "resolve_service_token_secret",
    "TokenAuditReport",
    "TokenAuditServiceReport",
    "audit_service_token_configs",
    "audit_service_tokens",
    "UpdateArtifact",
    "UpdateManifest",
    "UpdateVerificationError",
    "UpdateVerificationResult",
    "verify_update_bundle",
]

# dołącz opcjonalne symbole fingerprint w zależności od dostępności
__all__.extend(_fingerprint_v1_exports)
__all__.extend(_fingerprint_v2_exports)

# dołącz leniwe eksporty TLS
__all__.extend(sorted(_TLS_EXPORTS))
