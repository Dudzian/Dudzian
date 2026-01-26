"""Warstwa bezpieczeństwa i obsługi sekretów bota handlowego."""

from __future__ import annotations

from importlib import import_module
import importlib
from typing import TYPE_CHECKING

from bot_core.security.base import (
    SecretManager,
    SecretPayload,
    SecretStorage,
    SecretStorageError,
)
from bot_core.security.clock import ClockService
from bot_core.security.factory import (
    create_default_fingerprint_service,
    create_default_secret_storage,
)
from bot_core.security.hwid import HwIdProvider, HwIdProviderError
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
from bot_core.security.fingerprint_lock import (
    FingerprintLock,
    FingerprintLockError,
    load_fingerprint_lock,
    verify_local_hardware,
    write_fingerprint_lock,
)
from bot_core.security.rotation import RotationRegistry, RotationStatus
from bot_core.security.rotation_report import (
    RotationRecord,
    RotationSummary,
    build_rotation_summary_entry,
    write_rotation_summary,
)
_PYDANTIC_ERROR: Exception | None = None
_CRYPTOGRAPHY_ERROR: Exception | None = None
try:  # pragma: no cover - zależne od środowiska
    importlib.import_module("pydantic")
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - brak lub uszkodzone pydantic
    _PYDANTIC_ERROR = exc
    _HAS_PYDANTIC = False
else:  # pragma: no cover - pydantic dostępny
    _HAS_PYDANTIC = True

try:  # pragma: no cover - zależne od środowiska
    importlib.import_module("cryptography")
except (ModuleNotFoundError, ImportError) as exc:  # pragma: no cover - brak lub uszkodzone cryptography
    _CRYPTOGRAPHY_ERROR = exc
    _HAS_CRYPTOGRAPHY = False
else:  # pragma: no cover - cryptography dostępne
    _HAS_CRYPTOGRAPHY = True

if _HAS_PYDANTIC:  # pragma: no cover - zależne od środowiska
    from bot_core.security.install_validation import (
        FingerprintValidationResult,
        validate_fingerprint_document,
        validate_license_bundle,
    )
else:  # pragma: no cover - brak pydantic w light env
    class FingerprintValidationResult:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby walidować licencję."
            ) from _PYDANTIC_ERROR

    def validate_fingerprint_document(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby walidować licencję."
        ) from _PYDANTIC_ERROR

    def validate_license_bundle(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby walidować licencję."
        ) from _PYDANTIC_ERROR
from bot_core.security.messages import ValidationMessage, make_error, make_warning
if _HAS_CRYPTOGRAPHY:  # pragma: no cover - zależne od środowiska
    from bot_core.security.hardware_wallets import LedgerSigner, TrezorSigner
else:  # pragma: no cover - brak cryptography w light env
    class LedgerSigner:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "cryptography nie jest zainstalowane. Zainstaluj pakiet 'cryptography' aby użyć LedgerSigner."
            ) from _CRYPTOGRAPHY_ERROR

    class TrezorSigner:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "cryptography nie jest zainstalowane. Zainstaluj pakiet 'cryptography' aby użyć TrezorSigner."
            ) from _CRYPTOGRAPHY_ERROR
from bot_core.security.signing import (
    HmacTransactionSigner,
    TransactionSignerSelector,
    build_hmac_signature,
    build_transaction_signer_selector,
    canonical_json_bytes,
)
from bot_core.security.profiles import (
    UserProfile,
    load_profiles,
    save_profiles,
    upsert_profile,
    remove_profile,
    log_admin_event,
)
if _HAS_PYDANTIC:  # pragma: no cover - zależne od środowiska
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
else:  # pragma: no cover - brak pydantic w light env
    class ServiceToken:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć tokenów serwisowych."
            ) from _PYDANTIC_ERROR

    class ServiceTokenValidator:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć tokenów serwisowych."
            ) from _PYDANTIC_ERROR

    def build_service_token_validator(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć tokenów serwisowych."
        ) from _PYDANTIC_ERROR

    def resolve_service_token(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć tokenów serwisowych."
        ) from _PYDANTIC_ERROR

    def resolve_service_token_secret(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć tokenów serwisowych."
        ) from _PYDANTIC_ERROR

    class TokenAuditReport:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć audytu tokenów."
            ) from _PYDANTIC_ERROR

    class TokenAuditServiceReport:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć audytu tokenów."
            ) from _PYDANTIC_ERROR

    def audit_service_token_configs(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć audytu tokenów."
        ) from _PYDANTIC_ERROR

    def audit_service_tokens(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć audytu tokenów."
        ) from _PYDANTIC_ERROR
if _HAS_PYDANTIC:  # pragma: no cover - zależne od środowiska
    from bot_core.security.update import (
        UpdateArtifact,
        UpdateManifest,
        UpdateVerificationError,
        UpdateVerificationResult,
        verify_update_bundle,
    )
else:  # pragma: no cover - brak pydantic w light env
    class UpdateArtifact:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć aktualizacji."
            ) from _PYDANTIC_ERROR

    class UpdateManifest:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć aktualizacji."
            ) from _PYDANTIC_ERROR

    class UpdateVerificationError(RuntimeError):
        pass

    class UpdateVerificationResult:  # type: ignore[override]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError(
                "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć aktualizacji."
            ) from _PYDANTIC_ERROR

    def verify_update_bundle(*args, **kwargs):
        raise RuntimeError(
            "pydantic nie jest zainstalowany. Zainstaluj pakiet 'pydantic' aby użyć aktualizacji."
        ) from _PYDANTIC_ERROR

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
except (ModuleNotFoundError, ImportError):  # pragma: no cover
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
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    pass

# --- Leniwe eksporty TLS (zgodne z obiema gałęziami) -------------------------
if TYPE_CHECKING:  # pragma: no cover - tylko dla statycznych analizatorów
    from bot_core.security import tls_audit as _tls_audit  # noqa: F401
    from bot_core.security import tpm as _tpm  # noqa: F401

_TLS_EXPORTS = {
    "verify_certificate_key_pair",
    "audit_tls_entry",
    "audit_tls_assets",
    "audit_mtls_bundle",
}

_TPM_EXPORTS = {
    "validate_attestation",
    "TpmValidationError",
    "TpmValidationResult",
}


def __getattr__(name: str):
    if name in _TLS_EXPORTS:
        module = import_module("bot_core.security.tls_audit")
    elif name in _TPM_EXPORTS:
        module = import_module("bot_core.security.tpm")
    else:
        raise AttributeError(f"module 'bot_core.security' has no attribute '{name}'")
    value = getattr(module, name)
    globals()[name] = value  # cache
    return value


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
    "ClockService",
    "create_default_fingerprint_service",
    "HwIdProvider",
    "HwIdProviderError",
    "RotationRegistry",
    "RotationStatus",
    "RotationRecord",
    "RotationSummary",
    "build_rotation_summary_entry",
    "write_rotation_summary",
    "FingerprintValidationResult",
    "DeviceFingerprintGenerator",
    "FingerprintDocument",
    "FingerprintError",
    "FingerprintLock",
    "FingerprintLockError",
    "build_fingerprint_document",
    "get_local_fingerprint",
    "verify_document",
    "load_fingerprint_lock",
    "verify_local_hardware",
    "write_fingerprint_lock",
    "canonical_json_bytes",
    "build_hmac_signature",
    "build_transaction_signer_selector",
    "TransactionSignerSelector",
    "HmacTransactionSigner",
    "LedgerSigner",
    "TrezorSigner",
    "UserProfile",
    "load_profiles",
    "save_profiles",
    "upsert_profile",
    "log_admin_event",
    "remove_profile",
    "ValidationMessage",
    "make_error",
    "make_warning",
    "validate_fingerprint_document",
    "validate_license_bundle",
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
__all__.extend(sorted(_TLS_EXPORTS | _TPM_EXPORTS))
