"""Warstwa bezpieczeństwa i obsługi sekretów bota handlowego."""

from bot_core.security.base import (
    SecretManager,
    SecretPayload,
    SecretStorage,
    SecretStorageError,
)
from bot_core.security.factory import create_default_secret_storage
from bot_core.security.file_storage import EncryptedFileSecretStorage
from bot_core.security.keyring_storage import KeyringSecretStorage
from bot_core.security.rotation import RotationRegistry, RotationStatus

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
]
