"""Purpose-separated secret custody for the M0.7 Order authority."""

from __future__ import annotations

import base64
import hashlib
import hmac
from typing import Protocol

ORDER_AUTHENTICITY_ALGORITHM = "HMAC-SHA-256"
PRODUCTION_ORDER_AUTHENTICITY_PURPOSE = "CRYPTOHUNTER_M0_7_ORDER_AUTHORITY_PRODUCTION_ADMISSION_V1"
TEST_ORDER_AUTHENTICITY_PURPOSE = "CRYPTOHUNTER_M0_7_ORDER_AUTHORITY_TEST_ADMISSION_V1"


class OrderAuthoritySecretCustody(Protocol):
    """Opaque custody: key bytes may be provisioned and used, never retrieved."""

    def persist(self, custody_handle: str, key_material: bytes) -> None: ...
    def digest(self, custody_handle: str, payload: bytes) -> bytes | None: ...


class KeyringOrderAuthoritySecretCustody:
    """Order-only production custody backed by the canonical OS-keyring storage."""

    _PREFIX = "dudzian.m0-7-order-authority.v1:"

    def __init__(self) -> None:
        # Kept local so importing the pure Order validators does not require optional crypto/keyring.
        from bot_core.security.keyring_storage import KeyringSecretStorage

        self._storage = KeyringSecretStorage(service_name="dudzian.m0-7-order-authority")

    def persist(self, custody_handle: str, key_material: bytes) -> None:
        if type(key_material) is not bytes or len(key_material) != 32:
            raise RuntimeError("ORDER_AUTHORITY_CUSTODY_UNAVAILABLE")
        key = self._PREFIX + custody_handle
        if self._storage.get_secret(key) is not None:
            raise RuntimeError("ORDER_AUTHORITY_CUSTODY_UNAVAILABLE")
        self._storage.set_secret(key, base64.b64encode(key_material).decode("ascii"))

    def digest(self, custody_handle: str, payload: bytes) -> bytes | None:
        try:
            encoded = self._storage.get_secret(self._PREFIX + custody_handle)
            if encoded is None:
                return None
            key = base64.b64decode(encoded, validate=True)
        except Exception:
            return None
        if len(key) != 32:
            return None
        return hmac.digest(key, payload, "sha256")


class DeterministicTestOrderAuthoritySecretCustody:
    """Test-only cryptographic domain; incapable of producing production MACs."""

    _SEED = b"cryptohunter-m0.7-test-order-authority-not-production"

    def persist(self, custody_handle: str, key_material: bytes) -> None:
        # The supplied production-style random material is deliberately ignored.
        if type(custody_handle) is not str or not custody_handle:
            raise RuntimeError("TEST_ORDER_AUTHORITY_CUSTODY_UNAVAILABLE")

    def digest(self, custody_handle: str, payload: bytes) -> bytes | None:
        key = hmac.digest(self._SEED, custody_handle.encode("utf-8"), "sha256")
        return hmac.digest(key, payload, "sha256")


def production_order_authority_custody() -> OrderAuthoritySecretCustody:
    """Core-owned construction boundary; no caller-provided backend or secret."""

    return KeyringOrderAuthoritySecretCustody()
