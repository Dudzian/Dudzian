"""Core-owned M0.10 current-secret metadata use decision authority."""

from __future__ import annotations

from dataclasses import asdict, replace
from types import MappingProxyType
from typing import cast

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.initial_security import (
    InitialSecurityAuthority,
    SecretMetadataProjection,
    _ID_RE,
)

_SECRET_KINDS = frozenset({"API_KEY", "API_SECRET", "PASSPHRASE", "PRIVATE_KEY"})
_ENVIRONMENTS = frozenset({"PAPER", "TESTNET", "LIVE"})
_SECRET_OPERATIONS = ("PRIVATE_DATA", "ORDER_ENTRY")
_SECRET_STATES = frozenset({"AVAILABLE", "ROTATED", "REVOKED", "REPLACED"})


def secret_metadata_fingerprint(metadata: SecretMetadataProjection) -> str:
    """Return the canonical integrity hash, which does not establish authority."""
    return cast(
        str,
        canonical_json_sha256(
            {
                key: value
                for key, value in asdict(metadata).items()
                if key != "content_fingerprint_sha256"
            }
        ),
    )


def _valid_secret_reference(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("secure-store://"):
        return False
    locator = value[len("secure-store://") :]
    forbidden = (
        "api_key",
        "apikey",
        "secret",
        "password",
        "token",
        "private_key",
        "credential_value",
        "plaintext",
    )
    return (
        bool(locator)
        and not any(char.isspace() or char in "?#=" for char in locator)
        and not any(marker in locator.lower() for marker in forbidden)
    )


def _canonical_id(value: object, prefix: str) -> bool:
    return isinstance(value, str) and value.startswith(prefix) and bool(_ID_RE.fullmatch(value))


def _valid_permitted_operations(value: object) -> bool:
    return (
        isinstance(value, tuple)
        and bool(value)
        and all(type(item) is str and item in _SECRET_OPERATIONS for item in value)
        and len(value) == len(set(value))
        and value == tuple(item for item in _SECRET_OPERATIONS if item in value)
    )


def _valid_secret_metadata(metadata: object) -> bool:
    if not isinstance(metadata, SecretMetadataProjection):
        return False
    try:
        return bool(
            _valid_secret_reference(metadata.secret_reference)
            and metadata.secret_kind in _SECRET_KINDS
            and _canonical_id(metadata.exchange_account_id, "xacc_")
            and _canonical_id(metadata.credential_profile_id, "cred_")
            and isinstance(metadata.exchange_id, str)
            and metadata.exchange_id
            and metadata.environment in _ENVIRONMENTS
            and _valid_permitted_operations(metadata.permitted_operations)
            and isinstance(metadata.secret_revision, int)
            and not isinstance(metadata.secret_revision, bool)
            and metadata.secret_revision >= 1
            and metadata.state in _SECRET_STATES
            and isinstance(metadata.content_fingerprint_sha256, str)
            and len(metadata.content_fingerprint_sha256) == 64
            and all(char in "0123456789abcdef" for char in metadata.content_fingerprint_sha256)
            and secret_metadata_fingerprint(metadata) == metadata.content_fingerprint_sha256
        )
    except (AttributeError, TypeError, ValueError):
        return False


def _seed_trusted_secret_metadata(
    authority: SecretUseAuthority,
    metadata: SecretMetadataProjection,
    *,
    current: bool = True,
) -> None:
    """Publish metadata already accepted/designated by the upstream M0.5 owner."""
    if not isinstance(authority, SecretUseAuthority) or not _valid_secret_metadata(metadata):
        raise ValueError("SECRET_INVALID")
    scope = (metadata.exchange_account_id, metadata.credential_profile_id)
    with authority._state.lock:  # noqa: SLF001 - trusted owner boundary
        before = authority._state.snapshot  # noqa: SLF001 - trusted owner boundary
        accepted = dict(before.accepted_secret_metadata)
        designations = dict(before.current_secret_metadata)
        accepted[metadata.content_fingerprint_sha256] = metadata
        if current:
            designations[scope] = metadata.content_fingerprint_sha256
        authority._state.snapshot = replace(  # noqa: SLF001 - trusted owner boundary
            before,
            accepted_secret_metadata=MappingProxyType(accepted),
            current_secret_metadata=MappingProxyType(designations),
        )


class SecretUseAuthority:
    """Make a read-only secret-use decision from coherent shared Core state."""

    def __init__(self, security: InitialSecurityAuthority) -> None:
        self._state = security._state  # noqa: SLF001 - same exact semantic plane

    def validate_secret_use(self, untrusted: object, operation: object, environment: object) -> str:
        if not isinstance(untrusted, SecretMetadataProjection) or not _valid_secret_metadata(
            untrusted
        ):
            return "SECRET_INVALID"
        scope = (untrusted.exchange_account_id, untrusted.credential_profile_id)
        with self._state.lock:
            snapshot = self._state.snapshot
            fingerprint = snapshot.current_secret_metadata.get(scope)
            current = snapshot.accepted_secret_metadata.get(fingerprint or "")
            if current != untrusted or fingerprint != untrusted.content_fingerprint_sha256:
                return "SECRET_STALE"
            if untrusted.state == "REVOKED":
                return "SECRET_REVOKED"
            if untrusted.state in {"ROTATED", "REPLACED"}:
                return "SECRET_STALE"
            return (
                "SECRET_AVAILABLE"
                if operation in untrusted.permitted_operations
                and environment == untrusted.environment
                else "SECRET_UNAVAILABLE"
            )


__all__ = ["SecretUseAuthority", "secret_metadata_fingerprint"]
