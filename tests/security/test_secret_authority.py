from __future__ import annotations

from dataclasses import replace
import inspect
from types import MappingProxyType

import pytest

from bot_core.security.initial_security import SecretMetadataProjection
from bot_core.security.secret_authority import (
    SecretUseAuthority,
    _seed_trusted_secret_metadata,
    secret_metadata_fingerprint,
)
from tests.security.test_authentication import prepared

EXCHANGE_ACCOUNT = "xacc_01890f3e-7b12-7cc1-8f2a-0123456789ab"
CREDENTIAL_PROFILE = "cred_01890f3e-7b12-7cc1-8f2a-1123456789ab"


def secret(**changes: object) -> SecretMetadataProjection:
    metadata = SecretMetadataProjection(
        secret_reference="secure-store://opaque/test-profile",
        secret_kind="API_SECRET",
        exchange_account_id=EXCHANGE_ACCOUNT,
        credential_profile_id=CREDENTIAL_PROFILE,
        exchange_id="EXCHANGE_TEST",
        environment="TESTNET",
        permitted_operations=("PRIVATE_DATA", "ORDER_ENTRY"),
        secret_revision=1,
        state="AVAILABLE",
        content_fingerprint_sha256="",
    )
    metadata = replace(metadata, **changes)  # type: ignore[arg-type]
    return replace(metadata, content_fingerprint_sha256=secret_metadata_fingerprint(metadata))


def arranged(tmp_path):  # type: ignore[no-untyped-def]
    security, _, _ = prepared(tmp_path)
    return security, SecretUseAuthority(security)


def test_exact_schema_and_public_surface(tmp_path) -> None:  # type: ignore[no-untyped-def]
    security, authority = arranged(tmp_path)
    assert tuple(SecretMetadataProjection.__dataclass_fields__) == (
        "secret_reference",
        "secret_kind",
        "exchange_account_id",
        "credential_profile_id",
        "exchange_id",
        "environment",
        "permitted_operations",
        "secret_revision",
        "state",
        "content_fingerprint_sha256",
    )
    assert tuple(inspect.signature(SecretUseAuthority.validate_secret_use).parameters) == (
        "self",
        "untrusted",
        "operation",
        "environment",
    )
    assert not any(
        hasattr(SecretUseAuthority, name)
        for name in ("accept", "register", "seed", "set_current", "mark_current")
    )
    assert isinstance(security.snapshot.accepted_secret_metadata, MappingProxyType)
    assert isinstance(security.snapshot.current_secret_metadata, MappingProxyType)
    assert authority.validate_secret_use(object(), "PRIVATE_DATA", "TESTNET") == "SECRET_INVALID"


def test_self_hashed_but_unaccepted_is_stale_without_mutation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    security, authority = arranged(tmp_path)
    candidate = secret()
    before = security.snapshot
    assert authority.validate_secret_use(candidate, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
    assert security.snapshot == before


def test_current_available_operation_and_environment_fencing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    security, authority = arranged(tmp_path)
    candidate = secret()
    _seed_trusted_secret_metadata(authority, candidate)
    before = security.snapshot
    assert authority.validate_secret_use(candidate, "PRIVATE_DATA", "TESTNET") == "SECRET_AVAILABLE"
    assert authority.validate_secret_use(candidate, "ORDER_ENTRY", "TESTNET") == "SECRET_AVAILABLE"
    assert authority.validate_secret_use(candidate, "WITHDRAW", "TESTNET") == "SECRET_UNAVAILABLE"
    assert authority.validate_secret_use(candidate, "PRIVATE_DATA", "LIVE") == "SECRET_UNAVAILABLE"
    assert security.snapshot == before


def test_explicit_current_switch_fences_immutable_history(tmp_path) -> None:  # type: ignore[no-untyped-def]
    security, authority = arranged(tmp_path)
    previous = secret(secret_revision=9)
    current = secret(secret_revision=2, secret_reference="secure-store://opaque/new-profile")
    _seed_trusted_secret_metadata(authority, previous)
    _seed_trusted_secret_metadata(authority, current)
    assert authority.validate_secret_use(previous, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
    assert authority.validate_secret_use(current, "PRIVATE_DATA", "TESTNET") == "SECRET_AVAILABLE"
    assert previous.content_fingerprint_sha256 in security.snapshot.accepted_secret_metadata


@pytest.mark.parametrize(
    ("state", "expected"),
    (("ROTATED", "SECRET_STALE"), ("REPLACED", "SECRET_STALE"), ("REVOKED", "SECRET_REVOKED")),
)
def test_exact_current_non_available_states(tmp_path, state: str, expected: str) -> None:  # type: ignore[no-untyped-def]
    _, authority = arranged(tmp_path)
    candidate = secret(state=state)
    _seed_trusted_secret_metadata(authority, candidate)
    assert authority.validate_secret_use(candidate, "PRIVATE_DATA", "TESTNET") == expected


@pytest.mark.parametrize(
    "reference",
    (
        "keyring://opaque",
        "secure-store://",
        "secure-store://contains-secret",
        "secure-store://x?y",
        "secure-store://x#y",
        "secure-store://x=y",
    ),
)
def test_invalid_secret_reference_is_intrinsically_invalid(tmp_path, reference: str) -> None:  # type: ignore[no-untyped-def]
    _, authority = arranged(tmp_path)
    assert (
        authority.validate_secret_use(secret(secret_reference=reference), "PRIVATE_DATA", "TESTNET")
        == "SECRET_INVALID"
    )


@pytest.mark.parametrize(
    "operations",
    ((), ("PRIVATE_DATA", "PRIVATE_DATA"), ("ORDER_ENTRY", "PRIVATE_DATA"), ("WITHDRAW",)),
)
def test_permitted_operations_must_be_nonempty_unique_and_canonical(
    tmp_path, operations: tuple[str, ...]
) -> None:  # type: ignore[no-untyped-def]
    _, authority = arranged(tmp_path)
    assert (
        authority.validate_secret_use(
            secret(permitted_operations=operations), "PRIVATE_DATA", "TESTNET"
        )
        == "SECRET_INVALID"
    )


def test_fingerprint_integrity_precedes_membership(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _, authority = arranged(tmp_path)
    candidate = secret()
    tampered = replace(candidate, exchange_id="TAMPERED")
    assert authority.validate_secret_use(tampered, "PRIVATE_DATA", "TESTNET") == "SECRET_INVALID"
    recomputed = secret(exchange_id="TAMPERED")
    assert authority.validate_secret_use(recomputed, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"


def test_accepted_without_current_designation_is_stale(tmp_path) -> None:  # type: ignore[no-untyped-def]
    security, authority = arranged(tmp_path)
    candidate = secret()
    _seed_trusted_secret_metadata(authority, candidate, current=False)
    assert candidate.content_fingerprint_sha256 in security.snapshot.accepted_secret_metadata
    assert authority.validate_secret_use(candidate, "PRIVATE_DATA", "TESTNET") == "SECRET_STALE"
