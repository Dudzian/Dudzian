"""Journal and API unit tests below a mocked PostgreSQL close validator.

The fixture uses a private test-only membership map at an explicit validator
seam. It never registers capabilities in the real authority registry. These
tests exercise journal persistence and recovery, not DB-first security proofs.
Genuine PostgreSQL authorization, process crashes, and uncertain COMMIT remain
the responsibility of the separate real PostgreSQL lifecycle integration tests.
"""

from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path

import pytest

import bot_core.local_signing_custody as custody
import bot_core.local_signing_lifecycle_evidence as lifecycle
import bot_core.postgresql_custody_lifecycle as postgres_lifecycle
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_custody import (
    LocalSigningCustodyError,
    LocalSigningLifecycleError,
    LocalSigningMetadata,
    SigningKeyLifecycle,
    provision_local_signing_authority,
)
from bot_core.root_proof_issuer_substrate import ProviderRole, SecurityProfileIdentity
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_local_signing_custody import _admin, native_keyring, security


pytestmark = requires_native_custody_locking


@pytest.fixture
def authority(tmp_path: Path, native_keyring, security: SecurityProfileIdentity):
    return provision_local_signing_authority(
        tmp_path,
        _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING),
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )


@pytest.fixture
def close_authorization(monkeypatch):
    unit_authorizations = {}

    def validate(authorization: object, descriptor: bytes) -> None:
        if (
            type(authorization) is not postgres_lifecycle.DurableCloseAuthorization
            or type(descriptor) is not bytes
            or unit_authorizations.get(authorization) != descriptor
            or authorization.descriptor != descriptor
        ):
            raise postgres_lifecycle.CustodyLifecycleConflict(
                "positive committed exact close authorization required"
            )

    monkeypatch.setattr(postgres_lifecycle, "validate_durable_close_authorization", validate)

    def authorize(descriptor: bytes):
        authorization = postgres_lifecycle.DurableCloseAuthorization(descriptor)
        unit_authorizations[authorization] = descriptor
        return authorization

    return authorize


def _descriptor(
    path: Path,
    security: SecurityProfileIdentity,
    metadata: LocalSigningMetadata,
    operation_id: str = "lifecycle-op-1",
    target: SigningKeyLifecycle = SigningKeyLifecycle.VERIFY_ONLY,
) -> bytes:
    return canonical_json_bytes(
        {
            "protocol_version": 2,
            "operation_id": operation_id,
            "fence_key": "root-fence-1",
            "semantic_role": metadata.provider_role.value,
            "credential_identity": asdict(metadata.credential_identity()),
            "public_key_hex": metadata.public_key.hex(),
            "key_version": metadata.key_version,
            "lifecycle_generation": metadata.lifecycle_generation,
            "predecessor_state": metadata.lifecycle_state.value,
            "target_state": target.value,
            "custody_scope": {
                "directory": str(path),
                "security_profile": security.profile.value,
                "trust_domain": security.trust_domain,
            },
            "database_identity": {
                "system_identifier": "100",
                "database_oid": 200,
                "database_name": "test-db",
                "cluster_name": "custody-unit",
                "connection_host": "localhost",
                "connection_port": 5432,
                "server_address": "127.0.0.1",
                "server_port": 5432,
                "instance_id": "test-instance",
            },
        }
    )


def _journal(tmp_path: Path, operation_id: str = "lifecycle-op-1") -> Path:
    return (
        tmp_path
        / ".signing-lifecycle-operations"
        / f"{hashlib.sha256(operation_id.encode()).hexdigest()}.json"
    )


def _transition(
    tmp_path: Path,
    security: SecurityProfileIdentity,
    authority: LocalSigningMetadata,
    close_authorization: object,
    **changes,
):
    arguments = dict(
        security=security,
        role=authority.provider_role,
        credential_identity=authority.credential_identity(),
        expected_lifecycle_generation=authority.lifecycle_generation,
        target=SigningKeyLifecycle.VERIFY_ONLY,
        lifecycle_operation_id="lifecycle-op-1",
        operation_descriptor=_descriptor(
            tmp_path,
            security,
            authority,
            changes.get("lifecycle_operation_id", "lifecycle-op-1"),
            changes.get("target", SigningKeyLifecycle.VERIFY_ONLY),
        ),
        close_authorization=close_authorization,
    )
    arguments.update(changes)
    if callable(close_authorization):
        arguments["close_authorization"] = close_authorization(arguments["operation_descriptor"])
    return lifecycle.durable_transition_local_signing_credential_lifecycle(tmp_path, **arguments)


def test_receipt_derives_from_durable_exact_complete_record(
    tmp_path, security, authority, close_authorization
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)
    assert receipt.metadata == replace(
        authority, lifecycle_generation=2, lifecycle_state=SigningKeyLifecycle.VERIFY_ONLY
    )
    assert receipt.operation_id == "lifecycle-op-1"
    assert receipt.operation_descriptor == _descriptor(tmp_path, security, authority)
    assert receipt.evidence_bytes == _journal(tmp_path).read_bytes()
    assert (
        receipt.completion_fingerprint_sha256 == hashlib.sha256(receipt.evidence_bytes).hexdigest()
    )
    assert json.loads(receipt.evidence_bytes)["completion_state"] == "COMPLETE"
    with lifecycle.verified_durable_signing_lifecycle_completion(
        receipt, operation_descriptor=receipt.operation_descriptor
    ) as metadata:
        assert metadata == receipt.metadata


@pytest.mark.parametrize(
    ("target", "generation"),
    [(SigningKeyLifecycle.VERIFY_ONLY, 2), (SigningKeyLifecycle.REVOKED, 2)],
)
def test_legal_downgrades_are_exact_and_keep_key_material(
    tmp_path, security, authority, close_authorization, target, generation
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization, target=target)
    assert receipt.metadata.lifecycle_generation == generation
    assert receipt.metadata.lifecycle_state is target
    assert receipt.metadata.public_key == authority.public_key
    assert receipt.metadata.credential_identity() == authority.credential_identity()


def test_history_verify_only_to_revoked_is_the_only_next_step(
    tmp_path, security, native_keyring, close_authorization
) -> None:
    history = provision_local_signing_authority(
        tmp_path,
        _admin(tmp_path, security, ProviderRole.HISTORY_ATTESTATION_SIGNING),
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )
    first = _transition(tmp_path, security, history, close_authorization)
    final = _transition(
        tmp_path,
        security,
        first.metadata,
        close_authorization,
        lifecycle_operation_id="lifecycle-op-2",
        target=SigningKeyLifecycle.REVOKED,
    )
    assert final.metadata.lifecycle_generation == 3
    assert final.metadata.lifecycle_state is SigningKeyLifecycle.REVOKED


@pytest.mark.parametrize(
    ("cut", "expected_state"),
    [
        ("lifecycle_after_intent", SigningKeyLifecycle.ACTIVE),
        ("lifecycle_after_custody_transition", SigningKeyLifecycle.VERIFY_ONLY),
        ("lifecycle_after_completion", SigningKeyLifecycle.VERIFY_ONLY),
    ],
)
def test_restart_replays_each_durable_cut_without_extra_generation(
    tmp_path, security, authority, close_authorization, monkeypatch, cut, expected_state
) -> None:
    def crash(name: str) -> None:
        if name == cut:
            raise OSError("ack lost")

    with pytest.raises(OSError, match="ack lost"):
        _transition(tmp_path, security, authority, close_authorization, fault_injector=crash)
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        ).lifecycle_state
        is expected_state
    )
    lifecycle._COMPLETIONS.clear()
    receipt = _transition(tmp_path, security, authority, close_authorization)
    assert receipt.metadata.lifecycle_generation == 2
    completed_bytes = receipt.evidence_bytes

    def forbidden_transition(*args, **kwargs):
        pytest.fail("completed replay repeated custody CAS")

    monkeypatch.setattr(lifecycle, "_transition_exact_locked", forbidden_transition)
    assert (
        _transition(tmp_path, security, authority, close_authorization).evidence_bytes
        == completed_bytes
    )
    assert _journal(tmp_path).read_bytes() == completed_bytes


def test_missing_close_authorization_precedes_any_journal_or_custody_mutation(
    tmp_path, security, authority, close_authorization
) -> None:
    with pytest.raises(postgres_lifecycle.CustodyLifecycleConflict, match="positive committed"):
        _transition(tmp_path, security, authority, object())
    assert not _journal(tmp_path).exists()
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == authority
    )


@pytest.mark.parametrize(
    "change",
    [
        {"target": SigningKeyLifecycle.REVOKED},
        {"expected_lifecycle_generation": 2},
        {"lifecycle_operation_id": "another-operation"},
        {"role": ProviderRole.HISTORY_ATTESTATION_SIGNING},
    ],
)
def test_genuine_close_cannot_authorize_different_mutation_inputs(
    tmp_path, security, authority, close_authorization, change
) -> None:
    descriptor = _descriptor(tmp_path, security, authority)
    authorization = close_authorization(descriptor)
    with pytest.raises(LocalSigningLifecycleError, match="authorized"):
        _transition(
            tmp_path,
            security,
            authority,
            authorization,
            operation_descriptor=descriptor,
            **change,
        )
    assert not _journal(tmp_path).exists()
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == authority
    )


def test_genuine_close_cannot_change_a_different_credential_or_directory(
    tmp_path, security, authority, close_authorization, native_keyring
) -> None:
    original = _descriptor(tmp_path, security, authority)
    authorization = close_authorization(original)
    other_path = tmp_path / "other-custody"
    other = provision_local_signing_authority(
        other_path,
        _admin(other_path, security, ProviderRole.ROOT_PROOF_SIGNING),
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    with pytest.raises(LocalSigningLifecycleError, match="authorized"):
        _transition(
            tmp_path,
            security,
            authority,
            authorization,
            operation_descriptor=original,
            credential_identity=other.credential_identity(),
        )
    with pytest.raises(LocalSigningLifecycleError, match="authorized"):
        _transition(other_path, security, other, authorization, operation_descriptor=original)
    assert not _journal(tmp_path).exists()
    assert not _journal(other_path).exists()
    assert (
        custody._read_record(
            other_path / "root-proof-signing.json", security, authority.provider_role
        )
        == other
    )


@pytest.mark.parametrize(
    "change",
    [
        {"key_version": 2},
        {"predecessor_state": "VERIFY_ONLY", "target_state": "REVOKED"},
    ],
)
def test_authorized_descriptor_must_match_real_exact_predecessor(
    tmp_path, security, authority, close_authorization, change
) -> None:
    operation = json.loads(_descriptor(tmp_path, security, authority))
    operation.update(change)
    descriptor = canonical_json_bytes(operation)
    with pytest.raises(LocalSigningLifecycleError, match="predecessor"):
        _transition(
            tmp_path,
            security,
            authority,
            close_authorization(descriptor),
            operation_descriptor=descriptor,
            target=SigningKeyLifecycle(operation["target_state"]),
        )
    assert not _journal(tmp_path).exists()
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == authority
    )


def test_exact_close_dto_and_copied_close_are_not_authorization(
    tmp_path, security, authority, close_authorization
) -> None:
    import copy

    descriptor = _descriptor(tmp_path, security, authority)
    genuine = close_authorization(descriptor)
    forged = postgres_lifecycle.DurableCloseAuthorization(descriptor)
    for value in (forged, copy.copy(genuine)):
        with pytest.raises(postgres_lifecycle.CustodyLifecycleConflict, match="positive committed"):
            _transition(tmp_path, security, authority, value, operation_descriptor=descriptor)
    assert not _journal(tmp_path).exists()


@pytest.mark.parametrize(
    "change",
    [
        {"target": SigningKeyLifecycle.ACTIVE},
        {"expected_lifecycle_generation": True},
        {"expected_lifecycle_generation": 0},
        {"expected_lifecycle_generation": 99},
        {"role": ProviderRole.CHA_FRESHNESS_PROPOSER_SIGNING},
        {"role": ProviderRole.FRESHNESS_AUTHORITY_FINALIZATION_SIGNING},
        {"lifecycle_operation_id": "../escape"},
        {"operation_descriptor": b'{"scope":"a","scope":"b"}'},
        {"operation_descriptor": b'{"invalid":NaN}'},
        {"operation_descriptor": b"{}"},
        {"operation_descriptor": b'{ "operation_id": "a" }'},
    ],
)
def test_invalid_inputs_fail_before_custody_changes(
    tmp_path, security, authority, close_authorization, change
) -> None:
    with pytest.raises(LocalSigningCustodyError):
        _transition(tmp_path, security, authority, close_authorization, **change)
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == authority
    )


@pytest.mark.parametrize(
    "change",
    [
        {"target": SigningKeyLifecycle.REVOKED},
        {"expected_lifecycle_generation": 2},
        {"operation_descriptor": b'{"different":"request"}'},
    ],
)
def test_operation_identity_reuse_for_conflicting_request_fails_closed(
    tmp_path, security, authority, close_authorization, change
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)
    before = receipt.evidence_bytes
    with pytest.raises(LocalSigningCustodyError):
        _transition(tmp_path, security, authority, close_authorization, **change)
    assert _journal(tmp_path).read_bytes() == before
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == receipt.metadata
    )


def test_later_legal_descendant_cannot_be_completion_for_original_operation(
    tmp_path, security, authority, close_authorization
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)
    _transition(
        tmp_path,
        security,
        receipt.metadata,
        close_authorization,
        lifecycle_operation_id="lifecycle-op-2",
        target=SigningKeyLifecycle.REVOKED,
    )
    with pytest.raises(LocalSigningLifecycleError, match="custody conflicts"):
        _transition(tmp_path, security, authority, close_authorization)
    with pytest.raises(LocalSigningLifecycleError, match="custody conflicts"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            receipt, operation_descriptor=receipt.operation_descriptor
        ):
            pytest.fail("stale receipt authorized publication")


def test_intent_with_contradictory_custody_never_mints_completion(
    tmp_path, security, authority, close_authorization
) -> None:
    def crash(name: str) -> None:
        if name == "lifecycle_after_intent":
            raise OSError("crash")

    with pytest.raises(OSError):
        _transition(tmp_path, security, authority, close_authorization, fault_injector=crash)
    _transition(
        tmp_path,
        security,
        authority,
        close_authorization,
        lifecycle_operation_id="lifecycle-op-2",
        target=SigningKeyLifecycle.REVOKED,
    )
    with pytest.raises(LocalSigningLifecycleError, match="custody conflicts"):
        _transition(tmp_path, security, authority, close_authorization)
    assert json.loads(_journal(tmp_path).read_bytes())["completion_state"] == "INTENT"


@pytest.mark.parametrize("forged", [object(), {"completion_state": "COMPLETE"}, "a" * 64])
def test_dto_and_hash_do_not_prove_custody_completion(forged) -> None:
    with pytest.raises(LocalSigningLifecycleError, match="genuine durable"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            forged, operation_descriptor=b'{"irrelevant":"forgery"}'
        ):
            pytest.fail("forged receipt accepted")


def test_unissued_exact_type_and_public_constructor_do_not_prove_completion() -> None:
    with pytest.raises(TypeError, match="issued only"):
        lifecycle.DurableSigningLifecycleCompletion()
    forged = object.__new__(lifecycle.DurableSigningLifecycleCompletion)
    with pytest.raises(LocalSigningLifecycleError, match="genuine durable"):
        _ = forged.evidence_bytes


@pytest.mark.parametrize("which", ["journal", "directory", "custody"])
def test_symlink_substitutions_fail_closed(
    tmp_path, security, authority, close_authorization, which
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)
    if which == "journal":
        path = _journal(tmp_path)
    elif which == "directory":
        path = _journal(tmp_path).parent
    else:
        path = tmp_path / "root-proof-signing.json"
    moved = path.with_name(path.name + ".moved")
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=which == "directory")
    with pytest.raises(LocalSigningCustodyError):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            receipt, operation_descriptor=receipt.operation_descriptor
        ):
            pytest.fail("symlink accepted")


def test_completed_journal_tampering_fails_even_with_valid_json(
    tmp_path, security, authority, close_authorization
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)
    value = json.loads(receipt.evidence_bytes)
    value["operation_descriptor_b64"] = "e30="
    _journal(tmp_path).write_bytes(lifecycle._canonical(value))
    with pytest.raises(LocalSigningCustodyError):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            receipt, operation_descriptor=receipt.operation_descriptor
        ):
            pytest.fail("tampered evidence accepted")


def test_initial_fsync_failure_leaves_custody_active_and_recovery_resyncs_intent(
    tmp_path, security, authority, close_authorization, monkeypatch
) -> None:
    real_sync = custody.os.fsync
    calls = 0

    def fail_once(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fsync failure")
        real_sync(descriptor)

    monkeypatch.setattr(custody.os, "fsync", fail_once)
    # POSIX can surface the injected fsync error directly. On Windows the
    # qualified journal re-sync boundary deliberately wraps it as custody
    # durability failure; the exact original OSError must remain the cause.
    if os.name == "nt":
        with pytest.raises(LocalSigningCustodyError, match="lifecycle evidence durability failed") as failure:
            _transition(tmp_path, security, authority, close_authorization)
        assert type(failure.value.__cause__) is OSError
        assert str(failure.value.__cause__) == "fsync failure"
    else:
        with pytest.raises(OSError, match="fsync failure"):
            _transition(tmp_path, security, authority, close_authorization)
    assert (
        custody._read_record(
            tmp_path / "root-proof-signing.json", security, authority.provider_role
        )
        == authority
    )
    monkeypatch.setattr(custody.os, "fsync", real_sync)
    assert (
        _transition(
            tmp_path, security, authority, close_authorization
        ).metadata.lifecycle_generation
        == 2
    )


def test_completion_failure_recovers_exact_post_state_without_repeated_cas(
    tmp_path, security, authority, close_authorization, monkeypatch
) -> None:
    real_replace = lifecycle._replace_record

    def lose_ack(path: Path, payload: bytes) -> None:
        real_replace(path, payload)
        raise OSError("completion fsync ack lost")

    monkeypatch.setattr(lifecycle, "_replace_record", lose_ack)
    with pytest.raises(OSError):
        _transition(tmp_path, security, authority, close_authorization)
    monkeypatch.setattr(lifecycle, "_replace_record", real_replace)
    receipt = _transition(tmp_path, security, authority, close_authorization)
    assert receipt.metadata.lifecycle_generation == 2


def test_verification_fsync_failure_cannot_authorize_publication(
    tmp_path, security, authority, close_authorization, monkeypatch
) -> None:
    receipt = _transition(tmp_path, security, authority, close_authorization)

    def fail_sync(descriptor: int) -> None:
        raise OSError("sync unavailable")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(LocalSigningCustodyError, match="durability failed"):
        with lifecycle.verified_durable_signing_lifecycle_completion(
            receipt, operation_descriptor=receipt.operation_descriptor
        ):
            pytest.fail("unsynced evidence accepted")
