"""Negative coordinator API checks; PostgreSQL integration is separate.

These tests replace deployment qualification and SQL calls only to exercise
coordinator validation and error handling. Custody journals and transitions use
the real filesystem implementation beneath the mocked native keyring layer.
No mock deployment or response is presented as a qualified database authority.
"""

from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest

from bot_core import local_signing_custody as custody
from bot_core import postgresql_atomic_root_proof_commit as atomic_runtime
from bot_core import postgresql_custody_lifecycle as runtime
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_lifecycle_evidence import DurableSigningLifecycleCompletion
from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
)
from bot_core.root_proof_issuer_substrate import ProviderRole, SecurityProfileIdentity
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_local_signing_custody import _admin, native_keyring, security


pytestmark = requires_native_custody_locking


@pytest.fixture
def coordinator(tmp_path: Path, native_keyring, security, monkeypatch):
    metadata = custody.provision_local_signing_authority(
        tmp_path,
        _admin(tmp_path, security, ProviderRole.ROOT_PROOF_SIGNING),
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    identity = PostgreSQLAtomicRootProofPhysicalIdentity(
        101,
        102,
        "api_unit_test",
        "unit_cluster",
        "localhost",
        5432,
        "127.0.0.1",
        5432,
        "isolated-unit-test",
    )
    deployment = SimpleNamespace(
        atomic_deployment=SimpleNamespace(
            identity=identity, provisioning=SimpleNamespace(schema="api_unit_test")
        ),
        admin_connection=SimpleNamespace(dsn="dbname=api_unit_test"),
        evidence_key=b"a" * 32,
    )
    monkeypatch.setattr(runtime, "validate_custody_lifecycle_deployment", lambda value: None)
    administrator = runtime.PostgreSQLCustodyLifecycleAdministrator(deployment)
    snapshot = custody.LocalRootProofSigningProvider(
        tmp_path, keyring_index_path=tmp_path / "keyring-index.json", security=security
    ).credential_snapshot()
    fence_key = atomic_runtime._fence_key(
        "ROOT_PROOF_SIGNING", atomic_runtime._canonical(atomic_runtime._snapshot_data(snapshot))
    )
    request = runtime.prepare_custody_lifecycle_request(
        deployment,
        tmp_path,
        security=security,
        role=metadata.provider_role,
        administrative_operation_id="api-operation-1",
        target=custody.SigningKeyLifecycle.REVOKED,
        fence_key=fence_key,
    )
    return SimpleNamespace(
        administrator=administrator,
        request=request,
        deployment=deployment,
        before=metadata,
    )


def _pending(request):
    return {"operation": request.operation, "phase": "PENDING_CLOSED"}


def _completed(request):
    return {
        "operation": request.operation,
        "phase": "COMPLETED",
        "terminal_state": request.operation["target_state"],
    }


def _observe_custody(value):
    request = value.request
    return custody._read_record(
        request.directory / "root-proof-signing.json", request.security, request.role
    )


def _forbid_mutation(*args, **kwargs):
    pytest.fail("failed authority observation reached custody mutation")


def test_request_snapshots_descriptor_and_security_and_returns_deep_copies(coordinator, security):
    request = coordinator.request
    original = request.descriptor
    operation = request.operation
    operation["operation_id"] = "replacement"
    operation["credential_identity"]["credential_identity"] = "replacement"
    operation["database_identity"]["database_oid"] = 999
    operation["custody_scope"]["trust_domain"] = "replacement"
    assert request.descriptor == original
    assert request.operation_id == "api-operation-1"
    assert request.credential_identity == coordinator.before.credential_identity()
    assert request.operation["database_identity"] == runtime._database_identity(
        coordinator.deployment.atomic_deployment.identity
    )
    with pytest.raises(FrozenInstanceError):
        request.descriptor = b"{}"
    assert type(request.security) is SecurityProfileIdentity
    assert request.security is not security
    original_domain = request.security.trust_domain
    object.__setattr__(security, "trust_domain", "changed-caller-object")
    assert request.security.trust_domain == original_domain


def test_enrollment_requires_canonical_complete_custody_fence_before_database(
    coordinator, monkeypatch
):
    value = coordinator
    request = value.request
    monkeypatch.setattr(value.administrator, "_call", _forbid_mutation)
    arguments = dict(
        security=request.security, role=request.role, fence_key=request.operation["fence_key"]
    )
    binding = runtime.prepare_custody_lifecycle_binding(
        value.deployment.atomic_deployment.identity, request.directory, **arguments
    )
    assert binding["fence_key"] == request.operation["fence_key"]
    arguments["fence_key"] = "ROOT_PROOF_SIGNING:" + "0" * 64
    with pytest.raises(runtime.CustodyLifecycleConflict, match="exact custody credential"):
        runtime.prepare_custody_lifecycle_binding(
            value.deployment.atomic_deployment.identity, request.directory, **arguments
        )
    assert _observe_custody(value) == value.before


def test_history_enrollment_uses_original_credential_fence(coordinator):
    value = coordinator
    request = value.request
    role = ProviderRole.HISTORY_ATTESTATION_SIGNING
    history = custody.provision_local_signing_authority(
        request.directory,
        _admin(request.directory, request.security, role),
        security=request.security,
        role=role,
    )
    fence_key = atomic_runtime._fence_key(
        role.value,
        atomic_runtime.canonical_json_bytes(
            atomic_runtime._credential_material(history.credential_identity())
        ),
    )
    binding = runtime.prepare_custody_lifecycle_binding(
        value.deployment.atomic_deployment.identity,
        request.directory,
        security=request.security,
        role=role,
        fence_key=fence_key,
    )
    assert binding["fence_key"] == fence_key
    assert binding["public_key_hex"] == history.public_key.hex()
    assert binding["lifecycle_generation"] == 1


def test_postgresql_uint64_system_identity_remains_exact_in_canonical_descriptor(coordinator):
    request = coordinator.request
    operation = request.operation
    operation["database_identity"]["system_identifier"] = str(2**64 - 1)
    exact = runtime.CustodyLifecycleRequest(
        request.directory, request.security, request.role, canonical_json_bytes(operation)
    )
    assert exact.operation["database_identity"]["system_identifier"] == "18446744073709551615"


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("protocol_version", True),
        ("protocol_version", 1),
        ("operation_id", ""),
        ("operation_id", "  "),
        ("operation_id", 3),
        ("operation_id", "../escape"),
        ("operation_id", "a" * 129),
        ("fence_key", ""),
        ("key_version", True),
        ("key_version", 0),
        ("lifecycle_generation", True),
        ("lifecycle_generation", 0),
        ("lifecycle_generation", 1.0),
        ("lifecycle_generation", 2**63 - 1),
        ("predecessor_state", "REVOKED"),
        ("target_state", "ACTIVE"),
        ("target_state", "PENDING_CLOSED"),
        ("semantic_role", "HISTORY_ATTESTATION_SIGNING"),
        ("public_key_hex", "00"),
        ("public_key_hex", "00" * 32),
        ("public_key_hex", None),
        ("database_identity", {}),
    ],
)
def test_request_rejects_malformed_exact_binding(coordinator, field, invalid):
    request = coordinator.request
    operation = request.operation
    operation[field] = invalid
    with pytest.raises(runtime.CustodyLifecycleConflict):
        runtime.CustodyLifecycleRequest(
            request.directory,
            request.security,
            request.role,
            json.dumps(operation, sort_keys=True, separators=(",", ":")).encode(),
        )
    assert _observe_custody(coordinator) == coordinator.before


@pytest.mark.parametrize(
    "identity_change",
    [
        {"semantic_role": "HISTORY_ATTESTATION_SIGNING"},
        {"key_material_identity": "sha256:" + "0" * 64},
        {"credential_identity": ""},
        {"extra_field": "untrusted"},
    ],
)
def test_request_rejects_nested_credential_mismatch(coordinator, identity_change):
    request = coordinator.request
    operation = request.operation
    operation["credential_identity"].update(identity_change)
    with pytest.raises(runtime.CustodyLifecycleConflict):
        runtime.CustodyLifecycleRequest(
            request.directory, request.security, request.role, canonical_json_bytes(operation)
        )


@pytest.mark.parametrize(
    "descriptor",
    [
        b"{}",
        b'{"operation_id":"a","operation_id":"b"}',
        b'{"invalid":NaN}',
        b"not-json",
        b"\xff",
        b"[]",
        bytearray(b"{}"),
    ],
)
def test_request_rejects_noncanonical_or_nonexact_descriptor(coordinator, descriptor):
    request = coordinator.request
    with pytest.raises((runtime.CustodyLifecycleConflict, TypeError)):
        runtime.CustodyLifecycleRequest(
            request.directory, request.security, request.role, descriptor
        )


def test_prepare_never_grants_active_or_reverse_transition(coordinator):
    value = coordinator
    with pytest.raises(runtime.CustodyLifecycleConflict):
        runtime.prepare_custody_lifecycle_request(
            value.deployment,
            value.request.directory,
            security=value.request.security,
            role=value.request.role,
            administrative_operation_id="activation-attempt",
            target=custody.SigningKeyLifecycle.ACTIVE,
            fence_key=value.request.operation["fence_key"],
        )
    operation = value.request.operation
    operation.update(predecessor_state="VERIFY_ONLY", target_state="VERIFY_ONLY")
    with pytest.raises(runtime.CustodyLifecycleConflict):
        runtime.CustodyLifecycleRequest(
            value.request.directory,
            value.request.security,
            value.request.role,
            canonical_json_bytes(operation),
        )
    assert _observe_custody(value) == value.before


def test_request_for_another_physical_authority_never_calls_database(coordinator, monkeypatch):
    value = coordinator
    operation = value.request.operation
    operation["database_identity"]["system_identifier"] = str(
        int(operation["database_identity"]["system_identifier"]) + 1
    )
    foreign = runtime.CustodyLifecycleRequest(
        value.request.directory,
        value.request.security,
        value.request.role,
        canonical_json_bytes(operation),
    )
    monkeypatch.setattr(value.administrator, "_call", _forbid_mutation)
    with pytest.raises(runtime.CustodyLifecycleConflict, match="another physical authority"):
        value.administrator.begin_close(foreign)


@pytest.mark.parametrize(
    "response",
    [
        None,
        {"operation": {}, "phase": "PENDING_CLOSED"},
        {"phase": "ACTIVE"},
        {"phase": "COMPLETED", "terminal_state": "ACTIVE"},
    ],
)
def test_begin_close_has_no_capability_without_exact_positive_confirmation(
    coordinator, monkeypatch, response
):
    value = coordinator
    monkeypatch.setattr(value.administrator, "_call", lambda *args, **kwargs: response)
    with pytest.raises(runtime.CustodyLifecycleError):
        value.administrator.begin_close(value.request)
    assert _observe_custody(value) == value.before


def test_dto_or_copied_close_capability_cannot_mutate_custody(coordinator, monkeypatch):
    value = coordinator
    monkeypatch.setattr(
        value.administrator, "_call", lambda *args, **kwargs: _pending(value.request)
    )
    authentic = value.administrator.begin_close(value.request)
    for forged in (
        object(),
        runtime.DurableCloseAuthorization(value.request.descriptor),
        copy.copy(authentic),
    ):
        with pytest.raises(runtime.CustodyLifecycleConflict, match="positive committed"):
            value.administrator.mutate_custody(forged, value.request)
    assert _observe_custody(value) == value.before
    assert not (value.request.directory / ".signing-lifecycle-operations").exists()


@pytest.mark.parametrize("failure", ["absent", "unavailable", "conflict"])
def test_old_close_capability_rechecks_current_database_before_mutation(
    coordinator, monkeypatch, failure
):
    value = coordinator
    monkeypatch.setattr(
        value.administrator, "_call", lambda *args, **kwargs: _pending(value.request)
    )
    authorization = value.administrator.begin_close(value.request)

    def observe(*args, **kwargs):
        if failure == "unavailable":
            raise runtime.CustodyLifecycleUnavailable("database unavailable")
        if failure == "conflict":
            raise runtime.CustodyLifecycleConflict("authority drift")
        return None

    monkeypatch.setattr(value.administrator, "_call", observe)
    monkeypatch.setattr(
        runtime, "durable_transition_local_signing_credential_lifecycle", _forbid_mutation
    )
    with pytest.raises(runtime.CustodyLifecycleError):
        value.administrator.mutate_custody(authorization, value.request)
    assert _observe_custody(value) == value.before


def test_unknown_close_cannot_reach_mutation_without_positive_resolution(coordinator, monkeypatch):
    value = coordinator
    calls = []

    def call(name, args, *, write):
        calls.append(name)
        if name == "begin_lifecycle_close":
            raise runtime.CustodyLifecycleIndeterminate("lost close COMMIT response")
        return None

    monkeypatch.setattr(value.administrator, "_call", call)
    monkeypatch.setattr(
        runtime, "durable_transition_local_signing_credential_lifecycle", _forbid_mutation
    )
    with pytest.raises(runtime.CustodyLifecycleIndeterminate, match="no positive resolution"):
        value.administrator.execute(value.request)
    assert calls == [
        "read_lifecycle_operation",
        "read_lifecycle_operation",
        "begin_lifecycle_close",
        "read_lifecycle_operation",
    ]
    assert _observe_custody(value) == value.before


def test_unknown_close_executes_only_after_exact_positive_resolution(coordinator, monkeypatch):
    value = coordinator
    closed = False
    calls = []

    def call(name, args, *, write):
        nonlocal closed
        calls.append(name)
        if name == "begin_lifecycle_close":
            closed = True
            raise runtime.CustodyLifecycleIndeterminate("lost close COMMIT response")
        if name == "read_lifecycle_operation":
            return _pending(value.request) if closed else None
        return _completed(value.request)

    monkeypatch.setattr(value.administrator, "_call", call)
    assert value.administrator.execute(value.request)["phase"] == "COMPLETED"
    assert calls == [
        "read_lifecycle_operation",
        "read_lifecycle_operation",
        "begin_lifecycle_close",
        "read_lifecycle_operation",
        "read_lifecycle_operation",
        "complete_lifecycle_close",
    ]
    assert _observe_custody(value).lifecycle_generation == 2
    assert _observe_custody(value).lifecycle_state is custody.SigningKeyLifecycle.REVOKED


def test_unknown_completion_reconciles_without_repeating_irreversible_mutation(
    coordinator, monkeypatch
):
    value = coordinator
    completed = False

    def call(name, args, *, write):
        nonlocal completed
        if name == "complete_lifecycle_close":
            completed = True
            raise runtime.CustodyLifecycleIndeterminate("lost completion COMMIT response")
        return _completed(value.request) if completed else _pending(value.request)

    monkeypatch.setattr(value.administrator, "_call", call)
    assert value.administrator.execute(value.request)["phase"] == "COMPLETED"
    assert _observe_custody(value).lifecycle_generation == 2
    monkeypatch.setattr(
        runtime, "durable_transition_local_signing_credential_lifecycle", _forbid_mutation
    )
    assert value.administrator.execute(value.request)["phase"] == "COMPLETED"
    assert _observe_custody(value).lifecycle_generation == 2


def test_committed_terminal_replay_survives_later_descendant_but_old_receipt_cannot_publish(
    coordinator, monkeypatch
):
    value = coordinator
    operation = value.request.operation
    operation["target_state"] = "VERIFY_ONLY"
    request = runtime.CustodyLifecycleRequest(
        value.request.directory,
        value.request.security,
        value.request.role,
        canonical_json_bytes(operation),
    )
    monkeypatch.setattr(value.administrator, "_call", lambda *args, **kwargs: _pending(request))
    authorization = value.administrator.begin_close(request)
    receipt = value.administrator.mutate_custody(authorization, request)
    monkeypatch.setattr(value.administrator, "_call", lambda *args, **kwargs: _completed(request))
    assert value.administrator.publish_completion(request, receipt)["phase"] == "COMPLETED"
    custody.transition_local_signing_credential_lifecycle(
        request.directory,
        security=request.security,
        role=request.role,
        credential_identity=request.credential_identity,
        expected_lifecycle_generation=2,
        target=custody.SigningKeyLifecycle.REVOKED,
    )
    monkeypatch.setattr(
        runtime, "durable_transition_local_signing_credential_lifecycle", _forbid_mutation
    )
    assert value.administrator.execute(request)["terminal_state"] == "VERIFY_ONLY"
    assert _observe_custody(value).lifecycle_generation == 3
    assert _observe_custody(value).lifecycle_state is custody.SigningKeyLifecycle.REVOKED
    monkeypatch.setattr(value.administrator, "_call", _forbid_mutation)
    with pytest.raises(custody.LocalSigningLifecycleError, match="custody conflicts"):
        value.administrator.publish_completion(request, receipt)


@pytest.mark.parametrize(
    "forged", [object(), "caller-hash", {"completion_fingerprint_sha256": "0" * 64}]
)
def test_caller_completion_dto_or_hash_never_reaches_publication(coordinator, monkeypatch, forged):
    value = coordinator
    monkeypatch.setattr(value.administrator, "_call", _forbid_mutation)
    with pytest.raises(custody.LocalSigningLifecycleError, match="genuine durable"):
        value.administrator.publish_completion(value.request, forged)


def test_unregistered_exact_completion_class_never_reaches_publication(coordinator, monkeypatch):
    monkeypatch.setattr(coordinator.administrator, "_call", _forbid_mutation)
    counterfeit = object.__new__(DurableSigningLifecycleCompletion)
    with pytest.raises(custody.LocalSigningLifecycleError, match="genuine durable"):
        coordinator.administrator.publish_completion(coordinator.request, counterfeit)


def test_publication_binds_real_journal_and_held_custody_to_admin_mac(coordinator, monkeypatch):
    value = coordinator
    monkeypatch.setattr(
        value.administrator, "_call", lambda *args, **kwargs: _pending(value.request)
    )
    authorization = value.administrator.begin_close(value.request)
    completion = value.administrator.mutate_custody(authorization, value.request)
    observed_calls = []

    def publish(name, args, *, write):
        observed_calls.append(name)
        assert name == "complete_lifecycle_close"
        operation_id, evidence, tag = args
        assert operation_id == value.request.operation_id
        document = json.loads(evidence)
        assert canonical_json_bytes(document) == evidence
        assert document["operation"] == value.request.operation
        assert document["observed"]["lifecycle_state"] == "REVOKED"
        assert document["observed"]["lifecycle_generation"] == 2
        assert (
            document["local_evidence_sha256"]
            == hashlib.sha256(completion.evidence_bytes).hexdigest()
        )
        assert hmac.compare_digest(
            tag,
            hmac.digest(
                value.deployment.evidence_key, runtime._MAC_DOMAIN + evidence, hashlib.sha256
            ),
        )
        assert _observe_custody(value) == completion.metadata
        assert write is True
        return _completed(value.request)

    monkeypatch.setattr(value.administrator, "_call", publish)
    assert value.administrator.publish_completion(value.request, completion)["phase"] == "COMPLETED"
    assert observed_calls == ["complete_lifecycle_close"]


@pytest.mark.parametrize(
    ("stage", "write", "error"),
    [
        ("execute", True, runtime.CustodyLifecycleUnavailable),
        ("commit", True, runtime.CustodyLifecycleIndeterminate),
        ("commit", False, runtime.CustodyLifecycleUnavailable),
        ("post_commit", True, runtime.CustodyLifecycleIndeterminate),
    ],
)
def test_database_error_classification_preserves_unknown_outcome(
    coordinator, monkeypatch, stage, write, error
):
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args):
            if stage == "execute":
                raise psycopg.OperationalError("connection lost")
            return self

        def fetchone(self):
            return (_pending(coordinator.request),)

        def commit(self):
            if stage == "commit":
                raise psycopg.OperationalError("commit response lost")

    def cut(name, connection):
        if stage == "post_commit" and name.endswith("_commit"):
            raise psycopg.OperationalError("committed response lost")

    monkeypatch.setattr(runtime.psycopg, "connect", lambda *args, **kwargs: Connection())
    monkeypatch.setattr(
        runtime, "qualify_custody_lifecycle_authority", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(runtime, "_persistence_cut", cut)
    with pytest.raises(error):
        coordinator.administrator._call("begin_lifecycle_close", (), write=write)
