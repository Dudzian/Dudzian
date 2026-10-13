"""Legacy API isolation against qualified V3 enrollment and PostgreSQL 16.

The shared genuine INITIAL_BINDING fixture supplies immutable proof/history
bytes. Each test owns a distinct enrolled signing credential. Only the existing
native-keyring test seam is replaced; metadata, PostgreSQL, locks and fsync are
real. Retained-layout preparation is explicit privileged fixture setup, not a
new rotation implementation or application recovery path.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from bot_core import local_signing_custody as custody
from bot_core import postgresql_custody_lifecycle as lifecycle
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security import test_postgresql_custody_lifecycle as existing
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]

lineage_monkeypatch = existing.lineage_monkeypatch
lineage_path = existing.lineage_path
lineage_challenge = existing.lineage_challenge
lineage_integration = existing.lineage_integration
lineage_package = existing.lineage_package
lineage_lppi = existing.lineage_lppi
lineage_active = existing.lineage_active
lineage_committed = existing.lineage_committed
lineage_cha = existing.lineage_cha
lineage_reserved = existing.lineage_reserved
module_native_keyring = existing.module_native_keyring
genuine_attempt = existing.genuine_attempt
module_atomic_commit_boundary = existing.module_atomic_commit_boundary
lifecycle_boundary = existing.lifecycle_boundary

_ROLES = (ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING)
_DENIALS = (custody.LocalSigningCustodyError, lifecycle.CustodyLifecycleConflict)


@pytest.fixture(scope="module")
def isolation_pool(lifecycle_boundary):
    cases = tuple(lifecycle_boundary.lifecycle_cases)
    return {role: iter(tuple(case for case in cases if case.role is role)) for role in _ROLES}


@pytest.fixture(params=_ROLES, ids=lambda role: role.value)
def enrolled_case(lifecycle_boundary, isolation_pool, request):
    case = next(isolation_pool[request.param])
    assert case.role is request.param
    case.boundary = lifecycle_boundary
    case.request = existing._request(lifecycle_boundary, case)
    with psycopg.connect(BASE_DSN) as observer:
        assert observer.execute(
            sql.SQL(
                "SELECT credential_binding->>'semantic_role' "
                "FROM {}.lifecycle_credentials WHERE fence_key=%s"
            ).format(sql.Identifier(lifecycle_boundary.setup.schema)),
            (case.key,),
        ).fetchone() == (case.role.value,)
    return case


def _snapshot(case):
    """Physical metadata/journal bytes plus all durable administration effects."""
    with custody._custody_lock(case.directory, exclusive=False):
        local = tuple(
            (str(path.relative_to(case.directory)), path.read_bytes())
            for path in sorted(case.directory.rglob("*.json"))
        )
    with psycopg.connect(BASE_DSN) as observer:
        durable = tuple(
            observer.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1").format(
                    sql.Identifier(case.boundary.setup.schema, table)
                )
            ).fetchall()
            for table in (
                "fences",
                "lifecycle_credentials",
                "lifecycle_operations",
                "lifecycle_completions",
            )
        )
    return local, durable


def _legacy(case, route, *, identity=None, generation=None, target=None):
    security = case.boundary.security
    identity = identity or case.initial.credential_identity()
    generation = generation or case.initial.lifecycle_generation
    target = target or custody.SigningKeyLifecycle.REVOKED
    if route == "current-wrapper":
        return custody.transition_local_signing_lifecycle(
            case.directory, security=security, role=case.role, target=target
        )
    if route == "credential-wrapper":
        return custody.transition_local_signing_credential_lifecycle(
            case.directory,
            security=security,
            role=case.role,
            credential_identity=identity,
            expected_lifecycle_generation=generation,
            target=target,
        )
    with custody._custody_lock(case.directory, exclusive=True):
        if route == "internal-mutator":
            return custody._transition_exact_locked(
                case.directory, security, case.role, identity, generation, target
            )
        path, metadata = custody._resolve_credential_record(
            case.directory, security, case.role, identity
        )
        return custody._replace_record(
            path,
            custody._encode_record(
                replace(metadata, lifecycle_generation=generation + 1, lifecycle_state=target)
            ),
        )


@pytest.mark.parametrize(
    "route", ["current-wrapper", "credential-wrapper", "internal-mutator", "record-replacer"]
)
def test_enrolled_legacy_bypasses_leave_physical_metadata_and_db_unchanged(enrolled_case, route):
    case = enrolled_case
    before = _snapshot(case)
    with pytest.raises(_DENIALS):
        _legacy(case, route)
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("ACTIVE", 1)


@pytest.mark.parametrize("damage", ["deleted", "corrupted"])
def test_lost_enrollment_row_does_not_select_legacy_mode(enrolled_case, damage):
    case = enrolled_case
    with psycopg.connect(BASE_DSN) as bootstrap:
        # Simulate lost authority evidence using the separate test superuser.
        # Restore the immutable trigger before observing application behavior.
        bootstrap.execute(
            sql.SQL(
                "ALTER TABLE {}.lifecycle_credentials DISABLE TRIGGER lifecycle_immutable_row"
            ).format(sql.Identifier(case.boundary.setup.schema))
        )
        if damage == "deleted":
            bootstrap.execute(
                sql.SQL("DELETE FROM {}.lifecycle_credentials WHERE fence_key=%s").format(
                    sql.Identifier(case.boundary.setup.schema)
                ),
                (case.key,),
            )
        else:
            bootstrap.execute(
                sql.SQL(
                    "UPDATE {}.lifecycle_credentials SET credential_binding="
                    "jsonb_set(credential_binding,'{{public_key_hex}}',to_jsonb(%s::text)) "
                    "WHERE fence_key=%s"
                ).format(sql.Identifier(case.boundary.setup.schema)),
                ("corrupted-" + case.key, case.key),
            )
        bootstrap.execute(
            sql.SQL(
                "ALTER TABLE {}.lifecycle_credentials ENABLE ALWAYS TRIGGER lifecycle_immutable_row"
            ).format(sql.Identifier(case.boundary.setup.schema))
        )
        bootstrap.commit()
    before = _snapshot(case)
    for route in ("current-wrapper", "credential-wrapper", "internal-mutator", "record-replacer"):
        with pytest.raises(_DENIALS):
            _legacy(case, route)
        assert _snapshot(case) == before
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        case.boundary.lifecycle_admin.begin_close(case.request)
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("ACTIVE", 1)


def test_constructed_close_dto_cannot_authorize_lowest_mutation(enrolled_case):
    case = enrolled_case
    before = _snapshot(case)
    with pytest.raises(lifecycle.CustodyLifecycleIndeterminate):
        case.boundary.lifecycle_admin._authorize(case.request)
    assert _snapshot(case) == before
    forged = lifecycle.DurableCloseAuthorization(case.request.descriptor)
    with pytest.raises(_DENIALS), custody._custody_lock(case.directory, exclusive=True):
        custody._transition_exact_locked(
            case.directory,
            case.boundary.security,
            case.role,
            case.initial.credential_identity(),
            1,
            custody.SigningKeyLifecycle.REVOKED,
            close_authorization=forged,
            operation_descriptor=case.request.descriptor,
        )
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("ACTIVE", 1)


def test_genuine_close_without_durable_intent_cannot_skip_phase_b(enrolled_case):
    case = enrolled_case
    authorization = case.boundary.lifecycle_admin.begin_close(case.request)
    before = _snapshot(case)
    with pytest.raises(_DENIALS), custody._custody_lock(case.directory, exclusive=True):
        custody._transition_exact_locked(
            case.directory,
            case.boundary.security,
            case.role,
            case.initial.credential_identity(),
            1,
            custody.SigningKeyLifecycle.REVOKED,
            close_authorization=authorization,
            operation_descriptor=case.request.descriptor,
        )
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("PENDING_CLOSED", 1)
    assert case.boundary.lifecycle_admin.execute(case.request)["phase"] == "COMPLETED"


@pytest.mark.parametrize("changed", ["operation_id", "lifecycle_generation"])
def test_genuine_authorization_refuses_stale_operation_or_generation(enrolled_case, changed):
    case = enrolled_case
    authorization = case.boundary.lifecycle_admin.begin_close(case.request)
    operation = case.request.operation
    operation[changed] = "unrelated-close" if changed == "operation_id" else 2
    descriptor = canonical_json_bytes(operation)
    before = _snapshot(case)
    stale_request = lifecycle.CustodyLifecycleRequest(
        case.directory, case.boundary.security, case.role, descriptor
    )
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        case.boundary.lifecycle_admin.mutate_custody(authorization, stale_request)
    assert _snapshot(case) == before
    with pytest.raises(_DENIALS), custody._custody_lock(case.directory, exclusive=True):
        custody._transition_exact_locked(
            case.directory,
            case.boundary.security,
            case.role,
            case.initial.credential_identity(),
            1,
            custody.SigningKeyLifecycle.REVOKED,
            close_authorization=authorization,
            operation_descriptor=descriptor,
        )
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("PENDING_CLOSED", 1)


def test_exact_enrolled_retained_credential_has_the_same_admission(enrolled_case, tmp_path):
    case = enrolled_case
    security = case.boundary.security
    alternate = tmp_path / "privileged-retained-layout-bootstrap"
    administrator = custody.NativeKeyringSigningSecretAdministrator(
        index_path=alternate / "keyring-index.json", security=security, role=case.role
    )
    successor = custody.provision_local_signing_authority(
        alternate, administrator, security=security, role=case.role
    )
    alternate_request = lifecycle.prepare_custody_lifecycle_request(
        case.boundary.lifecycle_deployment,
        alternate,
        security=security,
        role=case.role,
        administrative_operation_id=case.request.operation_id + "-unregistered-successor",
        target=custody.SigningKeyLifecycle.REVOKED,
        fence_key=case.key,
    )
    alternate_bytes = tuple(
        (path.name, path.read_bytes()) for path in sorted(alternate.rglob("*.json"))
    )
    before_provisioned_alias = _snapshot(case)
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        case.boundary.lifecycle_admin.begin_close(alternate_request)
    assert _snapshot(case) == before_provisioned_alias
    assert (
        tuple((path.name, path.read_bytes()) for path in sorted(alternate.rglob("*.json")))
        == alternate_bytes
    )
    retained = custody._retained_path(case.directory, case.role, case.initial.credential_identity())
    retained.parent.mkdir(mode=0o700)
    current = custody._record_path(case.directory, case.role)
    # Privileged fixture preparation preserves the exact already-enrolled
    # credential and supplies a genuinely provisioned distinct current record.
    current.rename(retained)
    current.write_bytes(custody._encode_record(successor))
    current.chmod(0o600)
    with custody._custody_lock(case.directory, exclusive=False):
        path, observed = custody._resolve_credential_record(
            case.directory, security, case.role, case.initial.credential_identity()
        )
    assert path == retained
    assert observed == case.initial
    before = _snapshot(case)
    for route in ("credential-wrapper", "internal-mutator", "record-replacer"):
        with pytest.raises(_DENIALS):
            _legacy(case, route)
        assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("ACTIVE", 1)
    current_bytes = current.read_bytes()
    authentic_close = case.boundary.lifecycle_admin.begin_close(case.request)
    before_alternate_mutation = _snapshot(case)
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        case.boundary.lifecycle_admin.mutate_custody(authentic_close, alternate_request)
    assert _snapshot(case) == before_alternate_mutation
    assert (
        tuple((path.name, path.read_bytes()) for path in sorted(alternate.rglob("*.json")))
        == alternate_bytes
    )
    assert case.boundary.lifecycle_admin.execute(case.request)["phase"] == "COMPLETED"
    assert current.read_bytes() == current_bytes
    with custody._custody_lock(case.directory, exclusive=False):
        assert custody._read_record(retained, security, case.role) == replace(
            case.initial,
            lifecycle_state=custody.SigningKeyLifecycle.REVOKED,
            lifecycle_generation=2,
        )
    assert existing._fence(case.boundary, case.key) == ("REVOKED", 2)


@pytest.mark.parametrize("damage", ["deleted", "corrupted"])
def test_missing_or_corrupt_evidence_never_restores_legacy_permissions(enrolled_case, damage):
    case = enrolled_case
    request = existing._request(case.boundary, case, target=custody.SigningKeyLifecycle.VERIFY_ONLY)
    admin = case.boundary.lifecycle_admin
    completion = admin.mutate_custody(admin.begin_close(request), request)
    journal = (
        case.directory
        / ".signing-lifecycle-operations"
        / (hashlib.sha256(request.operation_id.encode()).hexdigest() + ".json")
    )
    if damage == "deleted":
        journal.unlink()
    else:
        journal.write_bytes(b"{corrupted enrolled custody evidence")
    custody._fsync_directory(journal.parent)
    before = _snapshot(case)
    for route in ("current-wrapper", "credential-wrapper", "internal-mutator", "record-replacer"):
        with pytest.raises(_DENIALS):
            _legacy(case, route, generation=2)
        assert _snapshot(case) == before
    with pytest.raises(_DENIALS):
        admin.publish_completion(request, completion)
    assert _snapshot(case) == before
    assert existing._fence(case.boundary, case.key) == ("PENDING_CLOSED", 1)


def test_fresh_python_process_keeps_legacy_refusal(enrolled_case):
    case = enrolled_case
    before = _snapshot(case)
    script = """
import sys
from pathlib import Path
from bot_core import local_signing_custody as custody
from bot_core.root_proof_issuer_substrate import ProviderRole, SecurityProfile, SecurityProfileIdentity
security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, sys.argv[2])
try:
    custody.transition_local_signing_lifecycle(
        Path(sys.argv[1]), security=security, role=ProviderRole(sys.argv[3]),
        target=custody.SigningKeyLifecycle.REVOKED,
    )
except custody.LocalSigningCustodyError:
    print("legacy-denied")
else:
    raise AssertionError("cold process regained legacy lifecycle privileges")
"""
    process_ids = []
    for _ in range(2):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(case.directory),
                case.boundary.security.trust_domain,
                case.role.value,
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        process_ids.append(result.stdout.strip())
        assert _snapshot(case) == before
    assert process_ids == ["legacy-denied", "legacy-denied"]


@existing.requires_fork
def test_independent_legacy_process_waits_for_phase_b_then_is_denied(enrolled_case):
    case = enrolled_case
    value = case.boundary
    request = existing._request(value, case, target=custody.SigningKeyLifecycle.VERIFY_ONLY)
    value.lifecycle_admin.begin_close(request)
    context = multiprocessing.get_context("fork")
    intent_written = context.Event()
    release_mutator = context.Event()
    legacy_started = context.Event()
    legacy_finished = context.Event()
    reader, writer = context.Pipe(duplex=False)

    def durable_administrator():
        admin = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment)

        def hold(name):
            if name == "lifecycle_after_intent":
                intent_written.set()
                assert release_mutator.wait(timeout=10)

        completion = admin.mutate_custody(
            admin.resolve_close(request), request, fault_injector=hold
        )
        admin.publish_completion(request, completion)
        writer.send(("durable", os.getpid()))

    def legacy_administrator():
        legacy_started.set()
        try:
            # After waiting for B, the wrapper reloads VERIFY_ONLY/gen2. Its
            # attempted ->REVOKED is a legal local CAS absent the new guard.
            _legacy(case, "current-wrapper")
        except custody.LocalSigningCustodyError:
            writer.send(("legacy-denied", os.getpid()))
            legacy_finished.set()
        else:
            raise AssertionError("concurrent legacy mutation succeeded")

    durable = context.Process(target=durable_administrator)
    legacy = context.Process(target=legacy_administrator)
    durable.start()
    try:
        assert intent_written.wait(timeout=10)
        legacy.start()
        assert legacy_started.wait(timeout=10)
        assert not legacy_finished.wait(timeout=0.1)
    finally:
        release_mutator.set()
    existing._join(durable)
    existing._join(legacy)
    messages = [reader.recv(), reader.recv()]
    assert {message[0] for message in messages} == {"durable", "legacy-denied"}
    assert len({message[1] for message in messages}) == 2
    assert existing._fence(value, case.key) == ("VERIFY_ONLY", 2)
    assert existing._metadata(case) == replace(
        case.initial,
        lifecycle_state=custody.SigningKeyLifecycle.VERIFY_ONLY,
        lifecycle_generation=2,
    )
    assert value.lifecycle_admin.execute(request)["phase"] == "COMPLETED"


def test_enrolled_provisioning_is_idempotent_and_grants_no_successor(enrolled_case):
    case = enrolled_case
    security = case.boundary.security
    administrator = custody.NativeKeyringSigningSecretAdministrator(
        index_path=case.directory / "keyring-index.json", security=security, role=case.role
    )
    before = _snapshot(case)
    assert (
        custody.provision_local_signing_authority(
            case.directory, administrator, security=security, role=case.role
        )
        == case.initial
    )
    with pytest.raises(custody.LocalSigningProvisioningConflict):
        custody.provision_local_signing_authority(
            case.directory,
            administrator,
            security=security,
            role=case.role,
            private_seed=b"x" * 32,
        )
    assert _snapshot(case) == before
