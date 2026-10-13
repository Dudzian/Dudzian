"""DB-first lifecycle proofs with PostgreSQL 16, real custody and processes.

The expensive immutable INITIAL_BINDING lineage is shared. Every administrative
case owns a different genuinely provisioned signing credential, so no case
reactivates or rolls back custody. Only the existing fixture's OS-keyring seam is
replaced; PostgreSQL transactions, CAS, locks, metadata and fsync run unchanged.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, replace
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Jsonb

from bot_core import local_signing_custody as custody
from bot_core import postgresql_atomic_root_proof_commit as atomic_runtime
from bot_core import postgresql_custody_lifecycle as lifecycle
from bot_core.authenticated_issuer_history import _credential_material
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.postgresql_custody_lifecycle_qualification import (
    CustodyLifecycleQualificationError,
    migrate_custody_lifecycle_v2,
    qualify_custody_lifecycle_authority,
    requalify_existing_custody_lifecycle_v2,
)
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_root_proof_retention import _snapshot_data
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security import test_postgresql_atomic_root_proof_commit as atomic_tests
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_postgresql_freshness_authority_core import _restart_controlled_postgresql

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]
requires_fork = pytest.mark.skipif(os.name != "posix", reason="genuine fixture process inheritance")

lineage_monkeypatch = atomic_tests.lineage_monkeypatch
lineage_path = atomic_tests.lineage_path
lineage_challenge = atomic_tests.lineage_challenge
lineage_integration = atomic_tests.lineage_integration
lineage_package = atomic_tests.lineage_package
lineage_lppi = atomic_tests.lineage_lppi
lineage_active = atomic_tests.lineage_active
lineage_committed = atomic_tests.lineage_committed
lineage_cha = atomic_tests.lineage_cha
lineage_reserved = atomic_tests.lineage_reserved
module_native_keyring = atomic_tests.module_native_keyring
genuine_attempt = atomic_tests.genuine_attempt
module_atomic_commit_boundary = atomic_tests.module_atomic_commit_boundary


@pytest.fixture(scope="module")
def lifecycle_boundary(module_atomic_commit_boundary, lineage_path):
    value = module_atomic_commit_boundary
    cases = []
    bindings = []
    for index in range(32):
        role = (
            ProviderRole.ROOT_PROOF_SIGNING
            if index % 2 == 0
            else ProviderRole.HISTORY_ATTESTATION_SIGNING
        )
        directory = lineage_path / f"lifecycle-custody-{index}"
        directory.mkdir(mode=0o700)
        administrator = custody.NativeKeyringSigningSecretAdministrator(
            index_path=directory / "keyring-index.json", security=value.security, role=role
        )
        metadata = custody.provision_local_signing_authority(
            directory, administrator, security=value.security, role=role
        )
        provider_type = (
            custody.LocalRootProofSigningProvider
            if role is ProviderRole.ROOT_PROOF_SIGNING
            else custody.LocalHistoryAttestationSigningProvider
        )
        provider = provider_type(
            directory, keyring_index_path=directory / "keyring-index.json", security=value.security
        )
        snapshot = provider.credential_snapshot()
        material = (
            _snapshot_data(snapshot)
            if role is ProviderRole.ROOT_PROOF_SIGNING
            else _credential_material(snapshot.credential_identity)
        )
        key = atomic_runtime._fence_key(role.value, canonical_json_bytes(material))
        atomic_tests._call_sql(value, "provision_fence", (key, role.value, 1), admin=True)
        atomic_tests._call_sql(value, "transition_fence", (key, 1, "ACTIVE", 1), admin=True)
        bindings.append(
            lifecycle.prepare_custody_lifecycle_binding(
                value.atomic_deployment.identity,
                directory,
                security=value.security,
                role=role,
                fence_key=key,
            )
        )
        cases.append(SimpleNamespace(directory=directory, role=role, key=key, initial=metadata))
    for role, key in (
        (ProviderRole.ROOT_PROOF_SIGNING, value.fences.root_key),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING, value.fences.history_key),
    ):
        bindings.append(
            lifecycle.prepare_custody_lifecycle_binding(
                value.atomic_deployment.identity,
                value.directory,
                security=value.security,
                role=role,
                fence_key=key,
            )
        )
    deployment = migrate_custody_lifecycle_v2(
        PostgreSQLConnectionConfig(BASE_DSN),
        value.atomic_deployment,
        atomic_tests._connection(value.setup.test_admin_role),
        tuple(bindings),
    )
    value.lifecycle_deployment = deployment
    value.lifecycle_admin = lifecycle.PostgreSQLCustodyLifecycleAdministrator(deployment)
    value.atomic_deployment = deployment.atomic_deployment
    value.commit = atomic_runtime.PostgreSQLAtomicRootProofCommit(deployment.atomic_deployment)
    value.lifecycle_cases = iter(cases)
    return value


@pytest.fixture
def lifecycle_case(lifecycle_boundary):
    case = next(lifecycle_boundary.lifecycle_cases)
    case.boundary = lifecycle_boundary
    case.request = _request(lifecycle_boundary, case)
    return case


def _request(value, case, *, target=custody.SigningKeyLifecycle.REVOKED, operation_id=None):
    return lifecycle.prepare_custody_lifecycle_request(
        value.lifecycle_deployment,
        case.directory,
        security=value.security,
        role=case.role,
        administrative_operation_id=operation_id or "stage9-" + uuid.uuid4().hex,
        target=target,
        fence_key=case.key,
    )


def _fence(value, key):
    with psycopg.connect(BASE_DSN) as observer:
        return observer.execute(
            sql.SQL("SELECT state,generation FROM {}.fences WHERE identity=%s").format(
                sql.Identifier(value.setup.schema)
            ),
            (key,),
        ).fetchone()


def _metadata(case):
    with custody._custody_lock(case.directory, exclusive=False):
        return custody._read_record(
            custody._record_path(case.directory, case.role), case.boundary.security, case.role
        )


def _join(process, *, expected=0):
    process.join(timeout=10)
    if process.is_alive():
        process.kill()
        process.join(timeout=5)
        pytest.fail("independent lifecycle process deadlocked")
    assert process.exitcode == expected


def _wait_until(predicate, message):
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    pytest.fail(message)


@pytest.mark.parametrize("target", ["VERIFY_ONLY", "REVOKED"])
def test_genuine_db_first_transition_retains_exact_identity_and_retry(lifecycle_case, target):
    case = lifecycle_case
    value = case.boundary
    request = _request(value, case, target=custody.SigningKeyLifecycle(target))
    authorization = value.lifecycle_admin.begin_close(request)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert _metadata(case) == case.initial
    completion = value.lifecycle_admin.mutate_custody(authorization, request)
    assert _metadata(case) == replace(
        case.initial, lifecycle_state=custody.SigningKeyLifecycle(target), lifecycle_generation=2
    )
    assert value.lifecycle_admin.reconcile_completion(request) is None
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    published = value.lifecycle_admin.publish_completion(request, completion)
    assert published["phase"] == "COMPLETED"
    assert _fence(value, case.key) == (target, 2)
    assert value.lifecycle_admin.execute(request) == published
    assert value.lifecycle_admin.publish_completion(request, completion) == published
    assert _metadata(case).public_key == case.initial.public_key
    assert _metadata(case).credential_identity() == case.initial.credential_identity()


def test_verify_only_can_only_advance_to_revoked(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    first = _request(value, case, target=custody.SigningKeyLifecycle.VERIFY_ONLY)
    value.lifecycle_admin.execute(first)
    second = _request(value, case)
    value.lifecycle_admin.execute(second)
    assert _fence(value, case.key) == ("REVOKED", 3)
    assert _metadata(case).lifecycle_generation == 3
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        _request(value, case, target=custody.SigningKeyLifecycle.ACTIVE)


@pytest.mark.parametrize("changed", ["operation_id", "target_state", "lifecycle_generation"])
def test_pending_close_rejects_other_ids_changed_replay_and_stale_generation(
    lifecycle_case, changed
):
    case = lifecycle_case
    value = case.boundary
    request = case.request
    value.lifecycle_admin.begin_close(request)
    payload = request.operation
    payload[changed] = {
        "operation_id": "different-" + uuid.uuid4().hex,
        "target_state": "VERIFY_ONLY",
        "lifecycle_generation": 2,
    }[changed]
    with pytest.raises(psycopg.Error):
        atomic_tests._call_sql(value, "begin_lifecycle_close", (Jsonb(payload),), admin=True)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert _metadata(case) == case.initial


def test_caller_dto_and_arbitrary_completion_mac_cannot_publish(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    authorization = value.lifecycle_admin.begin_close(case.request)
    with pytest.raises(custody.LocalSigningLifecycleError):
        value.lifecycle_admin.publish_completion(
            case.request, SimpleNamespace(evidence_bytes=b"{}")
        )
    with pytest.raises(psycopg.Error):
        atomic_tests._call_sql(
            value,
            "complete_lifecycle_close",
            (case.request.operation_id, b"{}", b"x" * 32),
            admin=True,
        )
    with pytest.raises(lifecycle.CustodyLifecycleConflict):
        value.lifecycle_admin.mutate_custody(
            lifecycle.DurableCloseAuthorization(case.request.descriptor), case.request
        )
    assert authorization.descriptor == case.request.descriptor
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert _metadata(case) == case.initial


@requires_fork
@pytest.mark.parametrize(
    "cut",
    ["lifecycle_after_intent", "lifecycle_after_custody_transition", "lifecycle_after_completion"],
)
def test_real_admin_process_crash_during_custody_recovers_exact_once(lifecycle_case, cut):
    case = lifecycle_case
    value = case.boundary
    value.lifecycle_admin.begin_close(case.request)

    def die_at_cut():
        admin = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment)
        authorization = admin.resolve_close(case.request)

        def die(name):
            if name == cut:
                os._exit(73)

        admin.mutate_custody(authorization, case.request, fault_injector=die)
        os._exit(74)

    process = multiprocessing.get_context("fork").Process(target=die_at_cut)
    process.start()
    _join(process, expected=73)
    observed = _metadata(case)
    expected_generation = 1 if cut == "lifecycle_after_intent" else 2
    assert observed.lifecycle_generation == expected_generation
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert value.lifecycle_admin.reconcile_completion(case.request) is None
    restarted = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment)
    result = restarted.execute(case.request)
    assert result["phase"] == "COMPLETED"
    assert _metadata(case).lifecycle_generation == 2
    assert _fence(value, case.key) == ("REVOKED", 2)
    assert restarted.execute(case.request) == result


@requires_fork
def test_crash_before_close_commit_rolls_back_without_custody_mutation(lifecycle_case):
    case = lifecycle_case
    value = case.boundary

    def uncommitted_client():
        def die(name, connection):
            if name == "after_begin_lifecycle_close_write":
                os._exit(75)

        lifecycle._persistence_cut = die
        value.lifecycle_admin.begin_close(case.request)

    process = multiprocessing.get_context("fork").Process(target=uncommitted_client)
    process.start()
    _join(process, expected=75)
    assert value.lifecycle_admin.resolve_close(case.request) is None
    assert _fence(value, case.key) == ("ACTIVE", 1)
    assert _metadata(case) == case.initial


@pytest.mark.parametrize("phase", ["begin_lifecycle_close", "complete_lifecycle_close"])
def test_actual_native_commit_response_loss_is_resolved_before_retry(
    lifecycle_case, monkeypatch, phase
):
    case = lifecycle_case
    value = case.boundary
    admin = value.lifecycle_admin
    calls = []

    def disconnect(name, connection):
        if name == "after_" + phase + "_write":
            with socket.fromfd(
                connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM
            ) as wire:
                wire.shutdown(socket.SHUT_RD)
            calls.append(connection.info.backend_pid)

    completion = None
    if phase == "complete_lifecycle_close":
        completion = admin.mutate_custody(admin.begin_close(case.request), case.request)
    monkeypatch.setattr(lifecycle, "_persistence_cut", disconnect)
    with pytest.raises(lifecycle.CustodyLifecycleIndeterminate):
        if completion is None:
            admin.begin_close(case.request)
        else:
            admin.publish_completion(case.request, completion)
    assert len(calls) == 1
    monkeypatch.setattr(lifecycle, "_persistence_cut", lambda *_: None)
    _wait_until(lambda: admin.resolve_close(case.request) is not None, "native COMMIT not retained")
    if completion is None:
        assert _metadata(case) == case.initial
        assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    result = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment).execute(
        case.request
    )
    assert result["phase"] == "COMPLETED"
    assert _metadata(case).lifecycle_generation == 2
    assert _fence(value, case.key) == ("REVOKED", 2)


def test_postgresql_restart_preserves_pending_close_and_local_completion(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    admin = value.lifecycle_admin
    authorization = admin.begin_close(case.request)
    _restart_controlled_postgresql()
    assert admin.resolve_close(case.request) is not None
    assert _metadata(case) == case.initial
    completion = admin.mutate_custody(authorization, case.request)
    _restart_controlled_postgresql()
    assert admin.reconcile_completion(case.request) is None
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    result = admin.publish_completion(case.request, completion)
    _restart_controlled_postgresql()
    assert admin.reconcile_completion(case.request) == result
    assert _fence(value, case.key) == ("REVOKED", 2)


@requires_fork
def test_admin_dies_after_durable_close_before_local_mutation(lifecycle_case):
    case = lifecycle_case
    value = case.boundary

    def close_then_die():
        def die(name, connection):
            if name == "after_begin_lifecycle_close_commit":
                os._exit(77)

        lifecycle._persistence_cut = die
        value.lifecycle_admin.begin_close(case.request)

    process = multiprocessing.get_context("fork").Process(target=close_then_die)
    process.start()
    _join(process, expected=77)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert _metadata(case) == case.initial
    restarted = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment)
    assert restarted.execute(case.request)["phase"] == "COMPLETED"
    assert _metadata(case).lifecycle_generation == 2


def test_missing_completion_journal_never_reopens_eligibility(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    completion = value.lifecycle_admin.mutate_custody(
        value.lifecycle_admin.begin_close(case.request), case.request
    )
    journal = (
        case.directory
        / ".signing-lifecycle-operations"
        / (hashlib.sha256(case.request.operation_id.encode()).hexdigest() + ".json")
    )
    journal.unlink()
    custody._fsync_directory(journal.parent)
    with pytest.raises(custody.LocalSigningCustodyError):
        value.lifecycle_admin.publish_completion(case.request, completion)
    with pytest.raises(custody.LocalSigningCustodyError):
        value.lifecycle_admin.execute(case.request)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert _metadata(case).lifecycle_state is custody.SigningKeyLifecycle.REVOKED
    assert value.lifecycle_admin.reconcile_completion(case.request) is None


def test_unavailable_admin_database_cannot_begin_custody_mutation(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    authorization = value.lifecycle_admin.begin_close(case.request)
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        administrator.execute(
            sql.SQL("ALTER ROLE {} NOLOGIN").format(sql.Identifier(value.setup.test_admin_role))
        )
        try:
            with pytest.raises(lifecycle.CustodyLifecycleUnavailable):
                value.lifecycle_admin.mutate_custody(authorization, case.request)
            assert _metadata(case) == case.initial
            assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
        finally:
            administrator.execute(
                sql.SQL("ALTER ROLE {} LOGIN").format(sql.Identifier(value.setup.test_admin_role))
            )


def test_unavailable_local_custody_cannot_publish_completion(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    authorization = value.lifecycle_admin.begin_close(case.request)
    path = custody._record_path(case.directory, case.role)
    unavailable = path.with_suffix(".unavailable")
    path.rename(unavailable)
    try:
        with pytest.raises(custody.LocalSigningCustodyError):
            value.lifecycle_admin.mutate_custody(authorization, case.request)
        assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
        assert value.lifecycle_admin.reconcile_completion(case.request) is None
    finally:
        unavailable.rename(path)


@requires_fork
def test_resolution_waits_for_live_backend_after_python_client_sigkill(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)

    def dying_client():
        with psycopg.connect(value.lifecycle_deployment.admin_connection.dsn) as connection:
            connection.execute(
                sql.SQL("SELECT {}.begin_lifecycle_close(%s)").format(
                    sql.Identifier(value.setup.schema)
                ),
                (Jsonb(case.request.operation),),
            ).fetchone()
            writer.send(connection.info.backend_pid)
            connection.execute("SELECT pg_catalog.pg_sleep(3)")
            connection.commit()

    process = context.Process(target=dying_client)
    process.start()
    assert reader.poll(10)
    backend_pid = reader.recv()

    def backend_sleeping():
        with psycopg.connect(BASE_DSN, autocommit=True) as observer:
            row = observer.execute(
                "SELECT wait_event FROM pg_catalog.pg_stat_activity WHERE pid=%s", (backend_pid,)
            ).fetchone()
            return row == ("PgSleep",)

    _wait_until(backend_sleeping, "backend did not enter independent pg_sleep")
    os.kill(process.pid, signal.SIGKILL)
    _join(process, expected=-signal.SIGKILL)
    assert backend_sleeping()
    resolved = []
    errors = []

    def resolve():
        try:
            resolved.append(value.lifecycle_admin.resolve_close(case.request))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=resolve)
    thread.start()
    atomic_tests._wait_for_role_lock(value.setup.test_admin_role)
    assert not resolved and not errors
    assert _metadata(case) == case.initial
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert not errors
    assert resolved == [None]
    assert _fence(value, case.key) == ("ACTIVE", 1)


def test_writer_and_unrelated_admin_cannot_escalate_or_edit_lifecycle(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    value.lifecycle_admin.begin_close(case.request)
    for role in (value.setup.issuer_writer_role, value.genuine.pair.setup.requester.admin_role):
        with psycopg.connect(atomic_tests._connection(role).dsn) as connection:
            with pytest.raises(InsufficientPrivilege):
                connection.execute(
                    sql.SQL("SELECT {}.begin_lifecycle_close(%s)").format(
                        sql.Identifier(value.setup.schema)
                    ),
                    (Jsonb(case.request.operation),),
                )
        with psycopg.connect(atomic_tests._connection(role).dsn) as connection:
            with pytest.raises(InsufficientPrivilege):
                connection.execute(
                    sql.SQL("SET ROLE {}").format(sql.Identifier(value.setup.schema_owner_role))
                )
    with psycopg.connect(value.writer_connection.dsn) as writer_connection:
        with pytest.raises(InsufficientPrivilege):
            writer_connection.execute(
                sql.SQL("UPDATE {}.fences SET state='ACTIVE' WHERE identity=%s").format(
                    sql.Identifier(value.setup.schema)
                ),
                (case.key,),
            )
    with psycopg.connect(value.lifecycle_deployment.admin_connection.dsn) as connection:
        qualify_custody_lifecycle_authority(
            connection, value.lifecycle_deployment, expected_role="admin"
        )
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)


def test_v2_admin_cannot_call_legacy_fence_activation(lifecycle_case):
    case = lifecycle_case
    value = case.boundary
    value.lifecycle_admin.begin_close(case.request)
    for function, arguments in (
        ("provision_fence", (case.key + "-replacement", case.role.value, 1)),
        ("transition_fence", (case.key, 1, "ACTIVE", 2)),
    ):
        with pytest.raises(InsufficientPrivilege):
            atomic_tests._call_sql(value, function, arguments, admin=True)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)


@pytest.mark.parametrize("phase", ["PENDING_CLOSED", "CUSTODY_CRASH", "COMPLETED"])
def test_fresh_admin_process_reconstructs_exact_deployment_and_retry(lifecycle_case, phase):
    case = lifecycle_case
    value = case.boundary
    value.lifecycle_admin.begin_close(case.request)
    if phase == "CUSTODY_CRASH":
        if os.name != "posix":
            pytest.skip("genuine crashed custody process inheritance requires POSIX")

        def crash_after_local_mutation():
            authorization = value.lifecycle_admin.resolve_close(case.request)

            def die(name):
                if name == "lifecycle_after_custody_transition":
                    os._exit(78)

            value.lifecycle_admin.mutate_custody(authorization, case.request, fault_injector=die)

        process = multiprocessing.get_context("fork").Process(target=crash_after_local_mutation)
        process.start()
        _join(process, expected=78)
        assert value.lifecycle_admin.reconcile_completion(case.request) is None
        assert _metadata(case).lifecycle_state is custody.SigningKeyLifecycle.REVOKED
    if phase == "COMPLETED":
        value.lifecycle_admin.execute(case.request)
    command = """
import json,sys
from pathlib import Path
from bot_core.postgresql_custody_lifecycle_qualification import requalify_existing_custody_lifecycle_v2
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_atomic_root_proof_qualification import PostgreSQLAtomicRootProofPhysicalIdentity
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_custody_lifecycle import CustodyLifecycleRequest,PostgreSQLCustodyLifecycleAdministrator
from bot_core.root_proof_issuer_substrate import ProviderRole,SecurityProfile,SecurityProfileIdentity
value=json.loads(sys.argv[1])
deployment=requalify_existing_custody_lifecycle_v2(
 PostgreSQLConnectionConfig(value['bootstrap']),
 PostgreSQLAtomicRootProofProvisioning(**value['provisioning']),
 PostgreSQLConnectionConfig(value['writer']),PostgreSQLConnectionConfig(value['admin']),
 expected_identity=PostgreSQLAtomicRootProofPhysicalIdentity(**value['identity']))
request=CustodyLifecycleRequest(Path(value['directory']),
 SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL,value['trust_domain']),
 ProviderRole(value['role']),bytes.fromhex(value['descriptor']))
admin=PostgreSQLCustodyLifecycleAdministrator(deployment)
first=admin.execute(request)
assert admin.execute(request)==first
print(json.dumps({'phase':first['phase'],'operation_id':first['operation']['operation_id']}))
"""
    arguments = json.dumps(
        {
            "bootstrap": BASE_DSN,
            "admin": value.lifecycle_deployment.admin_connection.dsn,
            "writer": value.writer_connection.dsn,
            "provisioning": asdict(value.setup),
            "identity": asdict(value.atomic_deployment.identity),
            "directory": str(case.directory),
            "trust_domain": value.security.trust_domain,
            "role": case.role.value,
            "descriptor": case.request.descriptor.hex(),
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", command, arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout) == {
        "phase": "COMPLETED",
        "operation_id": case.request.operation_id,
    }
    assert _metadata(case).lifecycle_generation == 2
    assert _fence(value, case.key) == ("REVOKED", 2)


@pytest.mark.parametrize("change", ["function-acl", "function-source", "membership", "schema"])
def test_v2_catalog_drift_refuses_live_qualification_and_cold_reconstruction(
    lifecycle_case, change
):
    case = lifecycle_case
    value = case.boundary
    function = sql.SQL("{}.begin_lifecycle_close(jsonb)").format(sql.Identifier(value.setup.schema))
    writer = sql.Identifier(value.setup.issuer_writer_role)
    owner = sql.Identifier(value.setup.schema_owner_role)
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        if change == "function-acl":
            apply = sql.SQL("GRANT EXECUTE ON FUNCTION {} TO {}").format(function, writer)
            restore = sql.SQL("REVOKE EXECUTE ON FUNCTION {} FROM {}").format(function, writer)
        elif change == "membership":
            apply = sql.SQL("GRANT {} TO {}").format(owner, writer)
            restore = sql.SQL("REVOKE {} FROM {}").format(owner, writer)
        elif change == "schema":
            table = sql.Identifier(value.setup.schema, "unreviewed_lifecycle_relation")
            apply = sql.SQL("CREATE TABLE {} (value boolean)").format(table)
            restore = sql.SQL("DROP TABLE {}").format(table)
        else:
            original = administrator.execute(
                "SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure(%s))",
                (value.setup.schema + ".begin_lifecycle_close(jsonb)",),
            ).fetchone()[0]
            apply = sql.SQL(original.replace("BEGIN", "BEGIN\n-- unreviewed lifecycle change", 1))
            restore = sql.SQL(original)
        administrator.execute(apply)
        try:
            with psycopg.connect(value.lifecycle_deployment.admin_connection.dsn) as connection:
                with pytest.raises(CustodyLifecycleQualificationError):
                    qualify_custody_lifecycle_authority(
                        connection, value.lifecycle_deployment, expected_role="admin"
                    )
            with pytest.raises(CustodyLifecycleQualificationError):
                requalify_existing_custody_lifecycle_v2(
                    PostgreSQLConnectionConfig(BASE_DSN),
                    value.setup,
                    value.writer_connection,
                    value.lifecycle_deployment.admin_connection,
                    expected_identity=value.atomic_deployment.identity,
                )
            assert _metadata(case) == case.initial
        finally:
            administrator.execute(restore)
    with psycopg.connect(value.lifecycle_deployment.admin_connection.dsn) as connection:
        qualify_custody_lifecycle_authority(
            connection, value.lifecycle_deployment, expected_role="admin"
        )
    assert _fence(value, case.key) == ("ACTIVE", 1)


def _restore_legacy_approval_for_fixture(value, package):
    """Model pre-migration legacy approval, never alter a lifecycle fence.

    Only this isolated test's bootstrap role can suspend immutability. Restore
    exact trigger modes before the genuine writer qualifies and attempts COMMIT.
    """
    with psycopg.connect(BASE_DSN) as administrator:
        table = sql.Identifier(value.setup.schema, "approved_commits")
        triggers = administrator.execute(
            "SELECT tgname,tgenabled FROM pg_catalog.pg_trigger "
            "WHERE tgrelid=pg_catalog.to_regclass(%s) AND NOT tgisinternal",
            (value.setup.schema + ".approved_commits",),
        ).fetchall()
        administrator.execute(sql.SQL("ALTER TABLE {} DISABLE TRIGGER USER").format(table))
        administrator.execute(
            sql.SQL("UPDATE {} SET package=%s WHERE attempt_key=%s").format(table),
            (Jsonb(package), value.preparation.locator.attempt),
        )
        modes = {"O": "ENABLE", "A": "ENABLE ALWAYS", "R": "ENABLE REPLICA", "D": "DISABLE"}
        for name, mode in triggers:
            administrator.execute(
                sql.SQL("ALTER TABLE {} {} TRIGGER {}").format(
                    table, sql.SQL(modes[mode]), sql.Identifier(name)
                )
            )
        administrator.commit()


@pytest.mark.parametrize("role", ["ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING"])
def test_legacy_alias_approval_cannot_bind_under_another_credentials_active_fence(
    lifecycle_case, role
):
    case = lifecycle_case
    value = case.boundary
    if case.role.value != role:
        case = next(value.lifecycle_cases)
        assert case.role.value == role
    assert _fence(value, case.key) == ("ACTIVE", 1)
    with psycopg.connect(BASE_DSN) as observer:
        original = observer.execute(
            sql.SQL("SELECT package FROM {}.approved_commits WHERE attempt_key=%s").format(
                sql.Identifier(value.setup.schema)
            ),
            (value.preparation.locator.attempt,),
        ).fetchone()[0]
    changed = dict(original)
    field = "root_fence_key" if role == "ROOT_PROOF_SIGNING" else "history_fence_key"
    assert case.key != original[field]
    changed[field] = case.key
    before = atomic_tests._snapshot(value)
    _restore_legacy_approval_for_fixture(value, changed)
    try:
        with pytest.raises(psycopg.errors.CheckViolation) as refused:
            atomic_tests._call_sql(value, "commit_root_proof", (value.preparation.locator.attempt,))
        assert refused.value.diag.message_primary == (
            "actual retained signer custody fence cannot be aliased"
        )
        assert atomic_tests._snapshot(value) == before
        assert _fence(value, case.key) == ("ACTIVE", 1)
        assert (
            value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
        )
    finally:
        _restore_legacy_approval_for_fixture(value, original)


@requires_fork
def test_independent_issuer_and_revoke_processes_have_one_durable_order(lifecycle_boundary):
    value = lifecycle_boundary
    case = SimpleNamespace(
        directory=value.directory, role=ProviderRole.ROOT_PROOF_SIGNING, key=value.fences.root_key
    )
    request = _request(value, case, target=custody.SigningKeyLifecycle.VERIFY_ONLY)
    context = multiprocessing.get_context("fork")
    entered = context.Event()
    release = context.Event()
    pending = context.Event()
    refused = context.Event()
    verify_only = context.Event()
    verify_denied = context.Event()
    revoked = context.Event()
    reader, writer = context.Pipe(duplex=False)

    def issuer():
        def hold(name, connection):
            if name == "before_commit":
                entered.set()
                assert release.wait(timeout=10)

        atomic_runtime._persistence_cut = hold
        runtime = atomic_runtime.PostgreSQLAtomicRootProofCommit(value.atomic_deployment)
        decision = runtime.commit(value.preparation)
        writer.send(("issuer", os.getpid(), decision.marker["decision_id"]))

    def revoker():
        assert entered.wait(timeout=10)
        admin = lifecycle.PostgreSQLCustodyLifecycleAdministrator(value.lifecycle_deployment)
        authorization = admin.begin_close(request)
        pending.set()
        assert refused.wait(timeout=10)
        completion = admin.mutate_custody(authorization, request)
        admin.publish_completion(request, completion)
        verify_only.set()
        assert verify_denied.wait(timeout=10)
        final_request = _request(value, case, operation_id=request.operation_id + "-revoke")
        admin.execute(final_request)
        revoked.set()
        writer.send(("admin", os.getpid(), final_request.operation_id))

    def closed_first_issuer():
        runtime = atomic_runtime.PostgreSQLAtomicRootProofCommit(value.atomic_deployment)
        for phase, generation, ready, finished in (
            ("PENDING_CLOSED", 1, pending, refused),
            ("VERIFY_ONLY", 2, verify_only, verify_denied),
            ("REVOKED", 3, revoked, None),
        ):
            assert ready.wait(timeout=10)
            assert _fence(value, case.key) == (phase, generation)
            try:
                runtime.commit(value.secondary.preparation)
            except atomic_runtime.AtomicRootProofCommitConflict as error:
                # This candidate has an old history predecessor after the first
                # winner. Prove refusal at eligibility before any history CAS.
                cause = error.__context__
                assert isinstance(cause, psycopg.errors.CheckViolation)
                assert cause.diag.message_primary == "DB-authoritative lifecycle fence closed"
                writer.send(("refused", os.getpid(), phase, cause.diag.message_primary))
                if finished is not None:
                    finished.set()
            else:
                os._exit(76)

    first = context.Process(target=issuer)
    second = context.Process(target=revoker)
    first.start()
    assert entered.wait(timeout=10)
    second.start()
    try:
        atomic_tests._wait_for_role_lock(value.setup.test_admin_role)
        assert not revoked.is_set()
        assert (
            value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
        )
    finally:
        release.set()
    _join(first)
    assert pending.wait(timeout=10)
    assert _fence(value, case.key) == ("PENDING_CLOSED", 1)
    assert value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.ACTIVE
    third = context.Process(target=closed_first_issuer)
    third.start()
    _join(third)
    _join(second)
    messages = [reader.recv() for _ in range(5)]
    assert {part[0] for part in messages} == {"issuer", "admin", "refused"}
    assert {part[2] for part in messages if part[0] == "refused"} == {
        "PENDING_CLOSED",
        "VERIFY_ONLY",
        "REVOKED",
    }
    assert len({part[1] for part in messages}) == 3
    decision = value.commit.reconcile(value.preparation.locator)
    atomic_tests._assert_committed(value, decision)
    assert _fence(value, case.key) == ("REVOKED", 3)
    assert value.signer.credential_snapshot().lifecycle_state is custody.SigningKeyLifecycle.REVOKED
    assert value.commit.commit(value.preparation) == decision

    with psycopg.connect(BASE_DSN) as observer:
        assert observer.execute(
            sql.SQL("SELECT count(*) FROM {}.decisions").format(sql.Identifier(value.setup.schema))
        ).fetchone() == (1,)
    assert value.lifecycle_admin.reconcile_completion(request)["phase"] == "COMPLETED"
