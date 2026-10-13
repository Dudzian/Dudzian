"""Crash-atomic Stage 9 write-set proofs against a genuine PostgreSQL 16.

The immutable INITIAL_BINDING lineage, genuine requester/claimant credentials,
authenticated signed CHA request, RESERVED row and signed STAGED bytes are
prepared once. Each case asserts real durable records before privileged test
administration restores the isolated fixture baseline. Faults interrupt actual
PostgreSQL writes or the native connection; no test replaces the transaction.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege

from bot_core import (
    postgresql_atomic_root_proof_commit as commit_runtime,
    postgresql_atomic_root_proof_migration as migration,
    postgresql_atomic_root_proof_qualification as qualification,
    postgresql_atomic_root_proof_schema as atomic_schema,
)
from bot_core.authenticated_issuer_history import attest_head
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from tests.security import (
    test_root_proof_atomic_composition_integration as existing_atomic,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.atomic_root_proof_fixture import prepare_second_atomic_candidate
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_postgresql_freshness_authority_core import _restart_controlled_postgresql

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]

lineage_monkeypatch = existing_atomic.lineage_monkeypatch
lineage_path = existing_atomic.lineage_path
lineage_challenge = existing_atomic.lineage_challenge
lineage_integration = existing_atomic.lineage_integration
lineage_package = existing_atomic.lineage_package
lineage_lppi = existing_atomic.lineage_lppi
lineage_active = existing_atomic.lineage_active
lineage_committed = existing_atomic.lineage_committed
lineage_cha = existing_atomic.lineage_cha
lineage_reserved = existing_atomic.lineage_reserved
module_native_keyring = existing_atomic.module_native_keyring


@pytest.fixture(scope="module")
def genuine_attempt(lineage_reserved, module_native_keyring, lineage_path, lineage_monkeypatch):
    tmp_path = lineage_path / "immutable-atomic-signed-attempt"
    tmp_path.mkdir()
    monkeypatch = lineage_monkeypatch
    yield from preflight_tests.installed_signed_attempt.__wrapped__(
        lineage_reserved,
        module_native_keyring,
        tmp_path,
        monkeypatch,
        SimpleNamespace(param=("requester-key", "claimant-key")),
    )


def _connection(role):
    return PostgreSQLConnectionConfig(make_conninfo(BASE_DSN, user=role))


def _call_sql(value, name, args, *, admin=False, commit=True):
    """One real connection owns every coordinator SQL statement and COMMIT."""
    config = _connection(value.setup.test_admin_role) if admin else value.writer_connection
    with psycopg.connect(config.dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        query = sql.SQL("SELECT {}.{}({})").format(
            sql.Identifier(value.setup.schema),
            sql.Identifier(name),
            sql.SQL(",").join(sql.Placeholder() for _ in args),
        )
        row = conn.execute(query, args).fetchone()
        if commit:
            conn.commit()
        else:
            conn.rollback()
        return None if row is None else row[0]


def _snapshot(value):
    """Read committed physical records through a fresh administrative backend."""
    pairs = (
        (value.genuine.entitlement_admin._schema, "lineages"),
        (value.genuine.entitlement_admin._schema, "history"),
        (value.intent_setup.schema, "intents"),
        (value.retention_setup.schema, "prepared"),
        (value.history_boundary.history_setup.schema, "streams"),
        (value.history_boundary.history_setup.schema, "records"),
        (value.history_boundary.history_setup.schema, "attestations"),
        (value.setup.schema, "decisions"),
    )
    orders = {
        "history": "1,2,3,4",
        "lineages": "1,2,3",
        "intents": "1,2",
        "prepared": "1,2",
        "streams": "1",
        "records": "1,2",
        "attestations": "1,2",
        "decisions": "1",
    }
    with psycopg.connect(BASE_DSN) as conn:
        return {
            pair: conn.execute(
                sql.SQL("SELECT * FROM {} ORDER BY {}").format(
                    sql.Identifier(*pair), sql.SQL(orders[pair[1]])
                )
            ).fetchall()
            for pair in pairs
        }


@pytest.fixture(scope="module")
def module_atomic_commit_boundary(
    genuine_attempt, lineage_active, lineage_path: Path, lineage_monkeypatch
):
    monkeypatch = lineage_monkeypatch
    path = lineage_path / "atomic-boundary"
    path.mkdir()
    source = existing_atomic.atomic_boundary.__wrapped__(genuine_attempt, path, monkeypatch)
    value = next(source)
    token = uuid.uuid4().hex[:12]
    installed = False
    try:
        value.reserved = value.intent.reserve(value.genuine.binding, value.genuine.opaque)
        value.prepared = existing_atomic._stage_reserved(value)
        value.preparation = commit_runtime.prepare_atomic_root_proof_commit(
            value.genuine.binding,
            value.genuine.opaque,
            signing_intent=value.intent,
            retention=value.retention,
            issuer_authority=value.genuine.authority,
            history=value.history,
        )
        value.secondary = prepare_second_atomic_candidate(
            value, lineage_active, lineage_path / "second-atomic-candidate", monkeypatch
        )
        value.head = attest_head(value.preparation.history_record, value.history_boundary.signer)
        value.secondary_head = attest_head(
            value.secondary.preparation.history_record, value.history_boundary.signer
        )
        value.secondary.head = value.secondary_head
        value.setup = atomic_schema.PostgreSQLAtomicRootProofProvisioning(
            f"arc_{token}",
            f"arco_{token}",
            f"arcw_{token}",
            f"arca_{token}",
            value.genuine.entitlement_admin._schema,
            value.intent_setup.schema,
            value.retention_setup.schema,
            value.history_boundary.history_setup.schema,
            value.auth.trust_domain,
            f"isolated-atomic-test-{token}",
            requester_schema=value.genuine.pair.requester._schema,
            claimant_schema=value.genuine.pair.claimant._schema,
        )
        value.writer_connection = _connection(value.setup.issuer_writer_role)
        legacy = migration.PostgreSQLAtomicRootProofLegacyConnections(
            value.genuine.authority.entitlement_registry._connection,
            existing_atomic._provider_pin(value.intent).connection,
            existing_atomic._provider_pin(value.retention).connection,
            value.history._connection,
            requester_runtime=value.genuine.pair.requester._connection,
            claimant_runtime=value.genuine.pair.claimant._connection,
        )
        value.atomic_deployment = migration.administer_atomic_root_proof_v1(
            PostgreSQLConnectionConfig(BASE_DSN),
            value.writer_connection,
            value.setup,
            legacy_connections=legacy,
        )
        installed = True
        value.commit = commit_runtime.PostgreSQLAtomicRootProofCommit(
            value.atomic_deployment, admin_connection=_connection(value.setup.test_admin_role)
        )
        value.fences = commit_runtime.AtomicRootProofCommitFences(
            value.preparation.root_fence_key,
            value.prepared.credential_snapshot.lifecycle_generation,
            commit_runtime.atomic_history_fence_key(value.head),
            value.history_boundary.signer.credential_snapshot().lifecycle_generation,
            value.preparation.trust_fence_key,
            1,
        )
        for key, semantic_role in (
            (value.fences.root_key, "ROOT_PROOF_SIGNING"),
            (value.fences.history_key, "HISTORY_ATTESTATION_SIGNING"),
            (value.fences.trust_key, "TRUST_ACTIVATION"),
        ):
            _call_sql(value, "provision_fence", (key, semantic_role, 1), admin=True)
            _call_sql(value, "transition_fence", (key, 1, "ACTIVE", 1), admin=True)
        value.commit.approve_prepared(
            value.preparation, value.head, value.history_boundary.signer, fences=value.fences
        )
        value.commit.approve_prepared(
            value.secondary.preparation,
            value.secondary_head,
            value.history_boundary.signer,
            fences=value.fences,
        )
        value.baseline = _fixture_data(value)
        yield value
    finally:
        if installed:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(value.setup.schema)
                    )
                )
                for role in (
                    value.setup.issuer_writer_role,
                    value.setup.test_admin_role,
                    value.setup.schema_owner_role,
                ):
                    conn.execute(sql.SQL("DROP OWNED BY {} CASCADE").format(sql.Identifier(role)))
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
        source.close()


def _data_tables(value):
    """Topological order for the strictly isolated fixture's source records."""
    registry = value.genuine.entitlement_admin._schema
    history = value.history_boundary.history_setup.schema
    tables = [(registry, "lineages"), (registry, "history")]
    for credential in (value.genuine.pair.requester, value.genuine.pair.claimant):
        tables.extend(
            (credential._schema, table)
            for table in ("credentials", "history", "current_credentials")
        )
    tables.extend(
        [
            (value.intent_setup.schema, "intents"),
            (value.retention_setup.schema, "prepared"),
            (history, "streams"),
            (history, "records"),
            (history, "attestations"),
            (value.setup.schema, "approved_commits"),
            (value.setup.schema, "fences"),
            (value.setup.schema, "decisions"),
        ]
    )
    return tables


def _fixture_data(value):
    with psycopg.connect(BASE_DSN) as conn:
        return {
            pair: conn.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1").format(sql.Identifier(*pair))
            ).fetchall()
            for pair in _data_tables(value)
        }


def _restore_isolated_baseline(value):
    """Test administrator teardown, never recovery or application behavior.

    Each case first asserts its actual durable transaction result through a
    fresh backend. Restoring the isolated fixture afterward permits sharing
    genuine immutable signed evidence without repeatedly running custody or
    the INITIAL_BINDING lineage. No runtime/production API can do this reset.
    """
    from psycopg.types.json import Jsonb

    if _fixture_data(value) == value.baseline:
        with psycopg.connect(value.writer_connection.dsn) as writer:
            qualification.qualify_atomic_root_proof_authority(writer, value.atomic_deployment)
        return
    tables = _data_tables(value)
    with psycopg.connect(BASE_DSN) as administrator:
        triggers = administrator.execute(
            "SELECT n.nspname,c.relname,t.tgname,t.tgenabled FROM pg_catalog.pg_trigger t "
            "JOIN pg_catalog.pg_class c ON c.oid=t.tgrelid "
            "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=ANY(%s)",
            ([schema for schema, _ in tables],),
        ).fetchall()
        for pair in tables:
            administrator.execute(
                sql.SQL("ALTER TABLE {} DISABLE TRIGGER ALL").format(sql.Identifier(*pair))
            )
        administrator.execute(
            sql.SQL("TRUNCATE {}").format(
                sql.SQL(",").join(sql.Identifier(*pair) for pair in tables)
            )
        )
        for pair in tables:
            for row in value.baseline[pair]:
                administrator.execute(
                    sql.SQL("INSERT INTO {} VALUES({})").format(
                        sql.Identifier(*pair), sql.SQL(",").join(sql.Placeholder() for _ in row)
                    ),
                    tuple(Jsonb(part) if isinstance(part, dict) else part for part in row),
                )
        modes = {"O": "ENABLE", "A": "ENABLE ALWAYS", "R": "ENABLE REPLICA", "D": "DISABLE"}
        for schema, table, trigger, mode in triggers:
            if (schema, table) in tables:
                administrator.execute(
                    sql.SQL("ALTER TABLE {} {} TRIGGER {}").format(
                        sql.Identifier(schema, table), sql.SQL(modes[mode]), sql.Identifier(trigger)
                    )
                )
        administrator.commit()
    assert _fixture_data(value) == value.baseline
    with psycopg.connect(value.writer_connection.dsn) as writer:
        qualification.qualify_atomic_root_proof_authority(writer, value.atomic_deployment)


@pytest.fixture
def atomic_commit_boundary(module_atomic_commit_boundary):
    value = module_atomic_commit_boundary
    assert _fixture_data(value) == value.baseline
    try:
        yield value
    finally:
        _restore_isolated_baseline(value)


def _assert_committed(value, decision, *, candidate=None):
    if candidate is None:
        preparation, prepared, reserved = value.preparation, value.prepared, value.reserved
        subject, head = value.genuine.subject, value.head
    else:
        preparation, prepared, reserved = (
            candidate.preparation,
            candidate.prepared,
            candidate.reserved,
        )
        subject, head = candidate.subject, candidate.head
    assert decision.state == "COMMITTED"
    assert decision.payload == prepared.canonical_payload
    assert decision.signature == prepared.signature
    assert decision.envelope == prepared.canonical_envelope
    snapshot = _snapshot(value)
    registry_schema = value.genuine.entitlement_admin._schema
    history_schema = value.history_boundary.history_setup.schema
    subject_history = [
        row for row in snapshot[(registry_schema, "history")] if row[0] == subject.lookup_handle
    ]
    assert len(subject_history) == 2
    assert subject_history[-1][5]["binding"]["kind"] == "BOUND"
    assert len(snapshot[(history_schema, "records")]) == 1
    assert len(snapshot[(history_schema, "attestations")]) == 1
    assert len(snapshot[(value.setup.schema, "decisions")]) == 1
    assert len(snapshot[(value.intent_setup.schema, "intents")]) == len(
        value.baseline[(value.intent_setup.schema, "intents")]
    )
    assert len(snapshot[(value.retention_setup.schema, "prepared")]) == len(
        value.baseline[(value.retention_setup.schema, "prepared")]
    )
    with psycopg.connect(BASE_DSN) as conn:
        retained = conn.execute(
            sql.SQL(
                "SELECT canonical_payload,signature,canonical_envelope FROM {}.prepared WHERE attempt_key=%s"
            ).format(sql.Identifier(value.retention_setup.schema)),
            (preparation.locator.attempt,),
        ).fetchone()
        assert tuple(bytes(part) for part in retained) == (
            prepared.canonical_payload,
            prepared.signature,
            prepared.canonical_envelope,
        )
        retained_head = conn.execute(
            sql.SQL("SELECT signature FROM {}.attestations").format(sql.Identifier(history_schema))
        ).fetchone()
        assert bytes(retained_head[0]) == head.signature
        marker = conn.execute(
            sql.SQL(
                "SELECT reserved_envelope,staged_envelope,canonical_payload,signature,"
                "canonical_record,canonical_head,history_signature,registry_revision,"
                "history_sequence,attempt_key,root_proof_key FROM {}.decisions"
            ).format(sql.Identifier(value.setup.schema))
        ).fetchone()
        assert tuple(bytes(part) for part in marker[:7]) == (
            reserved.canonical_envelope,
            prepared.canonical_envelope,
            prepared.canonical_payload,
            prepared.signature,
            canonical_json_bytes(preparation.history_record.unsigned_material()),
            head.canonical_attestation_bytes,
            head.signature,
        )
        assert marker[7:9] == (2, 1)
        assert bytes(marker[9]) == preparation.locator.attempt
        assert bytes(marker[10]) == preparation.locator.root_proof
    return snapshot


def test_real_atomic_commit_and_exact_retry_retain_original_bytes(atomic_commit_boundary):
    value = atomic_commit_boundary
    decision = value.commit.commit(value.preparation)
    committed = _assert_committed(value, decision)
    replay = value.commit.commit(value.preparation)
    assert replay == decision
    assert value.commit.reconcile(value.preparation.locator) == decision
    assert _snapshot(value) == committed


def test_two_parallel_deliveries_have_one_durable_winner(atomic_commit_boundary):
    value = atomic_commit_boundary
    barrier = threading.Barrier(2)

    def commit():
        barrier.wait(timeout=10)
        try:
            return value.commit.commit(value.preparation)
        except commit_runtime.AtomicRootProofCommitConflict:
            return value.commit.reconcile(value.preparation.locator)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(lambda _: commit(), range(2)))
    assert outcomes[0] == outcomes[1]
    _assert_committed(value, outcomes[0])


def test_changed_attempt_is_refused_and_cannot_mint_another_decision(atomic_commit_boundary):
    value = atomic_commit_boundary
    value.commit.commit(value.preparation)
    before = _snapshot(value)
    with pytest.raises(psycopg.Error):
        _call_sql(
            value,
            "commit_root_proof",
            (canonical_json_bytes("rpa_018f3e70-7b5d-7c21-8b9a-0123456789ac"),),
        )
    assert _snapshot(value) == before


@pytest.mark.parametrize(
    ("authority", "table", "event"),
    [
        ("entitlement", "history", "INSERT"),
        ("entitlement", "lineages", "UPDATE"),
        ("history", "records", "INSERT"),
        ("history", "streams", "UPDATE"),
        ("history", "attestations", "INSERT"),
        ("coordinator", "decisions", "INSERT"),
    ],
)
def test_fault_after_every_write_rolls_back_the_entire_physical_set(
    atomic_commit_boundary, authority, table, event
):
    value = atomic_commit_boundary
    before = _snapshot(value)
    schema = {
        "entitlement": value.genuine.entitlement_admin._schema,
        "history": value.history_boundary.history_setup.schema,
        "coordinator": value.setup.schema,
    }[authority]
    writer_pid = []
    ready = threading.Event()

    def blocked_commit():
        with psycopg.connect(value.writer_connection.dsn) as writer:
            writer.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            writer_pid.append(writer.info.backend_pid)
            ready.set()
            writer.execute(
                sql.SQL("SELECT {}.commit_root_proof(%s)").format(
                    sql.Identifier(value.setup.schema)
                ),
                (value.preparation.locator.attempt,),
            ).fetchone()
            writer.commit()

    # The pristine authority passes its live SQL guard, then waits on the
    # administrator's fence lock before opening a write-target relation.
    # Installing the fault while it waits makes the injected trigger fire
    # inside the admitted write set, instead of rejecting during preflight.
    with psycopg.connect(BASE_DSN) as administrator:
        first_fence = min(value.fences.root_key, value.fences.history_key, value.fences.trust_key)
        administrator.execute(
            sql.SQL("SELECT identity FROM {}.fences WHERE identity=%s FOR UPDATE").format(
                sql.Identifier(value.setup.schema)
            ),
            (first_fence,),
        ).fetchone()
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                committing = executor.submit(blocked_commit)
                try:
                    assert ready.wait(timeout=10)
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        waiting = administrator.execute(
                            "SELECT wait_event_type,wait_event FROM pg_catalog.pg_stat_activity WHERE pid=%s",
                            (writer_pid[0],),
                        ).fetchone()
                        if waiting == ("Lock", "transactionid"):
                            break
                        if committing.done():
                            committing.result()
                            pytest.fail("atomic writer did not block on the locked fence")
                        time.sleep(0.02)
                        administrator.execute("SELECT pg_catalog.pg_stat_clear_snapshot()")
                    else:
                        administrator.rollback()
                        pytest.fail(
                            "atomic writer did not reach the post-qualification lifecycle lock"
                        )
                    administrator.execute(
                        sql.SQL(
                            "CREATE FUNCTION {}.injected_write_failure() RETURNS trigger "
                            "LANGUAGE plpgsql AS $$BEGIN RAISE EXCEPTION 'atomic-write-cut'; END$$"
                        ).format(sql.Identifier(schema))
                    )
                    administrator.execute(
                        sql.SQL(
                            "CREATE TRIGGER injected_write_failure AFTER {} ON {} "
                            "FOR EACH ROW EXECUTE FUNCTION {}.injected_write_failure()"
                        ).format(
                            sql.SQL(event), sql.Identifier(schema, table), sql.Identifier(schema)
                        )
                    )
                    administrator.commit()
                    with pytest.raises(psycopg.Error, match="atomic-write-cut"):
                        committing.result(timeout=10)
                finally:
                    administrator.rollback()
        finally:
            administrator.rollback()
            administrator.execute(
                sql.SQL("DROP TRIGGER IF EXISTS injected_write_failure ON {}").format(
                    sql.Identifier(schema, table)
                )
            )
            administrator.execute(
                sql.SQL("DROP FUNCTION IF EXISTS {}.injected_write_failure()").format(
                    sql.Identifier(schema)
                )
            )
            administrator.commit()
            # PostgreSQL lazily clears relhastriggers after dropping the final
            # trigger. Restore the exact catalog pin before the next case.
            administrator.autocommit = True
            administrator.execute(sql.SQL("VACUUM {}").format(sql.Identifier(schema, table)))
    assert _snapshot(value) == before


def test_exception_before_commit_leaves_no_decision_and_not_found_is_indeterminate(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary
    before = _snapshot(value)

    def fail(name, connection):
        if name == "before_commit":
            raise RuntimeError("interrupted-before-commit")

    monkeypatch.setattr(commit_runtime, "_persistence_cut", fail)
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.reconcile(value.preparation.locator)


def test_backend_terminated_before_commit_rolls_back_all_writes(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary
    before = _snapshot(value)

    def kill_backend(name, connection):
        if name == "before_commit":
            with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
                terminated = administrator.execute(
                    "SELECT pg_catalog.pg_terminate_backend(%s)", (connection.info.backend_pid,)
                ).fetchone()
                assert terminated == (True,)

    monkeypatch.setattr(commit_runtime, "_persistence_cut", kill_backend)
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.reconcile(value.preparation.locator)


def test_committed_response_loss_recovers_exact_marker_without_second_history(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary

    def lose_response(name, connection):
        if name == "after_commit":
            raise RuntimeError("committed-response-lost")

    monkeypatch.setattr(commit_runtime, "_persistence_cut", lose_response)
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.commit(value.preparation)
    monkeypatch.setattr(commit_runtime, "_persistence_cut", lambda *_: None)
    restarted_client = commit_runtime.PostgreSQLAtomicRootProofCommit(
        value.atomic_deployment, admin_connection=_connection(value.setup.test_admin_role)
    )
    recovered = restarted_client.reconcile(value.preparation.locator)
    committed = _assert_committed(value, recovered)
    assert restarted_client.commit(value.preparation) == recovered
    assert _snapshot(value) == committed


def test_server_restart_recovers_exact_bytes_and_history(atomic_commit_boundary):
    value = atomic_commit_boundary
    decision = value.commit.commit(value.preparation)
    before = _assert_committed(value, decision)
    _restart_controlled_postgresql()
    restarted_client = commit_runtime.PostgreSQLAtomicRootProofCommit(
        value.atomic_deployment, admin_connection=_connection(value.setup.test_admin_role)
    )
    assert restarted_client.reconcile(value.preparation.locator) == decision
    assert restarted_client.commit(value.preparation) == decision
    assert _snapshot(value) == before


@pytest.mark.parametrize("fence", ["root_key", "history_key", "trust_key"])
@pytest.mark.parametrize("state", ["PENDING_CLOSED", "VERIFY_ONLY", "REVOKED"])
def test_closed_lifecycle_fence_refuses_bind(atomic_commit_boundary, fence, state):
    value = atomic_commit_boundary
    _call_sql(value, "transition_fence", (getattr(value.fences, fence), 1, state, 1), admin=True)
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


def _wait_for_role_lock(role):
    """Prove the competing administrator reached the actual DB lock wait."""
    deadline = time.monotonic() + 10
    with psycopg.connect(BASE_DSN, autocommit=True) as observer:
        while time.monotonic() < deadline:
            waits = observer.execute(
                "SELECT wait_event FROM pg_catalog.pg_stat_activity "
                "WHERE usename=%s AND wait_event_type='Lock'",
                (role,),
            ).fetchall()
            if any(wait[0] in {"transactionid", "advisory"} for wait in waits):
                return
            time.sleep(0.02)
    pytest.fail("the competing administrator did not block on the final transaction's DB lock")


def test_revoke_waits_for_final_commit_lock_and_never_crosses_the_eligibility_gap(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary
    entered = threading.Event()
    release = threading.Event()
    revoke_started = threading.Event()
    revoke_completed = threading.Event()

    def hold_commit(name, connection):
        if name == "before_commit":
            entered.set()
            assert release.wait(timeout=10)

    def revoke():
        assert entered.wait(timeout=10)
        revoke_started.set()
        _call_sql(value, "transition_fence", (value.fences.root_key, 1, "REVOKED", 1), admin=True)
        revoke_completed.set()

    monkeypatch.setattr(commit_runtime, "_persistence_cut", hold_commit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        committing = executor.submit(value.commit.commit, value.preparation)
        revoking = executor.submit(revoke)
        assert revoke_started.wait(timeout=10)
        try:
            _wait_for_role_lock(value.setup.test_admin_role)
            assert not revoke_completed.is_set()
        finally:
            release.set()
        decision = committing.result(timeout=10)
        revoking.result(timeout=10)
    _assert_committed(value, decision)
    assert revoke_completed.is_set()


def test_revoked_fence_cannot_be_reactivated(atomic_commit_boundary):
    value = atomic_commit_boundary
    _call_sql(value, "transition_fence", (value.fences.root_key, 1, "REVOKED", 1), admin=True)
    with pytest.raises(psycopg.Error):
        _call_sql(value, "transition_fence", (value.fences.root_key, 1, "ACTIVE", 2), admin=True)
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


@pytest.mark.parametrize(
    "change", ["table-dml", "membership", "function-source", "external-column"]
)
def test_post_migration_acl_membership_and_source_changes_fail_closed(
    atomic_commit_boundary, change
):
    value = atomic_commit_boundary
    before = _snapshot(value)
    role = sql.Identifier(value.setup.issuer_writer_role)
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        if change == "table-dml":
            apply = sql.SQL("GRANT INSERT ON {}.decisions TO {}").format(
                sql.Identifier(value.setup.schema), role
            )
            restore = sql.SQL("REVOKE INSERT ON {}.decisions FROM {}").format(
                sql.Identifier(value.setup.schema), role
            )
        elif change == "membership":
            apply = sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(value.setup.schema_owner_role), role
            )
            restore = sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(value.setup.schema_owner_role), role
            )
        elif change == "external-column":
            table = sql.Identifier("public", f"atomic_extra_{uuid.uuid4().hex[:10]}")
            administrator.execute(sql.SQL("CREATE TABLE {} (value integer)").format(table))
            apply = sql.SQL("GRANT UPDATE(value) ON {} TO {}").format(table, role)
            restore = sql.SQL("DROP TABLE {}").format(table)
        else:
            original = administrator.execute(
                "SELECT pg_catalog.pg_get_functiondef(p.oid) FROM pg_catalog.pg_proc p "
                "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
                "WHERE n.nspname=%s AND p.proname='commit_root_proof'",
                (value.setup.schema,),
            ).fetchone()[0]
            apply = sql.SQL(original.replace("BEGIN", "BEGIN\n-- altered-after-qualification", 1))
            restore = sql.SQL(original)
        administrator.execute(apply)
        try:
            with psycopg.connect(value.writer_connection.dsn) as writer:
                with pytest.raises(qualification.AtomicRootProofQualificationError):
                    qualification.qualify_atomic_root_proof_authority(
                        writer, value.atomic_deployment
                    )
            with pytest.raises(psycopg.Error):
                _call_sql(value, "commit_root_proof", (value.preparation.locator.attempt,))
            with pytest.raises(
                (
                    commit_runtime.AtomicRootProofCommitUnavailable,
                    commit_runtime.AtomicRootProofCommitCorrupt,
                    qualification.AtomicRootProofQualificationError,
                )
            ):
                value.commit.commit(value.preparation)
        finally:
            administrator.execute(restore)
    assert _snapshot(value) == before


def test_direct_writer_cannot_create_bound_or_history_without_exact_linkage(atomic_commit_boundary):
    value = atomic_commit_boundary
    before = _snapshot(value)
    for schema, table in (
        (value.genuine.entitlement_admin._schema, "history"),
        (value.history_boundary.history_setup.schema, "records"),
        (value.history_boundary.history_setup.schema, "attestations"),
        (value.setup.schema, "decisions"),
        (value.intent_setup.schema, "intents"),
        (value.retention_setup.schema, "prepared"),
    ):
        with psycopg.connect(value.writer_connection.dsn) as writer:
            with pytest.raises(InsufficientPrivilege):
                writer.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(schema, table)))
    assert _snapshot(value) == before


def test_missing_staged_linkage_rolls_back_bound_and_all_histories(atomic_commit_boundary):
    value = atomic_commit_boundary
    existing_atomic._privileged_mutation(
        SimpleNamespace(setup=value.retention_setup),
        sql.SQL("DELETE FROM {}.prepared WHERE attempt_key=%s").format(
            sql.Identifier(value.retention_setup.schema)
        ),
        (value.preparation.locator.attempt,),
    )
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


@pytest.mark.parametrize(
    "field", ["canonical_payload", "signature", "canonical_envelope", "root_proof_key"]
)
def test_inconsistent_reserved_staged_bytes_refuse_without_bind(atomic_commit_boundary, field):
    value = atomic_commit_boundary
    replacement = b"x" * 64 if field == "signature" else b"{}"
    existing_atomic._privileged_mutation(
        SimpleNamespace(setup=value.retention_setup),
        sql.SQL("UPDATE {}.prepared SET {}=%s WHERE attempt_key=%s").format(
            sql.Identifier(value.retention_setup.schema), sql.Identifier(field)
        ),
        (replacement, value.preparation.locator.attempt),
    )
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


def test_corrupted_history_head_rejects_cas_and_rolls_back_attempted_bind(atomic_commit_boundary):
    value = atomic_commit_boundary
    # The expected predecessor is genuinely prepared from the initial stream.
    # A privileged negative control advances the retained DB CAS coordinates;
    # the later history CAS must reject and roll back the preceding registry CAS.
    with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
        administrator.execute(
            sql.SQL("UPDATE {}.streams SET head_sequence=1,head_digest=%s").format(
                sql.Identifier(value.history_boundary.history_setup.schema)
            ),
            ("sha256:" + "f" * 64,),
        )
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


@pytest.mark.parametrize(
    "other_cluster", [False, True], ids=["different-database", "different-cluster"]
)
def test_matching_connection_labels_cannot_qualify_another_physical_database(
    atomic_commit_boundary, other_cluster
):
    value = atomic_commit_boundary
    other_database = f"atomic_other_{uuid.uuid4().hex[:10]}"
    if other_cluster:
        dsn = os.environ.get("CRYPTOHUNTER_TEST_POSTGRES_OTHER_DSN")
        assert dsn, "a genuinely distinct PostgreSQL 16 cluster is mandatory"
    else:
        with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
            administrator.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(other_database))
            )
        dsn = make_conninfo(BASE_DSN, dbname=other_database)
    foreign_admin_dsn = dsn
    created_role = False
    try:
        with psycopg.connect(foreign_admin_dsn, autocommit=True) as administrator:
            if other_cluster:
                administrator.execute(
                    sql.SQL(
                        "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER "
                        "NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                    ).format(sql.Identifier(value.setup.issuer_writer_role))
                )
                created_role = True
            administrator.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO {}").format(
                    sql.Identifier(value.setup.issuer_writer_role)
                )
            )
        with psycopg.connect(make_conninfo(dsn, user=value.setup.issuer_writer_role)) as foreign:
            assert foreign.info.server_version // 10000 == 16
            foreign_id = foreign.execute(
                "SELECT c.system_identifier,d.oid FROM pg_catalog.pg_control_system() c,"
                "pg_catalog.pg_database d WHERE d.datname=pg_catalog.current_database()"
            ).fetchone()
            identity = value.atomic_deployment.identity
            if other_cluster:
                assert foreign_id[0] != identity.system_identifier
            else:
                assert foreign_id[0] == identity.system_identifier
                assert foreign_id[1] != identity.database_oid
            with pytest.raises(
                qualification.AtomicRootProofQualificationError,
                match="physical database or service",
            ):
                qualification.qualify_atomic_root_proof_authority(foreign, value.atomic_deployment)
    finally:
        if created_role:
            with psycopg.connect(foreign_admin_dsn, autocommit=True) as administrator:
                administrator.execute(
                    sql.SQL("DROP OWNED BY {} CASCADE").format(
                        sql.Identifier(value.setup.issuer_writer_role)
                    )
                )
                administrator.execute(
                    sql.SQL("DROP ROLE {}").format(sql.Identifier(value.setup.issuer_writer_role))
                )
        if not other_cluster:
            with psycopg.connect(BASE_DSN, autocommit=True) as administrator:
                administrator.execute(
                    sql.SQL("DROP DATABASE {}").format(sql.Identifier(other_database))
                )


def test_every_legacy_write_api_is_unavailable_to_atomic_writer(atomic_commit_boundary):
    value = atomic_commit_boundary
    before = _snapshot(value)
    for schema, function in (
        (value.genuine.entitlement_admin._schema, "append_bind"),
        (value.history_boundary.history_setup.schema, "append_record"),
        (value.history_boundary.history_setup.schema, "retain_head"),
        (value.retention_setup.schema, "retain_prepared"),
        (value.intent_setup.schema, "reserve_intent"),
    ):
        with psycopg.connect(BASE_DSN) as administrator:
            argument_count = administrator.execute(
                "SELECT p.pronargs FROM pg_catalog.pg_proc p JOIN pg_catalog.pg_namespace n "
                "ON n.oid=p.pronamespace WHERE n.nspname=%s AND p.proname=%s",
                (schema, function),
            ).fetchone()[0]
        with psycopg.connect(value.writer_connection.dsn) as writer:
            with pytest.raises(InsufficientPrivilege):
                writer.execute(
                    sql.SQL("SELECT {}.{}({})").format(
                        sql.Identifier(schema),
                        sql.Identifier(function),
                        sql.SQL(",").join(sql.Placeholder() for _ in range(argument_count)),
                    ),
                    (None,) * argument_count,
                )
    assert _snapshot(value) == before


def test_original_registry_and_history_writers_cannot_finalize_standalone(atomic_commit_boundary):
    value = atomic_commit_boundary
    before = _snapshot(value)
    for connection, schema, function, argument_count in (
        (
            value.genuine.authority.entitlement_registry._connection,
            value.genuine.entitlement_admin._schema,
            "append_bind",
            5,
        ),
        (
            value.history._connection,
            value.history_boundary.history_setup.schema,
            "append_record",
            4,
        ),
        (value.history._connection, value.history_boundary.history_setup.schema, "retain_head", 4),
    ):
        with psycopg.connect(connection.dsn) as writer:
            with pytest.raises(InsufficientPrivilege):
                writer.execute(
                    sql.SQL("SELECT {}.{}({})").format(
                        sql.Identifier(schema),
                        sql.Identifier(function),
                        sql.SQL(",").join(sql.Placeholder() for _ in range(argument_count)),
                    ),
                    (None,) * argument_count,
                )
    assert _snapshot(value) == before


def test_privileged_bound_without_decision_is_rejected_by_deferred_linkage(atomic_commit_boundary):
    value = atomic_commit_boundary
    before = _snapshot(value)
    from psycopg.types.json import Jsonb

    from bot_core.postgresql_entitlement_registry import _state_json

    state = _state_json(value.preparation.bound_state)
    with psycopg.connect(BASE_DSN) as administrator:
        administrator.execute(
            sql.SQL(
                "INSERT INTO {}.history(lookup_handle,environment,trust_domain,revision,"
                "predecessor_revision,state,state_integrity) VALUES(%s,%s,%s,2,1,%s,md5(%s))"
            ).format(sql.Identifier(value.genuine.entitlement_admin._schema)),
            (
                value.genuine.subject.lookup_handle,
                value.auth.environment,
                value.auth.trust_domain,
                Jsonb(state),
                canonical_json_bytes(state).decode(),
            ),
        )
        with pytest.raises(psycopg.errors.CheckViolation, match="exact atomic decision linkage"):
            administrator.commit()
    assert _snapshot(value) == before


def test_bound_decision_preserves_legal_entitlement_admin_revoke(atomic_commit_boundary):
    value = atomic_commit_boundary
    decision = value.commit.commit(value.preparation)
    _assert_committed(value, decision)
    from psycopg.types.json import Jsonb

    from bot_core.entitlement_registry_contract import EntitlementLifecycle
    from bot_core.postgresql_entitlement_registry import _state_json

    state = replace(
        value.preparation.bound_state,
        lifecycle=EntitlementLifecycle.REVOKED,
        authoritative_state_revision=3,
        predecessor_revision=2,
    )
    with psycopg.connect(value.genuine.entitlement_admin._connection.dsn) as administrator:
        administrator.execute(
            sql.SQL("SELECT {}.revoke(%s,%s,%s,2,%s::json)").format(
                sql.Identifier(value.genuine.entitlement_admin._schema)
            ),
            (
                value.genuine.subject.lookup_handle,
                value.auth.environment,
                value.auth.trust_domain,
                Jsonb(_state_json(state)),
            ),
        ).fetchone()
        administrator.commit()
    snapshot = _snapshot(value)
    rows = [
        row
        for row in snapshot[(value.genuine.entitlement_admin._schema, "history")]
        if row[0] == value.genuine.subject.lookup_handle
    ]
    assert len(rows) == 3
    assert rows[-1][5]["lifecycle"] == "REVOKED"
    assert rows[-1][5]["binding"]["kind"] == "BOUND"
    assert len(snapshot[(value.setup.schema, "decisions")]) == 1
    assert value.commit.reconcile(value.preparation.locator) == decision


def test_actual_commit_receive_disconnect_recovers_durable_decision(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary
    actual_commit_calls = []

    def drop_commit_response(name, connection):
        if name == "before_commit":
            # This is the native libpq TCP connection, using the same pinned
            # database/service identity. The real COMMIT still goes to the DB;
            # its server response cannot be delivered to libpq afterward.
            with socket.fromfd(
                connection.pgconn.socket, socket.AF_INET, socket.SOCK_STREAM
            ) as wire:
                wire.shutdown(socket.SHUT_RD)
            actual_commit_calls.append(connection.info.backend_pid)

    monkeypatch.setattr(commit_runtime, "_persistence_cut", drop_commit_response)
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.commit(value.preparation)
    assert len(actual_commit_calls) == 1
    monkeypatch.setattr(commit_runtime, "_persistence_cut", lambda *_: None)
    deadline = time.monotonic() + 10
    while True:
        try:
            decision = value.commit.reconcile(value.preparation.locator)
            break
        except commit_runtime.AtomicRootProofCommitIndeterminate:
            if time.monotonic() >= deadline:
                pytest.fail("COMMIT was not durably retained after the native response disconnect")
            time.sleep(0.02)
    before = _assert_committed(value, decision)
    assert value.commit.commit(value.preparation.locator) == decision
    assert _snapshot(value) == before


def test_fresh_client_process_reconstructs_qualified_deployment_and_exact_retry(
    atomic_commit_boundary,
):
    value = atomic_commit_boundary
    decision = value.commit.commit(value.preparation)
    before = _assert_committed(value, decision)
    command = """
import json,sys
from bot_core.postgresql_atomic_root_proof_commit import AtomicRootProofCommitLocator,PostgreSQLAtomicRootProofCommit
from bot_core.postgresql_atomic_root_proof_schema import PostgreSQLAtomicRootProofProvisioning
from bot_core.postgresql_atomic_root_proof_migration import requalify_existing_atomic_root_proof_v1
from bot_core.postgresql_atomic_root_proof_qualification import PostgreSQLAtomicRootProofPhysicalIdentity
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
value=json.loads(sys.argv[1])
deployment=requalify_existing_atomic_root_proof_v1(
 PostgreSQLConnectionConfig(value['admin']),PostgreSQLConnectionConfig(value['writer']),
 PostgreSQLAtomicRootProofProvisioning(**value['provisioning']),
 expected_identity=PostgreSQLAtomicRootProofPhysicalIdentity(**value['identity']))
locator=AtomicRootProofCommitLocator(**{k:bytes.fromhex(v) for k,v in value['locator'].items()})
runtime=PostgreSQLAtomicRootProofCommit(deployment)
read=runtime.reconcile(locator)
retry=runtime.commit(locator)
assert read == retry
print(json.dumps({'payload':retry.payload.hex(),'signature':retry.signature.hex(),'envelope':retry.envelope.hex()}))
"""
    arguments = json.dumps(
        {
            "admin": BASE_DSN,
            "writer": value.writer_connection.dsn,
            "provisioning": asdict(value.setup),
            "identity": asdict(value.atomic_deployment.identity),
            "locator": {
                field: part.hex() for field, part in asdict(value.preparation.locator).items()
            },
        }
    )
    result = subprocess.run(
        [sys.executable, "-c", command, arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    recovered = json.loads(result.stdout)
    assert recovered == {
        "payload": decision.payload.hex(),
        "signature": decision.signature.hex(),
        "envelope": decision.envelope.hex(),
    }
    assert _snapshot(value) == before


def _transition_request_credential(value, role, lifecycle):
    from psycopg.types.json import Jsonb

    admin = getattr(value.genuine.pair, role + "_admin")
    principal = "CryptoHunterAccountAuthority" if role == "requester" else value.genuine.principal
    payload = {
        "operation": "TRANSITION",
        "principal_id": principal,
        "credential_id": None,
        "key_id": None,
        "key_version": None,
        "public_key_hex": None,
        "key_material_identity": None,
        "expected_revision": 1,
        "lifecycle": lifecycle,
    }
    with psycopg.connect(admin._connection.dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        conn.execute(
            sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(admin._schema)), (Jsonb(payload),)
        ).fetchone()
        conn.commit()


@pytest.mark.parametrize("role", ["requester", "claimant"])
@pytest.mark.parametrize("lifecycle", ["VERIFY_ONLY", "REVOKED"])
def test_final_db_request_credential_transition_refuses_stale_active_snapshot(
    atomic_commit_boundary, role, lifecycle
):
    value = atomic_commit_boundary
    _transition_request_credential(value, role, lifecycle)
    before = _snapshot(value)
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert _snapshot(value) == before


@pytest.mark.parametrize("role", ["requester", "claimant"])
def test_request_credential_admin_transition_waits_for_final_transaction(
    atomic_commit_boundary, monkeypatch, role
):
    value = atomic_commit_boundary
    entered = threading.Event()
    release = threading.Event()
    transition_started = threading.Event()
    transition_completed = threading.Event()

    def hold_commit(name, connection):
        if name == "before_commit":
            entered.set()
            assert release.wait(timeout=10)

    def transition():
        assert entered.wait(timeout=10)
        transition_started.set()
        _transition_request_credential(value, role, "REVOKED")
        transition_completed.set()

    monkeypatch.setattr(commit_runtime, "_persistence_cut", hold_commit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        committing = executor.submit(value.commit.commit, value.preparation)
        transitioning = executor.submit(transition)
        assert transition_started.wait(timeout=10)
        admin_role = getattr(value.genuine.pair.setup, role).admin_role
        try:
            _wait_for_role_lock(admin_role)
            assert not transition_completed.is_set()
        finally:
            release.set()
        decision = committing.result(timeout=10)
        transitioning.result(timeout=10)
    _assert_committed(value, decision)
    assert transition_completed.is_set()


@pytest.mark.parametrize(
    "field", ["scope", "attempt", "operation", "entitlement", "root_proof", "account"]
)
def test_changed_recovery_identity_cannot_write_or_reuse_an_existing_marker(
    atomic_commit_boundary, field
):
    value = atomic_commit_boundary
    value.commit.commit(value.preparation)
    before = _snapshot(value)
    locator = value.preparation.locator
    changed = replace(locator, **{field: getattr(locator, field) + b"-changed"})
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(changed)
    assert _snapshot(value) == before


def test_abrupt_backend_sigkill_before_commit_recovers_no_partial_write_set(
    atomic_commit_boundary, monkeypatch
):
    value = atomic_commit_boundary
    before = _snapshot(value)
    controller = os.environ.get("DUDZIAN_TEST_POSTGRES_CRASH_COMMAND", "")
    assert controller, "an explicitly controlled test PostgreSQL backend crash command is required"
    prefix = json.loads(controller)
    assert type(prefix) is list and prefix and all(type(part) is str and part for part in prefix)
    crashed = []

    def crash(name, connection):
        if name == "before_commit":
            backend = connection.info.backend_pid
            assert type(backend) is int and backend > 1
            # The fixed test-infrastructure argv targets the isolated service's
            # own PID namespace. The append-only numeric PID comes from libpq.
            subprocess.run([*prefix, str(backend)], check=True, capture_output=True, timeout=10)
            crashed.append(backend)

    monkeypatch.setattr(commit_runtime, "_persistence_cut", crash)
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.commit(value.preparation)
    assert len(crashed) == 1
    deadline = time.monotonic() + 30
    while True:
        try:
            with psycopg.connect(BASE_DSN, connect_timeout=2):
                break
        except psycopg.Error:
            if time.monotonic() >= deadline:
                pytest.fail("isolated PostgreSQL did not recover after the abrupt backend crash")
            time.sleep(0.02)
    assert _snapshot(value) == before
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.reconcile(value.preparation.locator)


def test_valid_advanced_history_refuses_genuinely_prepared_stale_predecessor(
    atomic_commit_boundary,
):
    value = atomic_commit_boundary
    winner = value.commit.commit(value.preparation)
    before = _assert_committed(value, winner)
    # Both independently genuine candidates observed the same predecessor
    # before migration. The first now has a valid signed durable record/head
    # and immutable marker; the second must lose its own registry BIND when
    # the shared history stream's predecessor CAS no longer matches.
    assert value.secondary.preparation.locator.entitlement != value.preparation.locator.entitlement
    assert value.secondary.preparation.history_record.predecessor_authenticated_digest == (
        value.preparation.history_record.predecessor_authenticated_digest
    )
    with pytest.raises(commit_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.secondary.preparation)
    assert _snapshot(value) == before
    secondary_rows = [
        row
        for row in before[(value.genuine.entitlement_admin._schema, "history")]
        if row[0] == value.secondary.subject.lookup_handle
    ]
    assert len(secondary_rows) == 1
    assert secondary_rows[0][3] == 1
    assert secondary_rows[0][5]["binding"]["kind"] == "UNBOUND"
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.reconcile(value.secondary.preparation.locator)
    assert value.commit.reconcile(value.preparation.locator) == winner


def test_parallel_distinct_genuine_operations_have_one_history_cas_winner(atomic_commit_boundary):
    value = atomic_commit_boundary
    barrier = threading.Barrier(2)
    primary = SimpleNamespace(
        preparation=value.preparation,
        prepared=value.prepared,
        reserved=value.reserved,
        subject=value.genuine.subject,
        head=value.head,
    )
    candidates = (primary, value.secondary)

    def commit(candidate):
        barrier.wait(timeout=10)
        try:
            return "COMMITTED", value.commit.commit(candidate.preparation), candidate
        except commit_runtime.AtomicRootProofCommitConflict:
            return "CONFLICT", None, candidate

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(commit, candidates))
    assert sorted(outcome[0] for outcome in outcomes) == ["COMMITTED", "CONFLICT"]
    winner = next(outcome for outcome in outcomes if outcome[0] == "COMMITTED")
    loser = next(outcome for outcome in outcomes if outcome[0] == "CONFLICT")
    snapshot = _assert_committed(value, winner[1], candidate=winner[2])
    loser_history = [
        row
        for row in snapshot[(value.genuine.entitlement_admin._schema, "history")]
        if row[0] == loser[2].subject.lookup_handle
    ]
    assert len(loser_history) == 1
    assert loser_history[0][3] == 1
    assert loser_history[0][5]["binding"]["kind"] == "UNBOUND"
    assert value.commit.reconcile(winner[2].preparation.locator) == winner[1]
    with pytest.raises(commit_runtime.AtomicRootProofCommitIndeterminate):
        value.commit.reconcile(loser[2].preparation.locator)
    assert _snapshot(value) == snapshot
