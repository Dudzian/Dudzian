"""Prove historical V2 omitted-signer issuance and the explicit V3 fix.

Each omission owns a complete independent genuine INITIAL_BINDING lineage and
original retained signatures. Historical V2 deliberately remains installable
only as an isolated NOT_READY predecessor. The real V3 upgrade must preserve its
old decision while refusing another valid retained attempt without enrollment.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg import sql
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Jsonb

import bot_core.postgresql_atomic_root_proof_commit as atomic_runtime
import bot_core.postgresql_custody_lifecycle as lifecycle
import bot_core.postgresql_custody_lifecycle_qualification as qualification
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import ProviderRole
from tests.security import test_postgresql_atomic_root_proof_commit as atomic_tests
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]

lineage_monkeypatch = atomic_tests.lineage_monkeypatch
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


@pytest.fixture(scope="module", params=["ROOT_PROOF_SIGNING", "HISTORY_ATTESTATION_SIGNING"])
def lineage_path(tmp_path_factory, request):
    return tmp_path_factory.mktemp("omitted-signer-" + request.param.lower())


@pytest.fixture(scope="module")
def omitted_enrollment_boundary(module_atomic_commit_boundary, lineage_path):
    value = module_atomic_commit_boundary
    omitted_role = (
        ProviderRole.ROOT_PROOF_SIGNING
        if "root_proof_signing" in lineage_path.name
        else ProviderRole.HISTORY_ATTESTATION_SIGNING
    )
    bindings = []
    omitted_key = None
    for role, key in (
        (ProviderRole.ROOT_PROOF_SIGNING, value.fences.root_key),
        (ProviderRole.HISTORY_ATTESTATION_SIGNING, value.fences.history_key),
    ):
        if role is omitted_role:
            omitted_key = key
            continue
        bindings.append(
            lifecycle.prepare_custody_lifecycle_binding(
                value.atomic_deployment.identity,
                value.directory,
                security=value.security,
                role=role,
                fence_key=key,
            )
        )
    old = qualification.migrate_custody_lifecycle_v2(
        PostgreSQLConnectionConfig(BASE_DSN),
        value.atomic_deployment,
        atomic_tests._connection(value.setup.test_admin_role),
        tuple(bindings),
        sql_contract_version=2,
    )
    value.atomic_deployment = old.atomic_deployment
    value.commit = atomic_runtime.PostgreSQLAtomicRootProofCommit(old.atomic_deployment)
    value.omitted_role = omitted_role
    value.omitted_key = omitted_key
    value.historical_deployment = old
    return value


def _issuance_tables(value):
    """Only physical issuance effects; custody, fences and enrollment stay intact."""
    return (
        (value.genuine.entitlement_admin._schema, "lineages"),
        (value.genuine.entitlement_admin._schema, "history"),
        (value.history_boundary.history_setup.schema, "streams"),
        (value.history_boundary.history_setup.schema, "records"),
        (value.history_boundary.history_setup.schema, "attestations"),
        (value.setup.schema, "decisions"),
    )


def _restore_only_isolated_issuance_fixture(value):
    """Test teardown after observing the exploit, never application recovery.

    Original atomic V1 regressions use the same privileged baseline restoration
    to share expensive genuine evidence. This narrower restoration changes no
    fence, lifecycle operation, enrollment, signed byte or local custody file.
    """
    tables = _issuance_tables(value)
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
            sql.SQL("TRUNCATE {}").format(sql.SQL(",").join(sql.Identifier(*p) for p in tables))
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


def _assert_missing_active_enrollment(value):
    with psycopg.connect(BASE_DSN) as observer:
        row = observer.execute(
            sql.SQL(
                "SELECT f.state,f.generation,c.fence_key FROM {}.fences f "
                "LEFT JOIN {}.lifecycle_credentials c ON c.fence_key=f.identity WHERE f.identity=%s"
            ).format(sql.Identifier(value.setup.schema), sql.Identifier(value.setup.schema)),
            (value.omitted_key,),
        ).fetchone()
    assert row == ("ACTIVE", 1, None)


def _administrative_snapshot(value):
    with psycopg.connect(BASE_DSN) as observer:
        return tuple(
            observer.execute(
                sql.SQL("SELECT * FROM {} ORDER BY 1").format(
                    sql.Identifier(value.setup.schema, table)
                )
            ).fetchall()
            for table in (
                "fences",
                "lifecycle_credentials",
                "lifecycle_operations",
                "lifecycle_completions",
            )
        )


def test_historical_missing_enrollment_commits_but_v3_refuses_another_valid_attempt(
    omitted_enrollment_boundary,
):
    value = omitted_enrollment_boundary
    _assert_missing_active_enrollment(value)
    assert value.historical_deployment.sql_contract_version == 2
    # The predecessor bug is demonstrated by a real durable COMMIT, not by
    # mocking eligibility or by editing the retained signed proof.
    original = value.commit.commit(value.preparation)
    committed = atomic_tests._assert_committed(value, original)
    assert len(committed[(value.setup.schema, "decisions")]) == 1
    assert len(committed[(value.history_boundary.history_setup.schema, "records")]) == 1
    assert len(committed[(value.history_boundary.history_setup.schema, "attestations")]) == 1
    upgraded = qualification.upgrade_custody_lifecycle_signer_enrollment_v3(
        PostgreSQLConnectionConfig(BASE_DSN), value.historical_deployment
    )
    assert upgraded.sql_contract_version == 3
    value.atomic_deployment = upgraded.atomic_deployment
    value.commit = atomic_runtime.PostgreSQLAtomicRootProofCommit(upgraded.atomic_deployment)
    assert atomic_tests._snapshot(value) == committed
    retry = value.commit.commit(value.preparation)
    assert retry == original
    assert (retry.payload, retry.signature, retry.envelope) == (
        value.prepared.canonical_payload,
        value.prepared.signature,
        value.prepared.canonical_envelope,
    )
    assert atomic_tests._snapshot(value) == committed
    _assert_missing_active_enrollment(value)

    administrative = _administrative_snapshot(value)
    _restore_only_isolated_issuance_fixture(value)
    assert _administrative_snapshot(value) == administrative
    before = atomic_tests._snapshot(value)
    for pair in _issuance_tables(value):
        assert before[pair] == value.baseline[pair]
    assert not before[(value.setup.schema, "decisions")]
    assert not before[(value.history_boundary.history_setup.schema, "records")]
    assert value.secondary.preparation.locator.attempt != value.preparation.locator.attempt
    with pytest.raises(psycopg.errors.CheckViolation) as denied:
        atomic_tests._call_sql(
            value, "commit_root_proof", (value.secondary.preparation.locator.attempt,)
        )
    assert "enrollment" in denied.value.diag.message_primary.lower()
    assert atomic_tests._snapshot(value) == before
    assert _administrative_snapshot(value) == administrative
    _assert_missing_active_enrollment(value)

    # Both original and distinct valid attempts are refused for new issuance.
    with pytest.raises(atomic_runtime.AtomicRootProofCommitConflict):
        value.commit.commit(value.preparation)
    assert atomic_tests._snapshot(value) == before
    for function, arguments in (
        ("provision_fence", (value.omitted_key + "-replacement", value.omitted_role.value, 1)),
        ("transition_fence", (value.omitted_key, 1, "ACTIVE", 2)),
    ):
        with pytest.raises(InsufficientPrivilege):
            atomic_tests._call_sql(value, function, arguments, admin=True)
    with psycopg.connect(value.writer_connection.dsn) as writer:
        with pytest.raises(InsufficientPrivilege):
            writer.execute(
                sql.SQL("SELECT * FROM {}.lifecycle_credentials").format(
                    sql.Identifier(value.setup.schema)
                )
            )
    assert qualification.ATOMIC_ISSUER_READY == "NOT_READY"
    assert qualification.PRODUCTION_LIFECYCLE_ISOLATION == "BLOCKED"
    assert lifecycle.ATOMIC_ISSUER_READY == "NOT_READY"
    assert atomic_tests._snapshot(value) == before
