"""Read-only Stage 9 qualification against two genuine PostgreSQL 16 clusters.

The full immutable InitialBinding lineage is retained once for this module.
Individual cases get isolated issuer schemas, while every observation uses
production provider types and PostgreSQL's actual physical/role identities.
OS keyring storage and a stable test-device fingerprint are custody seams;
the signing/verifying adapters, request signatures and all public authority
qualifications remain genuine.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from bot_core import (
    local_signing_custody as custody,
    postgresql_root_proof_retention as retention_storage,
    postgresql_root_proof_retention_schema as retention_schema,
    postgresql_root_proof_signing_intent as intent_storage,
    postgresql_root_proof_signing_intent_schema as intent_schema,
    root_proof_atomic_attempt_consistency as consistency,
    root_proof_atomic_composition as composition,
    root_proof_issuance_reconciliation as reconciliation,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.postgresql_authenticated_issuer_history import PostgreSQLAuthenticatedIssuerHistory
from bot_core.postgresql_checkpoint_authority import PostgreSQLCheckpointAuthority
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.postgresql_issuer_history_schema import (
    PostgreSQLIssuerHistoryProvisioning,
    provision_postgresql_issuer_history,
    provision_postgresql_issuer_history_stream,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
)
from bot_core.security import hwid
from tests.security import (
    test_postgresql_root_proof_signing_intent as existing_intent_tests,
    test_root_proof_issuer_runtime as preflight_tests,
)
from tests.security._local_signing_platform import requires_native_custody_locking
from tests.security.test_postgresql_checkpoint_authority import BASE_DSN
from tests.security.test_postgresql_root_proof_retention import _privileged_mutation
from tests.security.test_root_proof_issuance_reconciliation import _history_boundary
from tests.security.test_root_proof_issuer_orchestration import (
    _durable_snapshot,
    _prohibit_mutations,
)

pytestmark = [pytest.mark.external_postgresql, requires_native_custody_locking]

# These fixtures preserve the previously reviewed upstream harness unchanged.
lineage_monkeypatch = existing_intent_tests.lineage_monkeypatch
lineage_path = existing_intent_tests.lineage_path
lineage_challenge = existing_intent_tests.lineage_challenge
lineage_integration = existing_intent_tests.lineage_integration
lineage_package = existing_intent_tests.lineage_package
lineage_lppi = existing_intent_tests.lineage_lppi
lineage_active = existing_intent_tests.lineage_active
lineage_committed = existing_intent_tests.lineage_committed
lineage_cha = existing_intent_tests.lineage_cha
lineage_reserved = existing_intent_tests.lineage_reserved


@pytest.fixture(scope="module")
def module_native_keyring(lineage_monkeypatch):
    # Restarting a test container recreates host veth MACs. Preserve one test
    # device identity for the shared keyring lifetime, while keeping the real
    # master-key fingerprint binding and all cryptographic checks unchanged.
    fingerprint = hwid.get_local_fingerprint()
    lineage_monkeypatch.setattr(hwid, "get_local_fingerprint", lambda: fingerprint)
    return preflight_tests.native_keyring.__wrapped__(lineage_monkeypatch)


@pytest.fixture(scope="module")
def genuine_attempt(lineage_reserved, module_native_keyring, lineage_path, lineage_monkeypatch):
    """Authenticate and sign the CHA request once; never mutate this fixture."""
    yield from preflight_tests.installed_signed_attempt.__wrapped__(
        lineage_reserved,
        module_native_keyring,
        lineage_path,
        lineage_monkeypatch,
        SimpleNamespace(param=("requester-key", "claimant-key")),
    )


def _connection(role, *, dsn=BASE_DSN):
    return PostgreSQLConnectionConfig(make_conninfo(dsn, user=role))


def _provider_pin(provider):
    if type(provider) is intent_storage.PostgreSQLRootProofSigningIntent:
        connection = provider._PostgreSQLRootProofSigningIntent__connection
        schema = provider._PostgreSQLRootProofSigningIntent__schema
    elif type(provider) is retention_storage.PostgreSQLRootProofRetention:
        connection = provider._PostgreSQLRootProofRetention__connection
        schema = provider._PostgreSQLRootProofRetention__schema
    else:
        connection, schema = provider._connection, provider._schema
    return composition.AtomicRootProofAuthorityPin(
        connection, schema, conninfo_to_dict(connection.dsn)["user"]
    )


@pytest.fixture
def atomic_boundary(genuine_attempt, tmp_path: Path, monkeypatch):
    """Fresh genuine issuer ports with no intent, signature or staged proof."""
    auth = genuine_attempt.current.reservation.authorization
    token = uuid.uuid4().hex[:12]
    intent_setup = intent_schema.PostgreSQLRootProofSigningIntentProvisioning(
        f"aqi_{token}", f"aqio_{token}", f"aqiw_{token}", auth.trust_domain
    )
    retention_setup = retention_schema.PostgreSQLRootProofRetentionProvisioning(
        f"aqr_{token}", f"aqro_{token}", f"aqrw_{token}", auth.trust_domain
    )
    transaction_owner_role = f"aqtx_{token}"
    directory = tmp_path / "root-custody"
    directory.mkdir()
    security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, auth.trust_domain)
    administrator = custody.NativeKeyringSigningSecretAdministrator(
        index_path=directory / "keyring-index.json",
        security=security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
    )
    custody.provision_local_signing_authority(
        directory, administrator, security=security, role=ProviderRole.ROOT_PROOF_SIGNING
    )
    signer = custody.LocalRootProofSigningProvider(
        directory, keyring_index_path=directory / "keyring-index.json", security=security
    )
    history_administrator = custody.NativeKeyringSigningSecretAdministrator(
        index_path=directory / "keyring-index.json",
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )
    custody.provision_local_signing_authority(
        directory,
        history_administrator,
        security=security,
        role=ProviderRole.HISTORY_ATTESTATION_SIGNING,
    )
    history_signer = custody.LocalHistoryAttestationSigningProvider(
        directory, keyring_index_path=directory / "keyring-index.json", security=security
    )
    scope = retention_storage.RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL,
        auth.environment,
        auth.trust_domain,
        auth.product_scope,
        "IndependentAccountGenesisRootProofIssuer",
    )
    setups = []
    owner_created = False
    try:
        intent_schema.provision_postgresql_root_proof_signing_intent(
            PostgreSQLConnectionConfig(BASE_DSN), intent_setup
        )
        setups.append(
            (intent_setup.schema, intent_setup.issuer_writer_role, intent_setup.schema_owner_role)
        )
        retention_schema.provision_postgresql_root_proof_retention(
            PostgreSQLConnectionConfig(BASE_DSN), retention_setup
        )
        setups.append(
            (
                retention_setup.schema,
                retention_setup.runtime_role,
                retention_setup.schema_owner_role,
            )
        )
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            assert conn.info.server_version // 10000 == 16
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(sql.Identifier(transaction_owner_role))
            )
            owner_created = True
        with _history_boundary(genuine_attempt, monkeypatch) as history_boundary:
            history_boundary.signer = history_signer
            history_boundary.checkpoint = PostgreSQLCheckpointAuthority(
                history_boundary.checkpoint._connection,
                schema=history_boundary.checkpoint._schema,
                checkpoint_id=history_boundary.checkpoint_id,
                stream=history_boundary.stream,
                history=history_boundary.history,
                verification_authorities=(history_signer,),
            )
            history_boundary.runtime = reconciliation._compose_reconciliation_runtime(
                genuine_attempt.authority, history_boundary.history, history_boundary.checkpoint
            )
            monkeypatch.setattr(
                reconciliation,
                "_configured_reconciliation_runtime",
                lambda: history_boundary.runtime,
            )
            intent = intent_storage.PostgreSQLRootProofSigningIntent(
                _connection(intent_setup.issuer_writer_role),
                schema=intent_setup.schema,
                scope=scope,
                verification_authority=signer,
                issuer_authority=genuine_attempt.authority,
            )
            retention = retention_storage.PostgreSQLRootProofRetention(
                _connection(retention_setup.runtime_role),
                schema=retention_setup.schema,
                scope=scope,
                verification_authority=signer,
            )
            ports = {
                "entitlement": genuine_attempt.authority.entitlement_registry,
                "requester": genuine_attempt.authority.requester_registry,
                "claimant": genuine_attempt.authority.claimant_registry,
                "signing_intent": intent,
                "retention": retention,
                "history": history_boundary.history,
                "checkpoint": history_boundary.checkpoint,
            }
            pins = {name: _provider_pin(port) for name, port in ports.items()}
            deployment = composition.AtomicRootProofDeployment(
                scope=scope,
                **pins,
                transaction_owner=_connection(transaction_owner_role),
                transaction_owner_role=transaction_owner_role,
            )
            yield SimpleNamespace(
                genuine=genuine_attempt,
                auth=auth,
                intent=intent,
                retention=retention,
                signer=signer,
                directory=directory,
                security=security,
                scope=scope,
                history_boundary=history_boundary,
                history=history_boundary.history,
                checkpoint=history_boundary.checkpoint,
                intent_setup=intent_setup,
                retention_setup=retention_setup,
                deployment=deployment,
            )
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            for schema, writer, owner in reversed(setups):
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema))
                )
                for role in (writer, owner):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
            if owner_created:
                conn.execute(
                    sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(transaction_owner_role))
                )


def _qualify(value, *, deployment=None, history=None, checkpoint=None):
    return composition.qualify_atomic_root_proof_composition(
        value.deployment if deployment is None else deployment,
        issuance_authority=value.genuine.authority,
        signing_intent=value.intent,
        retention=value.retention,
        history=value.history if history is None else history,
        checkpoint=value.checkpoint if checkpoint is None else checkpoint,
    )


def _no_grant(report):
    assert report.state == "NOT_READY"
    assert report.authorizes_operations is False
    assert "ATOMIC_COORDINATOR_NOT_IMPLEMENTED" in report.failures
    assert "SAME_CONNECTION_PARTICIPATION_NOT_IMPLEMENTED" in report.failures
    for name in ("root_proof", "root_proof_bytes", "atomic_commit", "execute", "grant"):
        assert not hasattr(report, name)


def _snapshot(value):
    with psycopg.connect(BASE_DSN) as conn:
        intents = conn.execute(
            sql.SQL("SELECT * FROM {}.intents ORDER BY attempt_key").format(
                sql.Identifier(value.intent_setup.schema)
            )
        ).fetchall()
        retained = conn.execute(
            sql.SQL("SELECT * FROM {}.prepared ORDER BY attempt_key").format(
                sql.Identifier(value.retention_setup.schema)
            )
        ).fetchall()
    custody_files = tuple(
        (str(path.relative_to(value.directory)), path.read_bytes())
        for path in sorted(value.directory.rglob("*"))
        if path.is_file()
    )
    return _durable_snapshot(value.history_boundary), intents, retained, custody_files


def _deny_mutations(value, monkeypatch):
    called = _prohibit_mutations(value.history_boundary, monkeypatch)

    def forbidden(*args, **kwargs):
        called.append(True)
        pytest.fail("read-only atomic qualification invoked an issuer mutation")

    for target, names in (
        (intent_storage.PostgreSQLRootProofSigningIntent, ("reserve",)),
        (retention_storage.PostgreSQLRootProofRetention, ("retain_prepared",)),
        (custody._LocalSigningProviderBase, ("_sign", "_sign_result")),
        (intent_storage, ("_new_root_proof_id",)),
    ):
        for name in names:
            monkeypatch.setattr(target, name, forbidden)
    return called


def test_same_actual_database_and_genuine_qualified_roles_remain_not_ready_without_coordinator(
    atomic_boundary, monkeypatch
):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    report = _qualify(value)
    _no_grant(report)
    assert len(report.authorities) == 8
    assert all(observation.qualified for observation in report.authorities)
    identities = [observation.physical_identity for observation in report.authorities]
    assert all(identity is not None for identity in identities)
    assert (
        len({(identity.system_identifier, identity.database_oid) for identity in identities}) == 1
    )
    assert "COMPLETE_PHYSICAL_DATABASE_IDENTITY_REQUIRED" not in report.failures
    assert "CROSS_CLUSTER_AUTHORITIES" not in report.failures
    assert "CROSS_DATABASE_AUTHORITIES" not in report.failures
    assert "HISTORY_SIGNER_CREDENTIAL_EVIDENCE_UNQUALIFIED" not in report.failures
    assert calls == []
    assert _snapshot(value) == before


@pytest.mark.parametrize("privilege", ["dangerous-builtin", "sequence-update"])
def test_transaction_owner_broad_privileges_are_refused(atomic_boundary, monkeypatch, privilege):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    role = sql.Identifier(value.deployment.transaction_owner_role)
    schema = f"aq_extra_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        if privilege == "dangerous-builtin":
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION pg_catalog.pg_read_file(text) TO {}").format(
                    role
                )
            )
        else:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
            conn.execute(sql.SQL("CREATE SEQUENCE {}.counter").format(sql.Identifier(schema)))
            conn.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), role)
            )
            conn.execute(
                sql.SQL("GRANT UPDATE ON SEQUENCE {}.counter TO {}").format(
                    sql.Identifier(schema), role
                )
            )
        try:
            report = _qualify(value)
            _no_grant(report)
            expected = (
                "TRANSACTION_OWNER_EXTRA_FUNCTION_GRANT_FORBIDDEN"
                if privilege == "dangerous-builtin"
                else "TRANSACTION_OWNER_SEQUENCE_WRITE_FORBIDDEN"
            )
            assert expected in report.failures
            owner = next(
                item for item in report.authorities if item.authority == "transaction_owner"
            )
            assert owner.qualified is False
        finally:
            if privilege == "dangerous-builtin":
                conn.execute(
                    sql.SQL(
                        "REVOKE EXECUTE ON FUNCTION pg_catalog.pg_read_file(text) FROM {}"
                    ).format(role)
                )
            else:
                conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
    assert calls == []
    assert _snapshot(value) == before


@pytest.mark.parametrize(
    "lifecycle", [custody.SigningKeyLifecycle.VERIFY_ONLY, custody.SigningKeyLifecycle.REVOKED]
)
def test_current_root_custody_lifecycle_is_requalified_without_mutation(
    atomic_boundary, monkeypatch, lifecycle
):
    value = atomic_boundary
    custody.transition_local_signing_lifecycle(
        value.directory,
        security=value.security,
        role=ProviderRole.ROOT_PROOF_SIGNING,
        target=lifecycle,
    )
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    report = _qualify(value)
    _no_grant(report)
    assert "PINNED_CUSTODY_ACTIVE_LIFECYCLE_REQUIRED" in report.failures
    assert calls == []
    assert _snapshot(value) == before


@contextmanager
def _topology_permission(value):
    """Test-only observation permission cannot widen reviewed issuer authority."""
    roles = [
        getattr(value.deployment, name).login_role
        for name in (
            "entitlement",
            "requester",
            "claimant",
            "signing_intent",
            "retention",
            "history",
            "checkpoint",
        )
    ] + [value.deployment.transaction_owner_role]
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        for role in roles:
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO {}").format(
                    sql.Identifier(role)
                )
            )
        try:
            yield
        finally:
            for role in roles:
                conn.execute(
                    sql.SQL(
                        "REVOKE EXECUTE ON FUNCTION pg_catalog.pg_control_system() FROM {}"
                    ).format(sql.Identifier(role))
                )


def test_same_actual_database_with_complete_identity_remains_not_ready_and_preserves_exact_acls(
    atomic_boundary, monkeypatch
):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    with _topology_permission(value):
        report = _qualify(value)
        _no_grant(report)
        identities = [observation.physical_identity for observation in report.authorities]
        assert all(identity is not None for identity in identities)
        assert (
            len({(identity.system_identifier, identity.database_oid) for identity in identities})
            == 1
        )
        assert "CROSS_CLUSTER_AUTHORITIES" not in report.failures
        assert "CROSS_DATABASE_AUTHORITIES" not in report.failures
        # Original source contracts reject this additional builtin function grant.
        assert "AUTHORITY_SIGNING_INTENT_QUALIFICATION_FAILED" in report.failures
        assert "AUTHORITY_RETENTION_QUALIFICATION_FAILED" in report.failures
    assert calls == []
    assert _snapshot(value) == before


@contextmanager
def _foreign_history(value, *, other_cluster=False):
    """Same stream/scope/schema spelling, but an actual different database identity."""
    if other_cluster:
        dsn = os.environ.get("CRYPTOHUNTER_TEST_POSTGRES_OTHER_DSN")
        if not dsn:
            pytest.fail(
                "CRYPTOHUNTER_TEST_POSTGRES_OTHER_DSN is required for the real second cluster"
            )
    else:
        dsn = BASE_DSN
    token = uuid.uuid4().hex[:12]
    database = None if other_cluster else f"atomic_other_{token}"
    if database is not None:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
        dsn = make_conninfo(dsn, dbname=database)
    original = value.history_boundary.history_setup
    setup = PostgreSQLIssuerHistoryProvisioning(
        original.schema, f"aqho_{token}", f"aqhr_{token}", f"aqha_{token}", value.auth.trust_domain
    )
    installed = False
    try:
        provision_postgresql_issuer_history(PostgreSQLConnectionConfig(dsn), setup)
        installed = True
        provision_postgresql_issuer_history_stream(
            _connection(setup.admin_role, dsn=dsn),
            schema=setup.schema,
            stream=value.history_boundary.stream,
        )
        with psycopg.connect(dsn, autocommit=True) as conn:
            assert conn.info.server_version // 10000 == 16
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO {}").format(
                    sql.Identifier(setup.runtime_role)
                )
            )
        history = PostgreSQLAuthenticatedIssuerHistory(
            _connection(setup.runtime_role, dsn=dsn),
            schema=setup.schema,
            stream=value.history_boundary.stream,
        )
        checkpoint = PostgreSQLCheckpointAuthority(
            value.checkpoint._connection,
            schema=value.checkpoint._schema,
            checkpoint_id=value.history_boundary.checkpoint_id,
            stream=value.history_boundary.stream,
            history=history,
            verification_authorities=(value.history_boundary.signer,),
        )
        yield history, checkpoint, replace(value.deployment, history=_provider_pin(history))
    finally:
        if installed:
            with psycopg.connect(dsn, autocommit=True) as conn:
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
                )
                conn.execute(
                    sql.SQL(
                        "REVOKE EXECUTE ON FUNCTION pg_catalog.pg_control_system() FROM {}"
                    ).format(sql.Identifier(setup.runtime_role))
                )
                for role in (setup.runtime_role, setup.admin_role, setup.schema_owner_role):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))
        if database is not None:
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                conn.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(database)))


@pytest.mark.parametrize(
    "other_cluster", [False, True], ids=["different-database", "different-cluster"]
)
def test_matching_scope_and_schema_strings_cannot_replace_actual_database_identity(
    atomic_boundary, monkeypatch, other_cluster
):
    value = atomic_boundary
    with (
        _foreign_history(value, other_cluster=other_cluster) as (history, checkpoint, deployment),
        _topology_permission(value),
    ):
        before = _snapshot(value)
        calls = _deny_mutations(value, monkeypatch)
        report = _qualify(value, deployment=deployment, history=history, checkpoint=checkpoint)
        _no_grant(report)
        expected = "CROSS_CLUSTER_AUTHORITIES" if other_cluster else "CROSS_DATABASE_AUTHORITIES"
        assert expected in report.failures
        identities = {
            observation.authority: observation.physical_identity
            for observation in report.authorities
        }
        assert identities["history"] is not None and identities["retention"] is not None
        if other_cluster:
            assert (
                identities["history"].system_identifier != identities["retention"].system_identifier
            )
        else:
            assert (
                identities["history"].system_identifier == identities["retention"].system_identifier
            )
            assert identities["history"].database_oid != identities["retention"].database_oid
        assert calls == []
        assert _snapshot(value) == before


@pytest.mark.parametrize(
    "change", ["authority-membership", "transaction-membership", "table-write"]
)
def test_acl_and_role_escalations_refuse_qualification_without_durable_mutation(
    atomic_boundary, monkeypatch, change
):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        if change == "authority-membership":
            target = value.intent_setup.issuer_writer_role
            grant = sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(value.intent_setup.schema_owner_role), sql.Identifier(target)
            )
            revoke = sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(value.intent_setup.schema_owner_role), sql.Identifier(target)
            )
        elif change == "transaction-membership":
            target = value.deployment.transaction_owner_role
            grant = sql.SQL("GRANT {} TO {}").format(
                sql.Identifier(value.intent_setup.schema_owner_role), sql.Identifier(target)
            )
            revoke = sql.SQL("REVOKE {} FROM {}").format(
                sql.Identifier(value.intent_setup.schema_owner_role), sql.Identifier(target)
            )
        else:
            table = sql.Identifier(value.intent_setup.schema, "intents")
            target = sql.Identifier(value.intent_setup.issuer_writer_role)
            grant = sql.SQL("GRANT INSERT ON {} TO {}").format(table, target)
            revoke = sql.SQL("REVOKE INSERT ON {} FROM {}").format(table, target)
        conn.execute(grant)
        try:
            report = _qualify(value)
            _no_grant(report)
            if change == "transaction-membership":
                assert any(
                    "TRANSACTION_OWNER" in failure and "MEMBERSHIP" in failure
                    for failure in report.failures
                )
            else:
                assert "AUTHORITY_SIGNING_INTENT_QUALIFICATION_FAILED" in report.failures
        finally:
            conn.execute(revoke)
    assert calls == []
    assert _snapshot(value) == before


def test_configured_login_and_caller_identity_substitution_are_not_authority(
    atomic_boundary, monkeypatch
):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    pin = replace(value.deployment.retention, login_role=value.retention_setup.schema_owner_role)
    report = _qualify(value, deployment=replace(value.deployment, retention=pin))
    _no_grant(report)
    assert any("RETENTION" in failure for failure in report.failures)
    forged = dict(
        issuance_authority=value.genuine.authority,
        signing_intent=value.intent,
        retention=value.retention,
        history=SimpleNamespace(**vars(value.history)),
        checkpoint=value.checkpoint,
    )
    refused = composition.qualify_atomic_root_proof_composition(value.deployment, **forged)
    _no_grant(refused)
    assert "EXACT_INTERNAL_COMPOSITION_REQUIRED" in refused.failures
    assert calls == []
    assert _snapshot(value) == before


@pytest.mark.parametrize("failure", ["identity-permission-denied", "login-unavailable"])
def test_unknown_or_unavailable_physical_identity_fails_closed(
    atomic_boundary, monkeypatch, failure
):
    value = atomic_boundary
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    role = sql.Identifier(value.retention_setup.runtime_role)
    with _topology_permission(value), psycopg.connect(BASE_DSN, autocommit=True) as conn:
        if failure == "identity-permission-denied":
            conn.execute("REVOKE EXECUTE ON FUNCTION pg_catalog.pg_control_system() FROM PUBLIC")
            conn.execute(
                sql.SQL("REVOKE EXECUTE ON FUNCTION pg_catalog.pg_control_system() FROM {}").format(
                    role
                )
            )
        else:
            conn.execute(sql.SQL("ALTER ROLE {} NOLOGIN").format(role))
        try:
            report = _qualify(value)
            _no_grant(report)
            expected = (
                "PHYSICAL_IDENTITY_RETENTION_UNAVAILABLE"
                if failure == "identity-permission-denied"
                else "AUTHORITY_RETENTION_PROBE_UNAVAILABLE"
            )
            assert expected in report.failures
            assert (
                next(
                    item for item in report.authorities if item.authority == "retention"
                ).physical_identity
                is None
            )
        finally:
            if failure == "identity-permission-denied":
                conn.execute(
                    sql.SQL(
                        "GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO {}"
                    ).format(role)
                )
                conn.execute("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO PUBLIC")
            else:
                conn.execute(sql.SQL("ALTER ROLE {} LOGIN").format(role))
    assert calls == []
    assert _snapshot(value) == before


def test_restart_uses_new_providers_and_requalifies_current_authorities(
    atomic_boundary, monkeypatch
):
    value = atomic_boundary
    first = _qualify(value)
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    container = os.environ.get("CRYPTOHUNTER_TEST_POSTGRES_CONTAINER")
    if not container:
        pytest.fail(
            "CRYPTOHUNTER_TEST_POSTGRES_CONTAINER is required for the real PostgreSQL restart"
        )
    with psycopg.connect(BASE_DSN) as conn:
        started = conn.execute("SELECT pg_catalog.pg_postmaster_start_time()").fetchone()[0]
    docker_environment = os.environ.copy()
    for name in (
        "DOCKER_HOST",
        "DOCKER_CONTEXT",
        "DOCKER_TLS",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
    ):
        docker_environment.pop(name, None)
    subprocess.run(
        ["docker", "--host=unix:///var/run/docker.sock", "restart", container],
        check=True,
        capture_output=True,
        timeout=30,
        env=docker_environment,
    )
    deadline = time.monotonic() + 30
    while True:
        try:
            with psycopg.connect(make_conninfo(BASE_DSN, connect_timeout=1)) as conn:
                restarted = conn.execute("SELECT pg_catalog.pg_postmaster_start_time()").fetchone()[
                    0
                ]
            assert restarted > started
            break
        except psycopg.OperationalError:
            if time.monotonic() >= deadline:
                pytest.fail(
                    "PostgreSQL did not become available after the actual container restart"
                )
            time.sleep(0.25)
    value.intent = intent_storage.PostgreSQLRootProofSigningIntent(
        value.deployment.signing_intent.connection,
        schema=value.intent_setup.schema,
        scope=value.scope,
        verification_authority=value.signer,
        issuer_authority=value.genuine.authority,
    )
    value.retention = retention_storage.PostgreSQLRootProofRetention(
        value.deployment.retention.connection,
        schema=value.retention_setup.schema,
        scope=value.scope,
        verification_authority=value.signer,
    )
    value.history = PostgreSQLAuthenticatedIssuerHistory(
        value.deployment.history.connection,
        schema=value.history._schema,
        stream=value.history_boundary.stream,
    )
    value.checkpoint = PostgreSQLCheckpointAuthority(
        value.deployment.checkpoint.connection,
        schema=value.checkpoint._schema,
        checkpoint_id=value.history_boundary.checkpoint_id,
        stream=value.history_boundary.stream,
        history=value.history,
        verification_authorities=(value.history_boundary.signer,),
    )
    second = _qualify(value)
    _no_grant(second)
    assert second is not first
    assert second.authorities == first.authorities
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute("REVOKE EXECUTE ON FUNCTION pg_catalog.pg_control_system() FROM PUBLIC")
        try:
            current = _qualify(value)
            assert "PHYSICAL_IDENTITY_HISTORY_UNAVAILABLE" in current.failures
            assert (
                next(
                    item for item in current.authorities if item.authority == "history"
                ).physical_identity
                is None
            )
        finally:
            conn.execute("GRANT EXECUTE ON FUNCTION pg_catalog.pg_control_system() TO PUBLIC")
    assert calls == []
    assert _snapshot(value) == before


@pytest.fixture(scope="module")
def retained_reserved_boundary(genuine_attempt, lineage_path, lineage_monkeypatch):
    """Reserve one genuine immutable intent for every independent retention case."""
    path = lineage_path / "immutable-reserved-intent"
    path.mkdir()
    iterator = atomic_boundary.__wrapped__(genuine_attempt, path, lineage_monkeypatch)
    value = next(iterator)
    try:
        value.reserved = value.intent.reserve(value.genuine.binding, value.genuine.opaque)
        yield value
    finally:
        iterator.close()


@pytest.fixture
def consistency_boundary(retained_reserved_boundary):
    """Share only RESERVED; each case has an isolated, historically verified STAGED store."""
    shared = retained_reserved_boundary
    token = uuid.uuid4().hex[:12]
    setup = retention_schema.PostgreSQLRootProofRetentionProvisioning(
        f"aqs_{token}", f"aqso_{token}", f"aqsw_{token}", shared.auth.trust_domain
    )
    retention_schema.provision_postgresql_root_proof_retention(
        PostgreSQLConnectionConfig(BASE_DSN), setup
    )
    try:
        value = SimpleNamespace(**vars(shared))
        value.retention_setup = setup
        value.retention = retention_storage.PostgreSQLRootProofRetention(
            _connection(setup.runtime_role),
            schema=setup.schema,
            scope=value.scope,
            verification_authority=value.signer,
        )
        yield value
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(setup.schema))
            )
            for role in (setup.runtime_role, setup.schema_owner_role):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _stage_reserved(value, *, different_root=False):
    payload = value.reserved.canonical_payload
    if different_root:
        document = parse_canonical(payload)
        document["root_proof_id"] = "rpf_018f3e70-7b5e-7c21-8b9a-0123456789ab"
        assert document["root_proof_id"] != value.reserved.root_proof_id
        payload = canonical_json_bytes(document)
    # Offline fixture bytes use the generated test key directly. Production
    # signing operations are prohibited throughout every validator observation.
    _, fixture_private_key = value.signer._load_material()
    signature = fixture_private_key.sign(retention_storage.ROOT_PROOF_SIGNING_DOMAIN + payload)
    return value.retention.retain_prepared(
        authorization=value.auth,
        retained_request=value.genuine.retained,
        canonical_payload=payload,
        signature=signature,
    )


def _consistency_no_grant(report):
    assert report.disposition == "NOT_READY"
    assert report.authorizes_operations is False
    assert report.root_proof_issuance_available is False
    assert report.atomic_global_snapshot is False
    for name in (
        "root_proof",
        "root_proof_bytes",
        "signature",
        "authority_authenticated_evidence_reference",
    ):
        assert not hasattr(report, name)


@pytest.mark.parametrize(
    ("scenario", "expected"),
    [
        (
            "reserved-only",
            consistency.RootProofAtomicAttemptConsistencyOutcome.RESERVED_WITHOUT_STAGED,
        ),
        ("exact-staged", consistency.RootProofAtomicAttemptConsistencyOutcome.EXACT_MATCH),
        ("different-valid-root", consistency.RootProofAtomicAttemptConsistencyOutcome.CONFLICT),
        ("corrupted-signature", consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT),
        ("changed-payload", consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT),
        ("changed-binding-digest", consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT),
        ("changed-attempt-identity", consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT),
        ("changed-scope", consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT),
        (
            "changed-credential-version",
            consistency.RootProofAtomicAttemptConsistencyOutcome.CORRUPT,
        ),
    ],
)
def test_genuine_reserved_and_staged_consistency_never_authorizes_issuance(
    consistency_boundary, monkeypatch, scenario, expected
):
    value = consistency_boundary
    if scenario != "reserved-only":
        staged = _stage_reserved(value, different_root=scenario == "different-valid-root")
        if scenario in {
            "corrupted-signature",
            "changed-payload",
            "changed-binding-digest",
            "changed-attempt-identity",
            "changed-scope",
            "changed-credential-version",
        }:
            signature = staged.signature
            payload = staged.canonical_payload
            envelope = parse_canonical(staged.canonical_envelope)
            if scenario == "corrupted-signature":
                signature = bytes([signature[0] ^ 1]) + signature[1:]
            elif scenario == "changed-payload":
                document = parse_canonical(payload)
                document["account_id"] = "acct_018f3e70-7b5a-7c21-8b9a-0123456789ac"
                payload = canonical_json_bytes(document)
                _, fixture_private_key = value.signer._load_material()
                signature = fixture_private_key.sign(
                    retention_storage.ROOT_PROOF_SIGNING_DOMAIN + payload
                )
                envelope["canonical_payload_sha256"] = hashlib.sha256(payload).hexdigest()
            elif scenario == "changed-binding-digest":
                envelope["authorization"]["initial_binding_digest_sha256"] = "f" * 64
            elif scenario == "changed-attempt-identity":
                envelope["issuance_attempt_id"] = "rpa_018f3e70-7b5d-7c21-8b9a-0123456789ac"
            elif scenario == "changed-scope":
                envelope["scope"]["product_scope"] = "UnrelatedProduct"
            elif scenario == "changed-credential-version":
                envelope["credential_snapshot"]["key_version"] += 1
            envelope["signature_ed25519_base64url"] = retention_storage.encode_signature(signature)
            statement = sql.SQL(
                "UPDATE {}.prepared SET canonical_envelope=%s,canonical_payload=%s,signature=%s"
            ).format(sql.Identifier(value.retention_setup.schema))
            parameters = canonical_json_bytes(envelope), payload, signature
            with psycopg.connect(BASE_DSN, autocommit=True) as conn:
                with pytest.raises(
                    psycopg.errors.InsufficientPrivilege, match="STAGED bytes are immutable"
                ):
                    conn.execute(statement, parameters)
            # Existing immutable triggers stop normal SQL, including a direct
            # UPDATE by the test administrator. Simulate privileged corruption
            # through the existing harness, which restores ALWAYS triggers in
            # the same transaction before any qualification or validator read.
            _privileged_mutation(
                SimpleNamespace(setup=value.retention_setup), statement, parameters
            )
    before = _snapshot(value)
    calls = _deny_mutations(value, monkeypatch)
    report = consistency.validate_root_proof_atomic_attempt_consistency(
        value.intent,
        value.retention,
        binding=value.genuine.binding,
        signed_attempt=value.genuine.opaque,
    )
    _consistency_no_grant(report)
    assert report.outcome is expected
    if expected in {
        consistency.RootProofAtomicAttemptConsistencyOutcome.RESERVED_WITHOUT_STAGED,
        consistency.RootProofAtomicAttemptConsistencyOutcome.EXACT_MATCH,
    }:
        assert report.root_proof_id == value.reserved.root_proof_id
        assert len(report.observations) == 2
        assert report.observations[0] == report.observations[1]
    else:
        assert report.root_proof_id is None
    assert calls == []
    assert _snapshot(value) == before


def test_standalone_owner_probe_refuses_public_execute_on_protected_catalog_function():
    """A real PUBLIC grant must be caught even without an ACL naming the owner."""
    token = uuid.uuid4().hex[:12]
    owner_role = f"aq_public_{token}"
    connection = _connection(owner_role)
    scope = retention_storage.RootProofRetentionScope(
        SecurityProfile.PRODUCTION_LOCAL,
        "PRODUCTION",
        "standalone-owner-test",
        "CryptoHunter",
        "IndependentAccountGenesisRootProofIssuer",
    )
    names = (
        "entitlement",
        "requester",
        "claimant",
        "signing_intent",
        "retention",
        "history",
        "checkpoint",
    )
    pins = {
        name: composition.AtomicRootProofAuthorityPin(
            connection, f"absent_{index}_{token}", f"absent_role_{index}_{token}"
        )
        for index, name in enumerate(names)
    }
    deployment = composition.AtomicRootProofDeployment(
        scope=scope,
        **pins,
        transaction_owner=connection,
        transaction_owner_role=owner_role,
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "CREATE ROLE {} LOGIN NOINHERIT NOSUPERUSER NOCREATEDB "
                "NOCREATEROLE NOREPLICATION NOBYPASSRLS"
            ).format(sql.Identifier(owner_role))
        )
        original_acl = conn.execute(
            "SELECT proacl::text FROM pg_catalog.pg_proc "
            "WHERE oid='pg_catalog.pg_read_file(text)'::pg_catalog.regprocedure"
        ).fetchone()
        try:
            assert conn.execute(
                "SELECT pg_catalog.has_function_privilege(%s,"
                "'pg_catalog.pg_read_file(text)','EXECUTE')",
                (owner_role,),
            ).fetchone() == (False,)
            baseline, _ = composition._probe_transaction_owner(deployment)
            assert baseline.qualified is True
            conn.execute("GRANT EXECUTE ON FUNCTION pg_catalog.pg_read_file(text) TO PUBLIC")
            try:
                observation, failures = composition._probe_transaction_owner(deployment)
                assert observation.qualified is False
                assert "TRANSACTION_OWNER_RESTRICTED_CATALOG_EXECUTE_FORBIDDEN" in failures
                assert conn.execute(
                    "SELECT pg_catalog.has_function_privilege(%s,"
                    "'pg_catalog.pg_read_file(text)','EXECUTE')",
                    (owner_role,),
                ).fetchone() == (True,)
            finally:
                conn.execute("REVOKE EXECUTE ON FUNCTION pg_catalog.pg_read_file(text) FROM PUBLIC")
            assert (
                conn.execute(
                    "SELECT proacl::text FROM pg_catalog.pg_proc "
                    "WHERE oid='pg_catalog.pg_read_file(text)'::pg_catalog.regprocedure"
                ).fetchone()
                == original_acl
            )
        finally:
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(owner_role)))
