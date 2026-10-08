"""Real PostgreSQL checks for the separate pre-account public credential authorities."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Barrier

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege
from psycopg.types.json import Jsonb

import bot_core.postgresql_preaccount_credentials as credentials
from bot_core.postgresql_entitlement_registry import (
    PostgreSQLConnectionConfig,
    RegistryQualificationError,
)
from bot_core.root_proof_issuer_substrate import (
    ClaimantIdentityRegistry,
    CredentialSemanticRole,
    ProviderQualificationPolicy,
    ProviderRole,
    RequesterCredentialRegistry,
    SecurityProfile,
    public_key_material_identity,
)

pytestmark = pytest.mark.external_postgresql
BASE_DSN = os.environ.get(
    "DUDZIAN_TEST_POSTGRES_DSN",
    "host=127.0.0.1 port=55432 dbname=postgres user=postgres",
)
REQUESTER = "CryptoHunterAccountAuthority"
ENVIRONMENT = "PRODUCTION"
TRUST_DOMAIN = "td_preaccount"


def _connection(role: str) -> PostgreSQLConnectionConfig:
    return PostgreSQLConnectionConfig(make_conninfo(BASE_DSN, user=role))


@dataclass(frozen=True)
class _RegistryPair:
    setup: credentials.PostgreSQLPreaccountRegistryProvisioning
    requester: credentials.PostgreSQLRequesterCredentialRegistryProvider
    claimant: credentials.PostgreSQLClaimantIdentityRegistryProvider
    requester_admin: credentials.PostgreSQLRequesterCredentialProvisioningAdminProvider
    claimant_admin: credentials.PostgreSQLClaimantIdentityProvisioningAdminProvider


def _providers(setup: credentials.PostgreSQLPreaccountRegistryProvisioning) -> _RegistryPair:
    requester = setup.requester
    claimant = setup.claimant
    return _RegistryPair(
        setup,
        credentials.PostgreSQLRequesterCredentialRegistryProvider(
            _connection(requester.runtime_role),
            schema=requester.schema,
            environment=requester.environment,
            trust_domain=requester.trust_domain,
        ),
        credentials.PostgreSQLClaimantIdentityRegistryProvider(
            _connection(claimant.runtime_role),
            schema=claimant.schema,
            environment=claimant.environment,
            trust_domain=claimant.trust_domain,
        ),
        credentials.PostgreSQLRequesterCredentialProvisioningAdminProvider(
            _connection(requester.admin_role),
            schema=requester.schema,
            environment=requester.environment,
            trust_domain=requester.trust_domain,
        ),
        credentials.PostgreSQLClaimantIdentityProvisioningAdminProvider(
            _connection(claimant.admin_role),
            schema=claimant.schema,
            environment=claimant.environment,
            trust_domain=claimant.trust_domain,
        ),
    )


@contextmanager
def _isolated_pair():
    suffix = uuid.uuid4().hex[:10]
    requester = credentials.PostgreSQLCredentialRegistryProvisioning(
        f"pqr_{suffix}",
        f"pqro_{suffix}",
        f"pqrr_{suffix}",
        f"pqra_{suffix}",
        ENVIRONMENT,
        TRUST_DOMAIN,
    )
    claimant = credentials.PostgreSQLCredentialRegistryProvisioning(
        f"pqc_{suffix}",
        f"pqco_{suffix}",
        f"pqcr_{suffix}",
        f"pqca_{suffix}",
        ENVIRONMENT,
        TRUST_DOMAIN,
    )
    setup = credentials.PostgreSQLPreaccountRegistryProvisioning(requester, claimant)
    try:
        credentials.provision_postgresql_preaccount_registries(
            PostgreSQLConnectionConfig(BASE_DSN), setup
        )
        yield _providers(setup)
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            for authority in (requester, claimant):
                conn.execute(
                    sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                        sql.Identifier(authority.schema)
                    )
                )
            for authority in (requester, claimant):
                for role in (
                    authority.runtime_role,
                    authority.admin_role,
                    authority.schema_owner_role,
                ):
                    conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


@pytest.fixture
def registry():
    with _isolated_pair() as pair:
        yield pair


def _authority(pair: _RegistryPair, kind: str):
    if kind == "requester":
        return pair.setup.requester, pair.requester, pair.requester_admin, REQUESTER
    return pair.setup.claimant, pair.claimant, pair.claimant_admin, "deployment-security-authority"


def _provision(admin, principal: str, *, tag: str = "one", key: bytes = b"A" * 32):
    return admin.provision_credential(
        principal_id=principal,
        credential_id=f"credential-{tag}",
        key_id=f"key-{tag}",
        key_version=1,
        public_key=key,
    )


def _race(first, second):
    barrier = Barrier(2)

    def invoke(operation):
        barrier.wait(timeout=10)
        try:
            return operation()
        except credentials.CredentialConflictError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = (pool.submit(invoke, first), pool.submit(invoke, second))
        return tuple(future.result(timeout=30) for future in futures)


def test_real_server_durability_and_separate_authorities(registry: _RegistryPair) -> None:
    setup = registry.setup
    assert setup.requester.schema != setup.claimant.schema
    roles = tuple(
        role
        for authority in (setup.requester, setup.claimant)
        for role in (authority.schema_owner_role, authority.runtime_role, authority.admin_role)
    )
    assert len(set(roles)) == 6
    with psycopg.connect(BASE_DSN) as conn:
        row = conn.execute(
            "SELECT version(),current_setting('fsync'),current_setting('synchronous_commit')"
        ).fetchone()
        assert row is not None and "PostgreSQL 16" in row[0]
        assert row[1:] == ("on", "on")
        for authority in (setup.requester, setup.claimant):
            role_login = dict(
                conn.execute(
                    "SELECT rolname,rolcanlogin FROM pg_roles WHERE rolname=ANY(%s)",
                    ([authority.schema_owner_role, authority.runtime_role, authority.admin_role],),
                ).fetchall()
            )
            assert role_login == {
                authority.schema_owner_role: False,
                authority.runtime_role: True,
                authority.admin_role: True,
            }
            owners = conn.execute(
                "SELECT DISTINCT r.rolname FROM pg_class c "
                "JOIN pg_namespace n ON n.oid=c.relnamespace "
                "JOIN pg_roles r ON r.oid=c.relowner WHERE n.nspname=%s",
                (authority.schema,),
            ).fetchall()
            assert owners == [(authority.schema_owner_role,)]
            public_execute = conn.execute(
                "SELECT count(*) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
                "CROSS JOIN LATERAL aclexplode(coalesce(p.proacl,acldefault('f',p.proowner))) a "
                "WHERE n.nspname=%s AND a.grantee=0 AND a.privilege_type='EXECUTE'",
                (authority.schema,),
            ).fetchone()
            assert public_execute == (0,)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_schema_owner_cannot_authenticate_and_admin_functions_keep_owner_authority(
    registry, kind
) -> None:
    setup, runtime, admin, principal = _authority(registry, kind)
    with pytest.raises(psycopg.OperationalError, match="not permitted to log in"):
        with psycopg.connect(_connection(setup.schema_owner_role).dsn):
            pytest.fail("NOLOGIN schema owner unexpectedly authenticated")
    first = _provision(admin, principal)
    assert runtime.public_key(first.key_id) == first.public_key
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        function_owner = conn.execute(
            "SELECT r.rolname,r.rolcanlogin,p.prosecdef FROM pg_proc p "
            "JOIN pg_namespace n ON n.oid=p.pronamespace JOIN pg_roles r ON r.oid=p.proowner "
            "WHERE n.nspname=%s AND p.proname='apply'",
            (setup.schema,),
        ).fetchone()
        assert function_owner == (setup.schema_owner_role, False, True)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_schema_owner_login_escalation_is_detected_by_both_authorities(registry, kind) -> None:
    setup, _, _, _ = _authority(registry, kind)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("ALTER ROLE {} LOGIN").format(sql.Identifier(setup.schema_owner_role)))
    for provider in (
        registry.requester,
        registry.claimant,
        registry.requester_admin,
        registry.claimant_admin,
    ):
        with pytest.raises(RegistryQualificationError, match="role attributes"):
            provider.retained_history("qualification-probe")


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_exact_provider_roles_capabilities_and_authoritative_public_keys(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    record = _provision(admin, principal)
    protocol = RequesterCredentialRegistry if kind == "requester" else ClaimantIdentityRegistry
    assert isinstance(runtime, protocol)
    assert not isinstance(admin, protocol)
    expected_role = (
        ProviderRole.REQUESTER_CREDENTIAL_REGISTRY
        if kind == "requester"
        else ProviderRole.CLAIMANT_IDENTITY_REGISTRY
    )
    assert runtime.identity.role is expected_role
    assert runtime.capabilities.implemented
    assert runtime.capabilities.authoritative_reads
    assert runtime.capabilities.durable_state
    assert (
        ProviderQualificationPolicy().failures_for(
            SecurityProfile.PRODUCTION_LOCAL, runtime.identity, runtime.capabilities
        )
        == ()
    )
    for method in ("provision_credential", "rotate_credential", "transition_lifecycle", "sign"):
        assert not hasattr(runtime, method)
    assert "user=" not in repr(runtime)
    assert runtime.public_key(record.credential_id) == b"A" * 32
    assert runtime.public_key(record.key_id) == b"A" * 32
    identities = runtime.credential_identities()
    assert len(identities) == 1
    identity = identities[0]
    assert identity.semantic_role is (
        CredentialSemanticRole.ROOT_PROOF_REQUESTER
        if kind == "requester"
        else CredentialSemanticRole.ROOT_PROOF_CLAIMANT
    )
    assert identity.key_material_identity == public_key_material_identity(b"A" * 32)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_provider_cannot_authenticate_as_admin_owner_superuser_or_other_runtime(
    registry, kind
) -> None:
    setup, runtime, admin, _ = _authority(registry, kind)
    other = registry.setup.claimant if kind == "requester" else registry.setup.requester
    arguments = {"schema": setup.schema, "environment": ENVIRONMENT, "trust_domain": TRUST_DOMAIN}
    for role in ("postgres", setup.schema_owner_role, setup.admin_role, other.runtime_role):
        with pytest.raises(RegistryQualificationError):
            type(runtime)(_connection(role), **arguments)
    for role in ("postgres", setup.schema_owner_role, setup.runtime_role, other.admin_role):
        with pytest.raises(RegistryQualificationError):
            type(admin)(_connection(role), **arguments)
    laundering = PostgreSQLConnectionConfig(
        make_conninfo(BASE_DSN, options=f"-c role={setup.runtime_role}")
    )
    with pytest.raises(RegistryQualificationError):
        type(runtime)(laundering, **arguments)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_runtime_and_admin_cannot_write_authority_tables_or_ddl(registry, kind) -> None:
    setup, _, _, _ = _authority(registry, kind)
    statements = (
        sql.SQL("CREATE TABLE {}.unauthorized(value int)").format(sql.Identifier(setup.schema)),
        sql.SQL("UPDATE {}.credentials SET public_key=%s").format(sql.Identifier(setup.schema)),
        sql.SQL("DELETE FROM {}.history").format(sql.Identifier(setup.schema)),
        sql.SQL("DELETE FROM {}.current_credentials").format(sql.Identifier(setup.schema)),
        sql.SQL("DELETE FROM {}.metadata").format(sql.Identifier(setup.schema)),
        sql.SQL("ALTER TABLE {}.credentials ADD COLUMN unauthorized text").format(
            sql.Identifier(setup.schema)
        ),
        sql.SQL("DROP TABLE {}.history CASCADE").format(sql.Identifier(setup.schema)),
    )
    for role in (setup.runtime_role, setup.admin_role):
        with psycopg.connect(_connection(role).dsn, autocommit=True) as conn:
            for statement in statements:
                with pytest.raises(InsufficientPrivilege):
                    conn.execute(
                        statement, (b"Z" * 32,) if "public_key" in statement.as_string() else None
                    )
    with psycopg.connect(_connection(setup.runtime_role).dsn, autocommit=True) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute(
                sql.SQL("SELECT {}.apply('{{}}'::jsonb)").format(sql.Identifier(setup.schema))
            )


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize("role_field", ["runtime_role", "admin_role"])
def test_column_update_privilege_is_detected_live(registry, kind, role_field) -> None:
    setup, runtime, admin, _ = _authority(registry, kind)
    role = getattr(setup, role_field)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("GRANT UPDATE (public_key) ON {}.credentials TO {}").format(
                sql.Identifier(setup.schema), sql.Identifier(role)
            )
        )
    for provider in (runtime, admin):
        with pytest.raises(RegistryQualificationError):
            provider.retained_history("qualification-probe")


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize("role_field", ["runtime_role", "admin_role"])
def test_effective_database_create_privilege_is_detected_live(registry, kind, role_field) -> None:
    setup, runtime, admin, _ = _authority(registry, kind)
    role = getattr(setup, role_field)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        database = conn.execute("SELECT current_database()").fetchone()
        assert database is not None
        conn.execute(
            sql.SQL("GRANT CREATE ON DATABASE {} TO {}").format(
                sql.Identifier(database[0]), sql.Identifier(role)
            )
        )
    try:
        for provider in (runtime, admin):
            with pytest.raises(RegistryQualificationError):
                provider.retained_history("qualification-probe")
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("REVOKE CREATE ON DATABASE {} FROM {}").format(
                    sql.Identifier(database[0]), sql.Identifier(role)
                )
            )


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize("direction", ["incoming", "outgoing"])
def test_role_membership_escalation_is_detected_live(registry, kind, direction) -> None:
    setup, runtime, admin, _ = _authority(registry, kind)
    helper = f"pqm_{uuid.uuid4().hex[:10]}"
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE ROLE {} LOGIN NOINHERIT").format(sql.Identifier(helper)))
        member, authority = (
            (helper, setup.runtime_role)
            if direction == "incoming"
            else (setup.runtime_role, helper)
        )
        conn.execute(
            sql.SQL("GRANT {} TO {} WITH INHERIT FALSE, SET TRUE").format(
                sql.Identifier(authority), sql.Identifier(member)
            )
        )
    try:
        for provider in (runtime, admin):
            with pytest.raises(RegistryQualificationError):
                provider.retained_history("qualification-probe")
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("REVOKE {} FROM {}").format(
                    sql.Identifier(authority), sql.Identifier(member)
                )
            )
            conn.execute(sql.SQL("DROP ROLE {}").format(sql.Identifier(helper)))


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_namespace_and_unknown_schema_version_fail_closed(registry, kind) -> None:
    setup, runtime, _, _ = _authority(registry, kind)
    for environment, trust_domain in (("TEST", TRUST_DOMAIN), (ENVIRONMENT, "td_elsewhere")):
        with pytest.raises((RegistryQualificationError, ValueError)):
            type(runtime)(
                _connection(setup.runtime_role),
                schema=setup.schema,
                environment=environment,
                trust_domain=trust_domain,
            )
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL("UPDATE {}.metadata SET schema_version=999").format(
                sql.Identifier(setup.schema)
            )
        )
    with pytest.raises(RegistryQualificationError):
        runtime.retained_history("qualification-probe")


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_exact_provisioning_retries_and_material_conflicts(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)
    assert _provision(admin, principal) == first
    assert runtime.retained_history(principal) == (first,)
    with pytest.raises(credentials.CredentialConflictError):
        _provision(admin, principal, key=b"B" * 32)
    assert runtime.retained_history(principal) == (first,)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_concurrent_first_provisioning_has_one_durable_winner(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    results = _race(
        lambda: _provision(admin, principal, tag="left", key=b"L" * 32),
        lambda: _provision(admin, principal, tag="right", key=b"R" * 32),
    )
    winners = [result for result in results if not isinstance(result, Exception)]
    assert len(winners) == 1
    assert runtime.retained_history(principal) == (winners[0],)
    assert winners[0].lifecycle is credentials.CredentialLifecycle.ACTIVE


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_concurrent_rotation_retains_old_material_and_one_current_winner(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)

    def rotate(tag: str, key: bytes):
        return admin.rotate_credential(
            principal_id=principal,
            credential_id=f"credential-{tag}",
            key_id=f"key-{tag}",
            key_version=2,
            public_key=key,
            expected_revision=first.registry_revision,
        )

    results = _race(lambda: rotate("left", b"L" * 32), lambda: rotate("right", b"R" * 32))
    assert sum(not isinstance(result, Exception) for result in results) == 1
    history = runtime.retained_history(principal)
    assert len(history) == 3
    assert [item.registry_revision for item in history] == [1, 2, 3]
    assert [item.lifecycle_generation for item in history] == [1, 2, 3]
    assert [item.lifecycle for item in history] == [
        credentials.CredentialLifecycle.ACTIVE,
        credentials.CredentialLifecycle.VERIFY_ONLY,
        credentials.CredentialLifecycle.ACTIVE,
    ]
    assert [item.key_version for item in history] == [1, 1, 2]
    assert runtime.public_key(first.credential_id) == b"A" * 32
    assert runtime.historical_generation(first.credential_id, 1) == first
    if kind == "requester":
        assert runtime.historical_requester_credential(first.credential_id) == history[1]
        with pytest.raises(credentials.CredentialResolutionError):
            runtime.active_requester_credential("different-principal")
    else:
        assert runtime.historical_claimant(principal, 1) == first
        assert runtime.historical_claimant(principal, 2) == history[1]
        with pytest.raises(credentials.CredentialResolutionError):
            runtime.historical_claimant(principal, 999)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_concurrent_lifecycle_transition_and_stale_revision_fail_closed(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)
    results = _race(
        lambda: admin.transition_lifecycle(
            principal_id=principal,
            lifecycle=credentials.CredentialLifecycle.VERIFY_ONLY,
            expected_revision=first.registry_revision,
        ),
        lambda: admin.transition_lifecycle(
            principal_id=principal,
            lifecycle=credentials.CredentialLifecycle.REVOKED,
            expected_revision=first.registry_revision,
        ),
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    history = runtime.retained_history(principal)
    assert len(history) == 2
    assert history[-1].lifecycle is not credentials.CredentialLifecycle.ACTIVE
    assert history[-1].key_version == first.key_version
    assert history[-1].lifecycle_generation == first.lifecycle_generation + 1
    assert (
        admin.transition_lifecycle(
            principal_id=principal,
            lifecycle=history[-1].lifecycle,
            expected_revision=first.registry_revision,
        )
        == history[-1]
    )
    conflicting_lifecycle = (
        credentials.CredentialLifecycle.REVOKED
        if history[-1].lifecycle is credentials.CredentialLifecycle.VERIFY_ONLY
        else credentials.CredentialLifecycle.VERIFY_ONLY
    )
    with pytest.raises(credentials.CredentialConflictError):
        admin.transition_lifecycle(
            principal_id=principal,
            lifecycle=conflicting_lifecycle,
            expected_revision=first.registry_revision,
        )
    assert runtime.retained_history(principal) == history


def test_cross_role_material_alias_is_rejected_even_with_distinct_labels(registry) -> None:
    _provision(registry.requester_admin, REQUESTER, tag="requester")
    with pytest.raises(credentials.CredentialConflictError):
        _provision(registry.claimant_admin, "deployment-security-authority", tag="claimant")
    assert registry.claimant.retained_history("deployment-security-authority") == ()


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_historical_key_revocation_retains_current_active_replacement(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)
    rotated = admin.rotate_credential(
        principal_id=principal,
        credential_id="credential-next",
        key_id="key-next",
        key_version=2,
        public_key=b"B" * 32,
        expected_revision=first.registry_revision,
    )
    revoked = admin.transition_lifecycle(
        principal_id=principal,
        credential_id=first.credential_id,
        lifecycle=credentials.CredentialLifecycle.REVOKED,
        expected_revision=2,
    )
    assert revoked.credential_id == first.credential_id
    assert revoked.lifecycle is credentials.CredentialLifecycle.REVOKED
    assert revoked.registry_revision == 4 and revoked.lifecycle_generation == 4
    assert runtime.retained_history(principal)[-1] == revoked
    current = (
        runtime.active_requester_credential(principal)
        if kind == "requester"
        else runtime.resolve_claimant(principal)
    )
    current_key = current.requester_key_id if kind == "requester" else current.claimant_key_id
    assert current_key == rotated.key_id
    assert current.registry_revision == rotated.registry_revision
    assert runtime.public_key(first.credential_id) == first.public_key
    assert runtime.historical_generation(first.credential_id, 1) == first
    assert runtime.historical_generation(first.credential_id, 4) == revoked
    if kind == "requester":
        assert runtime.historical_requester_credential(first.credential_id) == revoked
    else:
        assert runtime.historical_claimant(principal, 4) == revoked


def test_concurrent_cross_role_material_alias_has_one_authoritative_winner(registry) -> None:
    results = _race(
        lambda: _provision(registry.requester_admin, REQUESTER, tag="requester"),
        lambda: _provision(
            registry.claimant_admin, "deployment-security-authority", tag="claimant"
        ),
    )
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert (
        len(registry.requester.retained_history(REQUESTER))
        + len(registry.claimant.retained_history("deployment-security-authority"))
        == 1
    )


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_reconnect_retains_exact_lifecycle_history_and_old_public_key(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)
    rotated = admin.rotate_credential(
        principal_id=principal,
        credential_id="credential-next",
        key_id="key-next",
        key_version=2,
        public_key=b"B" * 32,
        expected_revision=first.registry_revision,
    )
    revoked = admin.transition_lifecycle(
        principal_id=principal,
        lifecycle=credentials.CredentialLifecycle.REVOKED,
        expected_revision=rotated.registry_revision,
    )
    before = runtime.retained_history(principal)
    reconstructed = _providers(registry.setup)
    _, restarted, _, _ = _authority(reconstructed, kind)
    assert restarted.retained_history(principal) == before
    assert restarted.public_key(first.credential_id) == b"A" * 32
    assert restarted.public_key(rotated.credential_id) == b"B" * 32
    assert restarted.historical_generation(first.credential_id, 1) == first
    assert (
        restarted.historical_generation(revoked.credential_id, revoked.lifecycle_generation)
        == revoked
    )


def _mutation_payload(operation: str, principal: str) -> dict[str, object]:
    """Exact admin SQL request, used to test DB transactions without Python hooks."""
    transition = operation == "TRANSITION"
    return {
        "operation": operation,
        "principal_id": principal,
        "credential_id": None if transition else "credential-cutpoint",
        "key_id": None if transition else "key-cutpoint",
        "key_version": None if transition else (1 if operation == "PROVISION" else 2),
        "public_key_hex": None if transition else (b"Z" * 32).hex(),
        "key_material_identity": None if transition else public_key_material_identity(b"Z" * 32),
        "expected_revision": None if operation == "PROVISION" else 1,
        "lifecycle": "REVOKED" if transition else None,
    }


def _physical_snapshot(schema: str) -> tuple[tuple[object, ...], ...]:
    with psycopg.connect(BASE_DSN) as conn:
        return tuple(
            tuple(
                conn.execute(
                    sql.SQL("SELECT * FROM {}.{} ORDER BY 1,2").format(
                        sql.Identifier(schema), sql.Identifier(table)
                    )
                ).fetchall()
            )
            for table in ("credentials", "history", "current_credentials")
        )


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize(
    "invalid_setting", ["READ COMMITTED", "REPEATABLE READ", "sync_off", "forged_material"]
)
def test_direct_admin_sql_rejects_weak_transaction_or_forged_material(
    registry, kind, invalid_setting
) -> None:
    setup, runtime, _, principal = _authority(registry, kind)
    before = _physical_snapshot(setup.schema)
    payload = _mutation_payload("PROVISION", principal)
    if invalid_setting == "forged_material":
        payload["key_material_identity"] = public_key_material_identity(b"F" * 32)
    isolation = (
        invalid_setting
        if invalid_setting in ("READ COMMITTED", "REPEATABLE READ")
        else "SERIALIZABLE"
    )
    with psycopg.connect(_connection(setup.admin_role).dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("BEGIN ISOLATION LEVEL {}").format(sql.SQL(isolation)))
        if invalid_setting == "sync_off":
            conn.execute("SET LOCAL synchronous_commit=off")
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(setup.schema)),
                (Jsonb(payload),),
            )
    assert _physical_snapshot(setup.schema) == before
    assert runtime.retained_history(principal) == ()


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_direct_admin_sql_requires_actual_admin_session_login(registry, kind) -> None:
    setup, runtime, _, principal = _authority(registry, kind)
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
        conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(setup.admin_role)))
        with pytest.raises(InsufficientPrivilege, match="direct provisioning login"):
            conn.execute(
                sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(setup.schema)),
                (Jsonb(_mutation_payload("PROVISION", principal)),),
            )
    assert runtime.retained_history(principal) == ()


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize("field", ["principal_id", "credential_id", "key_id"])
def test_direct_admin_sql_rejects_unicode_whitespace_only_identity(registry, kind, field) -> None:
    setup, runtime, _, principal = _authority(registry, kind)
    payload = _mutation_payload("PROVISION", principal)
    payload[field] = "\u00a0\u2003"
    before = _physical_snapshot(setup.schema)
    with psycopg.connect(_connection(setup.admin_role).dsn, autocommit=True) as conn:
        conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute(
                sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(setup.schema)),
                (Jsonb(payload),),
            )
    assert _physical_snapshot(setup.schema) == before
    assert runtime.retained_history(principal) == ()


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_connection_search_path_is_pinned_before_catalog_qualification(registry, kind) -> None:
    setup, runtime, _, _ = _authority(registry, kind)
    shadow = f"shadow_{uuid.uuid4().hex[:10]}"
    try:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(shadow)))
            conn.execute(
                sql.SQL(
                    "CREATE FUNCTION {}.format_type(oid,integer) RETURNS text LANGUAGE SQL AS "
                    "$$ SELECT 'forged type'::text $$"
                ).format(sql.Identifier(shadow))
            )
            conn.execute(
                sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
                    sql.Identifier(shadow), sql.Identifier(setup.runtime_role)
                )
            )
        connection = PostgreSQLConnectionConfig(
            make_conninfo(
                _connection(setup.runtime_role).dsn, options=f"-c search_path={shadow},pg_catalog"
            )
        )
        with psycopg.connect(connection.dsn) as conn:
            assert conn.execute("SELECT format_type(23::oid,NULL)").fetchone() == ("forged type",)
        rebuilt = type(runtime)(
            connection, schema=setup.schema, environment=ENVIRONMENT, trust_domain=TRUST_DOMAIN
        )
        assert rebuilt.credential_identities() == ()
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(shadow)))


_CUTPOINTS = (
    ("PROVISION", "credentials", "TRUE"),
    ("PROVISION", "history", "TRUE"),
    ("PROVISION", "current_credentials", "TRUE"),
    ("ROTATE", "history", "NEW.lifecycle='VERIFY_ONLY'"),
    ("ROTATE", "credentials", "TRUE"),
    ("ROTATE", "history", "NEW.lifecycle='ACTIVE'"),
    ("ROTATE", "current_credentials", "TRUE"),
    ("TRANSITION", "history", "TRUE"),
    ("TRANSITION", "current_credentials", "TRUE"),
)


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize(("operation", "table", "condition"), _CUTPOINTS)
def test_real_sql_cutpoint_rolls_back_complete_mutation(
    registry, kind, operation, table, condition
) -> None:
    setup, runtime, admin, principal = _authority(registry, kind)
    if operation != "PROVISION":
        _provision(admin, principal)
    before = runtime.retained_history(principal)
    physical_before = _physical_snapshot(setup.schema)
    # The trigger injects a failure *inside* the reviewed PostgreSQL transaction.
    # Normal provider qualification rejects it; only the genuine admin login
    # calls the exact SQL API directly while the failpoint exists.
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        conn.execute(
            sql.SQL(
                "CREATE FUNCTION {}.test_abort() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'injected credential transaction cutpoint'; END $$"
            ).format(sql.Identifier(setup.schema))
        )
        conn.execute(
            sql.SQL(
                "CREATE TRIGGER test_abort AFTER INSERT OR UPDATE ON {}.{} "
                "FOR EACH ROW WHEN ({}) EXECUTE FUNCTION {}.test_abort()"
            ).format(
                sql.Identifier(setup.schema),
                sql.Identifier(table),
                sql.SQL(condition),
                sql.Identifier(setup.schema),
            )
        )
    try:
        with pytest.raises(RegistryQualificationError):
            runtime.retained_history(principal)
        with psycopg.connect(_connection(setup.admin_role).dsn, autocommit=True) as conn:
            conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
            with pytest.raises(psycopg.errors.RaiseException, match="injected credential"):
                conn.execute(
                    sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(setup.schema)),
                    (Jsonb(_mutation_payload(operation, principal)),),
                )
        assert _physical_snapshot(setup.schema) == physical_before
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP TRIGGER test_abort ON {}.{}").format(
                    sql.Identifier(setup.schema), sql.Identifier(table)
                )
            )
            conn.execute(
                sql.SQL("DROP FUNCTION {}.test_abort()").format(sql.Identifier(setup.schema))
            )
    rebuilt = _providers(registry.setup)
    _, recovered, _, _ = _authority(rebuilt, kind)
    assert recovered.retained_history(principal) == before
    assert sum(record.lifecycle is credentials.CredentialLifecycle.ACTIVE for record in before) <= 1


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize("operation", ["PROVISION", "ROTATE", "TRANSITION"])
def test_backend_loss_before_commit_has_no_partial_credential_or_history(
    registry, kind, operation
) -> None:
    setup, runtime, admin, principal = _authority(registry, kind)
    if operation != "PROVISION":
        _provision(admin, principal)
    before = runtime.retained_history(principal)
    physical_before = _physical_snapshot(setup.schema)
    with psycopg.connect(_connection(setup.admin_role).dsn, autocommit=True) as conn:
        conn.execute("BEGIN ISOLATION LEVEL SERIALIZABLE")
        backend = conn.execute("SELECT pg_backend_pid()").fetchone()
        assert backend is not None
        conn.execute(
            sql.SQL("SELECT {}.apply(%s)").format(sql.Identifier(setup.schema)),
            (Jsonb(_mutation_payload(operation, principal)),),
        )
        with psycopg.connect(BASE_DSN, autocommit=True) as bootstrap:
            assert bootstrap.execute(
                "SELECT pg_terminate_backend(%s)", (backend[0],)
            ).fetchone() == (True,)
        with pytest.raises(psycopg.OperationalError):
            conn.execute("COMMIT")
    assert _physical_snapshot(setup.schema) == physical_before
    rebuilt = _providers(registry.setup)
    _, recovered, _, _ = _authority(rebuilt, kind)
    assert recovered.retained_history(principal) == before


@pytest.mark.parametrize("kind", ["requester", "claimant"])
@pytest.mark.parametrize(
    "tamper",
    ["material_label", "public_key", "head_rewind", "history_gap", "active_alias", "trust_domain"],
)
def test_corrupt_retained_evidence_fails_closed_without_repair(registry, kind, tamper) -> None:
    setup, runtime, admin, principal = _authority(registry, kind)
    first = _provision(admin, principal)
    admin.rotate_credential(
        principal_id=principal,
        credential_id="credential-next",
        key_id="key-next",
        key_version=2,
        public_key=b"B" * 32,
        expected_revision=1,
    )
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        if tamper == "material_label":
            conn.execute(
                sql.SQL(
                    "UPDATE {}.credentials SET key_material_identity=%s WHERE credential_id=%s"
                ).format(sql.Identifier(setup.schema)),
                (public_key_material_identity(b"Z" * 32), first.credential_id),
            )
        elif tamper == "public_key":
            conn.execute(
                sql.SQL("UPDATE {}.credentials SET public_key=%s WHERE credential_id=%s").format(
                    sql.Identifier(setup.schema)
                ),
                (b"Z" * 32, first.credential_id),
            )
        elif tamper == "head_rewind":
            conn.execute(
                sql.SQL(
                    "UPDATE {}.current_credentials SET revision=1 WHERE principal_id=%s"
                ).format(sql.Identifier(setup.schema)),
                (principal,),
            )
        elif tamper == "history_gap":
            conn.execute(
                sql.SQL("DELETE FROM {}.history WHERE principal_id=%s AND revision=1").format(
                    sql.Identifier(setup.schema)
                ),
                (principal,),
            )
        elif tamper == "active_alias":
            conn.execute(
                sql.SQL(
                    "UPDATE {}.history SET lifecycle='ACTIVE' WHERE principal_id=%s AND revision=2"
                ).format(sql.Identifier(setup.schema)),
                (principal,),
            )
        else:
            conn.execute(
                sql.SQL(
                    "UPDATE {}.credentials SET trust_domain='td_elsewhere' WHERE credential_id=%s"
                ).format(sql.Identifier(setup.schema)),
                (first.credential_id,),
            )
    corrupt_snapshot = _physical_snapshot(setup.schema)
    for operation in (
        lambda: runtime.retained_history(principal),
        lambda: runtime.public_key(first.credential_id),
        runtime.credential_identities,
    ):
        with pytest.raises(credentials.CredentialResolutionError):
            operation()
    assert _physical_snapshot(setup.schema) == corrupt_snapshot


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_missing_or_inactive_credentials_cannot_authorize_new_flow(registry, kind) -> None:
    _, runtime, admin, principal = _authority(registry, kind)
    resolve = (
        runtime.active_requester_credential if kind == "requester" else runtime.resolve_claimant
    )
    with pytest.raises(credentials.CredentialResolutionError):
        resolve(principal)
    with pytest.raises(credentials.CredentialResolutionError):
        runtime.public_key("unregistered-identity")
    first = _provision(admin, principal)
    verify_only = admin.transition_lifecycle(
        principal_id=principal,
        lifecycle=credentials.CredentialLifecycle.VERIFY_ONLY,
        expected_revision=first.registry_revision,
    )
    with pytest.raises(credentials.CredentialResolutionError):
        resolve(principal)
    revoked = admin.transition_lifecycle(
        principal_id=principal,
        lifecycle=credentials.CredentialLifecycle.REVOKED,
        expected_revision=verify_only.registry_revision,
    )
    with pytest.raises(credentials.CredentialResolutionError):
        resolve(principal)
    assert runtime.public_key(first.credential_id) == b"A" * 32
    assert runtime.historical_generation(first.credential_id, 1) == first
    assert (
        runtime.historical_generation(first.credential_id, revoked.lifecycle_generation) == revoked
    )


@pytest.mark.parametrize("kind", ["requester", "claimant"])
def test_first_provisioning_race_in_separate_processes_uses_database_authority(
    registry, kind
) -> None:
    setup, runtime, _, principal = _authority(registry, kind)
    source = """
import json
import sys
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core import postgresql_preaccount_credentials as credentials
config = json.loads(sys.argv[1])
implementation = (credentials.PostgreSQLRequesterCredentialProvisioningAdminProvider
    if config['kind'] == 'requester' else credentials.PostgreSQLClaimantIdentityProvisioningAdminProvider)
admin = implementation(PostgreSQLConnectionConfig(config['dsn']), schema=config['schema'],
    environment='PRODUCTION', trust_domain=config['trust_domain'])
print('ready', flush=True)
sys.stdin.readline()
try:
    result = admin.provision_credential(principal_id=config['principal'],
        credential_id=config['tag'], key_id='key-' + config['tag'], key_version=1,
        public_key=bytes.fromhex(config['public_key']))
    print(json.dumps({'outcome': 'committed', 'credential_id': result.credential_id}), flush=True)
except credentials.CredentialConflictError:
    print(json.dumps({'outcome': 'conflict'}), flush=True)
"""
    processes = []
    try:
        for tag, key in (("process-left", b"L" * 32), ("process-right", b"R" * 32)):
            configuration = {
                "kind": kind,
                "dsn": _connection(setup.admin_role).dsn,
                "schema": setup.schema,
                "trust_domain": TRUST_DOMAIN,
                "principal": principal,
                "tag": tag,
                "public_key": key.hex(),
            }
            process = subprocess.Popen(
                [sys.executable, "-c", source, json.dumps(configuration)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append(process)
        for process in processes:
            assert process.stdout is not None and process.stdout.readline().strip() == "ready"
        for process in processes:
            assert process.stdin is not None
            process.stdin.write("go\n")
            process.stdin.flush()
        outcomes = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stderr
            outcomes.append(json.loads(stdout))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.communicate(timeout=10)
    assert sorted(item["outcome"] for item in outcomes) == ["committed", "conflict"]
    history = runtime.retained_history(principal)
    assert len(history) == 1 and history[0].lifecycle is credentials.CredentialLifecycle.ACTIVE
    winner = next(item for item in outcomes if item["outcome"] == "committed")
    assert history[0].credential_id == winner["credential_id"]
