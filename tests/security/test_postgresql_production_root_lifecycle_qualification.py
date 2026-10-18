"""Real PostgreSQL qualification of the independent closed production schema.

All role names, public credential facts and databases are disposable test
fixtures. No production key, enrollment, Root Proof or activation is created.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.types.json import Jsonb

from bot_core import postgresql_production_root_lifecycle_qualification as qualification
from bot_core import postgresql_production_root_lifecycle_schema as lifecycle

from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_production_root_lifecycle_qualification import (
    PostgreSQLProductionRootLifecyclePin,
    ProductionRootLifecycleQualificationError,
    pin_production_root_lifecycle_catalog,
    qualify_production_root_lifecycle_authority,
    qualify_production_root_lifecycle_resolution,
)
from bot_core.postgresql_production_root_lifecycle_schema import (
    PostgreSQLProductionRootLifecycleProvisioning,
    install_production_root_lifecycle_authority,
)

pytestmark = pytest.mark.external_postgresql
_BASE_DSN = os.environ.get(
    "CRYPTOHUNTER_TEST_POSTGRES_DSN",
    os.environ.get(
        "DUDZIAN_TEST_POSTGRES_DSN",
        "host=127.0.0.1 port=55432 dbname=postgres user=postgres connect_timeout=5",
    ),
)


@pytest.fixture
def production_qualification_boundary():
    suffix = uuid.uuid4().hex[:12]
    database = "plq_db_" + suffix
    config = PostgreSQLProductionRootLifecycleProvisioning(
        "plq_" + suffix,
        "plqo_" + suffix,
        "plqi_" + suffix,
        "plqa_" + suffix,
        "isolated-production-qualification-" + suffix,
        "isolated-production-qualification-trust-" + suffix,
    )
    dsn = make_conninfo(_BASE_DSN, dbname=database)
    roles = (config.schema_owner_role, config.issuer_role, config.administrator_role)
    with psycopg.connect(_BASE_DSN, autocommit=True) as bootstrap:
        bootstrap.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        with psycopg.connect(dsn, autocommit=True) as bootstrap:
            bootstrap.execute(
                sql.SQL("REVOKE CREATE,TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(database)
                )
            )
            bindings = tuple(
                {
                    "semantic_role": role,
                    "authority_identity": "isolated-authority-" + role,
                    "identity": "isolated-credential-" + role,
                    "public_material_identity": "sha256:"
                    + hashlib.sha256(role.encode()).hexdigest(),
                    "key_version": 1,
                    "lifecycle_generation": 1,
                }
                for role in (
                    "ROOT_PROOF_SIGNING",
                    "HISTORY_ATTESTATION_SIGNING",
                    "TRUST_ACTIVATION",
                )
            )
            identity = read_atomic_root_proof_physical_identity(bootstrap, config.installation_id)
            with bootstrap.transaction():
                bootstrap.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                install_production_root_lifecycle_authority(
                    bootstrap, config, bindings, bootstrap_operation_id="isolated-bootstrap"
                )
                pin = pin_production_root_lifecycle_catalog(
                    bootstrap, config, expected_identity=identity
                )
        yield SimpleNamespace(config=config, dsn=dsn, pin=pin, database=database)
    finally:
        with psycopg.connect(_BASE_DSN, autocommit=True) as bootstrap:
            bootstrap.execute(
                sql.SQL("DROP DATABASE {} WITH (FORCE)").format(sql.Identifier(database))
            )
            for role in reversed(roles):
                bootstrap.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def _qualify(value, *, semantic_role="issuer", pin=None, login=None, read_only=False):
    principal = login or (
        value.config.issuer_role if semantic_role == "issuer" else value.config.administrator_role
    )
    with psycopg.connect(make_conninfo(value.dsn, user=principal), autocommit=True) as conn:
        with conn.transaction():
            conn.execute(
                "SET TRANSACTION ISOLATION LEVEL SERIALIZABLE" + (" READ ONLY" if read_only else "")
            )
            qualify_production_root_lifecycle_authority(
                conn, pin or value.pin, expected_role=semantic_role
            )


@pytest.mark.parametrize("role", ["issuer", "administrator"])
def test_exact_direct_roles_qualify_in_read_only_transaction(
    production_qualification_boundary, role
):
    _qualify(production_qualification_boundary, semantic_role=role, read_only=True)


def test_runtime_cannot_self_pin(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(make_conninfo(value.dsn, user=value.config.issuer_role)) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        with pytest.raises(ProductionRootLifecycleQualificationError, match="bootstrap"):
            pin_production_root_lifecycle_catalog(
                conn, value.config, expected_identity=value.pin.identity
            )


@pytest.mark.parametrize("login_role", ["administrator", "issuer"])
def test_resolution_requires_direct_administrator_read_committed(
    production_qualification_boundary, login_role
):
    value = production_qualification_boundary
    principal = (
        value.config.administrator_role
        if login_role == "administrator"
        else value.config.issuer_role
    )
    with psycopg.connect(make_conninfo(value.dsn, user=principal), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            if login_role == "administrator":
                qualify_production_root_lifecycle_resolution(conn, value.pin)
            else:
                with pytest.raises(ProductionRootLifecycleQualificationError, match="direct-login"):
                    qualify_production_root_lifecycle_resolution(conn, value.pin)
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            with pytest.raises(ProductionRootLifecycleQualificationError, match="READ COMMITTED"):
                qualify_production_root_lifecycle_resolution(conn, value.pin)


def test_test_only_and_wrong_direct_roles_are_rejected(production_qualification_boundary):
    value = production_qualification_boundary
    with pytest.raises(ProductionRootLifecycleQualificationError, match="direct-login"):
        _qualify(value, login=value.config.administrator_role)
    with psycopg.connect(value.dsn, autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(value.config.issuer_role)))
            with pytest.raises(ProductionRootLifecycleQualificationError, match="direct-login"):
                qualify_production_root_lifecycle_authority(conn, value.pin)
    with pytest.raises(ProductionRootLifecycleQualificationError, match="installation pin"):
        qualify_production_root_lifecycle_authority(None, object())


@pytest.mark.parametrize(
    "field,value",
    [
        ("system_identifier", 1),
        ("database_oid", 1),
        ("database_name", "another-database"),
        ("connection_host", "another-endpoint"),
        ("connection_port", 1),
        ("server_address", "192.0.2.1"),
        ("server_port", 1),
    ],
)
def test_physical_database_and_endpoint_are_pinned(production_qualification_boundary, field, value):
    boundary = production_qualification_boundary
    identity = replace(boundary.pin.identity, **{field: value})
    with pytest.raises(ProductionRootLifecycleQualificationError, match="physical"):
        _qualify(boundary, pin=replace(boundary.pin, identity=identity))


def test_runtime_does_not_derive_its_expected_digest(production_qualification_boundary):
    value = production_qualification_boundary
    with pytest.raises(ProductionRootLifecycleQualificationError, match="catalog"):
        _qualify(value, pin=replace(value.pin, catalog_sha256="0" * 64))


@pytest.mark.parametrize(
    "alteration",
    [
        "ALTER ROLE {issuer} INHERIT",
        "ALTER ROLE {owner} LOGIN",
        "GRANT {owner} TO {issuer}",
        "GRANT UPDATE ON {schema}.fences TO {issuer}",
        "GRANT UPDATE(generation) ON {schema}.fences TO {issuer}",
        "GRANT SELECT(generation) ON {schema}.fences TO {issuer}",
        "GRANT EXECUTE ON FUNCTION {schema}.begin_close(jsonb) TO {issuer}",
        "GRANT EXECUTE ON FUNCTION {schema}.begin_close(jsonb) TO PUBLIC",
        "ALTER FUNCTION {schema}.assert_catalog() SECURITY INVOKER",
        "ALTER FUNCTION {schema}.assert_catalog() SET search_path=public",
        "ALTER ROLE {issuer} SET synchronous_commit=off",
        "ALTER TABLE {schema}.fences DISABLE TRIGGER lifecycle_update",
        "CREATE VIEW public.alternate_lifecycle AS SELECT * FROM {schema}.fences; GRANT UPDATE ON public.alternate_lifecycle TO {issuer}",
        "CREATE TABLE public.alternate_ownership(value integer); ALTER TABLE public.alternate_ownership OWNER TO {issuer}",
    ],
)
def test_privilege_source_and_alternate_write_paths_fail_closed(
    production_qualification_boundary, alteration
):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            sql.SQL(alteration).format(
                schema=sql.Identifier(value.config.schema),
                issuer=sql.Identifier(value.config.issuer_role),
                owner=sql.Identifier(value.config.schema_owner_role),
            )
        )
    with pytest.raises(ProductionRootLifecycleQualificationError):
        _qualify(value)


def test_modified_reviewed_function_source_is_rejected(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            sql.SQL(
                "CREATE OR REPLACE FUNCTION {}.assert_catalog() RETURNS void "
                "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS 'BEGIN RETURN; END'"
            ).format(sql.Identifier(value.config.schema))
        )
    with pytest.raises(ProductionRootLifecycleQualificationError, match="source"):
        _qualify(value)


def test_ordinary_maintenance_does_not_change_the_pin(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            sql.SQL("VACUUM ANALYZE {}.fences").format(sql.Identifier(value.config.schema))
        )
    _qualify(value)


def test_public_external_security_definer_is_refused(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            "CREATE FUNCTION public.alternate_lifecycle_authority() RETURNS void "
            "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS 'BEGIN RETURN; END'"
        )
    with pytest.raises(ProductionRootLifecycleQualificationError, match="alternate"):
        _qualify(value)


def test_builtin_settings_view_exception_is_pinned(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute("REVOKE UPDATE ON pg_catalog.pg_settings FROM PUBLIC")
    with pytest.raises(ProductionRootLifecycleQualificationError, match="catalog"):
        _qualify(value)


def test_public_temporary_privilege_is_refused(production_qualification_boundary):
    value = production_qualification_boundary
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            sql.SQL("GRANT TEMPORARY ON DATABASE {} TO PUBLIC").format(
                sql.Identifier(value.database)
            )
        )
    with pytest.raises(ProductionRootLifecycleQualificationError, match="alternate"):
        _qualify(value)


def test_serializable_transaction_and_exact_pin_types_are_required(
    production_qualification_boundary,
):
    value = production_qualification_boundary
    with psycopg.connect(
        make_conninfo(value.dsn, user=value.config.issuer_role), autocommit=True
    ) as conn:
        with pytest.raises(ProductionRootLifecycleQualificationError, match="SERIALIZABLE"):
            qualify_production_root_lifecycle_authority(conn, value.pin)
        with conn.transaction():
            with pytest.raises(ProductionRootLifecycleQualificationError, match="SERIALIZABLE"):
                qualify_production_root_lifecycle_authority(conn, value.pin)
    with pytest.raises(ValueError, match="distinct"):
        replace(value.pin, role_oids=tuple((name, 1) for name, _ in value.pin.role_oids))
    with pytest.raises(TypeError, match="production provisioning"):
        PostgreSQLProductionRootLifecyclePin(
            object(), value.pin.identity, "0" * 64, value.pin.role_oids
        )


# Completion publication stays blocked in this production catalog. The V1
# simulation below belongs exclusively to this disposable test database.
ROOT = "ROOT_PROOF_SIGNING"
IDENTITY = "isolated-credential-" + ROOT
REQUEST = {
    "operation_id": "isolated-close-1",
    "semantic_role": ROOT,
    "identity": IDENTITY,
    "generation": 1,
    "predecessor_state": "ACTIVE",
    "target_state": "VERIFY_ONLY",
}


@pytest.fixture
def boundary(production_qualification_boundary):
    return production_qualification_boundary


def _call(conn, value, function, arguments):
    return conn.execute(
        sql.SQL("SELECT {}.{}({})").format(
            sql.Identifier(value.config.schema),
            sql.Identifier(function),
            sql.SQL(",").join(sql.Placeholder() for _ in arguments),
        ),
        arguments,
    ).fetchone()[0]


def _refresh_manifest(conn, value):
    schema = sql.Identifier(value.config.schema)
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest DISABLE TRIGGER immutable_row").format(schema)
    )
    conn.execute(sql.SQL("DELETE FROM {}.catalog_manifest").format(schema))
    conn.execute(
        sql.SQL("ALTER TABLE {}.catalog_manifest ENABLE ALWAYS TRIGGER immutable_row").format(
            schema
        )
    )
    conn.execute(
        sql.SQL("INSERT INTO {}.catalog_manifest SELECT true,{}.catalog_snapshot()").format(
            schema, schema
        )
    )


def _seed_active_test_row(value):
    # Production has no activation port. Inject only disposable public test facts.
    schema = sql.Identifier(value.config.schema)
    with psycopg.connect(value.dsn) as conn:
        conn.execute(
            sql.SQL("ALTER TABLE {}.fences DISABLE TRIGGER lifecycle_update").format(schema)
        )
        conn.execute(
            sql.SQL(
                "UPDATE {}.fences SET state='ACTIVE',pending_operation_id=NULL WHERE semantic_role=%s"
            ).format(schema),
            (ROOT,),
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.fences ENABLE ALWAYS TRIGGER lifecycle_update").format(schema)
        )


def _begin(value):
    with psycopg.connect(make_conninfo(value.dsn, user=value.config.administrator_role)) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        return _call(conn, value, "begin_close", (Jsonb(REQUEST),))


def _receipt(pending):
    return json.dumps(
        {
            "operation_id": REQUEST["operation_id"],
            "request_sha256": pending["request_sha256"],
            "semantic_role": ROOT,
            "identity": IDENTITY,
            "public_material_identity": "sha256:" + hashlib.sha256(ROOT.encode()).hexdigest(),
            "key_version": 1,
            "generation": 2,
            "state": "VERIFY_ONLY",
            "custody_receipt_sha256": hashlib.sha256(b"missing-original-evidence").hexdigest(),
        },
        indent=2,
    ).encode()


def _qualify_completion_pin(value, pin):
    with psycopg.connect(make_conninfo(value.dsn, user=value.config.administrator_role)) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        qualification.qualify_production_root_lifecycle_authority(
            conn, pin, expected_role="administrator"
        )


@pytest.mark.parametrize("principal", ["administrator", "issuer", "bootstrap"])
@pytest.mark.parametrize("payload", ["forged_receipt", "evidence_hash_only"])
def test_direct_sql_cannot_publish_receipt_or_hash(boundary, principal, payload):
    _seed_active_test_row(boundary)
    pending = _begin(boundary)
    receipt = _receipt(pending)
    evidence = receipt if payload == "forged_receipt" else hashlib.sha256(receipt).digest()
    login = {
        "administrator": boundary.config.administrator_role,
        "issuer": boundary.config.issuer_role,
        "bootstrap": None,
    }[principal]
    dsn = make_conninfo(boundary.dsn, user=login) if login else boundary.dsn
    with psycopg.connect(dsn) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege) as rejected:
            _call(
                conn,
                boundary,
                "complete_close",
                (REQUEST["operation_id"], pending["request_sha256"], evidence),
            )
        if principal == "bootstrap":
            assert "INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED" in str(rejected.value)
    with psycopg.connect(boundary.dsn) as conn:
        assert conn.execute(
            sql.SQL("SELECT count(*) FROM {}.completions").format(
                sql.Identifier(boundary.config.schema)
            )
        ).fetchone() == (0,)
        assert conn.execute(
            sql.SQL(
                "SELECT state,generation,pending_operation_id FROM {}.fences WHERE semantic_role=%s"
            ).format(sql.Identifier(boundary.config.schema)),
            (ROOT,),
        ).fetchone() == ("PENDING_CLOSED", 1, REQUEST["operation_id"])


def test_forced_execute_grant_still_reaches_only_denial_tombstone(boundary):
    with psycopg.connect(boundary.dsn) as conn:
        conn.execute(
            sql.SQL(
                "GRANT EXECUTE ON FUNCTION {}.complete_close(text,text,bytea) TO PUBLIC,{}"
            ).format(
                sql.Identifier(boundary.config.schema),
                sql.Identifier(boundary.config.administrator_role),
            )
        )
        # Even a refreshed internal manifest must not turn an ACL drift into qualification.
        _refresh_manifest(conn, boundary)
    with pytest.raises(qualification.ProductionRootLifecycleQualificationError, match="ACL"):
        _qualify_completion_pin(boundary, boundary.pin)
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role)
    ) as conn:
        with pytest.raises(
            psycopg.errors.InsufficientPrivilege,
            match="INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED",
        ):
            _call(conn, boundary, "complete_close", ("nonexistent-operation", "0" * 64, b"{}"))


@pytest.mark.parametrize("drift", ["SECURITY INVOKER", "SET search_path=public"])
def test_completion_catalog_drift_blocks_runtime_and_direct_administration(boundary, drift):
    with psycopg.connect(boundary.dsn) as conn:
        conn.execute(
            sql.SQL("ALTER FUNCTION {}.complete_close(text,text,bytea) " + drift).format(
                sql.Identifier(boundary.config.schema)
            )
        )
    with pytest.raises(qualification.ProductionRootLifecycleQualificationError, match="source"):
        _qualify_completion_pin(boundary, boundary.pin)
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role)
    ) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="catalog drift"):
            _call(conn, boundary, "begin_close", (Jsonb(REQUEST),))


def _legacy_installation(value):
    """Build the reviewed V1 catalog only in the disposable test installation."""
    schema = sql.Identifier(value.config.schema)
    sources = lifecycle._reviewed_v1_function_sources(value.config)
    with psycopg.connect(value.dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        for signature in (
            "assert_catalog()",
            "begin_close(jsonb)",
            "read_operation(text,text,text)",
            "complete_close(text,text,bytea)",
        ):
            declaration, returns = lifecycle.PRODUCTION_LIFECYCLE_FUNCTION_DECLARATIONS[signature]
            conn.execute(
                sql.SQL(
                    "CREATE OR REPLACE FUNCTION {}.{}({}) RETURNS {} LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS {}"
                ).format(
                    schema,
                    sql.Identifier(signature.split("(")[0]),
                    sql.SQL(declaration),
                    sql.SQL(returns),
                    sql.Literal(sources[signature]),
                )
            )
        conn.execute(
            sql.SQL("GRANT EXECUTE ON FUNCTION {}.complete_close(text,text,bytea) TO {}").format(
                schema, sql.Identifier(value.config.administrator_role)
            )
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.metadata DISABLE TRIGGER immutable_row").format(schema)
        )
        conn.execute(sql.SQL("UPDATE {}.metadata SET schema_version=1").format(schema))
        conn.execute(
            sql.SQL("ALTER TABLE {}.metadata ENABLE ALWAYS TRIGGER immutable_row").format(schema)
        )
        _refresh_manifest(conn, value)
        qualification._check_reviewed_schema(
            conn, value.config, value.pin.role_oids, legacy_v1=True
        )
        # This emulates an externally retained V1 installation pin established
        # by the distinct bootstrap principal, never by a runtime connection.
        return replace(
            value.pin,
            catalog_sha256=qualification._catalog_fingerprint(
                conn, value.config, value.pin.role_oids
            ),
        )


@contextmanager
def _offline_migration_connection(value, *, login=None):
    """External maintenance ceremony for this disposable test database only."""
    dsn = make_conninfo(value.dsn, user=login) if login else value.dsn
    with psycopg.connect(dsn, autocommit=True) as conn:
        with psycopg.connect(_BASE_DSN, autocommit=True) as control:
            control.execute(
                sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS=false").format(
                    sql.Identifier(value.database)
                )
            )
        try:
            yield conn
        finally:
            with psycopg.connect(_BASE_DSN, autocommit=True) as control:
                control.execute(
                    sql.SQL("ALTER DATABASE {} ALLOW_CONNECTIONS=true").format(
                        sql.Identifier(value.database)
                    )
                )


def test_controlled_migration_keeps_exact_legacy_evidence_and_requires_new_pin(boundary):
    legacy_pin = _legacy_installation(boundary)
    _seed_active_test_row(boundary)
    pending = _begin(boundary)
    receipt = _receipt(pending)
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role)
    ) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        assert (
            _call(
                conn,
                boundary,
                "complete_close",
                (REQUEST["operation_id"], pending["request_sha256"], receipt),
            )["phase"]
            == "COMPLETED"
        )
    with _offline_migration_connection(boundary) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            assert (
                qualification.migrate_production_root_lifecycle_completion_block_v2(
                    conn, legacy_pin
                )
                is None
            )
    with pytest.raises(qualification.ProductionRootLifecycleQualificationError, match="catalog"):
        _qualify_completion_pin(boundary, legacy_pin)
    with psycopg.connect(boundary.dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        current_pin = qualification.pin_production_root_lifecycle_catalog(
            conn, boundary.config, expected_identity=legacy_pin.identity
        )
        assert current_pin.catalog_sha256 != legacy_pin.catalog_sha256
        assert conn.execute(
            sql.SQL(
                "SELECT receipt_bytes,receipt_sha256 FROM {}.completions WHERE operation_id=%s"
            ).format(sql.Identifier(boundary.config.schema)),
            (REQUEST["operation_id"],),
        ).fetchone() == (receipt, hashlib.sha256(receipt).digest())
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="immutable"):
            conn.execute(
                sql.SQL("UPDATE {}.completions SET receipt_bytes=%s").format(
                    sql.Identifier(boundary.config.schema)
                ),
                (b"replacement",),
            )
    _qualify_completion_pin(boundary, current_pin)
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role)
    ) as conn:
        assert (
            _call(
                conn,
                boundary,
                "read_operation",
                (ROOT, REQUEST["operation_id"], pending["request_sha256"]),
            )["phase"]
            == "LEGACY_UNVERIFIED_COMPLETION"
        )
    assert _begin(boundary)["phase"] == "LEGACY_UNVERIFIED_COMPLETION"
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role)
    ) as conn:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            _call(
                conn,
                boundary,
                "complete_close",
                (REQUEST["operation_id"], pending["request_sha256"], receipt),
            )


@pytest.mark.parametrize(
    "invalid",
    ["untrusted_digest", "wrong_physical_instance", "direct_runtime_login", "catalog_drift"],
)
def test_migration_requires_retained_pin_exact_catalog_and_bootstrap(boundary, invalid):
    legacy_pin = _legacy_installation(boundary)
    if invalid == "untrusted_digest":
        legacy_pin = replace(legacy_pin, catalog_sha256="0" * 64)
    elif invalid == "wrong_physical_instance":
        legacy_pin = replace(legacy_pin, identity=replace(legacy_pin.identity, system_identifier=1))
    elif invalid == "catalog_drift":
        with psycopg.connect(boundary.dsn) as conn:
            conn.execute(
                sql.SQL(
                    "ALTER FUNCTION {}.complete_close(text,text,bytea) SECURITY INVOKER"
                ).format(sql.Identifier(boundary.config.schema))
            )
    login = boundary.config.administrator_role if invalid == "direct_runtime_login" else None
    with _offline_migration_connection(boundary, login=login) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            with pytest.raises(qualification.ProductionRootLifecycleQualificationError):
                qualification.migrate_production_root_lifecycle_completion_block_v2(
                    conn, legacy_pin
                )
    with psycopg.connect(boundary.dsn) as conn:
        assert conn.execute(
            sql.SQL("SELECT schema_version FROM {}.metadata").format(
                sql.Identifier(boundary.config.schema)
            )
        ).fetchone() == (1,)


def test_uncommitted_migration_rolls_back_catalog_and_legacy_grants(boundary):
    legacy_pin = _legacy_installation(boundary)
    with _offline_migration_connection(boundary) as conn:
        with pytest.raises(RuntimeError, match="simulated crash before COMMIT"):
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                qualification.migrate_production_root_lifecycle_completion_block_v2(
                    conn, legacy_pin
                )
                raise RuntimeError("simulated crash before COMMIT")
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            qualification._check_reviewed_schema(
                conn, boundary.config, legacy_pin.role_oids, legacy_v1=True
            )
            assert (
                qualification._catalog_fingerprint(
                    conn, boundary.config, legacy_pin.role_oids, migration_offline=True
                )
                == legacy_pin.catalog_sha256
            )


def test_migration_requires_closed_connection_gate(boundary):
    legacy_pin = _legacy_installation(boundary)
    with psycopg.connect(boundary.dsn) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
        with pytest.raises(
            qualification.ProductionRootLifecycleQualificationError,
            match="closed database connection gate",
        ):
            qualification.migrate_production_root_lifecycle_completion_block_v2(conn, legacy_pin)


@pytest.mark.parametrize("subtransaction", [False, True])
def test_migration_rejects_gate_closed_without_separate_commit(boundary, subtransaction):
    legacy_pin = _legacy_installation(boundary)
    with psycopg.connect(boundary.dsn, autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
            # ALTER DATABASE correctly refuses to close its own database. A
            # privileged direct catalog edit must not bypass the commit gate.
            if subtransaction:
                with conn.transaction():
                    conn.execute(
                        "UPDATE pg_catalog.pg_database SET datallowconn=false "
                        "WHERE datname=pg_catalog.current_database()"
                    )
            else:
                conn.execute(
                    "UPDATE pg_catalog.pg_database SET datallowconn=false "
                    "WHERE datname=pg_catalog.current_database()"
                )
            with pytest.raises(
                qualification.ProductionRootLifecycleQualificationError,
                match="must commit before migration",
            ):
                qualification.migrate_production_root_lifecycle_completion_block_v2(
                    conn, legacy_pin
                )
            conn.execute(
                "UPDATE pg_catalog.pg_database SET datallowconn=true "
                "WHERE datname=pg_catalog.current_database()"
            )


def test_migration_rejects_remaining_old_runtime_connection(boundary):
    legacy_pin = _legacy_installation(boundary)
    with psycopg.connect(
        make_conninfo(boundary.dsn, user=boundary.config.administrator_role), autocommit=True
    ) as runtime:
        runtime.execute("SELECT 1")
        with _offline_migration_connection(boundary) as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                with pytest.raises(
                    qualification.ProductionRootLifecycleQualificationError,
                    match="must be drained",
                ):
                    qualification.migrate_production_root_lifecycle_completion_block_v2(
                        conn, legacy_pin
                    )
