"""Real PostgreSQL qualification of the independent closed production schema.

All role names, public credential facts and databases are disposable test
fixtures. No production key, enrollment, Root Proof or activation is created.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import replace
from types import SimpleNamespace

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo

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
