"""S9-RPI-11: real PostgreSQL 16 tests for non-authorizing CCA catalog inspection."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import psycopg
import pytest
from psycopg import sql

from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_independent_cca_reader_inspection import (
    IndependentCCAReaderCatalogError,
    IndependentCCAReaderCatalogPin,
    inspect_independent_cca_reader_catalog,
)
from tests.security.test_postgresql_independent_cca_reader_smoke import (
    isolated_cca_read,
)

pytestmark = pytest.mark.external_postgresql


def _synthetic_test_pin(value) -> IndependentCCAReaderCatalogPin:
    """Admin-created pin exists ONLY inside a disposable fixture, not production."""
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        identities = dict(
            bootstrap.execute(
                "SELECT rolname,oid FROM pg_catalog.pg_roles WHERE rolname=ANY(%s)",
                ([value.owner, value.reader, value.source],),
            ).fetchall()
        )
        function = bootstrap.execute(
            "SELECT p.oid,pg_catalog.pg_get_functiondef(p.oid) "
            "FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=%s AND p.proname='read_exact_close'",
            (value.schema,),
        ).fetchone()
        assert function is not None
        physical = read_atomic_root_proof_physical_identity(bootstrap, value.database)
    return IndependentCCAReaderCatalogPin(
        schema=value.schema,
        owner_role=value.owner,
        reader_role=value.reader,
        source_role=value.source,
        owner_oid=identities[value.owner],
        reader_oid=identities[value.reader],
        source_oid=identities[value.source],
        function_oid=function[0],
        function_definition_sha256=hashlib.sha256(function[1].encode("utf-8")).hexdigest(),
        physical_identity=physical,
    )


def _inspect(value, pin, *, role="reader", isolation="READ COMMITTED"):
    with psycopg.connect(value.login(role), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL " + isolation)
            return inspect_independent_cca_reader_catalog(conn, pin)


def test_synthetic_reader_catalog_is_read_only_inspectable_but_not_qualified(
    isolated_cca_read,
):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    result = _inspect(value, pin)
    assert result.function_oid == pin.function_oid
    assert result.catalog_sha256 == pin.function_definition_sha256
    assert result.status == "NOT_QUALIFIED"
    assert result.production_authorization == "BLOCKED"
    assert result.reader_installed is False


@pytest.mark.parametrize("role", ["source", "stranger"])
def test_source_or_stranger_cannot_substitute_for_direct_cca_login(isolated_cca_read, role):
    pin = _synthetic_test_pin(isolated_cca_read)
    with pytest.raises(IndependentCCAReaderCatalogError, match="direct-login"):
        _inspect(isolated_cca_read, pin, role=role)


def test_reader_catalog_inspection_rejects_stale_physical_identity(isolated_cca_read):
    pin = _synthetic_test_pin(isolated_cca_read)
    stale = replace(
        pin,
        physical_identity=replace(
            pin.physical_identity, database_oid=pin.physical_identity.database_oid + 1
        ),
    )
    with pytest.raises(IndependentCCAReaderCatalogError, match="physical"):
        _inspect(isolated_cca_read, stale)


def test_reader_catalog_inspection_rejects_role_oid_drift(isolated_cca_read):
    pin = _synthetic_test_pin(isolated_cca_read)
    with pytest.raises(IndependentCCAReaderCatalogError, match="OID"):
        _inspect(isolated_cca_read, replace(pin, reader_oid=pin.reader_oid + 100000))


def test_reader_catalog_inspection_rejects_sql_body_pin_drift(isolated_cca_read):
    pin = _synthetic_test_pin(isolated_cca_read)
    with pytest.raises(IndependentCCAReaderCatalogError, match="body"):
        _inspect(
            isolated_cca_read,
            replace(pin, function_definition_sha256="f" * 64),
        )


def test_reader_catalog_inspection_requires_read_committed_write_transaction(isolated_cca_read):
    pin = _synthetic_test_pin(isolated_cca_read)
    with pytest.raises(IndependentCCAReaderCatalogError, match="READ COMMITTED"):
        _inspect(isolated_cca_read, pin, isolation="REPEATABLE READ")
    with psycopg.connect(isolated_cca_read.login("reader"), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            with pytest.raises(IndependentCCAReaderCatalogError, match="READ WRITE"):
                inspect_independent_cca_reader_catalog(conn, pin)


def test_reader_catalog_inspection_rejects_unreviewed_public_grants(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.read_exact_close(text) TO PUBLIC").format(
                    sql.Identifier(value.schema)
                )
            )
            with pytest.raises(IndependentCCAReaderCatalogError, match="EXECUTE"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE EXECUTE ON FUNCTION {}.read_exact_close(text) FROM PUBLIC").format(
                    sql.Identifier(value.schema)
                )
            )


def test_reader_catalog_inspection_rejects_direct_cca_table_select(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("GRANT SELECT ON {}.operations TO {}").format(
                    sql.Identifier(value.schema), sql.Identifier(value.reader)
                )
            )
            with pytest.raises(IndependentCCAReaderCatalogError, match="direct relation"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE SELECT ON {}.operations FROM {}").format(
                    sql.Identifier(value.schema), sql.Identifier(value.reader)
                )
            )


@pytest.mark.parametrize("privilege", ["SELECT", "INSERT", "UPDATE", "REFERENCES"])
@pytest.mark.parametrize("grantee", ["reader", "PUBLIC"])
def test_reader_catalog_inspection_rejects_column_only_privileges(
    isolated_cca_read, privilege, grantee
):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    recipient = sql.SQL("PUBLIC") if grantee == "PUBLIC" else sql.Identifier(value.reader)
    relation = value.schema + ".operations"
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("GRANT {} (operation_id) ON {}.operations TO {}").format(
                    sql.SQL(privilege), sql.Identifier(value.schema), recipient
                )
            )
            assert bootstrap.execute(
                "SELECT pg_catalog.has_table_privilege("
                "%s,%s,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'),"
                "pg_catalog.has_any_column_privilege(%s,%s,%s)",
                (value.reader, relation, value.reader, relation, privilege),
            ).fetchone() == (False, True)
            with pytest.raises(IndependentCCAReaderCatalogError, match="direct column privileges"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE {} (operation_id) ON {}.operations FROM {}").format(
                    sql.SQL(privilege), sql.Identifier(value.schema), recipient
                )
            )


@pytest.mark.parametrize("privilege", ["SELECT", "UPDATE", "USAGE"])
@pytest.mark.parametrize("grantee", ["reader", "PUBLIC"])
def test_reader_catalog_inspection_rejects_sequence_privileges(
    isolated_cca_read, privilege, grantee
):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    recipient = sql.SQL("PUBLIC") if grantee == "PUBLIC" else sql.Identifier(value.reader)
    sequence = value.schema + ".inspection_sequence"
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        bootstrap.execute(
            sql.SQL("CREATE SEQUENCE {}.inspection_sequence").format(sql.Identifier(value.schema))
        )
        bootstrap.execute(
            sql.SQL("ALTER SEQUENCE {}.inspection_sequence OWNER TO {}").format(
                sql.Identifier(value.schema), sql.Identifier(value.owner)
            )
        )
        assert _inspect(value, pin).status == "NOT_QUALIFIED"
        try:
            bootstrap.execute(
                sql.SQL("GRANT {} ON SEQUENCE {}.inspection_sequence TO {}").format(
                    sql.SQL(privilege), sql.Identifier(value.schema), recipient
                )
            )
            assert bootstrap.execute(
                "SELECT pg_catalog.has_table_privilege("
                "%s,%s,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER'),"
                "pg_catalog.has_sequence_privilege(%s,%s,%s)",
                (value.reader, sequence, value.reader, sequence, privilege),
            ).fetchone() == (privilege != "USAGE", True)
            assert (
                bootstrap.execute(
                    "SELECT 1 FROM pg_catalog.pg_class c "
                    "JOIN pg_catalog.pg_namespace n ON n.oid=c.relnamespace "
                    "WHERE n.nspname=%s AND c.relkind IN ('r','p','v','m','f') "
                    "AND pg_catalog.has_table_privilege("
                    "%s,c.oid,'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')",
                    (value.schema, value.reader),
                ).fetchone()
                is None
            )
            expected = "direct sequence privileges" if privilege == "USAGE" else "direct relation"
            with pytest.raises(IndependentCCAReaderCatalogError, match=expected):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE {} ON SEQUENCE {}.inspection_sequence FROM {}").format(
                    sql.SQL(privilege), sql.Identifier(value.schema), recipient
                )
            )


def test_reader_catalog_inspection_rejects_unreviewed_definer_search_path(
    isolated_cca_read,
):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("ALTER FUNCTION {}.read_exact_close(text) SET search_path = public").format(
                    sql.Identifier(value.schema)
                )
            )
            with pytest.raises(IndependentCCAReaderCatalogError, match="search_path"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL(
                    "ALTER FUNCTION {}.read_exact_close(text) "
                    "SET search_path = pg_catalog, {}, pg_temp"
                ).format(sql.Identifier(value.schema), sql.Identifier(value.schema))
            )


def test_reader_catalog_inspection_rejects_untrusted_schema_create(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("GRANT CREATE ON SCHEMA {} TO {}").format(
                    sql.Identifier(value.schema), sql.Identifier(value.source)
                )
            )
            with pytest.raises(IndependentCCAReaderCatalogError, match="untrusted role"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE CREATE ON SCHEMA {} FROM {}").format(
                    sql.Identifier(value.schema), sql.Identifier(value.source)
                )
            )


def test_reader_catalog_inspection_rejects_new_role_membership(isolated_cca_read):
    value = isolated_cca_read
    pin = _synthetic_test_pin(value)
    with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
        try:
            bootstrap.execute(
                sql.SQL("GRANT {} TO {}").format(
                    sql.Identifier(value.reader), sql.Identifier(value.source)
                )
            )
            with pytest.raises(IndependentCCAReaderCatalogError, match="membership"):
                _inspect(value, pin)
        finally:
            bootstrap.execute(
                sql.SQL("REVOKE {} FROM {}").format(
                    sql.Identifier(value.reader), sql.Identifier(value.source)
                )
            )
