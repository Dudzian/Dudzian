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
        pin, physical_identity=replace(pin.physical_identity, database_oid=pin.physical_identity.database_oid + 1)
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
