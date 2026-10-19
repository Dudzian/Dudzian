"""S9-RPI-10: disposable PostgreSQL 16 CCA read/lock/ACL experiment.

This is NOT the installed CCA reader: no production role, migration, runtime
capability, completion, custody mutation, signer or enrollment is provided.
The superuser may populate only disposable fixture rows in a fresh database.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass

import psycopg
import pytest
from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import InsufficientPrivilege, LockNotAvailable
from psycopg.types.json import Jsonb

pytestmark = pytest.mark.external_postgresql
_BASE_DSN = os.environ.get(
    "CRYPTOHUNTER_TEST_POSTGRES_DSN",
    os.environ.get(
        "DUDZIAN_TEST_POSTGRES_DSN",
        "host=127.0.0.1 port=55432 dbname=postgres user=postgres connect_timeout=5",
    ),
)


@dataclass(frozen=True, slots=True)
class _IsolatedReader:
    dsn: str
    database: str
    schema: str
    owner: str
    reader: str
    source: str
    stranger: str
    close_xid: str

    def login(self, role: str) -> str:
        return make_conninfo(self.dsn, user=getattr(self, role))


def _function_call(value: _IsolatedReader) -> sql.Composed:
    return sql.SQL("SELECT {}.read_exact_close(%s)").format(sql.Identifier(value.schema))


def _fixture_function(value: _IsolatedReader) -> sql.Composed:
    """Create a deliberately disposable SECURITY DEFINER reference experiment."""
    return sql.SQL(
        """
        CREATE FUNCTION {}.read_exact_close(p_operation text) RETURNS jsonb
        LANGUAGE plpgsql SECURITY DEFINER
        SET search_path = pg_catalog, {}, pg_temp
        AS $rpi10$
        DECLARE op RECORD;
        BEGIN
          IF session_user IS DISTINCT FROM {}
          OR current_user IS DISTINCT FROM {}
          OR pg_catalog.current_setting('transaction_isolation') <> 'read committed'
          OR pg_catalog.current_setting('transaction_read_only') <> 'off'
          THEN
            RAISE EXCEPTION 'independent direct CCA reader and READ COMMITTED required'
              USING ERRCODE='42501';
          END IF;

          -- Every lifecycle administrator takes the slot before reading state.
          PERFORM 1 FROM {}.slots
          WHERE semantic_role='ROOT_PROOF_SIGNING' FOR UPDATE;
          IF NOT FOUND THEN
            RAISE EXCEPTION 'missing installed slot' USING ERRCODE='23514';
          END IF;

          SELECT operation_id,semantic_role,request::text AS request_text,
                 request_sha256,close_xid,phase
          INTO op FROM {}.operations WHERE operation_id=p_operation;
          IF NOT FOUND OR op.phase IS DISTINCT FROM 'PENDING_CLOSED'
          OR op.semantic_role IS DISTINCT FROM 'ROOT_PROOF_SIGNING'
          OR op.close_xid::text IS NOT DISTINCT FROM pg_catalog.pg_current_xact_id()::text
          THEN
            RAISE EXCEPTION 'unknown, changed or same-transaction close'
              USING ERRCODE='23514';
          END IF;

          RETURN pg_catalog.jsonb_build_object(
            'operation_id',op.operation_id,
            'semantic_role',op.semantic_role,
            'request_text',op.request_text,
            'request_sha256',op.request_sha256,
            'close_xid',op.close_xid,
            'observation_xid',pg_catalog.pg_current_xact_id()::text,
            'phase',op.phase);
        END;
        $rpi10$
        """
    ).format(
        sql.Identifier(value.schema),
        sql.Identifier(value.schema),
        sql.Literal(value.reader),
        sql.Literal(value.owner),
        sql.Identifier(value.schema),
        sql.Identifier(value.schema),
    )


@pytest.fixture
def isolated_cca_read():
    suffix = uuid.uuid4().hex[:12]
    database = "rpi10_db_" + suffix
    schema = "rpi10_" + suffix
    owner, reader, source, stranger = (
        prefix + suffix for prefix in ("rpi10_o_", "rpi10_c_", "rpi10_s_", "rpi10_x_")
    )
    dsn = make_conninfo(_BASE_DSN, dbname=database)
    with psycopg.connect(_BASE_DSN, autocommit=True) as bootstrap:
        bootstrap.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(database)))
    try:
        with psycopg.connect(dsn, autocommit=True) as bootstrap:
            if bootstrap.info.server_version // 10000 != 16:
                pytest.fail("isolated reader experiment requires PostgreSQL 16")
            bootstrap.execute(
                sql.SQL("REVOKE CREATE,TEMPORARY ON DATABASE {} FROM PUBLIC").format(
                    sql.Identifier(database)
                )
            )
            for role in (owner, reader, source, stranger):
                bootstrap.execute(
                    sql.SQL("CREATE ROLE {} {}").format(
                        sql.Identifier(role),
                        sql.SQL("NOLOGIN" if role == owner else "LOGIN"),
                    )
                )
            bootstrap.execute(
                sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                    sql.Identifier(schema), sql.Identifier(owner)
                )
            )
            bootstrap.execute(
                sql.SQL(
                    "CREATE TABLE {}.slots "
                    "(semantic_role text PRIMARY KEY, installed_marker text NOT NULL)"
                ).format(sql.Identifier(schema))
            )
            bootstrap.execute(
                sql.SQL(
                    "CREATE TABLE {}.operations (operation_id text PRIMARY KEY,"
                    "semantic_role text NOT NULL,request jsonb NOT NULL,"
                    "request_sha256 text NOT NULL,close_xid text NOT NULL,phase text NOT NULL)"
                ).format(sql.Identifier(schema))
            )
            for table in ("slots", "operations"):
                bootstrap.execute(
                    sql.SQL("ALTER TABLE {}.{} OWNER TO {}").format(
                        sql.Identifier(schema), sql.Identifier(table), sql.Identifier(owner)
                    )
                )
            bootstrap.execute(
                sql.SQL("INSERT INTO {}.slots VALUES ('ROOT_PROOF_SIGNING','fixture')").format(
                    sql.Identifier(schema)
                )
            )
            original = {
                "operation_id": "rpi10-close-1",
                "semantic_role": "ROOT_PROOF_SIGNING",
                "identity": "isolated-root-identity",
                "generation": 1,
                "predecessor_state": "ACTIVE",
                "target_state": "VERIFY_ONLY",
            }
            with bootstrap.transaction():
                close_xid = bootstrap.execute(
                    "SELECT pg_catalog.pg_current_xact_id()::text"
                ).fetchone()[0]
                bootstrap.execute(
                    sql.SQL(
                        "INSERT INTO {}.operations"
                        "(operation_id,semantic_role,request,request_sha256,close_xid,phase)"
                        "VALUES ('rpi10-close-1','ROOT_PROOF_SIGNING',%s,'',%s,'PENDING_CLOSED')"
                    ).format(sql.Identifier(schema)),
                    (Jsonb(original), close_xid),
                )
                bootstrap.execute(
                    sql.SQL(
                        "UPDATE {}.operations SET request_sha256="
                        "pg_catalog.encode(pg_catalog.sha256("
                        "pg_catalog.convert_to(request::text,'UTF8')),'hex')"
                    ).format(sql.Identifier(schema))
                )

            value = _IsolatedReader(dsn, database, schema, owner, reader, source, stranger, close_xid)
            # A single install transaction prevents an observable PUBLIC EXECUTE window.
            with bootstrap.transaction():
                bootstrap.execute(_fixture_function(value))
                bootstrap.execute(
                    sql.SQL("ALTER FUNCTION {}.read_exact_close(text) OWNER TO {}").format(
                        sql.Identifier(schema), sql.Identifier(owner)
                    )
                )
                bootstrap.execute(
                    sql.SQL("REVOKE ALL ON FUNCTION {}.read_exact_close(text) FROM PUBLIC").format(
                        sql.Identifier(schema)
                    )
                )
                bootstrap.execute(
                    sql.SQL("GRANT USAGE ON SCHEMA {} TO {},{},{}").format(
                        sql.Identifier(schema),
                        sql.Identifier(reader),
                        sql.Identifier(source),
                        sql.Identifier(stranger),
                    )
                )
                bootstrap.execute(
                    sql.SQL("GRANT EXECUTE ON FUNCTION {}.read_exact_close(text) TO {}").format(
                        sql.Identifier(schema), sql.Identifier(reader)
                    )
                )
        yield value
    finally:
        with psycopg.connect(_BASE_DSN, autocommit=True) as bootstrap:
            bootstrap.execute(
                sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                    sql.Identifier(database)
                )
            )
            for role in (stranger, source, reader, owner):
                bootstrap.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(role)))


def test_exact_committed_jsonb_text_and_independent_xid(isolated_cca_read):
    value = isolated_cca_read
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            result = conn.execute(_function_call(value), ("rpi10-close-1",)).fetchone()[0]
            retained = result["request_text"].encode("utf-8")
            assert result["request_sha256"] == hashlib.sha256(retained).hexdigest()
            assert result["close_xid"] == value.close_xid
            assert result["observation_xid"] != result["close_xid"]
            assert result["phase"] == "PENDING_CLOSED"
            assert result["semantic_role"] == "ROOT_PROOF_SIGNING"
            assert result["operation_id"] == "rpi10-close-1"
            # Original bytes mean the retained PostgreSQL jsonb::text rendering,
            # not client JSON or Python canonical_json_bytes serialization.
            with psycopg.connect(value.dsn, autocommit=True) as bootstrap:
                retained_in_database = bootstrap.execute(
                    sql.SQL(
                        "SELECT request::text FROM {}.operations WHERE operation_id=%s"
                    ).format(sql.Identifier(value.schema)),
                    ("rpi10-close-1",),
                ).fetchone()[0]
            assert retained_in_database == result["request_text"]


@pytest.mark.parametrize("role", ["source", "stranger"])
def test_non_cca_principal_cannot_execute(isolated_cca_read, role):
    value = isolated_cca_read
    with psycopg.connect(value.login(role), autocommit=True) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute(_function_call(value), ("rpi10-close-1",))


def test_cca_has_no_direct_table_access_or_mutation(isolated_cca_read):
    value = isolated_cca_read
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        for verb in ("SELECT * FROM", "UPDATE"):
            statement = (
                sql.SQL("SELECT * FROM {}.slots").format(sql.Identifier(value.schema))
                if verb.startswith("SELECT")
                else sql.SQL("UPDATE {}.slots SET installed_marker='mutated'").format(
                    sql.Identifier(value.schema)
                )
            )
            with pytest.raises(InsufficientPrivilege):
                conn.execute(statement)


def test_public_execute_closed_and_only_pinned_reader_granted(isolated_cca_read):
    value = isolated_cca_read
    with psycopg.connect(value.dsn, autocommit=True) as conn:
        rows = conn.execute(
            "SELECT r.rolname,a.grantee,a.privilege_type FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "CROSS JOIN LATERAL pg_catalog.aclexplode(p.proacl) a "
            "LEFT JOIN pg_catalog.pg_roles r ON r.oid=a.grantee "
            "WHERE n.nspname=%s AND p.proname='read_exact_close'",
            (value.schema,),
        ).fetchall()
        explicit_execute = {name for name, oid, privilege in rows if privilege == "EXECUTE"}
        assert value.reader in explicit_execute
        assert explicit_execute <= {value.reader, value.owner}
        assert all(oid != 0 for _, oid, _ in rows)
        assert value.source not in explicit_execute
        assert value.stranger not in explicit_execute
        attributes = conn.execute(
            "SELECT p.prosecdef,p.proconfig,p.proowner FROM pg_catalog.pg_proc p "
            "JOIN pg_catalog.pg_namespace n ON n.oid=p.pronamespace "
            "WHERE n.nspname=%s AND p.proname='read_exact_close'",
            (value.schema,),
        ).fetchone()
        assert attributes[0] is True
        assert attributes[1] == [f"search_path=pg_catalog, {value.schema}, pg_temp"]
        owner_oid = conn.execute(
            "SELECT oid FROM pg_catalog.pg_roles WHERE rolname=%s", (value.owner,)
        ).fetchone()[0]
        assert attributes[2] == owner_oid


def test_read_committed_required_and_missing_operation_fails_closed(isolated_cca_read):
    value = isolated_cca_read
    with psycopg.connect(value.login("reader"), autocommit=True) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            with pytest.raises(InsufficientPrivilege):
                conn.execute(_function_call(value), ("rpi10-close-1",))
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            with pytest.raises(psycopg.Error, match="unknown, changed"):
                conn.execute(_function_call(value), ("not-installed",))


def test_lock_is_held_until_caller_transaction_completes(isolated_cca_read):
    value = isolated_cca_read
    with (
        psycopg.connect(value.login("reader"), autocommit=True) as reader,
        psycopg.connect(value.dsn, autocommit=True) as concurrent_administrator,
    ):
        with reader.transaction():
            reader.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
            reader.execute(_function_call(value), ("rpi10-close-1",))
            with concurrent_administrator.transaction():
                concurrent_administrator.execute("SET LOCAL lock_timeout = '250ms'")
                with pytest.raises(LockNotAvailable):
                    concurrent_administrator.execute(
                        sql.SQL(
                            "UPDATE {}.slots SET installed_marker='concurrent' "
                            "WHERE semantic_role='ROOT_PROOF_SIGNING'"
                        ).format(sql.Identifier(value.schema))
                    )
        # A released lock allows another transaction to proceed without deadlock.
        with concurrent_administrator.transaction():
            concurrent_administrator.execute("SET LOCAL lock_timeout = '1s'")
            concurrent_administrator.execute(
                sql.SQL(
                    "UPDATE {}.slots SET installed_marker='after-release' "
                    "WHERE semantic_role='ROOT_PROOF_SIGNING'"
                ).format(sql.Identifier(value.schema))
            )
