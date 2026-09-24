"""Executable PostgreSQL 16 proof for the frozen SECURITY DEFINER caller model."""

from __future__ import annotations

import os
import uuid

import psycopg
from psycopg import sql
from psycopg.errors import InsufficientPrivilege
import pytest

pytestmark = pytest.mark.external_postgresql

BASE_DSN = os.environ.get(
    "ENTITLEMENT_REGISTRY_POSTGRES_ADMIN_DSN",
    "host=127.0.0.1 port=55432 dbname=postgres user=postgres",
)


def _dsn(role: str) -> str:
    return f"host=127.0.0.1 port=55432 dbname=postgres user={role}"


@pytest.fixture()
def boundary() -> dict[str, str]:
    suffix = uuid.uuid4().hex[:10]
    names = {
        "schema": f"fcp_{suffix}",
        "owner": f"fcpo_{suffix}",
        "verifier": f"fcpv_{suffix}",
        "runtime": f"fcpr_{suffix}",
        "outsider": f"fcpx_{suffix}",
    }
    identifiers = {key: sql.Identifier(value) for key, value in names.items()}
    with psycopg.connect(BASE_DSN, autocommit=True) as conn:
        for role in ("owner", "verifier", "runtime", "outsider"):
            login = sql.SQL("NOLOGIN") if role == "owner" else sql.SQL("LOGIN")
            conn.execute(
                sql.SQL(
                    "CREATE ROLE {} {} NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                ).format(identifiers[role], login)
            )
        conn.execute(
            sql.SQL("CREATE SCHEMA {} AUTHORIZATION {}").format(
                identifiers["schema"], identifiers["owner"]
            )
        )
        for function_name, caller in (("prepare_probe", "verifier"), ("cas_probe", "runtime")):
            conn.execute(
                sql.SQL(
                    "CREATE FUNCTION {}.{}() RETURNS TABLE(session_identity name,effective_identity name) "
                    "LANGUAGE plpgsql SECURITY DEFINER SET search_path=pg_catalog AS $fn$ "
                    "BEGIN IF session_user <> {} THEN RAISE EXCEPTION 'wrong login principal'; END IF; "
                    "RETURN QUERY SELECT session_user::name,current_user::name; END $fn$"
                ).format(
                    identifiers["schema"],
                    sql.Identifier(function_name),
                    sql.Literal(names[caller]),
                )
            )
            conn.execute(
                sql.SQL("ALTER FUNCTION {}.{}() OWNER TO {}").format(
                    identifiers["schema"], sql.Identifier(function_name), identifiers["owner"]
                )
            )
            conn.execute(
                sql.SQL("REVOKE ALL ON FUNCTION {}.{}() FROM PUBLIC").format(
                    identifiers["schema"], sql.Identifier(function_name)
                )
            )
            conn.execute(
                sql.SQL("GRANT EXECUTE ON FUNCTION {}.{}() TO {}").format(
                    identifiers["schema"], sql.Identifier(function_name), identifiers[caller]
                )
            )
        conn.execute(
            sql.SQL("GRANT USAGE ON SCHEMA {} TO {},{},{}").format(
                identifiers["schema"],
                identifiers["verifier"],
                identifiers["runtime"],
                identifiers["outsider"],
            )
        )
    try:
        yield names
    finally:
        with psycopg.connect(BASE_DSN, autocommit=True) as conn:
            conn.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(identifiers["schema"]))
            for role in ("outsider", "runtime", "verifier", "owner"):
                conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(identifiers[role]))


def _call(names: dict[str, str], login: str, function_name: str) -> tuple[str, str]:
    with psycopg.connect(_dsn(names[login])) as conn:
        row = conn.execute(
            sql.SQL("SELECT * FROM {}.{}()").format(
                sql.Identifier(names["schema"]), sql.Identifier(function_name)
            )
        ).fetchone()
        assert row is not None
        return str(row[0]), str(row[1])


def test_security_definer_principal_acl_and_no_set_role_path(boundary: dict[str, str]) -> None:
    names = boundary
    assert _call(names, "verifier", "prepare_probe") == (names["verifier"], names["owner"])
    assert _call(names, "runtime", "cas_probe") == (names["runtime"], names["owner"])

    with pytest.raises(InsufficientPrivilege):
        _call(names, "runtime", "prepare_probe")
    with pytest.raises(InsufficientPrivilege):
        _call(names, "verifier", "cas_probe")
    with pytest.raises(InsufficientPrivilege):
        _call(names, "outsider", "prepare_probe")

    with psycopg.connect(_dsn(names["runtime"])) as conn:
        with pytest.raises(InsufficientPrivilege):
            conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(names["verifier"])))

    with psycopg.connect(BASE_DSN) as conn:
        memberships = conn.execute(
            "SELECT count(*) FROM pg_auth_members m JOIN pg_roles parent ON parent.oid=m.roleid "
            "JOIN pg_roles child ON child.oid=m.member WHERE parent.rolname=ANY(%s) OR child.rolname=ANY(%s)",
            (list(names.values())[1:], list(names.values())[1:]),
        ).fetchone()
        assert memberships == (0,)
        public_execute = conn.execute(
            "SELECT count(*) FROM information_schema.routine_privileges "
            "WHERE specific_schema=%s AND grantee='PUBLIC' AND privilege_type='EXECUTE'",
            (names["schema"],),
        ).fetchone()
        assert public_execute == (0,)
        settings = dict(
            conn.execute(
                "SELECT name,setting FROM pg_settings WHERE name IN "
                "('server_version_num','fsync','synchronous_commit')"
            ).fetchall()
        )
        assert int(settings["server_version_num"]) >= 160000
        assert settings["fsync"] == "on"
        assert settings["synchronous_commit"] == "on"
