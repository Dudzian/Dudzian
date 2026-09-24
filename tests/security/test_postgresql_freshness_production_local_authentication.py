"""Real PostgreSQL >=16 proof of the exact PRODUCTION_LOCAL peer-auth boundary.

The cluster, HBA and catalog objects are isolated fixtures.  No production
FreshnessAuthority schema or implementation is installed by this test.
"""

from __future__ import annotations

import sys

import pytest

if not sys.platform.startswith("linux"):
    pytest.skip(
        "ten moduł jest natywnym dowodem Linux/Unix peer-auth z principalami systemowymi",
        allow_module_level=True,
    )

import os
from pathlib import Path
import pwd  # noqa: E402 -- guarded Linux-only import
import shutil
import socket
import subprocess
import tempfile

import psycopg

AUTHORITY_ROLES = (
    "freshness_schema_owner",
    "freshness_function_owner",
    "freshness_admin",
    "freshness_crypto_verifier",
    "freshness_runtime",
    "freshness_reader",
)
OS_IDENTITIES = (
    "os_freshness_crypto_verifier",
    "os_freshness_runtime",
    "os_freshness_outsider",
)


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def _as(identity: str, command: list[str], *, check: bool = True):
    return _run(["runuser", "-u", identity, "--", *command], check=check)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="module")
def isolated_peer_cluster():
    if os.geteuid() != 0:
        pytest.skip("the peer-auth proof needs root to switch among dedicated OS principals")
    bindir = Path(_run(["pg_config", "--bindir"]).stdout.strip())
    if not (bindir / "initdb").exists():
        pytest.skip("PostgreSQL server binaries are unavailable")

    created_users: list[str] = []
    for identity in OS_IDENTITIES:
        try:
            pwd.getpwnam(identity)
        except KeyError:
            _run(
                [
                    "useradd",
                    "--system",
                    "--no-create-home",
                    "--shell",
                    "/usr/sbin/nologin",
                    identity,
                ]
            )
            created_users.append(identity)

    base = Path(tempfile.mkdtemp(prefix="freshness-peer-gate-"))
    data, socket_dir = base / "data", base / "socket"
    socket_dir.mkdir()
    _run(["chown", "-R", "postgres:postgres", str(base)])
    base.chmod(0o711)
    port = _free_port()
    try:
        _as(
            "postgres",
            [str(bindir / "initdb"), "-D", str(data), "--auth-local=reject", "--auth-host=reject"],
        )
        (data / "pg_ident.conf").write_text(
            "freshness_authority_map os_freshness_crypto_verifier freshness_crypto_verifier\n"
            "freshness_authority_map os_freshness_runtime freshness_runtime\n"
            "freshness_admin_map root postgres\n",
            encoding="ascii",
        )
        (data / "pg_hba.conf").write_text(
            "local all postgres peer map=freshness_admin_map\n"
            "local freshness_gate freshness_crypto_verifier peer map=freshness_authority_map\n"
            "local freshness_gate freshness_runtime peer map=freshness_authority_map\n"
            "local freshness_gate all reject\n"
            "local all all reject\n",
            encoding="ascii",
        )
        with (data / "postgresql.conf").open("a", encoding="ascii") as config:
            config.write(
                f"\nlisten_addresses = ''\nunix_socket_directories = '{socket_dir}'\nport = {port}\n"
                "fsync = on\nsynchronous_commit = on\n"
            )
        _run(["chown", "postgres:postgres", str(data / "pg_ident.conf"), str(data / "pg_hba.conf")])
        # Redirect the server explicitly: otherwise its inherited captured
        # descriptors keep subprocess.run() waiting after pg_ctl has exited.
        _as(
            "postgres",
            [
                str(bindir / "pg_ctl"),
                "-D",
                str(data),
                "-l",
                str(base / "postgres.log"),
                "-w",
                "start",
            ],
        )
        admin = f"host={socket_dir} port={port} dbname=postgres user=postgres"
        with psycopg.connect(admin, autocommit=True) as conn:
            conn.execute("CREATE DATABASE freshness_gate")
        gate_admin = f"host={socket_dir} port={port} dbname=freshness_gate user=postgres"
        with psycopg.connect(gate_admin, autocommit=True) as conn:
            for role in AUTHORITY_ROLES:
                login = role in {"freshness_crypto_verifier", "freshness_runtime"}
                conn.execute(
                    f'CREATE ROLE "{role}" {"LOGIN" if login else "NOLOGIN"} NOINHERIT '
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS"
                )
            conn.execute("CREATE SCHEMA freshness_authority AUTHORIZATION freshness_schema_owner")
            conn.execute(
                "GRANT CREATE, USAGE ON SCHEMA freshness_authority TO freshness_function_owner"
            )
            conn.execute("CREATE TABLE freshness_authority.probe_state (value integer NOT NULL)")
            conn.execute(
                "ALTER TABLE freshness_authority.probe_state OWNER TO freshness_schema_owner"
            )
            conn.execute(
                "GRANT SELECT,INSERT,UPDATE,DELETE ON freshness_authority.probe_state TO freshness_function_owner"
            )
            for name, expected in (
                ("prepare_verified_freshness_candidate", "freshness_crypto_verifier"),
                ("compare_and_advance", "freshness_runtime"),
            ):
                conn.execute(f"""
                    CREATE FUNCTION freshness_authority.{name}()
                    RETURNS TABLE(session_identity name, effective_identity name)
                    LANGUAGE sql SECURITY DEFINER SET search_path = pg_catalog
                    AS $$ SELECT session_user::name, current_user::name
                          WHERE session_user = '{expected}' $$
                """)
                conn.execute(
                    f"ALTER FUNCTION freshness_authority.{name}() OWNER TO freshness_function_owner"
                )
                conn.execute(f"REVOKE ALL ON FUNCTION freshness_authority.{name}() FROM PUBLIC")
            conn.execute(
                "REVOKE CREATE ON SCHEMA freshness_authority FROM freshness_function_owner"
            )
            conn.execute(
                "GRANT USAGE ON SCHEMA freshness_authority TO freshness_crypto_verifier, freshness_runtime"
            )
            conn.execute(
                "GRANT EXECUTE ON FUNCTION freshness_authority.prepare_verified_freshness_candidate() TO freshness_crypto_verifier"
            )
            conn.execute(
                "GRANT EXECUTE ON FUNCTION freshness_authority.compare_and_advance() TO freshness_runtime"
            )
        yield {
            "bindir": bindir,
            "data": data,
            "socket": socket_dir,
            "port": port,
            "admin": gate_admin,
        }
    finally:
        if data.exists() and (data / "postmaster.pid").exists():
            _as(
                "postgres",
                [str(bindir / "pg_ctl"), "-D", str(data), "-m", "immediate", "-w", "stop"],
                check=False,
            )
        shutil.rmtree(base, ignore_errors=True)
        for identity in reversed(created_users):
            _run(["userdel", identity], check=False)


def _psql(cluster, os_identity: str, db_role: str, query: str):
    return _as(
        os_identity,
        [
            str(cluster["bindir"] / "psql"),
            "-XAt",
            "-h",
            str(cluster["socket"]),
            "-p",
            str(cluster["port"]),
            "-d",
            "freshness_gate",
            "-U",
            db_role,
            "-c",
            query,
        ],
        check=False,
    )


def test_new_connections_are_bound_to_kernel_peer_identity(isolated_peer_cluster):
    cluster = isolated_peer_cluster
    verifier = _psql(
        cluster, "os_freshness_crypto_verifier", "freshness_crypto_verifier", "SELECT session_user"
    )
    runtime = _psql(cluster, "os_freshness_runtime", "freshness_runtime", "SELECT session_user")
    assert verifier.returncode == 0 and verifier.stdout.strip() == "freshness_crypto_verifier"
    assert runtime.returncode == 0 and runtime.stdout.strip() == "freshness_runtime"

    # These are genuinely new connections, not SET ROLE probes.
    assert (
        _psql(cluster, "os_freshness_runtime", "freshness_crypto_verifier", "SELECT 1").returncode
        != 0
    )
    assert (
        _psql(cluster, "os_freshness_outsider", "freshness_crypto_verifier", "SELECT 1").returncode
        != 0
    )
    assert (
        _psql(cluster, "os_freshness_crypto_verifier", "freshness_runtime", "SELECT 1").returncode
        != 0
    )

    assert _psql(
        cluster,
        "os_freshness_crypto_verifier",
        "freshness_crypto_verifier",
        "SELECT * FROM freshness_authority.prepare_verified_freshness_candidate()",
    ).stdout.strip() == ("freshness_crypto_verifier|freshness_function_owner")
    assert _psql(
        cluster,
        "os_freshness_runtime",
        "freshness_runtime",
        "SELECT * FROM freshness_authority.compare_and_advance()",
    ).stdout.strip() == ("freshness_runtime|freshness_function_owner")


def test_hba_catalog_roles_owners_acl_and_durability(isolated_peer_cluster):
    cluster = isolated_peer_cluster
    expected_hba = [
        ("local", "all", "postgres", "peer", "map=freshness_admin_map"),
        (
            "local",
            "freshness_gate",
            "freshness_crypto_verifier",
            "peer",
            "map=freshness_authority_map",
        ),
        ("local", "freshness_gate", "freshness_runtime", "peer", "map=freshness_authority_map"),
        ("local", "freshness_gate", "all", "reject", None),
        ("local", "all", "all", "reject", None),
    ]
    with psycopg.connect(cluster["admin"]) as conn:
        hba = conn.execute(
            "SELECT type, array_to_string(database, ','), array_to_string(user_name, ','), auth_method, options[1] "
            "FROM pg_hba_file_rules ORDER BY line_number"
        ).fetchall()
        assert hba == expected_hba
        settings = dict(
            conn.execute(
                "SELECT name,setting FROM pg_settings WHERE name IN ('server_version_num','fsync','synchronous_commit','listen_addresses')"
            ).fetchall()
        )
        assert int(settings["server_version_num"]) >= 160000
        assert settings == {
            **settings,
            "fsync": "on",
            "synchronous_commit": "on",
            "listen_addresses": "",
        }

        roles = conn.execute(
            "SELECT rolname,rolcanlogin,rolinherit FROM pg_roles WHERE rolname=ANY(%s)",
            (list(AUTHORITY_ROLES),),
        ).fetchall()
        assert len(roles) == 6
        assert all(not inherit for _, _, inherit in roles)
        assert {name for name, login, _ in roles if login} == {
            "freshness_crypto_verifier",
            "freshness_runtime",
        }
        assert conn.execute(
            "SELECT count(*) FROM pg_auth_members m JOIN pg_roles p ON p.oid=m.roleid JOIN pg_roles c ON c.oid=m.member "
            "WHERE p.rolname=ANY(%s) OR c.rolname=ANY(%s)",
            (list(AUTHORITY_ROLES), list(AUTHORITY_ROLES)),
        ).fetchone() == (0,)

        owners = conn.execute("""
            SELECT n.nspowner, so.oid, so.rolname,
                   array_agg(DISTINCT po.rolname ORDER BY po.rolname),
                   bool_and(p.prosecdef), bool_and(p.proconfig = ARRAY['search_path=pg_catalog'])
            FROM pg_namespace n JOIN pg_roles so ON so.oid=n.nspowner
            JOIN pg_proc p ON p.pronamespace=n.oid JOIN pg_roles po ON po.oid=p.proowner
            WHERE n.nspname='freshness_authority' GROUP BY n.nspowner,so.oid,so.rolname
        """).fetchone()
        assert owners[1] == owners[0] and owners[2] == "freshness_schema_owner"
        assert owners[3:] == (["freshness_function_owner"], True, True)
        function_owner_oid = conn.execute(
            "SELECT oid FROM pg_roles WHERE rolname='freshness_function_owner'"
        ).fetchone()[0]
        assert function_owner_oid != owners[0]

        acl = conn.execute("""
            SELECT p.proname, x.grantee::regrole::text, x.privilege_type, x.is_grantable
            FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            CROSS JOIN LATERAL aclexplode(coalesce(p.proacl, acldefault('f',p.proowner))) x
            WHERE n.nspname='freshness_authority' ORDER BY 1,2
        """).fetchall()
        assert acl == [
            ("compare_and_advance", "freshness_function_owner", "EXECUTE", False),
            ("compare_and_advance", "freshness_runtime", "EXECUTE", False),
            ("prepare_verified_freshness_candidate", "freshness_crypto_verifier", "EXECUTE", False),
            ("prepare_verified_freshness_candidate", "freshness_function_owner", "EXECUTE", False),
        ]
        assert conn.execute("""
            SELECT
              has_function_privilege('freshness_crypto_verifier','freshness_authority.prepare_verified_freshness_candidate()','EXECUTE'),
              has_function_privilege('freshness_crypto_verifier','freshness_authority.compare_and_advance()','EXECUTE'),
              has_function_privilege('freshness_runtime','freshness_authority.prepare_verified_freshness_candidate()','EXECUTE'),
              has_function_privilege('freshness_runtime','freshness_authority.compare_and_advance()','EXECUTE')
        """).fetchone() == (True, False, False, True)
        usage = conn.execute(
            """
            SELECT r.rolname, has_schema_privilege(r.rolname,'freshness_authority','USAGE')
            FROM pg_roles r WHERE r.rolname=ANY(%s) ORDER BY r.rolname
        """,
            (list(AUTHORITY_ROLES),),
        ).fetchall()
        assert {name for name, allowed in usage if allowed} == {
            "freshness_schema_owner",
            "freshness_function_owner",
            "freshness_crypto_verifier",
            "freshness_runtime",
        }
        for role in ("freshness_runtime", "freshness_crypto_verifier", "freshness_admin"):
            assert conn.execute(
                "SELECT has_table_privilege(%s,'freshness_authority.probe_state','INSERT,UPDATE,DELETE,TRUNCATE')",
                (role,),
            ).fetchone() == (False,)
