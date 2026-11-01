"""Rzeczywisty PostgreSQL w nowym pustym klastrze; bez kwalifikacji produktu.

Jawny opt-in oraz katalog binariów PG >=16 są wymagane. Klaster powstaje
wyłącznie w tempfile, jako bieżący nie-root UID, bez TCP i bez zmian kont OS.
Mapowanie bieżącego UID jest fixture, nie podszywa się pod usługę CryptoHunter.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import psycopg
import pytest

from bot_core import linux_service_postgresql as adapter
from deployment.linux_physical_postgresql import validate_service_connection_observation

pytestmark = pytest.mark.skipif(
    sys.platform != "linux"
    or os.environ.get("CRYPTOHUNTER_RUN_LINUX_PG_ADAPTER_INTEGRATION") != "1",
    reason="Rzeczywisty tymczasowy PostgreSQL wymaga jawnego opt-in na Linux",
)


@pytest.fixture(scope="module")
def isolated_postgresql():
    import pwd

    assert os.getuid() != 0, "Fixture uruchamiaj jako nieuprzywilejowany UID"
    bindir = Path(os.environ["CRYPTOHUNTER_TEST_PG_BINDIR"]).resolve(strict=True)
    principal = pwd.getpwuid(os.getuid()).pw_name
    with tempfile.TemporaryDirectory(prefix="ch-pg-adapter-", dir="/tmp") as temp:
        root = Path(temp)
        data, sockets = root / "data", root / "socket"
        sockets.mkdir(mode=0o700)
        started = False

        def command(*arguments):
            return subprocess.run(
                [str(bindir / arguments[0]), *arguments[1:]],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )

        command(
            "initdb",
            "-D",
            str(data),
            "--auth-local=reject",
            "--auth-host=reject",
            "--no-locale",
            "--encoding=UTF8",
        )
        hba = data / "pg_hba.conf"
        ident = data / "pg_ident.conf"
        base_hba = (
            f"local all {principal} peer\n"
            "local freshness_gate freshness_crypto_verifier peer map=stage9_adapter_fixture\n"
            "local all all reject\n"
        )
        base_ident = f"stage9_adapter_fixture {principal} freshness_crypto_verifier\n"
        hba.write_text(base_hba)
        ident.write_text(base_ident)
        with (data / "postgresql.conf").open("a") as stream:
            stream.write(
                f"\nlisten_addresses=''\nunix_socket_directories='{sockets}'\nunix_socket_permissions=0700\nport=55441\nfsync=on\nsynchronous_commit=on\nfull_page_writes=on\n"
            )

        def connect(*, user=principal, database="postgres", autocommit=False):
            return psycopg.connect(
                host=str(sockets),
                port=55441,
                dbname=database,
                user=user,
                password=None,
                passfile="/dev/null",
                sslmode="disable",
                connect_timeout=3,
                autocommit=autocommit,
            )

        def reload():
            with connect(autocommit=True) as admin:
                before = admin.execute("SELECT pg_conf_load_time()").fetchone()[0]
                admin.execute("SELECT pg_reload_conf()")
                deadline = time.monotonic() + 5
                while admin.execute("SELECT pg_conf_load_time()").fetchone()[0] == before:
                    assert time.monotonic() < deadline, "Timeout reload fixture"
                    time.sleep(0.02)

        try:
            command("pg_ctl", "-D", str(data), "-l", str(root / "server.log"), "-w", "start")
            started = True
            with connect(autocommit=True) as admin:
                assert admin.info.server_version >= 160000
                admin.execute("CREATE DATABASE freshness_gate")
                admin.execute(
                    "CREATE ROLE freshness_crypto_verifier LOGIN NOSUPERUSER NOCREATEDB "
                    "NOCREATEROLE NOINHERIT NOREPLICATION NOBYPASSRLS"
                )
            with connect(database="freshness_gate", autocommit=True) as admin:
                admin.execute("CREATE SCHEMA freshness_authority")
                admin.execute("REVOKE ALL ON SCHEMA freshness_authority FROM PUBLIC")
                admin.execute(
                    "GRANT USAGE ON SCHEMA freshness_authority TO freshness_crypto_verifier"
                )
                admin.execute("CREATE TABLE freshness_authority.credentials(value text)")
                admin.execute("CREATE TABLE freshness_authority.prepared_verifications(value text)")
                admin.execute(
                    "INSERT INTO freshness_authority.credentials "
                    "VALUES ('jawny_fixture_bez_sekretu')"
                )
                admin.execute(
                    "CREATE FUNCTION freshness_authority.compare_and_advance"
                    "(text,jsonb,bytea,bytea) RETURNS integer LANGUAGE sql AS 'SELECT 1'"
                )
                admin.execute(
                    "REVOKE ALL ON FUNCTION freshness_authority.compare_and_advance"
                    "(text,jsonb,bytea,bytea) FROM PUBLIC"
                )
            yield {
                "connect": connect,
                "reload": reload,
                "command": command,
                "data": data,
                "hba": hba,
                "ident": ident,
                "base_hba": base_hba,
                "base_ident": base_ident,
                "principal": principal,
            }
        finally:
            if started:
                command("pg_ctl", "-D", str(data), "-m", "fast", "-w", "stop")


@pytest.fixture
def cluster(isolated_postgresql, monkeypatch):
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    cluster = isolated_postgresql
    try:
        yield cluster
    finally:
        cluster["hba"].write_text(cluster["base_hba"])
        cluster["ident"].write_text(cluster["base_ident"])
        cluster["reload"]()


def observe(cluster, connection):
    current = adapter.observe_current_process()
    return adapter.observe_service_connection(
        connection,
        challenge_id="a" * 64,
        installation_id="b" * 64,
        role="verifier",
        expected_uid=current["uid"],
        expected_gid=current["gid"],
        expected_os_principal=cluster["principal"],
        expected_database_role="freshness_crypto_verifier",
        expected_executable_sha256=current["exe_sha256"],
        expected_cgroup=current["cgroup"],
    )


@contextmanager
def verifier(cluster):
    connection = cluster["connect"](user="freshness_crypto_verifier", database="freshness_gate")
    try:
        yield connection
    finally:
        connection.close()


def test_real_peer_mapping_backend_four_denials_and_unchanged_state(cluster):
    with cluster["connect"](database="freshness_gate") as admin:
        before = admin.execute("TABLE freshness_authority.credentials").fetchall()
    with verifier(cluster) as connection:
        report = observe(cluster, connection)
        assert report["identity_row"][0] == "peer:" + cluster["principal"]
        assert report["identity_row"][3] == connection.info.backend_pid
        assert len(report["denials"]) == 4
        assert connection.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
        validate_service_connection_observation(
            report,
            challenge_id="a" * 64,
            installation_id="b" * 64,
            role="verifier",
            expected_process=adapter.observe_current_process(),
            expected_os_principal=cluster["principal"],
            expected_database_role="freshness_crypto_verifier",
        )
    with cluster["connect"](database="freshness_gate") as admin:
        assert admin.execute("TABLE freshness_authority.credentials").fetchall() == before


def test_real_trust_authentication_is_not_peer(cluster):
    cluster["hba"].write_text(
        cluster["base_hba"].replace("peer map=stage9_adapter_fixture", "trust")
    )
    cluster["reload"]()
    with verifier(cluster) as connection:
        with pytest.raises(adapter.ServicePostgreSQLObservationError, match="AUTHENTICATED_PEER"):
            observe(cluster, connection)


def test_real_wrong_ident_mapping_refuses_connection(cluster):
    cluster["ident"].write_text(f"stage9_adapter_fixture {cluster['principal']} forbidden_role\n")
    cluster["reload"]()
    with pytest.raises(psycopg.OperationalError):
        with verifier(cluster):
            pytest.fail("Błędne mapowanie peer zostało zaakceptowane")


def test_real_wrong_database_role_refuses(cluster):
    with cluster["connect"](database="freshness_gate") as connection:
        with pytest.raises(adapter.ServicePostgreSQLObservationError, match="AUTHENTICATED_PEER"):
            observe(cluster, connection)


def test_real_accidentally_granted_foreign_select_refuses(cluster):
    with cluster["connect"](database="freshness_gate", autocommit=True) as admin:
        admin.execute(
            "GRANT SELECT ON freshness_authority.credentials TO freshness_crypto_verifier"
        )
        try:
            with verifier(cluster) as connection:
                with pytest.raises(
                    adapter.ServicePostgreSQLObservationError, match="FORBIDDEN_OPERATION_SUCCEEDED"
                ):
                    observe(cluster, connection)
        finally:
            admin.execute(
                "REVOKE SELECT ON freshness_authority.credentials FROM freshness_crypto_verifier"
            )


def test_real_lock_timeout_is_not_permission_denial(cluster):
    with cluster["connect"](database="freshness_gate") as admin:
        admin.execute("LOCK TABLE freshness_authority.credentials IN ACCESS EXCLUSIVE MODE")
        with verifier(cluster) as connection:
            with pytest.raises(
                adapter.ServicePostgreSQLObservationError, match="EFFECTIVE_PRIVILEGE_DENIAL"
            ):
                observe(cluster, connection)
        admin.rollback()


def test_real_backend_connection_loss_is_not_denial(cluster):
    with verifier(cluster) as connection:
        with cluster["connect"](autocommit=True) as admin:
            admin.execute("SELECT pg_terminate_backend(%s)", (connection.info.backend_pid,))
        with pytest.raises(adapter.ServicePostgreSQLObservationError):
            observe(cluster, connection)


def test_real_isolated_database_restart_reconnects_and_requalifies(cluster):
    cluster["command"](
        "pg_ctl",
        "-D",
        str(cluster["data"]),
        "-l",
        str(cluster["data"].parent / "server.log"),
        "-m",
        "fast",
        "-w",
        "restart",
    )
    with verifier(cluster) as connection:
        assert observe(cluster, connection)["identity_row"][0] == "peer:" + cluster["principal"]
