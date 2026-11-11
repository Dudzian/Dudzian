"""Opt-in tests of the real PG16 host probe in a disposable non-root cluster.

Only fixture paths, port and the current OS/SQL principal replace the fixed
production constants. Authentication methods, HBA order/addresses and the SQL
policy remain unchanged. This cannot attest production paths, root/postgres
identities, systemd or the operator's Ubuntu VM. No system service/account or
existing PostgreSQL cluster is accessed.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest

from deployment import production_local_installer_host as probe

pytestmark = [
    pytest.mark.integration,
    pytest.mark.external_postgresql,
    pytest.mark.skipif(
        sys.platform != "linux"
        or os.environ.get("CRYPTOHUNTER_RUN_STAGE9_HOST_PG_INTEGRATION") != "1",
        reason="Disposable PG16 host-probe tests require explicit Linux opt-in",
    ),
]


def _unused_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


class PostgreSQLFixture:
    def __init__(self, root: Path, bindir: Path, principal: str, **overrides):
        self.root = root
        self.bindir = bindir
        self.principal = principal
        self.profile = overrides.get("profile", "ubuntu")
        self.expected_data = root / "var/lib/postgresql/16/main"
        self.data = root / "foreign-data" if overrides.get("wrong_data") else self.expected_data
        self.config_directory = root / "etc/postgresql/16/main"
        self.config = self.config_directory / "postgresql.conf"
        self.stock_socket = root / "socket.ubuntu"
        self.installed_socket = root / "socket.installed"
        self.socket = (
            root / "socket.foreign"
            if overrides.get("wrong_socket")
            else self.stock_socket
            if self.profile == "ubuntu"
            else self.installed_socket
        )
        self.expected_port = _unused_port()
        self.port = _unused_port() if overrides.get("wrong_port") else self.expected_port
        while overrides.get("wrong_port") and self.port == self.expected_port:
            self.port = _unused_port()
        self.listen = overrides.get("listen", "localhost" if self.profile == "ubuntu" else "")
        self.records = []
        self.started = False
        self.root.chmod(0o755)
        self.config_directory.mkdir(parents=True)
        (self.config_directory / "conf.d").mkdir(mode=0o755)
        self.socket.mkdir(mode=0o700)
        self.command(
            "initdb",
            "-D",
            str(self.data),
            "--no-locale",
            "--encoding=UTF8",
            "--auth-local=peer",
            "--auth-host=reject",
        )
        assert self.command("postgres", "--version").stdout.startswith("postgres (PostgreSQL) 16.")
        auth_directory = (
            self.config_directory
            if self.profile == "ubuntu"
            else root / "etc/cryptohunter/postgresql"
        )
        auth_directory.mkdir(parents=True, exist_ok=True)
        self.hba = auth_directory / "pg_hba.conf"
        self.ident = auth_directory / "pg_ident.conf"
        if self.profile == "ubuntu":
            hba = (
                f"local all {principal} peer\n"
                "local all all peer\n"
                "host all all 127.0.0.1/32 scram-sha-256\n"
                "host all all ::1/128 scram-sha-256\n"
                "local replication all peer\n"
                "host replication all 127.0.0.1/32 scram-sha-256\n"
                "host replication all ::1/128 scram-sha-256\n"
            )
            ident = "# Ubuntu baseline has no user maps\n"
        else:
            repository = Path(__file__).resolve().parents[2]
            hba = (
                (repository / "deployment/postgresql/pg_hba.conf")
                .read_text()
                .replace("local all postgres ", f"local all {principal} ")
            )
            ident = (
                (repository / "deployment/postgresql/pg_ident.conf")
                .read_text()
                .replace(
                    "freshness_admin_map root postgres",
                    f"freshness_admin_map {principal} {principal}",
                )
            )
        if self.profile == "installed":
            (self.config_directory / "pg_hba.conf").write_text("local all all peer\n")
            (self.config_directory / "pg_ident.conf").write_text("# inactive Ubuntu maps\n")
        self.hba.write_text(overrides.get("hba", hba), encoding="utf-8")
        self.ident.write_text(ident, encoding="utf-8")
        self.config.write_text(
            f"data_directory = '{self.data}'\n"
            f"hba_file = '{self.hba}'\nident_file = '{self.ident}'\n"
            f"port = {self.port}\nlisten_addresses = '{self.listen}'\n"
            f"unix_socket_directories = '{self.socket}'\n"
            "unix_socket_permissions = 0700\ninclude_dir = 'conf.d'\n"
            f"fsync = {overrides.get('fsync', 'on')}\n"
            f"synchronous_commit = {overrides.get('synchronous_commit', 'on')}\n",
            encoding="utf-8",
        )
        for directory in (self.root / "etc", *(self.root / "etc").rglob("*")):
            if directory.is_dir():
                directory.chmod(0o755)
        if self.profile == "installed":
            auth_directory.chmod(0o750)
        for ancestor in (self.root / "var", self.root / "var/lib"):
            if ancestor.exists():
                ancestor.chmod(0o755)

    def command(self, executable: str, *arguments: str):
        return subprocess.run(
            [str(self.bindir / executable), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def start(self):
        self.command(
            "pg_ctl",
            "-D",
            str(self.data),
            "-l",
            str(self.root / "server.log"),
            "-o",
            f"-c config_file={self.config}",
            "-w",
            "start",
        )
        self.started = True

    def stop(self):
        if self.started:
            self.command("pg_ctl", "-D", str(self.data), "-m", "fast", "-w", "stop")
            self.started = False

    def sql(self, statement: str, *, readonly=True):
        command = probe.POSTGRESQL_ROOT_CONFIG_QUERY
        connection = command[command.index("-d") + 1]
        if not readonly:
            connection = connection.replace(
                "default_transaction_read_only=on", "default_transaction_read_only=off"
            )
        return subprocess.run(
            [
                str(self.bindir / "psql"),
                "-X",
                "-w",
                "-qAt",
                "-h",
                str(self.socket),
                "-p",
                str(self.port),
                "-U",
                self.principal,
                "-d",
                connection,
                "-v",
                "ON_ERROR_STOP=1",
                "-c",
                statement,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )

    def fixture_sql(self, statement: str) -> str:
        # Exact substitutions adapt the fixed contract to an unprivileged fixture;
        # they do not replace HBA methods, order, address lists or other policy.
        substitutions = (
            ("/etc/postgresql/16/main", str(self.config_directory)),
            ("/etc/cryptohunter/postgresql", str(self.root / "etc/cryptohunter/postgresql")),
            ("/var/lib/postgresql/16/main", str(self.expected_data)),
            ("/var/run/postgresql", str(self.stock_socket)),
            ("/run/postgresql", str(self.installed_socket)),
            ("'5432'", repr(str(self.expected_port))),
            ("'peer:root'", repr("peer:" + self.principal)),
            ("'peer:postgres'", repr("peer:" + self.principal)),
            ("'postgres'", repr(self.principal)),
            ("'root'", repr(self.principal)),
            ('"postgres"', json.dumps(self.principal)),
            ('"root"', json.dumps(self.principal)),
        )
        for production, fixture in substitutions:
            statement = statement.replace(production, fixture)
        return statement

    def dispatch(self, arguments):
        assert arguments in (probe.POSTGRESQL_CONFIG_QUERY, probe.POSTGRESQL_ROOT_CONFIG_QUERY)
        result = self.sql(self.fixture_sql(arguments[-1]))
        self.records.append(result)
        return result.stdout.strip() if result.returncode == 0 else None

    def ready(self):
        return probe._postgresql_config_ready(os.getuid())

    def assert_refused(self, *, sql_denied=True):
        self.records.clear()
        if sql_denied:
            # Exercise SQL independently even if the static closure rejects the
            # same malformed file before the complete probe reaches its query.
            result = self.sql(self.fixture_sql(probe.POSTGRESQL_ROOT_CONFIG_QUERY[-1]))
            assert result.returncode != 0 or json.loads(result.stdout)["ready"] is not True
        assert not self.ready()

    def file_observation(self):
        result = {}
        for path in self.root.rglob("*"):
            if path.is_file():
                metadata = path.stat()
                result[str(path.relative_to(self.root))] = (
                    metadata.st_mode,
                    metadata.st_uid,
                    metadata.st_gid,
                    metadata.st_mtime_ns,
                    metadata.st_ctime_ns,
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
        return result


@pytest.fixture
def postgresql(monkeypatch):
    import pwd

    assert os.getuid() != 0, "Run disposable PostgreSQL fixtures as an unprivileged UID"
    bindir = Path(os.environ["CRYPTOHUNTER_TEST_PG_BINDIR"]).resolve(strict=True)
    principal = pwd.getpwuid(os.getuid()).pw_name
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    snapshot = probe._pg_snapshot

    @contextmanager
    def create(**options):
        with tempfile.TemporaryDirectory(prefix="stage9-host-pg-", dir="/tmp") as temp:
            cluster = PostgreSQLFixture(Path(temp), bindir, principal, **options)
            monkeypatch.setattr(
                probe,
                "_pg_snapshot",
                lambda postgres_uid: snapshot(
                    postgres_uid, root=cluster.root, owner_uid=os.getuid(), postgres_gid=os.getgid()
                ),
            )
            monkeypatch.setattr(probe, "_query", cluster.dispatch)
            try:
                cluster.start()
                yield cluster
            finally:
                cluster.stop()

    return create


@pytest.mark.parametrize("profile", ["ubuntu", "installed"])
def test_real_pg16_accepts_only_supported_profile_and_preserves_files(postgresql, profile):
    with postgresql(profile=profile) as cluster:
        # Warm the first backend cache before comparing configuration files.
        assert cluster.sql("SELECT 1").returncode == 0
        before = probe._pg_snapshot(os.getuid())
        all_files = cluster.file_observation()
        assert cluster.ready(), [item.stderr for item in cluster.records]
        assert probe._pg_snapshot(os.getuid()) == before
        assert cluster.file_observation() == all_files
        flags = cluster.sql("SELECT current_setting('transaction_read_only')")
        assert flags.returncode == 0 and flags.stdout.strip() == "on"


@pytest.mark.parametrize("listen", ["*", "0.0.0.0", "127.0.0.1"])
def test_real_pg16_rejects_unapproved_active_listeners(postgresql, listen):
    with postgresql(listen=listen) as cluster:
        cluster.assert_refused()


@pytest.mark.parametrize("option", ["wrong_socket", "wrong_port", "wrong_data"])
def test_real_pg16_rejects_wrong_active_location(postgresql, option):
    with postgresql(**{option: True}) as cluster:
        cluster.assert_refused()


@pytest.mark.parametrize("position", ["before", "after"])
def test_real_pg16_rejects_trust_beside_valid_peer(postgresql, position):
    with postgresql() as cluster:
        original = cluster.hba.read_text()
        trust = "local forbidden_db all trust\n"
        cluster.hba.write_text(trust + original if position == "before" else original + trust)
        cluster.assert_refused()


def test_real_pg16_rejects_malformed_and_unavailable_hba(postgresql):
    with postgresql() as cluster:
        cluster.hba.write_text(cluster.hba.read_text() + "invalid_hba_line\n")
        cluster.assert_refused()
    with postgresql() as cluster:
        changed = cluster.sql(
            "ALTER VIEW pg_catalog.pg_hba_file_rules RENAME TO stage9_fixture_hidden_hba_rules",
            readonly=False,
        )
        assert changed.returncode == 0, changed.stderr
        cluster.assert_refused()
        assert any(item.returncode != 0 for item in cluster.records)


def test_real_pg16_rejects_stale_disk_auth_despite_accepted_disk_peer(postgresql):
    with postgresql() as cluster:
        # Only the disposable fixture is restarted. Live HBA becomes trust, then
        # disk is restored without reload: the file view alone looks acceptable.
        original = cluster.hba.read_text()
        cluster.stop()
        cluster.hba.write_text(original.replace(" peer", " trust"))
        cluster.start()
        cluster.hba.write_text(original)
        auth = cluster.sql("SELECT system_user")
        assert auth.returncode == 0 and auth.stdout.strip() == ""
        cluster.assert_refused()


def test_real_pg16_rejects_disk_active_setting_drift(postgresql):
    with postgresql() as cluster:
        cluster.config.write_text(
            cluster.config.read_text().replace(
                "listen_addresses = 'localhost'", "listen_addresses = '*'"
            )
        )
        cluster.assert_refused()


def test_real_pg16_rejects_read_write_session(postgresql, monkeypatch):
    with postgresql() as cluster:
        original = cluster.sql
        monkeypatch.setattr(cluster, "sql", lambda statement: original(statement, readonly=False))
        cluster.assert_refused()


def test_real_pg16_rejects_removed_but_still_loaded_extra_trust(postgresql):
    with postgresql() as cluster:
        original = cluster.hba.read_text()
        cluster.stop()
        cluster.hba.write_text("local foreign_database all trust\n" + original)
        cluster.start()
        cluster.hba.write_text(original)  # Deliberately do not reload the fixture.
        auth = cluster.sql("SELECT system_user")
        assert auth.returncode == 0 and auth.stdout.strip() == "peer:" + cluster.principal
        disk = cluster.dispatch(probe.POSTGRESQL_ROOT_CONFIG_QUERY)
        assert json.loads(disk)["ready"] is True
        # The disk views and current peer identity cannot attest other loaded
        # rules. File ctime newer than postmaster startup closes that gap.
        cluster.assert_refused(sql_denied=False)


def test_real_pg16_requires_full_fixture_restart_after_even_valid_reload(postgresql):
    with postgresql() as cluster:
        cluster.config.write_text(cluster.config.read_text() + "# harmless fixture comment\n")
        cluster.command("pg_ctl", "-D", str(cluster.data), "reload")
        assert json.loads(cluster.dispatch(probe.POSTGRESQL_ROOT_CONFIG_QUERY))["ready"] is True
        cluster.assert_refused(sql_denied=False)
        cluster.stop()
        cluster.start()
        assert cluster.ready()


@pytest.mark.parametrize("profile", ["ubuntu", "installed"])
def test_real_pg16_rejects_unapproved_ident_mapping(postgresql, profile):
    with postgresql(profile=profile) as cluster:
        cluster.ident.write_text(cluster.ident.read_text() + "foreign_map foreign_os foreign_sql\n")
        cluster.assert_refused()


def test_real_pg16_rejects_empty_hba_include_without_emitted_foreign_rows(postgresql):
    with postgresql() as cluster:
        cluster.stop()
        (cluster.hba.parent / "auth-extra.conf").write_text("# empty included file\n")
        cluster.hba.write_text(cluster.hba.read_text() + "include auth-extra.conf\n")
        cluster.start()
        assert json.loads(cluster.dispatch(probe.POSTGRESQL_ROOT_CONFIG_QUERY))["ready"] is True
        # An empty include emits no view rows, so file_name checks alone cannot
        # establish a protected HBA closure. The static profile must deny it.
        cluster.assert_refused(sql_denied=False)


@pytest.mark.parametrize("setting", ["fsync", "synchronous_commit"])
def test_real_pg16_rejects_disabled_durability(postgresql, setting):
    with postgresql(**{setting: "off"}) as cluster:
        cluster.assert_refused()


def test_real_pg16_rejects_auto_conf_outside_protected_source_closure(postgresql):
    with postgresql() as cluster:
        (cluster.data / "postgresql.auto.conf").write_text("work_mem = '16MB'\n")
        cluster.assert_refused()


def test_real_pg16_rejects_empty_auto_include_without_emitted_foreign_rows(postgresql):
    with postgresql() as cluster:
        cluster.stop()
        (cluster.data / "empty-auto.conf").write_text("# no GUC rows\n")
        (cluster.data / "postgresql.auto.conf").write_text("include = 'empty-auto.conf'\n")
        cluster.start()
        assert json.loads(cluster.dispatch(probe.POSTGRESQL_ROOT_CONFIG_QUERY))["ready"] is True
        # The view omits an empty included file. Direct protected reading of
        # auto.conf must still forbid the dependency, even before postmaster start.
        cluster.assert_refused(sql_denied=False)


@pytest.mark.parametrize("method", ["md5", "password"])
def test_real_pg16_rejects_other_unapproved_hba_methods(postgresql, method):
    with postgresql() as cluster:
        cluster.hba.write_text(cluster.hba.read_text().replace("scram-sha-256", method))
        cluster.assert_refused()
