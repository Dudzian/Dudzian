"""Bootstrap security models; the opt-in test below uses an isolated real PG."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import os
import secrets
import socket
import subprocess
import sys
import time

import psycopg
from psycopg.conninfo import conninfo_to_dict, make_conninfo
import pytest

from deployment.windows_installer import provision


def _posix_bootstrap_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adapt only Win32 credential creation for the unprivileged Linux fixture."""
    monkeypatch.setattr(provision, "_require_bootstrap_system", lambda: None)

    def directory(security):
        path = security / (".postgres-bootstrap-" + secrets.token_hex(16))
        path.mkdir(mode=0o700)
        return path

    def write(path, value):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="ascii") as stream:
            stream.write(value)

    monkeypatch.setattr(provision, "_create_bootstrap_directory", directory)
    monkeypatch.setattr(provision, "_write_bootstrap_secret", write)
    monkeypatch.setattr(provision, "_qualify_bootstrap_directory", lambda _path: None)


@pytest.mark.parametrize(
    "failure_stage",
    (
        None,
        "POSTGRES_BOOTSTRAP_CREDENTIALS",
        "POSTGRES_INITDB",
        "POSTGRES_BOOTSTRAP_START",
        "POSTGRES_CREATE_DATABASE",
        "POSTGRES_PROVISION_FRESHNESS_AUTHORITY",
        "POSTGRES_QUALIFY_FRESHNESS_AUTHORITY",
        "POSTGRES_FINAL_HBA_WRITE",
        "POSTGRES_FINAL_HBA_QUALIFY",
        "POSTGRES_FINAL_ADMIN_REJECTION",
        "POSTGRES_BOOTSTRAP_STOP",
    ),
)
def test_bootstrap_model_requires_preestablished_credential_and_always_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None, capsys, caplog
) -> None:
    _posix_bootstrap_adapter(monkeypatch)
    monkeypatch.setenv("PGPASSWORD", "synthetic-unrelated-machine-credential")
    data, security = tmp_path / "data", tmp_path / "security"
    data.mkdir()
    security.mkdir()
    (data / "postgresql.conf").write_text("")
    protected: set[Path] = set()
    qualified: set[Path] = set()
    password: str | None = None
    password_path: Path | None = None
    commands: list[list[str]] = []
    authorized: list[str] = []
    denied: list[str] = []

    def protect(path, grants, *, inherit):
        assert grants == {provision.SYSTEM_SID: provision.FILE_ALL_ACCESS}
        assert inherit == path.is_dir()
        protected.add(path)

    def qualify(path, grants, *, inherit):
        assert path in protected
        qualified.add(path)

    original_secret_write = provision._write_bootstrap_secret

    def write_secret(path, value):
        assert path.parent in qualified  # no secret is written before ACL qualification
        original_secret_write(path, value)
        protected.add(path)
        qualified.add(path)

    def run(command, **kwargs):
        nonlocal password, password_path
        assert kwargs["stdout"] is subprocess.DEVNULL
        assert kwargs["stderr"] is subprocess.DEVNULL
        assert "env" not in kwargs and "input" not in kwargs
        commands.append(command)
        if command[0].endswith("initdb.exe"):
            assert "--auth-host=scram-sha-256" in command
            argument = next(value for value in command if value.startswith("--pwfile="))
            password_path = Path(argument.split("=", 1)[1])
            assert password_path in qualified
            password = password_path.read_text().strip()
            assert len(password) >= 48
            assert password not in repr(command) + repr(kwargs)
        elif command[1] == "start":
            assert password is not None
            assert password_path is not None and not password_path.exists()
            hba = (data / "pg_hba.conf").read_text()
            assert hba.startswith("hostssl all postgres 127.0.0.1/32 scram-sha-256\n")
            assert "trust" not in hba
            assert all(password not in arg for previous in commands for arg in previous)
            with pytest.raises(psycopg.OperationalError):
                connect("host=127.0.0.1 user=postgres dbname=postgres")

    class Connection:
        statement = ""

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def execute(self, statement):
            self.statement = statement
            return self

        def fetchone(self):
            return (1,) if self.statement == "SELECT 1" else ("postgres", "postgres")

        def fetchall(self):
            return []

    def connect(dsn, **_kwargs):
        options = conninfo_to_dict(dsn)
        passfile = options.get("passfile")
        if (data / "pg_hba.conf").read_text().splitlines() == list(provision.FINAL_HBA):
            denied.append("final")
            raise psycopg.OperationalError("administrator rejected")
        if passfile is None:
            denied.append("bootstrap")
            raise psycopg.OperationalError("bootstrap credential required")
        assert password is not None
        path = Path(passfile)
        assert path in qualified
        assert (
            path.read_text()
            == f"127.0.0.1:{provision.CONTRACT.postgresql_port}:*:postgres:{password}\n"
        )
        assert options["sslmode"] == "verify-full"
        assert options["password"] == ""
        assert options["sslrootcert"] == (security / "ca.crt").as_posix()
        assert options["connect_timeout"] == "5"
        assert password not in dsn
        authorized.append(options["dbname"])
        return Connection()

    def authority(config):
        with connect(config.dsn):
            pass

    def stage(value):
        if value == failure_stage:
            raise RuntimeError("injected bootstrap failure")

    original_create = provision._create_bootstrap_directory

    def create(security):
        path = original_create(security)
        protect(path, {provision.SYSTEM_SID: provision.FILE_ALL_ACCESS}, inherit=True)
        return path

    monkeypatch.setattr(provision, "_create_bootstrap_directory", create)
    monkeypatch.setattr(provision, "_protect", protect)
    monkeypatch.setattr(
        provision,
        "_qualify_bootstrap_directory",
        lambda path: qualify(path, {provision.SYSTEM_SID: provision.FILE_ALL_ACCESS}, inherit=True),
    )
    monkeypatch.setattr(provision, "_write_bootstrap_secret", write_secret)
    monkeypatch.setattr(provision.subprocess, "run", run)
    monkeypatch.setattr(provision, "provision_postgresql_freshness_authority", authority)
    monkeypatch.setattr(provision, "qualify_postgresql_freshness_authority", authority)
    monkeypatch.setattr(provision, "qualify_hba_rows", lambda *_args: None)
    monkeypatch.setitem(
        sys.modules,
        "psycopg",
        SimpleNamespace(connect=connect, OperationalError=psycopg.OperationalError),
    )

    if failure_stage is None:
        provision._configure_database(tmp_path, data, security, stage)
        assert authorized == ["postgres", "freshness_gate", "freshness_gate", "postgres"]
        assert denied == ["bootstrap", "final"]
        assert (data / "pg_hba.conf").read_text().splitlines() == list(provision.FINAL_HBA)
    else:
        with pytest.raises(RuntimeError, match="injected bootstrap failure"):
            provision._configure_database(tmp_path, data, security, stage)
    assert list(security.iterdir()) == []
    if password is not None:
        captured = capsys.readouterr()
        assert password not in captured.out + captured.err + caplog.text
        assert all(password not in value for value in os.environ.values())
    if any(command[1] == "start" for command in commands):
        assert commands[-1][1] == "stop"


@pytest.mark.parametrize("explicit_empty_password", (False, True))
def test_real_libpq_password_selection_requires_explicit_empty_with_passfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, explicit_empty_password: bool
) -> None:
    """Inspect real libpq selection before polling; no server or auth is needed."""
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    # The colon guarantees the ambient value differs from the URL-safe file password.
    ambient = secrets.token_urlsafe(48) + ":"
    file_password = secrets.token_urlsafe(48)
    monkeypatch.setenv("PGPASSWORD", ambient)
    passfile = tmp_path / "pgpass"
    passfile.write_text(f"127.0.0.1:1:postgres:postgres:{file_password}\n", encoding="ascii")
    passfile.chmod(0o600)
    options = {
        "host": "127.0.0.1",
        "hostaddr": "127.0.0.1",
        "port": "1",
        "dbname": "postgres",
        "user": "postgres",
        "passfile": str(passfile),
        "sslmode": "verify-full",
        # Even an unexpected local listener cannot authenticate without this CA.
        "sslrootcert": str(tmp_path / "absent-ca.crt"),
        "connect_timeout": "1",
    }
    if explicit_empty_password:
        options["password"] = ""
    connection = psycopg.pq.PGconn.connect_start(make_conninfo(**options).encode())
    try:
        selected_file = connection.password == file_password.encode()
        selected_ambient = connection.password == ambient.encode()
        assert selected_file is explicit_empty_password
        assert selected_ambient is (not explicit_empty_password)
    finally:
        connection.finish()


@pytest.mark.parametrize("fail_qualification", (False, True))
def test_bootstrap_acl_failure_prevents_any_credential_or_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fail_qualification: bool
) -> None:
    _posix_bootstrap_adapter(monkeypatch)
    security = tmp_path / "security"
    security.mkdir()

    def fail(*_args, **_kwargs):
        if fail_qualification:
            assert len(list(security.iterdir())) == 1
            assert not list(next(security.iterdir()).iterdir())
        else:
            assert not list(security.iterdir())
        raise OSError("controlled ACL failure")

    monkeypatch.setattr(provision, "_protect", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(provision, "_qualify_protected", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        provision,
        "_qualify_bootstrap_directory" if fail_qualification else "_create_bootstrap_directory",
        fail,
    )
    monkeypatch.setattr(
        provision.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("process before ACL")
    )
    with pytest.raises(OSError, match="controlled ACL failure"):
        provision._configure_database(tmp_path, tmp_path / "data", security)
    assert list(security.iterdir()) == []


@pytest.mark.parametrize("has_primary_failure", (False, True))
def test_bootstrap_credential_cleanup_failure_preserves_primary_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_primary_failure: bool
) -> None:
    _posix_bootstrap_adapter(monkeypatch)
    monkeypatch.setattr(provision, "_protect", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(provision, "_qualify_protected", lambda *_args, **_kwargs: None)
    real_rmtree = provision.shutil.rmtree
    primary = RuntimeError("controlled provisioning failure")

    def fail_cleanup(_path):
        raise OSError("private cleanup diagnostic")

    monkeypatch.setattr(provision.shutil, "rmtree", fail_cleanup)
    try:
        expected = RuntimeError if has_primary_failure else OSError
        with pytest.raises(expected) as captured:
            with provision._bootstrap_password_files(tmp_path):
                if has_primary_failure:
                    raise primary
        if has_primary_failure:
            assert captured.value is primary
            assert primary.__notes__ == ["bootstrap credential cleanup failed: OSError"]
    finally:
        for path in tmp_path.iterdir():
            real_rmtree(path)


@pytest.mark.skipif(
    sys.platform != "linux"
    or os.environ.get("CRYPTOHUNTER_RUN_WINDOWS_BOOTSTRAP_PG_INTEGRATION") != "1",
    reason="isolated real PostgreSQL bootstrap requires explicit Linux opt-in and PG binaries",
)
@pytest.mark.parametrize("failure_stage", (None, "POSTGRES_FINAL_HBA_WRITE"))
def test_isolated_native_postgresql_bootstrap_denies_independent_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_stage: str | None
) -> None:
    """Linux PG transport/schema test; it does not qualify Windows ACLs or SCM."""
    assert os.getuid() != 0
    _posix_bootstrap_adapter(monkeypatch)
    generated_secrets: list[str] = []
    original_password = provision.secrets.token_urlsafe

    def capture_password(size):
        value = original_password(size)
        generated_secrets.append(value)
        return value

    monkeypatch.setattr(provision.secrets, "token_urlsafe", capture_password)
    bindir = Path(os.environ["CRYPTOHUNTER_TEST_PG_BINDIR"]).resolve(strict=True)
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    monkeypatch.setattr(provision, "CONTRACT", replace(provision.CONTRACT, postgresql_port=port))
    for name in tuple(os.environ):
        if name.startswith("PG"):
            monkeypatch.delenv(name)
    data, security = tmp_path / "data", tmp_path / "security"
    security.mkdir(mode=0o700)

    # The Windows ACL boundary is modeled by owner-only POSIX permissions here.
    def protect(path, _grants, *, inherit):
        path.chmod(0o700 if inherit else 0o600)

    monkeypatch.setattr(provision, "_protect", protect)
    monkeypatch.setattr(provision, "_qualify_protected", lambda *_args, **_kwargs: None)
    provision._pki(security, {name: name for name in provision.SERVICES})
    real_run = subprocess.run

    def run(command, **kwargs):
        translated = [str(bindir / Path(command[0]).stem), *command[1:]]
        if translated[0].endswith("pg_ctl") and translated[1] == "start":
            # Keep the native fixture's sockets and logs within its own tempdir.
            with (data / "postgresql.conf").open("a") as stream:
                stream.write(f"\nunix_socket_directories='{tmp_path}'\n")
            translated.extend(["-l", str(tmp_path / "postgresql.log")])
        # Production discards child output; the fixture independently keeps its
        # PostgreSQL server log for transport assertions below.
        return real_run(translated, **kwargs)

    monkeypatch.setattr(provision.subprocess, "run", run)
    original_rejection = provision._prove_final_admin_rejection

    def rejection(provider, dsn, connection):
        # Windows reloads HBA for new sessions; Linux needs a SIGHUP first.
        before = connection.execute("SELECT pg_conf_load_time()").fetchone()[0]
        connection.execute("SELECT pg_reload_conf()")
        deadline = time.monotonic() + 5
        while connection.execute("SELECT pg_conf_load_time()").fetchone()[0] == before:
            assert time.monotonic() < deadline
            time.sleep(0.02)
        original_rejection(provider, dsn, connection)

    monkeypatch.setattr(provision, "_prove_final_admin_rejection", rejection)
    stages: list[str] = []

    def observe(stage):
        stages.append(stage)
        if stage == "POSTGRES_BOOTSTRAP_START":
            assert not list(security.glob(".postgres-bootstrap-*/initdb.password"))
        if stage == "POSTGRES_CREATE_DATABASE":
            passfile = next(security.glob(".postgres-bootstrap-*/pgpass"))
            dsn = (
                f"host=127.0.0.1 port={port} dbname=postgres user=postgres "
                f"passfile={passfile} sslmode=verify-full "
                f"sslrootcert={security / 'ca.crt'} connect_timeout=3"
            )
            with psycopg.connect(dsn, autocommit=True) as connection:
                ssl, version = connection.execute(
                    "SELECT ssl,version FROM pg_stat_ssl WHERE pid=pg_backend_pid()"
                ).fetchone()
                assert ssl is True and version in {"TLSv1.2", "TLSv1.3"}
                verifier = connection.execute(
                    "SELECT passwd FROM pg_shadow WHERE usename='postgres'"
                ).fetchone()[0]
                assert verifier.startswith("SCRAM-SHA-256$")
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(dsn.replace("sslmode=verify-full", "sslmode=disable"))
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(
                    dsn.replace(str(security / "ca.crt"), str(security / "runtime/client.crt"))
                )
            # A separate psql process has no access to the provisioning passfile.
            result = real_run(
                [
                    str(bindir / "psql"),
                    "-X",
                    "-w",
                    "-d",
                    f"host=127.0.0.1 port={port} dbname=postgres user=postgres "
                    "sslmode=require passfile=/dev/null connect_timeout=3",
                    "-c",
                    "SELECT session_user",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert result.returncode != 0
            assert "no password supplied" in result.stderr
            wrong_password = real_run(
                result.args,
                env={**os.environ, "PGPASSWORD": secrets.token_urlsafe(48)},
                capture_output=True,
                text=True,
                timeout=10,
            )
            assert wrong_password.returncode != 0
            assert "password authentication failed" in wrong_password.stderr
        if stage == failure_stage:
            raise RuntimeError("injected native bootstrap failure")

    if failure_stage is None:
        provision._configure_database(tmp_path, data, security, observe)
        assert "POSTGRES_FINAL_ADMIN_REJECTION" in stages
        assert (data / "pg_hba.conf").read_text().splitlines() == list(provision.FINAL_HBA)
    else:
        with pytest.raises(RuntimeError, match="injected native bootstrap failure"):
            provision._configure_database(tmp_path, data, security, observe)
    assert "POSTGRES_QUALIFY_FRESHNESS_AUTHORITY" in stages
    assert not list(security.glob(".postgres-bootstrap-*"))
    assert not (data / "postmaster.pid").exists()
    server_log = (tmp_path / "postgresql.log").read_text()
    assert generated_secrets
    assert all(value not in server_log for value in generated_secrets)
