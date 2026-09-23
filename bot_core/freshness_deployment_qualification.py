"""Binary, fail-closed qualification of the PRODUCTION_LOCAL deployment."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess

import psycopg

# Frozen deployment coordinates are repeated here deliberately: qualification must be
# importable on hosts that cannot import the POSIX-only verifier executable.
POSTGRES_SOCKET_DIRECTORY = "/run/postgresql"
POSTGRES_PORT = 5432
POSTGRES_DATABASE = "freshness_gate"
VERIFIER_SOCKET_PATH = "/run/cryptohunter/freshness/verifier.sock"
VERIFIER_USER = "os_freshness_crypto_verifier"
RUNTIME_USER = "os_freshness_runtime"
IPC_GROUP = "freshness_verifier_ipc"
from bot_core.postgresql_freshness_authority import (
    PostgreSQLConnectionConfig,
    qualify_postgresql_freshness_authority,
)

UNIT_NAME = "cryptohunter-freshness-verifier.service"
UNIT_PATH = Path("/etc/systemd/system") / UNIT_NAME
REVIEWED_UNIT = Path(__file__).resolve().parents[1] / "deployment/systemd" / UNIT_NAME
REVIEWED_DEPLOYMENT = Path(__file__).resolve().parents[1] / "deployment/postgresql"
EXPECTED_HBA = [
    ("local", "all", "postgres", "peer", "map=freshness_admin_map"),
    (
        "local",
        POSTGRES_DATABASE,
        "freshness_crypto_verifier",
        "peer",
        "map=freshness_authority_map",
    ),
    ("local", POSTGRES_DATABASE, "freshness_runtime", "peer", "map=freshness_authority_map"),
    ("local", POSTGRES_DATABASE, "all", "reject", None),
    ("local", "all", "all", "reject", None),
]
EXPECTED_IDENT = [
    ("freshness_authority_map", VERIFIER_USER, "freshness_crypto_verifier"),
    ("freshness_authority_map", RUNTIME_USER, "freshness_runtime"),
    ("freshness_admin_map", "root", "postgres"),
]
REVIEWED_CODE_DIGESTS = {
    "freshness_verifier_service.py": "88da8c2a19ac64527939a9aadd914de518bb0e6f2c84d0f85ddd1d85d0216c92",
    "freshness_semantic_verifier.py": "5cee08b44d446e9388d7c0e7834a54c008d7867f6289a943770602f807798d48",
    "account_genesis_freshness_authority.py": "cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b",
}


class DeploymentQualificationError(RuntimeError):
    """The live host is not the exact reviewed deployment."""


def _fail(condition: bool, message: str) -> None:
    if not condition:
        raise DeploymentQualificationError(message)


def _mode(path: Path, *, follow: bool = False) -> os.stat_result:
    try:
        return path.stat(follow_symlinks=follow)
    except OSError as exc:
        raise DeploymentQualificationError(f"missing or inaccessible path: {path}") from exc


def _immutable_file(
    path: Path, expected: bytes, online_uids: set[int], online_gids: set[int]
) -> None:
    status = _mode(path)
    _fail(stat.S_ISREG(status.st_mode) and not path.is_symlink(), f"not a real file: {path}")
    _fail(path.read_bytes() == expected, f"reviewed content mismatch: {path}")
    _fail(status.st_uid not in online_uids, f"online principal owns: {path}")
    writable = bool(status.st_mode & stat.S_IWOTH) or (
        status.st_gid in online_gids and bool(status.st_mode & stat.S_IWGRP)
    )
    _fail(not writable, f"online principal can write: {path}")


def _systemctl(*arguments: str) -> str:
    result = subprocess.run(["systemctl", *arguments], text=True, capture_output=True, check=False)
    if result.returncode:
        raise DeploymentQualificationError("system service manager query failed")
    return result.stdout.strip()


def qualify_production_local_deployment() -> None:
    """Raise on any drift; return only after the complete live-host proof."""
    if os.name != "posix":
        raise DeploymentQualificationError(
            "PRODUCTION_LOCAL deployment qualification requires POSIX"
        )

    # Native account databases exist only on POSIX. Importing them here keeps the
    # qualification contracts inspectable during cross-platform test collection.
    import grp
    import pwd

    verifier = pwd.getpwnam(VERIFIER_USER)
    runtime = pwd.getpwnam(RUNTIME_USER)
    ipc = grp.getgrnam(IPC_GROUP)
    _fail(verifier.pw_uid != runtime.pw_uid, "online UIDs are not distinct")
    _fail(verifier.pw_uid != 0 and runtime.pw_uid != 0, "online UID is root")
    _fail(
        VERIFIER_USER in ipc.gr_mem and RUNTIME_USER in ipc.gr_mem,
        "exact IPC group membership missing",
    )
    online_uids = {verifier.pw_uid, runtime.pw_uid}
    online_gids = {
        g.gr_gid for g in grp.getgrall() if VERIFIER_USER in g.gr_mem or RUNTIME_USER in g.gr_mem
    }
    online_gids.update({verifier.pw_gid, runtime.pw_gid})

    _immutable_file(UNIT_PATH, REVIEWED_UNIT.read_bytes(), online_uids, online_gids)
    for filename, installed in (
        ("pg_hba.conf", Path("/etc/cryptohunter/postgresql/pg_hba.conf")),
        ("pg_ident.conf", Path("/etc/cryptohunter/postgresql/pg_ident.conf")),
        (
            "production-local.conf",
            Path("/etc/postgresql/16/main/conf.d/cryptohunter-production-local.conf"),
        ),
    ):
        _immutable_file(
            installed, (REVIEWED_DEPLOYMENT / filename).read_bytes(), online_uids, online_gids
        )
    executable = Path("/usr/bin/python3")
    executable_status = _mode(executable, follow=True)
    _fail(stat.S_ISREG(executable_status.st_mode), "reviewed Python executable missing")
    code_directory = Path("/usr/lib/cryptohunter/bot_core")
    for filename, digest in REVIEWED_CODE_DIGESTS.items():
        source = code_directory / filename
        status = _mode(source)
        _fail(status.st_uid == 0 and not source.is_symlink(), f"unsafe source owner: {source}")
        _fail(
            not status.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
            f"source is group/world writable: {source}",
        )
        _fail(
            hashlib.sha256(source.read_bytes()).hexdigest() == digest,
            f"reviewed source digest mismatch: {source}",
        )
    shown = dict(
        line.split("=", 1)
        for line in _systemctl(
            "show", UNIT_NAME, "--property=User,Group,ExecStart,Environment,ActiveState"
        ).splitlines()
    )
    _fail(shown.get("User") == VERIFIER_USER, "wrong effective service User")
    _fail(shown.get("Group") == IPC_GROUP, "wrong effective service Group")
    _fail(shown.get("ActiveState") == "active", "verifier service is not active")
    _fail("freshness_verifier_service" in shown.get("ExecStart", ""), "wrong executable")
    _fail(shown.get("Environment") == "PYTHONNOUSERSITE=1", "unexpected environment")

    directory = Path(VERIFIER_SOCKET_PATH).parent
    directory_status = _mode(directory)
    _fail(
        not directory.is_symlink() and stat.S_ISDIR(directory_status.st_mode),
        "runtime directory substitution",
    )
    _fail(
        (directory_status.st_uid, directory_status.st_gid, stat.S_IMODE(directory_status.st_mode))
        == (verifier.pw_uid, ipc.gr_gid, 0o750),
        "runtime directory metadata",
    )
    socket_status = _mode(Path(VERIFIER_SOCKET_PATH))
    _fail(
        not Path(VERIFIER_SOCKET_PATH).is_symlink() and stat.S_ISSOCK(socket_status.st_mode),
        "verifier path is not a real Unix socket",
    )
    _fail(
        (socket_status.st_uid, socket_status.st_gid, stat.S_IMODE(socket_status.st_mode))
        == (verifier.pw_uid, ipc.gr_gid, 0o660),
        "verifier socket metadata",
    )

    dsn = (
        f"host={POSTGRES_SOCKET_DIRECTORY} port={POSTGRES_PORT} "
        f"dbname={POSTGRES_DATABASE} user=postgres sslmode=disable"
    )
    with psycopg.connect(dsn) as connection:
        settings = dict(
            connection.execute(
                "SELECT name,setting FROM pg_settings WHERE name=ANY(%s)",
                (["server_version_num", "listen_addresses", "fsync", "synchronous_commit"],),
            ).fetchall()
        )
        _fail(int(settings.get("server_version_num", "0")) >= 160000, "PostgreSQL <16")
        _fail(settings.get("listen_addresses") == "", "PostgreSQL TCP listener enabled")
        _fail(settings.get("fsync") == "on", "fsync is not on")
        _fail(settings.get("synchronous_commit") == "on", "synchronous_commit is not on")
        facts = connection.execute(
            "SELECT inet_client_addr(),inet_server_addr(),"
            "coalesce((SELECT ssl FROM pg_stat_ssl WHERE pid=pg_backend_pid()),false)"
        ).fetchone()
        _fail(facts == (None, None, False), "admin qualification used non-local transport")
        hba = connection.execute(
            "SELECT type,array_to_string(database,','),array_to_string(user_name,','),"
            "auth_method,options[1] FROM pg_hba_file_rules ORDER BY line_number"
        ).fetchall()
        _fail(hba == EXPECTED_HBA, "effective HBA rules differ")
        ident = connection.execute(
            "SELECT map_name,sys_name,pg_username FROM pg_ident_file_mappings ORDER BY line_number"
        ).fetchall()
        _fail(ident == EXPECTED_IDENT, "effective pg_ident mappings differ")
    qualify_postgresql_freshness_authority(PostgreSQLConnectionConfig(dsn))


def main() -> int:
    try:
        qualify_production_local_deployment()
    except Exception:
        print("FAIL CLOSED")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
