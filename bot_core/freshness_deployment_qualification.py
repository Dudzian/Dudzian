"""Binary, fail-closed qualification of the PRODUCTION_LOCAL deployment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any

import psycopg

from bot_core.freshness_service_identity import (
    ServiceIdentityError,
    _load_runtime,
    render_runtime_unit,
    validate_runtime_unit,
    qualify_freshness_service_identity,
)

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
    "linux_service_ipc.py": "099e6a3a08717b48e7a01606b9908f56f0eec40aced260d554d7dfc4562b451e",
    "linux_service_postgresql.py": "d09d359535e18408d8cee282cb757c4eca8a4e8c422bfce8fa804860ad59f0d1",
    "production_signer_ipc.py": "be755e67cb1ea4466c074e4c2c0dbe8e6528552f4a70fd7769a5e876c878f552",
    "freshness_verifier_service.py": "df23a3c2fcd1e04877192fe2fe376909a29aa5183a516485fbfdad5a72f9f057",
    "freshness_semantic_verifier.py": "64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1",
    "account_genesis_freshness_authority.py": "cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b",
}


class DeploymentQualificationError(RuntimeError):
    """The live host is not the exact reviewed deployment."""


SUBSTRATE_PROFILE = "SIGNED_FRESHNESS_SUBSTRATE"
FULL_PROFILE = "PRODUCTION_LOCAL"


def _load_application() -> dict[str, Any]:
    # This verifier imports no AccountGenesis authority and grants no live-host
    # or product qualification. Its independent signed inventory covers bytes.
    from deployment.freshness_application import verify_installed_application

    return verify_installed_application()


def qualify_signed_freshness_substrate() -> dict[str, Any]:
    """Verify signed installed bytes only; never return a production proof."""
    _fail(sys.platform == "linux", "signed Freshness substrate requires native Linux")
    try:
        result = _load_application()
    except (RuntimeError, OSError, ValueError) as exc:
        raise DeploymentQualificationError("signed application closure not qualified") from exc
    _fail(
        result.get("qualification_profile") == SUBSTRATE_PROFILE
        and result.get("scope") == "cryptohunter.freshness.substrate.v2"
        and result.get("production_qualification") == "WITHHELD",
        "ambiguous signed substrate qualification scope",
    )
    return result


def _qualify_reviewed_sources(code_directory: Path) -> None:
    """Full-product pins, including the independently frozen AccountGenesis."""
    for filename, digest in REVIEWED_CODE_DIGESTS.items():
        source = code_directory / filename
        status = _mode(source)
        _fail(
            status.st_uid == 0 and stat.S_ISREG(status.st_mode) and not source.is_symlink(),
            f"unsafe source owner: {source}",
        )
        _fail(
            not status.st_mode & (stat.S_IWGRP | stat.S_IWOTH),
            f"source is group/world writable: {source}",
        )
        _fail(
            hashlib.sha256(source.read_bytes()).hexdigest() == digest,
            f"reviewed source digest mismatch: {source}",
        )


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
    if sys.platform == "win32":
        raise DeploymentQualificationError(
            "PRODUCTION_LOCAL deployment qualification requires native Linux"
        )
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
    _qualify_reviewed_sources(Path("/usr/lib/cryptohunter/bot_core"))
    qualify_signed_freshness_substrate()
    try:
        approved_runtime = _load_runtime()
        reviewed_unit = REVIEWED_UNIT.read_bytes()
        if b"@FRESHNESS_RUNTIME_PYTHON@" in reviewed_unit:
            reviewed_unit = render_runtime_unit(reviewed_unit, approved_runtime)
        _immutable_file(
            UNIT_PATH,
            validate_runtime_unit(reviewed_unit, approved_runtime),
            online_uids,
            online_gids,
        )
        qualify_freshness_service_identity(uid=verifier.pw_uid, gid=ipc.gr_gid)
    except ServiceIdentityError as exc:
        raise DeploymentQualificationError("live verifier identity not qualified") from exc

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
        settings: dict[str, str] = dict(
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=(FULL_PROFILE, SUBSTRATE_PROFILE), default=FULL_PROFILE
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.profile == SUBSTRATE_PROFILE:
            print(json.dumps(qualify_signed_freshness_substrate(), sort_keys=True))
            return 0
        qualify_production_local_deployment()
    except Exception:
        print("FAIL CLOSED")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
