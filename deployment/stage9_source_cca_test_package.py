"""Offline, unprivileged TEST ONLY fixtures for physical installation comparison.

The separate inventory digest is a test package binding, never independent
production approval. Validation returns a retained byte snapshot for installation;
callers must not reopen user-controlled paths after validating it.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

from deployment.stage9_trusted_installation_contract import parse_manifest

ROLES = ("root_source", "history_source", "cca")
_SUFFIXES = ("r", "h", "c")
_SCHEMA = "CryptoHunter.SourceCCATestPackageV1"
_MAX_FILE = 8 * 1024 * 1024
_SOURCE = b"""/* Synthetic TEST ONLY: no sockets, custody, signing, database or credentials. */
#include <signal.h>
#include <unistd.h>
#ifndef CH_ROLE
#error CH_ROLE is required
#endif
const char ch_test_role[] = CH_ROLE;
static volatile sig_atomic_t stopping = 0;
static void stop(int number) { (void)number; stopping = 1; }
int main(void) {
    struct sigaction action = {0};
    sigset_t blocked, previous;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGTERM);
    sigaddset(&blocked, SIGINT);
    if (sigprocmask(SIG_BLOCK, &blocked, &previous)) return 2;
    action.sa_handler = stop;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGTERM, &action, 0) || sigaction(SIGINT, &action, 0)) return 2;
    while (!stopping) sigsuspend(&previous);
    return 0;
}
"""
_README = b"""TEST ONLY / SYNTHETIC / PRODUCTION_NO_GO
This package contains harmless signal-controlled Linux fixture processes.
The manifest and separate test pin are synthetic, not independent approval.
Run --check, review this package and record inventory.json SHA-256 separately.
Only an explicit --execute with the reviewed digest may install these fixtures.
Execution always performs owned-resource cleanup; afterwards run --inspect-cleanup
for the exact run ID and compare its independently observed, scoped report.
Rootless observation never proves full CLEAN. Full absence requires the explicit,
read-only --inspect-cleanup-admin mode invoked separately as root.
Unknown or replaced resources must be retained for operator review, not deleted.
Never run physical --execute in CI or Codex. See the repository operator runbook.
Stage 9 remains IN_PROGRESS / PRODUCTION_NO_GO even when checks MATCHED.
"""


class HarnessRefusal(ValueError):
    """Only a fixed diagnostic, never native error or caller path text."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _effective_uid() -> int:
    backend = getattr(os, "geteuid", None)
    if not callable(backend):
        raise HarnessRefusal("NATIVE_LINUX_REQUIRED")
    value = backend()
    if type(value) is not int or value < 0:
        raise HarnessRefusal("PACKAGE_METADATA_UNAVAILABLE")
    return value


def plan(run_id: str, uid_base: int = 62000) -> dict[str, Any]:
    """Derive every resource from a bounded, isolated one-run test namespace."""
    if type(run_id) is not str or re.fullmatch(r"[0-9a-f]{12}", run_id) is None:
        raise HarnessRefusal("RUN_ID_INVALID")
    if type(uid_base) is not int or not 60000 <= uid_base <= 64997:
        raise HarnessRefusal("UID_RANGE_INVALID")
    root = f"/run/ch-s9-test-{run_id}"
    roles = []
    for index, (role, suffix) in enumerate(zip(ROLES, _SUFFIXES, strict=True)):
        unit = f"ch-s9t-{run_id}-{suffix}.service"
        roles.append(
            {
                "role": role,
                "principal": f"chs9t_{run_id}_{suffix}",
                "unit": unit,
                "uid": uid_base + index,
                "gid": uid_base + index,
                "executable": f"{root}/{role}",
                "unit_path": f"/run/systemd/system/{unit}",
                "control_group": f"/system.slice/{unit}",
            }
        )
    return {"run_id": run_id, "uid_base": uid_base, "root": root, "roles": roles}


def unit_bytes(resource: dict[str, Any], run_id: str) -> bytes:
    """Canonical hardening; no operator-supplied unit directives are accepted."""
    return f"""[Unit]
Description=Stage9 SourceCCA TEST ONLY {run_id} {resource["role"]}
[Service]
Type=exec
User={resource["principal"]}
Group={resource["principal"]}
DynamicUser=no
SupplementaryGroups=
ExecStart={resource["executable"]}
NoNewPrivileges=yes
CapabilityBoundingSet=
AmbientCapabilities=
PrivateNetwork=yes
PrivateIPC=yes
PrivateMounts=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectSystem=strict
ProtectHome=yes
InaccessiblePaths=-/home -/root -/var/lib -/etc/cryptohunter -/run/cryptohunter -/opt/cryptohunter
RestrictAddressFamilies=AF_UNIX
RestrictNamespaces=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectKernelLogs=yes
ProtectControlGroups=yes
RestrictSUIDSGID=yes
LockPersonality=yes
RemoveIPC=yes
UMask=0077
Restart=no
KillMode=control-group
TimeoutStartSec=15s
TimeoutStopSec=5s
RuntimeMaxSec=180s
LimitCORE=0
StandardInput=null
StandardOutput=null
StandardError=null
UnsetEnvironment=LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
""".encode("ascii")


def _payload(resources: dict[str, Any], binaries: dict[str, bytes]) -> dict[str, bytes]:
    files = {"worker.c": _SOURCE, "README.txt": _README, **binaries}
    manifest_roles = []
    for resource in resources["roles"]:
        role = resource["role"]
        unit = unit_bytes(resource, resources["run_id"])
        files[f"{role}.service"] = unit
        manifest_roles.append(
            {
                "role": role,
                "installation_identity": f"TEST_ONLY-{resources['run_id']}-{role}",
                "unit": resource["unit"],
                "principal": resource["principal"],
                "uid": resource["uid"],
                "gid": resource["gid"],
                "supplementary_groups": [],
                "unit_file": {
                    "path": resource["unit_path"],
                    "sha256": _digest(unit),
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o644,
                },
                "executable": {
                    "path": resource["executable"],
                    "sha256": _digest(binaries[role]),
                    "uid": 0,
                    "gid": 0,
                    "mode": 0o755,
                },
                "control_group": resource["control_group"],
            }
        )
    manifest = _json(
        {
            "schema": "CryptoHunter.SourceCCAInstallationV1",
            "version": 1,
            "installation_id": f"TEST_ONLY-SYNTHETIC-{resources['run_id']}",
            "roles": manifest_roles,
        }
    )
    parse_manifest(manifest)
    files["manifest.json"] = manifest
    files["TEST_ONLY-pin.json"] = _json(
        {
            "schema": "CryptoHunter.SourceCCAInstallationApprovalPinV1",
            "version": 1,
            "installation_id": f"TEST_ONLY-SYNTHETIC-{resources['run_id']}",
            "approval_id": f"TEST_ONLY-NOT-INDEPENDENT-APPROVAL-{resources['run_id']}",
            "manifest_sha256": _digest(manifest),
        }
    )
    files["resources.json"] = _json(resources)
    return files


def _commit(value: Any) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise HarnessRefusal("COMMIT_INVALID")
    return value


def _compile(source: Path, target: Path, role: str) -> None:
    try:
        result = subprocess.run(
            [
                "/usr/bin/cc",
                "-D_POSIX_C_SOURCE=200809L",
                f'-DCH_ROLE="TEST_ONLY_{role}"',
                "-std=c11",
                "-Wall",
                "-Werror",
                "-Os",
                "-Wl,--build-id=none",
                str(source),
                "-o",
                str(target),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
            check=False,
            env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
        )
    except (OSError, subprocess.SubprocessError):
        raise HarnessRefusal("OFFLINE_COMPILER_UNAVAILABLE") from None
    if result.returncode != 0:
        raise HarnessRefusal("OFFLINE_COMPILER_FAILED")
    target.chmod(0o700)


def _write_new(path: Path, raw: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(raw)


def _open_directory(path: Path) -> int:
    absolute = path.absolute()
    if any(part in {".", ".."} for part in absolute.parts):
        raise HarnessRefusal("PACKAGE_DIRECTORY_UNSAFE")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in absolute.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def build_package(path: Path, run_id: str, uid_base: int, commit: str) -> dict[str, Any]:
    """Create a new local package without privileges, network or service changes."""
    if _effective_uid() == 0:
        raise HarnessRefusal("UNPRIVILEGED_PREPARATION_REQUIRED")
    resources = plan(run_id, uid_base)
    _commit(commit)
    path = Path(os.path.abspath(path))
    try:
        parent = _open_directory(path.parent)
    except OSError:
        raise HarnessRefusal("PACKAGE_CREATION_FAILED") from None
    try:
        os.mkdir(path.name, mode=0o700, dir_fd=parent)
    except FileExistsError:
        raise HarnessRefusal("PACKAGE_DESTINATION_EXISTS") from None
    except OSError:
        raise HarnessRefusal("PACKAGE_CREATION_FAILED") from None
    finally:
        os.close(parent)
    # On compiler failure leave the clearly isolated local package for review;
    # do not recursively delete any paths or claim that preparation completed.
    try:
        _write_new(path / "worker.c", _SOURCE)
        binaries = {}
        for role in ROLES:
            _compile(path / "worker.c", path / role, role)
            binaries[role] = (path / role).read_bytes()
        files = _payload(resources, binaries)
        for name, raw in files.items():
            if name != "worker.c" and name not in ROLES:
                _write_new(path / name, raw)
        inventory = _json(
            {
                "schema": _SCHEMA,
                "version": 1,
                "TEST_ONLY": True,
                "commit": commit,
                "run_id": run_id,
                "uid_base": uid_base,
                "files": {name: _digest(raw) for name, raw in files.items()},
            }
        )
        _write_new(path / "inventory.json", inventory)
        return validate_package(path, _digest(inventory))
    except HarnessRefusal:
        raise
    except (OSError, ValueError, TypeError):
        raise HarnessRefusal("PACKAGE_CREATION_FAILED") from None


def _metadata(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _no_acl(fd: int) -> None:
    backend = getattr(os, "listxattr", None)
    if not callable(backend):
        raise HarnessRefusal("PACKAGE_METADATA_UNAVAILABLE")
    if {"system.posix_acl_access", "system.posix_acl_default"}.intersection(backend(fd)):
        raise HarnessRefusal("PACKAGE_ACL_FORBIDDEN")


def _read_file(directory: int, name: str, owner: int) -> bytes:
    before = os.stat(name, dir_fd=directory, follow_symlinks=False)
    expected_mode = 0o700 if name in ROLES else 0o600
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_uid != owner
        or before.st_nlink != 1
        or stat.S_IMODE(before.st_mode) != expected_mode
        or not 0 < before.st_size <= _MAX_FILE
    ):
        raise HarnessRefusal("PACKAGE_FILE_UNSAFE")
    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    try:
        _no_acl(fd)
        if _metadata(before) != _metadata(os.fstat(fd)):
            raise HarnessRefusal("PACKAGE_CHANGED_DURING_READ")
        chunks = []
        total = 0
        while total <= _MAX_FILE:
            chunk = os.read(fd, min(65536, _MAX_FILE + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total != before.st_size or _metadata(before) != _metadata(os.fstat(fd)):
            raise HarnessRefusal("PACKAGE_CHANGED_DURING_READ")
        if _metadata(before) != _metadata(os.stat(name, dir_fd=directory, follow_symlinks=False)):
            raise HarnessRefusal("PACKAGE_CHANGED_DURING_READ")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _parse(raw: bytes) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in items:
            if key in result:
                raise HarnessRefusal("PACKAGE_SCHEMA_INVALID")
            result[key] = value
        return result

    def constant(_value: str) -> None:
        raise HarnessRefusal("PACKAGE_SCHEMA_INVALID")

    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def validate_package(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Read only a closed package, retaining validated bytes for the installer."""
    path = Path(os.path.abspath(path))
    if expected_sha256 is not None and (
        type(expected_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise HarnessRefusal("PACKAGE_DIGEST_INVALID")
    directory = -1
    try:
        before = path.lstat()
        if (
            not stat.S_ISDIR(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o700
            or before.st_uid == 0
            or (_effective_uid() != 0 and before.st_uid != _effective_uid())
        ):
            raise HarnessRefusal("PACKAGE_DIRECTORY_UNSAFE")
        directory = _open_directory(path)
        _no_acl(directory)
        if _metadata(before) != _metadata(os.fstat(directory)):
            raise HarnessRefusal("PACKAGE_CHANGED_DURING_READ")
        names = {
            "inventory.json",
            "worker.c",
            "README.txt",
            "manifest.json",
            "TEST_ONLY-pin.json",
            "resources.json",
            *ROLES,
            *(f"{role}.service" for role in ROLES),
        }
        if set(os.listdir(directory)) != names:
            raise HarnessRefusal("PACKAGE_CONTENTS_INVALID")
        raw = _read_file(directory, "inventory.json", before.st_uid)
        digest = _digest(raw)
        if expected_sha256 is not None and expected_sha256 != digest:
            raise HarnessRefusal("PACKAGE_DIGEST_MISMATCH")
        inventory = _parse(raw)
        if type(inventory) is not dict or set(inventory) != {
            "schema",
            "version",
            "TEST_ONLY",
            "commit",
            "run_id",
            "uid_base",
            "files",
        }:
            raise HarnessRefusal("PACKAGE_SCHEMA_INVALID")
        if (
            inventory["schema"] != _SCHEMA
            or type(inventory["version"]) is not int
            or inventory["version"] != 1
            or inventory["TEST_ONLY"] is not True
        ):
            raise HarnessRefusal("PACKAGE_SCHEMA_INVALID")
        commit = _commit(inventory["commit"])
        resources = plan(inventory["run_id"], inventory["uid_base"])
        if type(inventory["files"]) is not dict or set(inventory["files"]) != names - {
            "inventory.json"
        }:
            raise HarnessRefusal("PACKAGE_CONTENTS_INVALID")
        files = {
            name: _read_file(directory, name, before.st_uid)
            for name in sorted(names - {"inventory.json"})
        }
        if any(
            type(inventory["files"][name]) is not str
            or inventory["files"][name] != _digest(content)
            for name, content in files.items()
        ):
            raise HarnessRefusal("PACKAGE_FILE_DIGEST_MISMATCH")
        if any(not files[role].startswith(b"\x7fELF") for role in ROLES):
            raise HarnessRefusal("NATIVE_EXECUTABLE_REQUIRED")
        if len({_digest(files[role]) for role in ROLES}) != 3:
            raise HarnessRefusal("DISTINCT_EXECUTABLES_REQUIRED")
        if files != _payload(resources, {role: files[role] for role in ROLES}):
            raise HarnessRefusal("PACKAGE_CANONICAL_CONTENT_MISMATCH")
        if (
            set(os.listdir(directory)) != names
            or _metadata(before) != _metadata(os.fstat(directory))
            or _metadata(before) != _metadata(path.lstat())
        ):
            raise HarnessRefusal("PACKAGE_CHANGED_DURING_READ")
        files["inventory.json"] = raw
        return {"plan": resources, "commit": commit, "files": files, "package_sha256": digest}
    except HarnessRefusal:
        raise
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        raise HarnessRefusal("PACKAGE_VALIDATION_FAILED") from None
    finally:
        if directory >= 0:
            os.close(directory)
