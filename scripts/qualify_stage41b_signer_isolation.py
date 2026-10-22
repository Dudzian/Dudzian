"""Controlled physical Ubuntu qualification using synthetic keys only.

This program never runs pytest, installs a unit, modifies a production unit, or
opens a production secret. --check is read-only. --execute is an explicit root
harness for a dedicated physical host; it cannot produce production attestation.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import re
import secrets
import select
import shutil
import signal
import site
import socket
import stat
import struct
import subprocess
import sys
import time
from typing import Any, Iterator

REQUIRED_BOUNDARIES = (
    "host_preflight",
    "unit_collision_guard",
    "systemd_process_boundaries",
    "own_signature",
    "private_material_separation",
    "foreign_store_dac",
    "foreign_keyring_files",
    "foreign_mount",
    "foreign_dbus",
    "foreign_keyring",
    "foreign_custody",
    "foreign_custody_store",
    "foreign_ipc",
    "own_liveness",
    "identity_binding",
    "unauthorized_ipc",
    "artifact_secrecy",
    "service_liveness",
    "cleanup",
)
ROLES = ("root_signer", "history_signer")
PRODUCTION_UNITS = (
    "cryptohunter-root-proof-signer.service",
    "cryptohunter-history-attestation-signer.service",
    "cryptohunter-root-proof-issuer-check.service",
)
UNIT_FIELDS = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "FragmentPath",
    "DropInPaths",
    "Transient",
    "InvocationID",
    "MainPID",
    "ControlGroup",
    "Description",
    "User",
    "Group",
    "NoNewPrivileges",
    "PrivateMounts",
    "PrivateIPC",
    "PrivateNetwork",
)
MAX_ARTIFACT = 1024 * 1024
MAX_FRAME = 4 * MAX_ARTIFACT
HOST_PROGRAMS = {
    name: Path("/usr/bin") / name
    for name in (
        "systemctl",
        "systemd-run",
        "systemd-detect-virt",
        "journalctl",
        "dbus-daemon",
        "dbus-send",
        "gnome-keyring-daemon",
    )
}
HOST_ENV = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}


@contextmanager
def _defer_interrupts() -> Iterator[None]:
    """Publish ownership before a pending operator interrupt can run cleanup."""
    if sys.platform == "linux":
        previous = signal.pthread_sigmask(
            signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        )
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    else:
        raise QualificationBlocked("UBUNTU_24_04_REQUIRED")


def _host_argv(argv: list[str]) -> list[str]:
    """Only fixed, protected OS programs may receive privileged requests."""
    if not argv:
        raise QualificationBlocked("HOST_PROGRAM_NOT_ALLOWLISTED")
    name = Path(argv[0]).name
    program = HOST_PROGRAMS.get(name)
    if program is None or argv[0] not in {name, str(program)}:
        raise QualificationBlocked("HOST_PROGRAM_NOT_ALLOWLISTED")
    _check_protected(program.resolve(), directory=False)
    for directory in (program.parent, *program.parent.parents):
        _check_protected(directory, directory=True)
    for directory in (program.resolve().parent, *program.resolve().parent.parents):
        _check_protected(directory, directory=True)
    if program.is_symlink() and program.lstat().st_uid != 0:
        raise QualificationBlocked("UNTRUSTED_HOST_PROGRAM")
    return [str(program), *argv[1:]]


class QualificationBlocked(RuntimeError):
    """An unavailable prerequisite; its diagnostic text contains no backend data."""


class QualificationFailed(QualificationBlocked):
    """A completed observation proves a boundary was violated."""

    def __init__(self, boundary: str, reason: str) -> None:
        super().__init__(reason)
        self.boundary = boundary


def result(status: str, reason: str) -> dict[str, str]:
    return {"status": status, "reason": reason}


def aggregate(
    boundaries: dict[str, dict[str, str]], *, physical_executed: bool, cleanup_ok: bool
) -> str:
    """Incomplete, malformed, skipped or merely synthetic evidence never passes."""
    if type(boundaries) is not dict:
        return "BLOCKED"
    if type(physical_executed) is not bool or type(cleanup_ok) is not bool:
        return "BLOCKED"
    if set(boundaries) != set(REQUIRED_BOUNDARIES):
        return "BLOCKED"
    for value in boundaries.values():
        if (
            type(value) is not dict
            or type(value.get("status")) is not str
            or value["status"] not in {"PASS", "FAIL", "BLOCKED"}
            or type(value.get("reason")) is not str
            or not value["reason"]
        ):
            return "BLOCKED"
    if any(value["status"] == "FAIL" for value in boundaries.values()):
        return "FAIL"
    if not physical_executed or not cleanup_ok:
        return "BLOCKED"
    if all(value["status"] == "PASS" for value in boundaries.values()):
        return "STAGE41B_TEST_ONLY_PASS"
    return "BLOCKED"


def classify_denial(
    error_number: int | None, *, service_live: bool, host_path_exists: bool
) -> dict[str, str]:
    """Path absence is a separate observation, never a substitute for liveness."""
    if service_live is not True:
        return result("BLOCKED", "SERVICE_UNAVAILABLE")
    if error_number is None:
        return result("FAIL", "ACCESS_BREACH")
    if type(error_number) is not int:
        return result("BLOCKED", "UNEXPECTED_ERRNO")
    if error_number in {errno.EACCES, errno.EPERM}:
        return result("PASS", "PERMISSION_DENIED")
    if error_number == errno.ENOENT:
        if host_path_exists is True:
            return result("PASS", "NAMESPACE_PATH_ABSENT")
        return result("BLOCKED", "TARGET_MISSING")
    return result("BLOCKED", "UNEXPECTED_ERRNO")


def unit_is_unused(snapshot: dict[str, str]) -> bool:
    """Unknown/masked/installed/inactive-but-loaded names are all collisions."""
    expected = {
        "LoadState": "not-found",
        "ActiveState": "inactive",
        "FragmentPath": "",
        "DropInPaths": "",
        "Transient": "no",
        "InvocationID": "",
        "MainPID": "0",
        "ControlGroup": "",
    }
    return type(snapshot) is dict and all(
        type(snapshot.get(key)) is str and snapshot[key] == value for key, value in expected.items()
    )


def owns_unit(snapshot: dict[str, str], pin: dict[str, str]) -> bool:
    """Do not stop a same-name replacement or a unit whose origin is unknown."""
    if type(snapshot) is not dict or type(pin) is not dict:
        return False
    keys = ("Id", "Description", "InvocationID", "MainPID", "ControlGroup")
    if any(type(pin.get(key)) is not str or not pin[key] for key in keys):
        return False
    unit = pin["Id"]
    if unit not in PRODUCTION_UNITS[:2]:
        return False
    role = ROLES[PRODUCTION_UNITS.index(unit)]
    if re.fullmatch(r"Stage41B TEST ONLY [0-9a-f]{32} " + role, pin["Description"]) is None:
        return False
    if re.fullmatch(r"[0-9a-f]{32}", pin["InvocationID"]) is None:
        return False
    if not pin["MainPID"].isdigit() or int(pin["MainPID"]) <= 1:
        return False
    return (
        snapshot.get("Transient") == "yes"
        and snapshot.get("DropInPaths") == ""
        and pin.get("DropInPaths") == ""
        and snapshot.get("FragmentPath") == f"/run/systemd/transient/{unit}"
        and snapshot.get("LoadState") == "loaded"
        and all(snapshot.get(key) == pin[key] for key in keys)
    )


def owns_invocation(snapshot: dict[str, str], pin: dict[str, str]) -> bool:
    """The same root-created invocation, including a not-yet-started process."""
    if type(snapshot) is not dict or type(pin) is not dict:
        return False
    synthetic = dict(pin)
    # owns_unit applies all name/nonce/transient/fragment checks. A MainPID of
    # zero is permissible only for this separate startup/cleanup predicate.
    synthetic["MainPID"] = "2"
    observed = dict(snapshot)
    observed["MainPID"] = "2"
    if not owns_unit(observed, synthetic):
        return False
    if type(pin.get("MainPID")) is not str or not pin["MainPID"].isdigit():
        return False
    if type(snapshot.get("MainPID")) is not str or not snapshot["MainPID"].isdigit():
        return False
    if int(pin["MainPID"]) > 1 and snapshot["MainPID"] not in {pin["MainPID"], "0"}:
        return False
    return True


def _command(argv: list[str], *, timeout: float = 20) -> bytes:
    completed = subprocess.run(
        _host_argv(argv),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
        env=HOST_ENV,
    )
    if len(completed.stdout) > MAX_ARTIFACT or len(completed.stderr) > MAX_ARTIFACT:
        raise QualificationBlocked("HOST_OUTPUT_LIMIT")
    if completed.returncode:
        # Some systemd versions return a not-found exit code while still
        # returning the requested properties. Accept only a complete unused
        # observation; a dead manager or partial output still fails closed.
        allowed_missing = False
        if argv[:2] == ["systemctl", "show"] and completed.returncode in {1, 4}:
            try:
                values = dict(
                    line.split("=", 1) for line in completed.stdout.decode("utf-8").splitlines()
                )
                allowed_missing = unit_is_unused(values)
            except (UnicodeError, ValueError):
                pass
        if not allowed_missing:
            raise QualificationBlocked("HOST_COMMAND_UNAVAILABLE")
    return completed.stdout


def inspect_unit(name: str) -> dict[str, str]:
    raw = _command(
        ["systemctl", "show", name, "--no-pager", "--all", "--property=" + ",".join(UNIT_FIELDS)]
    )
    values: dict[str, str] = {}
    for line in raw.decode("utf-8", errors="strict").splitlines():
        key, separator, value = line.partition("=")
        if not separator or key in values or key not in UNIT_FIELDS:
            raise QualificationBlocked("MALFORMED_UNIT_OBSERVATION")
        values[key] = value
    if set(values) != set(UNIT_FIELDS):
        raise QualificationBlocked("INCOMPLETE_UNIT_OBSERVATION")
    return values


def _unit_name(run_id: str, role: str) -> str:
    if re.fullmatch(r"[0-9a-f]{32}", run_id) is None or role not in ROLES:
        raise QualificationBlocked("INVALID_TEST_IDENTITY")
    return PRODUCTION_UNITS[ROLES.index(role)]


def _state(run_id: str, role: str) -> Path:
    _unit_name(run_id, role)
    suffix = "root" if role == "root_signer" else "history"
    return Path(f"/run/ch-stage41b-{run_id}-{suffix}")


def _path_absent(path: Path) -> bool:
    """A failed metadata observation never establishes absence."""
    try:
        path.lstat()
    except FileNotFoundError:
        return True
    except OSError:
        raise QualificationBlocked("PATH_METADATA_UNAVAILABLE") from None
    return False


def _check_protected(path: Path, *, directory: bool) -> None:
    """Reject symlinked/writable runtime ancestry and executable source paths."""
    metadata = path.lstat()
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(metadata.st_mode) or metadata.st_uid != 0 or metadata.st_mode & 0o022:
        raise QualificationBlocked("ROOT_OWNED_PROTECTED_PATH_REQUIRED")


def _trusted_code(code_root: Path, interpreter: Path) -> None:
    # ProtectHome makes home checkouts deliberately unusable in the physical mode.
    if any(part in {"home", "root", "tmp"} for part in code_root.parts[1:2]):
        raise QualificationBlocked("PROTECTED_OPT_CHECKOUT_REQUIRED")
    for directory in (code_root, *code_root.parents):
        _check_protected(directory, directory=True)
    for file in (
        code_root / "deployment/stage41b_role_worker.py",
        code_root / "bot_core/local_signing_custody.py",
        interpreter.resolve(),
    ):
        _check_protected(file, directory=False)
    # Check import ancestry before importing even the prepared verifier. An
    # otherwise protected installation must not resolve a user-writable scripts
    # package or helper from the caller's current directory.
    for import_location in sys.path:
        candidate = Path(import_location or os.getcwd())
        if candidate.exists():
            for directory in (candidate, *candidate.parents):
                _check_protected(directory, directory=directory.is_dir())
    prepared = code_root.parent.parent == Path("/opt/cryptohunter-stage41b")
    if prepared:
        _check_protected(code_root / "scripts", directory=True)
        _check_protected(code_root / "scripts/prepare_stage41b_environment.py", directory=False)
        # The prepared inventory checks every directory and regular file in a
        # single no-follow walk and hashes pinned contents. It also replays the
        # Git object proof and authenticated wheel contents, so mutable mtime or
        # a root-owned filename alone cannot stand in for content integrity.
        from scripts.prepare_stage41b_environment import (
            PreparationBlocked,
            verify_installation,
        )

        try:
            verified = verify_installation(code_root.parent, code_root.parent.name)
        except (PreparationBlocked, OSError, ValueError, KeyError, TypeError):
            raise QualificationBlocked("PREPARED_ENVIRONMENT_INTEGRITY_FAILURE") from None
        if str(code_root) != verified["source"] or str(interpreter) != verified["interpreter"]:
            raise QualificationBlocked("PREPARED_EXECUTION_PATH_MISMATCH")
    # Every Python source loaded by the worker must be from this reviewed root
    # checkout, rather than a role-writable module or a user site directory.
    for package in () if prepared else (code_root / "deployment", code_root / "bot_core"):
        for entry in package.rglob("*"):
            if entry.is_dir():
                _check_protected(entry, directory=True)
            else:
                _check_protected(entry, directory=False)
    for directory in (interpreter.parent, *interpreter.parent.parents):
        _check_protected(directory, directory=True)
    for directory in (interpreter.resolve().parent, *interpreter.resolve().parent.parents):
        _check_protected(directory, directory=True)
    if interpreter.is_symlink() and interpreter.lstat().st_uid != 0:
        raise QualificationBlocked("ROOT_OWNED_INTERPRETER_REQUIRED")
    _check_protected(code_root / "scripts/qualify_stage41b_signer_isolation.py", directory=False)
    # Python reads existing bytecode and executable .pth files even with -B.
    # Require a small protected qualification venv, not an editable user env.
    for location in () if prepared else site.getsitepackages():
        package = Path(location)
        if not package.exists():
            continue
        for directory in (package, *package.parents):
            _check_protected(directory, directory=True)
        for entry in package.rglob("*"):
            _check_protected(entry, directory=entry.is_dir())


def _prepared_location(code_root: Path, interpreter: Path) -> dict[str, str]:
    installation = code_root.parent
    if (
        code_root.name != "source"
        or installation.parent != Path("/opt/cryptohunter-stage41b")
        or re.fullmatch(r"[0-9a-f]{40}", installation.name) is None
        or interpreter != installation / "venv/bin/python"
    ):
        raise QualificationBlocked("PREPARED_QUALIFICATION_ENVIRONMENT_REQUIRED")
    return {"commit": installation.name, "installation": str(installation)}


def host_preflight(code_root: Path, interpreter: Path) -> dict[str, str]:
    if sys.platform != "linux":
        raise QualificationBlocked("UBUNTU_24_04_REQUIRED")
    release = {}
    for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            release[key] = value.strip('"')
    if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        raise QualificationBlocked("UBUNTU_24_04_REQUIRED")
    if Path("/proc/1/comm").read_text(encoding="ascii").strip() != "systemd":
        raise QualificationBlocked("SYSTEMD_PID1_REQUIRED")
    if os.geteuid() != 0:
        raise QualificationBlocked("CONTROLLED_ROOT_HARNESS_REQUIRED")
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise QualificationBlocked("PIDFD_PROCESS_CONTROL_REQUIRED")
    for program in HOST_PROGRAMS:
        _host_argv([program])
    virtual = subprocess.run(
        _host_argv(["systemd-detect-virt", "--quiet"]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=5,
        check=False,
        env=HOST_ENV,
    )
    if virtual.returncode != 1:
        raise QualificationBlocked("DEDICATED_PHYSICAL_HOST_REQUIRED")
    _check_protected(Path("/run"), directory=True)
    _prepared_location(code_root, interpreter)
    _trusted_code(code_root, interpreter)
    for module in ("cryptography", "keyring", "secretstorage", "jeepney"):
        __import__(module)
    for name in PRODUCTION_UNITS:
        if not unit_is_unused(inspect_unit(name)):
            raise QualificationBlocked("PRODUCTION_UNIT_COLLISION")
        if not _path_absent(Path("/run/credentials") / name):
            raise QualificationBlocked("PRODUCTION_CREDENTIAL_PATH_COLLISION")
    # Read only the non-secret machine/DMI identity sources used by custody;
    # the original implementation validates them again inside each role.
    for path in (Path("/etc/machine-id"), Path("/sys/devices/virtual/dmi/id/product_uuid")):
        _check_protected(path, directory=False)
        if not path.read_bytes().strip():
            raise QualificationBlocked("CUSTODY_HWID_SOURCE_UNAVAILABLE")
    return result("PASS", "PHYSICAL_HOST_PREREQUISITES_READY")


def _check_unused_account(account: str) -> None:
    if sys.platform == "linux":
        import grp
        import pwd

        try:
            pwd.getpwnam(account)
        except KeyError:
            pass
        else:
            raise QualificationBlocked("TEST_USER_COLLISION")
        try:
            grp.getgrnam(account)
        except KeyError:
            pass
        else:
            raise QualificationBlocked("TEST_GROUP_COLLISION")
    else:
        raise QualificationBlocked("UBUNTU_24_04_REQUIRED")


def unit_properties(run_id: str, role: str, code_root: Path, python: Path) -> list[str]:
    """No production operation, key, password, or native secret in argv/env."""
    name = _unit_name(run_id, role)
    state = _state(run_id, role)
    hidden = Path(f"/run/cryptohunter-stage41b-hidden-{run_id}")
    other = ROLES[1 - ROLES.index(role)]
    properties = {
        "Description": f"Stage41B TEST ONLY {run_id} {role}",
        "Type": "simple",
        "DynamicUser": "yes",
        "User": "chs41b" + run_id[:16] + ("r" if role == "root_signer" else "h"),
        "Group": "chs41b" + run_id[:16] + ("r" if role == "root_signer" else "h"),
        "RuntimeDirectory": state.name,
        "RuntimeDirectoryMode": "0700",
        "NoNewPrivileges": "yes",
        "PrivateMounts": "yes",
        "PrivateIPC": "yes",
        "PrivateNetwork": "yes",
        "PrivateTmp": "yes",
        "ProtectSystem": "strict",
        "ProtectHome": "yes",
        "ReadWritePaths": str(state),
        "InaccessiblePaths": "-/etc/cryptohunter -/var/lib/cryptohunter -/run/cryptohunter",
        "RestrictAddressFamilies": "AF_UNIX",
        "RestrictNamespaces": "yes",
        "CapabilityBoundingSet": "",
        "AmbientCapabilities": "",
        "RemoveIPC": "yes",
        "ProtectKernelTunables": "yes",
        "ProtectKernelModules": "yes",
        "ProtectKernelLogs": "yes",
        "ProtectControlGroups": "yes",
        "RestrictSUIDSGID": "yes",
        "LockPersonality": "yes",
        "UMask": "0077",
        "Restart": "no",
        "TimeoutStopSec": "5s",
        "KillMode": "control-group",
        "RuntimeMaxSec": "180s",
        "TimeoutStartSec": "30s",
        "LimitCORE": "0",
        "LoadCredential": "custody-product-uuid:/sys/devices/virtual/dmi/id/product_uuid",
        "TemporaryFileSystem": f"{hidden}:ro",
        # Discard the test process tree's stdout/stderr before it enters a log
        # channel. Public results travel over the allowlisted test socket only.
        "StandardOutput": "null",
        "StandardError": "null",
        "UnsetEnvironment": "LD_PRELOAD LD_LIBRARY_PATH PYTHONHOME",
    }
    if any(
        "%" in value or "\n" in value or "\0" in value for value in map(str, (code_root, python))
    ):
        raise QualificationBlocked("UNSAFE_EXECUTION_PATH")
    argv = [
        "systemd-run",
        "--quiet",
        "--collect",
        f"--unit={name}",
        f"--working-directory={code_root}",
    ]
    argv += [f"--property={key}={value}" for key, value in properties.items()]
    argv += [
        "--setenv=PATH=/usr/bin:/bin",
        f"--setenv=PYTHONPATH={code_root}",
        "--setenv=PYTHONNOUSERSITE=1",
        "--setenv=PYTHONDONTWRITEBYTECODE=1",
    ]
    argv += [
        str(python),
        "-s",
        "-m",
        "deployment.stage41b_role_worker",
        "--role",
        role,
        "--run-id",
        run_id,
        "--state",
        str(state),
        "--socket",
        str(state / "control.sock"),
        "--hidden-foreign",
        str(hidden / other / "witness"),
    ]
    return argv


def _request(
    path: Path,
    operation: str,
    arguments: dict[str, Any] | None = None,
    *,
    expected_peer: tuple[int, int, int] | None = None,
) -> tuple[dict[str, Any], bytes]:
    if sys.platform != "linux":
        raise QualificationBlocked("UBUNTU_24_04_REQUIRED")
    request = {"operation": operation, "params": arguments or {}}
    raw = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(raw) > MAX_FRAME:
        raise QualificationBlocked("TEST_REQUEST_LIMIT")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
        peer.settimeout(15)
        peer.connect(str(path))
        if expected_peer is not None:
            actual = struct.unpack(
                "3i", peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            )
            if actual != expected_peer:
                raise QualificationBlocked("TEST_SOCKET_PEER_CHANGED")
        peer.sendall(raw)
        peer.shutdown(socket.SHUT_WR)
        reply = bytearray()
        while True:
            part = peer.recv(min(65536, MAX_FRAME + 1 - len(reply)))
            if not part:
                break
            reply.extend(part)
            if len(reply) > MAX_FRAME:
                raise QualificationBlocked("TEST_RESPONSE_LIMIT")
    value = json.loads(reply)
    if type(value) is not dict:
        raise QualificationBlocked("MALFORMED_TEST_RESPONSE")
    return value, raw + bytes(reply)


def validate_status_pair(
    statuses: dict[str, dict[str, Any]], pins: dict[str, dict[str, str]], run_id: str
) -> None:
    for role in ROLES:
        value, pin = statuses[role], pins[role]
        state = _state(run_id, role)
        uid, gid, pid = (value.get(key) for key in ("uid", "gid", "pid"))
        if (
            any(type(number) is not int or number <= 0 for number in (uid, gid, pid))
            or value.get("role") != role
            or value.get("run_id") != run_id
            or value.get("state") != str(state)
            or value.get("socket") != str(state / "control.sock")
            or value.get("bus") != str(state / "bus")
            or value.get("native_service_live") is not True
            or str(pid) != pin["MainPID"]
        ):
            raise QualificationBlocked("INVALID_ROLE_LIVENESS_OR_IDENTITY")
        if value.get("no_new_privs") is not True:
            raise QualificationFailed("systemd_process_boundaries", "NO_NEW_PRIVILEGES_DISABLED")
        proc = Path(f"/proc/{pid}")
        observed = {}
        for line in (proc / "status").read_text(encoding="ascii").splitlines():
            key, _, item = line.partition(":")
            observed[key] = item.strip()
        if (
            observed.get("Uid", "").split() != [str(uid)] * 4
            or observed.get("Gid", "").split() != [str(gid)] * 4
            or observed.get("NoNewPrivs") != "1"
            or any(
                int(observed.get(capability, "1"), 16) != 0
                for capability in ("CapEff", "CapPrm", "CapBnd", "CapAmb")
            )
            or any(group != str(gid) for group in observed.get("Groups", "").split())
        ):
            raise QualificationFailed(
                "systemd_process_boundaries", "HOST_PROCESS_IDENTITY_MISMATCH"
            )
        if (
            not (proc / "cgroup")
            .read_text(encoding="ascii")
            .strip()
            .endswith(":" + pin["ControlGroup"])
        ):
            raise QualificationBlocked("HOST_CGROUP_MISMATCH")
        for namespace in ("mnt", "ipc", "net"):
            inode = (proc / "ns" / namespace).stat().st_ino
            if value.get("namespaces", {}).get(namespace) != inode:
                raise QualificationBlocked("HOST_NAMESPACE_MISMATCH")
            if inode == Path(f"/proc/self/ns/{namespace}").stat().st_ino:
                raise QualificationFailed("systemd_process_boundaries", "HOST_NAMESPACE_SHARED")
        for child_key, executable_name in (
            ("bus_pid", "dbus-daemon"),
            ("keyring_pid", "gnome-keyring-daemon"),
        ):
            child_pid = value.get(child_key)
            if type(child_pid) is not int or child_pid <= 1 or child_pid == pid:
                raise QualificationBlocked("NATIVE_CHILD_IDENTITY_MISSING")
            child = Path(f"/proc/{child_pid}")
            if (child / "exe").resolve().name != executable_name:
                raise QualificationFailed(
                    "systemd_process_boundaries", "NATIVE_CHILD_EXECUTABLE_MISMATCH"
                )
            child_status = {}
            for line in (child / "status").read_text(encoding="ascii").splitlines():
                key, _, item = line.partition(":")
                child_status[key] = item.strip()
            if (
                child_status.get("Uid", "").split() != [str(uid)] * 4
                or child_status.get("Gid", "").split() != [str(gid)] * 4
                or child_status.get("NoNewPrivs") != "1"
                or any(
                    int(child_status.get(capability, "1"), 16) != 0
                    for capability in ("CapEff", "CapPrm", "CapBnd", "CapAmb")
                )
                or any(group != str(gid) for group in child_status.get("Groups", "").split())
                or not (child / "cgroup")
                .read_text(encoding="ascii")
                .strip()
                .endswith(":" + pin["ControlGroup"])
            ):
                raise QualificationFailed(
                    "systemd_process_boundaries", "NATIVE_CHILD_SECURITY_MISMATCH"
                )
            for namespace in ("mnt", "ipc", "net"):
                if (child / "ns" / namespace).stat().st_ino != value["namespaces"][namespace]:
                    raise QualificationFailed(
                        "systemd_process_boundaries", "NATIVE_CHILD_NAMESPACE_MISMATCH"
                    )
        for key in ("state", "bus", "metadata_path", "keyring_path", "private_index_path"):
            path = Path(value[key])
            if not path.is_relative_to(state) or path.is_symlink() or not path.exists():
                raise QualificationBlocked("NATIVE_RESOURCE_MISSING")
        for directory in (state, Path(value["keyring_path"])):
            metadata = directory.lstat()
            if (
                not stat.S_ISDIR(metadata.st_mode)
                or metadata.st_uid != uid
                or metadata.st_mode & 0o077
            ):
                raise QualificationBlocked("NATIVE_RESOURCE_PERMISSIONS")
        for key in ("NoNewPrivileges", "PrivateMounts", "PrivateIPC", "PrivateNetwork"):
            if pin.get(key) != "yes":
                raise QualificationFailed(
                    "systemd_process_boundaries", "SYSTEMD_ISOLATION_PROPERTY_MISSING"
                )
    root, history = (statuses[role] for role in ROLES)
    if any(
        root[key] == history[key]
        for key in (
            "uid",
            "gid",
            "pid",
            "public_key_b64",
            "credential_id",
            "secret_reference",
            "service_namespace",
        )
    ):
        raise QualificationFailed("systemd_process_boundaries", "CROSS_ROLE_IDENTITY_COLLISION")
    for namespace in ("mnt", "ipc", "net"):
        if root["namespaces"][namespace] == history["namespaces"][namespace]:
            raise QualificationFailed("systemd_process_boundaries", "CROSS_ROLE_NAMESPACE_SHARED")


def _verify_signatures(
    statuses: dict[str, dict[str, Any]], proofs: dict[str, dict[str, Any]], run_id: str
) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    for role in ROLES:
        proof = proofs[role]
        public = base64.b64decode(statuses[role]["public_key_b64"], validate=True)
        payload = (
            b"CryptoHunter Stage41B TEST ONLY\0"
            + run_id.encode("ascii")
            + b"\0"
            + role.encode("ascii")
        )
        signature = base64.b64decode(proof["signature_b64"], validate=True)
        if (
            proof.get("public_key_b64") != statuses[role]["public_key_b64"]
            or base64.b64decode(proof["payload_b64"], validate=True) != payload
        ):
            raise QualificationFailed("own_signature", "TEST_SIGNATURE_BINDING_MISMATCH")
        try:
            Ed25519PublicKey.from_public_bytes(public).verify(signature, payload)
        except InvalidSignature:
            raise QualificationFailed("own_signature", "OWN_KEY_SIGNATURE_INVALID") from None
        other = ROLES[1 - ROLES.index(role)]
        foreign = base64.b64decode(statuses[other]["public_key_b64"], validate=True)
        try:
            Ed25519PublicKey.from_public_bytes(foreign).verify(signature, payload)
        except InvalidSignature:
            continue
        raise QualificationFailed("private_material_separation", "CROSS_ROLE_SIGNATURE_COLLISION")


def _process_identity(pid: int) -> dict[str, Any]:
    """Read identity while a pidfd pins this process against PID reuse."""
    if type(pid) is not int or pid <= 1:
        raise QualificationBlocked("PROCESS_IDENTITY_UNPROVEN")
    proc = Path(f"/proc/{pid}")
    fields: dict[str, str] = {}
    for line in (proc / "status").read_text(encoding="ascii").splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value.strip()
    uids, gids = fields.get("Uid", "").split(), fields.get("Gid", "").split()
    if (
        len(uids) != 4
        or len(gids) != 4
        or any(re.fullmatch(r"[0-9]+", value) is None for value in (*uids, *gids))
        or len(set(uids)) != 1
        or len(set(gids)) != 1
        or int(uids[0]) == 0
        or int(gids[0]) == 0
    ):
        raise QualificationBlocked("PROCESS_IDENTITY_UNPROVEN")
    # comm may contain spaces or closing parentheses. Fields after its last
    # ')' begin at field 3; starttime is field 22 (index 19 in this suffix).
    raw_stat = (proc / "stat").read_text(encoding="ascii")
    tail = raw_stat.rsplit(")", 1)[-1].split()
    if len(tail) < 20 or re.fullmatch(r"[0-9]+", tail[19]) is None:
        raise QualificationBlocked("PROCESS_STARTTIME_UNPROVEN")
    cgroup = (proc / "cgroup").read_text(encoding="ascii").strip()
    if not cgroup.startswith("0::/") or "\n" in cgroup:
        raise QualificationBlocked("PROCESS_CGROUP_UNPROVEN")
    return {
        "pid": pid,
        "start_time": int(tail[19]),
        "uid": int(uids[0]),
        "gid": int(gids[0]),
        "control_group": cgroup.removeprefix("0::"),
    }


def _process_finished(handle: int, timeout: float = 0) -> bool:
    return bool(select.select([handle], [], [], timeout)[0])


class PhysicalRun:
    """An owned scope, never a generic root cleanup facility."""

    def __init__(self, run_id: str, code_root: Path, python: Path) -> None:
        self.run_id = run_id
        self.code_root = code_root
        self.python = python
        self.base = Path(f"/run/cryptohunter-stage41b-{run_id}")
        self.hidden = Path(f"/run/cryptohunter-stage41b-hidden-{run_id}")
        self.pins: dict[str, dict[str, str]] = {}
        self.attempted: list[str] = []
        self.owned_paths: list[tuple[Path, int, int, bool]] = []
        self.transcripts: list[bytes] = []
        self.journals: dict[str, bytes] = {}
        self.process_handles: dict[str, int] = {}
        self.process_identities: dict[str, dict[str, Any]] = {}
        self.cleanup_issues: list[dict[str, Any]] = []
        self.unresolved_startups: set[str] = set()

    def _issue(self, role: str | None, reason: str) -> None:
        issue: dict[str, Any] = {"reason": reason}
        if role is not None:
            issue.update(
                role=role,
                unit=_unit_name(self.run_id, role),
                runtime=str(_state(self.run_id, role)),
                credential_directory=str(Path("/run/credentials") / _unit_name(self.run_id, role)),
                expected_description=f"Stage41B TEST ONLY {self.run_id} {role}",
                expected_invocation_id=self.pins.get(role, {}).get("InvocationID", ""),
            )
        if issue not in self.cleanup_issues:
            self.cleanup_issues.append(issue)

    def _matches_role(self, snapshot: dict[str, str], role: str) -> bool:
        return (
            snapshot.get("Id") == _unit_name(self.run_id, role)
            and snapshot.get("Description") == f"Stage41B TEST ONLY {self.run_id} {role}"
            and owns_invocation(snapshot, snapshot)
        )

    def _capture_process(self, role: str, snapshot: dict[str, str]) -> None:
        if sys.platform != "linux":
            raise QualificationBlocked("PIDFD_PROCESS_CONTROL_REQUIRED")
        pid = int(snapshot["MainPID"])
        if pid <= 1:
            return
        if role in self.process_handles:
            if self.process_identities[role]["pid"] != pid:
                raise QualificationBlocked("TEST_PROCESS_PID_CHANGED")
            return
        handle = -1
        retained = False
        try:
            with _defer_interrupts():
                handle = os.pidfd_open(pid, 0)
            before = _process_identity(pid)
            if before["control_group"] != snapshot["ControlGroup"]:
                raise QualificationBlocked("TEST_PROCESS_CGROUP_CHANGED")
            observed = inspect_unit(_unit_name(self.run_id, role))
            if (
                not self._matches_role(observed, role)
                or not owns_unit(observed, snapshot)
                or _process_identity(pid) != before
                or _process_finished(handle)
            ):
                raise QualificationBlocked("TEST_PROCESS_CHANGED_DURING_PIN")
            with _defer_interrupts():
                self.process_handles[role] = handle
                self.process_identities[role] = before
                retained = True
        finally:
            if not retained and handle >= 0:
                os.close(handle)

    def _terminate_process(self, role: str) -> bool:
        """Signal only the pinned process, never a unit looked up by name."""
        if sys.platform != "linux":
            return False
        handle = self.process_handles.get(role)
        if handle is None:
            self._issue(role, "NO_PROVEN_PROCESS_HANDLE")
            return False
        if _process_finished(handle):
            return True
        pin = self.pins[role]
        observed = inspect_unit(_unit_name(self.run_id, role))
        if (
            not self._matches_role(observed, role)
            or not owns_invocation(observed, pin)
            or observed.get("MainPID") != str(self.process_identities[role]["pid"])
            or _process_identity(self.process_identities[role]["pid"])
            != self.process_identities[role]
        ):
            self._issue(role, "PROCESS_OR_INVOCATION_CHANGED_NO_SIGNAL_SENT")
            return False
        try:
            signal.pidfd_send_signal(handle, signal.SIGTERM, None, 0)
            if _process_finished(handle, 6):
                return True
            # The same descriptor continues to target the old process even if
            # a PID or unit name is reused while termination is pending.
            signal.pidfd_send_signal(handle, signal.SIGKILL, None, 0)
            if _process_finished(handle, 2):
                return True
        except ProcessLookupError:
            return _process_finished(handle)
        self._issue(role, "PINNED_PROCESS_DID_NOT_EXIT")
        return False

    def _remember(self, path: Path, directory: bool) -> None:
        info = path.lstat()
        self.owned_paths.append((path, info.st_dev, info.st_ino, directory))

    def create(self) -> None:
        if sys.platform != "linux":
            raise QualificationBlocked("UBUNTU_24_04_REQUIRED")
        for role in ROLES:
            if not unit_is_unused(inspect_unit(_unit_name(self.run_id, role))):
                raise QualificationBlocked("TEST_UNIT_COLLISION")
            if not _path_absent(_state(self.run_id, role)):
                raise QualificationBlocked("TEST_RUNTIME_COLLISION")
            if not _path_absent(Path("/run/credentials") / _unit_name(self.run_id, role)):
                raise QualificationBlocked("PRODUCTION_CREDENTIAL_PATH_COLLISION")
            _check_unused_account(
                "chs41b" + self.run_id[:16] + ("r" if role == "root_signer" else "h")
            )
        for path, mode in ((self.base, 0o700), (self.hidden, 0o755)):
            with _defer_interrupts():
                path.mkdir(mode=mode)
                self._remember(path, True)
        manifest = self.base / "owner.json"
        with _defer_interrupts(), manifest.open("x", encoding="utf-8") as stream:
            self._remember(manifest, False)
            json.dump(
                {
                    "stage": "41B_TEST_ONLY",
                    "run_id": self.run_id,
                    "units": [_unit_name(self.run_id, role) for role in ROLES],
                },
                stream,
                sort_keys=True,
            )
        for role in ROLES:
            directory = self.hidden / role
            with _defer_interrupts():
                directory.mkdir(mode=0o755)
                self._remember(directory, True)
            witness = directory / "witness"
            with _defer_interrupts(), witness.open("xb") as stream:
                self._remember(witness, False)
                stream.write(b"Stage41B public mount-namespace witness\n")
                witness.chmod(0o444)

    def _observe_started(self, role: str, *, wait_for_socket: bool, timeout: float = 20) -> bool:
        """Recover ownership even if systemd-run failed after creating a unit."""
        name = _unit_name(self.run_id, role)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            snapshot = inspect_unit(name)
            if unit_is_unused(snapshot):
                if wait_for_socket:
                    raise QualificationBlocked("TEST_SERVICE_EXITED_BEFORE_PIN")
            elif self._matches_role(snapshot, role):
                old_pin = self.pins.get(role)
                if old_pin is not None and not owns_invocation(snapshot, old_pin):
                    raise QualificationBlocked("TEST_UNIT_OR_PROCESS_REPLACED")
                if old_pin is None or int(old_pin["MainPID"]) <= 1:
                    self.pins[role] = snapshot
                try:
                    self._capture_process(role, snapshot)
                except (ProcessLookupError, FileNotFoundError):
                    if wait_for_socket:
                        raise QualificationBlocked(
                            "TEST_SERVICE_EXITED_BEFORE_PROCESS_PIN"
                        ) from None
                if not wait_for_socket:
                    return True
                if snapshot.get("ActiveState") in {"failed", "inactive"}:
                    raise QualificationBlocked("TEST_SERVICE_START_FAILED")
                if (
                    int(snapshot["MainPID"]) > 1
                    and role in self.process_handles
                    and not _process_finished(self.process_handles[role])
                    and (_state(self.run_id, role) / "control.sock").exists()
                ):
                    return True
            else:
                raise QualificationBlocked("TEST_UNIT_ORIGIN_MISMATCH")
            time.sleep(0.1)
        if wait_for_socket:
            raise QualificationBlocked("TEST_SERVICE_START_TIMEOUT")
        return False

    def start(self) -> None:
        for role in ROLES:
            name = _unit_name(self.run_id, role)
            if not unit_is_unused(inspect_unit(name)):
                raise QualificationBlocked("TEST_UNIT_COLLISION")
            self.attempted.append(role)
            try:
                _command(unit_properties(self.run_id, role, self.code_root, self.python))
            except BaseException:
                # Client failure does not roll back StartTransientUnit. Observe
                # the manager before propagating the original error/interrupt.
                self.unresolved_startups.add(role)
                try:
                    if self._observe_started(role, wait_for_socket=False, timeout=2):
                        self.unresolved_startups.discard(role)
                except Exception:
                    pass
                raise
            self._observe_started(role, wait_for_socket=True)

    def request(
        self, role: str, operation: str, arguments: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        identity = self.process_identities.get(role)
        expected = None if identity is None else (identity["pid"], identity["uid"], identity["gid"])
        value, transcript = _request(
            _state(self.run_id, role) / "control.sock",
            operation,
            arguments,
            expected_peer=expected,
        )
        self.transcripts.append(transcript)
        if value.get("status") == "FAIL" and value.get("reason") == "PRIVATE_MATERIAL_EXPOSED":
            raise QualificationFailed("artifact_secrecy", "PRIVATE_MATERIAL_EXPOSED")
        if len(self.transcripts) > 64:
            raise QualificationBlocked("TEST_TRANSCRIPT_LIMIT")
        return value

    def live_statuses(self) -> dict[str, dict[str, Any]]:
        statuses = {}
        for role in ROLES:
            snapshot = inspect_unit(_unit_name(self.run_id, role))
            if not owns_unit(snapshot, self.pins[role]) or snapshot.get("ActiveState") != "active":
                raise QualificationBlocked("TEST_SERVICE_NOT_LIVE")
            statuses[role] = self.request(role, "TEST_STATUS")
        validate_status_pair(statuses, self.pins, self.run_id)
        for role in ROLES:
            observed = inspect_unit(_unit_name(self.run_id, role))
            if (
                not owns_unit(observed, self.pins[role])
                or observed.get("ActiveState") != "active"
                or role not in self.process_handles
                or _process_finished(self.process_handles[role])
                or _process_identity(self.process_identities[role]["pid"])
                != self.process_identities[role]
            ):
                raise QualificationBlocked("TEST_PROCESS_CHANGED_DURING_OBSERVATION")
        return statuses

    def scan(self) -> bool:
        for role in ROLES:
            self.journals[role] = _command(
                [
                    "journalctl",
                    "_SYSTEMD_INVOCATION_ID=" + self.pins[role]["InvocationID"],
                    "--output=cat",
                    "--no-pager",
                    "--quiet",
                ]
            )
        artifacts = [*self.transcripts, *self.journals.values()]
        if any(len(artifact) > MAX_ARTIFACT for artifact in artifacts):
            raise QualificationBlocked("TEST_ARTIFACT_LIMIT")
        # Keep contiguous artifacts intact when possible. Adjacent overlapping
        # chunks preserve matches straddling a boundary (the longest native
        # record included by the worker is bounded well below this overlap).
        chunks = []
        for artifact in artifacts:
            for offset in range(0, max(1, len(artifact)), 32768):
                chunks.append(base64.b64encode(artifact[offset : offset + 65536]).decode("ascii"))
        for role in ROLES:
            for offset in range(0, len(chunks), 24):
                value = self.request(
                    role, "TEST_SCAN", {"artifacts_b64": chunks[offset : offset + 24]}
                )
                if (
                    value.get("status") != "PASS"
                    or value.get("reason") != "NO_PRIVATE_MATERIAL_IN_ARTIFACTS"
                ):
                    return False
        return True

    def _wait_collected(self, role: str) -> bool:
        deadline = time.monotonic() + 5
        name = _unit_name(self.run_id, role)
        while time.monotonic() < deadline:
            observed = inspect_unit(name)
            if unit_is_unused(observed):
                if not _path_absent(_state(self.run_id, role)) or not _path_absent(
                    Path("/run/credentials") / name
                ):
                    self._issue(role, "UNPROVEN_RUNTIME_OR_CREDENTIAL_REMAINS")
                    return False
                return True
            if not self._matches_role(observed, role) or not owns_invocation(
                observed, self.pins[role]
            ):
                self._issue(role, "UNIT_REPLACED_DURING_CLEANUP_NO_ACTION_TAKEN")
                return False
            time.sleep(0.1)
        self._issue(role, "MANAGER_COLLECTION_NOT_CONFIRMED")
        return False

    def cleanup(self) -> bool:
        # The caller ignores handled signals too, but this method is safe when
        # invoked directly after an exception or from another reviewed caller.
        clean = False
        with _defer_interrupts():
            try:
                clean = self._cleanup_owned()
            finally:
                for handle in self.process_handles.values():
                    try:
                        os.close(handle)
                    except OSError:
                        clean = False
                        self._issue(None, "PROCESS_HANDLE_CLOSE_UNCONFIRMED")
                self.process_handles.clear()
        return clean

    def _cleanup_owned(self) -> bool:
        clean = True
        for role in reversed(self.attempted):
            name = _unit_name(self.run_id, role)
            try:
                observed = inspect_unit(name)
                if unit_is_unused(observed):
                    if role in self.unresolved_startups:
                        # A queued manager request may outlive a timed-out
                        # client. One absent snapshot cannot prove rollback.
                        clean = False
                        self._issue(role, "STARTUP_OUTCOME_UNPROVEN")
                    if not _path_absent(_state(self.run_id, role)) or not _path_absent(
                        Path("/run/credentials") / name
                    ):
                        clean = False
                        self._issue(role, "UNPROVEN_RUNTIME_OR_CREDENTIAL_REMAINS")
                    continue
                pin = self.pins.get(role)
                # Recovery is permitted only for this exact unpredictable
                # description and transient origin after our recorded attempt.
                if not self._matches_role(observed, role) or (
                    pin is not None and not owns_invocation(observed, pin)
                ):
                    clean = False
                    self._issue(role, "FOREIGN_OR_REPLACED_UNIT_RETAINED")
                    continue
                if pin is None:
                    self.pins[role] = observed
                self.unresolved_startups.discard(role)
                if int(observed["MainPID"]) > 1:
                    try:
                        self._capture_process(role, observed)
                    except (ProcessLookupError, FileNotFoundError):
                        pass
                    if not self._terminate_process(role):
                        clean = False
                        continue
                elif observed.get("ActiveState") not in {"inactive", "failed", "deactivating"}:
                    self._issue(role, "NO_PROVEN_PROCESS_TARGET_WAIT_FOR_UNIT_TIMERS")
                    clean = False
                    continue
                if not self._wait_collected(role):
                    clean = False
            except Exception:
                clean = False
                self._issue(role, "UNIT_CLEANUP_OBSERVATION_OR_SIGNAL_UNAVAILABLE")
        # Filesystem cleanup is deliberately non-recursive. Every unlink/rmdir
        # requires its recorded device/inode, root ownership and expected type.
        for path, device, inode, directory in reversed(self.owned_paths):
            try:
                info = path.lstat()
                kind = stat.S_ISDIR if directory else stat.S_ISREG
                ancestry_intact = True
                for parent, parent_device, parent_inode, parent_is_directory in self.owned_paths:
                    if parent_is_directory and parent in path.parents:
                        metadata = parent.lstat()
                        if (
                            metadata.st_uid != 0
                            or not stat.S_ISDIR(metadata.st_mode)
                            or metadata.st_dev != parent_device
                            or metadata.st_ino != parent_inode
                        ):
                            ancestry_intact = False
                            break
                if (
                    info.st_uid != 0
                    or info.st_dev != device
                    or info.st_ino != inode
                    or not kind(info.st_mode)
                    or not ancestry_intact
                ):
                    clean = False
                    self._issue(None, "OWNED_PATH_IDENTITY_CHANGED_RETAINED")
                    continue
                if directory:
                    path.rmdir()
                else:
                    path.unlink()
            except Exception:
                clean = False
                self._issue(None, "OWNED_PATH_REMOVAL_UNCONFIRMED")
        return clean


def _execute(run: PhysicalRun, boundaries: dict[str, dict[str, str]]) -> None:
    run.create()
    boundaries["unit_collision_guard"] = result("PASS", "ALL_NAMES_UNUSED_AND_OWNED_SCOPE_CREATED")
    run.start()
    statuses = run.live_statuses()
    boundaries["systemd_process_boundaries"] = result(
        "PASS", "HOST_CONFIRMED_UID_GID_MNT_IPC_NET_NNP"
    )
    proofs = {role: run.request(role, "TEST_SIGN") for role in ROLES}
    _verify_signatures(statuses, proofs, run.run_id)
    boundaries["own_signature"] = result("PASS", "BOTH_NATIVE_KEYS_VERIFY_FIXED_TEST_CHALLENGE")
    boundaries["private_material_separation"] = result(
        "PASS", "DISTINCT_PUBLIC_KEYS_AND_CROSS_VERIFICATION_DENIAL"
    )
    per_role = {}
    for role in ROLES:
        other = ROLES[1 - ROLES.index(role)]
        # Both native services were independently observed live above; the
        # witness must also exist on the host before an ENOENT is meaningful.
        _check_protected(run.hidden / other / "witness", directory=False)
        per_role[role] = run.request(role, "TEST_PROBE", {"foreign": statuses[other]})
    after = run.live_statuses()
    if after != statuses:
        raise QualificationBlocked("LIVE_ROLE_FACTS_CHANGED")
    boundaries["service_liveness"] = result("PASS", "BOTH_NATIVE_SERVICES_LIVE_BEFORE_AND_AFTER")
    probe_reasons = {
        "foreign_store_dac": "PERMISSION_DENIED",
        "foreign_keyring_files": "PERMISSION_DENIED",
        "foreign_mount": "NAMESPACE_PATH_ABSENT",
        "foreign_dbus": "PERMISSION_DENIED",
        "foreign_keyring": "FOREIGN_SECRET_ABSENT_FROM_OWN_NATIVE_BUS",
        "foreign_custody": "FOREIGN_REFERENCE_AND_IDENTITY_REJECTED",
        "foreign_custody_store": "FOREIGN_ORIGINAL_PROVIDER_STORE_REJECTED",
        "foreign_ipc": "PERMISSION_DENIED",
        "own_liveness": "ORIGINAL_CUSTODY_READ_AND_BUS_OWNER",
        "identity_binding": "ROLE_AND_CREDENTIAL_BINDING_REJECTED",
        "unauthorized_ipc": "UNPRIVILEGED_PEER_REJECTED",
    }
    probe_names = set(probe_reasons)
    for name in probe_names:
        values = [per_role[role].get("boundaries", {}).get(name) for role in ROLES]
        if any(
            type(value) is not dict
            or value.get("status") not in {"PASS", "FAIL", "BLOCKED"}
            or type(value.get("reason")) is not str
            for value in values
        ):
            boundaries[name] = result("BLOCKED", "INCOMPLETE_ROLE_PROBE")
        elif any(value["status"] == "FAIL" for value in values):
            boundaries[name] = result("FAIL", "CROSS_ROLE_BOUNDARY_BREACH")
        elif any(value["status"] != "PASS" for value in values):
            boundaries[name] = result("BLOCKED", "ROLE_BOUNDARY_UNAVAILABLE")
        elif any(value["reason"] != probe_reasons[name] for value in values):
            boundaries[name] = result("BLOCKED", "EXPECTED_DENIAL_NOT_OBSERVED")
        else:
            boundaries[name] = result("PASS", "BOTH_ROLES_" + str(values[0]["reason"]))
    boundaries["artifact_secrecy"] = result(
        "PASS" if run.scan() else "FAIL", "INTERNAL_NATIVE_MATERIAL_SCAN"
    )
    run.live_statuses()


def inspect_cleanup(run_id: str) -> dict[str, Any]:
    """Inspect only known metadata. This mode never stops or removes anything."""
    _unit_name(run_id, ROLES[0])  # validate the nonce before constructing paths
    report: dict[str, Any] = {
        "stage": "41B_TEST_ONLY",
        "mode": "inspect-cleanup",
        "run_id": run_id,
        "status": "BLOCKED",
        "read_only": True,
        "units": {},
        "paths": [],
        "physical_qualification": "NOT_EXECUTED",
        "production_private_control": "BLOCKED",
        "stage9": "IN_PROGRESS / NO-GO",
        "authorizes_sign_bind_enrollment": False,
        "manual_review": (
            "Compare the exact test description and invocation ID with the execution receipt. "
            "Do not stop a replacement unit or remove an unproven path. "
            "This inspection does not authorize cleanup or production operations."
        ),
    }
    if sys.platform != "linux":
        report["reason"] = "UBUNTU_24_04_REQUIRED"
        return report
    all_absent = True
    paths = [
        Path(f"/run/cryptohunter-stage41b-{run_id}"),
        Path(f"/run/cryptohunter-stage41b-hidden-{run_id}"),
    ]
    for role in ROLES:
        name = _unit_name(run_id, role)
        entry: dict[str, Any] = {
            "unit": name,
            "expected_description": f"Stage41B TEST ONLY {run_id} {role}",
            "status": "BLOCKED",
            "reason": "UNIT_OBSERVATION_UNAVAILABLE",
            "inspect_command": [
                str(HOST_PROGRAMS["systemctl"]),
                "show",
                name,
                "--no-pager",
                "--property=Id,LoadState,ActiveState,Transient,FragmentPath,DropInPaths,Description,InvocationID,MainPID,ControlGroup",
            ],
        }
        try:
            observed = inspect_unit(name)
            if unit_is_unused(observed):
                entry.update(status="PASS", reason="UNIT_ABSENT")
            elif (
                observed.get("Id") == name
                and observed.get("Description") == entry["expected_description"]
                and owns_invocation(observed, observed)
            ):
                entry["reason"] = "MATCHING_TEST_UNIT_REMAINS_NO_ACTION_TAKEN"
                entry["observed_invocation_id"] = observed["InvocationID"]
                entry["observed_main_pid"] = observed["MainPID"]
            else:
                entry["reason"] = "FOREIGN_OR_UNPROVEN_UNIT_NO_ACTION_TAKEN"
        except (OSError, ValueError, QualificationBlocked, subprocess.SubprocessError):
            pass
        if entry["status"] != "PASS":
            all_absent = False
        report["units"][role] = entry
        paths.extend((_state(run_id, role), Path("/run/credentials") / name))
    for path in paths:
        entry = {"path": str(path), "status": "BLOCKED", "reason": "PATH_METADATA_UNAVAILABLE"}
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            entry.update(status="PASS", reason="PATH_ABSENT")
        except OSError:
            all_absent = False
        else:
            all_absent = False
            entry.update(reason="PATH_REMAINS_NO_ACTION_TAKEN")
            entry["metadata"] = {
                "uid": metadata.st_uid,
                "gid": metadata.st_gid,
                "mode": stat.S_IMODE(metadata.st_mode),
                "device": metadata.st_dev,
                "inode": metadata.st_ino,
                "type": "directory"
                if stat.S_ISDIR(metadata.st_mode)
                else "symlink"
                if stat.S_ISLNK(metadata.st_mode)
                else "other",
            }
        report["paths"].append(entry)
    if all_absent:
        report.update(status="PASS", reason="ALL_EXPECTED_RESOURCES_ABSENT")
    else:
        report["reason"] = "RESIDUAL_OR_UNPROVEN_RESOURCES_RETAINED"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--check", action="store_true", help="read-only host prerequisites; never a physical PASS"
    )
    modes.add_argument(
        "--execute",
        action="store_true",
        help="controlled root-only physical synthetic-key qualification",
    )
    modes.add_argument(
        "--inspect-cleanup",
        action="store_true",
        help="read-only metadata inspection for a previous run; never stops or removes resources",
    )
    parser.add_argument(
        "--run-id", help="exact 32-character lowercase hexadecimal nonce for inspection"
    )
    args = parser.parse_args(argv)
    if args.inspect_cleanup:
        if args.run_id is None or re.fullmatch(r"[0-9a-f]{32}", args.run_id) is None:
            parser.error("--inspect-cleanup requires an exact --run-id")
        inspection = inspect_cleanup(args.run_id)
        print(json.dumps(inspection, sort_keys=True, indent=2))
        return 0 if inspection["status"] == "PASS" else 2
    if args.run_id is not None:
        parser.error("--run-id is permitted only with --inspect-cleanup")
    boundaries = {
        key: result("BLOCKED", "PHYSICAL_CONTROL_NOT_EXECUTED") for key in REQUIRED_BOUNDARIES
    }
    run = PhysicalRun(
        secrets.token_hex(16), Path(__file__).resolve().parents[1], Path(sys.executable)
    )
    cleanup_ok = True
    physical_executed = False
    verified_context: dict[str, str] | None = None
    previous = {}
    try:
        boundaries["host_preflight"] = host_preflight(run.code_root, run.python)
        verified_context = _prepared_location(run.code_root, run.python)
        if args.execute:

            def interrupted(_number: int, _frame: Any) -> None:
                raise QualificationBlocked("PHYSICAL_RUN_INTERRUPTED")

            handled_signals = [signal.SIGTERM, signal.SIGINT]
            if sys.platform == "linux":
                handled_signals.append(signal.SIGHUP)
            for selected in handled_signals:
                previous[selected] = signal.signal(selected, interrupted)
            physical_executed = True
            _execute(run, boundaries)
    except Exception as error:
        reason = (
            str(error)
            if isinstance(error, QualificationBlocked)
            else "PHYSICAL_CONTROL_UNAVAILABLE"
        )
        # Exception/backend text is never trusted output: only fixed codes.
        if re.fullmatch(r"[A-Z][A-Z0-9_]{1,96}", reason) is None:
            reason = "PHYSICAL_CONTROL_UNAVAILABLE"
        if isinstance(error, QualificationFailed) and error.boundary in REQUIRED_BOUNDARIES:
            boundaries[error.boundary] = result("FAIL", reason)
        if boundaries["host_preflight"]["status"] != "PASS":
            boundaries["host_preflight"] = result("BLOCKED", reason)
        boundaries["service_liveness"] = result("BLOCKED", reason)
    finally:
        if args.execute and (run.owned_paths or run.attempted):
            # A second operator interrupt must not interrupt the owned-scope cleanup.
            for selected in previous:
                signal.signal(selected, signal.SIG_IGN)
            try:
                cleanup_ok = run.cleanup()
            except Exception:
                cleanup_ok = False
                run.cleanup_issues.append({"reason": "CLEANUP_UNEXPECTED_FAILURE"})
            boundaries["cleanup"] = result(
                "PASS" if cleanup_ok else "BLOCKED",
                "OWNED_SCOPE_REMOVED" if cleanup_ok else "UNVERIFIED_RESOURCE_RETAINED",
            )
        for selected, handler in previous.items():
            signal.signal(selected, handler)
    status = aggregate(boundaries, physical_executed=physical_executed, cleanup_ok=cleanup_ok)
    print(
        json.dumps(
            {
                "stage": "41B_TEST_ONLY",
                "status": status,
                "run_id": run.run_id,
                "mode": "execute" if args.execute else "check",
                "qualification_commit": None
                if verified_context is None
                else verified_context["commit"],
                "installation": None
                if verified_context is None
                else verified_context["installation"],
                "test_invocations": {role: pin["InvocationID"] for role, pin in run.pins.items()},
                "test_process_identities": run.process_identities,
                "cleanup_issues": run.cleanup_issues,
                "cleanup_inspection": [
                    str(run.python),
                    "-I",
                    "-B",
                    "-c",
                    'import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); runpy.run_module("scripts.qualify_stage41b_signer_isolation", run_name="__main__")',
                    str(run.code_root),
                    "--inspect-cleanup",
                    "--run-id",
                    run.run_id,
                ],
                "cleanup_review": (
                    "Run the read-only cleanup inspection and compare its description/invocation "
                    "with this receipt. Never stop a replacement unit or delete an unproven path."
                ),
                "boundaries": boundaries,
                "production_private_control": "BLOCKED",
                "stage9": "IN_PROGRESS / NO-GO",
                "authorizes_sign_bind_enrollment": False,
                "scope_paths_requiring_review": []
                if cleanup_ok
                else [str(run.base), str(run.hidden)],
                "resources_requiring_review": []
                if cleanup_ok
                else [
                    {"unit": _unit_name(run.run_id, role), "runtime": str(_state(run.run_id, role))}
                    for role in run.attempted
                ],
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0 if status == "STAGE41B_TEST_ONLY_PASS" else (1 if status == "FAIL" else 2)


if __name__ == "__main__":
    raise SystemExit(main())
