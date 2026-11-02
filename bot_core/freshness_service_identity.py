"""Odczytowe powiązanie efektywnej jednostki verifiera z żywym procesem Linux.

Nie jest to dowód pochodzenia hosta ani kwalifikacja funkcjonalna runtime.
Wymagamy zamrożonego ExecStart D-Bus, PIDFD i dokładnych obserwacji /proc.
"""

from __future__ import annotations

import hashlib
import json
import os
import select
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bot_core.linux_service_ipc import capture_process_identity
from bot_core.production_signer_ipc import ProductionSignerIPCError, _read_proc

UNIT_NAME = "cryptohunter-freshness-verifier.service"
EXECUTABLE = "/usr/bin/python3"
EXPECTED_ARGV = (
    EXECUTABLE,
    "-I",
    "-s",
    "-B",
    "/usr/lib/cryptohunter/deployment/freshness_verifier_entrypoint.py",
)
CGROUP = "/system.slice/" + UNIT_NAME
_OBJECT_PATH = "/org/freedesktop/systemd1/unit/cryptohunter_2dfreshness_2dverifier_2eservice"
_STATIC_PROPERTIES = {
    "Id": UNIT_NAME,
    "Names": UNIT_NAME,
    "LoadState": "loaded",
    "ActiveState": "active",
    "SubState": "running",
    "Type": "simple",
    "User": "os_freshness_crypto_verifier",
    "Group": "freshness_verifier_ipc",
    "DynamicUser": "no",
    "SupplementaryGroups": "",
    "Environment": "PYTHONNOUSERSITE=1",
    "EnvironmentFiles": "",
    "PassEnvironment": "",
    "UnsetEnvironment": "",
    "WorkingDirectory": "/usr/lib/cryptohunter",
    "RootDirectory": "",
    "RootImage": "",
    "FragmentPath": "/etc/systemd/system/" + UNIT_NAME,
    "DropInPaths": "",
    "NeedDaemonReload": "no",
    "ControlGroup": CGROUP,
    "NoNewPrivileges": "yes",
    "PrivateTmp": "yes",
    "ProtectSystem": "strict",
    "ProtectHome": "yes",
    "ProtectKernelTunables": "yes",
    "ProtectKernelModules": "yes",
    "ProtectKernelLogs": "yes",
    "ProtectControlGroups": "yes",
    "RestrictSUIDSGID": "yes",
    "LockPersonality": "yes",
    "RestrictNamespaces": "yes",
    "RestrictRealtime": "yes",
    "MemoryDenyWriteExecute": "yes",
    "CapabilityBoundingSet": "",
    "AmbientCapabilities": "",
    "RestrictAddressFamilies": "AF_UNIX",
    "SystemCallArchitectures": "native",
    "RuntimeDirectory": "cryptohunter/freshness",
    "RuntimeDirectoryMode": "0750",
    "UMask": "0027",
}
_DYNAMIC_PROPERTIES = ("MainPID", "ExecMainPID", "ExecMainStartTimestampMonotonic")
SYSTEMD_PROPERTIES = (*_STATIC_PROPERTIES, *_DYNAMIC_PROPERTIES)
_MAX_REPLY = 65536
_MAX_EXECUTABLE = 512 * 1024 * 1024


class ServiceIdentityError(RuntimeError):
    """Stały kod odmowy; dane hosta i stderr nie stanowią treści raportu."""


def _require(condition: bool, code: str) -> None:
    if not condition:
        raise ServiceIdentityError(code)


def _query(arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            arguments,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
            text=True,
            encoding="utf-8",
            errors="strict",
            cwd="/",
            env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"},
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise ServiceIdentityError("SERVICE_MANAGER_UNAVAILABLE") from exc
    _require(result.returncode == 0, "SERVICE_MANAGER_UNAVAILABLE")
    _require(len(result.stdout) <= _MAX_REPLY, "SERVICE_MANAGER_REPLY_TOO_LARGE")
    return result.stdout


def _systemctl(*arguments: str) -> str:
    return _query(["/usr/bin/systemctl", *arguments])


def _properties(raw: str) -> dict[str, str]:
    _require(
        type(raw) is str
        and len(raw) <= _MAX_REPLY
        and raw.isascii()
        and all(ord(char) >= 32 or char == "\n" for char in raw),
        "SYSTEMD_PROPERTIES_MALFORMED",
    )
    result: dict[str, str] = {}
    for line in raw.splitlines():
        key, separator, value = line.partition("=")
        _require(
            bool(separator) and key in SYSTEMD_PROPERTIES and key not in result,
            "SYSTEMD_PROPERTIES_MALFORMED",
        )
        result[key] = value
    _require(set(result) == set(SYSTEMD_PROPERTIES), "SYSTEMD_PROPERTIES_INCOMPLETE")
    _require(
        all(result[key] == value for key, value in _STATIC_PROPERTIES.items()),
        "EFFECTIVE_SYSTEMD_CONFIGURATION_CHANGED",
    )
    for key in _DYNAMIC_PROPERTIES:
        value = result[key]
        _require(
            value.isdecimal() and len(value) <= 20 and 0 < int(value) < 2**64,
            "SERVICE_PROCESS_BINDING_INVALID",
        )
    _require(
        result["MainPID"] == result["ExecMainPID"] and 1 < int(result["MainPID"]) < 2**31,
        "SERVICE_MAIN_PID_NOT_MATCHED",
    )
    return result


def _exec_start(raw: str, pid: int) -> tuple[Any, ...]:
    _require(type(raw) is str and len(raw) <= _MAX_REPLY, "EXECSTART_REPLY_MALFORMED")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            _require(key not in result, "EXECSTART_REPLY_MALFORMED")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
    except (ValueError, TypeError, RecursionError) as exc:
        raise ServiceIdentityError("EXECSTART_REPLY_MALFORMED") from exc
    _require(
        type(value) is dict
        and set(value) == {"type", "data"}
        and value["type"] == "a(sasbttttuii)"
        and type(value["data"]) is list
        and len(value["data"]) == 1,
        "EXACT_EXECSTART_REQUIRED",
    )
    command = value["data"][0]
    _require(type(command) is list and len(command) == 10, "EXACT_EXECSTART_REQUIRED")
    _require(
        command[0] == EXECUTABLE
        and type(command[1]) is list
        and tuple(command[1]) == EXPECTED_ARGV
        and command[2] is False,
        "EXECSTART_PROGRAM_OR_ARGUMENTS_CHANGED",
    )
    _require(
        all(type(item) is int and 0 <= item < 2**64 for item in command[3:])
        and command[3] > 0
        and command[4] > 0
        and command[5:7] == [0, 0]
        and command[7] == pid
        and command[8:] == [0, 0],
        "EXECSTART_PROCESS_NOT_MATCHED",
    )
    return (command[0], tuple(command[1]), *command[2:])


def _read_exec_start(pid: int) -> tuple[Any, ...]:
    return _exec_start(
        _query(
            [
                "/usr/bin/busctl",
                "--system",
                "--json=short",
                "get-property",
                "org.freedesktop.systemd1",
                _OBJECT_PATH,
                "org.freedesktop.systemd1.Service",
                "ExecStart",
            ]
        ),
        pid,
    )


def _metadata(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _program_observation(pid: int) -> dict[str, Any]:
    """Czytaj obraz przez jądrowe /proc/PID/exe, nie przez argv lub comm."""
    if sys.platform == "win32":
        raise ServiceIdentityError("NATIVE_LINUX_PROCESS_REQUIRED")
    directory = os.open(f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        raw = _read_proc(directory, "cmdline")
        _require(raw.endswith(b"\x00"), "PROCESS_ARGUMENTS_MALFORMED")
        argv = tuple(os.fsdecode(item) for item in raw[:-1].split(b"\x00"))
        expected_path = str(Path(EXECUTABLE).resolve(strict=True))
        _require(os.readlink("exe", dir_fd=directory) == expected_path, "PROCESS_PROGRAM_CHANGED")
        executable = os.open("exe", os.O_RDONLY | os.O_CLOEXEC, dir_fd=directory)
        expected: int | None = None
        try:
            expected = os.open(EXECUTABLE, os.O_RDONLY | os.O_CLOEXEC)
            before = os.fstat(expected)
            _require(
                stat.S_ISREG(before.st_mode)
                and before.st_uid == before.st_gid == 0
                and not before.st_mode & 0o022
                and bool(before.st_mode & 0o111)
                and 0 < before.st_size <= _MAX_EXECUTABLE,
                "EXPECTED_PYTHON_NOT_PROTECTED",
            )
            _require(
                _metadata(os.fstat(executable)) == _metadata(before),
                "PROCESS_PROGRAM_INODE_CHANGED",
            )
            digest = hashlib.sha256()
            total = 0
            while chunk := os.read(expected, 1024 * 1024):
                total += len(chunk)
                _require(total <= _MAX_EXECUTABLE, "EXPECTED_PYTHON_TOO_LARGE")
                digest.update(chunk)
            _require(
                total == before.st_size
                and _metadata(before)
                == _metadata(os.fstat(expected))
                == _metadata(os.fstat(executable))
                == _metadata(os.stat("exe", dir_fd=directory))
                == _metadata(os.stat(EXECUTABLE)),
                "PROCESS_PROGRAM_CHANGED_DURING_OBSERVATION",
            )
            return {
                "argv": argv,
                "exe_sha256": digest.hexdigest(),
                "exe_metadata": _metadata(before),
            }
        finally:
            if expected is not None:
                os.close(expected)
            os.close(executable)
    finally:
        os.close(directory)


def _check_pidfd(pidfd: int, pid: int) -> None:
    if sys.platform == "win32":
        raise ServiceIdentityError("NATIVE_LINUX_PIDFD_REQUIRED")
    directory = os.open("/proc/self/fdinfo", os.O_RDONLY | os.O_DIRECTORY)
    try:
        information = _read_proc(directory, str(pidfd))
    finally:
        os.close(directory)
    _require(
        [line.split()[1:] for line in information.splitlines() if line.startswith(b"Pid:")]
        == [[str(pid).encode("ascii")]],
        "MAIN_PIDFD_NOT_MATCHED",
    )
    poller = select.poll()
    poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
    _require(not poller.poll(0), "SERVICE_PROCESS_EXITED")


def qualify_freshness_service_identity(
    *, uid: int, gid: int, systemctl: Callable[..., str] = _systemctl
) -> None:
    """Odmów przy zmianie konfiguracji, argumentów, obrazu lub życia MainPID."""
    if sys.platform == "win32":
        raise ServiceIdentityError("NATIVE_LINUX_PIDFD_REQUIRED")
    _require(sys.platform == "linux", "NATIVE_LINUX_PIDFD_REQUIRED")
    _require(
        type(uid) is int and type(gid) is int and uid > 0 and gid > 0,
        "SERVICE_UID_GID_REQUIRED",
    )
    pidfd: int | None = None
    try:
        before = _properties(
            systemctl("show", UNIT_NAME, "--no-pager", "--property=" + ",".join(SYSTEMD_PROPERTIES))
        )
        pid = int(before["MainPID"])
        pidfd = os.pidfd_open(pid, 0)
        _check_pidfd(pidfd, pid)
        command = _read_exec_start(pid)
        first = capture_process_identity(pid)
        _require(
            first["pid"] == pid
            and first["uid"] == uid
            and first["gid"] == gid
            and set(first["groups"]) <= {gid}
            and first["cgroup"] == before["ControlGroup"] == CGROUP,
            "SERVICE_PROCESS_CREDENTIALS_OR_CGROUP_CHANGED",
        )
        program = _program_observation(pid)
        _require(program["argv"] == EXPECTED_ARGV, "PROCESS_ARGUMENTS_CHANGED")
        _require(first["exe_sha256"] == program["exe_sha256"], "PROCESS_PROGRAM_DIGEST_CHANGED")
        _check_pidfd(pidfd, pid)
        _require(capture_process_identity(pid) == first, "SERVICE_PROCESS_IDENTITY_CHANGED")
        _require(_program_observation(pid) == program, "SERVICE_PROCESS_PROGRAM_CHANGED")
        _require(_read_exec_start(pid) == command, "EFFECTIVE_EXECSTART_CHANGED")
        _require(
            _properties(
                systemctl(
                    "show", UNIT_NAME, "--no-pager", "--property=" + ",".join(SYSTEMD_PROPERTIES)
                )
            )
            == before,
            "EFFECTIVE_SERVICE_CHANGED",
        )
        _check_pidfd(pidfd, pid)
    except ServiceIdentityError:
        raise
    except (OSError, ValueError, TypeError, KeyError, ProductionSignerIPCError) as exc:
        raise ServiceIdentityError("SERVICE_IDENTITY_UNAVAILABLE") from exc
    finally:
        if pidfd is not None:
            os.close(pidfd)
