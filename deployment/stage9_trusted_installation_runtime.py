"""Read-only runtime correlation for the Stage 9 installation identity gate.

These sequential checks correlate a protected installation expectation with
systemd and procfs observations. They do not authenticate an IPC peer, attest
the host, or authorize SIGN/BIND, enrollment, or CCA publication.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import stat
import subprocess
import sys
from typing import cast

from deployment.stage9_native_linux_cgroup_membership_observation import (
    _parse_proc_cgroup_v2,
    _read_process_cgroup,
    _run_systemctl,
)
from deployment.stage9_native_linux_process_executable_observation import (
    _read_executable,
    _sanitize as _sanitize_executable,
)
from deployment.stage9_native_linux_process_identity_observation import (
    _SYSTEMD_PROPERTIES,
    _parse_stat,
    _parse_unit_hint,
    _read_proc_file,
    _read_proc_identity,
    _sanitize_identity,
)
from deployment.stage9_native_linux_unit_fragment_observation import _metadata
from deployment.stage9_trusted_installation_contract import RoleExpectation

_PROPERTIES = (
    *_SYSTEMD_PROPERTIES,
    "FragmentPath",
    "DropInPaths",
    "ControlGroup",
    "Names",
    "DynamicUser",
    "SupplementaryGroups",
    "NeedDaemonReload",
)
_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "SYSTEMD_UNAVAILABLE",
        "SYSTEMD_RESPONSE_INVALID",
        "SYSTEMD_IDENTITY_MISMATCH",
        "UNEXPECTED_SYSTEMD_DROPINS",
        "SYSTEMD_ALIASES_FORBIDDEN",
        "SYSTEMD_CONFIGURATION_CHANGED",
        "NSS_IDENTITY_UNAVAILABLE",
        "NSS_IDENTITY_MISMATCH",
        "PROCESS_UNAVAILABLE",
        "PROCESS_IDENTITY_MISMATCH",
        "PROCESS_CHANGED_DURING_OBSERVATION",
        "PROCESS_CGROUP_MISMATCH",
        "EXECUTABLE_OBJECT_MISMATCH",
        "EXECUTABLE_DIGEST_MISMATCH",
        "RUNTIME_OBSERVATION_FAILED",
    }
)
_ENV = {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}

# NSS modules can perform network lookups. Run them in a bounded child rather
# than letting an unbounded getpwnam/getgrouplist block the observing process.
# Only numeric metadata and a canonical-name comparison leave the child.
_NSS_WORKER = """import grp,json,os,pwd,sys
name = sys.argv[1]
account = pwd.getpwnam(name)
group = grp.getgrnam(name)
canonical = (pwd.getpwuid(account.pw_uid).pw_name == name
             and group.gr_gid == account.pw_gid
             and grp.getgrgid(account.pw_gid).gr_name == name)
print(json.dumps({"uid": account.pw_uid, "gid": account.pw_gid,
                  "groups": sorted(set(os.getgrouplist(name, account.pw_gid))),
                  "canonical": canonical}, separators=(",", ":")))
"""


class RuntimeRefusal(ValueError):
    """Only fixed public reason codes may cross the reporting boundary."""

    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "RUNTIME_OBSERVATION_FAILED"
        super().__init__(self.code)


def _run(command: tuple[str, ...]):
    return _run_systemctl(command, timeout=3, env=_ENV)


def _account(expectation: RoleExpectation) -> dict:
    try:
        response = _run((sys.executable, "-I", "-c", _NSS_WORKER, expectation.principal))
        if type(response.returncode) is not int or response.returncode != 0:
            raise RuntimeRefusal("NSS_IDENTITY_UNAVAILABLE")
        if type(response.stdout) is not str or not 0 < len(response.stdout) <= 8192:
            raise RuntimeRefusal("NSS_IDENTITY_UNAVAILABLE")
        observed = json.loads(response.stdout)
        if (
            type(observed) is not dict
            or set(observed) != {"uid", "gid", "groups", "canonical"}
            or type(observed["uid"]) is not int
            or type(observed["gid"]) is not int
            or observed["uid"] != expectation.uid
            or observed["gid"] != expectation.gid
            or observed["canonical"] is not True
            or type(observed["groups"]) is not list
            or any(type(value) is not int for value in observed["groups"])
            or observed["groups"] != [expectation.gid]
            or expectation.supplementary_groups != ()
        ):
            raise RuntimeRefusal("NSS_IDENTITY_MISMATCH")
        return observed
    except RuntimeRefusal:
        raise
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        raise RuntimeRefusal("NSS_IDENTITY_UNAVAILABLE") from exc


def _unit(expectation: RoleExpectation) -> tuple[int, dict[str, str]]:
    try:
        response = _run(
            (
                "/usr/bin/systemctl",
                "--system",
                "show",
                "--no-pager",
                f"--property={','.join(_PROPERTIES)}",
                expectation.unit,
            )
        )
        if type(response.returncode) is not int or response.returncode != 0:
            raise RuntimeRefusal("SYSTEMD_UNAVAILABLE")
        raw = response.stdout
        if (
            type(raw) is not str
            or not 0 < len(raw) <= 8192
            or not raw.isascii()
            or not raw.endswith("\n")
            or any(ord(char) < 32 and char != "\n" for char in raw)
        ):
            raise RuntimeRefusal("SYSTEMD_RESPONSE_INVALID")
        fields = {}
        for line in raw[:-1].split("\n"):
            key, separator, value = line.partition("=")
            if not separator or key not in _PROPERTIES or key in fields:
                raise RuntimeRefusal("SYSTEMD_RESPONSE_INVALID")
            fields[key] = value
        if set(fields) != set(_PROPERTIES):
            raise RuntimeRefusal("SYSTEMD_RESPONSE_INVALID")
        if fields["DropInPaths"]:
            raise RuntimeRefusal("UNEXPECTED_SYSTEMD_DROPINS")
        if fields["Names"] != expectation.unit:
            raise RuntimeRefusal("SYSTEMD_ALIASES_FORBIDDEN")
        if fields["NeedDaemonReload"] != "no":
            raise RuntimeRefusal("SYSTEMD_CONFIGURATION_CHANGED")
        for key, value in (
            ("FragmentPath", expectation.unit_file.path),
            ("ControlGroup", expectation.control_group),
            ("DynamicUser", "no"),
            ("SupplementaryGroups", ""),
        ):
            if fields[key] != value:
                raise RuntimeRefusal("SYSTEMD_IDENTITY_MISMATCH")
        hint = "".join(f"{key}={fields[key]}\n" for key in _SYSTEMD_PROPERTIES)
        try:
            pid = _parse_unit_hint(hint, unit=expectation.unit, account=expectation.principal)
        except ValueError as exc:
            raise RuntimeRefusal("SYSTEMD_IDENTITY_MISMATCH") from exc
        return pid, fields
    except RuntimeRefusal:
        raise
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        raise RuntimeRefusal("SYSTEMD_UNAVAILABLE") from exc


def _installed_executable(expectation: RoleExpectation, executable_fd: int):
    info = os.fstat(executable_fd)
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or (info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
        != (expectation.executable.uid, expectation.executable.gid, expectation.executable.mode)
        or not 0 < info.st_size <= 64 * 1024 * 1024
    ):
        raise RuntimeRefusal("EXECUTABLE_OBJECT_MISMATCH")
    return info


def _correlate_process(
    expectation: RoleExpectation, executable_fd: int, pid: int, directory: int, installed
) -> dict:
    before = _parse_stat(_read_proc_file(directory, "stat"), pid)
    if os.readlink("exe", dir_fd=directory) != expectation.executable.path:
        raise RuntimeRefusal("EXECUTABLE_OBJECT_MISMATCH")
    if _metadata(os.stat("exe", dir_fd=directory)) != _metadata(installed):
        raise RuntimeRefusal("EXECUTABLE_OBJECT_MISMATCH")
    identity = _sanitize_identity(
        _read_proc_identity(pid, expectation.uid, expectation.gid),
        pid=pid,
        uid=expectation.uid,
        gid=expectation.gid,
    )
    executable = _sanitize_executable(_read_executable(pid), pid=pid)
    if executable["sha256"] != expectation.executable.sha256:
        raise RuntimeRefusal("EXECUTABLE_DIGEST_MISMATCH")
    if executable["size"] != installed.st_size:
        raise RuntimeRefusal("EXECUTABLE_OBJECT_MISMATCH")
    cgroup_pid, cgroup_ticks, cgroup = _read_process_cgroup(pid)
    retained_cgroup = _parse_proc_cgroup_v2(_read_proc_file(directory, "cgroup"))
    if cgroup_pid != pid or cgroup != expectation.control_group or cgroup != retained_cgroup:
        raise RuntimeRefusal("PROCESS_CGROUP_MISMATCH")
    after = _parse_stat(_read_proc_file(directory, "stat"), pid)
    if {
        before,
        after,
        identity["start_time_ticks"],
        executable["start_time_ticks"],
        cgroup_ticks,
    } != {before}:
        raise RuntimeRefusal("PROCESS_CHANGED_DURING_OBSERVATION")
    if (
        os.readlink("exe", dir_fd=directory) != expectation.executable.path
        or _metadata(os.stat("exe", dir_fd=directory)) != _metadata(installed)
        or _metadata(os.fstat(executable_fd)) != _metadata(installed)
    ):
        raise RuntimeRefusal("EXECUTABLE_OBJECT_MISMATCH")
    return cast(dict, identity)


def observe_service(expectation: RoleExpectation, executable_fd: int) -> dict:
    """Correlate two observations without claiming atomicity or peer identity.

    The caller retains and checks the approved executable descriptor and its
    path bindings. A PID procfs directory is retained across both observations;
    every independently opened existing observer must match its start time.
    No pidfd_open result is treated as SO_PEERPIDFD authentication.
    """
    if sys.platform != "linux":
        raise RuntimeRefusal("NATIVE_LINUX_REQUIRED")
    try:
        installed = _installed_executable(expectation, executable_fd)
        account = _account(expectation)
        pid, unit = _unit(expectation)
        with ExitStack() as descriptors:
            directory = os.open(
                f"/proc/{pid}", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            descriptors.callback(os.close, directory)
            first = _correlate_process(expectation, executable_fd, pid, directory, installed)
            second = _correlate_process(expectation, executable_fd, pid, directory, installed)
            if first != second:
                raise RuntimeRefusal("PROCESS_CHANGED_DURING_OBSERVATION")
            if _unit(expectation) != (pid, unit):
                raise RuntimeRefusal("SYSTEMD_CONFIGURATION_CHANGED")
            if _account(expectation) != account:
                raise RuntimeRefusal("NSS_IDENTITY_MISMATCH")
        return first
    except RuntimeRefusal:
        raise
    except OSError as exc:
        raise RuntimeRefusal("PROCESS_UNAVAILABLE") from exc
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise RuntimeRefusal("PROCESS_IDENTITY_MISMATCH") from exc
