"""S9-RPI-27: read-only, untrusted Linux process executable object fingerprint.

Procfs executable metadata and bytes are NOT trusted executable provenance,
pidfd-bound peer authentication, an installation pin, or production approval.
"""

from __future__ import annotations

from contextlib import ExitStack
import hashlib
import json
import os
import stat
import sys

from deployment.stage9_native_linux_cgroup_membership_observation import _run_systemctl
from deployment.stage9_native_linux_process_identity_observation import (
    _PROPOSED_SYSTEMD_UNITS,
    _SYSTEMD_PROPERTIES,
    _parse_stat,
    _parse_unit_hint,
    _read_proc_file,
)

_MAX_EXE_BYTES = 64 * 1024 * 1024
_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "SYSTEMD_UNAVAILABLE",
        "MALFORMED_SYSTEMD_RESPONSE",
        "PROC_EXE_UNAVAILABLE",
        "EXECUTABLE_NOT_REGULAR_OR_SAFE",
        "EXECUTABLE_TOO_LARGE_OR_EMPTY",
        "EXECUTABLE_OBJECT_CHANGED",
        "PROCESS_CHANGED_DURING_OBSERVATION",
        "INVALID_PROCESS_EXECUTABLE_OBSERVATION",
        "SHARED_SYSTEMD_MAINPID_HINT",
        "PROCESS_OR_SYSTEMD_UNAVAILABLE",
        "BACKEND_OBSERVATION_FAILED",
    }
)
_BLOCKERS = (
    "PROC_EXE_DIGEST_IS_NOT_APPROVED_EXECUTABLE_PIN",
    "SYSTEMD_MAINPID_IS_UNTRUSTED_UNBOUND_HINT",
    "NO_PIDFD_SOCKET_PEER_BINDING_OR_INDEPENDENT_INSTALL_PROVENANCE",
    "PROCESS_NAMESPACE_AND_CROSS_AUTHORITY_ACCESS_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


class _Refusal(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


def _identity(item) -> tuple:
    """Exclude atime because this observer itself reads the executable."""
    return (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_uid,
        item.st_gid,
        item.st_nlink,
        item.st_size,
        item.st_ctime_ns,
        item.st_mtime_ns,
    )


def _safe_executable(item) -> bool:
    return (
        stat.S_ISREG(item.st_mode)
        and item.st_uid == 0
        and item.st_gid == 0
        and item.st_nlink >= 1
        and not (stat.S_IMODE(item.st_mode) & 0o022)
    )


def _read_executable(pid: int) -> dict:
    """Fingerprint only the kernel-reported /proc/PID/exe file object.

    The procfs 'exe' entry is a kernel magic link; it must be followed to
    read executable bytes. No target pathname or its contents are reported.
    This is sequential evidence, not pidfd-based authentication or attestation.
    """
    if sys.platform != "linux":
        raise _Refusal("NATIVE_LINUX_REQUIRED")
    if type(pid) is not int or not 1 < pid < 2**31:
        raise _Refusal("INVALID_PROCESS_EXECUTABLE_OBSERVATION")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK
    try:
        with ExitStack() as descriptors:
            directory = os.open(f"/proc/{pid}", directory_flags)
            descriptors.callback(os.close, directory)
            before_ticks = _parse_stat(_read_proc_file(directory, "stat"), pid)
            # os.stat follows ONLY the fixed kernel procfs executable link.
            before = os.stat("exe", dir_fd=directory)
            if not _safe_executable(before):
                raise _Refusal("EXECUTABLE_NOT_REGULAR_OR_SAFE")
            if not 0 < before.st_size <= _MAX_EXE_BYTES:
                raise _Refusal("EXECUTABLE_TOO_LARGE_OR_EMPTY")
            executable = os.open("exe", file_flags, dir_fd=directory)
            descriptors.callback(os.close, executable)
            opened = os.fstat(executable)
            if _identity(before) != _identity(opened):
                raise _Refusal("EXECUTABLE_OBJECT_CHANGED")

            digest = hashlib.sha256()
            length = 0
            while length <= _MAX_EXE_BYTES:
                chunk = os.read(executable, min(65536, _MAX_EXE_BYTES + 1 - length))
                if not chunk:
                    break
                length += len(chunk)
                if length > _MAX_EXE_BYTES:
                    raise _Refusal("EXECUTABLE_TOO_LARGE_OR_EMPTY")
                digest.update(chunk)

            after = os.fstat(executable)
            link_after = os.stat("exe", dir_fd=directory)
            after_ticks = _parse_stat(_read_proc_file(directory, "stat"), pid)
            if before_ticks != after_ticks:
                raise _Refusal("PROCESS_CHANGED_DURING_OBSERVATION")
            if (
                length != opened.st_size
                or _identity(before) != _identity(after)
                or _identity(before) != _identity(link_after)
            ):
                raise _Refusal("EXECUTABLE_OBJECT_CHANGED")
            return {
                "pid": pid,
                "start_time_ticks": before_ticks,
                "sha256": digest.hexdigest(),
                "size": length,
            }
    except OSError as exc:
        raise _Refusal("PROC_EXE_UNAVAILABLE") from exc


def _sanitize(observed, *, pid: int) -> dict:
    """Copy only fixed-shape builtin primitives from backend observations."""
    if type(observed) is not dict or set(observed) != {
        "pid",
        "start_time_ticks",
        "sha256",
        "size",
    }:
        raise _Refusal("INVALID_PROCESS_EXECUTABLE_OBSERVATION")
    seen_pid = observed["pid"]
    ticks = observed["start_time_ticks"]
    digest = observed["sha256"]
    size = observed["size"]
    if (
        type(seen_pid) is not int
        or seen_pid != pid
        or type(ticks) is not int
        or not 0 < ticks < 2**64
        or type(size) is not int
        or not 0 < size <= _MAX_EXE_BYTES
        or type(digest) is not str
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest)
    ):
        raise _Refusal("INVALID_PROCESS_EXECUTABLE_OBSERVATION")
    return {"pid": pid, "start_time_ticks": ticks, "sha256": digest, "size": size}


def inspect_host(*, runner=None, executable_reader=None) -> dict:
    """Three fixed proposed roles; output never gives a production GO."""
    synthetic = runner is not None or executable_reader is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxProcessExecutableObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "executable_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if runner is None:
        runner = _run_systemctl
    if executable_reader is None:
        executable_reader = _read_executable
    pid_hints = {}
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        try:
            command = (
                "/usr/bin/systemctl",
                "--system",
                "show",
                "--no-pager",
                f"--property={','.join(_SYSTEMD_PROPERTIES)}",
                unit,
            )
            response = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
            if type(response.returncode) is not int:
                raise _Refusal("MALFORMED_SYSTEMD_RESPONSE")
            if response.returncode:
                raise _Refusal("SYSTEMD_UNAVAILABLE")
            try:
                pid = _parse_unit_hint(response.stdout, unit=unit, account=account)
            except ValueError as exc:
                raise _Refusal("MALFORMED_SYSTEMD_RESPONSE") from exc
            # Retain the untrusted metadata even if executable observation fails.
            pid_hints[role] = pid
            item = _sanitize(executable_reader(pid), pid=pid)
            observed = {
                "observation": "UNPINNED_PROC_EXECUTABLE_DIGEST_ONLY",
                "pid_hint": item["pid"],
                "start_time_ticks": item["start_time_ticks"],
                "executable_sha256": item["sha256"],
                "executable_size": item["size"],
            }
        except UnicodeDecodeError:
            observed = {"observation": "NOT_QUALIFIED", "issues": ["MALFORMED_SYSTEMD_RESPONSE"]}
        except _Refusal as exc:
            code = getattr(exc, "code", None) if type(exc) is _Refusal else None
            if type(code) is not str or code not in _CODES:
                code = "BACKEND_OBSERVATION_FAILED"
            observed = {"observation": "NOT_QUALIFIED", "issues": [code]}
        except (OSError, TimeoutError):
            observed = {
                "observation": "NOT_QUALIFIED",
                "issues": ["PROCESS_OR_SYSTEMD_UNAVAILABLE"],
            }
        except Exception:
            observed = {"observation": "NOT_QUALIFIED", "issues": ["BACKEND_OBSERVATION_FAILED"]}
        result["executable_observations"][role] = {"unit": unit, **observed}
        if observed["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].extend(
                f"{role}.executable: {reason}" for reason in observed["issues"]
            )
    # Sequential MainPID hints do not establish a shared process identity.
    for index, left in enumerate(pid_hints):
        for right in list(pid_hints)[index + 1 :]:
            if pid_hints[left] == pid_hints[right]:
                result["observed_failures"].append(f"{left}/{right}: SHARED_SYSTEMD_MAINPID_HINT")
    return result


def main() -> int:
    print(json.dumps(inspect_host(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
