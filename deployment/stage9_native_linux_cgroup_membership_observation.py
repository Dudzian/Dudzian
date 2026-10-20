"""S9-RPI-25: read-only, untrusted Linux cgroup-v2 membership observation.

Matching cgroup metadata is NOT systemd unit ownership, socket-bound peer
authentication, installation approval or production qualification.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import sys
import time

from deployment.stage9_native_linux_process_identity_observation import (
    _PROPOSED_SYSTEMD_UNITS,
    _SYSTEMD_PROPERTIES,
    _parse_stat,
    _parse_unit_hint,
    _read_proc_file,
)

_PROPERTIES = (*_SYSTEMD_PROPERTIES, "ControlGroup")
_MAX_SYSTEMD_BYTES = 8192
_BLOCKERS = (
    "MAINPID_AND_CONTROLGROUP_ARE_NOT_TRUSTED_SERVICE_PROVENANCE",
    "PIDFD_AND_AUTHENTICATED_PEER_NOT_QUALIFIED",
    "UNIT_EXECUTABLE_INSTALLATION_AND_HOST_PIN_NOT_QUALIFIED",
    "CROSS_AUTHORITY_EFFECTIVE_ACCESS_NOT_QUALIFIED",
    "POSTGRESQL_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)
_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "MALFORMED_SYSTEMD_CGROUP_METADATA",
        "INVALID_SYSTEMD_CGROUP_PATH",
        "SYSTEMD_UNAVAILABLE",
        "OVERSIZED_SYSTEMD_RESPONSE",
        "MALFORMED_PROC_CGROUP_V2",
        "INVALID_PROC_CGROUP_IDENTITY",
        "PROCESS_CHANGED_DURING_OBSERVATION",
        "CGROUP_MEMBERSHIP_NOT_MATCHED",
        "CROSS_ROLE_CGROUP_OVERLAP",
        "SHARED_PROCESS_PID",
        "PROCESS_OR_SYSTEMD_UNAVAILABLE",
        "BACKEND_OBSERVATION_FAILED",
    }
)


class _Refusal(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


def _valid_path(value) -> bool:
    """Reject relative and ambiguous paths, rather than silently normalizing."""
    return (
        type(value) is str
        and 1 < len(value) <= 4096
        and value.startswith("/")
        and value.isascii()
        and all(33 <= ord(c) <= 126 for c in value)
        and all(component not in ("", ".", "..") for component in value[1:].split("/"))
    )


def _parse_unit_cgroup(raw: str, *, unit: str, account: str) -> tuple[int, str]:
    if (
        type(raw) is not str
        or len(raw) > _MAX_SYSTEMD_BYTES
        or not raw.isascii()
        or any(ord(char) < 32 and char != "\n" for char in raw)
    ):
        raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA")
    fields = {}
    for line in raw.splitlines():
        if "=" not in line:
            raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA")
        name, value = line.split("=", 1)
        if name not in _PROPERTIES or name in fields:
            raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA")
        fields[name] = value
    if set(fields) != set(_PROPERTIES):
        raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA")
    path = fields.pop("ControlGroup")
    if not _valid_path(path) or not path.endswith("/" + unit):
        raise _Refusal("INVALID_SYSTEMD_CGROUP_PATH")
    fixed = "\n".join(f"{name}={fields[name]}" for name in _SYSTEMD_PROPERTIES) + "\n"
    try:
        pid = _parse_unit_hint(fixed, unit=unit, account=account)
    except ValueError as exc:
        raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA") from exc
    return pid, path


def _parse_proc_cgroup_v2(raw: bytes) -> str:
    if type(raw) is not bytes or not raw or len(raw) > 8192:
        raise _Refusal("MALFORMED_PROC_CGROUP_V2")
    if raw.count(b"\n") != 1 or not raw.endswith(b"\n") or b"\r" in raw or b"\x00" in raw:
        raise _Refusal("MALFORMED_PROC_CGROUP_V2")
    fields = raw[:-1].split(b":", 2)
    if len(fields) != 3 or fields[0] != b"0" or fields[1] != b"":
        raise _Refusal("MALFORMED_PROC_CGROUP_V2")
    try:
        path = fields[2].decode("ascii")
    except UnicodeDecodeError as exc:
        raise _Refusal("MALFORMED_PROC_CGROUP_V2") from exc
    if not _valid_path(path):
        raise _Refusal("MALFORMED_PROC_CGROUP_V2")
    return path


def _read_process_cgroup(pid: int) -> tuple[int, int, str]:
    """Double-read procfs, without claiming an atomic snapshot."""
    if sys.platform != "linux":
        raise _Refusal("NATIVE_LINUX_REQUIRED")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(f"/proc/{pid}", flags)
    try:
        before = _parse_stat(_read_proc_file(fd, "stat"), pid)
        cg_before = _parse_proc_cgroup_v2(_read_proc_file(fd, "cgroup"))
        after = _parse_stat(_read_proc_file(fd, "stat"), pid)
        cg_after = _parse_proc_cgroup_v2(_read_proc_file(fd, "cgroup"))
        if before != after or cg_before != cg_after:
            raise _Refusal("PROCESS_CHANGED_DURING_OBSERVATION")
        return pid, before, cg_before
    finally:
        os.close(fd)


def _sanitize(observation, *, pid: int) -> tuple[int, str]:
    if type(observation) is not tuple or len(observation) != 3:
        raise _Refusal("INVALID_PROC_CGROUP_IDENTITY")
    observed_pid, start_ticks, path = observation
    if (
        type(observed_pid) is not int
        or observed_pid != pid
        or type(start_ticks) is not int
        or not 0 < start_ticks < 2**64
        or not _valid_path(path)
    ):
        raise _Refusal("INVALID_PROC_CGROUP_IDENTITY")
    return start_ticks, path


def _is_member(path: str, group: str) -> bool:
    """Require a whole component, not an arbitrary string prefix."""
    return path == group or path.startswith(group + "/")


def _run_systemctl(command, *, timeout=3, env=None, **_kwargs):
    """Bound stdout while reading; discard stderr without retaining private data."""
    deadline = time.monotonic() + timeout
    with subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=env,
        bufsize=0,
        close_fds=True,
    ) as process:
        try:
            parts = []
            size = 0
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not selector.select(remaining):
                        raise subprocess.TimeoutExpired(command, timeout)
                    part = os.read(process.stdout.fileno(), _MAX_SYSTEMD_BYTES + 1 - size)
                    if not part:
                        break
                    size += len(part)
                    if size > _MAX_SYSTEMD_BYTES:
                        raise _Refusal("OVERSIZED_SYSTEMD_RESPONSE")
                    parts.append(part)
            returncode = process.wait(timeout=max(0, deadline - time.monotonic()))
            return subprocess.CompletedProcess(
                command, returncode, stdout=b"".join(parts).decode("utf-8"), stderr=None
            )
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def _read_unit_cgroup(*, runner, unit: str, account: str) -> tuple[int, str]:
    command = (
        "/usr/bin/systemctl",
        "--system",
        "show",
        "--no-pager",
        f"--property={','.join(_PROPERTIES)}",
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
        raise _Refusal("MALFORMED_SYSTEMD_CGROUP_METADATA")
    if response.returncode:
        raise _Refusal("SYSTEMD_UNAVAILABLE")
    return _parse_unit_cgroup(response.stdout, unit=unit, account=account)


def inspect_host(*, runner=None, process_reader=None) -> dict:
    """Read fixed candidates only, never return a production GO."""
    synthetic = runner is not None or process_reader is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxCgroupMembershipObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": ("SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION"),
        "blockers": list(_BLOCKERS),
        "process_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if runner is None:
        runner = _run_systemctl
    if process_reader is None:
        process_reader = _read_process_cgroup
    groups = {}
    pids = {}
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        try:
            pid, group = _read_unit_cgroup(runner=runner, unit=unit, account=account)
            # Retain validated hints even if procfs later fails or does not match.
            groups[role] = group
            pids[role] = pid
            start_ticks, observed_group = _sanitize(process_reader(pid), pid=pid)
            if not _is_member(observed_group, group):
                raise _Refusal("CGROUP_MEMBERSHIP_NOT_MATCHED")
            observed = {
                "observation": "CGROUP_MEMBERSHIP_METADATA_ONLY",
                "pid": pid,
                "start_time_ticks": start_ticks,
            }
        except UnicodeDecodeError:
            observed = {
                "observation": "NOT_QUALIFIED",
                "issues": ["MALFORMED_SYSTEMD_CGROUP_METADATA"],
            }
        except _Refusal as exc:
            candidate = getattr(exc, "code", None) if type(exc) is _Refusal else None
            code = (
                candidate
                if type(candidate) is str and candidate in _CODES
                else "BACKEND_OBSERVATION_FAILED"
            )
            observed = {"observation": "NOT_QUALIFIED", "issues": [code]}
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            observed = {
                "observation": "NOT_QUALIFIED",
                "issues": ["PROCESS_OR_SYSTEMD_UNAVAILABLE"],
            }
        except Exception:
            observed = {
                "observation": "NOT_QUALIFIED",
                "issues": ["BACKEND_OBSERVATION_FAILED"],
            }
        result["process_observations"][role] = {"unit": unit, **observed}
        if observed["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].extend(
                f"{role}.cgroup: {reason}" for reason in observed["issues"]
            )
    for index, left in enumerate(groups):
        for right in list(groups)[index + 1 :]:
            a, b = groups[left], groups[right]
            if pids[left] == pids[right]:
                result["observed_failures"].append(f"{left}/{right}: SHARED_PROCESS_PID")
            if _is_member(a, b) or _is_member(b, a):
                result["observed_failures"].append(f"{left}/{right}: CROSS_ROLE_CGROUP_OVERLAP")
    return result


def main() -> int:
    print(json.dumps(inspect_host(), ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
