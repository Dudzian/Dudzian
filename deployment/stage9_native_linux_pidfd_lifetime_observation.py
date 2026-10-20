"""S9-RPI-28: untrusted read-only pidfd lifetime observations for proposed Source/CCA.

pidfd_open(MainPID) is NOT SO_PEERPIDFD, service ownership, authenticated IPC
peer identity, or an approved installation pin. This never promotes production.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
import select
import sys

from deployment.stage9_native_linux_cgroup_membership_observation import _run_systemctl
from deployment.stage9_native_linux_process_identity_observation import (
    _PROPOSED_SYSTEMD_UNITS,
    _SYSTEMD_PROPERTIES,
    _parse_stat,
    _parse_unit_hint,
    _read_proc_file,
)

_MAX_FDINFO_BYTES = 4096
_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "MALFORMED_SYSTEMD_RESPONSE",
        "SYSTEMD_UNAVAILABLE",
        "PIDFD_NOT_SUPPORTED",
        "PIDFD_OPEN_UNAVAILABLE",
        "PIDFD_FDINFO_UNAVAILABLE",
        "PIDFD_PROCFS_UNAVAILABLE",
        "PIDFD_CLOSE_FAILED",
        "PIDFD_NOT_LIVE",
        "PIDFD_POLL_FAILED",
        "PIDFD_FDINFO_MALFORMED",
        "PIDFD_PID_NOT_MATCHED",
        "PROCESS_CHANGED_DURING_OBSERVATION",
        "INVALID_PIDFD_LIFETIME_OBSERVATION",
        "SHARED_SYSTEMD_MAINPID_HINT",
        "BACKEND_OBSERVATION_FAILED",
    }
)
_BLOCKERS = (
    "SYSTEMD_MAINPID_IS_AN_UNTRUSTED_HINT",
    "PIDFD_OPEN_IS_NOT_SOCKET_BOUND_SO_PEERPIDFD",
    "PIDFD_AND_PROCFS_OBSERVATIONS_NOT_ATOMICALLY_IDENTITY_BOUND",
    "INDEPENDENTLY_PINNED_UNIT_EXECUTABLE_AND_HOST_NOT_QUALIFIED",
    "CROSS_ROLE_EFFECTIVE_ACCESS_AND_NATIVE_KEYRING_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


class _Refusal(ValueError):
    def __init__(self, code: str):
        self.code = code if type(code) is str and code in _CODES else "BACKEND_OBSERVATION_FAILED"
        super().__init__(self.code)


def _parse_pidfd_fdinfo(raw: bytes, *, pid: int) -> None:
    """Require one exact kernel Pid field, without logging other fdinfo content."""
    if type(raw) is not bytes or not raw or len(raw) > _MAX_FDINFO_BYTES:
        raise _Refusal("PIDFD_FDINFO_MALFORMED")
    if b"\x00" in raw or b"\r" in raw or not raw.endswith(b"\n"):
        raise _Refusal("PIDFD_FDINFO_MALFORMED")
    found = []
    for line in raw.splitlines():
        key, sep, value = line.partition(b":")
        if not sep or not key or not key.isascii() or not key.decode("ascii").isidentifier():
            raise _Refusal("PIDFD_FDINFO_MALFORMED")
        if key == b"Pid":
            if len(value) > 16:
                raise _Refusal("PIDFD_FDINFO_MALFORMED")
            parts = value.split()
            if (
                len(parts) != 1
                or not parts[0].isascii()
                or not parts[0].isdigit()
                or len(parts[0]) > 10
            ):
                raise _Refusal("PIDFD_FDINFO_MALFORMED")
            parsed = int(parts[0])
            if not 1 < parsed < 2**31:
                raise _Refusal("PIDFD_FDINFO_MALFORMED")
            found.append(parsed)
    if len(found) != 1:
        raise _Refusal("PIDFD_FDINFO_MALFORMED")
    if found[0] != pid:
        raise _Refusal("PIDFD_PID_NOT_MATCHED")


def _close_observation_fd(fd: int) -> None:
    """Attempt every registered close, but surface errors as fixed refusals."""
    try:
        os.close(fd)
    except OSError as exc:
        raise _Refusal("PIDFD_CLOSE_FAILED") from exc


def _read_pidfd_fdinfo(pidfd: int, *, pid: int) -> None:
    """Read only bounded kernel metadata for an already opened pidfd."""
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
    try:
        fd = os.open(f"/proc/self/fdinfo/{pidfd}", flags)
    except OSError as exc:
        raise _Refusal("PIDFD_FDINFO_UNAVAILABLE") from exc
    try:
        try:
            data = bytearray()
            while len(data) <= _MAX_FDINFO_BYTES:
                chunk = os.read(fd, min(1024, _MAX_FDINFO_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > _MAX_FDINFO_BYTES:
                    raise _Refusal("PIDFD_FDINFO_MALFORMED")
            _parse_pidfd_fdinfo(bytes(data), pid=pid)
        except OSError as exc:
            raise _Refusal("PIDFD_FDINFO_UNAVAILABLE") from exc
    finally:
        _close_observation_fd(fd)


def _poll_liveness(pidfd: int) -> None:
    """Any pidfd event means unknown/exited; do not assert identity from poll."""
    try:
        watcher = select.poll()
        watcher.register(pidfd, select.POLLIN | select.POLLERR | select.POLLHUP)
        events = watcher.poll(0)
    except (OSError, ValueError) as exc:
        raise _Refusal("PIDFD_POLL_FAILED") from exc
    if type(events) is not list or any(
        type(event) is not tuple
        or len(event) != 2
        or type(event[0]) is not int
        or event[0] != pidfd
        or type(event[1]) is not int
        for event in events
    ):
        raise _Refusal("PIDFD_POLL_FAILED")
    if events:
        raise _Refusal("PIDFD_NOT_LIVE")


def _read_pidfd_lifetime(pid: int) -> dict:
    """Pin a PID lifetime for observations, never a socket-authenticated peer.

    A procfs directory opened by numeric PID is NOT atomically bound to the
    pidfd. This is a best-effort diagnostic; PID reuse/ABA is not ruled out.
    """
    if sys.platform != "linux":
        raise _Refusal("NATIVE_LINUX_REQUIRED")
    if type(pid) is not int or not 1 < pid < 2**31:
        raise _Refusal("INVALID_PIDFD_LIFETIME_OBSERVATION")
    if not callable(getattr(os, "pidfd_open", None)) or not callable(
        getattr(select, "poll", None)
    ):
        raise _Refusal("PIDFD_NOT_SUPPORTED")
    with ExitStack() as descriptors:
        # Only failure of this syscall receives PIDFD_OPEN_UNAVAILABLE.
        try:
            pidfd = os.pidfd_open(pid, 0)
        except OSError as exc:
            raise _Refusal("PIDFD_OPEN_UNAVAILABLE") from exc
        descriptors.callback(_close_observation_fd, pidfd)
        _poll_liveness(pidfd)
        _read_pidfd_fdinfo(pidfd, pid=pid)
        try:
            directory = os.open(
                f"/proc/{pid}",
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
        except OSError as exc:
            raise _Refusal("PIDFD_PROCFS_UNAVAILABLE") from exc
        descriptors.callback(_close_observation_fd, directory)
        try:
            first = _parse_stat(_read_proc_file(directory, "stat"), pid)
            second = _parse_stat(_read_proc_file(directory, "stat"), pid)
        except OSError as exc:
            raise _Refusal("PIDFD_PROCFS_UNAVAILABLE") from exc
        if first != second:
            raise _Refusal("PROCESS_CHANGED_DURING_OBSERVATION")
        _read_pidfd_fdinfo(pidfd, pid=pid)
        _poll_liveness(pidfd)
        return {"pid": pid, "start_time_ticks": first, "pidfd_poll_alive": True}


def _sanitize(observation, *, pid: int) -> dict:
    if type(observation) is not dict or set(observation) != {
        "pid",
        "start_time_ticks",
        "pidfd_poll_alive",
    }:
        raise _Refusal("INVALID_PIDFD_LIFETIME_OBSERVATION")
    observed_pid = observation["pid"]
    ticks = observation["start_time_ticks"]
    alive = observation["pidfd_poll_alive"]
    if (
        type(observed_pid) is not int
        or observed_pid != pid
        or type(ticks) is not int
        or not 0 < ticks < 2**64
        or alive is not True
    ):
        raise _Refusal("INVALID_PIDFD_LIFETIME_OBSERVATION")
    return {"pid": pid, "start_time_ticks": ticks, "pidfd_poll_alive": True}


def inspect_host(*, runner=None, lifetime_reader=None) -> dict:
    """Only the three proposed roles; all outcomes remain NOT_QUALIFIED."""
    synthetic = runner is not None or lifetime_reader is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxPidfdLifetimeObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "pidfd_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if runner is None:
        runner = _run_systemctl
    if lifetime_reader is None:
        lifetime_reader = _read_pidfd_lifetime
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
            pid_hints[role] = pid
            record = _sanitize(lifetime_reader(pid), pid=pid)
            observed = {
                "observation": "UNTRUSTED_PIDFD_LIFETIME_METADATA_ONLY",
                "pid_hint": record["pid"],
                "start_time_ticks": record["start_time_ticks"],
                "pidfd_poll_alive": record["pidfd_poll_alive"],
            }
        except _Refusal as exc:
            code = getattr(exc, "code", None) if type(exc) is _Refusal else None
            if type(code) is not str or code not in _CODES:
                code = "BACKEND_OBSERVATION_FAILED"
            observed = {"observation": "NOT_QUALIFIED", "issues": [code]}
        except (OSError, TimeoutError):
            observed = {"observation": "NOT_QUALIFIED", "issues": ["BACKEND_OBSERVATION_FAILED"]}
        except Exception:
            observed = {"observation": "NOT_QUALIFIED", "issues": ["BACKEND_OBSERVATION_FAILED"]}
        result["pidfd_observations"][role] = {"unit": unit, **observed}
        if observed["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].extend(
                f"{role}.pidfd: {code}" for code in observed["issues"]
            )
    roles = list(pid_hints)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            if pid_hints[left] == pid_hints[right]:
                result["observed_failures"].append(f"{left}/{right}: SHARED_SYSTEMD_MAINPID_HINT")
    return result


def main() -> int:
    print(json.dumps(inspect_host(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
