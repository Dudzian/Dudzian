"""Read-only TEST/host observation of proposed Linux Source/CCA process identities.

No caller input can approve installation or Root Proof issuance. systemd's
MainPID is only an untrusted hint and is NOT socket-bound peer authentication.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys

from deployment.stage9_native_linux_source_cca_preflight import (
    _PROPOSED_SYSTEMD_UNITS,
    _native_accounts,
)

_NS_TYPES = ("mnt", "ipc", "net")
_CAP_FIELDS = ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
_SYSTEMD_PROPERTIES = ("Id", "LoadState", "ActiveState", "User", "Group", "MainPID")
_BLOCKERS = (
    "INDEPENDENTLY_APPROVED_INSTALLATION_PIN_MISSING",
    "MAINPID_IS_NOT_SOCKET_BOUND_PIDFD_OR_TRUSTED_PROCESS_PROVENANCE",
    "CROSS_ROLE_EFFECTIVE_ACCESS_AND_PRIVATE_KEY_CONTROL_NOT_QUALIFIED",
    "SYSTEMD_UNIT_BYTES_AND_PROCESS_EXECUTABLE_NOT_PINNED",
    "SOURCE_CCA_POSTGRESQL_COMMITTED_CLOSE_AND_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


def _blocked(reason: str) -> dict:
    return {"observation": "NOT_QUALIFIED", "issues": [reason]}


def _parse_unit_hint(raw: str, *, unit: str, account: str) -> int:
    """Return numeric PID hint only for a complete, exact unit response."""
    if type(raw) is not str or len(raw) > 8192 or "\x00" in raw:
        raise ValueError("MALFORMED_SYSTEMD_RESPONSE")
    properties = {}
    for line in raw.splitlines():
        if "=" not in line:
            raise ValueError("MALFORMED_SYSTEMD_RESPONSE")
        key, value = line.split("=", 1)
        if key not in _SYSTEMD_PROPERTIES or key in properties:
            raise ValueError("MALFORMED_SYSTEMD_RESPONSE")
        properties[key] = value
    if set(properties) != set(_SYSTEMD_PROPERTIES):
        raise ValueError("INCOMPLETE_SYSTEMD_RESPONSE")
    for key, expected in (
        ("Id", unit),
        ("LoadState", "loaded"),
        ("ActiveState", "active"),
        ("User", account),
        ("Group", account),
    ):
        if properties[key] != expected:
            raise ValueError(f"{key}_NOT_MATCHED")
    value = properties["MainPID"]
    if not value.isascii() or not value.isdecimal() or len(value) > 10:
        raise ValueError("INVALID_MAIN_PID")
    pid = int(value)
    if pid <= 1 or pid >= 2**31:
        raise ValueError("INVALID_MAIN_PID")
    return pid


def _parse_stat(raw: bytes, pid: int) -> int:
    """Linux procfs stat: field 22 is index 19 after '(comm)'."""
    if type(raw) is not bytes or len(raw) > 65536:
        raise ValueError("MALFORMED_PROC_STAT")
    close = raw.rfind(b")")
    if close < 2 or raw[close + 1 : close + 2] != b" ":
        raise ValueError("MALFORMED_PROC_STAT")
    if raw[: raw.find(b" ")] != str(pid).encode("ascii"):
        raise ValueError("PID_REUSED_OR_CHANGED")
    fields = raw[close + 2 :].split()
    if len(fields) < 20 or fields[0] in (b"Z", b"X", b"x"):
        raise ValueError("PROCESS_EXITED_OR_MALFORMED")
    if not fields[19].isascii() or not fields[19].isdecimal():
        raise ValueError("MALFORMED_PROC_STAT")
    start_time = int(fields[19])
    if start_time <= 0:
        raise ValueError("MALFORMED_PROC_STAT")
    return start_time


def _parse_status(raw: bytes, *, pid: int, uid: int, gid: int) -> dict:
    if type(raw) is not bytes or len(raw) > 65536:
        raise ValueError("MALFORMED_PROC_STATUS")
    required = {"Pid", "Tgid", "Uid", "Gid", "Groups", "NoNewPrivs", *_CAP_FIELDS}
    fields = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(b":")
        if not sep:
            continue
        try:
            name = key.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("MALFORMED_PROC_STATUS") from exc
        if name in required:
            if name in fields:
                raise ValueError("DUPLICATE_PROC_STATUS_FIELD")
            fields[name] = value.strip().split()
    if set(fields) != required:
        raise ValueError("INCOMPLETE_PROC_STATUS")
    try:
        observed_pid = tuple(int(item) for item in fields["Pid"])
        observed_tgid = tuple(int(item) for item in fields["Tgid"])
        uids = tuple(int(item) for item in fields["Uid"])
        gids = tuple(int(item) for item in fields["Gid"])
        groups = tuple(int(item) for item in fields["Groups"])
    except ValueError as exc:
        raise ValueError("MALFORMED_PROC_STATUS") from exc
    if observed_pid != (pid,) or observed_tgid != (pid,):
        raise ValueError("PROCESS_PID_NOT_MATCHED")
    if uids != (uid,) * 4 or gids != (gid,) * 4:
        raise ValueError("PROCESS_UID_GID_NOT_MATCHED")
    if set(groups) - {gid} or any(item < 0 for item in groups):
        raise ValueError("PROCESS_GROUPS_NOT_ISOLATED")
    if fields["NoNewPrivs"] != [b"1"]:
        raise ValueError("NO_NEW_PRIVILEGES_NOT_OBSERVED")
    for key in _CAP_FIELDS:
        values = fields[key]
        if len(values) != 1 or len(values[0]) > 16:
            raise ValueError("MALFORMED_PROC_CAPABILITIES")
        try:
            capabilities = int(values[0], 16)
        except ValueError as exc:
            raise ValueError("MALFORMED_PROC_CAPABILITIES") from exc
        if capabilities != 0:
            raise ValueError(f"{key.upper()}_NOT_EMPTY")
    return {"uid": uid, "gid": gid, "groups": sorted(set(groups))}


def _read_proc_file(directory: int, name: str) -> bytes:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, dir_fd=directory)
    try:
        parts = []
        size = 0
        while True:
            part = os.read(fd, min(8192, 65537 - size))
            if not part:
                break
            parts.append(part)
            size += len(part)
            if size > 65536:
                raise ValueError("OVERSIZED_PROC_RESPONSE")
        if size == 0:
            raise ValueError("EMPTY_PROC_RESPONSE")
        return b"".join(parts)
    finally:
        os.close(fd)


def _read_proc_identity(pid: int, uid: int, gid: int) -> dict:
    """Observe pinned /proc directory; verify starttime twice to detect churn.

    This is not an authenticated pidfd or proof the process belongs to systemd.
    """
    if sys.platform != "linux":
        raise ValueError("NATIVE_LINUX_REQUIRED")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(f"/proc/{pid}", flags)
    try:
        before = _parse_stat(_read_proc_file(directory, "stat"), pid)
        status = _parse_status(_read_proc_file(directory, "status"), pid=pid, uid=uid, gid=gid)
        namespaces = {}
        for name in _NS_TYPES:
            info = os.stat(f"ns/{name}", dir_fd=directory)
            if not stat.S_ISREG(info.st_mode) or info.st_ino <= 0:
                raise ValueError("NAMESPACE_NOT_OBSERVABLE")
            namespaces[name] = info.st_ino
        after_status = _parse_status(
            _read_proc_file(directory, "status"), pid=pid, uid=uid, gid=gid
        )
        after = _parse_stat(_read_proc_file(directory, "stat"), pid)
        if before != after or status != after_status:
            raise ValueError("PROCESS_CHANGED_DURING_OBSERVATION")
    finally:
        os.close(directory)
    return {"pid": pid, "start_time_ticks": before, **status, "namespaces": namespaces}


def inspect_host(*, runner=None, process_reader=None, accounts=None) -> dict:
    """Diagnostics only. Injected backends are clearly labeled as synthetic."""
    test_hooks = runner is not None or process_reader is not None or accounts is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxProcessIdentityObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if test_hooks else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "process_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if runner is None:
        runner = subprocess.run
    if process_reader is None:
        process_reader = _read_proc_identity
    if accounts is None:
        accounts = _native_accounts()
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        record = accounts.get(account)
        if record is None:
            observed = _blocked("NSS_ACCOUNT_NOT_OBSERVED")
        else:
            uid, gid, _groups = record
            if type(uid) is not int or type(gid) is not int or uid <= 0 or gid <= 0:
                observed = _blocked("INVALID_NSS_IDENTITY")
            else:
                command = (
                    "/usr/bin/systemctl",
                    "--system",
                    "show",
                    "--no-pager",
                    f"--property={','.join(_SYSTEMD_PROPERTIES)}",
                    unit,
                )
                try:
                    response = runner(
                        command,
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=3,
                        env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
                    )
                    if response.returncode:
                        raise ValueError("SYSTEMD_UNAVAILABLE")
                    pid = _parse_unit_hint(response.stdout, unit=unit, account=account)
                    identity = process_reader(pid, uid, gid)
                    if not isinstance(identity, dict) or set(identity) != {
                        "pid", "start_time_ticks", "uid", "gid", "groups", "namespaces"
                    }:
                        raise ValueError("INCOMPLETE_PROC_IDENTITY")
                    observed = {"observation": "PROCESS_METADATA_ONLY", "identity": identity}
                except UnicodeDecodeError:
                    observed = _blocked("MALFORMED_SYSTEMD_RESPONSE")
                except ValueError as exc:
                    observed = _blocked(str(exc))
                except (OSError, TimeoutError, subprocess.TimeoutExpired):
                    observed = _blocked("PROCESS_OR_SYSTEMD_UNAVAILABLE")
        result["process_observations"][role] = {"unit": unit, **observed}
        if observed["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].extend(
                f"{role}.process: {reason}" for reason in observed["issues"]
            )
    identities = {
        role: row["identity"]
        for role, row in result["process_observations"].items()
        if row["observation"] == "PROCESS_METADATA_ONLY"
    }
    for left_index, left in enumerate(identities):
        for right in list(identities)[left_index + 1 :]:
            a, b = identities[left], identities[right]
            if a["pid"] == b["pid"]:
                result["observed_failures"].append(f"{left}/{right}: SHARED_PROCESS_PID")
            if a["uid"] == b["uid"] or a["gid"] == b["gid"]:
                result["observed_failures"].append(f"{left}/{right}: SHARED_UID_OR_GID")
            if any(a["namespaces"][name] == b["namespaces"][name] for name in _NS_TYPES):
                result["observed_failures"].append(f"{left}/{right}: SHARED_PROCESS_NAMESPACE")
    return result


def main() -> int:
    print(json.dumps(inspect_host(), ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
