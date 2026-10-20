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
_REFUSAL_CODES = frozenset(
    {
        "NATIVE_LINUX_REQUIRED",
        "NSS_ACCOUNT_NOT_OBSERVED",
        "NSS_LOOKUP_UNAVAILABLE",
        "INVALID_NSS_IDENTITY",
        "MALFORMED_SYSTEMD_RESPONSE",
        "INCOMPLETE_SYSTEMD_RESPONSE",
        "INVALID_MAIN_PID",
        "SYSTEMD_UNAVAILABLE",
        "MALFORMED_PROC_STAT",
        "PID_REUSED_OR_CHANGED",
        "PROCESS_EXITED_OR_MALFORMED",
        "MALFORMED_PROC_STATUS",
        "DUPLICATE_PROC_STATUS_FIELD",
        "INCOMPLETE_PROC_STATUS",
        "PROCESS_PID_NOT_MATCHED",
        "PROCESS_UID_GID_NOT_MATCHED",
        "PROCESS_GROUPS_NOT_ISOLATED",
        "NO_NEW_PRIVILEGES_NOT_OBSERVED",
        "MALFORMED_PROC_CAPABILITIES",
        "OVERSIZED_PROC_RESPONSE",
        "EMPTY_PROC_RESPONSE",
        "NAMESPACE_NOT_OBSERVABLE",
        "PROCESS_CHANGED_DURING_OBSERVATION",
        "INVALID_PROC_IDENTITY",
        "PROCESS_OR_SYSTEMD_UNAVAILABLE",
        "BACKEND_OBSERVATION_FAILED",
        *(f"{field}_NOT_MATCHED" for field in _SYSTEMD_PROPERTIES if field != "MainPID"),
        *(f"{field.upper()}_NOT_EMPTY" for field in _CAP_FIELDS),
    }
)


def _refusal_code(value) -> str:
    if type(value) is str and value in _REFUSAL_CODES:
        return value
    return "BACKEND_OBSERVATION_FAILED"


class _ObservationRefusal(ValueError):
    """Local parser refusal; backend exception messages are never report data."""

    def __init__(self, code: str):
        self.code = _refusal_code(code)
        super().__init__(self.code)


def _blocked(reason: str) -> dict:
    return {"observation": "NOT_QUALIFIED", "issues": [_refusal_code(reason)]}


def _bounded_int(value, *, minimum=1, maximum=2**64 - 1) -> bool:
    return type(value) is int and minimum <= value <= maximum


def _sanitize_identity(identity, *, pid: int, uid: int, gid: int) -> dict:
    """Copy only bounded builtin numeric observations from either backend."""
    keys = {"pid", "start_time_ticks", "uid", "gid", "groups", "namespaces"}
    if (
        type(identity) is not dict
        or len(identity) != len(keys)
        or any(type(key) is not str for key in identity)
        or set(identity) != keys
    ):
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    observed_pid = identity["pid"]
    start_time = identity["start_time_ticks"]
    observed_uid, observed_gid = identity["uid"], identity["gid"]
    if not (
        _bounded_int(observed_pid, minimum=2, maximum=2**31 - 1)
        and _bounded_int(start_time)
        and _bounded_int(observed_uid, maximum=2**32 - 2)
        and _bounded_int(observed_gid, maximum=2**32 - 2)
    ):
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    if observed_pid != pid:
        raise _ObservationRefusal("PROCESS_PID_NOT_MATCHED")
    if observed_uid != uid or observed_gid != gid:
        raise _ObservationRefusal("PROCESS_UID_GID_NOT_MATCHED")
    groups = identity["groups"]
    if type(groups) is not list or len(groups) > 1:
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    groups = groups.copy()
    if len(groups) > 1:
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    if any(not _bounded_int(group, maximum=2**32 - 2) for group in groups):
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    if any(group != gid for group in groups):
        raise _ObservationRefusal("PROCESS_GROUPS_NOT_ISOLATED")
    namespaces = identity["namespaces"]
    if (
        type(namespaces) is not dict
        or len(namespaces) != len(_NS_TYPES)
        or any(type(key) is not str for key in namespaces)
        or set(namespaces) != set(_NS_TYPES)
    ):
        raise _ObservationRefusal("INVALID_PROC_IDENTITY")
    safe_namespaces = {}
    for name in _NS_TYPES:
        inode = namespaces[name]
        if not _bounded_int(inode):
            raise _ObservationRefusal("INVALID_PROC_IDENTITY")
        safe_namespaces[name] = inode
    return {
        "pid": observed_pid,
        "start_time_ticks": start_time,
        "uid": observed_uid,
        "gid": observed_gid,
        "groups": groups,
        "namespaces": safe_namespaces,
    }


def _parse_unit_hint(raw: str, *, unit: str, account: str) -> int:
    """Return numeric PID hint only for a complete, exact unit response."""
    if type(raw) is not str or len(raw) > 8192 or "\x00" in raw:
        raise _ObservationRefusal("MALFORMED_SYSTEMD_RESPONSE")
    properties = {}
    for line in raw.splitlines():
        if "=" not in line:
            raise _ObservationRefusal("MALFORMED_SYSTEMD_RESPONSE")
        key, value = line.split("=", 1)
        if key not in _SYSTEMD_PROPERTIES or key in properties:
            raise _ObservationRefusal("MALFORMED_SYSTEMD_RESPONSE")
        properties[key] = value
    if set(properties) != set(_SYSTEMD_PROPERTIES):
        raise _ObservationRefusal("INCOMPLETE_SYSTEMD_RESPONSE")
    for key, expected in (
        ("Id", unit),
        ("LoadState", "loaded"),
        ("ActiveState", "active"),
        ("User", account),
        ("Group", account),
    ):
        if properties[key] != expected:
            raise _ObservationRefusal(f"{key}_NOT_MATCHED")
    value = properties["MainPID"]
    if not value.isascii() or not value.isdecimal() or len(value) > 10:
        raise _ObservationRefusal("INVALID_MAIN_PID")
    pid = int(value)
    if pid <= 1 or pid >= 2**31:
        raise _ObservationRefusal("INVALID_MAIN_PID")
    return pid


def _parse_stat(raw: bytes, pid: int) -> int:
    """Linux procfs stat: field 22 is index 19 after '(comm)'."""
    if type(raw) is not bytes or len(raw) > 65536:
        raise _ObservationRefusal("MALFORMED_PROC_STAT")
    close = raw.rfind(b")")
    if close < 2 or raw[close + 1 : close + 2] != b" ":
        raise _ObservationRefusal("MALFORMED_PROC_STAT")
    if raw[: raw.find(b" ")] != str(pid).encode("ascii"):
        raise _ObservationRefusal("PID_REUSED_OR_CHANGED")
    fields = raw[close + 2 :].split()
    if len(fields) < 20 or fields[0] in (b"Z", b"X", b"x"):
        raise _ObservationRefusal("PROCESS_EXITED_OR_MALFORMED")
    if len(fields[19]) > 20 or not fields[19].isdigit():
        raise _ObservationRefusal("MALFORMED_PROC_STAT")
    start_time = int(fields[19])
    if not _bounded_int(start_time):
        raise _ObservationRefusal("MALFORMED_PROC_STAT")
    return start_time


def _parse_status(raw: bytes, *, pid: int, uid: int, gid: int) -> dict:
    if type(raw) is not bytes or len(raw) > 65536:
        raise _ObservationRefusal("MALFORMED_PROC_STATUS")
    required = {"Pid", "Tgid", "Uid", "Gid", "Groups", "NoNewPrivs", *_CAP_FIELDS}
    fields = {}
    for line in raw.splitlines():
        key, sep, value = line.partition(b":")
        if not sep:
            continue
        try:
            name = key.decode("ascii")
        except UnicodeDecodeError as exc:
            raise _ObservationRefusal("MALFORMED_PROC_STATUS") from exc
        if name in required:
            if name in fields:
                raise _ObservationRefusal("DUPLICATE_PROC_STATUS_FIELD")
            fields[name] = value.strip().split()
    if set(fields) != required:
        raise _ObservationRefusal("INCOMPLETE_PROC_STATUS")
    try:
        observed_pid = tuple(int(item) for item in fields["Pid"])
        observed_tgid = tuple(int(item) for item in fields["Tgid"])
        uids = tuple(int(item) for item in fields["Uid"])
        gids = tuple(int(item) for item in fields["Gid"])
        groups = tuple(int(item) for item in fields["Groups"])
    except ValueError as exc:
        raise _ObservationRefusal("MALFORMED_PROC_STATUS") from exc
    if observed_pid != (pid,) or observed_tgid != (pid,):
        raise _ObservationRefusal("PROCESS_PID_NOT_MATCHED")
    if uids != (uid,) * 4 or gids != (gid,) * 4:
        raise _ObservationRefusal("PROCESS_UID_GID_NOT_MATCHED")
    if set(groups) - {gid} or any(item < 0 for item in groups):
        raise _ObservationRefusal("PROCESS_GROUPS_NOT_ISOLATED")
    if fields["NoNewPrivs"] != [b"1"]:
        raise _ObservationRefusal("NO_NEW_PRIVILEGES_NOT_OBSERVED")
    for key in _CAP_FIELDS:
        values = fields[key]
        if len(values) != 1 or len(values[0]) > 16:
            raise _ObservationRefusal("MALFORMED_PROC_CAPABILITIES")
        try:
            capabilities = int(values[0], 16)
        except ValueError as exc:
            raise _ObservationRefusal("MALFORMED_PROC_CAPABILITIES") from exc
        if capabilities != 0:
            raise _ObservationRefusal(f"{key.upper()}_NOT_EMPTY")
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
                raise _ObservationRefusal("OVERSIZED_PROC_RESPONSE")
        if size == 0:
            raise _ObservationRefusal("EMPTY_PROC_RESPONSE")
        return b"".join(parts)
    finally:
        os.close(fd)


def _read_namespaces(directory: int) -> dict:
    namespaces = {}
    for name in _NS_TYPES:
        info = os.stat(f"ns/{name}", dir_fd=directory)
        if not stat.S_ISREG(info.st_mode) or not _bounded_int(info.st_ino):
            raise _ObservationRefusal("NAMESPACE_NOT_OBSERVABLE")
        namespaces[name] = info.st_ino
    return namespaces


def _read_proc_identity(pid: int, uid: int, gid: int) -> dict:
    """Repeat starttime, credentials and namespace observations to detect churn.

    Sequential rechecks do not make the snapshot atomic or detect every change.
    This is not an authenticated pidfd or proof the process belongs to systemd.
    """
    if sys.platform != "linux":
        raise _ObservationRefusal("NATIVE_LINUX_REQUIRED")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    directory = os.open(f"/proc/{pid}", flags)
    try:
        before = _parse_stat(_read_proc_file(directory, "stat"), pid)
        status = _parse_status(_read_proc_file(directory, "status"), pid=pid, uid=uid, gid=gid)
        namespaces = _read_namespaces(directory)
        after_status = _parse_status(
            _read_proc_file(directory, "status"), pid=pid, uid=uid, gid=gid
        )
        after = _parse_stat(_read_proc_file(directory, "stat"), pid)
        after_namespaces = _read_namespaces(directory)
        if before != after or status != after_status or namespaces != after_namespaces:
            raise _ObservationRefusal("PROCESS_CHANGED_DURING_OBSERVATION")
    finally:
        os.close(directory)
    return {"pid": pid, "start_time_ticks": before, **status, "namespaces": namespaces}


def _inspect_role(*, runner, process_reader, accounts, unit: str, account: str) -> dict:
    if type(accounts) is not dict:
        raise _ObservationRefusal("INVALID_NSS_IDENTITY")
    record = accounts.get(account)
    if record is None:
        raise _ObservationRefusal("NSS_ACCOUNT_NOT_OBSERVED")
    if type(record) not in (tuple, list) or len(record) != 3:
        raise _ObservationRefusal("INVALID_NSS_IDENTITY")
    uid, gid, _groups = record
    if not (_bounded_int(uid, maximum=2**32 - 2) and _bounded_int(gid, maximum=2**32 - 2)):
        raise _ObservationRefusal("INVALID_NSS_IDENTITY")
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
        raise _ObservationRefusal("MALFORMED_SYSTEMD_RESPONSE")
    if response.returncode:
        raise _ObservationRefusal("SYSTEMD_UNAVAILABLE")
    pid = _parse_unit_hint(response.stdout, unit=unit, account=account)
    identity = _sanitize_identity(process_reader(pid, uid, gid), pid=pid, uid=uid, gid=gid)
    return {"observation": "PROCESS_METADATA_ONLY", "identity": identity}


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
    nss_unavailable = False
    if accounts is None:
        try:
            accounts = _native_accounts()
        except Exception:
            nss_unavailable = True
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        try:
            if nss_unavailable:
                raise _ObservationRefusal("NSS_LOOKUP_UNAVAILABLE")
            observed = _inspect_role(
                runner=runner,
                process_reader=process_reader,
                accounts=accounts,
                unit=unit,
                account=account,
            )
        except UnicodeDecodeError:
            observed = _blocked("MALFORMED_SYSTEMD_RESPONSE")
        except _ObservationRefusal as exc:
            code = getattr(exc, "code", None) if type(exc) is _ObservationRefusal else None
            observed = _blocked(code)
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            observed = _blocked("PROCESS_OR_SYSTEMD_UNAVAILABLE")
        except Exception:
            observed = _blocked("BACKEND_OBSERVATION_FAILED")
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
