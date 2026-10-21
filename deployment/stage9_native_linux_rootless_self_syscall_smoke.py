"""S9-RPI-37: rootless native Linux ephemeral self-syscall smoke.

Real syscalls on freshly created TEST-only private files. All observations
remain untrusted; single-UID DAC behavior cannot prove cross-role isolation.
"""

from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import sys
import tempfile

_SCHEMA = "CryptoHunter.Stage9RootlessSelfSyscallSmokeV1"
_NAMES = (
    "owner_read",
    "owner_write",
    "owner_traverse",
    "denied_read",
    "denied_write",
    "denied_traverse",
    "missing_read",
)
_KNOWN_OUTCOMES = frozenset(
    {"SUCCESS", "DENIED_EACCES", "DENIED_EPERM", "ENOENT", "OTHER_FAILURE"}
)
_DENIED = frozenset({"DENIED_EACCES", "DENIED_EPERM"})
_BLOCKERS = (
    "EPHEMERAL_SAME_UID_SYSCALLS_ARE_NOT_CROSS_ROLE_DENIAL_PROOF",
    "NO_INDEPENDENT_PROCESS_UID_GID_OR_RESOURCE_PROVENANCE",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NO_APPROVED_SOURCE_CCA_SYSTEMD_OR_KEYRING_INSTALLATION",
    "NO_POSTGRESQL_COMMITTED_CLOSE_OR_CCA_DURABILITY",
    "PRODUCTION_ROOT_PROOF_ISSUANCE_AND_ENROLLMENT_BLOCKED",
)


def _classify(value: object) -> str:
    """Never forward exception details, errno strings or foreign data."""
    if type(value) is int:
        if value == errno.EACCES:
            return "DENIED_EACCES"
        if value == errno.EPERM:
            return "DENIED_EPERM"
        if value == errno.ENOENT:
            return "ENOENT"
    return "OTHER_FAILURE"


def _attempt(path: Path, operation: str) -> str:
    """Perform only a one-byte TEST read/write; do not follow the leaf symlink."""
    assert operation in ("read", "write")
    flags = os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK
    flags |= os.O_RDONLY if operation == "read" else os.O_WRONLY
    fd = -1
    try:
        fd = os.open(path, flags)
        if operation == "read":
            return "SUCCESS" if os.read(fd, 1) == b"T" else "OTHER_FAILURE"
        return "SUCCESS" if os.write(fd, b"T") == 1 else "OTHER_FAILURE"
    except OSError as exc:
        return _classify(exc.errno)
    finally:
        if fd >= 0:
            os.close(fd)


def _exercise() -> dict:
    """The sole real-I/O backend creates and removes its own private TEST root."""
    with tempfile.TemporaryDirectory(prefix="ch-s9-self-syscall-") as root:
        base = Path(root)
        allow = base / "allowed"
        deny = base / "denied"
        allow.mkdir(mode=0o700)
        deny.mkdir(mode=0o700)
        allow.chmod(0o700)
        deny.chmod(0o700)
        readable = allow / "readable"
        writable = allow / "writable"
        allow_leaf = allow / "child"
        denied_read = base / "no-read"
        denied_write = base / "no-write"
        denied_leaf = deny / "child"
        for path in (
            readable, writable, allow_leaf, denied_read, denied_write, denied_leaf
        ):
            path.write_bytes(b"T")
            path.chmod(0o600)

        # The owner always retains permission to restore the private TEST
        # directory; no installed principal, service or real secret is touched.
        try:
            denied_read.chmod(0o000)
            denied_write.chmod(0o000)
            deny.chmod(0o000)
            return {
                "owner_read": _attempt(readable, "read"),
                "owner_write": _attempt(writable, "write"),
                "owner_traverse": _attempt(allow_leaf, "read"),
                "denied_read": _attempt(denied_read, "read"),
                "denied_write": _attempt(denied_write, "write"),
                "denied_traverse": _attempt(denied_leaf, "read"),
                "missing_read": _attempt(base / "not-created", "read"),
            }
        finally:
            deny.chmod(0o700)
            denied_read.chmod(0o600)
            denied_write.chmod(0o600)


def inspect_local_syscalls(*, observer=None) -> dict:
    """Observe TEST-only semantics; synthetic observer never yields live label."""
    synthetic = observer is not None
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "NOT_QUALIFIED",
        "isolation": "NOT_ASSESSED",
        "blockers": list(_BLOCKERS),
        "smoke": {"observation": "NOT_QUALIFIED", "reason": "NOT_RUN"},
        "outcomes": {name: "NOT_ATTEMPTED" for name in _NAMES},
    }
    if not synthetic and (
        sys.platform != "linux" or not hasattr(os, "geteuid") or os.geteuid() == 0
    ):
        result["smoke"]["reason"] = "ROOTLESS_NATIVE_LINUX_REQUIRED"
        return result

    try:
        raw = (observer if synthetic else _exercise)()
    except Exception:
        result["smoke"]["reason"] = "TEST_FIXTURE_OR_CLEANUP_FAILED"
        return result
    if (
        type(raw) is not dict
        or set(raw) != set(_NAMES)
        or any(type(name) is not str for name in raw)
        or any(type(value) is not str or value not in _KNOWN_OUTCOMES for value in raw.values())
    ):
        result["smoke"]["reason"] = "INVALID_TEST_OBSERVATION"
        return result

    result["snapshot"] = (
        "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LOCAL_SELF_OBSERVATION"
    )
    result["outcomes"] = {name: raw[name] for name in _NAMES}
    expected = (
        all(raw[name] == "SUCCESS" for name in _NAMES[:3])
        and all(raw[name] in _DENIED for name in _NAMES[3:6])
        and raw["missing_read"] == "ENOENT"
    )
    result["smoke"] = (
        {
            "observation": "UNTRUSTED_LOCAL_SELF_SYSCALL_SEMANTICS_ONLY",
            "reason": "NO_CROSS_ROLE_ISOLATION_EVIDENCE",
        }
        if expected
        else {
            "observation": "NOT_QUALIFIED",
            "reason": "LOCAL_SYSCALL_OBSERVATION_MISMATCH",
        }
    )
    return result


def main(argv: list[str] | None = None) -> int:
    """No arguments, no configurable path, no sudo; always exit with code 2."""
    args = sys.argv[1:] if argv is None else argv
    if args:
        report = inspect_local_syscalls(observer=lambda: {})
        report["snapshot"] = "NOT_QUALIFIED"
        report["smoke"]["reason"] = "UNSUPPORTED_ARGUMENTS"
    else:
        report = inspect_local_syscalls()
    print(json.dumps(report, sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
