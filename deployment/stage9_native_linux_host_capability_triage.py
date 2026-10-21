"""S9-RPI-33: read-only, rootless Linux test-host capability triage.

Probes only the caller's process and public systemd manager metadata.
No probe outcome qualifies a Source/CCA service or production installation.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys

from deployment import stage9_native_linux_pidfd_lifetime_observation as pidfd

_NAMES = ("procfs", "namespaces", "pidfd_self", "systemd_manager")
_BLOCKERS = (
    "HOST_CAPABILITY_HINTS_ARE_NOT_PRODUCTION_ATTESTATION",
    "SOURCE_CCA_INSTALLATION_AND_ISOLATION_NOT_QUALIFIED",
    "AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY_NOT_QUALIFIED",
    "INDEPENDENT_UNIT_EXECUTABLE_TRUST_PINS_NOT_QUALIFIED",
    "EFFECTIVE_CROSS_ROLE_DENIAL_AND_KEYRING_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_AND_ENROLLMENT_BLOCKED",
)


def _procfs() -> bool:
    """Check bounded public self-stat access without following a path alias."""
    root = -1
    fd = -1
    try:
        root = os.open(
            f"/proc/{os.getpid()}",
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        if not stat.S_ISDIR(os.fstat(root).st_mode):
            return False
        fd = os.open(
            "stat",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=root,
        )
        raw = os.read(fd, 4097)
        return (
            type(raw) is bytes
            and 0 < len(raw) <= 4096
            and raw.startswith(f"{os.getpid()} (".encode("ascii"))
            and b") " in raw
            and b"\x00" not in raw
        )
    except (OSError, ValueError, OverflowError):
        return False
    finally:
        if fd >= 0:
            os.close(fd)
        if root >= 0:
            os.close(root)


def _namespaces() -> bool:
    """Inspect only the caller's three namespace inode hints."""
    try:
        for name in ("mnt", "ipc", "net"):
            node = os.stat(f"/proc/{os.getpid()}/ns/{name}")
            if (
                type(node.st_ino) is not int
                or node.st_ino <= 0
                or not stat.S_ISREG(node.st_mode)
            ):
                return False
    except (OSError, ValueError, OverflowError):
        return False
    return True


def _pidfd_self() -> bool:
    """Exercise the existing pidfd/procfs lifetime observer on self only."""
    try:
        record = pidfd._read_pidfd_lifetime(os.getpid())
    except Exception:
        return False
    return (
        type(record) is dict
        and set(record) == {"pid", "start_time_ticks", "pidfd_poll_alive"}
        and type(record["pid"]) is int
        and record["pid"] == os.getpid()
        and type(record["start_time_ticks"]) is int
        and 0 < record["start_time_ticks"] < 2**64
        and record["pidfd_poll_alive"] is True
    )


def _systemd_manager() -> bool:
    """Query a fixed, non-sensitive systemd manager property without elevation."""
    try:
        response = subprocess.run(
            (
                "/usr/bin/systemctl",
                "--system",
                "show",
                "--no-pager",
                "--property=Version",
            ),
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
            env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
        )
        if (
            type(response.returncode) is not int
            or response.returncode != 0
            or type(response.stdout) is not str
            or len(response.stdout) > 256
            or "\x00" in response.stdout
        ):
            return False
        lines = response.stdout.splitlines()
        if len(lines) != 1 or not lines[0].startswith("Version="):
            return False
        version = lines[0].removeprefix("Version=")
        return (
            0 < len(version) <= 128
            and version.isascii()
            and version[0] in "0123456789"
            and all(32 <= ord(char) < 127 for char in version)
        )
    except (OSError, ValueError, TimeoutError, subprocess.TimeoutExpired, UnicodeError):
        return False


_NATIVE_PROBES = {
    "procfs": _procfs,
    "namespaces": _namespaces,
    "pidfd_self": _pidfd_self,
    "systemd_manager": _systemd_manager,
}


def inspect_host(*, probes=None) -> dict:
    """Independently test fixed host capabilities; never emit raw observations."""
    synthetic = probes is not None
    report = {
        "schema": "CryptoHunter.Stage9NativeLinuxHostCapabilityTriageV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "capabilities": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        report["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return report

    if probes is None:
        probes = _NATIVE_PROBES
    if (
        type(probes) is not dict
        or set(probes) != set(_NAMES)
        or any(type(name) is not str or not callable(probes[name]) for name in _NAMES)
    ):
        report["observed_failures"].append("INVALID_PROBE_CONFIGURATION")
        return report

    for name in _NAMES:
        try:
            observed = probes[name]()
        except Exception:
            observed = None
        label = "OBSERVED_UNTRUSTED" if observed is True else "NOT_OBSERVED"
        reason = (
            "UNTRUSTED_CAPABILITY_HINT"
            if observed is True
            else "PROBE_UNAVAILABLE"
            if observed is False
            else "PROBE_FAILED_OR_MALFORMED"
        )
        report["capabilities"][name] = {"observation": label, "reason": reason}
        if observed is not True:
            report["observed_failures"].append(f"{name}: {reason}")
    return report


def main() -> int:
    print(json.dumps(inspect_host(), ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
