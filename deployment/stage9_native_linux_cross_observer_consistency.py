"""S9-RPI-29: correlate untrusted Source/CCA observations without promotion.

All underlying probes are sequential and have independent sampling windows.
A matching PID/starttime is never authenticated process ownership or a trust pin.
"""

from __future__ import annotations

import json
import sys

from deployment import stage9_native_linux_cgroup_membership_observation as cgroup
from deployment import stage9_native_linux_pidfd_lifetime_observation as pidfd
from deployment import stage9_native_linux_process_executable_observation as executable
from deployment import stage9_native_linux_process_identity_observation as identity
from deployment import stage9_native_linux_unit_fragment_observation as fragment

_ROLES = tuple((role, unit) for role, unit, _account in identity._PROPOSED_SYSTEMD_UNITS)
_SOURCES = ("identity", "cgroup", "fragment", "executable", "pidfd")
_NATIVE_OBSERVERS = {
    "identity": identity.inspect_host,
    "cgroup": cgroup.inspect_host,
    "fragment": fragment.inspect_host,
    "executable": executable.inspect_host,
    "pidfd": pidfd.inspect_host,
}
_SCHEMAS = {
    "identity": "CryptoHunter.Stage9NativeLinuxProcessIdentityObservationV1",
    "cgroup": "CryptoHunter.Stage9NativeLinuxCgroupMembershipObservationV1",
    "fragment": "CryptoHunter.Stage9NativeLinuxUnitFingerprintObservationV1",
    "executable": "CryptoHunter.Stage9NativeLinuxProcessExecutableObservationV1",
    "pidfd": "CryptoHunter.Stage9NativeLinuxPidfdLifetimeObservationV1",
}
_FIELDS = {
    "identity": ("process_observations", "PROCESS_METADATA_ONLY"),
    "cgroup": ("process_observations", "CGROUP_MEMBERSHIP_METADATA_ONLY"),
    "fragment": ("fragment_observations", "UNPINNED_FRAGMENT_DIGEST_ONLY"),
    "executable": ("executable_observations", "UNPINNED_PROC_EXECUTABLE_DIGEST_ONLY"),
    "pidfd": ("pidfd_observations", "UNTRUSTED_PIDFD_LIFETIME_METADATA_ONLY"),
}
_SUCCESS_KEYS = {
    "identity": frozenset({"unit", "observation", "identity"}),
    "cgroup": frozenset({"unit", "observation", "pid", "start_time_ticks"}),
    "fragment": frozenset({
        "unit", "observation", "pid_hint", "fragment_sha256", "fragment_size"
    }),
    "executable": frozenset({
        "unit", "observation", "pid_hint", "start_time_ticks",
        "executable_sha256", "executable_size",
    }),
    "pidfd": frozenset({
        "unit", "observation", "pid_hint", "start_time_ticks", "pidfd_poll_alive"
    }),
}
_BLOCKERS = (
    "ALL_INPUTS_UNTRUSTED_SEQUENTIAL_DIAGNOSTICS",
    "NO_ATOMIC_PIDFD_BINDING_ACROSS_OBSERVER_SNAPSHOTS",
    "NO_INDEPENDENT_HOST_UNIT_EXECUTABLE_INSTALLATION_PINS",
    "NO_SOCKET_BOUND_SO_PEERPIDFD_OR_CROSS_ROLE_ACCESS_PROOF",
    "POSTGRESQL_COMMITTED_CLOSE_AND_DURABILITY_NOT_QUALIFIED",
    "SIGN_BIND_ENROLLMENT_PRODUCTION_BLOCKED",
)
_TOP_KEYS = frozenset({
    "schema", "status", "qualification", "production_emission", "topology",
    "snapshot", "blockers", "observed_failures",
})


def _bounded_number(value: object, upper: int = 2**64 - 1) -> bool:
    return type(value) is int and 1 < value <= upper


def _digest(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _row_signature(source: str, row: object, *, unit: str) -> tuple[int, int | None]:
    """Whitelist only the numeric hints that can be compared across probes."""
    if type(row) is not dict or set(row) != _SUCCESS_KEYS[source]:
        raise ValueError("INVALID_SUBREPORT")
    if row["unit"] != unit or row["observation"] != _FIELDS[source][1]:
        raise ValueError("INVALID_SUBREPORT")
    if source == "identity":
        item = row["identity"]
        if type(item) is not dict or set(item) != {
            "pid", "start_time_ticks", "uid", "gid", "groups", "namespaces"
        }:
            raise ValueError("INVALID_SUBREPORT")
        if (
            not all(_bounded_number(item[k]) for k in ("pid", "start_time_ticks"))
            or not all(
                type(item[k]) is int and 0 <= item[k] <= 2**32 - 2
                for k in ("uid", "gid")
            )
            or type(item["groups"]) is not list
            or len(item["groups"]) > 1
            or any(
                type(g) is not int or g < 0 or g > 2**32 - 2
                for g in item["groups"]
            )
            or type(item["namespaces"]) is not dict
            or set(item["namespaces"]) != set(identity._NS_TYPES)
            or any(not _bounded_number(v) for v in item["namespaces"].values())
        ):
            raise ValueError("INVALID_SUBREPORT")
        pid, ticks = item["pid"], item["start_time_ticks"]
    else:
        pid = row["pid"] if source == "cgroup" else row["pid_hint"]
        ticks = None if source == "fragment" else row["start_time_ticks"]
    if not _bounded_number(pid, 2**31 - 1) or (ticks is not None and not _bounded_number(ticks)):
        raise ValueError("INVALID_SUBREPORT")
    if source == "fragment" and (
        not _digest(row["fragment_sha256"])
        or type(row["fragment_size"]) is not int
        or not 0 < row["fragment_size"] <= fragment._MAX_FRAGMENT_BYTES
    ):
        raise ValueError("INVALID_SUBREPORT")
    if source == "executable" and (
        not _digest(row["executable_sha256"])
        or type(row["executable_size"]) is not int
        or not 0 < row["executable_size"] <= executable._MAX_EXE_BYTES
    ):
        raise ValueError("INVALID_SUBREPORT")
    if source == "pidfd" and row["pidfd_poll_alive"] is not True:
        raise ValueError("INVALID_SUBREPORT")
    return pid, ticks


def _observe_source(source: str, record: object) -> tuple[dict, bool]:
    """Never forward arbitrary sub-observer data or failure details."""
    field, success = _FIELDS[source]
    if type(record) is not dict or set(record) != _TOP_KEYS | {field}:
        raise ValueError("INVALID_SUBREPORT")
    if any(
        record[key] != expected
        for key, expected in (
            ("schema", _SCHEMAS[source]),
            ("status", "BLOCKED"),
            ("qualification", "NOT_QUALIFIED"),
            ("production_emission", "BLOCKED"),
            ("topology", "PROPOSED_NOT_APPROVED"),
        )
    ):
        raise ValueError("INVALID_SUBREPORT")
    if (
        record["snapshot"] not in ("SYNTHETIC_TEST_INPUT", "UNTRUSTED_LIVE_OBSERVATION")
        or type(record["blockers"]) is not list
        or not 1 <= len(record["blockers"]) <= 30
        or any(type(s) is not str or len(s) > 120 for s in record["blockers"])
        or type(record["observed_failures"]) is not list
        or len(record["observed_failures"]) > 30
        or any(type(s) is not str or len(s) > 160 for s in record["observed_failures"])
    ):
        raise ValueError("INVALID_SUBREPORT")
    rows = record[field]
    if type(rows) is not dict or set(rows) != {r for r, _unit in _ROLES}:
        raise ValueError("INVALID_SUBREPORT")
    signatures = {}
    failed = bool(record["observed_failures"])
    for role, unit in _ROLES:
        row = rows[role]
        if type(row) is not dict or row.get("unit") != unit:
            raise ValueError("INVALID_SUBREPORT")
        if row.get("observation") == "NOT_QUALIFIED":
            if (
                set(row) != {"unit", "observation", "issues"}
                or type(row["issues"]) is not list
                or not 1 <= len(row["issues"]) <= 8
                or any(
                    type(issue) is not str
                    or not 1 <= len(issue) <= 90
                    or any(char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for char in issue)
                    for issue in row["issues"]
                )
            ):
                raise ValueError("INVALID_SUBREPORT")
            signatures[role] = None
            failed = True
        else:
            signatures[role] = _row_signature(source, row, unit=unit)
    return signatures, failed


def inspect_host(*, observers=None) -> dict:
    """Aggregate five existing untrusted reports, refusing all promotion."""
    synthetic = observers is not None
    report = {
        "schema": "CryptoHunter.Stage9NativeLinuxCrossObservationConsistencyV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "correlation": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        report["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return report
    if observers is None:
        observers = _NATIVE_OBSERVERS
    if (
        type(observers) is not dict
        or set(observers) != set(_SOURCES)
        or any(not callable(observers[name]) for name in _SOURCES)
    ):
        report["observed_failures"].append("INVALID_OBSERVER_CONFIGURATION")
        return report

    observations = {}
    for source in _SOURCES:
        try:
            # No raw subreport data is copied into the final JSON.
            observations[source] = _observe_source(source, observers[source]())
        except Exception:
            observations[source] = None
            report["observed_failures"].append(f"{source}: OBSERVER_FAILED_OR_INVALID")

    role_pid_sets = {}
    for role, _unit in _ROLES:
        signatures = {
            source: observations[source][0][role]
            for source in _SOURCES
            if observations[source] is not None and observations[source][0][role] is not None
        }
        issues = []
        for source in _SOURCES:
            if observations[source] is None or observations[source][0][role] is None:
                issues.append(f"{source}: NOT_OBSERVED")
            elif observations[source][1]:
                issues.append(f"{source}: SUBREPORT_ANOMALY")
        pids = {item[0] for item in signatures.values()}
        role_pid_sets[role] = pids
        ticks = {item[1] for item in signatures.values() if item[1] is not None}
        if len(pids) > 1:
            issues.append("CROSS_OBSERVER_MAINPID_MISMATCH")
        if len(ticks) > 1:
            issues.append("CROSS_OBSERVER_STARTTIME_MISMATCH")
        report["correlation"][role] = {
            "observation": (
                "NO_CROSS_SOURCE_CONTRADICTION_OBSERVED" if not issues else "NOT_QUALIFIED"
            ),
            "issues": issues,
            "sources_observed": len(signatures),
        }
        report["observed_failures"].extend(f"{role}: {code}" for code in issues)
    # Reject a shared numeric PID across roles even if a subordinate observer
    # fails to report the duplicate. This remains an untrusted metadata hint.
    for index, (left, _unit) in enumerate(_ROLES):
        for right, _other_unit in _ROLES[index + 1 :]:
            if role_pid_sets[left] & role_pid_sets[right]:
                for role in (left, right):
                    row = report["correlation"][role]
                    if "CROSS_ROLE_SHARED_PID_HINT" not in row["issues"]:
                        row["issues"].append("CROSS_ROLE_SHARED_PID_HINT")
                        row["observation"] = "NOT_QUALIFIED"
                        report["observed_failures"].append(
                            f"{role}: CROSS_ROLE_SHARED_PID_HINT"
                        )
    return report


def main() -> int:
    print(json.dumps(inspect_host(), ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
