"""S9-RPI-30: detect drift in two sequential, untrusted Source/CCA sweeps.

Never interprets repeated numeric hints or unpinned digests as process
attestation. The observation is strictly read-only and always production NO-GO.
"""

from __future__ import annotations

import json
import sys

from deployment import stage9_native_linux_cross_observer_consistency as consistency

_SOURCES = consistency._SOURCES
_ROLES = consistency._ROLES
_OBSERVERS = consistency._NATIVE_OBSERVERS
_FINGERPRINT_FIELDS = {
    "fragment": ("fragment_sha256", "fragment_size"),
    "executable": ("executable_sha256", "executable_size"),
}
_BLOCKERS = (
    "UNTRUSTED_NON_ATOMIC_SEQUENTIAL_SWEEPS",
    "TWO_PASS_MATCH_DOES_NOT_EXCLUDE_ABA_OR_PID_REUSE",
    "PIDFD_OPEN_HINT_NOT_SOCKET_BOUND_SO_PEERPIDFD",
    "UNPINNED_UNIT_AND_EXECUTABLE_DIGESTS_NOT_ATTESTATION",
    "CROSS_ROLE_AUTHORITY_AND_EFFECTIVE_DENIAL_UNQUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_UNQUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


def _sample(observers: dict) -> dict:
    """Collect exactly five closed-shape observations, without retries."""
    readings = {}
    for source in _SOURCES:
        try:
            record = observers[source]()
            signatures, anomalies = consistency._observe_source(source, record)
            fingerprints = {}
            if source in _FINGERPRINT_FIELDS:
                digest_key, size_key = _FINGERPRINT_FIELDS[source]
                rows = record[consistency._FIELDS[source][0]]
                for role, _unit in _ROLES:
                    if signatures[role] is not None:
                        row = rows[role]
                        fingerprints[role] = (row[digest_key], row[size_key])
            readings[source] = (signatures, anomalies, fingerprints)
        except Exception:
            # Invalid records and backend exceptions never flow into the report.
            readings[source] = None
    return readings


def _cross_source_issues(sample: dict, role: str) -> list[str]:
    """Check inconsistencies within one sweep, even if other sources failed."""
    signatures = [
        sample[source][0][role]
        for source in _SOURCES
        if sample[source] is not None and sample[source][0][role] is not None
    ]
    issues = []
    if len({signature[0] for signature in signatures}) > 1:
        issues.append("CROSS_SOURCE_PID_CONTRADICTION")
    if len({signature[1] for signature in signatures if signature[1] is not None}) > 1:
        issues.append("CROSS_SOURCE_STARTTIME_CONTRADICTION")
    return issues


def _cross_role_issues(sample: dict) -> set[str]:
    """Report numeric hint collisions between fixed roles in each sweep."""
    role_pids = {}
    for role, _unit in _ROLES:
        role_pids[role] = {
            signatures[role][0]
            for source in _SOURCES
            if (item := sample[source]) is not None
            if (signatures := item[0])[role] is not None
        }
    collisions = set()
    for index, (left, _unit) in enumerate(_ROLES):
        for right, _other_unit in _ROLES[index + 1 :]:
            if role_pids[left] & role_pids[right]:
                collisions.update((left, right))
    return collisions


def inspect_host(*, observers=None) -> dict:
    """Two strictly sequential sweeps, with no mutation and no trust promotion."""
    synthetic = observers is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxTwoPassDriftObservationV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION",
        "blockers": list(_BLOCKERS),
        "sweeps": 2,
        "drift_observations": {},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result
    if observers is None:
        observers = _OBSERVERS
    if (
        type(observers) is not dict
        or set(observers) != set(_SOURCES)
        or any(not callable(observers[source]) for source in _SOURCES)
    ):
        result["observed_failures"].append("INVALID_OBSERVER_CONFIGURATION")
        return result

    first = _sample(observers)
    second = _sample(observers)
    collisions = _cross_role_issues(first) | _cross_role_issues(second)
    for role, _unit in _ROLES:
        issues = []
        both_observed = 0
        for source in _SOURCES:
            old, new = first[source], second[source]
            if (
                old is None
                or new is None
                or old[0][role] is None
                or new[0][role] is None
            ):
                issues.append(f"{source}: INCOMPLETE_TWO_PASS_OBSERVATION")
                continue
            both_observed += 1
            if role in old[1] or role in new[1]:
                issues.append(f"{source}: SUBREPORT_ANOMALY")
            previous, current = old[0][role], new[0][role]
            if previous[0] != current[0]:
                issues.append(f"{source}: PID_CHANGED_BETWEEN_SWEEPS")
            if previous[1] != current[1]:
                issues.append(f"{source}: STARTTIME_CHANGED_BETWEEN_SWEEPS")
            if source in _FINGERPRINT_FIELDS and (
                old[2][role] != new[2][role]
            ):
                issues.append(f"{source}: UNPINNED_OBJECT_CHANGED_BETWEEN_SWEEPS")
        # Both sweeps are checked independently: a consistent *second*
        # sweep cannot erase inconsistencies present in the first sweep.
        for sweep in (first, second):
            for code in _cross_source_issues(sweep, role):
                if code not in issues:
                    issues.append(code)
        if role in collisions:
            issues.append("CROSS_ROLE_SHARED_PID_HINT")
        result["drift_observations"][role] = {
            "observation": (
                "NO_TWO_PASS_CONTRADICTION_OBSERVED" if not issues else "NOT_QUALIFIED"
            ),
            "sources_observed_both": both_observed,
            "issues": issues,
        }
        result["observed_failures"].extend(f"{role}: {issue}" for issue in issues)
    return result


def main() -> int:
    print(json.dumps(inspect_host(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
