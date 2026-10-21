"""S9-RPI-32: one-command, sanitized read-only Linux diagnostic handoff.

This is not host attestation, runtime authorization or production qualification.
It never forwards paths, identities, PIDs, namespaces or subordinate errors.
"""

from __future__ import annotations

import json
import sys

from deployment import stage9_native_linux_source_cca_preflight as preflight
from deployment import stage9_native_linux_two_pass_drift_observation as drift

_ROLES = tuple(role for role, _unit in drift._ROLES)
_REFERENCES = ("root_signer", "history_signer", "issuer")
_PATHS = frozenset(
    f"{role}.{kind}" for role in _ROLES for kind in ("configuration", "private_state")
)
_ACCOUNTS = {role: name for role, name, _subdir in preflight._PROPOSED}
_REFERENCE_ACCOUNTS = dict(preflight._KNOWN_REFERENCE_AUTHORITIES)
_UNITS = {role: unit for role, unit, _name in preflight._PROPOSED_SYSTEMD_UNITS}
_CANDIDATE_PATHS = {
    f"{role}.{kind}": f"{prefix}/{subdir}"
    for role, _name, subdir in preflight._PROPOSED
    for kind, prefix in (
        ("configuration", "/etc/cryptohunter"),
        ("private_state", "/var/lib/cryptohunter"),
    )
}
_PATH_FAILURES = frozenset(
    {
        "non-directory or symlink in path",
        "writable ancestor or target",
        "ancestor owner mismatch",
        "target identity, group or mode mismatch",
        "POSIX ACL inventory unavailable",
        "extended POSIX ACL present",
        "path missing or inaccessible",
    }
)
_UNIT_ISSUES = frozenset(
    {
        "SYSTEMD_UNAVAILABLE",
        "MALFORMED_SYSTEMD_RESPONSE",
        "UNEXPECTED_OR_DUPLICATE_PROPERTY",
        "INCOMPLETE_SYSTEMD_RESPONSE",
        "MAIN_PID_NOT_OBSERVED",
        "UNIT_FRAGMENT_NOT_ABSOLUTE",
        *(
            f"{field}_NOT_MATCHED"
            for field in (
                "Id",
                "LoadState",
                "ActiveState",
                "User",
                "Group",
                "DynamicUser",
                "NoNewPrivileges",
                "PrivateNetwork",
                "PrivateIPC",
                "PrivateMounts",
                "PrivateTmp",
                "ProtectSystem",
                "ProtectHome",
                "RestrictAddressFamilies",
                "SupplementaryGroups",
                "CapabilityBoundingSet",
                "AmbientCapabilities",
                "DropInPaths",
            )
        ),
    }
)
_COMMON = {
    "status": "BLOCKED",
    "qualification": "NOT_QUALIFIED",
    "production_emission": "BLOCKED",
    "topology": "PROPOSED_NOT_APPROVED",
}
_PREFLIGHT_KEYS = frozenset(
    {
        "schema",
        *_COMMON,
        "principal_observations",
        "reference_principal_observations",
        "unobserved_reference_principals",
        "systemd_observations",
        "paths",
        "observed_failures",
        "blockers",
    }
)
_DRIFT_KEYS = frozenset(
    {
        "schema",
        *_COMMON,
        "snapshot",
        "blockers",
        "sweeps",
        "sweeps_attempted",
        "drift_observations",
        "observed_failures",
    }
)
_BLOCKERS = (
    "UNTRUSTED_INDEPENDENT_NON_ATOMIC_HOST_OBSERVATIONS",
    "NO_APPROVED_PHYSICAL_HOST_OR_INSTALLATION_PIN",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NO_EFFECTIVE_CROSS_ROLE_DENIAL_OR_KEYRING_PROOF",
    "NO_POSTGRESQL_COMMITTED_CLOSE_OR_CCA_DURABILITY",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)


def _closed_header(report: object, schema: str, keys: frozenset[str]) -> bool:
    return (
        type(report) is dict
        and all(type(key) is str for key in report)
        and set(report) == keys
        and type(report["schema"]) is str
        and report["schema"] == schema
        and all(
            type(report[key]) is str and report[key] == expected
            for key, expected in _COMMON.items()
        )
        and type(report["blockers"]) is list
        and 1 <= len(report["blockers"]) <= 32
        and all(type(item) is str and 1 <= len(item) <= 128 for item in report["blockers"])
        and type(report["observed_failures"]) is list
        and len(report["observed_failures"]) <= 256
        and all(type(item) is str and 1 <= len(item) <= 180 for item in report["observed_failures"])
    )


def _principal_record(row: object, *, name: str) -> bool:
    """Validate the serialized NSS contract, without copying identity values."""
    if (
        type(row) is not dict
        or any(type(key) is not str for key in row)
        or set(row) != {"name", "uid", "gid", "supplementary_gids"}
        or type(row["name"]) is not str
        or row["name"] != name
        or any(
            type(row[key]) is not int or not 1 <= row[key] <= 2**32 - 2 for key in ("uid", "gid")
        )
        or type(row["supplementary_gids"]) is not list
        or len(row["supplementary_gids"]) > 65536
        or any(
            type(gid) is not int or not 1 <= gid <= 2**32 - 2 for gid in row["supplementary_gids"]
        )
    ):
        return False
    groups = row["supplementary_gids"]
    return groups == sorted(set(groups)) and row["gid"] not in groups


def _unit_record(row: object, *, unit: str) -> bool:
    if (
        type(row) is not dict
        or any(type(key) is not str for key in row)
        or type(row.get("unit")) is not str
        or row["unit"] != unit
    ):
        return False
    label = row.get("observation")
    if type(label) is not str:
        return False
    if label in ("NOT_OBSERVED_TEST_FIXTURE", "NOT_OBSERVED_NON_LINUX"):
        return set(row) == {"unit", "observation"}
    if (
        set(row) != {"unit", "observation", "issues"}
        or type(row["issues"]) is not list
        or len(row["issues"]) > len(_UNIT_ISSUES)
        or any(type(issue) is not str or issue not in _UNIT_ISSUES for issue in row["issues"])
        or len(set(row["issues"])) != len(row["issues"])
    ):
        return False
    return (label == "PROPOSED_DECLARATIONS_MATCHED_ONLY" and not row["issues"]) or (
        label == "NOT_QUALIFIED" and bool(row["issues"])
    )


def _path_record(row: object, *, path: str) -> bool:
    if (
        type(row) is not dict
        or any(type(key) is not str for key in row)
        or set(row) != {"path", "observation", "reason"}
        or any(type(row[key]) is not str for key in row)
        or row["path"] != path
    ):
        return False
    return (
        row["observation"] == "MATCHED_PROPOSED_METADATA_ONLY"
        and row["reason"] == "no private contents inspected"
    ) or (row["observation"] == "NOT_QUALIFIED" and row["reason"] in _PATH_FAILURES)


def _preflight_summary(record: object) -> dict:
    """Validate actual preflight records before issuing fixed diagnostic labels."""
    invalid = {"observation": "NOT_QUALIFIED", "reason": "INVALID_PREFLIGHT_REPORT"}
    if not _closed_header(
        record, "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1", _PREFLIGHT_KEYS
    ):
        return invalid
    if record["blockers"] not in (
        list(preflight._BLOCKERS),
        ["NATIVE_LINUX_REQUIRED", *preflight._BLOCKERS],
    ):
        return invalid

    principals = record["principal_observations"]
    references = record["reference_principal_observations"]
    missing = record["unobserved_reference_principals"]
    units = record["systemd_observations"]
    paths = record["paths"]
    if (
        any(type(value) is not dict for value in (principals, references, units, paths))
        or any(
            type(key) is not str
            for value in (principals, references, units, paths)
            for key in value
        )
        or not set(principals) <= set(_ROLES)
        or not set(references) <= set(_REFERENCES)
        or set(units) != set(_ROLES)
        or not set(paths) <= _PATHS
        or type(missing) is not list
        or len(missing) > len(_REFERENCES)
        or any(type(role) is not str or role not in _REFERENCES for role in missing)
        or len(set(missing)) != len(missing)
        or set(missing) & set(references)
        or any(not _principal_record(row, name=_ACCOUNTS[role]) for role, row in principals.items())
        or any(
            not _principal_record(row, name=_REFERENCE_ACCOUNTS[role])
            for role, row in references.items()
        )
        or any(not _unit_record(row, unit=_UNITS[role]) for role, row in units.items())
        or any(not _path_record(row, path=_CANDIDATE_PATHS[key]) for key, row in paths.items())
        or set(paths)
        != {f"{role}.{kind}" for role in principals for kind in ("configuration", "private_state")}
    ):
        return invalid

    # Reuse the preflight's separation rules, including reference memberships.
    # These comparisons only reject contradictions; they establish no provenance.
    accounts = {
        row["name"]: (row["uid"], row["gid"], (row["gid"], *row["supplementary_gids"]))
        for row in (*principals.values(), *references.values())
    }
    identities, identity_issues = preflight._inspect_accounts(accounts)
    _references, _missing, reference_issues = preflight._inspect_reference_authorities(
        accounts, identities
    )
    matching_units = sum(
        unit.get("observation") == "PROPOSED_DECLARATIONS_MATCHED_ONLY" for unit in units.values()
    )
    matching_paths = sum(
        path.get("observation") == "MATCHED_PROPOSED_METADATA_ONLY" for path in paths.values()
    )
    complete = (
        record["blockers"] == list(preflight._BLOCKERS)
        and len(principals) == len(_ROLES)
        and len(references) == len(_REFERENCES)
        and not missing
        and matching_units == len(_ROLES)
        and matching_paths == len(_PATHS)
        and not record["observed_failures"]
        and not identity_issues
        and not reference_issues
    )
    return {
        "observation": "UNTRUSTED_DECLARATIONS_ONLY" if complete else "NOT_QUALIFIED",
        "reason": "NO_ATTESTATION" if complete else "PREFLIGHT_INCOMPLETE_OR_ANOMALOUS",
        "principal_roles_seen": len(principals),
        "reference_roles_seen": len(references),
        "proposed_units_matching": matching_units,
        "proposed_paths_matching": matching_paths,
    }


def _drift_summary(record: object) -> dict:
    invalid = {"observation": "NOT_QUALIFIED", "reason": "INVALID_DRIFT_REPORT"}
    if not _closed_header(
        record, "CryptoHunter.Stage9NativeLinuxTwoPassDriftObservationV1", _DRIFT_KEYS
    ):
        return invalid
    if (
        type(record["snapshot"]) is not str
        or record["snapshot"] not in ("UNTRUSTED_LIVE_OBSERVATION", "SYNTHETIC_TEST_INPUT")
        or type(record["sweeps"]) is not int
        or record["sweeps"] != 2
        or type(record["sweeps_attempted"]) is not int
        or record["sweeps_attempted"] not in (0, 2)
        or type(record["drift_observations"]) is not dict
        or any(type(role) is not str for role in record["drift_observations"])
        or not set(record["drift_observations"]) <= set(_ROLES)
    ):
        return invalid

    sanitized_roles = {}
    for role, row in record["drift_observations"].items():
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != {"observation", "sources_observed_both", "issues"}
            or type(row["observation"]) is not str
            or row["observation"] not in ("NO_TWO_PASS_CONTRADICTION_OBSERVED", "NOT_QUALIFIED")
            or type(row["sources_observed_both"]) is not int
            or not 0 <= row["sources_observed_both"] <= len(drift._SOURCES)
            or type(row["issues"]) is not list
            or len(row["issues"]) > 64
            or any(type(issue) is not str or not 1 <= len(issue) <= 128 for issue in row["issues"])
        ):
            return invalid
        clean = (
            row["observation"] == "NO_TWO_PASS_CONTRADICTION_OBSERVED"
            and not row["issues"]
            and row["sources_observed_both"] == len(drift._SOURCES)
        )
        sanitized_roles[role] = {
            "observation": "UNTRUSTED_NO_CONTRADICTION" if clean else "NOT_QUALIFIED",
            "sources_observed_both": row["sources_observed_both"],
            "issue_count": len(row["issues"]),
        }
    complete = (
        record["sweeps_attempted"] == 2
        and set(sanitized_roles) == set(_ROLES)
        and not record["observed_failures"]
        and all(
            row["observation"] == "UNTRUSTED_NO_CONTRADICTION" for row in sanitized_roles.values()
        )
    )
    return {
        "observation": ("UNTRUSTED_TWO_PASS_METADATA_ONLY" if complete else "NOT_QUALIFIED"),
        "reason": "NO_ATTESTATION" if complete else "DRIFT_INCOMPLETE_OR_ANOMALOUS",
        "sweeps_attempted": record["sweeps_attempted"],
        "roles": sanitized_roles,
    }


def inspect_host(*, preflight_observer=None, drift_observer=None) -> dict:
    """Invoke each existing read-only diagnostic once and sanitize all outputs."""
    synthetic = preflight_observer is not None or drift_observer is not None
    result = {
        "schema": "CryptoHunter.Stage9NativeLinuxDiagnosticHandoffV1",
        **_COMMON,
        "snapshot": ("SYNTHETIC_TEST_INPUT" if synthetic else "UNTRUSTED_LIVE_OBSERVATION"),
        "blockers": list(_BLOCKERS),
        "preflight": {"observation": "NOT_QUALIFIED", "reason": "NOT_OBSERVED"},
        "two_pass": {"observation": "NOT_QUALIFIED", "reason": "NOT_OBSERVED"},
        "observed_failures": [],
    }
    if sys.platform != "linux":
        result["observed_failures"].append("NATIVE_LINUX_REQUIRED")
        return result

    preflight_runner = preflight.inspect_host if preflight_observer is None else preflight_observer
    drift_runner = drift.inspect_host if drift_observer is None else drift_observer
    for key, observer, summarizer in (
        ("preflight", preflight_runner, _preflight_summary),
        ("two_pass", drift_runner, _drift_summary),
    ):
        try:
            report = observer()
            # Propagate declared test provenance even if validation later fails.
            # A missing declaration cannot establish live provenance either.
            if key == "two_pass":
                snapshot = report.get("snapshot") if type(report) is dict else None
                if type(snapshot) is str and snapshot == "SYNTHETIC_TEST_INPUT":
                    result["snapshot"] = "SYNTHETIC_TEST_INPUT"
                elif result["snapshot"] != "SYNTHETIC_TEST_INPUT" and (
                    type(snapshot) is not str or snapshot != "UNTRUSTED_LIVE_OBSERVATION"
                ):
                    result["snapshot"] = "NOT_QUALIFIED"
            elif type(report) is dict:
                units = report.get("systemd_observations")
                if type(units) is dict and any(
                    type(units.get(role)) is dict
                    and units[role].get("observation") == "NOT_OBSERVED_TEST_FIXTURE"
                    for role in _ROLES
                ):
                    result["snapshot"] = "SYNTHETIC_TEST_INPUT"
            result[key] = summarizer(report)
        except Exception:
            if result["snapshot"] != "SYNTHETIC_TEST_INPUT":
                result["snapshot"] = "NOT_QUALIFIED"
            result[key] = {
                "observation": "NOT_QUALIFIED",
                "reason": "OBSERVER_EXCEPTION",
            }
        if result[key]["observation"] == "NOT_QUALIFIED":
            result["observed_failures"].append(
                "PREFLIGHT_NOT_QUALIFIED" if key == "preflight" else "TWO_PASS_NOT_QUALIFIED"
            )
    return result


def main() -> int:
    print(json.dumps(inspect_host(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
