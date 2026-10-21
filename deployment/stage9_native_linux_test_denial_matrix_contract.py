"""S9-RPI-34: closed, test-only Source/CCA cross-role denial matrix contract.

Pure validation of externally DECLARED fixture outcomes. No host probing,
privilege changes, authoritative installation evidence, or production GO.
"""

from __future__ import annotations

import json

from deployment import stage9_native_linux_source_cca_preflight as preflight

_ROLE_NAMES = tuple(role for role, _name, _subdir in preflight._PROPOSED) + tuple(
    role for role, _name in preflight._KNOWN_REFERENCE_AUTHORITIES
)
_OPERATIONS = ("read", "write", "traverse")
_CASES = tuple(
    (actor, owner, operation)
    for actor in _ROLE_NAMES
    for owner in _ROLE_NAMES
    if actor != owner
    for operation in _OPERATIONS
)
_SCHEMA = "CryptoHunter.Stage9TestOnlyCrossRoleDenialMatrixV1"
_BLOCKERS = (
    "DECLARED_TEST_OUTCOMES_ARE_NOT_EFFECTIVE_LINUX_PERMISSION_PROOF",
    "NO_INDEPENDENTLY_PINNED_TEST_PROCESS_OR_INSTALLATION_PROVENANCE",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NATIVE_KEYRING_AND_SOURCE_CUSTODY_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_SIGN_BIND_ENROLLMENT_BLOCKED",
)
_REQUIRED_KEYS = frozenset({"schema", "snapshot", "cells"})
_CELL_KEYS = frozenset({"actor", "owner", "operation", "observation"})
_OBSERVATIONS = frozenset({"CLAIMED_DENIED", "CLAIMED_ALLOWED", "NOT_OBSERVED"})


def expected_cases() -> tuple[tuple[str, str, str], ...]:
    """Return only fixed names for the later, separately reviewed TEST fixture."""
    return _CASES


def inspect_declared_matrix(record: object = None) -> dict:
    """Sanitize declared test outcomes. Never confirm any effective host denial."""
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "NOT_QUALIFIED",
        "blockers": list(_BLOCKERS),
        "matrix": {"observation": "NOT_QUALIFIED", "reason": "NO_TEST_OBSERVATIONS"},
        "counts": {
            "expected": len(_CASES),
            "claimed_denied": 0,
            "claimed_allowed": 0,
            "not_observed": 0,
            "missing": len(_CASES),
        },
    }
    if record is None:
        return result
    if (
        type(record) is not dict
        or any(type(key) is not str for key in record)
        or set(record) != _REQUIRED_KEYS
        or type(record["schema"]) is not str
        or record["schema"] != _SCHEMA
    ):
        result["matrix"]["reason"] = "INVALID_TEST_REPORT"
        return result
    snapshot = record["snapshot"]
    if type(snapshot) is not str or snapshot not in (
        "SYNTHETIC_TEST_INPUT",
        "UNTRUSTED_LIVE_OBSERVATION",
    ):
        result["matrix"]["reason"] = "INVALID_TEST_REPORT"
        return result
    result["snapshot"] = snapshot
    cells = record["cells"]
    if type(cells) is not list or len(cells) > len(_CASES):
        result["matrix"]["reason"] = "INVALID_TEST_REPORT"
        return result

    observations = {}
    expected = frozenset(_CASES)
    for cell in cells:
        if (
            type(cell) is not dict
            or any(type(key) is not str for key in cell)
            or set(cell) != _CELL_KEYS
            or any(type(cell[key]) is not str for key in _CELL_KEYS)
        ):
            result["matrix"]["reason"] = "INVALID_TEST_REPORT"
            return result
        key = (cell["actor"], cell["owner"], cell["operation"])
        if key not in expected or key in observations or cell["observation"] not in _OBSERVATIONS:
            result["matrix"]["reason"] = "INVALID_TEST_REPORT"
            return result
        observations[key] = cell["observation"]

    counts = {
        "expected": len(_CASES),
        "claimed_denied": sum(v == "CLAIMED_DENIED" for v in observations.values()),
        "claimed_allowed": sum(v == "CLAIMED_ALLOWED" for v in observations.values()),
        "not_observed": sum(v == "NOT_OBSERVED" for v in observations.values()),
        "missing": len(_CASES) - len(observations),
    }
    result["counts"] = counts
    if counts["claimed_allowed"]:
        reason = "DECLARED_CROSS_ROLE_ACCESS"
    elif counts["not_observed"] or counts["missing"]:
        reason = "INCOMPLETE_TEST_DENIAL_MATRIX"
    else:
        result["matrix"] = {
            "observation": "UNTRUSTED_DECLARED_DENIALS_ONLY",
            "reason": "NO_INDEPENDENT_HOST_ATTESTATION",
        }
        return result

    result["matrix"]["reason"] = reason
    return result


def main() -> int:
    """Without supplied observations this prints only a blocked fixture plan."""
    print(json.dumps(inspect_declared_matrix(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
