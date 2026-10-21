"""S9-RPI-35: interpret declared TEST syscall outcomes, never authorize Linux.

This pure adapter neither performs syscalls nor verifies their provenance.
It requires successful owner-positive controls before it can even summarize
all foreign-role denials as untrusted declared results.
"""

from __future__ import annotations

import json

from deployment import stage9_native_linux_test_denial_matrix_contract as denial

_SCHEMA = "CryptoHunter.Stage9TestOnlySyscallOutcomeClassifierV1"
_ROLES = tuple(dict.fromkeys(actor for actor, _, _ in denial.expected_cases()))
_OPERATIONS = ("read", "write", "traverse")
_CROSS_CASES = denial.expected_cases()
_OWNER_CONTROLS = tuple(
    (role, role, operation) for role in _ROLES for operation in _OPERATIONS
)
_EXPECTED = frozenset(_CROSS_CASES + _OWNER_CONTROLS)
_FIELDS = frozenset({"actor", "owner", "operation", "outcome"})
_REQUIRED = frozenset({"schema", "snapshot", "cases"})
_OUTCOMES = frozenset(
    {
        "SUCCESS",
        "DENIED_EACCES",
        "DENIED_EPERM",
        "ENOENT",
        "OTHER_FAILURE",
        "NOT_ATTEMPTED",
    }
)
_DENIED = frozenset({"DENIED_EACCES", "DENIED_EPERM"})
_BLOCKERS = (
    "DECLARED_SYSCALL_OUTCOMES_NOT_KERNEL_VERIFIED",
    "NO_INDEPENDENT_ACTOR_RESOURCE_OR_PROCESS_PROVENANCE",
    "POSITIVE_OWNER_CONTROLS_NOT_TRUSTED",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NATIVE_KEYRING_AND_PROTECTED_SOURCE_CUSTODY_NOT_QUALIFIED",
    "POSTGRESQL_COMMITTED_CLOSE_AND_CCA_DURABILITY_NOT_QUALIFIED",
    "PRODUCTION_ISSUANCE_AND_ENROLLMENT_BLOCKED",
)


def expected_cases() -> tuple[tuple[str, str, str], ...]:
    """Fixed TEST-only plan: 90 negative and 18 positive operations."""
    return _CROSS_CASES + _OWNER_CONTROLS


def inspect_declared_outcomes(record: object = None) -> dict:
    """Classify fixed synthetic/untrusted claims; never attest actual syscalls."""
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "NOT_QUALIFIED",
        "blockers": list(_BLOCKERS),
        "classification": {
            "observation": "NOT_QUALIFIED",
            "reason": "NO_TEST_OBSERVATIONS",
        },
        "denial_matrix": {
            "observation": "NOT_QUALIFIED",
            "reason": "NO_TEST_OBSERVATIONS",
        },
        "counts": {
            "cross_expected": len(_CROSS_CASES),
            "cross_denied": 0,
            "cross_allowed": 0,
            "cross_inconclusive": 0,
            "cross_missing": len(_CROSS_CASES),
            "owner_expected": len(_OWNER_CONTROLS),
            "owner_success": 0,
            "owner_failed": 0,
            "owner_missing": len(_OWNER_CONTROLS),
        },
    }
    if record is None:
        return result

    def reject() -> dict:
        result["classification"]["reason"] = "INVALID_TEST_REPORT"
        return result

    if (
        type(record) is not dict
        or set(record) != _REQUIRED
        or any(type(key) is not str for key in record)
        or record["schema"] != _SCHEMA
    ):
        return reject()
    snapshot = record["snapshot"]
    if type(snapshot) is not str or snapshot not in (
        "SYNTHETIC_TEST_INPUT",
        "UNTRUSTED_LIVE_OBSERVATION",
    ):
        return reject()
    result["snapshot"] = snapshot
    cases = record["cases"]
    if type(cases) is not list or len(cases) > len(_EXPECTED):
        return reject()

    observed = {}
    for item in cases:
        if (
            type(item) is not dict
            or set(item) != _FIELDS
            or any(type(key) is not str for key in item)
            or any(type(item[key]) is not str for key in _FIELDS)
        ):
            return reject()
        key = (item["actor"], item["owner"], item["operation"])
        if key not in _EXPECTED or key in observed or item["outcome"] not in _OUTCOMES:
            return reject()
        observed[key] = item["outcome"]

    cross = {key: value for key, value in observed.items() if key[0] != key[1]}
    owners = {key: value for key, value in observed.items() if key[0] == key[1]}
    counts = {
        "cross_expected": len(_CROSS_CASES),
        "cross_denied": sum(value in _DENIED for value in cross.values()),
        "cross_allowed": sum(value == "SUCCESS" for value in cross.values()),
        "cross_inconclusive": sum(
            value not in _DENIED and value != "SUCCESS" for value in cross.values()
        ),
        "cross_missing": len(_CROSS_CASES) - len(cross),
        "owner_expected": len(_OWNER_CONTROLS),
        "owner_success": sum(value == "SUCCESS" for value in owners.values()),
        "owner_failed": sum(value != "SUCCESS" for value in owners.values()),
        "owner_missing": len(_OWNER_CONTROLS) - len(owners),
    }
    result["counts"] = counts
    claims = {
        "schema": denial._SCHEMA,
        "snapshot": snapshot,
        "cells": [
            {
                "actor": actor,
                "owner": owner,
                "operation": operation,
                "observation": (
                    "CLAIMED_DENIED"
                    if value in _DENIED
                    else "CLAIMED_ALLOWED"
                    if value == "SUCCESS"
                    else "NOT_OBSERVED"
                ),
            }
            for (actor, owner, operation), value in cross.items()
        ],
    }
    matrix = denial.inspect_declared_matrix(claims)
    result["denial_matrix"] = matrix["matrix"]
    if counts["cross_allowed"]:
        reason = "DECLARED_CROSS_ROLE_ACCESS"
    elif counts["owner_failed"]:
        reason = "OWNER_CONTROL_FAILED_OR_INCONCLUSIVE"
    elif counts["owner_missing"]:
        reason = "OWNER_CONTROL_INCOMPLETE"
    elif counts["cross_inconclusive"] or counts["cross_missing"]:
        reason = "CROSS_ROLE_DENIAL_INCOMPLETE_OR_INCONCLUSIVE"
    else:
        result["classification"] = {
            "observation": "UNTRUSTED_DECLARED_DENIAL_WITH_OWNER_CONTROLS_ONLY",
            "reason": "NO_INDEPENDENT_KERNEL_EVIDENCE",
        }
        return result
    result["classification"]["reason"] = reason
    return result


def main() -> int:
    """Default CLI has no input and must stay blocked, with exit code 2."""
    print(json.dumps(inspect_declared_outcomes(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
