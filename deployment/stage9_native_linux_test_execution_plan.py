"""S9-RPI-38: deterministic offline TEST-only Linux isolation execution plan.

Creates and validates inert, fixed-label operation descriptors on Windows.
Never launches a process, touches filesystem targets, or attempts Linux access.
"""

from __future__ import annotations

import json

from deployment import stage9_native_linux_test_syscall_outcome_classifier as outcomes

_SCHEMA = "CryptoHunter.Stage9OfflineTestExecutionPlanV1"
_SNAPSHOT = "SYNTHETIC_TEST_PLAN"
_OPERATIONS = ("read", "write", "traverse")
_EXPECTED_FIELDS = frozenset(
    {
        "id",
        "phase",
        "actor",
        "owner",
        "operation",
        "expected",
        "owner_control",
    }
)
_TOP_FIELDS = frozenset({"schema", "snapshot", "steps"})
_BLOCKERS = (
    "OFFLINE_PLAN_IS_NOT_EXECUTION_OR_LINUX_HOST_EVIDENCE",
    "NO_APPROVED_TEMPORARY_TEST_PRINCIPALS_OR_RESOURCES",
    "NO_INDEPENDENT_EFFECTIVE_UID_GID_OR_PROCESS_BINDING",
    "NO_ACTUAL_KERNEL_DENIAL_OR_OWNER_CONTROL_PROOF",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NO_NATIVE_KEYRING_OR_PROTECTED_CUSTODY_PROOF",
    "NO_POSTGRESQL_COMMITTED_CLOSE_OR_CCA_DURABILITY_PROOF",
    "PRODUCTION_ISSUANCE_AND_ENROLLMENT_BLOCKED",
)


def _canonical_steps() -> tuple[dict[str, str], ...]:
    """Positive owner checks precede foreign denials, with fixed dependencies."""
    cases = outcomes.expected_cases()
    roles = tuple(dict.fromkeys(actor for actor, _, _ in cases))
    owner_keys = tuple(
        (role, role, operation) for role in roles for operation in _OPERATIONS
    )
    cross_keys = tuple(
        (actor, owner, operation)
        for actor, owner, operation in cases
        if actor != owner
    )
    controls = {
        (role, operation): f"OWN-{index:03d}"
        for index, (role, _owner, operation) in enumerate(owner_keys, start=1)
    }
    positives = tuple(
        {
            "id": controls[(role, operation)],
            "phase": "OWNER_CONTROL_FIRST",
            "actor": role,
            "owner": role,
            "operation": operation,
            "expected": "SUCCESS",
            "owner_control": "SELF",
        }
        for role, _owner, operation in owner_keys
    )
    negatives = tuple(
        {
            "id": f"CROSS-{index:03d}",
            "phase": "CROSS_ROLE_AFTER_OWNER",
            "actor": actor,
            "owner": owner,
            "operation": operation,
            "expected": "EACCES_OR_EPERM",
            "owner_control": controls[(owner, operation)],
        }
        for index, (actor, owner, operation) in enumerate(cross_keys, start=1)
    )
    return positives + negatives


_STEPS = _canonical_steps()
# All input still remains untrusted. These are local structural invariants.
assert len(_STEPS) == 108
assert sum(row["phase"] == "OWNER_CONTROL_FIRST" for row in _STEPS) == 18
assert sum(row["phase"] == "CROSS_ROLE_AFTER_OWNER" for row in _STEPS) == 90


def build_execution_plan() -> dict:
    """Return fresh inert descriptors for later reviewed native TEST adapters."""
    return {
        "schema": _SCHEMA,
        "snapshot": _SNAPSHOT,
        "steps": [row.copy() for row in _STEPS],
    }


def inspect_execution_plan(record: object = None) -> dict:
    """Check exact order, dependencies and complete coverage, never approve."""
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "NOT_QUALIFIED",
        "blockers": list(_BLOCKERS),
        "plan": {
            "observation": "NOT_QUALIFIED",
            "reason": "NO_TEST_EXECUTION_PLAN",
        },
        "counts": {
            "expected_total": len(_STEPS),
            "expected_owner_controls": 18,
            "expected_cross_role": 90,
            "accepted_owner_controls": 0,
            "accepted_cross_role": 0,
            "missing": len(_STEPS),
        },
    }
    if record is None:
        return result

    def invalid() -> dict:
        result["plan"]["reason"] = "INVALID_TEST_EXECUTION_PLAN"
        return result

    if (
        type(record) is not dict
        or set(record) != _TOP_FIELDS
        or any(type(key) is not str for key in record)
        or type(record["schema"]) is not str
        or record["schema"] != _SCHEMA
        or type(record["snapshot"]) is not str
        or record["snapshot"] != _SNAPSHOT
    ):
        return invalid()
    rows = record["steps"]
    if type(rows) is not list or len(rows) > len(_STEPS):
        return invalid()
    for row in rows:
        if (
            type(row) is not dict
            or set(row) != _EXPECTED_FIELDS
            or any(type(key) is not str for key in row)
            or any(type(value) is not str for value in row.values())
        ):
            return invalid()

    # A valid partial plan must be a canonical prefix. Do not treat a
    # reordered or tampered set of rows as a harmless incomplete run.
    if rows != list(_STEPS[: len(rows)]):
        return invalid()
    result["snapshot"] = _SNAPSHOT
    result["counts"] = {
        "expected_total": len(_STEPS),
        "expected_owner_controls": 18,
        "expected_cross_role": 90,
        "accepted_owner_controls": min(len(rows), 18),
        "accepted_cross_role": max(0, len(rows) - 18),
        "missing": len(_STEPS) - len(rows),
    }
    if len(rows) != len(_STEPS):
        result["plan"]["reason"] = "INCOMPLETE_TEST_EXECUTION_PLAN"
    else:
        result["plan"] = {
            "observation": "UNTRUSTED_OFFLINE_TEST_PLAN_ONLY",
            "reason": "NO_NATIVE_EXECUTION_OR_INDEPENDENT_HOST_EVIDENCE",
        }
    return result


def main() -> int:
    """Default CLI shows fixed planned counts, without executing any step."""
    print(json.dumps(inspect_execution_plan(build_execution_plan()), sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
