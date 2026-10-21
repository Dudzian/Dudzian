"""S9-RPI-37: reconcile TEST-only fixture and syscall outcome declarations.

Pure Windows-compatible offline validation. No Linux access, host actions,
attestation, trusted evidence, or production authorization.
"""

from __future__ import annotations

import json
import re

from deployment import stage9_native_linux_test_fixture_readiness as fixture
from deployment import stage9_native_linux_test_syscall_outcome_classifier as outcomes

_SCHEMA = "CryptoHunter.Stage9TestOnlyIsolationReconciliationV1"
_KEYS = frozenset(
    {"schema", "fixture", "outcomes", "fixture_session_ref", "outcomes_session_ref"}
)
_SESSION_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
_BLOCKERS = (
    "DECLARED_TEST_FIXTURE_AND_SYSCALL_OUTCOMES_ARE_UNTRUSTED",
    "NO_INDEPENDENT_TEST_PROCESS_OR_RESOURCE_IDENTITY",
    "NO_ACTUAL_LINUX_KERNEL_SYSCALL_EVIDENCE",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NO_NATIVE_KEYRING_AND_PROTECTED_SOURCE_CUSTODY_PROOF",
    "NO_POSTGRESQL_COMMITTED_CLOSE_OR_CCA_DURABILITY_PROOF",
    "PRODUCTION_ROOT_PROOF_AND_LEGAL_ENROLLMENT_BLOCKED",
)


def inspect_declared_handoff(record: object = None) -> dict:
    """Reconcile two closed TEST reports without elevating either to trust."""
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "NOT_QUALIFIED",
        "blockers": list(_BLOCKERS),
        "reconciliation": {
            "observation": "NOT_QUALIFIED",
            "reason": "NO_TEST_HANDOFF",
        },
        "counts": {
            "roles_expected": len(fixture.expected_roles()),
            "roles_declared": 0,
            "roles_missing": len(fixture.expected_roles()),
            "fixture_unsafe": 0,
            "fixture_collisions": 0,
            "cross_expected": len(outcomes._CROSS_CASES),
            "cross_denied": 0,
            "cross_allowed": 0,
            "cross_inconclusive": 0,
            "cross_missing": len(outcomes._CROSS_CASES),
            "owner_expected": len(outcomes._OWNER_CONTROLS),
            "owner_success": 0,
            "owner_failed": 0,
            "owner_missing": len(outcomes._OWNER_CONTROLS),
        },
    }
    if record is None:
        return result
    if (
        type(record) is not dict
        or any(type(key) is not str for key in record)
        or set(record) != _KEYS
        or type(record["schema"]) is not str
        or record["schema"] != _SCHEMA
        or type(record["fixture"]) is not dict
        or type(record["outcomes"]) is not dict
        or type(record["fixture_session_ref"]) is not str
        or type(record["outcomes_session_ref"]) is not str
        or _SESSION_PATTERN.fullmatch(record["fixture_session_ref"]) is None
        or _SESSION_PATTERN.fullmatch(record["outcomes_session_ref"]) is None
        or record["fixture_session_ref"] == "0" * 32
        or record["outcomes_session_ref"] == "0" * 32
    ):
        result["reconciliation"]["reason"] = "INVALID_TEST_HANDOFF"
        return result

    if record["fixture_session_ref"] != record["outcomes_session_ref"]:
        result["reconciliation"]["reason"] = "SESSION_DECLARATIONS_DISAGREE"
        return result

    f = fixture.inspect_declared_fixture(record["fixture"])
    o = outcomes.inspect_declared_outcomes(record["outcomes"])
    if (
        f["fixture"]["reason"] == "INVALID_TEST_FIXTURE"
        or o["classification"]["reason"] == "INVALID_TEST_REPORT"
    ):
        result["reconciliation"]["reason"] = "INVALID_TEST_HANDOFF"
        return result

    if f["snapshot"] != o["snapshot"]:
        result["reconciliation"]["reason"] = "SNAPSHOT_DECLARATIONS_DISAGREE"
        return result

    result["snapshot"] = f["snapshot"]
    fc, oc = f["counts"], o["counts"]
    result["counts"] = {
        "roles_expected": fc["expected_roles"],
        "roles_declared": fc["declared_roles"],
        "roles_missing": fc["missing_roles"],
        "fixture_unsafe": fc["unsafe_roles"],
        "fixture_collisions": fc["identity_collisions"],
        "cross_expected": oc["cross_expected"],
        "cross_denied": oc["cross_denied"],
        "cross_allowed": oc["cross_allowed"],
        "cross_inconclusive": oc["cross_inconclusive"],
        "cross_missing": oc["cross_missing"],
        "owner_expected": oc["owner_expected"],
        "owner_success": oc["owner_success"],
        "owner_failed": oc["owner_failed"],
        "owner_missing": oc["owner_missing"],
    }
    if oc["cross_allowed"]:
        reason = "DECLARED_CROSS_ROLE_ACCESS"
    elif f["fixture"]["observation"] != "UNTRUSTED_TEST_LAYOUT_DECLARATIONS_ONLY":
        reason = "TEST_FIXTURE_NOT_COMPLETE"
    elif (
        o["classification"]["observation"]
        != "UNTRUSTED_DECLARED_DENIAL_WITH_OWNER_CONTROLS_ONLY"
    ):
        reason = "SYSCALL_OUTCOMES_NOT_COMPLETE"
    elif o["denial_matrix"]["observation"] != "UNTRUSTED_DECLARED_DENIALS_ONLY":
        reason = "CROSS_ROLE_MATRIX_NOT_COMPLETE"
    else:
        result["reconciliation"] = {
            "observation": "UNTRUSTED_RECONCILED_TEST_DECLARATIONS_ONLY",
            "reason": "NO_INDEPENDENT_LINUX_EVIDENCE",
        }
        return result
    result["reconciliation"]["reason"] = reason
    return result


def main() -> int:
    """No arguments, no host actions: emit blocked plan with exit code 2."""
    print(json.dumps(inspect_declared_handoff(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
