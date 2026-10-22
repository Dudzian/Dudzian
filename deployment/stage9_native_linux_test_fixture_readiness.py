"""S9-RPI-36: pure TEST-only Linux fixture layout declaration validator.

Only checks internally consistent, untrusted TEST layout claims. It does not
inspect or provision host identities, protected files, or Linux services.
"""

from __future__ import annotations

import json

from deployment import stage9_native_linux_test_syscall_outcome_classifier as outcomes

_SCHEMA = "CryptoHunter.Stage9TestFixtureReadinessV1"
_ROLES = tuple(dict.fromkeys(actor for actor, _, _ in outcomes.expected_cases()))
_REQUIRED = frozenset({"schema", "snapshot", "roles"})
_ROLE_KEYS = frozenset(
    {
        "role",
        "uid",
        "gid",
        "effective_gids",
        "directory_owner_uid",
        "directory_owner_gid",
        "directory_mode",
        "file_owner_uid",
        "file_owner_gid",
        "file_mode",
        "symlink_free_claim",
        "acl_absent_claim",
        "unprivileged_claim",
        "private_parent_chain_claim",
    }
)
_CLAIMS = (
    "symlink_free_claim",
    "acl_absent_claim",
    "unprivileged_claim",
    "private_parent_chain_claim",
)
_ID_FIELDS = (
    "uid",
    "gid",
    "directory_owner_uid",
    "directory_owner_gid",
    "file_owner_uid",
    "file_owner_gid",
)
_BLOCKERS = (
    "TEST_FIXTURE_DECLARATIONS_ARE_NOT_HOST_EVIDENCE",
    "NO_INDEPENDENT_EFFECTIVE_UID_GID_OR_PROCESS_PROVENANCE",
    "NO_PINNED_INODES_PATH_COMPONENTS_OR_ACTUAL_PERMISSION_DENIALS",
    "NO_AUTHENTICATED_SOCKET_BOUND_PEER_IDENTITY",
    "NO_NATIVE_KEYRING_OR_PROTECTED_SOURCE_CUSTODY_PROOF",
    "NO_POSTGRESQL_COMMITTED_CLOSE_OR_CCA_DURABILITY_PROOF",
    "PRODUCTION_ROOT_PROOF_ISSUANCE_AND_ENROLLMENT_BLOCKED",
)


def expected_roles() -> tuple[str, ...]:
    """Fixed six role labels, not production principal names or host paths."""
    return _ROLES


def inspect_declared_fixture(record: object = None) -> dict:
    """Validate only closed, untrusted TEST fixture layout declarations."""
    result = {
        "schema": _SCHEMA,
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "snapshot": "NOT_QUALIFIED",
        "blockers": list(_BLOCKERS),
        "fixture": {"observation": "NOT_QUALIFIED", "reason": "NO_TEST_FIXTURE"},
        "counts": {
            "expected_roles": len(_ROLES),
            "declared_roles": 0,
            "missing_roles": len(_ROLES),
            "unsafe_roles": 0,
            "identity_collisions": 0,
            "planned_cross_role_cases": len(outcomes._CROSS_CASES),
            "planned_owner_controls": len(outcomes._OWNER_CONTROLS),
        },
    }
    if record is None:
        return result

    def invalid() -> dict:
        result["fixture"]["reason"] = "INVALID_TEST_FIXTURE"
        return result

    if (
        type(record) is not dict
        or any(type(key) is not str for key in record)
        or set(record) != _REQUIRED
        or type(record["schema"]) is not str
        or record["schema"] != _SCHEMA
    ):
        return invalid()
    snapshot = record["snapshot"]
    if type(snapshot) is not str or snapshot not in (
        "SYNTHETIC_TEST_INPUT",
        "UNTRUSTED_LIVE_OBSERVATION",
    ):
        return invalid()
    result["snapshot"] = snapshot
    rows = record["roles"]
    if type(rows) is not list or len(rows) > len(_ROLES):
        return invalid()

    observed = {}
    for row in rows:
        if (
            type(row) is not dict
            or any(type(key) is not str for key in row)
            or set(row) != _ROLE_KEYS
            or type(row["role"]) is not str
            or row["role"] not in _ROLES
            or row["role"] in observed
            or any(type(row[field]) is not int for field in _ID_FIELDS)
            or type(row["directory_mode"]) is not int
            or type(row["file_mode"]) is not int
            or any(type(row[field]) is not bool for field in _CLAIMS)
        ):
            return invalid()
        groups = row["effective_gids"]
        if (
            type(groups) is not list
            or not 1 <= len(groups) <= 16
            or any(type(group) is not int for group in groups)
            or len(set(groups)) != len(groups)
        ):
            return invalid()
        observed[row["role"]] = row

    unsafe = 0
    for row in observed.values():
        uid, gid = row["uid"], row["gid"]
        if (
            not 0 < uid < 2**32 - 1
            or not 0 < gid < 2**32 - 1
            or any(not 0 < row[field] < 2**32 - 1 for field in _ID_FIELDS)
            or row["effective_gids"] != [gid]
            or row["directory_owner_uid"] != uid
            or row["directory_owner_gid"] != gid
            or (row["file_owner_uid"], row["file_owner_gid"]) != (uid, gid)
            or row["directory_mode"] != 0o700
            or row["file_mode"] != 0o600
            or not all(row[field] for field in _CLAIMS)
        ):
            unsafe += 1

    identities = list(observed.values())
    collisions = sum(
        a["uid"] == b["uid"] or bool(set(a["effective_gids"]) & set(b["effective_gids"]))
        for i, a in enumerate(identities)
        for b in identities[i + 1 :]
    )
    result["counts"].update(
        {
            "declared_roles": len(observed),
            "missing_roles": len(_ROLES) - len(observed),
            "unsafe_roles": unsafe,
            "identity_collisions": collisions,
        }
    )
    if unsafe:
        reason = "DECLARED_TEST_RESOURCE_OR_PRIVILEGE_GAP"
    elif collisions:
        reason = "DECLARED_ROLE_IDENTITY_COLLISION"
    elif len(observed) != len(_ROLES):
        reason = "INCOMPLETE_TEST_FIXTURE"
    else:
        result["fixture"] = {
            "observation": "UNTRUSTED_TEST_LAYOUT_DECLARATIONS_ONLY",
            "reason": "NO_INDEPENDENT_PROCESS_OR_FILESYSTEM_EVIDENCE",
        }
        return result
    result["fixture"]["reason"] = reason
    return result


def main() -> int:
    """No-argument read-only CLI: always BLOCKED, always exit 2."""
    print(json.dumps(inspect_declared_fixture(), sort_keys=True, ensure_ascii=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
