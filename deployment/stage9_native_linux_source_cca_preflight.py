"""Read-only native Linux inventory for proposed Stage 9 source/CCA principals.

This is a diagnostic preflight, not an installation, an independently pinned
manifest, an effective-access proof or a production qualification. No result
from this module authorizes Root Proof issuance or enrollment.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import sys


# PROPOSED topology only. Changing these defaults cannot approve installation.
_PROPOSED = (
    ("root_source", "ch_root_custody_source", "root-custody-source"),
    ("history_source", "ch_history_custody_source", "history-custody-source"),
    ("cca", "ch_custody_completion", "custody-completion"),
)
_BLOCKERS = (
    "INDEPENDENTLY_APPROVED_INSTALLATION_PIN_MISSING",
    "RUNTIME_SERVICE_IDENTITY_AND_SYSTEMD_NOT_OBSERVED",
    "CROSS_ROLE_EFFECTIVE_ACCESS_AND_NATIVE_KEYRING_NOT_QUALIFIED",
    "INDEPENDENT_CREDENTIAL_HISTORY_AND_ROLLBACK_FLOOR_NOT_QUALIFIED",
    "CCA_COMMITTED_CLOSE_AND_DURABLE_EXECUTION_NOT_QUALIFIED",
    "PRODUCTION_ISSUER_AND_LEGAL_ENROLLMENT_BLOCKED",
)


def _mode(info):
    return stat.S_IMODE(info.st_mode)


def _relative(root: Path, location: str) -> Path:
    """Map a closed absolute candidate path into a temporary TEST root."""
    assert location.startswith("/") and ".." not in Path(location).parts
    return root / location.lstrip("/")


def _inspect_directory(root: Path, location: str, *, uid: int, gid: int, mode: int):
    """Only lstat metadata; never open private material or repair missing paths."""
    target = _relative(root, location)
    checked = []
    current = target
    while current != root:
        checked.append(current)
        current = current.parent
    checked.reverse()
    try:
        for index, node in enumerate(checked):
            info = node.lstat()
            if not stat.S_ISDIR(info.st_mode):
                return "NOT_QUALIFIED", "non-directory or symlink in path"
            if _mode(info) & 0o022:
                return "NOT_QUALIFIED", "writable ancestor or target"
            if index < len(checked) - 1:
                if info.st_uid != 0:
                    return "NOT_QUALIFIED", "ancestor not owned by root"
            else:
                if (info.st_uid, info.st_gid, _mode(info)) != (uid, gid, mode):
                    return "NOT_QUALIFIED", "target identity, group or mode mismatch"
            try:
                attributes = os.listxattr(node, follow_symlinks=False)
            except OSError:
                return "NOT_QUALIFIED", "POSIX ACL inventory unavailable"
            if any(name.startswith("system.posix_acl_") for name in attributes):
                return "NOT_QUALIFIED", "extended POSIX ACL present"
    except (OSError, ValueError):
        return "NOT_QUALIFIED", "path missing or inaccessible"
    return "MATCHED_PROPOSED_METADATA_ONLY", "no private contents inspected"


def _inspect_accounts(accounts):
    """Pure exact-role checks; a numeric identity is never an identity attestation."""
    issues = []
    identities = {}
    for role, name, _subdir in _PROPOSED:
        item = accounts.get(name)
        if item is None:
            issues.append(f"{role}: account missing or unobservable")
            continue
        uid, gid, groups = item
        if (
            type(uid) is not int
            or type(gid) is not int
            or type(groups) is not tuple
            or not groups
            or any(type(value) is not int for value in groups)
            or uid <= 0
            or gid <= 0
            or gid not in groups
        ):
            issues.append(f"{role}: malformed or privileged identity")
            continue
        identities[role] = {"name": name, "uid": uid, "gid": gid, "groups": groups}
    roles = list(identities)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            a, b = identities[left], identities[right]
            if a["uid"] == b["uid"]:
                issues.append(f"{left}/{right}: shared UID")
            if set(a["groups"]) & set(b["groups"]):
                issues.append(f"{left}/{right}: shared effective group")
    return identities, issues


def _native_accounts():
    # Resolve NSS via a read-only lookup; never create users or alter groups.
    import pwd

    accounts = {}
    for _role, name, _subdir in _PROPOSED:
        try:
            entry = pwd.getpwnam(name)
            groups = os.getgrouplist(name, entry.pw_gid)
        except (KeyError, OSError):
            continue
        accounts[name] = (entry.pw_uid, entry.pw_gid, tuple(sorted(set(groups))))
    return accounts


def inspect_host(*, root: Path = Path("/"), accounts=None):
    """Read-only snapshot. Even all matching proposals remain PRODUCTION_NO_GO."""
    if sys.platform != "linux":
        return {
            "schema": "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1",
            "status": "BLOCKED",
            "qualification": "NOT_QUALIFIED",
            "production_emission": "BLOCKED",
            "topology": "PROPOSED_NOT_APPROVED",
            "blockers": ["NATIVE_LINUX_REQUIRED", *_BLOCKERS],
            "principal_observations": {},
            "observed_failures": ["NATIVE_LINUX_REQUIRED"],
            "paths": {},
        }
    if accounts is None:
        accounts = _native_accounts()
    identities, issues = _inspect_accounts(accounts)
    paths = {}
    for role, _name, subdir in _PROPOSED:
        identity = identities.get(role)
        if identity is None:
            continue
        for kind, prefix, expected_uid, expected_gid, mode in (
            ("configuration", "/etc/cryptohunter", 0, identity["gid"], 0o750),
            ("private_state", "/var/lib/cryptohunter", identity["uid"], identity["gid"], 0o700),
        ):
            location = f"{prefix}/{subdir}"
            observed, explanation = _inspect_directory(
                root, location, uid=expected_uid, gid=expected_gid, mode=mode
            )
            paths[f"{role}.{kind}"] = {
                "path": location,
                "observation": observed,
                "reason": explanation,
            }
            if observed != "MATCHED_PROPOSED_METADATA_ONLY":
                issues.append(f"{role}.{kind}: {explanation}")
    return {
        "schema": "CryptoHunter.Stage9NativeLinuxSourceCCAPreflightV1",
        "status": "BLOCKED",
        "qualification": "NOT_QUALIFIED",
        "production_emission": "BLOCKED",
        "topology": "PROPOSED_NOT_APPROVED",
        "principal_observations": {
            role: {
                "name": identity["name"],
                "uid": identity["uid"],
                "gid": identity["gid"],
                "supplementary_gids": sorted(set(identity["groups"]) - {identity["gid"]}),
            }
            for role, identity in identities.items()
        },
        "paths": paths,
        "observed_failures": issues,
        "blockers": list(_BLOCKERS),
    }


def main():
    """No writable options, no user-controlled path, no live PASS exit code."""
    report = inspect_host()
    print(json.dumps(report, ensure_ascii=True, sort_keys=True))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
