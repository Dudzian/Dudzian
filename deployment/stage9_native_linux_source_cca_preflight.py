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
_KNOWN_REFERENCE_AUTHORITIES = (
    ("root_signer", "cryptohunter-root-proof-signer"),
    ("history_signer", "cryptohunter-history-attestation-signer"),
    ("issuer", "cryptohunter-root-proof-issuer"),
)
# Candidate names only; no unit is created, enabled, or considered approved.
_PROPOSED_SYSTEMD_UNITS = (
    ("root_source", "cryptohunter-root-custody-source.service", "ch_root_custody_source"),
    ("history_source", "cryptohunter-history-custody-source.service", "ch_history_custody_source"),
    ("cca", "cryptohunter-custody-completion.service", "ch_custody_completion"),
)
_SYSTEMD_PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "User",
    "Group",
    "DynamicUser",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
    "SupplementaryGroups",
    "NoNewPrivileges",
    "PrivateNetwork",
    "PrivateIPC",
    "PrivateMounts",
    "PrivateTmp",
    "ProtectSystem",
    "ProtectHome",
    "RestrictAddressFamilies",
    "CapabilityBoundingSet",
    "AmbientCapabilities",
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
    checked.append(root)
    checked.reverse()
    try:
        for index, node in enumerate(checked):
            info = node.lstat()
            if not stat.S_ISDIR(info.st_mode):
                return "NOT_QUALIFIED", "non-directory or symlink in path"
            if _mode(info) & 0o022:
                return "NOT_QUALIFIED", "writable ancestor or target"
            if index < len(checked) - 1:
                # A hermetic TEST root can belong to the test runner. The live
                # filesystem root and all mapped ancestors still require root.
                test_root = node == root and root != Path(root.anchor)
                ancestor_uid = os.geteuid() if test_root else 0
                if info.st_uid != ancestor_uid:
                    return "NOT_QUALIFIED", "ancestor owner mismatch"
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
        # S9-RPI-06 requires the exclusive primary service group. There is no
        # approved supplementary membership exception, including GID 0 or a
        # foreign signer group that is not shared by these three roles.
        supplementary_gids = sorted(set(groups) - {gid})
        if supplementary_gids:
            issues.append(f"{role}: unapproved supplementary GIDs {supplementary_gids}")
    roles = list(identities)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            a, b = identities[left], identities[right]
            if a["uid"] == b["uid"]:
                issues.append(f"{left}/{right}: shared UID")
            if set(a["groups"]) & set(b["groups"]):
                issues.append(f"{left}/{right}: shared effective group")
    return identities, issues


def _inspect_reference_authorities(accounts, identities):
    """Observe fixed signer/issuer NSS identities without granting authority.

    Missing service accounts and aliases are never inferred or auto-created.
    Comparisons use effective NSS group memberships, not mere principal names.
    """
    references = {}
    missing = []
    problems = []
    for role, name in _KNOWN_REFERENCE_AUTHORITIES:
        item = accounts.get(name)
        if item is None:
            missing.append(role)
            continue
        try:
            uid, gid, groups = item
        except (TypeError, ValueError):
            problems.append(f"{role}: malformed reference principal")
            continue
        if (
            type(uid) is not int
            or type(gid) is not int
            or type(groups) is not tuple
            or not groups
            or any(type(value) is not int or value < 0 for value in groups)
            or uid <= 0
            or gid <= 0
            or gid not in groups
            or 0 in groups
        ):
            problems.append(f"{role}: malformed or privileged reference principal")
            continue
        references[role] = {"name": name, "uid": uid, "gid": gid, "groups": groups}
    principals = {**identities, **references}
    roles = list(principals)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            # Pairwise checks between the three proposed authorities are
            # already handled by _inspect_accounts; do not double-report.
            if left in identities and right in identities:
                continue
            a, b = principals[left], principals[right]
            if a["uid"] == b["uid"]:
                problems.append(f"{left}/{right}: shared UID")
            if set(a["groups"]) & set(b["groups"]):
                problems.append(f"{left}/{right}: shared effective group")
    return references, missing, problems


def _native_accounts():
    # Resolve NSS via a read-only lookup; never create users or alter groups.
    import pwd

    accounts = {}
    fixed_names = [name for _role, name, _subdir in _PROPOSED]
    fixed_names.extend(name for _role, name in _KNOWN_REFERENCE_AUTHORITIES)
    for name in fixed_names:
        try:
            entry = pwd.getpwnam(name)
            groups = os.getgrouplist(name, entry.pw_gid)
        except (KeyError, OSError):
            continue
        accounts[name] = (entry.pw_uid, entry.pw_gid, tuple(sorted(set(groups))))
    return accounts


def _evaluate_systemd_show(raw: str, *, unit: str, account: str):
    """Interpret only fixed public systemd properties, never installed authority."""
    if type(raw) is not str or len(raw) > 32768 or "\x00" in raw:
        return {"observation": "NOT_QUALIFIED", "issues": ["MALFORMED_SYSTEMD_RESPONSE"]}
    properties = {}
    for line in raw.splitlines():
        if "=" not in line:
            return {"observation": "NOT_QUALIFIED", "issues": ["MALFORMED_SYSTEMD_RESPONSE"]}
        name, value = line.split("=", 1)
        if name not in _SYSTEMD_PROPERTIES or name in properties:
            return {"observation": "NOT_QUALIFIED", "issues": ["UNEXPECTED_OR_DUPLICATE_PROPERTY"]}
        properties[name] = value
    if set(properties) != set(_SYSTEMD_PROPERTIES):
        return {"observation": "NOT_QUALIFIED", "issues": ["INCOMPLETE_SYSTEMD_RESPONSE"]}
    issues = []
    for field, expected in (
        ("Id", unit),
        ("LoadState", "loaded"),
        ("ActiveState", "active"),
        ("User", account),
        ("Group", account),
        ("DynamicUser", "no"),
        ("NoNewPrivileges", "yes"),
        ("PrivateNetwork", "yes"),
        ("PrivateIPC", "yes"),
        ("PrivateMounts", "yes"),
        ("PrivateTmp", "yes"),
        ("ProtectSystem", "strict"),
        ("ProtectHome", "yes"),
        ("RestrictAddressFamilies", "AF_UNIX"),
        ("SupplementaryGroups", ""),
        ("CapabilityBoundingSet", ""),
        ("AmbientCapabilities", ""),
        ("DropInPaths", ""),
    ):
        if properties[field] != expected:
            issues.append(f"{field}_NOT_MATCHED")
    pid = properties["MainPID"]
    if not pid.isascii() or not pid.isdecimal() or len(pid) > 10 or int(pid) <= 1:
        issues.append("MAIN_PID_NOT_OBSERVED")
    fragment = properties["FragmentPath"]
    if not fragment.startswith("/") or ".." in Path(fragment).parts:
        issues.append("UNIT_FRAGMENT_NOT_ABSOLUTE")
    # Declared values are not kernel effective identity, trusted unit bytes,
    # override provenance, executable fingerprint or a protected host pin.
    return {
        "observation": ("PROPOSED_DECLARATIONS_MATCHED_ONLY" if not issues else "NOT_QUALIFIED"),
        "issues": issues,
    }


def _observe_proposed_systemd_units(*, root: Path, runner=None, use_host=False):
    """Read-only observation. Never invokes systemctl for synthetic fixtures."""
    roles = {role: unit for role, unit, _name in _PROPOSED_SYSTEMD_UNITS}
    if sys.platform != "linux":
        return {
            role: {"unit": unit, "observation": "NOT_OBSERVED_NON_LINUX"}
            for role, unit in roles.items()
        }
    if root != Path("/") or (not use_host and runner is None):
        return {
            role: {"unit": unit, "observation": "NOT_OBSERVED_TEST_FIXTURE"}
            for role, unit in roles.items()
        }
    import subprocess

    if runner is None:
        runner = subprocess.run
    observations = {}
    for role, unit, account in _PROPOSED_SYSTEMD_UNITS:
        command = (
            "/usr/bin/systemctl",
            "--system",
            "show",
            "--all",
            "--no-pager",
            f"--property={','.join(_SYSTEMD_PROPERTIES)}",
            unit,
        )
        try:
            result = runner(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
            if result.returncode != 0:
                entry = {"observation": "NOT_QUALIFIED", "issues": ["SYSTEMD_UNAVAILABLE"]}
            else:
                entry = _evaluate_systemd_show(result.stdout, unit=unit, account=account)
        except UnicodeDecodeError:
            entry = {"observation": "NOT_QUALIFIED", "issues": ["MALFORMED_SYSTEMD_RESPONSE"]}
        except (OSError, TimeoutError, subprocess.TimeoutExpired):
            entry = {"observation": "NOT_QUALIFIED", "issues": ["SYSTEMD_UNAVAILABLE"]}
        observations[role] = {"unit": unit, **entry}
    return observations


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
            "reference_principal_observations": {},
            "unobserved_reference_principals": [
                role for role, _name in _KNOWN_REFERENCE_AUTHORITIES
            ],
            "observed_failures": ["NATIVE_LINUX_REQUIRED"],
            "systemd_observations": _observe_proposed_systemd_units(root=root),
            "paths": {},
        }
    use_host_systemd = accounts is None
    if accounts is None:
        accounts = _native_accounts()
    identities, issues = _inspect_accounts(accounts)
    references, missing, reference_issues = _inspect_reference_authorities(accounts, identities)
    issues.extend(reference_issues)
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
    systemd_observations = _observe_proposed_systemd_units(root=root, use_host=use_host_systemd)
    if use_host_systemd and root == Path("/"):
        for role, observation in systemd_observations.items():
            if observation["observation"] == "NOT_QUALIFIED":
                issues.extend(f"{role}.systemd: {reason}" for reason in observation["issues"])
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
        "reference_principal_observations": {
            role: {
                "name": identity["name"],
                "uid": identity["uid"],
                "gid": identity["gid"],
                "supplementary_gids": sorted(set(identity["groups"]) - {identity["gid"]}),
            }
            for role, identity in references.items()
        },
        "unobserved_reference_principals": missing,
        "systemd_observations": systemd_observations,
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
