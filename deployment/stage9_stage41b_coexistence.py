"""Read-only diagnostic evidence. No result grants a preflight exception.

Invoke only from externally authenticated, protected operator code. In particular
metadata checks in this interpreter cannot authenticate its startup imports.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any

from deployment import stage9_source_cca_test_runtime as host
from scripts import prepare_stage41b_environment as preparation

COMMITS = (
    "782816540373d7b31f426dab645ad115320d1234",
    "26e17e7c23ea6cc1b265d57bfda1a1e6589d9e3f",
)
LIMIT = 64 * 1024 * 1024


def _stamp(info: os.stat_result) -> tuple[int, ...]:
    # Reading may itself update atime; only content/security identity changes
    # invalidate the observation. Preserve inode/device positions for matching.
    return (
        info.st_mode,
        info.st_ino,
        info.st_dev,
        info.st_nlink,
        info.st_uid,
        info.st_gid,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _read(path: Path) -> bytes:
    """No-follow, bounded, stable file read; never imports installed code."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > LIMIT:
            raise ValueError("INVALID_FILE")
        chunks = []
        size = 0
        while block := os.read(descriptor, min(65536, LIMIT + 1 - size)):
            chunks.append(block)
            size += len(block)
            if size > LIMIT:
                raise ValueError("OVERSIZED_FILE")
        after = os.fstat(descriptor)
        if _stamp(before) != _stamp(after) or _stamp(after) != _stamp(path.lstat()):
            raise ValueError("FILE_CHANGED")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _inventory(root: Path) -> dict[str, Any]:
    # Every ancestor is protected before walking; no untrusted directory can
    # replace a checked child. A second walk detects changes between reads.
    for directory in (*reversed(root.parents), root):
        host._protected(directory, directory=True)
    files: dict[str, str] = {}
    directories = []
    identities = []

    def failed(error: OSError) -> None:
        raise error

    for current, children, names in os.walk(root, followlinks=False, onerror=failed):
        directory = Path(current)
        info = host._protected(directory, directory=True)
        directories.append(directory.relative_to(root).as_posix())
        identities.append((str(directory), _stamp(info)))
        for child in children:
            host._protected(directory / child, directory=True)
        for name in names:
            path = directory / name
            info = host._protected(path, directory=False)
            files[path.relative_to(root).as_posix()] = hashlib.sha256(_read(path)).hexdigest()
            identities.append((str(path), _stamp(info)))
    return {"files": files, "directories": sorted(directories), "identities": sorted(identities)}


def installation(commit: str, expected: object) -> dict[str, Any]:
    result: dict[str, Any] = {"commit": commit, "integrity": "UNCERTAIN", "provenance": "UNCERTAIN"}
    try:
        if (
            commit not in COMMITS
            or type(expected) is not dict
            or set(expected) != {"files", "directories"}
        ):
            raise ValueError("INDEPENDENT_INVENTORY_REQUIRED")
        root = preparation.PREFIX / commit
        before = _inventory(root)
        # This is reviewed current code, never code imported from root/source.
        # It checks the receipt, Git Merkle proof against the full commit, wheel
        # lock and installed wheel members, plus interpreter and venv settings.
        preparation.verify_installation(root, commit)
        after = _inventory(root)
        if before != after:
            raise ValueError("INSTALLATION_CHANGED")
        result["integrity"] = "PASS"
        result["file_count_including_receipt"] = len(before["files"])
        result["directory_count"] = len(before["directories"])
        if expected != {key: before[key] for key in ("files", "directories")}:
            result["provenance"] = "BLOCKED"
            result["reason"] = "INDEPENDENT_INVENTORY_MISMATCH"
        else:
            result["provenance"] = "PASS"
            result["provenance_basis"] = "EXTERNALLY_PINNED_INVENTORY_AND_GIT_PROOF"
    except Exception:
        result["reason"] = "INSTALLATION_OBSERVATION_OR_VERIFICATION_FAILED"
    return result


def evidence(raw: bytes, external_sha256: str) -> dict[str, Any]:
    """The pin must come from independent review, never from the Dell receipt."""
    if (
        not re.fullmatch(r"[0-9a-f]{64}", external_sha256)
        or hashlib.sha256(raw).hexdigest() != external_sha256
    ):
        raise ValueError("EXTERNAL_EVIDENCE_DIGEST_MISMATCH")
    value = preparation._json(raw)
    if (
        type(value) is not dict
        or set(value) != {"schema", "installations"}
        or type(value["schema"]) is not int
        or value["schema"] != 1
    ):
        raise ValueError("EXTERNAL_EVIDENCE_SCHEMA")
    if type(value["installations"]) is not dict or set(value["installations"]) != set(COMMITS):
        raise ValueError("EXACT_TWO_INSTALLATIONS_REQUIRED")
    return value["installations"]


def _listing(path: Path) -> list[Path]:
    result = []
    for item in path.iterdir():
        result.append(item)
        if len(result) > 65536:
            raise ValueError("ENUMERATION_LIMIT")
    return sorted(result)


def _installation_metadata(root: Path) -> dict[str, Any]:
    """Protected lstat identities for correlation only; never read file contents.

    Private 0600 files need only directory search permission for lstat. Partial
    metadata is retained for positive matches; it cannot authenticate any bytes.
    Unsafe directories are not traversed and every gap remains explicit.
    """
    identities: list[tuple[str, tuple[int, ...]]] = []
    complete = True
    try:
        for parent in reversed(root.parents):
            host._protected(parent, directory=True)
        pending = [root]
        while pending:
            directory = pending.pop()
            try:
                info = host._protected(directory, directory=True)
                identities.append((str(directory), _stamp(info)))
                children = _listing(directory)
            except Exception:
                complete = False
                continue
            for child in children:
                if len(identities) + len(pending) >= 65536:
                    return {"identities": sorted(identities), "complete": False}
                try:
                    info = child.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        # Recheck protection before subsequent enumeration.
                        host._protected(child, directory=True)
                        pending.append(child)
                    else:
                        info = host._protected(child, directory=False)
                        identities.append((str(child), _stamp(info)))
                except Exception:
                    complete = False
    except Exception:
        complete = False
    return {"identities": sorted(identities), "complete": complete}


def _proc_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    if not raw or len(raw) > LIMIT:
        raise ValueError("PROC_CONTENT_UNAVAILABLE_OR_OVERSIZED")
    return raw


def _references(snapshot: tuple[Any, ...], inodes: set[tuple[int, int]]) -> bool:
    # Match device/inode for hardlink aliases, renamed/deleted files and maps
    # whose pathname differs from the installation pathname.
    if sys.platform != "linux":
        raise ValueError("NATIVE_LINUX_REQUIRED")
    # Preserve a positive pathname even if another field cannot be parsed.
    if str(preparation.PREFIX).encode() in repr(snapshot).encode():
        return True
    for _tid, _start, _status, cgroup, maps, links in snapshot:
        if any((device, inode) in inodes for _target, device, inode in links):
            return True
        if any(marker in cgroup for marker in (b"ch-stage41b-", b"cryptohunter-stage41b-")):
            return True
        for line in maps.splitlines():
            fields = line.split(None, 5)
            if len(fields) < 5:
                raise ValueError("MAPS_INVALID")
            major, minor = fields[3].split(b":")
            if (os.makedev(int(major, 16), int(minor, 16)), int(fields[4])) in inodes:
                return True
    return False


def _pid_snapshot(item: Path) -> tuple[Any, ...]:
    """All visible threads: credentials, executable, fd, maps, cwd and root.

    Paths can be renamed/deleted, so record inode identities as well as names.
    No absence claim follows from an ENOENT, a kernel thread, or a zombie.
    """
    records = []
    tasks = _listing(item / "task")
    if not tasks:
        raise ValueError("NO_THREADS")
    if len(tasks) > 1024:
        raise ValueError("THREAD_LIMIT")
    for task in tasks:
        start = _proc_bytes(task / "stat")
        status = _proc_bytes(task / "status")
        cgroup = _proc_bytes(task / "cgroup")
        maps = _proc_bytes(task / "maps")
        links = []
        for path in [task / name for name in ("exe", "cwd", "root")] + _listing(task / "fd"):
            target = os.readlink(path)
            info = path.stat()
            links.append((target, info.st_dev, info.st_ino))
        records.append((task.name, start, status, cgroup, maps, tuple(links)))
        if len(repr(records)) > LIMIT:
            raise ValueError("PROCESS_SNAPSHOT_LIMIT")
    return tuple(records)


def processes() -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": "UNCERTAIN",
        "scope": "VISIBLE_PROCFS_ALL_THREADS_EXE_CWD_ROOT_FD_MAPS",
        "full_confirmation": False,
        "stable_visible_snapshot": False,
        "matched_pids": [],
        "checked_pid_count": 0,
        "unavailable_pid_count": 0,
        "unstable_pid_count": 0,
        "matched_pid_count": 0,
        "observed_pid_count": 0,
        "observed_thread_count": 0,
        "limitations": [
            "NO_ATOMIC_HOST_SNAPSHOT",
            "NO_PROOF_FOR_COPIED_OR_CLOSED_FILES",
            "HOST_NAMESPACE_NOT_EXTERNALLY_ATTESTED",
        ],
    }
    checked: set[int] = set()
    unavailable: set[int] = set()
    unstable: set[int] = set()
    matched: set[int] = set()
    reasons: set[str] = set()
    threads: set[tuple[int, str]] = set()
    inodes: set[tuple[int, int]] = set()

    def metadata() -> list[dict[str, Any]]:
        observations = []
        for commit in COMMITS:
            try:
                observation = _installation_metadata(preparation.PREFIX / commit)
                inodes.update((info[2], info[1]) for _path, info in observation["identities"])
                if observation["complete"] is not True:
                    reasons.add("INSTALLATION_METADATA_INCOMPLETE")
                observations.append(observation)
            except Exception:
                reasons.add("INSTALLATION_METADATA_INCOMPLETE")
                observations.append({"identities": [], "complete": False})
        return observations

    def pids() -> dict[int, Path]:
        try:
            items = {
                int(path.name): path
                for path in _listing(Path("/proc"))
                if path.name.isascii() and path.name.isdigit()
            }
            if 1 not in items or os.getpid() not in items:
                reasons.add("PROCFS_INCOMPLETE")
            return items
        except Exception:
            reasons.add("PROCESS_ENUMERATION_UNAVAILABLE")
            return {}

    def sweep(items: dict[int, Path]) -> dict[int, str]:
        # Keep bounded fingerprints, not all procfs payloads between sweeps.
        observations = {}
        observed_size = 0
        for pid, item in items.items():
            if observed_size > LIMIT:
                unavailable.add(pid)
                reasons.add("PROCESS_SWEEP_LIMIT")
                continue
            try:
                snapshot = _pid_snapshot(item)
                if not snapshot:
                    raise ValueError("NO_THREADS")
                checked.add(pid)
                threads.update((pid, row[0]) for row in snapshot)
                # Both passes classify references; positive findings are sticky.
                if _references(snapshot, inodes):
                    matched.add(pid)
                raw = repr(snapshot).encode()
                observed_size += len(raw)
                if observed_size > LIMIT:
                    raise ValueError("PROCESS_SWEEP_LIMIT")
                observations[pid] = hashlib.sha256(raw).hexdigest()
            except FileNotFoundError:
                unstable.add(pid)
                reasons.add("PROCESS_DISAPPEARED_DURING_OBSERVATION")
            except Exception:
                unavailable.add(pid)
                reasons.add("PROCESS_OBSERVATION_UNAVAILABLE")
        return observations

    try:
        if not host._procfs_visibility():
            reasons.add("PROCFS_VISIBILITY_UNCERTAIN")
    except Exception:
        reasons.add("PROCFS_VISIBILITY_UNCERTAIN")
    before = metadata()
    initial = pids()
    first = sweep(initial)
    current = pids()
    unstable.update(initial.keys() ^ current.keys())
    # Retry initially unavailable PIDs and observe newly visible ones as well.
    second = sweep(initial | current)
    final = pids()
    after = metadata()
    metadata_stable = before == after and all(item["complete"] is True for item in before + after)
    if not metadata_stable:
        reasons.add("INSTALLATION_METADATA_INCOMPLETE_OR_CHANGED")
    unstable.update(current.keys() ^ final.keys())
    unstable.update(pid for pid in first.keys() & second.keys() if first[pid] != second[pid])
    if unstable:
        reasons.add("PROCESS_OBSERVATION_CHANGED")
    result.update(
        {
            "state": "BLOCKED" if matched else "UNCERTAIN",
            "matched_pids": sorted(matched),
            "checked_pid_count": len(checked),
            "unavailable_pid_count": len(unavailable),
            "unstable_pid_count": len(unstable),
            "matched_pid_count": len(matched),
            "unavailable_pids": sorted(unavailable),
            "unstable_pids": sorted(unstable),
            "observed_pid_count": len(checked),
            "observed_thread_count": len(threads),
            "stable_visible_snapshot": not reasons,
            "uncertainty": sorted(reasons),
            "installation_metadata": {
                "scope": "PROTECTED_PATH_METADATA_ONLY",
                "stable": metadata_stable,
                "observed_entry_count": sum(len(item["identities"]) for item in after),
                "integrity_verified": False,
                "provenance_verified": False,
            },
        }
    )
    if reasons:
        result["reason"] = "PROCESS_OBSERVATION_INCOMPLETE_OR_CHANGED"
    return result


def resources(plan: dict[str, Any]) -> dict[str, Any]:
    """Reuse exact name AND numeric NSS queries, unused-unit and cgroup gates."""
    result: dict[str, Any] = {"state": "UNCERTAIN", "scope": "KNOWN_RESOURCES_AND_SOURCE_CCA_PLAN"}
    try:
        before = host._collisions(plan)
        legacy = host._stage41b_diagnostics()
        accounts = {database: host._nss(database) for database in ("passwd", "group")}
        legacy_ids = {
            "uid": sorted(
                {int(row[2]) for row in accounts["passwd"] if row[0].startswith("chs41b")}
            ),
            "gid": sorted(
                {int(row[2]) for row in accounts["group"] if row[0].startswith("chs41b")}
                | {int(row[3]) for row in accounts["passwd"] if row[0].startswith("chs41b")}
            ),
        }
        aliases = {
            "passwd": [
                row[0]
                for row in accounts["passwd"]
                if int(row[2]) in legacy_ids["uid"] or int(row[3]) in legacy_ids["gid"]
            ],
            "group": [row[0] for row in accounts["group"] if int(row[2]) in legacy_ids["gid"]],
        }
        result["legacy_numeric_ids"] = legacy_ids
        result["legacy_identity_aliases"] = aliases
        if accounts != {database: host._nss(database) for database in ("passwd", "group")}:
            raise ValueError("NSS_OBSERVATION_CHANGED")
        after = host._collisions(plan)
        if before != after or legacy != host._stage41b_diagnostics():
            raise ValueError("RESOURCE_OBSERVATION_CHANGED")
        result["source_cca_checks"] = before
        result["legacy"] = legacy
        result["state"] = (
            "BLOCKED"
            if (
                not all(value is True for value in before.values())
                or any(aliases.values())
                or any(value.get("names") for value in legacy["units"].values())
                or any(
                    value.get("count_up_to_limit", 0) for value in legacy["runtime_paths"].values()
                )
            )
            else "UNCERTAIN"
        )
        # NSS enumeration and name-based legacy discovery are not a closed
        # resource inventory. Inactive units, aliases, mount/socket/network
        # reservations outside these known names still need independent review.
        result["limitations"] = [
            "LEGACY_RESOURCE_INVENTORY_NOT_CLOSED",
            "NSS_ENUMERATION_NOT_PROOF_OF_ABSENCE",
            "UNIT_CONTENT_AND_ALIASES_NOT_FULLY_VERIFIED",
        ]
    except Exception:
        result["reason"] = "RESOURCE_OBSERVATION_INCOMPLETE_OR_CHANGED"
    return result


def diagnose(plan: dict[str, Any], raw: bytes, pin: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": 1,
        "read_only": True,
        "qualification": "BLOCKED",
        "coexistence_authorized": False,
        "operator_authorization": "NOT_CONSUMED_DIAGNOSTIC_ONLY",
        "physical_test_status": "PHYSICAL_TEST_PENDING",
        "status": "IN_PROGRESS",
        "production_status": "PRODUCTION_NO_GO",
        "preflight_exception": False,
        "evidence_pin_verified": False,
    }
    try:
        expectations = evidence(raw, pin)
        report["evidence_pin_verified"] = True
        report["installations"] = [installation(commit, expectations[commit]) for commit in COMMITS]
    except Exception:
        report["installations"] = [
            {
                "commit": commit,
                "integrity": "UNCERTAIN",
                "provenance": "BLOCKED",
                "reason": "INDEPENDENT_EVIDENCE_UNAVAILABLE_OR_INVALID",
            }
            for commit in COMMITS
        ]
    report["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
    report["resources"] = resources(plan)
    report["process_observation"] = processes()
    report["blockers"] = [
        "NO_COMPLETE_INDEPENDENT_HOST_ISOLATION_PROOF",
        "NO_OTHER_INSTALLATIONS_GATE_RETAINED",
    ]
    return report
