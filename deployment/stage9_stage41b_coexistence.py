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


class _EnumerationError(ValueError):
    def __init__(self, partial: list[Path], issues: list[str]) -> None:
        super().__init__("PARTIAL_ENUMERATION")
        self.partial = partial
        self.issues = issues


def _listing(path: Path) -> list[Path]:
    result, issues = _partial_listing(path, 65536)
    if issues:
        raise _EnumerationError(result, issues)
    return result


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


PF_KTHREAD = 0x00200000
# Only documented operational fields are omitted from the stable fingerprint.
# All other stat fields (including flags, credentials-related fields and starttime)
# and every unrecognized status field remain significant.
_STAT_OPERATIONAL = {3, *range(10, 18), 35, 39, 42, 43, 44}
_STATUS_OPERATIONAL = {
    b"State",
    b"VmPeak",
    b"VmSize",
    b"VmHWM",
    b"VmRSS",
    b"RssAnon",
    b"RssFile",
    b"RssShmem",
    b"VmData",
    b"VmStk",
    b"VmExe",
    b"VmLib",
    b"VmPTE",
    b"VmSwap",
    b"voluntary_ctxt_switches",
    b"nonvoluntary_ctxt_switches",
}
_REASON_LIMIT = 65536


class _Issues(dict[str, int]):
    """Finite reason vocabulary with saturated event counts, never raw errors."""

    def append(self, code: str, count: int = 1) -> None:
        self[code] = min(_REASON_LIMIT, self.get(code, 0) + count)

    def extend(self, values: Any) -> None:
        if isinstance(values, dict):
            for code, count in values.items():
                self.append(code, count)
        else:
            for code in values:
                self.append(code)

    def remove(self, code: str) -> None:
        self.pop(code, None)


class _IdentityChanged(ValueError):
    pass


class _ProcReadError(ValueError):
    def __init__(self, reason: str, partial: bytes = b"") -> None:
        super().__init__(reason)
        self.reason = reason
        self.partial = partial


def _proc_bytes(path: Path, *, maximum: int | None = None) -> bytes:
    bound = LIMIT if maximum is None else min(LIMIT, max(0, maximum))
    with path.open("rb") as stream:
        raw = stream.read(bound + 1)
    if len(raw) > bound:
        raise _ProcReadError("LIMIT", raw[:bound])
    if not raw:
        raise _ProcReadError("EMPTY")
    return raw


def _failure(error: Exception) -> str:
    # Never expose exception messages: they can contain procfs payloads or paths.
    if isinstance(error, PermissionError):
        return "PERMISSION_DENIED"
    if isinstance(error, FileNotFoundError):
        return "NOT_FOUND"
    if isinstance(error, _IdentityChanged):
        return "IDENTITY_CHANGED"
    if isinstance(error, _ProcReadError):
        return error.reason
    if isinstance(error, ValueError):
        return "MALFORMED"
    return "IO_ERROR"


def _stat_fields(raw: bytes, expected: int) -> tuple[tuple[int, int, bool], tuple[bytes, ...]]:
    # comm may itself contain spaces, newlines and parentheses. The final ') '
    # separates it from the numeric suffix; splitting the whole line is unsafe.
    opening, closing = raw.find(b" ("), raw.rfind(b") ")
    if opening < 1 or closing < opening + 2:
        raise ValueError("STAT_STRUCTURE")
    prefix = raw[:opening]
    fields = raw[closing + 2 :].split()
    if (
        not re.fullmatch(rb"[1-9][0-9]*", prefix)
        or len(fields) != 50
        or fields[0] not in {b"R", b"S", b"D", b"Z", b"T", b"t", b"X", b"x", b"K", b"W", b"P", b"I"}
        or any(not re.fullmatch(rb"-?[0-9]+", value) for value in fields[1:])
    ):
        raise ValueError("STAT_STRUCTURE_OR_IDENTITY")
    flags, start = fields[6], fields[19]
    if (
        not flags.isdigit()
        or not start.isdigit()
        or len(flags) > 10
        or len(start) > 20
        or int(flags) > 0xFFFFFFFF
        or int(start) > 0xFFFFFFFFFFFFFFFF
    ):
        raise ValueError("STAT_FLAGS_OR_STARTTIME")
    if int(prefix) != expected:
        raise _IdentityChanged("STAT_PID_CHANGED")
    identity = (expected, int(start), bool(int(flags) & PF_KTHREAD))
    stable = (prefix, raw[opening + 2 : closing]) + tuple(
        value for number, value in enumerate(fields, 3) if number not in _STAT_OPERATIONAL
    )
    return identity, stable


def _status_fields(raw: bytes, expected: int, leader: int) -> tuple[tuple[bytes, bytes], ...]:
    values = {}
    for line in raw.splitlines():
        key, separator, value = line.partition(b":")
        value = b" ".join(value.split())
        if not separator or not re.fullmatch(rb"[A-Za-z_][A-Za-z0-9_]*", key) or key in values:
            raise ValueError("STATUS_STRUCTURE")
        if key in {b"Uid", b"Gid"} and (
            len(value.split()) != 4
            or any(
                not token.isdigit() or len(token) > 10 or int(token) > 0xFFFFFFFF
                for token in value.split()
            )
        ):
            raise ValueError("STATUS_CREDENTIALS")
        if (
            key in {b"Pid", b"Tgid"}
            and value != str(expected if key == b"Pid" else leader).encode()
        ):
            raise _IdentityChanged("STATUS_IDENTITY")
        if key == b"Groups" and any(
            not token.isdigit() or len(token) > 10 or int(token) > 0xFFFFFFFF
            for token in value.split()
        ):
            raise ValueError("STATUS_GROUPS")
        values[key] = value
    if not {b"Pid", b"Tgid", b"Uid", b"Gid", b"Groups"}.issubset(values):
        raise ValueError("STATUS_INCOMPLETE")
    return tuple(
        sorted((key, value) for key, value in values.items() if key not in _STATUS_OPERATIONAL)
    )


def _map_identity(line: bytes) -> tuple[int, int]:
    if sys.platform != "linux":
        raise ValueError("NATIVE_LINUX_REQUIRED")
    fields = line.split(None, 5)
    if (
        len(fields) < 5
        or not re.fullmatch(rb"[0-9a-fA-F]+-[0-9a-fA-F]+", fields[0])
        or not re.fullmatch(rb"[r-][w-][x-][ps]", fields[1])
        or not re.fullmatch(rb"[0-9a-fA-F]+", fields[2])
        or not re.fullmatch(rb"[0-9a-fA-F]+:[0-9a-fA-F]+", fields[3])
        or not fields[4].isdigit()
        or len(fields[3]) > 17
        or len(fields[4]) > 20
    ):
        raise ValueError("MAPS_STRUCTURE")
    major, minor = fields[3].split(b":")
    return os.makedev(int(major, 16), int(minor, 16)), int(fields[4])


def _reference_observation(
    snapshot: tuple[Any, ...],
    inodes: set[tuple[int, int]],
    observed: set[tuple[int, int]] | None = None,
    maximum: int = 65536,
) -> tuple[bool, _Issues]:
    if sys.platform != "linux":
        raise ValueError("NATIVE_LINUX_REQUIRED")
    # A positive name remains useful even when parsing or inode lookup fails.
    matched = str(preparation.PREFIX) in repr(snapshot)
    issues = _Issues()

    def remember(identity: tuple[int, int]) -> None:
        nonlocal matched
        if identity in inodes:
            matched = True
        if observed is not None and identity[1]:
            if len(observed) < maximum or identity in observed:
                observed.add(identity)
            else:
                issues.append("PROCESS_INODE_LIMIT")

    for row in snapshot:
        try:
            cgroup = row[3]
            if not isinstance(cgroup, bytes):
                raise ValueError("CGROUP_TYPE")
            if any(marker in cgroup for marker in (b"ch-stage41b-", b"cryptohunter-stage41b-")):
                matched = True
        except (TypeError, ValueError, IndexError):
            issues.append("CGROUP_MALFORMED")
        try:
            links = row[5]
            for link in links:
                try:
                    if type(link[1]) is int and type(link[2]) is int:
                        remember((link[1], link[2]))
                except (TypeError, ValueError, IndexError):
                    issues.append("LINK_MALFORMED")
        except (TypeError, ValueError, IndexError):
            issues.append("LINK_MALFORMED")
        try:
            maps = row[4]
            if not isinstance(maps, bytes):
                raise ValueError("MAPS_TYPE")
            for line in maps.splitlines():
                try:
                    remember(_map_identity(line))
                except (ValueError, OverflowError):
                    issues.append("MAPS_MALFORMED")
        except (TypeError, ValueError, IndexError):
            issues.append("MAPS_MALFORMED")
    return matched, issues


def _references(snapshot: tuple[Any, ...], inodes: set[tuple[int, int]]) -> bool:
    return _reference_observation(snapshot, inodes)[0]


def _partial_listing(path: Path, limit: int) -> tuple[list[Path], list[str]]:
    result: list[Path] = []
    issues = []
    try:
        for child in path.iterdir():
            if len(result) == limit:
                issues.append("LIMIT")
                break
            result.append(child)
    except Exception as error:
        issues.append(_failure(error))
    return sorted(result), issues


def _directory_identity(path: Path) -> tuple[int, ...]:
    info = path.stat()
    return (info.st_mode, info.st_dev, info.st_ino, info.st_uid, info.st_gid)


def _proc_readlink(path: Path) -> str:
    """Keep raw targets while allowing module-local observation fault injection."""
    return os.readlink(path)


def _pid_snapshot(item: Path) -> tuple[Any, ...]:
    """Keep every independently available source, even after an identity failure.

    The first six row positions retain the raw thread observations for matching;
    row seven holds only structured validation/security metadata. Nothing here is
    serialized into the public report. A partial observation never proves absence.
    """
    records: list[Any] = []
    remaining = LIMIT
    leader = int(item.name) if item.name.isascii() and item.name.isdigit() else -1

    def observe(task: Path, tid: int) -> tuple[Any, ...]:
        nonlocal remaining
        issues = _Issues()
        observed = 0

        def capture(path: Path, source: str) -> bytes:
            nonlocal remaining, observed
            if remaining <= 0:
                issues.append(source + "_LIMIT")
                return b""
            try:
                raw = _proc_bytes(path, maximum=remaining)
                observed += 1
            except Exception as error:
                issues.append(source + "_" + _failure(error))
                raw = error.partial if isinstance(error, _ProcReadError) else b""
                observed += bool(raw)
            remaining -= len(raw)
            return raw

        def identity(raw: bytes) -> tuple[int, int, bool] | None:
            try:
                return _stat_fields(raw, tid)[0]
            except ValueError as error:
                if raw:
                    issues.append(
                        "STAT_IDENTITY_CHANGED"
                        if isinstance(error, _IdentityChanged)
                        else "STAT_MALFORMED"
                    )
                return None

        def directory() -> tuple[int, ...] | None:
            try:
                return _directory_identity(task)
            except Exception as error:
                issues.append("THREAD_" + _failure(error))
                return None

        security_before = directory()
        start = capture(task / "stat", "STAT")
        before = identity(start)
        status = capture(task / "status", "STATUS")
        try:
            if status:
                _status_fields(status, tid, leader)
        except ValueError as error:
            issues.append(
                "STATUS_IDENTITY_CHANGED"
                if isinstance(error, _IdentityChanged)
                else "STATUS_MALFORMED"
            )
        cgroup = capture(task / "cgroup", "CGROUP")
        if cgroup and any(
            not re.fullmatch(rb"[0-9]+:[^:\n]*:/[^\n]*", line) for line in cgroup.splitlines()
        ):
            issues.append("CGROUP_MALFORMED")
        maps = capture(task / "maps", "MAPS")
        links = []

        def link(path: Path, source: str) -> None:
            nonlocal remaining, observed
            if remaining <= 0:
                issues.append(source + "_LIMIT")
                return
            target = None
            stamp = None
            try:
                target = _proc_readlink(path)
                observed += 1
            except Exception as error:
                issues.append(source + "_" + _failure(error))
            # Inode matching is independent from the symlink target read.
            try:
                info = path.stat()
                stamp = _stamp(info)
                observed += 1
            except Exception as error:
                issues.append(source + "_" + _failure(error))
            entry = (target, stamp[2] if stamp else None, stamp[1] if stamp else None, stamp)
            if target is not None or stamp is not None:
                links.append(entry)
                remaining -= len(repr(entry))

        for name in ("exe", "cwd", "root"):
            link(task / name, name.upper())
        descriptors, errors = _partial_listing(task / "fd", 65536)
        issues.extend("FD_" + error for error in errors)
        for path in descriptors:
            link(path, "FD")
            if remaining <= 0:
                issues.append("FD_LIMIT")
                break
        end = capture(task / "stat", "STAT")
        after = identity(end)
        security_after = directory()
        if before is not None and after is not None and before != after:
            issues.append("STAT_IDENTITY_CHANGED")
        if security_before != security_after:
            issues.append("THREAD_CONTENT_CHANGED")
        coherent = before is not None and before == after
        # Empty maps has meaning only after the same TID/starttime/flag was read
        # on both sides. KTHREAD never exempts other inaccessible resources.
        is_kernel = before is not None and before == after and before[2]
        empty_kernel_maps = is_kernel and "MAPS_EMPTY" in issues
        if empty_kernel_maps:
            issues.remove("MAPS_EMPTY")
        meta = {
            "issues": issues,
            "identity": before if coherent else None,
            "kernel_thread": is_kernel,
            "empty_kernel_maps": bool(empty_kernel_maps),
            "observed": observed,
            "security": (security_before, security_after),
            "end_stat": end,
        }
        return (str(tid), start, status, cgroup, maps, tuple(links), meta)

    records.append(observe(item, leader))
    tasks, errors = _partial_listing(item / "task", 1024)
    records[0][6]["issues"].extend("THREAD_" + error for error in errors)
    if not tasks:
        records[0][6]["issues"].append("THREAD_EMPTY")
    for task in tasks:
        if not task.name.isascii() or not task.name.isdigit():
            records[0][6]["issues"].append("THREAD_MALFORMED")
            continue
        if int(task.name) != leader:
            if remaining <= 0:
                records[0][6]["issues"].append("PROCESS_LIMIT")
                break
            records.append(observe(task, int(task.name)))
    # Bracket the entire thread enumeration with the leader identity too.
    try:
        closing = _proc_bytes(item / "stat", maximum=max(0, remaining))
        final_identity = _stat_fields(closing, leader)[0]
        if records[0][6]["identity"] != final_identity:
            records[0][6]["issues"].append("STAT_IDENTITY_CHANGED")
            records[0][6]["identity"] = None
            records[0][6]["kernel_thread"] = False
            records[0][6]["empty_kernel_maps"] = False
    except Exception as error:
        records[0][6]["issues"].append("STAT_" + _failure(error))
        records[0][6]["identity"] = None
        records[0][6]["kernel_thread"] = False
        records[0][6]["empty_kernel_maps"] = False
    return tuple(records)


def _fingerprints(snapshot: tuple[Any, ...], pid: int) -> tuple[dict[str, Any], _Issues]:
    identities: dict[int, tuple[int, int, bool]] = {}
    stable = []
    operational = []
    issues = _Issues()
    observed = False
    kernel = False
    empty_kernel = False
    for row in snapshot:
        try:
            tid, raw_stat, status, cgroup, maps, links = row[:6]
            meta = row[6] if len(row) > 6 else {}
            issues.extend(meta.get("issues", ()))
            observed |= bool(meta.get("observed", any(row[1:6])))
            identity, stable_stat = _stat_fields(raw_stat, int(tid))
            if not identities and int(tid) != pid:
                issues.append("STAT_MALFORMED")
            if "identity" not in meta or meta["identity"] is not None:
                identities[int(tid)] = identity
            if int(tid) == pid:
                kernel = meta.get("kernel_thread", identity[2])
                empty_kernel = meta.get("empty_kernel_maps", False)
            stable_status = _status_fields(status, int(tid), pid) if status else ()
            end_stat = meta.get("end_stat", raw_stat)
            _, end_stable = _stat_fields(end_stat, int(tid))
            if stable_stat != end_stable:
                issues.append("PROCESS_CONTENT_CHANGED")
            stable.append(
                (tid, stable_stat, stable_status, cgroup, maps, links, meta.get("security"))
            )
            operational.append((raw_stat, end_stat, status))
        except (TypeError, ValueError, IndexError):
            issues.append("PROCESS_MALFORMED")
            # Still compare unparseable data conservatively; matching is separate.
            stable.append(row[:6])
    return {
        "identities": identities,
        "bindings": hashlib.sha256(repr(stable).encode()).hexdigest(),
        "operational": hashlib.sha256(repr(operational).encode()).hexdigest(),
        "observed": observed,
        "kernel": kernel,
        "empty_kernel": empty_kernel,
    }, issues


def processes() -> dict[str, Any]:
    result: dict[str, Any] = {
        "state": "UNCERTAIN",
        "scope": "VISIBLE_PROCFS_ALL_THREADS_EXE_CWD_ROOT_FD_MAPS",
        "full_confirmation": False,
        "stable_visible_snapshot": False,
        "limitations": [
            "NO_ATOMIC_HOST_SNAPSHOT",
            "NO_PROOF_FOR_COPIED_OR_CLOSED_FILES",
            "HOST_NAMESPACE_NOT_EXTERNALLY_ATTESTED",
        ],
    }
    checked: set[int] = set()
    unavailable: set[int] = set()
    unstable: set[int] = set()
    identity_unstable: set[int] = set()
    bindings_unstable: set[int] = set()
    operationally_changed: set[int] = set()
    kernel: set[int] = set()
    empty_kernel: set[int] = set()
    matched: set[int] = set()
    reasons: set[str] = set()
    counts: dict[str, int] = {}
    threads: set[tuple[int, str]] = set()
    inodes: set[tuple[int, int]] = set()

    def reason(code: str, count: int = 1) -> None:
        reasons.add(code)
        counts[code] = min(_REASON_LIMIT, counts.get(code, 0) + count)

    def metadata() -> list[dict[str, Any]]:
        observations = []
        for commit in COMMITS:
            try:
                observation = _installation_metadata(preparation.PREFIX / commit)
                inodes.update((info[2], info[1]) for _path, info in observation["identities"])
                if observation["complete"] is not True:
                    reason("INSTALLATION_METADATA_INCOMPLETE")
                observations.append(observation)
            except Exception:
                reason("INSTALLATION_METADATA_INCOMPLETE")
                observations.append({"identities": [], "complete": False})
        return observations

    def pids() -> dict[int, Path]:
        try:
            paths = _listing(Path("/proc"))
        except _EnumerationError as error:
            paths = error.partial
            for code in error.issues:
                reason("PROCESS_ENUMERATION_" + code)
        except Exception as error:
            reason("PROCESS_ENUMERATION_" + _failure(error))
            paths = []
        items = {
            int(path.name): path for path in paths if path.name.isascii() and path.name.isdigit()
        }
        if 1 not in items or os.getpid() not in items:
            reason("PROCFS_INCOMPLETE")
        return items

    def sweep(items: dict[int, Path]) -> dict[int, dict[str, Any]]:
        observations = {}
        observed_size = 0
        observed_inodes = 0
        for pid, item in items.items():
            if observed_size > LIMIT:
                unavailable.add(pid)
                reason("PROCESS_SWEEP_LIMIT")
                continue
            try:
                snapshot = _pid_snapshot(item)
                if not snapshot:
                    raise ValueError("NO_THREADS")
                # Classify before validation, fingerprinting or size accounting.
                references: set[tuple[int, int]] = set()
                positive, issues = _reference_observation(
                    snapshot, inodes, references, 65536 - observed_inodes
                )
                observed_inodes += len(references)
                if positive:
                    matched.add(pid)
                fingerprint, validation = _fingerprints(snapshot, pid)
                fingerprint["inodes"] = references
                issues.extend(validation)
                if fingerprint["observed"]:
                    checked.add(pid)
                threads.update((pid, row[0]) for row in snapshot)
                for code, count in issues.items():
                    reason(code, count)
                    if code.endswith("IDENTITY_CHANGED"):
                        identity_unstable.add(pid)
                        unstable.add(pid)
                    elif code.endswith("CONTENT_CHANGED"):
                        bindings_unstable.add(pid)
                        unstable.add(pid)
                    else:
                        unavailable.add(pid)
                        if code.endswith("NOT_FOUND"):
                            unstable.add(pid)
                observed_size += len(repr(snapshot))
                if observed_size > LIMIT:
                    unavailable.add(pid)
                    reason("PROCESS_SWEEP_LIMIT")
                observations[pid] = fingerprint
            except Exception as error:
                code = _failure(error)
                reason("PROCESS_" + code)
                if isinstance(error, FileNotFoundError):
                    unstable.add(pid)
                else:
                    unavailable.add(pid)
        return observations

    try:
        if not host._procfs_visibility():
            reason("PROCFS_VISIBILITY_UNCERTAIN")
    except Exception:
        reason("PROCFS_VISIBILITY_UNCERTAIN")
    before = metadata()
    initial = pids()
    first = sweep(initial)
    current = pids()
    unstable.update(initial.keys() ^ current.keys())
    second = sweep(initial | current)
    final = pids()
    after = metadata()
    # Late readable installation metadata can identify an alias seen earlier.
    matched.update(
        pid
        for observations in (first, second)
        for pid, observation in observations.items()
        if observation["inodes"] & inodes
    )
    metadata_stable = before == after and all(item["complete"] is True for item in before + after)
    if not metadata_stable:
        reason("INSTALLATION_METADATA_INCOMPLETE_OR_CHANGED")
    unstable.update(current.keys() ^ final.keys())
    for pid in first.keys() & second.keys():
        left, right = first[pid], second[pid]
        if any(
            left["identities"][tid] != right["identities"][tid]
            for tid in left["identities"].keys() & right["identities"].keys()
        ):
            identity_unstable.add(pid)
            unstable.add(pid)
            reason("PROCESS_IDENTITY_CHANGED")
        if left["bindings"] != right["bindings"]:
            bindings_unstable.add(pid)
            unstable.add(pid)
            reason("PROCESS_CONTENT_CHANGED")
        elif left["operational"] != right["operational"]:
            operationally_changed.add(pid)
        leader_identity = left["identities"].get(pid)
        if (
            leader_identity is not None
            and leader_identity == right["identities"].get(pid)
            and left["kernel"]
            and right["kernel"]
            and pid not in identity_unstable
        ):
            kernel.add(pid)
            if left["empty_kernel"] or right["empty_kernel"]:
                empty_kernel.add(pid)
    if unstable:
        reason("PROCESS_OBSERVATION_CHANGED")
    visibility_complete = not any(
        code.startswith(("PROCFS_", "PROCESS_ENUMERATION_")) for code in reasons
    )
    result.update(
        {
            "state": "BLOCKED" if matched else "UNCERTAIN",
            "matched_pids": sorted(matched),
            "checked_pid_count": len(checked),
            "unavailable_pid_count": len(unavailable),
            "unstable_pid_count": len(unstable),
            "identity_unstable_pid_count": len(identity_unstable),
            "bindings_unstable_pid_count": len(bindings_unstable),
            "operationally_changed_pid_count": len(operationally_changed),
            "kernel_thread_pid_count": len(kernel),
            "empty_kernel_maps_pid_count": len(empty_kernel),
            "matched_pid_count": len(matched),
            "unavailable_pids": sorted(unavailable),
            "unstable_pids": sorted(unstable),
            "observed_pid_count": len(checked),
            "observed_thread_count": len(threads),
            "stable_visible_snapshot": not reasons,
            "stable_process_identities": visibility_complete
            and not unavailable
            and not unstable
            and not identity_unstable,
            "stable_installation_bindings": visibility_complete
            and not unavailable
            and not unstable
            and not bindings_unstable
            and metadata_stable,
            "uncertainty": sorted(reasons),
            "reason_counts": dict(sorted(counts.items())),
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
