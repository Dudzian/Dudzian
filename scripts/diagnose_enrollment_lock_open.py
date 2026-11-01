"""Bounded POSIX openat contention probe; temporary files confer no authority.

Compare the previous nonexclusive create-or-open operation with exclusive
creation followed only on EEXIST by a noncreating open. Two native controls
use an already existing file and a full pathname. All contenders start at a
barrier, every trial uses a new private directory, and no retry or delay can
hide an errno. A clean legacy result means NOT_REPRODUCED, never proof that
the platform defect is absent. This script does not use production state.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import platform
import stat
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier


def _object(info: os.stat_result) -> dict[str, int]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "type": stat.S_IFMT(info.st_mode),
        "mode": stat.S_IMODE(info.st_mode),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "links": info.st_nlink,
    }


def _differences(before: dict[str, int], after: dict[str, int]) -> dict:
    return {
        field: {"before": value, "after": after[field]}
        for field, value in before.items()
        if after[field] != value
    }


def _directory_observation(
    directory: int, path: Path, before: dict[str, int], link_delta: int
) -> dict:
    """Require stable qualification; only a separately measured link delta is allowed."""
    held = _object(os.fstat(directory))
    try:
        named = _object(path.stat(follow_symlinks=False))
    except OSError as exc:
        named = None
        pathname_error = errno.errorcode.get(exc.errno, str(exc.errno))
    else:
        pathname_error = None
    differences = _differences(before, held)
    failures = []
    for snapshot in (held, named):
        if snapshot is None:
            continue
        for field in ("device", "inode", "type", "mode", "uid", "gid"):
            if snapshot[field] != before[field]:
                failures.append("directory_" + field + "_changed")
        if (
            snapshot["type"] != stat.S_IFDIR
            or snapshot["mode"] != 0o700
            or snapshot["uid"] != os.geteuid()
        ):
            failures.append("directory_unqualified")
        if snapshot["links"] <= 0 or snapshot["links"] - before["links"] not in (0, link_delta):
            failures.append("directory_unexpected_links")
    same_directory = named is not None and all(
        held[field] == named[field] for field in ("device", "inode")
    )
    if not same_directory:
        failures.append("directory_path_binding_missing")
    return {
        "before": before,
        "directory_fd": held,
        "directory_path": named,
        "pathname_error": pathname_error,
        "same_directory": same_directory,
        "field_differences": differences,
        "permitted_metadata_changes": {
            field: change
            for field, change in differences.items()
            if field == "links" and not failures
        },
        "security_failures": failures,
        "qualified": not failures,
    }


def _lock_observation(descriptor: int, directory: int, path: Path) -> dict:
    held = os.fstat(descriptor)
    try:
        named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
    except OSError as exc:
        named = None
        pathname_error = errno.errorcode.get(exc.errno, str(exc.errno))
    else:
        pathname_error = None
    # Check the actual bytes as well as st_size, then recheck the held object.
    empty = stat.S_ISREG(held.st_mode) and os.read(descriptor, 1) == b""
    after = os.fstat(descriptor)
    failures = []
    for info in (held, after, named):
        if info is None:
            failures.append("lock_path_binding_missing")
        elif (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink != 1
            or stat.S_IMODE(info.st_mode) != 0o600
            or info.st_uid != os.geteuid()
            or info.st_size != 0
            or info.st_dev != os.fstat(directory).st_dev
        ):
            failures.append("lock_unqualified")
    if not empty:
        failures.append("lock_unexpected_contents")
    if _object(held) != _object(after) or named is None or _object(held) != _object(named):
        failures.append("lock_replaced_or_changed")
    return {
        "file_fd": _object(held),
        "file_fd_after_read": _object(after),
        "file_path": _object(named) if named is not None else None,
        "pathname_error": pathname_error,
        "empty_contents": empty,
        "security_failures": failures,
        "qualified": not failures,
    }


def _open(mode: str, directory: int, path: Path) -> int:
    flags = os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    if mode == "full_path_control":
        return os.open(path, flags | os.O_CREAT, 0o600)
    if mode == "existing_control":
        return os.open(path.name, flags, dir_fd=directory)
    if mode == "legacy":
        return os.open(path.name, flags | os.O_CREAT, 0o600, dir_fd=directory)
    try:
        return os.open(path.name, flags | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory)
    except FileExistsError:
        return os.open(path.name, flags, dir_fd=directory)


def _contend(
    mode: str,
    directory: int,
    path: Path,
    ready: Barrier,
    before: dict[str, int],
    link_delta: int,
) -> dict:
    ready.wait(timeout=30)
    try:
        descriptor = _open(mode, directory, path)
    except OSError as exc:
        return {
            "errno": errno.errorcode.get(exc.errno, str(exc.errno)),
            "directory": _directory_observation(directory, path.parent, before, link_delta),
        }
    try:
        observation = _lock_observation(descriptor, directory, path)
        observation["directory"] = _directory_observation(
            directory, path.parent, before, link_delta
        )
        return observation
    except OSError as exc:
        return {
            "errno": errno.errorcode.get(exc.errno, str(exc.errno)),
            "directory": _directory_observation(directory, path.parent, before, link_delta),
        }
    finally:
        os.close(descriptor)


def _metadata_control(root: Path) -> dict:
    """Independently measure entry creation/removal on this same temporary volume.

    Directory st_nlink describes filesystem directory metadata, rather than the
    single-link invariant imposed on a regular lock. Only the link transition
    actually measured here is permitted during the concurrent experiment.
    """
    path = root / "directory-metadata-control"
    path.mkdir(mode=0o700)
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    lock = path / ".pdsa-session.lock"
    try:
        before = _object(os.fstat(directory))
        initial = _directory_observation(directory, path, before, 0)
        descriptor = _open("exclusive_then_existing", directory, lock)
        try:
            lock_observation = _lock_observation(descriptor, directory, lock)
        finally:
            os.close(descriptor)
        after_create = _object(os.fstat(directory))
        link_delta = after_create["links"] - before["links"]
        created = _directory_observation(directory, path, before, link_delta)
        lock.unlink()
        after_remove = _object(os.fstat(directory))
        removed = _directory_observation(directory, path, before, link_delta)
        qualified = (
            created["qualified"]
            and initial["qualified"]
            and removed["qualified"]
            and lock_observation["qualified"]
            and after_remove == before
            and link_delta >= 0
        )
        return {
            "before": before,
            "directory_before": initial,
            "after_create": after_create,
            "after_remove": after_remove,
            "create_differences": _differences(before, after_create),
            "remove_differences": _differences(after_create, after_remove),
            "old_directory_comparison_would_fail": before != after_create,
            "directory_after_create": created,
            "directory_after_remove": removed,
            "lock": lock_observation,
            "permitted_entry_link_delta": link_delta if qualified else 0,
            "qualified": qualified,
        }
    finally:
        os.close(directory)
        lock.unlink(missing_ok=True)
        path.rmdir()


def probe(*, trials: int, workers: int) -> dict:
    """Exercise direct native opens; return counts and bounded identity samples."""
    if os.name != "posix" or not 1 <= trials <= 1000 or not 2 <= workers <= 64:
        raise ValueError("probe requires POSIX, 1..1000 trials and 2..64 workers")
    modes = ("legacy", "exclusive_then_existing", "existing_control", "full_path_control")
    results = {}
    with tempfile.TemporaryDirectory(prefix="enrollment-open-probe-") as temporary:
        metadata_control = _metadata_control(Path(temporary))
        link_delta = metadata_control["permitted_entry_link_delta"]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for mode in modes:
                errors: Counter[str] = Counter()
                successes = 0
                violations = 0
                same_directory_enoent = 0
                samples = []
                success_sample = None
                directory_samples = []
                directory_changes: Counter[str] = Counter()
                failure_reasons: Counter[str] = Counter()
                old_directory_comparison_failures = 0
                for trial in range(trials):
                    directory_path = Path(temporary) / f"{mode}-{trial}"
                    directory_path.mkdir(mode=0o700)
                    path = directory_path / ".pdsa-session.lock"
                    directory = os.open(
                        directory_path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
                    )
                    try:
                        before = _object(os.fstat(directory))
                        if mode == "existing_control":
                            descriptor = _open("exclusive_then_existing", directory, path)
                            os.close(descriptor)
                        ready = Barrier(workers)
                        futures = [
                            pool.submit(_contend, mode, directory, path, ready, before, link_delta)
                            for _ in range(workers)
                        ]
                        observations = [future.result(timeout=60) for future in futures]
                        identities = set()
                        for observation in observations:
                            directory_observation = observation["directory"]
                            violations += not directory_observation["qualified"]
                            failure_reasons.update(directory_observation["security_failures"])
                            if "errno" in observation:
                                errors[observation["errno"]] += 1
                                if (
                                    observation["errno"] == "ENOENT"
                                    and directory_observation["qualified"]
                                ):
                                    same_directory_enoent += 1
                                if len(samples) < 3:
                                    samples.append(observation)
                            else:
                                successes += 1
                                violations += not observation["qualified"]
                                failure_reasons.update(observation["security_failures"])
                                identity = observation["file_fd"]
                                identities.add((identity["device"], identity["inode"]))
                                if success_sample is None:
                                    success_sample = observation
                        violations += len(identities) > 1
                        if len(identities) > 1:
                            failure_reasons["multiple_lock_identities"] += 1
                        final_directory = _directory_observation(
                            directory, directory_path, before, link_delta
                        )
                        violations += not final_directory["qualified"]
                        failure_reasons.update(final_directory["security_failures"])
                        directory_changes.update(final_directory["field_differences"].keys())
                        old_directory_comparison_failures += (
                            before != final_directory["directory_fd"]
                        )
                        if len(directory_samples) < 3:
                            directory_samples.append(final_directory)
                        try:
                            descriptor = _open("existing_control", directory, path)
                        except OSError:
                            violations += 1
                            failure_reasons["final_lock_missing_or_unopenable"] += 1
                        else:
                            try:
                                final_lock = _lock_observation(descriptor, directory, path)
                                final_identity = final_lock["file_fd"]
                                final_matches = identities == {
                                    (final_identity["device"], final_identity["inode"])
                                }
                                violations += not final_lock["qualified"] or not final_matches
                                failure_reasons.update(final_lock["security_failures"])
                                if not final_matches:
                                    failure_reasons["final_lock_identity_changed"] += 1
                            finally:
                                os.close(descriptor)
                    finally:
                        os.close(directory)
                    path.unlink(missing_ok=True)
                    directory_path.rmdir()
                results[mode] = {
                    "attempts": workers * trials,
                    "successes": successes,
                    "errors": dict(errors),
                    "identity_or_metadata_violations": violations,
                    "security_failure_reasons": dict(failure_reasons),
                    "directory_field_changes": dict(directory_changes),
                    "old_directory_comparison_failures": old_directory_comparison_failures,
                    "directory_samples": directory_samples,
                    "enoent_with_same_live_directory": same_directory_enoent,
                    "error_samples": samples,
                    "success_sample": success_sample,
                }
    controls_pass = metadata_control["qualified"] and all(
        not result["errors"] and not result["identity_or_metadata_violations"]
        for mode, result in results.items()
        if mode != "legacy"
    )
    security_pass = metadata_control["qualified"] and all(
        not result["identity_or_metadata_violations"] for result in results.values()
    )
    legacy_errors_expected = not (set(results["legacy"]["errors"]) - {"ENOENT"})
    observed = results["legacy"]["enoent_with_same_live_directory"] > 0
    return {
        "schema": "CryptoHunter.EnrollmentOpenatProbe.v2",
        "system": platform.system(),
        "system_release": platform.release(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "trials_per_mode": trials,
        "workers": workers,
        "directory_metadata_control": metadata_control,
        "results": results,
        "controls_pass": controls_pass,
        "security_pass": security_pass,
        "legacy_errors_expected": legacy_errors_expected,
        "legacy_verdict": "REPRODUCED" if observed else "NOT_REPRODUCED",
        "production_state_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--workers", type=int, default=16)
    arguments = parser.parse_args()
    report = probe(trials=arguments.trials, workers=arguments.workers)
    print(json.dumps(report, indent=2, sort_keys=True))
    return (
        0
        if report["controls_pass"] and report["security_pass"] and report["legacy_errors_expected"]
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
