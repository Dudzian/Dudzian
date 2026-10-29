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
        "mode": stat.S_IMODE(info.st_mode),
        "links": info.st_nlink,
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


def _contend(mode: str, directory: int, path: Path, ready: Barrier) -> dict:
    ready.wait(timeout=30)
    try:
        descriptor = _open(mode, directory, path)
    except OSError as exc:
        # Pinning a descriptor does not by itself prevent unlinking a directory.
        # Capture both references at the actual error, before trial teardown.
        held = os.fstat(directory)
        named = path.parent.stat(follow_symlinks=False)
        return {
            "errno": errno.errorcode.get(exc.errno, str(exc.errno)),
            "directory_fd": _object(held),
            "directory_path": _object(named),
            "same_directory": (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino),
        }
    try:
        held = os.fstat(descriptor)
        named = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
        return {
            "file_fd": _object(held),
            "file_path": _object(named),
            "qualified": (
                stat.S_ISREG(held.st_mode)
                and held.st_nlink == 1
                and stat.S_IMODE(held.st_mode) == 0o600
                and held.st_size == 0
                and (held.st_dev, held.st_ino) == (named.st_dev, named.st_ino)
            ),
        }
    finally:
        os.close(descriptor)


def probe(*, trials: int, workers: int) -> dict:
    """Exercise direct native opens; return counts and bounded identity samples."""
    if os.name != "posix" or not 1 <= trials <= 1000 or not 2 <= workers <= 64:
        raise ValueError("probe requires POSIX, 1..1000 trials and 2..64 workers")
    modes = ("legacy", "exclusive_then_existing", "existing_control", "full_path_control")
    results = {}
    with tempfile.TemporaryDirectory(prefix="enrollment-open-probe-") as temporary:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for mode in modes:
                errors: Counter[str] = Counter()
                successes = 0
                violations = 0
                same_directory_enoent = 0
                samples = []
                success_sample = None
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
                            pool.submit(_contend, mode, directory, path, ready)
                            for _ in range(workers)
                        ]
                        observations = [future.result(timeout=60) for future in futures]
                        identities = set()
                        for observation in observations:
                            if "errno" in observation:
                                errors[observation["errno"]] += 1
                                if (
                                    observation["errno"] == "ENOENT"
                                    and observation["same_directory"]
                                ):
                                    same_directory_enoent += 1
                                if len(samples) < 3:
                                    samples.append(observation)
                            else:
                                successes += 1
                                violations += not observation["qualified"]
                                identity = observation["file_fd"]
                                identities.add((identity["device"], identity["inode"]))
                                if success_sample is None:
                                    success_sample = observation
                        violations += len(identities) > 1
                        violations += before != _object(os.fstat(directory))
                        violations += not path.exists()
                    finally:
                        os.close(directory)
                    path.unlink(missing_ok=True)
                    directory_path.rmdir()
                results[mode] = {
                    "attempts": workers * trials,
                    "successes": successes,
                    "errors": dict(errors),
                    "identity_or_metadata_violations": violations,
                    "enoent_with_same_live_directory": same_directory_enoent,
                    "error_samples": samples,
                    "success_sample": success_sample,
                }
    controls_pass = all(
        not result["errors"] and not result["identity_or_metadata_violations"]
        for mode, result in results.items()
        if mode != "legacy"
    )
    observed = results["legacy"]["enoent_with_same_live_directory"] > 0
    return {
        "schema": "CryptoHunter.EnrollmentOpenatProbe.v1",
        "system": platform.system(),
        "system_release": platform.release(),
        "macos_version": platform.mac_ver()[0],
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "trials_per_mode": trials,
        "workers": workers,
        "results": results,
        "controls_pass": controls_pass,
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
    return 0 if report["controls_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
