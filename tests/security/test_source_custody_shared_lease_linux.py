"""S9-RPI-17: cooperative Linux source read lease held through TEST CCA-style use.

This integration harness uses the existing source worker's flock inode and
same-UID source/floor test fixtures. Advisory flock is cooperative; this
does NOT qualify installed source custody, credential eligibility, independent
authority, prior durability or production completion.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import fcntl
import multiprocessing
import os
from pathlib import Path
import stat
import sys

import pytest

from bot_core.test_only_retained_source_original import TestOnlyRetainedSourceError
from tests.security import _custody_source_demo as source_worker
from tests.security.test_retained_source_real_linux_disk_bridge import (
    _LinuxFixtureOriginalReader,
    _REQUEST,
    _inspect,
    _receive,
)
from tests.security.test_retained_source_test_floor_witness import (
    _FloorGuardedReader,
    witnessed_source as witnessed_source,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(sys.platform != "linux", reason="requires Linux flock/dirfd"),
]


class TestOnlySourceLeaseError(RuntimeError):
    """An exact cooperative source lease could not be held and rechecked."""


@dataclass(frozen=True, slots=True)
class SyntheticSourceReadLease:
    """Cannot authorize a production CCA result, regardless of locking outcome."""

    status: str = field(default="NOT_QUALIFIED", init=False)
    production_authorization: str = field(default="BLOCKED", init=False)
    source_authority_qualified: bool = field(default=False, init=False)


def _file_identity(info):
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _require(condition, message):
    if not condition:
        raise TestOnlySourceLeaseError(message)


def _check_lock(root_fd, lock_fd, config, expected):
    """File check precedes directory check so inode swaps cannot be masked."""
    current = os.fstat(lock_fd)
    path = os.stat("lock", dir_fd=root_fd, follow_symlinks=False)
    _require(
        _file_identity(current) == expected and _file_identity(path) == expected,
        "test source lock inode/metadata changed",
    )
    _require(
        stat.S_ISREG(current.st_mode)
        and stat.S_IMODE(current.st_mode) == 0o600
        and current.st_nlink == 1
        and current.st_uid == config["uid"]
        and current.st_gid == config["gid"]
        and (current.st_dev, current.st_ino) == tuple(config["lock_inode"]),
        "test source lock type/owner/mode/pin invalid",
    )
    root = Path(config["root"])
    observed = os.fstat(root_fd)
    root_path = root.lstat()
    pinned = tuple(config["directories"][str(root)])
    _require(
        (observed.st_dev, observed.st_ino) == pinned
        and (root_path.st_dev, root_path.st_ino) == pinned
        and stat.S_ISDIR(observed.st_mode)
        and stat.S_IMODE(observed.st_mode) == 0o700
        and observed.st_uid == config["uid"]
        and observed.st_gid == config["gid"],
        "test source root directory changed",
    )


@contextmanager
def hold_synthetic_source_read_lease(config):
    """Hold the worker's shared flock throughout original+floor+policy use.

    No new lock inode or fallback is created. Both check barriers occur while
    the lock is held. This blocks *cooperating* exclusive worker transitions
    only, not filesystem mutation by other processes sharing the same UID.
    """
    if sys.platform != "linux":
        raise TestOnlySourceLeaseError("Linux test-only source lease required")
    root_fd = lock_fd = None
    locked = False
    try:
        root_fd = os.open(
            config["root"], os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        lock_fd = os.open(
            "lock",
            os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            dir_fd=root_fd,
        )
        expected = _file_identity(os.fstat(lock_fd))
        _check_lock(root_fd, lock_fd, config, expected)
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        locked = True
        _check_lock(root_fd, lock_fd, config, expected)
        try:
            yield SyntheticSourceReadLease()
        finally:
            _check_lock(root_fd, lock_fd, config, expected)
    except OSError as exc:
        raise TestOnlySourceLeaseError("test source lease OS boundary unavailable") from exc
    finally:
        if lock_fd is not None:
            try:
                if locked:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        if root_fd is not None:
            os.close(root_fd)


def _reader(config, original, pin):
    disk = _LinuxFixtureOriginalReader(config, original)
    return disk, _FloorGuardedReader(disk, pin)


def test_shared_source_lock_held_through_all_floor_and_policy_checks(witnessed_source):
    config, original, pin = witnessed_source
    disk, guarded = _reader(config, original, pin)
    with hold_synthetic_source_read_lease(config) as lease:
        assert lease.status == "NOT_QUALIFIED"
        assert lease.production_authorization == "BLOCKED"
        assert lease.source_authority_qualified is False
        with _inspect(guarded, original) as evidence:
            assert evidence.status == "NOT_QUALIFIED"
            assert evidence.production_authorization == "BLOCKED"
            assert disk.lookups == 2
        assert disk.lookups == 3


def _spawn_worker_without_inheriting_read_lock(config):
    """Spawn, not fork: otherwise child inherits parent's locked fd."""
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    process = ctx.Process(
        target=source_worker._worker,
        args=(child, None, None, config, _REQUEST, None, None),
    )
    process.start()
    child.close()
    return process, parent


def test_cooperating_source_writer_waits_for_reader_then_recovers_exact_original(
    witnessed_source,
):
    config, original, pin = witnessed_source
    disk, guarded = _reader(config, original, pin)
    process = connection = None
    try:
        with hold_synthetic_source_read_lease(config):
            process, connection = _spawn_worker_without_inheriting_read_lock(config)
            assert not connection.poll(0.5), "exclusive source worker bypassed shared flock"
            assert process.is_alive(), "cooperating worker exited while shared lease held"
            with _inspect(guarded, original) as evidence:
                assert evidence.original_envelope == original
                assert evidence.durability_attested is False
            assert disk.lookups == 3
            assert not connection.poll(0.2)
        status, response = _receive(process, connection)
        process = connection = None
        assert status == "ok" and response["recovered"] is True
        assert response["event"] == original
        assert response["production_emission"] == "BLOCKED"
    finally:
        if connection is not None:
            connection.close()
        if process is not None and process.is_alive():
            process.kill()
            process.join(5)


@pytest.mark.parametrize("attack", ["lock-inode", "lock-symlink", "lock-permission"])
def test_lock_replacement_symlink_or_permissions_refuse_source_lease(witnessed_source, attack):
    config, _original, _pin = witnessed_source
    path = Path(config["root"]) / "lock"
    if attack == "lock-inode":
        path.rename(path.with_name("saved-lock"))
        path.write_bytes(b"")
        path.chmod(0o600)
    elif attack == "lock-symlink":
        path.rename(path.with_name("saved-lock"))
        path.symlink_to("saved-lock")
    else:
        path.chmod(0o644)
    with pytest.raises(TestOnlySourceLeaseError):
        with hold_synthetic_source_read_lease(config):
            pytest.fail("changed lock inode/ACL may not become a source lease")


def test_lock_inode_substitution_during_consumer_refuses_successful_exit(witnessed_source):
    config, original, pin = witnessed_source
    _disk, guarded = _reader(config, original, pin)
    path = Path(config["root"]) / "lock"
    with pytest.raises(TestOnlySourceLeaseError, match="inode/metadata changed"):
        with hold_synthetic_source_read_lease(config):
            with _inspect(guarded, original):
                path.rename(path.with_name("saved-lock"))
                path.write_bytes(b"")
                path.chmod(0o600)


def test_noncooperating_mutation_bypasses_flock_but_exact_reader_denies(witnessed_source):
    config, original, pin = witnessed_source
    _disk, guarded = _reader(config, original, pin)
    path = Path(config["root"]) / "events" / "original"
    with hold_synthetic_source_read_lease(config):
        # Same-UID arbitrary filesystem writes are NOT serialized by flock.
        path.write_bytes(original[:-1])
        with pytest.raises(TestOnlyRetainedSourceError, match="lookup denied"):
            with _inspect(guarded, original):
                pytest.fail("uncooperative source mutation cannot authorize CCA")
