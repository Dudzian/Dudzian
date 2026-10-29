"""Delivery mechanics have no enrollment authority or hardware side effects."""

from __future__ import annotations

import ctypes
import errno
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from bot_core.licensing import enrollment_file_io as delivery

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX filesystem qualifications")


@pytest.mark.parametrize(
    ("platform", "symbol", "exclusive_flag"),
    [("linux", "renameat2", 0x1), ("darwin", "renameatx_np", 0x4)],
)
def test_native_exclusive_publish_uses_pinned_dirfd_and_no_replace(
    monkeypatch, platform: str, symbol: str, exclusive_flag: int
):
    """Both kernels publish through a native no-clobber operation."""

    class Rename:
        argtypes = None
        restype = None

        def __init__(self):
            self.arguments = None
            self.result = 0

        def __call__(self, *args):
            self.arguments = args
            return self.result

    native = Rename()
    monkeypatch.setattr(delivery.sys, "platform", platform)
    monkeypatch.setattr(
        delivery.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace(**{symbol: native})
    )
    delivery._rename_noreplace(17, "new.tmp", "final.json")
    assert native.arguments == (17, b"new.tmp", 17, b"final.json", exclusive_flag)
    assert native.argtypes == (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    assert native.restype is ctypes.c_int
    ctypes.set_errno(errno.EEXIST)
    native.result = -1
    with pytest.raises(FileExistsError):
        delivery._rename_noreplace(17, "new.tmp", "final.json")
    ctypes.set_errno(errno.EPERM)
    with pytest.raises(PermissionError):
        delivery._rename_noreplace(17, "new.tmp", "final.json")


def test_native_exclusive_publish_fails_closed_without_kernel_support(monkeypatch):
    monkeypatch.setattr(delivery.sys, "platform", "darwin")
    monkeypatch.setattr(delivery.ctypes, "CDLL", lambda *args, **kwargs: SimpleNamespace())
    with pytest.raises(delivery.EnrollmentFileError, match="native exclusive rename"):
        delivery._rename_noreplace(17, "new.tmp", "final.json")


def test_immutable_atomic_retry_and_permissions(tmp_path: Path):
    path = tmp_path / "artifact.json"
    delivery.write_artifact_file(path, b'{"artifact":"public"}')
    first = path.stat()
    delivery.write_artifact_file(path, b'{"artifact":"public"}')
    assert delivery.read_artifact_file(path) == b'{"artifact":"public"}'
    assert path.stat().st_ino == first.st_ino
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.stat().st_nlink == 1
    assert list(tmp_path.iterdir()) == [path]
    with pytest.raises(delivery.EnrollmentFileError, match="different bytes"):
        delivery.write_artifact_file(path, b"other")
    assert path.read_bytes() == b'{"artifact":"public"}'


def test_unsafe_existing_output_permissions_are_rejected_without_repair(tmp_path: Path):
    path = tmp_path / "artifact"
    path.write_bytes(b"public")
    path.chmod(0o644)
    assert delivery.read_artifact_file(path) == b"public"
    with pytest.raises(delivery.EnrollmentFileError, match="owner or permissions"):
        delivery.write_artifact_file(path, b"public")
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert path.read_bytes() == b"public"


@pytest.mark.skipif(
    getattr(os, "geteuid", lambda: -1)() != 0,
    reason="chown requires root in the isolated test directory",
)
def test_existing_output_must_belong_to_current_principal(tmp_path: Path):
    path = tmp_path / "artifact"
    path.write_bytes(b"public")
    path.chmod(0o600)
    os.chown(path, 65534, -1)
    assert delivery.read_artifact_file(path) == b"public"
    with pytest.raises(delivery.EnrollmentFileError, match="owner or permissions"):
        delivery.write_artifact_file(path, b"public")
    assert path.stat().st_uid == 65534


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "directory", "fifo"])
def test_rejects_nonregular_or_aliased_file(tmp_path: Path, kind: str):
    source = tmp_path / "source"
    source.write_bytes(b"public")
    path = tmp_path / "artifact"
    if kind == "symlink":
        path.symlink_to(source)
    elif kind == "hardlink":
        os.link(source, path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    with pytest.raises(delivery.EnrollmentFileError):
        delivery.read_artifact_file(path)
    with pytest.raises(delivery.EnrollmentFileError):
        delivery.write_artifact_file(path, b"replacement")
    assert source.read_bytes() == b"public"


def test_rejects_symlink_ancestor_and_path_traversal(tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "artifact").write_bytes(b"public")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    for path in (alias / "artifact", real / ".." / "real" / "artifact"):
        with pytest.raises(delivery.EnrollmentFileError):
            delivery.read_artifact_file(path)
        with pytest.raises(delivery.EnrollmentFileError):
            delivery.write_artifact_file(path, b"replacement")
    assert (real / "artifact").read_bytes() == b"public"


def test_size_limit_and_invalid_limits(tmp_path: Path):
    path = tmp_path / "artifact"
    path.write_bytes(b"x" * (delivery.MAXIMUM_ARTIFACT_BYTES + 1))
    with pytest.raises(delivery.EnrollmentFileError, match="size limit"):
        delivery.read_artifact_file(path)
    for maximum in (False, 0, -1, 1.5, delivery.MAXIMUM_ARTIFACT_BYTES + 1):
        with pytest.raises(delivery.EnrollmentFileError, match="limit"):
            delivery.read_artifact_file(path, maximum)
        with pytest.raises(delivery.EnrollmentFileError, match="limit"):
            delivery.write_artifact_file(tmp_path / "new", b"x", maximum)
    with pytest.raises(delivery.EnrollmentFileError):
        delivery.write_artifact_file(tmp_path / "new", b"xx", maximum=1)
    with pytest.raises(delivery.EnrollmentFileError):
        delivery.write_artifact_file(tmp_path / "new", bytearray(b"x"))
    assert not (tmp_path / "new").exists()


def test_genuine_missing_file_and_ancestor_remain_distinct_from_unsafe_files(tmp_path: Path):
    for path in (tmp_path / "missing", tmp_path / "absent" / "artifact"):
        with pytest.raises(FileNotFoundError):
            delivery.read_artifact_file(path)


def test_read_detects_modification_and_path_replacement(tmp_path: Path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"initial")
    native_read = os.read
    once = False

    def change_after_read(descriptor: int, size: int) -> bytes:
        nonlocal once
        raw = native_read(descriptor, size)
        if not once:
            once = True
            path.unlink()
            path.write_bytes(b"initial")
        return raw

    monkeypatch.setattr(os, "read", change_after_read)
    with pytest.raises(delivery.EnrollmentFileError, match="changed"):
        delivery.read_artifact_file(path)


def test_read_detects_size_growth(tmp_path: Path, monkeypatch):
    path = tmp_path / "artifact"
    path.write_bytes(b"initial")
    native_read = os.read
    once = False

    def grow_after_read(descriptor: int, size: int) -> bytes:
        nonlocal once
        raw = native_read(descriptor, size)
        if not once:
            once = True
            with path.open("ab") as stream:
                stream.write(b"x" * 20)
        return raw

    monkeypatch.setattr(os, "read", grow_after_read)
    with pytest.raises(delivery.EnrollmentFileError, match="size limit"):
        delivery.read_artifact_file(path, maximum=10)


def test_failed_write_leaves_no_partial_destination(tmp_path: Path, monkeypatch):
    native_write = os.write

    def interrupted(descriptor, raw):
        native_write(descriptor, raw[:1])
        raise OSError("disk unavailable")

    monkeypatch.setattr(os, "write", interrupted)
    with pytest.raises(delivery.EnrollmentFileError):
        delivery.write_artifact_file(tmp_path / "artifact", b"public")
    assert list(tmp_path.iterdir()) == []


def test_parallel_idempotent_writes_are_complete(tmp_path: Path):
    path = tmp_path / "artifact"
    raw = b"x" * 131073
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(delivery.write_artifact_file, path, raw) for _ in range(16)]
        for future in futures:
            future.result(timeout=5)
    assert delivery.read_artifact_file(path) == raw
    assert list(tmp_path.iterdir()) == [path]


def test_parallel_conflicting_writes_never_replace_winner(tmp_path: Path):
    path = tmp_path / "artifact"
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(delivery.write_artifact_file, path, raw) for raw in (b"one", b"two")
        ]
        successes = 0
        conflicts = 0
        for future in futures:
            try:
                future.result(timeout=5)
                successes += 1
            except delivery.EnrollmentFileError:
                conflicts += 1
    assert (successes, conflicts) == (1, 1)
    assert delivery.read_artifact_file(path) in (b"one", b"two")


def test_process_death_after_publication_does_not_wedge_retry(tmp_path: Path):
    """No finally block runs in the child: the destination must still be usable."""
    path = tmp_path / "artifact"
    native_rename = delivery._rename_noreplace

    def publish_and_exit():
        def die_after_rename(*args):
            native_rename(*args)
            os._exit(71)

        delivery._rename_noreplace = die_after_rename
        delivery.write_artifact_file(path, b"public")

    process = multiprocessing.get_context("fork").Process(target=publish_and_exit)
    process.start()
    process.join(timeout=5)
    if process.is_alive():
        process.kill()
        process.join()
    assert process.exitcode == 71
    assert path.stat().st_nlink == 1
    assert delivery.read_artifact_file(path) == b"public"
    delivery.write_artifact_file(path, b"public")
    assert list(tmp_path.iterdir()) == [path]


def test_create_directory_and_nested_session_io(tmp_path: Path):
    directory = tmp_path / "sessions" / "request"
    delivery.ensure_artifact_directory(directory)
    delivery.ensure_artifact_directory(directory)
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    with ThreadPoolExecutor(max_workers=1) as executor:

        def nested():
            with delivery.artifact_directory_lock(directory):
                with delivery.artifact_directory_lock(directory):
                    delivery.write_artifact_file(directory / "artifact", b"public")
                    return delivery.read_artifact_file(directory / "artifact")

        assert executor.submit(nested).result(timeout=5) == b"public"
    with pytest.raises(RuntimeError, match="caller failure"):
        with delivery.artifact_directory_lock(directory):
            raise RuntimeError("caller failure")


@pytest.mark.parametrize("replace", [False, True])
def test_session_directory_rename_cannot_redirect_nested_io(tmp_path: Path, replace: bool):
    session = tmp_path / "session"
    replacement = tmp_path / "replacement"
    delivery.ensure_artifact_directory(session)
    delivery.ensure_artifact_directory(replacement)
    delivery.write_artifact_file(replacement / "existing", b"unrelated")
    with delivery.artifact_directory_lock(session):
        session.rename(tmp_path / "original")
        if replace:
            replacement.rename(session)
        for operation in (
            lambda: delivery.read_artifact_file(session / "existing"),
            lambda: delivery.write_artifact_file(session / "new", b"public"),
            lambda: delivery.ensure_artifact_directory(session / "child"),
        ):
            with pytest.raises(delivery.EnrollmentFileError, match="active session"):
                operation()
        if replace:
            assert not (session / "new").exists()
            assert not (session / "child").exists()
        else:
            assert not session.exists()
    # Thread-local enforcement must end together with the lock context.
    delivery.ensure_artifact_directory(session)
    delivery.write_artifact_file(session / "new", b"public")


def test_session_lock_serializes_threads_and_survives_reopen(tmp_path: Path):
    entered = Event()
    release = Event()
    second = Event()

    def first():
        with delivery.artifact_directory_lock(tmp_path):
            entered.set()
            assert release.wait(timeout=5)

    def next_request():
        with delivery.artifact_directory_lock(tmp_path):
            second.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        one = executor.submit(first)
        assert entered.wait(timeout=5)
        two = executor.submit(next_request)
        assert not second.wait(timeout=0.05)
        release.set()
        one.result(timeout=5)
        two.result(timeout=5)
    assert second.is_set()
    with delivery.artifact_directory_lock(tmp_path):
        pass


def test_concurrent_first_session_lock_creation_has_one_immutable_inode(tmp_path, monkeypatch):
    contenders = Barrier(2)
    native_open = os.open
    attempts = []
    identities = []

    def synchronized_open(path, flags, mode=0o777, *, dir_fd=None):
        if path == ".pdsa-session.lock":
            attempts.append(flags)
            assert flags & os.O_NOFOLLOW
            if flags & os.O_CREAT:
                assert flags & os.O_EXCL
                contenders.wait(timeout=5)
        descriptor = native_open(path, flags, mode, dir_fd=dir_fd)
        if path == ".pdsa-session.lock":
            info = os.fstat(descriptor)
            identities.append((info.st_dev, info.st_ino))
        return descriptor

    monkeypatch.setattr(os, "open", synchronized_open)

    def enter_session():
        with delivery.artifact_directory_lock(tmp_path):
            delivery.write_artifact_file(tmp_path / "artifact", b"exact retry")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(enter_session) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert len(identities) == 2 and identities[0] == identities[1]
    assert sum(bool(flags & os.O_CREAT) for flags in attempts) == 2
    assert len(attempts) == 3  # two exclusive attempts, one existing-only open
    assert (tmp_path / "artifact").read_bytes() == b"exact retry"


@pytest.mark.parametrize("during_creation", [True, False])
def test_session_lock_enoent_is_not_retried_or_recreated(tmp_path, monkeypatch, during_creation):
    lock_path = tmp_path / ".pdsa-session.lock"
    if not during_creation:
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
    native_open = os.open
    attempts = []

    def disappeared(path, flags, mode=0o777, *, dir_fd=None):
        if path == ".pdsa-session.lock":
            attempts.append(flags)
            if during_creation or not flags & os.O_CREAT:
                if lock_path.exists():
                    lock_path.unlink()
                raise FileNotFoundError(errno.ENOENT, "missing lock", path)
        return native_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", disappeared)
    with pytest.raises(delivery.EnrollmentFileError, match="cannot be locked safely"):
        with delivery.artifact_directory_lock(tmp_path):
            pytest.fail("missing lock cannot establish a session")
    assert len(attempts) == (1 if during_creation else 2)
    assert not lock_path.exists()


def test_removed_pinned_directory_refuses_lock_creation(tmp_path, monkeypatch):
    session = tmp_path / "session"
    delivery.ensure_artifact_directory(session)
    native_open = os.open
    pinned_identity = None

    def remove_before_create(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal pinned_identity
        if path == ".pdsa-session.lock":
            info = os.fstat(dir_fd)
            pinned_identity = (info.st_dev, info.st_ino)
            session.rmdir()
            # An open descriptor pins the object, not its directory entry;
            # creating children in an unlinked directory must still fail.
            assert os.fstat(dir_fd).st_ino == pinned_identity[1]
        return native_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", remove_before_create)
    with pytest.raises(delivery.EnrollmentFileError, match="cannot be locked safely") as caught:
        with delivery.artifact_directory_lock(session):
            pytest.fail("a removed directory cannot establish a session")
    assert isinstance(caught.value.__cause__, FileNotFoundError)
    assert pinned_identity is not None
    assert not session.exists()


@pytest.mark.parametrize("change", ["replace", "permissions", "contents"])
def test_session_lock_changes_during_acquisition_are_rejected(tmp_path, monkeypatch, change):
    fcntl = delivery.importlib.import_module("fcntl")
    native_flock = fcntl.flock

    def replace_after_lock(descriptor, operation):
        native_flock(descriptor, operation)
        lock = tmp_path / ".pdsa-session.lock"
        if change == "replace":
            lock.rename(tmp_path / "original-lock")
            lock.write_bytes(b"")
            lock.chmod(0o600)
        elif change == "permissions":
            lock.chmod(0o644)
        else:
            lock.write_bytes(b"untrusted")

    monkeypatch.setattr(fcntl, "flock", replace_after_lock)
    with pytest.raises(delivery.EnrollmentFileError, match="lock changed during acquisition"):
        with delivery.artifact_directory_lock(tmp_path):
            pytest.fail("a substituted lock cannot establish a session")


def test_unprotected_directory_and_linked_lock_rejected(tmp_path: Path):
    tmp_path.chmod(0o777)
    for operation in (
        lambda: delivery.ensure_artifact_directory(tmp_path),
        lambda: delivery.write_artifact_file(tmp_path / "artifact", b"public"),
    ):
        with pytest.raises(delivery.EnrollmentFileError, match="untrusted modification"):
            operation()
    with pytest.raises(delivery.EnrollmentFileError):
        with delivery.artifact_directory_lock(tmp_path):
            pass
    tmp_path.chmod(0o700)
    source = tmp_path / "source"
    source.write_bytes(b"")
    source.chmod(0o600)
    os.link(source, tmp_path / ".pdsa-session.lock")
    with pytest.raises(delivery.EnrollmentFileError, match="exactly one link"):
        with delivery.artifact_directory_lock(tmp_path):
            pass


def test_windows_reads_native_qualified_handle_with_no_write_sharing(tmp_path, monkeypatch):
    """Native API simulation; physical NTFS remains a separate qualification."""
    from bot_core import windows_custody_filesystem as windows_fs

    calls = []
    handle = SimpleNamespace(Close=lambda: calls.append("closed"))
    info = (0, 1, 2, 3, 4, 0, 6, 1, 0, 9)

    @contextmanager
    def pinned(path):
        calls.append(path)
        yield (files, None, None), frozenset()

    def create(*arguments):
        assert arguments[2] == 0x1  # no delete/write sharing
        return handle

    files = SimpleNamespace(CreateFile=create, ReadFile=lambda _handle, _size: (0, b"public"))
    monkeypatch.setattr(windows_fs, "pinned_directory", pinned)
    monkeypatch.setattr(windows_fs, "_qualify", lambda *args, **kwargs: info)
    assert delivery._windows_read(tmp_path / "artifact", 6) == b"public"
    assert calls[-1] == "closed"
    with pytest.raises(delivery.EnrollmentFileError, match="size limit"):
        delivery._windows_read(tmp_path / "artifact", 5)
