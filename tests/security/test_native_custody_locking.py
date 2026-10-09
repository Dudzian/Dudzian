"""Native cross-process custody exclusion on POSIX and Windows."""

from __future__ import annotations

import errno
import multiprocessing
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from bot_core import local_signing_custody as custody
from tests.security._local_signing_platform import requires_native_custody_locking


def _lock_worker(path, exclusive, entered, release, outcome):
    try:
        with custody._custody_lock(Path(path), exclusive=exclusive):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("test release barrier timed out")
        outcome.put("released")
    except Exception as exc:
        outcome.put(type(exc).__name__)
        raise


@requires_native_custody_locking
@pytest.mark.parametrize("reader", [False, True])
@pytest.mark.parametrize("termination", ["normal", "abrupt"])
def test_native_two_process_exclusion_and_os_release(tmp_path, reader, termination):
    context = multiprocessing.get_context("spawn")
    first_entered, second_entered = context.Event(), context.Event()
    first_release, second_release = context.Event(), context.Event()
    outcomes = context.Queue()
    first = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, first_entered, first_release, outcomes)
    )
    second = context.Process(
        target=_lock_worker,
        args=(str(tmp_path), not reader, second_entered, second_release, outcomes),
    )
    try:
        first.start()
        assert first_entered.wait(20)
        second.start()
        assert not second_entered.wait(0.3)
        if termination == "normal":
            first_release.set()
            assert outcomes.get(timeout=10) == "released"
        else:
            first.terminate()
        first.join(10)
        assert not first.is_alive()
        assert second_entered.wait(20)
        second_release.set()
        assert outcomes.get(timeout=10) == "released"
        second.join(10)
        assert second.exitcode == 0
    finally:
        for process in (first, second):
            if process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(10)
        outcomes.close()


@requires_native_custody_locking
def test_exception_releases_native_custody_lock(tmp_path):
    with pytest.raises(RuntimeError, match="injected"):
        with custody._custody_lock(tmp_path, exclusive=True):
            raise RuntimeError("injected")
    # A different process must acquire it; a same-process probe is insufficient.
    context = multiprocessing.get_context("spawn")
    entered, release, outcome = context.Event(), context.Event(), context.Queue()
    release.set()
    process = context.Process(
        target=_lock_worker, args=(str(tmp_path), True, entered, release, outcome)
    )
    try:
        process.start()
        assert entered.wait(20)
        assert outcome.get(timeout=10) == "released"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
        process.join(10)
        outcome.close()


@pytest.mark.parametrize("exclusive", [False, True])
def test_windows_adapter_uses_one_byte_exclusive_ownership_and_explicit_unlock(
    tmp_path, monkeypatch, exclusive
):
    calls = []

    def locking(descriptor, mode, length):
        calls.append((mode, length, os.lseek(descriptor, 0, os.SEEK_CUR)))

    windows = SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    monkeypatch.setitem(sys.modules, "msvcrt", windows)
    monkeypatch.setattr(
        custody,
        "os",
        SimpleNamespace(
            **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}, **{}
        ),
    )
    monkeypatch.setattr(custody.os, "name", "nt")
    with pytest.raises(RuntimeError):
        with custody._custody_lock(tmp_path, exclusive=exclusive):
            raise RuntimeError("injected")
    assert calls == [(2, 1, 0), (0, 1, 0)]
    assert (tmp_path / custody._LOCK_FILENAME).read_bytes() == b"\0"


@pytest.mark.parametrize("error", [errno.EIO, errno.EACCES])
def test_windows_adapter_lock_error_never_yields_unlocked(tmp_path, monkeypatch, error):
    def locking(descriptor, mode, length):
        raise OSError(error, "injected lock failure")

    monkeypatch.setitem(
        sys.modules, "msvcrt", SimpleNamespace(LK_NBLCK=2, LK_UNLCK=0, locking=locking)
    )
    platform_os = SimpleNamespace(
        **{name: getattr(os, name) for name in dir(os) if not name.startswith("__")}
    )
    platform_os.name = "nt"
    monkeypatch.setattr(custody, "os", platform_os)
    clock = iter([0.0, 6.0])
    monkeypatch.setattr(custody, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(custody.LocalSigningCustodyError):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unlocked fallback")


@pytest.mark.skipif(os.name != "posix", reason="Unix mode bits are POSIX-specific")
def test_posix_custody_lock_rejects_world_readable_file(tmp_path):
    path = tmp_path / custody._LOCK_FILENAME
    path.write_bytes(b"\0")
    path.chmod(0o644)
    with pytest.raises(custody.LocalSigningCustodyError, match="permissions"):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unsafe permissions")


def test_unsupported_platform_has_no_unlocked_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(custody, "os", SimpleNamespace(name="unsupported"))
    with pytest.raises(custody.LocalSigningCustodyError, match="unavailable"):
        with custody._custody_lock(tmp_path, exclusive=True):
            pytest.fail("unsupported platform")
