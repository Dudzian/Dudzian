"""Hosted crash/fence/path tests, using real guarded upstream and durable publisher."""

from __future__ import annotations

import multiprocessing
import os
import stat
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.persistence import physical_durability
from deployment import windows_production_cha_account_reservation as installed
from tests.licensing import test_cha_account_reservation as runtime

wide = runtime.wide
challenge_harness = runtime.challenge_harness
integration = runtime.integration
package = runtime.package
lppi = runtime.lppi
active = runtime.active
committed = runtime.committed
cha = runtime.cha
reserved = runtime.reserved


def test_crash_after_memory_mint_before_commit_publishes_nothing(cha, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(installed, "_write", Mock(side_effect=RuntimeError("TEST_ONLY-crash")))
        with pytest.raises(RuntimeError, match="TEST_ONLY-crash"):
            installed.establish_installed_account_initial_binding(cha)
    assert not installed._state_path().exists()
    value = installed.establish_installed_account_initial_binding(cha)
    assert value.account_id.startswith("acct_")


def test_commit_before_response_lost_response_retains_exact_candidate(cha, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(
            installed,
            "_issue_verified_initial_binding",
            Mock(side_effect=RuntimeError("TEST_ONLY-lost-response")),
        )
        with pytest.raises(RuntimeError, match="TEST_ONLY-lost-response"):
            installed.establish_installed_account_initial_binding(cha)
    state = installed._read(installed._state_path())
    monkeypatch.setattr(installed, "mint_uuid7", runtime.forbidden)
    assert (
        installed.establish_installed_account_initial_binding(cha).account_id == state["account_id"]
    )


@pytest.mark.skipif(os.name != "posix", reason="physical POSIX fence failures")
def test_every_physical_fence_and_ambiguous_replace_fails_without_publication(cha, monkeypatch):
    fsync_original, replace_original = os.fsync, os.replace
    path = installed._state_path()
    for cut in ("temporary", "rename_before", "rename_after", "published", "directory"):
        calls = []

        def fsync(descriptor, cut=cut, calls=calls):
            kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
            calls.append(kind)
            # Directory-creation fences precede the write; fail only publication.
            if (
                (cut == "temporary" and kind == "file" and calls.count("file") == 1)
                or (cut == "published" and kind == "file" and calls.count("file") == 2)
                or (cut == "directory" and kind == "directory" and "file" in calls)
            ):
                raise OSError("TEST_ONLY-fence")
            return fsync_original(descriptor)

        def replace(source, target, cut=cut):
            if cut == "rename_before":
                raise OSError("TEST_ONLY-fence")
            replace_original(source, target)
            if cut == "rename_after":
                raise OSError("TEST_ONLY-ambiguous-replace")

        with monkeypatch.context() as patch:
            patch.setattr(physical_durability.os, "fsync", fsync)
            patch.setattr(physical_durability.os, "replace", replace)
            with pytest.raises(installed.AccountReservationError, match="PERSISTENCE_FAILED"):
                installed.establish_installed_account_initial_binding(cha)
        retained = installed._read(path)
        if cut in ("temporary", "rename_before"):
            assert retained is None
        else:
            assert retained["status"] == installed.STATUS
        assert not list(path.parent.glob(f".{path.name}.*"))
        with monkeypatch.context() as patch:
            if retained is not None:
                patch.setattr(installed, "mint_uuid7", runtime.forbidden)
            value = installed.establish_installed_account_initial_binding(cha)
            if retained is not None:
                assert value.account_id == retained["account_id"]
        path.unlink()


def test_loader_failed_fence_never_issues_then_repairs_exact_winner(reserved, monkeypatch):
    upstream, value, path, state = reserved
    identity = value.account_id
    with monkeypatch.context() as patch:
        patch.setattr(
            installed, "atomic_write_bytes_durably", Mock(side_effect=OSError("TEST_ONLY"))
        )
        with pytest.raises(installed.AccountReservationError, match="PERSISTENCE_FAILED"):
            installed.load_installed_account_initial_binding(upstream)
    assert path.read_bytes() == canonical_json_bytes(state)
    monkeypatch.setattr(installed, "mint_uuid7", runtime.forbidden)
    assert installed.load_installed_account_initial_binding(upstream).account_id == identity


def test_unsafe_oversized_noncanonical_and_unknown_records(reserved, monkeypatch, tmp_path):
    upstream, value, path, state = reserved
    original = path.read_bytes()
    for invalid in (
        b"{}",
        b" {}",
        original + b"\n",
        b"x" * (installed._MAX_BYTES + 1),
        canonical_json_bytes({**state, "unknown": True}),
    ):
        path.write_bytes(invalid)
        with pytest.raises(installed.AccountReservationError):
            installed.load_installed_account_initial_binding(upstream)
    path.write_bytes(original)
    for target in (path, path.with_suffix(".lock")):
        retained = target.read_bytes()
        target.unlink()
        alias = tmp_path / "TEST_ONLY-alias"
        alias.write_bytes(retained)
        if os.name == "posix":
            target.symlink_to(alias)
            with pytest.raises(installed.AccountReservationError):
                installed.load_installed_account_initial_binding(upstream)
            target.unlink()
        os.link(alias, target)
        with pytest.raises(installed.AccountReservationError):
            installed.load_installed_account_initial_binding(upstream)
        target.unlink()
        alias.unlink()
        target.write_bytes(retained)
    assert value.account_id == state["account_id"]
    information = SimpleNamespace(st_file_attributes=0x400, st_mode=stat.S_IFREG, st_nlink=1)
    with monkeypatch.context() as patch:
        patch.setattr(type(path), "lstat", Mock(return_value=information))
        with pytest.raises(installed.AccountReservationError):
            installed._safe(path)


@pytest.mark.skipif(os.name != "posix", reason="forked installed fixture and actual flock")
def test_process_race_one_retained_candidate_or_bounded_busy(cha):
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    qualified = context.Event()
    reader, writer = context.Pipe(duplex=False)

    def contender():
        ready.wait(10)
        raw = installed._source(cha)
        qualified.set()
        try:
            with installed._locked_state() as path:
                state = installed._reserve(path, raw)
            writer.send(("winner", state["account_id"]))
        except installed.AccountReservationError as exc:
            writer.send(("error", str(exc)))

    process = context.Process(target=contender)
    raw = installed._source(cha)
    with installed._locked_state() as path:
        process.start()
        ready.set()
        winner = installed._reserve(path, raw)
        assert qualified.wait(120), "upstream qualification timeout"
        assert reader.poll(5), "contender blocked at nonblocking lock"
        result = reader.recv()
        assert result == ("error", "CHA_ACCOUNT_RESERVATION_BUSY")
    process.join(10)
    assert process.exitcode == 0
    assert installed.load_installed_account_initial_binding(cha).account_id == winner["account_id"]


@pytest.mark.parametrize("busy", [True, False])
def test_hosted_windows_lock_uses_nonblocking_one_byte_lock_and_unlock(monkeypatch, tmp_path, busy):
    locking = Mock(side_effect=OSError("held") if busy else None)
    abi = SimpleNamespace(locking=locking, LK_NBLCK=2, LK_UNLCK=0)
    monkeypatch.setitem(sys.modules, "msvcrt", abi)
    monkeypatch.setattr(installed, "os", SimpleNamespace(**(vars(os) | {"name": "nt"})))
    monkeypatch.setattr(
        installed, "_state_path", lambda: tmp_path / "initial-cha-account-reservation.json"
    )
    if busy:
        with pytest.raises(installed.AccountReservationError, match="BUSY"):
            with installed._locked_state():
                pytest.fail("busy Windows lock entered its body")
        assert locking.call_count == 1
    else:
        with pytest.raises(RuntimeError, match="TEST_ONLY-body-failure"):
            with installed._locked_state():
                assert locking.call_count == 1
                raise RuntimeError("TEST_ONLY-body-failure")
        assert locking.call_count == 2
    first = locking.call_args_list[0].args
    assert first[1:] == (abi.LK_NBLCK, 1)
    assert (tmp_path / "initial-cha-account-reservation.lock").read_bytes() == b"\0"
    if not busy:
        assert locking.call_args_list[1].args == (first[0], abi.LK_UNLCK, 1)


@pytest.mark.parametrize(
    "kind", ["directory", "fifo", "symlink_parent", "oversized_lock", "relative"]
)
def test_path_boundaries_reject_before_state_or_lock_authority(monkeypatch, tmp_path, kind):
    from pathlib import Path

    path = tmp_path / "state" / "initial-cha-account-reservation.json"
    path.parent.mkdir()
    if kind == "relative":
        path = Path("TEST_ONLY-relative.json")
    elif kind == "symlink_parent":
        if os.name != "posix":
            pytest.skip("real POSIX symlink")
        alias = tmp_path / "TEST_ONLY-alias"
        alias.symlink_to(path.parent, target_is_directory=True)
        path = alias / path.name
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        if os.name != "posix":
            pytest.skip("real POSIX FIFO")
        os.mkfifo(path)
    elif kind == "oversized_lock":
        path.with_suffix(".lock").write_bytes(b"TEST_ONLY-oversized")
    monkeypatch.setattr(installed, "_state_path", lambda: path)
    with pytest.raises(installed.AccountReservationError):
        with installed._locked_state() as retained:
            installed._read(retained)
    assert not path.is_file()


def test_directory_creation_fence_failure_never_acquires_lock(monkeypatch, tmp_path):
    path = tmp_path / "TEST_ONLY-new-parent" / "child" / "initial-cha-account-reservation.json"
    monkeypatch.setattr(installed, "_state_path", lambda: path)
    original = installed.flush_created_directory_metadata
    with monkeypatch.context() as patch:
        patch.setattr(
            installed, "flush_created_directory_metadata", Mock(side_effect=OSError("TEST_ONLY"))
        )
        with pytest.raises(installed.AccountReservationError, match="PERSISTENCE_FAILED"):
            with installed._locked_state():
                pytest.fail("failed directory fence acquired authority")
    assert not path.with_suffix(".lock").exists()
    fenced = []

    def flush(parent):
        fenced.append(parent)
        return original(parent)

    monkeypatch.setattr(installed, "flush_created_directory_metadata", flush)
    with installed._locked_state():
        assert tmp_path in fenced and path.parent.parent in fenced
