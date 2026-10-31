"""Native probe classification must distinguish directory metadata from lock policy."""

from __future__ import annotations

import errno
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from types import SimpleNamespace

import pytest

from scripts import diagnose_enrollment_lock_open as probe

pytestmark = [
    pytest.mark.security,
    pytest.mark.skipif(os.name != "posix", reason="native POSIX descriptor probe"),
]


@pytest.fixture
def directory(tmp_path):
    path = tmp_path / "session"
    path.mkdir(mode=0o700)
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        yield path, descriptor, probe._object(os.fstat(descriptor))
    finally:
        os.close(descriptor)


def _changed(info, **changes):
    fields = {
        name: getattr(info, name)
        for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink", "st_size")
    }
    return SimpleNamespace(**(fields | changes))


def test_metadata_control_records_fd_path_and_reversible_entry_transition(tmp_path, monkeypatch):
    native_object = probe._object

    def simulated_directory_entry_count(info):
        snapshot = native_object(info)
        if stat.S_ISDIR(info.st_mode):
            # Deterministically exercise a filesystem that counts this entry.
            snapshot["links"] += (
                tmp_path / "directory-metadata-control" / ".pdsa-session.lock"
            ).exists()
        return snapshot

    monkeypatch.setattr(probe, "_object", simulated_directory_entry_count)
    result = probe._metadata_control(tmp_path)
    assert result["qualified"]
    assert result["permitted_entry_link_delta"] == 1
    assert result["old_directory_comparison_would_fail"]
    assert result["create_differences"] == {
        "links": {"before": result["before"]["links"], "after": result["after_create"]["links"]}
    }
    assert result["after_remove"] == result["before"]
    for key in ("directory_before", "directory_after_create", "directory_after_remove"):
        assert result[key]["qualified"]
        assert result[key]["same_directory"]
        assert result[key]["directory_fd"] == result[key]["directory_path"]


def test_only_independently_measured_directory_link_change_is_permitted(directory, monkeypatch):
    path, descriptor, before = directory
    held = os.fstat(descriptor)
    monkeypatch.setattr(
        probe.os, "fstat", lambda _descriptor: _changed(held, st_nlink=held.st_nlink + 1)
    )
    result = probe._directory_observation(descriptor, path, before, 1)
    assert result["qualified"]
    assert result["same_directory"]
    assert result["permitted_metadata_changes"] == {
        "links": {"before": held.st_nlink, "after": held.st_nlink + 1}
    }
    refused = probe._directory_observation(descriptor, path, before, 0)
    assert not refused["qualified"]
    assert "directory_unexpected_links" in refused["security_failures"]


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("st_ino", -1, "directory_inode_changed"),
        ("st_dev", -1, "directory_device_changed"),
        ("st_mode", stat.S_IFDIR | 0o755, "directory_mode_changed"),
        ("st_mode", stat.S_IFREG | 0o700, "directory_type_changed"),
        ("st_uid", -1, "directory_uid_changed"),
        ("st_gid", -1, "directory_gid_changed"),
        ("st_nlink", 0, "directory_unexpected_links"),
    ],
)
def test_security_fields_are_never_permitted_metadata(directory, monkeypatch, field, value, reason):
    path, descriptor, before = directory
    held = os.fstat(descriptor)
    monkeypatch.setattr(probe.os, "fstat", lambda _descriptor: _changed(held, **{field: value}))
    result = probe._directory_observation(descriptor, path, before, 1)
    assert not result["qualified"]
    assert reason in result["security_failures"]
    assert result["permitted_metadata_changes"] == {}


def test_actual_directory_replacement_refuses_even_with_qualified_pinned_inode(directory):
    path, descriptor, before = directory
    path.rename(path.with_name("original"))
    path.mkdir(mode=0o700)
    result = probe._directory_observation(descriptor, path, before, 0)
    assert not result["qualified"]
    assert not result["same_directory"]
    assert "directory_inode_changed" in result["security_failures"]


def test_unlinked_pinned_directory_has_no_qualified_path_binding(directory):
    path, descriptor, before = directory
    path.rmdir()
    result = probe._directory_observation(descriptor, path, before, 0)
    assert not result["qualified"]
    assert not result["same_directory"]
    assert result["pathname_error"] == "ENOENT"
    assert "directory_path_binding_missing" in result["security_failures"]


def test_directory_symlink_rebinding_refuses(directory):
    path, descriptor, before = directory
    original = path.with_name("original")
    path.rename(original)
    path.symlink_to(original, target_is_directory=True)
    result = probe._directory_observation(descriptor, path, before, 0)
    assert not result["qualified"]
    assert "directory_type_changed" in result["security_failures"]


@pytest.mark.parametrize("mutation", ["replacement", "contents", "hardlink", "mode", "owner"])
def test_lock_policy_is_independent_of_permitted_directory_metadata(
    directory, monkeypatch, mutation
):
    path, directory_fd, _before = directory
    lock = path / ".pdsa-session.lock"
    descriptor = probe._open("exclusive_then_existing", directory_fd, lock)
    try:
        if mutation == "replacement":
            lock.unlink()
            lock.write_bytes(b"")
            lock.chmod(0o600)
        elif mutation == "contents":
            lock.write_bytes(b"unexpected")
        elif mutation == "hardlink":
            os.link(lock, path / "alias")
        elif mutation == "mode":
            lock.chmod(0o644)
        else:
            native_fstat = probe.os.fstat

            def wrong_owner(fd):
                info = native_fstat(fd)
                return _changed(info, st_uid=info.st_uid + 1) if fd == descriptor else info

            monkeypatch.setattr(probe.os, "fstat", wrong_owner)
        result = probe._lock_observation(descriptor, directory_fd, lock)
        assert not result["qualified"]
        assert result["security_failures"]
        if mutation == "replacement":
            assert "lock_replaced_or_changed" in result["security_failures"]
        elif mutation == "contents":
            assert "lock_unexpected_contents" in result["security_failures"]
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_nonregular_lock_refuses_without_reading_or_blocking(directory, kind):
    path, directory_fd, _before = directory
    lock = path / ".pdsa-session.lock"
    if kind == "directory":
        lock.mkdir(mode=0o700)
        descriptor = os.open(lock, os.O_RDONLY | os.O_DIRECTORY)
    else:
        os.mkfifo(lock, 0o600)
        descriptor = os.open(lock, os.O_RDWR | os.O_NONBLOCK)
    try:
        result = probe._lock_observation(descriptor, directory_fd, lock)
        assert not result["qualified"]
        assert "lock_unqualified" in result["security_failures"]
    finally:
        os.close(descriptor)


def test_lock_symlink_is_never_followed(directory):
    path, directory_fd, _before = directory
    source = path / "source"
    source.write_bytes(b"")
    lock = path / ".pdsa-session.lock"
    lock.symlink_to(source)
    with pytest.raises(OSError):
        probe._open("exclusive_then_existing", directory_fd, lock)


@pytest.mark.parametrize("create_error", [errno.ENOENT, errno.EACCES, errno.EEXIST])
def test_no_retry_or_recreation_after_unsafe_open_failure(monkeypatch, create_error):
    attempts = []

    def fail_open(path, flags, *args, **kwargs):
        attempts.append(flags)
        code = create_error if len(attempts) == 1 else errno.ENOENT
        raise OSError(code, "controlled native refusal")

    monkeypatch.setattr(probe.os, "open", fail_open)
    with pytest.raises(OSError):
        probe._open("exclusive_then_existing", 17, Path(".pdsa-session.lock"))
    assert len(attempts) == (2 if create_error == errno.EEXIST else 1)
    assert attempts[0] & os.O_EXCL and attempts[0] & os.O_CREAT
    assert all(flags & os.O_NOFOLLOW for flags in attempts)
    if len(attempts) == 2:
        assert not attempts[1] & (os.O_EXCL | os.O_CREAT)


def test_concurrent_first_creation_and_native_controls_share_one_qualified_lock():
    report = probe.probe(trials=3, workers=4)
    assert report["controls_pass"]
    assert report["security_pass"]
    assert report["directory_metadata_control"]["qualified"]
    legacy = report["results"]["legacy"]
    assert legacy["attempts"] == legacy["successes"] + sum(legacy["errors"].values()) == 12
    for mode, result in report["results"].items():
        if mode == "legacy":
            continue
        assert result["attempts"] == result["successes"] == 12
        assert result["errors"] == {}
        assert result["identity_or_metadata_violations"] == 0
        assert result["security_failure_reasons"] == {}
        assert result["success_sample"]["qualified"]
        assert result["success_sample"]["directory"]["qualified"]


def test_barrier_synchronized_existing_lock_access_keeps_native_checks(directory):
    path, directory_fd, before = directory
    lock = path / ".pdsa-session.lock"
    descriptor = probe._open("exclusive_then_existing", directory_fd, lock)
    os.close(descriptor)
    delta = os.fstat(directory_fd).st_nlink - before["links"]
    ready = Barrier(4)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(
                probe._contend, "existing_control", directory_fd, lock, ready, before, delta
            )
            for _ in range(4)
        ]
        observations = [future.result(timeout=10) for future in futures]
    assert all(result["qualified"] and result["directory"]["qualified"] for result in observations)
    assert (
        len({(result["file_fd"]["device"], result["file_fd"]["inode"]) for result in observations})
        == 1
    )


@pytest.mark.parametrize(
    ("unsafe", "native_error", "expected_exit"),
    [(False, "ENOENT", 0), (True, "ENOENT", 1), (False, "EACCES", 1)],
)
def test_legacy_errno_reproducer_never_exempts_security_failures(
    monkeypatch, capsys, unsafe, native_error, expected_exit
):
    native_contend = probe._contend
    injection_guard = Lock()
    injected = False

    def legacy_observation(mode, *args):
        nonlocal injected
        observation = native_contend(mode, *args)
        with injection_guard:
            if mode != "legacy" or injected:
                return observation
            injected = True
        directory = observation["directory"]
        if unsafe:
            directory = directory | {
                "qualified": False,
                "security_failures": ["directory_inode_changed"],
            }
        return {"errno": native_error, "directory": directory}

    monkeypatch.setattr(probe, "_contend", legacy_observation)
    report = probe.probe(trials=1, workers=4)
    assert report["controls_pass"]
    assert report["security_pass"] is (not unsafe)
    assert report["legacy_errors_expected"] is (native_error == "ENOENT")
    assert report["results"]["legacy"]["errors"] == {native_error: 1}
    assert bool(report["results"]["legacy"]["identity_or_metadata_violations"]) is unsafe
    monkeypatch.setattr(probe, "probe", lambda **kwargs: report)
    monkeypatch.setattr(probe.sys, "argv", ["diagnose_enrollment_lock_open.py"])
    assert probe.main() == expected_exit
    assert '"security_pass"' in capsys.readouterr().out
