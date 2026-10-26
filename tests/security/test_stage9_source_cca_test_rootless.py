"""Rootless scope evidence and strict procfs regressions; no host mutation."""

from __future__ import annotations

import os
import io
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_source_cca_test_runtime as runtime
from deployment.stage9_source_cca_test_package import HarnessRefusal, plan
from tests.security.test_stage9_source_cca_test_probes import (
    physical_preflight as physical_preflight,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="Linux-only procfs identity and cgroup observations"
)
_ROOTLESS_OBSERVER = runtime.rootless_process_observation


def _status(pid, *, uids=(0, 0, 0, 0), gids=(0, 0, 0, 0), groups=()):
    return (
        f"Name:\tinert\nState:\tS (sleeping)\nTgid:\t{pid}\nPid:\t{pid}\n"
        f"PPid:\t1\nUid:\t{' '.join(map(str, uids))}\n"
        f"Gid:\t{' '.join(map(str, gids))}\nGroups:\t{' '.join(map(str, groups))}\n"
    )


def _stat(pid, *, start=100):
    fields = ["S", *("0" for _ in range(49))]
    fields[19] = str(start)
    return f"{pid} (inert) {' '.join(fields)}\n"


@pytest.fixture
def procfs(monkeypatch, tmp_path):
    directory = tmp_path / "proc"
    directory.mkdir()
    original_iterdir = Path.iterdir
    original_readlink = os.readlink
    original_open = os.open
    denied_executables = set()
    executable_reads = []

    def iterdir(path):
        return original_iterdir(directory if path == Path("/proc") else path)

    def open_directory(path, *args, **kwargs):
        return original_open(
            directory if path == "/proc" or path == Path("/proc") else path, *args, **kwargs
        )

    def readlink(path, *args, **kwargs):
        candidate = Path(path)
        if candidate.parent.parent == directory and candidate.name == "exe":
            executable_reads.append(int(candidate.parent.name))
            if int(candidate.parent.name) in denied_executables:
                raise PermissionError("PRIVATE executable metadata")
        return original_readlink(path, *args, **kwargs)

    def process(
        pid,
        *,
        uids=(0, 0, 0, 0),
        gids=(0, 0, 0, 0),
        groups=(),
        cgroup="/init.scope",
        executable="/usr/bin/inert",
        deny_executable=False,
        status=None,
        start=100,
    ):
        item = directory / str(pid)
        item.mkdir()
        (item / "status").write_text(
            _status(pid, uids=uids, gids=gids, groups=groups) if status is None else status,
            encoding="ascii",
        )
        (item / "stat").write_text(_stat(pid, start=start), encoding="ascii")
        (item / "cgroup").write_text(f"0::{cgroup}\n", encoding="ascii")
        if executable is not None:
            (item / "exe").symlink_to(executable)
        if deny_executable:
            denied_executables.add(pid)
        return item

    monkeypatch.setattr(Path, "iterdir", iterdir)
    monkeypatch.setattr(runtime.os, "readlink", readlink)
    monkeypatch.setattr(runtime.os, "open", open_directory)
    monkeypatch.setattr(runtime.os, "getpid", lambda: 999)
    process(999, uids=(1000,) * 4, gids=(1000,) * 4, cgroup="/user.slice/session.scope")
    return SimpleNamespace(
        directory=directory,
        process=process,
        reads=executable_reads,
        denied=denied_executables,
    )


def _scoped(resources):
    return runtime.rootless_process_observation(resources)


def test_unrelated_root_process_permission_denied_executable_is_excluded(procfs):
    procfs.process(1, cgroup="/init.scope", deny_executable=True)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "SCOPED_ABSENCE_OBSERVED"
    assert report["matched_pids"] == [] and report["uncertain_pids"] == []
    assert report["scope"] == "VISIBLE_UID_GID_CGROUP"
    assert report["excluded_processes"] == 2
    assert report["full_confirmation"] is False
    assert report["executable_scan_complete"] is False
    assert 1 not in procfs.reads


def test_unrelated_root_process_does_not_block_rootless_preflight(
    procfs, physical_preflight, monkeypatch
):
    # The host fixture virtualizes host availability only; exercise the actual
    # new procfs observer through preflight rather than its fixture stand-in.
    monkeypatch.setattr(runtime, "rootless_process_observation", _ROOTLESS_OBSERVER)
    procfs.process(1, deny_executable=True)
    report = runtime.preflight(plan("012345abcdef"))
    assert report["ready"] is True
    assert report["read_only"] is True
    assert report["checks"]["no_test_processes"] is True
    assert report["blockers"] == []
    assert 1 not in procfs.reads


@pytest.mark.parametrize(
    "identity",
    [
        "real_uid",
        "effective_uid",
        "saved_uid",
        "filesystem_uid",
        "real_gid",
        "effective_gid",
        "saved_gid",
        "filesystem_gid",
        "supplementary_group",
    ],
)
def test_reserved_identity_is_detected_without_executable_access(procfs, identity):
    resources = plan("012345abcdef")
    number = resources["roles"][0]["uid"]
    uids = [0] * 4
    gids = [0] * 4
    groups = []
    if identity == "supplementary_group":
        groups = [number]
    else:
        index = ("real", "effective", "saved", "filesystem").index(identity.rsplit("_", 1)[0])
        (uids if identity.endswith("uid") else gids)[index] = number
    procfs.process(42, uids=uids, gids=gids, groups=groups, deny_executable=True)
    report = _scoped(resources)
    assert report["state"] == "RESIDUAL_PROCESS_FOUND"
    assert 42 in report["matched_pids"]
    assert report["full_confirmation"] is False
    assert 42 not in procfs.reads


@pytest.mark.parametrize("descendant", ["", "/child"])
def test_owned_cgroup_detects_old_process_even_after_uid_change(procfs, descendant):
    resources = plan("012345abcdef")
    procfs.process(
        42, cgroup=resources["roles"][0]["control_group"] + descendant, deny_executable=True
    )
    report = _scoped(resources)
    assert report["state"] == "RESIDUAL_PROCESS_FOUND"
    assert 42 in report["matched_pids"]


@pytest.mark.parametrize("filename", ["status", "stat", "cgroup"])
def test_unreadable_potential_test_metadata_is_uncertain(procfs, monkeypatch, filename):
    resources = plan("012345abcdef")
    item = procfs.process(42, uids=(resources["roles"][0]["uid"],) * 4, deny_executable=True)
    original = runtime._read_proc_file
    inode = item.stat().st_ino

    def read(directory, name):
        if os.fstat(directory).st_ino == inode and name == filename:
            raise PermissionError("PRIVATE process metadata")
        return original(directory, name)

    monkeypatch.setattr(runtime, "_read_proc_file", read)
    report = _scoped(resources)
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]
    assert report["complete"] is False and report["full_confirmation"] is False
    assert "PRIVATE" not in str(report)


@pytest.mark.parametrize(
    "replacement",
    [
        "Uid:\t0 0 0\n",
        "Uid:\t0 0 0 NaN\n",
        "Uid:\t0 0 0 0\nUid:\t0 0 0 0\n",
        "Gid:\t0 0 0\n",
        "Gid:\t0 0 0 bad\n",
        "Groups:\tbad\n",
        "Pid:\t999\n",
        "Pid:\tNaN\n",
        "Tgid:\t999\n",
    ],
)
def test_ambiguous_identity_never_proves_scope_absence(procfs, replacement):
    key = replacement.partition(":")[0]
    status = (
        "".join(line + "\n" for line in _status(42).splitlines() if not line.startswith(key + ":"))
        + replacement
    )
    procfs.process(42, status=status, deny_executable=True)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]
    assert report["complete"] is False


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "0::/init.scope\n0::/system.slice\n",
        "1:cpu:/init.scope\n",
        "0::relative\n",
        "0::/system.slice/../init.scope\n",
        "0::/init.scope\x00\n",
    ],
)
def test_ambiguous_cgroup_never_proves_scope_absence(procfs, raw):
    item = procfs.process(42, deny_executable=True)
    (item / "cgroup").write_text(raw, encoding="ascii")
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]


def test_pid_starttime_change_is_uncertain(procfs, monkeypatch):
    item = procfs.process(42, deny_executable=True)
    original = runtime._read_proc_file
    inode = item.stat().st_ino
    reads = 0

    def read(directory, name):
        nonlocal reads
        if os.fstat(directory).st_ino == inode and name == "stat":
            reads += 1
            return _stat(42, start=100 if reads == 1 else 101).encode("ascii")
        return original(directory, name)

    monkeypatch.setattr(runtime, "_read_proc_file", read)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]


def test_missing_metadata_of_still_present_pid_is_not_a_vanished_process(procfs):
    item = procfs.process(42, deny_executable=True)
    (item / "status").unlink()
    report = _scoped(plan("012345abcdef"))
    assert item.is_dir()
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]


def test_missing_self_pid_prevents_scope_absence(procfs, monkeypatch):
    monkeypatch.setattr(runtime.os, "getpid", lambda: 123456)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert report["complete"] is False


def test_strict_scan_still_requires_unrelated_root_executable_observation(procfs):
    procfs.process(1, deny_executable=True)
    with pytest.raises(HarnessRefusal, match="PROCESS_OBSERVATION_UNAVAILABLE"):
        runtime._processes(plan("012345abcdef"))
    assert 1 in procfs.reads


def test_strict_scan_still_detects_executable_only_identity(procfs):
    resources = plan("012345abcdef")
    procfs.process(42, executable=resources["root"] + "/root_source")
    assert runtime._processes(resources) == [42]


def test_administrative_confirmation_retains_strict_executable_fail_closed(procfs, monkeypatch):
    procfs.process(1, deny_executable=True)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    report = runtime.process_observation(plan("012345abcdef"), administrative=True)
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False
    assert report["complete"] is False
    assert 1 in procfs.reads


def test_administrative_confirmation_can_prove_full_scope_only_after_full_scan(procfs, monkeypatch):
    procfs.process(1)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    report = runtime.process_observation(plan("012345abcdef"), administrative=True)
    assert report["matched_pids"] == []
    assert report["complete"] is True
    assert report["full_confirmation"] is True
    assert report["executable_scan_complete"] is True
    assert 1 in procfs.reads


def test_rootless_cleanup_scope_is_explicitly_not_full_clean(procfs, monkeypatch):
    procfs.process(1, deny_executable=True)
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"resources_absent": True})
    report = runtime.inspect_cleanup(plan("012345abcdef"))
    assert report["scope_clear"] is True
    assert report["clean"] is False
    assert report["state"] == "SCOPED_ABSENCE_OBSERVED"
    assert "ADMINISTRATIVE_INSPECTION_REQUIRED" in report["blockers"]


def test_rootless_cleanup_detects_active_old_test_process(procfs, monkeypatch):
    resources = plan("012345abcdef")
    procfs.process(42, uids=(resources["roles"][0]["uid"],) * 4, deny_executable=True)
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"resources_absent": True})
    report = runtime.inspect_cleanup(resources)
    assert report["scope_clear"] is False
    assert report["clean"] is False
    assert report["state"] == "RESIDUALS_FOUND"


def test_rootless_cleanup_collision_cannot_be_overruled_by_scope_process_absence(
    procfs, monkeypatch
):
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"unit_unused": False})
    report = runtime.inspect_cleanup(plan("012345abcdef"))
    assert report["scope_clear"] is False
    assert report["clean"] is False
    assert "unit_unused" in report["blockers"]


@pytest.mark.parametrize(
    "raw,visible",
    [
        (b"1 0 0:1 / /proc rw - proc proc rw\n", True),
        (b"1 0 0:1 / /proc rw - proc proc rw,hidepid=0\n", True),
        (b"1 0 0:1 / /proc rw,hidepid=1 - proc proc rw\n", False),
        (b"1 0 0:1 / /proc rw - proc proc rw,hidepid=2\n", False),
        (b"1 0 0:1 / /proc rw - proc proc rw,hidepid=invisible\n", False),
        (b"1 0 0:1 / /proc rw - proc proc rw,hidepid=ptraceable\n", False),
        (b"1 0 0:1 / /proc rw - proc proc rw,hidepid=unknown\n", False),
        (b"1 0 0:1 /42 /proc rw - proc proc rw\n", False),
        (b"1 0 0:1 / /proc rw - tmpfs tmpfs rw\n", False),
        (b"1 0 0:1 / /proc rw - proc proc rw\n" * 2, False),
        (b"", False),
    ],
    ids=[
        "visible",
        "visible-explicit",
        "hidepid-1",
        "hidepid-2",
        "invisible",
        "ptraceable",
        "unknown-policy",
        "subtree",
        "wrong-fs",
        "ambiguous-mount",
        "unavailable-mount",
    ],
)
def test_procfs_visibility_requires_full_visible_proc_mount(monkeypatch, raw, visible):
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: io.BytesIO(raw))
    assert runtime._procfs_visibility() is visible


def test_hidden_pid_policy_is_explicit_uncertainty(procfs, monkeypatch):
    monkeypatch.setattr(runtime, "_procfs_visibility", lambda: False)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert "PROCFS_VISIBILITY_UNCERTAIN" in report["uncertainty"]
    assert report["complete"] is False


def test_oversized_mountinfo_cannot_establish_visibility(monkeypatch):
    monkeypatch.setattr(
        Path, "open", lambda *_args, **_kwargs: io.BytesIO(b"x" * (1024 * 1024 + 1))
    )
    assert runtime._procfs_visibility() is False


def test_rootless_privilege_does_not_implicitly_upgrade_scope(procfs, monkeypatch):
    procfs.process(1, deny_executable=True)
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    report = runtime.process_observation(plan("012345abcdef"))
    assert report["state"] == "SCOPED_ABSENCE_OBSERVED"
    assert report["full_confirmation"] is False
    assert not procfs.reads


def test_administrative_observation_never_downgrades_without_root(procfs, monkeypatch):
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 1000)
    report = runtime.process_observation(plan("012345abcdef"), administrative=True)
    assert report["state"] == "UNCERTAIN"
    assert report["full_confirmation"] is False
    assert report["complete"] is False
    assert report["uncertainty"] == ["ADMINISTRATIVE_PROCESS_OBSERVATION_REQUIRED"]


def test_executable_only_old_process_requires_admin_and_never_rootless_clean(procfs, monkeypatch):
    resources = plan("012345abcdef")
    procfs.process(42, executable=resources["root"] + "/root_source")
    monkeypatch.setattr(runtime, "_collisions", lambda _: {"resources_absent": True})
    limited = runtime.inspect_cleanup(resources)
    assert limited["scope_clear"] is True and limited["clean"] is False
    monkeypatch.setattr(runtime.os, "geteuid", lambda: 0)
    full = runtime.inspect_cleanup(resources, administrative=True)
    assert full["clean"] is False and full["scope_clear"] is False
    assert full["process_observation"]["matched_pids"] == [42]


@pytest.mark.parametrize("relevant", [False, True])
def test_stable_zombie_is_classified_by_identity_without_false_parser_refusal(procfs, relevant):
    resources = plan("012345abcdef")
    uid = resources["roles"][0]["uid"] if relevant else 0
    item = procfs.process(42, uids=(uid,) * 4, cgroup="/", deny_executable=True)
    (item / "stat").write_text(_stat(42).replace(") S ", ") Z "), encoding="ascii")
    report = _scoped(resources)
    assert report["complete"] is True
    assert report["state"] == ("RESIDUAL_PROCESS_FOUND" if relevant else "SCOPED_ABSENCE_OBSERVED")


def test_host_process_started_in_first_clock_tick_has_valid_stable_identity(procfs):
    procfs.process(1, start=0, cgroup="/", deny_executable=True)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "SCOPED_ABSENCE_OBSERVED"
    assert report["full_confirmation"] is False


@pytest.mark.parametrize("field", ["status", "cgroup"])
def test_changing_scope_metadata_is_uncertainty(procfs, monkeypatch, field):
    item = procfs.process(42, deny_executable=True)
    inode = item.stat().st_ino
    original = runtime._read_proc_file
    reads = 0

    def read(directory, name):
        nonlocal reads
        if os.fstat(directory).st_ino == inode and name == field:
            reads += 1
            if reads == 2:
                return (
                    _status(42, gids=(1000,) * 4).encode()
                    if field == "status"
                    else b"0::/changed.scope\n"
                )
        return original(directory, name)

    monkeypatch.setattr(runtime, "_read_proc_file", read)
    report = _scoped(plan("012345abcdef"))
    assert report["state"] == "UNCERTAIN"
    assert 42 in report["uncertain_pids"]


def test_unavailable_resource_metadata_is_uncertain_in_cleanup(procfs, monkeypatch):
    def unavailable(_):
        raise PermissionError("PRIVATE resource metadata")

    monkeypatch.setattr(runtime, "_collisions", unavailable)
    report = runtime.inspect_cleanup(plan("012345abcdef"))
    assert report["clean"] is False and report["scope_clear"] is False
    assert report["state"] == "UNCERTAIN"
    assert "PRIVATE" not in str(report)
