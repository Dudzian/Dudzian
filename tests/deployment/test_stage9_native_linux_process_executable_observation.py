"""S9-RPI-27: procfs exe fingerprints stay untrusted and production blocked."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_process_executable_observation as probe


def _systemd(unit, account, pid=4321):
    return (
        f"Id={unit}\n"
        "LoadState=loaded\n"
        "ActiveState=active\n"
        f"User={account}\n"
        f"Group={account}\n"
        f"MainPID={pid}\n"
    )


def _runner(command, **kwargs):
    assert command[:4] == ("/usr/bin/systemctl", "--system", "show", "--no-pager")
    assert kwargs["timeout"] == 3
    assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    unit = command[-1]
    index = next(
        i for i, entry in enumerate(probe._PROPOSED_SYSTEMD_UNITS) if entry[1] == unit
    )
    _, _, account = probe._PROPOSED_SYSTEMD_UNITS[index]
    return SimpleNamespace(
        returncode=0, stdout=_systemd(unit, account, 4321 + index)
    )


def _synthetic(pid):
    return {"pid": pid, "start_time_ticks": 98765, "sha256": "ab" * 32, "size": 400}


def _assert_blocked(result):
    assert result["status"] == result["production_emission"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert "PRIVATE_EXECUTABLE_SECRET" not in json.dumps(result, allow_nan=False)


def _proc_stat(pid, ticks):
    fields = ["S", *(str(i) for i in range(1, 19)), str(ticks)]
    return f"{pid} (synthetic worker) ".encode() + " ".join(fields).encode() + b"\n"


@pytest.fixture
def procfs(tmp_path, monkeypatch):
    """Use real descriptor-relative syscalls in an isolated fake procfs tree."""
    if sys.platform != "linux":
        pytest.skip("Linux procfs descriptor exercise")
    pid = 4321
    proc = tmp_path / "proc"
    proc.mkdir()
    binary = proc / "real-binary"
    content = b"ELF PRIVATE_EXECUTABLE_SECRET \x00\xf0\n"
    binary.write_bytes(content)
    (proc / "exe").symlink_to(binary.name)
    (proc / "stat").write_bytes(_proc_stat(pid, 98765))

    original_open = os.open
    original_stat = os.stat
    original_fstat = os.fstat
    original_close = os.close
    original_read = os.read
    opened = {}
    closed = []
    reads = []
    hooks = {}

    def root_info(info):
        fields = {
            field: getattr(info, field)
            for field in (
                "st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                "st_ctime_ns", "st_mtime_ns",
            )
        }
        return SimpleNamespace(**fields, st_uid=0, st_gid=0)

    def open_file(name, flags, *, dir_fd=None):
        if name == f"/proc/{pid}":
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_DIRECTORY
            name = str(proc)
        if name == "exe":
            assert dir_fd in opened
            assert flags & os.O_NONBLOCK
            assert not (flags & os.O_NOFOLLOW)  # kernel magic link must be followed
            before_open = hooks.get("before_exe_open")
            if before_open:
                before_open()
        fd = original_open(name, flags, dir_fd=dir_fd)
        opened[fd] = name
        return fd

    def stat_file(name, *, dir_fd=None):
        info = root_info(original_stat(name, dir_fd=dir_fd))
        callback = hooks.get("stat")
        if callback:
            callback(name, info)
        return info

    def fstat_file(fd):
        info = root_info(original_fstat(fd))
        callback = hooks.get("fstat")
        if callback:
            callback(fd, info)
        return info

    def read_file(fd, size):
        if opened.get(fd) != "exe":
            return original_read(fd, size)
        callback = hooks.get("read")
        if callback:
            callback(fd)
        data = original_read(fd, size)
        reads.append((size, data))
        return data

    def close_file(fd):
        original_close(fd)
        closed.append(fd)

    overrides = {
        "open": open_file, "stat": stat_file, "fstat": fstat_file,
        "read": read_file, "close": close_file,
    }
    monkeypatch.setattr(probe, "os", SimpleNamespace(**(vars(os) | overrides)))
    fixture = SimpleNamespace(
        proc=proc, binary=binary, content=content, pid=pid,
        opened=opened, closed=closed, reads=reads, hooks=hooks
    )
    yield fixture
    assert sorted(opened) == sorted(closed)
    for fd in closed:
        with pytest.raises(OSError):
            original_fstat(fd)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux observation")
def test_three_fixed_roles_produce_untrusted_digests_only():
    report = probe.inspect_host(runner=_runner, executable_reader=_synthetic)
    _assert_blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert not report["observed_failures"]
    assert len(report["executable_observations"]) == 3
    assert all(
        row["observation"] == "UNPINNED_PROC_EXECUTABLE_DIGEST_ONLY"
        for row in report["executable_observations"].values()
    )
    assert "PRIVATE_EXECUTABLE_SECRET" not in json.dumps(report)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_nonlinux_does_not_invoke_any_backend(monkeypatch, capsys, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden(*_args, **_kwargs):
        pytest.fail("A non-Linux backend was called")

    report = probe.inspect_host(runner=forbidden, executable_reader=forbidden)
    _assert_blocked(report)
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]
    assert probe.main() == 2
    _assert_blocked(json.loads(capsys.readouterr().out))


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed observations")
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"pid": 4321, "start_time_ticks": 1, "sha256": "ab" * 32},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "ab" * 32, "size": 1,
         "extra": "PRIVATE_EXECUTABLE_SECRET"},
        {"pid": True, "start_time_ticks": 1, "sha256": "ab" * 32, "size": 1},
        {"pid": 1, "start_time_ticks": 1, "sha256": "ab" * 32, "size": 1},
        {"pid": 4321, "start_time_ticks": 0, "sha256": "ab" * 32, "size": 1},
        {"pid": 4321, "start_time_ticks": True, "sha256": "ab" * 32, "size": 1},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "AB" * 32, "size": 1},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "a" * 63, "size": 1},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "ab" * 32, "size": True},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "ab" * 32, "size": 0},
        {"pid": 4321, "start_time_ticks": 1, "sha256": "ab" * 32, "size": 2**64},
    ],
)
def test_injected_observations_are_strictly_sanitized(value):
    report = probe.inspect_host(runner=_runner, executable_reader=lambda _pid: value)
    _assert_blocked(report)
    assert all(
        row["issues"] == ["INVALID_PROCESS_EXECUTABLE_OBSERVATION"]
        for row in report["executable_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic backend refusals")
@pytest.mark.parametrize("backend", ["runner", "executable_reader"])
@pytest.mark.parametrize("error", [
    ValueError("PRIVATE_EXECUTABLE_SECRET"),
    RuntimeError("PRIVATE_EXECUTABLE_SECRET"),
    PermissionError("PRIVATE_EXECUTABLE_SECRET"),
])
def test_backend_failures_never_leak_private_content(backend, error):
    def broken(*_args, **_kwargs):
        raise error

    kwargs = {"runner": _runner, "executable_reader": _synthetic}
    kwargs[backend] = broken
    report = probe.inspect_host(**kwargs)
    _assert_blocked(report)
    assert len(report["observed_failures"]) == 3


def test_regular_kernel_executable_fd_hashes_once(procfs):
    observed = probe._read_executable(procfs.pid)
    assert observed == {
        "pid": procfs.pid,
        "start_time_ticks": 98765,
        "sha256": hashlib.sha256(procfs.content).hexdigest(),
        "size": len(procfs.content),
    }
    assert [data for _, data in procfs.reads] == [procfs.content, b""]
    assert "exe" in procfs.opened.values()


def test_procfs_executable_retarget_during_open_refuses_before_read(procfs):
    other = procfs.proc / "other-binary"
    other.write_bytes(b"ELF PRIVATE_EXECUTABLE_SECRET changed\n")

    def swap():
        (procfs.proc / "exe").unlink()
        (procfs.proc / "exe").symlink_to(other.name)

    procfs.hooks["before_exe_open"] = swap
    with pytest.raises(ValueError, match="EXECUTABLE_OBJECT_CHANGED"):
        probe._read_executable(procfs.pid)
    assert not procfs.reads


def test_process_starttime_drift_refuses_executable_digest(procfs, monkeypatch):
    calls = 0

    def changed_stat(_fd, _name):
        nonlocal calls
        calls += 1
        return _proc_stat(procfs.pid, 98765 + (calls == 2))

    monkeypatch.setattr(probe, "_read_proc_file", changed_stat)
    with pytest.raises(ValueError, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        probe._read_executable(procfs.pid)
    assert calls == 2


@pytest.mark.parametrize("invalid", ["nonregular", "unsafe", "zero", "oversized", "deleted"])
def test_executable_unsafe_initial_metadata_refuses_without_read(procfs, invalid):
    def tamper(name, info):
        if name != "exe":
            return
        if invalid == "nonregular":
            info.st_mode = stat.S_IFDIR | 0o755
        elif invalid == "unsafe":
            info.st_uid = 1000
        elif invalid == "zero":
            info.st_size = 0
        elif invalid == "oversized":
            info.st_size = probe._MAX_EXE_BYTES + 1
        else:
            info.st_nlink = 0

    procfs.hooks["stat"] = tamper
    with pytest.raises(ValueError):
        probe._read_executable(procfs.pid)
    assert not procfs.reads


def test_executable_digest_is_bounded_with_one_extra_byte(procfs, monkeypatch):
    monkeypatch.setattr(probe, "_MAX_EXE_BYTES", 128)
    procfs.binary.write_bytes(b"x" * 128)
    assert probe._read_executable(procfs.pid)["size"] == 128
    assert procfs.reads[-1] == (1, b"")


def test_executable_growth_refuses_after_maximum_plus_one(procfs, monkeypatch):
    monkeypatch.setattr(probe, "_MAX_EXE_BYTES", 128)
    procfs.binary.write_bytes(b"x" * 128)
    grown = False

    def append(_fd):
        nonlocal grown
        if not grown:
            grown = True
            with procfs.binary.open("ab") as f:
                f.write(b"y" * 100)

    procfs.hooks["read"] = append
    with pytest.raises(ValueError, match="EXECUTABLE_TOO_LARGE_OR_EMPTY"):
        probe._read_executable(procfs.pid)
    assert sum(len(data) for _, data in procfs.reads) == 129


@pytest.mark.parametrize("phase", ["stat", "fstat", "read"])
def test_executable_io_error_releases_all_descriptors(procfs, phase):
    def bad(*_args):
        raise OSError("PRIVATE_EXECUTABLE_SECRET")
    procfs.hooks[phase] = bad
    with pytest.raises(ValueError, match="PROC_EXE_UNAVAILABLE"):
        probe._read_executable(procfs.pid)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux CLI observation")
def test_readonly_executable_observations_never_promote_cli(monkeypatch, capsys):
    original = probe.inspect_host
    monkeypatch.setattr(
        probe,
        "inspect_host",
        lambda: original(runner=_runner, executable_reader=_synthetic),
    )
    assert probe.main() == 2
    _assert_blocked(json.loads(capsys.readouterr().out))
