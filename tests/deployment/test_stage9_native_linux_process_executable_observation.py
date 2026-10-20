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
from deployment import stage9_native_linux_process_identity_observation as identity_probe


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
    index = next(i for i, entry in enumerate(probe._PROPOSED_SYSTEMD_UNITS) if entry[1] == unit)
    _, _, account = probe._PROPOSED_SYSTEMD_UNITS[index]
    return SimpleNamespace(returncode=0, stdout=_systemd(unit, account, 4321 + index))


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
    active = {}
    opened = []
    closed = []
    reads = []
    hooks = {}

    def root_info(info):
        fields = {
            field: getattr(info, field)
            for field in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_size",
                "st_ctime_ns",
                "st_mtime_ns",
            )
        }
        return SimpleNamespace(**fields, st_uid=0, st_gid=0)

    def open_file(name, flags, *, dir_fd=None):
        if name == f"/proc/{pid}":
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_DIRECTORY
            name = str(proc)
        if name == "exe":
            assert dir_fd in active
            assert flags & os.O_NONBLOCK
            assert not (flags & os.O_NOFOLLOW)  # kernel magic link must be followed
            before_open = hooks.get("before_exe_open")
            if before_open:
                before_open()
        fd = original_open(name, flags, dir_fd=dir_fd)
        active[fd] = name
        opened.append((fd, name))
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
        if active.get(fd) != "exe":
            return original_read(fd, size)
        callback = hooks.get("read")
        if callback:
            callback(fd)
        data = original_read(fd, size)
        reads.append((size, data))
        return data

    def close_file(fd):
        original_close(fd)
        del active[fd]
        closed.append(fd)

    def assert_no_leaks():
        assert not active, f"Leaked descriptors: {active}"
        assert sorted(fd for fd, _name in opened) == sorted(closed)

    overrides = {
        "open": open_file,
        "stat": stat_file,
        "fstat": fstat_file,
        "read": read_file,
        "close": close_file,
    }
    tracked_os = SimpleNamespace(**(vars(os) | overrides))
    monkeypatch.setattr(probe, "os", tracked_os)
    monkeypatch.setattr(identity_probe, "os", tracked_os)
    fixture = SimpleNamespace(
        proc=proc,
        binary=binary,
        content=content,
        pid=pid,
        active=active,
        opened=opened,
        closed=closed,
        reads=reads,
        hooks=hooks,
        assert_no_leaks=assert_no_leaks,
    )
    yield fixture
    assert_no_leaks()


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
        {
            "pid": 4321,
            "start_time_ticks": 1,
            "sha256": "ab" * 32,
            "size": 1,
            "extra": "PRIVATE_EXECUTABLE_SECRET",
        },
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
@pytest.mark.parametrize(
    "error",
    [
        ValueError("PRIVATE_EXECUTABLE_SECRET"),
        RuntimeError("PRIVATE_EXECUTABLE_SECRET"),
        PermissionError("PRIVATE_EXECUTABLE_SECRET"),
    ],
)
def test_backend_failures_never_leak_private_content(backend, error):
    def broken(*_args, **_kwargs):
        raise error

    kwargs = {"runner": _runner, "executable_reader": _synthetic}
    kwargs[backend] = broken
    report = probe.inspect_host(**kwargs)
    _assert_blocked(report)
    assert len(report["observed_failures"]) == 3


def test_descriptor_history_preserves_reused_fd_numbers(procfs):
    first = probe._read_executable(procfs.pid)
    first_opened = list(procfs.opened)
    first_closed = list(procfs.closed)
    assert [name for _fd, name in first_opened] == [str(procfs.proc), "stat", "exe", "stat"]
    assert first_opened[1][0] == first_opened[2][0]
    assert not procfs.active

    assert probe._read_executable(procfs.pid) == first
    # Linux reuses the lowest available descriptors after the first read closes.
    assert procfs.opened == first_opened * 2
    assert procfs.closed == first_closed * 2
    assert not procfs.active
    procfs.assert_no_leaks()


def test_descriptor_leak_check_detects_an_open_descriptor(procfs):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = probe.os.open(f"/proc/{procfs.pid}", flags)
    try:
        assert stat.S_ISDIR(os.fstat(fd).st_mode)
        with pytest.raises(AssertionError, match="Leaked descriptors"):
            procfs.assert_no_leaks()
    finally:
        probe.os.close(fd)
    procfs.assert_no_leaks()


def test_descriptor_leak_check_ignores_unrelated_fd_reuse(procfs):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = probe.os.open(f"/proc/{procfs.pid}", flags)
    probe.os.close(fd)

    unrelated = os.open(procfs.proc, flags)
    try:
        assert unrelated == fd
        procfs.assert_no_leaks()
    finally:
        os.close(unrelated)


def test_regular_kernel_executable_fd_hashes_once(procfs):
    observed = probe._read_executable(procfs.pid)
    assert observed == {
        "pid": procfs.pid,
        "start_time_ticks": 98765,
        "sha256": hashlib.sha256(procfs.content).hexdigest(),
        "size": len(procfs.content),
    }
    assert [data for _, data in procfs.reads] == [procfs.content, b""]
    assert "exe" in [name for _fd, name in procfs.opened]


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


def test_executable_link_retarget_after_open_is_rejected(procfs):
    replacement = procfs.proc / "replacement-binary"
    replacement.write_bytes(procfs.content)
    replacement.chmod(0o644)
    changed = False

    def retarget(_fd):
        nonlocal changed
        if not changed:
            changed = True
            link = procfs.proc / "exe"
            link.unlink()
            link.symlink_to(replacement.name)

    procfs.hooks["read"] = retarget
    with pytest.raises(ValueError, match="EXECUTABLE_OBJECT_CHANGED"):
        probe._read_executable(procfs.pid)
    assert procfs.reads


def test_executable_metadata_change_after_read_refuses_digest(procfs):
    reads_complete = False

    def observed_read(_fd):
        nonlocal reads_complete
        reads_complete = True

    def changed_stat(name, info):
        if name == "exe" and reads_complete:
            info.st_ino += 1

    procfs.hooks["read"] = observed_read
    procfs.hooks["stat"] = changed_stat
    with pytest.raises(ValueError, match="EXECUTABLE_OBJECT_CHANGED"):
        probe._read_executable(procfs.pid)
    assert procfs.reads


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed systemd")
def test_invalid_systemd_response_never_reaches_procfs():
    def fake_runner(command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="PRIVATE_EXECUTABLE_SECRET\nMainPID=999\n",
        )

    def forbidden(_pid):
        pytest.fail("Malformed systemd response reached procfs")

    report = probe.inspect_host(runner=fake_runner, executable_reader=forbidden)
    _assert_blocked(report)
    assert all(
        row["issues"] == ["MALFORMED_SYSTEMD_RESPONSE"]
        for row in report["executable_observations"].values()
    )


def _runner_with_mainpid_hints(pids):
    def runner(command, **kwargs):
        response = _runner(command, **kwargs)
        index = next(
            i for i, entry in enumerate(probe._PROPOSED_SYSTEMD_UNITS) if entry[1] == command[-1]
        )
        _, unit, account = probe._PROPOSED_SYSTEMD_UNITS[index]
        response.stdout = _systemd(unit, account, pids[index])
        return response

    return runner


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic MainPID metadata collision")
@pytest.mark.parametrize("shared_indices", [(0, 1), (0, 2), (1, 2), (0, 1, 2)])
def test_shared_mainpid_hints_are_metadata_anomalies_even_with_different_starttimes(shared_indices):
    pids = [4321, 4322, 4323]
    for index in shared_indices:
        pids[index] = 7777
    reads = []

    def reader(pid):
        reads.append(pid)
        return {**_synthetic(pid), "start_time_ticks": 98765 + len(reads)}

    report = probe.inspect_host(runner=_runner_with_mainpid_hints(pids), executable_reader=reader)
    _assert_blocked(report)
    assert reads == pids
    roles = [probe._PROPOSED_SYSTEMD_UNITS[index][0] for index in shared_indices]
    assert report["observed_failures"] == [
        f"{left}/{right}: SHARED_SYSTEMD_MAINPID_HINT"
        for index, left in enumerate(roles)
        for right in roles[index + 1 :]
    ]
    assert all(
        row["observation"] == "UNPINNED_PROC_EXECUTABLE_DIGEST_ONLY"
        for row in report["executable_observations"].values()
    )
    assert "SYSTEMD_MAINPID_IS_UNTRUSTED_UNBOUND_HINT" in report["blockers"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic MainPID metadata collision")
@pytest.mark.parametrize("failed_index", [0, 2])
@pytest.mark.parametrize(
    "failure, issue",
    [
        ("refusal", "PROC_EXE_UNAVAILABLE"),
        ("permission", "PROCESS_OR_SYSTEMD_UNAVAILABLE"),
        ("invalid", "INVALID_PROCESS_EXECUTABLE_OBSERVATION"),
        ("unexpected", "BACKEND_OBSERVATION_FAILED"),
    ],
)
def test_shared_mainpid_hint_survives_earlier_or_later_executable_failure(
    failed_index, failure, issue
):
    pids = [7777, 4322, 7777]
    reads = []

    def reader(pid):
        reads.append(pid)
        if len(reads) - 1 == failed_index:
            if failure == "refusal":
                raise probe._Refusal("PROC_EXE_UNAVAILABLE")
            if failure == "permission":
                raise PermissionError("PRIVATE_EXECUTABLE_SECRET")
            if failure == "unexpected":
                raise RuntimeError("PRIVATE_EXECUTABLE_SECRET")
            return {"extra": "PRIVATE_EXECUTABLE_SECRET"}
        return _synthetic(pid)

    report = probe.inspect_host(runner=_runner_with_mainpid_hints(pids), executable_reader=reader)
    _assert_blocked(report)
    assert reads == pids
    roles = [entry[0] for entry in probe._PROPOSED_SYSTEMD_UNITS]
    assert report["observed_failures"] == [
        f"{roles[failed_index]}.executable: {issue}",
        f"{roles[0]}/{roles[2]}: SHARED_SYSTEMD_MAINPID_HINT",
    ]
    assert report["executable_observations"][roles[failed_index]]["issues"] == [issue]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic invalid MainPID metadata")
@pytest.mark.parametrize("invalid_index", [0, 2])
def test_malformed_systemd_metadata_does_not_supply_a_mainpid_hint(invalid_index):
    pids = [7777, 4322, 7777]
    runner = _runner_with_mainpid_hints(pids)
    reads = []

    def malformed_runner(command, **kwargs):
        response = runner(command, **kwargs)
        if command[-1] == probe._PROPOSED_SYSTEMD_UNITS[invalid_index][1]:
            response.stdout += "Unexpected=PRIVATE_EXECUTABLE_SECRET\n"
        return response

    def reader(pid):
        reads.append(pid)
        return _synthetic(pid)

    report = probe.inspect_host(runner=malformed_runner, executable_reader=reader)
    _assert_blocked(report)
    assert reads == [pid for index, pid in enumerate(pids) if index != invalid_index]
    role = probe._PROPOSED_SYSTEMD_UNITS[invalid_index][0]
    assert report["observed_failures"] == [f"{role}.executable: MALFORMED_SYSTEMD_RESPONSE"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic MainPID collision CLI")
def test_mainpid_metadata_collision_keeps_cli_blocked(monkeypatch, capsys):
    original = probe.inspect_host
    monkeypatch.setattr(
        probe,
        "inspect_host",
        lambda: original(
            runner=_runner_with_mainpid_hints([7777, 4322, 7777]),
            executable_reader=_synthetic,
        ),
    )
    assert probe.main() == 2
    report = json.loads(capsys.readouterr().out)
    _assert_blocked(report)
    left = probe._PROPOSED_SYSTEMD_UNITS[0][0]
    right = probe._PROPOSED_SYSTEMD_UNITS[2][0]
    assert report["observed_failures"] == [f"{left}/{right}: SHARED_SYSTEMD_MAINPID_HINT"]
