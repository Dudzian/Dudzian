"""S9-RPI-28: observational-only pidfd lifetime checks, always production blocked."""

from __future__ import annotations

import json
import os
import select
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_pidfd_lifetime_observation as probe


def _reply(unit: str, account: str, pid: int) -> str:
    return (
        f"Id={unit}\n"
        "LoadState=loaded\n"
        "ActiveState=active\n"
        f"User={account}\n"
        f"Group={account}\n"
        f"MainPID={pid}\n"
    )


def _runner_for(pids):
    def run(cmd, **kwargs):
        assert cmd[:4] == ("/usr/bin/systemctl", "--system", "show", "--no-pager")
        assert kwargs["timeout"] == 3
        assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
        for i, (_role, unit, account) in enumerate(probe._PROPOSED_SYSTEMD_UNITS):
            if cmd[-1] == unit:
                return SimpleNamespace(returncode=0, stdout=_reply(unit, account, pids[i]))
        pytest.fail("Unknown systemd candidate")

    return run


def _reader(pid):
    return {"pid": pid, "start_time_ticks": 234567, "pidfd_poll_alive": True}


def _blocked(result):
    assert result["status"] == "BLOCKED"
    assert result["qualification"] == "NOT_QUALIFIED"
    assert result["production_emission"] == "BLOCKED"
    assert result["topology"] == "PROPOSED_NOT_APPROVED"
    assert "PRIVATE_EXECUTABLE_SECRET" not in json.dumps(result, allow_nan=False)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux host inventory")
def test_three_roles_observe_pidfd_without_promoting_production():
    result = probe.inspect_host(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=_reader
    )
    _blocked(result)
    assert result["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert not result["observed_failures"]
    assert len(result["pidfd_observations"]) == 3
    assert all(
        record["observation"] == "UNTRUSTED_PIDFD_LIFETIME_METADATA_ONLY"
        for record in result["pidfd_observations"].values()
    )
    assert "PIDFD_OPEN_IS_NOT_SOCKET_BOUND_SO_PEERPIDFD" in result["blockers"]


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_nonlinux_never_invokes_any_native_backend(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Backend invoked on non-Linux platform")

    result = probe.inspect_host(runner=forbidden, lifetime_reader=forbidden)
    _blocked(result)
    assert result["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic MainPID collision")
@pytest.mark.parametrize("pids", [[7777, 7777, 4323], [4321, 7777, 7777], [7777] * 3])
def test_shared_untrusted_mainpid_hint_detected(pids):
    result = probe.inspect_host(runner=_runner_for(pids), lifetime_reader=_reader)
    _blocked(result)
    roles = [entry[0] for entry in probe._PROPOSED_SYSTEMD_UNITS]
    expected = [
        f"{roles[i]}/{roles[j]}: SHARED_SYSTEMD_MAINPID_HINT"
        for i in range(3)
        for j in range(i + 1, 3)
        if pids[i] == pids[j]
    ]
    assert result["observed_failures"] == expected


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed native backend")
def test_mainpid_collision_is_retained_even_if_pidfd_open_fails():
    reads = []

    def reader(pid):
        reads.append(pid)
        if len(reads) == 1:
            raise probe._Refusal("PIDFD_OPEN_UNAVAILABLE")
        return _reader(pid)

    result = probe.inspect_host(
        runner=_runner_for([7777, 4322, 7777]), lifetime_reader=reader
    )
    _blocked(result)
    roles = [entry[0] for entry in probe._PROPOSED_SYSTEMD_UNITS]
    assert result["observed_failures"] == [
        f"{roles[0]}.pidfd: PIDFD_OPEN_UNAVAILABLE",
        f"{roles[0]}/{roles[2]}: SHARED_SYSTEMD_MAINPID_HINT",
    ]
    assert reads == [7777, 4322, 7777]


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malformed native backend")
@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        {"pid": 4321, "start_time_ticks": 1},
        {"pid": 4321, "start_time_ticks": 1, "pidfd_poll_alive": True, "secret": "X"},
        {"pid": True, "start_time_ticks": 1, "pidfd_poll_alive": True},
        {"pid": 4322, "start_time_ticks": 1, "pidfd_poll_alive": True},
        {"pid": 4321, "start_time_ticks": True, "pidfd_poll_alive": True},
        {"pid": 4321, "start_time_ticks": 0, "pidfd_poll_alive": True},
        {"pid": 4321, "start_time_ticks": 1, "pidfd_poll_alive": 1},
        {"pid": 4321, "start_time_ticks": 1, "pidfd_poll_alive": False},
        {"pid": 4321, "start_time_ticks": 1, "pidfd_poll_alive": "TRUE"},
    ],
)
def test_untrusted_backend_output_is_strictly_sanitized(value):
    result = probe.inspect_host(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=lambda _pid: value
    )
    _blocked(result)
    assert all(
        row["issues"] == ["INVALID_PIDFD_LIFETIME_OBSERVATION"]
        for row in result["pidfd_observations"].values()
    )


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"Pid: 123\nPid: 123\n",
        b"Name: Pid\n",
        b"Pid: -1\n",
        b"Pid: 0\n",
        b"Pid: 1\n",
        b"Pid: 123 456\n",
        b"Pid: 0123 \x00\n",
        b"Pid: 123\r\n",
        b"Pid: 123",
        b"Pid: abc\n",
        b"Pid: 123\n\n",
        b"Pid: 123\nbroken\n",
        b"Pid: 123\n" + b"x" * 4097,
    ],
)
def test_fdinfo_malformed_or_ambiguous_rejected(raw):
    with pytest.raises(ValueError, match="PIDFD_FDINFO_MALFORMED"):
        probe._parse_pidfd_fdinfo(raw, pid=123)


def test_fdinfo_different_valid_pid_refused():
    with pytest.raises(ValueError, match="PIDFD_PID_NOT_MATCHED"):
        probe._parse_pidfd_fdinfo(b"pos:\t0\nflags:\t02000002\nPid:\t124\nNSpid:\t124\n", pid=123)


def test_fdinfo_unique_exact_pid_accepted():
    probe._parse_pidfd_fdinfo(
        b"pos:\t0\nflags:\t02000002\nmnt_id:\t5\nino:\t0\nPid:\t123\nNSpid:\t123\n",
        pid=123,
    )


class _Poll:
    def __init__(self, events):
        self.events = events
        self.fd = None
        self.mask = None

    def register(self, fd, mask):
        self.fd = fd
        self.mask = mask

    def poll(self, timeout):
        assert timeout == 0
        return self.events


@pytest.mark.parametrize(
    "events,error",
    [
        ([], None),
        ([(51, getattr(select, "POLLIN", 1))], "PIDFD_NOT_LIVE"),
        ([(51, getattr(select, "POLLHUP", 16))], "PIDFD_NOT_LIVE"),
        ([(51, getattr(select, "POLLERR", 8))], "PIDFD_NOT_LIVE"),
        ([("51", 1)], "PIDFD_POLL_FAILED"),
        ([(52, 1)], "PIDFD_POLL_FAILED"),
        (None, "PIDFD_POLL_FAILED"),
    ],
)
def test_pidfd_poll_result_strictly_checked(monkeypatch, events, error):
    poller = _Poll(events)
    monkeypatch.setattr(probe, "select", SimpleNamespace(
        poll=lambda: poller, POLLIN=getattr(select, "POLLIN", 1),
        POLLERR=getattr(select, "POLLERR", 8), POLLHUP=getattr(select, "POLLHUP", 16)
    ))
    if error is not None:
        with pytest.raises(ValueError, match=error):
            probe._poll_liveness(51)
    else:
        probe._poll_liveness(51)
    assert poller.fd == 51


@pytest.mark.skipif(sys.platform != "linux", reason="Native Linux pidfd needed")
def test_pidfd_open_of_current_process_is_a_readonly_observation():
    if not callable(getattr(os, "pidfd_open", None)):
        pytest.skip("pidfd_open not supported on this Python/Linux host")
    try:
        observed = probe._read_pidfd_lifetime(os.getpid())
    except probe._Refusal as error:
        if error.code in {"PIDFD_OPEN_UNAVAILABLE", "PIDFD_NOT_SUPPORTED"}:
            pytest.skip("Native pidfd opening denied/unavailable on this host")
        raise
    assert observed["pid"] == os.getpid()
    assert observed["pidfd_poll_alive"] is True
    assert observed["start_time_ticks"] > 0


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic missing pidfd API")
def test_missing_pidfd_api_never_downgrades_to_pid_only(monkeypatch):
    monkeypatch.setattr(probe, "os", SimpleNamespace(pidfd_open=None))
    with pytest.raises(ValueError, match="PIDFD_NOT_SUPPORTED"):
        probe._read_pidfd_lifetime(4321)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic pidfd lifecycle")
def test_pidfd_and_proc_descriptors_are_closed_on_stat_failure(monkeypatch, tmp_path):
    """Even when procfs refuses mid-observation, pidfd and proc directory close."""
    real_open, real_close = os.open, os.close
    opened, closed = [], []
    proc = tmp_path / "proc"
    proc.mkdir()

    def tracked_open(path, flags, *, dir_fd=None):
        if path == "/proc/4321":
            path = proc
        fd = real_open(path, flags, dir_fd=dir_fd)
        opened.append(fd)
        return fd

    def pidfd_open(_pid, _flags):
        fd = real_open("/dev/null", os.O_RDONLY)
        opened.append(fd)
        return fd

    def tracked_close(fd):
        real_close(fd)
        closed.append(fd)

    mocked_os = SimpleNamespace(**(vars(os) | {
        "open": tracked_open,
        "close": tracked_close,
        "pidfd_open": pidfd_open,
    }))
    monkeypatch.setattr(probe, "os", mocked_os)
    monkeypatch.setattr(probe, "_poll_liveness", lambda _fd: None)
    monkeypatch.setattr(probe, "_read_pidfd_fdinfo", lambda _fd, *, pid: None)

    def fail(_fd, _name):
        raise probe._Refusal("PROCESS_CHANGED_DURING_OBSERVATION")

    monkeypatch.setattr(probe, "_read_proc_file", fail)
    with pytest.raises(ValueError, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        probe._read_pidfd_lifetime(4321)
    assert sorted(opened) == sorted(closed)


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux CLI")
def test_cli_remains_blocked_even_when_lifetime_observation_succeeds(monkeypatch, capsys):
    original = probe.inspect_host
    monkeypatch.setattr(probe, "inspect_host", lambda: original(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=_reader
    ))
    assert probe.main() == 2
    _blocked(json.loads(capsys.readouterr().out))



@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic malicious refusal code")
def test_malformed_backend_refusal_code_fails_closed_without_crash():
    def bad_reader(_pid):
        refusal = probe._Refusal("PIDFD_NOT_LIVE")
        refusal.code = ["PRIVATE_EXECUTABLE_SECRET"]
        raise refusal

    report = probe.inspect_host(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=bad_reader
    )
    _blocked(report)
    assert all(
        row["issues"] == ["BACKEND_OBSERVATION_FAILED"]
        for row in report["pidfd_observations"].values()
    )


@pytest.fixture
def simulated_pidfd_lifetime(monkeypatch, tmp_path):
    """Exercise real descriptor cleanup while injecting predictable kernel events."""
    if sys.platform != "linux":
        pytest.skip("Synthetic Linux pidfd lifetime")
    native_open, native_close = os.open, os.close
    opened, closed, events = [], [], []
    settings = {"pidfd_open_error": False, "proc_open_error": False, "close_error": False}

    def fake_pidfd_open(pid, flags):
        assert (pid, flags) == (4321, 0)
        if settings["pidfd_open_error"]:
            raise PermissionError("PRIVATE_EXECUTABLE_SECRET")
        fd = native_open("/dev/null", os.O_RDONLY)
        opened.append(fd)
        return fd

    def fake_open(path, flags, *, dir_fd=None):
        assert path == "/proc/4321"
        if settings["proc_open_error"]:
            raise PermissionError("PRIVATE_EXECUTABLE_SECRET")
        fd = native_open(tmp_path, flags, dir_fd=dir_fd)
        opened.append(fd)
        return fd

    def fake_close(fd):
        native_close(fd)
        closed.append(fd)
        if settings["close_error"]:
            settings["close_error"] = False
            raise OSError("PRIVATE_EXECUTABLE_SECRET")

    mock_os = SimpleNamespace(
        **(vars(os) | {
            "pidfd_open": fake_pidfd_open,
            "open": fake_open,
            "close": fake_close,
        })
    )
    monkeypatch.setattr(probe, "os", mock_os)
    monkeypatch.setattr(probe, "_read_proc_file", lambda fd, name: b"synthetic stat")
    monkeypatch.setattr(probe, "_parse_stat", lambda raw, pid: 76543)
    monkeypatch.setattr(
        probe, "_poll_liveness", lambda _fd: events.append("poll")
    )
    monkeypatch.setattr(
        probe, "_read_pidfd_fdinfo", lambda _fd, *, pid: events.append("fdinfo")
    )
    fixture = SimpleNamespace(
        settings=settings, opened=opened, closed=closed, events=events
    )
    yield fixture
    assert sorted(opened) == sorted(closed)
    for fd in closed:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_pidfd_lifetime_happy_path_checks_both_ends(simulated_pidfd_lifetime):
    fs = simulated_pidfd_lifetime
    observed = probe._read_pidfd_lifetime(4321)
    assert observed == {
        "pid": 4321, "start_time_ticks": 76543, "pidfd_poll_alive": True
    }
    assert fs.events == ["poll", "fdinfo", "fdinfo", "poll"]


def test_process_exit_between_first_and_second_poll_refuses(simulated_pidfd_lifetime, monkeypatch):
    fs = simulated_pidfd_lifetime
    polls = []

    class SequentialPoll:
        def register(self, fd, _mask):
            self.fd = fd

        def poll(self, timeout):
            assert timeout == 0
            polls.append(self.fd)
            if len(polls) == 1:
                return []
            return [(self.fd, select.POLLIN)]

    monkeypatch.setattr(probe, "select", SimpleNamespace(
        poll=SequentialPoll, POLLIN=select.POLLIN,
        POLLERR=select.POLLERR, POLLHUP=select.POLLHUP
    ))
    native_poll = probe._poll_liveness

    def observed_poll(pidfd):
        fs.events.append("poll")
        native_poll(pidfd)

    monkeypatch.setattr(probe, "_poll_liveness", observed_poll)
    with pytest.raises(probe._Refusal, match="^PIDFD_NOT_LIVE$"):
        probe._read_pidfd_lifetime(4321)
    assert len(polls) == 2
    assert fs.events == ["poll", "fdinfo", "fdinfo", "poll"]


def test_second_fdinfo_pid_change_is_rejected(simulated_pidfd_lifetime, monkeypatch):
    fs = simulated_pidfd_lifetime

    def fdinfo(_pidfd, *, pid):
        fs.events.append("fdinfo")
        value = 4321 if fs.events.count("fdinfo") == 1 else 9876
        probe._parse_pidfd_fdinfo(f"Pid:\t{value}\n".encode(), pid=pid)

    monkeypatch.setattr(probe, "_read_pidfd_fdinfo", fdinfo)
    with pytest.raises(probe._Refusal, match="^PIDFD_PID_NOT_MATCHED$"):
        probe._read_pidfd_lifetime(4321)
    assert fs.events == ["poll", "fdinfo", "fdinfo"]


def test_procfs_starttime_change_is_refused(simulated_pidfd_lifetime, monkeypatch):
    reads = []

    def stat(_raw, _pid):
        reads.append(len(reads))
        return 76543 + reads[-1]

    monkeypatch.setattr(probe, "_parse_stat", stat)
    with pytest.raises(probe._Refusal, match="^PROCESS_CHANGED_DURING_OBSERVATION$"):
        probe._read_pidfd_lifetime(4321)
    assert len(reads) == 2


@pytest.mark.parametrize(
    "operation,expected",
    [
        ("pidfd_open_error", "PIDFD_OPEN_UNAVAILABLE"),
        ("proc_open_error", "PIDFD_PROCFS_UNAVAILABLE"),
        ("close_error", "PIDFD_CLOSE_FAILED"),
    ],
)
def test_native_error_codes_are_specific_to_failure_stage(
    simulated_pidfd_lifetime, operation, expected
):
    fs = simulated_pidfd_lifetime
    fs.settings[operation] = True
    with pytest.raises(probe._Refusal, match=f"^{expected}$"):
        probe._read_pidfd_lifetime(4321)
    if operation == "pidfd_open_error":
        assert not fs.opened
    elif operation == "proc_open_error":
        assert len(fs.opened) == 1
    else:
        assert len(fs.opened) == 2


def test_proc_stat_read_error_is_not_pidfd_open_error(
    simulated_pidfd_lifetime, monkeypatch
):
    def unavailable(_fd, _name):
        raise OSError("PRIVATE_EXECUTABLE_SECRET")

    monkeypatch.setattr(probe, "_read_proc_file", unavailable)
    with pytest.raises(probe._Refusal, match="^PIDFD_PROCFS_UNAVAILABLE$"):
        probe._read_pidfd_lifetime(4321)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux fdinfo syscall fixture")
@pytest.mark.parametrize("stage", ["open", "read", "close"])
def test_fdinfo_io_errors_are_separate_from_pidfd_open(
    monkeypatch, tmp_path, stage
):
    original_open, original_read, original_close = os.open, os.read, os.close
    opened, closed = [], []
    fixture = tmp_path / "fdinfo"
    fixture.write_bytes(b"Pid:\t4321\n")

    def fdinfo_open(path, flags):
        assert path == "/proc/self/fdinfo/99"
        if stage == "open":
            raise PermissionError("PRIVATE_EXECUTABLE_SECRET")
        fd = original_open(fixture, flags)
        opened.append(fd)
        return fd

    def fdinfo_read(fd, length):
        if stage == "read":
            raise OSError("PRIVATE_EXECUTABLE_SECRET")
        return original_read(fd, length)

    def fdinfo_close(fd):
        original_close(fd)
        closed.append(fd)
        if stage == "close":
            raise OSError("PRIVATE_EXECUTABLE_SECRET")

    monkeypatch.setattr(
        probe,
        "os",
        SimpleNamespace(
            **(vars(os) | {
                "open": fdinfo_open,
                "read": fdinfo_read,
                "close": fdinfo_close,
            })
        ),
    )
    expected = "PIDFD_CLOSE_FAILED" if stage == "close" else "PIDFD_FDINFO_UNAVAILABLE"
    with pytest.raises(probe._Refusal, match=f"^{expected}$"):
        probe._read_pidfd_fdinfo(99, pid=4321)
    assert opened == closed


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux host inventory")
def test_error_classification_is_exported_as_fixed_refusal_codes():
    failures = {
        4321: "PIDFD_OPEN_UNAVAILABLE",
        4322: "PIDFD_FDINFO_UNAVAILABLE",
        4323: "PIDFD_PROCFS_UNAVAILABLE",
    }

    def failed_reader(pid):
        raise probe._Refusal(failures[pid])

    report = probe.inspect_host(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=failed_reader
    )
    _blocked(report)
    assert [
        item["issues"]
        for item in report["pidfd_observations"].values()
    ] == [[failure] for failure in failures.values()]
    assert len(report["observed_failures"]) == 3


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux host inventory")
@pytest.mark.parametrize("failure", [PermissionError, TimeoutError, subprocess.TimeoutExpired])
def test_systemctl_io_failure_is_not_labeled_as_pidfd_open(failure):
    def unavailable(*_args, **_kwargs):
        if failure is subprocess.TimeoutExpired:
            raise failure("/usr/bin/systemctl", 3)
        raise failure("PRIVATE_EXECUTABLE_SECRET")

    def forbidden(_pid):
        pytest.fail("systemctl failure must not invoke pidfd reader")

    result = probe.inspect_host(runner=unavailable, lifetime_reader=forbidden)
    _blocked(result)
    assert all(
        row["issues"] == ["SYSTEMD_UNAVAILABLE"]
        for row in result["pidfd_observations"].values()
    )


@pytest.mark.skipif(sys.platform != "linux", reason="Synthetic Linux host inventory")
def test_injected_unclassified_backend_oserror_is_fail_closed():
    def unavailable(_pid):
        raise OSError("PRIVATE_EXECUTABLE_SECRET")

    result = probe.inspect_host(
        runner=_runner_for([4321, 4322, 4323]), lifetime_reader=unavailable
    )
    _blocked(result)
    assert all(
        row["issues"] == ["BACKEND_OBSERVATION_FAILED"]
        for row in result["pidfd_observations"].values()
    )
