"""S9-RPI-28: observational-only pidfd lifetime checks, always production blocked."""

from __future__ import annotations

import json
import os
import select
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
        ([(51, select.POLLIN)], "PIDFD_NOT_LIVE"),
        ([(51, select.POLLHUP)], "PIDFD_NOT_LIVE"),
        ([(51, select.POLLERR)], "PIDFD_NOT_LIVE"),
        ([("51", 1)], "PIDFD_POLL_FAILED"),
        ([(52, 1)], "PIDFD_POLL_FAILED"),
        (None, "PIDFD_POLL_FAILED"),
    ],
)
def test_pidfd_poll_result_strictly_checked(monkeypatch, events, error):
    poller = _Poll(events)
    monkeypatch.setattr(probe, "select", SimpleNamespace(
        poll=lambda: poller, POLLIN=select.POLLIN,
        POLLERR=select.POLLERR, POLLHUP=select.POLLHUP
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
