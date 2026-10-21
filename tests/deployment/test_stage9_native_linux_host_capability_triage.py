"""S9-RPI-33: host capability hints cannot qualify a production source/CCA."""

from __future__ import annotations

import json
import os
import stat
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_native_linux_host_capability_triage as triage


def _probes(**overrides):
    probes = {name: (lambda: True) for name in triage._NAMES}
    for name, result in overrides.items():
        probes[name] = lambda result=result: result
    return probes


def _assert_blocked(report):
    assert report["schema"] == "CryptoHunter.Stage9NativeLinuxHostCapabilityTriageV1"
    assert report["status"] == report["production_emission"] == "BLOCKED"
    assert report["qualification"] == "NOT_QUALIFIED"
    assert report["topology"] == "PROPOSED_NOT_APPROVED"
    assert report["blockers"]
    assert set(report) == {
        "schema",
        "status",
        "qualification",
        "production_emission",
        "topology",
        "snapshot",
        "blockers",
        "capabilities",
        "observed_failures",
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
def test_matching_synthetic_probes_are_untrusted_and_invoke_each_once():
    calls = []
    probes = {}

    for name in triage._NAMES:

        def observer(name=name):
            calls.append(name)
            return True

        probes[name] = observer

    report = triage.inspect_host(probes=probes)
    _assert_blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["observed_failures"] == []
    assert calls == list(triage._NAMES)
    assert report["capabilities"] == {
        name: {
            "observation": "OBSERVED_UNTRUSTED",
            "reason": "UNTRUSTED_CAPABILITY_HINT",
        }
        for name in triage._NAMES
    }


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
@pytest.mark.parametrize("name", triage._NAMES)
def test_false_probe_is_scoped_without_dropping_other_observations(name):
    report = triage.inspect_host(probes=_probes(**{name: False}))
    _assert_blocked(report)
    assert report["observed_failures"] == [f"{name}: PROBE_UNAVAILABLE"]
    assert report["capabilities"][name] == {
        "observation": "NOT_OBSERVED",
        "reason": "PROBE_UNAVAILABLE",
    }
    for other in set(triage._NAMES) - {name}:
        assert report["capabilities"][other]["observation"] == "OBSERVED_UNTRUSTED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
@pytest.mark.parametrize("invalid", [None, 1, "yes", {}, []])
def test_malformed_probe_value_is_not_accepted_as_positive(invalid):
    report = triage.inspect_host(probes=_probes(procfs=invalid))
    _assert_blocked(report)
    assert report["observed_failures"] == ["procfs: PROBE_FAILED_OR_MALFORMED"]
    assert report["capabilities"]["procfs"]["observation"] == "NOT_OBSERVED"


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
def test_backend_exceptions_are_independently_sanitized():
    order = []

    def failed():
        order.append("procfs")
        raise RuntimeError("PRIVATE_HOST_ENVIRONMENT")

    def available():
        order.append("systemd_manager")
        return True

    probes = _probes(namespaces=False, pidfd_self=False)
    probes["procfs"] = failed
    probes["systemd_manager"] = available
    report = triage.inspect_host(probes=probes)
    _assert_blocked(report)
    assert order == ["procfs", "systemd_manager"]
    assert len(report["observed_failures"]) == 3
    assert report["capabilities"]["systemd_manager"]["observation"] == "OBSERVED_UNTRUSTED"
    assert "PRIVATE_HOST_ENVIRONMENT" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"procfs": lambda: True},
        23,
        {
            "procfs": lambda: True,
            "namespaces": True,
            "pidfd_self": lambda: True,
            "systemd_manager": lambda: True,
        },
    ],
)
def test_invalid_probe_configuration_refuses_without_running_any_probes(invalid):
    report = triage.inspect_host(probes=invalid)
    _assert_blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["capabilities"] == {}
    assert report["observed_failures"] == ["INVALID_PROBE_CONFIGURATION"]


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_non_linux_refusal_precedes_probe_execution(monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)

    def never():
        pytest.fail("non-Linux must not call any Linux probes")

    report = triage.inspect_host(probes=dict.fromkeys(triage._NAMES, never))
    _assert_blocked(report)
    assert report["capabilities"] == {}
    assert report["observed_failures"] == ["NATIVE_LINUX_REQUIRED"]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
def test_cli_always_exits_two_even_with_all_capabilities(monkeypatch, capsys):
    real = triage.inspect_host
    monkeypatch.setattr(triage, "inspect_host", lambda: real(probes=_probes()))
    assert triage.main() == 2
    result = json.loads(capsys.readouterr().out)
    _assert_blocked(result)
    assert len(result["capabilities"]) == len(triage._NAMES)
    assert result["observed_failures"] == []


@pytest.mark.skipif(sys.platform != "linux", reason="Linux self process probe")
def test_real_procfs_and_namespaces_probes_never_return_private_metadata():
    for probe in (triage._procfs, triage._namespaces):
        outcome = probe()
        assert type(outcome) is bool


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd probe")
@pytest.mark.parametrize("bad", [False, None, {"pid": 123}, {"pid": True}])
def test_pidfd_self_refuses_missing_or_malformed_lifetime(monkeypatch, bad):
    monkeypatch.setattr(triage.pidfd, "_read_pidfd_lifetime", lambda _: bad)
    assert triage._pidfd_self() is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux pidfd probe")
def test_pidfd_self_checks_actual_pid_without_emitting_it(monkeypatch):
    observed = []

    def reader(p):
        observed.append(p)
        return {"pid": p, "start_time_ticks": 42, "pidfd_poll_alive": True}

    monkeypatch.setattr(triage.pidfd, "_read_pidfd_lifetime", reader)
    assert triage._pidfd_self() is True
    assert observed == [os.getpid()]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic manager probe")
def test_systemd_manager_command_is_fixed_read_only_with_timeout_and_no_shell(monkeypatch):
    invoked = []

    def runner(args, **kwargs):
        invoked.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="Version=255.4\n")

    monkeypatch.setattr(triage.subprocess, "run", runner)
    assert triage._systemd_manager() is True
    assert len(invoked) == 1
    command, kwargs = invoked[0]
    assert command == (
        "/usr/bin/systemctl",
        "--system",
        "show",
        "--no-pager",
        "--property=Version",
    )
    assert kwargs["timeout"] == 3
    assert kwargs["check"] is False
    assert kwargs["env"] == {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    assert kwargs["stdout"] is triage.subprocess.PIPE
    assert kwargs["stderr"] is triage.subprocess.DEVNULL
    assert "capture_output" not in kwargs
    assert kwargs["text"] is True
    assert "shell" not in kwargs


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic manager probe")
@pytest.mark.parametrize(
    "stdout,returncode",
    [
        ("", 0),
        ("Version=\n", 0),
        ("Version=255\nPRIVATE_SECRET=1\n", 0),
        ("Version=PRIVATE_SECRET\n", 0),
        ("Version=256\0SECRET\n", 0),
        ("Version=256\n", 1),
        ("Version=" + "1" * 300, 0),
        ("Version=" + "1" * 129, 0),
        ("Version=256\tPRIVATE_SECRET\n", 0),
        ("Version=256\x7fPRIVATE_SECRET\n", 0),
        ("Version=256é\n", 0),
        (None, 0),
        (42, 0),
        ("Version=256\n", True),
    ],
)
def test_malformed_systemd_manager_metadata_is_refused(monkeypatch, stdout, returncode):
    def runner(*args, **kwargs):
        return SimpleNamespace(stdout=stdout, returncode=returncode)

    monkeypatch.setattr(triage.subprocess, "run", runner)
    assert triage._systemd_manager() is False


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic manager probe")
def test_systemd_manager_exception_is_sanitized(monkeypatch):
    def runner(*args, **kwargs):
        raise OSError("PRIVATE_HOST_ENVIRONMENT")

    monkeypatch.setattr(triage.subprocess, "run", runner)
    assert triage._systemd_manager() is False


_SELF_PID = 654321
_PRIVATE = "PRIVATE_HOST_ENVIRONMENT"


@pytest.fixture
def linux_probe(monkeypatch):
    """Run synthetic backends independently of the test runner's host OS."""
    monkeypatch.setattr(triage.sys, "platform", "linux")


def _assert_probe_refused(name, probe, *, reason="PROBE_UNAVAILABLE", probes=None):
    if probes is None:
        probes = _probes()
    probes[name] = probe
    report = triage.inspect_host(probes=probes)
    _assert_blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert report["observed_failures"] == [f"{name}: {reason}"]
    assert report["capabilities"][name] == {
        "observation": "NOT_OBSERVED",
        "reason": reason,
    }
    for other in set(triage._NAMES) - {name}:
        assert report["capabilities"][other]["observation"] == "OBSERVED_UNTRUSTED"
    encoded = json.dumps(report)
    for private in (_PRIVATE, str(_SELF_PID), "/proc/", "Version=", "st_ino"):
        assert private not in encoded


def _procfs_backend(
    monkeypatch, *, failures=(), raw=f"{_SELF_PID} ({_PRIVATE}) S 1\n".encode(), mode=stat.S_IFDIR
):
    opened = []
    closed = []
    reads = []

    def fail(operation):
        if operation in failures:
            raise OSError(_PRIVATE)

    def opener(path, flags, *, dir_fd=None):
        fail("root_open" if dir_fd is None else "stat_open")
        opened.append((path, flags, dir_fd))
        return 71 if dir_fd is None else 72

    def fstat(fd):
        assert fd == 71
        fail("fstat")
        return SimpleNamespace(st_mode=mode)

    def reader(fd, limit):
        reads.append((fd, limit))
        fail("read")
        return raw

    def closer(fd):
        closed.append(fd)
        fail("stat_close" if fd == 72 else "root_close")

    backend = SimpleNamespace(
        getpid=lambda: _SELF_PID,
        open=opener,
        fstat=fstat,
        read=reader,
        close=closer,
        O_RDONLY=os.O_RDONLY,
        O_DIRECTORY=getattr(os, "O_DIRECTORY", 65536),
        O_CLOEXEC=getattr(os, "O_CLOEXEC", 524288),
        O_NOFOLLOW=getattr(os, "O_NOFOLLOW", 131072),
    )
    monkeypatch.setattr(triage, "os", backend)
    return opened, closed, reads


def test_procfs_read_is_self_only_bounded_and_closes_in_reverse_order(monkeypatch):
    opened, closed, reads = _procfs_backend(monkeypatch)
    assert triage._procfs() is True
    assert opened == [
        (
            f"/proc/{_SELF_PID}",
            triage.os.O_RDONLY | triage.os.O_DIRECTORY | triage.os.O_CLOEXEC | triage.os.O_NOFOLLOW,
            None,
        ),
        ("stat", triage.os.O_RDONLY | triage.os.O_CLOEXEC | triage.os.O_NOFOLLOW, 71),
    ]
    assert reads == [(72, 4097)]
    assert closed == [72, 71]


@pytest.mark.parametrize(
    "failures,expected_closed",
    [
        (("root_open",), []),
        (("fstat",), [71]),
        (("stat_open",), [71]),
        (("read",), [72, 71]),
        (("stat_close",), [72, 71]),
        (("root_close",), [72, 71]),
        (("stat_close", "root_close"), [72, 71]),
    ],
)
def test_procfs_failures_refuse_and_attempt_every_acquired_close(
    linux_probe, monkeypatch, capsys, failures, expected_closed
):
    _, closed, _ = _procfs_backend(monkeypatch, failures=failures)
    _assert_probe_refused("procfs", triage._procfs)
    assert closed == expected_closed
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"PRIVATE_HOST_ENVIRONMENT",
        b"654322 (PRIVATE_HOST_ENVIRONMENT) S 1\n",
        b"654321 (PRIVATE_HOST_ENVIRONMENT S 1\n",
        b"654321 (PRIVATE_HOST_ENVIRONMENT) S\x00\n",
        b"654321 (PRIVATE_HOST_ENVIRONMENT) S " + b"x" * 4096,
        "654321 (PRIVATE_HOST_ENVIRONMENT) S 1\n",
        bytearray(b"654321 (PRIVATE_HOST_ENVIRONMENT) S 1\n"),
        None,
        42,
    ],
)
def test_malformed_procfs_read_is_private_and_closes_descriptors(linux_probe, monkeypatch, raw):
    _, closed, _ = _procfs_backend(monkeypatch, raw=raw)
    _assert_probe_refused("procfs", triage._procfs)
    assert closed == [72, 71]


def test_procfs_non_directory_refuses_before_file_open(linux_probe, monkeypatch):
    opened, closed, reads = _procfs_backend(monkeypatch, mode=stat.S_IFREG)
    _assert_probe_refused("procfs", triage._procfs)
    assert len(opened) == 1
    assert closed == [71]
    assert reads == []


@pytest.mark.parametrize("failed_namespace", ["mnt", "ipc", "net"])
def test_namespace_read_failure_is_independent_and_sanitized(
    linux_probe, monkeypatch, capsys, failed_namespace
):
    calls = []

    def reader(path):
        calls.append(path)
        if path.endswith(f"/{failed_namespace}"):
            raise OSError(_PRIVATE)
        return SimpleNamespace(st_ino=100, st_mode=stat.S_IFREG)

    monkeypatch.setattr(triage, "os", SimpleNamespace(getpid=lambda: _SELF_PID, stat=reader))
    _assert_probe_refused("namespaces", triage._namespaces)
    expected = ("mnt", "ipc", "net")
    assert calls == [f"/proc/{_SELF_PID}/ns/{name}" for name in expected[: len(calls)]]
    assert len(calls) == expected.index(failed_namespace) + 1
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize(
    "inode,mode",
    [
        (0, stat.S_IFREG),
        (-1, stat.S_IFREG),
        (True, stat.S_IFREG),
        (1.0, stat.S_IFREG),
        ("PRIVATE_HOST_ENVIRONMENT", stat.S_IFREG),
        (None, stat.S_IFREG),
        (100, stat.S_IFDIR),
        (100, stat.S_IFLNK),
        (100, 0),
        (100, True),
        (100, "PRIVATE_HOST_ENVIRONMENT"),
        (100, None),
    ],
)
def test_malformed_namespace_metadata_is_refused(linux_probe, monkeypatch, inode, mode):
    def reader(path):
        return SimpleNamespace(st_ino=inode, st_mode=mode)

    monkeypatch.setattr(triage, "os", SimpleNamespace(getpid=lambda: _SELF_PID, stat=reader))
    _assert_probe_refused("namespaces", triage._namespaces)


@pytest.mark.parametrize("metadata", [None, {}, SimpleNamespace(st_ino=100)])
def test_incomplete_namespace_backend_response_refuses_the_observation(
    linux_probe, monkeypatch, metadata
):
    monkeypatch.setattr(
        triage, "os", SimpleNamespace(getpid=lambda: _SELF_PID, stat=lambda _: metadata)
    )
    _assert_probe_refused("namespaces", triage._namespaces, reason="PROBE_FAILED_OR_MALFORMED")


@pytest.mark.parametrize(
    "changes",
    [
        {"pid": True},
        {"pid": _SELF_PID + 1},
        {"pid": str(_SELF_PID)},
        {"start_time_ticks": True},
        {"start_time_ticks": 0},
        {"start_time_ticks": -1},
        {"start_time_ticks": 2**64},
        {"start_time_ticks": 1.5},
        {"start_time_ticks": float("nan")},
        {"start_time_ticks": float("inf")},
        {"start_time_ticks": None},
        {"pidfd_poll_alive": 1},
        {"pidfd_poll_alive": False},
        {"pidfd_poll_alive": None},
        {"private_extra_field": _PRIVATE},
    ],
)
def test_pidfd_complete_but_invalid_record_is_refused(linux_probe, monkeypatch, changes):
    record = {"pid": _SELF_PID, "start_time_ticks": 42, "pidfd_poll_alive": True, **changes}
    monkeypatch.setattr(triage, "os", SimpleNamespace(getpid=lambda: _SELF_PID))
    monkeypatch.setattr(triage.pidfd, "_read_pidfd_lifetime", lambda _: record)
    _assert_probe_refused("pidfd_self", triage._pidfd_self)


def test_pidfd_backend_exception_refuses_without_private_output(linux_probe, monkeypatch, capsys):
    def reader(pid):
        raise RuntimeError(_PRIVATE)

    monkeypatch.setattr(triage.pidfd, "_read_pidfd_lifetime", reader)
    _assert_probe_refused("pidfd_self", triage._pidfd_self)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failed_probe", triage._NAMES)
@pytest.mark.parametrize("error_type", [OSError, ValueError, RuntimeError])
def test_each_backend_exception_preserves_other_observations(
    linux_probe, capsys, failed_probe, error_type
):
    calls = []
    probes = {}
    for name in triage._NAMES:

        def reader(name=name):
            calls.append(name)
            if name == failed_probe:
                raise error_type(_PRIVATE)
            return True

        probes[name] = reader
    _assert_probe_refused(
        failed_probe, probes[failed_probe], reason="PROBE_FAILED_OR_MALFORMED", probes=probes
    )
    assert calls == list(triage._NAMES)
    assert capsys.readouterr() == ("", "")


def test_partial_observations_keep_complete_blocked_envelope(linux_probe):
    report = triage.inspect_host(
        probes=_probes(
            procfs=True, namespaces=None, pidfd_self=False, systemd_manager={"raw": _PRIVATE}
        )
    )
    _assert_blocked(report)
    assert report["snapshot"] == "SYNTHETIC_TEST_INPUT"
    assert set(report["capabilities"]) == set(triage._NAMES)
    assert report["capabilities"]["procfs"]["observation"] == "OBSERVED_UNTRUSTED"
    assert report["observed_failures"] == [
        "namespaces: PROBE_FAILED_OR_MALFORMED",
        "pidfd_self: PROBE_UNAVAILABLE",
        "systemd_manager: PROBE_FAILED_OR_MALFORMED",
    ]
    assert _PRIVATE not in json.dumps(report)


@pytest.mark.parametrize(
    "error",
    [
        OSError(_PRIVATE),
        ValueError(_PRIVATE),
        TimeoutError(_PRIVATE),
        triage.subprocess.TimeoutExpired("PRIVATE_COMMAND", 3, output=_PRIVATE, stderr=_PRIVATE),
        UnicodeError(_PRIVATE),
    ],
)
def test_systemd_exception_and_timeout_details_are_not_reported(
    linux_probe, monkeypatch, capsys, error
):
    def runner(*args, **kwargs):
        raise error

    monkeypatch.setattr(triage.subprocess, "run", runner)
    _assert_probe_refused("systemd_manager", triage._systemd_manager)
    assert capsys.readouterr() == ("", "")
