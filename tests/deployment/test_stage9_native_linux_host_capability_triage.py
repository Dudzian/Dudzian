"""S9-RPI-33: host capability hints cannot qualify a production source/CCA."""

from __future__ import annotations

import json
import os
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
    assert (
        report["capabilities"]["systemd_manager"]["observation"]
        == "OBSERVED_UNTRUSTED"
    )
    assert "PRIVATE_HOST_ENVIRONMENT" not in json.dumps(report)


@pytest.mark.skipif(sys.platform != "linux", reason="Linux synthetic capability triage")
@pytest.mark.parametrize(
    "invalid",
    [
        {},
        {"procfs": lambda: True},
        23,
        {"procfs": lambda: True, "namespaces": True,
         "pidfd_self": lambda: True, "systemd_manager": lambda: True},
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
def test_systemd_manager_command_is_fixed_read_only_bounded_and_not_shell(monkeypatch):
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
    assert kwargs["capture_output"] is True
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
