"""Synthetic runtime correlation exercises, never physical isolation evidence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from deployment import stage9_trusted_installation_runtime as runtime
from deployment.stage9_trusted_installation_contract import FileExpectation, RoleExpectation

_PID = 4321
_TICKS = 98765


def _proc_stat(ticks=_TICKS):
    fields = ["S", *(str(index) for index in range(1, 19)), str(ticks)]
    return f"{_PID} (synthetic runtime) ".encode() + " ".join(fields).encode() + b"\n"


def _systemd(expectation):
    return {
        "Id": expectation.unit,
        "LoadState": "loaded",
        "ActiveState": "active",
        "User": expectation.principal,
        "Group": expectation.principal,
        "MainPID": str(_PID),
        "FragmentPath": expectation.unit_file.path,
        "DropInPaths": "",
        "ControlGroup": expectation.control_group,
        "Names": expectation.unit,
        "DynamicUser": "no",
        "SupplementaryGroups": "",
        "NeedDaemonReload": "no",
    }


@pytest.fixture
def observation(tmp_path, monkeypatch):
    if sys.platform != "linux":
        pytest.skip("Linux descriptor-relative runtime checks")
    executable = tmp_path / "synthetic-executable"
    content = b"synthetic executable bytes\n"
    executable.write_bytes(content)
    executable.chmod(0o755)
    info = executable.stat()
    expected_file = FileExpectation(
        str(executable), hashlib.sha256(content).hexdigest(), info.st_uid, info.st_gid, 0o755
    )
    # File metadata is owned by the unprivileged test runner. These explicitly
    # constructed dataclasses bypass manifest parsing solely for adapter tests;
    # the production parser requires protected root-owned installation files.
    expectation = RoleExpectation(
        "root_source",
        "synthetic-installation-identity",
        "synthetic-root-source.service",
        "synthetic_root_source",
        31001,
        32001,
        (),
        FileExpectation("/etc/systemd/system/synthetic-root-source.service", "a" * 64, 0, 0, 0o644),
        expected_file,
        "/system.slice/synthetic-root-source.service",
    )
    proc = tmp_path / "synthetic-proc"
    proc.mkdir()
    (proc / "exe").symlink_to(executable)
    (proc / "stat").write_bytes(_proc_stat())
    (proc / "cgroup").write_text(f"0::{expectation.control_group}\n")
    snapshot = {
        "pid": _PID,
        "start_time_ticks": _TICKS,
        "uid": expectation.uid,
        "gid": expectation.gid,
        "groups": [expectation.gid],
        "namespaces": {"mnt": 101, "ipc": 102, "net": 103},
    }
    state = SimpleNamespace(
        expectation=expectation,
        executable=executable,
        proc=proc,
        unit=_systemd(expectation),
        unit_returncode=0,
        unit_raw=None,
        unit_calls=0,
        account_calls=0,
        account={
            "uid": expectation.uid,
            "gid": expectation.gid,
            "groups": [expectation.gid],
            "canonical": True,
        },
        account_returncode=0,
        identity=snapshot,
        executable_observation={
            "pid": _PID,
            "start_time_ticks": _TICKS,
            "sha256": expected_file.sha256,
            "size": len(content),
        },
        cgroup=(_PID, _TICKS, expectation.control_group),
    )
    original_open = os.open

    def mapped_open(path, flags, *args, **kwargs):
        if path == f"/proc/{_PID}":
            path = proc
        return original_open(path, flags, *args, **kwargs)

    def runner(command, **kwargs):
        assert kwargs == {"timeout": 3, "env": {"LC_ALL": "C", "PATH": "/usr/bin:/bin"}}
        if command[0] == sys.executable:
            assert command[1:3] == ("-I", "-c")
            assert command[-1] == expectation.principal
            state.account_calls += 1
            return SimpleNamespace(
                returncode=state.account_returncode, stdout=json.dumps(state.account)
            )
        assert command[:4] == ("/usr/bin/systemctl", "--system", "show", "--no-pager")
        assert command[-1] == expectation.unit
        state.unit_calls += 1
        raw = state.unit_raw
        if raw is None:
            raw = "".join(f"{name}={value}\n" for name, value in state.unit.items())
        return SimpleNamespace(returncode=state.unit_returncode, stdout=raw)

    monkeypatch.setattr(runtime.os, "open", mapped_open)
    monkeypatch.setattr(runtime, "_run_systemctl", runner)
    monkeypatch.setattr(runtime, "_read_proc_identity", lambda *_args: state.identity.copy())
    monkeypatch.setattr(
        runtime, "_read_executable", lambda _pid: state.executable_observation.copy()
    )
    monkeypatch.setattr(runtime, "_read_process_cgroup", lambda _pid: state.cgroup)
    fd = os.open(executable, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    state.fd = fd
    try:
        yield state
    finally:
        os.close(fd)


def _observe(state):
    return runtime.observe_service(state.expectation, state.fd)


def test_matching_synthetic_observations_correlate_twice(observation):
    result = _observe(observation)
    assert result == observation.identity
    assert observation.unit_calls == observation.account_calls == 2
    assert "qualification" not in result
    assert str(observation.executable) not in json.dumps(result)
    os.fstat(observation.fd)  # The caller still owns the retained descriptor.


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("Names", "synthetic-root-source.service alias.service", "SYSTEMD_ALIASES_FORBIDDEN"),
        ("DropInPaths", "/private/override.conf", "UNEXPECTED_SYSTEMD_DROPINS"),
        ("NeedDaemonReload", "yes", "SYSTEMD_CONFIGURATION_CHANGED"),
        ("FragmentPath", "/private/other.service", "SYSTEMD_IDENTITY_MISMATCH"),
        ("ControlGroup", "/system.slice/other.service", "SYSTEMD_IDENTITY_MISMATCH"),
        ("DynamicUser", "yes", "SYSTEMD_IDENTITY_MISMATCH"),
        ("SupplementaryGroups", "unexpected", "SYSTEMD_IDENTITY_MISMATCH"),
        ("User", "other_account", "SYSTEMD_IDENTITY_MISMATCH"),
        ("Group", "other_group", "SYSTEMD_IDENTITY_MISMATCH"),
        ("MainPID", "0", "SYSTEMD_IDENTITY_MISMATCH"),
        ("MainPID", "True", "SYSTEMD_IDENTITY_MISMATCH"),
    ],
)
def test_systemd_mismatches_refused(observation, field, value, code):
    observation.unit[field] = value
    with pytest.raises(runtime.RuntimeRefusal, match=code):
        _observe(observation)


@pytest.mark.parametrize(
    "raw",
    ["", "Id=one\nId=two\n", "Unknown=x\n", "PRIVATE\x00PATH", False, 7, [], {}, "é\n"],
)
def test_malformed_systemd_is_refused(observation, raw):
    observation.unit_raw = raw
    with pytest.raises(runtime.RuntimeRefusal, match="SYSTEMD_RESPONSE_INVALID"):
        _observe(observation)


def test_systemd_unavailable(observation):
    observation.unit_returncode = 1
    with pytest.raises(runtime.RuntimeRefusal, match="SYSTEMD_UNAVAILABLE"):
        _observe(observation)


def test_boolean_systemd_returncode_is_not_success(observation):
    observation.unit_returncode = False
    with pytest.raises(runtime.RuntimeRefusal, match="SYSTEMD_UNAVAILABLE"):
        _observe(observation)


def test_missing_systemd_property_is_refused(observation):
    del observation.unit["Names"]
    with pytest.raises(runtime.RuntimeRefusal, match="SYSTEMD_RESPONSE_INVALID"):
        _observe(observation)


def test_stale_pid_is_refused(observation):
    observation.unit["MainPID"] = "2147483646"
    with pytest.raises(runtime.RuntimeRefusal, match="PROCESS_UNAVAILABLE"):
        _observe(observation)


@pytest.mark.parametrize("field", ["uid", "gid", "groups", "canonical"])
def test_nss_identity_and_group_aliases_refused(observation, field):
    observation.account[field] = {
        "uid": observation.expectation.uid + 1,
        "gid": observation.expectation.gid + 1,
        "groups": [observation.expectation.gid, 999],
        "canonical": False,
    }[field]
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_MISMATCH"):
        _observe(observation)


def test_nss_unavailable(observation):
    observation.account_returncode = 1
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_UNAVAILABLE"):
        _observe(observation)


@pytest.mark.parametrize("value", [None, False, 1, [], {"canonical": True}])
def test_malformed_nss_primitive_is_refused(observation, value):
    observation.account = value
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_MISMATCH"):
        _observe(observation)


def test_boolean_nss_returncode_is_not_success(observation):
    observation.account_returncode = False
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_UNAVAILABLE"):
        _observe(observation)


def test_timeout_is_bounded_and_sanitized(observation, monkeypatch):
    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("PRIVATE_CUSTODY_PATH", 3)

    monkeypatch.setattr(runtime, "_run_systemctl", timeout)
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_UNAVAILABLE") as error:
        _observe(observation)
    assert "PRIVATE_CUSTODY_PATH" not in str(error.value)


@pytest.mark.parametrize("source", ["identity", "executable", "cgroup", "retained_stat"])
def test_process_start_time_changes_refused(observation, source):
    if source == "identity":
        observation.identity["start_time_ticks"] += 1
    elif source == "executable":
        observation.executable_observation["start_time_ticks"] += 1
    elif source == "cgroup":
        observation.cgroup = (_PID, _TICKS + 1, observation.expectation.control_group)
    else:
        (observation.proc / "stat").write_bytes(_proc_stat(_TICKS + 1))
    with pytest.raises(runtime.RuntimeRefusal, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        _observe(observation)


@pytest.mark.parametrize("field", ["uid", "gid", "groups"])
def test_effective_process_principal_must_match(observation, field):
    observation.identity[field] = [999] if field == "groups" else 999
    with pytest.raises(runtime.RuntimeRefusal, match="PROCESS_IDENTITY_MISMATCH"):
        _observe(observation)


def test_process_namespace_change_between_sweeps_refused(observation, monkeypatch):
    calls = 0

    def identity(*_args):
        nonlocal calls
        calls += 1
        result = observation.identity.copy()
        result["namespaces"] = {"mnt": 101 + calls, "ipc": 102, "net": 103}
        return result

    monkeypatch.setattr(runtime, "_read_proc_identity", identity)
    with pytest.raises(runtime.RuntimeRefusal, match="PROCESS_CHANGED_DURING_OBSERVATION"):
        _observe(observation)


def test_executable_other_object_with_same_bytes_refused(observation):
    other = observation.executable.with_name("other-executable")
    other.write_bytes(observation.executable.read_bytes())
    other.chmod(0o755)
    (observation.proc / "exe").unlink()
    (observation.proc / "exe").symlink_to(other)
    with pytest.raises(runtime.RuntimeRefusal, match="EXECUTABLE_OBJECT_MISMATCH"):
        _observe(observation)


def test_executable_hardlink_refused(observation):
    os.link(observation.executable, observation.executable.with_name("hardlink"))
    with pytest.raises(runtime.RuntimeRefusal, match="EXECUTABLE_OBJECT_MISMATCH"):
        _observe(observation)


def test_executable_digest_must_match(observation):
    observation.executable_observation["sha256"] = "f" * 64
    with pytest.raises(runtime.RuntimeRefusal, match="EXECUTABLE_DIGEST_MISMATCH"):
        _observe(observation)


def test_executable_replacement_during_observation_refused(observation, monkeypatch):
    def changed(_pid):
        replacement = observation.executable.with_name("replacement")
        replacement.write_bytes(observation.executable.read_bytes())
        replacement.chmod(0o755)
        replacement.replace(observation.executable)
        return observation.executable_observation.copy()

    monkeypatch.setattr(runtime, "_read_executable", changed)
    with pytest.raises(runtime.RuntimeRefusal, match="EXECUTABLE_OBJECT_MISMATCH"):
        _observe(observation)


@pytest.mark.parametrize("nested", [False, True])
def test_process_service_membership_must_match_exact_cgroup(observation, nested):
    group = observation.expectation.control_group + "/nested" if nested else "/other.service"
    observation.cgroup = (_PID, _TICKS, group)
    with pytest.raises(runtime.RuntimeRefusal, match="PROCESS_CGROUP_MISMATCH"):
        _observe(observation)


def test_systemd_changed_between_sweeps_refused(observation, monkeypatch):
    original = runtime._unit
    calls = 0

    def changed(expectation):
        nonlocal calls
        calls += 1
        result = original(expectation)
        return (_PID + 1, result[1]) if calls == 2 else result

    monkeypatch.setattr(runtime, "_unit", changed)
    with pytest.raises(runtime.RuntimeRefusal, match="SYSTEMD_CONFIGURATION_CHANGED"):
        _observe(observation)


def test_policy_does_not_permit_supplementary_group_override(observation):
    observation.expectation = replace(observation.expectation, supplementary_groups=(999,))
    with pytest.raises(runtime.RuntimeRefusal, match="NSS_IDENTITY_MISMATCH"):
        _observe(observation)


def test_non_linux_refused(observation, monkeypatch):
    monkeypatch.setattr(runtime.sys, "platform", "win32")
    with pytest.raises(runtime.RuntimeRefusal, match="NATIVE_LINUX_REQUIRED"):
        _observe(observation)


def test_unknown_error_code_sanitized():
    assert str(runtime.RuntimeRefusal("PRIVATE_CUSTODY_PATH")) == "RUNTIME_OBSERVATION_FAILED"
