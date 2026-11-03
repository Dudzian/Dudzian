"""Modele granic harnessu; te testy nie są wykonaniem natywnej próby VM."""

from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from deployment import stage9_vm_native as native

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Modele obserwacji natywnego Linux")


def state(**changes):
    result = {
        "LoadState": "loaded",
        "ActiveState": "active",
        "SubState": "running",
        "Result": "success",
        "ExecMainCode": "0",
        "ExecMainStatus": "0",
        "MainPID": "811",
        "ExecMainPID": "811",
        "FragmentPath": str(native.UNIT_PATH),
        "DropInPaths": "",
    }
    return {**result, **changes}


@pytest.fixture
def systemd(monkeypatch):
    calls = []
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(native, "os", SimpleNamespace(geteuid=lambda: 0))
    monkeypatch.setattr(native, "_protected_unit", lambda: calls.append("protected"))
    monkeypatch.setattr(native, "_command", lambda arguments: calls.append(arguments) or "")
    monkeypatch.setattr(native, "_state", lambda: state())
    monkeypatch.setattr(native, "_service_observation", lambda: {"pidfd": True})
    return calls


def test_readonly_systemd_observation_does_not_start_or_reload(systemd):
    result = native.observe_systemd()
    assert result["status"] == "PASS"
    assert systemd == [
        "protected",
        ["/usr/bin/systemd-analyze", "verify", "--man=no", str(native.UNIT_PATH)],
    ]


def test_explicit_start_validates_before_each_mutation(systemd):
    assert native.observe_systemd(start=True)["status"] == "PASS"
    assert systemd == [
        "protected",
        ["/usr/bin/systemd-analyze", "verify", "--man=no", str(native.UNIT_PATH)],
        ["/usr/bin/systemctl", "daemon-reload"],
        "protected",
        ["/usr/bin/systemctl", "start", native.UNIT],
    ]


def test_changed_unit_never_reloads_or_starts(systemd, monkeypatch):
    def refuse():
        raise native.NativeObservationError("UNIT_DROPIN_NOT_APPROVED")

    monkeypatch.setattr(native, "_protected_unit", refuse)
    assert native.observe_systemd(start=True)["reason"] == "UNIT_DROPIN_NOT_APPROVED"
    assert systemd == []


@pytest.mark.parametrize(
    "changes",
    [{"ActiveState": "failed"}, {"SubState": "dead"}, {"MainPID": "0"}, {"ExecMainPID": "812"}],
)
def test_inactive_or_wrong_process_is_blocked(systemd, monkeypatch, changes):
    monkeypatch.setattr(native, "_state", lambda: state(**changes))
    result = native.observe_systemd()
    assert result["status"] == "BLOCKED"
    assert result["reason"] == "SERVICE_NOT_RUNNING"


def test_changed_service_state_cannot_pass(systemd, monkeypatch):
    states = iter([state(), state(Result="timeout"), state()])
    monkeypatch.setattr(native, "_state", lambda: next(states))
    assert native.observe_systemd()["reason"] == "SERVICE_STATE_CHANGED"


def test_errors_are_sanitized_and_unavailable_state_does_not_mask_failure(systemd, monkeypatch):
    def fail():
        raise RuntimeError("password=do-not-log-private-value")

    monkeypatch.setattr(native, "_service_observation", fail)
    result = native.observe_systemd()
    assert result["reason"] == "SYSTEMD_IDENTITY_REFUSED"
    assert "private-value" not in json.dumps(result)


def test_nonlinux_or_nonroot_cannot_invoke_native_actions(systemd, monkeypatch):
    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="win32"))
    assert native.observe_systemd(start=True)["status"] == "BLOCKED"
    assert native.observe_postgresql_ipc()["reason"] == "NATIVE_ROOT_REQUIRED"
    assert systemd == []


def test_start_flag_is_strict_boolean(systemd):
    assert native.observe_systemd(start="yes")["reason"] == "SYSTEMD_START_SELECTION_INVALID"
    assert systemd == []


@pytest.mark.parametrize("failure", [OSError("secret"), subprocess.TimeoutExpired("secret", 1)])
def test_native_command_timeout_is_blocked_without_stderr(monkeypatch, failure):
    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(native.subprocess, "run", run)
    with pytest.raises(native.NativeObservationError, match="NATIVE_COMMAND_UNAVAILABLE"):
        native._command(["/usr/bin/systemctl", "daemon-reload"])


def test_native_command_sanitized_environment_and_fail_closed_reply(monkeypatch):
    def run(arguments, **options):
        assert arguments == ["/usr/bin/systemctl", "show"]
        assert options["env"] == native.SAFE_ENV
        assert options["stdin"] == subprocess.DEVNULL
        assert options["stderr"] == subprocess.DEVNULL
        assert options["timeout"] == 30
        return SimpleNamespace(returncode=0, stdout="ok")

    monkeypatch.setattr(native.subprocess, "run", run)
    assert native._command(["/usr/bin/systemctl", "show"]) == "ok"
    with pytest.raises(native.NativeObservationError, match="NATIVE_COMMAND_NOT_APPROVED"):
        native._command(["/bin/sh", "-c", "false"])
    monkeypatch.setattr(
        native.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="secret")
    )
    with pytest.raises(native.NativeObservationError, match="NATIVE_COMMAND_REFUSED"):
        native._command(["/usr/bin/systemctl", "show"])
    monkeypatch.setattr(
        native.subprocess,
        "run",
        lambda *_a, **_kw: SimpleNamespace(returncode=0, stdout="x" * 65537),
    )
    with pytest.raises(native.NativeObservationError, match="NATIVE_COMMAND_REPLY_TOO_LARGE"):
        native._command(["/usr/bin/systemctl", "show"])


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"DropInPaths": "/tmp/foreign.conf"}, "UNIT_DROPIN_NOT_APPROVED"),
        ({"FragmentPath": "/tmp/foreign.service"}, "UNIT_FRAGMENT_CHANGED"),
        ({"MainPID": "password=secret"}, "UNIT_STATE_INVALID"),
        ({"Result": "private diagnostic"}, "UNIT_STATE_INVALID"),
    ],
)
def test_state_parser_rejects_foreign_or_unstructured_values(monkeypatch, changes, reason):
    raw = "\n".join(f"{key}={value}" for key, value in state(**changes).items())
    monkeypatch.setattr(native, "_command", lambda _arguments: raw)
    with pytest.raises(native.NativeObservationError, match=reason):
        native._state()


def test_state_parser_accepts_exact_fields_and_refuses_partial_or_duplicate(monkeypatch):
    raw = "\n".join(f"{key}={value}" for key, value in state().items())
    monkeypatch.setattr(native, "_command", lambda _arguments: raw)
    assert native._state() == state()
    for malformed in (raw + "\nMainPID=811", raw.split("\n", 1)[1], "password=secret"):
        monkeypatch.setattr(native, "_command", lambda _arguments, value=malformed: value)
        with pytest.raises(native.NativeObservationError):
            native._state()


@pytest.fixture
def protected_unit(monkeypatch):
    from deployment import freshness_runtime, production_local_installer_host

    class UnitPath:
        parents = (Path("/etc/systemd/system"),)

        def lstat(self):
            return SimpleNamespace(
                st_mode=stat.S_IFREG | 0o644, st_uid=0, st_gid=0, st_nlink=1, st_size=10
            )

        def read_bytes(self):
            return b"unit"

        def __str__(self):
            return "/etc/systemd/system/cryptohunter-freshness-verifier.service"

    path = UnitPath()
    runtime = SimpleNamespace(executable=Path("/approved/python"))
    checked = []
    original_lstat = Path.lstat

    def lstat_path(current):
        if current == Path("/etc/systemd/system"):
            return SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0, st_gid=0)
        return original_lstat(current)

    monkeypatch.setattr(Path, "lstat", lstat_path)
    monkeypatch.setattr(native, "UNIT_PATH", path)
    monkeypatch.setattr(native, "_state", lambda: state())
    original_query = production_local_installer_host._query
    monkeypatch.setattr(
        production_local_installer_host,
        "_query",
        lambda _arguments: "LoadState=loaded\nFragmentPath=" + str(path) + "\nDropInPaths=",
    )
    monkeypatch.setattr(native.os.path, "lexists", lambda _path: False)
    monkeypatch.setattr(freshness_runtime, "verify_installed_runtime", lambda: runtime)
    monkeypatch.setattr(
        freshness_runtime,
        "validate_runtime_unit",
        lambda content, program: checked.append((content, program)),
    )
    return SimpleNamespace(path=path, runtime=runtime, checked=checked, query=original_query)


def test_protected_unit_reuses_current_exact_runtime_contract(protected_unit):
    assert native._protected_unit() == protected_unit.runtime
    assert protected_unit.checked == [(b"unit", "/approved/python")]


def test_protected_unit_refuses_dropins_before_start(protected_unit, monkeypatch):
    monkeypatch.setattr(native.os.path, "lexists", lambda path: str(path).endswith("service.d"))
    with pytest.raises(native.NativeObservationError, match="UNIT_DROPIN_NOT_APPROVED"):
        native._protected_unit()


def test_new_unit_may_await_explicit_daemon_reload(protected_unit, monkeypatch):
    from deployment import production_local_installer_host

    monkeypatch.setattr(
        production_local_installer_host,
        "_query",
        lambda _arguments: "LoadState=not-found\nFragmentPath=\nDropInPaths=",
    )
    assert native._protected_unit() == protected_unit.runtime


@pytest.mark.parametrize("code", [1, 4])
def test_notfound_exit_status_reuses_existing_exact_host_query(protected_unit, monkeypatch, code):
    from deployment import production_local_installer_host as host

    monkeypatch.setattr(host, "_query", protected_unit.query)

    def run(arguments, **_options):
        assert arguments == host.UNIT_QUERY
        return SimpleNamespace(
            returncode=code, stdout="LoadState=not-found\nFragmentPath=\nDropInPaths=\n"
        )

    monkeypatch.setattr(host.subprocess, "run", run)
    assert native._protected_unit() == protected_unit.runtime


@pytest.mark.parametrize(
    "code,raw",
    [
        (2, "LoadState=not-found\nFragmentPath=\nDropInPaths=\n"),
        (1, "LoadState=not-found\nFragmentPath=\n"),
        (4, "LoadState=not-found\nFragmentPath=/tmp/foreign\nDropInPaths=\n"),
    ],
)
def test_notfound_exception_never_accepts_other_errors(protected_unit, monkeypatch, code, raw):
    from deployment import production_local_installer_host as host

    monkeypatch.setattr(host, "_query", protected_unit.query)
    monkeypatch.setattr(
        host.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(returncode=code, stdout=raw)
    )
    with pytest.raises(native.NativeObservationError, match="UNIT_ORIGIN_UNAVAILABLE"):
        native._protected_unit()


def test_origin_parser_does_not_accept_duplicate_loaded_fields(protected_unit, monkeypatch):
    from deployment import production_local_installer_host as host

    monkeypatch.setattr(host, "_query", lambda _a: "LoadState=loaded\nLoadState=loaded\n")
    with pytest.raises(native.NativeObservationError, match="UNIT_ORIGIN_INVALID"):
        native._protected_unit()


def test_protected_unit_refuses_symlink_or_foreign_owner(protected_unit, monkeypatch):
    monkeypatch.setattr(
        protected_unit.path,
        "lstat",
        lambda: SimpleNamespace(st_mode=stat.S_IFLNK | 0o777, st_uid=0, st_gid=0),
    )
    with pytest.raises(native.NativeObservationError, match="UNIT_PATH_NOT_PROTECTED"):
        native._protected_unit()


@pytest.fixture
def service_flow(monkeypatch):
    import grp
    import pwd

    from bot_core import freshness_service_identity as identity
    from deployment import freshness_runtime, linux_physical_native

    runtime = SimpleNamespace(
        executable=Path("/approved/python"),
        base_executable=Path("/usr/bin/python3.12"),
        base_executable_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )
    process = {
        "pid": 811,
        "uid": 901,
        "gid": 902,
        "start_time_ticks": 44,
        "exe_sha256": "a" * 64,
        "cgroup": identity.CGROUP,
    }
    kernel = {
        **process,
        "no_new_privs": 1,
        "capabilities": {
            key: "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb")
        },
    }
    program = {"argv": ("/approved/python", "-I"), "exe_sha256": "a" * 64, "exe_metadata": (1, 2)}
    properties = {"MainPID": "811", "ControlGroup": identity.CGROUP}
    closed = []
    monkeypatch.setattr(
        native, "os", SimpleNamespace(pidfd_open=lambda pid, flags: 33, close=closed.append)
    )
    monkeypatch.setattr(pwd, "getpwnam", lambda _name: SimpleNamespace(pw_uid=901))
    monkeypatch.setattr(grp, "getgrnam", lambda _name: SimpleNamespace(gr_gid=902))
    monkeypatch.setattr(freshness_runtime, "verify_installed_runtime", lambda: runtime)
    monkeypatch.setattr(identity, "qualify_freshness_service_identity", lambda **_kw: None)
    monkeypatch.setattr(identity, "_systemctl", lambda *_a: "properties")
    monkeypatch.setattr(identity, "_properties", lambda _value: properties)
    monkeypatch.setattr(identity, "_check_pidfd", lambda _fd, _pid: None)
    monkeypatch.setattr(identity, "capture_process_identity", lambda _pid: dict(process))
    monkeypatch.setattr(identity, "_program_observation", lambda _pid, _runtime: dict(program))
    monkeypatch.setattr(identity, "_read_exec_start", lambda _pid, _runtime: ("exec",))
    monkeypatch.setattr(identity, "_read_bootstrap_start", lambda: ("pre",))
    monkeypatch.setattr(identity, "expected_runtime_argv", lambda _runtime: program["argv"])
    monkeypatch.setattr(linux_physical_native, "process_observation", lambda _pid: dict(kernel))
    return SimpleNamespace(kernel=kernel, process=process, closed=closed, identity=identity)


def test_service_evidence_keeps_real_qualifier_and_pinned_metadata(service_flow):
    evidence = native._service_observation()
    assert evidence["pidfd"] is True
    assert evidence["proc_exe"] == "/usr/bin/python3.12"
    assert evidence["program"]["exe_metadata"] == (1, 2)
    assert service_flow.closed == [33]


def test_service_capability_or_process_drift_refuses_and_closes_pidfd(service_flow):
    service_flow.kernel["capabilities"]["CapEff"] = "0000000000000001"
    with pytest.raises(native.NativeObservationError, match="SERVICE_PROCESS_BINDING_CHANGED"):
        native._service_observation()
    assert service_flow.closed == [33]


def test_service_replacement_during_evidence_refuses(service_flow, monkeypatch):
    records = iter([dict(service_flow.process), {**service_flow.process, "pid": 812}])
    monkeypatch.setattr(
        service_flow.identity, "capture_process_identity", lambda _pid: next(records)
    )
    with pytest.raises(native.NativeObservationError, match="SERVICE_CHANGED_DURING_OBSERVATION"):
        native._service_observation()
    assert service_flow.closed == [33]


@pytest.fixture
def ipc_flow(monkeypatch):
    from bot_core import linux_service_ipc
    from deployment import linux_physical_ipc

    monkeypatch.setattr(native, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(native, "os", SimpleNamespace(geteuid=lambda: 0))
    monkeypatch.setattr(linux_service_ipc, "load_installed_qualification_policy", lambda: "b" * 64)
    process = {
        "pid": 811,
        "uid": 901,
        "gid": 902,
        "start_time_ticks": 44,
        "exe_sha256": "a" * 64,
        "cgroup": "/system.slice/" + native.UNIT,
    }
    monkeypatch.setattr(native, "_service_observation", lambda: {"process": dict(process)})
    observation = {"response": {"postgresql": {"service": "real-observation-model"}}}
    calls = []

    def collect(path, **context):
        assert path == native.SOCKET_PATH
        assert context["expected"].pid == 811
        assert context["expected_cgroup"] == process["cgroup"]
        assert len(context["challenge_id"]) == 64
        assert context["installation_id"] == "b" * 64
        calls.append(context)
        return observation

    monkeypatch.setattr(linux_physical_ipc, "collect_verifier_observation", collect)
    monkeypatch.setattr(
        linux_physical_ipc,
        "validate_verifier_observation",
        lambda value, **context: calls.append(context),
    )
    return SimpleNamespace(
        observation=observation,
        calls=calls,
        ipc=linux_service_ipc,
        collector=linux_physical_ipc,
        process=process,
    )


def test_real_partial_exchange_cannot_promote_full_ipc_or_production(ipc_flow):
    result = native.observe_postgresql_ipc()
    assert result["status"] == result["production_qualification"] == "BLOCKED"
    assert result["diagnostic_exchange"] == result["postgresql_service_connection"] == "PASS"
    assert result["evidence"] == ipc_flow.observation
    assert ipc_flow.calls[0] == ipc_flow.calls[1]


def test_no_policy_does_not_create_or_collect_anything(ipc_flow, monkeypatch):
    monkeypatch.setattr(ipc_flow.ipc, "load_installed_qualification_policy", lambda: None)
    assert native.observe_postgresql_ipc()["reason"] == "QUALIFICATION_POLICY_NOT_INSTALLED"
    assert ipc_flow.calls == []


def test_missing_service_postgres_observation_is_blocked(ipc_flow):
    ipc_flow.observation["response"]["postgresql"] = None
    assert native.observe_postgresql_ipc()["postgresql_service_connection"] == "BLOCKED"


def test_swapped_endpoint_or_signature_authority_refusal_never_passes(ipc_flow, monkeypatch):
    def refuse(*_a, **_kw):
        raise ValueError("private endpoint / password")

    monkeypatch.setattr(ipc_flow.collector, "collect_verifier_observation", refuse)
    result = native.observe_postgresql_ipc()
    assert result == {"status": "FAIL", "reason": "POSTGRESQL_IPC_OBSERVATION_REFUSED"}


def test_process_swap_during_ipc_refuses(ipc_flow, monkeypatch):
    services = iter([{"process": ipc_flow.process}, {"process": {**ipc_flow.process, "pid": 812}}])
    monkeypatch.setattr(native, "_service_observation", lambda: next(services))
    assert native.observe_postgresql_ipc()["reason"] == "SERVICE_CHANGED_DURING_IPC"
