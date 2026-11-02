"""Odczytowe modele i odmowy tożsamości usługi, bez kwalifikacji produkcyjnej."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from bot_core import (
    freshness_deployment_qualification as qualification,
    freshness_service_identity as identity,
)
from bot_core.production_signer_ipc import ProductionSignerIPCError

PID = 54123
UID = 910
GID = 911
LINUX = pytest.mark.skipif(sys.platform != "linux", reason="Odczyt /proc i pidfd wymaga Linux")


def properties(**changes):
    result = dict(identity._STATIC_PROPERTIES)
    result.update(MainPID=str(PID), ExecMainPID=str(PID), ExecMainStartTimestampMonotonic="300")
    result.update(changes)
    return result


def properties_reply(value=None):
    return "\n".join(f"{key}={item}" for key, item in (value or properties()).items()) + "\n"


def exec_command():
    return [identity.EXECUTABLE, list(identity.EXPECTED_ARGV), False, 200, 300, 0, 0, PID, 0, 0]


def exec_reply(command=None):
    return json.dumps(
        {"type": "a(sasbttttuii)", "data": [exec_command() if command is None else command]}
    )


def process_identity(**changes):
    result = {
        "pid": PID,
        "uid": UID,
        "gid": GID,
        "groups": [GID],
        "start_time_ticks": 300,
        "exe_sha256": "a" * 64,
        "cgroup": identity.CGROUP,
    }
    result.update(changes)
    return result


@pytest.fixture
def flow(monkeypatch):
    """Zastąp granice hosta; sam parser i logika kwalifikacji pozostają rzeczywiste."""
    state = SimpleNamespace(
        properties=[properties(), properties()],
        commands=[exec_reply(), exec_reply()],
        processes=[process_identity(), process_identity()],
        programs=[
            {"argv": identity.EXPECTED_ARGV, "exe_sha256": "a" * 64, "exe_metadata": (1, 2)},
            {"argv": identity.EXPECTED_ARGV, "exe_sha256": "a" * 64, "exe_metadata": (1, 2)},
        ],
        calls=[],
        closed=[],
        checks=[],
    )

    def systemctl(*arguments):
        state.calls.append(arguments)
        return properties_reply(state.properties.pop(0))

    def query(arguments):
        assert arguments == [
            "/usr/bin/busctl",
            "--system",
            "--json=short",
            "get-property",
            "org.freedesktop.systemd1",
            identity._OBJECT_PATH,
            "org.freedesktop.systemd1.Service",
            "ExecStart",
        ]
        return state.commands.pop(0)

    monkeypatch.setattr(identity, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(
        identity,
        "os",
        SimpleNamespace(
            pidfd_open=lambda pid, flags: 93
            if (pid, flags) == (PID, 0)
            else pytest.fail("Obcy PID"),
            close=state.closed.append,
        ),
    )
    monkeypatch.setattr(identity, "_check_pidfd", lambda fd, pid: state.checks.append((fd, pid)))
    monkeypatch.setattr(identity, "_query", query)
    monkeypatch.setattr(identity, "capture_process_identity", lambda pid: state.processes.pop(0))
    monkeypatch.setattr(identity, "_program_observation", lambda pid: state.programs.pop(0))
    state.run = lambda: identity.qualify_freshness_service_identity(
        uid=UID, gid=GID, systemctl=systemctl
    )
    return state


def test_exact_effective_unit_and_process_binding_accepts_only_complete_observation(flow):
    assert flow.run() is None
    assert len(flow.calls) == 2
    assert flow.calls[0] == flow.calls[1]
    assert flow.calls[0][:3] == ("show", identity.UNIT_NAME, "--no-pager")
    assert set(flow.calls[0][3].removeprefix("--property=").split(",")) == set(
        identity.SYSTEMD_PROPERTIES
    )
    assert flow.checks == [(93, PID)] * 3
    assert flow.closed == [93]


@pytest.mark.parametrize(
    "field,value",
    [
        (
            "DropInPaths",
            "/etc/systemd/system/cryptohunter-freshness-verifier.service.d/foreign.conf",
        ),
        ("FragmentPath", "/usr/lib/systemd/system/cryptohunter-freshness-verifier.service"),
        ("User", "root"),
        ("Group", "root"),
        ("ControlGroup", "/system.slice/foreign.service"),
        ("ActiveState", "inactive"),
        ("SubState", "exited"),
        ("Type", "forking"),
        ("DynamicUser", "yes"),
        ("SupplementaryGroups", "sudo"),
        ("EnvironmentFiles", "/etc/foreign.conf"),
        ("PassEnvironment", "PYTHONPATH"),
        ("Environment", "PYTHONNOUSERSITE=1 PYTHONPATH=/tmp"),
        ("WorkingDirectory", "/tmp"),
        ("RootDirectory", "/foreign"),
        ("RootImage", "/foreign.raw"),
        ("NeedDaemonReload", "yes"),
        ("NoNewPrivileges", "no"),
        ("ProtectSystem", "no"),
        ("ProtectHome", "no"),
        ("PrivateTmp", "no"),
        ("MemoryDenyWriteExecute", "no"),
        ("CapabilityBoundingSet", "cap_sys_admin"),
        ("AmbientCapabilities", "cap_setuid"),
        ("RestrictAddressFamilies", "AF_UNIX AF_INET"),
        ("RuntimeDirectoryMode", "0777"),
        ("UMask", "0000"),
    ],
)
def test_effective_configuration_drift_refuses_before_process_lookup(flow, field, value):
    flow.properties[0][field] = value
    with pytest.raises(
        identity.ServiceIdentityError, match="EFFECTIVE_SYSTEMD_CONFIGURATION_CHANGED"
    ):
        flow.run()
    assert flow.checks == []
    assert flow.closed == []


@pytest.mark.parametrize(
    "command",
    [
        ["/bin/echo", ["/bin/echo", identity.EXPECTED_ARGV[-1]], False, 200, 300, 0, 0, PID, 0, 0],
        [
            "/bin/sh",
            ["/bin/sh", "-c", " ".join(identity.EXPECTED_ARGV)],
            False,
            200,
            300,
            0,
            0,
            PID,
            0,
            0,
        ],
        ["/usr/bin/python3.12", list(identity.EXPECTED_ARGV), False, 200, 300, 0, 0, PID, 0, 0],
        [
            identity.EXECUTABLE,
            list(identity.EXPECTED_ARGV) + ["--foreign"],
            False,
            200,
            300,
            0,
            0,
            PID,
            0,
            0,
        ],
        [
            identity.EXECUTABLE,
            list(identity.EXPECTED_ARGV[:3]) + list(identity.EXPECTED_ARGV[4:]),
            False,
            200,
            300,
            0,
            0,
            PID,
            0,
            0,
        ],
        [
            identity.EXECUTABLE,
            [identity.EXECUTABLE, "-s", "-I", "-B", identity.EXPECTED_ARGV[-1]],
            False,
            200,
            300,
            0,
            0,
            PID,
            0,
            0,
        ],
        [identity.EXECUTABLE, list(identity.EXPECTED_ARGV), True, 200, 300, 0, 0, PID, 0, 0],
    ],
)
def test_alternative_binary_embedded_path_or_changed_arguments_are_refused(flow, command):
    flow.commands[0] = exec_reply(command)
    with pytest.raises(
        identity.ServiceIdentityError, match="EXECSTART_PROGRAM_OR_ARGUMENTS_CHANGED"
    ):
        flow.run()
    assert flow.closed == [93]


@pytest.mark.parametrize(
    "index,value",
    [(3, 0), (4, 0), (5, 1), (6, 1), (7, PID + 1), (8, 1), (9, 1), (7, True), (4, 1.0)],
)
def test_execstart_must_describe_the_running_main_process(index, value):
    command = exec_command()
    command[index] = value
    with pytest.raises(identity.ServiceIdentityError, match="EXECSTART_PROCESS_NOT_MATCHED"):
        identity._exec_start(exec_reply(command), PID)


@pytest.mark.parametrize(
    "reply",
    [
        "not-json",
        "null",
        "[]",
        "{}",
        '{"type":"a(sasbttttuii)","type":"a(sasbttttuii)","data":[]}',
        json.dumps({"type": "as", "data": [exec_command()]}),
        json.dumps({"type": "a(sasbttttuii)", "data": []}),
        json.dumps({"type": "a(sasbttttuii)", "data": [exec_command(), exec_command()]}),
        json.dumps({"type": "a(sasbttttuii)", "data": [[exec_command()]]}),
        json.dumps({"type": "a(sasbttttuii)", "data": [exec_command()[:-1]]}),
    ],
)
def test_execstart_malformed_multiple_or_call_style_wrapped_value_refuses(reply):
    with pytest.raises(identity.ServiceIdentityError):
        identity._exec_start(reply, PID)


def test_get_property_uses_array_value_without_method_call_wrapper():
    # Natywny busctl get-property wypisuje data jako samą wartość array.
    observed = identity._exec_start(exec_reply(), PID)
    assert observed == tuple([identity.EXECUTABLE, identity.EXPECTED_ARGV, *exec_command()[2:]])


@pytest.mark.parametrize(
    "change",
    [
        {"pid": PID + 1},
        {"uid": UID + 1},
        {"gid": GID + 1},
        {"groups": [GID, 0]},
        {"cgroup": "/user.slice/foreign.scope"},
    ],
)
def test_kernel_credentials_or_cgroup_drift_refuse(flow, change):
    flow.processes[0].update(change)
    with pytest.raises(
        identity.ServiceIdentityError, match="SERVICE_PROCESS_CREDENTIALS_OR_CGROUP_CHANGED"
    ):
        flow.run()
    assert flow.closed == [93]


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_time_ticks", 301),
        ("pid", PID + 1),
        ("uid", UID + 1),
        ("gid", GID + 1),
        ("groups", []),
        ("cgroup", "/other"),
        ("exe_sha256", "b" * 64),
    ],
)
def test_reused_pid_or_changed_identity_during_observation_refuses(flow, field, value):
    flow.processes[1][field] = value
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_PROCESS_IDENTITY_CHANGED"):
        flow.run()
    assert flow.closed == [93]


@pytest.mark.parametrize(
    "argv",
    [
        ("/bin/echo", identity.EXPECTED_ARGV[-1]),
        identity.EXPECTED_ARGV + ("--foreign",),
        (identity.EXECUTABLE, "-I", "-s", "-c", identity.EXPECTED_ARGV[-1]),
    ],
)
def test_actual_proc_argv_must_match_effective_execstart(flow, argv):
    flow.programs[0]["argv"] = argv
    with pytest.raises(identity.ServiceIdentityError, match="PROCESS_ARGUMENTS_CHANGED"):
        flow.run()
    assert flow.closed == [93]


def test_process_program_digest_must_match_independent_proc_identity(flow):
    flow.programs[0]["exe_sha256"] = "b" * 64
    with pytest.raises(identity.ServiceIdentityError, match="PROCESS_PROGRAM_DIGEST_CHANGED"):
        flow.run()


def test_program_replacement_during_observation_refuses(flow):
    flow.programs[1]["exe_metadata"] = (7, 8)
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_PROCESS_PROGRAM_CHANGED"):
        flow.run()


def test_effective_execstart_change_during_observation_refuses(flow):
    command = exec_command()
    command[3] += 1
    flow.commands[1] = exec_reply(command)
    with pytest.raises(identity.ServiceIdentityError, match="EFFECTIVE_EXECSTART_CHANGED"):
        flow.run()


def test_mainpid_change_between_unit_observations_refuses(flow):
    flow.properties[1].update(MainPID=str(PID + 1), ExecMainPID=str(PID + 1))
    with pytest.raises(identity.ServiceIdentityError, match="EFFECTIVE_SERVICE_CHANGED"):
        flow.run()


@pytest.mark.parametrize("boundary", ["pidfd", "capture", "program"])
def test_unavailable_native_boundary_refuses_and_closes_open_handle(flow, monkeypatch, boundary):
    def unavailable(*_arguments):
        raise PermissionError("Syntetyczna niedostępność granicy")

    if boundary == "pidfd":
        monkeypatch.setattr(identity.os, "pidfd_open", unavailable)
    elif boundary == "capture":
        monkeypatch.setattr(identity, "capture_process_identity", unavailable)
    else:
        monkeypatch.setattr(identity, "_program_observation", unavailable)
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_IDENTITY_UNAVAILABLE"):
        flow.run()
    assert flow.closed == ([] if boundary == "pidfd" else [93])


def test_capture_credential_alias_error_remains_fail_closed(flow, monkeypatch):
    def ambiguous(*_arguments):
        raise ProductionSignerIPCError("QUALIFIER_PROCESS_CREDENTIALS_AMBIGUOUS")

    monkeypatch.setattr(identity, "capture_process_identity", ambiguous)
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_IDENTITY_UNAVAILABLE"):
        flow.run()


def test_pidfd_lifetime_failure_stops_later_observations(flow, monkeypatch):
    def exited(*_arguments):
        raise identity.ServiceIdentityError("SERVICE_PROCESS_EXITED")

    monkeypatch.setattr(identity, "_check_pidfd", exited)
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_PROCESS_EXITED"):
        flow.run()
    assert len(flow.commands) == 2
    assert flow.closed == [93]


@pytest.mark.parametrize("uid,gid", [(0, GID), (UID, 0), (True, GID), (UID, -1), ("910", GID)])
def test_root_or_non_exact_expected_identity_is_refused(monkeypatch, uid, gid):
    monkeypatch.setattr(identity, "sys", SimpleNamespace(platform="linux"))
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_UID_GID_REQUIRED"):
        identity.qualify_freshness_service_identity(uid=uid, gid=gid)


@pytest.mark.parametrize(
    "reply",
    [
        properties_reply() + "MainPID=100\n",
        properties_reply().replace("DropInPaths=\n", ""),
        properties_reply() + "Unknown=anything\n",
        properties_reply().replace("MainPID=54123", "MainPID=0"),
        properties_reply().replace("ExecMainPID=54123", "ExecMainPID=54124"),
        properties_reply().replace("MainPID=54123", "MainPID=+54123"),
        properties_reply().replace("Id=", "Id=\t", 1),
    ],
)
def test_incomplete_ambiguous_or_malformed_effective_properties_refuse(reply):
    with pytest.raises(identity.ServiceIdentityError):
        identity._properties(reply)


@pytest.mark.parametrize(
    "error",
    [OSError("niedostępny"), subprocess.TimeoutExpired("busctl", 15), UnicodeError("kodowanie")],
)
def test_service_query_errors_never_return_success(monkeypatch, error):
    def unavailable(*_arguments, **_keywords):
        raise error

    monkeypatch.setattr(identity.subprocess, "run", unavailable)
    with pytest.raises(identity.ServiceIdentityError, match="SERVICE_MANAGER_UNAVAILABLE"):
        identity._query(["/usr/bin/busctl"])


@pytest.mark.parametrize("returncode,stdout", [(1, ""), (0, "x" * 65537)])
def test_service_query_nonzero_or_unbounded_reply_refuses(monkeypatch, returncode, stdout):
    monkeypatch.setattr(
        identity.subprocess,
        "run",
        lambda *_args, **_kw: SimpleNamespace(returncode=returncode, stdout=stdout),
    )
    with pytest.raises(identity.ServiceIdentityError):
        identity._query(["/usr/bin/busctl"])


def test_service_query_uses_fixed_executable_and_isolated_environment(monkeypatch):
    calls = []

    def run(arguments, **options):
        calls.append((arguments, options))
        return SimpleNamespace(returncode=0, stdout="MainPID=54123\n")

    monkeypatch.setattr(identity.subprocess, "run", run)
    assert identity._systemctl("show", identity.UNIT_NAME) == "MainPID=54123\n"
    arguments, options = calls[0]
    assert arguments == ["/usr/bin/systemctl", "show", identity.UNIT_NAME]
    assert options["env"] == {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C", "LC_ALL": "C"}
    assert options["cwd"] == "/"
    assert options["timeout"] == 15
    assert options["stdin"] == options["stderr"] == subprocess.DEVNULL
    assert "shell" not in options


def test_live_qualifier_propagates_identity_refusal_before_database(monkeypatch):
    """Zamodeluj wcześniejsze granice, by sprawdzić tylko negatywną kompozycję."""
    verifier = SimpleNamespace(pw_uid=UID, pw_gid=913)
    runtime = SimpleNamespace(pw_uid=UID + 1, pw_gid=914)
    group = SimpleNamespace(
        gr_gid=GID,
        gr_mem=[qualification.VERIFIER_USER, qualification.RUNTIME_USER],
    )
    monkeypatch.setitem(
        sys.modules,
        "pwd",
        SimpleNamespace(
            getpwnam=lambda name: verifier if name == qualification.VERIFIER_USER else runtime
        ),
    )
    monkeypatch.setitem(
        sys.modules, "grp", SimpleNamespace(getgrnam=lambda name: group, getgrall=lambda: [group])
    )
    monkeypatch.setattr(qualification, "os", SimpleNamespace(name="posix"))
    monkeypatch.setattr(qualification, "_immutable_file", Mock())
    monkeypatch.setattr(
        qualification, "_mode", lambda *_args, **_kw: SimpleNamespace(st_mode=stat.S_IFREG | 0o755)
    )
    # Tylko test granicy kompozycji; nie przypisuje PASS kwalifikacji z pominiętymi pinami.
    monkeypatch.setattr(qualification, "REVIEWED_CODE_DIGESTS", {})
    refusal = identity.ServiceIdentityError("EXECSTART_PROGRAM_OR_ARGUMENTS_CHANGED")
    qualify_identity = Mock(side_effect=refusal)
    monkeypatch.setattr(qualification, "qualify_freshness_service_identity", qualify_identity)
    connect = Mock(side_effect=AssertionError("Baza nie może zostać dotknięta po odmowie procesu"))
    monkeypatch.setattr(qualification.psycopg, "connect", connect)
    with pytest.raises(qualification.DeploymentQualificationError) as error:
        qualification.qualify_production_local_deployment()
    qualify_identity.assert_called_once_with(uid=UID, gid=GID)
    connect.assert_not_called()
    assert error.value.__cause__ is refusal


@pytest.fixture
def native_binary_metadata(monkeypatch):
    """Modeluj wyłącznie root-owned metadata, zachowując odczyt prawdziwego ELF."""
    changes = {}

    def observed(value):
        fields = {
            name: getattr(value, name)
            for name in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_uid",
                "st_gid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        }
        fields.update(st_uid=0, st_gid=0)
        fields.update(changes)
        return SimpleNamespace(**fields)

    proxy = SimpleNamespace(
        O_RDONLY=os.O_RDONLY,
        O_DIRECTORY=os.O_DIRECTORY,
        O_NOFOLLOW=os.O_NOFOLLOW,
        O_CLOEXEC=os.O_CLOEXEC,
        open=os.open,
        close=os.close,
        read=os.read,
        fsdecode=os.fsdecode,
        readlink=os.readlink,
        fstat=lambda fd: observed(os.fstat(fd)),
        stat=lambda *args, **kw: observed(os.stat(*args, **kw)),
    )
    monkeypatch.setattr(identity, "os", proxy)
    return changes


@pytest.fixture
def native_python():
    code = "import time; print('ready', flush=True); time.sleep(20)"
    child = subprocess.Popen(
        [identity.EXECUTABLE, "-I", "-s", "-B", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline() == b"ready\n"
        yield child, code
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


@LINUX
def test_native_proc_image_and_argv_with_modelled_root_ownership(
    native_python, native_binary_metadata
):
    child, code = native_python
    actual = identity._program_observation(child.pid)
    assert actual["argv"] == (identity.EXECUTABLE, "-I", "-s", "-B", "-c", code)
    assert (
        actual["exe_sha256"] == hashlib.sha256(Path(identity.EXECUTABLE).read_bytes()).hexdigest()
    )
    assert actual["exe_metadata"][0:2] == (
        os.stat(identity.EXECUTABLE).st_dev,
        os.stat(identity.EXECUTABLE).st_ino,
    )


@LINUX
@pytest.mark.parametrize(
    "changes",
    [
        {"st_uid": 1000},
        {"st_gid": 1000},
        {"st_mode": stat.S_IFREG | 0o775},
        {"st_mode": stat.S_IFREG | 0o644},
        {"st_mode": stat.S_IFDIR | 0o755},
        {"st_size": 0},
        {"st_size": 512 * 1024 * 1024 + 1},
    ],
)
def test_native_proc_refuses_unprotected_expected_program_metadata(
    native_python, native_binary_metadata, changes
):
    native_binary_metadata.update(changes)
    with pytest.raises(identity.ServiceIdentityError, match="EXPECTED_PYTHON_NOT_PROTECTED"):
        identity._program_observation(native_python[0].pid)


@LINUX
@pytest.mark.parametrize(
    "change_at,code",
    [
        (2, "PROCESS_PROGRAM_INODE_CHANGED"),
        (3, "PROCESS_PROGRAM_CHANGED_DURING_OBSERVATION"),
    ],
)
def test_native_executable_inode_replacement_refuses_before_or_after_read(
    native_python, native_binary_metadata, monkeypatch, change_at, code
):
    original = identity.os.fstat
    calls = 0

    def metadata(fd):
        nonlocal calls
        calls += 1
        value = original(fd)
        if calls == change_at:
            value.st_ino += 1
        return value

    monkeypatch.setattr(identity.os, "fstat", metadata)
    with pytest.raises(identity.ServiceIdentityError, match=code):
        identity._program_observation(native_python[0].pid)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_unsupported_platform_refuses_before_native_access(monkeypatch, platform):
    monkeypatch.setattr(identity, "sys", SimpleNamespace(platform=platform))
    with pytest.raises(identity.ServiceIdentityError, match="NATIVE_LINUX_PIDFD_REQUIRED"):
        identity.qualify_freshness_service_identity(uid=UID, gid=GID)


@LINUX
def test_native_pidfd_is_bound_to_live_child_and_refuses_after_exit():
    child = subprocess.Popen(
        [identity.EXECUTABLE, "-I", "-s", "-B", "-c", "import time; time.sleep(20)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    descriptor = os.pidfd_open(child.pid, 0)
    try:
        identity._check_pidfd(descriptor, child.pid)
        with pytest.raises(identity.ServiceIdentityError, match="MAIN_PIDFD_NOT_MATCHED"):
            identity._check_pidfd(descriptor, child.pid + 1)
        child.terminate()
        child.wait(timeout=5)
        with pytest.raises(
            identity.ServiceIdentityError, match="MAIN_PIDFD_NOT_MATCHED|SERVICE_PROCESS_EXITED"
        ):
            identity._check_pidfd(descriptor, child.pid)
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        os.close(descriptor)


@LINUX
def test_native_program_refuses_when_another_binary_is_approved(monkeypatch):
    child = subprocess.Popen(
        [
            identity.EXECUTABLE,
            "-I",
            "-s",
            "-B",
            "-c",
            "import time; print('ready', flush=True); time.sleep(20)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        assert child.stdout.readline() == b"ready\n"
        monkeypatch.setattr(identity, "EXECUTABLE", "/bin/sleep")
        with pytest.raises(identity.ServiceIdentityError, match="PROCESS_PROGRAM_CHANGED"):
            identity._program_observation(child.pid)
    finally:
        child.terminate()
        child.wait(timeout=5)
