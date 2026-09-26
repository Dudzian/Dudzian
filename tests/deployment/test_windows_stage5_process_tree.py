from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

import deployment.windows_process_tree as tree_module
import deployment.windows_process_tree_child as child_module
from deployment.windows_process_tree import (
    CHILD_EXIT_TIMEOUT,
    ChildJob,
    JOB_PREFIX,
    ProcessTreeError,
    job_security_attributes,
    job_name,
    reviewed_python_executable,
)
from deployment.windows_process_tree_qualification import (
    ProcessTreeQualificationError,
    qualify,
    qualify_absent,
)


class Handle:
    def __init__(self, events=None):
        self.events = events
        self.closed = False

    def Close(self):
        self.closed = True
        if self.events is not None:
            self.events.append("close")


class QualifierJobApi:
    JOB_OBJECT_QUERY = 1
    JobObjectExtendedLimitInformation = 2
    JobObjectBasicProcessIdList = 3
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self, pids=(20, 30), flags=0x2000, open_error=None):
        self.pids, self.flags, self.open_error = pids, flags, open_error
        self.open_calls = []

    def OpenJobObject(self, *args):
        self.open_calls.append(args)
        if self.open_error:
            raise self.open_error
        return Handle()

    def QueryInformationJobObject(self, _handle, info):
        if info == self.JobObjectExtendedLimitInformation:
            return {"BasicLimitInformation": {"LimitFlags": self.flags}}
        return {"ProcessIdList": list(self.pids)}


class NativeError(Exception):
    def __init__(self, winerror):
        self.winerror = winerror


class StartJobApi:
    JOB_OBJECT_QUERY = 0x0004
    JobObjectExtendedLimitInformation = 2
    JobObjectBasicProcessIdList = 3
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000

    def __init__(self, events, *, query_flag=0x2000, assignment_error=None):
        self.events = events
        self.query_flag = query_flag
        self.assignment_error = assignment_error
        self.handle = Handle(events)
        self.set_payload = None
        self.extended_queries = 0

    def CreateJobObject(self, security, name):
        self.events.append(("create", security, name))
        return self.handle

    def QueryInformationJobObject(self, _handle, info):
        if info == self.JobObjectExtendedLimitInformation:
            self.extended_queries += 1
            self.events.append("query_limits")
            flag = 0 if self.extended_queries == 1 else self.query_flag
            return {"BasicLimitInformation": {"LimitFlags": flag}}
        raise AssertionError("membership query uses patched bounded helper")

    def SetInformationJobObject(self, _handle, info, payload):
        self.events.append("set_limits")
        self.set_payload = (info, payload)

    def AssignProcessToJobObject(self, _handle, process):
        self.events.append(("assign", process.pid))
        if self.assignment_error:
            raise self.assignment_error


class StartApi:
    def __init__(self, events, last_error=0):
        self.events, self.last_error = events, last_error

    def GetLastError(self):
        return self.last_error

    def OpenProcess(self, _access, _inherit, pid):
        self.events.append(("open_process", pid))
        process = Handle(self.events)
        process.pid = pid
        return process


CON = SimpleNamespace(PROCESS_SET_QUOTA=1, PROCESS_TERMINATE=2, PROCESS_QUERY_LIMITED_INFORMATION=4)
REVIEWED_PYTHON = r"C:\hostedtoolcache\windows\Python\3.11.x\x64\python.exe"


class SecurityApi:
    ACL_REVISION = 2
    WinBuiltinAdministratorsSid = 26
    WinLocalSystemSid = 22

    def __init__(self):
        self.created_sids = []
        self.acl = SimpleNamespace(aces=[])
        self.acl.AddAccessAllowedAce = lambda revision, mask, sid: self.acl.aces.append(
            (revision, mask, sid)
        )
        self.descriptor = SimpleNamespace(dacl_call=None)
        self.descriptor.SetSecurityDescriptorDacl = lambda present, dacl, defaulted: setattr(
            self.descriptor, "dacl_call", (present, dacl, defaulted)
        )

    def CreateWellKnownSid(self, sid_type, domain):
        assert domain is None
        self.created_sids.append(sid_type)
        return f"SID:{sid_type}"

    def ACL(self):
        return self.acl

    def SECURITY_DESCRIPTOR(self):
        return self.descriptor


class PyWinTypesApi:
    def __init__(self):
        self.attributes = SimpleNamespace(SECURITY_DESCRIPTOR=None, bInheritHandle=None)

    def SECURITY_ATTRIBUTES(self):
        return self.attributes


def load_service_module(monkeypatch):
    class ServiceFramework:
        def __init__(self, _args):
            pass

    modules = {
        "servicemanager": SimpleNamespace(
            LogInfoMsg=lambda _message: None, LogErrorMsg=lambda _message: None
        ),
        "win32api": SimpleNamespace(),
        "win32con": SimpleNamespace(),
        "win32event": SimpleNamespace(INFINITE=-1),
        "win32job": SimpleNamespace(),
        "win32security": SimpleNamespace(),
        "win32service": SimpleNamespace(SERVICE_STOP_PENDING=3, SERVICE_AUTO_START=2),
        "win32serviceutil": SimpleNamespace(ServiceFramework=ServiceFramework),
        "pywintypes": SimpleNamespace(),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "_windows_test_service_startup_test", Path("deployment/windows_test_service.py")
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    original_os_name = os.name
    os.name = "nt"
    try:
        spec.loader.exec_module(module)
    finally:
        os.name = original_os_name
    return module


def write_marker(path: Path):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "service_pid": 10,
                "child_pid": 20,
                "grandchild_pid": 30,
                "job_name": job_name(10),
            }
        ),
        encoding="utf-8",
    )


def prepare_start(
    monkeypatch,
    tmp_path,
    *,
    last_error=0,
    query_flag=0x2000,
    assignment_error=None,
    memberships=((222,), (222, 333)),
    python_executable=REVIEWED_PYTHON,
):
    events = []
    api = StartApi(events, last_error)
    jobs = StartJobApi(events, query_flag=query_flag, assignment_error=assignment_error)
    security = SecurityApi()
    pywintypes = PyWinTypesApi()
    popen_calls = []

    class Popen:
        pid = 222

        def __init__(self, command, text, creationflags):
            self.command = command
            self.alive = True
            popen_calls.append((command, creationflags))
            events.append("popen_child")

        def terminate(self):
            events.append("terminate_child")
            self.alive = False

        def wait(self, timeout):
            events.append(("wait_child", timeout))
            self.alive = False
            return 0

        def kill(self):
            events.append("kill_child")
            self.alive = False

    calls = iter(memberships)

    def wait_exact(_job, expected, *, win32job, timeout=tree_module.TREE_TIMEOUT):
        observed = set(next(calls))
        events.append(("membership", observed))
        if observed != expected:
            raise ProcessTreeError("job PID set mismatch")

    monkeypatch.setattr(tree_module.subprocess, "Popen", Popen)
    monkeypatch.setattr(tree_module, "wait_for_exact_pids", wait_exact)
    monkeypatch.setattr(tree_module.os, "getpid", lambda: 111)
    child_job = ChildJob(
        tmp_path,
        python_executable=python_executable,
        win32api=api,
        win32con=CON,
        win32job=jobs,
        win32security=security,
        pywintypes=pywintypes,
    )
    release = child_job._release_gate
    monkeypatch.setattr(
        child_job, "_release_gate", lambda gate: (events.append("gate_release"), release(gate))[1]
    )
    monkeypatch.setattr(
        child_job, "_wait_for_grandchild", lambda gate: events.append("grandchild_created") or 333
    )
    return child_job, jobs, events, popen_calls


def test_job_name_depends_only_on_positive_service_pid():
    assert job_name(42) == JOB_PREFIX + "42"
    for invalid in (0, -1, True):
        with pytest.raises(ProcessTreeError):
            job_name(invalid)


def test_child_job_requires_explicit_interpreter(tmp_path):
    with pytest.raises(TypeError):
        ChildJob(
            tmp_path,
            win32api=object(),
            win32con=object(),
            win32job=object(),
            win32security=object(),
            pywintypes=object(),
        )
    with pytest.raises(ProcessTreeError, match="required"):
        ChildJob(
            tmp_path,
            python_executable="",
            win32api=object(),
            win32con=object(),
            win32job=object(),
            win32security=object(),
            pywintypes=object(),
        )


def test_job_security_is_explicit_non_inheritable_and_query_only():
    security = SecurityApi()
    pywintypes = PyWinTypesApi()
    jobs = SimpleNamespace(JOB_OBJECT_QUERY=0x0004)

    attributes = job_security_attributes(
        pywintypes=pywintypes, win32job=jobs, win32security=security
    )

    assert attributes is pywintypes.attributes
    assert attributes.bInheritHandle is False
    assert attributes.SECURITY_DESCRIPTOR is security.descriptor
    assert security.descriptor.dacl_call == (True, security.acl, False)
    assert security.created_sids == [
        security.WinBuiltinAdministratorsSid,
        security.WinLocalSystemSid,
    ]
    assert security.acl.aces == [
        (security.ACL_REVISION, jobs.JOB_OBJECT_QUERY, "SID:26"),
        (security.ACL_REVISION, jobs.JOB_OBJECT_QUERY, "SID:22"),
    ]
    forbidden = 0x0001 | 0x0002 | 0x0008 | 0x00040000 | 0x00080000
    assert all(
        mask == jobs.JOB_OBJECT_QUERY and mask & forbidden == 0
        for _revision, mask, _sid in security.acl.aces
    )


def test_existing_job_fails_closed_before_child(monkeypatch, tmp_path):
    child_job, jobs, _events, popen = prepare_start(monkeypatch, tmp_path, last_error=183)
    with pytest.raises(ProcessTreeError, match="already exists"):
        child_job.start()
    assert jobs.handle.closed
    assert popen == []
    create = next(event for event in _events if isinstance(event, tuple) and event[0] == "create")
    assert create[1] is not None
    assert not any("SetNamedSecurityInfo" in str(event) for event in _events)


def test_child_job_start_sets_exact_limit_and_enforces_reviewed_interpreter_order(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(sys, "executable", r"C:\Python311\pythonservice.exe")
    record = tmp_path / "windows-acceptance-ownership.json"
    record.write_text(
        json.dumps(
            {
                "ownership_phase": "SERVICE_PROVEN",
                "strict_create_result": "CREATED",
                "service_name": "CryptoHunterBackend",
                "service_identity": r"NT SERVICE\CryptoHunterBackend",
                "python_executable": REVIEWED_PYTHON,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == REVIEWED_PYTHON)
    reviewed = reviewed_python_executable(record)
    child_job, jobs, events, commands = prepare_start(
        monkeypatch, tmp_path, python_executable=reviewed
    )
    result = child_job.start()
    command, creationflags = commands[0]
    assert command[0] == REVIEWED_PYTHON
    assert command[0] != sys.executable
    assert creationflags == tree_module._DETACHED_PROCESS == 0x00000008
    assert creationflags & 0x01000000 == 0  # CREATE_BREAKAWAY_FROM_JOB
    info, payload = jobs.set_payload
    assert info == jobs.JobObjectExtendedLimitInformation
    assert payload["BasicLimitInformation"]["LimitFlags"] == jobs.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    assert ("assign", 222) in events
    assert all(event != ("assign", 111) for event in events)
    ordered = [
        "popen_child",
        ("open_process", 222),
        ("assign", 222),
        ("membership", {222}),
        "gate_release",
        "grandchild_created",
        ("membership", {222, 333}),
    ]
    assert [events.index(event) for event in ordered] == sorted(
        events.index(event) for event in ordered
    )
    assert (tmp_path / "process-tree.gate").read_text(encoding="ascii") == "assigned"
    assert result == json.loads((tmp_path / "process-tree.json").read_text(encoding="utf-8"))
    assert set(result) == {
        "schema_version",
        "service_pid",
        "child_pid",
        "grandchild_pid",
        "job_name",
    }
    assert "run_token" not in result


def test_query_back_wrong_flag_fails_before_child(monkeypatch, tmp_path):
    child_job, jobs, _events, popen = prepare_start(monkeypatch, tmp_path, query_flag=0x2001)
    with pytest.raises(ProcessTreeError, match="query-back"):
        child_job.start()
    assert jobs.handle.closed
    assert popen == []


def test_assignment_failure_does_not_release_gate(monkeypatch, tmp_path):
    child_job, jobs, events, commands = prepare_start(
        monkeypatch, tmp_path, assignment_error=NativeError(5)
    )
    with pytest.raises(NativeError):
        child_job.start()
    assert not (tmp_path / "process-tree.gate").exists()
    assert commands
    assert "terminate_child" in events
    assert ("wait_child", CHILD_EXIT_TIMEOUT) in events
    assert "kill_child" not in events
    assert jobs.handle.closed
    assert events.index("close") < events.index("terminate_child")
    assert child_job.child is not None and not child_job.child.alive
    assert not any((tmp_path / name).exists() for name in tree_module.TRANSIENT_NAMES)


def test_failure_after_assignment_uses_job_close_then_bounded_wait(monkeypatch, tmp_path):
    child_job, jobs, events, _ = prepare_start(monkeypatch, tmp_path, memberships=((999,),))
    with pytest.raises(ProcessTreeError, match="PID set"):
        child_job.start()
    assert jobs.handle.closed
    assert events.index("close") < events.index(("wait_child", CHILD_EXIT_TIMEOUT))
    assert "terminate_child" in events
    assert "gate_release" not in events
    assert child_job.child is not None and not child_job.child.alive


def test_failure_after_gate_uses_kill_on_close_without_manual_terminate(monkeypatch, tmp_path):
    child_job, jobs, events, _ = prepare_start(
        monkeypatch, tmp_path, memberships=((222,), (222, 333, 444))
    )
    with pytest.raises(ProcessTreeError, match="PID set"):
        child_job.start()
    assert "gate_release" in events and "grandchild_created" in events
    assert jobs.handle.closed
    assert "terminate_child" not in events and "kill_child" not in events
    assert events.index("close") < events.index(("wait_child", CHILD_EXIT_TIMEOUT))
    assert child_job.child is not None and not child_job.child.alive
    assert not any((tmp_path / name).exists() for name in tree_module.TRANSIENT_NAMES)


@pytest.mark.parametrize("memberships", [((999,),), ((222,), (222,)), ((222,), (222, 333, 444))])
def test_start_fails_on_non_exact_membership(monkeypatch, tmp_path, memberships):
    child_job, _jobs, _events, _ = prepare_start(monkeypatch, tmp_path, memberships=memberships)
    with pytest.raises(ProcessTreeError, match="PID set"):
        child_job.start()


@pytest.mark.parametrize("pids", [(20,), (30,), (20, 30, 40), (10, 20, 30)])
def test_read_only_qualifier_requires_exact_child_set(tmp_path, pids):
    path = tmp_path / "process-tree.json"
    write_marker(path)
    with pytest.raises(ProcessTreeQualificationError, match="PID set"):
        qualify(path, 10, win32job=QualifierJobApi(pids))


def test_read_only_qualifier_passes_exact_contract(tmp_path):
    path = tmp_path / "process-tree.json"
    write_marker(path)
    jobs = QualifierJobApi()
    assert qualify(path, 10, win32job=jobs) == {"WINDOWS_PROCESS_TREE": "PASS"}
    assert jobs.open_calls == [(jobs.JOB_OBJECT_QUERY, False, job_name(10))]
    source = Path("deployment/windows_process_tree_qualification.py").read_text(encoding="utf-8")
    assert "SetNamedSecurityInfo" not in source
    assert "SetSecurityDescriptorDacl" not in source


def test_read_only_qualifier_requires_exact_kill_limit(tmp_path):
    path = tmp_path / "process-tree.json"
    write_marker(path)
    with pytest.raises(ProcessTreeQualificationError, match="KILL_ON_JOB_CLOSE"):
        qualify(path, 10, win32job=QualifierJobApi(flags=0x2001))


def test_qualify_absent_distinguishes_not_found_from_access_denied():
    assert qualify_absent(10, win32job=QualifierJobApi(open_error=NativeError(2))) == {
        "JOB_ABSENT": "PASS"
    }
    with pytest.raises(ProcessTreeQualificationError, match="could not be proven"):
        qualify_absent(10, win32job=QualifierJobApi(open_error=NativeError(5)))


def test_wait_for_exact_pids_is_bounded(monkeypatch):
    clock = iter((0.0, 0.0, 0.2))
    monkeypatch.setattr(tree_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(tree_module.time, "sleep", lambda _: None)
    with pytest.raises(ProcessTreeError, match="PID set mismatch"):
        tree_module.wait_for_exact_pids(
            Handle(), {20}, win32job=QualifierJobApi(pids=()), timeout=0.1
        )


def test_reviewed_python_authority_and_no_fallback(monkeypatch, tmp_path):
    record = tmp_path / "windows-acceptance-ownership.json"
    record.write_text(
        json.dumps(
            {
                "ownership_phase": "SERVICE_PROVEN",
                "strict_create_result": "CREATED",
                "service_name": "CryptoHunterBackend",
                "service_identity": r"NT SERVICE\CryptoHunterBackend",
                "python_executable": REVIEWED_PYTHON,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == REVIEWED_PYTHON)
    assert reviewed_python_executable(record) == REVIEWED_PYTHON


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "malformed",
        "phase",
        "create",
        "service",
        "identity",
        "empty",
        "relative",
        "missing_file",
    ],
)
def test_invalid_reviewed_python_authority_fails_before_popen(monkeypatch, tmp_path, mutation):
    record_path = tmp_path / "windows-acceptance-ownership.json"
    value = {
        "ownership_phase": "SERVICE_PROVEN",
        "strict_create_result": "CREATED",
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "python_executable": REVIEWED_PYTHON,
    }
    if mutation == "malformed":
        record_path.write_text("{", encoding="utf-8")
    elif mutation != "missing":
        changes = {
            "phase": ("ownership_phase", "INTENT_CREATED"),
            "create": ("strict_create_result", "ALREADY_EXISTS"),
            "service": ("service_name", "Foreign"),
            "identity": ("service_identity", "SYSTEM"),
            "empty": ("python_executable", ""),
            "relative": ("python_executable", r"Python\python.exe"),
        }
        if mutation in changes:
            key, replacement = changes[mutation]
            value[key] = replacement
        record_path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(Path, "is_file", lambda _self: mutation != "missing_file")
    popen = []
    monkeypatch.setattr(tree_module.subprocess, "Popen", lambda *args, **kwargs: popen.append(args))
    with pytest.raises(ProcessTreeError):
        reviewed_python_executable(record_path)
    assert popen == []


def test_grandchild_uses_child_interpreter(monkeypatch, tmp_path):
    gate = tmp_path / "process-tree.gate"
    gate.write_text("assigned", encoding="ascii")
    commands = []

    class Grandchild:
        pid = 444

        def __init__(self, command, creationflags):
            commands.append((command, creationflags))

        def wait(self):
            return 0

    monkeypatch.setattr(child_module.subprocess, "Popen", Grandchild)
    monkeypatch.setattr(child_module.sys, "executable", REVIEWED_PYTHON)
    assert child_module.main(["child", str(gate)]) == 0
    assert commands == [
        (
            [REVIEWED_PYTHON, child_module.__file__, "--grandchild"],
            child_module._DETACHED_PROCESS,
        )
    ]
    assert child_module._DETACHED_PROCESS == 0x00000008
    assert child_module._DETACHED_PROCESS & 0x01000000 == 0  # CREATE_BREAKAWAY_FROM_JOB


def test_grandchild_is_not_spawned_before_gate(monkeypatch, tmp_path):
    gate = tmp_path / "process-tree.gate"
    popen_calls = []
    clock = iter((0.0, 0.0, child_module.TIMEOUT + 1.0))
    monkeypatch.setattr(child_module.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(child_module.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(
        child_module.subprocess,
        "Popen",
        lambda *args, **kwargs: popen_calls.append((args, kwargs)),
    )

    assert child_module.main(["child", str(gate)]) == 2
    assert popen_calls == []


@pytest.mark.parametrize("failure_point", ["reviewed_python_executable", "ChildJob.start"])
def test_service_logs_process_tree_start_failure_and_propagates(
    monkeypatch, tmp_path, failure_point
):
    service_module = load_service_module(monkeypatch)
    messages = []

    class Logger:
        def info(self, *_args):
            pass

        def exception(self, message, *args):
            messages.append(message % args)

    class FailingJob:
        def __init__(self, *_args, **_kwargs):
            pass

        def start(self):
            raise ValueError("child start exploded")

        def close(self):
            pass

    service = service_module.CryptoHunterBackendTestService.__new__(
        service_module.CryptoHunterBackendTestService
    )
    service.marker = tmp_path / "scm-health.txt"
    service.machine_root = tmp_path
    service.ownership_record = tmp_path / "windows-acceptance-ownership.json"
    service.tree = None
    service.logger = None
    service.stop_event = object()
    monkeypatch.setattr(service_module, "configure_service_logger", lambda _path: Logger())
    monkeypatch.setattr(service_module, "close_service_logger", lambda _logger: None)
    monkeypatch.setattr(service_module, "ChildJob", FailingJob)
    if failure_point == "reviewed_python_executable":
        monkeypatch.setattr(
            service_module,
            "reviewed_python_executable",
            lambda _record: (_ for _ in ()).throw(RuntimeError("authority exploded")),
        )
        expected_type, expected_message = "RuntimeError", "authority exploded"
    else:
        monkeypatch.setattr(
            service_module, "reviewed_python_executable", lambda _record: REVIEWED_PYTHON
        )
        expected_type, expected_message = "ValueError", "child start exploded"

    with pytest.raises(Exception, match=expected_message):
        service.SvcDoRun()

    diagnostic = "\n".join(messages)
    assert "PROCESS_TREE_START_FAILURE" in diagnostic
    assert f"exception_type={expected_type}" in diagnostic
    assert f"exception_message={expected_message}" in diagnostic
    assert "service_pid=" in diagnostic
    assert "run_token" not in diagnostic
    assert not service.marker.exists()


def test_service_logger_configuration_failure_uses_event_log_and_propagates(monkeypatch, tmp_path):
    service_module = load_service_module(monkeypatch)
    event_messages = []
    monkeypatch.setattr(service_module.servicemanager, "LogErrorMsg", event_messages.append)
    monkeypatch.setattr(
        service_module,
        "configure_service_logger",
        lambda _path: (_ for _ in ()).throw(OSError("persistent logger exploded")),
    )
    service = service_module.CryptoHunterBackendTestService.__new__(
        service_module.CryptoHunterBackendTestService
    )
    service.marker = tmp_path / "scm-health.txt"
    service.machine_root = tmp_path
    service.ownership_record = tmp_path / "windows-acceptance-ownership.json"
    service.tree = None
    service.logger = None
    service.stop_event = object()

    with pytest.raises(OSError, match="persistent logger exploded"):
        service.SvcDoRun()

    assert len(event_messages) == 1
    diagnostic = event_messages[0]
    assert "PROCESS_TREE_START_FAILURE" in diagnostic
    assert "service_name=CryptoHunterBackend" in diagnostic
    assert "exception_type=OSError" in diagnostic
    assert "exception_message=persistent logger exploded" in diagnostic
    assert "service_pid=" in diagnostic
    assert "run_token" not in diagnostic
    assert not service.marker.exists()


def test_powershell_lifecycle_contract_is_specific_and_bounded():
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    assert 'Get-CimInstance Win32_Process -Filter "ProcessId=$($Tree.child_pid)"' in source
    assert "[int]$child.ParentProcessId -ne [int]$Tree.service_pid" in source
    assert "[int]$grandchild.ParentProcessId -ne [int]$Tree.child_pid" in source
    body = source[source.index("function Wait-TreeGone") : source.index("function Assert-LogEvent")]
    assert "for ($i = 0; $i -lt 60; $i++)" in body
    assert 'Wait-TreeGone $tree1 "GRACEFUL_STOP"' in source
    assert 'Wait-TreeGone $crashedTree "CRASH_RESTART"' in source
    assert 'Wait-TreeGone $tree3 "POST_RECOVERY_GRACEFUL_STOP"' in source
    assert "Stop-Process -Id $Tree.child_pid" not in source
    assert "Stop-Process -Id $Tree.grandchild_pid" not in source
    final_stop = source.index('Wait-TreeGone $tree3 "POST_RECOVERY_GRACEFUL_STOP"')
    for result in (
        "WINDOWS_PROCESS_TREE",
        "WINDOWS_NO_ORPHAN_CHILDREN",
        "WINDOWS_PERSISTENT_LOGGING",
    ):
        assert source.index(f'$result.{result} = "PASS"') > final_stop
    cleanup = source[
        source.index("function Remove-OwnedProcessTreeArtifacts") : source.index(
            "function Read-MarkerPid"
        )
    ]
    assert "Get-Item -LiteralPath $target -Force" in cleanup
    assert "$item.PSIsContainer" in cleanup
    assert "[IO.FileAttributes]::ReparsePoint" in cleanup
    assert cleanup.index("unsafe process-tree transient") < cleanup.index(
        "Remove-Item -LiteralPath $target"
    )
    assert "Remove-Item -LiteralPath $runtime" not in source


def test_start_timeout_diagnostic_is_safe_bounded_and_precedes_cleanup():
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    diagnostic = source[
        source.index("function Get-ProcessTreeStartupDiagnostic") : source.index(
            "function Assert-Tree"
        )
    ]
    assert "Win32_Service" in diagnostic
    assert "health_marker=" in diagnostic and "health_pid=" in diagnostic
    assert "process_tree_json=" in diagnostic and "process_tree_gate=" in diagnostic
    for name in ("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.log.4"):
        assert f'"{name}"' in diagnostic
    assert "Select-String" in diagnostic and '"PROCESS_TREE_START_FAILURE"' in diagnostic
    assert "ReparsePoint" in diagnostic and "$item.PSIsContainer" in diagnostic
    assert "Get-WinEvent -FilterHashtable $eventQuery -MaxEvents 128" in diagnostic
    assert 'LogName = "Application"; StartTime = $SinceUtc' in diagnostic
    assert "service_name=CryptoHunterBackend" in diagnostic
    assert '"service_pid=$ServicePid(?:\\s|$)"' in diagnostic
    assert "Where-Object { $_ -match 'PROCESS_TREE_START_FAILURE' }" in diagnostic
    assert "Select-Object -First 1" in diagnostic
    assert "$messageLine.Count -ne 1" in diagnostic
    assert "event=$messageLine" in diagnostic
    assert "$event.Message" in diagnostic
    assert "$event |" not in diagnostic
    assert "ownershipPath" not in diagnostic
    assert "WindowsAcceptanceRunToken" not in diagnostic
    catch = source.index('($stage -ceq "PROCESS_TREE_START"')
    cleanup = source.index("} finally {", catch)
    capture = source.index("foreach ($line in @(", catch)
    lookup = source.index("Get-ProcessTreeStartupDiagnostic $startupServicePid", capture)
    assert catch < capture < lookup < source.index("Write-Output $line", capture) < cleanup
    assert '$primaryFailure = "[$stage] $($_.Exception.Message)"' in source[:cleanup]
    assert "$diagnosticFailures.Add" in source[catch:cleanup]
    report = source[source.index("$reportedFailures =") :]
    assert report.index("$primaryFailure") < report.index("$startupDiagnostics")
    assert report.index("$primaryFailure") < report.index("$diagnosticFailures")


def test_event_log_lookup_failure_is_secondary_to_process_tree_primary():
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    diagnostic = source[
        source.index("function Get-ProcessTreeStartupDiagnostic") : source.index(
            "function Assert-Tree"
        )
    ]
    event_lookup = diagnostic.index("Get-WinEvent")
    event_failure = diagnostic.index("$script:diagnosticFailures.Add", event_lookup)
    assert event_lookup < event_failure
    assert "throw" not in diagnostic[event_failure:]
    report = source[source.index("$reportedFailures =") :]
    assert report.index("$primaryFailure") < report.index("$diagnosticFailures")
