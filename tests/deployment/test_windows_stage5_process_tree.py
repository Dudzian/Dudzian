from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import deployment.windows_process_tree as tree_module
from deployment.windows_process_tree import ChildJob, JOB_PREFIX, ProcessTreeError, job_name
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

    def OpenJobObject(self, *_):
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

    def CreateJobObject(self, _security, name):
        self.events.append(("create", name))
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
):
    events = []
    api = StartApi(events, last_error)
    jobs = StartJobApi(events, query_flag=query_flag, assignment_error=assignment_error)
    popen_calls = []

    class Popen:
        pid = 222

        def __init__(self, command, text):
            popen_calls.append(command)
            events.append("popen_child")

    calls = iter(memberships)

    def wait_exact(_job, expected, *, win32job, timeout=tree_module.TREE_TIMEOUT):
        observed = set(next(calls))
        events.append(("membership", observed))
        if observed != expected:
            raise ProcessTreeError("job PID set mismatch")

    monkeypatch.setattr(tree_module.subprocess, "Popen", Popen)
    monkeypatch.setattr(tree_module, "wait_for_exact_pids", wait_exact)
    monkeypatch.setattr(tree_module.os, "getpid", lambda: 111)
    child_job = ChildJob(tmp_path, win32api=api, win32con=CON, win32job=jobs)
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


def test_existing_job_fails_closed_before_child(monkeypatch, tmp_path):
    child_job, jobs, _events, popen = prepare_start(monkeypatch, tmp_path, last_error=183)
    with pytest.raises(ProcessTreeError, match="already exists"):
        child_job.start()
    assert jobs.handle.closed
    assert popen == []


def test_child_job_start_sets_exact_limit_and_enforces_order(monkeypatch, tmp_path):
    child_job, jobs, events, _ = prepare_start(monkeypatch, tmp_path)
    result = child_job.start()
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
    child_job, _jobs, _events, _ = prepare_start(
        monkeypatch, tmp_path, assignment_error=NativeError(5)
    )
    with pytest.raises(NativeError):
        child_job.start()
    assert not (tmp_path / "process-tree.gate").exists()


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
    assert qualify(path, 10, win32job=QualifierJobApi()) == {"WINDOWS_PROCESS_TREE": "PASS"}


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
