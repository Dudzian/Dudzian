from __future__ import annotations

import json
import sys
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace

import pytest

from deployment.platforms.windows import WindowsDeploymentNotQualified, static_path_layout
from deployment.windows_dacl_qualification import (
    ACE_FLAGS,
    ADMINISTRATORS_SID,
    FILE_ALL_ACCESS,
    FILE_GENERIC_EXECUTE,
    FILE_GENERIC_READ,
    FILE_GENERIC_WRITE,
    DELETE,
    SYSTEM_SID,
    WindowsDaclQualificationError,
    expected_aces,
    is_safe_descendant,
    qualify_directory_dacl,
    qualify_record,
)
import deployment.windows_dacl_provision as provisioner
import deployment.windows_dacl_qualification as qualifier


class Api:
    @staticmethod
    def GetFullPathName(value: str) -> str:
        return value


class Dacl:
    def __init__(self, aces):
        self.aces = aces

    def GetAceCount(self):
        return len(self.aces)

    def GetAce(self, index):
        return self.aces[index]


class Descriptor:
    def __init__(self, owner, dacl, control):
        self.owner, self.dacl, self.control = owner, dacl, control

    def GetSecurityDescriptorOwner(self):
        return self.owner

    def GetSecurityDescriptorDacl(self):
        return self.dacl

    def GetSecurityDescriptorControl(self):
        return self.control, 1


def security(descriptor):
    return SimpleNamespace(
        OWNER_SECURITY_INFORMATION=1,
        DACL_SECURITY_INFORMATION=2,
        SE_FILE_OBJECT=1,
        SE_DACL_PRESENT=4,
        SE_DACL_PROTECTED=8,
        ACCESS_ALLOWED_ACE_TYPE=0,
        INHERITED_ACE=16,
        GetNamedSecurityInfo=lambda *_: descriptor,
        ConvertSidToStringSid=lambda sid: sid,
    )


def exact_descriptor(role="CONFIG", service_sid="S-1-5-80-123"):
    service_mask = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
    if role != "CONFIG":
        service_mask |= FILE_GENERIC_WRITE | DELETE
    aces = [
        ((0, ACE_FLAGS), FILE_ALL_ACCESS, ADMINISTRATORS_SID),
        ((0, ACE_FLAGS), FILE_ALL_ACCESS, SYSTEM_SID),
        ((0, ACE_FLAGS), service_mask, service_sid),
    ]
    return Descriptor(ADMINISTRATORS_SID, Dacl(aces), 4 | 8)


def test_frozen_layout_and_separator_safe_descendants():
    assert static_path_layout(r"D:\Apps", r"E:\Data", r"F:\Users\u\Local") == (
        PureWindowsPath(r"D:\Apps\CryptoHunter"),
        PureWindowsPath(r"E:\Data\CryptoHunter\State"),
        PureWindowsPath(r"E:\Data\CryptoHunter\Config"),
        PureWindowsPath(r"E:\Data\CryptoHunter\Logs"),
        PureWindowsPath(r"E:\Data\CryptoHunter\Runtime"),
        PureWindowsPath(r"E:\Data\CryptoHunter\Updates"),
        PureWindowsPath(r"F:\Users\u\Local\CryptoHunter"),
    )
    root = r"C:\ProgramData\CryptoHunter"
    assert is_safe_descendant(root + r"\State", root, Api)
    assert not is_safe_descendant(root + "-Evil", root, Api)


@pytest.mark.parametrize("role", ["CONFIG", "STATE", "RUNTIME"])
def test_exact_dacl_passes(role):
    sid = "S-1-5-80-123"
    qualify_directory_dacl(
        "ignored", role, sid, win32security=security(exact_descriptor(role, sid))
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "unprotected",
        "null",
        "unknown",
        "inherited",
        "wrong_owner",
        "service_write",
        "write_dac",
        "write_owner",
        "all_access",
        "everyone",
        "users",
        "authenticated_users",
    ],
)
def test_non_exact_config_dacl_fails_closed(mutation):
    sid = "S-1-5-80-123"
    descriptor = exact_descriptor("CONFIG", sid)
    if mutation == "unprotected":
        descriptor.control = 4
    elif mutation == "null":
        descriptor.dacl = None
    elif mutation == "unknown":
        descriptor.dacl.aces[2] = ((1, ACE_FLAGS), FILE_GENERIC_READ, sid)
    elif mutation == "inherited":
        descriptor.dacl.aces[2] = ((0, ACE_FLAGS | 16), FILE_GENERIC_READ, sid)
    elif mutation == "wrong_owner":
        descriptor.owner = SYSTEM_SID
    elif mutation == "service_write":
        descriptor.dacl.aces[2] = ((0, ACE_FLAGS), FILE_GENERIC_READ | FILE_GENERIC_WRITE, sid)
    elif mutation == "write_dac":
        descriptor.dacl.aces[2] = ((0, ACE_FLAGS), FILE_GENERIC_READ | 0x40000, sid)
    elif mutation == "write_owner":
        descriptor.dacl.aces[2] = ((0, ACE_FLAGS), FILE_GENERIC_READ | 0x80000, sid)
    elif mutation == "all_access":
        descriptor.dacl.aces[2] = ((0, ACE_FLAGS), FILE_ALL_ACCESS, sid)
    else:
        bad = {"everyone": "S-1-1-0", "users": "S-1-5-32-545", "authenticated_users": "S-1-5-11"}[
            mutation
        ]
        descriptor.dacl.aces.append(((0, ACE_FLAGS), FILE_GENERIC_READ, bad))
    with pytest.raises(WindowsDaclQualificationError):
        qualify_directory_dacl("ignored", "CONFIG", sid, win32security=security(descriptor))


def test_qualifier_source_has_no_mutation_or_directory_creation():
    import inspect
    import deployment.windows_dacl_qualification as module

    source = inspect.getsource(module)
    for mutation in (
        "windows_dacl_provision",
        "SetNamedSecurityInfo",
        "SetFileSecurity",
        "AddAccessAllowedAce",
        ".mkdir(",
        ".unlink(",
        ".rmdir(",
        ".write_text(",
        ".replace(",
        ".chmod(",
        "icacls",
    ):
        assert mutation not in source


def valid_record(**updates):
    record = {
        "run_token": "token",
        "ownership_phase": "SERVICE_PROVEN",
        "strict_create_result": "CREATED",
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "service_sid": "S-1-5-80-123",
    }
    record.update(updates)
    return record


class ReadOnlySecurity:
    OWNER_SECURITY_INFORMATION = 1
    DACL_SECURITY_INFORMATION = 2
    SE_FILE_OBJECT = 1
    SE_DACL_PRESENT = 4
    SE_DACL_PROTECTED = 8
    ACCESS_ALLOWED_ACE_TYPE = 0
    INHERITED_ACE = 16

    def __init__(self, descriptors):
        self.descriptors = descriptors

    def LookupAccountName(self, system, identity):
        return "S-1-5-80-123", None, None

    def ConvertSidToStringSid(self, sid):
        return sid

    def GetNamedSecurityInfo(self, path, *_):
        return self.descriptors[str(path)]


def test_read_only_record_qualifier_passes_exact_authority_and_dacls(monkeypatch):
    paths = SimpleNamespace(configuration="Config", state="State", runtime="Runtime")
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: paths)
    security_api = ReadOnlySecurity(
        {
            "Config": exact_descriptor("CONFIG"),
            "State": exact_descriptor("STATE"),
            "Runtime": exact_descriptor("RUNTIME"),
        }
    )
    assert qualify_record(
        valid_record(),
        run_token="token",
        win32api=Api,
        win32file=FileApi(),
        win32security=security_api,
    ) == {
        "NATIVE_PATHS": "PASS",
        "CONFIG_DACL": "PASS",
        "STATE_DACL": "PASS",
        "RUNTIME_DACL": "PASS",
        "READ_ONLY_QUALIFIER": "PASS",
    }


@pytest.mark.parametrize(
    "updates,token",
    [
        ({}, "wrong"),
        ({"ownership_phase": "INTENT_CREATED"}, "token"),
        ({"strict_create_result": "ALREADY_EXISTS"}, "token"),
    ],
)
def test_read_only_record_qualifier_rejects_invalid_ownership(monkeypatch, updates, token):
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: pytest.fail("path read"))
    with pytest.raises(WindowsDaclQualificationError):
        qualify_record(
            valid_record(**updates),
            run_token=token,
            win32api=Api,
            win32file=FileApi(),
            win32security=ReadOnlySecurity({}),
        )


def test_read_only_record_qualifier_rejects_live_sid_mismatch(monkeypatch):
    api = ReadOnlySecurity({})
    api.LookupAccountName = lambda *_: ("S-1-5-80-wrong", None, None)
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: pytest.fail("path read"))
    with pytest.raises(WindowsDaclQualificationError, match="SID mismatches"):
        qualify_record(
            valid_record(), run_token="token", win32api=Api, win32file=FileApi(), win32security=api
        )


@pytest.mark.parametrize("broken_role", ["CONFIG", "STATE", "RUNTIME"])
def test_read_only_record_qualifier_rejects_each_dacl_mismatch(monkeypatch, broken_role):
    paths = SimpleNamespace(configuration="CONFIG", state="STATE", runtime="RUNTIME")
    descriptors = {role: exact_descriptor(role) for role in ("CONFIG", "STATE", "RUNTIME")}
    descriptors[broken_role].owner = SYSTEM_SID
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: paths)
    with pytest.raises(WindowsDaclQualificationError, match="owner SID"):
        qualify_record(
            valid_record(),
            run_token="token",
            win32api=Api,
            win32file=FileApi(),
            win32security=ReadOnlySecurity(descriptors),
        )


def test_provisioner_has_no_qualification_pass_authority():
    import inspect

    source = inspect.getsource(provisioner.provision)
    assert "qualify_stage4" not in source
    assert "WINDOWS_ACL_QUALIFICATION" not in source
    assert "CONFIG_PROVISIONED" in source


def test_scm_probe_separates_provision_from_read_only_proof():
    probe = (Path(__file__).resolve().parents[2] / "deployment/windows_scm_probe.ps1").read_text()
    provision = probe.index("-m deployment.windows_dacl_provision provision")
    qualification = probe.index("-m deployment.windows_dacl_qualification")
    pass_assignment = probe.index('$result.WINDOWS_ACL_QUALIFICATION = "PASS"')
    assert provision < qualification < pass_assignment


def test_public_platform_acl_boundary_executes_dacl_qualification(monkeypatch):
    monkeypatch.setattr(qualifier.os, "name", "nt")
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: "paths")
    called = {}
    monkeypatch.setattr(
        qualifier,
        "qualify_stage4",
        lambda paths, sid, **_: called.update(paths=paths, sid=sid) or {"CONFIG_DACL": "PASS"},
    )
    monkeypatch.setitem(__import__("sys").modules, "win32api", SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "win32file", SimpleNamespace())
    monkeypatch.setitem(__import__("sys").modules, "win32security", SimpleNamespace())
    from deployment.platforms.windows import qualify_acl

    assert qualify_acl("S-1-5-80-123") == {"CONFIG_DACL": "PASS"}
    assert called == {"paths": "paths", "sid": "S-1-5-80-123"}


class FileApi:
    FILE_ATTRIBUTE_REPARSE_POINT = 0x400

    def __init__(self, reparse: Path | None = None):
        self.reparse = reparse

    def GetFileAttributes(self, value):
        return self.FILE_ATTRIBUTE_REPARSE_POINT if Path(value) == self.reparse else 0


class RecordingSecurity:
    ACL_REVISION = 2
    ACL_REVISION_DS = 4
    OWNER_SECURITY_INFORMATION = 1
    DACL_SECURITY_INFORMATION = 2
    PROTECTED_DACL_SECURITY_INFORMATION = 4
    SE_FILE_OBJECT = 1

    def __init__(self):
        self.dacls = []
        self.set_calls = []

    def LookupAccountName(self, system, identity):
        return "S-1-5-80-123", None, None

    def ConvertSidToStringSid(self, sid):
        return sid

    def ConvertStringSidToSid(self, sid):
        return sid

    def ACL(self):
        calls = []
        self.dacls.append(calls)

        class RecordingAcl:
            @staticmethod
            def AddAccessAllowedAceEx(revision, flags, mask, sid):
                calls.append(SimpleNamespace(revision=revision, flags=flags, mask=mask, sid=sid))

        return RecordingAcl()

    def SetNamedSecurityInfo(self, path, object_type, info, owner, group, dacl, sacl):
        self.set_calls.append((path, object_type, info, owner, group, dacl, sacl))


class NativeTestError(Exception):
    def __init__(self, message="native token failure", *, winerror=5, funcname="FakeNative"):
        super().__init__(message)
        self.winerror = winerror
        self.funcname = funcname


class FailingSecurity(RecordingSecurity):
    def __init__(self, failure, exception_type=NativeTestError):
        super().__init__()
        self.failure = failure
        self.exception_type = exception_type
        self.operations = []

    def _call(self, operation):
        self.operations.append(operation)
        if self.failure == operation:
            raise self.exception_type()

    def LookupAccountName(self, system, identity):
        self._call("LookupAccountName")
        return super().LookupAccountName(system, identity)

    def ConvertSidToStringSid(self, sid):
        self._call("ConvertSidToStringSid")
        return super().ConvertSidToStringSid(sid)

    def ConvertStringSidToSid(self, sid):
        operation = f"ConvertStringSidToSid:{sid}"
        self._call(operation)
        return super().ConvertStringSidToSid(sid)

    def ACL(self):
        self._call("ACL")
        parent = self
        acl = super().ACL()
        original = acl.AddAccessAllowedAceEx

        def add(revision, flags, mask, sid):
            parent._call("AddAccessAllowedAceEx")
            return original(revision, flags, mask, sid)

        acl.AddAccessAllowedAceEx = add
        return acl

    def SetNamedSecurityInfo(self, path, object_type, info, owner, group, dacl, sacl):
        self._call("SetNamedSecurityInfo")
        return super().SetNamedSecurityInfo(path, object_type, info, owner, group, dacl, sacl)


def provision_fixture(tmp_path, monkeypatch, security_api):
    targets = SimpleNamespace(
        configuration=tmp_path / "Config",
        state=tmp_path / "State",
        runtime=tmp_path / "Runtime",
    )
    monkeypatch.setattr(provisioner, "qualify_native_paths", lambda **_: targets)
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(valid_record()), encoding="utf-8")
    return record_path, targets


def test_provisioner_uses_ds_revision_and_preserves_exact_aces(tmp_path, monkeypatch):
    targets = SimpleNamespace(
        configuration=tmp_path / "Config",
        state=tmp_path / "State",
        runtime=tmp_path / "Runtime",
    )
    monkeypatch.setattr(provisioner, "qualify_native_paths", lambda **_: targets)
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(valid_record()), encoding="utf-8")
    security_api = RecordingSecurity()

    provisioner.provision(
        record_path,
        "token",
        win32api=Api,
        win32file=FileApi(),
        win32security=security_api,
    )

    assert len(security_api.dacls) == 3
    recorded_ace_calls = [call for dacl in security_api.dacls for call in dacl]
    assert len(recorded_ace_calls) == 9
    assert all(call.revision == security_api.ACL_REVISION_DS for call in recorded_ace_calls)
    assert all(call.revision != security_api.ACL_REVISION for call in recorded_ace_calls)
    for role, calls in zip(("CONFIG", "STATE", "RUNTIME"), security_api.dacls, strict=True):
        assert {(call.sid, call.mask, call.flags) for call in calls} == expected_aces(
            role, "S-1-5-80-123"
        )


@pytest.mark.parametrize(
    "failure,diagnostic_operation",
    [
        ("LookupAccountName", "LookupAccountName"),
        ("ConvertSidToStringSid", "ConvertSidToStringSid"),
        (f"ConvertStringSidToSid:{ADMINISTRATORS_SID}", "ConvertStringSidToSid"),
        (f"ConvertStringSidToSid:{SYSTEM_SID}", "ConvertStringSidToSid"),
        ("ConvertStringSidToSid:S-1-5-80-123", "ConvertStringSidToSid"),
        ("ACL", "ACL"),
        ("AddAccessAllowedAceEx", "AddAccessAllowedAceEx"),
        ("SetNamedSecurityInfo", "SetNamedSecurityInfo"),
    ],
)
def test_each_native_failure_is_normalized_and_stops_security_calls(
    tmp_path, monkeypatch, failure, diagnostic_operation
):
    security_api = FailingSecurity(failure)
    record_path, _ = provision_fixture(tmp_path, monkeypatch, security_api)

    with pytest.raises(provisioner.WindowsDaclProvisionError) as caught:
        provisioner.provision(
            record_path, "token", win32api=Api, win32file=FileApi(), win32security=security_api
        )

    diagnostic = str(caught.value)
    assert diagnostic.startswith(f"DACL_NATIVE_FAILURE operation={diagnostic_operation}")
    assert "error_type=NativeTestError" in diagnostic
    assert "winerror=5" in diagnostic
    assert "funcname=FakeNative" in diagnostic
    assert "token" not in diagnostic
    assert security_api.operations[-1] == failure


def test_add_ace_type_error_is_controlled_and_reports_contract_arguments(tmp_path, monkeypatch):
    security_api = FailingSecurity("AddAccessAllowedAceEx", TypeError)
    record_path, _ = provision_fixture(tmp_path, monkeypatch, security_api)

    with pytest.raises(provisioner.WindowsDaclProvisionError) as caught:
        provisioner.provision(
            record_path, "token", win32api=Api, win32file=FileApi(), win32security=security_api
        )

    diagnostic = str(caught.value)
    assert "operation=AddAccessAllowedAceEx" in diagnostic
    assert "role=CONFIG" in diagnostic
    assert any(
        f"principal_sid={sid}" in diagnostic
        for sid in (ADMINISTRATORS_SID, SYSTEM_SID, "S-1-5-80-123")
    )
    assert f"revision={security_api.ACL_REVISION_DS}" in diagnostic
    assert f"flags={ACE_FLAGS}" in diagnostic
    assert "mask=" in diagnostic
    assert "error_type=TypeError" in diagnostic
    assert "token" not in diagnostic
    assert security_api.operations[-1] == "AddAccessAllowedAceEx"


def test_cli_normalizes_pywintypes_style_error_without_traceback(tmp_path, monkeypatch, capsys):
    class PywinError(NativeTestError):
        pass

    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(valid_record()), encoding="utf-8")
    monkeypatch.setattr(provisioner.os, "name", "nt")
    monkeypatch.setitem(sys.modules, "win32api", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32file", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "win32security", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pywintypes", SimpleNamespace(error=PywinError))

    def fail(*args, **kwargs):
        return provisioner._native_call(
            "LookupAccountName", lambda: (_ for _ in ()).throw(PywinError()), secret="token"
        )

    monkeypatch.setattr(provisioner, "provision", fail)
    assert (
        provisioner.main(["provision", "--record", str(record_path), "--run-token", "token"]) == 1
    )
    output = capsys.readouterr()
    assert output.err == ""
    assert "Traceback" not in output.out
    assert "DACL_NATIVE_FAILURE operation=LookupAccountName" in output.out
    assert "error_type=PywinError" in output.out
    assert "token" not in output.out


def cleanup_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    targets = SimpleNamespace(
        configuration=tmp_path / "Config",
        state=tmp_path / "State",
        runtime=tmp_path / "Runtime",
    )
    monkeypatch.setattr(provisioner, "qualify_native_paths", lambda **_: targets)
    sid = "S-1-5-80-123"
    record = {
        "run_token": "token",
        "ownership_phase": "SERVICE_PROVEN",
        "service_sid": sid,
        "path_security_plan": provisioner._plan(targets, sid, Api),
    }
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    for item in record["path_security_plan"]["targets"]:
        target = Path(item["path"])
        target.mkdir()
        sentinel = {
            "run_token": "token",
            "canonical_path": item["path"],
            "service_sid": sid,
            "role": item["role"],
        }
        (target / provisioner.SENTINEL).write_text(json.dumps(sentinel), encoding="utf-8")
    return record_path, record


def test_cleanup_exact_sentinels_is_allowed(tmp_path, monkeypatch):
    record_path, record = cleanup_fixture(tmp_path, monkeypatch)
    provisioner.cleanup(record_path, "token", win32api=Api, win32file=FileApi())
    assert all(not Path(item["path"]).exists() for item in record["path_security_plan"]["targets"])
    assert "path_security_plan" not in json.loads(record_path.read_text())


@pytest.mark.parametrize("failure", ["token", "foreign", "reparse"])
def test_cleanup_mismatch_preserves_targets_and_fails(tmp_path, monkeypatch, failure):
    record_path, record = cleanup_fixture(tmp_path, monkeypatch)
    first = Path(record["path_security_plan"]["targets"][0]["path"])
    file_api = FileApi(first if failure == "reparse" else None)
    if failure == "token":
        sentinel = first / provisioner.SENTINEL
        value = json.loads(sentinel.read_text())
        value["run_token"] = "foreign"
        sentinel.write_text(json.dumps(value))
    elif failure == "foreign":
        (first / "foreign.txt").write_text("do not delete")
    with pytest.raises(provisioner.WindowsDaclProvisionError):
        provisioner.cleanup(record_path, "token", win32api=Api, win32file=file_api)
    assert first.exists()
