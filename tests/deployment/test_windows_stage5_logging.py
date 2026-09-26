from __future__ import annotations

import inspect
import json
import logging
import shutil
import subprocess
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import SimpleNamespace

import pytest

import deployment.windows_logging_provision as provisioner
import deployment.windows_logging_qualification as qualifier
from deployment.windows_dacl_qualification import (
    ACE_FLAGS,
    ADMINISTRATORS_SID,
    DELETE,
    FILE_ALL_ACCESS,
    FILE_GENERIC_EXECUTE,
    FILE_GENERIC_READ,
    FILE_GENERIC_WRITE,
    SYSTEM_SID,
    WindowsDaclQualificationError,
)
from deployment.windows_logging_provision import (
    SENTINEL,
    WindowsLoggingProvisionError,
    cleanup,
    expected_logging_aces,
    provision,
)
from deployment.windows_logging_qualification import (
    WindowsLoggingQualificationError,
    qualify_record,
)
from deployment.windows_persistent_logging import (
    BACKUP_COUNT,
    LOG_FILE_NAME,
    LOG_FORMAT,
    MAX_BYTES,
    close_service_logger,
    configure_service_logger,
    create_handler,
)

SERVICE_SID = "S-1-5-80-123"
TOKEN = "a" * 64


class Api:
    @staticmethod
    def GetFullPathName(value):
        return str(value)


class FileApi:
    def __init__(self, reparses=()):
        self.reparses = {str(path) for path in reparses}

    def GetFileAttributes(self, path):
        return 0x400 if str(path) in self.reparses else 0


class Acl:
    def __init__(self, aces=None):
        self.aces = list(aces or [])

    def AddAccessAllowedAceEx(self, revision, flags, mask, sid):
        self.aces.append(((0, flags), mask, sid, revision))

    def GetAceCount(self):
        return len(self.aces)

    def GetAce(self, index):
        value = self.aces[index]
        return value[:3]


class Descriptor:
    def __init__(self, owner, dacl, control=12):
        self.owner, self.dacl, self.control = owner, dacl, control

    def GetSecurityDescriptorOwner(self):
        return self.owner

    def GetSecurityDescriptorDacl(self):
        return self.dacl

    def GetSecurityDescriptorControl(self):
        return self.control, 1


class Security:
    ACL_REVISION_DS = 4
    OWNER_SECURITY_INFORMATION = 1
    DACL_SECURITY_INFORMATION = 2
    PROTECTED_DACL_SECURITY_INFORMATION = 4
    SE_FILE_OBJECT = 1
    SE_DACL_PRESENT = 4
    SE_DACL_PROTECTED = 8
    ACCESS_ALLOWED_ACE_TYPE = 0
    INHERITED_ACE = 16

    def __init__(self):
        self.set_call = None
        self.descriptor = None
        self.created_acls = []

    def LookupAccountName(self, _system, _identity):
        return SERVICE_SID, None, None

    def ConvertSidToStringSid(self, sid):
        return sid

    def ConvertStringSidToSid(self, sid):
        return sid

    def ACL(self):
        acl = Acl()
        self.created_acls.append(acl)
        return acl

    def SetNamedSecurityInfo(self, path, kind, info, owner, group, dacl, sacl):
        self.set_call = (path, kind, info, owner, group, dacl, sacl)
        self.descriptor = Descriptor(owner, dacl)

    def GetNamedSecurityInfo(self, *_):
        return self.descriptor


def record(**updates):
    value = {
        "run_token": TOKEN,
        "ownership_phase": "SERVICE_PROVEN",
        "strict_create_result": "CREATED",
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "service_sid": SERVICE_SID,
        "path_security_plan": {"frozen": True},
        "grants": ["one"],
        "service_path_name": "owned",
    }
    value.update(updates)
    return value


def patch_paths(monkeypatch, logs):
    paths = SimpleNamespace(logs=logs)
    monkeypatch.setattr(provisioner, "qualify_native_paths", lambda **_: paths)
    monkeypatch.setattr(qualifier, "qualify_native_paths", lambda **_: paths)
    monkeypatch.setattr(provisioner, "windows_path_identity", lambda path, _api: str(path))
    monkeypatch.setattr(qualifier, "windows_path_identity", lambda path, _api: str(path))
    return paths


def provisioned(tmp_path, monkeypatch):
    logs = tmp_path / "Logs"
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)
    security = Security()
    provision(path, TOKEN, win32api=Api, win32file=FileApi(), win32security=security)
    return path, logs, security


def test_logging_provision_absent_creates_plan_before_mutation_and_exact_acl(tmp_path, monkeypatch):
    logs = tmp_path / "Logs"
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)
    security = Security()
    original_save = provisioner._save_record
    observations = []

    def save_before_create(record_path, value):
        observations.append(("save", logs.exists(), "logging_security_plan" in value))
        original_save(record_path, value)

    monkeypatch.setattr(provisioner, "_save_record", save_before_create)
    provision(path, TOKEN, win32api=Api, win32file=FileApi(), win32security=security)
    saved = json.loads(path.read_text(encoding="utf-8"))
    plan = saved["logging_security_plan"]
    assert observations[0] == ("save", False, True)
    assert observations[1] == ("save", True, True)
    assert plan == {"role": "LOGS", "canonical_path": str(logs), "service_sid": SERVICE_SID}
    assert json.loads((logs / SENTINEL).read_text(encoding="utf-8")) == {
        "run_token": TOKEN,
        "canonical_path": str(logs),
        "service_sid": SERVICE_SID,
        "role": "LOGS",
    }
    _, _, info, owner, _, dacl, _ = security.set_call
    assert owner == ADMINISTRATORS_SID
    assert info & security.PROTECTED_DACL_SECURITY_INFORMATION
    assert {
        (sid, mask, flags)
        for (_type, flags), mask, sid, revision in dacl.aces
        if revision == security.ACL_REVISION_DS
    } == expected_logging_aces(SERVICE_SID)
    service_mask = next(
        mask for sid, mask, _ in expected_logging_aces(SERVICE_SID) if sid == SERVICE_SID
    )
    assert service_mask == FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE
    assert not service_mask & 0x40000
    assert not service_mask & 0x80000
    assert service_mask != FILE_ALL_ACCESS


@pytest.mark.parametrize(
    "checkpoint",
    [
        "PLAN_SAVED",
        "DIRECTORY_CREATED",
        "OWNERSHIP_PROOF_SAVED",
        "DACL_PROVISIONED",
        "TARGET_PUBLISHED",
    ],
)
def test_cleanup_recovers_each_interrupted_owned_provision_state(tmp_path, monkeypatch, checkpoint):
    logs = tmp_path / "Logs"
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)

    def interrupt(observed):
        if observed == checkpoint:
            raise RuntimeError(f"interrupted at {checkpoint}")

    with pytest.raises(RuntimeError, match="interrupted"):
        provision(
            path,
            TOKEN,
            win32api=Api,
            win32file=FileApi(),
            win32security=Security(),
            checkpoint=interrupt,
        )
    assert "logging_security_plan" in json.loads(path.read_text(encoding="utf-8"))
    cleanup(path, TOKEN, win32api=Api, win32file=FileApi())
    assert not logs.exists()
    assert "logging_security_plan" not in json.loads(path.read_text(encoding="utf-8"))


def test_cleanup_recovers_failed_dacl_provisioning(tmp_path, monkeypatch):
    class FailingSecurity(Security):
        def SetNamedSecurityInfo(self, *args):
            raise OSError("DACL failure")

    logs = tmp_path / "Logs"
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)
    with pytest.raises(OSError, match="DACL failure"):
        provision(
            path,
            TOKEN,
            win32api=Api,
            win32file=FileApi(),
            win32security=FailingSecurity(),
        )
    assert (logs.with_name(".stage5-logging-staging") / SENTINEL).is_file()
    cleanup(path, TOKEN, win32api=Api, win32file=FileApi())
    assert not logs.exists()
    assert "logging_security_plan" not in json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("reparse", [False, True])
def test_logging_provision_preexisting_fails_closed(tmp_path, monkeypatch, reparse):
    logs = tmp_path / "Logs"
    logs.mkdir()
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)
    file_api = FileApi([logs] if reparse else [])
    with pytest.raises(WindowsLoggingProvisionError, match="pre-exists"):
        provision(path, TOKEN, win32api=Api, win32file=file_api, win32security=Security())
    assert logs.exists()


def test_logging_provision_preserves_preexisting_staging_target(tmp_path, monkeypatch):
    logs = tmp_path / "Logs"
    staging = tmp_path / ".stage5-logging-staging"
    staging.mkdir()
    foreign = staging / "foreign.txt"
    foreign.write_text("foreign", encoding="utf-8")
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)
    with pytest.raises(WindowsLoggingProvisionError, match="staging target pre-exists"):
        provision(path, TOKEN, win32api=Api, win32file=FileApi(), win32security=Security())
    assert foreign.exists()
    assert "logging_security_plan" not in json.loads(path.read_text(encoding="utf-8"))


def test_expected_logging_acl_is_exact():
    assert expected_logging_aces(SERVICE_SID) == {
        (ADMINISTRATORS_SID, FILE_ALL_ACCESS, ACE_FLAGS),
        (SYSTEM_SID, FILE_ALL_ACCESS, ACE_FLAGS),
        (
            SERVICE_SID,
            FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE,
            ACE_FLAGS,
        ),
    }


def exact_qualified(tmp_path, monkeypatch):
    path, logs, security = provisioned(tmp_path, monkeypatch)
    value = json.loads(path.read_text(encoding="utf-8"))
    return value, logs, security


def test_read_only_logging_qualifier_passes_exact_dacl(tmp_path, monkeypatch):
    value, logs, security = exact_qualified(tmp_path, monkeypatch)
    assert qualify_record(
        value, run_token=TOKEN, win32api=Api, win32file=FileApi(), win32security=security
    ) == {"LOGS_DACL": "PASS", "READ_ONLY_QUALIFIER": "PASS"}
    assert logs.exists()


@pytest.mark.parametrize("mutation", ["owner", "inherited", "extra", "mask"])
def test_read_only_logging_qualifier_rejects_acl_mutations(tmp_path, monkeypatch, mutation):
    value, _logs, security = exact_qualified(tmp_path, monkeypatch)
    if mutation == "owner":
        security.descriptor.owner = SYSTEM_SID
    elif mutation == "inherited":
        security.descriptor.dacl.aces[0] = (
            (0, ACE_FLAGS | security.INHERITED_ACE),
            FILE_ALL_ACCESS,
            ADMINISTRATORS_SID,
            4,
        )
    elif mutation == "extra":
        security.descriptor.dacl.aces.append(((0, ACE_FLAGS), FILE_GENERIC_READ, "S-1-1-0", 4))
    else:
        security.descriptor.dacl.aces[-1] = ((0, ACE_FLAGS), FILE_GENERIC_READ, SERVICE_SID, 4)
    with pytest.raises(WindowsDaclQualificationError):
        qualify_record(
            value, run_token=TOKEN, win32api=Api, win32file=FileApi(), win32security=security
        )


def test_read_only_logging_qualifier_rejects_plan_mismatch(tmp_path, monkeypatch):
    value, _logs, security = exact_qualified(tmp_path, monkeypatch)
    value["logging_security_plan"]["role"] = "STATE"
    with pytest.raises(WindowsLoggingQualificationError, match="plan mismatch"):
        qualify_record(
            value, run_token=TOKEN, win32api=Api, win32file=FileApi(), win32security=security
        )


def test_qualifier_source_has_no_mutation_api():
    source = inspect.getsource(qualifier)
    for forbidden in (
        "SetNamedSecurityInfo",
        "AddAccessAllowedAce",
        ".mkdir(",
        ".unlink(",
        ".rmdir(",
        ".write_text(",
        ".replace(",
        "icacls",
    ):
        assert forbidden not in source


def test_exact_owned_cleanup_removes_only_logging_plan(tmp_path, monkeypatch):
    path, logs, _security = provisioned(tmp_path, monkeypatch)
    (logs / "backend.log").write_text("event", encoding="utf-8")
    before = json.loads(path.read_text(encoding="utf-8"))
    cleanup(path, TOKEN, win32api=Api, win32file=FileApi())
    after = json.loads(path.read_text(encoding="utf-8"))
    assert not logs.exists()
    assert "logging_security_plan" not in after
    for key in ("path_security_plan", "grants", "service_path_name", "ownership_phase"):
        assert after[key] == before[key]


@pytest.mark.parametrize("foreign_kind", ["file", "directory", "reparse", "sentinel"])
def test_cleanup_preserves_foreign_or_unproven_content(tmp_path, monkeypatch, foreign_kind):
    path, logs, _security = provisioned(tmp_path, monkeypatch)
    file_api = FileApi()
    if foreign_kind == "file":
        (logs / "foreign.txt").write_text("x", encoding="utf-8")
    elif foreign_kind == "directory":
        (logs / "foreign").mkdir()
    elif foreign_kind == "reparse":
        item = logs / "backend.log"
        item.write_text("x", encoding="utf-8")
        file_api = FileApi([item])
    else:
        (logs / SENTINEL).write_text("{}", encoding="utf-8")
    with pytest.raises((WindowsLoggingProvisionError, json.JSONDecodeError)):
        cleanup(path, TOKEN, win32api=Api, win32file=file_api)
    assert logs.exists()


def test_cleanup_run_token_mismatch_preserves_logs(tmp_path, monkeypatch):
    path, logs, _security = provisioned(tmp_path, monkeypatch)
    with pytest.raises(WindowsLoggingProvisionError, match="authority"):
        cleanup(path, "wrong", win32api=Api, win32file=FileApi())
    assert logs.exists()


def test_interrupted_empty_directory_with_foreign_content_is_preserved(tmp_path, monkeypatch):
    logs = tmp_path / "Logs"
    path = tmp_path / "ownership.json"
    path.write_text(json.dumps(record()), encoding="utf-8")
    patch_paths(monkeypatch, logs)

    def interrupt(name):
        if name == "DIRECTORY_CREATED":
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError):
        provision(
            path,
            TOKEN,
            win32api=Api,
            win32file=FileApi(),
            win32security=Security(),
            checkpoint=interrupt,
        )
    staging = logs.with_name(".stage5-logging-staging")
    (staging / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(WindowsLoggingProvisionError, match="ownership proof"):
        cleanup(path, TOKEN, win32api=Api, win32file=FileApi())
    assert staging.exists() and (staging / "foreign.txt").exists()


def test_crash_and_final_logging_proofs_recheck_all_historical_events():
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    helper = source[
        source.index("function Assert-LogEvent") : source.index(
            "function Remove-OwnedProcessTreeArtifacts"
        )
    ]
    for name in ("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.log.4"):
        assert f'"{name}"' in helper
    crash = source[
        source.index('$stage = "PERSISTENT_LOGGING_CRASH_RESTART"') : source.index(
            "$verifyRecovery = @("
        )
    ]
    assert crash.count('Assert-LogEvent "SERVICE_START" $pid1') == 1
    assert crash.count('Assert-LogEvent "SERVICE_STOP" $pid1') == 1
    assert crash.count('Assert-LogEvent "SERVICE_START" $pid2') == 1
    assert crash.count('Assert-LogEvent "SERVICE_START" $currentPid') == 1
    assert 'Assert-LogEvent "SERVICE_STOP" $pid2' not in crash
    final = source[
        source.index('$stage = "PERSISTENT_LOGGING_FINAL_STOP"') : source.index("$allowedLogs = @(")
    ]
    for proof in (
        'Assert-LogEvent "SERVICE_START" $pid1',
        'Assert-LogEvent "SERVICE_STOP" $pid1',
        'Assert-LogEvent "SERVICE_START" $pid2',
        'Assert-LogEvent "SERVICE_START" $currentPid',
        'Assert-LogEvent "SERVICE_STOP" $currentPid',
    ):
        assert proof in final
    pass_assignment = source.index('$result.WINDOWS_PERSISTENT_LOGGING = "PASS"')
    assert pass_assignment > source.index(
        'Assert-LogEvent "SERVICE_STOP" $currentPid',
        source.index('$stage = "PERSISTENT_LOGGING_FINAL_STOP"'),
    )


def test_log_event_helper_uses_non_reserved_service_pid_and_exact_log_proof():
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    helper = source[
        source.index("function Assert-LogEvent") : source.index(
            "function Remove-OwnedProcessTreeArtifacts"
        )
    ]
    assert "[int]$Pid" not in helper
    assert "$Pid" not in helper
    assert "[int]$ServicePid" in helper
    assert '"pid=$ServicePid .*$Event"' in helper
    assert '"log event missing event=$Event pid=$ServicePid"' in helper


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="requires PowerShell Core")
def test_log_event_helper_accepts_positional_pid_without_automatic_pid_collision(tmp_path):
    source = Path("deployment/windows_scm_probe.ps1").read_text(encoding="utf-8")
    helper = source[
        source.index("function Assert-LogEvent") : source.index(
            "function Remove-OwnedProcessTreeArtifacts"
        )
    ]
    logs = tmp_path / "Logs"
    logs.mkdir()
    (logs / "backend.log").write_text(
        "2026-01-01 00:00:00 INFO pid=4242 SERVICE_START\n", encoding="utf-8"
    )
    script = tmp_path / "assert-log-event.ps1"
    escaped_logs = str(logs).replace("'", "''")
    script.write_text(
        f"$logs = '{escaped_logs}'\n{helper}\nAssert-LogEvent 'SERVICE_START' 4242\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["pwsh", "-NoProfile", "-NonInteractive", "-File", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Cannot overwrite variable Pid" not in output


def test_frozen_bounded_append_utf8_contract(tmp_path):
    handler = create_handler(tmp_path)
    try:
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == MAX_BYTES == 5 * 1024 * 1024
        assert handler.backupCount == BACKUP_COUNT == 4
        assert handler.mode == "a"
        assert handler.encoding.lower().replace("-", "") == "utf8"
        assert handler.baseFilename.endswith(LOG_FILE_NAME)
        assert all(
            token in LOG_FORMAT
            for token in ("%(asctime)s", "%(levelname)s", "%(process)d", "%(message)s")
        )
        record = logging.LogRecord("x", logging.INFO, "", 0, "EVENT", (), None)
        rendered = handler.format(record)
        assert "Z level=INFO pid=" in rendered and rendered.endswith("EVENT")
    finally:
        handler.close()


def test_rotation_is_bounded_and_reopen_preserves_content(tmp_path):
    logger = logging.getLogger("stage5-test")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    logger.addHandler(create_handler(tmp_path, max_bytes=80, backup_count=2))
    for number in range(30):
        logger.info("record %s %s", number, "x" * 30)
    for handler in logger.handlers:
        handler.close()
    assert {p.name for p in tmp_path.iterdir()} <= {"backend.log", "backend.log.1", "backend.log.2"}
    current = (tmp_path / "backend.log").read_text(encoding="utf-8")
    service = configure_service_logger(tmp_path)
    service.info("AFTER_REOPEN")
    close_service_logger(service)
    assert current in (tmp_path / "backend.log").read_text(encoding="utf-8")
    assert "AFTER_REOPEN" in (tmp_path / "backend.log").read_text(encoding="utf-8")


def test_configure_replaces_and_closes_old_handler(tmp_path):
    logger = logging.getLogger("cryptohunter.windows.acceptance")
    old = create_handler(tmp_path)
    logger.handlers[:] = [old]
    configured = configure_service_logger(tmp_path)
    assert old.stream is None
    assert len(configured.handlers) == 1 and configured.handlers[0] is not old
    close_service_logger(configured)


def test_acceptance_service_messages_never_include_run_token():
    source = Path("deployment/windows_test_service.py").read_text(encoding="utf-8")
    assert 'logger.info("SERVICE_START")' in source
    assert 'logger.info("PROCESS_TREE_READY")' in source
    assert 'logger.info("SERVICE_STOP")' in source
    assert "run_token" not in source
