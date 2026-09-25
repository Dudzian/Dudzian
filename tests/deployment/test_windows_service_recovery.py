from __future__ import annotations

import ntpath
from copy import deepcopy

import pytest

from deployment.windows_service_recovery import (
    RESET_PERIOD_SECONDS,
    RecoveryQualificationError,
    configure_recovery,
)


class FakeScm:
    SC_MANAGER_CONNECT = 1
    SERVICE_QUERY_CONFIG = 2
    SERVICE_CHANGE_CONFIG = 4
    SERVICE_START = 16
    SERVICE_ALL_ACCESS = 983551
    SERVICE_AUTO_START = 2
    SERVICE_CONFIG_FAILURE_ACTIONS = 2
    SC_ACTION_NONE = 0
    SC_ACTION_RESTART = 1

    def __init__(self) -> None:
        self.changed = False
        self.requested_access = None
        self.policy = {"ResetPeriod": 0, "RebootMsg": "", "Command": "", "Actions": []}
        self.binary_path = '"C:\\host.exe"'
        self.start_name = r"NT SERVICE\CryptoHunterBackend"
        self.start_type = self.SERVICE_AUTO_START

    def OpenSCManager(self, *_args):
        return "manager"

    def OpenService(self, _manager, _name, desired_access):
        self.requested_access = desired_access
        return "service"

    def CloseServiceHandle(self, _handle):
        return None

    def QueryServiceConfig(self, _service):
        return (
            16,
            self.start_type,
            1,
            self.binary_path,
            None,
            0,
            (),
            self.start_name,
            "CryptoHunter",
        )

    def ChangeServiceConfig2(self, _service, level, policy):
        assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
        self.changed = True
        self.policy = deepcopy(policy)

    def QueryServiceConfig2(self, _service, _level):
        return deepcopy(self.policy)


class FakeSecurity:
    def __init__(self, sid: str = "S-1-5-80-test") -> None:
        self.sid = sid
        self.lookups: list[str] = []

    def LookupAccountName(self, _system, name):
        self.lookups.append(name)
        return self.sid, "domain", 5

    def ConvertSidToStringSid(self, sid):
        return sid


class FakeWin32Api:
    @staticmethod
    def GetFullPathName(path):
        return ntpath.abspath(path)


def configure(scm: FakeScm, *, mutate: bool = True, security=None):
    return configure_recovery(
        record(),
        run_token="a" * 64,
        win32service=scm,
        win32security=security or FakeSecurity(),
        win32api=FakeWin32Api(),
        mutate=mutate,
    )


def record() -> dict[str, object]:
    return {
        "run_token": "a" * 64,
        "ownership_phase": "SERVICE_PROVEN",
        "strict_create_result": "CREATED",
        "service_name": "CryptoHunterBackend",
        "service_identity": r"NT SERVICE\CryptoHunterBackend",
        "service_path_name": '"C:\\host.exe" service.py',
        "service_host": r"C:\host.exe",
        "service_sid": "S-1-5-80-test",
    }


def test_exact_bounded_recovery_policy_is_written_and_queried() -> None:
    scm = FakeScm()
    result = configure(scm)
    assert scm.changed
    assert result["start_type"] == scm.SERVICE_AUTO_START
    assert scm.policy == {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [
            (scm.SC_ACTION_RESTART, 1000),
            (scm.SC_ACTION_RESTART, 5000),
            (scm.SC_ACTION_RESTART, 30000),
            (scm.SC_ACTION_NONE, 0),
        ],
    }


def test_mutation_requests_query_change_and_start_without_all_access() -> None:
    scm = FakeScm()
    configure(scm)
    assert scm.requested_access == (
        scm.SERVICE_QUERY_CONFIG | scm.SERVICE_CHANGE_CONFIG | scm.SERVICE_START
    )
    assert scm.requested_access != scm.SERVICE_ALL_ACCESS


@pytest.mark.parametrize(
    "field,value",
    [
        ("ownership_phase", "INTENT_CREATED"),
        ("strict_create_result", "ALREADY_EXISTS"),
        ("service_name", "ForeignService"),
    ],
)
def test_unproven_or_mismatched_ownership_blocks_mutation(field: str, value: str) -> None:
    owned = record()
    owned[field] = value
    scm = FakeScm()
    with pytest.raises(RecoveryQualificationError):
        configure_recovery(
            owned,
            run_token="a" * 64,
            win32service=scm,
            win32security=FakeSecurity(),
            win32api=FakeWin32Api(),
        )
    assert not scm.changed


def test_verify_only_never_mutates() -> None:
    scm = FakeScm()
    scm.policy = {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    configure(scm, mutate=False)
    assert not scm.changed
    assert scm.requested_access == scm.SERVICE_QUERY_CONFIG
    assert not scm.requested_access & scm.SERVICE_CHANGE_CONFIG
    assert not scm.requested_access & scm.SERVICE_START


def test_service_host_windows_identity_allows_different_case() -> None:
    scm = FakeScm()
    scm.binary_path = '"c:\\HOST.EXE"'
    configure(scm)
    assert scm.changed


def test_alternate_account_text_is_authorized_by_exact_sid() -> None:
    scm = FakeScm()
    scm.start_name = r".\CryptoHunterBackend"
    security = FakeSecurity()
    configure(scm, security=security)
    assert security.lookups == [scm.start_name]
    assert scm.changed


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("binary_path", '"C:\\foreign.exe"'),
        ("binary_path", '"C:\\host.exe" --foreign'),
        ("binary_path", ""),
        ("start_name", ""),
        ("start_type", 3),
    ],
)
def test_scm_qualification_mismatch_blocks_mutation(attribute: str, value) -> None:
    scm = FakeScm()
    setattr(scm, attribute, value)
    with pytest.raises(RecoveryQualificationError):
        configure(scm)
    assert not scm.changed


def test_resolved_sid_mismatch_blocks_mutation() -> None:
    scm = FakeScm()
    with pytest.raises(RecoveryQualificationError):
        configure(scm, security=FakeSecurity("S-1-5-80-foreign"))
    assert not scm.changed


def test_verify_only_performs_host_account_and_autostart_qualification() -> None:
    scenarios = (
        ("binary_path", '"C:\\foreign.exe"', FakeSecurity()),
        ("start_name", "", FakeSecurity()),
        ("start_type", 3, FakeSecurity()),
        ("start_name", r"NT SERVICE\CryptoHunterBackend", FakeSecurity("foreign")),
    )
    for attribute, value, security in scenarios:
        scm = FakeScm()
        scm.policy = {
            "ResetPeriod": RESET_PERIOD_SECONDS,
            "RebootMsg": "",
            "Command": "",
            "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
        }
        setattr(scm, attribute, value)
        with pytest.raises(RecoveryQualificationError):
            configure(scm, mutate=False, security=security)
        assert not scm.changed


def test_crash_interval_has_no_manual_start_and_checks_new_matching_pid() -> None:
    probe = open("deployment/windows_scm_probe.ps1", encoding="utf-8").read()
    interval = probe[
        probe.index('$stage = "CRASH_RESTART"') : probe.index(
            '$stage = "POST_RECOVERY_GRACEFUL_STOP"'
        )
    ]
    assert "Start-Service" not in interval
    assert "Restart-Service" not in interval
    assert "win32serviceutil.StartService" not in interval
    assert "$currentPid -ne $originalPid" in interval
    assert "$markerPid -eq $currentPid" in interval
    assert "Stop-Process -Id $originalPid -Force" in interval


def test_post_recovery_graceful_stop_has_negative_restart_proof() -> None:
    probe = open("deployment/windows_scm_probe.ps1", encoding="utf-8").read()
    interval = probe[probe.index('$stage = "POST_RECOVERY_GRACEFUL_STOP"') :]
    assert "Stop-Service $service" in interval
    assert 'Status.ToString() -cne "Stopped"' in interval
    assert "Start-Sleep -Milliseconds 2000" in interval
