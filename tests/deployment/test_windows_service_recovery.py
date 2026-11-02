from __future__ import annotations

import ntpath
from copy import deepcopy

import pytest

from deployment.windows_service_recovery import (
    FAILURE_ACTIONS_ON_NON_CRASH_FAILURES,
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
    SERVICE_CONFIG_FAILURE_ACTIONS_FLAG = 4
    SC_ACTION_NONE = 0
    SC_ACTION_RESTART = 1

    def __init__(self) -> None:
        self.changed = False
        self.requested_access = None
        self.policy = {"ResetPeriod": 0, "RebootMsg": "", "Command": "", "Actions": []}
        self.binary_path = '"C:\\host.exe"'
        self.start_name = r"NT SERVICE\CryptoHunterBackend"
        self.start_type = self.SERVICE_AUTO_START
        self.stored_flag = False
        self.flag_writes: list[bool] = []
        self.query_levels: list[int] = []
        self.closed: list[str] = []

    def OpenSCManager(self, *_args):
        return "manager"

    def OpenService(self, _manager, _name, desired_access):
        self.requested_access = desired_access
        return "service"

    def CloseServiceHandle(self, handle):
        self.closed.append(handle)

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
        self.changed = True
        if level == self.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            assert type(policy) is bool
            self.flag_writes.append(policy)
            self.stored_flag = policy
        else:
            assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
            self.policy = deepcopy(policy)

    def QueryServiceConfig2(self, _service, level):
        self.query_levels.append(level)
        if level == self.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            return self.stored_flag
        assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
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


@pytest.mark.parametrize("initial_flag", [False, True])
def test_exact_bounded_recovery_policy_is_written_and_queried(initial_flag: bool) -> None:
    scm = FakeScm()
    scm.stored_flag = initial_flag
    result = configure(scm)
    assert scm.changed
    assert result["start_type"] == scm.SERVICE_AUTO_START
    assert FAILURE_ACTIONS_ON_NON_CRASH_FAILURES is True
    assert scm.flag_writes == [True]
    assert scm.stored_flag is True
    assert result["failure_actions_on_non_crash_failures"] is True
    assert scm.query_levels == [
        scm.SERVICE_CONFIG_FAILURE_ACTIONS,
        scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG,
    ]
    assert scm.closed == ["service", "manager"]
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
    scm.stored_flag = True
    scm.policy = {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    configure(scm, mutate=False)
    assert not scm.changed
    assert scm.flag_writes == []
    assert scm.query_levels == [
        scm.SERVICE_CONFIG_FAILURE_ACTIONS,
        scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG,
    ]
    assert scm.requested_access == scm.SERVICE_QUERY_CONFIG
    assert not scm.requested_access & scm.SERVICE_CHANGE_CONFIG
    assert not scm.requested_access & scm.SERVICE_START


def test_service_host_windows_identity_allows_different_case() -> None:
    scm = FakeScm()
    scm.binary_path = '"c:\\HOST.EXE"'
    configure(scm)
    assert scm.changed


def test_non_string_native_full_path_fails_before_recovery_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scm = FakeScm()
    monkeypatch.setattr(FakeWin32Api, "GetFullPathName", staticmethod(lambda _path: None))
    with pytest.raises(RecoveryQualificationError, match="non-string path"):
        configure(scm)
    assert not scm.changed
    assert scm.flag_writes == []
    assert scm.closed == ["service", "manager"]


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


class RecoveryTriggerScm(FakeScm):
    """Model the documented SCM trigger, separately from its failure actions.

    This is not native Windows evidence. FAILURE_ACTIONS_FLAG is a bool in
    pywin32 312; changing its stored value takes effect at the next system boot.
    See docs/windows_service_recovery_contract.md for the pinned API sources.
    """

    def __init__(self, *, stored_flag: bool, active_flag: bool) -> None:
        super().__init__()
        self.stored_flag = stored_flag
        self.active_flag = active_flag

    def first_action_after_exit(self, *, reported_stopped: bool, win32_exit: int):
        # Only a new failure is modeled, with no previously queued restart.
        failure = not reported_stopped or (self.active_flag and win32_exit != 0)
        return self.policy["Actions"][0] if failure and self.policy["Actions"] else None


@pytest.mark.parametrize("initial_flag", [False, True])
@pytest.mark.parametrize(
    "reported_stopped,win32_exit",
    [(False, 0), (False, 31), (True, 0), (True, 1066), (True, 31)],
    ids=["crash-exit-zero", "crash-exit-error", "normal-stop", "host-error-stop", "error-stop"],
)
def test_configured_policy_restarts_crash_and_error_stop_after_modeled_system_boot(
    initial_flag: bool,
    reported_stopped: bool,
    win32_exit: int,
) -> None:
    """Exercise the real setter against a model; no native boot is performed."""
    scm = RecoveryTriggerScm(stored_flag=initial_flag, active_flag=initial_flag)
    configure(scm)
    assert scm.flag_writes == [True]
    assert scm.stored_flag is True
    assert scm.active_flag is initial_flag
    before_boot = (1, 1000) if not reported_stopped or (initial_flag and win32_exit != 0) else None
    assert (
        scm.first_action_after_exit(reported_stopped=reported_stopped, win32_exit=win32_exit)
        == before_boot
    )
    # Model the documented activation boundary, without rebooting any host.
    scm.active_flag = scm.stored_flag
    assert scm.policy["Actions"] == [(1, 1000), (1, 5000), (1, 30000), (0, 0)]
    expected = None if reported_stopped and win32_exit == 0 else (1, 1000)
    assert (
        scm.first_action_after_exit(reported_stopped=reported_stopped, win32_exit=win32_exit)
        == expected
    )


@pytest.mark.parametrize("active_flag", [False, True])
def test_stored_flag_readback_does_not_prove_current_boot_recovery(
    active_flag: bool,
) -> None:
    scm = RecoveryTriggerScm(stored_flag=False, active_flag=active_flag)
    result = configure(scm)
    assert result["failure_actions_on_non_crash_failures"] is True
    assert scm.active_flag is active_flag
    expected = (1, 1000) if active_flag else None
    assert scm.first_action_after_exit(reported_stopped=True, win32_exit=1066) == expected


@pytest.mark.parametrize("flag", [False, True])
def test_trigger_flag_without_failure_actions_cannot_queue_recovery(flag: bool) -> None:
    scm = RecoveryTriggerScm(stored_flag=flag, active_flag=flag)
    assert scm.first_action_after_exit(reported_stopped=True, win32_exit=1066) is None
    assert scm.first_action_after_exit(reported_stopped=False, win32_exit=31) is None


@pytest.mark.parametrize("mutate", [False, True])
@pytest.mark.parametrize("observed_flag", [False, None, 0, 1, "True", {}, {"flag": True}])
def test_non_crash_flag_readback_must_be_exact_true(
    monkeypatch: pytest.MonkeyPatch, mutate: bool, observed_flag: object
) -> None:
    scm = FakeScm()
    scm.policy = {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    query = scm.QueryServiceConfig2

    def query_flag(service, level):
        if level == scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            return observed_flag
        return query(service, level)

    monkeypatch.setattr(scm, "QueryServiceConfig2", query_flag)
    with pytest.raises(RecoveryQualificationError, match="non-crash failure-actions flag"):
        configure(scm, mutate=mutate)
    assert scm.closed == ["service", "manager"]
    assert scm.changed is mutate
    assert scm.flag_writes == ([True] if mutate else [])
    assert scm.requested_access == (
        scm.SERVICE_QUERY_CONFIG | scm.SERVICE_CHANGE_CONFIG | scm.SERVICE_START
        if mutate
        else scm.SERVICE_QUERY_CONFIG
    )


@pytest.mark.parametrize(
    "operation,mutate", [("change_flag", True), ("query_flag", True), ("query_flag", False)]
)
def test_non_crash_flag_api_failure_propagates_and_closes_handles(
    monkeypatch: pytest.MonkeyPatch, operation: str, mutate: bool
) -> None:
    scm = FakeScm()
    scm.policy = {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    original = PermissionError("flag configuration denied")
    change = scm.ChangeServiceConfig2
    query = scm.QueryServiceConfig2

    def change_flag(service, level, policy):
        if level == scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            raise original
        return change(service, level, policy)

    def query_flag(service, level):
        if level == scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            raise original
        return query(service, level)

    monkeypatch.setattr(
        scm,
        "ChangeServiceConfig2" if operation == "change_flag" else "QueryServiceConfig2",
        change_flag if operation == "change_flag" else query_flag,
    )
    with pytest.raises(PermissionError) as raised:
        configure(scm, mutate=mutate)
    assert raised.value is original
    assert scm.closed == ["service", "manager"]
    if not mutate:
        assert not scm.changed
        assert scm.flag_writes == []
