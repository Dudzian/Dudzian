from __future__ import annotations

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
            self.SERVICE_AUTO_START,
            1,
            '"C:\\host.exe" service.py',
            None,
            0,
            (),
            r"NT SERVICE\CryptoHunterBackend",
            "CryptoHunter",
        )

    def ChangeServiceConfig2(self, _service, level, policy):
        assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
        self.changed = True
        self.policy = deepcopy(policy)

    def QueryServiceConfig2(self, _service, _level):
        return deepcopy(self.policy)


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
    result = configure_recovery(record(), run_token="a" * 64, win32service=scm)
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
    configure_recovery(record(), run_token="a" * 64, win32service=scm)
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
        ("service_path_name", '"C:\\foreign.exe"'),
    ],
)
def test_unproven_or_mismatched_ownership_blocks_mutation(field: str, value: str) -> None:
    owned = record()
    owned[field] = value
    scm = FakeScm()
    with pytest.raises(RecoveryQualificationError):
        configure_recovery(owned, run_token="a" * 64, win32service=scm)
    assert not scm.changed


def test_verify_only_never_mutates() -> None:
    scm = FakeScm()
    scm.policy = {
        "ResetPeriod": RESET_PERIOD_SECONDS,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    configure_recovery(record(), run_token="a" * 64, win32service=scm, mutate=False)
    assert not scm.changed
    assert scm.requested_access == scm.SERVICE_QUERY_CONFIG
    assert not scm.requested_access & scm.SERVICE_CHANGE_CONFIG
    assert not scm.requested_access & scm.SERVICE_START


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
