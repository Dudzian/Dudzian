"""Native SCM boundaries are modeled; these tests never start a service."""

from __future__ import annotations

from copy import deepcopy
import sys
from typing import Any

import pytest

from deployment.windows_installer import provision
from deployment.windows_installer.contract import CONTRACT


class ProvisionScm:
    SC_MANAGER_CONNECT = 1
    SERVICE_QUERY_CONFIG = 1
    SERVICE_CHANGE_CONFIG = 2
    SERVICE_START = 16
    SERVICE_CONFIG_FAILURE_ACTIONS = 2
    SERVICE_CONFIG_FAILURE_ACTIONS_FLAG = 4
    SC_ACTION_RESTART = 1
    SC_ACTION_NONE = 0

    def __init__(self, initial_flag: bool = False) -> None:
        self.flag: Any = initial_flag
        self.policy: dict[str, Any] = {}
        self.closed: list[str] = []
        self.calls: list[tuple[str, int]] = []
        self.query_override: dict[int, Any] = {}
        self.failure: tuple[str, int] | None = None

    def OpenSCManager(self, machine: Any, database: Any, rights: int) -> str:
        assert machine is None and database is None and rights == self.SC_MANAGER_CONNECT
        return "manager"

    def OpenService(self, manager: str, name: str, rights: int) -> str:
        assert manager == "manager" and name == CONTRACT.backend_service
        assert rights == self.SERVICE_QUERY_CONFIG | self.SERVICE_CHANGE_CONFIG | self.SERVICE_START
        self._call("open", rights)
        return "backend"

    def _call(self, operation: str, level: int) -> None:
        self.calls.append((operation, level))
        if self.failure == (operation, level):
            raise PermissionError("modeled native SCM denial")

    def ChangeServiceConfig2(self, service: str, level: int, value: Any) -> None:
        assert service == "backend"
        self._call("change", level)
        if level == self.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            assert type(value) is bool
            self.flag = value
        else:
            assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
            self.policy = deepcopy(value)

    def QueryServiceConfig2(self, service: str, level: int) -> Any:
        assert service == "backend"
        self._call("query", level)
        if level in self.query_override:
            return self.query_override[level]
        if level == self.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG:
            return self.flag
        assert level == self.SERVICE_CONFIG_FAILURE_ACTIONS
        return deepcopy(self.policy)

    def CloseServiceHandle(self, handle: str) -> None:
        self.closed.append(handle)


@pytest.mark.parametrize("initial_flag", [False, True])
def test_provision_writes_and_reads_bounded_actions_and_noncrash_flag(
    monkeypatch: pytest.MonkeyPatch, initial_flag: bool
) -> None:
    scm = ProvisionScm(initial_flag)
    monkeypatch.setitem(sys.modules, "win32service", scm)
    provision._recovery()
    assert scm.flag is True
    assert scm.policy == {
        "ResetPeriod": 86400,
        "RebootMsg": "",
        "Command": "",
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
    }
    assert scm.calls == [
        ("open", 19),
        ("change", scm.SERVICE_CONFIG_FAILURE_ACTIONS),
        ("change", scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG),
        ("query", scm.SERVICE_CONFIG_FAILURE_ACTIONS),
        ("query", scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG),
    ]
    assert scm.closed == ["backend", "manager"]


@pytest.mark.parametrize("observed", [False, None, 0, 1, "True", {"flag": True}])
def test_provision_rejects_wrong_or_malformed_flag_readback(
    monkeypatch: pytest.MonkeyPatch, observed: Any
) -> None:
    scm = ProvisionScm()
    scm.query_override[scm.SERVICE_CONFIG_FAILURE_ACTIONS_FLAG] = observed
    monkeypatch.setitem(sys.modules, "win32service", scm)
    with pytest.raises(provision.ProvisionError, match="non-crash failure flag"):
        provision._recovery()
    assert scm.closed == ["backend", "manager"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("ResetPeriod", 0),
        ("Actions", [(1, 1000)]),
        ("Command", "unexpected-command"),
        ("RebootMsg", "unexpected-reboot"),
    ],
)
def test_provision_rejects_action_policy_drift(
    monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    scm = ProvisionScm()
    observed = {
        "ResetPeriod": 86400,
        "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)],
        "Command": "",
        "RebootMsg": "",
    }
    observed[field] = value
    scm.query_override[scm.SERVICE_CONFIG_FAILURE_ACTIONS] = observed
    monkeypatch.setitem(sys.modules, "win32service", scm)
    with pytest.raises(provision.ProvisionError, match="recovery read-back mismatch"):
        provision._recovery()
    assert scm.closed == ["backend", "manager"]


@pytest.mark.parametrize(
    "failure",
    [("open", 19), ("change", 2), ("change", 4), ("query", 2), ("query", 4)],
)
def test_native_api_denial_propagates_and_releases_opened_handles(
    monkeypatch: pytest.MonkeyPatch, failure: tuple[str, int]
) -> None:
    scm = ProvisionScm()
    scm.failure = failure
    monkeypatch.setitem(sys.modules, "win32service", scm)
    with pytest.raises(PermissionError, match="modeled native SCM denial"):
        provision._recovery()
    assert scm.calls[-1] == failure
    assert scm.closed == (["manager"] if failure[0] == "open" else ["backend", "manager"])
