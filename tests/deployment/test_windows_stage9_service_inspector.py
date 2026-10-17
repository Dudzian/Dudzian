"""Emulated administrator native reads; no Windows qualification is claimed."""

from __future__ import annotations

import copy
from pathlib import PureWindowsPath
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import windows_stage9_postgresql_passive as passive
from deployment import windows_stage9_service_evidence as evidence
from deployment.windows_installer.contract import CONTRACT


@pytest.fixture
def native(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    name = CONTRACT.postgresql_service
    root = PureWindowsPath(r"C:\Program Files\CryptoHunter")
    state = SimpleNamespace(
        host={
            "hostname": "native-host",
            "machine_guid": "native-machine-guid",
            "hardware_uuid": "native-hardware-uuid",
        },
        paths={"program_files": str(root), "program_data": r"C:\ProgramData\CryptoHunter"},
        service={
            "service_type": 16,
            "account": rf"NT SERVICE\{name}",
            "command_line": f'"{root / "PostgreSQL" / (name + ".exe")}"',
            "process_id": 30,
            "state": 4,
        },
        process={
            "process_id": 30,
            "process_path": str(root / "PostgreSQL" / (name + ".exe")),
            "process_started": "2026-10-16T00:00:00+00:00",
            "parent_process_id": 4,
        },
        listeners=[
            {
                "address": CONTRACT.postgresql_host,
                "port": CONTRACT.postgresql_port,
                "process_id": 31,
                "parent_process_id": 30,
                "process_path": str(root / "PostgreSQL/bin/postgres.exe"),
                "process_started": "2026-10-16T00:00:01+00:00",
            }
        ],
        events=[],
        failure=None,
    )
    state.observation = {
        "host_identity": copy.deepcopy(state.host),
        "paths": copy.deepcopy(state.paths),
        "services": [
            {
                "name": name,
                "account": state.service["account"],
                "command_line": state.service["command_line"],
                "state": "Running",
                "second_state": 4,
                "process_id": 30,
                "second_process_id": 30,
                "process_path": state.process["process_path"],
                "process_started": state.process["process_started"],
            }
        ],
        "listeners": copy.deepcopy(state.listeners),
    }

    def check(stage: str) -> None:
        state.events.append(stage)
        if state.failure == stage:
            raise PermissionError("private native access diagnostic")

    class RegistryKey:
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *_args: Any) -> None:
            state.events.append("registry-close")

    def registry_open(root_key: int, path: str, reserved: int, rights: int) -> RegistryKey:
        assert (root_key, path, reserved, rights) == (
            1,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            0x20119,
        )
        check("registry-open")
        return RegistryKey()

    def registry_query(_key: Any, field: str) -> Any:
        assert field == "MachineGuid"
        check("registry-query")
        return state.host["machine_guid"], None

    def manager_open(machine: Any, database: Any, rights: int) -> str:
        assert (machine, database, rights) == (None, None, 1)
        check("manager-open")
        return "manager"

    def service_open(manager: str, selected: str, rights: int) -> str:
        assert (manager, selected, rights) == ("manager", name, 5)
        check("service-open")
        return "postgresql-service"

    def query_config(handle: str) -> Any:
        assert handle == "postgresql-service"
        check("service-config")
        return (
            state.service["service_type"],
            None,
            None,
            state.service["command_line"],
            None,
            None,
            None,
            state.service["account"],
        )

    def query_status(handle: str) -> Any:
        assert handle == "postgresql-service"
        check("service-status")
        return {"CurrentState": state.service["state"], "ProcessId": state.service["process_id"]}

    def process_details(pid: int) -> Any:
        assert pid == state.service["process_id"]
        check("process-query")
        return copy.deepcopy(state.process)

    def listeners() -> Any:
        check("listeners-query")
        return copy.deepcopy(state.listeners)

    api = SimpleNamespace(GetComputerName=lambda: state.host["hostname"])
    registry = SimpleNamespace(
        HKEY_LOCAL_MACHINE=1,
        KEY_READ=0x20019,
        KEY_WOW64_64KEY=0x100,
        OpenKey=registry_open,
        QueryValueEx=registry_query,
    )
    scm = SimpleNamespace(
        SC_MANAGER_CONNECT=1,
        SERVICE_QUERY_CONFIG=1,
        SERVICE_QUERY_STATUS=4,
        OpenSCManager=manager_open,
        OpenService=service_open,
        QueryServiceConfig=query_config,
        QueryServiceStatusEx=query_status,
        CloseServiceHandle=lambda handle: state.events.append("close-" + handle),
    )
    for module_name, module in (("win32api", api), ("winreg", registry), ("win32service", scm)):
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setattr(evidence, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(passive, "_native_x64", lambda: check("native-architecture"))
    monkeypatch.setattr(passive, "_hardware_uuid", lambda: state.host["hardware_uuid"])
    monkeypatch.setattr(
        passive,
        "_known_folder",
        lambda index: state.paths["program_files" if index == 0x26 else "program_data"],
    )
    monkeypatch.setattr(passive, "_process_details", process_details)
    monkeypatch.setattr(passive, "_native_listeners", listeners)
    return state


def test_reads_native_host_paths_scm_process_and_listener_before_returning_binding(
    native: SimpleNamespace,
) -> None:
    result = evidence._inspect_host(native.observation)
    assert result == {
        "host_identity": native.host,
        "paths": native.paths,
        "listeners": native.listeners,
        "postgresql_service": native.process,
    }
    assert result["listeners"] is not native.observation["listeners"]
    assert native.events == [
        "native-architecture",
        "registry-open",
        "registry-query",
        "registry-close",
        "manager-open",
        "service-open",
        "service-config",
        "service-status",
        "process-query",
        "close-postgresql-service",
        "close-manager",
        "listeners-query",
    ]


@pytest.mark.parametrize("field", ("hostname", "machine_guid", "hardware_uuid"))
def test_observation_cannot_assert_host_identity(native: SimpleNamespace, field: str) -> None:
    native.observation["host_identity"][field] = "saved or caller-asserted identity"
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"
    assert "manager-open" not in native.events


@pytest.mark.parametrize("field", ("program_files", "program_data"))
def test_observation_cannot_select_product_or_credential_root(
    native: SimpleNamespace, field: str
) -> None:
    native.observation["paths"][field] = r"C:\Temp\copied-product"
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"
    assert "manager-open" not in native.events


@pytest.mark.parametrize(
    "field,value",
    [
        ("service_type", 32),
        ("account", "LocalSystem"),
        ("state", 1),
        ("process_id", 0),
        ("process_id", True),
        ("command_line", r'"C:\Temp\CryptoHunterPostgreSQL.exe"'),
    ],
)
def test_native_scm_contradiction_cannot_be_overridden_by_saved_service_row(
    native: SimpleNamespace,
    field: str,
    value: Any,
) -> None:
    native.service[field] = value
    with pytest.raises(passive.PassivePostgreSQLError):
        evidence._inspect_host(native.observation)
    assert native.events[-2:] == ["close-postgresql-service", "close-manager"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_id", 999),
        ("second_process_id", 999),
        ("state", "Stopped"),
        ("second_state", 1),
        ("account", "LocalSystem"),
        ("command_line", '"C:\\Temp\\copy.exe"'),
        ("process_path", r"C:\Temp\copy.exe"),
        ("process_started", "2026-10-17T00:00:00+00:00"),
    ],
)
def test_caller_service_fields_cannot_replace_native_scm_and_process_reads(
    native: SimpleNamespace,
    field: str,
    value: Any,
) -> None:
    native.observation["services"][0][field] = value
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"
    assert native.events[-2:] == ["close-postgresql-service", "close-manager"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_id", 999),
        ("process_path", r"C:\Temp\copy.exe"),
        ("process_started", "2026-10-17T00:00:00+00:00"),
    ],
)
def test_same_scm_pid_does_not_hide_process_image_or_creation_change(
    native: SimpleNamespace,
    field: str,
    value: Any,
) -> None:
    native.process[field] = value
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"
    assert native.events[-2:] == ["close-postgresql-service", "close-manager"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("address", "0.0.0.0"),
        ("port", 5432),
        ("process_id", 999),
        ("parent_process_id", 999),
        ("process_path", r"C:\Temp\postgres.exe"),
        ("process_started", "2026-10-15T00:00:00+00:00"),
        ("process_started", "2026-10-17T00:00:00+00:00"),
    ],
)
def test_native_listener_must_be_private_and_belong_to_the_canonical_service_lifetime(
    native: SimpleNamespace,
    field: str,
    value: Any,
) -> None:
    native.listeners[0][field] = value
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"
    assert "close-postgresql-service" in native.events and "close-manager" in native.events


@pytest.mark.parametrize("kind", ("none", "duplicate", "saved-none", "saved-duplicate"))
def test_missing_or_additional_listener_cannot_supply_private_endpoint_authority(
    native: SimpleNamespace,
    kind: str,
) -> None:
    rows = native.observation["listeners"] if kind.startswith("saved-") else native.listeners
    if kind.endswith("none"):
        rows.clear()
    else:
        rows.append(copy.deepcopy(rows[0]))
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"


@pytest.mark.parametrize(
    "field,value",
    [
        ("address", "0.0.0.0"),
        ("port", 5432),
        ("process_id", 999),
        ("parent_process_id", 999),
        ("process_path", r"C:\Temp\postgres.exe"),
        ("process_started", "2026-10-17T00:00:00+00:00"),
    ],
)
def test_saved_listener_fields_cannot_replace_native_endpoint_reads(
    native: SimpleNamespace,
    field: str,
    value: Any,
) -> None:
    native.observation["listeners"][0][field] = value
    with pytest.raises(passive.PassivePostgreSQLError) as failure:
        evidence._inspect_host(native.observation)
    assert failure.value.status == "FAIL"


@pytest.mark.parametrize(
    "stage,closed",
    [
        ("native-architecture", []),
        ("registry-open", []),
        ("registry-query", ["registry-close"]),
        ("manager-open", ["registry-close"]),
        ("service-open", ["registry-close", "close-manager"]),
        ("service-config", ["registry-close", "close-postgresql-service", "close-manager"]),
        ("service-status", ["registry-close", "close-postgresql-service", "close-manager"]),
        ("process-query", ["registry-close", "close-postgresql-service", "close-manager"]),
        ("listeners-query", ["registry-close", "close-postgresql-service", "close-manager"]),
    ],
)
def test_native_query_failures_release_every_already_opened_handle(
    native: SimpleNamespace,
    stage: str,
    closed: list[str],
) -> None:
    native.failure = stage
    with pytest.raises(PermissionError):
        evidence._inspect_host(native.observation)
    assert [event for event in native.events if "close" in event] == closed


def test_missing_platform_fails_before_any_native_read(
    native: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evidence, "os", SimpleNamespace(name="posix"))
    with pytest.raises(passive.PassivePostgreSQLError):
        evidence._inspect_host(native.observation)
    assert native.events == []
