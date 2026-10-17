from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import windows_stage9_host_inspection as host
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_postgresql_auth_contract import final_hba_lines, ident_lines


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fresh() -> dict[str, Any]:
    return {
        "os": "Windows",
        "native_x64": True,
        "host_identity": {
            "hostname": "approved-host",
            "machine_guid": "0a16b42b-ded1-4eac-938d-bd03ea794ad0",
            "hardware_uuid": "3bc7c43f-af07-4bc3-a377-a5bde9550479",
        },
        "paths": {
            "program_files": r"C:\Program Files\CryptoHunter",
            "program_data": r"C:\ProgramData\CryptoHunter",
            "program_files_exists": False,
            "program_data_exists": False,
        },
        "products": [],
        "related_products": [],
        "all_user_msi": {
            "complete": True,
            "user_sid": "S-1-1-0",
            "context_mask": 7,
            "family_products": [],
            "conflicting_products": [],
        },
        "services": [],
        "enrollment": {"state": "absent", "files": []},
        "conflicts": [],
        "residue": [],
        "listeners": [],
    }


def acl_tree(root: str) -> dict[str, Any]:
    paths = [*reversed(PureWindowsPath(root).parents), PureWindowsPath(root)]
    records = [
        {
            "path": str(path),
            "final_path": str(path),
            "kind": "directory",
            "attributes": 16,
            "file_id": f"12345678:{index:016x}",
            "link_count": 1,
            "length": 0,
            "last_write_ticks": 1,
            "change_ticks": 1,
            "owner_sid": "S-1-5-18",
            "inheritance_disabled": True,
            "dacl_present": True,
            "dacl_canonical": True,
            "aces": [
                {
                    "sid": "S-1-5-18",
                    "rights": 0x1F01FF,
                    "access_type": "Allow",
                    "inherited": False,
                    "inheritance_flags": 0,
                    "propagation_flags": 0,
                }
            ],
        }
        for index, path in enumerate(paths, 1)
    ]
    return {
        "schema_version": 1,
        "root": root,
        "complete": True,
        "errors": [],
        "before": records,
        "after": copy.deepcopy(records),
    }


def saved_sql_claim(facts: dict[str, Any], kind: str) -> dict[str, Any]:
    if kind == "legacy":
        return copy.deepcopy(facts["postgresql"]["effective_sql"])
    # Reports are serializable explanations, including when callers add plausible
    # PASS/provenance claims. Neither their shape nor their values are authority.
    summary = json.loads(json.dumps(host.passive_sql.summarize_passive_postgresql(None, facts)))
    if kind == "forged_summary":
        summary["status"] = "PASS"
        summary["source"] = "live_passive_postgresql"
        summary["identity"] = CONTRACT.backend_service
        summary["role"] = "freshness_runtime"
        summary["provenance"] = {
            "source": "native-current-service-token",
            "evidence_mode": "live",
            "authenticated": True,
        }
        for component in summary["components"].values():
            component["status"] = "PASS"
    return summary


def simulated_partial_sql_receipt(facts: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise receipt/report plumbing without native tokens, SQL or service starts."""
    monkeypatch.setattr(
        host.passive_sql,
        "_native_binding",
        lambda _: {"identity": CONTRACT.backend_service, "role": "freshness_runtime"},
    )
    monkeypatch.setattr(host.passive_sql, "_connect", lambda _: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(
        host.passive_sql,
        "_collect_sql",
        lambda *args: {
            name: {
                "status": "BLOCKED"
                if name in {"active_hba", "cross_role_authentication"}
                else "PASS",
                "detail": "Simulated partial SQL evidence for integration testing.",
            }
            for name in host.passive_sql.COMPONENTS
        },
    )
    return host.passive_sql.collect_passive_postgresql(facts)


@pytest.fixture
def installed(tmp_path: Path) -> tuple[dict[str, Any], tuple[Path, Path, Path], dict[str, Any]]:
    facts = fresh()
    payload = tmp_path / "payload"
    payload.mkdir()
    backend = payload / "CryptoHunterBackend.exe"
    backend.write_bytes(b"qualified-backend")
    (payload / "support.dll").write_bytes(b"qualified-support")
    msi = tmp_path / "qualified.msi"
    msi.write_bytes(b"qualified-msi-not-executed")
    manifest_path = tmp_path / "installer-manifest.json"
    manifest = {
        "architecture": "x64",
        "product_version": "1.2.3",
        "msi": {"sha256": sha(msi)},
        "production_executables": {"CryptoHunterBackend.exe": sha(backend)},
        "installed_files": {item.name: sha(item) for item in payload.iterdir()},
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt_path = tmp_path / "clean-install-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "source_revision": "a" * 40,
                "ci_run_id": "100",
                "ci_provider": "https://github.com",
                "runner_os": "Windows",
                "product_version": "1.2.3",
                "manifest_sha256": sha(manifest_path),
                "msi_sha256": sha(msi),
            }
        ),
        encoding="utf-8",
    )
    facts["qualified_product_code"] = "{D79EA891-1C35-4CBC-B761-C5A5A46D8C71}"
    facts["products"] = [
        {
            "product_code": facts["qualified_product_code"],
            "version": "1.2.3",
            "name": "CryptoHunter",
            "windows_installer": 1,
            "view": "Registry64",
        }
    ]
    facts["related_products"] = [
        {"product_code": facts["qualified_product_code"], "version": "1.2.3", "state": 5}
    ]
    facts["all_user_msi"]["family_products"] = [
        {
            "product_code": facts["qualified_product_code"],
            "version": "1.2.3",
            "context": 4,
            "user_sid": "",
        }
    ]
    facts["payload_root"] = str(payload)
    facts["paths"]["program_files_exists"] = True
    facts["paths"]["program_data_exists"] = True
    facts["readiness"] = {}
    for index, name in enumerate(host.SERVICES, 1):
        binary = (
            PureWindowsPath(facts["paths"]["program_files"])
            / ("PostgreSQL" if name == CONTRACT.postgresql_service else "")
            / f"{name}.exe"
        )
        facts["services"].append(
            {
                "name": name,
                "command_line": f'"{binary}"',
                "account": rf"NT SERVICE\{name}",
                "service_type": 16,
                "start_type": 2 if index == 1 else 3,
                "dependencies": [CONTRACT.postgresql_service]
                if name == CONTRACT.backend_service
                else [],
                "state": "Running",
                "process_id": index * 10,
                "second_process_id": index * 10,
                "second_state": 4,
                "process_path": str(binary),
                "process_started": "2026-10-16T00:00:00+00:00",
                "recovery": {
                    "Command": "",
                    "RebootMsg": "",
                    "ResetPeriod": 86400,
                    "Actions": [(1, 1000), (1, 5000), (1, 30000), (0, 0)]
                    if name == CONTRACT.backend_service
                    else [],
                },
            }
        )
        if index > 1:
            facts["readiness"][name] = {
                "written": "2026-10-16T00:00:01+00:00",
                "marker": {
                    "pid": index * 10,
                    "db_role": "freshness_runtime" if index == 2 else "freshness_crypto_verifier",
                    "corehost_lock": True,
                    "cross_key_denied": True,
                    "cross_roles_denied": True,
                },
            }
    facts["listeners"] = [
        {
            "address": "127.0.0.1",
            "port": 55432,
            "process_id": 11,
            "parent_process_id": 10,
            "process_path": str(
                PureWindowsPath(facts["paths"]["program_files"]) / "PostgreSQL/bin/postgres.exe"
            ),
        }
    ]
    facts["final_services"] = copy.deepcopy(facts["services"])
    rows = [
        (
            1,
            "hostssl",
            ["freshness_gate"],
            ["freshness_crypto_verifier"],
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            ["map=stage8_cert", "clientcert=verify-full"],
            None,
        ),
        (
            2,
            "hostssl",
            ["freshness_gate"],
            ["freshness_runtime"],
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            ["map=stage8_cert", "clientcert=verify-full"],
            None,
        ),
        (
            3,
            "host",
            ["freshness_gate"],
            ["all"],
            "127.0.0.1",
            "255.255.255.255",
            "reject",
            None,
            None,
        ),
        (4, "host", ["all"], ["all"], "127.0.0.1", "255.255.255.255", "reject", None, None),
        (5, "host", ["all"], ["all"], "0.0.0.0", "0.0.0.0", "reject", None, None),
        (6, "host", ["all"], ["all"], "::", "::", "reject", None, None),
    ]
    facts["postgresql"] = {
        "hba_lines": list(final_hba_lines("freshness_gate")),
        "ident_lines": list(ident_lines()),
        "effective_sql": {
            "listen_addresses": "127.0.0.1",
            "port": 55432,
            "ssl": True,
            "server_version_num": 170011,
            "hba_rows": rows,
            "session_roles": ["freshness_runtime", "freshness_crypto_verifier"],
            "cross_roles_denied": True,
        },
    }
    facts["product_dacl"] = "PASS"
    facts["service_sids"] = [f"S-1-5-80-{index}-2-3-4-5" for index in range(1, 4)]
    facts["acl_trees"] = {
        name: acl_tree(facts["paths"][name]) for name in ("program_files", "program_data")
    }
    arguments = {
        "expected_manifest_sha256": sha(manifest_path),
        "expected_msi_sha256": sha(msi),
        "expected_receipt_sha256": sha(receipt_path),
        "source_revision": "a" * 40,
        "ci_run_id": "100",
    }
    return facts, (msi, manifest_path, receipt_path), arguments


def test_fresh_snapshot_never_claims_live_pass() -> None:
    report = host.inspect_existing_installation(fresh())
    assert report["status"] == "UNVERIFIED"
    assert report["evidence_mode"] == "rehearsal"
    assert report["run_stage10"] is False
    assert report["install_execution_authorized"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("products", [{"version": "1.0"}]),
        (
            "related_products",
            [{"product_code": "hidden-registration", "version": "1.0", "state": 5}],
        ),
        ("services", [{"name": "CryptoHunterBackend"}]),
        ("enrollment", {"state": "present", "files": ["corehost.sqlite"]}),
        ("enrollment", {"state": "unknown"}),
        ("enrollment", None),
        ("conflicts", ["old-service"]),
        ("residue", ["transaction.json"]),
        ("listeners", [{"port": 55432}]),
        ("products", None),
        ("native_x64", False),
    ],
)
def test_existing_unknown_or_conflicting_host_stays_blocked(field: str, value: Any) -> None:
    facts = fresh()
    facts[field] = value
    assert host.inspect_existing_installation(facts)["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "key,value",
    [
        ("program_files_exists", True),
        ("program_data_exists", True),
        ("program_files", r"C:\Program Files\..\CryptoHunter"),
        ("program_data", r"\\remote\CryptoHunter"),
    ],
)
def test_existing_or_unsafe_paths_stay_blocked(key: str, value: Any) -> None:
    facts = fresh()
    facts["paths"][key] = value
    assert host.inspect_existing_installation(facts)["status"] == "BLOCKED"


def test_live_calls_collector_and_cannot_accept_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(host, "collect_host_observation", fresh)
    assert host.run_host_inspection(live=True)["status"] == "PASS"
    assert host.run_host_inspection(live=True, snapshot=fresh())["status"] == "FAIL"
    monkeypatch.setattr(
        host, "collect_host_observation", lambda: (_ for _ in ()).throw(PermissionError("secret"))
    )
    report = host.run_host_inspection(live=True)
    assert report["status"] == "BLOCKED"
    assert "secret" not in json.dumps(report)
    assert host.run_host_inspection()["status"] == "BLOCKED"


def test_complete_post_install_rehearsal_remains_unverified(installed: Any) -> None:
    facts, paths, arguments = installed
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert report["status"] == "UNVERIFIED", report
    assert {item["name"] for item in report["checks"]} == {
        "qualified_inputs",
        "installed_files",
        "msi_registration",
        "runtime_provenance",
        "scm_configuration",
        "postgresql",
        "product_acl",
        "runtime_readiness",
        "final_scm_process_identity",
    }
    assert all(item["status"] == "UNVERIFIED" for item in report["checks"])


@pytest.mark.parametrize("index", range(3))
def test_each_qualified_input_is_rehashed_before_host_access(
    installed: Any, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    _facts, paths, arguments = installed
    paths[index].write_bytes(b"tampered")
    monkeypatch.setattr(
        host, "_collect_post_install", lambda _: pytest.fail("must fail before host access")
    )
    assert host.verify_post_install(*paths, live=True, **arguments)["status"] == "FAIL"


@pytest.mark.parametrize(
    "change,check",
    [
        ("file_sha", "installed_files"),
        ("extra_file", "installed_files"),
        ("version", "msi_registration"),
        ("product_code", "msi_registration"),
        ("mixed_run", "runtime_provenance"),
        ("service_account", "scm_configuration"),
        ("service_arguments", "scm_configuration"),
        ("automatic_backend", "scm_configuration"),
        ("dependencies", "scm_configuration"),
        ("recovery", "scm_configuration"),
        ("listener", "postgresql"),
        ("listener_pid", "postgresql"),
        ("postgresql_wrapper", "postgresql"),
        ("postgresql_pid", "postgresql"),
        ("hba", "postgresql"),
        ("effective_hba", "postgresql"),
        ("roles", "postgresql"),
        ("postgresql_version", "postgresql"),
        ("pid", "runtime_readiness"),
        ("process_path", "runtime_readiness"),
        ("stale_marker", "runtime_readiness"),
        ("lock", "runtime_readiness"),
        ("cross_key", "runtime_readiness"),
        ("cross_role", "runtime_readiness"),
        ("unsafe_acl", "product_acl"),
    ],
)
def test_post_install_negative_contracts(installed: Any, change: str, check: str) -> None:
    facts, paths, arguments = installed
    backend = facts["services"][1]
    marker = facts["readiness"][CONTRACT.backend_service]
    if change == "file_sha":
        (Path(facts["payload_root"]) / "support.dll").write_bytes(b"changed")
    elif change == "extra_file":
        (Path(facts["payload_root"]) / "extra.dll").write_bytes(b"extra")
    elif change == "version":
        facts["products"][0]["version"] = "9.9.9"
    elif change == "product_code":
        facts["products"][0]["product_code"] = "{00000000-0000-0000-0000-000000000001}"
    elif change == "mixed_run":
        arguments["ci_run_id"] = "101"
    elif change == "service_account":
        backend["account"] = "LocalSystem"
    elif change == "service_arguments":
        backend["command_line"] += " --unsafe"
    elif change == "automatic_backend":
        backend["start_type"] = 2
    elif change == "dependencies":
        backend["dependencies"] = []
    elif change == "recovery":
        backend["recovery"]["Command"] = "reboot.exe"
    elif change == "listener":
        facts["listeners"][0]["address"] = "0.0.0.0"
    elif change == "listener_pid":
        facts["listeners"][0]["parent_process_id"] = 999
    elif change == "postgresql_wrapper":
        facts["services"][0]["process_path"] = r"C:\malicious-wrapper.exe"
    elif change == "postgresql_pid":
        facts["services"][0]["second_process_id"] = 999
    elif change == "hba":
        facts["postgresql"]["hba_lines"].append("host all all 0.0.0.0/0 trust")
    elif change == "effective_hba":
        facts["postgresql"]["effective_sql"]["hba_rows"] = []
    elif change == "roles":
        facts["postgresql"]["effective_sql"]["session_roles"] = ["postgres"]
    elif change == "postgresql_version":
        facts["postgresql"]["effective_sql"]["server_version_num"] = 170010
    elif change == "pid":
        marker["marker"]["pid"] = 999
    elif change == "process_path":
        backend["process_path"] = r"C:\malware.exe"
    elif change == "stale_marker":
        marker["written"] = "2026-10-15T00:00:00+00:00"
    elif change == "lock":
        marker["marker"]["corehost_lock"] = False
    elif change == "cross_key":
        marker["marker"]["cross_key_denied"] = False
    elif change == "cross_role":
        marker["marker"]["cross_roles_denied"] = False
    elif change == "unsafe_acl":
        facts["acl_trees"]["program_files"]["before"][-1]["aces"][0]["sid"] = "S-1-1-0"
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert next(item for item in report["checks"] if item["name"] == check)["status"] == "FAIL", (
        report
    )


def test_fresh_install_does_not_start_backend_to_obtain_readiness(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    facts["services"][1]["state"] = "Stopped"
    facts["services"][1]["process_id"] = 0
    facts["final_services"] = copy.deepcopy(facts["services"])
    facts["readiness"] = {}
    facts["postgresql"]["effective_sql"] = None
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *a, **kw: pytest.fail("must not start services or query keys"),
    )
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert report["status"] == "BLOCKED", report
    assert report["legal_production_enrollment"] == "NOT_PERFORMED"
    assert facts["services"][1]["start_type"] == 3


def test_post_install_host_access_and_snapshot_in_live_mode_fail_closed(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    assert (
        host.verify_post_install(*paths, live=True, observation=facts, **arguments)["status"]
        == "FAIL"
    )
    monkeypatch.setattr(
        host, "_collect_post_install", lambda _: (_ for _ in ()).throw(PermissionError("private"))
    )
    report = host.verify_post_install(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED"
    assert "private" not in json.dumps(report)
    facts["enrollment"] = {"state": "present", "files": ["State/corehost.sqlite"]}
    assert host.verify_post_install(*paths, observation=facts, **arguments)["status"] == "BLOCKED"


@pytest.mark.parametrize("kind", ["legacy", "exported_summary", "forged_summary"])
def test_live_post_install_rejects_saved_sql_claims_from_collector_seam(
    installed: Any, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    facts, paths, arguments = installed
    facts["postgresql"]["effective_sql"] = saved_sql_claim(facts, kind)
    initial_services = copy.deepcopy(facts["services"])
    monkeypatch.setattr(host, "_collect_post_install", lambda _: facts)
    monkeypatch.setattr(host, "_powershell", lambda _: facts["final_services"])
    monkeypatch.setattr(
        host,
        "Path",
        lambda value: Path(facts["payload_root"])
        if value == facts["paths"]["program_files"]
        else Path(value),
    )
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("passive qualification cannot start processes"),
    )
    report = host.verify_post_install(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED", report
    assert (
        next(item for item in report["checks"] if item["name"] == "postgresql")["status"]
        == "BLOCKED"
    )
    assert all(
        component["status"] != "PASS"
        for component in report["postgresql_evidence"]["components"].values()
    )
    assert report["missing_authorities"]
    assert facts["services"] == initial_services
    assert report["run_stage10"] is False


@pytest.mark.parametrize("exported", [False, True])
def test_partial_passive_sql_cannot_complete_live_post_install(
    installed: Any, monkeypatch: pytest.MonkeyPatch, exported: bool
) -> None:
    facts, paths, arguments = installed
    receipt = simulated_partial_sql_receipt(facts, monkeypatch)
    summary = host.passive_sql.summarize_passive_postgresql(receipt, facts)
    assert summary["components"]["effective_configuration"]["status"] == "PASS"
    facts["postgresql"]["effective_sql"] = json.loads(json.dumps(summary)) if exported else receipt
    monkeypatch.setattr(host, "_collect_post_install", lambda _: facts)
    monkeypatch.setattr(host, "_powershell", lambda _: facts["final_services"])
    monkeypatch.setattr(
        host,
        "Path",
        lambda value: Path(facts["payload_root"])
        if value == facts["paths"]["program_files"]
        else Path(value),
    )
    monkeypatch.setattr(
        host.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("passive qualification cannot start processes"),
    )
    report = host.verify_post_install(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED", report
    components = report["postgresql_evidence"]["components"]
    if exported:
        assert all(item["status"] == "UNVERIFIED" for item in components.values())
    else:
        assert components["effective_configuration"]["status"] == "PASS"
        assert components["active_hba"]["status"] == "BLOCKED"
        assert components["cross_role_authentication"]["status"] == "BLOCKED"
        assert set(report["missing_authorities"]) == {
            "passive PostgreSQL active_hba",
            "passive PostgreSQL cross_role_authentication",
        }


@pytest.mark.parametrize(
    "expected",
    [
        {},
        {"../escape": "a" * 64},
        {"C:/escape": "a" * 64},
        {"bad\\escape": "a" * 64},
        {"file": "invalid"},
        {"./file": "a" * 64},
    ],
)
def test_installed_file_manifest_rejects_unsafe_paths(tmp_path: Path, expected: Any) -> None:
    with pytest.raises(host.HostInspectionError):
        host._installed_files(tmp_path, expected)


def test_installed_symlink_is_not_accepted(tmp_path: Path) -> None:
    original = tmp_path / "original"
    original.write_bytes(b"data")
    (tmp_path / "linked").symlink_to(original)
    with pytest.raises(host.HostInspectionError, match="redirected"):
        host._installed_files(tmp_path, {"original": sha(original), "linked": sha(original)})


def test_readonly_msi_product_code_uses_database_mode_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    view = SimpleNamespace(
        Execute=lambda _: calls.append("execute"),
        Fetch=lambda: SimpleNamespace(StringData=lambda _: "{product-code}"),
        Close=lambda: calls.append("close"),
    )
    database = SimpleNamespace(OpenView=lambda query: view)

    def open_database(path: str, mode: int) -> Any:
        calls.append(mode)
        return database

    installer = SimpleNamespace(OpenDatabase=open_database)
    monkeypatch.setattr(
        host.importlib, "import_module", lambda _: SimpleNamespace(Dispatch=lambda _: installer)
    )
    assert host._msi_product_code(tmp_path / "qualified.msi") == "{product-code}"
    assert calls == [0, "execute", "close"]


def test_native_scm_boundary_requests_query_rights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def open_service(manager: str, name: str, rights: int) -> str:
        calls.append(rights)
        return "service"

    scm = SimpleNamespace(
        SC_MANAGER_CONNECT=1,
        SERVICE_QUERY_CONFIG=1,
        SERVICE_QUERY_STATUS=4,
        SERVICE_CONFIG_FAILURE_ACTIONS=2,
        OpenSCManager=lambda *args: "manager",
        OpenService=open_service,
        QueryServiceConfig=lambda _: (16, 3, 0, "path", 0, 0, ["dependency"], "account"),
        QueryServiceStatusEx=lambda _: {"ProcessId": 5, "CurrentState": 4},
        QueryServiceConfig2=lambda *args: {"Actions": []},
        CloseServiceHandle=lambda handle: calls.append(handle),
    )
    monkeypatch.setattr(host.importlib, "import_module", lambda _: scm)
    assert host._service_configuration("service")["second_process_id"] == 5
    assert calls == [5, "service", "manager"]


def test_collector_script_has_no_mutating_commands() -> None:
    for forbidden in (
        "Win32_Product",
        "Start-Service",
        "Set-Service",
        "Remove-Item",
        "msiexec",
        "Restart-Computer",
        "ChangeServiceConfig",
    ):
        assert forbidden not in host._HOST_SCRIPT


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_id", 999),
        ("process_started", "2026-10-16T00:05:00+00:00"),
        ("process_path", r"C:\different.exe"),
        ("state", "Stopped"),
    ],
)
def test_service_churn_during_payload_checks_fails_closed(
    installed: Any, field: str, value: Any
) -> None:
    facts, paths, arguments = installed
    facts["final_services"][1][field] = value
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert (
        next(item for item in report["checks"] if item["name"] == "final_scm_process_identity")[
            "status"
        ]
        == "FAIL"
    )


def test_input_changed_during_verification_is_not_accepted(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    monkeypatch.setattr(
        host, "_product_acl", lambda _: paths[0].write_bytes(b"replaced-during-check")
    )
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert report["status"] == "FAIL"
    assert any(item["name"] == "qualified_input_stability" for item in report["checks"])
    assert report["checks"][-1]["name"] == "final_scm_process_identity"


def test_upgrade_family_discovery_does_not_depend_on_display_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def related_products(code: str) -> list[str]:
        calls.append(code)
        return ["{hidden-product}"]

    installer = SimpleNamespace(
        RelatedProducts=related_products,
        ProductInfo=lambda code, field: "1.2.3",
        ProductState=lambda code: 5,
    )
    monkeypatch.setattr(
        host.importlib, "import_module", lambda _: SimpleNamespace(Dispatch=lambda _: installer)
    )
    assert host._related_products() == [
        {"product_code": "{hidden-product}", "version": "1.2.3", "state": 5}
    ]
    assert calls == ["{" + CONTRACT.upgrade_code + "}"]


def test_qualified_input_rejects_redirected_parent(tmp_path: Path) -> None:
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "qualified.msi").write_bytes(b"msi")
    (tmp_path / "alias").symlink_to(actual, target_is_directory=True)
    with pytest.raises(host.HostInspectionError, match="redirected"):
        host._sha(tmp_path / "alias" / "qualified.msi")


@pytest.mark.parametrize(
    "failure", ["OpenService", "QueryServiceConfig", "QueryServiceStatusEx", "QueryServiceConfig2"]
)
def test_native_scm_closes_read_handles_on_every_failure(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    closed: list[str] = []

    def fail(*args: Any) -> Any:
        raise PermissionError("native read denied")

    scm = SimpleNamespace(
        SC_MANAGER_CONNECT=1,
        SERVICE_QUERY_CONFIG=1,
        SERVICE_QUERY_STATUS=4,
        SERVICE_CONFIG_FAILURE_ACTIONS=2,
        OpenSCManager=lambda *args: "manager",
        OpenService=lambda *args: "service",
        QueryServiceConfig=lambda _: (16, 3, 0, "path", 0, 0, [], "account"),
        QueryServiceStatusEx=lambda _: {"ProcessId": 5, "CurrentState": 4},
        QueryServiceConfig2=lambda *args: {"Actions": []},
        CloseServiceHandle=closed.append,
    )
    setattr(scm, failure, fail)
    monkeypatch.setattr(host.importlib, "import_module", lambda _: scm)
    with pytest.raises(PermissionError):
        host._service_configuration("service")
    assert closed == (["manager"] if failure == "OpenService" else ["service", "manager"])


def test_empty_readonly_msi_query_closes_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    closed = []
    view = SimpleNamespace(
        Execute=lambda _: None, Fetch=lambda: None, Close=lambda: closed.append(True)
    )
    installer = SimpleNamespace(OpenDatabase=lambda *args: SimpleNamespace(OpenView=lambda _: view))
    monkeypatch.setattr(
        host.importlib, "import_module", lambda _: SimpleNamespace(Dispatch=lambda _: installer)
    )
    with pytest.raises(host.HostInspectionError, match="ProductCode"):
        host._msi_product_code(tmp_path / "qualified.msi")
    assert closed == [True]


def test_native_host_collector_composes_independent_registration_and_scm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = fresh()
    facts["services"] = [{"name": CONTRACT.backend_service}]
    monkeypatch.setattr(host, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(host, "_powershell", lambda _: facts)
    monkeypatch.setattr(host, "_related_products", lambda: [{"product_code": "hidden"}])
    monkeypatch.setattr(host, "_all_user_msi_inventory", lambda: fresh()["all_user_msi"])
    monkeypatch.setattr(host, "_service_configuration", lambda _: {"second_process_id": 42})
    result = host.collect_host_observation()
    assert result["related_products"] == [{"product_code": "hidden"}]
    assert result["services"][0]["second_process_id"] == 42
    monkeypatch.setattr(host, "_powershell", lambda _: [])
    with pytest.raises(host.HostInspectionError, match="malformed"):
        host.collect_host_observation()
    monkeypatch.setattr(host, "os", SimpleNamespace(name="posix"))
    with pytest.raises(host.HostInspectionError, match="Windows"):
        host.collect_host_observation()


def test_post_collector_uses_only_passive_product_qualifier(
    installed: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, _arguments = installed
    root = Path(facts["payload_root"])
    machine = tmp_path / "machine"
    data = machine / "PostgreSQL" / "Data"
    data.mkdir(parents=True)
    (data / "pg_hba.conf").write_text(
        "\n".join(final_hba_lines("freshness_gate")), encoding="utf-8"
    )
    (data / "pg_ident.conf").write_text("\n".join(ident_lines()), encoding="utf-8")
    runtime = machine / "Runtime"
    runtime.mkdir()
    (runtime / "backend-readiness.json").write_text(
        json.dumps(facts["readiness"][CONTRACT.backend_service]["marker"]), encoding="utf-8"
    )
    native_paths = facts["paths"]
    monkeypatch.setattr(
        host,
        "Path",
        lambda value: root
        if value == native_paths["program_files"]
        else machine
        if value == native_paths["program_data"]
        else Path(value),
    )
    monkeypatch.setattr(host, "collect_host_observation", lambda: copy.deepcopy(facts))
    monkeypatch.setattr(host, "_msi_product_code", lambda _: facts["qualified_product_code"])
    reads = []
    provision = SimpleNamespace(qualify_dacl=lambda pf, pd: reads.append((pf, pd)))
    security = SimpleNamespace(
        LookupAccountName=lambda *args: ("sid",),
        ConvertSidToStringSid=lambda sid: "S-1-5-80-1-2-3-4-5",
    )
    original_import = host.importlib.import_module
    monkeypatch.setattr(
        host.importlib,
        "import_module",
        lambda name: provision
        if name.endswith(".provision")
        else security
        if name == "win32security"
        else original_import(name),
    )
    monkeypatch.setattr(host, "_powershell", lambda script, **kwargs: {"complete": True})
    result = host._collect_post_install(paths[0])
    assert reads == [(root, machine)]
    assert result["product_dacl"] == "PASS"
    summary = host.passive_sql.summarize_passive_postgresql(
        result["postgresql"]["effective_sql"], result
    )
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert result["postgresql"]["effective_sql"] is not None
    assert not isinstance(result["postgresql"]["effective_sql"], dict)
    assert CONTRACT.backend_service in result["readiness"]
    assert CONTRACT.verifier_service not in result["readiness"]
    assert set(result["acl_trees"]) == {"program_files", "program_data"}


@pytest.mark.parametrize(
    "identity",
    [
        None,
        {},
        {"hostname": "host", "machine_guid": "invalid", "hardware_uuid": "invalid"},
        {
            "hostname": "host",
            "machine_guid": "00000000-0000-0000-0000-000000000000",
            "hardware_uuid": "3bc7c43f-af07-4bc3-a377-a5bde9550479",
        },
    ],
)
def test_unknown_native_identity_blocks(identity: Any) -> None:
    facts = fresh()
    facts["host_identity"] = identity
    assert host.inspect_existing_installation(facts)["status"] == "BLOCKED"


@pytest.mark.parametrize(
    "state,context,cache_present", [("1", 2, True), ("5", 8, True), ("5", 2, False)]
)
def test_all_user_msi_unknown_advertised_or_cacheless_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str, context: int, cache_present: bool
) -> None:
    cache = tmp_path / "cached.msi"
    cache.write_bytes(b"cache")
    product = SimpleNamespace(
        ProductCode="{D79EA891-1C35-4CBC-B761-C5A5A46D8C71}",
        Context=context,
        UserSid="S-1-5-21-1-2-3-1001",
        InstallProperty=lambda key: state
        if key == "State"
        else str(cache)
        if cache_present
        else "",
    )
    installer = SimpleNamespace(ProductsEx=lambda *args: [product])
    monkeypatch.setattr(
        host.importlib, "import_module", lambda _: SimpleNamespace(Dispatch=lambda _: installer)
    )
    with pytest.raises(host.HostInspectionError, match="unavailable"):
        host._all_user_msi_inventory()


def test_all_user_msi_enumerates_every_context_and_detects_another_users_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cached.msi"
    cache.write_bytes(b"cache")
    code = "{D79EA891-1C35-4CBC-B761-C5A5A46D8C71}"
    product = SimpleNamespace(
        ProductCode=code,
        Context=2,
        UserSid="S-1-5-21-1-2-3-1001",
        InstallProperty=lambda key: "5" if key == "State" else str(cache),
    )
    calls = []

    def products_ex(code: str, sid: str, context: int) -> list[Any]:
        calls.append((code, sid, context))
        return [product]

    installer = SimpleNamespace(ProductsEx=products_ex)
    monkeypatch.setattr(
        host.importlib, "import_module", lambda _: SimpleNamespace(Dispatch=lambda _: installer)
    )
    values = {
        "ProductCode": code,
        "UpgradeCode": "{" + CONTRACT.upgrade_code + "}",
        "ProductName": "renamed-product",
        "ProductVersion": "1.2.3",
    }
    monkeypatch.setattr(host, "_msi_property", lambda installer, path, key: values[key])
    inventory = host._all_user_msi_inventory()
    assert calls == [("", "S-1-1-0", 7)]
    assert inventory["complete"] is True
    assert inventory["family_products"][0]["context"] == 2
    facts = fresh()
    facts["all_user_msi"] = inventory
    assert host.inspect_existing_installation(facts)["status"] == "BLOCKED"
    values["UpgradeCode"] = None
    values["ProductName"] = "another-vendor"
    assert host._all_user_msi_inventory()["family_products"] == []
    values["ProductName"] = "CryptoHunter Legacy"
    assert len(host._all_user_msi_inventory()["conflicting_products"]) == 1


@pytest.mark.parametrize(
    "inventory",
    [
        None,
        {"complete": False},
        {
            "complete": True,
            "user_sid": "current-user",
            "context_mask": 7,
            "family_products": [],
            "conflicting_products": [],
        },
    ],
)
def test_incomplete_all_user_inventory_never_becomes_fresh_pass(inventory: Any) -> None:
    facts = fresh()
    facts["all_user_msi"] = inventory
    assert host.inspect_existing_installation(facts)["status"] == "BLOCKED"


def test_program_files_is_not_writable_by_product_services(installed: Any) -> None:
    facts, paths, arguments = installed
    tree = facts["acl_trees"]["program_files"]
    for phase in ("before", "after"):
        tree[phase][-1]["aces"].append(
            {
                "sid": facts["service_sids"][1],
                "rights": 2,
                "access_type": "Allow",
                "inherited": False,
                "inheritance_flags": 0,
                "propagation_flags": 0,
            }
        )
    report = host.verify_post_install(*paths, observation=facts, **arguments)
    assert (
        next(item for item in report["checks"] if item["name"] == "product_acl")["status"] == "FAIL"
    )
