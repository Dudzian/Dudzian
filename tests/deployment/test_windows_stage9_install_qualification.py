"""Hermetic phase boundaries; no installation, enrollment or service starts."""

from __future__ import annotations

import copy
import json
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

from deployment import (
    windows_stage9_host_inspection as host,
    windows_stage9_install_qualification as qualification,
)
from deployment.windows_installer.contract import CONTRACT
from tests.deployment import test_windows_stage9_host_inspection as host_tests

installed_source = host_tests.installed


@pytest.fixture
def installed(installed_source: Any) -> Any:
    facts, paths, arguments = installed_source
    security = PureWindowsPath(facts["paths"]["program_data"]) / "Security"
    facts["postgresql"]["postgresql.conf"] = [
        "# initdb defaults and installer-owned endpoint/TLS settings",
        "listen_addresses='127.0.0.1'",
        "port=55432",
        "ssl=on",
        f"ssl_ca_file='{(security / 'ca.crt').as_posix()}'",
        f"ssl_cert_file='{(security / 'server/server.crt').as_posix()}'",
        f"ssl_key_file='{(security / 'server/server.key').as_posix()}'",
    ]
    facts["postgresql"]["postgresql.auto.conf"] = ["# no ALTER SYSTEM overrides"]
    return facts, paths, arguments


def check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["name"] == name)


def clean_base(facts: dict[str, Any]) -> None:
    for service in facts["services"][1:]:
        service.update(
            state="Stopped",
            process_id=0,
            second_process_id=0,
            second_state=1,
            process_path=None,
            process_started=None,
        )
        service["recovery"]["Actions"] = []
    facts.pop("readiness")
    facts["postgresql"]["effective_sql"] = None
    facts["final_services"] = copy.deepcopy(facts["services"])


def no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("passive phase must not start processes or services")

    monkeypatch.setattr(host.subprocess, "run", forbidden)


def test_clean_base_does_not_require_or_read_enrolled_runtime(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    no_processes(monkeypatch)
    monkeypatch.setattr(
        host, "_readiness", lambda *args: pytest.fail("base must not read readiness")
    )
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert report["status"] == "BLOCKED", report
    assert all(
        item["status"] == "UNVERIFIED"
        for item in report["checks"]
        if item["name"] != "postgresql_effective_sql"
    )
    assert check(report, "postgresql_effective_sql")["status"] == "BLOCKED"
    assert report["qualification_phase"] == "clean-base-installation"
    assert report["legal_production_enrollment"] == "NOT_PERFORMED"
    assert report["run_stage10"] is False
    assert report["install_execution_authorized"] is False


def test_simulated_native_base_collects_without_runtime_markers(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    calls = []

    def collect(
        msi: Path,
        *,
        include_readiness: bool,
        include_configuration: bool,
        include_service_evidence: bool,
    ) -> Any:
        assert include_service_evidence is False
        calls.append((msi, include_readiness, include_configuration))
        return facts

    monkeypatch.setattr(host, "_collect_post_install", collect)
    monkeypatch.setattr(host, "_powershell", lambda _: facts["final_services"])
    monkeypatch.setattr(
        qualification,
        "Path",
        lambda value: Path(facts["payload_root"])
        if value == facts["paths"]["program_files"]
        else Path(value),
    )
    no_processes(monkeypatch)
    report = qualification.qualify_clean_base_installation(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED", report
    assert check(report, "postgresql_effective_sql")["status"] == "BLOCKED"
    for name in (
        "qualified_inputs",
        "installed_files",
        "msi_registration",
        "artifact_provenance",
        "scm_configuration",
        "postgresql_static",
        "postgresql_static_configuration",
        "product_acl",
        "base_services_stopped",
        "enrollment_absent",
        "qualified_input_stability",
        "final_scm_process_identity",
    ):
        assert check(report, name)["status"] == "PASS", report
    assert all(
        component["status"] != "PASS"
        for component in report["postgresql_evidence"]["components"].values()
    )
    assert calls == [(paths[0], False, True)]
    assert report["stage10"] == "BLOCKED / NOT_RUN"


@pytest.mark.parametrize(
    "qualifier",
    [qualification.qualify_clean_base_installation, qualification.qualify_enrolled_runtime],
)
def test_live_phase_rejects_saved_facts_before_collection(
    installed: Any, monkeypatch: pytest.MonkeyPatch, qualifier: Any
) -> None:
    facts, paths, arguments = installed
    monkeypatch.setattr(
        host, "_collect_post_install", lambda *a, **kw: pytest.fail("must reject snapshot first")
    )
    no_processes(monkeypatch)
    report = qualifier(*paths, live=True, observation=facts, **arguments)
    assert report["status"] == "FAIL"
    assert check(report, "live_collection")["status"] == "FAIL"


@pytest.mark.parametrize("enrolled", [False, True])
@pytest.mark.parametrize("kind", ["legacy", "exported_summary", "forged_summary"])
def test_live_phase_rejects_saved_sql_claims_from_collector_seam(
    installed: Any, monkeypatch: pytest.MonkeyPatch, enrolled: bool, kind: str
) -> None:
    facts, paths, arguments = installed
    claim = host_tests.saved_sql_claim(facts, kind)
    if enrolled:
        monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", lambda _: None)
        facts["services"][1]["start_type"] = 2
        facts["final_services"] = copy.deepcopy(facts["services"])
    else:
        clean_base(facts)
    facts["postgresql"]["effective_sql"] = claim
    initial_services = copy.deepcopy(facts["services"])
    monkeypatch.setattr(host, "collect_host_observation", lambda: facts)
    monkeypatch.setattr(host, "_collect_post_install", lambda *args, **kwargs: facts)
    monkeypatch.setattr(host, "_powershell", lambda _: facts["final_services"])
    monkeypatch.setattr(
        qualification,
        "Path",
        lambda value: Path(facts["payload_root"])
        if value == facts["paths"]["program_files"]
        else Path(value),
    )
    no_processes(monkeypatch)
    qualifier = (
        qualification.qualify_enrolled_runtime
        if enrolled
        else qualification.qualify_clean_base_installation
    )
    report = qualifier(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED", report
    assert check(report, "postgresql_effective_sql")["status"] == "BLOCKED"
    assert all(
        component["status"] != "PASS"
        for component in report["postgresql_evidence"]["components"].values()
    )
    assert report["missing_authorities"]
    assert facts["services"] == initial_services
    assert report["run_stage10"] is False
    if not enrolled:
        assert check(report, "base_services_stopped")["status"] == "PASS"
        assert all(service["start_type"] == 3 for service in facts["services"][1:])


@pytest.mark.parametrize("enrolled", [False, True])
@pytest.mark.parametrize("exported", [False, True])
def test_partial_passive_sql_cannot_complete_live_qualification(
    installed: Any, monkeypatch: pytest.MonkeyPatch, enrolled: bool, exported: bool
) -> None:
    facts, paths, arguments = installed
    if enrolled:
        monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", lambda _: None)
        facts["services"][1]["start_type"] = 2
        facts["final_services"] = copy.deepcopy(facts["services"])
    else:
        clean_base(facts)
    initial_services = copy.deepcopy(facts["services"])
    receipt = host_tests.simulated_partial_sql_receipt(facts, monkeypatch)
    summary = host.passive_sql.summarize_passive_postgresql(receipt, facts)
    assert summary["components"]["effective_configuration"]["status"] == "PASS"
    facts["postgresql"]["effective_sql"] = json.loads(json.dumps(summary)) if exported else receipt
    monkeypatch.setattr(host, "collect_host_observation", lambda: facts)

    def collect(*args: Any, **kwargs: Any) -> Any:
        assert kwargs["include_service_evidence"] is enrolled
        return facts

    monkeypatch.setattr(host, "_collect_post_install", collect)
    monkeypatch.setattr(host, "_powershell", lambda _: facts["final_services"])
    monkeypatch.setattr(
        qualification,
        "Path",
        lambda value: Path(facts["payload_root"])
        if value == facts["paths"]["program_files"]
        else Path(value),
    )
    no_processes(monkeypatch)
    qualifier = (
        qualification.qualify_enrolled_runtime
        if enrolled
        else qualification.qualify_clean_base_installation
    )
    report = qualifier(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED", report
    assert check(report, "postgresql_effective_sql")["status"] == "BLOCKED"
    assert check(report, "postgresql_effective_configuration")["status"] == (
        "UNVERIFIED" if exported else "PASS"
    )
    assert check(report, "postgresql_active_hba")["status"] != "PASS"
    assert check(report, "postgresql_cross_role_authentication")["status"] != "PASS"
    assert facts["services"] == initial_services
    assert report["stage10"] == "BLOCKED / NOT_RUN"
    if not enrolled:
        assert check(report, "base_services_stopped")["status"] == "PASS"
        assert all(service["start_type"] == 3 for service in facts["services"][1:])


@pytest.mark.parametrize("index", range(3))
def test_input_tamper_fails_before_native_collection(
    installed: Any, monkeypatch: pytest.MonkeyPatch, index: int
) -> None:
    _facts, paths, arguments = installed
    paths[index].write_bytes(b"changed")
    monkeypatch.setattr(
        host, "_collect_post_install", lambda *a, **kw: pytest.fail("must rehash first")
    )
    assert (
        qualification.qualify_clean_base_installation(*paths, live=True, **arguments)["status"]
        == "FAIL"
    )


@pytest.mark.parametrize(
    "field,value,gate",
    [
        ("state", "Running", "base_services_stopped"),
        ("process_id", 91, "base_services_stopped"),
        ("second_state", 4, "base_services_stopped"),
        ("process_path", r"C:\unexpected.exe", "base_services_stopped"),
        ("process_started", "2026-10-16T00:00:00+00:00", "base_services_stopped"),
        ("start_type", 2, "scm_configuration"),
        ("account", "LocalSystem", "scm_configuration"),
        ("dependencies", [], "scm_configuration"),
        ("recovery", {"Actions": [(1, 1000)]}, "scm_configuration"),
    ],
)
def test_clean_base_scm_negative_paths(installed: Any, field: str, value: Any, gate: str) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    facts["services"][1][field] = value
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert check(report, gate)["status"] == "FAIL", report


@pytest.mark.parametrize(
    "change,gate,status",
    [
        ("enrollment", "enrollment_absent", "BLOCKED"),
        ("extra_payload", "installed_files", "FAIL"),
        ("version", "msi_registration", "FAIL"),
        ("hba", "postgresql_static", "FAIL"),
        ("listener", "postgresql_static", "FAIL"),
        ("acl", "product_acl", "FAIL"),
        ("run", "artifact_provenance", "FAIL"),
        ("churn", "final_scm_process_identity", "FAIL"),
    ],
)
def test_clean_base_reuses_existing_negative_authorities(
    installed: Any, change: str, gate: str, status: str
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    if change == "enrollment":
        facts["enrollment"] = {"state": "present", "files": ["State/corehost.sqlite"]}
    elif change == "extra_payload":
        (Path(facts["payload_root"]) / "extra.dll").write_bytes(b"extra")
    elif change == "version":
        facts["products"][0]["version"] = "9.9.9"
    elif change == "hba":
        facts["postgresql"]["hba_lines"].append("host all all 0.0.0.0/0 trust")
    elif change == "listener":
        facts["listeners"][0]["address"] = "0.0.0.0"
    elif change == "acl":
        facts["acl_trees"]["program_files"]["before"][-1]["link_count"] = 2
    elif change == "run":
        arguments["ci_run_id"] = "other-run"
    elif change == "churn":
        facts["final_services"][0]["process_id"] = 99
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert check(report, gate)["status"] == status, report


def test_unenrolled_runtime_blocks_before_runtime_collection(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    facts["legal_production_enrollment"] = "CONFIRMED"
    facts["approved"] = True
    monkeypatch.setattr(host, "collect_host_observation", lambda: facts)
    monkeypatch.setattr(
        host, "_collect_post_install", lambda *a, **kw: pytest.fail("no runtime before enrollment")
    )
    no_processes(monkeypatch)
    report = qualification.qualify_enrolled_runtime(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED"
    assert check(report, "legal_enrollment_authority")["status"] == "BLOCKED"
    assert {item["name"] for item in report["checks"]} == {
        "qualified_inputs",
        "legal_enrollment_authority",
    }
    assert report["legal_production_enrollment"] == "NOT_PERFORMED"


def test_marker_cannot_replace_missing_accepted_enrollment_authority(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    facts["enrollment"] = {"state": "present", "files": ["State/enrollment-finalization.json"]}

    def missing() -> None:
        raise RuntimeError("secret untrusted error")

    monkeypatch.setattr(
        qualification.importlib,
        "import_module",
        lambda _: SimpleNamespace(load_windows_external_provisioning_handoff=missing),
    )
    no_processes(monkeypatch)
    report = qualification.qualify_enrolled_runtime(*paths, observation=facts, **arguments)
    assert report["status"] == "BLOCKED"
    assert "secret untrusted error" not in json.dumps(report)


def test_static_hba_and_markers_cannot_replace_effective_service_token_sql(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    # Fake only the legal-authority boundary to exercise the later SQL gate.
    monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", lambda _: None)
    facts["services"][1]["start_type"] = 2
    facts["final_services"] = copy.deepcopy(facts["services"])
    facts["postgresql"]["effective_sql"] = None
    no_processes(monkeypatch)
    report = qualification.qualify_enrolled_runtime(*paths, observation=facts, **arguments)
    assert report["status"] == "BLOCKED", report
    gate = check(report, "postgresql_effective_sql")
    assert gate["status"] == "BLOCKED"
    assert "service-token SQL authority is unavailable" in gate["details"]
    assert report["missing_authorities"]
    assert check(report, "runtime_mtls_readiness")["status"] == "UNVERIFIED"


@pytest.mark.parametrize(
    "change,gate",
    [
        ("hba", "postgresql_effective_sql"),
        ("role", "postgresql_effective_sql"),
        ("tls", "postgresql_effective_sql"),
        ("pid", "runtime_mtls_readiness"),
        ("backend_start", "scm_configuration"),
    ],
)
def test_enrolled_runtime_uses_existing_effective_authorities(
    installed: Any, monkeypatch: pytest.MonkeyPatch, change: str, gate: str
) -> None:
    facts, paths, arguments = installed
    monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", lambda _: None)
    facts["services"][1]["start_type"] = 2
    if change == "hba":
        facts["postgresql"]["effective_sql"]["hba_rows"] = []
    elif change == "role":
        facts["postgresql"]["effective_sql"]["session_roles"] = ["postgres"]
    elif change == "tls":
        facts["postgresql"]["effective_sql"]["ssl"] = False
    elif change == "pid":
        facts["readiness"][CONTRACT.backend_service]["marker"]["pid"] = 999
    elif change == "backend_start":
        facts["services"][1]["start_type"] = 3
    facts["final_services"] = copy.deepcopy(facts["services"])
    no_processes(monkeypatch)
    report = qualification.qualify_enrolled_runtime(*paths, observation=facts, **arguments)
    assert check(report, gate)["status"] == "FAIL", report


@pytest.mark.parametrize(
    "change", ["ssl", "port", "include", "auto", "duplicate", "missing", "unavailable"]
)
def test_static_postgresql_configuration_is_fail_closed(installed: Any, change: str) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    lines = facts["postgresql"]["postgresql.conf"]
    if change == "ssl":
        lines[3] = "ssl=off"
    elif change == "port":
        lines[2] = "port=5432"
    elif change == "include":
        lines.append("include_if_exists='untrusted.conf'")
    elif change == "auto":
        facts["postgresql"]["postgresql.auto.conf"].append("ssl=off")
    elif change == "duplicate":
        lines.insert(0, "ssl=off")
    elif change == "missing":
        lines.pop()
    elif change == "unavailable":
        facts["postgresql"]["postgresql.auto.conf"] = None
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert check(report, "postgresql_static_configuration")["status"] == (
        "BLOCKED" if change == "unavailable" else "FAIL"
    )


@pytest.mark.parametrize(
    "change", [None, "phase", "claim", "account", "device", "schema", "malformed"]
)
def test_legal_enrollment_uses_canonical_accepted_identity_and_committed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str | None
) -> None:
    state = tmp_path / "State"
    state.mkdir()
    finalization = state / "enrollment-finalization.json"
    record = {
        "schema_version": 1,
        "phase": "COMMITTED",
        "claim_reference": "accepted-claim",
        "account_id": "account",
        "device_installation_id": "device",
    }
    field = {
        "phase": "phase",
        "claim": "claim_reference",
        "account": "account_id",
        "device": "device_installation_id",
        "schema": "schema_version",
    }.get(change or "")
    if field:
        record[field] = "wrong"
    finalization.write_text("[]" if change == "malformed" else json.dumps(record), encoding="utf-8")
    accepted = object()
    calls = []

    def resolve(value: object) -> Any:
        assert value is accepted
        calls.append("accepted canonical authority")
        return (
            "accepted-claim",
            SimpleNamespace(account_id="account", device_installation_id="device"),
            None,
        )

    def committed(machine: Path) -> None:
        assert machine == tmp_path
        calls.append("committed ownership")

    composition = SimpleNamespace(
        load_windows_external_provisioning_handoff=lambda: accepted,
        resolve_accepted_handoff=resolve,
    )
    provision = SimpleNamespace(
        _committed_record=committed, ENROLLMENT_FINALIZATION="enrollment-finalization.json"
    )
    monkeypatch.setattr(
        qualification.importlib,
        "import_module",
        lambda name: composition if name.endswith("corehost_composition") else provision,
    )
    facts = {"enrollment": {"state": "present"}, "paths": {"program_data": str(tmp_path)}}
    if change is None:
        qualification._confirmed_legal_enrollment(facts)
    else:
        with pytest.raises(host.HostInspectionError, match="enrollment identity"):
            qualification._confirmed_legal_enrollment(facts)
    assert calls == ["accepted canonical authority", "committed ownership"]


def test_inputs_changed_during_passive_base_checks_fail_closed(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    original = qualification._base_services

    def mutate(value: dict[str, Any]) -> None:
        original(value)
        paths[0].write_bytes(b"substitution during inspection")

    monkeypatch.setattr(qualification, "_base_services", mutate)
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert check(report, "qualified_input_stability")["status"] == "FAIL"


def test_missing_native_access_blocks_without_disclosing_exception(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _facts, paths, arguments = installed

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise PermissionError("sensitive internal detail")

    monkeypatch.setattr(host, "_collect_post_install", missing)
    report = qualification.qualify_clean_base_installation(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED"
    assert "sensitive internal detail" not in json.dumps(report)


def test_native_collector_can_skip_readiness_and_collect_static_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, machine = tmp_path / "payload", tmp_path / "machine"
    root.mkdir()
    data = machine / "PostgreSQL" / "Data"
    data.mkdir(parents=True)
    for name in ("pg_hba.conf", "pg_ident.conf", "postgresql.conf", "postgresql.auto.conf"):
        (data / name).write_text("# static configuration\n", encoding="utf-8")
    (machine / "Runtime").mkdir()
    (machine / "Runtime" / "backend-readiness.json").write_text(
        "invalid readiness must not be read", encoding="utf-8"
    )
    facts = {"paths": {"program_files": str(root), "program_data": str(machine)}}
    monkeypatch.setattr(
        host.service_sql,
        "collect_service_postgresql",
        lambda *_: pytest.fail("clean-base cannot contact application evidence endpoints"),
    )
    monkeypatch.setattr(host, "collect_host_observation", lambda: facts)
    monkeypatch.setattr(host, "_msi_product_code", lambda _: "qualified-code")
    monkeypatch.setattr(host, "_powershell", lambda *args, **kwargs: {})
    security = SimpleNamespace(
        ConvertSidToStringSid=lambda sid: sid,
        LookupAccountName=lambda *args: ("S-1-5-80-1-2-3-4-5", None, None),
    )
    provision = SimpleNamespace(qualify_dacl=lambda *args: None)
    monkeypatch.setattr(
        host.importlib,
        "import_module",
        lambda name: security if name == "win32security" else provision,
    )
    no_processes(monkeypatch)
    result = host._collect_post_install(
        tmp_path / "qualified.msi", include_readiness=False, include_configuration=True
    )
    assert result["readiness"] == {}
    assert result["postgresql"]["postgresql.conf"] == ["# static configuration"]
    assert result["postgresql"]["postgresql.auto.conf"] == ["# static configuration"]
    summary = host.passive_sql.summarize_passive_postgresql(
        result["postgresql"]["effective_sql"], result
    )
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert result["postgresql"]["effective_sql"] is not None
    assert not isinstance(result["postgresql"]["effective_sql"], dict)


@pytest.mark.parametrize("value", [None, [], "PASS", {"effective_sql": None}])
def test_malformed_postgresql_observation_returns_a_closed_report(
    installed: Any, value: Any
) -> None:
    facts, paths, arguments = installed
    clean_base(facts)
    facts["postgresql"] = value
    report = qualification.qualify_clean_base_installation(*paths, observation=facts, **arguments)
    assert report["status"] in {"BLOCKED", "FAIL"}
    assert check(report, "postgresql_effective_sql")["status"] == "BLOCKED"


def test_enrollment_disappearing_before_runtime_reads_stays_blocked(
    installed: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    facts, paths, arguments = installed
    facts["enrollment"] = {"state": "present"}
    runtime_facts = copy.deepcopy(facts)
    runtime_facts["enrollment"] = {"state": "absent", "files": []}

    def confirm(value: dict[str, Any]) -> None:
        if value["enrollment"]["state"] != "present":
            raise host.HostInspectionError("confirmed legal enrollment is unavailable")

    monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", confirm)
    monkeypatch.setattr(host, "collect_host_observation", lambda: facts)
    monkeypatch.setattr(host, "_collect_post_install", lambda *args, **kwargs: runtime_facts)
    monkeypatch.setattr(
        host, "_readiness", lambda *args: pytest.fail("runtime legal gate must fail first")
    )
    no_processes(monkeypatch)
    report = qualification.qualify_enrolled_runtime(*paths, live=True, **arguments)
    assert report["status"] == "BLOCKED"
    assert check(report, "host_access")["status"] == "BLOCKED"
