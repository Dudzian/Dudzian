"""Częściowe fakty verifiera są walidowane, lecz nie otwierają bramki produktu."""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest

from deployment import linux_physical_collector as collector
from deployment import linux_physical_evidence as evidence
from deployment import linux_physical_ipc as ipc
from deployment import linux_physical_native as native
from deployment.linux_physical_observations import (
    ObservationError,
    validate_functional_observations,
    validate_installation,
)
from tests.deployment.test_linux_physical_collector import installation_fixture
from tests.deployment.test_linux_physical_evidence import _complete_transport
from tests.deployment.test_linux_verifier_observation import (
    CHALLENGE_ID,
    INSTALLATION_ID,
    SERVER_CGROUP,
    valid_verifier_observation,
)


def _installation(tmp_path):
    value = installation_fixture(tmp_path)
    value["functional_qualification"] = {"installation_id": INSTALLATION_ID}
    value["configuration"].append(
        {
            "path": "/etc/cryptohunter/linux-qualification.json",
            "sha256": evidence.digest({"schema_version": 1, "installation_id": INSTALLATION_ID}),
            "uid": 0,
            "gid": 0,
            "mode": 0o644,
        }
    )
    value["ipc"][0]["path"] = "/run/cryptohunter/freshness/verifier.sock"
    verifier = value["units"][1]
    verifier["name"] = SERVER_CGROUP.rsplit("/", 1)[1]
    verifier["unit_file"]["path"] = "/etc/systemd/system/" + verifier["name"]
    value["clean_paths"][2] = verifier["unit_file"]["path"]
    verifier["executable_sha256"] = "e" * 64
    validate_installation(value)
    return value


def _raw(challenge_id=CHALLENGE_ID):
    value = valid_verifier_observation()
    value["request"]["challenge_id"] = value["response"]["challenge_id"] = challenge_id
    return {"schema_version": 1, "status": "PARTIAL_BLOCKED", "verifier": value}


@pytest.mark.parametrize("fault", ["identity", "policy", "mode", "socket", "unknown"])
def test_optional_adapter_requires_exact_protected_installation_contract(tmp_path, fault):
    value = _installation(tmp_path)
    if fault == "identity":
        value["functional_qualification"]["installation_id"] = "not-a-digest"
    elif fault == "policy":
        value["configuration"].pop()
    elif fault == "mode":
        value["configuration"][-1]["mode"] = 0o600
    elif fault == "socket":
        value["ipc"][0]["path"] = "/run/foreign.sock"
    else:
        value["functional_qualification"]["command"] = "not-permitted"
    with pytest.raises(ObservationError):
        validate_installation(value)


@pytest.mark.parametrize("fault", ["challenge", "installation", "program", "cgroup", "pass"])
def test_pipeline_rejects_rebound_or_pass_only_partial_facts(tmp_path, fault):
    installation = _installation(tmp_path)
    value = _raw()
    if fault == "challenge":
        value["verifier"]["request"]["challenge_id"] = "0" * 64
    elif fault == "installation":
        value["verifier"]["response"]["installation_id"] = "0" * 64
    elif fault == "program":
        value["verifier"]["server_process"]["exe_sha256"] = "0" * 64
    elif fault == "cgroup":
        value["verifier"]["server_process"]["cgroup"] = "/foreign.service"
    else:
        value = {"status": "PASS"}
    with pytest.raises(ValueError):
        validate_functional_observations(value, installation, challenge_id=CHALLENGE_ID)


@pytest.mark.parametrize("changed", [False, True])
def test_native_adapter_binds_exchange_to_observed_systemd_process(tmp_path, monkeypatch, changed):
    installation = _installation(tmp_path)
    raw = _raw()
    driver = native.NativeDriver(installation, CHALLENGE_ID)
    process = raw["verifier"]["server_process"]
    properties = {
        "MainPID": str(process["pid"]),
        "ActiveState": "active",
        "ControlGroup": SERVER_CGROUP,
    }
    reads = 0

    def read_properties(_name):
        nonlocal reads
        reads += 1
        return {**properties, "MainPID": "999"} if changed and reads > 1 else properties

    monkeypatch.setattr(native.sys, "platform", "linux")
    # Model ścieżek kolektora Linux pozostaje POSIX także na Windows.
    monkeypatch.setattr(native, "Path", PurePosixPath)
    monkeypatch.setattr(native.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(native.os, "getegid", lambda: 0, raising=False)
    monkeypatch.setattr(
        native,
        "read_regular",
        lambda *_a, **_k: evidence.canonical_bytes(
            {"schema_version": 1, "installation_id": INSTALLATION_ID}
        ),
    )
    monkeypatch.setattr(driver, "properties", read_properties)
    monkeypatch.setattr(native, "process_observation", lambda _pid: process)
    calls = []

    def collect(path, **kwargs):
        calls.append((path, kwargs))
        return raw["verifier"]

    monkeypatch.setattr(ipc, "collect_verifier_observation", collect)
    if changed:
        with pytest.raises(ObservationError, match="zmieniła"):
            driver.functional_observations()
    else:
        assert driver.functional_observations() == raw
    assert calls[0][0] == "/run/cryptohunter/freshness/verifier.sock"
    assert calls[0][1]["expected"].pid == process["pid"]
    assert calls[0][1]["expected_cgroup"] == SERVER_CGROUP


def _context(tmp_path):
    context = _complete_transport(tmp_path, current=True)
    installation = _installation(tmp_path)
    context["host"]["installation"] = installation
    materials = context["host"]["materials"]
    materials["installation_contract_sha256"] = evidence.digest(installation)
    materials["configuration_sha256"] = evidence.digest(
        {row["path"]: row["sha256"] for row in installation["configuration"]}
    )
    context["challenge"] = evidence.issue_challenge(
        context["policy"],
        source_revision=materials["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
        host_id=context["host"]["host_id"],
        now=datetime.now(timezone.utc) - timedelta(seconds=30),
    )
    return context


def _diagnostic(context, monkeypatch):
    installation = context["host"]["installation"]
    identity = context["bundle"]["records"][0]["payload"]["host"]
    monkeypatch.setattr(collector, "host_identity", lambda: identity)
    monkeypatch.setattr(collector, "collect_materials", lambda _: context["host"]["materials"])
    return collector.collect_functional_diagnostic(
        challenge=context["challenge"],
        installation=installation,
        key_bytes=context["key"],
        driver=SimpleNamespace(
            functional_observations=lambda: _raw(context["challenge"]["challenge_id"])
        ),
    )


def test_readonly_collector_and_independent_verifier_keep_partial_domain(tmp_path, monkeypatch):
    context = _context(tmp_path)
    record = _diagnostic(context, monkeypatch)
    result = evidence.verify_functional_diagnostic(
        record,
        context["challenge"],
        context["policy"],
        source_revision=context["challenge"]["source_revision"],
        ci_run_id="12345",
        ci_run_attempt="2",
    )
    assert result["status"] == "PARTIAL_BLOCKED"
    assert result["production_qualification"] == "BLOCKED"
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.verify_bundle(
            record,
            context["challenge"],
            context["policy"],
            source_revision=context["challenge"]["source_revision"],
            ci_run_id="12345",
            ci_run_attempt="2",
        )


@pytest.mark.parametrize(
    "fault", ["unsigned", "signed_pass", "challenge", "host", "materials", "time", "kind", "ci"]
)
def test_diagnostic_rejects_tampering_even_with_valid_host_signature(tmp_path, monkeypatch, fault):
    context = _context(tmp_path)
    record = _diagnostic(context, monkeypatch)
    payload = copy.deepcopy(record["payload"])
    if fault in {"unsigned", "signed_pass"}:
        payload["observations"] = {"status": "PASS"}
    elif fault == "challenge":
        payload["challenge_id"] = "0" * 64
    elif fault == "host":
        payload["host"]["machine_id_sha256"] = "0" * 64
    elif fault == "materials":
        payload["materials"]["collector_sha256"] = "0" * 64
    elif fault == "time":
        payload["observed_at_utc"] = "2000-01-01T00:00:00Z"
    elif fault == "kind":
        payload["kind"] = "LIVE_PHYSICAL_LINUX"
    record = (
        {**record, "payload": payload}
        if fault == "unsigned"
        else evidence.sign_record(payload, context["key"])
    )
    with pytest.raises(evidence.LinuxEvidenceError):
        evidence.verify_functional_diagnostic(
            record,
            context["challenge"],
            context["policy"],
            source_revision=context["challenge"]["source_revision"],
            ci_run_id="999" if fault == "ci" else "12345",
            ci_run_attempt="2",
        )


@pytest.mark.parametrize("fault", [None, "hba", "ident", "process", "missing"])
def test_snapshot_replays_raw_pg_rules_and_correlates_service_identity(tmp_path, fault):
    from tests.deployment.test_linux_physical_collector import _snapshot as snapshot_fixture
    from deployment.linux_physical_observations import _snapshot

    installation = _installation(tmp_path)
    hba = [
        [
            "local",
            ["freshness_gate"],
            ["freshness_crypto_verifier"],
            None,
            None,
            "peer",
            ["map=reviewed"],
            None,
        ]
    ]
    ident = [["reviewed", "ch_verifier", "freshness_crypto_verifier", None]]
    installation["postgres"]["expected_hba_sha256"] = evidence.digest(hba)
    installation["postgres"]["expected_ident_sha256"] = evidence.digest(ident)
    snapshot = snapshot_fixture(installation, step=0)
    facts = _raw()
    process = snapshot["units"][1]["processes"][0]
    server = {
        key: process[key]
        for key in ("pid", "uid", "gid", "groups", "start_time_ticks", "exe_sha256")
    }
    server["cgroup"] = SERVER_CGROUP
    facts["verifier"]["server_process"] = copy.deepcopy(server)
    facts["verifier"]["response"]["server_process"] = copy.deepcopy(server)
    facts["verifier"]["server_peer_credentials"] = {
        key: server[key] for key in ("pid", "uid", "gid")
    }
    snapshot["functional_observations"] = facts
    pg = snapshot["postgres"]["administrative_observation"]
    pg["hba_rules"], pg["ident_mappings"] = hba, ident
    if fault == "hba":
        pg["hba_rules"][0][5] = "trust"
    elif fault == "ident":
        pg["ident_mappings"][0][2] = "postgres"
    elif fault == "process":
        facts["verifier"]["server_process"]["start_time_ticks"] += 1
        facts["verifier"]["response"]["server_process"]["start_time_ticks"] += 1
    elif fault == "missing":
        del snapshot["functional_observations"]
    if fault:
        with pytest.raises(ValueError):
            _snapshot(snapshot, installation, updated=False, challenge_id=CHALLENGE_ID)
    else:
        _snapshot(snapshot, installation, updated=False, challenge_id=CHALLENGE_ID)
