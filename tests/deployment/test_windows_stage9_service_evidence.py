"""Emulated authenticated-service bridge tests; no native Windows qualification."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import PureWindowsPath
from typing import Any, Iterator

import pytest

from deployment import windows_stage9_postgresql_passive as passive
from deployment import windows_stage9_service_evidence as evidence
from deployment import windows_stage9_install_qualification as qualification
from deployment.windows_installer.contract import CONTRACT


SERVICES = (CONTRACT.backend_service, CONTRACT.verifier_service)
CREATED = "2026-10-16T00:00:00+00:00"
POSTMASTER_STARTED = "2026-10-16T00:00:01+00:00"
SYSTEM_IDENTIFIER = "7448802115599552300"
REAL_REQUIRE_ENROLLMENT = evidence._require_enrollment


def observation() -> dict[str, Any]:
    root = PureWindowsPath(r"C:\Program Files\CryptoHunter")
    services = []
    for index, name in enumerate((CONTRACT.postgresql_service, *SERVICES), 1):
        image = root / ("PostgreSQL" if index == 1 else "") / f"{name}.exe"
        services.append(
            {
                "name": name,
                "account": rf"NT SERVICE\{name}",
                "command_line": f'"{image}"',
                "state": "Running",
                "process_id": index * 10,
                "second_process_id": index * 10,
                "second_state": 4,
                "process_path": str(image),
                "process_started": CREATED,
            }
        )
    return {
        "host_identity": {
            "hostname": "approved-host",
            "machine_guid": "0a16b42b-ded1-4eac-938d-bd03ea794ad0",
            "hardware_uuid": "3bc7c43f-af07-4bc3-a377-a5bde9550479",
        },
        "paths": {
            "program_files": str(root),
            "program_data": r"C:\ProgramData\CryptoHunter",
        },
        "services": services,
        "listeners": [
            {
                "address": CONTRACT.postgresql_host,
                "port": CONTRACT.postgresql_port,
                "process_id": 11,
                "parent_process_id": 10,
                "process_path": str(root / "PostgreSQL/bin/postgres.exe"),
                "process_started": CREATED,
            }
        ],
    }


def captured_rows(name: str, backend_pid: int) -> dict[str, Any]:
    role = "freshness_runtime" if name == CONTRACT.backend_service else "freshness_crypto_verifier"
    hba = [
        [
            index,
            "hostssl",
            ["freshness_gate"],
            [login],
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            ["map=stage8_cert", "clientcert=verify-full"],
            None,
        ]
        for index, login in enumerate(("freshness_crypto_verifier", "freshness_runtime"), 1)
    ]
    hba.extend(
        [index, "host", databases, ["all"], address, mask, "reject", None, None]
        for index, databases, address, mask in (
            (3, ["freshness_gate"], "127.0.0.1", "255.255.255.255"),
            (4, ["all"], "127.0.0.1", "255.255.255.255"),
            (5, ["all"], "0.0.0.0", "0.0.0.0"),
            (6, ["all"], "::", "::"),
        )
    )
    return {
        "identity": [
            [
                "freshness_gate",
                role,
                role,
                CONTRACT.postgresql_host,
                CONTRACT.postgresql_port,
                backend_pid,
                POSTMASTER_STARTED,
                170011,
                CONTRACT.postgresql_host,
                CONTRACT.postgresql_port,
                True,
                True,
                True,
            ]
        ],
        "ssl": [[True, "TLSv1.3", "TLS_AES_256_GCM_SHA384", 256, f"/CN={name}"]],
        "data": [[r"C:\ProgramData\CryptoHunter\PostgreSQL\Data"]],
        "control": [[SYSTEM_IDENTIFIER]],
        "hba": hba,
        "roles": sorted(
            [login, False, False, False, False, True, False, False]
            for login in ("freshness_runtime", "freshness_crypto_verifier")
        ),
        "membership": [[0]],
    }


class EmulatedBridge:
    def __init__(self) -> None:
        self.facts = observation()
        self.identities = {
            row["name"]: {
                "name": row["name"],
                "process_id": row["process_id"],
                "token_sid": f"S-1-5-80-{index}",
                "process_path": row["process_path"],
                "process_started": row["process_started"],
            }
            for index, row in enumerate(self.facts["services"][1:], 1)
        }
        self.mutate: Any = lambda _name, _payload: None
        self.connects: list[str] = []
        self.closed: list[str] = []
        self.acknowledged: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.payloads: list[dict[str, Any]] = []
        self.unavailable: set[str] = set()
        self.ack_failure: set[str] = set()
        self.restarted: set[str] = set()

    def payload(self, name: str, request: dict[str, Any]) -> dict[str, Any]:
        own = copy.deepcopy(self.identities[name])
        own.update(
            identity=name,
            role="freshness_runtime"
            if name == CONTRACT.backend_service
            else "freshness_crypto_verifier",
            certificate_identity=name,
            paths=copy.deepcopy(self.facts["paths"]),
        )
        backend_pid = 12 if name == CONTRACT.backend_service else 13
        rows = captured_rows(name, backend_pid)
        counts = {
            "identity": 2,
            "ssl": 2,
            "data": 2,
            "control": 3,
            "hba": 1,
            "roles": 1,
            "membership": 1,
        }
        other = (
            "freshness_crypto_verifier" if name == CONTRACT.backend_service else "freshness_runtime"
        )
        payload = {
            "version": 1,
            "nonce": request["nonce"],
            "collection_id": request["collection_id"],
            "service": name,
            "process": own,
            "backend_pid": backend_pid,
            "queries": {
                key: [{"rows": copy.deepcopy(rows[key])} for _ in range(count)]
                for key, count in counts.items()
            },
            "probes": {
                role: {"status": "DENIED", "sqlstate": "28000"} for role in (other, "postgres")
            },
            "post_probe": {
                key: {"rows": copy.deepcopy(rows[key])}
                for key in ("identity", "ssl", "control", "data")
            },
        }
        self.mutate(name, payload)
        self.payloads.append(copy.deepcopy(payload))
        return payload

    @contextmanager
    def connect(self, name: str, facts: dict[str, Any]) -> Iterator[Any]:
        assert facts is self.facts
        self.connects.append(name)
        if name in self.unavailable:
            raise PermissionError("canonical service is unavailable")
        row = next(item for item in facts["services"] if item["name"] == name)
        if row["state"] != "Running":
            raise PermissionError("canonical service is stopped")
        pinned = copy.deepcopy(self.identities[name])
        bridge = self

        class Peer:
            identity = pinned
            request_nonce: str | None = None

            def assert_current(self) -> None:
                bridge.verify(name, facts, pinned)

            def request(self, request: dict[str, Any]) -> dict[str, Any]:
                self.request_nonce = request["nonce"]
                bridge.requests.append(copy.deepcopy(request))
                return bridge.payload(name, request)

            def finish(self, nonce: str) -> None:
                assert nonce == self.request_nonce
                if name in bridge.ack_failure:
                    raise TimeoutError("service cleanup acknowledgement unavailable")
                bridge.acknowledged.append(name)

        try:
            yield Peer()
        finally:
            self.closed.append(name)

    def verify(self, name: str, facts: dict[str, Any], pinned: dict[str, Any]) -> None:
        assert facts == self.facts
        if name in self.restarted or self.identities[name] != pinned:
            raise RuntimeError("canonical service restarted")

    def inspect(self, _facts: dict[str, Any]) -> dict[str, Any]:
        return {
            key: copy.deepcopy(self.facts[key]) for key in ("host_identity", "paths", "listeners")
        } | {"postgresql_service": copy.deepcopy(self.facts["services"][0])}

    def backend(self, pid: int) -> dict[str, Any]:
        return {
            "process_id": pid,
            "parent_process_id": 11,
            "process_path": self.facts["listeners"][0]["process_path"],
            "process_started": "2026-10-16T00:00:02+00:00",
        }

    def collect(self) -> tuple[Any, dict[str, Any]]:
        receipt = evidence.collect_service_postgresql(self.facts)
        return receipt, passive.summarize_passive_postgresql(receipt, self.facts)


@pytest.fixture
def bridge(monkeypatch: pytest.MonkeyPatch) -> EmulatedBridge:
    bridge = EmulatedBridge()
    monkeypatch.setattr(evidence, "_require_enrollment", lambda _: None)
    monkeypatch.setattr(evidence.pipe, "connect_service", bridge.connect)
    monkeypatch.setattr(evidence.pipe, "verify_service", bridge.verify)
    monkeypatch.setattr(evidence, "_inspect_host", bridge.inspect)
    monkeypatch.setattr(passive, "_process_details", bridge.backend)
    return bridge


@pytest.mark.parametrize(
    "enrollment",
    [
        None,
        {"state": "absent", "files": []},
        {"state": "present", "committed": True, "accepted": True, "status": "PASS"},
    ],
    ids=["missing", "absent", "asserted-marker"],
)
def test_real_enrollment_gate_blocks_unavailable_authority_before_host_inspection_and_ipc(
    monkeypatch: pytest.MonkeyPatch, enrollment: dict[str, Any] | None
) -> None:
    facts = observation()
    if enrollment is not None:
        facts["enrollment"] = enrollment
    private = "private accepted-claim-path C:\\Security\\enrollment password=secret"
    calls: list[Any] = []
    connects: list[Any] = []
    inspections: list[Any] = []

    def unavailable(received: dict[str, Any]) -> None:
        calls.append(received)
        raise RuntimeError(private)

    def forbidden_connect(*args: Any) -> Any:
        connects.append(args)
        pytest.fail("unaccepted enrollment reached the authenticated service pipe")

    def forbidden_inspection(*args: Any) -> Any:
        inspections.append(args)
        pytest.fail("unaccepted enrollment reached native host inspection")

    monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", unavailable)
    monkeypatch.setattr(evidence.pipe, "connect_service", forbidden_connect)
    monkeypatch.setattr(evidence, "_inspect_host", forbidden_inspection)
    receipt = evidence.collect_service_postgresql(facts)
    summary = passive.summarize_passive_postgresql(receipt, facts)
    assert calls == [facts]
    assert calls[0] is facts
    assert connects == []
    assert inspections == []
    assert summary["status"] == "BLOCKED"
    assert all(item["status"] == "BLOCKED" for item in summary["components"].values())
    assert private not in json.dumps(summary)
    assert "committed enrollment is unavailable" in json.dumps(summary)


def test_real_gate_delegates_to_canonical_qualifier_with_the_exact_observation(
    bridge: EmulatedBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    monkeypatch.setattr(evidence, "_require_enrollment", REAL_REQUIRE_ENROLLMENT)
    monkeypatch.setattr(qualification, "_confirmed_legal_enrollment", calls.append)
    _receipt, summary = bridge.collect()
    assert calls == [bridge.facts]
    assert calls[0] is bridge.facts
    assert summary["components"]["service_authentication"]["status"] == "PASS"
    assert summary["components"]["cross_role_authentication"]["status"] == "PASS"


def test_two_authenticated_services_produce_bound_readonly_evidence(bridge: EmulatedBridge) -> None:
    receipt, summary = bridge.collect()
    assert summary["status"] == "BLOCKED"
    assert summary["components"]["active_hba"]["status"] == "BLOCKED"
    for component in set(passive.COMPONENTS) - {"active_hba"}:
        passive.require_passive_postgresql_component(receipt, bridge.facts, component)
    assert set(bridge.connects) == set(SERVICES)
    assert set(bridge.closed) == set(SERVICES)
    assert set(bridge.acknowledged) == set(SERVICES)
    assert len({request["nonce"] for request in bridge.requests}) == 2
    assert len({request["collection_id"] for request in bridge.requests}) == 1


@pytest.mark.parametrize("field", ["nonce", "collection_id"])
def test_replayed_or_other_collection_payload_has_no_authority(
    bridge: EmulatedBridge, field: str
) -> None:
    bridge.mutate = lambda _name, payload: payload.update({field: "0" * 64})
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert set(bridge.closed) == set(bridge.connects)


@pytest.mark.parametrize(
    "field,value",
    [
        ("process_id", 999),
        ("token_sid", "S-1-5-18"),
        ("process_path", r"C:\Temp\CryptoHunterBackend.exe"),
        ("process_started", "2026-10-16T00:00:30+00:00"),
    ],
)
def test_payload_process_cannot_replace_authenticated_pipe_peer(
    bridge: EmulatedBridge, field: str, value: Any
) -> None:
    bridge.mutate = lambda _name, payload: payload["process"].update({field: value})
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


@pytest.mark.parametrize("mode", ["missing", "stopped"])
def test_unavailable_verifier_leaves_authentication_and_active_hba_unproved(
    bridge: EmulatedBridge, mode: str
) -> None:
    if mode == "missing":
        bridge.unavailable.add(CONTRACT.verifier_service)
    else:
        bridge.facts["services"][2]["state"] = "Stopped"
        bridge.facts["services"][2]["process_id"] = 0
        bridge.facts["services"][2]["second_process_id"] = 0
        bridge.facts["services"][2]["second_state"] = 1
    _receipt, summary = bridge.collect()
    assert summary["status"] != "PASS"
    for name in ("service_authentication", "cross_role_authentication", "active_hba"):
        assert summary["components"][name]["status"] != "PASS"


@pytest.mark.parametrize("field", ["control", "identity"])
def test_services_must_observe_the_same_system_and_postmaster(
    bridge: EmulatedBridge, field: str
) -> None:
    def mutate(name: str, payload: dict[str, Any]) -> None:
        if name != CONTRACT.verifier_service:
            return
        index = 0 if field == "control" else 6
        value = "7448802115599552301" if field == "control" else "2026-10-16T00:00:03+00:00"
        for batch in payload["queries"][field]:
            batch["rows"][0][index] = value
        payload["post_probe"][field]["rows"][0][index] = value

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"


@pytest.mark.parametrize("sqlstate", [None, "08006", "57014", "42501"])
def test_network_or_nonauthentication_error_never_proves_cross_role_denial(
    bridge: EmulatedBridge, sqlstate: str | None
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        for probe in payload["probes"].values():
            probe.update(status="BLOCKED", sqlstate=sqlstate)

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["components"]["cross_role_authentication"]["status"] != "PASS"
    assert summary["components"]["active_hba"]["status"] != "PASS"


@pytest.mark.parametrize("sqlstate", ["42501", "08006", "42P01"])
def test_catalog_errors_keep_sqlstate_policy_and_export_no_server_diagnostics(
    bridge: EmulatedBridge, sqlstate: str
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["queries"]["hba"] = [{"sqlstate": sqlstate}]

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["components"]["parsed_hba"]["status"] == (
        "FAIL" if sqlstate == "42P01" else "BLOCKED"
    )


def test_missing_cleanup_acknowledgement_cannot_issue_passing_evidence(
    bridge: EmulatedBridge,
) -> None:
    bridge.ack_failure.add(CONTRACT.verifier_service)
    _receipt, summary = bridge.collect()
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert set(bridge.closed) == set(bridge.connects)


def test_previously_authenticated_wire_response_cannot_be_replayed(
    bridge: EmulatedBridge,
) -> None:
    _receipt, summary = bridge.collect()
    assert summary["components"]["cross_role_authentication"]["status"] == "PASS"
    captured = {payload["service"]: payload for payload in bridge.payloads}

    def replay(name: str, payload: dict[str, Any]) -> None:
        payload.clear()
        payload.update(copy.deepcopy(captured[name]))

    bridge.mutate = replay
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


@pytest.mark.parametrize("sqlstate", [None, "08006", "57014"])
def test_claimed_denial_with_nonauthentication_sqlstate_is_rejected(
    bridge: EmulatedBridge, sqlstate: str | None
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["probes"]["postgres"].update(status="DENIED", sqlstate=sqlstate)

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert summary["components"]["cross_role_authentication"]["status"] == "FAIL"


@pytest.mark.parametrize("sqlstate", [True, 28000, 28000.0, [], {}, ["28000"], "2800", "28p01"])
def test_cross_role_probe_rejects_invalid_sqlstate_types_and_shapes(
    bridge: EmulatedBridge, sqlstate: Any
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["probes"]["postgres"].update(status="BLOCKED", sqlstate=sqlstate)

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


@pytest.mark.parametrize("sqlstate", ["28000", "28P01", "08006", "42501"])
def test_accepted_probe_cannot_also_claim_a_server_error_sqlstate(
    bridge: EmulatedBridge, sqlstate: str
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["probes"]["postgres"].update(status="ACCEPTED", sqlstate=sqlstate)

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


def test_post_probe_schema_failure_survives_later_permission_denial(
    bridge: EmulatedBridge,
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["post_probe"]["identity"] = {"sqlstate": "42P01"}
        payload["post_probe"]["ssl"] = {"sqlstate": "42501"}

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert summary["components"]["cross_role_authentication"]["status"] == "FAIL"


def test_actual_forbidden_role_acceptance_contradicts_authentication_separation(
    bridge: EmulatedBridge,
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["probes"]["postgres"].update(status="ACCEPTED", sqlstate=None)

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert summary["components"]["cross_role_authentication"]["status"] == "FAIL"


@pytest.mark.parametrize(
    "query,index,value",
    [
        ("identity", 5, 999),
        ("identity", 6, "2026-10-16T00:00:03+00:00"),
        ("control", 0, "7448802115599552301"),
        ("data", 0, r"C:\Other\PostgreSQL\Data"),
        ("ssl", 4, "/CN=OtherService"),
    ],
)
def test_post_probe_session_instance_and_certificate_must_remain_unchanged(
    bridge: EmulatedBridge, query: str, index: int, value: Any
) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["post_probe"][query]["rows"][0][index] = value

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


def test_capture_trace_cannot_add_an_unapproved_query(bridge: EmulatedBridge) -> None:
    def mutate(_name: str, payload: dict[str, Any]) -> None:
        payload["queries"]["arbitrary_sql"] = [{"rows": []}]

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())


def test_contradictory_catalog_evidence_survives_later_transport_failure(
    bridge: EmulatedBridge,
) -> None:
    def mutate(name: str, payload: dict[str, Any]) -> None:
        if name == CONTRACT.backend_service:
            payload["queries"]["hba"] = [{"sqlstate": "42P01"}]
        else:
            raise TimeoutError("second service transport unavailable")

    bridge.mutate = mutate
    _receipt, summary = bridge.collect()
    assert summary["status"] == "FAIL"
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert set(bridge.closed) == set(bridge.connects)


@pytest.mark.parametrize("phase", ["during_capture", "receipt_use"])
def test_native_postgresql_restart_revokes_evidence_even_if_observation_is_unchanged(
    bridge: EmulatedBridge, monkeypatch: pytest.MonkeyPatch, phase: str
) -> None:
    native = bridge.inspect(bridge.facts)
    monkeypatch.setattr(evidence, "_inspect_host", lambda _: copy.deepcopy(native))
    if phase == "during_capture":

        def restart(_name: str, _payload: dict[str, Any]) -> None:
            native["listeners"][0]["process_id"] += 1

        bridge.mutate = restart
        _receipt, summary = bridge.collect()
        assert summary["status"] == "FAIL"
        assert all(item["status"] != "PASS" for item in summary["components"].values())
    else:
        receipt, summary = bridge.collect()
        assert summary["components"]["cross_role_authentication"]["status"] == "PASS"
        native["listeners"][0]["process_id"] += 1
        assert passive.summarize_passive_postgresql(receipt, bridge.facts)["status"] == "UNVERIFIED"


def test_expired_response_is_not_accepted_even_when_the_wire_facts_are_valid(
    bridge: EmulatedBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = {"now": 100.0}
    monkeypatch.setattr(evidence.time, "monotonic", lambda: clock["now"])

    def expire(_name: str, _payload: dict[str, Any]) -> None:
        clock["now"] += 31.0

    bridge.mutate = expire
    _receipt, summary = bridge.collect()
    assert all(item["status"] != "PASS" for item in summary["components"].values())
    assert set(bridge.closed) == set(bridge.connects)


def test_service_restart_after_collection_revokes_receipt(bridge: EmulatedBridge) -> None:
    receipt, summary = bridge.collect()
    assert summary["components"]["cross_role_authentication"]["status"] == "PASS"
    bridge.restarted.add(CONTRACT.backend_service)
    assert passive.summarize_passive_postgresql(receipt, bridge.facts)["status"] == "UNVERIFIED"


def test_saved_reports_copied_observations_and_expiry_carry_no_live_authority(
    bridge: EmulatedBridge, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt, summary = bridge.collect()
    assert summary["components"]["cross_role_authentication"]["status"] == "PASS"
    for saved in (copy.deepcopy(summary), json.loads(json.dumps(summary))):
        assert passive.summarize_passive_postgresql(saved, bridge.facts)["status"] == "UNVERIFIED"
    assert (
        passive.summarize_passive_postgresql(receipt, copy.deepcopy(bridge.facts))["status"]
        == "UNVERIFIED"
    )
    created = passive._RECEIPTS[receipt]["created"]
    monkeypatch.setattr(passive.time, "monotonic", lambda: created + passive._MAX_AGE + 1)
    assert passive.summarize_passive_postgresql(receipt, bridge.facts)["status"] == "UNVERIFIED"
