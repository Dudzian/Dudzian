from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from ui.pyside_app import preview_block_p_desktop_exe_build_readiness_matrix as matrix
import ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract as contract
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_matrix import (
    build_preview_block_p_desktop_exe_build_readiness_matrix,
)

EXPECTED_DECISION_FIELDS = [
    "source_matrix_exists_and_accepted",
    "contract_definitions_complete",
    "contract_satisfied",
    "no_readiness_row_satisfied",
    "no_blocker_resolved",
    "no_evidence_collected",
    "no_evidence_validated",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_enabled",
    "orders_enabled",
    "only_source_only_18_7_handoff_allowed",
    "build_performed_by_18_6",
    "packaging_authorized_by_18_6",
    "build_authorized_by_18_6",
    "artifact_creation_authorized_by_18_6",
    "release_performed_by_18_6",
    "runtime_enabled_by_18_6",
    "orders_enabled_by_18_6",
]
EXPECTED_CONTRACT_BOUNDARY_FIELDS = [
    "reads_18_5_only",
    "source_only",
    "plain_data",
    "static_contract",
    "can_feed_only_18_7_build_readiness_read_model",
    "repo_rescan",
    "filesystem_scan",
    "environment_scan",
    "secret_file_read",
    "dependency_import",
    "dependency_resolution",
    "pyside_import",
    "qml_load",
    "qt_plugin_discovery",
    "entrypoint_selection",
    "entrypoint_validation",
    "evidence_collection",
    "evidence_validation",
    "blocker_resolution",
    "readiness_approval",
    "build_gate_approval",
    "build_readiness_grant",
    "packaging_authorization",
    "build_authorization",
    "spec_file_creation",
    "build_command_creation",
    "build_command_execution",
    "packaging_performed",
    "artifact_created",
    "artifact_scanned",
    "artifact_signed",
    "installer_created",
    "release_performed",
    "runtime_started",
    "orders_enabled",
    "network_opened",
    "credentials_read",
    "qml_bridge_gateway_controller_changed",
]
BY_18_6_FIELDS = [
    "build_performed_by_18_6",
    "packaging_authorized_by_18_6",
    "build_authorized_by_18_6",
    "artifact_creation_authorized_by_18_6",
    "release_performed_by_18_6",
    "runtime_enabled_by_18_6",
    "orders_enabled_by_18_6",
]


class HashBomb:
    hash_calls = 0

    def __hash__(self) -> int:
        type(self).hash_calls += 1
        raise RuntimeError


def test_real_18_5_builder_is_accepted(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    sources: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    def builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        source = build_preview_block_p_desktop_exe_build_readiness_matrix()
        sources.append(source)
        snapshots.append(copy.deepcopy(source))
        return source

    monkeypatch.setattr(matrix, "READINESS_ROWS", copy.deepcopy(contract.SOURCE_ROWS))
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", builder
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()

    assert calls == 1
    assert len(sources) == 1
    assert len(snapshots) == 1
    assert sources[0] == snapshots[0]
    assert contract._source_accepted(sources[0]) is True
    assert contract._integrity(payload) is True
    assert json.dumps(payload)
    assert payload["status"] == contract.STATUS
    assert payload["source_18_5_accepted"] is True
    assert payload["build_readiness_contract_artifact_complete"] is True
    assert payload["ready_for_block_p_7"] is True
    assert len(payload["build_readiness_contract_rows"]) == 17
    assert len(payload["build_readiness_acceptance_rules"]) == 6

    summary = payload["build_readiness_contract_summary"]
    assert summary["unique_requirement_count"] == 8
    assert summary["unique_blocker_count"] == 12
    assert summary["unique_evidence_count"] == 12
    assert summary["readiness_clause_count"] == 17
    assert summary["defined_readiness_clause_count"] == 17
    assert summary["satisfied_readiness_clause_count"] == 0
    assert summary["observed_readiness_count"] == 0
    assert summary["validated_readiness_count"] == 0
    assert summary["satisfied_readiness_count"] == 0
    assert summary["ready_readiness_count"] == 0
    assert summary["acceptance_rule_count"] == 6
    assert summary["satisfied_acceptance_rule_count"] == 0
    assert summary["packaging_authorized"] is False
    assert summary["build_authorized"] is False
    assert summary["artifact_creation_authorized"] is False
    assert summary["release_authorized"] is False
    assert summary["runtime_authorized"] is False
    assert summary["orders_authorized"] is False

    for row in payload["build_readiness_contract_rows"]:
        assert row["contract_clause_defined"] is True
        assert row["contract_clause_satisfied"] is False
        assert row["build_readiness_granted"] is False
        assert row["packaging_authorization_granted"] is False
        assert row["build_authorization_granted"] is False
        assert row["artifact_creation_authorization_granted"] is False

    capability_state = payload["capability_contract_state"]
    assert set(capability_state["source_capability_build_readiness_state"].values()) == {"blocked"}
    assert set(capability_state["contract_capability_state"].values()) == {"blocked"}


def test_nominal_decision_and_boundaries_have_canonical_order() -> None:
    payload = contract._nominal()
    decision = payload["fail_closed_build_readiness_contract_decision"]
    boundaries = payload["contract_boundaries"]

    assert list(decision) == EXPECTED_DECISION_FIELDS
    assert len(decision) == 22
    assert list(boundaries) == EXPECTED_CONTRACT_BOUNDARY_FIELDS
    assert len(boundaries) == 38
    for field in EXPECTED_CONTRACT_BOUNDARY_FIELDS[:5]:
        assert boundaries[field] is True
    for field in EXPECTED_CONTRACT_BOUNDARY_FIELDS[5:]:
        assert boundaries[field] is False
    for field in BY_18_6_FIELDS:
        assert decision[field] is False


def test_hash_bomb_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    source = copy.deepcopy(contract._source_template())
    source["readiness_rows"][0]["source_requirement_ids"][0] = HashBomb()
    HashBomb.hash_calls = 0
    assert contract._source_accepted(source) is False
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert payload == contract._blocked()
    assert HashBomb.hash_calls == 0


def test_blocked_is_exact() -> None:
    blocked = contract._blocked()
    assert blocked == contract._blocked()
    assert contract._integrity(blocked) is True
    assert json.dumps(blocked)

    reordered = dict(reversed(list(blocked.items())))
    assert contract._integrity(reordered) is False

    blocked["extra"] = False
    assert contract._integrity(blocked) is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda source: source.__setitem__("ready_for_build_execution", True),
        lambda source: source.__setitem__("integrity_valid", False),
        lambda source: source.__setitem__("source_18_4_accepted", False),
        lambda source: source.__setitem__("status", "wrong"),
        lambda source: source.__setitem__("unexpected_top_level_field", False),
        lambda source: source.pop("integrity_valid"),
        lambda source: (
            source.clear()
            or source.update(dict(reversed(list(contract._source_template().items()))))
        ),
    ],
)
def test_source_top_level_tampering_returns_canonical_blocked(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    source = copy.deepcopy(contract._source_template())
    mutate(source)
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_build_readiness_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_build_readiness_contract()
    assert payload == contract._blocked()
    assert contract._integrity(payload) is True


def test_exact_plain_accepts_matching_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.append(trusted)
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_lists() -> None:
    actual: list[object] = []
    trusted: list[object] = []
    actual.append(actual)
    trusted.extend([trusted, "extra"])
    assert contract._exact_plain(actual, trusted) is False


def test_exact_plain_accepts_matching_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    assert contract._exact_plain(actual, trusted) is True


def test_exact_plain_rejects_different_cyclic_dicts() -> None:
    actual: dict[str, Any] = {}
    trusted: dict[str, Any] = {}
    actual["self"] = actual
    trusted["self"] = trusted
    trusted["extra"] = False
    assert contract._exact_plain(actual, trusted) is False


def test_blocked_rejects_all_top_level_mutations() -> None:
    blocked = contract._blocked()
    assert blocked["source_18_5_accepted"] is False
    for field in blocked:
        payload = copy.deepcopy(blocked)
        payload.pop(field)
        assert contract._integrity(payload) is False
    for field in (
        "source_18_5_accepted",
        "build_readiness_contract_artifact_complete",
        "ready_for_block_p_7",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "integrity_valid",
    ):
        payload = copy.deepcopy(blocked)
        payload[field] = not payload[field]
        assert contract._integrity(payload) is False


def test_main_sections_reject_missing_extra_and_reordered_fields() -> None:
    sections = [
        "block_p_desktop_exe_build_readiness_matrix_reference",
        "source_matrix_preservation",
        "build_readiness_contract_summary",
        "build_readiness_contract_principles",
        "capability_contract_state",
        "fail_closed_build_readiness_contract_decision",
        "non_execution_contract_evidence",
        "contract_boundaries",
        "source_boundaries",
    ]
    for name in sections:
        payload = copy.deepcopy(contract._nominal())
        section = payload[name]
        section.pop(next(iter(section)))
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name]["unexpected_test_field"] = False
        assert contract._integrity(payload) is False
        payload = copy.deepcopy(contract._nominal())
        payload[name] = dict(reversed(list(payload[name].items())))
        assert contract._integrity(payload) is False
