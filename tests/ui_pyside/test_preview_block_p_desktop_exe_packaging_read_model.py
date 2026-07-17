from __future__ import annotations

import json
from typing import Any

import pytest

from ui.pyside_app import preview_block_p_desktop_exe_packaging_read_model as read_model
from ui.pyside_app.preview_block_p_desktop_exe_packaging_contract import (
    build_preview_block_p_desktop_exe_packaging_contract,
)


def _payload() -> dict[str, Any]:
    return read_model.build_preview_block_p_desktop_exe_packaging_read_model()


def test_expected_source_is_the_accepted_18_3_snapshot() -> None:
    assert read_model.EXPECTED_SOURCE == build_preview_block_p_desktop_exe_packaging_contract()
    assert list(read_model.EXPECTED_SOURCE) == read_model.TOP_LEVEL_FIELDS_18_3
    assert read_model.SOURCE_IDENTITY_EXPECTED == {
        key: read_model.EXPECTED_SOURCE[key] for key in read_model.SOURCE_IDENTITY_EXPECTED
    }


def test_nominal_read_model_is_plain_complete_and_fail_closed() -> None:
    payload = _payload()
    assert list(payload) == read_model.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == read_model.SCHEMA_VERSION
    assert payload["block"] == "P"
    assert payload["step"] == "18.4"
    assert payload["status"] == read_model.STATUS
    assert (
        payload["block_p_desktop_exe_packaging_read_model_status"]
        == read_model.PACKAGING_READ_MODEL_STATUS
    )
    assert (
        payload["block_p_desktop_exe_packaging_read_model_decision"]
        == read_model.PACKAGING_READ_MODEL_STATUS.upper()
    )
    assert payload["packaging_read_model_artifact_complete"] is True
    assert payload["ready_for_block_p_5"] is True
    assert [
        len(payload[name])
        for name in (
            "domain_contract_read_model_rows",
            "scope_contract_read_model_rows",
            "requirement_contract_read_model_rows",
            "blocker_read_model_rows",
            "evidence_read_model_rows",
            "acceptance_rule_read_model_rows",
        )
    ] == [6, 3, 8, 12, 12, 6]
    assert payload["packaging_read_model_summary"]["contract_satisfied"] is False
    assert payload["packaging_contract_overview"]["build_ready"] is False
    assert payload["non_execution_read_model_evidence"]["source_builder_call_count"] == 1
    json.dumps(payload, sort_keys=True)


def test_exact_handoff_and_all_operational_flags_are_false() -> None:
    payload = _payload()
    assert payload["future_steps"][0]["step"] == "18.5"
    assert payload["read_model_boundaries"]["can_feed_only_18_5_build_readiness_matrix"] is True
    assert all(
        value is False
        for key, value in payload["read_model_boundaries"].items()
        if key
        not in {
            "reads_18_3_only",
            "source_only",
            "plain_data",
            "static_read_model",
            "can_feed_only_18_5_build_readiness_matrix",
        }
    )
    assert all(
        value == "blocked"
        for value in payload["capability_read_model_state"][
            "packaging_read_model_capabilities"
        ].values()
    )


def test_invalid_source_is_blocked_without_mutation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    source["packaging_contract_summary"]["build_ready"] = True
    snapshot = json.dumps(source, sort_keys=True)
    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_packaging_contract", lambda: source
    )
    payload = _payload()
    assert payload["status"] == read_model.BLOCKED_STATUS
    assert payload["ready_for_block_p_5"] is False
    assert payload["non_execution_read_model_evidence"]["contract_summary_valid"] is False
    assert json.dumps(source, sort_keys=True) == snapshot


def test_builder_is_called_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    source = build_preview_block_p_desktop_exe_packaging_contract()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(read_model, "build_preview_block_p_desktop_exe_packaging_contract", fake)
    assert _payload()["non_execution_read_model_evidence"]["source_builder_call_count"] == 1
    assert calls == 1


def test_graph_filters_dangling_links_when_dependency_sections_are_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (
        ("contract_evidence_requirement_rows", "evidence_read_model_rows", "required_evidence_ids"),
        (
            "unresolved_blocker_contract_rows",
            "blocker_read_model_rows",
            "source_unresolved_condition_ids",
        ),
        (
            "packaging_scope_contract_rows",
            "scope_contract_read_model_rows",
            "source_affected_scope_ids",
        ),
        (
            "packaging_requirement_contract_rows",
            "requirement_contract_read_model_rows",
            "contract_requirement_ids",
        ),
    )
    for section, output_family, link in cases:
        source = build_preview_block_p_desktop_exe_packaging_contract()
        source[section] = []
        monkeypatch.setattr(
            read_model, "build_preview_block_p_desktop_exe_packaging_contract", lambda: source
        )
        payload = _payload()
        assert payload["status"] == read_model.BLOCKED_STATUS
        assert payload[output_family] == []
        assert (
            payload["non_execution_read_model_evidence"]["output_referential_integrity_valid"]
            is True
        )
        if section == "contract_evidence_requirement_rows":
            for family in (
                "blocker_read_model_rows",
                "scope_contract_read_model_rows",
                "requirement_contract_read_model_rows",
                "acceptance_rule_read_model_rows",
            ):
                assert all(not row["required_evidence_ids"] for row in payload[family])
        if section == "unresolved_blocker_contract_rows":
            assert payload["evidence_read_model_rows"] == []
            for family, field in (
                ("scope_contract_read_model_rows", "source_unresolved_blocker_ids"),
                ("requirement_contract_read_model_rows", "source_unresolved_condition_ids"),
                ("acceptance_rule_read_model_rows", "required_blocker_ids"),
            ):
                assert all(not row[field] for row in payload[family])
        if section == "packaging_scope_contract_rows":
            assert all(not row[link] for row in payload["blocker_read_model_rows"])
        if section == "packaging_requirement_contract_rows":
            assert all(not row[link] for row in payload["blocker_read_model_rows"])


def test_blocker_contract_requirement_links_are_preserved_exactly() -> None:
    payload = _payload()
    source = build_preview_block_p_desktop_exe_packaging_contract()
    assert [row["contract_requirement_ids"] for row in payload["blocker_read_model_rows"]] == [
        row["contract_requirement_ids"] for row in source["unresolved_blocker_contract_rows"]
    ]
    assert all(
        len(row["contract_requirement_ids"]) == 1 for row in payload["blocker_read_model_rows"]
    )


def test_graph_states_are_isolated_from_non_graph_capability_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    source["real_capability_contract_state"] = {}
    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_packaging_contract", lambda: source
    )
    payload = _payload()
    evidence = payload["non_execution_read_model_evidence"]
    assert payload["status"] == read_model.BLOCKED_STATUS
    assert evidence["source_graph_integrity_valid"] is True
    assert evidence["output_graph_integrity_valid"] is True
    assert evidence["contract_definition_graph_complete"] is True
    assert payload["source_contract_preservation"]["referential_integrity_preserved"] is True
    assert payload["packaging_read_model_summary"]["contract_definitions_complete"] is True
    assert (
        payload["packaging_contract_overview"]["read_model_state"]
        == "contract_defined_handoff_blocked_by_non_graph_source"
    )
    assert payload["packaging_contract_overview"]["next_required_source_only_step"] == ""


def test_incomplete_evidence_graph_blocks_acceptance_definitions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    source["contract_evidence_requirement_rows"] = []
    monkeypatch.setattr(
        read_model, "build_preview_block_p_desktop_exe_packaging_contract", lambda: source
    )
    payload = _payload()
    evidence = payload["non_execution_read_model_evidence"]
    assert evidence["source_graph_integrity_valid"] is False
    assert evidence["output_graph_integrity_valid"] is True
    assert evidence["contract_definition_graph_complete"] is False
    assert payload["source_contract_preservation"]["referential_integrity_preserved"] is False
    assert (
        payload["packaging_contract_overview"]["read_model_state"]
        == "contract_overview_blocked_by_invalid_source"
    )
    assert payload["packaging_contract_overview"]["next_required_source_only_step"] == ""
    assert all(
        row["source_blocker_preserved"] is False for row in payload["blocker_read_model_rows"]
    )
    assert all(row["rule_defined"] is False for row in payload["acceptance_rule_read_model_rows"])


def test_source_graph_validator_rejects_identifier_and_link_breaks() -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    names = (
        "packaging_scope_contract_rows",
        "packaging_requirement_contract_rows",
        "unresolved_blocker_contract_rows",
        "contract_evidence_requirement_rows",
        "contract_acceptance_rule_rows",
    )
    rows = [read_model._copy_plain(source[name]) for name in names]
    assert read_model._source_referential_integrity_valid(*rows) is True
    mutations = (
        (0, 0, "scope_id", ""),
        (1, 0, "contract_requirement_id", ""),
        (2, 0, "blocker_id", ""),
        (3, 0, "blocker_id", "missing"),
        (4, 0, "acceptance_rule_id", ""),
        (0, 0, "source_unresolved_blocker_ids", ["missing"]),
        (1, 0, "contract_clause_ids", ["missing"]),
        (2, 0, "required_evidence_ids", ["missing"]),
        (4, 0, "required_blocker_ids", ["missing"]),
    )
    for family, index, field, value in mutations:
        altered = [read_model._copy_plain(source[name]) for name in names]
        altered[family][index][field] = value
        assert read_model._source_referential_integrity_valid(*altered) is False


def test_semantic_pairing_rejects_existing_but_swapped_relationships() -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    names = (
        "packaging_scope_contract_rows",
        "packaging_requirement_contract_rows",
        "unresolved_blocker_contract_rows",
        "contract_evidence_requirement_rows",
        "contract_acceptance_rule_rows",
    )
    for family, field in (
        (3, "blocker_id"),
        (2, "required_evidence_ids"),
        (2, "contract_requirement_ids"),
        (0, "contract_clause_ids"),
        (1, "required_evidence_ids"),
    ):
        rows = [read_model._copy_plain(source[name]) for name in names]
        other = 2 if field == "contract_requirement_ids" else 1
        rows[family][0][field], rows[family][other][field] = (
            rows[family][other][field],
            rows[family][0][field],
        )
        assert read_model._source_referential_integrity_valid(*rows) is False


def test_output_integrity_rejects_semantic_swaps() -> None:
    for family, field in (
        ("evidence_read_model_rows", "blocker_id"),
        ("blocker_read_model_rows", "required_evidence_ids"),
        ("blocker_read_model_rows", "contract_requirement_ids"),
        ("scope_contract_read_model_rows", "contract_clause_ids"),
        ("requirement_contract_read_model_rows", "required_evidence_ids"),
    ):
        payload = _payload()
        rows = payload[family]
        other = 2 if field == "contract_requirement_ids" else 1
        rows[0][field], rows[other][field] = rows[other][field], rows[0][field]
        assert read_model._output_integrity(payload) is False


def test_semantic_validator_rejects_parallel_length_and_acceptance_swaps() -> None:
    source = build_preview_block_p_desktop_exe_packaging_contract()
    names = (
        "packaging_scope_contract_rows",
        "packaging_requirement_contract_rows",
        "unresolved_blocker_contract_rows",
        "contract_evidence_requirement_rows",
        "contract_acceptance_rule_rows",
    )
    for family, field in (
        (0, "contract_clause_ids"),
        (0, "required_evidence_ids"),
        (1, "contract_clause_ids"),
        (1, "required_evidence_ids"),
    ):
        rows = [read_model._copy_plain(source[name]) for name in names]
        rows[family][0][field].append(rows[family][0][field][0])
        assert read_model._source_referential_integrity_valid(*rows) is False
    rows = [read_model._copy_plain(source[name]) for name in names]
    rows[4][0]["required_blocker_ids"], rows[4][2]["required_blocker_ids"] = (
        rows[4][2]["required_blocker_ids"],
        rows[4][0]["required_blocker_ids"],
    )
    assert read_model._source_referential_integrity_valid(*rows) is False


def test_output_integrity_rejects_clause_swap_and_family_reorder() -> None:
    payload = _payload()
    rows = payload["blocker_read_model_rows"]
    rows[0]["contract_clause_id"], rows[1]["contract_clause_id"] = (
        rows[1]["contract_clause_id"],
        rows[0]["contract_clause_id"],
    )
    assert read_model._output_integrity(payload) is False
    for family in (
        "scope_contract_read_model_rows",
        "requirement_contract_read_model_rows",
        "blocker_read_model_rows",
        "evidence_read_model_rows",
        "acceptance_rule_read_model_rows",
    ):
        payload = _payload()
        payload[family][0], payload[family][1] = payload[family][1], payload[family][0]
        assert read_model._output_integrity(payload) is False


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("domain_contract_read_model_rows", "build_ready", True),
        ("scope_contract_read_model_rows", "scope_ready", True),
        ("requirement_contract_read_model_rows", "build_authorized", True),
        ("blocker_read_model_rows", "resolved", True),
        ("blocker_read_model_rows", "evidence_validated", True),
        ("evidence_read_model_rows", "collected", True),
        ("evidence_read_model_rows", "validated", True),
        ("acceptance_rule_read_model_rows", "rule_satisfied", True),
        ("acceptance_rule_read_model_rows", "grants_build_authorization", True),
    ],
)
def test_output_integrity_rejects_operational_grants(
    family: str, field: str, value: object
) -> None:
    payload = _payload()
    payload[family][0][field] = value
    assert read_model._output_integrity(payload) is False


@pytest.mark.parametrize(
    "family",
    [
        "domain_contract_read_model_rows",
        "scope_contract_read_model_rows",
        "requirement_contract_read_model_rows",
        "blocker_read_model_rows",
        "evidence_read_model_rows",
        "acceptance_rule_read_model_rows",
    ],
)
def test_output_integrity_requires_exact_row_schema(family: str) -> None:
    payload = _payload()
    row = payload[family][0]
    value = row.pop(next(reversed(row)))
    row["unexpected"] = value
    assert read_model._output_integrity(payload) is False


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("scope_contract_read_model_rows", "read_model_state", "wrong"),
        ("scope_contract_read_model_rows", "read_model_result", "wrong"),
        ("scope_contract_read_model_rows", "failure_policy", "open"),
        ("requirement_contract_read_model_rows", "read_model_state", "wrong"),
        ("requirement_contract_read_model_rows", "read_model_result", "wrong"),
        ("requirement_contract_read_model_rows", "failure_policy", "open"),
        ("blocker_read_model_rows", "read_model_severity", "low"),
        ("blocker_read_model_rows", "read_model_state", "wrong"),
        ("blocker_read_model_rows", "read_model_result", "wrong"),
        ("blocker_read_model_rows", "failure_policy", "open"),
        ("evidence_read_model_rows", "read_model_state", "wrong"),
        ("evidence_read_model_rows", "read_model_result", "wrong"),
        ("evidence_read_model_rows", "failure_policy", "open"),
        ("acceptance_rule_read_model_rows", "failure_policy", "open"),
    ],
)
def test_output_integrity_rejects_state_tampering(family: str, field: str, value: object) -> None:
    payload = _payload()
    payload[family][0][field] = value
    assert read_model._output_integrity(payload) is False


class _Text(str):
    pass


class _Links(list[str]):
    pass


class _Row(dict[str, object]):
    pass


@pytest.mark.parametrize(
    ("family", "field", "value"),
    [
        ("domain_contract_read_model_rows", "domain_id", _Text("desktop_application_entrypoint")),
        (
            "scope_contract_read_model_rows",
            "source_unresolved_blocker_ids",
            _Links(
                [
                    "final_desktop_entrypoint_not_selected",
                    "desktop_entrypoint_validation_not_performed",
                ]
            ),
        ),
        ("domain_contract_read_model_rows", "unresolved_condition_count", True),
        ("domain_contract_read_model_rows", "source_contract_preserved", 1),
    ],
)
def test_output_integrity_rejects_exact_type_confusion(
    family: str, field: str, value: object
) -> None:
    payload = _payload()
    payload[family][0][field] = value
    assert read_model._output_integrity(payload) is False


def test_output_integrity_rejects_dict_subclass_row() -> None:
    payload = _payload()
    payload["domain_contract_read_model_rows"][0] = _Row(
        payload["domain_contract_read_model_rows"][0]
    )
    assert read_model._output_integrity(payload) is False
