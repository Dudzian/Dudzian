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
