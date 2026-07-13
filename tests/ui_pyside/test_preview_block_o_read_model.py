from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_o_entry_contract import build_preview_block_o_entry_contract
from ui.pyside_app import preview_block_o_read_model as read_model
from ui.pyside_app.preview_block_o_read_model import (
    BLOCKED_STATUS,
    _REAL_CAPABILITY_KEYS,
    build_preview_block_o_read_model,
)


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def payload_from(monkeypatch: pytest.MonkeyPatch, source: dict[str, Any]) -> dict[str, Any]:
    original = clone(source)
    calls = {"count": 0}

    def fake_builder() -> dict[str, Any]:
        calls["count"] += 1
        return source

    monkeypatch.setattr(read_model, "build_preview_block_o_entry_contract", fake_builder)
    payload = build_preview_block_o_read_model()
    assert calls["count"] == 1
    assert source == original
    json.dumps(payload)
    return payload


def assert_read_model_blocked(payload: dict[str, Any]) -> None:
    assert payload["block_o_read_model_ready"] is False
    assert payload["ready_for_block_o_2"] is False
    assert payload["status"] == BLOCKED_STATUS
    assert payload["block_o_entry_contract_reference"]["block_o_read_model_ready_by_17_1"] is False
    assert payload["block_o_entry_contract_reference"]["ready_for_functional_preview_17_2"] is False
    assert payload["read_model_summary"]["block_o_read_model_ready"] is False
    assert payload["read_model_summary"]["ready_for_functional_preview_17_2"] is False
    assert payload["fail_closed_read_decision"]["block_o_read_model_in_17_1"] == "blocked"
    assert (
        payload["fail_closed_read_decision"]["block_o_execution_authorization_matrix_in_17_2"]
        == "blocked"
    )
    assert payload["source_boundaries"]["block_o_entry_contract_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_2"] is False
    assert (
        payload["block_o_entry_contract_reference"]["source_block_o_entry_contract_read_by_17_1"]
        is True
    )
    assert payload["block_o_entry_contract_reference"]["block_o_read_model_built_by_17_1"] is True
    assert payload["read_model_summary"]["block_o_read_model_built"] is True
    assert payload["non_execution_read_evidence"]["source_block_o_entry_contract_read"] is True
    assert payload["non_execution_read_evidence"]["block_o_read_model_built"] is True
    assert payload["non_execution_read_evidence"]["source_block_o_entry_contract_accepted"] is False
    assert payload["non_execution_read_evidence"]["block_o_remains_open"] is False


def test_identity_order_reference_and_json_serializable():
    payload = build_preview_block_o_read_model()
    assert list(payload) == read_model._TOP_LEVEL_FIELDS
    assert payload["schema_version"] == read_model.SCHEMA_VERSION
    assert payload["block_o_read_model_kind"] == read_model.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.1"
    assert payload["block_o_read_model_status"] == read_model.READ_MODEL_STATUS
    assert payload["block_o_read_model_decision"] == read_model.READ_MODEL_DECISION
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.2"
    assert payload["next_step_title"] == "BLOCK O EXECUTION AUTHORIZATION MATRIX"
    assert payload["future_steps"] == [
        "functional_preview_17_2_block_o_execution_authorization_matrix"
    ]
    assert (
        list(payload["block_o_entry_contract_reference"])
        == read_model._REFERENCE_SOURCE_FIELDS
        + read_model._REFERENCE_HANDOFF_FIELDS
        + read_model._FALSE_BY_17_1_FIELDS
    )
    json.dumps(payload)


def test_read_model_preserves_accepted_17_0_state():
    payload = build_preview_block_o_read_model()
    assert payload["block_o_read_model_ready"] is True
    assert payload["ready_for_block_o_2"] is True
    assert payload["read_model_summary"]["all_capabilities_blocked"] is True
    assert payload["read_model_summary"]["all_requirements_missing"] is True
    assert payload["read_model_summary"]["all_invariants_preserved"] is True
    assert payload["read_model_summary"]["all_execution_unauthorized"] is True
    assert (
        payload["fail_closed_read_decision"]["block_o_execution_authorization_matrix_in_17_2"]
        == "allowed"
    )


def test_capability_state_remains_40_blocked_and_not_ready():
    state = build_preview_block_o_read_model()["capability_read_state"]
    assert state["overall"]["total_capability_count"] == 40
    assert state["overall"]["ready_capability_count"] == 0
    assert state["overall"]["blocked_capability_count"] == 40
    assert state["overall"]["all_capabilities_blocked"] is True
    assert state["enabled_by_block_o_read_model"] is False


def test_invariants_remain_12_preserved():
    state = build_preview_block_o_read_model()["invariant_read_state"]
    assert state["invariant_count"] == 12
    assert state["preserved_invariant_count"] == 12
    assert state["failed_invariant_count"] == 0
    assert state["all_invariants_preserved"] is True


def test_requirements_remain_7_missing():
    state = build_preview_block_o_read_model()["requirement_read_state"]
    assert state["requirement_count"] == 7
    assert state["missing_requirement_count"] == 7
    assert state["present_requirement_count"] == 0
    assert state["satisfied_requirement_count"] == 0


def test_exe_direction_remains_desktop_exe_and_not_ready():
    source = build_preview_block_o_entry_contract()
    payload = build_preview_block_o_read_model()
    source_exe = source["exe_direction_entry_contract"]
    read_exe = payload["exe_direction_read_state"]
    assert (
        list(read_exe)
        == read_model._EXPECTED_SOURCE_EXE_FIELDS + read_model._EXE_READ_STATE_17_1_FIELDS
    )
    for key, value in source_exe.items():
        assert read_exe[key] == value
    assert read_exe["final_product_direction"] == "desktop_exe"
    assert read_exe["build_readiness_classification"] == "not_ready"
    assert read_exe["packaging_readiness_classification"] == "not_ready"
    assert read_exe["release_readiness_classification"] == "not_ready"
    assert read_exe["read_result"] == "exe_direction_read_execution_not_ready"
    assert read_exe["read_model_is_not_execution_authorization"] is True
    assert read_exe["read_by_block_o_read_model"] is True
    assert read_exe["recalculated_by_block_o_read_model"] is False
    assert read_exe["block_o_read_model_confirms_exe_direction"] is True
    assert read_exe["block_o_read_model_is_not_execution_authorization"] is True
    assert (
        read_exe["block_o_read_model_result"] == "exe_direction_read_preserved_execution_not_ready"
    )


def test_fail_closed_decision_keeps_all_real_capabilities_blocked():
    decision = build_preview_block_o_read_model()["fail_closed_read_decision"]
    assert list(decision["real_capability_status"]) == _REAL_CAPABILITY_KEYS
    assert set(decision["real_capability_status"].values()) == {"blocked"}
    assert decision["real_capability_status_inherited_from_17_0"] is True
    assert decision["real_capability_status_modified_by_17_1"] is False


def test_non_execution_evidence_and_boundaries():
    payload = build_preview_block_o_read_model()
    evidence = payload["non_execution_read_evidence"]
    assert evidence["source_block_o_entry_contract_read"] is True
    assert evidence["all_capability_states_blocked"] is True
    assert all(value is False for key, value in evidence.items() if key.endswith("_by_17_1"))
    assert all(value is True for value in payload["read_model_boundaries"].values())


def test_builder_calls_17_0_exactly_once(monkeypatch):
    payload = payload_from(monkeypatch, build_preview_block_o_entry_contract())
    assert payload["block_o_read_model_ready"] is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", "sentinel"),
        ("block_o_entry_contract_kind", "sentinel"),
        ("block", "N"),
        ("step", "17.x"),
        ("status", "sentinel"),
        ("block_o_entry_contract_decision", "sentinel"),
        ("block_o_opened", False),
        ("ready_for_block_o_1", False),
        ("next_step", "sentinel"),
        ("next_step_title", "sentinel"),
        ("future_steps", ["sentinel"]),
    ],
)
def test_source_identity_sentinels_block(monkeypatch, key, value):
    source = build_preview_block_o_entry_contract()
    source[key] = value
    assert_read_model_blocked(payload_from(monkeypatch, source))


def test_source_identity_extra_missing_reordered_fields_block(monkeypatch):
    for mutate in (
        lambda s: s.update({"sentinel": True}),
        lambda s: s.pop("status"),
        lambda s: s.update({"schema_version": s.pop("schema_version")}),
    ):
        source = build_preview_block_o_entry_contract()
        mutate(source)
        assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "key",
    [
        "block_n_closure_audit_reference",
        "entry_contract_summary",
        "inherited_block_n_closure_summary",
        "inherited_capability_state",
        "inherited_invariant_state",
        "inherited_requirement_state",
        "exe_direction_entry_contract",
        "fail_closed_entry_decision",
        "non_execution_entry_evidence",
        "entry_contract_boundaries",
        "source_boundaries",
    ],
)
def test_missing_and_non_dict_sections_block(monkeypatch, key):
    for replacement in (None, "sentinel"):
        source = build_preview_block_o_entry_contract()
        if replacement is None:
            source.pop(key)
        else:
            source[key] = replacement
        assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["entry_contract_summary"].__setitem__("any_gate_open_now", True),
        lambda s: s["entry_contract_summary"].__setitem__(
            "all_execution_capabilities_blocked", False
        ),
        lambda s: s["inherited_capability_state"]["packaging_release"].__setitem__(
            "ready_capability_count", 1
        ),
        lambda s: s["inherited_capability_state"]["runtime_safety"].__setitem__(
            "execution_authorized", True
        ),
        lambda s: s["inherited_invariant_state"].__setitem__("all_invariants_preserved", False),
        lambda s: s["inherited_requirement_state"].__setitem__("present_requirement_count", 1),
        lambda s: s["exe_direction_entry_contract"].__setitem__("runtime_gate_open_now", True),
        lambda s: s["fail_closed_entry_decision"]["real_capability_status"].__setitem__(
            _REAL_CAPABILITY_KEYS[0], "allowed"
        ),
    ],
)
def test_safety_state_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"].__setitem__(0, None),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"].__setitem__(
            0, []
        ),
        lambda s: s["inherited_invariant_state"].__setitem__(
            "source_invariant_read_rows", "sentinel"
        ),
        lambda s: s["inherited_requirement_state"].__setitem__(
            "source_requirement_read_rows", None
        ),
    ],
)
def test_row_type_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_capability_state"]["overall"].__setitem__(
            "total_capability_count", 40.0
        ),
        lambda s: s["inherited_capability_state"]["packaging_release"].__setitem__(
            "capability_count", 22.0
        ),
        lambda s: s["inherited_capability_state"]["runtime_safety"].__setitem__(
            "ready_capability_count", False
        ),
        lambda s: s["inherited_invariant_state"].__setitem__("invariant_count", 12.0),
        lambda s: s["inherited_requirement_state"].__setitem__("requirement_count", 7.0),
    ],
)
def test_integer_type_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_block_n_closure_summary"].__setitem__("block_n_step_count", 8.0),
        lambda s: s["inherited_block_n_closure_summary"].__setitem__(
            "completed_block_n_step_count", 8.0
        ),
        lambda s: s["inherited_block_n_closure_summary"].__setitem__("source_block_n_closed", 1),
        lambda s: s["inherited_block_n_closure_summary"].__setitem__(
            "all_block_n_steps_complete", 1
        ),
        lambda s: s["inherited_block_n_closure_summary"].__setitem__(
            "block_n_closure_preserved", 1
        ),
    ],
)
def test_block_n_exact_type_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_block_n_closure_summary"].__setitem__(
            "read_by_block_o_read_model", True
        ),
        lambda s: s["inherited_capability_state"].__setitem__("read_by_block_o_read_model", True),
        lambda s: s["inherited_invariant_state"].__setitem__("read_by_block_o_read_model", True),
        lambda s: s["inherited_requirement_state"].__setitem__("read_by_block_o_read_model", True),
        lambda s: s["exe_direction_entry_contract"].__setitem__("read_by_block_o_read_model", True),
        lambda s: s["source_boundaries"].__setitem__("can_feed_17_2", True),
    ],
)
def test_field_shadowing_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda m: m.clear(),
        lambda m: m.pop(_REAL_CAPABILITY_KEYS[0]),
        lambda m: m.update({"extra": "blocked"}),
        lambda m: (
            m.__setitem__("x", m.pop(_REAL_CAPABILITY_KEYS[0])),
            m.__setitem__(_REAL_CAPABILITY_KEYS[0], "blocked"),
        ),
        lambda m: m.__setitem__(_REAL_CAPABILITY_KEYS[0], "allowed"),
    ],
)
def test_real_capability_map_shape_and_values_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source["fail_closed_entry_decision"]["real_capability_status"])
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize("replacement", [None, "sentinel", list(_REAL_CAPABILITY_KEYS)])
def test_real_capability_map_missing_or_non_dict_blocks_with_empty_final_map(
    monkeypatch, replacement
):
    source = build_preview_block_o_entry_contract()
    if replacement is None:
        source["fail_closed_entry_decision"].pop("real_capability_status")
    else:
        source["fail_closed_entry_decision"]["real_capability_status"] = replacement
    payload = payload_from(monkeypatch, source)
    assert_read_model_blocked(payload)
    if replacement is not None:
        assert payload["fail_closed_read_decision"]["real_capability_status"] == {}
        assert (
            payload["fail_closed_read_decision"]["real_capability_status_inherited_from_17_0"]
            is False
        )
        assert (
            payload["fail_closed_read_decision"]["real_capability_status_modified_by_17_1"] is False
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["block_n_closure_audit_reference"].__setitem__("gate_opened_by_17_0", True),
        lambda s: s["inherited_capability_state"]["packaging_release"].__setitem__(
            "required_requirement_ids", ["sentinel"]
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].__setitem__(
            "failure_policy", "fail_open"
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "source_present", True
        ),
        lambda s: s["exe_direction_entry_contract"].__setitem__("matrix_result", "sentinel"),
        lambda s: s["fail_closed_entry_decision"].__setitem__(
            "sentinel_extra_policy", "fail_closed"
        ),
        lambda s: s["non_execution_entry_evidence"].__setitem__(
            "all_capability_states_blocked", False
        ),
        lambda s: s["entry_contract_boundaries"].pop("cannot_run_runtime"),
        lambda s: [
            s["source_boundaries"].pop(key)
            for key in list(s["source_boundaries"])
            if key.startswith("forbidden_")
        ],
    ],
)
def test_exact_lineage_bypass_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].__setitem__(
            "invariant_id", "sentinel"
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].__setitem__(
            "source_contract_id", "sentinel"
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].__setitem__(
            "read_result", "sentinel"
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"].reverse(),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"].__setitem__(
            1, s["inherited_invariant_state"]["source_invariant_read_rows"][0]
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].__setitem__(
            "extra", "sentinel"
        ),
        lambda s: s["inherited_invariant_state"]["source_invariant_read_rows"][0].pop("notes"),
    ],
)
def test_exact_invariant_row_lineage_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("domain", "sentinel_domain"),
        ("display_name", "Sentinel Display Name"),
    ],
)
def test_exact_invariant_domain_and_display_name_sentinels_block(monkeypatch, field, value):
    source = build_preview_block_o_entry_contract()
    source["inherited_invariant_state"]["source_invariant_read_rows"][0][field] = value
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "requirement_id", "sentinel"
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "display_name", "sentinel"
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "applicable_domains", ["sentinel"]
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "source_required", False
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "completed", True
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "missing_blocks_execution", False
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "failure_policy", "fail_open"
        ),
        lambda s: s["inherited_requirement_state"]["source_requirement_read_rows"][0].__setitem__(
            "read_result", "sentinel"
        ),
    ],
)
def test_exact_requirement_row_lineage_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("read_by_block_o_read_model", True),
        ("recalculated_by_block_o_read_model", False),
        ("block_o_read_model_confirms_exe_direction", True),
        ("block_o_read_model_is_not_execution_authorization", True),
        ("block_o_read_model_result", "exe_direction_read_preserved_execution_not_ready"),
    ],
)
def test_exe_field_shadowing_all_17_1_fields_block(monkeypatch, field, value):
    source = build_preview_block_o_entry_contract()
    source["exe_direction_entry_contract"][field] = value
    payload = payload_from(monkeypatch, source)
    assert field in source["exe_direction_entry_contract"]
    assert_read_model_blocked(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("read_result", "sentinel_source_read_result"),
        ("read_model_is_not_execution_authorization", False),
    ],
)
def test_exe_inherited_source_semantics_sentinels_block(monkeypatch, field, value):
    source = build_preview_block_o_entry_contract()
    source["exe_direction_entry_contract"][field] = value
    payload = payload_from(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert payload["exe_direction_read_state"][field] == value
    assert (
        payload["exe_direction_read_state"]["block_o_read_model_result"]
        == "exe_direction_source_invalid_execution_blocked"
    )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["fail_closed_entry_decision"].pop("missing_operator_confirmation_policy"),
        lambda s: s["fail_closed_entry_decision"].__setitem__("extra_policy", "fail_closed"),
        lambda s: s["fail_closed_entry_decision"].update(
            {
                "missing_block_n_closure_audit_policy": s["fail_closed_entry_decision"].pop(
                    "missing_block_n_closure_audit_policy"
                )
            }
        ),
        lambda s: s["fail_closed_entry_decision"].__setitem__(
            "missing_operator_confirmation_policy", "fail_open"
        ),
        lambda s: s["fail_closed_entry_decision"].pop("real_capability_status_inherited_from_16_8"),
        lambda s: s["fail_closed_entry_decision"].__setitem__(
            "block_o_entry_contract_in_17_0", "sentinel"
        ),
    ],
)
def test_exact_fail_closed_decision_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["non_execution_entry_evidence"].__setitem__(
            "source_block_n_closure_audit_read", False
        ),
        lambda s: s["non_execution_entry_evidence"].__setitem__("extra", False),
        lambda s: s["non_execution_entry_evidence"].pop("gate_opened"),
        lambda s: s["non_execution_entry_evidence"].update(
            {
                "source_block_n_closure_audit_read": s["non_execution_entry_evidence"].pop(
                    "source_block_n_closure_audit_read"
                )
            }
        ),
    ],
)
def test_exact_evidence_shape_sentinels_block(monkeypatch, mutator):
    source = build_preview_block_o_entry_contract()
    mutator(source)
    assert_read_model_blocked(payload_from(monkeypatch, source))


def assert_local_read_valid(payload: dict[str, Any], section: str) -> None:
    if section == "block_n":
        assert payload["block_n_closure_read_state"]["read_by_block_o_read_model"] is True
        assert (
            payload["block_n_closure_read_state"]["read_result"]
            == "block_n_closure_read_preserved_execution_blocked"
        )
    if section == "capability":
        assert payload["capability_read_state"]["read_by_block_o_read_model"] is True
        assert (
            payload["capability_read_state"]["read_result"]
            == "capability_state_read_all_blocked_execution_unauthorized"
        )
    if section == "invariant":
        assert payload["invariant_read_state"]["read_by_block_o_read_model"] is True
        assert (
            payload["invariant_read_state"]["read_result"]
            == "invariants_read_all_preserved_execution_blocked"
        )
    if section == "requirement":
        assert payload["requirement_read_state"]["read_by_block_o_read_model"] is True
        assert (
            payload["requirement_read_state"]["read_result"]
            == "requirements_read_all_missing_execution_blocked"
        )
    if section == "exe":
        assert payload["exe_direction_read_state"]["read_by_block_o_read_model"] is True
        assert (
            payload["exe_direction_read_state"]["block_o_read_model_result"]
            == "exe_direction_read_preserved_execution_not_ready"
        )


def test_cross_section_invariant_invalid_keeps_other_local_reads_valid(monkeypatch):
    source = build_preview_block_o_entry_contract()
    source["inherited_invariant_state"]["source_invariant_read_rows"][0]["domain"] = (
        "sentinel_domain"
    )
    payload = payload_from(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert payload["invariant_read_state"]["read_by_block_o_read_model"] is False
    assert (
        payload["invariant_read_state"]["read_result"]
        == "invariant_source_invalid_execution_blocked"
    )
    assert_local_read_valid(payload, "block_n")
    assert_local_read_valid(payload, "capability")
    assert_local_read_valid(payload, "requirement")
    assert_local_read_valid(payload, "exe")
    assert payload["non_execution_read_evidence"]["all_invariant_states_read"] is False
    assert payload["non_execution_read_evidence"]["block_n_closure_read"] is True
    assert payload["non_execution_read_evidence"]["all_capability_states_read"] is True
    assert payload["non_execution_read_evidence"]["all_requirement_states_read"] is True
    assert payload["non_execution_read_evidence"]["exe_direction_read"] is True


def test_cross_section_exe_invalid_keeps_other_local_reads_valid(monkeypatch):
    source = build_preview_block_o_entry_contract()
    source["exe_direction_entry_contract"]["runtime_gate_open_now"] = True
    payload = payload_from(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert payload["exe_direction_read_state"]["read_by_block_o_read_model"] is False
    assert (
        payload["exe_direction_read_state"]["block_o_read_model_result"]
        == "exe_direction_source_invalid_execution_blocked"
    )
    assert_local_read_valid(payload, "block_n")
    assert_local_read_valid(payload, "capability")
    assert_local_read_valid(payload, "invariant")
    assert_local_read_valid(payload, "requirement")
    assert payload["non_execution_read_evidence"]["exe_direction_read"] is False
    assert payload["non_execution_read_evidence"]["all_invariant_states_read"] is True


def test_cross_section_capability_invalid_keeps_other_local_reads_valid(monkeypatch):
    source = build_preview_block_o_entry_contract()
    source["inherited_capability_state"]["runtime_safety"]["execution_authorized"] = True
    payload = payload_from(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert payload["capability_read_state"]["read_by_block_o_read_model"] is False
    assert (
        payload["capability_read_state"]["read_result"]
        == "capability_source_invalid_execution_blocked"
    )
    assert_local_read_valid(payload, "block_n")
    assert_local_read_valid(payload, "invariant")
    assert_local_read_valid(payload, "requirement")
    assert_local_read_valid(payload, "exe")
    assert payload["non_execution_read_evidence"]["all_capability_states_read"] is False
    assert payload["non_execution_read_evidence"]["all_capability_states_blocked"] is False
    assert payload["non_execution_read_evidence"]["block_n_closure_read"] is True


def test_ast_source_guard():
    tree = ast.parse(Path("ui/pyside_app/preview_block_o_read_model.py").read_text())
    assert [node for node in ast.walk(tree) if isinstance(node, ast.Import)] == []
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert len(imports) == 3
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        ("ui.pyside_app.preview_block_o_entry_contract", ["build_preview_block_o_entry_contract"]),
    ]
    calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("build_preview_block_o_entry_contract") == 1
    forbidden_calls = {
        "build_preview_block_n_closure_audit",
        "build_preview_block_n",
        "build_preview_block_m",
        "subprocess",
        "requests",
        "urllib",
        "socket",
        "pathlib",
        "Path",
        "git",
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert called_names.isdisjoint(forbidden_calls)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert attribute_calls == {"get", "items", "replace", "startswith", "update", "upper"}
    forbidden_attributes = {
        "read_text",
        "write_text",
        "open",
        "run",
        "request",
        "validate",
        "authorize",
        "package",
        "release",
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
    }
    assert attribute_calls.isdisjoint(forbidden_attributes)
    imported_modules = {node.module for node in imports}
    assert not any(
        "runtime" in str(module).lower()
        or "gateway" in str(module).lower()
        or "controller" in str(module).lower()
        or "qml" in str(module).lower()
        or "bridge" in str(module).lower()
        for module in imported_modules
    )
