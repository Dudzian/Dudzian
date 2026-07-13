from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_o_read_model import build_preview_block_o_read_model
from ui.pyside_app import preview_block_o_execution_authorization_matrix as matrix
from ui.pyside_app.preview_block_o_execution_authorization_matrix import (
    BLOCKED_STATUS,
    build_preview_block_o_execution_authorization_matrix,
)


class PretendBlocked:
    def __eq__(self, other: object) -> bool:
        return other == "blocked"


class PretendRealCapabilityKey:
    def __init__(self, expected: str) -> None:
        self.expected = expected

    def __eq__(self, other: object) -> bool:
        return other == self.expected

    def __hash__(self) -> int:
        return hash(self.expected)


def clone(value: Any) -> Any:
    return json.loads(json.dumps(value))


def payload_from(monkeypatch: pytest.MonkeyPatch, source: dict[str, Any]) -> dict[str, Any]:
    original = clone(source)
    calls = {"count": 0}

    def fake_builder() -> dict[str, Any]:
        calls["count"] += 1
        return source

    monkeypatch.setattr(matrix, "build_preview_block_o_read_model", fake_builder)
    payload = build_preview_block_o_execution_authorization_matrix()
    assert calls["count"] == 1
    assert source == original
    json.dumps(payload)
    return payload


def payload_from_non_json(
    monkeypatch: pytest.MonkeyPatch, source: dict[str, Any]
) -> dict[str, Any]:
    calls = {"count": 0}

    def fake_builder() -> dict[str, Any]:
        calls["count"] += 1
        return source

    monkeypatch.setattr(matrix, "build_preview_block_o_read_model", fake_builder)
    payload = build_preview_block_o_execution_authorization_matrix()
    assert calls["count"] == 1
    json.dumps(payload)
    return payload


def assert_matrix_blocked(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_matrix_ready"] is False
    assert payload["ready_for_block_o_3"] is False
    assert payload["execution_authorization_matrix_status"] == BLOCKED_STATUS
    assert payload["execution_authorization_matrix_decision"] == BLOCKED_STATUS.upper()
    assert payload["status"] == BLOCKED_STATUS
    assert payload["non_execution_matrix_evidence"]["source_block_o_read_model_read"] is True
    assert payload["non_execution_matrix_evidence"]["execution_authorization_matrix_built"] is True
    assert payload["non_execution_matrix_evidence"]["source_block_o_read_model_accepted"] is False
    assert payload["non_execution_matrix_evidence"]["block_o_remains_open"] is False
    assert (
        payload["fail_closed_matrix_decision"]["execution_authorization_contract_in_17_3"]
        == "blocked"
    )
    assert (
        payload["fail_closed_matrix_decision"]["execution_authorization_granted_by_17_2"] is False
    )
    assert payload["source_boundaries"]["block_o_read_model_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_3"] is False


def move_first_key_to_end(value: dict[str, Any]) -> dict[str, Any]:
    first_key = next(iter(value))
    value[first_key] = value.pop(first_key)
    return value


def reorder_first_row(section: dict[str, Any], rows_key: str) -> None:
    move_first_key_to_end(section[rows_key][0])


def test_identity_order_reference_and_json_serializable():
    payload = build_preview_block_o_execution_authorization_matrix()
    assert list(payload) == matrix.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == matrix.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_matrix_kind"] == matrix.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.2"
    assert payload["execution_authorization_matrix_status"] == matrix.MATRIX_STATUS
    assert payload["execution_authorization_matrix_decision"] == matrix.MATRIX_DECISION
    assert payload["execution_authorization_matrix_ready"] is True
    assert payload["ready_for_block_o_3"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.3"
    assert payload["next_step_title"] == "BLOCK O EXECUTION AUTHORIZATION CONTRACT"
    assert payload["status"] == matrix.STATUS
    reference = payload["block_o_read_model_reference"]
    assert reference["schema_version"] == "preview_block_o_read_model.v1"
    assert reference["source_block_o_read_model_step"] == "FUNCTIONAL-PREVIEW-17.1"
    assert reference["source_block_o_read_model_read_by_17_2"] is True
    assert reference["execution_authorization_matrix_ready_by_17_2"] is True
    assert all(
        value is False
        for key, value in reference.items()
        if key.endswith("_by_17_2")
        and key
        not in {
            "source_block_o_read_model_read_by_17_2",
            "execution_authorization_matrix_built_by_17_2",
            "execution_authorization_matrix_ready_by_17_2",
        }
    )
    json.dumps(payload)


def test_two_domain_rows_are_unauthorized():
    rows = build_preview_block_o_execution_authorization_matrix()["domain_authorization_rows"]
    assert [row["domain"] for row in rows] == ["packaging_release", "runtime_safety"]
    assert rows[0]["source_capability_count"] == 22
    assert rows[1]["source_capability_count"] == 18
    for row in rows:
        assert row["source_read_capability_count"] == row["source_blocked_capability_count"]
        assert row["source_ready_capability_count"] == 0
        assert row["requirements_complete"] is False
        assert row["source_execution_authorized"] is False
        assert row["authorization_condition_met"] is False
        assert row["execution_authorized_by_matrix"] is False
        assert row["authorization_classification"] == "blocked_missing_required_conditions"


def test_seven_requirement_rows_are_missing_and_unauthorized():
    rows = build_preview_block_o_execution_authorization_matrix()["requirement_authorization_rows"]
    assert [row["requirement_id"] for row in rows] == [
        row["requirement_id"] for row in matrix.EXPECTED_REQUIREMENT["source_requirement_read_rows"]
    ]
    assert len(rows) == 7
    for row in rows:
        assert row["required"] is True
        assert row["source_present"] is False
        assert row["source_completed"] is False
        assert row["source_satisfied"] is False
        assert row["missing"] is True
        assert row["execution_authorized_by_matrix"] is False
        assert row["matrix_result"] == "missing_requirement_execution_unauthorized"


def test_invariant_guard_and_exe_guard_preserve_inherited_fields():
    payload = build_preview_block_o_execution_authorization_matrix()
    inv = payload["invariant_authorization_guard"]
    exe = payload["exe_authorization_guard"]
    for key, value in matrix.EXPECTED_INVARIANT.items():
        assert inv[key] == value
    for key, value in matrix.EXPECTED_EXE.items():
        assert exe[key] == value
    assert inv["invariants_alone_authorize_execution"] is False
    assert inv["execution_authorized_by_matrix"] is False
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["runtime_gate_open_now"] is False
    assert exe["matrix_result"] == matrix.EXPECTED_EXE["matrix_result"]
    assert exe["read_result"] == matrix.EXPECTED_EXE["read_result"]
    assert exe["block_o_read_model_result"] == matrix.EXPECTED_EXE["block_o_read_model_result"]
    assert exe["block_o_authorization_matrix_confirms_desktop_exe"] is True
    assert exe["execution_authorized_by_matrix"] is False


def test_exact_blocked_real_capability_map_and_summary_evidence_boundaries():
    payload = build_preview_block_o_execution_authorization_matrix()
    state = payload["real_capability_authorization_state"]
    assert list(state["real_capability_status"]) == matrix.REAL_CAPABILITY_KEYS
    assert set(state["real_capability_status"].values()) == {"blocked"}
    assert state["all_real_capabilities_blocked"] is True
    assert state["execution_authorized_by_matrix"] is False
    summary = payload["matrix_summary"]
    assert summary["domain_row_count"] == 2
    assert summary["requirement_row_count"] == 7
    assert summary["all_authorization_conditions_unmet"] is True
    assert summary["packaging_build_release_performed_by_17_2"] is False
    evidence = payload["non_execution_matrix_evidence"]
    assert evidence["source_block_o_read_model_read"] is True
    assert evidence["source_block_o_read_model_accepted"] is True
    assert all(payload["matrix_boundaries"].values())


def test_fail_closed_decision_nominal_and_single_builder_call(monkeypatch):
    payload = payload_from(monkeypatch, build_preview_block_o_read_model())
    decision = payload["fail_closed_matrix_decision"]
    assert decision["block_o_read_model_in_17_1"] == "preserved"
    assert decision["execution_authorization_matrix_in_17_2"] == "ready"
    assert decision["execution_authorization_contract_in_17_3"] == "allowed"
    assert decision["only_source_only_17_3_handoff_allowed"] is True
    assert decision["execution_authorization_granted_by_17_2"] is False


def test_fail_closed_decision_field_order_and_real_capability_provenance():
    decision = build_preview_block_o_execution_authorization_matrix()["fail_closed_matrix_decision"]
    assert list(decision) == [
        "missing_source_policy",
        "missing_domain_state_policy",
        "missing_requirement_state_policy",
        "missing_invariant_state_policy",
        "missing_exe_state_policy",
        "missing_real_capability_state_policy",
        "missing_confirmation_policy",
        "missing_validation_policy",
        "missing_credentials_policy",
        "missing_future_gate_policy",
        "failed_matrix_policy",
        "block_o_read_model_in_17_1",
        "execution_authorization_matrix_in_17_2",
        "execution_authorization_contract_in_17_3",
        "only_source_only_17_3_handoff_allowed",
        "real_capability_status",
        "real_capability_status_inherited_from_17_1",
        "real_capability_status_modified_by_17_2",
        "execution_authorization_granted_by_17_2",
    ]
    assert list(decision["real_capability_status"]) == matrix.REAL_CAPABILITY_KEYS
    assert set(decision["real_capability_status"].values()) == {"blocked"}
    assert decision["real_capability_status_inherited_from_17_1"] is True
    assert decision["real_capability_status_modified_by_17_2"] is False


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", "sentinel"),
        ("block_o_read_model_kind", "sentinel"),
        ("block", "N"),
        ("step", "17.x"),
        ("block_o_read_model_ready", False),
        ("ready_for_block_o_2", False),
        ("next_step", "sentinel"),
        ("next_step_title", "sentinel"),
        ("status", "sentinel"),
        ("future_steps", ["sentinel"]),
    ],
)
def test_identity_scalar_sentinels_block(monkeypatch, key, value):
    source = build_preview_block_o_read_model()
    source[key] = value
    assert_matrix_blocked(payload_from(monkeypatch, source))


def test_identity_extra_missing_reordered_top_level_fields_block(monkeypatch):
    for mutate in (
        lambda source: source.update({"sentinel": True}),
        lambda source: source.pop("status"),
        lambda source: source.update({"schema_version": source.pop("schema_version")}),
    ):
        source = build_preview_block_o_read_model()
        mutate(source)
        assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "key",
    [
        "block_o_entry_contract_reference",
        "read_model_summary",
        "block_n_closure_read_state",
        "capability_read_state",
        "invariant_read_state",
        "requirement_read_state",
        "exe_direction_read_state",
        "fail_closed_read_decision",
        "non_execution_read_evidence",
        "read_model_boundaries",
        "source_boundaries",
    ],
)
def test_missing_and_non_dict_sections_block(monkeypatch, key):
    for value in (None, "sentinel"):
        source = build_preview_block_o_read_model()
        if value is None:
            source.pop(key)
        else:
            source[key] = value
        assert_matrix_blocked(payload_from(monkeypatch, source))


def test_missing_invariant_section_uses_honest_invalid_guard(monkeypatch):
    source = build_preview_block_o_read_model()
    source.pop("invariant_read_state")
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    guard = payload["invariant_authorization_guard"]
    assert guard["read_by_execution_authorization_matrix"] is False
    assert guard["invariants_preserved_for_future_authorization"] is False
    assert guard["execution_authorized_by_matrix"] is False
    assert (
        guard["block_o_authorization_matrix_result"]
        == "invariant_source_invalid_execution_unauthorized"
    )
    assert "invariant_count" not in guard
    assert "source_invariant_read_rows" not in guard


def test_missing_exe_section_uses_honest_invalid_guard(monkeypatch):
    source = build_preview_block_o_read_model()
    source.pop("exe_direction_read_state")
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    guard = payload["exe_authorization_guard"]
    assert guard["read_by_execution_authorization_matrix"] is False
    assert guard["block_o_authorization_matrix_confirms_desktop_exe"] is False
    assert guard["execution_authorized_by_matrix"] is False
    assert (
        guard["block_o_authorization_matrix_result"] == "exe_source_invalid_execution_unauthorized"
    )
    assert "final_product_direction" not in guard
    assert "read_result" not in guard


def test_missing_source_boundaries_uses_honest_invalid_boundaries(monkeypatch):
    source = build_preview_block_o_read_model()
    source.pop("source_boundaries")
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    boundaries = payload["source_boundaries"]
    assert "source_block_n_closure_audit" not in boundaries
    assert "can_feed_17_2" not in boundaries
    assert boundaries["source_block_o_read_model"] == "FUNCTIONAL-PREVIEW-17.1"
    assert boundaries["block_o_read_model_source_preserved"] is False
    assert boundaries["can_build_execution_authorization_matrix"] is False
    assert boundaries["can_feed_17_3"] is False


def test_non_json_invariant_dict_is_not_copied_to_blocked_payload(monkeypatch):
    source = build_preview_block_o_read_model()
    sentinel = object()
    source["invariant_read_state"]["invariant_count"] = sentinel
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert source["invariant_read_state"]["invariant_count"] is sentinel
    guard = payload["invariant_authorization_guard"]
    assert "invariant_count" not in guard
    assert guard["read_by_execution_authorization_matrix"] is False


def test_non_json_exe_dict_is_not_copied_to_blocked_payload(monkeypatch):
    source = build_preview_block_o_read_model()
    sentinel = object()
    source["exe_direction_read_state"]["runtime_gate_open_now"] = sentinel
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert source["exe_direction_read_state"]["runtime_gate_open_now"] is sentinel
    guard = payload["exe_authorization_guard"]
    assert "runtime_gate_open_now" not in guard
    assert guard["read_by_execution_authorization_matrix"] is False


def test_non_json_source_boundaries_dict_is_not_copied_to_blocked_payload(monkeypatch):
    source = build_preview_block_o_read_model()
    sentinel = object()
    source["source_boundaries"]["sentinel_non_json"] = sentinel
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert source["source_boundaries"]["sentinel_non_json"] is sentinel
    boundaries = payload["source_boundaries"]
    assert "sentinel_non_json" not in boundaries
    assert "source_block_n_closure_audit" not in boundaries
    assert boundaries["can_feed_17_3"] is False


def test_non_json_scalar_reference_field_becomes_none_in_blocked_payload(monkeypatch):
    source = build_preview_block_o_read_model()
    sentinel = object()
    source["schema_version"] = sentinel
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert source["schema_version"] is sentinel
    assert payload["block_o_read_model_reference"]["schema_version"] is None


def real_capability_map_with_custom_key() -> dict[Any, Any]:
    nominal = clone(matrix.EXPECTED_FAIL["real_capability_status"])
    first_key = next(iter(nominal))
    custom_key_map = {PretendRealCapabilityKey(first_key): nominal[first_key]}
    for key, value in list(nominal.items())[1:]:
        custom_key_map[key] = value
    return custom_key_map


def assert_real_capability_bypass_blocked(payload: dict[str, Any]) -> None:
    assert_matrix_blocked(payload)
    assert payload["real_capability_authorization_state"]["all_real_capabilities_blocked"] is False
    assert payload["real_capability_authorization_state"]["real_capability_status"] == {}
    assert payload["fail_closed_matrix_decision"]["real_capability_status"] == {}
    json.dumps(payload)


def test_real_capability_custom_value_equality_bypass_blocks_and_stays_json_safe(monkeypatch):
    source = build_preview_block_o_read_model()
    sentinel = PretendBlocked()
    source["fail_closed_read_decision"]["real_capability_status"]["release_execution"] = sentinel
    payload = payload_from_non_json(monkeypatch, source)
    assert (
        source["fail_closed_read_decision"]["real_capability_status"]["release_execution"]
        is sentinel
    )
    assert_real_capability_bypass_blocked(payload)


def test_real_capability_custom_key_equality_bypass_blocks_and_stays_json_safe(monkeypatch):
    source = build_preview_block_o_read_model()
    custom_key_map = real_capability_map_with_custom_key()
    custom_key = next(iter(custom_key_map))
    source["fail_closed_read_decision"]["real_capability_status"] = custom_key_map
    payload = payload_from_non_json(monkeypatch, source)
    assert next(iter(source["fail_closed_read_decision"]["real_capability_status"])) is custom_key
    assert_real_capability_bypass_blocked(payload)


def test_real_capability_helper_rejects_custom_equality_bypasses():
    nominal_map = clone(matrix.EXPECTED_FAIL["real_capability_status"])
    custom_value_map = clone(matrix.EXPECTED_FAIL["real_capability_status"])
    custom_value_map["release_execution"] = PretendBlocked()
    custom_key_map = real_capability_map_with_custom_key()

    assert matrix._real_capability_status_is_exactly_blocked(nominal_map) is True
    assert matrix._real_capability_status_is_exactly_blocked(custom_value_map) is False
    assert matrix._real_capability_status_is_exactly_blocked(custom_key_map) is False


def test_cyclic_invariant_dict_blocks_without_recursion_error(monkeypatch):
    source = build_preview_block_o_read_model()
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    source["invariant_read_state"] = cyclic
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert (
        payload["invariant_authorization_guard"]["read_by_execution_authorization_matrix"] is False
    )
    json.dumps(payload)
    assert source["invariant_read_state"] is cyclic
    assert cyclic["self"] is cyclic


def test_cyclic_exe_list_blocks_and_is_not_copied(monkeypatch):
    source = build_preview_block_o_read_model()
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    source["exe_direction_read_state"]["runtime_gate_open_now"] = cyclic
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    guard = payload["exe_authorization_guard"]
    assert guard["read_by_execution_authorization_matrix"] is False
    assert "runtime_gate_open_now" not in guard
    json.dumps(payload)
    assert source["exe_direction_read_state"]["runtime_gate_open_now"] is cyclic
    assert cyclic[0] is cyclic


def test_cyclic_source_boundaries_block_with_only_17_2_fields(monkeypatch):
    source = build_preview_block_o_read_model()
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    source["source_boundaries"] = cyclic
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert payload["source_boundaries"] == {
        "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
        "block_o_read_model_source_preserved": False,
        "can_build_execution_authorization_matrix": False,
        "can_feed_17_3": False,
    }
    json.dumps(payload)
    assert source["source_boundaries"] is cyclic
    assert cyclic["self"] is cyclic


def test_all_plain_json_is_cycle_safe_and_allows_shared_acyclic_references():
    assert matrix._all_plain_json(matrix.EXPECTED_FAIL["real_capability_status"]) is True

    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    assert matrix._all_plain_json(cyclic_dict) is False

    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert matrix._all_plain_json(cyclic_list) is False

    shared = ["value"]
    assert matrix._all_plain_json({"first": shared, "second": shared}) is True


def deep_dict(depth: int = 1500) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root
    for _ in range(depth):
        child: dict[str, Any] = {}
        current["child"] = child
        current = child
    return root


def deep_list(depth: int = 1500) -> list[Any]:
    root: list[Any] = []
    current = root
    for _ in range(depth):
        child: list[Any] = []
        current.append(child)
        current = child
    return root


def test_deep_invariant_dict_blocks_without_diagnostic_copy(monkeypatch):
    source = build_preview_block_o_read_model()
    root = deep_dict()
    first_child = root["child"]
    source["invariant_read_state"] = root
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    guard = payload["invariant_authorization_guard"]
    assert guard["read_by_execution_authorization_matrix"] is False
    assert "child" not in guard
    json.dumps(payload)
    assert source["invariant_read_state"] is root
    assert root["child"] is first_child


def test_deep_exe_container_blocks_without_diagnostic_copy(monkeypatch):
    source = build_preview_block_o_read_model()
    root = deep_list()
    first_child = root[0]
    source["exe_direction_read_state"]["runtime_gate_open_now"] = root
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    guard = payload["exe_authorization_guard"]
    assert guard["read_by_execution_authorization_matrix"] is False
    assert "runtime_gate_open_now" not in guard
    json.dumps(payload)
    assert source["exe_direction_read_state"]["runtime_gate_open_now"] is root
    assert root[0] is first_child


def test_deep_source_boundaries_block_with_only_17_2_fields(monkeypatch):
    source = build_preview_block_o_read_model()
    root = deep_dict()
    first_child = root["child"]
    source["source_boundaries"] = root
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert payload["source_boundaries"] == {
        "source_block_o_read_model": "FUNCTIONAL-PREVIEW-17.1",
        "block_o_read_model_source_preserved": False,
        "can_build_execution_authorization_matrix": False,
        "can_feed_17_3": False,
    }
    json.dumps(payload)
    assert source["source_boundaries"] is root
    assert root["child"] is first_child


def test_deep_reference_scalar_container_becomes_none(monkeypatch):
    source = build_preview_block_o_read_model()
    root = deep_list()
    first_child = root[0]
    source["schema_version"] = root
    payload = payload_from_non_json(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert payload["block_o_read_model_reference"]["schema_version"] is None
    json.dumps(payload)
    assert source["schema_version"] is root
    assert root[0] is first_child


def test_all_plain_json_depth_limit_preserves_unlimited_and_shared_behavior():
    deep_value = deep_dict()
    assert matrix._all_plain_json(deep_value) is True
    assert (
        matrix._all_plain_json(
            deep_value,
            max_depth=matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH,
        )
        is False
    )
    nominal_source = build_preview_block_o_read_model()
    assert (
        matrix._all_plain_json(
            nominal_source,
            max_depth=matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH,
        )
        is True
    )
    shared = ["value"]
    assert (
        matrix._all_plain_json(
            {"first": shared, "second": shared},
            max_depth=matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH,
        )
        is True
    )


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "block_o_entry_contract_reference",
            lambda section: section.update({"block_o_read_model_ready_by_17_1": False}),
        ),
        (
            "read_model_summary",
            lambda section: section.update({"all_execution_unauthorized": False}),
        ),
        ("block_n_closure_read_state", lambda section: section.update({"block_n_step_count": 8.0})),
        (
            "capability_read_state",
            lambda section: section["overall"].update({"execution_authorized": True}),
        ),
        (
            "invariant_read_state",
            lambda section: section["source_invariant_read_rows"][0].update({"domain": "sentinel"}),
        ),
        (
            "requirement_read_state",
            lambda section: section["source_requirement_read_rows"][0].update(
                {"source_present": True}
            ),
        ),
        (
            "exe_direction_read_state",
            lambda section: section.update({"runtime_gate_open_now": True}),
        ),
        (
            "fail_closed_read_decision",
            lambda section: section.update(
                {"block_o_execution_authorization_matrix_in_17_2": "blocked"}
            ),
        ),
        (
            "non_execution_read_evidence",
            lambda section: section.update({"all_execution_authorization_false": False}),
        ),
        ("read_model_boundaries", lambda section: section.pop("cannot_authorize")),
        ("source_boundaries", lambda section: section.pop("can_feed_17_2")),
    ],
)
def test_lineage_and_malformed_sentinels_block(monkeypatch, section, mutate):
    source = build_preview_block_o_read_model()
    mutate(source[section])
    assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "capability_read_state",
            lambda section: section["overall"].update({"total_capability_count": 40.0}),
        ),
        ("invariant_read_state", lambda section: section.update({"invariant_count": 12.0})),
        ("requirement_read_state", lambda section: section.update({"requirement_count": 7.0})),
        (
            "invariant_read_state",
            lambda section: section["source_invariant_read_rows"][0].pop("domain"),
        ),
        (
            "requirement_read_state",
            lambda section: section["source_requirement_read_rows"][0].pop("present"),
        ),
        (
            "requirement_read_state",
            lambda section: section.update({"source_requirement_read_rows": {}}),
        ),
    ],
)
def test_type_and_row_shape_sentinels_block(monkeypatch, section, mutate):
    source = build_preview_block_o_read_model()
    mutate(source[section])
    assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        (
            "read_model_summary",
            lambda section: section.update({"block_o_entry_contract_available": 1}),
        ),
        (
            "capability_read_state",
            lambda section: section["overall"].update({"ready_capability_count": False}),
        ),
        (
            "requirement_read_state",
            lambda section: section["source_requirement_read_rows"][0].update({"required": 1}),
        ),
    ],
)
def test_bool_int_bypass_sentinels_block(monkeypatch, section, mutate):
    source = build_preview_block_o_read_model()
    mutate(source[section])
    assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("read_model_summary", move_first_key_to_end),
        (
            "capability_read_state",
            lambda section: move_first_key_to_end(section["packaging_release"]),
        ),
        ("invariant_read_state", move_first_key_to_end),
        (
            "invariant_read_state",
            lambda section: reorder_first_row(section, "source_invariant_read_rows"),
        ),
        ("requirement_read_state", move_first_key_to_end),
        (
            "requirement_read_state",
            lambda section: reorder_first_row(section, "source_requirement_read_rows"),
        ),
        ("exe_direction_read_state", move_first_key_to_end),
        ("fail_closed_read_decision", move_first_key_to_end),
        ("non_execution_read_evidence", move_first_key_to_end),
        ("source_boundaries", move_first_key_to_end),
    ],
)
def test_reordered_source_sections_block(monkeypatch, section, mutate):
    source = build_preview_block_o_read_model()
    mutate(source[section])
    assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda status: status.clear(),
        lambda status: status.pop(next(iter(status))),
        lambda status: status.update({"extra": "blocked"}),
        lambda status: status.update({next(iter(status)): status.pop(next(iter(status)))}),
        lambda status: status.update({next(iter(status)): "allowed"}),
    ],
)
def test_real_capability_map_sentinels_block(monkeypatch, mutate):
    source = build_preview_block_o_read_model()
    mutate(source["fail_closed_read_decision"]["real_capability_status"])
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert payload["real_capability_authorization_state"]["all_real_capabilities_blocked"] is False


@pytest.mark.parametrize("value", [{}, "sentinel", list(matrix.REAL_CAPABILITY_KEYS)])
def test_real_capability_map_missing_field_string_and_list_block(monkeypatch, value):
    source = build_preview_block_o_read_model()
    if value == {}:
        source["fail_closed_read_decision"].pop("real_capability_status")
    else:
        source["fail_closed_read_decision"]["real_capability_status"] = value
    assert_matrix_blocked(payload_from(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "field"),
    [("invariant_read_state", field) for field in matrix.INVARIANT_GUARD_FIELDS_17_2]
    + [("exe_direction_read_state", field) for field in matrix.EXE_GUARD_FIELDS_17_2]
    + [("source_boundaries", field) for field in matrix.SOURCE_BOUNDARY_FIELDS_17_2],
)
def test_field_shadowing_17_2_blocks(monkeypatch, section, field):
    source = build_preview_block_o_read_model()
    source[section][field] = True
    assert_matrix_blocked(payload_from(monkeypatch, source))


def test_cross_section_invalid_invariant_keeps_exe_and_real_valid(monkeypatch):
    source = build_preview_block_o_read_model()
    source["invariant_read_state"]["invariant_count"] = 11
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert (
        payload["invariant_authorization_guard"]["block_o_authorization_matrix_result"]
        == "invariant_source_invalid_execution_unauthorized"
    )
    assert (
        payload["exe_authorization_guard"]["block_o_authorization_matrix_confirms_desktop_exe"]
        is True
    )
    assert payload["real_capability_authorization_state"]["all_real_capabilities_blocked"] is True


def test_cross_section_invalid_exe_keeps_invariant_and_real_valid(monkeypatch):
    source = build_preview_block_o_read_model()
    source["exe_direction_read_state"]["runtime_gate_open_now"] = True
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert (
        payload["exe_authorization_guard"]["block_o_authorization_matrix_result"]
        == "exe_source_invalid_execution_unauthorized"
    )
    assert (
        payload["invariant_authorization_guard"]["invariants_preserved_for_future_authorization"]
        is True
    )
    assert payload["real_capability_authorization_state"]["all_real_capabilities_blocked"] is True


def test_cross_section_invalid_requirement_keeps_invariant_exe_and_real_valid(monkeypatch):
    source = build_preview_block_o_read_model()
    source["requirement_read_state"]["requirement_count"] = 6
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert payload["requirement_authorization_rows"] == []
    assert payload["matrix_summary"]["all_authorization_conditions_unmet"] is False
    for row in payload["domain_authorization_rows"]:
        assert row["authorization_classification"] == "source_invalid"
        assert row["all_capabilities_read"] is True
        assert row["all_capabilities_not_ready"] is True
        assert row["all_capabilities_blocked"] is True
    assert (
        payload["invariant_authorization_guard"]["invariants_preserved_for_future_authorization"]
        is True
    )
    assert (
        payload["exe_authorization_guard"]["block_o_authorization_matrix_confirms_desktop_exe"]
        is True
    )
    assert payload["real_capability_authorization_state"]["all_real_capabilities_blocked"] is True


def test_cross_section_invalid_real_map_keeps_domain_requirement_invariant_and_exe_valid(
    monkeypatch,
):
    source = build_preview_block_o_read_model()
    source["fail_closed_read_decision"]["real_capability_status"]["release_execution"] = "allowed"
    payload = payload_from(monkeypatch, source)
    assert_matrix_blocked(payload)
    assert (
        payload["real_capability_authorization_state"]["state_result"]
        == "real_capability_source_invalid_execution_unauthorized"
    )
    assert len(payload["requirement_authorization_rows"]) == 7
    assert all(
        row["authorization_classification"] == "blocked_missing_required_conditions"
        for row in payload["domain_authorization_rows"]
    )
    assert (
        payload["invariant_authorization_guard"]["invariants_preserved_for_future_authorization"]
        is True
    )
    assert (
        payload["exe_authorization_guard"]["block_o_authorization_matrix_confirms_desktop_exe"]
        is True
    )


def test_ast_guard():
    tree = ast.parse(Path(matrix.__file__).read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    import_from = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == []
    assert len(import_from) == 3
    assert [node.module for node in import_from] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_read_model",
    ]
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_calls = [node.func.id for node in call_nodes if isinstance(node.func, ast.Name)]
    attribute_calls = [
        node.func.attr for node in call_nodes if isinstance(node.func, ast.Attribute)
    ]
    assert sorted(set(name_calls)) == [
        "_all_plain_json",
        "_build_reference",
        "_copy_plain",
        "_domain_rows",
        "_exact_plain_matches",
        "_guard",
        "_identity_valid",
        "_no_shadowing",
        "_plain_dict_section",
        "_real_capability_status_is_exactly_blocked",
        "_requirement_rows",
        "_section_valid",
        "all",
        "any",
        "build_preview_block_o_read_model",
        "id",
        "len",
        "list",
        "type",
        "zip",
    ]
    assert sorted(set(attribute_calls)) == [
        "append",
        "get",
        "items",
        "pop",
        "update",
        "upper",
    ]
    assert name_calls.count("build_preview_block_o_read_model") == 1
    forbidden_calls = {
        "build_preview_block_o_entry_contract",
        "build_preview_block_n_closure_audit",
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "subprocess",
        "network",
        "socket",
        "runtime",
        "orders",
        "validation",
        "authorization",
        "packaging",
        "build",
        "release",
        "pyinstaller",
        "qml",
        "bridge",
        "gateway",
        "controller",
    }
    assert not (set(name_calls) & forbidden_calls)
    assert not (set(attribute_calls) & forbidden_calls)
