from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

import ui.pyside_app.preview_block_o_execution_authorization_contract as contract
from ui.pyside_app.preview_block_o_execution_authorization_matrix import (
    build_preview_block_o_execution_authorization_matrix,
)


def build_payload() -> dict[str, Any]:
    return contract.build_preview_block_o_execution_authorization_contract()


def assert_contract_blocked(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_contract_ready"] is False
    assert payload["ready_for_block_o_4"] is False
    assert payload["execution_authorization_contract_status"] == contract.BLOCKED_STATUS
    assert payload["execution_authorization_contract_decision"] == contract.BLOCKED_STATUS.upper()
    assert payload["status"] == contract.BLOCKED_STATUS
    reference = payload["block_o_execution_authorization_matrix_reference"]
    assert reference["source_matrix_read_by_17_3"] is True
    assert reference["execution_authorization_contract_built_by_17_3"] is True
    assert reference["execution_authorization_contract_ready_by_17_3"] is False
    assert reference["ready_for_functional_preview_17_4"] is False
    evidence = payload["non_execution_contract_evidence"]
    assert evidence["source_matrix_read"] is True
    assert evidence["execution_authorization_contract_built"] is True
    assert evidence["source_matrix_accepted"] is False
    assert evidence["block_o_remains_open"] is False
    decision = payload["fail_closed_contract_decision"]
    assert decision["execution_authorization_contract_in_17_3"] == "blocked"
    assert decision["execution_authorization_read_model_in_17_4"] == "blocked"
    assert decision["execution_authorization_granted_by_17_3"] is False
    assert payload["source_boundaries"]["matrix_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_4"] is False
    json.dumps(payload)


def test_identity_order_reference_and_json_serializable() -> None:
    payload = build_payload()
    assert list(payload) == contract.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_contract_kind"] == contract.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.3"
    assert payload["execution_authorization_contract_status"] == contract.CONTRACT_STATUS
    assert payload["execution_authorization_contract_decision"] == contract.CONTRACT_DECISION
    assert payload["execution_authorization_contract_ready"] is True
    assert payload["ready_for_block_o_4"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.4"
    assert payload["status"] == contract.STATUS
    assert payload["future_steps"] == contract.FUTURE_STEPS
    json.dumps(payload)


def test_source_matrix_is_consumed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    source = build_preview_block_o_execution_authorization_matrix()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(source)

    monkeypatch.setattr(contract, "build_preview_block_o_execution_authorization_matrix", fake)
    assert build_payload()["execution_authorization_contract_ready"] is True
    assert calls == 1


def test_two_domain_authorization_contract_rows() -> None:
    rows = build_payload()["domain_authorization_contract_rows"]
    assert len(rows) == 2
    assert [row["domain"] for row in rows] == ["packaging_release", "runtime_safety"]
    for row in rows:
        assert row["source_requirements_complete"] is False
        assert row["source_domain_ready"] is False
        assert row["source_execution_authorized_by_matrix"] is False
        assert row["contract_condition_satisfied"] is False
        assert row["execution_authorized_by_contract"] is False
        assert row["failure_policy"] == "fail_closed"


def test_seven_requirement_authorization_contract_rows() -> None:
    rows = build_payload()["requirement_authorization_contract_rows"]
    assert len(rows) == 7
    for row in rows:
        assert row["source_required"] is True
        assert row["source_present"] is False
        assert row["source_completed"] is False
        assert row["source_satisfied"] is False
        assert row["source_missing"] is True
        assert row["contract_condition_satisfied"] is False
        assert row["execution_authorized_by_contract"] is False


def test_invariant_authorization_contract() -> None:
    guard = build_payload()["invariant_authorization_contract"]
    assert guard["all_invariants_preserved"] is True
    assert guard["read_by_execution_authorization_contract"] is True
    assert guard["recalculated_by_execution_authorization_contract"] is False
    assert guard["invariants_alone_satisfy_contract"] is False
    assert guard["execution_authorized_by_contract"] is False


def test_exe_authorization_contract_preserves_inherited_lineage() -> None:
    guard = build_payload()["exe_authorization_contract"]
    assert guard["final_product_direction"] == "desktop_exe"
    assert (
        guard["block_o_authorization_matrix_result"]
        == "desktop_exe_direction_preserved_build_packaging_release_unauthorized"
    )
    assert guard["block_o_entry_contract_confirms_exe_direction"] is True
    assert guard["contract_confirms_desktop_exe_direction"] is True
    assert guard["execution_authorized_by_contract"] is False


def test_real_capability_contract_preserves_exact_blocked_map() -> None:
    state = build_payload()["real_capability_authorization_contract"]
    assert state["real_capability_status"] == contract.REAL_CAPABILITY_STATUS
    assert list(state["real_capability_status"]) == list(contract.REAL_CAPABILITY_STATUS)
    assert state["all_real_capabilities_contracted_blocked"] is True
    assert state["execution_authorized_by_contract"] is False


def test_fail_closed_contract_decision() -> None:
    decision = build_payload()["fail_closed_contract_decision"]
    assert decision["missing_source_matrix_policy"] == "fail_closed"
    assert decision["block_o_execution_authorization_matrix_in_17_2"] == "preserved"
    assert decision["execution_authorization_contract_in_17_3"] == "ready"
    assert decision["execution_authorization_read_model_in_17_4"] == "allowed"
    assert decision["execution_authorization_granted_by_17_3"] is False


def test_contract_summary_evidence_and_boundaries() -> None:
    payload = build_payload()
    summary = payload["contract_summary"]
    assert summary["source_matrix_accepted"] is True
    assert summary["all_contract_conditions_unsatisfied"] is True
    assert summary["all_execution_authorizations_false"] is True
    evidence = payload["non_execution_contract_evidence"]
    assert evidence["source_matrix_read"] is True
    assert evidence["source_matrix_accepted"] is True
    assert all(payload["contract_boundaries"].values())


@pytest.mark.parametrize("key", contract.EXPECTED_SOURCE_TOP_LEVEL_FIELDS)
def test_identity_missing_top_level_blocks(monkeypatch: pytest.MonkeyPatch, key: str) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source.pop(key)
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


@pytest.mark.parametrize(
    ("section", "replacement"),
    [
        ("block_o_read_model_reference", []),
        ("matrix_summary", []),
        ("domain_authorization_rows", {}),
        ("requirement_authorization_rows", {}),
        ("invariant_authorization_guard", []),
        ("exe_authorization_guard", []),
        ("real_capability_authorization_state", []),
        ("fail_closed_matrix_decision", []),
        ("non_execution_matrix_evidence", []),
        ("matrix_boundaries", []),
        ("source_boundaries", []),
    ],
)
def test_missing_or_non_dict_sections_block(
    monkeypatch: pytest.MonkeyPatch, section: str, replacement: Any
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source[section] = replacement
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["domain_authorization_rows"][0].__setitem__(
            "execution_authorized_by_matrix", True
        ),
        lambda s: s["requirement_authorization_rows"][0].__setitem__(
            "authorization_condition_met", True
        ),
        lambda s: s["invariant_authorization_guard"].__setitem__(
            "execution_authorized_by_matrix", True
        ),
        lambda s: s["exe_authorization_guard"].__setitem__("execution_authorized_by_matrix", True),
        lambda s: s["real_capability_authorization_state"]["real_capability_status"].__setitem__(
            next(iter(s["real_capability_authorization_state"]["real_capability_status"])),
            "allowed",
        ),
    ],
)
def test_authorization_bypass_blocks(monkeypatch: pytest.MonkeyPatch, mutate: Any) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    mutate(source)
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


@pytest.mark.parametrize("field", contract.INVARIANT_CONTRACT_FIELDS_17_3)
def test_invariant_field_shadowing_blocks(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["invariant_authorization_guard"][field] = False
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


@pytest.mark.parametrize("field", contract.EXE_CONTRACT_FIELDS_17_3)
def test_exe_field_shadowing_blocks(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["exe_authorization_guard"][field] = False
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


def test_cross_section_isolation_invalid_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["invariant_authorization_guard"] = {"bad": True}
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    assert (
        payload["invariant_authorization_contract"]["read_by_execution_authorization_contract"]
        is False
    )
    assert payload["exe_authorization_contract"]["read_by_execution_authorization_contract"] is True
    assert (
        payload["real_capability_authorization_contract"][
            "all_real_capabilities_contracted_blocked"
        ]
        is True
    )


def test_cross_section_isolation_invalid_real_map(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["real_capability_authorization_state"]["real_capability_status"]["release_execution"] = (
        "allowed"
    )
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    assert payload["real_capability_authorization_contract"]["real_capability_status"] == {}
    assert len(payload["requirement_authorization_contract_rows"]) == 7
    assert payload["exe_authorization_contract"]["read_by_execution_authorization_contract"] is True


@pytest.mark.parametrize("bad", [object(), set(), tuple(), {"x": object()}])
def test_non_json_source_blocks(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["matrix_summary"] = bad
    before = repr(source)
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())
    assert repr(source) == before


def test_cycle_and_depth_block_without_recursion_error(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["source_boundaries"]["cycle"] = source["source_boundaries"]
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())
    deep: dict[str, Any] = {}
    cursor = deep
    for _ in range(1500):
        cursor["x"] = {}
        cursor = cursor["x"]
    source = build_preview_block_o_execution_authorization_matrix()
    source["source_boundaries"] = deep
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    assert_contract_blocked(build_payload())


def test_ast_guard() -> None:
    module_text = Path(contract.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_text)
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert [node.module for node in imports] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_matrix",
    ]
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_calls = sorted({node.func.id for node in call_nodes if isinstance(node.func, ast.Name)})
    attribute_calls = sorted(
        {node.func.attr for node in call_nodes if isinstance(node.func, ast.Attribute)}
    )
    assert name_calls == [
        "_all_plain_json",
        "_copy_plain",
        "_domain_contract_rows",
        "_exact_plain_matches",
        "_exe_contract",
        "_invariant_contract",
        "_matrix_reference",
        "_no_shadowing",
        "_plain_dict_section",
        "_plain_list_section",
        "_real_contract",
        "_requirement_contract_rows",
        "_safe_top_level_source",
        "_section_valid",
        "_source_identity_valid",
        "all",
        "any",
        "build_preview_block_o_execution_authorization_matrix",
        "enumerate",
        "id",
        "len",
        "list",
        "reversed",
        "set",
        "str",
        "type",
        "zip",
    ]
    assert attribute_calls == [
        "add",
        "append",
        "endswith",
        "get",
        "items",
        "keys",
        "pop",
        "remove",
        "update",
        "upper",
    ]
    builder_calls = [
        node
        for node in call_nodes
        if isinstance(node.func, ast.Name)
        and node.func.id == "build_preview_block_o_execution_authorization_matrix"
    ]
    assert len(builder_calls) == 1
    forbidden_name_calls = {"open", "eval", "exec", "compile", "__import__"}
    assert not forbidden_name_calls.intersection(name_calls)
    forbidden_attribute_calls = {"run", "Popen", "system", "start", "submit", "cancel", "replace"}
    assert not forbidden_attribute_calls.intersection(attribute_calls)


EXACT_SECTION_CASES: list[tuple[str, Any]] = [
    ("block_o_read_model_reference", contract.EXPECTED_REFERENCE),
    ("matrix_summary", contract.EXPECTED_SUMMARY),
    ("domain_authorization_rows", contract.EXPECTED_DOMAINS),
    ("requirement_authorization_rows", contract.EXPECTED_REQUIREMENTS),
    ("invariant_authorization_guard", contract.EXPECTED_INVARIANT_GUARD),
    ("exe_authorization_guard", contract.EXPECTED_EXE_GUARD),
    ("real_capability_authorization_state", contract.EXPECTED_REAL_CAPABILITY_STATE),
    ("fail_closed_matrix_decision", contract.EXPECTED_FAIL_CLOSED_DECISION),
    ("non_execution_matrix_evidence", contract.EXPECTED_EVIDENCE),
    ("matrix_boundaries", contract.EXPECTED_MATRIX_BOUNDARIES),
    ("source_boundaries", contract.EXPECTED_SOURCE_BOUNDARIES),
    ("future_steps", contract.EXPECTED_FUTURE_STEPS),
]


def _blocked_from_source(monkeypatch: pytest.MonkeyPatch, source: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    return payload


def _mutate_first_scalar(container: Any) -> None:
    if type(container) is dict:
        first_key = next(iter(container))
        value = container[first_key]
        if type(value) is bool:
            container[first_key] = 1 if value is True else 0
        elif type(value) is int:
            container[first_key] = float(value)
        elif type(value) is str:
            container[first_key] = f"{value}_sentinel"
        elif type(value) is list:
            container[first_key] = tuple(value)
        elif type(value) is dict:
            _mutate_first_scalar(value)
        else:
            container[first_key] = object()
    elif type(container) is list:
        first = container[0]
        if type(first) is dict:
            _mutate_first_scalar(first)
        elif type(first) is str:
            container[0] = f"{first}_sentinel"
        elif type(first) is bool:
            container[0] = 1 if first is True else 0
        elif type(first) is int:
            container[0] = float(first)


@pytest.mark.parametrize("field", contract.EXPECTED_IDENTITY)
def test_exact_identity_scalar_fields_block(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    current = source[field]
    if type(current) is bool:
        source[field] = 1 if current is True else 0
    elif type(current) is str:
        source[field] = f"{current}_sentinel"
    elif type(current) is list:
        source[field] = [*current, "sentinel"]
    else:
        source[field] = object()
    _blocked_from_source(monkeypatch, source)


def test_extra_and_reordered_top_level_block(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["extra_17_3_sentinel"] = False
    _blocked_from_source(monkeypatch, source)
    source = build_preview_block_o_execution_authorization_matrix()
    value = source.pop("schema_version")
    source["schema_version"] = value
    _blocked_from_source(monkeypatch, source)


@pytest.mark.parametrize(("section", "expected"), EXACT_SECTION_CASES)
def test_exact_sections_missing_non_container_extra_missing_reordered_and_value_block(
    monkeypatch: pytest.MonkeyPatch, section: str, expected: Any
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source.pop(section)
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    source[section] = {} if type(expected) is list else []
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    if type(source[section]) is dict:
        source[section]["extra_17_3_sentinel"] = False
    else:
        source[section].append("extra_17_3_sentinel")
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    if type(source[section]) is dict:
        source[section].pop(next(iter(source[section])))
    else:
        source[section].pop()
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    if type(source[section]) is dict:
        first_key = next(iter(source[section]))
        first_value = source[section].pop(first_key)
        source[section][first_key] = first_value
    else:
        if len(source[section]) == 1:
            source[section] = [*source[section], "sentinel"]
        else:
            source[section] = list(reversed(source[section]))
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    _mutate_first_scalar(source[section])
    _blocked_from_source(monkeypatch, source)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["matrix_summary"].__setitem__("runtime_started_by_17_2", True),
        lambda s: s["block_o_read_model_reference"].__setitem__(
            "authorization_granted_by_17_2", True
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__(
            "execution_authorized_by_matrix", True
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__(
            "all_real_capabilities_blocked", False
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__(
            "real_capability_status_inherited_from_17_1", False
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__(
            "real_capability_status_modified_by_17_2", True
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__(
            "real_capabilities_opened_by_17_2", True
        ),
        lambda s: s["real_capability_authorization_state"].__setitem__("state_result", "sentinel"),
        lambda s: s["fail_closed_matrix_decision"].__setitem__(
            "execution_authorization_granted_by_17_2", True
        ),
        lambda s: s["non_execution_matrix_evidence"].__setitem__(
            "all_execution_authorizations_false", False
        ),
        lambda s: s["matrix_boundaries"].__setitem__("cannot_grant_authorization", False),
        lambda s: s.__setitem__("source_boundaries", {"anything": True}),
    ],
)
def test_exact_authorization_bypass_sentinels_block(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    mutate(source)
    _blocked_from_source(monkeypatch, source)


class PretendBlocked(str):
    def __eq__(self, other: object) -> bool:
        return other == "blocked"


class PretendRealCapabilityKey(str):
    def __eq__(self, other: object) -> bool:
        return other == "release_execution"

    def __hash__(self) -> int:
        return hash("release_execution")


@pytest.mark.parametrize("custom_mode", ["value", "key"])
def test_custom_equality_real_capability_blocks_and_clears_maps(
    monkeypatch: pytest.MonkeyPatch, custom_mode: str
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    status = source["real_capability_authorization_state"]["real_capability_status"]
    if custom_mode == "value":
        status["release_execution"] = PretendBlocked("blocked")
    else:
        value = status.pop("release_execution")
        status[PretendRealCapabilityKey("release_execution")] = value
    payload = _blocked_from_source(monkeypatch, source)
    assert payload["real_capability_authorization_contract"]["real_capability_status"] == {}
    assert payload["fail_closed_contract_decision"]["real_capability_status"] == {}
    json.dumps(payload)


@pytest.mark.parametrize("field", contract.SOURCE_BOUNDARY_FIELDS_17_3)
def test_source_boundary_field_shadowing_blocks(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["source_boundaries"][field] = False
    payload = _blocked_from_source(monkeypatch, source)
    assert payload["source_boundaries"]["matrix_source_preserved"] is False


def test_cross_section_isolation_invalid_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["exe_authorization_guard"] = {"bad": True}
    payload = _blocked_from_source(monkeypatch, source)
    assert (
        payload["exe_authorization_contract"]["read_by_execution_authorization_contract"] is False
    )
    assert (
        payload["invariant_authorization_contract"]["read_by_execution_authorization_contract"]
        is True
    )
    assert (
        payload["real_capability_authorization_contract"][
            "all_real_capabilities_contracted_blocked"
        ]
        is True
    )


def test_cross_section_isolation_invalid_requirements_preserves_domain_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["requirement_authorization_rows"] = []
    payload = _blocked_from_source(monkeypatch, source)
    assert payload["requirement_authorization_contract_rows"] == []
    for row in payload["domain_authorization_contract_rows"]:
        assert row["source_capability_count"] in (22, 18)
        assert row["source_all_capabilities_read"] is True
        assert row["source_all_capabilities_not_ready"] is True
        assert row["source_all_capabilities_blocked"] is True
        assert row["contract_classification"] == "source_invalid"
        assert row["execution_authorized_by_contract"] is False
    assert (
        payload["invariant_authorization_contract"]["read_by_execution_authorization_contract"]
        is True
    )
    assert payload["exe_authorization_contract"]["read_by_execution_authorization_contract"] is True
    assert (
        payload["real_capability_authorization_contract"][
            "all_real_capabilities_contracted_blocked"
        ]
        is True
    )


def test_cross_section_isolation_non_json_summary_keeps_other_local_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["matrix_summary"]["bad"] = object()
    payload = _blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_contract_evidence"]
    assert evidence["summary_read_valid"] is False
    assert evidence["exe_read_valid"] is True
    assert evidence["invariant_read_valid"] is True
    assert evidence["real_capability_map_valid"] is True


def test_valid_source_boundaries_are_copied_on_independent_global_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source["matrix_summary"]["runtime_started_by_17_2"] = True
    payload = _blocked_from_source(monkeypatch, source)
    assert payload["source_boundaries"]["source_block_o_read_model"] == "FUNCTIONAL-PREVIEW-17.1"
    assert payload["source_boundaries"]["matrix_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_4"] is False


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("matrix_summary", object()),
        ("matrix_summary", lambda: None),
        ("matrix_summary", set()),
        ("matrix_summary", tuple()),
    ],
)
def test_malformed_object_callable_set_tuple_block(
    monkeypatch: pytest.MonkeyPatch, section: str, value: Any
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    source[section] = value
    _blocked_from_source(monkeypatch, source)


def test_cyclic_list_and_shared_acyclic_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    cycle: list[Any] = []
    cycle.append(cycle)
    source["future_steps"] = cycle
    _blocked_from_source(monkeypatch, source)

    source = build_preview_block_o_execution_authorization_matrix()
    shared = {"shared": True}
    source["matrix_summary"]["shared_a"] = shared
    source["matrix_summary"]["shared_b"] = shared
    _blocked_from_source(monkeypatch, source)


def test_deep_invariant_exe_reference_and_source_boundaries_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for section in [
        "invariant_authorization_guard",
        "exe_authorization_guard",
        "block_o_read_model_reference",
        "source_boundaries",
    ]:
        source = build_preview_block_o_execution_authorization_matrix()
        deep: dict[str, Any] = {}
        cursor = deep
        for _ in range(1500):
            cursor["x"] = {}
            cursor = cursor["x"]
        source[section] = deep
        _blocked_from_source(monkeypatch, source)


class BombTopLevelKey(str):
    def __eq__(self, other: object) -> bool:
        raise RuntimeError("custom equality must not run")

    def __hash__(self) -> int:
        return hash(str(self))


class LyingTopLevelKey(str):
    equality_called = False

    def __eq__(self, other: object) -> bool:
        type(self).equality_called = True
        return other == "schema_version"

    def __hash__(self) -> int:
        return hash(str(self))


def _replace_top_level_key(source: dict[Any, Any], old_key: str, new_key: Any) -> dict[Any, Any]:
    replaced: dict[Any, Any] = {}
    for key, value in source.items():
        replaced[new_key if key == old_key else key] = value
    return replaced


def test_bomb_top_level_identity_key_blocks_without_equality_or_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    raw_source = _replace_top_level_key(source, "schema_version", BombTopLevelKey("schema_version"))
    raw_keys_before = list(raw_source.keys())
    raw_repr_before = repr(raw_source)
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: raw_source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    assert list(raw_source.keys()) == raw_keys_before
    assert repr(raw_source) == raw_repr_before
    json.dumps(payload)


def test_bomb_top_level_section_key_blocks_and_preserves_other_local_validity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = build_preview_block_o_execution_authorization_matrix()
    raw_source = _replace_top_level_key(source, "matrix_summary", BombTopLevelKey("matrix_summary"))
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: raw_source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    evidence = payload["non_execution_contract_evidence"]
    assert evidence["summary_read_valid"] is False
    assert evidence["exe_read_valid"] is True
    assert evidence["invariant_read_valid"] is True
    assert evidence["real_capability_map_valid"] is True
    json.dumps(payload)


def test_lying_top_level_key_is_rejected_without_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    LyingTopLevelKey.equality_called = False
    source = build_preview_block_o_execution_authorization_matrix()
    raw_source = _replace_top_level_key(
        source, "schema_version", LyingTopLevelKey("schema_version")
    )
    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_matrix", lambda: raw_source
    )
    payload = build_payload()
    assert_contract_blocked(payload)
    assert LyingTopLevelKey.equality_called is False
    json.dumps(payload)


def test_iterative_all_plain_json_deep_cycle_and_shared_reference() -> None:
    deep: dict[str, Any] = {}
    cursor = deep
    for _ in range(1500):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    assert contract._all_plain_json(deep) is True
    assert (
        contract._all_plain_json(
            deep,
            max_depth=contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH,
        )
        is False
    )

    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    assert contract._all_plain_json(cyclic_dict) is False

    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert contract._all_plain_json(cyclic_list) is False

    shared = {"ok": True}
    assert contract._all_plain_json({"a": shared, "b": shared}) is True
    assert contract._all_plain_json(contract.EXPECTED_SUMMARY) is True
