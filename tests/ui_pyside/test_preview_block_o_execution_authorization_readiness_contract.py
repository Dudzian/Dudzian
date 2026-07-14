from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app import preview_block_o_execution_authorization_readiness_contract as contract
from ui.pyside_app.preview_block_o_execution_authorization_readiness_matrix import (
    build_preview_block_o_execution_authorization_readiness_matrix,
)


def build_with_source(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        contract,
        "build_preview_block_o_execution_authorization_readiness_matrix",
        lambda: source,
    )
    return contract.build_preview_block_o_execution_authorization_readiness_contract()


def assert_blocked_json_safe(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_readiness_contract_ready"] is False
    assert payload["ready_for_block_o_7"] is False
    assert payload["status"] == contract.BLOCKED_STATUS
    json.dumps(payload)


def test_expected_source_matches_current_17_5_builder() -> None:
    assert (
        contract.EXPECTED_SOURCE == build_preview_block_o_execution_authorization_readiness_matrix()
    )


def test_nominal_identity_order_reference_and_json() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    assert list(payload) == contract.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_readiness_contract_kind"] == contract.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.6"
    assert (
        payload["execution_authorization_readiness_contract_status"]
        == contract.READINESS_CONTRACT_STATUS
    )
    assert (
        payload["execution_authorization_readiness_contract_decision"]
        == contract.READINESS_CONTRACT_DECISION
    )
    assert payload["execution_authorization_readiness_contract_ready"] is True
    assert payload["ready_for_block_o_7"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.7"
    assert payload["status"] == contract.STATUS
    ref = payload["block_o_execution_authorization_readiness_matrix_reference"]
    assert (
        ref["source_block_o_execution_authorization_readiness_matrix_step"]
        == "FUNCTIONAL-PREVIEW-17.5"
    )
    assert ref["source_readiness_matrix_read_by_17_6"] is True
    assert ref["execution_authorization_readiness_contract_ready_by_17_6"] is True
    assert ref["ready_for_functional_preview_17_7"] is True
    assert ref["validation_performed_by_17_6"] is False
    assert ref["runtime_started_by_17_6"] is False
    json.dumps(payload)


def test_builder_calls_source_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return copy.deepcopy(contract.EXPECTED_SOURCE)

    monkeypatch.setattr(
        contract, "build_preview_block_o_execution_authorization_readiness_matrix", fake
    )
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    assert calls == 1
    assert payload["ready_for_block_o_7"] is True


def test_domain_requirement_rows_and_flags() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    domains = payload["domain_authorization_readiness_contract_rows"]
    assert [r["domain"] for r in domains] == ["packaging_release", "runtime_safety"]
    assert [r["source_capability_count"] for r in domains] == [22, 18]
    for row in domains:
        assert row["contract_requires_all_requirements_ready"] is True
        assert row["invariants_ready_for_contract"] is True
        assert row["readiness_contract_condition_satisfied"] is False
        assert row["execution_ready_by_readiness_contract"] is False
        assert row["execution_authorized_by_readiness_contract"] is False
        assert row["readiness_contract_classification"] == "contracted_missing_readiness_conditions"
    reqs = payload["requirement_authorization_readiness_contract_rows"]
    assert len(reqs) == 7
    assert [r["requirement_id"] for r in reqs] == [
        r["requirement_id"]
        for r in contract.EXPECTED_SOURCE["requirement_authorization_readiness_rows"]
    ]
    for row in reqs:
        assert row["contract_requires_present"] is True
        assert row["requirement_present_for_contract"] is False
        assert row["requirement_completed_for_contract"] is False
        assert row["requirement_satisfied_for_contract"] is False
        assert row["future_explicit_step_ready_for_contract"] is False
        assert row["execution_authorized_by_readiness_contract"] is False


def test_invariant_exe_real_capability_fail_closed_summary_evidence_boundaries() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    inv = payload["invariant_authorization_readiness_contract"]
    assert inv["source_readiness_guard_preserved"] is True
    assert inv["invariants_preserved_for_readiness_contract"] is True
    assert inv["invariants_alone_satisfy_readiness_contract"] is False
    exe = payload["exe_authorization_readiness_contract"]
    assert exe["contract_confirms_desktop_exe_direction"] is True
    assert exe["contract_is_not_build_readiness_grant"] is True
    assert exe["build_ready_by_readiness_contract"] is False
    real = payload["real_capability_authorization_readiness_contract"]
    assert (
        real["real_capability_status"]
        == contract.EXPECTED_SOURCE["real_capability_authorization_readiness_state"][
            "real_capability_status"
        ]
    )
    assert set(real["real_capability_status"].values()) == {"blocked"}
    assert real["all_real_capabilities_blocked_for_readiness_contract"] is True
    fail = payload["fail_closed_readiness_contract_decision"]
    assert fail["block_o_execution_authorization_readiness_matrix_in_17_5"] == "preserved"
    assert fail["execution_authorization_readiness_contract_in_17_6"] == "ready"
    assert fail["execution_authorization_readiness_read_model_in_17_7"] == "allowed"
    assert fail["execution_readiness_granted_by_17_6"] is False
    summary = payload["readiness_contract_summary"]
    assert summary["all_contract_readiness_conditions_false"] is True
    assert summary["all_contract_authorization_grants_false"] is True
    evidence = payload["non_execution_readiness_contract_evidence"]
    assert all(evidence[k] is True for k in evidence if k.endswith("_valid"))
    assert evidence["validation_performed_by_17_6"] is False
    assert set(payload["readiness_contract_boundaries"].values()) == {True}


@pytest.mark.parametrize("mutation", ["identity", "extra", "missing", "reordered"])
def test_top_level_sentinels_block(monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    if mutation == "identity":
        source["step"] = "17.x"
    elif mutation == "extra":
        source["extra"] = True
    elif mutation == "missing":
        source.pop("future_steps")
    else:
        source = {k: source[k] for k in reversed(list(source))}
    assert_blocked_json_safe(build_with_source(monkeypatch, source))


@pytest.mark.parametrize(
    "key",
    [
        "readiness_matrix_summary",
        "domain_authorization_readiness_rows",
        "requirement_authorization_readiness_rows",
        "invariant_authorization_readiness_guard",
        "exe_authorization_readiness_guard",
        "real_capability_authorization_readiness_state",
        "fail_closed_readiness_decision",
        "non_execution_readiness_evidence",
        "readiness_matrix_boundaries",
        "source_boundaries",
        "future_steps",
    ],
)
def test_section_extra_missing_reordered_type_value_block(
    monkeypatch: pytest.MonkeyPatch, key: str
) -> None:
    for mutate in ("type", "value"):
        source = copy.deepcopy(contract.EXPECTED_SOURCE)
        source[key] = [] if mutate == "type" else copy.deepcopy(source[key])
        if mutate == "value":
            if type(source[key]) is dict:
                source[key][next(iter(source[key]))] = "wrong"
            elif type(source[key]) is list:
                source[key] = list(reversed(source[key])) + ["unexpected"]
        assert_blocked_json_safe(build_with_source(monkeypatch, source))


def test_invalid_requirements_preserve_domain_counts_and_invariant_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    source["requirement_authorization_readiness_rows"] = []
    payload = build_with_source(monkeypatch, source)
    assert_blocked_json_safe(payload)
    rows = payload["domain_authorization_readiness_contract_rows"]
    assert [r["source_capability_count"] for r in rows] == [22, 18]
    assert all(r["invariants_ready_for_contract"] is True for r in rows)
    assert all(r["readiness_contract_classification"] == "source_invalid" for r in rows)
    assert payload["readiness_contract_summary"]["all_contract_readiness_conditions_false"] is False
    assert (
        payload["invariant_authorization_readiness_contract"][
            "invariants_preserved_for_readiness_contract"
        ]
        is True
    )


def test_shadowing_bomb_lying_keys_subclasses_and_malformed_are_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BombKey(str):
        def __eq__(self, other: object) -> bool:
            raise AssertionError("boom")

        def __hash__(self) -> int:
            return str.__hash__(self)

    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    cases: list[Any] = [object(), lambda: None, {"x"}, ("x",), DictSubclass(), ListSubclass()]
    for bad in cases:
        assert_blocked_json_safe(build_with_source(monkeypatch, bad))
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    source["invariant_authorization_readiness_guard"][
        "read_by_execution_authorization_readiness_contract"
    ] = True
    assert_blocked_json_safe(build_with_source(monkeypatch, source))
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    source[BombKey("schema_version")] = source.pop("schema_version")
    assert_blocked_json_safe(build_with_source(monkeypatch, source))


def test_custom_blocked_key_value_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    class Liar(str):
        def __eq__(self, other: object) -> bool:
            return True

        def __hash__(self) -> int:
            return str.__hash__(self)

    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    real = source["real_capability_authorization_readiness_state"]["real_capability_status"]
    real[Liar("fake")] = "blocked"
    assert_blocked_json_safe(build_with_source(monkeypatch, source))
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    source["real_capability_authorization_readiness_state"]["real_capability_status"]["build"] = (
        Liar("blocked")
    )
    assert_blocked_json_safe(build_with_source(monkeypatch, source))


def test_cycles_depth_and_shared_references(monkeypatch: pytest.MonkeyPatch) -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    assert contract._all_plain_json(cyclic) is False
    deep: dict[str, Any] = {}
    node = deep
    for i in range(1500):
        node["x"] = {}
        node = node["x"]
    assert contract._all_plain_json(deep) is True
    assert (
        contract._all_plain_json(deep, max_depth=contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    )
    shared: dict[str, Any] = {"x": 1}
    assert contract._all_plain_json({"a": shared, "b": shared}) is True
    source = copy.deepcopy(contract.EXPECTED_SOURCE)
    source["readiness_matrix_summary"] = cyclic
    assert_blocked_json_safe(build_with_source(monkeypatch, source))


def test_ast_guard_and_forbidden_raw_tokens() -> None:
    path = Path(contract.__file__)
    text = path.read_text()
    assert "create_order" not in text
    assert "fetch_balance" not in text
    assert "ccxt" not in text
    tree = ast.parse(text)
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert len(imports) == 3
    assert [i.module for i in imports] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_readiness_matrix",
    ]
    builder_calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "build_preview_block_o_execution_authorization_readiness_matrix"
    ]
    assert len(builder_calls) == 1
    name_calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    attribute_calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert name_calls == {
        "_all_plain_json",
        "_contains_owned_field",
        "_copy_plain",
        "_domain_rows",
        "_exact_plain_matches",
        "_matrix_reference",
        "_no_shadowing",
        "_plain_dict_section",
        "_plain_list_section",
        "_requirement_rows",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "_state_with_fields",
        "all",
        "build_preview_block_o_execution_authorization_readiness_matrix",
        "enumerate",
        "id",
        "len",
        "list",
        "reversed",
        "set",
        "type",
        "zip",
    }
    assert attribute_calls == {
        "add",
        "append",
        "get",
        "items",
        "keys",
        "pop",
        "remove",
        "update",
        "upper",
    }
    forbidden = {
        "open",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "system",
        "popen",
        "run",
        "call",
        "check_call",
        "check_output",
        "socket",
        "urlopen",
        "request",
        "getenv",
        "environ",
        "validate",
        "validation",
        "confirm",
        "confirmation",
        "grant_readiness",
        "grant_authorization",
        "authorize",
        "open_gate",
        "mutate_gate",
        "runtime",
        "submit_order",
        "cancel_order",
        "replace_order",
        "package",
        "pyinstaller",
        "build",
        "artifact",
        "release",
        "qml",
        "bridge",
        "gateway",
        "controller",
    }
    assert not (name_calls & forbidden)
    assert not (attribute_calls & forbidden)


def test_direct_smoke_contract_false_flags_and_split_real_keys() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    json.dumps(payload)
    assert payload["ready_for_block_o_7"] is True
    assert len(payload["domain_authorization_readiness_contract_rows"]) == 2
    assert len(payload["requirement_authorization_readiness_contract_rows"]) == 7
    encoded = json.dumps(
        payload["real_capability_authorization_readiness_contract"]["real_capability_status"]
    )
    assert "create_order" in encoded and "fetch_balance" in encoded and "ccxt" in encoded
    for section in payload.values():
        if type(section) is dict:
            for key, value in section.items():
                if key.endswith(
                    (
                        "condition_satisfied",
                        "ready_by_readiness_contract",
                        "authorized_by_readiness_contract",
                    )
                ):
                    assert value is False


def clone_source() -> dict[str, Any]:
    return copy.deepcopy(contract.EXPECTED_SOURCE)


def assert_malformed_blocks_without_mutation(
    monkeypatch: pytest.MonkeyPatch, source: dict[str, Any], *, compare_source: bool = True
) -> dict[str, Any]:
    before = copy.deepcopy(source) if compare_source else None
    payload = build_with_source(monkeypatch, source)
    if compare_source:
        assert source == before
    assert_blocked_json_safe(payload)
    return payload


def test_exact_contract_row_schemas_and_preserved_source_facts() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    for row in payload["domain_authorization_readiness_contract_rows"]:
        assert list(row) == contract.DOMAIN_READINESS_CONTRACT_ROW_FIELDS
        assert row["source_readiness_classification"] == "missing_requirements_execution_not_ready"
        assert row["source_readiness_result"].endswith("execution_unauthorized")
    for row in payload["requirement_authorization_readiness_contract_rows"]:
        assert list(row) == contract.REQUIREMENT_READINESS_CONTRACT_ROW_FIELDS
        assert row["source_requirement_readiness_condition_met"] is False
        assert row["source_missing_blocks_execution_for_readiness"] is True
        assert row["source_future_explicit_step_required"] is True
        assert row["source_readiness_classification"] == "missing_requirement_execution_not_ready"
        assert (
            row["source_readiness_result"] == "missing_requirement_readiness_execution_unauthorized"
        )
        assert "requirement_ready_for_contract" not in row


def iter_container_ids(value: Any) -> list[int]:
    result: list[int] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            result.append(id(item))
            stack.extend(item.values())
        elif type(item) is list:
            result.append(id(item))
            stack.extend(item)
    return result


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    payload = contract.build_preview_block_o_execution_authorization_readiness_contract()
    ids = iter_container_ids(payload)
    assert len(ids) == len(set(ids))
    real_map = payload["real_capability_authorization_readiness_contract"]["real_capability_status"]
    decision_map = payload["fail_closed_readiness_contract_decision"]["real_capability_status"]
    assert real_map == decision_map
    assert real_map is not decision_map
    key = next(iter(real_map))
    real_map[key] = "changed"
    assert decision_map[key] == "blocked"


def test_independent_builder_calls_do_not_share_mutable_payload_or_expected_source() -> None:
    first = contract.build_preview_block_o_execution_authorization_readiness_contract()
    second = contract.build_preview_block_o_execution_authorization_readiness_contract()
    assert first is not second
    first_domain_ids = first["domain_authorization_readiness_contract_rows"][0][
        "required_requirement_ids"
    ]
    second_domain_ids = second["domain_authorization_readiness_contract_rows"][0][
        "required_requirement_ids"
    ]
    assert first_domain_ids is not second_domain_ids
    first_domain_ids.append("mutated")
    assert "mutated" not in second_domain_ids
    assert (
        "mutated"
        not in contract.EXPECTED_SOURCE["domain_authorization_readiness_rows"][0][
            "required_requirement_ids"
        ]
    )
    first_real_map = first["real_capability_authorization_readiness_contract"][
        "real_capability_status"
    ]
    second_real_map = second["real_capability_authorization_readiness_contract"][
        "real_capability_status"
    ]
    real_key = next(iter(first_real_map))
    first_real_map[real_key] = "changed"
    assert second_real_map[real_key] == "blocked"
    assert (
        contract.EXPECTED_SOURCE["real_capability_authorization_readiness_state"][
            "real_capability_status"
        ][real_key]
        == "blocked"
    )


def test_summary_separates_readiness_and_authorization_grants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nominal = contract.build_preview_block_o_execution_authorization_readiness_contract()[
        "readiness_contract_summary"
    ]
    assert nominal["all_source_readiness_conditions_false"] is True
    assert nominal["all_source_readiness_grants_false"] is True
    assert nominal["all_source_authorization_grants_false"] is True
    assert nominal["all_contract_readiness_conditions_false"] is True
    assert nominal["all_contract_readiness_grants_false"] is True
    assert nominal["all_contract_authorization_grants_false"] is True
    source = clone_source()
    source["fail_closed_readiness_decision"]["failed_readiness_matrix_policy"] = "open"
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source)
    summary = payload["readiness_contract_summary"]
    assert summary["all_source_readiness_conditions_false"] is True
    assert summary["all_source_readiness_grants_false"] is False
    assert summary["all_source_authorization_grants_false"] is False
    assert summary["all_contract_readiness_conditions_false"] is True
    assert summary["all_contract_readiness_grants_false"] is True
    assert summary["all_contract_authorization_grants_false"] is True
    source = clone_source()
    source["source_boundaries"]["allowed_imports_only"] = False
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source)
    summary = payload["readiness_contract_summary"]
    assert summary["all_contract_readiness_conditions_false"] is True
    assert summary["all_contract_authorization_grants_false"] is True


@pytest.mark.parametrize(
    ("section", "fields"),
    [
        ("invariant_authorization_readiness_guard", contract.INVARIANT_FIELDS_17_6),
        ("exe_authorization_readiness_guard", contract.EXE_FIELDS_17_6),
        ("source_boundaries", contract.SOURCE_BOUNDARY_FIELDS_17_6),
    ],
)
def test_all_owned_fields_are_shadowing_guarded(
    monkeypatch: pytest.MonkeyPatch, section: str, fields: list[str]
) -> None:
    for field in fields:
        source = clone_source()
        source[section][field] = "shadowed"
        payload = assert_malformed_blocks_without_mutation(monkeypatch, source)
        assert payload["non_execution_readiness_contract_evidence"][
            "source_boundaries_read_valid"
        ] in (
            True,
            False,
        )


class _CollisionKey:
    equality_calls = 0

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return False


class BombTopLevelKey(_CollisionKey):
    __hash__ = _CollisionKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise AssertionError("top-level equality called")


class LyingTopLevelKey(_CollisionKey):
    __hash__ = _CollisionKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return True


class BombNestedKey(_CollisionKey):
    __hash__ = _CollisionKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise AssertionError("nested equality called")


class LyingNestedKey(_CollisionKey):
    __hash__ = _CollisionKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return True


@pytest.mark.parametrize("key_type", [BombTopLevelKey, LyingTopLevelKey])
def test_bomb_lying_top_level_identity_collision_does_not_call_custom_equality(
    monkeypatch: pytest.MonkeyPatch, key_type: type[_CollisionKey]
) -> None:
    key_type.equality_calls = 0
    source = clone_source()
    value = source.pop("schema_version")
    collision_key = key_type("schema_version")
    source[collision_key] = value
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source, compare_source=False)
    assert key_type.equality_calls == 0
    assert any(key is collision_key for key in source)
    evidence = payload["non_execution_readiness_contract_evidence"]
    assert evidence["summary_read_valid"] is True
    assert evidence["domain_rows_read_valid"] is True


@pytest.mark.parametrize("key_type", [BombTopLevelKey, LyingTopLevelKey])
def test_bomb_lying_top_level_section_collision_does_not_call_custom_equality(
    monkeypatch: pytest.MonkeyPatch, key_type: type[_CollisionKey]
) -> None:
    key_type.equality_calls = 0
    source = clone_source()
    value = source.pop("readiness_matrix_summary")
    collision_key = key_type("readiness_matrix_summary")
    source[collision_key] = value
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source, compare_source=False)
    assert key_type.equality_calls == 0
    assert any(key is collision_key for key in source)
    evidence = payload["non_execution_readiness_contract_evidence"]
    assert evidence["summary_read_valid"] is False
    assert evidence["domain_rows_read_valid"] is True


@pytest.mark.parametrize("key_type", [BombNestedKey, LyingNestedKey])
@pytest.mark.parametrize(
    ("section", "target_field", "invalid_flag", "valid_flags"),
    [
        (
            "invariant_authorization_readiness_guard",
            "contract_requires_invariants_preserved",
            "invariant_read_valid",
            ["exe_read_valid", "source_boundaries_read_valid"],
        ),
        (
            "exe_authorization_readiness_guard",
            "contract_is_not_build_readiness_grant",
            "exe_read_valid",
            ["invariant_read_valid", "source_boundaries_read_valid"],
        ),
        (
            "source_boundaries",
            "can_feed_17_7",
            "source_boundaries_read_valid",
            ["invariant_read_valid", "exe_read_valid"],
        ),
    ],
)
def test_bomb_lying_nested_owned_field_collision_does_not_call_custom_equality(
    monkeypatch: pytest.MonkeyPatch,
    key_type: type[_CollisionKey],
    section: str,
    target_field: str,
    invalid_flag: str,
    valid_flags: list[str],
) -> None:
    key_type.equality_calls = 0
    source = clone_source()
    collision_key = key_type(target_field)
    source[section][collision_key] = "value"
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source, compare_source=False)
    assert key_type.equality_calls == 0
    assert any(key is collision_key for key in source[section])
    evidence = payload["non_execution_readiness_contract_evidence"]
    assert evidence[invalid_flag] is False
    for flag in valid_flags:
        assert evidence[flag] is True


def test_targeted_section_and_row_malformed_sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    mutations: list[tuple[str, Any]] = []
    for section in [
        "readiness_matrix_summary",
        "invariant_authorization_readiness_guard",
        "exe_authorization_readiness_guard",
        "source_boundaries",
    ]:
        source = clone_source()
        source[section]["unexpected"] = True
        mutations.append(("extra", source))
        source = clone_source()
        source[section].pop(next(iter(source[section])))
        mutations.append(("missing", source))
        source = clone_source()
        source[section] = {k: source[section][k] for k in reversed(list(source[section]))}
        mutations.append(("reordered", source))
    source = clone_source()
    source["readiness_matrix_summary"]["readiness_matrix_built"] = 1
    mutations.append(("bool_to_int", source))
    source = clone_source()
    source["domain_authorization_readiness_rows"][0]["source_capability_count"] = 22.0
    mutations.append(("int_to_float", source))
    for rows_key in [
        "domain_authorization_readiness_rows",
        "requirement_authorization_readiness_rows",
    ]:
        source = clone_source()
        source[rows_key] = source[rows_key][:-1]
        mutations.append(("wrong_count", source))
        source = clone_source()
        source[rows_key] = list(reversed(source[rows_key]))
        mutations.append(("wrong_order", source))
        source = clone_source()
        source[rows_key][0] = []
        mutations.append(("non_dict_row", source))
        source = clone_source()
        source[rows_key][0]["unexpected"] = True
        mutations.append(("extra_row_field", source))
        source = clone_source()
        source[rows_key][0].pop(next(iter(source[rows_key][0])))
        mutations.append(("missing_row_field", source))
        source = clone_source()
        source[rows_key][0]["readiness_classification"] = "wrong"
        mutations.append(("changed_classification", source))
        source = clone_source()
        source[rows_key][0]["readiness_result"] = "wrong"
        mutations.append(("changed_result", source))
    for _, source in mutations:
        assert_malformed_blocks_without_mutation(monkeypatch, source)


def test_depth_cycle_and_subclass_targeted_sentinels(monkeypatch: pytest.MonkeyPatch) -> None:
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert_blocked_json_safe(build_with_source(monkeypatch, {"schema_version": cyclic_dict}))
    assert_blocked_json_safe(build_with_source(monkeypatch, {"schema_version": cyclic_list}))
    deep_dict: dict[str, Any] = {}
    node = deep_dict
    for _ in range(1500):
        node["x"] = {}
        node = node["x"]
    deep_list: list[Any] = []
    current = deep_list
    for _ in range(1500):
        child: list[Any] = []
        current.append(child)
        current = child
    for section in [
        "invariant_authorization_readiness_guard",
        "exe_authorization_readiness_guard",
        "source_boundaries",
    ]:
        source = clone_source()
        source[section] = deep_dict
        assert_malformed_blocks_without_mutation(monkeypatch, source, compare_source=False)
    source = clone_source()
    source["schema_version"] = deep_list
    assert_malformed_blocks_without_mutation(monkeypatch, source, compare_source=False)
    shared: dict[str, Any] = {"x": 1}
    assert contract._all_plain_json({"a": shared, "b": shared}) is True
    assert_malformed_blocks_without_mutation(monkeypatch, DictSubclass())
    assert_malformed_blocks_without_mutation(monkeypatch, ListSubclass())


@pytest.mark.parametrize(
    "section",
    [
        "domain_authorization_readiness_rows",
        "requirement_authorization_readiness_rows",
        "invariant_authorization_readiness_guard",
        "exe_authorization_readiness_guard",
        "real_capability_authorization_readiness_state",
        "fail_closed_readiness_decision",
        "source_boundaries",
    ],
)
def test_invalid_sections_are_blocked_and_isolated(
    monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    source = clone_source()
    source[section] = {} if type(source[section]) is list else []
    payload = assert_malformed_blocks_without_mutation(monkeypatch, source)
    summary = payload["readiness_contract_summary"]
    if section == "fail_closed_readiness_decision":
        assert summary["all_contract_readiness_conditions_false"] is True
        assert summary["all_source_readiness_grants_false"] is False
    if section == "source_boundaries":
        assert summary["all_contract_readiness_conditions_false"] is True
    if section in {
        "domain_authorization_readiness_rows",
        "requirement_authorization_readiness_rows",
        "invariant_authorization_readiness_guard",
        "exe_authorization_readiness_guard",
        "real_capability_authorization_readiness_state",
    }:
        assert summary["all_contract_readiness_conditions_false"] is False
