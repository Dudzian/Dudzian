from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app import preview_block_o_execution_authorization_readiness_read_model as rm
from ui.pyside_app.preview_block_o_execution_authorization_readiness_contract import (
    build_preview_block_o_execution_authorization_readiness_contract,
)


def _payload_with(source: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    calls = 0

    def fake_builder() -> Any:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        rm, "build_preview_block_o_execution_authorization_readiness_contract", fake_builder
    )
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    assert calls == 1
    json.dumps(payload)
    return payload


def _ids(value: Any) -> list[int]:
    stack = [value]
    out = []
    while stack:
        item = stack.pop()
        if type(item) is dict:
            out.append(id(item))
            stack.extend(item.values())
        elif type(item) is list:
            out.append(id(item))
            stack.extend(item)
    return out


def _blocked(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_readiness_read_model_ready"] is False
    assert payload["ready_for_block_o_8"] is False
    assert payload["status"] == rm.BLOCKED_STATUS
    json.dumps(payload)


def _mutated(key: str, value: Any) -> dict[str, Any]:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source[key] = value
    return source


def test_expected_source_matches_current_17_6() -> None:
    assert rm.EXPECTED_SOURCE == build_preview_block_o_execution_authorization_readiness_contract()


def test_identity_order_reference_and_json_serializable() -> None:
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    assert list(payload) == rm.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == rm.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_readiness_read_model_kind"] == rm.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.7"
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.8"
    assert payload["next_step_title"] == "BLOCK O CLOSURE AUDIT"
    assert (
        payload["execution_authorization_readiness_read_model_status"]
        == rm.READINESS_READ_MODEL_STATUS
    )
    assert (
        payload["execution_authorization_readiness_read_model_decision"]
        == rm.READINESS_READ_MODEL_DECISION
    )
    assert payload["execution_authorization_readiness_read_model_ready"] is True
    ref = payload["block_o_execution_authorization_readiness_contract_reference"]
    assert ref["schema_version"] == rm.EXPECTED_SOURCE["schema_version"]
    assert (
        ref["source_block_o_execution_authorization_readiness_contract_step"]
        == "FUNCTIONAL-PREVIEW-17.6"
    )
    assert ref["source_readiness_contract_read_by_17_7"] is True
    assert ref["ready_for_functional_preview_17_8"] is True
    assert all(
        value is False
        for key, value in ref.items()
        if key.endswith("_by_17_7")
        and key
        not in {
            "source_readiness_contract_read_by_17_7",
            "execution_authorization_readiness_read_model_built_by_17_7",
            "execution_authorization_readiness_read_model_ready_by_17_7",
        }
    )
    json.dumps(payload)


def test_source_builder_called_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload_with(copy.deepcopy(rm.EXPECTED_SOURCE), monkeypatch)
    assert payload["status"] == rm.STATUS


def test_two_domain_contract_read_rows() -> None:
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    rows = payload["domain_authorization_readiness_contract_read_rows"]
    assert len(rows) == 2
    assert [row["domain"] for row in rows] == ["packaging_release", "runtime_safety"]
    assert [row["source_capability_count"] for row in rows] == [22, 18]
    assert [row["read_result"] for row in rows] == [
        "packaging_release_readiness_contract_read_execution_not_ready_unauthorized",
        "runtime_safety_readiness_contract_read_execution_not_ready_unauthorized",
    ]
    for index, row in enumerate(rows):
        source_keys = [
            key
            for key in rm.EXPECTED_SOURCE["domain_authorization_readiness_contract_rows"][index]
            if key not in rm.DOMAIN_READ_FIELDS_17_7
        ]
        assert list(row) == ["read_row_id", *source_keys, *rm.DOMAIN_READ_FIELDS_17_7]
        assert row["readiness_classification"] == "readiness_contract_read_execution_not_ready"
        assert row["failure_policy"] == "fail_closed"
        assert row["read_invariants_ready_for_contract"] is True
        assert row["read_execution_ready"] is False
        assert row["read_execution_authorized"] is False


def test_seven_requirement_contract_read_rows() -> None:
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    rows = payload["requirement_authorization_readiness_contract_read_rows"]
    assert len(rows) == 7
    for index, row in enumerate(rows):
        source_keys = [
            key
            for key in rm.EXPECTED_SOURCE["requirement_authorization_readiness_contract_rows"][
                index
            ]
            if key not in rm.REQUIREMENT_READ_FIELDS_17_7
        ]
        assert list(row) == ["read_row_id", *source_keys, *rm.REQUIREMENT_READ_FIELDS_17_7]
        assert (
            row["readiness_classification"]
            == "missing_requirement_readiness_contract_read_execution_not_ready"
        )
        assert (
            row["read_result"]
            == "missing_requirement_readiness_contract_read_execution_not_ready_unauthorized"
        )
        assert row["read_execution_ready"] is False
        assert row["read_execution_authorized"] is False


def test_invariant_read_state() -> None:
    state = rm.build_preview_block_o_execution_authorization_readiness_read_model()[
        "invariant_authorization_readiness_contract_read_state"
    ]
    for key, value in rm.EXPECTED_SOURCE["invariant_authorization_readiness_contract"].items():
        assert state[key] == value
    assert state["source_readiness_contract_preserved"] is True
    assert state["read_invariants_preserved_for_readiness_contract"] is True
    assert state["read_invariants_alone_satisfy_readiness_contract"] is False
    assert state["read_execution_authorized"] is False


def test_exe_read_state_preserves_lineage() -> None:
    state = rm.build_preview_block_o_execution_authorization_readiness_read_model()[
        "exe_authorization_readiness_contract_read_state"
    ]
    for key, value in rm.EXPECTED_SOURCE["exe_authorization_readiness_contract"].items():
        assert state[key] == value
    assert state["read_confirms_desktop_exe_direction"] is True
    assert state["read_build_ready_by_readiness_contract"] is False
    assert state["read_packaging_ready_by_readiness_contract"] is False
    assert state["read_release_ready_by_readiness_contract"] is False


def test_real_capability_read_state_preserves_blocked_map() -> None:
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    state = payload["real_capability_authorization_readiness_contract_read_state"]
    expected = rm.EXPECTED_SOURCE["real_capability_authorization_readiness_contract"][
        "real_capability_status"
    ]
    assert state["real_capability_status"] == expected
    assert state["real_capability_status"] is not expected
    assert state["real_capability_status_inherited_from_17_6"] is True
    assert state["real_capabilities_opened_by_17_7"] is False
    assert state["all_real_capabilities_read_as_blocked"] is True


def test_fail_closed_read_decision() -> None:
    decision = rm.build_preview_block_o_execution_authorization_readiness_read_model()[
        "fail_closed_readiness_contract_read_decision"
    ]
    assert decision["block_o_execution_authorization_readiness_contract_in_17_6"] == "preserved"
    assert decision["execution_authorization_readiness_read_model_in_17_7"] == "ready"
    assert decision["block_o_closure_audit_in_17_8"] == "allowed"
    assert decision["only_source_only_17_8_handoff_allowed"] is True
    assert decision["execution_readiness_granted_by_17_7"] is False
    assert decision["execution_authorization_granted_by_17_7"] is False


def test_summary_evidence_and_boundaries() -> None:
    payload = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    summary = payload["readiness_read_model_summary"]
    assert summary["all_source_contract_conditions_false"] is True
    assert summary["all_source_readiness_grants_false"] is True
    assert summary["all_read_authorization_grants_false"] is True
    evidence = payload["non_execution_readiness_read_evidence"]
    for key in [
        "reference_read_valid",
        "domain_rows_read_valid",
        "requirement_rows_read_valid",
        "source_boundaries_read_valid",
    ]:
        assert evidence[key] is True
    assert evidence["validation_performed_by_17_7"] is False
    boundaries = payload["readiness_read_model_boundaries"]
    assert boundaries["reads_17_6_only"] is True
    assert boundaries["cannot_grant_authorization"] is True
    assert boundaries["can_feed_only_source_only_17_8_closure_audit"] is True


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    container_ids = _ids(rm.build_preview_block_o_execution_authorization_readiness_read_model())
    assert len(container_ids) == len(set(container_ids))


def test_independent_builder_calls_do_not_share_state() -> None:
    first = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    second = rm.build_preview_block_o_execution_authorization_readiness_read_model()
    assert not (set(_ids(first)) & set(_ids(second)))
    first["readiness_read_model_summary"]["source_only"] = False
    assert second["readiness_read_model_summary"]["source_only"] is True
    assert rm.EXPECTED_SOURCE == build_preview_block_o_execution_authorization_readiness_contract()


def test_forbidden_raw_tokens_absent() -> None:
    source = Path(rm.__file__).read_text(encoding="utf-8")
    for token in ("create" + "_order", "fetch" + "_balance", "c" + "cxt"):
        assert token not in source


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: {**s, "extra": True},
        lambda s: {k: s[k] for k in reversed(list(s))},
        lambda s: {k: v for k, v in s.items() if k != "schema_version"},
    ],
)
def test_top_level_extra_missing_reordered_blocks(
    mutation: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _blocked(_payload_with(mutation(copy.deepcopy(rm.EXPECTED_SOURCE)), monkeypatch))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "changed"),
        ("execution_authorization_readiness_contract_ready", 1),
        ("ready_for_block_o_7", 1.0),
        ("step", "17.5"),
    ],
)
def test_identity_scalar_and_bool_number_bypass_blocks(
    key: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _blocked(_payload_with(_mutated(key, value), monkeypatch))


def test_invalid_requirements_isolated_preserves_domain_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source["requirement_authorization_readiness_contract_rows"] = []
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    rows = payload["domain_authorization_readiness_contract_read_rows"]
    assert [row["source_capability_count"] for row in rows] == [22, 18]
    assert all(row["read_invariants_ready_for_contract"] is True for row in rows)
    assert all(row["readiness_classification"] == "source_invalid" for row in rows)
    summary = payload["readiness_read_model_summary"]
    assert summary["all_source_contract_conditions_false"] is False
    assert summary["all_read_readiness_grants_false"] is False


@pytest.mark.parametrize(
    "section",
    [
        "domain_authorization_readiness_contract_rows",
        "invariant_authorization_readiness_contract",
        "exe_authorization_readiness_contract",
        "real_capability_authorization_readiness_contract",
        "fail_closed_readiness_contract_decision",
        "source_boundaries",
    ],
)
def test_invalid_sections_block_without_destroying_unrelated_local_validity(
    section: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _mutated(section, {})
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    evidence = payload["non_execution_readiness_read_evidence"]
    assert evidence["requirement_rows_read_valid"] is True
    if section == "fail_closed_readiness_contract_decision":
        summary = payload["readiness_read_model_summary"]
        assert summary["all_source_contract_conditions_false"] is True
        assert summary["all_source_readiness_grants_false"] is False
        assert summary["all_read_readiness_grants_false"] is True


class DictSubclass(dict[str, Any]):
    pass


class ListSubclass(list[Any]):
    pass


@pytest.mark.parametrize(
    "bad", [object(), set(), tuple(), lambda: None, DictSubclass(), ListSubclass()]
)
def test_unsupported_values_block_json_safe(bad: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source["readiness_contract_summary"] = bad
    _blocked(_payload_with(source, monkeypatch))


def test_cycle_deep_and_shared_plain_json() -> None:
    cycle: dict[str, Any] = {}
    cycle["self"] = cycle
    assert rm._all_plain_json(cycle) is False
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert rm._all_plain_json(cyclic_list) is False
    deep: dict[str, Any] = {}
    cur = deep
    for _ in range(1500):
        nxt: dict[str, Any] = {}
        cur["x"] = nxt
        cur = nxt
    assert rm._all_plain_json(deep) is True
    assert rm._all_plain_json(deep, max_depth=64) is False
    deep_list: list[Any] = []
    cur_list = deep_list
    for _ in range(1500):
        nxt_list: list[Any] = []
        cur_list.append(nxt_list)
        cur_list = nxt_list
    assert rm._all_plain_json(deep_list) is True
    assert rm._all_plain_json(deep_list, max_depth=rm.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    shared: dict[str, Any] = {"x": []}
    assert rm._all_plain_json({"a": shared["x"], "b": shared["x"]}) is True


def _deep_dict_1500() -> dict[str, Any]:
    result: dict[str, Any] = {}
    current = result
    for _ in range(1500):
        child: dict[str, Any] = {}
        current["child"] = child
        current = child
    return result


def _deep_list_1500() -> list[Any]:
    result: list[Any] = []
    current = result
    for _ in range(1500):
        child: list[Any] = []
        current.append(child)
        current = child
    return result


@pytest.mark.parametrize(
    "section,value,expected_invalid_flag",
    [
        (
            "invariant_authorization_readiness_contract",
            _deep_dict_1500(),
            "invariant_read_valid",
        ),
        ("exe_authorization_readiness_contract", _deep_dict_1500(), "exe_read_valid"),
        ("schema_version", _deep_dict_1500(), None),
        ("source_boundaries", _deep_dict_1500(), "source_boundaries_read_valid"),
    ],
)
def test_builder_level_deep_values_block_json_safe(
    section: str,
    value: Any,
    expected_invalid_flag: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    original = copy.deepcopy(source)
    source[section] = value
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    if expected_invalid_flag is not None:
        assert payload["non_execution_readiness_read_evidence"][expected_invalid_flag] is False
    assert source != original
    source[section] = original[section]
    assert source == original


@pytest.mark.parametrize(
    "section,value,expected_invalid_flag",
    [
        (
            "invariant_authorization_readiness_contract",
            (lambda: (lambda d: (d.__setitem__("self", d), d)[1])({}))(),
            "invariant_read_valid",
        ),
        (
            "domain_authorization_readiness_contract_rows",
            (lambda: (lambda items: (items.append(items), items)[1])([]))(),
            "domain_rows_read_valid",
        ),
    ],
)
def test_builder_level_cyclic_sections_block_json_safe(
    section: str,
    value: Any,
    expected_invalid_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source[section] = value
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    assert payload["non_execution_readiness_read_evidence"][expected_invalid_flag] is False


@pytest.mark.parametrize("field", rm.SOURCE_BOUNDARY_FIELDS_17_7)
def test_every_source_boundary_owned_field_is_shadowing_guarded(field: str) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source["source_boundaries"][field] = "shadowed"
    assert rm._no_shadowing(source) is False


@pytest.mark.parametrize(
    "section,field",
    [
        *[
            ("invariant_authorization_readiness_contract", field)
            for field in rm.INVARIANT_READ_FIELDS_17_7
        ],
        *[("exe_authorization_readiness_contract", field) for field in rm.EXE_READ_FIELDS_17_7],
        *[("source_boundaries", field) for field in rm.SOURCE_BOUNDARY_FIELDS_17_7],
    ],
)
def test_all_owned_fields_shadowing_blocks_json_safe(
    section: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source[section][field] = "shadowed"
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)


class BombKey:
    called = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        raise AssertionError("bomb equality called")


class LyingKey(BombKey):
    __hash__ = BombKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        return True


class BombNestedKey:
    equality_calls = 0

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise AssertionError("nested equality called")


class LyingNestedKey(BombNestedKey):
    __hash__ = BombNestedKey.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return True


@pytest.mark.parametrize("key_cls", [BombKey, LyingKey])
@pytest.mark.parametrize("target", ["schema_version", "readiness_contract_summary"])
def test_custom_top_level_keys_do_not_call_equality(
    key_cls: type[BombKey], target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_cls.called = False
    source: dict[Any, Any] = copy.deepcopy(rm.EXPECTED_SOURCE)
    source[key_cls(target)] = source.pop(target)
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    assert key_cls.called is False


@pytest.mark.parametrize("key_cls", [BombNestedKey, LyingNestedKey])
@pytest.mark.parametrize(
    "section,target,invalid_flag,independent_flag",
    [
        (
            "invariant_authorization_readiness_contract",
            "read_execution_ready",
            "invariant_read_valid",
            "exe_read_valid",
        ),
        (
            "invariant_authorization_readiness_contract",
            "read_execution_authorized",
            "invariant_read_valid",
            "requirement_rows_read_valid",
        ),
        (
            "exe_authorization_readiness_contract",
            "read_execution_ready",
            "exe_read_valid",
            "invariant_read_valid",
        ),
        (
            "exe_authorization_readiness_contract",
            "read_execution_authorized",
            "exe_read_valid",
            "domain_rows_read_valid",
        ),
        *[
            ("source_boundaries", field, "source_boundaries_read_valid", "summary_read_valid")
            for field in rm.SOURCE_BOUNDARY_FIELDS_17_7
        ],
    ],
)
def test_bomb_and_lying_nested_key_collisions_do_not_call_equality(
    key_cls: type[BombNestedKey],
    section: str,
    target: str,
    invalid_flag: str,
    independent_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_cls.equality_calls = 0
    source: dict[Any, Any] = copy.deepcopy(rm.EXPECTED_SOURCE)
    original = copy.deepcopy(source)
    source[section][key_cls(target)] = source[section].pop(target, "shadowed")
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    assert key_cls.equality_calls == 0
    evidence = payload["non_execution_readiness_read_evidence"]
    assert evidence[invalid_flag] is False
    assert evidence[independent_flag] is True
    json.dumps(payload)
    assert rm.EXPECTED_SOURCE == original


def _crafted_combined_shadowing_source(
    section: str, owned_field: str, key_cls: type[BombNestedKey]
) -> dict[str, Any]:
    source: dict[str, Any] = copy.deepcopy(rm.EXPECTED_SOURCE)
    original_section = source[section]
    keys = list(original_section)
    first_key = keys[0]
    removed_key = keys[-1]
    crafted: dict[Any, Any] = {
        key_cls(first_key): original_section[first_key],
    }
    for key, value in original_section.items():
        if key not in {first_key, removed_key}:
            crafted[key] = value
    crafted[owned_field] = False
    source[section] = crafted
    return source


@pytest.mark.parametrize("key_cls", [BombNestedKey, LyingNestedKey])
@pytest.mark.parametrize(
    "section,owned_field,invalid_flag,independent_flag",
    [
        (
            "invariant_authorization_readiness_contract",
            "read_execution_ready",
            "invariant_read_valid",
            "exe_read_valid",
        ),
        (
            "exe_authorization_readiness_contract",
            "read_execution_ready",
            "exe_read_valid",
            "invariant_read_valid",
        ),
        (
            "source_boundaries",
            "can_feed_17_8",
            "source_boundaries_read_valid",
            "summary_read_valid",
        ),
    ],
)
def test_combined_nested_key_and_exact_string_owned_field_blocks_without_equality(
    key_cls: type[BombNestedKey],
    section: str,
    owned_field: str,
    invalid_flag: str,
    independent_flag: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_cls.equality_calls = 0
    source = _crafted_combined_shadowing_source(section, owned_field, key_cls)
    original_section = source[section]
    original_key_ids = [id(key) for key in original_section]
    original_plain_items = {
        key: copy.deepcopy(value) for key, value in original_section.items() if type(key) is str
    }
    payload = _payload_with(source, monkeypatch)
    assert key_cls.equality_calls == 0
    _blocked(payload)
    json.dumps(payload)
    evidence = payload["non_execution_readiness_read_evidence"]
    assert evidence[invalid_flag] is False
    assert evidence[independent_flag] is True
    assert source[section] is original_section
    assert [id(key) for key in source[section]] == original_key_ids
    assert {
        key: value for key, value in source[section].items() if type(key) is str
    } == original_plain_items


@pytest.mark.parametrize("key_cls", [BombNestedKey, LyingNestedKey])
@pytest.mark.parametrize(
    "section,owned_field",
    [
        ("invariant_authorization_readiness_contract", "read_execution_ready"),
        ("exe_authorization_readiness_contract", "read_execution_ready"),
        ("source_boundaries", "can_feed_17_8"),
    ],
)
def test_direct_no_shadowing_combined_nested_collision_has_no_equality_calls(
    key_cls: type[BombNestedKey], section: str, owned_field: str
) -> None:
    key_cls.equality_calls = 0
    source = _crafted_combined_shadowing_source(section, owned_field, key_cls)
    assert rm._no_shadowing(source) is False
    assert key_cls.equality_calls == 0


@pytest.mark.parametrize(
    "section,field",
    [
        ("invariant_authorization_readiness_contract", "read_execution_ready"),
        ("exe_authorization_readiness_contract", "read_execution_authorized"),
        ("source_boundaries", "can_feed_17_8"),
    ],
)
def test_owned_field_shadowing_blocks(
    section: str, field: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    source[section][field] = True
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)


def _first_scalar_key(value: dict[str, Any]) -> str:
    for key, item in value.items():
        if type(item) in (str, int, bool):
            return key
    raise AssertionError("missing scalar key")


def _mutate_dict_section(section: str, mutation: str) -> dict[str, Any]:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    value = copy.deepcopy(source[section])
    if mutation == "extra_field":
        value["extra_field"] = True
    elif mutation == "missing_field":
        value.pop(next(iter(value)))
    elif mutation == "reordered_fields":
        value = {key: value[key] for key in reversed(list(value))}
    elif mutation == "wrong_container_type":
        value = []
    elif mutation == "changed_scalar":
        key = _first_scalar_key(value)
        value[key] = "changed" if type(value[key]) is not str else value[key] + "_changed"
    elif mutation == "bool_to_int":
        key = next(key for key, item in value.items() if type(item) is bool)
        value[key] = 1
    elif mutation == "int_to_float":
        int_key = next((key for key, item in value.items() if type(item) is int), None)
        if int_key is None:
            value["extra_float"] = 1.0
        else:
            value[int_key] = 1.0
    source[section] = value
    return source


@pytest.mark.parametrize(
    "section",
    [
        "readiness_contract_summary",
        "invariant_authorization_readiness_contract",
        "exe_authorization_readiness_contract",
        "real_capability_authorization_readiness_contract",
        "fail_closed_readiness_contract_decision",
        "non_execution_readiness_contract_evidence",
        "readiness_contract_boundaries",
        "source_boundaries",
    ],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "extra_field",
        "missing_field",
        "reordered_fields",
        "wrong_container_type",
        "changed_scalar",
        "bool_to_int",
        "int_to_float",
    ],
)
def test_exact_dict_section_mutation_matrix_blocks_json_safe(
    section: str, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _mutate_dict_section(section, mutation)
    original = copy.deepcopy(source)
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    assert source == original


def _mutate_row_section(section: str, mutation: str) -> dict[str, Any]:
    source = copy.deepcopy(rm.EXPECTED_SOURCE)
    rows = copy.deepcopy(source[section])
    if mutation == "wrong_row_count":
        rows = rows[:-1]
    elif mutation == "wrong_row_order":
        rows = list(reversed(rows))
    elif mutation == "wrong_container_type":
        rows = {}
    elif mutation == "non_dict_row":
        rows[0] = []
    elif mutation == "extra_row_field":
        rows[0]["extra_row_field"] = True
    elif mutation == "missing_row_field":
        rows[0].pop(next(iter(rows[0])))
    elif mutation == "changed_classification":
        rows[0]["readiness_contract_classification"] = "changed"
    elif mutation == "changed_result":
        rows[0]["readiness_contract_result"] = "changed"
    elif mutation == "changed_readiness_flag":
        rows[0]["execution_ready_by_readiness_contract"] = True
    elif mutation == "changed_authorization_flag":
        rows[0]["execution_authorized_by_readiness_contract"] = True
    source[section] = rows
    return source


@pytest.mark.parametrize(
    "section",
    [
        "domain_authorization_readiness_contract_rows",
        "requirement_authorization_readiness_contract_rows",
    ],
)
@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_row_count",
        "wrong_row_order",
        "wrong_container_type",
        "non_dict_row",
        "extra_row_field",
        "missing_row_field",
        "changed_classification",
        "changed_result",
        "changed_readiness_flag",
        "changed_authorization_flag",
    ],
)
def test_exact_row_mutation_matrix_blocks_json_safe(
    section: str, mutation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _mutate_row_section(section, mutation)
    original = copy.deepcopy(source)
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)
    assert source == original


@pytest.mark.parametrize(
    "source",
    [
        DictSubclass(rm.EXPECTED_SOURCE),
        _mutated(
            "readiness_contract_summary",
            DictSubclass(rm.EXPECTED_SOURCE["readiness_contract_summary"]),
        ),
        _mutated(
            "domain_authorization_readiness_contract_rows",
            ListSubclass(rm.EXPECTED_SOURCE["domain_authorization_readiness_contract_rows"]),
        ),
        (
            lambda prepared: (
                prepared["domain_authorization_readiness_contract_rows"][0].__setitem__(
                    "required_requirement_ids",
                    ListSubclass(
                        prepared["domain_authorization_readiness_contract_rows"][0][
                            "required_requirement_ids"
                        ]
                    ),
                ),
                prepared,
            )[1]
        )(copy.deepcopy(rm.EXPECTED_SOURCE)),
    ],
)
def test_real_container_subclasses_block_without_exception(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = _payload_with(source, monkeypatch)
    _blocked(payload)


def test_ast_guard_exact_imports_and_calls() -> None:
    tree = ast.parse(Path(rm.__file__).read_text(encoding="utf-8"))
    assert [node for node in ast.walk(tree) if isinstance(node, ast.Import)] == []
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_readiness_contract",
    ]
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_call_list = [name.id for name in calls if isinstance(name, ast.Name)]
    name_calls = set(name_call_list)
    assert name_calls == {
        "_all_plain_json",
        "_contract_reference",
        "_copy_plain",
        "_domain_rows",
        "_exact_plain_matches",
        "_no_shadowing",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_plain_list_section",
        "_requirement_rows",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "_state_with_fields",
        "all",
        "build_preview_block_o_execution_authorization_readiness_contract",
        "id",
        "len",
        "list",
        "range",
        "reversed",
        "set",
        "type",
        "zip",
    }
    assert (
        name_call_list.count("build_preview_block_o_execution_authorization_readiness_contract")
        == 1
    )
    assert "build_preview_block_o_execution_authorization_readiness_matrix" not in name_calls
    for forbidden in {
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
        "grant",
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
        "exec",
        "eval",
        "compile",
        "input",
        "__import__",
    }:
        assert forbidden not in name_calls
    attribute_calls = {attr.attr for attr in calls if isinstance(attr, ast.Attribute)}
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
    for forbidden in {
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
        "grant",
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
    }:
        assert forbidden not in attribute_calls
