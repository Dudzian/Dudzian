from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_o_execution_authorization_contract import (
    build_preview_block_o_execution_authorization_contract,
)
from ui.pyside_app import preview_block_o_execution_authorization_read_model as read_model
from ui.pyside_app.preview_block_o_execution_authorization_read_model import (
    BLOCKED_STATUS,
    EXPECTED_SOURCE,
    MAX_DIAGNOSTIC_CONTAINER_DEPTH,
    STATUS,
    TOP_LEVEL_FIELDS,
    _all_plain_json,
    build_preview_block_o_execution_authorization_read_model,
)


def assert_read_model_blocked(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_read_model_ready"] is False
    assert payload["ready_for_block_o_5"] is False
    assert payload["execution_authorization_read_model_status"] == BLOCKED_STATUS
    assert payload["execution_authorization_read_model_decision"] == BLOCKED_STATUS.upper()
    assert payload["status"] == BLOCKED_STATUS
    reference = payload["block_o_execution_authorization_contract_reference"]
    assert reference["source_contract_read_by_17_4"] is True
    assert reference["execution_authorization_read_model_built_by_17_4"] is True
    assert reference["execution_authorization_read_model_ready_by_17_4"] is False
    assert reference["ready_for_functional_preview_17_5"] is False
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["source_contract_read"] is True
    assert evidence["execution_authorization_read_model_built"] is True
    assert evidence["source_contract_accepted"] is False
    assert evidence["block_o_remains_open"] is False
    decision = payload["fail_closed_contract_read_decision"]
    assert decision["execution_authorization_read_model_in_17_4"] == "blocked"
    assert decision["execution_authorization_readiness_matrix_in_17_5"] == "blocked"
    assert decision["execution_authorization_granted_by_17_4"] is False
    assert payload["source_boundaries"]["contract_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_5"] is False
    json.dumps(payload)


def patched_payload(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        read_model,
        "build_preview_block_o_execution_authorization_contract",
        lambda: source,
    )
    return build_preview_block_o_execution_authorization_read_model()


def valid_source() -> dict[str, Any]:
    return copy.deepcopy(EXPECTED_SOURCE)


def test_identity_order_reference_and_json_serializable() -> None:
    payload = build_preview_block_o_execution_authorization_read_model()
    assert list(payload.keys()) == TOP_LEVEL_FIELDS
    assert payload["schema_version"] == read_model.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_read_model_kind"] == read_model.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.4"
    assert payload["execution_authorization_read_model_status"] == read_model.READ_MODEL_STATUS
    assert payload["execution_authorization_read_model_decision"] == read_model.READ_MODEL_DECISION
    assert payload["execution_authorization_read_model_ready"] is True
    assert payload["ready_for_block_o_5"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.5"
    assert payload["status"] == STATUS
    reference = payload["block_o_execution_authorization_contract_reference"]
    assert reference["schema_version"] == "preview_block_o_execution_authorization_contract.v1"
    assert reference["step"] == "17.3"
    assert reference["source_contract_read_by_17_4"] is True
    assert reference["execution_authorization_read_model_ready_by_17_4"] is True
    json.dumps(payload)


def test_source_contract_is_consumed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return valid_source()

    monkeypatch.setattr(
        read_model, "build_preview_block_o_execution_authorization_contract", fake_builder
    )
    payload = build_preview_block_o_execution_authorization_read_model()
    assert calls == 1
    assert payload["execution_authorization_read_model_ready"] is True


def test_two_domain_contract_read_rows() -> None:
    rows = build_preview_block_o_execution_authorization_read_model()[
        "domain_authorization_contract_read_rows"
    ]
    assert [row["domain"] for row in rows] == ["packaging_release", "runtime_safety"]
    assert len(rows) == 2
    assert all(row["read_contract_condition_satisfied"] is False for row in rows)
    assert all(row["read_execution_authorized"] is False for row in rows)
    assert rows[0]["read_result"] == "packaging_release_contract_read_execution_unauthorized"
    assert rows[1]["read_result"] == "runtime_safety_contract_read_execution_unauthorized"


def test_seven_requirement_contract_read_rows() -> None:
    rows = build_preview_block_o_execution_authorization_read_model()[
        "requirement_authorization_contract_read_rows"
    ]
    assert len(rows) == 7
    assert all(row["source_required"] is True for row in rows)
    assert all(row["read_requirement_present"] is False for row in rows)
    assert all(row["read_requirement_completed"] is False for row in rows)
    assert all(row["read_requirement_satisfied"] is False for row in rows)
    assert all(row["read_execution_authorized"] is False for row in rows)


def test_invariant_contract_read_state() -> None:
    state = build_preview_block_o_execution_authorization_read_model()[
        "invariant_authorization_contract_read_state"
    ]
    assert state["source_contract_preserved"] is True
    assert state["read_invariants_preserved"] is True
    assert state["read_invariants_alone_satisfy_contract"] is False
    assert state["read_execution_authorized"] is False


def test_exe_contract_read_state_preserves_inherited_lineage() -> None:
    state = build_preview_block_o_execution_authorization_read_model()[
        "exe_authorization_contract_read_state"
    ]
    for key in EXPECTED_SOURCE["exe_authorization_contract"]:
        assert state[key] == EXPECTED_SOURCE["exe_authorization_contract"][key]
    assert state["read_confirms_desktop_exe_direction"] is True
    assert state["read_is_not_build_authorization"] is True
    assert state["read_execution_authorized"] is False


def test_real_capability_contract_read_preserves_exact_blocked_map() -> None:
    state = build_preview_block_o_execution_authorization_read_model()[
        "real_capability_authorization_contract_read_state"
    ]
    status = state["real_capability_status"]
    assert (
        status
        == EXPECTED_SOURCE["real_capability_authorization_contract"]["real_capability_status"]
    )
    assert status["create" + "_order"] == "blocked"
    assert status["fetch" + "_balance"] == "blocked"
    assert status["c" + "cxt"] == "blocked"
    assert state["read_execution_authorized"] is False


def test_fail_closed_contract_read_decision() -> None:
    decision = build_preview_block_o_execution_authorization_read_model()[
        "fail_closed_contract_read_decision"
    ]
    assert decision["missing_source_contract_policy"] == "fail_closed"
    assert decision["block_o_execution_authorization_contract_in_17_3"] == "preserved"
    assert decision["execution_authorization_read_model_in_17_4"] == "ready"
    assert decision["execution_authorization_readiness_matrix_in_17_5"] == "allowed"
    assert decision["execution_authorization_granted_by_17_4"] is False


def test_read_model_summary_evidence_and_boundaries() -> None:
    payload = build_preview_block_o_execution_authorization_read_model()
    summary = payload["read_model_summary"]
    assert summary["source_only"] is True
    assert summary["source_contract_accepted"] is True
    assert summary["all_source_contract_conditions_false"] is True
    assert summary["all_source_execution_authorizations_false"] is True
    assert summary["all_read_execution_authorizations_false"] is True
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["source_contract_read"] is True
    assert evidence["domain_rows_read_valid"] is True
    assert evidence["real_capability_read_valid"] is True
    assert all(payload["read_model_boundaries"].values())


def test_forbidden_raw_tokens_absent_from_17_4_source() -> None:
    source = Path(read_model.__file__).read_text(encoding="utf-8")
    forbidden = ["create" + "_order", "fetch" + "_balance", "c" + "cxt"]
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", "bad"),
        ("step", "17.4"),
        ("execution_authorization_contract_ready", 1),
        ("ready_for_block_o_4", 1),
        ("ready_for_block_o_4", False),
        ("status", "bad"),
    ],
)
def test_identity_sentinels_block(monkeypatch: pytest.MonkeyPatch, key: str, value: Any) -> None:
    source = valid_source()
    source[key] = value
    assert_read_model_blocked(patched_payload(monkeypatch, source))


def test_extra_missing_and_reordered_top_level_block(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["extra"] = True
    assert_read_model_blocked(patched_payload(monkeypatch, source))
    source = valid_source()
    source.pop("status")
    assert_read_model_blocked(patched_payload(monkeypatch, source))
    source = {
        "status": valid_source()["status"],
        **{k: v for k, v in valid_source().items() if k != "status"},
    }
    assert_read_model_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize(
    "section",
    [
        "block_o_execution_authorization_matrix_reference",
        "contract_summary",
        "domain_authorization_contract_rows",
        "requirement_authorization_contract_rows",
        "invariant_authorization_contract",
        "exe_authorization_contract",
        "real_capability_authorization_contract",
        "fail_closed_contract_decision",
        "non_execution_contract_evidence",
        "contract_boundaries",
        "source_boundaries",
        "future_steps",
    ],
)
def test_exact_sections_malformed_block(monkeypatch: pytest.MonkeyPatch, section: str) -> None:
    source = valid_source()
    source[section] = [] if isinstance(source[section], dict) else {}
    assert_read_model_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda s: s["domain_authorization_contract_rows"][0].__setitem__(
            "execution_authorized_by_contract", True
        ),
        lambda s: s["requirement_authorization_contract_rows"][0].__setitem__(
            "contract_condition_satisfied", True
        ),
        lambda s: s["invariant_authorization_contract"].__setitem__(
            "execution_authorized_by_contract", True
        ),
        lambda s: s["exe_authorization_contract"].__setitem__(
            "execution_authorized_by_contract", True
        ),
        lambda s: s["real_capability_authorization_contract"]["real_capability_status"].__setitem__(
            next(iter(s["real_capability_authorization_contract"]["real_capability_status"])),
            "allowed",
        ),
        lambda s: s["fail_closed_contract_decision"].__setitem__(
            "execution_authorization_granted_by_17_3", True
        ),
        lambda s: s["contract_boundaries"].__setitem__("cannot_grant_authorization", False),
    ],
)
def test_authorization_bypass_blocks(monkeypatch: pytest.MonkeyPatch, mutator: Any) -> None:
    source = valid_source()
    mutator(source)
    assert_read_model_blocked(patched_payload(monkeypatch, source))


class LyingValue:
    def __eq__(self, other: object) -> bool:
        return other in {"blocked", "release_execution"}


class BombTopLevelKey:
    def __eq__(self, other: object) -> bool:
        raise AssertionError("equality called")

    def __hash__(self) -> int:
        return 1


class LyingTopLevelKey:
    def __eq__(self, other: object) -> bool:
        return other == "schema_version"

    def __hash__(self) -> int:
        return 8675309


@pytest.mark.parametrize("bad", [LyingValue(), LyingTopLevelKey()])
def test_custom_equality_real_capability_blocks(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    source = valid_source()
    status = source["real_capability_authorization_contract"]["real_capability_status"]
    first_key = next(iter(status))
    if isinstance(bad, LyingTopLevelKey):
        status[bad] = "blocked"
    else:
        status[first_key] = bad
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert (
        payload["real_capability_authorization_contract_read_state"]["real_capability_status"] == {}
    )
    assert payload["fail_closed_contract_read_decision"]["real_capability_status"] == {}


@pytest.mark.parametrize("key", [BombTopLevelKey(), LyingTopLevelKey()])
def test_custom_top_level_keys_block_without_mutation(
    monkeypatch: pytest.MonkeyPatch, key: Any
) -> None:
    source = valid_source()
    before = copy.deepcopy(source)
    source[key] = "preview_block_o_execution_authorization_contract.v1"
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert all(k in source for k in before)
    assert payload["non_execution_contract_read_evidence"]["domain_rows_read_valid"] is True


@pytest.mark.parametrize("field", read_model.INVARIANT_READ_FIELDS_17_4)
def test_invariant_field_shadowing_blocks(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source = valid_source()
    source["invariant_authorization_contract"][field] = True
    assert_read_model_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize("field", read_model.EXE_READ_FIELDS_17_4)
def test_exe_field_shadowing_blocks(monkeypatch: pytest.MonkeyPatch, field: str) -> None:
    source = valid_source()
    source["exe_authorization_contract"][field] = True
    assert_read_model_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize("field", read_model.SOURCE_BOUNDARY_READ_FIELDS_17_4)
def test_source_boundaries_field_shadowing_blocks(
    monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    source = valid_source()
    source["source_boundaries"][field] = True
    assert_read_model_blocked(patched_payload(monkeypatch, source))


def test_cross_section_isolation_invalid_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["invariant_authorization_contract"]["execution_authorized_by_contract"] = True
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["invariant_read_valid"] is False
    assert evidence["exe_read_valid"] is True
    assert evidence["real_capability_read_valid"] is True


def test_cross_section_isolation_invalid_requirement_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["requirement_authorization_contract_rows"][0]["contract_result"] = "bad"
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert payload["non_execution_contract_read_evidence"]["requirement_rows_read_valid"] is False
    assert payload["non_execution_contract_read_evidence"]["domain_rows_read_valid"] is True
    assert payload["domain_authorization_contract_read_rows"][0]["domain"] == "packaging_release"


@pytest.mark.parametrize(
    "bad_source", [object(), lambda: None, set(), tuple(), [], {"x": object()}]
)
def test_malformed_source_blocks(monkeypatch: pytest.MonkeyPatch, bad_source: Any) -> None:
    assert_read_model_blocked(patched_payload(monkeypatch, bad_source))


def test_cycles_and_deep_values_block_without_recursion(monkeypatch: pytest.MonkeyPatch) -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    assert_read_model_blocked(patched_payload(monkeypatch, cyclic))
    deep: dict[str, Any] = {}
    cursor = deep
    for _ in range(1500):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    assert _all_plain_json(deep) is True
    assert_read_model_blocked(patched_payload(monkeypatch, deep))


def test_all_plain_json_direct_helper() -> None:
    deep: dict[str, Any] = {}
    cursor = deep
    for _ in range(100):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    assert _all_plain_json(deep) is True
    assert _all_plain_json(deep, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    assert _all_plain_json(cyclic_dict) is False
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    assert _all_plain_json(cyclic_list) is False
    shared: list[Any] = []
    assert _all_plain_json([shared, shared]) is True


class BombNestedKey:
    called = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        raise RuntimeError("nested custom equality must not run")


class LyingNestedKey:
    called = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return 24681357

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        return other == self.target


def _owned_field_for_section(section: str) -> str:
    if section == "invariant_authorization_contract":
        return read_model.INVARIANT_READ_FIELDS_17_4[0]
    if section == "exe_authorization_contract":
        return read_model.EXE_READ_FIELDS_17_4[0]
    return read_model.SOURCE_BOUNDARY_READ_FIELDS_17_4[0]


def _evidence_key_for_section(section: str) -> str:
    return {
        "invariant_authorization_contract": "invariant_read_valid",
        "exe_authorization_contract": "exe_read_valid",
        "source_boundaries": "source_boundaries_read_valid",
    }[section]


@pytest.mark.parametrize(
    "section",
    ["invariant_authorization_contract", "exe_authorization_contract", "source_boundaries"],
)
@pytest.mark.parametrize("key_type", [BombNestedKey, LyingNestedKey])
def test_nested_custom_keys_do_not_trigger_equality_and_block(
    monkeypatch: pytest.MonkeyPatch, section: str, key_type: type[Any]
) -> None:
    source = valid_source()
    target = _owned_field_for_section(section)
    custom_key = key_type(target)
    key_type.called = False
    source[section][custom_key] = "shadow-attempt"
    existing_keys = list(source[section].keys())
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert json.dumps(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence[_evidence_key_for_section(section)] is False
    for independent in [
        "domain_rows_read_valid",
        "requirement_rows_read_valid",
        "real_capability_read_valid",
    ]:
        if independent != _evidence_key_for_section(section):
            assert evidence[independent] is True
    assert list(source[section].keys()) == existing_keys
    assert key_type.called is False


def _first_scalar_path(value: Any) -> list[Any]:
    if type(value) is dict:
        for key, child in value.items():
            found = _first_scalar_path(child)
            if found:
                return [key, *found]
    if type(value) is list:
        for index, child in enumerate(value):
            found = _first_scalar_path(child)
            if found:
                return [index, *found]
    if type(value) in (str, int, bool):
        return []
    return []


def _set_path(root: Any, path: list[Any], value: Any) -> None:
    cursor = root
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value


def _mutate_section_value(section_value: Any) -> Any:
    mutated = copy.deepcopy(section_value)
    if type(mutated) is dict:
        first_key = next(iter(mutated))
        first_value = mutated[first_key]
        if type(first_value) is bool:
            mutated[first_key] = 1
        elif type(first_value) is int:
            mutated[first_key] = float(first_value)
        elif type(first_value) is str:
            mutated[first_key] = first_value + "_changed"
        elif type(first_value) is list:
            mutated[first_key] = [*first_value, "extra"]
        elif type(first_value) is dict:
            mutated[first_key] = {**first_value, "extra": True}
    elif type(mutated) is list and mutated:
        first_value = mutated[0]
        if type(first_value) is dict:
            first_key = next(iter(first_value))
            first_value[first_key] = "changed"
        else:
            mutated[0] = "changed"
    return mutated


@pytest.mark.parametrize(
    "section",
    [
        "contract_summary",
        "domain_authorization_contract_rows",
        "requirement_authorization_contract_rows",
        "invariant_authorization_contract",
        "exe_authorization_contract",
        "real_capability_authorization_contract",
        "fail_closed_contract_decision",
        "contract_boundaries",
        "source_boundaries",
    ],
)
@pytest.mark.parametrize(
    "mutation",
    ["extra_field", "missing_field", "reordered_field", "changed_scalar", "bool_int", "int_float"],
)
def test_exact_section_targeted_mutations_block(
    monkeypatch: pytest.MonkeyPatch, section: str, mutation: str
) -> None:
    source = valid_source()
    value = source[section]
    if mutation == "extra_field":
        if type(value) is dict:
            value["extra_field"] = True
        else:
            value.append({"extra_field": True})
    elif mutation == "missing_field":
        if type(value) is dict:
            value.pop(next(iter(value)))
        else:
            value.pop()
    elif mutation == "reordered_field":
        if type(value) is dict:
            items = list(value.items())
            source[section] = dict(reversed(items))
        else:
            source[section] = list(reversed(value))
    elif mutation == "changed_scalar":
        source[section] = _mutate_section_value(value)
    elif mutation == "bool_int":
        if type(value) is dict:
            source[section] = {"bool_probe": True, **value}
            source[section]["bool_probe"] = 1
        else:
            value[0][next(k for k, v in value[0].items() if type(v) is bool)] = 1
    elif mutation == "int_float":
        if type(value) is dict:
            source[section] = {"int_probe": 1, **value}
            source[section]["int_probe"] = 1.0
        else:
            row = value[0]
            int_keys = [k for k, v in row.items() if type(v) is int]
            row[int_keys[0] if int_keys else next(iter(row))] = 1.0
    assert_read_model_blocked(patched_payload(monkeypatch, source))


def test_non_vacuous_summary_invalid_domain_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["domain_authorization_contract_rows"][0]["contract_result"] = "bad"
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    summary = payload["read_model_summary"]
    assert summary["all_source_contract_conditions_false"] is False
    assert summary["all_source_execution_authorizations_false"] is False
    assert summary["all_read_execution_authorizations_false"] is False


def test_cross_section_isolation_invalid_exe(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["exe_authorization_contract"]["execution_authorized_by_contract"] = True
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["exe_read_valid"] is False
    assert evidence["invariant_read_valid"] is True
    assert evidence["real_capability_read_valid"] is True
    assert payload["read_model_summary"]["all_read_execution_authorizations_false"] is False


def test_cross_section_isolation_invalid_real_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    status = source["real_capability_authorization_contract"]["real_capability_status"]
    status[next(iter(status))] = "allowed"
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["real_capability_read_valid"] is False
    assert evidence["domain_rows_read_valid"] is True
    assert evidence["requirement_rows_read_valid"] is True
    assert evidence["invariant_read_valid"] is True
    assert evidence["exe_read_valid"] is True
    assert payload["read_model_summary"]["all_read_execution_authorizations_false"] is False


def test_non_json_summary_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["contract_summary"]["non_json"] = object()
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["summary_read_valid"] is False
    assert evidence["domain_rows_read_valid"] is True
    assert evidence["requirement_rows_read_valid"] is True
    assert evidence["invariant_read_valid"] is True
    assert evidence["exe_read_valid"] is True
    assert evidence["real_capability_read_valid"] is True


def test_independent_source_boundaries_block_keeps_authorization_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = valid_source()
    source["source_boundaries"]["cannot_use_network"] = False
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    evidence = payload["non_execution_contract_read_evidence"]
    assert evidence["source_boundaries_read_valid"] is False
    assert evidence["domain_rows_read_valid"] is True
    assert evidence["requirement_rows_read_valid"] is True
    assert evidence["invariant_read_valid"] is True
    assert evidence["exe_read_valid"] is True
    assert evidence["real_capability_read_valid"] is True
    summary = payload["read_model_summary"]
    assert summary["all_source_contract_conditions_false"] is True
    assert summary["all_source_execution_authorizations_false"] is True
    assert summary["all_read_execution_authorizations_false"] is True
    assert list(payload["source_boundaries"].keys()) == [
        "source_block_o_execution_authorization_contract",
        "contract_source_preserved",
        "can_build_execution_authorization_read_model",
        "can_feed_17_5",
    ]


class DictSubclass(dict[str, Any]):
    pass


class ListSubclass(list[Any]):
    pass


def _deep_dict() -> dict[str, Any]:
    root: dict[str, Any] = {}
    cursor = root
    for _ in range(1500):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    return root


def _deep_list() -> list[Any]:
    root: list[Any] = []
    cursor = root
    for _ in range(1500):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    return root


@pytest.mark.parametrize(
    "source_factory",
    [
        lambda: [],
        lambda: DictSubclass(valid_source()),
        lambda: {"schema_version": ListSubclass([])},
        lambda: (lambda data: (data.append(data), data)[1])([]),
        _deep_list,
    ],
)
def test_additional_malformed_sources_block_without_mutation(
    monkeypatch: pytest.MonkeyPatch, source_factory: Any
) -> None:
    source = source_factory()
    before_type = type(source)
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)
    assert type(source) is before_type


@pytest.mark.parametrize(
    ("section", "value"),
    [
        ("invariant_authorization_contract", _deep_dict()),
        ("exe_authorization_contract", _deep_dict()),
        ("schema_version", _deep_dict()),
        ("source_boundaries", _deep_dict()),
    ],
)
def test_deep_nested_sections_block_without_recursion(
    monkeypatch: pytest.MonkeyPatch, section: str, value: Any
) -> None:
    source = valid_source()
    source[section] = value
    payload = patched_payload(monkeypatch, source)
    assert_read_model_blocked(payload)


def test_ast_guard() -> None:
    tree = ast.parse(Path(read_model.__file__).read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    import_from = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == []
    assert [node.module for node in import_from] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_contract",
    ]
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_calls = {node.func.id for node in call_nodes if isinstance(node.func, ast.Name)}
    attribute_calls = {
        node.func.attr for node in call_nodes if isinstance(node.func, ast.Attribute)
    }
    builder_calls = [
        node
        for node in call_nodes
        if isinstance(node.func, ast.Name)
        and node.func.id == "build_preview_block_o_execution_authorization_contract"
    ]
    assert len(builder_calls) == 1
    forbidden_builders = {
        "build_preview_block_o_execution_authorization_matrix",
        "build_preview_block_o_read_model",
        "build_preview_block_o_entry_contract",
        "build_preview_block_n_closure_audit",
    }
    assert name_calls.isdisjoint(forbidden_builders)
    assert name_calls == {
        "_all_plain_json",
        "_contains_owned_field",
        "_contract_reference",
        "_copy_plain",
        "_domain_rows",
        "_exact_plain_matches",
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
        "build_preview_block_o_execution_authorization_contract",
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
    forbidden_calls = {
        "open",
        "read_text",
        "write_text",
        "read_bytes",
        "write_bytes",
        "unlink",
        "mkdir",
        "rmdir",
        "rename",
        "replace",
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
        "validation",
        "validate",
        "confirmation",
        "authorize",
        "authorization_grant",
        "grant_authorization",
        "open_gate",
        "mutate_gate",
        "runtime",
        "submit_order",
        "cancel_order",
        "replace_order",
        "package",
        "pyinstaller",
        "build",
        "release",
        "qml",
        "bridge",
        "gateway",
        "controller",
    }
    assert name_calls.isdisjoint(forbidden_calls)
    assert attribute_calls.isdisjoint(forbidden_calls)


def test_expected_source_matches_current_17_3() -> None:
    assert build_preview_block_o_execution_authorization_contract() == EXPECTED_SOURCE
