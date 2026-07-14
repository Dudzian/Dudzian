from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app import (
    preview_block_o_execution_authorization_readiness_matrix as readiness_matrix,
)
from ui.pyside_app.preview_block_o_execution_authorization_read_model import (
    build_preview_block_o_execution_authorization_read_model,
)
from ui.pyside_app.preview_block_o_execution_authorization_readiness_matrix import (
    BLOCKED_STATUS,
    EXPECTED_SOURCE,
    MAX_DIAGNOSTIC_CONTAINER_DEPTH,
    STATUS,
    TOP_LEVEL_FIELDS,
    _all_plain_json,
    build_preview_block_o_execution_authorization_readiness_matrix,
)


def assert_readiness_matrix_blocked(payload: dict[str, Any]) -> None:
    assert payload["execution_authorization_readiness_matrix_ready"] is False
    assert payload["ready_for_block_o_6"] is False
    assert payload["status"] == BLOCKED_STATUS
    reference = payload["block_o_execution_authorization_read_model_reference"]
    assert reference["source_read_model_read_by_17_5"] is True
    assert reference["execution_authorization_readiness_matrix_built_by_17_5"] is True
    assert reference["execution_authorization_readiness_matrix_ready_by_17_5"] is False
    assert reference["ready_for_functional_preview_17_6"] is False
    decision = payload["fail_closed_readiness_decision"]
    assert decision["execution_authorization_readiness_matrix_in_17_5"] == "blocked"
    assert decision["execution_authorization_readiness_contract_in_17_6"] == "blocked"
    assert decision["execution_readiness_granted_by_17_5"] is False
    assert decision["execution_authorization_granted_by_17_5"] is False
    assert payload["source_boundaries"]["read_model_source_preserved"] is False
    assert payload["source_boundaries"]["can_feed_17_6"] is False
    json.dumps(payload)


def valid_source() -> dict[str, Any]:
    return copy.deepcopy(EXPECTED_SOURCE)


def patched_payload(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        readiness_matrix,
        "build_preview_block_o_execution_authorization_read_model",
        lambda: source,
    )
    return build_preview_block_o_execution_authorization_readiness_matrix()


def test_identity_order_reference_and_json_serializable() -> None:
    payload = build_preview_block_o_execution_authorization_readiness_matrix()
    assert list(payload.keys()) == TOP_LEVEL_FIELDS
    assert payload["schema_version"] == readiness_matrix.SCHEMA_VERSION
    assert payload["block_o_execution_authorization_readiness_matrix_kind"] == readiness_matrix.KIND
    assert payload["block"] == "O"
    assert payload["step"] == "17.5"
    assert (
        payload["execution_authorization_readiness_matrix_status"]
        == readiness_matrix.READINESS_MATRIX_STATUS
    )
    assert (
        payload["execution_authorization_readiness_matrix_decision"]
        == readiness_matrix.READINESS_MATRIX_DECISION
    )
    assert payload["execution_authorization_readiness_matrix_ready"] is True
    assert payload["ready_for_block_o_6"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.6"
    assert payload["next_step_title"] == "BLOCK O EXECUTION AUTHORIZATION READINESS CONTRACT"
    assert payload["status"] == STATUS
    reference = payload["block_o_execution_authorization_read_model_reference"]
    assert reference["schema_version"] == "preview_block_o_execution_authorization_read_model.v1"
    assert reference["step"] == "17.4"
    assert reference["source_read_model_read_by_17_5"] is True
    assert reference["execution_authorization_readiness_matrix_ready_by_17_5"] is True
    json.dumps(payload)


def test_source_read_model_consumed_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_builder() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return valid_source()

    monkeypatch.setattr(
        readiness_matrix, "build_preview_block_o_execution_authorization_read_model", fake_builder
    )
    payload = build_preview_block_o_execution_authorization_readiness_matrix()
    assert calls == 1
    assert payload["execution_authorization_readiness_matrix_ready"] is True


def test_two_domain_authorization_readiness_rows() -> None:
    rows = build_preview_block_o_execution_authorization_readiness_matrix()[
        "domain_authorization_readiness_rows"
    ]
    assert [row["domain"] for row in rows] == ["packaging_release", "runtime_safety"]
    assert len(rows) == 2
    for row in rows:
        assert row["source_requirements_complete"] is False
        assert row["source_read_contract_condition_satisfied"] is False
        assert row["source_read_execution_authorized"] is False
        assert row["domain_readiness_condition_met"] is False
        assert row["domain_ready_for_execution"] is False
        assert row["execution_authorized_by_readiness_matrix"] is False
        assert row["readiness_classification"] == "missing_requirements_execution_not_ready"
    assert (
        rows[0]["readiness_result"]
        == "packaging_release_readiness_missing_requirements_execution_unauthorized"
    )
    assert (
        rows[1]["readiness_result"]
        == "runtime_safety_readiness_missing_requirements_execution_unauthorized"
    )


def test_seven_requirement_authorization_readiness_rows() -> None:
    rows = build_preview_block_o_execution_authorization_readiness_matrix()[
        "requirement_authorization_readiness_rows"
    ]
    assert len(rows) == 7
    for row in rows:
        assert row["source_required"] is True
        assert row["source_present"] is False
        assert row["source_missing"] is True
        assert row["source_read_requirement_satisfied"] is False
        assert row["requirement_readiness_condition_met"] is False
        assert row["requirement_ready_for_execution"] is False
        assert row["execution_authorized_by_readiness_matrix"] is False
        assert row["readiness_classification"] == "missing_requirement_execution_not_ready"
        assert row["readiness_result"] == "missing_requirement_readiness_execution_unauthorized"


def test_invariant_authorization_readiness_guard() -> None:
    guard = build_preview_block_o_execution_authorization_readiness_matrix()[
        "invariant_authorization_readiness_guard"
    ]
    assert guard["read_invariants_preserved"] is True
    assert guard["source_read_state_preserved"] is True
    assert guard["invariants_preserved_for_readiness"] is True
    assert guard["invariants_alone_make_execution_ready"] is False
    assert guard["readiness_condition_met"] is False
    assert guard["execution_authorized_by_readiness_matrix"] is False


def test_exe_authorization_readiness_guard_preserves_lineage() -> None:
    guard = build_preview_block_o_execution_authorization_readiness_matrix()[
        "exe_authorization_readiness_guard"
    ]
    for key, value in EXPECTED_SOURCE["exe_authorization_contract_read_state"].items():
        assert guard[key] == value
    assert guard["desktop_exe_direction_preserved_for_readiness"] is True
    assert guard["build_ready_for_execution"] is False
    assert guard["packaging_ready_for_execution"] is False
    assert guard["release_ready_for_execution"] is False
    assert guard["execution_authorized_by_readiness_matrix"] is False


def test_real_capability_readiness_preserves_exact_blocked_map() -> None:
    state = build_preview_block_o_execution_authorization_readiness_matrix()[
        "real_capability_authorization_readiness_state"
    ]
    status = state["real_capability_status"]
    expected = EXPECTED_SOURCE["real_capability_authorization_contract_read_state"][
        "real_capability_status"
    ]
    assert status == expected
    assert list(status.keys()) == list(expected.keys())
    assert status["create" + "_order"] == "blocked"
    assert status["fetch" + "_balance"] == "blocked"
    assert status["c" + "cxt"] == "blocked"
    assert state["execution_authorized_by_readiness_matrix"] is False


def test_fail_closed_readiness_decision() -> None:
    decision = build_preview_block_o_execution_authorization_readiness_matrix()[
        "fail_closed_readiness_decision"
    ]
    assert all(value == "fail_closed" for key, value in decision.items() if key.endswith("policy"))
    assert decision["block_o_execution_authorization_read_model_in_17_4"] == "preserved"
    assert decision["execution_authorization_readiness_matrix_in_17_5"] == "ready"
    assert decision["execution_authorization_readiness_contract_in_17_6"] == "allowed"
    assert decision["execution_readiness_granted_by_17_5"] is False
    assert decision["execution_authorization_granted_by_17_5"] is False


def test_readiness_summary_evidence_and_boundaries() -> None:
    payload = build_preview_block_o_execution_authorization_readiness_matrix()
    summary = payload["readiness_matrix_summary"]
    assert summary["source_read_model_accepted"] is True
    assert summary["all_source_contract_conditions_false"] is True
    assert summary["all_source_read_authorizations_false"] is True
    assert summary["all_readiness_conditions_false"] is True
    assert summary["all_execution_authorizations_false"] is True
    evidence = payload["non_execution_readiness_evidence"]
    assert evidence["source_read_model_read"] is True
    assert evidence["execution_authorization_readiness_matrix_built"] is True
    assert evidence["source_read_model_accepted"] is True
    assert all(value is True for value in payload["readiness_matrix_boundaries"].values())


def test_forbidden_raw_tokens_absent_from_17_5_source() -> None:
    source = Path(readiness_matrix.__file__).read_text(encoding="utf-8")
    assert "create_order" not in source
    assert "fetch_balance" not in source
    assert "ccxt" not in source


def test_expected_source_matches_current_17_4() -> None:
    assert build_preview_block_o_execution_authorization_read_model() == EXPECTED_SOURCE


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", "wrong"),
        ("block", "P"),
        ("step", "17.5"),
        ("execution_authorization_read_model_ready", 1),
        ("ready_for_block_o_5", 1),
        ("status", "wrong"),
    ],
)
def test_identity_sentinels_block(monkeypatch: pytest.MonkeyPatch, key: str, value: Any) -> None:
    source = valid_source()
    source[key] = value
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


def test_extra_missing_and_reordered_top_level_block(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["extra"] = True
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))
    source = valid_source()
    del source["status"]
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))
    source = valid_source()
    value = source.pop("schema_version")
    source["schema_version"] = value
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutate"),
    [
        ("read_model_summary", lambda s: s.update({"x": True})),
        (
            "domain_authorization_contract_read_rows",
            lambda s: s[0].update({"read_execution_authorized": True}),
        ),
        (
            "requirement_authorization_contract_read_rows",
            lambda s: s[0].update({"read_requirement_satisfied": True}),
        ),
        (
            "invariant_authorization_contract_read_state",
            lambda s: s.update({"read_execution_authorized": True}),
        ),
        (
            "exe_authorization_contract_read_state",
            lambda s: s.update({"read_execution_authorized": True}),
        ),
        (
            "real_capability_authorization_contract_read_state",
            lambda s: s["real_capability_status"].update(
                {next(iter(s["real_capability_status"])): "allowed"}
            ),
        ),
        (
            "fail_closed_contract_read_decision",
            lambda s: s.update({"execution_authorization_granted_by_17_4": True}),
        ),
        ("read_model_boundaries", lambda s: s.update({"cannot_grant_authorization": False})),
    ],
)
def test_exact_section_and_authorization_sentinels_block(
    monkeypatch: pytest.MonkeyPatch, section: str, mutate: Any
) -> None:
    source = valid_source()
    mutate(source[section])
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)


def test_invalid_requirements_preserve_domain_source_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    source["requirement_authorization_contract_read_rows"][0]["read_requirement_satisfied"] = True
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    rows = payload["domain_authorization_readiness_rows"]
    assert rows[0]["source_capability_count"] == 22
    assert rows[1]["source_capability_count"] == 18
    assert all(row["readiness_classification"] == "source_invalid" for row in rows)
    assert all(row["domain_ready_for_execution"] is False for row in rows)


@pytest.mark.parametrize(
    "section",
    [
        "invariant_authorization_contract_read_state",
        "exe_authorization_contract_read_state",
        "real_capability_authorization_contract_read_state",
        "source_boundaries",
    ],
)
def test_cross_section_isolation(monkeypatch: pytest.MonkeyPatch, section: str) -> None:
    source = valid_source()
    source[section]["unexpected"] = True
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    evidence = payload["non_execution_readiness_evidence"]
    if section != "invariant_authorization_contract_read_state":
        assert evidence["invariant_read_valid"] is True
    if section != "exe_authorization_contract_read_state":
        assert evidence["exe_read_valid"] is True
    if section != "real_capability_authorization_contract_read_state":
        assert evidence["real_capability_read_valid"] is True


def test_invalid_source_boundaries_keeps_readiness_aggregates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = valid_source()
    source["source_boundaries"]["extra"] = True
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert payload["source_boundaries"] == {
        "source_block_o_execution_authorization_read_model": "FUNCTIONAL-PREVIEW-17.4",
        "read_model_source_preserved": False,
        "can_build_execution_authorization_readiness_matrix": False,
        "can_feed_17_6": False,
    }
    assert payload["readiness_matrix_summary"]["all_readiness_conditions_false"] is True


class LyingStr(str):
    called = False

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        return True

    def __hash__(self) -> int:
        return str.__hash__(self)


@pytest.mark.parametrize("replacement", [object(), lambda: None, {"x"}, ("x",)])
def test_malformed_values_block_json_safe(
    monkeypatch: pytest.MonkeyPatch, replacement: Any
) -> None:
    source = valid_source()
    before = copy.deepcopy(source)
    source["read_model_summary"] = replacement
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert source != before or replacement is source.get("read_model_summary")


def test_cycles_depth_shared_and_subclasses(monkeypatch: pytest.MonkeyPatch) -> None:
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    for replacement in [cyclic, [cyclic], {"a": {"b": []}}, dict(valid_source()), list([])]:
        source = valid_source()
        source["read_model_summary"] = replacement
        payload = patched_payload(monkeypatch, source)
        if replacement == {"a": {"b": []}}:
            assert payload["non_execution_readiness_evidence"]["summary_read_valid"] is False
        else:
            assert_readiness_matrix_blocked(payload)
    deep: Any = {}
    cursor = deep
    for index in range(MAX_DIAGNOSTIC_CONTAINER_DEPTH + 2):
        cursor["x"] = {}
        cursor = cursor["x"]
    source = valid_source()
    source["source_boundaries"] = deep
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))
    shared: list[Any] = []
    assert _all_plain_json([shared, shared]) is True


def test_custom_keys_do_not_call_equality_and_block(monkeypatch: pytest.MonkeyPatch) -> None:
    source: dict[Any, Any] = valid_source()
    source[LyingStr("schema_version")] = source.pop("schema_version")
    LyingStr.called = False
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert LyingStr.called is False


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("invariant_authorization_contract_read_state", field)
        for field in readiness_matrix.INVARIANT_READINESS_FIELDS_17_5
    ]
    + [
        ("exe_authorization_contract_read_state", field)
        for field in readiness_matrix.EXE_READINESS_FIELDS_17_5
    ]
    + [
        ("source_boundaries", field)
        for field in readiness_matrix.SOURCE_BOUNDARY_READINESS_FIELDS_17_5
    ],
)
def test_field_shadowing_blocks(monkeypatch: pytest.MonkeyPatch, section: str, field: str) -> None:
    source = valid_source()
    source[section][field] = False
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


def test_custom_blocked_value_blocks_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    source = valid_source()
    first_key = next(
        iter(source["real_capability_authorization_contract_read_state"]["real_capability_status"])
    )
    source["real_capability_authorization_contract_read_state"]["real_capability_status"][
        first_key
    ] = LyingStr("blocked")
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert payload["real_capability_authorization_readiness_state"]["real_capability_status"] == {}
    assert payload["fail_closed_readiness_decision"]["real_capability_status"] == {}


class BombTopLevelKey:
    called = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        raise RuntimeError("top-level custom equality must not run")


class LyingTopLevelKey(BombTopLevelKey):
    def __hash__(self) -> int:
        return BombTopLevelKey.__hash__(self)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        return True


class BombNestedKey:
    called = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        raise RuntimeError("nested custom equality must not run")


class LyingNestedKey(BombNestedKey):
    def __hash__(self) -> int:
        return BombNestedKey.__hash__(self)

    def __eq__(self, other: object) -> bool:
        type(self).called = True
        return True


class DictSubclass(dict[str, Any]):
    pass


class ListSubclass(list[Any]):
    pass


def test_invalid_requirements_preserve_invariant_readiness_fact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = valid_source()
    source["requirement_authorization_contract_read_rows"][0]["read_requirement_satisfied"] = True
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    evidence = payload["non_execution_readiness_evidence"]
    assert evidence["requirement_rows_read_valid"] is False
    assert evidence["invariant_read_valid"] is True
    assert evidence["domain_rows_read_valid"] is True
    rows = payload["domain_authorization_readiness_rows"]
    assert [row["source_capability_count"] for row in rows] == [22, 18]
    for row in rows:
        assert row["invariants_ready_for_execution"] is True
        assert row["requirements_ready_for_execution"] is False
        assert row["future_explicit_gate_ready"] is False
        assert row["domain_readiness_condition_met"] is False
        assert row["domain_ready_for_execution"] is False
        assert row["execution_authorized_by_readiness_matrix"] is False
        assert row["readiness_classification"] == "source_invalid"


def _assert_source_equal_without_custom_key(
    current: dict[Any, Any], expected: dict[str, Any], removed: str | None = None
) -> None:
    plain_current = {key: value for key, value in current.items() if type(key) is str}
    plain_expected = copy.deepcopy(expected)
    if removed is not None:
        plain_expected.pop(removed)
    assert plain_current == plain_expected


@pytest.mark.parametrize(
    ("key_class", "replaced_key", "expected_valid_flags"),
    [
        (BombTopLevelKey, "schema_version", {"summary_read_valid": True}),
        (BombTopLevelKey, "read_model_summary", {"domain_rows_read_valid": True}),
        (LyingTopLevelKey, "schema_version", {"summary_read_valid": True}),
    ],
)
def test_bomb_and_lying_top_level_keys_are_projected_without_equality(
    monkeypatch: pytest.MonkeyPatch,
    key_class: type[BombTopLevelKey],
    replaced_key: str,
    expected_valid_flags: dict[str, bool],
) -> None:
    source: dict[Any, Any] = valid_source()
    before = copy.deepcopy(source)
    key_class.called = False
    source[key_class(replaced_key)] = source.pop(replaced_key)
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert key_class.called is False
    _assert_source_equal_without_custom_key(source, before, removed=replaced_key)
    evidence = payload["non_execution_readiness_evidence"]
    for key, value in expected_valid_flags.items():
        assert evidence[key] is value


@pytest.mark.parametrize(
    ("key_class", "section", "owned_field", "invalid_flag", "valid_flags"),
    [
        (
            BombNestedKey,
            "invariant_authorization_contract_read_state",
            readiness_matrix.INVARIANT_READINESS_FIELDS_17_5[0],
            "invariant_read_valid",
            ["exe_read_valid", "real_capability_read_valid", "source_boundaries_read_valid"],
        ),
        (
            LyingNestedKey,
            "invariant_authorization_contract_read_state",
            readiness_matrix.INVARIANT_READINESS_FIELDS_17_5[1],
            "invariant_read_valid",
            ["exe_read_valid", "real_capability_read_valid", "source_boundaries_read_valid"],
        ),
        (
            BombNestedKey,
            "exe_authorization_contract_read_state",
            readiness_matrix.EXE_READINESS_FIELDS_17_5[0],
            "exe_read_valid",
            ["invariant_read_valid", "real_capability_read_valid", "source_boundaries_read_valid"],
        ),
        (
            LyingNestedKey,
            "exe_authorization_contract_read_state",
            readiness_matrix.EXE_READINESS_FIELDS_17_5[1],
            "exe_read_valid",
            ["invariant_read_valid", "real_capability_read_valid", "source_boundaries_read_valid"],
        ),
        (
            BombNestedKey,
            "source_boundaries",
            readiness_matrix.SOURCE_BOUNDARY_READINESS_FIELDS_17_5[0],
            "source_boundaries_read_valid",
            ["invariant_read_valid", "exe_read_valid", "real_capability_read_valid"],
        ),
        (
            LyingNestedKey,
            "source_boundaries",
            readiness_matrix.SOURCE_BOUNDARY_READINESS_FIELDS_17_5[1],
            "source_boundaries_read_valid",
            ["invariant_read_valid", "exe_read_valid", "real_capability_read_valid"],
        ),
    ],
)
def test_bomb_and_lying_nested_keys_do_not_call_equality(
    monkeypatch: pytest.MonkeyPatch,
    key_class: type[BombNestedKey],
    section: str,
    owned_field: str,
    invalid_flag: str,
    valid_flags: list[str],
) -> None:
    source = valid_source()
    before = copy.deepcopy(source)
    key_class.called = False
    source[section][key_class(owned_field)] = False
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    assert key_class.called is False
    source[section].pop(next(key for key in source[section] if type(key) is key_class))
    assert source == before
    evidence = payload["non_execution_readiness_evidence"]
    assert evidence[invalid_flag] is False
    for flag in valid_flags:
        assert evidence[flag] is True


def test_real_container_subclasses_block_json_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    dict_subclass_source = DictSubclass(valid_source())
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, dict_subclass_source))

    source = valid_source()
    source["read_model_summary"] = DictSubclass(source["read_model_summary"])
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))

    source = valid_source()
    source["domain_authorization_contract_read_rows"] = ListSubclass(
        source["domain_authorization_contract_read_rows"]
    )
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))

    source = valid_source()
    source["domain_authorization_contract_read_rows"][0]["required_requirement_ids"] = ListSubclass(
        source["domain_authorization_contract_read_rows"][0]["required_requirement_ids"]
    )
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


def _deep_dict(levels: int) -> dict[str, Any]:
    root: dict[str, Any] = {}
    cursor = root
    for _ in range(levels):
        child: dict[str, Any] = {}
        cursor["x"] = child
        cursor = child
    return root


def _deep_list(levels: int) -> list[Any]:
    root: list[Any] = []
    cursor = root
    for _ in range(levels):
        child: list[Any] = []
        cursor.append(child)
        cursor = child
    return root


def test_direct_deep_cycle_and_shared_plain_json() -> None:
    deep_dict = _deep_dict(1500)
    deep_list = _deep_list(1500)
    assert _all_plain_json(deep_dict) is True
    assert _all_plain_json(deep_list) is True
    assert _all_plain_json(deep_dict, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    assert _all_plain_json(deep_list, max_depth=MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    cyclic_dict: dict[str, Any] = {}
    cyclic_dict["self"] = cyclic_dict
    cyclic_list: list[Any] = []
    cyclic_list.append(cyclic_list)
    shared: list[Any] = []
    shared_acyclic = [shared, shared]
    assert _all_plain_json(cyclic_dict) is False
    assert _all_plain_json(cyclic_list) is False
    assert _all_plain_json(shared_acyclic) is True


@pytest.mark.parametrize(
    ("section", "replacement", "invalid_flag"),
    [
        ("invariant_authorization_contract_read_state", _deep_dict(80), "invariant_read_valid"),
        ("exe_authorization_contract_read_state", _deep_dict(80), "exe_read_valid"),
        ("schema_version", _deep_dict(80), ""),
        ("source_boundaries", _deep_dict(80), "source_boundaries_read_valid"),
    ],
)
def test_builder_deep_sections_and_scalar_reference_block_json_safe(
    monkeypatch: pytest.MonkeyPatch, section: str, replacement: Any, invalid_flag: str
) -> None:
    source = valid_source()
    source[section] = replacement
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    evidence = payload["non_execution_readiness_evidence"]
    if invalid_flag in evidence:
        assert evidence[invalid_flag] is False


def _first_key(mapping: dict[str, Any]) -> str:
    return next(iter(mapping))


def _mutate_extra_field(source: dict[str, Any], section: str) -> None:
    value = source[section]
    if type(value) is dict:
        value["extra_field"] = True
    else:
        value[0]["extra_field"] = True


def _mutate_missing_field(source: dict[str, Any], section: str) -> None:
    value = source[section]
    if type(value) is dict:
        value.pop(_first_key(value))
    else:
        value[0].pop(_first_key(value[0]))


def _mutate_reordered_field(source: dict[str, Any], section: str) -> None:
    value = source[section]
    target = value if type(value) is dict else value[0]
    key = _first_key(target)
    item = target.pop(key)
    target[key] = item


def _mutate_changed_scalar(source: dict[str, Any], section: str) -> None:
    value = source[section]
    target = value if type(value) is dict else value[0]
    for key, item in target.items():
        if type(item) is str:
            target[key] = item + "_changed"
            return
        if type(item) is bool:
            target[key] = not item
            return
        if type(item) is int:
            target[key] = item + 1
            return


def _mutate_bool_to_int(source: dict[str, Any], section: str) -> None:
    value = source[section]
    target = value if type(value) is dict else value[0]
    for key, item in target.items():
        if type(item) is bool:
            target[key] = 1 if item else 0
            return


def _mutate_int_to_float(source: dict[str, Any], section: str) -> None:
    value = source[section]
    target = value if type(value) is dict else value[0]
    for key, item in target.items():
        if type(item) is int and type(item) is not bool:
            target[key] = float(item)
            return
    target["unexpected_float"] = 1.0


SECTION_MUTATORS = [
    _mutate_extra_field,
    _mutate_missing_field,
    _mutate_reordered_field,
    _mutate_changed_scalar,
    _mutate_bool_to_int,
    _mutate_int_to_float,
]


@pytest.mark.parametrize(
    "section",
    [
        "read_model_summary",
        "domain_authorization_contract_read_rows",
        "requirement_authorization_contract_read_rows",
        "invariant_authorization_contract_read_state",
        "exe_authorization_contract_read_state",
        "real_capability_authorization_contract_read_state",
        "fail_closed_contract_read_decision",
        "read_model_boundaries",
        "source_boundaries",
    ],
)
@pytest.mark.parametrize("mutator", SECTION_MUTATORS)
def test_exact_section_mutation_matrix_blocks(
    monkeypatch: pytest.MonkeyPatch, section: str, mutator: Any
) -> None:
    source = valid_source()
    mutator(source, section)
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutator"),
    [
        ("domain_authorization_contract_read_rows", lambda rows: rows.pop()),
        ("domain_authorization_contract_read_rows", lambda rows: rows.reverse()),
        ("domain_authorization_contract_read_rows", lambda rows: rows.__setitem__(0, "bad_row")),
        (
            "domain_authorization_contract_read_rows",
            lambda rows: rows[0].update({"extra_row_field": True}),
        ),
        ("domain_authorization_contract_read_rows", lambda rows: rows[0].pop("domain")),
        ("requirement_authorization_contract_read_rows", lambda rows: rows.pop()),
        ("requirement_authorization_contract_read_rows", lambda rows: rows.reverse()),
        (
            "requirement_authorization_contract_read_rows",
            lambda rows: rows.__setitem__(0, "bad_row"),
        ),
        (
            "requirement_authorization_contract_read_rows",
            lambda rows: rows[0].update({"extra_row_field": True}),
        ),
        (
            "requirement_authorization_contract_read_rows",
            lambda rows: rows[0].pop("requirement_id"),
        ),
    ],
)
def test_row_shape_mutation_matrix_blocks(
    monkeypatch: pytest.MonkeyPatch, section: str, mutator: Any
) -> None:
    source = valid_source()
    mutator(source[section])
    assert_readiness_matrix_blocked(patched_payload(monkeypatch, source))


@pytest.mark.parametrize(
    ("section", "mutate", "false_claims", "true_claims"),
    [
        (
            "domain_authorization_contract_read_rows",
            lambda source: source["domain_authorization_contract_read_rows"][0].update(
                {"domain": "bad"}
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_domains_not_ready",
                "all_execution_authorizations_false",
            ],
            ["seven_requirement_rows_read", "invariants_preserved", "desktop_exe_preserved"],
        ),
        (
            "requirement_authorization_contract_read_rows",
            lambda source: source["requirement_authorization_contract_read_rows"][0].update(
                {"read_requirement_satisfied": True}
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_requirements_not_ready",
                "all_execution_authorizations_false",
            ],
            ["two_domain_rows_read", "invariants_preserved", "desktop_exe_preserved"],
        ),
        (
            "invariant_authorization_contract_read_state",
            lambda source: source["invariant_authorization_contract_read_state"].update(
                {"read_execution_authorized": True}
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_execution_authorizations_false",
            ],
            ["two_domain_rows_read", "seven_requirement_rows_read", "desktop_exe_preserved"],
        ),
        (
            "exe_authorization_contract_read_state",
            lambda source: source["exe_authorization_contract_read_state"].update(
                {"read_execution_authorized": True}
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_execution_authorizations_false",
            ],
            ["two_domain_rows_read", "seven_requirement_rows_read", "invariants_preserved"],
        ),
        (
            "real_capability_authorization_contract_read_state",
            lambda source: source["real_capability_authorization_contract_read_state"][
                "real_capability_status"
            ].update(
                {
                    next(
                        iter(
                            source["real_capability_authorization_contract_read_state"][
                                "real_capability_status"
                            ]
                        )
                    ): "allowed"
                }
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_execution_authorizations_false",
                "real_capabilities_blocked",
            ],
            ["two_domain_rows_read", "seven_requirement_rows_read", "invariants_preserved"],
        ),
        (
            "fail_closed_contract_read_decision",
            lambda source: source["fail_closed_contract_read_decision"].update(
                {"execution_authorization_granted_by_17_4": True}
            ),
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_execution_authorizations_false",
            ],
            [
                "all_readiness_conditions_false",
                "two_domain_rows_read",
                "seven_requirement_rows_read",
            ],
        ),
        (
            "source_boundaries",
            lambda source: source["source_boundaries"].update({"extra": True}),
            [],
            [
                "all_source_contract_conditions_false",
                "all_source_read_authorizations_false",
                "all_readiness_conditions_false",
                "all_execution_authorizations_false",
                "two_domain_rows_read",
                "seven_requirement_rows_read",
                "invariants_preserved",
                "desktop_exe_preserved",
            ],
        ),
    ],
)
def test_non_vacuous_aggregate_claims_by_invalid_section(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    mutate: Any,
    false_claims: list[str],
    true_claims: list[str],
) -> None:
    source = valid_source()
    mutate(source)
    payload = patched_payload(monkeypatch, source)
    assert_readiness_matrix_blocked(payload)
    summary = payload["readiness_matrix_summary"]
    for claim in false_claims:
        assert summary[claim] is False, section
    for claim in true_claims:
        assert summary[claim] is True, section
    if section == "source_boundaries":
        assert payload["source_boundaries"]["read_model_source_preserved"] is False
        assert payload["non_execution_readiness_evidence"]["source_boundaries_read_valid"] is False


def test_ast_imports_builder_call_and_call_allowlist() -> None:
    source = Path(readiness_matrix.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_execution_authorization_read_model",
    ]
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    name_calls = {node.func.id for node in call_nodes if isinstance(node.func, ast.Name)}
    attribute_calls = {
        node.func.attr for node in call_nodes if isinstance(node.func, ast.Attribute)
    }
    assert name_calls == {
        "_all_plain_json",
        "_contains_owned_field",
        "_copy_plain",
        "_domain_rows",
        "_exact_plain_matches",
        "_no_shadowing",
        "_plain_dict_section",
        "_plain_list_section",
        "_read_model_reference",
        "_requirement_rows",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "_state_with_fields",
        "all",
        "build_preview_block_o_execution_authorization_read_model",
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
    builder_calls = [
        node
        for node in call_nodes
        if isinstance(node.func, ast.Name)
        and node.func.id == "build_preview_block_o_execution_authorization_read_model"
    ]
    assert len(builder_calls) == 1
    forbidden_names = {
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
        "confirm",
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
    forbidden_builder_fragments = (
        "17_3",
        "17_2",
        "17_1",
        "17_0",
        "16_",
    )
    for node in call_nodes:
        if isinstance(node.func, ast.Name):
            assert node.func.id not in forbidden_names
            assert not (
                node.func.id.startswith("build_preview")
                and any(fragment in node.func.id for fragment in forbidden_builder_fragments)
            )
        if isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_names
