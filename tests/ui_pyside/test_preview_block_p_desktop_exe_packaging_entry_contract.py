from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
from typing import Any

import pytest

import ui.pyside_app.preview_block_p_desktop_exe_packaging_entry_contract as block_p
from ui.pyside_app.preview_block_o_closure_audit import build_preview_block_o_closure_audit


SENTINEL = object()


def _payload(source: Any = SENTINEL) -> dict[str, Any]:
    if source is SENTINEL:
        return block_p.build_preview_block_p_desktop_exe_packaging_entry_contract()
    old = block_p.build_preview_block_o_closure_audit
    calls = {"count": 0}

    def fake() -> Any:
        calls["count"] += 1
        return source

    block_p.build_preview_block_o_closure_audit = fake
    try:
        payload = block_p.build_preview_block_p_desktop_exe_packaging_entry_contract()
    finally:
        block_p.build_preview_block_o_closure_audit = old
    assert calls["count"] == 1
    return payload


def _assert_blocked(payload: dict[str, Any]) -> None:
    assert payload["block_p_opened"] is False
    assert payload["ready_for_block_p_1"] is False
    assert payload["status"] == block_p.BLOCKED_STATUS
    assert block_p._all_plain_json(payload)
    json.dumps(payload)


def _container_ids(value: Any) -> list[int]:
    found: list[int] = []
    stack = [value]
    while stack:
        item = stack.pop()
        if type(item) is dict:
            found.append(id(item))
            stack.extend(item.values())
        elif type(item) is list:
            found.append(id(item))
            stack.extend(item)
    return found


def test_expected_source_matches_current_17_8() -> None:
    assert block_p.EXPECTED_SOURCE == build_preview_block_o_closure_audit()


def test_identity_order_reference_and_json_serializable() -> None:
    payload = _payload()
    assert list(payload) == block_p.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == block_p.SCHEMA_VERSION
    assert payload["block"] == "P"
    assert payload["step"] == "18.0"
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-18.1"
    ref = payload["block_o_closure_audit_reference"]
    assert ref["source_block_o_closure_audit_step"] == "FUNCTIONAL-PREVIEW-17.8"
    assert ref["block_p_opened_by_18_0"] is True
    json.dumps(payload)


def test_source_builder_called_exactly_once() -> None:
    calls = {"count": 0}

    def fake() -> dict[str, Any]:
        calls["count"] += 1
        return copy.deepcopy(block_p.EXPECTED_SOURCE)

    old = block_p.build_preview_block_o_closure_audit
    block_p.build_preview_block_o_closure_audit = fake
    try:
        assert (
            block_p.build_preview_block_p_desktop_exe_packaging_entry_contract()["block_p_opened"]
            is True
        )
    finally:
        block_p.build_preview_block_o_closure_audit = old
    assert calls["count"] == 1


def test_block_p_opened_source_only() -> None:
    payload = _payload()
    assert payload["block_p_opened"] is True
    assert payload["ready_for_block_p_1"] is True
    assert payload["entry_contract_summary"]["source_only"] is True
    assert payload["entry_contract_boundaries"]["build_tool_executed"] is False


def test_desktop_exe_product_direction() -> None:
    direction = _payload()["desktop_exe_product_direction"]
    assert direction["final_product_type"] == "desktop_exe"
    assert direction["desktop_ui_framework"] == "pyside6_qml"
    assert direction["desktop_application_entrypoint_requires_inventory"] is True
    assert direction["desktop_application_entrypoint_approved"] is False
    assert direction["desktop_artifact_created"] is False


def test_three_packaging_scope_rows() -> None:
    rows = _payload()["desktop_exe_packaging_scope_contract"]
    assert [row["scope_id"] for row in rows] == [
        "desktop_application_entrypoint",
        "qt_qml_runtime_bundle",
        "windows_exe_artifact_pipeline",
    ]
    assert all(row["classification"] == "packaging_scope_not_inventoried" for row in rows)
    assert all(row["scope_authorized"] is False for row in rows)


def test_eight_packaging_requirement_rows() -> None:
    rows = _payload()["desktop_exe_packaging_requirement_rows"]
    assert [row["requirement_id"] for row in rows] == [
        "desktop_application_entrypoint_inventory",
        "qml_asset_inventory",
        "qt_plugin_inventory",
        "python_dependency_inventory",
        "packaging_profile_alignment",
        "secret_and_local_data_exclusion_policy",
        "windows_target_toolchain_confirmation",
        "future_explicit_build_execution_gate",
    ]
    assert all(row["missing"] is True for row in rows)
    assert all(row["build_authorized"] is False for row in rows)


def test_cli_preview_plan_not_approved_as_final_desktop_contract() -> None:
    direction = _payload()["desktop_exe_product_direction"]
    assert direction["cli_preview_plan_detected_as_separate_scope"] is True
    assert direction["cli_preview_plan_is_final_desktop_contract"] is False
    assert direction["cli_preview_entrypoint_approved_as_final_desktop_entrypoint"] is False
    assert direction["cli_preview_artifact_approved_as_final_product"] is False


def test_real_capabilities_all_blocked() -> None:
    state = _payload()["real_capability_entry_state"]
    caps = state["block_p_capabilities"]
    assert caps
    assert set(caps.values()) == {"blocked"}
    assert state["real_capability_status_modified_by_18_0"] is False
    assert state["all_real_capabilities_blocked_at_block_p_entry"] is True


def test_fail_closed_entry_decision() -> None:
    decision = _payload()["fail_closed_entry_decision"]
    assert decision["block_o_closure_audit_in_17_8"] == "preserved"
    assert decision["block_p_entry_contract_in_18_0"] == "opened"
    assert decision["desktop_exe_packaging_source_inventory_in_18_1"] == "allowed"
    assert decision["build_executed_by_18_0"] is False
    assert decision["orders_enabled_by_18_0"] is False


def test_summary_evidence_and_boundaries() -> None:
    payload = _payload()
    summary = payload["entry_contract_summary"]
    evidence = payload["non_execution_entry_evidence"]
    boundaries = payload["entry_contract_boundaries"]
    assert summary["all_packaging_requirements_missing"] is True
    assert summary["all_real_capabilities_blocked"] is True
    assert evidence["source_builder_call_count"] == 1
    assert all(evidence["local_validity"].values())
    assert boundaries["reads_17_8_only"] is True
    assert boundaries["packaging_performed"] is False
    assert boundaries["runtime_started"] is False


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    ids = _container_ids(_payload())
    assert len(ids) == len(set(ids))


def test_independent_builder_calls_do_not_share_state() -> None:
    first = _payload()
    second = _payload()
    first["entry_contract_summary"]["source_only"] = False
    first["real_capability_entry_state"]["block_p_capabilities"]["runtime_activation"] = "open"
    assert second["entry_contract_summary"]["source_only"] is True
    assert (
        second["real_capability_entry_state"]["block_p_capabilities"]["runtime_activation"]
        == "blocked"
    )
    assert block_p.EXPECTED_SOURCE == build_preview_block_o_closure_audit()


def test_forbidden_raw_tokens_absent() -> None:
    text = Path(block_p.__file__).read_text()
    for token in ("create" + "_order", "fetch" + "_balance", "c" + "cxt"):
        assert token not in text


def test_exact_ast_guard() -> None:
    tree = ast.parse(Path(block_p.__file__).read_text())
    assert not [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert [node.module for node in imports] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_o_closure_audit",
    ]
    name_calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert name_calls == {
        "_all_plain_json",
        "_capability_map",
        "_copy_plain",
        "_exact_plain_matches",
        "_no_shadowing",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_plain_list_section",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "all",
        "bool",
        "build_preview_block_o_closure_audit",
        "enumerate",
        "id",
        "len",
        "list",
        "set",
        "type",
    }
    builder_calls = [
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "build_preview_block_o_closure_audit"
    ]
    assert len(builder_calls) == 1
    assert "build_preview_block_o_execution_authorization_readiness_read_model" not in name_calls
    forbidden = {
        "open",
        "compile",
        "exec",
        "eval",
        "getattr",
        "setattr",
        "hasattr",
        "__import__",
        "Path",
        "run",
        "Popen",
        "check_call",
        "check_output",
        "import_module",
        "load",
        "validate",
        "confirm",
        "grant",
        "package",
        "build",
        "release",
        "create",
        "start",
        "connect",
    }
    assert not (name_calls & forbidden)
    attribute_calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert attribute_calls == {
        "add",
        "append",
        "get",
        "items",
        "keys",
        "pop",
        "remove",
        "upper",
        "values",
    }


class Bomb:
    equality_calls = 0

    def __init__(self, target: str) -> None:
        self.target = target
        self.armed = False

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        if self.armed:
            raise AssertionError("bomb equality called")
        return False


class Lying(Bomb):
    __hash__ = Bomb.__hash__

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        if self.armed:
            return True
        return False


def _custom_first_section(
    field: str, owned: str, key_cls: type[Bomb]
) -> tuple[dict[str, Any], dict[Any, Any], list[Any], list[Any], Bomb]:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    original = source[field]
    custom = key_cls(owned)
    new_section: dict[Any, Any] = {custom: False}
    for key, value in original.items():
        if key != owned and len(new_section) < len(original) - 1:
            new_section[key] = value
    new_section[owned] = copy.deepcopy(original[owned]) if owned in original else False
    assert len(new_section) == len(original)
    assert next(iter(new_section)) is custom
    assert next(reversed(new_section)) == owned
    section_object = new_section
    keys_snapshot = list(new_section.keys())
    values_snapshot = copy.deepcopy(list(new_section.values()))
    custom_key_object = custom
    source[field] = section_object
    key_cls.equality_calls = 0
    return source, section_object, keys_snapshot, values_snapshot, custom_key_object


def _assert_custom_section_unchanged(
    source: dict[str, Any],
    section: str,
    section_object: dict[Any, Any],
    keys_snapshot: list[Any],
    values_snapshot: list[Any],
    custom_key_object: Bomb,
) -> None:
    current_section = source[section]
    assert current_section is section_object
    current_keys = list(current_section.keys())
    assert len(current_keys) == len(keys_snapshot)
    assert current_keys[0] is custom_key_object
    assert keys_snapshot[0] is custom_key_object
    for current, expected in zip(current_keys[1:], keys_snapshot[1:], strict=True):
        assert type(current) is str
        assert type(expected) is str
        assert current == expected
    current_values = list(current_section.values())
    assert len(current_values) == len(values_snapshot)
    assert current_values == values_snapshot


@pytest.mark.parametrize("key_cls", [Bomb, Lying])
@pytest.mark.parametrize("field", ["schema_version", "closure_audit_summary"])
def test_custom_top_level_keys_block_without_equality(key_cls: type[Bomb], field: str) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    custom = key_cls(field)
    source = {custom: "bad", **source}
    custom.armed = True
    key_cls.equality_calls = 0
    payload = _payload(source)
    _assert_blocked(payload)
    assert key_cls.equality_calls == 0
    assert any(id(key) == id(custom) for key in source)


@pytest.mark.parametrize("key_cls", [Bomb, Lying])
@pytest.mark.parametrize(
    ("section", "owned"),
    [
        ("invariant_closure_audit_state", "block_p_opened"),
        ("exe_closure_audit_state", "can_open_block_p"),
        *[("source_boundaries", field) for field in block_p.SOURCE_BOUNDARY_FIELDS_18_0],
    ],
)
def test_bomb_custom_first_nested_shadowing_blocks_without_equality(
    key_cls: type[Bomb], section: str, owned: str
) -> None:
    (
        source,
        section_object,
        keys_snapshot,
        values_snapshot,
        custom,
    ) = _custom_first_section(section, owned, key_cls)
    custom.armed = True
    original = block_p.EXPECTED_SOURCE[section]
    crafted = source[section]
    assert len(crafted) == len(original)
    assert next(iter(crafted)) is custom
    assert block_p._no_shadowing(source) is False
    payload = _payload(source)
    _assert_blocked(payload)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    if section == "invariant_closure_audit_state":
        assert local["invariant_valid"] is False
        assert local["exe_valid"] is True
    elif section == "exe_closure_audit_state":
        assert local["exe_valid"] is False
        assert local["invariant_valid"] is True
    else:
        assert local["source_boundaries_valid"] is False
        assert local["summary_valid"] is True
    assert key_cls.equality_calls == 0
    _assert_custom_section_unchanged(
        source, section, section_object, keys_snapshot, values_snapshot, custom
    )


def test_capability_maps_are_independent() -> None:
    payload = _payload()
    real = payload["real_capability_entry_state"]["block_p_capabilities"]
    fail = payload["fail_closed_entry_decision"]["block_p_capabilities"]
    assert real == fail
    assert real is not fail


def _deep_dict(depth: int) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root
    for _ in range(depth):
        nxt: dict[str, Any] = {}
        current["x"] = nxt
        current = nxt
    return root


def _deep_list(depth: int) -> list[Any]:
    root: list[Any] = []
    current = root
    for _ in range(depth):
        nxt: list[Any] = []
        current.append(nxt)
        current = nxt
    return root


@pytest.mark.parametrize(
    "bad_source",
    [
        None,
        [],
        type("DictSub", (dict,), {})(block_p.EXPECTED_SOURCE),
        {**block_p.EXPECTED_SOURCE, "extra": True},
        {key: value for key, value in list(block_p.EXPECTED_SOURCE.items())[1:]},
        dict(reversed(list(block_p.EXPECTED_SOURCE.items()))),
        {**block_p.EXPECTED_SOURCE, "status": "changed"},
        {**block_p.EXPECTED_SOURCE, "block_o_closure_audit_decision": "changed"},
        {**block_p.EXPECTED_SOURCE, "block_o_closure_audit_ready": 1},
        {**block_p.EXPECTED_SOURCE, "step": 17.8},
        {
            **block_p.EXPECTED_SOURCE,
            "closure_audit_summary": type("DictSub", (dict,), {})(
                block_p.EXPECTED_SOURCE["closure_audit_summary"]
            ),
        },
        {
            **block_p.EXPECTED_SOURCE,
            "domain_closure_audit_rows": type("ListSub", (list,), {})(
                block_p.EXPECTED_SOURCE["domain_closure_audit_rows"]
            ),
        },
        {**block_p.EXPECTED_SOURCE, "future_steps": [type("ListSub", (list,), {})([])]},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": object()},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": lambda: None},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": set()},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": tuple()},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": 1.25},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": _deep_dict(1500)},
        {**block_p.EXPECTED_SOURCE, "closure_evidence": _deep_list(1500)},
    ],
)
def test_malformed_sources_block_and_stay_json_safe(bad_source: Any) -> None:
    can_snapshot = block_p._all_plain_json(bad_source, block_p.MAX_DIAGNOSTIC_CONTAINER_DEPTH)
    snapshot = copy.deepcopy(bad_source) if can_snapshot else bad_source
    payload = _payload(bad_source)
    _assert_blocked(payload)
    if can_snapshot:
        assert bad_source == snapshot


def test_cycles_rejected_shared_acyclic_allowed_and_unbounded_deep_no_recursion_error() -> None:
    cyc: dict[str, Any] = {}
    cyc["self"] = cyc
    assert block_p._all_plain_json(cyc) is False
    lst: list[Any] = []
    lst.append(lst)
    assert block_p._all_plain_json(lst) is False
    shared: list[Any] = [1]
    assert block_p._all_plain_json({"a": shared, "b": shared}) is True
    assert block_p._all_plain_json(_deep_dict(1500)) is True
    assert block_p._all_plain_json(_deep_list(1500)) is True
    assert block_p._all_plain_json(_deep_dict(65), block_p.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False


@pytest.mark.parametrize(
    "field",
    [
        "closure_evidence",
        "closure_boundaries",
        "source_boundaries",
        "future_steps",
        "fail_closed_closure_decision",
    ],
)
def test_invalid_section_isolation_preserves_independent_inherited_facts(field: str) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source[field] = [] if type(source[field]) is dict else {}
    payload = _payload(source)
    _assert_blocked(payload)
    inherited = payload["inherited_block_o_closure_summary"]
    assert len(inherited["domain_closure_audit_rows"]) == 2
    assert len(inherited["requirement_closure_audit_rows"]) == 7
    assert inherited["invariant_closure_audit_state"]
    assert inherited["exe_closure_audit_state"]
    assert inherited["real_capability_closure_audit_state"]


@pytest.mark.parametrize("field", ["domain_closure_audit_rows", "requirement_closure_audit_rows"])
def test_invalid_domain_or_requirement_degrades_dependent_claim_only(field: str) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source[field] = []
    payload = _payload(source)
    _assert_blocked(payload)
    inherited = payload["inherited_block_o_closure_summary"]
    assert inherited[field] == []
    assert inherited["invariant_closure_audit_state"]


class BombValue:
    equality_calls = 0
    armed = False

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("identity value equality called")
        return False


class LyingValue(BombValue):
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


IDENTITY_FIELDS = list(block_p.SOURCE_IDENTITY_EXPECTED.keys())


@pytest.mark.parametrize("value_cls", [BombValue, LyingValue])
@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_top_level_custom_identity_values_do_not_call_equality(
    value_cls: type[BombValue], field: str
) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    custom_value = value_cls()
    value_cls.armed = False
    source[field] = custom_value
    value_cls.equality_calls = 0
    value_cls.armed = True
    try:
        payload = _payload(source)
    finally:
        value_cls.armed = False
    assert value_cls.equality_calls == 0
    _assert_blocked(payload)
    json.dumps(payload)
    assert source[field] is custom_value


def _identity_mutations(field: str) -> list[Any]:
    expected = block_p.SOURCE_IDENTITY_EXPECTED[field]
    if type(expected) is bool:
        values: list[Any] = [not expected, 1]
    else:
        values = [f"{expected}_changed", 123]
    if field == "step":
        values.append(17.8)
    values.extend([{}, [], object(), lambda: None])
    return values


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_exact_scalar_identity_mutation_matrix(field: str) -> None:
    for value in _identity_mutations(field):
        source = copy.deepcopy(block_p.EXPECTED_SOURCE)
        source[field] = value
        before_plain = copy.deepcopy(source) if block_p._all_plain_json(source) else None
        payload = _payload(source)
        _assert_blocked(payload)
        if before_plain is not None:
            assert source == before_plain
        else:
            assert source[field] is value


def test_inherited_capability_map_is_exact_inner_map() -> None:
    payload = _payload()
    inherited_caps = payload["real_capability_entry_state"]["inherited_block_o_capabilities"]
    expected_caps = block_p.EXPECTED_SOURCE["real_capability_closure_audit_state"][
        "real_capability_status"
    ]
    assert inherited_caps == expected_caps
    assert "real_capability_status" not in inherited_caps
    assert set(inherited_caps.values()) == {"blocked"}


def test_invalid_real_capability_source_makes_all_claim_non_vacuous_false() -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source["real_capability_closure_audit_state"] = {}
    payload = _payload(source)
    _assert_blocked(payload)
    state = payload["real_capability_entry_state"]
    assert state["real_capability_status_inherited_from_17_8"] is False
    assert state["inherited_block_o_capabilities"] == {}
    assert state["block_p_capabilities"]
    assert set(state["block_p_capabilities"].values()) == {"blocked"}
    assert state["all_real_capabilities_blocked_at_block_p_entry"] is False
    assert payload["entry_contract_summary"]["all_real_capabilities_blocked"] is False


def test_capability_maps_are_four_independent_mutable_maps() -> None:
    payload = _payload()
    inherited = payload["real_capability_entry_state"]["inherited_block_o_capabilities"]
    block_p_caps = payload["real_capability_entry_state"]["block_p_capabilities"]
    fail_caps = payload["fail_closed_entry_decision"]["block_p_capabilities"]
    expected_caps = block_p.EXPECTED_SOURCE["real_capability_closure_audit_state"][
        "real_capability_status"
    ]
    assert len({id(inherited), id(block_p_caps), id(fail_caps), id(expected_caps)}) == 4
    inherited_key = next(iter(inherited))
    block_p_key = next(iter(block_p_caps))
    fail_key = next(iter(fail_caps))
    inherited[inherited_key] = "changed_inherited"
    block_p_caps[block_p_key] = "changed_block_p"
    fail_caps[fail_key] = "changed_fail"
    assert expected_caps[inherited_key] == "blocked"
    assert block_p_caps.get(inherited_key) != "changed_inherited"
    assert fail_caps.get(block_p_key) != "changed_block_p"
    assert inherited.get(fail_key) != "changed_fail"


def test_independent_builder_calls_do_not_share_capability_maps() -> None:
    first = _payload()
    second = _payload()
    paths = [
        ("real_capability_entry_state", "inherited_block_o_capabilities"),
        ("real_capability_entry_state", "block_p_capabilities"),
        ("fail_closed_entry_decision", "block_p_capabilities"),
    ]
    for section, field in paths:
        assert first[section][field] is not second[section][field]
        first_key = next(iter(first[section][field]))
        first[section][field][first_key] = "mutated"
        assert second[section][field][first_key] == "blocked"


@pytest.mark.parametrize(
    ("field", "validity_name"),
    [
        ("closure_audit_summary", "summary_valid"),
        ("domain_closure_audit_rows", "domain_rows_valid"),
        ("invariant_closure_audit_state", "invariant_valid"),
        ("source_boundaries", "source_boundaries_valid"),
    ],
)
def test_builder_level_cycles_block_without_recursion_error(field: str, validity_name: str) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    if type(source[field]) is list:
        cyclic: list[Any] = []
        cyclic.append(cyclic)
    else:
        cyclic_dict: dict[str, Any] = {}
        cyclic_dict["self"] = cyclic_dict
        cyclic = cyclic_dict
    source[field] = cyclic
    payload = _payload(source)
    _assert_blocked(payload)
    assert payload["non_execution_entry_evidence"]["local_validity"][validity_name] is False
    if type(cyclic) is list:
        assert cyclic[0] is cyclic
    else:
        assert cyclic["self"] is cyclic


def test_exact_scope_rows_field_order_and_values() -> None:
    rows = _payload()["desktop_exe_packaging_scope_contract"]
    expected_ids = [
        (
            "desktop_application_entrypoint",
            "desktop_application_entrypoint_not_inventoried_build_unauthorized",
        ),
        ("qt_qml_runtime_bundle", "qt_qml_runtime_bundle_not_inventoried_build_unauthorized"),
        (
            "windows_exe_artifact_pipeline",
            "windows_exe_artifact_pipeline_not_inventoried_build_unauthorized",
        ),
    ]
    expected_fields = [
        "scope_id",
        "scope_required",
        "scope_inventory_present",
        "scope_inventory_complete",
        "scope_validated",
        "scope_ready",
        "scope_authorized",
        "requires_future_explicit_step",
        "failure_policy",
        "classification",
        "result",
    ]
    assert len(rows) == 3
    for row, (scope_id, result) in zip(rows, expected_ids, strict=True):
        assert type(row) is dict
        assert list(row) == expected_fields
        assert row == {
            "scope_id": scope_id,
            "scope_required": True,
            "scope_inventory_present": False,
            "scope_inventory_complete": False,
            "scope_validated": False,
            "scope_ready": False,
            "scope_authorized": False,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "classification": "packaging_scope_not_inventoried",
            "result": result,
        }


def test_exact_requirement_rows_field_order_and_values() -> None:
    rows = _payload()["desktop_exe_packaging_requirement_rows"]
    expected_ids = [
        "desktop_application_entrypoint_inventory",
        "qml_asset_inventory",
        "qt_plugin_inventory",
        "python_dependency_inventory",
        "packaging_profile_alignment",
        "secret_and_local_data_exclusion_policy",
        "windows_target_toolchain_confirmation",
        "future_explicit_build_execution_gate",
    ]
    expected_fields = [
        "requirement_id",
        "required",
        "present",
        "completed",
        "satisfied",
        "missing",
        "missing_blocks_build",
        "requires_future_explicit_step",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "failure_policy",
        "classification",
        "result",
    ]
    assert len(rows) == 8
    for row, requirement_id in zip(rows, expected_ids, strict=True):
        assert type(row) is dict
        assert list(row) == expected_fields
        assert row == {
            "requirement_id": requirement_id,
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "missing": True,
            "missing_blocks_build": True,
            "requires_future_explicit_step": True,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "failure_policy": "fail_closed",
            "classification": "missing_packaging_requirement",
            "result": "missing_packaging_requirement_build_unauthorized",
        }


DICT_SECTION_VALIDITY = {
    "block_o_execution_authorization_readiness_read_model_reference": "reference_valid",
    "closure_audit_summary": "summary_valid",
    "invariant_closure_audit_state": "invariant_valid",
    "exe_closure_audit_state": "exe_valid",
    "real_capability_closure_audit_state": "real_capability_valid",
    "fail_closed_closure_decision": "fail_closed_valid",
    "closure_evidence": "evidence_valid",
    "closure_boundaries": "closure_boundaries_valid",
    "source_boundaries": "source_boundaries_valid",
}


def _first_scalar_key(section: dict[str, Any]) -> str:
    for key, value in section.items():
        if type(value) in (str, bool, int):
            return key
    raise AssertionError("no scalar key found")


def _changed_scalar(value: Any) -> Any:
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return f"{value}_changed"
    return "changed"


def _assert_blocked_plain_unmutated(source: Any, before: Any) -> dict[str, Any]:
    payload = _payload(source)
    _assert_blocked(payload)
    assert block_p._all_plain_json(payload)
    assert source == before
    return payload


def _dict_section_cases(section_name: str) -> list[dict[str, Any]]:
    source_section = block_p.EXPECTED_SOURCE[section_name]
    scalar_key = _first_scalar_key(source_section)
    missing_key = next(iter(source_section))
    reversed_section = dict(reversed(list(source_section.items())))
    extra_section = copy.deepcopy(source_section)
    extra_section["mutation_extra_field"] = True
    missing_section = copy.deepcopy(source_section)
    del missing_section[missing_key]
    changed_section = copy.deepcopy(source_section)
    changed_section[scalar_key] = _changed_scalar(changed_section[scalar_key])
    bool_bypass_section = copy.deepcopy(source_section)
    bool_key = next(
        (key for key, value in source_section.items() if type(value) is bool), scalar_key
    )
    bool_bypass_section[bool_key] = 1
    return [
        {section_name: extra_section},
        {section_name: missing_section},
        {section_name: reversed_section},
        {section_name: []},
        {section_name: changed_section},
        {section_name: bool_bypass_section},
        {section_name: type("DictSub", (dict,), {})(source_section)},
    ]


@pytest.mark.parametrize("section_name", list(DICT_SECTION_VALIDITY))
def test_exact_source_dict_section_mutation_matrix(section_name: str) -> None:
    for mutation in _dict_section_cases(section_name):
        source = copy.deepcopy(block_p.EXPECTED_SOURCE)
        source.update(mutation)
        before = copy.deepcopy(source)
        payload = _assert_blocked_plain_unmutated(source, before)
        local = payload["non_execution_entry_evidence"]["local_validity"]
        assert local[DICT_SECTION_VALIDITY[section_name]] is False
        assert any(
            value is True
            for key, value in local.items()
            if key != DICT_SECTION_VALIDITY[section_name]
        )


def _row_cases(list_name: str, changed_values: dict[str, Any]) -> list[Any]:
    rows = block_p.EXPECTED_SOURCE[list_name]
    first_row = rows[0]
    first_key = next(iter(first_row))
    cases: list[Any] = [
        {},
        type("ListSub", (list,), {})(rows),
        rows[:-1],
        [*rows, copy.deepcopy(rows[0])],
        list(reversed(rows)),
        ["not_a_dict", *rows[1:]],
        [type("DictSub", (dict,), {})(first_row), *rows[1:]],
    ]
    extra_row = copy.deepcopy(first_row)
    extra_row["mutation_extra_field"] = True
    cases.append([extra_row, *rows[1:]])
    missing_row = copy.deepcopy(first_row)
    del missing_row[first_key]
    cases.append([missing_row, *rows[1:]])
    cases.append([dict(reversed(list(first_row.items()))), *rows[1:]])
    for key, value in changed_values.items():
        changed_row = copy.deepcopy(first_row)
        changed_row[key] = value
        cases.append([changed_row, *rows[1:]])
    return cases


@pytest.mark.parametrize(
    "rows_mutation",
    _row_cases(
        "domain_closure_audit_rows",
        {
            "domain": "changed_domain",
            "closure_classification": "changed_classification",
            "closure_result": "changed_result",
            "source_capability_count": 999,
            "block_o_scope_complete_for_domain": False,
            "domain_closed_in_block_o": False,
            "execution_ready_after_block_o_closure": True,
            "execution_authorized_after_block_o_closure": True,
            "failure_policy": "open",
        },
    ),
)
def test_exact_domain_source_row_mutation_matrix(rows_mutation: Any) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source["domain_closure_audit_rows"] = rows_mutation
    before = copy.deepcopy(source)
    payload = _assert_blocked_plain_unmutated(source, before)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    assert local["domain_rows_valid"] is False
    assert local["requirement_rows_valid"] is True
    assert local["invariant_valid"] is True


@pytest.mark.parametrize(
    "rows_mutation",
    _row_cases(
        "requirement_closure_audit_rows",
        {
            "requirement_id": "changed_requirement",
            "closure_classification": "changed_classification",
            "closure_result": "changed_result",
            "source_missing": False,
            "source_satisfied": True,
            "requirement_closed_as_missing_in_block_o": False,
            "requirement_ready_after_block_o_closure": True,
            "execution_authorized_after_block_o_closure": True,
            "failure_policy": "open",
        },
    ),
)
def test_exact_requirement_source_row_mutation_matrix(rows_mutation: Any) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source["requirement_closure_audit_rows"] = rows_mutation
    before = copy.deepcopy(source)
    payload = _assert_blocked_plain_unmutated(source, before)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    assert local["requirement_rows_valid"] is False
    assert local["domain_rows_valid"] is True
    assert local["invariant_valid"] is True


def _capability_status_cases() -> list[dict[str, Any]]:
    expected = block_p.EXPECTED_SOURCE["real_capability_closure_audit_state"][
        "real_capability_status"
    ]
    first_key = next(iter(expected))
    cases: list[dict[str, Any]] = [{}]
    missing = copy.deepcopy(expected)
    del missing[first_key]
    cases.append(missing)
    extra = copy.deepcopy(expected)
    extra["mutation_extra_capability"] = "blocked"
    cases.append(extra)
    cases.append(dict(reversed(list(expected.items()))))
    changed_open = copy.deepcopy(expected)
    changed_open[first_key] = "open"
    cases.append(changed_open)
    changed_bool = copy.deepcopy(expected)
    changed_bool[first_key] = True
    cases.append(changed_bool)
    cases.append(type("DictSub", (dict,), {})(expected))
    return cases


@pytest.mark.parametrize("capability_status", _capability_status_cases())
def test_real_capability_source_mutation_matrix(capability_status: dict[str, Any]) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source["real_capability_closure_audit_state"]["real_capability_status"] = capability_status
    before = copy.deepcopy(source)
    payload = _assert_blocked_plain_unmutated(source, before)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    state = payload["real_capability_entry_state"]
    summary = payload["entry_contract_summary"]
    assert local["real_capability_valid"] is False
    assert state["inherited_block_o_capabilities"] == {}
    assert state["real_capability_status_inherited_from_17_8"] is False
    assert state["all_real_capabilities_blocked_at_block_p_entry"] is False
    assert summary["all_real_capabilities_blocked"] is False
    assert state["block_p_capabilities"]
    assert set(state["block_p_capabilities"].values()) == {"blocked"}


@pytest.mark.parametrize(
    ("section", "field", "local_key"),
    [
        ("domain_closure_audit_rows", "execution_ready_after_block_o_closure", "domain_rows_valid"),
        (
            "domain_closure_audit_rows",
            "execution_authorized_after_block_o_closure",
            "domain_rows_valid",
        ),
        (
            "requirement_closure_audit_rows",
            "requirement_ready_after_block_o_closure",
            "requirement_rows_valid",
        ),
        (
            "requirement_closure_audit_rows",
            "execution_authorized_after_block_o_closure",
            "requirement_rows_valid",
        ),
        (
            "invariant_closure_audit_state",
            "execution_ready_after_block_o_closure",
            "invariant_valid",
        ),
        (
            "invariant_closure_audit_state",
            "execution_authorized_after_block_o_closure",
            "invariant_valid",
        ),
        ("exe_closure_audit_state", "build_ready_after_block_o_closure", "exe_valid"),
        ("exe_closure_audit_state", "packaging_ready_after_block_o_closure", "exe_valid"),
        ("exe_closure_audit_state", "release_ready_after_block_o_closure", "exe_valid"),
        ("exe_closure_audit_state", "execution_ready_after_block_o_closure", "exe_valid"),
        ("exe_closure_audit_state", "execution_authorized_after_block_o_closure", "exe_valid"),
        (
            "fail_closed_closure_decision",
            "execution_readiness_granted_by_17_8",
            "fail_closed_valid",
        ),
        (
            "fail_closed_closure_decision",
            "execution_authorization_granted_by_17_8",
            "fail_closed_valid",
        ),
        ("fail_closed_closure_decision", "runtime_enabled_by_17_8", "fail_closed_valid"),
        ("fail_closed_closure_decision", "orders_enabled_by_17_8", "fail_closed_valid"),
        (
            "fail_closed_closure_decision",
            "packaging_build_release_enabled_by_17_8",
            "fail_closed_valid",
        ),
    ],
)
def test_source_readiness_authorization_bypass_matrix(
    section: str, field: str, local_key: str
) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    if type(source[section]) is list:
        source[section][0][field] = True
    else:
        source[section][field] = True
    before = copy.deepcopy(source)
    payload = _assert_blocked_plain_unmutated(source, before)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    assert local[local_key] is False
    assert payload["block_p_opened"] is False
    summary = payload["entry_contract_summary"]
    boundaries = payload["entry_contract_boundaries"]
    decision = payload["fail_closed_entry_decision"]
    for key in [
        "build_execution_authorized",
        "artifact_creation_authorized",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ]:
        assert summary[key] is False
    for key in [
        "build_tool_executed",
        "packaging_performed",
        "artifact_created",
        "release_performed",
        "runtime_started",
        "orders_enabled",
    ]:
        assert boundaries[key] is False
    for key in [
        "build_approved_by_18_0",
        "build_executed_by_18_0",
        "artifact_created_by_18_0",
        "release_approved_by_18_0",
        "runtime_enabled_by_18_0",
        "orders_enabled_by_18_0",
    ]:
        assert decision[key] is False


def _future_step_cases() -> list[Any]:
    steps = block_p.EXPECTED_SOURCE["future_steps"]
    first = steps[0]
    cases: list[Any] = [
        {},
        type("ListSub", (list,), {})(steps),
        [],
        [*steps, first],
        [{"step": first}],
        ["changed_future_step"],
        [123],
        [True],
        [None],
        list(reversed([*steps, "second_step"])),
        [first, "mutation_extra_row"],
    ]
    return cases


@pytest.mark.parametrize("future_steps", _future_step_cases())
def test_future_steps_mutation_matrix(future_steps: Any) -> None:
    source = copy.deepcopy(block_p.EXPECTED_SOURCE)
    source["future_steps"] = future_steps
    before = copy.deepcopy(source)
    payload = _assert_blocked_plain_unmutated(source, before)
    local = payload["non_execution_entry_evidence"]["local_validity"]
    assert local["future_steps_valid"] is False
    assert payload["block_p_opened"] is False


def test_custom_section_unchanged_detects_nested_list_mutation() -> None:
    source, section_object, keys_snapshot, values_snapshot, custom = _custom_first_section(
        "invariant_closure_audit_state", "block_p_opened", Bomb
    )
    custom.armed = True
    nested_list = next(value for value in section_object.values() if type(value) is list)
    nested_list.append({"mutation_sentinel": True})
    with pytest.raises(AssertionError):
        _assert_custom_section_unchanged(
            source,
            "invariant_closure_audit_state",
            section_object,
            keys_snapshot,
            values_snapshot,
            custom,
        )


def test_custom_section_unchanged_detects_nested_dict_mutation() -> None:
    source, section_object, keys_snapshot, values_snapshot, custom = _custom_first_section(
        "source_boundaries", "source_block_o_closure_audit", Bomb
    )
    custom.armed = True
    nested_dict = next(value for value in section_object.values() if type(value) is dict)
    nested_dict["mutation_sentinel"] = True
    with pytest.raises(AssertionError):
        _assert_custom_section_unchanged(
            source,
            "source_boundaries",
            section_object,
            keys_snapshot,
            values_snapshot,
            custom,
        )


def test_custom_section_unchanged_detects_scalar_mutation() -> None:
    source, section_object, keys_snapshot, values_snapshot, custom = _custom_first_section(
        "source_boundaries", "source_block_o_closure_audit", Bomb
    )
    custom.armed = True
    scalar_key = next(key for key in list(section_object)[1:] if type(section_object[key]) is bool)
    section_object[scalar_key] = not section_object[scalar_key]
    with pytest.raises(AssertionError):
        _assert_custom_section_unchanged(
            source,
            "source_boundaries",
            section_object,
            keys_snapshot,
            values_snapshot,
            custom,
        )


@pytest.mark.parametrize("key_cls", [Bomb, Lying])
@pytest.mark.parametrize(
    ("section", "owned"),
    [
        ("invariant_closure_audit_state", "block_p_opened"),
        ("exe_closure_audit_state", "can_open_block_p"),
        *[("source_boundaries", field) for field in block_p.SOURCE_BOUNDARY_FIELDS_18_0],
    ],
)
def test_direct_no_shadowing_custom_first_matrix_no_mutation(
    key_cls: type[Bomb], section: str, owned: str
) -> None:
    source, section_object, keys_snapshot, values_snapshot, custom = _custom_first_section(
        section, owned, key_cls
    )
    custom.armed = True
    key_cls.equality_calls = 0
    try:
        result = block_p._no_shadowing(source)
    finally:
        custom.armed = False
    assert result is False
    assert key_cls.equality_calls == 0
    _assert_custom_section_unchanged(
        source, section, section_object, keys_snapshot, values_snapshot, custom
    )


@pytest.mark.parametrize("key_cls", [Bomb, Lying])
@pytest.mark.parametrize(
    ("section", "owned"),
    [
        ("invariant_closure_audit_state", "block_p_opened"),
        ("exe_closure_audit_state", "can_open_block_p"),
        *[("source_boundaries", field) for field in block_p.SOURCE_BOUNDARY_FIELDS_18_0],
    ],
)
def test_builder_level_custom_first_matrix_no_mutation(
    key_cls: type[Bomb], section: str, owned: str
) -> None:
    source, section_object, keys_snapshot, values_snapshot, custom = _custom_first_section(
        section, owned, key_cls
    )
    custom.armed = True
    key_cls.equality_calls = 0
    try:
        payload = _payload(source)
    finally:
        custom.armed = False
    _assert_blocked(payload)
    assert block_p._all_plain_json(payload)
    assert key_cls.equality_calls == 0
    local = payload["non_execution_entry_evidence"]["local_validity"]
    if section == "invariant_closure_audit_state":
        assert local["invariant_valid"] is False
        assert local["exe_valid"] is True
    elif section == "exe_closure_audit_state":
        assert local["exe_valid"] is False
        assert local["invariant_valid"] is True
    else:
        assert local["source_boundaries_valid"] is False
        assert local["summary_valid"] is True
    _assert_custom_section_unchanged(
        source, section, section_object, keys_snapshot, values_snapshot, custom
    )
