from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app import preview_block_p_desktop_exe_packaging_contract as contract
from ui.pyside_app.preview_block_p_desktop_exe_packaging_inventory_matrix import (
    build_preview_block_p_desktop_exe_packaging_inventory_matrix,
)


def _payload() -> dict[str, Any]:
    return contract.build_preview_block_p_desktop_exe_packaging_contract()


def _assert_plain(value: Any) -> None:
    json.dumps(value, sort_keys=True)
    assert contract._all_plain_json(value, contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH)


def test_expected_source_matches_current_18_2() -> None:
    assert (
        contract.EXPECTED_SOURCE == build_preview_block_p_desktop_exe_packaging_inventory_matrix()
    )
    assert list(contract.EXPECTED_SOURCE) == contract.TOP_LEVEL_FIELDS_18_2


def test_top_level_identity_and_reference() -> None:
    payload = _payload()
    assert list(payload) == contract.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == contract.SCHEMA_VERSION
    assert payload["block_p_desktop_exe_packaging_contract_kind"] == contract.KIND
    assert payload["block"] == "P"
    assert payload["step"] == "18.3"
    assert payload["packaging_contract_artifact_complete"] is True
    assert payload["ready_for_block_p_4"] is True
    ref = payload["block_p_desktop_exe_packaging_inventory_matrix_reference"]
    assert (
        ref["source_block_p_desktop_exe_packaging_inventory_matrix_step"]
        == "FUNCTIONAL-PREVIEW-18.2"
    )
    assert ref["source_inventory_matrix_read_by_18_3"] is True
    assert ref["build_performed"] is False
    _assert_plain(payload)


def test_builder_called_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    source = build_preview_block_p_desktop_exe_packaging_inventory_matrix()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_packaging_inventory_matrix", fake
    )
    payload = contract.build_preview_block_p_desktop_exe_packaging_contract()
    assert calls == 1
    assert payload["non_execution_contract_evidence"]["source_builder_call_count"] == 1


def test_counts_principles_domains_and_future_steps() -> None:
    payload = _payload()
    summary = payload["packaging_contract_summary"]
    assert summary["scope_contract_count"] == 3
    assert summary["requirement_contract_count"] == 8
    assert summary["blocker_contract_count"] == 12
    assert summary["evidence_requirement_count"] == 12
    assert summary["acceptance_rule_count"] == 6
    assert len(payload["contract_principles"]) == 8
    assert payload["desktop_entrypoint_contract"]["candidate_paths"] == [
        "ui/pyside_app/__main__.py",
        "ui/pyside_app/app.py",
    ]
    assert payload["qml_bundle_contract"]["pyside_qml_file_count"] == 24
    assert payload["qml_bundle_contract"]["shared_qml_file_count"] == 107
    assert payload["python_dependency_contract"]["build_tool_candidates"] == [
        "pyinstaller",
        "briefcase",
    ]
    assert payload["future_steps"] == [
        {
            "step": "18.4",
            "title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.5",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS MATRIX",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.6",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.7",
            "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
            "source_only": True,
            "build_performed": False,
        },
        {
            "step": "18.8",
            "title": "BLOCK P CLOSURE AUDIT",
            "source_only": True,
            "build_performed": False,
        },
    ]


def test_scope_requirement_blocker_evidence_acceptance_integrity() -> None:
    payload = _payload()
    blockers = payload["unresolved_blocker_contract_rows"]
    evidence = payload["contract_evidence_requirement_rows"]
    scopes = payload["packaging_scope_contract_rows"]
    reqs = payload["packaging_requirement_contract_rows"]
    acceptance = payload["contract_acceptance_rule_rows"]
    blocker_ids = [b["blocker_id"] for b in blockers]
    evidence_ids = [e["evidence_id"] for e in evidence]
    assert blocker_ids == [
        r["blocker_id"] for r in contract.EXPECTED_SOURCE["unresolved_contract_blocker_rows"]
    ]
    assert len(blocker_ids) == len(set(blocker_ids)) == 12
    assert len(evidence_ids) == len(set(evidence_ids)) == 12
    assert [s["unresolved_blocker_count"] for s in scopes] == [2, 5, 5]
    for row in blockers:
        assert row["contract_clause_defined"] is True
        assert row["resolved_by_18_3"] is False
        assert row["blocks_build_readiness"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["required_evidence_ids"][0] in evidence_ids
    for row in evidence:
        assert row["blocker_id"] in blocker_ids
        assert row["required_artifacts"]
        assert row["collected_by_18_3"] is False
        assert row["validated_by_18_3"] is False
    for row in scopes + reqs:
        assert row["ready_for_read_model"] is True
        assert row["build_ready"] is False
    for rule in acceptance:
        assert set(rule["required_blocker_ids"]).issubset(blocker_ids)
        assert set(rule["required_evidence_ids"]).issubset(evidence_ids)
        assert rule["rule_satisfied_by_18_3"] is False
        assert rule["grants_build_authorization"] is False


def test_capabilities_and_zero_operational_grants() -> None:
    payload = _payload()
    caps = payload["real_capability_contract_state"]
    assert caps["packaging_contract_capabilities"]
    assert set(caps["packaging_contract_capabilities"].values()) == {"blocked"}
    summary = payload["packaging_contract_summary"]
    for key in [
        "any_blocker_resolved",
        "all_blockers_resolved",
        "all_required_evidence_collected",
        "all_required_evidence_validated",
        "contract_satisfied",
        "desktop_entrypoint_selected",
        "desktop_entrypoint_validated",
        "qml_bundle_validated",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_created",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ]:
        assert summary[key] is False


def _mutable_ids(value: Any) -> list[int]:
    stack = [value]
    ids = []
    while stack:
        item = stack.pop()
        if type(item) is dict:
            ids.append(id(item))
            stack.extend(item.values())
        elif type(item) is list:
            ids.append(id(item))
            stack.extend(item)
    return ids


def test_no_shared_mutable_containers_and_independent_calls() -> None:
    one = _payload()
    two = _payload()
    assert len(_mutable_ids(one)) == len(set(_mutable_ids(one)))
    one["packaging_scope_contract_rows"][0]["scope_id"] = "mutated"
    assert two["packaging_scope_contract_rows"][0]["scope_id"] == "desktop_application_entrypoint"
    assert (
        contract.EXPECTED_SOURCE["packaging_scope_matrix_rows"][0]["scope_id"]
        == "desktop_application_entrypoint"
    )


@pytest.mark.parametrize("bad", [None, 1, 1.0, object(), set(), tuple()])
def test_malformed_source_blocks(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_packaging_inventory_matrix", lambda: bad
    )
    payload = contract.build_preview_block_p_desktop_exe_packaging_contract()
    assert payload["packaging_contract_artifact_complete"] is False
    assert payload["ready_for_block_p_4"] is False
    assert payload["status"] == contract.BLOCKED_STATUS
    _assert_plain(payload)


def test_cycle_deep_shared_plain_helpers() -> None:
    cyc: dict[str, Any] = {}
    cyc["x"] = cyc
    assert contract._all_plain_json(cyc, contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    deep: Any = []
    for _ in range(1500):
        deep = [deep]
    assert contract._all_plain_json(deep, 2000) is True
    assert contract._all_plain_json(deep, contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    shared: list[Any] = []
    assert (
        contract._all_plain_json([shared, shared], contract.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is True
    )


class Bomb:
    equality_calls = 0
    armed = False

    def __init__(self, target: str) -> None:
        self.target = target

    def __hash__(self) -> int:
        return hash(self.target)

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("custom equality called")
        return False


def test_custom_keys_do_not_trigger_equality(monkeypatch: pytest.MonkeyPatch) -> None:
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    custom = Bomb("schema_version")
    source = {custom: "shadow", **source}
    Bomb.armed = True
    try:
        payload = None
        monkeypatch.setattr(
            contract, "build_preview_block_p_desktop_exe_packaging_inventory_matrix", lambda: source
        )
        payload = contract.build_preview_block_p_desktop_exe_packaging_contract()
    finally:
        Bomb.armed = False
    assert Bomb.equality_calls == 0
    assert payload is not None
    assert payload["packaging_contract_artifact_complete"] is False


def test_forbidden_raw_tokens_absent() -> None:
    text = Path(contract.__file__).read_text()
    assert "ccxt" not in text
    assert "create_order" not in text
    assert "fetch_balance" not in text


def test_exact_ast_guard() -> None:
    tree = ast.parse(Path(contract.__file__).read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert [n.module for n in imports] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_p_desktop_exe_packaging_inventory_matrix",
    ]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    name_call_list = [n.func.id for n in calls if isinstance(n.func, ast.Name)]
    assert name_call_list.count("build_preview_block_p_desktop_exe_packaging_inventory_matrix") == 1
    assert not any("source_inventory" in name for name in name_call_list)
    forbidden = {
        "open",
        "read_text",
        "read_bytes",
        "write_text",
        "write_bytes",
        "glob",
        "rglob",
        "iterdir",
        "exists",
        "resolve",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "getenv",
        "home",
        "expanduser",
        "environ",
        "run",
        "Popen",
        "call",
        "check_call",
        "check_output",
        "system",
        "__import__",
        "import_module",
        "find_spec",
        "QGuiApplication",
        "QQmlApplicationEngine",
        "PyInstaller",
        "briefcase",
        "package",
        "build",
        "freeze",
        "sign",
        "upload",
        "publish",
        "release",
        "exec",
        "start",
        "submit_order",
        "cancel_order",
        "replace_order",
    }
    name_calls = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    attribute_calls = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    assert name_calls == {
        "_acceptance_rows",
        "_all_plain_json",
        "_blocker_contract_rows",
        "_capability_map_known_blocked",
        "_contract_principles",
        "_copy_plain",
        "_evidence_rows",
        "_exact_plain_matches",
        "_get_exact_string_key",
        "_no_shadowing",
        "_nonempty_unique",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_plain_list_section",
        "_requirement_contract_rows",
        "_safe_top_level_source",
        "_scalar_reference",
        "_scope_contract_rows",
        "_section_valid",
        "_source_identity_valid",
        "_source_referential_integrity_valid",
        "all",
        "bool",
        "build_preview_block_p_desktop_exe_packaging_inventory_matrix",
        "id",
        "len",
        "list",
        "set",
        "type",
        "zip",
    }
    assert attribute_calls == {
        "add",
        "append",
        "discard",
        "get",
        "issubset",
        "items",
        "keys",
        "pop",
        "upper",
        "values",
    }
    assert forbidden.isdisjoint(name_calls)
    assert forbidden.isdisjoint(attribute_calls)


def test_source_identity_expected_assignment_is_literal_dict() -> None:
    tree = ast.parse(Path(contract.__file__).read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SOURCE_IDENTITY_EXPECTED"
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    assert isinstance(assignment.value, ast.Dict)
    assert not [node for node in ast.walk(assignment.value) if isinstance(node, ast.DictComp)]
    assert not [node for node in ast.walk(assignment.value) if isinstance(node, ast.Subscript)]


LOCAL_VALIDITY_BY_SECTION = {
    "block_p_desktop_exe_packaging_source_inventory_reference": "source_reference_valid",
    "inventory_matrix_summary": "matrix_summary_valid",
    "source_inventory_preservation": "source_preservation_valid",
    "desktop_entrypoint_matrix_rows": "entrypoint_rows_valid",
    "qml_bundle_matrix_rows": "qml_rows_valid",
    "python_dependency_matrix_rows": "dependency_rows_valid",
    "packaging_metadata_matrix_rows": "metadata_rows_valid",
    "existing_preview_packaging_matrix_rows": "preview_rows_valid",
    "artifact_exclusion_policy_matrix_rows": "policy_rows_valid",
    "inventory_finding_matrix_rows": "finding_rows_valid",
    "packaging_scope_matrix_rows": "scope_rows_valid",
    "packaging_requirement_matrix_rows": "requirement_rows_valid",
    "unresolved_contract_blocker_rows": "blocker_rows_valid",
    "real_capability_matrix_state": "real_capability_valid",
    "fail_closed_matrix_decision": "fail_closed_valid",
    "non_execution_matrix_evidence": "evidence_valid",
    "matrix_boundaries": "matrix_boundaries_valid",
    "source_boundaries": "source_boundaries_valid",
    "future_steps": "future_steps_valid",
}


def _blocked_from_source(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        contract, "build_preview_block_p_desktop_exe_packaging_inventory_matrix", lambda: source
    )
    payload = contract.build_preview_block_p_desktop_exe_packaging_contract()
    assert payload["packaging_contract_artifact_complete"] is False
    assert payload["ready_for_block_p_4"] is False
    assert payload["status"] == contract.BLOCKED_STATUS
    _assert_plain(payload)
    return payload


class DictSubclass(dict[str, Any]):
    pass


class ListSubclass(list[Any]):
    pass


def _deep_value() -> list[Any]:
    value: Any = []
    for _ in range(1500):
        value = [value]
    return value


def _cycle_value(expected: Any) -> Any:
    if type(expected) is dict:
        value: dict[str, Any] = {}
        value["self"] = value
        return value
    value_list: list[Any] = []
    value_list.append(value_list)
    return value_list


def _wrong_type(expected: Any) -> Any:
    return [] if type(expected) is dict else {}


def _empty_container(expected: Any) -> Any:
    return {} if type(expected) is dict else []


def _subclass_container(expected: Any) -> Any:
    return DictSubclass(expected) if type(expected) is dict else ListSubclass(expected)


@pytest.mark.parametrize("section, validity_key", LOCAL_VALIDITY_BY_SECTION.items())
@pytest.mark.parametrize("variant", ["wrong", "empty", "cycle", "deep", "subclass"])
def test_section_local_isolation_matrix(
    monkeypatch: pytest.MonkeyPatch, section: str, validity_key: str, variant: str
) -> None:
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    expected = contract.EXPECTED_SOURCE[section]
    if variant == "wrong":
        source[section] = _wrong_type(expected)
    elif variant == "empty":
        source[section] = _empty_container(expected)
    elif variant == "cycle":
        source[section] = _cycle_value(expected)
    elif variant == "deep":
        source[section] = _deep_value()
    else:
        source[section] = _subclass_container(expected)
    payload = _blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_contract_evidence"]
    assert evidence[validity_key] is False
    for other_section, other_key in LOCAL_VALIDITY_BY_SECTION.items():
        if other_section != section:
            assert evidence[other_key] is True
    if section != "desktop_entrypoint_matrix_rows":
        assert payload["desktop_entrypoint_contract"]["candidate_paths"] == [
            "ui/pyside_app/__main__.py",
            "ui/pyside_app/app.py",
        ]
    if section != "qml_bundle_matrix_rows":
        assert payload["qml_bundle_contract"]["pyside_qml_file_count"] == 24
        assert payload["qml_bundle_contract"]["shared_qml_file_count"] == 107
    if section != "python_dependency_matrix_rows":
        assert payload["python_dependency_contract"]["declared_dependency_count"] == 25
    if section != "existing_preview_packaging_matrix_rows":
        assert (
            payload["preview_packaging_separation_contract"]["cli_preview_entrypoint"]
            == "scripts/run_local_bot.py"
        )
    if section != "artifact_exclusion_policy_matrix_rows":
        assert (
            payload["artifact_exclusion_contract"]["policy_source"]
            == "scripts/safe_exe_preview_build_plan.py"
        )
    assert (
        payload["source_boundaries"]["source_block_p_desktop_exe_packaging_inventory_matrix"]
        == "FUNCTIONAL-PREVIEW-18.2"
    )


def test_full_scalar_reference_and_invalid_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = _payload()
    ref = payload["block_p_desktop_exe_packaging_inventory_matrix_reference"]
    assert list(ref)[:12] == [
        "schema_version",
        "block_p_desktop_exe_packaging_inventory_matrix_kind",
        "block",
        "step",
        "block_p_desktop_exe_packaging_inventory_matrix_status",
        "block_p_desktop_exe_packaging_inventory_matrix_decision",
        "inventory_matrix_artifact_complete",
        "ready_for_block_p_3",
        "next_step",
        "next_step_title",
        "status",
        "source_identity_valid",
    ]
    for key, value in contract.SOURCE_IDENTITY_EXPECTED.items():
        assert ref[key] == value
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    source["schema_version"] = "wrong"
    blocked = _blocked_from_source(monkeypatch, source)
    blocked_ref = blocked["block_p_desktop_exe_packaging_inventory_matrix_reference"]
    assert blocked_ref["source_identity_valid"] is False
    assert blocked_ref["schema_version"] == ""
    assert blocked_ref["inventory_matrix_artifact_complete"] is False


def test_source_boundaries_and_artifact_policy_are_exact() -> None:
    payload = _payload()
    assert payload["source_boundaries"] == {
        "source_block_p_desktop_exe_packaging_inventory_matrix": "FUNCTIONAL-PREVIEW-18.2",
        "inventory_matrix_preserved": True,
        "can_build_desktop_exe_packaging_contract": True,
        "packaging_contract_artifact_complete": True,
        "can_build_desktop_exe_packaging_read_model": True,
        "can_feed_18_4": True,
    }
    artifact = payload["artifact_exclusion_contract"]
    assert artifact["policy_source"] == "scripts/safe_exe_preview_build_plan.py"
    assert artifact["policy_version"] == "security_packaging_artifact_policy.v1"
    assert artifact["denied_patterns"] == contract.DENIED_ARTIFACT_PATTERNS_18_3
    artifact["denied_patterns"].append("mutated")
    assert contract.DENIED_ARTIFACT_PATTERNS_18_3[-1] == "*keychain*"


def test_acceptance_rule_links_are_meaningful_and_ordered() -> None:
    rows = _payload()["contract_acceptance_rule_rows"]
    assert [row["acceptance_rule_id"] for row in rows] == list(contract.ACCEPTANCE_RULE_LINKS)
    for row in rows:
        links = contract.ACCEPTANCE_RULE_LINKS[row["acceptance_rule_id"]]
        assert row["required_blocker_ids"] == links["blockers"]
        assert row["required_evidence_ids"] == links["evidence"]
    assert rows[0]["required_evidence_ids"] == []
    assert rows[1]["required_blocker_ids"] == []


def test_evidence_artifacts_are_concrete_unique_and_ordered() -> None:
    rows = _payload()["contract_evidence_requirement_rows"]
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        expected = contract.EVIDENCE_ARTIFACTS[row["blocker_id"]]
        assert row["required_artifacts"] == expected
        assert "future_validation_result" not in row["required_artifacts"]
        assert not any(item.startswith("future_record_for_") for item in row["required_artifacts"])
        artifact_tuple = tuple(row["required_artifacts"])
        assert artifact_tuple not in seen
        seen.add(artifact_tuple)


def test_non_vacuous_capability_claims_and_invalid_source_caps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    caps = _payload()["real_capability_contract_state"]
    for claim in [
        "inherited_block_o_capabilities_known_blocked",
        "inherited_block_p_capabilities_known_blocked",
        "source_inventory_capabilities_known_blocked",
        "inventory_matrix_capabilities_known_blocked",
        "packaging_contract_capabilities_known_blocked",
        "all_real_capabilities_blocked_at_18_3",
    ]:
        assert caps[claim] is True
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    source["real_capability_matrix_state"] = {}
    blocked = _blocked_from_source(monkeypatch, source)
    blocked_caps = blocked["real_capability_contract_state"]
    assert blocked_caps["inherited_block_o_capabilities"] == {}
    assert blocked_caps["inherited_block_p_capabilities"] == {}
    assert blocked_caps["source_inventory_capabilities"] == {}
    assert blocked_caps["inventory_matrix_capabilities"] == {}
    assert blocked_caps["inherited_block_o_capabilities_known_blocked"] is False
    assert blocked_caps["inherited_block_p_capabilities_known_blocked"] is False
    assert blocked_caps["source_inventory_capabilities_known_blocked"] is False
    assert blocked_caps["inventory_matrix_capabilities_known_blocked"] is False
    assert blocked_caps["packaging_contract_capabilities_known_blocked"] is True
    assert blocked_caps["all_real_capabilities_blocked_at_18_3"] is False


class BombValue:
    equality_calls = 0
    armed = False

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("custom value equality called")
        return False


class LyingValue(BombValue):
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


class Lying(Bomb):
    equality_calls = 0
    __hash__ = Bomb.__hash__

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


@pytest.mark.parametrize("value_type", [BombValue, LyingValue])
@pytest.mark.parametrize("field", list(contract.SOURCE_IDENTITY_EXPECTED))
def test_custom_identity_values_do_not_call_equality(
    monkeypatch: pytest.MonkeyPatch, value_type: type[BombValue], field: str
) -> None:
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    custom = value_type()
    source[field] = custom
    before_keys = list(source.keys())
    value_type.armed = True
    try:
        payload = _blocked_from_source(monkeypatch, source)
    finally:
        value_type.armed = False
    assert value_type.equality_calls == 0
    assert source[field] is custom
    assert list(source.keys()) == before_keys
    assert payload["non_execution_contract_evidence"]["identity_valid"] is False


def _custom_first_section(
    source: dict[str, Any], section_name: str, target: str, key_type: type[Bomb]
) -> tuple[dict[Any, Any], Bomb, int, list[Any], list[Any]]:
    original = source if section_name == "root" else source[section_name]
    original_key_count = len(original)
    target_present = target in original
    custom_key = key_type(target)
    items = list(original.items())
    without = [(key, value) for key, value in items if key != target]
    ordinary_items = without[:-1] if target_present else without[:-2]
    section: dict[Any, Any] = {custom_key: "shadow"}
    for key, value in ordinary_items:
        section[key] = value
    if target_present and type(original[target]) is bool:
        section[target] = not original[target]
    elif target_present and type(original[target]) is str:
        section[target] = original[target] + "_shadow"
    else:
        section[target] = True
    assert len(section) == original_key_count
    assert next(iter(section)) is custom_key
    assert list(section.keys())[-1] == target
    if section_name == "root":
        source.clear()
        source.update(section)
    else:
        source[section_name] = section  # type: ignore[assignment]
    return (
        section,
        custom_key,
        original_key_count,
        list(section.keys()),
        [
            contract._copy_plain(v)
            for v in section.values()
            if type(v) in (dict, list, str, int, bool) or v is None
        ],
    )


@pytest.mark.parametrize("key_type", [Bomb, Lying])
@pytest.mark.parametrize(
    "section_name,target",
    [("root", "schema_version"), ("root", "inventory_matrix_summary")]
    + [("inventory_matrix_summary", key) for key in contract.SUMMARY_OWNED_FIELDS_18_3]
    + [("fail_closed_matrix_decision", key) for key in contract.FAIL_CLOSED_OWNED_FIELDS_18_3]
    + [("source_boundaries", key) for key in contract.SOURCE_BOUNDARY_FIELDS_18_3],
)
def test_custom_first_keys_do_not_call_equality_or_mutate(
    monkeypatch: pytest.MonkeyPatch, key_type: type[Bomb], section_name: str, target: str
) -> None:
    direct_source = contract._copy_plain(contract.EXPECTED_SOURCE)
    direct_section, direct_custom_key, direct_count, direct_keys, direct_values = (
        _custom_first_section(direct_source, section_name, target, key_type)
    )
    assert len(direct_section) == direct_count
    assert next(iter(direct_section)) is direct_custom_key
    assert list(direct_section.keys())[-1] == target
    key_type.armed = True
    try:
        direct = contract._no_shadowing(direct_source)
    finally:
        key_type.armed = False
    assert key_type.equality_calls == 0
    if section_name != "root":
        assert direct is False
    assert len(direct_section) == direct_count
    assert list(direct_section.keys()) == direct_keys
    assert [
        contract._copy_plain(v)
        for v in direct_section.values()
        if type(v) in (dict, list, str, int, bool) or v is None
    ] == direct_values

    builder_source = contract._copy_plain(contract.EXPECTED_SOURCE)
    builder_section, builder_custom_key, builder_count, builder_keys, builder_values = (
        _custom_first_section(builder_source, section_name, target, key_type)
    )
    key_type.equality_calls = 0
    key_type.armed = True
    try:
        payload = _blocked_from_source(monkeypatch, builder_source)
    finally:
        key_type.armed = False
    assert key_type.equality_calls == 0
    assert len(builder_section) == builder_count
    assert next(iter(builder_section)) is builder_custom_key
    assert list(builder_section.keys()) == builder_keys
    assert [
        contract._copy_plain(v)
        for v in builder_section.values()
        if type(v) in (dict, list, str, int, bool) or v is None
    ] == builder_values
    assert payload["packaging_contract_artifact_complete"] is False


def test_mutation_sensitivity_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    cases = []
    nested_list = contract._copy_plain(contract.EXPECTED_SOURCE)
    nested_list["desktop_entrypoint_matrix_rows"][0]["path"] = "mutated"
    cases.append((nested_list, "entrypoint_rows_valid"))
    nested_dict = contract._copy_plain(contract.EXPECTED_SOURCE)
    nested_dict["inventory_matrix_summary"]["build_ready"] = True
    cases.append((nested_dict, "matrix_summary_valid"))
    scalar = contract._copy_plain(contract.EXPECTED_SOURCE)
    scalar["status"] = "mutated"
    cases.append((scalar, "identity_valid"))
    reordered = contract._copy_plain(contract.EXPECTED_SOURCE)
    reordered["future_steps"] = list(reversed(reordered["future_steps"]))
    cases.append((reordered, "future_steps_valid"))
    deleted = contract._copy_plain(contract.EXPECTED_SOURCE)
    deleted["inventory_matrix_summary"].pop("build_ready")
    cases.append((deleted, "matrix_summary_valid"))
    added = contract._copy_plain(contract.EXPECTED_SOURCE)
    added["inventory_matrix_summary"]["extra"] = False
    cases.append((added, "matrix_summary_valid"))
    owned = contract._copy_plain(contract.EXPECTED_SOURCE)
    owned["source_boundaries"]["can_build_desktop_exe_packaging_contract"] = False
    cases.append((owned, "source_boundaries_valid"))
    custom = contract._copy_plain(contract.EXPECTED_SOURCE)
    custom[Bomb("schema_version")] = "shadow"  # type: ignore[index]
    cases.append((custom, "all_top_level_keys_exact_str"))
    for source, validity_key in cases:
        payload = _blocked_from_source(monkeypatch, source)
        assert payload["non_execution_contract_evidence"][validity_key] is False


def _assert_referential_integrity(payload: dict[str, Any]) -> None:
    scopes = payload["packaging_scope_contract_rows"]
    requirements = payload["packaging_requirement_contract_rows"]
    blockers = payload["unresolved_blocker_contract_rows"]
    evidence = payload["contract_evidence_requirement_rows"]
    acceptance = payload["contract_acceptance_rule_rows"]
    scope_ids = [row["scope_id"] for row in scopes]
    requirement_ids = [row["requirement_id"] for row in requirements]
    blocker_ids = [row["blocker_id"] for row in blockers]
    clause_ids = [row["contract_clause_id"] for row in blockers]
    evidence_ids = [row["evidence_id"] for row in evidence]
    acceptance_ids = [row["acceptance_rule_id"] for row in acceptance]
    for ids in [scope_ids, requirement_ids, blocker_ids, clause_ids, evidence_ids, acceptance_ids]:
        assert all(type(item) is str and item for item in ids)
        assert len(ids) == len(set(ids))
    source_finding_ids = {
        row["finding_id"] for row in contract.EXPECTED_SOURCE["inventory_finding_matrix_rows"]
    }
    for blocker in blockers:
        assert set(blocker["source_finding_ids"]).issubset(source_finding_ids)
        assert set(blocker["source_affected_scope_ids"]).issubset(scope_ids)
        assert len(blocker["required_evidence_ids"]) == 1
        assert blocker["required_evidence_ids"][0] in evidence_ids
    for scope in scopes:
        assert set(scope["source_unresolved_blocker_ids"]).issubset(blocker_ids)
        assert set(scope["contract_clause_ids"]).issubset(clause_ids)
        assert set(scope["required_evidence_ids"]).issubset(evidence_ids)
    for requirement in requirements:
        assert set(requirement["source_unresolved_condition_ids"]).issubset(blocker_ids)
        assert set(requirement["contract_clause_ids"]).issubset(clause_ids)
        assert set(requirement["required_evidence_ids"]).issubset(evidence_ids)
    for rule in acceptance:
        assert set(rule["required_blocker_ids"]).issubset(blocker_ids)
        assert set(rule["required_evidence_ids"]).issubset(evidence_ids)


def test_referential_integrity_nominal_and_blocked_sections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_referential_integrity(_payload())
    for section in LOCAL_VALIDITY_BY_SECTION:
        source = contract._copy_plain(contract.EXPECTED_SOURCE)
        source[section] = _wrong_type(contract.EXPECTED_SOURCE[section])
        _assert_referential_integrity(_blocked_from_source(monkeypatch, source))


def test_no_aliasing_constants_and_link_lists() -> None:
    payload = _payload()
    other = _payload()
    assert len(_mutable_ids(payload)) == len(set(_mutable_ids(payload)))
    payload["artifact_exclusion_contract"]["denied_patterns"].append("mutated")
    assert (
        other["artifact_exclusion_contract"]["denied_patterns"]
        == contract.DENIED_ARTIFACT_PATTERNS_18_3
    )
    assert contract.DENIED_ARTIFACT_PATTERNS_18_3[-1] == "*keychain*"
    payload["unresolved_blocker_contract_rows"][0]["required_evidence_ids"].append("mutated")
    assert other["unresolved_blocker_contract_rows"][0]["required_evidence_ids"] == [
        "evidence_final_desktop_entrypoint_selection"
    ]
    payload["contract_evidence_requirement_rows"][0]["required_artifacts"].append("mutated")
    assert (
        other["contract_evidence_requirement_rows"][0]["required_artifacts"]
        == contract.EVIDENCE_ARTIFACTS["final_desktop_entrypoint_not_selected"]
    )
    payload["contract_acceptance_rule_rows"][0]["required_blocker_ids"].append("mutated")
    assert other["contract_acceptance_rule_rows"][0]["required_blocker_ids"] == list(
        contract.BLOCKER_TO_EVIDENCE
    )


def _valid_source_matrix_ids_for_payload(
    source: dict[str, Any], payload: dict[str, Any]
) -> set[str]:
    evidence = payload["non_execution_contract_evidence"]
    section_pairs = [
        ("desktop_entrypoint_matrix_rows", "entrypoint_rows_valid"),
        ("qml_bundle_matrix_rows", "qml_rows_valid"),
        ("python_dependency_matrix_rows", "dependency_rows_valid"),
        ("packaging_metadata_matrix_rows", "metadata_rows_valid"),
        ("existing_preview_packaging_matrix_rows", "preview_rows_valid"),
        ("artifact_exclusion_policy_matrix_rows", "policy_rows_valid"),
    ]
    row_ids: set[str] = set()
    for section, validity in section_pairs:
        if evidence[validity]:
            row_ids.update(row["matrix_row_id"] for row in source[section])
    return row_ids


def _assert_dependency_aware_output_integrity(
    payload: dict[str, Any], source: dict[str, Any]
) -> None:
    emitted_contract_requirement_ids = [
        row["contract_requirement_id"] for row in payload["packaging_requirement_contract_rows"]
    ]
    assert all(type(row_id) is str and row_id for row_id in emitted_contract_requirement_ids)
    assert len(emitted_contract_requirement_ids) == len(set(emitted_contract_requirement_ids))
    emitted_contract_requirement_id_set = set(emitted_contract_requirement_ids)
    emitted_blocker_ids = {row["blocker_id"] for row in payload["unresolved_blocker_contract_rows"]}
    emitted_clause_ids = {
        row["contract_clause_id"] for row in payload["unresolved_blocker_contract_rows"]
    }
    emitted_evidence_ids = {
        row["evidence_id"] for row in payload["contract_evidence_requirement_rows"]
    }
    available_source_row_ids = _valid_source_matrix_ids_for_payload(source, payload)
    for scope in payload["packaging_scope_contract_rows"]:
        assert set(scope["source_supporting_matrix_row_ids"]).issubset(available_source_row_ids)
        assert set(scope["source_unresolved_blocker_ids"]).issubset(emitted_blocker_ids)
        assert set(scope["contract_clause_ids"]).issubset(emitted_clause_ids)
        assert set(scope["required_evidence_ids"]).issubset(emitted_evidence_ids)
    for requirement in payload["packaging_requirement_contract_rows"]:
        assert set(requirement["contract_clause_ids"]).issubset(emitted_clause_ids)
        assert set(requirement["required_evidence_ids"]).issubset(emitted_evidence_ids)
    for blocker in payload["unresolved_blocker_contract_rows"]:
        assert len(blocker["contract_requirement_ids"]) == len(
            set(blocker["contract_requirement_ids"])
        )
        assert set(blocker["contract_requirement_ids"]).issubset(
            emitted_contract_requirement_id_set
        )
    for rule in payload["contract_acceptance_rule_rows"]:
        assert set(rule["required_blocker_ids"]).issubset(emitted_blocker_ids)
        assert set(rule["required_evidence_ids"]).issubset(emitted_evidence_ids)


SUPPORT_SECTION_CASES = [
    (
        "desktop_entrypoint_matrix_rows",
        "desktop_application_entrypoint",
        [],
        ["qt_qml_runtime_bundle", "windows_exe_artifact_pipeline"],
    ),
    (
        "qml_bundle_matrix_rows",
        "qt_qml_runtime_bundle",
        ["setuptools_ui_package_discovery", "qml_package_data_declaration"],
        ["desktop_application_entrypoint", "windows_exe_artifact_pipeline"],
    ),
    (
        "packaging_metadata_matrix_rows",
        "qt_qml_runtime_bundle",
        [
            "default_qml_entrypoint",
            "pyside_qml_root",
            "shared_qml_root",
            "styles_module",
            "windows_shared_qml_import_path",
        ],
        ["desktop_application_entrypoint", "windows_exe_artifact_pipeline"],
    ),
    (
        "python_dependency_matrix_rows",
        "windows_exe_artifact_pipeline",
        ["safe_exe_preview_build_plan", "windows_preview_profile", "artifact_exclusion_policy"],
        ["desktop_application_entrypoint", "qt_qml_runtime_bundle"],
    ),
    (
        "existing_preview_packaging_matrix_rows",
        "windows_exe_artifact_pipeline",
        [
            "project_dependency_declarations",
            "desktop_optional_dependency_declarations",
            "dependency_resolution",
            "desktop_build_tool_candidates",
            "artifact_exclusion_policy",
        ],
        ["desktop_application_entrypoint", "qt_qml_runtime_bundle"],
    ),
    (
        "artifact_exclusion_policy_matrix_rows",
        "windows_exe_artifact_pipeline",
        [
            "project_dependency_declarations",
            "desktop_optional_dependency_declarations",
            "dependency_resolution",
            "desktop_build_tool_candidates",
            "safe_exe_preview_build_plan",
            "windows_preview_profile",
        ],
        ["desktop_application_entrypoint", "qt_qml_runtime_bundle"],
    ),
]


@pytest.mark.parametrize(
    "section, affected_scope_id, expected_supporting_ids, unaffected_scope_ids",
    SUPPORT_SECTION_CASES,
)
def test_dependency_aware_scope_supporting_links_for_invalid_sections(
    monkeypatch: pytest.MonkeyPatch,
    section: str,
    affected_scope_id: str,
    expected_supporting_ids: list[str],
    unaffected_scope_ids: list[str],
) -> None:
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    source[section] = []
    snapshot = contract._copy_plain(source)
    payload = _blocked_from_source(monkeypatch, source)
    assert source == snapshot
    assert payload["source_matrix_preservation"]["referential_integrity_preserved"] is False
    scopes = {row["scope_id"]: row for row in payload["packaging_scope_contract_rows"]}
    affected = scopes[affected_scope_id]
    assert affected["source_supporting_matrix_row_ids"] == expected_supporting_ids
    assert affected["source_scope_preserved"] is False
    assert affected["contract_definition_complete"] is False
    assert affected["ready_for_read_model"] is False
    assert (
        affected["contract_classification"] == "scope_contract_source_support_not_fully_preserved"
    )
    assert (
        affected["contract_result"]
        == "scope_contract_definition_blocked_by_invalid_supporting_source"
    )
    for scope_id in unaffected_scope_ids:
        assert scopes[scope_id]["source_scope_preserved"] is True
        assert scopes[scope_id]["ready_for_read_model"] is True
    if section == "python_dependency_matrix_rows":
        dependency = payload["python_dependency_contract"]
        assert dependency["declared_dependency_count"] == 0
        assert dependency["optional_desktop_dependency_count"] == 0
        assert dependency["build_tool_candidates"] == []
        assert dependency["contract_defined"] is False
    _assert_dependency_aware_output_integrity(payload, source)


def test_empty_requirements_do_not_create_dangling_blocker_requirement_links(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = contract._copy_plain(contract.EXPECTED_SOURCE)
    source["packaging_requirement_matrix_rows"] = []
    snapshot = contract._copy_plain(source)
    payload = _blocked_from_source(monkeypatch, source)
    assert source == snapshot
    assert payload["packaging_requirement_contract_rows"] == []
    assert payload["unresolved_blocker_contract_rows"]
    assert all(
        row["contract_requirement_ids"] == [] for row in payload["unresolved_blocker_contract_rows"]
    )
    assert payload["source_matrix_preservation"]["referential_integrity_preserved"] is False
    assert payload["desktop_entrypoint_contract"]["contract_defined"] is True
    assert payload["qml_bundle_contract"]["contract_defined"] is True
    assert payload["python_dependency_contract"]["contract_defined"] is True
    assert payload["artifact_exclusion_contract"]["contract_defined"] is True
    _assert_dependency_aware_output_integrity(payload, source)


def test_output_integrity_for_nominal_and_invalid_link_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nominal_source = contract._copy_plain(contract.EXPECTED_SOURCE)
    _assert_dependency_aware_output_integrity(_payload(), nominal_source)
    for section in [
        "desktop_entrypoint_matrix_rows",
        "qml_bundle_matrix_rows",
        "python_dependency_matrix_rows",
        "packaging_metadata_matrix_rows",
        "existing_preview_packaging_matrix_rows",
        "artifact_exclusion_policy_matrix_rows",
        "inventory_finding_matrix_rows",
        "packaging_scope_matrix_rows",
        "packaging_requirement_matrix_rows",
        "unresolved_contract_blocker_rows",
        "real_capability_matrix_state",
    ]:
        source = contract._copy_plain(contract.EXPECTED_SOURCE)
        source[section] = []
        _assert_dependency_aware_output_integrity(_blocked_from_source(monkeypatch, source), source)


EXPECTED_REQUIREMENT_CONTRACT_LINKS = [
    (
        "desktop_application_entrypoint_inventory",
        "contract_desktop_application_entrypoint_inventory",
        [
            "clause_final_desktop_entrypoint_not_selected",
            "clause_desktop_entrypoint_validation_not_performed",
        ],
        [
            "evidence_final_desktop_entrypoint_selection",
            "evidence_desktop_entrypoint_validation",
        ],
    ),
    (
        "qml_asset_inventory",
        "contract_qml_asset_inventory",
        [
            "clause_qml_bundle_validation_not_performed",
            "clause_windows_shared_qml_import_path_unresolved",
            "clause_ui_package_discovery_missing",
            "clause_qml_package_data_missing",
        ],
        [
            "evidence_qml_bundle_validation",
            "evidence_windows_shared_qml_import_path",
            "evidence_ui_package_discovery",
            "evidence_qml_package_data",
        ],
    ),
    (
        "qt_plugin_inventory",
        "contract_qt_plugin_inventory",
        ["clause_qt_plugin_inventory_missing"],
        ["evidence_qt_plugin_inventory"],
    ),
    (
        "python_dependency_inventory",
        "contract_python_dependency_inventory",
        ["clause_dependency_resolution_not_performed"],
        ["evidence_windows_dependency_resolution"],
    ),
    (
        "packaging_profile_alignment",
        "contract_packaging_profile_alignment",
        ["clause_final_desktop_packaging_profile_not_aligned"],
        ["evidence_final_windows_profile_alignment"],
    ),
    (
        "secret_and_local_data_exclusion_policy",
        "contract_secret_and_local_data_exclusion_policy",
        ["clause_secret_and_local_data_exclusion_policy_not_validated"],
        ["evidence_artifact_exclusion_validation"],
    ),
    (
        "windows_target_toolchain_confirmation",
        "contract_windows_target_toolchain_confirmation",
        ["clause_windows_toolchain_not_confirmed"],
        ["evidence_windows_toolchain_confirmation"],
    ),
    (
        "future_explicit_build_execution_gate",
        "contract_future_explicit_build_execution_gate",
        ["clause_future_explicit_build_execution_gate_missing"],
        ["evidence_future_explicit_build_execution_gate"],
    ),
]

EXPECTED_BLOCKER_CONTRACT_REQUIREMENT_LINKS = {
    "final_desktop_entrypoint_not_selected": ["contract_desktop_application_entrypoint_inventory"],
    "desktop_entrypoint_validation_not_performed": [
        "contract_desktop_application_entrypoint_inventory"
    ],
    "qml_bundle_validation_not_performed": ["contract_qml_asset_inventory"],
    "windows_shared_qml_import_path_unresolved": ["contract_qml_asset_inventory"],
    "qt_plugin_inventory_missing": ["contract_qt_plugin_inventory"],
    "ui_package_discovery_missing": ["contract_qml_asset_inventory"],
    "qml_package_data_missing": ["contract_qml_asset_inventory"],
    "final_desktop_packaging_profile_not_aligned": ["contract_packaging_profile_alignment"],
    "dependency_resolution_not_performed": ["contract_python_dependency_inventory"],
    "secret_and_local_data_exclusion_policy_not_validated": [
        "contract_secret_and_local_data_exclusion_policy"
    ],
    "windows_toolchain_not_confirmed": ["contract_windows_target_toolchain_confirmation"],
    "future_explicit_build_execution_gate_missing": [
        "contract_future_explicit_build_execution_gate"
    ],
}


def test_exact_requirement_contract_ids_and_blocker_links() -> None:
    payload = _payload()
    requirement_rows = payload["packaging_requirement_contract_rows"]
    blocker_rows = payload["unresolved_blocker_contract_rows"]
    assert [
        (
            row["requirement_id"],
            row["contract_requirement_id"],
            row["contract_clause_ids"],
            row["required_evidence_ids"],
        )
        for row in requirement_rows
    ] == EXPECTED_REQUIREMENT_CONTRACT_LINKS
    assert [row["contract_requirement_id"] for row in requirement_rows] == [
        "contract_desktop_application_entrypoint_inventory",
        "contract_qml_asset_inventory",
        "contract_qt_plugin_inventory",
        "contract_python_dependency_inventory",
        "contract_packaging_profile_alignment",
        "contract_secret_and_local_data_exclusion_policy",
        "contract_windows_target_toolchain_confirmation",
        "contract_future_explicit_build_execution_gate",
    ]
    assert {
        row["blocker_id"]: row["contract_requirement_ids"] for row in blocker_rows
    } == EXPECTED_BLOCKER_CONTRACT_REQUIREMENT_LINKS
    raw_requirement_ids = {row["requirement_id"] for row in requirement_rows}
    for blocker in blocker_rows:
        assert set(blocker["contract_requirement_ids"]).isdisjoint(raw_requirement_ids)
