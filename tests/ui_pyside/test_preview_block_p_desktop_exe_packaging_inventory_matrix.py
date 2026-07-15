from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app import preview_block_p_desktop_exe_packaging_inventory_matrix as matrix
from ui.pyside_app.preview_block_p_desktop_exe_packaging_source_inventory import (
    build_preview_block_p_desktop_exe_packaging_source_inventory,
)


def _payload() -> dict[str, Any]:
    return matrix.build_preview_block_p_desktop_exe_packaging_inventory_matrix()


def _assert_plain(value: Any) -> None:
    json.dumps(value, sort_keys=True)
    assert matrix._all_plain_json(value, matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH)


def test_expected_source_matches_current_18_1() -> None:
    assert matrix.EXPECTED_SOURCE == build_preview_block_p_desktop_exe_packaging_source_inventory()
    assert list(matrix.EXPECTED_SOURCE) == matrix.TOP_LEVEL_FIELDS_18_1


def test_identity_order_reference_and_json_serializable() -> None:
    payload = _payload()
    assert list(payload) == matrix.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == matrix.SCHEMA_VERSION
    assert payload["block_p_desktop_exe_packaging_inventory_matrix_kind"] == matrix.KIND
    assert payload["block"] == "P"
    assert payload["step"] == "18.2"
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-18.3"
    ref = payload["block_p_desktop_exe_packaging_source_inventory_reference"]
    assert (
        ref["source_block_p_desktop_exe_packaging_source_inventory_step"]
        == "FUNCTIONAL-PREVIEW-18.1"
    )
    assert ref["source_inventory_read_by_18_2"] is True
    _assert_plain(payload)


def test_source_builder_called_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    source = build_preview_block_p_desktop_exe_packaging_source_inventory()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return source

    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_source_inventory", fake
    )
    payload = matrix.build_preview_block_p_desktop_exe_packaging_inventory_matrix()
    assert calls == 1
    assert payload["non_execution_matrix_evidence"]["source_builder_call_count"] == 1


def test_inventory_matrix_artifact_complete_source_only() -> None:
    p = _payload()
    assert p["inventory_matrix_artifact_complete"] is True
    assert p["ready_for_block_p_3"] is True
    assert p["inventory_matrix_summary"]["source_only"] is True
    assert p["matrix_boundaries"]["repo_rescan"] is False


def test_source_inventory_counts_preserved() -> None:
    s = _payload()["source_inventory_preservation"]
    assert s["desktop_entrypoint_row_count"] == 4
    assert s["pyside_qml_file_count"] == 24
    assert s["shared_qml_file_count"] == 107
    assert s["additional_qml_support_asset_count"] == 0
    assert s["deploy_packaging_source_file_count"] == 46
    assert s["deployment_documentation_file_count"] == 17
    assert s["config_reference_row_count"] == 6
    assert s["inventory_finding_count"] == 11


def test_four_desktop_entrypoint_matrix_rows() -> None:
    rows = _payload()["desktop_entrypoint_matrix_rows"]
    assert [r["path"] for r in rows] == [
        "ui/pyside_app/__main__.py",
        "ui/pyside_app/app.py",
        "scripts/run_local_bot.py",
        "scripts/operator_preview_bundle.py",
    ]
    assert len(rows) == 4
    assert rows[0]["desktop_entrypoint_candidate"] is True
    assert rows[1]["selected_as_final_desktop_entrypoint"] is False
    assert rows[2]["excluded_from_final_desktop_contract"] is True


def test_five_qml_bundle_matrix_rows() -> None:
    rows = _payload()["qml_bundle_matrix_rows"]
    assert [r["matrix_row_id"] for r in rows] == [
        "default_qml_entrypoint",
        "pyside_qml_root",
        "shared_qml_root",
        "styles_module",
        "windows_shared_qml_import_path",
    ]
    assert rows[-1]["unresolved_condition_present"] is True
    assert len(rows) == 5


def test_four_python_dependency_matrix_rows() -> None:
    rows = _payload()["python_dependency_matrix_rows"]
    assert len(rows) == 4
    assert rows[2]["matrix_classification"] == "dependency_resolution_not_performed"
    assert rows[3]["selection_performed"] is False


def test_four_packaging_metadata_matrix_rows() -> None:
    rows = _payload()["packaging_metadata_matrix_rows"]
    assert len(rows) == 4
    assert rows[0]["matrix_classification"] == "setuptools_ui_package_discovery_missing"
    assert rows[1]["matrix_classification"] == "qml_package_data_declaration_missing"


def test_four_existing_preview_packaging_matrix_rows() -> None:
    rows = _payload()["existing_preview_packaging_matrix_rows"]
    assert len(rows) == 4
    assert all(r["source_scope"] == "cli_preview" for r in rows)
    assert all(r["targets_final_desktop_entrypoint"] is False for r in rows)


def test_artifact_exclusion_policy_matrix_row() -> None:
    row = _payload()["artifact_exclusion_policy_matrix_rows"][0]
    assert row["matrix_row_id"] == "artifact_exclusion_policy"
    assert row["policy_applied"] is False
    assert row["build_ready"] is False


def test_eleven_inventory_finding_matrix_rows() -> None:
    rows = _payload()["inventory_finding_matrix_rows"]
    assert len(rows) == 11
    assert rows[6]["finding_id"] == "desktop_build_tools_declared_as_optional_dependencies"
    assert rows[6]["contract_blocker_present"] is False
    assert all(r["approved"] is False for r in rows)


def test_three_packaging_scope_matrix_rows() -> None:
    rows = _payload()["packaging_scope_matrix_rows"]
    assert len(rows) == 3
    assert [r["unresolved_condition_count"] for r in rows] == [2, 5, 5]
    assert all(r["failure_policy"] == "fail_closed" for r in rows)


def test_eight_packaging_requirement_matrix_rows() -> None:
    rows = _payload()["packaging_requirement_matrix_rows"]
    assert len(rows) == 8
    assert rows[0]["inventory_requirement_satisfied"] is True
    assert rows[2]["missing_inventory"] is True
    assert all(r["build_authorized"] is False for r in rows)


def test_twelve_unresolved_contract_blocker_rows() -> None:
    rows = _payload()["unresolved_contract_blocker_rows"]
    assert len(rows) == 12
    assert rows[0]["blocker_id"] == "final_desktop_entrypoint_not_selected"
    assert all(r["classification"] == "unresolved_packaging_contract_condition" for r in rows)


def test_inventory_complete_does_not_mean_contract_satisfied() -> None:
    summary = _payload()["inventory_matrix_summary"]
    assert summary["inventory_matrix_artifact_complete"] is True
    assert summary["packaging_contract_conditions_satisfied"] is False
    assert summary["build_ready"] is False


def test_all_real_capabilities_blocked_non_vacuously() -> None:
    caps = _payload()["real_capability_matrix_state"]
    for key in [
        "inherited_block_o_capabilities",
        "inherited_block_p_capabilities",
        "source_inventory_capabilities",
        "inventory_matrix_capabilities",
    ]:
        assert caps[key]
        assert set(caps[key].values()) == {"blocked"}
    assert caps["all_real_capabilities_blocked_at_18_2"] is True


def test_no_approval_build_packaging_release_runtime_orders() -> None:
    p = _payload()
    summary = p["inventory_matrix_summary"]
    for key in [
        "desktop_entrypoint_approved",
        "build_ready",
        "packaging_authorized",
        "build_authorized",
        "artifact_creation_authorized",
        "release_authorized",
        "runtime_authorized",
        "orders_authorized",
    ]:
        assert summary[key] is False
    assert p["fail_closed_matrix_decision"]["build_executed_by_18_2"] is False


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


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    ids = _mutable_ids(_payload())
    assert len(ids) == len(set(ids))


def test_independent_builder_calls_do_not_share_state() -> None:
    one = _payload()
    two = _payload()
    one["desktop_entrypoint_matrix_rows"][0]["path"] = "mutated"
    assert two["desktop_entrypoint_matrix_rows"][0]["path"] == "ui/pyside_app/__main__.py"
    assert (
        matrix.EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"][0]["path"]
        == "ui/pyside_app/__main__.py"
    )


def test_forbidden_raw_tokens_absent() -> None:
    text = Path(matrix.__file__).read_text()
    assert "ccxt" not in text
    assert "create_order" not in text
    assert "fetch_balance" not in text


def test_exact_ast_guard() -> None:
    tree = ast.parse(Path(matrix.__file__).read_text())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert [n.module for n in imports] == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_p_desktop_exe_packaging_source_inventory",
    ]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    name_call_list = [n.func.id for n in calls if isinstance(n.func, ast.Name)]
    name_calls = {n.func.id for n in calls if isinstance(n.func, ast.Name)}
    attribute_calls = {n.func.attr for n in calls if isinstance(n.func, ast.Attribute)}
    assert name_call_list.count("build_preview_block_p_desktop_exe_packaging_source_inventory") == 1
    assert name_calls == {
        "_all_plain_json",
        "_blocker_rows",
        "_copy_plain",
        "_dependency_rows",
        "_entrypoint_rows",
        "_exact_plain_matches",
        "_finding_rows",
        "_future_steps",
        "_matrix_capabilities",
        "_metadata_rows",
        "_no_shadowing",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_preview_rows",
        "_qml_rows",
        "_requirement_rows",
        "_safe_top_level_source",
        "_scalar_reference",
        "_scope_rows",
        "_section_valid",
        "_source_identity_valid",
        "all",
        "bool",
        "build_preview_block_p_desktop_exe_packaging_source_inventory",
        "enumerate",
        "id",
        "len",
        "list",
        "range",
        "set",
        "type",
        "zip",
    }
    assert attribute_calls == {
        "add",
        "append",
        "discard",
        "get",
        "items",
        "keys",
        "pop",
        "upper",
        "values",
    }
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
        "load",
        "loads",
        "safe_load",
        "parse",
        "walk_packages",
        "iter_modules",
        "QGuiApplication",
        "QQmlApplicationEngine",
        "loadFromModule",
        "addImportPath",
        "setImportPathList",
        "PyInstaller",
        "briefcase",
        "package",
        "build",
        "freeze",
        "collect_data_files",
        "collect_submodules",
        "sign",
        "upload",
        "publish",
        "release",
        "exec",
        "start",
        "run_runtime",
        "submit_order",
        "cancel_order",
        "replace_order",
    }
    assert forbidden.isdisjoint(name_calls)
    assert forbidden.isdisjoint(attribute_calls)


@pytest.mark.parametrize("bad", [None, 1, 1.0, object(), set(), tuple()])
def test_malformed_source_non_dict_blocks(monkeypatch: pytest.MonkeyPatch, bad: Any) -> None:
    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_source_inventory", lambda: bad
    )
    payload = matrix.build_preview_block_p_desktop_exe_packaging_inventory_matrix()
    assert payload["inventory_matrix_artifact_complete"] is False
    _assert_plain(payload)


def test_cycle_deep_shared_plain_helpers() -> None:
    cyc: dict[str, Any] = {}
    cyc["x"] = cyc
    assert matrix._all_plain_json(cyc, matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    deep: Any = []
    for _ in range(1500):
        deep = [deep]
    assert matrix._all_plain_json(deep, 2000) is True
    assert matrix._all_plain_json(deep, matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    shared: list[Any] = []
    assert matrix._all_plain_json([shared, shared], matrix.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is True


class BombValue:
    equality_calls = 0
    armed = False

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            raise AssertionError("custom identity equality called")
        return False


class LyingValue(BombValue):
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


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
            raise AssertionError("custom key equality called")
        return False


class Lying(Bomb):
    equality_calls = 0
    __hash__ = Bomb.__hash__

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


def _plain_source() -> dict[str, Any]:
    return matrix._copy_plain(matrix.EXPECTED_SOURCE)


def _blocked_from_source(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    monkeypatch.setattr(
        matrix, "build_preview_block_p_desktop_exe_packaging_source_inventory", lambda: source
    )
    payload = matrix.build_preview_block_p_desktop_exe_packaging_inventory_matrix()
    assert payload["inventory_matrix_artifact_complete"] is False
    _assert_plain(payload)
    return payload


def _deep_plain_values_without_custom(section: dict[Any, Any], custom_key: object) -> list[Any]:
    values = []
    for key, value in section.items():
        if key is custom_key:
            continue
        values.append(matrix._copy_plain(value))
    return values


def _assert_custom_first_not_mutated(
    section: dict[Any, Any],
    section_before: dict[str, Any],
    custom_key: object,
    owned_field: str,
    value_snapshot: list[Any],
) -> None:
    assert section is section_before["object"]
    keys = list(section.keys())
    assert len(keys) == section_before["key_count"]
    assert keys[0] is custom_key
    assert keys[-1] == owned_field
    assert [type(key) for key in keys[1:]] == section_before["key_types"]
    assert keys[1:] == section_before["keys_without_custom"]
    assert len(list(section.values())) == section_before["value_count"]
    assert _deep_plain_values_without_custom(section, custom_key) == value_snapshot


def _custom_first_section(
    source: dict[str, Any], section_name: str, target: str, key_type: type[Bomb]
) -> tuple[dict[Any, Any], Bomb, dict[str, Any], list[Any]]:
    original = source[section_name]
    assert type(original) is dict
    custom_key = key_type(target)
    original_key_count = len(original)
    pairs = list(original.items())
    target_present = target in original
    without_target = [(key, value) for key, value in pairs if key != target]
    protected = set(matrix.SOURCE_IDENTITY_EXPECTED) if section_name == "root" else set()
    removable = [item for item in without_target if item[0] not in protected]
    protected_items = [item for item in without_target if item[0] in protected]
    remove_count = 1 if target_present else 2
    without_target = protected_items + removable[:-remove_count]
    section: dict[Any, Any] = {custom_key: "shadow"}
    for key, value in without_target:
        section[key] = value
    if target_present and type(original[target]) is bool:
        section[target] = not original[target]
    elif target_present and type(original[target]) is str:
        section[target] = original[target] + "_shadow"
    else:
        section[target] = original[target] if target_present else True
    assert len(section) == original_key_count
    assert next(iter(section)) is custom_key
    assert list(section.keys())[-1] == target
    source[section_name] = section  # type: ignore[assignment]
    before = {
        "object": section,
        "original_key_count": original_key_count,
        "key_count": original_key_count,
        "key_types": [type(key) for key in list(section.keys())[1:]],
        "keys_without_custom": list(section.keys())[1:],
        "value_count": len(list(section.values())),
    }
    values = _deep_plain_values_without_custom(section, custom_key)
    return section, custom_key, before, values


def _all_matrix_row_ids(payload: dict[str, Any]) -> set[str]:
    row_ids: set[str] = set()
    for section in [
        "desktop_entrypoint_matrix_rows",
        "qml_bundle_matrix_rows",
        "python_dependency_matrix_rows",
        "packaging_metadata_matrix_rows",
        "existing_preview_packaging_matrix_rows",
        "artifact_exclusion_policy_matrix_rows",
    ]:
        for row in payload[section]:
            row_ids.add(row["matrix_row_id"])
    return row_ids


def _assert_referential_integrity(payload: dict[str, Any]) -> None:
    blocker_rows = payload["unresolved_contract_blocker_rows"]
    blocker_ids = {row["blocker_id"] for row in blocker_rows}
    finding_rows = payload["inventory_finding_matrix_rows"]
    finding_ids = {row["finding_id"] for row in finding_rows}
    scope_rows = payload["packaging_scope_matrix_rows"]
    scope_ids = {row["scope_id"] for row in scope_rows}
    all_matrix_row_ids = _all_matrix_row_ids(payload)
    assert len(blocker_ids) == len(blocker_rows)
    assert "" not in blocker_ids
    for blocker in blocker_rows:
        assert blocker["affected_scope_ids"]
        if finding_rows:
            assert set(blocker["source_finding_ids"]) <= finding_ids
        else:
            assert blocker["source_finding_ids"] == []
        assert set(blocker["affected_scope_ids"]) <= scope_ids
    for scope in scope_rows:
        support = scope["supporting_matrix_row_ids"]
        assert len(support) == len(set(support))
        assert "" not in support
        assert set(support) <= all_matrix_row_ids
        assert set(scope["unresolved_condition_ids"]) <= blocker_ids
        for blocker_id in scope["unresolved_condition_ids"]:
            blocker = next(row for row in blocker_rows if row["blocker_id"] == blocker_id)
            assert scope["scope_id"] in blocker["affected_scope_ids"]
    for requirement in payload["packaging_requirement_matrix_rows"]:
        assert requirement["unresolved_condition_ids"]
        assert set(requirement["unresolved_condition_ids"]) <= blocker_ids


def test_styles_module_paths_come_from_18_1_source() -> None:
    rows = _payload()["qml_bundle_matrix_rows"]
    assert rows[3]["source_paths"] == [
        "ui/pyside_app/qml/Styles/qmldir",
        "ui/pyside_app/qml/Styles/DesignSystem.qml",
    ]


def test_artifact_policy_source_and_version_come_from_valid_source() -> None:
    row = _payload()["artifact_exclusion_policy_matrix_rows"][0]
    assert row["policy_source"] == "scripts/safe_exe_preview_build_plan.py"
    assert row["policy_version"] == "security_packaging_artifact_policy.v1"


def test_blocker_scope_requirement_finding_referential_integrity() -> None:
    payload = _payload()
    _assert_referential_integrity(payload)
    blockers = {row["blocker_id"]: row for row in payload["unresolved_contract_blocker_rows"]}
    assert blockers["final_desktop_entrypoint_not_selected"]["source_finding_ids"] == [
        "desktop_module_launcher_observed",
        "desktop_application_main_observed",
    ]
    assert blockers["qt_plugin_inventory_missing"]["source_finding_ids"] == []
    assert blockers["future_explicit_build_execution_gate_missing"]["source_finding_ids"] == []


def test_scope_and_requirement_unresolved_ids_are_exact_blockers() -> None:
    payload = _payload()
    blocker_ids = {row["blocker_id"] for row in payload["unresolved_contract_blocker_rows"]}
    for row in payload["packaging_scope_matrix_rows"]:
        assert set(row["unresolved_condition_ids"]) <= blocker_ids
    for row in payload["packaging_requirement_matrix_rows"]:
        assert row["unresolved_condition_ids"]
        assert set(row["unresolved_condition_ids"]) <= blocker_ids
    windows = payload["packaging_scope_matrix_rows"][2]
    assert "artifact_exclusion_policy_not_validated" not in windows["unresolved_condition_ids"]
    assert (
        "secret_and_local_data_exclusion_policy_not_validated"
        in windows["unresolved_condition_ids"]
    )


def test_metadata_preview_local_valid_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _plain_source()
    source["packaging_metadata_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_matrix_evidence"]
    preservation = payload["source_inventory_preservation"]
    assert evidence["packaging_metadata_valid"] is False
    assert evidence["preview_packaging_valid"] is True
    assert preservation["deploy_packaging_source_file_count"] == 0
    assert preservation["deployment_documentation_file_count"] == 0
    assert preservation["cli_preview_remains_separate"] is True

    source = _plain_source()
    source["existing_cli_preview_packaging_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_matrix_evidence"]
    preservation = payload["source_inventory_preservation"]
    assert evidence["packaging_metadata_valid"] is True
    assert evidence["preview_packaging_valid"] is False
    assert preservation["deploy_packaging_source_file_count"] == 46
    assert preservation["deployment_documentation_file_count"] == 17
    assert preservation["cli_preview_remains_separate"] is False


@pytest.mark.parametrize(
    "section",
    ["source_inventory_summary", "fail_closed_inventory_decision", "inventory_boundaries"],
)
def test_preserved_approval_and_build_claims_require_local_valid_sections(
    monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    source = _plain_source()
    source[section] = {}
    payload = _blocked_from_source(monkeypatch, source)
    preservation = payload["source_inventory_preservation"]
    assert preservation["all_source_approvals_false"] is (section == "inventory_boundaries")
    assert preservation["all_source_build_runtime_order_flags_false"] is False


def test_source_identity_expected_is_literal_dict() -> None:
    tree = ast.parse(Path(matrix.__file__).read_text())
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "SOURCE_IDENTITY_EXPECTED"
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Dict)
    assert not isinstance(assignments[0].value, ast.DictComp)


@pytest.mark.parametrize("value_type", [BombValue, LyingValue])
@pytest.mark.parametrize("field", list(matrix.SOURCE_IDENTITY_EXPECTED))
def test_per_field_custom_identity_values_do_not_call_equality(
    monkeypatch: pytest.MonkeyPatch, value_type: type[BombValue], field: str
) -> None:
    source = _plain_source()
    custom = value_type()
    source[field] = custom
    key_order = list(source.keys())
    remaining = {key: matrix._copy_plain(value) for key, value in source.items() if key != field}
    value_type.equality_calls = 0
    value_type.armed = True
    try:
        payload = _blocked_from_source(monkeypatch, source)
    finally:
        value_type.armed = False
    assert value_type.equality_calls == 0
    assert source[field] is custom
    assert list(source.keys()) == key_order
    assert {key: value for key, value in source.items() if key != field} == remaining
    _assert_plain(payload)


def _custom_key_cases() -> list[tuple[str, str, str]]:
    return [
        ("top", "", "schema_version"),
        ("top", "", "source_inventory_summary"),
        *[
            ("section", "source_inventory_summary", field)
            for field in matrix.SUMMARY_OWNED_FIELDS_18_2
        ],
        *[
            ("section", "fail_closed_inventory_decision", field)
            for field in matrix.FAIL_CLOSED_OWNED_FIELDS_18_2
        ],
        *[("section", "source_boundaries", field) for field in matrix.SOURCE_BOUNDARY_FIELDS_18_2],
    ]


@pytest.mark.parametrize("key_type", [Bomb, Lying])
@pytest.mark.parametrize(("level", "section_name", "target"), _custom_key_cases())
def test_custom_first_direct_and_builder_probes_do_not_mutate(
    monkeypatch: pytest.MonkeyPatch,
    key_type: type[Bomb],
    level: str,
    section_name: str,
    target: str,
) -> None:
    direct_source = _plain_source()
    if level == "top":
        section, custom_key, before, values = _custom_first_section(
            {"root": direct_source}, "root", target, key_type
        )
        direct_source = section  # type: ignore[assignment]
    else:
        section, custom_key, before, values = _custom_first_section(
            direct_source, section_name, target, key_type
        )
    assert len(section) == before["original_key_count"]
    key_type.equality_calls = 0
    key_type.armed = True
    try:
        if level == "top":
            _, all_keys_exact = matrix._safe_top_level_source(direct_source)  # type: ignore[arg-type]
            assert all_keys_exact is False
        else:
            assert matrix._no_shadowing(direct_source) is False
    finally:
        key_type.armed = False
    assert key_type.equality_calls == 0
    _assert_custom_first_not_mutated(section, before, custom_key, target, values)

    builder_source = _plain_source()
    if level == "top":
        section, custom_key, before, values = _custom_first_section(
            {"root": builder_source}, "root", target, key_type
        )
        builder_source = section  # type: ignore[assignment]
    else:
        section, custom_key, before, values = _custom_first_section(
            builder_source, section_name, target, key_type
        )
    assert len(section) == before["original_key_count"]
    key_type.equality_calls = 0
    key_type.armed = True
    try:
        payload = _blocked_from_source(monkeypatch, builder_source)
    finally:
        key_type.armed = False
    assert key_type.equality_calls == 0
    if level == "top" and target == "schema_version":
        assert payload["non_execution_matrix_evidence"]["identity_valid"] is False
    else:
        assert payload["non_execution_matrix_evidence"]["identity_valid"] is True
    assert payload["non_execution_matrix_evidence"]["entrypoint_rows_valid"] is True
    _assert_custom_first_not_mutated(section, before, custom_key, target, values)


def test_no_mutation_assertion_helper_is_sensitive() -> None:
    source = _plain_source()
    section, custom_key, before, values = _custom_first_section(
        source, "source_inventory_summary", "inventory_matrix_artifact_complete", Bomb
    )
    _assert_custom_first_not_mutated(
        section, before, custom_key, "inventory_matrix_artifact_complete", values
    )
    mutations = []
    mutations.append(lambda s: s.setdefault("nested", []).append("x"))
    mutations.append(lambda s: s.setdefault("nested_dict", {}).__setitem__("x", 1))
    mutations.append(lambda s: s.__setitem__("source_only", False))
    mutations.append(lambda s: s.update({k: s.pop(k) for k in [next(iter(s.keys()))]}))
    mutations.append(lambda s: s.pop("source_only"))
    mutations.append(lambda s: s.__setitem__("added", True))
    mutations.append(lambda s: s.__setitem__("inventory_matrix_artifact_complete", False))
    mutations.append(
        lambda s: s.__setitem__(Bomb("inventory_matrix_artifact_complete"), s.pop(custom_key))
    )
    for mutate in mutations:
        source = _plain_source()
        section, custom_key, before, values = _custom_first_section(
            source, "source_inventory_summary", "inventory_matrix_artifact_complete", Bomb
        )
        mutate(section)
        with pytest.raises(AssertionError):
            _assert_custom_first_not_mutated(
                section, before, custom_key, "inventory_matrix_artifact_complete", values
            )


def _malformed_cases() -> list[tuple[str, Any, str]]:
    class DictSubclass(dict):
        pass

    class ListSubclass(list):
        pass

    cases: list[tuple[str, Any, str]] = []
    cases.extend(
        (name, value, "identity_valid")
        for name, value in [
            ("source non-dict", None),
            ("float", 1.0),
            ("object", object()),
            ("callable", lambda: None),
            ("set", set()),
            ("tuple", tuple()),
        ]
    )
    source = _plain_source()
    cases.append(("top-level dict subclass", DictSubclass(source), "identity_valid"))
    for name, mutator in [
        ("top-level extra", lambda s: s.__setitem__("extra", True)),
        ("top-level missing", lambda s: s.pop("status")),
        ("top-level reordered", lambda s: s.update({"schema_version": s.pop("schema_version")})),
        ("status mutation", lambda s: s.__setitem__("status", "bad")),
        (
            "decision mutation",
            lambda s: s.__setitem__(
                "block_p_desktop_exe_packaging_source_inventory_decision", "bad"
            ),
        ),
        ("bool/int bypass", lambda s: s.__setitem__("ready_for_block_p_2", 1)),
        ("int/float bypass", lambda s: s.__setitem__("ready_for_block_p_2", 1.0)),
        ("custom top-level key", lambda s: s.__setitem__(Bomb("schema_version"), "x")),
    ]:
        source = _plain_source()
        mutator(source)
        cases.append(
            (
                name,
                source,
                ""
                if name in {"top-level extra", "top-level reordered", "custom top-level key"}
                else "identity_valid",
            )
        )
    section_cases = [
        (
            "section dict subclass",
            "source_inventory_summary",
            DictSubclass(matrix.EXPECTED_SOURCE["source_inventory_summary"]),
            "summary_valid",
        ),
        (
            "section list subclass",
            "desktop_entrypoint_inventory_rows",
            ListSubclass(matrix.EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"]),
            "entrypoint_rows_valid",
        ),
        (
            "section extra",
            "source_inventory_summary",
            {**matrix.EXPECTED_SOURCE["source_inventory_summary"], "x": True},
            "summary_valid",
        ),
        ("section missing", "source_inventory_summary", {}, "summary_valid"),
        (
            "section reordered",
            "source_inventory_summary",
            {
                k: matrix.EXPECTED_SOURCE["source_inventory_summary"][k]
                for k in reversed(matrix.EXPECTED_SOURCE["source_inventory_summary"])
            },
            "summary_valid",
        ),
        ("row non-dict", "desktop_entrypoint_inventory_rows", ["bad"], "entrypoint_rows_valid"),
        (
            "row dict subclass",
            "desktop_entrypoint_inventory_rows",
            [DictSubclass(matrix.EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"][0])],
            "entrypoint_rows_valid",
        ),
        ("row count", "desktop_entrypoint_inventory_rows", [], "entrypoint_rows_valid"),
        (
            "row order",
            "desktop_entrypoint_inventory_rows",
            list(reversed(matrix.EXPECTED_SOURCE["desktop_entrypoint_inventory_rows"])),
            "entrypoint_rows_valid",
        ),
        (
            "changed QML count",
            "qml_source_inventory",
            {**matrix.EXPECTED_SOURCE["qml_source_inventory"], "pyside_qml_source_file_count": 25},
            "qml_inventory_valid",
        ),
        (
            "changed QML path",
            "qml_source_inventory",
            {**matrix.EXPECTED_SOURCE["qml_source_inventory"], "default_qml_entrypoint": "bad"},
            "qml_inventory_valid",
        ),
        (
            "changed dependency spec",
            "python_dependency_inventory",
            {
                **matrix.EXPECTED_SOURCE["python_dependency_inventory"],
                "project_dependency_specs": ["bad"],
            },
            "python_dependency_inventory_valid",
        ),
        (
            "changed deployment path",
            "packaging_metadata_inventory",
            {
                **matrix.EXPECTED_SOURCE["packaging_metadata_inventory"],
                "deploy_packaging_source_files": ["bad"],
            },
            "packaging_metadata_valid",
        ),
        (
            "changed finding source path",
            "inventory_findings",
            [{**matrix.EXPECTED_SOURCE["inventory_findings"][0], "source_paths": ["bad"]}],
            "inventory_findings_valid",
        ),
        (
            "changed source approval",
            "source_inventory_summary",
            {
                **matrix.EXPECTED_SOURCE["source_inventory_summary"],
                "desktop_entrypoint_approved": True,
            },
            "summary_valid",
        ),
        (
            "changed build flag",
            "fail_closed_inventory_decision",
            {
                **matrix.EXPECTED_SOURCE["fail_closed_inventory_decision"],
                "build_executed_by_18_1": True,
            },
            "fail_closed_valid",
        ),
        ("capability empty", "real_capability_inventory_state", {}, "real_capability_valid"),
        (
            "capability missing",
            "real_capability_inventory_state",
            {"inventory_capabilities": {}},
            "real_capability_valid",
        ),
        (
            "capability extra",
            "real_capability_inventory_state",
            {**matrix.EXPECTED_SOURCE["real_capability_inventory_state"], "x": {}},
            "real_capability_valid",
        ),
        (
            "capability reordered",
            "real_capability_inventory_state",
            {
                k: matrix.EXPECTED_SOURCE["real_capability_inventory_state"][k]
                for k in reversed(matrix.EXPECTED_SOURCE["real_capability_inventory_state"])
            },
            "real_capability_valid",
        ),
        (
            "capability open",
            "real_capability_inventory_state",
            {
                **matrix.EXPECTED_SOURCE["real_capability_inventory_state"],
                "inventory_capabilities": {"x": "open"},
            },
            "real_capability_valid",
        ),
        (
            "capability True",
            "real_capability_inventory_state",
            {
                **matrix.EXPECTED_SOURCE["real_capability_inventory_state"],
                "inventory_capabilities": {"x": True},
            },
            "real_capability_valid",
        ),
        (
            "owned-field shadowing",
            "source_inventory_summary",
            {
                **matrix.EXPECTED_SOURCE["source_inventory_summary"],
                "inventory_matrix_artifact_complete": True,
            },
            "summary_valid",
        ),
    ]
    for name, key, value, validity in section_cases:
        source = _plain_source()
        source[key] = value
        cases.append((name, source, validity))
    cyc: dict[str, Any] = {}
    cyc["x"] = cyc
    source = _plain_source()
    source["source_inventory_summary"] = cyc
    cases.append(("cycle", source, "summary_valid"))
    deep: Any = []
    for _ in range(1500):
        deep = [deep]
    source = _plain_source()
    source["source_inventory_summary"] = deep
    cases.append(("deep 1500", source, "summary_valid"))
    shared: list[Any] = []
    source = _plain_source()
    source["source_inventory_summary"] = {"a": shared, "b": shared}
    cases.append(("shared acyclic references", source, "summary_valid"))
    return cases


@pytest.mark.parametrize(("name", "source", "invalid_key"), _malformed_cases())
def test_builder_level_malformed_source_matrix(
    monkeypatch: pytest.MonkeyPatch, name: str, source: Any, invalid_key: str
) -> None:
    before = matrix._copy_plain(source) if matrix._all_plain_json(source, 64) else None
    payload = _blocked_from_source(monkeypatch, source)
    if invalid_key:
        assert payload["non_execution_matrix_evidence"][invalid_key] is False, name
    assert payload["non_execution_matrix_evidence"]["entrypoint_rows_valid"] in (True, False)
    if before is not None:
        assert source == before


def test_exact_output_rows_and_field_order() -> None:
    payload = _payload()
    expected_orders = {
        "desktop_entrypoint_matrix_rows": [
            "matrix_row_id",
            "source_inventory_row_id",
            "path",
            "source_kind",
            "observed_role",
            "observed_symbol",
            "source_observed",
            "source_inventory_preserved",
            "desktop_entrypoint_candidate",
            "excluded_from_final_desktop_contract",
            "selection_required",
            "validation_required",
            "selected_as_final_desktop_entrypoint",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "qml_bundle_matrix_rows": [
            "matrix_row_id",
            "source_paths",
            "source_inventory_present",
            "source_inventory_complete",
            "source_inventory_preserved",
            "matrix_evaluated",
            "validation_required",
            "validation_performed",
            "unresolved_condition_present",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "python_dependency_matrix_rows": [
            "matrix_row_id",
            "source_inventory_present",
            "source_inventory_preserved",
            "declaration_inventory_complete",
            "resolution_required",
            "resolution_performed",
            "selection_required",
            "selection_performed",
            "validated",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "packaging_metadata_matrix_rows": [
            "matrix_row_id",
            "source_inventory_present",
            "required_declaration_present",
            "inventory_complete",
            "unresolved_condition_present",
            "validation_required",
            "validation_performed",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "existing_preview_packaging_matrix_rows": [
            "matrix_row_id",
            "source_inventory_present",
            "source_scope",
            "targets_run_local_bot",
            "targets_final_desktop_entrypoint",
            "final_desktop_profile_aligned",
            "reusable_as_final_desktop_contract",
            "profile_validation_performed",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "artifact_exclusion_policy_matrix_rows": [
            "matrix_row_id",
            "policy_source",
            "policy_version",
            "policy_observed",
            "denied_patterns_inventory_preserved",
            "policy_application_required",
            "policy_applied",
            "desktop_bundle_validation_required",
            "desktop_bundle_validation_performed",
            "approved_for_packaging_contract",
            "approved_for_build",
            "build_ready",
            "matrix_classification",
            "matrix_result",
        ],
        "inventory_finding_matrix_rows": [
            "finding_id",
            "source_paths",
            "source_observation_preserved",
            "source_severity_classification",
            "matrix_evaluated",
            "requires_packaging_contract_action",
            "contract_blocker_present",
            "resolved",
            "approved",
            "matrix_classification",
            "matrix_result",
        ],
        "packaging_scope_matrix_rows": [
            "scope_id",
            "source_inventory_artifact_present",
            "inventory_matrix_evaluated",
            "supporting_matrix_row_ids",
            "resolved_condition_count",
            "unresolved_condition_ids",
            "unresolved_condition_count",
            "ready_for_packaging_contract",
            "scope_ready",
            "scope_authorized",
            "build_ready",
            "failure_policy",
            "matrix_classification",
            "matrix_result",
        ],
        "packaging_requirement_matrix_rows": [
            "requirement_id",
            "required",
            "source_inventory_observed",
            "inventory_requirement_satisfied",
            "matrix_evaluated",
            "packaging_contract_requirement_satisfied",
            "build_requirement_satisfied",
            "missing_inventory",
            "unresolved_for_contract",
            "unresolved_condition_ids",
            "requires_future_explicit_step",
            "build_ready",
            "packaging_authorized",
            "build_authorized",
            "failure_policy",
            "matrix_classification",
            "matrix_result",
        ],
        "unresolved_contract_blocker_rows": [
            "blocker_id",
            "source_finding_ids",
            "affected_scope_ids",
            "blocker_present",
            "resolved",
            "requires_18_3_contract",
            "blocks_build_readiness",
            "blocks_packaging_authorization",
            "blocks_build_authorization",
            "failure_policy",
            "classification",
            "result",
        ],
    }
    for section, order in expected_orders.items():
        for row in payload[section]:
            assert list(row) == order
            assert row.get("build_ready", False) is False
            assert row.get("approved_for_build", False) is False
            assert row.get("packaging_authorized", False) is False
            assert row.get("build_authorized", False) is False
    _assert_referential_integrity(payload)


DEPENDENT_EXPECTATIONS = {
    "packaging_metadata_inventory": {
        "validity": "packaging_metadata_valid",
        "requirements_true": [
            "desktop_application_entrypoint_inventory",
            "qml_asset_inventory",
            "python_dependency_inventory",
            "packaging_profile_alignment",
        ],
        "scopes": {
            "desktop_application_entrypoint": (True, True),
            "qt_qml_runtime_bundle": (True, False),
        },
    },
    "desktop_entrypoint_inventory_rows": {
        "validity": "entrypoint_rows_valid",
        "requirements_false": ["desktop_application_entrypoint_inventory"],
        "scopes": {"desktop_application_entrypoint": (False, False)},
    },
    "qml_source_inventory": {
        "validity": "qml_inventory_valid",
        "requirements_false": ["qml_asset_inventory"],
        "scopes": {"qt_qml_runtime_bundle": (False, False)},
    },
    "python_dependency_inventory": {
        "validity": "python_dependency_inventory_valid",
        "requirements_false": ["python_dependency_inventory"],
        "scopes": {"windows_exe_artifact_pipeline": (True, False)},
    },
    "existing_cli_preview_packaging_inventory": {
        "validity": "preview_packaging_valid",
        "requirements_false": ["packaging_profile_alignment"],
        "scopes": {"windows_exe_artifact_pipeline": (True, False)},
    },
    "config_and_runtime_reference_inventory_rows": {
        "validity": "config_reference_rows_valid",
        "requirements_false": ["secret_and_local_data_exclusion_policy"],
        "scopes": {"windows_exe_artifact_pipeline": (True, False)},
    },
    "artifact_exclusion_policy_inventory": {
        "validity": "artifact_exclusion_policy_valid",
        "requirements_false": ["secret_and_local_data_exclusion_policy"],
        "scopes": {"windows_exe_artifact_pipeline": (True, False)},
    },
}


@pytest.mark.parametrize("section", list(DEPENDENT_EXPECTATIONS))
def test_local_global_isolation_degrades_only_dependent_rows_and_scopes(
    monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    source = _plain_source()
    source[section] = {}
    payload = _blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_matrix_evidence"]
    expected = DEPENDENT_EXPECTATIONS[section]
    assert evidence["source_accepted"] is False
    assert evidence[expected["validity"]] is False
    requirements = {
        row["requirement_id"]: row for row in payload["packaging_requirement_matrix_rows"]
    }
    for requirement_id in expected.get("requirements_true", []):
        assert requirements[requirement_id]["source_inventory_observed"] is True
        assert requirements[requirement_id]["inventory_requirement_satisfied"] is True
    for requirement_id in expected.get("requirements_false", []):
        assert requirements[requirement_id]["source_inventory_observed"] is False
        assert requirements[requirement_id]["inventory_requirement_satisfied"] is False
    scopes = {row["scope_id"]: row for row in payload["packaging_scope_matrix_rows"]}
    for scope_id, (source_present, evaluated) in expected["scopes"].items():
        assert scopes[scope_id]["source_inventory_artifact_present"] is source_present
        assert scopes[scope_id]["inventory_matrix_evaluated"] is evaluated
    _assert_referential_integrity(payload)


def test_referential_integrity_for_blocked_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    for section in [
        "inventory_findings",
        "packaging_metadata_inventory",
        "existing_cli_preview_packaging_inventory",
        "artifact_exclusion_policy_inventory",
        "source_inventory_summary",
    ]:
        source = _plain_source()
        source[section] = {}
        payload = _blocked_from_source(monkeypatch, source)
        _assert_referential_integrity(payload)
        if section == "inventory_findings":
            assert payload["inventory_finding_matrix_rows"] == []
            assert all(
                row["source_finding_ids"] == []
                for row in payload["unresolved_contract_blocker_rows"]
            )
            assert all(
                row["affected_scope_ids"] for row in payload["unresolved_contract_blocker_rows"]
            )


def test_blocked_artifact_policy_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _plain_source()
    source["artifact_exclusion_policy_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    row = payload["artifact_exclusion_policy_matrix_rows"][0]
    assert row["policy_source"] == ""
    assert row["policy_version"] == ""
    assert row["policy_observed"] is False
    assert row["denied_patterns_inventory_preserved"] is False
    assert row["matrix_classification"] == "artifact_exclusion_policy_source_not_preserved"
    assert row["matrix_result"] == "artifact_exclusion_policy_matrix_blocked_by_invalid_source"


def test_nested_mutation_sensitivity_uses_existing_nested_values() -> None:
    source = _plain_source()
    source["source_inventory_summary"] = {
        "nested_list": [{"value": 1}],
        "nested_dict": {"inner": {"value": 1}},
        "scalar": "stable",
        "inventory_matrix_artifact_complete": False,
    }
    section, custom_key, before, values = _custom_first_section(
        source, "source_inventory_summary", "inventory_matrix_artifact_complete", Bomb
    )
    section["nested_list"][0]["value"] = 2
    with pytest.raises(AssertionError):
        _assert_custom_first_not_mutated(
            section, before, custom_key, "inventory_matrix_artifact_complete", values
        )
    source = _plain_source()
    source["source_inventory_summary"] = {
        "nested_list": [{"value": 1}],
        "nested_dict": {"inner": {"value": 1}},
        "scalar": "stable",
        "inventory_matrix_artifact_complete": False,
    }
    section, custom_key, before, values = _custom_first_section(
        source, "source_inventory_summary", "inventory_matrix_artifact_complete", Bomb
    )
    section["nested_dict"]["inner"]["value"] = 2
    with pytest.raises(AssertionError):
        _assert_custom_first_not_mutated(
            section, before, custom_key, "inventory_matrix_artifact_complete", values
        )


def _requirement(payload: dict[str, Any], requirement_id: str) -> dict[str, Any]:
    return next(
        row
        for row in payload["packaging_requirement_matrix_rows"]
        if row["requirement_id"] == requirement_id
    )


def _scope(payload: dict[str, Any], scope_id: str) -> dict[str, Any]:
    return next(
        row for row in payload["packaging_scope_matrix_rows"] if row["scope_id"] == scope_id
    )


def test_nominal_supporting_ids_are_exact_and_meaningful() -> None:
    payload = _payload()
    assert _scope(payload, "desktop_application_entrypoint")["supporting_matrix_row_ids"] == [
        "desktop_module_launcher_matrix",
        "desktop_application_main_matrix",
    ]
    assert _scope(payload, "qt_qml_runtime_bundle")["supporting_matrix_row_ids"] == [
        "default_qml_entrypoint",
        "pyside_qml_root",
        "shared_qml_root",
        "styles_module",
        "windows_shared_qml_import_path",
        "setuptools_ui_package_discovery",
        "qml_package_data_declaration",
    ]
    assert _scope(payload, "windows_exe_artifact_pipeline")["supporting_matrix_row_ids"] == [
        "project_dependency_declarations",
        "desktop_optional_dependency_declarations",
        "dependency_resolution",
        "desktop_build_tool_candidates",
        "safe_exe_preview_build_plan",
        "windows_preview_profile",
        "artifact_exclusion_policy",
    ]
    assert _requirement(payload, "future_explicit_build_execution_gate")["matrix_evaluated"] is True
    _assert_referential_integrity(payload)


def test_supporting_integrity_for_blocked_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    for section in [
        "desktop_entrypoint_inventory_rows",
        "qml_source_inventory",
        "python_dependency_inventory",
        "packaging_metadata_inventory",
        "existing_cli_preview_packaging_inventory",
        "artifact_exclusion_policy_inventory",
        "inventory_findings",
        "source_inventory_summary",
    ]:
        source = _plain_source()
        source[section] = {}
        payload = _blocked_from_source(monkeypatch, source)
        future_gate = _requirement(payload, "future_explicit_build_execution_gate")
        assert future_gate["source_inventory_observed"] is False
        assert future_gate["inventory_requirement_satisfied"] is False
        assert future_gate["matrix_evaluated"] is True
        assert future_gate["missing_inventory"] is True
        assert future_gate["unresolved_for_contract"] is True
        assert future_gate["requires_future_explicit_step"] is True
        _assert_referential_integrity(payload)


def test_invalid_entrypoint_source_has_no_dangling_supporting_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _plain_source()
    source["desktop_entrypoint_inventory_rows"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    assert payload["desktop_entrypoint_matrix_rows"] == []
    desktop_scope = _scope(payload, "desktop_application_entrypoint")
    assert desktop_scope["supporting_matrix_row_ids"] == []
    assert desktop_scope["source_inventory_artifact_present"] is False
    assert desktop_scope["inventory_matrix_evaluated"] is False
    assert _scope(payload, "qt_qml_runtime_bundle")["supporting_matrix_row_ids"] == [
        "default_qml_entrypoint",
        "pyside_qml_root",
        "shared_qml_root",
        "styles_module",
        "windows_shared_qml_import_path",
        "setuptools_ui_package_discovery",
        "qml_package_data_declaration",
    ]
    _assert_referential_integrity(payload)


def test_invalid_qml_source_uses_blocked_rows_without_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _plain_source()
    source["qml_source_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    assert len(payload["qml_bundle_matrix_rows"]) == 5
    for row in payload["qml_bundle_matrix_rows"]:
        assert row["source_paths"] == []
        assert row["source_inventory_present"] is False
        assert row["source_inventory_complete"] is False
        assert row["source_inventory_preserved"] is False
        assert row["matrix_evaluated"] is False
        assert row["matrix_classification"] == "qml_source_inventory_not_preserved"
        assert row["matrix_result"] == "qml_bundle_matrix_blocked_by_invalid_source"
        assert row["approved_for_build"] is False
    qml_scope = _scope(payload, "qt_qml_runtime_bundle")
    assert set(qml_scope["supporting_matrix_row_ids"]) <= _all_matrix_row_ids(payload)
    assert "setuptools_ui_package_discovery" in qml_scope["supporting_matrix_row_ids"]
    _assert_referential_integrity(payload)


def test_invalid_dependency_source_uses_blocked_source_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _plain_source()
    source["python_dependency_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    rows = {row["matrix_row_id"]: row for row in payload["python_dependency_matrix_rows"]}
    for row_id in [
        "project_dependency_declarations",
        "desktop_optional_dependency_declarations",
        "desktop_build_tool_candidates",
    ]:
        assert rows[row_id]["source_inventory_present"] is False
        assert rows[row_id]["source_inventory_preserved"] is False
        assert rows[row_id]["declaration_inventory_complete"] is False
        assert rows[row_id]["matrix_classification"] == "dependency_inventory_source_not_preserved"
        assert rows[row_id]["matrix_result"] == "dependency_matrix_blocked_by_invalid_source"
    assert (
        rows["dependency_resolution"]["matrix_classification"]
        == "dependency_resolution_not_performed"
    )
    assert rows["dependency_resolution"]["matrix_result"] == "dependency_resolution_pending"
    assert set(
        _scope(payload, "windows_exe_artifact_pipeline")["supporting_matrix_row_ids"]
    ) <= _all_matrix_row_ids(payload)


def test_invalid_metadata_source_uses_blocked_metadata_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _plain_source()
    source["packaging_metadata_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    for row in payload["packaging_metadata_matrix_rows"]:
        assert row["source_inventory_present"] is False
        assert row["inventory_complete"] is False
        assert row["matrix_classification"] == "packaging_metadata_source_not_preserved"
        assert row["matrix_result"] == "packaging_metadata_matrix_blocked_by_invalid_source"
    qml_scope = _scope(payload, "qt_qml_runtime_bundle")
    assert qml_scope["source_inventory_artifact_present"] is True
    assert qml_scope["inventory_matrix_evaluated"] is False
    _assert_referential_integrity(payload)


def test_invalid_preview_source_uses_blocked_preview_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    source = _plain_source()
    source["existing_cli_preview_packaging_inventory"] = {}
    payload = _blocked_from_source(monkeypatch, source)
    for row in payload["existing_preview_packaging_matrix_rows"]:
        assert row["source_inventory_present"] is False
        assert row["source_scope"] == ""
        assert row["targets_run_local_bot"] is False
        assert row["targets_final_desktop_entrypoint"] is False
        assert row["matrix_classification"] == "cli_preview_packaging_source_not_preserved"
        assert row["matrix_result"] == "preview_packaging_matrix_blocked_by_invalid_source"
    assert _scope(payload, "windows_exe_artifact_pipeline")["inventory_matrix_evaluated"] is False
    _assert_referential_integrity(payload)


def test_supporting_lists_do_not_alias_payloads_or_constants() -> None:
    one = _payload()
    two = _payload()
    blocker_snapshot = matrix._copy_plain(matrix.BLOCKER_RELATION_ROWS)
    expected_snapshot = matrix._copy_plain(matrix.EXPECTED_SOURCE)
    row_ids_before = _all_matrix_row_ids(one)
    other_scope_support = matrix._copy_plain(
        one["packaging_scope_matrix_rows"][1]["supporting_matrix_row_ids"]
    )
    two_support = matrix._copy_plain(
        two["packaging_scope_matrix_rows"][0]["supporting_matrix_row_ids"]
    )
    one["packaging_scope_matrix_rows"][0]["supporting_matrix_row_ids"].append("mutated")
    assert two["packaging_scope_matrix_rows"][0]["supporting_matrix_row_ids"] == two_support
    assert one["packaging_scope_matrix_rows"][1]["supporting_matrix_row_ids"] == other_scope_support
    assert _all_matrix_row_ids(one) == row_ids_before
    assert matrix.BLOCKER_RELATION_ROWS == blocker_snapshot
    assert matrix.EXPECTED_SOURCE == expected_snapshot
