from __future__ import annotations

import ast
import copy
import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

from ui.pyside_app.preview_block_p_desktop_exe_packaging_entry_contract import (
    build_preview_block_p_desktop_exe_packaging_entry_contract,
)
from ui.pyside_app import preview_block_p_desktop_exe_packaging_source_inventory as inv

ROOT = Path(__file__).resolve().parents[2]
QML_EXT = {
    ".qml",
    ".js",
    ".mjs",
    ".qmltypes",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".webp",
    ".ico",
    ".ttf",
    ".otf",
}


def build() -> dict[str, Any]:
    return inv.build_preview_block_p_desktop_exe_packaging_source_inventory()


def qml_files(root: str) -> list[str]:
    base = ROOT / root
    return sorted(
        p.relative_to(ROOT).as_posix()
        for p in base.rglob("*")
        if p.is_file() and (p.name == "qmldir" or p.suffix.lower() in QML_EXT)
    )


def test_expected_source_matches_current_18_0() -> None:
    source = build_preview_block_p_desktop_exe_packaging_entry_contract()
    assert source == inv.EXPECTED_SOURCE
    assert list(source) == inv.TOP_LEVEL_FIELDS_18_0
    for key, value in inv.SOURCE_IDENTITY_EXPECTED.items():
        assert source[key] == value


def test_identity_order_reference_and_json_serializable() -> None:
    payload = build()
    assert list(payload) == inv.TOP_LEVEL_FIELDS
    assert payload["schema_version"] == inv.SCHEMA_VERSION
    assert payload["block"] == "P"
    assert payload["step"] == "18.1"
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-18.2"
    json.dumps(payload)


def test_source_builder_called_exactly_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real = build_preview_block_p_desktop_exe_packaging_entry_contract()

    def fake() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return real

    monkeypatch.setattr(inv, "build_preview_block_p_desktop_exe_packaging_entry_contract", fake)
    assert build()["source_inventory_artifact_complete"] is True
    assert calls == 1


def test_source_inventory_artifact_complete_source_only() -> None:
    p = build()
    assert p["source_inventory_artifact_complete"] is True
    assert p["ready_for_block_p_2"] is True
    summary = p["source_inventory_summary"]
    assert summary["source_only"] is True
    assert summary["static_inventory"] is True
    assert summary["inventory_validated"] is False
    assert summary["build_ready"] is False


def test_four_desktop_entrypoint_inventory_rows() -> None:
    rows = build()["desktop_entrypoint_inventory_rows"]
    assert [r["path"] for r in rows] == [
        "ui/pyside_app/__main__.py",
        "ui/pyside_app/app.py",
        "scripts/run_local_bot.py",
        "scripts/operator_preview_bundle.py",
    ]
    assert [r["desktop_entrypoint_candidate"] for r in rows] == [True, True, False, False]
    assert rows[1]["uses_qguiapplication"] is True
    assert rows[1]["default_qml_path"] == "ui/pyside_app/qml/MainWindow.qml"


def test_qml_source_inventory_matches_current_repo() -> None:
    q = build()["qml_source_inventory"]
    assert inv.PYSIDE_QML_SOURCE_FILES_18_1 == qml_files("ui/pyside_app/qml")
    assert inv.SHARED_QML_SOURCE_FILES_18_1 == qml_files("ui/qml")
    assert q["pyside_qml_source_files"] == inv.PYSIDE_QML_SOURCE_FILES_18_1
    assert q["shared_qml_source_files"] == inv.SHARED_QML_SOURCE_FILES_18_1
    assert q["pyside_qml_source_file_count"] == len(inv.PYSIDE_QML_SOURCE_FILES_18_1)
    assert q["shared_qml_source_file_count"] == len(inv.SHARED_QML_SOURCE_FILES_18_1)


def test_main_window_import_inventory() -> None:
    text = (ROOT / "ui/pyside_app/qml/MainWindow.qml").read_text()
    parsed = [line.strip() for line in text.splitlines() if line.startswith("import ")]
    assert parsed == [
        "import QtQuick",
        "import QtQuick.Controls",
        "import QtQuick.Layouts",
        "import QtQuick.Effects",
        'import "components" as Components',
        'import "components/layout" as LayoutComponents',
        "import Styles 1.0 as StylesModule",
        'import "views" as Views',
    ]
    assert build()["qml_source_inventory"]["main_window_imports"] == [
        "QtQuick",
        "QtQuick.Controls",
        "QtQuick.Layouts",
        "QtQuick.Effects",
        "components",
        "components/layout",
        "Styles 1.0",
        "views",
    ]


def test_styles_qmldir_inventory() -> None:
    text = (ROOT / "ui/pyside_app/qml/Styles/qmldir").read_text()
    obs = build()["qml_source_inventory"]["styles_module_observation"]
    assert "module Styles" in text
    assert "DesignSystem 1.0 DesignSystem.qml" in text
    assert obs == {
        "module": "Styles",
        "type_name": "DesignSystem",
        "version": "1.0",
        "source": "DesignSystem.qml",
    }


def test_shared_qml_platform_condition_observed() -> None:
    text = (ROOT / "ui/pyside_app/app.py").read_text()
    assert 'sys.platform != "win32"' in text
    obs = build()["qml_source_inventory"]["windows_shared_qml_import_path_observation"]
    assert obs["shared_qml_root_added_on_non_windows"] is True
    assert obs["shared_qml_root_added_on_windows"] is False


def test_config_and_runtime_reference_inventory() -> None:
    rows = build()["config_and_runtime_reference_inventory_rows"]
    assert len(rows) == 6
    assert rows[0]["path"] == "ui/config/example.yaml"
    assert rows[1]["source_path"] == "data/sample_ohlcv/trend.csv"
    assert rows[1]["existence_observed_in_repo"] == (ROOT / "data/sample_ohlcv/trend.csv").exists()
    assert all(row.get("secret_content_read") is False for row in rows[2:5])
    assert rows[5]["value_copied_to_inventory"] is False


def pyproject() -> dict[str, Any]:
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_project_dependency_inventory_matches_pyproject() -> None:
    deps = pyproject()["project"]["dependencies"]
    assert inv.PROJECT_DEPENDENCY_SPECS_18_1 == deps
    assert build()["python_dependency_inventory"]["project_dependency_specs"] == deps


def test_desktop_optional_dependency_inventory_matches_pyproject() -> None:
    deps = pyproject()["project"]["optional-dependencies"]["desktop"]
    assert inv.DESKTOP_OPTIONAL_DEPENDENCY_SPECS_18_1 == deps
    assert build()["python_dependency_inventory"]["desktop_optional_dependency_specs"] == deps


def deployment_files(root: str) -> list[str]:
    allowed = {
        ".py",
        ".toml",
        ".md",
        ".txt",
        ".lock",
        ".yaml",
        ".yml",
        ".json",
        ".key",
        ".whl",
        ".gitkeep",
    }
    base = ROOT / root
    return sorted(
        p.relative_to(ROOT).as_posix()
        for p in base.rglob("*")
        if p.is_file() and (p.name == ".gitkeep" or p.suffix.lower() in allowed)
    )


def test_packaging_metadata_inventory_matches_pyproject() -> None:
    data = pyproject()
    meta = build()["packaging_metadata_inventory"]
    assert (
        inv.SETUPTOOLS_PACKAGE_INCLUDE_PATTERNS_18_1
        == data["tool"]["setuptools"]["packages"]["find"]["include"]
    )
    assert inv.SETUPTOOLS_PACKAGE_DATA_KEYS_18_1 == list(
        data["tool"]["setuptools"]["package-data"].keys()
    )
    assert meta["ui_package_discovery_pattern_present"] is False
    assert meta["qml_package_data_declaration_present"] is False
    assert inv.DEPLOY_PACKAGING_SOURCE_FILES_18_1 == deployment_files("deploy/packaging")
    assert inv.DEPLOYMENT_DOCUMENTATION_FILES_18_1 == deployment_files("docs/deployment")
    assert meta["deploy_packaging_source_files"] == inv.DEPLOY_PACKAGING_SOURCE_FILES_18_1
    assert meta["deployment_documentation_files"] == inv.DEPLOYMENT_DOCUMENTATION_FILES_18_1
    assert meta["all_deployment_inventory_paths_unique"] is True


def test_existing_cli_preview_plan_inventory() -> None:
    text = (ROOT / "scripts/safe_exe_preview_build_plan.py").read_text()
    rows = build()["existing_cli_preview_packaging_inventory"]["rows"]
    assert "scripts/run_local_bot.py" in text
    assert rows[0]["current_plan_entrypoint"] == "scripts/run_local_bot.py"
    assert rows[0]["current_artifact_type"] == "cli-preview-exe"
    assert rows[0]["current_selected_build_tool"] == "pyinstaller"
    assert rows[0]["build_command_executed"] is False


def test_preview_packaging_profiles_inventory() -> None:
    rows = build()["existing_cli_preview_packaging_inventory"]["rows"][1:]
    expected = {
        "windows": {
            "platform": "windows",
            "pyinstaller": {
                "entrypoint": r"..\..\..\..\scripts\run_local_bot.py",
                "runtime_name": "dudzian-bot-preview",
                "hidden_imports": [
                    "bot_core.runtime.bootstrap",
                    "bot_core.runtime.pipeline",
                    "bot_core.runtime.config",
                ],
                "dist_dir": r"..\..\..\..\dist\preview\windows",
                "work_dir": r"..\..\..\..\var\build\preview\pyinstaller\windows",
            },
            "briefcase": {
                "project": r"..\..\..\..\ui\briefcase",
                "app": "BotTradingShell",
                "output_dir": r"..\..\..\..\dist\preview\briefcase\windows",
            },
        },
        "linux": {
            "platform": "linux",
            "pyinstaller": {
                "entrypoint": "../../../../scripts/run_local_bot.py",
                "runtime_name": "dudzian-bot-preview",
                "hidden_imports": [
                    "bot_core.runtime.bootstrap",
                    "bot_core.runtime.pipeline",
                    "bot_core.runtime.config",
                ],
                "dist_dir": "../../../../dist/preview/linux",
                "work_dir": "../../../../var/build/preview/pyinstaller/linux",
            },
            "briefcase": {
                "project": "../../../../ui/briefcase",
                "app": "BotTradingShell",
                "output_dir": "../../../../dist/preview/briefcase/linux",
            },
        },
        "macos": {
            "platform": "macos",
            "pyinstaller": {
                "entrypoint": "../../../../scripts/run_local_bot.py",
                "runtime_name": "dudzian-bot-preview",
                "hidden_imports": [
                    "bot_core.runtime.bootstrap",
                    "bot_core.runtime.pipeline",
                    "bot_core.runtime.config",
                ],
                "dist_dir": "../../../../dist/preview/macos",
                "work_dir": "../../../../var/build/preview/pyinstaller/macos",
            },
            "briefcase": {
                "project": "../../../../ui/briefcase",
                "app": "BotTradingShell",
                "output_dir": "../../../../dist/preview/briefcase/macos",
            },
        },
    }
    for row in rows:
        data = tomllib.loads((ROOT / row["path"]).read_text())
        assert data == expected[row["profile_platform"]]
        assert row["profile_pyinstaller_entrypoint_targets_run_local_bot"] is True
        assert row["profile_targets_desktop_pyside_entrypoint"] is False


def test_artifact_exclusion_policy_inventory() -> None:
    tree = ast.parse((ROOT / "scripts/safe_exe_preview_build_plan.py").read_text())
    assigned = next(
        n.value
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "DENIED_ARTIFACT_PATTERNS" for t in n.targets)
    )
    assert ast.literal_eval(assigned) == inv.DENIED_ARTIFACT_PATTERNS_18_1
    assert (
        build()["artifact_exclusion_policy_inventory"]["denied_artifact_patterns"]
        == inv.DENIED_ARTIFACT_PATTERNS_18_1
    )


def test_eleven_inventory_findings() -> None:
    findings = build()["inventory_findings"]
    expected_paths = {
        "desktop_module_launcher_observed": ["ui/pyside_app/__main__.py"],
        "desktop_application_main_observed": ["ui/pyside_app/app.py"],
        "default_qml_entrypoint_observed": ["ui/pyside_app/qml/MainWindow.qml"],
        "two_qml_source_roots_observed": ["ui/pyside_app/qml", "ui/qml"],
        "shared_qml_import_path_is_platform_conditional": [
            "ui/pyside_app/app.py",
            "ui/pyside_app/qml/MainWindow.qml",
            "ui/qml",
        ],
        "cli_preview_plan_targets_non_desktop_entrypoint": [
            "scripts/safe_exe_preview_build_plan.py",
            "deploy/packaging/profiles/preview/windows.toml",
            "deploy/packaging/profiles/preview/linux.toml",
            "deploy/packaging/profiles/preview/macos.toml",
        ],
        "desktop_build_tools_declared_as_optional_dependencies": ["pyproject.toml"],
        "ui_package_discovery_not_declared_in_current_setuptools_include": ["pyproject.toml"],
        "qml_package_data_not_declared_in_current_setuptools_metadata": ["pyproject.toml"],
        "example_config_references_local_secret_paths": ["ui/config/example.yaml"],
        "example_config_contains_sensitive_field_reference": ["ui/config/example.yaml"],
    }
    assert [f["finding_id"] for f in findings] == list(expected_paths)
    for finding in findings:
        assert finding["source_paths"] == expected_paths[finding["finding_id"]]
        assert finding["severity_classification"] == "inventory_observation"


def test_all_real_capabilities_blocked_non_vacuously() -> None:
    state = build()["real_capability_inventory_state"]
    assert state["inherited_18_0_capabilities_known_blocked"] is True
    assert state["inventory_capabilities_known_blocked"] is True
    assert state["all_real_capabilities_blocked_at_18_1"] is True
    assert state["inventory_capabilities"]
    assert set(state["inventory_capabilities"].values()) == {"blocked"}


def test_no_build_packaging_release_runtime_orders() -> None:
    p = build()
    decision = p["fail_closed_inventory_decision"]
    boundaries = p["inventory_boundaries"]
    for key in [
        "build_ready_by_18_1",
        "packaging_authorized_by_18_1",
        "build_authorized_by_18_1",
        "build_executed_by_18_1",
        "artifact_created_by_18_1",
        "release_authorized_by_18_1",
        "runtime_enabled_by_18_1",
        "orders_enabled_by_18_1",
    ]:
        assert decision[key] is False
    for key in [
        "packaging_performed",
        "artifact_created",
        "release_performed",
        "runtime_started",
        "orders_enabled",
        "network_opened",
    ]:
        assert boundaries[key] is False


def test_nominal_payload_has_no_shared_mutable_containers() -> None:
    seen: set[int] = set()
    stack = [build()]
    while stack:
        item = stack.pop()
        if type(item) in (dict, list):
            assert id(item) not in seen
            seen.add(id(item))
            stack.extend(item.values() if type(item) is dict else item)


def test_independent_builder_calls_do_not_share_state() -> None:
    a = build()
    b = build()
    a["qml_source_inventory"]["pyside_qml_source_files"].append("x")
    assert "x" not in b["qml_source_inventory"]["pyside_qml_source_files"]
    assert "x" not in inv.PYSIDE_QML_SOURCE_FILES_18_1


def test_forbidden_raw_tokens_absent() -> None:
    text = (
        ROOT / "ui/pyside_app/preview_block_p_desktop_exe_packaging_source_inventory.py"
    ).read_text()
    assert "create_order" not in text
    assert "fetch_balance" not in text
    assert "ccxt" not in text


def test_exact_ast_guard() -> None:
    tree = ast.parse(
        (
            ROOT / "ui/pyside_app/preview_block_p_desktop_exe_packaging_source_inventory.py"
        ).read_text()
    )
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Import)]
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == [
        "__future__",
        "typing",
        "ui.pyside_app.preview_block_p_desktop_exe_packaging_entry_contract",
    ]
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    name_calls = sorted({n.func.id for n in calls if isinstance(n.func, ast.Name)})
    attribute_calls = sorted({n.func.attr for n in calls if isinstance(n.func, ast.Attribute)})
    assert name_calls == [
        "_all_plain_json",
        "_capabilities",
        "_copy_plain",
        "_entry_rows",
        "_exact_plain_matches",
        "_future_steps",
        "_no_shadowing",
        "_owned_fields_are_unshadowed",
        "_plain_dict_section",
        "_safe_top_level_source",
        "_scalar_reference",
        "_section_valid",
        "_source_identity_valid",
        "all",
        "bool",
        "build_preview_block_p_desktop_exe_packaging_entry_contract",
        "id",
        "len",
        "list",
        "set",
        "type",
        "zip",
    ]
    assert attribute_calls == [
        "add",
        "append",
        "discard",
        "get",
        "items",
        "keys",
        "pop",
        "update",
        "upper",
        "values",
    ]
    assert (
        sum(
            1
            for n in calls
            if isinstance(n.func, ast.Name)
            and n.func.id == "build_preview_block_p_desktop_exe_packaging_entry_contract"
        )
        == 1
    )
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
    assert not forbidden.intersection(name_calls)
    assert not forbidden.intersection(attribute_calls)


@pytest.mark.parametrize(
    "bad_source", [None, [], {"schema_version": 1}, {"schema_version": 1.0}, {"x": object()}]
)
def test_malformed_sources_block_without_exception(
    monkeypatch: pytest.MonkeyPatch, bad_source: Any
) -> None:
    monkeypatch.setattr(
        inv, "build_preview_block_p_desktop_exe_packaging_entry_contract", lambda: bad_source
    )
    payload = build()
    assert payload["source_inventory_artifact_complete"] is False
    assert payload["status"] == inv.BLOCKED_STATUS
    json.dumps(payload)


class BombValue:
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        raise AssertionError("custom equality must not be called")


class LyingValue:
    equality_calls = 0

    def __eq__(self, other: object) -> bool:
        type(self).equality_calls += 1
        return True


def test_bomb_and_lying_identity_values_block_without_custom_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for cls in (BombValue, LyingValue):
        source = build_preview_block_p_desktop_exe_packaging_entry_contract()
        for key in inv.SOURCE_IDENTITY_EXPECTED:
            source[key] = cls()
        cls.equality_calls = 0
        monkeypatch.setattr(
            inv, "build_preview_block_p_desktop_exe_packaging_entry_contract", lambda: source
        )
        payload = build()
        assert payload["source_inventory_artifact_complete"] is False
        assert payload["status"] == inv.BLOCKED_STATUS
        assert cls.equality_calls == 0
        json.dumps(payload)


def test_cycle_depth_subclass_float_and_shared_reference_plain_json_coverage() -> None:
    cycle: list[Any] = []
    cycle.append(cycle)
    assert inv._all_plain_json(cycle, inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    deep: Any = []
    for _ in range(1500):
        deep = [deep]
    assert inv._all_plain_json(deep, 2000) is True
    assert inv._all_plain_json(deep, inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    shared = ["ok"]
    assert inv._all_plain_json([shared, shared], inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is True

    class DictSubclass(dict[str, Any]):
        pass

    class ListSubclass(list[Any]):
        pass

    assert inv._all_plain_json(DictSubclass(), inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    assert inv._all_plain_json(ListSubclass(), inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False
    assert inv._all_plain_json(1.0, inv.MAX_DIAGNOSTIC_CONTAINER_DEPTH) is False


def clone_plain(value: Any) -> Any:
    return json.loads(json.dumps(value))


def blocked_from_source(monkeypatch: pytest.MonkeyPatch, source: Any) -> dict[str, Any]:
    snapshot = (
        clone_plain(source) if type(source) is dict and inv._all_plain_json(source, 2000) else None
    )
    monkeypatch.setattr(
        inv, "build_preview_block_p_desktop_exe_packaging_entry_contract", lambda: source
    )
    payload = build()
    assert payload["source_inventory_artifact_complete"] is False
    assert payload["ready_for_block_p_2"] is False
    assert payload["status"] == inv.BLOCKED_STATUS
    json.dumps(payload)
    if snapshot is not None:
        assert source == snapshot
    return payload


def test_ast_entrypoint_sources_without_importing_modules() -> None:
    main_tree = ast.parse((ROOT / "ui/pyside_app/__main__.py").read_text())
    assert any(
        isinstance(node, ast.ImportFrom)
        and node.module == "app"
        and node.level == 1
        and [alias.name for alias in node.names] == ["main"]
        for node in main_tree.body
    )
    if_node = next(node for node in main_tree.body if isinstance(node, ast.If))
    assert ast.unparse(if_node.test) == "__name__ == '__main__'"
    assert any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "SystemExit"
        and node.exc.args
        and isinstance(node.exc.args[0], ast.Call)
        and isinstance(node.exc.args[0].func, ast.Name)
        and node.exc.args[0].func.id == "main"
        for node in if_node.body
    )

    app_text = (ROOT / "ui/pyside_app/app.py").read_text()
    app_tree = ast.parse(app_text)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in app_tree.body)
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "AppOptions" for node in app_tree.body
    )
    assert any(
        isinstance(node, ast.ClassDef) and node.name == "BotPysideApplication"
        for node in app_tree.body
    )
    calls = [node for node in ast.walk(app_tree) if isinstance(node, ast.Call)]
    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "QGuiApplication" for node in calls
    )
    assert any(
        isinstance(node.func, ast.Name) and node.func.id == "QQmlApplicationEngine"
        for node in calls
    )
    assert "ui/config/example.yaml" in app_text
    assert '"qml" / "MainWindow.qml"' in (ROOT / "ui/pyside_app/config.py").read_text()
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "load" for node in calls)
    assert any(isinstance(node.func, ast.Attribute) and node.func.attr == "exec" for node in calls)


def test_existing_cli_preview_plan_ast_inventory() -> None:
    text = (ROOT / "scripts/safe_exe_preview_build_plan.py").read_text()
    tree = ast.parse(text)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_payload"
    )
    assigned: dict[str, Any] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {
                    "allowed_entrypoint",
                    "optional_entrypoint",
                    "build_tool_candidates",
                }:
                    assigned[target.id] = ast.literal_eval(node.value)
    assert assigned == {
        "allowed_entrypoint": "scripts/run_local_bot.py",
        "optional_entrypoint": "scripts/operator_preview_bundle.py",
        "build_tool_candidates": ["pyinstaller", "briefcase"],
    }
    assert (
        'selected_build_tool = "pyinstaller" if "pyinstaller" in build_tool_candidates else "undecided"'
        in text
    )
    for literal in [
        '"build_command_preview"',
        '"dudzian-bot-preview"',
        '"cli-preview-exe"',
        '"build_command_execution_allowed": False',
        '"build_command_executed": False',
        'ARTIFACT_EXCLUDE_POLICY_VERSION = "security_packaging_artifact_policy.v1"',
    ]:
        assert literal in text


MUTATION_CASES: list[tuple[str, list[Any], Any]] = [
    ("changed source status", ["block_p_desktop_exe_packaging_entry_contract_status"], "changed"),
    (
        "changed source decision",
        ["block_p_desktop_exe_packaging_entry_contract_decision"],
        "changed",
    ),
    ("entry_contract_summary empty", ["entry_contract_summary"], {}),
    (
        "summary build execution authorized",
        ["entry_contract_summary", "build_execution_authorized"],
        True,
    ),
    ("summary build ready", ["entry_contract_summary", "build_command_approved"], True),
    (
        "desktop product direction changed",
        ["desktop_exe_product_direction", "product_direction"],
        "changed",
    ),
    ("scope row count changed", ["desktop_exe_packaging_scope_contract"], []),
    ("scope row order changed", ["desktop_exe_packaging_scope_contract"], "reverse"),
    ("scope row field order changed", ["desktop_exe_packaging_scope_contract", 0], "reorder"),
    (
        "scope_authorized true",
        ["desktop_exe_packaging_scope_contract", 0, "scope_authorized"],
        True,
    ),
    ("scope_ready true", ["desktop_exe_packaging_scope_contract", 0, "scope_ready"], True),
    (
        "scope classification changed",
        ["desktop_exe_packaging_scope_contract", 0, "classification"],
        "changed",
    ),
    ("scope result changed", ["desktop_exe_packaging_scope_contract", 0, "result"], "changed"),
    ("requirement row count changed", ["desktop_exe_packaging_requirement_rows"], []),
    ("requirement row order changed", ["desktop_exe_packaging_requirement_rows"], "reverse"),
    ("requirement field order changed", ["desktop_exe_packaging_requirement_rows", 0], "reorder"),
    ("missing false", ["desktop_exe_packaging_requirement_rows", 0, "missing"], False),
    ("satisfied true", ["desktop_exe_packaging_requirement_rows", 0, "satisfied"], True),
    ("requirement build ready", ["desktop_exe_packaging_requirement_rows", 0, "build_ready"], True),
    (
        "requirement packaging authorized",
        ["desktop_exe_packaging_requirement_rows", 0, "packaging_authorized"],
        True,
    ),
    (
        "requirement build authorized",
        ["desktop_exe_packaging_requirement_rows", 0, "build_authorized"],
        True,
    ),
    (
        "requirement classification changed",
        ["desktop_exe_packaging_requirement_rows", 0, "classification"],
        "changed",
    ),
    (
        "requirement result changed",
        ["desktop_exe_packaging_requirement_rows", 0, "result"],
        "changed",
    ),
    ("fail closed empty", ["fail_closed_entry_decision"], {}),
    ("build approved", ["fail_closed_entry_decision", "build_approved_by_18_0"], True),
    ("packaging approved", ["fail_closed_entry_decision", "packaging_approved_by_18_0"], True),
    ("runtime enabled", ["fail_closed_entry_decision", "runtime_enabled_by_18_0"], True),
    ("orders enabled", ["fail_closed_entry_decision", "orders_enabled_by_18_0"], True),
    ("future empty", ["future_steps"], []),
    ("future reordered", ["future_steps"], "reverse"),
    ("future source only false", ["future_steps", 0, "source_only"], False),
    ("future build performed true", ["future_steps", 0, "build_performed"], True),
    ("source boundary changed", ["source_boundaries", "can_feed_18_1"], False),
    ("entry boundary changed", ["entry_contract_boundaries", "build_tool_executed"], True),
    ("evidence changed", ["non_execution_entry_evidence", "build_performed"], True),
]


def mutate_path(source: dict[str, Any], path: list[Any], value: Any) -> None:
    target: Any = source
    for part in path[:-1]:
        target = target[part]
    last = path[-1]
    if value == "reverse":
        target[last] = list(reversed(target[last]))
    elif value == "reorder":
        target[last] = dict(reversed(list(target[last].items())))
    else:
        target[last] = value


@pytest.mark.parametrize(("case", "path", "value"), MUTATION_CASES)
def test_exact_source_mutations_block(
    monkeypatch: pytest.MonkeyPatch, case: str, path: list[Any], value: Any
) -> None:
    del case
    source = clone_plain(inv.EXPECTED_SOURCE)
    mutate_path(source, path, value)
    payload = blocked_from_source(monkeypatch, source)
    assert payload["non_execution_inventory_evidence"]["source_accepted"] is False


def test_empty_and_open_capability_sources_block_non_vacuously(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    source["real_capability_entry_state"] = {}
    payload = blocked_from_source(monkeypatch, source)
    state = payload["real_capability_inventory_state"]
    assert state["inherited_18_0_capabilities"] == {
        "inherited_block_o_capabilities": {},
        "block_p_capabilities": {},
    }
    assert state["inherited_18_0_capabilities_known_blocked"] is False
    assert state["all_real_capabilities_blocked_at_18_1"] is False
    assert state["inventory_capabilities"]
    assert set(state["inventory_capabilities"].values()) == {"blocked"}

    for replacement in ["open", True, "ready", "authorized"]:
        source = clone_plain(inv.EXPECTED_SOURCE)
        first = next(iter(source["real_capability_entry_state"]["inherited_block_o_capabilities"]))
        source["real_capability_entry_state"]["inherited_block_o_capabilities"][first] = replacement
        payload = blocked_from_source(monkeypatch, source)
        assert (
            payload["real_capability_inventory_state"]["all_real_capabilities_blocked_at_18_1"]
            is False
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


class Lying(Bomb):
    equality_calls = 0
    __hash__ = Bomb.__hash__

    def __eq__(self, other: object) -> bool:
        if type(self).armed:
            type(self).equality_calls += 1
            return True
        return False


def with_custom_first(
    section: dict[Any, Any], cls: type[Bomb], target: str, owned: str
) -> tuple[dict[Any, Any], list[Any], list[Any], Bomb]:
    custom_key = cls(target)
    ordinary_items = [
        (key, value) for key, value in section.items() if type(key) is str and key != owned
    ]
    if owned in section:
        kept_items = ordinary_items[:-1]
    else:
        kept_items = ordinary_items[:-2]
    crafted: dict[Any, Any] = {custom_key: "sentinel"}
    for key, value in kept_items:
        crafted[key] = clone_plain(value)
    crafted[owned] = False if owned == "can_build_desktop_exe_packaging_source_inventory" else True
    assert len(crafted) == len(section)
    assert next(iter(crafted)) is custom_key
    assert list(crafted.keys())[-1] == owned
    keys_snapshot = list(crafted.keys())
    values_snapshot = copy.deepcopy(list(crafted.values()))
    return crafted, keys_snapshot, values_snapshot, custom_key


def section_at(source: dict[str, Any], section_path: list[str]) -> dict[Any, Any]:
    section: Any = source
    for part in section_path:
        section = section[part]
    assert type(section) is dict
    return section


def assert_custom_section_unchanged(
    source: dict[str, Any],
    section_path: list[str],
    section_object: dict[Any, Any],
    keys_snapshot: list[Any],
    values_snapshot: list[Any],
    custom_key_object: Bomb,
) -> None:
    current = section_at(source, section_path)
    assert current is section_object
    current_keys = list(current.keys())
    assert len(current_keys) == len(keys_snapshot)
    assert current_keys[0] is custom_key_object
    assert keys_snapshot[0] is custom_key_object
    assert current_keys[-1] == keys_snapshot[-1]
    assert type(current_keys[-1]) is str
    for left, right in zip(current_keys[1:], keys_snapshot[1:], strict=True):
        assert type(left) is str
        assert left == right
    current_values = list(current.values())
    assert len(current_values) == len(values_snapshot)
    assert current_values == values_snapshot


@pytest.mark.parametrize("cls", [Bomb, Lying])
def test_custom_first_shadowing_matrix_direct_and_builder(
    monkeypatch: pytest.MonkeyPatch, cls: type[Bomb]
) -> None:
    for field_list, section_path, validity_key in [
        (
            inv.INVARIANT_OWNED_FIELDS_18_1[:3],
            ["inherited_block_o_closure_summary", "invariant_closure_audit_state"],
            "inherited_closure_valid",
        ),
        (
            inv.EXE_OWNED_FIELDS_18_1[:3],
            ["inherited_block_o_closure_summary", "exe_closure_audit_state"],
            "inherited_closure_valid",
        ),
        (inv.SOURCE_BOUNDARY_FIELDS_18_1, ["source_boundaries"], "source_boundaries_valid"),
    ]:
        for owned in field_list:
            source = clone_plain(inv.EXPECTED_SOURCE)
            section = source
            for part in section_path[:-1]:
                section = section[part]
            original = section[section_path[-1]]
            cls.armed = False
            crafted, keys_snapshot, values_snapshot, custom_key = with_custom_first(
                original, cls, owned, owned
            )
            section[section_path[-1]] = crafted
            cls.equality_calls = 0
            cls.armed = True
            try:
                assert inv._no_shadowing(source) is False
                payload = blocked_from_source(monkeypatch, source)
                assert payload["non_execution_inventory_evidence"][validity_key] is False
                assert payload["non_execution_inventory_evidence"]["summary_valid"] is True
                assert cls.equality_calls == 0
                assert_custom_section_unchanged(
                    source, section_path, crafted, keys_snapshot, values_snapshot, custom_key
                )
            finally:
                cls.armed = False


@pytest.mark.parametrize("field", list(inv.SOURCE_IDENTITY_EXPECTED))
@pytest.mark.parametrize("cls", [BombValue, LyingValue])
def test_each_custom_identity_value_blocks_without_custom_equality(
    monkeypatch: pytest.MonkeyPatch, field: str, cls: type[Any]
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    source[field] = cls()
    cls.equality_calls = 0
    payload = blocked_from_source(monkeypatch, source)
    assert payload["source_inventory_artifact_complete"] is False
    assert cls.equality_calls == 0


def deep_dict(depth: int) -> dict[str, Any]:
    root: dict[str, Any] = {}
    current = root
    for _ in range(depth):
        child: dict[str, Any] = {}
        current["x"] = child
        current = child
    return root


def deep_list(depth: int) -> list[Any]:
    value: list[Any] = []
    for _ in range(depth):
        value = [value]
    return value


def assert_blocked_plain(payload: dict[str, Any]) -> None:
    assert payload["source_inventory_artifact_complete"] is False
    assert payload["ready_for_block_p_2"] is False
    assert payload["status"] == inv.BLOCKED_STATUS
    assert inv._all_plain_json(payload, 2000) is True
    json.dumps(payload)


REFERENCE_MUTATIONS: list[tuple[str, list[Any], Any]] = [
    ("empty reference", ["block_o_closure_audit_reference"], {}),
    ("missing reference field", ["block_o_closure_audit_reference", "schema_version"], "delete"),
    ("extra reference field", ["block_o_closure_audit_reference", "extra"], "extra"),
    ("reordered reference fields", ["block_o_closure_audit_reference"], "reorder"),
    ("changed reference schema", ["block_o_closure_audit_reference", "schema_version"], "changed"),
    (
        "changed reference kind",
        ["block_o_closure_audit_reference", "block_o_closure_audit_kind"],
        "changed",
    ),
    ("changed reference block", ["block_o_closure_audit_reference", "block"], "changed"),
    ("changed reference step", ["block_o_closure_audit_reference", "step"], "changed"),
    ("changed reference status", ["block_o_closure_audit_reference", "status"], "changed"),
    (
        "changed source step",
        ["block_o_closure_audit_reference", "source_block_o_closure_audit_step"],
        "changed",
    ),
    (
        "changed source-read flag",
        ["block_o_closure_audit_reference", "source_block_o_closure_audit_read_by_18_0"],
        False,
    ),
    (
        "changed static-only flag",
        ["block_o_closure_audit_reference", "static_block_p_entry_contract_only"],
        False,
    ),
    (
        "changed Block P opened flag",
        ["block_o_closure_audit_reference", "block_p_opened_by_18_0"],
        False,
    ),
    (
        "readiness flag changed",
        ["block_o_closure_audit_reference", "ready_for_functional_preview_18_1"],
        False,
    ),
    (
        "operational false changed",
        ["block_o_closure_audit_reference", "build_executed_by_18_0"],
        True,
    ),
]


def mutate_reference_path(source: dict[str, Any], path: list[Any], value: Any) -> None:
    target: Any = source
    for part in path[:-1]:
        target = target[part]
    last = path[-1]
    if value == "delete":
        del target[last]
    elif value == "extra":
        target[last] = True
    elif value == "reorder":
        target[last] = dict(reversed(list(target[last].items())))
    else:
        target[last] = value


@pytest.mark.parametrize(("case", "path", "value"), REFERENCE_MUTATIONS)
def test_block_o_closure_reference_mutations_block(
    monkeypatch: pytest.MonkeyPatch, case: str, path: list[Any], value: Any
) -> None:
    del case
    source = clone_plain(inv.EXPECTED_SOURCE)
    mutate_reference_path(source, path, value)
    payload = blocked_from_source(monkeypatch, source)
    evidence = payload["non_execution_inventory_evidence"]
    assert evidence["identity_valid"] is True
    assert evidence["block_o_closure_reference_valid"] is False


def test_block_o_closure_reference_subclass_cycle_and_depth_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DictSubclass(dict[str, Any]):
        pass

    cases: list[Any] = [DictSubclass(inv.EXPECTED_SOURCE["block_o_closure_audit_reference"])]
    cyclic: dict[str, Any] = {}
    cyclic["self"] = cyclic
    cases.append(cyclic)
    cases.append(deep_dict(1500))
    for bad_reference in cases:
        source = clone_plain(inv.EXPECTED_SOURCE)
        source["block_o_closure_audit_reference"] = bad_reference
        payload = blocked_from_source(monkeypatch, source)
        assert payload["non_execution_inventory_evidence"]["identity_valid"] is True
        assert (
            payload["non_execution_inventory_evidence"]["block_o_closure_reference_valid"] is False
        )


CAPABILITY_BAD_VALUES: list[tuple[str, Any]] = [
    ("cyclic dict", None),
    ("cyclic nested list", None),
    ("deep dict", "deep_dict"),
    ("deep list", "deep_list"),
    ("dict subclass", "dict_subclass"),
    ("list subclass", "list_subclass"),
    ("float", 1.0),
    ("custom object", object()),
    ("callable", lambda: None),
    ("set", {"x"}),
    ("tuple", ("x",)),
]


def bad_capability_value(kind: str, fallback: Any) -> Any:
    if kind == "cyclic dict":
        value: dict[str, Any] = {}
        value["self"] = value
        return value
    if kind == "cyclic nested list":
        value_list: list[Any] = []
        value_list.append(value_list)
        return {"nested": value_list}
    if fallback == "deep_dict":
        return deep_dict(1500)
    if fallback == "deep_list":
        return {"nested": deep_list(1500)}
    if fallback == "dict_subclass":

        class DictSubclass(dict[str, Any]):
            pass

        return DictSubclass({"x": "blocked"})
    if fallback == "list_subclass":

        class ListSubclass(list[Any]):
            pass

        return {"nested": ListSubclass(["x"])}
    return fallback


@pytest.mark.parametrize("map_name", ["inherited_block_o_capabilities", "block_p_capabilities"])
@pytest.mark.parametrize(("case", "value"), CAPABILITY_BAD_VALUES)
def test_builder_level_bad_capability_maps_do_not_recurse(
    monkeypatch: pytest.MonkeyPatch, map_name: str, case: str, value: Any
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    bad_value = bad_capability_value(case, value)
    source["real_capability_entry_state"][map_name] = bad_value
    payload = blocked_from_source(monkeypatch, source)
    state = payload["real_capability_inventory_state"]
    assert payload["non_execution_inventory_evidence"]["real_capability_valid"] is False
    assert state["inherited_18_0_capabilities"] == {
        "inherited_block_o_capabilities": {},
        "block_p_capabilities": {},
    }
    assert state["inherited_block_o_capabilities_known_blocked"] is False
    assert state["inherited_block_p_capabilities_known_blocked"] is False
    assert state["inherited_18_0_capabilities_known_blocked"] is False
    assert state["all_real_capabilities_blocked_at_18_1"] is False
    assert state["inventory_capabilities"]
    assert set(state["inventory_capabilities"].values()) == {"blocked"}


@pytest.mark.parametrize(
    "section_name",
    [
        "block_o_closure_audit_reference",
        "entry_contract_summary",
        "inherited_block_o_closure_summary",
        "desktop_exe_product_direction",
        "desktop_exe_packaging_scope_contract",
        "desktop_exe_packaging_requirement_rows",
        "fail_closed_entry_decision",
        "non_execution_entry_evidence",
        "entry_contract_boundaries",
        "source_boundaries",
        "future_steps",
    ],
)
@pytest.mark.parametrize("bad_kind", ["cycle", "deep"])
def test_builder_level_cycle_depth_matrix_for_sections(
    monkeypatch: pytest.MonkeyPatch, section_name: str, bad_kind: str
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    if bad_kind == "cycle":
        if type(inv.EXPECTED_SOURCE[section_name]) is list:
            bad_list: list[Any] = []
            bad_list.append(bad_list)
            source[section_name] = bad_list
        else:
            bad_dict: dict[str, Any] = {}
            bad_dict["self"] = bad_dict
            source[section_name] = bad_dict
    else:
        source[section_name] = (
            deep_list(1500) if type(inv.EXPECTED_SOURCE[section_name]) is list else deep_dict(1500)
        )
    payload = blocked_from_source(monkeypatch, source)
    assert payload["non_execution_inventory_evidence"]["source_accepted"] is False


def custom_section_fixture(
    cls: type[Bomb], section_path: list[str], owned: str
) -> tuple[dict[str, Any], dict[Any, Any], list[Any], list[Any], Bomb]:
    source = clone_plain(inv.EXPECTED_SOURCE)
    section = source
    for part in section_path[:-1]:
        section = section[part]
    crafted, keys_snapshot, values_snapshot, custom_key = with_custom_first(
        section[section_path[-1]], cls, owned, owned
    )
    section[section_path[-1]] = crafted
    return source, crafted, keys_snapshot, values_snapshot, custom_key


@pytest.mark.parametrize("cls", [Bomb, Lying])
def test_custom_key_direct_matrix_separate(cls: type[Bomb]) -> None:
    section_path = ["inherited_block_o_closure_summary", "invariant_closure_audit_state"]
    source, crafted, keys_snapshot, values_snapshot, custom_key = custom_section_fixture(
        cls,
        section_path,
        inv.INVARIANT_OWNED_FIELDS_18_1[0],
    )
    cls.armed = False
    cls.equality_calls = 0
    cls.armed = True
    try:
        assert inv._no_shadowing(source) is False
        assert cls.equality_calls == 0
        assert_custom_section_unchanged(
            source,
            section_path,
            crafted,
            keys_snapshot,
            values_snapshot,
            custom_key,
        )
    finally:
        cls.armed = False


@pytest.mark.parametrize("cls", [Bomb, Lying])
def test_custom_key_builder_matrix_separate(
    monkeypatch: pytest.MonkeyPatch, cls: type[Bomb]
) -> None:
    source, crafted, keys_snapshot, values_snapshot, custom_key = custom_section_fixture(
        cls,
        ["inherited_block_o_closure_summary", "exe_closure_audit_state"],
        inv.EXE_OWNED_FIELDS_18_1[0],
    )
    cls.armed = False
    cls.equality_calls = 0
    cls.armed = True
    try:
        payload = blocked_from_source(monkeypatch, source)
        assert payload["non_execution_inventory_evidence"]["inherited_closure_valid"] is False
        assert payload["non_execution_inventory_evidence"]["summary_valid"] is True
        assert cls.equality_calls == 0
        assert_custom_section_unchanged(
            source,
            ["inherited_block_o_closure_summary", "exe_closure_audit_state"],
            crafted,
            keys_snapshot,
            values_snapshot,
            custom_key,
        )
    finally:
        cls.armed = False


MUTATION_SENSITIVITY_CASES = [
    "nested list append",
    "nested dict insertion",
    "scalar replacement",
    "key order mutation",
    "key deletion",
    "key addition",
    "owned field replacement",
    "custom key replacement",
]


@pytest.mark.parametrize("mutation", MUTATION_SENSITIVITY_CASES)
def test_custom_section_no_mutation_helper_sensitivity(mutation: str) -> None:
    section_path = ["inherited_block_o_closure_summary", "invariant_closure_audit_state"]
    owned = inv.INVARIANT_OWNED_FIELDS_18_1[0]
    if mutation == "nested dict insertion":
        section_path = ["inherited_block_o_closure_summary"]
        owned = "source_inventory_artifact_complete"
    source, crafted, keys_snapshot, values_snapshot, custom_key = custom_section_fixture(
        Bomb,
        section_path,
        owned,
    )
    if mutation == "nested list append":
        list_key = next(key for key, value in crafted.items() if type(value) is list)
        crafted[list_key].append("mutation")
    elif mutation == "nested dict insertion":
        dict_key = next(key for key, value in crafted.items() if type(value) is dict)
        crafted[dict_key]["mutation"] = True
    elif mutation == "scalar replacement":
        scalar_key = next(
            key for key, value in crafted.items() if type(key) is str and type(value) is bool
        )
        crafted[scalar_key] = not crafted[scalar_key]
    elif mutation == "key order mutation":
        items = list(crafted.items())
        crafted.clear()
        for key, value in reversed(items):
            crafted[key] = value
    elif mutation == "key deletion":
        crafted.pop(next(key for key in crafted if type(key) is str))
    elif mutation == "key addition":
        crafted["new_key"] = True
    elif mutation == "owned field replacement":
        crafted[inv.INVARIANT_OWNED_FIELDS_18_1[0]] = "changed"
    else:
        items = list(crafted.items())
        crafted.clear()
        crafted[Bomb(inv.INVARIANT_OWNED_FIELDS_18_1[0])] = "sentinel"
        for key, value in items[1:]:
            crafted[key] = value
    with pytest.raises(AssertionError):
        assert_custom_section_unchanged(
            source,
            section_path,
            crafted,
            keys_snapshot,
            values_snapshot,
            custom_key,
        )


@pytest.mark.parametrize("target", ["schema_version", "entry_contract_summary"])
@pytest.mark.parametrize("cls", [Bomb, Lying])
def test_top_level_bomb_lying_keys(
    monkeypatch: pytest.MonkeyPatch, target: str, cls: type[Bomb]
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    custom_key = cls(target)
    crafted: dict[Any, Any] = {custom_key: "sentinel"}
    for key, value in source.items():
        if key != target:
            crafted[key] = value
    crafted[target] = source[target]
    keys_snapshot = list(crafted.keys())
    cls.armed = False
    cls.equality_calls = 0
    cls.armed = True
    try:
        payload = blocked_from_source(monkeypatch, crafted)
        assert_blocked_plain(payload)
        assert cls.equality_calls == 0
        assert next(iter(crafted)) is custom_key
        assert list(crafted.keys()) == keys_snapshot
    finally:
        cls.armed = False


@pytest.mark.parametrize("field", list(inv.SOURCE_IDENTITY_EXPECTED))
@pytest.mark.parametrize("cls", [BombValue, LyingValue])
def test_each_custom_identity_value_preserves_source_without_custom_equality(
    monkeypatch: pytest.MonkeyPatch, field: str, cls: type[Any]
) -> None:
    source = clone_plain(inv.EXPECTED_SOURCE)
    custom_value = cls()
    source[field] = custom_value
    top_keys = list(source.keys())
    plain_others = {key: clone_plain(value) for key, value in source.items() if key != field}
    cls.equality_calls = 0
    payload = blocked_from_source(monkeypatch, source)
    assert payload["source_inventory_artifact_complete"] is False
    assert source[field] is custom_value
    assert list(source.keys()) == top_keys
    for key, value in plain_others.items():
        assert source[key] == value
    assert cls.equality_calls == 0


def test_capability_map_no_aliasing_mutations() -> None:
    first = build()
    second = build()
    state = first["real_capability_inventory_state"]
    inherited_o = state["inherited_18_0_capabilities"]["inherited_block_o_capabilities"]
    inherited_p = state["inherited_18_0_capabilities"]["block_p_capabilities"]
    inventory = state["inventory_capabilities"]
    inherited_o["mutated"] = "open"
    inherited_p["mutated"] = "ready"
    inventory["mutated"] = "authorized"
    second_state = second["real_capability_inventory_state"]
    assert (
        "mutated"
        not in second_state["inherited_18_0_capabilities"]["inherited_block_o_capabilities"]
    )
    assert "mutated" not in second_state["inherited_18_0_capabilities"]["block_p_capabilities"]
    assert "mutated" not in second_state["inventory_capabilities"]
    assert (
        "mutated"
        not in inv.EXPECTED_SOURCE["real_capability_entry_state"]["inherited_block_o_capabilities"]
    )
    assert (
        "mutated" not in inv.EXPECTED_SOURCE["real_capability_entry_state"]["block_p_capabilities"]
    )
