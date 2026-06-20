"""Source-only tests for BLOK D closure and BLOK E controlled wiring start."""

from __future__ import annotations

import importlib
import json
from dataclasses import is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ui.pyside_app.preview_block_d_closure_audit import (
    AUDIT_KIND,
    BLOCK_STATUS,
    CLOSURE_DECISION,
    RECOMMENDED_FUTURE_INTEGRATION_POINT,
    SCHEMA_VERSION,
    build_preview_block_d_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ANTI_DUPLICATION_AUDIT = (
    REPO_ROOT / "docs" / "functional_preview" / "block_d_anti_duplication_integration_audit.md"
)
QML_BRIDGE = REPO_ROOT / "ui" / "pyside_app" / "qml_bridge.py"
APP = REPO_ROOT / "ui" / "pyside_app" / "app.py"
QML_FILES = (
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml",
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml",
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "PaperTerminal.qml",
)
BAT_LAUNCHERS = (
    REPO_ROOT / "run_ui_preview_visible_doubleclick.bat",
    REPO_ROOT / "scripts" / "windows" / "run_ui_preview_visible.bat",
)
PIPELINE_MODULE_NAMES = (
    "ui.pyside_app.preview_action_dispatch_contract",
    "ui.pyside_app.preview_action_dispatch_audit",
    "ui.pyside_app.preview_action_dispatch_catalog",
    "ui.pyside_app.preview_action_dispatch_selection",
    "ui.pyside_app.preview_action_dispatch_bridge_snapshot",
    "ui.pyside_app.preview_action_dispatch_bridge_provider",
    "ui.pyside_app.preview_action_dispatch_qt_bridge",
    "ui.pyside_app.preview_action_dispatch_qt_bridge_registration",
)
FALSE_BOUNDARY_FLAGS = (
    "execution_allowed",
    "execution_performed",
    "lifecycle_execution_allowed",
    "command_dispatch_allowed",
    "order_generation_allowed",
    "order_submission_allowed",
    "live_mode_allowed",
    "testnet_mode_allowed",
    "account_fetch_allowed",
    "secrets_allowed",
    "export_cloud_allowed",
)
SIMPLE_TYPES = (dict, list, str, bool, int, type(None))


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    assert not is_dataclass(value)
    assert not isinstance(value, MappingProxyType)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _source_files_under(*roots: Path) -> tuple[Path, ...]:
    source_suffixes = {".bat", ".json", ".md", ".py", ".qml", ".txt", ".yaml", ".yml"}
    return tuple(
        sorted(
            path
            for root in roots
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in source_suffixes
        )
    )


def _paths_containing(pattern: str, *roots: Path) -> set[str]:
    return {
        path.relative_to(REPO_ROOT).as_posix()
        for path in _source_files_under(*roots)
        if pattern in _source(path)
    }


def test_closure_audit_returns_expected_schema_kind_block_and_status() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["schema_version"] == SCHEMA_VERSION
    assert evidence["audit_kind"] == AUDIT_KIND
    assert evidence["block"] == "D"
    assert evidence["block_status"] == BLOCK_STATUS
    assert evidence["runtime_mode"] == "paper"
    assert evidence["paper_only"] is True
    assert evidence["local_only"] is True


def test_closure_decision_is_exact_close_block_d_bridge_ready_not_wired() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["closure_decision"] == CLOSURE_DECISION
    assert evidence["closure_decision"] == "CLOSE_BLOCK_D_AS_BRIDGE_READY_NOT_WIRED"
    assert evidence["ready_for_block_e"] is True


def test_bridge_is_ready_and_block_e_wired_to_central_context() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["bridge_ready"] is True
    assert evidence["block_e_wiring_started"] is True
    assert evidence["bridge_wired_to_real_startup"] is True
    assert evidence["boundary_checks"]["bridge_ready"] is True
    assert evidence["boundary_checks"]["block_e_wiring_started"] is True
    assert evidence["boundary_checks"]["bridge_registered_in_central_context"] is True


def test_qml_does_not_consume_bridge_and_real_context_property_registered_once() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["qml_consumes_bridge"] is True
    assert evidence["real_startup_context_property_registered"] is True
    assert evidence["registered_context_property_name"] == "paperRuntimeActionDispatchBridge"
    assert evidence["execution_still_disabled_after_wiring"] is True
    assert evidence["boundary_checks"]["real_context_property_registered_once"] is True


def test_recommended_integration_point_is_qml_context_bridge_install() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["recommended_future_integration_point"] == RECOMMENDED_FUTURE_INTEGRATION_POINT
    assert (
        evidence["recommended_future_integration_point"]
        == "ui/pyside_app/qml_bridge.py::QmlContextBridge.install()"
    )


def test_launch_path_preserved_and_anti_duplication_audit_exists() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert evidence["launch_path_preserved"] is True
    assert evidence["anti_duplication_audit_present"] is True
    assert ANTI_DUPLICATION_AUDIT.is_file()


def test_all_block_d_pipeline_modules_exist_and_import() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    assert all(evidence["pipeline_modules_present"].values())
    for module_name in PIPELINE_MODULE_NAMES:
        module = importlib.import_module(module_name)
        assert module is not None


def test_evidence_is_deterministic_copy_safe_json_serializable_and_simple() -> None:
    first = build_preview_block_d_closure_audit(REPO_ROOT)
    second = build_preview_block_d_closure_audit(REPO_ROOT)

    assert first == second
    first["boundary_checks"]["execution_disabled"] = False
    first["next_block_gate"]["forbidden_integration_points"].clear()
    reread = build_preview_block_d_closure_audit(REPO_ROOT)
    assert reread == second
    assert reread["boundary_checks"]["execution_disabled"] is True
    _assert_simple_types_only(reread)
    assert json.loads(json.dumps(reread, sort_keys=True)) == reread


def test_execution_live_testnet_account_secrets_export_flags_are_false() -> None:
    evidence = build_preview_block_d_closure_audit(REPO_ROOT)

    for flag in FALSE_BOUNDARY_FLAGS:
        assert evidence[flag] is False


def test_next_block_gate_has_one_required_integration_point_and_forbidden_points() -> None:
    gate = build_preview_block_d_closure_audit(REPO_ROOT)["next_block_gate"]

    assert gate["allowed_next_scope"] == "read_only_qml_snapshot_consumption"
    assert gate["required_integration_point"] == RECOMMENDED_FUTURE_INTEGRATION_POINT
    assert list(gate).count("required_integration_point") == 1
    forbidden = gate["forbidden_integration_points"]
    for expected in (
        ".bat launchers",
        "ui/pyside_app/app.py",
        "ui/pyside_app/qml/MainWindow.qml",
        "ui/pyside_app/qml/views/PaperTerminal.qml",
        "ui/pyside_app/preview_state_bridge.py",
        "runtime/order/trading/live/testnet/account/secrets/export paths",
    ):
        assert expected in forbidden


def test_controlled_wiring_source_guard_for_qml_bridge_and_real_startup() -> None:
    qml_bridge_source = _source(QML_BRIDGE)
    app_source = _source(APP)

    assert qml_bridge_source.count("paperRuntimeActionDispatchBridge") == 0
    assert "preview_action_dispatch_qt_bridge_registration" in qml_bridge_source
    assert qml_bridge_source.count("register_paper_runtime_action_dispatch_qt_bridge") == 2
    assert "preview_action_dispatch_qt_bridge_registration" not in app_source
    assert "register_paper_runtime_action_dispatch_qt_bridge" not in app_source


def test_qml_bridge_property_is_limited_to_operator_dashboard_read_only_snapshot() -> None:
    qml_consumers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in QML_FILES
        if "paperRuntimeActionDispatchBridge" in _source(path)
    ]

    assert qml_consumers == ["ui/pyside_app/qml/views/OperatorDashboard.qml"]
    operator_source = _source(
        REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
    )
    assert "paperRuntimeActionDispatchBridge.snapshot" in operator_source
    for path in BAT_LAUNCHERS:
        assert "paperRuntimeActionDispatchBridge" not in _source(path)


def test_context_property_constant_limited_to_allowed_block_d_sources() -> None:
    paths = _paths_containing(
        "PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY",
        REPO_ROOT / "ui" / "pyside_app",
        REPO_ROOT / "tests",
        REPO_ROOT / "docs",
    )

    assert paths <= {
        "ui/pyside_app/preview_action_dispatch_qt_bridge_registration.py",
        "tests/ui_pyside/test_preview_action_dispatch_qt_bridge_registration.py",
        "tests/ui_pyside/test_preview_block_d_closure_audit.py",
        "tests/ui_pyside/test_qml_context_bridge_action_dispatch_wiring.py",
        "ui/pyside_app/qml_bridge.py",
    }
