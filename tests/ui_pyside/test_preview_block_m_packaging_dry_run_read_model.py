"""Tests for FUNCTIONAL-PREVIEW-15.5 Block M packaging dry-run read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_dry_run_contract import (
    build_preview_block_m_packaging_dry_run_contract,
)
from ui.pyside_app.preview_block_m_packaging_dry_run_read_model import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_DECISION,
    BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_KIND,
    PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_6,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_dry_run_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_dry_run_read_model.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_dry_run_read_model_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_read_model_status",
    "block_m_packaging_dry_run_read_model_decision",
    "ready_for_block_m_6",
    "next_step",
    "next_step_title",
    "packaging_dry_run_contract_reference",
    "packaging_dry_run_read_summary",
    "dry_run_prerequisite_read_rows",
    "dry_run_execution_read_model",
    "dry_run_simulation_read_rows",
    "dry_run_artifact_policy_read_model",
    "runtime_safety_carryover_read_rows",
    "exe_direction_dry_run_read_model",
    "fail_closed_dry_run_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_TRUE_FLAGS = [
    "packaging_dry_run_contract_available",
    "packaging_dry_run_read_model_built",
    "ready_for_block_m_6",
    "exe_direction_preserved",
    "dry_run_read_model_static_only",
    "dry_run_read_model_ready_for_future_matrix",
    "dry_run_contract_read_only",
]
SUMMARY_FALSE_FLAGS = [
    "dry_run_satisfied_now",
    "dry_run_can_execute_now",
    "packaging_ready_now",
    "packaging_can_execute_now",
    "pyinstaller_can_start_now",
    "build_command_can_execute_now",
    "build_artifact_can_be_created_now",
    "installer_can_change_now",
    "release_workflow_can_change_now",
    "artifact_smoke_test_can_run_now",
    "artifact_signing_can_run_now",
    "artifact_publishing_can_run_now",
    "dependency_freeze_can_run_now",
    "asset_discovery_can_run_now",
    "qml_asset_discovery_can_run_now",
    "packaging_filesystem_io_allowed_now",
    "safe_to_activate_runtime_now",
    "safe_to_start_paper_runtime_now",
    "safe_to_start_testnet_runtime_now",
    "safe_to_start_live_canary_now",
    "safe_to_enable_live_trading_now",
    "safe_to_generate_orders_now",
    "safe_to_" + "sub" + "mit_orders_now",
    "safe_to_" + "can" + "cel_orders_now",
    "safe_to_" + "re" + "place_orders_now",
    "safe_to_access_private_endpoints_now",
    "safe_to_open_network_io_now",
    "safe_to_read_credentials_now",
    "safe_for_filesystem_io_now",
    "safe_for_config_env_secrets_now",
    "safe_to_change_qml_bridge_now",
]
RUNTIME_CAPABILITY_IDS = {
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gate_execution",
    "gate_state_mutation",
    "order_generation",
    "order_submission",
    "order_cancel",
    "order_replace",
    "private_endpoints",
    "network_io",
    "credential_read",
    "filesystem_io",
    "config_env_secrets",
    "qml_bridge",
}
BLOCKED_DECISION_KEYS = [
    "packaging_dry_run_execution_in_15_5",
    "packaging_execution_in_15_5",
    "pyinstaller_execution_in_15_5",
    "build_command_execution_in_15_5",
    "build_artifact_creation_in_15_5",
    "installer_change_in_15_5",
    "release_workflow_change_in_15_5",
    "artifact_smoke_test_in_15_5",
    "artifact_signing_in_15_5",
    "artifact_publishing_in_15_5",
    "packaging_filesystem_io_in_15_5",
    "packaging_environment_probe_in_15_5",
    "dependency_freeze_in_15_5",
    "asset_discovery_in_15_5",
    "qml_asset_discovery_in_15_5",
    "runtime_activation_in_15_5",
    "paper_runtime_start_in_15_5",
    "testnet_runtime_start_in_15_5",
    "live_canary_start_in_15_5",
    "live_trading_in_15_5",
    "order_generation_in_15_5",
    "order_submission_in_15_5",
    "order_cancel_in_15_5",
    "order_replace_in_15_5",
    "private_endpoint_in_15_5",
    "network_io_in_15_5",
    "credential_read_in_15_5",
    "config_env_secret_read_in_15_5",
    "qml_bridge_change_in_15_5",
]
NON_EXECUTION_FALSE_FLAGS = [
    "packaging_dry_run_executed",
    "packaging_executed",
    "runtime_activated",
    "paper_runtime_started",
    "testnet_runtime_started",
    "live_canary_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "gate_state_mutated",
    "mode_activated",
    "order_generated",
    "order_submitted",
    "private_endpoint_accessed",
    "network_io_performed",
    "filesystem_io_performed",
    "credential_read_performed",
    "live_trading_started",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "installer_changed",
    "release_workflow_changed",
    "artifact_smoke_test_executed",
    "artifact_signed",
    "artifact_published",
    "dependency_freeze_performed",
    "asset_discovery_performed",
    "qml_asset_discovery_performed",
    "qml_bridge_changed",
]
BOUNDARY_TRUE_FLAGS = [
    "packaging_dry_run_read_model_is_plain_data_only",
    "packaging_dry_run_read_model_is_source_only",
    "packaging_dry_run_read_model_reads_dry_run_contract_only",
    "packaging_dry_run_read_model_preserves_exe_direction_without_packaging",
    "packaging_dry_run_read_model_can_feed_15_6_packaging_artifact_policy_matrix",
    "packaging_dry_run_read_model_cannot_execute_dry_run",
    "packaging_dry_run_read_model_cannot_package_exe",
    "packaging_dry_run_read_model_cannot_start_pyinstaller",
    "packaging_dry_run_read_model_cannot_execute_build_commands",
    "packaging_dry_run_read_model_cannot_create_build_artifacts",
    "packaging_dry_run_read_model_cannot_change_installers",
    "packaging_dry_run_read_model_cannot_change_release_workflows",
    "packaging_dry_run_read_model_cannot_run_artifact_smoke_tests",
    "packaging_dry_run_read_model_cannot_sign_artifacts",
    "packaging_dry_run_read_model_cannot_publish_artifacts",
    "packaging_dry_run_read_model_cannot_probe_packaging_environment",
    "packaging_dry_run_read_model_cannot_freeze_dependencies",
    "packaging_dry_run_read_model_cannot_discover_assets",
    "packaging_dry_run_read_model_cannot_discover_qml_assets",
    "packaging_dry_run_read_model_cannot_perform_filesystem_io",
    "packaging_dry_run_read_model_cannot_activate_runtime",
    "packaging_dry_run_read_model_cannot_start_paper_runtime",
    "packaging_dry_run_read_model_cannot_start_testnet_runtime",
    "packaging_dry_run_read_model_cannot_start_live_canary",
    "packaging_dry_run_read_model_cannot_enable_live_trading",
    "packaging_dry_run_read_model_cannot_generate_orders",
    "packaging_dry_run_read_model_cannot_" + "sub" + "mit_orders",
    "packaging_dry_run_read_model_cannot_" + "can" + "cel_orders",
    "packaging_dry_run_read_model_cannot_" + "re" + "place_orders",
    "packaging_dry_run_read_model_cannot_access_private_endpoints",
    "packaging_dry_run_read_model_cannot_open_network_io",
    "packaging_dry_run_read_model_cannot_read_credentials",
    "packaging_dry_run_read_model_cannot_start_runtime_loop",
    "packaging_dry_run_read_model_cannot_execute_runtime_gates",
    "packaging_dry_run_read_model_cannot_mutate_gate_state",
    "packaging_dry_run_read_model_cannot_read_config_env_or_secrets",
    "packaging_dry_run_read_model_cannot_change_ui_bridge",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_m_packaging_dry_run_read_model()


def test_payload_is_json_serializable_and_top_level_fields_are_stable() -> None:
    payload = _payload()
    json.dumps(payload, sort_keys=True)
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_5() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_SCHEMA_VERSION
    assert (
        payload["block_m_packaging_dry_run_read_model_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_KIND
    )
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.5"
    assert (
        payload["block_m_packaging_dry_run_read_model_status"]
        == BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_STATUS
    )
    assert (
        payload["block_m_packaging_dry_run_read_model_decision"]
        == BLOCK_M_PACKAGING_DRY_RUN_READ_MODEL_DECISION
    )
    assert payload["ready_for_block_m_6"] is READY_FOR_BLOCK_M_6 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.6"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING ARTIFACT POLICY MATRIX"
    assert payload["status"] == STATUS


def test_packaging_dry_run_contract_reference_points_to_15_4() -> None:
    payload = _payload()
    contract = build_preview_block_m_packaging_dry_run_contract()
    reference = payload["packaging_dry_run_contract_reference"]
    for key in [
        "schema_version",
        "block_m_packaging_dry_run_contract_kind",
        "block",
        "step",
        "block_m_packaging_dry_run_contract_status",
        "block_m_packaging_dry_run_contract_decision",
        "ready_for_block_m_5",
        "next_step",
        "next_step_title",
    ]:
        assert reference[key] == contract[key]
    assert reference["source_packaging_dry_run_contract_step"] == "FUNCTIONAL-PREVIEW-15.4"
    assert reference["source_packaging_dry_run_contract_read_by_15_5_read_model"] is True
    assert reference["packaging_dry_run_contract_available_before_read_model"] is True
    assert reference["static_packaging_dry_run_contract_only"] is True
    for key, value in reference.items():
        if (
            key.endswith("_by_15_5")
            or key.endswith("_opened_by_15_5")
            or key.endswith("_accessed_by_15_5")
            or key.endswith("_performed_by_15_5")
        ):
            assert value is False


def test_summary_preserves_exe_direction_and_blocks_unsafe_paths() -> None:
    summary = _payload()["packaging_dry_run_read_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_prerequisite_read_rows_are_unsatisfied_15_4_derivatives() -> None:
    payload = _payload()
    contract_rows = build_preview_block_m_packaging_dry_run_contract()[
        "dry_run_prerequisite_contract"
    ]
    rows = payload["dry_run_prerequisite_read_rows"]
    assert len(rows) == len(contract_rows)
    for row, source in zip(rows, contract_rows, strict=True):
        assert row["source_id"] == source["source_id"]
        assert row["display_name"] == source["display_name"]
        assert row["notes"] == source["notes"]
        assert row["source_failure_policy"] == source["source_failure_policy"]
        assert row["read_row_type"] == "packaging_dry_run_static_prerequisite_read_row"
        assert row["source_required_before_dry_run"] is True
        assert row["source_satisfied_in_15_4"] is False
        assert row["source_checked_by_15_4"] is False
        assert row["required_before_future_dry_run"] is True
        assert row["satisfied_in_15_5"] is False
        assert row["checked_by_15_5"] is False
        assert row["read_by_15_5"] is True
        assert row["requires_future_step"] is True
        assert row["failure_policy"] == "fail_closed"


def test_execution_read_model_blocks_packaging_and_environment_actions() -> None:
    model = _payload()["dry_run_execution_read_model"]
    assert model["dry_run_execution_read_model_built"] is True
    assert model["source_dry_run_not_performed_by_15_4"] is True
    assert model["source_packaging_not_performed_by_15_4"] is True
    assert model["dry_run_requires_future_explicit_gate"] is True
    assert model["dry_run_requires_future_operator_confirmation"] is True
    assert model["dry_run_not_performed_by_15_5"] is True
    assert model["packaging_not_performed_by_15_5"] is True
    for key, value in model.items():
        if key.endswith("_allowed_now"):
            assert value is False


def test_simulation_read_rows_are_static_only() -> None:
    rows = _payload()["dry_run_simulation_read_rows"]
    source_rows = build_preview_block_m_packaging_dry_run_contract()[
        "dry_run_simulation_plan_contract"
    ]
    assert len(rows) == len(source_rows)
    for row, source in zip(rows, source_rows, strict=True):
        assert row["simulation_step_id"] == source["simulation_step_id"]
        assert row["display_name"] == source["display_name"]
        assert row["source_planned_for_future_dry_run"] is True
        assert row["source_executed_in_15_4"] is False
        assert row["source_allowed_now"] is False
        assert row["read_model_row_type"] == "packaging_dry_run_static_simulation_read_row"
        assert row["read_by_15_5"] is True
        assert row["planned_for_future_dry_run"] is True
        assert row["executed_in_15_5"] is False
        assert row["allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True


def test_artifact_policy_read_model_blocks_artifact_changes() -> None:
    model = _payload()["dry_run_artifact_policy_read_model"]
    assert model["artifact_policy_read_model_built"] is True
    assert model["source_no_artifact_created_by_15_4"] is True
    assert model["artifact_policy_requires_future_gate"] is True
    assert model["artifact_policy_requires_future_operator_confirmation"] is True
    assert model["no_artifact_created_by_15_5"] is True
    assert model["no_artifact_mutated_by_15_5"] is True
    assert model["no_artifact_deleted_by_15_5"] is True
    for key, value in model.items():
        if (
            key.endswith("_allowed_now")
            or key.endswith("_finalized_now")
            or key.endswith("_selected_now")
        ):
            assert value is False


def test_runtime_safety_carryover_rows_block_runtime_and_external_capabilities() -> None:
    rows = _payload()["runtime_safety_carryover_read_rows"]
    assert {row["capability_id"] for row in rows} == RUNTIME_CAPABILITY_IDS
    for row in rows:
        assert row["source_gate_allowed_now"] is False
        assert row["source_dry_run_contract_allowed_now"] is False
        assert row["source_dry_run_contract_executed_now"] is False
        assert row["read_model_allowed_now"] is False
        assert row["read_model_executed_now"] is False
        assert row["blocked_in_15_5"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_read_model_preserves_direction_without_starting_packaging() -> None:
    model = _payload()["exe_direction_dry_run_read_model"]
    assert model["final_product_direction"] == "desktop_exe"
    assert model["exe_direction_preserved"] is True
    assert model["dry_run_read_model_confirms_exe_direction"] is True
    true_flags = [
        "packaging_deferred_to_future_explicit_block",
        "dry_run_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_dry_run_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]
    for key in true_flags:
        assert model[key] is True
    for key, value in model.items():
        if key.endswith("_now"):
            assert value is False


def test_fail_closed_decision_blocks_all_15_5_forbidden_paths() -> None:
    decision = _payload()["fail_closed_dry_run_read_decision"]
    assert decision["missing_packaging_dry_run_contract_policy"] == "fail_closed"
    assert decision["missing_dry_run_read_row_policy"] == "fail_closed"
    assert decision["missing_operator_confirmation_policy"] == "fail_closed"
    assert decision["missing_runtime_safety_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_execution_evidence_remains_false_for_execution_paths() -> None:
    evidence = _payload()["non_execution_evidence"]
    assert evidence["source_packaging_dry_run_contract_read"] is True
    assert evidence["packaging_dry_run_read_model_built"] is True
    assert evidence["packaging_dry_run_read_model_only"] is True
    for key in NON_EXECUTION_FALSE_FLAGS:
        assert evidence[key] is False
    for key, value in evidence.items():
        if key.startswith("source_contract_"):
            assert value is False


def test_read_model_boundaries_are_closed() -> None:
    boundaries = _payload()["read_model_boundaries"]
    for key in BOUNDARY_TRUE_FLAGS:
        assert boundaries[key] is True
    for key, value in boundaries.items():
        if key.startswith("packaging_dry_run_read_model_cannot_"):
            assert value is True


def test_source_boundaries_point_to_15_4_contract() -> None:
    source = _payload()["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_packaging_dry_run_contract"] == "FUNCTIONAL-PREVIEW-15.4"
    for key in [
        "forbidden_packaging_calls_present",
        "forbidden_pyinstaller_calls_present",
        "forbidden_build_calls_present",
        "forbidden_runtime_calls_present",
        "forbidden_io_calls_present",
        "forbidden_network_calls_present",
        "forbidden_private_endpoint_calls_present",
        "forbidden_ui_bridge_calls_present",
    ]:
        assert source[key] is False
    nested = source["source_packaging_dry_run_contract_boundaries"]
    assert nested["allowed_imports_only"] is True
    assert nested["source_packaging_gate_contract"] == "FUNCTIONAL-PREVIEW-15.3"


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", tuple(alias.name for alias in node.names)))
    assert imports == [
        ("__future__", ("annotations",)),
        ("typing", ("Any", "Final")),
        (
            "ui.pyside_app.preview_block_m_packaging_dry_run_contract",
            ("build_preview_block_m_packaging_dry_run_contract",),
        ),
    ]


def test_source_call_guard_blocks_forbidden_execution_and_io_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_call_names = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "loads",
        "dumps",
        "request",
        "get",
        "post",
        "run",
        "Popen",
        "getaddrinfo",
        "create_connection",
        "TradingController",
        "DecisionEnvelope",
        "activate",
        "start",
        "execute",
        "mutate",
        "PyInstaller",
        "build",
    }
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.add(func.id)
            elif isinstance(func, ast.Attribute):
                calls.add(func.attr)
    assert forbidden_call_names.isdisjoint(calls)
    assert all(
        call in {"build_preview_block_m_packaging_dry_run_contract", "update"}
        or call.startswith("_build_")
        for call in calls
    )


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    for token in forbidden:
        assert token not in source
