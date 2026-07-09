"""Tests for FUNCTIONAL-PREVIEW-15.4 Block M packaging dry-run contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_dry_run_contract import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_DECISION,
    BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_KIND,
    PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_5,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_dry_run_contract,
)
from ui.pyside_app.preview_block_m_packaging_gate_contract import (
    build_preview_block_m_packaging_gate_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_dry_run_contract.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_dry_run_contract_kind",
    "block",
    "step",
    "block_m_packaging_dry_run_contract_status",
    "block_m_packaging_dry_run_contract_decision",
    "ready_for_block_m_5",
    "next_step",
    "next_step_title",
    "packaging_gate_contract_reference",
    "packaging_dry_run_summary",
    "dry_run_prerequisite_contract",
    "dry_run_execution_blocked_contract",
    "dry_run_simulation_plan_contract",
    "dry_run_artifact_policy_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_dry_run_contract",
    "fail_closed_dry_run_decision",
    "non_execution_evidence",
    "dry_run_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_TRUE_FLAGS = [
    "packaging_gate_contract_available",
    "packaging_dry_run_contract_built",
    "ready_for_block_m_5",
    "exe_direction_preserved",
    "dry_run_contract_static_only",
    "dry_run_ready_for_future_read_model",
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
    "packaging_dry_run_execution_in_15_4",
    "packaging_execution_in_15_4",
    "pyinstaller_execution_in_15_4",
    "build_command_execution_in_15_4",
    "build_artifact_creation_in_15_4",
    "installer_change_in_15_4",
    "release_workflow_change_in_15_4",
    "artifact_smoke_test_in_15_4",
    "artifact_signing_in_15_4",
    "artifact_publishing_in_15_4",
    "packaging_filesystem_io_in_15_4",
    "packaging_environment_probe_in_15_4",
    "dependency_freeze_in_15_4",
    "asset_discovery_in_15_4",
    "qml_asset_discovery_in_15_4",
    "runtime_activation_in_15_4",
    "paper_runtime_start_in_15_4",
    "testnet_runtime_start_in_15_4",
    "live_canary_start_in_15_4",
    "live_trading_in_15_4",
    "order_generation_in_15_4",
    "order_submission_in_15_4",
    "order_cancel_in_15_4",
    "order_replace_in_15_4",
    "private_endpoint_in_15_4",
    "network_io_in_15_4",
    "credential_read_in_15_4",
    "config_env_secret_read_in_15_4",
    "qml_bridge_change_in_15_4",
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
    "packaging_dry_run_contract_is_plain_data_only",
    "packaging_dry_run_contract_is_source_only",
    "packaging_dry_run_contract_reads_packaging_gate_contract_only",
    "packaging_dry_run_contract_preserves_exe_direction_without_packaging",
    "packaging_dry_run_contract_can_feed_15_5_packaging_dry_run_read_model",
    "packaging_dry_run_contract_cannot_execute_dry_run",
    "packaging_dry_run_contract_cannot_package_exe",
    "packaging_dry_run_contract_cannot_start_pyinstaller",
    "packaging_dry_run_contract_cannot_execute_build_commands",
    "packaging_dry_run_contract_cannot_create_build_artifacts",
    "packaging_dry_run_contract_cannot_change_installers",
    "packaging_dry_run_contract_cannot_change_release_workflows",
    "packaging_dry_run_contract_cannot_run_artifact_smoke_tests",
    "packaging_dry_run_contract_cannot_sign_artifacts",
    "packaging_dry_run_contract_cannot_publish_artifacts",
    "packaging_dry_run_contract_cannot_probe_packaging_environment",
    "packaging_dry_run_contract_cannot_freeze_dependencies",
    "packaging_dry_run_contract_cannot_discover_assets",
    "packaging_dry_run_contract_cannot_discover_qml_assets",
    "packaging_dry_run_contract_cannot_perform_filesystem_io",
    "packaging_dry_run_contract_cannot_activate_runtime",
    "packaging_dry_run_contract_cannot_start_paper_runtime",
    "packaging_dry_run_contract_cannot_start_testnet_runtime",
    "packaging_dry_run_contract_cannot_start_live_canary",
    "packaging_dry_run_contract_cannot_enable_live_trading",
    "packaging_dry_run_contract_cannot_generate_orders",
    "packaging_dry_run_contract_cannot_" + "sub" + "mit_orders",
    "packaging_dry_run_contract_cannot_" + "can" + "cel_orders",
    "packaging_dry_run_contract_cannot_" + "re" + "place_orders",
    "packaging_dry_run_contract_cannot_access_private_endpoints",
    "packaging_dry_run_contract_cannot_open_network_io",
    "packaging_dry_run_contract_cannot_read_credentials",
    "packaging_dry_run_contract_cannot_start_runtime_loop",
    "packaging_dry_run_contract_cannot_execute_runtime_gates",
    "packaging_dry_run_contract_cannot_mutate_gate_state",
    "packaging_dry_run_contract_cannot_read_config_env_or_secrets",
    "packaging_dry_run_contract_cannot_change_ui_bridge",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_m_packaging_dry_run_contract()


def test_payload_is_json_serializable_and_top_level_fields_are_stable() -> None:
    payload = _payload()

    assert json.loads(json.dumps(payload, sort_keys=True)) == payload
    assert list(payload) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_4() -> None:
    payload = _payload()

    assert payload["schema_version"] == PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_SCHEMA_VERSION
    assert (
        payload["block_m_packaging_dry_run_contract_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_KIND
    )
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.4"
    assert (
        payload["block_m_packaging_dry_run_contract_status"]
        == BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_STATUS
    )
    assert (
        payload["block_m_packaging_dry_run_contract_decision"]
        == BLOCK_M_PACKAGING_DRY_RUN_CONTRACT_DECISION
    )
    assert payload["ready_for_block_m_5"] is READY_FOR_BLOCK_M_5 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.5"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING DRY RUN READ MODEL"
    assert payload["future_steps"] == ["functional_preview_15_5_packaging_dry_run_read_model"]
    assert payload["status"] == STATUS


def test_packaging_gate_contract_reference_points_to_15_3() -> None:
    payload = _payload()
    gate = build_preview_block_m_packaging_gate_contract()
    reference = payload["packaging_gate_contract_reference"]

    for key in [
        "schema_version",
        "block_m_packaging_gate_contract_kind",
        "block",
        "step",
        "block_m_packaging_gate_contract_status",
        "block_m_packaging_gate_contract_decision",
        "ready_for_block_m_4",
        "next_step",
        "next_step_title",
    ]:
        assert reference[key] == gate[key]
    assert reference["source_packaging_gate_contract_step"] == "FUNCTIONAL-PREVIEW-15.3"
    assert reference["source_packaging_gate_contract_read_by_15_4_dry_run_contract"] is True
    assert reference["packaging_gate_contract_available_before_dry_run_contract"] is True
    assert reference["static_packaging_gate_contract_only"] is True
    for key, value in reference.items():
        if (
            key.endswith("_by_15_4")
            or key.endswith("_opened_by_15_4")
            or key.endswith("_accessed_by_15_4")
        ):
            assert value is False


def test_packaging_dry_run_summary_preserves_exe_direction_and_blocks_execution() -> None:
    summary = _payload()["packaging_dry_run_summary"]

    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_dry_run_prerequisites_are_gate_rows_and_unsatisfied() -> None:
    payload = _payload()
    gate = build_preview_block_m_packaging_gate_contract()
    rows = payload["dry_run_prerequisite_contract"]

    assert len(rows) == len(gate["packaging_prerequisite_gate_rows"])
    assert {row["source_id"] for row in rows} == {
        row["prerequisite_id"] for row in gate["packaging_prerequisite_gate_rows"]
    }
    for row in rows:
        assert row["dry_run_row_type"] == "packaging_dry_run_static_prerequisite_row"
        assert row["required_before_dry_run"] is True
        assert row["satisfied_in_15_4"] is False
        assert row["checked_by_15_4"] is False
        assert row["requires_future_step"] is True
        assert row["source_failure_policy"] == "fail_closed"
        assert row["failure_policy"] == "fail_closed"


def test_dry_run_execution_blocked_contract_blocks_all_packaging_paths() -> None:
    blocked = _payload()["dry_run_execution_blocked_contract"]

    assert blocked["dry_run_execution_contract_built"] is True
    assert blocked["dry_run_requires_future_explicit_gate"] is True
    assert blocked["dry_run_requires_future_operator_confirmation"] is True
    assert blocked["dry_run_not_performed_by_15_4"] is True
    assert blocked["packaging_not_performed_by_15_4"] is True
    for key, value in blocked.items():
        if key.endswith("_allowed_now"):
            assert value is False


def test_dry_run_simulation_plan_is_static_and_not_executed() -> None:
    rows = _payload()["dry_run_simulation_plan_contract"]

    assert len(rows) >= 12
    assert {row["simulation_step_id"] for row in rows} >= {
        "validate_gate_state",
        "validate_pyinstaller_spec_presence",
        "validate_build_environment_contract",
        "validate_dependency_freeze_contract",
        "validate_asset_inclusion_contract",
        "validate_qml_asset_inclusion_contract",
        "validate_runtime_disabled_contract",
        "validate_no_live_credentials_contract",
        "validate_no_network_build_contract",
        "validate_artifact_naming_contract",
        "validate_smoke_policy_contract",
        "validate_rollback_delete_policy_contract",
    }
    for row in rows:
        assert row["planned_for_future_dry_run"] is True
        assert row["executed_in_15_4"] is False
        assert row["allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True


def test_dry_run_artifact_policy_disallows_artifact_changes() -> None:
    policy = _payload()["dry_run_artifact_policy_contract"]

    assert policy["artifact_policy_contract_built"] is True
    assert policy["artifact_policy_requires_future_gate"] is True
    assert policy["artifact_policy_requires_future_operator_confirmation"] is True
    assert policy["no_artifact_created_by_15_4"] is True
    for key, value in policy.items():
        if (
            key.endswith("_allowed_now")
            or key.endswith("_finalized_now")
            or key.endswith("_selected_now")
        ):
            assert value is False


def test_runtime_safety_carryover_blocks_runtime_order_private_network_and_qml() -> None:
    rows = _payload()["runtime_safety_carryover_contract"]

    assert {row["capability_id"] for row in rows} == RUNTIME_CAPABILITY_IDS
    for row in rows:
        assert row["source_gate_allowed_now"] is False
        assert row["dry_run_contract_allowed_now"] is False
        assert row["dry_run_contract_executed_now"] is False
        assert row["blocked_in_15_4"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_contract_preserves_exe_and_starts_nothing() -> None:
    contract = _payload()["exe_direction_dry_run_contract"]

    assert contract["final_product_direction"] == "desktop_exe"
    assert contract["exe_direction_preserved"] is True
    assert contract["dry_run_contract_confirms_exe_direction"] is True
    assert contract["packaging_deferred_to_future_explicit_block"] is True
    assert contract["dry_run_deferred_to_future_explicit_block"] is True
    assert contract["future_packaging_requires_explicit_gate"] is True
    assert contract["future_dry_run_requires_explicit_gate"] is True
    assert contract["future_packaging_requires_separate_prompt"] is True
    assert contract["future_packaging_must_not_use_live_credentials"] is True
    assert contract["future_packaging_must_not_enable_runtime_by_itself"] is True
    for key, value in contract.items():
        if key.endswith("_now"):
            assert value is False


def test_fail_closed_decision_blocks_all_forbidden_15_4_paths() -> None:
    decision = _payload()["fail_closed_dry_run_decision"]

    assert decision["missing_packaging_gate_contract_policy"] == "fail_closed"
    assert decision["missing_dry_run_check_policy"] == "fail_closed"
    assert decision["missing_operator_confirmation_policy"] == "fail_closed"
    assert decision["missing_runtime_safety_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_execution_evidence_shows_no_dry_run_packaging_runtime_or_io() -> None:
    evidence = _payload()["non_execution_evidence"]

    assert evidence["source_packaging_gate_contract_read"] is True
    assert evidence["packaging_dry_run_contract_built"] is True
    assert evidence["packaging_dry_run_contract_only"] is True
    for key in NON_EXECUTION_FALSE_FLAGS:
        assert evidence[key] is False


def test_dry_run_boundaries_are_closed() -> None:
    boundaries = _payload()["dry_run_boundaries"]

    for key in BOUNDARY_TRUE_FLAGS:
        assert boundaries[key] is True
    for key, value in boundaries.items():
        if key.startswith("packaging_dry_run_contract_cannot_"):
            assert value is True


def test_source_boundaries_point_to_15_3() -> None:
    boundaries = _payload()["source_boundaries"]

    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_packaging_gate_contract"] == "FUNCTIONAL-PREVIEW-15.3"
    for key, value in boundaries.items():
        if key.startswith("forbidden_"):
            assert value is False
    nested = boundaries["source_packaging_gate_contract_boundaries"]
    assert nested["allowed_imports_only"] is True
    assert nested["source_packaging_readiness_matrix"] == "FUNCTIONAL-PREVIEW-15.2"


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]

    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert {alias.name for alias in imports[1].names} == {"Any", "Final"}
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_m_packaging_gate_contract"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_m_packaging_gate_contract"
    ]


def test_source_call_guard_blocks_io_network_runtime_orders_and_packaging_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_call_names = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "requests",
        "subprocess",
        "urlopen",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "getaddrinfo",
        "create_connection",
        "TradingController",
        "DecisionEnvelope",
        "activate",
        "start",
        "execute",
        "mutate",
        "PyInstaller",
        "packaging",
        "build",
    }
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = {
        func.id if isinstance(func, ast.Name) else func.attr
        for func in calls
        if isinstance(func, ast.Name | ast.Attribute)
    }

    assert "build_preview_block_m_packaging_gate_contract" in call_names
    assert forbidden_call_names.isdisjoint(call_names)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden_tokens = [
        "create_order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]

    for token in forbidden_tokens:
        assert token not in source
