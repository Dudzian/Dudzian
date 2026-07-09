"""Tests for FUNCTIONAL-PREVIEW-15.3 Block M packaging gate contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_gate_contract import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_GATE_CONTRACT_DECISION,
    BLOCK_M_PACKAGING_GATE_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_KIND,
    PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_4,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_gate_contract,
)
from ui.pyside_app.preview_block_m_packaging_readiness_matrix import (
    build_preview_block_m_packaging_readiness_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_gate_contract.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_gate_contract_kind",
    "block",
    "step",
    "block_m_packaging_gate_contract_status",
    "block_m_packaging_gate_contract_decision",
    "ready_for_block_m_4",
    "next_step",
    "next_step_title",
    "packaging_readiness_matrix_reference",
    "packaging_gate_summary",
    "packaging_gate_checklist",
    "packaging_gate_decision_table",
    "packaging_prerequisite_gate_rows",
    "packaging_execution_blocked_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_gate_contract",
    "fail_closed_packaging_gate_decision",
    "non_execution_evidence",
    "gate_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_TRUE_FLAGS = [
    "packaging_readiness_matrix_available",
    "packaging_gate_contract_built",
    "ready_for_block_m_4",
    "exe_direction_preserved",
    "packaging_gate_static_only",
    "packaging_gate_ready_for_future_contract",
]
SUMMARY_FALSE_FLAGS = [
    "packaging_gate_satisfied_now",
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
CHECK_IDS = {
    "packaging_gate_explicitly_approved",
    "pyinstaller_spec_exists_and_reviewed",
    "build_environment_defined",
    "dependency_freeze_defined",
    "asset_inclusion_list_defined",
    "qml_asset_inclusion_list_defined",
    "runtime_disabled_during_packaging",
    "no_live_credentials_embedded",
    "no_network_required_during_build",
    "installer_policy_defined",
    "release_artifact_naming_defined",
    "built_artifact_smoke_policy_defined",
    "rollback_delete_artifact_policy_defined",
    "signing_policy_defined",
    "manual_packaging_confirmation_required",
}
DECISION_GATE_IDS = {
    "packaging_gate",
    "pyinstaller_gate",
    "build_command_gate",
    "artifact_creation_gate",
    "installer_mutation_gate",
    "release_workflow_mutation_gate",
    "artifact_smoke_gate",
    "signing_gate",
    "publishing_gate",
    "filesystem_io_gate",
    "credential_exclusion_gate",
    "network_free_build_gate",
    "runtime_disabled_gate",
    "qml_bridge_unchanged_gate",
}
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
    "packaging_execution_in_15_3",
    "pyinstaller_execution_in_15_3",
    "build_command_execution_in_15_3",
    "build_artifact_creation_in_15_3",
    "installer_change_in_15_3",
    "release_workflow_change_in_15_3",
    "artifact_smoke_test_in_15_3",
    "artifact_signing_in_15_3",
    "artifact_publishing_in_15_3",
    "packaging_filesystem_io_in_15_3",
    "packaging_environment_probe_in_15_3",
    "dependency_freeze_in_15_3",
    "asset_discovery_in_15_3",
    "qml_asset_discovery_in_15_3",
    "runtime_activation_in_15_3",
    "paper_runtime_start_in_15_3",
    "testnet_runtime_start_in_15_3",
    "live_canary_start_in_15_3",
    "live_trading_in_15_3",
    "order_generation_in_15_3",
    "order_submission_in_15_3",
    "order_cancel_in_15_3",
    "order_replace_in_15_3",
    "private_endpoint_in_15_3",
    "network_io_in_15_3",
    "credential_read_in_15_3",
    "config_env_secret_read_in_15_3",
    "qml_bridge_change_in_15_3",
]
NON_EXECUTION_FALSE_FLAGS = [
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
    "packaging_gate_contract_is_plain_data_only",
    "packaging_gate_contract_is_source_only",
    "packaging_gate_contract_reads_readiness_matrix_only",
    "packaging_gate_contract_preserves_exe_direction_without_packaging",
    "packaging_gate_contract_can_feed_15_4_packaging_dry_run_contract",
    "packaging_gate_contract_cannot_package_exe",
    "packaging_gate_contract_cannot_start_pyinstaller",
    "packaging_gate_contract_cannot_execute_build_commands",
    "packaging_gate_contract_cannot_create_build_artifacts",
    "packaging_gate_contract_cannot_change_installers",
    "packaging_gate_contract_cannot_change_release_workflows",
    "packaging_gate_contract_cannot_run_artifact_smoke_tests",
    "packaging_gate_contract_cannot_sign_artifacts",
    "packaging_gate_contract_cannot_publish_artifacts",
    "packaging_gate_contract_cannot_probe_packaging_environment",
    "packaging_gate_contract_cannot_freeze_dependencies",
    "packaging_gate_contract_cannot_discover_assets",
    "packaging_gate_contract_cannot_discover_qml_assets",
    "packaging_gate_contract_cannot_perform_filesystem_io",
    "packaging_gate_contract_cannot_activate_runtime",
    "packaging_gate_contract_cannot_start_paper_runtime",
    "packaging_gate_contract_cannot_start_testnet_runtime",
    "packaging_gate_contract_cannot_start_live_canary",
    "packaging_gate_contract_cannot_enable_live_trading",
    "packaging_gate_contract_cannot_generate_orders",
    "packaging_gate_contract_cannot_" + "sub" + "mit_orders",
    "packaging_gate_contract_cannot_" + "can" + "cel_orders",
    "packaging_gate_contract_cannot_" + "re" + "place_orders",
    "packaging_gate_contract_cannot_access_private_endpoints",
    "packaging_gate_contract_cannot_open_network_io",
    "packaging_gate_contract_cannot_read_credentials",
    "packaging_gate_contract_cannot_start_runtime_loop",
    "packaging_gate_contract_cannot_execute_runtime_gates",
    "packaging_gate_contract_cannot_mutate_gate_state",
    "packaging_gate_contract_cannot_read_config_env_or_secrets",
    "packaging_gate_contract_cannot_change_ui_bridge",
]


def _payload() -> dict[str, Any]:
    return build_preview_block_m_packaging_gate_contract()


def test_payload_is_json_serializable() -> None:
    json.dumps(_payload(), sort_keys=True)


def test_top_level_fields_are_exact_and_ordered() -> None:
    assert list(_payload()) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_3() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_SCHEMA_VERSION
    assert (
        payload["block_m_packaging_gate_contract_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_GATE_CONTRACT_KIND
    )
    assert payload["block"] == BLOCK_ID == "M"
    assert payload["step"] == STEP_ID == "15.3"
    assert (
        payload["block_m_packaging_gate_contract_status"] == BLOCK_M_PACKAGING_GATE_CONTRACT_STATUS
    )
    assert (
        payload["block_m_packaging_gate_contract_decision"]
        == BLOCK_M_PACKAGING_GATE_CONTRACT_DECISION
    )
    assert payload["ready_for_block_m_4"] is READY_FOR_BLOCK_M_4 is True
    assert payload["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.4"
    assert payload["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING DRY RUN CONTRACT"
    assert payload["status"] == STATUS


def test_packaging_readiness_matrix_reference_points_to_15_2() -> None:
    reference = _payload()["packaging_readiness_matrix_reference"]
    assert reference["source_packaging_readiness_matrix_step"] == "FUNCTIONAL-PREVIEW-15.2"
    assert reference["step"] == "15.2"
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-15.3"
    assert reference["source_packaging_readiness_matrix_read_by_15_3_gate_contract"] is True
    assert reference["packaging_readiness_matrix_available_before_gate_contract"] is True
    assert reference["static_readiness_matrix_only"] is True
    for key, value in reference.items():
        if (
            key.endswith("_by_15_3")
            or key.endswith("_opened_by_15_3")
            or key.endswith("_read_by_15_3")
            or key.endswith("_accessed_by_15_3")
            or key.endswith("_performed_by_15_3")
        ):
            if key != "source_packaging_readiness_matrix_read_by_15_3_gate_contract":
                assert value is False


def test_packaging_gate_summary_preserves_exe_and_blocks_everything_now() -> None:
    summary = _payload()["packaging_gate_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_packaging_gate_checklist_is_static_unsatisfied_and_future_gated() -> None:
    rows = _payload()["packaging_gate_checklist"]
    assert {row["check_id"] for row in rows} == CHECK_IDS
    for row in rows:
        assert row["required_before_packaging"] is True
        assert row["satisfied_in_15_3"] is False
        assert row["checked_by_15_3"] is False
        assert row["requires_future_step"] is True
        assert row["failure_policy"] == "fail_closed"


def test_packaging_gate_decision_table_blocks_gate_and_capability_paths() -> None:
    rows = _payload()["packaging_gate_decision_table"]
    assert {row["gate_id"] for row in rows} == DECISION_GATE_IDS
    for row in rows:
        assert row["gate_required"] is True
        assert row["gate_satisfied_now"] is False
        assert row["gate_checked_now"] is False
        assert row["gate_execution_allowed_now"] is False
        assert row["capability_allowed_now"] is False
        assert row["capability_executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_packaging_prerequisite_gate_rows_derive_from_15_2_and_fail_closed() -> None:
    matrix_rows = build_preview_block_m_packaging_readiness_matrix()[
        "packaging_prerequisite_matrix"
    ]
    rows = _payload()["packaging_prerequisite_gate_rows"]
    assert [row["prerequisite_id"] for row in rows] == [
        row["prerequisite_id"] for row in matrix_rows
    ]
    for row in rows:
        assert row["gate_row_type"] == "packaging_prerequisite_static_gate_row"
        assert row["source_required_before_packaging"] is True
        assert row["source_satisfied_in_15_2"] is False
        assert row["source_checked_by_15_2"] is False
        assert row["required_before_packaging"] is True
        assert row["satisfied_in_15_3"] is False
        assert row["checked_by_15_3"] is False
        assert row["requires_future_step"] is True
        assert row["failure_policy"] == "fail_closed"


def test_packaging_execution_blocked_contract_blocks_execution_paths() -> None:
    contract = _payload()["packaging_execution_blocked_contract"]
    assert contract["packaging_execution_contract_built"] is True
    assert contract["packaging_requires_future_explicit_gate"] is True
    assert contract["packaging_requires_future_operator_confirmation"] is True
    assert contract["packaging_not_performed_by_15_3"] is True
    for key, value in contract.items():
        if key.endswith("_allowed_now"):
            assert value is False


def test_runtime_safety_carryover_contract_blocks_runtime_paths() -> None:
    rows = _payload()["runtime_safety_carryover_contract"]
    assert {row["capability_id"] for row in rows} == RUNTIME_CAPABILITY_IDS
    for row in rows:
        assert row["source_matrix_allowed_now"] is False
        assert row["gate_contract_allowed_now"] is False
        assert row["gate_contract_executed_now"] is False
        assert row["blocked_in_15_3"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_gate_contract_preserves_direction_without_execution() -> None:
    contract = _payload()["exe_direction_gate_contract"]
    assert contract["final_product_direction"] == "desktop_exe"
    assert contract["exe_direction_preserved"] is True
    assert contract["gate_contract_confirms_exe_direction"] is True
    assert contract["packaging_deferred_to_future_explicit_block"] is True
    assert contract["future_packaging_requires_explicit_gate"] is True
    assert contract["future_packaging_requires_separate_prompt"] is True
    assert contract["future_packaging_must_not_use_live_credentials"] is True
    assert contract["future_packaging_must_not_enable_runtime_by_itself"] is True
    for key, value in contract.items():
        if key.endswith("_now"):
            assert value is False


def test_fail_closed_packaging_gate_decision_blocks_forbidden_paths() -> None:
    decision = _payload()["fail_closed_packaging_gate_decision"]
    assert decision["missing_packaging_readiness_matrix_policy"] == "fail_closed"
    assert decision["missing_packaging_check_policy"] == "fail_closed"
    assert decision["missing_operator_confirmation_policy"] == "fail_closed"
    assert decision["missing_runtime_safety_policy"] == "fail_closed"
    for key in BLOCKED_DECISION_KEYS:
        assert decision[key] == "blocked"


def test_non_execution_evidence_stays_false_for_forbidden_activity() -> None:
    evidence = _payload()["non_execution_evidence"]
    assert evidence["source_packaging_readiness_matrix_read"] is True
    assert evidence["packaging_gate_contract_built"] is True
    assert evidence["packaging_gate_contract_only"] is True
    for key in NON_EXECUTION_FALSE_FLAGS:
        assert evidence[key] is False
    for key, value in evidence.items():
        if key.startswith("source_matrix_"):
            assert value is False


def test_gate_boundaries_are_closed() -> None:
    boundaries = _payload()["gate_boundaries"]
    for key in BOUNDARY_TRUE_FLAGS:
        assert boundaries[key] is True
    assert all(
        value is True
        for key, value in boundaries.items()
        if key.startswith("packaging_gate_contract_cannot_")
    )


def test_source_boundaries_point_to_15_2() -> None:
    boundaries = _payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_packaging_readiness_matrix"] == "FUNCTIONAL-PREVIEW-15.2"
    assert (
        boundaries["source_packaging_readiness_matrix_boundaries"]["allowed_imports_only"] is True
    )
    for key, value in boundaries.items():
        if key.startswith("forbidden_"):
            assert value is False


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append((module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((alias.name, None))
    assert imports == [
        ("__future__", "annotations"),
        ("typing", "Any"),
        ("typing", "Final"),
        (
            "ui.pyside_app.preview_block_m_packaging_readiness_matrix",
            "build_preview_block_m_packaging_readiness_matrix",
        ),
    ]


def test_source_call_guard_blocks_forbidden_calls() -> None:
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
        "urllib",
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
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                seen.add(func.id)
            elif isinstance(func, ast.Attribute):
                seen.add(func.attr)
    assert seen.isdisjoint(forbidden_call_names)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden_literals = [
        "create_order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    for token in forbidden_literals:
        assert token not in source
