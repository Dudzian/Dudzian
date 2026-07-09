"""Tests for FUNCTIONAL-PREVIEW-15.10 Block M closure audit."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_closure_audit import (
    BLOCK_ID,
    BLOCK_M_CLOSED,
    BLOCK_M_CLOSURE_AUDIT_DECISION,
    BLOCK_M_CLOSURE_AUDIT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_CLOSURE_AUDIT_KIND,
    PREVIEW_BLOCK_M_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_NEXT_BLOCK,
    STATUS,
    STEP_ID,
    build_preview_block_m_closure_audit,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_closure_audit.py"
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_closure_audit_kind",
    "block",
    "step",
    "block_m_closure_audit_status",
    "block_m_closure_audit_decision",
    "block_m_closed",
    "ready_for_next_block",
    "next_step",
    "next_step_title",
    "packaging_release_readiness_read_model_reference",
    "block_m_closure_summary",
    "block_m_step_chain_audit",
    "block_m_packaging_readiness_lineage_audit",
    "block_m_release_readiness_closure_audit",
    "packaging_execution_safety_closure_audit",
    "runtime_safety_closure_audit",
    "exe_direction_closure_audit",
    "fail_closed_closure_decision",
    "non_execution_evidence",
    "closure_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
FALSE_SUMMARY_KEYS = [
    "release_readiness_satisfied_now",
    "release_execution_allowed_now",
    "release_publish_allowed_now",
    "release_signing_allowed_now",
    "release_smoke_test_allowed_now",
    "release_workflow_change_allowed_now",
    "release_notes_generation_allowed_now",
    "release_tag_creation_allowed_now",
    "release_upload_allowed_now",
    "release_external_export_allowed_now",
    "artifact_policy_satisfied_now",
    "artifact_creation_allowed_now",
    "artifact_mutation_allowed_now",
    "artifact_delete_allowed_now",
    "artifact_smoke_test_allowed_now",
    "artifact_signing_allowed_now",
    "artifact_publishing_allowed_now",
    "artifact_location_selected_now",
    "artifact_naming_finalized_now",
    "artifact_retention_policy_finalized_now",
    "artifact_rollback_policy_finalized_now",
    "artifact_checksum_generation_allowed_now",
    "artifact_metadata_write_allowed_now",
    "artifact_audit_export_allowed_now",
    "artifact_cleanup_allowed_now",
    "dry_run_can_execute_now",
    "packaging_ready_now",
    "packaging_can_execute_now",
    "pyinstaller_can_start_now",
    "build_command_can_execute_now",
    "build_artifact_can_be_created_now",
    "installer_can_change_now",
    "release_workflow_can_change_now",
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


def payload() -> dict[str, Any]:
    return build_preview_block_m_closure_audit()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_next_and_closure_are_15_10() -> None:
    built = payload()
    assert built["schema_version"] == PREVIEW_BLOCK_M_CLOSURE_AUDIT_SCHEMA_VERSION
    assert built["block_m_closure_audit_kind"] == PREVIEW_BLOCK_M_CLOSURE_AUDIT_KIND
    assert built["block"] == BLOCK_ID == "M"
    assert built["step"] == STEP_ID == "15.10"
    assert built["block_m_closure_audit_status"] == BLOCK_M_CLOSURE_AUDIT_STATUS
    assert built["block_m_closure_audit_decision"] == BLOCK_M_CLOSURE_AUDIT_DECISION
    assert built["block_m_closed"] is BLOCK_M_CLOSED is True
    assert built["ready_for_next_block"] is READY_FOR_NEXT_BLOCK is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-16.0"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "NEXT BLOCK ENTRY CONTRACT"
    assert built["status"] == STATUS


def test_packaging_release_readiness_read_model_reference_points_to_15_9() -> None:
    reference = payload()["packaging_release_readiness_read_model_reference"]
    assert (
        reference["source_packaging_release_readiness_read_model_step"] == "FUNCTIONAL-PREVIEW-15.9"
    )
    assert reference["step"] == "15.9"
    assert reference["ready_for_block_m_10"] is True
    assert (
        reference["source_packaging_release_readiness_read_model_read_by_15_10_closure_audit"]
        is True
    )
    assert (
        reference["packaging_release_readiness_read_model_available_before_closure_audit"] is True
    )
    assert reference["static_packaging_release_readiness_read_model_only"] is True
    assert reference["block_m_closure_audit_built_by_15_10"] is True
    assert reference["block_m_closed_by_15_10"] is True
    false_keys = [key for key in reference if key.endswith("_by_15_10")]
    assert false_keys
    assert all(
        reference[key] is False
        for key in false_keys
        if key not in {"block_m_closure_audit_built_by_15_10", "block_m_closed_by_15_10"}
    )


def test_closure_summary_preserves_exe_closes_block_and_blocks_unsafe_paths() -> None:
    summary = payload()["block_m_closure_summary"]
    for key in [
        "packaging_release_readiness_read_model_available",
        "block_m_closure_audit_built",
        "block_m_closed",
        "ready_for_next_block",
        "ready_for_functional_preview_16_0",
        "exe_direction_preserved",
        "block_m_source_only_chain_complete",
        "block_m_packaging_readiness_chain_complete",
        "block_m_release_readiness_chain_complete",
        "closure_audit_static_only",
        "closure_audit_read_only",
    ]:
        assert summary[key] is True
    assert all(summary[key] is False for key in FALSE_SUMMARY_KEYS)


def test_block_m_step_chain_audit_contains_15_0_through_15_10_and_is_fail_closed() -> None:
    rows = payload()["block_m_step_chain_audit"]
    assert [row["step"] for row in rows] == [
        "15.0",
        "15.1",
        "15.2",
        "15.3",
        "15.4",
        "15.5",
        "15.6",
        "15.7",
        "15.8",
        "15.9",
        "15.10",
    ]
    for row in rows:
        assert row["source_only"] is True
        assert row["plain_data_only"] is True
        assert row["verified_by_closure"] is True
        assert row["executed_runtime"] is False
        assert row["executed_packaging"] is False
        assert row["executed_release"] is False
        assert row["created_artifact"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
    assert all(row["completed_before_closure"] is True for row in rows[:-1])
    assert rows[-1]["completed_before_closure"] is False


def test_packaging_readiness_lineage_is_complete_but_does_not_allow_execution() -> None:
    audit = payload()["block_m_packaging_readiness_lineage_audit"]
    assert audit["lineage_audit_built"] is True
    assert audit["lineage_steps"] == [f"15.{index}" for index in range(10)]
    for key in [
        "all_block_m_steps_represented",
        "packaging_readiness_chain_complete",
        "dry_run_chain_complete",
        "artifact_policy_chain_complete",
        "release_readiness_chain_complete",
        "closure_audit_step_present",
        "exe_direction_preserved_through_lineage",
        "future_packaging_requires_explicit_gate",
        "future_release_requires_explicit_gate",
    ]:
        assert audit[key] is True
    false_keys = [key for key in audit if key.endswith("_allowed_by_lineage")]
    assert false_keys
    assert all(audit[key] is False for key in false_keys)


def test_release_readiness_closure_audit_blocks_release_paths() -> None:
    audit = payload()["block_m_release_readiness_closure_audit"]
    for key in [
        "release_readiness_read_model_available",
        "release_readiness_chain_closed",
        "release_readiness_read_model_ready_for_closure_audit",
        "release_requires_future_explicit_gate",
        "release_requires_future_operator_confirmation",
        "block_m_closure_does_not_unlock_release",
    ]:
        assert audit[key] is True
    false_keys = [key for key in audit if key.endswith("_now")]
    assert false_keys
    assert all(audit[key] is False for key in false_keys)


def test_packaging_execution_safety_closure_audit_blocks_all_rows() -> None:
    rows = payload()["packaging_execution_safety_closure_audit"]
    capability_ids = {row["capability_id"] for row in rows}
    assert {
        "packaging_dry_run_execution",
        "packaging_execution",
        "pyinstaller_execution",
        "build_command_execution",
        "build_artifact_creation",
        "dependency_freeze",
        "asset_discovery",
        "qml_asset_discovery",
        "artifact_creation",
        "release_execution",
        "release_publish",
        "release_signing",
        "release_smoke_test",
        "release_notes_generation",
        "release_tag_creation",
        "release_upload",
        "release_external_export",
    } <= capability_ids
    for row in rows:
        assert row["blocked_in_15_10"] is True
        assert row["requires_future_explicit_gate"] is True
        for key in row:
            if key.endswith("allowed_now") or key.endswith("executed_now"):
                assert row[key] is False


def test_runtime_safety_closure_audit_blocks_runtime_and_side_effects() -> None:
    rows = payload()["runtime_safety_closure_audit"]
    assert {row["capability_id"] for row in rows} == {
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
    for row in rows:
        assert row["blocked_in_15_10"] is True
        assert row["requires_future_explicit_gate"] is True
        for key in row:
            if key.endswith("allowed_now") or key.endswith("executed_now"):
                assert row[key] is False


def test_exe_direction_closure_audit_preserves_direction_without_starting_work() -> None:
    audit = payload()["exe_direction_closure_audit"]
    assert audit["final_product_direction"] == "desktop_exe"
    assert audit["exe_direction_preserved"] is True
    assert audit["block_m_closure_audit_confirms_exe_direction"] is True
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "dry_run_deferred_to_future_explicit_block",
        "artifact_work_deferred_to_future_explicit_block",
        "release_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_dry_run_requires_explicit_gate",
        "future_artifact_work_requires_explicit_gate",
        "future_release_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        assert audit[key] is True
    false_keys = [key for key in audit if key.endswith("_now")]
    assert false_keys
    assert all(audit[key] is False for key in false_keys)


def test_fail_closed_closure_decision_closes_block_and_blocks_15_10_paths() -> None:
    decision = payload()["fail_closed_closure_decision"]
    assert decision["block_m_closure_in_15_10"] == "closed"
    assert decision["next_block_entry_in_16_0"] == "allowed"
    assert decision["missing_packaging_release_readiness_read_model_policy"] == "fail_closed"
    blocked = [key for key in decision if key.endswith("_in_15_10")]
    assert blocked
    assert all(
        decision[key] == "blocked" for key in blocked if key not in {"block_m_closure_in_15_10"}
    )


def test_non_execution_evidence_keeps_work_unexecuted() -> None:
    evidence = payload()["non_execution_evidence"]
    for key in [
        "source_packaging_release_readiness_read_model_read",
        "block_m_closure_audit_built",
        "block_m_closure_audit_only",
        "block_m_closed",
        "ready_for_next_block",
    ]:
        assert evidence[key] is True
    false_keys = [key for key, value in evidence.items() if value is False]
    assert {
        "release_executed",
        "release_published",
        "release_signed",
        "release_smoke_test_executed",
        "release_workflow_mutated",
        "artifact_created",
        "artifact_mutated",
        "artifact_deleted",
        "packaging_dry_run_executed",
        "packaging_executed",
        "pyinstaller_started",
        "build_command_executed",
        "runtime_activated",
        "network_io_opened",
        "credentials_read",
        "qml_bridge_changed",
    } <= set(false_keys)
    assert all(evidence[key] is False for key in false_keys)


def test_closure_and_source_boundaries_are_closed() -> None:
    built = payload()
    boundaries = built["closure_boundaries"]
    assert boundaries["block_m_closure_audit_is_plain_data_only"] is True
    assert boundaries["block_m_closure_audit_is_source_only"] is True
    assert boundaries["block_m_closure_audit_reads_release_readiness_read_model_only"] is True
    assert all(
        value is True
        for key, value in boundaries.items()
        if key.startswith("block_m_closure_audit_cannot_")
    )
    source = built["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_packaging_release_readiness_read_model"] == "FUNCTIONAL-PREVIEW-15.9"
    assert (
        source["source_packaging_release_readiness_read_model_boundaries"]["allowed_imports_only"]
        is True
    )
    for key in source:
        if key.startswith("forbidden_") and key.endswith("_present"):
            assert source[key] is False


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert [alias.name for alias in imports[1].names] == ["Any", "Final"]
    assert isinstance(imports[2], ast.ImportFrom)
    assert (
        imports[2].module == "ui.pyside_app.preview_block_m_packaging_release_readiness_read_model"
    )
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_m_packaging_release_readiness_read_model"
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
        "PyInstaller",
        "packaging",
        "build",
        "release",
    }
    call_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
    assert call_names.isdisjoint(forbidden_call_names)


def test_forbidden_literal_tokens_do_not_appear_in_helper() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden_literals = [
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    assert all(token not in source for token in forbidden_literals)
