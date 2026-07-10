"""Tests for FUNCTIONAL-PREVIEW-16.1 Block N read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_n_read_model import (
    BLOCK_ID,
    BLOCK_N_READ_MODEL_DECISION,
    BLOCK_N_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_N_READ_MODEL_KIND,
    PREVIEW_BLOCK_N_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_N_2,
    STATUS,
    STEP_ID,
    build_preview_block_n_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_n_read_model.py"
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_read_model_kind",
    "block",
    "step",
    "block_n_read_model_status",
    "block_n_read_model_decision",
    "ready_for_block_n_2",
    "next_step",
    "next_step_title",
    "block_n_entry_contract_reference",
    "block_n_read_summary",
    "block_m_closure_handoff_read_model",
    "block_n_entry_readiness_read_model",
    "packaging_release_safety_read_rows",
    "runtime_safety_read_rows",
    "exe_direction_read_model",
    "fail_closed_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
PACKAGING_IDS = {
    "packaging_dry_run_execution",
    "packaging_execution",
    "pyinstaller_execution",
    "build_command_execution",
    "build_artifact_creation",
    "dependency_freeze",
    "asset_discovery",
    "qml_asset_discovery",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke_test",
    "artifact_signing",
    "artifact_publishing",
    "release_execution",
    "release_publish",
    "release_signing",
    "release_smoke_test",
    "release_notes_generation",
    "release_tag_creation",
    "release_upload",
    "release_external_export",
}
RUNTIME_IDS = {
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
SUMMARY_FALSE_KEYS = [
    "release_execution_allowed_now",
    "release_publish_allowed_now",
    "release_signing_allowed_now",
    "release_smoke_test_allowed_now",
    "release_workflow_change_allowed_now",
    "release_notes_generation_allowed_now",
    "release_tag_creation_allowed_now",
    "release_upload_allowed_now",
    "release_external_export_allowed_now",
    "artifact_creation_allowed_now",
    "artifact_mutation_allowed_now",
    "artifact_delete_allowed_now",
    "artifact_smoke_test_allowed_now",
    "artifact_signing_allowed_now",
    "artifact_publishing_allowed_now",
    "artifact_location_selected_now",
    "artifact_naming_finalized_now",
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
    return build_preview_block_n_read_model()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_next_and_readiness_are_16_1() -> None:
    built = payload()
    assert built["schema_version"] == PREVIEW_BLOCK_N_READ_MODEL_SCHEMA_VERSION
    assert built["block_n_read_model_kind"] == PREVIEW_BLOCK_N_READ_MODEL_KIND
    assert built["block"] == BLOCK_ID == "N"
    assert built["step"] == STEP_ID == "16.1"
    assert built["block_n_read_model_status"] == BLOCK_N_READ_MODEL_STATUS
    assert built["block_n_read_model_decision"] == BLOCK_N_READ_MODEL_DECISION
    assert built["ready_for_block_n_2"] is READY_FOR_BLOCK_N_2 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-16.2"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "BLOCK N SAFETY GATE MATRIX"
    assert built["status"] == STATUS


def test_block_n_entry_contract_reference_points_to_16_0_and_blocks_16_1_work() -> None:
    reference = payload()["block_n_entry_contract_reference"]
    assert reference["source_block_n_entry_contract_step"] == "FUNCTIONAL-PREVIEW-16.0"
    assert reference["step"] == "16.0"
    assert reference["block"] == "N"
    assert reference["block_n_opened"] is True
    assert reference["ready_for_block_n_1"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-16.1"
    for key in [
        "source_block_n_entry_contract_read_by_16_1_read_model",
        "block_n_entry_contract_available_before_read_model",
        "static_block_n_entry_contract_only",
        "block_n_opened_before_read_model",
        "block_n_read_model_built_by_16_1",
        "ready_for_functional_preview_16_2",
    ]:
        assert reference[key] is True
    false_keys = [key for key in reference if key.endswith("_by_16_1")]
    assert false_keys
    assert all(
        reference[key] is False for key in false_keys if key != "block_n_read_model_built_by_16_1"
    )


def test_block_n_read_summary_preserves_closure_exe_and_blocks_unsafe_paths() -> None:
    summary = payload()["block_n_read_summary"]
    for key in [
        "block_n_entry_contract_available",
        "block_n_opened",
        "block_n_read_model_built",
        "ready_for_block_n_2",
        "ready_for_functional_preview_16_2",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "previous_block_closure_handoff_preserved",
        "block_n_read_model_static_only",
        "block_n_read_model_read_only",
    ]:
        assert summary[key] is True
    assert all(summary[key] is False for key in SUMMARY_FALSE_KEYS)


def test_block_m_handoff_read_model_is_m_to_n_without_unlocking_execution() -> None:
    handoff = payload()["block_m_closure_handoff_read_model"]
    assert handoff["previous_block"] == "M"
    assert handoff["current_block"] == "N"
    assert handoff["previous_block_closure_step"] == "FUNCTIONAL-PREVIEW-15.10"
    assert handoff["current_block_entry_step"] == "FUNCTIONAL-PREVIEW-16.0"
    assert handoff["current_block_read_model_step"] == "FUNCTIONAL-PREVIEW-16.1"
    for key in [
        "previous_block_closed",
        "current_block_opened",
        "handoff_source_only",
        "handoff_plain_data_only",
        "handoff_read_by_16_1",
        "handoff_preserved_by_read_model",
        "handoff_does_not_unlock_packaging",
        "handoff_does_not_unlock_release",
        "handoff_does_not_unlock_runtime",
        "handoff_requires_future_explicit_gate_for_packaging",
        "handoff_requires_future_explicit_gate_for_release",
        "handoff_requires_future_explicit_gate_for_runtime",
    ]:
        assert handoff[key] is True


def test_entry_readiness_read_model_is_ready_for_16_2_without_execution() -> None:
    readiness = payload()["block_n_entry_readiness_read_model"]
    assert all(readiness.values())
    assert readiness["ready_for_block_n_2"] is True
    assert readiness["ready_for_functional_preview_16_2"] is True


def test_packaging_release_safety_read_rows_block_all_capabilities() -> None:
    rows = payload()["packaging_release_safety_read_rows"]
    assert {row["capability_id"] for row in rows} == PACKAGING_IDS
    for row in rows:
        assert row["source_allowed_now"] is False
        assert row["source_entry_contract_allowed_now"] is False
        assert row["source_entry_contract_executed_now"] is False
        assert row["read_model_allowed_now"] is False
        assert row["read_model_executed_now"] is False
        assert row["blocked_in_16_1"] is True
        assert row["requires_future_explicit_gate"] is True


def test_runtime_safety_read_rows_block_all_capabilities() -> None:
    rows = payload()["runtime_safety_read_rows"]
    assert {row["capability_id"] for row in rows} == RUNTIME_IDS
    for row in rows:
        assert row["source_read_model_allowed_now"] is False
        assert row["source_entry_contract_allowed_now"] is False
        assert row["source_entry_contract_executed_now"] is False
        assert row["read_model_allowed_now"] is False
        assert row["read_model_executed_now"] is False
        assert row["blocked_in_16_1"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_preserved_without_starting_packaging_release_or_artifact_work() -> None:
    exe = payload()["exe_direction_read_model"]
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["exe_direction_preserved"] is True
    assert exe["block_n_read_model_confirms_exe_direction"] is True
    false_keys = [key for key, value in exe.items() if key.endswith("_now")]
    assert false_keys
    assert all(exe[key] is False for key in false_keys)
    assert exe["future_packaging_requires_explicit_gate"] is True
    assert exe["future_release_requires_explicit_gate"] is True


def test_fail_closed_read_decision_allows_16_2_and_blocks_16_1_execution_paths() -> None:
    decision = payload()["fail_closed_read_decision"]
    assert decision["block_n_read_model_in_16_1"] == "ready"
    assert decision["block_n_safety_gate_matrix_in_16_2"] == "allowed"
    blocked = {key: value for key, value in decision.items() if key.endswith("_in_16_1")}
    assert blocked
    assert all(
        value == "blocked" for key, value in blocked.items() if key != "block_n_read_model_in_16_1"
    )


def test_non_execution_evidence_shows_no_forbidden_execution() -> None:
    evidence = payload()["non_execution_evidence"]
    for key in [
        "source_block_n_entry_contract_read",
        "block_n_read_model_built",
        "block_n_read_model_only",
        "block_n_opened",
        "ready_for_block_n_2",
    ]:
        assert evidence[key] is True
    for key in [
        "release_executed",
        "release_published",
        "release_signed",
        "artifact_created",
        "artifact_mutated",
        "artifact_deleted",
        "packaging_dry_run_executed",
        "packaging_executed",
        "pyinstaller_started",
        "build_command_executed",
        "runtime_activated",
        "orders_enabled",
        "private_endpoint_accessed",
        "network_io_opened",
        "credentials_read",
        "filesystem_io_performed",
        "qml_bridge_changed",
    ]:
        assert evidence[key] is False


def test_read_model_boundaries_are_closed() -> None:
    boundaries = payload()["read_model_boundaries"]
    assert boundaries["block_n_read_model_is_plain_data_only"] is True
    assert boundaries["block_n_read_model_is_source_only"] is True
    assert boundaries["block_n_read_model_reads_block_n_entry_contract_only"] is True
    cannot_keys = [key for key in boundaries if key.startswith("block_n_read_model_cannot_")]
    assert cannot_keys
    assert all(boundaries[key] is True for key in cannot_keys)


def test_source_boundaries_point_to_16_0() -> None:
    boundaries = payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_n_entry_contract"] == "FUNCTIONAL-PREVIEW-16.0"
    for key in [key for key in boundaries if key.startswith("forbidden_")]:
        assert boundaries[key] is False
    subset = boundaries["source_block_n_entry_contract_boundaries"]
    assert subset["allowed_imports_only"] is True
    assert (
        subset["entry_boundary_subset"]["block_n_entry_contract_can_feed_16_1_block_n_read_model"]
        is True
    )


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_n_entry_contract"
    assert imports[2].names[0].name == "build_preview_block_n_entry_contract"


def test_source_call_guard_blocks_forbidden_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_names = {
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
    }
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                called.add(func.id)
            elif isinstance(func, ast.Attribute):
                called.add(func.attr)
    assert forbidden_names.isdisjoint(called)


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
    assert all(token not in source for token in forbidden)
