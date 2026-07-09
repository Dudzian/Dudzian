"""Tests for FUNCTIONAL-PREVIEW-16.0 Block N entry contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_n_entry_contract import (
    BLOCK_ID,
    BLOCK_N_ENTRY_CONTRACT_DECISION,
    BLOCK_N_ENTRY_CONTRACT_STATUS,
    BLOCK_N_OPENED,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_N_ENTRY_CONTRACT_KIND,
    PREVIEW_BLOCK_N_ENTRY_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_N_1,
    STATUS,
    STEP_ID,
    build_preview_block_n_entry_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_n_entry_contract.py"
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_entry_contract_kind",
    "block",
    "step",
    "block_n_entry_contract_status",
    "block_n_entry_contract_decision",
    "block_n_opened",
    "ready_for_block_n_1",
    "next_step",
    "next_step_title",
    "block_m_closure_audit_reference",
    "block_n_entry_summary",
    "previous_block_closure_handoff",
    "block_n_entry_readiness_contract",
    "packaging_release_safety_carryover",
    "runtime_safety_carryover",
    "exe_direction_carryover",
    "fail_closed_entry_decision",
    "non_execution_evidence",
    "entry_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_FALSE_KEYS = [
    "release_execution_allowed_now",
    "release_publish_allowed_now",
    "release_signing_allowed_now",
    "release_smoke_test_allowed_now",
    "release_workflow_change_allowed_now",
    "artifact_creation_allowed_now",
    "artifact_mutation_allowed_now",
    "artifact_delete_allowed_now",
    "artifact_smoke_test_allowed_now",
    "artifact_signing_allowed_now",
    "artifact_publishing_allowed_now",
    "dry_run_can_execute_now",
    "packaging_ready_now",
    "packaging_can_execute_now",
    "pyinstaller_can_start_now",
    "build_command_can_execute_now",
    "build_artifact_can_be_created_now",
    "dependency_freeze_can_run_now",
    "asset_discovery_can_run_now",
    "qml_asset_discovery_can_run_now",
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
    return build_preview_block_n_entry_contract()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_next_and_entry_are_16_0() -> None:
    built = payload()
    assert built["schema_version"] == PREVIEW_BLOCK_N_ENTRY_CONTRACT_SCHEMA_VERSION
    assert built["block_n_entry_contract_kind"] == PREVIEW_BLOCK_N_ENTRY_CONTRACT_KIND
    assert built["block"] == BLOCK_ID == "N"
    assert built["step"] == STEP_ID == "16.0"
    assert built["block_n_entry_contract_status"] == BLOCK_N_ENTRY_CONTRACT_STATUS
    assert built["block_n_entry_contract_decision"] == BLOCK_N_ENTRY_CONTRACT_DECISION
    assert built["block_n_opened"] is BLOCK_N_OPENED is True
    assert built["ready_for_block_n_1"] is READY_FOR_BLOCK_N_1 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-16.1"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "BLOCK N READ MODEL"
    assert built["status"] == STATUS


def test_block_m_closure_audit_reference_points_to_15_10_and_16_0_boundaries() -> None:
    reference = payload()["block_m_closure_audit_reference"]
    assert reference["source_block_m_closure_audit_step"] == "FUNCTIONAL-PREVIEW-15.10"
    assert reference["step"] == "15.10"
    assert reference["block"] == "M"
    assert reference["block_m_closed"] is True
    assert reference["ready_for_next_block"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-16.0"
    for key in [
        "source_block_m_closure_audit_read_by_16_0_entry_contract",
        "block_m_closure_audit_available_before_block_n_entry",
        "static_block_m_closure_audit_only",
        "block_m_closed_before_block_n_entry",
        "block_n_entry_contract_built_by_16_0",
        "block_n_opened_by_16_0",
        "ready_for_functional_preview_16_1",
    ]:
        assert reference[key] is True
    false_keys = [key for key in reference if key.endswith("_by_16_0")]
    assert false_keys
    assert all(
        reference[key] is False
        for key in false_keys
        if key not in {"block_n_entry_contract_built_by_16_0", "block_n_opened_by_16_0"}
    )


def test_block_n_entry_summary_opens_block_preserves_exe_and_blocks_unsafe_paths() -> None:
    summary = payload()["block_n_entry_summary"]
    for key in [
        "block_m_closure_audit_available",
        "block_m_closed",
        "block_n_entry_contract_built",
        "block_n_opened",
        "ready_for_block_n_1",
        "ready_for_functional_preview_16_1",
        "exe_direction_preserved",
        "previous_block_closure_consumed",
        "next_block_entry_static_only",
        "next_block_entry_read_only",
    ]:
        assert summary[key] is True
    assert all(summary[key] is False for key in SUMMARY_FALSE_KEYS)


def test_previous_block_closure_handoff_is_m_to_n_and_does_not_unlock_execution() -> None:
    handoff = payload()["previous_block_closure_handoff"]
    assert handoff["previous_block"] == "M"
    assert handoff["next_block"] == "N"
    assert handoff["previous_block_closure_step"] == "FUNCTIONAL-PREVIEW-15.10"
    assert handoff["next_block_entry_step"] == "FUNCTIONAL-PREVIEW-16.0"
    assert handoff["previous_block_closed"] is True
    assert handoff["next_block_entry_allowed"] is True
    assert handoff["next_block_opened"] is True
    for key in [
        "handoff_does_not_unlock_packaging",
        "handoff_does_not_unlock_release",
        "handoff_does_not_unlock_runtime",
        "handoff_requires_future_explicit_gate_for_packaging",
        "handoff_requires_future_explicit_gate_for_release",
        "handoff_requires_future_explicit_gate_for_runtime",
    ]:
        assert handoff[key] is True


def test_entry_readiness_contract_is_ready_for_16_1_without_execution() -> None:
    readiness = payload()["block_n_entry_readiness_contract"]
    assert all(readiness.values())
    assert readiness["ready_for_functional_preview_16_1"] is True
    assert readiness["entry_contract_does_not_execute_packaging"] is True
    assert readiness["entry_contract_does_not_execute_release"] is True
    assert readiness["entry_contract_does_not_execute_runtime"] is True


def test_packaging_release_safety_carryover_blocks_all_capabilities() -> None:
    rows = payload()["packaging_release_safety_carryover"]
    expected = {
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
    assert {row["capability_id"] for row in rows} == expected
    for row in rows:
        assert row["source_allowed_now"] is False
        assert row["source_closure_allowed_now"] is False
        assert row["source_closure_executed_now"] is False
        assert row["entry_contract_allowed_now"] is False
        assert row["entry_contract_executed_now"] is False
        assert row["blocked_in_16_0"] is True
        assert row["requires_future_explicit_gate"] is True


def test_runtime_safety_carryover_blocks_runtime_live_order_private_and_io_paths() -> None:
    rows = payload()["runtime_safety_carryover"]
    expected = {
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
    assert {row["capability_id"] for row in rows} == expected
    for row in rows:
        assert row["source_read_model_allowed_now"] is False
        assert row["source_closure_allowed_now"] is False
        assert row["source_closure_executed_now"] is False
        assert row["entry_contract_allowed_now"] is False
        assert row["entry_contract_executed_now"] is False
        assert row["blocked_in_16_0"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_carryover_preserves_direction_without_starting_packaging_or_release() -> (
    None
):
    carryover = payload()["exe_direction_carryover"]
    assert carryover["final_product_direction"] == "desktop_exe"
    assert carryover["exe_direction_preserved"] is True
    assert carryover["block_n_entry_confirms_exe_direction"] is True
    false_keys = [key for key in carryover if key.endswith("_now")]
    assert false_keys
    assert all(carryover[key] is False for key in false_keys)
    true_keys = [key for key in carryover if key.endswith("explicit_gate")]
    assert true_keys
    assert all(carryover[key] is True for key in true_keys)


def test_fail_closed_entry_decision_opens_only_block_n_and_blocks_16_0_execution() -> None:
    decision = payload()["fail_closed_entry_decision"]
    assert decision["block_n_entry_in_16_0"] == "opened"
    assert decision["block_n_read_model_in_16_1"] == "allowed"
    assert decision["missing_block_m_closure_audit_policy"] == "fail_closed"
    blocked = {key: value for key, value in decision.items() if key.endswith("_in_16_0")}
    assert blocked
    assert all(
        value == "blocked" for key, value in blocked.items() if key not in {"block_n_entry_in_16_0"}
    )


def test_non_execution_evidence_keeps_release_artifact_runtime_network_and_bridge_false() -> None:
    evidence = payload()["non_execution_evidence"]
    for key in [
        "source_block_m_closure_audit_read",
        "block_n_entry_contract_built",
        "block_n_entry_contract_only",
        "block_n_opened",
        "ready_for_block_n_1",
    ]:
        assert evidence[key] is True
    for key in [
        "release_executed",
        "release_published",
        "release_signed",
        "release_smoke_test_executed",
        "artifact_created",
        "artifact_mutated",
        "artifact_deleted",
        "artifact_smoke_test_executed",
        "packaging_dry_run_executed",
        "packaging_executed",
        "runtime_activated",
        "runtime_loop_started",
        "runtime_gate_executed",
        "gate_state_mutated",
        "order_generated",
        "order_submitted",
        "private_endpoint_accessed",
        "network_io_performed",
        "credential_read_performed",
        "live_trading_started",
        "qml_bridge_changed",
    ]:
        assert evidence[key] is False


def test_entry_boundaries_are_closed() -> None:
    boundaries = payload()["entry_boundaries"]
    assert boundaries["block_n_entry_contract_is_plain_data_only"] is True
    assert boundaries["block_n_entry_contract_is_source_only"] is True
    assert boundaries["block_n_entry_contract_opens_block_n"] is True
    cannot_keys = [key for key in boundaries if key.startswith("block_n_entry_contract_cannot_")]
    assert cannot_keys
    assert all(boundaries[key] is True for key in cannot_keys)


def test_source_boundaries_point_to_15_10_and_forbidden_calls_absent() -> None:
    boundaries = payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_m_closure_audit"] == "FUNCTIONAL-PREVIEW-15.10"
    assert boundaries["source_block_m_closure_audit_boundaries"]["allowed_imports_only"] is True
    false_keys = [key for key in boundaries if key.startswith("forbidden_")]
    assert false_keys
    assert all(boundaries[key] is False for key in false_keys)


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", tuple(alias.name for alias in node.names)))
        elif isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
    assert imports == [
        ("__future__", ("annotations",)),
        ("typing", ("Any", "Final")),
        ("ui.pyside_app.preview_block_m_closure_audit", ("build_preview_block_m_closure_audit",)),
    ]


def test_source_call_guard_blocks_io_network_runtime_release_packaging_and_bridge_calls() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    called_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                called_names.add(function.id)
            elif isinstance(function, ast.Attribute):
                called_names.add(function.attr)
    forbidden = {
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
        "activate",
        "start",
        "execute",
        "mutate",
        "release",
        "build_exe",
        "build_command",
        "package_exe",
        "bridge",
    }
    assert called_names.isdisjoint(forbidden)


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
