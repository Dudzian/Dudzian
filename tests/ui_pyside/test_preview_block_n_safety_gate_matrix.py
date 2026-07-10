"""Tests for FUNCTIONAL-PREVIEW-16.2 Block N safety gate matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_n_safety_gate_matrix import (
    BLOCK_ID,
    BLOCK_N_SAFETY_GATE_MATRIX_DECISION,
    BLOCK_N_SAFETY_GATE_MATRIX_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_KIND,
    PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_N_3,
    STATUS,
    STEP_ID,
    build_preview_block_n_safety_gate_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_n_safety_gate_matrix.py"
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_matrix_status",
    "block_n_safety_gate_matrix_decision",
    "ready_for_block_n_3",
    "next_step",
    "next_step_title",
    "block_n_read_model_reference",
    "safety_gate_summary",
    "packaging_release_gate_rows",
    "runtime_safety_gate_rows",
    "cross_domain_invariant_gate_rows",
    "exe_direction_safety_gate",
    "fail_closed_gate_decision",
    "non_execution_evidence",
    "gate_matrix_boundaries",
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
INVARIANT_IDS = [
    "block_m_closure_preserved",
    "block_n_entry_preserved",
    "exe_direction_preserved_without_execution",
    "no_live_credentials_embedded",
    "no_network_required_for_static_matrix",
    "runtime_disabled_during_packaging_and_release",
    "operator_confirmation_required_before_execution",
    "artifact_validation_required_before_release",
    "release_rollback_policy_required",
    "release_publication_requires_future_explicit_gate",
    "packaging_environment_validation_deferred",
    "filesystem_side_effects_forbidden_in_16_2",
]
SUMMARY_TRUE_KEYS = [
    "block_n_read_model_available",
    "block_n_opened",
    "block_n_safety_gate_matrix_built",
    "ready_for_block_n_3",
    "ready_for_functional_preview_16_3",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "safety_gate_matrix_static_only",
    "safety_gate_matrix_read_only",
    "all_execution_gates_fail_closed",
    "packaging_release_gate_rows_built",
    "runtime_safety_gate_rows_built",
    "cross_domain_invariant_gate_rows_built",
]


def payload() -> dict[str, Any]:
    return build_preview_block_n_safety_gate_matrix()


def test_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_next_and_readiness_are_16_2() -> None:
    built = payload()
    assert built["schema_version"] == PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_SCHEMA_VERSION
    assert built["block_n_safety_gate_matrix_kind"] == PREVIEW_BLOCK_N_SAFETY_GATE_MATRIX_KIND
    assert built["block"] == BLOCK_ID == "N"
    assert built["step"] == STEP_ID == "16.2"
    assert built["block_n_safety_gate_matrix_status"] == BLOCK_N_SAFETY_GATE_MATRIX_STATUS
    assert built["block_n_safety_gate_matrix_decision"] == BLOCK_N_SAFETY_GATE_MATRIX_DECISION
    assert built["ready_for_block_n_3"] is READY_FOR_BLOCK_N_3 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-16.3"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "BLOCK N SAFETY GATE CONTRACT"
    assert built["status"] == STATUS


def test_block_n_read_model_reference_points_to_16_1() -> None:
    reference = payload()["block_n_read_model_reference"]
    assert reference["source_block_n_read_model_step"] == "FUNCTIONAL-PREVIEW-16.1"
    assert reference["step"] == "16.1"
    assert reference["block"] == "N"
    assert reference["ready_for_block_n_2"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-16.2"
    assert reference["next_step_title"] == "BLOCK N SAFETY GATE MATRIX"
    for key in [
        "source_block_n_read_model_read_by_16_2_gate_matrix",
        "block_n_read_model_available_before_gate_matrix",
        "static_block_n_read_model_only",
        "block_n_safety_gate_matrix_built_by_16_2",
        "ready_for_functional_preview_16_3",
    ]:
        assert reference[key] is True
    false_keys = [key for key in reference if key.endswith("_by_16_2")]
    assert false_keys
    assert all(
        reference[key] is False
        for key in false_keys
        if key != "block_n_safety_gate_matrix_built_by_16_2"
    )


def test_safety_gate_summary_and_no_execution_gate_open() -> None:
    summary = payload()["safety_gate_summary"]
    assert all(summary[key] is True for key in SUMMARY_TRUE_KEYS)
    false_keys = [
        key for key in summary if key.endswith("_open_now") or key.endswith("_present_now")
    ]
    assert false_keys
    assert all(summary[key] is False for key in false_keys)


def test_packaging_release_gate_rows_exact_static_closed_fail_closed() -> None:
    rows = payload()["packaging_release_gate_rows"]
    assert {row["capability_id"] for row in rows} == PACKAGING_IDS
    assert [row["capability_id"] for row in rows] == [
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
    ]
    for row in rows:
        assert row["gate_id"] == "block_n_" + row["capability_id"] + "_gate"
        assert row["domain"] == "packaging_release"
        assert row["source_blocked_in_16_1"] is True
        assert row["source_allowed_now"] is False
        assert row["source_executed_now"] is False
        assert row["required_before_execution"] is True
        assert row["static_gate_row"] is True
        assert row["gate_evaluated_by_16_2"] is False
        assert row["gate_condition_met"] is False
        assert row["gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["operator_confirmation_present"] is False
        assert row["environment_validation_present"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["gate_result"] == "blocked"


def test_runtime_gate_rows_exact_static_closed_fail_closed() -> None:
    rows = payload()["runtime_safety_gate_rows"]
    assert {row["capability_id"] for row in rows} == RUNTIME_IDS
    for row in rows:
        assert row["domain"] == "runtime_safety"
        assert row["source_blocked_in_16_1"] is True
        assert row["source_allowed_now"] is False
        assert row["source_executed_now"] is False
        assert row["required_before_execution"] is True
        assert row["static_gate_row"] is True
        assert row["gate_evaluated_by_16_2"] is False
        assert row["gate_condition_met"] is False
        assert row["gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["operator_confirmation_present"] is False
        assert row["runtime_validation_present"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["gate_result"] == "blocked"


def test_cross_domain_invariants_preserved_without_execution() -> None:
    rows = payload()["cross_domain_invariant_gate_rows"]
    assert [row["invariant_id"] for row in rows] == INVARIANT_IDS
    for row in rows:
        assert row["domain"] == "cross_domain"
        assert row["static_invariant"] is True
        assert row["source_evidence_available"] is True
        assert row["invariant_preserved_by_16_2"] is True
        assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["gate_result"] == "preserved_but_execution_blocked"


def test_exe_direction_preserved_but_not_execution_authorization() -> None:
    gate = payload()["exe_direction_safety_gate"]
    assert gate["final_product_direction"] == "desktop_exe"
    assert gate["exe_direction_preserved"] is True
    assert gate["block_n_safety_gate_matrix_confirms_exe_direction"] is True
    assert gate["exe_direction_is_not_execution_authorization"] is True
    false_keys = [key for key in gate if key.endswith("_open_now") or key.endswith("_started_now")]
    false_keys += [
        "build_command_added_now",
        "build_command_executed_now",
        "workflow_changed_for_packaging_now",
        "installer_changed_now",
        "release_artifact_created_now",
        "release_executed_now",
        "release_published_now",
        "artifact_created_now",
        "artifact_mutated_now",
        "artifact_deleted_now",
        "artifact_smoke_test_executed_now",
        "artifact_signed_now",
        "artifact_published_now",
    ]
    assert all(gate[key] is False for key in false_keys)
    true_future_keys = [
        key
        for key in gate
        if key.startswith("future_") or key.endswith("_deferred_to_future_explicit_block")
    ]
    assert true_future_keys
    assert all(gate[key] is True for key in true_future_keys)


def test_fail_closed_decision_allows_only_source_only_16_3_and_blocks_real_capabilities() -> None:
    decision = payload()["fail_closed_gate_decision"]
    assert decision["block_n_safety_gate_matrix_in_16_2"] == "ready"
    assert decision["block_n_safety_gate_contract_in_16_3"] == "allowed"
    policy_keys = [key for key in decision if key.endswith("_policy")]
    assert all(decision[key] == "fail_closed" for key in policy_keys)
    blocked = [key for key in decision if key.endswith("_in_16_2")]
    blocked.remove("block_n_safety_gate_matrix_in_16_2")
    assert blocked
    assert all(decision[key] == "blocked" for key in blocked)


def test_non_execution_evidence_and_boundaries() -> None:
    built = payload()
    evidence = built["non_execution_evidence"]
    for key in [
        "source_block_n_read_model_read",
        "block_n_safety_gate_matrix_built",
        "block_n_safety_gate_matrix_only",
        "block_n_opened",
        "ready_for_block_n_3",
        "all_execution_gates_fail_closed",
    ]:
        assert evidence[key] is True
    assert all(
        value is False
        for key, value in evidence.items()
        if key
        not in {
            "source_block_n_read_model_read",
            "block_n_safety_gate_matrix_built",
            "block_n_safety_gate_matrix_only",
            "block_n_opened",
            "ready_for_block_n_3",
            "all_execution_gates_fail_closed",
        }
    )
    boundaries = built["gate_matrix_boundaries"]
    assert all(boundaries[key] is True for key in boundaries)


def test_source_boundaries_point_to_16_1() -> None:
    boundaries = payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_n_read_model"] == "FUNCTIONAL-PREVIEW-16.1"
    assert boundaries["source_block_n_read_model_boundaries"]["allowed_imports_only"] is True
    assert (
        boundaries["source_block_n_read_model_boundaries"]["source_block_n_entry_contract"]
        == "FUNCTIONAL-PREVIEW-16.0"
    )
    assert boundaries["source_block_n_read_model_boundaries"]["can_feed_16_2_boundary"] is True
    forbidden_keys = [key for key in boundaries if key.startswith("forbidden_")]
    assert forbidden_keys
    assert all(boundaries[key] is False for key in forbidden_keys)


def test_source_import_guard() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    modules = {
        getattr(node, "module", None) for node in imports if isinstance(node, ast.ImportFrom)
    }
    assert modules == {"__future__", "typing", "ui.pyside_app.preview_block_n_read_model"}


def test_source_call_guard_and_forbidden_literal_tokens() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
        "TradingController",
        "DecisionEnvelope",
    }
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    called_names = {func.id for func in calls if isinstance(func, ast.Name)}
    called_attrs = {func.attr for func in calls if isinstance(func, ast.Attribute)}
    assert not forbidden_call_names & called_names
    assert not forbidden_call_names & called_attrs
    forbidden_literals = [
        "create_" + "order",
        "submit_" + "order",
        "cancel_" + "order",
        "replace_" + "order",
        "fetch_" + "balance",
        "cc" + "xt",
    ]
    for token in forbidden_literals:
        assert token not in source


def test_no_row_opens_allows_or_performs_execution() -> None:
    built = payload()
    rows = (
        built["packaging_release_gate_rows"]
        + built["runtime_safety_gate_rows"]
        + built["cross_domain_invariant_gate_rows"]
    )
    for row in rows:
        if "gate_open_now" in row:
            assert row["gate_open_now"] is False
        if "execution_gate_open_now" in row:
            assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
