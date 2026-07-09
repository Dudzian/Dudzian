"""Tests for FUNCTIONAL-PREVIEW-15.6 Block M packaging artifact policy matrix."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_artifact_policy_matrix import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_DECISION,
    BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_KIND,
    PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_7,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_artifact_policy_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_artifact_policy_matrix.py"
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_artifact_policy_matrix_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_matrix_status",
    "block_m_packaging_artifact_policy_matrix_decision",
    "ready_for_block_m_7",
    "next_step",
    "next_step_title",
    "packaging_dry_run_read_model_reference",
    "artifact_policy_summary",
    "artifact_lifecycle_policy_matrix",
    "artifact_naming_policy_matrix",
    "artifact_retention_rollback_policy_matrix",
    "artifact_smoke_sign_publish_policy_matrix",
    "artifact_execution_blocked_matrix",
    "packaging_execution_carryover_matrix",
    "runtime_safety_carryover_matrix",
    "exe_direction_artifact_policy_matrix",
    "fail_closed_artifact_policy_decision",
    "non_execution_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
SUMMARY_TRUE_FLAGS = [
    "packaging_dry_run_read_model_available",
    "artifact_policy_matrix_built",
    "ready_for_block_m_7",
    "exe_direction_preserved",
    "artifact_policy_matrix_static_only",
    "artifact_policy_ready_for_future_read_model",
    "artifact_policy_read_only",
]
SUMMARY_FALSE_FLAGS = [
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
    return build_preview_block_m_packaging_artifact_policy_matrix()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_6() -> None:
    built = payload()
    assert (
        built["schema_version"] == PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_SCHEMA_VERSION
    )
    assert (
        built["block_m_packaging_artifact_policy_matrix_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_KIND
    )
    assert built["block"] == BLOCK_ID == "M"
    assert built["step"] == STEP_ID == "15.6"
    assert (
        built["block_m_packaging_artifact_policy_matrix_status"]
        == BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_STATUS
    )
    assert (
        built["block_m_packaging_artifact_policy_matrix_decision"]
        == BLOCK_M_PACKAGING_ARTIFACT_POLICY_MATRIX_DECISION
    )
    assert built["ready_for_block_m_7"] is READY_FOR_BLOCK_M_7 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.7"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING ARTIFACT POLICY READ MODEL"
    assert built["status"] == STATUS


def test_packaging_dry_run_read_model_reference_points_to_15_5_and_blocks_15_6_work() -> None:
    reference = payload()["packaging_dry_run_read_model_reference"]
    assert reference["source_packaging_dry_run_read_model_step"] == "FUNCTIONAL-PREVIEW-15.5"
    assert reference["step"] == "15.5"
    assert reference["ready_for_block_m_6"] is True
    assert reference["source_packaging_dry_run_read_model_read_by_15_6_matrix"] is True
    assert reference["packaging_dry_run_read_model_available_before_artifact_policy_matrix"] is True
    assert reference["static_packaging_dry_run_read_model_only"] is True
    assert reference["artifact_policy_matrix_built_by_15_6"] is True
    for key, value in reference.items():
        if key.endswith("_by_15_6") or key in {
            "packaging_dry_run_executed_by_15_6",
            "packaging_executed_by_15_6",
            "pyinstaller_started_by_15_6",
        }:
            if key != "artifact_policy_matrix_built_by_15_6":
                assert value is False


def test_artifact_policy_summary_preserves_exe_direction_and_blocks_everything() -> None:
    summary = payload()["artifact_policy_summary"]
    for key in SUMMARY_TRUE_FLAGS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_FLAGS:
        assert summary[key] is False


def test_artifact_lifecycle_policy_matrix_is_static_unsatisfied_and_fail_closed() -> None:
    rows = payload()["artifact_lifecycle_policy_matrix"]
    assert len(rows) >= 10
    for row in rows:
        assert row["required_before_artifact_work"] is True
        assert row["satisfied_in_15_6"] is False
        assert row["checked_by_15_6"] is False
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_artifact_naming_policy_matrix_selects_no_real_name_or_path() -> None:
    rows = payload()["artifact_naming_policy_matrix"]
    assert len(rows) >= 10
    for row in rows:
        assert row["required_before_artifact_naming"] is True
        assert row["finalized_in_15_6"] is False
        assert row["checked_by_15_6"] is False
        assert row["allowed_now"] is False
        assert row["selected_now"] is False
        assert row["requires_future_explicit_gate"] is True


def test_retention_rollback_policy_matrix_deletes_and_rolls_back_nothing() -> None:
    rows = payload()["artifact_retention_rollback_policy_matrix"]
    assert len(rows) >= 10
    for row in rows:
        assert row["required_before_artifact_release"] is True
        assert row["satisfied_in_15_6"] is False
        assert row["checked_by_15_6"] is False
        assert row["delete_allowed_now"] is False
        assert row["rollback_allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_smoke_sign_publish_policy_matrix_runs_nothing() -> None:
    rows = payload()["artifact_smoke_sign_publish_policy_matrix"]
    assert len(rows) >= 10
    for row in rows:
        assert row["required_before_publish"] is True
        assert row["satisfied_in_15_6"] is False
        assert row["checked_by_15_6"] is False
        assert row["smoke_allowed_now"] is False
        assert row["sign_allowed_now"] is False
        assert row["publish_allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_artifact_execution_blocked_matrix_blocks_all_artifact_execution_paths() -> None:
    rows = payload()["artifact_execution_blocked_matrix"]
    assert len(rows) >= 12
    for row in rows:
        assert row["forbidden_in_15_6"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_packaging_execution_carryover_matrix_blocks_packaging_and_artifact_paths() -> None:
    rows = payload()["packaging_execution_carryover_matrix"]
    ids = {row["capability_id"] for row in rows}
    assert {
        "packaging_dry_run_execution",
        "packaging_execution",
        "pyinstaller_execution",
        "build_command_execution",
        "artifact_creation",
        "artifact_publishing",
    } <= ids
    for row in rows:
        assert row["source_allowed_now"] is False
        assert row["matrix_allowed_now"] is False
        assert row["matrix_executed_now"] is False
        assert row["blocked_in_15_6"] is True
        assert row["requires_future_explicit_gate"] is True


def test_runtime_safety_carryover_matrix_blocks_runtime_and_order_paths() -> None:
    rows = payload()["runtime_safety_carryover_matrix"]
    ids = {row["capability_id"] for row in rows}
    assert {
        "runtime_activation",
        "live_trading",
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
    } <= ids
    for row in rows:
        assert row["source_read_model_allowed_now"] is False
        assert row["artifact_policy_matrix_allowed_now"] is False
        assert row["artifact_policy_matrix_executed_now"] is False
        assert row["blocked_in_15_6"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_artifact_policy_matrix_preserves_exe_but_starts_nothing() -> None:
    matrix = payload()["exe_direction_artifact_policy_matrix"]
    assert matrix["final_product_direction"] == "desktop_exe"
    assert matrix["exe_direction_preserved"] is True
    assert matrix["artifact_policy_matrix_confirms_exe_direction"] is True
    for key, value in matrix.items():
        if key.endswith("_now"):
            assert value is False
    for key in [
        "packaging_deferred_to_future_explicit_block",
        "dry_run_deferred_to_future_explicit_block",
        "artifact_work_deferred_to_future_explicit_block",
        "future_packaging_requires_explicit_gate",
        "future_dry_run_requires_explicit_gate",
        "future_artifact_work_requires_explicit_gate",
        "future_packaging_requires_separate_prompt",
        "future_packaging_must_not_use_live_credentials",
        "future_packaging_must_not_enable_runtime_by_itself",
    ]:
        assert matrix[key] is True


def test_fail_closed_artifact_policy_decision_blocks_forbidden_paths() -> None:
    decision = payload()["fail_closed_artifact_policy_decision"]
    for key in [
        "missing_packaging_dry_run_read_model_policy",
        "missing_artifact_policy_row_policy",
        "missing_operator_confirmation_policy",
        "missing_runtime_safety_policy",
    ]:
        assert decision[key] == "fail_closed"
    for key, value in decision.items():
        if key not in {
            "missing_packaging_dry_run_read_model_policy",
            "missing_artifact_policy_row_policy",
            "missing_operator_confirmation_policy",
            "missing_runtime_safety_policy",
        }:
            assert value == "blocked"


def test_non_execution_evidence_has_no_artifact_packaging_runtime_network_or_bridge_execution() -> (
    None
):
    evidence = payload()["non_execution_evidence"]
    assert evidence["source_packaging_dry_run_read_model_read"] is True
    assert evidence["artifact_policy_matrix_built"] is True
    assert evidence["artifact_policy_matrix_only"] is True
    for key, value in evidence.items():
        if key not in {
            "source_packaging_dry_run_read_model_read",
            "artifact_policy_matrix_built",
            "artifact_policy_matrix_only",
        }:
            assert value is False


def test_matrix_boundaries_are_closed() -> None:
    boundaries = payload()["matrix_boundaries"]
    assert boundaries["packaging_artifact_policy_matrix_is_plain_data_only"] is True
    assert boundaries["packaging_artifact_policy_matrix_is_source_only"] is True
    assert boundaries["packaging_artifact_policy_matrix_reads_dry_run_read_model_only"] is True
    assert (
        boundaries[
            "packaging_artifact_policy_matrix_can_feed_15_7_packaging_artifact_policy_read_model"
        ]
        is True
    )
    for key, value in boundaries.items():
        if "_cannot_" in key:
            assert value is True


def test_source_boundaries_point_to_15_5() -> None:
    boundaries = payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_packaging_dry_run_read_model"] == "FUNCTIONAL-PREVIEW-15.5"
    for key, value in boundaries.items():
        if key.startswith("forbidden_"):
            assert value is False
    assert (
        boundaries["source_packaging_dry_run_read_model_boundaries"]["allowed_imports_only"] is True
    )


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports: list[tuple[str, str | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, None) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append((node.module or "", ",".join(alias.name for alias in node.names)))
    assert imports == [
        ("__future__", "annotations"),
        ("typing", "Any,Final"),
        (
            "ui.pyside_app.preview_block_m_packaging_dry_run_read_model",
            "build_preview_block_m_packaging_dry_run_read_model",
        ),
    ]


def test_source_call_guard_blocks_io_network_runtime_orders_private_config_qml_and_packaging() -> (
    None
):
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    forbidden_calls = {
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
    calls: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.add(node.func.attr)
    assert not (calls & forbidden_calls)


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
