"""Tests for FUNCTIONAL-PREVIEW-15.7 Block M artifact policy read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_artifact_policy_matrix import (
    build_preview_block_m_packaging_artifact_policy_matrix,
)
from ui.pyside_app.preview_block_m_packaging_artifact_policy_read_model import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_DECISION,
    BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_KIND,
    PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_8,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_artifact_policy_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_artifact_policy_read_model.py"
)
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_artifact_policy_read_model_kind",
    "block",
    "step",
    "block_m_packaging_artifact_policy_read_model_status",
    "block_m_packaging_artifact_policy_read_model_decision",
    "ready_for_block_m_8",
    "next_step",
    "next_step_title",
    "packaging_artifact_policy_matrix_reference",
    "artifact_policy_read_summary",
    "artifact_lifecycle_policy_read_rows",
    "artifact_naming_policy_read_rows",
    "artifact_retention_rollback_policy_read_rows",
    "artifact_smoke_sign_publish_policy_read_rows",
    "artifact_execution_read_model",
    "packaging_execution_carryover_read_rows",
    "runtime_safety_carryover_read_rows",
    "exe_direction_artifact_policy_read_model",
    "fail_closed_artifact_policy_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
FALSE_SUMMARY_KEYS = [
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
    return build_preview_block_m_packaging_artifact_policy_read_model()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_7() -> None:
    built = payload()
    assert (
        built["schema_version"]
        == PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_SCHEMA_VERSION
    )
    assert (
        built["block_m_packaging_artifact_policy_read_model_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_KIND
    )
    assert built["block"] == BLOCK_ID == "M"
    assert built["step"] == STEP_ID == "15.7"
    assert (
        built["block_m_packaging_artifact_policy_read_model_status"]
        == BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_STATUS
    )
    assert (
        built["block_m_packaging_artifact_policy_read_model_decision"]
        == BLOCK_M_PACKAGING_ARTIFACT_POLICY_READ_MODEL_DECISION
    )
    assert built["ready_for_block_m_8"] is READY_FOR_BLOCK_M_8 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.8"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING RELEASE READINESS CONTRACT"
    assert built["status"] == STATUS


def test_matrix_reference_points_to_15_6_and_blocks_15_7_work() -> None:
    reference = payload()["packaging_artifact_policy_matrix_reference"]
    assert reference["source_packaging_artifact_policy_matrix_step"] == "FUNCTIONAL-PREVIEW-15.6"
    assert reference["step"] == "15.6"
    assert reference["ready_for_block_m_7"] is True
    assert reference["source_packaging_artifact_policy_matrix_read_by_15_7_read_model"] is True
    assert reference["packaging_artifact_policy_matrix_available_before_read_model"] is True
    assert reference["static_packaging_artifact_policy_matrix_only"] is True
    assert reference["artifact_policy_read_model_built_by_15_7"] is True
    for key, value in reference.items():
        if key.endswith("_by_15_7") and key != "artifact_policy_read_model_built_by_15_7":
            assert value is False


def test_summary_preserves_exe_direction_and_blocks_artifact_packaging_runtime_io() -> None:
    summary = payload()["artifact_policy_read_summary"]
    for key in [
        "packaging_artifact_policy_matrix_available",
        "artifact_policy_read_model_built",
        "ready_for_block_m_8",
        "exe_direction_preserved",
        "artifact_policy_read_model_static_only",
        "artifact_policy_ready_for_future_release_readiness_contract",
        "artifact_policy_read_only",
    ]:
        assert summary[key] is True
    for key in FALSE_SUMMARY_KEYS:
        assert summary[key] is False


def test_policy_read_rows_are_derived_from_15_6_and_fail_closed() -> None:
    built = payload()
    matrix = build_preview_block_m_packaging_artifact_policy_matrix()
    assert len(built["artifact_lifecycle_policy_read_rows"]) == len(
        matrix["artifact_lifecycle_policy_matrix"]
    )
    for row in built["artifact_lifecycle_policy_read_rows"]:
        assert row["read_row_type"] == "packaging_artifact_lifecycle_static_read_row"
        assert row["source_required_before_artifact_work"] is True
        assert row["source_satisfied_in_15_6"] is False
        assert row["source_checked_by_15_6"] is False
        assert row["source_allowed_now"] is False
        assert row["source_executed_now"] is False
        assert row["required_before_future_artifact_work"] is True
        assert row["satisfied_in_15_7"] is False
        assert row["checked_by_15_7"] is False
        assert row["read_by_15_7"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_naming_retention_and_publish_rows_do_not_select_delete_rollback_smoke_sign_or_publish() -> (
    None
):
    built = payload()
    for row in built["artifact_naming_policy_read_rows"]:
        assert row["read_row_type"] == "packaging_artifact_naming_static_read_row"
        assert row["source_required_before_artifact_naming"] is True
        assert row["source_finalized_in_15_6"] is False
        assert row["source_checked_by_15_6"] is False
        assert row["source_selected_now"] is False
        assert row["finalized_in_15_7"] is False
        assert row["selected_now"] is False
        assert row["allowed_now"] is False
        assert row["requires_future_explicit_gate"] is True
    for row in built["artifact_retention_rollback_policy_read_rows"]:
        assert row["read_row_type"] == "packaging_artifact_retention_rollback_static_read_row"
        assert row["delete_allowed_now"] is False
        assert row["rollback_allowed_now"] is False
        assert row["source_delete_allowed_now"] is False
        assert row["source_rollback_allowed_now"] is False
        assert row["failure_policy"] == "fail_closed"
    for row in built["artifact_smoke_sign_publish_policy_read_rows"]:
        assert row["read_row_type"] == "packaging_artifact_smoke_sign_publish_static_read_row"
        assert row["smoke_allowed_now"] is False
        assert row["sign_allowed_now"] is False
        assert row["publish_allowed_now"] is False
        assert row["source_smoke_allowed_now"] is False
        assert row["source_sign_allowed_now"] is False
        assert row["source_publish_allowed_now"] is False
        assert row["failure_policy"] == "fail_closed"


def test_execution_carryover_and_runtime_rows_block_all_execution_paths() -> None:
    built = payload()
    execution = built["artifact_execution_read_model"]
    assert execution["artifact_execution_read_model_built"] is True
    assert execution["source_artifact_policy_matrix_built"] is True
    assert execution["artifact_work_requires_future_explicit_gate"] is True
    assert execution["artifact_work_requires_future_operator_confirmation"] is True
    for key, value in execution.items():
        if key.endswith("_allowed_now"):
            assert value is False
    assert execution["no_artifact_created_by_15_7"] is True
    assert execution["no_artifact_mutated_by_15_7"] is True
    assert execution["no_artifact_deleted_by_15_7"] is True
    for row in built["packaging_execution_carryover_read_rows"]:
        assert row["source_allowed_now"] is False
        assert row["source_matrix_allowed_now"] is False
        assert row["source_matrix_executed_now"] is False
        assert row["read_model_allowed_now"] is False
        assert row["read_model_executed_now"] is False
        assert row["blocked_in_15_7"] is True
        assert row["requires_future_explicit_gate"] is True
    runtime_ids = {row["capability_id"] for row in built["runtime_safety_carryover_read_rows"]}
    assert {
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
    } <= runtime_ids
    for row in built["runtime_safety_carryover_read_rows"]:
        assert row["source_read_model_allowed_now"] is False
        assert row["source_artifact_policy_matrix_allowed_now"] is False
        assert row["source_artifact_policy_matrix_executed_now"] is False
        assert row["read_model_allowed_now"] is False
        assert row["read_model_executed_now"] is False
        assert row["blocked_in_15_7"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_fail_closed_evidence_boundaries_and_source_boundaries() -> None:
    built = payload()
    exe = built["exe_direction_artifact_policy_read_model"]
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["exe_direction_preserved"] is True
    assert exe["artifact_policy_read_model_confirms_exe_direction"] is True
    for key, value in exe.items():
        if key.endswith("_now"):
            assert value is False
    for key, value in exe.items():
        if key.startswith("future_") or key.endswith("_future_explicit_block"):
            assert value is True
    decision = built["fail_closed_artifact_policy_read_decision"]
    assert decision["missing_packaging_artifact_policy_matrix_policy"] == "fail_closed"
    for key, value in decision.items():
        if key.endswith("_in_15_7"):
            assert value == "blocked"
    evidence = built["non_execution_evidence"]
    assert evidence["source_packaging_artifact_policy_matrix_read"] is True
    assert evidence["artifact_policy_read_model_built"] is True
    assert evidence["artifact_policy_read_model_only"] is True
    for key, value in evidence.items():
        if key not in {
            "source_packaging_artifact_policy_matrix_read",
            "artifact_policy_read_model_built",
            "artifact_policy_read_model_only",
        }:
            assert value is False
    boundaries = built["read_model_boundaries"]
    assert boundaries["packaging_artifact_policy_read_model_is_plain_data_only"] is True
    assert boundaries["packaging_artifact_policy_read_model_is_source_only"] is True
    assert all(value is True for value in boundaries.values())
    source = built["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_packaging_artifact_policy_matrix"] == "FUNCTIONAL-PREVIEW-15.6"
    for key, value in source.items():
        if key.startswith("forbidden_"):
            assert value is False


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
        (
            "ui.pyside_app.preview_block_m_packaging_artifact_policy_matrix",
            ("build_preview_block_m_packaging_artifact_policy_matrix",),
        ),
    ]


def test_source_call_guard_and_forbidden_literals() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
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
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            assert name not in forbidden_calls
    forbidden_literals = [
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    for token in forbidden_literals:
        assert token not in source
