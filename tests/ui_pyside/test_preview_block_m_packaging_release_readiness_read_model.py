"""Tests for FUNCTIONAL-PREVIEW-15.9 Block M release readiness read model."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_release_readiness_contract import (
    build_preview_block_m_packaging_release_readiness_contract,
)
from ui.pyside_app.preview_block_m_packaging_release_readiness_read_model import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_DECISION,
    BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_KIND,
    PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_10,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_release_readiness_read_model,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_release_readiness_read_model.py"
)
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_release_readiness_read_model_kind",
    "block",
    "step",
    "block_m_packaging_release_readiness_read_model_status",
    "block_m_packaging_release_readiness_read_model_decision",
    "ready_for_block_m_10",
    "next_step",
    "next_step_title",
    "packaging_release_readiness_contract_reference",
    "release_readiness_read_summary",
    "release_readiness_checklist_read_rows",
    "release_prerequisite_read_rows",
    "release_artifact_readiness_read_model",
    "release_smoke_sign_publish_readiness_read_model",
    "release_rollback_readiness_read_model",
    "release_execution_read_model",
    "packaging_execution_carryover_read_rows",
    "runtime_safety_carryover_read_rows",
    "exe_direction_release_readiness_read_model",
    "fail_closed_release_readiness_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
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
    return build_preview_block_m_packaging_release_readiness_read_model()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_9() -> None:
    built = payload()
    assert (
        built["schema_version"]
        == PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_SCHEMA_VERSION
    )
    assert (
        built["block_m_packaging_release_readiness_read_model_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_KIND
    )
    assert built["block"] == BLOCK_ID == "M"
    assert built["step"] == STEP_ID == "15.9"
    assert (
        built["block_m_packaging_release_readiness_read_model_status"]
        == BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_STATUS
    )
    assert (
        built["block_m_packaging_release_readiness_read_model_decision"]
        == BLOCK_M_PACKAGING_RELEASE_READINESS_READ_MODEL_DECISION
    )
    assert built["ready_for_block_m_10"] is READY_FOR_BLOCK_M_10 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.10"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "BLOCK M CLOSURE AUDIT"
    assert built["status"] == STATUS


def test_contract_reference_points_to_15_8_and_blocks_15_9_work() -> None:
    reference = payload()["packaging_release_readiness_contract_reference"]
    assert (
        reference["source_packaging_release_readiness_contract_step"] == "FUNCTIONAL-PREVIEW-15.8"
    )
    assert reference["step"] == "15.8"
    assert reference["ready_for_block_m_9"] is True
    assert reference["source_packaging_release_readiness_contract_read_by_15_9_read_model"] is True
    assert reference["packaging_release_readiness_contract_available_before_read_model"] is True
    assert reference["release_readiness_read_model_built_by_15_9"] is True
    false_keys = [key for key in reference if key.endswith("_by_15_9")]
    assert false_keys
    assert all(
        reference[key] is False
        for key in false_keys
        if key != "release_readiness_read_model_built_by_15_9"
    )


def test_release_readiness_read_summary_preserves_exe_and_blocks_all_execution() -> None:
    summary = payload()["release_readiness_read_summary"]
    assert summary["packaging_release_readiness_contract_available"] is True
    assert summary["release_readiness_read_model_built"] is True
    assert summary["ready_for_block_m_10"] is True
    assert summary["exe_direction_preserved"] is True
    assert summary["release_readiness_read_model_static_only"] is True
    assert summary["release_readiness_ready_for_block_m_closure_audit"] is True
    assert summary["release_readiness_read_only"] is True
    assert all(summary[key] is False for key in FALSE_SUMMARY_KEYS)


def test_release_readiness_checklist_read_rows_derive_from_15_8_and_are_fail_closed() -> None:
    built_rows = payload()["release_readiness_checklist_read_rows"]
    source_rows = build_preview_block_m_packaging_release_readiness_contract()[
        "release_readiness_checklist"
    ]
    assert [row["check_id"] for row in built_rows] == [row["check_id"] for row in source_rows]
    for row in built_rows:
        assert row["read_row_type"] == "packaging_release_readiness_static_checklist_read_row"
        assert row["source_required_before_release"] is True
        assert row["source_satisfied_in_15_8"] is False
        assert row["source_checked_by_15_8"] is False
        assert row["source_allowed_now"] is False
        assert row["source_executed_now"] is False
        assert row["required_before_future_release"] is True
        assert row["satisfied_in_15_9"] is False
        assert row["checked_by_15_9"] is False
        assert row["read_by_15_9"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_release_prerequisite_read_rows_derive_from_15_8_and_do_not_execute() -> None:
    built_rows = payload()["release_prerequisite_read_rows"]
    source_rows = build_preview_block_m_packaging_release_readiness_contract()[
        "release_prerequisite_contract_rows"
    ]
    assert [row["source_id"] for row in built_rows] == [row["source_id"] for row in source_rows]
    for row in built_rows:
        assert row["read_row_type"] == "packaging_release_readiness_static_prerequisite_read_row"
        assert row["source_read_by_15_7"] is True
        assert row["source_allowed_now"] is False
        assert row["source_requires_future_explicit_gate"] is True
        assert row["source_required_before_release"] is True
        assert row["source_satisfied_in_15_8"] is False
        assert row["source_checked_by_15_8"] is False
        assert row["required_before_future_release"] is True
        assert row["satisfied_in_15_9"] is False
        assert row["checked_by_15_9"] is False
        assert row["read_by_15_9"] is True
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_artifact_smoke_sign_publish_rollback_and_execution_models_block_work() -> None:
    built = payload()
    artifact = built["release_artifact_readiness_read_model"]
    assert artifact["release_artifact_readiness_read_model_built"] is True
    assert artifact["source_release_artifact_readiness_contract_built"] is True
    assert artifact["source_artifact_execution_read_model_built"] is True
    assert all(value is False for key, value in artifact.items() if key.endswith("allowed_now"))
    assert artifact["artifact_work_requires_future_explicit_gate"] is True
    assert artifact["no_artifact_created_by_15_9"] is True

    smoke = built["release_smoke_sign_publish_readiness_read_model"]
    assert smoke["release_smoke_sign_publish_readiness_read_model_built"] is True
    assert smoke["source_release_smoke_sign_publish_readiness_contract_built"] is True
    assert smoke["smoke_sign_publish_requires_future_explicit_gate"] is True
    assert all(value is False for key, value in smoke.items() if key.endswith("allowed_now"))

    rollback = built["release_rollback_readiness_read_model"]
    assert rollback["release_rollback_readiness_read_model_built"] is True
    assert rollback["source_release_rollback_readiness_contract_built"] is True
    assert rollback["rollback_allowed_now"] is False
    assert rollback["delete_allowed_now"] is False
    assert rollback["cleanup_allowed_now"] is False
    assert rollback["rollback_requires_future_explicit_gate"] is True

    execution = built["release_execution_read_model"]
    assert execution["release_execution_read_model_built"] is True
    assert execution["source_release_execution_blocked_contract_built"] is True
    assert execution["release_requires_future_explicit_gate"] is True
    assert all(value is False for key, value in execution.items() if key.endswith("allowed_now"))


def test_carryover_rows_block_packaging_and_runtime_safety() -> None:
    built = payload()
    packaging_rows = built["packaging_execution_carryover_read_rows"]
    assert len(packaging_rows) == len(
        build_preview_block_m_packaging_release_readiness_contract()[
            "packaging_execution_carryover_contract"
        ]
    )
    for row in packaging_rows:
        assert row["blocked_in_15_9"] is True
        assert row["requires_future_explicit_gate"] is True
        assert all(
            value is False for key, value in row.items() if "allowed" in key or "executed" in key
        )

    runtime_rows = built["runtime_safety_carryover_read_rows"]
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
    assert {row["capability_id"] for row in runtime_rows} == expected
    for row in runtime_rows:
        assert row["blocked_in_15_9"] is True
        assert row["requires_future_explicit_gate"] is True
        assert all(
            value is False for key, value in row.items() if "allowed" in key or "executed" in key
        )


def test_exe_direction_read_model_preserves_direction_without_starting_work() -> None:
    model = payload()["exe_direction_release_readiness_read_model"]
    assert model["final_product_direction"] == "desktop_exe"
    assert model["exe_direction_preserved"] is True
    assert model["release_readiness_read_model_confirms_exe_direction"] is True
    false_keys = [key for key in model if key.endswith("_now")]
    assert false_keys
    assert all(model[key] is False for key in false_keys)
    assert model["future_packaging_requires_explicit_gate"] is True
    assert model["future_release_requires_explicit_gate"] is True


def test_fail_closed_decision_non_execution_evidence_boundaries_and_source_boundaries() -> None:
    built = payload()
    decision = built["fail_closed_release_readiness_read_decision"]
    assert decision["missing_packaging_release_readiness_contract_policy"] == "fail_closed"
    assert all(value in {"fail_closed", "blocked"} for value in decision.values())
    assert decision["release_execution_in_15_9"] == "blocked"
    assert decision["order_" + "sub" + "mission_in_15_9"] == "blocked"
    assert decision["order_" + "can" + "cel_in_15_9"] == "blocked"
    assert decision["order_" + "re" + "place_in_15_9"] == "blocked"

    evidence = built["non_execution_evidence"]
    assert evidence["source_packaging_release_readiness_contract_read"] is True
    assert evidence["release_readiness_read_model_built"] is True
    assert evidence["release_readiness_read_model_only"] is True
    assert all(
        value is False
        for key, value in evidence.items()
        if key
        not in {
            "source_packaging_release_readiness_contract_read",
            "release_readiness_read_model_built",
            "release_readiness_read_model_only",
        }
    )

    boundaries = built["read_model_boundaries"]
    assert all(value is True for value in boundaries.values())
    assert (
        boundaries["packaging_release_readiness_read_model_cannot_" + "sub" + "mit_orders"] is True
    )
    assert (
        boundaries["packaging_release_readiness_read_model_cannot_" + "can" + "cel_orders"] is True
    )
    assert (
        boundaries["packaging_release_readiness_read_model_cannot_" + "re" + "place_orders"] is True
    )

    source = built["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_packaging_release_readiness_contract"] == "FUNCTIONAL-PREVIEW-15.8"
    assert (
        source["source_packaging_release_readiness_contract_boundaries"][
            "contract_boundary_subset"
        ][
            "packaging_release_readiness_contract_can_feed_15_9_packaging_release_readiness_read_model"
        ]
        is True
    )
    assert all(value is False for key, value in source.items() if key.startswith("forbidden_"))


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert [alias.name for alias in imports[1].names] == ["Any", "Final"]
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_m_packaging_release_readiness_contract"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_m_packaging_release_readiness_contract"
    ]


def test_source_call_guard_blocks_forbidden_calls_and_literal_tokens() -> None:
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
        "release",
    }
    seen_calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                seen_calls.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                seen_calls.add(node.func.attr)
    assert not (seen_calls & forbidden_call_names)

    forbidden_literals = [
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "balance" + "_fetch",
        "cc" + "xt",
    ]
    for token in forbidden_literals:
        assert token not in source
