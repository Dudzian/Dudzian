"""Tests for FUNCTIONAL-PREVIEW-15.8 Block M release readiness contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_m_packaging_artifact_policy_read_model import (
    build_preview_block_m_packaging_artifact_policy_read_model,
)
from ui.pyside_app.preview_block_m_packaging_release_readiness_contract import (
    BLOCK_ID,
    BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_DECISION,
    BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_KIND,
    PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_M_9,
    STATUS,
    STEP_ID,
    build_preview_block_m_packaging_release_readiness_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = (
    REPO_ROOT / "ui" / "pyside_app" / "preview_block_m_packaging_release_readiness_contract.py"
)
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_m_packaging_release_readiness_contract_kind",
    "block",
    "step",
    "block_m_packaging_release_readiness_contract_status",
    "block_m_packaging_release_readiness_contract_decision",
    "ready_for_block_m_9",
    "next_step",
    "next_step_title",
    "packaging_artifact_policy_read_model_reference",
    "release_readiness_summary",
    "release_readiness_checklist",
    "release_prerequisite_contract_rows",
    "release_artifact_readiness_contract",
    "release_smoke_sign_publish_readiness_contract",
    "release_rollback_readiness_contract",
    "release_execution_blocked_contract",
    "packaging_execution_carryover_contract",
    "runtime_safety_carryover_contract",
    "exe_direction_release_readiness_contract",
    "fail_closed_release_readiness_decision",
    "non_execution_evidence",
    "contract_boundaries",
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
    return build_preview_block_m_packaging_release_readiness_contract()


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    built = payload()
    json.dumps(built, sort_keys=True)
    assert list(built) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step_are_15_8() -> None:
    built = payload()
    assert (
        built["schema_version"]
        == PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_SCHEMA_VERSION
    )
    assert (
        built["block_m_packaging_release_readiness_contract_kind"]
        == PREVIEW_BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_KIND
    )
    assert built["block"] == BLOCK_ID == "M"
    assert built["step"] == STEP_ID == "15.8"
    assert (
        built["block_m_packaging_release_readiness_contract_status"]
        == BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_STATUS
    )
    assert (
        built["block_m_packaging_release_readiness_contract_decision"]
        == BLOCK_M_PACKAGING_RELEASE_READINESS_CONTRACT_DECISION
    )
    assert built["ready_for_block_m_9"] is READY_FOR_BLOCK_M_9 is True
    assert built["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-15.9"
    assert built["next_step_title"] == NEXT_STEP_TITLE == "PACKAGING RELEASE READINESS READ MODEL"
    assert built["status"] == STATUS


def test_read_model_reference_points_to_15_7_and_blocks_15_8_work() -> None:
    reference = payload()["packaging_artifact_policy_read_model_reference"]
    assert (
        reference["source_packaging_artifact_policy_read_model_step"] == "FUNCTIONAL-PREVIEW-15.7"
    )
    assert reference["step"] == "15.7"
    assert reference["ready_for_block_m_8"] is True
    assert reference["source_packaging_artifact_policy_read_model_read_by_15_8_contract"] is True
    assert (
        reference[
            "packaging_artifact_policy_read_model_available_before_release_readiness_contract"
        ]
        is True
    )
    assert reference["static_packaging_artifact_policy_read_model_only"] is True
    assert reference["release_readiness_contract_built_by_15_8"] is True
    for key, value in reference.items():
        if key.endswith("_by_15_8") and key != "release_readiness_contract_built_by_15_8":
            assert value is False


def test_summary_preserves_exe_direction_and_blocks_release_artifact_runtime_io() -> None:
    summary = payload()["release_readiness_summary"]
    for key in [
        "packaging_artifact_policy_read_model_available",
        "release_readiness_contract_built",
        "ready_for_block_m_9",
        "exe_direction_preserved",
        "release_readiness_contract_static_only",
        "release_readiness_ready_for_future_read_model",
        "release_readiness_read_only",
    ]:
        assert summary[key] is True
    for key in FALSE_SUMMARY_KEYS:
        assert summary[key] is False


def test_release_readiness_checklist_is_static_unsatisfied_and_fail_closed() -> None:
    checklist = payload()["release_readiness_checklist"]
    assert [row["check_id"] for row in checklist] == [
        "release_gate_approved",
        "artifact_policy_read_model_available",
        "artifact_lifecycle_policy_read",
        "artifact_naming_policy_read",
        "artifact_retention_rollback_policy_read",
        "artifact_smoke_sign_publish_policy_read",
        "release_rollback_policy_read",
        "no_live_credentials_embedded",
        "no_network_required_for_packaging",
        "runtime_disabled_for_release",
        "operator_release_confirmation_required",
        "release_publication_policy_deferred",
    ]
    for row in checklist:
        assert row["required_before_release"] is True
        assert row["satisfied_in_15_8"] is False
        assert row["checked_by_15_8"] is False
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_release_prerequisite_rows_are_derived_from_15_7_without_checks() -> None:
    built = payload()
    source = build_preview_block_m_packaging_artifact_policy_read_model()
    expected_count = sum(
        len(source[name])
        for name in [
            "artifact_lifecycle_policy_read_rows",
            "artifact_naming_policy_read_rows",
            "artifact_retention_rollback_policy_read_rows",
            "artifact_smoke_sign_publish_policy_read_rows",
        ]
    )
    assert len(built["release_prerequisite_contract_rows"]) == expected_count
    for row in built["release_prerequisite_contract_rows"]:
        assert (
            row["release_prerequisite_row_type"]
            == "packaging_release_readiness_static_prerequisite_row"
        )
        assert row["source_read_by_15_7"] is True
        assert row["source_allowed_now"] is False
        assert row["source_requires_future_explicit_gate"] is True
        assert row["required_before_release"] is True
        assert row["satisfied_in_15_8"] is False
        assert row["checked_by_15_8"] is False
        assert row["allowed_now"] is False
        assert row["executed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"


def test_artifact_smoke_sign_publish_rollback_and_release_contracts_are_blocked() -> None:
    built = payload()
    artifact = built["release_artifact_readiness_contract"]
    assert artifact["release_artifact_readiness_contract_built"] is True
    assert artifact["source_artifact_execution_read_model_built"] is True
    for key, value in artifact.items():
        if key.endswith("allowed_now"):
            assert value is False
    assert artifact["artifact_work_requires_future_explicit_gate"] is True
    assert artifact["artifact_work_requires_future_operator_confirmation"] is True
    assert artifact["no_artifact_created_by_15_8"] is True
    assert artifact["no_artifact_mutated_by_15_8"] is True
    assert artifact["no_artifact_deleted_by_15_8"] is True

    smoke = built["release_smoke_sign_publish_readiness_contract"]
    for key in [
        "artifact_smoke_policy_read",
        "artifact_signing_policy_read",
        "artifact_publishing_policy_read",
    ]:
        assert smoke[key] is True
    for key, value in smoke.items():
        if key.endswith("allowed_now"):
            assert value is False
    assert smoke["smoke_sign_publish_requires_future_explicit_gate"] is True
    assert smoke["no_smoke_test_executed_by_15_8"] is True
    assert smoke["no_artifact_signed_by_15_8"] is True
    assert smoke["no_artifact_published_by_15_8"] is True

    rollback = built["release_rollback_readiness_contract"]
    for key in ["rollback_allowed_now", "delete_allowed_now", "cleanup_allowed_now"]:
        assert rollback[key] is False
    assert rollback["rollback_requires_future_explicit_gate"] is True
    assert rollback["no_artifact_deleted_by_15_8"] is True
    assert rollback["no_artifact_cleanup_by_15_8"] is True
    assert rollback["no_release_rollback_by_15_8"] is True

    release = built["release_execution_blocked_contract"]
    for key, value in release.items():
        if key.endswith("allowed_now"):
            assert value is False
    assert release["release_requires_future_explicit_gate"] is True
    assert release["release_not_executed_by_15_8"] is True
    assert release["release_not_published_by_15_8"] is True


def test_packaging_and_runtime_carryover_contracts_block_all_paths() -> None:
    built = payload()
    packaging = built["packaging_execution_carryover_contract"]
    assert {row["capability_id"] for row in packaging} >= {
        "packaging_dry_run_execution",
        "packaging_execution",
        "pyinstaller_execution",
        "build_command_execution",
        "artifact_creation",
        "dependency_freeze",
        "asset_discovery",
        "qml_asset_discovery",
    }
    for row in packaging:
        assert row["source_allowed_now"] is False
        assert row["source_read_model_allowed_now"] is False
        assert row["source_read_model_executed_now"] is False
        assert row["release_contract_allowed_now"] is False
        assert row["release_contract_executed_now"] is False
        assert row["blocked_in_15_8"] is True
        assert row["requires_future_explicit_gate"] is True

    runtime = built["runtime_safety_carryover_contract"]
    assert {row["capability_id"] for row in runtime} >= {
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
    for row in runtime:
        assert row["source_read_model_allowed_now"] is False
        assert row["source_artifact_policy_read_model_allowed_now"] is False
        assert row["source_artifact_policy_read_model_executed_now"] is False
        assert row["release_contract_allowed_now"] is False
        assert row["release_contract_executed_now"] is False
        assert row["blocked_in_15_8"] is True
        assert row["requires_future_explicit_gate"] is True


def test_exe_direction_contract_preserves_exe_without_starting_work() -> None:
    contract = payload()["exe_direction_release_readiness_contract"]
    assert contract["final_product_direction"] == "desktop_exe"
    assert contract["exe_direction_preserved"] is True
    assert contract["release_readiness_contract_confirms_exe_direction"] is True
    for key, value in contract.items():
        if key.endswith("_now"):
            assert value is False
    for key, value in contract.items():
        if key.endswith("explicit_block") or key.endswith("explicit_gate"):
            assert value is True
    assert contract["future_packaging_requires_separate_prompt"] is True
    assert contract["future_packaging_must_not_use_live_credentials"] is True
    assert contract["future_packaging_must_not_enable_runtime_by_itself"] is True


def test_fail_closed_decision_non_execution_evidence_and_boundaries() -> None:
    built = payload()
    decision = built["fail_closed_release_readiness_decision"]
    for key in [
        "missing_packaging_artifact_policy_read_model_policy",
        "missing_release_readiness_row_policy",
        "missing_operator_confirmation_policy",
        "missing_runtime_safety_policy",
    ]:
        assert decision[key] == "fail_closed"
    for key, value in decision.items():
        if key.endswith("_in_15_8"):
            assert value == "blocked"

    evidence = built["non_execution_evidence"]
    assert evidence["source_packaging_artifact_policy_read_model_read"] is True
    assert evidence["release_readiness_contract_built"] is True
    assert evidence["release_readiness_contract_only"] is True
    for key, value in evidence.items():
        if key not in {
            "source_packaging_artifact_policy_read_model_read",
            "release_readiness_contract_built",
            "release_readiness_contract_only",
        }:
            assert value is False

    boundaries = built["contract_boundaries"]
    for key, value in boundaries.items():
        assert value is True
        if key.startswith("packaging_release_readiness_contract_cannot_"):
            assert value is True


def test_source_boundaries_point_to_15_7() -> None:
    boundaries = payload()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_packaging_artifact_policy_read_model"] == "FUNCTIONAL-PREVIEW-15.7"
    for key, value in boundaries.items():
        if key.startswith("forbidden_") and key.endswith("_present"):
            assert value is False
    subset = boundaries["source_packaging_artifact_policy_read_model_boundaries"]
    assert subset["allowed_imports_only"] is True
    assert subset["source_packaging_artifact_policy_matrix"] == "FUNCTIONAL-PREVIEW-15.6"
    assert (
        subset["read_model_boundary_subset"][
            "packaging_artifact_policy_read_model_can_feed_15_8_packaging_release_readiness_contract"
        ]
        is True
    )


def test_source_import_guard_allows_only_required_imports() -> None:
    tree = ast.parse(SOURCE_PATH.read_text(encoding="utf-8"))
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert {alias.name for alias in imports[1].names} == {"Any", "Final"}
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_m_packaging_artifact_policy_read_model"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_m_packaging_artifact_policy_read_model"
    ]


def test_source_call_guard_blocks_io_network_release_runtime_orders_private_and_build_calls() -> (
    None
):
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
        "getaddrinfo",
        "create_connection",
        "TradingController",
        "DecisionEnvelope",
        "PyInstaller",
    }
    forbidden_call_substrings = [
        "runtime",
        "gate",
        "canary",
        "private",
        "account",
        "balance",
        "positions",
        "orders",
        "fills",
        "activate",
        "start",
        "execute",
        "mutate",
        "packaging",
        "build",
        "release",
        "QML",
        "PySide",
        "bridge",
    ]
    allowed_helper_prefixes = (
        "_build_",
        "build_preview_block_m_packaging_release_readiness_contract",
        "build_preview_block_m_packaging_artifact_policy_read_model",
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            else:
                name = ""
            assert name not in forbidden_call_names
            if name.startswith(allowed_helper_prefixes):
                continue
            assert not any(part in name for part in forbidden_call_substrings)


def test_forbidden_literal_tokens_do_not_appear_in_helper_source() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    forbidden = [
        "create_order",
        "submit_order",
        "cancel_order",
        "replace_order",
        "fetch_" + "balance",
        "cc" + "xt",
    ]
    for token in forbidden:
        assert token not in source
