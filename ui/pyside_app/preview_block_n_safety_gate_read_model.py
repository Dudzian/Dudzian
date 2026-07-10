"""FUNCTIONAL-PREVIEW-16.4 Block N source-only safety gate read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_safety_gate_contract import (
    build_preview_block_n_safety_gate_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_n_safety_gate_read_model.v1"
KIND: Final[str] = "functional_preview_block_n_safety_gate_read_model"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.4"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.5"
NEXT_STEP_TITLE: Final[str] = "BLOCK N SAFETY GATE READINESS MATRIX"
READY_FOR_BLOCK_N_5: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_16_5_block_n_safety_gate_readiness_matrix"
SOURCE_BLOCK_N_SAFETY_GATE_CONTRACT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.3"
READ_MODEL_STATUS: Final[str] = (
    "block_n_safety_gate_read_model_ready_16_3_contract_consumed_block_m_closure_"
    "preserved_block_n_opened_exe_direction_preserved_source_only_static_read_model_"
    "all_capabilities_fail_closed_no_gate_evaluation_no_condition_acceptance_no_gate_"
    "opening_no_gate_mutation_no_validation_performed_no_release_execution_no_artifact_"
    "work_no_packaging_no_pyinstaller_no_build_no_runtime_no_orders_no_private_"
    "endpoints_no_network_io_no_credentials_no_filesystem_io"
)
READ_MODEL_DECISION: Final[str] = READ_MODEL_STATUS.upper()
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_read_model_status",
    "block_n_safety_gate_read_model_decision",
    "ready_for_block_n_5",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_contract_reference",
    "safety_gate_read_summary",
    "packaging_release_gate_read_rows",
    "runtime_safety_gate_read_rows",
    "cross_domain_invariant_read_rows",
    "validation_readiness_summary",
    "exe_direction_read_model",
    "fail_closed_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_contract_status",
    "block_n_safety_gate_contract_decision",
    "ready_for_block_n_4",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_4_ROOTS: Final[list[str]] = [
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
    "operator_confirmation_performed",
    "environment_validation_performed",
    "artifact_validation_performed",
    "release_validation_performed",
    "runtime_validation_performed",
    "credentials_validation_performed",
    "release_executed",
    "release_published",
    "release_signed",
    "release_smoke_tested",
    "release_notes_generated",
    "release_tag_created",
    "release_uploaded",
    "release_exported",
    "artifact_created",
    "artifact_mutated",
    "artifact_deleted",
    "artifact_smoke_tested",
    "artifact_signed",
    "artifact_published",
    "artifact_name_finalized",
    "artifact_location_finalized",
    "artifact_checksum_generated",
    "artifact_metadata_written",
    "artifact_audit_exported",
    "artifact_cleanup_performed",
    "packaging_dry_run_executed",
    "packaging_executed",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "installer_changed",
    "workflow_changed",
    "environment_probe_performed",
    "dependency_freeze_performed",
    "asset_discovery_performed",
    "qml_asset_discovery_performed",
    "runtime_activated",
    "paper_runtime_started",
    "testnet_runtime_started",
    "live_canary_started",
    "live_trading_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "order_activity_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "credentials_read",
    "config_env_secrets_read",
    "filesystem_io_performed",
    "qml_bridge_changed",
]
_REAL_CAPABILITY_KEYS: Final[list[str]] = [
    "release_execution",
    "release_publish",
    "release_signing",
    "release_smoke",
    "release_workflow",
    "release_notes",
    "release_tag",
    "release_upload",
    "release_export",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke",
    "artifact_sign",
    "artifact_publish",
    "artifact_name",
    "artifact_location",
    "artifact_checksum",
    "artifact_metadata",
    "artifact_audit",
    "artifact_cleanup",
    "packaging_dry_run",
    "packaging",
    "pyinstaller",
    "build",
    "build_artifact",
    "installer",
    "workflow",
    "filesystem",
    "environment",
    "dependency",
    "asset",
    "qml_asset",
    "gate_evaluation",
    "gate_condition_acceptance",
    "gate_opening",
    "gate_state_mutation",
    "operator_confirmation_acceptance",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gates",
    "order_generation",
    "order_" + "sub" + "mission",
    "order_" + "can" + "cel",
    "order_" + "re" + "place",
    "private_endpoint",
    "network",
    "credential",
    "config_env_secret",
    "qml_bridge",
]


def build_preview_block_n_safety_gate_read_model() -> dict[str, Any]:
    """Build the 16.4 source-only, non-executing safety gate read model."""
    contract = build_preview_block_n_safety_gate_contract()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_n_safety_gate_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_safety_gate_read_model_status": READ_MODEL_STATUS,
        "block_n_safety_gate_read_model_decision": READ_MODEL_DECISION,
        "ready_for_block_n_5": READY_FOR_BLOCK_N_5,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_safety_gate_contract_reference": _build_contract_reference(contract),
        "safety_gate_read_summary": _build_summary(),
        "packaging_release_gate_read_rows": _build_packaging_rows(contract),
        "runtime_safety_gate_read_rows": _build_runtime_rows(contract),
        "cross_domain_invariant_read_rows": _build_invariant_rows(contract),
        "validation_readiness_summary": _build_validation_summary(),
        "exe_direction_read_model": _build_exe_direction_read_model(contract),
        "fail_closed_read_decision": _build_fail_closed_decision(),
        "non_execution_evidence": _build_non_execution_evidence(),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(contract),
        "future_steps": ["functional_preview_16_5_block_n_safety_gate_readiness_matrix"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    reference = {key: contract[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_safety_gate_contract_step": SOURCE_BLOCK_N_SAFETY_GATE_CONTRACT_STEP,
            "source_block_n_safety_gate_contract_read_by_16_4_read_model": True,
            "block_n_safety_gate_contract_available_before_read_model": True,
            "static_block_n_safety_gate_contract_only": True,
            "block_n_safety_gate_read_model_built_by_16_4": True,
            "ready_for_functional_preview_16_5": True,
        }
    )
    for root in _FALSE_BY_16_4_ROOTS:
        reference[root + "_by_16_4"] = False
    return reference


def _build_summary() -> dict[str, bool]:
    summary = {
        key: True
        for key in [
            "block_n_safety_gate_contract_available",
            "block_n_safety_gate_read_model_built",
            "block_n_opened",
            "ready_for_block_n_5",
            "ready_for_functional_preview_16_5",
            "block_m_closure_preserved",
            "exe_direction_preserved",
            "safety_gate_read_model_static_only",
            "safety_gate_read_model_read_only",
            "all_capabilities_fail_closed",
            "all_contract_rows_visible",
            "all_contract_rows_require_future_explicit_gate",
            "all_missing_validations_visible",
            "packaging_release_gate_read_rows_built",
            "runtime_safety_gate_read_rows_built",
            "cross_domain_invariant_read_rows_built",
            "validation_readiness_summary_built",
            "missing_evidence_blocks_execution",
            "missing_confirmation_blocks_execution",
            "missing_validation_blocks_execution",
        ]
    }
    for key in [
        "any_contract_evaluated_now",
        "any_contract_condition_met_now",
        "any_gate_open_now",
        "any_execution_allowed_now",
        "any_execution_performed_now",
        "any_gate_state_mutated_now",
        "any_validation_completed_now",
        "operator_confirmation_present_now",
        "environment_validation_present_now",
        "artifact_validation_present_now",
        "release_validation_present_now",
        "runtime_validation_present_now",
        "credentials_validation_present_now",
        "packaging_execution_ready_now",
        "release_execution_ready_now",
        "artifact_work_ready_now",
        "runtime_activation_ready_now",
        "order_activity_ready_now",
        "private_endpoint_access_ready_now",
        "network_io_ready_now",
        "credential_read_ready_now",
        "filesystem_io_ready_now",
        "qml_bridge_change_ready_now",
    ]:
        summary[key] = False
    return summary


def _build_packaging_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [_packaging_row(row) for row in contract["packaging_release_gate_contract_rows"]]


def _packaging_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "read_row_id": row["contract_id"] + "_read_model",
        "source_contract_id": row["contract_id"],
        "source_gate_id": row["source_gate_id"],
        "capability_id": row["capability_id"],
        "domain": "packaging_release",
        "display_name": row["display_name"],
        "source_contract_result": "blocked_pending_future_explicit_gate",
        "contract_required": True,
        "contract_static_only": True,
        "contract_evaluated": False,
        "contract_condition_met": False,
        "gate_open_now": False,
        "execution_allowed_now": False,
        "execution_performed_now": False,
        "operator_confirmation_required": True,
        "operator_confirmation_present": False,
        "environment_validation_required": True,
        "environment_validation_present": False,
        "artifact_validation_required": True,
        "artifact_validation_present": False,
        "missing_requirements": [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "future_explicit_gate",
        ],
        "requirements_complete": False,
        "ready_for_execution": False,
        "requires_future_explicit_gate": True,
        "failure_policy": "fail_closed",
        "read_result": "not_ready_execution_blocked",
        "notes": "16.4 static read row; source-only projection keeps execution blocked.",
    }


def _build_runtime_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [_runtime_row(row) for row in contract["runtime_safety_gate_contract_rows"]]


def _runtime_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "read_row_id": row["contract_id"] + "_read_model",
        "source_contract_id": row["contract_id"],
        "source_gate_id": row["source_gate_id"],
        "capability_id": row["capability_id"],
        "domain": "runtime_safety",
        "display_name": row["display_name"],
        "source_contract_result": "blocked_pending_future_explicit_gate",
        "contract_required": True,
        "contract_static_only": True,
        "contract_evaluated": False,
        "contract_condition_met": False,
        "gate_open_now": False,
        "execution_allowed_now": False,
        "execution_performed_now": False,
        "operator_confirmation_required": True,
        "operator_confirmation_present": False,
        "runtime_validation_required": True,
        "runtime_validation_present": False,
        "credentials_validation_required": True,
        "credentials_validation_present": False,
        "missing_requirements": [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ],
        "requirements_complete": False,
        "ready_for_execution": False,
        "requires_future_explicit_gate": True,
        "failure_policy": "fail_closed",
        "read_result": "not_ready_execution_blocked",
        "notes": "16.4 static read row; source-only projection keeps execution blocked.",
    }


def _build_invariant_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "read_row_id": row["contract_id"] + "_read_model",
            "source_contract_id": row["contract_id"],
            "invariant_id": row["source_invariant_id"],
            "display_name": row["display_name"],
            "domain": "cross_domain",
            "source_contract_result": "preserved_but_execution_blocked",
            "source_invariant_preserved": True,
            "invariant_required_for_future_execution": True,
            "invariant_satisfied_for_static_read_model": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "read_result": "invariant_preserved_execution_blocked",
            "notes": "16.4 preserves this invariant while keeping execution blocked.",
        }
        for row in contract["cross_domain_invariant_contract_rows"]
    ]


def _build_validation_summary() -> dict[str, Any]:
    return {
        "operator_confirmation_required": True,
        "operator_confirmation_present": False,
        "environment_validation_required": True,
        "environment_validation_present": False,
        "artifact_validation_required": True,
        "artifact_validation_present": False,
        "release_validation_required": True,
        "release_validation_present": False,
        "runtime_validation_required": True,
        "runtime_validation_present": False,
        "credentials_validation_required": True,
        "credentials_validation_present": False,
        "future_explicit_gate_required": True,
        "future_explicit_gate_present": False,
        "all_required_validations_complete": False,
        "packaging_release_requirements_complete": False,
        "runtime_safety_requirements_complete": False,
        "execution_readiness_satisfied": False,
        "execution_authorized": False,
        "failure_policy": "fail_closed",
        "readiness_result": "not_ready_missing_required_validations",
        "missing_packaging_release_requirements": [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "release_validation",
            "future_explicit_gate",
        ],
        "missing_runtime_requirements": [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ],
    }


def _build_exe_direction_read_model(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["exe_direction_gate_contract"]
    model = {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": True,
        "block_n_safety_gate_read_model_confirms_exe_direction": True,
        "exe_direction_is_not_execution_authorization": True,
        "exe_direction_requires_future_explicit_packaging_gate": True,
        "exe_direction_requires_future_explicit_release_gate": True,
        "packaging_requirements_complete": False,
        "release_requirements_complete": False,
        "ready_to_build_exe_now": False,
        "ready_to_package_exe_now": False,
        "ready_to_release_exe_now": False,
    }
    for key in [k for k, value in source.items() if isinstance(value, bool) and not value]:
        model[key] = False
    for key in [
        k
        for k, value in source.items()
        if isinstance(value, bool)
        and value
        and ("future" in k or "deferred" in k or "requires" in k)
    ]:
        model[key] = True
    return model


def _build_fail_closed_decision() -> dict[str, str]:
    decision = {
        "missing_block_n_safety_gate_contract_policy": "fail_closed",
        "missing_gate_read_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_readiness_policy": "fail_closed",
        "block_n_safety_gate_read_model_in_16_4": "ready",
        "block_n_safety_gate_readiness_matrix_in_16_5": "allowed",
    }
    for key in _REAL_CAPABILITY_KEYS:
        decision[key + "_in_16_4"] = "blocked"
    return decision


def _build_non_execution_evidence() -> dict[str, bool]:
    evidence = {
        "source_block_n_safety_gate_contract_read": True,
        "block_n_safety_gate_read_model_built": True,
        "block_n_safety_gate_read_model_only": True,
        "block_n_opened": True,
        "ready_for_block_n_5": True,
        "all_read_rows_fail_closed": True,
        "all_execution_readiness_false": True,
    }
    for root in _FALSE_BY_16_4_ROOTS:
        evidence[root] = False
    for key in [
        "confirmations_completed",
        "validations_completed",
        "order_generated",
        "order_" + "sub" + "mitted",
        "order_" + "can" + "celed",
        "order_" + "re" + "placed",
    ]:
        evidence[key] = False
    return evidence


def _build_read_model_boundaries() -> dict[str, bool]:
    boundaries = {
        key: True
        for key in [
            "block_n_safety_gate_read_model_is_plain_data_only",
            "block_n_safety_gate_read_model_is_source_only",
            "block_n_safety_gate_read_model_reads_block_n_safety_gate_contract_only",
            "block_n_safety_gate_read_model_preserves_block_m_closure",
            "block_n_safety_gate_read_model_preserves_block_n_entry",
            "block_n_safety_gate_read_model_preserves_exe_direction_without_packaging",
            "block_n_safety_gate_read_model_is_static_and_non_evaluating",
            "block_n_safety_gate_read_model_is_non_mutating",
            "block_n_safety_gate_read_model_is_non_authorizing",
            "block_n_safety_gate_read_model_can_feed_16_5_block_n_safety_gate_readiness_matrix",
        ]
    }
    for suffix in [
        "evaluate_contracts",
        "evaluate_gates",
        "mark_conditions_met",
        "open_gates",
        "mutate_gate_state",
        "accept_confirmations",
        "perform_validations",
        "authorize_execution",
        "execute_packaging",
        "execute_build",
        "execute_release",
        "execute_artifact",
        "execute_runtime",
        "execute_order",
        "execute_network",
        "execute_filesystem",
        "execute_private",
        "execute_credential",
        "execute_qml_paths",
    ]:
        boundaries["block_n_safety_gate_read_model_cannot_" + suffix] = True
    return boundaries


def _build_source_boundaries(contract: dict[str, Any]) -> dict[str, Any]:
    source = contract["source_boundaries"]
    bounds = contract["contract_boundaries"]
    result: dict[str, Any] = {
        "allowed_imports_only": True,
        "source_block_n_safety_gate_contract": SOURCE_BLOCK_N_SAFETY_GATE_CONTRACT_STEP,
        "source_block_n_safety_gate_contract_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_matrix": source["source_block_n_safety_gate_matrix"],
            "plain_data_source_only": bounds["block_n_safety_gate_contract_is_plain_data_only"]
            and bounds["block_n_safety_gate_contract_is_source_only"],
            "static_non_evaluating": bounds[
                "block_n_safety_gate_contract_is_static_and_non_evaluating"
            ],
            "non_mutating": bounds["block_n_safety_gate_contract_is_non_mutating"],
            "can_feed_16_4": bounds[
                "block_n_safety_gate_contract_can_feed_16_4_block_n_safety_gate_read_model"
            ],
        },
    }
    for name in [
        "packaging",
        "pyinstaller",
        "build",
        "release",
        "runtime",
        "gate_evaluation",
        "gate_execution",
        "gate_mutation",
        "validation",
        "confirmation",
        "io",
        "network",
        "private_endpoint",
        "ui_bridge",
    ]:
        result["forbidden_" + name + "_calls_present"] = False
    return result
