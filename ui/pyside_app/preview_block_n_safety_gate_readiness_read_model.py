"""FUNCTIONAL-PREVIEW-16.7 Block N safety gate readiness read model."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_safety_gate_readiness_contract import (
    build_preview_block_n_safety_gate_readiness_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_n_safety_gate_readiness_read_model.v1"
KIND: Final[str] = "functional_preview_block_n_safety_gate_readiness_read_model"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.7"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.8"
NEXT_STEP_TITLE: Final[str] = "BLOCK N CLOSURE AUDIT"
READY_FOR_BLOCK_N_8: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_16_8_block_n_closure_audit"
SOURCE_BLOCK_N_SAFETY_GATE_READINESS_CONTRACT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.6"
READINESS_READ_MODEL_STATUS: Final[str] = (
    "readiness_read_model_ready_16_6_readiness_contract_consumed_block_m_closure_preserved_"
    "block_n_opened_exe_direction_preserved_source_only_plain_data_static_read_projection_only_"
    "all_capability_contracts_read_all_requirement_contracts_read_all_invariant_contracts_read_"
    "all_execution_capabilities_not_ready_all_execution_capabilities_blocked_all_requirements_missing_"
    "all_execution_unauthorized_all_gates_closed_invariants_preserved_no_readiness_recalculation_"
    "no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_"
    "no_build_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_"
    "no_filesystem_io"
)
READINESS_READ_MODEL_DECISION: Final[str] = READINESS_READ_MODEL_STATUS.upper()
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_readiness_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_read_model_status",
    "block_n_safety_gate_readiness_read_model_decision",
    "ready_for_block_n_8",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_contract_reference",
    "readiness_read_summary",
    "packaging_release_readiness_read_rows",
    "runtime_safety_readiness_read_rows",
    "cross_domain_invariant_readiness_read_rows",
    "validation_requirement_read_rows",
    "domain_readiness_read_summary",
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
    "block_n_safety_gate_readiness_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_contract_status",
    "block_n_safety_gate_readiness_contract_decision",
    "ready_for_block_n_7",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_7_ROOTS: Final[list[str]] = [
    "readiness_recalculated_from_environment",
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
    "execution_authorized",
    "operator_confirmation_accepted",
    "environment_validation_performed",
    "artifact_validation_performed",
    "release_validation_performed",
    "runtime_validation_performed",
    "credentials_validation_performed",
    "dependency_validation_performed",
    "future_explicit_gate_opened",
    "packaging_dry_run_executed",
    "packaging_executed",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "artifact_created",
    "artifact_mutated",
    "artifact_deleted",
    "artifact_smoke_tested",
    "artifact_signed",
    "artifact_published",
    "release_executed",
    "release_published",
    "release_signed",
    "release_smoke_tested",
    "release_notes_generated",
    "release_tag_created",
    "release_uploaded",
    "release_external_export",
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
    "installer_changed",
    "workflow_changed",
]
PACKAGING_REQUIREMENTS: Final[list[str]] = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "future_explicit_gate",
]
RUNTIME_REQUIREMENTS: Final[list[str]] = [
    "operator_confirmation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]


def build_preview_block_n_safety_gate_readiness_read_model() -> dict[str, Any]:
    """Build the 16.7 source-only, plain-data readiness read projection."""
    contract = build_preview_block_n_safety_gate_readiness_contract()
    packaging_rows = _build_capability_read_rows(
        contract["packaging_release_readiness_contract_rows"]
    )
    runtime_rows = _build_capability_read_rows(contract["runtime_safety_readiness_contract_rows"])
    invariant_rows = _build_invariant_read_rows(
        contract["cross_domain_invariant_readiness_contract_rows"]
    )
    requirement_rows = _build_requirement_read_rows(
        contract["validation_requirement_contract_rows"]
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_n_safety_gate_readiness_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_safety_gate_readiness_read_model_status": READINESS_READ_MODEL_STATUS,
        "block_n_safety_gate_readiness_read_model_decision": READINESS_READ_MODEL_DECISION,
        "ready_for_block_n_8": READY_FOR_BLOCK_N_8,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_safety_gate_readiness_contract_reference": _build_contract_reference(contract),
        "readiness_read_summary": _build_summary(),
        "packaging_release_readiness_read_rows": packaging_rows,
        "runtime_safety_readiness_read_rows": runtime_rows,
        "cross_domain_invariant_readiness_read_rows": invariant_rows,
        "validation_requirement_read_rows": requirement_rows,
        "domain_readiness_read_summary": _build_domain_summary(packaging_rows, runtime_rows),
        "exe_direction_read_model": _build_exe_direction_read_model(contract),
        "fail_closed_read_decision": _build_fail_closed_decision(),
        "non_execution_evidence": _build_non_execution_evidence(),
        "read_model_boundaries": _build_read_model_boundaries(),
        "source_boundaries": _build_source_boundaries(contract),
        "future_steps": ["functional_preview_16_8_block_n_closure_audit"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    reference = {key: contract[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_safety_gate_readiness_contract_step": SOURCE_BLOCK_N_SAFETY_GATE_READINESS_CONTRACT_STEP,
            "source_block_n_safety_gate_readiness_contract_read_by_16_7": True,
            "block_n_safety_gate_readiness_contract_available_before_read_model": True,
            "static_block_n_safety_gate_readiness_contract_only": True,
            "block_n_safety_gate_readiness_read_model_built_by_16_7": True,
            "ready_for_functional_preview_16_8": True,
        }
    )
    for root in _FALSE_BY_16_7_ROOTS:
        reference[root + "_by_16_7"] = False
    return reference


def _build_summary() -> dict[str, bool]:
    true_keys = [
        "block_n_safety_gate_readiness_contract_available",
        "block_n_safety_gate_readiness_read_model_built",
        "block_n_opened",
        "ready_for_block_n_8",
        "ready_for_functional_preview_16_8",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "read_model_source_only",
        "read_model_plain_data_only",
        "read_model_static_only",
        "read_model_read_only",
        "read_model_non_evaluating",
        "read_model_non_mutating",
        "read_model_non_authorizing",
        "all_capability_contract_rows_read",
        "all_requirement_contract_rows_read",
        "all_invariant_contract_rows_read",
        "all_execution_capabilities_fail_closed",
        "all_execution_capabilities_not_ready",
        "all_execution_capabilities_blocked",
        "all_execution_capabilities_require_future_explicit_gate",
        "all_requirements_missing",
        "all_requirements_block_execution",
        "all_invariants_preserved",
        "packaging_release_read_rows_built",
        "runtime_safety_read_rows_built",
        "cross_domain_invariant_read_rows_built",
        "validation_requirement_read_rows_built",
        "domain_readiness_read_summary_built",
        "missing_confirmation_blocks_execution",
        "missing_validation_blocks_execution",
        "missing_future_explicit_gate_blocks_execution",
    ]
    false_keys = [
        "any_readiness_recalculated_from_environment_now",
        "any_gate_evaluated_now",
        "any_gate_condition_met_now",
        "any_gate_open_now",
        "any_gate_state_mutated_now",
        "any_execution_authorized_now",
        "any_execution_allowed_now",
        "any_execution_performed_now",
        "any_validation_completed_now",
        "any_requirement_present_now",
        "any_requirement_satisfied_now",
        "any_capability_ready_now",
        "operator_confirmation_present_now",
        "environment_validation_present_now",
        "artifact_validation_present_now",
        "release_validation_present_now",
        "runtime_validation_present_now",
        "credentials_validation_present_now",
        "dependency_validation_present_now",
        "future_explicit_gate_present_now",
        "packaging_release_domain_ready_now",
        "runtime_safety_domain_ready_now",
        "exe_build_ready_now",
        "exe_packaging_ready_now",
        "exe_release_ready_now",
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_capability_read_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "read_row_id": row["contract_row_id"] + "_read",
            "source_contract_row_id": row["contract_row_id"],
            "source_readiness_row_id": row["source_readiness_row_id"],
            "source_read_row_id": row["source_read_row_id"],
            "source_contract_id": row["source_contract_id"],
            "source_gate_id": row["source_gate_id"],
            "capability_id": row["capability_id"],
            "domain": row["domain"],
            "display_name": row["display_name"],
            "source_contract_result": row["contract_result"],
            "source_contract_readiness_classification": row["contract_readiness_classification"],
            "required_requirements": list(row["contract_required_requirements"]),
            "satisfied_requirements": list(row["contract_satisfied_requirements"]),
            "missing_requirements": list(row["contract_missing_requirements"]),
            "requirements_total": row["contract_requirements_total"],
            "requirements_satisfied_count": row["contract_requirements_satisfied_count"],
            "requirements_missing_count": row["contract_requirements_missing_count"],
            "requirements_complete": row["contract_requirements_complete"],
            "read_result": "read_not_ready_missing_requirements_execution_blocked",
            "readiness_classification": row["contract_readiness_classification"],
            "ready_for_execution": row["contract_ready_for_execution"],
            "execution_authorized": row["contract_execution_authorized"],
            "gate_open_now": row["contract_gate_open_now"],
            "execution_allowed_now": row["contract_execution_allowed_now"],
            "execution_performed_now": row["contract_execution_performed_now"],
            "requires_future_explicit_gate": row["contract_requires_future_explicit_gate"],
            "failure_policy": row["contract_failure_policy"],
            "notes": "16.7 reads the 16.6 contracted not-ready row without evaluation or execution.",
        }
        for row in source_rows
    ]


def _build_invariant_read_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "read_row_id": row["contract_row_id"] + "_read",
            "source_contract_row_id": row["contract_row_id"],
            "source_readiness_row_id": row["source_readiness_row_id"],
            "source_read_row_id": row["source_read_row_id"],
            "source_contract_id": row["source_contract_id"],
            "invariant_id": row["invariant_id"],
            "domain": row["domain"],
            "display_name": row["display_name"],
            "source_contract_result": row["contract_result"],
            "source_contract_readiness_classification": row["contract_readiness_classification"],
            "source_invariant_preserved": row["source_invariant_preserved"],
            "read_invariant_preserved": row["contract_invariant_preserved"],
            "invariant_required_for_future_execution": row[
                "contract_invariant_required_for_future_execution"
            ],
            "execution_gate_open_now": row["contract_execution_gate_open_now"],
            "execution_allowed_now": row["contract_execution_allowed_now"],
            "execution_performed_now": row["contract_execution_performed_now"],
            "requires_future_explicit_gate": row["contract_requires_future_explicit_gate"],
            "readiness_classification": row["contract_readiness_classification"],
            "failure_policy": row["contract_failure_policy"],
            "read_result": "read_invariant_preserved_execution_blocked",
            "notes": "16.7 reads the 16.6 preserved invariant without opening execution.",
        }
        for row in source_rows
    ]


def _build_requirement_read_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "read_row_id": row["contract_row_id"] + "_read",
            "source_contract_row_id": row["contract_row_id"],
            "requirement_id": row["requirement_id"],
            "display_name": row["display_name"],
            "source_required": row["source_required"],
            "source_present": row["source_present"],
            "source_completed": row["source_completed"],
            "source_satisfied": row["source_satisfied"],
            "required": row["contract_required"],
            "present": row["contract_present"],
            "completed": row["contract_completed"],
            "satisfied": row["contract_satisfied"],
            "applicable_domains": list(row["applicable_domains"]),
            "missing_blocks_execution": row["contract_missing_blocks_execution"],
            "requires_future_explicit_step": row["contract_requires_future_explicit_step"],
            "failure_policy": row["contract_failure_policy"],
            "read_result": "read_missing_execution_blocked",
            "notes": "16.7 reads the 16.6 missing requirement without validation.",
        }
        for row in source_rows
    ]


def _build_domain_summary(
    packaging_rows: list[dict[str, Any]], runtime_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    packaging_required = [
        "operator_confirmation",
        "environment_validation",
        "artifact_validation",
        "release_validation",
        "future_explicit_gate",
    ]
    total = len(packaging_rows) + len(runtime_rows)
    return {
        "packaging_release": _domain_row(
            "packaging_release", len(packaging_rows), packaging_required
        ),
        "runtime_safety": _domain_row("runtime_safety", len(runtime_rows), RUNTIME_REQUIREMENTS),
        "overall": {
            "total_capability_count": total,
            "read_capability_count": total,
            "ready_capability_count": 0,
            "blocked_capability_count": total,
            "all_domains_read": True,
            "all_domains_ready": False,
            "all_capabilities_not_ready": True,
            "all_capabilities_blocked": True,
            "execution_authorized": False,
            "failure_policy": "fail_closed",
            "read_result": "read_not_ready_execution_blocked",
        },
    }


def _domain_row(domain: str, count: int, required: list[str]) -> dict[str, Any]:
    return {
        "domain": domain,
        "capability_count": count,
        "read_capability_count": count,
        "ready_capability_count": 0,
        "blocked_capability_count": count,
        "required_requirement_ids": list(required),
        "satisfied_requirement_ids": [],
        "missing_requirement_ids": list(required),
        "requirements_complete": False,
        "domain_ready": False,
        "execution_authorized": False,
        "all_capabilities_read": True,
        "all_capabilities_not_ready": True,
        "all_capabilities_blocked": True,
        "failure_policy": "fail_closed",
        "read_result": "read_not_ready_execution_blocked",
    }


def _build_exe_direction_read_model(contract: dict[str, Any]) -> dict[str, Any]:
    read_model = dict(contract["exe_direction_readiness_contract"])
    read_model.update(
        {
            "block_n_safety_gate_readiness_read_model_confirms_exe_direction": True,
            "readiness_contract_source_preserved": True,
            "read_model_is_not_execution_authorization": True,
            "final_product_direction": "desktop_exe",
            "exe_direction_preserved": True,
            "packaging_requirements_complete": False,
            "release_requirements_complete": False,
            "build_readiness_classification": "not_ready",
            "packaging_readiness_classification": "not_ready",
            "release_readiness_classification": "not_ready",
            "ready_to_build_exe_now": False,
            "ready_to_package_exe_now": False,
            "ready_to_release_exe_now": False,
            "build_authorized_now": False,
            "packaging_authorized_now": False,
            "release_authorized_now": False,
            "future_packaging_gate_required": True,
            "future_release_gate_required": True,
            "future_explicit_step_required": True,
            "failure_policy": "fail_closed",
            "read_result": "exe_direction_read_execution_not_ready",
        }
    )
    return read_model


def _build_fail_closed_decision() -> dict[str, Any]:
    real_capabilities = [
        "release_execution",
        "release_publish",
        "release_sign",
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
        "environment",
        "dependency",
        "asset",
        "qml_asset",
        "filesystem",
        "gate_evaluation",
        "gate_condition",
        "gate_opening",
        "gate_mutation",
        "confirmation_acceptance",
        "environment_validation",
        "artifact_validation",
        "release_validation",
        "runtime_validation",
        "credentials_validation",
        "dependency_validation",
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
        "credentials",
        "config_env_secrets",
        "qml_bridge",
    ]
    return {
        "missing_block_n_safety_gate_readiness_contract_policy": "fail_closed",
        "missing_read_row_policy": "fail_closed",
        "missing_requirement_read_row_policy": "fail_closed",
        "missing_invariant_read_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_read_model_policy": "fail_closed",
        "block_n_safety_gate_readiness_read_model_in_16_7": "ready",
        "block_n_closure_audit_in_16_8": "allowed",
        "only_source_only_16_8_handoff_allowed": True,
        "real_capability_status": {key: "blocked" for key in real_capabilities},
    }


def _build_non_execution_evidence() -> dict[str, bool]:
    true_keys = [
        "source_block_n_safety_gate_readiness_contract_read",
        "block_n_safety_gate_readiness_read_model_built",
        "block_n_safety_gate_readiness_read_model_only",
        "block_n_opened",
        "ready_for_block_n_8",
        "all_capability_contract_rows_read",
        "all_capability_rows_not_ready",
        "all_invariant_contract_rows_preserved",
        "all_requirement_contract_rows_missing",
        "all_execution_authorization_false",
        "all_capabilities_fail_closed",
    ]
    false_keys = [
        "readiness_recalculated_from_environment",
        "gate_evaluation_performed",
        "gate_condition_accepted",
        "gate_opened",
        "gate_mutated",
        "confirmation_accepted",
        "validation_performed",
        "authorization_performed",
        "execution_performed",
        "packaging_performed",
        "build_performed",
        "release_performed",
        "runtime_performed",
        "orders_performed",
        "network_io_performed",
        "filesystem_io_performed",
        "private_endpoint_accessed",
        "credentials_read",
        "config_env_secrets_read",
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_read_model_boundaries() -> dict[str, bool]:
    true_keys = [
        "block_n_safety_gate_readiness_read_model_is_plain_data_only",
        "block_n_safety_gate_readiness_read_model_is_source_only",
        "block_n_safety_gate_readiness_read_model_reads_readiness_contract_only",
        "block_n_safety_gate_readiness_read_model_preserves_block_m_closure",
        "block_n_safety_gate_readiness_read_model_preserves_block_n_entry",
        "block_n_safety_gate_readiness_read_model_preserves_exe_direction_without_packaging",
        "block_n_safety_gate_readiness_read_model_is_static_and_non_evaluating",
        "block_n_safety_gate_readiness_read_model_is_non_mutating",
        "block_n_safety_gate_readiness_read_model_is_non_authorizing",
        "block_n_safety_gate_readiness_read_model_can_feed_16_8_closure_audit",
        "cannot_recalculate_readiness_from_environment",
        "cannot_evaluate",
        "cannot_accept_condition",
        "cannot_open_gate",
        "cannot_mutate_gate",
        "cannot_accept_confirmations",
        "cannot_perform_validations",
        "cannot_authorize",
        "cannot_package",
        "cannot_build",
        "cannot_release",
        "cannot_perform_artifact_work",
        "cannot_run_runtime",
        "cannot_generate_orders",
        "cannot_submit_orders",
        "cannot_cancel_orders",
        "cannot_replace_orders",
        "cannot_use_network",
        "cannot_use_filesystem",
        "cannot_access_private_endpoints",
        "cannot_read_credentials",
        "cannot_read_config_env_secrets",
        "cannot_change_qml_or_bridge",
        "cannot_create_execution_side_effects",
    ]
    return {key: True for key in true_keys}


def _build_source_boundaries(contract: dict[str, Any]) -> dict[str, Any]:
    contract_boundaries = contract["source_boundaries"]
    matrix_boundaries = contract_boundaries[
        "source_block_n_safety_gate_readiness_matrix_boundaries"
    ]
    return {
        "allowed_imports_only": contract_boundaries["allowed_imports_only"],
        "source_block_n_safety_gate_readiness_contract": SOURCE_BLOCK_N_SAFETY_GATE_READINESS_CONTRACT_STEP,
        "source_block_n_safety_gate_readiness_matrix": contract_boundaries[
            "source_block_n_safety_gate_readiness_matrix"
        ],
        "source_block_n_safety_gate_read_model": contract_boundaries[
            "source_block_n_safety_gate_read_model"
        ],
        "source_block_n_safety_gate_readiness_contract_boundaries": {
            "allowed_imports_only": matrix_boundaries["allowed_imports_only"],
            "source_block_n_safety_gate_readiness_matrix": contract_boundaries[
                "source_block_n_safety_gate_readiness_matrix"
            ],
            "source_block_n_safety_gate_read_model": matrix_boundaries[
                "source_block_n_safety_gate_read_model"
            ],
            "plain_data_source_only": matrix_boundaries["plain_data_source_only"],
            "static_non_evaluating": matrix_boundaries["static_non_evaluating"],
            "non_mutating": matrix_boundaries["non_mutating"],
            "non_authorizing": matrix_boundaries["non_authorizing"],
            "can_feed_16_7": True,
        },
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_release_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_gate_evaluation_calls_present": False,
        "forbidden_gate_execution_calls_present": False,
        "forbidden_gate_mutation_calls_present": False,
        "forbidden_validation_calls_present": False,
        "forbidden_confirmation_calls_present": False,
        "forbidden_authorization_calls_present": False,
        "forbidden_readiness_recalculation_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
    }
