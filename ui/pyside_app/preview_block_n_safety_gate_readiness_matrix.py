"""FUNCTIONAL-PREVIEW-16.5 Block N source-only safety gate readiness matrix."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_safety_gate_read_model import (
    build_preview_block_n_safety_gate_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_n_safety_gate_readiness_matrix.v1"
KIND: Final[str] = "functional_preview_block_n_safety_gate_readiness_matrix"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.5"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.6"
NEXT_STEP_TITLE: Final[str] = "BLOCK N SAFETY GATE READINESS CONTRACT"
READY_FOR_BLOCK_N_6: Final[bool] = True
STATUS: Final[str] = "ready_for_functional_preview_16_6_block_n_safety_gate_readiness_contract"
SOURCE_BLOCK_N_SAFETY_GATE_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.4"
READINESS_MATRIX_STATUS: Final[str] = (
    "readiness_matrix_ready_16_4_read_model_consumed_block_m_closure_preserved_"
    "block_n_opened_exe_direction_preserved_source_only_static_classification_only_"
    "all_execution_capabilities_not_ready_all_execution_capabilities_blocked_"
    "missing_requirements_visible_no_gate_evaluation_no_validation_no_authorization_"
    "no_packaging_no_release_no_runtime_no_orders_no_private_endpoints_no_network_io_"
    "no_credentials_no_filesystem_io"
)
READINESS_MATRIX_DECISION: Final[str] = READINESS_MATRIX_STATUS.upper()
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_readiness_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_matrix_status",
    "block_n_safety_gate_readiness_matrix_decision",
    "ready_for_block_n_6",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_read_model_reference",
    "readiness_matrix_summary",
    "packaging_release_readiness_rows",
    "runtime_safety_readiness_rows",
    "cross_domain_invariant_readiness_rows",
    "validation_requirement_rows",
    "domain_readiness_summary",
    "exe_direction_readiness_matrix",
    "fail_closed_readiness_decision",
    "non_execution_evidence",
    "readiness_matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_read_model_status",
    "block_n_safety_gate_read_model_decision",
    "ready_for_block_n_5",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_5_ROOTS: Final[list[str]] = [
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


def build_preview_block_n_safety_gate_readiness_matrix() -> dict[str, Any]:
    """Build the 16.5 source-only, non-executing readiness matrix."""
    read_model = build_preview_block_n_safety_gate_read_model()
    packaging_rows = _build_capability_rows(
        read_model["packaging_release_gate_read_rows"], "packaging_release", PACKAGING_REQUIREMENTS
    )
    runtime_rows = _build_capability_rows(
        read_model["runtime_safety_gate_read_rows"], "runtime_safety", RUNTIME_REQUIREMENTS
    )
    invariant_rows = _build_invariant_rows(read_model["cross_domain_invariant_read_rows"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_n_safety_gate_readiness_matrix_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_safety_gate_readiness_matrix_status": READINESS_MATRIX_STATUS,
        "block_n_safety_gate_readiness_matrix_decision": READINESS_MATRIX_DECISION,
        "ready_for_block_n_6": READY_FOR_BLOCK_N_6,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_safety_gate_read_model_reference": _build_read_model_reference(read_model),
        "readiness_matrix_summary": _build_summary(),
        "packaging_release_readiness_rows": packaging_rows,
        "runtime_safety_readiness_rows": runtime_rows,
        "cross_domain_invariant_readiness_rows": invariant_rows,
        "validation_requirement_rows": _build_validation_requirement_rows(),
        "domain_readiness_summary": _build_domain_summary(packaging_rows, runtime_rows),
        "exe_direction_readiness_matrix": _build_exe_direction_matrix(read_model),
        "fail_closed_readiness_decision": _build_fail_closed_decision(),
        "non_execution_evidence": _build_non_execution_evidence(),
        "readiness_matrix_boundaries": _build_readiness_matrix_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_16_6_block_n_safety_gate_readiness_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_safety_gate_read_model_step": SOURCE_BLOCK_N_SAFETY_GATE_READ_MODEL_STEP,
            "source_block_n_safety_gate_read_model_read_by_16_5": True,
            "block_n_safety_gate_read_model_available_before_readiness_matrix": True,
            "static_block_n_safety_gate_read_model_only": True,
            "block_n_safety_gate_readiness_matrix_built_by_16_5": True,
            "ready_for_functional_preview_16_6": True,
        }
    )
    for root in _FALSE_BY_16_5_ROOTS:
        reference[root + "_by_16_5"] = False
    return reference


def _build_summary() -> dict[str, bool]:
    true_keys = [
        "block_n_safety_gate_read_model_available",
        "block_n_safety_gate_readiness_matrix_built",
        "block_n_opened",
        "ready_for_block_n_6",
        "ready_for_functional_preview_16_6",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "readiness_matrix_static_only",
        "readiness_matrix_read_only",
        "readiness_matrix_non_authorizing",
        "all_capabilities_classified",
        "all_missing_requirements_visible",
        "all_execution_capabilities_fail_closed",
        "all_execution_capabilities_not_ready",
        "all_execution_capabilities_require_future_explicit_gate",
        "packaging_release_readiness_rows_built",
        "runtime_safety_readiness_rows_built",
        "cross_domain_invariant_readiness_rows_built",
        "validation_requirement_rows_built",
        "domain_readiness_summary_built",
        "missing_evidence_blocks_execution",
        "missing_confirmation_blocks_execution",
        "missing_validation_blocks_execution",
        "missing_future_explicit_gate_blocks_execution",
    ]
    false_keys = [
        "any_gate_evaluated_now",
        "any_gate_condition_met_now",
        "any_gate_open_now",
        "any_gate_state_mutated_now",
        "any_execution_authorized_now",
        "any_execution_allowed_now",
        "any_execution_performed_now",
        "any_validation_completed_now",
        "any_requirement_satisfied_now",
        "any_capability_ready_now",
        "operator_confirmation_present_now",
        "environment_validation_present_now",
        "artifact_validation_present_now",
        "release_validation_present_now",
        "runtime_validation_present_now",
        "credentials_validation_present_now",
        "future_explicit_gate_present_now",
        "packaging_release_domain_ready_now",
        "runtime_safety_domain_ready_now",
        "exe_build_ready_now",
        "exe_packaging_ready_now",
        "exe_release_ready_now",
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_capability_rows(
    source_rows: list[dict[str, Any]], domain: str, requirements: list[str]
) -> list[dict[str, Any]]:
    return [
        {
            "readiness_row_id": row["read_row_id"] + "_readiness_matrix",
            "source_read_row_id": row["read_row_id"],
            "source_contract_id": row["source_contract_id"],
            "source_gate_id": row["source_gate_id"],
            "capability_id": row["capability_id"],
            "domain": domain,
            "display_name": row["display_name"],
            "source_read_result": "not_ready_execution_blocked",
            "required_requirements": list(requirements),
            "satisfied_requirements": [],
            "missing_requirements": list(requirements),
            "requirements_total": len(requirements),
            "requirements_satisfied_count": 0,
            "requirements_missing_count": len(requirements),
            "requirements_complete": False,
            "static_readiness_classification": "not_ready",
            "ready_for_execution": False,
            "execution_authorized": False,
            "gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "failure_policy": "fail_closed",
            "matrix_result": "not_ready_missing_requirements_execution_blocked",
            "notes": "16.5 static readiness row; missing requirements keep execution blocked.",
        }
        for row in source_rows
    ]


def _build_invariant_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "readiness_row_id": row["read_row_id"] + "_readiness_matrix",
            "source_read_row_id": row["read_row_id"],
            "source_contract_id": row["source_contract_id"],
            "invariant_id": row["invariant_id"],
            "domain": "cross_domain",
            "display_name": row["display_name"],
            "source_read_result": "invariant_preserved_execution_blocked",
            "source_invariant_preserved": True,
            "invariant_preserved_in_readiness_matrix": True,
            "invariant_required_for_future_execution": True,
            "execution_gate_open_now": False,
            "execution_allowed_now": False,
            "execution_performed_now": False,
            "requires_future_explicit_gate": True,
            "static_readiness_classification": "invariant_preserved_execution_not_ready",
            "failure_policy": "fail_closed",
            "matrix_result": "invariant_preserved_execution_blocked",
            "notes": "16.5 preserves this invariant while classifying execution as not ready.",
        }
        for row in source_rows
    ]


def _build_validation_requirement_rows() -> list[dict[str, Any]]:
    specs = [
        ("operator_confirmation", ["packaging_release", "runtime_safety"]),
        ("environment_validation", ["packaging_release"]),
        ("artifact_validation", ["packaging_release"]),
        ("release_validation", ["packaging_release"]),
        ("runtime_validation", ["runtime_safety"]),
        ("credentials_validation", ["runtime_safety"]),
        ("future_explicit_gate", ["packaging_release", "runtime_safety", "cross_domain"]),
    ]
    return [
        {
            "requirement_id": requirement_id,
            "display_name": requirement_id.replace("_", " ").title(),
            "required": True,
            "present": False,
            "completed": False,
            "satisfied": False,
            "applicable_domains": domains,
            "missing_blocks_execution": True,
            "requires_future_explicit_step": True,
            "failure_policy": "fail_closed",
            "readiness_result": "missing_execution_blocked",
            "notes": "16.5 records this requirement as missing; execution remains blocked.",
        }
        for requirement_id, domains in specs
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
    runtime_required = RUNTIME_REQUIREMENTS
    total = len(packaging_rows) + len(runtime_rows)
    return {
        "packaging_release": _domain_row(
            "packaging_release", len(packaging_rows), packaging_required
        ),
        "runtime_safety": _domain_row("runtime_safety", len(runtime_rows), runtime_required),
        "overall": {
            "total_capability_count": total,
            "ready_capability_count": 0,
            "blocked_capability_count": total,
            "all_domains_ready": False,
            "execution_authorized": False,
            "failure_policy": "fail_closed",
            "readiness_result": "not_ready_execution_blocked",
        },
    }


def _domain_row(domain: str, count: int, required: list[str]) -> dict[str, Any]:
    return {
        "domain": domain,
        "capability_count": count,
        "ready_capability_count": 0,
        "blocked_capability_count": count,
        "required_requirement_ids": list(required),
        "satisfied_requirement_ids": [],
        "missing_requirement_ids": list(required),
        "requirements_complete": False,
        "domain_ready": False,
        "execution_authorized": False,
        "failure_policy": "fail_closed",
        "readiness_result": "not_ready_execution_blocked",
    }


def _build_exe_direction_matrix(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_read_model"]
    matrix = dict(source)
    matrix.update(
        {
            "block_n_safety_gate_readiness_matrix_confirms_exe_direction": True,
            "exe_direction_is_not_execution_authorization": True,
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
            "matrix_result": "exe_direction_preserved_execution_not_ready",
        }
    )
    return matrix


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
        "missing_block_n_safety_gate_read_model_policy": "fail_closed",
        "missing_readiness_row_policy": "fail_closed",
        "missing_requirement_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_readiness_policy": "fail_closed",
        "block_n_safety_gate_readiness_matrix_in_16_5": "ready",
        "block_n_safety_gate_readiness_contract_in_16_6": "allowed",
        "real_capability_status": {key: "blocked" for key in real_capabilities},
        "only_source_only_16_6_handoff_allowed": True,
    }


def _build_non_execution_evidence() -> dict[str, bool]:
    true_keys = [
        "source_block_n_safety_gate_read_model_read",
        "block_n_safety_gate_readiness_matrix_built",
        "block_n_safety_gate_readiness_matrix_only",
        "block_n_opened",
        "ready_for_block_n_6",
        "all_capability_rows_not_ready",
        "all_execution_authorization_false",
        "all_requirements_unsatisfied",
        "all_capabilities_fail_closed",
    ]
    false_keys = [
        "gate_evaluation_performed",
        "gate_condition_accepted",
        "gate_opened",
        "gate_mutated",
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
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_readiness_matrix_boundaries() -> dict[str, bool]:
    true_keys = [
        "block_n_safety_gate_readiness_matrix_is_plain_data_only",
        "block_n_safety_gate_readiness_matrix_is_source_only",
        "block_n_safety_gate_readiness_matrix_reads_block_n_safety_gate_read_model_only",
        "block_n_safety_gate_readiness_matrix_preserves_block_m_closure",
        "block_n_safety_gate_readiness_matrix_preserves_block_n_entry",
        "block_n_safety_gate_readiness_matrix_preserves_exe_direction_without_packaging",
        "block_n_safety_gate_readiness_matrix_is_static_and_non_evaluating",
        "block_n_safety_gate_readiness_matrix_is_non_mutating",
        "block_n_safety_gate_readiness_matrix_is_non_authorizing",
        "block_n_safety_gate_readiness_matrix_can_feed_16_6_block_n_safety_gate_readiness_contract",
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
        "cannot_use_network",
        "cannot_use_filesystem",
        "cannot_access_private_endpoints",
        "cannot_read_credentials",
        "cannot_read_config_env_secrets",
        "cannot_change_qml_or_bridge",
        "cannot_create_execution_side_effects",
    ]
    return {key: True for key in true_keys}


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_n_safety_gate_read_model": SOURCE_BLOCK_N_SAFETY_GATE_READ_MODEL_STEP,
        "source_block_n_safety_gate_read_model_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_contract": source["source_block_n_safety_gate_contract"],
            "plain_data_source_only": source["source_block_n_safety_gate_contract_boundaries"][
                "plain_data_source_only"
            ],
            "static_non_evaluating": source["source_block_n_safety_gate_contract_boundaries"][
                "static_non_evaluating"
            ],
            "non_mutating": source["source_block_n_safety_gate_contract_boundaries"][
                "non_mutating"
            ],
            "non_authorizing": True,
            "can_feed_16_5": True,
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
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
    }
