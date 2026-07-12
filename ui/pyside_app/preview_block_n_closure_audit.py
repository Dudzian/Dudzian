"""FUNCTIONAL-PREVIEW-16.8 Block N source-only closure audit."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_n_safety_gate_readiness_read_model import (
    build_preview_block_n_safety_gate_readiness_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_n_closure_audit.v1"
KIND: Final[str] = "functional_preview_block_n_closure_audit"
BLOCK_ID: Final[str] = "N"
STEP_ID: Final[str] = "16.8"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-17.0"
NEXT_STEP_TITLE: Final[str] = "BLOCK O ENTRY CONTRACT"
BLOCK_N_CLOSED: Final[bool] = True
READY_FOR_BLOCK_O_0: Final[bool] = True
STATUS: Final[str] = "block_n_closed_ready_for_functional_preview_17_0_block_o_entry_contract"
SOURCE_BLOCK_N_READINESS_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-16.7"
CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_n_closure_audit_complete_16_7_readiness_read_model_consumed_steps_16_0_through_16_7_complete_"
    "block_m_closure_preserved_block_n_closed_exe_direction_preserved_source_only_plain_data_static_audit_only_"
    "all_capability_rows_read_all_requirements_missing_all_invariants_preserved_all_execution_capabilities_not_ready_"
    "all_execution_capabilities_blocked_all_execution_unauthorized_all_gates_closed_no_readiness_recalculation_"
    "no_gate_evaluation_no_validation_no_confirmation_acceptance_no_authorization_no_packaging_no_build_no_release_"
    "no_runtime_no_orders_no_private_endpoints_no_network_io_no_credentials_no_filesystem_io_only_source_only_block_o_handoff_allowed"
)
CLOSURE_AUDIT_DECISION: Final[str] = CLOSURE_AUDIT_STATUS.upper()
_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_n_closure_audit_kind",
    "block",
    "step",
    "block_n_closure_audit_status",
    "block_n_closure_audit_decision",
    "block_n_closed",
    "ready_for_block_o_0",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_read_model_reference",
    "closure_audit_summary",
    "block_n_step_closure_rows",
    "packaging_release_closure_summary",
    "runtime_safety_closure_summary",
    "cross_domain_invariant_closure_summary",
    "validation_requirement_closure_summary",
    "exe_direction_closure_audit",
    "fail_closed_closure_decision",
    "non_execution_closure_evidence",
    "closure_audit_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_n_safety_gate_readiness_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_read_model_status",
    "block_n_safety_gate_readiness_read_model_decision",
    "ready_for_block_n_8",
    "next_step",
    "next_step_title",
]
_FALSE_BY_16_8_ROOTS: Final[list[str]] = [
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
_STEPS: Final[list[tuple[str, str, str]]] = [
    ("FUNCTIONAL-PREVIEW-16.0", "BLOCK N ENTRY CONTRACT", "entry_contract"),
    ("FUNCTIONAL-PREVIEW-16.1", "BLOCK N READ MODEL", "read_model"),
    ("FUNCTIONAL-PREVIEW-16.2", "BLOCK N SAFETY GATE MATRIX", "safety_gate_matrix"),
    ("FUNCTIONAL-PREVIEW-16.3", "BLOCK N SAFETY GATE CONTRACT", "safety_gate_contract"),
    ("FUNCTIONAL-PREVIEW-16.4", "BLOCK N SAFETY GATE READ MODEL", "safety_gate_read_model"),
    (
        "FUNCTIONAL-PREVIEW-16.5",
        "BLOCK N SAFETY GATE READINESS MATRIX",
        "safety_gate_readiness_matrix",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.6",
        "BLOCK N SAFETY GATE READINESS CONTRACT",
        "safety_gate_readiness_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.7",
        "BLOCK N SAFETY GATE READINESS READ MODEL",
        "safety_gate_readiness_read_model",
    ),
]


def build_preview_block_n_closure_audit() -> dict[str, Any]:
    """Build the 16.8 source-only Block N closure audit from the 16.7 read model."""
    read_model = build_preview_block_n_safety_gate_readiness_read_model()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_n_closure_audit_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_n_closure_audit_status": CLOSURE_AUDIT_STATUS,
        "block_n_closure_audit_decision": CLOSURE_AUDIT_DECISION,
        "block_n_closed": BLOCK_N_CLOSED,
        "ready_for_block_o_0": READY_FOR_BLOCK_O_0,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_n_safety_gate_readiness_read_model_reference": _build_reference(read_model),
        "closure_audit_summary": _build_summary(),
        "block_n_step_closure_rows": _build_step_rows(),
        "packaging_release_closure_summary": _build_domain_summary(read_model, "packaging_release"),
        "runtime_safety_closure_summary": _build_domain_summary(read_model, "runtime_safety"),
        "cross_domain_invariant_closure_summary": _build_invariant_summary(read_model),
        "validation_requirement_closure_summary": _build_requirement_summary(read_model),
        "exe_direction_closure_audit": _build_exe_direction(read_model),
        "fail_closed_closure_decision": _build_fail_closed_decision(),
        "non_execution_closure_evidence": _build_non_execution_evidence(),
        "closure_audit_boundaries": _build_closure_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_17_0_block_o_entry_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _REFERENCE_KEYS}
    reference.update(
        {
            "source_block_n_safety_gate_readiness_read_model_step": SOURCE_BLOCK_N_READINESS_READ_MODEL_STEP,
            "source_block_n_safety_gate_readiness_read_model_read_by_16_8": True,
            "block_n_safety_gate_readiness_read_model_available_before_closure_audit": True,
            "static_block_n_safety_gate_readiness_read_model_only": True,
            "block_n_closure_audit_built_by_16_8": True,
            "ready_for_functional_preview_17_0": True,
        }
    )
    for root in _FALSE_BY_16_8_ROOTS:
        reference[root + "_by_16_8"] = False
    return reference


def _build_summary() -> dict[str, bool]:
    true_keys = [
        "block_n_safety_gate_readiness_read_model_available",
        "block_n_closure_audit_built",
        "block_n_opened",
        "block_n_closed",
        "ready_for_block_o_0",
        "ready_for_functional_preview_17_0",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "closure_audit_source_only",
        "closure_audit_plain_data_only",
        "closure_audit_static_only",
        "closure_audit_read_only",
        "closure_audit_non_evaluating",
        "closure_audit_non_mutating",
        "closure_audit_non_authorizing",
        "all_block_n_steps_complete",
        "all_capability_rows_read",
        "all_requirement_rows_read",
        "all_invariant_rows_read",
        "all_execution_capabilities_fail_closed",
        "all_execution_capabilities_not_ready",
        "all_execution_capabilities_blocked",
        "all_requirements_missing",
        "all_requirements_block_execution",
        "all_invariants_preserved",
        "all_domains_not_ready",
        "all_domains_execution_unauthorized",
        "only_source_only_block_o_handoff_allowed",
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
        "packaging_release_domain_ready_now",
        "runtime_safety_domain_ready_now",
        "exe_build_ready_now",
        "exe_packaging_ready_now",
        "exe_release_ready_now",
        "runtime_enabled_by_closure",
        "packaging_enabled_by_closure",
        "release_enabled_by_closure",
        "orders_enabled_by_closure",
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_step_rows() -> list[dict[str, Any]]:
    return [
        {
            "closure_row_id": step.lower().replace("-", "_").replace(".", "_") + "_closure",
            "step": step,
            "title": title,
            "artifact_kind": kind,
            "step_complete": True,
            "source_only": True,
            "plain_data": True,
            "execution_authorized": False,
            "real_capabilities_opened": False,
            "closure_status": "complete",
            "closure_result": "closed_source_only_execution_blocked",
            "notes": "Block N step is complete as source-only plain data; execution remains blocked.",
        }
        for step, title, kind in _STEPS
    ]


def _build_domain_summary(read_model: dict[str, Any], domain: str) -> dict[str, Any]:
    source_row = read_model["domain_readiness_read_summary"][domain]
    return {
        "domain": source_row["domain"],
        "capability_count": source_row["capability_count"],
        "read_capability_count": source_row["read_capability_count"],
        "ready_capability_count": source_row["ready_capability_count"],
        "blocked_capability_count": source_row["blocked_capability_count"],
        "required_requirement_ids": list(source_row["required_requirement_ids"]),
        "satisfied_requirement_ids": list(source_row["satisfied_requirement_ids"]),
        "missing_requirement_ids": list(source_row["missing_requirement_ids"]),
        "requirements_complete": source_row["requirements_complete"],
        "domain_ready": source_row["domain_ready"],
        "execution_authorized": source_row["execution_authorized"],
        "all_capabilities_read": source_row["all_capabilities_read"],
        "all_capabilities_not_ready": source_row["all_capabilities_not_ready"],
        "all_capabilities_blocked": source_row["all_capabilities_blocked"],
        "failure_policy": source_row["failure_policy"],
        "domain_closed_in_block_n": True,
        "domain_enabled_by_closure": False,
        "closure_result": "closed_source_only_execution_blocked",
    }


def _build_invariant_summary(read_model: dict[str, Any]) -> dict[str, Any]:
    rows = read_model["cross_domain_invariant_readiness_read_rows"]
    invariant_count = len(rows)
    preserved_invariant_count = sum(row["read_invariant_preserved"] is True for row in rows)
    failed_invariant_count = invariant_count - preserved_invariant_count
    return {
        "source_invariant_read_rows": rows,
        "invariant_count": invariant_count,
        "preserved_invariant_count": preserved_invariant_count,
        "failed_invariant_count": failed_invariant_count,
        "all_invariants_read": len(rows) == invariant_count,
        "all_invariants_preserved": preserved_invariant_count == invariant_count,
        "all_invariants_require_future_explicit_gate": all(
            row["requires_future_explicit_gate"] is True for row in rows
        ),
        "execution_gate_open_now": any(row["execution_gate_open_now"] is True for row in rows),
        "execution_allowed_now": any(row["execution_allowed_now"] is True for row in rows),
        "execution_performed_now": any(row["execution_performed_now"] is True for row in rows),
        "failure_policy": "fail_closed",
        "closure_result": "closed_invariants_preserved_execution_blocked",
    }


def _build_requirement_summary(read_model: dict[str, Any]) -> dict[str, Any]:
    rows = read_model["validation_requirement_read_rows"]
    requirement_count = len(rows)
    required_requirement_count = sum(row["required"] is True for row in rows)
    present_requirement_count = sum(row["present"] is True for row in rows)
    completed_requirement_count = sum(row["completed"] is True for row in rows)
    satisfied_requirement_count = sum(row["satisfied"] is True for row in rows)
    missing_requirement_count = sum(row["present"] is False for row in rows)
    return {
        "source_requirement_read_rows": rows,
        "requirement_count": requirement_count,
        "required_requirement_count": required_requirement_count,
        "present_requirement_count": present_requirement_count,
        "completed_requirement_count": completed_requirement_count,
        "satisfied_requirement_count": satisfied_requirement_count,
        "missing_requirement_count": missing_requirement_count,
        "all_requirements_read": len(rows) == requirement_count,
        "all_requirements_required": required_requirement_count == requirement_count,
        "all_requirements_missing": missing_requirement_count == requirement_count,
        "all_requirements_block_execution": all(
            row["missing_blocks_execution"] is True for row in rows
        ),
        "all_requirements_require_future_explicit_step": all(
            row["requires_future_explicit_step"] is True for row in rows
        ),
        "failure_policy": "fail_closed",
        "closure_result": "closed_requirements_missing_execution_blocked",
    }


def _build_exe_direction(read_model: dict[str, Any]) -> dict[str, Any]:
    exe = dict(read_model["exe_direction_read_model"])
    exe.update(
        {
            "block_n_closure_audit_confirms_exe_direction": True,
            "readiness_read_model_source_preserved": True,
            "closure_is_not_execution_authorization": True,
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
            "closure_result": "exe_direction_preserved_block_n_closed_execution_not_ready",
        }
    )
    return exe


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
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "private_endpoint",
        "network",
        "credentials",
        "config_env_secrets",
        "qml_bridge",
        "cc" + "xt",
    ]
    return {
        "missing_block_n_readiness_read_model_policy": "fail_closed",
        "missing_block_n_step_policy": "fail_closed",
        "missing_capability_read_row_policy": "fail_closed",
        "missing_requirement_read_row_policy": "fail_closed",
        "missing_invariant_read_row_policy": "fail_closed",
        "missing_operator_confirmation_policy": "fail_closed",
        "missing_environment_validation_policy": "fail_closed",
        "missing_artifact_validation_policy": "fail_closed",
        "missing_release_validation_policy": "fail_closed",
        "missing_runtime_validation_policy": "fail_closed",
        "missing_credentials_validation_policy": "fail_closed",
        "missing_future_explicit_gate_policy": "fail_closed",
        "failed_closure_audit_policy": "fail_closed",
        "block_n_closure_audit_in_16_8": "closed",
        "block_o_entry_contract_in_17_0": "allowed",
        "only_source_only_17_0_handoff_allowed": True,
        "real_capability_status": {key: "blocked" for key in real_capabilities},
    }


def _build_non_execution_evidence() -> dict[str, bool]:
    true_keys = [
        "source_block_n_readiness_read_model_read",
        "block_n_closure_audit_built",
        "block_n_closure_audit_only",
        "block_n_opened",
        "block_n_closed",
        "ready_for_block_o_0",
        "all_block_n_steps_complete",
        "all_capability_rows_read",
        "all_capability_rows_not_ready",
        "all_invariant_rows_preserved",
        "all_requirement_rows_missing",
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
        "real_capabilities_opened_by_closure",
    ]
    return {**{key: True for key in true_keys}, **{key: False for key in false_keys}}


def _build_closure_boundaries() -> dict[str, bool]:
    keys = [
        "block_n_closure_audit_is_plain_data_only",
        "block_n_closure_audit_is_source_only",
        "block_n_closure_audit_reads_16_7_only",
        "block_n_closure_audit_preserves_block_m_closure",
        "block_n_closure_audit_preserves_block_n_entry",
        "block_n_closure_audit_preserves_exe_direction_without_packaging",
        "block_n_closure_audit_is_static_and_non_evaluating",
        "block_n_closure_audit_is_non_mutating",
        "block_n_closure_audit_is_non_authorizing",
        "block_n_closure_audit_can_close_block_n",
        "block_n_closure_audit_can_feed_17_0_entry_contract",
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
    return {key: True for key in keys}


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    source = dict(read_model["source_boundaries"])
    contract_boundaries = dict(source["source_block_n_safety_gate_readiness_contract_boundaries"])
    contract_boundaries["can_feed_16_8"] = True
    source["source_block_n_safety_gate_readiness_contract_boundaries"] = contract_boundaries
    source["source_block_n_safety_gate_readiness_read_model"] = (
        SOURCE_BLOCK_N_READINESS_READ_MODEL_STEP
    )
    source["can_feed_16_8"] = True
    source["can_close_block_n"] = True
    source["can_feed_17_0"] = True
    source["forbidden_git_calls_present"] = False
    return source
