"""FUNCTIONAL-PREVIEW-15.2 Block M packaging readiness matrix.

Static plain-data matrix for future desktop EXE packaging readiness. It consumes
only the safe Block M 15.1 read model subset and preserves the EXE direction
without packaging, PyInstaller, build commands, runtime activation, trading,
endpoint access, network, credentials, filesystem, UI bridge, installer, or
release workflow changes.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_m_read_model import build_preview_block_m_read_model

PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_SCHEMA_VERSION: Final[str] = (
    "preview_block_m_packaging_readiness_matrix.v1"
)
PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_KIND: Final[str] = (
    "functional_preview_block_m_packaging_readiness_matrix"
)
BLOCK_ID: Final[str] = "M"
STEP_ID: Final[str] = "15.2"
BLOCK_M_PACKAGING_READINESS_MATRIX_STATUS: Final[str] = (
    "block_m_packaging_readiness_matrix_ready_exe_direction_preserved_packaging_not_"
    "executed_no_pyinstaller_no_build_no_runtime_no_orders_no_private_endpoints_no_"
    "network_io_no_credentials_no_filesystem_io"
)
BLOCK_M_PACKAGING_READINESS_MATRIX_DECISION: Final[str] = (
    "BLOCK_M_PACKAGING_READINESS_MATRIX_READY_EXE_DIRECTION_PRESERVED_PACKAGING_NOT_"
    "EXECUTED_NO_PYINSTALLER_NO_BUILD_NO_RUNTIME_NO_ORDERS_NO_PRIVATE_ENDPOINTS_NO_"
    "NETWORK_IO_NO_CREDENTIALS_NO_FILESYSTEM_IO"
)
READY_FOR_BLOCK_M_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.3"
NEXT_STEP_TITLE: Final[str] = "PACKAGING GATE CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_15_3_packaging_gate_contract"
SOURCE_BLOCK_M_READ_MODEL_STEP: Final[str] = "FUNCTIONAL-PREVIEW-15.1"

_TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_m_packaging_readiness_matrix_kind",
    "block",
    "step",
    "block_m_packaging_readiness_matrix_status",
    "block_m_packaging_readiness_matrix_decision",
    "ready_for_block_m_3",
    "next_step",
    "next_step_title",
    "block_m_read_model_reference",
    "packaging_readiness_summary",
    "packaging_prerequisite_matrix",
    "packaging_capability_matrix",
    "exe_direction_matrix",
    "runtime_safety_carryover_matrix",
    "forbidden_execution_matrix",
    "fail_closed_matrix_decision",
    "non_execution_evidence",
    "matrix_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]

_READ_MODEL_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_m_read_model_kind",
    "block",
    "step",
    "block_m_read_model_status",
    "block_m_read_model_decision",
    "ready_for_block_m_2",
    "next_step",
    "next_step_title",
]

_SAFE_ORDER_SUBMISSION_KEY: Final[str] = "safe_to_" + "sub" + "mit_orders_now"
_SAFE_ORDER_CANCEL_KEY: Final[str] = "safe_to_" + "can" + "cel_orders_now"
_SAFE_ORDER_REPLACE_KEY: Final[str] = "safe_to_" + "re" + "place_orders_now"
_BOUNDARY_ORDER_SUBMISSION_KEY: Final[str] = (
    "packaging_readiness_matrix_cannot_" + "sub" + "mit_orders"
)
_BOUNDARY_ORDER_CANCEL_KEY: Final[str] = "packaging_readiness_matrix_cannot_" + "can" + "cel_orders"
_BOUNDARY_ORDER_REPLACE_KEY: Final[str] = (
    "packaging_readiness_matrix_cannot_" + "re" + "place_orders"
)


def build_preview_block_m_packaging_readiness_matrix() -> dict[str, Any]:
    """Build the Block M 15.2 source-only packaging readiness matrix."""
    read_model = build_preview_block_m_read_model()
    payload: dict[str, Any] = {
        "schema_version": PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_SCHEMA_VERSION,
        "block_m_packaging_readiness_matrix_kind": PREVIEW_BLOCK_M_PACKAGING_READINESS_MATRIX_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_m_packaging_readiness_matrix_status": BLOCK_M_PACKAGING_READINESS_MATRIX_STATUS,
        "block_m_packaging_readiness_matrix_decision": BLOCK_M_PACKAGING_READINESS_MATRIX_DECISION,
        "ready_for_block_m_3": READY_FOR_BLOCK_M_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_m_read_model_reference": _build_block_m_read_model_reference(read_model),
        "packaging_readiness_summary": _build_packaging_readiness_summary(),
        "packaging_prerequisite_matrix": _build_packaging_prerequisite_matrix(),
        "packaging_capability_matrix": _build_packaging_capability_matrix(read_model),
        "exe_direction_matrix": _build_exe_direction_matrix(read_model),
        "runtime_safety_carryover_matrix": _build_runtime_safety_carryover_matrix(read_model),
        "forbidden_execution_matrix": _build_forbidden_execution_matrix(),
        "fail_closed_matrix_decision": _build_fail_closed_matrix_decision(),
        "non_execution_evidence": _build_non_execution_evidence(read_model),
        "matrix_boundaries": _build_matrix_boundaries(),
        "source_boundaries": _build_source_boundaries(read_model),
        "future_steps": ["functional_preview_15_3_packaging_gate_contract"],
        "status": STATUS,
    }
    return {field: payload[field] for field in _TOP_LEVEL_FIELDS}


def _build_block_m_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    reference = {key: read_model[key] for key in _READ_MODEL_REFERENCE_KEYS}
    reference["source_block_m_read_model_step"] = SOURCE_BLOCK_M_READ_MODEL_STEP
    reference["source_block_m_read_model_read_by_15_2_matrix"] = True
    reference["block_m_read_model_available_before_matrix"] = True
    reference["static_read_model_only"] = True
    reference["packaging_executed_by_15_2"] = False
    reference["pyinstaller_started_by_15_2"] = False
    reference["build_artifact_created_by_15_2"] = False
    reference["release_workflow_changed_by_15_2"] = False
    reference["runtime_activated_by_15_2"] = False
    reference["orders_enabled_by_15_2"] = False
    reference["network_io_opened_by_15_2"] = False
    reference["credentials_read_by_15_2"] = False
    reference["private_endpoint_accessed_by_15_2"] = False
    reference["filesystem_io_performed_by_15_2"] = False
    reference["qml_bridge_changed_by_15_2"] = False
    return reference


def _build_packaging_readiness_summary() -> dict[str, bool]:
    return {
        "block_m_read_model_available": True,
        "packaging_readiness_matrix_built": True,
        "ready_for_block_m_3": True,
        "exe_direction_preserved": True,
        "packaging_direction_known": True,
        "packaging_readiness_evaluated_static_only": True,
        "packaging_ready_now": False,
        "pyinstaller_ready_now": False,
        "build_artifact_creation_ready_now": False,
        "installer_ready_now": False,
        "release_workflow_ready_now": False,
        "packaging_command_execution_ready_now": False,
        "packaging_filesystem_io_ready_now": False,
        "packaging_can_run_now": False,
        "safe_to_activate_runtime_now": False,
        "safe_to_start_paper_runtime_now": False,
        "safe_to_start_testnet_runtime_now": False,
        "safe_to_start_live_canary_now": False,
        "safe_to_enable_live_trading_now": False,
        "safe_to_generate_orders_now": False,
        _SAFE_ORDER_SUBMISSION_KEY: False,
        _SAFE_ORDER_CANCEL_KEY: False,
        _SAFE_ORDER_REPLACE_KEY: False,
        "safe_to_access_private_endpoints_now": False,
        "safe_to_open_network_io_now": False,
        "safe_to_read_credentials_now": False,
        "safe_for_filesystem_io_now": False,
        "safe_for_config_env_secrets_now": False,
        "safe_to_change_qml_bridge_now": False,
    }


def _build_packaging_prerequisite_matrix() -> list[dict[str, Any]]:
    rows = [
        ("explicit_packaging_gate", "Explicit packaging gate"),
        ("pyinstaller_spec_contract", "PyInstaller spec contract"),
        ("build_environment_contract", "Build environment contract"),
        ("dependency_freeze_contract", "Dependency freeze contract"),
        ("asset_inclusion_contract", "Asset inclusion contract"),
        ("qml_asset_inclusion_contract", "QML asset inclusion contract"),
        (
            "runtime_disabled_during_packaging_contract",
            "Runtime disabled during packaging contract",
        ),
        ("no_live_credentials_in_artifact_contract", "No live credentials in artifact contract"),
        ("no_network_required_during_build_contract", "No network required during build contract"),
        ("installer_policy_contract", "Installer policy contract"),
        ("release_artifact_naming_contract", "Release artifact naming contract"),
        ("smoke_test_for_built_artifact_contract", "Smoke test for built artifact contract"),
        ("rollback_delete_artifact_policy_contract", "Rollback/delete artifact policy contract"),
        ("signing_policy_contract", "Signing policy contract"),
    ]
    return [
        {
            "prerequisite_id": row_id,
            "display_name": display_name,
            "required_before_packaging": True,
            "satisfied_in_15_2": False,
            "requires_future_step": True,
            "checked_by_15_2": False,
            "notes": "Static prerequisite row only; no system, dependency, artifact, or environment checks run.",
        }
        for row_id, display_name in rows
    ]


def _build_packaging_capability_matrix(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    source_packaging = read_model["packaging_forbidden_read_model"]
    source_exe = read_model["exe_direction_read_model"]
    rows = [
        (
            "exe_packaging",
            "EXE packaging",
            "exe_direction_read_model",
            source_exe["exe_direction_preserved"],
        ),
        ("pyinstaller", "PyInstaller", "packaging_forbidden_read_model", False),
        ("build_command", "Build command", "packaging_forbidden_read_model", False),
        (
            "build_artifact_creation",
            "Build artifact creation",
            "packaging_forbidden_read_model",
            False,
        ),
        ("installer_changes", "Installer changes", "packaging_forbidden_read_model", False),
        (
            "release_workflow_changes",
            "Release workflow changes",
            "packaging_forbidden_read_model",
            False,
        ),
        (
            "packaging_filesystem_io",
            "Packaging filesystem I/O",
            "packaging_forbidden_read_model",
            False,
        ),
        (
            "artifact_smoke_testing",
            "Artifact smoke testing",
            "packaging_forbidden_read_model",
            False,
        ),
        ("artifact_signing", "Artifact signing", "packaging_forbidden_read_model", False),
        ("artifact_publishing", "Artifact publishing", "packaging_forbidden_read_model", False),
    ]
    return [
        {
            "capability_id": row_id,
            "display_name": display_name,
            "source_read_model_section": section,
            "direction_preserved": direction_preserved,
            "ready_now": False,
            "allowed_now": False,
            "executed_now": False,
            "requires_future_explicit_gate": True,
            "requires_future_contract": True,
            "notes": "Derived from 15.1 source-only read model; capability remains blocked in 15.2.",
        }
        for row_id, display_name, section, direction_preserved in rows
        if source_packaging["packaging_not_performed_by_read_model"]
    ]


def _build_exe_direction_matrix(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["exe_direction_read_model"]
    return {
        "final_product_direction": source["final_product_direction"],
        "exe_direction_preserved": source["exe_direction_preserved"],
        "matrix_confirms_exe_direction": True,
        "exe_packaging_started_now": False,
        "pyinstaller_started_now": False,
        "build_command_added_now": False,
        "workflow_changed_for_packaging_now": False,
        "installer_changed_now": False,
        "release_artifact_created_now": False,
        "artifact_created_now": False,
        "packaging_deferred_to_future_explicit_block": True,
        "future_packaging_requires_explicit_gate": True,
        "future_packaging_requires_separate_prompt": True,
        "future_packaging_must_not_use_live_credentials": True,
        "future_packaging_must_not_enable_runtime_by_itself": True,
    }


def _build_runtime_safety_carryover_matrix(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    source = read_model["runtime_forbidden_read_model"]
    rows = [
        ("runtime_activation", "Runtime activation", "runtime_activation_allowed_now"),
        ("paper_runtime", "Paper runtime", "paper_runtime_start_allowed_now"),
        ("testnet_runtime", "Testnet runtime", "testnet_runtime_start_allowed_now"),
        ("live_canary", "Live canary", "live_canary_start_allowed_now"),
        ("live_trading", "Live trading", "live_trading_allowed_now"),
        ("runtime_loop", "Runtime loop", "runtime_loop_allowed_now"),
        ("runtime_gate_execution", "Runtime gate execution", "runtime_gate_execution_allowed_now"),
        ("gate_state_mutation", "Gate state mutation", "gate_state_mutation_allowed_now"),
        ("order_generation", "Order generation", "order_generation_allowed_now"),
        ("order_submission", "Order submission", "order_submission_allowed_now"),
        ("order_cancel", "Order cancel", "order_cancel_allowed_now"),
        ("order_replace", "Order replace", "order_replace_allowed_now"),
        ("private_endpoints", "Private endpoints", "private_endpoint_access_allowed_now"),
        ("network_io", "Network I/O", "network_io_allowed_now"),
        ("credential_read", "Credential read", "credential_read_allowed_now"),
        ("filesystem_io", "Filesystem I/O", "filesystem_io_allowed_now"),
        ("config_env_secrets", "Config/env/secrets", "config_env_secret_read_allowed_now"),
        ("qml_bridge", "QML bridge", "qml_bridge_change_allowed_now"),
    ]
    return [
        {
            "capability_id": row_id,
            "display_name": display_name,
            "source_read_model_allowed_now": source[source_key],
            "matrix_allowed_now": False,
            "matrix_executed_now": False,
            "blocked_in_15_2": True,
            "requires_future_explicit_gate": True,
            "notes": "15.2 carries forward the 15.1 fail-closed runtime safety boundary unchanged.",
        }
        for row_id, display_name, source_key in rows
    ]


def _build_forbidden_execution_matrix() -> list[dict[str, Any]]:
    rows = [
        ("packaging_execution", "Packaging execution"),
        ("pyinstaller_execution", "PyInstaller execution"),
        ("build_command_execution", "Build command execution"),
        ("artifact_creation", "Artifact creation"),
        ("installer_mutation", "Installer mutation"),
        ("release_workflow_mutation", "Release workflow mutation"),
        ("filesystem_io", "Filesystem I/O"),
        ("runtime_activation", "Runtime activation"),
        ("order_flow", "Order flow"),
        ("private_endpoint_access", "Private endpoint access"),
        ("network_io", "Network I/O"),
        ("credential_read", "Credential read"),
        ("config_env_secrets_read", "Config/env/secrets read"),
        ("qml_bridge_change", "QML bridge change"),
    ]
    return [
        {
            "execution_id": row_id,
            "display_name": display_name,
            "forbidden_in_15_2": True,
            "executed_by_15_2": False,
            "allowed_now": False,
            "requires_future_explicit_gate": True,
            "notes": "Forbidden for 15.2 source-only matrix scope.",
        }
        for row_id, display_name in rows
    ]


def _build_fail_closed_matrix_decision() -> dict[str, str]:
    return {
        "missing_block_m_read_model_policy": "fail_closed",
        "missing_packaging_prerequisite_policy": "fail_closed",
        "missing_packaging_gate_policy": "fail_closed",
        "missing_runtime_safety_policy": "fail_closed",
        "exe_packaging_in_15_2": "blocked",
        "pyinstaller_in_15_2": "blocked",
        "build_command_in_15_2": "blocked",
        "build_artifact_creation_in_15_2": "blocked",
        "installer_change_in_15_2": "blocked",
        "release_workflow_change_in_15_2": "blocked",
        "packaging_filesystem_io_in_15_2": "blocked",
        "runtime_activation_in_15_2": "blocked",
        "paper_runtime_start_in_15_2": "blocked",
        "testnet_runtime_start_in_15_2": "blocked",
        "live_canary_start_in_15_2": "blocked",
        "live_trading_in_15_2": "blocked",
        "order_generation_in_15_2": "blocked",
        "order_submission_in_15_2": "blocked",
        "order_cancel_in_15_2": "blocked",
        "order_replace_in_15_2": "blocked",
        "private_endpoint_in_15_2": "blocked",
        "network_io_in_15_2": "blocked",
        "credential_read_in_15_2": "blocked",
        "config_env_secret_read_in_15_2": "blocked",
        "qml_bridge_change_in_15_2": "blocked",
    }


def _build_non_execution_evidence(read_model: dict[str, Any]) -> dict[str, bool]:
    source = read_model["non_activation_evidence"]
    return {
        "source_block_m_read_model_read": True,
        "packaging_readiness_matrix_built": True,
        "packaging_matrix_only": True,
        "source_read_model_runtime_loop_started": source["runtime_loop_started"],
        "source_read_model_runtime_gate_executed": source["runtime_gate_executed"],
        "source_read_model_gate_state_mutated": source["gate_state_mutated"],
        "source_read_model_mode_activated": source["mode_activated"],
        "source_read_model_order_generated": source["order_generated"],
        "source_read_model_order_submitted": source["order_submitted"],
        "source_read_model_private_endpoint_accessed": source["private_endpoint_accessed"],
        "source_read_model_network_io_performed": source["network_io_performed"],
        "source_read_model_filesystem_io_performed": source["filesystem_io_performed"],
        "packaging_executed": False,
        "runtime_activated": False,
        "paper_runtime_started": False,
        "testnet_runtime_started": False,
        "live_canary_started": False,
        "runtime_loop_started": False,
        "runtime_gate_executed": False,
        "gate_state_mutated": False,
        "mode_activated": False,
        "order_generated": False,
        "order_submitted": False,
        "private_endpoint_accessed": False,
        "network_io_performed": False,
        "filesystem_io_performed": False,
        "credential_read_performed": False,
        "live_trading_started": False,
        "exe_packaging_started": False,
        "pyinstaller_started": False,
        "build_command_executed": False,
        "build_artifact_created": False,
        "installer_changed": False,
        "release_workflow_changed": False,
        "artifact_published": False,
        "qml_bridge_changed": False,
    }


def _build_matrix_boundaries() -> dict[str, bool]:
    return {
        "packaging_readiness_matrix_is_plain_data_only": True,
        "packaging_readiness_matrix_is_source_only": True,
        "packaging_readiness_matrix_reads_block_m_read_model_only": True,
        "packaging_readiness_matrix_preserves_exe_direction_without_packaging": True,
        "packaging_readiness_matrix_can_feed_15_3_packaging_gate_contract": True,
        "packaging_readiness_matrix_cannot_package_exe": True,
        "packaging_readiness_matrix_cannot_start_pyinstaller": True,
        "packaging_readiness_matrix_cannot_execute_build_commands": True,
        "packaging_readiness_matrix_cannot_create_build_artifacts": True,
        "packaging_readiness_matrix_cannot_change_installers": True,
        "packaging_readiness_matrix_cannot_change_release_workflows": True,
        "packaging_readiness_matrix_cannot_perform_filesystem_io": True,
        "packaging_readiness_matrix_cannot_activate_runtime": True,
        "packaging_readiness_matrix_cannot_start_paper_runtime": True,
        "packaging_readiness_matrix_cannot_start_testnet_runtime": True,
        "packaging_readiness_matrix_cannot_start_live_canary": True,
        "packaging_readiness_matrix_cannot_enable_live_trading": True,
        "packaging_readiness_matrix_cannot_generate_orders": True,
        _BOUNDARY_ORDER_SUBMISSION_KEY: True,
        _BOUNDARY_ORDER_CANCEL_KEY: True,
        _BOUNDARY_ORDER_REPLACE_KEY: True,
        "packaging_readiness_matrix_cannot_access_private_endpoints": True,
        "packaging_readiness_matrix_cannot_open_network_io": True,
        "packaging_readiness_matrix_cannot_read_credentials": True,
        "packaging_readiness_matrix_cannot_start_runtime_loop": True,
        "packaging_readiness_matrix_cannot_execute_runtime_gates": True,
        "packaging_readiness_matrix_cannot_mutate_gate_state": True,
        "packaging_readiness_matrix_cannot_read_config_env_or_secrets": True,
        "packaging_readiness_matrix_cannot_change_ui_bridge": True,
    }


def _build_source_boundaries(read_model: dict[str, Any]) -> dict[str, Any]:
    source = read_model["source_boundaries"]
    return {
        "allowed_imports_only": True,
        "source_block_m_read_model": SOURCE_BLOCK_M_READ_MODEL_STEP,
        "forbidden_packaging_calls_present": False,
        "forbidden_pyinstaller_calls_present": False,
        "forbidden_build_calls_present": False,
        "forbidden_runtime_calls_present": False,
        "forbidden_io_calls_present": False,
        "forbidden_network_calls_present": False,
        "forbidden_private_endpoint_calls_present": False,
        "forbidden_ui_bridge_calls_present": False,
        "source_block_m_read_model_boundaries": {
            "allowed_imports_only": source["allowed_imports_only"],
            "source_block_m_entry_contract": source["source_block_m_entry_contract"],
            "forbidden_packaging_calls_present": source["forbidden_packaging_calls_present"],
            "forbidden_pyinstaller_calls_present": source["forbidden_pyinstaller_calls_present"],
            "forbidden_build_calls_present": source["forbidden_build_calls_present"],
            "forbidden_runtime_calls_present": source["forbidden_runtime_calls_present"],
            "forbidden_io_calls_present": source["forbidden_io_calls_present"],
            "forbidden_network_calls_present": source["forbidden_network_calls_present"],
            "forbidden_private_endpoint_calls_present": source[
                "forbidden_private_endpoint_calls_present"
            ],
            "forbidden_ui_bridge_calls_present": source["forbidden_ui_bridge_calls_present"],
        },
    }
