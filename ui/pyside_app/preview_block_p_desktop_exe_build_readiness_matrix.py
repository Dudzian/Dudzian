"""FUNCTIONAL-PREVIEW-18.5: fail-closed desktop EXE build-readiness matrix."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_packaging_read_model import (
    SCHEMA_VERSION as READ_MODEL_SCHEMA_VERSION,
    STATUS as READ_MODEL_STATUS,
    _handoff_integrity,
    _output_integrity,
    build_preview_block_p_desktop_exe_packaging_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_build_readiness_matrix.v1"
BLOCKED_STATUS: Final[str] = "blocked_for_desktop_exe_build_readiness_source_not_accepted"
STATUS: Final[str] = "blocked_desktop_exe_build_readiness_evidence_not_collected"
ROW_FIELDS: Final[list[str]] = [
    "readiness_id",
    "source_requirement_ids",
    "source_blocker_ids",
    "required_evidence_ids",
    "observed",
    "validated",
    "satisfied",
    "ready",
    "blocks_build",
    "blocks_packaging",
    "blocks_artifact_creation",
    "readiness_state",
    "readiness_result",
    "failure_policy",
]
READINESS_ROWS: Final[list[dict[str, Any]]] = [
    {
        "readiness_id": "final_desktop_entrypoint",
        "source_requirement_ids": ["desktop_application_entrypoint_inventory"],
        "source_blocker_ids": ["final_desktop_entrypoint_not_selected"],
        "required_evidence_ids": ["evidence_final_desktop_entrypoint_selection"],
    },
    {
        "readiness_id": "entrypoint_source_validation",
        "source_requirement_ids": ["desktop_application_entrypoint_inventory"],
        "source_blocker_ids": ["desktop_entrypoint_validation_not_performed"],
        "required_evidence_ids": ["evidence_desktop_entrypoint_validation"],
    },
    {
        "readiness_id": "windows_launch_smoke_prerequisite",
        "source_requirement_ids": ["desktop_application_entrypoint_inventory"],
        "source_blocker_ids": ["desktop_entrypoint_validation_not_performed"],
        "required_evidence_ids": ["evidence_desktop_entrypoint_validation"],
    },
    {
        "readiness_id": "qml_roots_assets",
        "source_requirement_ids": ["qml_asset_inventory"],
        "source_blocker_ids": ["qml_bundle_validation_not_performed"],
        "required_evidence_ids": ["evidence_qml_bundle_validation"],
    },
    {
        "readiness_id": "windows_shared_qml_import_path",
        "source_requirement_ids": ["qml_asset_inventory"],
        "source_blocker_ids": ["windows_shared_qml_import_path_unresolved"],
        "required_evidence_ids": ["evidence_windows_shared_qml_import_path"],
    },
    {
        "readiness_id": "qt_plugin_inventory",
        "source_requirement_ids": ["qt_plugin_inventory"],
        "source_blocker_ids": ["qt_plugin_inventory_missing"],
        "required_evidence_ids": ["evidence_qt_plugin_inventory"],
    },
    {
        "readiness_id": "ui_package_discovery",
        "source_requirement_ids": ["qml_asset_inventory"],
        "source_blocker_ids": ["ui_package_discovery_missing"],
        "required_evidence_ids": ["evidence_ui_package_discovery"],
    },
    {
        "readiness_id": "qml_package_data",
        "source_requirement_ids": ["qml_asset_inventory"],
        "source_blocker_ids": ["qml_package_data_missing"],
        "required_evidence_ids": ["evidence_qml_package_data"],
    },
    {
        "readiness_id": "windows_dependency_lock_resolution",
        "source_requirement_ids": ["python_dependency_inventory"],
        "source_blocker_ids": ["dependency_resolution_not_performed"],
        "required_evidence_ids": ["evidence_windows_dependency_resolution"],
    },
    {
        "readiness_id": "final_packaging_profile_alignment",
        "source_requirement_ids": ["packaging_profile_alignment"],
        "source_blocker_ids": ["final_desktop_packaging_profile_not_aligned"],
        "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
    },
    {
        "readiness_id": "windows_python_qt_toolchain_versions",
        "source_requirement_ids": ["windows_target_toolchain_confirmation"],
        "source_blocker_ids": ["windows_toolchain_not_confirmed"],
        "required_evidence_ids": ["evidence_windows_toolchain_confirmation"],
    },
    {
        "readiness_id": "secret_local_data_exclusion_application",
        "source_requirement_ids": ["secret_and_local_data_exclusion_policy"],
        "source_blocker_ids": ["secret_and_local_data_exclusion_policy_not_validated"],
        "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
    },
    {
        "readiness_id": "final_bundle_denied_scan",
        "source_requirement_ids": ["secret_and_local_data_exclusion_policy"],
        "source_blocker_ids": ["secret_and_local_data_exclusion_policy_not_validated"],
        "required_evidence_ids": ["evidence_artifact_exclusion_validation"],
    },
    {
        "readiness_id": "explicit_build_gate",
        "source_requirement_ids": ["future_explicit_build_execution_gate"],
        "source_blocker_ids": ["future_explicit_build_execution_gate_missing"],
        "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
    },
    {
        "readiness_id": "exact_build_command",
        "source_requirement_ids": ["future_explicit_build_execution_gate"],
        "source_blocker_ids": ["future_explicit_build_execution_gate_missing"],
        "required_evidence_ids": ["evidence_future_explicit_build_execution_gate"],
    },
    {
        "readiness_id": "artifact_output_name_version_policy",
        "source_requirement_ids": ["packaging_profile_alignment"],
        "source_blocker_ids": ["final_desktop_packaging_profile_not_aligned"],
        "required_evidence_ids": ["evidence_final_windows_profile_alignment"],
    },
    {
        "readiness_id": "post_build_launch_smoke",
        "source_requirement_ids": ["desktop_application_entrypoint_inventory"],
        "source_blocker_ids": ["desktop_entrypoint_validation_not_performed"],
        "required_evidence_ids": ["evidence_desktop_entrypoint_validation"],
    },
]


TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block",
    "step",
    "status",
    "source_18_4_accepted",
    "ready_for_build_execution",
    "readiness_rows",
    "capability_build_readiness_state",
    "boundaries",
    "integrity_valid",
]
CAPABILITY_FIELDS: Final[list[str]] = [
    "build",
    "packaging",
    "artifact_creation",
    "runtime",
    "orders",
]
BOUNDARY_FIELDS: Final[list[str]] = [
    "source_only",
    "plain_data",
    "build_performed",
    "filesystem_scanned",
    "network_opened",
    "runtime_started",
    "orders_enabled",
]


def _string_list(value: Any) -> bool:
    return (
        type(value) is list
        and bool(value)
        and len(value) == len(set(value))
        and all(type(x) is str and x for x in value)
    )


def _integrity_core(payload: Any) -> bool:
    if type(payload) is not dict or list(payload) != TOP_LEVEL_FIELDS:
        return False
    expected_types = {
        "schema_version": str,
        "block": str,
        "step": str,
        "status": str,
        "source_18_4_accepted": bool,
        "ready_for_build_execution": bool,
        "readiness_rows": list,
        "capability_build_readiness_state": dict,
        "boundaries": dict,
        "integrity_valid": bool,
    }
    if any(type(payload[key]) is not expected for key, expected in expected_types.items()):
        return False
    if (payload["schema_version"], payload["block"], payload["step"]) != (
        SCHEMA_VERSION,
        "P",
        "18.5",
    ):
        return False
    if (
        type(payload["source_18_4_accepted"]) is not bool
        or payload["ready_for_build_execution"] is not False
    ):
        return False
    caps, boundaries = payload["capability_build_readiness_state"], payload["boundaries"]
    if (
        type(caps) is not dict
        or list(caps) != CAPABILITY_FIELDS
        or any(type(caps[k]) is not str or caps[k] != "blocked" for k in CAPABILITY_FIELDS)
    ):
        return False
    if type(boundaries) is not dict or list(boundaries) != BOUNDARY_FIELDS:
        return False
    if (
        boundaries["source_only"] is not True
        or boundaries["plain_data"] is not True
        or any(boundaries[k] is not False for k in BOUNDARY_FIELDS[2:])
    ):
        return False
    rows = payload["readiness_rows"]
    accepted = payload["source_18_4_accepted"]
    if type(rows) is not list or payload["status"] != (STATUS if accepted else BLOCKED_STATUS):
        return False
    if not accepted:
        return rows == []
    if [row.get("readiness_id") if type(row) is dict else None for row in rows] != [
        row["readiness_id"] for row in READINESS_ROWS
    ]:
        return False
    for row, expected in zip(rows, READINESS_ROWS):
        if (
            type(row) is not dict
            or list(row) != ROW_FIELDS
            or any(
                type(row[k]) is not expected_type
                for k, expected_type in {
                    "readiness_id": str,
                    "observed": bool,
                    "validated": bool,
                    "satisfied": bool,
                    "ready": bool,
                    "blocks_build": bool,
                    "blocks_packaging": bool,
                    "blocks_artifact_creation": bool,
                    "readiness_state": str,
                    "readiness_result": str,
                    "failure_policy": str,
                }.items()
            )
            or any(row[k] != v for k, v in expected.items())
        ):
            return False
        if not all(
            _string_list(row[k])
            for k in ("source_requirement_ids", "source_blocker_ids", "required_evidence_ids")
        ):
            return False
        if not all(
            type(row[k]) is bool and row[k] is False
            for k in ("observed", "validated", "satisfied", "ready")
        ):
            return False
        if not all(
            type(row[k]) is bool and row[k] is True
            for k in ("blocks_build", "blocks_packaging", "blocks_artifact_creation")
        ):
            return False
        if (row["readiness_state"], row["readiness_result"], row["failure_policy"]) != (
            "required_not_observed",
            "future_explicit_evidence_required",
            "fail_closed",
        ):
            return False
    return True


def _integrity(payload: Any) -> bool:
    return _integrity_core(payload) and type(payload) is dict and payload["integrity_valid"] is True


def _source_accepted(source: Any) -> bool:
    return (
        type(source) is dict
        and source.get("schema_version") == READ_MODEL_SCHEMA_VERSION
        and source.get("block") == "P"
        and source.get("step") == "18.4"
        and source.get("status") == READ_MODEL_STATUS
        and source.get("ready_for_block_p_5") is True
        and source.get("packaging_read_model_artifact_complete") is True
        and _output_integrity(source)
        and _handoff_integrity(source)
    )


def build_preview_block_p_desktop_exe_build_readiness_matrix() -> dict[str, Any]:
    source = build_preview_block_p_desktop_exe_packaging_read_model()
    accepted = _source_accepted(source)
    rows = (
        [
            {
                **deepcopy(row),
                "observed": False,
                "validated": False,
                "satisfied": False,
                "ready": False,
                "blocks_build": True,
                "blocks_packaging": True,
                "blocks_artifact_creation": True,
                "readiness_state": "required_not_observed",
                "readiness_result": "future_explicit_evidence_required",
                "failure_policy": "fail_closed",
            }
            for row in READINESS_ROWS
        ]
        if accepted
        else []
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block": "P",
        "step": "18.5",
        "status": STATUS if accepted else BLOCKED_STATUS,
        "source_18_4_accepted": accepted,
        "ready_for_build_execution": False,
        "readiness_rows": rows,
        "capability_build_readiness_state": {
            "build": "blocked",
            "packaging": "blocked",
            "artifact_creation": "blocked",
            "runtime": "blocked",
            "orders": "blocked",
        },
        "boundaries": {
            "source_only": True,
            "plain_data": True,
            "build_performed": False,
            "filesystem_scanned": False,
            "network_opened": False,
            "runtime_started": False,
            "orders_enabled": False,
        },
        "integrity_valid": False,
    }
    payload["integrity_valid"] = _integrity_core(payload)
    return payload
