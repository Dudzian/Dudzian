"""FUNCTIONAL-PREVIEW-19.1 Block Q Windows build environment inventory."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_q_windows_desktop_build_execution_entry_contract import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_0,
    build_preview_block_q_windows_desktop_build_execution_entry_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_build_environment_inventory.v1"
KIND: Final[str] = "functional_preview_block_q_windows_build_environment_inventory"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.1"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-19.2"
NEXT_STEP_TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT MATRIX"
STATUS: Final[str] = (
    "source_19_0_accepted_11_row_source_only_environment_inventory_defined_actual_environment_"
    "not_inspected_source_only_handoff_to_19_2_no_build_no_packaging_no_artifact_no_runtime_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_1_source_19_0_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()

SOURCE_19_0_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_desktop_build_execution_entry_contract_kind",
    "block",
    "step",
    "block_q_windows_desktop_build_execution_entry_contract_status",
    "source_18_8_accepted",
    "block_q_windows_desktop_build_execution_entry_contract_decision",
    "entry_contract_artifact_complete",
    "ready_for_block_q_1",
    "next_step",
    "next_step_title",
    "block_p_closure_audit_reference",
    "source_closure_preservation",
    "block_q_scope_definition",
    "windows_build_environment_requirements",
    "build_execution_evidence_requirements",
    "build_execution_authorization_state",
    "non_execution_entry_evidence",
    "entry_contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
SOURCE_19_0_IDENTITY_VALUES: Final[tuple[tuple[str, str], ...]] = (
    ("schema_version", SOURCE_SCHEMA_VERSION),
    ("kind", SOURCE_KIND),
    ("block", "Q"),
    ("step", "19.0"),
    ("status", SOURCE_STATUS),
)
SOURCE_19_0_REQUIRED_TRUE_FIELDS: Final[tuple[str, ...]] = (
    "source_18_8_accepted",
    "entry_contract_artifact_complete",
    "ready_for_block_q_1",
    "integrity_valid",
)
ENVIRONMENT_REQUIREMENT_TRUE_FIELDS: Final[tuple[str, ...]] = (
    "windows_host_required",
    "supported_windows_version_must_be_confirmed",
    "exact_python_version_must_be_confirmed",
    "exact_pyside_version_must_be_confirmed",
    "exact_packaging_tool_and_version_must_be_selected",
    "desktop_entrypoint_must_be_confirmed",
    "qml_assets_must_be_confirmed",
    "qt_plugins_must_be_confirmed",
    "dependency_lock_must_be_resolved",
    "secret_and_local_data_exclusion_must_be_confirmed",
    "output_name_and_version_policy_must_be_confirmed",
)
ENVIRONMENT_REQUIREMENT_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "environment_inventory_performed",
    "dependency_resolution_performed",
    "pyside_import_performed",
    "qml_load_performed",
    "qt_plugin_discovery_performed",
)
SOURCE_EXEC_AUTH_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "dependency_resolution_authorized",
    "build_command_creation_authorized",
    "build_command_execution_authorized",
    "packaging_authorized",
    "artifact_creation_authorized",
    "artifact_scan_authorized",
    "artifact_signing_authorized",
    "installer_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
    "authorization_granted_by_19_0",
)
SOURCE_BOUNDARY_TRUE_FIELDS: Final[tuple[str, ...]] = (
    "reads_18_8_only",
    "source_only",
    "plain_data",
    "static_entry_contract",
    "can_feed_only_19_1_windows_build_environment_inventory",
)
SOURCE_BOUNDARY_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "repo_rescan",
    "filesystem_scan",
    "environment_scan",
    "windows_build",
    "packaging",
    "artifact_creation",
    "artifact_scan",
    "artifact_signing",
    "installer_creation",
    "release",
    "runtime",
    "network",
    "credentials",
    "orders",
)
ENVIRONMENT_INVENTORY_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
    ("windows_host", "windows_host_required", "platform"),
    ("supported_windows_version", "supported_windows_version_must_be_confirmed", "platform"),
    ("python_version", "exact_python_version_must_be_confirmed", "runtime"),
    ("pyside_version", "exact_pyside_version_must_be_confirmed", "framework"),
    (
        "packaging_tool_and_version",
        "exact_packaging_tool_and_version_must_be_selected",
        "packaging",
    ),
    ("desktop_entrypoint", "desktop_entrypoint_must_be_confirmed", "application"),
    ("qml_assets", "qml_assets_must_be_confirmed", "assets"),
    ("qt_plugins", "qt_plugins_must_be_confirmed", "plugins"),
    ("dependency_lock", "dependency_lock_must_be_resolved", "dependencies"),
    (
        "secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_must_be_confirmed",
        "security",
    ),
    (
        "output_name_and_version_policy",
        "output_name_and_version_policy_must_be_confirmed",
        "output",
    ),
)
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_inventory_kind",
    "block",
    "step",
    "block_q_windows_build_environment_inventory_status",
    "source_19_0_accepted",
    "block_q_windows_build_environment_inventory_decision",
    "environment_inventory_artifact_complete",
    "environment_observation_complete",
    "ready_for_block_q_2",
    "next_step",
    "next_step_title",
    "block_q_19_0_entry_contract_reference",
    "source_entry_contract_preservation",
    "environment_inventory_scope",
    "environment_inventory_rows",
    "environment_inventory_summary",
    "build_execution_authorization_state",
    "non_execution_inventory_evidence",
    "inventory_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
BLOCKED_FIELDS: Final[tuple[str, ...]] = TOP_LEVEL_FIELDS
AUTHORIZATION_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "environment_scan_authorized",
    "windows_host_inspection_authorized",
    "dependency_resolution_authorized",
    "pyside_import_authorized",
    "qml_load_authorized",
    "qt_plugin_discovery_authorized",
    "build_command_creation_authorized",
    "build_command_execution_authorized",
    "packaging_authorized",
    "artifact_creation_authorized",
    "artifact_scan_authorized",
    "artifact_signing_authorized",
    "installer_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
    "authorization_granted_by_19_1",
)


def _exact_str_keyed_dict(value: Any) -> bool:
    try:
        return type(value) is dict and all(type(key) is str for key in value)
    except Exception:
        return False


def _exact_plain(value: Any, expected: Any) -> bool:
    try:
        pending = [(value, expected)]
        visited: set[tuple[int, int]] = set()
        while pending:
            actual, trusted = pending.pop()
            if type(actual) is not type(trusted):
                return False
            if type(trusted) is dict:
                pair = (id(actual), id(trusted))
                if pair in visited:
                    continue
                visited.add(pair)
                keys = list(actual)
                if not all(type(k) is str for k in keys) or keys != list(trusted):
                    return False
                for key in reversed(list(trusted)):
                    pending.append((actual[key], trusted[key]))
            elif type(trusted) is list:
                pair = (id(actual), id(trusted))
                if pair in visited:
                    continue
                visited.add(pair)
                if len(actual) != len(trusted):
                    return False
                pending.extend(zip(reversed(actual), reversed(trusted)))
            elif type(trusted) not in (str, bool, int, type(None)) or actual != trusted:
                return False
        return True
    except Exception:
        return False


def _source_accepted(source: Any) -> bool:
    try:
        if type(source) is not dict or not all(type(k) is str for k in source):
            return False
        if tuple(source) != SOURCE_19_0_TOP_LEVEL_FIELDS:
            return False
        if not (
            type(source["schema_version"]) is str
            and source["schema_version"] == SOURCE_SCHEMA_VERSION
            and type(source["block_q_windows_desktop_build_execution_entry_contract_kind"]) is str
            and source["block_q_windows_desktop_build_execution_entry_contract_kind"] == SOURCE_KIND
            and type(source["block"]) is str
            and source["block"] == "Q"
            and type(source["step"]) is str
            and source["step"] == "19.0"
            and type(source["status"]) is str
            and source["status"] == SOURCE_STATUS
            and type(source["block_q_windows_desktop_build_execution_entry_contract_status"]) is str
            and source["block_q_windows_desktop_build_execution_entry_contract_status"]
            == SOURCE_STATUS
            and source["source_18_8_accepted"] is True
            and source["entry_contract_artifact_complete"] is True
            and source["ready_for_block_q_1"] is True
            and type(source["next_step"]) is str
            and source["next_step"] == "FUNCTIONAL-PREVIEW-19.1"
            and type(source["next_step_title"]) is str
            and source["next_step_title"] == "WINDOWS BUILD ENVIRONMENT INVENTORY"
            and source["integrity_valid"] is True
        ):
            return False
        requirements = source["windows_build_environment_requirements"]
        auth = source["build_execution_authorization_state"]
        boundaries = source["entry_contract_boundaries"]
        future_steps = source["future_steps"]
        if not (
            _exact_str_keyed_dict(requirements)
            and _exact_str_keyed_dict(auth)
            and _exact_str_keyed_dict(boundaries)
            and type(future_steps) is list
        ):
            return False
        for key in ENVIRONMENT_REQUIREMENT_TRUE_FIELDS:
            if type(requirements.get(key)) is not bool or requirements[key] is not True:
                return False
        for key in ENVIRONMENT_REQUIREMENT_FALSE_FIELDS:
            if type(requirements.get(key)) is not bool or requirements[key] is not False:
                return False
        if auth.get("environment_inventory_authorized") is not True:
            return False
        if auth.get("only_source_only_19_1_handoff_allowed") is not True:
            return False
        for key in SOURCE_EXEC_AUTH_FALSE_FIELDS:
            if type(auth.get(key)) is not bool or auth[key] is not False:
                return False
        for key in SOURCE_BOUNDARY_TRUE_FIELDS:
            if type(boundaries.get(key)) is not bool or boundaries[key] is not True:
                return False
        for key in SOURCE_BOUNDARY_FALSE_FIELDS:
            if type(boundaries.get(key)) is not bool or boundaries[key] is not False:
                return False
        if len(future_steps) != 1 or not _exact_str_keyed_dict(future_steps[0]):
            return False
        expected_future = {
            "next_step": "FUNCTIONAL-PREVIEW-19.1",
            "next_step_title": "WINDOWS BUILD ENVIRONMENT INVENTORY",
            "source_only": True,
            "physical_build_performed": False,
        }
        if not _exact_plain(future_steps[0], expected_future):
            return False
        return _integrity_19_0(source) is True
    except Exception:
        return False


def _environment_inventory_rows() -> list[dict[str, Any]]:
    return [
        {
            "inventory_id": inventory_id,
            "requirement_field": requirement_field,
            "category": category,
            "required": True,
            "declared_by_19_0": True,
            "collection_status": "not_collected",
            "validation_status": "not_validated",
            "resolution_status": "blocked",
            "observed_value": "",
            "source_only_definition": True,
        }
        for inventory_id, requirement_field, category in ENVIRONMENT_INVENTORY_SPECS
    ]


def _inventory_summary(full: bool) -> dict[str, Any]:
    count = len(ENVIRONMENT_INVENTORY_SPECS) if full else 0
    return {
        "inventory_row_count": count,
        "required_count": count,
        "collected_count": 0,
        "validated_count": 0,
        "resolved_count": 0,
        "blocked_count": count,
        "environment_observation_complete": False,
        "inventory_definition_complete": full,
    }


def _authorization_state(nominal: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "environment_matrix_definition_authorized": nominal,
        "only_source_only_19_2_handoff_allowed": nominal,
    }
    payload.update({field: False for field in AUTHORIZATION_FALSE_FIELDS})
    return payload


def _trusted_source_stub() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_q_windows_desktop_build_execution_entry_contract_kind": SOURCE_KIND,
        "block": "Q",
        "step": "19.0",
        "block_q_windows_desktop_build_execution_entry_contract_status": SOURCE_STATUS,
        "source_18_8_accepted": True,
        "entry_contract_artifact_complete": True,
        "ready_for_block_q_1": True,
        "status": SOURCE_STATUS,
        "integrity_valid": True,
    }


def _non_execution_evidence(nominal: bool) -> dict[str, Any]:
    return {
        "source_read": nominal,
        "environment_inventory_definition_built": nominal,
        "repo_scan_performed": False,
        "filesystem_scan_performed": False,
        "environment_scan_performed": False,
        "windows_host_inspected": False,
        "windows_version_collected": False,
        "python_version_collected": False,
        "pyside_version_collected": False,
        "packaging_tool_selected": False,
        "dependency_resolution_performed": False,
        "pyside_import_performed": False,
        "qml_load_performed": False,
        "qt_plugin_discovery_performed": False,
        "build_command_created": False,
        "build_command_executed": False,
        "packaging_performed": False,
        "artifact_created": False,
        "artifact_scanned": False,
        "artifact_signed": False,
        "installer_created": False,
        "release_performed": False,
        "runtime_started": False,
        "network_opened": False,
        "credentials_read": False,
        "orders_enabled": False,
    }


def _inventory_boundaries(nominal: bool) -> dict[str, Any]:
    return {
        "reads_19_0_only": nominal,
        "source_only": nominal,
        "plain_data": nominal,
        "static_inventory": nominal,
        "can_feed_only_19_2_windows_build_environment_matrix": nominal,
        "repo_scan": False,
        "filesystem_scan": False,
        "environment_scan": False,
        "windows_host_inspection": False,
        "dependency_resolution": False,
        "pyside_import": False,
        "qml_load": False,
        "qt_plugin_discovery": False,
        "windows_build": False,
        "packaging": False,
        "artifact_creation": False,
        "artifact_scan": False,
        "artifact_signing": False,
        "installer_creation": False,
        "release": False,
        "runtime": False,
        "network": False,
        "credentials": False,
        "orders": False,
    }


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_inventory_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_inventory_status": STATUS,
        "source_19_0_accepted": True,
        "block_q_windows_build_environment_inventory_decision": DECISION,
        "environment_inventory_artifact_complete": True,
        "environment_observation_complete": False,
        "ready_for_block_q_2": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_q_19_0_entry_contract_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_q_windows_desktop_build_execution_entry_contract_kind"],
            "block": "Q",
            "step": "19.0",
            "status": source["status"],
            "source_18_8_accepted": True,
            "entry_contract_artifact_complete": True,
            "ready_for_block_q_1": True,
            "integrity_valid": True,
            "source_top_level_fields": list(SOURCE_19_0_TOP_LEVEL_FIELDS),
        },
        "source_entry_contract_preservation": {
            "preserves_19_0_payload": True,
            "preserves_19_0_environment_requirements": True,
            "preserves_19_0_zero_execution_authorizations": True,
            "preserves_19_0_non_execution_state": True,
            "preserves_source_only_handoff": True,
            "source_entry_contract_modified": False,
            "source_entry_contract_reinterpreted": False,
        },
        "environment_inventory_scope": {
            "inventory_defined": True,
            "source_only_inventory": True,
            "inventory_row_count": 11,
            "actual_windows_host_inspected": False,
            "actual_windows_version_collected": False,
            "actual_python_version_collected": False,
            "actual_pyside_version_collected": False,
            "actual_packaging_tool_selected": False,
            "actual_entrypoint_validated": False,
            "actual_qml_assets_validated": False,
            "actual_qt_plugins_discovered": False,
            "actual_dependencies_resolved": False,
            "actual_secret_exclusion_validated": False,
            "actual_output_policy_confirmed": False,
        },
        "environment_inventory_rows": _environment_inventory_rows(),
        "environment_inventory_summary": _inventory_summary(True),
        "build_execution_authorization_state": _authorization_state(True),
        "non_execution_inventory_evidence": _non_execution_evidence(True),
        "inventory_boundaries": _inventory_boundaries(True),
        "source_boundaries": {
            "source_step": "19.0",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "block_p_builders_read": False,
            "earlier_block_q_builders_read": False,
        },
        "future_steps": [
            {
                "next_step": NEXT_STEP,
                "next_step_title": NEXT_STEP_TITLE,
                "source_only": True,
                "environment_scan_performed": False,
                "physical_build_performed": False,
            }
        ],
        "status": STATUS,
        "integrity_valid": True,
    }
    return {field: payload[field] for field in TOP_LEVEL_FIELDS}


def _blocked() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_inventory_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_inventory_status": BLOCKED_STATUS,
        "source_19_0_accepted": False,
        "block_q_windows_build_environment_inventory_decision": BLOCKED_DECISION,
        "environment_inventory_artifact_complete": False,
        "environment_observation_complete": False,
        "ready_for_block_q_2": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_0_entry_contract_reference": {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.0",
            "status": "",
            "source_18_8_accepted": False,
            "entry_contract_artifact_complete": False,
            "ready_for_block_q_1": False,
            "integrity_valid": False,
            "source_top_level_fields": [],
        },
        "source_entry_contract_preservation": {
            "preserves_19_0_payload": False,
            "preserves_19_0_environment_requirements": False,
            "preserves_19_0_zero_execution_authorizations": False,
            "preserves_19_0_non_execution_state": False,
            "preserves_source_only_handoff": False,
            "source_entry_contract_modified": False,
            "source_entry_contract_reinterpreted": False,
        },
        "environment_inventory_scope": {
            "inventory_defined": False,
            "source_only_inventory": False,
            "inventory_row_count": 0,
            "actual_windows_host_inspected": False,
            "actual_windows_version_collected": False,
            "actual_python_version_collected": False,
            "actual_pyside_version_collected": False,
            "actual_packaging_tool_selected": False,
            "actual_entrypoint_validated": False,
            "actual_qml_assets_validated": False,
            "actual_qt_plugins_discovered": False,
            "actual_dependencies_resolved": False,
            "actual_secret_exclusion_validated": False,
            "actual_output_policy_confirmed": False,
        },
        "environment_inventory_rows": [],
        "environment_inventory_summary": _inventory_summary(False),
        "build_execution_authorization_state": _authorization_state(False),
        "non_execution_inventory_evidence": _non_execution_evidence(False),
        "inventory_boundaries": _inventory_boundaries(False),
        "source_boundaries": {
            "source_step": "19.0",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "block_p_builders_read": False,
            "earlier_block_q_builders_read": False,
        },
        "future_steps": [],
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }
    return {field: payload[field] for field in BLOCKED_FIELDS}


def _canonical_nominal() -> dict[str, Any]:
    return _nominal(_trusted_source_stub())


def _canonical_blocked() -> dict[str, Any]:
    return _blocked()


def _integrity(payload: Any) -> bool:
    try:
        return _exact_plain(payload, _canonical_nominal()) or _exact_plain(
            payload, _canonical_blocked()
        )
    except Exception:
        return False


def build_preview_block_q_windows_build_environment_inventory() -> dict[str, Any]:
    try:
        source = build_preview_block_q_windows_desktop_build_execution_entry_contract()
    except Exception:
        return _blocked()
    try:
        return _nominal(source) if _source_accepted(source) else _blocked()
    except Exception:
        return _blocked()
