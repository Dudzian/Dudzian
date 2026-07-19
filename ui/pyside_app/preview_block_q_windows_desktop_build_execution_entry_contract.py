"""FUNCTIONAL-PREVIEW-19.0 Block Q Windows build execution entry contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_p_closure_audit import (
    CLOSURE_AUDIT_STATUS as SOURCE_CLOSURE_AUDIT_STATUS,
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_18_8,
    build_preview_block_p_closure_audit,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_desktop_build_execution_entry_contract.v1"
KIND: Final[str] = "functional_preview_block_q_windows_desktop_build_execution_entry_contract"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.0"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-19.1"
NEXT_STEP_TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT INVENTORY"
STATUS: Final[str] = (
    "source_18_8_accepted_block_q_entry_contract_defined_source_only_handoff_to_19_1_"
    "no_build_no_packaging_no_artifact_no_release_no_runtime_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_0_source_18_8_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()

SOURCE_18_8_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_p_closure_audit_kind",
    "block",
    "step",
    "block_p_closure_audit_status",
    "block_p_closure_audit_decision",
    "block_p_closure_audit_ready",
    "source_18_7_accepted",
    "block_p_source_only_design_closed",
    "source_18_7_reference",
    "source_acceptance",
    "stage_audit_rows",
    "closure_summary",
    "closure_findings",
    "preservation_audit",
    "readiness_audit",
    "capability_audit",
    "authorization_audit",
    "non_execution_audit",
    "closure_decision",
    "closure_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
STAGE_STEPS: Final[tuple[str, ...]] = (
    "18.0",
    "18.1",
    "18.2",
    "18.3",
    "18.4",
    "18.5",
    "18.6",
    "18.7",
)
AUTHORIZATION_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
    "authorization_granted_by_18_8",
)
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
BLOCKED_FIELDS: Final[tuple[str, ...]] = (
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
    "windows_build_environment_requirements",
    "build_execution_evidence_requirements",
    "build_execution_authorization_state",
    "non_execution_entry_evidence",
    "entry_contract_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
EVIDENCE_IDS: Final[tuple[str, ...]] = (
    "evidence_windows_host",
    "evidence_windows_version",
    "evidence_python_version",
    "evidence_pyside_version",
    "evidence_packaging_tool_selection",
    "evidence_desktop_entrypoint_validation",
    "evidence_qml_bundle_validation",
    "evidence_qt_plugin_inventory",
    "evidence_dependency_resolution",
    "evidence_secret_exclusion_validation",
    "evidence_build_command_approval",
    "evidence_build_output",
    "evidence_artifact_launch_smoke",
)
EXEC_AUTH_FIELDS: Final[tuple[str, ...]] = (
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
SOURCE_18_8_IDENTITY_VALUES: Final[tuple[tuple[str, str], ...]] = (
    ("schema_version", SOURCE_SCHEMA_VERSION),
    ("block_p_closure_audit_kind", SOURCE_KIND),
    ("status", SOURCE_STATUS),
)


def _trusted_source_stub() -> dict[str, Any]:
    return dict(SOURCE_18_8_IDENTITY_VALUES)


def _canonical_nominal() -> dict[str, Any]:
    return _nominal(_trusted_source_stub())


def _canonical_blocked() -> dict[str, Any]:
    return _blocked()


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


def _exact_str_keyed_dict(value: Any) -> bool:
    try:
        return type(value) is dict and all(type(key) is str for key in value)
    except Exception:
        return False


def _source_accepted(source: Any) -> bool:
    try:
        if type(source) is not dict or not all(type(k) is str for k in source):
            return False
        if tuple(source) != SOURCE_18_8_TOP_LEVEL_FIELDS:
            return False
        if not (
            type(source["schema_version"]) is str
            and source["schema_version"] == SOURCE_SCHEMA_VERSION
            and type(source["block_p_closure_audit_kind"]) is str
            and source["block_p_closure_audit_kind"] == SOURCE_KIND
            and type(source["block"]) is str
            and source["block"] == "P"
            and type(source["step"]) is str
            and source["step"] == "18.8"
            and type(source["status"]) is str
            and source["status"] == SOURCE_STATUS
            and type(source["block_p_closure_audit_status"]) is str
            and source["block_p_closure_audit_status"] == SOURCE_CLOSURE_AUDIT_STATUS
            and type(source["block_p_closure_audit_decision"]) is str
            and source["block_p_closure_audit_decision"] == SOURCE_CLOSURE_AUDIT_STATUS.upper()
        ):
            return False
        closure_summary = source["closure_summary"]
        closure_decision = source["closure_decision"]
        auth = source["authorization_audit"]
        capability_audit = source["capability_audit"]
        if not (
            _exact_str_keyed_dict(closure_summary)
            and _exact_str_keyed_dict(closure_decision)
            and _exact_str_keyed_dict(auth)
            and _exact_str_keyed_dict(capability_audit)
        ):
            return False
        caps = capability_audit.get("capability_state")
        if not _exact_str_keyed_dict(caps):
            return False
        for key in (
            "source_18_7_accepted",
            "closure_audit_complete",
            "block_p_source_only_design_closed",
        ):
            if type(closure_summary.get(key)) is not bool or closure_summary[key] is not True:
                return False
        if (
            source["source_18_7_accepted"] is not True
            or source["block_p_source_only_design_closed"] is not True
        ):
            return False
        if source["integrity_valid"] is not True:
            return False
        stage_rows = source["stage_audit_rows"]
        if type(stage_rows) is not list or len(stage_rows) != len(STAGE_STEPS):
            return False
        stage_steps: list[str] = []
        for row in stage_rows:
            if not _exact_str_keyed_dict(row):
                return False
            step = row.get("step")
            if type(step) is not str:
                return False
            stage_steps.append(step)
        if tuple(stage_steps) != STAGE_STEPS:
            return False
        if type(source["closure_findings"]) is not list or len(source["closure_findings"]) != 14:
            return False
        for key in (
            "physical_build_completed",
            "desktop_exe_build_ready",
            "desktop_exe_built",
            "desktop_exe_packaged",
            "desktop_exe_released",
            "runtime_enabled",
            "orders_enabled",
        ):
            if type(closure_decision.get(key)) is not bool or closure_decision[key] is not False:
                return False
        for key in AUTHORIZATION_FALSE_FIELDS:
            if type(auth.get(key)) is not bool or auth[key] is not False:
                return False
        for value in caps.values():
            if type(value) is not str or value != "blocked":
                return False
        if type(source["future_steps"]) is not list or source["future_steps"] != []:
            return False
        if "next_step" in source or "next_step_title" in source:
            return False
        return _integrity_18_8(source) is True
    except Exception:
        return False


def _evidence_requirements() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": evidence_id,
            "required": True,
            "collected": False,
            "validated": False,
            "source_only_definition": True,
        }
        for evidence_id in EVIDENCE_IDS
    ]


def _authorization_state(environment_inventory_authorized: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"environment_inventory_authorized": environment_inventory_authorized}
    payload.update({field: False for field in EXEC_AUTH_FIELDS})
    payload["only_source_only_19_1_handoff_allowed"] = environment_inventory_authorized
    return payload


def _environment_requirements(full: bool) -> dict[str, Any]:
    if not full:
        return {}
    return {
        "windows_host_required": True,
        "supported_windows_version_must_be_confirmed": True,
        "exact_python_version_must_be_confirmed": True,
        "exact_pyside_version_must_be_confirmed": True,
        "exact_packaging_tool_and_version_must_be_selected": True,
        "desktop_entrypoint_must_be_confirmed": True,
        "qml_assets_must_be_confirmed": True,
        "qt_plugins_must_be_confirmed": True,
        "dependency_lock_must_be_resolved": True,
        "secret_and_local_data_exclusion_must_be_confirmed": True,
        "output_name_and_version_policy_must_be_confirmed": True,
        "environment_inventory_performed": False,
        "dependency_resolution_performed": False,
        "pyside_import_performed": False,
        "qml_load_performed": False,
        "qt_plugin_discovery_performed": False,
    }


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_desktop_build_execution_entry_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_desktop_build_execution_entry_contract_status": STATUS,
        "source_18_8_accepted": True,
        "block_q_windows_desktop_build_execution_entry_contract_decision": DECISION,
        "entry_contract_artifact_complete": True,
        "ready_for_block_q_1": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_closure_audit_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_p_closure_audit_kind"],
            "block": "P",
            "step": "18.8",
            "status": source["status"],
            "source_18_7_accepted": True,
            "closure_audit_complete": True,
            "block_p_source_only_design_closed": True,
            "integrity_valid": True,
            "source_top_level_fields": list(SOURCE_18_8_TOP_LEVEL_FIELDS),
        },
        "source_closure_preservation": {
            "preserves_18_8_payload": True,
            "preserves_all_8_stage_rows": True,
            "preserves_all_14_closure_findings": True,
            "preserves_source_only_closure": True,
            "preserves_blocked_capabilities": True,
            "preserves_zero_authorizations": True,
            "preserves_no_physical_build": True,
            "source_closure_modified": False,
            "source_closure_reinterpreted": False,
        },
        "block_q_scope_definition": {
            "block_q_defined": True,
            "block_q_source_only_entry_defined": True,
            "windows_desktop_build_execution_path_defined": True,
            "physical_build_performed": False,
            "packaging_performed": False,
            "artifact_created": False,
            "artifact_scanned": False,
            "artifact_signed": False,
            "installer_created": False,
            "release_performed": False,
            "runtime_started": False,
            "orders_enabled": False,
            "future_explicit_windows_build_environment_inventory": True,
            "future_explicit_build_evidence_collection": True,
            "future_explicit_build_authorization": True,
            "future_explicit_build_command": True,
            "future_explicit_artifact_validation": True,
        },
        "windows_build_environment_requirements": _environment_requirements(True),
        "build_execution_evidence_requirements": _evidence_requirements(),
        "build_execution_authorization_state": _authorization_state(True),
        "non_execution_entry_evidence": {
            "source_read": True,
            "entry_contract_built": True,
            "repo_scan_performed": False,
            "filesystem_scan_performed": False,
            "environment_scan_performed": False,
            "windows_environment_inspected": False,
            "dependency_resolution_performed": False,
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
        },
        "entry_contract_boundaries": {
            "reads_18_8_only": True,
            "source_only": True,
            "plain_data": True,
            "static_entry_contract": True,
            "can_feed_only_19_1_windows_build_environment_inventory": True,
            "repo_rescan": False,
            "filesystem_scan": False,
            "environment_scan": False,
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
        },
        "source_boundaries": {
            "source_step": "18.8",
            "source_block": "P",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "earlier_block_p_builders_read": False,
        },
        "future_steps": [
            {
                "next_step": NEXT_STEP,
                "next_step_title": NEXT_STEP_TITLE,
                "source_only": True,
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
        "block_q_windows_desktop_build_execution_entry_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_desktop_build_execution_entry_contract_status": BLOCKED_STATUS,
        "source_18_8_accepted": False,
        "block_q_windows_desktop_build_execution_entry_contract_decision": BLOCKED_DECISION,
        "entry_contract_artifact_complete": False,
        "ready_for_block_q_1": False,
        "next_step": "",
        "next_step_title": "",
        "windows_build_environment_requirements": {},
        "build_execution_evidence_requirements": [],
        "build_execution_authorization_state": _authorization_state(False),
        "non_execution_entry_evidence": {
            "source_read": False,
            "entry_contract_built": False,
            "repo_scan_performed": False,
            "filesystem_scan_performed": False,
            "environment_scan_performed": False,
            "windows_environment_inspected": False,
            "dependency_resolution_performed": False,
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
        },
        "entry_contract_boundaries": {
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
        },
        "future_steps": [],
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }
    return {field: payload[field] for field in BLOCKED_FIELDS}


def _integrity(payload: Any) -> bool:
    try:
        return _exact_plain(payload, _canonical_nominal()) or _exact_plain(
            payload, _canonical_blocked()
        )
    except Exception:
        return False


def build_preview_block_q_windows_desktop_build_execution_entry_contract() -> dict[str, Any]:
    try:
        source = build_preview_block_p_closure_audit()
    except Exception:
        return _blocked()
    try:
        return _nominal(source) if _source_accepted(source) else _blocked()
    except Exception:
        return _blocked()
