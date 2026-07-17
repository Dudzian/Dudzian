"""FUNCTIONAL-PREVIEW-18.6 static fail-closed build-readiness contract."""

from __future__ import annotations
import json
from copy import deepcopy
from typing import Any, Final
from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_matrix import (
    SCHEMA_VERSION as MATRIX_SCHEMA_VERSION,
    STATUS as MATRIX_STATUS,
    READINESS_ROWS,
    CAPABILITY_FIELDS,
    _integrity as _matrix_integrity,
    build_preview_block_p_desktop_exe_build_readiness_matrix,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_build_readiness_contract.v1"
KIND: Final[str] = "functional_preview_block_p_desktop_exe_build_readiness_contract"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.6"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-18.7"
NEXT_STEP_TITLE: Final[str] = "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL"
STATUS: Final[str] = (
    "ready_for_functional_preview_18_7_source_18_5_accepted_contract_defined_build_not_ready"
)
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_7_source_18_5_rejected_contract_not_built"
)
BUILD_READINESS_CONTRACT_STATUS: Final[str] = (
    "source_18_5_accepted_17_clauses_defined_zero_readiness_evidence_or_authorization_only_source_only_handoff_to_18_7_allowed"
)
TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_build_readiness_contract_kind",
    "block",
    "step",
    "block_p_desktop_exe_build_readiness_contract_status",
    "source_18_5_accepted",
    "block_p_desktop_exe_build_readiness_contract_decision",
    "build_readiness_contract_artifact_complete",
    "ready_for_block_p_7",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_build_readiness_matrix_reference",
    "source_matrix_preservation",
    "build_readiness_contract_summary",
    "build_readiness_contract_principles",
    "build_readiness_contract_rows",
    "build_readiness_acceptance_rules",
    "capability_contract_state",
    "fail_closed_build_readiness_contract_decision",
    "non_execution_contract_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
]
BLOCKED_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_build_readiness_contract_kind",
    "block",
    "step",
    "block_p_desktop_exe_build_readiness_contract_status",
    "source_18_5_accepted",
    "build_readiness_contract_artifact_complete",
    "ready_for_block_p_7",
    "build_readiness_contract_rows",
    "build_readiness_acceptance_rules",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "status",
    "integrity_valid",
]
SOURCE_ROWS: Final[list[dict[str, Any]]] = deepcopy(READINESS_ROWS)


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
                pending.extend((actual[k], trusted[k]) for k in trusted)
            elif type(trusted) is list:
                pair = (id(actual), id(trusted))
                if pair in visited:
                    continue
                visited.add(pair)
                if len(actual) != len(trusted):
                    return False
                pending.extend(zip(actual, trusted))
            elif type(trusted) not in (str, bool, int, type(None)) or actual != trusted:
                return False
        return True
    except Exception:
        return False


def _source_template() -> dict[str, Any]:
    rows = [
        {
            **deepcopy(r),
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
        for r in SOURCE_ROWS
    ]
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "block": "P",
        "step": "18.5",
        "status": MATRIX_STATUS,
        "source_18_4_accepted": True,
        "ready_for_build_execution": False,
        "readiness_rows": rows,
        "capability_build_readiness_state": {k: "blocked" for k in CAPABILITY_FIELDS},
        "boundaries": {
            "source_only": True,
            "plain_data": True,
            "build_performed": False,
            "filesystem_scanned": False,
            "network_opened": False,
            "runtime_started": False,
            "orders_enabled": False,
        },
        "integrity_valid": True,
    }


def _source_accepted(source: Any) -> bool:
    if not _exact_plain(source, _source_template()):
        return False
    try:
        return _matrix_integrity(source) is True
    except Exception:
        return False


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "readiness_id": r["readiness_id"],
            "source_requirement_ids": deepcopy(r["source_requirement_ids"]),
            "source_blocker_ids": deepcopy(r["source_blocker_ids"]),
            "required_evidence_ids": deepcopy(r["required_evidence_ids"]),
            "contract_clause_id": "contract_" + r["readiness_id"],
            "source_observed": False,
            "source_validated": False,
            "source_satisfied": False,
            "source_ready": False,
            "source_blocks_build": True,
            "source_blocks_packaging": True,
            "source_blocks_artifact_creation": True,
            "source_readiness_state": "required_not_observed",
            "source_readiness_result": "future_explicit_evidence_required",
            "source_failure_policy": "fail_closed",
            "required_observation": True,
            "required_validation": True,
            "required_satisfaction": True,
            "required_ready_state": True,
            "contract_clause_defined": True,
            "contract_clause_satisfied": False,
            "build_readiness_granted": False,
            "packaging_authorization_granted": False,
            "build_authorization_granted": False,
            "artifact_creation_authorization_granted": False,
            "failure_policy": "fail_closed",
        }
        for r in SOURCE_ROWS
    ]


def _ids(name: str) -> list[str]:
    return list(dict.fromkeys(x for r in SOURCE_ROWS for x in r[name]))


def _rules() -> list[dict[str, Any]]:
    clauses = ["contract_" + r["readiness_id"] for r in SOURCE_ROWS]
    return [
        {
            "rule_id": "all_readiness_clauses_satisfied",
            "required_contract_clause_ids": clauses,
            "rule_defined": True,
            "rule_satisfied": False,
            "grants_build_readiness": False,
        },
        {
            "rule_id": "all_required_evidence_collected_and_validated",
            "required_evidence_ids": _ids("required_evidence_ids"),
            "evidence_collected": False,
            "evidence_validated": False,
            "rule_defined": True,
            "rule_satisfied": False,
        },
        {
            "rule_id": "all_source_blockers_resolved",
            "required_blocker_ids": _ids("source_blocker_ids"),
            "blockers_resolved": False,
            "rule_defined": True,
            "rule_satisfied": False,
        },
        {
            "rule_id": "all_readiness_rows_observed_validated_satisfied_ready",
            "rule_defined": True,
            "observed": False,
            "validated": False,
            "satisfied": False,
            "ready": False,
            "rule_satisfied": False,
        },
        {
            "rule_id": "explicit_future_build_execution_gate",
            "rule_defined": True,
            "gate_present": False,
            "gate_approved": False,
            "rule_satisfied": False,
        },
        {
            "rule_id": "build_authorization_requires_all_previous_rules",
            "rule_defined": True,
            "rule_satisfied": False,
            "grants_packaging_authorization": False,
            "grants_build_authorization": False,
            "grants_artifact_creation_authorization": False,
        },
    ]


def _nominal() -> dict[str, Any]:
    rows = _rows()
    summary = {
        "source_18_5_accepted": True,
        "source_build_readiness_matrix_preserved": True,
        "source_only": True,
        "plain_data": True,
        "static_contract": True,
        "build_readiness_contract_artifact_complete": True,
        "ready_for_block_p_7": True,
        "readiness_clause_count": 17,
        "defined_readiness_clause_count": 17,
        "satisfied_readiness_clause_count": 0,
        "unique_requirement_count": len(_ids("source_requirement_ids")),
        "unique_blocker_count": len(_ids("source_blocker_ids")),
        "unique_evidence_count": len(_ids("required_evidence_ids")),
        "observed_readiness_count": 0,
        "validated_readiness_count": 0,
        "satisfied_readiness_count": 0,
        "ready_readiness_count": 0,
        "acceptance_rule_count": 6,
        "satisfied_acceptance_rule_count": 0,
        "contract_definitions_complete": True,
        "build_readiness_contract_satisfied": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
        "only_source_only_18_7_handoff_allowed": True,
    }
    false = [
        "repo_rescan",
        "filesystem_scan",
        "environment_scan",
        "secret_file_read",
        "dependency_import",
        "dependency_resolution",
        "pyside_import",
        "qml_load",
        "qt_plugin_discovery",
        "entrypoint_selection",
        "entrypoint_validation",
        "evidence_collection",
        "evidence_validation",
        "blocker_resolution",
        "readiness_approval",
        "build_gate_approval",
        "build_readiness_grant",
        "packaging_authorization",
        "build_authorization",
        "spec_file_creation",
        "build_command_creation",
        "build_command_execution",
        "packaging",
        "artifact_creation",
        "artifact_scan",
        "artifact_signing",
        "installer_creation",
        "release",
        "runtime",
        "orders",
        "network",
        "credentials_read",
    ]
    ref = {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "kind": "functional_preview_block_p_desktop_exe_build_readiness_matrix",
        "block": "P",
        "step": "18.5",
        "status": MATRIX_STATUS,
        "source_18_4_accepted": True,
        "integrity_valid": True,
        "ready_for_build_execution": False,
        "readiness_row_count": 17,
        "source_read_by_18_6": True,
        "source_available_before_contract": True,
        "static_build_readiness_contract_only": True,
        "contract_built_by_18_6": True,
        "ready_for_functional_preview_18_7": True,
        **{k: False for k in false},
    }
    caps = {
        "source_capability_build_readiness_state": {k: "blocked" for k in CAPABILITY_FIELDS},
        "contract_capability_state": {k: "blocked" for k in CAPABILITY_FIELDS},
        "source_capabilities_known_blocked": True,
        "contract_capabilities_known_blocked": True,
        "all_real_capabilities_blocked_at_18_6": True,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_build_readiness_contract_kind": KIND,
        "block": "P",
        "step": "18.6",
        "block_p_desktop_exe_build_readiness_contract_status": BUILD_READINESS_CONTRACT_STATUS,
        "source_18_5_accepted": True,
        "block_p_desktop_exe_build_readiness_contract_decision": BUILD_READINESS_CONTRACT_STATUS.upper(),
        "build_readiness_contract_artifact_complete": True,
        "ready_for_block_p_7": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_desktop_exe_build_readiness_matrix_reference": ref,
        "source_matrix_preservation": {
            "preserves_source_identity": True,
            "preserves_all_17_readiness_rows": True,
            "preserves_row_order": True,
            "preserves_source_requirement_links": True,
            "preserves_source_blocker_links": True,
            "preserves_required_evidence_links": True,
            "preserves_readiness_flags": True,
            "preserves_capability_state": True,
            "preserves_boundaries": True,
            "preserves_failure_policy": True,
            "preserves_source_status": True,
            "preserves_referential_integrity": True,
            "source_matrix_modified": False,
            "source_matrix_recalculated": False,
            "source_readiness_reinterpreted": False,
            "source_links_modified": False,
            "source_capabilities_modified": False,
            "source_boundaries_modified": False,
            "repo_rescanned": False,
        },
        "build_readiness_contract_summary": summary,
        "build_readiness_contract_principles": {
            "all_17_readiness_clauses_required": True,
            "all_required_evidence_must_be_collected": True,
            "all_required_evidence_must_be_validated": True,
            "all_source_blockers_must_be_resolved": True,
            "all_readiness_rows_must_be_observed": True,
            "all_readiness_rows_must_be_validated": True,
            "all_readiness_rows_must_be_satisfied": True,
            "all_readiness_rows_must_be_ready": True,
            "explicit_future_build_gate_required": True,
            "partial_readiness_never_authorizes_build": True,
            "missing_evidence_fails_closed": True,
            "invalid_evidence_fails_closed": True,
            "unknown_fields_fail_closed": True,
            "source_tampering_fails_closed": True,
            "contract_does_not_execute_build": True,
            "contract_does_not_authorize_packaging": True,
            "contract_does_not_authorize_build": True,
            "contract_does_not_authorize_artifact_creation": True,
        },
        "build_readiness_contract_rows": rows,
        "build_readiness_acceptance_rules": _rules(),
        "capability_contract_state": caps,
        "fail_closed_build_readiness_contract_decision": {
            "source_matrix_exists_and_accepted": True,
            "contract_definitions_complete": True,
            "no_readiness_row_satisfied": True,
            "no_blocker_resolved": True,
            "no_evidence_collected": True,
            "no_evidence_validated": True,
            "only_source_only_18_7_handoff_allowed": True,
            "contract_satisfied": False,
            "build_ready": False,
            "packaging_authorized": False,
            "build_authorized": False,
            "artifact_creation_authorized": False,
            "release_authorized": False,
            "runtime_enabled": False,
            "orders_enabled": False,
        },
        "non_execution_contract_evidence": {
            "source_read": True,
            "contract_built": True,
            "evidence_collection_performed": False,
            "evidence_validation_performed": False,
            "blocker_resolution_performed": False,
            "build_performed": False,
            "packaging_performed": False,
            "artifact_creation_performed": False,
            "release_performed": False,
            "runtime_performed": False,
            "orders_performed": False,
            "network_performed": False,
            "credentials_performed": False,
        },
        "contract_boundaries": {
            "reads_18_5_only": True,
            "source_only": True,
            "plain_data": True,
            "static_contract": True,
            "can_feed_only_18_7_build_readiness_read_model": True,
            "network_opened": False,
            "repo_rescan": False,
            "filesystem_scan": False,
            "environment_scan": False,
            "secret_file_read": False,
            "dependency_import": False,
            "dependency_resolution": False,
            "pyside_import": False,
            "qml_load": False,
            "qt_plugin_discovery": False,
            "entrypoint_selection": False,
            "entrypoint_validation": False,
            "evidence_collection": False,
            "evidence_validation": False,
            "blocker_resolution": False,
            "readiness_approval": False,
            "build_gate_approval": False,
            "build_readiness_grant": False,
            "packaging_authorization": False,
            "build_authorization": False,
            "spec_file_creation": False,
            "build_command_creation": False,
            "build_command_execution": False,
            "packaging_performed": False,
            "artifact_created": False,
            "artifact_scanned": False,
            "artifact_signed": False,
            "installer_created": False,
            "release_performed": False,
            "runtime_started": False,
            "orders_enabled": False,
            "credentials_read": False,
            "qml_bridge_gateway_controller_changed": False,
        },
        "source_boundaries": {
            "source_block_p_desktop_exe_build_readiness_matrix": "FUNCTIONAL-PREVIEW-18.5",
            "build_readiness_matrix_preserved": True,
            "can_build_desktop_exe_build_readiness_contract": True,
            "build_readiness_contract_artifact_complete": True,
            "can_build_desktop_exe_build_readiness_read_model": True,
            "can_feed_18_7": True,
        },
        "future_steps": [
            {
                "step": "18.7",
                "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
                "source_only": True,
                "build_performed": False,
            },
            {
                "step": "18.8",
                "title": "BLOCK P CLOSURE AUDIT",
                "source_only": True,
                "build_performed": False,
            },
        ],
        "status": STATUS,
        "integrity_valid": True,
    }
    return payload


def _blocked() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_build_readiness_contract_kind": KIND,
        "block": "P",
        "step": "18.6",
        "block_p_desktop_exe_build_readiness_contract_status": BLOCKED_STATUS,
        "source_18_5_accepted": False,
        "build_readiness_contract_artifact_complete": False,
        "ready_for_block_p_7": False,
        "build_readiness_contract_rows": [],
        "build_readiness_acceptance_rules": [],
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }


def _integrity(payload: Any) -> bool:
    if type(payload) is not dict:
        return False
    try:
        keys = list(payload)
    except Exception:
        return False
    if not all(type(k) is str for k in keys):
        return False
    expected = (
        _nominal() if keys == TOP_LEVEL_FIELDS else _blocked() if keys == BLOCKED_FIELDS else None
    )
    if expected is None or not _exact_plain(payload, expected):
        return False
    try:
        json.dumps(payload)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def build_preview_block_p_desktop_exe_build_readiness_contract() -> dict[str, Any]:
    source = build_preview_block_p_desktop_exe_build_readiness_matrix()
    return _nominal() if _source_accepted(source) else _blocked()
