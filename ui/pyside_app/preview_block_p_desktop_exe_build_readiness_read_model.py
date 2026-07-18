"""FUNCTIONAL-PREVIEW-18.7 source-only build-readiness read model."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_contract import (
    BLOCKED_STATUS as SOURCE_BLOCKED_STATUS,
    BUILD_READINESS_CONTRACT_STATUS,
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    SOURCE_ROWS,
    CAPABILITY_FIELDS,
    MATRIX_SCHEMA_VERSION,
    MATRIX_STATUS,
    STATUS as SOURCE_STATUS,
    TOP_LEVEL_FIELDS as SOURCE_TOP_LEVEL_FIELDS,
    _integrity as _source_integrity,
    build_preview_block_p_desktop_exe_build_readiness_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_desktop_exe_build_readiness_read_model.v1"
KIND: Final[str] = "functional_preview_block_p_desktop_exe_build_readiness_read_model"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.7"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-18.8"
NEXT_STEP_TITLE: Final[str] = "BLOCK P CLOSURE AUDIT"
STATUS: Final[str] = "ready_for_functional_preview_18_8_block_p_closure_audit_source_only"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_8_block_p_build_readiness_read_model_source_18_6_rejected"
)
READ_MODEL_STATUS: Final[str] = (
    "source_18_6_consumed_build_readiness_contract_preserved_source_only_plain_data_read_model_"
    "complete_17_clauses_6_acceptance_rules_8_requirements_12_blockers_12_evidence_"
    "all_states_fail_closed_only_source_only_handoff_to_18_8_allowed"
)
TOP_LEVEL_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_build_readiness_read_model_kind",
    "block",
    "step",
    "block_p_desktop_exe_build_readiness_read_model_status",
    "source_18_6_accepted",
    "block_p_desktop_exe_build_readiness_read_model_decision",
    "read_model_artifact_complete",
    "ready_for_block_p_8",
    "next_step",
    "next_step_title",
    "block_p_desktop_exe_build_readiness_contract_reference",
    "source_contract_preservation",
    "build_readiness_read_model_summary",
    "readiness_clause_read_rows",
    "acceptance_rule_read_rows",
    "capability_read_model_state",
    "fail_closed_readiness_decision_view",
    "non_execution_read_model_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
]

SOURCE_CONTRACT_BOUNDARY_FIELDS: Final[list[str]] = [
    "reads_18_5_only",
    "source_only",
    "plain_data",
    "static_contract",
    "can_feed_only_18_7_build_readiness_read_model",
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
    "packaging_performed",
    "artifact_created",
    "artifact_scanned",
    "artifact_signed",
    "installer_created",
    "release_performed",
    "runtime_started",
    "orders_enabled",
    "network_opened",
    "credentials_read",
    "qml_bridge_gateway_controller_changed",
]

BLOCKED_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_desktop_exe_build_readiness_read_model_kind",
    "block",
    "step",
    "block_p_desktop_exe_build_readiness_read_model_status",
    "source_18_6_accepted",
    "read_model_artifact_complete",
    "ready_for_block_p_8",
    "readiness_clause_read_rows",
    "acceptance_rule_read_rows",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "status",
    "integrity_valid",
]


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
                pending.extend((actual[k], trusted[k]) for k in reversed(list(trusted)))
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


def _source_ids(name: str) -> list[str]:
    result: list[str] = []
    for row in SOURCE_ROWS:
        for item in row[name]:
            if item not in result:
                result.append(item)
    return result


def _source_rows_template() -> list[dict[str, Any]]:
    return [
        {
            "readiness_id": row["readiness_id"],
            "source_requirement_ids": deepcopy(row["source_requirement_ids"]),
            "source_blocker_ids": deepcopy(row["source_blocker_ids"]),
            "required_evidence_ids": deepcopy(row["required_evidence_ids"]),
            "contract_clause_id": "contract_" + row["readiness_id"],
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
        for row in SOURCE_ROWS
    ]


def _source_rules_template() -> list[dict[str, Any]]:
    clauses = ["contract_" + row["readiness_id"] for row in SOURCE_ROWS]
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
            "required_evidence_ids": _source_ids("required_evidence_ids"),
            "evidence_collected": False,
            "evidence_validated": False,
            "rule_defined": True,
            "rule_satisfied": False,
        },
        {
            "rule_id": "all_source_blockers_resolved",
            "required_blocker_ids": _source_ids("source_blocker_ids"),
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


def _source_reference_template() -> dict[str, Any]:
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
    return {
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
        **{key: False for key in false},
    }


def _source_summary_template() -> dict[str, Any]:
    return {
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
        "unique_requirement_count": len(_source_ids("source_requirement_ids")),
        "unique_blocker_count": len(_source_ids("source_blocker_ids")),
        "unique_evidence_count": len(_source_ids("required_evidence_ids")),
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


def _source_decision_template() -> dict[str, Any]:
    return {
        "source_matrix_exists_and_accepted": True,
        "contract_definitions_complete": True,
        "contract_satisfied": False,
        "no_readiness_row_satisfied": True,
        "no_blocker_resolved": True,
        "no_evidence_collected": True,
        "no_evidence_validated": True,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_enabled": False,
        "orders_enabled": False,
        "only_source_only_18_7_handoff_allowed": True,
        "build_performed_by_18_6": False,
        "packaging_authorized_by_18_6": False,
        "build_authorized_by_18_6": False,
        "artifact_creation_authorized_by_18_6": False,
        "release_performed_by_18_6": False,
        "runtime_enabled_by_18_6": False,
        "orders_enabled_by_18_6": False,
    }


def _source_contract_boundaries_template() -> dict[str, bool]:
    return {key: index < 5 for index, key in enumerate(SOURCE_CONTRACT_BOUNDARY_FIELDS)}


def _source_boundaries_template() -> dict[str, Any]:
    return {
        "source_block_p_desktop_exe_build_readiness_matrix": "FUNCTIONAL-PREVIEW-18.5",
        "build_readiness_matrix_preserved": True,
        "can_build_desktop_exe_build_readiness_contract": True,
        "build_readiness_contract_artifact_complete": True,
        "can_build_desktop_exe_build_readiness_read_model": True,
        "can_feed_18_7": True,
    }


def _trusted_source_template() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_p_desktop_exe_build_readiness_contract_kind": SOURCE_KIND,
        "block": "P",
        "step": "18.6",
        "block_p_desktop_exe_build_readiness_contract_status": BUILD_READINESS_CONTRACT_STATUS,
        "source_18_5_accepted": True,
        "block_p_desktop_exe_build_readiness_contract_decision": BUILD_READINESS_CONTRACT_STATUS.upper(),
        "build_readiness_contract_artifact_complete": True,
        "ready_for_block_p_7": True,
        "next_step": "FUNCTIONAL-PREVIEW-18.7",
        "next_step_title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL",
        "block_p_desktop_exe_build_readiness_matrix_reference": _source_reference_template(),
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
        "build_readiness_contract_summary": _source_summary_template(),
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
        "build_readiness_contract_rows": _source_rows_template(),
        "build_readiness_acceptance_rules": _source_rules_template(),
        "capability_contract_state": {
            "source_capability_build_readiness_state": {
                key: "blocked" for key in CAPABILITY_FIELDS
            },
            "contract_capability_state": {key: "blocked" for key in CAPABILITY_FIELDS},
            "source_capabilities_known_blocked": True,
            "contract_capabilities_known_blocked": True,
            "all_real_capabilities_blocked_at_18_6": True,
        },
        "fail_closed_build_readiness_contract_decision": _source_decision_template(),
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
        "contract_boundaries": _source_contract_boundaries_template(),
        "source_boundaries": _source_boundaries_template(),
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
        "status": SOURCE_STATUS,
        "integrity_valid": True,
    }


def _source_accepted(source: Any) -> bool:
    if not _exact_plain(source, _trusted_source_template()):
        return False
    if source.get("source_18_5_accepted") is not True:
        return False
    if source.get("build_readiness_contract_artifact_complete") is not True:
        return False
    if source.get("ready_for_block_p_7") is not True:
        return False
    if source.get("integrity_valid") is not True:
        return False
    try:
        return _source_integrity(source) is True
    except Exception:
        return False


def _ids(rows: list[dict[str, Any]], name: str) -> list[str]:
    result: list[str] = []
    for row in rows:
        for item in row[name]:
            if item not in result:
                result.append(item)
    return result


def _read_rows(source: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for index, row in enumerate(source["build_readiness_contract_rows"], start=1):
        rows.append(
            {
                "read_model_row_id": f"read_model_{row['contract_clause_id']}",
                "read_by_18_7": True,
                "visible_in_read_model": True,
                "readiness_id": row["readiness_id"],
                "source_requirement_ids": deepcopy(row["source_requirement_ids"]),
                "source_blocker_ids": deepcopy(row["source_blocker_ids"]),
                "required_evidence_ids": deepcopy(row["required_evidence_ids"]),
                "contract_clause_id": row["contract_clause_id"],
                "source_observed": row["source_observed"],
                "source_validated": row["source_validated"],
                "source_satisfied": row["source_satisfied"],
                "source_ready": row["source_ready"],
                "source_blocks_build": row["source_blocks_build"],
                "source_blocks_packaging": row["source_blocks_packaging"],
                "source_blocks_artifact_creation": row["source_blocks_artifact_creation"],
                "source_readiness_state": row["source_readiness_state"],
                "source_readiness_result": row["source_readiness_result"],
                "source_failure_policy": row["source_failure_policy"],
                "contract_clause_defined": row["contract_clause_defined"],
                "contract_clause_satisfied": row["contract_clause_satisfied"],
                "build_readiness_granted": row["build_readiness_granted"],
                "packaging_authorization_granted": row["packaging_authorization_granted"],
                "build_authorization_granted": row["build_authorization_granted"],
                "artifact_creation_authorization_granted": row[
                    "artifact_creation_authorization_granted"
                ],
                "failure_policy": row["failure_policy"],
                "source_order": index,
            }
        )
    return rows


def _rule_reads(source: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"read_by_18_7": True, "visible_in_read_model": True, **deepcopy(rule)}
        for rule in source["build_readiness_acceptance_rules"]
    ]


def _nominal(source: dict[str, Any] | None = None) -> dict[str, Any]:
    src = _trusted_source_template() if source is None else deepcopy(source)
    rows = _read_rows(src)
    rules = _rule_reads(src)
    summary = {
        "source_18_6_accepted": True,
        "source_contract_preserved": True,
        "read_model_complete": True,
        "ready_for_block_p_8": True,
        "readiness_clause_count": 17,
        "defined_readiness_clause_count": 17,
        "satisfied_readiness_clause_count": 0,
        "unique_requirement_count": len(_ids(rows, "source_requirement_ids")),
        "unique_blocker_count": len(_ids(rows, "source_blocker_ids")),
        "unique_evidence_count": len(_ids(rows, "required_evidence_ids")),
        "observed_readiness_count": 0,
        "validated_readiness_count": 0,
        "satisfied_readiness_count": 0,
        "ready_readiness_count": 0,
        "acceptance_rule_count": 6,
        "satisfied_acceptance_rule_count": 0,
        "build_readiness_contract_satisfied": False,
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
    }
    boundaries = {
        "reads_18_6_only": True,
        "source_only": True,
        "plain_data": True,
        "static_read_model": True,
        "can_feed_only_18_8_closure_audit": True,
    }
    for key in SOURCE_CONTRACT_BOUNDARY_FIELDS:
        if key not in {"source_only", "plain_data"}:
            boundaries[key] = False
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_build_readiness_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_desktop_exe_build_readiness_read_model_status": READ_MODEL_STATUS,
        "source_18_6_accepted": True,
        "block_p_desktop_exe_build_readiness_read_model_decision": READ_MODEL_STATUS.upper(),
        "read_model_artifact_complete": True,
        "ready_for_block_p_8": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_p_desktop_exe_build_readiness_contract_reference": {
            "schema_version": SOURCE_SCHEMA_VERSION,
            "kind": SOURCE_KIND,
            "block": "P",
            "step": "18.6",
            "status": SOURCE_STATUS,
            "contract_status": BUILD_READINESS_CONTRACT_STATUS,
            "integrity_valid": True,
            "source_18_5_accepted": True,
            "build_readiness_contract_artifact_complete": True,
            "ready_for_block_p_7": True,
            "source_top_level_fields": deepcopy(SOURCE_TOP_LEVEL_FIELDS),
        },
        "source_contract_preservation": {
            "preserves_18_6_payload": True,
            "preserves_all_17_readiness_clauses": True,
            "preserves_all_6_acceptance_rules": True,
            "preserves_row_order": True,
            "preserves_rule_order": True,
            "preserves_requirement_blocker_evidence_links": True,
            "source_contract_modified": False,
            "source_contract_reinterpreted": False,
        },
        "build_readiness_read_model_summary": summary,
        "readiness_clause_read_rows": rows,
        "acceptance_rule_read_rows": rules,
        "capability_read_model_state": {
            "source_capability_build_readiness_state": deepcopy(
                src["capability_contract_state"]["source_capability_build_readiness_state"]
            ),
            "contract_capability_state": deepcopy(
                src["capability_contract_state"]["contract_capability_state"]
            ),
            "read_model_capability_view": deepcopy(
                src["capability_contract_state"]["contract_capability_state"]
            ),
        },
        "fail_closed_readiness_decision_view": {
            **deepcopy(src["fail_closed_build_readiness_contract_decision"]),
            "read_model_built_by_18_7": True,
            "source_read_by_18_7": True,
            "nothing_executed_by_18_7": True,
            "nothing_authorized_by_18_7": True,
            "only_source_only_18_8_handoff_allowed": True,
        },
        "non_execution_read_model_evidence": {
            "source_read": True,
            "read_model_built": True,
            "source_builder_call_count": 1,
            "build_performed": False,
            "packaging_performed": False,
            "artifact_created": False,
            "release_performed": False,
            "runtime_started": False,
            "network_opened": False,
            "credentials_read": False,
            "orders_enabled": False,
        },
        "read_model_boundaries": boundaries,
        "source_boundaries": deepcopy(src["contract_boundaries"]),
        "future_steps": [
            {
                "step": "18.8",
                "title": NEXT_STEP_TITLE,
                "source_only": True,
                "build_performed": False,
            }
        ],
        "status": STATUS,
        "integrity_valid": True,
    }
    return payload


def _blocked() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "block_p_desktop_exe_build_readiness_read_model_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_desktop_exe_build_readiness_read_model_status": BLOCKED_STATUS,
        "source_18_6_accepted": False,
        "read_model_artifact_complete": False,
        "ready_for_block_p_8": False,
        "readiness_clause_read_rows": [],
        "acceptance_rule_read_rows": [],
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


def build_preview_block_p_desktop_exe_build_readiness_read_model() -> dict[str, Any]:
    source = build_preview_block_p_desktop_exe_build_readiness_contract()
    return _nominal(source) if _source_accepted(source) else _blocked()
