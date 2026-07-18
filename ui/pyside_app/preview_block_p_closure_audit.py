"""FUNCTIONAL-PREVIEW-18.8 Block P source-only closure audit."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_block_p_desktop_exe_build_readiness_read_model import (
    BLOCKED_STATUS as SOURCE_BLOCKED_STATUS,
    KIND as SOURCE_KIND,
    READ_MODEL_STATUS as SOURCE_READ_MODEL_STATUS,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_18_7,
    build_preview_block_p_desktop_exe_build_readiness_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_p_closure_audit.v1"
KIND: Final[str] = "functional_preview_block_p_closure_audit"
BLOCK_ID: Final[str] = "P"
STEP_ID: Final[str] = "18.8"
STATUS: Final[str] = "block_p_source_only_design_closed_real_build_fail_closed"
BLOCKED_STATUS: Final[str] = (
    "blocked_for_functional_preview_18_8_block_p_closure_audit_source_18_7_rejected"
)
CLOSURE_AUDIT_STATUS: Final[str] = (
    "source_18_7_consumed_block_p_source_only_design_closed_18_0_through_18_7_preserved_"
    "17_clauses_6_rules_8_requirements_12_blockers_12_evidence_all_real_capabilities_blocked_"
    "no_build_no_packaging_no_artifact_no_release_no_runtime_no_orders_no_next_stage_started"
)
TOP_LEVEL_FIELDS: Final[list[str]] = [
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
]
BLOCKED_FIELDS: Final[list[str]] = [
    "schema_version",
    "block_p_closure_audit_kind",
    "block",
    "step",
    "source_18_7_accepted",
    "closure_audit_complete",
    "block_p_source_only_design_closed",
    "stage_audit_rows",
    "closure_findings",
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
    "status",
    "integrity_valid",
]

_STAGE_ROWS: Final[list[dict[str, Any]]] = [
    {"step": "18.0", "title": "BLOCK P DESKTOP EXE PACKAGING ENTRY CONTRACT"},
    {"step": "18.1", "title": "BLOCK P DESKTOP EXE PACKAGING SOURCE INVENTORY"},
    {"step": "18.2", "title": "BLOCK P DESKTOP EXE PACKAGING INVENTORY MATRIX"},
    {"step": "18.3", "title": "BLOCK P DESKTOP EXE PACKAGING CONTRACT"},
    {"step": "18.4", "title": "BLOCK P DESKTOP EXE PACKAGING READ MODEL"},
    {"step": "18.5", "title": "BLOCK P DESKTOP EXE BUILD READINESS MATRIX"},
    {"step": "18.6", "title": "BLOCK P DESKTOP EXE BUILD READINESS CONTRACT"},
    {"step": "18.7", "title": "BLOCK P DESKTOP EXE BUILD READINESS READ MODEL"},
]
_CAPABILITY_FIELDS: Final[list[str]] = [
    "desktop_exe_entry_contract",
    "packaging_source_inventory",
    "packaging_inventory_matrix",
    "packaging_contract",
    "packaging_read_model",
    "build_readiness_matrix",
    "build_readiness_contract",
    "build_readiness_read_model",
    "desktop_exe_build",
    "desktop_exe_packaging",
    "artifact_creation",
    "release",
    "runtime",
    "orders",
]
_AUTH_KEYS: Final[list[str]] = [
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
]

SOURCE_18_7_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
)
SOURCE_18_7_REQUIRED_TRUE_FIELDS: Final[tuple[str, ...]] = (
    "source_18_6_accepted",
    "read_model_artifact_complete",
    "ready_for_block_p_8",
    "integrity_valid",
)
SOURCE_18_7_SUMMARY_ZERO_COUNT_FIELDS: Final[tuple[str, ...]] = (
    "satisfied_readiness_clause_count",
    "satisfied_acceptance_rule_count",
    "observed_readiness_count",
    "validated_readiness_count",
    "satisfied_readiness_count",
    "ready_readiness_count",
)
SOURCE_18_7_SUMMARY_FALSE_AUTHORIZATION_FIELDS: Final[tuple[str, ...]] = (
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_authorized",
    "orders_authorized",
)
SOURCE_18_7_DECISION_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "build_ready",
    "packaging_authorized",
    "build_authorized",
    "artifact_creation_authorized",
    "release_authorized",
    "runtime_enabled",
    "orders_enabled",
)
SOURCE_18_7_REQUIRED_TRUE_BOUNDARY_FIELDS: Final[tuple[str, ...]] = (
    "reads_18_6_only",
    "source_only",
    "plain_data",
    "static_read_model",
    "can_feed_only_18_8_closure_audit",
)
SOURCE_18_7_CAPABILITY_SECTION_FIELDS: Final[tuple[str, ...]] = (
    "source_capability_build_readiness_state",
    "contract_capability_state",
    "read_model_capability_view",
)

_COUNT_KEYS: Final[dict[str, int]] = {
    "readiness_clause_count": 17,
    "acceptance_rule_count": 6,
    "unique_requirement_count": 8,
    "unique_blocker_count": 12,
    "unique_evidence_count": 12,
    "satisfied_readiness_clause_count": 0,
    "satisfied_acceptance_rule_count": 0,
    "observed_readiness_count": 0,
    "validated_readiness_count": 0,
    "satisfied_readiness_count": 0,
    "ready_readiness_count": 0,
}


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


def _stage_rows() -> list[dict[str, Any]]:
    return [
        {
            **deepcopy(row),
            "source_only": True,
            "schema_defined": True,
            "artifact_defined": True,
            "preserved_by_following_stage": True,
            "build_performed": False,
            "runtime_started": False,
        }
        for row in _STAGE_ROWS
    ]


def _findings() -> list[dict[str, Any]]:
    names = [
        ("source_chain_18_0_18_7_preserved", "18.7"),
        ("17_readiness_clauses_defined", "18.7"),
        ("6_acceptance_rules_defined", "18.7"),
        ("8_requirements_referenced", "18.7"),
        ("12_blockers_referenced", "18.7"),
        ("12_evidence_items_referenced", "18.7"),
        ("no_evidence_collected", "18.7"),
        ("no_evidence_validated", "18.7"),
        ("no_blocker_resolved", "18.7"),
        ("no_readiness_granted", "18.7"),
        ("no_build_authorization_granted", "18.8"),
        ("no_packaging_authorization_granted", "18.8"),
        ("no_artifact_authorization_granted", "18.8"),
        ("no_physical_artifact_created", "18.8"),
    ]
    return [
        {
            "finding_id": name,
            "defined": True,
            "passed": True,
            "source_step": step,
            "failure_policy": "fail_closed",
        }
        for name, step in names
    ]


def _exact_scalar(value: Any, expected: str | bool | int | None) -> bool:
    if type(value) is not type(expected):
        return False
    try:
        return value == expected
    except Exception:
        return False


def _source_accepted(source: Any) -> bool:
    try:
        if type(source) is not dict:
            return False
        root_keys = tuple(source)
        if not all(type(key) is str for key in root_keys):
            return False
        if root_keys != SOURCE_18_7_TOP_LEVEL_FIELDS:
            return False
        if not (
            _exact_scalar(source["schema_version"], SOURCE_SCHEMA_VERSION)
            and _exact_scalar(
                source["block_p_desktop_exe_build_readiness_read_model_kind"], SOURCE_KIND
            )
            and _exact_scalar(source["block"], "P")
            and _exact_scalar(source["step"], "18.7")
            and _exact_scalar(
                source["block_p_desktop_exe_build_readiness_read_model_status"],
                SOURCE_READ_MODEL_STATUS,
            )
            and _exact_scalar(source["status"], SOURCE_STATUS)
        ):
            return False
        if any(source[field] is not True for field in SOURCE_18_7_REQUIRED_TRUE_FIELDS):
            return False

        summary = source["build_readiness_read_model_summary"]
        if type(summary) is not dict:
            return False
        for field, expected in _COUNT_KEYS.items():
            value = summary[field]
            if type(value) is not int or value != expected:
                return False
        if any(
            summary[field] is not False for field in SOURCE_18_7_SUMMARY_FALSE_AUTHORIZATION_FIELDS
        ):
            return False
        for field in SOURCE_18_7_SUMMARY_ZERO_COUNT_FIELDS:
            value = summary[field]
            if type(value) is not int or value != 0:
                return False

        caps = source["capability_read_model_state"]
        if type(caps) is not dict:
            return False
        cap_keys = tuple(caps)
        if not all(type(key) is str for key in cap_keys):
            return False
        if cap_keys != SOURCE_18_7_CAPABILITY_SECTION_FIELDS:
            return False
        for section in SOURCE_18_7_CAPABILITY_SECTION_FIELDS:
            mapping = caps[section]
            if type(mapping) is not dict:
                return False
            for key, value in mapping.items():
                if type(key) is not str or type(value) is not str:
                    return False
                if value != "blocked":
                    return False

        decision = source["fail_closed_readiness_decision_view"]
        if type(decision) is not dict:
            return False
        if any(decision[field] is not False for field in SOURCE_18_7_DECISION_FALSE_FIELDS):
            return False
        if decision["only_source_only_18_8_handoff_allowed"] is not True:
            return False

        boundaries = source["read_model_boundaries"]
        if type(boundaries) is not dict:
            return False
        if any(
            boundaries[field] is not True for field in SOURCE_18_7_REQUIRED_TRUE_BOUNDARY_FIELDS
        ):
            return False
        return _integrity_18_7(source) is True
    except Exception:
        return False


def _nominal(source: dict[str, Any] | None = None) -> dict[str, Any]:
    source_ref = {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "kind": SOURCE_KIND,
        "block": "P",
        "step": "18.7",
        "status": SOURCE_STATUS,
        "source_18_7_read_by_18_8": True,
        "source_top_level_fields": list(SOURCE_18_7_TOP_LEVEL_FIELDS),
    }
    summary = {
        "source_18_7_accepted": True,
        "block_p_stage_count": 8,
        "all_block_p_stages_defined": True,
        "all_block_p_stages_source_only": True,
        "all_block_p_artifacts_plain_data": True,
        "all_block_p_handoffs_preserved": True,
        "closure_audit_complete": True,
        "block_p_source_only_design_closed": True,
        **_COUNT_KEYS,
        "build_ready": False,
        **{k: False for k in _AUTH_KEYS},
    }
    cap_state = {key: "blocked" for key in _CAPABILITY_FIELDS}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "block_p_closure_audit_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_p_closure_audit_status": CLOSURE_AUDIT_STATUS,
        "block_p_closure_audit_decision": CLOSURE_AUDIT_STATUS.upper(),
        "block_p_closure_audit_ready": True,
        "source_18_7_accepted": True,
        "block_p_source_only_design_closed": True,
        "source_18_7_reference": source_ref,
        "source_acceptance": {
            "accepted": True,
            "source_18_6_accepted": True,
            "read_model_artifact_complete": True,
            "ready_for_block_p_8": True,
            "integrity_valid": True,
        },
        "stage_audit_rows": _stage_rows(),
        "closure_summary": summary,
        "closure_findings": _findings(),
        "preservation_audit": {
            "source_chain_18_0_18_7_preserved": True,
            "read_model_18_7_preserved": True,
            "source_modified_by_18_8": False,
        },
        "readiness_audit": {**_COUNT_KEYS, "build_ready": False},
        "capability_audit": {
            "capability_state": cap_state,
            "all_capabilities_known": True,
            "all_real_capabilities_blocked": True,
            "capability_state_modified": False,
        },
        "authorization_audit": {
            **{k: False for k in _AUTH_KEYS},
            "authorization_granted_by_18_8": False,
        },
        "non_execution_audit": {
            "source_read": True,
            "closure_audit_built": True,
            "repo_scan_performed": False,
            "filesystem_scan_performed": False,
            "environment_scan_performed": False,
            "evidence_collection_performed": False,
            "evidence_validation_performed": False,
            "blocker_resolution_performed": False,
            "build_performed": False,
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
        "closure_decision": {
            "block_p_source_only_design_closed": True,
            "desktop_exe_build_ready": False,
            "desktop_exe_built": False,
            "desktop_exe_packaged": False,
            "desktop_exe_released": False,
            "runtime_enabled": False,
            "orders_enabled": False,
            "physical_build_completed": False,
        },
        "closure_boundaries": {
            "reads_18_7_only": True,
            "source_only": True,
            "plain_data": True,
            "static_closure_audit": True,
            "closes_block_p_source_only_design": True,
            "build_performed": False,
            "packaging_performed": False,
            "artifact_created": False,
            "release_performed": False,
            "runtime_started": False,
            "orders_enabled": False,
        },
        "source_boundaries": {
            "source_step": "FUNCTIONAL-PREVIEW-18.7",
            "reads_18_7_only": True,
            "source_only": True,
            "repo_scan_performed": False,
            "filesystem_scan_performed": False,
            "environment_scan_performed": False,
        },
        "future_steps": [],
        "status": STATUS,
        "integrity_valid": True,
    }
    return payload


def _blocked() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "block_p_closure_audit_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "source_18_7_accepted": False,
        "closure_audit_complete": False,
        "block_p_source_only_design_closed": False,
        "stage_audit_rows": [],
        "closure_findings": [],
        "build_ready": False,
        "packaging_authorized": False,
        "build_authorized": False,
        "artifact_creation_authorized": False,
        "release_authorized": False,
        "runtime_authorized": False,
        "orders_authorized": False,
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


def build_preview_block_p_closure_audit() -> dict[str, Any]:
    try:
        source = build_preview_block_p_desktop_exe_build_readiness_read_model()
    except Exception:
        return _blocked()

    try:
        return _nominal(source) if _source_accepted(source) else _blocked()
    except Exception:
        return _blocked()


__all__ = [
    "SCHEMA_VERSION",
    "KIND",
    "BLOCK_ID",
    "STEP_ID",
    "STATUS",
    "BLOCKED_STATUS",
    "TOP_LEVEL_FIELDS",
    "BLOCKED_FIELDS",
    "SOURCE_18_7_TOP_LEVEL_FIELDS",
    "build_preview_block_p_closure_audit",
    "_integrity",
]
