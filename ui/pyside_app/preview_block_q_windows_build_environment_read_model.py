"""FUNCTIONAL-PREVIEW-19.4 Block Q Windows build environment read model."""

from __future__ import annotations
from typing import Any, Final
from ui.pyside_app.preview_block_q_windows_build_environment_contract import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_3,
    build_preview_block_q_windows_build_environment_contract,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_build_environment_read_model.v1"
KIND: Final[str] = "functional_preview_block_q_windows_build_environment_read_model"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.4"
NEXT_STEP: Final[str] = ""
NEXT_STEP_TITLE: Final[str] = ""
STATUS: Final[str] = (
    "source_19_3_accepted_11_row_environment_read_model_defined_0_actual_reads_0_evidence_environment_build_not_ready_handoff_blocked_next_step_contract_missing"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_4_source_19_3_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()
SOURCE_19_3_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_contract_kind",
    "block",
    "step",
    "block_q_windows_build_environment_contract_status",
    "source_19_2_accepted",
    "block_q_windows_build_environment_contract_decision",
    "environment_contract_artifact_complete",
    "environment_observation_complete",
    "environment_build_ready",
    "ready_for_block_q_4",
    "next_step",
    "next_step_title",
    "block_q_19_2_matrix_reference",
    "source_matrix_preservation",
    "environment_contract_scope",
    "environment_contract_rows",
    "environment_contract_summary",
    "build_execution_authorization_state",
    "non_execution_contract_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
CONTRACT_ROW_FIELDS: Final[tuple[str, ...]] = (
    "contract_id",
    "matrix_id",
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "source_matrix_state",
    "source_blocker_code",
    "source_evidence_required",
    "source_evidence_collected",
    "source_evidence_validated",
    "acceptance_rule",
    "satisfaction_rule",
    "current_contract_state",
    "requirement_satisfied",
    "build_gate_open",
    "source_only_definition",
)
READ_MODEL_ROW_FIELDS: Final[tuple[str, ...]] = (
    "read_model_id",
    "contract_id",
    "matrix_id",
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "source_contract_state",
    "source_requirement_satisfied",
    "source_build_gate_open",
    "read_required",
    "read_status",
    "observed_value",
    "evidence_collected",
    "evidence_validated",
    "read_model_state",
    "source_only_definition",
)
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_read_model_kind",
    "block",
    "step",
    "block_q_windows_build_environment_read_model_status",
    "source_19_3_accepted",
    "block_q_windows_build_environment_read_model_decision",
    "environment_read_model_artifact_complete",
    "environment_observation_complete",
    "environment_build_ready",
    "ready_for_block_q_5",
    "next_step",
    "next_step_title",
    "block_q_19_3_contract_reference",
    "source_contract_preservation",
    "environment_read_model_scope",
    "environment_read_model_rows",
    "environment_read_model_summary",
    "build_execution_authorization_state",
    "non_execution_read_model_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
SPECS: Final[tuple[tuple[str, str, str, str, str, str], ...]] = (
    (
        "read_model_windows_host",
        "contract_windows_host",
        "matrix_windows_host",
        "windows_host",
        "windows_host_required",
        "platform",
    ),
    (
        "read_model_supported_windows_version",
        "contract_supported_windows_version",
        "matrix_supported_windows_version",
        "supported_windows_version",
        "supported_windows_version_must_be_confirmed",
        "platform",
    ),
    (
        "read_model_python_version",
        "contract_python_version",
        "matrix_python_version",
        "python_version",
        "exact_python_version_must_be_confirmed",
        "runtime",
    ),
    (
        "read_model_pyside_version",
        "contract_pyside_version",
        "matrix_pyside_version",
        "pyside_version",
        "exact_pyside_version_must_be_confirmed",
        "framework",
    ),
    (
        "read_model_packaging_tool_and_version",
        "contract_packaging_tool_and_version",
        "matrix_packaging_tool_and_version",
        "packaging_tool_and_version",
        "exact_packaging_tool_and_version_must_be_selected",
        "packaging",
    ),
    (
        "read_model_desktop_entrypoint",
        "contract_desktop_entrypoint",
        "matrix_desktop_entrypoint",
        "desktop_entrypoint",
        "desktop_entrypoint_must_be_confirmed",
        "application",
    ),
    (
        "read_model_qml_assets",
        "contract_qml_assets",
        "matrix_qml_assets",
        "qml_assets",
        "qml_assets_must_be_confirmed",
        "assets",
    ),
    (
        "read_model_qt_plugins",
        "contract_qt_plugins",
        "matrix_qt_plugins",
        "qt_plugins",
        "qt_plugins_must_be_confirmed",
        "plugins",
    ),
    (
        "read_model_dependency_lock",
        "contract_dependency_lock",
        "matrix_dependency_lock",
        "dependency_lock",
        "dependency_lock_must_be_resolved",
        "dependencies",
    ),
    (
        "read_model_secret_and_local_data_exclusion",
        "contract_secret_and_local_data_exclusion",
        "matrix_secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_must_be_confirmed",
        "security",
    ),
    (
        "read_model_output_name_and_version_policy",
        "contract_output_name_and_version_policy",
        "matrix_output_name_and_version_policy",
        "output_name_and_version_policy",
        "output_name_and_version_policy_must_be_confirmed",
        "output",
    ),
)
EXECUTION_FALSE_FIELDS: Final[tuple[str, ...]] = (
    "environment_scan",
    "windows_host_inspection",
    "dependency_resolution",
    "pyside_import",
    "qml_load",
    "qt_plugin_discovery",
    "build_command_creation",
    "build_command_execution",
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


def _exact_dict(value: Any) -> bool:
    try:
        return type(value) is dict and all(type(k) is str for k in list(value))
    except Exception:
        return False


def _scalar(value: Any, typ: type, expected: Any) -> bool:
    if type(value) is not typ:
        return False
    if typ is bool:
        return value is expected
    return value == expected


def _exact_plain(a: Any, b: Any) -> bool:
    try:
        pending: list[tuple[Any, Any]] = [(a, b)]
        seen: list[tuple[int, int]] = []
        amap: list[tuple[int, int]] = []
        bmap: list[tuple[int, int]] = []
        while pending:
            x, y = pending.pop()
            if type(x) is not type(y):
                return False
            if type(y) is dict:
                xid, yid = id(x), id(y)
                for ax, by in amap:
                    if ax == xid and by != yid:
                        return False
                for by, ax in bmap:
                    if by == yid and ax != xid:
                        return False
                pair = (xid, yid)
                if pair in seen:
                    continue
                seen.append(pair)
                amap.append((xid, yid))
                bmap.append((yid, xid))
                kx = list(x)
                ky = list(y)
                if not all(type(k) is str for k in kx) or kx != ky:
                    return False
                for k in reversed(ky):
                    pending.append((x[k], y[k]))
            elif type(y) is list:
                xid, yid = id(x), id(y)
                for ax, by in amap:
                    if ax == xid and by != yid:
                        return False
                for by, ax in bmap:
                    if by == yid and ax != xid:
                        return False
                pair = (xid, yid)
                if pair in seen:
                    continue
                seen.append(pair)
                amap.append((xid, yid))
                bmap.append((yid, xid))
                if len(x) != len(y):
                    return False
                for i in range(len(y) - 1, -1, -1):
                    pending.append((x[i], y[i]))
            elif type(y) not in (str, bool, int, type(None)) or not _scalar(x, type(y), y):
                return False
        return True
    except Exception:
        return False


def _contract_rows() -> list[dict[str, Any]]:
    rows = []
    for _, cid, mid, iid, req, cat in SPECS:
        rows.append(
            {
                "contract_id": cid,
                "matrix_id": mid,
                "inventory_id": iid,
                "requirement_field": req,
                "category": cat,
                "required": True,
                "source_matrix_state": "blocked",
                "source_blocker_code": "environment_evidence_not_collected",
                "source_evidence_required": True,
                "source_evidence_collected": False,
                "source_evidence_validated": False,
                "acceptance_rule": "requires_collected_and_validated_environment_evidence",
                "satisfaction_rule": "requires_satisfied_requirement_and_validated_evidence",
                "current_contract_state": "blocked",
                "requirement_satisfied": False,
                "build_gate_open": False,
                "source_only_definition": True,
            }
        )
    return rows


def _summary_source() -> dict[str, Any]:
    return {
        "contract_row_count": 11,
        "required_count": 11,
        "satisfied_count": 0,
        "unsatisfied_count": 11,
        "open_gate_count": 0,
        "closed_gate_count": 11,
        "evidence_required_count": 11,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "environment_contract_definition_complete": True,
        "environment_build_ready": False,
    }


def _scope_source() -> dict[str, Any]:
    return {
        "contract_defined": True,
        "source_only_contract": True,
        "contract_row_count": 11,
        "source_matrix_row_count": 11,
        "one_to_one_matrix_mapping": True,
        "environment_observed": False,
        "requirements_satisfied": False,
        "all_contract_gates_open": False,
        "environment_build_ready": False,
        "actual_windows_host_inspected": False,
        "actual_dependencies_resolved": False,
        "actual_pyside_imported": False,
        "actual_qml_loaded": False,
        "actual_qt_plugins_discovered": False,
        "build_command_created": False,
        "build_command_executed": False,
    }


def _auth_source() -> dict[str, Any]:
    d = {
        "environment_read_model_definition_authorized": True,
        "only_source_only_19_4_handoff_allowed": True,
    }
    for f in (
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
        "authorization_granted_by_19_3",
    ):
        d[f] = False
    return d


def _evidence_fields() -> tuple[str, ...]:
    return (
        "repo_scan_performed",
        "filesystem_scan_performed",
        "environment_scan_performed",
        "windows_host_inspected",
        "windows_version_collected",
        "python_version_collected",
        "pyside_version_collected",
        "packaging_tool_selected",
        "dependency_resolution_performed",
        "pyside_import_performed",
        "qml_load_performed",
        "qt_plugin_discovery_performed",
        "build_command_created",
        "build_command_executed",
        "packaging_performed",
        "artifact_created",
        "artifact_scanned",
        "artifact_signed",
        "installer_created",
        "release_performed",
        "runtime_started",
        "network_opened",
        "credentials_read",
        "orders_enabled",
    )


def _evidence_source() -> dict[str, Any]:
    d = {"source_read": True, "environment_contract_definition_built": True}
    d.update({f: False for f in _evidence_fields()})
    return d


def _bound_source() -> dict[str, Any]:
    d = {
        "reads_19_2_only": True,
        "source_only": True,
        "plain_data": True,
        "static_contract": True,
        "can_feed_only_19_4_windows_build_environment_read_model": True,
        "one_to_one_source_mapping_required": True,
    }
    for f in (
        "repo_scan",
        "filesystem_scan",
        "environment_scan",
        "windows_host_inspection",
        "dependency_resolution",
        "pyside_import",
        "qml_load",
        "qt_plugin_discovery",
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
    ):
        d[f] = False
    return d


def _trusted_source_stub() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_q_windows_build_environment_contract_kind": SOURCE_KIND,
        "block": "Q",
        "step": "19.3",
        "block_q_windows_build_environment_contract_status": SOURCE_STATUS,
        "source_19_2_accepted": True,
        "block_q_windows_build_environment_contract_decision": SOURCE_STATUS.upper(),
        "environment_contract_artifact_complete": True,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_4": True,
        "next_step": "FUNCTIONAL-PREVIEW-19.4",
        "next_step_title": "WINDOWS BUILD ENVIRONMENT READ MODEL",
        "block_q_19_2_matrix_reference": {
            "schema_version": "preview_block_q_windows_build_environment_matrix.v1",
            "kind": "functional_preview_block_q_windows_build_environment_matrix",
            "block": "Q",
            "step": "19.2",
            "status": "source_19_1_accepted_11_row_environment_matrix_defined_0_requirements_satisfied_11_requirements_blocked_environment_not_observed_source_only_handoff_to_19_3_no_build_no_packaging_no_artifact_no_runtime_no_orders",
            "source_19_1_accepted": True,
            "environment_matrix_artifact_complete": True,
            "environment_observation_complete": False,
            "environment_build_ready": False,
            "ready_for_block_q_3": True,
            "integrity_valid": True,
            "source_top_level_fields": [
                "schema_version",
                "block_q_windows_build_environment_matrix_kind",
                "block",
                "step",
                "block_q_windows_build_environment_matrix_status",
                "source_19_1_accepted",
                "block_q_windows_build_environment_matrix_decision",
                "environment_matrix_artifact_complete",
                "environment_observation_complete",
                "environment_build_ready",
                "ready_for_block_q_3",
                "next_step",
                "next_step_title",
                "block_q_19_1_inventory_reference",
                "source_inventory_preservation",
                "environment_matrix_scope",
                "environment_matrix_rows",
                "environment_matrix_summary",
                "build_execution_authorization_state",
                "non_execution_matrix_evidence",
                "matrix_boundaries",
                "source_boundaries",
                "future_steps",
                "status",
                "integrity_valid",
            ],
            "source_matrix_row_count": 11,
        },
        "source_matrix_preservation": {
            "preserves_19_2_payload": False,
            "preserves_all_11_matrix_rows": True,
            "preserves_matrix_order": True,
            "preserves_one_to_one_mapping": True,
            "preserves_blocked_matrix_state": True,
            "preserves_zero_environment_evidence": True,
            "preserves_zero_execution_authorizations": True,
            "preserves_source_only_handoff": True,
            "source_matrix_modified": False,
            "source_matrix_reinterpreted": False,
        },
        "environment_contract_scope": _scope_source(),
        "environment_contract_rows": _contract_rows(),
        "environment_contract_summary": _summary_source(),
        "build_execution_authorization_state": _auth_source(),
        "non_execution_contract_evidence": _evidence_source(),
        "contract_boundaries": _bound_source(),
        "source_boundaries": {
            "source_step": "19.2",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "block_p_builders_read": False,
            "block_q_19_0_builder_read": False,
            "block_q_19_1_builder_read": False,
            "other_block_q_builders_read": False,
        },
        "future_steps": [
            {
                "next_step": "FUNCTIONAL-PREVIEW-19.4",
                "next_step_title": "WINDOWS BUILD ENVIRONMENT READ MODEL",
                "source_only": True,
                "environment_observation_performed": False,
                "physical_build_performed": False,
            }
        ],
        "status": SOURCE_STATUS,
        "integrity_valid": True,
    }


def _source_accepted(s: Any) -> bool:
    try:
        if not _exact_dict(s):
            return False
        if tuple(s) != SOURCE_19_3_TOP_LEVEL_FIELDS:
            return False
        for k, t, v in (
            ("schema_version", str, SOURCE_SCHEMA_VERSION),
            ("block_q_windows_build_environment_contract_kind", str, SOURCE_KIND),
            ("block", str, "Q"),
            ("step", str, "19.3"),
            ("block_q_windows_build_environment_contract_status", str, SOURCE_STATUS),
            ("source_19_2_accepted", bool, True),
            ("block_q_windows_build_environment_contract_decision", str, SOURCE_STATUS.upper()),
            ("environment_contract_artifact_complete", bool, True),
            ("environment_observation_complete", bool, False),
            ("environment_build_ready", bool, False),
            ("ready_for_block_q_4", bool, True),
            ("next_step", str, "FUNCTIONAL-PREVIEW-19.4"),
            ("next_step_title", str, "WINDOWS BUILD ENVIRONMENT READ MODEL"),
            ("status", str, SOURCE_STATUS),
            ("integrity_valid", bool, True),
        ):
            if not _scalar(s[k], t, v):
                return False
        rows = s["environment_contract_rows"]
        _rows(rows)
        if not _exact_plain(s["environment_contract_summary"], _summary_source()):
            return False
        if not _exact_plain(s["environment_contract_scope"], _scope_source()):
            return False
        if not _exact_plain(s["build_execution_authorization_state"], _auth_source()):
            return False
        if not _exact_plain(s["non_execution_contract_evidence"], _evidence_source()):
            return False
        if not _exact_plain(s["contract_boundaries"], _bound_source()):
            return False
        if not _exact_plain(s["source_boundaries"], _trusted_source_stub()["source_boundaries"]):
            return False
        if not _exact_plain(s["future_steps"], _trusted_source_stub()["future_steps"]):
            return False
        if not _exact_plain(
            s["block_q_19_2_matrix_reference"],
            _trusted_source_stub()["block_q_19_2_matrix_reference"],
        ):
            return False
        if not _exact_plain(
            s["source_matrix_preservation"], _trusted_source_stub()["source_matrix_preservation"]
        ):
            return False
        return _integrity_19_3(s) is True
    except Exception:
        return False


def _rows(source_rows: Any) -> list[dict[str, Any]]:
    if type(source_rows) is not list or len(source_rows) != len(SPECS):
        raise ValueError("invalid contract rows")
    out = []
    for index in range(len(SPECS)):
        row = source_rows[index]
        if not _exact_dict(row) or tuple(row) != CONTRACT_ROW_FIELDS:
            raise ValueError("invalid contract row")
        rid, cid, mid, iid, req, cat = SPECS[index]
        for k, t, v in (
            ("contract_id", str, cid),
            ("matrix_id", str, mid),
            ("inventory_id", str, iid),
            ("requirement_field", str, req),
            ("category", str, cat),
            ("required", bool, True),
            ("source_matrix_state", str, "blocked"),
            ("source_blocker_code", str, "environment_evidence_not_collected"),
            ("source_evidence_required", bool, True),
            ("source_evidence_collected", bool, False),
            ("source_evidence_validated", bool, False),
            ("acceptance_rule", str, "requires_collected_and_validated_environment_evidence"),
            ("satisfaction_rule", str, "requires_satisfied_requirement_and_validated_evidence"),
            ("current_contract_state", str, "blocked"),
            ("requirement_satisfied", bool, False),
            ("build_gate_open", bool, False),
            ("source_only_definition", bool, True),
        ):
            if not _scalar(row[k], t, v):
                raise ValueError("contract row mismatch")
        out.append(
            {
                "read_model_id": rid,
                "contract_id": row["contract_id"],
                "matrix_id": row["matrix_id"],
                "inventory_id": row["inventory_id"],
                "requirement_field": row["requirement_field"],
                "category": row["category"],
                "required": row["required"],
                "source_contract_state": row["current_contract_state"],
                "source_requirement_satisfied": row["requirement_satisfied"],
                "source_build_gate_open": row["build_gate_open"],
                "read_required": row["required"],
                "read_status": "pending",
                "observed_value": "",
                "evidence_collected": False,
                "evidence_validated": False,
                "read_model_state": row["current_contract_state"],
                "source_only_definition": row["source_only_definition"],
            }
        )
    return out


def _summary(n: bool) -> dict[str, Any]:
    c = 11 if n else 0
    return {
        "read_model_row_count": c,
        "required_count": c,
        "unread_pending_blocked_count": c,
        "read_complete_count": 0,
        "evidence_required_count": c,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "read_model_definition_complete": n,
        "environment_build_ready": False,
    }


def _scope(n: bool) -> dict[str, Any]:
    return {
        "read_model_defined": n,
        "source_only_definition": n,
        "read_results_included": False,
        "actual_environment_observed": False,
        "environment_build_ready": False,
    }


def _auth(n: bool) -> dict[str, Any]:
    d = {f + "_authorized": False for f in EXECUTION_FALSE_FIELDS}
    d["source_only_next_step_authorized"] = False
    d["next_step_contract_missing"] = True
    return d


def _evidence(n: bool) -> dict[str, Any]:
    d = {
        "source_read": n,
        "read_model_definition_built": n,
        "environment_evidence_collected": False,
        "environment_evidence_validated": False,
    }
    d.update({f + "_performed": False for f in EXECUTION_FALSE_FIELDS})
    return d


def _bound(n: bool) -> dict[str, Any]:
    d = {
        "reads_19_3_only": n,
        "source_only": n,
        "plain_data": n,
        "next_step_contract_missing": True,
    }
    d.update({f: False for f in EXECUTION_FALSE_FIELDS})
    return d


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    p = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_read_model_kind": KIND,
        "block": "Q",
        "step": "19.4",
        "block_q_windows_build_environment_read_model_status": STATUS,
        "source_19_3_accepted": True,
        "block_q_windows_build_environment_read_model_decision": DECISION,
        "environment_read_model_artifact_complete": True,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_5": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_3_contract_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_q_windows_build_environment_contract_kind"],
            "block": "Q",
            "step": "19.3",
            "status": source["status"],
            "source_top_level_fields": list(SOURCE_19_3_TOP_LEVEL_FIELDS),
            "source_contract_row_count": 11,
            "integrity_valid": True,
        },
        "source_contract_preservation": {
            "preserves_19_3_payload": False,
            "preserves_all_11_contract_rows": True,
            "preserves_contract_order": True,
            "preserves_one_to_one_mapping": True,
            "source_contract_modified": False,
        },
        "environment_read_model_scope": _scope(True),
        "environment_read_model_rows": _rows(source["environment_contract_rows"]),
        "environment_read_model_summary": _summary(True),
        "build_execution_authorization_state": _auth(True),
        "non_execution_read_model_evidence": _evidence(True),
        "read_model_boundaries": _bound(True),
        "source_boundaries": {
            "source_step": "19.3",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "only_public_19_3_builder_read": True,
            "block_p_builders_read": False,
            "earlier_block_q_builders_read": False,
        },
        "future_steps": [],
        "status": STATUS,
        "integrity_valid": True,
    }
    return {k: p[k] for k in TOP_LEVEL_FIELDS}


def _blocked() -> dict[str, Any]:
    p = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_read_model_kind": KIND,
        "block": "Q",
        "step": "19.4",
        "block_q_windows_build_environment_read_model_status": BLOCKED_STATUS,
        "source_19_3_accepted": False,
        "block_q_windows_build_environment_read_model_decision": BLOCKED_DECISION,
        "environment_read_model_artifact_complete": False,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_5": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_3_contract_reference": {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.3",
            "status": "",
            "source_top_level_fields": [],
            "source_contract_row_count": 0,
            "integrity_valid": False,
        },
        "source_contract_preservation": {
            "preserves_19_3_payload": False,
            "preserves_all_11_contract_rows": False,
            "preserves_contract_order": False,
            "preserves_one_to_one_mapping": False,
            "source_contract_modified": False,
        },
        "environment_read_model_scope": _scope(False),
        "environment_read_model_rows": [],
        "environment_read_model_summary": _summary(False),
        "build_execution_authorization_state": _auth(False),
        "non_execution_read_model_evidence": _evidence(False),
        "read_model_boundaries": _bound(False),
        "source_boundaries": {
            "source_step": "19.3",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "only_public_19_3_builder_read": True,
            "block_p_builders_read": False,
            "earlier_block_q_builders_read": False,
        },
        "future_steps": [],
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }
    return {k: p[k] for k in TOP_LEVEL_FIELDS}


def _canonical_nominal() -> dict[str, Any]:
    return _nominal(_trusted_source_stub())


def _canonical_blocked() -> dict[str, Any]:
    return _blocked()


def _integrity(payload: Any) -> bool:
    try:
        return _exact_plain(payload, _canonical_blocked()) or _exact_plain(
            payload, _canonical_nominal()
        )
    except Exception:
        return False


def build_preview_block_q_windows_build_environment_read_model() -> dict[str, Any]:
    try:
        source = build_preview_block_q_windows_build_environment_contract()
    except Exception:
        return _blocked()
    try:
        if not _source_accepted(source):
            return _blocked()
        return _nominal(source)
    except Exception:
        return _blocked()
