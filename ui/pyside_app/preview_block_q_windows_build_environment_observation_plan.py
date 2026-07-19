"""FUNCTIONAL-PREVIEW-19.5 Block Q Windows build observation plan."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_q_windows_build_environment_read_model import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_4,
    build_preview_block_q_windows_build_environment_read_model,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_build_environment_observation_plan.v1"
KIND: Final[str] = "functional_preview_block_q_windows_build_environment_observation_plan"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.5"
TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT OBSERVATION PLAN"
NEXT_STEP: Final[str] = ""
NEXT_STEP_TITLE: Final[str] = ""
STATUS: Final[str] = (
    "source_19_4_accepted_11_row_observation_plan_defined_0_observations_authorized_0_observations_performed_0_evidence_"
    "environment_build_not_ready_source_only_handoff_blocked_next_step_contract_missing_no_build_no_packaging_no_artifact_no_runtime_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_5_source_19_4_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()

SOURCE_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
READ_ROW_FIELDS: Final[tuple[str, ...]] = (
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
ROW_FIELDS: Final[tuple[str, ...]] = (
    "observation_plan_id",
    "read_model_id",
    "contract_id",
    "matrix_id",
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "observation_target",
    "observation_method",
    "expected_value_type",
    "evidence_required",
    "observation_authorized",
    "observation_status",
    "evidence_status",
    "plan_state",
    "source_only_definition",
)
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
    "schema_version",
    "block_q_windows_build_environment_observation_plan_kind",
    "block",
    "step",
    "title",
    "block_q_windows_build_environment_observation_plan_status",
    "source_19_4_accepted",
    "block_q_windows_build_environment_observation_plan_decision",
    "observation_plan_artifact_complete",
    "actual_environment_observation_performed",
    "environment_build_ready",
    "ready_for_block_q_6",
    "next_step",
    "next_step_title",
    "next_step_contract_missing",
    "block_q_19_4_read_model_reference",
    "source_read_model_preservation",
    "observation_plan_scope",
    "observation_plan_rows",
    "observation_plan_summary",
    "observation_authorization_state",
    "non_execution_observation_plan_evidence",
    "observation_plan_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
    "integrity_valid",
)
SPECS: Final[tuple[tuple[str, str, str, str], ...]] = (
    (
        "observation_plan_windows_host",
        "windows_host_identity",
        "windows_platform_identity_query",
        "string",
    ),
    (
        "observation_plan_supported_windows_version",
        "windows_version",
        "windows_version_query",
        "string",
    ),
    (
        "observation_plan_python_version",
        "python_interpreter_version",
        "python_version_query",
        "string",
    ),
    (
        "observation_plan_pyside_version",
        "pyside_package_version",
        "python_package_metadata_query",
        "string",
    ),
    (
        "observation_plan_packaging_tool_and_version",
        "packaging_tool_and_version",
        "packaging_configuration_query",
        "string",
    ),
    (
        "observation_plan_desktop_entrypoint",
        "desktop_entrypoint",
        "repository_configuration_read",
        "string",
    ),
    ("observation_plan_qml_assets", "qml_asset_manifest", "repository_manifest_read", "list"),
    ("observation_plan_qt_plugins", "qt_plugin_requirements", "repository_manifest_read", "list"),
    ("observation_plan_dependency_lock", "dependency_lock_state", "dependency_lock_read", "object"),
    (
        "observation_plan_secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_policy",
        "repository_policy_read",
        "object",
    ),
    (
        "observation_plan_output_name_and_version_policy",
        "output_name_and_version_policy",
        "repository_policy_read",
        "object",
    ),
)
READ_IDS: Final[tuple[tuple[str, str, str, str, str, str], ...]] = (
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
FORBIDDEN: Final[tuple[str, ...]] = (
    "repo_runtime_scan",
    "filesystem_runtime_scan",
    "environment_scan",
    "windows_host_inspection",
    "windows_version_read",
    "interpreter_observation",
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
    "application_runtime",
    "network",
    "credentials",
    "orders",
)
AUTHORIZATION_FIELDS: Final[tuple[str, ...]] = (
    "repo_runtime_scan_authorized",
    "filesystem_runtime_scan_authorized",
    "environment_scan_authorized",
    "windows_host_inspection_authorized",
    "windows_version_read_authorized",
    "interpreter_observation_authorized",
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
    "application_runtime_authorized",
    "network_open_authorized",
    "credentials_read_authorized",
    "orders_authorized",
)


def _scalar(value: Any, typ: type, expected: Any) -> bool:
    if type(value) is not typ:
        return False
    if typ is bool:
        return value is expected
    return value == expected


def _exact_plain(a: Any, b: Any) -> bool:
    try:
        pending = [(a, b)]
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
                if (xid, yid) in seen:
                    continue
                seen.append((xid, yid))
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
                if (xid, yid) in seen:
                    continue
                seen.append((xid, yid))
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


def _source_rows() -> list[dict[str, Any]]:
    rows = []
    for rid, cid, mid, iid, req, cat in READ_IDS:
        rows.append(
            {
                "read_model_id": rid,
                "contract_id": cid,
                "matrix_id": mid,
                "inventory_id": iid,
                "requirement_field": req,
                "category": cat,
                "required": True,
                "source_contract_state": "blocked",
                "source_requirement_satisfied": False,
                "source_build_gate_open": False,
                "read_required": True,
                "read_status": "pending",
                "observed_value": "",
                "evidence_collected": False,
                "evidence_validated": False,
                "read_model_state": "blocked",
                "source_only_definition": True,
            }
        )
    return rows


def _trusted_source_19_4() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_q_windows_build_environment_read_model_kind": SOURCE_KIND,
        "block": "Q",
        "step": "19.4",
        "block_q_windows_build_environment_read_model_status": SOURCE_STATUS,
        "source_19_3_accepted": True,
        "block_q_windows_build_environment_read_model_decision": SOURCE_STATUS.upper(),
        "environment_read_model_artifact_complete": True,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_5": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_3_contract_reference": {
            "schema_version": "preview_block_q_windows_build_environment_contract.v1",
            "kind": "functional_preview_block_q_windows_build_environment_contract",
            "block": "Q",
            "step": "19.3",
            "status": "source_19_2_accepted_11_row_environment_contract_defined_0_requirements_satisfied_11_contract_gates_closed_environment_evidence_not_collected_environment_build_not_ready_source_only_handoff_to_19_4_no_build_no_packaging_no_artifact_no_runtime_no_orders",
            "source_top_level_fields": [
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
            ],
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
        "environment_read_model_scope": {
            "read_model_defined": True,
            "source_only_definition": True,
            "read_results_included": False,
            "actual_environment_observed": False,
            "environment_build_ready": False,
        },
        "environment_read_model_rows": _source_rows(),
        "environment_read_model_summary": {
            "read_model_row_count": 11,
            "required_count": 11,
            "unread_pending_blocked_count": 11,
            "read_complete_count": 0,
            "evidence_required_count": 11,
            "evidence_collected_count": 0,
            "evidence_validated_count": 0,
            "environment_observation_complete": False,
            "read_model_definition_complete": True,
            "environment_build_ready": False,
        },
        "build_execution_authorization_state": _auth19(),
        "non_execution_read_model_evidence": _evidence19(True),
        "read_model_boundaries": _bound19(),
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
        "status": SOURCE_STATUS,
        "integrity_valid": True,
    }


def _auth19() -> dict[str, Any]:
    names = (
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
    d = {f + "_authorized": False for f in names}
    d["source_only_next_step_authorized"] = False
    d["next_step_contract_missing"] = True
    return d


def _evidence19(n: bool) -> dict[str, Any]:
    d = {
        "source_read": n,
        "read_model_definition_built": n,
        "environment_evidence_collected": False,
        "environment_evidence_validated": False,
    }
    for f in (
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
    ):
        d[f + "_performed"] = False
    return d


def _bound19() -> dict[str, Any]:
    d = {
        "reads_19_3_only": True,
        "source_only": True,
        "plain_data": True,
        "next_step_contract_missing": True,
    }
    for f in (
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
    ):
        d[f] = False
    return d


def _source_accepted(source: Any) -> bool:
    if not _exact_plain(source, _trusted_source_19_4()):
        return False
    return _integrity_19_4(source) is True


def _rows(source_rows: Any) -> list[dict[str, Any]]:
    if type(source_rows) is not list or len(source_rows) != len(SPECS):
        raise ValueError("invalid source rows")
    out = []
    for i in range(len(SPECS)):
        row = source_rows[i]
        if (
            type(row) is not dict
            or not all(type(k) is str for k in list(row))
            or tuple(row) != READ_ROW_FIELDS
        ):
            raise ValueError("invalid source row")
        rid, cid, mid, iid, req, cat = READ_IDS[i]
        for k, t, v in (
            ("read_model_id", str, rid),
            ("contract_id", str, cid),
            ("matrix_id", str, mid),
            ("inventory_id", str, iid),
            ("requirement_field", str, req),
            ("category", str, cat),
            ("required", bool, True),
            ("read_status", str, "pending"),
            ("observed_value", str, ""),
            ("evidence_collected", bool, False),
            ("evidence_validated", bool, False),
            ("read_model_state", str, "blocked"),
            ("source_only_definition", bool, True),
        ):
            if not _scalar(row[k], t, v):
                raise ValueError("row mismatch")
        pid, target, method, value_type = SPECS[i]
        out.append(
            {
                "observation_plan_id": pid,
                "read_model_id": row["read_model_id"],
                "contract_id": row["contract_id"],
                "matrix_id": row["matrix_id"],
                "inventory_id": row["inventory_id"],
                "requirement_field": row["requirement_field"],
                "category": row["category"],
                "required": row["required"],
                "observation_target": target,
                "observation_method": method,
                "expected_value_type": value_type,
                "evidence_required": True,
                "observation_authorized": False,
                "observation_status": "not_performed",
                "evidence_status": "not_collected",
                "plan_state": "defined_but_not_authorized",
                "source_only_definition": row["source_only_definition"],
            }
        )
    return out


def _summary(n: bool) -> dict[str, Any]:
    c = 11 if n else 0
    return {
        "observation_plan_row_count": c,
        "required_count": c,
        "authorized_observation_count": 0,
        "performed_observation_count": 0,
        "evidence_required_count": c,
        "evidence_collected_count": 0,
        "plan_definition_complete": n,
        "actual_environment_observation_complete": False,
        "environment_build_ready": False,
    }


def _auth(n: bool) -> dict[str, Any]:
    d = {f: False for f in AUTHORIZATION_FIELDS}
    d["observation_plan_definition_authorized"] = n
    d["next_step_authorized"] = False
    d["next_step_contract_missing"] = True
    return d


def _evidence(n: bool) -> dict[str, Any]:
    d = {
        "source_read": n,
        "observation_plan_definition_built": n,
        "environment_evidence_collected": False,
        "environment_evidence_validated": False,
    }
    d.update({f + "_performed": False for f in FORBIDDEN})
    return d


def _bound(n: bool) -> dict[str, Any]:
    d = {
        "reads_19_4_only": n,
        "one_to_one_read_model_mapping_required": n,
        "plain_data": n,
        "source_only": n,
        "no_runtime_execution": True,
        "no_environment_mutation": True,
        "no_source_mutation": True,
        "no_network": True,
        "no_credentials": True,
        "no_orders": True,
    }
    d.update({f: False for f in FORBIDDEN})
    return d


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    p = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_observation_plan_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "title": TITLE,
        "block_q_windows_build_environment_observation_plan_status": STATUS,
        "source_19_4_accepted": True,
        "block_q_windows_build_environment_observation_plan_decision": DECISION,
        "observation_plan_artifact_complete": True,
        "actual_environment_observation_performed": False,
        "environment_build_ready": False,
        "ready_for_block_q_6": False,
        "next_step": "",
        "next_step_title": "",
        "next_step_contract_missing": True,
        "block_q_19_4_read_model_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_q_windows_build_environment_read_model_kind"],
            "block": "Q",
            "step": "19.4",
            "status": source["status"],
            "source_top_level_fields": list(SOURCE_TOP_LEVEL_FIELDS),
            "source_read_model_row_count": 11,
            "integrity_valid": True,
        },
        "source_read_model_preservation": {
            "preserves_19_4_payload": False,
            "preserves_all_11_read_model_rows": True,
            "preserves_read_model_order": True,
            "preserves_one_to_one_mapping": True,
            "source_read_model_modified": False,
        },
        "observation_plan_scope": {
            "observation_plan_defined": True,
            "source_only_definition": True,
            "actual_environment_observed": False,
            "observation_results_included": False,
            "environment_build_ready": False,
        },
        "observation_plan_rows": _rows(source["environment_read_model_rows"]),
        "observation_plan_summary": _summary(True),
        "observation_authorization_state": _auth(True),
        "non_execution_observation_plan_evidence": _evidence(True),
        "observation_plan_boundaries": _bound(True),
        "source_boundaries": {
            "source_step": "19.4",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "only_public_19_4_builder_read": True,
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
        "block_q_windows_build_environment_observation_plan_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "title": TITLE,
        "block_q_windows_build_environment_observation_plan_status": BLOCKED_STATUS,
        "source_19_4_accepted": False,
        "block_q_windows_build_environment_observation_plan_decision": BLOCKED_DECISION,
        "observation_plan_artifact_complete": False,
        "actual_environment_observation_performed": False,
        "environment_build_ready": False,
        "ready_for_block_q_6": False,
        "next_step": "",
        "next_step_title": "",
        "next_step_contract_missing": True,
        "block_q_19_4_read_model_reference": {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.4",
            "status": "",
            "source_top_level_fields": [],
            "source_read_model_row_count": 0,
            "integrity_valid": False,
        },
        "source_read_model_preservation": {
            "preserves_19_4_payload": False,
            "preserves_all_11_read_model_rows": False,
            "preserves_read_model_order": False,
            "preserves_one_to_one_mapping": False,
            "source_read_model_modified": False,
        },
        "observation_plan_scope": {
            "observation_plan_defined": False,
            "source_only_definition": False,
            "actual_environment_observed": False,
            "observation_results_included": False,
            "environment_build_ready": False,
        },
        "observation_plan_rows": [],
        "observation_plan_summary": _summary(False),
        "observation_authorization_state": _auth(False),
        "non_execution_observation_plan_evidence": _evidence(False),
        "observation_plan_boundaries": _bound(False),
        "source_boundaries": {
            "source_step": "19.4",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "only_public_19_4_builder_read": True,
            "block_p_builders_read": False,
            "earlier_block_q_builders_read": False,
        },
        "future_steps": [],
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }
    return {k: p[k] for k in TOP_LEVEL_FIELDS}


def _canonical_nominal() -> dict[str, Any]:
    return _nominal(_trusted_source_19_4())


def _canonical_blocked() -> dict[str, Any]:
    return _blocked()


def _integrity(payload: Any) -> bool:
    return _exact_plain(payload, _canonical_nominal()) or _exact_plain(
        payload, _canonical_blocked()
    )


def build_preview_block_q_windows_build_environment_observation_plan() -> dict[str, Any]:
    try:
        source = build_preview_block_q_windows_build_environment_read_model()
    except Exception:
        return _blocked()
    try:
        if not _source_accepted(source):
            return _blocked()
        return _nominal(source)
    except Exception:
        return _blocked()
