"""FUNCTIONAL-PREVIEW-19.3 Block Q Windows build environment contract."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_q_windows_build_environment_matrix import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_2,
    build_preview_block_q_windows_build_environment_matrix,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_build_environment_contract.v1"
KIND: Final[str] = "functional_preview_block_q_windows_build_environment_contract"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.3"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-19.4"
NEXT_STEP_TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT READ MODEL"
STATUS: Final[str] = (
    "source_19_2_accepted_11_row_environment_contract_defined_0_requirements_satisfied_"
    "11_contract_gates_closed_environment_evidence_not_collected_environment_build_not_ready_"
    "source_only_handoff_to_19_4_no_build_no_packaging_no_artifact_no_runtime_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_3_source_19_2_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()

SOURCE_19_2_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
)
SOURCE_19_2_MATRIX_SPECS: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("matrix_windows_host", "windows_host", "windows_host_required", "platform"),
    (
        "matrix_supported_windows_version",
        "supported_windows_version",
        "supported_windows_version_must_be_confirmed",
        "platform",
    ),
    (
        "matrix_python_version",
        "python_version",
        "exact_python_version_must_be_confirmed",
        "runtime",
    ),
    (
        "matrix_pyside_version",
        "pyside_version",
        "exact_pyside_version_must_be_confirmed",
        "framework",
    ),
    (
        "matrix_packaging_tool_and_version",
        "packaging_tool_and_version",
        "exact_packaging_tool_and_version_must_be_selected",
        "packaging",
    ),
    (
        "matrix_desktop_entrypoint",
        "desktop_entrypoint",
        "desktop_entrypoint_must_be_confirmed",
        "application",
    ),
    ("matrix_qml_assets", "qml_assets", "qml_assets_must_be_confirmed", "assets"),
    ("matrix_qt_plugins", "qt_plugins", "qt_plugins_must_be_confirmed", "plugins"),
    (
        "matrix_dependency_lock",
        "dependency_lock",
        "dependency_lock_must_be_resolved",
        "dependencies",
    ),
    (
        "matrix_secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion",
        "secret_and_local_data_exclusion_must_be_confirmed",
        "security",
    ),
    (
        "matrix_output_name_and_version_policy",
        "output_name_and_version_policy",
        "output_name_and_version_policy_must_be_confirmed",
        "output",
    ),
)
SOURCE_MATRIX_ROW_FIELDS: Final[tuple[str, ...]] = (
    "matrix_id",
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "source_declared_by_19_0",
    "source_collection_status",
    "source_validation_status",
    "source_resolution_status",
    "source_observed_value",
    "requirement_satisfied",
    "matrix_state",
    "blocker_code",
    "evidence_required",
    "evidence_collected",
    "evidence_validated",
    "source_only_definition",
)
ENVIRONMENT_CONTRACT_SPECS: Final[tuple[tuple[str, str, str, str, str], ...]] = tuple(
    ("contract" + m[6:], m, i, r, c) for m, i, r, c in SOURCE_19_2_MATRIX_SPECS
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
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
    "authorization_granted_by_19_3",
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
        actual_to_trusted: dict[int, int] = {}
        trusted_to_actual: dict[int, int] = {}
        while pending:
            actual, trusted = pending.pop()
            if type(actual) is not type(trusted):
                return False
            if type(trusted) is dict:
                actual_id = id(actual)
                trusted_id = id(trusted)
                if actual_to_trusted.get(actual_id, trusted_id) != trusted_id:
                    return False
                if trusted_to_actual.get(trusted_id, actual_id) != actual_id:
                    return False
                actual_to_trusted[actual_id] = trusted_id
                trusted_to_actual[trusted_id] = actual_id
                pair = (actual_id, trusted_id)
                if pair in visited:
                    continue
                visited.add(pair)
                keys = list(actual)
                if not all(type(k) is str for k in keys) or keys != list(trusted):
                    return False
                for key in reversed(list(trusted)):
                    pending.append((actual[key], trusted[key]))
            elif type(trusted) is list:
                actual_id = id(actual)
                trusted_id = id(trusted)
                if actual_to_trusted.get(actual_id, trusted_id) != trusted_id:
                    return False
                if trusted_to_actual.get(trusted_id, actual_id) != actual_id:
                    return False
                actual_to_trusted[actual_id] = trusted_id
                trusted_to_actual[trusted_id] = actual_id
                pair = (actual_id, trusted_id)
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


def _has_unique(items: list[str]) -> bool:
    seen: list[str] = []
    for item in items:
        if item in seen:
            return False
        seen.append(item)
    return True


def _source_accepted(source: Any) -> bool:
    try:
        if not _exact_str_keyed_dict(source) or tuple(source) != SOURCE_19_2_TOP_LEVEL_FIELDS:
            return False
        if not (
            type(source["schema_version"]) is str
            and source["schema_version"] == SOURCE_SCHEMA_VERSION
            and type(source["block_q_windows_build_environment_matrix_kind"]) is str
            and source["block_q_windows_build_environment_matrix_kind"] == SOURCE_KIND
            and type(source["block"]) is str
            and source["block"] == "Q"
            and type(source["step"]) is str
            and source["step"] == "19.2"
            and type(source["status"]) is str
            and source["status"] == SOURCE_STATUS
            and type(source["block_q_windows_build_environment_matrix_status"]) is str
            and source["block_q_windows_build_environment_matrix_status"] == SOURCE_STATUS
            and source["source_19_1_accepted"] is True
            and source["environment_matrix_artifact_complete"] is True
            and source["environment_observation_complete"] is False
            and source["environment_build_ready"] is False
            and source["ready_for_block_q_3"] is True
            and type(source["next_step"]) is str
            and source["next_step"] == "FUNCTIONAL-PREVIEW-19.3"
            and type(source["next_step_title"]) is str
            and source["next_step_title"] == "WINDOWS BUILD ENVIRONMENT CONTRACT"
            and source["integrity_valid"] is True
        ):
            return False
        rows = source["environment_matrix_rows"]
        if type(rows) is not list or len(rows) != 11:
            return False
        mids: list[str] = []
        iids: list[str] = []
        reqs: list[str] = []
        for row, spec in zip(rows, SOURCE_19_2_MATRIX_SPECS):
            if not _exact_str_keyed_dict(row) or tuple(row) != SOURCE_MATRIX_ROW_FIELDS:
                return False
            matrix_id, inventory_id, requirement_field, category = spec
            checks = (
                ("matrix_id", str, matrix_id),
                ("inventory_id", str, inventory_id),
                ("requirement_field", str, requirement_field),
                ("category", str, category),
                ("required", bool, True),
                ("source_declared_by_19_0", bool, True),
                ("source_collection_status", str, "not_collected"),
                ("source_validation_status", str, "not_validated"),
                ("source_resolution_status", str, "blocked"),
                ("source_observed_value", str, ""),
                ("requirement_satisfied", bool, False),
                ("matrix_state", str, "blocked"),
                ("blocker_code", str, "environment_evidence_not_collected"),
                ("evidence_required", bool, True),
                ("evidence_collected", bool, False),
                ("evidence_validated", bool, False),
                ("source_only_definition", bool, True),
            )
            for key, typ, expected in checks:
                if type(row[key]) is not typ:
                    return False
                if typ is bool:
                    if row[key] is not expected:
                        return False
                elif row[key] != expected:
                    return False
            mids.append(row["matrix_id"])
            iids.append(row["inventory_id"])
            reqs.append(row["requirement_field"])
        if not (_has_unique(mids) and _has_unique(iids) and _has_unique(reqs)):
            return False
        if not _exact_plain(source["environment_matrix_summary"], _source_summary(True)):
            return False
        if not _exact_plain(source["environment_matrix_scope"], _source_scope(True)):
            return False
        if not _exact_plain(
            source["build_execution_authorization_state"], _source_authorization(True)
        ):
            return False
        if not _exact_plain(source["non_execution_matrix_evidence"], _source_evidence(True)):
            return False
        if not _exact_plain(source["matrix_boundaries"], _source_boundaries(True)):
            return False
        if not _exact_plain(
            source["future_steps"],
            [
                {
                    "next_step": "FUNCTIONAL-PREVIEW-19.3",
                    "next_step_title": "WINDOWS BUILD ENVIRONMENT CONTRACT",
                    "source_only": True,
                    "environment_observation_performed": False,
                    "physical_build_performed": False,
                }
            ],
        ):
            return False
        return _integrity_19_2(source) is True
    except Exception:
        return False


def _source_summary(nominal: bool) -> dict[str, Any]:
    count = 11 if nominal else 0
    return {
        "matrix_row_count": count,
        "required_count": count,
        "satisfied_count": 0,
        "unsatisfied_count": count,
        "ready_count": 0,
        "blocked_count": count,
        "evidence_required_count": count,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "environment_matrix_definition_complete": nominal,
        "environment_build_ready": False,
    }


def _source_scope(nominal: bool) -> dict[str, Any]:
    return {
        "matrix_defined": nominal,
        "source_only_matrix": nominal,
        "matrix_row_count": 11 if nominal else 0,
        "source_inventory_row_count": 11 if nominal else 0,
        "one_to_one_inventory_mapping": nominal,
        "environment_observed": False,
        "requirements_satisfied": False,
        "build_environment_ready": False,
        "actual_windows_host_inspected": False,
        "actual_dependencies_resolved": False,
        "actual_pyside_imported": False,
        "actual_qml_loaded": False,
        "actual_qt_plugins_discovered": False,
        "build_command_created": False,
        "build_command_executed": False,
    }


def _source_authorization(nominal: bool) -> dict[str, Any]:
    data = {
        "environment_contract_definition_authorized": nominal,
        "only_source_only_19_3_handoff_allowed": nominal,
    }
    for f in (*AUTHORIZATION_FALSE_FIELDS[:-1], "authorization_granted_by_19_2"):
        data[f] = False
    return data


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


def _source_evidence(nominal: bool) -> dict[str, Any]:
    data = {"source_read": nominal, "environment_matrix_definition_built": nominal}
    data.update({f: False for f in _evidence_fields()})
    return data


def _source_boundaries(nominal: bool) -> dict[str, Any]:
    data = {
        f: nominal
        for f in (
            "reads_19_1_only",
            "source_only",
            "plain_data",
            "static_matrix",
            "can_feed_only_19_3_windows_build_environment_contract",
            "one_to_one_source_mapping_required",
        )
    }
    data.update(
        {
            f: False
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
            )
        }
    )
    return data


def _environment_contract_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    if type(source_rows) is not list or len(source_rows) != len(ENVIRONMENT_CONTRACT_SPECS):
        raise ValueError("source matrix rows do not match the 19.3 contract boundary")
    for index, (
        contract_id,
        matrix_id,
        inventory_id,
        requirement_field,
        category,
    ) in enumerate(ENVIRONMENT_CONTRACT_SPECS):
        source_row = source_rows[index]
        if (
            type(source_row) is not dict
            or type(source_row.get("matrix_id")) is not str
            or source_row["matrix_id"] != matrix_id
            or type(source_row.get("inventory_id")) is not str
            or source_row["inventory_id"] != inventory_id
            or type(source_row.get("requirement_field")) is not str
            or source_row["requirement_field"] != requirement_field
            or type(source_row.get("category")) is not str
            or source_row["category"] != category
        ):
            raise ValueError("source matrix row does not match local 19.3 contract spec")
        rows.append(
            {
                "contract_id": contract_id,
                "matrix_id": source_row["matrix_id"],
                "inventory_id": source_row["inventory_id"],
                "requirement_field": source_row["requirement_field"],
                "category": source_row["category"],
                "required": source_row["required"],
                "source_matrix_state": source_row["matrix_state"],
                "source_blocker_code": source_row["blocker_code"],
                "source_evidence_required": source_row["evidence_required"],
                "source_evidence_collected": source_row["evidence_collected"],
                "source_evidence_validated": source_row["evidence_validated"],
                "acceptance_rule": "requires_collected_and_validated_environment_evidence",
                "satisfaction_rule": "requires_satisfied_requirement_and_validated_evidence",
                "current_contract_state": "blocked",
                "requirement_satisfied": source_row["requirement_satisfied"],
                "build_gate_open": False,
                "source_only_definition": source_row["source_only_definition"],
            }
        )
    return rows


def _summary(nominal: bool) -> dict[str, Any]:
    c = 11 if nominal else 0
    return {
        "contract_row_count": c,
        "required_count": c,
        "satisfied_count": 0,
        "unsatisfied_count": c,
        "open_gate_count": 0,
        "closed_gate_count": c,
        "evidence_required_count": c,
        "evidence_collected_count": 0,
        "evidence_validated_count": 0,
        "environment_observation_complete": False,
        "environment_contract_definition_complete": nominal,
        "environment_build_ready": False,
    }


def _scope(nominal: bool) -> dict[str, Any]:
    return {
        "contract_defined": nominal,
        "source_only_contract": nominal,
        "contract_row_count": 11 if nominal else 0,
        "source_matrix_row_count": 11 if nominal else 0,
        "one_to_one_matrix_mapping": nominal,
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


def _authorization(nominal: bool) -> dict[str, Any]:
    data = {
        "environment_read_model_definition_authorized": nominal,
        "only_source_only_19_4_handoff_allowed": nominal,
    }
    data.update({f: False for f in AUTHORIZATION_FALSE_FIELDS})
    return data


def _evidence(nominal: bool) -> dict[str, Any]:
    data = {"source_read": nominal, "environment_contract_definition_built": nominal}
    data.update({f: False for f in _evidence_fields()})
    return data


def _contract_boundaries(nominal: bool) -> dict[str, Any]:
    data = {
        f: nominal
        for f in (
            "reads_19_2_only",
            "source_only",
            "plain_data",
            "static_contract",
            "can_feed_only_19_4_windows_build_environment_read_model",
            "one_to_one_source_mapping_required",
        )
    }
    data.update(
        {
            f: False
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
            )
        }
    )
    return data


def _trusted_source_matrix_rows() -> list[dict[str, Any]]:
    return [
        {
            "matrix_id": m,
            "inventory_id": i,
            "requirement_field": r,
            "category": c,
            "required": True,
            "source_declared_by_19_0": True,
            "source_collection_status": "not_collected",
            "source_validation_status": "not_validated",
            "source_resolution_status": "blocked",
            "source_observed_value": "",
            "requirement_satisfied": False,
            "matrix_state": "blocked",
            "blocker_code": "environment_evidence_not_collected",
            "evidence_required": True,
            "evidence_collected": False,
            "evidence_validated": False,
            "source_only_definition": True,
        }
        for m, i, r, c in SOURCE_19_2_MATRIX_SPECS
    ]


def _trusted_source_stub() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_q_windows_build_environment_matrix_kind": SOURCE_KIND,
        "status": SOURCE_STATUS,
        "environment_matrix_rows": _trusted_source_matrix_rows(),
    }


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_contract_status": STATUS,
        "source_19_2_accepted": True,
        "block_q_windows_build_environment_contract_decision": DECISION,
        "environment_contract_artifact_complete": True,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_4": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_q_19_2_matrix_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_q_windows_build_environment_matrix_kind"],
            "block": "Q",
            "step": "19.2",
            "status": source["status"],
            "source_19_1_accepted": True,
            "environment_matrix_artifact_complete": True,
            "environment_observation_complete": False,
            "environment_build_ready": False,
            "ready_for_block_q_3": True,
            "integrity_valid": True,
            "source_top_level_fields": list(SOURCE_19_2_TOP_LEVEL_FIELDS),
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
        "environment_contract_scope": _scope(True),
        "environment_contract_rows": _environment_contract_rows(source["environment_matrix_rows"]),
        "environment_contract_summary": _summary(True),
        "build_execution_authorization_state": _authorization(True),
        "non_execution_contract_evidence": _evidence(True),
        "contract_boundaries": _contract_boundaries(True),
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
                "next_step": NEXT_STEP,
                "next_step_title": NEXT_STEP_TITLE,
                "source_only": True,
                "environment_observation_performed": False,
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
        "block_q_windows_build_environment_contract_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_contract_status": BLOCKED_STATUS,
        "source_19_2_accepted": False,
        "block_q_windows_build_environment_contract_decision": BLOCKED_DECISION,
        "environment_contract_artifact_complete": False,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_4": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_2_matrix_reference": {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.2",
            "status": "",
            "source_19_1_accepted": False,
            "environment_matrix_artifact_complete": False,
            "environment_observation_complete": False,
            "environment_build_ready": False,
            "ready_for_block_q_3": False,
            "integrity_valid": False,
            "source_top_level_fields": [],
            "source_matrix_row_count": 0,
        },
        "source_matrix_preservation": {
            "preserves_19_2_payload": False,
            "preserves_all_11_matrix_rows": False,
            "preserves_matrix_order": False,
            "preserves_one_to_one_mapping": False,
            "preserves_blocked_matrix_state": False,
            "preserves_zero_environment_evidence": False,
            "preserves_zero_execution_authorizations": False,
            "preserves_source_only_handoff": False,
            "source_matrix_modified": False,
            "source_matrix_reinterpreted": False,
        },
        "environment_contract_scope": _scope(False),
        "environment_contract_rows": [],
        "environment_contract_summary": _summary(False),
        "build_execution_authorization_state": _authorization(False),
        "non_execution_contract_evidence": _evidence(False),
        "contract_boundaries": _contract_boundaries(False),
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
        "future_steps": [],
        "status": BLOCKED_STATUS,
        "integrity_valid": True,
    }
    return {field: payload[field] for field in TOP_LEVEL_FIELDS}


def _canonical_nominal() -> dict[str, Any]:
    return _nominal(_trusted_source_stub())


def _canonical_blocked() -> dict[str, Any]:
    return _blocked()


def _integrity(payload: Any) -> bool:
    try:
        if _exact_plain(payload, _canonical_blocked()):
            return True
    except Exception:
        pass
    try:
        return _exact_plain(payload, _canonical_nominal())
    except Exception:
        return False


def build_preview_block_q_windows_build_environment_contract() -> dict[str, Any]:
    try:
        source = build_preview_block_q_windows_build_environment_matrix()
    except Exception:
        return _blocked()
    try:
        return _nominal(source) if _source_accepted(source) else _blocked()
    except Exception:
        return _blocked()
