"""FUNCTIONAL-PREVIEW-19.2 Block Q Windows build environment matrix."""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_q_windows_build_environment_inventory import (
    KIND as SOURCE_KIND,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    STATUS as SOURCE_STATUS,
    _integrity as _integrity_19_1,
    build_preview_block_q_windows_build_environment_inventory,
)

SCHEMA_VERSION: Final[str] = "preview_block_q_windows_build_environment_matrix.v1"
KIND: Final[str] = "functional_preview_block_q_windows_build_environment_matrix"
BLOCK_ID: Final[str] = "Q"
STEP_ID: Final[str] = "19.2"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-19.3"
NEXT_STEP_TITLE: Final[str] = "WINDOWS BUILD ENVIRONMENT CONTRACT"
STATUS: Final[str] = (
    "source_19_1_accepted_11_row_environment_matrix_defined_0_requirements_satisfied_"
    "11_requirements_blocked_environment_not_observed_source_only_handoff_to_19_3_"
    "no_build_no_packaging_no_artifact_no_runtime_no_orders"
)
BLOCKED_STATUS: Final[str] = "blocked_for_functional_preview_19_2_source_19_1_rejected"
DECISION: Final[str] = STATUS.upper()
BLOCKED_DECISION: Final[str] = BLOCKED_STATUS.upper()

SOURCE_19_1_TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
SOURCE_19_1_INVENTORY_SPECS: Final[tuple[tuple[str, str, str], ...]] = (
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
SOURCE_ROW_FIELDS: Final[tuple[str, ...]] = (
    "inventory_id",
    "requirement_field",
    "category",
    "required",
    "declared_by_19_0",
    "collection_status",
    "validation_status",
    "resolution_status",
    "observed_value",
    "source_only_definition",
)
TOP_LEVEL_FIELDS: Final[tuple[str, ...]] = (
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
ENVIRONMENT_MATRIX_SPECS: Final[tuple[tuple[str, str, str, str], ...]] = (
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
MATRIX_ROW_FIELDS: Final[tuple[str, ...]] = (
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
    "authorization_granted_by_19_2",
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
        if tuple(source) != SOURCE_19_1_TOP_LEVEL_FIELDS:
            return False
        if not (
            type(source["schema_version"]) is str
            and source["schema_version"] == SOURCE_SCHEMA_VERSION
            and type(source["block_q_windows_build_environment_inventory_kind"]) is str
            and source["block_q_windows_build_environment_inventory_kind"] == SOURCE_KIND
            and type(source["block"]) is str
            and source["block"] == "Q"
            and type(source["step"]) is str
            and source["step"] == "19.1"
            and type(source["status"]) is str
            and source["status"] == SOURCE_STATUS
            and type(source["block_q_windows_build_environment_inventory_status"]) is str
            and source["block_q_windows_build_environment_inventory_status"] == SOURCE_STATUS
            and source["source_19_0_accepted"] is True
            and source["environment_inventory_artifact_complete"] is True
            and source["environment_observation_complete"] is False
            and source["ready_for_block_q_2"] is True
            and type(source["next_step"]) is str
            and source["next_step"] == "FUNCTIONAL-PREVIEW-19.2"
            and type(source["next_step_title"]) is str
            and source["next_step_title"] == "WINDOWS BUILD ENVIRONMENT MATRIX"
            and source["integrity_valid"] is True
        ):
            return False
        rows = source["environment_inventory_rows"]
        summary = source["environment_inventory_summary"]
        scope = source["environment_inventory_scope"]
        auth = source["build_execution_authorization_state"]
        evidence = source["non_execution_inventory_evidence"]
        boundaries = source["inventory_boundaries"]
        future_steps = source["future_steps"]
        if not (type(rows) is list and len(rows) == 11):
            return False
        seen_ids: list[str] = []
        seen_req: list[str] = []
        for row, spec in zip(rows, SOURCE_19_1_INVENTORY_SPECS):
            if not _exact_str_keyed_dict(row) or tuple(row) != SOURCE_ROW_FIELDS:
                return False
            expected_inventory_id, expected_requirement, expected_category = spec
            inventory_id = row["inventory_id"]
            requirement_field = row["requirement_field"]
            category = row["category"]
            required = row["required"]
            declared_by_19_0 = row["declared_by_19_0"]
            collection_status = row["collection_status"]
            validation_status = row["validation_status"]
            resolution_status = row["resolution_status"]
            observed_value = row["observed_value"]
            source_only_definition = row["source_only_definition"]
            if type(inventory_id) is not str or inventory_id != expected_inventory_id:
                return False
            if type(requirement_field) is not str or requirement_field != expected_requirement:
                return False
            if type(category) is not str or category != expected_category:
                return False
            if type(required) is not bool or required is not True:
                return False
            if type(declared_by_19_0) is not bool or declared_by_19_0 is not True:
                return False
            if type(collection_status) is not str or collection_status != "not_collected":
                return False
            if type(validation_status) is not str or validation_status != "not_validated":
                return False
            if type(resolution_status) is not str or resolution_status != "blocked":
                return False
            if type(observed_value) is not str or observed_value != "":
                return False
            if type(source_only_definition) is not bool or source_only_definition is not True:
                return False
            if inventory_id in seen_ids or requirement_field in seen_req:
                return False
            seen_ids.append(inventory_id)
            seen_req.append(requirement_field)
        if not all(_exact_str_keyed_dict(x) for x in (summary, scope, auth, evidence, boundaries)):
            return False
        if not _exact_plain(
            summary,
            {
                "inventory_row_count": 11,
                "required_count": 11,
                "collected_count": 0,
                "validated_count": 0,
                "resolved_count": 0,
                "blocked_count": 11,
                "environment_observation_complete": False,
                "inventory_definition_complete": True,
            },
        ):
            return False
        for k, expected in (
            ("inventory_defined", True),
            ("source_only_inventory", True),
            ("inventory_row_count", 11),
        ):
            value = scope.get(k)
            if type(value) is not type(expected):
                return False
            if type(expected) is bool:
                if value is not expected:
                    return False
            elif value != expected:
                return False
        for k, v in scope.items():
            if k.startswith("actual_") and v is not False:
                return False
        if (
            auth.get("environment_matrix_definition_authorized") is not True
            or auth.get("only_source_only_19_2_handoff_allowed") is not True
        ):
            return False
        for k, v in auth.items():
            if (
                k
                not in (
                    "environment_matrix_definition_authorized",
                    "only_source_only_19_2_handoff_allowed",
                )
                and v is not False
            ):
                return False
        if (
            evidence.get("source_read") is not True
            or evidence.get("environment_inventory_definition_built") is not True
        ):
            return False
        for k, v in evidence.items():
            if (
                k not in ("source_read", "environment_inventory_definition_built")
                and v is not False
            ):
                return False
        for k in (
            "reads_19_0_only",
            "source_only",
            "plain_data",
            "static_inventory",
            "can_feed_only_19_2_windows_build_environment_matrix",
        ):
            if boundaries.get(k) is not True:
                return False
        for k, v in boundaries.items():
            if (
                k
                not in (
                    "reads_19_0_only",
                    "source_only",
                    "plain_data",
                    "static_inventory",
                    "can_feed_only_19_2_windows_build_environment_matrix",
                )
                and v is not False
            ):
                return False
        if len(future_steps) != 1 or not _exact_str_keyed_dict(future_steps[0]):
            return False
        if not _exact_plain(
            future_steps[0],
            {
                "next_step": "FUNCTIONAL-PREVIEW-19.2",
                "next_step_title": "WINDOWS BUILD ENVIRONMENT MATRIX",
                "source_only": True,
                "environment_scan_performed": False,
                "physical_build_performed": False,
            },
        ):
            return False
        return _integrity_19_1(source) is True
    except Exception:
        return False


def _environment_matrix_rows(source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "matrix_id": matrix_id,
            "inventory_id": inventory_id,
            "requirement_field": requirement_field,
            "category": category,
            "required": True,
            "source_declared_by_19_0": True,
            "source_collection_status": source_rows[index]["collection_status"],
            "source_validation_status": source_rows[index]["validation_status"],
            "source_resolution_status": source_rows[index]["resolution_status"],
            "source_observed_value": source_rows[index]["observed_value"],
            "requirement_satisfied": False,
            "matrix_state": "blocked",
            "blocker_code": "environment_evidence_not_collected",
            "evidence_required": True,
            "evidence_collected": False,
            "evidence_validated": False,
            "source_only_definition": True,
        }
        for index, (matrix_id, inventory_id, requirement_field, category) in enumerate(
            ENVIRONMENT_MATRIX_SPECS
        )
    ]


def _summary(nominal: bool) -> dict[str, Any]:
    count = len(ENVIRONMENT_MATRIX_SPECS) if nominal else 0
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


def _authorization(nominal: bool) -> dict[str, Any]:
    data = {
        "environment_contract_definition_authorized": nominal,
        "only_source_only_19_3_handoff_allowed": nominal,
    }
    data.update({field: False for field in AUTHORIZATION_FALSE_FIELDS})
    return data


def _evidence(nominal: bool) -> dict[str, Any]:
    fields = (
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
    data = {"source_read": nominal, "environment_matrix_definition_built": nominal}
    data.update({field: False for field in fields})
    return data


def _matrix_boundaries(nominal: bool) -> dict[str, Any]:
    positive = (
        "reads_19_1_only",
        "source_only",
        "plain_data",
        "static_matrix",
        "can_feed_only_19_3_windows_build_environment_contract",
        "one_to_one_source_mapping_required",
    )
    negative = (
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
    data = {field: nominal for field in positive}
    data.update({field: False for field in negative})
    return data


def _scope(nominal: bool) -> dict[str, Any]:
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


def _trusted_source_inventory_rows() -> list[dict[str, Any]]:
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
        for inventory_id, requirement_field, category in SOURCE_19_1_INVENTORY_SPECS
    ]


def _trusted_source_stub() -> dict[str, Any]:
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "block_q_windows_build_environment_inventory_kind": SOURCE_KIND,
        "status": SOURCE_STATUS,
        "environment_inventory_rows": _trusted_source_inventory_rows(),
    }


def _nominal(source: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "block_q_windows_build_environment_matrix_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_matrix_status": STATUS,
        "source_19_1_accepted": True,
        "block_q_windows_build_environment_matrix_decision": DECISION,
        "environment_matrix_artifact_complete": True,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_3": True,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_q_19_1_inventory_reference": {
            "schema_version": source["schema_version"],
            "kind": source["block_q_windows_build_environment_inventory_kind"],
            "block": "Q",
            "step": "19.1",
            "status": source["status"],
            "source_19_0_accepted": True,
            "environment_inventory_artifact_complete": True,
            "environment_observation_complete": False,
            "ready_for_block_q_2": True,
            "integrity_valid": True,
            "source_top_level_fields": list(SOURCE_19_1_TOP_LEVEL_FIELDS),
            "source_inventory_row_count": 11,
        },
        "source_inventory_preservation": {
            "preserves_19_1_payload": True,
            "preserves_all_11_inventory_rows": True,
            "preserves_inventory_order": True,
            "preserves_inventory_requirements": True,
            "preserves_zero_observations": True,
            "preserves_zero_execution_authorizations": True,
            "preserves_source_only_handoff": True,
            "source_inventory_modified": False,
            "source_inventory_reinterpreted": False,
        },
        "environment_matrix_scope": _scope(True),
        "environment_matrix_rows": _environment_matrix_rows(source["environment_inventory_rows"]),
        "environment_matrix_summary": _summary(True),
        "build_execution_authorization_state": _authorization(True),
        "non_execution_matrix_evidence": _evidence(True),
        "matrix_boundaries": _matrix_boundaries(True),
        "source_boundaries": {
            "source_step": "19.1",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "block_p_builders_read": False,
            "block_q_19_0_builder_read": False,
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
        "block_q_windows_build_environment_matrix_kind": KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_q_windows_build_environment_matrix_status": BLOCKED_STATUS,
        "source_19_1_accepted": False,
        "block_q_windows_build_environment_matrix_decision": BLOCKED_DECISION,
        "environment_matrix_artifact_complete": False,
        "environment_observation_complete": False,
        "environment_build_ready": False,
        "ready_for_block_q_3": False,
        "next_step": "",
        "next_step_title": "",
        "block_q_19_1_inventory_reference": {
            "schema_version": "",
            "kind": "",
            "block": "Q",
            "step": "19.1",
            "status": "",
            "source_19_0_accepted": False,
            "environment_inventory_artifact_complete": False,
            "environment_observation_complete": False,
            "ready_for_block_q_2": False,
            "integrity_valid": False,
            "source_top_level_fields": [],
            "source_inventory_row_count": 0,
        },
        "source_inventory_preservation": {
            "preserves_19_1_payload": False,
            "preserves_all_11_inventory_rows": False,
            "preserves_inventory_order": False,
            "preserves_inventory_requirements": False,
            "preserves_zero_observations": False,
            "preserves_zero_execution_authorizations": False,
            "preserves_source_only_handoff": False,
            "source_inventory_modified": False,
            "source_inventory_reinterpreted": False,
        },
        "environment_matrix_scope": _scope(False),
        "environment_matrix_rows": [],
        "environment_matrix_summary": _summary(False),
        "build_execution_authorization_state": _authorization(False),
        "non_execution_matrix_evidence": _evidence(False),
        "matrix_boundaries": _matrix_boundaries(False),
        "source_boundaries": {
            "source_step": "19.1",
            "source_block": "Q",
            "source_integrity_required": True,
            "source_mutation_allowed": False,
            "block_p_builders_read": False,
            "block_q_19_0_builder_read": False,
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


def build_preview_block_q_windows_build_environment_matrix() -> dict[str, Any]:
    try:
        source = build_preview_block_q_windows_build_environment_inventory()
    except Exception:
        return _blocked()
    try:
        return _nominal(source) if _source_accepted(source) else _blocked()
    except Exception:
        return _blocked()
