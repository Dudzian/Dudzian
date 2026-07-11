from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_safety_gate_readiness_contract import (
    build_preview_block_n_safety_gate_readiness_contract,
)
from ui.pyside_app.preview_block_n_safety_gate_readiness_matrix import (
    build_preview_block_n_safety_gate_readiness_matrix,
)

MODULE_PATH = Path("ui/pyside_app/preview_block_n_safety_gate_readiness_contract.py")
UNCHANGED_PATHS = [
    Path("ui/pyside_app/preview_block_n_safety_gate_readiness_matrix.py"),
    Path("tests/ui_pyside/test_preview_block_n_safety_gate_readiness_matrix.py"),
    Path("tests/test_local_gateway_validation.py"),
    Path("tests/ui_pyside/test_source_smoke.py"),
    Path("ui/pyside_app/qml/MainWindow.qml"),
]
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_contract_status",
    "block_n_safety_gate_readiness_contract_decision",
    "ready_for_block_n_7",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_matrix_reference",
    "readiness_contract_summary",
    "packaging_release_readiness_contract_rows",
    "runtime_safety_readiness_contract_rows",
    "cross_domain_invariant_readiness_contract_rows",
    "validation_requirement_contract_rows",
    "domain_readiness_contract_summary",
    "exe_direction_readiness_contract",
    "fail_closed_readiness_contract_decision",
    "non_execution_contract_evidence",
    "readiness_contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
CAPABILITY_ROW_FIELDS = [
    "contract_row_id",
    "source_readiness_row_id",
    "source_read_row_id",
    "source_contract_id",
    "source_gate_id",
    "capability_id",
    "domain",
    "display_name",
    "source_matrix_result",
    "source_static_readiness_classification",
    "contract_required_requirements",
    "contract_satisfied_requirements",
    "contract_missing_requirements",
    "contract_requirements_total",
    "contract_requirements_satisfied_count",
    "contract_requirements_missing_count",
    "contract_requirements_complete",
    "contract_readiness_classification",
    "contract_ready_for_execution",
    "contract_execution_authorized",
    "contract_gate_open_now",
    "contract_execution_allowed_now",
    "contract_execution_performed_now",
    "contract_requires_future_explicit_gate",
    "contract_failure_policy",
    "contract_result",
    "notes",
]
INVARIANT_ROW_FIELDS = [
    "contract_row_id",
    "source_readiness_row_id",
    "source_read_row_id",
    "source_contract_id",
    "invariant_id",
    "domain",
    "display_name",
    "source_matrix_result",
    "source_static_readiness_classification",
    "source_invariant_preserved",
    "contract_invariant_preserved",
    "contract_invariant_required_for_future_execution",
    "contract_execution_gate_open_now",
    "contract_execution_allowed_now",
    "contract_execution_performed_now",
    "contract_requires_future_explicit_gate",
    "contract_readiness_classification",
    "contract_failure_policy",
    "contract_result",
    "notes",
]
REQUIREMENT_ROW_FIELDS = [
    "contract_row_id",
    "requirement_id",
    "display_name",
    "source_required",
    "source_present",
    "source_completed",
    "source_satisfied",
    "contract_required",
    "contract_present",
    "contract_completed",
    "contract_satisfied",
    "applicable_domains",
    "contract_missing_blocks_execution",
    "contract_requires_future_explicit_step",
    "contract_failure_policy",
    "contract_result",
    "notes",
]
PACKAGING_IDS = [
    "packaging_dry_run_execution",
    "packaging_execution",
    "pyinstaller_execution",
    "build_command_execution",
    "build_artifact_creation",
    "dependency_freeze",
    "asset_discovery",
    "qml_asset_discovery",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke_test",
    "artifact_signing",
    "artifact_publishing",
    "release_execution",
    "release_publish",
    "release_signing",
    "release_smoke_test",
    "release_notes_generation",
    "release_tag_creation",
    "release_upload",
    "release_external_export",
]
RUNTIME_IDS = [
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gate_execution",
    "gate_state_mutation",
    "order_generation",
    "order_" + "sub" + "mission",
    "order_" + "can" + "cel",
    "order_" + "re" + "place",
    "private_endpoints",
    "network_io",
    "credential_read",
    "filesystem_io",
    "config_env_secrets",
    "qml_bridge",
]
INVARIANT_IDS = [
    "block_m_closure_preserved",
    "block_n_entry_preserved",
    "exe_direction_preserved_without_execution",
    "no_live_credentials_embedded",
    "no_network_required_for_static_matrix",
    "runtime_disabled_during_packaging_and_release",
    "operator_confirmation_required_before_execution",
    "artifact_validation_required_before_release",
    "release_rollback_policy_required",
    "release_publication_requires_future_explicit_gate",
    "packaging_environment_validation_deferred",
    "filesystem_side_effects_forbidden_in_16_2",
]
REQUIREMENT_DOMAINS = {
    "operator_confirmation": ["packaging_release", "runtime_safety"],
    "environment_validation": ["packaging_release"],
    "artifact_validation": ["packaging_release"],
    "release_validation": ["packaging_release"],
    "runtime_validation": ["runtime_safety"],
    "credentials_validation": ["runtime_safety"],
    "future_explicit_gate": ["packaging_release", "runtime_safety", "cross_domain"],
}

REFERENCE_FALSE_ROOTS = [
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
    "readiness_recalculated_from_environment",
    "execution_authorized",
    "operator_confirmation_accepted",
    "environment_validation_performed",
    "artifact_validation_performed",
    "release_validation_performed",
    "runtime_validation_performed",
    "credentials_validation_performed",
    "dependency_validation_performed",
    "future_explicit_gate_opened",
    "packaging_dry_run_executed",
    "packaging_executed",
    "pyinstaller_started",
    "build_command_executed",
    "build_artifact_created",
    "artifact_created",
    "artifact_mutated",
    "artifact_deleted",
    "artifact_smoke_tested",
    "artifact_signed",
    "artifact_published",
    "release_executed",
    "release_published",
    "release_signed",
    "release_smoke_tested",
    "release_notes_generated",
    "release_tag_created",
    "release_uploaded",
    "release_external_export",
    "runtime_activated",
    "paper_runtime_started",
    "testnet_runtime_started",
    "live_canary_started",
    "live_trading_started",
    "runtime_loop_started",
    "runtime_gate_executed",
    "order_activity_enabled",
    "private_endpoint_accessed",
    "network_io_opened",
    "credentials_read",
    "config_env_secrets_read",
    "filesystem_io_performed",
    "qml_bridge_changed",
    "installer_changed",
    "workflow_changed",
]
EXPECTED_REFERENCE_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_matrix_status",
    "block_n_safety_gate_readiness_matrix_decision",
    "ready_for_block_n_6",
    "next_step",
    "next_step_title",
    "source_block_n_safety_gate_readiness_matrix_step",
    "source_block_n_safety_gate_readiness_matrix_read_by_16_6",
    "block_n_safety_gate_readiness_matrix_available_before_contract",
    "static_block_n_safety_gate_readiness_matrix_only",
    "block_n_safety_gate_readiness_contract_built_by_16_6",
    "ready_for_functional_preview_16_7",
    *[root + "_by_16_6" for root in REFERENCE_FALSE_ROOTS],
]
SUMMARY_TRUE_KEYS = {
    "block_n_safety_gate_readiness_matrix_available",
    "block_n_safety_gate_readiness_contract_built",
    "block_n_opened",
    "ready_for_block_n_7",
    "ready_for_functional_preview_16_7",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "readiness_contract_source_only",
    "readiness_contract_plain_data_only",
    "readiness_contract_static_only",
    "readiness_contract_read_only",
    "readiness_contract_non_evaluating",
    "readiness_contract_non_mutating",
    "readiness_contract_non_authorizing",
    "all_capabilities_contracted",
    "all_requirements_contracted",
    "all_invariants_contracted",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_execution_capabilities_require_future_explicit_gate",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_preserved",
    "packaging_release_readiness_contract_rows_built",
    "runtime_safety_readiness_contract_rows_built",
    "cross_domain_invariant_readiness_contract_rows_built",
    "validation_requirement_contract_rows_built",
    "domain_readiness_contract_summary_built",
    "missing_confirmation_blocks_execution",
    "missing_validation_blocks_execution",
    "missing_future_explicit_gate_blocks_execution",
}
SUMMARY_FALSE_KEYS = {
    "any_gate_evaluated_now",
    "any_gate_condition_met_now",
    "any_gate_open_now",
    "any_gate_state_mutated_now",
    "any_readiness_recalculated_from_environment_now",
    "any_execution_authorized_now",
    "any_execution_allowed_now",
    "any_execution_performed_now",
    "any_validation_completed_now",
    "any_requirement_present_now",
    "any_requirement_satisfied_now",
    "any_capability_ready_now",
    "operator_confirmation_present_now",
    "environment_validation_present_now",
    "artifact_validation_present_now",
    "release_validation_present_now",
    "runtime_validation_present_now",
    "credentials_validation_present_now",
    "dependency_validation_present_now",
    "future_explicit_gate_present_now",
    "packaging_release_domain_ready_now",
    "runtime_safety_domain_ready_now",
    "exe_build_ready_now",
    "exe_packaging_ready_now",
    "exe_release_ready_now",
}
EVIDENCE_TRUE_KEYS = {
    "source_block_n_safety_gate_readiness_matrix_read",
    "block_n_safety_gate_readiness_contract_built",
    "block_n_safety_gate_readiness_contract_only",
    "block_n_opened",
    "ready_for_block_n_7",
    "all_capability_rows_contracted",
    "all_capability_rows_not_ready",
    "all_invariant_rows_preserved",
    "all_requirement_rows_missing",
    "all_execution_authorization_false",
    "all_capabilities_fail_closed",
}
EVIDENCE_FALSE_KEYS = {
    "gate_evaluation_performed",
    "gate_condition_accepted",
    "gate_opened",
    "gate_mutated",
    "readiness_recalculated_from_environment",
    "confirmation_accepted",
    "validation_performed",
    "authorization_performed",
    "execution_performed",
    "packaging_performed",
    "build_performed",
    "release_performed",
    "runtime_performed",
    "orders_performed",
    "network_io_performed",
    "filesystem_io_performed",
    "private_endpoint_accessed",
    "credentials_read",
    "config_env_secrets_read",
}


def _contract() -> dict:
    return build_preview_block_n_safety_gate_readiness_contract()


def _matrix() -> dict:
    return build_preview_block_n_safety_gate_readiness_matrix()


def _source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_json_serializable_top_level_identity_status_and_reference() -> None:
    contract = _contract()
    json.dumps(contract, sort_keys=True)
    assert list(contract) == TOP_LEVEL_FIELDS
    assert contract["schema_version"] == "preview_block_n_safety_gate_readiness_contract.v1"
    assert contract["block_n_safety_gate_readiness_contract_kind"] == (
        "functional_preview_block_n_safety_gate_readiness_contract"
    )
    assert contract["block"] == "N"
    assert contract["step"] == "16.6"
    assert "readiness_contract_ready" in contract["block_n_safety_gate_readiness_contract_status"]
    assert (
        "16_5_readiness_matrix_consumed"
        in contract["block_n_safety_gate_readiness_contract_status"]
    )
    assert (
        "ALL_EXECUTION_CAPABILITIES_BLOCKED"
        in contract["block_n_safety_gate_readiness_contract_decision"]
    )
    assert contract["ready_for_block_n_7"] is True
    assert contract["next_step"] == "FUNCTIONAL-PREVIEW-16.7"
    assert contract["next_step_title"] == "BLOCK N SAFETY GATE READINESS READ MODEL"
    assert contract["status"] == (
        "ready_for_functional_preview_16_7_block_n_safety_gate_readiness_read_model"
    )
    reference = contract["block_n_safety_gate_readiness_matrix_reference"]
    assert list(reference) == EXPECTED_REFERENCE_FIELDS
    assert reference["source_block_n_safety_gate_readiness_matrix_step"] == (
        "FUNCTIONAL-PREVIEW-16.5"
    )
    assert reference["source_block_n_safety_gate_readiness_matrix_read_by_16_6"] is True
    assert reference["block_n_safety_gate_readiness_contract_built_by_16_6"] is True
    assert reference["ready_for_functional_preview_16_7"] is True
    for root in REFERENCE_FALSE_ROOTS:
        assert reference[root + "_by_16_6"] is False


def test_summary_source_only_static_fail_closed_not_ready() -> None:
    summary = _contract()["readiness_contract_summary"]
    assert set(summary) == SUMMARY_TRUE_KEYS | SUMMARY_FALSE_KEYS
    for key in SUMMARY_TRUE_KEYS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_KEYS:
        assert summary[key] is False


def test_capability_contract_rows_preserve_16_5_lineage_and_contract_fail_closed() -> None:
    contract = _contract()
    matrix = _matrix()
    row_sets = [
        (
            contract["packaging_release_readiness_contract_rows"],
            matrix["packaging_release_readiness_rows"],
            PACKAGING_IDS,
            "packaging_release",
        ),
        (
            contract["runtime_safety_readiness_contract_rows"],
            matrix["runtime_safety_readiness_rows"],
            RUNTIME_IDS,
            "runtime_safety",
        ),
    ]
    for rows, source_rows, expected_ids, domain in row_sets:
        source_rows_by_id = {row["readiness_row_id"]: row for row in source_rows}
        assert [row["capability_id"] for row in rows] == expected_ids
        for row in rows:
            source = source_rows_by_id[row["source_readiness_row_id"]]
            assert list(row) == CAPABILITY_ROW_FIELDS
            assert row["contract_row_id"] == row["source_readiness_row_id"] + "_contract"
            assert row["source_read_row_id"] == source["source_read_row_id"]
            assert row["source_contract_id"] == source["source_contract_id"]
            assert row["source_gate_id"] == source["source_gate_id"]
            assert row["capability_id"] == source["capability_id"]
            assert row["domain"] == domain
            assert row["display_name"] == source["display_name"]
            assert row["source_matrix_result"] == source["matrix_result"]
            assert (
                row["source_static_readiness_classification"]
                == source["static_readiness_classification"]
            )
            assert row["contract_required_requirements"] == source["required_requirements"]
            assert row["contract_satisfied_requirements"] == []
            assert row["contract_missing_requirements"] == row["contract_required_requirements"]
            assert row["contract_requirements_total"] == 4
            assert row["contract_requirements_satisfied_count"] == 0
            assert row["contract_requirements_missing_count"] == 4
            assert row["contract_requirements_complete"] is False
            assert row["contract_readiness_classification"] == "not_ready"
            assert row["contract_ready_for_execution"] is False
            assert row["contract_execution_authorized"] is False
            assert row["contract_gate_open_now"] is False
            assert row["contract_execution_allowed_now"] is False
            assert row["contract_execution_performed_now"] is False
            assert row["contract_requires_future_explicit_gate"] is True
            assert row["contract_failure_policy"] == "fail_closed"
            assert row["contract_result"] == (
                "contracted_not_ready_missing_requirements_execution_blocked"
            )


def test_invariant_contract_rows_preserve_16_5_lineage_and_full_semantics() -> None:
    contract = _contract()
    matrix = _matrix()
    invariants = contract["cross_domain_invariant_readiness_contract_rows"]
    source_by_id = {
        row["readiness_row_id"]: row for row in matrix["cross_domain_invariant_readiness_rows"]
    }
    assert [row["invariant_id"] for row in invariants] == INVARIANT_IDS
    for row in invariants:
        source = source_by_id[row["source_readiness_row_id"]]
        assert list(row) == INVARIANT_ROW_FIELDS
        assert row["contract_row_id"] == row["source_readiness_row_id"] + "_contract"
        assert row["source_read_row_id"] == source["source_read_row_id"]
        assert row["source_contract_id"] == source["source_contract_id"]
        assert row["invariant_id"] == source["invariant_id"]
        assert row["domain"] == "cross_domain"
        assert row["display_name"] == source["display_name"]
        assert row["source_matrix_result"] == source["matrix_result"]
        assert (
            row["source_static_readiness_classification"]
            == source["static_readiness_classification"]
        )
        assert row["source_invariant_preserved"] == source["source_invariant_preserved"]
        assert row["contract_invariant_preserved"] is True
        assert row["contract_invariant_required_for_future_execution"] is True
        assert row["contract_execution_gate_open_now"] is False
        assert row["contract_execution_allowed_now"] is False
        assert row["contract_execution_performed_now"] is False
        assert row["contract_requires_future_explicit_gate"] is True
        assert row["contract_readiness_classification"] == "invariant_preserved_execution_not_ready"
        assert row["contract_failure_policy"] == "fail_closed"
        assert row["contract_result"] == "contracted_invariant_preserved_execution_blocked"


def test_requirement_contract_rows_preserve_16_5_lineage_and_full_semantics() -> None:
    contract = _contract()
    matrix = _matrix()
    requirements = contract["validation_requirement_contract_rows"]
    source_by_id = {row["requirement_id"]: row for row in matrix["validation_requirement_rows"]}
    assert [row["requirement_id"] for row in requirements] == list(REQUIREMENT_DOMAINS)
    for row in requirements:
        source = source_by_id[row["requirement_id"]]
        assert list(row) == REQUIREMENT_ROW_FIELDS
        assert row["contract_row_id"] == row["requirement_id"] + "_readiness_contract"
        assert row["display_name"] == source["display_name"]
        assert row["source_required"] == source["required"]
        assert row["source_present"] == source["present"]
        assert row["source_completed"] == source["completed"]
        assert row["source_satisfied"] == source["satisfied"]
        assert row["contract_required"] is True
        assert row["contract_present"] is False
        assert row["contract_completed"] is False
        assert row["contract_satisfied"] is False
        assert row["applicable_domains"] == source["applicable_domains"]
        assert row["applicable_domains"] == REQUIREMENT_DOMAINS[row["requirement_id"]]
        assert row["contract_missing_blocks_execution"] is True
        assert row["contract_requires_future_explicit_step"] is True
        assert row["contract_failure_policy"] == "fail_closed"
        assert row["contract_result"] == "contracted_missing_execution_blocked"


def test_domain_summaries_are_complete_and_fail_closed() -> None:
    summary = _contract()["domain_readiness_contract_summary"]
    expected_required = {
        "packaging_release": [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "release_validation",
            "future_explicit_gate",
        ],
        "runtime_safety": [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ],
    }
    expected_counts = {"packaging_release": 22, "runtime_safety": 18}
    for domain, capability_count in expected_counts.items():
        row = summary[domain]
        assert row["domain"] == domain
        assert row["capability_count"] == capability_count
        assert row["contracted_capability_count"] == row["capability_count"]
        assert row["ready_capability_count"] == 0
        assert row["blocked_capability_count"] == row["capability_count"]
        assert row["required_requirement_ids"] == expected_required[domain]
        assert row["satisfied_requirement_ids"] == []
        assert row["missing_requirement_ids"] == row["required_requirement_ids"]
        assert row["requirements_complete"] is False
        assert row["domain_ready"] is False
        assert row["execution_authorized"] is False
        assert row["all_capabilities_contracted"] is True
        assert row["all_capabilities_not_ready"] is True
        assert row["all_capabilities_blocked"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["contract_result"] == "contracted_not_ready_execution_blocked"
    overall = summary["overall"]
    assert overall["total_capability_count"] == 40
    assert overall["contracted_capability_count"] == 40
    assert overall["ready_capability_count"] == 0
    assert overall["blocked_capability_count"] == 40
    assert overall["all_domains_contracted"] is True
    assert overall["all_domains_ready"] is False
    assert overall["all_capabilities_not_ready"] is True
    assert overall["all_capabilities_blocked"] is True
    assert overall["execution_authorized"] is False
    assert overall["failure_policy"] == "fail_closed"
    assert overall["contract_result"] == "contracted_not_ready_execution_blocked"


def test_exe_direction_preserves_16_5_source_and_adds_16_6_contract_fields() -> None:
    matrix = _matrix()
    contract = _contract()
    source_exe = matrix["exe_direction_readiness_matrix"]
    contract_exe = contract["exe_direction_readiness_contract"]
    for key, value in source_exe.items():
        assert contract_exe[key] == value
    assert contract_exe["block_n_safety_gate_readiness_contract_confirms_exe_direction"] is True
    assert contract_exe["readiness_matrix_source_preserved"] is True
    assert contract_exe["build_readiness_classification"] == "not_ready"
    assert contract_exe["packaging_readiness_classification"] == "not_ready"
    assert contract_exe["release_readiness_classification"] == "not_ready"
    assert contract_exe["ready_to_build_exe_now"] is False
    assert contract_exe["ready_to_package_exe_now"] is False
    assert contract_exe["ready_to_release_exe_now"] is False
    assert contract_exe["build_authorized_now"] is False
    assert contract_exe["packaging_authorized_now"] is False
    assert contract_exe["release_authorized_now"] is False
    assert contract_exe["future_packaging_gate_required"] is True
    assert contract_exe["future_release_gate_required"] is True
    assert contract_exe["future_explicit_step_required"] is True
    assert contract_exe["failure_policy"] == "fail_closed"
    assert contract_exe["contract_result"] == "exe_direction_contracted_execution_not_ready"


def test_fail_closed_decision_evidence_and_boundaries_are_complete() -> None:
    contract = _contract()
    decision = contract["fail_closed_readiness_contract_decision"]
    assert all(value == "fail_closed" for key, value in decision.items() if key.endswith("_policy"))
    assert decision["block_n_safety_gate_readiness_contract_in_16_6"] == "ready"
    assert decision["block_n_safety_gate_readiness_read_model_in_16_7"] == "allowed"
    assert decision["only_source_only_16_7_handoff_allowed"] is True
    assert set(decision["real_capability_status"].values()) == {"blocked"}
    assert decision["real_capability_status"]["order_" + "sub" + "mission"] == "blocked"
    assert decision["real_capability_status"]["order_" + "can" + "cel"] == "blocked"
    assert decision["real_capability_status"]["order_" + "re" + "place"] == "blocked"

    evidence = contract["non_execution_contract_evidence"]
    assert set(evidence) == EVIDENCE_TRUE_KEYS | EVIDENCE_FALSE_KEYS
    for key in EVIDENCE_TRUE_KEYS:
        assert evidence[key] is True
    for key in EVIDENCE_FALSE_KEYS:
        assert evidence[key] is False

    assert all(value is True for value in contract["readiness_contract_boundaries"].values())
    source_boundaries = contract["source_boundaries"]
    assert source_boundaries["allowed_imports_only"] is True
    assert source_boundaries["source_block_n_safety_gate_readiness_matrix"] == (
        "FUNCTIONAL-PREVIEW-16.5"
    )
    assert source_boundaries["source_block_n_safety_gate_read_model"] == "FUNCTIONAL-PREVIEW-16.4"
    nested = source_boundaries["source_block_n_safety_gate_readiness_matrix_boundaries"]
    assert nested["allowed_imports_only"] is True
    assert nested["plain_data_source_only"] is True
    assert nested["static_non_evaluating"] is True
    assert nested["non_mutating"] is True
    assert nested["non_authorizing"] is True
    assert nested["can_feed_16_6"] is True
    assert all(
        value is False for key, value in source_boundaries.items() if key.startswith("forbidden_")
    )


def test_import_ast_literal_and_side_effect_guards() -> None:
    tree = ast.parse(_source())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    calls = []
    strings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            calls.append(func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", ""))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            strings.append(node.value)
    assert len(imports) == 3
    assert all(isinstance(node, ast.ImportFrom) for node in imports)
    assert [(node.module, [alias.name for alias in node.names]) for node in imports] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        (
            "ui.pyside_app.preview_block_n_safety_gate_readiness_matrix",
            ["build_preview_block_n_safety_gate_readiness_matrix"],
        ),
    ]
    forbidden_calls = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "subprocess",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "getaddrinfo",
        "create_connection",
        "activate",
        "start",
        "execute",
        "mutate",
        "evaluate",
        "validate",
        "confirm",
        "authorize",
        "TradingController",
        "DecisionEnvelope",
        "PyInstaller",
        "packaging",
        "build",
        "release",
    }
    assert not (forbidden_calls & set(calls))
    assert calls.count("build_preview_block_n_safety_gate_readiness_matrix") == 1
    forbidden_literals = {
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    }
    assert not (forbidden_literals & set(strings))


def test_required_protected_scope_files_exist() -> None:
    for path in UNCHANGED_PATHS:
        assert path.exists()
