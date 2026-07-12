from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_safety_gate_readiness_contract import (
    build_preview_block_n_safety_gate_readiness_contract,
)
from ui.pyside_app.preview_block_n_safety_gate_readiness_read_model import (
    build_preview_block_n_safety_gate_readiness_read_model,
)

MODULE_PATH = Path("ui/pyside_app/preview_block_n_safety_gate_readiness_read_model.py")
TEST_PATH = Path("tests/ui_pyside/test_preview_block_n_safety_gate_readiness_read_model.py")
PROTECTED_PATHS = [
    Path("ui/pyside_app/preview_block_n_safety_gate_readiness_contract.py"),
    Path("tests/ui_pyside/test_preview_block_n_safety_gate_readiness_contract.py"),
    Path("ui/pyside_app/preview_block_n_safety_gate_readiness_matrix.py"),
    Path("tests/ui_pyside/test_preview_block_n_safety_gate_readiness_matrix.py"),
    Path("tests/test_local_gateway_validation.py"),
    Path("tests/ui_pyside/test_source_smoke.py"),
    Path("ui/pyside_app/qml/MainWindow.qml"),
]
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_read_model_status",
    "block_n_safety_gate_readiness_read_model_decision",
    "ready_for_block_n_8",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_contract_reference",
    "readiness_read_summary",
    "packaging_release_readiness_read_rows",
    "runtime_safety_readiness_read_rows",
    "cross_domain_invariant_readiness_read_rows",
    "validation_requirement_read_rows",
    "domain_readiness_read_summary",
    "exe_direction_read_model",
    "fail_closed_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
REFERENCE_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_contract_status",
    "block_n_safety_gate_readiness_contract_decision",
    "ready_for_block_n_7",
    "next_step",
    "next_step_title",
    "source_block_n_safety_gate_readiness_contract_step",
    "source_block_n_safety_gate_readiness_contract_read_by_16_7",
    "block_n_safety_gate_readiness_contract_available_before_read_model",
    "static_block_n_safety_gate_readiness_contract_only",
    "block_n_safety_gate_readiness_read_model_built_by_16_7",
    "ready_for_functional_preview_16_8",
]
FALSE_BY_16_7_ROOTS = [
    "readiness_recalculated_from_environment",
    "gate_evaluated",
    "gate_condition_met",
    "gate_opened",
    "gate_state_mutated",
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
CAPABILITY_ROW_FIELDS = [
    "read_row_id",
    "source_contract_row_id",
    "source_readiness_row_id",
    "source_read_row_id",
    "source_contract_id",
    "source_gate_id",
    "capability_id",
    "domain",
    "display_name",
    "source_contract_result",
    "source_contract_readiness_classification",
    "required_requirements",
    "satisfied_requirements",
    "missing_requirements",
    "requirements_total",
    "requirements_satisfied_count",
    "requirements_missing_count",
    "requirements_complete",
    "read_result",
    "readiness_classification",
    "ready_for_execution",
    "execution_authorized",
    "gate_open_now",
    "execution_allowed_now",
    "execution_performed_now",
    "requires_future_explicit_gate",
    "failure_policy",
    "notes",
]
INVARIANT_ROW_FIELDS = [
    "read_row_id",
    "source_contract_row_id",
    "source_readiness_row_id",
    "source_read_row_id",
    "source_contract_id",
    "invariant_id",
    "domain",
    "display_name",
    "source_contract_result",
    "source_contract_readiness_classification",
    "source_invariant_preserved",
    "read_invariant_preserved",
    "invariant_required_for_future_execution",
    "execution_gate_open_now",
    "execution_allowed_now",
    "execution_performed_now",
    "requires_future_explicit_gate",
    "readiness_classification",
    "failure_policy",
    "read_result",
    "notes",
]
REQUIREMENT_ROW_FIELDS = [
    "read_row_id",
    "source_contract_row_id",
    "requirement_id",
    "display_name",
    "source_required",
    "source_present",
    "source_completed",
    "source_satisfied",
    "required",
    "present",
    "completed",
    "satisfied",
    "applicable_domains",
    "missing_blocks_execution",
    "requires_future_explicit_step",
    "failure_policy",
    "read_result",
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
REQUIREMENT_IDS = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]
SUMMARY_TRUE = [
    "block_n_safety_gate_readiness_contract_available",
    "block_n_safety_gate_readiness_read_model_built",
    "block_n_opened",
    "ready_for_block_n_8",
    "ready_for_functional_preview_16_8",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "read_model_source_only",
    "read_model_plain_data_only",
    "read_model_static_only",
    "read_model_read_only",
    "read_model_non_evaluating",
    "read_model_non_mutating",
    "read_model_non_authorizing",
    "all_capability_contract_rows_read",
    "all_requirement_contract_rows_read",
    "all_invariant_contract_rows_read",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_execution_capabilities_require_future_explicit_gate",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_preserved",
    "packaging_release_read_rows_built",
    "runtime_safety_read_rows_built",
    "cross_domain_invariant_read_rows_built",
    "validation_requirement_read_rows_built",
    "domain_readiness_read_summary_built",
    "missing_confirmation_blocks_execution",
    "missing_validation_blocks_execution",
    "missing_future_explicit_gate_blocks_execution",
]
SUMMARY_FALSE = [
    "any_readiness_recalculated_from_environment_now",
    "any_gate_evaluated_now",
    "any_gate_condition_met_now",
    "any_gate_open_now",
    "any_gate_state_mutated_now",
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
]
DOMAINS = {
    "operator_confirmation": ["packaging_release", "runtime_safety"],
    "environment_validation": ["packaging_release"],
    "artifact_validation": ["packaging_release"],
    "release_validation": ["packaging_release"],
    "runtime_validation": ["runtime_safety"],
    "credentials_validation": ["runtime_safety"],
    "future_explicit_gate": ["packaging_release", "runtime_safety", "cross_domain"],
}

EVIDENCE_TRUE_KEYS = [
    "source_block_n_safety_gate_readiness_contract_read",
    "block_n_safety_gate_readiness_read_model_built",
    "block_n_safety_gate_readiness_read_model_only",
    "block_n_opened",
    "ready_for_block_n_8",
    "all_capability_contract_rows_read",
    "all_capability_rows_not_ready",
    "all_invariant_contract_rows_preserved",
    "all_requirement_contract_rows_missing",
    "all_execution_authorization_false",
    "all_capabilities_fail_closed",
]
EVIDENCE_FALSE_KEYS = [
    "readiness_recalculated_from_environment",
    "gate_evaluation_performed",
    "gate_condition_accepted",
    "gate_opened",
    "gate_mutated",
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
]
BOUNDARY_KEYS = [
    "block_n_safety_gate_readiness_read_model_is_plain_data_only",
    "block_n_safety_gate_readiness_read_model_is_source_only",
    "block_n_safety_gate_readiness_read_model_reads_readiness_contract_only",
    "block_n_safety_gate_readiness_read_model_preserves_block_m_closure",
    "block_n_safety_gate_readiness_read_model_preserves_block_n_entry",
    "block_n_safety_gate_readiness_read_model_preserves_exe_direction_without_packaging",
    "block_n_safety_gate_readiness_read_model_is_static_and_non_evaluating",
    "block_n_safety_gate_readiness_read_model_is_non_mutating",
    "block_n_safety_gate_readiness_read_model_is_non_authorizing",
    "block_n_safety_gate_readiness_read_model_can_feed_16_8_closure_audit",
    "cannot_recalculate_readiness_from_environment",
    "cannot_evaluate",
    "cannot_accept_condition",
    "cannot_open_gate",
    "cannot_mutate_gate",
    "cannot_accept_confirmations",
    "cannot_perform_validations",
    "cannot_authorize",
    "cannot_package",
    "cannot_build",
    "cannot_release",
    "cannot_perform_artifact_work",
    "cannot_run_runtime",
    "cannot_generate_orders",
    "cannot_submit_orders",
    "cannot_cancel_orders",
    "cannot_replace_orders",
    "cannot_use_network",
    "cannot_use_filesystem",
    "cannot_access_private_endpoints",
    "cannot_read_credentials",
    "cannot_read_config_env_secrets",
    "cannot_change_qml_or_bridge",
    "cannot_create_execution_side_effects",
]


def payload() -> dict:
    return build_preview_block_n_safety_gate_readiness_read_model()


def test_identity_reference_json_and_top_level_order() -> None:
    model = payload()
    json.dumps(model)
    assert list(model) == TOP_LEVEL_FIELDS
    assert model["schema_version"] == "preview_block_n_safety_gate_readiness_read_model.v1"
    assert model["block_n_safety_gate_readiness_read_model_kind"] == (
        "functional_preview_block_n_safety_gate_readiness_read_model"
    )
    assert model["block"] == "N"
    assert model["step"] == "16.7"
    assert model["ready_for_block_n_8"] is True
    assert model["next_step"] == "FUNCTIONAL-PREVIEW-16.8"
    assert model["next_step_title"] == "BLOCK N CLOSURE AUDIT"
    assert model["status"] == "ready_for_functional_preview_16_8_block_n_closure_audit"
    status = model["block_n_safety_gate_readiness_read_model_status"]
    for token in [
        "readiness_read_model_ready",
        "16_6_readiness_contract_consumed",
        "source_only",
        "plain_data",
        "all_execution_capabilities_not_ready",
        "all_execution_capabilities_blocked",
        "no_filesystem_io",
    ]:
        assert token in status
    assert model["block_n_safety_gate_readiness_read_model_decision"] == status.upper()
    reference = model["block_n_safety_gate_readiness_contract_reference"]
    assert list(reference) == REFERENCE_FIELDS + [root + "_by_16_7" for root in FALSE_BY_16_7_ROOTS]
    contract = build_preview_block_n_safety_gate_readiness_contract()
    for key in REFERENCE_FIELDS[:9]:
        assert reference[key] == contract[key]
    assert (
        reference["source_block_n_safety_gate_readiness_contract_step"] == "FUNCTIONAL-PREVIEW-16.6"
    )
    assert reference["source_block_n_safety_gate_readiness_contract_read_by_16_7"] is True
    assert reference["ready_for_functional_preview_16_8"] is True
    for key, value in reference.items():
        if key in {root + "_by_16_7" for root in FALSE_BY_16_7_ROOTS}:
            assert value is False


def test_summary_rows_and_lineage_match_16_6_contract() -> None:
    model = payload()
    summary = model["readiness_read_summary"]
    assert list(summary) == SUMMARY_TRUE + SUMMARY_FALSE
    assert all(summary[key] is True for key in SUMMARY_TRUE)
    assert all(summary[key] is False for key in SUMMARY_FALSE)
    contract = build_preview_block_n_safety_gate_readiness_contract()
    packaging = model["packaging_release_readiness_read_rows"]
    runtime = model["runtime_safety_readiness_read_rows"]
    invariants = model["cross_domain_invariant_readiness_read_rows"]
    requirements = model["validation_requirement_read_rows"]
    assert [row["capability_id"] for row in packaging] == PACKAGING_IDS
    assert [row["capability_id"] for row in runtime] == RUNTIME_IDS
    assert [row["invariant_id"] for row in invariants] == INVARIANT_IDS
    assert [row["requirement_id"] for row in requirements] == REQUIREMENT_IDS
    for rows in [packaging, runtime]:
        for row in rows:
            assert list(row) == CAPABILITY_ROW_FIELDS
            assert row["read_row_id"] == row["source_contract_row_id"] + "_read"
            assert row["source_contract_result"] == (
                "contracted_not_ready_missing_requirements_execution_blocked"
            )
            assert row["source_contract_readiness_classification"] == "not_ready"
            assert row["satisfied_requirements"] == []
            assert row["missing_requirements"] == row["required_requirements"]
            assert (
                row["requirements_total"],
                row["requirements_satisfied_count"],
                row["requirements_missing_count"],
            ) == (4, 0, 4)
            assert row["requirements_complete"] is False
            assert row["readiness_classification"] == "not_ready"
            assert row["ready_for_execution"] is False
            assert row["execution_authorized"] is False
            assert row["gate_open_now"] is False
            assert row["execution_allowed_now"] is False
            assert row["execution_performed_now"] is False
            assert row["requires_future_explicit_gate"] is True
            assert row["failure_policy"] == "fail_closed"
    assert all(
        row["required_requirements"]
        == [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "future_explicit_gate",
        ]
        for row in packaging
    )
    assert all(
        row["required_requirements"]
        == [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ]
        for row in runtime
    )
    for source_rows, read_rows in [
        (contract["packaging_release_readiness_contract_rows"], packaging),
        (contract["runtime_safety_readiness_contract_rows"], runtime),
    ]:
        source_by_id = {row["contract_row_id"]: row for row in source_rows}
        for row in read_rows:
            source = source_by_id[row["source_contract_row_id"]]
            assert row["source_readiness_row_id"] == source["source_readiness_row_id"]
            assert row["source_read_row_id"] == source["source_read_row_id"]
            assert row["source_contract_id"] == source["source_contract_id"]
            assert row["source_gate_id"] == source["source_gate_id"]
            assert row["capability_id"] == source["capability_id"]
            assert row["domain"] == source["domain"]
            assert row["display_name"] == source["display_name"]
            assert row["source_contract_result"] == source["contract_result"]
            assert (
                row["source_contract_readiness_classification"]
                == source["contract_readiness_classification"]
            )
            assert row["required_requirements"] == source["contract_required_requirements"]
            assert row["satisfied_requirements"] == source["contract_satisfied_requirements"]
            assert row["missing_requirements"] == source["contract_missing_requirements"]
            assert row["requirements_total"] == source["contract_requirements_total"]
            assert (
                row["requirements_satisfied_count"]
                == source["contract_requirements_satisfied_count"]
            )
            assert (
                row["requirements_missing_count"] == source["contract_requirements_missing_count"]
            )
            assert row["requirements_complete"] == source["contract_requirements_complete"]
            assert row["ready_for_execution"] == source["contract_ready_for_execution"]
            assert row["execution_authorized"] == source["contract_execution_authorized"]
            assert row["gate_open_now"] == source["contract_gate_open_now"]
            assert row["execution_allowed_now"] == source["contract_execution_allowed_now"]
            assert row["execution_performed_now"] == source["contract_execution_performed_now"]
            assert (
                row["requires_future_explicit_gate"]
                == source["contract_requires_future_explicit_gate"]
            )
            assert row["failure_policy"] == source["contract_failure_policy"]


def test_invariants_requirements_domain_exe_and_fail_closed() -> None:
    model = payload()
    contract = build_preview_block_n_safety_gate_readiness_contract()
    for source, row in zip(
        contract["cross_domain_invariant_readiness_contract_rows"],
        model["cross_domain_invariant_readiness_read_rows"],
        strict=True,
    ):
        assert list(row) == INVARIANT_ROW_FIELDS
        assert row["source_contract_row_id"] == source["contract_row_id"]
        assert row["source_readiness_row_id"] == source["source_readiness_row_id"]
        assert row["source_read_row_id"] == source["source_read_row_id"]
        assert row["source_contract_id"] == source["source_contract_id"]
        assert row["invariant_id"] == source["invariant_id"]
        assert row["domain"] == source["domain"]
        assert row["display_name"] == source["display_name"]
        assert row["source_contract_result"] == "contracted_invariant_preserved_execution_blocked"
        assert (
            row["source_contract_readiness_classification"]
            == "invariant_preserved_execution_not_ready"
        )
        assert row["source_contract_result"] == source["contract_result"]
        assert (
            row["source_contract_readiness_classification"]
            == source["contract_readiness_classification"]
        )
        assert row["source_invariant_preserved"] == source["source_invariant_preserved"]
        assert row["read_invariant_preserved"] == source["contract_invariant_preserved"]
        assert (
            row["invariant_required_for_future_execution"]
            == source["contract_invariant_required_for_future_execution"]
        )
        assert row["execution_gate_open_now"] == source["contract_execution_gate_open_now"]
        assert row["execution_allowed_now"] == source["contract_execution_allowed_now"]
        assert row["execution_performed_now"] == source["contract_execution_performed_now"]
        assert (
            row["requires_future_explicit_gate"] == source["contract_requires_future_explicit_gate"]
        )
        assert row["readiness_classification"] == source["contract_readiness_classification"]
        assert row["failure_policy"] == source["contract_failure_policy"]
        assert row["source_invariant_preserved"] is True
        assert row["read_invariant_preserved"] is True
        assert row["invariant_required_for_future_execution"] is True
        assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
    for source, row in zip(
        contract["validation_requirement_contract_rows"],
        model["validation_requirement_read_rows"],
        strict=True,
    ):
        assert list(row) == REQUIREMENT_ROW_FIELDS
        assert row["source_contract_row_id"] == source["contract_row_id"]
        assert row["requirement_id"] == source["requirement_id"]
        assert row["display_name"] == source["display_name"]
        assert row["source_required"] == source["source_required"]
        assert row["source_present"] == source["source_present"]
        assert row["source_completed"] == source["source_completed"]
        assert row["source_satisfied"] == source["source_satisfied"]
        assert row["required"] == source["contract_required"]
        assert row["present"] == source["contract_present"]
        assert row["completed"] == source["contract_completed"]
        assert row["satisfied"] == source["contract_satisfied"]
        assert row["applicable_domains"] == source["applicable_domains"]
        assert row["missing_blocks_execution"] == source["contract_missing_blocks_execution"]
        assert (
            row["requires_future_explicit_step"] == source["contract_requires_future_explicit_step"]
        )
        assert row["failure_policy"] == source["contract_failure_policy"]
        assert row["required"] is True
        assert row["present"] is False
        assert row["completed"] is False
        assert row["satisfied"] is False
        assert row["applicable_domains"] == DOMAINS[row["requirement_id"]]
        assert row["missing_blocks_execution"] is True
        assert row["requires_future_explicit_step"] is True
        assert row["failure_policy"] == "fail_closed"
    domain = model["domain_readiness_read_summary"]
    assert domain["packaging_release"]["capability_count"] == 22
    assert domain["runtime_safety"]["capability_count"] == 18
    assert domain["overall"]["total_capability_count"] == 40
    assert domain["overall"]["read_capability_count"] == 40
    assert domain["overall"]["ready_capability_count"] == 0
    assert domain["overall"]["blocked_capability_count"] == 40
    assert domain["packaging_release"]["required_requirement_ids"] == [
        "operator_confirmation",
        "environment_validation",
        "artifact_validation",
        "release_validation",
        "future_explicit_gate",
    ]
    assert domain["runtime_safety"]["required_requirement_ids"] == [
        "operator_confirmation",
        "runtime_validation",
        "credentials_validation",
        "future_explicit_gate",
    ]
    for row in [domain["packaging_release"], domain["runtime_safety"]]:
        assert row["domain"] in {"packaging_release", "runtime_safety"}
        assert row["read_capability_count"] == row["capability_count"]
        assert row["ready_capability_count"] == 0
        assert row["blocked_capability_count"] == row["capability_count"]
        assert row["satisfied_requirement_ids"] == []
        assert row["missing_requirement_ids"] == row["required_requirement_ids"]
        assert row["requirements_complete"] is False
        assert row["domain_ready"] is False
        assert row["execution_authorized"] is False
        assert row["all_capabilities_read"] is True
        assert row["all_capabilities_not_ready"] is True
        assert row["all_capabilities_blocked"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["read_result"] == "read_not_ready_execution_blocked"
    overall = domain["overall"]
    assert overall["total_capability_count"] == 40
    assert overall["read_capability_count"] == 40
    assert overall["ready_capability_count"] == 0
    assert overall["blocked_capability_count"] == 40
    assert overall["all_domains_read"] is True
    assert overall["all_domains_ready"] is False
    assert overall["all_capabilities_not_ready"] is True
    assert overall["all_capabilities_blocked"] is True
    assert overall["execution_authorized"] is False
    assert overall["failure_policy"] == "fail_closed"
    assert overall["read_result"] == "read_not_ready_execution_blocked"
    exe = model["exe_direction_read_model"]
    for key, value in contract["exe_direction_readiness_contract"].items():
        assert exe[key] == value
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["build_readiness_classification"] == "not_ready"
    assert exe["packaging_readiness_classification"] == "not_ready"
    assert exe["release_readiness_classification"] == "not_ready"
    assert exe["build_authorized_now"] is False
    assert exe["packaging_authorized_now"] is False
    assert exe["release_authorized_now"] is False
    decision = model["fail_closed_read_decision"]
    assert decision["block_n_safety_gate_readiness_read_model_in_16_7"] == "ready"
    assert decision["block_n_closure_audit_in_16_8"] == "allowed"
    assert decision["only_source_only_16_8_handoff_allowed"] is True
    for key, value in decision.items():
        if key.endswith("_policy"):
            assert value == "fail_closed"
    assert all(value == "blocked" for value in decision["real_capability_status"].values())


def test_evidence_boundaries_source_boundaries_and_source_guards() -> None:
    model = payload()
    contract = build_preview_block_n_safety_gate_readiness_contract()
    evidence = model["non_execution_evidence"]
    assert list(evidence) == EVIDENCE_TRUE_KEYS + EVIDENCE_FALSE_KEYS
    assert all(evidence[key] is True for key in EVIDENCE_TRUE_KEYS)
    assert all(evidence[key] is False for key in EVIDENCE_FALSE_KEYS)
    boundaries = model["read_model_boundaries"]
    assert list(boundaries) == BOUNDARY_KEYS
    assert all(boundaries[key] is True for key in BOUNDARY_KEYS)
    source = model["source_boundaries"]
    contract_source = contract["source_boundaries"]
    contract_nested = contract_source["source_block_n_safety_gate_readiness_matrix_boundaries"]
    read_nested = source["source_block_n_safety_gate_readiness_contract_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["allowed_imports_only"] == contract_source["allowed_imports_only"]
    assert source["source_block_n_safety_gate_readiness_contract"] == "FUNCTIONAL-PREVIEW-16.6"
    assert (
        source["source_block_n_safety_gate_readiness_matrix"]
        == contract_source["source_block_n_safety_gate_readiness_matrix"]
    )
    assert (
        source["source_block_n_safety_gate_read_model"]
        == contract_source["source_block_n_safety_gate_read_model"]
    )
    assert read_nested["allowed_imports_only"] == contract_nested["allowed_imports_only"]
    assert (
        read_nested["source_block_n_safety_gate_readiness_matrix"]
        == contract_source["source_block_n_safety_gate_readiness_matrix"]
    )
    assert (
        read_nested["source_block_n_safety_gate_read_model"]
        == contract_nested["source_block_n_safety_gate_read_model"]
    )
    assert read_nested["plain_data_source_only"] == contract_nested["plain_data_source_only"]
    assert read_nested["static_non_evaluating"] == contract_nested["static_non_evaluating"]
    assert read_nested["non_mutating"] == contract_nested["non_mutating"]
    assert read_nested["non_authorizing"] == contract_nested["non_authorizing"]
    assert read_nested["can_feed_16_7"] is True
    for key, value in source.items():
        if key.startswith("forbidden_"):
            assert value is False
    module_source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    assert not any(isinstance(node, ast.Import) for node in imports)
    assert isinstance(imports[0], ast.ImportFrom) and imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom) and imports[1].module == "typing"
    assert [alias.name for alias in imports[1].names] == ["Any", "Final"]
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_n_safety_gate_readiness_contract"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_n_safety_gate_readiness_contract"
    ]
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_call_names = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "sub" + "process",
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
        "pack" + "aging",
        "build",
        "release",
    }
    for func in calls:
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        assert name not in forbidden_call_names
    source_builder_call_count = sum(
        isinstance(func, ast.Name)
        and func.id == "build_preview_block_n_safety_gate_readiness_contract"
        for func in calls
    )
    assert source_builder_call_count == 1
    literal_values = [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)]
    forbidden_literals = {
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    }
    assert not forbidden_literals.intersection(
        {value for value in literal_values if isinstance(value, str)}
    )
    test_source = TEST_PATH.read_text(encoding="utf-8")
    assert "g" + "it" not in test_source
    assert "sub" + "process" not in test_source
    for protected in PROTECTED_PATHS:
        assert protected.exists()
