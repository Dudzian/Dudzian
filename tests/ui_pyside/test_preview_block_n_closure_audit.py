import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_closure_audit import build_preview_block_n_closure_audit
from ui.pyside_app.preview_block_n_safety_gate_readiness_read_model import (
    build_preview_block_n_safety_gate_readiness_read_model,
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_closure_audit_kind",
    "block",
    "step",
    "block_n_closure_audit_status",
    "block_n_closure_audit_decision",
    "block_n_closed",
    "ready_for_block_o_0",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_readiness_read_model_reference",
    "closure_audit_summary",
    "block_n_step_closure_rows",
    "packaging_release_closure_summary",
    "runtime_safety_closure_summary",
    "cross_domain_invariant_closure_summary",
    "validation_requirement_closure_summary",
    "exe_direction_closure_audit",
    "fail_closed_closure_decision",
    "non_execution_closure_evidence",
    "closure_audit_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
REFERENCE_SOURCE_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_read_model_status",
    "block_n_safety_gate_readiness_read_model_decision",
    "ready_for_block_n_8",
    "next_step",
    "next_step_title",
]
REFERENCE_HANDOFF_FIELDS = [
    "source_block_n_safety_gate_readiness_read_model_step",
    "source_block_n_safety_gate_readiness_read_model_read_by_16_8",
    "block_n_safety_gate_readiness_read_model_available_before_closure_audit",
    "static_block_n_safety_gate_readiness_read_model_only",
    "block_n_closure_audit_built_by_16_8",
    "ready_for_functional_preview_17_0",
]
FALSE_BY_16_8_FIELDS = [
    "readiness_recalculated_from_environment_by_16_8",
    "gate_evaluated_by_16_8",
    "gate_condition_met_by_16_8",
    "gate_opened_by_16_8",
    "gate_state_mutated_by_16_8",
    "execution_authorized_by_16_8",
    "operator_confirmation_accepted_by_16_8",
    "environment_validation_performed_by_16_8",
    "artifact_validation_performed_by_16_8",
    "release_validation_performed_by_16_8",
    "runtime_validation_performed_by_16_8",
    "credentials_validation_performed_by_16_8",
    "dependency_validation_performed_by_16_8",
    "future_explicit_gate_opened_by_16_8",
    "packaging_dry_run_executed_by_16_8",
    "packaging_executed_by_16_8",
    "pyinstaller_started_by_16_8",
    "build_command_executed_by_16_8",
    "build_artifact_created_by_16_8",
    "artifact_created_by_16_8",
    "artifact_mutated_by_16_8",
    "artifact_deleted_by_16_8",
    "artifact_smoke_tested_by_16_8",
    "artifact_signed_by_16_8",
    "artifact_published_by_16_8",
    "release_executed_by_16_8",
    "release_published_by_16_8",
    "release_signed_by_16_8",
    "release_smoke_tested_by_16_8",
    "release_notes_generated_by_16_8",
    "release_tag_created_by_16_8",
    "release_uploaded_by_16_8",
    "release_external_export_by_16_8",
    "runtime_activated_by_16_8",
    "paper_runtime_started_by_16_8",
    "testnet_runtime_started_by_16_8",
    "live_canary_started_by_16_8",
    "live_trading_started_by_16_8",
    "runtime_loop_started_by_16_8",
    "runtime_gate_executed_by_16_8",
    "order_activity_enabled_by_16_8",
    "private_endpoint_accessed_by_16_8",
    "network_io_opened_by_16_8",
    "credentials_read_by_16_8",
    "config_env_secrets_read_by_16_8",
    "filesystem_io_performed_by_16_8",
    "qml_bridge_changed_by_16_8",
    "installer_changed_by_16_8",
    "workflow_changed_by_16_8",
]
EXPECTED_REFERENCE_FIELDS = (
    REFERENCE_SOURCE_FIELDS + REFERENCE_HANDOFF_FIELDS + FALSE_BY_16_8_FIELDS
)
SUMMARY_TRUE_KEYS = [
    "block_n_safety_gate_readiness_read_model_available",
    "block_n_closure_audit_built",
    "block_n_opened",
    "block_n_closed",
    "ready_for_block_o_0",
    "ready_for_functional_preview_17_0",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "closure_audit_source_only",
    "closure_audit_plain_data_only",
    "closure_audit_static_only",
    "closure_audit_read_only",
    "closure_audit_non_evaluating",
    "closure_audit_non_mutating",
    "closure_audit_non_authorizing",
    "all_block_n_steps_complete",
    "all_capability_rows_read",
    "all_requirement_rows_read",
    "all_invariant_rows_read",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_preserved",
    "all_domains_not_ready",
    "all_domains_execution_unauthorized",
    "only_source_only_block_o_handoff_allowed",
]
SUMMARY_FALSE_KEYS = [
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
    "packaging_release_domain_ready_now",
    "runtime_safety_domain_ready_now",
    "exe_build_ready_now",
    "exe_packaging_ready_now",
    "exe_release_ready_now",
    "runtime_enabled_by_closure",
    "packaging_enabled_by_closure",
    "release_enabled_by_closure",
    "orders_enabled_by_closure",
]
EVIDENCE_TRUE_KEYS = [
    "source_block_n_readiness_read_model_read",
    "block_n_closure_audit_built",
    "block_n_closure_audit_only",
    "block_n_opened",
    "block_n_closed",
    "ready_for_block_o_0",
    "all_block_n_steps_complete",
    "all_capability_rows_read",
    "all_capability_rows_not_ready",
    "all_invariant_rows_preserved",
    "all_requirement_rows_missing",
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
    "real_capabilities_opened_by_closure",
]
CLOSURE_BOUNDARY_KEYS = [
    "block_n_closure_audit_is_plain_data_only",
    "block_n_closure_audit_is_source_only",
    "block_n_closure_audit_reads_16_7_only",
    "block_n_closure_audit_preserves_block_m_closure",
    "block_n_closure_audit_preserves_block_n_entry",
    "block_n_closure_audit_preserves_exe_direction_without_packaging",
    "block_n_closure_audit_is_static_and_non_evaluating",
    "block_n_closure_audit_is_non_mutating",
    "block_n_closure_audit_is_non_authorizing",
    "block_n_closure_audit_can_close_block_n",
    "block_n_closure_audit_can_feed_17_0_entry_contract",
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
DOMAIN_SOURCE_FIELDS = [
    "domain",
    "capability_count",
    "read_capability_count",
    "ready_capability_count",
    "blocked_capability_count",
    "required_requirement_ids",
    "satisfied_requirement_ids",
    "missing_requirement_ids",
    "requirements_complete",
    "domain_ready",
    "execution_authorized",
    "all_capabilities_read",
    "all_capabilities_not_ready",
    "all_capabilities_blocked",
    "failure_policy",
]

INHERITED_FORBIDDEN_SOURCE_FIELDS = [
    "forbidden_packaging_calls_present",
    "forbidden_pyinstaller_calls_present",
    "forbidden_build_calls_present",
    "forbidden_release_calls_present",
    "forbidden_runtime_calls_present",
    "forbidden_gate_evaluation_calls_present",
    "forbidden_gate_execution_calls_present",
    "forbidden_gate_mutation_calls_present",
    "forbidden_validation_calls_present",
    "forbidden_confirmation_calls_present",
    "forbidden_authorization_calls_present",
    "forbidden_readiness_recalculation_calls_present",
    "forbidden_io_calls_present",
    "forbidden_network_calls_present",
    "forbidden_private_endpoint_calls_present",
    "forbidden_ui_bridge_calls_present",
]
EXPECTED_REAL_CAPABILITY_KEYS = [
    "release_execution",
    "release_publish",
    "release_sign",
    "release_smoke",
    "release_workflow",
    "release_notes",
    "release_tag",
    "release_upload",
    "release_export",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke",
    "artifact_sign",
    "artifact_publish",
    "artifact_name",
    "artifact_location",
    "artifact_checksum",
    "artifact_metadata",
    "artifact_audit",
    "artifact_cleanup",
    "packaging_dry_run",
    "packaging",
    "pyinstaller",
    "build",
    "build_artifact",
    "installer",
    "workflow",
    "environment",
    "dependency",
    "asset",
    "qml_asset",
    "filesystem",
    "gate_evaluation",
    "gate_condition",
    "gate_opening",
    "gate_mutation",
    "confirmation_acceptance",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "dependency_validation",
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gates",
    "order_generation",
    "create_" + "order",
    "sub" + "mit_order",
    "can" + "cel_order",
    "re" + "place_order",
    "fetch" + "_balance",
    "private_endpoint",
    "network",
    "credentials",
    "config_env_secrets",
    "qml_bridge",
    "cc" + "xt",
]
STEP_DATA = [
    ("FUNCTIONAL-PREVIEW-16.0", "BLOCK N ENTRY CONTRACT", "entry_contract"),
    ("FUNCTIONAL-PREVIEW-16.1", "BLOCK N READ MODEL", "read_model"),
    ("FUNCTIONAL-PREVIEW-16.2", "BLOCK N SAFETY GATE MATRIX", "safety_gate_matrix"),
    ("FUNCTIONAL-PREVIEW-16.3", "BLOCK N SAFETY GATE CONTRACT", "safety_gate_contract"),
    ("FUNCTIONAL-PREVIEW-16.4", "BLOCK N SAFETY GATE READ MODEL", "safety_gate_read_model"),
    (
        "FUNCTIONAL-PREVIEW-16.5",
        "BLOCK N SAFETY GATE READINESS MATRIX",
        "safety_gate_readiness_matrix",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.6",
        "BLOCK N SAFETY GATE READINESS CONTRACT",
        "safety_gate_readiness_contract",
    ),
    (
        "FUNCTIONAL-PREVIEW-16.7",
        "BLOCK N SAFETY GATE READINESS READ MODEL",
        "safety_gate_readiness_read_model",
    ),
]
ROW_KEYS = [
    "closure_row_id",
    "step",
    "title",
    "artifact_kind",
    "step_complete",
    "source_only",
    "plain_data",
    "execution_authorized",
    "real_capabilities_opened",
    "closure_status",
    "closure_result",
    "notes",
]


def test_closure_audit_identity_order_and_reference():
    audit = build_preview_block_n_closure_audit()
    source = build_preview_block_n_safety_gate_readiness_read_model()
    json.dumps(audit)
    assert list(audit) == TOP_LEVEL_FIELDS
    assert audit["schema_version"] == "preview_block_n_closure_audit.v1"
    assert audit["block_n_closure_audit_kind"] == "functional_preview_block_n_closure_audit"
    assert audit["block"] == "N"
    assert audit["step"] == "16.8"
    assert audit["block_n_closed"] is True
    assert audit["ready_for_block_o_0"] is True
    assert audit["next_step"] == "FUNCTIONAL-PREVIEW-17.0"
    assert audit["next_step_title"] == "BLOCK O ENTRY CONTRACT"
    assert (
        audit["status"] == "block_n_closed_ready_for_functional_preview_17_0_block_o_entry_contract"
    )
    status = audit["block_n_closure_audit_status"]
    for token in [
        "block_n_closure_audit_complete",
        "16_7_readiness_read_model_consumed",
        "no_runtime",
        "only_source_only_block_o_handoff_allowed",
    ]:
        assert token in status
    assert audit["block_n_closure_audit_decision"] == status.upper()
    reference = audit["block_n_safety_gate_readiness_read_model_reference"]
    assert list(reference) == EXPECTED_REFERENCE_FIELDS
    for key in REFERENCE_SOURCE_FIELDS:
        assert reference[key] == source[key]
    assert (
        reference["source_block_n_safety_gate_readiness_read_model_step"]
        == "FUNCTIONAL-PREVIEW-16.7"
    )
    assert reference["source_block_n_safety_gate_readiness_read_model_read_by_16_8"] is True
    assert (
        reference["block_n_safety_gate_readiness_read_model_available_before_closure_audit"] is True
    )
    assert reference["static_block_n_safety_gate_readiness_read_model_only"] is True
    assert reference["block_n_closure_audit_built_by_16_8"] is True
    assert reference["ready_for_functional_preview_17_0"] is True
    for key in FALSE_BY_16_8_FIELDS:
        assert reference[key] is False


def test_summary_rows_domains_and_source_lineage():
    audit = build_preview_block_n_closure_audit()
    source = build_preview_block_n_safety_gate_readiness_read_model()
    summary = audit["closure_audit_summary"]
    assert list(summary) == SUMMARY_TRUE_KEYS + SUMMARY_FALSE_KEYS
    for key in SUMMARY_TRUE_KEYS:
        assert summary[key] is True
    for key in SUMMARY_FALSE_KEYS:
        assert summary[key] is False
    rows = audit["block_n_step_closure_rows"]
    assert len(rows) == 8
    for row, (step, title, kind) in zip(rows, STEP_DATA, strict=True):
        assert list(row) == ROW_KEYS
        assert (
            row["closure_row_id"] == step.lower().replace("-", "_").replace(".", "_") + "_closure"
        )
        assert row["step"] == step
        assert row["title"] == title
        assert row["artifact_kind"] == kind
        assert row["step_complete"] is True
        assert row["source_only"] is True
        assert row["plain_data"] is True
        assert row["execution_authorized"] is False
        assert row["real_capabilities_opened"] is False
        assert row["closure_status"] == "complete"
        assert row["closure_result"] == "closed_source_only_execution_blocked"
    domain_cases = [
        ("packaging_release", "packaging_release_closure_summary", (22, 22, 0, 22)),
        ("runtime_safety", "runtime_safety_closure_summary", (18, 18, 0, 18)),
    ]
    for domain, closure_key, expected_counts in domain_cases:
        source_domain = source["domain_readiness_read_summary"][domain]
        closure_domain = audit[closure_key]
        for key in DOMAIN_SOURCE_FIELDS:
            assert closure_domain[key] == source_domain[key]
        assert (
            closure_domain["capability_count"],
            closure_domain["read_capability_count"],
            closure_domain["ready_capability_count"],
            closure_domain["blocked_capability_count"],
        ) == expected_counts
        assert closure_domain["satisfied_requirement_ids"] == []
        assert (
            closure_domain["missing_requirement_ids"] == closure_domain["required_requirement_ids"]
        )
        assert closure_domain["requirements_complete"] is False
        assert closure_domain["domain_ready"] is False
        assert closure_domain["execution_authorized"] is False
        assert closure_domain["all_capabilities_read"] is True
        assert closure_domain["all_capabilities_not_ready"] is True
        assert closure_domain["all_capabilities_blocked"] is True
        assert closure_domain["failure_policy"] == "fail_closed"
        assert closure_domain["domain_closed_in_block_n"] is True
        assert closure_domain["domain_enabled_by_closure"] is False
        assert closure_domain["closure_result"] == "closed_source_only_execution_blocked"
    assert source["domain_readiness_read_summary"]["overall"]["read_capability_count"] == 40
    assert source["domain_readiness_read_summary"]["overall"]["blocked_capability_count"] == 40


def test_invariants_requirements_exe_fail_closed_and_boundaries():
    audit = build_preview_block_n_closure_audit()
    source = build_preview_block_n_safety_gate_readiness_read_model()
    invariant_rows = source["cross_domain_invariant_readiness_read_rows"]
    invariants = audit["cross_domain_invariant_closure_summary"]
    preserved = sum(row["read_invariant_preserved"] is True for row in invariant_rows)
    assert invariants["source_invariant_read_rows"] == invariant_rows
    assert invariants["invariant_count"] == len(invariant_rows)
    assert invariants["preserved_invariant_count"] == preserved
    assert invariants["failed_invariant_count"] == len(invariant_rows) - preserved
    assert invariants["all_invariants_read"] is True
    assert invariants["all_invariants_preserved"] == all(
        row["read_invariant_preserved"] is True for row in invariant_rows
    )
    assert invariants["all_invariants_require_future_explicit_gate"] == all(
        row["requires_future_explicit_gate"] is True for row in invariant_rows
    )
    assert invariants["execution_gate_open_now"] == any(
        row["execution_gate_open_now"] is True for row in invariant_rows
    )
    assert invariants["execution_allowed_now"] == any(
        row["execution_allowed_now"] is True for row in invariant_rows
    )
    assert invariants["execution_performed_now"] == any(
        row["execution_performed_now"] is True for row in invariant_rows
    )
    assert invariants["invariant_count"] == 12
    assert invariants["preserved_invariant_count"] == 12
    assert invariants["failed_invariant_count"] == 0
    assert invariants["failure_policy"] == "fail_closed"
    assert invariants["closure_result"] == "closed_invariants_preserved_execution_blocked"
    requirement_rows = source["validation_requirement_read_rows"]
    requirements = audit["validation_requirement_closure_summary"]
    assert requirements["source_requirement_read_rows"] == requirement_rows
    assert requirements["requirement_count"] == len(requirement_rows)
    assert requirements["required_requirement_count"] == sum(
        row["required"] is True for row in requirement_rows
    )
    assert requirements["present_requirement_count"] == sum(
        row["present"] is True for row in requirement_rows
    )
    assert requirements["completed_requirement_count"] == sum(
        row["completed"] is True for row in requirement_rows
    )
    assert requirements["satisfied_requirement_count"] == sum(
        row["satisfied"] is True for row in requirement_rows
    )
    assert requirements["missing_requirement_count"] == sum(
        row["present"] is False for row in requirement_rows
    )
    assert requirements["all_requirements_read"] is True
    assert requirements["all_requirements_required"] == all(
        row["required"] is True for row in requirement_rows
    )
    assert requirements["all_requirements_missing"] == all(
        row["present"] is False for row in requirement_rows
    )
    assert requirements["all_requirements_block_execution"] == all(
        row["missing_blocks_execution"] is True for row in requirement_rows
    )
    assert requirements["all_requirements_require_future_explicit_step"] == all(
        row["requires_future_explicit_step"] is True for row in requirement_rows
    )
    assert requirements["requirement_count"] == 7
    assert requirements["required_requirement_count"] == 7
    assert requirements["present_requirement_count"] == 0
    assert requirements["completed_requirement_count"] == 0
    assert requirements["satisfied_requirement_count"] == 0
    assert requirements["missing_requirement_count"] == 7
    assert requirements["failure_policy"] == "fail_closed"
    assert requirements["closure_result"] == "closed_requirements_missing_execution_blocked"
    exe = audit["exe_direction_closure_audit"]
    for key, value in source["exe_direction_read_model"].items():
        assert exe[key] == value
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["build_readiness_classification"] == "not_ready"
    assert exe["packaging_readiness_classification"] == "not_ready"
    assert exe["release_readiness_classification"] == "not_ready"
    assert exe["build_authorized_now"] is False
    assert exe["packaging_authorized_now"] is False
    assert exe["release_authorized_now"] is False
    assert exe["future_packaging_gate_required"] is True
    assert exe["future_release_gate_required"] is True
    assert exe["closure_result"] == "exe_direction_preserved_block_n_closed_execution_not_ready"
    fail_closed = audit["fail_closed_closure_decision"]
    assert all(
        value == "fail_closed" for key, value in fail_closed.items() if key.endswith("_policy")
    )
    assert fail_closed["block_n_closure_audit_in_16_8"] == "closed"
    assert fail_closed["block_o_entry_contract_in_17_0"] == "allowed"
    assert fail_closed["only_source_only_17_0_handoff_allowed"] is True
    real_capability_status = fail_closed["real_capability_status"]
    assert list(real_capability_status) == EXPECTED_REAL_CAPABILITY_KEYS
    assert all(real_capability_status[key] == "blocked" for key in EXPECTED_REAL_CAPABILITY_KEYS)
    evidence = audit["non_execution_closure_evidence"]
    assert list(evidence) == EVIDENCE_TRUE_KEYS + EVIDENCE_FALSE_KEYS
    for key in EVIDENCE_TRUE_KEYS:
        assert evidence[key] is True
    for key in EVIDENCE_FALSE_KEYS:
        assert evidence[key] is False
    boundaries = audit["closure_audit_boundaries"]
    assert list(boundaries) == CLOSURE_BOUNDARY_KEYS
    assert all(boundaries[key] is True for key in CLOSURE_BOUNDARY_KEYS)
    source_boundaries = source["source_boundaries"]
    source_nested = source_boundaries["source_block_n_safety_gate_readiness_contract_boundaries"]
    closure_boundaries = audit["source_boundaries"]
    assert closure_boundaries["allowed_imports_only"] == source_boundaries["allowed_imports_only"]
    assert (
        closure_boundaries["source_block_n_safety_gate_readiness_contract"]
        == source_boundaries["source_block_n_safety_gate_readiness_contract"]
    )
    assert (
        closure_boundaries["source_block_n_safety_gate_readiness_matrix"]
        == source_boundaries["source_block_n_safety_gate_readiness_matrix"]
    )
    assert (
        closure_boundaries["source_block_n_safety_gate_read_model"]
        == source_boundaries["source_block_n_safety_gate_read_model"]
    )
    closure_nested = closure_boundaries["source_block_n_safety_gate_readiness_contract_boundaries"]
    for key in [
        "allowed_imports_only",
        "source_block_n_safety_gate_readiness_matrix",
        "source_block_n_safety_gate_read_model",
        "plain_data_source_only",
        "static_non_evaluating",
        "non_mutating",
        "non_authorizing",
    ]:
        assert closure_nested[key] == source_nested[key]
    assert closure_nested["can_feed_16_7"] == source_nested["can_feed_16_7"]
    assert closure_nested["can_feed_16_8"] is True
    assert (
        closure_boundaries["source_block_n_safety_gate_readiness_read_model"]
        == "FUNCTIONAL-PREVIEW-16.7"
    )
    assert closure_boundaries["can_feed_16_8"] is True
    assert closure_boundaries["can_close_block_n"] is True
    assert closure_boundaries["can_feed_17_0"] is True
    for key in INHERITED_FORBIDDEN_SOURCE_FIELDS:
        assert closure_boundaries[key] == source_boundaries[key]
        assert closure_boundaries[key] is False
    assert "forbidden_git_calls_present" not in source_boundaries
    assert closure_boundaries["forbidden_git_calls_present"] is False
    assert all(
        value is False for key, value in closure_boundaries.items() if key.startswith("forbidden_")
    )


def test_source_import_call_and_literal_guards():
    path = Path("ui/pyside_app/preview_block_n_closure_audit.py")
    text = path.read_text()
    tree = ast.parse(text)
    imports = [node for node in tree.body if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    assert not any(isinstance(node, ast.Import) for node in imports)
    assert isinstance(imports[0], ast.ImportFrom) and imports[0].module == "__future__"
    assert [alias.name for alias in imports[0].names] == ["annotations"]
    assert isinstance(imports[1], ast.ImportFrom) and imports[1].module == "typing"
    assert [alias.name for alias in imports[1].names] == ["Any", "Final"]
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_n_safety_gate_readiness_read_model"
    assert [alias.name for alias in imports[2].names] == [
        "build_preview_block_n_safety_gate_readiness_read_model"
    ]
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    builder_calls = [
        node
        for node in calls
        if isinstance(node.func, ast.Name)
        and node.func.id == "build_preview_block_n_safety_gate_readiness_read_model"
    ]
    assert len(builder_calls) == 1
    forbidden_calls = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "exists",
        "stat",
        "glob",
        "rglob",
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
        "git",
    }
    called_names = set()
    for node in calls:
        if isinstance(node.func, ast.Name):
            called_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_names.add(node.func.attr)
    assert not (called_names & forbidden_calls)
    forbidden_literals = [
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    constants = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]
    for literal in forbidden_literals:
        assert literal not in constants


def test_protected_files_exist_and_test_has_no_subprocess_or_git_calls():
    for protected in [
        "ui/pyside_app/preview_block_n_safety_gate_readiness_read_model.py",
        "tests/ui_pyside/test_preview_block_n_safety_gate_readiness_read_model.py",
        "ui/pyside_app/preview_block_n_safety_gate_readiness_contract.py",
        "tests/ui_pyside/test_preview_block_n_safety_gate_readiness_contract.py",
        "ui/pyside_app/preview_block_n_safety_gate_readiness_matrix.py",
        "tests/ui_pyside/test_preview_block_n_safety_gate_readiness_matrix.py",
        "tests/test_local_gateway_validation.py",
        "tests/ui_pyside/test_source_smoke.py",
        "ui/pyside_app/qml/MainWindow.qml",
    ]:
        assert Path(protected).exists()
    tree = ast.parse(Path(__file__).read_text())
    forbidden_modules = {"subprocess", "git"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(alias.name.split(".")[0] not in forbidden_modules for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_modules
    called_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name | ast.Attribute):
            called_names.add(node.func.id if isinstance(node.func, ast.Name) else node.func.attr)
    assert "subprocess" not in called_names
    assert "git" not in called_names
