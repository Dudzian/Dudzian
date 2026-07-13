import ast
import json
from pathlib import Path

import pytest

from ui.pyside_app.preview_block_n_closure_audit import build_preview_block_n_closure_audit
import ui.pyside_app.preview_block_o_entry_contract as entry_module
from ui.pyside_app.preview_block_o_entry_contract import build_preview_block_o_entry_contract

HELPER = Path("ui/pyside_app/preview_block_o_entry_contract.py")
TEST = Path("tests/ui_pyside/test_preview_block_o_entry_contract.py")
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_o_entry_contract_kind",
    "block",
    "step",
    "block_o_entry_contract_status",
    "block_o_entry_contract_decision",
    "block_o_opened",
    "ready_for_block_o_1",
    "next_step",
    "next_step_title",
    "block_n_closure_audit_reference",
    "entry_contract_summary",
    "inherited_block_n_closure_summary",
    "inherited_capability_state",
    "inherited_invariant_state",
    "inherited_requirement_state",
    "exe_direction_entry_contract",
    "fail_closed_entry_decision",
    "non_execution_entry_evidence",
    "entry_contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
REFERENCE_SOURCE_FIELDS = [
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
]
REFERENCE_HANDOFF_FIELDS = [
    "source_block_n_closure_audit_step",
    "source_block_n_closure_audit_read_by_17_0",
    "block_n_closure_audit_available_before_block_o_entry",
    "static_block_n_closure_audit_only",
    "block_o_entry_contract_built_by_17_0",
    "block_o_opened_by_17_0",
    "ready_for_functional_preview_17_1",
]
FALSE_BY_17_0_FIELDS = [
    f"{root}_by_17_0"
    for root in [
        "readiness_recalculated_from_environment",
        "closure_recalculated",
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
]

BLOCKED_STATUS_VALUE = "blocked_for_functional_preview_17_1_block_o_entry_source_not_accepted"

EXPECTED_BLOCK_N_STEPS = [
    "FUNCTIONAL-PREVIEW-16.0",
    "FUNCTIONAL-PREVIEW-16.1",
    "FUNCTIONAL-PREVIEW-16.2",
    "FUNCTIONAL-PREVIEW-16.3",
    "FUNCTIONAL-PREVIEW-16.4",
    "FUNCTIONAL-PREVIEW-16.5",
    "FUNCTIONAL-PREVIEW-16.6",
    "FUNCTIONAL-PREVIEW-16.7",
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

EXPECTED_INHERITED_FORBIDDEN_FIELDS = [
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
    "forbidden_git_calls_present",
]

SUMMARY_TRUE_KEYS = [
    "block_n_closure_audit_available",
    "block_n_closed",
    "block_o_entry_contract_built",
    "block_o_opened",
    "ready_for_block_o_1",
    "ready_for_functional_preview_17_1",
    "block_m_closure_preserved",
    "exe_direction_preserved",
    "entry_contract_source_only",
    "entry_contract_plain_data_only",
    "entry_contract_static_only",
    "entry_contract_read_only",
    "entry_contract_non_evaluating",
    "entry_contract_non_mutating",
    "entry_contract_non_authorizing",
    "all_block_n_steps_preserved",
    "all_capabilities_inherited",
    "all_execution_capabilities_fail_closed",
    "all_execution_capabilities_not_ready",
    "all_execution_capabilities_blocked",
    "all_requirements_inherited",
    "all_requirements_missing",
    "all_requirements_block_execution",
    "all_invariants_inherited",
    "all_invariants_preserved",
    "all_domains_not_ready",
    "all_domains_execution_unauthorized",
    "only_source_only_17_1_handoff_allowed",
]
SUMMARY_FALSE_KEYS = [
    "any_closure_recalculated_now",
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
    "runtime_enabled_by_block_o_entry",
    "packaging_enabled_by_block_o_entry",
    "release_enabled_by_block_o_entry",
    "orders_enabled_by_block_o_entry",
]
EVIDENCE_TRUE_KEYS = [
    "source_block_n_closure_audit_read",
    "block_n_closure_preserved",
    "block_o_entry_contract_built",
    "block_o_entry_contract_only",
    "block_o_opened",
    "ready_for_block_o_1",
    "all_block_n_steps_inherited",
    "all_capability_states_inherited",
    "all_capability_states_not_ready",
    "all_invariant_states_inherited",
    "all_invariant_states_preserved",
    "all_requirement_states_inherited",
    "all_requirement_states_missing",
    "all_execution_authorization_false",
    "all_capabilities_fail_closed",
]
EVIDENCE_FALSE_KEYS = [
    "closure_recalculated",
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
    "real_capabilities_opened_by_block_o_entry",
]
BOUNDARY_KEYS = [
    "block_o_entry_contract_is_plain_data_only",
    "block_o_entry_contract_is_source_only",
    "block_o_entry_contract_reads_16_8_only",
    "block_o_entry_contract_preserves_block_m_closure",
    "block_o_entry_contract_preserves_block_n_closure",
    "block_o_entry_contract_preserves_exe_direction_without_packaging",
    "block_o_entry_contract_is_static_and_non_evaluating",
    "block_o_entry_contract_is_non_mutating",
    "block_o_entry_contract_is_non_authorizing",
    "block_o_entry_contract_can_open_block_o",
    "block_o_entry_contract_can_feed_17_1_read_model",
    "cannot_recalculate_block_n_closure",
    "cannot_recalculate_readiness_from_environment",
    "cannot_evaluate",
    "cannot_accept_condition",
    "cannot_open_real_gate",
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
PROTECTED_FILES = [
    "ui/pyside_app/preview_block_n_closure_audit.py",
    "tests/ui_pyside/test_preview_block_n_closure_audit.py",
    "ui/pyside_app/preview_block_n_safety_gate_readiness_read_model.py",
    "tests/ui_pyside/test_preview_block_n_safety_gate_readiness_read_model.py",
    "tests/ui_pyside/test_source_smoke.py",
    "ui/pyside_app/qml/MainWindow.qml",
    "tests/test_local_gateway_validation.py",
]

BLOCK_N_CLOSURE_SUMMARY_FIELDS = [
    "source_block_n_closed",
    "source_ready_for_block_o_0",
    "source_next_step",
    "source_next_step_title",
    "block_n_step_count",
    "completed_block_n_step_count",
    "all_block_n_steps_complete",
    "all_block_n_steps_source_only",
    "all_block_n_steps_plain_data",
    "all_block_n_steps_execution_unauthorized",
    "all_block_n_steps_real_capabilities_closed",
    "block_n_closure_preserved",
    "block_o_entry_does_not_reopen_block_n",
    "closure_result",
]
CAPABILITY_OVERALL_FIELDS = [
    "total_capability_count",
    "read_capability_count",
    "ready_capability_count",
    "blocked_capability_count",
    "all_capabilities_inherited",
    "all_capabilities_read",
    "all_capabilities_not_ready",
    "all_capabilities_blocked",
    "execution_authorized",
    "enabled_by_block_o_entry",
    "failure_policy",
    "entry_result",
]
FAIL_CLOSED_ENTRY_DECISION_FIELDS = [
    "missing_block_n_closure_audit_policy",
    "missing_block_n_step_closure_policy",
    "missing_inherited_capability_state_policy",
    "missing_inherited_requirement_state_policy",
    "missing_inherited_invariant_state_policy",
    "missing_operator_confirmation_policy",
    "missing_environment_validation_policy",
    "missing_artifact_validation_policy",
    "missing_release_validation_policy",
    "missing_runtime_validation_policy",
    "missing_credentials_validation_policy",
    "missing_future_explicit_gate_policy",
    "failed_block_o_entry_contract_policy",
    "block_n_closure_audit_in_16_8",
    "block_o_entry_contract_in_17_0",
    "block_o_read_model_in_17_1",
    "only_source_only_17_1_handoff_allowed",
    "real_capability_status",
    "real_capability_status_inherited_from_16_8",
    "real_capability_status_modified_by_17_0",
]


def call_name(node):
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return ""


def copy_plain_data(value):
    return json.loads(json.dumps(value))


def assert_entry_blocked(payload):
    assert payload["block_o_opened"] is False
    assert payload["ready_for_block_o_1"] is False
    assert payload["status"] == BLOCKED_STATUS_VALUE
    assert (
        payload["block_o_entry_contract_decision"]
        == payload["block_o_entry_contract_status"].upper()
    )
    assert payload["block_n_closure_audit_reference"]["block_o_opened_by_17_0"] is False
    assert payload["block_n_closure_audit_reference"]["ready_for_functional_preview_17_1"] is False
    assert payload["entry_contract_summary"]["block_o_opened"] is False
    assert payload["entry_contract_summary"]["ready_for_block_o_1"] is False
    assert payload["entry_contract_summary"]["ready_for_functional_preview_17_1"] is False
    assert payload["entry_contract_summary"]["only_source_only_17_1_handoff_allowed"] is False
    assert payload["fail_closed_entry_decision"]["block_o_entry_contract_in_17_0"] == "blocked"
    assert payload["fail_closed_entry_decision"]["block_o_read_model_in_17_1"] == "blocked"
    assert payload["fail_closed_entry_decision"]["only_source_only_17_1_handoff_allowed"] is False
    assert payload["non_execution_entry_evidence"]["block_n_closure_preserved"] is False
    assert payload["non_execution_entry_evidence"]["block_o_opened"] is False
    assert payload["non_execution_entry_evidence"]["ready_for_block_o_1"] is False
    assert payload["source_boundaries"]["can_open_block_o"] is False
    assert payload["source_boundaries"]["can_feed_17_1"] is False
    assert payload["source_boundaries"]["block_n_closure_audit_source_preserved"] is False


def payloads():
    return build_preview_block_o_entry_contract(), build_preview_block_n_closure_audit()


def test_identity_order_reference_and_json_serializable():
    payload, source = payloads()
    json.dumps(payload)
    assert source["schema_version"] == "preview_block_n_closure_audit.v1"
    assert source["block_n_closure_audit_kind"] == "functional_preview_block_n_closure_audit"
    assert source["block"] == "N"
    assert source["step"] == "16.8"
    assert (
        source["status"]
        == "block_n_closed_ready_for_functional_preview_17_0_block_o_entry_contract"
    )
    assert source["future_steps"] == ["functional_preview_17_0_block_o_entry_contract"]
    assert (
        source["block_n_closure_audit_decision"] == source["block_n_closure_audit_status"].upper()
    )
    assert list(payload) == TOP_LEVEL_FIELDS
    assert payload["schema_version"] == "preview_block_o_entry_contract.v1"
    assert payload["block_o_entry_contract_kind"] == "functional_preview_block_o_entry_contract"
    assert payload["block"] == "O"
    assert payload["step"] == "17.0"
    assert "block_o_entry_contract_ready" in payload["block_o_entry_contract_status"]
    assert (
        payload["block_o_entry_contract_decision"]
        == payload["block_o_entry_contract_status"].upper()
    )
    assert payload["block_o_opened"] is True
    assert payload["ready_for_block_o_1"] is True
    assert payload["next_step"] == "FUNCTIONAL-PREVIEW-17.1"
    assert payload["next_step_title"] == "BLOCK O READ MODEL"
    assert payload["status"] == "ready_for_functional_preview_17_1_block_o_read_model"
    assert payload["entry_contract_summary"]["exe_direction_preserved"] is True
    reference = payload["block_n_closure_audit_reference"]
    assert (
        list(reference) == REFERENCE_SOURCE_FIELDS + REFERENCE_HANDOFF_FIELDS + FALSE_BY_17_0_FIELDS
    )
    for key in REFERENCE_SOURCE_FIELDS:
        assert reference[key] == source[key]
    assert reference["source_block_n_closure_audit_step"] == "FUNCTIONAL-PREVIEW-16.8"
    for key in REFERENCE_HANDOFF_FIELDS[1:]:
        assert reference[key] is True
    for key in FALSE_BY_17_0_FIELDS:
        assert reference[key] is False


def test_summary_and_block_n_closure_are_inherited_from_eight_rows():
    payload, source = payloads()
    summary = payload["entry_contract_summary"]
    assert list(summary) == SUMMARY_TRUE_KEYS + SUMMARY_FALSE_KEYS
    assert all(summary[key] is True for key in SUMMARY_TRUE_KEYS)
    assert payload["entry_contract_summary"]["block_m_closure_preserved"] is True
    assert all(summary[key] is False for key in SUMMARY_FALSE_KEYS)
    rows = source["block_n_step_closure_rows"]
    inherited = payload["inherited_block_n_closure_summary"]
    assert list(inherited) == BLOCK_N_CLOSURE_SUMMARY_FIELDS
    assert inherited["source_block_n_closed"] == source["block_n_closed"]
    assert inherited["source_ready_for_block_o_0"] == source["ready_for_block_o_0"]
    assert inherited["source_ready_for_block_o_0"] is True
    assert inherited["source_next_step"] == source["next_step"]
    assert inherited["source_next_step"] == "FUNCTIONAL-PREVIEW-17.0"
    assert inherited["source_next_step_title"] == source["next_step_title"]
    assert inherited["source_next_step_title"] == "BLOCK O ENTRY CONTRACT"
    exact_step_count = len(rows) == len(EXPECTED_BLOCK_N_STEPS)
    expected_step_chain_present = [row["step"] for row in rows] == EXPECTED_BLOCK_N_STEPS
    all_steps_complete = (
        exact_step_count
        and expected_step_chain_present
        and all(row["step_complete"] is True for row in rows)
    )
    all_steps_source_only = (
        exact_step_count
        and expected_step_chain_present
        and all(row["source_only"] is True for row in rows)
    )
    all_steps_plain_data = (
        exact_step_count
        and expected_step_chain_present
        and all(row["plain_data"] is True for row in rows)
    )
    all_steps_execution_unauthorized = (
        exact_step_count
        and expected_step_chain_present
        and all(row["execution_authorized"] is False for row in rows)
    )
    all_real_capabilities_closed = (
        exact_step_count
        and expected_step_chain_present
        and all(row["real_capabilities_opened"] is False for row in rows)
    )
    assert [row["step"] for row in rows] == EXPECTED_BLOCK_N_STEPS
    assert inherited["block_n_step_count"] == len(rows) == 8
    assert (
        inherited["completed_block_n_step_count"]
        == sum(row["step_complete"] is True for row in rows)
        == 8
    )
    assert all_steps_complete is True
    assert all_steps_source_only is True
    assert all_steps_plain_data is True
    assert all_steps_execution_unauthorized is True
    assert all_real_capabilities_closed is True
    assert inherited["all_block_n_steps_complete"] == all_steps_complete
    assert inherited["all_block_n_steps_source_only"] == all_steps_source_only
    assert inherited["all_block_n_steps_plain_data"] == all_steps_plain_data
    assert inherited["all_block_n_steps_execution_unauthorized"] == all_steps_execution_unauthorized
    assert inherited["all_block_n_steps_real_capabilities_closed"] == all_real_capabilities_closed
    assert inherited["block_n_closure_preserved"] == (
        source["block_n_closed"] is True
        and all_steps_complete
        and all_steps_source_only
        and all_steps_plain_data
        and all_steps_execution_unauthorized
        and all_real_capabilities_closed
    )
    assert inherited["block_n_closure_preserved"] is True
    assert inherited["block_o_entry_does_not_reopen_block_n"] == (
        source["block_n_closed"] is True
        and exact_step_count
        and expected_step_chain_present
        and all_steps_execution_unauthorized
        and all_real_capabilities_closed
    )
    assert inherited["block_o_entry_does_not_reopen_block_n"] is True
    assert inherited["closure_result"] == "block_n_closure_inherited_execution_blocked"


def test_capability_invariant_requirement_and_exe_state_preserve_16_8():
    payload, source = payloads()
    capability = payload["inherited_capability_state"]
    for name, source_key in [
        ("packaging_release", "packaging_release_closure_summary"),
        ("runtime_safety", "runtime_safety_closure_summary"),
    ]:
        inherited = capability[name]
        for key, value in source[source_key].items():
            assert inherited[key] == value
        assert inherited["inherited_by_block_o_entry"] is True
        assert inherited["enabled_by_block_o_entry"] is False
    assert capability["packaging_release"]["inherited_by_block_o_entry"] is True
    assert capability["runtime_safety"]["inherited_by_block_o_entry"] is True
    overall = capability["overall"]
    assert list(overall) == CAPABILITY_OVERALL_FIELDS
    assert overall["total_capability_count"] == 40
    assert overall["read_capability_count"] == 40
    assert overall["ready_capability_count"] == 0
    assert overall["blocked_capability_count"] == 40
    assert overall["execution_authorized"] is False
    assert overall["enabled_by_block_o_entry"] is False
    assert overall["failure_policy"] == "fail_closed"
    invariant = payload["inherited_invariant_state"]
    source_invariant = source["cross_domain_invariant_closure_summary"]
    assert list(invariant) == [
        *source_invariant.keys(),
        "inherited_by_block_o_entry",
        "revalidated_by_block_o_entry",
        "entry_result",
    ]
    for key, value in source_invariant.items():
        assert invariant[key] == value
    assert invariant["invariant_count"] == 12
    assert invariant["preserved_invariant_count"] == 12
    assert invariant["failed_invariant_count"] == 0
    assert invariant["inherited_by_block_o_entry"] is True
    assert invariant["entry_result"] == "invariants_inherited_execution_blocked"
    requirement = payload["inherited_requirement_state"]
    source_requirement = source["validation_requirement_closure_summary"]
    assert list(requirement) == [
        *source_requirement.keys(),
        "inherited_by_block_o_entry",
        "validated_by_block_o_entry",
        "entry_result",
    ]
    for key, value in source_requirement.items():
        assert requirement[key] == value
    assert requirement["requirement_count"] == 7
    assert requirement["missing_requirement_count"] == 7
    assert requirement["satisfied_requirement_count"] == 0
    assert requirement["inherited_by_block_o_entry"] is True
    assert requirement["entry_result"] == "requirements_inherited_missing_execution_blocked"
    source_exe = source["exe_direction_closure_audit"]
    entry_exe = payload["exe_direction_entry_contract"]
    assert list(entry_exe) == [
        *source_exe.keys(),
        "block_o_entry_contract_confirms_exe_direction",
        "block_n_closure_source_preserved",
        "entry_contract_is_not_execution_authorization",
        "entry_result",
    ]
    for key, value in source_exe.items():
        assert entry_exe[key] == value
    assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is True
    assert entry_exe["block_n_closure_source_preserved"] is True
    assert entry_exe["entry_contract_is_not_execution_authorization"] is True
    assert entry_exe["entry_result"] == "exe_direction_inherited_block_o_opened_execution_not_ready"
    assert entry_exe["final_product_direction"] == source_exe["final_product_direction"]
    assert (
        entry_exe["build_readiness_classification"] == source_exe["build_readiness_classification"]
    )
    assert (
        entry_exe["packaging_readiness_classification"]
        == source_exe["packaging_readiness_classification"]
    )
    assert (
        entry_exe["release_readiness_classification"]
        == source_exe["release_readiness_classification"]
    )
    assert entry_exe["build_authorized_now"] == source_exe["build_authorized_now"]
    assert entry_exe["packaging_authorized_now"] == source_exe["packaging_authorized_now"]
    assert entry_exe["release_authorized_now"] == source_exe["release_authorized_now"]
    assert (
        entry_exe["future_packaging_gate_required"] == source_exe["future_packaging_gate_required"]
    )
    assert entry_exe["future_release_gate_required"] == source_exe["future_release_gate_required"]


def test_fail_closed_decision_evidence_boundaries_source_lineage_and_future_steps():
    payload, source = payloads()
    decision = payload["fail_closed_entry_decision"]
    assert list(decision) == FAIL_CLOSED_ENTRY_DECISION_FIELDS
    policy_keys = [key for key in decision if key.endswith("_policy")]
    assert policy_keys
    assert all(decision[key] == "fail_closed" for key in policy_keys)
    assert decision["block_n_closure_audit_in_16_8"] == "preserved"
    assert decision["block_o_entry_contract_in_17_0"] == "opened"
    assert decision["block_o_read_model_in_17_1"] == "allowed"
    assert decision["only_source_only_17_1_handoff_allowed"] is True
    assert (
        decision["real_capability_status"]
        == source["fail_closed_closure_decision"]["real_capability_status"]
    )
    assert list(decision["real_capability_status"]) == list(
        source["fail_closed_closure_decision"]["real_capability_status"]
    )
    assert list(decision["real_capability_status"]) == EXPECTED_REAL_CAPABILITY_KEYS
    assert all(value == "blocked" for value in decision["real_capability_status"].values())
    assert decision["real_capability_status_inherited_from_16_8"] is True
    assert decision["real_capability_status_modified_by_17_0"] is False
    evidence = payload["non_execution_entry_evidence"]
    assert list(evidence) == EVIDENCE_TRUE_KEYS + EVIDENCE_FALSE_KEYS
    assert all(evidence[key] is True for key in EVIDENCE_TRUE_KEYS)
    assert all(evidence[key] is False for key in EVIDENCE_FALSE_KEYS)
    boundaries = payload["entry_contract_boundaries"]
    assert list(boundaries) == BOUNDARY_KEYS
    assert all(value is True for value in boundaries.values())
    source_boundaries = payload["source_boundaries"]
    for key, value in source["source_boundaries"].items():
        assert source_boundaries[key] == value
    assert source_boundaries["source_block_n_closure_audit"] == "FUNCTIONAL-PREVIEW-16.8"
    assert source_boundaries["block_n_closure_audit_source_preserved"] is True
    assert source_boundaries["can_open_block_o"] is True
    assert source_boundaries["can_feed_17_1"] is True
    for key in source["source_boundaries"]:
        if key.startswith("forbidden_"):
            assert source_boundaries[key] == source["source_boundaries"][key]
    source_forbidden_fields = [key for key in source_boundaries if key.startswith("forbidden_")]
    assert source_forbidden_fields == EXPECTED_INHERITED_FORBIDDEN_FIELDS
    assert payload["future_steps"] == ["functional_preview_17_1_block_o_read_model"]


def test_source_identity_sentinels_block_entry_without_sanitizing(monkeypatch):
    cases = [
        ("schema_version", "sentinel.schema"),
        ("block_n_closure_audit_kind", "sentinel_kind"),
        ("block", "SENTINEL"),
        ("step", "99.9"),
        ("status", "sentinel_status"),
    ]
    reference_fields = {
        "schema_version",
        "block_n_closure_audit_kind",
        "block",
        "step",
    }
    for key, value in cases:
        source = copy_plain_data(build_preview_block_n_closure_audit())
        source[key] = value
        monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

        payload = build_preview_block_o_entry_contract()

        if key in reference_fields:
            assert payload["block_n_closure_audit_reference"][key] == value
        assert source[key] == value
        assert_entry_blocked(payload)


def test_source_closure_status_sentinel_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_closure_audit_status"] = "sentinel_closure_status"
    source["block_n_closure_audit_decision"] = "SENTINEL_CLOSURE_STATUS"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    reference = payload["block_n_closure_audit_reference"]
    assert reference["block_n_closure_audit_status"] == "sentinel_closure_status"
    assert reference["block_n_closure_audit_decision"] == "SENTINEL_CLOSURE_STATUS"
    assert_entry_blocked(payload)


def test_source_closure_decision_mismatch_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_closure_audit_decision"] = "SENTINEL_DECISION"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["block_n_closure_audit_reference"]["block_n_closure_audit_decision"]
        == "SENTINEL_DECISION"
    )
    assert_entry_blocked(payload)


def test_source_future_steps_sentinel_blocks_entry_without_payload_expansion(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["future_steps"] = ["functional_preview_sentinel"]
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["future_steps"] == ["functional_preview_sentinel"]
    assert payload["future_steps"] == ["functional_preview_17_1_block_o_read_model"]
    assert "future_steps" not in payload["block_n_closure_audit_reference"]
    assert_entry_blocked(payload)


def test_exe_direction_source_fields_are_not_sanitized(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["final_product_direction"] = (
        "sentinel_inherited_direction"
    )
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    entry_exe = payload["exe_direction_entry_contract"]
    assert entry_exe["final_product_direction"] == "sentinel_inherited_direction"
    assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is False
    assert entry_exe["entry_result"] == "exe_direction_not_confirmed_block_o_entry_blocked"
    assert payload["entry_contract_summary"]["exe_direction_preserved"] is False
    assert_entry_blocked(payload)


def test_block_n_closure_preservation_is_derived_from_source(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_closed"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    inherited = payload["inherited_block_n_closure_summary"]
    assert inherited["source_block_n_closed"] is False
    assert inherited["block_n_closure_preserved"] is False
    assert inherited["block_o_entry_does_not_reopen_block_n"] is False
    assert inherited["closure_result"] == "block_n_closure_not_preserved_execution_blocked"
    assert_entry_blocked(payload)


def test_empty_block_n_closure_rows_fail_closed(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_step_closure_rows"] = []
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_block_n_closure_summary"]

    assert inherited["block_n_step_count"] == 0
    assert inherited["completed_block_n_step_count"] == 0
    assert inherited["all_block_n_steps_complete"] is False
    assert inherited["all_block_n_steps_source_only"] is False
    assert inherited["all_block_n_steps_plain_data"] is False
    assert inherited["all_block_n_steps_execution_unauthorized"] is False
    assert inherited["all_block_n_steps_real_capabilities_closed"] is False
    assert inherited["block_n_closure_preserved"] is False
    assert inherited["block_o_entry_does_not_reopen_block_n"] is False
    assert inherited["closure_result"] == "block_n_closure_not_preserved_execution_blocked"
    assert payload["entry_contract_summary"]["all_block_n_steps_preserved"] is False
    assert_entry_blocked(payload)


def test_incomplete_block_n_closure_rows_fail_closed(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_step_closure_rows"] = source["block_n_step_closure_rows"][:-1]
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_block_n_closure_summary"]

    assert inherited["block_n_step_count"] == 7
    assert inherited["completed_block_n_step_count"] == 7
    assert inherited["all_block_n_steps_complete"] is False
    assert inherited["block_n_closure_preserved"] is False
    assert inherited["block_o_entry_does_not_reopen_block_n"] is False
    assert inherited["closure_result"] == "block_n_closure_not_preserved_execution_blocked"
    assert payload["entry_contract_summary"]["all_block_n_steps_preserved"] is False
    assert_entry_blocked(payload)


def test_reordered_block_n_closure_rows_fail_closed(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    rows = source["block_n_step_closure_rows"]
    rows[0], rows[1] = rows[1], rows[0]
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_block_n_closure_summary"]

    assert inherited["block_n_step_count"] == 8
    assert inherited["completed_block_n_step_count"] == 8
    assert inherited["all_block_n_steps_complete"] is False
    assert inherited["block_n_closure_preserved"] is False
    assert inherited["block_o_entry_does_not_reopen_block_n"] is False
    assert inherited["closure_result"] == "block_n_closure_not_preserved_execution_blocked"
    assert payload["entry_contract_summary"]["all_block_n_steps_preserved"] is False
    assert_entry_blocked(payload)


def test_source_not_ready_for_block_o_0_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["ready_for_block_o_0"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["inherited_block_n_closure_summary"]["source_ready_for_block_o_0"] is False
    entry_exe = payload["exe_direction_entry_contract"]
    assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is True
    assert entry_exe["entry_result"] == "exe_direction_inherited_block_o_entry_blocked"
    assert_entry_blocked(payload)


def test_invalid_block_o_entry_target_handoff_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["next_step"] = "FUNCTIONAL-PREVIEW-SENTINEL"
    source["next_step_title"] = "SENTINEL TITLE"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["block_n_closure_audit_reference"]["next_step"] == "FUNCTIONAL-PREVIEW-SENTINEL"
    assert payload["block_n_closure_audit_reference"]["next_step_title"] == "SENTINEL TITLE"
    inherited = payload["inherited_block_n_closure_summary"]
    assert inherited["source_next_step"] == "FUNCTIONAL-PREVIEW-SENTINEL"
    assert inherited["source_next_step_title"] == "SENTINEL TITLE"
    assert_entry_blocked(payload)


def test_source_handoff_decision_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"]["block_o_entry_contract_in_17_0"] = "blocked"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["fail_closed_closure_decision"]["block_o_entry_contract_in_17_0"] == "blocked"
    assert_entry_blocked(payload)


def test_source_handoff_allowed_flag_false_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"]["only_source_only_17_0_handoff_allowed"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["fail_closed_closure_decision"]["only_source_only_17_0_handoff_allowed"] is False
    assert_entry_blocked(payload)


def test_empty_real_capability_status_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"]["real_capability_status"] = {}
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["fail_closed_entry_decision"]["real_capability_status"] == {}
    assert payload["entry_contract_summary"]["all_execution_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_missing_required_forbidden_flag_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["source_boundaries"].pop("forbidden_runtime_calls_present")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert "forbidden_runtime_calls_present" not in payload["source_boundaries"]
    assert_entry_blocked(payload)


def test_true_required_forbidden_flag_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["source_boundaries"]["forbidden_runtime_calls_present"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["source_boundaries"]["forbidden_runtime_calls_present"] is True
    assert_entry_blocked(payload)


def test_real_capability_status_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    real_status = source["fail_closed_closure_decision"]["real_capability_status"]
    first_key = next(iter(real_status))
    real_status[first_key] = "sentinel_not_blocked"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["fail_closed_entry_decision"]["real_capability_status"][first_key]
        == "sentinel_not_blocked"
    )
    assert payload["entry_contract_summary"]["all_execution_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_missing_real_capability_key_blocks_entry_without_rebuilding(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    real_status = source["fail_closed_closure_decision"]["real_capability_status"]
    removed_key = next(iter(real_status))
    real_status.pop(removed_key)
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited_status = payload["fail_closed_entry_decision"]["real_capability_status"]

    assert removed_key not in inherited_status
    assert list(inherited_status) != EXPECTED_REAL_CAPABILITY_KEYS
    assert payload["entry_contract_summary"]["all_execution_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_extra_real_capability_key_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    real_status = source["fail_closed_closure_decision"]["real_capability_status"]
    real_status["sentinel_unknown_capability"] = "blocked"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        "sentinel_unknown_capability"
        in payload["fail_closed_entry_decision"]["real_capability_status"]
    )
    assert payload["entry_contract_summary"]["all_execution_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_reordered_real_capability_keys_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    real_status = source["fail_closed_closure_decision"]["real_capability_status"]
    items = list(real_status.items())
    first = items.pop(0)
    items.append(first)
    source["fail_closed_closure_decision"]["real_capability_status"] = dict(items)
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited_status = payload["fail_closed_entry_decision"]["real_capability_status"]

    assert all(value == "blocked" for value in inherited_status.values())
    assert len(inherited_status) == len(EXPECTED_REAL_CAPABILITY_KEYS)
    assert list(inherited_status) != EXPECTED_REAL_CAPABILITY_KEYS
    assert payload["entry_contract_summary"]["all_execution_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_extra_forbidden_flag_true_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["source_boundaries"]["forbidden_sentinel_unknown_calls_present"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["source_boundaries"]["forbidden_sentinel_unknown_calls_present"] is True
    assert_entry_blocked(payload)


def test_extra_forbidden_flag_false_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["source_boundaries"]["forbidden_sentinel_unknown_calls_present"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["source_boundaries"]["forbidden_sentinel_unknown_calls_present"] is False
    assert_entry_blocked(payload)


def test_block_m_closure_not_preserved_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_summary"]["block_m_closure_preserved"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_summary"]["block_m_closure_preserved"] is False
    assert payload["entry_contract_summary"]["block_m_closure_preserved"] is False
    assert_entry_blocked(payload)


def test_source_summary_not_ready_for_17_0_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_summary"]["ready_for_functional_preview_17_0"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_summary"]["ready_for_functional_preview_17_0"] is False
    assert_entry_blocked(payload)


def test_extra_source_top_level_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["sentinel_top_level"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["sentinel_top_level"] is True
    assert_entry_blocked(payload)


def test_block_n_row_metadata_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_step_closure_rows"][0]["closure_status"] = "sentinel_status"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_block_n_closure_summary"]

    assert source["block_n_step_closure_rows"][0]["closure_status"] == "sentinel_status"
    assert inherited["block_n_closure_preserved"] is False
    assert_entry_blocked(payload)


def test_requirement_row_only_contradiction_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    requirement = source["validation_requirement_closure_summary"]
    requirement["source_requirement_read_rows"][0]["present"] = True
    requirement["source_requirement_read_rows"][0]["satisfied"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_requirement_state"]

    assert inherited["source_requirement_read_rows"][0]["present"] is True
    assert inherited["source_requirement_read_rows"][0]["satisfied"] is True
    assert inherited["present_requirement_count"] == 0
    assert inherited["satisfied_requirement_count"] == 0
    assert_entry_blocked(payload)


def test_invariant_row_only_contradiction_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    invariant = source["cross_domain_invariant_closure_summary"]
    invariant["source_invariant_read_rows"][0]["read_invariant_preserved"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_invariant_state"]

    assert inherited["source_invariant_read_rows"][0]["read_invariant_preserved"] is False
    assert inherited["preserved_invariant_count"] == 12
    assert inherited["failed_invariant_count"] == 0
    assert_entry_blocked(payload)


def test_closure_summary_false_field_true_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_summary"]["any_execution_authorized_now"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_summary"]["any_execution_authorized_now"] is True
    assert_entry_blocked(payload)


def test_closure_summary_true_field_false_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_summary"]["all_execution_capabilities_blocked"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_summary"]["all_execution_capabilities_blocked"] is False
    assert_entry_blocked(payload)


def test_source_evidence_runtime_performed_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["non_execution_closure_evidence"]["runtime_performed"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["non_execution_closure_evidence"]["runtime_performed"] is True
    assert_entry_blocked(payload)


def test_source_evidence_true_claim_false_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["non_execution_closure_evidence"]["all_capabilities_fail_closed"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["non_execution_closure_evidence"]["all_capabilities_fail_closed"] is False
    assert_entry_blocked(payload)


def test_source_closure_boundary_false_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_boundaries"]["cannot_run_runtime"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_boundaries"]["cannot_run_runtime"] is False
    assert_entry_blocked(payload)


def test_source_closure_boundary_extra_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["closure_audit_boundaries"]["sentinel_boundary"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["closure_audit_boundaries"]["sentinel_boundary"] is True
    assert_entry_blocked(payload)


def test_capability_domain_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["packaging_release_closure_summary"]["requirements_complete"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_capability_state"]["packaging_release"]["requirements_complete"] is True
    )
    assert_entry_blocked(payload)


def test_runtime_domain_closure_result_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["runtime_safety_closure_summary"]["closure_result"] = "sentinel_result"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_capability_state"]["runtime_safety"]["closure_result"]
        == "sentinel_result"
    )
    assert_entry_blocked(payload)


def test_requirement_sentinel_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    requirement = source["validation_requirement_closure_summary"]
    requirement["source_requirement_read_rows"][0]["present"] = True
    requirement["source_requirement_read_rows"][0]["satisfied"] = True
    requirement["present_requirement_count"] = 1
    requirement["satisfied_requirement_count"] = 1
    requirement["missing_requirement_count"] = 6
    requirement["all_requirements_missing"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_requirement_state"]

    assert inherited["present_requirement_count"] == 1
    assert inherited["satisfied_requirement_count"] == 1
    assert inherited["all_requirements_missing"] is False
    assert payload["entry_contract_summary"]["all_requirements_missing"] is False
    assert_entry_blocked(payload)


def test_invariant_sentinel_blocks_entry_without_sanitizing(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    invariant = source["cross_domain_invariant_closure_summary"]
    invariant["source_invariant_read_rows"][0]["read_invariant_preserved"] = False
    invariant["preserved_invariant_count"] = 11
    invariant["failed_invariant_count"] = 1
    invariant["all_invariants_preserved"] = False
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    inherited = payload["inherited_invariant_state"]

    assert inherited["source_invariant_read_rows"][0]["read_invariant_preserved"] is False
    assert inherited["failed_invariant_count"] == 1
    assert inherited["all_invariants_preserved"] is False
    assert payload["entry_contract_summary"]["all_invariants_preserved"] is False
    assert_entry_blocked(payload)


def test_fail_closed_decision_policy_lineage_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    decision = source["fail_closed_closure_decision"]
    decision["missing_runtime_validation_policy"] = "fail_open"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["fail_closed_entry_decision"]["real_capability_status"]
        == decision["real_capability_status"]
    )
    assert (
        source["fail_closed_closure_decision"]["missing_runtime_validation_policy"] == "fail_open"
    )
    assert_entry_blocked(payload)


def test_fail_closed_decision_shape_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    decision = source["fail_closed_closure_decision"]
    decision.pop("missing_runtime_validation_policy")
    decision["sentinel_policy"] = "fail_closed"
    items = list(decision.items())
    items[0], items[1] = items[1], items[0]
    source["fail_closed_closure_decision"] = dict(items)
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert "missing_runtime_validation_policy" not in source["fail_closed_closure_decision"]
    assert source["fail_closed_closure_decision"]["sentinel_policy"] == "fail_closed"
    assert_entry_blocked(payload)


def test_readiness_reference_lineage_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    reference = source["block_n_safety_gate_readiness_read_model_reference"]
    reference["step"] = "SENTINEL"
    reference["gate_opened_by_16_8"] = True
    reference["sentinel_extra_reference_field"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert source["block_n_safety_gate_readiness_read_model_reference"]["step"] == "SENTINEL"
    assert (
        source["block_n_safety_gate_readiness_read_model_reference"]["gate_opened_by_16_8"] is True
    )
    assert_entry_blocked(payload)


def test_source_boundary_full_lineage_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    boundaries = source["source_boundaries"]
    boundaries["can_feed_17_0"] = False
    boundaries["source_block_n_safety_gate_readiness_read_model"] = "FUNCTIONAL-PREVIEW-SENTINEL"
    boundaries["source_block_n_safety_gate_readiness_contract"] = "FUNCTIONAL-PREVIEW-SENTINEL"
    nested = boundaries["source_block_n_safety_gate_readiness_contract_boundaries"]
    nested["can_feed_16_7"] = False
    nested.pop("non_authorizing")
    nested["sentinel_extra_nested_boundary"] = True
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["source_boundaries"]["can_feed_17_0"] is False
    assert payload["source_boundaries"]["can_open_block_o"] is False
    assert payload["source_boundaries"]["can_feed_17_1"] is False
    assert_entry_blocked(payload)


def test_reordered_source_boundary_fields_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    items = list(source["source_boundaries"].items())
    items[0], items[1] = items[1], items[0]
    source["source_boundaries"] = dict(items)
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert list(source["source_boundaries"])[0] == "source_block_n_safety_gate_readiness_contract"
    assert_entry_blocked(payload)


def test_invariant_exact_row_lineage_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    rows = source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"]
    rows[0]["invariant_id"] = "sentinel_invariant"
    rows[1]["sentinel_extra_row_field"] = True
    rows[2], rows[3] = rows[3], rows[2]
    rows[-1] = copy_plain_data(rows[0])
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_invariant_state"]["source_invariant_read_rows"][0]["invariant_id"]
        == "sentinel_invariant"
    )
    assert_entry_blocked(payload)


def test_requirement_exact_row_lineage_sentinels_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    requirement = source["validation_requirement_closure_summary"]
    rows = requirement["source_requirement_read_rows"]
    rows[0]["requirement_id"] = "sentinel_requirement"
    rows[1]["sentinel_extra_row_field"] = True
    rows[2], rows[3] = rows[3], rows[2]
    rows[4]["present"] = False
    rows[4]["satisfied"] = True
    requirement["satisfied_requirement_count"] = 1
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_requirement_state"]["source_requirement_read_rows"][0]["requirement_id"]
        == "sentinel_requirement"
    )
    assert_entry_blocked(payload)


def test_capability_domain_exact_shape_and_requirement_ids_block_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    domain = source["packaging_release_closure_summary"]
    domain["required_requirement_ids"] = ["sentinel_requirement"]
    domain["missing_requirement_ids"] = ["sentinel_requirement"]
    domain["sentinel_extra_domain_field"] = True
    runtime_items = list(source["runtime_safety_closure_summary"].items())
    runtime_items[0], runtime_items[1] = runtime_items[1], runtime_items[0]
    source["runtime_safety_closure_summary"] = dict(runtime_items)
    source["runtime_safety_closure_summary"].pop("closure_result")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["inherited_capability_state"]["packaging_release"][
        "required_requirement_ids"
    ] == ["sentinel_requirement"]
    assert_entry_blocked(payload)


def test_readiness_reference_exact_status_and_decision_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    reference = source["block_n_safety_gate_readiness_read_model_reference"]
    reference["block_n_safety_gate_readiness_read_model_status"] = "sentinel_status"
    reference["block_n_safety_gate_readiness_read_model_decision"] = "SENTINEL_STATUS"
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        source["block_n_safety_gate_readiness_read_model_reference"][
            "block_n_safety_gate_readiness_read_model_status"
        ]
        == "sentinel_status"
    )
    assert (
        source["block_n_safety_gate_readiness_read_model_reference"][
            "block_n_safety_gate_readiness_read_model_decision"
        ]
        == "SENTINEL_STATUS"
    )
    assert_entry_blocked(payload)


def test_exe_source_shape_sentinels_block_entry_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    exe = source["exe_direction_closure_audit"]
    exe["sentinel_extra_exe_field"] = False
    exe.pop("failure_policy")
    items = list(exe.items())
    items[0], items[1] = items[1], items[0]
    source["exe_direction_closure_audit"] = dict(items)
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()
    entry_exe = payload["exe_direction_entry_contract"]

    assert "failure_policy" not in entry_exe
    assert entry_exe["sentinel_extra_exe_field"] is False
    assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is False
    assert_entry_blocked(payload)


def test_missing_block_n_row_step_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_step_closure_rows"][0].pop("step")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert payload["inherited_block_n_closure_summary"]["block_n_closure_preserved"] is False
    assert_entry_blocked(payload)


def test_missing_source_boundaries_nested_field_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["source_boundaries"].pop("source_block_n_safety_gate_readiness_contract_boundaries")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        "source_block_n_safety_gate_readiness_contract_boundaries"
        not in payload["source_boundaries"]
    )
    assert_entry_blocked(payload)


def test_invariant_row_full_semantics_sentinels_block_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    rows = source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"]
    rows[0]["failure_policy"] = "fail_open"
    rows[1]["source_contract_id"] = "sentinel_contract_id"
    rows[2]["source_contract_result"] = "sentinel_result"
    rows[3].pop("invariant_id")
    rows[4].pop("failure_policy")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_invariant_state"]["source_invariant_read_rows"][0]["failure_policy"]
        == "fail_open"
    )
    assert_entry_blocked(payload)


def test_requirement_row_full_semantics_sentinels_block_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    rows = source["validation_requirement_closure_summary"]["source_requirement_read_rows"]
    rows[0]["failure_policy"] = "fail_open"
    rows[1]["source_contract_row_id"] = "sentinel_contract_row"
    rows[2]["applicable_domains"] = ["sentinel_domain"]
    rows[3]["source_present"] = True
    rows[4].pop("requirement_id")
    rows[5].pop("failure_policy")
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)

    payload = build_preview_block_o_entry_contract()

    assert (
        payload["inherited_requirement_state"]["source_requirement_read_rows"][0]["failure_policy"]
        == "fail_open"
    )
    assert_entry_blocked(payload)


def assert_exe_sentinel_blocked(payload, field, value):
    entry_exe = payload["exe_direction_entry_contract"]
    assert entry_exe[field] == value
    assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is False
    assert payload["entry_contract_summary"]["exe_direction_preserved"] is False
    assert entry_exe["entry_result"] == "exe_direction_not_confirmed_block_o_entry_blocked"
    assert_entry_blocked(payload)


def build_with_source(monkeypatch, source):
    monkeypatch.setattr(entry_module, "build_preview_block_n_closure_audit", lambda: source)
    return build_preview_block_o_entry_contract()


def test_exe_runtime_gate_open_now_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["runtime_gate_open_now"] = True
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "runtime_gate_open_now", True)


def test_exe_release_gate_open_now_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["release_gate_open_now"] = True
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "release_gate_open_now", True)


def test_exe_release_executed_now_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["release_executed_now"] = True
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "release_executed_now", True)


def test_exe_artifact_created_now_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["artifact_created_now"] = True
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "artifact_created_now", True)


def test_exe_future_packaging_live_credentials_guard_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["future_packaging_must_not_use_live_credentials"] = False
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "future_packaging_must_not_use_live_credentials", False)


def test_exe_matrix_result_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["matrix_result"] = "sentinel_result"
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "matrix_result", "sentinel_result")


def test_exe_closure_result_sentinel_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["exe_direction_closure_audit"]["closure_result"] = "sentinel_result"
    payload = build_with_source(monkeypatch, source)
    assert_exe_sentinel_blocked(payload, "closure_result", "sentinel_result")


def test_invariant_summary_extra_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["cross_domain_invariant_closure_summary"]["sentinel_extra_summary_field"] = True
    payload = build_with_source(monkeypatch, source)
    assert payload["inherited_invariant_state"]["sentinel_extra_summary_field"] is True
    assert_entry_blocked(payload)


def test_invariant_summary_missing_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["cross_domain_invariant_closure_summary"].pop("closure_result")
    payload = build_with_source(monkeypatch, source)
    assert "closure_result" not in payload["inherited_invariant_state"]
    assert_entry_blocked(payload)


def test_invariant_summary_reordered_fields_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    items = list(source["cross_domain_invariant_closure_summary"].items())
    items[0], items[1] = items[1], items[0]
    source["cross_domain_invariant_closure_summary"] = dict(items)
    payload = build_with_source(monkeypatch, source)
    assert list(payload["inherited_invariant_state"])[0] == "invariant_count"
    assert_entry_blocked(payload)


def test_requirement_summary_extra_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["validation_requirement_closure_summary"]["sentinel_extra_summary_field"] = True
    payload = build_with_source(monkeypatch, source)
    assert payload["inherited_requirement_state"]["sentinel_extra_summary_field"] is True
    assert_entry_blocked(payload)


def test_requirement_summary_missing_field_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["validation_requirement_closure_summary"].pop("closure_result")
    payload = build_with_source(monkeypatch, source)
    assert "closure_result" not in payload["inherited_requirement_state"]
    assert_entry_blocked(payload)


def test_requirement_summary_reordered_fields_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    items = list(source["validation_requirement_closure_summary"].items())
    items[0], items[1] = items[1], items[0]
    source["validation_requirement_closure_summary"] = dict(items)
    payload = build_with_source(monkeypatch, source)
    assert list(payload["inherited_requirement_state"])[0] == "requirement_count"
    assert_entry_blocked(payload)


def test_missing_real_capability_status_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"].pop("real_capability_status")
    payload = build_with_source(monkeypatch, source)
    assert "real_capability_status" not in source["fail_closed_closure_decision"]
    decision = payload["fail_closed_entry_decision"]
    assert decision["real_capability_status"] == {}
    assert decision["real_capability_status_inherited_from_16_8"] is False
    assert decision["real_capability_status_modified_by_17_0"] is False
    assert_entry_blocked(payload)


def test_missing_block_n_closure_transition_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"].pop("block_n_closure_audit_in_16_8")
    payload = build_with_source(monkeypatch, source)
    assert "block_n_closure_audit_in_16_8" not in source["fail_closed_closure_decision"]
    assert_entry_blocked(payload)


def test_missing_block_o_entry_transition_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"].pop("block_o_entry_contract_in_17_0")
    payload = build_with_source(monkeypatch, source)
    assert "block_o_entry_contract_in_17_0" not in source["fail_closed_closure_decision"]
    assert_entry_blocked(payload)


def test_missing_source_only_handoff_flag_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"].pop("only_source_only_17_0_handoff_allowed")
    payload = build_with_source(monkeypatch, source)
    assert "only_source_only_17_0_handoff_allowed" not in source["fail_closed_closure_decision"]
    assert_entry_blocked(payload)


def test_invariant_closure_result_sentinel_alone_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["cross_domain_invariant_closure_summary"]["closure_result"] = (
        "sentinel_invariant_closure_result"
    )
    payload = build_with_source(monkeypatch, source)

    assert (
        payload["inherited_invariant_state"]["closure_result"]
        == "sentinel_invariant_closure_result"
    )
    assert_entry_blocked(payload)


def test_requirement_closure_result_sentinel_alone_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["validation_requirement_closure_summary"]["closure_result"] = (
        "sentinel_requirement_closure_result"
    )
    payload = build_with_source(monkeypatch, source)

    assert (
        payload["inherited_requirement_state"]["closure_result"]
        == "sentinel_requirement_closure_result"
    )
    assert_entry_blocked(payload)


def test_non_dict_real_capability_status_blocks_without_inherited_provenance(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["fail_closed_closure_decision"]["real_capability_status"] = "sentinel_not_a_mapping"
    payload = build_with_source(monkeypatch, source)
    decision = payload["fail_closed_entry_decision"]

    assert (
        source["fail_closed_closure_decision"]["real_capability_status"] == "sentinel_not_a_mapping"
    )
    assert decision["real_capability_status"] == {}
    assert decision["real_capability_status_inherited_from_16_8"] is False
    assert decision["real_capability_status_modified_by_17_0"] is False
    assert_entry_blocked(payload)


def test_real_capability_key_list_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    sentinel = list(EXPECTED_REAL_CAPABILITY_KEYS)
    source["fail_closed_closure_decision"]["real_capability_status"] = sentinel
    payload = build_with_source(monkeypatch, source)

    assert source["fail_closed_closure_decision"]["real_capability_status"] == sentinel
    decision = payload["fail_closed_entry_decision"]
    assert decision["real_capability_status"] == {}
    assert decision["real_capability_status_inherited_from_16_8"] is False
    assert decision["real_capability_status_modified_by_17_0"] is False
    assert_entry_blocked(payload)


@pytest.mark.parametrize(
    "section_key",
    [
        "block_n_step_closure_rows",
        "packaging_release_closure_summary",
        "runtime_safety_closure_summary",
        "cross_domain_invariant_closure_summary",
        "validation_requirement_closure_summary",
        "exe_direction_closure_audit",
        "fail_closed_closure_decision",
        "source_boundaries",
    ],
)
def test_missing_top_level_source_section_blocks_without_exception(monkeypatch, section_key):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source.pop(section_key)

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert section_key not in source
    assert_entry_blocked(payload)
    if section_key in [
        "packaging_release_closure_summary",
        "runtime_safety_closure_summary",
    ]:
        domain_name = (
            "packaging_release"
            if section_key == "packaging_release_closure_summary"
            else "runtime_safety"
        )
        domain = payload["inherited_capability_state"][domain_name]
        assert domain["inherited_by_block_o_entry"] is False
        overall = payload["inherited_capability_state"]["overall"]
        assert overall["total_capability_count"] == 0
        assert overall["read_capability_count"] == 0
        assert overall["ready_capability_count"] == 0
        assert overall["blocked_capability_count"] == 0
        assert overall["all_capabilities_inherited"] is False
        assert overall["entry_result"] == "inherited_source_invalid_execution_blocked"
        assert payload["entry_contract_summary"]["all_capabilities_inherited"] is False
    if section_key == "cross_domain_invariant_closure_summary":
        state = payload["inherited_invariant_state"]
        assert state["inherited_by_block_o_entry"] is False
        assert state["entry_result"] == "invariant_source_unavailable_execution_blocked"
        assert payload["entry_contract_summary"]["all_invariants_inherited"] is False
    if section_key == "validation_requirement_closure_summary":
        state = payload["inherited_requirement_state"]
        assert state["inherited_by_block_o_entry"] is False
        assert state["entry_result"] == "requirement_source_unavailable_execution_blocked"
        assert payload["entry_contract_summary"]["all_requirements_inherited"] is False
    if section_key == "exe_direction_closure_audit":
        entry_exe = payload["exe_direction_entry_contract"]
        assert entry_exe["block_o_entry_contract_confirms_exe_direction"] is False
        assert entry_exe["entry_result"] == ("exe_direction_not_confirmed_block_o_entry_blocked")
    if section_key == "fail_closed_closure_decision":
        decision = payload["fail_closed_entry_decision"]
        assert decision["real_capability_status"] == {}
        assert decision["real_capability_status_inherited_from_16_8"] is False
        assert decision["real_capability_status_modified_by_17_0"] is False
    if section_key == "source_boundaries":
        assert payload["source_boundaries"]["block_n_closure_audit_source_preserved"] is False
        assert payload["source_boundaries"]["can_open_block_o"] is False
        assert payload["source_boundaries"]["can_feed_17_1"] is False


@pytest.mark.parametrize(
    ("scalar_key", "summary_key"),
    [
        ("block_n_closed", "source_block_n_closed"),
        ("ready_for_block_o_0", "source_ready_for_block_o_0"),
        ("next_step", "source_next_step"),
        ("next_step_title", "source_next_step_title"),
    ],
)
def test_missing_top_level_identity_scalar_blocks_without_exception(
    monkeypatch,
    scalar_key,
    summary_key,
):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source.pop(scalar_key)

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert scalar_key not in source
    assert payload["block_n_closure_audit_reference"][scalar_key] is None
    assert payload["inherited_block_n_closure_summary"][summary_key] is None
    assert_entry_blocked(payload)


@pytest.mark.parametrize(
    ("section_key", "sentinel"),
    [
        ("packaging_release_closure_summary", []),
        ("cross_domain_invariant_closure_summary", "sentinel_non_mapping"),
        ("validation_requirement_closure_summary", None),
        ("source_boundaries", []),
    ],
)
def test_non_dict_top_level_source_section_blocks_without_exception(
    monkeypatch,
    section_key,
    sentinel,
):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source[section_key] = sentinel

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source[section_key] == sentinel
    assert_entry_blocked(payload)
    if section_key == "packaging_release_closure_summary":
        domain = payload["inherited_capability_state"]["packaging_release"]
        assert domain["inherited_by_block_o_entry"] is False
        assert (
            payload["inherited_capability_state"]["overall"]["all_capabilities_inherited"] is False
        )
        assert payload["entry_contract_summary"]["all_capabilities_inherited"] is False
    if section_key == "cross_domain_invariant_closure_summary":
        state = payload["inherited_invariant_state"]
        assert state["inherited_by_block_o_entry"] is False
        assert state["entry_result"] == "invariant_source_unavailable_execution_blocked"
        assert payload["entry_contract_summary"]["all_invariants_inherited"] is False
    if section_key == "validation_requirement_closure_summary":
        state = payload["inherited_requirement_state"]
        assert state["inherited_by_block_o_entry"] is False
        assert state["entry_result"] == "requirement_source_unavailable_execution_blocked"
        assert payload["entry_contract_summary"]["all_requirements_inherited"] is False
    if section_key == "source_boundaries":
        assert payload["source_boundaries"]["block_n_closure_audit_source_preserved"] is False
        assert payload["source_boundaries"]["can_open_block_o"] is False
        assert payload["source_boundaries"]["can_feed_17_1"] is False


@pytest.mark.parametrize(
    ("source_key", "domain_name"),
    [
        ("packaging_release_closure_summary", "packaging_release"),
        ("runtime_safety_closure_summary", "runtime_safety"),
    ],
)
def test_capability_domain_owned_field_shadowing_blocks_entry(
    monkeypatch,
    source_key,
    domain_name,
):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    domain = source[source_key]
    domain["inherited_by_block_o_entry"] = True
    domain["enabled_by_block_o_entry"] = False

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source[source_key]["inherited_by_block_o_entry"] is True
    assert source[source_key]["enabled_by_block_o_entry"] is False
    assert payload["inherited_capability_state"][domain_name]["inherited_by_block_o_entry"] is True
    assert payload["inherited_capability_state"]["overall"]["all_capabilities_inherited"] is False
    assert_entry_blocked(payload)


def test_invariant_owned_field_shadowing_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    state = source["cross_domain_invariant_closure_summary"]
    state["inherited_by_block_o_entry"] = True
    state["revalidated_by_block_o_entry"] = False
    state["entry_result"] = "invariants_inherited_execution_blocked"

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert state["inherited_by_block_o_entry"] is True
    assert state["revalidated_by_block_o_entry"] is False
    assert state["entry_result"] == "invariants_inherited_execution_blocked"
    assert_entry_blocked(payload)


def test_requirement_owned_field_shadowing_blocks_entry(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    state = source["validation_requirement_closure_summary"]
    state["inherited_by_block_o_entry"] = True
    state["validated_by_block_o_entry"] = False
    state["entry_result"] = "requirements_inherited_missing_execution_blocked"

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert state["inherited_by_block_o_entry"] is True
    assert state["validated_by_block_o_entry"] is False
    assert state["entry_result"] == "requirements_inherited_missing_execution_blocked"
    assert_entry_blocked(payload)


def test_malformed_block_n_row_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["block_n_step_closure_rows"][0] = None

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source["block_n_step_closure_rows"][0] is None
    assert_entry_blocked(payload)


def test_malformed_invariant_row_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"][0] = 7

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"][0] == 7
    assert_entry_blocked(payload)


def test_malformed_requirement_row_blocks_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["validation_requirement_closure_summary"]["source_requirement_read_rows"][0] = []

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source["validation_requirement_closure_summary"]["source_requirement_read_rows"][0] == []
    assert_entry_blocked(payload)


def test_non_list_invariant_rows_block_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"] = None

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source["cross_domain_invariant_closure_summary"]["source_invariant_read_rows"] is None
    assert_entry_blocked(payload)


def test_non_list_requirement_rows_block_without_exception(monkeypatch):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source["validation_requirement_closure_summary"]["source_requirement_read_rows"] = (
        "sentinel_non_list"
    )

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert (
        source["validation_requirement_closure_summary"]["source_requirement_read_rows"]
        == "sentinel_non_list"
    )
    assert_entry_blocked(payload)


@pytest.mark.parametrize(
    ("section_key", "field_key", "sentinel"),
    [
        ("packaging_release_closure_summary", "capability_count", 22.0),
        ("runtime_safety_closure_summary", "ready_capability_count", False),
        ("cross_domain_invariant_closure_summary", "invariant_count", 12.0),
        ("validation_requirement_closure_summary", "requirement_count", 7.0),
    ],
)
def test_numeric_type_drift_blocks_without_exception(
    monkeypatch,
    section_key,
    field_key,
    sentinel,
):
    source = copy_plain_data(build_preview_block_n_closure_audit())
    source[section_key][field_key] = sentinel

    payload = build_with_source(monkeypatch, source)

    json.dumps(payload)
    assert source[section_key][field_key] == sentinel
    assert type(source[section_key][field_key]) is type(sentinel)
    assert_entry_blocked(payload)


def test_helper_source_guards_and_protected_files_exist():
    tree = ast.parse(HELPER.read_text())
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import)]
    import_from = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == []
    assert len(import_from) == 3
    assert [(node.module, [alias.name for alias in node.names]) for node in import_from] == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        ("ui.pyside_app.preview_block_n_closure_audit", ["build_preview_block_n_closure_audit"]),
    ]
    call_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    calls = [call_name(node) for node in call_nodes]
    attribute_calls = [
        call_name(node) for node in call_nodes if isinstance(node.func, ast.Attribute)
    ]
    assert calls.count("build_preview_block_n_closure_audit") == 1
    allowed_calls = {
        "build_preview_block_n_closure_audit",
        "dict",
        "len",
        "list",
        "sum",
        "all",
        "any",
        "update",
        "upper",
        "values",
        "items",
        "startswith",
        "endswith",
        "_block_n_closure_rows_are_expected",
        "_invariant_state_is_self_consistent",
        "_requirement_state_is_self_consistent",
        "_capability_domain_is_expected",
        "_source_closure_summary_is_expected",
        "_source_non_execution_evidence_is_expected",
        "_source_closure_boundaries_are_expected",
        "_source_fail_closed_decision_is_expected",
        "_source_readiness_reference_is_expected",
        "_source_boundaries_are_expected",
        "_plain_dict_section",
        "_plain_dict_section_is_present",
        "_plain_dict_section_has_exact_fields",
        "_plain_list_section",
        "type",
        "enumerate",
        "replace",
        "lower",
        "_block_n_closure_source_identity_is_expected",
        "_build_reference",
        "_build_summary",
        "_build_block_n_summary",
        "_build_capability_state",
        "_build_invariant_state",
        "_build_requirement_state",
        "_build_exe_direction",
        "_build_fail_closed_decision",
        "_build_entry_source_acceptance",
        "_real_capability_status_is_exactly_blocked",
        "_exe_direction_source_is_expected",
        "_build_non_execution_evidence",
        "_build_entry_source_acceptance",
        "_build_entry_boundaries",
        "_build_source_boundaries",
    }
    assert set(calls) <= allowed_calls
    assert set(attribute_calls) <= {
        "update",
        "upper",
        "values",
        "items",
        "startswith",
        "endswith",
        "replace",
        "lower",
    }
    source = HELPER.read_text()
    forbidden_literals = [
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    assert all(token not in source for token in forbidden_literals)
    assert "block_n_closure_inherited_" + "block_o_opened_execution_blocked" not in source
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
    assert not (set(calls) & forbidden_calls)
    test_tree = ast.parse(TEST.read_text())
    forbidden_test_import_roots = {"git", "subprocess"}
    for node in ast.walk(test_tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_test_import_roots
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            assert node.module.split(".")[0] not in forbidden_test_import_roots
    test_calls = [call_name(node) for node in ast.walk(test_tree) if isinstance(node, ast.Call)]
    assert "git" not in test_calls
    assert "subprocess" not in test_calls
    for protected in PROTECTED_FILES:
        assert Path(protected).exists()
