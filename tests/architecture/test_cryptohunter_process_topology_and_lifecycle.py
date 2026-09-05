"""Contract tests for CryptoHunter M0.3 process topology and lifecycle."""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from collections import Counter
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, cast

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = (
    ROOT / "docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json"
)
ARCH_README = ROOT / "docs/architecture/cryptohunter_product_architecture/README.md"
MAIN_README = ROOT / "README.md"
SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{16}|api[_-]?secret\s*[:=]\s*['\"][^'\"]+|password\s*[:=]\s*['\"][^'\"]+|pin\s*[:=]\s*['\"][0-9]{4,}|BEGIN (RSA|EC|OPENSSH) PRIVATE KEY)",
    re.IGNORECASE,
)
REQUIRED_SHUTDOWN_FIELDS = {
    "name",
    "description",
    "allowed_client_roles",
    "requires_core_reachable",
    "requires_operator_authentication",
    "requires_secondary_confirmation",
    "stops_desktop_shell",
    "stops_tray_agent",
    "stops_strategies",
    "stops_core",
    "triggers_kill_switch",
    "blocks_new_order_intents",
    "persists_checkpoint",
    "expected_core_result",
    "invariants",
}


def load_contract() -> dict:
    return cast(dict[Any, Any], json.loads(CONTRACT.read_text(encoding="utf-8")))


def startup_recovery_contract() -> dict[str, Any]:
    return cast(dict[str, Any], load_contract()["corehost_startup_recovery_contract"])


def roles_by_name(data: dict) -> dict:
    return {role["name"]: role for role in data["process_roles"]}


def modes_by_name(data: dict) -> dict:
    return {mode["name"]: mode for mode in data["deployment_modes"]}


def intents_by_name(data: dict) -> dict:
    return {intent["name"]: intent for intent in data["shutdown_intents"]}


def test_architecture_readme_is_m0_source_of_truth() -> None:
    arch = ARCH_README.read_text(encoding="utf-8")
    main = MAIN_README.read_text(encoding="utf-8")
    assert "## Status M0.1 — closed" in arch
    assert "## Status M0.2 — closed" in arch
    assert "## Status M0.3 — closed" in arch
    assert "## Status M0.3 — under audit" not in arch
    assert "process_topology_and_lifecycle.md" in arch
    assert "process_topology_and_lifecycle.json" in arch
    assert "M0.14" in arch
    assert "## CryptoHunter Product Architecture Contract (M0)" not in main


def test_schema_baseline_roles_modes_and_health_states() -> None:
    data = load_contract()
    assert data["schema_version"] == "cryptohunter.process_topology_and_lifecycle.v1"
    assert data["m0_element"] == "M0.3"
    assert re.fullmatch(r"[0-9a-f]{40}", data["baseline_commit"])
    assert set(roles_by_name(data)) == {"core_host", "tray_agent", "desktop_shell", "bootstrapper"}
    modes = modes_by_name(data)
    assert set(modes) == {"desktop_user_session", "windows_service"}
    assert modes["desktop_user_session"]["status"] == "initial"
    assert modes["windows_service"]["status"] == "future_not_implemented"
    assert data["process_health_states"] == [
        "NOT_STARTED",
        "STARTING",
        "HEALTHY",
        "DEGRADED",
        "STOPPING",
        "STOPPED",
        "CRASHED",
    ]


def test_corehost_initialized_startup_recovery_has_one_exact_order() -> None:
    contract = startup_recovery_contract()
    assert contract["expands_runtime_session_order_step"] == "recover writable StateStore"
    assert contract["preconditions"] == [
        "canonical CoreHost scope resolved",
        "process lock held",
        "current RuntimeSession active-process manifestation exists",
        "mutable StateStore open",
    ]
    assert contract["initialized_order"] == [
        "read_verified_snapshot_initial",
        "bind_corehost_scope_to_verified_state_store_identity",
        "recover_m0_3_protected_freshness",
        "read_verified_snapshot_after_protected_recovery",
        "recover_migrations_in_sealed_schema_chain_order",
        "read_verified_snapshot_after_migration_recovery",
        "recover_secret_handoffs_in_handoff_id_order",
        "read_verified_snapshot_final",
        "publish_fresh_process_local_durable_evidence",
        "recovery_complete",
        "durably_publish_current_runtime_session_history",
        "determine_startup_readiness",
        "READY_only_if_all_later_gates_pass",
    ]


def test_corehost_empty_startup_skips_initialized_authority_pipeline() -> None:
    empty = startup_recovery_contract()["empty_or_uninitialized"]
    assert empty["order"] == [
        "read_verified_snapshot_initial_returns_none",
        "skip_initialized_recovery",
        "determine_startup_readiness",
    ]
    assert empty["skips"] == [
        "M0.3 protected recovery",
        "Migration recovery",
        "SecretHandoff recovery",
        "LocalDurableEvidence publication",
    ]
    assert empty["possible_readiness"] == "SETUP_REQUIRED"
    assert empty["consumes_bootstrap"] is False
    assert "durable RuntimeSession history" in empty["must_not_create"]
    assert "security authority" in empty["must_not_create"]


def test_corehost_protected_recovery_is_first_initialized_authority_gate() -> None:
    protected = startup_recovery_contract()["protected_freshness"]
    assert protected["startup_owned"] is True
    assert protected["mandatory_for_initialized"] is True
    assert protected["production_owner"] == (
        "ProtectedFreshnessHandoffCoordinator.recover_protected_state(scope)"
    )
    assert protected["new_ordinary_prepare"] is False
    assert protected["exact_committed_current"] == "VERIFY_WITH_ZERO_UNNECESSARY_MUTATION"
    assert protected["prepared"] == "EXISTING_PROTOCOL_DECIDES_EXACT_ABORT_OR_FINALIZE"
    assert protected["missing_malformed_mismatched_or_unresolved"] == "FAIL_CLOSED_NO_READY"
    assert protected["initial_gate_is_session_wide_authority_token"] is False
    assert protected["later_mutations_keep_existing_protected_preflight"] is True


def test_corehost_migration_startup_recovery_reuses_sealed_authority() -> None:
    migration = startup_recovery_contract()["migration"]
    assert migration["startup_owned"] is True
    assert migration["discovery_source"] == "fresh verified StateStore snapshot"
    assert migration["declaration_mints_execution_authority"] is False
    assert migration["missing_registry_entry"] == "FAIL_CLOSED"
    assert migration["states"] == {
        "COMPLETED": "TERMINAL_ZERO_EFFECT",
        "FAILED": "TERMINAL_FAILURE_NO_SQL_REPLAY",
        "PREPARED": "RESOLVE_WITH_EXISTING_PRODUCTION_MIGRATION_AUTHORITY",
        "APPLYING": "RESUME_WITH_EXISTING_PRODUCTION_MIGRATION_AUTHORITY",
        "DURABLE_MIGRATED": "COMPLETE_WITHOUT_STRUCTURAL_SQL_REPLAY",
    }
    assert migration["already_materialized_structural_sql_replay"] is False
    assert migration["corehost_executes_sql"] is False
    assert migration["multiple_family_order"] == (
        "SEMANTIC_EXACT_SOURCE_TO_TARGET_SCHEMA_CHAIN_FROM_SEALED_REGISTRY"
    )
    assert migration["forbidden_order_sources"] == [
        "SQLite row order",
        "record storage order",
        "lexicographic migration_id",
        "caller order",
    ]


def test_corehost_secret_handoff_startup_behavior_is_closed_by_state() -> None:
    secret = startup_recovery_contract()["secret_handoff"]
    assert secret["startup_owned"] is True
    assert secret["states"] == {
        "PREPARED": "RECONCILE_NEVER_BLIND_RETRY_INITIAL_MUTATION",
        "COMMITTED": "EXACT_HANDOFF_IDEMPOTENT_CLEANUP_REDELIVERY",
        "CLEANUP_PENDING": "TERMINAL_ZERO_EXTERNAL_MUTATION",
        "UNKNOWN_RECONCILIATION": "UNRESOLVED_ZERO_BLIND_MUTATION_NO_READY",
    }
    assert secret["multiple_family_order"] == (
        "DETERMINISTIC_ONLY_NON_AUTHORITATIVE_ASCENDING_CANONICAL_HANDOFF_ID"
    )
    assert secret["invalid_family"] == "FAIL_CLOSED"


def test_corehost_recovery_requires_fresh_observations_and_final_evidence() -> None:
    contract = startup_recovery_contract()
    initial = contract["initial_observation"]
    assert initial["only_source"] == (
        "read_verified_snapshot generation-pinned durable StateStore snapshot"
    )
    assert initial["invalid_or_corrupt"] == "FAIL_CLOSED_NO_AUTOMATIC_PHYSICAL_RESTORE"
    assert contract["freshness_between_stages"] == {
        "after_every_authority_changing_recovery": "fresh read_verified_snapshot required",
        "stale_snapshot_reuse": False,
        "top_level_m0_3_gate_skips_subsystem_protected_preflight": False,
    }
    evidence = contract["local_durable_evidence"]
    assert evidence["source"] == "final fresh verified StateStore snapshot"
    assert evidence["persisted"] is False
    assert evidence["equals_READY"] is False
    assert evidence["equals_RuntimeSession_history_publication"] is False


def test_corehost_recovery_complete_and_runtime_session_boundaries_are_exact() -> None:
    contract = startup_recovery_contract()
    complete = contract["recovery_complete"]
    assert complete["derivation"] == (
        "exact observations from production authority owners; never caller boolean"
    )
    assert len(complete["initialized_requires"]) == 11
    assert "no SecretHandoff UNKNOWN_RECONCILIATION exists" in complete["initialized_requires"]
    assert complete["any_false"] == "NO_READY_AND_FAIL_CLOSED"
    runtime = contract["runtime_session_boundary"]
    assert runtime["active_process"] == "EPHEMERAL RUNTIME created before StateStore open"
    assert runtime["durable_publication"] == "AFTER_RECOVERY_COMPLETE_BEFORE_READINESS"
    assert runtime["old_active_process_restored"] is False
    assert runtime["old_durable_history_preserved"] is True
    assert contract["readiness_boundary"]["READY_requires"] == (
        "successful recovery_complete plus existing later readiness gates"
    )


def test_corehost_startup_recovery_ownership_and_restore_separation_are_closed() -> None:
    contract = startup_recovery_contract()
    assert set(contract["ownership"]) == {
        "CoreHost",
        "SQLiteStateStore",
        "M0.3 ProtectedFreshness authority",
        "MigrationRegistry and migration coordinators",
        "SecretHandoff production orchestrator",
        "LocalDurableEvidenceRegistry",
        "TrustedPhysicalRestoreCoordinator",
        "M0.3 CoreHost readiness contract",
    }
    assert contract["caller_boolean_authority"] is False
    restore = contract["physical_restore_boundary"]
    assert restore["normal_startup_auto_restore"] is False
    assert restore["requires"] == "separate explicit restore operation authority and artifact"


def test_corehost_startup_recovery_failure_matrix_is_uniformly_fail_closed() -> None:
    rows = startup_recovery_contract()["failure_matrix"]
    assert [row["failure"] for row in rows] == [
        "STATESTORE_INTEGRITY_FAILURE",
        "COREHOST_SCOPE_MISMATCH",
        "PROTECTED_AUTHORITY_MISSING",
        "PROTECTED_AUTHORITY_MALFORMED",
        "PROTECTED_SCOPE_MISMATCH",
        "PROTECTED_PREPARED_UNRESOLVED",
        "MIGRATION_FAMILY_INVALID",
        "MIGRATION_AUTHORITY_MISSING",
        "MIGRATION_RECOVERY_FAILED",
        "SECRET_FAMILY_INVALID",
        "SECRET_RECONCILIATION_REQUIRED",
        "SECRET_RECOVERY_FAILED",
        "FINAL_VERIFIED_SNAPSHOT_FAILED",
        "FINAL_EVIDENCE_PUBLICATION_FAILED",
    ]
    for row in rows:
        assert row["READY"] == "FORBIDDEN"
        assert row["caller_bypass"] is False
        assert row["automatic_physical_restore"] is False
        assert row["corehost_cleanup"] == (
            "CLOSE_STATESTORE_CLOSE_RUNTIMESESSION_RELEASE_PROCESS_LOCK"
        )
        assert row["rollback_completed_external_effects"] is False


def test_process_role_authority_flags() -> None:
    data = load_contract()
    roles = roles_by_name(data)
    assert roles["core_host"]["owns_trading_state"] is True
    assert roles["core_host"]["may_access_exchange_api"] is True
    assert roles["core_host"]["may_survive_desktop_shell_exit"] is True
    assert roles["core_host"]["may_apply_runtime_mutations"] is True
    assert roles["core_host"]["may_request_runtime_mutations"] is False
    assert roles["core_host"]["may_issue_ipc_commands"] is False
    for client in ("desktop_shell", "tray_agent"):
        assert roles[client]["may_access_exchange_api"] is False
        assert roles[client]["owns_trading_state"] is False
        assert roles[client]["may_apply_runtime_mutations"] is False
        assert roles[client]["may_request_runtime_mutations"] is True
        assert roles[client]["may_issue_ipc_commands"] is True
    bootstrapper = roles["bootstrapper"]
    assert bootstrapper["may_apply_runtime_mutations"] is False
    assert bootstrapper["may_request_runtime_mutations"] is False
    assert bootstrapper["may_issue_ipc_commands"] is False
    assert bootstrapper["may_consume_maintenance_authorization"] is True
    for role in roles.values():
        assert "may_mutate_runtime" not in role
        assert "may_issue_runtime_commands" not in role
    invariants = "\n".join(data["invariants"])
    assert "only CoreHost applies mutable trading state mutations" in invariants
    assert "DesktopShell and TrayAgent may only send commands requesting mutations" in invariants
    assert "a correctly authenticated command may still be rejected by Core" in invariants
    assert "Bootstrapper does not send runtime or trading commands" in invariants


def test_state_axes_are_complete_disjoint_and_used_by_applicability() -> None:
    data = load_contract()
    process = set(data["process_health_states"])
    reachability = set(data["core_ipc_reachability_states"])
    observation = set(data["core_state_observation_sources"])
    confidence = set(data["core_state_confidence_states"])
    windows = set(data["desktop_window_states"])
    auth = set(data["operator_interface_authentication_states"])
    restart = set(data["supervision_restart_states"])
    assert process == {
        "NOT_STARTED",
        "STARTING",
        "HEALTHY",
        "DEGRADED",
        "STOPPING",
        "STOPPED",
        "CRASHED",
    }
    assert reachability == {"REACHABLE", "UNREACHABLE"}
    assert observation == {
        "CORE_HANDSHAKE",
        "PROCESS_LOCK",
        "CONNECTION_DESCRIPTOR",
        "TRAY_SUPERVISOR",
        "CACHED_SNAPSHOT",
        "NONE",
    }
    assert confidence == {"CONFIRMED_CURRENT", "STALE", "UNKNOWN"}
    assert windows == {"VISIBLE", "HIDDEN", "CLOSED"}
    assert auth == {"LOCKED", "AUTHENTICATED"}
    assert restart == {"NONE", "SCHEDULED", "IN_PROGRESS", "EXHAUSTED"}
    all_states = process | reachability | windows | auth | restart
    assert "UNLOCKED" not in all_states
    assert not {"REACHABLE", "UNREACHABLE"} & process
    assert not {"VISIBLE", "HIDDEN", "CLOSED"} & auth
    assert not {"LOCKED", "AUTHENTICATED"} & windows
    for intent in data["shutdown_intents"]:
        applicability = intent["applicability"]
        assert set(applicability["core_process_health_states"]) <= process
        assert set(applicability["core_ipc_reachability_states"]) <= reachability
        assert set(applicability["tray_process_health_states"]) <= process
        assert set(applicability["core_supervision_restart_states"]) <= restart
        assert set(applicability["core_state_observation_sources"]) <= observation
        assert set(applicability["core_state_confidence_states"]) <= confidence
        assert set(applicability["desktop_window_states"]) <= windows
        assert set(applicability["operator_authentication_states"]) <= auth


def test_intent_evaluation_pipeline_and_predicate_constraints() -> None:
    data = load_contract()
    assert [step["step_id"] for step in data["intent_evaluation_pipeline"]] == [
        "validate_trigger_kind_and_event_source",
        "validate_client_role",
        "validate_operator_interface_authentication",
        "validate_command_authorization_context",
        "validate_required_core_ipc_reachability",
        "match_exactly_one_state_case",
        "require_operator_acknowledgement_if_needed",
        "require_secondary_confirmation_if_needed",
        "submit_versioned_command_to_core",
        "core_revalidates_execution_authorization",
        "accepted_or_rejected_result_is_audited",
    ]
    rules = "\n".join(data["intent_evaluation_rules"])
    assert "allowed true does not bypass client role validation" in rules
    assert "allowed true does not bypass operator authentication" in rules
    assert "allowed true does not bypass authorization context" in rules
    assert "zero or multiple matching state_cases means reject" in rules
    constraints = data["state_predicate_constraints"]
    assert [
        "NO_ACTIVE_RUNTIME_CONFIRMED",
        "ACTIVE_RUNTIME_PRESENT",
        "RUNTIME_ACTIVITY_UNKNOWN",
    ] in constraints["mutually_exclusive_groups"]
    implications = {item["predicate"]: item for item in constraints["implications"]}
    assert set(
        implications["TRAY_PROCESS_CONFIRMED_RUNNING"]["requires_tray_process_health_states"]
    ) == {"HEALTHY", "DEGRADED"}
    assert set(implications["SUPERVISED_RESTART_PENDING"]["requires_core_lifecycle_pair_ids"]) == {
        "crashed_restart_scheduled",
        "supervised_restart_in_progress",
    }
    assert set(implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_lifecycle_pair_ids"]) == {
        "not_started",
        "stopped",
    }
    assert implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_state_confidence_states"] == [
        "CONFIRMED_CURRENT"
    ]
    assert set(
        implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_state_observation_sources"]
    ) == {"PROCESS_LOCK", "CONNECTION_DESCRIPTOR", "TRAY_SUPERVISOR"}
    assert {
        "not_started",
        "stopped",
        "crashed_unscheduled",
        "crashed_restart_scheduled",
        "crashed_restart_exhausted",
    } <= set(implications["ACTIVE_RUNTIME_PRESENT"]["forbidden_core_lifecycle_pair_ids"])
    assert implications["CORE_STATE_OBSERVATION_STALE"]["cannot_alone_confirm"] == [
        "NO_ACTIVE_RUNTIME_CONFIRMED"
    ]
    axis_rules = {
        rule["rule_id"]: rule for rule in data["state_axis_consistency_constraints"]["rules"]
    }
    assert axis_rules["core_ipc_reachable_requires_handshake_current"][
        "requires_core_state_observation_sources"
    ] == ["CORE_HANDSHAKE"]
    assert axis_rules["core_ipc_reachable_requires_handshake_current"][
        "requires_core_state_confidence_states"
    ] == ["CONFIRMED_CURRENT"]
    assert (
        "CORE_HANDSHAKE" in axis_rules["core_handshake_requires_reachable_current"]["when"].values()
    )
    assert axis_rules["core_ipc_unreachable_forbids_current_handshake"][
        "forbidden_core_state_observation_sources"
    ] == ["CORE_HANDSHAKE"]
    assert set(
        axis_rules["tray_supervisor_requires_running_tray_and_predicate"][
            "requires_tray_process_health_states"
        ]
    ) == {"HEALTHY", "DEGRADED"}
    assert axis_rules["none_source_requires_unknown"]["requires_core_state_confidence_states"] == [
        "UNKNOWN"
    ]


def test_requires_core_reachable_matches_reachability_axis() -> None:
    for intent in load_contract()["shutdown_intents"]:
        reachability = set(intent["applicability"]["core_ipc_reachability_states"])
        if intent["requires_core_reachable"] is True:
            assert reachability == {"REACHABLE"}, intent["name"]
            assert not any(
                "unreachable" in condition.lower() and "not" not in condition.lower()
                for condition in intent["applicability"]["conditions"]
            ), intent["name"]
        if intent["trigger_kind"] == "operating_system_event":
            assert reachability == {"REACHABLE", "UNREACHABLE"}


def test_process_lock_order_is_before_mutable_initialization() -> None:
    data = load_contract()
    steps = {step["step_id"]: step["order"] for step in data["startup_sequence"]}
    assert (
        steps["resolve_device_installation_and_state_store_identity"]
        < steps["acquire_local_process_lock"]
    )
    assert steps["acquire_local_process_lock"] < steps["create_runtime_session"]
    assert steps["acquire_local_process_lock"] < steps["open_state_store"]
    assert steps["handle_lock_busy"] < steps["create_runtime_session"]
    busy = next(step for step in data["startup_sequence"] if step["step_id"] == "handle_lock_busy")
    assert "do not open mutable state store" in busy["description"]
    assert "do not create RuntimeSession" in busy["description"]
    assert "do not initialize adapters" in busy["description"]
    invariants = "\n".join(data["invariants"])
    assert "process lock precedes mutable state-store open" in invariants
    assert "process lock precedes RuntimeSession creation" in invariants
    assert "losing process-lock contender performs no mutable initialization" in invariants
    assert "stale lock recovery requires PID/start nonce verification" in invariants


def test_close_window_tray_hud_and_shutdown_intents() -> None:
    data = load_contract()
    policy = data["window_close_policy"]["active_core_or_strategies"]
    assert policy["close_x_stops_core"] is False
    assert policy["default_safe_option"] == "background"
    assert policy["rememberable_options"] == ["background"]
    assert set(policy["options"]) == {
        "background",
        "stop_core_with_secondary_confirmation",
        "cancel",
    }
    assert policy["tray_agent_required_for_safe_hide"] is True
    assert data["hud_contract"]["read_only"] is True
    assert "full balances" in data["hud_contract"]["locked_hides"]


def test_shutdown_intents_have_complete_machine_contract() -> None:
    data = load_contract()
    expected = {
        "CLOSE_DESKTOP_SHELL",
        "HIDE_TO_BACKGROUND",
        "STOP_STRATEGIES",
        "PAUSE_STRATEGIES",
        "STOP_CORE_GRACEFULLY",
        "EXIT_TRAY_AGENT",
        "TRIGGER_KILL_SWITCH",
        "OS_SESSION_LOGOFF",
        "OS_SHUTDOWN",
        "ENTER_MAINTENANCE_MODE",
    }
    intents = intents_by_name(data)
    assert set(intents) == expected
    process_roles = set(roles_by_name(data))
    event_sources = set(data["external_event_sources"])
    for intent in intents.values():
        assert REQUIRED_SHUTDOWN_FIELDS <= intent.keys()
        assert {
            "trigger_kind",
            "applicability",
            "confirmation_policy",
            "operator_acknowledgement_policy",
            "event_sources",
            "handoff_clients",
            "evaluation_route",
            "requires_command_authorization_context",
            "requires_core_command_submission",
        } <= intent.keys()
        assert intent["trigger_kind"] in {
            "client_command",
            "operating_system_event",
            "supervised_lifecycle_event",
        }
        assert intent["description"]
        assert isinstance(intent["invariants"], list) and intent["invariants"]
        assert set(intent["allowed_client_roles"]) <= process_roles
        assert set(intent["event_sources"]) <= event_sources
        assert set(intent["applicability"]) == {
            "core_process_health_states",
            "core_ipc_reachability_states",
            "core_supervision_restart_states",
            "tray_process_health_states",
            "desktop_window_states",
            "operator_authentication_states",
            "conditions",
            "core_state_observation_sources",
            "core_state_confidence_states",
        }
        assert isinstance(intent["applicability"]["conditions"], list)
        assert set(intent["confirmation_policy"]) == {"mode", "conditions"}
        mode = intent["confirmation_policy"]["mode"]
        assert mode in {"never", "always", "conditional"}
        if mode == "always":
            assert intent["requires_secondary_confirmation"] is True
        if mode == "never":
            assert intent["requires_secondary_confirmation"] is False
        if mode == "conditional":
            assert intent["requires_secondary_confirmation"] is None
        acknowledgement = intent["operator_acknowledgement_policy"]
        assert set(acknowledgement) == {"mode", "conditions", "message_requirements"}
        assert acknowledgement["mode"] in {"never", "always", "conditional"}
        assert isinstance(acknowledgement["conditions"], list)
        assert isinstance(acknowledgement["message_requirements"], list)
        assert isinstance(intent["state_cases"], list) and intent["state_cases"]
        pair_ids = {pair["pair_id"] for pair in data["valid_core_lifecycle_pairs"]}
        predicates = {predicate["predicate_id"] for predicate in data["state_predicates"]}
        case_ids = set()
        for case in intent["state_cases"]:
            assert {
                "case_id",
                "case_kind",
                "core_lifecycle_pair_ids",
                "core_ipc_reachability_states",
                "core_state_observation_sources",
                "core_state_confidence_states",
                "tray_process_health_states",
                "allowed",
                "requires_operator_acknowledgement",
                "requires_secondary_confirmation",
                "required_predicates",
                "forbidden_predicates",
                "reason",
            } <= case.keys()
            assert case["case_kind"] in {"normal", "forbidden", "forbidden_catch_all"}
            assert (case["case_kind"] == "normal") == case["allowed"]
            if case["case_kind"] in {"forbidden", "forbidden_catch_all"}:
                assert case["allowed"] is False
            assert case["case_id"] not in case_ids
            case_ids.add(case["case_id"])
            assert set(case["core_lifecycle_pair_ids"]) <= pair_ids
            assert set(case["core_ipc_reachability_states"]) <= set(
                data["core_ipc_reachability_states"]
            )
            assert set(case["tray_process_health_states"]) <= set(data["process_health_states"])
            assert set(case["core_state_observation_sources"]) <= set(
                data["core_state_observation_sources"]
            )
            assert set(case["core_state_confidence_states"]) <= set(
                data["core_state_confidence_states"]
            )
            assert set(case["required_predicates"]) <= predicates
            assert set(case["forbidden_predicates"]) <= predicates
            assert isinstance(case["allowed"], bool)
            assert isinstance(case["requires_operator_acknowledgement"], bool)
            assert case["reason"]


def test_shutdown_intent_semantics() -> None:
    intents = intents_by_name(load_contract())
    close = intents["CLOSE_DESKTOP_SHELL"]
    assert close["stops_desktop_shell"] is True and close["stops_core"] is False
    assert close["stops_tray_agent"] is False and close["stops_strategies"] is False
    close_conditions = "\n".join(close["applicability"]["conditions"])
    assert (
        "silent ordinary close allowed only when Core process health is NOT_STARTED or STOPPED"
        in close_conditions
    )
    assert "silent ordinary close requires core supervision restart state NONE" in close_conditions
    acknowledgement = close["operator_acknowledgement_policy"]
    assert acknowledgement["mode"] == "conditional"
    assert "Core IPC reachability is UNREACHABLE" in acknowledgement["conditions"]
    assert "Core process health is CRASHED" in acknowledgement["conditions"]
    assert (
        "core supervision restart state is SCHEDULED or IN_PROGRESS"
        in acknowledgement["conditions"]
    )
    assert (
        "must not display Core stopped only because IPC is UNREACHABLE"
        in acknowledgement["message_requirements"]
    )
    background = intents["HIDE_TO_BACKGROUND"]
    assert background["stops_desktop_shell"] is True and background["stops_core"] is False
    assert "requires active TrayAgent" in "\n".join(background["invariants"])
    background_cases = {case["case_id"]: case for case in background["state_cases"]}
    assert set(background_cases["active_reachable_core"]["tray_process_health_states"]) == {
        "HEALTHY",
        "DEGRADED",
    }
    assert background["expected_core_result"] == "desktop_hidden_core_lifecycle_unchanged"
    assert "Core remains active" not in background["invariants"]
    assert "HIDE_TO_BACKGROUND does not change Core process health" in background["invariants"]
    assert (
        "HIDE_TO_BACKGROUND does not change supervision restart state" in background["invariants"]
    )
    background_conditions = "\n".join(background["applicability"]["conditions"])
    assert "TrayAgent must actually be running" in background_conditions
    assert "CRASHED is allowed only with a running TrayAgent" in background_conditions
    assert "SCHEDULED or IN_PROGRESS means Tray supervises restart" in background_conditions
    assert "EXHAUSTED is shown as operator-action-required state" in background_conditions
    assert "no silent hide when both Core and Tray are unreachable" in background_conditions
    assert background["operator_acknowledgement_policy"]["mode"] == "conditional"
    assert (
        "Core IPC reachability is UNREACHABLE"
        in background["operator_acknowledgement_policy"]["conditions"]
    )
    assert (
        "Core process health is CRASHED"
        in background["operator_acknowledgement_policy"]["conditions"]
    )
    assert (
        "core supervision restart state is SCHEDULED or IN_PROGRESS"
        in background["operator_acknowledgement_policy"]["conditions"]
    )
    stop_strategies = intents["STOP_STRATEGIES"]
    assert stop_strategies["stops_strategies"] is True and stop_strategies["stops_core"] is False
    assert stop_strategies["triggers_kill_switch"] is False
    pause = intents["PAUSE_STRATEGIES"]
    assert pause["blocks_new_order_intents"] is True and pause["stops_core"] is False
    graceful = intents["STOP_CORE_GRACEFULLY"]
    assert graceful["requires_core_reachable"] is True
    assert graceful["requires_secondary_confirmation"] is True
    assert graceful["confirmation_policy"]["mode"] == "always"
    assert graceful["blocks_new_order_intents"] is True
    assert graceful["persists_checkpoint"] is True
    assert graceful["triggers_kill_switch"] is False
    tray_exit = intents["EXIT_TRAY_AGENT"]
    assert tray_exit["stops_tray_agent"] is True and tray_exit["stops_core"] is False
    assert tray_exit["requires_secondary_confirmation"] is None
    assert tray_exit["confirmation_policy"]["mode"] == "conditional"
    exit_confirmation = "\n".join(tray_exit["confirmation_policy"]["conditions"])
    assert "core supervision restart state is NONE" in exit_confirmation
    assert (
        "secondary confirmation required when core supervision restart state is SCHEDULED or IN_PROGRESS"
        in exit_confirmation
    )
    assert (
        "secondary confirmation required for active, unknown, unreachable, or restarting Core"
        in exit_confirmation
    )
    assert (
        "no secondary confirmation only when Core process health is NOT_STARTED or STOPPED"
        in exit_confirmation
    )
    assert (
        "EXHAUSTED warning says Tray is needed to present operator-action-required state"
        in exit_confirmation
    )
    assert "cannot be remembered automatically" in "\n".join(tray_exit["invariants"])
    kill = intents["TRIGGER_KILL_SWITCH"]
    assert kill["triggers_kill_switch"] is True and kill["stops_core"] is False
    assert (
        kill["blocks_new_order_intents"] is True
        and kill["requires_operator_authentication"] is True
    )
    assert kill["requires_secondary_confirmation"] is None
    assert kill["confirmation_policy"]["mode"] == "conditional"
    assert "policy cannot be weakened by UI" in "\n".join(kill["confirmation_policy"]["conditions"])
    logoff = intents["OS_SESSION_LOGOFF"]
    assert "desktop_user_session does not guarantee continued operation" in "\n".join(
        logoff["invariants"]
    )
    shutdown = intents["OS_SHUTDOWN"]
    assert shutdown["blocks_new_order_intents"] is True and shutdown["persists_checkpoint"] is True
    assert "does not promise completion of all network operations" in "\n".join(
        shutdown["invariants"]
    )
    maintenance = intents["ENTER_MAINTENANCE_MODE"]
    assert maintenance["blocks_new_order_intents"] is True
    assert "STOPPED" not in maintenance["applicability"]["core_process_health_states"]
    assert maintenance["applicability"]["core_process_health_states"] == ["HEALTHY", "DEGRADED"]
    assert maintenance["applicability"]["core_ipc_reachability_states"] == ["REACHABLE"]
    assert "A stopped Core cannot acknowledge ENTER_MAINTENANCE_MODE" in "\n".join(
        maintenance["invariants"]
    )
    assert "does not reset kill switch" in "\n".join(maintenance["invariants"])


def test_maintenance_handoff_and_bootstrapper_consistency() -> None:
    data = load_contract()
    roles = roles_by_name(data)
    bootstrapper = roles["bootstrapper"]
    assert bootstrapper["may_issue_ipc_commands"] is False
    assert bootstrapper["may_request_runtime_mutations"] is False
    assert bootstrapper["may_apply_runtime_mutations"] is False
    assert bootstrapper["may_consume_maintenance_authorization"] is True
    maintenance = intents_by_name(data)["ENTER_MAINTENANCE_MODE"]
    assert "bootstrapper" not in maintenance["allowed_client_roles"]
    assert maintenance["handoff_clients"] == ["bootstrapper"]
    invariants = "\n".join(maintenance["invariants"])
    assert "DesktopShell or TrayAgent authorizes maintenance" in invariants
    assert "Core confirms entering maintenance" in invariants
    assert "Bootstrapper may execute update/restart workflow only from issued handoff" in invariants
    assert (
        "Bootstrapper cannot switch a running Core into maintenance without authorization"
        in invariants
    )


def test_system_events_are_not_client_roles() -> None:
    data = load_contract()
    assert set(data["external_event_sources"]) == {
        "operating_system",
        "windows_session_manager",
        "future_windows_service_manager",
    }
    intents = intents_by_name(data)
    for name in ("OS_SESSION_LOGOFF", "OS_SHUTDOWN"):
        intent = intents[name]
        assert intent["trigger_kind"] == "operating_system_event"
        assert intent["allowed_client_roles"] == []
        assert set(intent["event_sources"]) == {"operating_system", "windows_session_manager"}
    for name, intent in intents.items():
        if name not in {"OS_SESSION_LOGOFF", "OS_SHUTDOWN"}:
            assert intent["trigger_kind"] == "client_command"
            assert "os" not in intent["allowed_client_roles"]


def test_device_installation_id_is_canonical_identifier() -> None:
    data = load_contract()
    assert "device_installation_id" in data["discovery_contract"]["descriptor_may_contain"]
    assert "installation_id" not in data["discovery_contract"]["descriptor_may_contain"]
    assert "device_installation_id" in data["ipc_contract"]["handshake_fields"]
    raw = CONTRACT.read_text(encoding="utf-8")
    allowed_legacy_phrase = "M0.3 does not create an installation_id alias"
    without_allowed_legacy = raw.replace(allowed_legacy_phrase, "")
    assert not re.search(r"(?<!device_)installation_id", without_allowed_legacy)
    assert allowed_legacy_phrase in raw
    assert any(
        "all IPC, discovery and persistence references use the M0.2 device_installation_id name"
        == item
        for item in data["invariants"]
    )


def test_startup_readiness_states_and_first_run_gates() -> None:
    data = load_contract()
    states = {state["name"]: state for state in data["startup_readiness_states"]}
    assert set(states) == {
        "SETUP_REQUIRED",
        "CREDENTIALS_OPTIONAL",
        "RECONCILIATION_REQUIRED",
        "OPERATOR_ACTION_REQUIRED",
        "READY",
        "BLOCKED",
    }
    setup_rules = "\n".join(states["SETUP_REQUIRED"]["rules"])
    assert "no private exchange connections" in setup_rules
    assert "no strategies" in setup_rules
    assert "no order entry" in setup_rules
    assert "no exchange-secret loading" in setup_rules
    credentials = "\n".join(states["CREDENTIALS_OPTIONAL"]["rules"])
    assert "Paper may run without exchange accounts" in credentials
    assert "Testnet requires configured ExchangeAccount and credential reference" in credentials
    assert "strategies are not resumed" in "\n".join(states["RECONCILIATION_REQUIRED"]["rules"])
    assert any(
        "first-run incomplete never starts exchange-private connections or strategies" == item
        for item in data["invariants"]
    )


def test_autostart_policy() -> None:
    policy = load_contract()["autostart_policy"]
    assert set(policy) == {
        "start_core_on_logon",
        "start_tray_on_logon",
        "open_desktop_shell_on_logon",
        "show_hud_on_logon",
        "resume_runtime_after_reconciliation",
        "invariants",
    }
    for name in (
        "start_core_on_logon",
        "start_tray_on_logon",
        "open_desktop_shell_on_logon",
        "show_hud_on_logon",
        "resume_runtime_after_reconciliation",
    ):
        assert isinstance(policy[name]["default_enabled"], bool), name
        assert (
            isinstance(policy[name]["activation_conditions"], list)
            and policy[name]["activation_conditions"]
        ), name
    assert policy["start_core_on_logon"]["default_enabled"] is True
    assert policy["start_core_on_logon"]["activation_conditions"] == ["wizard_completed"]
    assert policy["start_tray_on_logon"]["default_enabled"] is True
    assert policy["open_desktop_shell_on_logon"]["default_enabled"] is False
    assert policy["show_hud_on_logon"]["default_enabled"] is False
    resume = policy["resume_runtime_after_reconciliation"]
    assert resume["default_enabled"] is False
    assert resume["user_choice_required"] is True
    assert set(resume["activation_conditions"]) == {
        "reconciliation_passed",
        "execution_lease_active",
        "capability_allowed",
        "policy_allowed",
        "kill_switch_not_triggered",
    }
    invariants = "\n".join(policy["invariants"])
    assert "autostart Core does not imply automatic strategy resume" in invariants
    assert (
        "resume requires reconciliation, lease, capability, policy and non-triggered kill switch"
        in invariants
    )
    assert "HUD requires TrayAgent" in invariants
    assert "open DesktopShell is not required for Core operation" in invariants
    assert "before wizard completion autostart does not start trading" in invariants


def test_deployment_mode_lifecycle_invariants() -> None:
    data = load_contract()
    modes = modes_by_name(data)
    assert (
        "does not promise operation after user logoff"
        in modes["desktop_user_session"]["non_guarantees"]
    )
    assert any("lock screen" in item for item in modes["desktop_user_session"]["guarantees"])
    assert any("compatible" in item for item in modes["windows_service"]["guarantees"])
    assert any("M0.2 domain identifiers" in item for item in data["invariants"])


def test_ipc_discovery_handshake_reconnect_and_commands() -> None:
    data = load_contract()
    ipc = data["ipc_contract"]
    assert ipc["canonical_logical_interface"] == "versioned Protobuf/gRPC"
    assert ipc["default_transport_scope"] == "local-only"
    required = {
        "protocol_version",
        "client_role",
        "client_version",
        "client_instance_id",
        "device_installation_id",
        "supported_capabilities",
        "requested_workspace_id",
        "authentication_authorization_reference",
        "core_runtime_session_id",
        "server_capabilities",
        "compatibility_result",
    }
    assert set(ipc["handshake_fields"]) == required
    reconnect = "\n".join(ipc["heartbeat_and_reconnect"])
    assert "full snapshot" in reconnect
    assert "command_id" in reconnect
    assert "not replayed" in reconnect
    descriptor_forbidden = set(data["discovery_contract"]["descriptor_must_not_contain"])
    assert {
        "api_keys",
        "api_secrets",
        "pin",
        "recovery_tokens",
        "biometric_data",
        "plaintext_operator_password",
    } <= descriptor_forbidden
    assert data["discovery_contract"]["secrets_in_command_line"] is False
    authority = "\n".join(data["authority_boundaries"])
    assert "CoreHost is the only authority for mutable trading state" in authority
    assert "GUI must not show command success before Core acknowledgement" in authority
    assert "command_id" in authority and "correlation_id" in authority


def test_windows_session_lock_contract_and_tray_exit_safety() -> None:
    data = load_contract()
    failures = {entry["failure"]: entry for entry in data["failure_matrix"]}
    lock = failures["Windows lock screen"]
    required = "\n".join(lock["required_behavior"])
    forbidden = "\n".join(lock["forbidden_behavior"])
    assert "CoreHost continues" in required
    assert "TrayAgent continues" in required
    assert "operator_interface_authentication = LOCKED" in required
    assert "sensitive views are masked immediately" in required
    assert "HUD enters locked visibility policy" in required
    assert "no new manual commands until CryptoHunter re-authentication" in required
    assert "Windows unlock alone does not automatically authenticate CryptoHunter" in required
    assert "treat Windows unlock alone as CryptoHunter authentication" in forbidden
    tray_rules = "\n".join(data["tray_contract"]["rules"])
    assert "EXIT_TRAY_AGENT requires secondary confirmation when Core is active" in tray_rules
    assert "Core remains running without tray icon and notifications" in tray_rules
    assert "EXIT_TRAY_AGENT can be cancelled by the user" in tray_rules
    assert "EXIT_TRAY_AGENT cannot be remembered as an automatic choice" in tray_rules
    assert "relaunching CryptoHunter should discover the existing Core" in tray_rules


def test_single_instance_failure_restart_security_and_evidence() -> None:
    data = load_contract()
    assert any(
        "local process lock blocks second Core" in item
        for item in data["single_instance_policy"]["core_host"]
    )
    assert any(
        "does not replace ExchangeAccount ExecutionLease" in item
        for item in data["single_instance_policy"]["core_host"]
    )
    failures = {entry["failure"]: entry for entry in data["failure_matrix"]}
    assert "stop Core" in "\n".join(failures["DesktopShell crash"]["forbidden_behavior"])
    assert "stop Core" in "\n".join(failures["TrayAgent crash"]["forbidden_behavior"])
    assert any("reconciles" in item for item in failures["CoreHost crash"]["recovery_behavior"])
    restart_rules = "\n".join(data["restart_policy"]["rules"])
    assert "no automatic kill switch reset" in restart_rules
    assert "no silent fallback" in restart_rules
    assert "no secrets in command line" in data["security_rules"]
    for path in data["evidence_paths"]:
        evidence = Path(path)
        assert not evidence.is_absolute(), path
        assert (ROOT / evidence).exists(), path


def test_shutdown_intent_state_cases_are_authoritative() -> None:
    data = load_contract()
    invariants = "\n".join(data["invariants"])
    assert (
        "implementation must not interpret applicability as a free Cartesian product" in invariants
    )
    assert "lifecycle applicability is determined by exactly one matching state_case" in invariants
    assert (
        "final intent executability requires all global role, authentication, authorization, trigger and Core gates"
        in invariants
    )
    assert "missing matching state_case means intent is forbidden and fail-closed" in invariants
    assert (
        "more than one matching state_case is a contract error, not a priority mechanism"
        in invariants
    )
    intents = intents_by_name(data)
    close_cases = {case["case_id"]: case for case in intents["CLOSE_DESKTOP_SHELL"]["state_cases"]}
    assert {
        "silent_confirmed_inactive",
        "unreachable_stale_snapshot",
        "unreachable_no_observation",
        "crashed_unscheduled",
        "restarting_with_tray_available_independent_current",
        "restarting_with_tray_available_tray_supervisor",
        "restarting_with_tray_available_ipc_unreachable",
        "restarting_with_tray_available_ipc_reachable",
        "restarting_without_tray",
        "restarting_without_tray_scheduled",
        "restart_exhausted",
        "active_reachable_core",
    } <= set(close_cases)
    assert close_cases["silent_confirmed_inactive"]["core_lifecycle_pair_ids"] == [
        "not_started",
        "stopped",
    ]
    assert set(close_cases["silent_confirmed_inactive"]["required_predicates"]) == {
        "NO_ACTIVE_RUNTIME_CONFIRMED",
        "INDEPENDENT_OBSERVATION_PROOF_VALIDATED",
    }
    assert close_cases["unreachable_stale_snapshot"]["requires_operator_acknowledgement"] is True
    assert (
        "RUNTIME_ACTIVITY_UNKNOWN"
        in close_cases["unreachable_stale_snapshot"]["required_predicates"]
    )
    assert close_cases["restarting_with_tray_available_independent_current"][
        "core_lifecycle_pair_ids"
    ] == ["crashed_restart_scheduled"]
    assert close_cases["restarting_with_tray_available_independent_current"]["allowed"] is False
    assert close_cases["restarting_with_tray_available_ipc_unreachable"][
        "core_lifecycle_pair_ids"
    ] == ["supervised_restart_in_progress"]
    assert close_cases["restarting_with_tray_available_ipc_reachable"][
        "core_ipc_reachability_states"
    ] == ["REACHABLE"]
    assert close_cases["restarting_without_tray"]["allowed"] is False
    assert (
        "TRAY_PROCESS_CONFIRMED_RUNNING"
        in close_cases["restarting_without_tray"]["forbidden_predicates"]
    )
    assert close_cases["active_reachable_core"]["allowed"] is False

    hide = intents["HIDE_TO_BACKGROUND"]
    assert "Core and Tray remain active" not in hide["description"]
    assert "Core remains active" not in hide["invariants"]
    hide_cases = {case["case_id"]: case for case in hide["state_cases"]}
    assert {
        "active_reachable_core",
        "unreachable_stale_snapshot",
        "unreachable_no_observation",
        "crashed_unscheduled_independent_current",
        "crashed_unscheduled_tray_supervisor",
        "crashed_restart_scheduled_independent_current",
        "crashed_restart_scheduled_tray_supervisor",
        "restart_in_progress_ipc_unreachable",
        "restart_in_progress_ipc_reachable",
        "restart_exhausted_independent_current",
        "restart_exhausted_tray_supervisor",
        "tray_unavailable_reachable_core",
        "tray_unavailable_independent_current",
        "tray_unavailable_cached_stale",
        "tray_unavailable_no_observation",
    } <= set(hide_cases)
    assert hide_cases["active_reachable_core"]["core_lifecycle_pair_ids"] == [
        "ordinary_starting",
        "healthy",
        "degraded",
        "stopping",
    ]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_lifecycle_pair_ids"] == [
        "supervised_restart_in_progress"
    ]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_ipc_reachability_states"] == [
        "UNREACHABLE"
    ]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_state_observation_sources"] == [
        "TRAY_SUPERVISOR"
    ]
    assert hide_cases["restart_in_progress_ipc_reachable"]["core_ipc_reachability_states"] == [
        "REACHABLE"
    ]
    assert hide_cases["restart_in_progress_ipc_reachable"]["core_state_observation_sources"] == [
        "CORE_HANDSHAKE"
    ]
    assert (
        hide_cases["restart_in_progress_ipc_reachable"]["requires_operator_acknowledgement"] is True
    )
    assert (
        "TRAY_PROCESS_CONFIRMED_RUNNING"
        in hide_cases["restart_in_progress_ipc_reachable"]["required_predicates"]
    )
    assert hide_cases["restart_exhausted_independent_current"]["core_lifecycle_pair_ids"] == [
        "crashed_restart_exhausted"
    ]
    assert (
        "OPERATOR_ACTION_REQUIRED" in hide_cases["restart_exhausted_independent_current"]["reason"]
    )
    assert hide_cases["tray_unavailable_reachable_core"]["allowed"] is False

    for name in [
        "STOP_STRATEGIES",
        "PAUSE_STRATEGIES",
        "STOP_CORE_GRACEFULLY",
        "TRIGGER_KILL_SWITCH",
        "ENTER_MAINTENANCE_MODE",
    ]:
        cases = intents[name]["state_cases"]
        pairs = {pair["pair_id"]: pair for pair in data["valid_core_lifecycle_pairs"]}
        assert cases and all(
            pairs[pair_id]["core_supervision_restart_state"] == "NONE"
            for case in cases
            for pair_id in case["core_lifecycle_pair_ids"]
        ), name
        assert all(case["core_ipc_reachability_states"] == ["REACHABLE"] for case in cases), name
        assert all(
            case["core_state_observation_sources"] == ["CORE_HANDSHAKE"] for case in cases
        ), name
        assert all(
            case["core_state_confidence_states"] == ["CONFIRMED_CURRENT"] for case in cases
        ), name

    exit_cases = {case["case_id"]: case for case in intents["EXIT_TRAY_AGENT"]["state_cases"]}
    assert {
        "inactive_core",
        "unreachable_stale_snapshot",
        "unreachable_no_observation",
        "active_core",
        "crashed_unscheduled",
        "crashed_restart_scheduled",
        "restart_in_progress_ipc_unreachable",
        "restart_in_progress_ipc_reachable",
        "restart_exhausted",
    } <= set(exit_cases)
    assert exit_cases["inactive_core"]["requires_secondary_confirmation"] is False
    assert exit_cases["inactive_core"]["core_ipc_reachability_states"] == ["UNREACHABLE"]
    assert exit_cases["unreachable_stale_snapshot"]["requires_secondary_confirmation"] is True
    assert (
        "NO_ACTIVE_RUNTIME_CONFIRMED"
        in exit_cases["unreachable_stale_snapshot"]["forbidden_predicates"]
    )


def test_restart_policy_supervision_state_transitions_and_readiness_mapping() -> None:
    data = load_contract()
    expected_pairs = {
        ("not_started", "NOT_STARTED", "NONE"),
        ("ordinary_starting", "STARTING", "NONE"),
        ("supervised_restart_in_progress", "STARTING", "IN_PROGRESS"),
        ("healthy", "HEALTHY", "NONE"),
        ("degraded", "DEGRADED", "NONE"),
        ("stopping", "STOPPING", "NONE"),
        ("stopped", "STOPPED", "NONE"),
        ("crashed_unscheduled", "CRASHED", "NONE"),
        ("crashed_restart_scheduled", "CRASHED", "SCHEDULED"),
        ("crashed_restart_exhausted", "CRASHED", "EXHAUSTED"),
    }
    actual_pairs = {
        (p["pair_id"], p["core_process_health_state"], p["core_supervision_restart_state"])
        for p in data["valid_core_lifecycle_pairs"]
    }
    assert actual_pairs == expected_pairs
    assert not any(
        health == "CRASHED" and restart == "IN_PROGRESS" for _, health, restart in actual_pairs
    )
    restart_states = set(data["supervision_restart_states"])
    process_states = set(data["process_health_states"])
    pair_ids = {p["pair_id"] for p in data["valid_core_lifecycle_pairs"]}
    transitions = data["restart_policy"]["supervision_restart_state_transitions"]
    observed = set()
    for item in transitions:
        assert {
            "from_restart_state",
            "required_core_process_health_states",
            "from_core_lifecycle_pair_ids",
            "event",
            "to_restart_state",
            "resulting_core_process_health_states",
            "to_core_lifecycle_pair_ids",
        } <= item.keys()
        assert item["from_restart_state"] in restart_states
        assert item["to_restart_state"] in restart_states
        assert item["from_restart_state"] != "CRASHED"
        assert item["to_restart_state"] != "CRASHED"
        assert set(item["required_core_process_health_states"]) <= process_states
        assert set(item["resulting_core_process_health_states"]) <= process_states
        assert set(item["from_core_lifecycle_pair_ids"]) <= pair_ids
        assert set(item["to_core_lifecycle_pair_ids"]) <= pair_ids
        observed.add(
            (
                item["from_restart_state"],
                tuple(item["required_core_process_health_states"]),
                item["event"],
                item["to_restart_state"],
                tuple(item["resulting_core_process_health_states"]),
            )
        )
    assert ("NONE", ("CRASHED",), "crash_detected", "SCHEDULED", ("CRASHED",)) in observed
    assert (
        "SCHEDULED",
        ("CRASHED",),
        "backoff_elapsed_restart_attempt_begins",
        "IN_PROGRESS",
        ("STARTING",),
    ) in observed
    assert (
        "IN_PROGRESS",
        ("STARTING",),
        "core_started_successfully",
        "NONE",
        ("HEALTHY", "DEGRADED"),
    ) in observed
    assert (
        "IN_PROGRESS",
        ("STARTING",),
        "restart_attempt_failed",
        "SCHEDULED",
        ("CRASHED",),
    ) in observed
    assert any(item[3] == "EXHAUSTED" for item in observed)
    assert data["restart_policy"]["readiness_mapping"] == {"EXHAUSTED": "OPERATOR_ACTION_REQUIRED"}


def _case_matches(
    case: dict,
    pair_id: str,
    ipc_reachability: str,
    observation_source: str,
    confidence: str,
    tray_health: str,
    predicates: set[str],
) -> bool:
    return (
        pair_id in case["core_lifecycle_pair_ids"]
        and ipc_reachability in case["core_ipc_reachability_states"]
        and observation_source in case["core_state_observation_sources"]
        and confidence in case["core_state_confidence_states"]
        and tray_health in case["tray_process_health_states"]
        and set(case["required_predicates"]) <= predicates
        and not (set(case["forbidden_predicates"]) & predicates)
    )


def _powerset(items: list[str]) -> list[set[str]]:
    return [set(combo) for size in range(len(items) + 1) for combo in combinations(items, size)]


RULE_FIELDS = {
    "rule_id",
    "when",
    "requires_core_ipc_reachability_states",
    "forbidden_core_ipc_reachability_states",
    "requires_core_state_observation_sources",
    "forbidden_core_state_observation_sources",
    "allowed_core_state_confidence_states",
    "requires_core_state_confidence_states",
    "forbidden_core_state_confidence_states",
    "requires_tray_process_health_states",
    "forbidden_tray_process_health_states",
    "requires_core_lifecycle_pair_ids",
    "forbidden_core_lifecycle_pair_ids",
    "requires_predicates",
    "forbidden_predicates",
}
CONTEXT_FIELD_BY_AXIS = {
    "core_lifecycle_pair_id": "pair_id",
    "core_ipc_reachability_state": "ipc_reachability",
    "core_state_observation_source": "observation_source",
    "core_state_confidence_state": "confidence",
    "tray_process_health_state": "tray_health",
}
RULE_FIELD_TO_CONTEXT = {
    "core_ipc_reachability_states": "ipc_reachability",
    "core_state_observation_sources": "observation_source",
    "core_state_confidence_states": "confidence",
    "tray_process_health_states": "tray_health",
    "core_lifecycle_pair_ids": "pair_id",
}


def _context(
    pair_id: str,
    ipc_reachability: str,
    observation_source: str,
    confidence: str,
    tray_health: str,
    predicates: set[str],
) -> dict:
    return {
        "pair_id": pair_id,
        "ipc_reachability": ipc_reachability,
        "observation_source": observation_source,
        "confidence": confidence,
        "tray_health": tray_health,
        "predicates": predicates,
    }


_VALIDATED_CONTRACT_IDS: set[int] = set()


def _validate_constraint_references(data: dict) -> None:
    if id(data) in _VALIDATED_CONTRACT_IDS:
        return
    pair_ids = {pair["pair_id"] for pair in data["valid_core_lifecycle_pairs"]}
    predicates = {predicate["predicate_id"] for predicate in data["state_predicates"]}
    observation = set(data["core_state_observation_sources"])
    confidence = set(data["core_state_confidence_states"])
    reachability = set(data["core_ipc_reachability_states"])
    tray = set(data["process_health_states"])
    allowed_values = {
        "core_lifecycle_pair_ids": pair_ids,
        "predicates": predicates,
        "core_state_observation_sources": observation,
        "core_state_confidence_states": confidence,
        "core_ipc_reachability_states": reachability,
        "tray_process_health_states": tray,
    }
    seen_rule_ids: set[str] = set()
    for rule in data["state_axis_consistency_constraints"]["rules"]:
        assert set(rule) <= RULE_FIELDS, rule
        assert isinstance(rule["rule_id"], str) and rule["rule_id"]
        assert rule["rule_id"] not in seen_rule_ids
        seen_rule_ids.add(rule["rule_id"])
        when = rule.get("when", {})
        assert isinstance(when, dict)
        when_allowed_values = {
            "core_lifecycle_pair_id": pair_ids,
            "core_ipc_reachability_state": reachability,
            "core_state_observation_source": observation,
            "core_state_confidence_state": confidence,
            "tray_process_health_state": tray,
            "predicate": predicates,
        }
        for axis, value in when.items():
            assert axis in CONTEXT_FIELD_BY_AXIS or axis == "predicate", (rule["rule_id"], axis)
            assert isinstance(value, str), (rule["rule_id"], axis, value)
            assert value in when_allowed_values[axis], (rule["rule_id"], axis, value)
        for field, values in rule.items():
            if field in {"rule_id", "when"}:
                continue
            assert isinstance(values, list) and all(isinstance(v, str) for v in values), (
                rule["rule_id"],
                field,
            )
            suffix = (
                field.replace("requires_", "").replace("forbidden_", "").replace("allowed_", "")
            )
            assert suffix in allowed_values, (rule["rule_id"], field)
            assert set(values) <= allowed_values[suffix], (rule["rule_id"], field, values)
    constraints = data["state_predicate_constraints"]
    for mapping_name in [
        "required_predicates_by_lifecycle_pair",
        "forbidden_predicates_by_lifecycle_pair",
    ]:
        for pair_id, values in constraints[mapping_name].items():
            assert pair_id in pair_ids
            assert isinstance(values, list) and set(values) <= predicates
    proof = data["independent_observation_proof_contract"]
    assert set(proof) == {
        "predicate",
        "required_checks",
        "allowed_observation_sources",
        "forbidden_observation_sources",
    }
    assert proof["predicate"] in predicates
    assert set(proof["required_checks"]) == {
        "PID_VALIDATED",
        "START_NONCE_VALIDATED",
        "SOURCE_FRESHNESS_VALIDATED",
        "DEVICE_INSTALLATION_IDENTITY_MATCHED",
        "STATE_STORE_IDENTITY_MATCHED",
    }
    allowed_sources = set(proof["allowed_observation_sources"])
    forbidden_sources = set(proof["forbidden_observation_sources"])
    assert allowed_sources <= observation
    assert forbidden_sources <= observation
    assert not allowed_sources & forbidden_sources
    assert allowed_sources | forbidden_sources == observation
    _VALIDATED_CONTRACT_IDS.add(id(data))


def _rule_when_matches(rule: dict, ctx: dict) -> bool:
    for axis, expected in rule.get("when", {}).items():
        if axis == "predicate":
            if expected not in ctx["predicates"]:
                return False
        elif ctx[CONTEXT_FIELD_BY_AXIS[axis]] != expected:
            return False
    return True


def _rule_allows(rule: dict, ctx: dict) -> bool:
    if not _rule_when_matches(rule, ctx):
        return True
    for field, values in rule.items():
        if field in {"rule_id", "when"}:
            continue
        if field in {"requires_predicates"} and not set(values) <= ctx["predicates"]:
            return False
        if field in {"forbidden_predicates"} and set(values) & ctx["predicates"]:
            return False
        if field.startswith("requires_") and field != "requires_predicates":
            axis = field.removeprefix("requires_")
            if ctx[RULE_FIELD_TO_CONTEXT[axis]] not in values:
                return False
        if field.startswith("forbidden_") and field != "forbidden_predicates":
            axis = field.removeprefix("forbidden_")
            if ctx[RULE_FIELD_TO_CONTEXT[axis]] in values:
                return False
        if field.startswith("allowed_"):
            axis = field.removeprefix("allowed_")
            if ctx[RULE_FIELD_TO_CONTEXT[axis]] not in values:
                return False
    return True


def _semantically_valid_combination(
    data: dict,
    pair_id: str,
    ipc_reachability: str,
    observation_source: str,
    confidence: str,
    tray_health: str,
    predicates: set[str],
) -> bool:
    ctx = _context(
        pair_id, ipc_reachability, observation_source, confidence, tray_health, predicates
    )
    _validate_constraint_references(data)
    for rule in data["state_axis_consistency_constraints"]["rules"]:
        if not _rule_allows(rule, ctx):
            return False
    constraints = data["state_predicate_constraints"]
    for pair, required in constraints["required_predicates_by_lifecycle_pair"].items():
        if pair_id == pair and not set(required) <= predicates:
            return False
    for pair, forbidden in constraints["forbidden_predicates_by_lifecycle_pair"].items():
        if pair_id == pair and set(forbidden) & predicates:
            return False
    proof = data["independent_observation_proof_contract"]
    proof_predicate = proof["predicate"]
    if (
        proof_predicate in predicates
        and observation_source in proof["forbidden_observation_sources"]
    ):
        return False
    if (
        observation_source in proof["allowed_observation_sources"]
        and confidence == "CONFIRMED_CURRENT"
        and proof_predicate not in predicates
    ):
        return False
    for group in constraints["mutually_exclusive_groups"]:
        if len(set(group) & predicates) > 1:
            return False
    for implication in constraints["implications"]:
        if implication["predicate"] not in predicates:
            continue
        if (
            "requires_core_lifecycle_pair_ids" in implication
            and pair_id not in implication["requires_core_lifecycle_pair_ids"]
        ):
            return False
        if (
            "forbidden_core_lifecycle_pair_ids" in implication
            and pair_id in implication["forbidden_core_lifecycle_pair_ids"]
        ):
            return False
        if (
            "requires_core_state_confidence_states" in implication
            and confidence not in implication["requires_core_state_confidence_states"]
        ):
            return False
        if (
            "requires_core_state_observation_sources" in implication
            and observation_source not in implication["requires_core_state_observation_sources"]
        ):
            return False
        if (
            "requires_tray_process_health_states" in implication
            and tray_health not in implication["requires_tray_process_health_states"]
        ):
            return False
    for invalid in constraints["invalid_combinations"]:
        pair_matches = (
            "core_lifecycle_pair_ids" not in invalid
            or pair_id in invalid["core_lifecycle_pair_ids"]
        )
        ipc_matches = (
            "core_ipc_reachability_states" not in invalid
            or ipc_reachability in invalid["core_ipc_reachability_states"]
        )
        observation_matches = (
            "core_state_observation_sources" not in invalid
            or observation_source in invalid["core_state_observation_sources"]
        )
        confidence_matches = (
            "core_state_confidence_states" not in invalid
            or confidence in invalid["core_state_confidence_states"]
        )
        if pair_matches and ipc_matches and observation_matches and confidence_matches:
            return False
    return True


def test_state_cases_are_deterministic_for_enumerated_combinations() -> None:
    data = load_contract()
    predicates = [predicate["predicate_id"] for predicate in data["state_predicates"]]
    predicate_sets = _powerset(predicates)
    saw_multi_predicate_case = False
    witnessed_cases: set[tuple[str, str]] = set()
    for intent in data["shutdown_intents"]:
        for pair in data["valid_core_lifecycle_pairs"]:
            for reachability in data["core_ipc_reachability_states"]:
                for observation_source in data["core_state_observation_sources"]:
                    for confidence in data["core_state_confidence_states"]:
                        for tray_health in data["process_health_states"]:
                            for active_predicates in predicate_sets:
                                if not _semantically_valid_combination(
                                    data,
                                    pair["pair_id"],
                                    reachability,
                                    observation_source,
                                    confidence,
                                    tray_health,
                                    active_predicates,
                                ):
                                    continue
                                if len(active_predicates) >= 2:
                                    saw_multi_predicate_case = True
                                matches = [
                                    case
                                    for case in intent["state_cases"]
                                    if _case_matches(
                                        case,
                                        pair["pair_id"],
                                        reachability,
                                        observation_source,
                                        confidence,
                                        tray_health,
                                        active_predicates,
                                    )
                                ]
                                for case in matches:
                                    witnessed_cases.add((intent["name"], case["case_id"]))
                                assert len(matches) <= 1, (
                                    intent["name"],
                                    pair["pair_id"],
                                    reachability,
                                    observation_source,
                                    confidence,
                                    tray_health,
                                    active_predicates,
                                    [case["case_id"] for case in matches],
                                )
    for intent in data["shutdown_intents"]:
        for case in intent["state_cases"]:
            is_forbidden_catch_all = case["case_kind"] == "forbidden_catch_all"
            assert (intent["name"], case["case_id"]) in witnessed_cases or is_forbidden_catch_all, (
                intent["name"],
                case["case_id"],
            )
    assert saw_multi_predicate_case
    assert not _semantically_valid_combination(
        data, "stopped", "REACHABLE", "CORE_HANDSHAKE", "CONFIRMED_CURRENT", "HEALTHY", set()
    )
    assert not _semantically_valid_combination(
        data,
        "crashed_unscheduled",
        "REACHABLE",
        "CORE_HANDSHAKE",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        set(),
    )
    close_cases = intents_by_name(data)["CLOSE_DESKTOP_SHELL"]["state_cases"]
    stopped_confirmed = [
        case["case_id"]
        for case in close_cases
        if _case_matches(
            case,
            "stopped",
            "UNREACHABLE",
            "PROCESS_LOCK",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            {"NO_ACTIVE_RUNTIME_CONFIRMED", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
        )
    ]
    assert stopped_confirmed == ["silent_confirmed_inactive"]
    stopped_stale = [
        case["case_id"]
        for case in close_cases
        if _case_matches(
            case,
            "stopped",
            "UNREACHABLE",
            "CACHED_SNAPSHOT",
            "STALE",
            "HEALTHY",
            {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
        )
    ]
    assert stopped_stale == ["unreachable_stale_snapshot"]
    exit_cases = intents_by_name(data)["EXIT_TRAY_AGENT"]["state_cases"]
    stopped_unreachable_with_no_active_runtime = [
        case["case_id"]
        for case in exit_cases
        if _case_matches(
            case,
            "stopped",
            "UNREACHABLE",
            "PROCESS_LOCK",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            {"NO_ACTIVE_RUNTIME_CONFIRMED", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
        )
    ]
    assert stopped_unreachable_with_no_active_runtime == ["inactive_core"]
    stopped_unreachable_unknown = [
        case["case_id"]
        for case in exit_cases
        if _case_matches(
            case,
            "stopped",
            "UNREACHABLE",
            "CACHED_SNAPSHOT",
            "STALE",
            "HEALTHY",
            {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
        )
    ]
    assert stopped_unreachable_unknown == ["unreachable_stale_snapshot"]
    hide_cases = intents_by_name(data)["HIDE_TO_BACKGROUND"]["state_cases"]
    matches = [
        case
        for case in hide_cases
        if _case_matches(
            case,
            "supervised_restart_in_progress",
            "REACHABLE",
            "CORE_HANDSHAKE",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            {"TRAY_PROCESS_CONFIRMED_RUNNING", "SUPERVISED_RESTART_PENDING"},
        )
    ]
    assert len(matches) == 1
    assert matches[0]["case_id"] == "restart_in_progress_ipc_reachable"
    assert matches[0]["allowed"] is True
    assert matches[0]["requires_operator_acknowledgement"] is True
    unreachable_matches = [
        case
        for case in hide_cases
        if _case_matches(
            case,
            "supervised_restart_in_progress",
            "UNREACHABLE",
            "TRAY_SUPERVISOR",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            {"TRAY_PROCESS_CONFIRMED_RUNNING", "SUPERVISED_RESTART_PENDING"},
        )
    ]
    assert [case["case_id"] for case in unreachable_matches] == [
        "restart_in_progress_ipc_unreachable"
    ]
    for intent in data["shutdown_intents"]:
        for case in intent["state_cases"]:
            if case["requires_operator_acknowledgement"] or case["allowed"]:
                assert (
                    case["required_predicates"]
                    or case["forbidden_predicates"]
                    or (
                        case["core_state_observation_sources"]
                        and case["core_state_confidence_states"]
                    )
                    or intent["trigger_kind"] == "operating_system_event"
                ), (intent["name"], case["case_id"])


def test_restart_provenance_and_observation_proof_regressions() -> None:
    data = load_contract()
    intents = intents_by_name(data)
    assert not _semantically_valid_combination(
        data,
        "supervised_restart_in_progress",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "STALE",
        "STOPPED",
        {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
    )
    close_cases = intents["CLOSE_DESKTOP_SHELL"]["state_cases"]
    assert not any(
        case["allowed"]
        and _case_matches(
            case,
            "supervised_restart_in_progress",
            "UNREACHABLE",
            "CACHED_SNAPSHOT",
            "STALE",
            "STOPPED",
            {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
        )
        for case in close_cases
    )
    generic_close = next(
        case for case in close_cases if case["case_id"] == "unreachable_stale_snapshot"
    )
    assert not _case_matches(
        generic_close,
        "supervised_restart_in_progress",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "STALE",
        "STOPPED",
        {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
    )

    scheduled_without_tray = {
        "SUPERVISED_RESTART_PENDING",
        "RUNTIME_ACTIVITY_UNKNOWN",
        "CORE_STATE_OBSERVATION_STALE",
    }
    assert _semantically_valid_combination(
        data,
        "crashed_restart_scheduled",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "STALE",
        "STOPPED",
        scheduled_without_tray,
    )
    matches = [
        case
        for case in close_cases
        if _case_matches(
            case,
            "crashed_restart_scheduled",
            "UNREACHABLE",
            "CACHED_SNAPSHOT",
            "STALE",
            "STOPPED",
            scheduled_without_tray,
        )
    ]
    assert [case["case_id"] for case in matches] == ["restarting_without_tray_scheduled_stale"]
    assert not any(case["allowed"] for case in matches)
    assert generic_close["case_id"] not in [case["case_id"] for case in matches]

    assert not _semantically_valid_combination(
        data,
        "stopped",
        "UNREACHABLE",
        "PROCESS_LOCK",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"NO_ACTIVE_RUNTIME_CONFIRMED"},
    )
    proof_predicates = {"NO_ACTIVE_RUNTIME_CONFIRMED", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"}
    assert _semantically_valid_combination(
        data,
        "stopped",
        "UNREACHABLE",
        "PROCESS_LOCK",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        proof_predicates,
    )
    proof_matches = [
        case
        for case in close_cases
        if _case_matches(
            case,
            "stopped",
            "UNREACHABLE",
            "PROCESS_LOCK",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            proof_predicates,
        )
    ]
    assert [case["case_id"] for case in proof_matches] == ["silent_confirmed_inactive"]

    assert not _semantically_valid_combination(
        data,
        "stopped",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"CORE_STATE_OBSERVATION_STALE"},
    )
    for intent_name, case_id in [
        ("CLOSE_DESKTOP_SHELL", "unreachable_stale_snapshot"),
        ("CLOSE_DESKTOP_SHELL", "unreachable_no_observation"),
        ("HIDE_TO_BACKGROUND", "unreachable_stale_snapshot"),
        ("HIDE_TO_BACKGROUND", "unreachable_no_observation"),
        ("EXIT_TRAY_AGENT", "unreachable_stale_snapshot"),
        ("EXIT_TRAY_AGENT", "unreachable_no_observation"),
    ]:
        case = next(
            case for case in intents[intent_name]["state_cases"] if case["case_id"] == case_id
        )
        assert "crashed_restart_scheduled" not in case["core_lifecycle_pair_ids"]
        assert "supervised_restart_in_progress" not in case["core_lifecycle_pair_ids"]
    assert not _semantically_valid_combination(
        data,
        "stopped",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "STALE",
        "HEALTHY",
        {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED", "CORE_STATE_OBSERVATION_STALE"},
    )
    assert not _semantically_valid_combination(
        data,
        "healthy",
        "REACHABLE",
        "CORE_HANDSHAKE",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"SUPERVISED_RESTART_PENDING"},
    )


def _matching_witness_predicate_sets(
    data: dict,
    case: dict,
    pair_id: str,
    reachability: str,
    observation_source: str,
    confidence: str,
    tray_health: str,
) -> list[set[str]]:
    predicates = [predicate["predicate_id"] for predicate in data["state_predicates"]]
    return [
        active_predicates
        for active_predicates in _powerset(predicates)
        if _semantically_valid_combination(
            data,
            pair_id,
            reachability,
            observation_source,
            confidence,
            tray_health,
            active_predicates,
        )
        and _case_matches(
            case,
            pair_id,
            reachability,
            observation_source,
            confidence,
            tray_health,
            active_predicates,
        )
    ]


def test_state_cases_have_no_dead_axis_variants() -> None:
    data = load_contract()
    witnessed_cases: set[tuple[str, str]] = set()
    witnessed_axes: dict[tuple[str, str], dict[str, set[str]]] = {}
    for intent in data["shutdown_intents"]:
        for case in intent["state_cases"]:
            if case["case_kind"] == "forbidden_catch_all":
                continue
            key = (intent["name"], case["case_id"])
            witnessed_axes[key] = {
                "pairs": set(),
                "reachability": set(),
                "observations": set(),
                "confidence": set(),
                "tray": set(),
            }
            for pair_id in case["core_lifecycle_pair_ids"]:
                for reachability in case["core_ipc_reachability_states"]:
                    for observation_source in case["core_state_observation_sources"]:
                        for confidence in case["core_state_confidence_states"]:
                            for tray_health in case["tray_process_health_states"]:
                                witnesses = _matching_witness_predicate_sets(
                                    data,
                                    case,
                                    pair_id,
                                    reachability,
                                    observation_source,
                                    confidence,
                                    tray_health,
                                )
                                assert witnesses, (
                                    intent["name"],
                                    case["case_id"],
                                    pair_id,
                                    reachability,
                                    observation_source,
                                    confidence,
                                    tray_health,
                                )
                                witnessed_cases.add(key)
                                witnessed_axes[key]["pairs"].add(pair_id)
                                witnessed_axes[key]["reachability"].add(reachability)
                                witnessed_axes[key]["observations"].add(observation_source)
                                witnessed_axes[key]["confidence"].add(confidence)
                                witnessed_axes[key]["tray"].add(tray_health)
            assert key in witnessed_cases
            assert witnessed_axes[key]["pairs"] == set(case["core_lifecycle_pair_ids"])
            assert witnessed_axes[key]["reachability"] == set(case["core_ipc_reachability_states"])
            assert witnessed_axes[key]["observations"] == set(
                case["core_state_observation_sources"]
            )
            assert witnessed_axes[key]["confidence"] == set(case["core_state_confidence_states"])
            assert witnessed_axes[key]["tray"] == set(case["tray_process_health_states"])


def test_invalid_when_references_are_rejected() -> None:
    base = load_contract()
    replacements = {
        "core_lifecycle_pair_id": "not_a_lifecycle_pair",
        "core_ipc_reachability_state": "MAYBE_REACHABLE",
        "core_state_observation_source": "TELEPATHY",
        "core_state_confidence_state": "FRESHISH",
        "tray_process_health_state": "MISSING_TRAY_DIMENSION",
        "predicate": "MADE_UP_PREDICATE",
    }
    for axis, bad_value in replacements.items():
        data = deepcopy(base)
        data["state_axis_consistency_constraints"]["rules"].append(
            {"rule_id": f"bad_{axis}", "when": {axis: bad_value}, "requires_predicates": []}
        )
        try:
            _validate_constraint_references(data)
        except AssertionError:
            pass
        else:
            raise AssertionError(axis)


def test_stale_snapshot_and_no_observation_are_separate_generic_cases() -> None:
    data = load_contract()
    predicates_stale = {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"}
    predicates_none = {"RUNTIME_ACTIVITY_UNKNOWN"}
    assert _semantically_valid_combination(
        data, "healthy", "UNREACHABLE", "NONE", "UNKNOWN", "HEALTHY", predicates_none
    )
    assert _semantically_valid_combination(
        data, "healthy", "UNREACHABLE", "CACHED_SNAPSHOT", "STALE", "HEALTHY", predicates_stale
    )
    expected = {
        "CLOSE_DESKTOP_SHELL": ("unreachable_no_observation", "unreachable_stale_snapshot"),
        "HIDE_TO_BACKGROUND": ("unreachable_no_observation", "unreachable_stale_snapshot"),
        "EXIT_TRAY_AGENT": ("unreachable_no_observation", "unreachable_stale_snapshot"),
    }
    for intent_name, (none_case, stale_case) in expected.items():
        cases = intents_by_name(data)[intent_name]["state_cases"]
        intent_none_predicates = predicates_none | (
            {"TRAY_PROCESS_CONFIRMED_RUNNING"} if intent_name == "HIDE_TO_BACKGROUND" else set()
        )
        intent_stale_predicates = predicates_stale | (
            {"TRAY_PROCESS_CONFIRMED_RUNNING"} if intent_name == "HIDE_TO_BACKGROUND" else set()
        )
        none_matches = [
            case["case_id"]
            for case in cases
            if _case_matches(
                case, "healthy", "UNREACHABLE", "NONE", "UNKNOWN", "HEALTHY", intent_none_predicates
            )
        ]
        stale_matches = [
            case["case_id"]
            for case in cases
            if _case_matches(
                case,
                "healthy",
                "UNREACHABLE",
                "CACHED_SNAPSHOT",
                "STALE",
                "HEALTHY",
                intent_stale_predicates,
            )
        ]
        assert none_matches == [none_case]
        assert stale_matches == [stale_case]


def test_tray_supervisor_current_is_separate_from_independent_proof() -> None:
    data = load_contract()
    tray_predicates = {"TRAY_PROCESS_CONFIRMED_RUNNING"}
    assert _semantically_valid_combination(
        data,
        "crashed_unscheduled",
        "UNREACHABLE",
        "TRAY_SUPERVISOR",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        tray_predicates,
    )
    assert not _semantically_valid_combination(
        data,
        "crashed_unscheduled",
        "UNREACHABLE",
        "TRAY_SUPERVISOR",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        tray_predicates | {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
    )
    hide_cases = intents_by_name(data)["HIDE_TO_BACKGROUND"]["state_cases"]
    matches = [
        case["case_id"]
        for case in hide_cases
        if _case_matches(
            case,
            "crashed_unscheduled",
            "UNREACHABLE",
            "TRAY_SUPERVISOR",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            tray_predicates,
        )
    ]
    assert matches == ["crashed_unscheduled_tray_supervisor"]


def test_os_event_best_effort_paths_cover_unreachable_observations() -> None:
    data = load_contract()
    checks = [
        ("NONE", "UNKNOWN", {"SUPERVISED_RESTART_PENDING"}, "os_event_no_observation"),
        (
            "CACHED_SNAPSHOT",
            "STALE",
            {"SUPERVISED_RESTART_PENDING", "CORE_STATE_OBSERVATION_STALE"},
            "os_event_cached_stale",
        ),
        (
            "PROCESS_LOCK",
            "CONFIRMED_CURRENT",
            {"SUPERVISED_RESTART_PENDING", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
            "os_event_independent_current",
        ),
    ]
    for intent_name in ["OS_SESSION_LOGOFF", "OS_SHUTDOWN"]:
        cases = intents_by_name(data)[intent_name]["state_cases"]
        for source, confidence, predicates, expected_case in checks:
            assert _semantically_valid_combination(
                data,
                "supervised_restart_in_progress",
                "UNREACHABLE",
                source,
                confidence,
                "STOPPED",
                predicates,
            )
            matches = [
                case
                for case in cases
                if _case_matches(
                    case,
                    "supervised_restart_in_progress",
                    "UNREACHABLE",
                    source,
                    confidence,
                    "STOPPED",
                    predicates,
                )
            ]
            assert [case["case_id"] for case in matches] == [expected_case]
            assert matches[0]["allowed"] is True
            assert matches[0]["case_kind"] == "normal"


def test_closed_observation_axis_rejects_unsupported_source_confidence_pairs() -> None:
    data = load_contract()
    invalid = [
        (
            "PROCESS_LOCK",
            "STALE",
            {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED", "CORE_STATE_OBSERVATION_STALE"},
        ),
        ("PROCESS_LOCK", "UNKNOWN", {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED"}),
        (
            "CONNECTION_DESCRIPTOR",
            "STALE",
            {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED", "CORE_STATE_OBSERVATION_STALE"},
        ),
        ("CONNECTION_DESCRIPTOR", "UNKNOWN", {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED"}),
        (
            "TRAY_SUPERVISOR",
            "STALE",
            {"TRAY_PROCESS_CONFIRMED_RUNNING", "CORE_STATE_OBSERVATION_STALE"},
        ),
        ("TRAY_SUPERVISOR", "UNKNOWN", {"TRAY_PROCESS_CONFIRMED_RUNNING"}),
        ("CACHED_SNAPSHOT", "UNKNOWN", set()),
        ("CACHED_SNAPSHOT", "CONFIRMED_CURRENT", {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED"}),
        ("NONE", "STALE", {"CORE_STATE_OBSERVATION_STALE"}),
        ("NONE", "CONFIRMED_CURRENT", {"INDEPENDENT_OBSERVATION_PROOF_VALIDATED"}),
    ]
    for source, confidence, predicates in invalid:
        assert not _semantically_valid_combination(
            data, "healthy", "UNREACHABLE", source, confidence, "HEALTHY", predicates
        ), (source, confidence)


def test_closed_observation_axis_accepts_only_canonical_pairs() -> None:
    data = load_contract()
    assert _semantically_valid_combination(
        data,
        "healthy",
        "UNREACHABLE",
        "PROCESS_LOCK",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"ACTIVE_RUNTIME_PRESENT", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
    )
    assert _semantically_valid_combination(
        data,
        "healthy",
        "UNREACHABLE",
        "CONNECTION_DESCRIPTOR",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"ACTIVE_RUNTIME_PRESENT", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"},
    )
    assert _semantically_valid_combination(
        data,
        "healthy",
        "UNREACHABLE",
        "TRAY_SUPERVISOR",
        "CONFIRMED_CURRENT",
        "HEALTHY",
        {"ACTIVE_RUNTIME_PRESENT", "TRAY_PROCESS_CONFIRMED_RUNNING"},
    )
    assert _semantically_valid_combination(
        data,
        "healthy",
        "UNREACHABLE",
        "CACHED_SNAPSHOT",
        "STALE",
        "HEALTHY",
        {"RUNTIME_ACTIVITY_UNKNOWN", "CORE_STATE_OBSERVATION_STALE"},
    )
    assert _semantically_valid_combination(
        data, "healthy", "UNREACHABLE", "NONE", "UNKNOWN", "HEALTHY", {"RUNTIME_ACTIVITY_UNKNOWN"}
    )


def test_close_desktop_shell_tray_supervisor_current_crash_paths() -> None:
    data = load_contract()
    cases = intents_by_name(data)["CLOSE_DESKTOP_SHELL"]["state_cases"]
    checks = [
        (
            "crashed_unscheduled",
            {"TRAY_PROCESS_CONFIRMED_RUNNING"},
            "crashed_unscheduled_tray_supervisor",
        ),
        (
            "crashed_restart_exhausted",
            {"TRAY_PROCESS_CONFIRMED_RUNNING"},
            "restart_exhausted_tray_supervisor",
        ),
    ]
    for pair_id, predicates, expected_case in checks:
        assert _semantically_valid_combination(
            data,
            pair_id,
            "UNREACHABLE",
            "TRAY_SUPERVISOR",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            predicates,
        )
        matches = [
            case
            for case in cases
            if _case_matches(
                case,
                pair_id,
                "UNREACHABLE",
                "TRAY_SUPERVISOR",
                "CONFIRMED_CURRENT",
                "HEALTHY",
                predicates,
            )
        ]
        assert [case["case_id"] for case in matches] == [expected_case]
        assert matches[0]["allowed"] is True
        assert matches[0]["requires_operator_acknowledgement"] is True


def test_exit_tray_agent_tray_supervisor_current_paths() -> None:
    data = load_contract()
    cases = intents_by_name(data)["EXIT_TRAY_AGENT"]["state_cases"]
    checks = [
        (
            "crashed_unscheduled",
            {"TRAY_PROCESS_CONFIRMED_RUNNING"},
            "crashed_unscheduled_tray_supervisor",
        ),
        (
            "crashed_restart_scheduled",
            {"TRAY_PROCESS_CONFIRMED_RUNNING", "SUPERVISED_RESTART_PENDING"},
            "crashed_restart_scheduled_tray_supervisor",
        ),
        (
            "supervised_restart_in_progress",
            {"TRAY_PROCESS_CONFIRMED_RUNNING", "SUPERVISED_RESTART_PENDING"},
            "restart_in_progress_tray_supervisor",
        ),
        (
            "crashed_restart_exhausted",
            {"TRAY_PROCESS_CONFIRMED_RUNNING"},
            "restart_exhausted_tray_supervisor",
        ),
    ]
    for pair_id, predicates, expected_case in checks:
        assert _semantically_valid_combination(
            data,
            pair_id,
            "UNREACHABLE",
            "TRAY_SUPERVISOR",
            "CONFIRMED_CURRENT",
            "HEALTHY",
            predicates,
        )
        matches = [
            case
            for case in cases
            if _case_matches(
                case,
                pair_id,
                "UNREACHABLE",
                "TRAY_SUPERVISOR",
                "CONFIRMED_CURRENT",
                "HEALTHY",
                predicates,
            )
        ]
        assert [case["case_id"] for case in matches] == [expected_case]
        assert matches[0]["allowed"] is True
        assert matches[0]["requires_operator_acknowledgement"] is True
        assert matches[0]["requires_secondary_confirmation"] is True


def _semantic_contexts(data: dict) -> list[tuple[str, str, str, str, str, set[str]]]:
    predicates = [predicate["predicate_id"] for predicate in data["state_predicates"]]
    contexts = []
    for pair in data["valid_core_lifecycle_pairs"]:
        for reachability in data["core_ipc_reachability_states"]:
            for observation_source in data["core_state_observation_sources"]:
                for confidence in data["core_state_confidence_states"]:
                    for tray_health in data["process_health_states"]:
                        for active_predicates in _powerset(predicates):
                            if _semantically_valid_combination(
                                data,
                                pair["pair_id"],
                                reachability,
                                observation_source,
                                confidence,
                                tray_health,
                                active_predicates,
                            ):
                                contexts.append(
                                    (
                                        pair["pair_id"],
                                        reachability,
                                        observation_source,
                                        confidence,
                                        tray_health,
                                        active_predicates,
                                    )
                                )
    return contexts


def test_os_event_best_effort_full_coverage() -> None:
    data = load_contract()
    contexts = _semantic_contexts(data)
    for intent_name in ["OS_SESSION_LOGOFF", "OS_SHUTDOWN"]:
        cases = intents_by_name(data)[intent_name]["state_cases"]
        matched = unmatched = conflicts = 0
        for (
            pair_id,
            reachability,
            observation_source,
            confidence,
            tray_health,
            predicates,
        ) in contexts:
            matches = [
                case
                for case in cases
                if _case_matches(
                    case,
                    pair_id,
                    reachability,
                    observation_source,
                    confidence,
                    tray_health,
                    predicates,
                )
            ]
            if len(matches) == 1:
                matched += 1
            elif not matches:
                unmatched += 1
            else:
                conflicts += 1
        assert matched == len(contexts)
        assert unmatched == 0
        assert conflicts == 0


def test_independent_observation_proof_required_checks_are_complete() -> None:
    data = load_contract()
    assert set(data["independent_observation_proof_contract"]["required_checks"]) == {
        "PID_VALIDATED",
        "START_NONCE_VALIDATED",
        "SOURCE_FRESHNESS_VALIDATED",
        "DEVICE_INSTALLATION_IDENTITY_MATCHED",
        "STATE_STORE_IDENTITY_MATCHED",
    }


def test_state_case_axes_are_covered_by_intent_applicability() -> None:
    data = load_contract()
    pairs = {pair["pair_id"]: pair for pair in data["valid_core_lifecycle_pairs"]}
    for intent in data["shutdown_intents"]:
        applicability = intent["applicability"]
        applicable_health = set(applicability["core_process_health_states"])
        applicable_restart = set(applicability["core_supervision_restart_states"])
        for case in intent["state_cases"]:
            for pair_id in case["core_lifecycle_pair_ids"]:
                pair = pairs[pair_id]
                assert pair["core_process_health_state"] in applicable_health, (
                    intent["name"],
                    case["case_id"],
                    pair_id,
                )
                assert pair["core_supervision_restart_state"] in applicable_restart, (
                    intent["name"],
                    case["case_id"],
                    pair_id,
                )
            assert set(case["core_ipc_reachability_states"]) <= set(
                applicability["core_ipc_reachability_states"]
            ), (intent["name"], case["case_id"])
            assert set(case["core_state_observation_sources"]) <= set(
                applicability["core_state_observation_sources"]
            ), (intent["name"], case["case_id"])
            assert set(case["core_state_confidence_states"]) <= set(
                applicability["core_state_confidence_states"]
            ), (intent["name"], case["case_id"])
            assert set(case["tray_process_health_states"]) <= set(
                applicability["tray_process_health_states"]
            ), (intent["name"], case["case_id"])
    assert (
        "TRAY_SUPERVISOR"
        in intents_by_name(data)["EXIT_TRAY_AGENT"]["applicability"][
            "core_state_observation_sources"
        ]
    )


def test_exit_tray_agent_confirmation_policy_matches_state_cases() -> None:
    data = load_contract()
    exit_intent = intents_by_name(data)["EXIT_TRAY_AGENT"]
    cases = {case["case_id"]: case for case in exit_intent["state_cases"]}
    inactive = cases["inactive_core"]
    assert inactive["requires_secondary_confirmation"] is False
    assert set(inactive["core_lifecycle_pair_ids"]) == {"not_started", "stopped"}
    assert inactive["core_ipc_reachability_states"] == ["UNREACHABLE"]
    assert {"NO_ACTIVE_RUNTIME_CONFIRMED", "INDEPENDENT_OBSERVATION_PROOF_VALIDATED"} <= set(
        inactive["required_predicates"]
    )
    assert {
        "ACTIVE_RUNTIME_PRESENT",
        "RUNTIME_ACTIVITY_UNKNOWN",
        "SUPERVISED_RESTART_PENDING",
    } <= set(inactive["forbidden_predicates"])
    assert [
        case_id
        for case_id, case in cases.items()
        if case["requires_secondary_confirmation"] is False
    ] == ["inactive_core"]
    for case_id, case in cases.items():
        if case_id == "inactive_core" or case["case_kind"] != "normal":
            continue
        risky = (
            "REACHABLE" in case["core_ipc_reachability_states"]
            or "UNREACHABLE" in case["core_ipc_reachability_states"]
            and (
                "CACHED_SNAPSHOT" in case["core_state_observation_sources"]
                or "NONE" in case["core_state_observation_sources"]
            )
            or any(pair_id.startswith("crashed") for pair_id in case["core_lifecycle_pair_ids"])
            or "crashed_restart_scheduled" in case["core_lifecycle_pair_ids"]
            or "supervised_restart_in_progress" in case["core_lifecycle_pair_ids"]
            or "crashed_restart_exhausted" in case["core_lifecycle_pair_ids"]
            or "TRAY_SUPERVISOR" in case["core_state_observation_sources"]
        )
        if risky:
            assert case["requires_secondary_confirmation"] is True, case_id
    for case_id, case in cases.items():
        if "TRAY_SUPERVISOR" in case["core_state_observation_sources"]:
            reason = case["reason"]
            assert "Core is not stopped by EXIT_TRAY_AGENT" in reason
            assert "Core may restart without Tray icon, HUD, or notifications" in reason
            assert "user can cancel" in reason
    confirmation_conditions = "\n".join(exit_intent["confirmation_policy"]["conditions"])
    applicability_conditions = "\n".join(exit_intent["applicability"]["conditions"])
    combined_conditions = f"{confirmation_conditions}\n{applicability_conditions}"
    unconditional_unreachable_text = (
        "secondary confirmation required when Core IPC reachability is UNREACHABLE"
    )
    assert "no secondary confirmation only for inactive_core" in confirmation_conditions
    assert "no secondary confirmation only for inactive_core" in applicability_conditions
    assert unconditional_unreachable_text not in exit_intent["applicability"]["conditions"]
    assert unconditional_unreachable_text not in applicability_conditions
    assert (
        "all UNREACHABLE cases except machine-confirmed inactive_core require secondary confirmation"
        in combined_conditions
    )


def _passes_global_pipeline(
    intent: dict, *, client_role: str, operator_authenticated: bool, authorization_context: bool
) -> bool:
    return (
        client_role in intent["allowed_client_roles"]
        and (not intent["requires_operator_authentication"] or operator_authenticated)
        and authorization_context
    )


def test_global_pipeline_blocks_unauthenticated_or_disallowed_clients() -> None:
    data = load_contract()
    stop = intents_by_name(data)["STOP_STRATEGIES"]
    assert _passes_global_pipeline(
        stop, client_role="desktop_shell", operator_authenticated=True, authorization_context=True
    )
    assert not _passes_global_pipeline(
        stop, client_role="desktop_shell", operator_authenticated=False, authorization_context=True
    )
    assert not _passes_global_pipeline(
        stop, client_role="bootstrapper", operator_authenticated=True, authorization_context=True
    )
    assert not _passes_global_pipeline(
        stop, client_role="desktop_shell", operator_authenticated=True, authorization_context=False
    )


def test_intent_evaluation_routes() -> None:
    data = load_contract()
    routes = {route["route_id"]: route for route in data["intent_evaluation_routes"]}
    assert set(routes) == {
        "LOCAL_SHELL_ACTION",
        "CORE_COMMAND",
        "OPERATING_SYSTEM_EVENT",
        "MAINTENANCE_HANDOFF",
    }
    intents = intents_by_name(data)
    for name in ("CLOSE_DESKTOP_SHELL", "HIDE_TO_BACKGROUND", "EXIT_TRAY_AGENT"):
        intent = intents[name]
        assert intent["evaluation_route"] == "LOCAL_SHELL_ACTION"
        assert intent["requires_core_command_submission"] is False
        assert intent["requires_command_authorization_context"] is False
        assert (
            "core_revalidates_execution_authorization"
            not in routes["LOCAL_SHELL_ACTION"]["pipeline_steps"]
        )
    for name in (
        "STOP_STRATEGIES",
        "PAUSE_STRATEGIES",
        "STOP_CORE_GRACEFULLY",
        "TRIGGER_KILL_SWITCH",
        "ENTER_MAINTENANCE_MODE",
    ):
        intent = intents[name]
        assert intent["evaluation_route"] == "CORE_COMMAND"
        assert intent["requires_core_command_submission"] is True
        assert intent["requires_command_authorization_context"] is True
        steps = routes["CORE_COMMAND"]["pipeline_steps"]
        assert "validate_operator_interface_authentication" in steps
        assert "validate_command_authorization_context" in steps
        assert "validate_required_core_ipc_reachability" in steps
        assert "core_revalidates_execution_authorization" in steps
        assert "accepted_or_rejected_result_is_audited" in steps
    for name in ("OS_SESSION_LOGOFF", "OS_SHUTDOWN"):
        intent = intents[name]
        assert intent["evaluation_route"] == "OPERATING_SYSTEM_EVENT"
        assert intent["allowed_client_roles"] == []
        assert intent["requires_operator_authentication"] is False
        assert intent["requires_command_authorization_context"] is False
        assert set(intent["event_sources"]) == {"operating_system", "windows_session_manager"}
        assert "validate_client_role" not in routes["OPERATING_SYSTEM_EVENT"]["pipeline_steps"]


def test_contract_contains_no_values_that_look_like_real_secrets() -> None:
    raw = CONTRACT.read_text(encoding="utf-8")
    assert not SECRET_RE.search(raw)


# Independent immutable expectation and pure M0.3 first-run bootstrap reference model.
def _bootstrap_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _bootstrap_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_bootstrap_freeze(item) for item in value)
    return value


FIRST_RUN_BOOTSTRAP_EXPECTED = _bootstrap_freeze(
    json.loads(r"""{
  "status": "closed",
  "authority_owner": "external_product_provisioning_boundary",
  "consumer": "CoreHost",
  "nature": "ephemeral protected security/provisioning handoff; not a durable M0.2 entity",
  "implementation_neutral_examples": [
    "protected installer handoff",
    "package provisioning",
    "SaaS or device provisioning",
    "other product provisioning mechanism"
  ],
  "non_authorities": [
    "bootstrapper",
    "tray_agent",
    "desktop_shell",
    "raw_caller",
    "caller_boolean",
    "self_hash",
    "ordinary_RuntimeSession",
    "arbitrary_local_admin_claim"
  ],
  "bootstrapper_role": {
    "mode": "transport_and_discovery_only",
    "may_pass": "opaque protected reference",
    "cannot": [
      "mint",
      "accept",
      "elevate"
    ]
  },
  "identity_boundary": {
    "provisioned_identity_fields": [
      "account_id",
      "device_installation_id"
    ],
    "canonical_state_store_identity_required": true,
    "does_not_imply": [
      "device_TRUSTED",
      "operator_authenticated",
      "PIN_verified",
      "LIVE_authorized"
    ],
    "RuntimeSession_during_SETUP_REQUIRED": true,
    "noncanonical_alias_forbidden": true
  },
  "claim_schema": {
    "ordered_fields": [
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "bootstrap_generation",
      "bootstrap_revision",
      "issued_at_utc",
      "expires_at_utc",
      "challenge_fingerprint_sha256",
      "provisioning_context_fingerprint_sha256",
      "claim_fingerprint_sha256"
    ],
    "ephemeral_key": "claim_fingerprint_sha256",
    "durable_bootstrap_id": false,
    "fingerprint_input_fields": [
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "bootstrap_generation",
      "bootstrap_revision",
      "issued_at_utc",
      "expires_at_utc",
      "challenge_fingerprint_sha256",
      "provisioning_context_fingerprint_sha256"
    ]
  },
  "acceptance_authority": {
    "required": "claim fingerprint resolves through a pre-existing Core-visible accepted provisioning membership registry established independently before caller claim validation",
    "public_validator_inputs": [
      "untrusted claim",
      "opaque Core authority state reference",
      "now_utc",
      "requested authority purpose"
    ],
    "public_validator_excludes": [
      "caller-owned registry",
      "caller-created membership binding",
      "caller-created Core state projection"
    ],
    "membership_schema": "ProvisioningMembershipBinding",
    "registry_binding": [
      "claim_fingerprint_sha256",
      "complete_claim_content_fingerprint_sha256",
      "authority_source",
      "provisioning_context_fingerprint_sha256",
      "exact accepted immutable claim content"
    ],
    "authority_source": "external_product_provisioning_boundary",
    "owner_string_alone_authority": false,
    "hash_semantics": "integrity only; recomputed self-hash cannot create membership",
    "nominal_type_semantics": "dataclass, JSON shape, owner string or state fingerprint alone is not authority",
    "boolean_authority": false,
    "CoreHost_may_mint": false,
    "registry_key_binding": "registry key == binding.claim_fingerprint_sha256 == claim.claim_fingerprint_sha256"
  },
  "eligibility": {
    "startup_readiness": "SETUP_REQUIRED",
    "exact_bindings": [
      "resolved account_id",
      "resolved device_installation_id",
      "intended first operator_id",
      "bootstrap generation",
      "bootstrap revision",
      "provisioning membership"
    ],
    "required_absence": [
      "accepted/current first OperatorIdentity for bootstrap scope",
      "completed initial-security setup",
      "consumed bootstrap generation"
    ],
    "time_rule": "canonical UTC issued_at_utc <= now_utc <= expires_at_utc and expires_at_utc > issued_at_utc",
    "caller_first_run_flag_authority": false
  },
  "one_shot_transition": {
    "input": "accepted provisioning membership plus exact current PRE_INITIAL_SECURITY Core state and SETUP_REQUIRED",
    "output": "BootstrapTransitionResult with INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED and INITIAL_SECURITY_ESTABLISHMENT_ONLY purpose plus atomically derived post-state consumption",
    "consumed_schema": "ConsumedBootstrapAuthority",
    "consumes": [
      "account_id",
      "device_installation_id",
      "bootstrap_generation",
      "bootstrap_revision",
      "claim_fingerprint_sha256",
      "challenge_fingerprint_sha256"
    ],
    "semantic_atomicity": "compare current state, authorize and append exact consumed binding as one semantic transition; at most one success per accepted claim",
    "does_not_set_readiness": "READY"
  },
  "terminal_fencing": {
    "replay": false,
    "modified_claim_recomputed_hash_reuses_membership": false,
    "after_completed_setup": false,
    "second_device_enrollment": false,
    "reinstall_or_recovery_bypass": false,
    "forbidden_authorizations": [
      "ordinary privileged operations",
      "RiskPolicy",
      "kill switch",
      "ProductCapabilities",
      "exchange operations",
      "ExecutionLease",
      "LIVE"
    ],
    "one_shot_rule": "exact consumed binding or terminal completed lifecycle denies reuse",
    "caller_cannot_clear_consumption": true
  },
  "failure_outcome": {
    "missing_or_invalid": "SETUP_REQUIRED remains; deny bootstrap",
    "side_effects_forbidden": [
      "private exchange connection",
      "exchange-secret loading",
      "strategy start",
      "order entry"
    ]
  },
  "audit_boundary": {
    "uses": "canonical M0.2 AuditEvent; M0.7 causation/correlation when available",
    "ordering": "RuntimeSession is created after process-lock ownership and may exist before bootstrap validation; no artificial pre-RuntimeSession event requirement",
    "allowed": [
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "claim/reference fingerprint",
      "bootstrap generation",
      "reason code",
      "causation_id",
      "correlation_id"
    ],
    "forbidden": [
      "raw bootstrap secret",
      "PIN",
      "biometric material",
      "API credentials",
      "protected provisioning payload"
    ]
  },
  "milestone_ownership": {
    "M0.3": "pre-existing provisioning authority handoff, startup ordering and one-shot scope",
    "M0.10": "first OperatorIdentity acceptance, initial TRUSTED designation, PIN verifier, optional platform biometric semantics and first Core-issued authentication proof",
    "M0.11": "durably atomic compare-and-consume, persistence, migrations, backup and crash recovery for accepted/current membership and consumed facts"
  },
  "invariants": [
    "DeviceInstallation identity is not M0.10 TRUSTED designation",
    "Bootstrapper transports but never issues or accepts authority",
    "CoreHost consumes but cannot self-issue provisioning membership",
    "valid bootstrap does not mean READY",
    "current LIVE remains denied",
    "no caller boolean or self-hash authority"
  ],
  "startup_identity_resolution": {
    "mode_source": "accepted/current installation state; never caller first_run boolean",
    "EXISTING_INSTALLATION": "resolve canonical account_id, device_installation_id and state-store identity from accepted local/Core state under later persistence boundary; first-run provisioning handoff not required again",
    "FIRST_INSTALLATION": "only when accepted installation identity is absent, external product provisioning handoff supplies canonical account_id and device_installation_id for state-store/process-lock scope and later setup",
    "common_non_implications": [
      "device_TRUSTED",
      "operator_authenticated",
      "PIN_verified",
      "LIVE_authorized"
    ]
  },
  "executable_schemas": {
    "FirstRunBootstrapClaim": [
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "bootstrap_generation",
      "bootstrap_revision",
      "issued_at_utc",
      "expires_at_utc",
      "challenge_fingerprint_sha256",
      "provisioning_context_fingerprint_sha256",
      "claim_fingerprint_sha256"
    ],
    "ProvisioningMembershipBinding": [
      "claim_fingerprint_sha256",
      "complete_claim_content_fingerprint_sha256",
      "authority_source",
      "provisioning_context_fingerprint_sha256"
    ],
    "ConsumedBootstrapAuthority": [
      "account_id",
      "device_installation_id",
      "bootstrap_generation",
      "bootstrap_revision",
      "claim_fingerprint_sha256",
      "challenge_fingerprint_sha256"
    ],
    "CoreCurrentBootstrapState": [
      "state_fingerprint_sha256",
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "startup_readiness",
      "initial_security_lifecycle",
      "first_operator_presence",
      "expected_generation",
      "expected_revision",
      "consumed_authorities",
      "state_revision"
    ],
    "BootstrapTransitionResult": [
      "outcome",
      "authority_purpose",
      "pre_state_fingerprint_sha256",
      "post_state_fingerprint_sha256",
      "consumed_authority"
    ]
  },
  "core_current_state_authority": {
    "schema": "CoreCurrentBootstrapState",
    "lifecycle_states": [
      "PRE_INITIAL_SECURITY",
      "INITIAL_SECURITY_COMPLETED"
    ],
    "readiness_states": [
      "SETUP_REQUIRED"
    ],
    "acceptance": "opaque state reference must resolve in pre-existing Core-owned accepted registry and be the exact current designation for account/device scope",
    "caller_projection_authority": false,
    "stale_accepted_history_authority": false,
    "state_fingerprint_input_fields": [
      "account_id",
      "device_installation_id",
      "intended_operator_id",
      "startup_readiness",
      "initial_security_lifecycle",
      "first_operator_presence",
      "expected_generation",
      "expected_revision",
      "consumed_authorities",
      "state_revision"
    ],
    "first_operator_presence_states": [
      "ABSENT",
      "PRESENT"
    ],
    "lifecycle_presence_invariants": {
      "PRE_INITIAL_SECURITY": "ABSENT",
      "INITIAL_SECURITY_COMPLETED": "PRESENT"
    },
    "future_m010_fence": "accepting first OperatorIdentity must replace current PRE_INITIAL_SECURITY/ABSENT designation; historical PRE may remain audit-only"
  },
  "authority_purpose_registry": [
    "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
  ],
  "scope_consumer_policy": {
    "allowed": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "all_other_purposes": "BOOTSTRAP_SCOPE_DENIED",
    "result_alone_authority": false,
    "requires_current_post_state": true,
    "requires_exact_pre_to_post_transition": true
  },
  "transition_result_authority": {
    "transport_object_authority": false,
    "nominal_type_authority": false,
    "caller_created_success_authority": false,
    "consumer_revalidation": [
      "exact accepted historical PRE registry key/internal/recomputed fingerprint",
      "exact accepted and current POST registry key/internal/recomputed fingerprint",
      "exact PRE to POST single-consumption transition",
      "exact new ConsumedBootstrapAuthority binding"
    ],
    "registry_key_equals_internal_state_fingerprint": true,
    "consumed_history_validation": "every entry fully validates canonical IDs, positive non-bool generation/revision, lowercase SHA-256, state scope, uniqueness and generation uniqueness"
  }
}""")
)


def _bootstrap_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _bootstrap_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_bootstrap_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class FirstRunBootstrapClaim:
    account_id: str
    device_installation_id: str
    intended_operator_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    issued_at_utc: str
    expires_at_utc: str
    challenge_fingerprint_sha256: str
    provisioning_context_fingerprint_sha256: str
    claim_fingerprint_sha256: str


@dataclass(frozen=True)
class ProvisioningMembershipBinding:
    claim_fingerprint_sha256: str
    complete_claim_content_fingerprint_sha256: str
    authority_source: str
    provisioning_context_fingerprint_sha256: str


@dataclass(frozen=True)
class ConsumedBootstrapAuthority:
    account_id: str
    device_installation_id: str
    bootstrap_generation: int
    bootstrap_revision: int
    claim_fingerprint_sha256: str
    challenge_fingerprint_sha256: str


@dataclass(frozen=True)
class CoreCurrentBootstrapState:
    state_fingerprint_sha256: str
    account_id: str
    device_installation_id: str
    intended_operator_id: str
    startup_readiness: str
    initial_security_lifecycle: str
    first_operator_presence: str
    expected_generation: int
    expected_revision: int
    consumed_authorities: tuple[ConsumedBootstrapAuthority, ...]
    state_revision: int


@dataclass(frozen=True)
class BootstrapTransitionResult:
    outcome: str
    authority_purpose: str | None
    pre_state_fingerprint_sha256: str | None
    post_state_fingerprint_sha256: str | None
    consumed_authority: ConsumedBootstrapAuthority | None


BOOTSTRAP_AUTHORITY_DATACLASSES = {
    "FirstRunBootstrapClaim": FirstRunBootstrapClaim,
    "ProvisioningMembershipBinding": ProvisioningMembershipBinding,
    "ConsumedBootstrapAuthority": ConsumedBootstrapAuthority,
    "CoreCurrentBootstrapState": CoreCurrentBootstrapState,
    "BootstrapTransitionResult": BootstrapTransitionResult,
}

# These registries model pre-existing Core-visible authority only. They are not persistence.
_ACCEPTED_PROVISIONING_MEMBERSHIPS: dict[str, ProvisioningMembershipBinding] = {}
_ACCEPTED_PROVISIONING_CLAIMS: dict[str, FirstRunBootstrapClaim] = {}
_ACCEPTED_CORE_STATES: dict[str, CoreCurrentBootstrapState] = {}
_CURRENT_CORE_STATE_BY_SCOPE: dict[tuple[str, str], str] = {}


def _canonical_json(value: Any) -> bytes:
    if isinstance(value, tuple):
        value = list(value)
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _claim_content(claim: FirstRunBootstrapClaim) -> dict[str, Any]:
    content = asdict(claim)
    content.pop("claim_fingerprint_sha256")
    return content


def _claim_fingerprint(claim: FirstRunBootstrapClaim) -> str:
    return _fingerprint(_claim_content(claim))


def _state_content(state: CoreCurrentBootstrapState) -> dict[str, Any]:
    content = asdict(state)
    content.pop("state_fingerprint_sha256")
    return content


def _state_fingerprint(state: CoreCurrentBootstrapState) -> str:
    return _fingerprint(_state_content(state))


def _canonical_utc(value: str) -> datetime | None:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def _canonical_id(value: object, prefix: str) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(
            rf"{prefix}_[0-9a-f]{{8}}-[0-9a-f]{{4}}-7[0-9a-f]{{3}}-[89ab][0-9a-f]{{3}}-[0-9a-f]{{12}}",
            value,
        )
        is not None
    )


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _trusted_provisioning_fixture(
    claim: FirstRunBootstrapClaim, state: CoreCurrentBootstrapState
) -> str:
    """Establish authority before public validation; never called by the validator."""
    _ACCEPTED_PROVISIONING_MEMBERSHIPS.clear()
    _ACCEPTED_PROVISIONING_CLAIMS.clear()
    _ACCEPTED_CORE_STATES.clear()
    _CURRENT_CORE_STATE_BY_SCOPE.clear()
    binding = ProvisioningMembershipBinding(
        claim.claim_fingerprint_sha256,
        _fingerprint(_claim_content(claim)),
        "external_product_provisioning_boundary",
        claim.provisioning_context_fingerprint_sha256,
    )
    _ACCEPTED_PROVISIONING_MEMBERSHIPS[claim.claim_fingerprint_sha256] = binding
    _ACCEPTED_PROVISIONING_CLAIMS[claim.claim_fingerprint_sha256] = claim
    _ACCEPTED_CORE_STATES[state.state_fingerprint_sha256] = state
    _CURRENT_CORE_STATE_BY_SCOPE[(state.account_id, state.device_installation_id)] = (
        state.state_fingerprint_sha256
    )
    return state.state_fingerprint_sha256


def _denied(outcome: str) -> BootstrapTransitionResult:
    return BootstrapTransitionResult(outcome, None, None, None, None)


def validate_consumed_bootstrap_authority(
    item: object, state: CoreCurrentBootstrapState | None = None
) -> bool:
    if not isinstance(item, ConsumedBootstrapAuthority):
        return False
    if not all(
        (
            _canonical_id(item.account_id, "acct"),
            _canonical_id(item.device_installation_id, "dev"),
            _positive_int(item.bootstrap_generation),
            _positive_int(item.bootstrap_revision),
            isinstance(item.claim_fingerprint_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", item.claim_fingerprint_sha256),
            isinstance(item.challenge_fingerprint_sha256, str)
            and re.fullmatch(r"[0-9a-f]{64}", item.challenge_fingerprint_sha256),
        )
    ):
        return False
    return state is None or (item.account_id, item.device_installation_id) == (
        state.account_id,
        state.device_installation_id,
    )


def _validate_consumed_history(state: CoreCurrentBootstrapState) -> bool:
    items = state.consumed_authorities
    if not isinstance(items, tuple) or not all(
        validate_consumed_bootstrap_authority(item, state) for item in items
    ):
        return False
    return len(items) == len(set(items)) and len(
        {item.bootstrap_generation for item in items}
    ) == len(items)


def _accepted_state(reference: object) -> tuple[str, CoreCurrentBootstrapState | None]:
    if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-f]{64}", reference):
        return "MALFORMED_CORE_BOOTSTRAP_STATE", None
    state = _ACCEPTED_CORE_STATES.get(reference)
    if state is None:
        return "BOOTSTRAP_AUTHORITY_DENIED", None
    if state.state_fingerprint_sha256 != reference or _state_fingerprint(state) != reference:
        return "CONTRACT_INCONSISTENT", None
    return "VALID", state


def authorize_and_consume_first_run_bootstrap(
    claim: object, core_state_reference: object, now_utc: object, requested_purpose: object
) -> BootstrapTransitionResult:
    if not isinstance(claim, FirstRunBootstrapClaim):
        return _denied("BOOTSTRAP_AUTHORITY_DENIED")
    if not isinstance(core_state_reference, str) or not re.fullmatch(
        r"[0-9a-f]{64}", core_state_reference
    ):
        return _denied("MALFORMED_CORE_BOOTSTRAP_STATE")
    if not isinstance(now_utc, str) or not isinstance(requested_purpose, str):
        return _denied("MALFORMED_BOOTSTRAP_INPUT")
    state_status, state = _accepted_state(core_state_reference)
    if state is None:
        return _denied(state_status)
    current_ref = _CURRENT_CORE_STATE_BY_SCOPE.get((state.account_id, state.device_installation_id))
    if current_ref != core_state_reference:
        current = _ACCEPTED_CORE_STATES.get(current_ref or "")
        if current and any(
            item.claim_fingerprint_sha256 == claim.claim_fingerprint_sha256
            for item in current.consumed_authorities
        ):
            return _denied("BOOTSTRAP_REPLAY_DENIED")
        return _denied("STALE_CORE_BOOTSTRAP_STATE")
    if not all(
        (
            _canonical_id(state.account_id, "acct"),
            _canonical_id(state.device_installation_id, "dev"),
            _canonical_id(state.intended_operator_id, "op"),
        )
    ):
        return _denied("MALFORMED_CORE_BOOTSTRAP_STATE")
    lifecycle_presence = {"PRE_INITIAL_SECURITY": "ABSENT", "INITIAL_SECURITY_COMPLETED": "PRESENT"}
    if (
        state.startup_readiness != "SETUP_REQUIRED"
        or lifecycle_presence.get(state.initial_security_lifecycle) != state.first_operator_presence
    ):
        return _denied("MALFORMED_CORE_BOOTSTRAP_STATE")
    if not all(
        (
            _positive_int(state.expected_generation),
            _positive_int(state.expected_revision),
            _positive_int(state.state_revision),
        )
    ):
        return _denied("MALFORMED_CORE_BOOTSTRAP_STATE")
    if not _validate_consumed_history(state):
        return _denied("MALFORMED_CORE_BOOTSTRAP_STATE")
    if requested_purpose != "INITIAL_SECURITY_ESTABLISHMENT_ONLY":
        return _denied("BOOTSTRAP_SCOPE_DENIED")
    if state.initial_security_lifecycle == "INITIAL_SECURITY_COMPLETED":
        return _denied("BOOTSTRAP_NOT_ELIGIBLE")
    if not all(
        (
            _canonical_id(claim.account_id, "acct"),
            _canonical_id(claim.device_installation_id, "dev"),
            _canonical_id(claim.intended_operator_id, "op"),
        )
    ):
        return _denied("MALFORMED_BOOTSTRAP_CLAIM")
    if not _positive_int(claim.bootstrap_generation) or not _positive_int(claim.bootstrap_revision):
        return _denied("MALFORMED_BOOTSTRAP_CLAIM")
    hashes = (
        claim.challenge_fingerprint_sha256,
        claim.provisioning_context_fingerprint_sha256,
        claim.claim_fingerprint_sha256,
    )
    if not all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
        return _denied("MALFORMED_BOOTSTRAP_CLAIM")
    issued, expires, now = (
        _canonical_utc(claim.issued_at_utc),
        _canonical_utc(claim.expires_at_utc),
        _canonical_utc(now_utc),
    )
    if issued is None or expires is None or now is None or expires <= issued:
        return _denied("MALFORMED_BOOTSTRAP_CLAIM")
    if now < issued:
        return _denied("BOOTSTRAP_NOT_YET_VALID")
    if now > expires:
        return _denied("BOOTSTRAP_EXPIRED")
    if _claim_fingerprint(claim) != claim.claim_fingerprint_sha256:
        return _denied("BOOTSTRAP_AUTHORITY_DENIED")
    binding = _ACCEPTED_PROVISIONING_MEMBERSHIPS.get(claim.claim_fingerprint_sha256)
    accepted_claim = _ACCEPTED_PROVISIONING_CLAIMS.get(claim.claim_fingerprint_sha256)
    if binding is None or accepted_claim != claim:
        return _denied("BOOTSTRAP_AUTHORITY_DENIED")
    if (
        binding.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256
        or binding.authority_source != "external_product_provisioning_boundary"
        or binding.complete_claim_content_fingerprint_sha256 != _fingerprint(_claim_content(claim))
        or binding.provisioning_context_fingerprint_sha256
        != claim.provisioning_context_fingerprint_sha256
    ):
        return _denied("BOOTSTRAP_AUTHORITY_DENIED")
    exact = (
        claim.account_id,
        claim.device_installation_id,
        claim.intended_operator_id,
        claim.bootstrap_generation,
        claim.bootstrap_revision,
    )
    expected = (
        state.account_id,
        state.device_installation_id,
        state.intended_operator_id,
        state.expected_generation,
        state.expected_revision,
    )
    if exact != expected:
        return _denied("BOOTSTRAP_BINDING_DENIED")
    consumed = ConsumedBootstrapAuthority(
        claim.account_id,
        claim.device_installation_id,
        claim.bootstrap_generation,
        claim.bootstrap_revision,
        claim.claim_fingerprint_sha256,
        claim.challenge_fingerprint_sha256,
    )
    if consumed in state.consumed_authorities or any(
        item.bootstrap_generation == consumed.bootstrap_generation
        for item in state.consumed_authorities
    ):
        return _denied("BOOTSTRAP_REPLAY_DENIED")
    post = replace(
        state,
        state_fingerprint_sha256="0" * 64,
        consumed_authorities=state.consumed_authorities + (consumed,),
        state_revision=state.state_revision + 1,
    )
    post = replace(post, state_fingerprint_sha256=_state_fingerprint(post))
    _ACCEPTED_CORE_STATES[post.state_fingerprint_sha256] = post
    _CURRENT_CORE_STATE_BY_SCOPE[(post.account_id, post.device_installation_id)] = (
        post.state_fingerprint_sha256
    )
    return BootstrapTransitionResult(
        "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED",
        "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        state.state_fingerprint_sha256,
        post.state_fingerprint_sha256,
        consumed,
    )


def consume_bootstrap_result_for_purpose(result: object, purpose: object) -> str:
    if (
        not isinstance(result, BootstrapTransitionResult)
        or result.outcome != "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED"
        or purpose != "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        or result.authority_purpose != purpose
        or not isinstance(result.consumed_authority, ConsumedBootstrapAuthority)
    ):
        return "BOOTSTRAP_SCOPE_DENIED"
    pre_status, pre = _accepted_state(result.pre_state_fingerprint_sha256)
    post_status, post = _accepted_state(result.post_state_fingerprint_sha256)
    if pre_status != "VALID" or post_status != "VALID" or pre is None or post is None:
        return "BOOTSTRAP_SCOPE_DENIED"
    if (
        _CURRENT_CORE_STATE_BY_SCOPE.get((post.account_id, post.device_installation_id))
        != result.post_state_fingerprint_sha256
    ):
        return "BOOTSTRAP_SCOPE_DENIED"
    stable_fields = (
        "account_id",
        "device_installation_id",
        "intended_operator_id",
        "startup_readiness",
        "initial_security_lifecycle",
        "first_operator_presence",
        "expected_generation",
        "expected_revision",
    )
    if any(getattr(pre, field) != getattr(post, field) for field in stable_fields):
        return "BOOTSTRAP_SCOPE_DENIED"
    if post.state_revision != pre.state_revision + 1:
        return "BOOTSTRAP_SCOPE_DENIED"
    if len(post.consumed_authorities) != len(pre.consumed_authorities) + 1:
        return "BOOTSTRAP_SCOPE_DENIED"
    if post.consumed_authorities[:-1] != pre.consumed_authorities:
        return "BOOTSTRAP_SCOPE_DENIED"
    if post.consumed_authorities[-1] != result.consumed_authority:
        return "BOOTSTRAP_SCOPE_DENIED"
    if not _validate_consumed_history(pre) or not _validate_consumed_history(post):
        return "BOOTSTRAP_SCOPE_DENIED"
    return "BOOTSTRAP_PURPOSE_ACCEPTED"


def _bootstrap_claim() -> FirstRunBootstrapClaim:
    claim = FirstRunBootstrapClaim(
        "acct_018f0000-0000-7000-8000-000000000001",
        "dev_018f0000-0000-7000-8000-000000000002",
        "op_018f0000-0000-7000-8000-000000000003",
        1,
        1,
        "2026-08-10T10:00:00Z",
        "2026-08-10T10:05:00Z",
        "a" * 64,
        "b" * 64,
        "0" * 64,
    )
    return replace(claim, claim_fingerprint_sha256=_claim_fingerprint(claim))


def _bootstrap_state() -> CoreCurrentBootstrapState:
    claim = _bootstrap_claim()
    state = CoreCurrentBootstrapState(
        "0" * 64,
        claim.account_id,
        claim.device_installation_id,
        claim.intended_operator_id,
        "SETUP_REQUIRED",
        "PRE_INITIAL_SECURITY",
        "ABSENT",
        1,
        1,
        (),
        1,
    )
    return replace(state, state_fingerprint_sha256=_state_fingerprint(state))


def _genuine_authority() -> tuple[FirstRunBootstrapClaim, str]:
    claim, state = _bootstrap_claim(), _bootstrap_state()
    return claim, _trusted_provisioning_fixture(claim, state)


def validate_bootstrap_machine_root(root: object) -> str:
    return (
        "VALID"
        if root == _bootstrap_thaw(FIRST_RUN_BOOTSTRAP_EXPECTED)
        else "CONTRACT_INCONSISTENT"
    )


def test_bootstrap_machine_root_and_all_authority_schemas_are_exact() -> None:
    root = load_contract()["first_run_bootstrap_authority_contract"]
    assert validate_bootstrap_machine_root(root) == "VALID"
    schemas = root["executable_schemas"]
    assert set(schemas) == set(BOOTSTRAP_AUTHORITY_DATACLASSES)
    for name, model in BOOTSTRAP_AUTHORITY_DATACLASSES.items():
        assert schemas[name] == [field.name for field in fields(model)]


@pytest.mark.parametrize("raw", [{}, {"bootstrap": True}, {"trusted": True}, {"accepted": True}])
def test_raw_caller_boolean_shadows_and_nominal_types_are_denied(raw: object) -> None:
    assert (
        authorize_and_consume_first_run_bootstrap(
            raw, raw, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_fake_registry_membership_and_owner_string_cannot_be_public_authority() -> None:
    claim = _bootstrap_claim()
    fake = ProvisioningMembershipBinding(
        claim.claim_fingerprint_sha256,
        _fingerprint(_claim_content(claim)),
        "external_product_provisioning_boundary",
        claim.provisioning_context_fingerprint_sha256,
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, fake, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "MALFORMED_CORE_BOOTSTRAP_STATE"
    )


def test_self_hash_without_preexisting_membership_is_denied() -> None:
    _ACCEPTED_PROVISIONING_MEMBERSHIPS.clear()
    _ACCEPTED_PROVISIONING_CLAIMS.clear()
    _ACCEPTED_CORE_STATES.clear()
    _CURRENT_CORE_STATE_BY_SCOPE.clear()
    claim = _bootstrap_claim()
    state = _bootstrap_state()
    _ACCEPTED_CORE_STATES[state.state_fingerprint_sha256] = state
    _CURRENT_CORE_STATE_BY_SCOPE[(state.account_id, state.device_installation_id)] = (
        state.state_fingerprint_sha256
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim,
            state.state_fingerprint_sha256,
            "2026-08-10T10:01:00Z",
            "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_genuine_preexisting_membership_pre_to_consumed_then_replay_denied() -> None:
    claim, pre_ref = _genuine_authority()
    first = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert first.outcome == "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED"
    assert first.consumed_authority == ConsumedBootstrapAuthority(
        claim.account_id,
        claim.device_installation_id,
        1,
        1,
        claim.claim_fingerprint_sha256,
        claim.challenge_fingerprint_sha256,
    )
    assert first.post_state_fingerprint_sha256 != pre_ref
    second = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert second.outcome == "BOOTSTRAP_REPLAY_DENIED"


@pytest.mark.parametrize("state_input", [{}, object(), True, "bad"])
def test_malformed_or_caller_created_core_state_fails_without_exception(
    state_input: object,
) -> None:
    claim, _ = _genuine_authority()
    assert authorize_and_consume_first_run_bootstrap(
        claim, state_input, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    ).outcome in {"BOOTSTRAP_AUTHORITY_DENIED", "MALFORMED_CORE_BOOTSTRAP_STATE"}


def test_caller_created_setup_required_state_and_recomputed_hash_is_not_registered_authority() -> (
    None
):
    claim, _ = _genuine_authority()
    fake = replace(_bootstrap_state(), state_fingerprint_sha256="0" * 64, state_revision=2)
    fake = replace(fake, state_fingerprint_sha256=_state_fingerprint(fake))
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim,
            fake.state_fingerprint_sha256,
            "2026-08-10T10:01:00Z",
            "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("startup_readiness", "UNKNOWN"),
        ("expected_revision", True),
        ("expected_generation", 0),
        ("consumed_authorities", []),
        ("state_revision", False),
    ],
)
def test_malformed_registered_core_state_fails_closed(field: str, value: Any) -> None:
    claim, state = _bootstrap_claim(), _bootstrap_state()
    bad = replace(state, **{field: value}, state_fingerprint_sha256="0" * 64)  # type: ignore[arg-type]
    bad = replace(bad, state_fingerprint_sha256=_state_fingerprint(bad))
    ref = _trusted_provisioning_fixture(claim, bad)
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "MALFORMED_CORE_BOOTSTRAP_STATE"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_id", "acct_018f0000-0000-7000-8000-000000000099"),
        ("device_installation_id", "dev_018f0000-0000-7000-8000-000000000099"),
        ("bootstrap_revision", 2),
        ("challenge_fingerprint_sha256", "c" * 64),
    ],
)
def test_claim_substitutions_cannot_reuse_genuine_membership(field: str, value: Any) -> None:
    original, pre_ref = _genuine_authority()
    changed = replace(original, **{field: value}, claim_fingerprint_sha256="0" * 64)  # type: ignore[arg-type]
    changed = replace(changed, claim_fingerprint_sha256=_claim_fingerprint(changed))
    assert (
        authorize_and_consume_first_run_bootstrap(
            changed, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_fake_new_binding_for_modified_claim_is_not_preexisting_authority() -> None:
    original, pre_ref = _genuine_authority()
    changed = replace(
        original,
        intended_operator_id="op_018f0000-0000-7000-8000-000000000099",
        claim_fingerprint_sha256="0" * 64,
    )
    changed = replace(changed, claim_fingerprint_sha256=_claim_fingerprint(changed))
    fake = ProvisioningMembershipBinding(
        changed.claim_fingerprint_sha256,
        _fingerprint(_claim_content(changed)),
        "external_product_provisioning_boundary",
        changed.provisioning_context_fingerprint_sha256,
    )
    del fake
    assert (
        authorize_and_consume_first_run_bootstrap(
            changed, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_wrong_authority_source_and_modified_accepted_content_are_denied() -> None:
    claim, ref = _genuine_authority()
    binding = _ACCEPTED_PROVISIONING_MEMBERSHIPS[claim.claim_fingerprint_sha256]
    _ACCEPTED_PROVISIONING_MEMBERSHIPS[claim.claim_fingerprint_sha256] = replace(
        binding, authority_source="CoreHost"
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )
    claim, ref = _genuine_authority()
    _ACCEPTED_PROVISIONING_CLAIMS[claim.claim_fingerprint_sha256] = replace(
        claim, expires_at_utc="2026-08-10T10:04:00Z"
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_completed_setup_stale_state_and_cleared_consumption_cannot_restore_authority() -> None:
    claim, pre_ref = _genuine_authority()
    first = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    post = _ACCEPTED_CORE_STATES[cast(str, first.post_state_fingerprint_sha256)]
    forged = replace(
        post,
        state_fingerprint_sha256="0" * 64,
        consumed_authorities=(),
        initial_security_lifecycle="PRE_INITIAL_SECURITY",
    )
    forged = replace(forged, state_fingerprint_sha256=_state_fingerprint(forged))
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim,
            forged.state_fingerprint_sha256,
            "2026-08-10T10:01:00Z",
            "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )
    completed = replace(
        post,
        state_fingerprint_sha256="0" * 64,
        initial_security_lifecycle="INITIAL_SECURITY_COMPLETED",
        first_operator_presence="PRESENT",
    )
    completed = replace(completed, state_fingerprint_sha256=_state_fingerprint(completed))
    _ACCEPTED_CORE_STATES[completed.state_fingerprint_sha256] = completed
    _CURRENT_CORE_STATE_BY_SCOPE[(completed.account_id, completed.device_installation_id)] = (
        completed.state_fingerprint_sha256
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim,
            completed.state_fingerprint_sha256,
            "2026-08-10T10:01:00Z",
            "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        ).outcome
        == "BOOTSTRAP_NOT_ELIGIBLE"
    )


@pytest.mark.parametrize(
    "purpose",
    [
        "ORDINARY_PRIVILEGED_OPERATION",
        "RISK_POLICY",
        "KILL_SWITCH",
        "PRODUCT_CAPABILITIES",
        "EXECUTION_LEASE",
        "LIVE",
    ],
)
def test_successful_bootstrap_cannot_escape_initial_security_purpose(purpose: str) -> None:
    claim, pre_ref = _genuine_authority()
    result = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert consume_bootstrap_result_for_purpose(result, purpose) == "BOOTSTRAP_SCOPE_DENIED"


def test_existing_installation_startup_does_not_require_first_run_handoff() -> None:
    root = load_contract()["first_run_bootstrap_authority_contract"]
    existing = root["startup_identity_resolution"]["EXISTING_INSTALLATION"]
    assert "first-run provisioning handoff not required again" in existing
    assert (
        "accepted/current installation state; never caller first_run boolean"
        == root["startup_identity_resolution"]["mode_source"]
    )


def test_device_identity_remains_distinct_from_trusted_designation() -> None:
    root = load_contract()["first_run_bootstrap_authority_contract"]
    assert "device_TRUSTED" in root["identity_boundary"]["does_not_imply"]
    assert "device_TRUSTED" in root["startup_identity_resolution"]["common_non_implications"]


def test_startup_order_and_losing_contender_remain_fail_closed() -> None:
    data = load_contract()
    steps = {item["step_id"]: item["order"] for item in data["startup_sequence"]}
    assert (
        steps["resolve_device_installation_and_state_store_identity"]
        < steps["acquire_local_process_lock"]
        < steps["create_runtime_session"]
        < steps["open_state_store"]
        < steps["verify_integrity"]
        < steps["determine_startup_readiness"]
        < steps["validate_first_run_bootstrap_authority_if_required"]
    )
    busy = next(item for item in data["startup_sequence"] if item["step_id"] == "handle_lock_busy")
    assert all(
        text in busy["description"]
        for text in (
            "do not open mutable state store",
            "do not create RuntimeSession",
            "do not initialize adapters",
        )
    )


@pytest.mark.parametrize(
    "mutation",
    [
        ("acceptance_authority", "required", "self hash is enough"),
        ("acceptance_authority", "CoreHost_may_mint", True),
        ("acceptance_authority", "registry_binding", ["claim_fingerprint_sha256"]),
        ("one_shot_transition", "consumes", ["bootstrap_generation"]),
        ("terminal_fencing", "one_shot_rule", "reusable"),
        ("terminal_fencing", "after_completed_setup", True),
        ("terminal_fencing", "forbidden_authorizations", ["LIVE"]),
        (
            "startup_identity_resolution",
            "EXISTING_INSTALLATION",
            "requires first-run provisioning handoff",
        ),
        ("identity_boundary", "does_not_imply", ["operator_authenticated"]),
        ("transition_result_authority", "nominal_type_authority", True),
        ("scope_consumer_policy", "requires_current_post_state", False),
        ("scope_consumer_policy", "requires_exact_pre_to_post_transition", False),
        ("acceptance_authority", "registry_key_binding", "dict key only"),
        ("transition_result_authority", "registry_key_equals_internal_state_fingerprint", False),
        ("transition_result_authority", "consumed_history_validation", "nominal type only"),
        ("core_current_state_authority", "first_operator_presence_states", ["PRESENT"]),
        (
            "core_current_state_authority",
            "lifecycle_presence_invariants",
            {"PRE_INITIAL_SECURITY": "PRESENT"},
        ),
    ],
)
def test_bootstrap_semantic_machine_mutations_fail_contract_attestation(
    mutation: tuple[str, str, Any],
) -> None:
    root = deepcopy(load_contract()["first_run_bootstrap_authority_contract"])
    section, field, value = mutation
    root[section][field] = value
    assert validate_bootstrap_machine_root(root) == "CONTRACT_INCONSISTENT"


def test_bootstrap_root_remains_exactly_compatible_with_m02() -> None:
    vocabulary = json.loads(
        (
            ROOT
            / "docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json"
        ).read_text()
    )
    entities = {item["canonical_name"]: item for item in vocabulary["entity_kinds"]}
    assert (
        entities["CryptoHunterAccount"]["id_field"],
        entities["CryptoHunterAccount"]["id_prefix"],
    ) == ("account_id", "acct")
    assert (
        entities["DeviceInstallation"]["id_field"],
        entities["DeviceInstallation"]["id_prefix"],
        entities["DeviceInstallation"]["parent"],
    ) == ("device_installation_id", "dev", "CryptoHunterAccount")
    assert (
        entities["OperatorIdentity"]["id_field"],
        entities["OperatorIdentity"]["id_prefix"],
        entities["OperatorIdentity"]["parent"],
    ) == ("operator_id", "op", "CryptoHunterAccount")
    relationships = {
        (item["from"], item["to"], item["cardinality"]) for item in vocabulary["relationships"]
    }
    assert ("DeviceInstallation", "RuntimeSession", "one_to_many") in relationships
    assert ("DeviceInstallation", "AuditEvent", "one_to_many") in relationships
    assert vocabulary["identifier_policy"]["persistent_id_format"] == "<prefix>_<uuidv7>"


def test_manual_success_transition_result_is_not_authority() -> None:
    consumed = ConsumedBootstrapAuthority(
        "acct_018f0000-0000-7000-8000-000000000001",
        "dev_018f0000-0000-7000-8000-000000000002",
        1,
        1,
        "a" * 64,
        "b" * 64,
    )
    forged = BootstrapTransitionResult(
        "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED",
        "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
        "c" * 64,
        "d" * 64,
        consumed,
    )
    assert (
        consume_bootstrap_result_for_purpose(forged, "INITIAL_SECURITY_ESTABLISHMENT_ONLY")
        == "BOOTSTRAP_SCOPE_DENIED"
    )


def test_genuine_transition_result_is_revalidated_and_accepted() -> None:
    claim, pre_ref = _genuine_authority()
    result = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert (
        consume_bootstrap_result_for_purpose(result, "INITIAL_SECURITY_ESTABLISHMENT_ONLY")
        == "BOOTSTRAP_PURPOSE_ACCEPTED"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("pre_state_fingerprint_sha256", "f" * 64),
        ("post_state_fingerprint_sha256", "e" * 64),
        ("consumed_authority", None),
    ],
)
def test_genuine_transition_result_tamper_is_denied(field: str, value: Any) -> None:
    claim, pre_ref = _genuine_authority()
    result = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert (
        consume_bootstrap_result_for_purpose(
            replace(result, **{field: value}), "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        )
        == "BOOTSTRAP_SCOPE_DENIED"
    )  # type: ignore[arg-type]


def test_different_consumed_binding_and_noncurrent_post_are_denied() -> None:
    claim, pre_ref = _genuine_authority()
    result = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert result.consumed_authority is not None
    wrong = replace(result.consumed_authority, bootstrap_revision=2)
    assert (
        consume_bootstrap_result_for_purpose(
            replace(result, consumed_authority=wrong), "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        )
        == "BOOTSTRAP_SCOPE_DENIED"
    )
    _CURRENT_CORE_STATE_BY_SCOPE[(claim.account_id, claim.device_installation_id)] = pre_ref
    assert (
        consume_bootstrap_result_for_purpose(result, "INITIAL_SECURITY_ESTABLISHMENT_ONLY")
        == "BOOTSTRAP_SCOPE_DENIED"
    )


def test_self_hashed_unaccepted_post_and_coordinated_extra_post_mutation_are_denied() -> None:
    claim, pre_ref = _genuine_authority()
    result = authorize_and_consume_first_run_bootstrap(
        claim, pre_ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    post = _ACCEPTED_CORE_STATES[cast(str, result.post_state_fingerprint_sha256)]
    forged = replace(post, state_fingerprint_sha256="0" * 64, expected_revision=2)
    forged = replace(forged, state_fingerprint_sha256=_state_fingerprint(forged))
    forged_result = replace(result, post_state_fingerprint_sha256=forged.state_fingerprint_sha256)
    assert (
        consume_bootstrap_result_for_purpose(forged_result, "INITIAL_SECURITY_ESTABLISHMENT_ONLY")
        == "BOOTSTRAP_SCOPE_DENIED"
    )
    _ACCEPTED_CORE_STATES[forged.state_fingerprint_sha256] = forged
    _CURRENT_CORE_STATE_BY_SCOPE[(forged.account_id, forged.device_installation_id)] = (
        forged.state_fingerprint_sha256
    )
    assert (
        consume_bootstrap_result_for_purpose(forged_result, "INITIAL_SECURITY_ESTABLISHMENT_ONLY")
        == "BOOTSTRAP_SCOPE_DENIED"
    )


def test_membership_internal_claim_fingerprint_is_exact() -> None:
    claim, ref = _genuine_authority()
    binding = _ACCEPTED_PROVISIONING_MEMBERSHIPS[claim.claim_fingerprint_sha256]
    _ACCEPTED_PROVISIONING_MEMBERSHIPS[claim.claim_fingerprint_sha256] = replace(
        binding, claim_fingerprint_sha256="c" * 64
    )
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "BOOTSTRAP_AUTHORITY_DENIED"
    )


def test_core_registry_key_must_equal_internal_and_recomputed_fingerprint() -> None:
    claim, ref = _genuine_authority()
    state = _ACCEPTED_CORE_STATES[ref]
    changed = replace(state, state_fingerprint_sha256="0" * 64, state_revision=2)
    changed = replace(changed, state_fingerprint_sha256=_state_fingerprint(changed))
    _ACCEPTED_CORE_STATES[ref] = changed
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "CONTRACT_INCONSISTENT"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("account_id", "acct_bad"),
        ("device_installation_id", "dev_bad"),
        ("bootstrap_generation", True),
        ("bootstrap_generation", 0),
        ("bootstrap_revision", True),
        ("bootstrap_revision", 0),
        ("claim_fingerprint_sha256", "bad"),
        ("challenge_fingerprint_sha256", "bad"),
    ],
)
def test_consumed_authority_full_validation(field: str, value: Any) -> None:
    valid = ConsumedBootstrapAuthority(
        "acct_018f0000-0000-7000-8000-000000000001",
        "dev_018f0000-0000-7000-8000-000000000002",
        1,
        1,
        "a" * 64,
        "b" * 64,
    )
    assert not validate_consumed_bootstrap_authority(replace(valid, **{field: value}))  # type: ignore[arg-type]


def test_consumed_history_scope_duplicates_and_generation_collisions_fail() -> None:
    state = _bootstrap_state()
    valid = ConsumedBootstrapAuthority(
        state.account_id, state.device_installation_id, 1, 1, "a" * 64, "b" * 64
    )
    wrong_account = replace(valid, account_id="acct_018f0000-0000-7000-8000-000000000099")
    wrong_device = replace(valid, device_installation_id="dev_018f0000-0000-7000-8000-000000000099")
    assert not validate_consumed_bootstrap_authority(wrong_account, state)
    assert not validate_consumed_bootstrap_authority(wrong_device, state)
    assert not _validate_consumed_history(replace(state, consumed_authorities=(valid, valid)))
    collision = replace(valid, bootstrap_revision=2, claim_fingerprint_sha256="c" * 64)
    assert not _validate_consumed_history(replace(state, consumed_authorities=(valid, collision)))


def test_pre_lifecycle_requires_first_operator_absent() -> None:
    claim, state = (
        _bootstrap_claim(),
        replace(
            _bootstrap_state(), state_fingerprint_sha256="0" * 64, first_operator_presence="PRESENT"
        ),
    )
    state = replace(state, state_fingerprint_sha256=_state_fingerprint(state))
    ref = _trusted_provisioning_fixture(claim, state)
    assert (
        authorize_and_consume_first_run_bootstrap(
            claim, ref, "2026-08-10T10:01:00Z", "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
        ).outcome
        == "MALFORMED_CORE_BOOTSTRAP_STATE"
    )


# Pure semantic model for the M0.3 protected external restore-freshness boundary.
@dataclass(frozen=True)
class RestoreFreshnessAuthority:
    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    lifecycle: str
    committed_generation: int | None
    committed_state_fingerprint_sha256: str | None
    prepared_generation: int | None
    prepared_state_fingerprint_sha256: str | None
    prepared_transaction_fingerprint_sha256: str | None
    authority_revision: int
    authority_source: str
    content_fingerprint_sha256: str


@dataclass(frozen=True)
class _LocalDurableStateEvidence:
    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    durability_state: str
    evidence_revision: int
    evidence_fingerprint_sha256: str


_RESTORE_AUTHORITIES: dict[str, RestoreFreshnessAuthority] = {}
_CURRENT_RESTORE_AUTHORITY_BY_SCOPE: dict[tuple[str, str, str], str] = {}
_RETIRED_RESTORE_AUTHORITY_REFERENCES: set[str] = set()
_LOCAL_DURABLE_EVIDENCE: dict[str, _LocalDurableStateEvidence] = {}
_CURRENT_LOCAL_EVIDENCE_BY_SCOPE: dict[tuple[str, str, str], str] = {}
_RESTORE_AVAILABLE = True


def _sha(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _scope() -> tuple[str, str, str]:
    return (
        "acct_018f0000-0000-7000-8000-000000000001",
        "dev_018f0000-0000-7000-8000-000000000002",
        "c" * 64,
    )


def _authority_content(value: RestoreFreshnessAuthority) -> tuple[object, ...]:
    return tuple(
        getattr(value, f.name) for f in fields(value) if f.name != "content_fingerprint_sha256"
    )


def _authority_fingerprint(value: RestoreFreshnessAuthority) -> str:
    return _fingerprint(_authority_content(value))


def _evidence_content(value: _LocalDurableStateEvidence) -> tuple[object, ...]:
    return tuple(
        getattr(value, f.name) for f in fields(value) if f.name != "evidence_fingerprint_sha256"
    )


def _evidence_fingerprint(value: _LocalDurableStateEvidence) -> str:
    return _fingerprint(_evidence_content(value))


def _valid_scope(value: object) -> bool:
    uuidv7 = r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
    return (
        isinstance(value, tuple)
        and len(value) == 3
        and isinstance(value[0], str)
        and re.fullmatch(rf"acct_{uuidv7}", value[0]) is not None
        and isinstance(value[1], str)
        and re.fullmatch(rf"dev_{uuidv7}", value[1]) is not None
        and _sha(value[2])
    )


def _valid_authority(value: RestoreFreshnessAuthority) -> bool:
    committed_absent = (
        value.committed_generation is None and value.committed_state_fingerprint_sha256 is None
    )
    committed_present = _positive(value.committed_generation) and _sha(
        value.committed_state_fingerprint_sha256
    )
    prepared_absent = (
        value.prepared_generation is None
        and value.prepared_state_fingerprint_sha256 is None
        and value.prepared_transaction_fingerprint_sha256 is None
    )
    prepared_present = (
        _positive(value.prepared_generation)
        and _sha(value.prepared_state_fingerprint_sha256)
        and _sha(value.prepared_transaction_fingerprint_sha256)
    )
    lifecycle_ok = bool(
        value.lifecycle == "UNINITIALIZED"
        and committed_absent
        and prepared_absent
        or value.lifecycle == "PREPARED"
        and prepared_present
        and (
            committed_absent
            and value.prepared_generation == 1
            or committed_present
            and value.prepared_generation == cast(int, value.committed_generation) + 1
        )
        or value.lifecycle == "COMMITTED"
        and committed_present
        and prepared_absent
    )
    return (
        _valid_scope(
            (
                value.account_id,
                value.device_installation_id,
                value.state_store_identity_fingerprint_sha256,
            )
        )
        and lifecycle_ok
        and _positive(value.authority_revision)
        and value.authority_source == "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY"
        and value.content_fingerprint_sha256 == _authority_fingerprint(value)
    )


def _valid_evidence(value: _LocalDurableStateEvidence) -> bool:
    return (
        _valid_scope(
            (
                value.account_id,
                value.device_installation_id,
                value.state_store_identity_fingerprint_sha256,
            )
        )
        and _positive(value.generation)
        and _sha(value.state_fingerprint_sha256)
        and _sha(value.transaction_fingerprint_sha256)
        and value.durability_state == "DURABLE_COMMITTED"
        and _positive(value.evidence_revision)
        and value.evidence_fingerprint_sha256 == _evidence_fingerprint(value)
    )


def _install_protected_fixture(value: RestoreFreshnessAuthority, kind: str) -> str:
    if _scope() in _CURRENT_RESTORE_AUTHORITY_BY_SCOPE:
        raise ValueError("initial protected membership already exists for scope")
    value = replace(value, content_fingerprint_sha256=_authority_fingerprint(value))
    reference = _fingerprint((kind, value.content_fingerprint_sha256))
    _RESTORE_AUTHORITIES[reference] = value
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope_from_authority(value)] = reference
    return reference


def _initial_protected_scope_membership_fixture() -> str:
    value = RestoreFreshnessAuthority(
        *_scope(),
        "UNINITIALIZED",
        None,
        None,
        None,
        None,
        None,
        1,
        "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY",
        "0" * 64,
    )
    return _install_protected_fixture(value, "initial-protected-scope-membership")


def _preexisting_committed_restore_authority_fixture(
    generation: int, state_fingerprint: str
) -> str:
    # TEST SETUP ONLY; NOT A LEGAL PROVISIONING OR ADVANCE TRANSITION.
    value = RestoreFreshnessAuthority(
        *_scope(),
        "COMMITTED",
        generation,
        state_fingerprint,
        None,
        None,
        None,
        1,
        "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY",
        "0" * 64,
    )
    return _install_protected_fixture(value, "preexisting-committed-test-setup")


def _trusted_external_reference_replacement_fixture(old_reference: object) -> str | None:
    current = _resolve_authority(old_reference)
    if current is None or current.lifecycle not in {"UNINITIALIZED", "COMMITTED"}:
        return None
    replacement = replace(
        current,
        authority_revision=current.authority_revision + 1,
        content_fingerprint_sha256="0" * 64,
    )
    replacement = replace(
        replacement, content_fingerprint_sha256=_authority_fingerprint(replacement)
    )
    new_reference = _fingerprint(
        (
            "protected-reference-replacement",
            cast(str, old_reference),
            replacement.content_fingerprint_sha256,
        )
    )
    _RESTORE_AUTHORITIES[new_reference] = replacement
    _RETIRED_RESTORE_AUTHORITY_REFERENCES.add(cast(str, old_reference))
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope_from_authority(replacement)] = new_reference
    return new_reference


def _malformed_protected_authority_fixture(value: RestoreFreshnessAuthority) -> str:
    value = replace(value, content_fingerprint_sha256=_authority_fingerprint(value))
    reference = _fingerprint(
        ("malformed-protected-scope-membership", value.content_fingerprint_sha256)
    )
    _RESTORE_AUTHORITIES[reference] = value
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope_from_authority(value)] = reference
    return reference


def _trusted_local_durable_evidence_fixture(
    generation: int,
    state_fingerprint: str,
    transaction_fingerprint: str,
    *,
    scope: tuple[str, str, str] | None = None,
    durability_state: str = "DURABLE_COMMITTED",
    make_current: bool = True,
) -> str:
    exact_scope = _scope() if scope is None else scope
    value = _LocalDurableStateEvidence(
        *exact_scope,
        generation,
        state_fingerprint,
        transaction_fingerprint,
        durability_state,
        1,
        "0" * 64,
    )
    value = replace(value, evidence_fingerprint_sha256=_evidence_fingerprint(value))
    reference = _fingerprint(
        ("local-durable-evidence", value.evidence_fingerprint_sha256, len(_LOCAL_DURABLE_EVIDENCE))
    )
    _LOCAL_DURABLE_EVIDENCE[reference] = value
    if make_current:
        _CURRENT_LOCAL_EVIDENCE_BY_SCOPE[exact_scope] = reference
    return reference


def _resolve_authority(reference: object) -> RestoreFreshnessAuthority | None:
    if not _RESTORE_AVAILABLE or not isinstance(reference, str):
        return None
    value = _RESTORE_AUTHORITIES.get(reference)
    if reference in _RETIRED_RESTORE_AUTHORITY_REFERENCES:
        return None
    if value is None or not _valid_authority(value):
        return None
    return (
        value
        if _CURRENT_RESTORE_AUTHORITY_BY_SCOPE.get(_scope_from_authority(value)) == reference
        else None
    )


def _resolve_current_evidence(reference: object) -> _LocalDurableStateEvidence | None:
    if not isinstance(reference, str):
        return None
    value = _LOCAL_DURABLE_EVIDENCE.get(reference)
    if value is None or not _valid_evidence(value):
        return None
    scope = (
        value.account_id,
        value.device_installation_id,
        value.state_store_identity_fingerprint_sha256,
    )
    return value if _CURRENT_LOCAL_EVIDENCE_BY_SCOPE.get(scope) == reference else None


def prepare_restore_transition(
    reference: object,
    scope: object,
    expected_generation: object,
    expected_state_fingerprint: object,
    next_generation: object,
    candidate_state_fingerprint: object,
    transaction_fingerprint: object,
) -> str:
    current = _resolve_authority(reference)
    if current is None:
        return "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    if not _valid_scope(scope) or scope != _scope_from_authority(current):
        return "RESTORE_SCOPE_MISMATCH"
    if (
        not _positive(next_generation)
        or not _sha(candidate_state_fingerprint)
        or not _sha(transaction_fingerprint)
    ):
        return "RESTORE_CANDIDATE_MALFORMED"
    if current.lifecycle == "UNINITIALIZED":
        if (
            expected_generation is not None
            or expected_state_fingerprint is not None
            or next_generation != 1
        ):
            return "RESTORE_GENESIS_DENIED"
    else:
        if current.lifecycle == "PREPARED":
            pending = (
                current.prepared_generation,
                current.prepared_state_fingerprint_sha256,
                current.prepared_transaction_fingerprint_sha256,
            )
            candidate = (next_generation, candidate_state_fingerprint, transaction_fingerprint)
            if current.committed_generation is None:
                if expected_generation is not None or expected_state_fingerprint is not None:
                    return "RESTORE_EXPECTED_CURRENT_MISMATCH"
            elif (
                expected_generation != current.committed_generation
                or expected_state_fingerprint != current.committed_state_fingerprint_sha256
            ):
                return "RESTORE_EXPECTED_CURRENT_MISMATCH"
            return "RESTORE_PREPARED" if pending == candidate else "RESTORE_INCOMPATIBLE_PREPARE"
        if (
            expected_generation != current.committed_generation
            or expected_state_fingerprint != current.committed_state_fingerprint_sha256
        ):
            return "RESTORE_EXPECTED_CURRENT_MISMATCH"
        if next_generation != cast(int, current.committed_generation) + 1:
            return "RESTORE_GENERATION_DENIED"
    updated = replace(
        current,
        lifecycle="PREPARED",
        prepared_generation=cast(int, next_generation),
        prepared_state_fingerprint_sha256=cast(str, candidate_state_fingerprint),
        prepared_transaction_fingerprint_sha256=cast(str, transaction_fingerprint),
        authority_revision=current.authority_revision + 1,
        content_fingerprint_sha256="0" * 64,
    )
    _RESTORE_AUTHORITIES[cast(str, reference)] = replace(
        updated, content_fingerprint_sha256=_authority_fingerprint(updated)
    )
    return "RESTORE_PREPARED"


def _scope_from_authority(value: RestoreFreshnessAuthority) -> tuple[str, str, str]:
    return (
        value.account_id,
        value.device_installation_id,
        value.state_store_identity_fingerprint_sha256,
    )


def finalize_restore_transition(
    reference: object, scope: object, evidence_reference: object
) -> str:
    current = _resolve_authority(reference)
    evidence = _resolve_current_evidence(evidence_reference)
    if (
        current is None
        or current.lifecycle != "PREPARED"
        or evidence is None
        or not _valid_scope(scope)
    ):
        return "RESTORE_FINALIZE_DENIED"
    exact = (
        evidence.account_id,
        evidence.device_installation_id,
        evidence.state_store_identity_fingerprint_sha256,
        evidence.generation,
        evidence.state_fingerprint_sha256,
        evidence.transaction_fingerprint_sha256,
    )
    required = (
        *_scope_from_authority(current),
        current.prepared_generation,
        current.prepared_state_fingerprint_sha256,
        current.prepared_transaction_fingerprint_sha256,
    )
    if scope != _scope_from_authority(current) or exact != required:
        return "RESTORE_FINALIZE_DENIED"
    updated = replace(
        current,
        lifecycle="COMMITTED",
        committed_generation=evidence.generation,
        committed_state_fingerprint_sha256=evidence.state_fingerprint_sha256,
        prepared_generation=None,
        prepared_state_fingerprint_sha256=None,
        prepared_transaction_fingerprint_sha256=None,
        authority_revision=current.authority_revision + 1,
        content_fingerprint_sha256="0" * 64,
    )
    _RESTORE_AUTHORITIES[cast(str, reference)] = replace(
        updated, content_fingerprint_sha256=_authority_fingerprint(updated)
    )
    return "RESTORE_FINALIZED"


def abort_restore_transition(reference: object, scope: object, evidence_reference: object) -> str:
    current = _resolve_authority(reference)
    evidence = _resolve_current_evidence(evidence_reference)
    if (
        current is None
        or current.lifecycle != "PREPARED"
        or current.committed_generation is None
        or evidence is None
        or not _valid_scope(scope)
    ):
        return "RESTORE_ABORT_DENIED"
    exact = (
        *_scope_from_authority(current),
        current.committed_generation,
        current.committed_state_fingerprint_sha256,
    )
    observed = (
        evidence.account_id,
        evidence.device_installation_id,
        evidence.state_store_identity_fingerprint_sha256,
        evidence.generation,
        evidence.state_fingerprint_sha256,
    )
    if scope != _scope_from_authority(current) or observed != exact:
        return "RESTORE_ABORT_DENIED"
    updated = replace(
        current,
        lifecycle="COMMITTED",
        prepared_generation=None,
        prepared_state_fingerprint_sha256=None,
        prepared_transaction_fingerprint_sha256=None,
        authority_revision=current.authority_revision + 1,
        content_fingerprint_sha256="0" * 64,
    )
    _RESTORE_AUTHORITIES[cast(str, reference)] = replace(
        updated, content_fingerprint_sha256=_authority_fingerprint(updated)
    )
    return "RESTORE_ABORTED"


def reconcile_restore_authority(
    reference: object, scope: object, evidence_reference: object
) -> str:
    current = _resolve_authority(reference)
    evidence = _resolve_current_evidence(evidence_reference)
    if current is None or not _valid_scope(scope) or scope != _scope_from_authority(current):
        return "NO_READY_AUTHORITY"
    if current.lifecycle == "UNINITIALIZED":
        return "NO_CURRENT_STATESTORE_AUTHORITY"
    if (
        current.lifecycle == "PREPARED"
        and current.committed_generation is None
        and evidence is None
    ):
        return "GENESIS_PENDING_RECOVERY_REQUIRED"
    if evidence is None:
        return "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"
    local = (evidence.generation, evidence.state_fingerprint_sha256)
    committed = (current.committed_generation, current.committed_state_fingerprint_sha256)
    pending = (
        current.prepared_generation,
        current.prepared_state_fingerprint_sha256,
        current.prepared_transaction_fingerprint_sha256,
    )
    if current.lifecycle == "COMMITTED" and local == committed:
        return "CURRENT_AUTHORITY"
    if current.lifecycle == "PREPARED" and local == committed:
        return "ABORT_PENDING_THROUGH_PROTECTED_PROTOCOL"
    if (
        current.lifecycle == "PREPARED"
        and (
            evidence.generation,
            evidence.state_fingerprint_sha256,
            evidence.transaction_fingerprint_sha256,
        )
        == pending
    ):
        return "FINALIZE_MATCHING_PENDING"
    return "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"


@pytest.fixture(autouse=True)
def _clear_restore_authority_model() -> None:
    global _RESTORE_AVAILABLE
    _RESTORE_AUTHORITIES.clear()
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE.clear()
    _RETIRED_RESTORE_AUTHORITY_REFERENCES.clear()
    _LOCAL_DURABLE_EVIDENCE.clear()
    _CURRENT_LOCAL_EVIDENCE_BY_SCOPE.clear()
    _RESTORE_AVAILABLE = True


def test_restore_machine_root_closes_genesis_evidence_abort_and_separation() -> None:
    root = load_contract()["restore_freshness_authority_contract"]
    assert root["authority_owner"] == "EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY"
    assert root["scope_membership_lifecycle"]["states"] == [
        "UNINITIALIZED",
        "PREPARED",
        "COMMITTED",
    ]
    assert (
        root["local_durable_evidence_handoff"]["owner"]
        == "M0.11 local durable StateStore observation boundary"
    )
    assert root["separation"]["not_first_run_bootstrap_authority"] is True


def test_legal_genesis_requires_durable_evidence_and_reaches_committed_one() -> None:
    ref = _initial_protected_scope_membership_fixture()
    assert (
        reconcile_restore_authority(ref, _scope(), "missing") == "NO_CURRENT_STATESTORE_AUTHORITY"
    )
    assert (
        prepare_restore_transition(ref, _scope(), None, None, 1, "a" * 64, "b" * 64)
        == "RESTORE_PREPARED"
    )
    assert finalize_restore_transition(ref, _scope(), "missing") == "RESTORE_FINALIZE_DENIED"
    evidence = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZED"
    assert _RESTORE_AUTHORITIES[ref].committed_generation == 1
    assert reconcile_restore_authority(ref, _scope(), evidence) == "CURRENT_AUTHORITY"


@pytest.mark.parametrize(
    "expected_generation,expected_state,next_generation",
    [(0, None, 1), (None, "a" * 64, 1), (None, None, 0), (None, None, True), (None, None, 2)],
)
def test_genesis_rejects_fake_previous_authority_and_nonexact_one(
    expected_generation: object, expected_state: object, next_generation: object
) -> None:
    ref = _initial_protected_scope_membership_fixture()
    assert prepare_restore_transition(
        ref, _scope(), expected_generation, expected_state, next_generation, "a" * 64, "b" * 64
    ) in {"RESTORE_GENESIS_DENIED", "RESTORE_CANDIDATE_MALFORMED"}


def test_matching_strings_without_durable_membership_cannot_finalize() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    manual = _LocalDurableStateEvidence(
        *_scope(), 2, "b" * 64, "d" * 64, "DURABLE_COMMITTED", 1, "0" * 64
    )
    manual = replace(manual, evidence_fingerprint_sha256=_evidence_fingerprint(manual))
    assert finalize_restore_transition(ref, _scope(), manual) == "RESTORE_FINALIZE_DENIED"
    assert (
        finalize_restore_transition(ref, _scope(), manual.evidence_fingerprint_sha256)
        == "RESTORE_FINALIZE_DENIED"
    )
    assert _RESTORE_AUTHORITIES[ref].lifecycle == "PREPARED"
    assert _RESTORE_AUTHORITIES[ref].committed_generation == 1


@pytest.mark.parametrize(
    "scope,generation,state,transaction,durability,current",
    [
        (
            ("acct_wrong", _scope()[1], _scope()[2]),
            2,
            "b" * 64,
            "d" * 64,
            "DURABLE_COMMITTED",
            True,
        ),
        ((_scope()[0], "dev_wrong", _scope()[2]), 2, "b" * 64, "d" * 64, "DURABLE_COMMITTED", True),
        ((_scope()[0], _scope()[1], "e" * 64), 2, "b" * 64, "d" * 64, "DURABLE_COMMITTED", True),
        (_scope(), 3, "b" * 64, "d" * 64, "DURABLE_COMMITTED", True),
        (_scope(), 2, "e" * 64, "d" * 64, "DURABLE_COMMITTED", True),
        (_scope(), 2, "b" * 64, "e" * 64, "DURABLE_COMMITTED", True),
        (_scope(), 2, "b" * 64, "d" * 64, "PREPARED_ONLY", True),
        (_scope(), 2, "b" * 64, "d" * 64, "DURABLE_COMMITTED", False),
    ],
)
def test_wrong_non_durable_and_stale_evidence_cannot_finalize(
    scope: tuple[str, str, str],
    generation: int,
    state: str,
    transaction: str,
    durability: str,
    current: bool,
) -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    evidence = _trusted_local_durable_evidence_fixture(
        generation,
        state,
        transaction,
        scope=scope,
        durability_state=durability,
        make_current=current,
    )
    assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZE_DENIED"
    assert _RESTORE_AUTHORITIES[ref].committed_generation == 1


def test_exact_accepted_current_durable_evidence_finalizes_existing_installation() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    evidence = _trusted_local_durable_evidence_fixture(2, "b" * 64, "d" * 64)
    assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZED"


def test_executable_abort_clears_pending_and_allows_different_candidate() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    committed = _trusted_local_durable_evidence_fixture(1, "a" * 64, "c" * 64)
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    assert (
        reconcile_restore_authority(ref, _scope(), committed)
        == "ABORT_PENDING_THROUGH_PROTECTED_PROTOCOL"
    )
    revision = _RESTORE_AUTHORITIES[ref].authority_revision
    assert abort_restore_transition(ref, _scope(), committed) == "RESTORE_ABORTED"
    after = _RESTORE_AUTHORITIES[ref]
    assert (after.committed_generation, after.committed_state_fingerprint_sha256) == (1, "a" * 64)
    assert (
        after.prepared_generation,
        after.prepared_state_fingerprint_sha256,
        after.prepared_transaction_fingerprint_sha256,
    ) == (None, None, None)
    assert after.authority_revision == revision + 1
    assert after.content_fingerprint_sha256 == _authority_fingerprint(after)
    assert (
        prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "e" * 64, "f" * 64)
        == "RESTORE_PREPARED"
    )


@pytest.mark.parametrize(
    "case", ["no_pending", "candidate", "wrong_scope", "caller", "wrong_committed"]
)
def test_abort_denials_do_not_clear_or_advance(case: str) -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    committed = _trusted_local_durable_evidence_fixture(1, "a" * 64, "c" * 64)
    if case == "no_pending":
        assert abort_restore_transition(ref, _scope(), committed) == "RESTORE_ABORT_DENIED"
        return
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    evidence: object = committed
    scope: object = _scope()
    if case == "candidate":
        evidence = _trusted_local_durable_evidence_fixture(2, "b" * 64, "d" * 64)
    if case == "wrong_scope":
        scope = (_scope()[0], _scope()[1], "e" * 64)
    if case == "caller":
        evidence = _LocalDurableStateEvidence(
            *_scope(), 1, "a" * 64, "c" * 64, "DURABLE_COMMITTED", 1, "e" * 64
        )
    if case == "wrong_committed":
        evidence = _trusted_local_durable_evidence_fixture(1, "e" * 64, "c" * 64)
    assert abort_restore_transition(ref, scope, evidence) == "RESTORE_ABORT_DENIED"
    assert _RESTORE_AUTHORITIES[ref].prepared_generation == 2
    assert _RESTORE_AUTHORITIES[ref].committed_generation == 1


def test_crash_after_local_commit_before_finalize_recovers_by_exact_evidence() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
    evidence = _trusted_local_durable_evidence_fixture(2, "b" * 64, "d" * 64)
    assert reconcile_restore_authority(ref, _scope(), evidence) == "FINALIZE_MATCHING_PENDING"
    assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZED"


def test_committed_anchor_cannot_repeat_genesis_or_return_uninitialized() -> None:
    ref = _preexisting_committed_restore_authority_fixture(3, "a" * 64)
    assert (
        prepare_restore_transition(ref, _scope(), None, None, 1, "b" * 64, "d" * 64)
        == "RESTORE_EXPECTED_CURRENT_MISMATCH"
    )
    assert _RESTORE_AUTHORITIES[ref].lifecycle == "COMMITTED"


def test_missing_membership_is_not_uninitialized_and_cannot_be_self_provisioned() -> None:
    assert reconcile_restore_authority("missing", _scope(), "missing") == "NO_READY_AUTHORITY"
    uninitialized = _initial_protected_scope_membership_fixture()
    assert (
        reconcile_restore_authority(uninitialized, _scope(), "missing")
        == "NO_CURRENT_STATESTORE_AUTHORITY"
    )


def test_full_b1_rollback_after_legal_advances_is_denied() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    old = _trusted_local_durable_evidence_fixture(1, "a" * 64, "c" * 64)
    for generation, state, transaction in ((2, "b" * 64, "d" * 64), (3, "e" * 64, "f" * 64)):
        current = _RESTORE_AUTHORITIES[ref]
        assert (
            prepare_restore_transition(
                ref,
                _scope(),
                current.committed_generation,
                current.committed_state_fingerprint_sha256,
                generation,
                state,
                transaction,
            )
            == "RESTORE_PREPARED"
        )
        evidence = _trusted_local_durable_evidence_fixture(generation, state, transaction)
        assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZED"
    _CURRENT_LOCAL_EVIDENCE_BY_SCOPE[_scope()] = old
    assert (
        reconcile_restore_authority(ref, _scope(), old)
        == "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"
    )
    assert _RESTORE_AUTHORITIES[ref].committed_generation == 3


@pytest.mark.parametrize(
    "function,args",
    [
        (
            prepare_restore_transition,
            (object(), object(), object(), object(), object(), object(), object()),
        ),
        (finalize_restore_transition, (object(), object(), object())),
        (abort_restore_transition, (object(), object(), object())),
        (reconcile_restore_authority, (object(), object(), object())),
    ],
)
def test_malformed_protocol_inputs_fail_closed_without_python_exception(
    function: Any, args: tuple[object, ...]
) -> None:
    assert function(*args) in {
        "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED",
        "RESTORE_FINALIZE_DENIED",
        "RESTORE_ABORT_DENIED",
        "NO_READY_AUTHORITY",
    }


def test_non_authorities_implementation_neutrality_handoff_and_live_denial() -> None:
    root = load_contract()["restore_freshness_authority_contract"]
    assert {
        "BackupEnvelope",
        "first_run_bootstrap_authority_contract",
        "RuntimeSession",
        "DesktopShell",
        "TrayAgent",
    } <= set(root["non_authorities"])
    assert "implementation_mechanisms_not_selected_by_contract" in root["location"]
    assert root["location"]["conforming_implementation_may_use_any_mechanism"] is True
    assert root["milestone_ownership"]["M0.11_may_mint_M0.3_membership"] is False
    assert root["current_live_policy"] == "DENIED"


@pytest.mark.parametrize(
    "external_generation,local_generation,local_state",
    [(2, 1, "a" * 64), (4, 1, "a" * 64), (1, 2, "b" * 64)],
)
def test_external_ahead_and_store_ahead_without_pending_remain_fail_closed(
    external_generation: int, local_generation: int, local_state: str
) -> None:
    external_state = "e" * 64 if external_generation > 1 else "a" * 64
    ref = _preexisting_committed_restore_authority_fixture(external_generation, external_state)
    evidence = _trusted_local_durable_evidence_fixture(local_generation, local_state, "d" * 64)
    assert (
        reconcile_restore_authority(ref, _scope(), evidence)
        == "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"
    )


@pytest.mark.parametrize(
    "next_generation",
    [0, True, 1, 3],
)
def test_existing_committed_generation_cannot_rollback_skip_or_repeat(
    next_generation: object,
) -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    assert prepare_restore_transition(
        ref, _scope(), 1, "a" * 64, next_generation, "b" * 64, "d" * 64
    ) in {"RESTORE_CANDIDATE_MALFORMED", "RESTORE_GENERATION_DENIED"}


def test_incompatible_second_prepare_and_missing_external_authority_remain_denied() -> None:
    ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    assert (
        prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "b" * 64, "d" * 64)
        == "RESTORE_PREPARED"
    )
    assert (
        prepare_restore_transition(ref, _scope(), 1, "a" * 64, 2, "e" * 64, "f" * 64)
        == "RESTORE_INCOMPATIBLE_PREPARE"
    )
    assert (
        prepare_restore_transition("missing", _scope(), None, None, 1, "b" * 64, "d" * 64)
        == "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        {"lifecycle": "UNINITIALIZED", "prepared_state_fingerprint_sha256": "a" * 64},
        {"lifecycle": "UNINITIALIZED", "prepared_transaction_fingerprint_sha256": "b" * 64},
        {
            "lifecycle": "PREPARED",
            "committed_state_fingerprint_sha256": "c" * 64,
            "prepared_generation": 1,
            "prepared_state_fingerprint_sha256": "a" * 64,
            "prepared_transaction_fingerprint_sha256": "b" * 64,
        },
        {
            "lifecycle": "COMMITTED",
            "committed_generation": 1,
            "committed_state_fingerprint_sha256": "c" * 64,
            "prepared_state_fingerprint_sha256": "a" * 64,
        },
        {
            "lifecycle": "COMMITTED",
            "committed_generation": 1,
            "committed_state_fingerprint_sha256": "c" * 64,
            "prepared_transaction_fingerprint_sha256": "b" * 64,
        },
        {"lifecycle": "COMMITTED", "committed_generation": 1},
        {"lifecycle": "COMMITTED", "committed_state_fingerprint_sha256": "c" * 64},
        {"lifecycle": "PREPARED", "prepared_generation": 1},
        {
            "lifecycle": "PREPARED",
            "prepared_state_fingerprint_sha256": "a" * 64,
            "prepared_transaction_fingerprint_sha256": "b" * 64,
        },
        {"lifecycle": "UNKNOWN"},
    ],
)
def test_recomputed_hash_cannot_legalize_malformed_authority_projection(
    mutation: dict[str, object],
) -> None:
    base_ref = _initial_protected_scope_membership_fixture()
    malformed = replace(
        _RESTORE_AUTHORITIES[base_ref],
        **mutation,  # type: ignore[arg-type]
        content_fingerprint_sha256="0" * 64,
    )
    reference = _malformed_protected_authority_fixture(malformed)
    assert _resolve_authority(reference) is None
    assert (
        prepare_restore_transition(reference, _scope(), None, None, 1, "a" * 64, "b" * 64)
        == "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    )
    assert reconcile_restore_authority(reference, _scope(), "missing") == "NO_READY_AUTHORITY"


def test_genesis_crash_without_local_evidence_retains_exact_pending_fence() -> None:
    ref = _initial_protected_scope_membership_fixture()
    assert (
        prepare_restore_transition(ref, _scope(), None, None, 1, "a" * 64, "b" * 64)
        == "RESTORE_PREPARED"
    )
    pending = _RESTORE_AUTHORITIES[ref]
    assert (
        reconcile_restore_authority(ref, _scope(), "missing") == "GENESIS_PENDING_RECOVERY_REQUIRED"
    )
    assert abort_restore_transition(ref, _scope(), "missing") == "RESTORE_ABORT_DENIED"
    assert (
        prepare_restore_transition(ref, _scope(), None, None, 1, "c" * 64, "d" * 64)
        == "RESTORE_INCOMPATIBLE_PREPARE"
    )
    assert _RESTORE_AUTHORITIES[ref] == pending


def test_genesis_pending_resumes_only_with_exact_durable_evidence() -> None:
    ref = _initial_protected_scope_membership_fixture()
    prepare_restore_transition(ref, _scope(), None, None, 1, "a" * 64, "b" * 64)
    evidence = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    assert reconcile_restore_authority(ref, _scope(), evidence) == "FINALIZE_MATCHING_PENDING"
    assert finalize_restore_transition(ref, _scope(), evidence) == "RESTORE_FINALIZED"
    assert _RESTORE_AUTHORITIES[ref].lifecycle == "COMMITTED"


def test_genesis_local_mismatch_fails_closed_and_preserves_pending() -> None:
    ref = _initial_protected_scope_membership_fixture()
    prepare_restore_transition(ref, _scope(), None, None, 1, "a" * 64, "b" * 64)
    pending = _RESTORE_AUTHORITIES[ref]
    mismatch = _trusted_local_durable_evidence_fixture(1, "c" * 64, "d" * 64)
    assert (
        reconcile_restore_authority(ref, _scope(), mismatch)
        == "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"
    )
    assert finalize_restore_transition(ref, _scope(), mismatch) == "RESTORE_FINALIZE_DENIED"
    assert abort_restore_transition(ref, _scope(), mismatch) == "RESTORE_ABORT_DENIED"
    assert (
        prepare_restore_transition(ref, _scope(), None, None, 1, "c" * 64, "d" * 64)
        == "RESTORE_INCOMPATIBLE_PREPARE"
    )
    assert _RESTORE_AUTHORITIES[ref] == pending


@pytest.mark.parametrize(
    "genesis,expected_generation,expected_state,expected_outcome",
    [
        (True, None, None, "RESTORE_PREPARED"),
        (True, 9, None, "RESTORE_EXPECTED_CURRENT_MISMATCH"),
        (True, None, "f" * 64, "RESTORE_EXPECTED_CURRENT_MISMATCH"),
        (False, 1, "a" * 64, "RESTORE_PREPARED"),
        (False, 9, "a" * 64, "RESTORE_EXPECTED_CURRENT_MISMATCH"),
        (False, 1, "f" * 64, "RESTORE_EXPECTED_CURRENT_MISMATCH"),
    ],
)
def test_idempotent_prepare_retry_requires_exact_expected_current(
    genesis: bool,
    expected_generation: int | None,
    expected_state: str | None,
    expected_outcome: str,
) -> None:
    ref = (
        _initial_protected_scope_membership_fixture()
        if genesis
        else _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    )
    initial_generation = None if genesis else 1
    initial_state = None if genesis else "a" * 64
    next_generation = 1 if genesis else 2
    assert (
        prepare_restore_transition(
            ref,
            _scope(),
            initial_generation,
            initial_state,
            next_generation,
            "b" * 64,
            "d" * 64,
        )
        == "RESTORE_PREPARED"
    )
    pending = _RESTORE_AUTHORITIES[ref]
    assert (
        prepare_restore_transition(
            ref,
            _scope(),
            expected_generation,
            expected_state,
            next_generation,
            "b" * 64,
            "d" * 64,
        )
        == expected_outcome
    )
    assert _RESTORE_AUTHORITIES[ref] == pending


def test_trusted_external_replacement_then_legal_g1_g2_g3_denies_old_b1() -> None:
    old_ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    old_b1 = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    new_ref = _trusted_external_reference_replacement_fixture(old_ref)
    assert isinstance(new_ref, str)
    assert _RESTORE_AUTHORITIES[new_ref].committed_generation == 1
    assert _RESTORE_AUTHORITIES[new_ref].committed_state_fingerprint_sha256 == "a" * 64
    for generation, state, transaction in ((2, "c" * 64, "d" * 64), (3, "e" * 64, "f" * 64)):
        current = _RESTORE_AUTHORITIES[new_ref]
        assert (
            prepare_restore_transition(
                new_ref,
                _scope(),
                current.committed_generation,
                current.committed_state_fingerprint_sha256,
                generation,
                state,
                transaction,
            )
            == "RESTORE_PREPARED"
        )
        evidence = _trusted_local_durable_evidence_fixture(generation, state, transaction)
        assert finalize_restore_transition(new_ref, _scope(), evidence) == "RESTORE_FINALIZED"
    _CURRENT_LOCAL_EVIDENCE_BY_SCOPE[_scope()] = old_b1
    assert reconcile_restore_authority(old_ref, _scope(), old_b1) == "NO_READY_AUTHORITY"
    assert (
        reconcile_restore_authority(new_ref, _scope(), old_b1)
        == "FAIL_CLOSED_AUTHORITATIVE_RESTORE_REQUIRED"
    )


def test_stale_external_reference_cannot_prepare_finalize_abort_or_reconcile() -> None:
    old_ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    committed = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    new_ref = _trusted_external_reference_replacement_fixture(old_ref)
    assert isinstance(new_ref, str)
    old_snapshot = _RESTORE_AUTHORITIES[old_ref]
    assert (
        prepare_restore_transition(old_ref, _scope(), 1, "a" * 64, 2, "c" * 64, "d" * 64)
        == "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    )
    assert finalize_restore_transition(old_ref, _scope(), committed) == "RESTORE_FINALIZE_DENIED"
    assert abort_restore_transition(old_ref, _scope(), committed) == "RESTORE_ABORT_DENIED"
    assert reconcile_restore_authority(old_ref, _scope(), committed) == "NO_READY_AUTHORITY"
    assert _RESTORE_AUTHORITIES[old_ref] == old_snapshot
    assert _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope()] == new_ref


def test_public_consumer_cannot_reselect_stale_current_designation() -> None:
    old_ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    new_ref = _trusted_external_reference_replacement_fixture(old_ref)
    assert isinstance(new_ref, str)
    public = {
        name for name, value in globals().items() if callable(value) and not name.startswith("_")
    }
    assert (
        not {
            "set_current_restore_authority",
            "select_current_membership",
            "replace_restore_authority",
            "rotate_anchor",
            "force_current",
            "prefer_newer",
        }
        & public
    )
    assert _resolve_authority(old_ref) is None
    assert _resolve_authority(new_ref) == _RESTORE_AUTHORITIES[new_ref]


@pytest.mark.parametrize(
    "account_id,device_installation_id",
    [
        ("acct_wrong", _scope()[1]),
        (_scope()[0], "dev_wrong"),
        ("acct_", _scope()[1]),
        (_scope()[0], "dev_"),
        ("acct_018F0000-0000-7000-8000-000000000001", _scope()[1]),
        (_scope()[0], "dev_018F0000-0000-7000-8000-000000000002"),
        ("acct_018f0000-0000-4000-8000-000000000001", _scope()[1]),
        (_scope()[0], "dev_018f0000-0000-7000-7000-000000000002"),
        ("acct_018f0000-0000-7000-8000-00000000001", _scope()[1]),
        (_scope()[0], "dev_018f0000-0000-7000-8000-000000000002x"),
        ("dev_018f0000-0000-7000-8000-000000000001", _scope()[1]),
        (_scope()[0], "acct_018f0000-0000-7000-8000-000000000002"),
    ],
)
def test_recomputed_hash_cannot_legalize_noncanonical_m02_scope(
    account_id: str, device_installation_id: str
) -> None:
    valid_ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    malformed = replace(
        _RESTORE_AUTHORITIES[valid_ref],
        account_id=account_id,
        device_installation_id=device_installation_id,
        content_fingerprint_sha256="0" * 64,
    )
    reference = _malformed_protected_authority_fixture(malformed)
    evidence = _trusted_local_durable_evidence_fixture(
        1,
        "a" * 64,
        "b" * 64,
        scope=(account_id, device_installation_id, _scope()[2]),
    )
    assert _resolve_authority(reference) is None
    assert _resolve_current_evidence(evidence) is None
    assert (
        prepare_restore_transition(
            reference,
            (account_id, device_installation_id, _scope()[2]),
            1,
            "a" * 64,
            2,
            "b" * 64,
            "c" * 64,
        )
        == "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    )
    assert finalize_restore_transition(reference, _scope(), evidence) == "RESTORE_FINALIZE_DENIED"
    assert abort_restore_transition(reference, _scope(), evidence) == "RESTORE_ABORT_DENIED"
    assert reconcile_restore_authority(reference, _scope(), evidence) == "NO_READY_AUTHORITY"


def test_exact_canonical_m02_uuidv7_scope_remains_valid_for_authority_and_evidence() -> None:
    reference = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    evidence = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    assert _resolve_authority(reference) == _RESTORE_AUTHORITIES[reference]
    assert _resolve_current_evidence(evidence) == _LOCAL_DURABLE_EVIDENCE[evidence]
    assert reconcile_restore_authority(reference, _scope(), evidence) == "CURRENT_AUTHORITY"


def test_reference_replacement_api_cannot_accept_generation_or_state_candidate() -> None:
    assert _trusted_external_reference_replacement_fixture.__code__.co_argcount == 1
    old_ref = _preexisting_committed_restore_authority_fixture(3, "c" * 64)
    with pytest.raises(ValueError, match="already exists"):
        _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    with pytest.raises(ValueError, match="already exists"):
        _preexisting_committed_restore_authority_fixture(3, "x" * 64)
    assert _resolve_authority(old_ref) == _RESTORE_AUTHORITIES[old_ref]
    assert _RESTORE_AUTHORITIES[old_ref].committed_generation == 3
    assert _RESTORE_AUTHORITIES[old_ref].committed_state_fingerprint_sha256 == "c" * 64


@pytest.mark.parametrize("genesis", [True, False])
def test_ordinary_reference_replacement_is_denied_while_prepared(genesis: bool) -> None:
    reference = (
        _initial_protected_scope_membership_fixture()
        if genesis
        else _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    )
    expected_generation = None if genesis else 1
    expected_state = None if genesis else "a" * 64
    next_generation = 1 if genesis else 2
    prepare_restore_transition(
        reference,
        _scope(),
        expected_generation,
        expected_state,
        next_generation,
        "b" * 64,
        "d" * 64,
    )
    pending = _RESTORE_AUTHORITIES[reference]
    assert _trusted_external_reference_replacement_fixture(reference) is None
    assert _RESTORE_AUTHORITIES[reference] == pending
    assert _resolve_authority(reference) == pending


def test_uninitialized_reference_replacement_preserves_empty_projection_and_genesis() -> None:
    old_ref = _initial_protected_scope_membership_fixture()
    new_ref = _trusted_external_reference_replacement_fixture(old_ref)
    assert isinstance(new_ref, str)
    replacement = _RESTORE_AUTHORITIES[new_ref]
    assert replacement.lifecycle == "UNINITIALIZED"
    assert replacement.committed_generation is None
    assert replacement.committed_state_fingerprint_sha256 is None
    assert replacement.prepared_generation is None
    assert replacement.prepared_state_fingerprint_sha256 is None
    assert replacement.prepared_transaction_fingerprint_sha256 is None
    assert _resolve_authority(old_ref) is None
    assert (
        prepare_restore_transition(new_ref, _scope(), None, None, 1, "a" * 64, "b" * 64)
        == "RESTORE_PREPARED"
    )


def test_retired_reference_survives_current_map_rollback_corruption_and_denies_all_operations() -> (
    None
):
    old_ref = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    b1 = _trusted_local_durable_evidence_fixture(1, "a" * 64, "b" * 64)
    new_ref = _trusted_external_reference_replacement_fixture(old_ref)
    assert isinstance(new_ref, str)
    assert old_ref in _RETIRED_RESTORE_AUTHORITY_REFERENCES
    old_snapshot, new_snapshot = _RESTORE_AUTHORITIES[old_ref], _RESTORE_AUTHORITIES[new_ref]
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope()] = old_ref  # corruption simulation only
    assert _resolve_authority(old_ref) is None
    assert reconcile_restore_authority(old_ref, _scope(), b1) == "NO_READY_AUTHORITY"
    assert (
        prepare_restore_transition(old_ref, _scope(), 1, "a" * 64, 2, "c" * 64, "d" * 64)
        == "RESTORE_AUTHORITY_UNAVAILABLE_OR_DENIED"
    )
    assert finalize_restore_transition(old_ref, _scope(), b1) == "RESTORE_FINALIZE_DENIED"
    assert abort_restore_transition(old_ref, _scope(), b1) == "RESTORE_ABORT_DENIED"
    assert _RESTORE_AUTHORITIES[old_ref] == old_snapshot
    assert _RESTORE_AUTHORITIES[new_ref] == new_snapshot


def test_replacement_chain_retires_every_predecessor_terminally() -> None:
    r1 = _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    r2 = _trusted_external_reference_replacement_fixture(r1)
    assert isinstance(r2, str)
    r3 = _trusted_external_reference_replacement_fixture(r2)
    assert isinstance(r3, str)
    assert {r1, r2} <= _RETIRED_RESTORE_AUTHORITY_REFERENCES
    assert _resolve_authority(r3) == _RESTORE_AUTHORITIES[r3]
    for retired in (r1, r2):
        _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope()] = retired  # corruption simulation only
        assert _resolve_authority(retired) is None
    _CURRENT_RESTORE_AUTHORITY_BY_SCOPE[_scope()] = r3
    assert _resolve_authority(r3) == _RESTORE_AUTHORITIES[r3]


def test_initial_and_preexisting_fixtures_have_disjoint_test_only_semantics() -> None:
    initial = _initial_protected_scope_membership_fixture()
    assert _RESTORE_AUTHORITIES[initial].lifecycle == "UNINITIALIZED"
    assert _RESTORE_AUTHORITIES[initial].committed_generation is None
    with pytest.raises(ValueError, match="already exists"):
        _preexisting_committed_restore_authority_fixture(1, "a" * 64)
    public = {
        name for name, value in globals().items() if callable(value) and not name.startswith("_")
    }
    assert (
        not {"unretire_reference", "restore_old_reference", "clear_retired", "reuse_reference"}
        & public
    )


def test_corehost_scope_uses_verified_logical_store_identity_not_a_path_hash() -> None:
    binding = startup_recovery_contract()["scope_binding"]
    assert binding["identity_source"] == (
        "verified StateStore metadata for initialized existing stores"
    )
    assert binding["path_hash_required"] is False
    assert binding["opened_store_path_binding"] == (
        "opened SQLiteStateStore.path must equal resolved CoreHostScope.state_store_path "
        "using existing P1A Path.resolve locator normalization; no hash comparison"
    )
    assert binding["protected_scope_source"] == {
        "account_id": "verified StateStore metadata after exact equality with CoreHostScope",
        "device_installation_id": (
            "verified StateStore metadata after exact equality with CoreHostScope"
        ),
        "state_store_identity_fingerprint_sha256": (
            "verified StateStore metadata; never rederived from filesystem path"
        ),
    }
    assert "path" not in " ".join(binding["required_equal_fields"]).lower()


def test_corehost_store_substitution_and_empty_identity_fail_safe() -> None:
    contract = startup_recovery_contract()
    substitution = contract["store_substitution"]
    assert substitution["path_match_sufficient"] is False
    assert substitution["stale_wrong_or_unknown_identity"] == "FAIL_CLOSED"
    assert "exact M0.3 triple" in substitution["security_anchor"]
    empty = contract["empty_or_uninitialized"]
    assert empty["established_state_store_identity_fingerprint"] is False
    assert empty["p1b_generates_state_store_identity_fingerprint"] is False
    assert empty["identity_creation_owner"] == (
        "future authorized StateStore genesis / first-run creation flow"
    )


def test_corehost_recovery_complete_uses_sealed_current_physical_schema_registry() -> None:
    gate = startup_recovery_contract()["recovery_complete"]["physical_schema_gate"]
    assert gate == {
        "actual": "SQLiteStateStore.sqlite_schema_fingerprint()",
        "expected": (
            "persistence_versioning_migrations_backup_and_recovery.json#/"
            "state_store_physical_schema_registry entry for final verified metadata "
            "state_store_schema_version"
        ),
        "required_current_version": (
            "final metadata state_store_schema_version equals sealed "
            "current_state_store_schema_version"
        ),
        "mismatch": "FAIL_CLOSED",
        "match_alone_sufficient": False,
    }
