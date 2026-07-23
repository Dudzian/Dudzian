"""Contract tests for CryptoHunter M0.3 process topology and lifecycle."""

from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs/architecture/cryptohunter_product_architecture/process_topology_and_lifecycle.json"
ARCH_README = ROOT / "docs/architecture/cryptohunter_product_architecture/README.md"
MAIN_README = ROOT / "README.md"
SECRET_RE = re.compile(r"(AKIA[0-9A-Z]{16}|api[_-]?secret\s*[:=]\s*['\"][^'\"]+|password\s*[:=]\s*['\"][^'\"]+|pin\s*[:=]\s*['\"][0-9]{4,}|BEGIN (RSA|EC|OPENSSH) PRIVATE KEY)", re.IGNORECASE)
REQUIRED_SHUTDOWN_FIELDS = {
    "name", "description", "allowed_client_roles", "requires_core_reachable",
    "requires_operator_authentication", "requires_secondary_confirmation",
    "stops_desktop_shell", "stops_tray_agent", "stops_strategies", "stops_core",
    "triggers_kill_switch", "blocks_new_order_intents", "persists_checkpoint",
    "expected_core_result", "invariants",
}


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


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
    assert "## Status M0.3 — under audit" in arch
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
    assert data["process_health_states"] == ["NOT_STARTED", "STARTING", "HEALTHY", "DEGRADED", "STOPPING", "STOPPED", "CRASHED"]


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
    assert process == {"NOT_STARTED", "STARTING", "HEALTHY", "DEGRADED", "STOPPING", "STOPPED", "CRASHED"}
    assert reachability == {"REACHABLE", "UNREACHABLE"}
    assert observation == {"CORE_HANDSHAKE", "PROCESS_LOCK", "CONNECTION_DESCRIPTOR", "TRAY_SUPERVISOR", "CACHED_SNAPSHOT", "NONE"}
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
    assert ["NO_ACTIVE_RUNTIME_CONFIRMED", "ACTIVE_RUNTIME_PRESENT", "RUNTIME_ACTIVITY_UNKNOWN"] in constraints["mutually_exclusive_groups"]
    implications = {item["predicate"]: item for item in constraints["implications"]}
    assert set(implications["TRAY_PROCESS_CONFIRMED_RUNNING"]["requires_tray_process_health_states"]) == {"HEALTHY", "DEGRADED"}
    assert set(implications["SUPERVISED_RESTART_PENDING"]["requires_core_lifecycle_pair_ids"]) == {"crashed_restart_scheduled", "supervised_restart_in_progress"}
    assert set(implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_lifecycle_pair_ids"]) == {"not_started", "stopped"}
    assert implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_state_confidence_states"] == ["CONFIRMED_CURRENT"]
    assert set(implications["NO_ACTIVE_RUNTIME_CONFIRMED"]["requires_core_state_observation_sources"]) == {"PROCESS_LOCK", "CONNECTION_DESCRIPTOR", "TRAY_SUPERVISOR"}
    assert {"not_started", "stopped", "crashed_unscheduled", "crashed_restart_scheduled", "crashed_restart_exhausted"} <= set(implications["ACTIVE_RUNTIME_PRESENT"]["forbidden_core_lifecycle_pair_ids"])
    assert implications["CORE_STATE_OBSERVATION_STALE"]["cannot_alone_confirm"] == ["NO_ACTIVE_RUNTIME_CONFIRMED"]
    axis_rules = {rule["rule_id"]: rule for rule in data["state_axis_consistency_constraints"]["rules"]}
    assert axis_rules["core_ipc_reachable_requires_handshake_current"]["requires_core_state_observation_sources"] == ["CORE_HANDSHAKE"]
    assert axis_rules["core_ipc_reachable_requires_handshake_current"]["requires_core_state_confidence_states"] == ["CONFIRMED_CURRENT"]
    assert "CORE_HANDSHAKE" in axis_rules["core_handshake_requires_reachable_current"]["when"].values()
    assert axis_rules["core_ipc_unreachable_forbids_current_handshake"]["forbidden_core_state_observation_sources"] == ["CORE_HANDSHAKE"]
    assert set(axis_rules["tray_supervisor_requires_running_tray_and_predicate"]["requires_tray_process_health_states"]) == {"HEALTHY", "DEGRADED"}
    assert axis_rules["none_source_requires_unknown"]["requires_core_state_confidence_states"] == ["UNKNOWN"]


def test_requires_core_reachable_matches_reachability_axis() -> None:
    for intent in load_contract()["shutdown_intents"]:
        reachability = set(intent["applicability"]["core_ipc_reachability_states"])
        if intent["requires_core_reachable"] is True:
            assert reachability == {"REACHABLE"}, intent["name"]
            assert not any("unreachable" in condition.lower() and "not" not in condition.lower() for condition in intent["applicability"]["conditions"]), intent["name"]
        if intent["trigger_kind"] == "operating_system_event":
            assert reachability == {"REACHABLE", "UNREACHABLE"}


def test_process_lock_order_is_before_mutable_initialization() -> None:
    data = load_contract()
    steps = {step["step_id"]: step["order"] for step in data["startup_sequence"]}
    assert steps["resolve_device_installation_and_state_store_identity"] < steps["acquire_local_process_lock"]
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
    assert set(policy["options"]) == {"background", "stop_core_with_secondary_confirmation", "cancel"}
    assert policy["tray_agent_required_for_safe_hide"] is True
    assert data["hud_contract"]["read_only"] is True
    assert "full balances" in data["hud_contract"]["locked_hides"]


def test_shutdown_intents_have_complete_machine_contract() -> None:
    data = load_contract()
    expected = {"CLOSE_DESKTOP_SHELL", "HIDE_TO_BACKGROUND", "STOP_STRATEGIES", "PAUSE_STRATEGIES", "STOP_CORE_GRACEFULLY", "EXIT_TRAY_AGENT", "TRIGGER_KILL_SWITCH", "OS_SESSION_LOGOFF", "OS_SHUTDOWN", "ENTER_MAINTENANCE_MODE"}
    intents = intents_by_name(data)
    assert set(intents) == expected
    process_roles = set(roles_by_name(data))
    event_sources = set(data["external_event_sources"])
    for intent in intents.values():
        assert REQUIRED_SHUTDOWN_FIELDS <= intent.keys()
        assert {"trigger_kind", "applicability", "confirmation_policy", "operator_acknowledgement_policy", "event_sources", "handoff_clients", "evaluation_route", "requires_command_authorization_context", "requires_core_command_submission"} <= intent.keys()
        assert intent["trigger_kind"] in {"client_command", "operating_system_event", "supervised_lifecycle_event"}
        assert intent["description"]
        assert isinstance(intent["invariants"], list) and intent["invariants"]
        assert set(intent["allowed_client_roles"]) <= process_roles
        assert set(intent["event_sources"]) <= event_sources
        assert set(intent["applicability"]) == {"core_process_health_states", "core_ipc_reachability_states", "core_supervision_restart_states", "tray_process_health_states", "desktop_window_states", "operator_authentication_states", "conditions", "core_state_observation_sources", "core_state_confidence_states"}
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
            assert {"case_id", "core_lifecycle_pair_ids", "core_ipc_reachability_states", "core_state_observation_sources", "core_state_confidence_states", "tray_process_health_states", "allowed", "requires_operator_acknowledgement", "requires_secondary_confirmation", "required_predicates", "forbidden_predicates", "reason"} <= case.keys()
            assert case["case_id"] not in case_ids
            case_ids.add(case["case_id"])
            assert set(case["core_lifecycle_pair_ids"]) <= pair_ids
            assert set(case["core_ipc_reachability_states"]) <= set(data["core_ipc_reachability_states"])
            assert set(case["tray_process_health_states"]) <= set(data["process_health_states"])
            assert set(case["core_state_observation_sources"]) <= set(data["core_state_observation_sources"])
            assert set(case["core_state_confidence_states"]) <= set(data["core_state_confidence_states"])
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
    assert "silent ordinary close allowed only when Core process health is NOT_STARTED or STOPPED" in close_conditions
    assert "silent ordinary close requires core supervision restart state NONE" in close_conditions
    acknowledgement = close["operator_acknowledgement_policy"]
    assert acknowledgement["mode"] == "conditional"
    assert "Core IPC reachability is UNREACHABLE" in acknowledgement["conditions"]
    assert "Core process health is CRASHED" in acknowledgement["conditions"]
    assert "core supervision restart state is SCHEDULED or IN_PROGRESS" in acknowledgement["conditions"]
    assert "must not display Core stopped only because IPC is UNREACHABLE" in acknowledgement["message_requirements"]
    background = intents["HIDE_TO_BACKGROUND"]
    assert background["stops_desktop_shell"] is True and background["stops_core"] is False
    assert "requires active TrayAgent" in "\n".join(background["invariants"])
    background_cases = {case["case_id"]: case for case in background["state_cases"]}
    assert set(background_cases["active_reachable_core"]["tray_process_health_states"]) == {"HEALTHY", "DEGRADED"}
    assert background["expected_core_result"] == "desktop_hidden_core_lifecycle_unchanged"
    assert "Core remains active" not in background["invariants"]
    assert "HIDE_TO_BACKGROUND does not change Core process health" in background["invariants"]
    assert "HIDE_TO_BACKGROUND does not change supervision restart state" in background["invariants"]
    background_conditions = "\n".join(background["applicability"]["conditions"])
    assert "TrayAgent must actually be running" in background_conditions
    assert "CRASHED is allowed only with a running TrayAgent" in background_conditions
    assert "SCHEDULED or IN_PROGRESS means Tray supervises restart" in background_conditions
    assert "EXHAUSTED is shown as operator-action-required state" in background_conditions
    assert "no silent hide when both Core and Tray are unreachable" in background_conditions
    assert background["operator_acknowledgement_policy"]["mode"] == "conditional"
    assert "Core IPC reachability is UNREACHABLE" in background["operator_acknowledgement_policy"]["conditions"]
    assert "Core process health is CRASHED" in background["operator_acknowledgement_policy"]["conditions"]
    assert "core supervision restart state is SCHEDULED or IN_PROGRESS" in background["operator_acknowledgement_policy"]["conditions"]
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
    assert "secondary confirmation required when core supervision restart state is SCHEDULED or IN_PROGRESS" in exit_confirmation
    assert "secondary confirmation required for active, unknown, unreachable, or restarting Core" in exit_confirmation
    assert "no secondary confirmation only when Core process health is NOT_STARTED or STOPPED" in exit_confirmation
    assert "EXHAUSTED warning says Tray is needed to present operator-action-required state" in exit_confirmation
    assert "cannot be remembered automatically" in "\n".join(tray_exit["invariants"])
    kill = intents["TRIGGER_KILL_SWITCH"]
    assert kill["triggers_kill_switch"] is True and kill["stops_core"] is False
    assert kill["blocks_new_order_intents"] is True and kill["requires_operator_authentication"] is True
    assert kill["requires_secondary_confirmation"] is None
    assert kill["confirmation_policy"]["mode"] == "conditional"
    assert "policy cannot be weakened by UI" in "\n".join(kill["confirmation_policy"]["conditions"])
    logoff = intents["OS_SESSION_LOGOFF"]
    assert "desktop_user_session does not guarantee continued operation" in "\n".join(logoff["invariants"])
    shutdown = intents["OS_SHUTDOWN"]
    assert shutdown["blocks_new_order_intents"] is True and shutdown["persists_checkpoint"] is True
    assert "does not promise completion of all network operations" in "\n".join(shutdown["invariants"])
    maintenance = intents["ENTER_MAINTENANCE_MODE"]
    assert maintenance["blocks_new_order_intents"] is True
    assert "STOPPED" not in maintenance["applicability"]["core_process_health_states"]
    assert maintenance["applicability"]["core_process_health_states"] == ["HEALTHY", "DEGRADED"]
    assert maintenance["applicability"]["core_ipc_reachability_states"] == ["REACHABLE"]
    assert "A stopped Core cannot acknowledge ENTER_MAINTENANCE_MODE" in "\n".join(maintenance["invariants"])
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
    assert "Bootstrapper cannot switch a running Core into maintenance without authorization" in invariants


def test_system_events_are_not_client_roles() -> None:
    data = load_contract()
    assert set(data["external_event_sources"]) == {"operating_system", "windows_session_manager", "future_windows_service_manager"}
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
    assert not re.search(r'(?<!device_)installation_id', without_allowed_legacy)
    assert allowed_legacy_phrase in raw
    assert any("all IPC, discovery and persistence references use the M0.2 device_installation_id name" == item for item in data["invariants"])


def test_startup_readiness_states_and_first_run_gates() -> None:
    data = load_contract()
    states = {state["name"]: state for state in data["startup_readiness_states"]}
    assert set(states) == {"SETUP_REQUIRED", "CREDENTIALS_OPTIONAL", "RECONCILIATION_REQUIRED", "OPERATOR_ACTION_REQUIRED", "READY", "BLOCKED"}
    setup_rules = "\n".join(states["SETUP_REQUIRED"]["rules"])
    assert "no private exchange connections" in setup_rules
    assert "no strategies" in setup_rules
    assert "no order entry" in setup_rules
    assert "no exchange-secret loading" in setup_rules
    credentials = "\n".join(states["CREDENTIALS_OPTIONAL"]["rules"])
    assert "Paper may run without exchange accounts" in credentials
    assert "Testnet requires configured ExchangeAccount and credential reference" in credentials
    assert "strategies are not resumed" in "\n".join(states["RECONCILIATION_REQUIRED"]["rules"])
    assert any("first-run incomplete never starts exchange-private connections or strategies" == item for item in data["invariants"])


def test_autostart_policy() -> None:
    policy = load_contract()["autostart_policy"]
    assert set(policy) == {"start_core_on_logon", "start_tray_on_logon", "open_desktop_shell_on_logon", "show_hud_on_logon", "resume_runtime_after_reconciliation", "invariants"}
    for name in ("start_core_on_logon", "start_tray_on_logon", "open_desktop_shell_on_logon", "show_hud_on_logon", "resume_runtime_after_reconciliation"):
        assert isinstance(policy[name]["default_enabled"], bool), name
        assert isinstance(policy[name]["activation_conditions"], list) and policy[name]["activation_conditions"], name
    assert policy["start_core_on_logon"]["default_enabled"] is True
    assert policy["start_core_on_logon"]["activation_conditions"] == ["wizard_completed"]
    assert policy["start_tray_on_logon"]["default_enabled"] is True
    assert policy["open_desktop_shell_on_logon"]["default_enabled"] is False
    assert policy["show_hud_on_logon"]["default_enabled"] is False
    resume = policy["resume_runtime_after_reconciliation"]
    assert resume["default_enabled"] is False
    assert resume["user_choice_required"] is True
    assert set(resume["activation_conditions"]) == {"reconciliation_passed", "execution_lease_active", "capability_allowed", "policy_allowed", "kill_switch_not_triggered"}
    invariants = "\n".join(policy["invariants"])
    assert "autostart Core does not imply automatic strategy resume" in invariants
    assert "resume requires reconciliation, lease, capability, policy and non-triggered kill switch" in invariants
    assert "HUD requires TrayAgent" in invariants
    assert "open DesktopShell is not required for Core operation" in invariants
    assert "before wizard completion autostart does not start trading" in invariants


def test_deployment_mode_lifecycle_invariants() -> None:
    data = load_contract()
    modes = modes_by_name(data)
    assert "does not promise operation after user logoff" in modes["desktop_user_session"]["non_guarantees"]
    assert any("lock screen" in item for item in modes["desktop_user_session"]["guarantees"])
    assert any("compatible" in item for item in modes["windows_service"]["guarantees"])
    assert any("M0.2 domain identifiers" in item for item in data["invariants"])


def test_ipc_discovery_handshake_reconnect_and_commands() -> None:
    data = load_contract()
    ipc = data["ipc_contract"]
    assert ipc["canonical_logical_interface"] == "versioned Protobuf/gRPC"
    assert ipc["default_transport_scope"] == "local-only"
    required = {"protocol_version", "client_role", "client_version", "client_instance_id", "device_installation_id", "supported_capabilities", "requested_workspace_id", "authentication_authorization_reference", "core_runtime_session_id", "server_capabilities", "compatibility_result"}
    assert set(ipc["handshake_fields"]) == required
    reconnect = "\n".join(ipc["heartbeat_and_reconnect"])
    assert "full snapshot" in reconnect
    assert "command_id" in reconnect
    assert "not replayed" in reconnect
    descriptor_forbidden = set(data["discovery_contract"]["descriptor_must_not_contain"])
    assert {"api_keys", "api_secrets", "pin", "recovery_tokens", "biometric_data", "plaintext_operator_password"} <= descriptor_forbidden
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
    assert any("local process lock blocks second Core" in item for item in data["single_instance_policy"]["core_host"])
    assert any("does not replace ExchangeAccount ExecutionLease" in item for item in data["single_instance_policy"]["core_host"])
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
    assert "implementation must not interpret applicability as a free Cartesian product" in invariants
    assert "lifecycle applicability is determined by exactly one matching state_case" in invariants
    assert "final intent executability requires all global role, authentication, authorization, trigger and Core gates" in invariants
    assert "missing matching state_case means intent is forbidden and fail-closed" in invariants
    assert "more than one matching state_case is a contract error, not a priority mechanism" in invariants
    intents = intents_by_name(data)
    close_cases = {case["case_id"]: case for case in intents["CLOSE_DESKTOP_SHELL"]["state_cases"]}
    assert {"silent_confirmed_inactive", "unreachable_unknown_core", "crashed_unscheduled", "restarting_with_tray_available", "restarting_with_tray_available_ipc_unreachable", "restarting_with_tray_available_ipc_reachable", "restarting_without_tray", "restarting_without_tray_scheduled", "restart_exhausted", "active_reachable_core"} <= set(close_cases)
    assert close_cases["silent_confirmed_inactive"]["core_lifecycle_pair_ids"] == ["not_started", "stopped"]
    assert close_cases["silent_confirmed_inactive"]["required_predicates"] == ["NO_ACTIVE_RUNTIME_CONFIRMED"]
    assert close_cases["unreachable_unknown_core"]["requires_operator_acknowledgement"] is True
    assert "RUNTIME_ACTIVITY_UNKNOWN" in close_cases["unreachable_unknown_core"]["required_predicates"]
    assert close_cases["restarting_with_tray_available"]["core_lifecycle_pair_ids"] == ["crashed_restart_scheduled"]
    assert close_cases["restarting_with_tray_available"]["allowed"] is False
    assert close_cases["restarting_with_tray_available_ipc_unreachable"]["core_lifecycle_pair_ids"] == ["supervised_restart_in_progress"]
    assert close_cases["restarting_with_tray_available_ipc_reachable"]["core_ipc_reachability_states"] == ["REACHABLE"]
    assert close_cases["restarting_without_tray"]["allowed"] is False
    assert "TRAY_PROCESS_CONFIRMED_RUNNING" in close_cases["restarting_without_tray"]["forbidden_predicates"]
    assert close_cases["active_reachable_core"]["allowed"] is False

    hide = intents["HIDE_TO_BACKGROUND"]
    assert "Core and Tray remain active" not in hide["description"]
    assert "Core remains active" not in hide["invariants"]
    hide_cases = {case["case_id"]: case for case in hide["state_cases"]}
    assert {"active_reachable_core", "unreachable_core", "crashed_unscheduled", "crashed_restart_scheduled", "restart_in_progress_ipc_unreachable", "restart_in_progress_ipc_reachable", "restart_exhausted", "tray_unavailable"} <= set(hide_cases)
    assert hide_cases["active_reachable_core"]["core_lifecycle_pair_ids"] == ["ordinary_starting", "healthy", "degraded", "stopping"]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_lifecycle_pair_ids"] == ["supervised_restart_in_progress"]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_ipc_reachability_states"] == ["UNREACHABLE"]
    assert hide_cases["restart_in_progress_ipc_unreachable"]["core_state_observation_sources"] == ["TRAY_SUPERVISOR"]
    assert hide_cases["restart_in_progress_ipc_reachable"]["core_ipc_reachability_states"] == ["REACHABLE"]
    assert hide_cases["restart_in_progress_ipc_reachable"]["core_state_observation_sources"] == ["CORE_HANDSHAKE"]
    assert hide_cases["restart_in_progress_ipc_reachable"]["requires_operator_acknowledgement"] is True
    assert "TRAY_PROCESS_CONFIRMED_RUNNING" in hide_cases["restart_in_progress_ipc_reachable"]["required_predicates"]
    assert hide_cases["restart_exhausted"]["core_lifecycle_pair_ids"] == ["crashed_restart_exhausted"]
    assert "OPERATOR_ACTION_REQUIRED" in hide_cases["restart_exhausted"]["reason"]
    assert hide_cases["tray_unavailable"]["allowed"] is False

    for name in ["STOP_STRATEGIES", "PAUSE_STRATEGIES", "STOP_CORE_GRACEFULLY", "TRIGGER_KILL_SWITCH", "ENTER_MAINTENANCE_MODE"]:
        cases = intents[name]["state_cases"]
        pairs = {pair["pair_id"]: pair for pair in data["valid_core_lifecycle_pairs"]}
        assert cases and all(pairs[pair_id]["core_supervision_restart_state"] == "NONE" for case in cases for pair_id in case["core_lifecycle_pair_ids"]), name
        assert all(case["core_ipc_reachability_states"] == ["REACHABLE"] for case in cases), name
        assert all(case["core_state_observation_sources"] == ["CORE_HANDSHAKE"] for case in cases), name
        assert all(case["core_state_confidence_states"] == ["CONFIRMED_CURRENT"] for case in cases), name

    exit_cases = {case["case_id"]: case for case in intents["EXIT_TRAY_AGENT"]["state_cases"]}
    assert {"inactive_core", "unreachable_core", "active_core", "crashed_unscheduled", "crashed_restart_scheduled", "restart_in_progress_ipc_unreachable", "restart_in_progress_ipc_reachable", "restart_exhausted"} <= set(exit_cases)
    assert exit_cases["inactive_core"]["requires_secondary_confirmation"] is False
    assert exit_cases["inactive_core"]["core_ipc_reachability_states"] == ["UNREACHABLE"]
    assert exit_cases["unreachable_core"]["requires_secondary_confirmation"] is True
    assert "NO_ACTIVE_RUNTIME_CONFIRMED" in exit_cases["unreachable_core"]["forbidden_predicates"]


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
    actual_pairs = {(p["pair_id"], p["core_process_health_state"], p["core_supervision_restart_state"]) for p in data["valid_core_lifecycle_pairs"]}
    assert actual_pairs == expected_pairs
    assert not any(health == "CRASHED" and restart == "IN_PROGRESS" for _, health, restart in actual_pairs)
    restart_states = set(data["supervision_restart_states"])
    process_states = set(data["process_health_states"])
    pair_ids = {p["pair_id"] for p in data["valid_core_lifecycle_pairs"]}
    transitions = data["restart_policy"]["supervision_restart_state_transitions"]
    observed = set()
    for item in transitions:
        assert {"from_restart_state", "required_core_process_health_states", "from_core_lifecycle_pair_ids", "event", "to_restart_state", "resulting_core_process_health_states", "to_core_lifecycle_pair_ids"} <= item.keys()
        assert item["from_restart_state"] in restart_states
        assert item["to_restart_state"] in restart_states
        assert item["from_restart_state"] != "CRASHED"
        assert item["to_restart_state"] != "CRASHED"
        assert set(item["required_core_process_health_states"]) <= process_states
        assert set(item["resulting_core_process_health_states"]) <= process_states
        assert set(item["from_core_lifecycle_pair_ids"]) <= pair_ids
        assert set(item["to_core_lifecycle_pair_ids"]) <= pair_ids
        observed.add((item["from_restart_state"], tuple(item["required_core_process_health_states"]), item["event"], item["to_restart_state"], tuple(item["resulting_core_process_health_states"])))
    assert ("NONE", ("CRASHED",), "crash_detected", "SCHEDULED", ("CRASHED",)) in observed
    assert ("SCHEDULED", ("CRASHED",), "backoff_elapsed_restart_attempt_begins", "IN_PROGRESS", ("STARTING",)) in observed
    assert ("IN_PROGRESS", ("STARTING",), "core_started_successfully", "NONE", ("HEALTHY", "DEGRADED")) in observed
    assert ("IN_PROGRESS", ("STARTING",), "restart_attempt_failed", "SCHEDULED", ("CRASHED",)) in observed
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


def _semantically_valid_combination(
    data: dict,
    pair_id: str,
    ipc_reachability: str,
    observation_source: str,
    confidence: str,
    tray_health: str,
    predicates: set[str],
) -> bool:
    if ipc_reachability == "REACHABLE" and (observation_source != "CORE_HANDSHAKE" or confidence != "CONFIRMED_CURRENT"):
        return False
    if observation_source == "CORE_HANDSHAKE" and (ipc_reachability != "REACHABLE" or confidence != "CONFIRMED_CURRENT"):
        return False
    if ipc_reachability == "UNREACHABLE" and observation_source == "CORE_HANDSHAKE":
        return False
    if observation_source == "TRAY_SUPERVISOR" and (
        tray_health not in {"HEALTHY", "DEGRADED"} or "TRAY_PROCESS_CONFIRMED_RUNNING" not in predicates
    ):
        return False
    if observation_source == "CACHED_SNAPSHOT" and confidence == "CONFIRMED_CURRENT":
        return False
    if observation_source == "NONE" and confidence != "UNKNOWN":
        return False
    if observation_source in {"PROCESS_LOCK", "CONNECTION_DESCRIPTOR"} and ipc_reachability == "REACHABLE":
        return False
    constraints = data["state_predicate_constraints"]
    for group in constraints["mutually_exclusive_groups"]:
        if len(set(group) & predicates) > 1:
            return False
    implications = {item["predicate"]: item for item in constraints["implications"]}
    if "TRAY_PROCESS_CONFIRMED_RUNNING" in predicates and tray_health not in implications["TRAY_PROCESS_CONFIRMED_RUNNING"]["requires_tray_process_health_states"]:
        return False
    if "SUPERVISED_RESTART_PENDING" in predicates and pair_id not in implications["SUPERVISED_RESTART_PENDING"]["requires_core_lifecycle_pair_ids"]:
        return False
    if "NO_ACTIVE_RUNTIME_CONFIRMED" in predicates:
        rule = implications["NO_ACTIVE_RUNTIME_CONFIRMED"]
        if pair_id not in rule["requires_core_lifecycle_pair_ids"]:
            return False
        if confidence not in rule["requires_core_state_confidence_states"]:
            return False
        if observation_source not in rule["requires_core_state_observation_sources"]:
            return False
    if "ACTIVE_RUNTIME_PRESENT" in predicates and pair_id in implications["ACTIVE_RUNTIME_PRESENT"]["forbidden_core_lifecycle_pair_ids"]:
        return False
    for invalid in constraints["invalid_combinations"]:
        pair_matches = "core_lifecycle_pair_ids" not in invalid or pair_id in invalid["core_lifecycle_pair_ids"]
        ipc_matches = "core_ipc_reachability_states" not in invalid or ipc_reachability in invalid["core_ipc_reachability_states"]
        observation_matches = "core_state_observation_sources" not in invalid or observation_source in invalid["core_state_observation_sources"]
        confidence_matches = "core_state_confidence_states" not in invalid or confidence in invalid["core_state_confidence_states"]
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
                                if not _semantically_valid_combination(data, pair["pair_id"], reachability, observation_source, confidence, tray_health, active_predicates):
                                    continue
                                if len(active_predicates) >= 2:
                                    saw_multi_predicate_case = True
                                matches = [
                                    case
                                    for case in intent["state_cases"]
                                    if _case_matches(case, pair["pair_id"], reachability, observation_source, confidence, tray_health, active_predicates)
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
            is_forbidden_catch_all = case["allowed"] is False and (
                not case["required_predicates"] or case["forbidden_predicates"]
            )
            assert (intent["name"], case["case_id"]) in witnessed_cases or is_forbidden_catch_all, (intent["name"], case["case_id"])
    assert saw_multi_predicate_case
    assert not _semantically_valid_combination(data, "stopped", "REACHABLE", "CORE_HANDSHAKE", "CONFIRMED_CURRENT", "HEALTHY", set())
    assert not _semantically_valid_combination(data, "crashed_unscheduled", "REACHABLE", "CORE_HANDSHAKE", "CONFIRMED_CURRENT", "HEALTHY", set())
    close_cases = intents_by_name(data)["CLOSE_DESKTOP_SHELL"]["state_cases"]
    stopped_confirmed = [
        case["case_id"]
        for case in close_cases
        if _case_matches(case, "stopped", "UNREACHABLE", "PROCESS_LOCK", "CONFIRMED_CURRENT", "HEALTHY", {"NO_ACTIVE_RUNTIME_CONFIRMED"})
    ]
    assert stopped_confirmed == ["silent_confirmed_inactive"]
    stopped_stale = [
        case["case_id"]
        for case in close_cases
        if _case_matches(case, "stopped", "UNREACHABLE", "CACHED_SNAPSHOT", "STALE", "HEALTHY", {"RUNTIME_ACTIVITY_UNKNOWN"})
    ]
    assert stopped_stale == ["unreachable_unknown_core"]
    exit_cases = intents_by_name(data)["EXIT_TRAY_AGENT"]["state_cases"]
    stopped_unreachable_with_no_active_runtime = [
        case["case_id"]
        for case in exit_cases
        if _case_matches(case, "stopped", "UNREACHABLE", "PROCESS_LOCK", "CONFIRMED_CURRENT", "HEALTHY", {"NO_ACTIVE_RUNTIME_CONFIRMED"})
    ]
    assert stopped_unreachable_with_no_active_runtime == ["inactive_core"]
    stopped_unreachable_unknown = [
        case["case_id"]
        for case in exit_cases
        if _case_matches(case, "stopped", "UNREACHABLE", "CACHED_SNAPSHOT", "STALE", "HEALTHY", {"RUNTIME_ACTIVITY_UNKNOWN"})
    ]
    assert stopped_unreachable_unknown == ["unreachable_core"]
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
    assert [case["case_id"] for case in unreachable_matches] == ["restart_in_progress_ipc_unreachable"]
    for intent in data["shutdown_intents"]:
        for case in intent["state_cases"]:
            if case["requires_operator_acknowledgement"] or case["allowed"]:
                assert case["required_predicates"] or case["forbidden_predicates"] or (
                    case["core_state_observation_sources"] and case["core_state_confidence_states"]
                ) or intent["trigger_kind"] == "operating_system_event", (intent["name"], case["case_id"])


def _passes_global_pipeline(intent: dict, *, client_role: str, operator_authenticated: bool, authorization_context: bool) -> bool:
    return (
        client_role in intent["allowed_client_roles"]
        and (not intent["requires_operator_authentication"] or operator_authenticated)
        and authorization_context
    )


def test_global_pipeline_blocks_unauthenticated_or_disallowed_clients() -> None:
    data = load_contract()
    stop = intents_by_name(data)["STOP_STRATEGIES"]
    assert _passes_global_pipeline(stop, client_role="desktop_shell", operator_authenticated=True, authorization_context=True)
    assert not _passes_global_pipeline(stop, client_role="desktop_shell", operator_authenticated=False, authorization_context=True)
    assert not _passes_global_pipeline(stop, client_role="bootstrapper", operator_authenticated=True, authorization_context=True)
    assert not _passes_global_pipeline(stop, client_role="desktop_shell", operator_authenticated=True, authorization_context=False)


def test_intent_evaluation_routes() -> None:
    data = load_contract()
    routes = {route["route_id"]: route for route in data["intent_evaluation_routes"]}
    assert set(routes) == {"LOCAL_SHELL_ACTION", "CORE_COMMAND", "OPERATING_SYSTEM_EVENT", "MAINTENANCE_HANDOFF"}
    intents = intents_by_name(data)
    for name in ("CLOSE_DESKTOP_SHELL", "HIDE_TO_BACKGROUND", "EXIT_TRAY_AGENT"):
        intent = intents[name]
        assert intent["evaluation_route"] == "LOCAL_SHELL_ACTION"
        assert intent["requires_core_command_submission"] is False
        assert intent["requires_command_authorization_context"] is False
        assert "core_revalidates_execution_authorization" not in routes["LOCAL_SHELL_ACTION"]["pipeline_steps"]
    for name in ("STOP_STRATEGIES", "PAUSE_STRATEGIES", "STOP_CORE_GRACEFULLY", "TRIGGER_KILL_SWITCH", "ENTER_MAINTENANCE_MODE"):
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
