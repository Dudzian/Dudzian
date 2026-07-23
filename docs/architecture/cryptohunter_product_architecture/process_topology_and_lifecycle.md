# CryptoHunter M0.3 — Process Topology and Lifecycle Contract

Status: `under audit`
Baseline commit: `9c3cb7588c399c533f48317a36bee84c68acba9f`

This Markdown is the human-readable companion to `process_topology_and_lifecycle.json`; the JSON remains the machine-validated source of truth.

## Core lifecycle pair registry
- `not_started` = `NOT_STARTED` + `NONE` — Core process is not started and has no supervised restart pending.
- `ordinary_starting` = `STARTING` + `NONE` — Normal user/session startup is in progress.
- `supervised_restart_in_progress` = `STARTING` + `IN_PROGRESS` — Supervisor is actively starting Core after a crash.
- `healthy` = `HEALTHY` + `NONE` — Core is healthy with no restart pending.
- `degraded` = `DEGRADED` + `NONE` — Core is degraded but running with no restart pending.
- `stopping` = `STOPPING` + `NONE` — Core is stopping under normal lifecycle.
- `stopped` = `STOPPED` + `NONE` — Core is stopped with no restart pending.
- `crashed_unscheduled` = `CRASHED` + `NONE` — Core has crashed and no restart has been scheduled yet.
- `crashed_restart_scheduled` = `CRASHED` + `SCHEDULED` — Core has crashed and restart is scheduled after backoff.
- `crashed_restart_exhausted` = `CRASHED` + `EXHAUSTED` — Core has crashed and restart attempts are exhausted; operator action is required.

The registry is closed for M0.3: `CRASHED + IN_PROGRESS`, `HEALTHY + SCHEDULED`, `HEALTHY + EXHAUSTED`, `STOPPED + IN_PROGRESS` and every pair not listed above are invalid.

## State predicates
- `NO_ACTIVE_RUNTIME_CONFIRMED` — Independent process-lock/descriptor proof confirms no active runtime; cached STOPPED alone is insufficient when reachability is UNREACHABLE.
- `ACTIVE_RUNTIME_PRESENT` — Core runtime is known to be active.
- `RUNTIME_ACTIVITY_UNKNOWN` — Runtime activity is unknown, typically after IPC loss or stale observation.
- `TRAY_PROCESS_CONFIRMED_RUNNING` — TrayAgent process is confirmed running and able to present status.
- `SUPERVISED_RESTART_PENDING` — Supervisor has scheduled or is performing a Core restart.
- `CORE_STATE_OBSERVATION_STALE` — Observed Core state may be stale and cannot be used as current truth.

## Axis consistency constraints
- `core_ipc_reachable_requires_handshake_current`: {'rule_id': 'core_ipc_reachable_requires_handshake_current', 'when': {'core_ipc_reachability_state': 'REACHABLE'}, 'requires_core_state_observation_sources': ['CORE_HANDSHAKE'], 'requires_core_state_confidence_states': ['CONFIRMED_CURRENT'], 'forbidden_core_state_observation_sources': ['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR', 'CACHED_SNAPSHOT', 'NONE']}
- `core_handshake_requires_reachable_current`: {'rule_id': 'core_handshake_requires_reachable_current', 'when': {'core_state_observation_source': 'CORE_HANDSHAKE'}, 'requires_core_ipc_reachability_states': ['REACHABLE'], 'requires_core_state_confidence_states': ['CONFIRMED_CURRENT']}
- `core_ipc_unreachable_forbids_current_handshake`: {'rule_id': 'core_ipc_unreachable_forbids_current_handshake', 'when': {'core_ipc_reachability_state': 'UNREACHABLE'}, 'forbidden_core_state_observation_sources': ['CORE_HANDSHAKE']}
- `tray_supervisor_requires_running_tray_and_predicate`: {'rule_id': 'tray_supervisor_requires_running_tray_and_predicate', 'when': {'core_state_observation_source': 'TRAY_SUPERVISOR'}, 'requires_tray_process_health_states': ['HEALTHY', 'DEGRADED'], 'requires_predicates': ['TRAY_PROCESS_CONFIRMED_RUNNING'], 'forbidden_tray_process_health_states': ['NOT_STARTED', 'STARTING', 'STOPPING', 'STOPPED', 'CRASHED']}
- `cached_snapshot_never_current`: {'rule_id': 'cached_snapshot_never_current', 'when': {'core_state_observation_source': 'CACHED_SNAPSHOT'}, 'allowed_core_state_confidence_states': ['STALE', 'UNKNOWN'], 'forbidden_core_state_confidence_states': ['CONFIRMED_CURRENT']}
- `none_source_requires_unknown`: {'rule_id': 'none_source_requires_unknown', 'when': {'core_state_observation_source': 'NONE'}, 'requires_core_state_confidence_states': ['UNKNOWN'], 'forbidden_core_state_confidence_states': ['STALE', 'CONFIRMED_CURRENT']}
- `process_lock_descriptor_are_unreachable_independent_proofs`: {'rule_id': 'process_lock_descriptor_are_unreachable_independent_proofs', 'when': {'core_state_observation_source': ['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']}, 'requires_core_ipc_reachability_states': ['UNREACHABLE'], 'confirmed_current_requires': ['validated PID', 'validated start_nonce', 'validated process lock or descriptor freshness']}

## Predicate constraints
- mutually exclusive: ['NO_ACTIVE_RUNTIME_CONFIRMED', 'ACTIVE_RUNTIME_PRESENT', 'RUNTIME_ACTIVITY_UNKNOWN']
- implication for `TRAY_PROCESS_CONFIRMED_RUNNING`: {'predicate': 'TRAY_PROCESS_CONFIRMED_RUNNING', 'requires_tray_process_health_states': ['HEALTHY', 'DEGRADED']}
- implication for `SUPERVISED_RESTART_PENDING`: {'predicate': 'SUPERVISED_RESTART_PENDING', 'requires_core_lifecycle_pair_ids': ['crashed_restart_scheduled', 'supervised_restart_in_progress']}
- implication for `NO_ACTIVE_RUNTIME_CONFIRMED`: {'predicate': 'NO_ACTIVE_RUNTIME_CONFIRMED', 'requires_core_lifecycle_pair_ids': ['not_started', 'stopped'], 'requires_core_state_confidence_states': ['CONFIRMED_CURRENT'], 'requires_core_state_observation_sources': ['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR']}
- implication for `ACTIVE_RUNTIME_PRESENT`: {'predicate': 'ACTIVE_RUNTIME_PRESENT', 'forbidden_core_lifecycle_pair_ids': ['not_started', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']}
- implication for `CORE_STATE_OBSERVATION_STALE`: {'predicate': 'CORE_STATE_OBSERVATION_STALE', 'cannot_alone_confirm': ['NO_ACTIVE_RUNTIME_CONFIRMED']}
- invalid combination: {'core_lifecycle_pair_ids': ['not_started', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted'], 'core_ipc_reachability_states': ['REACHABLE'], 'reason': 'Stopped, not-started and crashed Core do not have a confirmed reachable Core IPC endpoint.'}
- invalid combination: {'core_state_observation_sources': ['CACHED_SNAPSHOT'], 'core_state_confidence_states': ['CONFIRMED_CURRENT'], 'reason': 'Cached snapshot cannot provide CONFIRMED_CURRENT after IPC loss.'}

## Intent evaluation routes
### `LOCAL_SHELL_ACTION`
Local DesktopShell/Tray lifecycle action; validates trigger/client role, uses intent-specific auth only, records local audit, and does not submit a Core command unless later contract explicitly adds one.

Pipeline:
- `validate_trigger_kind_and_event_source`
- `validate_client_role`
- `validate_operator_interface_authentication`
- `match_exactly_one_state_case`
- `require_operator_acknowledgement_if_needed`
- `require_secondary_confirmation_if_needed`
- `record_local_audit_event`

### `CORE_COMMAND`
Versioned command to reachable Core; requires role, operator authentication, authorization context, Core IPC, Core revalidation, acknowledgement and audit.

Pipeline:
- `validate_trigger_kind_and_event_source`
- `validate_client_role`
- `validate_operator_interface_authentication`
- `validate_command_authorization_context`
- `validate_required_core_ipc_reachability`
- `match_exactly_one_state_case`
- `require_operator_acknowledgement_if_needed`
- `require_secondary_confirmation_if_needed`
- `submit_versioned_command_to_core`
- `core_revalidates_execution_authorization`
- `accepted_or_rejected_result_is_audited`

### `OPERATING_SYSTEM_EVENT`
Trusted OS event; no client role/operator auth/command auth required; Core notification/checkpoint is best-effort only if Core IPC is available; result is audited.

Pipeline:
- `validate_trigger_kind_and_event_source`
- `match_exactly_one_state_case`
- `best_effort_core_notification_if_ipc_available`
- `accepted_or_rejected_result_is_audited`

### `MAINTENANCE_HANDOFF`
Bootstrapper stage after Core-issued MaintenanceAuthorization; cannot bypass prior operator/Core authorization.

Pipeline:
- `validate_maintenance_authorization_handoff`
- `perform_authorized_update_or_restart_step`
- `accepted_or_rejected_result_is_audited`

## Shared intent evaluation pipeline steps
1. `validate_trigger_kind_and_event_source` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND', 'OPERATING_SYSTEM_EVENT']
2. `validate_client_role` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND']
3. `validate_operator_interface_authentication` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND']
4. `validate_command_authorization_context` applies to ['CORE_COMMAND']
5. `validate_required_core_ipc_reachability` applies to ['CORE_COMMAND']
6. `match_exactly_one_state_case` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND', 'OPERATING_SYSTEM_EVENT']
7. `require_operator_acknowledgement_if_needed` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND']
8. `require_secondary_confirmation_if_needed` applies to ['LOCAL_SHELL_ACTION', 'CORE_COMMAND']
9. `submit_versioned_command_to_core` applies to ['CORE_COMMAND']
10. `core_revalidates_execution_authorization` applies to ['CORE_COMMAND']
11. `accepted_or_rejected_result_is_audited` applies to ['CORE_COMMAND', 'OPERATING_SYSTEM_EVENT', 'MAINTENANCE_HANDOFF']

Rules: state_case determines lifecycle applicability, not full permission to execute; allowed true does not bypass client role validation; allowed true does not bypass operator authentication; allowed true does not bypass authorization context; zero or multiple matching state_cases means reject; Core is the final authority; UI success is shown only after Core acknowledgement

## State axes
- process_health_states: NOT_STARTED, STARTING, HEALTHY, DEGRADED, STOPPING, STOPPED, CRASHED
- core_ipc_reachability_states: REACHABLE, UNREACHABLE
- core_state_observation_sources: CORE_HANDSHAKE, PROCESS_LOCK, CONNECTION_DESCRIPTOR, TRAY_SUPERVISOR, CACHED_SNAPSHOT, NONE
- core_state_confidence_states: CONFIRMED_CURRENT, STALE, UNKNOWN
- desktop_window_states: VISIBLE, HIDDEN, CLOSED
- operator_interface_authentication_states: LOCKED, AUTHENTICATED
- supervision_restart_states: NONE, SCHEDULED, IN_PROGRESS, EXHAUSTED

## Shutdown intents and deterministic state cases
Applicability is a documentation union only. `state_cases` are authoritative for lifecycle applicability; global route gates still validate trigger, role, authentication, authorization and Core acknowledgement. State cases are deterministic, mutually exclusive for a given lifecycle pair/Core IPC reachability/observation/confidence/tray/predicate combination, and fail closed when no case matches. There is no first-match-wins behavior.

### `CLOSE_DESKTOP_SHELL`
Route: `LOCAL_SHELL_ACTION`; requires_command_authorization_context=False; requires_core_command_submission=False.
Close only the DesktopShell window/process. Expected result: `no_core_call_required_or_core_state_unchanged`.

- `silent_confirmed_inactive`: pairs=['not_started', 'stopped']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['NO_ACTIVE_RUNTIME_CONFIRMED']; forbidden_predicates=['SUPERVISED_RESTART_PENDING', 'ACTIVE_RUNTIME_PRESENT', 'RUNTIME_ACTIVITY_UNKNOWN', 'CORE_STATE_OBSERVATION_STALE']; allowed=true; ack=false; secondary=false; reason=Confirmed inactive Core may close silently..

- `unreachable_unknown_core`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['CACHED_SNAPSHOT', 'NONE']; confidence=['STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['RUNTIME_ACTIVITY_UNKNOWN']; forbidden_predicates=['NO_ACTIVE_RUNTIME_CONFIRMED', 'SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=false; reason=Unreachable Core has unknown runtime activity or stale observation; warning acknowledgement required..

- `crashed_unscheduled`: pairs=['crashed_unscheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=Crashed unscheduled Core can be closed only after warning..

- `restarting_with_tray_available`: pairs=['crashed_restart_scheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=false; ack=true; secondary=false; reason=route_to_background_policy: scheduled restart with Tray available must use HIDE_TO_BACKGROUND or controlled shutdown/cancel path..

- `restarting_without_tray_scheduled`: pairs=['crashed_restart_scheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING']; allowed=false; ack=true; secondary=false; reason=forbidden: scheduled restart cannot be closed without a confirmed running TrayAgent; GUI remains visible or opens controlled shutdown/cancel path..

- `restarting_with_tray_available_ipc_unreachable`: pairs=['supervised_restart_in_progress']; ipc=['UNREACHABLE']; observation=['TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=false; ack=true; secondary=false; reason=route_to_background_policy: supervised restart with Tray available must use HIDE_TO_BACKGROUND or controlled shutdown/cancel path..

- `restarting_with_tray_available_ipc_reachable`: pairs=['supervised_restart_in_progress']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=false; ack=true; secondary=false; reason=route_to_background_policy: reachable supervised restart with Tray available must use HIDE_TO_BACKGROUND or controlled shutdown/cancel path..

- `restarting_without_tray`: pairs=['supervised_restart_in_progress']; ipc=['UNREACHABLE']; observation=['CACHED_SNAPSHOT', 'NONE']; confidence=['STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STARTING', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING']; allowed=false; ack=true; secondary=false; reason=forbidden: supervised restart cannot be closed without a confirmed running TrayAgent; GUI remains visible or opens controlled shutdown/cancel path..

- `restart_exhausted`: pairs=['crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=Restart exhausted needs visible operator action; warning required..

- `active_reachable_core`: pairs=['ordinary_starting', 'healthy', 'degraded', 'stopping']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=[]; allowed=false; ack=false; secondary=false; reason=use HIDE_TO_BACKGROUND or controlled shutdown dialog.

### `HIDE_TO_BACKGROUND`
Route: `LOCAL_SHELL_ACTION`; requires_command_authorization_context=False; requires_core_command_submission=False.
Hide or exit DesktopShell while TrayAgent remains active and the current Core lifecycle or supervised-restart lifecycle continues unchanged. Expected result: `desktop_hidden_core_lifecycle_unchanged`.

- `active_reachable_core`: pairs=['ordinary_starting', 'healthy', 'degraded', 'stopping']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=[]; allowed=true; ack=false; secondary=false; reason=Active reachable Core can continue under Tray visibility..

- `unreachable_core`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['CACHED_SNAPSHOT', 'NONE']; confidence=['STALE', 'UNKNOWN']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'RUNTIME_ACTIVITY_UNKNOWN']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=false; reason=Unreachable Core requires acknowledgement; Tray remains responsible for status..

- `crashed_unscheduled`: pairs=['crashed_unscheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=Crashed unscheduled state is shown by Tray; do not claim the crashed Core is running..

- `crashed_restart_scheduled`: pairs=['crashed_restart_scheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=Scheduled restart is supervised by Tray..

- `restart_in_progress_ipc_unreachable`: pairs=['supervised_restart_in_progress']; ipc=['UNREACHABLE']; observation=['TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=STARTING + IN_PROGRESS is supervised restart; Tray remains active and acknowledgement is required. IPC unreachable; Tray supervisor confirms supervised restart..

- `restart_in_progress_ipc_reachable`: pairs=['supervised_restart_in_progress']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING', 'SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=STARTING + IN_PROGRESS is supervised restart; Tray remains active and acknowledgement is required. IPC reachable through Core handshake during supervised restart..

- `restart_exhausted`: pairs=['crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR']; confidence=['CONFIRMED_CURRENT']; tray=['HEALTHY', 'DEGRADED']; required_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=false; reason=OPERATOR_ACTION_REQUIRED must remain visible through Tray/HUD..

- `tray_unavailable`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['REACHABLE', 'UNREACHABLE']; observation=['CORE_HANDSHAKE', 'PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'CACHED_SNAPSHOT', 'NONE']; confidence=['CONFIRMED_CURRENT', 'STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=['TRAY_PROCESS_CONFIRMED_RUNNING']; allowed=false; ack=false; secondary=false; reason=Tray unavailable: no silent hide; remain in GUI or choose another safe path..

### `STOP_STRATEGIES`
Route: `CORE_COMMAND`; requires_command_authorization_context=True; requires_core_command_submission=True.
Request Core to stop strategy execution. Expected result: `strategies_stopped_core_running`.

- `reachable_running_core`: pairs=['healthy', 'degraded']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=true; reason=STOP_STRATEGIES is allowed only for reachable Core with restart state NONE..

### `PAUSE_STRATEGIES`
Route: `CORE_COMMAND`; requires_command_authorization_context=True; requires_core_command_submission=True.
Pause generation of new OrderIntent according to later execution contract. Expected result: `new_order_intents_paused_core_running`.

- `reachable_running_core`: pairs=['healthy', 'degraded']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=false; secondary=false; reason=PAUSE_STRATEGIES is allowed only for reachable Core with restart state NONE..

### `STOP_CORE_GRACEFULLY`
Route: `CORE_COMMAND`; requires_command_authorization_context=True; requires_core_command_submission=True.
Controlled graceful Core shutdown request. Expected result: `core_stopped_after_controlled_shutdown`.

- `reachable_running_core`: pairs=['ordinary_starting', 'healthy', 'degraded']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=true; reason=Graceful Core stop requires reachable Core with restart state NONE..

### `EXIT_TRAY_AGENT`
Route: `LOCAL_SHELL_ACTION`; requires_command_authorization_context=False; requires_core_command_submission=False.
Exit TrayAgent while leaving Core state unchanged. Expected result: `tray_exited_core_unchanged`.

- `inactive_core`: pairs=['not_started', 'stopped']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['NO_ACTIVE_RUNTIME_CONFIRMED']; forbidden_predicates=['RUNTIME_ACTIVITY_UNKNOWN', 'CORE_STATE_OBSERVATION_STALE', 'ACTIVE_RUNTIME_PRESENT', 'SUPERVISED_RESTART_PENDING']; allowed=true; ack=false; secondary=false; reason=Confirmed inactive Core with no active runtime; Tray may exit without secondary confirmation..

- `unreachable_core`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['CACHED_SNAPSHOT', 'NONE']; confidence=['STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['RUNTIME_ACTIVITY_UNKNOWN']; forbidden_predicates=['NO_ACTIVE_RUNTIME_CONFIRMED', 'SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=true; reason=Unreachable Core has unknown activity/stale observation; exiting Tray needs confirmation..

- `active_core`: pairs=['ordinary_starting', 'healthy', 'degraded', 'stopping']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Active Core would continue without Tray notifications..

- `crashed_unscheduled`: pairs=['crashed_unscheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Crashed state still needs Tray visibility..

- `crashed_restart_scheduled`: pairs=['crashed_restart_scheduled']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Scheduled restart pending requires Tray visibility and confirmation..

- `restart_in_progress_ipc_unreachable`: pairs=['supervised_restart_in_progress']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Supervised restart in progress with IPC unreachable requires Tray visibility and confirmation..

- `restart_in_progress_ipc_reachable`: pairs=['supervised_restart_in_progress']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['SUPERVISED_RESTART_PENDING']; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Supervised restart in progress with IPC reachable still requires confirmation before exiting Tray..

- `restart_exhausted`: pairs=['crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=true; secondary=true; reason=Exiting Tray loses visible OPERATOR_ACTION_REQUIRED state..

### `TRIGGER_KILL_SWITCH`
Route: `CORE_COMMAND`; requires_command_authorization_context=True; requires_core_command_submission=True.
Trigger kill switch to block execution while Core remains available for read/reconciliation. Expected result: `kill_switch_triggered_core_running`.

- `reachable_running_core`: pairs=['ordinary_starting', 'healthy', 'degraded', 'stopping']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=none; reason=Kill switch trigger is allowed only for reachable Core with restart state NONE in M0.3..

### `OS_SESSION_LOGOFF`
Route: `OPERATING_SYSTEM_EVENT`; requires_command_authorization_context=False; requires_core_command_submission=False.
Handle Windows user logoff in desktop_user_session. Expected result: `graceful_session_shutdown_best_effort`.

- `os_best_effort_ipc_reachable`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=false; secondary=false; reason=Operating-system lifecycle event is handled best-effort for any canonical lifecycle pair..

- `os_best_effort_ipc_unreachable`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR', 'CACHED_SNAPSHOT', 'NONE']; confidence=['CONFIRMED_CURRENT', 'STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=false; secondary=false; reason=Operating-system lifecycle event is handled best-effort for any canonical lifecycle pair..

### `OS_SHUTDOWN`
Route: `OPERATING_SYSTEM_EVENT`; requires_command_authorization_context=False; requires_core_command_submission=False.
Handle Windows shutdown/restart. Expected result: `checkpoint_best_effort_shutdown`.

- `os_best_effort_ipc_reachable`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=false; secondary=false; reason=Operating-system lifecycle event is handled best-effort for any canonical lifecycle pair..

- `os_best_effort_ipc_unreachable`: pairs=['not_started', 'ordinary_starting', 'supervised_restart_in_progress', 'healthy', 'degraded', 'stopping', 'stopped', 'crashed_unscheduled', 'crashed_restart_scheduled', 'crashed_restart_exhausted']; ipc=['UNREACHABLE']; observation=['PROCESS_LOCK', 'CONNECTION_DESCRIPTOR', 'TRAY_SUPERVISOR', 'CACHED_SNAPSHOT', 'NONE']; confidence=['CONFIRMED_CURRENT', 'STALE', 'UNKNOWN']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=[]; forbidden_predicates=[]; allowed=true; ack=false; secondary=false; reason=Operating-system lifecycle event is handled best-effort for any canonical lifecycle pair..

### `ENTER_MAINTENANCE_MODE`
Route: `CORE_COMMAND`; requires_command_authorization_context=True; requires_core_command_submission=True.
Enter maintenance/update/backup preparation mode. Expected result: `maintenance_mode_entered`.

- `reachable_running_core`: pairs=['healthy', 'degraded']; ipc=['REACHABLE']; observation=['CORE_HANDSHAKE']; confidence=['CONFIRMED_CURRENT']; tray=['NOT_STARTED', 'STARTING', 'HEALTHY', 'DEGRADED', 'STOPPING', 'STOPPED', 'CRASHED']; required_predicates=['ACTIVE_RUNTIME_PRESENT']; forbidden_predicates=['SUPERVISED_RESTART_PENDING']; allowed=true; ack=true; secondary=true; reason=Maintenance entry requires reachable running Core with restart state NONE and issued handoff..

## Restart policy transitions
- `crash_detected`: ['crashed_unscheduled'] / `NONE` + ['CRASHED'] -> ['crashed_restart_scheduled'] / `SCHEDULED` + ['CRASHED']
- `backoff_elapsed_restart_attempt_begins`: ['crashed_restart_scheduled'] / `SCHEDULED` + ['CRASHED'] -> ['supervised_restart_in_progress'] / `IN_PROGRESS` + ['STARTING']
- `core_started_successfully`: ['supervised_restart_in_progress'] / `IN_PROGRESS` + ['STARTING'] -> ['healthy', 'degraded'] / `NONE` + ['HEALTHY', 'DEGRADED']
- `crash_loop_limit_exceeded`: ['crashed_restart_scheduled'] / `SCHEDULED` + ['CRASHED'] -> ['crashed_restart_exhausted'] / `EXHAUSTED` + ['CRASHED']
- `crash_loop_limit_exceeded`: ['supervised_restart_in_progress'] / `IN_PROGRESS` + ['STARTING'] -> ['crashed_restart_exhausted'] / `EXHAUSTED` + ['CRASHED']
- `restart_attempt_failed`: ['supervised_restart_in_progress'] / `IN_PROGRESS` + ['STARTING'] -> ['crashed_restart_scheduled'] / `SCHEDULED` + ['CRASHED']

## Invariants
- CLOSE_DESKTOP_SHELL does not stop Core
- HIDE_TO_BACKGROUND does not stop Core
- STOP_STRATEGIES does not stop Core
- STOP_CORE_GRACEFULLY is not a kill switch
- TRIGGER_KILL_SWITCH leaves Core active for read/reconciliation
- EXIT_TRAY_AGENT does not stop Core without separate explicit choice
- OS_SHUTDOWN checkpoints and controlled-stops within available time
- User logoff in desktop_user_session may stop user processes; no post-logout promise
- Maintenance mode blocks new trading commands before update
- Windows lock screen does not stop Core in desktop_user_session
- Future migration to windows_service must not require changing M0.2 domain identifiers or command/read-model contracts
- Service mode keeps compatible IPC
- No silent environment fallback from Live/Testnet to Paper
- process lock precedes mutable state-store open
- process lock precedes RuntimeSession creation
- losing process-lock contender performs no mutable initialization
- stale lock recovery requires PID/start nonce verification and must not blindly take over the state store
- first-run incomplete never starts exchange-private connections or strategies
- M0.3 does not create an installation_id alias
- all IPC, discovery and persistence references use the M0.2 device_installation_id name
- only CoreHost applies mutable trading state mutations
- DesktopShell and TrayAgent may only send commands requesting mutations
- a correctly authenticated command may still be rejected by Core
- Bootstrapper does not send runtime or trading commands
- UNREACHABLE never implies STOPPED
- A stopped Core cannot acknowledge ENTER_MAINTENANCE_MODE
- EXHAUSTED does not start Core without operator action
- GUI crash does not change Core supervision restart state
- Tray restart does not change Core supervision restart state
- implementation must not interpret applicability as a free Cartesian product
- missing matching state_case means intent is forbidden and fail-closed
- for one combination of core lifecycle pair, reachability, Tray health, and required predicates at most one state_case may match per intent
- more than one matching state_case is a contract error, not a priority mechanism
- state_cases do not use first match wins
- CRASHED + SCHEDULED is not equivalent to durable stopped
- STARTING + IN_PROGRESS is supervised restart and is not equivalent to ordinary startup or durable stopped
- lifecycle applicability is determined by exactly one matching state_case, while final intent executability requires all global role, authentication, authorization, trigger and Core gates
- Core IPC REACHABLE requires CORE_HANDSHAKE and CONFIRMED_CURRENT
- CORE_HANDSHAKE with Core IPC UNREACHABLE is invalid
- Core-mutating commands use only REACHABLE CORE_HANDSHAKE CONFIRMED_CURRENT state cases
- ordinary CLOSE_DESKTOP_SHELL cannot close a pending restart without a confirmed running TrayAgent

## M0.3 audit fix: restart provenance, independent proof, and state-case validation

JSON is the machine source of truth for this section.  The M0.3 status remains **under audit**.

### Bidirectional restart predicate contract

`SUPERVISED_RESTART_PENDING` is bidirectional: `crashed_restart_scheduled` and `supervised_restart_in_progress` require it, while non-restart lifecycle pairs forbid it.  `crashed_restart_exhausted` is not a pending restart.  Implementations validate lifecycle-pair predicate requirements before matching any `state_case`.

Generic stale/unknown cases no longer contain `crashed_restart_scheduled` or `supervised_restart_in_progress`; restart lifecycle pairs are handled only by explicit restart cases.  Pending restart with a confirmed running Tray is not silently closed: close is forbidden and must route to hide-to-background or controlled shutdown/cancel.  Pending restart without a running Tray is also forbidden; the GUI remains visible or presents controlled shutdown/cancel.

### Independent observation proof

`INDEPENDENT_OBSERVATION_PROOF_VALIDATED` means all independent observation checks succeeded: `PID_VALIDATED`, `START_NONCE_VALIDATED`, `SOURCE_FRESHNESS_VALIDATED`, `DEVICE_INSTALLATION_IDENTITY_MATCHED`, and `STATE_STORE_IDENTITY_MATCHED`.  The only observation sources allowed for this proof are `PROCESS_LOCK` and `CONNECTION_DESCRIPTOR`; it is forbidden for `CACHED_SNAPSHOT`, `NONE`, `CORE_HANDSHAKE`, and `TRAY_SUPERVISOR`.

`PROCESS_LOCK + CONFIRMED_CURRENT` and `CONNECTION_DESCRIPTOR + CONFIRMED_CURRENT` require `INDEPENDENT_OBSERVATION_PROOF_VALIDATED`.  `CORE_HANDSHAKE` uses IPC confirmation, and `TRAY_SUPERVISOR` requires `TRAY_PROCESS_CONFIRMED_RUNNING` with a healthy or degraded Tray.

### Stale observation and fail-closed matching

`CORE_STATE_OBSERVATION_STALE` requires `STALE` confidence and cannot coexist with `CONFIRMED_CURRENT`.  `CACHED_SNAPSHOT + STALE` requires the stale predicate; `NONE` requires `UNKNOWN` and cannot use stale observation as current proof.  Stale observation cannot authorize silent close or mutating Core commands; it may only produce warning, acknowledgement, forbidden, or fail-closed paths.

### Generic constraint interpreter and `case_kind`

The test contract interprets `state_axis_consistency_constraints` generically instead of reimplementing rule-specific semantics.  It also executes lifecycle-pair predicate requirements and the independent observation proof contract before matching state cases.

Every state case has exactly one `case_kind`: `normal`, `forbidden`, or `forbidden_catch_all`.  `normal` means `allowed=true`; `forbidden` and `forbidden_catch_all` mean `allowed=false`.  Normal and forbidden cases must have at least one semantically valid witness.  Missing matches fail closed, multiple matches are contract errors, and there is no first-match-wins behavior.

### State-case variant coverage and OS best-effort refinement

Normal and forbidden `state_case` entries must not contain dead variants.  Each declared lifecycle pair, IPC reachability state, observation source, confidence state, Tray health state, and every declared axis product must have at least one semantically valid predicate witness; otherwise the variant must be moved to a separate case or removed.  `forbidden_catch_all` remains the only broad fallback kind and is validated separately.

Generic unreachable handling is split by provenance: `CACHED_SNAPSHOT + STALE` uses `unreachable_stale_snapshot` and requires both `RUNTIME_ACTIVITY_UNKNOWN` and `CORE_STATE_OBSERVATION_STALE`; `NONE + UNKNOWN` uses `unreachable_no_observation`, requires only `RUNTIME_ACTIVITY_UNKNOWN`, and forbids both `CORE_STATE_OBSERVATION_STALE` and `INDEPENDENT_OBSERVATION_PROOF_VALIDATED`.  Generic unknown cases still exclude restart lifecycle pairs.

Independent current proof and Tray-supervisor current observation are disjoint.  Cases that use `PROCESS_LOCK` or `CONNECTION_DESCRIPTOR` with `CONFIRMED_CURRENT` require `INDEPENDENT_OBSERVATION_PROOF_VALIDATED`; cases that use `TRAY_SUPERVISOR` with `CONFIRMED_CURRENT` require `TRAY_PROCESS_CONFIRMED_RUNNING` and forbid independent proof.

`HIDE_TO_BACKGROUND` no longer uses a broad `tray_unavailable` catch-all.  It has separate forbidden paths for reachable Core, independent current observation, cached stale observation, and no observation, so each no-Tray variant has its own valid witness and no restart-specific overlap.

`OS_SESSION_LOGOFF` and `OS_SHUTDOWN` now expose five best-effort paths: reachable IPC handshake, independent current proof, Tray supervisor current observation, cached stale observation, and no observation.  Trusted OS events do not require operator authentication, client role, or command authorization context merely because Core IPC or an observation source is missing.

The generic constraint interpreter validates every `when` reference against the corresponding JSON axis, enforces unique non-empty `rule_id` values, and validates the independent observation proof contract keys, required checks, source disjointness, and full observation-source coverage.  JSON remains the machine source of truth.


### Closed observation-axis coverage and Tray-supervisor current routes

The observation axis is closed by machine rules: `CORE_HANDSHAKE` is only `REACHABLE + CONFIRMED_CURRENT`; `PROCESS_LOCK` and `CONNECTION_DESCRIPTOR` are only `UNREACHABLE + CONFIRMED_CURRENT` with `INDEPENDENT_OBSERVATION_PROOF_VALIDATED`; `TRAY_SUPERVISOR` is only `UNREACHABLE + CONFIRMED_CURRENT` with a healthy/degraded Tray and `TRAY_PROCESS_CONFIRMED_RUNNING`; `CACHED_SNAPSHOT` is only `UNREACHABLE + STALE` with `CORE_STATE_OBSERVATION_STALE`; and `NONE` is only `UNREACHABLE + UNKNOWN` without stale or independent proof predicates.

`CLOSE_DESKTOP_SHELL` has explicit Tray-supervisor current paths for `crashed_unscheduled` and `crashed_restart_exhausted`.  These paths require a running Tray, forbid independent proof, allow close only with acknowledgement, and do not claim that a crashed Core is running.

`EXIT_TRAY_AGENT` has explicit Tray-supervisor current paths for `crashed_unscheduled`, `crashed_restart_scheduled`, `supervised_restart_in_progress`, and `crashed_restart_exhausted`.  Each path requires acknowledgement and secondary confirmation, states that Core is not stopped by exiting Tray, warns that Core may restart without Tray icon, HUD, or notifications, and says the user can cancel.

`OS_SESSION_LOGOFF` and `OS_SHUTDOWN` must match every semantically valid context through exactly one of the five best-effort paths.  An unmatched OS context or multiple matches are contract errors; there is no first-match-wins fallback.  JSON remains the machine source of truth.

### Applicability aggregation and EXIT_TRAY_AGENT confirmation policy

Intent `applicability` is an aggregate of axis values used by its `state_cases`; it is not a Cartesian product of allowed runtime combinations.  Every lifecycle health/restart pair, IPC reachability value, observation source, confidence value, and Tray health value declared by a state case must be present in the owning intent's applicability aggregate, while exact matching remains defined only by the state cases.

`EXIT_TRAY_AGENT` includes `TRAY_SUPERVISOR` in its applicability observation-source aggregate because its explicit Tray-supervisor current cases use that source.  `inactive_core` is the only `EXIT_TRAY_AGENT` state case that may skip secondary confirmation: it is limited to `not_started`/`stopped`, `UNREACHABLE`, `NO_ACTIVE_RUNTIME_CONFIRMED`, `INDEPENDENT_OBSERVATION_PROOF_VALIDATED`, and forbids pending restart, active runtime, and unknown runtime predicates.

All other `EXIT_TRAY_AGENT` normal cases that involve active Core, unknown state, stale observation, crash, scheduled restart, restart in progress, restart exhausted, reachable Core, or Tray-supervisor current crash/restart observation require secondary confirmation.  The confirmation policy therefore says that all `UNREACHABLE` cases except machine-confirmed `inactive_core` require secondary confirmation; JSON remains the machine source of truth.
