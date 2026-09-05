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

## First-run bootstrap authority

M0.3 zamyka implementacyjnie neutralną **external product provisioning boundary** jako pre-existing trust anchor dla pierwszego uruchomienia. Jest to ephemeral protected provisioning handoff, a nie nowa trwała encja M0.2. Przyszła implementacja może użyć chronionego handoffu instalatora, package provisioning, SaaS/device provisioning albo innego mechanizmu produktowego; M0.3 nie wybiera Windows API, TPM, Secure Enclave, kryptografii instalatora ani SaaS enrollment.

Bootstrapper pełni wyłącznie rolę transport/discovery: może przenieść opaque protected reference, lecz nie może jej mintować, akceptować ani użyć do elevation. DesktopShell i TrayAgent nie są authority. CoreHost jest konsumentem i walidatorem, ale nie może sam wydać provisioning membership. Hash dowodzi integralności claimu, nie jego membership; raw caller, caller boolean, self-hash, zwykły RuntimeSession ani arbitrary local-admin claim nie mogą ustanowić authority.

Provisioning handoff wiąże exact M0.2 `account_id`, `device_installation_id` i intended first `operator_id`, generation/revision, canonical UTC validity window, challenge fingerprint oraz provisioning-context fingerprint. Fingerprint kompletnego immutable content jest ephemeral key. Core wymaga exact immutable membership w pre-existing registry widocznym z external product provisioning boundary. Nie istnieje durable bootstrap authorization ID.

Provisioned `DeviceInstallation` identity umożliwia bezpieczne rozstrzygnięcie canonical state-store identity i process lock. Nie oznacza stanu M0.10 `TRUSTED`, authenticated operatora, verified PIN ani LIVE authorization. RuntimeSession może istnieć w ograniczonym control-plane podczas `SETUP_REQUIRED`.

Bootstrap eligibility jest wyprowadzane wyłącznie z accepted Core/product state: `SETUP_REQUIRED`, exact account/device/operator/generation/revision, brak accepted first OperatorIdentity, brak completed initial-security setup i brak consumed generation. Valid claim autoryzuje dokładnie jeden późniejszy M0.10 initial-security establishment transition; nie ustawia `READY`. Consumption fence obejmuje generation, revision, claim fingerprint i challenge fingerprint. Replay, zmieniony claim z nowym self-hashem, completed setup, second-device enrollment oraz reinstall/recovery bypass są zabronione.

Bootstrap authority nie autoryzuje normalnych privileged operations, RiskPolicy, kill switch, ProductCapabilities, exchange operations, ExecutionLease ani LIVE. Missing/invalid authority pozostawia `SETUP_REQUIRED` i zabrania private exchange connection, exchange-secret loading, strategy start oraz order entry.

### Startup ordering

1. Pre-existing provisioning handoff dostarcza bezpieczną canonical account/device/state-store identity bez przyznawania trust.
2. Core przejmuje process lock; przegrany contender nie otwiera mutable store i nie tworzy RuntimeSession.
3. Po locku Core tworzy RuntimeSession, otwiera store i sprawdza integrity.
4. Core wyprowadza readiness; dopiero dla `SETUP_REQUIRED` waliduje external provisioning membership.
5. Valid bootstrap udostępnia wyłącznie późniejszą bramkę M0.10, nadal nie `READY`.
6. Secret references, connections i trading pozostają za setup/security/readiness gates.

Bootstrap acceptance, rejection i późniejsze completion używają canonical M0.2 `AuditEvent`; po utworzeniu RuntimeSession audit może go referencjonować. Gdy M0.7 jest dostępne, zachowuje causation/correlation. Safe payload może zawierać canonical IDs, reference fingerprint, generation i reason code, ale nigdy raw bootstrap secret, PIN, biometric material, API credentials ani protected provisioning payload.

M0.10 pozostaje właścicielem first OperatorIdentity acceptance, initial device `TRUSTED` designation, PIN verifier, optional platform biometric semantics i pierwszego Core-issued AuthenticationProof. M0.11 pozostaje właścicielem durability, migrations, backup i crash recovery accepted/consumed facts.

### Exact executable authority closure

Public bootstrap validation consumes only an untrusted claim, an opaque reference to pre-existing current Core state, canonical current time and the single closed purpose `INITIAL_SECURITY_ESTABLISHMENT_ONLY`. It never accepts a caller-owned registry, membership binding or Core-state projection. A nominal owner string is insufficient. `ProvisioningMembershipBinding` exact-binds the claim fingerprint, complete immutable claim-content fingerprint, external authority source and provisioning-context fingerprint; the accepted claim content is independently held by the Core-visible provisioning boundary.

`CoreCurrentBootstrapState` is also pre-existing Core authority: an opaque reference must resolve in the accepted registry and be the exact current designation for its account/device scope. A caller-created `SETUP_REQUIRED` record, a recomputed state hash, cleared consumption, or accepted-but-stale history has no authority.

The pure transition compares the current PRE state and membership, then derives one POST state containing an exact `ConsumedBootstrapAuthority`. The consumed fence includes account, device, generation, revision, claim fingerprint and challenge fingerprint. The returned `BootstrapTransitionResult` is usable only for `INITIAL_SECURITY_ESTABLISHMENT_ONLY`; it cannot be consumed as normal privileged, policy, lease or LIVE authority. M0.3 specifies semantic compare-and-consume; M0.11 must later make accepted/current designation and consumption durably atomic and crash-safe.

Startup identity resolution has two derived modes. For an existing installation, canonical account/device/state-store identity resolves from accepted local/Core state under the later persistence boundary and does not require the first-run handoff again. Only when accepted installation identity is absent may the external provisioning handoff supply canonical identity for first-installation process-lock scope. Neither mode grants M0.10 device trust or authentication authority.

### Transition-result revalidation and first-operator absence

`BootstrapTransitionResult` is an untrusted transport object: its nominal type, success strings and content alone are never authority. The initial-security consumer re-resolves an exact accepted historical PRE state and an exact accepted/current POST state, requires registry keys to equal internal and recomputed state fingerprints, proves a single-field-set-preserving PRE→POST transition with exactly one appended consumed binding, and verifies that binding against the current POST history.

Every consumed-history entry is fully validated for canonical account/device IDs, positive non-boolean generation/revision, lowercase SHA-256 fingerprints, exact state scope, tuple ordering, exact uniqueness and generation uniqueness. `CoreCurrentBootstrapState.first_operator_presence` is closed to `ABSENT|PRESENT`: `PRE_INITIAL_SECURITY` requires `ABSENT`, while `INITIAL_SECURITY_COMPLETED` requires `PRESENT`. A future M0.10 first-operator acceptance must replace the current PRE/ABSENT designation; historical PRE remains audit-only.

## Protected external restore-freshness authority

M0.3 owns the implementation-neutral `/restore_freshness_authority_contract`. Its owner is `EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY`; CoreHost is only a consumer through a Core-owned trusted protocol path. Protected scope membership and its state live outside the restorable canonical StateStore, every `BackupEnvelope`, normal user-editable configuration, DesktopShell and TrayAgent state. Exact scope is canonical `account_id`, canonical `device_installation_id`, and `state_store_identity_fingerprint_sha256` read from verified M0.11 StateStore metadata, never derived from a filesystem path and never a new M0.2 entity. M0.11 owns the immutable logical StateStore identity semantics; M0.3 consumes the exact triple without minting or re-keying it.

The contract does not select TPM, Secure Enclave, Windows API, macOS Keychain, Linux secret service, SaaS, a hardware counter, keyring, or cryptographic primitive. A future conforming implementation may use any mechanism satisfying the protected external authority contract; no mechanism is a canonical requirement or domain authority. If the platform cannot provide a conforming mechanism, authoritative restore fails closed.

### Exact protected current designation and canonical scope

The protected current-designation key is exact `(account_id, device_installation_id, state_store_identity_fingerprint_sha256)`, with exactly one current membership per scope. The `EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY` alone owns designation. The accepted protected registry may retain historical records, but an authority-critical resolver requires an existing opaque reference, a structurally and semantically valid record, canonical scope, and equality with the boundary-owned current reference for that record scope. Accepted membership, a valid fingerprint, or historical reference is not current authority. Caller, backup, CoreHost and M0.11 cannot select current membership.

Initial external scope provisioning, normal freshness advance, and protected reference replacement are distinct operations. Initial provisioning applies only when no current membership exists and establishes exact-scope `UNINITIALIZED`, never a magic committed generation. Normal advance remains exclusively `COMMITTED G -> PREPARED G+1 -> accepted current durable local G+1 -> FINALIZE -> COMMITTED G+1`.

Protected reference replacement changes opaque identity only. It is allowed only for stable `UNINITIALIZED` or `COMMITTED` current records and copies exact scope, lifecycle, committed generation/state fingerprint, and every prepared field unchanged; only membership-local revision may advance. It cannot rollback, skip, change same-generation content, clear pending, return COMMITTED to UNINITIALIZED, or replace PREPARED. PREPARED must first be legally FINALIZED or normally ABORTED where allowed. The old reference becomes terminal `RETIRED` and remains accepted history but never authority. Retirement and current designation are one semantically atomic external-boundary transition: exact current non-retired stable OLD becomes terminally retired while NEW becomes accepted and sole current with the same freshness projection. The resolver rejects RETIRED references even if corruption points the current map back to them. Missing current records, retired-current mappings, or malformed replacement observations fail closed without Core/M0.11 auto-heal. There is no unretire transition and no additional anchor-for-anchor layer. The scope-to-current-reference mapping itself is protected state inside `EXTERNAL_PRODUCT_PROTECTED_STATE_BOUNDARY`, outside StateStore, backup, user config, Core registry authority and M0.11 mutable state; designation rollback is corruption and fails closed, never a legal transition.

Canonical scope validation projects the read-only M0.2 identifier contract exactly: `account_id` is lowercase `acct_<uuidv7>`, `device_installation_id` is lowercase `dev_<uuidv7>`, and the suffix matches `[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}`. The state-store identity fingerprint is exact lowercase SHA-256. Prefix-only validation is insufficient, and local durable evidence uses the same canonical scope validation before exact matching.

### Scope membership, genesis and protected projection

Protected exact-scope membership has the closed lifecycle `UNINITIALIZED`, `PREPARED`, `COMMITTED`. `MISSING` is not a lifecycle state: it means no protected membership, no READY or self-provisioning, and requires trusted external provisioning/recovery. `UNINITIALIZED` means pre-existing external membership exists but committed generation and fingerprint are absent, no StateStore authority exists, and READY is denied. CoreHost, M0.11, Bootstrapper, UI, Tray, raw caller, backup, and first-run bootstrap cannot mint membership.

Legal genesis is `UNINITIALIZED -> PREPARED generation 1 -> accepted current DURABLE_COMMITTED local evidence -> COMMITTED generation 1`. GENESIS PREPARE requires exact membership and scope, Core-owned trusted path, absent committed state, exact next generation 1, and canonical candidate state/transaction fingerprints. There is no generation-0 authority and no magic pre-seeded COMMITTED generation 1. Once historically committed, an anchor cannot return to UNINITIALIZED or repeat genesis. Existing installations retain normal `COMMITTED G -> PREPARED G+1 -> durable local G+1 -> COMMITTED G+1` behavior, with no rollback or skip.

Protected projection contains scope, lifecycle, optional committed generation/state fingerprint, optional prepared generation/state/transaction fingerprints, authority revision/source, and content fingerprint. Present generations are positive non-boolean integers. Every authoritative StateStore semantic transaction after genesis participates. Fingerprints/signatures prove integrity only, never protected membership.

### Trusted local durable evidence, FINALIZE and ABORT

M0.3 models an implementation-neutral handoff owned by the M0.11 local durable StateStore observation boundary. Its private evidence record exact-binds scope, generation, state fingerprint, transaction fingerprint, `DURABLE_COMMITTED`, evidence revision and evidence fingerprint. It is neither M0.2 entity nor external authority and cannot mint membership. Protocol consumers receive only an opaque reference resolving to exact accepted **and current** membership in a private Core/M0.11-owned durable-observation registry. Caller-created or self-hashed objects and booleans such as `durable`, `fsync_ok`, `committed`, `is_current`, or `trusted` have no authority.

PREPARE creates protected pending membership but never current StateStore authority. FINALIZE accepts only an opaque evidence reference resolving exact accepted current `DURABLE_COMMITTED` evidence whose scope, generation, state fingerprint and transaction fingerprint equal existing protected PREPARED membership. Matching caller strings alone are denied. Success commits prepared generation/fingerprint, terminally clears pending, increments revision and recomputes content fingerprint; older generations never regain current authority.

Executable protected ABORT applies only to PREPARED over already COMMITTED G when accepted current durable local observation proves the store is exact committed G and does not prove durable candidate G+1. It keeps committed generation/fingerprint, clears all prepared fields, increments revision, and recomputes content fingerprint. Raw values, caller evidence, wrong scope/current state, no pending, or durable candidate G+1 are denied. After legal ABORT a different G+1 candidate may be prepared.

### Exact projection and genesis crash fencing

Field presence is closed. `UNINITIALIZED` has no committed or prepared generation/fingerprint fields. Genesis `PREPARED` has no committed fields, has prepared generation exactly 1, and requires both canonical prepared fingerprints. Normal `PREPARED` has positive non-boolean committed G with its fingerprint plus prepared G+1 with both fingerprints. `COMMITTED` has positive committed generation/fingerprint and no prepared fields. Any dangling optional field, unknown lifecycle, or missing paired field is semantic class `MALFORMED_PROTECTED_AUTHORITY` and fails closed through existing controlled M0.3 denial/no-READY outcomes even when the content fingerprint is correctly recomputed. M0.11 later maps that class to its final persistence failure registry; M0.3 defines no such taxonomy.

Genesis crash recovery is `RESUME_EXACT_PENDING_FAIL_CLOSED`. After `PREPARED 1` with no accepted/current local evidence, outcome is `GENESIS_PENDING_RECOVERY_REQUIRED`: no READY, no automatic return to UNINITIALIZED, no ordinary Core/M0.11 ABORT, and no replacement candidate. Absence of local evidence cannot prove that a durable generation 1 never existed. Exact accepted/current `DURABLE_COMMITTED` evidence matching pending may resume and FINALIZE. Mismatching evidence fails closed, retains pending, and requires trusted external recovery; any future clearing is a separate protected external recovery authority.

An idempotent PREPARE retry never bypasses expected-current validation. Genesis exact retry requires expected generation/fingerprint both absent. Normal exact retry requires expected generation/fingerprint equal protected committed current. Wrong expected current is denied before idempotent success; a different candidate is incompatible and leaves pending unchanged.

### Crash, rollback and ownership

Crash before PREPARE leaves COMMITTED G. Crash after PREPARE but before local commit is recovered by exact durable observation of G followed by protected ABORT. Crash after local commit but before FINALIZE is recovered only through exact accepted current evidence matching pending G+1. External committed ahead/store behind, store ahead without pending, mismatch, excessive gap, or unavailable/malformed/corrupt/reset authority fail closed with no READY. Inputs are type-validated and malformed references, scope, generations, fingerprints or evidence references produce controlled denial.

A cryptographically valid B1 at G cannot become current after legal protected commits reach G+N, even after loss of the newer local store. Backup is candidate persisted state only and cannot mint/advance/rollback membership, reconstruct missing membership, create pending, or finalize. It cannot resurrect consumed bootstrap, revoked device/operator/access authority, old security/session epochs, kill-switch state, leases, idempotency outcomes, or capability authority.

First-run bootstrap remains a separate ephemeral one-shot `INITIAL_SECURITY_ESTABLISHMENT_ONLY` contract. Its claim/reference does not mint restore membership and bootstrap generation is not restore generation, even if a future implementation shares broader provisioning infrastructure. Restore freshness grants no operator authentication, device TRUSTED, PIN/biometric verification, admin role, entitlement, LiveAccessGrant, RiskDecision, ExecutionLease, kill-switch reset, ProductCapability, accounting authority, or LIVE permission; current LIVE remains denied.

M0.3 owns external membership existence/semantics, exact scope, genesis, monotonic PREPARE/FINALIZE/ABORT and non-authority rules. M0.11 owns local evidence production, matching StateStore metadata, durable transactions, backup, restore validation/orchestration, migrations, failure registry and history-only recovery, but cannot mint M0.3 membership. Existing M0.10 security, M0.9 risk/lease, M0.4 capability and M0.8 accounting ownership remains unchanged. The canonical startup prefix remains unchanged; for initialized state, freshness reconciliation is subordinate to store integrity, is the first recovery gate, and precedes readiness as expanded by `corehost_startup_recovery_contract`.

## CoreHost startup recovery contract

The authoritative machine-readable contract is `corehost_startup_recovery_contract` in the companion JSON. It expands `recover writable StateStore` into one initialized startup pipeline: initial verified snapshot and exact scope binding; M0.3 protected recovery; fresh verification; sealed-registry Migration recovery; fresh verification; deterministic-only `handoff_id`-ordered SecretHandoff recovery; final verification; process-local evidence; recovery complete; durable RuntimeSession history; and only then readiness.

The pre-open `CoreHostScope.state_store_path` is only the normalized local filesystem locator selecting the SQLite handle. After a verified initialized snapshot is read, account and device metadata must exactly match `CoreHostScope`; the protected scope third dimension is the existing `state_store_identity_fingerprint_sha256` read from verified metadata. It is never rederived from the path and path equality grants no protected authority. Substitution at the same path therefore still requires exact M0.3 accepted/current authority for the substituted store’s verified `(account_id, device_installation_id, state_store_identity_fingerprint_sha256)` triple. Empty P1B has no established identity and does not create one.

An empty or genuinely uninitialized store skips protected, Migration, SecretHandoff, evidence, durable RuntimeSession history, bootstrap consumption, and security authority creation, then proceeds to readiness determination where `SETUP_REQUIRED` may be derived. Invalid initialized state fails closed without automatic physical restore. Subsystem production owners retain all semantic authority; CoreHost owns only sequencing and live handles.

## CoreHost startup recovery — sealed physical schema gate

Recovery-complete porównuje `SQLiteStateStore.sqlite_schema_fingerprint()` z wpisem `state_store_physical_schema_registry` wskazanym przez finalne zweryfikowane `StateStoreMetadata.state_store_schema_version`. Finalna wersja metadata musi być równa statycznemu `current_state_store_schema_version`. Zgodność fizycznego schematu jest konieczna, ale niewystarczająca i nie ustanawia żadnej authority.


## S8B-P1C-A1 — CoreHost RuntimeSession and readiness contract

Canonical owner: `process_topology_and_lifecycle.json#/corehost_runtime_session_and_readiness_contract` (M0.3). M0.2 zachowuje ownership identity/schema, a M0.11 carriera i chronionego protokołu transakcyjnego.

```json
{
  "contract_id": "S8B-P1C-A1",
  "semantic_owner": "M0.3 CoreHost startup sequencing and startup-readiness boundary",
  "dependencies": {
    "M0.2": "RuntimeSession domain identity/schema and identifier_policy in canonical_domain_vocabulary.json",
    "M0.11": "RuntimeSession canonical identity/history carrier and protected StateStore transaction_protocol in persistence_versioning_migrations_backup_and_recovery.json",
    "referenced_not_copied": [
      "M0.4 ProductCapabilities/environment policy gates",
      "M0.5 ExchangeAccount connection/configuration readiness",
      "M0.8 portfolio/accounting reconciliation",
      "M0.9 ExecutionLease/risk/kill-switch prerequisites",
      "M0.10 authorization/security state"
    ]
  },
  "active_runtime_session": {
    "manifestation": "EPHEMERAL PROCESS-LOCAL OBJECT",
    "created": "exactly once per CoreHost start attempt, after exact process-lock acquisition and before StateStore open",
    "canonical_fields": [
      "runtime_session_id",
      "device_installation_id"
    ],
    "runtime_session_id": {
      "policy": "M0.2 identifier_policy",
      "source": "fresh identity generated through Core-owned production path",
      "read_from_durable_history": false,
      "selected_from_latest_history": false,
      "deterministically_derived_from_StateStore": false,
      "is_authority": false,
      "caller_arbitrary_id_is_production_authority": false,
      "test_seam": "factory may reproduce a valid test object; nominal injection does not establish production authority"
    },
    "device_binding": "RuntimeSession.device_installation_id exact-matches CoreHostScope.device_installation_id",
    "device_mismatch": "FAIL_CLOSED_BEFORE_DURABLE_RUNTIMESESSION_PUBLICATION",
    "restored_from_history": false
  },
  "durable_carrier": {
    "representation_registry_key": "RuntimeSession canonical identity/history",
    "classification": "DURABLE IMMUTABLE / APPEND-ONLY HISTORY",
    "new_representation_kind": false,
    "current_designation_created": false,
    "record_key": "runtime_session_id",
    "payload": {
      "fact_kind": "RuntimeSession",
      "upstream_payload": {
        "runtime_session_id": "<current active runtime_session_id>",
        "device_installation_id": "<current CoreHost device_installation_id>"
      },
      "upstream_payload_fingerprint_sha256": "<existing exact canonical fingerprint>"
    },
    "forbidden_added_fields": [
      "timestamp",
      "PID",
      "hostname",
      "process start clock",
      "path",
      "account_id"
    ],
    "meaning": "historical identity fact only",
    "does_not_prove": [
      "process alive",
      "process-lock ownership",
      "authentication",
      "DeviceTrust",
      "READY",
      "LIVE",
      "current-session designation"
    ],
    "active_process_truth": "process-local"
  },
  "publication": {
    "branch": "INITIALIZED_ONLY",
    "preconditions": [
      "exact original process lock still held",
      "exact current RuntimeSession handle",
      "exact current StateStore handle",
      "CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE"
    ],
    "insufficient_inputs": [
      "caller boolean",
      "string recovered",
      "cached result",
      "history presence"
    ],
    "protected_scope": [
      "verified metadata account_id",
      "verified metadata device_installation_id",
      "verified metadata state_store_identity_fingerprint_sha256"
    ],
    "forbidden_scope_sources": [
      "path",
      "session ID",
      "caller scope override"
    ],
    "mutation_kind": "NEW SEMANTIC DURABLE MUTATION AFTER P1B RECOVERY-COMPLETE",
    "transaction_protocol_pointer": "persistence_versioning_migrations_backup_and_recovery.json#/transaction_protocol",
    "sequence": [
      "resolve exact current M0.3",
      "M0.3 PREPARE G+1",
      "prepare local semantic transaction",
      "all-or-nothing local durable commit G+1",
      "publish fresh LocalDurableEvidence",
      "M0.3 FINALIZE",
      "success after required durability/finality"
    ],
    "local_mutation": {
      "immutable_history_appends": [
        "exactly one new RuntimeSession canonical identity/history carrier for current runtime_session_id"
      ],
      "current_designation_writes": 0,
      "overwrites_old_history": false,
      "deletes_old_history": false,
      "metadata_and_descriptor": "existing generic StateStoreMetadata and StateStoreTransactionDescriptor contract"
    },
    "forbidden_paths": [
      "raw SQL",
      "direct SQLite INSERT",
      "unprotected append",
      "caller durable=True",
      "manual metadata generation bump"
    ],
    "immutable_scope_fields": [
      "account_id",
      "device_installation_id",
      "state_store_identity_fingerprint_sha256",
      "state_store_schema_version"
    ],
    "immutable_scope_exception": "only generation/fingerprint/current durable commit facts legally updated by generic transaction_protocol",
    "success": {
      "carrier_durable": true,
      "state_store_generation": "G+1",
      "m0_3": "exact COMMITTED/current for G+1",
      "local_durable_evidence": "fresh and matching G+1",
      "old_history_preserved": true,
      "next_step": "determine_startup_readiness"
    },
    "p1b_evidence": {
      "generation": "G",
      "may_substitute_for_post_publication_G_plus_1": false,
      "readiness_source": "post-publication finalized current state under existing transaction_protocol"
    }
  },
  "empty_uninitialized": {
    "classification": "CoreHostRecoveryClassification.EMPTY_UNINITIALIZED",
    "publication_count": 0,
    "next_step": "determine_startup_readiness",
    "disposition": "SETUP_REQUIRED",
    "must_not_create": [
      "durable RuntimeSession history",
      "StateStore identity",
      "StateStore genesis",
      "bootstrap consumption",
      "M0.10 security authority"
    ],
    "READY": "FORBIDDEN",
    "corehost_may_remain_active_for_setup": true,
    "setup_required_mints_bootstrap_consumption_authority": false
  },
  "idempotency_and_collision": {
    "repeated_start_same_running_corehost": {
      "publication_count": 1,
      "runtime_session_id_count": 1,
      "basis": "process-local CoreHost.start() idempotency; never durable-history adoption"
    },
    "preexisting_record_key_equal_current_runtime_session_id": "FAIL_CLOSED_RUNTIMESESSION_ID_COLLISION",
    "adopt_old_entry": false,
    "restore_previous_active_session": false,
    "fresh_process_requires_fresh_identity": true
  },
  "crash_matrix": {
    "BEFORE_M0_3_PREPARE": {
      "new_history_durable": false,
      "state_store": "old G remains",
      "restart": "creates NEW runtime_session_id and ordinary startup proceeds from G"
    },
    "PREPARED_G_PLUS_1_BEFORE_LOCAL_COMMIT": {
      "new_history_durable": false,
      "restart": "creates NEW runtime_session_id before StateStore; P1B resolves pending with frozen M0.3 recovery",
      "exact_local_G": "existing ABORT semantics apply when it proves pre-candidate state",
      "after_recovery": "publish only new process own RuntimeSession history",
      "blind_retry_old_session": false
    },
    "LOCAL_COMMIT_G_PLUS_1_BEFORE_FINALIZE": {
      "crashed_session_history": "durable historical fact",
      "restart": "creates NEW runtime_session_id; P1B observes exact local G+1 plus PREPARED G+1 and FINALIZEs exact pending authority",
      "old_process_restored": false,
      "after_recovery": "current process publishes own history at later protected generation",
      "duplicate_business_commit": false
    },
    "AFTER_FINALIZE_BEFORE_READINESS": {
      "old_history": "remains durable",
      "restart": "creates NEW runtime_session_id, runs P1B, then publishes own history",
      "old_process_restored": false
    }
  },
  "failure": {
    "unresolved_publication_failure": "FAIL_CLOSED",
    "READY": "FORBIDDEN",
    "startup_readiness_success_disposition": "FORBIDDEN",
    "cleanup": "CLOSE_EXACT_CURRENT_STATESTORE_CLOSE_EXACT_RUNTIMESESSION_RELEASE_EXACT_PROCESS_LOCK per existing CoreHost failed-start ownership semantics",
    "rollback_completed_external_effects": false,
    "delete_already_durable_history": false
  },
  "startup_readiness": {
    "classification": "PROCESS-LOCAL NON-AUTHORITY",
    "dispositions": [
      "SETUP_REQUIRED",
      "PROCEED_TO_LATER_STARTUP_GATES"
    ],
    "READY_outcome_present": false,
    "separation": [
      "recovery result",
      "startup readiness disposition",
      "final M0.2/Core runtime READY state"
    ],
    "not_equal": [
      "INITIALIZED_RECOVERY_COMPLETE != READY",
      "durable RuntimeSession history != READY",
      "LocalDurableEvidence != READY",
      "startup readiness disposition != READY"
    ],
    "initialized_requires": [
      "CoreHostRecoveryClassification.INITIALIZED_RECOVERY_COMPLETE",
      "successful durable current RuntimeSession history publication finalized at G+1"
    ],
    "initialized_disposition": "PROCEED_TO_LATER_STARTUP_GATES",
    "initialized_disposition_means": "P1C boundary passed only",
    "initialized_disposition_does_not_mean": [
      "READY",
      "authenticated",
      "authorized",
      "reconciled",
      "leased",
      "LIVE-capable"
    ],
    "authority_boolean_inputs_forbidden": [
      "recovered=True",
      "session_published=True",
      "authenticated=True",
      "authorized=True",
      "reconciled=True",
      "lease_valid=True",
      "kill_switch_clear=True",
      "capability_ok=True",
      "ready=True"
    ],
    "derivation": "CoreHost-owned exact process state plus verified/persisted outputs of their semantic owners"
  },
  "final_ready_boundary": {
    "p1c_mints_READY": false,
    "final_owner": "separate later CoreHost startup_sequence ready_gate",
    "required_later_gates": [
      {
        "gate": "startup readiness disposition",
        "owner_milestone": "M0.3",
        "artifact": "process_topology_and_lifecycle.json",
        "json_pointer": "/corehost_runtime_session_and_readiness_contract/startup_readiness"
      },
      {
        "gate": "ExchangeAccount connection/configuration readiness",
        "owner_milestone": "M0.5",
        "artifact": "exchange_accounts_and_instruments.json",
        "json_pointer": "/exchange_account_contract"
      },
      {
        "gate": "portfolio/accounting reconciliation",
        "owner_milestone": "M0.8",
        "artifact": "ledger_portfolio_capital_and_pnl.json",
        "json_pointer": "/reconciliation_protocol"
      },
      {
        "gate": "authorization/security state",
        "owner_milestone": "M0.10",
        "artifact": "identity_device_authentication_and_secrets.json",
        "json_pointer": "/authority"
      },
      {
        "gate": "ExecutionLease and risk prerequisites",
        "owner_milestone": "M0.9",
        "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "json_pointer": "/execution_lease_contract"
      },
      {
        "gate": "kill-switch state",
        "owner_milestone": "M0.9",
        "artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "json_pointer": "/kill_switch_contract"
      },
      {
        "gate": "ProductCapabilities",
        "owner_milestone": "M0.4",
        "artifact": "environment_and_product_capabilities.json",
        "json_pointer": "/ProductCapabilities"
      },
      {
        "gate": "environment/policy readiness",
        "owner_milestone": "M0.4",
        "artifact": "environment_and_product_capabilities.json",
        "json_pointer": "/environment_readiness"
      }
    ],
    "connect_accounts_step_remains_after_p1c": true,
    "ready_gate_remains_after_later_gates": true,
    "missing_owner_or_api": "NOT_YET_READY / PROCEED_TO_LATER_STARTUP_GATES",
    "default_allow_forbidden": [
      "True",
      "default allow",
      "dummy grant",
      "placeholder ready",
      "test-only authority",
      "not implemented therefore pass"
    ]
  },
  "state_model_separation": [
    "process_health_state",
    "M0.2 runtime_state",
    "P1C startup readiness disposition",
    "exchange connection state",
    "operator GUI lock state"
  ],
  "production_implementation_boundary": {
    "may_implement_next": [
      "Core-created active RuntimeSession identity/object",
      "exact durable RuntimeSession history publisher",
      "integration after P1B recovery-complete",
      "empty -> SETUP_REQUIRED",
      "initialized plus successful publication -> PROCEED_TO_LATER_STARTUP_GATES",
      "process-local typed startup disposition",
      "fail-closed errors and cleanup"
    ],
    "deferred": [
      "bootstrap consumption",
      "first OperatorIdentity",
      "DeviceTrust transition",
      "PIN setup/authentication",
      "LiveAccessGrant",
      "ExchangeAccount connect",
      "reconciliation engine",
      "ExecutionLease issuance",
      "risk gate",
      "final ready_gate success",
      "LIVE"
    ]
  },
  "ownership_domains": {
    "runtime_session_identity_and_schema": {
      "owner_milestone": "M0.2",
      "artifact": "canonical_domain_vocabulary.json",
      "json_pointers": [
        "/entity_kinds",
        "/identifier_policy"
      ]
    },
    "durable_carrier_and_transaction_protocol": {
      "owner_milestone": "M0.11",
      "artifact": "persistence_versioning_migrations_backup_and_recovery.json",
      "json_pointers": [
        "/backup_contract/representation_registry/RuntimeSession canonical identity~1history",
        "/transaction_protocol",
        "/runtime_session_persistence/publication"
      ]
    },
    "startup_sequencing_phase_boundaries_and_readiness": {
      "owner_milestone": "M0.3",
      "artifact": "process_topology_and_lifecycle.json",
      "json_pointers": [
        "/corehost_startup_recovery_contract/recover_writable_state_store_phase_binding",
        "/corehost_runtime_session_and_readiness_contract/startup_readiness"
      ]
    }
  },
  "phase_boundaries": {
    "P1B_RECOVERY": {
      "starts_at": "read_verified_snapshot_initial",
      "ends_at_inclusive": "recovery_complete",
      "steps": [
        "read_verified_snapshot_initial",
        "bind_corehost_scope_to_verified_state_store_identity",
        "recover_m0_3_protected_freshness",
        "read_verified_snapshot_after_protected_recovery",
        "recover_migrations_in_sealed_schema_chain_order",
        "read_verified_snapshot_after_migration_recovery",
        "recover_secret_handoffs_in_handoff_id_order",
        "read_verified_snapshot_final",
        "publish_fresh_process_local_durable_evidence",
        "recovery_complete"
      ],
      "includes_runtime_session_publication": false,
      "includes_startup_readiness": false,
      "includes_READY": false
    },
    "P1C": {
      "starts_after": "recovery_complete",
      "steps": [
        "durably_publish_current_runtime_session_history",
        "determine_startup_readiness"
      ],
      "terminal_dispositions": [
        "SETUP_REQUIRED",
        "PROCEED_TO_LATER_STARTUP_GATES"
      ],
      "includes_final_READY": false
    },
    "LATER_STARTUP": {
      "starts_after": "PROCEED_TO_LATER_STARTUP_GATES",
      "uses_existing_later_gates": true,
      "final_step": "ready_gate"
    }
  },
  "global_initialized_order": [
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
    "PROCEED_TO_LATER_STARTUP_GATES",
    "later_startup_gates",
    "ready_gate",
    "READY_only_if_all_later_gates_pass"
  ],
  "global_initialized_order_authority": "M0.3 SINGLE AUTHORITATIVE GLOBAL ORDER"
}
```
