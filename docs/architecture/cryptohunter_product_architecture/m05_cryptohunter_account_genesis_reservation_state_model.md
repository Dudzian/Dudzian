# M0.5 CryptoHunterAccount genesis reservation operation state model

This file is a deterministic complete projection of `m05_cryptohunter_account_genesis_reservation_state_model.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_CRYPTOHUNTER_ACCOUNT_GENESIS_RESERVATION_OPERATION_STATE_MODEL",
  "iteration": "CONTRACT / DESIGN FREEZE ONLY",
  "repository_head_examined": "bf1433649f0c547ed5c6c21d70f9bca13f7cf438",
  "reviewed_head_supplied": "76afe748f3d130684c9daa0f9e0b4368c233f227",
  "provenance": {
    "source": "GIT",
    "reviewed_head_available_locally": false,
    "classification": "UNKNOWN",
    "relationship_to_reviewed_sha": "UNKNOWN",
    "actual_local_source_tree": "bf1433649f0c547ed5c6c21d70f9bca13f7cf438",
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_advancement_allowed": false,
    "artifacts_examined": [
      "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json",
      "m05_cryptohunter_account_genesis_authority_model.json",
      "m05_cryptohunter_account_root_of_trust_reconciliation.json",
      "m05_cryptohunter_account_authority_contract_design.json"
    ],
    "rule": "The unavailable supplied reviewed SHA is not equated with the actual local source tree; no content-equivalence claim is made."
  },
  "identity_boundaries": {
    "account_id": "entity identity only",
    "account_id_is_operation_identity": false,
    "reservation_is_genuine_account": false,
    "reserved_account_id_is_authority": false,
    "genuine_account_exists_only_after_durable_genesis_commit": true,
    "reservation_identity": "separate conceptual identity; exact reservation_id requirement DESIGN_BLOCKED",
    "logical_operation_identity": "authority-bound identity; exact type and issuer NOT_FROZEN",
    "account_entity_identity": "account_id",
    "all_three_identical": false,
    "subject_cardinality": "NOT_FROZEN"
  },
  "candidate_state_models": {
    "A_immutable_operation_journal_authenticated_current_index": {
      "status": "VIABLE_REQUIREMENTS / NOT_SELECTED",
      "gap": "freshness and owners unavailable"
    },
    "B_immutable_reservation_journal_immutable_genesis_journal": {
      "status": "VIABLE_REQUIREMENTS / NOT_SELECTED",
      "gap": "atomic cross-journal closure and freshness unavailable"
    },
    "C_unified_append_only_account_genesis_operation_journal": {
      "status": "PREFERRED_SHAPE / NOT_SELECTED",
      "gap": "root, owner, authentication keys and freshness authority unavailable"
    },
    "D_current_only_state_external_freshness_anchor": {
      "status": "INSUFFICIENT_BY_DEFAULT",
      "gap": "history closure absent; external anchor unavailable"
    },
    "E_DESIGN_BLOCKED": {
      "status": "SELECTED"
    }
  },
  "selected_or_blocked_state_model": {
    "selection": "DESIGN_BLOCKED",
    "reason": "No genuine root-of-trust, operation/reservation owner, authenticated durable protocol, or external freshness authority is available. Structural safety requirements are frozen, but no production state model/schema is selected.",
    "exact_schema": "DESIGN_BLOCKED",
    "durable_history_model": "DESIGN_BLOCKED"
  },
  "logical_operation_object": {
    "status": "REQUIRED_SEMANTICS_FROZEN / EXACT_SCHEMA_DESIGN_BLOCKED",
    "required_facts": [
      "operation identity",
      "authority domain",
      "reservation identity/account_id binding",
      "canonical semantic request binding",
      "root-proof provenance binding",
      "created/prepared generation",
      "current state",
      "commit reference when committed",
      "authenticated predecessor"
    ],
    "operation_identity_issuance": "NOT_FROZEN",
    "AUTHORITY_ISSUED_GENESIS_REQUEST_ID": "PREFERRED_CONCEPTUAL_CANDIDATE / NOT_SELECTED",
    "retention": "INDEFINITE_SECURITY_HISTORY_REQUIRED; bounded TTL FORBIDDEN until replay-safe proof exists",
    "caller_supplied_identity_alone_establishes_genuine_operation": false,
    "caller_may_mint_genuine_operation_identity": false,
    "operation_identity_issuance_timing": "NOT_FROZEN",
    "issuance_timing_candidates": {
      "BEFORE_RESERVATION": "CANDIDATE / NOT_SELECTED",
      "ATOMIC_WITH_RESERVATION": "CANDIDATE / NOT_SELECTED",
      "INSIDE_FIRST_PREPARE": "CONDITIONAL_CANDIDATE / NOT_SELECTED"
    },
    "authority_binding_deadline": "before any durable state is trusted after restart, it must be covered by some genuine authority-bound recovery identity: operation identity, or reservation recovery identity under a frozen handoff protocol",
    "preexisting_operation_identity_before_reservation_required": false,
    "INSIDE_FIRST_PREPARE_prerequisite": "AUTHORITY_BOUND_RESERVATION_RECOVERY_IDENTITY + AUTHENTICATED_CANONICAL_REQUEST_BINDING + DETERMINISTIC_HANDOFF_TO_OPERATION_IDENTITY",
    "INSIDE_FIRST_PREPARE_without_prerequisite": "restart before PREPARED -> FAIL_CLOSED; caller assertion cannot recover the operation and acct_B cannot be allocated",
    "caller_command_ID_is_recovery_identity": false,
    "caller_command_ID_is_handoff_authority": false
  },
  "operation_state_machine": {
    "status": "STRUCTURAL_SEMANTICS_FROZEN / NAMES_AND_SCHEMA_DESIGN_BLOCKED",
    "conceptual_states": [
      "NEW_OR_UNOBSERVED",
      "RESERVED",
      "PREPARED",
      "COMMITTED",
      "ABORTED"
    ],
    "RESERVED_PREPARED_merge": "PERMITTED_ONLY_IF the single durable transition binds acceptance, canonical request and reservation atomically; NOT_SELECTED",
    "PREPARED": "durably accepted operation with canonical request, authority domain, root-proof provenance, and exactly one reservation/account_id bound; recoverable after restart; not genuine account",
    "COMMITTED": "terminal successful outcome referencing the logically atomic authenticated genesis decision closure; it does not select physical storage atomicity",
    "ABORTED": "immutable terminal unsuccessful operation outcome retaining authenticated history and reservation tombstone; same operation cannot restart as new",
    "forbidden_transition": "UNKNOWN -> COMMITTED without genuine genesis decision",
    "terminal_states": [
      "COMMITTED",
      "ABORTED"
    ],
    "terminal_outcome_immutable": true,
    "ABORTED_to_RELEASED_operation_transition": "FORBIDDEN",
    "post_terminal_reservation_disposition": "tracked only by reservation state; never an operation state or operation outcome mutation",
    "RESERVED": "trusted/recoverable logical-operation state only when an authority-bound operation identity exists or a frozen reservation-to-operation recovery binding provides equivalent authority; reservation RESERVED alone does not imply operation RESERVED"
  },
  "reservation_state_machine": {
    "status": "STRUCTURAL_SEMANTICS_FROZEN / OWNER_AND_SCHEMA_DESIGN_BLOCKED",
    "conceptual_states": [
      "UNOBSERVED",
      "RESERVED",
      "BOUND_PREPARED",
      "CONSUMED_COMMITTED",
      "ABORTED_HELD",
      "RELEASED_TOMBSTONED"
    ],
    "owner_candidates": {
      "A_future_CryptoHunterAccountAuthority": "NOT_AVAILABLE / NOT_SELECTED",
      "B_AccountIdReservationAuthority": "NOT_FOUND / NOT_SELECTED",
      "C_external_issuer": "NOT_FOUND / NOT_SELECTED",
      "D_atomic_issuer_account_authority": "NOT_FOUND / NOT_SELECTED",
      "E_DESIGN_BLOCKED": "SELECTED"
    },
    "owner": "DESIGN_BLOCKED",
    "account_id_as_durable_reservation_key": "NOT_FROZEN",
    "separate_reservation_id": "NOT_FROZEN",
    "missing_lookup": "FAIL_CLOSED; never permission to allocate a new account_id",
    "restart_safe_reservation_recovery_identity": "REQUIRED_IF_RESERVATION_PRECEDES_OPERATION_ID",
    "durable_reservation_serializable_authority_scope_required": true,
    "serializable_authority_scope_exact_key": "DESIGN_BLOCKED",
    "account_id_alone_proves_operation_ownership": false,
    "account_id_key_requires_authenticated_authority_binding": true
  },
  "operation_reservation_binding": {
    "invariant": "one logical operation -> at most one reservation/account candidate",
    "same_operation_two_account_ids": "STRUCTURALLY_FORBIDDEN_OR_FAIL_CLOSED",
    "retry_same_operation": "recover same reservation, account_id, canonical request and final outcome; no reallocation",
    "changed_semantic_request": "CONFLICT / CONTRACT_INCONSISTENT; no mutation",
    "new_operation": "requires authority proof of distinctness; different caller ID or account_id alone is insufficient",
    "same_account_different_operations": "at most one commit under per-account_id CAS; deterministic loser is CONFLICT or ALREADY_ADMITTED according to authenticated committed fact; no LWW"
  },
  "genesis_commit_binding": {
    "genuine_at": "IFF authenticated fresh recovery closure proves one logically atomic authenticated genesis decision binding the exact genesis record + exact operation COMMITTED outcome + exact reservation CONSUMED_COMMITTED",
    "only_COMMITTED_establishes_genuine_account": true,
    "PREPARED_grants_account_authority": false,
    "partial_visibility": "must not publish account or resolver until recovery closure",
    "account_genuine_operation_prepared_after_recovery": "FORBIDDEN",
    "crash_after_genesis_record_write": "POSSIBLE ONLY AS PARTIAL/UNPUBLISHED DURABLE STATE if the future physical model permits separate writes; restart verifies authenticated fresh closure and either recovers the same COMMITTED genesis or FAIL_CLOSED; row existence alone never publishes an account",
    "logical_commit_closure_invariant": "FROZEN",
    "physical_transaction_interpretation": "NOT_FROZEN; logical closure does not select a physical DB transaction",
    "genesis_record_exists_without_verified_commit_closure_is_genuine_account": false,
    "partial_genesis_row_authority": "NOT SUFFICIENT FOR AUTHORITY",
    "partial_genesis_PREPARED_BOUND_PREPARED_is_COMMITTED": false,
    "physical_genesis_row_is_commit_point": false
  },
  "abort_release_semantics": {
    "status": "FROZEN_AT_SAFETY_SEMANTICS / AUTHORIZER_DESIGN_BLOCKED",
    "ABORT": "records terminal failure without erasing history; reservation becomes ABORTED_HELD and is not reusable",
    "RELEASE": "reservation-disposition-only authenticated post-abort transition from ABORTED_HELD to RELEASED_TOMBSTONED; preserves operation history and immutable ABORTED outcome",
    "same_operation_may_restart": false,
    "abort_permanently_consumes_operation_identity": true,
    "caller_may_unilaterally_abort_or_erase_history": false,
    "authorizer": "DESIGN_BLOCKED",
    "RELEASE_belongs_to": "reservation disposition only",
    "RELEASE_preserves_operation_terminal_ABORTED": true,
    "RELEASE_preserves_operation_history": true,
    "RELEASE_may_alter_only_reservation_disposition": true,
    "RELEASE_changes_operation_identity": false,
    "RELEASE_changes_operation_terminal_outcome": false,
    "RELEASE_turns_same_operation_into_NEW": false
  },
  "reuse_semantics": {
    "account_id_after_abort": "NOT_FROZEN; default DENY/FAIL_CLOSED",
    "reuse_without_authenticated_release": "FORBIDDEN",
    "authenticated_release_implies_reuse_permission": false,
    "required_before_any_future_reuse": "a separately frozen authenticated reuse protocol plus genuinely new authority-proven operation; currently unavailable"
  },
  "retry_matrix": [
    {
      "case": "same operation, same semantics, RESERVED/PREPARED",
      "outcome": "recover same reservation/account_id; continue or return authenticated terminal outcome; never reallocate"
    },
    {
      "case": "same operation, changed canonical semantics",
      "outcome": "CONFLICT / CONTRACT_INCONSISTENT"
    },
    {
      "case": "same operation, COMMITTED",
      "outcome": "return same committed outcome"
    },
    {
      "case": "same operation, ABORTED",
      "outcome": "always return ABORTED terminal operation outcome, whether reservation is ABORTED_HELD or RELEASED_TOMBSTONED"
    },
    {
      "case": "missing reservation lookup for known/uncertain operation",
      "outcome": "FAIL_CLOSED"
    },
    {
      "case": "claimed new operation",
      "outcome": "require authority proof of distinctness before reservation"
    }
  ],
  "crash_matrix": [
    {
      "point": "before reservation",
      "restart_outcome": "authenticate operation history; if truly unobserved, only authority-proven operation may begin; uncertainty FAIL_CLOSED"
    },
    {
      "point": "after reservation",
      "restart_outcome": "if authority-bound operation identity exists, recover same operation + reservation + account_id; else if genuine authority-bound reservation recovery identity with frozen deterministic handoff exists, recover reservation and deterministically recover/bind the same operation; else FAIL_CLOSED; always never allocate acct_B"
    },
    {
      "point": "after PREPARED",
      "restart_outcome": "recover PREPARED binding and resume/reconcile; it is not genuine"
    },
    {
      "point": "after partial genesis record write (only if future physical model permits)",
      "restart_outcome": "treat as partial/unpublished; verify authenticated fresh closure, then recover the same COMMITTED genesis or FAIL_CLOSED; never publish from row existence and never remint"
    },
    {
      "point": "after operation COMMITTED",
      "restart_outcome": "verify closure/freshness and return the same committed outcome"
    },
    {
      "point": "before response",
      "restart_outcome": "retry returns the durable terminal outcome"
    },
    {
      "point": "during ABORT/RELEASE",
      "restart_outcome": "replay authenticated transition using expected generation; ambiguous/torn state FAIL_CLOSED and no reuse"
    }
  ],
  "concurrency": {
    "per_operation_serialization": "REQUIRED",
    "per_reservation_serialization": "REQUIRED",
    "per_account_id_serialization": "REQUIRED",
    "same_account_two_operations": "at most one commit",
    "loser": "deterministic CONFLICT or ALREADY_ADMITTED; DENY on unverifiable state",
    "last_writer_wins": false,
    "global_sequence": "persistence ordering only; not a singleton-account rule",
    "per_operation_serialization_enforceable_when": "genuine operation identity or equivalent frozen recovery binding exists",
    "before_operation_serialization_enforceable": "reservation scope prevents concurrent remint",
    "durable_reservation_serializable_authority_scope": "REQUIRED_BEFORE_RESTART_SURVIVAL / exact key DESIGN_BLOCKED"
  },
  "CAS": {
    "required": true,
    "scopes": [
      "logical operation",
      "reservation",
      "account_id/genesis"
    ],
    "mechanism": "expected authenticated generation/predecessor; stale writers fail closed"
  },
  "durability": {
    "security_history": "append-only authenticated history required conceptually",
    "current_only_row_sufficient": false,
    "idempotency_retention": "for lifetime of account and indefinitely for terminal/tombstone security facts unless a future replay-safe retirement protocol is frozen",
    "bounded_TTL_safe": false
  },
  "authentication": {
    "security_critical_state_authenticated": true,
    "public_SHA_sufficient": false,
    "local_hashes_sufficient": false,
    "M012_HMAC_convention": "CONCEPTUALLY_REUSABLE_PATTERN_ONLY",
    "key_and_owner": "DESIGN_BLOCKED",
    "coherent_local_rewrite_trusted": false
  },
  "freshness": {
    "valid_prefix_rollback_must_be_rejected": true,
    "external_freshness_authority": "NOT_AVAILABLE / DESIGN_BLOCKED",
    "local_generation_alone_sufficient": false
  },
  "rollback": {
    "COMMITTED_to_PREPARED_valid_prefix": "MUST_FAIL_CLOSED",
    "split_brain_prepared_reservation_committed_genesis": "reconcile only with authenticated fresh closure to same COMMITTED operation; otherwise FAIL_CLOSED; never acct_B",
    "coherent_rewrite_operation_reservation_genesis_local_hashes": "NOT_TRUSTED"
  },
  "restart": {
    "status": "ALGORITHM_ORDER_FROZEN / PRODUCTION_CLOSURE_DESIGN_BLOCKED",
    "order": [
      "verify durable storage",
      "verify authentication",
      "verify journal closure",
      "verify freshness",
      "reconstruct operations/reservations",
      "reconcile committed genesis facts",
      "publish resolver only after closure"
    ],
    "current_row_trusted_directly": false,
    "authority_available_on_failed_closure": false,
    "failure_mode": "FAIL_CLOSED"
  },
  "environment_isolation": {
    "required_separation": [
      "storage",
      "authentication domain",
      "keys",
      "freshness state"
    ],
    "TEST_prepared_to_PRODUCTION_commit": "DENY",
    "cross_environment_identity_reuse": "DENY unless a future explicit authenticated migration contract is frozen"
  },
  "M011_interaction": {
    "role": "derived carrier only; NOT AUTHORITY",
    "PREPARED_or_RESERVED_stored_solely_in_projection": false,
    "restored_projection_recreates_authority": false
  },
  "M03_interaction": {
    "device_first_run_generation_revision_reused": false,
    "reusable_identity_for_account_genesis": false,
    "ownership_migration": "NOT_PROVEN"
  },
  "cross_artifact_parity": {
    "parity": "PASS",
    "account_id_is_operation_identity": false,
    "reservation_is_authority": false,
    "same_operation_may_remint_after_crash": false,
    "different_ids_prove_distinct_operations": false,
    "subject_cardinality": "NOT_FROZEN",
    "entity_uniqueness": "account_id / ENTITY_UNIQUENESS_KEY_FROZEN",
    "logical_idempotency_identity": "NOT_FROZEN",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "genuine_account_only_after_genuine_genesis_commit": true,
    "prior_commit_does_not_select_physical_DB_transaction": true
  },
  "mandatory_redteam": {
    "PREPARED_reservation_treated_as_genuine_account": "FAIL",
    "missing_reservation_lookup_allocates_new_account_id": "FAIL",
    "same_operation_owns_acct_A_and_acct_B": "FAIL",
    "valid_prefix_PREPARED_accepted_after_COMMITTED": "FAIL",
    "caller_abort_erases_security_history": "FAIL",
    "ABORTED_reservation_silently_reused": "FAIL",
    "current_only_local_hashes_prove_authenticity": "FAIL",
    "crash_after_genesis_write_creates_second_genesis": "FAIL",
    "TEST_prepared_operation_to_PRODUCTION_commit": "FAIL",
    "M011_restore_recreates_PREPARED_authority": "FAIL",
    "ABORTED_operation_transitions_to_RELEASED_operation": "FAIL",
    "reservation_RELEASE_mutates_operation_terminal_result": "FAIL",
    "genesis_row_alone_publishes_genuine_account": "FAIL",
    "partial_genesis_PREPARED_operation_is_COMMITTED": "FAIL",
    "physical_genesis_row_treated_as_commit_point_while_model_blocked": "FAIL",
    "caller_command_ID_becomes_genuine_operation_identity": "FAIL",
    "state_model_requires_operation_ID_before_reservation_while_timing_NOT_FROZEN": "FAIL",
    "INSIDE_FIRST_PREPARE_restartable_reservation_without_authority_recovery_identity_recovers_normally": "FAIL",
    "account_id_alone_proves_retry_operation_ownership": "FAIL",
    "caller_command_ID_provides_reservation_to_operation_handoff_authority": "FAIL",
    "crash_after_reservation_unidentified_operation_allocates_new_account_id": "FAIL"
  },
  "result": {
    "primary_result": "ACCOUNT_GENESIS_RESERVATION_STATE_MODEL_DESIGN_BLOCKED",
    "intrinsic_blockers": [
      "exact operation identity and issuance protocol NOT_FROZEN",
      "authority-bound reservation recovery identity and deterministic handoff protocol DESIGN_BLOCKED",
      "reservation identity/key, owner and reuse protocol NOT_FROZEN",
      "authenticated append-only schema, keys and physical commit protocol not selected",
      "external freshness/anti-rollback authority unavailable"
    ],
    "upstream_blockers": [
      "genuine root-of-trust DESIGN_BLOCKED",
      "CryptoHunterAccountAuthority NOT_AVAILABLE"
    ],
    "frozen_despite_blockers": [
      "state safety meanings",
      "one operation to at most one reservation/account candidate",
      "atomic genesis/COMMITTED/consumption relationship",
      "retry, concurrency and fail-closed restart invariants",
      "authentication and freshness requirements",
      "abort retains history and no silent reuse"
    ]
  },
  "implementation_allowed": {
    "CryptoHunterAccountAuthority": false,
    "ProvisioningBoundary": false,
    "WorkspaceAuthority": false,
    "InstrumentAuthority": false,
    "WCP_Authority": false,
    "FullFillAuthority": false,
    "M0.8": false
  },
  "preserved_status": {
    "genesis_uniqueness_idempotency_design": "GIT",
    "entity_uniqueness": "account_id / ENTITY_UNIQUENESS_KEY_FROZEN",
    "selected_logical_idempotency_identity": "NOT_FROZEN",
    "distinct_account_ids_imply_distinct_operations": false,
    "crash_retry_may_remint": false,
    "per_account_id_serialization": "REQUIRED",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "production_M0.5": "NOT_AVAILABLE"
  },
  "candidate_commit_models": {
    "selection": "DESIGN_BLOCKED",
    "physical_commit_model": "NOT_SELECTED / DESIGN_BLOCKED",
    "A_SINGLE_PHYSICAL_ATOMIC_TRANSACTION": {
      "status": "VIABLE / NOT_SELECTED",
      "semantics": "genesis record + operation COMMITTED + reservation CONSUMED_COMMITTED become durable atomically; a durable externally visible partial genesis state cannot occur"
    },
    "B_MULTI_STEP_PREPARE_FINALIZE_WITH_RECOVERY_CLOSURE": {
      "status": "VIABLE / NOT_SELECTED",
      "semantics": "physical writes may be separate, but a partial genesis row is unpublished and grants no authority until authenticated fresh closure proves the complete decision"
    },
    "C_OTHER_FROZEN_PROTOCOL": {
      "status": "NOT_FOUND / NOT_SELECTED"
    },
    "D_DESIGN_BLOCKED": {
      "status": "SELECTED"
    },
    "physical_storage_atomicity": "NOT_FROZEN",
    "logical_authority_atomicity": "FROZEN through authenticated fresh recovery closure"
  },
  "recovery_identity": {
    "universal_invariant": "every durable / restart-recoverable pre-commit state MUST have an authority-bound recovery identity",
    "must_resolve": [
      "same operation",
      "same reservation",
      "same account_id",
      "same canonical request binding"
    ],
    "caller_controlled_identity_sufficient": false,
    "logical_operation_identity_and_reservation_recovery_identity_are_distinct_semantic_roles": true,
    "logical_operation_identity_equals_reservation_recovery_identity": "NOT_FROZEN; may be the same object only after explicit frozen model selection",
    "trusted_identity_options": [
      "authority-bound logical operation identity",
      "genuine authority-bound reservation recovery identity under a frozen deterministic handoff protocol"
    ],
    "identity_gap_outcome": "FAIL_CLOSED",
    "identity_gap_may_allocate_new_account_id": false,
    "identity_gap_may_recover_operation_by_caller_assertion": false
  },
  "pre_PREPARED_recovery_models": {
    "A_OPERATION_ID_BEFORE_OR_WITH_RESERVATION": {
      "status": "VIABLE / NOT_SELECTED",
      "semantics": "authority-bound operation identity exists before the reservation becomes restart-trusted"
    },
    "B_RESERVATION_FIRST_WITH_AUTHORITY_BOUND_RECOVERY_IDENTITY": {
      "status": "CONDITIONAL_CANDIDATE / NOT_SELECTED",
      "semantics": "durable reservation has its own genuine authority-bound recovery identity, authenticated canonical request binding, and deterministic one-way handoff to the later operation identity",
      "prerequisite": "AUTHORITY_BOUND_RESERVATION_RECOVERY_IDENTITY + AUTHENTICATED_CANONICAL_REQUEST_BINDING + DETERMINISTIC_HANDOFF_TO_OPERATION_IDENTITY"
    },
    "C_DESIGN_BLOCKED": {
      "status": "SELECTED"
    },
    "selection": "DESIGN_BLOCKED"
  },
  "critical_pre_PREPARED_identity_gap": {
    "operation_identity_issuance": "INSIDE_FIRST_PREPARE",
    "sequence": [
      "request O",
      "durable reservation acct_A",
      "crash before PREPARED",
      "retry arrives"
    ],
    "authority_bound_reservation_recovery_identity": "ABSENT",
    "outcome": "FAIL_CLOSED",
    "recover_O_by_caller_assertion": false,
    "allocate_acct_B": false
  }
}
```
