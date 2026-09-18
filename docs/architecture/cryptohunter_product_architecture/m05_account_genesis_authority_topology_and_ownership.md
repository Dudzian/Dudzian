# M0.5 Account genesis authority topology and ownership

This file is a deterministic complete projection of `m05_account_genesis_authority_topology_and_ownership.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_AND_OWNERSHIP_DESIGN",
  "iteration": "DESIGN / RECONCILIATION ONLY",
  "repository_head_examined": "2f840313801165b542eb07c8360f4f056eb576c3",
  "reviewed_head_supplied": "18107c75b75ad410c9ea481b2df4947566921a3e",
  "provenance": {
    "source": "GIT",
    "reviewed_head_available_locally": false,
    "classification": "UNKNOWN",
    "relationship_to_reviewed_sha": "UNKNOWN",
    "actual_local_source_tree": "2f840313801165b542eb07c8360f4f056eb576c3",
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_advancement_allowed": false,
    "rule": "Supplied reviewed SHA is unavailable locally; it is not equated with the current tree and no formal status advancement is made."
  },
  "frozen_inputs": {
    "security_substrate": "ACCOUNT_GENESIS_SECURITY_SUBSTRATE_CONTRACT_FROZEN",
    "root_of_trust": "DESIGN_BLOCKED",
    "reservation_owner": "DESIGN_BLOCKED",
    "account_id_mint_owner": "NOT_FROZEN",
    "genesis_decision_authority": "NOT_FOUND",
    "account_id_is_operation_identity": false,
    "reservation_is_authority": false,
    "PREPARED_is_genuine": false,
    "genuine_at": "COMMITTED closure only",
    "ABORTED": "terminal",
    "RELEASE": "reservation-only",
    "subject_cardinality": "NOT_FROZEN",
    "reuse_after_abort": "NOT_FROZEN / default DENY",
    "valid_MAC_is_authorization": false,
    "valid_MAC_is_freshness": false
  },
  "authority_roles": {
    "separation_invariant": "technical state owner != root-proof issuer != domain authorization decision != security-substrate owner unless an explicit frozen protocol safely joins roles",
    "authority_existence_invariant": "authority can exist before account != authority may self-authorize first account",
    "final_decision_invariant": "For one logical operation, at most one genuine authority decision can establish account genesis.",
    "semantic_vs_storage": "The semantic decision owner and storage transaction coordinator are distinct roles even if a future protocol assigns both to one component.",
    "reservation_invariant": "reservation owner != genuine account authority result; reserved account_id != genuine account_id"
  },
  "ownership_matrix": {
    "logical_operation_identity": {
      "owner": "DESIGN_BLOCKED",
      "status": "NOT_FROZEN",
      "evidence": "AUTHORITY_ISSUED_GENESIS_REQUEST_ID is preferred conceptual candidate, not selected",
      "authority_scope": "pre-account logical operation",
      "may_mint_trust": "NO",
      "notes": "Caller identity and root-proof ID are insufficient."
    },
    "account_id_mint": {
      "owner": "NOT_FROZEN",
      "status": "DESIGN_BLOCKED",
      "evidence": "Canonical provenance stops upstream; ProvisioningBoundary supplies but is not proven to mint account_id",
      "authority_scope": "candidate entity identity",
      "may_mint_trust": "NO",
      "notes": "Minting may later belong to AccountAuthority while admission still requires independent proof."
    },
    "reservation": {
      "owner": "DESIGN_BLOCKED",
      "status": "DESIGN_BLOCKED",
      "evidence": "Reservation state model selected DESIGN_BLOCKED",
      "authority_scope": "candidate account_id exclusion",
      "may_mint_trust": "NO",
      "notes": "Reservation never establishes a genuine account."
    },
    "reservation_recovery": {
      "owner": "DESIGN_BLOCKED",
      "status": "DESIGN_BLOCKED",
      "evidence": "Recovery identity required if reservation precedes operation identity",
      "authority_scope": "pre-PREPARED recovery",
      "may_mint_trust": "NO",
      "notes": "Identity gap -> FAIL_CLOSED and never acct_B."
    },
    "root_proof_issuance": {
      "owner": "M03_EXTERNAL_PRODUCT_PROVISIONING_MEMBERSHIP conditional candidate",
      "status": "BLOCKED_UPSTREAM",
      "evidence": "Production external issuer/registry is NOT_FOUND / NOT_AVAILABLE",
      "authority_scope": "independent pre-account root evidence",
      "may_mint_trust": "CONDITIONAL",
      "notes": "It supplies evidence only; issuance, account ID minting and genesis admission are not implied."
    },
    "root_proof_validation": {
      "owner": "DESIGN_BLOCKED",
      "status": "NOT_FROZEN",
      "evidence": "No genuine production ProvisioningBoundary / RootProofAuthority implementation or validator contract",
      "authority_scope": "proof verification/admission input",
      "may_mint_trust": "NO",
      "notes": "Verification is distinct from issuance and final genesis decision."
    },
    "genesis_semantic_decision": {
      "owner": "NOT_FOUND",
      "status": "DESIGN_BLOCKED",
      "evidence": "Account genesis authority model and root reconciliation do not identify an owner",
      "authority_scope": "PREPARED -> COMMITTED semantic decision",
      "may_mint_trust": "YES, only after independently rooted authority is frozen",
      "notes": "Exactly one future final owner is required; none exists today."
    },
    "durable_commit_coordination": {
      "owner": "DESIGN_BLOCKED",
      "status": "NOT_FROZEN",
      "evidence": "Logical closure is frozen but physical atomic protocol is not selected",
      "authority_scope": "authenticated durable closure",
      "may_mint_trust": "NO",
      "notes": "Coordinator cannot substitute for semantic authorization."
    },
    "ABORT": {
      "owner": "DESIGN_BLOCKED",
      "status": "DESIGN_BLOCKED",
      "evidence": "Safety meaning frozen; authorizer blocked",
      "authority_scope": "terminal logical-operation disposition",
      "may_mint_trust": "NO",
      "notes": "Caller may request cancellation but cannot erase or rewrite history."
    },
    "RELEASE": {
      "owner": "DESIGN_BLOCKED",
      "status": "DESIGN_BLOCKED",
      "evidence": "Reservation disposition is frozen; reservation owner/authorizer blocked",
      "authority_scope": "post-ABORT reservation disposition",
      "may_mint_trust": "NO",
      "notes": "Cannot change ABORTED or imply reuse permission."
    },
    "historical_resolver": {
      "owner": "DESIGN_BLOCKED",
      "status": "DESIGN_BLOCKED",
      "evidence": "Canonical account history owner is absent; M0.11 is projection carrier only",
      "authority_scope": "current and exact committed genesis resolution",
      "may_mint_trust": "NO",
      "notes": "Must verify historical identity and provenance after authenticated closure."
    },
    "custody": {
      "owner": "AccountGenesis-specific security component; exact component NOT_FROZEN",
      "status": "SECURITY_ROLE_FROZEN / OWNER_NOT_FROZEN",
      "evidence": "Security substrate contract",
      "authority_scope": "secret custody only",
      "may_mint_trust": "NO",
      "notes": "KeyringSecretStorage does not authorize genesis."
    },
    "freshness": {
      "owner": "AccountGenesis-specific security component; exact component NOT_FROZEN",
      "status": "SECURITY_ROLE_FROZEN / OWNER_NOT_FROZEN",
      "evidence": "Atomic multi-lineage anchor selected; exact component remains open",
      "authority_scope": "rollback/freshness comparison only",
      "may_mint_trust": "NO",
      "notes": "Freshness acceptance does not decide domain admission."
    }
  },
  "candidate_topologies": {
    "A_UNIFIED_CRYPTOHUNTER_ACCOUNT_AUTHORITY": "VIABLE_SHAPE / NOT_SELECTED",
    "B_SEPARATE_RESERVATION_AUTHORITY": "VIABLE_ONLY_WITH_FROZEN_HANDOFF_AND_ATOMICITY / NOT_SELECTED",
    "C_EXTERNAL_ISSUER_MINTS_OR_RESERVES_ACCOUNT_ID": "CONDITIONAL / NOT_SELECTED",
    "D_ATOMIC_ISSUER_ACCOUNT_AUTHORITY_PROTOCOL": "CONDITIONAL / NOT_SELECTED",
    "E_OTHER_CANONICALLY_SUPPORTED_MODEL": "NOT_FOUND",
    "F_DESIGN_BLOCKED": "SELECTED"
  },
  "topology_analysis": {
    "A_UNIFIED_CRYPTOHUNTER_ACCOUNT_AUTHORITY": {
      "benefit": "Single serialization and closure boundary can simplify operation, reservation, commit, abort/release and history.",
      "cost": "Concentration risks laundering technical ownership into self-authorization.",
      "mandatory_constraint": "MUST reject first-account creation using only locally generated evidence; independent accepted root proof remains required.",
      "root_validation": "May be a distinct dependency/interface; issuance remains external.",
      "candidate_id_technical_ownership": "AccountAuthority may mint/reserve candidate IDs",
      "technical_ownership_is_self_authorization_permission": false,
      "genuine_genesis_precondition": "independent accepted root proof"
    },
    "B_SEPARATE_RESERVATION_AUTHORITY": {
      "flow": [
        "reservation commit",
        "authenticated handoff",
        "genesis decision",
        "reservation consumption"
      ],
      "costs": [
        "cross-authority atomicity",
        "independent freshness heads",
        "split-brain recovery",
        "rollback ambiguity",
        "orphan reservation",
        "double consumption",
        "ownership cycles"
      ],
      "split_brain": "CONSUMED/PREPARED and COMMITTED/RESERVED are never publishable genuine closure.",
      "selection_reason": "No authenticated handoff, shared decision recovery, or atomic protocol exists; conceptual neatness is insufficient."
    },
    "C_EXTERNAL_ISSUER_MINTS_OR_RESERVES_ACCOUNT_ID": {
      "unresolved": [
        "UUIDv7 syntax and uniqueness owner",
        "retry serializer",
        "reservation history owner",
        "issuer replay handler",
        "issuer rollback handler",
        "reservation consumer",
        "collision handling across issuers"
      ],
      "injection_gate": "Issuer identity, account_id, operation, environment, reservation state and proof provenance must be bound by a frozen protocol; arbitrary caller account_id is forbidden."
    },
    "D_ATOMIC_ISSUER_ACCOUNT_AUTHORITY_PROTOCOL": {
      "requirement": "Explicit atomic prepare/commit/abort/recovery and one final decision are required.",
      "current": "No protocol, participant ownership, or shared freshness semantics is frozen."
    },
    "E_OTHER_CANONICALLY_SUPPORTED_MODEL": "No canonical evidence found.",
    "F_DESIGN_BLOCKED": "Selected because the reviewed SHA is unavailable and current canonical evidence does not own root validation, operation identity, mint/reservation, semantic decision, or atomic closure."
  },
  "selected_or_blocked_topology": {
    "selection": "F_DESIGN_BLOCKED",
    "selected_authority_topology": "DESIGN_BLOCKED",
    "reason": "No candidate has a complete non-circular root, ownership, serialization, recovery and authenticated atomic commit contract.",
    "one_final_owner_requirement": "FROZEN",
    "one_final_owner_identity": "NOT_FOUND"
  },
  "root_proof_boundary": {
    "candidate": "M03_EXTERNAL_PRODUCT_PROVISIONING_MEMBERSHIP",
    "candidate_status": "VIABLE_ONLY_AFTER_ADDITIONAL_AUTHORITY / BLOCKED_UPSTREAM",
    "issuer": "NOT_FOUND / conditional external provisioning issuer",
    "validator": "DESIGN_BLOCKED",
    "issuance_is_validation": false,
    "issuance_implies_account_id_mint": false,
    "proof_id_is_operation_id": false,
    "consumption_semantics": {
      "single_use": "NOT_FROZEN",
      "multi_use": "NOT_FROZEN",
      "account_bound": "claim evidence binds account_id but AccountGenesis admission contract is NOT_FROZEN",
      "operation_bound": "NOT_FROZEN",
      "subject_bound": "NOT_FROZEN"
    },
    "replay": "R -> O1/acct_A and R -> O2/acct_B is NOT_FROZEN; depends on future consumption, business cardinality and issuer policy.",
    "non_circular": "Proof for acct_A MUST NOT depend on acct_A, its OperatorIdentity, DeviceInstallation, Workspace, or authority legitimate only within acct_A.",
    "first_account_self_authorization": {
      "locally_generated_evidence_only": "FORBIDDEN",
      "independent_accepted_root_proof_required": true,
      "technical_ownership_is_self_authorization_permission": false
    },
    "non_circular_root": {
      "target_account": "acct_A",
      "created_account_itself_allowed": false,
      "account_scoped_operator_identity_allowed": false,
      "account_scoped_device_installation_allowed": false,
      "account_scoped_workspace_allowed": false,
      "account_scoped_authority_allowed": false,
      "legitimacy_may_depend_on_entity_requiring_target_account_to_exist": false
    }
  },
  "operation_identity_ownership": {
    "owner": "DESIGN_BLOCKED",
    "AUTHORITY_ISSUED_GENESIS_REQUEST_ID": "PREFERRED_CONCEPTUAL_CANDIDATE / NOT_SELECTED",
    "same_as_reservation_owner": "NOT_FROZEN",
    "root_proof_id_substitution": "FORBIDDEN without proven exact issuance/retry semantics"
  },
  "account_id_mint_ownership": {
    "owner": "NOT_FROZEN",
    "AccountAuthority_may_mint_in_principle": true,
    "condition": "Independent accepted root proof must authorize admission before commit; mint does not authorize genesis.",
    "ProvisioningBoundary_mints": "NOT_PROVEN",
    "caller_arbitrary_id": "FORBIDDEN"
  },
  "reservation_ownership": {
    "owner": "DESIGN_BLOCKED",
    "reserved_is_genuine": false,
    "same_account_race": "O1 -> acct_A and O2 -> acct_A: at most one COMMITTED",
    "reuse_after_abort": "NOT_FROZEN / default DENY"
  },
  "reservation_recovery_ownership": {
    "owner": "DESIGN_BLOCKED",
    "required_when": "reservation can precede operation identity",
    "identity_gap": "FAIL_CLOSED",
    "identity_gap_allocates_acct_B": false
  },
  "genesis_decision_ownership": {
    "owner": "NOT_FOUND",
    "status": "DESIGN_BLOCKED",
    "exactly_one_required": true,
    "dual_independent_commit": "FORBIDDEN",
    "same_operation_split": "O -> acct_A and O -> acct_B: FORBIDDEN / FAIL_CLOSED"
  },
  "commit_coordination": {
    "owner": "DESIGN_BLOCKED",
    "semantic_owner_equals_storage_coordinator": "NOT_FROZEN",
    "publish_gate": "Only authenticated, fresh, complete COMMITTED closure may publish a genuine account.",
    "partial_split_brain_publish": "FORBIDDEN"
  },
  "abort_release_ownership": {
    "ABORT_owner": "DESIGN_BLOCKED",
    "RELEASE_owner": "DESIGN_BLOCKED",
    "caller_request_is_authorization": false,
    "ABORTED_terminal": true,
    "RELEASE_reservation_only": true,
    "RELEASE_changes_terminal_outcome": false,
    "RELEASE_allows_reuse": false
  },
  "historical_resolver_ownership": {
    "owner": "DESIGN_BLOCKED",
    "requirements": [
      "resolve current account",
      "resolve exact committed genesis",
      "verify historical account identity/provenance"
    ],
    "M0.11": "projection carrier only; never authority"
  },
  "security_substrate_boundary": {
    "custody_owner": "AccountGenesis-specific security component; exact component NOT_FROZEN",
    "freshness_owner": "AccountGenesis-specific security component; exact component NOT_FROZEN",
    "domain_authority_owner": "DESIGN_BLOCKED",
    "owners_are_semantically_identical": false,
    "does_not_decide": [
      "who may create account",
      "whether root proof is valid",
      "whether reservation is authorized",
      "whether genesis should occur"
    ],
    "security_owner_implies_AccountAuthority": false
  },
  "M03_boundary": {
    "external_membership": "conditional root-proof candidate only",
    "accepted_first_device_membership_equals_genesis_authority": false,
    "ProvisioningBoundary_account_id_mint": "NOT_PROVEN",
    "FirstRunBootstrapAuthority": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "FirstRunBootstrapAuthority_is_AccountAuthority": false
  },
  "M011_boundary": {
    "role": "projection carrier only",
    "may_mint_or_recreate_authority": false,
    "restored_projection_establishes_genesis": false
  },
  "concurrency": {
    "one_final_genesis_decision": "at most one genuine decision per logical operation",
    "same_account": "O1/acct_A and O2/acct_A -> at most one COMMITTED",
    "same_operation_split": "O/acct_A and O/acct_B -> FORBIDDEN / FAIL_CLOSED",
    "serialization_owner": "DESIGN_BLOCKED",
    "last_writer_wins": false
  },
  "replay": {
    "root_proof_replay": "NOT_FROZEN",
    "operation_retry": "Must resolve exact prior outcome under future authority-issued identity.",
    "caller_command_identity_sufficient": false,
    "account_id_reuse_after_abort": "NOT_FROZEN / default DENY"
  },
  "restart": {
    "order": [
      "verify security substrate",
      "verify authenticated history and freshness",
      "recover operation and reservation identity",
      "reconcile one genesis decision closure",
      "publish resolver only after complete COMMITTED closure"
    ],
    "identity_gap": "FAIL_CLOSED",
    "split_brain": "FAIL_CLOSED / DO_NOT_PUBLISH",
    "rollback_auto_repair": "FORBIDDEN"
  },
  "cross_authority_atomicity": {
    "status": "DESIGN_BLOCKED",
    "B_C_D_require_explicit_protocol": true,
    "reservation_CONSUMED_account_PREPARED": "FAIL_CLOSED / DO_NOT_PUBLISH",
    "account_COMMITTED_reservation_RESERVED": "FAIL_CLOSED / DO_NOT_PUBLISH",
    "independent_commits": "FORBIDDEN"
  },
  "cross_artifact_parity": {
    "parity": "PASS",
    "artifacts": [
      "m05_account_genesis_security_substrate_contract.json",
      "m05_cryptohunter_account_genesis_reservation_state_model.json",
      "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json",
      "m05_cryptohunter_account_genesis_authority_model.json",
      "m05_cryptohunter_account_root_of_trust_reconciliation.json"
    ],
    "account_id_not_operation_identity": true,
    "reservation_not_authority": true,
    "PREPARED_not_genuine": true,
    "ABORTED_terminal": true,
    "RELEASE_reservation_only": true,
    "subject_cardinality": "NOT_FROZEN",
    "root_of_trust": "DESIGN_BLOCKED",
    "security_substrate_authorizes_genesis": false
  },
  "mandatory_redteam": {
    "AccountAuthority_self_authorizes_without_independent_root": "FAIL",
    "reservation_owner_implies_genuine_account": "FAIL",
    "root_issuer_implies_account_id_mint_without_protocol": "FAIL",
    "security_substrate_owner_becomes_domain_authority": "FAIL",
    "FirstRunBootstrapAuthority_becomes_AccountAuthority": "FAIL",
    "two_authorities_independently_commit_same_genesis": "FAIL",
    "COMMITTED_with_independent_RESERVED_is_published": "FAIL",
    "caller_arbitrary_account_id": "FAIL",
    "account_scoped_identity_as_pre_account_root": "FAIL"
  },
  "result": {
    "primary_result": "ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_DESIGN_BLOCKED",
    "root_of_trust_solved": false,
    "formal_advancement": "WITHHELD",
    "reason": "Current evidence freezes safety separations but not the required authority owners or protocol."
  },
  "implementation_allowed": {
    "CryptoHunterAccountAuthority": false,
    "AccountIdReservationAuthority": false,
    "WorkspaceAuthority": false,
    "FullFillAuthority": false,
    "M0.8": false
  },
  "preserved_status": {
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "AccountIdReservationAuthority": "NOT_AVAILABLE",
    "WorkspaceAuthority": "NOT_AVAILABLE",
    "FullFillAuthority": "NOT_AVAILABLE",
    "production M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "root_of_trust": "DESIGN_BLOCKED",
    "reservation_owner": "DESIGN_BLOCKED",
    "account_id_mint_owner": "NOT_FROZEN",
    "genesis_decision_authority": "NOT_FOUND"
  }
}
```
