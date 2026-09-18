# M0.5 Account genesis root-proof admission binding contract

This file is a deterministic complete projection of `m05_account_genesis_root_proof_admission_binding_contract.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_ACCOUNT_GENESIS_ROOT_PROOF_ADMISSION_BINDING_CONTRACT",
  "iteration": "DESIGN_RECONCILIATION_ONLY",
  "repository_head_examined": "09f6e33f113652d3c3b8ddf072ef0e38aa739a43",
  "reviewed_head_supplied": "74355d07f6fde26e26126c2130422ea4a71917b8",
  "provenance": {
    "source": "GIT",
    "reviewed_head_available_locally": false,
    "classification": "UNKNOWN",
    "relationship_to_reviewed_sha": "UNKNOWN",
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_advancement_allowed": false,
    "basis": [
      "bot_core/runtime/first_run_bootstrap.py",
      "m05_account_genesis_authority_topology_and_ownership.json",
      "m05_cryptohunter_account_root_of_trust_reconciliation.json",
      "m05_cryptohunter_account_genesis_authority_model.json",
      "m05_cryptohunter_account_genesis_subject_identity_discovery.json",
      "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json"
    ],
    "rule": "Unavailable reviewed SHA is not equated with the current tree. This artifact freezes safety constraints and records unresolved semantics; it creates no production authority."
  },
  "frozen_inputs": {
    "independent_accepted_first_account_root": "REQUIRED",
    "self_authorization_using_local_evidence": "FORBIDDEN",
    "non_circular_root": "FROZEN",
    "root_proof_issuer": "NOT_FOUND / conditional external provisioning issuer",
    "root_proof_validator": "DESIGN_BLOCKED",
    "root_proof_consumption_semantics": "NOT_FROZEN",
    "genesis_final_decision_owner": "NOT_FOUND",
    "subject_cardinality": "NOT_FROZEN",
    "proof_id_is_operation_id": false,
    "issuance_is_validation": false,
    "issuance_implies_account_id_mint": false,
    "proof_acceptance_implies_reservation": false,
    "proof_acceptance_implies_COMMITTED": false
  },
  "existing_candidate_inventory": {
    "candidate": "M03_EXTERNAL_PRODUCT_PROVISIONING_MEMBERSHIP",
    "claim_schema": "FirstRunBootstrapClaim",
    "membership_schema": "ProvisioningMembershipBinding",
    "authority_source_exact": "external_product_provisioning_boundary",
    "rows": [
      {
        "field": "account_id",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "none",
        "account_binding": "YES",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "exact candidate account_id; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "device_installation_id",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "none",
        "account_binding": "INDIRECT_ONLY",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "first-device child identity; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "intended_operator_id",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "none",
        "account_binding": "INDIRECT_ONLY",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "intended account-scoped operator; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "bootstrap_generation",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "generation exists only for bootstrap",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "positive bootstrap generation; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "bootstrap_revision",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "revision exists only for bootstrap",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "positive bootstrap revision; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "issued_at_utc",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "not-before input",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "bootstrap issuance time; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "expires_at_utc",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "bootstrap expiry input",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "bootstrap expiry time; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "challenge_fingerprint_sha256",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "challenge may vary; no genesis policy",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "NOT_EXPLICIT",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "public challenge content fingerprint; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "provisioning_context_fingerprint_sha256",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "none stated",
        "account_binding": "NO",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "OPAQUE_FINGERPRINT_ONLY",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "opaque provisioning-context fingerprint; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "claim_fingerprint_sha256",
        "source": "FirstRunBootstrapClaim",
        "owner": "external_product_provisioning_boundary supplies accepted claim",
        "authenticated": "ONLY_WHEN exact claim resolves through separately accepted ProvisioningMembershipBinding; public SHA alone is not authenticity",
        "freshness_semantics": "none",
        "account_binding": "INDIRECT_CONTENT_BINDING",
        "operation_binding": "NO: no AccountGenesis logical operation identity or canonical genesis request",
        "environment_binding": "INDIRECT_ONLY",
        "historical_retention": "M0.3 consumed history retains only a subset; original accepted claim/membership durable history is NOT_PROVEN",
        "usable_for_AccountGenesis": "CONDITIONAL_ONLY",
        "gap": "canonical public SHA-256 of all other claim fields; authentication, genesis scope, operation binding, environment semantics, and durable provenance are incomplete"
      },
      {
        "field": "claim_fingerprint_sha256",
        "source": "ProvisioningMembershipBinding",
        "owner": "external_product_provisioning_boundary (abstract port; production owner NOT_FOUND)",
        "authenticated": "NOMINAL accepted membership lookup only; production signature/MAC/authenticated registry NOT_FOUND",
        "freshness_semantics": "No lifecycle, expiry, revocation, monotonic issuer state, or stale-after semantics in binding",
        "account_binding": "INDIRECT_ONLY",
        "operation_binding": "NO",
        "environment_binding": "NOT_EXPLICIT; authority_source is not a trust-domain/environment field",
        "historical_retention": "production durable authenticated membership history NOT_FOUND",
        "usable_for_AccountGenesis": "NO_AS_STANDALONE_ROOT_PROOF",
        "gap": "references exact accepted claim; nominal shape/literal/hash cannot establish issuer authority"
      },
      {
        "field": "complete_claim_content_fingerprint_sha256",
        "source": "ProvisioningMembershipBinding",
        "owner": "external_product_provisioning_boundary (abstract port; production owner NOT_FOUND)",
        "authenticated": "NOMINAL accepted membership lookup only; production signature/MAC/authenticated registry NOT_FOUND",
        "freshness_semantics": "No lifecycle, expiry, revocation, monotonic issuer state, or stale-after semantics in binding",
        "account_binding": "INDIRECT_ONLY",
        "operation_binding": "NO",
        "environment_binding": "NOT_EXPLICIT; authority_source is not a trust-domain/environment field",
        "historical_retention": "production durable authenticated membership history NOT_FOUND",
        "usable_for_AccountGenesis": "NO_AS_STANDALONE_ROOT_PROOF",
        "gap": "recomputed complete claim content SHA-256; nominal shape/literal/hash cannot establish issuer authority"
      },
      {
        "field": "authority_source",
        "source": "ProvisioningMembershipBinding",
        "owner": "external_product_provisioning_boundary (abstract port; production owner NOT_FOUND)",
        "authenticated": "NOMINAL accepted membership lookup only; production signature/MAC/authenticated registry NOT_FOUND",
        "freshness_semantics": "No lifecycle, expiry, revocation, monotonic issuer state, or stale-after semantics in binding",
        "account_binding": "NO",
        "operation_binding": "NO",
        "environment_binding": "NOT_EXPLICIT; authority_source is not a trust-domain/environment field",
        "historical_retention": "production durable authenticated membership history NOT_FOUND",
        "usable_for_AccountGenesis": "NO_AS_STANDALONE_ROOT_PROOF",
        "gap": "literal external_product_provisioning_boundary; nominal shape/literal/hash cannot establish issuer authority"
      },
      {
        "field": "provisioning_context_fingerprint_sha256",
        "source": "ProvisioningMembershipBinding",
        "owner": "external_product_provisioning_boundary (abstract port; production owner NOT_FOUND)",
        "authenticated": "NOMINAL accepted membership lookup only; production signature/MAC/authenticated registry NOT_FOUND",
        "freshness_semantics": "No lifecycle, expiry, revocation, monotonic issuer state, or stale-after semantics in binding",
        "account_binding": "INDIRECT_ONLY",
        "operation_binding": "NO",
        "environment_binding": "NOT_EXPLICIT; authority_source is not a trust-domain/environment field",
        "historical_retention": "production durable authenticated membership history NOT_FOUND",
        "usable_for_AccountGenesis": "NO_AS_STANDALONE_ROOT_PROOF",
        "gap": "must equal claim context fingerprint; nominal shape/literal/hash cannot establish issuer authority"
      }
    ],
    "assessment": "VIABLE_ONLY_AFTER_ADDITIONAL_AUTHORITY; insufficient as current AccountGenesis root proof"
  },
  "proof_object_layers": {
    "raw_external_claim": {
      "object": "FirstRunBootstrapClaim supplied by caller",
      "authority": "NONE by shape/hash alone",
      "conversion": "must resolve byte-for-semantic exact accepted claim and membership"
    },
    "accepted_external_membership": {
      "object": "ProvisioningMembershipBinding plus separately resolved exact FirstRunBootstrapClaim",
      "authority": "conditional external_product_provisioning_boundary membership",
      "scope": "INITIAL_SECURITY_ESTABLISHMENT_ONLY consumer contract; not AccountGenesis evidence"
    },
    "root_proof_validation_receipt": {
      "object": "NOT_FOUND; future immutable receipt is useful but schema/name/owner are NOT_FROZEN",
      "must_not_be_fabricated_from": "public SHA, local HMAC, structural validity, or current membership fallback"
    },
    "AccountGenesis_admission_evidence": {
      "object": "NOT_FOUND",
      "requirement": "must retain exact accepted root-proof provenance and exact genesis binding/decision linkage independently of current issuer state"
    }
  },
  "candidate_models": {
    "A_EXISTING_ACCEPTED_PROVISIONING_MEMBERSHIP_AS_ROOT_PROOF": {
      "status": "INSUFFICIENT",
      "reason": "exact account binding exists, but issuer authentication, environment, genesis operation/request binding, revocation, replay and historical provenance are absent"
    },
    "B_ACCOUNT_GENESIS_SPECIFIC_ROOT_PROOF_DERIVED_FROM_EXTERNAL_MEMBERSHIP": {
      "status": "POSSIBLE_NOT_SELECTED",
      "requirements": "authenticated derivation chain, exact domain/bindings/freshness, non-circular issuer and historical resolution"
    },
    "C_EXTERNAL_ISSUER_ACCOUNT_GENESIS_PROOF": {
      "status": "POSSIBLE_NOT_SELECTED",
      "requirements": "trusted issuer contract, authentication, domain, binding, freshness, replay and history"
    },
    "D_VALIDATION_RECEIPT_MINTED_BY_FUTURE_ROOT_PROOF_VALIDATOR": {
      "status": "POSSIBLE_NOT_SELECTED",
      "requirements": "validator owner/trust roots, immutable receipt identity/schema and replay/history semantics"
    },
    "E_ATOMIC_ISSUER_ACCOUNT_PROTOCOL_PROOF": {
      "status": "POSSIBLE_NOT_SELECTED",
      "requirements": "frozen issuer/reservation/admission atomic protocol without collapsing authority roles"
    },
    "F_DESIGN_BLOCKED": {
      "status": "SELECTED",
      "reason": "no candidate has a complete evidenced contract and validator owner remains DESIGN_BLOCKED"
    }
  },
  "selected_or_blocked_model": {
    "selection": "F_DESIGN_BLOCKED",
    "existing_candidate_disposition": "M03 candidate is conditional input evidence, not a sufficient AccountGenesis root proof",
    "intrinsic_blockers": [
      "missing AccountGenesis operation/request binding model",
      "consumption/replay/cardinality semantics NOT_FROZEN",
      "validator and final decision owner NOT_FOUND/DESIGN_BLOCKED",
      "account-id mint/reservation handoff protocol NOT_FROZEN"
    ],
    "upstream_blockers": [
      "production ProvisioningBoundary/authenticated issuer NOT_AVAILABLE",
      "issuer trust-domain keys/registry NOT_FOUND",
      "membership lifecycle/revocation and anti-rollback anchor NOT_FOUND"
    ]
  },
  "binding_requirements": {
    "rule": "An accepted future proof MUST bind every semantically required fact directly or through one authenticated, historically resolvable derivation chain. Fields below are requirements, not invented DTO fields.",
    "requirements": {
      "issuer_identity": "REQUIRED / field model NOT_FROZEN",
      "issuer_authority_domain": "REQUIRED / NOT_FOUND",
      "proof_identity": "REQUIRED / claim fingerprint is content identity only",
      "proof_generation_revision": "CONDITIONAL; M0.3 bootstrap values exist but AccountGenesis meaning NOT_FROZEN",
      "environment_trust_domain": "REQUIRED / NOT_AVAILABLE",
      "candidate_account_id_or_reservation_identity": "REQUIRED before commit / binding mechanism NOT_FROZEN",
      "logical_operation_or_equivalent": "REQUIRED for retry convergence / identity NOT_FOUND",
      "canonical_request_semantics_fingerprint": "REQUIRED unless selected authenticated equivalent completely binds semantics / NOT_FOUND",
      "provisioning_root_provenance": "REQUIRED / PARTIAL",
      "issued_accepted_time": "CONDITIONAL / M0.3 issued time exists, acceptance time absent",
      "freshness_expiry": "REQUIRED policy / AccountGenesis semantics NOT_FROZEN"
    }
  },
  "account_binding": {
    "current_candidate": "FirstRunBootstrapClaim directly binds exact account_id",
    "options": {
      "A_proof_then_authenticated_handoff_to_reservation": "POSSIBLE_NOT_SELECTED",
      "B_direct_exact_account_id": "CURRENT_CANDIDATE_HAS_FIELD_BUT_END_TO_END_AUTHORITY_INSUFFICIENT",
      "C_atomic_proof_reservation": "POSSIBLE_NOT_SELECTED",
      "D_NOT_FROZEN": "SELECTED"
    },
    "mismatch_rule": "If an accepted proof/derivation binds acct_A, use for acct_B MUST fail. Current candidate can detect content mismatch only after genuine membership authenticity exists.",
    "proof_accepted_is_account_id_minted": false,
    "account_id_minted_is_proof_accepted": false,
    "both_imply_account_genuine": false
  },
  "operation_binding": {
    "proof_id_is_operation_id": false,
    "current_candidate_binds_logical_operation": false,
    "current_candidate_binds_reservation": false,
    "current_candidate_binds_canonical_genesis_request": false,
    "future_requirement": "stable authority-issued logical operation identity or authenticated equivalent must make an exact retry converge on the same decision/result",
    "changed_semantics_rule": "MUST fail only after an operation/request-bound model is selected; current root proof is not operation-bound, so this is not claimed as current behavior",
    "status": "DESIGN_BLOCKED"
  },
  "environment_binding": {
    "requirement": "TEST root proof MUST NOT authorize PRODUCTION AccountGenesis",
    "current_candidate": "NOT_AVAILABLE; provisioning_context_fingerprint_sha256 is opaque and has no frozen environment/trust-domain semantics",
    "structural_or_literal_authority_source_is_sufficient": false,
    "disposition": "FAIL_CLOSED",
    "test_proof_authorizes_production": false
  },
  "authentication": {
    "current_source": "NOMINAL accepted membership authority through ProvisioningBoundary abstract port",
    "external_signature": "NOT_FOUND",
    "external_MAC": "NOT_FOUND",
    "authenticated_membership_registry": "NOT_FOUND",
    "future_validator_receipt": "NOT_FOUND",
    "public_SHA_is_authority": false,
    "local_AccountGenesis_HMAC_role": "may authenticate stored receipt bytes only; cannot establish external proof genuineness",
    "unknown_issuer": "DENY / FAIL_CLOSED"
  },
  "freshness": {
    "authentication_is_freshness": false,
    "generation_revision": "AVAILABLE_FOR_BOOTSTRAP_ONLY",
    "issued_expiry": "FirstRunBootstrapAuthority enforces issued_at_utc <= now_utc <= expires_at_utc for bootstrap consumption",
    "acceptance_timestamp": "NOT_FOUND",
    "stale_after": "NOT_FOUND",
    "monotonic_issuer_state": "NOT_FOUND",
    "AccountGenesis_commit_time_policy": "NOT_FROZEN"
  },
  "revocation": {
    "membership_lifecycle": "NOT_FOUND",
    "revocation_status_or_generation": "NOT_FOUND",
    "valid_T1_revoked_or_stale_T2": "NOT_FROZEN",
    "acceptance_snapshot_rule": "NOT_FROZEN",
    "disposition_when_unverifiable": "DENY / FAIL_CLOSED"
  },
  "validation_semantics": {
    "valid_root_proof_means": "Only the exact claims explicitly guaranteed by a future trusted issuer/validator contract, for its exact authority domain, authenticated content, bindings and validity point; that contract is not currently complete.",
    "does_not_prove": [
      "account_id is genuine",
      "account genesis COMMITTED",
      "business subject may own exactly one account",
      "reservation exists or was consumed",
      "logical operation is unique",
      "issuer owns account_id mint",
      "validation owner owns final genesis decision"
    ],
    "proof_verification_is_issuance": false,
    "proof_validated_is_account_committed": false
  },
  "validation_output": {
    "immutable_receipt_useful": true,
    "name_AcceptedAccountGenesisRootProof": "NOT_FROZEN",
    "owner": "DESIGN_BLOCKED",
    "minimum_candidate_contents": [
      "receipt identity and schema version",
      "validator identity/authority domain and decision",
      "exact issuer/proof identity and authenticated evidence reference",
      "environment/trust domain",
      "exact account_id or reservation handoff",
      "logical operation and canonical request binding or authenticated equivalent",
      "validation time and applicable freshness/revocation snapshot",
      "immutable provenance and predecessor/head binding"
    ],
    "warning": "A receipt can attest validation only; it cannot mint external issuer authority, account_id, reservation, or COMMITTED genesis."
  },
  "consumption_semantics": {
    "single_use": "NOT_FROZEN",
    "multi_use": "NOT_FROZEN",
    "account_bound": "admission NOT_FROZEN",
    "operation_bound": "NOT_FROZEN",
    "subject_bound": "NOT_FROZEN",
    "decision_conditions": {
      "single_use_valid_if": "future business/admission authority freezes one proof to at most one terminal logical genesis and durable atomic consumption/recovery",
      "multi_use_valid_if": "issuer scope and subject/cardinality policy explicitly allow reuse and each operation/account binding is authenticated and historically retained",
      "account_bound_valid_if": "proof or authenticated handoff binds exact account_id/reservation before commit",
      "operation_bound_valid_if": "proof or authenticated handoff binds stable logical operation plus canonical semantics",
      "subject_bound_valid_if": "canonical subject schema and subject/account cardinality are frozen"
    }
  },
  "replay": {
    "R_to_O1_acct_A_then_O2_acct_B": "DOMAIN_SEMANTICS_NOT_FROZEN",
    "one_external_R_validated_twice": "NOT_FROZEN: same receipt, conflict, or two receipts depends on validator identity/consumption contract",
    "same_proof_same_logical_operation_retry": "FROZEN REQUIREMENT: future contract must return/resolve the same terminal result and MUST NOT create a second account",
    "blocker": "stable logical operation identity and durable decision resolver are NOT_FOUND",
    "same_proof_changed_semantic_request": "CONDITIONAL: fail when selected proof/handoff is operation/request-bound; no current rule invented"
  },
  "historical_provenance": {
    "requirement": "Every future committed genesis MUST immutably identify and resolve the exact accepted root proof, issuer/authority domain, authenticated derivation/validation evidence, bindings and commit-time validity decision without current-state substitution.",
    "current_availability": "NOT_AVAILABLE",
    "current_issuer_state_is_historical_substitute": false,
    "stored_HMAC_is_external_genuineness": false
  },
  "restart": {
    "requirement": "Recover the exact accepted proof/receipt/provenance and same logical decision/result, or fail closed.",
    "caller_recreates_equivalent_proof": false,
    "silent_issuer_switch": false,
    "current_membership_substitutes_missing_history": false,
    "current_availability": "NOT_AVAILABLE"
  },
  "rollback": {
    "requirement": "Rollback of accepted proof/receipt/history MUST be detected by future AccountGenesis freshness closure or recovery MUST fail closed.",
    "separate_weaker_trust_path": false,
    "current_root_proof_anti_rollback": "NOT_AVAILABLE"
  },
  "authority_boundary": {
    "separations": [
      "root-proof issuance != root-proof validation",
      "root-proof validation != account_id mint",
      "account_id mint != reservation",
      "reservation != genesis final decision",
      "proof acceptance != reservation consumption"
    ],
    "issuer": "NOT_FOUND / conditional external provisioning issuer",
    "validator_options": {
      "A_future_CryptoHunterAccountAuthority_internal": "NOT_SELECTED",
      "B_separate_RootProofValidationAuthority": "NOT_SELECTED",
      "C_ProvisioningBoundary_verifies_own_evidence": "NOT_SELECTED",
      "D_external_issuer_returns_accepted_proof": "NOT_SELECTED",
      "E_DESIGN_BLOCKED": "SELECTED"
    },
    "validator_owner": "DESIGN_BLOCKED",
    "genesis_final_decision_owner": "NOT_FOUND",
    "non_circular_rule": "Root for acct_A MUST be independent of acct_A and every child/account-scoped authority whose legitimacy requires acct_A."
  },
  "failure_semantics": {
    "public_literals": "NOT_FROZEN; do not invent API error literals",
    "policy": "Every unverifiable critical proof state is DENY / FAIL_CLOSED; never fallback to local self-authorization.",
    "classes": {
      "proof malformed": "DENY / FAIL_CLOSED",
      "issuer unknown": "DENY / FAIL_CLOSED",
      "authentication invalid": "DENY / FAIL_CLOSED",
      "proof stale": "DENY / FAIL_CLOSED; exact public literal NOT_FROZEN",
      "proof revoked": "DENY / FAIL_CLOSED; lifecycle semantics NOT_FROZEN",
      "environment mismatch": "DENY / FAIL_CLOSED",
      "account binding mismatch": "DENY / FAIL_CLOSED when binding exists; missing required binding blocks admission",
      "operation binding mismatch": "DENY / FAIL_CLOSED when operation-bound model exists; otherwise contract DESIGN_BLOCKED",
      "proof already consumed": "DENY only if future single-use model selected; otherwise NOT_FROZEN",
      "root circularity detected": "DENY / FAIL_CLOSED"
    }
  },
  "root_proof_capability_matrix": {
    "independent pre-account provenance": {
      "status": "PARTIAL",
      "evidence": "port intent says pre-existing external authority, but independence and production establishment are unproven"
    },
    "issuer authenticity": {
      "status": "NOT_AVAILABLE",
      "evidence": "signature, MAC, authenticated issuer registry and trust roots are NOT_FOUND"
    },
    "proof uniqueness": {
      "status": "PARTIAL",
      "evidence": "claim fingerprint identifies content; it is not an authority-issued unique root-proof identity"
    },
    "account candidate binding": {
      "status": "AVAILABLE",
      "evidence": "accepted complete FirstRunBootstrapClaim binds exact account_id"
    },
    "operation binding": {
      "status": "NOT_AVAILABLE",
      "evidence": "no AccountGenesis logical operation identity; proof_id_is_operation_id=false"
    },
    "request-semantic binding": {
      "status": "NOT_AVAILABLE",
      "evidence": "no canonical AccountGenesis request/fingerprint"
    },
    "environment binding": {
      "status": "NOT_AVAILABLE",
      "evidence": "no explicit production/test trust domain; opaque context hash has no frozen semantics"
    },
    "freshness": {
      "status": "PARTIAL",
      "evidence": "issued_at/expires_at and generation/revision are enforced for bootstrap, not frozen for AccountGenesis"
    },
    "revocation": {
      "status": "NOT_AVAILABLE",
      "evidence": "no membership revocation/lifecycle semantics"
    },
    "replay semantics": {
      "status": "NOT_FROZEN",
      "evidence": "bootstrap consumption replay is fenced, AccountGenesis proof consumption is not"
    },
    "historical verification": {
      "status": "PARTIAL",
      "evidence": "consumed bootstrap subset survives, but exact accepted claim/membership and issuer authentication history do not"
    },
    "restart recovery": {
      "status": "PARTIAL",
      "evidence": "M0.11 restores consumed bootstrap state, not exact authenticated root-proof/admission provenance"
    },
    "anti-rollback": {
      "status": "NOT_AVAILABLE",
      "evidence": "no independent provisioning membership/root-proof freshness anchor"
    },
    "end-to-end genuine root-proof usability": {
      "status": "NOT_AVAILABLE",
      "evidence": "content-level account binding does not compensate for missing genuine issuer authenticity, environment and admission semantics"
    }
  },
  "cross_artifact_parity": {
    "parity": "PASS",
    "independent_root_required": true,
    "circular_root_forbidden": true,
    "proof_id_is_operation_id": false,
    "issuer_implies_account_id_mint_owner": false,
    "subject_cardinality": "NOT_FROZEN",
    "same_proof_replay_semantics": "NOT_FROZEN",
    "M03_FirstRunBootstrapAuthority_scope": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "M03_membership_equals_genesis_authority": false
  },
  "mandatory_redteam": {
    "locally_generated_AccountAuthority_proof_accepted": "FAIL",
    "OperatorIdentity_child_of_acct_A_proves_acct_A": "FAIL",
    "TEST_proof_used_in_PRODUCTION": "FAIL",
    "structurally_valid_unknown_issuer_accepted": "FAIL",
    "public_SHA_as_authenticity": "FAIL",
    "proof_bound_acct_A_used_for_acct_B": "FAIL_IF_ACCOUNT_BINDING_EXISTS / REQUIRED_FAIL_FOR_CURRENT_CLAIM_FIELD",
    "current_membership_substitutes_missing_historical_proof": "FAIL",
    "accepted_proof_implies_COMMITTED_account": "FAIL",
    "proof_id_automatically_becomes_logical_operation_id": "FAIL"
  },
  "result": {
    "primary_result": "ACCOUNT_GENESIS_ROOT_PROOF_INSUFFICIENT_SEMANTICS",
    "intrinsic_status": "DESIGN_BLOCKED",
    "upstream_status": "BLOCKED_UPSTREAM",
    "formal_advancement": "WITHHELD",
    "reason": "The M0.3 candidate binds exact claim content and account_id and has bootstrap-only timing/replay fences, but it lacks proven issuer authentication, environment, AccountGenesis operation/request/consumption policy, complete historical provenance, validator ownership and rollback closure."
  },
  "implementation_allowed": {
    "ProvisioningBoundary": false,
    "RootProofValidator": false,
    "CryptoHunterAccountAuthority": false,
    "AccountIdReservationAuthority": false,
    "WorkspaceAuthority": false,
    "FullFillAuthority": false,
    "M0.8": false
  },
  "preserved_status": {
    "independent accepted first-account root": "REQUIRED",
    "self-authorization using local evidence": "FORBIDDEN",
    "non-circular root": "FROZEN",
    "root-proof issuer": "NOT_FOUND / conditional external provisioning issuer",
    "root-proof validator": "DESIGN_BLOCKED",
    "root-proof consumption semantics": "NOT_FROZEN",
    "genesis final decision owner": "NOT_FOUND",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "WorkspaceAuthority": "NOT_AVAILABLE",
    "FullFillAuthority": "NOT_AVAILABLE",
    "production M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE"
  },
  "account_binding_security": {
    "content_binding": "AVAILABLE: FirstRunBootstrapClaim.account_id binds the exact candidate account_id in canonical claim content",
    "issuer_authenticity": "NOT_AVAILABLE",
    "end_to_end_genuine_root_proof": "NOT_AVAILABLE",
    "exact_bound_account_mismatch": "DENY / FAIL_CLOSED",
    "proof_bound_account_may_be_rebound": false,
    "content_bound_account_id_must_match_candidate": true
  },
  "non_circular_root": {
    "target_account": "acct_A",
    "created_account_itself_allowed": false,
    "account_scoped_operator_identity_allowed": false,
    "account_scoped_device_installation_allowed": false,
    "account_scoped_workspace_allowed": false,
    "account_scoped_authority_allowed": false,
    "legitimacy_may_depend_on_entity_requiring_target_account_to_exist": false
  },
  "m03_critical_behavior_evidence": {
    "source": "bot_core/runtime/first_run_bootstrap.py::FirstRunBootstrapAuthority.consume",
    "accepted_claim_equals_supplied_claim": "REQUIRED",
    "binding_claim_fingerprint_matches_claim": "REQUIRED",
    "binding_complete_claim_content_fingerprint_matches_recomputed_content": "REQUIRED",
    "binding_authority_source_matches_AUTHORITY_SOURCE": "REQUIRED",
    "binding_provisioning_context_matches_claim": "REQUIRED",
    "claim_account_device_operator_generation_revision_match_state": "REQUIRED",
    "issued_at_lte_now_lte_expires_at": "REQUIRED",
    "replay_consumption_denied": "REQUIRED",
    "purpose": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "cryptographic_issuer_authenticity_inferred": false
  }
}
```
