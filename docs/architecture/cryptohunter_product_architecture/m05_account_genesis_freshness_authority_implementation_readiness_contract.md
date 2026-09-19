# M0.5 AccountGenesis FreshnessAuthority production substrate / implementation readiness

Ten plik jest deterministyczną, kompletną projekcją `m05_account_genesis_freshness_authority_implementation_readiness_contract.json`. JSON jest źródłem prawdy.

```json
{
  "artifact": "M05_ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_IMPLEMENTATION_READINESS_CONTRACT",
  "iteration": "DESIGN / RECONCILIATION ONLY",
  "readiness_result": "FRESHNESS_AUTHORITY_IMPLEMENTATION_READINESS_CAN_BE_FROZEN",
  "implementation_scope": {
    "FreshnessAuthority_implemented": false,
    "CryptoHunterAccountAuthority_implemented": false,
    "concrete_production_provider_selected": false,
    "FreshnessAuthority_implementation_allowed_after_iteration": false,
    "CHA_implementation_allowed_after_iteration": false,
    "reason": "design contract is frozen, but production CAS/history/custody and dedicated local authority adapter implementations do not exist"
  },
  "provenance": {
    "classification": "UNKNOWN",
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_project_advancement": "WITHHELD",
    "reviewed_SHA_supplied": "NOT_SUPPLIED",
    "design_freeze_is_formal_advancement": false
  },
  "frozen_inputs": {
    "freshness_model": "F_HYBRID_CAS_PLUS_SIGNED_FINALIZATION_RECEIPT",
    "update_unit": "FULL_AUTHORITATIVE_DOCUMENT",
    "winner_rule": "EXACTLY_ONE_SUCCESSFUL_N_TO_N_PLUS_1",
    "predecessor_binding": [
      "exact generation",
      "exact document digest",
      "exact complete head set"
    ],
    "accepted_equals_proposed": true,
    "finalization_request_id_bound": true,
    "physical_protocol": "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT",
    "CAS_loser": "MODEL_A_ATOMIC_REPREPARE",
    "projection_is_authority": false,
    "Catalog_authority_or_custody_reuse": false,
    "TEST_PRODUCTION_domains_distinct": true
  },
  "actual_current_capability": {
    "production_CAS_backend": "NOT_AVAILABLE",
    "authoritative_history_backend": "NOT_AVAILABLE",
    "durable_receipt_store": "NOT_AVAILABLE",
    "dedicated_local_authority_adapter": "NOT_AVAILABLE",
    "freshness_or_proposer_custody_provider": "NOT_SELECTED",
    "evidence": [
      {
        "path": "bot_core/security/keyring_storage.py",
        "finding": "ordinary keyring get/set/rotation; no authoritative CAS semantics"
      },
      {
        "path": "bot_core/persistence/state_store.py",
        "finding": "M0.11 SQLite projection store, not AccountGenesis authority"
      },
      {
        "path": "bot_core/instruments/source_producer_membership.py",
        "finding": "domain-specific local SQLite journal; not reusable as freshness authority capability"
      },
      {
        "path": "bot_core/instruments/catalog_admission_receipt.py",
        "finding": "Catalog-specific receipt authority/custody is forbidden for reuse"
      },
      {
        "path": "pyproject.toml",
        "finding": "cryptography and PyNaCl are available; no production remote CAS provider dependency establishes required semantics"
      }
    ]
  },
  "production_CAS_backend_contract": {
    "interface_operations": [
      "compare_and_swap_full_document",
      "read_authoritative_current",
      "lookup_decision",
      "lookup_original_decision_identity",
      "prove_immediate_successor",
      "verify_successor_chain",
      "prove_constructive_non_acceptance",
      "recover_finalization_receipt",
      "read_key_lifecycle_history"
    ],
    "CAS_preconditions": [
      "exact predecessor generation",
      "exact predecessor digest",
      "exact complete predecessor head set",
      "exact authority domain and authority_id"
    ],
    "atomic_effect": "one indivisible durable commit of accepted full document, decision identity, receipt, successor link and lifecycle reference",
    "winner_cardinality": "exactly one winner per exact predecessor",
    "correctness": [
      "linearizable or equivalently single-copy atomic",
      "multi-process",
      "crash/restart durable",
      "authoritative reread"
    ],
    "forbidden_substitutes": [
      "read-then-write",
      "process mutex",
      "timestamp check",
      "ordinary keyring set()",
      "last-write-wins",
      "rename without CAS",
      "local pre-write digest comparison",
      "caller optimistic assumption"
    ]
  },
  "retained_authoritative_history": {
    "status": "SEMANTICS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE",
    "owner": "FreshnessAuthority",
    "properties": [
      "durable",
      "authenticated",
      "authority-controlled",
      "append-only monotonic or equivalently rollback-protected",
      "cryptographically linked to currently trusted authority root/state"
    ],
    "lookups": [
      "exact positive historical acceptance",
      "OriginalDecisionIdentity",
      "exact immediate successor for old predecessor",
      "successor chain",
      "constructive authoritative non-acceptance proof",
      "historical receipt",
      "key lifecycle history",
      "superseded candidate conflict"
    ],
    "negative_lookup_is_non_acceptance_proof": false,
    "current_head_alone_sufficient": false,
    "constructive_non_acceptance": "authenticated proof identifies the accepted immediate successor (or authenticated bounded authority state proving no decision), binds queried predecessor/candidate/request and chains to current trusted authority state; mere absence/error is never proof"
  },
  "finalization_receipt_durability": {
    "authority_owner": "FreshnessAuthority",
    "storage": "same authoritative retained decision history as the CAS decision",
    "commit_rule": "receipt record and exact acceptance decision are one atomic durable authority transaction",
    "lost_response_recovery": "lookup by authenticated OriginalDecisionIdentity and finalization_request_id, then authoritative reread and chain verification",
    "exact_binding": [
      "authority domain",
      "authority_id",
      "predecessor generation/digest/complete heads",
      "accepted full-document digest",
      "finalization_request_id",
      "decision sequence/id",
      "freshness key_id/version"
    ],
    "history_confirmation": "receipt decision link and successor chain must verify to current trusted authority state",
    "rotation": "VERIFY_ONLY keys verify historical receipts; ACTIVE signs new decisions",
    "revocation": "REVOKED cannot sign new decisions; historical verification requires independent pre-compromise evidence and authenticated lifecycle history; revocation alone neither erases nor validates history",
    "memory_only_allowed": false
  },
  "canonical_representation": {
    "status": "FROZEN_NOW",
    "serialization": "RFC 8785 JSON Canonicalization Scheme (JCS), restricted to the contract schema",
    "text_encoding": "UTF-8, no BOM",
    "strings": "JSON strings; Unicode code points preserved exactly; no application Unicode normalization; schema values requiring normalization are rejected before serialization",
    "bytes": "base64url without padding",
    "integers": "JSON safe integers only, decimal JCS form, schema bounds; floating point values forbidden in cryptographic preimages",
    "object_order": "JCS lexicographic property ordering",
    "array_order": "semantic and preserved; set-like fields must be schema-sorted before JCS",
    "null_absence": "null forbidden unless a field schema explicitly declares null; absent and null are distinct; required fields may not be omitted",
    "domain_separator": "ASCII literal encoded as UTF-8, followed by one 0x00 byte, then canonical payload bytes; separator is versioned and object-type specific",
    "cross_process_rule": "identical validated semantics MUST yield byte-identical preimage and digest; noncanonical input rejected",
    "invalid_Unicode_or_lone_surrogate": "REJECT before canonicalization in accordance with JCS/I-JSON constraints",
    "duplicate_object_keys": "REJECT during parsing before canonicalization",
    "set_like_array_rule": "each set-like field has a schema-declared comparator; sort by ascending unsigned lexicographic order of each element canonical UTF-8 byte string; equal canonical elements are duplicates and REJECT; arrays not declared set-like preserve semantic order",
    "base64url_validation": "RFC 4648 URL-safe alphabet without padding; decode then re-encode equality required; padding and noncanonical forms REJECT",
    "digest_hex_validation": "exactly 64 lowercase hexadecimal ASCII characters; uppercase and mixed case REJECT",
    "floats_in_cryptographic_preimages": "REJECT"
  },
  "cryptographic_profile": {
    "digest": {
      "algorithm": "SHA-256",
      "representation": "lowercase hexadecimal, exactly 64 ASCII characters",
      "algorithm_identifier": "sha256-v1",
      "algorithm_agility": "new version requires explicit migration; no caller choice"
    },
    "authentication": {
      "receipt": "Ed25519",
      "proposer": "Ed25519",
      "signature_representation": "base64url without padding over exactly 64 signature bytes",
      "key_id_version_binding": "authenticated inside signed preimage",
      "provider_selection": "NOT_FROZEN / NOT_SELECTED",
      "verifier_selection": "only authenticated lifecycle lineage; aliases, headers and callers cannot select provider/key",
      "authoritative_document": "Ed25519",
      "authoritative_document_signing_role": "FreshnessAuthority authority/finalization signing key role; same role as receipt, never CHA proposer role"
    }
  },
  "key_custody": {
    "freshness_boundary": "independently administered non-exportable signing capability preferred; exact provider NOT_FROZEN",
    "separate_roles": [
      "CHA proposer key",
      "FreshnessAuthority finalization key",
      "storage master key",
      "Catalog authority keys"
    ],
    "role_reuse_forbidden": true,
    "caller_selected_authority_key": false,
    "lifecycle": [
      "ACTIVE may sign and verify",
      "VERIFY_ONLY may verify history but not sign",
      "REVOKED may not sign or ordinarily establish trust; historical acceptance needs independent pre-compromise proof"
    ],
    "requirements": [
      "authenticated key_id and version",
      "rotation lineage",
      "compromise revocation",
      "historical verification",
      "no unauthenticated alias/provider selection",
      "disjoint PRODUCTION and TEST roots, keys, namespace and state"
    ]
  },
  "proposer_trust_lineage": {
    "status": "FROZEN_NOW / IMPLEMENTATION_NOT_AVAILABLE",
    "bootstrap_owner": "offline Product Security provisioning authority, independent of candidate and created account",
    "trust_root_storage": "pinned PRODUCTION/TEST-specific root in authenticated deployment configuration plus authority-controlled monotonic lineage journal",
    "first_identity": "installed only by an authenticated provisioning ceremony before candidate admission",
    "rotation": "old ACTIVE provisioning authority signs successor key/version and effective boundary; append to monotonic lineage before use",
    "VERIFY_ONLY_history": true,
    "REVOKED_history": "cannot authenticate new candidates; historical candidates require independent pre-compromise acceptance/receipt evidence and authenticated lifecycle state",
    "TOFU_allowed": false,
    "candidate_self_provision_allowed": false,
    "candidate_embedded_key_is_trust": false
  },
  "failure_model": {
    "CAS_ACCEPTED": "verified receipt plus authoritative reread matching exact candidate",
    "ALREADY_ACCEPTED_EXACT": "historical exact decision and receipt recovered; idempotent success only after verification",
    "CAS_CONFLICT": "different immediate successor proven; invoke durable Model A conflict classification/REPREPARE",
    "PREDECESSOR_MISMATCH": "no send/retry until authoritative reread and reconciliation",
    "UNAVAILABLE": "retry/reconcile; never publish",
    "TIMEOUT_AFTER_SEND": "OUTCOME_UNKNOWN",
    "TRANSPORT_ERROR_AFTER_SEND": "OUTCOME_UNKNOWN unless proven not sent",
    "MALFORMED_AUTHORITY_RESPONSE": "OUTCOME_UNKNOWN and fail closed",
    "UNVERIFIABLE_RECEIPT": "OUTCOME_UNKNOWN and fail closed",
    "STALE_KEY": "refresh authenticated lifecycle and repropose only if no acceptance",
    "REVOKED_KEY": "reject new transition; reconcile possible historical decision independently",
    "HISTORY_UNAVAILABLE": "OUTCOME_UNKNOWN / fail closed",
    "FORK_OR_TAMPER": "security failure, quarantine and operator escalation",
    "outcome_unknown_rule": "never FAILED or SUCCESS without authoritative reconciliation"
  },
  "local_authority_adapter": {
    "status": "REQUIREMENTS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE",
    "owner": "CryptoHunterAccountAuthority dedicated transactional storage adapter",
    "atomic_transitions": [
      "INITIAL_BINDING",
      "PREPARED",
      "REPREPARE supersession",
      "FINAL_COMMIT"
    ],
    "requirements": [
      "durable uniqueness",
      "expected-state/version fences",
      "storage-level cross-process serialization",
      "crash/restart atomicity",
      "retained local authority history",
      "idempotent exact replay",
      "fail-closed fork/tamper detection"
    ],
    "M011_SQLiteStateStore_role": "PROJECTION_ONLY / NOT_AUTHORITY",
    "SQLite_future_option": "possible only as a new dedicated adapter proven against this contract; existing M0.11 store is not promoted",
    "process_local_lock_sufficient": false
  },
  "implementation_readiness_matrix": [
    {
      "capability": "authoritative full-document CAS",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "build/select conforming production backend"
    },
    {
      "capability": "authoritative reread",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "backend must expose authenticated linearizable decision reread"
    },
    {
      "capability": "retained decision history",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement authenticated rollback-resistant authority journal"
    },
    {
      "capability": "immediate-successor historical proof",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement proof lookup and chain verification"
    },
    {
      "capability": "durable finalization receipts",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "persist receipts atomically with the accepted CAS decision"
    },
    {
      "capability": "canonical serialization",
      "current_status": "semantic preimages frozen; concrete representation previously NOT_FROZEN",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement RFC 8785 JCS profile and conformance vectors"
    },
    {
      "capability": "digest algorithm",
      "current_status": "NOT_FROZEN",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement SHA-256 profile"
    },
    {
      "capability": "receipt auth algorithm/provider",
      "current_status": "algorithm NOT_FROZEN; provider NOT_SELECTED",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES_ALGORITHM_AND_BOUNDARY_PROVIDER_NOT_SELECTED",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement Ed25519 and select production custody provider"
    },
    {
      "capability": "proposer auth algorithm/provider",
      "current_status": "algorithm NOT_FROZEN; provider NOT_SELECTED",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES_ALGORITHM_AND_BOUNDARY_PROVIDER_NOT_SELECTED",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement Ed25519 and select production custody provider"
    },
    {
      "capability": "freshness key custody",
      "current_status": "generic keyring exists but is not authority custody",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES_CAPABILITY_BOUNDARY_PROVIDER_NOT_SELECTED",
      "implementation_available": "NO",
      "blocker_or_next_action": "select independently administered production provider"
    },
    {
      "capability": "freshness key lifecycle storage",
      "current_status": "NOT_AVAILABLE",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement authenticated monotonic lifecycle journal"
    },
    {
      "capability": "proposer trust lineage",
      "current_status": "C_PROPOSER_KEY_LINEAGE_NOT_YET_FROZEN",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "implement offline-provisioned signed lineage registry"
    },
    {
      "capability": "local transactional AccountGenesis adapter",
      "current_status": "NOT_AVAILABLE; M0.11 store is projection only",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "build dedicated authority adapter; SQLite may be evaluated separately"
    },
    {
      "capability": "multi-process fencing",
      "current_status": "no conforming AccountGenesis authority primitive",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "backend and adapter need storage-level transactions/fences"
    },
    {
      "capability": "TEST/PRODUCTION separation",
      "current_status": "semantics already frozen; implementation absent",
      "must_be_frozen_before_implementation": "YES",
      "frozen_now": "YES",
      "implementation_available": "NO",
      "blocker_or_next_action": "provision disjoint roots, keys, namespaces and state"
    }
  ],
  "remaining_prerequisites": [
    "select/build and qualify production CAS plus retained-history backend",
    "select production custody providers and provision separated roots",
    "implement canonical codec and cryptographic conformance vectors",
    "implement FreshnessAuthority against the frozen interface",
    "implement dedicated local AccountGenesis authority adapter",
    "implement proposer lineage registry and provisioning ceremony",
    "complete independent root-proof issuer prerequisite before CHA implementation"
  ],
  "downstream_status": {
    "WorkspaceAuthority": "UNCHANGED / NOT_ADVANCED",
    "FullFillAuthority": "UNCHANGED / NOT_ADVANCED",
    "M0.8": "UNCHANGED / NOT_ADVANCED",
    "production_M0.5": "BLOCKED_ON_IMPLEMENTATIONS_AND_ROOT_PROOF",
    "design_freeze": "READY",
    "formal_project_advancement": "WITHHELD"
  },
  "redteam_mutations": [
    "read_compare_write_accepted_as_CAS",
    "process_mutex_accepted_as_global_CAS",
    "two_successful_successors",
    "current_head_only_sufficient_history",
    "negative_receipt_lookup_proves_non_acceptance",
    "receipt_only_in_process_memory",
    "receipt_not_bound_to_exact_CAS_decision",
    "noncanonical_serialization_accepted",
    "processes_hash_same_semantics_differently",
    "unauthenticated_key_alias_selects_verifier",
    "proposer_key_reused_as_freshness_key",
    "Catalog_key_reused",
    "TEST_key_or_state_accepted_in_PRODUCTION",
    "TOFU_proposer_admitted",
    "candidate_self_provisions_proposer_trust",
    "revoked_key_finalizes_new_transition",
    "M011_projection_promoted_to_authority",
    "local_adapter_process_lock_only",
    "timeout_treated_as_failed",
    "timeout_treated_as_success"
  ],
  "domain_separation_profile": {
    "profile_version": "account-genesis-freshness-crypto-profile-v1",
    "preimage_format": "ASCII(exact_domain_literal) || 0x00 || canonical_bytes",
    "exact_literals": {
      "freshness_document_digest": "cryptohunter.account-genesis.freshness-document-digest.v1",
      "authoritative_document_authentication": "cryptohunter.account-genesis.freshness-document-authentication.v1",
      "cha_proposer_authentication": "cryptohunter.account-genesis.cha-proposer-authentication.v1",
      "complete_semantic_head_set_digest": "cryptohunter.account-genesis.complete-semantic-head-set-digest.v1",
      "finalization_receipt_authentication": "cryptohunter.account-genesis.freshness-finalization-receipt-authentication.v1",
      "original_decision_identity": "cryptohunter.account-genesis.original-decision-identity.v1"
    },
    "all_literals_distinct": true,
    "environment_binding": "TEST/PRODUCTION remains inside canonical authenticated payload",
    "immutability": "literal immutable for profile v1; change requires explicit new protocol/profile version",
    "runtime_or_caller_configurable": false
  },
  "cross_artifact_redteam_mutations": [
    "readiness_SHA256_freshness_NOT_FROZEN",
    "readiness_Ed25519_proposer_freshness_NOT_FROZEN",
    "readiness_Ed25519_receipt_freshness_NOT_FROZEN",
    "readiness_JCS_freshness_different_encoding",
    "document_digest_domain_changed",
    "proposer_domain_reused_as_receipt_domain",
    "semantic_head_domain_reused_as_document_domain",
    "empty_domain_separator",
    "unversioned_domain_separator",
    "caller_configurable_domain_separator",
    "authoritative_document_auth_unspecified",
    "proposer_key_reused_for_authoritative_document",
    "invalid_unicode_lone_surrogate_accepted",
    "padded_base64url_accepted",
    "uppercase_digest_accepted",
    "duplicate_JSON_keys_accepted",
    "set_like_array_order_implementation_dependent"
  ],
  "cross_artifact_status_redteam_mutations": [
    "physical_frozen_freshness_says_F_DESIGN_BLOCKED",
    "physical_frozen_freshness_unblocked_false",
    "freshness_frozen_physical_not_frozen",
    "freshness_reason_claims_canonical_crypto_NOT_FROZEN",
    "freshness_reason_claims_physical_design_blockers_remain",
    "implementation_available_inferred_from_physical_design_freeze",
    "production_M05_available_from_physical_design_freeze"
  ]
}
```
