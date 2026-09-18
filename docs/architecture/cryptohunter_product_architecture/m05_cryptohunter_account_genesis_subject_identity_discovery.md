# M0.5 CryptoHunterAccount genesis subject identity discovery

This file is a deterministic complete projection of `m05_cryptohunter_account_genesis_subject_identity_discovery.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_CRYPTOHUNTER_ACCOUNT_GENESIS_SUBJECT_IDENTITY_DISCOVERY",
  "iteration": "DISCOVERY_ONLY",
  "repository_head_examined": "95e798f2e40a5b54ea0529f184078debe6b701d7",
  "provenance": {
    "reviewed_head_supplied": "50f733578cf270e334fed248f7ee9d42d388b058",
    "reviewed_head_available_locally": false,
    "classification": "UNKNOWN",
    "basis": [
      "m05_cryptohunter_account_genesis_authority_model.json",
      "m05_cryptohunter_account_root_of_trust_reconciliation.json",
      "process_topology_and_lifecycle.json",
      "canonical_domain_vocabulary.json",
      "identity_device_authentication_and_secrets.json",
      "bot_core/runtime/first_run_bootstrap.py"
    ],
    "rule": "The supplied SHA is unavailable locally and its relationship to the inspected current tree is UNKNOWN. Current-tree findings are discovery evidence only and cannot advance formal status or create authority.",
    "current_tree_evidence": true,
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_advancement_allowed": false,
    "relationship_to_reviewed_sha": "UNKNOWN",
    "current_local_tree_inspected": "YES",
    "path_content_comparison_evidence": "NOT_PERFORMED"
  },
  "subject_requirements": [
    "exists_before_CryptoHunterAccount_genesis",
    "independent_from_account_id",
    "stable_across_retries",
    "stable_across_devices",
    "stable_across_process_restart",
    "authority_bound_non_caller_mintable",
    "production_test_domain_semantics_known",
    "distinguishes_same_from_different_subjects",
    "durably_referenceable_in_genesis_provenance"
  ],
  "semantic_search_scope": {
    "searched_concepts": [
      "subject/principal/customer/tenant/subscriber",
      "subscription/license/entitlement/billing",
      "owner/deployment/installation/enrollment",
      "issuer/external identity/provisioning membership"
    ],
    "scope": "entire repository, including canonical architecture, runtime, persistence, packaging, licensing, tests and UI",
    "method": "semantic and identifier search; matches were evaluated as authority, entitlement, device, attempt context, or subject"
  },
  "candidate_matrix": [
    {
      "candidate": "intended_operator_id",
      "independent_of_account": false,
      "pre_account": false,
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": false,
      "durable": true,
      "result": "NOT_VIABLE_ACCOUNT_CHILD",
      "rejection_reason": "OperatorIdentity parent is CryptoHunterAccount; using it to create that parent is circular, and reverse uniqueness is absent.",
      "evidence": {
        "source_file": "canonical_domain_vocabulary.json",
        "source_pointer": "entity_kinds[canonical_name=OperatorIdentity]",
        "field_schema": "operator_id / intended_operator_id",
        "authority_owner": "NOT_FOUND for pre-account mint; external boundary only accepts a claim containing it",
        "parent_scope": "CryptoHunterAccount",
        "lifecycle": "persistent account child",
        "pre_account_availability": "NOT_PROVEN; genuine identity requires account parent",
        "durability": "OperatorIdentity is persistent after parent exists; intended value is claim content",
        "authenticity": "claim membership binds value but does not prove independent operator genesis",
        "stability": "reverse uniqueness and cross-account/cross-device stability NOT_SPECIFIED"
      }
    },
    {
      "candidate": "provisioning_context_fingerprint_sha256",
      "independent_of_account": "UNKNOWN",
      "pre_account": "UNKNOWN",
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": "CONDITIONAL_MEMBERSHIP_BINDING",
      "durable": "UNKNOWN",
      "result": "UNKNOWN",
      "rejection_reason": "The exact fingerprinted object, constructor, namespace, retry/device stability, and embedded fields are not specified.",
      "evidence": {
        "source_file": "process_topology_and_lifecycle.json",
        "source_pointer": "first_run_bootstrap_authority_contract.claim_schema and acceptance_authority.registry_binding",
        "field_schema": "64-lowercase-hex provisioning_context_fingerprint_sha256",
        "authority_owner": "external_product_provisioning_boundary membership binds the value; constructor/semantic object absent",
        "parent_scope": "provisioning attempt/context, not a declared subject namespace",
        "lifecycle": "ephemeral protected handoff",
        "pre_account_availability": "present in accepted handoff, but subject meaning NOT_DEFINED",
        "durability": "binding may be durably carried; stable semantic referent NOT_PROVEN",
        "authenticity": "membership authenticity is stronger than public hash, but hash alone is integrity only",
        "stability": "retry, restart, cross-device, nonce/challenge/account/device inclusion all NOT_SPECIFIED"
      }
    },
    {
      "candidate": "claim_fingerprint_sha256",
      "independent_of_account": false,
      "pre_account": true,
      "stable_across_retry": false,
      "stable_across_device": false,
      "authority_bound": false,
      "durable": true,
      "result": "ATTEMPT_CONTEXT_FINGERPRINT_ONLY",
      "rejection_reason": "It fingerprints claim content including account_id, device_installation_id, timestamps, challenge, generation/revision, operator and provisioning context.",
      "evidence": {
        "source_file": "bot_core/runtime/first_run_bootstrap.py",
        "source_pointer": "claim_content_fingerprint",
        "field_schema": "SHA-256 of all FirstRunBootstrapClaim fields except itself",
        "authority_owner": "external membership accepts exact claim; public hash is not authority",
        "parent_scope": "one complete claim/attempt",
        "lifecycle": "ephemeral key; durable_bootstrap_id=false",
        "pre_account_availability": "yes as claim content, but not independent",
        "durability": "consumption history can retain it",
        "authenticity": "registry membership required",
        "stability": "changes with account_id, device_installation_id, operator, generation, revision, times, challenge, or context"
      }
    },
    {
      "candidate": "device_installation_id",
      "independent_of_account": false,
      "pre_account": false,
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": false,
      "authority_bound": "UPSTREAM_VALUE_ONLY",
      "durable": true,
      "result": "NOT_VIABLE_ACCOUNT_CHILD",
      "rejection_reason": "DeviceInstallation parent is CryptoHunterAccount and identity varies by device.",
      "evidence": {
        "source_file": "canonical_domain_vocabulary.json",
        "source_pointer": "entity_kinds[canonical_name=DeviceInstallation]",
        "authority_owner": "NOT_FOUND",
        "lifecycle": "persistent product entity",
        "durability": true,
        "authenticity": "NOT_APPLICABLE",
        "field_schema": "device_installation_id",
        "parent_scope": "CryptoHunterAccount",
        "pre_account_availability": "value may be carried before genesis; genuine entity cannot pre-exist parent",
        "stability": "device-specific"
      }
    },
    {
      "candidate": "account_id",
      "independent_of_account": false,
      "pre_account": false,
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": "MINT_OWNER_NOT_FROZEN",
      "durable": true,
      "result": "CIRCULAR_IDENTITY_BEING_CREATED",
      "rejection_reason": "It is the identity whose genuine genesis is under decision, not an independent subject.",
      "evidence": {
        "source_file": "m05_cryptohunter_account_genesis_authority_model.json",
        "source_pointer": "account_id_state_machine",
        "field_schema": "account_id",
        "authority_owner": "NOT_FOUND",
        "parent_scope": "root entity",
        "lifecycle": "genuine only at genesis commit",
        "pre_account_availability": "candidate syntax/value only",
        "durability": "genuine commit required",
        "authenticity": "candidate is not authority",
        "stability": "does not correlate acct_A with acct_B"
      }
    },
    {
      "candidate": "external provisioning subject candidate",
      "independent_of_account": "UNKNOWN",
      "pre_account": "UNKNOWN",
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": "UNKNOWN",
      "durable": "UNKNOWN",
      "result": "PROVENANCE_STOPS_UPSTREAM",
      "rejection_reason": "The issuer exists conceptually, but no issuer-side subject field, namespace, schema, ownership rule, or durable mapping is exposed.",
      "evidence": {
        "source_file": "m05_cryptohunter_account_root_of_trust_reconciliation.json",
        "source_pointer": "m03_existing_authorities.external_product_provisioning_boundary",
        "field_schema": "NO_SUBJECT_FIELD",
        "authority_owner": "external_product_provisioning_boundary",
        "parent_scope": "external / absent",
        "lifecycle": "NOT_SPECIFIED",
        "pre_account_availability": "required but NOT_PROVEN",
        "durability": "NOT_PROVEN",
        "authenticity": "upstream validator not implemented locally",
        "stability": "NOT_SPECIFIED"
      }
    },
    {
      "candidate": "license/subscription candidate",
      "independent_of_account": "UNKNOWN",
      "pre_account": "UNKNOWN",
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": "LICENSE_ISSUERS_EXIST_OUTSIDE_GENESIS_CONTRACT",
      "durable": "VARIES",
      "result": "NOT_A_CANONICAL_SUBJECT",
      "rejection_reason": "Repository licensing represents access/feature/device entitlements; no lifetime one-license-per-subject identity or subject mapping exists.",
      "evidence": {
        "source_file": "bot_core/security/license.py and deploy/packaging/build_pyinstaller_bundle.py",
        "source_pointer": "LicenseValidationResult/license_id and encrypted license store",
        "field_schema": "license_id, fingerprint, validity/features",
        "authority_owner": "license signing/validation mechanisms",
        "parent_scope": "entitlement/device fingerprint",
        "lifecycle": "validity and replacement/rotation capable",
        "pre_account_availability": "possible entitlement, not canonical account subject",
        "durability": "license stores/registries exist",
        "authenticity": "signed/HMAC mechanisms exist",
        "stability": "not proven across renewals, replacements or devices"
      }
    },
    {
      "candidate": "deployment/install owner candidate",
      "independent_of_account": "UNKNOWN",
      "pre_account": "UNKNOWN",
      "stable_across_retry": "UNKNOWN",
      "stable_across_device": "UNKNOWN",
      "authority_bound": "UNKNOWN",
      "durable": "UNKNOWN",
      "result": "NO_SUBJECT_CONCEPT_FOUND",
      "rejection_reason": "No independent deployment owner, installation owner, or machine-enrollment subject contract connects installation to account genesis.",
      "evidence": {
        "source_file": "repository semantic search; canonical_domain_vocabulary.json",
        "source_pointer": "no canonical owner entity; DeviceInstallation is the nearest entity",
        "field_schema": "NO_FIELD",
        "authority_owner": "NOT_FOUND",
        "parent_scope": "DeviceInstallation is account child",
        "lifecycle": "NOT_DEFINED",
        "pre_account_availability": "NOT_FOUND",
        "durability": "NOT_FOUND",
        "authenticity": "NOT_FOUND",
        "stability": "NOT_FOUND"
      }
    }
  ],
  "intended_operator_analysis": {
    "classification": "NOT_VIABLE_ACCOUNT_CHILD",
    "mint_owner": "NOT_FOUND before account genesis",
    "may_exist_before_account": "NOT_PROVEN as genuine OperatorIdentity",
    "genuine_pre_account_authority": false,
    "same_operator_multiple_accounts": "NOT_SPECIFIED",
    "account_scoped": true,
    "parent_requires_existing_account": true,
    "reverse_uniqueness": "NOT_SPECIFIED",
    "circularity_test": "FAIL: op_A requires genuine acct_A parent, so op_A cannot authorize acct_A genesis"
  },
  "provisioning_context_analysis": {
    "classification": "UNKNOWN",
    "exact_semantic_object": "NOT_DEFINED",
    "constructor": "NOT_FOUND",
    "authenticator": "accepted ProvisioningMembershipBinding owned contractually by external_product_provisioning_boundary; public SHA alone is not authenticity",
    "stable_across_retries": "NOT_PROVEN",
    "stable_across_devices": "NOT_PROVEN",
    "changes_per_attempt": "NOT_SPECIFIED",
    "same_customer_same_value": "NOT_SPECIFIED",
    "embedded_account_device_nonce_challenge": "NOT_SPECIFIED",
    "conclusion": "Cannot be accepted as STABLE_SUBJECT_FINGERPRINT."
  },
  "external_provisioning_subject_analysis": {
    "classification": "EXTERNAL_SUBJECT_REQUIRED_BUT_SCHEMA_ABSENT",
    "issuer": "external_product_provisioning_boundary",
    "issuer_is_subject": false,
    "subject_namespace": "NOT_FOUND",
    "issuer_side_subject_key": "NOT_FOUND",
    "independence_and_stability": "NOT_PROVEN",
    "upstream_dependency_status": "BLOCKED_UPSTREAM",
    "conditional_semantics": "IF the selected account-genesis model requires a stable external subject identity, the current upstream provisioning contract does not provide one.",
    "unconditional_requirement_asserted": false
  },
  "license_subscription_analysis": {
    "classification": "ENTITLEMENT_NOT_SUBJECT",
    "findings": [
      "license_id identifies a license/entitlement",
      "machine fingerprint can bind an installation",
      "validity/features and registries concern access"
    ],
    "lifetime_subject_rule": "NOT_FOUND",
    "rotation_attack": "license_1 -> license_2 would split subject S; no canonical mapping prevents this"
  },
  "deployment_owner_analysis": {
    "classification": "NO_SUBJECT_CONCEPT_FOUND",
    "deployment_identity": "NOT_FOUND",
    "installation_owner": "NOT_FOUND",
    "machine_enrollment_subject": "NOT_FOUND",
    "DeviceInstallation": "NOT_VIABLE_ACCOUNT_CHILD"
  },
  "cross_device_stability": {
    "result": "NOT_PROVEN",
    "reason": "No candidate is defined to remain equal for subject S across dev_1 and dev_2."
  },
  "retry_stability": {
    "result": "NOT_PROVEN",
    "reason": "No authority-issued subject key is required to remain equal across retry/crash; arbitrary caller identifiers are denied."
  },
  "authority_provenance": {
    "result": "PROVENANCE_STOPS_UPSTREAM",
    "authority_source_is_subject": false,
    "public_hash_is_authority": false,
    "caller_minted_subject_allowed": false,
    "detail": "Membership proves acceptance by an issuer contract, not the identity of the customer/subject."
  },
  "same_subject_duplicate_attack": {
    "scenario": "same S: claim_1 -> acct_A; claim_2 -> acct_B",
    "classification": "DOMAIN_SEMANTICS_NOT_FROZEN",
    "if_ONE_SUBJECT_ONE_ACCOUNT": "acct_A/acct_B = deterministic conflict/duplicate; stable subject identity becomes mandatory",
    "if_ONE_SUBJECT_MANY_ACCOUNTS": "acct_A and acct_B may both be genuine; no false duplicate rejection",
    "if_no_external_subject_model": "comparison may be irrelevant",
    "automatic_security_violation": false
  },
  "distinct_subject_attack": {
    "scenario": "S1 -> acct_A; S2 -> acct_B",
    "subject_distinction": "UNAVAILABLE",
    "classification": "CONSEQUENCE_DEPENDS_ON_SELECTED_MODEL",
    "account_genesis_impossible_inferred": false,
    "global_singleton_serialization_allowed": false
  },
  "cross_artifact_parity": {
    "OperatorIdentity_parent": "CryptoHunterAccount",
    "DeviceInstallation_parent": "CryptoHunterAccount",
    "formal_contract_migration_identified": false,
    "FirstRunBootstrapAuthority_scope": "INITIAL_SECURITY_ESTABLISHMENT_ONLY",
    "root_reconciliation": "ROOT_PROOF_INSUFFICIENT_SEMANTICS",
    "genesis_model": "ACCOUNT_GENESIS_MODEL_DESIGN_BLOCKED",
    "contradiction": false
  },
  "result": {
    "primary_result": "ACCOUNT_GENESIS_SUBJECT_IDENTITY_INSUFFICIENT_SEMANTICS",
    "canonical_subject_identity": "NOT_FOUND",
    "subject_account_cardinality": "NOT_FROZEN",
    "upstream_subject_schema": "ABSENT",
    "external_subject_schema_status": "EXTERNAL_SUBJECT_REQUIRED_BUT_SCHEMA_ABSENT",
    "upstream_dependency_status": "CONDITIONAL_BLOCKER_IF_SUBJECT_IDENTITY_REQUIRED",
    "reason": "Canonical subject-to-account cardinality, account uniqueness key, and whether external subject identity participates in genesis are NOT_FROZEN.",
    "result_scope": "CURRENT_TREE_ONLY",
    "formal_status_advancement": "WITHHELD"
  },
  "impact_on_genesis_model": {
    "per_subject_serialization_required": "NOT_FROZEN",
    "per_subject_serialization_design": "NOT_APPLICABLE_UNTIL_CARDINALITY_SELECTED",
    "logical_account_genesis_idempotency_basis": "NOT_FROZEN",
    "idempotency_candidates": [
      "account_id reservation identity",
      "external provisioning decision ID",
      "authority-issued genesis request identity",
      "external subject identity",
      "atomic issuer decision"
    ],
    "same_subject_acct_A_acct_B": "DOMAIN_SEMANTICS_NOT_FROZEN",
    "missing_external_subject_identity_alone_blocks_genesis": false,
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE"
  },
  "implementation_allowed": {
    "CryptoHunterAccountAuthority": false,
    "ProvisioningBoundary": false,
    "WorkspaceAuthority": false,
    "InstrumentAuthority": false,
    "WCP Authority": false,
    "FullFillAuthority": false,
    "M0.8": false
  },
  "preserved_status": {
    "M0.12": "ACCEPTED / AVAILABLE",
    "M0.7 structural Full Fill v2": "ACCEPTED / STRUCTURAL AVAILABLE",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "WorkspaceAuthority": "NOT_AVAILABLE",
    "InstrumentAuthority": "NOT_AVAILABLE",
    "WCP Authority": "NOT_AVAILABLE / DESIGN_BLOCKED",
    "FullFillAuthority": "NOT_AVAILABLE",
    "production M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "mandatory_redteam_mutations": {
    "intended_operator_without_pre_account_independence": "FAIL",
    "device_installation_as_subject": "FAIL",
    "claim_SHA_unique_therefore_subject": "FAIL",
    "provisioning_context_without_stability_semantics": "FAIL",
    "authority_source_as_subject": "FAIL",
    "license_ID_as_lifetime_subject": "FAIL",
    "caller_arbitrary_subject_becomes_genuine": "FAIL",
    "different_subjects_globally_serialized": "FAIL",
    "one_subject_one_account_without_canonical_evidence": "FAIL",
    "same_subject_acct_A_acct_B_security_conflict_while_cardinality_unresolved": "FAIL",
    "missing_subject_ID_alone_makes_genesis_impossible": "FAIL",
    "global_singleton_serialization_for_all_subjects": "FAIL",
    "one_to_many_rejects_legitimate_second_account": "FAIL"
  },
  "subject_account_cardinality": {
    "canonical_external_subject_exists": "NOT_FOUND",
    "subject_to_account_cardinality": "NOT_FROZEN",
    "one_subject_one_account": "NOT_PROVEN",
    "multiple_accounts_per_subject_allowed": "NOT_PROVEN",
    "multiple_subjects_per_account_allowed": "NOT_PROVEN",
    "account_uniqueness_key": "NOT_FROZEN",
    "vocabulary_invariant": "CryptoHunterAccount purpose root SaaS/customer account does not prove one human/customer/organization may own exactly one CryptoHunterAccount",
    "missing_subject_identity_alone_makes_genesis_impossible": false
  },
  "candidate_domain_models": {
    "A_ONE_EXTERNAL_SUBJECT_ONE_CRYPTOHUNTER_ACCOUNT": {
      "status": "NOT_SELECTED / NOT_PROVEN",
      "consequence": "stable external subject identity required; acct_A/acct_B is deterministic conflict/duplicate"
    },
    "B_ONE_EXTERNAL_SUBJECT_MANY_CRYPTOHUNTER_ACCOUNTS": {
      "status": "NOT_SELECTED / NOT_PROVEN",
      "consequence": "acct_A and acct_B may both be genuine; subject identity cannot be the account uniqueness key alone"
    },
    "C_ACCOUNT_ID_IS_THE_ONLY_ACCOUNT_UNIQUENESS_KEY": {
      "status": "NOT_SELECTED / NOT_PROVEN",
      "consequence": "subject comparison is irrelevant to account uniqueness; authoritative account_id mint/reservation remains separately unresolved"
    },
    "D_EXTERNAL_PROVISIONING_DECISION_DEFINES_ACCOUNT_GENESIS_IDEMPOTENCY_WITHOUT_CUSTOMER_WIDE_UNIQUENESS": {
      "status": "NOT_SELECTED / NOT_PROVEN",
      "consequence": "requires a stable authority-issued provisioning decision identity, which is not currently exposed"
    },
    "E_OTHER_CANONICALLY_SUPPORTED_MODEL": {
      "status": "NOT_FOUND",
      "consequence": "no other canonical model found"
    },
    "F_DESIGN_BLOCKED": {
      "status": "CURRENT",
      "consequence": "no model may be selected without canonical cardinality, uniqueness and authority evidence"
    },
    "selected_model": "NOT_FROZEN"
  },
  "subject_cardinality_parity": {
    "discovery_subject_to_account_cardinality": "NOT_FROZEN",
    "genesis_model_subject_to_account_cardinality": "NOT_FROZEN",
    "canonical_vocabulary_subject_to_account_cardinality": "NOT_DEFINED",
    "one_to_one_inferred_from_root_SaaS_customer_account": false,
    "parity": "PASS"
  }
}
```
