# M0.5 — executable PRODUCTION_LOCAL authentication gate

> Deterministyczna projekcja `m05_account_genesis_freshness_authority_production_local_authentication_gate.json`. JSON jest źródłem prawdy.

```json
{
  "artifact": "M05_ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_PRODUCTION_LOCAL_AUTHENTICATION_GATE",
  "schema_version": 1,
  "decision": "IMPLEMENTATION_AUTHORIZATION_GATE_PASSED",
  "canonical_model": "ISOLATED_SEMANTIC_VERIFIER_WITH_DB_AUTHENTICATED_ONE_TIME_PREPARATION",
  "authentication_binding": {
    "transport": "local PostgreSQL Unix-domain socket only",
    "method": "peer with an explicit pg_ident map",
    "hba_order": [
      "local all postgres peer map=freshness_admin_map (offline cluster administration; superuser is outside the claimed threat boundary)",
      "local <authority_database> freshness_crypto_verifier peer map=freshness_authority_map",
      "local <authority_database> freshness_runtime peer map=freshness_authority_map",
      "local <authority_database> all reject",
      "local all all reject"
    ],
    "identity_map": {
      "os_freshness_crypto_verifier": "freshness_crypto_verifier",
      "os_freshness_runtime": "freshness_runtime"
    },
    "network_listeners": false,
    "caller_supplied_identity_allowed": false,
    "caller_supplied_role_token_header_or_GUC_allowed": false,
    "alternate_HBA_rule_allowed": false,
    "boundary": "the kernel-supplied peer operating-system identity, not SET ROLE denial"
  },
  "roles": {
    "freshness_schema_owner": {
      "login": false,
      "inherit": false,
      "purpose": "offline schema ownership"
    },
    "freshness_function_owner": {
      "login": false,
      "inherit": false,
      "purpose": "reviewed SECURITY DEFINER mutation function ownership"
    },
    "freshness_admin": {
      "login": false,
      "inherit": false,
      "purpose": "offline provisioning and lifecycle administration through reviewed boundaries; no post-provisioning raw authority DML"
    },
    "freshness_crypto_verifier": {
      "login": true,
      "inherit": false,
      "direct_execute": [
        "prepare_verified_freshness_candidate"
      ],
      "raw_authority_DML": false
    },
    "freshness_runtime": {
      "login": true,
      "inherit": false,
      "direct_execute": [
        "compare_and_advance",
        "reviewed reads"
      ],
      "raw_authority_DML": false
    },
    "freshness_reader": {
      "login": false,
      "inherit": false,
      "purpose": "reviewed read authority only"
    }
  },
  "database_qualification": {
    "minimum_postgresql_major": 16,
    "fsync": "on",
    "synchronous_commit": "on",
    "schema_owner": "freshness_schema_owner",
    "mutation_function_owner": "freshness_function_owner",
    "owners_must_have_distinct_OIDs": true,
    "unexpected_memberships": 0,
    "inherited_authority": false,
    "set_role_paths_between_authority_roles": false,
    "public_execute": false,
    "execute_grant_option": false,
    "fixed_function_search_path": "pg_catalog",
    "security_definer_required": true
  },
  "scope_limits": {
    "protects_against_database_superuser": false,
    "protects_against_host_root": false,
    "protects_against_coordinated_host_rollback": false
  },
  "preserved_invariants": {
    "retained_PostgreSQL_lifecycle_history_is_sole_ACTIVE_eligibility_authority": true,
    "preparation_freezes_ACTIVE": false,
    "proposer_identity_exact_binding": true,
    "one_time_preparation": true,
    "receipt_atomic_with_accepted_decision": true,
    "FreshnessFinalizationReceipt_V1_unchanged": true,
    "OriginalDecisionIdentity_is_separate": true,
    "key_version_is_lifecycle_generation": false,
    "cross_role_key_separation_uses_exact_raw_32_byte_Ed25519_public_key_material": true,
    "material_alias_dimensions_cannot_bypass_separation": [
      "credential_id",
      "key_id",
      "key_version",
      "semantic role label",
      "stored material identity label"
    ]
  },
  "final_flags": {
    "FreshnessAuthority_implementation_allowed_after_iteration": true,
    "FreshnessAuthority_implemented": false,
    "production_substrate_implemented": false,
    "PRODUCTION_LOCAL_RUNTIME_AVAILABLE": false,
    "ROOT_PROOF_ISSUER_IMPLEMENTED": false
  }
}
```
