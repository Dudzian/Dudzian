# M0.5 CryptoHunterAccount authority discovery

This file is a deterministic complete projection of `m05_cryptohunter_account_authority_discovery.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_CRYPTOHUNTER_ACCOUNT_AUTHORITY_DISCOVERY",
  "repository_head_supplied": "87dd8dca0368cdeb5f26ce65b1d72eeb06abcbfc",
  "repository_head_examined": "365ab30fc5e6cd5d5c162bef410b1ed73c21c3a3",
  "provenance": {
    "classification": "MISMATCH",
    "supplied_commit_available": false,
    "mode": "CURRENT_TREE_DISCOVERY",
    "formal_authority_status_minted": false,
    "reason": "The supplied object is absent from the local object database; findings describe the checked-out current tree and do not mint ACCEPTED or AVAILABLE."
  },
  "canonical_entity": {
    "status": "FOUND",
    "canonical_name": "CryptoHunterAccount",
    "purpose": "root SaaS/customer account",
    "parent": "none",
    "persistence": true,
    "saas_sync_candidate": true,
    "secret_policy": "must not contain secrets",
    "direct_relationships": {
      "DeviceInstallation": "one_to_many",
      "OperatorIdentity": "one_to_many",
      "Workspace": "one_to_many"
    },
    "not_proven_owned_or_scoped": [
      "Portfolio",
      "ExchangeAccount",
      "billing",
      "license",
      "subscription",
      "secrets"
    ],
    "sources": [
      "canonical_domain_vocabulary.json#/entity_kinds/0",
      "canonical_domain_vocabulary.json#/relationships"
    ]
  },
  "canonical_id_contract": {
    "status": "PARTIALLY_FROZEN",
    "field": "account_id",
    "prefix": "acct_",
    "grammar": "acct_<uuidv7>",
    "uuid_version": 7,
    "immutable": true,
    "scope": "top-level/root; no parent",
    "persistent": true,
    "syntax_grants_authority": false,
    "evidence_note": "The vocabulary freezes persistent ID syntax and ID immutability, not admission or genuineness."
  },
  "production_authority": {
    "status": "NOT_FOUND",
    "module": null,
    "class_or_function": null,
    "admission_api": null,
    "resolver_api": null,
    "durable_store": null,
    "history_model": null,
    "semantic_search": {
      "roots": [
        "bot_core",
        "tests/architecture",
        "docs/audits",
        "docs/architecture/cryptohunter_product_architecture"
      ],
      "concepts": [
        "CryptoHunterAccount",
        "account registry/store/journal/repository/resolver",
        "accepted account projection",
        "account admission service",
        "identity/tenant authority",
        "account lifecycle owner",
        "tenant/account relationships and lifecycle repositories"
      ],
      "nearby_non_authorities": [
        "bot_core.persistence.state_store validates and carries scoped PersistenceRecord values but StateStoreSnapshot explicitly carries no authority",
        "PersistentEntityIdentityProjection is a minimal backup/recovery representation of a pre-existing identity",
        "license/subscription parsing is not CryptoHunterAccount admission or resolution",
        "ExchangeAccount lifecycle and identity contracts concern a different entity"
      ]
    },
    "production_admission_authority": "NOT_FOUND",
    "production_genuine_account_resolver": "NOT_FOUND",
    "production_genuine_record_writer": {
      "status": "NOT_FOUND",
      "reason": "Generic PersistenceRecord construction and SQLiteStateStore mutation can technically carry a structurally valid projection, but no production upstream admission authority proof identifies a genuine account source/writer.",
      "generic_store_api_is_genuine_writer": false
    }
  },
  "accepted_record_schema": {
    "status": "NOT_FROZEN",
    "immutable_fields": [],
    "projection_found": {
      "name": "PersistentEntityIdentityProjection",
      "fields": [
        "entity_kind",
        "entity_id",
        "parent_scope_bindings"
      ],
      "for_account_parent_scope_bindings": {},
      "is_authority": false,
      "is_accepted_record": false
    },
    "reason": "The identity projection is a backup carrier and no genuine accepted-account record is defined."
  },
  "admission": {
    "status": "NOT_FROZEN",
    "who_may_create": "NOT_FOUND",
    "creation_command": "NOT_FOUND",
    "accepted_event": "NOT_FOUND",
    "issuer_admin_or_root_actor": "NOT_FOUND",
    "authentication_proof": "NOT_FOUND",
    "idempotency_identity": "NOT_FOUND",
    "caller_supplied_id_is_admission": false
  },
  "trusted_actor": {
    "status": "NOT_FOUND",
    "local_product_installation_authority": "NOT_FROZEN",
    "licensed_user_identity": "NOT_FROZEN",
    "deployment_owner": "NOT_FROZEN",
    "administrative_root": "NOT_FROZEN",
    "canonical_parent_entity": "NONE_DECLARED",
    "self_bootstrapping_authority_allowed": false
  },
  "lifecycle": {
    "status": "NOT_FROZEN",
    "states": {
      "ACTIVE": "NOT_DEFINED",
      "CLOSED": "NOT_DEFINED",
      "REVOKED": "NOT_DEFINED",
      "DELETED": "NOT_DEFINED",
      "SUSPENDED": "NOT_DEFINED",
      "SUPERSEDED": "NOT_DEFINED"
    },
    "exchange_or_workspace_lifecycle_inference_allowed": false
  },
  "current_resolver": {
    "status": "NOT_FOUND",
    "raw_mapping_is_resolver": false,
    "identity_projection_is_resolver": false,
    "kind": "production genuine account resolver"
  },
  "historical_resolver": {
    "status": "NOT_FOUND",
    "workspace_original_parent_after_account_change": "NOT_FROZEN",
    "gap": "No account history/resolution or later-lifecycle effect on historical Workspace resolution is specified.",
    "kind": "production genuine historical account resolver"
  },
  "workspace_parent_binding": {
    "canonical_relationship": "CryptoHunterAccount -> Workspace one_to_many",
    "required_proof": "Workspace.parent CryptoHunterAccount ID must resolve through a genuine CryptoHunterAccount authority",
    "caller_parent_id_is_proof": false,
    "PersistentEntityIdentityProjection_is_proof": false,
    "raw_account_mapping_is_proof": false,
    "binding_immutability": "UNRESOLVED",
    "immutability_gap": "Canonical vocabulary freezes persistent IDs but does not explicitly freeze immutability of the Workspace-to-CryptoHunterAccount parent edge."
  },
  "durability": {
    "current_record_carrier": {
      "status": "FOUND",
      "implementation": "bot_core.persistence.state_store.SQLiteStateStore",
      "table": "state_store_current_records",
      "representation": "CryptoHunterAccount current record",
      "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
      "carrier_strategy": "PERSISTENCE_RECORD",
      "representation_schema": "PersistentEntityIdentityProjection",
      "durability_class": "DURABLE AUTHORITATIVE CURRENT STATE",
      "grants_authority": false,
      "adds_new_domain_facts": false,
      "restorable_authority": false,
      "interpretation": "Durability classification of a current-state carrier; it is not CryptoHunterAccount admission authority."
    },
    "generic_M011_persistence_tables": {
      "status": "FOUND",
      "tables": [
        "state_store_metadata",
        "state_store_current_records",
        "state_store_immutable_history",
        "state_store_transaction_descriptors"
      ]
    },
    "CryptoHunterAccount_specific_authority_tables": {
      "status": "NOT_FOUND"
    },
    "account_authority_store": {
      "status": "NOT_FOUND"
    },
    "account_authority_journal": {
      "status": "NOT_FOUND"
    },
    "account_authority_history": {
      "status": "NOT_FOUND",
      "current_representation_is_history": false
    },
    "account_authority_head": {
      "status": "NOT_FOUND"
    },
    "backup_projection": "FOUND_NON_AUTHORITY",
    "restart_semantics": "NOT_FROZEN"
  },
  "authentication": {
    "classification": "NONE",
    "public_sha_only_is_authenticity": false,
    "backup_authentication_is_account_authority_authentication": false,
    "future_environment_separation": {
      "separate_auth_domains": "NOT_FROZEN",
      "separate_keys": "NOT_FROZEN",
      "separate_durable_stores": "NOT_FROZEN"
    }
  },
  "rollback": {
    "rollback_protection": "NOT_FROZEN",
    "coherent_prefix_restore_detected": "NOT_FROZEN",
    "authority_implementation_permission": false
  },
  "restart": {
    "status": "NOT_FROZEN",
    "revalidation": "NOT_FROZEN",
    "fail_closed_rules": "NOT_FROZEN"
  },
  "concurrency": {
    "status": "NOT_FROZEN",
    "serialized_transition": "NOT_FROZEN",
    "cas_or_generation": "NOT_FROZEN",
    "idempotency_key": "NOT_FROZEN",
    "duplicate_request_semantics": "NOT_FROZEN",
    "conflicting_duplicate_semantics": "NOT_FROZEN",
    "state_store_generation_is_account_admission_concurrency": false
  },
  "self_mint": {
    "scenario": "valid account ID syntax + valid projection shape + valid public fingerprint",
    "result": "NO",
    "current_result_basis": "NO_CURRENT_AUTHORITY_PATH",
    "future_self_mint_resistance": "UNPROVEN",
    "caller_selected_account_id_grants_accepted_identity": false
  },
  "contract_completeness": {
    "primary_result": "INSUFFICIENT_SEMANTICS",
    "status": "INCOMPLETE",
    "upstream_dependency_status": "NO_UPSTREAM_BLOCKER_IDENTIFIED",
    "intrinsic_gaps_remain": true
  },
  "intrinsic_blockers": [
    "accepted account record schema",
    "trusted admission actor/root of trust",
    "creation command and accepted event",
    "authenticated admission proof and idempotency",
    "lifecycle and current/historical resolution",
    "authority durability/history/restart",
    "rollback protection and freshness",
    "concurrency and duplicate semantics",
    "Workspace parent-edge immutability",
    "explicit reuse-or-separation decision for the existing M0.11 current-record carrier"
  ],
  "upstream_blockers": [],
  "implementation_allowed": false,
  "impact_on_workspace": {
    "dependency": "CryptoHunterAccount genuine resolver -> Workspace admission parent binding",
    "workspace_status": "NOT_AVAILABLE / INSUFFICIENT_SEMANTICS + BLOCKED_UPSTREAM on CryptoHunterAccount authority/resolver",
    "account_available_implies_workspace_implementation_allowed": false,
    "reason": "Workspace retains independent intrinsic semantic gaps."
  },
  "next_stage": "Freeze the CryptoHunterAccount root-of-trust, accepted record/admission, lifecycle/resolvers, authenticated durable history, restart, rollback/freshness, and concurrency contracts before any authority implementation.",
  "preserved_status": {
    "M0.12_1.46.0": "ACCEPTED",
    "CatalogRuntimeAcceptanceAuthority": "ACCEPTED / AVAILABLE",
    "M0.7_structural_Full_Fill_v2": "ACCEPTED / AVAILABLE_FOR_STRUCTURAL_VALIDATION",
    "WorkspaceAuthority": "NOT_AVAILABLE / INSUFFICIENT_SEMANTICS",
    "InstrumentAuthority": "NOT_AVAILABLE / INSUFFICIENT_SEMANTICS",
    "WorkspaceCatalogProjectionAuthority": "NOT_AVAILABLE / DESIGN_BLOCKED",
    "M0.7_FullFillAuthority": "NOT_AVAILABLE",
    "production_M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "authority_laundering_denials": {
    "state_store_is_CryptoHunterAccountAuthority": false,
    "PersistenceRecord_is_account_authority": false,
    "PersistentEntityIdentityProjection_is_account_authority": false,
    "restored_record_mints_account_authority": false
  },
  "restore_semantics": {
    "StateStoreSnapshot_carries_authority": false,
    "restore_can_recreate_genuine_account_authority_by_itself": false,
    "required_basis": "Revalidation of an existing upstream fact under the M0.11 contract; backup/restore is not genesis or admission.",
    "upstream_genuine_fact_source": "NOT_FOUND"
  },
  "duplicate_store_policy": {
    "future_design_requirement": "MUST inventory and reuse, or explicitly separate, M0.11 carrier semantics before introducing any second durable current-account store.",
    "silently_create_second_current_account_persistence_source": false,
    "status": "REQUIRED_BEFORE_AUTHORITY_DESIGN"
  }
}
```
