# M0.5 WCP authority state model design

This file is a deterministic complete projection of `m05_wcp_authority_state_model_design.json`. JSON is the source of truth.

```json
{
  "iteration": "CONTRACT_DESIGN_FREEZE_ONLY",
  "discovery_commit_supplied": "d33bc33d1678009b621c836479e359e47ec38740",
  "repository_head_examined": "07156448589022e024d0dfa3121894c1798309d6",
  "provenance": {
    "mode": "CURRENT_TREE_PLUS_PRIOR_DISCOVERY",
    "formal_authority_status_minted": false,
    "rule": "Heuristic/discovery evidence may classify or design but cannot mint ACCEPTED or AVAILABLE."
  },
  "preserved_status": {
    "M0.12_1.46.0": "ACCEPTED",
    "CatalogRuntimeAcceptanceAuthority": "ACCEPTED_AVAILABLE",
    "M0.7_structural_Full_Fill_v2": "ACCEPTED_AVAILABLE_FOR_STRUCTURAL_VALIDATION",
    "M0.7_FullFillAuthority": "NOT_AVAILABLE",
    "WorkspaceCatalogProjectionAuthority": "NOT_AVAILABLE",
    "production_M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "artifact": "M05_WCP_AUTHORITY_STATE_MODEL_DESIGN",
  "result": "DESIGN_BLOCKED",
  "implementation_allowed": false,
  "authority_boundary": {
    "owner": "future WorkspaceCatalogProjectionAuthority (not implemented)",
    "owns": [
      "workspace_catalog_projection_id minting",
      "projection admission",
      "projection generation/head serialization"
    ],
    "references": [
      "genuine Workspace",
      "genuine ASCAT",
      "exact authoritative Instrument versions",
      "workspace selection-policy authorization"
    ],
    "must_not_own": [
      "Workspace admission",
      "Instrument minting/versioning",
      "TradingUniverse authority"
    ]
  },
  "selection_policy": {
    "projection_completeness": "workspace-selected subset",
    "owner": "NOT_FOUND",
    "omission_is_deletion": false,
    "semantic_admission": "BLOCKED"
  },
  "id_contract": {
    "minted_by": "WCP authority only; caller forbidden",
    "immutable_scope": [
      "workspace_id"
    ],
    "cross_workspace_reuse": "DENY",
    "stronger ID grammar": "NOT_FROZEN"
  },
  "options": {
    "A_immutable_journal_authenticated_head": "VIABLE_CANDIDATE; requires external freshness authority and crash protocol not frozen",
    "B_immutable_chain_current_index": "PREFERRED_CANDIDATE; requires authenticated chain/head plus external monotonic freshness anchor not frozen",
    "C_authenticated_current_only_external_anchor": "REJECTED_FOR_NOW; history/replacement and complete rollback proof absent",
    "D": "SELECTED: DESIGN_BLOCKED until exact freshness/auth/crash ownership is frozen"
  },
  "history_semantics": {
    "every_accepted_projection_historically_resolvable": "REQUIRED_CANDIDATE_NOT_FROZEN",
    "generation": "strict G+1 candidate",
    "predecessor": "exact prior authenticated record candidate",
    "replacement_semantics": "NOT_FROZEN",
    "current_only_authorized": false
  },
  "authentication": {
    "content_fingerprint": "structural integrity only; never authenticity",
    "required": "keyed authenticated state (repository HMAC convention candidate), distinct production/test domains and key material",
    "coherent_SQL_rewrite": "must fail authentication and freshness verification",
    "public_SHA_authenticity": false,
    "exact_key_owner_rotation_recovery": "NOT_FROZEN"
  },
  "rollback_freshness": {
    "valid_prefix_A_B_after_A_B_C": "MUST_DETECT_BEFORE_PUBLISH",
    "required_mechanism": "external non-rollbackable monotonic freshness anchor exact-bound to authenticated generation/head",
    "available_frozen_mechanism": "NOT_FOUND",
    "protection_result": "NOT_PROVEN",
    "design_consequence": "DESIGN_BLOCKED"
  },
  "environment_isolation": {
    "authentication_domain": "production and test distinct",
    "key_material": "production and test distinct",
    "durable_storage": "production and test distinct",
    "test_to_production_laundering": "DENY"
  },
  "crash_matrix": {
    "before_prepare": "old current remains",
    "after_prepare": "recover/verify prepared intent; do not publish new",
    "after_durable_record": "record remains unaccepted until authenticated head and anchor protocol completes",
    "before_head": "old current only",
    "after_head_before_external_anchor": "do not publish; recovery/fail closed",
    "before_external_anchor": "do not publish",
    "after_external_anchor": "publish only after exact local finalized state re-verifies; mismatch fail closed"
  },
  "restart_algorithm": [
    "open isolated durable state",
    "verify keyed authentication",
    "verify complete generation/predecessor/member closure",
    "verify external freshness anchor equals authenticated head",
    "resolve genuine upstream Workspace, exact ASCAT and exact Instrument versions",
    "rebuild current projection",
    "publish AVAILABLE only after every step succeeds"
  ],
  "upstream_mutation": {
    "historical_projection": "immutable; pinned IDs, exact versions, source tuples and selection proof never silently mutate",
    "workspace_lifecycle_change": "affects future/current usability per unfrozen policy, not historical bytes",
    "new_instrument_version": "no implicit upgrade",
    "producer_revocation": "no historical rewrite; runtime eligibility policy NOT_FROZEN",
    "new_catalog_snapshot": "no implicit replacement"
  },
  "semantic_admission_tuple": [
    "genuine Workspace",
    "genuine exact ASCAT",
    "exact authoritative current or historical Instrument versions",
    "exact ASCAT member source-tuple closure",
    "genuine workspace selection-policy authorization"
  ],
  "self_mint": "valid syntax + genuine ASCAT ID + valid Instrument IDs/bindings + correct public fingerprint remains DENY absent all genuine admission dependencies",
  "concurrency": {
    "serialization_owner": "future WCP authority per workspace",
    "CAS": "expected authenticated head generation and fingerprint required",
    "same_retry": "idempotent only for exact request identity and exact ordered instrument_ids",
    "same_members_different_order": "different request because order is fingerprint-significant",
    "conflict": "DENY/CONFLICT; never silent last-writer-wins",
    "exact_idempotency_key_schema": "NOT_FROZEN"
  },
  "missing_semantics": [
    "selection-policy owner",
    "exact historical/replacement model",
    "external freshness-anchor authority/API",
    "HMAC key ownership/rotation/recovery",
    "prepare/finalize transaction and crash recovery contract",
    "exact idempotency request identity"
  ],
  "negative_requirements": [
    "WCP raw fingerprint grants authority -> FAIL",
    "current-only WCP store without rollback semantics -> FAIL",
    "valid-prefix rollback accepted -> FAIL",
    "coherent SQL rewrite accepted -> FAIL",
    "production/test auth domain shared -> FAIL",
    "caller chooses wcat ID -> FAIL",
    "cross-workspace WCP ID reuse -> FAIL",
    "concurrent admissions silently last-writer-win -> FAIL"
  ]
}
```
