# M0.7 structural Full Fill v2 implementation status

This is a deterministic projection of `m07_structural_full_fill_v2_status.json`; JSON is the source of truth.

```json
{
  "schema_version": "1.0.0",
  "artifact_kind": "M07_STRUCTURAL_FULL_FILL_V2_IMPLEMENTATION_STATUS",
  "implemented_commit": "RECORDED_BY_GIT_COMMIT_CONTAINING_THIS_ARTIFACT",
  "v1_status": "CLOSED_AVAILABLE_FOR_HISTORICAL_STRUCTURAL_VALIDATION",
  "v2_contract_status": "IMPLEMENTED_CANDIDATE",
  "v2_validator_status": "IMPLEMENTED_CANDIDATE",
  "public_validator_symbol": "bot_core.execution.m07_fill_validation.validate_structural_fill",
  "validator_input_mode": "RAW_FILL_ONLY",
  "v2_field_count": 20,
  "v2_fingerprint_domain": "cryptohunter.m0.7.full_fill.v2",
  "snapshot_id_grammar": {
    "type": "exact_str",
    "prefix": "ascat_",
    "suffix": "non_empty",
    "syntax_grants_trust": false
  },
  "fee_structural_policy": {
    "NONE": "quantity exactly 0; asset reference null",
    "CHARGE": "positive quantity; exact AssetReference shape; no namespace comparison",
    "namespace_comparison": "NONE",
    "future_semantic_rule": "FullFillAuthority must compare namespace with resolved historical Instrument.source_exchange_id"
  },
  "semantic_authority_enabled": false,
  "historical_instrument_resolution_enabled": false,
  "workspace_projection_resolution_enabled": false,
  "legacy_policy": "exact v1 field set retains legacy undomained 18-input SHA-256; no automatic upgrade",
  "preserved_status": {
    "M0.7_FullFillAuthority": "NOT_AVAILABLE",
    "semantic_Fill_admission": "NOT_AVAILABLE",
    "WorkspaceCatalogProjection_authority": "NOT_AVAILABLE",
    "production_M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "canonical_contract_file": "commands_events_order_lifecycle_and_idempotency.json",
  "canonical_v1_pointer": "/fill_contract",
  "canonical_v2_pointer": "/fill_contract_versions/v2"
}
```
