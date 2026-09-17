# M0.7/M0.5 Fill–Instrument binding reconciliation

This file is a deterministic complete projection of `m07_m05_fill_instrument_binding_reconciliation.json`. JSON is the source of truth.

```json
{
  "schema_version": "1.0.0",
  "artifact_kind": "M07_M05_FILL_INSTRUMENT_BINDING_RECONCILIATION",
  "repository_head_examined": "3d38c17eb03d4f6baeae38e7cf4c8627b35ec0ef",
  "accepted_discovery_amendment": "0473d3e88224a515390be3857033d28981c6f215",
  "provenance_mode": "CURRENT_WORKTREE_CONTRACT_SEMANTICS_WITH_ACCEPTED_AMENDMENT_REFERENCE",
  "current_m07_fill_binding": {
    "schema_field_count": 19,
    "available_instrument_reference_fields": [
      "instrument_id",
      "instrument_metadata_version",
      "workspace_id"
    ],
    "stale_required_matches": [
      "workspace_id",
      "exchange_id",
      "environment"
    ],
    "valid_direct_instrument_matches": [
      "workspace_id"
    ],
    "forbidden_direct_instrument_matches": [
      "exchange_id",
      "environment"
    ],
    "reason": "Current Instrument has workspace_id but has neither execution exchange_id nor environment; PAPER must preserve source_exchange_id rather than rewrite it."
  },
  "current_m05_instrument_identity": {
    "workspace_scope": "workspace_id",
    "source_product_identity": [
      "source_exchange_id",
      "market_type",
      "venue_symbol"
    ],
    "version_field": "metadata_version",
    "snapshot_binding_field": "accepted_source_catalog_snapshot_id",
    "execution_environment_is_instrument_field": false,
    "exchange_id_is_instrument_field": false,
    "exchange_id_equals_source_exchange_id_assumption": "FORBIDDEN"
  },
  "current_m05_historical_resolution": {
    "canonical_resolver_input": [
      "instrument_id",
      "metadata_version",
      "accepted_source_catalog_snapshot_id"
    ],
    "storage_primary_key": "instrument_history_by_id map key is instrument_id; value is an ordered non-empty version history",
    "public_authority_resolver_input": "No production public resolver exists; frozen contract requires exact instrument_id + metadata_version + accepted_source_catalog_snapshot_id.",
    "record_provenance_attributes": [
      "workspace_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol",
      "accepted_source_catalog_snapshot_id",
      "source_adapter_family_id"
    ],
    "validation_context_keys": [
      "instrument_history_by_id",
      "workspace_catalog_projections_by_id",
      "accepted_source_catalog_snapshots_by_id",
      "source_producer_membership_authority"
    ],
    "current_fallback": "FORBIDDEN",
    "latest_snapshot_fallback": "FORBIDDEN",
    "symbol_only_lookup": "FORBIDDEN"
  },
  "execution_scope_owner": {
    "environment": "genuine Order + ExchangeAccount + ExecutionRoute execution graph",
    "exchange_id": "genuine Order + ExchangeAccount + ExecutionRoute execution graph",
    "exchange_account_id": "genuine Order + ExchangeAccount + ExecutionRoute execution graph",
    "execution_route_id": "genuine Order + ExecutionRoute execution graph",
    "portfolio_id": "genuine Order + ExchangeAccount parent scope",
    "workspace_id": "genuine Order workspace scope; also exact continuity constraint to historical Instrument and WorkspaceCatalogProjection"
  },
  "source_scope_owner": {
    "workspace_id": "M0.5 Instrument and genuine WorkspaceCatalogProjection",
    "source_exchange_id": "M0.5 Instrument; exact ASCAT member and WorkspaceCatalogProjection member binding",
    "market_type": "M0.5 Instrument; exact ASCAT member and WorkspaceCatalogProjection member binding",
    "venue_symbol": "M0.5 Instrument; exact ASCAT member and WorkspaceCatalogProjection member binding",
    "source_metadata_version_id": "genuine ASCAT member, exact-bound by WorkspaceCatalogProjection member binding",
    "accepted_source_catalog_snapshot_id": "Core-owned AcceptedSourceCatalogSnapshot identity, referenced by Instrument and WorkspaceCatalogProjection",
    "instrument_metadata_version": "M0.5 Instrument historical record metadata_version, exact-bound by WorkspaceCatalogProjection member binding"
  },
  "composite_resolver_candidate": {
    "name": "M05PrevalidatedInstrumentHistory",
    "production_authority": "NOT_FOUND_TEST_ONLY",
    "requested_input_available_from_fill": [
      "instrument_id",
      "instrument_metadata_version",
      "workspace_id"
    ],
    "required_frozen_input_missing_from_fill_or_admission_envelope": [
      "accepted_source_catalog_snapshot_id"
    ],
    "status": "INSUFFICIENT",
    "caller_may_supply_snapshot_as_trusted_proof": false,
    "order_bridge": {
      "supported_equalities": [
        "Fill.instrument_id == Order.instrument_id",
        "Fill execution scope == Order execution scope"
      ],
      "missing_binding": "Order admission does not freeze accepted_source_catalog_snapshot_id or an equivalent exact source-provenance handle.",
      "source_identity_fields_need_not_be_duplicated_if_exact_snapshot binding is authority-bound": true
    }
  },
  "composite_uniqueness_proof": {
    "claim": "instrument_id + instrument_metadata_version uniquely selects a version inside a structurally valid history map, but does not satisfy the canonical historical resolution contract.",
    "status": "FAILED_FOR_CANONICAL_RESOLUTION",
    "countercondition": "M0.5 explicitly requires accepted_source_catalog_snapshot_id as a third exact selector; it is both a record attribute and a resolver selector.",
    "accepted_source_catalog_snapshot_id_role": "BOTH",
    "toctou": "NOT_CLOSED_BY_CURRENT_FILL_FIELDS; metadata_version is pinned but exact source snapshot selector is absent, and current/latest fallback is forbidden."
  },
  "workspace_catalog_projection_dependency": {
    "contractual_bridge": "A genuine projection exact-binds workspace_id and accepted_source_catalog_snapshot_id; its member binds instrument_id, instrument metadata_version, source_exchange_id, market_type, venue_symbol, and source_metadata_version_id.",
    "raw_mapping_is_proof": false,
    "projection_mints_source_acceptance": false,
    "runtime_authority": "NOT_AVAILABLE",
    "runtime_writer": "NOT_IMPLEMENTED"
  },
  "accepted_source_catalog_dependency": {
    "exact_binding": [
      "accepted_source_catalog_snapshot_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol",
      "source_metadata_version_id"
    ],
    "producer_membership_provenance_required": true,
    "caller_snapshot_id_is_proof": false,
    "current_catalog_fallback": false,
    "latest_snapshot_fallback": false,
    "contract_declared_runtime_writer": "NOT_IMPLEMENTED",
    "contract_metadata_interpretation": "HISTORICAL_ONLY_NOT_CURRENT_PRODUCTION_AVAILABILITY_PROOF",
    "current_production_catalog_runtime_authority": "AVAILABLE",
    "authority": "ACCEPTED_AVAILABLE",
    "authority_symbol": "bot_core.instruments.catalog_runtime_acceptance.CatalogRuntimeAcceptanceAuthority",
    "public_ingestion_boundary": "fetch_catalog_once",
    "catalog_admission_receipt_authority": "ACCEPTED_AVAILABLE",
    "source_producer_membership_authority": "ACCEPTED_AVAILABLE",
    "need_to_implement_new_catalog_runtime_authority": false,
    "reason": "The current production tree contains the genuine CatalogRuntimeAcceptanceAuthority even though frozen M0.5 contract metadata still declares runtime_writer NOT_IMPLEMENTED."
  },
  "fill_schema_migration_required": {
    "value": true,
    "minimal_immutable_binding_to_design": "accepted_source_catalog_snapshot_id, or an authority-owned admission-envelope reference that deterministically supplies that exact selector and is cryptographically/durably bound to the accepted Fill",
    "not_yet_added_fields": [
      "accepted_source_catalog_snapshot_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol",
      "source_metadata_version_id"
    ],
    "reason": "The 19-field Fill and current Order admission envelope do not carry the third selector required by frozen M0.5 historical resolution."
  },
  "asset_reference_drift_impact": {
    "classification": "BLOCKING",
    "reason": "M0.5 asset_namespace exact-binds source_exchange_id and PAPER never rewrites it, while M0.7 fee validation currently compares asset_namespace to execution exchange_id. Base/quote/settlement references come from the same exact historical Instrument; FIFO remains deferred but cannot safely consume the stale binding.",
    "affected": [
      "fee_asset_reference",
      "base_asset_reference",
      "quote_asset_reference",
      "settlement_asset_reference",
      "future FIFO accounting"
    ]
  },
  "entity_kinds_drift_impact": {
    "classification": "NONBLOCKING",
    "reason": "Current vocabulary still defines Instrument as a Workspace-owned source-product projection, Fill as an Order child, and LedgerEntry as downstream of Fill. The fingerprint drift is an upstream extension/metadata change and adds no alternative Fill-to-Instrument resolver semantics."
  },
  "reconciliation_result": "M07_FILL_CONTRACT_MIGRATION_REQUIRED",
  "candidate_next_stage": "M07_FILL_CONTRACT_MIGRATION_DESIGN_DISCOVERY",
  "blockers": [
    "Design and freeze the minimal authority-bound accepted_source_catalog_snapshot_id binding without silently adding all source fields to Fill.",
    "Reconcile fee_asset_reference.asset_namespace against resolved source_exchange_id rather than execution exchange_id.",
    "Implement the missing WorkspaceCatalogProjection production runtime authority; use the existing ACCEPTED / AVAILABLE CatalogRuntimeAcceptanceAuthority as the genuine ASCAT source and do not build a competing Catalog authority.",
    "Replace test-only M05PrevalidatedInstrumentHistory with a production non-caller-mintable M0.5 historical Instrument authority/resolver.",
    "Only after semantic reconciliation, update M0.8 cross-contract fingerprints in a later implementation iteration."
  ],
  "preserved_status": {
    "M0.7_OrderAuthority_kernel": "ACCEPTED_AVAILABLE",
    "M0.7_structural_Full_Fill_foundation": "CLOSED",
    "production_M0.7_FullFillAuthority": "NOT_FOUND",
    "M0.7_semantic_SUBMIT_ORDER": "BLOCKED_UPSTREAM",
    "M0.8": "NOT_AVAILABLE",
    "production_M0.5": "NOT_AVAILABLE",
    "M0.12_1.46.0": "ACCEPTED",
    "CatalogRuntimeAcceptanceAuthority": "ACCEPTED_AVAILABLE",
    "CatalogAdmissionReceiptAuthority": "ACCEPTED_AVAILABLE",
    "AcceptedSourceProducerMembership": "ACCEPTED_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "catalog_authority_invariant": {
    "accepted_source_catalog_runtime_owner": "CatalogRuntimeAcceptanceAuthority",
    "competing_catalog_runtime_authority_forbidden": true,
    "catalog_admission_receipt_authority_required": true,
    "source_producer_membership_authority_required": true
  }
}
```
