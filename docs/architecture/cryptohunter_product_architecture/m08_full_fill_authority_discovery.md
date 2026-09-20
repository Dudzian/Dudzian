# M0.8 Full Fill authority discovery

This file is a deterministic complete projection of `m08_full_fill_authority_discovery.json`. JSON is the source of truth.

```json
{
  "schema_version": "1.0.0",
  "artifact_kind": "M0.8_FULL_FILL_AUTHORITY_DISCOVERY",
  "repository_head_examined": "de4fb0cbd06721c162150460dc32aec190dd707c",
  "provenance_mode": "HYBRID_HEURISTIC",
  "equivalence_class": "MISMATCH",
  "formal_acceptance_allowed": false,
  "candidate_implementation_allowed": false,
  "m07_equivalence": {
    "reviewed_exact_commits": [
      "7dbfe87acad28eb92ef1878ed9976e012e8207ae",
      "565d1927b3e8ce27642c69eb9d585ac156ecec2f"
    ],
    "exact_objects_available": false,
    "canonical_merge": "de4fb0cbd06721c162150460dc32aec190dd707c",
    "lineage_commit": "c8fa0d5fd7f5cca69ccfb4583d6566d63179b9bc",
    "authority_relevant_paths_compared": [
      "bot_core/orders/__init__.py",
      "bot_core/orders/authority.py",
      "bot_core/orders/custody.py",
      "bot_core/security/keyring_storage.py",
      "bot_core/execution/m07_fill_validation.py",
      "bot_core/execution/__init__.py",
      "bot_core/accounting/__init__.py",
      "bot_core/accounting/authority.py",
      "docs/architecture/cryptohunter_product_architecture/commands_events_order_lifecycle_and_idempotency.json",
      "docs/architecture/cryptohunter_product_architecture/m07_order_authority_kernel_status.json",
      "docs/architecture/cryptohunter_product_architecture/m07_semantic_admission_upstream_discovery.json",
      "docs/architecture/cryptohunter_product_architecture/m07_semantic_admission_upstream_discovery.md",
      "docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json",
      "docs/architecture/cryptohunter_product_architecture/ledger_portfolio_capital_and_pnl.json",
      "tests/architecture/test_m07_order_authority.py",
      "tests/architecture/test_m07_semantic_admission_upstream_discovery.py",
      "tests/architecture/test_m07_accepted_full_fill_authority.py",
      "tests/architecture/test_cryptohunter_commands_events_order_lifecycle_and_idempotency.py",
      "tests/architecture/test_cryptohunter_ledger_portfolio_capital_and_pnl.py",
      "tests/architecture/test_m08_internal_accounting_authority.py"
    ],
    "semantic_differences": [
      "M0.7 Fill instrument_binding.required_matches still references execution exchange_id/environment while canonical M0.5 Instrument source model no longer carries those fields."
    ],
    "historical_metadata_differences": [
      "m07_semantic_admission_upstream_discovery.json and its Markdown projection retain the original blocked exact-base checkout metadata; these literals are historical provenance, not an architecture-semantic mismatch"
    ],
    "required_semantics": {
      "production_order_authority_kernel": "ACCEPTED_AVAILABLE",
      "production_order_history": "ACCEPTED_AVAILABLE",
      "historical_order_resolver": "ACCEPTED_AVAILABLE",
      "semantic_submit_order": "BLOCKED_UPSTREAM",
      "M0.7": "NOT_AVAILABLE"
    },
    "runtime_invariants_confirmed": [
      "HMAC-SHA-256 production admission",
      "production/test purpose separation",
      "KeyringOrderAuthoritySecretCustody",
      "public SHA is not admission proof",
      "coherent SQL mint denied",
      "coherent rewrite denied",
      "exact request -> outcome -> ORDER_PLANNED binding",
      "Fill identity dedupe",
      "external trade dedupe",
      "Fill cumulative progression",
      "type(authority) is OrderAuthority production boundary",
      "production submit_order fail-closes upstream"
    ]
  },
  "structural_fill_foundation": {
    "status": "ACCEPTED_CLOSED",
    "validator": "bot_core.execution.m07_fill_validation.validate_structural_fill",
    "fingerprint_symbol": "bot_core.execution.m07_fill_validation.canonical_fill_fingerprint",
    "scope": "structural shape, fee rules, canonical encoding, and fingerprint integrity only; no admission authority"
  },
  "canonical_fill_schema": {
    "status": "FOUND",
    "contract": "docs/architecture/cryptohunter_product_architecture/commands_events_order_lifecycle_and_idempotency.json",
    "json_pointer": "/fill_contract",
    "exact_fields": [
      "fill_id",
      "order_id",
      "environment",
      "workspace_id",
      "portfolio_id",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "instrument_metadata_version",
      "execution_route_id",
      "venue_trade_id",
      "side",
      "executed_quantity",
      "execution_price",
      "executed_at_utc",
      "fee_kind",
      "fee_quantity",
      "fee_asset_reference",
      "fill_fingerprint_sha256"
    ],
    "python_types": {
      "fill_id": "str",
      "order_id": "str",
      "environment": "str enum",
      "workspace_id": "str",
      "portfolio_id": "str",
      "exchange_account_id": "str",
      "exchange_id": "str",
      "instrument_id": "str",
      "instrument_metadata_version": "int (bool forbidden)",
      "execution_route_id": "str",
      "venue_trade_id": "str",
      "side": "str enum",
      "executed_quantity": "str",
      "execution_price": "str",
      "executed_at_utc": "str",
      "fee_kind": "str enum",
      "fee_quantity": "str",
      "fee_asset_reference": "dict[str,str] | None",
      "fill_fingerprint_sha256": "str"
    },
    "nullable_fields": [
      "fee_asset_reference"
    ],
    "decimal_rule": "canonical non-negative string regex (0|[1-9][0-9]*)(\\.[0-9]*[1-9])?; quantity and price positive; float forbidden",
    "timestamp_rule": "executed_at_utc is canonical UTC RFC3339 Z, whole seconds or fractional seconds ending non-zero",
    "fingerprint": {
      "algorithm": "SHA-256",
      "domain": [
        "fill_id",
        "order_id",
        "environment",
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "exchange_id",
        "instrument_id",
        "instrument_metadata_version",
        "execution_route_id",
        "venue_trade_id",
        "side",
        "executed_quantity",
        "execution_price",
        "executed_at_utc",
        "fee_kind",
        "fee_quantity",
        "fee_asset_reference"
      ],
      "excluded": [
        "fill_fingerprint_sha256"
      ],
      "canonicalization": [
        "UTF-8",
        "NFC",
        "lexicographically sorted JSON object keys",
        "comma/colon JSON separators",
        "canonical decimals",
        "canonical UTC Z timestamp"
      ]
    },
    "production_validator": "bot_core.execution.m07_fill_validation.validate_structural_fill",
    "test_oracles": [
      "tests.architecture.test_m07_accepted_full_fill_authority.frozen_fill_fingerprint",
      "tests.architecture.test_cryptohunter_commands_events_order_lifecycle_and_idempotency.validate_fill"
    ]
  },
  "production_full_fill_authority": {
    "status": "NOT_FOUND",
    "distinctions": {
      "structural_validator": "FOUND",
      "DTO": "raw Mapping only; no production accepted DTO",
      "execution_router_result": "not authority",
      "adapter_response": "not authority",
      "test_authority": "test-local executable contract only",
      "journal": "NOT_FOUND for accepted Fill",
      "projection": "NOT_FOUND in production",
      "genuine_production_admission_authority": "NOT_FOUND"
    },
    "architectural_owner": "M0.7"
  },
  "production_fill_store": "NOT_FOUND",
  "fill_history": "NOT_FOUND",
  "historical_fill_resolver": "NOT_FOUND",
  "current_fill_projection": "NOT_APPLICABLE_IMMUTABLE_FACT",
  "fill_identity": {
    "core": [
      "fill_id"
    ],
    "external_trade_identity": [
      "environment",
      "exchange_account_id",
      "exchange_id",
      "venue_trade_id"
    ],
    "economic_identity_fields": [
      "fill_id",
      "order_id",
      "environment",
      "workspace_id",
      "portfolio_id",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "instrument_metadata_version",
      "execution_route_id",
      "venue_trade_id",
      "side",
      "executed_quantity",
      "execution_price",
      "executed_at_utc",
      "fee_kind",
      "fee_quantity",
      "fee_asset_reference"
    ],
    "fingerprint_field": "fill_fingerprint_sha256"
  },
  "dedupe_semantics": {
    "same fill_id + same economics": "REPLAY_SUCCESS / NO_EFFECT",
    "same fill_id + different economics": "FILL_IDENTITY_CONFLICT",
    "same external trade identity + same economics": "REPLAY_SUCCESS / NO_EFFECT; retain first fact",
    "same external trade identity + different economics": "FILL_IDENTITY_CONFLICT",
    "new fill_id + same external trade": "REPLAY_SUCCESS only for identical economics; otherwise FILL_IDENTITY_CONFLICT",
    "same fill_id + new venue_trade_id": "FILL_IDENTITY_CONFLICT"
  },
  "fill_order_binding": {
    "exact_fields": [
      "order_id",
      "environment",
      "workspace_id",
      "portfolio_id",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "execution_route_id"
    ],
    "side_binding": "side is stored on Fill; contract event referential rules do not list side as an envelope equality field",
    "required_authority": "genuine accepted Order lifecycle history plus Core-owned accepted_fill_ids_by_order_id",
    "production_genuine_order_history": "NOT_AVAILABLE because semantic SUBMIT_ORDER is BLOCKED_UPSTREAM"
  },
  "order_state_requirement": {
    "basis": "current aggregate lifecycle transition plus complete accepted Fill sequence; no Fill aggregate_version field and no exact historical Order version is frozen in Full Fill",
    "direct_fill_sources": [
      "SUBMISSION_PENDING",
      "ACKNOWLEDGED",
      "PARTIALLY_FILLED",
      "CANCEL_PENDING",
      "REPLACE_PENDING",
      "RECONCILIATION_REQUIRED"
    ],
    "not_direct_fill_sources": [
      "PLANNED",
      "FILLED",
      "CANCELLED",
      "REJECTED",
      "EXPIRED",
      "REPLACED"
    ],
    "note": "DISPATCHED is an event; its target state is SUBMISSION_PENDING"
  },
  "late_fill_semantics": {
    "CANCEL_PENDING": "direct PARTIAL/FULL Fill lifecycle mutation permitted",
    "REPLACE_PENDING": "direct PARTIAL/FULL Fill lifecycle mutation permitted",
    "CANCELLED": "terminal: identical fact replay only; other facts require reconciliation/audit and never regress terminal state",
    "REPLACED": "terminal: identical fact replay only; other facts require reconciliation/audit and never regress terminal state",
    "EXPIRED": "terminal: identical fact replay only; other facts require reconciliation/audit and never regress terminal state",
    "FILLED": "terminal: identical fact replay only; other facts require reconciliation/audit and never regress terminal state"
  },
  "fill_progression_ownership": {
    "full_fill_admission_history_authority_owner": "M0.7",
    "fill_identity_dedupe_owner": "M0.7",
    "external_venue_trade_dedupe_owner": "M0.7",
    "accepted_fill_ids_by_order_id_owner": "M0.7",
    "cumulative_order_execution_progression_owner": "M0.7",
    "partial_full_overfill_lifecycle_enforcement_owner": "M0.7",
    "M0.8_role": "DOWNSTREAM_ACCOUNTING_CONSUMER_ONLY: consumes genuine accepted M0.7 Full Fill and derives accounting postings, FIFO, and P&L; does not mint, admit, deduplicate, or own canonical Fill history",
    "M0.8_fill_authority_owner": false,
    "M0.8_independent_external_venue_trade_admission": false
  },
  "fill_instrument_binding": {
    "fill_fields": [
      "instrument_id",
      "instrument_metadata_version"
    ],
    "fields_absent_from_fill": [
      "accepted_source_catalog_snapshot_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol",
      "source_metadata_version_id"
    ],
    "stale_m07_required_matches": [
      "workspace_id",
      "exchange_id",
      "environment"
    ],
    "current_m05_execution_fields": [],
    "current_m05_source_identity_fields": [
      "workspace_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol"
    ],
    "historical_resolution_key": [
      "instrument_id",
      "metadata_version",
      "accepted_source_catalog_snapshot_id"
    ],
    "additional_source_metadata_binding_required": true,
    "option_a_status": "UNPROVEN: M05PrevalidatedInstrumentHistory is test-only and the M0.7 reference resolver validates the obsolete exchange_id/environment Instrument shape; no non-caller-mintable production composite carries the canonical snapshot/projection/source-member proof",
    "option_b_status": "UNRESOLVED: reconcile the contract before deciding whether M0.7 Fill schema itself requires migration; do not add fields silently",
    "execution_scope_owner": "Fill -> genuine Order / ExchangeAccount / ExecutionRoute graph owns execution exchange_id and environment",
    "source_instrument_owner": "canonical M0.5 Catalog graph owns source_exchange_id, market_type, venue_symbol, accepted_source_catalog_snapshot_id, source_metadata_version_id binding, and historical source provenance",
    "direct_execution_field_comparison_forbidden": true,
    "M0.7_fill_migration_required": "UNRESOLVED",
    "historical_instrument_authority": "NOT_AVAILABLE",
    "production_dependency": "M0.5"
  },
  "fill_account_binding": {
    "fields": [
      "exchange_account_id",
      "environment",
      "exchange_id"
    ],
    "market_type": "NOT_FROZEN",
    "credential_or_profile_generation": "NOT_FROZEN",
    "resolution_mode": "no production historical ExchangeAccount resolver requirement is frozen in /fill_contract; exact binding to Order/event envelope is frozen"
  },
  "execution_route_binding": {
    "fields": [
      "execution_route_id"
    ],
    "adapter_family_id": "NOT_FROZEN",
    "adapter_implementation_release_generation": "NOT_FROZEN",
    "dispatch_route_proof": "binding to accepted Order/event envelope required; separate adapter release proof NOT_FROZEN"
  },
  "execution_producer_authenticity": {
    "status": "NOT_FOUND",
    "catalog_source_producer_membership_is_execution_proof": false,
    "reason": "no production execution Fill producer admission/authentication authority or authenticated adapter-session binding found"
  },
  "venue_trade_provenance": {
    "classification": "NOT_FOUND",
    "caller_supplied_venue_trade_id_is_proof": false
  },
  "fee_semantics": {
    "NONE": "fee_quantity exactly 0 and fee_asset_reference null",
    "CHARGE": "positive fee_quantity and non-null exact/explicit-alias asset reference with asset_namespace == exchange_id",
    "negative": "forbidden",
    "zero_charge": "forbidden",
    "third_asset": "legal and preserved",
    "base_or_quote_only": false,
    "maker_rebate": "unsupported fail-closed"
  },
  "fifo_accounting_authority": {
    "status": "NOT_FOUND",
    "rule": "ACCOUNTING_SPOT_FIFO_V1",
    "production_partial_foundation": "bot_core.accounting.authority.AccountingAuthority exists for deposit/withdrawal journals but fill is explicitly BLOCKED_SOURCE_TYPES and no accept_fill path exists",
    "test_oracle": "tests/architecture/test_cryptohunter_ledger_portfolio_capital_and_pnl.py contains executable FIFO model, not production authority",
    "required_fill_facts": [
      "fill_id",
      "order_id",
      "scope",
      "side",
      "executed_quantity",
      "execution_price",
      "executed_at_utc",
      "fee semantics",
      "resolved historical base/quote/settlement assets"
    ],
    "downstream_distinction": [
      "accepted execution Fill",
      "internal accounting journal",
      "observed balance fact",
      "reconciliation result are separate authorities"
    ],
    "future_m08_role": "CONSUMES genuine accepted M0.7 Full Fill context/projection to derive accounting postings, FIFO cost basis, and P&L",
    "forbidden": [
      "create Fill authority",
      "mint accepted Fill",
      "maintain competing canonical Fill history",
      "perform independent venue-trade admission"
    ],
    "required_upstream_context": "genuine M0.7 composite trusted accepted Full Fill and Core-owned accepted_fill_ids_by_order_id evidence"
  },
  "self_mint_fill_authority_fact": {
    "result": "NO",
    "paths": [
      "bot_core.execution.m07_fill_validation.validate_structural_fill returns only a structurally valid mapping",
      "bot_core.execution.__init__ exports no AcceptedFillAuthority/CoreAcceptedFillProjection",
      "bot_core.accounting.authority.BLOCKED_SOURCE_TYPES includes fill and no accept_fill API exists"
    ],
    "qualification": "No production authority fact can currently be minted at all; a self-hashed raw Fill is not genuine admission."
  },
  "timestamp_semantics": {
    "executed_at_utc": "immutable venue/execution economic time carried by Fill; authenticity owner not implemented",
    "occurred_at_utc": "separate Order event time",
    "observed_at_utc": "NOT_FROZEN",
    "received_at_utc": "NOT_FROZEN",
    "accepted_at_utc": "NOT_FROZEN"
  },
  "authority_validation_order": "ORDERING_UNSPECIFIED beyond frozen dependencies: structural Fill, nominal M0.5 historical Instrument resolution, complete accepted Fill/Order history, dedupe/progression, then accounting downstream",
  "cross_authority_toctou": {
    "order_change_during_validation": "NOT_FOUND",
    "instrument_history_concurrent_resolution": "NOT_FOUND",
    "duplicate_fill_race": "NOT_FOUND",
    "two_fills_final_quantity_race": "NOT_FOUND",
    "cancel_or_replace_ack_vs_fill": "OrderAuthority serializes its own event journal with RLock/SQL transaction, but no cross-authority Fill transaction exists"
  },
  "concurrency_semantics": {
    "same_fill_simultaneously": "contract says replay, production atomic mechanism NOT_FOUND",
    "same_venue_trade_simultaneously": "contract says replay/conflict by economics, production atomic mechanism NOT_FOUND",
    "two_unique_fills_racing": "progression contract frozen, production cross-authority mechanism NOT_FOUND",
    "final_vs_partial": "progression contract frozen, production cross-authority mechanism NOT_FOUND",
    "fill_vs_terminal_transition": "terminal rule frozen, production cross-authority mechanism NOT_FOUND"
  },
  "durability_requirements_from_repo_conventions": [
    "append-only authenticated journal",
    "production/test domain separation",
    "restart replay validation",
    "head and valid-prefix rollback disclosure",
    "atomic authority fence/CAS or transaction",
    "coherent SQL mint/rewrite denial"
  ],
  "fill_authority_kernel_buildability": {
    "classification": "BLOCKED_BY_CROSS_CONTRACT_FILL_INSTRUMENT_BINDING",
    "reason": "A durable journal must not seal accepted Fill identity/evidence until immutable M0.5 source-catalog provenance and the execution/source identity split are reconciled; otherwise it commits the wrong referential proof and creates a migration hazard.",
    "required_next_decision": "Choose and freeze a non-caller-mintable canonical composite resolver or explicitly migrate the M0.7 Fill contract.",
    "semantic_admission_blockers": [
      "cross-contract Fill/Instrument binding inconsistency",
      "production M0.5 exact historical Instrument authority",
      "production genuine Order creation/history rooted in semantic SUBMIT_ORDER",
      "execution Fill producer authenticity and venue-trade provenance"
    ]
  },
  "semantic_production_fill_admission": "NOT_AVAILABLE",
  "M0.8_status": "NOT_AVAILABLE",
  "next_buildable_stage": "M07_M05_FILL_INSTRUMENT_BINDING_RECONCILIATION",
  "next_blocker": "CROSS_CONTRACT_FILL_INSTRUMENT_BINDING_INCONSISTENT",
  "existing_test_inventory": {
    "tests/architecture/test_m07_accepted_full_fill_authority.py": "proves structural schema/fingerprint/fee rules and absence of self-enrolling Fill/FIFO path; does not prove production admission",
    "tests/architecture/test_m07_order_authority.py": "proves durable Order kernel, event-level fill dedupe/progression and tamper resistance; does not prove genuine Fill producer or Full Fill store",
    "tests/architecture/test_cryptohunter_commands_events_order_lifecycle_and_idempotency.py": "test oracle for frozen Full Fill composite semantics; its in-test projection/resolver is not production authority",
    "tests/architecture/test_cryptohunter_ledger_portfolio_capital_and_pnl.py": "executable accounting/FIFO contract model; does not prove production Fill accounting path",
    "tests/architecture/test_m08_internal_accounting_authority.py": "proves production deposit/withdrawal accounting kernel and explicit Fill rejection; does not prove Fill/FIFO authority"
  },
  "preserved_status": {
    "M0.7_OrderAuthority_kernel": "ACCEPTED_AVAILABLE",
    "M0.7_semantic_SUBMIT_ORDER": "BLOCKED_UPSTREAM",
    "M0.7": "NOT_AVAILABLE",
    "M0.8_structural_foundation": "CLOSED",
    "M0.8": "NOT_AVAILABLE",
    "M0.12_1.46.0": "ACCEPTED",
    "Workspace": "WITHHELD",
    "production_M0.5": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  },
  "cross_contract_consistency": {
    "source_contract": "docs/architecture/cryptohunter_product_architecture/ledger_portfolio_capital_and_pnl.json",
    "json_pointer": "/cross_contract_dependencies",
    "canonical_hash": "SHA-256 over UTF-8 NFC-normalized JSON with sorted keys and comma/colon separators, identical to executable M0.8 oracle",
    "overall_status": "NO_DEPENDENCY_FINGERPRINT_DRIFT",
    "dependencies": [
      {
        "contract": "canonical_domain_vocabulary.json",
        "json_pointer": "/public_trading_environments",
        "declared_fingerprint": "b0114e386bf72439199ec8155d65a57dc7e00cabe79dba805f53221ba7713103",
        "current_recomputed_fingerprint": "b0114e386bf72439199ec8155d65a57dc7e00cabe79dba805f53221ba7713103",
        "status": "MATCH"
      },
      {
        "contract": "canonical_domain_vocabulary.json",
        "json_pointer": "/entity_kinds",
        "declared_fingerprint": "bb92436a67abe42975d763b06717962009d3f4bc8a6066b227f34c8e7d178e9b",
        "current_recomputed_fingerprint": "bb92436a67abe42975d763b06717962009d3f4bc8a6066b227f34c8e7d178e9b",
        "status": "MATCH"
      },
      {
        "contract": "canonical_domain_vocabulary.json",
        "json_pointer": "/relationships",
        "declared_fingerprint": "03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9",
        "current_recomputed_fingerprint": "03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9",
        "status": "MATCH"
      },
      {
        "contract": "canonical_domain_vocabulary.json",
        "json_pointer": "/identifier_policy",
        "declared_fingerprint": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
        "current_recomputed_fingerprint": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
        "status": "MATCH"
      },
      {
        "contract": "exchange_accounts_and_instruments.json",
        "json_pointer": "/asset_reference_contract",
        "declared_fingerprint": "6aaa3985e58acd4f7ee82cbd941e67507dd76a3186e7969bf8c62db1d4d41ee0",
        "current_recomputed_fingerprint": "6aaa3985e58acd4f7ee82cbd941e67507dd76a3186e7969bf8c62db1d4d41ee0",
        "status": "MATCH"
      },
      {
        "contract": "exchange_accounts_and_instruments.json",
        "json_pointer": "/decimal_policy",
        "declared_fingerprint": "e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48",
        "current_recomputed_fingerprint": "e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48",
        "status": "MATCH"
      },
      {
        "contract": "exchange_accounts_and_instruments.json",
        "json_pointer": "/instrument_contract/record_fields",
        "declared_fingerprint": "39b5d32a1d887863cbc7f7a652e163b6b78d02a6ae21b1e4447e97437e18bf16",
        "current_recomputed_fingerprint": "39b5d32a1d887863cbc7f7a652e163b6b78d02a6ae21b1e4447e97437e18bf16",
        "status": "MATCH"
      },
      {
        "contract": "exchange_accounts_and_instruments.json",
        "json_pointer": "/instrument_contract/trusted_history_contract",
        "declared_fingerprint": "d3bdb205695163ec2834a065bb872d9ca39f1ad1b0d60c592d7273f42c380c65",
        "current_recomputed_fingerprint": "d3bdb205695163ec2834a065bb872d9ca39f1ad1b0d60c592d7273f42c380c65",
        "status": "MATCH"
      },
      {
        "contract": "exchange_accounts_and_instruments.json",
        "json_pointer": "/instrument_type_registry",
        "declared_fingerprint": "e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8",
        "current_recomputed_fingerprint": "e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8",
        "status": "MATCH"
      },
      {
        "contract": "strategy_market_data_and_execution_routing.json",
        "json_pointer": "/current_edition_execution_pair_policy",
        "declared_fingerprint": "7dd12317c8dab7bc2d19e751f39800db7895f737a42335df1e22927d465ec0b9",
        "current_recomputed_fingerprint": "7dd12317c8dab7bc2d19e751f39800db7895f737a42335df1e22927d465ec0b9",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/fill_contract",
        "declared_fingerprint": "1846393d14f684fc2462eb12b5d92f9c18cfa8a0163b2a2f41fe88c913324d1b",
        "current_recomputed_fingerprint": "1846393d14f684fc2462eb12b5d92f9c18cfa8a0163b2a2f41fe88c913324d1b",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/command_registry/SUBMIT_ORDER",
        "declared_fingerprint": "864bc27bf55228d08b6592f2042a3f9a1f447eae661382bde0b380602369d748",
        "current_recomputed_fingerprint": "864bc27bf55228d08b6592f2042a3f9a1f447eae661382bde0b380602369d748",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/event_contract",
        "declared_fingerprint": "c63d514a4546161798de2f7f90441821cd71653fba433c3577c6d7b630e4b6b2",
        "current_recomputed_fingerprint": "c63d514a4546161798de2f7f90441821cd71653fba433c3577c6d7b630e4b6b2",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/order_lifecycle",
        "declared_fingerprint": "48a514419aaa0863078e69ffc50c3acd4b37a06d873d257275fb68874cc840dd",
        "current_recomputed_fingerprint": "48a514419aaa0863078e69ffc50c3acd4b37a06d873d257275fb68874cc840dd",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/idempotency_contract",
        "declared_fingerprint": "43ba37d976eb5948970d289cc71cd06da82cb90803f9e5f61cc84109f00e9a62",
        "current_recomputed_fingerprint": "43ba37d976eb5948970d289cc71cd06da82cb90803f9e5f61cc84109f00e9a62",
        "status": "MATCH"
      },
      {
        "contract": "commands_events_order_lifecycle_and_idempotency.json",
        "json_pointer": "/closed_request_policy",
        "declared_fingerprint": "f3a523b01cbce2bafabeaad261777e3a97fc4f60db751deee0e77cb912f93e7c",
        "current_recomputed_fingerprint": "f3a523b01cbce2bafabeaad261777e3a97fc4f60db751deee0e77cb912f93e7c",
        "status": "MATCH"
      }
    ],
    "drifted_pointers": [],
    "red_test": {
      "test": "tests/architecture/test_cryptohunter_ledger_portfolio_capital_and_pnl.py::test_contract_and_real_upstream_dependencies_are_valid",
      "result": "VALID",
      "classification": "NO_RELEVANT_CROSS_CONTRACT_FINGERPRINT_DRIFT"
    },
    "historical_target_contract": "CLOSED_AT_ORIGINAL_BASELINE",
    "current_upstream_compatibility": "DEPENDENCY_FINGERPRINTS_MATCH"
  },
  "current_m05_instrument_model": {
    "sources": [
      "docs/architecture/cryptohunter_product_architecture/exchange_accounts_and_instruments.json",
      "bot_core/instruments/catalog_projection_oracle.py",
      "tests/architecture/test_m06_workspace_catalog_source_chain.py"
    ],
    "record_fields": [
      "instrument_id",
      "workspace_id",
      "source_exchange_id",
      "market_type",
      "instrument_type",
      "venue_symbol",
      "display_symbol",
      "base_asset_reference",
      "quote_asset_reference",
      "settlement_asset_reference",
      "trading_status",
      "price_tick",
      "quantity_step",
      "min_quantity",
      "max_quantity",
      "min_notional",
      "max_notional",
      "contract_size",
      "contract_value_currency",
      "derivative_settlement_type",
      "expiry_at_utc",
      "strike_price",
      "option_side",
      "accepted_source_catalog_snapshot_id",
      "metadata_version",
      "observed_at_utc",
      "effective_at_utc",
      "stale_after_utc",
      "source_adapter_family_id"
    ],
    "immutable_identity_fields": [
      "workspace_id",
      "source_exchange_id",
      "market_type",
      "venue_symbol"
    ],
    "execution_fields_present": [],
    "execution_fields_absent": [
      "exchange_id",
      "environment"
    ],
    "source_exchange_id_semantics": "exact source venue namespace; immutable; never rewritten for PAPER execution",
    "market_type_semantics": "immutable source product dimension and accepted source snapshot scope",
    "venue_symbol_semantics": "exact venue identifier; no trim, case-fold, Unicode normalization, or symbol-only lookup",
    "instrument_metadata_version_field": "metadata_version on Instrument; positive non-boolean, unique and strictly increasing in history",
    "source_metadata_version_id_relationship": "not an Instrument record field; required on WorkspaceCatalogProjection member binding and exact-matched to the AcceptedSourceCatalogSnapshot member tuple (source_exchange_id, market_type, venue_symbol, source_metadata_version_id)",
    "accepted_source_catalog_snapshot_id_relationship": "Instrument metadata version exact-binds its accepted source snapshot; projection and source member binding must match that same snapshot",
    "historical_resolution_key": [
      "instrument_id",
      "metadata_version",
      "accepted_source_catalog_snapshot_id"
    ],
    "additional_source_binding": [
      "workspace projection member binding source_exchange_id",
      "market_type",
      "venue_symbol",
      "source_metadata_version_id",
      "accepted source snapshot membership and producer provenance"
    ],
    "current_history_coherence": "one immutable (workspace_id, source_exchange_id, market_type, venue_symbol) tuple; current metadata_version strictly greater than complete history; exact historical resolution never falls back to current"
  },
  "fill_instrument_binding_status": "CONTRACT_INCONSISTENT",
  "M0.8_historical_target_contract": "CLOSED_AT_ORIGINAL_BASELINE",
  "M0.8_current_upstream_compatibility": "DEPENDENCY_FINGERPRINTS_MATCH_SEMANTIC_FILL_BINDING_INCONSISTENT",
  "single_fill_authority_invariant": {
    "canonical_accepted_fill_authority_owner": "M0.7",
    "canonical_accepted_fill_history_owner": "M0.7",
    "m08_is_downstream_consumer_only": true,
    "competing_fill_authority_forbidden": true,
    "m08_may_mint_or_admit_fill": false,
    "m08_may_deduplicate_external_venue_trade_independently": false,
    "journal_separation": "M0.7 Fill journal/history != M0.8 LedgerEntry/accounting journal; a downstream accepted-source reference/evidence never becomes Fill authority"
  },
  "authority_boundary_red_team": {
    "scenario": "F1 appears genuine in an M0.8 accounting path but is absent from M0.7 accepted Fill history",
    "classification": "IMPOSSIBLE_BY_AUTHORITY_BOUNDARY",
    "required_result": "M0.8 rejects F1 as an accounting source; only a genuine accepted M0.7 Full Fill context/projection is consumable"
  }
}
```
