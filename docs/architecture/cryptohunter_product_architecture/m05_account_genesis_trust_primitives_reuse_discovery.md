# M0.5 account-genesis trust-primitives reuse discovery

This file is a deterministic complete projection of `m05_account_genesis_trust_primitives_reuse_discovery.json`. JSON is the source of truth.

```json
{
  "artifact": "M05_ACCOUNT_GENESIS_TRUST_PRIMITIVES_REUSE_DISCOVERY",
  "iteration": "DISCOVERY / RECONCILIATION ONLY",
  "repository_head_examined": "dde37719fdeecf745c803dca58534a7f40a01a9e",
  "reviewed_head_supplied": "2a88c726cacebb718748b4872177e9ca639df8af",
  "provenance": {
    "source": "GIT",
    "reviewed_head_available_locally": false,
    "classification": "UNKNOWN",
    "relationship_to_reviewed_sha": "UNKNOWN",
    "actual_local_source_tree": "dde37719fdeecf745c803dca58534a7f40a01a9e",
    "finding_scope": "CURRENT_TREE_ONLY",
    "formal_advancement_allowed": false,
    "rule": "The supplied reviewed commit is unavailable locally; current-tree evidence is not claimed equivalent."
  },
  "m012_authentication_inventory": [
    {
      "mechanism": "Catalog admission receipt and finalization HMAC",
      "module": "bot_core.instruments.catalog_admission_receipt",
      "class": "CatalogAdmissionReceiptAuthority",
      "algorithm": "HMAC-SHA-256",
      "key_owner": "CatalogAdmissionReceiptAuthority via SecretStorageCatalogAdmissionReceiptSecureCustody",
      "key_source": "secrets.token_bytes(32), persisted through production KeyringSecretStorage",
      "key_identifier": "cark_<uuid>; opaque catalog-material-<uuid> custody handle",
      "domain_separation": {
        "receipt_purpose": "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_RECEIPT_V1",
        "authority_domain": "cryptohunter.catalog-admission-receipt.production.v1",
        "lifecycle_purpose": "CRYPTOHUNTER_M0_12_CATALOG_RECEIPT_KEY_LIFECYCLE_V1",
        "finalization_purpose": "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_FINALIZATION_V1",
        "algorithm": "HMAC-SHA-256",
        "key_id_covered": true,
        "generation_covered": "receipt_sequence/finalization_sequence and key_revision in their respective authenticated structures"
      },
      "production_test_separation": "Exact production types, distinct production/test authority domains, production OS-keyring custody versus deterministic test seed/file anchor",
      "record_coverage": "Exact canonical receipt fields excluding receipt_mac; exact finalization fields excluding finalization_mac; lifecycle key set and revision",
      "preimage_fingerprint_domain": "purpose NUL authority_domain NUL canonical payload (receipt); purpose NUL canonical body (finalization); separate lifecycle purpose/domain",
      "verification_entry_point": "CatalogAdmissionReceiptAuthority.verify plus fail-closed _replay during construction and every operation",
      "restart_behavior": "Replays lifecycle, receipt and finalization chains, verifies all HMACs, then compares external anchor before returning authority"
    },
    {
      "mechanism": "Source producer membership integrity chain",
      "module": "bot_core.instruments.source_producer_membership",
      "class": "SourceProducerMembershipAuthority",
      "algorithm": "SHA-256 fingerprints/digest chain; NOT HMAC authentication",
      "key_owner": "NONE",
      "key_source": "NONE",
      "key_identifier": "NONE",
      "domain_separation": "Production carrier domain and record fingerprint literals",
      "production_test_separation": "Distinct carrier classes/domains and exact production type checks",
      "record_coverage": "Public record fingerprints and journal sequence/predecessor",
      "preimage_fingerprint_domain": "cryptohunter.m0.12.source-producer-membership-* and membership journal row domain",
      "verification_entry_point": "carrier read/_validated_rows and authority _replay",
      "restart_behavior": "Validates local SQLite chain/head and immutable release declarations; whole-database rollback remains out of scope"
    },
    {
      "mechanism": "Catalog runtime admission closure",
      "module": "bot_core.instruments.catalog_runtime_acceptance",
      "class": "CatalogRuntimeAcceptanceAuthority",
      "algorithm": "Catalog public SHA-256 chains plus CatalogAdmissionReceiptAuthority HMAC receipt/finalization",
      "key_owner": "CatalogAdmissionReceiptAuthority only",
      "key_source": "No separate runtime key",
      "key_identifier": "Receipt key_id",
      "domain_separation": "cryptohunter.catalog_runtime_acceptance.production.v1 plus catalog receipt purposes/domain",
      "production_test_separation": "Constructor requires exact production membership carrier/authority and exact production receipt authority",
      "record_coverage": "Snapshot commitment, membership commitment, receipt MAC and authenticated finalization bind durable snapshot sequence/digest",
      "preimage_fingerprint_domain": "Catalog-specific commitment and runtime domains",
      "verification_entry_point": "CatalogRuntimeAcceptanceAuthority._replay calls receipt replay and validates exact closure",
      "restart_behavior": "Validates domains/local chains, receipt authentication/freshness anchor, membership semantics and receipt-finalization closure before snapshots are exposed"
    }
  ],
  "m012_keyring_inventory": {
    "implementation": "bot_core.security.keyring_storage.KeyringSecretStorage",
    "catalog_service_name": "dudzian.catalog-admission-receipt",
    "catalog_namespace_prefix": "dudzian.catalog-admission-receipt.v1:",
    "secret_slots": "per-key opaque custody handles, lifecycle root handle, authority-anchor-v1",
    "owner": "CatalogAdmissionReceiptAuthority security substrate; not an account or reservation owner",
    "scope": "catalog-specific, not a generic application HMAC authority",
    "environment_specific": "Production uses native OS keyring and HWID-bound encrypted values; test authority uses deterministic seed and a sidecar anchor file",
    "keyring_anti_rollback": "NOT_FOUND; the Catalog anchor detects mismatch with current SQLite, but no independent monotonic protection for rollback of the entire OS keyring is evidenced"
  },
  "m012_freshness_inventory": {
    "primitive": "Catalog receipt authority external keyring anchor",
    "implementation": "SecretStorageCatalogAdmissionReceiptSecureCustody.read_anchor/write_anchor and _CatalogAdmissionReceiptAuthorityBase._validate_anchor/_publish_anchor",
    "storage_location": "OS keyring service dudzian.catalog-admission-receipt, slot dudzian.catalog-admission-receipt.v1:authority-anchor-v1; outside Catalog SQLite",
    "owner": "CatalogAdmissionReceiptAuthority through catalog-only custody",
    "covered_state": [
      "authority domain/purposes/algorithm",
      "key revision and lifecycle_state_mac",
      "receipt committed sequence/digest/last id",
      "finalization committed sequence/digest/last id"
    ],
    "update_protocol": "Validate/replay, commit SQLite transaction, then publish canonical anchor, then replay and compare; anchor write is after database COMMIT",
    "compare_protocol": "Reconstruct exact anchor mapping from SQLite and compare canonical bytes with hmac.compare_digest",
    "failure_behavior": "Missing, malformed or unequal enrolled anchor raises CatalogAdmissionReceiptAuthorityUnavailable; callers fail closed",
    "restart_behavior": "Constructor _replay validates anchor after HMAC and chain replay",
    "atomicity_with_local_journal": "NOT_ATOMIC: SQLite commit and keyring anchor publication are separate; crash/failure can cause availability failure, not stale-state acceptance",
    "valid_prefix_rollback": "Detected when SQLite receipt/finalization/lifecycle prefix is restored but newer keyring anchor remains",
    "actually_external": "External to mutable Catalog SQLite/journal, but local to the same machine/OS keyring; not a TPM or remote monotonic anchor",
    "coordinated_rollback": "SQLite plus historical keyring anchor rollback is explicitly outside the locally detectable boundary"
  },
  "threat_boundary": {
    "local_DB_rollback": "DETECTED only when the newer external keyring anchor remains",
    "journal_valid_prefix_rollback": "DETECTED for Catalog receipt/finalization lineage against retained anchor; membership-only whole-DB rollback is not independently detected",
    "coherent_local_rewrite": "HMAC forgery rejected; a rewrite retaining genuine old HMACs is rejected only if newer anchor remains",
    "keyring_rollback": "NOT_PROTECTED when coherently rolled back with SQLite",
    "filesystem_snapshot_rollback": "NOT_PROVEN; OS keyring may be outside a DB snapshot, but no general filesystem guarantee",
    "machine_wide_rollback": "NOT_PROTECTED / explicitly requires TPM or remote monotonic anchor",
    "authentication_is_freshness": false
  },
  "production_test_isolation": {
    "key_material": "Production random keys in OS keyring; test deterministic derived keys",
    "keyring_namespace": "Production catalog service/prefix; test has no OS-keyring namespace",
    "storage_path": "SQLite paths are caller-selected, but metadata authority domains reject cross-domain reuse; test anchor is a sidecar file",
    "authority_domain": "cryptohunter.catalog-admission-receipt.production.v1 versus cryptohunter.catalog-admission-receipt.test.v1",
    "freshness_anchor": "Production OS-keyring catalog slot versus test sidecar .anchor",
    "exact_type_gate": "Production receipt and runtime constructors reject test authority/store types",
    "TEST_authentication_material_accepted_in_PRODUCTION": false,
    "result": "PROVEN_FOR_M012_CATALOG_BOUNDARY; future account genesis still requires its own production/test domains and custody namespaces"
  },
  "domain_separation": {
    "catalog_hmac_as_account_genesis_hmac": "MUST_FAIL",
    "reason": "Catalog preimages and validators require Catalog-specific purpose, authority domain, record shape and key identifiers. No AccountGenesis validator exists and reuse of these literals is forbidden.",
    "future_account_domain": "DISTINCT_DOMAIN_REQUIRED / exact literal NOT_FROZEN",
    "m012_record_domain_reuse": "FORBIDDEN",
    "same_raw_key_safety": "NOT_AUTHORIZED_BY_CURRENT_DESIGN even with a new domain; current custody and lifecycle are catalog-only"
  },
  "key_reuse_policy": {
    "CROSS_AUTHORITY_KEY_REUSE_POLICY": "NOT_FROZEN",
    "same_raw_key_different_domain": "No explicit cross-authority policy or production API found",
    "decision": "Do not reuse Catalog raw keys or custody handles for AccountGenesis"
  },
  "key_rotation": {
    "rotation_owner": "CatalogAdmissionReceiptAuthority",
    "key_generation": "rotate creates random 32-byte key; prior ACTIVE becomes VERIFY_ONLY; key_revision increments and lifecycle is HMAC-sealed",
    "old_key_verification": "VERIFY_ONLY keys continue historical verification",
    "revocation": "REVOKED keys cause verify to reject receipts; lifecycle revision/anchor changes",
    "retirement": "No deletion/retirement state found",
    "restart_behavior": "Lifecycle HMAC, revision, key states and anchor are replayed/verified",
    "KEY_ROTATION_SEMANTICS": "FOUND_FOR_CATALOG_ONLY; account-genesis semantics NOT_FOUND"
  },
  "freshness_anchor_scope": {
    "model": "one catalog anchor slot -> one Catalog receipt/finalization/key-lifecycle lineage",
    "multiple_independent_journals": "NOT_SUPPORTED_BY_CURRENT_INTERFACE",
    "namespace_or_slot_parameter": "No caller-selectable authority lineage slot in production custody",
    "shared_global_monotonic_anchor": false,
    "separate_per_authority_slot_model": "PATTERN_CONCEPT_ONLY; current catalog-only class does not expose a generalized service",
    "account_lineage_direct_coverage": false
  },
  "cross_authority_anchor_analysis": {
    "scenario": "Catalog C10 remains while Account A8 is rolled back to A5",
    "current_anchor_detection": "NO: Catalog anchor contains no account lineage/generation/head",
    "single_shared_anchor_sufficient": false,
    "cross_authority_reuse": "NOT_PROVEN / NOT_COMPATIBLE as implemented",
    "concurrency": "No evidence for multi-lineage CAS, atomic slot update, race handling or lost-update prevention; the current anchor is a single catalog mapping written after Catalog SQLite commit",
    "required_future_property": "Independent authenticated namespace/slot and freshness comparison for each account authority lineage, or a newly designed atomic multi-lineage service"
  },
  "CAS_and_generation_reuse": {
    "expected_generation": "Receipt append derives next sequence under BEGIN IMMEDIATE; no caller-facing expected-generation CAS",
    "predecessor": "Receipt/finalization and runtime/membership chains bind previous digest; membership grants bind previous membership id/generation",
    "compare_and_set": "SQLite BEGIN IMMEDIATE plus replay/head checks provides authority-internal serialization, not a generalized CAS API",
    "transactional_sequence": "Yes within each local SQLite journal; Catalog snapshot and membership rows share a transaction, while receipt metadata and external anchor are separate boundaries",
    "account_genesis_result": "REUSABLE_PATTERN_ONLY; does not directly provide operation/reservation/account_id CAS"
  },
  "restart_order_parity": {
    "result": "PARTIAL_MATCH",
    "m012_observed_order": [
      "validate durable schemas/domains and replay local chains",
      "verify lifecycle and receipt/finalization HMAC authentication",
      "verify journal heads/closure",
      "compare external Catalog anchor",
      "reconstruct membership/Catalog semantic state",
      "verify receipt-finalization-to-snapshot reconciliation",
      "expose snapshots after constructor succeeds"
    ],
    "difference": "Catalog-specific stores/closure and reconstruction are split across collaborating authorities; no account operation/reservation/genesis reconciliation or account resolver publication exists"
  },
  "account_genesis_requirement_matrix": [
    {
      "requirement": "authenticated operation state",
      "classification": "REUSABLE_WITH_NEW_DOMAIN",
      "evidence": "HMAC canonical-envelope pattern is production proven, but record schema/key owner/domain must be new"
    },
    {
      "requirement": "authenticated reservation state",
      "classification": "REUSABLE_WITH_NEW_DOMAIN",
      "evidence": "Same primitive pattern can cover exact records only under a new authority-specific design"
    },
    {
      "requirement": "authenticated predecessor",
      "classification": "REUSABLE_WITH_NEW_DOMAIN",
      "evidence": "Receipt/finalization HMAC payloads cover previous digest and sequence"
    },
    {
      "requirement": "expected generation / CAS",
      "classification": "REUSABLE_PATTERN_ONLY",
      "evidence": "Internal SQLite serialization and sequences exist; no generalized expected-generation interface"
    },
    {
      "requirement": "valid-prefix rollback rejection",
      "classification": "REUSABLE_PATTERN_ONLY",
      "evidence": "Catalog-only keyring anchor proves the pattern; current single slot cannot cover an independent account lineage"
    },
    {
      "requirement": "freshness verification before resolver publication",
      "classification": "REUSABLE_PATTERN_ONLY",
      "evidence": "Catalog construction fails before snapshots are exposed; no account resolver implementation/path exists"
    },
    {
      "requirement": "TEST/PRODUCTION isolation",
      "classification": "REUSABLE_PATTERN_ONLY",
      "evidence": "Exact-type/domain/key-custody split is proven for Catalog and must be independently instantiated for AccountGenesis"
    }
  ],
  "authentication_reuse_result": {
    "result": "ACCOUNT_GENESIS_AUTH_PRIMITIVE_PATTERN_REUSE_ONLY",
    "production_authentication_primitive_found": true,
    "direct_reuse_available": false,
    "new_cryptographic_domain_required": true,
    "rationale": "The accepted HMAC construction is inseparable from catalog-specific custody, schema, purpose, type gates and owner; no production-capable generalized AccountGenesis path exists."
  },
  "freshness_reuse_result": {
    "result": "ACCOUNT_GENESIS_FRESHNESS_PRIMITIVE_PATTERN_REUSE_ONLY",
    "production_freshness_primitive_found": true,
    "independent_account_lineage_supported": false,
    "rationale": "The accepted external-to-SQLite compare pattern exists, but its one catalog slot covers only the Catalog lineage and is not machine-wide rollback resistant."
  },
  "authority_boundary": {
    "invariant": "REUSING A SECURITY PRIMITIVE MUST NOT EXPAND THE SOURCE AUTHORITY'S DOMAIN.",
    "CatalogRuntimeAcceptanceAuthority": "MUST_REMAIN_SCOPED",
    "CatalogAdmissionReceiptAuthority": "MUST_REMAIN_SCOPED",
    "SourceProducerMembershipAuthority": "MUST_REMAIN_SCOPED",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "primitive_availability_does_not_create_account_authority": true
  },
  "root_of_trust_impact": {
    "root_of_trust": "DESIGN_BLOCKED",
    "solved": false,
    "reason": "Authentication/freshness substrate does not decide who may create the first account."
  },
  "reservation_owner_impact": {
    "AccountIdReservationAuthority": "NOT_AVAILABLE",
    "solved": false,
    "m012_key_owner_is_reservation_owner": false,
    "reason": "Security substrate ownership is not domain authority ownership."
  },
  "mandatory_redteam": {
    "catalog_HMAC_as_AccountGenesis_without_distinct_domain": "FAIL",
    "TEST_authentication_material_in_PRODUCTION": "FAIL",
    "valid_HMAC_implies_fresh_state": "FAIL",
    "local_journal_and_local_anchor_rollback_considered_safe": "FAIL",
    "single_shared_anchor_assumed_for_independent_lineages": "FAIL",
    "primitive_reuse_expands_CatalogRuntimeAcceptanceAuthority": "FAIL",
    "m012_key_owner_implies_reservation_owner": "FAIL",
    "coherent_local_rewrite_with_stale_freshness_accepted": "FAIL"
  },
  "result": {
    "primary_result": "M05_ACCOUNT_GENESIS_TRUST_PRIMITIVES_REUSE_DISCOVERY_COMPLETE",
    "authentication": "ACCOUNT_GENESIS_AUTH_PRIMITIVE_PATTERN_REUSE_ONLY",
    "freshness": "ACCOUNT_GENESIS_FRESHNESS_PRIMITIVE_PATTERN_REUSE_ONLY",
    "account_authority_owner": "DESIGN_BLOCKED",
    "production_M0.5": "NOT_AVAILABLE"
  },
  "implementation_allowed": {
    "CryptoHunterAccountAuthority": false,
    "AccountIdReservationAuthority": false,
    "ProvisioningBoundary": false,
    "WorkspaceAuthority": false,
    "InstrumentAuthority": false,
    "WCP Authority": false,
    "FullFillAuthority": false,
    "M0.8": false
  },
  "preserved_status": {
    "M0.12": "ACCEPTED / AVAILABLE",
    "AcceptedSourceProducerMembership": "ACCEPTED / AVAILABLE",
    "CatalogAdmissionReceiptAuthority": "ACCEPTED / AVAILABLE",
    "CatalogRuntimeAcceptanceAuthority": "ACCEPTED / AVAILABLE",
    "M0.7 structural Full Fill v2": "ACCEPTED / STRUCTURAL AVAILABLE",
    "Account genesis reservation state model": "GIT current-tree design review / ACCOUNT_GENESIS_RESERVATION_STATE_MODEL_DESIGN_BLOCKED",
    "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
    "WorkspaceAuthority": "NOT_AVAILABLE",
    "InstrumentAuthority": "NOT_AVAILABLE",
    "WCP Authority": "NOT_AVAILABLE / DESIGN_BLOCKED",
    "FullFillAuthority": "NOT_AVAILABLE",
    "production M0.5": "NOT_AVAILABLE",
    "M0.8": "NOT_AVAILABLE",
    "C25": "BLOCKED",
    "S9D": "OPEN"
  }
}
```
