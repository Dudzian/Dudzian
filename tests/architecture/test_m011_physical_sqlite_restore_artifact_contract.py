"""Frozen architecture assertions for physical SQLite restore candidates."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
MACHINE_PATH = (
    ROOT
    / "docs/architecture/cryptohunter_product_architecture"
    / "persistence_versioning_migrations_backup_and_recovery.json"
)
MACHINE = json.loads(MACHINE_PATH.read_text())
CONTRACT = MACHINE["physical_sqlite_restore_artifact_contract"]
MANIFEST_FIELDS = [
    "artifact_format_version",
    "account_id",
    "device_installation_id",
    "environment",
    "state_store_identity_fingerprint_sha256",
    "state_store_schema_version",
    "local_protected_freshness_generation",
    "state_fingerprint_sha256",
    "transaction_fingerprint_sha256",
    "backup_envelope_fingerprint_sha256",
    "physical_artifact_sha256",
    "physical_artifact_byte_length",
    "sqlite_schema_fingerprint_sha256",
    "manifest_fingerprint_sha256",
]


def test_trust_root_audit_is_historical_and_implementation_remains_pending() -> None:
    assert CONTRACT["status"] == (
        "BACKUP_ARTIFACT_AUTHENTICATION_AUTHORITY_ARCHITECTURE_DEFINED_IMPLEMENTATION_PENDING"
    )
    audit = CONTRACT["trust_root_audit"]
    assert audit["decision"] == "NO_SUITABLE_EXISTING_INDEPENDENT_TRUST_ROOT"
    assert audit["reuse_decision"].startswith("NONE;")
    missing = audit["missing_prerequisite"]
    assert missing["name"] == "BackupArtifactAuthenticationAuthority"
    assert set(missing["required_properties"]) == {
        "PRE_EXISTING",
        "INDEPENDENT_OF_RESTORE_CANDIDATE",
        "CANDIDATE_CANNOT_MINT_OR_REPLACE",
        "NO_RAW_KEY_EXPOSURE",
        "PURPOSE_SEPARATED",
    }
    assert "invented signing or MAC key" in audit["forbidden_shortcuts"]


def test_verification_authority_is_external_to_four_component_candidate_bundle() -> None:
    model = CONTRACT["artifact_model"]
    assert len(model["outer_bundle_components"]) == 4
    assert "BackupArtifactAuthenticationAuthorityCapability" not in model["outer_bundle_components"]
    assert model["contains_candidate_carried_data_and_proof_only"] is True
    assert model["may_carry_trust_authority_or_capability"] is False
    boundary = CONTRACT["verification_input_boundary"]
    assert boundary["invariant"] == (
        "RESTORE_CANDIDATE_MAY_CARRY_AUTHENTICATION_PROOF_BUT_NEVER_AUTHENTICATION_AUTHORITY"
    )
    assert boundary["authentication_proof"]["candidate_carried"] is True
    capability = boundary["verification_capability"]
    assert capability["classification"] == (
        "PRE_EXISTING_EXTERNAL_LOCAL_SECURITY_AUTHORITY_CAPABILITY"
    )
    assert capability["production_exists"] is False
    assert capability["process_local"] is True
    assert capability["pre_existing"] is True
    assert capability["durable"] is False
    assert capability["candidate_carried"] is False
    assert capability["serializable_into_bundle"] is False
    assert capability["acquisition"] == "INDEPENDENTLY_FROM_RESTORE_CANDIDATE"
    assert capability["unavailable"] == "RESTORE_REJECTED"


def test_post_structural_restore_requires_physical_artifact_not_semantic_replay() -> None:
    policy = CONTRACT["materialization_policy"]
    assert policy["invariants"] == [
        "POST_STRUCTURAL_MIGRATION_RESTORE_REQUIRES_TRUSTED_PHYSICAL_STATE_ARTIFACT",
        "BACKUP_ENVELOPE_ALONE_CANNOT_RECONSTRUCT_ARBITRARY_MIGRATION_DDL_DML_EFFECTS",
        "RESTORE_MUST_NOT_REPLAY_MIGRATION_SQL_TO_MANUFACTURE_MATERIALIZATION",
    ]
    assert policy["restore_executes_migration_SQL"] is False
    assert CONTRACT["failure_policy"]["no_semantic_only_fallback_post_structural"] is True


def test_outer_bundle_keeps_database_bytes_out_of_semantic_records() -> None:
    model = CONTRACT["artifact_model"]
    assert model["bundle_name"] == "TrustedPhysicalBackupArtifact"
    assert model["outer_bundle_components"] == [
        "semantic BackupEnvelope",
        "self-contained physical SQLite database artifact",
        "immutable PhysicalSQLiteArtifactManifest",
        "PhysicalArtifactAuthenticationProof",
    ]
    assert model["raw_sqlite_bytes_are_PersistenceRecord"] is False
    assert model["physical_artifact_is_M0_3_authority"] is False
    assert model["candidate_only"] is True


def test_manifest_has_exact_minimum_fields_and_canonical_fingerprint() -> None:
    manifest = CONTRACT["manifest"]
    assert manifest["additional_fields"] is False
    assert manifest["exact_fields"] == MANIFEST_FIELDS
    assert manifest["timestamp_authority"] is False
    assert manifest["fingerprint"] == {
        "field": "manifest_fingerprint_sha256",
        "projection": "all exact manifest fields except manifest_fingerprint_sha256",
        "algorithm": "SHA-256",
        "encoding": "UTF-8",
        "canonical_json": {
            "sort_keys": True,
            "separators": [",", ":"],
            "ensure_ascii": False,
            "allow_nan": False,
        },
        "digest": "lowercase hexadecimal",
        "meaning": "STRUCTURAL_BINDING_ONLY",
    }


def test_semantic_manifest_and_physical_bytes_bind_one_consistency_point() -> None:
    snapshot = CONTRACT["consistent_snapshot"]
    assert snapshot["required_class"] == "SQLITE_CONSISTENT_SNAPSHOT_PRIMITIVE"
    assert (
        "same exact durable StateStore generation, state fingerprint, and transaction fingerprint"
        in snapshot["semantic_physical_rule"]
    )
    assert snapshot["mismatched_consistency_points"] == "RESTORE_REJECTED"
    assert "physical bytes/length" in CONTRACT["manifest"]["bindings"]


def test_live_wal_copy_is_forbidden_and_output_has_no_sidecar_dependency() -> None:
    snapshot = CONTRACT["consistent_snapshot"]
    assert snapshot["output"] == "SELF_CONTAINED_DATABASE_IMAGE_NO_EXTERNAL_WAL_DEPENDENCY"
    assert snapshot["forbidden"] == [
        "naive filesystem copy of a live WAL-mode database",
        "copying only main database while committed state may remain in WAL",
        "concurrent database-page reads without a SQLite-consistent snapshot primitive",
    ]
    sidecars = CONTRACT["sqlite_validation"]["wal_sidecar_policy"]
    assert sidecars["source_WAL_or_SHM_authoritative_input"] is False
    assert sidecars["candidate_may_attach_live_location_sidecars"] is False


def test_hashes_are_integrity_only_and_proof_is_authenticity_gate() -> None:
    boundary = CONTRACT["integrity_authenticity"]
    assert boundary["physical_artifact_sha256"] == "BYTE_INTEGRITY_ONLY"
    assert boundary["manifest_fingerprint_sha256"] == "STRUCTURAL_BINDING_ONLY"
    assert boundary["backup_envelope_fingerprint_sha256"] == "CANDIDATE_INTRINSIC_INTEGRITY_ONLY"
    assert boundary["independent_authentication_proof"] == "PHYSICAL_ARTIFACT_AUTHENTICITY_GATE"
    assert boundary["hash_possession_grants_authority"] is False
    assert boundary["candidate_may_carry_or_mint_trust_root"] is False


def test_authenticated_candidate_still_grants_no_lifecycle_or_live_authority() -> None:
    denied = set(CONTRACT["integrity_authenticity"]["valid_proof_does_not_establish"])
    assert denied == {
        "M0.3 authority",
        "current protected membership",
        "current lifecycle authority",
        "SecretHandoff external authority",
        "restore promotion",
        "LIVE readiness",
    }
    assert CONTRACT["restore_workflow"]["stage_2_stage_3_m0_3_remain_conjunctive"] is True


def test_physical_admission_is_outer_eligibility_prerequisite_not_authority_stage() -> None:
    admission = CONTRACT["physical_artifact_admission_boundary"]
    stages = CONTRACT["restore_workflow"]["canonical_authority_stages"]
    assert admission["classification"] == ("PRE_LIFECYCLE_STAGE_ARTIFACT_ADMISSION_PREREQUISITE")
    assert admission["lifecycle_authority_stage"] is False
    assert set(stages) <= set(admission["not"])
    assert admission["successful_admission_grants"] == (
        "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
    )
    assert set(admission["successful_admission_does_not_grant"]) == {
        "restore promotion",
        "current membership",
        "lifecycle authority",
        "M0.3 authority",
        "SecretHandoff authority",
        "LIVE readiness",
        "business-resume permission",
    }
    assert admission["successful_authentication_establishes_restore_authority"] is False
    assert admission["reorders_four_lifecycle_stages"] is False


def test_authentication_precedes_sqlite_open_and_failure_rejects_without_fallback() -> None:
    admission = CONTRACT["physical_artifact_admission_boundary"]
    steps = admission["ordered_steps"]
    names = [step["step"] for step in steps]
    proof = names.index("EXTERNAL_PROOF_VERIFICATION")
    physical = names.index("AUTHENTICATED_CANDIDATE_INTRINSIC_PHYSICAL_VALIDATION")
    assert proof < physical
    assert all(not step["opens_SQLite_database"] for step in steps[:physical])
    assert steps[physical]["opens_SQLite_database"] is True
    assert admission["invariants"] == [
        "UNAUTHENTICATED_PHYSICAL_SQLITE_BYTES_MUST_NOT_BE_OPENED_AS_A_RESTORE_DATABASE",
        "PHYSICAL_ARTIFACT_ADMISSION_DOES_NOT_CHANGE_STAGE_1_AUTHORITY_CLASSIFICATION",
    ]
    assert admission["authentication_failure_or_unavailability"] == "RESTORE_REJECTED"
    assert admission["fallback"] is False
    assert steps[proof]["candidate_self_authentication"] is False


def test_authenticated_intrinsic_physical_validation_computes_facts_not_authority() -> None:
    admission = CONTRACT["physical_artifact_admission_boundary"]
    intrinsic = admission["intrinsic_physical_validation"]
    assert intrinsic == {
        "classification": "AUTHENTICATED_CANDIDATE_INTRINSIC_PHYSICAL_VALIDATION",
        "authority": False,
        "establishes_current_authorization": False,
    }
    workflow = CONTRACT["restore_workflow"]
    assert "physical_verification_role" not in workflow
    assert workflow["physical_artifact_authentication_role"] == (
        "PRE_LIFECYCLE_STAGE_ADMISSION_AUTHENTICITY_GATE"
    )
    assert workflow["successful_authentication_establishes_restore_authority"] is False


def test_sqlite_validation_requires_full_integrity_and_exact_semantics() -> None:
    validation = CONTRACT["sqlite_validation"]
    assert validation["context"] == "ISOLATED_FILESYSTEM_CONTEXT_NEVER_LIVE_STORE_LOCATION"
    assert validation["full_integrity_check"].startswith(
        "PRAGMA integrity_check MUST return exactly ok"
    )
    assert validation["quick_check_alone_sufficient"] is False
    required = set(validation["required"])
    assert {
        "full SQLite integrity succeeds",
        "expected StateStore database identity is present",
        "exact account_id, device_installation_id, and environment",
        "exact state_store_schema_version",
        "exact sqlite_schema_fingerprint_sha256",
        "semantic snapshot exactly matches BackupEnvelope bindings",
        "exact generation, state fingerprint, and transaction fingerprint",
        "no unexpected attached database dependency",
    } <= required


def test_existing_schema_algorithm_is_reused_but_does_not_authenticate_dml() -> None:
    schema = CONTRACT["sqlite_validation"]["schema_fingerprint"]
    assert schema["algorithm"] == "REUSE_EXISTING_MIGRATION_SQLITE_SCHEMA_FINGERPRINT"
    assert schema["candidate_only_derivation_forbidden"] is True
    assert schema["authenticates_arbitrary_DML_rows"] is False
    assert schema["candidate_computation_before_stage_2"] == (
        "ALLOWED_INTRINSIC_FACT_COLLECTION_ONLY"
    )
    assert schema["MigrationExecutionAuthority_comparison_before_stage_2"] is False
    assert schema["authoritative_comparison_location"].endswith(
        "/MIGRATION_RESTORE_AUTHORITY_REVALIDATION"
    )
    assert (
        "exact physical artifact SHA-256" in CONTRACT["materialization_policy"]["dml_authenticity"]
    )


def test_stage_2_composition_places_all_applicable_components_before_stage_3() -> None:
    workflow = CONTRACT["restore_workflow"]
    assert workflow["canonical_authority_stages"] == [
        "STRUCTURAL_PERSISTENCE_RECORD_VALIDATION",
        "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
        "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
        "M0_3_RESTORE_FRESHNESS",
    ]
    composition = workflow["stage_2_composition"]
    assert composition["stage"] == "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION"
    migration, secret_handoff = composition["components"]
    assert migration["name"] == "MIGRATION_RESTORE_AUTHORITY_REVALIDATION"
    assert migration["applicability"] == "Migration family records require authority"
    assert migration["authority_checks"] == [
        "MigrationDefinition binding",
        "MigrationExecutionAuthority binding",
        "operation-plan fingerprint binding where applicable",
        "independently sealed pre/target sqlite schema fingerprint comparison",
        "required migration declaration/materialization relation",
        "candidate-computed SQLite schema fingerprint compared with independently sealed expected fingerprint appropriate to lifecycle/materialization state",
    ]
    assert migration["authority_source"] == (
        "PRE_EXISTING_SEALED_MIGRATION_AUTHORITY_NEVER_CANDIDATE_DERIVED"
    )
    assert migration["migration_SQL_execution"] is False
    assert migration["reconstructs_authority_from_candidate"] is False
    assert migration["schema_fingerprint_authenticates_arbitrary_DML_rows"] is False
    assert secret_handoff == {
        "name": "SECRET_HANDOFF_INITIAL_STAGE_2",
        "applicability": "one or more SecretHandoff family records exist",
        "action": "freshly observe every candidate SecretHandoff descriptor through pre-existing read-only external authority",
    }
    assert composition["invariants"] == [
        "ALL_APPLICABLE_STAGE_2_AUTHORITY_COMPONENTS_COMPLETE_BEFORE_STAGE_3",
        "INITIAL_STAGE_2_SECRET_HANDOFF_OBSERVATION_IS_PART_OF_STAGE_2_NOT_A_POST_STAGE_3_FENCE",
    ]
    assert composition["invocation_count"]["SECRET_HANDOFF_INITIAL_STAGE_2"] == (
        "EXACTLY_ONCE_WHEN_APPLICABLE_PER_RESTORE_PATH"
    )
    assert composition["not_new_restore_stage"] is True
    steps = workflow["ordered_steps"]
    admission = steps.index(
        "complete physical_artifact_admission_boundary when physical artifact is required"
    )
    stage_1 = steps.index(
        "perform canonical Stage-1 STRUCTURAL_PERSISTENCE_RECORD_VALIDATION intrinsic semantic candidate/carrier validation"
    )
    stage_2 = steps.index(
        "perform every applicable Stage-2 component defined by stage_2_composition"
    )
    stage_3 = steps.index("perform current-designation Stage-3 relational validation")
    m03 = steps.index("perform pre-install M0.3 restore authority/freshness assessment")
    assert admission < stage_1 < stage_2 < stage_3 < m03


def test_pre_stage_2_migration_fact_collection_never_consults_authority() -> None:
    policy = CONTRACT["restore_workflow"]["pre_stage_2_candidate_fact_policy"]
    assert policy["allowed"] == [
        "compute candidate sqlite_schema_fingerprint_sha256",
        "validate intrinsic physical structure",
        "collect candidate migration/materialization facts",
    ]
    assert policy["forbidden"] == [
        "accept candidate schema as authorized because it matches candidate-carried data",
        "compare candidate materialization against MigrationExecutionAuthority as an authority decision",
        "treat migration materialization as authority-validated",
    ]
    assert policy["MigrationExecutionAuthority_consulted_for_authority_decision"] is False


def test_m0_3_assessment_remains_fourth_authority_stage() -> None:
    workflow = CONTRACT["restore_workflow"]
    interpretation = workflow["authority_stage_interpretation"]
    assert interpretation["M0_3_RESTORE_FRESHNESS"] == (
        "PRE_INSTALL_M0_3_RESTORE_AUTHORITY_AND_FRESHNESS_ASSESSMENT"
    )
    assert interpretation["filesystem_install_location_defines_authority_stage_order"] is False
    assessment = workflow["pre_install_M0_3_assessment"]
    assert assessment["outcomes"] == [
        "DENIED",
        "AUTHORIZED_COMMITTED_TARGET",
        "AUTHORIZED_PREPARED_TARGET",
    ]
    assert assessment["new_M0_3_states"] is False
    assert assessment["new_PREPARE_or_ABORT"] is False


def test_ordinary_installing_path_has_stage_2_before_stage_3_and_exact_fences() -> None:
    workflow = CONTRACT["restore_workflow"]
    path = workflow["ordinary_installing_path"]
    admission = path.index("complete physical artifact admission prerequisite when required")
    stage_1 = path.index(
        "perform canonical Stage-1 intrinsic semantic candidate/carrier validation"
    )
    initial_in_stage_2 = path.index(
        "perform all applicable Stage-2 authority components including exactly one INITIAL_STAGE_2 SecretHandoff observation when applicable"
    )
    stage_3 = path.index("current-designation Stage-3 relational validation")
    m03 = path.index(
        "pre-install M0.3 restore authority/freshness assessment passes as AUTHORIZED_COMMITTED_TARGET"
    )
    pre_install = path.index(
        "PRE_INSTALL SecretHandoff freshness fence when applicable immediately before atomic install"
    )
    install = path.index("atomic candidate install")
    reopen = path.index("reopen installed StateStore and verify required local facts")
    final = path.index("FINAL_PROMOTION SecretHandoff freshness fence when applicable")
    assert (
        admission
        < stage_1
        < initial_in_stage_2
        < stage_3
        < m03
        < pre_install
        < install
        < reopen
        < final
    )
    assert pre_install + 1 == install
    assert workflow["ordinary_installing_path_forbidden_M0_3_actions"] == ["PREPARE", "ABORT"]


def test_existing_prepared_installing_path_has_stage_2_before_stage_3() -> None:
    workflow = CONTRACT["restore_workflow"]
    path = workflow["existing_external_M0_3_PREPARED_installing_path"]
    admission = path.index("complete physical artifact admission prerequisite when required")
    stage_1 = path.index(
        "perform canonical Stage-1 intrinsic semantic candidate/carrier validation"
    )
    initial_in_stage_2 = path.index(
        "perform all applicable Stage-2 authority components including exactly one INITIAL_STAGE_2 SecretHandoff observation when applicable"
    )
    stage_3 = path.index("current-designation Stage-3 relational validation")
    m03 = path.index("pre-install M0.3 assessment identifies AUTHORIZED_PREPARED_TARGET")
    pre_install = path.index(
        "PRE_INSTALL SecretHandoff freshness fence when applicable immediately before atomic install"
    )
    install = path.index("atomic candidate install")
    reopen = path.index("reopen installed StateStore and verify required local facts")
    before_finalize = path.index(
        "fresh FINAL_PROMOTION-type SecretHandoff observation immediately before existing M0.3 recovery/finalization"
    )
    finalize = path.index("perform existing M0.3 recovery/finalization")
    after_finalize = path.index(
        "after successful finalization freshly invoke FINAL_PROMOTION again immediately before restore promotion/readiness/evidence release"
    )
    promotion = path.index("promotion/readiness only after every remaining gate succeeds")
    assert (
        admission
        < stage_1
        < initial_in_stage_2
        < stage_3
        < m03
        < pre_install
        < install
        < reopen
        < before_finalize
        < finalize
        < after_finalize
        < promotion
    )
    assert pre_install + 1 == install
    rules = workflow["existing_external_M0_3_PREPARED_rules"]
    assert rules["pre_finalization_observation_reuse_at_final_promotion"] == "FORBIDDEN"
    assert rules["fence_types"] == ["INITIAL_STAGE_2", "PRE_INSTALL", "FINAL_PROMOTION"]
    assert rules["fourth_fence_type_created"] is False
    assert rules["cross_resource_ACID"] is False
    assert rules["completed_M0_3_finalization_rollback_if_later_fence_fails"] == (
        "NOT_CLAIMED_AND_FORBIDDEN"
    )


def test_noop_defers_to_existing_contract_without_install_or_duplicate_sequence() -> None:
    workflow = CONTRACT["restore_workflow"]
    noop = workflow["NOOP_ALREADY_CURRENT_existing_external_M0_3_PREPARED"]
    assert noop["physical_artifact_role"] == "NOT_INVOLVED_IN_TRUE_NOOP_PATH"
    assert noop["duplicate_competing_NOOP_contract"] is False
    assert noop["authority_prefix"] == [
        "perform canonical Stage-1 intrinsic semantic candidate/carrier validation",
        "perform all applicable Stage-2 authority components including exactly one INITIAL_STAGE_2 SecretHandoff observation when applicable",
        "perform current-designation Stage-3 relational validation",
        "perform M0.3 restore assessment/freshness",
    ]
    assert noop["PRE_INSTALL"] == "NOT_APPLICABLE_NO_INSTALL_OCCURS"
    assert noop["atomic_install"] == "NOT_APPLICABLE"
    assert noop["physical_artifact_materialization"] == "NOT_APPLICABLE"
    existing = MACHINE["backup_contract"]["restore_lifecycle_authority_contract"][
        "secret_handoff_restore_authority"
    ]["freshness_and_fencing_contract"]["NOOP_ALREADY_CURRENT"]
    assert existing["existing_external_M0_3_PREPARED_sequence"] == [
        "INITIAL_STAGE_2",
        "fresh FINAL_PROMOTION invocation immediately before M0.3 recovery/finalization",
        "fresh FINAL_PROMOTION invocation again after successful finalization and immediately before successful NOOP return or readiness",
    ]
    assert existing["prepared_path_fence_type_rule"] == (
        "same FINAL_PROMOTION fence TYPE invoked twice; no fourth fence TYPE"
    )
    assert noop["pre_finalization_observation_reuse"] == "FORBIDDEN"


def test_physical_workflow_agrees_with_lifecycle_and_secret_handoff_contracts() -> None:
    workflow = CONTRACT["restore_workflow"]
    lifecycle = MACHINE["backup_contract"]["restore_lifecycle_authority_contract"]
    responsibilities = lifecycle["ordered_stage_responsibilities"]
    freshness = lifecycle["secret_handoff_restore_authority"]["freshness_and_fencing_contract"]
    assert workflow["canonical_authority_stages"] == responsibilities["ordered_stages"]
    assert (
        CONTRACT["physical_artifact_admission_boundary"]["classification"]
        not in (responsibilities["ordered_stages"])
    )
    assert responsibilities["STRUCTURAL_PERSISTENCE_RECORD_VALIDATION"] == (
        "intrinsic carriers and envelope only"
    )
    assert responsibilities["ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION"] == [
        "pre-existing sealed Migration authority",
        "pre-existing read-only SecretHandoff external authority",
    ]
    roles = workflow["secret_handoff_fence_roles"]
    assert roles["INITIAL_STAGE_2"]["stage_membership"] == (
        "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION"
    )
    assert roles["INITIAL_STAGE_2"]["before"] == "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION"
    assert roles["PRE_INSTALL"]["stage_membership"] is False
    assert roles["FINAL_PROMOTION"]["stage_membership"] is False
    assert roles["later_repeated_observations_reorder_canonical_authority_stages"] is False
    assert freshness["required_restore_time_fences"]["ordered_names"] == [
        "INITIAL_STAGE_2",
        "PRE_INSTALL",
        "FINAL_PROMOTION",
    ]
    assert freshness["required_restore_time_fences"]["INITIAL_STAGE_2"]["timing"] == (
        "before M0.3 freshness"
    )


def test_atomic_install_never_exposes_partial_or_unverified_store() -> None:
    install = CONTRACT["atomic_install"]
    assert install["candidate_completely_validated_before_replacement"] is True
    assert install["pre_install_failure_live_effect"] == "UNCHANGED"
    assert install["partially_copied_database_may_become_live"] is False
    assert install["installed_store_reopened_and_verified_before_readiness"] is True
    assert install["cross_resource_ACID"] is False
    assert install["external_SecretHandoff_or_M0_3_atomic_with_filesystem_replacement"] is False
    assert set(install["install_itself_grants"].values()) == {False}
    assert install["post_install_failure"] == (
        "MAY_LEAVE_VALID_INSTALLED_LOCAL_CANDIDATE_NOT_PROMOTED_NOT_READY"
    )
    assert install["post_install_rollback_guarantee"] is False


def test_backup_creation_is_same_point_and_all_or_nothing() -> None:
    creation = CONTRACT["backup_creation"]
    assert creation["ordered_steps"] == [
        "obtain one physical image with SQLITE_CONSISTENT_SNAPSHOT_PRIMITIVE",
        "derive BackupEnvelope from the same represented durable consistency point or prove exact generation/state/transaction identity",
        "compute physical_artifact_sha256 and byte length",
        "build canonical immutable PhysicalSQLiteArtifactManifest",
        "compute manifest_fingerprint_sha256",
        "obtain authenticate capability for exact trusted authority scope",
        "HMAC exact domain-separated canonical manifest payload with current ACTIVE key",
        "build PhysicalArtifactAuthenticationProof",
        "output TrustedPhysicalBackupArtifact only after every component succeeds",
    ]
    assert creation["mixed_generation_output_valid"] is False
    assert creation["authentication_authority_missing_or_unavailable"] == "BACKUP_CREATION_FAILED"
    assert creation["mutate_business_state_solely_to_create_backup"] is False


def test_authentication_and_all_binding_failures_reject_without_repair() -> None:
    failure = CONTRACT["failure_policy"]
    assert failure["outcome"] == "RESTORE_REJECTED"
    assert failure["no_repair"] is True
    assert failure["no_SQL_replay"] is True
    assert failure["no_authority_minting"] is True
    conditions = set(failure["fail_closed_conditions"])
    assert {
        "authentication proof missing",
        "authentication authority unavailable",
        "authentication proof invalid",
        "BackupEnvelope and manifest mismatch",
        "manifest and physical database mismatch",
        "generation mismatch",
        "state fingerprint mismatch",
        "transaction fingerprint mismatch",
        "required migration materialization mismatch",
    } <= conditions


def test_representation_classification_keeps_raw_secrets_and_authority_out() -> None:
    classification = CONTRACT["representation_classification"]
    assert classification["raw_secret_payload_field_count"] == 0
    representations = classification["representations"]
    for name, item in representations.items():
        assert item["contains_raw_secrets"] is False, name
        if name != "BackupArtifactAuthenticationAuthorityCapability":
            assert item["authority"] is False, name
            assert item["restorable_authority"] is False, name
    capability = representations["BackupArtifactAuthenticationAuthorityCapability"]
    assert capability["durable"] is False
    assert capability["candidate_carried"] is False
    assert capability["permitted_in_BackupEnvelope"] is False
    assert capability["permitted_in_StateStore"] is False
