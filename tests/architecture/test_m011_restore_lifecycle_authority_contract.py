"""Frozen architecture assertions for M0.11 lifecycle restore authority."""

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
CONTRACT = MACHINE["backup_contract"]["restore_lifecycle_authority_contract"]
EXPECTED_RESTORE_STAGES = [
    "STRUCTURAL_PERSISTENCE_RECORD_VALIDATION",
    "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
    "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
    "M0_3_RESTORE_FRESHNESS",
]
FRESHNESS = CONTRACT["secret_handoff_restore_authority"]["freshness_and_fencing_contract"]


def test_bundle_is_pre_existing_process_local_and_never_candidate_derived() -> None:
    bundle = CONTRACT["bundle"]
    assert bundle["name"] == "RestoreLifecycleAuthorityBundle"
    assert bundle["classification"] == "PROCESS-LOCAL / PRE-EXISTING TRUSTED INPUT"
    assert bundle["contains"] == [
        "migration_restore_authority",
        "secret_handoff_restore_authority",
    ]
    assert bundle["candidate_may_populate"] is False
    assert set(bundle["not"]) == {
        "PersistenceRecord",
        "BackupEnvelope",
        "durable StateStore state",
        "candidate-derived",
        "restore result",
        "M0.3 authority",
    }
    assert (
        "RestoreLifecycleAuthorityBundle"
        not in MACHINE["backup_contract"]["representation_registry"]
    )


def test_authority_is_optional_only_when_corresponding_family_is_absent() -> None:
    policy = CONTRACT["bundle"]["family_presence_policy"]
    assert policy == {
        "records_absent_and_corresponding_authority_absent": "ALLOWED_GATE_NOT_APPLICABLE",
        "any_corresponding_records_present_and_authority_absent": "RESTORE_REJECTED",
        "any_corresponding_records_present_and_authority_unavailable": "RESTORE_REJECTED",
    }


def test_migration_execution_authority_is_independent_and_exact() -> None:
    migration = CONTRACT["migration_restore_authority"]
    execution = migration["MigrationExecutionAuthority"]
    assert migration["required_components"] == [
        "MigrationDefinition",
        "MigrationExecutionAuthority",
        "trusted deterministic planner",
    ]
    assert execution["classification"] == "EXACT IMMUTABLE PRE-EXISTING NON-DURABLE AUTHORITY"
    assert execution["fields"] == [
        "migration_id",
        "source_schema_version",
        "target_schema_version",
        "ordered_path",
        "rollback_policy",
        "migration_definition_fingerprint_sha256",
        "operation_plan_fingerprint_sha256",
        "pre_sqlite_schema_fingerprint_sha256",
        "target_sqlite_schema_fingerprint_sha256",
    ]
    assert execution["exact_operations_required_in_authority"] is False
    assert execution["hash_possession_grants_authority"] is False
    assert set(execution["excluded_from"]) == {
        "PersistenceRecord",
        "BackupEnvelope",
        "StateStore",
        "restore candidate",
    }


def test_candidate_reconstructed_plan_cannot_supply_execution_authority() -> None:
    boundary = CONTRACT["migration_restore_authority"]["declaration_authority_boundary"]
    assert boundary["candidate_operations_are_independent_authority"] is False
    assert boundary["candidate_reconstructed_plan_is_authority"] is False
    assert boundary["assert_matches_definition_and_candidate_reconstructed_plan_proves_only"] == [
        "declaration intrinsic consistency",
        "static MigrationDefinition equality",
    ]
    assert "exact MigrationExecutionAuthority" in boundary["required_conjunctive_bindings"]
    assert boundary["unknown_migration_id"] == "RESTORE_REJECTED"
    assert boundary["known_definition_missing_execution_authority"] == "RESTORE_REJECTED"
    assert boundary["execution_authority_declaration_mismatch"] == "RESTORE_REJECTED"


def test_runtime_planner_must_match_the_same_sealed_authority_before_effects() -> None:
    registry = CONTRACT["migration_restore_authority"]["future_registry_contract"]
    assert registry["per_migration_id"] == [
        "MigrationDefinition",
        "MigrationExecutionAuthority",
        "trusted deterministic planner",
    ]
    assert (
        "before M0.3 PREPARE, local migration SQL, or declaration persistence"
        in registry["planner_result_gate"]
    )
    assert registry["candidate_specific_plan_without_sealed_match"] == "NOT_RESTORE_AUTHORIZABLE"


def test_migration_declaration_cardinality_is_state_and_predecessor_aware() -> None:
    cardinality = CONTRACT["migration_restore_authority"][
        "declaration_cardinality_by_current_state"
    ]
    assert cardinality["PREPARED"]["allowed_counts"] == [0]
    assert cardinality["APPLYING"]["allowed_counts"] == [0, 1]
    assert cardinality["DURABLE_MIGRATED"]["allowed_counts"] == [1]
    assert cardinality["COMPLETED"]["allowed_counts"] == [1]
    assert cardinality["FAILED"]["by_predecessor"] == {
        "PREPARED": [0],
        "APPLYING": [0, 1],
        "DURABLE_MIGRATED": [1],
    }
    assert set(CONTRACT["migration_restore_authority"]["cardinality_failures"].values()) == {
        "RESTORE_REJECTED"
    }


def test_migration_relations_remain_necessary_but_insufficient() -> None:
    migration = CONTRACT["migration_restore_authority"]
    assert len(migration["local_relational_requirements"]) == 12
    assert migration["local_relations_are_sufficient_authority"] is False
    assert migration["MigrationRecord"] == {
        "durability": "EPHEMERAL ONLY",
        "restore_authority": False,
        "BackupEnvelope_membership": False,
    }


def test_post_structural_restore_requires_a_future_physical_artifact() -> None:
    physical = CONTRACT["physical_sqlite_restore_prerequisite"]
    assert (
        physical["current_BackupEnvelope_preserves_complete_physical_SQLite_database_image"]
        is False
    )
    assert physical["migration_execution_effect_kinds"] == ["DDL", "DML"]
    assert physical["invariants"] == [
        "POST_STRUCTURAL_MIGRATION_RESTORE_REQUIRES_TRUSTED_PHYSICAL_STATE_ARTIFACT",
        "BACKUP_ENVELOPE_ALONE_CANNOT_RECONSTRUCT_ARBITRARY_MIGRATION_DDL_DML_EFFECTS",
        "RESTORE_MUST_NOT_REPLAY_MIGRATION_SQL_TO_MANUFACTURE_MATERIALIZATION",
    ]
    assert physical["candidate_with_declaration_additionally_requires"] == (
        "trusted physical StateStore materialization proof"
    )
    assert physical["production_stage_2_status"] == (
        "INTENTIONALLY_BLOCKED_FOR_POST_STRUCTURAL_MIGRATION_CANDIDATES"
    )
    assert physical["deferred_to"] == "NEXT_SEPARATE_ARCHITECTURE_CORRECTIVE"
    assert physical["partial_production_restore_integration_permitted"] is False


def test_secret_restore_port_is_read_only_and_separate_from_mutation_api() -> None:
    port = CONTRACT["secret_handoff_restore_authority"]["port"]
    assert port["name"] == "SecretHandoffRestoreAuthorityPort"
    assert port["owner"] == "EXTERNAL PROTECTED SECURE-STORE / SECRET-RESOURCE AUTHORITY BOUNDARY"
    assert port["method"] == "observe(descriptor) -> SecretHandoffRestoreObservation"
    assert set(port["properties"]) == {
        "READ_ONLY",
        "SIDE_EFFECT_FREE",
        "NO_RETRY_OF_BUSINESS_MUTATION",
        "NO_BEGIN",
        "NO_CLEANUP",
        "NO_LIFECYCLE_WRITE",
        "NO_M0.3_ACTION",
    }
    assert "SecretExternalResourcePort mutation API" in port["not"]


def test_secret_restore_observation_has_exact_fields_and_closed_states() -> None:
    observation = CONTRACT["secret_handoff_restore_authority"]["observation"]
    assert observation["exact_fields"] == [
        "handoff_id",
        "scope",
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "external_state",
    ]
    assert observation["external_state_registry"] == [
        "NOT_STARTED",
        "COMMITTED",
        "UNRESOLVED",
        "CLEANUP_ACCEPTED_OR_SATISFIED",
    ]
    assert observation["candidate_may_carry_or_mint"] is False


def test_secret_state_relation_matrix_is_exact() -> None:
    secret = CONTRACT["secret_handoff_restore_authority"]
    assert secret["state_relation_matrix"] == {
        "PREPARED": ["NOT_STARTED", "COMMITTED", "UNRESOLVED"],
        "COMMITTED": ["COMMITTED", "CLEANUP_ACCEPTED_OR_SATISFIED"],
        "CLEANUP_PENDING": ["CLEANUP_ACCEPTED_OR_SATISFIED"],
        "UNKNOWN_RECONCILIATION": ["UNRESOLVED"],
    }
    assert secret["any_other_state_pair"] == "RESTORE_REJECTED"
    assert len(secret["required_exact_bindings"]) == 4


def test_secret_missing_unavailable_unknown_or_mismatched_authority_fails_closed() -> None:
    policy = CONTRACT["secret_handoff_restore_authority"]["availability_policy"]
    assert policy == {
        "no_family_records_and_no_port": "ALLOWED_GATE_NOT_APPLICABLE",
        "any_family_record_and_missing_port": "RESTORE_REJECTED",
        "authority_call_unavailable_or_error": "RESTORE_REJECTED",
        "unknown_handoff": "RESTORE_REJECTED",
        "scope_or_fingerprint_mismatch": "RESTORE_REJECTED",
        "fallback_to_SecretExternalResourcePort_reconcile": "FORBIDDEN",
    }
    assert (
        CONTRACT["secret_handoff_restore_authority"]["local_relations_establish_external_authority"]
        is False
    )


def test_secret_restore_requires_current_authoritative_observation() -> None:
    semantics = FRESHNESS["observation_semantics"]
    assert semantics == {
        "required_result": "CURRENT_AUTHORITATIVE_OBSERVATION_AT_CALL_TIME",
        "prohibited_substitutes": [
            "cached previous observation",
            "BackupEnvelope-carried observation",
            "StateStore-carried observation",
            "caller-provided observation",
            "stale process-local accepted observation",
        ],
        "current_authority_unavailable": "RESTORE_REJECTED",
        "durable_observation_record": "FORBIDDEN",
        "authority_revision": "FORBIDDEN",
    }


def test_secret_restore_has_exactly_three_defined_fence_types() -> None:
    fences = FRESHNESS["required_restore_time_fences"]
    assert fences["exact_count"] == 3
    assert fences["ordered_names"] == [
        "INITIAL_STAGE_2",
        "PRE_INSTALL",
        "FINAL_PROMOTION",
    ]
    assert fences["exact_count_semantics"] == (
        "EXACTLY_THREE_DEFINED_FENCE_TYPES; not a maximum invocation count; a fence "
        "TYPE may be invoked more than once on one restore path when multiple "
        "authority-sensitive boundaries require a current observation"
    )


def test_prepared_m0_3_reuses_the_final_promotion_fence_type() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["fence_type"] == "FINAL_PROMOTION"
    assert prepared["fourth_restore_time_fence_created"] is False
    assert prepared["mapping"] == (
        "REPEATED_INVOCATION_OF_THE_SAME_FINAL_PROMOTION_FENCE_TYPE_AT_TWO_"
        "AUTHORITY_SENSITIVE_BOUNDARIES"
    )


def test_prepared_m0_3_final_promotion_sequence_is_exact() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["required_sequence"] == [
        (
            "fresh FINAL_PROMOTION-type observation of EVERY descriptor immediately "
            "before existing M0.3 recovery/finalization"
        ),
        "perform existing M0.3 recovery/finalization",
        (
            "after successful finalization, freshly invoke FINAL_PROMOTION again "
            "immediately before actual restore promotion, readiness, or evidence release"
        ),
    ]


def test_prepared_m0_3_forbids_pre_finalization_observation_reuse() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["pre_finalization_observation_reuse"] == (
        "FORBIDDEN_AS_PROOF_OF_CURRENT_SECRET_HANDOFF_AUTHORITY_AFTER_M0.3_FINALIZATION"
    )
    assert prepared["reason"] == (
        "SecretHandoff external authority is mutable and no frozen cross-resource ACID "
        "transaction spans SecretHandoff authority and M0.3 finalization"
    )


def test_prepared_m0_3_pre_finalization_failure_fails_closed() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["pre_finalization_observation_failure"] == [
        "DO_NOT_PERFORM_M0.3_FINALIZATION",
        "FAIL_CLOSED",
    ]


def test_prepared_m0_3_post_finalization_failure_fails_closed_without_rollback() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["post_finalization_pre_promotion_observation_failure"] == [
        "DO_NOT_GRANT_LIVE_READINESS",
        "DO_NOT_GRANT_BUSINESS_RESUME_PERMISSION",
        "DO_NOT_PROMOTE_RESTORE_AUTHORITY",
        "DO_NOT_EMIT_OR_USE_FRESH_EVIDENCE_AS_PROMOTION_AUTHORIZATION",
    ]
    assert prepared["completed_M0_3_finalization_rollback_claim"] == "FORBIDDEN"
    assert prepared["SecretHandoff_authority_mutation"] == "FORBIDDEN"


def test_prepared_m0_3_secret_handoff_is_only_a_prerequisite_gate() -> None:
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["SecretHandoff_authority_role"] == (
        "PREREQUISITE_GATE_ONLY_DOES_NOT_AUTHORIZE_M0.3_FINALIZE"
    )
    assert prepared["new_M0_3_PREPARE"] == "FORBIDDEN"
    assert prepared["M0_3_ABORT"] == "FORBIDDEN"


def test_noop_prepared_m0_3_uses_final_promotion_twice_without_a_new_fence() -> None:
    noop = FRESHNESS["NOOP_ALREADY_CURRENT"]
    assert noop["PRE_INSTALL"] == "NOT_APPLICABLE_NO_INSTALL_OCCURS"
    assert noop["existing_external_M0_3_PREPARED_sequence"] == [
        "INITIAL_STAGE_2",
        "fresh FINAL_PROMOTION invocation immediately before M0.3 recovery/finalization",
        (
            "fresh FINAL_PROMOTION invocation again after successful finalization and "
            "immediately before successful NOOP return or readiness"
        ),
    ]
    assert noop["prepared_path_fence_type_rule"] == (
        "same FINAL_PROMOTION fence TYPE invoked twice; no fourth fence TYPE"
    )

    fences = FRESHNESS["required_restore_time_fences"]
    assert fences["exact_count"] == 3
    assert fences["ordered_names"] == [
        "INITIAL_STAGE_2",
        "PRE_INSTALL",
        "FINAL_PROMOTION",
    ]
    prepared = FRESHNESS["existing_external_M0_3_PREPARED_finalize_gate"]
    assert prepared["fence_type"] == "FINAL_PROMOTION"


def test_secret_restore_fence_family_applicability_is_exact() -> None:
    applicability = FRESHNESS["family_applicability"]
    assert applicability["zero_SecretHandoff_family_records"] == (
        "ALL_SECRET_HANDOFF_FENCES_NOT_APPLICABLE; SecretHandoffRestoreAuthorityPort not required"
    )
    assert applicability["one_or_more_SecretHandoff_family_records"] == {
        "descriptor_coverage": (
            "every fence APPLICABLE_TO_CURRENT_RESTORE_PATH must observe EVERY descriptor"
        ),
        "skip_policy": "NO_APPLICABLE_FENCE_MAY_BE_SKIPPED",
        "absent_path_operation": ("operation-specific fence is NOT_APPLICABLE, not satisfied"),
    }


def test_installing_restore_fence_mapping_is_exact() -> None:
    installing = FRESHNESS["path_applicability_matrix"]["INSTALLING_RESTORE"]
    assert installing == {
        "applicable_fences": ["INITIAL_STAGE_2", "PRE_INSTALL", "FINAL_PROMOTION"],
        "FINAL_PROMOTION_prerequisites": [
            "atomic StateStore install has completed",
            "installed StateStore has been reopened and verified",
        ],
    }


def test_noop_restore_fence_surfaces_remain_aligned() -> None:
    matrix_noop = FRESHNESS["path_applicability_matrix"]["NOOP_ALREADY_CURRENT"]
    contract_noop = FRESHNESS["NOOP_ALREADY_CURRENT"]
    expected_fences = ["INITIAL_STAGE_2", "FINAL_PROMOTION"]
    expected_pre_install = "NOT_APPLICABLE_NO_INSTALL_OCCURS"

    assert matrix_noop == {
        "applicable_fences": expected_fences,
        "PRE_INSTALL": expected_pre_install,
        "FINAL_PROMOTION_prerequisites": [
            "current live StateStore has been freshly verified",
            "no installation or installed-store reopen is required",
        ],
    }
    assert contract_noop["applicable_fences"] == matrix_noop["applicable_fences"]
    assert contract_noop["PRE_INSTALL"] == matrix_noop["PRE_INSTALL"]
    assert contract_noop["FINAL_PROMOTION_prerequisite"] == (
        "current live StateStore freshly verified; no installation or installed-store "
        "reopen required"
    )


def test_generic_final_promotion_fails_closed_without_protected_outcomes() -> None:
    final_promotion = FRESHNESS["required_restore_time_fences"]["FINAL_PROMOTION"]
    assert final_promotion == {
        "timing": (
            "immediately before granting final restore promotion, LIVE readiness, "
            "business-resume permission, or usable fresh evidence"
        ),
        "action": (
            "freshly observe EVERY candidate SecretHandoff descriptor with current "
            "authoritative observations"
        ),
        "incompatible_or_unavailable": "FAIL_CLOSED",
        "must_not_gain": [
            "LIVE readiness",
            "business-resume permission",
            "restore authority promotion",
            "fresh evidence usable to authorize protected finalization/promotion",
        ],
        "external_secret_system_mutation": "FORBIDDEN",
    }


def test_secret_restore_fences_preserve_the_read_only_boundary() -> None:
    boundary = FRESHNESS["read_only_side_effect_boundary"]
    assert boundary == {
        "every_fence": "READ_ONLY",
        "forbidden_calls": ["begin()", "cleanup()"],
        "forbidden_effects": [
            "advance SecretHandoff lifecycle",
            "write SecretHandoff records",
            "retry initial mutation",
            "retry cleanup",
            "mint M0.3 authority",
            "repair candidate state",
        ],
        "observation_requires_mutation_to_establish_truth": "RESTORE_REJECTED",
    }


def test_restore_stage_order_and_responsibilities_remain_exact() -> None:
    stages = CONTRACT["ordered_stage_responsibilities"]
    assert stages["ordered_stages"] == EXPECTED_RESTORE_STAGES
    assert (
        MACHINE["backup_contract"]["restore_candidate_revalidation_orchestration"]["ordered_stages"]
        == EXPECTED_RESTORE_STAGES
    )
    assert stages["stage_2_authority_reconstructed_from_stage_1"] is False
    assert "final restore freshness owner" in stages["M0_3_RESTORE_FRESHNESS"]


def test_stage_2_forbids_business_mutation_and_authority_minting() -> None:
    boundary = CONTRACT["assessment_side_effect_boundary"]
    assert boundary["permitted"] == [
        "read-only static plan comparison",
        "read-only SecretHandoff observation",
    ]
    assert set(boundary["forbidden"]) >= {
        "execute migration SQL",
        "call MigrationExecutionCoordinator.execute()",
        "call DurableMigrationCompletionCoordinator.resume_to_completion()",
        "call SecretHandoff begin()",
        "call SecretHandoff cleanup()",
        "create M0.3 PREPARE",
        "ABORT M0.3",
        "mint authority",
        "repair candidate history",
    }


def test_noop_corrupt_and_prepared_paths_cannot_bypass_authority_gates() -> None:
    paths = CONTRACT["path_invariants"]
    assert (
        paths[
            "NO_RESTORE_PATH_MAY_REACH_M0_3_RESTORE_FRESHNESS_BEFORE_M0_11_LIFECYCLE_STAGE_2_AND_STAGE_3"
        ]
        is True
    )
    noop = paths["NOOP_ALREADY_CURRENT"]
    assert "lifecycle semantic authority revalidation" in noop
    assert "current relational revalidation" in noop
    assert "M0.3 freshness" in noop
    assert "corrupt store supplies no lifecycle authority" in paths["CORRUPT_OR_UNREADABLE"]
    assert "precede existing protected FINALIZE recovery" in paths["EXTERNALLY_PREPARED_M0_3"]
    assert "cannot authorize FINALIZE" in paths["EXTERNALLY_PREPARED_M0_3"]


def test_future_restore_sequence_preserves_semantic_stage_order() -> None:
    sequence = MACHINE["backup_contract"]["descriptor_preservation_contract"][
        "future_restore_sequence"
    ]
    stage_2 = "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION"
    stage_3 = "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION"
    freshness = "apply pre-existing external M0.3 restore freshness gate"

    assert sequence.count(stage_2) == 1
    assert sequence.count(stage_3) == 1
    assert sequence.count(freshness) == 1
    assert sequence.index(stage_2) < sequence.index(stage_3) < sequence.index(freshness)
    assert sequence.index("verify full chain and local candidate") < sequence.index(stage_2)


def test_promotion_gates_preserve_authority_order() -> None:
    gates = MACHINE["restore_contract"]["promotion_gates"]
    ordered_gates = [
        "canonical IDs/history consistency",
        "M0.11 lifecycle aspect-specific semantic authority revalidation",
        "M0.11 current-designation relational revalidation",
        "external M0.3 current membership",
        "protected authority available, well-formed, current and non-retired",
    ]

    assert gates.count(ordered_gates[1]) == 1
    assert gates.count(ordered_gates[2]) == 1
    assert [gates.index(gate) for gate in ordered_gates] == sorted(
        gates.index(gate) for gate in ordered_gates
    )


def test_all_complete_restore_surfaces_put_stage_2_and_3_before_freshness() -> None:
    orchestration = MACHINE["backup_contract"]["restore_candidate_revalidation_orchestration"][
        "ordered_stages"
    ]
    responsibilities = CONTRACT["ordered_stage_responsibilities"]["ordered_stages"]
    future = MACHINE["backup_contract"]["descriptor_preservation_contract"][
        "future_restore_sequence"
    ]
    gates = MACHINE["restore_contract"]["promotion_gates"]
    surface_markers = [
        (
            orchestration,
            "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
            "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
            "M0_3_RESTORE_FRESHNESS",
        ),
        (
            responsibilities,
            "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
            "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
            "M0_3_RESTORE_FRESHNESS",
        ),
        (
            future,
            "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
            "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
            "apply pre-existing external M0.3 restore freshness gate",
        ),
        (
            gates,
            "M0.11 lifecycle aspect-specific semantic authority revalidation",
            "M0.11 current-designation relational revalidation",
            "external M0.3 current membership",
        ),
    ]

    for surface, stage_2, stage_3, freshness in surface_markers:
        assert surface.index(stage_2) < surface.index(stage_3) < surface.index(freshness)
