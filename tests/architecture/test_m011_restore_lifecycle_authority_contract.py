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


def test_restore_stage_order_and_responsibilities_remain_exact() -> None:
    stages = CONTRACT["ordered_stage_responsibilities"]
    expected = [
        "STRUCTURAL_PERSISTENCE_RECORD_VALIDATION",
        "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
        "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
        "M0_3_RESTORE_FRESHNESS",
    ]
    assert stages["ordered_stages"] == expected
    assert (
        MACHINE["backup_contract"]["restore_candidate_revalidation_orchestration"]["ordered_stages"]
        == expected
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
    assert "MUST execute lifecycle semantic authority revalidation" in paths["NOOP_ALREADY_CURRENT"]
    assert "corrupt store supplies no lifecycle authority" in paths["CORRUPT_OR_UNREADABLE"]
    assert "precede existing protected FINALIZE recovery" in paths["EXTERNALLY_PREPARED_M0_3"]
    assert "cannot authorize FINALIZE" in paths["EXTERNALLY_PREPARED_M0_3"]
