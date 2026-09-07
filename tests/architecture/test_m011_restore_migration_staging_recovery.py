"""Executable oracle for the source-durable restore/migration workspace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CANONICAL = json.loads(
    (DOCS / "persistence_versioning_migrations_backup_and_recovery.json").read_text()
)
CONTRACT = CANONICAL["restore_migration_staging_recovery_contract"]

MANIFEST_FIELDS = [
    "staging_manifest_schema_version",
    "staging_id",
    "account_id",
    "device_installation_id",
    "environment",
    "state_store_identity_fingerprint_sha256",
    "source_backup_envelope_fingerprint_sha256",
    "source_state_store_schema_version",
    "source_generation",
    "source_state_fingerprint_sha256",
    "source_transaction_fingerprint_sha256",
    "source_history_tail_fingerprint_sha256",
    "source_staged_sqlite_artifact_fingerprint_sha256",
    "migration_id",
    "migration_definition_fingerprint_sha256",
    "operation_plan_fingerprint_sha256",
    "target_state_store_schema_version",
    "manifest_fingerprint_sha256",
]
PHASES = [
    "SOURCE_READY",
    "PREPARED",
    "APPLYING_SOURCE",
    "APPLYING_MATERIALIZED",
    "DURABLE_MIGRATED",
    "COMPLETED",
    "FAILED",
]


def test_corrective_names_all_three_previous_sequencing_contradictions() -> None:
    contradictions = CONTRACT["confirmed_contradictions"]
    assert set(contradictions) == {
        "execution_order",
        "generation_model",
        "byte_hash_model",
        "promotion_path_topology",
        "windows_initial_staged_sibling_publication",
        "windows_control_directory_publication",
    }
    assert (
        "external COMMITTED verification complete before execute returns"
        in contradictions["execution_order"]
    )
    assert "not source generation G+1" in contradictions["generation_model"]
    assert "mutable workspace" in contradictions["byte_hash_model"]
    assert "same-directory promotion" in contradictions["promotion_path_topology"]
    assert "MoveFileExW" in contradictions["windows_initial_staged_sibling_publication"]
    assert "directory chain" in contradictions["windows_control_directory_publication"]


def test_corrective_freezes_manifest_sidecar_and_sqlite_in_live_parent() -> None:
    topology = CONTRACT["physical_topology"]
    assert topology["control_directory"] is None
    assert topology["new_control_directory_created"] is False
    assert topology["no_new_durable_control_directory_exists"] is True
    assert topology["manifest_path"] == (
        "<live.parent>/.<live.name>.restore-migration-<staging_id>.manifest.json"
    )
    assert topology["manifest_parent_is_live_parent"] is True
    assert topology["all_durable_artifact_parents_are_live_parent"] is True
    assert topology["staged_sqlite_path"] == (
        "<live.parent>/.<live.name>.restore-migration-<staging_id>.sqlite3"
    )
    assert topology["same_directory_invariant"] == (
        "Path(staged_sqlite_path).parent.resolve() == Path(live_path).parent.resolve()"
    )
    assert topology["distinct_path_invariant"] == (
        "Path(staged_sqlite_path).resolve() != Path(live_path).resolve()"
    )
    assert ".cryptohunter-restore-staging" not in json.dumps(CONTRACT)
    assert topology["new_directory_creation_fence_required"] is False
    assert topology["volume_handle_required"] is False
    assert topology["administrator_privileges_required"] is False


def test_physical_paths_are_deterministic_and_collision_safe() -> None:
    topology = CONTRACT["physical_topology"]

    def staged(live: Path, staging_id: str) -> Path:
        return live.parent / f".{live.name}.restore-migration-{staging_id}.sqlite3"

    def unpublished(live: Path, staging_id: str) -> Path:
        return live.parent / (f".{live.name}.restore-migration-{staging_id}.unpublished.sqlite3")

    def manifest(live: Path, staging_id: str) -> Path:
        return live.parent / f".{live.name}.restore-migration-{staging_id}.manifest.json"

    staging_id = "a" * 64
    alpha = Path("root/alpha.sqlite3")
    beta = Path("root/beta.sqlite3")
    alpha_staged, beta_staged = staged(alpha, staging_id), staged(beta, staging_id)
    assert alpha_staged.parent == alpha.parent
    assert beta_staged.parent == beta.parent
    assert alpha_staged != alpha and beta_staged != beta
    assert staging_id in alpha_staged.name and staging_id in beta_staged.name
    assert alpha.name in alpha_staged.name and beta.name in beta_staged.name
    assert alpha_staged != beta_staged
    assert unpublished(alpha, staging_id) != unpublished(beta, staging_id)
    assert alpha.name in unpublished(alpha, staging_id).name
    assert beta.name in unpublished(beta, staging_id).name
    assert manifest(alpha, staging_id) != manifest(beta, staging_id)
    assert alpha.name in manifest(alpha, staging_id).name
    assert beta.name in manifest(beta, staging_id).name
    policy = topology["collision_policy"]
    assert "alpha.sqlite3 and beta.sqlite3" in policy
    assert "unpublished" in policy
    assert "staged" in policy
    assert "manifest sidecar" in policy
    assert "control director" not in policy.lower()
    assert topology["authority_identity_unchanged"] is True


def test_one_sqlite_artifact_uses_two_sequential_paths_without_copy() -> None:
    topology = CONTRACT["physical_topology"]
    assert topology["single_sqlite_artifact"] is True
    assert topology["materialize_once_at_unpublished_path"] is True
    assert topology["durably_rename_same_file_to_staged_sibling_before_manifest"] is True
    assert topology["prepublication_same_file_rename"] is True
    assert topology["prepublication_byte_copy"] is False
    assert topology["two_complete_sqlite_artifacts_exist_concurrently"] is False
    assert topology["pre_promotion_copy"] is False
    assert topology["promotion_clone"] is False
    assert topology["cross_directory_rename"] is False
    assert CONTRACT["artifact"]["new_durable_control_directory_exists"] is False
    assert topology["prepublication_publication_owner"] == (
        "publish_file_atomically_durably(unpublished_materialization_path, staged_sqlite_path)"
    )
    assert topology["source_ready_invariant"] == {
        "unpublished_materialization_path_exists": False,
        "staged_sqlite_path_exists": True,
        "manifest_path_exists": True,
        "all_parents_equal_preexisting_live_parent": True,
        "control_directory_exists": False,
    }
    assert topology["lifecycle_and_promotion_paths"] == {
        "P2A": "staged_sqlite_path only",
        "P2B_F1_through_F4": "staged_sqlite_path only",
        "unpublished_path_participates": False,
    }


def test_unpublished_and_final_paths_are_deterministic_same_directory_siblings() -> None:
    topology = CONTRACT["physical_topology"]
    staging_id = "b" * 64
    live = Path("root/live.sqlite3")
    unpublished = live.parent / (f".{live.name}.restore-migration-{staging_id}.unpublished.sqlite3")
    staged = live.parent / f".{live.name}.restore-migration-{staging_id}.sqlite3"
    manifest = live.parent / f".{live.name}.restore-migration-{staging_id}.manifest.json"
    assert topology["unpublished_materialization_path"].endswith(
        ".restore-migration-<staging_id>.unpublished.sqlite3"
    )
    assert unpublished.parent == staged.parent == manifest.parent == live.parent
    assert len({unpublished, staged, manifest, live}) == 4
    assert live.name in unpublished.name and staging_id in unpublished.name
    assert live.name in staged.name and staging_id in staged.name
    assert live.name in manifest.name and staging_id in manifest.name
    assert topology["unpublished_same_directory_invariant"].startswith(
        "Path(unpublished_materialization_path).parent.resolve()"
    )


def test_workspace_is_source_seeded_durable_and_never_authority() -> None:
    artifact = CONTRACT["artifact"]
    assert artifact["canonical_name"] == "RestoreMigrationStagingArtifact"
    assert artifact["classification"] == "SOURCE-SEEDED DURABLE RESTORE/MIGRATION WORKSPACE"
    assert artifact["integrity_recovery_carrier"] is True
    assert artifact["authority"] is False
    assert artifact["authoritative_state_store_member"] is False
    assert artifact["same_sqlite_path_for_all_protected_transitions"] is True
    assert artifact["created_before_first_protected_migration_transition"] is True


def test_manifest_is_closed_immutable_source_only_binding() -> None:
    manifest = CONTRACT["manifest"]
    assert manifest["semantic_role"] == (
        "IMMUTABLE SOURCE / WORKSPACE IDENTITY MANIFEST; NOT LIFECYCLE AUTHORITY"
    )
    assert manifest["exact_ordered_fields"] == MANIFEST_FIELDS
    assert set(manifest["field_contracts"]) == set(MANIFEST_FIELDS)
    assert manifest["additional_fields"] is False
    assert manifest["source_only"] is True
    assert manifest["records_current_phase"] is False
    assert manifest["validation_grants_authority"] is False
    assert manifest["forbidden_fields"] == [
        "target_generation",
        "target_state_fingerprint_sha256",
        "target_transaction_fingerprint_sha256",
        "target_history_tail_fingerprint_sha256",
        "staged_sqlite_artifact_fingerprint_sha256",
    ]


def test_source_byte_hash_is_initial_only_and_current_integrity_uses_normal_owners() -> None:
    byte_hash = CONTRACT["source_byte_fingerprint"]
    assert byte_hash["applies_only_to"] == (
        "initial exact source-v1 materialization before any migration lifecycle mutation"
    )
    assert byte_hash["compare_against_mutated_current_workspace"] is False
    assert byte_hash["current_workspace_integrity_after_transition"] == [
        "SQLiteStateStore.read_verified_snapshot()",
        "descriptor lineage from exact source anchor",
        "external protected freshness",
        "sealed migration authority",
    ]


def test_staging_identity_is_deterministic_without_unknown_target_generation() -> None:
    identity = CONTRACT["staging_identity"]
    assert identity["exact_projection_order"] == [
        "state_store_identity_fingerprint_sha256",
        "source_backup_envelope_fingerprint_sha256",
        "migration_id",
        "target_state_store_schema_version",
    ]
    assert identity["depends_on_target_generation"] is False
    assert identity["caller_restore_attempt_id_permitted"] is False
    assert identity["arbitrary_sqlite_scanning"] is False
    assert "STAGING_CONFLICT" in identity["conflict"]


def test_exact_source_is_durable_before_first_protected_transition() -> None:
    order = CONTRACT["initial_durability_order"]
    assert order[0] == "authenticate exact BackupEnvelope v1"
    assert "manifest_path directly under pre-existing live.parent" in order[1]
    assert "one exact v1 SQLite at unpublished_materialization_path" in order[2]
    assert "source descriptor anchor at unpublished_materialization_path" in order[3]
    assert order[5:10] == [
        "fsync unpublished SQLite regular file",
        "publish_file_atomically_durably(unpublished_materialization_path, staged_sqlite_path) using the existing same-file durable rename owner",
        "require unpublished_materialization_path absent and staged_sqlite_path present",
        "compute source PhysicalSQLiteArtifact fingerprint from staged_sqlite_path",
        "build exact 18-field immutable source-only manifest",
    ]
    assert order[10] == (
        "atomic_write_bytes_durably(manifest_path, canonical manifest bytes) in pre-existing "
        "live.parent"
    )
    assert not any("directory creation" in step.lower() for step in order)
    assert order[-1] == "only now may PREPARED migration transition begin"


def test_existing_physical_durability_primitive_owns_prepublication_rename() -> None:
    import inspect

    from bot_core.persistence import physical_durability

    publish = inspect.getsource(physical_durability.publish_file_atomically_durably)
    atomic_write = inspect.getsource(physical_durability.atomic_write_bytes_durably)
    windows = inspect.getsource(physical_durability._move_file_ex_windows)
    directory = inspect.getsource(physical_durability.flush_created_directory_metadata)
    assert "os.replace(source, destination)" in publish
    assert "_move_file_ex_windows(source, destination)" in publish
    assert "_flush_directory_posix(destination.parent)" in publish
    assert "MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH" in windows
    assert "MoveFileExW" in windows
    assert 'if platform == "nt":\n        return' in directory
    assert "FlushFileBuffers" not in directory
    assert "publish_file_atomically_durably(temporary, target)" in atomic_write


def test_source_anchor_and_restricted_suffix_prove_workspace_lineage() -> None:
    lineage = CONTRACT["source_lineage"]
    assert lineage["anchor_required"] is True
    assert lineage["exact_anchor_fields"] == {
        "target_generation": "manifest.source_generation",
        "post_state_fingerprint_sha256": "manifest.source_state_fingerprint_sha256",
        "transaction_fingerprint_sha256": "manifest.source_transaction_fingerprint_sha256",
        "post_history_tail_fingerprint_sha256": "manifest.source_history_tail_fingerprint_sha256",
    }
    assert lineage["predecessor_history_prefix_rewritten"] is False
    assert lineage["scope_only_match_sufficient"] is False
    suffix = CONTRACT["workspace_suffix"]
    assert suffix["unrelated_domain_writes"] is False
    assert "exact sealed structural migration execution edge" in suffix["allowed_mutations"]


def test_closed_phase_classifier_covers_every_restart_point() -> None:
    phase_authority = CONTRACT["phase_authority"]
    phases = phase_authority["closed_phases"]
    assert list(phases) == PHASES
    assert phase_authority["manifest_declares_current_phase"] is False
    assert phase_authority["only_installable_phase"] == "COMPLETED"
    assert phases["SOURCE_READY"]["schema_version"] == 1
    assert phases["PREPARED"]["schema_version"] == 1
    assert phases["APPLYING_SOURCE"]["allowed_next"] == (
        "MigrationExecutionCoordinator.execute(migration_id)"
    )
    assert phases["APPLYING_MATERIALIZED"]["schema_version"] == 2
    assert "record_durable_migrated" in phases["APPLYING_MATERIALIZED"]["allowed_next"]
    assert "DurableMigrationCompletionCoordinator" in phases["DURABLE_MIGRATED"]["allowed_next"]
    assert phases["COMPLETED"]["installable"] is True
    assert phases["FAILED"]["installable"] is False
    assert all(not phases[name]["installable"] for name in PHASES if name != "COMPLETED")


def test_final_generation_is_read_from_completed_staged_snapshot() -> None:
    final = CONTRACT["final_generation"]
    assert final["fixed_source_plus_one"] is False
    assert final["authority"] == (
        "fresh COMPLETED staged snapshot.metadata.protected_freshness_generation"
    )
    assert "never compare final external membership to source v1" in final["external_match"]
    determinism = CONTRACT["determinism"]
    assert determinism["fixed_final_generation_assumption"] is False
    assert determinism["final_generation_source"] == "exact durable transition sequence"
    assert all(
        token in determinism["nondeterminism_audit"]
        for token in ("no timestamp", "UUID", "random nonce")
    )


def test_external_resume_is_phase_driven_and_fail_closed() -> None:
    external = CONTRACT["external_restart_classification"]
    assert set(external) == {
        "COMMITTED_EXACT_STAGED_CURRENT",
        "PREPARED_FOR_NEXT_STAGED_CANDIDATE",
        "COMMITTED_PREDECESSOR_WHILE_LOCAL_PREDECESSOR",
        "EXTERNAL_AHEAD_OF_STAGED_CURRENT",
        "EXTERNAL_DIFFERENT_SAME_GENERATION_OR_SCOPE",
    }
    assert "recover_protected_state" in external["PREPARED_FOR_NEXT_STAGED_CANDIDATE"]
    assert external["EXTERNAL_AHEAD_OF_STAGED_CURRENT"].startswith("FAIL_CLOSED")
    assert external["EXTERNAL_DIFFERENT_SAME_GENERATION_OR_SCOPE"] == "FAIL_CLOSED"


def test_existing_protected_order_keeps_committed_state_in_durable_workspace() -> None:
    durability = CONTRACT["sqlite_lifecycle_durability"]
    assert durability["workspace_may_remain_open"] is True
    assert durability["close_between_protected_transitions_required"] is False
    assert durability["new_per_transition_filesystem_protocol"] is False
    assert "after every external COMMITTED transition" in durability["critical_property"]
    protected = CONTRACT["protected_freshness"]
    assert protected["semantics_changed"] is False
    assert protected["new_callback_or_hook"] is False
    assert protected["exact_existing_order"] == [
        "external PREPARE",
        "local durable StateStore mutation",
        "publish local evidence",
        "external FINALIZE",
        "verify external COMMITTED",
    ]


def test_only_completed_workspace_can_cross_all_final_install_fences() -> None:
    install = CONTRACT["final_install"]
    assert "closed phase COMPLETED" in install["requirements"]
    assert (
        "fresh external COMMITTED matching current COMPLETED staged generation/state"
        in install["requirements"]
    )
    assert (
        "final external re-read exact current COMPLETED staged generation/state"
        in install["requirements"]
    )
    assert install["source_backup_equals_final_external_required"] is False
    assert install["raw_v1_install"] is False
    assert install["operation"].startswith("atomic_replace")


def test_crash_matrix_c0_through_c7_is_complete() -> None:
    matrix = CONTRACT["crash_matrix"]
    assert list(matrix) == ["C0", "C1", "C2", "C3", "C4", "C5", "C6A", "C6B", "C7"]
    assert "SOURCE_READY" in matrix["C1"]["restart"]
    assert "protected recovery" in matrix["C2"]["restart"]
    assert "schema-v1" not in matrix["C3"]["restart"]  # phase is carried, not guessed
    assert "without rematerializing backup" in matrix["C3"]["restart"]
    assert "exact declaration/edge" in matrix["C4"]["restart"]
    assert "never replay structural migration" in matrix["C5"]["restart"]
    assert "repeat F1 through F4" in matrix["C6A"]["restart"]
    assert "no migration replay" in matrix["C6B"]["restart"]
    assert "idempotent cleanup" in matrix["C7"]["restart"]


def test_cleanup_and_missing_workspace_never_roll_back_or_replay() -> None:
    cleanup = CONTRACT["cleanup"]
    assert cleanup["unconditional_finally_cleanup"] is False
    assert cleanup["before_first_transition_only_if"] == [
        "external remains exact source",
        "workspace abandoned before PREPARED",
        "live unchanged",
    ]
    assert cleanup["after_prepared_or_later"].startswith("workspace survives ordinary failure")
    assert CONTRACT["missing_workspace"].startswith("manifest present + staged sibling missing")
    assert (
        "RESTORE_MIGRATION_STAGING_LOST / MANUAL_RECOVERY_REQUIRED" in CONTRACT["missing_workspace"]
    )
    assert "no source replay or external rollback" in CONTRACT["missing_workspace"]


def test_restore_only_orchestrates_existing_migration_owners() -> None:
    owners = CONTRACT["ordinary_owners"]
    assert owners["restore_owns_migration_sql"] is False
    assert owners["structural_execution"] == "MigrationExecutionCoordinator"
    assert owners["lifecycle"] == "DurableMigrationLifecycleCoordinator"
    assert owners["completion"] == "DurableMigrationCompletionCoordinator"
    assert owners["migration_restore_authority"] == (
        "SealedMigrationRestoreAuthority remains read-only"
    )
    assert CONTRACT["protected_freshness"]["external_rollback"] is False


def test_backup_security_and_m010_identity_remain_unchanged() -> None:
    preserved = CONTRACT["preserved_contracts"]
    assert preserved["source_backup_required_on_every_resume"] is True
    assert preserved["backup_envelope_outer_schema_version"] == 1
    assert preserved["secret_handoff_fences_preserved"] is True
    assert preserved["non_durable_security_authority_restorable"] is False
    raw = (DOCS / "identity_device_authentication_and_secrets.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "dd7a23a4f40001a929ab16135876b5d1cb850d3a4f136c7e7d21e01a1c38e61e"
    )


def test_production_wal_primitives_match_frozen_final_promotion_boundary() -> None:
    import inspect

    from bot_core.persistence.state_store import SQLiteStateStore

    constructor = inspect.getsource(SQLiteStateStore.__init__)
    prepare = inspect.getsource(SQLiteStateStore.prepare_for_atomic_install)
    replace = inspect.getsource(SQLiteStateStore.atomic_replace)
    assert "PRAGMA journal_mode = WAL" in constructor
    assert "PRAGMA synchronous = FULL" in constructor
    assert "PRAGMA wal_checkpoint(TRUNCATE)" in prepare
    assert "self.close()" in prepare
    assert "wal_checkpoint" not in replace
    assert "source.parent.resolve() != target.parent.resolve()" in replace
    assert "same-directory isolated store" in replace
    assert "os.replace(source, target)" in replace


def test_final_promotion_requires_existing_checkpoint_close_owner() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    assert boundary["invariant"] == (
        "NO OPEN / UNCHECKPOINTED STAGING MAY CROSS FINAL ATOMIC PROMOTION"
    )
    assert boundary["production_storage"] == {
        "journal_mode": "WAL",
        "synchronous": "FULL",
        "intermediate_durability": (
            "existing transactional SQLiteStateStore commits under WAL + FULL"
        ),
    }
    primitives = boundary["existing_primitives"]
    assert "wal_checkpoint(TRUNCATE)" in primitives["prepare_for_atomic_install"]
    assert "does not checkpoint source WAL" in primitives["atomic_replace"]
    assert "read-only immutable" in primitives["read_isolated_verified_snapshot"]
    assert "proves zero registered SQLiteStateStore handles" in primitives["installation_gate"]


def test_final_promotion_sequence_is_exact_f1_through_f4() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    assert boundary["ordering"] == [
        "F1_OPEN_FINAL_SEMANTIC_VERIFICATION",
        "F2_CHECKPOINT_AND_CLOSE_ORCHESTRATOR_HANDLE",
        "F2B_ACQUIRE_STAGING_PATH_GATE",
        "F3_CLOSED_MAIN_CONTINUITY",
        "F4_FINAL_LIVE_AND_EXTERNAL_FENCES",
    ]
    sequence = boundary["sequence"]
    f1 = sequence["F1_OPEN_FINAL_SEMANTIC_VERIFICATION"]
    f2 = sequence["F2_CHECKPOINT_AND_CLOSE_ORCHESTRATOR_HANDLE"]
    f2b = sequence["F2B_ACQUIRE_STAGING_PATH_GATE"]
    f3 = sequence["F3_CLOSED_MAIN_CONTINUITY"]
    f4 = sequence["F4_FINAL_LIVE_AND_EXTERNAL_FENCES"]
    assert "phase exactly COMPLETED" in f1
    assert "staged read_verified_snapshot() captured as completed_snapshot" in f1
    assert "staged.prepare_for_atomic_install()" in f2[0]
    assert "path-wide staging-handle quiescence" in f2[3]
    assert "installation_gate(staged_sqlite_path)" in f2b[0]
    assert "hold staging path gate continuously" in f2b[2]
    assert "require WAL checkpoint TRUNCATE success" in f2
    assert "closed_staged_snapshot equals completed_snapshot" in f3
    assert any("without staged -wal or -shm" in item for item in f3)
    assert any("do not reopen mutable/registered" in item for item in f3)
    assert "installation_gate(live_path)" in f4[0]
    assert any("external authority re-read after source is checkpointed" in item for item in f4)
    assert "both gates are held" in f4[-2]
    assert "atomic_replace(staged_sqlite_path, live_path)" in f4[-2]
    assert "source.parent == target.parent by construction" in f4[-2]


def test_c0_c6b_and_c7_use_the_single_deterministic_sibling() -> None:
    publication = CONTRACT["publication_boundary"]
    cleanup = CONTRACT["cleanup"]
    matrix = CONTRACT["crash_matrix"]
    assert "sole durable publication boundary" in publication["final_manifest"]
    assert "pre-existing live.parent" in publication["final_manifest"]
    assert "unpublished_materialization_path" in publication["manifest_absent"]
    assert "rename outcome uncertainty" in publication["manifest_absent"]
    assert "never C0-delete or rematerialize" in publication["manifest_present"]
    assert publication["arbitrary_globbing"] is False
    assert cleanup["c0_exact_artifacts"] == [
        "unpublished_materialization_path",
        "unpublished -wal",
        "unpublished -shm",
        "staged_sqlite_path",
        "staged -wal",
        "staged -shm",
        "manifest transient temp owned by exact manifest_path publication",
    ]
    assert cleanup["c0_requires_manifest_absent"] is True
    assert cleanup["manifest_present_missing_sibling_is_c0"] is False
    assert "unpublished only or staged only" in cleanup["c0_rename_outcomes"]
    assert cleanup["valid_manifest_with_unpublished_path"].startswith("STAGING_CONFLICT")
    assert publication["protected_transition_before_source_ready"] is False
    assert "manifest sidecar present" in matrix["C6B"]["restart"]
    assert "manifest sidecar may remain" in matrix["C7"]["restart"]
    assert "ALREADY_INSTALLED_EXACT_TARGET" in matrix["C7"]["restart"]
    assert "exact manifest_path" in cleanup["post_promotion"]
    assert "no per-staging directory" in cleanup["post_promotion"]
    assert "no arbitrary glob" in cleanup["manifest_temp_cleanup"]


def test_closed_source_and_windows_handle_fences_are_explicit() -> None:
    requirements = CONTRACT["wal_promotion_boundary"]["source_requirements_at_replace"]
    assert requirements["open_process_local_staged_handles"] == 0
    assert requirements["requires_staged_wal"] is False
    assert requirements["requires_staged_shm"] is False
    assert requirements["mutable_reopen_after_checkpoint"] is False
    assert "no Unix open-inode rename assumption" in requirements["windows_safe"]


def test_checkpoint_and_closed_main_failures_never_install_or_cleanup() -> None:
    failure = CONTRACT["wal_promotion_boundary"]["failure_policy"]
    assert failure["checkpoint_failure"].startswith("FAIL_CLOSED; preserve staging; no install")
    assert failure["closed_main_mismatch"].startswith("FAIL_CLOSED; preserve staging")
    assert "no install or external rollback" in failure["final_fence_failure"]


def test_completed_is_semantic_phase_not_new_promotion_lifecycle_state() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    assert (
        "runtime-only CHECKPOINTED_AND_CLOSED_FOR_PROMOTION"
        in boundary["completed_phase_semantics"]
    )
    assert "CHECKPOINTED_AND_CLOSED_FOR_PROMOTION" in boundary["runtime_only_observations"]
    assert set(boundary["no_new_durable_state"].values()) == {False}
    assert CONTRACT["phase_authority"]["only_installable_phase"] == "COMPLETED"


def test_c6a_c6b_and_c7_freeze_closed_main_crash_recovery() -> None:
    matrix = CONTRACT["crash_matrix"]
    assert list(matrix) == ["C0", "C1", "C2", "C3", "C4", "C5", "C6A", "C6B", "C7"]
    assert "repeat F1 through F4" in matrix["C6A"]["restart"]
    assert "closed complete durable workspace survives" in matrix["C6B"]["restart"]
    assert "no migration replay" in matrix["C6B"]["restart"]
    assert "never new live main requiring old staging WAL" in matrix["C7"]["restart"]


def test_post_install_verifies_exact_completed_snapshot_before_cleanup() -> None:
    post = CONTRACT["wal_promotion_boundary"]["post_install"]
    assert post == [
        "open live through normal SQLiteStateStore",
        "installed.read_verified_snapshot() equals completed_snapshot",
        "fresh external authority remains exact completed target",
        "only after both proofs may staging manifest cleanup occur",
        "post-install mismatch => FAIL_CLOSED / MANUAL_RECOVERY_REQUIRED; no external rollback",
    ]
    install = CONTRACT["final_install"]
    assert "closed, checkpointed" in install["operation"]
    assert "completed_snapshot" in install["post_install"]


def test_source_only_manifest_stays_exactly_eighteen_fields() -> None:
    manifest = CONTRACT["manifest"]
    assert len(manifest["exact_ordered_fields"]) == 18
    assert manifest["exact_ordered_fields"] == MANIFEST_FIELDS
    assert set(manifest["forbidden_fields"]) == {
        "target_generation",
        "target_state_fingerprint_sha256",
        "target_transaction_fingerprint_sha256",
        "target_history_tail_fingerprint_sha256",
        "staged_sqlite_artifact_fingerprint_sha256",
    }


def test_prepare_closes_only_invoking_instance_and_staging_gate_proves_quiescence(
    tmp_path: Path,
) -> None:
    import sqlite3

    import pytest

    from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError

    staged_path = tmp_path / "staged.sqlite3"
    first = SQLiteStateStore(staged_path)
    second = SQLiteStateStore(staged_path)
    try:
        first.prepare_for_atomic_install()
        with pytest.raises(sqlite3.ProgrammingError):
            first.read_metadata()
        assert second.read_metadata() is None
        with pytest.raises(StateStoreError, match="pre-existing process-local handles"):
            with SQLiteStateStore.installation_gate(staged_path):
                raise AssertionError("gate admitted a remaining registered staging handle")
        second.close()
        with SQLiteStateStore.installation_gate(staged_path):
            pass
    finally:
        first.close()
        second.close()


def test_f2_and_f2b_separate_instance_close_from_path_quiescence() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    primitive = boundary["existing_primitives"]["prepare_for_atomic_install"]
    assert "close only that invoking instance" in primitive
    assert "does not by itself prove path-wide handle quiescence" in primitive
    sequence = boundary["sequence"]
    assert (
        "do not yet claim path-wide staging-handle quiescence"
        in sequence["F2_CHECKPOINT_AND_CLOSE_ORCHESTRATOR_HANDLE"]
    )
    assert (
        "successful held installation_gate(staged_sqlite_path)"
        in boundary["source_requirements_at_replace"]["quiescence_proof"]
    )


def test_dual_gate_order_and_lifetime_are_closed() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    assert boundary["ordering"] == [
        "F1_OPEN_FINAL_SEMANTIC_VERIFICATION",
        "F2_CHECKPOINT_AND_CLOSE_ORCHESTRATOR_HANDLE",
        "F2B_ACQUIRE_STAGING_PATH_GATE",
        "F3_CLOSED_MAIN_CONTINUITY",
        "F4_FINAL_LIVE_AND_EXTERNAL_FENCES",
    ]
    dual = boundary["dual_gate_protocol"]
    assert dual["lock_order"] == ["STAGING_PATH_GATE", "LIVE_PATH_GATE"]
    assert "no third lock framework" in dual["lock_framework"]
    assert "held continuously through F3, F4, and atomic_replace" in dual["staging_gate_lifetime"]
    assert "held through atomic_replace" in dual["live_gate_lifetime"]
    assert "same class-level path RLock" in dual["constructor_race_fence"]


def test_f3_immutable_reader_and_both_replace_gates_are_explicit() -> None:
    boundary = CONTRACT["wal_promotion_boundary"]
    f3 = boundary["sequence"]["F3_CLOSED_MAIN_CONTINUITY"]
    assert f3[0] == "execute inside continuously held staging installation_gate"
    assert "not a registered mutable SQLiteStateStore handle" in f3[2]
    requirements = boundary["source_requirements_at_replace"]
    assert requirements["staging_path_gate_held"] is True
    assert requirements["live_path_gate_held"] is True
    assert requirements["open_process_local_staged_handles"] == 0
    assert requirements["registered_mutable_live_handles"] == 0
    assert requirements["gate_order"] == "STAGING_PATH_GATE before LIVE_PATH_GATE"


def test_staging_path_not_quiescent_is_retryable_fail_closed() -> None:
    failure = CONTRACT["wal_promotion_boundary"]["failure_policy"]["staging_path_not_quiescent"]
    assert failure.startswith("STAGING_PATH_NOT_QUIESCENT / FAIL_CLOSED")
    assert "preserve staging" in failure
    assert "no atomic_replace" in failure
    assert "no cleanup" in failure
    assert "no external rollback" in failure
    assert "retry may succeed after the foreign handle closes" in failure


def test_c6b_and_c7_include_gate_acquisition_and_dual_gate_replace() -> None:
    matrix = CONTRACT["crash_matrix"]
    assert "before or after acquiring staging path gate" in matrix["C6B"]["point"]
    assert "process crash releases process-local gate" in matrix["C6B"]["restart"]
    assert "acquire fresh staging gate" in matrix["C6B"]["restart"]
    assert "no migration replay" in matrix["C6B"]["restart"]
    assert "staging and live path gates both held" in matrix["C7"]["point"]
    assert "old live plus intact completed same-directory sibling" in matrix["C7"]["restart"]
    assert "exact new live with sibling absent" in matrix["C7"]["restart"]
