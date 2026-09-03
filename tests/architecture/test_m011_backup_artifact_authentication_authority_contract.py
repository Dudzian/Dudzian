"""Permanent architecture guards for the M0.11 backup authentication authority."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
MACHINE = json.loads(
    (
        ROOT
        / "docs/architecture/cryptohunter_product_architecture"
        / "persistence_versioning_migrations_backup_and_recovery.json"
    ).read_text()
)
AUTH = MACHINE["backup_artifact_authentication_authority_contract"]
PHYSICAL = MACHINE["physical_sqlite_restore_artifact_contract"]
PURPOSE = "CRYPTOHUNTER_M0_11_PHYSICAL_BACKUP_AUTH_V1"


def test_exact_authority_scope_algorithm_and_key_generation() -> None:
    assert AUTH["name"] == "BackupArtifactAuthenticationAuthority"
    assert AUTH["classification"] == "PRE_EXISTING_EXTERNAL_LOCAL_SECURITY_AUTHORITY"
    assert AUTH["ownership_and_scope"]["exact_scope"] == [
        "account_id",
        "device_installation_id",
        "environment",
        "purpose",
    ]
    assert AUTH["ownership_and_scope"]["purpose"] == PURPOSE
    assert AUTH["cryptography"]["algorithm"] == "HMAC-SHA-256"
    key = AUTH["key_material"]
    assert (key["logical_name"], key["bits"], key["generation"]) == (
        "BACKUP_AUTHENTICATION_KEY",
        256,
        "CRYPTOGRAPHICALLY_SECURE_RANDOM_GENERATOR",
    )
    assert not key["candidate_derived"] and not key["deterministically_identifier_derived"]


def test_distinct_custody_and_raw_key_non_exportability() -> None:
    custody = AUTH["secure_custody"]
    assert custody["name"] == "BackupArtifactAuthenticationSecureCustody"
    assert custody["decision"] == (
        "DISTINCT_PURPOSE_SEPARATED_SECURE_CUSTODY_NAMESPACE_AND_BACKEND_INTERFACE_REQUIRED"
    )
    assert custody["backend_unavailable"] == "FAIL_CLOSED"
    raw = AUTH["key_material"]["raw_material_policy"]
    assert "non-exportable through product/application APIs" in raw
    assert {"never serialized into BackupEnvelope", "never stored in StateStore"} <= set(raw)
    assert AUTH["candidate_secret_accounting"] == {
        "raw_secret_payload_field_count": 0,
        "authority_internal_secret_custody_is_backup_payload_field": False,
    }


def test_exact_key_states_permissions_rotation_retention_and_terminal_revocation() -> None:
    lifecycle = AUTH["lifecycle"]
    assert lifecycle["exact_states"] == ["ACTIVE", "VERIFY_ONLY", "REVOKED"]
    assert lifecycle["active_cardinality"] == {
        "minimum": 0,
        "maximum": 1,
        "meaning": "AT_MOST_ONE_ACTIVE_KEY_PER_PROVISIONED_SCOPE",
        "multiple_ACTIVE": "AUTHORITY_STATE_INVALID_FAIL_CLOSED",
        "zero_ACTIVE_allowed_after": [
            "revocation of sole ACTIVE key",
            "explicit destructive removal of active key material",
            "administrative security response",
        ],
        "zero_ACTIVE_is_unprovisioned": False,
    }
    assert lifecycle["permissions"] == {
        "ACTIVE": {"authenticate": True, "verify": True},
        "VERIFY_ONLY": {"authenticate": False, "verify": True},
        "REVOKED": {"authenticate": False, "verify": False},
    }
    rotation = lifecycle["rotation"]
    assert rotation["atomic_steps"] == [
        "generate new 256-bit cryptographically random key",
        "allocate new authority_key_id",
        "store new key in secure custody",
        "make new key ACTIVE",
        "demote previous ACTIVE to VERIFY_ONLY",
    ]
    assert rotation["failure_before_commit"] == "PREVIOUS_ACTIVE_REMAINS_AUTHORITATIVE"
    assert rotation["historical_VERIFY_ONLY_verification"] is True
    assert lifecycle["retention"]["ordinary_rotation_destroys_old_key"] is False
    assert lifecycle["revocation"]["terminal"] is True
    assert lifecycle["revocation"]["candidate_override"] is False
    assert lifecycle["revocation"]["fallback_to_another_key"] is False


def test_authority_loss_provisioning_and_verify_never_auto_provisions() -> None:
    recovery = AUTH["recovery"]
    assert recovery["invariant"] == (
        "BACKUP_CANDIDATE_CANNOT_RECOVER_BACKUP_AUTHENTICATION_AUTHORITY"
    )
    assert recovery["model"] == "SAME_TRUST_DOMAIN_RECOVERY_ONLY"
    assert (
        "newly provisioned key does not validate historical proofs"
        in recovery["authority_loss_effect"]
    )
    provisioning = AUTH["provisioning"]
    assert provisioning["candidate_dependent"] is False
    assert provisioning["verify_may_auto_provision"] is False
    assert provisioning["restore_without_authority"] == (
        "RESTORE_REJECTED_AUTHORITY_NOT_PROVISIONED"
    )


def test_least_privilege_capabilities_and_current_status_verification() -> None:
    capabilities = AUTH["capabilities"]
    assert capabilities["backup_creation"] == "AUTHENTICATE_ONLY_CAPABILITY"
    assert capabilities["restore"] == "VERIFY_ONLY_CAPABILITY"
    assert capabilities["administration"]["separate_surface"] is True
    assert capabilities["administration"]["exposed_to_restore"] is False
    verify = capabilities["verify"]
    assert (
        "read one fresh internally consistent committed authority scope snapshot "
        "at the invocation linearization point" in verify["ordered_steps"]
    )
    assert verify["cached_status_sufficient"] is False
    assert verify["mutates_authority"] is False
    assert "provisioning" in verify["forbidden_side_effects"]


def test_exact_domain_separated_payload_and_proof_schema() -> None:
    payload = AUTH["cryptography"]["authentication_payload"]
    assert payload["exact_expression"] == (f'UTF8("{PURPOSE}") + 0x00 + CANONICAL_MANIFEST_UTF8')
    assert payload["canonical_json"] == {
        "encoding": "UTF-8",
        "sort_keys": True,
        "separators": [",", ":"],
        "ensure_ascii": False,
        "allow_nan": False,
    }
    assert payload["physical_hash_only_sufficient"] is False
    proof = AUTH["proof"]
    assert proof["exact_fields"] == [
        "proof_version",
        "algorithm",
        "purpose",
        "authority_key_id",
        "manifest_fingerprint_sha256",
        "authentication_tag_hex",
    ]
    assert proof["additional_fields"] is False
    assert proof["constants"] == {
        "proof_version": 1,
        "algorithm": "HMAC-SHA-256",
        "purpose": PURPOSE,
    }
    assert proof["authentication_tag_hex"] == {
        "encoding": "lowercase hexadecimal",
        "pattern": "^[0-9a-f]{64}$",
        "length": 64,
        "bytes": 32,
    }


def test_only_verified_passes_and_result_is_ephemeral_non_authority() -> None:
    result = AUTH["verification_result"]
    assert result["exact_statuses"] == [
        "VERIFIED",
        "INVALID_PROOF",
        "UNKNOWN_KEY",
        "REVOKED_KEY",
        "AUTHORITY_UNAVAILABLE",
        "AUTHORITY_NOT_PROVISIONED",
    ]
    assert result["passes_authenticity_gate"] == ["VERIFIED"]
    assert result["all_other_outcome"] == "RESTORE_REJECTED"
    assert result["durable"] is False
    assert result["ephemeral_process_local"] is True
    assert result["authority"] is False


def test_cross_scope_and_secret_authority_reuse_are_forbidden() -> None:
    cross = AUTH["cross_scope_rejection"]
    assert cross["reject_under"] == ["account B", "device D2", "TESTNET", "PAPER"]
    assert cross["lookup_scope_bound"] is True
    assert cross["cross_environment_or_install_fallback"] is False
    forbidden = set(AUTH["secret_separation"]["key_is_not"])
    assert {"exchange API key", "user PIN", "M0.3 key or reference"} <= forbidden
    assert AUTH["secret_separation"]["SecretHandoff_rotates_authority"] is False


def test_representation_classification_keeps_authority_outside_candidate() -> None:
    classifications = AUTH["representation_classification"]
    assert set(classifications) == {
        "BackupArtifactAuthenticationAuthority",
        "BackupArtifactAuthenticationAuthorityCapability",
        "BackupArtifactAuthenticationSecureCustody",
        "BackupAuthenticationKey",
        "BackupAuthenticationKeyMetadata",
        "PhysicalArtifactAuthenticationProof",
        "BackupArtifactVerificationResult",
    }
    expected_fields = {
        "durable",
        "candidate_carried",
        "authority",
        "restorable_authority",
        "contains_raw_secret",
        "permitted_in_BackupEnvelope",
        "permitted_in_StateStore",
        "process_local",
        "external_secure_custody",
    }
    assert all(set(value) == expected_fields for value in classifications.values())
    assert classifications["BackupAuthenticationKey"]["contains_raw_secret"] is True
    assert classifications["BackupAuthenticationKey"]["candidate_carried"] is False
    assert classifications["PhysicalArtifactAuthenticationProof"]["candidate_carried"] is True
    assert classifications["PhysicalArtifactAuthenticationProof"]["authority"] is False


def test_cross_contract_creation_admission_and_lifecycle_stage_separation() -> None:
    assert len(PHYSICAL["artifact_model"]["outer_bundle_components"]) == 4
    assert PHYSICAL["verification_input_boundary"]["verification_capability"]["operation"] == (
        "BackupArtifactAuthenticationAuthorityCapability.verify"
    )
    assert PHYSICAL["backup_creation"]["authority_operation"] == (
        "BackupArtifactAuthenticationAuthorityCapability.authenticate"
    )
    admission = PHYSICAL["physical_artifact_admission_boundary"]
    assert admission["accepted_verification_status"] == "VERIFIED"
    assert admission["successful_admission_grants"] == (
        "ELIGIBLE_FOR_FURTHER_RESTORE_VALIDATION_ONLY"
    )
    steps = admission["ordered_steps"]
    open_index = next(i for i, step in enumerate(steps) if step["opens_SQLite_database"])
    verify_index = next(
        i for i, step in enumerate(steps) if step["step"] == "EXTERNAL_PROOF_VERIFICATION"
    )
    assert verify_index < open_index
    assert PHYSICAL["integrity_authenticity"]["physical_artifact_sha256"] == ("BYTE_INTEGRITY_ONLY")
    assert PHYSICAL["integrity_authenticity"]["valid_proof_does_not_establish"]
    assert PHYSICAL["restore_workflow"]["canonical_authority_stages"] == [
        "STRUCTURAL_PERSISTENCE_RECORD_VALIDATION",
        "ASPECT_SPECIFIC_SEMANTIC_AUTHORITY_REVALIDATION",
        "CURRENT_DESIGNATION_RELATIONAL_REVALIDATION",
        "M0_3_RESTORE_FRESHNESS",
    ]


def test_architecture_defined_but_production_implementation_pending() -> None:
    pending = "BACKUP_ARTIFACT_AUTHENTICATION_AUTHORITY_ARCHITECTURE_DEFINED_IMPLEMENTATION_PENDING"
    assert AUTH["architecture_status"] == pending
    assert PHYSICAL["status"] == pending
    assert AUTH["production_implementation_exists"] is False
    assert PHYSICAL["trust_root_audit"]["historical_finding_retained"] is True
    model = PHYSICAL["artifact_model"]
    assert model["production_use_blocked_until_authentication_authority_implemented"] is True
    assert model["architecture_prerequisite_defined"] is True
    assert model["production_authentication_capability_exists"] is False
    assert "production_use_blocked_until_authentication_prerequisite_closed" not in model


def test_scope_conditions_are_not_key_states() -> None:
    lifecycle = AUTH["lifecycle"]
    conditions = lifecycle["scope_conditions"]
    assert conditions["classification"] == "AUTHORITY_SCOPE_CONDITIONS_NOT_KEY_STATES"
    assert conditions["exact_conditions"] == [
        "UNPROVISIONED_SCOPE",
        "PROVISIONED_WITH_ACTIVE",
        "PROVISIONED_WITHOUT_ACTIVE",
    ]
    assert conditions["UNPROVISIONED_SCOPE"] == {"key_ring_exists": False}
    assert conditions["PROVISIONED_WITH_ACTIVE"] == {
        "key_ring_exists": True,
        "ACTIVE_key_count": 1,
    }
    assert conditions["PROVISIONED_WITHOUT_ACTIVE"] == {
        "key_ring_exists": True,
        "ACTIVE_key_count": 0,
        "may_contain": ["VERIFY_ONLY", "REVOKED"],
    }
    assert conditions["candidate_carried"] is False
    assert conditions["lifecycle_records"] is False
    assert conditions["restore_authority"] is False
    assert not set(conditions["exact_conditions"]) & set(lifecycle["exact_states"])


def test_authenticate_distinguishes_scope_conditions_and_fails_closed() -> None:
    authenticate = AUTH["capabilities"]["authenticate"]
    assert authenticate["requires_state"] == "EXACTLY_ONE_ACTIVE_KEY_AT_CALL_TIME"
    assert authenticate["scope_condition_outcomes"] == {
        "UNPROVISIONED_SCOPE": "AUTHORITY_NOT_PROVISIONED",
        "PROVISIONED_WITHOUT_ACTIVE": "NO_ACTIVE_KEY",
        "PROVISIONED_WITH_ACTIVE": "AUTHENTICATE_WITH_SOLE_ACTIVE_KEY",
        "MULTIPLE_ACTIVE_KEYS": "AUTHORITY_STATE_INVALID_FAIL_CLOSED",
    }
    assert authenticate["VERIFY_ONLY_or_REVOKED_selection"] is False
    assert authenticate["silent_winner_selection"] is False


def test_zero_active_scope_can_verify_retained_verify_only_key() -> None:
    verify = AUTH["capabilities"]["verify"]
    assert verify["requires_ACTIVE_key_in_scope"] is False
    assert verify["zero_ACTIVE_with_retained_VERIFY_ONLY"] == (
        "VERIFY_ALLOWED_FOR_EXACT_VERIFY_ONLY_KEY"
    )
    assert AUTH["lifecycle"]["permissions"]["VERIFY_ONLY"]["verify"] is True
    assert AUTH["lifecycle"]["permissions"]["REVOKED"]["verify"] is False


def test_sole_active_revocation_intentionally_leaves_zero_active() -> None:
    revocation = AUTH["lifecycle"]["revocation"]
    assert "ACTIVE -> REVOKED" in revocation["allowed_transitions"]
    assert revocation["sole_ACTIVE_transition_result"] == "PROVISIONED_WITHOUT_ACTIVE"
    assert revocation["automatic_replacement"] is False
    assert revocation["revocation_implies_rotation"] is False
    assert revocation["post_sole_ACTIVE_revocation"] == {
        "backup_creation": "NO_ACTIVE_KEY",
        "revoked_key_proof": "REVOKED_KEY",
        "other_VERIFY_ONLY_proofs": "MAY_VERIFY_INDEPENDENTLY",
    }
    assert revocation["terminal"] is True


def test_normal_rotation_commits_one_active_without_visible_dual_active_state() -> None:
    rotation = AUTH["lifecycle"]["rotation"]
    assert rotation["precondition"] == "PROVISIONED_WITH_ACTIVE"
    assert rotation["committed_active_cardinality"] == 1
    assert rotation["externally_visible_two_ACTIVE_state"] is False
    assert rotation["atomic_steps"][-2:] == [
        "make new key ACTIVE",
        "demote previous ACTIVE to VERIFY_ONLY",
    ]
    assert rotation["failure_before_commit"] == "PREVIOUS_ACTIVE_REMAINS_AUTHORITATIVE"


def test_zero_active_rekey_is_admin_only_and_future_facing() -> None:
    rekey = AUTH["lifecycle"]["rekey"]
    assert rekey["operation"] == "rekey(scope)"
    assert rekey["precondition"] == "PROVISIONED_WITHOUT_ACTIVE"
    assert rekey["authority_admin_only"] is True
    assert rekey["steps"] == [
        "generate new independent 256-bit cryptographically random key",
        "allocate new authority_key_id",
        "persist new key securely",
        "atomically make new key the sole ACTIVE key",
    ]
    assert rekey["postcondition"] == "PROVISIONED_WITH_ACTIVE"
    assert rekey["existing_VERIFY_ONLY_unchanged"] is True
    assert rekey["existing_REVOKED_unchanged"] is True
    assert rekey["resurrects_REVOKED"] is False
    assert rekey["validates_historical_REVOKED_proofs"] is False
    assert rekey["derives_from_old_material"] is False
    assert rekey["candidate_or_restore_can_trigger"] is False
    assert rekey["first_use_provisioning"] is False
    assert rekey["recovers_old_authority_material"] is False
    assert rekey["effect"] == "FUTURE_BACKUP_AUTHENTICATION_ONLY"


def test_provision_rotate_and_rekey_have_disjoint_admin_preconditions() -> None:
    provisioning = AUTH["provisioning"]
    administration = AUTH["capabilities"]["administration"]
    assert provisioning["precondition"] == "UNPROVISIONED_SCOPE"
    assert administration["operation_preconditions"] == {
        "provision(scope)": "UNPROVISIONED_SCOPE",
        "rotate(scope)": "PROVISIONED_WITH_ACTIVE",
        "rekey(scope)": "PROVISIONED_WITHOUT_ACTIVE",
    }
    assert administration["candidate_selects_operation"] is False
    assert administration["exposed_to_restore"] is False
    forbidden = AUTH["capabilities"]["verify"]["forbidden_side_effects"]
    assert {"provisioning", "rotation", "key promotion", "key resurrection"} <= set(forbidden)


def test_multiple_active_corruption_rejects_authenticate_and_verify() -> None:
    corruption = AUTH["lifecycle"]["multiple_active_corruption"]
    assert corruption["invariant"] == (
        "MULTIPLE_ACTIVE_KEYS_FOR_ONE_SCOPE = AUTHORITY_STATE_INVALID_FAIL_CLOSED"
    )
    assert corruption["authenticate"] == "AUTHORITY_STATE_INVALID_FAIL_CLOSED"
    assert corruption["verify"] == "AUTHORITY_UNAVAILABLE"
    assert corruption["verification_result_mapping"] == "AUTHORITY_UNAVAILABLE"
    assert corruption["winner_selection"] is False
    assert corruption["forbidden_selection"] == [
        "arbitrary",
        "lexicographic authority_key_id",
        "latest authority_key_id",
        "candidate-selected",
    ]
    assert corruption["repair"] == "TRUSTED_ADMINISTRATION_OUTSIDE_RESTORE"


def test_rekey_after_loss_preserves_same_trust_domain_recovery_boundary() -> None:
    recovery = AUTH["recovery"]
    assert recovery["model"] == "SAME_TRUST_DOMAIN_RECOVERY_ONLY"
    assert recovery["invariant"] == (
        "BACKUP_CANDIDATE_CANNOT_RECOVER_BACKUP_AUTHENTICATION_AUTHORITY"
    )
    assert recovery["new_active_after_loss_or_reset"] == "FUTURE_BACKUPS_ONLY"
    assert recovery["new_authority_key_id_validates_destroyed_key_proofs"] is False
    assert recovery["candidate_reconstructs_lost_secret"] is False


def test_authority_operations_are_linearizable_per_exact_scope() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    assert consistency["invariant"] == (
        "AUTHORITY_OPERATIONS_ARE_LINEARIZABLE_PER_EXACT_AUTHORITY_SCOPE"
    )
    assert consistency["serialization_scope"] == "EXACT_AUTHORITY_SCOPE"
    assert consistency["exact_scope"] == [
        "account_id",
        "device_installation_id",
        "environment",
        "purpose",
    ]
    assert consistency["same_scope_rule"] == "ONE_SERIALIZED_AUTHORITATIVE_ORDER"
    assert consistency["different_scopes_may_proceed_independently"] is True
    assert consistency["global_product_lock_required"] is False
    assert set(consistency["covered_operations"]) == {
        "provision",
        "rotate",
        "rekey",
        "revoke",
        "explicit destructive key removal",
        "authenticate authority-state read",
        "verify authority-state read",
    }


def test_atomic_scope_snapshot_and_external_authority_revision() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    snapshot = consistency["scope_snapshot"]
    assert snapshot["name"] == "BackupAuthenticationAuthorityScopeSnapshot"
    assert snapshot["classification"] == "EXTERNAL_SECURE_CUSTODY_AUTHORITY_METADATA"
    assert snapshot["atomic_committed_view"] is True
    assert snapshot["one_committed_snapshot_per_scope"] is True
    assert {
        "exact authority scope",
        "authority_revision",
        "whether key ring exists",
        "all authority_key_id entries relevant to scope",
        "current lifecycle state of each known key",
        "exact ACTIVE cardinality",
    } <= set(snapshot["determines"])
    assert snapshot["raw_secret_bytes_duplicated"] is False
    assert snapshot["product_level_raw_key_reference_exposed"] is False
    assert not any(
        snapshot[field]
        for field in (
            "StateStore_data",
            "BackupEnvelope_data",
            "candidate_carried",
            "restore_authority",
            "lifecycle_restore_record",
        )
    )
    revision = consistency["authority_revision"]
    assert revision["type"] == "positive integer"
    assert revision["bool_allowed"] is False
    assert revision["first_successful_provision"] == "ABSENT_TO_REVISION_1"
    assert revision["first_successful_provision"] == "ABSENT_TO_REVISION_1"
    assert revision["subsequent_successful_admin_mutation"] == "EXACTLY_R_TO_R_PLUS_1"
    assert revision["subsequent_revision_precondition"] == "R >= 1"
    assert revision["may_decrease"] is False
    assert revision["unprovisioned_scope_revision"] is None
    assert revision["candidate_carried"] is False
    assert revision["restore_authority"] is False


def test_cas_rejects_stale_revision_and_prevents_intermediate_two_active() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    mutation = consistency["atomic_mutation"]
    assert mutation["primitive"] == "ATOMIC_COMPARE_AND_SWAP_SCOPE_SNAPSHOT"
    assert mutation["stale_outcome"] == "STALE_AUTHORITY_REVISION"
    assert mutation["automatic_reread_and_reapply"] is False
    assert mutation["successful_snapshot_must_satisfy"] == "0 <= ACTIVE_count <= 1"
    assert mutation["torn_authoritative_state"] is False
    prevention = consistency["active_cardinality_prevention"]
    assert prevention["normal_rotate_single_transition"] == [
        "old ACTIVE -> VERIFY_ONLY",
        "new key -> ACTIVE",
    ]
    assert prevention["authoritative_intermediate_two_ACTIVE_snapshot"] is False
    assert prevention["textual_preparation_steps_are_separate_commits"] is False


def test_staged_key_and_crash_before_commit_are_non_authoritative() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    staged = consistency["staged_key_material"]
    assert staged["invariant"] == (
        "STAGED_KEY_MATERIAL_IS_NON_AUTHORITATIVE_UNTIL_REFERENCED_BY_"
        "SUCCESSFULLY_COMMITTED_SCOPE_SNAPSHOT"
    )
    assert staged["before_commit_authority"] is False
    assert staged["may_authenticate"] is False
    assert staged["may_verify"] is False
    assert staged["changes_scope_condition_or_ACTIVE_cardinality"] is False
    assert staged["candidate_can_discover_select_or_promote"] is False
    assert staged["restore_may_repair_or_adopt"] is False
    crash = consistency["crash_before_commit"]
    assert crash["rule"] == "PREVIOUS_COMMITTED_SCOPE_SNAPSHOT_REMAINS_EXACTLY_AUTHORITATIVE"
    assert crash["outcomes"] == {
        "provision": "UNPROVISIONED_SCOPE",
        "rotate": "PREVIOUS_ACTIVE_REMAINS_ACTIVE",
        "rekey": "PROVISIONED_WITHOUT_ACTIVE",
        "revoke": "TARGET_KEY_RETAINS_PRIOR_LIFECYCLE_STATE",
        "explicit destructive key removal": (
            "PRIOR_METADATA_AND_KEY_MATERIAL_REMAIN_AUTHORITATIVE"
        ),
    }
    assert crash["partial_authority_transition"] is False


def test_committed_response_loss_requires_reconciliation_not_blind_retry() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    lost = consistency["commit_succeeded_response_lost"]
    assert lost["rule"] == ("SUCCESSFULLY_COMMITTED_AUTHORITY_TRANSITION_REMAINS_AUTHORITATIVE")
    assert "R_PLUS_1" not in lost["rule"]
    assert lost["caller_may_assume_failure"] is False
    assert lost["first_provision_committed_response_lost"] == {
        "committed_transition": "ABSENT_TO_REVISION_1",
        "authoritative_result": "REVISION_1_REMAINS_AUTHORITATIVE",
        "caller_may_assume_provision_failed": False,
        "reconciliation": "FRESH_EXACT_SCOPE_RECONCILIATION_REQUIRED",
        "blind_repeat": False,
    }
    assert lost["subsequent_mutation_committed_response_lost"] == {
        "committed_transition": "R_TO_R_PLUS_1",
        "authoritative_result": "R_PLUS_1_REMAINS_AUTHORITATIVE",
        "reconciliation": "FRESH_EXACT_SCOPE_RECONCILIATION_REQUIRED",
        "blind_repeat": False,
    }
    assert lost["invariant"] == "UNKNOWN_ADMIN_MUTATION_OUTCOME_NO_BLIND_RETRY"
    assert set(lost["blind_retry_forbidden"]) == {
        "rotate",
        "rekey",
        "provision",
        "revoke",
        "explicit destructive key removal",
    }
    assert lost["required_next_action"] == (
        "READ_COMMITTED_SCOPE_SNAPSHOT_AND_RECONCILE_REVISION_AND_STATE"
    )
    retry = consistency["retry_policy"]
    assert retry["known_pre_commit_failure"] == "MAY_RETRY_FROM_FRESH_COMMITTED_SNAPSHOT"
    assert retry["stale_revision"] == "FRESH_READ_AND_REEVALUATION_REQUIRED"
    assert retry["unknown_post_commit_outcome"] == (
        "STATE_RECONCILIATION_REQUIRED_BEFORE_ANOTHER_MUTATION"
    )
    assert retry["blind_at_least_once_retry"] is False
    assert retry["candidate_visible_mutation_id"] is False


def test_concurrent_rotate_rekey_and_provision_allow_at_most_one_commit() -> None:
    concurrent = AUTH["consistency_and_concurrency"]["concurrent_administration"]
    rotate = concurrent["rotate_vs_rotate"]
    assert rotate["same_start_revision"] == "AT_MOST_ONE_COMMITS_R_TO_R_PLUS_1"
    assert rotate["loser"] == "STALE_AUTHORITY_REVISION"
    assert rotate["reinterpret_against_R_plus_1"] is False
    rekey = concurrent["rekey_vs_rekey"]
    assert rekey["same_zero_active_revision"] == "AT_MOST_ONE_COMMITS_R_TO_R_PLUS_1"
    assert rekey["loser"] == "STALE_AUTHORITY_REVISION"
    assert rekey["loser_staged_key"] == "NON_AUTHORITATIVE"
    assert rekey["second_ACTIVE_commit"] is False
    provision = concurrent["provision_vs_provision"]
    assert provision["winner"] == "ESTABLISHES_REVISION_1"
    assert provision["loser"] == "ALREADY_PROVISIONED_OR_STALE_FAIL_CLOSED"
    assert provision["last_writer_wins"] is False
    assert provision["winner_replaced"] is False


def test_rotate_revoke_order_is_linearized_and_stale_admin_cannot_roll_back() -> None:
    concurrent = AUTH["consistency_and_concurrency"]["concurrent_administration"]
    race = concurrent["rotate_vs_revoke"]
    assert race["rotate_first"].startswith("new key ACTIVE; old key VERIFY_ONLY")
    assert race["sole_ACTIVE_revoke_first"].startswith("scope PROVISIONED_WITHOUT_ACTIVE")
    assert race["stale_rotate_undoes_revocation"] is False
    assert concurrent["stale_rekey_revoke_or_other_admin_cannot"] == [
        "resurrect revoked keys",
        "overwrite newer ACTIVE",
        "change newer lifecycle decisions",
        "roll authority state backward",
    ]


def test_authenticate_and_verify_have_precise_read_linearization_points() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    authenticate = consistency["authenticate_linearization"]
    assert authenticate["snapshot"] == "ONE_INTERNALLY_CONSISTENT_COMMITTED_SCOPE_SNAPSHOT"
    assert authenticate["linearization_point"] == (
        "ATOMIC_COMMITTED_SCOPE_SNAPSHOT_READ_DURING_INVOCATION"
    )
    assert authenticate["outcomes"]["PROVISIONED_WITHOUT_ACTIVE"] == "NO_ACTIVE_KEY"
    assert authenticate["later_admin_commit_retroactively_invalidates_operation"] is False
    assert authenticate["later_revocation_may_make_proof_unverifiable"] is True
    assert authenticate["cross_resource_ACID_with_backup_publication"] is False
    verify = consistency["verify_linearization"]
    assert verify["snapshot"] == (
        "ONE_FRESH_INTERNALLY_CONSISTENT_COMMITTED_SCOPE_SNAPSHOT_PER_INVOCATION"
    )
    assert verify["revocation_before_read"] == "REVOKED_KEY"
    assert verify["verify_read_before_revocation_commit"] == (
        "MAY_COMPLETE_USING_PRIOR_ACTIVE_OR_VERIFY_ONLY_STATUS"
    )
    assert verify["later_invocations_after_revocation"] == "MUST_OBSERVE_REVOKED"
    assert verify["cached_previous_invocation_status_sufficient"] is False
    assert verify["wall_clock_atomicity_across_in_flight_call_claimed"] is False


def test_missing_committed_key_material_and_destructive_removal_fail_closed() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    material = consistency["key_material_read_consistency"]
    assert material["ACTIVE_or_VERIFY_ONLY_requires_readable_authority_owned_material"] is True
    assert material["metadata_valid_but_material_missing"] == "AUTHORITY_UNAVAILABLE"
    assert material["unknown_key_id"] == "UNKNOWN_KEY"
    assert material["fallback_to_another_key"] is False
    removal = consistency["destructive_key_removal"]
    assert removal["trusted_administration_only"] is True
    assert removal["revision_fenced_serialized_mutation"] is True
    assert removal["safe_order"] == [
        "atomically commit R+1 metadata transition marking the key REVOKED and "
        "authority-internally non-usable",
        "only after that commit physically destroy secret bytes",
    ]
    assert removal["committed_usable_entry_may_reference_already_destroyed_material"] is False
    assert removal["destroyed_material_resurrection"] is False
    assert removal["candidate_or_restore_access"] is False


def test_authority_revision_is_absent_from_all_candidate_contracts() -> None:
    revision = AUTH["consistency_and_concurrency"]["authority_revision"]
    assert revision["PhysicalArtifactAuthenticationProof_field"] is False
    assert revision["PhysicalSQLiteArtifactManifest_field"] is False
    assert revision["BackupEnvelope_field"] is False
    assert revision["M0_3_generation"] is False
    assert revision["migration_revision"] is False
    assert revision["StateStore_generation"] is False
    assert "authority_revision" not in AUTH["proof"]["exact_fields"]
    assert "authority_revision" not in PHYSICAL["manifest"]["exact_fields"]
    assert "authority_revision" not in MACHINE["backup_contract"]["schema"]["required"]


def test_secure_custody_contract_requires_atomic_implementation_independent_semantics() -> None:
    custody = AUTH["secure_custody"]
    assert custody["required_consistency_semantics"] == [
        "atomic committed per-scope snapshot read",
        "revision-fenced scope update",
        "immutable key-material staging",
        "durable key-material persistence before committed snapshot reference",
        "committed metadata visibility",
        "no torn authoritative scope state",
    ]
    assert custody["implementation_independent"] is True
    assert custody["mandated_mechanisms"] == []


def test_first_provision_uses_absent_scope_create_not_numeric_revision_cas() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    revision = consistency["authority_revision"]
    assert revision["unprovisioned_scope_revision"] is None
    assert revision["numeric_zero_for_unprovisioned"] is False
    assert revision["null_plus_one_arithmetic"] is False
    assert revision["first_successful_provision"] == "ABSENT_TO_REVISION_1"
    create = consistency["first_provision_atomic_create"]
    assert create["operation"] == "provision(scope)"
    assert create["precondition"] == "UNPROVISIONED_SCOPE"
    assert create["primitive"] == "ATOMIC_CREATE_SCOPE_SNAPSHOT_IF_ABSENT"
    assert create["equivalent_name"] == "COMPARE_ABSENT_AND_CREATE_REVISION_1"
    assert create["linearization_point"] == "SUCCESSFUL_ATOMIC_FIRST_SCOPE_SNAPSHOT_CREATE"
    assert create["numeric_R_plus_1_used"] is False
    assert create["result_revision"] == 1
    assert create["result_ACTIVE_count"] == 1


def test_first_provision_durably_stages_key_before_absent_scope_commit() -> None:
    create = AUTH["consistency_and_concurrency"]["first_provision_atomic_create"]
    assert create["flow"] == [
        "read one authoritative scope-absence view proving no committed key ring or scope "
        "snapshot exists for exact scope",
        "validate UNPROVISIONED_SCOPE provision precondition",
        "generate new 256-bit key material and authority_key_id",
        "durably persist immutable staged key material in secure custody",
        "construct first snapshot with authority_revision 1 and exactly one ACTIVE key",
        "atomically create snapshot only if scope remains unprovisioned and committed snapshot "
        "remains absent",
        "otherwise fail closed because another provision won",
    ]
    assert create["stale_outcome"] == "ALREADY_PROVISIONED_OR_STALE_FAIL_CLOSED"
    assert create["implementation_independent"] is True


def test_subsequent_mutations_retain_numeric_revision_cas_with_positive_r() -> None:
    mutation = AUTH["consistency_and_concurrency"]["atomic_mutation"]
    assert mutation["primitive"] == "ATOMIC_COMPARE_AND_SWAP_SCOPE_SNAPSHOT"
    assert mutation["applies_only_when"] == "NUMERIC_COMMITTED_AUTHORITY_REVISION_R_EXISTS"
    assert mutation["numeric_revision_precondition"] == "R >= 1"
    assert mutation["excludes_first_provision"] is True
    assert mutation["covered_operations"] == [
        "rotate",
        "rekey",
        "revoke",
        "explicit destructive key removal",
        "any future authority-state mutation",
    ]
    assert (
        "durably persist newly generated immutable staged secret key material when needed"
        in (mutation["flow"])
    )


def test_concurrent_provision_loser_cannot_reinterpret_or_overwrite() -> None:
    provision = AUTH["consistency_and_concurrency"]["concurrent_administration"][
        "provision_vs_provision"
    ]
    assert provision["commit_cardinality"] == "AT_MOST_ONE_ABSENT_TO_REVISION_1_COMMIT"
    assert provision["winner"] == "ESTABLISHES_REVISION_1"
    assert provision["loser"] == "ALREADY_PROVISIONED_OR_STALE_FAIL_CLOSED"
    assert provision["loser_cannot"] == [
        "overwrite winner",
        "create revision 2 by automatic reinterpretation",
        "convert command into rotate",
        "replace winner ACTIVE key",
        "automatically retry against newly provisioned scope",
    ]
    assert provision["last_writer_wins"] is False


def test_uncertain_first_provision_requires_fresh_reconciliation() -> None:
    unknown = AUTH["consistency_and_concurrency"]["commit_succeeded_response_lost"]
    assert unknown["invariant"] == "UNKNOWN_ADMIN_MUTATION_OUTCOME_NO_BLIND_RETRY"
    provision = unknown["uncertain_first_provision"]
    assert provision["required"] == "FRESH_EXACT_SCOPE_AUTHORITY_STATE_READ_AND_RECONCILIATION"
    assert provision["if_revision_1_exists"] == "TREAT_AS_PROVISIONED_AND_RECONCILE"
    assert provision["if_scope_absent"] == (
        "NEW_PROVISION_MAY_BE_CONSIDERED_ONLY_AFTER_FRESH_RECONCILIATION"
    )
    assert provision["blind_repeat_original_command"] is False


def test_key_material_is_durable_before_any_referencing_snapshot_commit() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    staged = consistency["staged_key_material"]
    assert staged["durability_invariant"] == (
        "COMMITTED_SCOPE_SNAPSHOT_MUST_NEVER_REFERENCE_NEW_KEY_MATERIAL_THAT_WAS_NOT_"
        "DURABLY_PERSISTED_BEFORE_THE_SNAPSHOT_COMMIT"
    )
    assert staged["key_producing_operations"] == ["provision", "rotate", "rekey"]
    assert staged["required_order"] == [
        "generate key",
        "durably persist immutable staged key material",
        "commit authoritative scope snapshot referencing that key",
    ]
    assert staged["durable_persistence_failure"] == "DO_NOT_COMMIT_SCOPE_SNAPSHOT"
    assert staged["durability_failure_outcomes"] == {
        "provision": "UNPROVISIONED_SCOPE",
        "rotate": "PREVIOUS_ACTIVE_REMAINS_ACTIVE",
        "rekey": "PROVISIONED_WITHOUT_ACTIVE",
    }
    assert staged["post_commit_new_key_is_durably_available"] is True
    assert staged["volatile_only_key_may_be_committed"] is False


def test_orphan_gc_cannot_delete_committed_or_inflight_eligible_material() -> None:
    staged = AUTH["consistency_and_concurrency"]["staged_key_material"]
    assert staged["crash_before_snapshot_commit"] == "ORPHAN_NON_AUTHORITATIVE_CUSTODY_OBJECT"
    assert staged["may_authenticate"] is False
    assert staged["may_verify"] is False
    gc = staged["orphan_gc"]
    assert gc["delete_only_if"] == [
        "not referenced by any committed authority scope snapshot",
        "not required by an in-progress authority mutation that may still commit",
    ]
    assert gc["may_delete_committed_ACTIVE_or_VERIFY_ONLY_material"] is False
    assert gc["must_coordinate_with_in_progress_commit_eligibility"] is True
    assert gc["specific_mechanism_mandated"] is False


def test_successful_commit_cannot_depend_on_volatile_new_key_material() -> None:
    durability = AUTH["consistency_and_concurrency"]["commit_success_key_durability"]
    assert durability["rule"] == (
        "EVERY_NEWLY_REFERENCED_USABLE_KEY_IS_DURABLY_AVAILABLE_BEFORE_SNAPSHOT_COMMIT"
    )
    assert durability["applies_to"] == ["provision", "rotate", "rekey"]
    assert durability["crash_immediately_after_commit"] == (
        "COMMITTED_METADATA_AND_REFERENCED_KEY_MATERIAL_REMAIN_DURABLE"
    )
    assert (
        durability["normal_success_can_produce_committed_metadata_with_volatile_only_missing_key"]
        is False
    )
    assert (
        AUTH["consistency_and_concurrency"]["key_material_read_consistency"][
            "metadata_valid_but_material_missing"
        ]
        == "AUTHORITY_UNAVAILABLE"
    )


def test_snapshot_bootstrap_existence_and_candidate_token_exclusion() -> None:
    consistency = AUTH["consistency_and_concurrency"]
    existence = consistency["scope_snapshot"]["bootstrap_existence_semantics"]
    assert existence["UNPROVISIONED_SCOPE"] == "NO_COMMITTED_SCOPE_SNAPSHOT"
    assert existence["PROVISIONED_SCOPE"] == "EXACTLY_ONE_CURRENT_COMMITTED_SCOPE_SNAPSHOT"
    assert existence["historical_internal_revisions_may_be_retained_for_audit"] is True
    assert existence["historical_revision_is_current_authority"] is False
    assert set(consistency["candidate_visibility"].values()) == {False}


def test_markdown_is_exact_projection_of_canonical_json() -> None:
    parts = [
        "# Persistence Versioning, Migrations, Backup and Recovery\n\n"
        "Canonical machine-readable source: "
        "`persistence_versioning_migrations_backup_and_recovery.json`.\n"
    ]
    for key, value in MACHINE.items():
        parts.append(
            f"\n## `{key}`\n\n```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
        )
    projection = (
        ROOT
        / "docs/architecture/cryptohunter_product_architecture"
        / "persistence_versioning_migrations_backup_and_recovery.md"
    )
    assert projection.read_text() == "".join(parts)
