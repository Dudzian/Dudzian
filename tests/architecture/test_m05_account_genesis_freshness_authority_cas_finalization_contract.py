"""Wykonywalny kontrakt freshness authority, CAS i finalizacji AccountGenesis."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_freshness_authority_cas_finalization_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_freshness_authority_cas_finalization_contract.md"
OWNER = "Future AccountGenesis freshness / anti-rollback authority"
EXPECTED_PAYLOAD_FIELDS = {
    "schema_version", "environment", "trust_domain", "authority_id", "generation",
    "predecessor_generation", "predecessor_document_digest",
    "complete_semantic_head_set", "freshness_authority_key_id",
    "freshness_authority_key_version",
    "finalization_request_id",
}
EXPECTED_CAS_ARGUMENTS = {
    "expected_generation", "expected_document_digest", "expected_complete_heads",
    "proposed_document_payload", "proposed_document_digest", "proposer_authentication",
}
EXPECTED_CAS_PRECONDITIONS = {
    "current_full_document_matches_expected_generation",
    "current_full_document_matches_expected_digest",
    "current_complete_heads_match_expected_heads",
    "candidate_generation_is_exact_successor",
    "candidate_predecessor_matches_expected_document",
    "candidate_complete_heads_bound",
    "candidate_authentication_required",
    "environment_match_required",
    "trust_domain_match_required",
    "authority_id_match_required",
    "finalization_request_identity_match_required",
    "proposed_payload_digest_matches_proposed_document_digest",
    "proposed_payload_exactly_matches_CAS_candidate_identity",
    "accepted_payload_must_equal_proposed_payload",
    "accepted_document_digest_must_equal_proposed_document_digest",
    "authority_may_not_rewrite_candidate_payload",
    "authorized_proposer_identity_match_required",
    "trusted_proposer_key_required",
    "proposer_environment_match_required",
    "proposer_trust_domain_match_required",
    "proposer_document_digest_binding_required",
    "proposer_key_lifecycle_eligible_required",
}
EXPECTED_FINALIZATION_RECEIPT_FIELDS = {
    "schema_version", "environment", "trust_domain", "authority_id",
    "exact_predecessor_generation", "exact_predecessor_document_digest",
    "accepted_generation", "accepted_document_digest", "complete_semantic_head_digest",
    "finalization_request_id", "receipt_id", "freshness_authority_key_id",
    "freshness_authority_key_version", "authentication_tag_or_signature",
}
EXPECTED_FINALIZATION_RECEIPT_AUTH_BINDING = {
    "finalization-receipt-authentication-domain-separator", "schema_version",
    "environment", "trust_domain", "authority_id", "exact_predecessor_generation",
    "exact_predecessor_document_digest", "accepted_generation",
    "accepted_document_digest", "complete_semantic_head_digest",
    "finalization_request_id", "receipt_id", "freshness_authority_key_id",
    "freshness_authority_key_version",
}
EXPECTED_REPLAY_IDENTITY = {
    "environment", "trust_domain", "authority_id", "finalization_request_id",
    "exact_predecessor_generation", "exact_predecessor_document_digest",
    "exact_proposed_document_digest", "exact_complete_semantic_head_set_digest",
}
EXPECTED_ORIGINAL_DECISION_IDENTITY = {
    "environment", "trust_domain", "authority_id", "exact_predecessor_generation",
    "exact_predecessor_document_digest", "accepted_generation",
    "accepted_document_digest", "complete_semantic_head_digest", "finalization_request_id",
}
EXPECTED_BOOTSTRAP_DENY = {
    "acct_A", "OperatorIdentity of acct_A", "DeviceInstallation of acct_A",
    "Workspace of acct_A", "account-scoped authority of acct_A",
}
EXPECTED_PROPOSER_AUTH_FIELDS = {
    "schema_version", "proposer_identity", "environment", "trust_domain",
    "proposer_key_id", "proposer_key_version",
    "signed_or_authenticated_document_digest", "authentication_tag_or_signature",
}


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis freshness authority CAS and finalization contract\n\n"
        "Ten plik jest deterministyczną, kompletną projekcją "
        "`m05_account_genesis_freshness_authority_cas_finalization_contract.json`. "
        "JSON jest źródłem prawdy.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_CAS_AND_FINALIZATION_CONTRACT"
    )
    provenance = value["provenance"]
    assert provenance["actual_repository_HEAD_inspected"] == value["repository_head_examined"]
    assert provenance["reviewed_SHA_supplied"] == "NOT_SUPPLIED"
    assert provenance["classification"] == "UNKNOWN"
    assert provenance["finding_scope"] == "CURRENT_TREE_ONLY"
    assert provenance["formal_advancement_allowed"] is False
    assert provenance["formal_project_advancement"] == "WITHHELD"

    frozen = value["frozen_inputs"]
    ownership = value["ownership_boundary"]
    assert frozen["semantic_authority_owner"] == "CryptoHunterAccountAuthority"
    assert frozen["generation_candidate_proposer"] == "CryptoHunterAccountAuthority"
    assert frozen["freshness_generation_authority_owner"] == OWNER
    assert frozen["freshness_owner_equals_genesis_semantic_decision_owner"] is False
    assert frozen["coordinator_may_unilaterally_advance_authoritative_generation"] is False
    assert ownership["authoritative_freshness_owner"] == OWNER
    assert ownership["CHA_may_accept_authoritative_successor"] is False

    bootstrap = value["authority_bootstrap"]
    assert set(bootstrap["must_not_derive_from"]) == EXPECTED_BOOTSTRAP_DENY
    assert bootstrap["root_proof_issuer_same_role"] is False
    identity = value["authority_identity"]
    assert identity["account_id_allowed"] is False
    assert identity["unknown_authority"] == "REJECT"

    state = value["candidate_vs_authoritative_state"]
    assert state["candidate_anchor_document_equals_authoritative_freshness_document"] is False
    assert state["candidate_is_freshness"] is False
    assert state["acceptor"] == OWNER
    assert state["without_acceptance"] == "no authoritative N+1"
    assert state["method_returned_true_is_historical_proof"] is False

    document = value["anchor_document_contract"]
    assert document["payload_type"] == "AccountGenesisFreshnessDocumentPayload"
    assert set(document["payload_exact_semantic_fields"]) == EXPECTED_PAYLOAD_FIELDS
    assert document["envelope_exact_fields"] == [
        "payload", "document_digest", "authentication_tag_or_signature",
    ]
    assert document["document_digest_semantics"] == (
        "HASH(domain_separator || canonical(AccountGenesisFreshnessDocumentPayload))"
    )
    assert document["document_digest_field_excluded_from_own_preimage"] is True
    assert document["authentication_material_excluded_from_document_digest_preimage"] is True
    assert document["separate_candidate_payload_digest_present"] is False
    assert document["candidate_or_proposed_digest_name"] == "proposed_document_digest"
    assert document["semantic_preimage"] == "FROZEN"
    assert document["finalization_request_id_is_in_digest_preimage"] is True
    assert document["changing_finalization_request_id_changes_document_identity"] is True
    proposed = document["proposed_candidate_object"]
    assert proposed["type"] == "ProposedAccountGenesisFreshnessDocument"
    assert proposed["exact_fields"] == [
        "payload", "document_digest", "proposer_authentication",
    ]
    assert proposed["payload_type"] == "AccountGenesisFreshnessDocumentPayload"
    assert proposed["document_digest"] == (
        "HASH(domain_separator || canonical(payload))"
    )
    assert proposed["proposer_authentication"].startswith(
        "semantic authentication binding over domain_separator, exact payload and document_digest"
    )
    assert proposed["single_source_of_candidate_truth"] is True
    assert proposed["proposer_authentication_type"] == "ProposerAuthentication"
    assert set(proposed["proposer_authentication_exact_fields"]) == EXPECTED_PROPOSER_AUTH_FIELDS
    assert proposed["timestamps_present"] is False
    assert document["unknown_fields"] == "REJECT"
    assert document["timestamps_required"] is False

    cas = value["CAS_contract"]
    assert cas["conceptual_operation"] == "compare_and_advance"
    assert set(cas["arguments"]) == EXPECTED_CAS_ARGUMENTS
    assert set(cas["preconditions"]) == EXPECTED_CAS_PRECONDITIONS
    assert all(cas["preconditions"].values())
    assert cas["compare_and_advance_proposed_document_digest_required"] is True
    assert cas["freshness_authority_may_modify_proposed_payload"] is False
    assert set(cas["payload_fields_validated"]) == EXPECTED_PAYLOAD_FIELDS
    race = cas["key_lifecycle_race"]
    assert race["if_key_binding_no_longer_valid_or_eligible"] == (
        "REJECT / REQUIRE NEW PROPOSAL"
    )
    assert race["authority_may_rewrite_K1_V1_to_K2_V2_and_accept"] is False
    assert cas["proposer_authentication_failure_no_fallback"] is True
    assert cas["generation_gap"].startswith("FORBIDDEN")
    assert cas["read_compare_set_emulation"] == "FORBIDDEN / TOCTOU"
    assert cas["availability"] == "NO COMMITTED PUBLICATION"
    assert set(cas["statuses"]) == {
        "ACCEPTED", "ALREADY_ACCEPTED_EXACT", "CAS_CONFLICT", "PREDECESSOR_MISMATCH",
        "INVALID_DOCUMENT", "INVALID_AUTHENTICATION", "WRONG_DOMAIN", "UNAVAILABLE",
        "OUTCOME_UNKNOWN",
    }

    evidence = value["finalization_evidence"]
    assert evidence["required"].startswith("FreshnessFinalizationReceipt")
    assert set(evidence["fields"]) == EXPECTED_FINALIZATION_RECEIPT_FIELDS
    assert evidence["accepted_document_digest_binding"] == (
        "FreshnessFinalizationReceipt.accepted_document_digest == exact "
        "proposed_document_digest accepted by the same CAS decision"
    )
    assert evidence["complete_semantic_head_digest_semantics"] == (
        "HASH(semantic-head-domain || canonical(complete_semantic_head_set))"
    )
    assert evidence["complete_semantic_head_digest_semantic_preimage"] == "FROZEN"
    assert evidence["key_field_semantics"].startswith(
        "freshness_authority_key_id/version identify only the freshness authority key"
    )
    receipt_auth = evidence["authentication_boundary"]
    assert receipt_auth["object_type"] == "FreshnessFinalizationReceipt"
    assert set(receipt_auth["authentication_binding"]) == (
        EXPECTED_FINALIZATION_RECEIPT_AUTH_BINDING
    )
    assert receipt_auth["authentication_tag_or_signature_in_own_preimage"] is False
    assert receipt_auth["semantic_preimage"] == "FROZEN"
    assert receipt_auth["cryptographic_algorithm"] == "NOT_FROZEN"
    assert receipt_auth["provider"] == "NOT_FROZEN"
    assert receipt_auth["canonical_byte_encoding"] == "NOT_FROZEN"
    assert receipt_auth["schema_version_authenticated_before_interpretation"] is True
    assert receipt_auth[
        "authenticated_key_identity_drives_receipt_verification_historical_verification_and_lifecycle"
    ] is True
    assert receipt_auth["lifecycle_or_verifier_identity_may_use_unauthenticated_metadata"] is False
    assert receipt_auth[
        "provider_alias_or_material_reuse_may_bypass_receipt_key_lifecycle"
    ] is False
    assert receipt_auth[
        "changing_freshness_authority_key_id_or_version_requires_new_authentication"
    ] is True
    assert receipt_auth["K1_receipt_may_be_treated_as_K2_receipt"] is False
    assert receipt_auth["key_identity_rule"] == (
        "freshness_authority_key_id and freshness_authority_key_version used for receipt "
        "verification, historical verification and lifecycle evaluation MUST exactly equal "
        "the authenticated values in the FreshnessFinalizationReceipt preimage"
    )
    assert receipt_auth["accepted_document_digest_rule"] == (
        "authenticated accepted_document_digest == exact proposed_document_digest accepted "
        "by the same CAS decision"
    )
    assert evidence["full_document_alone_sufficient"] is False
    assert evidence["receipt_authenticity_alone_establishes_freshness"] is False
    direction = value["direction_proof"]
    assert direction["local_state_alone_may_mint_direction"] is False
    assert direction["receipt_without_monotonic_single_successor_state"] is False

    recovery = value["recovery_semantics"]
    assert recovery["local_N_plus_1_authority_N"] == "CONDITIONAL_AUTOMATIC_EXACT_CAS_RETRY"
    assert recovery["condition_failure"] == "FAIL_CLOSED"
    assert "exact proposed_document_payload, proposed_document_digest" in recovery["conditions"][0]
    assert recovery["missing_exact_proposed_candidate_identity"] == "FAIL_CLOSED"
    assert recovery["authority_N_plus_1_local_N"].startswith("FAIL_CLOSED")
    ambiguous = value["ambiguous_outcome"]
    assert ambiguous["initial_status"] == "OUTCOME_UNKNOWN"
    assert "never assume success or failure" in ambiguous["mandatory_action"]
    assert ambiguous["different_successor_accepted"].startswith("CAS_CONFLICT")
    assert value["idempotency"]["exact_replay"].startswith("ALREADY_ACCEPTED_EXACT")
    assert set(value["idempotency"]["identity_requires"]) == EXPECTED_REPLAY_IDENTITY
    assert value["idempotency"]["ALREADY_ACCEPTED_EXACT_receipt_binding"] == (
        "receipt.accepted_document_digest == original proposed_document_digest and exact "
        "finalization_request_id matches"
    )
    assert "never ALREADY_ACCEPTED_EXACT" in value["idempotency"][
        "different_finalization_request_id_for_otherwise_same_successor_semantics"
    ]
    lookup = value["idempotency"]["historical_lookup_contract"]
    assert value["idempotency"]["receipt_reissue"].startswith(
        "original receipt with unchanged authenticated preimage"
    )
    assert lookup["exact_same_authoritative_decision_required"] is True
    assert set(lookup["exact_bindings"]) == {
        "environment", "trust_domain", "authority_id", "exact_predecessor_generation",
        "exact_predecessor_document_digest", "accepted_generation",
        "accepted_document_digest", "complete_semantic_head_digest",
        "finalization_request_id", "original_decision_identity",
    }
    assert lookup["may_create_second_authority_decision"] is False
    assert lookup["currently_trusted_authority_evidence_required"] is True
    assert lookup["may_derive_trust_solely_from_historical_REVOKED_key"] is False
    assert lookup["may_rewrite_original_candidate_or_document_identity"] is False
    assert lookup["may_change_finalization_request_id"] is False
    assert lookup["may_change_accepted_document_digest"] is False
    assert lookup["original_receipt_authenticated_preimage_unchanged"] is True
    assert lookup["alternate_transport_provider_algorithm"] == "NOT_FROZEN"
    assert lookup["original_decision_identity_type"] == "OriginalDecisionIdentity"
    assert set(lookup["original_decision_identity_semantic_tuple"]) == (
        EXPECTED_ORIGINAL_DECISION_IDENTITY
    )
    assert lookup[
        "missing_currently_trusted_independent_root_may_yield_ALREADY_ACCEPTED_EXACT"
    ] is False

    identity = value["original_decision_identity_contract"]
    assert identity["type"] == "OriginalDecisionIdentity"
    assert set(identity["semantic_tuple"]) == EXPECTED_ORIGINAL_DECISION_IDENTITY
    assert identity["semantic_preimage"] == "FROZEN"
    assert identity["canonical_byte_encoding"] == "NOT_FROZEN"
    assert identity["hash_algorithm"] == "NOT_FROZEN"
    assert identity["is_caller_selected"] is False
    assert identity["is_lookup_selected"] is False
    assert identity["is_receipt_id"] is False
    assert identity["may_be_minted_during_historical_lookup"] is False
    assert identity["changing_finalization_request_id_changes_identity"] is True
    assert identity["changing_accepted_document_digest_changes_identity"] is True
    assert identity["historical_lookup_must_reference_already_existing_identity"] is True

    independent = value["revoked_history_independent_proof_contract"]
    assert independent["applies_to_all_allowed_proof_shapes"] is True
    assert independent["currently_trusted_authority_root_or_state_required"] is True
    assert independent["may_derive_trust_solely_from_revoked_historical_key"] is False
    assert independent["revoked_key_authentication_may_participate_as_historical_artifact"] is True
    assert independent["revoked_key_authentication_may_be_ultimate_trust_root"] is False
    assert independent["proof_must_bind_exact_same_original_decision_identity"] is True
    assert independent[
        "proof_must_be_verifiable_without_treating_revoked_key_as_current_trust_anchor"
    ] is True
    assert independent["proof_cannot_be_self_corroboration_of_same_revoked_credential"] is True
    assert independent["same_revoked_key_signing_receipt_and_corroborating_record_is_independent"] is False
    assert independent["missing_currently_trusted_independent_root"] == (
        "UNAVAILABLE / FAIL_CLOSED / NO LOCAL COMMITTED CLOSURE"
    )
    assert independent["valid_signature_or_tag_under_revoked_K1_alone_establishes_corroboration"] is False
    shapes = independent["proof_shapes"]
    retained = shapes["retained_monotonic_authority_decision_record"]
    assert retained["belongs_to_authority_controlled_monotonic_decision_history"] is True
    assert retained["binds_exact_original_decision_identity"] is True
    assert retained["history_integrity_currently_trusted_independently_of_revoked_key"] is True
    assert retained["record_authenticated_solely_by_revoked_key_is_sufficient"] is False
    assert retained["caller_or_local_projection_may_manufacture_record"] is False
    assert retained["local_DB_copy_alone_is_authority_proof"] is False
    successor = shapes["authenticated_successor_chain_evidence"]
    assert successor["commits_to_exact_original_decision_identity"] is True
    assert successor["verification_reaches_currently_trusted_authority_root_or_state"] is True
    assert successor["trust_path_terminating_only_in_revoked_key_is_sufficient"] is False
    assert successor[
        "same_revoked_key_may_sign_receipt_and_chain_to_establish_independence"
    ] is False
    assert successor["may_rewrite_original_candidate_request_or_document_identity"] is False
    attested = shapes["currently_trusted_authority_attested_historical_lookup"]
    assert attested["currently_trusted_authority_evidence_required"] is True
    assert attested["may_derive_trust_solely_from_historical_REVOKED_key"] is False

    concurrency = value["concurrency"]
    assert concurrency["successful_successors_per_predecessor"] == 1
    assert concurrency["same_generation_unequal_documents"] == "FORBIDDEN"
    assert concurrency["last_write_wins"] is False
    assert concurrency["timestamp_arbitration"] is False
    assert concurrency["lexicographic_digest_choice"] is False
    multi = value["multi_lineage_atomicity"]
    assert multi["full_document_atomic_CAS_required"] is True
    assert multi["independent_per_lineage_CAS"] is False

    lifecycle = value["key_lifecycle"]
    assert lifecycle["states"] == ["ACTIVE", "VERIFY_ONLY", "REVOKED"]
    assert lifecycle["ACTIVE"] == "may finalize new transitions and verify history"
    assert lifecycle["VERIFY_ONLY"] == (
        "normal retired/rotated historical key; historical verification allowed; "
        "new finalization forbidden"
    )
    assert lifecycle["REVOKED"] == (
        "compromise or suspected-compromise terminal security state; new finalization "
        "forbidden; revoked-key authentication alone cannot establish historical acceptance"
    )
    assert lifecycle["planned_rotation_target"] == "VERIFY_ONLY"
    assert lifecycle["planned_rotation_may_target_REVOKED"] is False
    assert lifecycle["revoked_state_terminal"] is True
    assert lifecycle["REVOKED_may_transition_to_ACTIVE"] is False
    assert lifecycle["REVOKED_may_transition_to_VERIFY_ONLY"] is False
    assert lifecycle["historical_receipt_under_REVOKED_requires_independent_authority_proof"] is True
    assert lifecycle["revoked_receipt_authentication_alone_establishes_historical_acceptance"] is False
    assert lifecycle["revoked_receipt_authentication_alone_establishes_ALREADY_ACCEPTED_EXACT"] is False
    assert lifecycle["revoked_receipt_authentication_alone_establishes_recovery_direction"] is False
    assert lifecycle["revoked_receipt_authentication_alone_establishes_local_COMMITTED"] is False
    assert lifecycle["revoked_receipt_authentication_alone_establishes_genuine_prior_finalization"] is False
    assert lifecycle["missing_independent_revoked_history_proof"] == (
        "UNAVAILABLE / FAIL_CLOSED / NO LOCAL COMMITTED CLOSURE"
    )
    assert lifecycle["independent_proof_must_bind_exact_same_decision_identity"] is True
    assert set(lifecycle["revoked_historical_independent_proof_abstract_options"]) == {
        "retained monotonic authority decision record",
        "authenticated successor-chain evidence committing to exact historical document and decision",
        "currently trusted authority-attested historical lookup",
    }
    assert lifecycle["wall_clock_timestamp_arbitration"].startswith("FORBIDDEN")
    assert lifecycle["historical_receipt_verification"].startswith("required after K+1")
    assert lifecycle["independent_key_boundary"] is True
    assert value["authentication_model"]["rule"] == "authenticity != freshness"
    assert value["authentication_model"]["valid_HMAC_alone_establishes_freshness"] is False
    assert value["authentication_model"][
        "successful_writer_return_without_atomic_CAS_establishes_freshness"
    ] is False

    proposer = value["proposer_authentication_boundary"]
    assert proposer["authorized_proposer_identity"] == "CryptoHunterAccountAuthority"
    assert proposer["candidate_creator_must_equal_authorized_proposer"] is True
    assert proposer["trusted_proposer_verification_lineage"] == (
        "C_PROPOSER_KEY_LINEAGE_NOT_YET_FROZEN"
    )
    assert proposer["exact_key_provider"] == "NOT_FROZEN"
    assert proposer["proposer_verification_trust_available_before_AccountGenesis"] == "REQUIRED"
    assert proposer["created_account_may_provision_its_own_proposer_trust"] is False
    assert set(proposer["trust_must_not_derive_from"]) == EXPECTED_BOOTSTRAP_DENY
    assert proposer["unknown_proposer_identity"] == "INVALID_AUTHENTICATION / REJECT"
    assert proposer["untrusted_or_unknown_proposer_key"] == "INVALID_AUTHENTICATION / REJECT"
    assert proposer["wrong_environment_or_trust_domain"] == "WRONG_DOMAIN / REJECT"
    assert proposer["self_asserted_proposer_identity"] == "FORBIDDEN"
    assert proposer["cryptographic_validity_alone_authorizes_proposer"] is False
    assert proposer["candidate_carried_key_may_establish_trust"] is False
    assert proposer["TOFU"] == "FORBIDDEN"
    assert proposer["authentication_algorithm"] == "NOT_FROZEN"
    assert set(proposer["authentication_binding"]) == {
        "proposer-authentication-domain-separator", "schema_version", "proposer_identity",
        "environment", "trust_domain", "proposer_key_id", "proposer_key_version",
        "exact proposed_document_digest",
    }
    assert proposer["authentication_tag_or_signature_in_own_preimage"] is False
    assert proposer["authenticated_key_identity_drives_verification_and_lifecycle"] is True
    assert proposer["lifecycle_eligibility_may_use_unauthenticated_key_metadata"] is False
    assert proposer["provider_alias_or_material_reuse_may_bypass_authenticated_key_lifecycle"] is False
    assert proposer["changing_proposer_key_id_or_version_requires_new_authentication"] is True
    assert proposer["P1_authentication_may_be_treated_as_P2_authentication"] is False
    assert proposer["schema_version_authenticated_before_interpretation"] is True
    assert proposer["key_identity_rule"] == (
        "proposer_key_id and proposer_key_version used for verification and lifecycle "
        "eligibility MUST exactly equal the values in the authenticated "
        "ProposerAuthentication preimage"
    )
    assert proposer["valid_proposer_authentication_establishes_authoritative_freshness"] is False
    assert proposer["valid_proposer_authentication_establishes_genuine_account"] is False
    assert proposer["valid_proposer_authentication_establishes_genesis_COMMITTED"] is False
    assert proposer["must_be_distinct_security_roles"] is True
    assert proposer["proposer_key_lifecycle"]["states"] == ["ACTIVE", "VERIFY_ONLY", "REVOKED"]
    assert proposer["proposer_key_lifecycle"]["ACTIVE"] == (
        "may authenticate new candidates and historical provenance"
    )
    assert proposer["proposer_key_lifecycle"]["VERIFY_ONLY"].startswith(
        "historical provenance verification only"
    )
    assert proposer["proposer_key_lifecycle"]["REVOKED"] == (
        "cannot authorize new candidate"
    )
    assert proposer["proposer_rotation_race"]["if_P1_not_eligible_at_CAS"] == (
        "INVALID_AUTHENTICATION / REQUIRE NEW PROPOSAL"
    )
    assert proposer["proposer_rotation_race"]["authority_may_rewrite_P1_to_P2_and_accept"] is False
    assert proposer["proposer_rotation_race"][
        "key_identity_version_authenticated_before_lifecycle_eligibility"
    ] is True

    domain = value["domain_isolation"]
    assert domain["TEST_PRODUCTION_separated"] is True
    assert domain["TEST_receipt_in_PRODUCTION"] == "REJECT / WRONG_DOMAIN"
    rollback = value["rollback_boundary"]
    assert rollback["local_DB_only_rollback_authority_newer"].startswith("DETECTABLE")
    assert rollback["authority_only_rollback_local_newer"].startswith("DETECTABLE")
    assert rollback["both_local_and_authority_rolled_back"].startswith("NOT_DETECTABLE")

    models = value["candidate_models"]
    assert set(models) == {
        "A_OS_KEYRING_SINGLE_DOCUMENT_CAS", "B_SEPARATE_LOCAL_MONOTONIC_STORE",
        "C_REMOTE_TRANSACTIONAL_FRESHNESS_SERVICE",
        "D_SIGNED_FINALIZATION_RECEIPT_AUTHORITY",
        "E_HARDWARE_MONOTONIC_COUNTER_OR_TPM_STYLE_ROOT",
        "F_HYBRID_CAS_PLUS_SIGNED_FINALIZATION_RECEIPT", "G_DESIGN_BLOCKED",
    }
    selected = value["selected_or_blocked_model"]
    assert selected["selected"] == "F_HYBRID_CAS_PLUS_SIGNED_FINALIZATION_RECEIPT"
    assert selected["backend"] == "NOT_FROZEN"
    assert selected["current_atomic_CAS_available"] is False
    assert "blind keyring set" in selected["Catalog_anchor"]

    assert set(value["mandatory_redteam"]) == set(MANDATORY_REDTEAM_MUTATIONS)
    assert set(value["proposer_authentication_redteam"]) == set(PROPOSER_AUTH_MUTATIONS)
    assert set(value["finalization_receipt_authentication_redteam"]) == set(
        RECEIPT_AUTH_MUTATIONS
    )
    assert set(value["freshness_finalization_lifecycle_redteam"]) == set(
        FRESHNESS_LIFECYCLE_MUTATIONS
    )
    assert set(value["revoked_history_independent_proof_redteam"]) == set(
        REVOKED_HISTORY_PROOF_MUTATIONS
    )
    assert all(
        result == {"mutation": "REJECT", "expected": "contract validation fails"}
        for result in value["mandatory_redteam"].values()
    )
    assert len(value["mandatory_redteam"]) == 15
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_CONTRACT_FROZEN"
    )
    assert value["result"]["physical_persistence_unblocked"] is False
    assert value["implementation_allowed"]["FreshnessAuthority"] == "NO"
    assert value["implementation_allowed"]["CryptoHunterAccountAuthority"] == "NO"
    assert value["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"


def test_contract_and_deterministic_projection() -> None:
    value = load()
    validate(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


MANDATORY_REDTEAM_MUTATIONS = {
    "CHA becomes freshness authority owner":
        (("ownership_boundary", "authoritative_freshness_owner"), "CryptoHunterAccountAuthority"),
    "valid HMAC alone establishes freshness":
        (("authentication_model", "valid_HMAC_alone_establishes_freshness"), True),
    "writer success without CAS establishes freshness":
        (("authentication_model", "successful_writer_return_without_atomic_CAS_establishes_freshness"), True),
    "read-compare-write emulation accepted as atomic CAS":
        (("CAS_contract", "read_compare_set_emulation"), "ALLOWED"),
    "two N+1 successors accepted": (("concurrency", "successful_successors_per_predecessor"), 2),
    "N -> N+2 accepted": (("CAS_contract", "generation_gap"), "ALLOWED"),
    "same generation unequal documents accepted":
        (("concurrency", "same_generation_unequal_documents"), "ACCEPT"),
    "TEST receipt accepted in PRODUCTION":
        (("domain_isolation", "TEST_receipt_in_PRODUCTION"), "ACCEPT"),
    "receipt from unknown authority accepted": (("authority_identity", "unknown_authority"), "ACCEPT"),
    "receipt authenticity without monotonic state considered freshness":
        (("finalization_evidence", "receipt_authenticity_alone_establishes_freshness"), True),
    "freshness authority bootstrapped from acct_A":
        (("authority_bootstrap", "must_not_derive_from"), ["Workspace of acct_A"]),
    "ambiguous write assumed failed without reread":
        (("ambiguous_outcome", "mandatory_action"), "assume failure"),
    "ambiguous write assumed successful without evidence":
        (("ambiguous_outcome", "initial_status"), "ACCEPTED"),
    "anchor-ahead reconstructs missing local authority history":
        (("recovery_semantics", "authority_N_plus_1_local_N"), "RECONSTRUCT_FROM_ANCHOR"),
    "local state alone authorizes recovery direction":
        (("direction_proof", "local_state_alone_may_mint_direction"), True),
}

PROPOSER_AUTH_MUTATIONS = {
    "arbitrary self-signed proposer accepted as CryptoHunterAccountAuthority":
        ("boundary", "candidate_carried_key_may_establish_trust", True),
    "unknown proposer key accepted":
        ("boundary", "untrusted_or_unknown_proposer_key", "ACCEPT"),
    "wrong proposer identity accepted":
        ("boundary", "authorized_proposer_identity", "SomeOtherAuthority"),
    "TEST proposer credential accepted in PRODUCTION":
        ("remove_binding", "environment", None),
    "proposer authentication key treated as freshness-finalization authority":
        ("boundary", "must_be_distinct_security_roles", False),
    "proposer identity binding removed":
        ("remove_field", "proposer_identity", None),
    "proposed digest binding removed from proposer authentication":
        ("remove_binding", "exact proposed_document_digest", None),
    "candidate-carried key enables TOFU":
        ("boundary", "TOFU", "ALLOWED"),
    "proposer key id removed from authentication binding":
        ("remove_binding", "proposer_key_id", None),
    "proposer key version removed from authentication binding":
        ("remove_binding", "proposer_key_version", None),
    "proposer schema version removed from authentication binding":
        ("remove_binding", "schema_version", None),
    "REVOKED proposer key may authorize new candidate":
        ("lifecycle", "REVOKED", "may authorize new candidate"),
    "VERIFY_ONLY proposer key may authorize new candidate":
        ("lifecycle", "VERIFY_ONLY", "may authorize new candidate"),
    "P1 authentication may be relabeled as P2 without new authentication":
        ("boundary", "P1_authentication_may_be_treated_as_P2_authentication", True),
    "proposer key lifecycle eligibility may use unauthenticated key metadata":
        ("boundary", "lifecycle_eligibility_may_use_unauthenticated_key_metadata", True),
}

RECEIPT_AUTH_MUTATIONS = {
    "receipt key id removed from authentication binding":
        ("remove_binding", "freshness_authority_key_id", None),
    "receipt key version removed from authentication binding":
        ("remove_binding", "freshness_authority_key_version", None),
    "receipt schema version removed from authentication binding":
        ("remove_binding", "schema_version", None),
    "receipt accepted document digest removed from authentication binding":
        ("remove_binding", "accepted_document_digest", None),
    "receipt predecessor digest removed from authentication binding":
        ("remove_binding", "exact_predecessor_document_digest", None),
    "receipt finalization request id removed from authentication binding":
        ("remove_binding", "finalization_request_id", None),
    "receipt authority id removed from authentication binding":
        ("remove_binding", "authority_id", None),
    "receipt id removed from authentication binding":
        ("remove_binding", "receipt_id", None),
    "K1 receipt may be relabeled as K2 without new authentication":
        ("set", "K1_receipt_may_be_treated_as_K2_receipt", True),
    "receipt lifecycle eligibility may use unauthenticated key metadata":
        ("set", "lifecycle_or_verifier_identity_may_use_unauthenticated_metadata", True),
    "provider alias may bypass receipt key lifecycle":
        ("set", "provider_alias_or_material_reuse_may_bypass_receipt_key_lifecycle", True),
    "authentication tag is included in its own preimage":
        ("set", "authentication_tag_or_signature_in_own_preimage", True),
}

FRESHNESS_LIFECYCLE_MUTATIONS = {
    "REVOKED freshness key may finalize new transition":
        ("lifecycle", "REVOKED", "may finalize new transitions"),
    "REVOKED receipt signature alone establishes historical acceptance":
        ("lifecycle", "revoked_receipt_authentication_alone_establishes_historical_acceptance", True),
    "REVOKED receipt alone yields ALREADY_ACCEPTED_EXACT":
        ("lifecycle", "revoked_receipt_authentication_alone_establishes_ALREADY_ACCEPTED_EXACT", True),
    "REVOKED receipt alone authorizes recovery direction":
        ("lifecycle", "revoked_receipt_authentication_alone_establishes_recovery_direction", True),
    "REVOKED receipt alone closes local COMMITTED state":
        ("lifecycle", "revoked_receipt_authentication_alone_establishes_local_COMMITTED", True),
    "planned rotation moves ACTIVE directly to REVOKED":
        ("lifecycle", "planned_rotation_target", "REVOKED"),
    "REVOKED key may transition back to ACTIVE":
        ("lifecycle", "REVOKED_may_transition_to_ACTIVE", True),
    "REVOKED key may transition back to VERIFY_ONLY":
        ("lifecycle", "REVOKED_may_transition_to_VERIFY_ONLY", True),
    "missing independent proof for revoked historical receipt is accepted":
        ("lifecycle", "missing_independent_revoked_history_proof", "ACCEPT"),
    "historical receipt under VERIFY_ONLY is rejected merely because key is not ACTIVE":
        ("lifecycle", "VERIFY_ONLY", "historical verification forbidden"),
    "historical lookup result may derive trust solely from revoked key":
        ("lookup", "may_derive_trust_solely_from_historical_REVOKED_key", True),
    "historical lookup result may represent a second authority decision":
        ("lookup", "may_create_second_authority_decision", True),
    "historical lookup may rewrite original finalization_request_id":
        ("lookup", "may_change_finalization_request_id", True),
    "historical lookup may rewrite accepted_document_digest":
        ("lookup", "may_change_accepted_document_digest", True),
}

REVOKED_HISTORY_PROOF_MUTATIONS = {
    "retained decision record trusted solely because revoked key authenticates it":
        ("retained", "record_authenticated_solely_by_revoked_key_is_sufficient", True),
    "successor chain terminates only in revoked key":
        ("successor", "trust_path_terminating_only_in_revoked_key_is_sufficient", True),
    "revoked receipt self-corroborates as independent proof":
        ("common", "proof_cannot_be_self_corroboration_of_same_revoked_credential", False),
    "same revoked key signs receipt and corroborating history record":
        ("common", "same_revoked_key_signing_receipt_and_corroborating_record_is_independent", True),
    "local DB projection accepted as retained authority decision record":
        ("retained", "local_DB_copy_alone_is_authority_proof", True),
    "independent proof need not reach currently trusted authority root":
        ("common", "currently_trusted_authority_root_or_state_required", False),
    "independent proof need not bind exact original decision identity":
        ("common", "proof_must_bind_exact_same_original_decision_identity", False),
    "original decision identity may be caller selected":
        ("identity", "is_caller_selected", True),
    "original decision identity may be minted during lookup":
        ("identity", "may_be_minted_during_historical_lookup", True),
    "receipt_id is treated as original authority decision identity":
        ("identity", "is_receipt_id", True),
    "changing finalization_request_id preserves original decision identity":
        ("identity", "changing_finalization_request_id_changes_identity", False),
    "changing accepted_document_digest preserves original decision identity":
        ("identity", "changing_accepted_document_digest_changes_identity", False),
    "historical lookup can use revoked key as ultimate trust root":
        ("lookup", "may_derive_trust_solely_from_historical_REVOKED_key", True),
    "missing currently trusted independent root still yields ALREADY_ACCEPTED_EXACT":
        ("lookup", "missing_currently_trusted_independent_root_may_yield_ALREADY_ACCEPTED_EXACT", True),
}


@pytest.mark.parametrize("attack", MANDATORY_REDTEAM_MUTATIONS)
def test_mandatory_redteam_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    path, replacement = MANDATORY_REDTEAM_MUTATIONS[attack]
    mutated[path[0]][path[1]] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("attack", PROPOSER_AUTH_MUTATIONS)
def test_proposer_authentication_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    operation, field, replacement = PROPOSER_AUTH_MUTATIONS[attack]
    if operation == "boundary":
        mutated["proposer_authentication_boundary"][field] = replacement
    elif operation == "remove_binding":
        mutated["proposer_authentication_boundary"]["authentication_binding"].remove(field)
    elif operation == "lifecycle":
        mutated["proposer_authentication_boundary"]["proposer_key_lifecycle"][field] = replacement
    else:
        mutated["anchor_document_contract"]["proposed_candidate_object"][
            "proposer_authentication_exact_fields"
        ].remove(field)
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("attack", RECEIPT_AUTH_MUTATIONS)
def test_finalization_receipt_authentication_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    operation, field, replacement = RECEIPT_AUTH_MUTATIONS[attack]
    boundary = mutated["finalization_evidence"]["authentication_boundary"]
    if operation == "remove_binding":
        boundary["authentication_binding"].remove(field)
    else:
        boundary[field] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("attack", FRESHNESS_LIFECYCLE_MUTATIONS)
def test_freshness_finalization_lifecycle_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    target, field, replacement = FRESHNESS_LIFECYCLE_MUTATIONS[attack]
    if target == "lifecycle":
        mutated["key_lifecycle"][field] = replacement
    else:
        mutated["idempotency"]["historical_lookup_contract"][field] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("attack", REVOKED_HISTORY_PROOF_MUTATIONS)
def test_revoked_history_independent_proof_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    target, field, replacement = REVOKED_HISTORY_PROOF_MUTATIONS[attack]
    common = mutated["revoked_history_independent_proof_contract"]
    if target == "common":
        common[field] = replacement
    elif target == "retained":
        common["proof_shapes"]["retained_monotonic_authority_decision_record"][field] = replacement
    elif target == "successor":
        common["proof_shapes"]["authenticated_successor_chain_evidence"][field] = replacement
    elif target == "identity":
        mutated["original_decision_identity_contract"][field] = replacement
    else:
        mutated["idempotency"]["historical_lookup_contract"][field] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    "field",
    [
        "authority_id", "exact_predecessor_document_digest", "accepted_document_digest",
        "complete_semantic_head_digest", "finalization_request_id",
        "freshness_authority_key_id", "freshness_authority_key_version",
        "authentication_tag_or_signature",
    ],
)
def test_removing_security_critical_receipt_field_fails(field: str) -> None:
    mutated = deepcopy(load())
    mutated["finalization_evidence"]["fields"].remove(field)
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    "precondition",
    [
        "authority_id_match_required", "finalization_request_identity_match_required",
        "candidate_complete_heads_bound", "candidate_generation_is_exact_successor",
        "current_full_document_matches_expected_digest",
    ],
)
def test_weakening_security_critical_cas_precondition_fails(precondition: str) -> None:
    mutated = deepcopy(load())
    mutated["CAS_contract"]["preconditions"][precondition] = False
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    "binding",
    [
        "finalization_request_id", "authority_id", "exact_predecessor_generation",
        "exact_predecessor_document_digest", "exact_proposed_document_digest",
        "exact_complete_semantic_head_set_digest",
    ],
)
def test_removing_exact_replay_binding_fails(binding: str) -> None:
    mutated = deepcopy(load())
    mutated["idempotency"]["identity_requires"].remove(binding)
    with pytest.raises(AssertionError):
        validate(mutated)


CANDIDATE_BINDING_MUTATIONS = {
    "CAS omits proposed_document_payload":
        ("remove_argument", "proposed_document_payload"),
    "CAS omits proposed_document_digest":
        ("remove_argument", "proposed_document_digest"),
    "freshness authority rewrites proposed payload":
        ("set", ("CAS_contract", "freshness_authority_may_modify_proposed_payload", True)),
    "accepted digest differs from proposed digest":
        ("set_precondition", "accepted_document_digest_must_equal_proposed_document_digest"),
    "key lifecycle silently rewrites candidate identity":
        ("set_race", "authority_may_rewrite_K1_V1_to_K2_V2_and_accept"),
    "request ID excluded from digest identity":
        ("set", ("anchor_document_contract", "finalization_request_id_is_in_digest_preimage", False)),
    "request ID change preserves document identity":
        ("set", ("anchor_document_contract", "changing_finalization_request_id_changes_document_identity", False)),
}


@pytest.mark.parametrize("attack", CANDIDATE_BINDING_MUTATIONS)
def test_candidate_binding_mutations_fail(attack: str) -> None:
    mutated = deepcopy(load())
    operation, detail = CANDIDATE_BINDING_MUTATIONS[attack]
    if operation == "remove_argument":
        mutated["CAS_contract"]["arguments"].remove(detail)
    elif operation == "set_precondition":
        mutated["CAS_contract"]["preconditions"][detail] = False
    elif operation == "set_race":
        mutated["CAS_contract"]["key_lifecycle_race"][detail] = True
    else:
        section, field, replacement = detail
        mutated[section][field] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


def test_cross_artifact_parity_reads_actual_sources() -> None:
    contract = load()
    topology = json.loads(
        (DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.json")
        .read_text(encoding="utf-8")
    )
    physical = json.loads(
        (DOCS / "m05_account_genesis_physical_persistence_crash_atomicity_contract.json")
        .read_text(encoding="utf-8")
    )
    substrate = json.loads(
        (DOCS / "m05_account_genesis_security_substrate_contract.json")
        .read_text(encoding="utf-8")
    )
    root_proof = json.loads(
        (DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
        .read_text(encoding="utf-8")
    )
    operation_binding = json.loads(
        (DOCS / "m05_account_genesis_operation_identity_request_binding_contract.json")
        .read_text(encoding="utf-8")
    )

    topology_roles = {row["role"]: row for row in topology["role_matrix"]}
    freshness = topology_roles["freshness owner"]
    root_issuer = topology_roles["independent root-proof issuer"]
    assert freshness["candidate_owner"] == contract["ownership_boundary"][
        "authoritative_freshness_owner"
    ]
    assert freshness["must_be_independent_from_created_account"] == "YES"
    assert topology["separation_invariants"][
        "freshness_owner_is_genesis_semantic_decision_owner"
    ] is False
    assert root_issuer["candidate_owner"] != freshness["candidate_owner"]

    physical_freshness = physical["freshness_ownership_boundary"]
    assert physical_freshness["owner"] == freshness["candidate_owner"]
    assert physical_freshness[
        "coordinator_may_unilaterally_advance_authoritative_generation"
    ] is False
    assert physical["CAS_generation_contract"]["current_external_CAS_available"] is False
    assert physical["selected_or_blocked_protocol"]["selection"] == "F_DESIGN_BLOCKED"
    assert physical["authority_vs_storage_roles"]["generation_candidate_proposer"] == (
        "CryptoHunterAccountAuthority"
    )

    assert substrate["anchor_models"]["selected"] == "C. atomic multi-lineage anchor document"
    assert substrate["anchor_models"]["catalog_anchor_reused"] is False
    assert substrate["anchor_atomicity"]["local_and_external_atomic"] is False
    assert substrate["anchor_mismatch_semantics"]["automatic_repair"].startswith("FORBIDDEN")
    separation = substrate["production_test_separation"]
    assert separation["authority_domain"] == "DISTINCT"
    assert separation["key_material"] == "DISTINCT"
    assert separation["freshness_state"] == "DISTINCT"
    assert substrate["cryptographic_domains"]["environment_rule"].startswith(
        "Every preimage binds exactly one PRODUCTION or TEST authority domain"
    )
    assert topology["non_circular_root"]["account_scoped_authority_allowed"] is False
    assert operation_binding["implementation_allowed"]["CryptoHunterAccountAuthority"] is False

    assert root_proof["frozen_inputs"]["root_proof_issuer"] == (
        "NOT_FOUND / conditional external provisioning issuer"
    )
    assert root_proof["authority_boundary"]["issuer"] != freshness["candidate_owner"]
    assert root_proof["non_circular_root"]["account_scoped_authority_allowed"] is False
