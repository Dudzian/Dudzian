"""Wykonywalny kontrakt fizycznej trwałości i crash atomicity AccountGenesis."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_physical_persistence_crash_atomicity_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_physical_persistence_crash_atomicity_contract.md"
OWNER = "CryptoHunterAccountAuthority"
PROVENANCE_CLASSIFICATIONS = {
    "EXACT_COMMIT",
    "PATH_CONTENT_EQUIVALENT",
    "HIGH_CONFIDENCE_LINEAGE_EQUIVALENT",
    "LIKELY_EQUIVALENT",
    "UNKNOWN",
    "MISMATCH",
}


def load(name: str | None = None) -> dict:
    return json.loads((DOCS / name if name else MACHINE).read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis physical persistence and crash atomicity contract\n\n"
        "Ten plik jest deterministyczną, kompletną projekcją "
        "`m05_account_genesis_physical_persistence_crash_atomicity_contract.json`. "
        "JSON jest źródłem prawdy.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_PHYSICAL_PERSISTENCE_AND_CRASH_ATOMICITY_CONTRACT"
    )
    assert value["iteration"] == "DESIGN / RECONCILIATION ONLY"
    provenance = value["provenance"]
    assert provenance["actual_repository_HEAD_inspected"] == (
        value["repository_head_examined"]
    )
    assert provenance["classification"] in PROVENANCE_CLASSIFICATIONS
    assert value["reviewed_sha_supplied"] == provenance["reviewed_SHA_supplied"]
    assert value["reviewed_sha_supplied"] == "NOT_SUPPLIED"
    if provenance["reviewed_SHA_supplied"] == "NOT_SUPPLIED":
        assert provenance["reviewed_sha_state"] == "NOT_SUPPLIED"
        assert provenance["availability"] == "NOT_APPLICABLE"
        assert provenance["classification"] == "UNKNOWN"
        assert provenance["finding_scope"] == "CURRENT_TREE_ONLY"
        assert provenance["formal_advancement_allowed"] is False
        assert provenance["formal_project_advancement"] == "WITHHELD"

    frozen = value["frozen_inputs"]
    assert frozen["topology_result"] == "ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_FROZEN"
    assert frozen["selected_topology"] == "A_SINGLE_ACCOUNT_GENESIS_COORDINATOR"
    assert frozen["sole_coordinator"] == OWNER
    assert frozen["logical_commit_closure"] == "FROZEN"
    assert frozen["freshness_implementation"] == "NOT_AVAILABLE"
    assert frozen["external_root_proof_issuer"] == "NOT_AVAILABLE"
    assert frozen["required_genuine_closure"][-3:] == [
        "operation COMMITTED", "reservation CONSUMED_COMMITTED",
        "authenticated fresh history",
    ]

    roles = value["authority_vs_storage_roles"]
    assert roles["semantic_authority_owner"] == OWNER
    assert roles["physical_protocol_coordinator"] == OWNER
    assert roles["transaction_coordinator"] == OWNER
    assert roles["recovery_reconciler"] == OWNER
    assert roles["generation_candidate_proposer"] == OWNER
    assert "generation_owner" not in roles
    assert roles["local_authority_storage_adapter"].endswith("NOT_AVAILABLE")
    assert roles["external_anchor_writer"].endswith("NOT_AVAILABLE")

    freshness = value["freshness_ownership_boundary"]
    assert freshness["owner"] == (
        "Future AccountGenesis freshness / anti-rollback authority"
    )
    assert freshness["owner"] != roles["semantic_authority_owner"]
    assert freshness["owner_status"] == (
        "OWNER_CLASS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE"
    )
    assert freshness["implementation"] == "NOT_AVAILABLE"
    assert freshness["must_be_independent_from_created_account"] == "YES"
    assert freshness["owner_equals_genesis_semantic_decision_owner"] is False
    assert freshness["coordinator_may_propose_successor"] is True
    assert freshness[
        "coordinator_may_unilaterally_advance_authoritative_generation"
    ] is False
    assert freshness["candidate_generation_becomes_authoritative_on_construction"] is False
    assert freshness[
        "authoritative_generation_exists_only_after_independent_freshness_acceptance"
    ] is True
    assert freshness["valid_local_HMAC_establishes_freshness"] is False
    assert freshness[
        "external_anchor_writer_automatically_is_freshness_authority"
    ] is False
    assert freshness["successful_external_write_alone_establishes_freshness"] is False
    assert freshness["custody_writer_itself_is_freshness_authority"].startswith(
        "NOT_FROZEN"
    )

    records = {item["record"]: item["authority"] for item in value["authoritative_records"]}
    assert all(records.values())
    assert "accepted root-proof validation evidence and provenance" in records
    projection = value["projection_boundary"]
    assert projection["M0.11_SQLiteStateStore_CryptoHunterAccount"] == (
        "PROJECTION_CARRIER_ONLY / NOT_AUTHORITY"
    )
    assert projection["same_SQLite_file_changes_role"] is False
    assert projection["projection_may_replace_lost_authority_history"] is False

    candidates = value["candidate_protocols"]
    assert set(candidates) == {
        "A_SINGLE_SQLITE_AUTHORITY_TRANSACTION_PLUS_POSTCOMMIT_EXTERNAL_ANCHOR",
        "B_APPEND_ONLY_LOCAL_JOURNAL_PLUS_EXTERNAL_ANCHOR_FINALIZATION",
        "C_PREPARE_LOCAL_THEN_ANCHOR_THEN_LOCAL_FINALIZE",
        "D_LOCAL_COMMIT_PLUS_AUTHENTICATED_FINALIZATION_RECEIPT",
        "E_OTHER_CANONICALLY_SUPPORTED_PROTOCOL", "F_DESIGN_BLOCKED",
    }
    selected = value["selected_or_blocked_protocol"]
    assert selected["selection"] == (
        "INITIAL_BINDING_THEN_PREPARED_THEN_FRESHNESS_CAS_THEN_LOCAL_FINAL_COMMIT"
    )
    assert selected["protocol_result"] == "PHYSICAL_PROTOCOL_CAN_NOW_BE_FROZEN"
    assert selected["point_of_no_return"] == (
        "exact authoritative freshness acceptance N -> N+1"
    )
    assert selected["logical_authority_atomicity_equals_single_physical_transaction"] is False
    assert selected["no_partial_physical_state_publishable"] is True

    phases = {item["phase"]: item for item in value["physical_phases"]}
    assert set(phases) == {
        "NO_DURABLE_STATE", "INITIAL_BINDING_DURABLE", "LOCAL_PREPARED",
        "CAS_CONFLICT_CONFIRMED", "REPREPARED",
        "FRESHNESS_OUTCOME_UNKNOWN", "AUTHORITY_ACCEPTED_LOCAL_FINALIZATION_PENDING",
        "LOCAL_COMMITTED", "PUBLISHED",
    }
    assert phases["LOCAL_PREPARED"]["publishable"] is False
    assert phases["CAS_CONFLICT_CONFIRMED"]["publishable"] is False
    assert phases["REPREPARED"]["publishable"] is False
    assert phases["AUTHORITY_ACCEPTED_LOCAL_FINALIZATION_PENDING"]["publishable"] is False
    assert phases["LOCAL_COMMITTED"]["publishable"] is True

    transaction = value["local_transaction_contract"]
    assert transaction["INITIAL_BINDING_partial_state"].startswith("IMPOSSIBLE")
    assert len(transaction["INITIAL_BINDING_atomic_records"]) == 6
    required_fragments = (
        "operation", "canonical request", "reservation",
        "account", "root-proof", "genesis", "semantic heads",
        "finalization",
    )
    all_atomic_records = sum(
        (
            transaction["INITIAL_BINDING_atomic_records"],
            transaction["PREPARED_atomic_records"],
            transaction["FINAL_COMMIT_atomic_records"],
        ),
        [],
    )
    assert all(
        any(fragment in record for record in all_atomic_records)
        for fragment in required_fragments
    )
    assert transaction["operation_COMMITTED_with_reservation_RESERVED"] == (
        "FAIL_CLOSED / DO_NOT_PUBLISH"
    )
    assert transaction["partial_commit"] == "MUST ROLL BACK OR BE DETECTED AND FAIL CLOSED"

    anchor = value["external_anchor_contract"]
    assert anchor["writer_status"] == "NOT_AVAILABLE"
    assert anchor["Catalog_anchor_or_custody_reused"] is False
    assert anchor["current_KeyringSecretStorage_set_secret_has_atomic_CAS"] is False
    assert anchor["successful_write_followed_by_unreadable"] == "FAIL_CLOSED"
    assert anchor["same_generation_different_heads"] == "CONFLICT / FAIL_CLOSED"
    assert anchor["publication_sequence"][-1].startswith("authenticate and compare exact")

    cas = value["CAS_generation_contract"]
    assert cas["required"] is True
    assert cas["successful_writers_per_predecessor_generation"] == 1
    assert cas["current_external_CAS_available"] is False
    assert "never blind retry" in cas["concurrent_loser"]

    publication = value["publishability_rule"]
    assert publication["local_COMMITTED_before_anchor_confirmation"] == "NOT_PUBLISHABLE"
    assert publication["PREPARED_publishable"] is False
    assert publication["anchor_confirmation_without_local_history_publishable"] is False
    assert "COMMITTED_PENDING_EXTERNAL_FRESHNESS_semantic_state" not in publication
    assert publication["unresolved_CAS_conflict"] == "NOT_PUBLISHABLE"
    assert value["projection_update_order"] == [
        "commit atomic INITIAL_BINDING",
        "commit exact durable PREPARED intent",
        "complete external authoritative full-document CAS and obtain/verify exact finalization evidence",
        "commit complete authoritative local closure",
        "publish deterministic COMMITTED result as genuine authority",
        "update/rebuild CryptoHunterAccount projection idempotently",
    ]

    matrix = {row["crash_point"]: row for row in value["crash_matrix"]}
    expected_points = {
        "before INITIAL_BINDING commit", "after INITIAL_BINDING commit",
        "before PREPARED commit", "after PREPARED before CAS",
        "CAS timeout or transport ambiguity",
        "authority exact N+1 while local PREPARED",
        "authority different N+1 while local PREPARED",
        "after CAS conflict before durable conflict classification",
        "after durable CAS_CONFLICT_CONFIRMED", "during atomic REPREPARE",
        "after new REPREPARED before new CAS",
        "authority changes again before loser next CAS",
        "stale process attempts old candidate after supersession",
        "old finalization receipt appears after supersession",
        "authoritative reread unavailable during conflict resolution",
        "current authority N+2 or later during old-candidate recovery",
        "old candidate won but response was lost and authority later advanced",
        "after authority acceptance before local COMMIT", "during local final COMMIT",
        "after local COMMIT before publication", "same generation unequal digest",
        "authority unavailable or evidence unverifiable",
        "after publication before projection update", "after projection update",
    }
    assert set(matrix) == expected_points
    required_columns = {
        "crash_point", "durable_facts_present", "external_anchor_state", "restart_action",
        "publishable", "may_retry", "may_allocate_new_account",
        "manual_or_external_recovery_required",
    }
    assert all(set(row) == required_columns for row in matrix.values())
    assert matrix["after INITIAL_BINDING commit"]["may_allocate_new_account"] == "NO"
    assert matrix["before INITIAL_BINDING commit"]["durable_facts_present"].startswith("none")
    assert matrix["after PREPARED before CAS"]["publishable"] == "NO"
    assert matrix["after authority acceptance before local COMMIT"]["publishable"] == "NO"

    conflict = value["CAS_conflict_loser_resolution"]
    policy = "REPREPARE_SAME_LOGICAL_OPERATION_ATOMIC_SUPERSESSION"
    transition = (
        "LOCAL_PREPARED(old candidate) -> CAS_CONFLICT_CONFIRMED -> "
        "REPREPARED(new candidate)"
    )
    assert conflict["selected_model"] == "A_REPREPARE_REBASE_SAME_LOGICAL_OPERATION"
    assert conflict["normative_policy"] == policy
    assert conflict["exact_transition"] == transition
    assert conflict["authoritative_reread_required"] is True
    assert "FAIL_CLOSED" in conflict["unavailable_or_unverifiable_reread"]
    identities = conflict["identity_preservation"]
    assert identities["logical_operation_identity"] == "PRESERVED_EXACT"
    assert identities["reservation_identity"] == "PRESERVED_EXACT"
    assert identities["account_id"] == "PRESERVED_EXACT"
    assert identities["new_logical_operation"] is False
    assert identities["new_reservation"] is False
    assert identities["remint_account_id"] is False
    candidate = conflict["new_candidate_identity"]
    assert candidate["required"] is True
    assert candidate["proposed_document_digest"].startswith("NEW")
    assert candidate["finalization_request_id"].startswith("NEW")
    old = conflict["old_candidate_disposition"]
    assert old["status"] == "SUPERSEDED_RETAINED_HISTORY"
    assert old["sendable_after_supersession"] is False
    assert old["mutable_in_place"] is False
    assert old["may_equal_new_candidate_identity"] is False
    receipt = conflict["old_receipt_disposition"]
    assert receipt["may_close_new_candidate"] is False
    atomic = conflict["atomic_reprepare"]
    assert atomic["required"] is True
    assert atomic["crash_atomicity"].startswith("complete old PREPARED or complete new")
    assert atomic["silent_overwrite"] is False
    exact_transition = transition + (
        " by one atomic durable supersession transaction preserving "
        "operation/reservation/account/request identities"
    )
    assert exact_transition in value["allowed_physical_transitions"]
    conflict_row = matrix["authority different N+1 while local PREPARED"]
    assert conflict_row["restart_action"] == conflict["crash_matrix_policy"]

    non_acceptance = value["authoritative_non_acceptance_proof"]
    assert non_acceptance["required"] is True
    assert non_acceptance["entry_gate"] == (
        "AUTHORITATIVE_DIFFERENT_UNIQUE_SUCCESSOR_PROVEN"
    )
    assert non_acceptance["negative_lookup_is_proof"] is False
    assert non_acceptance["receipt_not_found_is_proof"] is False
    assert non_acceptance["missing_historical_lookup_is_proof"] is False
    assert non_acceptance["timeout_is_proof"] is False
    assert non_acceptance["unavailable_history_is_proof"] is False
    assert non_acceptance["unverifiable_history_is_proof"] is False
    assert non_acceptance[
        "current_generation_greater_than_old_predecessor_is_proof"
    ] is False
    direct = non_acceptance["direct_different_successor"]
    assert direct["allowed"] is True
    assert direct["accepted_generation_is_exact_N_plus_1"] is True
    assert direct["predecessor_generation_matches_old_N"] is True
    assert direct["predecessor_document_digest_matches_old_D_N"] is True
    assert direct["predecessor_complete_heads_match_old_H_N"] is True
    assert direct["accepted_document_identity_differs_from_old_candidate"] is True
    assert direct["single_successor_invariant_authority_authenticated"] is True
    assert direct["environment_trust_domain_authority_match_required"] is True
    history = non_acceptance["historical_successor_chain"]
    assert history["allowed"] is True
    assert history["required_when_current_generation_is_N_plus_2_or_later"] is True
    assert history["current_head_alone_sufficient"] is False
    assert history["exact_immediate_successor_of_old_predecessor_required"] is True
    assert history["exact_old_predecessor_generation_digest_heads_binding_required"] is True
    assert history["currently_trusted_authority_evidence_required"] is True
    assert history["verification_reaches_currently_trusted_authority_state_or_root"] is True
    assert history["no_missing_generation_required"] is True
    assert history["no_fork_required"] is True
    assert history["revoked_only_trust_sufficient"] is False
    assert history["local_projection_sufficient"] is False
    assert non_acceptance["incomplete_history"].startswith("FAIL_CLOSED")
    assert non_acceptance["unavailable_history"].startswith("FAIL_CLOSED")
    assert non_acceptance["unverifiable_history"].startswith("FAIL_CLOSED")
    assert non_acceptance["same_generation_unequal_document"] == (
        "FAIL_CLOSED / FORK_OR_TAMPER"
    )
    positive = non_acceptance["positive_old_acceptance"]
    assert positive["exact_receipt_for_old_candidate"].startswith("FINALIZE_OLD")
    assert positive["exact_authenticated_historical_acceptance"].startswith(
        "FINALIZE_OLD"
    )
    assert positive["positive_old_acceptance_evidence_forbids_reprepare"] is True

    recovery = value["restart_recovery"]
    assert recovery["local_N_plus_1_anchor_N"].startswith("Impossible")
    assert "exact durable authenticated PREPARED" in recovery["anchor_N_plus_1_local_N"]
    assert recovery["same_generation_unequal_heads"] == "FAIL_CLOSED"
    assert recovery["automatic_repair"].startswith("Only exact PREPARED")
    ambiguous = value["ambiguous_anchor_write"]
    assert ambiguous["classification"] == "WRITE_OUTCOME_UNKNOWN"
    assert ambiguous["on_exception_or_timeout"].startswith("Reread")
    assert ambiguous["successful_return_without_verified_reread"] == "NOT_PUBLISHABLE"

    concurrency = value["concurrency"]
    assert concurrency["authority_global_anchor_CAS"].startswith("REQUIRED")
    assert concurrency["process_local_mutex_sufficient"] is False
    assert concurrency["O1_O2_same_N"] == conflict["normative_policy"]
    assert concurrency["CAS_conflict_crash_matrix_policy"] == conflict["crash_matrix_policy"]
    assert concurrency["loser_resolution_transition"] == transition
    assert "does not become one-account-per-subject" in concurrency["serialization_scope"]
    multi = value["multi_lineage_anchor"]
    assert len(multi["heads"]) == 5
    assert multi["partial_document"] == "REJECT / FAIL_CLOSED"
    assert multi["mixed_closure_heads"] == "REJECT / FAIL_CLOSED"
    lost = value["lost_update_prevention"]
    assert lost["conflict_policy"] == "CAS_CONFLICT / NO_LAST_WRITE_WINS"
    assert lost["blind_overwrite"] is False
    assert lost["timestamp_arbitration"] is False
    assert lost["CAS_loser_policy"] == conflict["normative_policy"]
    assert lost["old_candidate_blind_retry"] is False
    assert lost["old_candidate_in_place_rebase"] is False

    assert value["abort_persistence"]["operation_ABORTED_terminal_immutable"] is True
    assert value["abort_persistence"]["crash_between_ABORT_and_RELEASE"] == "ABORTED_HELD"
    assert value["abort_persistence"]["after_point_of_no_return"].startswith("FORBIDDEN")
    release = value["release_persistence"]
    assert release["scope"] == "RESERVATION_ONLY"
    assert release["may_rewrite_operation_ABORTED"] is False
    assert release["account_id_reusable"] is False
    assert release["accepted_account_id_reusable"] is False
    assert release["accepted_operation_restartable"] is False
    proof = value["root_proof_evidence_persistence"]
    assert proof["required_before_publication"] is True
    assert proof["exact_historical_provenance_required"] is True
    assert proof["missing_provenance"] == "FAIL_CLOSED / NOT_PUBLISHABLE"

    keys = value["key_lifecycle_interaction"]
    assert keys["authority_commit_and_key_rotation_same_operation"] is False
    assert "exclusive maintenance boundary" in keys["storage_master_key_rotation"]
    sqlite = value["sqlite_durability_evidence"]
    assert sqlite["SQLiteStateStore_role"] == "M0.11 projection only"
    assert sqlite["SQLiteStateStore_configuration"]["journal_mode"] == "WAL required"
    assert sqlite["SQLiteStateStore_configuration"]["synchronous"] == "FULL"
    assert "does not prove hardware fsync" in sqlite["claim_limit"]
    rollback = value["rollback_boundary"]
    assert rollback["coordinated_DB_and_keyring_rollback"] == "OUT_OF_SCOPE / NOT_DETECTED"

    parity = value["cross_artifact_parity"]
    assert parity["coordinator"] == OWNER
    assert parity["operation_state_required"] == "COMMITTED"
    assert parity["reservation_state_required"] == "CONSUMED_COMMITTED"
    assert parity["PREPARED_is_genuine"] is False
    assert parity["external_freshness_required"] is True
    assert parity["TEST_persistence_or_anchor_accepted_as_PRODUCTION"] is False
    assert parity["M0.11_projection_role"] == "PROJECTION_CARRIER_ONLY / NOT_AUTHORITY"

    assert set(value["mandatory_redteam"]) == {
        "local_committed_published_before_external_confirmation",
        "operation_committed_reservation_reserved", "anchor_ahead_auto_creates_history",
        "local_ahead_auto_accepted_without_direction_proof",
        "same_generation_unequal_heads_accepted", "two_writers_finalize_same_generation",
        "last_write_wins_anchor_conflict", "partial_multi_lineage_anchor_accepted",
        "proof_provenance_lost_but_publishable", "aborted_rewritten_by_release",
        "test_accepted_as_production", "m011_projection_as_authority_after_loss",
        "catalog_anchor_reused",
        "partial_initial_binding_accepted", "prepared_published",
        "point_of_no_return_at_request_send", "abort_after_authority_acceptance",
        "release_enables_accepted_account_reuse",
        "process_local_mutex_as_cross_process_authority",
        "cas_loser_silently_rewrites_old_prepared_predecessor",
        "cas_loser_keeps_old_digest_after_predecessor_change",
        "cas_loser_treats_new_candidate_as_already_accepted_exact_old",
        "cas_loser_remints_account_id", "cas_loser_creates_new_reservation",
        "cas_loser_starts_new_logical_operation_as_retry",
        "stale_old_candidate_remains_sendable_after_supersession",
        "partial_reprepare_produces_mixed_old_new_candidate",
        "conflict_path_absent_from_allowed_transitions",
        "concurrency_policy_contradicts_crash_matrix",
        "cas_loser_abort_enables_account_id_reuse",
        "unresolved_conflict_permits_publication",
        "unavailable_authoritative_reread_permits_reprepare_or_abort",
        "old_receipt_from_superseded_candidate_closes_new_candidate",
        "receipt_not_found_treated_as_non_acceptance_proof",
        "timeout_treated_as_non_acceptance_proof",
        "unavailable_historical_lookup_allows_reprepare",
        "current_generation_ahead_alone_allows_reprepare",
        "N_plus_2_head_without_verified_immediate_history_allows_reprepare",
        "local_projection_proves_non_acceptance",
        "revoked_only_history_proves_non_acceptance",
        "same_generation_unequal_document_is_valid_successor_proof",
        "incomplete_successor_chain_allows_reprepare",
        "positive_exact_old_receipt_allows_reprepare",
        "positive_exact_old_history_allows_reprepare",
        "different_successor_wrong_predecessor_digest_allows_reprepare",
        "different_successor_wrong_predecessor_heads_allows_reprepare",
        "different_successor_other_trust_domain_allows_reprepare",
    }
    result = value["result"]
    assert result["primary_result"] == "PHYSICAL_PROTOCOL_CAN_NOW_BE_FROZEN"
    assert result["physical_protocol_frozen"] is True
    assert result["initial_binding_removes_prior_identity_recovery_blocker"] is True
    assert result["remaining_design_blockers"] == []
    assert result["CAS_conflict_loser_resolution_frozen"] is True
    assert result["authoritative_non_acceptance_proof_frozen"] is True
    assert value["implementation_allowed"]["FreshnessAuthority"] is False
    assert value["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
    assert value["preserved_status"] == {
        "WorkspaceAuthority": "NOT_AVAILABLE", "FullFillAuthority": "NOT_AVAILABLE",
        "M0.8": "BLOCKED", "production M0.5": "NOT_AVAILABLE",
        "freshness implementation": "NOT_AVAILABLE",
        "external root-proof issuer": "NOT_AVAILABLE",
        "physical persistence protocol": "FROZEN ABSTRACT DESIGN",
    }


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("classification", "EXACT_COMMIT"),
        ("classification", "CURRENT_TREE_ONLY"),
        ("classification", "NO_REVIEWED_SHA_TO_COMPARE"),
        ("formal_advancement_allowed", True),
        ("finding_scope", "FORMAL_REVIEW"),
        ("reviewed_SHA_supplied", "a" * 40),
        ("formal_project_advancement", "ALLOWED"),
        ("formal_project_advancement", "ACCEPTED"),
    ],
)
def test_no_reviewed_sha_provenance_mutations_fail(
    field: str, unsafe_value: object
) -> None:
    mutated = deepcopy(load())
    mutated["provenance"][field] = unsafe_value
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    "mutation", sorted(load()["freshness_ownership_redteam"])
)
def test_freshness_ownership_mutations_fail(mutation: str) -> None:
    mutated = deepcopy(load())
    specification = mutated["freshness_ownership_redteam"][mutation]
    target = mutated
    for component in specification["path"][:-1]:
        target = target[component]
    target[specification["path"][-1]] = specification["unsafe_value"]
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("mutation", sorted(load()["mandatory_redteam"]))
def test_mandatory_redteam_mutations_fail(mutation: str) -> None:
    mutated = deepcopy(load())
    specification = mutated["mandatory_redteam"][mutation]
    target = mutated
    for component in specification["path"][:-1]:
        target = target[component]
    target[specification["path"][-1]] = specification["unsafe_value"]
    with pytest.raises(AssertionError):
        validate(mutated)


def test_cross_artifact_parity_uses_actual_frozen_values() -> None:
    contract = load()
    topology = load("m05_account_genesis_authority_topology_resolution_after_binding_freeze.json")
    substrate = load("m05_account_genesis_security_substrate_contract.json")
    binding = load("m05_account_genesis_operation_identity_request_binding_contract.json")
    reservation = load("m05_cryptohunter_account_genesis_reservation_state_model.json")
    proof = load("m05_account_genesis_root_proof_admission_binding_contract.json")

    assert contract["cross_artifact_parity"]["coordinator"] == (
        topology["logical_commit_closure"]["coordinator"]
    )
    topology_freshness_owner = next(
        role for role in topology["role_matrix"] if role["role"] == "freshness owner"
    )
    freshness = contract["freshness_ownership_boundary"]
    assert topology_freshness_owner["candidate_owner"] == freshness["owner"]
    assert topology_freshness_owner[
        "must_be_independent_from_created_account"
    ] == "YES"
    assert freshness["owner"] != contract["authority_vs_storage_roles"][
        "semantic_authority_owner"
    ]
    assert topology["separation_invariants"][
        "freshness_owner_is_genesis_semantic_decision_owner"
    ] == freshness["owner_equals_genesis_semantic_decision_owner"]
    assert contract["projection_boundary"]["M0.11_SQLiteStateStore_CryptoHunterAccount"] == (
        topology["ownership_resolution"]["M0.11_SQLiteStateStore_role"]
    )
    assert contract["restart_recovery"]["same_generation_unequal_heads"] in (
        substrate["anchor_mismatch_semantics"]["equal_generation_unequal_heads"]
    )
    assert substrate["anchor_atomicity"]["local_and_external_atomic"] is False
    assert substrate["freshness_lineage"]["model"] == (
        contract["external_anchor_contract"]["model"]
    )
    assert binding["state_machine_binding"]["PREPARED_request_mutation"] == "FORBIDDEN"
    assert contract["cross_artifact_parity"]["reservation_state_required"] in (
        reservation["genesis_commit_binding"]["genuine_at"]
    )
    assert proof["historical_provenance"]["current_availability"] == (
        "SEMANTICS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE"
    )
    assert "exact accepted root proof" in proof["historical_provenance"]["requirement"]
    assert proof["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
