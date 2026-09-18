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
    assert selected["selection"] == "F_DESIGN_BLOCKED"
    assert selected["protocol_result"] == (
        "ACCOUNT_GENESIS_PHYSICAL_PERSISTENCE_DESIGN_BLOCKED"
    )
    assert selected["logical_authority_atomicity_equals_single_physical_transaction"] is False
    assert selected["no_partial_physical_state_publishable"] is True

    phases = {item["phase"]: item for item in value["physical_phases"]}
    assert set(phases) == {
        "NO_DURABLE_STATE", "RESERVATION_DURABLE", "LOCAL_PREPARED",
        "LOCAL_CLOSURE_COMMITTED_ANCHOR_UNCONFIRMED",
        "EXTERNAL_ANCHOR_WRITE_AMBIGUOUS", "EXTERNAL_ANCHOR_CONFIRMED", "PUBLISHED",
    }
    assert phases["LOCAL_PREPARED"]["publishable"] is False
    assert phases["LOCAL_CLOSURE_COMMITTED_ANCHOR_UNCONFIRMED"]["publishable"] is False
    assert phases["EXTERNAL_ANCHOR_CONFIRMED"]["publishable"] is True

    transaction = value["local_transaction_contract"]
    required_fragments = (
        "operation transition", "canonical request", "reservation transition",
        "account candidate", "root-proof", "genesis history", "semantic heads",
        "pending external-anchor",
    )
    assert all(
        any(fragment in record for record in transaction["atomic_records"])
        for fragment in required_fragments
    )
    assert transaction["operation_COMMITTED_with_reservation_RESERVED"] == (
        "FAIL_CLOSED / DO_NOT_PUBLISH"
    )
    assert transaction["partial_commit"] == "MUST_ROLL_BACK_OR_BE_DETECTED_AND_FAIL_CLOSED"

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
    assert publication["COMMITTED_PENDING_EXTERNAL_FRESHNESS_semantic_state"].startswith(
        "FORBIDDEN"
    )
    assert value["projection_update_order"] == [
        "commit complete authoritative local closure",
        "complete external full-document CAS and authenticated exact reread",
        "publish deterministic COMMITTED result as genuine authority",
        "update/rebuild CryptoHunterAccount projection idempotently",
    ]

    matrix = {row["crash_point"]: row for row in value["crash_matrix"]}
    expected_points = {
        "before reservation commit", "after reservation commit", "after PREPARED",
        "during local closure transaction", "after local closure commit before anchor write",
        "during anchor write", "after anchor write before anchor reread",
        "after anchor reread before publication", "after publication before projection update",
        "after projection update",
    }
    assert set(matrix) == expected_points
    required_columns = {
        "crash_point", "durable_facts_present", "external_anchor_state", "restart_action",
        "publishable", "may_retry", "may_allocate_new_account",
        "manual_or_external_recovery_required",
    }
    assert all(set(row) == required_columns for row in matrix.values())
    assert matrix["after PREPARED"]["may_allocate_new_account"] == "NO"
    assert matrix["after local closure commit before anchor write"]["publishable"] == "NO"
    assert matrix["after anchor write before anchor reread"]["publishable"] == "NO"

    recovery = value["restart_recovery"]
    assert recovery["local_N_plus_1_anchor_N"].startswith("FAIL_CLOSED")
    assert recovery["anchor_N_plus_1_local_N"].startswith("FAIL_CLOSED")
    assert recovery["same_generation_unequal_heads"] == "FAIL_CLOSED"
    assert recovery["automatic_repair"].startswith("FORBIDDEN")
    ambiguous = value["ambiguous_anchor_write"]
    assert ambiguous["classification"] == "WRITE_OUTCOME_UNKNOWN"
    assert ambiguous["on_exception_or_timeout"].startswith("Reread")
    assert ambiguous["successful_return_without_verified_reread"] == "NOT_PUBLISHABLE"

    concurrency = value["concurrency"]
    assert concurrency["authority_global_anchor_CAS"].startswith("REQUIRED")
    assert "does not become one-account-per-subject" in concurrency["serialization_scope"]
    multi = value["multi_lineage_anchor"]
    assert len(multi["heads"]) == 5
    assert multi["partial_document"] == "REJECT / FAIL_CLOSED"
    assert multi["mixed_closure_heads"] == "REJECT / FAIL_CLOSED"
    lost = value["lost_update_prevention"]
    assert lost["conflict_policy"] == "CAS_CONFLICT / NO_LAST_WRITE_WINS"
    assert lost["blind_overwrite"] is False
    assert lost["timestamp_arbitration"] is False

    assert value["abort_persistence"]["operation_ABORTED_terminal_immutable"] is True
    assert value["abort_persistence"]["crash_between_ABORT_and_RELEASE"] == "ABORTED_HELD"
    release = value["release_persistence"]
    assert release["scope"] == "RESERVATION_ONLY"
    assert release["may_rewrite_operation_ABORTED"] is False
    assert release["account_id_reusable"] is False
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
    }
    result = value["result"]
    assert result["primary_result"] == "ACCOUNT_GENESIS_PHYSICAL_PERSISTENCE_DESIGN_BLOCKED"
    assert result["physical_protocol_frozen"] is False
    assert value["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
    assert value["preserved_status"] == {
        "WorkspaceAuthority": "NOT_AVAILABLE", "FullFillAuthority": "NOT_AVAILABLE",
        "M0.8": "BLOCKED", "production M0.5": "NOT_AVAILABLE",
        "freshness implementation": "NOT_AVAILABLE",
        "external root-proof issuer": "NOT_AVAILABLE",
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
    assert proof["historical_provenance"]["current_availability"] == "NOT_AVAILABLE"
    assert "exact accepted root proof" in proof["historical_provenance"]["requirement"]
    assert proof["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
