"""Executable contract for the M0.5 genesis reservation/operation model."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_genesis_reservation_state_model.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_genesis_reservation_state_model.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 CryptoHunterAccount genesis reservation operation state model\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_genesis_reservation_state_model.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["reviewed_head_supplied"] == ("76afe748f3d130684c9daa0f9e0b4368c233f227")
    provenance = value["provenance"]
    assert provenance["classification"] in {
        "EXACT_COMMIT",
        "PATH_CONTENT_EQUIVALENT",
        "HIGH_CONFIDENCE_LINEAGE_EQUIVALENT",
        "LIKELY_EQUIVALENT",
        "UNKNOWN",
        "MISMATCH",
    }
    assert provenance["reviewed_head_available_locally"] is False
    assert provenance["classification"] == "UNKNOWN"
    assert value["repository_head_examined"] != value["reviewed_head_supplied"]

    identity = value["identity_boundaries"]
    assert identity["account_id"] == "entity identity only"
    assert identity["account_id_is_operation_identity"] is False
    assert identity["reservation_is_genuine_account"] is False
    assert identity["reserved_account_id_is_authority"] is False
    assert identity["subject_cardinality"] == "NOT_FROZEN"

    assert value["candidate_state_models"]["E_DESIGN_BLOCKED"]["status"] == "SELECTED"
    assert value["selected_or_blocked_state_model"]["selection"] == "DESIGN_BLOCKED"
    operation = value["operation_state_machine"]
    assert "not genuine account" in operation["PREPARED"]
    assert "terminal successful" in operation["COMMITTED"]
    assert operation["forbidden_transition"].startswith("UNKNOWN -> COMMITTED")
    assert operation["terminal_states"] == ["COMMITTED", "ABORTED"]
    assert operation["terminal_outcome_immutable"] is True
    assert "RELEASED" not in operation["conceptual_states"]
    assert operation["ABORTED_to_RELEASED_operation_transition"] == "FORBIDDEN"
    assert "reservation RESERVED alone does not imply operation RESERVED" in operation["RESERVED"]
    reservation = value["reservation_state_machine"]
    assert reservation["owner"] == "DESIGN_BLOCKED"
    assert reservation["account_id_as_durable_reservation_key"] == "NOT_FROZEN"
    assert reservation["separate_reservation_id"] == "NOT_FROZEN"
    assert reservation["missing_lookup"].startswith("FAIL_CLOSED")
    assert reservation["restart_safe_reservation_recovery_identity"] == (
        "REQUIRED_IF_RESERVATION_PRECEDES_OPERATION_ID"
    )
    assert reservation["durable_reservation_serializable_authority_scope_required"] is True
    assert reservation["serializable_authority_scope_exact_key"] == "DESIGN_BLOCKED"
    assert reservation["account_id_alone_proves_operation_ownership"] is False

    binding = value["operation_reservation_binding"]
    assert "at most one" in binding["invariant"]
    assert "CONFLICT" in binding["changed_semantic_request"]
    assert "no LWW" in binding["same_account_different_operations"]
    commit = value["genesis_commit_binding"]
    assert commit["only_COMMITTED_establishes_genuine_account"] is True
    assert commit["PREPARED_grants_account_authority"] is False
    assert commit["logical_commit_closure_invariant"] == "FROZEN"
    assert "authenticated fresh recovery closure" in commit["genuine_at"]
    assert (
        commit["genesis_record_exists_without_verified_commit_closure_is_genuine_account"] is False
    )
    assert commit["partial_genesis_row_authority"] == "NOT SUFFICIENT FOR AUTHORITY"
    assert commit["partial_genesis_PREPARED_BOUND_PREPARED_is_COMMITTED"] is False
    assert commit["physical_genesis_row_is_commit_point"] is False
    assert "PARTIAL/UNPUBLISHED" in commit["crash_after_genesis_record_write"]

    commit_models = value["candidate_commit_models"]
    assert commit_models["selection"] == "DESIGN_BLOCKED"
    assert commit_models["physical_commit_model"] == "NOT_SELECTED / DESIGN_BLOCKED"
    assert commit_models["D_DESIGN_BLOCKED"]["status"] == "SELECTED"
    assert commit_models["physical_storage_atomicity"] == "NOT_FROZEN"
    assert commit_models["logical_authority_atomicity"].startswith("FROZEN")

    abort = value["abort_release_semantics"]
    assert abort["same_operation_may_restart"] is False
    assert abort["caller_may_unilaterally_abort_or_erase_history"] is False
    assert abort["RELEASE_belongs_to"] == "reservation disposition only"
    assert abort["RELEASE_preserves_operation_terminal_ABORTED"] is True
    assert abort["RELEASE_preserves_operation_history"] is True
    assert abort["RELEASE_may_alter_only_reservation_disposition"] is True
    assert abort["RELEASE_changes_operation_identity"] is False
    assert abort["RELEASE_changes_operation_terminal_outcome"] is False
    assert abort["RELEASE_turns_same_operation_into_NEW"] is False
    assert value["reuse_semantics"]["reuse_without_authenticated_release"] == "FORBIDDEN"
    assert value["reuse_semantics"]["authenticated_release_implies_reuse_permission"] is False

    assert len(value["crash_matrix"]) == 7
    assert len(value["retry_matrix"]) == 6
    aborted_retry = next(
        item for item in value["retry_matrix"] if item["case"] == "same operation, ABORTED"
    )
    assert aborted_retry["outcome"].startswith("always return ABORTED")
    reservation_crash = next(
        item for item in value["crash_matrix"] if item["point"] == "after reservation"
    )
    assert "else FAIL_CLOSED" in reservation_crash["restart_outcome"]
    assert "never allocate acct_B" in reservation_crash["restart_outcome"]
    assert value["concurrency"]["per_operation_serialization"] == "REQUIRED"
    assert value["concurrency"]["per_reservation_serialization"] == "REQUIRED"
    assert value["concurrency"]["per_account_id_serialization"] == "REQUIRED"
    assert value["concurrency"]["last_writer_wins"] is False
    assert value["concurrency"]["per_operation_serialization_enforceable_when"] == (
        "genuine operation identity or equivalent frozen recovery binding exists"
    )
    assert value["concurrency"]["before_operation_serialization_enforceable"] == (
        "reservation scope prevents concurrent remint"
    )
    assert value["CAS"]["required"] is True

    assert value["durability"]["current_only_row_sufficient"] is False
    assert value["durability"]["bounded_TTL_safe"] is False
    assert value["authentication"]["public_SHA_sufficient"] is False
    assert value["authentication"]["local_hashes_sufficient"] is False
    assert value["freshness"]["valid_prefix_rollback_must_be_rejected"] is True
    assert "DESIGN_BLOCKED" in value["freshness"]["external_freshness_authority"]
    assert value["rollback"]["COMMITTED_to_PREPARED_valid_prefix"] == "MUST_FAIL_CLOSED"
    assert value["restart"]["order"] == [
        "verify durable storage",
        "verify authentication",
        "verify journal closure",
        "verify freshness",
        "reconstruct operations/reservations",
        "reconcile committed genesis facts",
        "publish resolver only after closure",
    ]
    assert value["restart"]["failure_mode"] == "FAIL_CLOSED"

    assert value["environment_isolation"]["TEST_prepared_to_PRODUCTION_commit"] == "DENY"
    assert value["M011_interaction"]["restored_projection_recreates_authority"] is False
    assert value["M03_interaction"]["reusable_identity_for_account_genesis"] is False
    parity = value["cross_artifact_parity"]
    assert parity["parity"] == "PASS"
    assert parity["different_ids_prove_distinct_operations"] is False
    assert parity["same_operation_may_remint_after_crash"] is False
    assert parity["logical_idempotency_identity"] == "NOT_FROZEN"
    assert set(value["mandatory_redteam"].values()) == {"FAIL"}
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_RESERVATION_STATE_MODEL_DESIGN_BLOCKED"
    )
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert value["preserved_status"]["production_M0.5"] == "NOT_AVAILABLE"

    identity = value["logical_operation_object"]
    assert identity["operation_identity_issuance"] == "NOT_FROZEN"
    assert identity["operation_identity_issuance_timing"] == "NOT_FROZEN"
    assert set(identity["issuance_timing_candidates"]) == {
        "BEFORE_RESERVATION",
        "ATOMIC_WITH_RESERVATION",
        "INSIDE_FIRST_PREPARE",
    }
    assert identity["issuance_timing_candidates"]["INSIDE_FIRST_PREPARE"].startswith(
        "CONDITIONAL_CANDIDATE"
    )
    assert identity["INSIDE_FIRST_PREPARE_prerequisite"] == (
        "AUTHORITY_BOUND_RESERVATION_RECOVERY_IDENTITY + "
        "AUTHENTICATED_CANONICAL_REQUEST_BINDING + "
        "DETERMINISTIC_HANDOFF_TO_OPERATION_IDENTITY"
    )
    assert identity["caller_may_mint_genuine_operation_identity"] is False
    assert identity["preexisting_operation_identity_before_reservation_required"] is False
    assert identity["caller_command_ID_is_recovery_identity"] is False
    assert identity["caller_command_ID_is_handoff_authority"] is False

    recovery = value["recovery_identity"]
    assert recovery["universal_invariant"].startswith("every durable")
    assert set(recovery["must_resolve"]) == {
        "same operation",
        "same reservation",
        "same account_id",
        "same canonical request binding",
    }
    assert recovery["caller_controlled_identity_sufficient"] is False
    assert (
        recovery[
            "logical_operation_identity_and_reservation_recovery_identity_are_distinct_semantic_roles"
        ]
        is True
    )
    assert recovery["logical_operation_identity_equals_reservation_recovery_identity"].startswith(
        "NOT_FROZEN"
    )
    assert recovery["identity_gap_outcome"] == "FAIL_CLOSED"
    assert recovery["identity_gap_may_allocate_new_account_id"] is False
    assert recovery["identity_gap_may_recover_operation_by_caller_assertion"] is False

    models = value["pre_PREPARED_recovery_models"]
    assert models["selection"] == "DESIGN_BLOCKED"
    assert models["C_DESIGN_BLOCKED"]["status"] == "SELECTED"
    assert models["B_RESERVATION_FIRST_WITH_AUTHORITY_BOUND_RECOVERY_IDENTITY"][
        "status"
    ].startswith("CONDITIONAL_CANDIDATE")
    gap = value["critical_pre_PREPARED_identity_gap"]
    assert gap["outcome"] == "FAIL_CLOSED"
    assert gap["recover_O_by_caller_assertion"] is False
    assert gap["allocate_acct_B"] is False


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


def test_cross_artifact_parity() -> None:
    value = load()
    uniqueness = json.loads(
        (DOCS / "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json").read_text(
            encoding="utf-8"
        )
    )
    authority = json.loads(
        (DOCS / "m05_cryptohunter_account_genesis_authority_model.json").read_text(encoding="utf-8")
    )
    parity = value["cross_artifact_parity"]
    assert (
        parity["entity_uniqueness"] == "account_id / " + uniqueness["entity_uniqueness"]["result"]
    )
    assert (
        parity["subject_cardinality"]
        == authority["account_subject_identity"]["subject_to_account_cardinality"]
    )
    assert (
        "successful durable account-genesis commit"
        in authority["account_id_state_machine"]["genuine_at"]
    )
    assert parity["genuine_account_only_after_genuine_genesis_commit"] is True
    assert parity["prior_commit_does_not_select_physical_DB_transaction"] is True
    assert (
        uniqueness["reservation_relationship"]["same_logical_operation_may_remint_after_crash"]
        is False
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["genesis_commit_binding"].update(PREPARED_grants_account_authority=True),
        lambda d: d["reservation_state_machine"].update(missing_lookup="allocate new account_id"),
        lambda d: d["operation_reservation_binding"].update(
            invariant="one operation -> multiple reservations"
        ),
        lambda d: d["rollback"].update(COMMITTED_to_PREPARED_valid_prefix="ACCEPT"),
        lambda d: d["abort_release_semantics"].update(
            caller_may_unilaterally_abort_or_erase_history=True
        ),
        lambda d: d["reuse_semantics"].update(reuse_without_authenticated_release="PERMITTED"),
        lambda d: d["authentication"].update(local_hashes_sufficient=True),
        lambda d: d["genesis_commit_binding"].update(
            crash_after_genesis_record_write="mint another account"
        ),
        lambda d: d["environment_isolation"].update(TEST_prepared_to_PRODUCTION_commit="ALLOW"),
        lambda d: d["M011_interaction"].update(restored_projection_recreates_authority=True),
        lambda d: d["operation_state_machine"].update(
            ABORTED_to_RELEASED_operation_transition="ALLOWED"
        ),
        lambda d: d["abort_release_semantics"].update(
            RELEASE_changes_operation_terminal_outcome=True
        ),
        lambda d: d["genesis_commit_binding"].update(
            genesis_record_exists_without_verified_commit_closure_is_genuine_account=True
        ),
        lambda d: d["genesis_commit_binding"].update(
            partial_genesis_PREPARED_BOUND_PREPARED_is_COMMITTED=True
        ),
        lambda d: d["genesis_commit_binding"].update(physical_genesis_row_is_commit_point=True),
        lambda d: d["logical_operation_object"].update(
            caller_may_mint_genuine_operation_identity=True
        ),
        lambda d: d["logical_operation_object"].update(
            preexisting_operation_identity_before_reservation_required=True
        ),
        lambda d: d["recovery_identity"].update(identity_gap_outcome="recover operation normally"),
        lambda d: d["reservation_state_machine"].update(
            account_id_alone_proves_operation_ownership=True
        ),
        lambda d: d["logical_operation_object"].update(caller_command_ID_is_handoff_authority=True),
        lambda d: d["critical_pre_PREPARED_identity_gap"].update(allocate_acct_B=True),
    ],
)
def test_mandatory_redteam_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)
