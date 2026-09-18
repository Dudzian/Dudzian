"""Executable contract checks for M0.5 account-genesis uniqueness/idempotency."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    return (
        "# M0.5 CryptoHunterAccount genesis uniqueness and idempotency design\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
    )


def candidates(value: dict) -> dict[str, dict]:
    return {item["candidate"]: item for item in value["idempotency_candidates"]}


def validate(value: dict) -> None:
    assert value["repository_head_examined"] == "1aa8fb905b378755bb316f47a81aee5c95876d3c"
    assert value["reviewed_head_supplied"] == "6b0a196f9c5c509ae6ab58550ca43939a1c1a727"
    provenance = value["provenance"]
    assert provenance["reviewed_head_available_locally"] is False
    assert provenance["relationship_to_reviewed_sha"] == "UNKNOWN"
    assert provenance["classification"] == "UNKNOWN"
    assert provenance["current_tree_evidence"] is True
    assert provenance["finding_scope"] == "CURRENT_TREE_ONLY"
    assert provenance["formal_advancement_allowed"] is False
    assert provenance["current_local_tree_inspected"] == "YES"
    assert provenance["path_content_comparison_evidence"] == "NOT_PERFORMED"
    assert provenance["current_tree_findings"] == "AVAILABLE FOR DISCOVERY ONLY"
    assert provenance["formal_advancement_from_reviewed_sha"] == "WITHHELD"
    if (
        provenance["reviewed_head_available_locally"] is False
        and provenance["relationship_to_reviewed_sha"] == "UNKNOWN"
    ):
        assert value["repository_head_examined"] != value["reviewed_head_supplied"]
    entity = value["entity_uniqueness"]
    assert entity["result"] == "ENTITY_UNIQUENESS_KEY_FROZEN"
    assert entity["key"] == "account_id"
    assert entity["account_id_alone_sufficient_to_prevent_duplicate_entity_genesis"] is True
    assert entity["account_id_is_logical_operation_idempotency_identity"] is False
    assert "at most one" in entity["invariant"]

    business = value["business_uniqueness_boundary"]
    assert business["subject_to_account_cardinality"] == "NOT_FROZEN"
    assert business["one_subject_one_account"] == "NOT_PROVEN"
    assert business["multiple_accounts_per_subject_allowed"] == "NOT_PROVEN"
    assert business["missing_subject_identity_alone_makes_genesis_impossible"] is False
    assert business["different_account_ids_automatically_duplicate_for_same_subject"] is False
    assert business["idempotency_may_enforce_business_uniqueness"] is False

    matrix = candidates(value)
    assert {
        "ACCOUNT_ID_RESERVATION_IDENTITY",
        "EXTERNAL_PROVISIONING_DECISION_ID",
        "AUTHORITY_ISSUED_GENESIS_REQUEST_ID",
        "ROOT_PROOF_DECISION_ID",
        "ATOMIC_ISSUER_DECISION_ID",
        "EXTERNAL_SUBJECT_IDENTITY",
        "CALLER_COMMAND_ID",
        "DESIGN_BLOCKED",
    } == matrix.keys()
    dimensions = {
        "authority_owner",
        "caller_controllability",
        "stability_across_retries",
        "stability_across_crash_restart",
        "uniqueness_scope",
        "durability",
        "authentication",
        "retry_returns_same_account",
        "two_legitimate_account_creations_can_use_different_ids",
    }
    assert all(dimensions <= item.keys() for item in matrix.values())
    assert matrix["CALLER_COMMAND_ID"]["status"] == "INSUFFICIENT_BY_DEFAULT"
    assert matrix["DESIGN_BLOCKED"]["status"] == "SELECTED"
    assert value["selected_or_blocked_idempotency_model"]["selection"] == "DESIGN_BLOCKED"
    assert value["selected_or_blocked_idempotency_model"]["authority_owner"] == "NOT_FOUND"

    scope = value["serialization_scope"]
    assert scope["status"] == "PARTIALLY_FROZEN"
    assert "per account_id" in scope["minimum"]
    assert scope["distinct_account_operation_relationship"] == "NOT_FROZEN"
    assert scope["cross_account_logical_operation_serialization"].startswith(
        "DESIGN_BLOCKED"
    )
    assert scope["last_writer_wins"] is False
    assert value["same_account_concurrency"]["same_account_id_two_genesis"] == "FORBIDDEN"
    relation = value["logical_operation_relationship"]
    assert relation["current_default_without_authority_bound_provenance"] == "UNKNOWN"
    assert relation["different_account_ids_imply_distinct_operations"] is False
    assert relation["different_idempotency_ids_imply_distinct_operations"] is False
    assert relation["caller_command_ids_prove_distinct_operations"] is False
    assert relation["distinct_account_ids_may_represent_independent_legitimate_genesis"].startswith(
        "CONDITIONAL"
    )
    assert value["distinct_account_concurrency"]["global_singleton_serialization"] is False
    assert value["distinct_account_concurrency"][
        "same_external_subject_alone_causes_duplicate_or_conflict"
    ] is False

    retry = value["retry_semantics"]
    assert "exact replay" in retry["same_identity_same_semantic_request"]
    assert "CONFLICT" in retry["same_identity_different_semantic_request"]
    assert value["reservation_relationship"]["reservation_is_account_authority"] is False
    assert value["reservation_relationship"]["missing_reservation_lookup_allows_new_id"] is False
    assert value["reservation_relationship"]["same_logical_operation_may_remint_after_crash"] is False
    assert value["crash_before_commit"]["silent_remint"] == "FORBIDDEN"
    assert value["crash_before_commit"]["different_account_id_alone_proves_new_operation"] is False
    assert value["critical_retry_split_scenario"]["acct_B_automatically_independent_new_genesis"] is False
    assert value["crash_after_commit"]["new_account"] == "FORBIDDEN"
    assert value["authentication"]["public_SHA_sufficient"] is False
    assert value["rollback"]["valid_prefix_A_B_after_A_B_C"] == "MUST_FAIL_CLOSED"
    assert value["environment_isolation"]["TEST_to_PRODUCTION_account_genesis"] == "DENY"

    m03 = value["M03_interaction"]
    assert m03["existing_exact_decision_id"] == "NOT_FOUND"
    assert m03["membership_id_field"] == "NOT_FOUND"
    assert m03["reusable_identity_for_account_genesis"] is False
    assert m03["ownership_expanded"] is False
    assert value["cross_artifact_parity"]["parity"] == "PASS"
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_UNIQUENESS_IDEMPOTENCY_INSUFFICIENT_SEMANTICS"
    )
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert set(value["mandatory_redteam_mutations"].values()) == {"FAIL"}


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


def test_cross_artifact_parity() -> None:
    value = load()
    genesis = json.loads(
        (DOCS / "m05_cryptohunter_account_genesis_authority_model.json").read_text()
    )
    subject = json.loads(
        (DOCS / "m05_cryptohunter_account_genesis_subject_identity_discovery.json").read_text()
    )
    parity = value["cross_artifact_parity"]
    assert parity["subject_cardinality"] == genesis["account_subject_identity"][
        "subject_to_account_cardinality"
    ]
    assert parity["subject_cardinality"] == subject["subject_account_cardinality"][
        "subject_to_account_cardinality"
    ]
    assert parity["missing_subject_identity_alone_makes_genesis_impossible"] is False
    assert parity["caller_selected_genuine_account_id"] == "FORBIDDEN"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["entity_uniqueness"].update(
            invariant="one genuine account_id -> two immutable account genesis facts"
        ),
        lambda d: d["retry_semantics"].update(
            same_identity_different_semantic_request="create new account"
        ),
        lambda d: d["crash_before_commit"].update(silent_remint="ALLOWED"),
        lambda d: d["reservation_relationship"].update(reservation_is_account_authority=True),
        lambda d: candidates(d)["CALLER_COMMAND_ID"].update(status="SELECTED"),
        lambda d: d["distinct_account_concurrency"].update(
            global_singleton_serialization=True
        ),
        lambda d: d["business_uniqueness_boundary"].update(
            subject_to_account_cardinality="ONE"
        ),
        lambda d: d["distinct_account_concurrency"].update(
            same_external_subject_alone_causes_duplicate_or_conflict=True
        ),
        lambda d: d["authentication"].update(public_SHA_sufficient=True),
        lambda d: d["logical_operation_relationship"].update(
            different_account_ids_imply_distinct_operations=True
        ),
        lambda d: d["logical_operation_relationship"].update(
            different_idempotency_ids_imply_distinct_operations=True
        ),
        lambda d: d["critical_retry_split_scenario"].update(
            acct_B_automatically_independent_new_genesis=True
        ),
        lambda d: d["logical_operation_relationship"].update(
            caller_command_ids_prove_distinct_operations=True
        ),
        lambda d: d.update(repository_head_examined=d["reviewed_head_supplied"]),
        lambda d: d["provenance"].update(classification="EXACT_COMMIT"),
        lambda d: d["provenance"].update(formal_advancement_allowed=True),
        lambda d: d["environment_isolation"].update(
            TEST_to_PRODUCTION_account_genesis="ALLOW"
        ),
    ],
)
def test_mandatory_redteam_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)
