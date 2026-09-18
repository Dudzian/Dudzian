"""Executable contract for the M0.5 account-genesis authority topology design."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_authority_topology_and_ownership.json"
MARKDOWN = DOCS / "m05_account_genesis_authority_topology_and_ownership.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 Account genesis authority topology and ownership\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_authority_topology_and_ownership.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_AND_OWNERSHIP_DESIGN"
    )
    assert value["reviewed_head_supplied"] == (
        "18107c75b75ad410c9ea481b2df4947566921a3e"
    )
    provenance = value["provenance"]
    assert provenance["source"] == "GIT"
    assert provenance["reviewed_head_available_locally"] is False
    assert provenance["classification"] == "UNKNOWN"
    assert provenance["formal_advancement_allowed"] is False
    assert value["repository_head_examined"] != value["reviewed_head_supplied"]

    frozen = value["frozen_inputs"]
    assert frozen["security_substrate"] == (
        "ACCOUNT_GENESIS_SECURITY_SUBSTRATE_CONTRACT_FROZEN"
    )
    assert frozen["root_of_trust"] == "DESIGN_BLOCKED"
    assert frozen["account_id_is_operation_identity"] is False
    assert frozen["reservation_is_authority"] is False
    assert frozen["PREPARED_is_genuine"] is False
    assert frozen["ABORTED"] == "terminal"
    assert frozen["RELEASE"] == "reservation-only"
    assert frozen["subject_cardinality"] == "NOT_FROZEN"
    assert frozen["valid_MAC_is_authorization"] is False
    assert frozen["valid_MAC_is_freshness"] is False

    candidates = value["candidate_topologies"]
    assert set(candidates) == {
        "A_UNIFIED_CRYPTOHUNTER_ACCOUNT_AUTHORITY",
        "B_SEPARATE_RESERVATION_AUTHORITY",
        "C_EXTERNAL_ISSUER_MINTS_OR_RESERVES_ACCOUNT_ID",
        "D_ATOMIC_ISSUER_ACCOUNT_AUTHORITY_PROTOCOL",
        "E_OTHER_CANONICALLY_SUPPORTED_MODEL",
        "F_DESIGN_BLOCKED",
    }
    assert candidates["F_DESIGN_BLOCKED"] == "SELECTED"
    unified = value["topology_analysis"][
        "A_UNIFIED_CRYPTOHUNTER_ACCOUNT_AUTHORITY"
    ]
    assert unified["candidate_id_technical_ownership"] == (
        "AccountAuthority may mint/reserve candidate IDs"
    )
    assert unified["technical_ownership_is_self_authorization_permission"] is False
    assert unified["genuine_genesis_precondition"] == "independent accepted root proof"
    selected = value["selected_or_blocked_topology"]
    assert selected["selection"] == "F_DESIGN_BLOCKED"
    assert selected["one_final_owner_requirement"] == "FROZEN"
    assert selected["one_final_owner_identity"] == "NOT_FOUND"

    matrix = value["ownership_matrix"]
    assert set(matrix) == {
        "logical_operation_identity",
        "account_id_mint",
        "reservation",
        "reservation_recovery",
        "root_proof_issuance",
        "root_proof_validation",
        "genesis_semantic_decision",
        "durable_commit_coordination",
        "ABORT",
        "RELEASE",
        "historical_resolver",
        "custody",
        "freshness",
    }
    required = {"owner", "status", "evidence", "authority_scope", "may_mint_trust", "notes"}
    assert all(set(row) == required for row in matrix.values())
    assert matrix["logical_operation_identity"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["account_id_mint"]["owner"] == "NOT_FROZEN"
    assert matrix["reservation"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["reservation_recovery"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["root_proof_validation"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["genesis_semantic_decision"]["owner"] == "NOT_FOUND"
    assert matrix["durable_commit_coordination"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["ABORT"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["RELEASE"]["owner"] == "DESIGN_BLOCKED"
    assert matrix["historical_resolver"]["owner"] == "DESIGN_BLOCKED"

    root = value["root_proof_boundary"]
    assert root["candidate"] == "M03_EXTERNAL_PRODUCT_PROVISIONING_MEMBERSHIP"
    assert root["issuance_is_validation"] is False
    assert root["issuance_implies_account_id_mint"] is False
    assert root["proof_id_is_operation_id"] is False
    assert set(root["consumption_semantics"].values()) >= {"NOT_FROZEN"}
    assert "NOT_FROZEN" in root["replay"]
    self_authorization = root["first_account_self_authorization"]
    assert self_authorization["independent_accepted_root_proof_required"] is True
    assert self_authorization["locally_generated_evidence_only"] == "FORBIDDEN"
    assert (
        self_authorization["technical_ownership_is_self_authorization_permission"]
        is False
    )
    non_circular = root["non_circular_root"]
    assert non_circular["target_account"] == "acct_A"
    assert non_circular["created_account_itself_allowed"] is False
    assert non_circular["account_scoped_operator_identity_allowed"] is False
    assert non_circular["account_scoped_device_installation_allowed"] is False
    assert non_circular["account_scoped_workspace_allowed"] is False
    assert non_circular["account_scoped_authority_allowed"] is False
    assert (
        non_circular[
            "legitimacy_may_depend_on_entity_requiring_target_account_to_exist"
        ]
        is False
    )
    assert value["account_id_mint_ownership"]["ProvisioningBoundary_mints"] == (
        "NOT_PROVEN"
    )
    assert value["account_id_mint_ownership"]["caller_arbitrary_id"] == "FORBIDDEN"
    assert value["reservation_recovery_ownership"]["identity_gap"] == "FAIL_CLOSED"
    assert value["reservation_recovery_ownership"]["identity_gap_allocates_acct_B"] is False

    genesis = value["genesis_decision_ownership"]
    assert genesis["owner"] == "NOT_FOUND"
    assert genesis["exactly_one_required"] is True
    assert genesis["dual_independent_commit"] == "FORBIDDEN"
    assert value["commit_coordination"]["partial_split_brain_publish"] == "FORBIDDEN"
    abort_release = value["abort_release_ownership"]
    assert abort_release["caller_request_is_authorization"] is False
    assert abort_release["ABORTED_terminal"] is True
    assert abort_release["RELEASE_reservation_only"] is True
    assert abort_release["RELEASE_changes_terminal_outcome"] is False
    assert abort_release["RELEASE_allows_reuse"] is False

    substrate = value["security_substrate_boundary"]
    assert substrate["owners_are_semantically_identical"] is False
    assert substrate["security_owner_implies_AccountAuthority"] is False
    assert len(substrate["does_not_decide"]) == 4
    assert value["M03_boundary"]["accepted_first_device_membership_equals_genesis_authority"] is False
    assert value["M03_boundary"]["FirstRunBootstrapAuthority"] == (
        "INITIAL_SECURITY_ESTABLISHMENT_ONLY"
    )
    assert value["M03_boundary"]["FirstRunBootstrapAuthority_is_AccountAuthority"] is False
    assert value["M011_boundary"]["may_mint_or_recreate_authority"] is False

    assert value["concurrency"]["last_writer_wins"] is False
    assert "at most one COMMITTED" in value["concurrency"]["same_account"]
    atomicity = value["cross_authority_atomicity"]
    assert atomicity["status"] == "DESIGN_BLOCKED"
    assert atomicity["reservation_CONSUMED_account_PREPARED"].startswith("FAIL_CLOSED")
    assert atomicity["account_COMMITTED_reservation_RESERVED"].startswith("FAIL_CLOSED")
    assert atomicity["independent_commits"] == "FORBIDDEN"

    parity = value["cross_artifact_parity"]
    assert parity["parity"] == "PASS"
    assert parity["account_id_not_operation_identity"] is True
    assert parity["reservation_not_authority"] is True
    assert parity["PREPARED_not_genuine"] is True
    assert parity["ABORTED_terminal"] is True
    assert parity["RELEASE_reservation_only"] is True
    assert parity["root_of_trust"] == "DESIGN_BLOCKED"
    assert parity["security_substrate_authorizes_genesis"] is False
    redteam = value["mandatory_redteam"]
    assert set(redteam.values()) == {"FAIL"}
    assert redteam["AccountAuthority_self_authorizes_without_independent_root"] == (
        "FAIL"
    )
    assert self_authorization["independent_accepted_root_proof_required"] is True
    assert redteam["account_scoped_identity_as_pre_account_root"] == "FAIL"
    assert all(
        non_circular[field] is False
        for field in (
            "created_account_itself_allowed",
            "account_scoped_operator_identity_allowed",
            "account_scoped_device_installation_allowed",
            "account_scoped_workspace_allowed",
            "account_scoped_authority_allowed",
        )
    )
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_DESIGN_BLOCKED"
    )
    assert value["result"]["root_of_trust_solved"] is False
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert value["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        (("root_proof_boundary", "issuance_implies_account_id_mint"), True),
        (("frozen_inputs", "reservation_is_authority"), True),
        (("security_substrate_boundary", "security_owner_implies_AccountAuthority"), True),
        (("M03_boundary", "FirstRunBootstrapAuthority_is_AccountAuthority"), True),
        (("genesis_decision_ownership", "dual_independent_commit"), "ALLOWED"),
        (("commit_coordination", "partial_split_brain_publish"), "ALLOWED"),
        (("account_id_mint_ownership", "caller_arbitrary_id"), "ALLOWED"),
        (("M03_boundary", "accepted_first_device_membership_equals_genesis_authority"), True),
        (("reservation_recovery_ownership", "identity_gap"), "ALLOCATE_ACCT_B"),
        (
            (
                "root_proof_boundary",
                "first_account_self_authorization",
                "independent_accepted_root_proof_required",
            ),
            False,
        ),
        (
            (
                "root_proof_boundary",
                "first_account_self_authorization",
                "locally_generated_evidence_only",
            ),
            "ALLOWED",
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "created_account_itself_allowed",
            ),
            True,
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "account_scoped_operator_identity_allowed",
            ),
            True,
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "account_scoped_device_installation_allowed",
            ),
            True,
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "account_scoped_workspace_allowed",
            ),
            True,
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "account_scoped_authority_allowed",
            ),
            True,
        ),
        (
            (
                "root_proof_boundary",
                "non_circular_root",
                "legitimacy_may_depend_on_entity_requiring_target_account_to_exist",
            ),
            True,
        ),
    ],
)
def test_mandatory_redteam_mutations_fail(
    path: tuple[str, ...], unsafe_value: object
) -> None:
    mutated = deepcopy(load())
    target = mutated
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = unsafe_value
    with pytest.raises(AssertionError):
        validate(mutated)


def test_cross_artifact_status_parity() -> None:
    names = [
        "m05_account_genesis_security_substrate_contract.json",
        "m05_cryptohunter_account_genesis_reservation_state_model.json",
        "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json",
        "m05_cryptohunter_account_genesis_authority_model.json",
        "m05_cryptohunter_account_root_of_trust_reconciliation.json",
    ]
    artifacts = {
        name: json.loads((DOCS / name).read_text(encoding="utf-8")) for name in names
    }
    contract = load()
    assert artifacts[names[0]]["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_SECURITY_SUBSTRATE_CONTRACT_FROZEN"
    )
    assert artifacts[names[1]]["preserved_status"][
        "selected_logical_idempotency_identity"
    ] == "NOT_FROZEN"
    assert artifacts[names[2]]["M03_interaction"][
        "reusable_identity_for_account_genesis"
    ] is False
    assert artifacts[names[3]]["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_MODEL_DESIGN_BLOCKED"
    )
    assert artifacts[names[4]]["account_id_mint_owner"]["existing_owner"] == "NOT_FOUND"
    assert contract["result"]["root_of_trust_solved"] is False
    assert all(allowed is False for allowed in contract["implementation_allowed"].values())
