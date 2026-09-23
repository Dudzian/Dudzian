"""Wykonywalny kontrakt rozstrzygnięcia topologii AccountGenesis."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.json"
MARKDOWN = DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.md"
OWNER = "CryptoHunterAccountAuthority"


EXPECTED_ROLE_OWNERS = {
    "independent root-proof issuer": "EXTERNAL_INDEPENDENT_ROOT_PROOF_ISSUER / NOT_AVAILABLE",
    "root-proof validator": OWNER,
    "logical operation identity issuer": OWNER,
    "logical operation identity owner": OWNER,
    "account_id mint owner": OWNER,
    "reservation owner": OWNER,
    "reservation recovery identity owner": OWNER,
    "canonical request authority/binding owner": OWNER,
    "genesis semantic decision owner": OWNER,
    "durable commit coordinator": OWNER,
    "ABORT authorizer": OWNER,
    "RELEASE authorizer": OWNER,
    "historical resolver owner": OWNER,
    "security custody owner": "Dedicated AccountGenesis security custody wrapper",
    "freshness owner": "Future AccountGenesis freshness / anti-rollback authority",
}


EXPECTED_ROLE_STATUSES = {
    "independent root-proof issuer": "OWNER_CLASS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE",
    "root-proof validator": "FROZEN",
    "logical operation identity issuer": "FROZEN",
    "logical operation identity owner": "FROZEN",
    "account_id mint owner": "FROZEN",
    "reservation owner": "FROZEN",
    "reservation recovery identity owner": "FROZEN",
    "canonical request authority/binding owner": "FROZEN",
    "genesis semantic decision owner": "FROZEN",
    "durable commit coordinator": "FROZEN",
    "ABORT authorizer": "FROZEN",
    "RELEASE authorizer": "FROZEN",
    "historical resolver owner": "FROZEN",
    "security custody owner": "FROZEN_ROLE / IMPLEMENTATION_GATED",
    "freshness owner": "OWNER_CLASS_FROZEN / IMPLEMENTATION_NOT_AVAILABLE",
}


MANDATORY_REDTEAM_MUTATIONS = {
    "operation_committed_reservation_reserved": (
        (
            "logical_commit_closure",
            "state_consistency",
            "operation_COMMITTED_with_reservation_RESERVED",
        ),
        "PUBLISH",
    ),
    "reservation_consumed_no_genesis": (
        (
            "logical_commit_closure",
            "state_consistency",
            "reservation_CONSUMED_COMMITTED_without_operation_COMMITTED",
        ),
        "PUBLISH",
    ),
    "validator_accepts_no_commit": (
        ("root_proof_validation_role", "root_proof_validation_evidence_implies_COMMITTED"),
        True,
    ),
    "commit_without_proof": (
        (
            "root_proof_validation_role",
            "independent_accepted_root_proof_validation_evidence_required_for_COMMITTED",
        ),
        False,
    ),
    "two_final_committers": (("decision", "final_genesis_decision_owner_count"), 2),
    "self_issue_and_validate": (
        (
            "separation_invariants",
            "genesis_decision_owner_may_self_issue_independent_root_proof",
        ),
        True,
    ),
    "child_authorizes_parent": (
        ("non_circular_root", "account_scoped_operator_identity_allowed"),
        True,
    ),
    "reservation_as_root": (
        ("separation_invariants", "reservation_establishes_genuine_account"),
        True,
    ),
    "mint_as_authorization": (
        (
            "separation_invariants",
            "minting_candidate_account_id_authorizes_genuine_account",
        ),
        True,
    ),
}


def load(name: str | None = None) -> dict:
    path = DOCS / name if name else MACHINE
    return json.loads(path.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis authority topology resolution after binding freeze\n\n"
        "Ten plik jest deterministyczną, kompletną projekcją "
        "`m05_account_genesis_authority_topology_resolution_after_binding_freeze.json`. "
        "JSON jest źródłem prawdy.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_RESOLUTION_AFTER_BINDING_FREEZE"
    )
    provenance = value["provenance"]
    assert provenance["actual_repository_HEAD_inspected"] == value["repository_head_examined"]
    assert provenance["reviewed_SHA_supplied"] == "NOT_SUPPLIED"
    assert provenance["availability"] == "NOT_APPLICABLE"
    assert provenance["reviewed_sha_state"] == "NOT_SUPPLIED"
    assert provenance["classification"] == "UNKNOWN"
    assert provenance["finding_scope"] == "CURRENT_TREE_ONLY"
    assert provenance["formal_advancement_allowed"] is False
    assert provenance["formal_project_advancement"] == "WITHHELD"

    decision = value["decision"]
    assert decision["selected_authority_topology"] == "A_SINGLE_ACCOUNT_GENESIS_COORDINATOR"
    assert decision["topology_result"] == "ACCOUNT_GENESIS_AUTHORITY_TOPOLOGY_FROZEN"
    assert decision["exactly_one_final_genesis_semantic_decision_owner"] == "REQUIRED"
    assert decision["owner_identity"] == OWNER
    assert decision["final_genesis_decision_owner_count"] == 1
    assert decision["missing_external_root_proof_blocks_topology_selection"] is False
    assert decision["missing_external_root_proof_blocks_COMMITTED_execution"] is True
    assert decision["production_implementation"] is False
    assert decision["design_result_current_tree"] == "FROZEN"
    assert decision["formal_project_advancement"] == "WITHHELD"

    roles = {row["role"]: row for row in value["role_matrix"]}
    assert set(roles) == set(EXPECTED_ROLE_OWNERS)
    required_columns = {
        "role",
        "candidate_owner",
        "authority_scope",
        "may_create_trust",
        "may_only_validate_trust",
        "may_mint_identity",
        "may_mutate_durable_authority_state",
        "must_be_independent_from_created_account",
        "status",
        "evidence",
    }
    assert all(set(row) == required_columns for row in roles.values())
    assert {role: row["candidate_owner"] for role, row in roles.items()} == EXPECTED_ROLE_OWNERS
    assert {role: row["status"] for role, row in roles.items()} == EXPECTED_ROLE_STATUSES
    assert (
        roles["independent root-proof issuer"]["must_be_independent_from_created_account"] == "YES"
    )
    assert roles["root-proof validator"]["may_create_trust"] == "NO"

    decision_owner = roles["genesis semantic decision owner"]["candidate_owner"]
    assert roles["freshness owner"]["candidate_owner"] != decision_owner
    assert roles["independent root-proof issuer"]["candidate_owner"] != decision_owner

    separation = value["separation_invariants"]
    false_invariants = [
        "root_proof_issuer_is_automatically_genesis_decision_owner",
        "genesis_decision_owner_may_self_issue_independent_root_proof",
        "technical_ownership_grants_root_proof_issuance",
        "minting_candidate_account_id_authorizes_genuine_account",
        "reservation_establishes_genuine_account",
        "root_validator_may_mint_proof",
        "root_validator_may_create_trust_from_local_account_state",
        "custody_owner_is_domain_authorization_owner",
        "freshness_owner_is_genesis_semantic_decision_owner",
        "valid_HMAC_is_authorization",
        "valid_HMAC_is_freshness",
        "CatalogAdmissionReceiptAuthority_has_AccountGenesis_role",
    ]
    assert all(separation[key] is False for key in false_invariants)
    assert separation["unknown_issuer_environment_or_domain"] == "FAIL_CLOSED"
    assert roles["security custody owner"]["candidate_owner"] != (
        "CatalogAdmissionReceiptAuthority"
    )

    ownership = value["ownership_resolution"]
    assert ownership["logical_operation_identity_issuer"] == OWNER
    assert ownership["caller_may_issue_genuine_logical_operation_identity"] is False
    assert ownership["caller_command_id_is_genuine_logical_operation_identity"] is False
    assert ownership["M0.11_SQLiteStateStore_role"] == ("PROJECTION_CARRIER_ONLY / NOT_AUTHORITY")

    validation = value["root_proof_validation_role"]
    assert (
        roles["root-proof validator"]["candidate_owner"]
        == (validation["host_component"])
        == decision["owner_identity"]
        == OWNER
    )
    assert validation["is_separate_domain_authority"] is False
    assert validation["may_issue_proof"] is False
    assert validation["may_create_independent_root_trust"] is False
    assert validation["may_validate_evidence"] is True
    assert validation["validation_result_is_final_genesis_decision"] is False
    assert validation["root_proof_validation_evidence_implies_COMMITTED"] is False
    assert (
        validation["independent_accepted_root_proof_validation_evidence_required_for_COMMITTED"]
        is True
    )
    assert validation["accepted_proof_without_genesis_COMMITTED"] == "DO_NOT_PUBLISH"

    non_circular = value["non_circular_root"]
    assert non_circular == {
        "created_account_itself_allowed": False,
        "account_scoped_operator_identity_allowed": False,
        "account_scoped_device_installation_allowed": False,
        "account_scoped_workspace_allowed": False,
        "account_scoped_authority_allowed": False,
    }

    representation = value["owner_vs_representation"]
    assert representation["logical_operation_identity_owner"] == "FROZEN"
    assert representation["logical_operation_identity_exact_syntax"] == "NOT_FROZEN"
    assert representation["logical_operation_identity_issuance_timing"] == "NOT_FROZEN"
    assert representation["reservation_owner"] == "FROZEN"
    assert representation["exact_reservation_identity_schema_key"] == "NOT_FROZEN"
    assert representation["caller_selected_genuine_ids"] == "FORBIDDEN"

    closure = value["logical_commit_closure"]
    assert (
        roles["durable commit coordinator"]["candidate_owner"]
        == (closure["coordinator"])
        == decision["owner_identity"]
        == OWNER
    )
    assert roles["account_id mint owner"]["candidate_owner"] == OWNER
    assert roles["reservation owner"]["candidate_owner"] == OWNER
    assert roles["ABORT authorizer"]["candidate_owner"] == OWNER
    assert roles["RELEASE authorizer"]["candidate_owner"] == OWNER
    assert roles["historical resolver owner"]["candidate_owner"] == OWNER
    assert closure["required"] == [
        "exact operation",
        "exact canonical request",
        "exact reservation",
        "exact account_id",
        "accepted independent root-proof validation evidence",
        "COMMITTED genesis",
    ]
    assert closure["physical_persistence_protocol"] == "NOT_SELECTED / DESIGN_BLOCKED"
    assert closure["partial_closure"] == "FAIL_CLOSED / DO_NOT_PUBLISH"
    consistency = closure["state_consistency"]
    assert consistency["operation_state_required_for_genuine_account"] == "COMMITTED"
    assert consistency["reservation_state_required_for_genuine_account"] == "CONSUMED_COMMITTED"
    assert consistency["genuine_account_only_after_complete_logical_closure"] is True
    assert consistency["operation_COMMITTED_with_reservation_RESERVED"] == (
        "FAIL_CLOSED / DO_NOT_PUBLISH"
    )
    assert consistency["operation_COMMITTED_with_reservation_ABORTED_HELD"] == (
        "FAIL_CLOSED / DO_NOT_PUBLISH"
    )
    assert consistency["reservation_CONSUMED_COMMITTED_without_operation_COMMITTED"] == (
        "FAIL_CLOSED / DO_NOT_PUBLISH"
    )
    assert consistency["accepted_root_proof_without_operation_COMMITTED"] == "DO_NOT_PUBLISH"

    candidates = value["candidate_topologies"]
    assert set(candidates) == {
        "A_SINGLE_ACCOUNT_GENESIS_COORDINATOR",
        "B_SEPARATE_OPERATION_AND_ACCOUNT_AUTHORITY",
        "C_SEPARATE_RESERVATION_AUTHORITY",
        "D_EXTERNAL_PROVISIONING_OWNS_OPERATION_OR_ACCOUNT_ID",
        "E_SEPARATE_ROOT_PROOF_VALIDATOR_AUTHORITY_ALSO_OWNS_FINAL_DECISION",
        "F_DESIGN_BLOCKED",
    }
    assert candidates["A_SINGLE_ACCOUNT_GENESIS_COORDINATOR"]["status"] == "SELECTED"
    separate_validator = candidates[
        "E_SEPARATE_ROOT_PROOF_VALIDATOR_AUTHORITY_ALSO_OWNS_FINAL_DECISION"
    ]
    assert separate_validator["status"] == "REJECTED"
    assert "separate proof-validation authority" in separate_validator["note"]
    assert candidates["F_DESIGN_BLOCKED"]["status"] == "NOT_SELECTED"
    assert all(candidates[key]["advantages"] for key in candidates)
    assert all(candidates[key]["required_invariants"] for key in candidates)
    assert all(candidates[key]["failure_modes"] for key in candidates)

    frozen = value["frozen_contracts"]
    assert frozen["account_id != operation identity"] is True
    assert frozen["proof_id != operation identity"] is True
    assert frozen["caller command_id != operation identity"] is True
    assert frozen["one operation -> at most one reservation/account candidate"] is True
    assert frozen["same operation cannot remint acct_B"] is True
    assert frozen["ABORTED terminal"] is True
    assert frozen["RELEASE reservation-only"] is True
    assert frozen["PREPARED != genuine account"] is True
    assert frozen["COMMITTED requires authenticated fresh logical closure"] is True
    assert frozen["independent first-account root required"] is True
    assert frozen["subject/account cardinality"] == "NOT_FROZEN"
    assert frozen["operation identity issuance timing"] == "NOT_FROZEN"

    redteam = value["mandatory_redteam"]
    assert len(redteam) == 9
    assert {case["id"] for case in redteam} == set(MANDATORY_REDTEAM_MUTATIONS)
    assert all(case["expected"] == "FAIL_CLOSED / DO_NOT_PUBLISH" for case in redteam)
    assert value["implementation_gates"] == {
        "CryptoHunterAccountAuthority": "NO",
        "WorkspaceAuthority": "NOT_AVAILABLE",
        "FullFillAuthority": "NOT_AVAILABLE",
        "M0.8": "BLOCKED / canonical current status",
        "production M0.5": "NOT_AVAILABLE",
    }


def validate_cross_artifact_parity(value: dict) -> None:
    """Wyprowadza zgodność z wartości źródłowych, nie z ich polami PASS."""
    uniqueness = load("m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json")
    reservation = load("m05_cryptohunter_account_genesis_reservation_state_model.json")
    substrate = load("m05_account_genesis_security_substrate_contract.json")
    proof = load("m05_account_genesis_root_proof_admission_binding_contract.json")
    operation = load("m05_account_genesis_operation_identity_request_binding_contract.json")
    old_topology = load("m05_account_genesis_authority_topology_and_ownership.json")
    root = load("m05_cryptohunter_account_root_of_trust_reconciliation.json")
    authority = load("m05_cryptohunter_account_genesis_authority_model.json")

    assert (
        uniqueness["entity_uniqueness"]["account_id_is_logical_operation_idempotency_identity"]
        is False
    )
    assert reservation["recovery_identity"]["identity_gap_may_allocate_new_account_id"] is False
    assert reservation["operation_state_machine"]["ABORTED"].startswith("immutable terminal")
    assert (
        reservation["abort_release_semantics"]["RELEASE_belongs_to"]
        == "reservation disposition only"
    )
    upstream_closure = reservation["genesis_commit_binding"]
    consistency = value["logical_commit_closure"]["state_consistency"]
    assert (
        consistency["operation_state_required_for_genuine_account"]
        in upstream_closure["genuine_at"]
    )
    assert (
        consistency["reservation_state_required_for_genuine_account"]
        in upstream_closure["genuine_at"]
    )
    assert upstream_closure["logical_commit_closure_invariant"] == "FROZEN"
    assert upstream_closure["only_COMMITTED_establishes_genuine_account"] is True
    assert consistency["genuine_account_only_after_complete_logical_closure"] is True
    assert substrate["frozen_inputs"]["valid_MAC_is_freshness"] is False
    assert substrate["frozen_inputs"]["catalog_authority_as_account_authority"] == "FORBIDDEN"
    assert proof["operation_binding"]["proof_id_is_operation_id"] is False
    assert (
        "root-proof issuance != root-proof validation" in proof["authority_boundary"]["separations"]
    )
    assert operation["issuance_timing"]["status"] == "NOT_FROZEN"
    assert old_topology["selected_or_blocked_topology"]["one_final_owner_requirement"] == "FROZEN"
    supersedes = value["ownership_resolution"]["supersedes_prior_owner_resolution"]
    assert supersedes == {
        "logical operation identity owner": True,
        "account_id mint owner": True,
        "reservation owner": True,
        "root-proof validator owner": True,
        "final genesis decision owner": True,
    }
    old_owners = old_topology["ownership_matrix"]
    assert old_owners["logical_operation_identity"]["owner"] == "DESIGN_BLOCKED"
    assert old_owners["account_id_mint"]["owner"] == "NOT_FROZEN"
    assert old_owners["reservation"]["owner"] == "DESIGN_BLOCKED"
    assert old_owners["root_proof_validation"]["owner"] == "DESIGN_BLOCKED"
    assert old_owners["genesis_semantic_decision"]["owner"] == "NOT_FOUND"
    upstream_non_circular = old_topology["root_proof_boundary"]["non_circular_root"]
    for key, expected in value["non_circular_root"].items():
        assert upstream_non_circular[key] is expected is False
    assert root["production_provisioning_authority_availability"] == "NOT_FOUND / NOT_AVAILABLE"
    assert authority["frozen_input_contracts"]["caller_selected_genuine_account_id"] == "FORBIDDEN"
    validate(value)


def test_contract_and_projection() -> None:
    value = load()
    validate(value)
    validate_cross_artifact_parity(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("decision", "owner_identity"), "SecondAuthority"),
        (("decision", "missing_external_root_proof_blocks_topology_selection"), True),
        (("decision", "missing_external_root_proof_blocks_COMMITTED_execution"), False),
        (
            (
                "separation_invariants",
                "genesis_decision_owner_may_self_issue_independent_root_proof",
            ),
            True,
        ),
        (
            ("separation_invariants", "minting_candidate_account_id_authorizes_genuine_account"),
            True,
        ),
        (("separation_invariants", "reservation_establishes_genuine_account"), True),
        (("separation_invariants", "valid_HMAC_is_authorization"), True),
        (("logical_commit_closure", "partial_closure"), "PUBLISH"),
        (
            (
                "logical_commit_closure",
                "state_consistency",
                "reservation_state_required_for_genuine_account",
            ),
            "RESERVED",
        ),
        (("root_proof_validation_role", "accepted_proof_without_genesis_COMMITTED"), "PUBLISH"),
        (("root_proof_validation_role", "validation_result_is_final_genesis_decision"), True),
        (("root_proof_validation_role", "is_separate_domain_authority"), True),
        (("provenance", "classification"), "NO_REVIEWED_SHA_TO_COMPARE"),
        (("provenance", "formal_advancement_allowed"), True),
        (("implementation_gates", "CryptoHunterAccountAuthority"), "YES"),
    ],
)
def test_security_mutations_fail(path: tuple[str, ...], replacement: object) -> None:
    mutated = deepcopy(load())
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("attack_id", MANDATORY_REDTEAM_MUTATIONS)
def test_every_mandatory_redteam_mutates_security_semantics(attack_id: str) -> None:
    mutated = deepcopy(load())
    path, replacement = MANDATORY_REDTEAM_MUTATIONS[attack_id]
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    ("role", "candidate_owner"),
    [
        ("logical operation identity issuer", "CALLER"),
        ("logical operation identity issuer", "EXTERNAL_UNTRUSTED_COMPONENT"),
        ("security custody owner", "CatalogAdmissionReceiptAuthority"),
        ("freshness owner", OWNER),
        ("root-proof validator", "SeparateValidatorAuthority"),
        ("durable commit coordinator", "SecondCoordinator"),
        ("historical resolver owner", "SQLiteStateStore"),
    ],
)
def test_every_frozen_role_owner_mutation_fails(role: str, candidate_owner: str) -> None:
    mutated = deepcopy(load())
    row = next(item for item in mutated["role_matrix"] if item["role"] == role)
    row["candidate_owner"] = candidate_owner
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize(
    ("role", "status"),
    [
        ("logical operation identity issuer", "NOT_FROZEN"),
        ("reservation owner", "DESIGN_BLOCKED"),
        ("freshness owner", "FROZEN"),
    ],
)
def test_every_frozen_role_status_mutation_fails(role: str, status: str) -> None:
    mutated = deepcopy(load())
    row = next(item for item in mutated["role_matrix"] if item["role"] == role)
    row["status"] = status
    with pytest.raises(AssertionError):
        validate(mutated)
