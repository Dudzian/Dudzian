"""Executable checks for the M0.5 AccountGenesis root-proof binding design."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import inspect
import json
from pathlib import Path

import pytest

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    FirstRunBootstrapAuthority,
    FirstRunBootstrapClaim,
    ProvisioningMembershipBinding,
)


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_root_proof_admission_binding_contract.md"
TOPOLOGY = DOCS / "m05_account_genesis_authority_topology_and_ownership.json"
RECONCILIATION = DOCS / "m05_cryptohunter_account_root_of_trust_reconciliation.json"
GENESIS_MODEL = DOCS / "m05_cryptohunter_account_genesis_authority_model.json"
SUBJECT = DOCS / "m05_cryptohunter_account_genesis_subject_identity_discovery.json"
IDEMPOTENCY = DOCS / "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 Account genesis root-proof admission binding contract\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_root_proof_admission_binding_contract.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_cross_artifact_parity(
    contract: dict,
    topology: dict,
    reconciliation: dict,
    genesis_model: dict,
    subject: dict,
    idempotency: dict,
) -> None:
    """Derive parity from the actual upstream artifacts, not self-reporting."""
    root = topology["root_proof_boundary"]
    assert root["first_account_self_authorization"][
        "independent_accepted_root_proof_required"
    ] is True
    assert root["first_account_self_authorization"][
        "locally_generated_evidence_only"
    ] == "FORBIDDEN"
    assert root["non_circular_root"] == contract["non_circular_root"]
    assert root["proof_id_is_operation_id"] is False
    assert root["proof_id_is_operation_id"] == contract["operation_binding"][
        "proof_id_is_operation_id"
    ]
    assert root["issuance_implies_account_id_mint"] is False
    assert topology["account_id_mint_ownership"]["ProvisioningBoundary_mints"] == (
        "NOT_PROVEN"
    )
    assert reconciliation["account_id_mint_owner"]["external_boundary_ownership"] == (
        "NOT_PROVEN"
    )
    assert reconciliation["account_id_mint_owner"]["supplies_is_not_mints"] is True
    assert reconciliation["root_candidate_evaluation"]["result"] == (
        "VIABLE_ONLY_AFTER_ADDITIONAL_AUTHORITY"
    )
    assert genesis_model["account_subject_identity"]["subject_to_account_cardinality"] == (
        "NOT_FROZEN"
    )
    assert subject["result"]["subject_account_cardinality"] == "NOT_FROZEN"
    assert idempotency["cross_artifact_parity"]["subject_cardinality"] == "NOT_FROZEN"
    assert root["consumption_semantics"]["single_use"] == "NOT_FROZEN"
    assert root["consumption_semantics"]["multi_use"] == "NOT_FROZEN"
    assert contract["cross_artifact_parity"]["same_proof_replay_semantics"] == (
        "NOT_FROZEN"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_ROOT_PROOF_ADMISSION_BINDING_CONTRACT"
    )
    assert value["reviewed_head_supplied"] == (
        "74355d07f6fde26e26126c2130422ea4a71917b8"
    )
    assert value["repository_head_examined"] != value["reviewed_head_supplied"]
    assert value["provenance"]["source"] == "GIT"
    assert value["provenance"]["reviewed_head_available_locally"] is False
    assert value["provenance"]["formal_advancement_allowed"] is False

    frozen = value["frozen_inputs"]
    assert frozen["independent_accepted_first_account_root"] == "REQUIRED"
    assert frozen["self_authorization_using_local_evidence"] == "FORBIDDEN"
    assert frozen["non_circular_root"] == "FROZEN"
    assert frozen["proof_id_is_operation_id"] is False
    assert frozen["issuance_is_validation"] is False
    assert frozen["issuance_implies_account_id_mint"] is False
    assert frozen["proof_acceptance_implies_reservation"] is False
    assert frozen["proof_acceptance_implies_COMMITTED"] is False

    models = value["candidate_models"]
    assert set(models) == {
        "A_EXISTING_ACCEPTED_PROVISIONING_MEMBERSHIP_AS_ROOT_PROOF",
        "B_ACCOUNT_GENESIS_SPECIFIC_ROOT_PROOF_DERIVED_FROM_EXTERNAL_MEMBERSHIP",
        "C_EXTERNAL_ISSUER_ACCOUNT_GENESIS_PROOF",
        "D_VALIDATION_RECEIPT_MINTED_BY_FUTURE_ROOT_PROOF_VALIDATOR",
        "E_ATOMIC_ISSUER_ACCOUNT_PROTOCOL_PROOF",
        "F_DESIGN_BLOCKED",
    }
    assert models["F_DESIGN_BLOCKED"]["status"] == "SELECTED"
    assert value["selected_or_blocked_model"]["selection"] == "F_DESIGN_BLOCKED"

    account = value["account_binding"]
    assert account["options"]["D_NOT_FROZEN"] == "SELECTED"
    assert account["proof_accepted_is_account_id_minted"] is False
    assert account["account_id_minted_is_proof_accepted"] is False
    assert account["both_imply_account_genuine"] is False
    account_security = value["account_binding_security"]
    assert account_security["content_binding"].startswith("AVAILABLE:")
    assert account_security["issuer_authenticity"] == "NOT_AVAILABLE"
    assert account_security["end_to_end_genuine_root_proof"] == "NOT_AVAILABLE"
    assert account_security["exact_bound_account_mismatch"] == "DENY / FAIL_CLOSED"
    assert account_security["proof_bound_account_may_be_rebound"] is False
    assert account_security["content_bound_account_id_must_match_candidate"] is True
    operation = value["operation_binding"]
    assert operation["proof_id_is_operation_id"] is False
    assert operation["proof_id_is_operation_id"] == frozen["proof_id_is_operation_id"]
    assert operation["current_candidate_binds_logical_operation"] is False
    assert operation["current_candidate_binds_canonical_genesis_request"] is False

    authentication = value["authentication"]
    assert authentication["public_SHA_is_authority"] is False
    assert authentication["external_signature"] == "NOT_FOUND"
    assert authentication["external_MAC"] == "NOT_FOUND"
    assert authentication["unknown_issuer"] == "DENY / FAIL_CLOSED"
    assert value["freshness"]["authentication_is_freshness"] is False
    assert value["revocation"]["valid_T1_revoked_or_stale_T2"] == "NOT_FROZEN"
    assert value["environment_binding"][
        "structural_or_literal_authority_source_is_sufficient"
    ] is False
    assert value["environment_binding"]["disposition"] == "FAIL_CLOSED"
    assert value["environment_binding"]["test_proof_authorizes_production"] is False

    validation = value["validation_semantics"]
    assert validation["proof_verification_is_issuance"] is False
    assert validation["proof_validated_is_account_committed"] is False
    assert set(validation["does_not_prove"]) >= {
        "account_id is genuine",
        "account genesis COMMITTED",
        "reservation exists or was consumed",
        "logical operation is unique",
    }
    assert value["validation_output"]["name_AcceptedAccountGenesisRootProof"] == (
        "NOT_FROZEN"
    )
    assert value["validation_output"]["owner"] == "DESIGN_BLOCKED"

    consumption = value["consumption_semantics"]
    assert consumption["single_use"] == "NOT_FROZEN"
    assert consumption["multi_use"] == "NOT_FROZEN"
    assert consumption["operation_bound"] == "NOT_FROZEN"
    replay = value["replay"]
    assert replay["R_to_O1_acct_A_then_O2_acct_B"] == (
        "DOMAIN_SEMANTICS_NOT_FROZEN"
    )
    assert replay["one_external_R_validated_twice"].startswith("NOT_FROZEN")
    assert "MUST NOT create a second account" in replay[
        "same_proof_same_logical_operation_retry"
    ]

    assert value["historical_provenance"]["current_availability"] == "NOT_AVAILABLE"
    assert value["historical_provenance"][
        "current_issuer_state_is_historical_substitute"
    ] is False
    assert value["restart"]["current_membership_substitutes_missing_history"] is False
    assert value["rollback"]["separate_weaker_trust_path"] is False
    assert value["rollback"]["current_root_proof_anti_rollback"] == "NOT_AVAILABLE"

    non_circular = value["non_circular_root"]
    assert non_circular["target_account"] == "acct_A"
    assert all(
        non_circular[field] is False
        for field in (
            "created_account_itself_allowed",
            "account_scoped_operator_identity_allowed",
            "account_scoped_device_installation_allowed",
            "account_scoped_workspace_allowed",
            "account_scoped_authority_allowed",
            "legitimacy_may_depend_on_entity_requiring_target_account_to_exist",
        )
    )

    boundary = value["authority_boundary"]
    assert boundary["validator_options"]["E_DESIGN_BLOCKED"] == "SELECTED"
    assert boundary["validator_owner"] == "DESIGN_BLOCKED"
    assert boundary["genesis_final_decision_owner"] == "NOT_FOUND"
    assert len(boundary["separations"]) == 5
    failures = value["failure_semantics"]
    assert failures["public_literals"].startswith("NOT_FROZEN")
    assert all(
        outcome.startswith("DENY")
        for name, outcome in failures["classes"].items()
        if name not in {"proof already consumed", "operation binding mismatch"}
    )

    matrix = value["root_proof_capability_matrix"]
    assert set(matrix) == {
        "independent pre-account provenance",
        "issuer authenticity",
        "proof uniqueness",
        "account candidate binding",
        "operation binding",
        "request-semantic binding",
        "environment binding",
        "freshness",
        "revocation",
        "replay semantics",
        "historical verification",
        "restart recovery",
        "anti-rollback",
        "end-to-end genuine root-proof usability",
    }
    allowed = {"AVAILABLE", "PARTIAL", "NOT_AVAILABLE", "NOT_FROZEN", "NOT_APPLICABLE"}
    assert {item["status"] for item in matrix.values()} <= allowed
    assert matrix["account candidate binding"]["status"] == "AVAILABLE"
    assert matrix["issuer authenticity"]["status"] == "NOT_AVAILABLE"
    assert matrix["replay semantics"]["status"] == "NOT_FROZEN"
    assert matrix["end-to-end genuine root-proof usability"]["status"] == (
        "NOT_AVAILABLE"
    )

    parity = value["cross_artifact_parity"]
    assert parity["parity"] == "PASS"
    assert parity["independent_root_required"] is True
    assert parity["circular_root_forbidden"] is True
    assert parity["proof_id_is_operation_id"] is False
    assert parity["issuer_implies_account_id_mint_owner"] is False
    assert parity["subject_cardinality"] == "NOT_FROZEN"
    assert parity["same_proof_replay_semantics"] == "NOT_FROZEN"

    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_ROOT_PROOF_INSUFFICIENT_SEMANTICS"
    )
    assert value["result"]["intrinsic_status"] == "DESIGN_BLOCKED"
    assert value["result"]["upstream_status"] == "BLOCKED_UPSTREAM"
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert value["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"

    evidence = value["m03_critical_behavior_evidence"]
    assert evidence["purpose"] == INITIAL_SECURITY_ESTABLISHMENT_ONLY
    assert evidence["cryptographic_issuer_authenticity_inferred"] is False
    assert set(evidence.values()) >= {"REQUIRED"}


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


def test_m03_inventory_matches_exact_runtime_schemas() -> None:
    value = load()
    inventory = value["existing_candidate_inventory"]
    assert inventory["authority_source_exact"] == AUTHORITY_SOURCE
    expected = [field.name for field in fields(FirstRunBootstrapClaim)] + [
        field.name for field in fields(ProvisioningMembershipBinding)
    ]
    assert [row["field"] for row in inventory["rows"]] == expected
    required = {
        "field",
        "source",
        "owner",
        "authenticated",
        "freshness_semantics",
        "account_binding",
        "operation_binding",
        "environment_binding",
        "historical_retention",
        "usable_for_AccountGenesis",
        "gap",
    }
    assert all(set(row) == required for row in inventory["rows"])


def test_m03_critical_behavior_evidence_matches_production_source() -> None:
    source = inspect.getsource(FirstRunBootstrapAuthority.consume)
    required_fragments = (
        "accepted_claim != claim",
        "binding.claim_fingerprint_sha256 != claim.claim_fingerprint_sha256",
        "binding.complete_claim_content_fingerprint_sha256 != computed_claim",
        "binding.authority_source != AUTHORITY_SOURCE",
        "binding.provisioning_context_fingerprint_sha256",
        "!= claim.provisioning_context_fingerprint_sha256",
        "claim.intended_operator_id",
        "claim.bootstrap_generation",
        "claim.bootstrap_revision",
        "if now < issued:",
        "if now > expires:",
        'if purpose != INITIAL_SECURITY_ESTABLISHMENT_ONLY:',
        'if consumed in state.consumed_authorities or any(',
        '_deny("BOOTSTRAP_REPLAY_DENIED")',
    )
    assert all(fragment in source for fragment in required_fragments)
    evidence = load()["m03_critical_behavior_evidence"]
    assert evidence["accepted_claim_equals_supplied_claim"] == "REQUIRED"
    assert evidence["issued_at_lte_now_lte_expires_at"] == "REQUIRED"
    assert evidence["replay_consumption_denied"] == "REQUIRED"
    assert evidence["cryptographic_issuer_authenticity_inferred"] is False


def test_cross_artifact_parity_is_derived_from_actual_sources() -> None:
    validate_cross_artifact_parity(
        load(),
        read_json(TOPOLOGY),
        read_json(RECONCILIATION),
        read_json(GENESIS_MODEL),
        read_json(SUBJECT),
        read_json(IDEMPOTENCY),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda topology: topology["root_proof_boundary"][
            "first_account_self_authorization"
        ].update(independent_accepted_root_proof_required=False),
        lambda topology: topology["root_proof_boundary"]["non_circular_root"].update(
            account_scoped_operator_identity_allowed=True
        ),
    ],
    ids=["independent-root-drift", "operator-child-circularity-drift"],
)
def test_cross_artifact_parity_mutations_fail(mutation) -> None:
    topology = read_json(TOPOLOGY)
    mutation(topology)
    with pytest.raises(AssertionError):
        validate_cross_artifact_parity(
            load(),
            topology,
            read_json(RECONCILIATION),
            read_json(GENESIS_MODEL),
            read_json(SUBJECT),
            read_json(IDEMPOTENCY),
        )


@pytest.mark.parametrize(
    ("path", "unsafe"),
    [
        (("frozen_inputs", "self_authorization_using_local_evidence"), "ALLOWED"),
        (("frozen_inputs", "independent_accepted_first_account_root"), "OPTIONAL"),
        (("frozen_inputs", "proof_id_is_operation_id"), True),
        (("operation_binding", "proof_id_is_operation_id"), True),
        (("authentication", "public_SHA_is_authority"), True),
        (("authentication", "unknown_issuer"), "ALLOW"),
        (("environment_binding", "disposition"), "ALLOW"),
        (("environment_binding", "test_proof_authorizes_production"), True),
        (("account_binding", "proof_accepted_is_account_id_minted"), True),
        (("account_binding_security", "proof_bound_account_may_be_rebound"), True),
        (("account_binding_security", "exact_bound_account_mismatch"), "ALLOW"),
        (("account_binding_security", "content_bound_account_id_must_match_candidate"), False),
        (("frozen_inputs", "proof_acceptance_implies_COMMITTED"), True),
        (("validation_semantics", "proof_validated_is_account_committed"), True),
        (("consumption_semantics", "single_use"), "FROZEN"),
        (("replay", "R_to_O1_acct_A_then_O2_acct_B"), "FORBIDDEN"),
        (("authority_boundary", "validator_owner"), "ProvisioningBoundary"),
        (("historical_provenance", "current_issuer_state_is_historical_substitute"), True),
        (("restart", "current_membership_substitutes_missing_history"), True),
        (("non_circular_root", "created_account_itself_allowed"), True),
        (("non_circular_root", "account_scoped_operator_identity_allowed"), True),
        (("non_circular_root", "account_scoped_device_installation_allowed"), True),
        (("non_circular_root", "account_scoped_workspace_allowed"), True),
        (("non_circular_root", "account_scoped_authority_allowed"), True),
        (
            (
                "non_circular_root",
                "legitimacy_may_depend_on_entity_requiring_target_account_to_exist",
            ),
            True,
        ),
        (("implementation_allowed", "CryptoHunterAccountAuthority"), True),
    ],
)
def test_unsafe_contract_mutations_fail(path: tuple[str, ...], unsafe: object) -> None:
    value = deepcopy(load())
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe
    with pytest.raises(AssertionError):
        validate(value)


@pytest.mark.parametrize(
    ("attack", "path", "unsafe"),
    [
        (
            "locally_generated_AccountAuthority_proof_accepted",
            ("frozen_inputs", "self_authorization_using_local_evidence"),
            "ALLOWED",
        ),
        (
            "OperatorIdentity_child_of_acct_A_proves_acct_A",
            ("non_circular_root", "account_scoped_operator_identity_allowed"),
            True,
        ),
        (
            "TEST_proof_used_in_PRODUCTION",
            ("environment_binding", "test_proof_authorizes_production"),
            True,
        ),
        (
            "structurally_valid_unknown_issuer_accepted",
            ("authentication", "unknown_issuer"),
            "ALLOW",
        ),
        ("public_SHA_as_authenticity", ("authentication", "public_SHA_is_authority"), True),
        (
            "proof_bound_acct_A_used_for_acct_B",
            ("account_binding_security", "proof_bound_account_may_be_rebound"),
            True,
        ),
        (
            "current_membership_substitutes_missing_historical_proof",
            ("restart", "current_membership_substitutes_missing_history"),
            True,
        ),
        (
            "accepted_proof_implies_COMMITTED_account",
            ("frozen_inputs", "proof_acceptance_implies_COMMITTED"),
            True,
        ),
        (
            "proof_id_automatically_becomes_logical_operation_id",
            ("operation_binding", "proof_id_is_operation_id"),
            True,
        ),
    ],
)
def test_each_mandatory_redteam_attack_has_an_executable_mutation(
    attack: str, path: tuple[str, ...], unsafe: object
) -> None:
    value = deepcopy(load())
    assert attack in value["mandatory_redteam"]
    target = value
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = unsafe
    with pytest.raises(AssertionError):
        validate(value)
