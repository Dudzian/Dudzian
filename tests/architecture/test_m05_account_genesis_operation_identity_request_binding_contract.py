"""Executable contract checks for AccountGenesis operation/request binding."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_operation_identity_request_binding_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_operation_identity_request_binding_contract.md"


def read_json(name: str) -> dict:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis operation identity and request binding contract\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_operation_identity_request_binding_contract.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_OPERATION_IDENTITY_REQUEST_BINDING_CONTRACT"
    )
    assert value["reviewed_head_supplied"] == (
        "5ed5af00659f2414657dfa80af84907cdd2265e4"
    )
    assert value["repository_head_examined"] != value["reviewed_head_supplied"]
    assert value["provenance"]["source"] == "GIT"
    assert value["provenance"]["formal_advancement_allowed"] is False

    frozen = value["frozen_inputs"]
    assert frozen["account_id_is_logical_operation_identity"] is False
    assert frozen["root_proof_id_is_logical_operation_identity"] is False
    assert frozen["caller_command_id_is_genuine_logical_operation_identity"] is False
    assert frozen["different_operation_ids_prove_distinct_operations"] is False
    assert frozen["subject_cardinality"] == "NOT_FROZEN"

    assert value["selected_or_blocked_model"]["selection"] == "F_DESIGN_BLOCKED"
    assert value["identity_roles"]["identity_owner"] == "DESIGN_BLOCKED"
    assert value["issuance_timing"]["status"] == "NOT_FROZEN"
    assert value["issuance_timing"]["INSIDE_FIRST_PREPARE"].startswith(
        "CONDITIONAL_CANDIDATE"
    )

    request = value["canonical_request"]
    assert request["closed_semantic_field_set_required"] is True
    assert request[
        "frozen_semantic_requirement_may_disappear_from_fingerprint_contract"
    ] is False
    assert request["unknown_semantic_fields"] == "REJECT"
    assert request["ambiguous_optional_fields"] == "FORBIDDEN"
    assert request["required_semantic_slots"] == [
        "request_schema_version",
        "environment/trust_domain",
        "requested_genesis_action",
        "product/account_scope",
        "candidate account_id",
        "reservation relationship",
        "root-proof reference or handoff expectation",
    ]
    taxonomy = {row["field"]: row["classification"] for row in value["request_field_taxonomy"]}
    assert taxonomy["request_schema_version"] == "SEMANTIC_FINGERPRINTED"
    assert taxonomy["environment/trust_domain"] == "SEMANTIC_FINGERPRINTED"
    assert taxonomy["requested_genesis_action"] == "SEMANTIC_FINGERPRINTED"
    assert taxonomy["product/account_scope"] == "SEMANTIC_FINGERPRINTED"
    assert taxonomy["account_id"] == "SEMANTIC_FINGERPRINTED"
    assert taxonomy["reservation_identity/relationship"] == "NOT_FROZEN"
    assert taxonomy["root_proof_id or root-proof handoff slot"] == "NOT_FROZEN"
    assert taxonomy["correlation_id"] == "NON_SEMANTIC_TRANSPORT"
    assert taxonomy["causation_id"] == "NON_SEMANTIC_TRANSPORT"
    assert taxonomy["caller command_id"] == "NON_SEMANTIC_TRANSPORT"
    assert taxonomy["subject/cardinality"] == "NOT_ALLOWED"
    assert taxonomy["unknown fields"] == "NOT_ALLOWED"

    fingerprint = value["fingerprint_contract"]
    assert fingerprint["algorithm"] == "SHA-256"
    assert fingerprint["is_authority"] is False
    assert fingerprint["is_freshness"] is False
    assert fingerprint["is_operation_ownership"] is False
    assert fingerprint["is_authorization"] is False
    assert fingerprint["closed_field_set"] is True
    assert fingerprint["explicit_unique_domain_separation"] == "REQUIRED"
    assert fingerprint["domain_status"] == "EXACT_DOMAIN_LITERAL_NOT_FROZEN"
    assert fingerprint["schema_version_in_preimage"] is True
    assert fingerprint["environment_in_preimage"] is True
    assert value["versioning"]["silent_reinterpretation"] is False

    progression = value["idempotency_progression"]
    assert progression["same_operation_same_canonical_request"] == (
        "SAME_OPERATION_SEMANTICS_AND_SAME_BOUND_IDENTITIES"
    )
    assert progression["always_same_transient_state_or_result"] is False
    assert progression["idempotency_prevents_duplicate_side_effects"] is True
    assert progression[
        "idempotency_prevents_legitimate_first_time_state_progression"
    ] is False
    assert progression["duplicate_reservation"] == "FORBIDDEN"
    assert progression["duplicate_account_candidate"] == "FORBIDDEN"
    assert progression["duplicate_genesis_commit"] == "FORBIDDEN"
    assert progression[
        "duplicate_already_recorded_proof_binding_or_consumption"
    ] == "FORBIDDEN"
    assert progression[
        "first_not_yet_performed_authorized_transition_on_recovered_nonterminal_operation"
    ] == "ALLOWED"
    assert progression["progression_preconditions"] == [
        "same operation identity",
        "same canonical request",
        "same reservation/account_id",
        "authenticated CAS/generation",
        "root-proof contract",
        "state-machine legality",
    ]

    matrix = {row["case"]: row["outcome"] for row in value["retry_matrix"]}
    assert matrix["same operation ID + same request + NEW_OR_UNOBSERVED"].startswith(
        "AUTHORITY_PROOF_REQUIRED"
    )
    assert matrix["same operation ID + same request + RESERVED"].startswith(
        "RECOVER_SAME_OPERATION_AND_CONTINUE_OR_RETURN_TERMINAL_OUTCOME"
    )
    assert "same reservation/account_id/request" in matrix[
        "same operation ID + same request + RESERVED"
    ]
    assert "never reallocate" in matrix[
        "same operation ID + same request + RESERVED"
    ]
    assert matrix["same operation ID + same request + PREPARED"].startswith(
        "RECOVER_SAME_OPERATION_AND_CONTINUE_OR_RETURN_TERMINAL_OUTCOME"
    )
    assert "same terminal decision" in matrix[
        "same operation ID + same request + PREPARED"
    ]
    assert matrix["same operation ID + same request + COMMITTED"] == (
        "RETURN_SAME_COMMITTED_OUTCOME; NO_NEW_SIDE_EFFECTS; no new genesis, "
        "proof consumption, reservation, or account"
    )
    assert matrix["same operation ID + same request + ABORTED"] == (
        "RETURN_SAME_ABORTED_OUTCOME; NO_NEW_SIDE_EFFECTS; no restart whether "
        "reservation is ABORTED_HELD or RELEASED_TOMBSTONED"
    )
    assert matrix["same operation ID + different request"] == (
        "CONFLICT / CONTRACT_INCONSISTENT; NO_MUTATION at every durable state"
    )
    assert matrix["different operation ID + same request"].startswith(
        "UNKNOWN / AUTHORITY_PROOF_REQUIRED"
    )
    assert matrix["caller supplies new command ID after crash"].startswith(
        "FAIL_CLOSED"
    )

    assert value["account_binding"]["account_id_substitution"] == "FORBIDDEN"
    assert value["account_binding"]["same_operation_may_remint_after_crash"] is False
    assert value["account_binding"]["missing_recovery_identity_may_allocate_acct_B"] is False
    assert value["reservation_binding"]["row_existence_proves_operation_ownership"] is False
    assert value["reservation_binding"]["substitution"].startswith("DENY")
    assert value["root_proof_binding"]["proof_id_is_operation_id"] is False
    assert value["root_proof_binding"]["after_PREPARED_substitution"].startswith("DENY")
    handoff = value["canonical_request"]["root_proof_handoff_progression"]
    assert handoff[
        "authorized_proof_arrival_satisfying_previously_bound_handoff_is_request_mutation"
    ] is False
    assert handoff["classification"] == (
        "AUTHENTICATED_OPERATION_STATE_AND_PROVENANCE_PROGRESSION"
    )
    assert value["root_proof_binding"][
        "already_durable_binding_or_consumption_retry"
    ] == "FORBIDDEN"
    assert value["root_proof_binding"]["single_use"] == "NOT_FROZEN"
    assert value["root_proof_binding"]["multi_use"] == "NOT_FROZEN"
    assert value["root_proof_binding"]["cross_operation_reuse"] == "NOT_FROZEN"
    assert value["environment_isolation"]["TEST_request_accepted_as_PRODUCTION"] is False
    assert value["state_machine_binding"]["PREPARED_request_mutation"] == "FORBIDDEN"
    assert value["abort_release"]["ABORTED"].startswith("terminal")
    assert value["abort_release"]["RELEASE"].startswith("reservation-only")
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_OPERATION_REQUEST_CONTRACT_PARTIALLY_FROZEN"
    )
    assert value["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
    assert frozen["reservation_identity_is_logical_operation_identity"] == (
        "NO_UNLESS_EXPLICITLY_SELECTED_BY_FUTURE_PROTOCOL"
    )
    assert value["authentication"]["operation_request_records_authenticated"] is True
    assert value["authentication"]["valid_HMAC_is_authorized_genesis"] is False
    assert value["authentication"]["valid_HMAC_is_freshness"] is False
    assert value["freshness"]["required"] is True
    assert value["freshness"]["available"] == "NOT_AVAILABLE"
    assert value["freshness"]["anti_rollback_required"] is True
    isolation = value["environment_isolation"]
    assert isolation[
        "canonical_request_must_bind_exact_environment_before_production_commit"
    ] is True
    assert isolation["operation_identity_spaces_must_be_isolated"] is True
    assert isolation["authenticated_record_spaces_must_be_isolated"] is True


def validate_cross_artifact_parity(value: dict) -> None:
    """Read upstream sources and derive parity rather than trusting their PASS."""
    uniqueness = read_json("m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json")
    reservation = read_json("m05_cryptohunter_account_genesis_reservation_state_model.json")
    proof = read_json("m05_account_genesis_root_proof_admission_binding_contract.json")
    topology = read_json("m05_account_genesis_authority_topology_and_ownership.json")
    subject = read_json("m05_cryptohunter_account_genesis_subject_identity_discovery.json")

    assert uniqueness["entity_uniqueness"]["account_id_is_logical_operation_idempotency_identity"] is False
    assert uniqueness["business_uniqueness_boundary"]["subject_to_account_cardinality"] == "NOT_FROZEN"
    assert value["frozen_inputs"]["different_operation_ids_prove_distinct_operations"] == (
        reservation["cross_artifact_parity"]["different_ids_prove_distinct_operations"]
    ) == False
    assert value["account_binding"]["same_operation_may_remint_after_crash"] == (
        reservation["cross_artifact_parity"]["same_operation_may_remint_after_crash"]
    ) == False
    recovery = reservation["recovery_identity"]
    assert recovery[
        "logical_operation_identity_and_reservation_recovery_identity_are_distinct_semantic_roles"
    ] is True
    assert recovery[
        "logical_operation_identity_equals_reservation_recovery_identity"
    ] == (
        "NOT_FROZEN; may be the same object only after explicit frozen model selection"
    )
    assert reservation["operation_state_machine"]["ABORTED"].startswith("immutable terminal")
    assert reservation["abort_release_semantics"]["RELEASE_belongs_to"] == (
        "reservation disposition only"
    )
    assert value["root_proof_binding"]["proof_id_is_operation_id"] == (
        proof["operation_binding"]["proof_id_is_operation_id"]
    ) == False
    assert proof["result"]["primary_result"] == "ACCOUNT_GENESIS_ROOT_PROOF_INSUFFICIENT_SEMANTICS"
    assert topology["cross_artifact_parity"]["account_id_not_operation_identity"] is True
    assert topology["cross_artifact_parity"]["RELEASE_reservation_only"] is True
    assert value["frozen_inputs"]["subject_cardinality"] == (
        subject["result"]["subject_account_cardinality"]
    ) == "NOT_FROZEN"
    assert value["frozen_inputs"]["account_id_is_logical_operation_identity"] == (
        uniqueness["entity_uniqueness"][
            "account_id_is_logical_operation_idempotency_identity"
        ]
    ) == False
    assert value["cross_artifact_parity"]["actual_sources_validated"] is True

    upstream_retry = {
        row["case"]: row["outcome"] for row in reservation["retry_matrix"]
    }
    current_retry = {row["case"]: row["outcome"] for row in value["retry_matrix"]}
    assert upstream_retry["same operation, same semantics, RESERVED/PREPARED"] == (
        "recover same reservation/account_id; continue or return authenticated "
        "terminal outcome; never reallocate"
    )
    assert current_retry["same operation ID + same request + RESERVED"].startswith(
        "RECOVER_SAME_OPERATION_AND_CONTINUE_OR_RETURN_TERMINAL_OUTCOME"
    )
    assert current_retry["same operation ID + same request + PREPARED"].startswith(
        "RECOVER_SAME_OPERATION_AND_CONTINUE_OR_RETURN_TERMINAL_OUTCOME"
    )
    assert upstream_retry["same operation, COMMITTED"] == "return same committed outcome"
    assert current_retry["same operation ID + same request + COMMITTED"].startswith(
        "RETURN_SAME_COMMITTED_OUTCOME"
    )
    assert upstream_retry["same operation, ABORTED"].startswith(
        "always return ABORTED terminal operation outcome"
    )
    assert current_retry["same operation ID + same request + ABORTED"].startswith(
        "RETURN_SAME_ABORTED_OUTCOME"
    )


def test_contract_and_projection() -> None:
    value = load()
    validate(value)
    validate_cross_artifact_parity(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


MANDATORY_MUTATIONS = {
    "caller command ID promoted to genuine operation ID": lambda v: v[
        "frozen_inputs"
    ].__setitem__("caller_command_id_is_genuine_logical_operation_identity", True),
    "account_id promoted to operation ID": lambda v: v["frozen_inputs"].__setitem__(
        "account_id_is_logical_operation_identity", True
    ),
    "same genuine operation identity + changed request accepted": lambda v: next(
        row
        for row in v["retry_matrix"]
        if row["case"] == "same operation ID + different request"
    ).__setitem__("outcome", "ACCEPT"),
    "acct_A changes to acct_B inside same operation": lambda v: v[
        "account_binding"
    ].__setitem__("account_id_substitution", "ALLOW"),
    "reservation_A changes to reservation_B without authenticated migration": lambda v: v[
        "reservation_binding"
    ].__setitem__("substitution", "ALLOW"),
    "proof_A silently replaced by proof_B after PREPARED": lambda v: v[
        "root_proof_binding"
    ].__setitem__("after_PREPARED_substitution", "ALLOW"),
    "TEST operation/request accepted as PRODUCTION": lambda v: v[
        "environment_isolation"
    ].__setitem__("TEST_request_accepted_as_PRODUCTION", True),
    "request fingerprint treated as authority": lambda v: v[
        "fingerprint_contract"
    ].__setitem__("is_authority", True),
    "missing recovery identity after crash causes new account allocation": lambda v: v[
        "account_binding"
    ].__setitem__("missing_recovery_identity_may_allocate_acct_B", True),
    "unknown request field silently ignored": lambda v: v[
        "canonical_request"
    ].__setitem__("unknown_semantic_fields", "IGNORE"),
    "preterminal same-operation same-request retry treated as immutable result with progression forbidden": lambda v: next(
        row
        for row in v["retry_matrix"]
        if row["case"] == "same operation ID + same request + RESERVED"
    ).__setitem__("outcome", "RETURN_SAME_RESULT_ONLY / NO_PROGRESS"),
    "COMMITTED retry may execute genesis/proof consumption again": lambda v: next(
        row
        for row in v["retry_matrix"]
        if row["case"] == "same operation ID + same request + COMMITTED"
    ).__setitem__("outcome", "EXECUTE_GENESIS_AND_PROOF_CONSUMPTION_AGAIN"),
    "ABORTED retry may restart operation": lambda v: next(
        row
        for row in v["retry_matrix"]
        if row["case"] == "same operation ID + same request + ABORTED"
    ).__setitem__("outcome", "RESTART_OPERATION"),
    "authorized proof arrival satisfying existing handoff treated as canonical request mutation": lambda v: v[
        "canonical_request"
    ]["root_proof_handoff_progression"].__setitem__(
        "authorized_proof_arrival_satisfying_previously_bound_handoff_is_request_mutation",
        True,
    ),
}

HARDENING_MUTATIONS = [
    lambda v: v["canonical_request"]["required_semantic_slots"].remove(
        "candidate account_id"
    ),
    lambda v: v["canonical_request"]["required_semantic_slots"].remove(
        "reservation relationship"
    ),
    lambda v: v["canonical_request"]["required_semantic_slots"].remove(
        "root-proof reference or handoff expectation"
    ),
    lambda v: next(
        row for row in v["request_field_taxonomy"] if row["field"] == "account_id"
    ).__setitem__("classification", "NON_SEMANTIC_TRANSPORT"),
    lambda v: v["account_binding"].__setitem__(
        "same_operation_may_remint_after_crash", True
    ),
    lambda v: v["frozen_inputs"].__setitem__(
        "reservation_identity_is_logical_operation_identity", "YES"
    ),
    lambda v: v["reservation_binding"].__setitem__(
        "row_existence_proves_operation_ownership", True
    ),
    lambda v: v["authentication"].__setitem__(
        "operation_request_records_authenticated", False
    ),
    lambda v: v["freshness"].__setitem__("anti_rollback_required", False),
    lambda v: v["freshness"].__setitem__("required", False),
    lambda v: v["environment_isolation"].__setitem__(
        "operation_identity_spaces_must_be_isolated", False
    ),
    lambda v: v["environment_isolation"].__setitem__(
        "authenticated_record_spaces_must_be_isolated", False
    ),
    lambda v: v["fingerprint_contract"].__setitem__(
        "explicit_unique_domain_separation", "NOT_REQUIRED"
    ),
    lambda v: v["fingerprint_contract"].__setitem__("closed_field_set", False),
]


@pytest.mark.parametrize("attack,mutate", MANDATORY_MUTATIONS.items())
def test_mandatory_redteam_mutations_are_rejected(attack, mutate) -> None:
    attacked = deepcopy(load())
    mutate(attacked)
    with pytest.raises(AssertionError):
        validate(attacked)


def test_mandatory_redteam_cases_have_exact_executable_mapping() -> None:
    assert set(load()["mandatory_redteam"]["cases"]) == set(MANDATORY_MUTATIONS)


@pytest.mark.parametrize("mutate", HARDENING_MUTATIONS)
def test_hardening_mutations_are_rejected(mutate) -> None:
    attacked = deepcopy(load())
    mutate(attacked)
    with pytest.raises(AssertionError):
        validate(attacked)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda v: v["account_binding"].__setitem__(
            "same_operation_may_remint_after_crash", True
        ),
        lambda v: v["frozen_inputs"].__setitem__(
            "subject_cardinality", "FROZEN_ONE_TO_ONE"
        ),
    ],
)
def test_cross_artifact_mutations_are_rejected(mutate) -> None:
    attacked = deepcopy(load())
    mutate(attacked)
    with pytest.raises(AssertionError):
        validate_cross_artifact_parity(attacked)
