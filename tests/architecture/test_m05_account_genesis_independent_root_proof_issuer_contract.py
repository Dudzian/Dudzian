"""Executable contract for the independent pre-account root-proof issuer."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
)

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.md"


def load(path: Path = MACHINE) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis independent root-proof issuer contract\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_independent_root_proof_issuer_contract.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(v: dict) -> None:
    assert v["result"] == "ACCOUNT_GENESIS_ROOT_PROOF_ISSUER_CONTRACT_CAN_BE_FROZEN"
    assert v["provenance"] == {"classification": "UNKNOWN", "finding_scope": "CURRENT_TREE_ONLY", "formal_project_advancement": "WITHHELD"}
    a = v["authority_model"]
    assert a["selected"] == "INDEPENDENT_PRE_ACCOUNT_ROOT_PROOF_ISSUER"
    assert not any(a[k] for k in ("root_proof_issuer_equals_CHA", "root_proof_issuer_equals_FreshnessAuthority", "root_proof_issuer_equals_Catalog_authority", "root_proof_issuer_equals_local_projection"))
    assert "never by candidate data, TOFU" in a["trust_rule"]
    assert v["m03_decision"]["classification"] == "M03_ONLY_PROVISIONING_INPUT"
    assert v["m03_decision"]["genuine_pre_account_entitlement_available"] is False
    e = v["bootstrap_entitlement"]
    assert e["cardinality"].startswith("exactly one logical operation and one account_id")
    assert e["reusable"] is e["transferable_between_trust_domains"] is e["caller_may_choose_or_replace"] is False
    assert "distinct registries" in e["test_production_separation"]
    issue = v["issuance_protocol"]
    assert issue["point"] == "B_AFTER_INITIAL_BINDING_BEFORE_PREPARED"
    assert issue["issuer_may_choose_account_or_operation_or_reservation"] is False
    fields = set(v["root_proof_object"]["signed_payload_fields"])
    assert {"environment", "trust_domain", "account_id", "logical_operation_id", "canonical_genesis_request_fingerprint_sha256", "bootstrap_entitlement_id"} <= fields
    ident = v["identity"]
    assert ident["caller_selected"] is False
    assert ident["root_proof_id_is_logical_operation_identity"] is False
    replay = v["replay_and_consumption"]
    assert replay["model"] == "C_STATEFUL_SINGLE_BIND_AT_ISSUANCE_PLUS_IMMUTABLE_EXACT_OPERATION_PROOF"
    assert replay["one_proof_more_than_one_operation"] is replay["one_proof_more_than_one_account"] is replay["parallel_proofs_same_entitlement"] is False
    assert replay["mutable_proof_consumed_state"] is replay["global_consumption_serialization_required"] is False
    assert v["cross_authority_atomicity"]["model"].startswith("NO_DISTRIBUTED_TRANSACTION")
    assert v["cross_authority_atomicity"]["post_issuance_callback"] is False
    keys = v["key_lifecycle"]
    assert keys["ACTIVE"] == {"issue": True, "historical_verify": True}
    assert keys["VERIFY_ONLY"] == {"issue": False, "historical_verify": True}
    assert keys["REVOKED"]["issue"] is keys["REVOKED"]["proof_alone_is_ultimate_historical_trust"] is False
    crypto = v["crypto"]
    assert crypto["domain_literal"] == "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1"
    assert (crypto["canonicalization"], crypto["digest"], crypto["signature"], crypto["signature_encoding"], crypto["digest_encoding"]) == ("RFC 8785 / JCS over signed_payload only", "SHA-256", "Ed25519", "base64url without padding", "lowercase 64-character hexadecimal")
    assert crypto["key_separation_required"] is True
    accepted = v["cha_validation"]
    assert accepted["timing"] == "before PREPARED"
    assert accepted["success_is_committed"] is accepted["success_is_freshness"] is accepted["success_is_publication"] is False
    assert set(v["fail_closed"].values()) <= {"REJECT", "FAIL_CLOSED"}
    assert v["concurrency"]["process_local_mutex_sufficient"] is False
    status = v["implementation_status"]
    assert status["FreshnessAuthority_implementation_allowed"] is status["CryptoHunterAccountAuthority_implementation_allowed"] is False
    assert status["production_M05"] == "BLOCKED"
    assert status["RootProofIssuer_implementation_allowed_by_this_artifact"] is False


def test_contract_and_projection() -> None:
    value = load()
    validate(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


def test_cross_artifact_contracts_and_m03_implementation() -> None:
    v = load()
    names = set(v["cross_artifact_inputs"])
    for name in names - {"bot_core/runtime/first_run_bootstrap.py"}:
        assert (DOCS / name).is_file()
    physical = load(DOCS / "m05_account_genesis_physical_persistence_crash_atomicity_contract.json")
    operation = load(DOCS / "m05_account_genesis_operation_identity_request_binding_contract.json")
    reservation = load(DOCS / "m05_cryptohunter_account_genesis_reservation_state_model.json")
    topology = load(DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.json")
    substrate = load(DOCS / "m05_account_genesis_security_substrate_contract.json")
    freshness = load(DOCS / "m05_account_genesis_freshness_authority_implementation_readiness_contract.json")
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    assert physical["selected_or_blocked_protocol"]["selection"] == "INITIAL_BINDING_THEN_PREPARED_THEN_FRESHNESS_CAS_THEN_LOCAL_FINAL_COMMIT"
    assert operation["root_proof_binding"]["proof_id_is_operation_id"] is False
    assert reservation["identity_boundaries"]["account_id_is_operation_identity"] is False
    assert topology["root_proof_validation_role"]["may_issue_proof"] is False
    assert substrate["authentication_algorithm"]["selected"].startswith("HMAC-SHA-256")
    assert freshness["implementation_scope"]["FreshnessAuthority_implementation_allowed_after_iteration"] is False
    validate_issuer_admission_parity(v, admission)
    assert AUTHORITY_SOURCE == "external_product_provisioning_boundary"
    assert INITIAL_SECURITY_ESTABLISHMENT_ONLY == "INITIAL_SECURITY_ESTABLISHMENT_ONLY"


MUTATIONS = {
    "cha_self_issues_root_proof": ("authority_model.root_proof_issuer_equals_CHA", True),
    "freshness_credential_reused_without_distinct_role": ("authority_model.root_proof_issuer_equals_FreshnessAuthority", True),
    "catalog_key_reused": ("authority_model.root_proof_issuer_equals_Catalog_authority", True),
    "tofu_issuer_accepted": ("authority_model.trust_rule", "TOFU allowed"),
    "candidate_carried_issuer_key_accepted": ("authority_model.trust_rule", "candidate key allowed"),
    "test_proof_accepted_in_production": ("bootstrap_entitlement.test_production_separation", "shared"),
    "acct_a_proof_accepted_for_acct_b": ("replay_and_consumption.one_proof_more_than_one_account", True),
    "operation_o1_proof_accepted_for_o2": ("replay_and_consumption.one_proof_more_than_one_operation", True),
    "request_r1_proof_accepted_for_r2": ("root_proof_object.signed_payload_fields", ["schema_version"]),
    "root_proof_id_treated_as_operation_identity": ("identity.root_proof_id_is_logical_operation_identity", True),
    "caller_selects_genuine_root_proof_id": ("identity.caller_selected", True),
    "same_proof_authorizes_two_account_ids": ("root_proof_object.signed_payload_fields", ["logical_operation_id"]),
    "same_entitlement_authorizes_two_operations": ("bootstrap_entitlement.cardinality", "unlimited"),
    "stateless_proof_without_cardinality_authority": ("replay_and_consumption.model", "STATELESS"),
    "process_local_mutex_as_global_replay_protection": ("concurrency.process_local_mutex_sufficient", True),
    "crash_allows_second_proof_or_account": ("cross_authority_atomicity.post_issuance_callback", True),
    "revoked_credential_issues_new_proof": ("key_lifecycle.REVOKED.issue", True),
    "verify_only_credential_issues_new_proof": ("key_lifecycle.VERIFY_ONLY.issue", True),
    "wrong_bootstrap_entitlement_accepted": ("root_proof_object.signed_payload_fields", ["account_id"]),
    "unavailable_issuer_history_accepted": ("fail_closed.unverifiable_history", "ACCEPT"),
    "root_proof_alone_creates_committed": ("cha_validation.success_is_committed", True),
    "root_proof_alone_establishes_freshness": ("cha_validation.success_is_freshness", True),
}


def mutate(value: dict, name: str) -> dict:
    changed = deepcopy(value)
    path, replacement = MUTATIONS[name]
    target = changed
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    assert target[parts[-1]] != replacement
    target[parts[-1]] = replacement
    return changed


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_redteam_mutations_are_real_and_rejected(name: str) -> None:
    source = load()
    assert set(source["redteam_mutations"]) == set(MUTATIONS)
    changed = mutate(source, name)
    assert changed != source
    with pytest.raises(AssertionError):
        validate(changed)


def validate_issuer_admission_parity(issuer: dict, admission: dict) -> None:
    """Require both normative artifacts to express one replay/admission contract."""
    replay = issuer["replay_and_consumption"]
    consumption = admission["consumption_semantics"]
    assert consumption["source_of_truth"] == MACHINE.name
    assert admission["frozen_inputs"]["root_proof_consumption_semantics"] == replay["model"]
    assert consumption["model"] == replay["model"]
    assert consumption["max_logical_operations"] == 1
    assert consumption["max_account_ids"] == 1
    assert consumption["post_issuance_CONSUMED_state"] == replay["mutable_proof_consumed_state"] is False
    assert consumption["post_issuance_issuer_callback"] == issuer["cross_authority_atomicity"]["post_issuance_callback"] is False
    assert admission["operation_binding"]["issuance_timing"] == issuer["issuance_protocol"]["point"]
    assert admission["operation_binding"]["current_candidate_binds_logical_operation"] is True
    assert admission["operation_binding"]["current_candidate_binds_canonical_genesis_request"] is True
    assert consumption["account_bound"] is consumption["operation_bound"] is consumption["request_fingerprint_bound"] is True
    assert admission["replay"]["same_proof_same_logical_operation_retry"].startswith("IDEMPOTENT_SAME_PROOF")
    assert admission["replay"]["same_proof_changed_semantic_request"] == "REJECT"
    assert admission["validation_semantics"]["proof_validated_is_account_committed"] is False
    assert admission["validation_semantics"]["proof_validated_is_freshness"] is False
    assert admission["result"]["semantic_status"] == "SEMANTICS_FROZEN"
    assert admission["result"]["production_status"] == "PRODUCTION_ISSUER_ENTITLEMENT_AND_CLAIMANT_AUTHORITY_NOT_AVAILABLE"
    assert issuer["implementation_status"]["RootProofIssuer_implementation_allowed_by_this_artifact"] is False


AUTHORIZATION_MUTATIONS = {
    "issuer_consumption_model_vs_admission_not_frozen": ("admission", "frozen_inputs.root_proof_consumption_semantics", "NOT_FROZEN"),
    "admission_allows_second_operation": ("admission", "consumption_semantics.max_logical_operations", 2),
    "admission_requires_post_issuance_consumed": ("admission", "consumption_semantics.post_issuance_CONSUMED_state", True),
    "public_entitlement_id_is_authorization": ("issuer", "claimant_authorization.public_entitlement_id_is_authorization", True),
    "wrong_claimant_with_handle_binds": ("issuer", "claimant_authorization.authorization_rule", "handle wins"),
    "unauthenticated_issuance_requester": ("issuer", "issuance_request.requester_authentication", "NONE"),
    "unauthenticated_initial_binding_tuple": ("issuer", "initial_binding_authenticity.caller_supplied_tuple_alone_sufficient", True),
    "wrong_cha_requester_key": ("issuer", "initial_binding_authenticity.key_rule", "any key"),
    "issuance_request_cross_domain_replay": ("issuer", "issuance_request.cross_protocol_replay", True),
    "entitlement_subject_changed_before_bind": ("issuer", "bootstrap_entitlement.claimant_key_binding", "mutable after lookup"),
    "unauthorized_first_writer_wins": ("issuer", "concurrency.unauthorized_first_bind_possible", True),
    "caller_selected_entitlement_alias": ("issuer", "bootstrap_entitlement.alias_rule", "caller selects target"),
    "bind_before_claimant_authorization": ("issuer", "bind_authorization_order", ["1_BIND", "2_AUTHORIZE"]),
    "lost_response_different_tuple_returns_old_proof": ("issuer", "lost_response.different_attempt_id", "RETURN_OLD_PROOF"),
    "test_request_against_production_entitlement": ("issuer", "bootstrap_entitlement.test_production_separation", "TEST MAY AUTHORIZE PRODUCTION"),
}


def validate_authorization(value: dict) -> None:
    entitlement = value["bootstrap_entitlement"]
    assert entitlement["claimant_model"] == "A_SUBJECT_BOUND_ENTITLEMENT"
    assert entitlement["entitlement_id_is_authorization"] is False
    assert "exact claimant_key_id" in entitlement["claimant_key_binding"]
    assert "caller data cannot redirect" in entitlement["alias_rule"]
    claimant = value["claimant_authorization"]
    assert claimant["selected_model"] == "A_SUBJECT_BOUND_ENTITLEMENT"
    assert claimant["currently_available"] is False
    assert claimant["public_entitlement_id_is_authorization"] is False
    assert claimant["authorization_rule"].startswith("both genuine CHA")
    request = value["issuance_request"]
    assert request["submitter"].startswith("genuine CryptoHunterAccountAuthority only")
    assert request["requester_credential_model"] == "DISTINCT_CHA_ROOT_PROOF_ISSUANCE_REQUESTER_CREDENTIAL"
    assert request["credential_role"] == "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
    assert request["key_material_reuse_with_freshness_proposer"] is False
    assert request["custody_reuse_with_freshness_proposer"] is False
    assert request["requester_authentication"].startswith("Ed25519 signature verified")
    assert request["domain_literal"] == "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUEST_V1"
    assert request["domain_literal"] != value["crypto"]["domain_literal"]
    assert request["cross_protocol_replay"] is request["root_proof_domain_reuse"] is False
    binding = value["initial_binding_authenticity"]
    assert binding["model"] == "CHA_AUTHENTICATED_EXACT_INITIAL_BINDING_DIGEST_AND_REFERENCE"
    assert binding["caller_supplied_tuple_alone_sufficient"] is False
    assert binding["key_rule"].startswith("BRANCHED: NEW_BIND_AUTHORIZATION")
    assert value["bind_authorization_order"][:4] == ["1_RESOLVE_AUTHORITATIVE_ENTITLEMENT", "2_AUTHENTICATE_ISSUANCE_REQUESTER", "3_VERIFY_ENTITLEMENT_CLAIMANT_AUTHORIZATION", "4_VERIFY_GENUINE_INITIAL_BINDING_EXACT_TUPLE"]
    assert value["lost_response"]["different_attempt_id"].startswith("FAIL_CLOSED / NOT_EXACT_RETRY")
    assert value["concurrency"]["unauthorized_first_bind_possible"] is False
    assert "distinct registries" in entitlement["test_production_separation"]


def mutate_named(value: dict, path: str, replacement: object) -> dict:
    changed = deepcopy(value)
    target = changed
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    assert target[parts[-1]] != replacement
    target[parts[-1]] = replacement
    return changed


def test_issuer_admission_parity_and_claimant_authorization() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    validate(issuer)
    validate_authorization(issuer)
    validate_issuer_admission_parity(issuer, admission)
    assert set(issuer["authorization_redteam_mutations"]) == set(AUTHORIZATION_MUTATIONS)


@pytest.mark.parametrize("name", sorted(AUTHORIZATION_MUTATIONS))
def test_authorization_and_cross_artifact_mutations_rejected(name: str) -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    artifact, path, replacement = AUTHORIZATION_MUTATIONS[name]
    if artifact == "issuer":
        changed = mutate_named(issuer, path, replacement)
        assert changed != issuer
        with pytest.raises(AssertionError):
            validate_authorization(changed)
    else:
        changed = mutate_named(admission, path, replacement)
        assert changed != admission
        with pytest.raises(AssertionError):
            validate_issuer_admission_parity(issuer, changed)

FRESHNESS = DOCS / "m05_account_genesis_freshness_authority_cas_finalization_contract.json"
PHYSICAL = DOCS / "m05_account_genesis_physical_persistence_crash_atomicity_contract.json"
OPERATION = DOCS / "m05_account_genesis_operation_identity_request_binding_contract.json"
READINESS = DOCS / "m05_account_genesis_freshness_authority_implementation_readiness_contract.json"

ROLE_EVIDENCE_MUTATIONS = {
    "freshness_proposer_as_issuance_requester": ("issuer", "freshness_proposer_role_parity.freshness_proposer_accepted_as_root_proof_requester", True),
    "issuance_requester_as_freshness_proposer": ("freshness", "proposer_authentication_boundary.proposer_key_role", "also root-proof requester"),
    "same_key_material_two_aliases": ("issuer", "issuance_request.key_material_reuse_with_freshness_proposer", True),
    "requester_namespace_aliases_freshness_proposer": ("issuer", "freshness_proposer_role_parity.key_id_namespace_shared", True),
    "same_custody_role_is_sufficient": ("issuer", "issuance_request.custody_reuse_with_freshness_proposer", True),
    "revoked_proposer_accepted_as_requester": ("issuer", "issuance_request.lifecycle.REVOKED", "may authenticate new request"),
    "verify_only_requester_authenticates_new_request": ("issuer", "issuance_request.lifecycle.VERIFY_ONLY", "may authenticate new request"),
    "prepared_root_proof_id_only": ("admission", "validation_output.fields", ["root_proof_id"]),
    "evidence_omits_root_proof_signed_digest": ("admission", "validation_output.fields", ["schema_version"]),
    "evidence_omits_initial_binding_digest_reference": ("admission", "validation_output.fields", ["root_proof_signed_payload_digest_sha256"]),
    "evidence_omits_entitlement_generation": ("admission", "validation_output.fields", ["initial_binding_digest_sha256"]),
    "evidence_omits_claimant_principal_key": ("admission", "validation_output.fields", ["entitlement_generation"]),
    "evidence_omits_requester_key_identity": ("admission", "validation_output.fields", ["provisioning_principal_id"]),
    "evidence_omits_issuer_history_decision": ("admission", "validation_output.fields", ["cha_root_proof_issuance_requester_credential_id"]),
    "current_issuer_state_substitutes_history": ("admission", "admission_recovery.current_issuer_state_substitutes_retained_history", True),
    "caller_reconstructs_evidence_after_crash": ("admission", "admission_recovery.caller_may_reconstruct_missing_evidence", True),
    "evidence_o1_acct_a_reused_o2_acct_b": ("admission", "admission_recovery.continue_PREPARED_only_if", "may mismatch operation/account"),
    "validation_record_claims_committed": ("admission", "validation_output.implies_COMMITTED", True),
    "validation_record_claims_freshness": ("admission", "validation_output.implies_freshness", True),
    "admission_final_owner_not_found": ("admission", "authority_boundary.genesis_final_decision_owner", "NOT_FOUND"),
    "operation_reason_root_proof_timing_blocked": ("operation", "result.reason", "root-proof timing semantics remain blocked"),
    "validation_output_design_blocked": ("admission", "validation_output.owner", "DESIGN_BLOCKED"),
    "admission_evidence_not_found": ("admission", "existing_candidate_inventory.AccountGenesis_admission_evidence.object", "NOT_FOUND"),
}


def validate_role_and_prepared_evidence(
    issuer: dict,
    admission: dict,
    freshness: dict,
    physical: dict,
    topology: dict,
    operation: dict,
    readiness: dict,
) -> None:
    request = issuer["issuance_request"]
    proposer = freshness["proposer_authentication_boundary"]
    parity = issuer["freshness_proposer_role_parity"]
    assert proposer["proposer_key_role"] == "authenticate origin of new candidate as CryptoHunterAccountAuthority only"
    assert request["requester_credential_model"] == "DISTINCT_CHA_ROOT_PROOF_ISSUANCE_REQUESTER_CREDENTIAL"
    assert request["credential_role"] == "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1"
    assert request["key_material_reuse_with_freshness_proposer"] is False
    assert request["custody_reuse_with_freshness_proposer"] is False
    assert parity["roles_distinct"] is True
    assert parity["key_material_shared"] is parity["custody_shared"] is parity["key_id_namespace_shared"] is False
    assert parity["freshness_proposer_accepted_as_root_proof_requester"] is False
    assert parity["root_proof_requester_accepted_as_freshness_proposer"] is False
    assert request["lifecycle"]["VERIFY_ONLY"].startswith("historical")
    assert request["lifecycle"]["REVOKED"].startswith("cannot authenticate a new")
    assert len({request["domain_literal"], issuer["crypto"]["domain_literal"], issuer["claimant_authorization"]["domain_literal"], proposer["domain_literal"]}) == 4

    evidence = admission["validation_output"]
    required = {
        "schema_version", "environment", "trust_domain", "logical_operation_id", "account_id",
        "reservation_identity", "reservation_relation", "canonical_genesis_request_fingerprint_sha256",
        "initial_binding_reference", "initial_binding_digest_sha256", "root_proof_id",
        "root_proof_signed_payload_digest_sha256", "root_proof_signature_or_immutable_signed_proof_reference",
        "root_proof_issuer_id", "issuer_key_id", "issuer_key_version", "bootstrap_entitlement_id",
        "entitlement_generation", "bound_decision_identity_or_reference", "provisioning_principal_id",
        "claimant_key_id", "claimant_key_version", "cha_root_proof_issuance_requester_credential_id",
        "cha_root_proof_issuance_requester_key_version", "issuer_retained_history_decision_reference",
        "lifecycle_checkpoint_reference", "validation_policy_profile_version", "validation_outcome",
    }
    assert required <= set(evidence["fields"])
    assert evidence["object"] == "RootProofAdmissionEvidenceV1"
    assert evidence["separate_validation_receipt"] == "SEPARATE_VALIDATION_RECEIPT_NOT_REQUIRED"
    assert evidence["owner"] == "CryptoHunterAccountAuthority local AccountGenesis authority record"
    assert evidence["validation_outcome"] == "ACCEPTED_FOR_PREPARED_ONLY"
    assert evidence["implies_COMMITTED"] is evidence["implies_freshness"] is False
    assert evidence["claimant_binding_model"].startswith("AUTHORITATIVE_INDIRECT")
    assert "atomically stored with PREPARED" in evidence["persistence"]
    recovery = admission["admission_recovery"]
    assert recovery["current_issuer_state_substitutes_retained_history"] is False
    assert recovery["caller_may_reconstruct_missing_evidence"] is False
    assert recovery["continue_PREPARED_only_if"].startswith("durable RootProofAdmissionEvidenceV1 exactly matches")
    assert admission["existing_candidate_inventory"]["AccountGenesis_admission_evidence"]["object"] == "RootProofAdmissionEvidenceV1"
    assert admission["authority_boundary"]["genesis_final_decision_owner"] == "CryptoHunterAccountAuthority"
    assert admission["authority_boundary"]["root_proof_validation_is_subordinate"] is True
    assert topology["decision"]["owner_identity"] == "CryptoHunterAccountAuthority"
    assert topology["root_proof_validation_role"]["validation_result_is_final_genesis_decision"] is False
    assert "accepted root-proof validation evidence and provenance" in {row["record"] for row in physical["authoritative_records"]}
    assert "accepted root-proof validation evidence/reference with exact provenance" in physical["local_transaction_contract"]["PREPARED_atomic_records"]
    reason = operation["result"]["reason"]
    assert "root-proof timing semantics remain blocked" not in reason
    assert operation["result"]["root_proof_timing_status"] == "FROZEN_AFTER_INITIAL_BINDING_BEFORE_PREPARED"
    assert readiness["implementation_scope"]["FreshnessAuthority_implementation_allowed_after_iteration"] is False
    assert issuer["implementation_status"]["RootProofIssuer_implementation_allowed_by_this_artifact"] is False


def test_freshness_role_and_prepared_evidence_cross_artifact_contract() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    validate_role_and_prepared_evidence(
        issuer, admission, load(FRESHNESS), load(PHYSICAL),
        load(DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.json"),
        load(OPERATION), load(READINESS),
    )
    assert set(issuer["key_role_redteam_mutations"]) | set(admission["admission_evidence_redteam_mutations"]) == set(ROLE_EVIDENCE_MUTATIONS)


@pytest.mark.parametrize("name", sorted(ROLE_EVIDENCE_MUTATIONS))
def test_key_role_and_prepared_evidence_mutations_rejected(name: str) -> None:
    values = {
        "issuer": load(),
        "admission": load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"),
        "freshness": load(FRESHNESS),
        "physical": load(PHYSICAL),
        "topology": load(DOCS / "m05_account_genesis_authority_topology_resolution_after_binding_freeze.json"),
        "operation": load(OPERATION),
        "readiness": load(READINESS),
    }
    artifact, path, replacement = ROLE_EVIDENCE_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_role_and_prepared_evidence(
            values["issuer"], values["admission"], values["freshness"], values["physical"],
            values["topology"], values["operation"], values["readiness"],
        )

ROTATION_RECOVERY_MUTATIONS = {
    "send_before_durable_attempt": ("physical", "root_proof_issuance_attempt_recovery.persist_before_send", False),
    "crash_loses_signed_request": ("issuer", "root_proof_issuance_attempt.fields", ["issuance_attempt_id"]),
    "requester_verify_only_cannot_recover_bound": ("issuer", "credential_rotation_recovery.EXACT_ALREADY_BOUND_RECOVERY.requester_eligibility", "ACTIVE only"),
    "requester_verify_only_creates_unbound_bind": ("issuer", "credential_rotation_recovery.NEW_BIND_AUTHORIZATION.requester_eligibility", "ACTIVE or VERIFY_ONLY"),
    "claimant_verify_only_prevents_bound_recovery": ("issuer", "credential_rotation_recovery.EXACT_ALREADY_BOUND_RECOVERY.claimant_eligibility", "ACTIVE only"),
    "claimant_verify_only_claims_unbound": ("issuer", "credential_rotation_recovery.NEW_BIND_AUTHORIZATION.claimant_eligibility", "ACTIVE or VERIFY_ONLY"),
    "current_requester_substitutes_historical": ("issuer", "credential_rotation_recovery.current_key_substitution", True),
    "current_claimant_substitutes_historical": ("issuer", "credential_rotation_recovery.claimant_rotation", "use current claimant silently"),
    "reconstruct_missing_requester_signature": ("issuer", "root_proof_issuance_attempt.fields", ["claimant_authorization_signature_bytes"]),
    "reconstruct_missing_claimant_signature": ("issuer", "root_proof_issuance_attempt.fields", ["requester_signature_bytes"]),
    "new_attempt_without_authoritative_reconciliation": ("issuer", "lost_response_reconciliation.authoritative_read_required", False),
    "supersede_without_authoritative_unbound": ("issuer", "lost_response_reconciliation.AUTHORITATIVELY_UNBOUND", "supersede without read"),
    "prefer_newest_over_bound_winner": ("issuer", "lost_response_reconciliation.BOUND_TO_ANOTHER_DURABLE_ATTEMPT_SAME_OPERATION", "prefer newest"),
    "adopt_bound_different_tuple": ("issuer", "lost_response_reconciliation.BOUND_TO_DIFFERENT_OPERATION_ACCOUNT_REQUEST", "ADOPT"),
    "verify_only_recovery_is_new_authorization": ("issuer", "credential_rotation_recovery.EXACT_ALREADY_BOUND_RECOVERY.creates_new_bound_decision", True),
    "revoked_requester_signature_alone": ("issuer", "credential_rotation_recovery.REVOKED_historical.signature_alone_sufficient", True),
    "revoked_claimant_signature_alone": ("issuer", "credential_rotation_recovery.REVOKED_historical.missing_evidence", "ACCEPT signature"),
    "admission_omits_winning_attempt": ("admission", "validation_output.fields", ["root_proof_id"]),
    "attempt_id_is_operation_id": ("issuer", "root_proof_issuance_attempt.identity_separation.is_logical_operation_identity", True),
    "process_memory_reconstructs_attempt": ("issuer", "root_proof_issuance_attempt.reconstruction_sources_forbidden", ["caller input"]),
}


def validate_rotation_recovery(issuer: dict, admission: dict, physical: dict) -> None:
    rotation = issuer["credential_rotation_recovery"]
    assert rotation["operations_are_distinct"] == ["NEW_BIND_AUTHORIZATION", "EXACT_ALREADY_BOUND_RECOVERY"]
    new = rotation["NEW_BIND_AUTHORIZATION"]
    assert new["authoritative_precondition"].endswith("authoritatively UNBOUND")
    assert new["requester_eligibility"] == new["claimant_eligibility"] == new["issuer_signing_key_eligibility"] == "ACTIVE only"
    assert new["VERIFY_ONLY_allowed"] is new["REVOKED_allowed"] is False
    historical = rotation["EXACT_ALREADY_BOUND_RECOVERY"]
    assert "ACTIVE or VERIFY_ONLY" in historical["requester_eligibility"]
    assert "ACTIVE or VERIFY_ONLY" in historical["claimant_eligibility"]
    assert historical["creates_new_bound_decision"] is historical["resigning_allowed"] is False
    assert rotation["REVOKED_historical"]["signature_alone_sufficient"] is False
    assert rotation["REVOKED_historical"]["missing_evidence"].startswith("FAIL_CLOSED")
    assert rotation["current_key_substitution"] is False
    assert rotation["claimant_rotation"].startswith("VERIFY_ONLY may verify")

    attempt = issuer["root_proof_issuance_attempt"]
    required = {
        "schema_version", "environment", "trust_domain", "issuance_attempt_id", "logical_operation_id", "account_id",
        "reservation_identity", "reservation_relation", "canonical_genesis_request_fingerprint_sha256",
        "initial_binding_reference", "initial_binding_digest_sha256", "bootstrap_entitlement_id", "entitlement_generation",
        "requester_principal_id", "requester_credential_role", "requester_key_id", "requester_key_version",
        "provisioning_principal_id", "claimant_key_id", "claimant_key_version", "root_proof_issuance_request_signed_payload",
        "canonical_signed_payload_bytes_or_immutable_reference", "signed_payload_digest_sha256", "requester_signature_bytes",
        "claimant_authorization_signature_bytes", "issuance_request_domain_and_profile_version",
        "claimant_authorization_domain_and_profile_version", "attempt_state",
    }
    assert required <= set(attempt["fields"])
    assert attempt["persist_before_send"] is True
    assert attempt["external_send_without_durable_exact_attempt"] == "FORBIDDEN"
    assert attempt["no_response_implies_NOT_SENT"] is False
    assert attempt["identity_separation"]["is_logical_operation_identity"] is False
    assert set(attempt["reconstruction_sources_forbidden"]) == {"caller input", "current requester key", "current claimant key", "current entitlement lookup", "process memory"}

    reconcile = issuer["lost_response_reconciliation"]
    assert reconcile["authoritative_read_required"] is True
    assert reconcile["AUTHORITATIVELY_UNBOUND"].startswith("retry original only if original requester and claimant remain ACTIVE")
    assert "actual authoritative BOUND winner" in reconcile["BOUND_TO_ANOTHER_DURABLE_ATTEMPT_SAME_OPERATION"]
    assert reconcile["BOUND_TO_DIFFERENT_OPERATION_ACCOUNT_REQUEST"].startswith("FAIL_CLOSED")
    assert reconcile["silent_rewrite_historical_bytes"] is False

    fields = set(admission["validation_output"]["fields"])
    assert {"winning_root_proof_issuance_attempt_id", "winning_root_proof_issuance_attempt_digest_sha256"} <= fields
    assert admission["validation_output"]["winning_attempt_rule"].startswith("must reference exact actual issuer BOUND winner")
    assert admission["admission_recovery"]["winning_attempt"].startswith("actual authoritative BOUND attempt controls")

    recovery = physical["root_proof_issuance_attempt_recovery"]
    assert recovery["macro_phase_change"] is False
    assert recovery["position"] == "between INITIAL_BINDING and PREPARED"
    assert recovery["persist_before_send"] is True
    assert recovery["external_send_without_durable_attempt"] == "FORBIDDEN"
    assert recovery["missing_response_implies_not_sent"] is False
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"

    matrix = issuer["issuance_attempt_crash_matrix"]
    assert len(matrix) == 9
    assert {row["case"][0] for row in matrix} == set("ABCDEFGHI")
    assert all(set(row) == {"case", "new_bind_allowed", "historical_lookup_allowed", "winner", "resigning_allowed", "PREPARED_allowed", "fail_closed"} for row in matrix)
    assert all(row["resigning_allowed"] is False for row in matrix)
    by_case = {row["case"]: row for row in matrix}
    assert by_case["D_BOUND_RESPONSE_LOST_REQUESTER_VERIFY_ONLY"]["PREPARED_allowed"] is True
    assert by_case["E_BOUND_RESPONSE_LOST_CLAIMANT_VERIFY_ONLY"]["PREPARED_allowed"] is True
    assert by_case["F_UNBOUND_REQUESTER_VERIFY_ONLY"]["new_bind_allowed"] is False
    assert by_case["H_BOUND_OLD_ATTEMPT_NEWER_LOCAL_EXISTS"]["winner"].startswith("old actual authoritative BOUND")
    assert by_case["I_ISSUER_HISTORY_UNAVAILABLE"]["PREPARED_allowed"] is False


def test_rotation_recovery_and_crash_matrix_contract() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    physical = load(PHYSICAL)
    validate_rotation_recovery(issuer, admission, physical)
    assert set(issuer["rotation_recovery_redteam_mutations"]) == set(ROTATION_RECOVERY_MUTATIONS)


@pytest.mark.parametrize("name", sorted(ROTATION_RECOVERY_MUTATIONS))
def test_rotation_recovery_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "admission": load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"), "physical": load(PHYSICAL)}
    artifact, path, replacement = ROTATION_RECOVERY_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_rotation_recovery(values["issuer"], values["admission"], values["physical"])

ATTEMPT_IDENTITY_MUTATIONS = {
    "caller_selects_issuance_attempt_id": ("issuer", "root_proof_issuance_attempt_identity.issuance_attempt_id.caller_selected", True),
    "replacement_reuses_old_attempt_id": ("issuer", "root_proof_issuance_attempt_identity.replacement.new_issuance_attempt_id_required", False),
    "same_attempt_id_different_immutable_payload": ("issuer", "root_proof_issuance_attempt_identity.issuance_attempt_id.same_id_unequal_payload", "ACCEPT"),
    "attempt_digest_includes_mutable_attempt_state": ("issuer", "root_proof_issuance_attempt_identity.mutable_fields_excluded", ["retry_counters"]),
    "legal_state_transition_changes_attempt_digest": ("issuer", "root_proof_issuance_attempt_identity.conformance_vector.state_changes_digest", True),
    "attempt_digest_is_attempt_id_only": ("issuer", "root_proof_issuance_attempt_identity.digest.meaning", "hash ID only"),
    "attempt_digest_is_request_digest_only": ("issuer", "root_proof_issuance_attempt_identity.digest.meaning", "request digest only"),
    "attempt_digest_omits_initial_binding_digest": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["issuance_attempt_id"]),
    "attempt_digest_omits_entitlement_generation": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["initial_binding_digest_sha256"]),
    "attempt_digest_omits_requester_key_version": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["entitlement_generation"]),
    "attempt_digest_omits_claimant_key_version": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["requester_key_version"]),
    "attempt_digest_omits_requester_signature": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["claimant_key_version"]),
    "attempt_digest_omits_claimant_signature": ("issuer", "root_proof_issuance_attempt_identity.immutable_fields", ["requester_signature_base64url"]),
    "attempt_digest_uses_noncanonical_process_specific_serialization": ("issuer", "root_proof_issuance_attempt_identity.digest.canonicalization", "native JSON"),
    "attempt_digest_has_no_object_domain_separator": ("issuer", "root_proof_issuance_attempt_identity.digest.preimage", "JCS(payload)"),
    "attempt_digest_reuses_root_proof_domain": ("issuer", "root_proof_issuance_attempt_identity.digest.domain_literal", "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1"),
    "admission_evidence_uses_mutable_row_digest": ("admission", "validation_output.winning_attempt_digest_semantics.exact_meaning", "mutable row digest"),
    "admission_evidence_digest_not_equal_winning_attempt_identity_digest": ("admission", "validation_output.winning_attempt_digest_semantics.consistency", "need not match"),
    "supersession_rewrites_old_immutable_attempt_payload": ("issuer", "root_proof_issuance_attempt_identity.replacement.old_immutable_payload_rewritten", True),
    "current_attempt_digest_substituted_for_authoritative_winner_digest": ("admission", "validation_output.winning_attempt_rule", "use current attempt"),
}


def attempt_identity_digest(contract: dict, payload: dict) -> tuple[str, str, str]:
    profile = contract["root_proof_issuance_attempt_identity"]["digest"]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    preimage = profile["domain_literal"].encode("ascii") + b"\x00" + canonical.encode("utf-8")
    return canonical, preimage.hex(), hashlib.sha256(preimage).hexdigest()


def validate_attempt_identity(issuer: dict, admission: dict, physical: dict) -> None:
    identity = issuer["root_proof_issuance_attempt_identity"]
    attempt_id = identity["issuance_attempt_id"]
    assert identity["object"] == "RootProofIssuanceAttemptIdentityPayloadV1"
    assert attempt_id["issuer"] == "CryptoHunterAccountAuthority local AccountGenesis authority boundary"
    assert attempt_id["representation"] == "rpa_<canonical lowercase UUIDv7>"
    assert attempt_id["caller_selected"] is attempt_id["process_selected_unauthenticated"] is attempt_id["reusable"] is False
    assert attempt_id["same_id_unequal_payload"] == "FAIL_CLOSED / CORRUPTION_OR_TAMPER"
    required = {
        "schema_version", "environment", "trust_domain", "issuance_attempt_id", "logical_operation_id", "account_id",
        "reservation_identity", "reservation_relation", "canonical_genesis_request_fingerprint_sha256",
        "initial_binding_reference", "initial_binding_digest_sha256", "bootstrap_entitlement_id", "entitlement_generation",
        "requester_principal_id", "requester_credential_role", "requester_key_id", "requester_key_version",
        "provisioning_principal_id", "claimant_key_id", "claimant_key_version",
        "root_proof_issuance_request_signed_payload_digest_sha256", "root_proof_issuance_request_canonical_bytes_reference",
        "requester_signature_base64url", "claimant_authorization_signature_base64url",
        "issuance_request_domain_and_profile_version", "claimant_authorization_domain_and_profile_version",
    }
    assert set(identity["immutable_fields"]) == required
    assert set(identity["mutable_fields_excluded"]) == {"attempt_state", "wall_clock_timestamps", "retry_counters", "diagnostic_text", "transport_error_strings", "current_process_metadata"}
    digest = identity["digest"]
    assert digest["meaning"].startswith("SHA-256 digest of RootProofIssuanceAttemptIdentityPayloadV1 only")
    assert (digest["canonicalization"], digest["text_encoding"], digest["algorithm"], digest["representation"]) == ("RFC 8785 / JCS", "UTF-8", "SHA-256", "lowercase 64-character hexadecimal")
    assert digest["domain_literal"] == "CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_ATTEMPT_IDENTITY_V1"
    assert digest["preimage"] == "ASCII(domain_literal) || 0x00 || UTF8(JCS(RootProofIssuanceAttemptIdentityPayloadV1))"
    assert digest["domain_literal"] not in {issuer["crypto"]["domain_literal"], issuer["issuance_request"]["domain_literal"], issuer["claimant_authorization"]["domain_literal"]}
    replacement = identity["replacement"]
    assert replacement["authoritative_UNBOUND_required"] is replacement["old_id_and_digest_preserved"] is replacement["new_issuance_attempt_id_required"] is replacement["new_immutable_digest_required"] is True
    assert replacement["old_immutable_payload_rewritten"] is False
    assert identity["concurrency"]["process_local_mutex_establishes_uniqueness"] is False
    assert identity["concurrency"]["authoritative_BOUND_winner_controls"] is True

    vector = identity["conformance_vector"]
    canonical, preimage_hex, calculated = attempt_identity_digest(issuer, vector["payload"])
    assert canonical == vector["canonical_jcs_utf8"]
    assert preimage_hex == vector["preimage_hex"]
    assert calculated == vector["digest_sha256"]
    assert vector["state_changes_digest"] is False
    for field, original in vector["payload"].items():
        changed = deepcopy(vector["payload"])
        changed[field] = original + "x" if isinstance(original, str) else original + 1
        assert attempt_identity_digest(issuer, changed)[2] != calculated
    for _state in vector["legal_attempt_states"]:
        assert attempt_identity_digest(issuer, vector["payload"])[2] == calculated

    semantics = admission["validation_output"]["winning_attempt_digest_semantics"]
    assert semantics["field"] == "winning_root_proof_issuance_attempt_digest_sha256"
    assert semantics["exact_meaning"].startswith("SHA-256 immutable RootProofIssuanceAttemptIdentityPayloadV1 digest")
    assert set(semantics["forbidden_meanings"]) == {"mutable attempt row digest", "attempt_state digest", "issuance_attempt_id hash", "request digest alone", "implementation-specific serialization"}
    assert semantics["consistency"].endswith("MUST mutually match")
    assert admission["validation_output"]["winning_attempt_rule"].startswith("must reference exact actual issuer BOUND winner")
    recovery = physical["root_proof_issuance_attempt_recovery"]
    assert recovery["persist_before_send"] is True
    assert recovery["identity_contract"].startswith("authority-issued rpa_<canonical lowercase UUIDv7>")
    assert recovery["replacement"].startswith("authoritative UNBOUND first creates a new durable reservation with NOT_YET_DEFINED digest")
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"
    assert len(issuer["issuance_attempt_crash_matrix"]) == 9


def test_attempt_identity_digest_contract_and_conformance_vector() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    physical = load(PHYSICAL)
    validate_attempt_identity(issuer, admission, physical)
    assert set(issuer["attempt_identity_redteam_mutations"]) == set(ATTEMPT_IDENTITY_MUTATIONS)


@pytest.mark.parametrize("name", sorted(ATTEMPT_IDENTITY_MUTATIONS))
def test_attempt_identity_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "admission": load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"), "physical": load(PHYSICAL)}
    artifact, path, replacement = ATTEMPT_IDENTITY_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_attempt_identity(values["issuer"], values["admission"], values["physical"])

END_TO_END_ATTEMPT_MUTATIONS = {
    "issuance_request_omits_attempt_id": ("issuer", "issuance_request.signed_payload_fields", ["schema_version"]),
    "requester_signature_does_not_cover_attempt_id": ("issuer", "issuance_request.attempt_id_binding.requester_signature_covers", False),
    "claimant_signature_does_not_cover_attempt_id": ("issuer", "issuance_request.attempt_id_binding.claimant_signature_covers", False),
    "issuer_bound_history_omits_attempt_id": ("issuer", "issuer_bound_attempt_linkage.bound_history_field", "NONE"),
    "same_external_request_two_attempt_ids_treated_exact_retry": ("issuer", "issuer_bound_attempt_linkage.different_attempt_id_same_other_fields", "EXACT_RETRY"),
    "admission_attempt_id_differs_from_issuer_bound_attempt_id": ("admission", "validation_output.winning_attempt_end_to_end_consistency.admission_winning_attempt_id", "MAY DIFFER"),
    "admission_digest_for_attempt_B_with_issuer_bound_attempt_A": ("admission", "validation_output.winning_attempt_end_to_end_consistency.immutable_digest", "any local digest"),
    "two_CHA_processes_create_two_current_attempts_same_tuple": ("issuer", "local_current_attempt_fence.crash_two_current_same_tuple", True),
    "process_local_mutex_used_as_creation_fence": ("issuer", "local_current_attempt_fence.global_serialization", "process-local mutex"),
    "current_attempt_fence_not_durable": ("physical", "root_proof_issuance_attempt_recovery.reservation_atomicity", "best effort"),
    "replacement_switch_not_atomic": ("issuer", "local_current_attempt_fence.replacement_reservation_switch_atomic", False),
    "replacement_reuses_old_signed_request_with_new_attempt_id": ("issuer", "local_current_attempt_fence.AUTHORITATIVELY_UNBOUND_REPLACEMENT_RESERVATION", "rewrite old request"),
    "delayed_old_attempt_wins_but_newest_local_selected": ("issuer", "local_current_attempt_fence.AUTHORITATIVELY_BOUND", "newest local wins"),
    "new_attempt_wins_but_old_local_attempt_selected": ("issuer", "attempt_creation_concurrency_matrix", []),
    "issuer_history_cannot_distinguish_attempts_but_PREPARED_allowed": ("issuer", "issuer_bound_attempt_linkage.ambiguous_history", "ALLOW PREPARED"),
    "exact_retry_identity_ignores_attempt_id": ("issuer", "issuer_bound_attempt_linkage.exact_retry_fields", ["logical_operation_id"]),
    "caller_can_override_bound_attempt_id": ("issuer", "issuer_bound_attempt_linkage.caller_may_override_bound_attempt_id", True),
    "current_pointer_overrides_authoritative_bound_winner": ("admission", "validation_output.winning_attempt_end_to_end_consistency.current_pointer_is_authoritative", True),
    "crash_leaves_two_current_attempts": ("physical", "root_proof_issuance_attempt_recovery.crash_two_current_attempts_same_tuple", "ALLOWED"),
    "equivalent_request_content_substitutes_different_attempt_identity": ("admission", "validation_output.winning_attempt_end_to_end_consistency.signed_request_identity", "semantic equivalence accepted"),
}


def validate_end_to_end_attempt_binding(issuer: dict, admission: dict, physical: dict) -> None:
    request = issuer["issuance_request"]
    assert "issuance_attempt_id" in request["signed_payload_fields"]
    binding = request["attempt_id_binding"]
    assert binding["required"] is binding["requester_signature_covers"] is binding["claimant_signature_covers"] is True
    assert binding["attempt_digest_not_in_request"].startswith("prevents signature/digest cycle")
    assert "issuance_attempt_id" in issuer["root_proof_object"]["signed_payload_fields"]
    assert issuer["root_proof_object"]["issuance_attempt_id_binding"]["required"] is True

    linkage = issuer["issuer_bound_attempt_linkage"]
    assert linkage["model"] == "END_TO_END_ISSUANCE_ATTEMPT_ID_BINDING"
    assert linkage["bound_history_field"] == "bound_issuance_attempt_id"
    assert linkage["requester_signature_covers_attempt_id"] is linkage["claimant_signature_covers_attempt_id"] is linkage["root_proof_signed_payload_covers_attempt_id"] is True
    assert "issuance_attempt_id" in linkage["exact_retry_fields"]
    assert linkage["different_attempt_id_same_other_fields"].startswith("NOT_EXACT_RETRY")
    assert linkage["caller_may_override_bound_attempt_id"] is False
    assert linkage["ambiguous_history"].startswith("FAIL_CLOSED")
    assert set(linkage["winner_selection_forbidden"]) == {"newest local record", "process identity", "timestamps", "insertion order", "current attempt pointer", "equivalent request content"}

    fence = issuer["local_current_attempt_fence"]
    assert fence["model"] == "PER_OPERATION_CURRENT_ISSUANCE_ATTEMPT_FENCE"
    assert fence["field"] == "current_root_proof_issuance_attempt_id"
    assert fence["global_serialization"].startswith("durable local authority transaction/CAS")
    assert set(fence["exact_idempotency_key_fields"]) == {"environment", "trust_domain", "logical_operation_id", "account_id", "canonical_genesis_request_fingerprint_sha256", "bootstrap_entitlement_id", "entitlement_generation", "requester_principal_id", "requester_credential_role", "requester_key_id", "requester_key_version", "provisioning_principal_id", "claimant_key_id", "claimant_key_version", "initial_binding_digest_sha256"}
    assert set(fence["forbidden_idempotency_inputs"]) == {"wall-clock time", "thread id", "process id", "caller-generated random token"}
    assert fence["EXACT_CURRENT_RESERVED"].startswith("return same authority-owned issuance_attempt_id")
    assert "NOT_YET_DEFINED" in fence["EXACT_CURRENT_RESERVED"]
    assert fence["EXACT_CURRENT_IMMUTABLE"].startswith("return same issuance_attempt_id, same DEFINED immutable digest")
    assert fence["CURRENT_OUTCOME_AMBIGUOUS"].startswith("no replacement before authoritative reconciliation")
    assert fence["AUTHORITATIVELY_UNBOUND_REPLACEMENT_RESERVATION"].startswith("after authenticated authoritative UNBOUND reconciliation")
    assert fence["AUTHORITATIVELY_BOUND"].startswith("issuer_history.bound_issuance_attempt_id wins")
    assert fence["crash_two_current_same_tuple"] is False
    assert fence["replacement_reservation_switch_atomic"] is fence["superseded_history_retained"] is True
    matrix = issuer["attempt_creation_concurrency_matrix"]
    assert len(matrix) == 5
    assert matrix[0]["result"].startswith("one authority-owned rpa_id and one RootProofIssuanceAttemptReservationV1 installed")
    assert matrix[0]["immutable_digest_status"] == "NOT_YET_DEFINED"
    assert matrix[0]["requester_signature_present"] is matrix[0]["claimant_signature_present"] is False
    assert matrix[0]["external_send_eligible"] is False
    assert matrix[1]["identical_identity"].startswith("one immutable digest durable")
    assert matrix[1]["unequal_identity"] == "FAIL_CLOSED / CORRUPTION_OR_TAMPER"
    assert matrix[2]["step"] == "AUTHORITATIVE_UNBOUND_REPLACEMENT_RESERVATION"
    assert matrix[2]["immutable_digest_status"] == "NOT_YET_DEFINED"
    assert matrix[2]["external_send_eligible"] is False
    assert matrix[3]["step"] == "REPLACEMENT_SIGNED_FINALIZATION"
    assert matrix[3]["immutable_digest_status"] == "DEFINED"
    assert matrix[4]["result"].startswith("select exact bound_issuance_attempt_id")
    assert matrix[4]["current_pointer_overrides_winner"] is False

    consistency = admission["validation_output"]["winning_attempt_end_to_end_consistency"]
    assert consistency["admission_winning_attempt_id"] == "MUST equal issuer_history.bound_issuance_attempt_id"
    assert consistency["local_attempt_id"] == "MUST equal admission and issuer history ID"
    assert consistency["immutable_digest"].startswith("MUST resolve the local immutable identity payload for that exact ID")
    assert consistency["signed_request_identity"].startswith("MUST exactly match issuer-retained signed RootProofIssuanceRequest identity")
    assert consistency["current_pointer_is_authoritative"] is False

    recovery = physical["root_proof_issuance_attempt_recovery"]
    assert recovery["reservation_atomicity"].startswith("current fence + RootProofIssuanceAttemptReservationV1 creation/install are one")
    assert recovery["immutable_finalization_atomicity"].startswith("later local authority transaction verifies exact current non-superseded reservation")
    assert recovery["replacement_reservation_switch_atomicity"].startswith("old supersession + durable exact authenticated authoritative-UNBOUND evidence")
    assert recovery["replacement_immutable_finalization_atomicity"].startswith("later reuse immutable_finalization_atomicity")
    assert recovery["crash_two_current_attempts_same_tuple"] == "FORBIDDEN"
    assert recovery["authority_winner"].startswith("issuer_history.bound_issuance_attempt_id overrides")
    assert recovery["persist_before_send"] is True
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"
    assert len(issuer["issuance_attempt_crash_matrix"]) == 9


def semantic_authorization_fixture(**overrides: object) -> dict:
    value = {
        "environment": "TEST",
        "trust_domain": "td_example",
        "logical_operation_id": "ago_018f3e70-7b5b-7c21-8b9a-0123456789ab",
        "account_id": "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab",
        "canonical_genesis_request_fingerprint_sha256": "11" * 32,
        "bootstrap_entitlement_id": "ent_018f3e70-7b5e-7c21-8b9a-0123456789ab",
        "entitlement_generation": 1,
        "requester_principal_id": "CryptoHunterAccountAuthority",
        "requester_credential_role": "ACCOUNT_GENESIS_ROOT_PROOF_ISSUANCE_REQUESTER_V1",
        "requester_key_id": "rpr_test_1",
        "requester_key_version": 1,
        "provisioning_principal_id": "prv_018f3e70-7b5f-7c21-8b9a-0123456789ab",
        "claimant_key_id": "clm_test_1",
        "claimant_key_version": 1,
        "initial_binding_digest_sha256": "22" * 32,
    }
    value.update(overrides)
    return value


def derive_exact_idempotency_key(contract: dict, request: dict) -> tuple[object, ...]:
    fields = contract["local_current_attempt_fence"]["exact_idempotency_key_fields"]
    assert set(fields) == set(request)
    return tuple(request[field] for field in fields)


class SemanticCurrentAttemptAuthority:
    def __init__(self, contract: dict, id_factory) -> None:
        self.contract = contract
        self.id_factory = id_factory
        self.current_attempt_by_operation: dict[tuple[object, ...], dict] = {}
        self.attempt_history_by_operation: dict[tuple[object, ...], list[dict]] = {}
        self.authority_revision = 7
        self.bound_attempt_ids: set[str] = set()

    def operation_slot(self, request: dict) -> tuple[object, ...]:
        fields = self.contract["local_current_attempt_fence"]["operation_slot_key_fields"]
        return tuple(request[field] for field in fields)

    def reserve_or_resolve(self, request: dict) -> str:
        slot = self.operation_slot(request)
        exact_key = derive_exact_idempotency_key(self.contract, request)
        current = self.current_attempt_by_operation.get(slot)
        if current is None:
            attempt_id = self.id_factory()
            self.current_attempt_by_operation[slot] = {
                "exact_key": exact_key, "attempt_id": attempt_id,
                "state": "RESERVED_AWAITING_SIGNATURES", "digest": None,
                "requester_signature": None, "claimant_signature": None,
            }
            return attempt_id
        if current["exact_key"] == exact_key:
            return current["attempt_id"]
        return "RECONCILIATION_REQUIRED"

    def replace_after_authoritative_unbound(
        self, old_request: dict, replacement_request: dict, expected_old_id: str,
        reconciliation_evidence: dict | None,
    ) -> str:
        slot = self.operation_slot(old_request)
        assert slot == self.operation_slot(replacement_request)
        assert reconciliation_evidence is not None, "FAIL_CLOSED / EVIDENCE_REQUIRED"
        evidence = reconciliation_evidence
        assert evidence["authenticated"] is True, "FAIL_CLOSED / UNAUTHENTICATED"
        assert evidence["outcome"] == "AUTHORITATIVELY_UNBOUND", "FAIL_CLOSED / OUTCOME_UNKNOWN"
        exact = {
            "environment": old_request["environment"],
            "trust_domain": old_request["trust_domain"],
            "bootstrap_entitlement_id": old_request["bootstrap_entitlement_id"],
            "entitlement_generation": old_request["entitlement_generation"],
            "logical_operation_id": old_request["logical_operation_id"],
            "account_id": old_request["account_id"],
            "canonical_genesis_request_fingerprint_sha256": old_request["canonical_genesis_request_fingerprint_sha256"],
            "old_issuance_attempt_id": expected_old_id,
            "initial_binding_reference": "initial_binding_v1",
            "initial_binding_digest_sha256": old_request["initial_binding_digest_sha256"],
        }
        assert all(evidence[field] == value for field, value in exact.items()), "FAIL_CLOSED / EVIDENCE_MISMATCH"
        assert evidence["authoritative_state_revision"] == self.authority_revision, "FAIL_CLOSED / STALE_EVIDENCE"
        assert expected_old_id not in self.bound_attempt_ids, "FAIL_CLOSED / LATER_BOUND"
        replacement_key = derive_exact_idempotency_key(self.contract, replacement_request)
        current = self.current_attempt_by_operation[slot]
        if current["attempt_id"] != expected_old_id:
            if current["exact_key"] == replacement_key and current["reconciliation_evidence_digest"] == evidence["authority_authenticated_evidence_digest_sha256"]:
                return current["attempt_id"]
            return "RECONCILIATION_REQUIRED"
        old_history = dict(current)
        old_history["state"] = "SUPERSEDED_AFTER_AUTHORITATIVE_UNBOUND_RECONCILIATION"
        old_history["replacement_authorization_evidence_reference"] = evidence["authority_authenticated_evidence_reference"]
        old_history["replacement_authorization_evidence_digest_sha256"] = evidence["authority_authenticated_evidence_digest_sha256"]
        self.attempt_history_by_operation.setdefault(slot, []).append(old_history)
        replacement_id = self.id_factory()
        self.current_attempt_by_operation[slot] = {
            "exact_key": replacement_key,
            "attempt_id": replacement_id,
            "state": "RESERVED_AWAITING_SIGNATURES",
            "digest": None,
            "requester_signature": None,
            "claimant_signature": None,
            "superseded_old_attempt_id": expected_old_id,
            "reconciliation_evidence_reference": evidence["authority_authenticated_evidence_reference"],
            "reconciliation_evidence_digest": evidence["authority_authenticated_evidence_digest_sha256"],
        }
        return replacement_id

    def sign_requester(self, request: dict, attempt_id: str) -> None:
        current = self.current_attempt_by_operation[self.operation_slot(request)]
        assert current["attempt_id"] == attempt_id
        current["requester_signature"] = "requester-signature"
        current["state"] = "REQUEST_SIGNED_BY_REQUESTER"

    def sign_claimant(self, request: dict, attempt_id: str) -> None:
        current = self.current_attempt_by_operation[self.operation_slot(request)]
        assert current["attempt_id"] == attempt_id
        assert current["requester_signature"] is not None
        current["claimant_signature"] = "claimant-signature"
        current["state"] = "CLAIMANT_AUTHORIZED"

    def finalize_immutable(self, request: dict, attempt_id: str, digest: str) -> None:
        current = self.current_attempt_by_operation[self.operation_slot(request)]
        assert current["attempt_id"] == attempt_id
        assert not current["state"].startswith("SUPERSEDED")
        assert current["requester_signature"] and current["claimant_signature"]
        if current["digest"] is not None:
            assert current["digest"] == digest, "FAIL_CLOSED / CORRUPTION_OR_TAMPER"
            return
        current["digest"] = digest
        current["state"] = "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"

    def resolve_current(self, request: dict) -> dict:
        current = self.current_attempt_by_operation[self.operation_slot(request)]
        return {
            "issuance_attempt_id": current["attempt_id"],
            "state": current["state"],
            "immutable_attempt_digest_status": (
                "NOT_YET_DEFINED" if current["digest"] is None else f"DEFINED({current['digest']})"
            ),
        }

    def external_send_eligible(self, request: dict) -> bool:
        current = self.current_attempt_by_operation[self.operation_slot(request)]
        return current["state"] == "SIGNED_IMMUTABLE_DURABLE_NOT_SENT" and current["digest"] is not None


def deterministic_id_factory():
    ids = iter([
        "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab",
        "rpa_018f3e70-7b6a-7c21-8b9a-0123456789ab",
    ])
    return lambda: next(ids)


def reconciliation_evidence_fixture(request: dict, old_id: str, **overrides: object) -> dict:
    evidence = {
        "schema_version": "RootProofIssuanceReconciliationEvidenceV1",
        "environment": request["environment"],
        "trust_domain": request["trust_domain"],
        "issuer_authority_identity": "IndependentAccountGenesisRootProofIssuer",
        "issuer_registry_identity": "root-proof-entitlement-registry-v1",
        "bootstrap_entitlement_id": request["bootstrap_entitlement_id"],
        "entitlement_generation": request["entitlement_generation"],
        "logical_operation_id": request["logical_operation_id"],
        "account_id": request["account_id"],
        "canonical_genesis_request_fingerprint_sha256": request["canonical_genesis_request_fingerprint_sha256"],
        "old_issuance_attempt_id": old_id,
        "initial_binding_reference": "initial_binding_v1",
        "initial_binding_digest_sha256": request["initial_binding_digest_sha256"],
        "outcome": "AUTHORITATIVELY_UNBOUND",
        "authoritative_state_identity": "entitlement-state-v1",
        "authoritative_state_revision": 7,
        "authority_authenticated_evidence_reference": "issuer-history://reconciliation/7",
        "authority_authenticated_evidence_digest_sha256": "c" * 64,
        "verification_profile_version": "ROOT_PROOF_RECONCILIATION_V1",
        "authenticated": True,
    }
    evidence.update(overrides)
    return evidence


def test_two_processes_resolve_same_reserved_attempt_before_signing() -> None:
    contract = load()
    request = semantic_authorization_fixture()
    authority = SemanticCurrentAttemptAuthority(contract, deterministic_id_factory())
    p1 = authority.reserve_or_resolve(request)
    p2 = authority.reserve_or_resolve(request)
    assert p1 == p2
    assert len(authority.current_attempt_by_operation) == 1
    reserved = next(iter(authority.current_attempt_by_operation.values()))
    assert reserved["state"] == "RESERVED_AWAITING_SIGNATURES"
    assert reserved["digest"] is None
    assert reserved["requester_signature"] is reserved["claimant_signature"] is None


def test_current_attempt_authority_isolation_and_ambiguous_change() -> None:
    contract = load()
    base = semantic_authorization_fixture()
    fields = contract["local_current_attempt_fence"]["exact_idempotency_key_fields"]
    assert derive_exact_idempotency_key(contract, base) == tuple(base[field] for field in fields)
    authority = SemanticCurrentAttemptAuthority(contract, deterministic_id_factory())
    first = authority.reserve_or_resolve(base)
    operation_2 = semantic_authorization_fixture(logical_operation_id="ago_018f3e70-7b6b-7c21-8b9a-0123456789ab")
    second = authority.reserve_or_resolve(operation_2)
    assert first != second
    assert len(authority.current_attempt_by_operation) == 2
    variants = [
        {"account_id": "acct_018f3e70-7b6c-7c21-8b9a-0123456789ab"},
        {"canonical_genesis_request_fingerprint_sha256": "44" * 32},
        {"entitlement_generation": 2}, {"requester_key_version": 2}, {"claimant_key_version": 2},
    ]
    for change in variants:
        changed = semantic_authorization_fixture(**change)
        assert derive_exact_idempotency_key(contract, changed) != derive_exact_idempotency_key(contract, base)
        assert authority.reserve_or_resolve(changed) == "RECONCILIATION_REQUIRED"
    assert len(authority.current_attempt_by_operation) == 2


def test_reservation_signing_finalization_sequence() -> None:
    contract = load()
    request = semantic_authorization_fixture()
    authority = SemanticCurrentAttemptAuthority(contract, deterministic_id_factory())
    attempt_id = authority.reserve_or_resolve(request)
    assert authority.external_send_eligible(request) is False
    authority.sign_requester(request, attempt_id)
    assert authority.external_send_eligible(request) is False
    authority.sign_claimant(request, attempt_id)
    assert authority.external_send_eligible(request) is False
    authority.finalize_immutable(request, attempt_id, "a" * 64)
    assert authority.external_send_eligible(request) is True
    with pytest.raises(AssertionError):
        authority.sign_requester(request, "rpa_018f3e70-7b7a-7c21-8b9a-0123456789ab")


def test_end_to_end_attempt_binding_contract() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    physical = load(PHYSICAL)
    validate_end_to_end_attempt_binding(issuer, admission, physical)
    assert set(issuer["end_to_end_attempt_binding_redteam_mutations"]) == set(END_TO_END_ATTEMPT_MUTATIONS)


@pytest.mark.parametrize("name", sorted(END_TO_END_ATTEMPT_MUTATIONS))
def test_end_to_end_attempt_binding_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "admission": load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"), "physical": load(PHYSICAL)}
    artifact, path, replacement = END_TO_END_ATTEMPT_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_end_to_end_attempt_binding(values["issuer"], values["admission"], values["physical"])

CAS_LOST_RESPONSE_MUTATIONS = {
    "idempotency_key_uses_field_names_only": ("local_current_attempt_fence.idempotency_key_derivation", "tuple of field names"),
    "idempotency_key_ignores_field_values": ("local_current_attempt_fence.field_names_only_key_allowed", True),
    "all_operations_share_one_current_attempt_slot": ("local_current_attempt_fence.global_singleton_current_attempt_registry", True),
    "different_operation_reuses_existing_attempt": ("local_current_attempt_fence.different_operation_slots_isolated", False),
    "different_account_reuses_existing_attempt": ("local_current_attempt_fence.different_account_may_converge", True),
    "changed_request_fingerprint_treated_exact_same": ("local_current_attempt_fence.exact_idempotency_key_fields", ["logical_operation_id"]),
    "changed_entitlement_generation_treated_exact_same": ("local_current_attempt_fence.exact_idempotency_key_fields", ["canonical_genesis_request_fingerprint_sha256"]),
    "changed_requester_key_version_treated_exact_same": ("local_current_attempt_fence.exact_idempotency_key_fields", ["entitlement_generation"]),
    "changed_claimant_key_version_treated_exact_same": ("local_current_attempt_fence.exact_idempotency_key_fields", ["requester_key_version"]),
    "changed_credentials_create_second_current_before_reconciliation": ("local_current_attempt_fence.changed_exact_authorization_may_bypass_current_fence", True),
    "process_local_dictionary_singleton_claimed_as_authority": ("local_current_attempt_fence.process_local_dictionary_is_authority", True),
    "candidate_loser_remains_externally_sendable": ("local_current_attempt_fence.candidate_loser_externally_sendable", True),
    "lost_response_ignores_attempt_id": ("lost_response.recovery_requires_fields", ["logical_operation_id"]),
    "lost_response_different_attempt_id_returns_old_proof": ("lost_response.different_attempt_id", "RETURN OLD PROOF"),
    "bound_recovery_does_not_compare_bound_attempt_id": ("lost_response.requested_attempt_id_must_equal", "no comparison"),
    "semantic_equivalence_substitutes_attempt_identity": ("lost_response.semantic_equivalence_substitutes_attempt_identity", True),
    "current_pointer_attempt_id_substitutes_bound_attempt_id": ("lost_response.current_pointer_substitutes_bound_attempt_id", True),
}


def validate_cas_and_lost_response(contract: dict) -> None:
    fence = contract["local_current_attempt_fence"]
    assert fence["operation_slot_key_fields"] == ["environment", "trust_domain", "logical_operation_id"]
    assert fence["idempotency_key_derivation"].startswith("ordered tuple of ACTUAL REQUEST VALUES")
    assert fence["global_singleton_current_attempt_registry"] is False
    assert fence["field_names_only_key_allowed"] is False
    assert fence["process_local_dictionary_is_authority"] is False
    assert fence["different_operation_slots_isolated"] is True
    assert fence["different_account_may_converge"] is False
    assert fence["changed_exact_authorization_may_bypass_current_fence"] is False
    assert fence["candidate_loser_externally_sendable"] is False
    assert fence["changed_current_requires"].startswith("RECONCILIATION_REQUIRED")
    expected = {
        "environment", "trust_domain", "logical_operation_id", "account_id",
        "canonical_genesis_request_fingerprint_sha256", "bootstrap_entitlement_id",
        "entitlement_generation", "requester_principal_id", "requester_credential_role",
        "requester_key_id", "requester_key_version", "provisioning_principal_id",
        "claimant_key_id", "claimant_key_version", "initial_binding_digest_sha256",
    }
    assert set(fence["exact_idempotency_key_fields"]) == expected

    lost = contract["lost_response"]
    retry = set(contract["issuer_bound_attempt_linkage"]["exact_retry_fields"])
    required = set(lost["recovery_requires_fields"])
    assert retry <= required
    assert {"issuance_attempt_id", "initial_binding_reference", "initial_binding_digest_sha256", "signed_root_proof_issuance_request_identity"} <= required
    assert lost["requested_attempt_id_must_equal"] == "issuer_history.bound_issuance_attempt_id"
    assert lost["same_attempt_id_success"].startswith("return byte-identical")
    assert lost["different_attempt_id"].startswith("FAIL_CLOSED / NOT_EXACT_RETRY")
    assert lost["semantic_equivalence_substitutes_attempt_identity"] is False
    assert lost["current_pointer_substitutes_bound_attempt_id"] is False
    assert lost["enforcement_boundary"].startswith("issuer recovery boundary")


def recover_lost_response(contract: dict, requested_attempt_id: str, bound_attempt_id: str) -> str:
    validate_cas_and_lost_response(contract)
    if requested_attempt_id != bound_attempt_id:
        return "FAIL_CLOSED / NOT_EXACT_RETRY"
    return "BYTE_IDENTICAL_PROOF_AND_HISTORY"


def test_lost_response_same_attempt_succeeds_and_different_attempt_rejects() -> None:
    contract = load()
    attempt_a = "rpa_018f3e70-7b5a-7c21-8b9a-0123456789ab"
    attempt_b = "rpa_018f3e70-7b6a-7c21-8b9a-0123456789ab"
    assert recover_lost_response(contract, attempt_a, attempt_a) == "BYTE_IDENTICAL_PROOF_AND_HISTORY"
    assert recover_lost_response(contract, attempt_b, attempt_a) == "FAIL_CLOSED / NOT_EXACT_RETRY"


def test_cas_and_lost_response_contract() -> None:
    contract = load()
    validate_cas_and_lost_response(contract)
    assert set(contract["cas_and_lost_response_redteam_mutations"]) == set(CAS_LOST_RESPONSE_MUTATIONS)


@pytest.mark.parametrize("name", sorted(CAS_LOST_RESPONSE_MUTATIONS))
def test_cas_and_lost_response_mutations_rejected(name: str) -> None:
    contract = load()
    path, replacement = CAS_LOST_RESPONSE_MUTATIONS[name]
    changed = mutate_named(contract, path, replacement)
    assert changed != contract
    with pytest.raises(AssertionError):
        validate_cas_and_lost_response(changed)

RESERVATION_SEQUENCING_MUTATIONS = {
    "process_supplies_candidate_rpa_id_to_authority": ("root_proof_issuance_attempt_reservation.process_proposes_candidate_rpa_id", True),
    "rpa_id_not_durable_before_requester_signature": ("root_proof_issuance_attempt_reservation.purpose", "sign then persist ID"),
    "rpa_id_not_durable_before_claimant_signature": ("signed_attempt_finalization.ordered_steps", ["obtain claimant authorization signature", "load reservation"]),
    "immutable_attempt_persist_required_in_same_tx_as_first_id_mint": ("root_proof_issuance_attempt_reservation.NO_CURRENT_atomic_transaction", ["persist immutable attempt"]),
    "local_transaction_spans_external_claimant_signing": ("root_proof_issuance_attempt_reservation.transaction_excludes", ["external issuer calls"]),
    "crash_after_id_mint_before_signature_remints_new_id": ("reservation_crash_recovery.reservation_durable_no_requester_signature", "remint"),
    "crash_after_requester_signature_before_claimant_remints_new_id": ("reservation_crash_recovery.requester_signature_claimant_missing", "remint"),
    "crash_after_both_signatures_before_attempt_persist_remints_new_id": ("reservation_crash_recovery.both_signatures_attempt_not_persisted", "remint"),
    "losing_process_can_sign_non_authoritative_rpa_id": ("signed_attempt_finalization.signing_non_reserved_id_forbidden", False),
    "losing_process_candidate_digest_exists_before_authority_selection": ("root_proof_issuance_attempt_reservation.immutable_attempt_digest_present", True),
    "reservation_rpa_id_differs_from_signed_request_rpa_id": ("signed_attempt_finalization.completion_equality_fields", ["requester_key_version"]),
    "reservation_requester_key_version_differs_from_completed_attempt": ("signed_attempt_finalization.completion_equality_fields", ["issuance_attempt_id"]),
    "reservation_claimant_key_version_differs_from_completed_attempt": ("signed_attempt_finalization.completion_equality_fields", ["requester_key_version"]),
    "reservation_entitlement_generation_differs_from_completed_attempt": ("signed_attempt_finalization.completion_equality_fields", ["claimant_key_version"]),
    "external_send_allowed_from_reserved_only_state": ("signed_attempt_finalization.external_send_eligible_only_after", "RESERVED_AWAITING_SIGNATURES"),
    "placeholder_attempt_digest_created_at_reservation": ("root_proof_issuance_attempt_reservation.placeholder_digest_forbidden", False),
    "pre_send_reservation_superseded_without_proof_send_was_impossible": ("pre_send_reservation_supersession.requires", []),
    "current_pointer_requires_completed_digest_before_it_can_reference_reserved_id": ("root_proof_issuance_attempt_reservation.current_pointer_may_reference_reserved_without_digest", False),
}


def validate_reservation_sequencing(issuer: dict, physical: dict) -> None:
    reservation = issuer["root_proof_issuance_attempt_reservation"]
    assert reservation["object"] == "RootProofIssuanceAttemptReservationV1"
    assert reservation["owner"] == "CryptoHunterAccountAuthority local AccountGenesis authority boundary"
    assert reservation["purpose"].startswith("durably establish authority-owned issuance_attempt_id before any")
    required = {
        "schema_version", "environment", "trust_domain", "logical_operation_id", "account_id",
        "canonical_genesis_request_fingerprint_sha256", "initial_binding_reference", "initial_binding_digest_sha256",
        "bootstrap_entitlement_id", "entitlement_generation", "requester_principal_id", "requester_credential_role",
        "requester_key_id", "requester_key_version", "provisioning_principal_id", "claimant_key_id",
        "claimant_key_version", "exact_idempotency_key", "issuance_attempt_id", "reservation_state",
    }
    assert set(reservation["fields"]) == required
    assert reservation["signatures_present"] is reservation["immutable_attempt_digest_present"] is False
    assert reservation["placeholder_digest_forbidden"] is True
    assert reservation["API"] == "reserve_or_resolve_attempt_id(exact_authorization_tuple)"
    assert reservation["process_proposes_candidate_rpa_id"] is False
    assert reservation["authority_id_factory"].startswith("inside durable CHA reservation transaction")
    assert reservation["exact_retry"].startswith("returns same durable authority-issued issuance_attempt_id")
    assert reservation["current_pointer_may_reference_reserved_without_digest"] is True
    assert reservation["NO_CURRENT_atomic_transaction"] == ["derive per-operation slot from actual values", "derive exact idempotency identity", "verify no incompatible current/reservation", "mint authority-owned rpa_id", "persist RootProofIssuanceAttemptReservationV1", "install current_root_proof_issuance_attempt_id"]
    assert set(reservation["transaction_excludes"]) == {"requester signing", "claimant signing", "remote calls", "non-local HSM/network interaction", "external issuer calls"}

    finalization = issuer["signed_attempt_finalization"]
    assert finalization["model"] == "SIGNED_IMMUTABLE_ATTEMPT_FINALIZATION"
    assert finalization["ordered_steps"][0] == "load exact durable reserved rpa_id"
    assert finalization["ordered_steps"][-1] == "enable external issuer send"
    equality = set(finalization["completion_equality_fields"])
    assert {"issuance_attempt_id", "requester_key_version", "claimant_key_version", "entitlement_generation", "initial_binding_digest_sha256"} <= equality
    assert finalization["immutable_digest_defined_at"].startswith("after exact request and both signatures")
    assert finalization["external_send_eligible_only_after"] == "IMMUTABLE_ATTEMPT_DURABLE"
    assert finalization["signing_non_reserved_id_forbidden"] is True

    crash = issuer["reservation_crash_recovery"]
    assert crash["reservation_durable_no_requester_signature"].startswith("recover same reserved rpa_id")
    assert crash["requester_signature_claimant_missing"].startswith("recover same reserved rpa_id")
    assert crash["both_signatures_attempt_not_persisted"].startswith("reservation remains authoritative")
    supersession = issuer["pre_send_reservation_supersession"]
    assert supersession["distinct_from_post_send_replacement"] is True
    assert set(supersession["requires"]) == {"proof no external send was possible", "proof no completed externally-sendable attempt exists", "proof no issuer BOUND decision can exist"}
    assert supersession["old_history_retained"] is supersession["new_rpa_id_required"] is True
    assert supersession["unresolved_reservation_allows_hidden_new_id"] is False

    parity = physical["root_proof_issuance_attempt_recovery"]
    assert parity["two_step_local_model"] == ["durable authority-owned RootProofIssuanceAttemptReservationV1", "exact requester and claimant signing for reserved rpa_id", "durable completed immutable RootProofIssuanceAttemptV1", "external issuer send"]
    assert parity["authority_id_reservation_precedes_signing"] is True
    assert parity["immutable_attempt_persistence_precedes_send"] is True
    assert parity["macro_phase_change"] is False
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"


def test_reservation_sequencing_contract() -> None:
    issuer = load()
    physical = load(PHYSICAL)
    validate_reservation_sequencing(issuer, physical)
    assert set(issuer["reservation_sequencing_redteam_mutations"]) == set(RESERVATION_SEQUENCING_MUTATIONS)


@pytest.mark.parametrize("name", sorted(RESERVATION_SEQUENCING_MUTATIONS))
def test_reservation_sequencing_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "physical": load(PHYSICAL)}
    artifact = "issuer"
    path, replacement = RESERVATION_SEQUENCING_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_reservation_sequencing(values["issuer"], values["physical"])

TWO_STEP_RECONCILIATION_MUTATIONS = {
    "exact_reserved_retry_requires_immutable_digest": ("issuer", "local_current_attempt_fence.EXACT_CURRENT_RESERVED", "digest required"),
    "placeholder_digest_returned_for_reserved_retry": ("issuer", "current_attempt_resolution.reserved_semantics", "return zero digest"),
    "reserved_current_lookup_treated_as_corrupt_because_digest_missing": ("issuer", "current_attempt_resolution.missing_digest_field", "NOT_YET_DEFINED is corrupt"),
    "reservation_and_immutable_finalization_forced_same_transaction": ("issuer", "immutable_finalization_atomicity.transaction", "same reservation transaction"),
    "physical_current_fence_claims_immutable_creation_same_tx": ("physical", "root_proof_issuance_attempt_recovery.reservation_atomicity", "fence + immutable persist same transaction"),
    "initial_race_claims_digest_exists_before_signing": ("issuer", "attempt_creation_concurrency_matrix", []),
    "validator_requires_digest_for_RESERVED_AWAITING_SIGNATURES": ("issuer", "current_attempt_resolution.digest_status_union", ["DEFINED"]),
    "validator_requires_old_one_step_current_attempt_atomicity": ("physical", "root_proof_issuance_attempt_recovery.immutable_finalization_atomicity", "same first transaction"),
    "two_finalizers_same_rpa_unequal_payload_both_accept": ("issuer", "immutable_finalization_atomicity.same_rpa_unequal_identity", "ACCEPT"),
    "stale_superseded_reservation_can_finalize": ("issuer", "immutable_finalization_atomicity.stale_superseded_may_finalize", True),
    "exact_duplicate_finalization_mints_new_digest_identity": ("issuer", "immutable_finalization_atomicity.same_rpa_identical_identity", "mint new digest"),
    "pre_send_superseded_reservation_required_to_have_digest": ("issuer", "pre_send_reservation_supersession.historical_digest_semantics", "digest required"),
}


def validate_two_step_reconciliation(issuer: dict, admission: dict, physical: dict) -> None:
    fence = issuer["local_current_attempt_fence"]
    assert "EXACT_CURRENT_EXISTS" not in fence
    assert "NOT_YET_DEFINED" in fence["EXACT_CURRENT_RESERVED"]
    assert "same DEFINED immutable digest" in fence["EXACT_CURRENT_IMMUTABLE"]
    resolution = issuer["current_attempt_resolution"]
    assert resolution["object"] == "CurrentRootProofIssuanceAttemptResolutionV1"
    assert resolution["fields"] == ["issuance_attempt_id", "state", "immutable_attempt_digest_status"]
    assert resolution["digest_status_union"] == ["NOT_YET_DEFINED", "DEFINED(<lowercase 64-character SHA-256 hex>)"]
    assert resolution["reserved_semantics"].startswith("NOT_YET_DEFINED is authoritative and valid")
    assert set(resolution["forbidden_not_yet_defined_representations"]) == {"empty string", "zero digest", "all-zero digest", "synthetic placeholder hash", "ambiguous null"}
    assert resolution["missing_digest_field"] == "MALFORMED / FAIL_CLOSED; distinct from explicit NOT_YET_DEFINED"

    finalization = issuer["immutable_finalization_atomicity"]
    assert finalization["transaction"].startswith("later local authority transaction distinct from reservation transaction")
    assert finalization["signing_outside_transaction"] is True
    assert finalization["same_rpa_identical_identity"].startswith("idempotently converge")
    assert finalization["same_rpa_unequal_identity"] == "FAIL_CLOSED / CORRUPTION_OR_TAMPER"
    assert finalization["stale_superseded_may_finalize"] is False
    assert finalization["reservation_generation_fence_required"] is True

    matrix = issuer["attempt_creation_concurrency_matrix"]
    assert len(matrix) == 5
    assert matrix[0]["immutable_digest_status"] == "NOT_YET_DEFINED"
    assert matrix[0]["requester_signature_present"] is matrix[0]["claimant_signature_present"] is False
    assert matrix[0]["external_send_eligible"] is False
    assert matrix[1]["step"] == "SIGNED_FINALIZATION_RACE"
    assert matrix[1]["identical_identity"].startswith("one immutable digest durable")
    assert matrix[1]["unequal_identity"] == "FAIL_CLOSED / CORRUPTION_OR_TAMPER"

    outputs = issuer["reservation_crash_resolution_outputs"]
    assert outputs["after_reservation_before_signatures"]["digest_status"] == "NOT_YET_DEFINED"
    assert outputs["after_requester_before_claimant"]["digest_status"] == "NOT_YET_DEFINED"
    assert outputs["after_both_signatures_before_finalization"]["digest_status"].startswith("NOT_YET_DEFINED")
    assert outputs["after_immutable_finalization"]["digest_status"] == "DEFINED(same immutable digest)"
    assert issuer["pre_send_reservation_supersession"]["historical_digest_semantics"].startswith("superseded proven-unsendable reservation may permanently retain NOT_YET_DEFINED")

    recovery = physical["root_proof_issuance_attempt_recovery"]
    assert "current_attempt_fence_atomicity" not in recovery
    assert recovery["reservation_atomicity"].startswith("current fence + RootProofIssuanceAttemptReservationV1 creation/install")
    assert recovery["immutable_finalization_atomicity"].startswith("later local authority transaction")
    assert set(recovery["reservation_transaction_includes"].values()) == {False}
    assert all(recovery["ordering"].values())
    assert admission["validation_output"]["object"] == "RootProofAdmissionEvidenceV1"
    assert "winning_root_proof_issuance_attempt_digest_sha256" in admission["validation_output"]["fields"]
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"


def test_state_sensitive_reserved_and_immutable_exact_retry() -> None:
    contract = load()
    request = semantic_authorization_fixture()
    authority = SemanticCurrentAttemptAuthority(contract, deterministic_id_factory())
    attempt_id = authority.reserve_or_resolve(request)
    assert authority.reserve_or_resolve(request) == attempt_id
    reserved = authority.resolve_current(request)
    assert reserved["immutable_attempt_digest_status"] == "NOT_YET_DEFINED"
    assert authority.external_send_eligible(request) is False
    authority.sign_requester(request, attempt_id)
    authority.sign_claimant(request, attempt_id)
    authority.finalize_immutable(request, attempt_id, "a" * 64)
    assert authority.reserve_or_resolve(request) == attempt_id
    immutable = authority.resolve_current(request)
    assert immutable["immutable_attempt_digest_status"] == f"DEFINED({'a' * 64})"
    assert authority.external_send_eligible(request) is True


def test_immutable_finalization_race_and_superseded_history() -> None:
    contract = load()
    request = semantic_authorization_fixture()
    authority = SemanticCurrentAttemptAuthority(contract, deterministic_id_factory())
    attempt_id = authority.reserve_or_resolve(request)
    authority.sign_requester(request, attempt_id)
    authority.sign_claimant(request, attempt_id)
    authority.finalize_immutable(request, attempt_id, "a" * 64)
    authority.finalize_immutable(request, attempt_id, "a" * 64)
    with pytest.raises(AssertionError, match="FAIL_CLOSED"):
        authority.finalize_immutable(request, attempt_id, "b" * 64)

    old = {"attempt_id": "rpa_old", "state": "SUPERSEDED_PRE_SEND_PROVEN_UNSENDABLE", "digest": None}
    assert old["digest"] is None
    assert old["state"] == "SUPERSEDED_PRE_SEND_PROVEN_UNSENDABLE"


def test_two_step_reconciliation_contract() -> None:
    issuer = load()
    admission = load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json")
    physical = load(PHYSICAL)
    validate_two_step_reconciliation(issuer, admission, physical)
    assert set(issuer["two_step_reconciliation_redteam_mutations"]) == set(TWO_STEP_RECONCILIATION_MUTATIONS)


@pytest.mark.parametrize("name", sorted(TWO_STEP_RECONCILIATION_MUTATIONS))
def test_two_step_reconciliation_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "admission": load(DOCS / "m05_account_genesis_root_proof_admission_binding_contract.json"), "physical": load(PHYSICAL)}
    artifact, path, replacement = TWO_STEP_RECONCILIATION_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_two_step_reconciliation(values["issuer"], values["admission"], values["physical"])

REPLACEMENT_TWO_STEP_MUTATIONS = {
    "replacement_switch_persists_immutable_digest_in_first_transaction": ("issuer", "replacement_reservation_transition.immutable_attempt_present", True),
    "replacement_switch_requires_signed_request_before_new_rpa_reservation": ("issuer", "replacement_reservation_transition.signing_after_commit", False),
    "replacement_new_rpa_not_durable_before_requester_signature": ("issuer", "replacement_reservation_transition.requester_signature_present", True),
    "replacement_new_rpa_not_durable_before_claimant_signature": ("issuer", "replacement_reservation_transition.claimant_signature_present", True),
    "replacement_initial_digest_status_defined": ("issuer", "replacement_reservation_transition.immutable_attempt_digest_status", "DEFINED"),
    "replacement_transaction_includes_requester_signing": ("physical", "root_proof_issuance_attempt_recovery.replacement_reservation_transaction_includes.requester_signing", True),
    "replacement_transaction_includes_claimant_signing": ("physical", "root_proof_issuance_attempt_recovery.replacement_reservation_transaction_includes.claimant_signing", True),
    "replacement_transaction_includes_immutable_attempt_persistence": ("physical", "root_proof_issuance_attempt_recovery.replacement_reservation_transaction_includes.immutable_attempt_persistence", True),
    "replacement_transaction_includes_external_send": ("physical", "root_proof_issuance_attempt_recovery.replacement_reservation_transaction_includes.external_send", True),
    "concurrent_replacements_install_two_current_reservations": ("issuer", "replacement_concurrency.two_current_replacements_possible", True),
    "losing_replacement_reservation_can_be_signed": ("issuer", "replacement_concurrency.loser_may_sign", True),
    "old_superseded_attempt_can_finalize_after_replacement_switch": ("issuer", "immutable_finalization_atomicity.stale_superseded_may_finalize", True),
    "crash_after_replacement_reservation_remints_new_rpa": ("issuer", "replacement_crash_recovery.R2_REPLACEMENT_RESERVED_BEFORE_REQUESTER_SIGNATURE_CRASH", "remint"),
    "crash_between_replacement_signatures_remints_new_rpa": ("issuer", "replacement_crash_recovery.R3_REQUESTER_SIGNED_BEFORE_CLAIMANT_CRASH", "remint"),
    "replacement_mutates_old_immutable_attempt_credentials": ("issuer", "replacement_concurrency.old_attempt_credentials_mutated", True),
    "replacement_current_pointer_switch_occurs_only_after_digest_exists": ("issuer", "replacement_reservation_transition.current_switch_requires_digest", True),
}


def validate_replacement_two_step(issuer: dict, physical: dict) -> None:
    fence = issuer["local_current_attempt_fence"]
    assert "AUTHORITATIVELY_UNBOUND_REPLACEMENT" not in fence
    assert fence["AUTHORITATIVELY_UNBOUND_REPLACEMENT_RESERVATION"].startswith("after authenticated authoritative UNBOUND reconciliation")
    transition = issuer["replacement_reservation_transition"]
    assert transition["model"] == "REPLACEMENT_RESERVATION_TRANSITION"
    assert transition["commit_state"] == "RESERVED_AWAITING_SIGNATURES"
    assert transition["immutable_attempt_digest_status"] == "NOT_YET_DEFINED"
    assert transition["requester_signature_present"] is transition["claimant_signature_present"] is transition["immutable_attempt_present"] is transition["external_send_eligible"] is False
    assert transition["signing_after_commit"] is True
    assert transition["immutable_finalization_after_signatures"] == "reuse immutable_finalization_atomicity"
    assert transition["current_switch_requires_digest"] is False
    assert set(transition["old_history_never_rewritten"]) == {"old rpa_id", "old signed request", "old signatures", "old immutable digest if defined", "old send/outcome history", "authoritative reconciliation evidence"}

    crash = issuer["replacement_crash_recovery"]
    assert crash["R1_UNBOUND_PROVEN_BEFORE_TX_CRASH"].startswith("old remains authoritative current")
    assert crash["R2_REPLACEMENT_RESERVED_BEFORE_REQUESTER_SIGNATURE_CRASH"].startswith("new rpa_id current; NOT_YET_DEFINED; recover same ID")
    assert crash["R3_REQUESTER_SIGNED_BEFORE_CLAIMANT_CRASH"].startswith("same new rpa_id; NOT_YET_DEFINED")
    assert crash["R4_BOTH_SIGNATURES_BEFORE_FINALIZATION_CRASH"].startswith("same replacement reservation authoritative current")
    assert crash["R5_FINALIZATION_COMMITTED_BEFORE_SEND_CRASH"].startswith("same new rpa_id and same DEFINED")
    assert crash["R6_OLD_STALE_FINALIZATION"].startswith("FAIL_CLOSED")
    concurrency = issuer["replacement_concurrency"]
    assert concurrency["two_current_replacements_possible"] is concurrency["loser_may_sign"] is concurrency["loser_externally_sendable"] is False
    assert concurrency["new_credentials_bound_to_new_reservation"] is True
    assert concurrency["old_attempt_credentials_mutated"] is False

    matrix = issuer["attempt_creation_concurrency_matrix"]
    reserved = next(row for row in matrix if row["step"] == "AUTHORITATIVE_UNBOUND_REPLACEMENT_RESERVATION")
    finalized = next(row for row in matrix if row["step"] == "REPLACEMENT_SIGNED_FINALIZATION")
    assert reserved["immutable_digest_status"] == "NOT_YET_DEFINED"
    assert reserved["requester_signature_present"] is reserved["claimant_signature_present"] is reserved["external_send_eligible"] is False
    assert reserved["two_current_replacements"] is False
    assert finalized["immutable_digest_status"] == "DEFINED"
    assert finalized["external_send_eligible"] is True

    recovery = physical["root_proof_issuance_attempt_recovery"]
    assert "replacement_switch_atomicity" not in recovery
    assert recovery["replacement_reservation_switch_atomicity"].startswith("old supersession + durable exact authenticated authoritative-UNBOUND evidence")
    assert recovery["replacement_immutable_finalization_atomicity"].startswith("later reuse immutable_finalization_atomicity")
    included = recovery["replacement_reservation_transaction_includes"]
    assert included == {"old_supersession": True, "new_reservation_persistence": True, "current_pointer_switch": True, "requester_signing": False, "claimant_signing": False, "immutable_attempt_persistence": False, "external_send": False, "authenticated_unbound_evidence_binding": True}
    assert issuer["immutable_finalization_atomicity"]["stale_superseded_may_finalize"] is False
    assert physical["result"]["selected_protocol"] == "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT"


def test_replacement_reservation_finalization_and_race() -> None:
    issuer = load()
    request = semantic_authorization_fixture()
    authority = SemanticCurrentAttemptAuthority(issuer, deterministic_id_factory())
    old_id = authority.reserve_or_resolve(request)
    replacement_request = semantic_authorization_fixture(requester_key_version=2, claimant_key_version=2)
    evidence = reconciliation_evidence_fixture(request, old_id)
    new_id = authority.replace_after_authoritative_unbound(request, replacement_request, old_id, evidence)
    competing_process_result = authority.replace_after_authoritative_unbound(
        request, replacement_request, old_id, evidence
    )
    assert new_id != old_id
    assert competing_process_result == new_id
    assert len(authority.current_attempt_by_operation) == 1
    assert authority.reserve_or_resolve(replacement_request) == new_id
    assert authority.resolve_current(replacement_request)["immutable_attempt_digest_status"] == "NOT_YET_DEFINED"
    assert authority.external_send_eligible(replacement_request) is False
    old_history = authority.attempt_history_by_operation[authority.operation_slot(request)]
    assert len(old_history) == 1
    assert old_history[0]["attempt_id"] == old_id
    assert old_history[0]["state"] == "SUPERSEDED_AFTER_AUTHORITATIVE_UNBOUND_RECONCILIATION"
    assert old_history[0]["replacement_authorization_evidence_digest_sha256"] == "c" * 64
    with pytest.raises(AssertionError):
        authority.finalize_immutable(request, old_id, "a" * 64)
    authority.sign_requester(replacement_request, new_id)
    authority.sign_claimant(replacement_request, new_id)
    authority.finalize_immutable(replacement_request, new_id, "b" * 64)
    assert authority.resolve_current(replacement_request)["immutable_attempt_digest_status"] == f"DEFINED({'b' * 64})"
    assert authority.external_send_eligible(replacement_request) is True


def test_replacement_two_step_contract() -> None:
    issuer = load()
    physical = load(PHYSICAL)
    validate_replacement_two_step(issuer, physical)
    assert set(issuer["replacement_two_step_redteam_mutations"]) == set(REPLACEMENT_TWO_STEP_MUTATIONS)


@pytest.mark.parametrize("name", sorted(REPLACEMENT_TWO_STEP_MUTATIONS))
def test_replacement_two_step_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "physical": load(PHYSICAL)}
    artifact, path, replacement = REPLACEMENT_TWO_STEP_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_replacement_two_step(values["issuer"], values["physical"])


RECONCILIATION_EVIDENCE_MUTATIONS = {
    "replacement_without_reconciliation_evidence": ("issuer", "replacement_reservation_transition.evidence_required", False),
    "caller_boolean_unbound_authorizes_replacement": ("issuer", "replacement_reservation_transition.caller_assertion_sufficient", True),
    "method_name_unbound_is_treated_as_evidence": ("issuer", "replacement_reservation_transition.evidence_object", "method name"),
    "timeout_treated_as_unbound": ("issuer", "reconciliation_evidence_contract.non_authoritative_outcomes", ["connection error"]),
    "not_found_treated_as_unbound": ("issuer", "reconciliation_evidence_contract.non_authoritative_outcomes", ["timeout"]),
    "missing_history_treated_as_unbound": ("issuer", "reconciliation_evidence_contract.non_authoritative_outcomes", ["timeout", "NOT_FOUND"]),
    "unauthenticated_unbound_snapshot_accepted": ("issuer", "reconciliation_evidence_contract.authentication_required", False),
    "reconciliation_wrong_old_rpa_id_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["environment"]),
    "reconciliation_wrong_entitlement_generation_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["old_issuance_attempt_id"]),
    "reconciliation_wrong_operation_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["old_issuance_attempt_id", "entitlement_generation"]),
    "reconciliation_wrong_account_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["logical_operation_id"]),
    "reconciliation_wrong_request_fingerprint_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["account_id"]),
    "reconciliation_wrong_initial_binding_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["canonical_genesis_request_fingerprint_sha256"]),
    "reconciliation_wrong_environment_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["trust_domain"]),
    "reconciliation_wrong_trust_domain_accepted": ("issuer", "replacement_reservation_transition.evidence_exact_binding_fields", ["environment"]),
    "stale_unbound_evidence_after_later_bound_accepted": ("issuer", "reconciliation_evidence_contract.stale_after_later_bound_authorizes", True),
    "replacement_commit_omits_reconciliation_evidence_reference": ("issuer", "replacement_reservation_transition.durable_replacement_binding_fields", ["old_issuance_attempt_id", "new_issuance_attempt_id"]),
    "crash_reconstructs_reconciliation_evidence_from_current_lookup": ("issuer", "reconciliation_evidence_contract.forbidden_reconstruction_sources", ["caller"]),
    "concurrent_replacement_accepts_incompatible_evidence": ("issuer", "replacement_concurrency.incompatible_evidence_after_switch", "ACCEPT"),
    "superseded_history_loses_unbound_evidence_provenance": ("issuer", "reconciliation_evidence_contract.durable_relation", "NONE"),
}


def validate_reconciliation_evidence(issuer: dict, physical: dict) -> None:
    evidence = issuer["reconciliation_evidence_contract"]
    assert evidence["object"] == "RootProofIssuanceReconciliationEvidenceV1"
    assert evidence["required_outcome"] == "AUTHORITATIVELY_UNBOUND"
    assert evidence["positive_result"] is evidence["absence_never_proves_unbound"] is evidence["authentication_required"] is True
    assert set(evidence["non_authoritative_outcomes"]) == {"timeout", "connection error", "missing response", "NOT_FOUND", "absent local proof", "absent local receipt", "empty history", "caller boolean/string UNBOUND", "process memory", "cached non-authoritative projection", "stale issuer snapshot"}
    assert evidence["non_authoritative_result"] == "OUTCOME_UNKNOWN / FAIL_CLOSED"
    required = {"schema_version", "environment", "trust_domain", "issuer_authority_identity", "issuer_registry_identity", "bootstrap_entitlement_id", "entitlement_generation", "logical_operation_id", "account_id", "canonical_genesis_request_fingerprint_sha256", "old_issuance_attempt_id", "initial_binding_reference", "initial_binding_digest_sha256", "outcome", "authoritative_state_identity", "authoritative_state_revision", "authority_authenticated_evidence_reference", "authority_authenticated_evidence_digest_sha256", "verification_profile_version"}
    assert set(evidence["required_fields"]) == required
    assert evidence["wall_clock_establishes_ordering"] is False
    assert evidence["anti_staleness_model"] == "CURRENT_AUTHORITY_REVISION_AND_RETAINED_HISTORY_FENCE"
    assert evidence["stale_after_later_bound_authorizes"] is False
    assert set(evidence["forbidden_reconstruction_sources"]) == {"caller", "process memory", "current issuer lookup", "current credentials"}
    assert evidence["durable_relation"] == "old_issuance_attempt_id --superseded_by_authoritative_unbound_evidence--> new_issuance_attempt_id"
    transition = issuer["replacement_reservation_transition"]
    assert transition["evidence_required"] is True
    assert transition["caller_assertion_sufficient"] is False
    assert transition["evidence_object"] == evidence["object"]
    exact = {"environment", "trust_domain", "bootstrap_entitlement_id", "entitlement_generation", "logical_operation_id", "account_id", "canonical_genesis_request_fingerprint_sha256", "old_issuance_attempt_id", "initial_binding_reference", "initial_binding_digest_sha256"}
    assert set(transition["evidence_exact_binding_fields"]) == exact
    assert set(transition["durable_replacement_binding_fields"]) == {"old_issuance_attempt_id", "replacement_authorization_evidence_reference", "replacement_authorization_evidence_digest_sha256", "new_issuance_attempt_id"}
    assert transition["mismatch_result"].startswith("FAIL_CLOSED")
    concurrency = issuer["replacement_concurrency"]
    assert concurrency["same_evidence_converges"] is concurrency["evidence_identity_must_match_current_replacement"] is True
    assert concurrency["incompatible_evidence_after_switch"].startswith("FAIL_CLOSED")
    recovery = physical["root_proof_issuance_attempt_recovery"]
    parity = recovery["replacement_reconciliation_evidence"]
    assert parity["object"] == evidence["object"]
    assert parity["obtained_before_local_switch"] is parity["durably_bound_in_local_switch"] is True
    assert parity["distributed_transaction_with_issuer_claimed"] is False
    assert set(parity["bound_relation"]) == set(transition["durable_replacement_binding_fields"])
    assert recovery["replacement_reservation_transaction_includes"]["authenticated_unbound_evidence_binding"] is True


def test_reconciliation_evidence_state_machine() -> None:
    issuer = load()
    request = semantic_authorization_fixture()
    replacement = semantic_authorization_fixture(requester_key_version=2)
    authority = SemanticCurrentAttemptAuthority(issuer, deterministic_id_factory())
    old_id = authority.reserve_or_resolve(request)
    valid = reconciliation_evidence_fixture(request, old_id)
    before = deepcopy(authority.current_attempt_by_operation)
    invalid_evidence = [
        None,
        reconciliation_evidence_fixture(request, old_id, outcome="OUTCOME_UNKNOWN"),
        reconciliation_evidence_fixture(request, "rpa_wrong"),
        reconciliation_evidence_fixture(request, old_id, authenticated=False),
        reconciliation_evidence_fixture(request, old_id, entitlement_generation=2),
        reconciliation_evidence_fixture(request, old_id, logical_operation_id="ago_wrong"),
        reconciliation_evidence_fixture(request, old_id, account_id="acct_wrong"),
        reconciliation_evidence_fixture(request, old_id, canonical_genesis_request_fingerprint_sha256="ff" * 32),
        reconciliation_evidence_fixture(request, old_id, initial_binding_digest_sha256="ee" * 32),
        reconciliation_evidence_fixture(request, old_id, environment="PROD"),
        reconciliation_evidence_fixture(request, old_id, trust_domain="td_wrong"),
    ]
    for invalid in invalid_evidence:
        with pytest.raises(AssertionError):
            authority.replace_after_authoritative_unbound(request, replacement, old_id, invalid)
        assert authority.current_attempt_by_operation == before
    stale = reconciliation_evidence_fixture(request, old_id, authoritative_state_revision=6)
    with pytest.raises(AssertionError, match="STALE_EVIDENCE"):
        authority.replace_after_authoritative_unbound(request, replacement, old_id, stale)
    authority.bound_attempt_ids.add(old_id)
    with pytest.raises(AssertionError, match="LATER_BOUND"):
        authority.replace_after_authoritative_unbound(request, replacement, old_id, valid)
    authority.bound_attempt_ids.clear()
    new_id = authority.replace_after_authoritative_unbound(request, replacement, old_id, valid)
    assert authority.replace_after_authoritative_unbound(request, replacement, old_id, valid) == new_id
    incompatible = reconciliation_evidence_fixture(request, old_id, authority_authenticated_evidence_digest_sha256="d" * 64)
    assert authority.replace_after_authoritative_unbound(request, replacement, old_id, incompatible) == "RECONCILIATION_REQUIRED"
    current = authority.current_attempt_by_operation[authority.operation_slot(request)]
    history = authority.attempt_history_by_operation[authority.operation_slot(request)][0]
    assert (history["attempt_id"], history["replacement_authorization_evidence_digest_sha256"], current["attempt_id"]) == (old_id, "c" * 64, new_id)


def test_reconciliation_evidence_contract_and_mutations() -> None:
    issuer, physical = load(), load(PHYSICAL)
    validate_reconciliation_evidence(issuer, physical)
    assert set(issuer["reconciliation_evidence_redteam_mutations"]) == set(RECONCILIATION_EVIDENCE_MUTATIONS)


@pytest.mark.parametrize("name", sorted(RECONCILIATION_EVIDENCE_MUTATIONS))
def test_reconciliation_evidence_mutations_rejected(name: str) -> None:
    values = {"issuer": load(), "physical": load(PHYSICAL)}
    artifact, path, replacement = RECONCILIATION_EVIDENCE_MUTATIONS[name]
    original = values[artifact]
    values[artifact] = mutate_named(original, path, replacement)
    assert values[artifact] != original
    with pytest.raises(AssertionError):
        validate_reconciliation_evidence(values["issuer"], values["physical"])
