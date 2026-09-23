"""Wykonywalny kontrakt readiness produkcyjnego FreshnessAuthority."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
STEM = "m05_account_genesis_freshness_authority_implementation_readiness_contract"
MACHINE = DOCS / f"{STEM}.json"
MARKDOWN = DOCS / f"{STEM}.md"
JCS = "RFC 8785 JSON Canonicalization Scheme (JCS), restricted to the contract schema"
SIGNATURE = "base64url without padding over exactly 64 signature bytes"
DIGEST = "lowercase hexadecimal, exactly 64 ASCII characters"


def load(name: str | None = None) -> dict:
    return json.loads((DOCS / name if name else MACHINE).read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return (
        "# M0.5 AccountGenesis FreshnessAuthority production substrate / "
        "implementation readiness\n\nTen plik jest deterministyczną, kompletną "
        f"projekcją `{STEM}.json`. JSON jest źródłem prawdy.\n\n"
        f"```json\n{body}```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == (
        "M05_ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_IMPLEMENTATION_READINESS_CONTRACT"
    )
    assert value["iteration"] == "DESIGN / RECONCILIATION ONLY"
    assert value["readiness_result"] == (
        "FRESHNESS_AUTHORITY_IMPLEMENTATION_READINESS_CAN_BE_FROZEN"
    )
    provenance = value["provenance"]
    assert provenance == {
        "classification": "UNKNOWN",
        "finding_scope": "CURRENT_TREE_ONLY",
        "formal_project_advancement": "WITHHELD",
        "reviewed_SHA_supplied": "NOT_SUPPLIED",
        "design_freeze_is_formal_advancement": False,
    }

    scope = value["implementation_scope"]
    assert scope["FreshnessAuthority_implemented"] is False
    assert scope["CryptoHunterAccountAuthority_implemented"] is False
    assert scope["concrete_production_provider_selected"] is False
    assert scope["FreshnessAuthority_implementation_allowed_after_iteration"] is True
    assert scope["CHA_implementation_allowed_after_iteration"] is False

    assert value["frozen_inputs"]["TEST_PRODUCTION_domains_distinct"] is True

    cas = value["production_CAS_backend_contract"]
    assert cas["atomic_effect"].startswith("one indivisible durable commit")
    assert cas["winner_cardinality"] == "exactly one winner per exact predecessor"
    assert set(cas["CAS_preconditions"]) == {
        "exact predecessor generation",
        "exact predecessor digest",
        "exact complete predecessor head set",
        "exact authority domain and authority_id",
    }
    assert {"multi-process", "crash/restart durable"} <= set(cas["correctness"])
    assert {"read-then-write", "process mutex"} <= set(cas["forbidden_substitutes"])

    history = value["retained_authoritative_history"]
    assert history["current_head_alone_sufficient"] is False
    assert history["negative_lookup_is_non_acceptance_proof"] is False
    assert "exact immediate successor for old predecessor" in history["lookups"]
    assert (
        "cryptographically linked to currently trusted authority root/state"
        in (history["properties"])
    )

    receipt = value["finalization_receipt_durability"]
    assert receipt["authority_owner"] == "FreshnessAuthority"
    assert receipt["memory_only_allowed"] is False
    assert receipt["commit_rule"].startswith("receipt record and exact acceptance decision")
    assert set(receipt["exact_binding"]) >= {
        "accepted full-document digest",
        "finalization_request_id",
        "decision sequence/id",
    }

    canonical = value["canonical_representation"]
    assert canonical["serialization"].startswith("RFC 8785")
    assert canonical["text_encoding"] == "UTF-8, no BOM"
    assert canonical["bytes"] == "base64url without padding"
    assert canonical["integers"].startswith("JSON safe integers only")
    assert canonical["object_order"].startswith("JCS lexicographic")
    assert canonical["cross_process_rule"].startswith(
        "identical validated semantics MUST yield byte-identical"
    )
    assert canonical["invalid_Unicode_or_lone_surrogate"].startswith("REJECT")
    assert canonical["duplicate_object_keys"].startswith("REJECT")
    assert canonical["floats_in_cryptographic_preimages"] == "REJECT"
    assert "schema-declared comparator" in canonical["set_like_array_rule"]
    assert "padding and noncanonical forms REJECT" in canonical["base64url_validation"]
    assert "uppercase and mixed case REJECT" in canonical["digest_hex_validation"]

    crypto = value["cryptographic_profile"]
    assert crypto["digest"]["algorithm"] == "SHA-256"
    assert crypto["authentication"]["receipt"] == "Ed25519"
    assert crypto["authentication"]["proposer"] == "Ed25519"
    assert crypto["authentication"]["authoritative_document"] == "Ed25519"
    assert crypto["authentication"]["provider_selection"] == "NOT_FROZEN / NOT_SELECTED"
    assert crypto["authentication"]["verifier_selection"].startswith(
        "only authenticated lifecycle lineage"
    )
    assert crypto["authentication"]["signature_representation"] == SIGNATURE
    assert "same role as receipt" in crypto["authentication"]["authoritative_document_signing_role"]

    domains = value["domain_separation_profile"]
    assert domains["preimage_format"] == ("ASCII(exact_domain_literal) || 0x00 || canonical_bytes")
    literals = domains["exact_literals"]
    assert len(literals) == 6
    assert len(set(literals.values())) == len(literals)
    assert all(literal and literal.endswith(".v1") for literal in literals.values())
    assert domains["runtime_or_caller_configurable"] is False

    custody = value["key_custody"]
    assert custody["role_reuse_forbidden"] is True
    assert custody["caller_selected_authority_key"] is False
    assert len(custody["separate_roles"]) == 4
    assert "REVOKED may not sign" in custody["lifecycle"][2]

    lineage = value["proposer_trust_lineage"]
    assert lineage["status"] == "FROZEN_NOW / IMPLEMENTATION_NOT_AVAILABLE"
    assert lineage["bootstrap_owner"].startswith("offline Product Security")
    assert lineage["VERIFY_ONLY_history"] is True
    assert lineage["TOFU_allowed"] is False
    assert lineage["candidate_self_provision_allowed"] is False

    failures = value["failure_model"]
    assert failures["TIMEOUT_AFTER_SEND"] == "OUTCOME_UNKNOWN"
    assert failures["TRANSPORT_ERROR_AFTER_SEND"].startswith("OUTCOME_UNKNOWN")
    assert failures["outcome_unknown_rule"].startswith("never FAILED or SUCCESS")
    assert "quarantine" in failures["FORK_OR_TAMPER"]

    local = value["local_authority_adapter"]
    assert set(local["atomic_transitions"]) == {
        "INITIAL_BINDING",
        "PREPARED",
        "REPREPARE supersession",
        "FINAL_COMMIT",
    }
    assert local["M011_SQLiteStateStore_role"] == "PROJECTION_ONLY / NOT_AUTHORITY"
    assert local["process_local_lock_sufficient"] is False

    matrix = {row["capability"]: row for row in value["implementation_readiness_matrix"]}
    required = {
        "authoritative full-document CAS",
        "authoritative reread",
        "retained decision history",
        "immediate-successor historical proof",
        "durable finalization receipts",
        "canonical serialization",
        "digest algorithm",
        "receipt auth algorithm/provider",
        "proposer auth algorithm/provider",
        "freshness key custody",
        "freshness key lifecycle storage",
        "proposer trust lineage",
        "local transactional AccountGenesis adapter",
        "multi-process fencing",
        "TEST/PRODUCTION separation",
    }
    assert set(matrix) == required
    assert all(row["must_be_frozen_before_implementation"] == "YES" for row in matrix.values())
    assert all(row["frozen_now"].startswith("YES") for row in matrix.values())
    assert all(row["implementation_available"] == "NO" for row in matrix.values())

    assert value["actual_current_capability"]["production_CAS_backend"] == "NOT_AVAILABLE"
    assert value["downstream_status"]["production_M0.5"].startswith("BLOCKED")


def validate_cross_artifact(readiness: dict, freshness: dict, physical: dict) -> None:
    """Egzekwuj jedną concrete reprezentację w obu normatywnych artefaktach."""
    validate(readiness)
    canonical = readiness["canonical_representation"]
    crypto = readiness["cryptographic_profile"]
    domains = readiness["domain_separation_profile"]["exact_literals"]
    refinement = freshness["concrete_representation_refinement"]
    document = freshness["anchor_document_contract"]
    evidence = freshness["finalization_evidence"]
    receipt = evidence["authentication_boundary"]
    proposer = freshness["proposer_authentication_boundary"]

    assert refinement["canonical_profile"]["serialization"] == canonical["serialization"]
    for readiness_key, freshness_key in (
        ("invalid_Unicode_or_lone_surrogate", "invalid_Unicode_or_lone_surrogate"),
        ("duplicate_object_keys", "duplicate_object_keys"),
        ("set_like_array_rule", "set_like_array_rule"),
        ("base64url_validation", "base64url"),
        ("digest_hex_validation", "digest_hex"),
        ("floats_in_cryptographic_preimages", "floats_in_cryptographic_preimages"),
    ):
        assert refinement["canonical_profile"][freshness_key] == canonical[readiness_key]
    assert refinement["digest_algorithm"] == crypto["digest"]["algorithm"]
    assert refinement["digest_representation"] == crypto["digest"]["representation"]
    assert (
        refinement["signature_representation"]
        == crypto["authentication"]["signature_representation"]
    )
    assert document["canonical_byte_encoding"] == canonical["serialization"]
    assert document["hash_algorithm"] == crypto["digest"]["algorithm"]
    assert document["digest_representation"] == crypto["digest"]["representation"]
    assert (
        document["proposed_candidate_object"]["authentication_algorithm"]
        == crypto["authentication"]["proposer"]
    )
    assert proposer["authentication_algorithm"] == crypto["authentication"]["proposer"]
    assert receipt["cryptographic_algorithm"] == crypto["authentication"]["receipt"]
    assert (
        document["authoritative_document_authentication"]["algorithm"]
        == crypto["authentication"]["authoritative_document"]
    )
    assert (
        receipt["signature_representation"] == crypto["authentication"]["signature_representation"]
    )
    assert (
        proposer["signature_representation"] == crypto["authentication"]["signature_representation"]
    )
    assert (
        evidence["complete_semantic_head_digest_canonical_byte_encoding"]
        == canonical["serialization"]
    )
    assert evidence["complete_semantic_head_digest_hash_algorithm"] == crypto["digest"]["algorithm"]
    assert freshness["concrete_representation_refinement"]["domain_separators"] == domains
    assert refinement["domain_rules"]["runtime_or_caller_configurable"] is False
    assert document["document_digest_domain_literal"] == domains["freshness_document_digest"]
    assert (
        document["authoritative_document_authentication"]["domain_literal"]
        == domains["authoritative_document_authentication"]
    )
    assert proposer["domain_literal"] == domains["cha_proposer_authentication"]
    assert (
        evidence["complete_semantic_head_digest_domain_literal"]
        == domains["complete_semantic_head_set_digest"]
    )
    assert receipt["domain_literal"] == domains["finalization_receipt_authentication"]
    assert document["authoritative_document_authentication"]["proposer_key_role_allowed"] is False

    physical_result = physical["result"]
    physical_selection = physical["selected_or_blocked_protocol"]
    physical_status = physical["preserved_status"]
    assert physical_result["physical_protocol_frozen"] is True
    assert physical_result["selected_protocol"] == readiness["frozen_inputs"]["physical_protocol"]
    assert physical_selection["protocol_result"] == physical_result["primary_result"]
    assert (
        freshness["result"]["physical_persistence_unblocked"]
        == physical_result["physical_protocol_frozen"]
    )
    assert (
        freshness["preserved_status"]["physical persistence protocol"]
        == physical_status["physical persistence protocol"]
    )
    assert (
        freshness["preserved_status"]["physical persistence result"]
        == physical_selection["protocol_result"]
    )
    reason = freshness["result"]["reason"]
    assert "canonical serialization/key algorithm review" not in reason
    assert "physical persistence blockers" not in reason
    assert "no conforming CAS/history backend" in reason
    assert physical_result["implementation_and_backend_available"] is False
    assert freshness["implementation_allowed"]["FreshnessAuthority"] == "NO"
    assert freshness["implementation_allowed"]["CryptoHunterAccountAuthority"] == "NO"
    assert freshness["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"


MUTATIONS = {
    "read_compare_write_accepted_as_CAS": lambda x: x["production_CAS_backend_contract"][
        "forbidden_substitutes"
    ].remove("read-then-write"),
    "process_mutex_accepted_as_global_CAS": lambda x: x["production_CAS_backend_contract"][
        "forbidden_substitutes"
    ].remove("process mutex"),
    "two_successful_successors": lambda x: x["production_CAS_backend_contract"].__setitem__(
        "winner_cardinality", "two winners allowed"
    ),
    "current_head_only_sufficient_history": lambda x: x[
        "retained_authoritative_history"
    ].__setitem__("current_head_alone_sufficient", True),
    "negative_receipt_lookup_proves_non_acceptance": lambda x: x[
        "retained_authoritative_history"
    ].__setitem__("negative_lookup_is_non_acceptance_proof", True),
    "receipt_only_in_process_memory": lambda x: x["finalization_receipt_durability"].__setitem__(
        "memory_only_allowed", True
    ),
    "receipt_not_bound_to_exact_CAS_decision": lambda x: x["finalization_receipt_durability"][
        "exact_binding"
    ].remove("decision sequence/id"),
    "noncanonical_serialization_accepted": lambda x: x["canonical_representation"].__setitem__(
        "serialization", "implementation-defined JSON"
    ),
    "processes_hash_same_semantics_differently": lambda x: x[
        "canonical_representation"
    ].__setitem__("cross_process_rule", "process-local output allowed"),
    "unauthenticated_key_alias_selects_verifier": lambda x: x["cryptographic_profile"][
        "authentication"
    ].__setitem__("verifier_selection", "caller alias"),
    "proposer_key_reused_as_freshness_key": lambda x: x["key_custody"].__setitem__(
        "role_reuse_forbidden", False
    ),
    "Catalog_key_reused": lambda x: x["key_custody"]["separate_roles"].pop(),
    "TEST_key_or_state_accepted_in_PRODUCTION": lambda x: x["frozen_inputs"].__setitem__(
        "TEST_PRODUCTION_domains_distinct", False
    ),
    "TOFU_proposer_admitted": lambda x: x["proposer_trust_lineage"].__setitem__(
        "TOFU_allowed", True
    ),
    "candidate_self_provisions_proposer_trust": lambda x: x["proposer_trust_lineage"].__setitem__(
        "candidate_self_provision_allowed", True
    ),
    "revoked_key_finalizes_new_transition": lambda x: x["key_custody"]["lifecycle"].__setitem__(
        2, "REVOKED may sign"
    ),
    "M011_projection_promoted_to_authority": lambda x: x["local_authority_adapter"].__setitem__(
        "M011_SQLiteStateStore_role", "AUTHORITY"
    ),
    "local_adapter_process_lock_only": lambda x: x["local_authority_adapter"].__setitem__(
        "process_local_lock_sufficient", True
    ),
    "timeout_treated_as_failed": lambda x: x["failure_model"].__setitem__(
        "TIMEOUT_AFTER_SEND", "FAILED"
    ),
    "timeout_treated_as_success": lambda x: x["failure_model"].__setitem__(
        "TIMEOUT_AFTER_SEND", "SUCCESS"
    ),
}


CROSS_ARTIFACT_MUTATIONS = {
    "readiness_SHA256_freshness_NOT_FROZEN": lambda r, f: f["anchor_document_contract"].__setitem__(
        "hash_algorithm", "NOT_FROZEN"
    ),
    "readiness_Ed25519_proposer_freshness_NOT_FROZEN": lambda r, f: f[
        "proposer_authentication_boundary"
    ].__setitem__("authentication_algorithm", "NOT_FROZEN"),
    "readiness_Ed25519_receipt_freshness_NOT_FROZEN": lambda r, f: f["finalization_evidence"][
        "authentication_boundary"
    ].__setitem__("cryptographic_algorithm", "NOT_FROZEN"),
    "readiness_JCS_freshness_different_encoding": lambda r, f: f[
        "anchor_document_contract"
    ].__setitem__("canonical_byte_encoding", "implementation-defined JSON"),
    "document_digest_domain_changed": lambda r, f: f["anchor_document_contract"].__setitem__(
        "document_digest_domain_literal", "cryptohunter.changed.v1"
    ),
    "proposer_domain_reused_as_receipt_domain": lambda r, f: f[
        "proposer_authentication_boundary"
    ].__setitem__(
        "domain_literal",
        f["finalization_evidence"]["authentication_boundary"]["domain_literal"],
    ),
    "semantic_head_domain_reused_as_document_domain": lambda r, f: f[
        "finalization_evidence"
    ].__setitem__(
        "complete_semantic_head_digest_domain_literal",
        f["anchor_document_contract"]["document_digest_domain_literal"],
    ),
    "empty_domain_separator": lambda r, f: f["anchor_document_contract"].__setitem__(
        "document_digest_domain_literal", ""
    ),
    "unversioned_domain_separator": lambda r, f: f["anchor_document_contract"].__setitem__(
        "document_digest_domain_literal", "cryptohunter.account-genesis.document"
    ),
    "caller_configurable_domain_separator": lambda r, f: f["concrete_representation_refinement"][
        "domain_rules"
    ].__setitem__("runtime_or_caller_configurable", True),
    "authoritative_document_auth_unspecified": lambda r, f: f["anchor_document_contract"][
        "authoritative_document_authentication"
    ].__setitem__("algorithm", "NOT_FROZEN"),
    "proposer_key_reused_for_authoritative_document": lambda r, f: f["anchor_document_contract"][
        "authoritative_document_authentication"
    ].__setitem__("proposer_key_role_allowed", True),
    "invalid_unicode_lone_surrogate_accepted": lambda r, f: f["concrete_representation_refinement"][
        "canonical_profile"
    ].__setitem__("invalid_Unicode_or_lone_surrogate", "ACCEPT"),
    "padded_base64url_accepted": lambda r, f: f["concrete_representation_refinement"][
        "canonical_profile"
    ].__setitem__("base64url", "padding accepted"),
    "uppercase_digest_accepted": lambda r, f: f["concrete_representation_refinement"][
        "canonical_profile"
    ].__setitem__("digest_hex", "uppercase accepted"),
    "duplicate_JSON_keys_accepted": lambda r, f: f["concrete_representation_refinement"][
        "canonical_profile"
    ].__setitem__("duplicate_object_keys", "ACCEPT"),
    "set_like_array_order_implementation_dependent": lambda r, f: f[
        "concrete_representation_refinement"
    ]["canonical_profile"].__setitem__("set_like_array_rule", "implementation-dependent"),
}


CROSS_ARTIFACT_STATUS_MUTATIONS = {
    "physical_frozen_freshness_says_F_DESIGN_BLOCKED": lambda r, f, p: f[
        "preserved_status"
    ].__setitem__("physical persistence protocol", "F_DESIGN_BLOCKED"),
    "physical_frozen_freshness_unblocked_false": lambda r, f, p: f["result"].__setitem__(
        "physical_persistence_unblocked", False
    ),
    "freshness_frozen_physical_not_frozen": lambda r, f, p: p["result"].__setitem__(
        "physical_protocol_frozen", False
    ),
    "freshness_reason_claims_canonical_crypto_NOT_FROZEN": lambda r, f, p: f["result"].__setitem__(
        "reason", "canonical serialization/key algorithm review does not exist"
    ),
    "freshness_reason_claims_physical_design_blockers_remain": lambda r, f, p: f[
        "result"
    ].__setitem__("reason", "physical persistence blockers remain"),
    "implementation_available_inferred_from_physical_design_freeze": lambda r, f, p: f[
        "implementation_allowed"
    ].__setitem__("FreshnessAuthority", "YES"),
    "production_M05_available_from_physical_design_freeze": lambda r, f, p: f[
        "preserved_status"
    ].__setitem__("production M0.5", "AVAILABLE"),
}


def test_contract_and_markdown_are_valid_and_in_sync() -> None:
    contract = load()
    validate(contract)
    assert MARKDOWN.read_text(encoding="utf-8") == render(contract)
    assert contract["redteam_mutations"] == list(MUTATIONS)
    assert contract["cross_artifact_redteam_mutations"] == list(CROSS_ARTIFACT_MUTATIONS)
    assert contract["cross_artifact_status_redteam_mutations"] == list(
        CROSS_ARTIFACT_STATUS_MUTATIONS
    )


@pytest.mark.parametrize("mutation_name", MUTATIONS)
def test_each_named_redteam_mutation_is_applied_and_rejected(mutation_name: str) -> None:
    original = load()
    mutated = deepcopy(original)
    MUTATIONS[mutation_name](mutated)
    assert mutated != original, f"mutation {mutation_name} was not applied"
    with pytest.raises(AssertionError):
        validate(mutated)


@pytest.mark.parametrize("mutation_name", CROSS_ARTIFACT_MUTATIONS)
def test_cross_artifact_crypto_mutations_are_applied_and_rejected(
    mutation_name: str,
) -> None:
    readiness = load()
    freshness = load("m05_account_genesis_freshness_authority_cas_finalization_contract.json")
    physical = load("m05_account_genesis_physical_persistence_crash_atomicity_contract.json")
    mutated_readiness = deepcopy(readiness)
    mutated_freshness = deepcopy(freshness)
    mutated_physical = deepcopy(physical)
    CROSS_ARTIFACT_MUTATIONS[mutation_name](mutated_readiness, mutated_freshness)
    assert (mutated_readiness, mutated_freshness, mutated_physical) != (
        readiness,
        freshness,
        physical,
    ), f"mutation {mutation_name} was not applied"
    with pytest.raises(AssertionError):
        validate_cross_artifact(mutated_readiness, mutated_freshness, mutated_physical)


@pytest.mark.parametrize("mutation_name", CROSS_ARTIFACT_STATUS_MUTATIONS)
def test_cross_artifact_status_mutations_are_applied_and_rejected(
    mutation_name: str,
) -> None:
    readiness = load()
    freshness = load("m05_account_genesis_freshness_authority_cas_finalization_contract.json")
    physical = load("m05_account_genesis_physical_persistence_crash_atomicity_contract.json")
    mutated_readiness = deepcopy(readiness)
    mutated_freshness = deepcopy(freshness)
    mutated_physical = deepcopy(physical)
    CROSS_ARTIFACT_STATUS_MUTATIONS[mutation_name](
        mutated_readiness, mutated_freshness, mutated_physical
    )
    assert (mutated_readiness, mutated_freshness, mutated_physical) != (
        readiness,
        freshness,
        physical,
    ), f"mutation {mutation_name} was not applied"
    with pytest.raises(AssertionError):
        validate_cross_artifact(mutated_readiness, mutated_freshness, mutated_physical)


def test_cross_artifact_frozen_inputs_and_boundaries() -> None:
    contract = load()
    freshness = load("m05_account_genesis_freshness_authority_cas_finalization_contract.json")
    physical = load("m05_account_genesis_physical_persistence_crash_atomicity_contract.json")
    topology = load("m05_account_genesis_authority_topology_resolution_after_binding_freeze.json")
    security = load("m05_account_genesis_security_substrate_contract.json")
    operation = load("m05_account_genesis_operation_identity_request_binding_contract.json")
    reservation = load("m05_cryptohunter_account_genesis_reservation_state_model.json")
    root_proof = load("m05_account_genesis_root_proof_admission_binding_contract.json")

    validate_cross_artifact(contract, freshness, physical)

    frozen = contract["frozen_inputs"]
    assert freshness["selected_or_blocked_model"]["selected"] == frozen["freshness_model"]
    assert physical["selected_or_blocked_protocol"]["selection"] == (
        "INITIAL_BINDING_THEN_PREPARED_THEN_FRESHNESS_CAS_THEN_LOCAL_FINAL_COMMIT"
    )
    assert physical["projection_boundary"]["M0.11_SQLiteStateStore_CryptoHunterAccount"].startswith(
        "PROJECTION_CARRIER_ONLY"
    )
    assert (
        topology["separation_invariants"]["freshness_owner_is_genesis_semantic_decision_owner"]
        is False
    )
    assert security["anchor_models"]["catalog_anchor_reused"] is False
    assert security["production_test_separation"]["authority_domain"] == "DISTINCT"
    assert operation["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
    assert reservation["preserved_status"]["production_M0.5"] == "NOT_AVAILABLE"
    assert root_proof["authority_boundary"]["issuer"] != "CryptoHunterAccountAuthority"
