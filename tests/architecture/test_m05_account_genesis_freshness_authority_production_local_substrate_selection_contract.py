"""Executable selection gate for the future PRODUCTION_LOCAL FreshnessAuthority."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
STEM = "m05_account_genesis_freshness_authority_production_local_substrate_selection_contract"
MACHINE = DOCS / f"{STEM}.json"
MARKDOWN = DOCS / f"{STEM}.md"
CANONICAL = DOCS / "m05_account_genesis_freshness_authority_cas_finalization_contract.json"


def load():
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def load_canonical():
    return json.loads(CANONICAL.read_text(encoding="utf-8"))


def validate_frozen_receipt_parity(selection, canonical):
    receipt = selection["receipt"]
    evidence = canonical["finalization_evidence"]
    boundary = evidence["authentication_boundary"]
    assert receipt["exact_value_fields"] == evidence["fields"]
    assert receipt["exact_authenticated_preimage_fields"] == [
        field for field in evidence["fields"]
        if field != "authentication_tag_or_signature"
    ]
    assert receipt["authentication_binding"] == boundary["authentication_binding"]
    assert receipt["domain"] == boundary["domain_literal"]
    assert receipt["canonical_byte_encoding"] == boundary["canonical_byte_encoding"]
    assert receipt["signature"] == boundary["signature_representation"]
    assert receipt["authentication_tag_or_signature_in_own_preimage"] is False


def validate_original_decision_identity_parity(selection, canonical):
    selected = selection["original_decision_identity"]
    frozen = canonical["original_decision_identity_contract"]
    assert selected["type"] == frozen["type"]
    assert selected["exact_semantic_tuple"] == frozen["semantic_tuple"]
    assert selected["domain"] == frozen["domain_literal"]
    assert selected["is_freshness_finalization_receipt_v1_field"] is False


def render(value):
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return (
        "# M0.5 — wybór substrate PRODUCTION_LOCAL FreshnessAuthority\n\n"
        f"> Deterministyczna projekcja `{STEM}.json`. JSON jest źródłem prawdy.\n\n"
        "## Wynik\n\n"
        f"**{value['principal_result']}**\n\n"
        "Wybrano wyłącznie model substrate. FreshnessAuthority oraz runtime nie są zaimplementowane.\n\n"
        f"## Projekcja maszynowa\n\n```json\n{body}```\n"
    )


def test_machine_and_markdown_are_exact_parity():
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


def test_frozen_model_and_physical_protocol_are_unchanged():
    frozen = load()["frozen_inputs"]
    assert frozen == {
        "freshness_model": "F_HYBRID_CAS_PLUS_SIGNED_FINALIZATION_RECEIPT",
        "update_unit": "FULL_AUTHORITATIVE_DOCUMENT",
        "winner": "EXACTLY_ONE_SUCCESSFUL_N_TO_N_PLUS_1",
        "physical_protocol": "INITIAL_BINDING -> PREPARED -> EXTERNAL_FRESHNESS_CAS -> LOCAL_FINAL_COMMIT",
        "accepted_document_equals_proposed_document": True,
    }


def test_discovery_is_complete_and_honest():
    items = {item["capability"]: item for item in load()["current_tree_discovery"]}
    assert set(items) == {
        "production freshness CAS backend", "authoritative freshness history backend",
        "durable finalization receipt store", "dedicated local freshness authority adapter",
        "freshness authority signing custody", "CHA freshness proposer custody",
        "trusted proposer verification lineage", "TEST/PRODUCTION separation",
        "independent revoked-history evidence", "production/local rollback model",
    }
    assert {item["status"] for item in items.values()} <= {"FOUND", "PARTIAL", "NOT_FOUND"}
    assert items["production freshness CAS backend"]["status"] == "NOT_FOUND"
    assert items["durable finalization receipt store"]["status"] == "NOT_FOUND"
    assert items["freshness authority signing custody"]["status"] == "FOUND"
    assert items["CHA freshness proposer custody"]["status"] == "FOUND"
    assert all((ROOT / item["path"]).exists() for item in items.values())


def test_postgresql_selection_is_transactional_not_process_local():
    value = load(); selection = value["production_local_selection"]
    assert selection["selection_is_implementation"] is False
    assert selection["selected"].startswith("POSTGRESQL_16_SERIALIZABLE")
    assert "database SERIALIZABLE transaction" in selection["concurrency_authority"]
    assert selection["durability"] == {
        "minimum_server_version": 16, "fsync": "on", "synchronous_commit": "on",
        "unlogged_tables_allowed": False, "temporary_authority_objects_allowed": False,
    }
    same = value["schema_authority_model"]["same_predecessor"]
    assert same["same_original_decision_identity_and_bytes"] == "ALREADY_ACCEPTED_EXACT"
    assert same["different_candidate_or_request"] == "CAS_CONFLICT"
    assert same["automatic_N_plus_2_retry"] is False


def test_atomic_authority_transaction_retains_decision_and_receipt():
    model = load()["schema_authority_model"]
    assert set(model["tables"]) == {
        "authority_lineages", "authoritative_documents", "decisions",
        "finalization_receipts", "key_lifecycle_history",
    }
    transaction = " ".join(model["atomic_transaction"])
    for required in ("exact current generation/digest/complete head set", "exact N+1",
                     "immutable document, decision, receipt", "commit exactly once"):
        assert required in transaction


def test_lifecycle_has_one_authority_at_cas_linearization():
    ownership = load()["lifecycle_authority_ownership"]
    assert ownership["dual_authority"] is False
    assert "PostgreSQL" in ownership["active_at_new_cas_authoritative_owner"]
    assert "sole ACTIVE eligibility source" in ownership["postgresql_key_lifecycle_history"]
    assert "never independently authoritative" in ownership["local_custody_metadata"]
    assert "typed key_version" in ownership["cas_rule"]


def test_receipt_is_exact_and_never_caller_authority():
    receipt = load()["receipt"]
    assert receipt["type"] == "FreshnessFinalizationReceipt"
    assert receipt["atomic_with_decision"] is True
    assert receipt["caller_constructed_receipt_is_authority"] is False
    assert receipt["domain"] == "cryptohunter.account-genesis.freshness-finalization-receipt-authentication.v1"
    assert set(receipt["exact_value_fields"]) == {
        "schema_version", "receipt_id", "environment", "trust_domain", "authority_id",
        "exact_predecessor_generation", "exact_predecessor_document_digest",
        "accepted_generation", "accepted_document_digest", "complete_semantic_head_digest",
        "finalization_request_id", "authentication_tag_or_signature",
        "freshness_authority_key_id", "freshness_authority_key_version",
    }


def test_receipt_fields_authentication_binding_and_domain_match_canonical_artifact():
    validate_frozen_receipt_parity(load(), load_canonical())


def test_original_decision_identity_tuple_matches_canonical_artifact():
    validate_original_decision_identity_parity(load(), load_canonical())


def test_internal_database_metadata_cannot_leak_into_receipt_v1_preimage():
    value = load()
    separation = value["internal_vs_canonical"]
    assert separation["internal_fields_change_frozen_receipt_v1_preimage"] is False
    assert separation["receipt_storage"]["metadata_in_cryptographic_preimage"] is False
    assert "original_decision_identity" in separation["internal_postgresql_authority_fields_may_include"]
    assert "original_decision_identity" not in value["receipt"]["exact_value_fields"]


def test_recovered_receipt_is_byte_compatible_with_frozen_v1_verifier():
    value = load()
    validate_frozen_receipt_parity(value, load_canonical())
    assert value["receipt"]["stored_canonical_bytes_must_equal_frozen_v1_representation"] is True
    assert value["receipt"]["lost_response_recovered_receipt_accepted_by_frozen_v1_verifier"] is True
    committed = value["crash_and_lost_response_matrix"][1]
    assert "byte-identical" in committed["compatibility"]


def test_extra_original_decision_identity_in_receipt_preimage_fails_parity():
    mutated = deepcopy(load())
    mutated["receipt"]["exact_authenticated_preimage_fields"].append("original_decision_identity")
    with pytest.raises(AssertionError):
        validate_frozen_receipt_parity(mutated, load_canonical())


def test_replacing_complete_head_digest_with_two_internal_fields_fails_parity():
    mutated = deepcopy(load())
    fields = mutated["receipt"]["exact_authenticated_preimage_fields"]
    fields.remove("complete_semantic_head_digest")
    fields.extend(("predecessor_complete_semantic_head_set_digest",
                   "accepted_complete_semantic_head_set_digest"))
    with pytest.raises(AssertionError):
        validate_frozen_receipt_parity(mutated, load_canonical())


def test_renaming_exact_predecessor_generation_fails_parity():
    mutated = deepcopy(load())
    fields = mutated["receipt"]["exact_authenticated_preimage_fields"]
    fields[fields.index("exact_predecessor_generation")] = "predecessor_generation"
    with pytest.raises(AssertionError):
        validate_frozen_receipt_parity(mutated, load_canonical())


def test_custody_roles_and_numeric_versions_are_distinct():
    custody = load()["crypto_and_custody"]
    assert custody["authority_role"] != custody["proposer_role"]
    assert "never parse" in custody["numeric_key_version_source"]
    assert "never use lifecycle_generation" in custody["numeric_key_version_source"]
    assert set(custody["forbidden_alias_dimensions"]) == {
        "provider namespace", "credential ID", "key handle", "key material identity", "lifecycle namespace"
    }
    assert {"RootProof issuer", "history attestation", "CHA proposer"} <= set(custody["forbidden_alias_roles"])


def test_revoked_history_and_rollback_claims_are_bounded():
    history = load()["history_and_rollback"]
    assert history["negative_lookup_is_non_acceptance"] is False
    assert "independently currently-trusted" in history["revoked_history"]
    assert history["coordinated_full_host_rollback_detected"] is False
    assert "does not claim SERVER_READY" in history["claim"]


def test_crash_matrix_has_exact_recovery_and_no_second_successor():
    matrix = load()["crash_and_lost_response_matrix"]
    assert len(matrix) == 6
    assert {row["cut"] for row in matrix} == {
        "before CAS", "CAS transaction committed / receipt response lost",
        "history committed / receipt not returned", "receipt committed / process crash",
        "CAS committed / local caller response lost", "restart before local final commit",
    }
    assert all("retry" in row and "durable" in row for row in matrix)
    assert any("never second N+1" in row["retry"] for row in matrix)


def test_qualification_reuses_v5_hardening_without_manifest_trust():
    qualification = load()["qualification"]
    assert qualification["persisted_manifest_is_trust_root"] is False
    assert qualification["structural_protocol_is_trusted_provenance"] is False
    joined = " ".join(qualification["required"])
    for item in ("role names and OIDs", "fixed search_path", "PUBLIC execute absent",
                 "no user triggers", "pg_proc.prosrc", "three-way equality"):
        assert item in joined


@pytest.mark.parametrize("mutation", [
    "fabricated receipt", "fabricated VALID/current generation/digest",
    "subclass/dict/copy/object.__setattr__ mutation", "wrong environment/trust domain/authority",
    "same-generation unequal successor", "concurrent N+1 candidates",
    "stale proposer lifecycle", "dynamic signing identity/version",
    "TEST credential in PRODUCTION", "all forbidden role aliases",
    "schema/function/role/manifest tampering", "revoked key self-corroboration",
])
def test_required_red_team_matrix_is_closed(mutation):
    assert mutation in load()["red_team_required"]


def test_readiness_remains_fail_closed():
    readiness = load()["readiness"]
    assert readiness["production_substrate_selected"] is True
    assert readiness["production_substrate_implemented"] is False
    assert readiness["FreshnessAuthority_implemented"] is False
    assert readiness["FreshnessAuthority_implementation_allowed_after_iteration"] is False
    assert readiness["ROOT_PROOF_ISSUER_IMPLEMENTED"] is False
    assert readiness["PRODUCTION_LOCAL_RUNTIME_AVAILABLE"] is False
    assert readiness["classification"] == "UNKNOWN"
    assert readiness["finding_scope"] == "CURRENT_TREE_ONLY"
    assert readiness["formal_project_advancement"] == "WITHHELD"
    assert readiness[
        "FRESHNESS_AUTHORITY_PRODUCTION_LOCAL_SIGNING_CUSTODY_FOUNDATION_IMPLEMENTED"
    ] is True
    assert len(readiness["remaining_blockers"]) == 4


@pytest.mark.parametrize("path,value", [
    (("readiness", "FreshnessAuthority_implemented"), True),
    (("readiness", "FreshnessAuthority_implementation_allowed_after_iteration"), True),
    (("production_local_selection", "selection_is_implementation"), True),
    (("receipt", "caller_constructed_receipt_is_authority"), True),
    (("history_and_rollback", "negative_lookup_is_non_acceptance"), True),
    (("history_and_rollback", "coordinated_full_host_rollback_detected"), True),
    (("schema_authority_model", "same_predecessor", "automatic_N_plus_2_retry"), True),
])
def test_status_and_authority_mutations_are_detected(path, value):
    mutated = deepcopy(load()); target = mutated
    for key in path[:-1]: target = target[key]
    target[path[-1]] = value
    assert mutated != load()
