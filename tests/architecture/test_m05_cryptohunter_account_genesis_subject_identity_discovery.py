"""Executable checks for account-genesis subject identity discovery."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_genesis_subject_identity_discovery.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_genesis_subject_identity_discovery.md"
ALLOWED_PROVENANCE = {
    "EXACT_COMMIT",
    "PATH_CONTENT_EQUIVALENT",
    "HIGH_CONFIDENCE_LINEAGE_EQUIVALENT",
    "LIKELY_EQUIVALENT",
    "UNKNOWN",
    "MISMATCH",
}


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    return (
        "# M0.5 CryptoHunterAccount genesis subject identity discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_genesis_subject_identity_discovery.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
    )


def candidates(value: dict) -> dict[str, dict]:
    return {item["candidate"]: item for item in value["candidate_matrix"]}


def validate(value: dict) -> None:
    assert value["repository_head_examined"] == "95e798f2e40a5b54ea0529f184078debe6b701d7"
    provenance = value["provenance"]
    assert provenance["classification"] in ALLOWED_PROVENANCE
    assert provenance["classification"] == "UNKNOWN"
    assert provenance["reviewed_head_available_locally"] is False
    assert provenance["current_tree_evidence"] is True
    assert provenance["finding_scope"] == "CURRENT_TREE_ONLY"
    assert provenance["formal_advancement_allowed"] is False
    assert provenance["relationship_to_reviewed_sha"] == "UNKNOWN"
    assert provenance["current_local_tree_inspected"] == "YES"
    assert provenance["path_content_comparison_evidence"] == "NOT_PERFORMED"
    required = value["subject_requirements"]
    assert len(required) == 9
    matrix = candidates(value)
    assert {
        "intended_operator_id",
        "provisioning_context_fingerprint_sha256",
        "claim_fingerprint_sha256",
        "device_installation_id",
        "account_id",
        "external provisioning subject candidate",
        "license/subscription candidate",
        "deployment/install owner candidate",
    } <= matrix.keys()
    for item in matrix.values():
        assert {
            "independent_of_account",
            "pre_account",
            "stable_across_retry",
            "stable_across_device",
            "authority_bound",
            "durable",
            "result",
            "rejection_reason",
            "evidence",
        } <= item.keys()
    assert value["intended_operator_analysis"]["classification"] == "NOT_VIABLE_ACCOUNT_CHILD"
    assert value["intended_operator_analysis"]["parent_requires_existing_account"] is True
    assert value["intended_operator_analysis"]["genuine_pre_account_authority"] is False
    assert matrix["device_installation_id"]["result"] == "NOT_VIABLE_ACCOUNT_CHILD"
    assert matrix["claim_fingerprint_sha256"]["result"] == ("ATTEMPT_CONTEXT_FINGERPRINT_ONLY")
    assert value["provisioning_context_analysis"]["classification"] == "UNKNOWN"
    assert value["provisioning_context_analysis"]["stable_across_retries"] == "NOT_PROVEN"
    assert value["provisioning_context_analysis"]["stable_across_devices"] == "NOT_PROVEN"
    assert value["external_provisioning_subject_analysis"]["classification"] == (
        "EXTERNAL_SUBJECT_REQUIRED_BUT_SCHEMA_ABSENT"
    )
    assert value["authority_provenance"]["authority_source_is_subject"] is False
    assert value["authority_provenance"]["public_hash_is_authority"] is False
    assert value["authority_provenance"]["caller_minted_subject_allowed"] is False
    assert value["license_subscription_analysis"]["classification"] == "ENTITLEMENT_NOT_SUBJECT"
    assert value["deployment_owner_analysis"]["DeviceInstallation"] == "NOT_VIABLE_ACCOUNT_CHILD"
    assert value["cross_device_stability"]["result"] == "NOT_PROVEN"
    assert value["retry_stability"]["result"] == "NOT_PROVEN"
    cardinality = value["subject_account_cardinality"]
    assert cardinality["subject_to_account_cardinality"] == "NOT_FROZEN"
    assert cardinality["one_subject_one_account"] == "NOT_PROVEN"
    assert cardinality["multiple_accounts_per_subject_allowed"] == "NOT_PROVEN"
    assert cardinality["multiple_subjects_per_account_allowed"] == "NOT_PROVEN"
    assert cardinality["account_uniqueness_key"] == "NOT_FROZEN"
    assert cardinality["missing_subject_identity_alone_makes_genesis_impossible"] is False
    assert value["candidate_domain_models"]["selected_model"] == "NOT_FROZEN"
    assert value["same_subject_duplicate_attack"]["classification"] == (
        "DOMAIN_SEMANTICS_NOT_FROZEN"
    )
    assert value["same_subject_duplicate_attack"]["automatic_security_violation"] is False
    assert value["distinct_subject_attack"]["subject_distinction"] == "UNAVAILABLE"
    assert value["distinct_subject_attack"]["account_genesis_impossible_inferred"] is False
    assert value["distinct_subject_attack"]["global_singleton_serialization_allowed"] is False
    assert value["cross_artifact_parity"]["OperatorIdentity_parent"] == "CryptoHunterAccount"
    assert value["cross_artifact_parity"]["DeviceInstallation_parent"] == "CryptoHunterAccount"
    assert value["subject_cardinality_parity"]["parity"] == "PASS"
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_SUBJECT_IDENTITY_INSUFFICIENT_SEMANTICS"
    )
    assert value["result"]["upstream_dependency_status"] == (
        "CONDITIONAL_BLOCKER_IF_SUBJECT_IDENTITY_REQUIRED"
    )
    assert value["result"]["canonical_subject_identity"] == "NOT_FOUND"
    assert value["result"]["result_scope"] == "CURRENT_TREE_ONLY"
    assert value["result"]["formal_status_advancement"] == "WITHHELD"
    assert value["impact_on_genesis_model"]["per_subject_serialization_required"] == "NOT_FROZEN"
    assert value["impact_on_genesis_model"]["logical_account_genesis_idempotency_basis"] == (
        "NOT_FROZEN"
    )
    assert (
        value["impact_on_genesis_model"]["missing_external_subject_identity_alone_blocks_genesis"]
        is False
    )
    assert "A pre-account external subject is required" not in value["result"]["reason"]
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert set(value["mandatory_redteam_mutations"].values()) == {"FAIL"}


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


def test_canonical_parent_parity() -> None:
    value = load()
    vocabulary = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    parents = {item["canonical_name"]: item["parent"] for item in vocabulary["entity_kinds"]}
    assert value["cross_artifact_parity"]["OperatorIdentity_parent"] == parents["OperatorIdentity"]
    assert (
        value["cross_artifact_parity"]["DeviceInstallation_parent"] == parents["DeviceInstallation"]
    )


def test_subject_cardinality_parity_with_genesis_model_and_vocabulary() -> None:
    value = load()
    genesis = json.loads(
        (DOCS / "m05_cryptohunter_account_genesis_authority_model.json").read_text()
    )
    vocabulary = json.loads((DOCS / "canonical_domain_vocabulary.json").read_text())
    parity = value["subject_cardinality_parity"]
    assert (
        parity["discovery_subject_to_account_cardinality"]
        == (value["subject_account_cardinality"]["subject_to_account_cardinality"])
    )
    assert (
        parity["genesis_model_subject_to_account_cardinality"]
        == (genesis["account_subject_identity"]["subject_to_account_cardinality"])
    )
    account = next(
        item
        for item in vocabulary["entity_kinds"]
        if item["canonical_name"] == "CryptoHunterAccount"
    )
    assert account["purpose"] == "root SaaS/customer account"
    assert parity["canonical_vocabulary_subject_to_account_cardinality"] == "NOT_DEFINED"
    assert parity["one_to_one_inferred_from_root_SaaS_customer_account"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["intended_operator_analysis"].update(
            classification="VIABLE_PRE_ACCOUNT_SUBJECT"
        ),
        lambda d: candidates(d)["device_installation_id"].update(result="ACCEPTED_SUBJECT"),
        lambda d: candidates(d)["claim_fingerprint_sha256"].update(
            result="STABLE_SUBJECT_FINGERPRINT"
        ),
        lambda d: d["provisioning_context_analysis"].update(
            classification="STABLE_SUBJECT_FINGERPRINT"
        ),
        lambda d: d["authority_provenance"].update(authority_source_is_subject=True),
        lambda d: d["license_subscription_analysis"].update(classification="LIFETIME_SUBJECT"),
        lambda d: d["authority_provenance"].update(caller_minted_subject_allowed=True),
        lambda d: d["same_subject_duplicate_attack"].update(classification="SECURITY_CONFLICT"),
        lambda d: d["distinct_subject_attack"].update(global_singleton_serialization_allowed=True),
        lambda d: d["subject_account_cardinality"].update(one_subject_one_account=True),
        lambda d: d["subject_account_cardinality"].update(
            missing_subject_identity_alone_makes_genesis_impossible=True
        ),
        lambda d: d["impact_on_genesis_model"].update(
            missing_external_subject_identity_alone_blocks_genesis=True
        ),
        lambda d: d["result"].update(reason="A pre-account external subject is required"),
    ],
)
def test_mandatory_redteam_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["provenance"].update(classification="GIT_CURRENT_TREE_EVIDENCE"),
        lambda d: d["provenance"].update(formal_advancement_allowed=True),
        lambda d: d["provenance"].update(classification="EXACT_COMMIT"),
        lambda d: d["provenance"].update(classification="PATH_CONTENT_EQUIVALENT"),
    ],
)
def test_invalid_provenance_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)


def test_preserved_status_and_implementation_gates() -> None:
    value = load()
    assert value["preserved_status"] == {
        "M0.12": "ACCEPTED / AVAILABLE",
        "M0.7 structural Full Fill v2": "ACCEPTED / STRUCTURAL AVAILABLE",
        "CryptoHunterAccountAuthority": "NOT_AVAILABLE",
        "WorkspaceAuthority": "NOT_AVAILABLE",
        "InstrumentAuthority": "NOT_AVAILABLE",
        "WCP Authority": "NOT_AVAILABLE / DESIGN_BLOCKED",
        "FullFillAuthority": "NOT_AVAILABLE",
        "production M0.5": "NOT_AVAILABLE",
        "M0.8": "NOT_AVAILABLE",
        "C25": "BLOCKED",
        "S9D": "OPEN",
    }
