from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "architecture" / "cryptohunter_product_architecture"
STEM = "m05_account_genesis_freshness_crypto_to_postgres_cas_authorization_contract"


def _load(name: str) -> dict[str, object]:
    return json.loads((DOCS / name).read_text(encoding="utf-8"))


def _contract() -> dict[str, object]:
    return _load(f"{STEM}.json")


def test_machine_and_markdown_projection_are_exact() -> None:
    raw = (DOCS / f"{STEM}.json").read_text(encoding="utf-8")
    markdown = (DOCS / f"{STEM}.md").read_text(encoding="utf-8")
    assert markdown.endswith(f"```json\n{raw}```\n")
    assert _contract()["artifact"] == (
        "M05_ACCOUNT_GENESIS_FRESHNESS_CRYPTO_TO_POSTGRES_CAS_AUTHORIZATION_CONTRACT"
    )


def test_selected_model_is_isolated_semantic_verifier_plus_db_preparation() -> None:
    contract = _contract()
    selected = contract["selected_model"]
    assert selected["name"] == "ISOLATED_SEMANTIC_VERIFIER_WITH_DB_AUTHENTICATED_ONE_TIME_PREPARATION"
    assert selected["composition"] == [
        "C_ISOLATED_LOCAL_VERIFIER_PROCESS",
        "D_TWO_PHASE_DB_PREPARATION",
    ]
    assert "PostgreSQL authenticates the exact login role" in selected["nonforgeable_basis"]
    candidates = contract["considered_models"]
    assert candidates["A_VERIFY_INSIDE_POSTGRESQL"]["status"] == "NOT_SELECTED"
    assert candidates["B_HMAC_VERIFIER_EVIDENCE"]["status"] == "NOT_SELECTED"
    assert candidates["C_ISOLATED_LOCAL_VERIFIER_PROCESS"]["status"].startswith("SELECTED")
    assert candidates["D_TWO_PHASE_DB_PREPARATION"]["status"].startswith("SELECTED")


def test_raw_dml_boolean_token_security_definer_and_session_assertions_are_rejected() -> None:
    contract = _contract()
    rejected = contract["rejected_models"]
    assert all(item["status"] == "REJECTED" for item in rejected.values())
    assert contract["problem_statement"]["runtime_raw_authority_dml"] is False
    assert contract["problem_statement"]["caller_constructible_verification_assertion"] is False
    boundary = contract["mutation_boundary"]
    assert boundary["runtime_raw_dml"] is False
    assert boundary["verifier_raw_dml"] is False
    assert boundary["admin_arbitrary_raw_dml"] is False


def test_runtime_cannot_mint_evidence_or_obtain_generic_crypto_capability() -> None:
    contract = _contract()
    verifier = contract["verifier_api"]
    assert verifier["operation"] == "verify_exact_freshness_candidate_and_prepare"
    assert verifier["generic_sign_or_mac_available"] is False
    roles = contract["authority_roles"]
    assert "no raw table DML" in roles["freshness_runtime"]
    assert "no prepare execute" in roles["freshness_runtime"]
    assert "no raw table DML" in roles["freshness_crypto_verifier"]
    assert "no CAS execute" in roles["freshness_crypto_verifier"]


def test_evidence_binds_exact_candidate_domain_identities_and_roles() -> None:
    evidence = _contract()["evidence_schema"]
    fields = set(evidence["exact_fields"])
    assert {
        "schema_version",
        "security_profile",
        "environment",
        "trust_domain",
        "authority_id",
        "operation_type",
        "expected_predecessor_generation",
        "expected_predecessor_document_digest",
        "expected_predecessor_complete_semantic_head_digest",
        "proposed_document_digest",
        "original_decision_identity",
        "proposer_identity",
        "proposer_credential_role_identity",
        "proposer_key_version",
        "proposer_public_key_material_identity",
        "proposer_authentication_digest",
        "finalization_credential_role_identity",
        "finalization_key_version",
        "finalization_public_key_material_identity",
        "authoritative_document_authentication_digest",
        "receipt_id",
        "receipt_canonical_digest",
        "finalization_request_id",
        "preparation_id",
        "verifier_authority_identity",
        "verifier_authority_version",
    } <= fields
    assert evidence["excluded_assertions"] == ["signature_valid", "verified", "trusted", "ACTIVE"]
    assert evidence["cross_scope_reuse"] == "REJECT"
    assert evidence["substitution"] == "REJECT"


def test_postgresql_lifecycle_remains_sole_cas_authority_and_races_are_ordered() -> None:
    lifecycle = _contract()["lifecycle_and_ordering"]
    assert lifecycle["authority"].startswith("PostgreSQL retained lifecycle history")
    assert "never freezes lifecycle" in lifecycle["evidence_scope"]
    assert lifecycle["proposer_revoke_race"]["revoke_before_CAS"].startswith("REJECT")
    assert lifecycle["proposer_revoke_race"]["CAS_before_revoke"].startswith("acceptance may commit")
    assert lifecycle["proposer_rotation_race"]["rotation_before_CAS"].startswith("old proposer")
    assert lifecycle["finalization_signer_race"]["signer_loses_ACTIVE_before_CAS"].startswith("REJECT")


def test_one_time_new_decision_and_lost_response_replay_are_distinct() -> None:
    replay = _contract()["replay_semantics"]
    assert replay["new_decision"].startswith("Preparation is one-time consumable")
    assert "at most one physical decision append" in replay["concurrent_consumption"]
    assert "no new preparation and no N+2" in replay["lost_response_after_acceptance"]
    assert len(_contract()["crash_semantics"]) == 6


def test_postgresql_crypto_discovery_does_not_claim_pgcrypto_ed25519() -> None:
    discovery = _contract()["discovery"]
    assert discovery["postgresql_16_stock"] == "No native Ed25519 verification SQL primitive was found."
    assert "no Ed25519" in discovery["pgcrypto_1_3"]
    assert discovery["external_ed25519_extension_required_for_model_A"] is True
    assert discovery["external_extension_selected"] is False


def test_production_local_limitations_and_server_ready_migration_are_explicit() -> None:
    local = _contract()["production_local"]
    assert "host root can impersonate the verifier or steal its DB credential" in local["limitations"]
    assert "coordinated full-host rollback remains undetected without independent checkpoint" in local["limitations"]
    assert any("independently administered" in item for item in local["server_ready_migration"])


def test_cross_artifact_parity_without_redefining_frozen_types() -> None:
    contract = _contract()
    cas = _load("m05_account_genesis_freshness_authority_cas_finalization_contract.json")
    readiness = _load("m05_account_genesis_freshness_authority_implementation_readiness_contract.json")
    selection = _load(
        "m05_account_genesis_freshness_authority_production_local_substrate_selection_contract.json"
    )
    parity = contract["cross_artifact_parity"]
    assert cas["finalization_evidence"]["authentication_boundary"]["object_type"] == "FreshnessFinalizationReceipt"
    assert cas["original_decision_identity_contract"]["type"] == "OriginalDecisionIdentity"
    assert readiness["finalization_receipt_durability"]["commit_rule"] == (
        "receipt record and exact acceptance decision are one atomic durable authority transaction"
    )
    assert selection["lifecycle_authority_ownership"]["postgresql_key_lifecycle_history"].endswith(
        "sole ACTIVE eligibility source for a new CAS"
    )
    assert parity["freshness_receipt_v1"].startswith("Referenced unchanged")
    assert parity["original_decision_identity"].startswith("Referenced unchanged")
    assert "distinct" in parity["key_version"]


def test_no_implementation_is_authorized_and_flags_remain_false() -> None:
    contract = _contract()
    allowed = contract["implementation_allowed"]
    assert allowed["canonical_model_selected"] is True
    assert allowed["design_and_architecture_tests_only"] is True
    assert all(
        allowed[name] is False
        for name in (
            "postgresql_schema",
            "postgresql_tables",
            "sql_functions",
            "verifier_service",
            "FreshnessAuthority",
            "RootProofIssuer",
            "AccountGenesis_runtime_wiring",
        )
    )
    flags = contract["preserved_flags"]
    assert flags["FRESHNESS_AUTHORITY_PRODUCTION_LOCAL_SIGNING_CUSTODY_FOUNDATION_IMPLEMENTED"] is True
    assert flags["production_substrate_selected"] is True
    assert flags["production_substrate_implemented"] is False
    assert flags["FreshnessAuthority_implemented"] is False
    assert flags["FreshnessAuthority_implementation_allowed_after_iteration"] is True
    assert flags["ROOT_PROOF_ISSUER_IMPLEMENTED"] is False
    assert flags["PRODUCTION_LOCAL_RUNTIME_AVAILABLE"] is False
    assert flags["classification"] == "UNKNOWN"
    assert flags["finding_scope"] == "CURRENT_TREE_ONLY"
    assert flags["formal_project_advancement"] == "WITHHELD"


def test_security_definer_uses_session_user_for_caller_and_current_user_for_owner() -> None:
    contract = _contract()
    boundary = contract["mutation_boundary"]
    auth = boundary["invoker_authentication"]
    assert "session_user" in boundary["preparation_caller"] and "current_user" in boundary["preparation_caller"]
    assert "session_user" in boundary["cas_caller"] and "current_user" in boundary["cas_caller"]
    assert auth["inside_security_definer_current_user"].startswith("MUST equal")
    assert auth["inside_preparation_session_user"].endswith("LOGIN name/OID")
    assert auth["inside_cas_session_user"].endswith("LOGIN name/OID")
    assert auth["caller_supplied_principal_fields"] == "FORBIDDEN"
    assert auth["public_execute"] == "FORBIDDEN"
    assert "no grant option" in auth["execute_acl"]
    assert "pg_auth_members" in auth["memberships"]


def test_cross_role_same_raw_ed25519_key_material_is_unconditionally_rejected() -> None:
    contract = _contract()
    invariant = contract["evidence_schema"]["cross_role_key_material_invariant"]
    assert invariant["rule"] == (
        "proposer_public_key_material_identity MUST NOT equal "
        "finalization_public_key_material_identity"
    )
    assert invariant["same_exact_raw_ed25519_public_key_material"].startswith("REJECT")
    assert "exact canonical raw 32-byte Ed25519 public key material" in invariant["comparison_basis"]
    assert invariant["aliases_do_not_bypass"] == [
        "different credential_id",
        "different key_id",
        "different key_version",
        "different semantic role label",
    ]
    assert "simultaneous proposer/finalization registration" in invariant[
        "retained_authority_requirement"
    ]
    assert any(
        "same raw Ed25519 public key material identity" in item
        for item in contract["verifier_api"]["fail_closed_before_preparation"]
    )


def test_identity_and_exact_semantic_role_laundering_are_rejected() -> None:
    evidence = _contract()["evidence_schema"]
    assert "MUST exactly equal" in evidence["proposer_identity_binding"]
    roles = evidence["semantic_roles"]
    assert roles["proposer"] == "ACCOUNT_GENESIS_FRESHNESS_PROPOSER_SIGNING_V1"
    assert roles["finalization"] == (
        "ACCOUNT_GENESIS_FRESHNESS_AUTHORITY_FINALIZATION_SIGNING_V1"
    )
    assert roles["arbitrary_text_allowed"] is False
