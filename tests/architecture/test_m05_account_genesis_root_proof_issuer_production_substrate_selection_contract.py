"""Wykonywalny kontrakt wyboru production substrate Root-Proof Issuer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
STEM = "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract"
MACHINE = DOCS / f"{STEM}.json"
MARKDOWN = DOCS / f"{STEM}.md"
REQUIRED = {
    "deployment_trust_root_provider", "entitlement_registry_backend",
    "claimant_identity_registry", "requester_credential_registry",
    "root_proof_signing_key_custody", "global_serialization_CAS",
    "retained_authenticated_history", "history_checkpoint_or_anti_rollback",
    "reconciliation_evidence_source", "durable_local_attempt_storage",
    "TEST_PRODUCTION_separation",
}
MUTATIONS = {
    "CHA_directly_mutates_entitlement_registry",
    "issuer_directly_commits_account",
    "checkpoint_same_rollback_domain_without_independent_anchor",
    "ordinary_DB_durability_claimed_as_anti_rollback",
    "root_proof_signing_key_plaintext_config",
    "requester_key_reused_as_issuer_signing_key",
    "claimant_key_reused_as_requester_key",
    "requester_key_reused_as_freshness_proposer_key",
    "requester_key_reused_as_Catalog_authority_key",
    "requester_key_reused_as_storage_security_key",
    "issuer_signing_key_reused_as_freshness_proposer_key",
    "issuer_signing_key_reused_as_Catalog_authority_key",
    "claimant_key_reused_as_issuer_signing_key",
    "TEST_namespace_reused_for_PRODUCTION",
    "process_local_CAS_selected_for_multi_host_registry",
    "last_write_wins_selected_as_BIND",
    "checkpoint_unavailable_but_issuance_allowed",
    "registry_unavailable_but_signing_only_fallback_allowed",
    "NOT_FOUND_used_as_authoritative_UNBOUND",
    "restored_BOUND_to_UNBOUND_state_accepted",
    "restored_REVOKED_to_ACTIVE_state_accepted",
    "CHA_attempt_store_shares_authority_write_role_with_issuer_registry",
    "unknown_schema_auto_migrated",
    "caller_selects_entitlement_record",
    "local_timestamp_used_to_resolve_split_brain",
    "implementation_allowed_with_DECISION_REQUIRED_dependency",
    "compromised_issuer_key_becomes_VERIFY_ONLY",
    "compromised_requester_key_becomes_VERIFY_ONLY",
    "compromised_claimant_key_becomes_VERIFY_ONLY",
    "compromised_deployment_root_becomes_VERIFY_ONLY",
    "compromised_checkpoint_key_becomes_VERIFY_ONLY",
    "REVOKED_to_VERIFY_ONLY_allowed",
    "REVOKED_to_ACTIVE_allowed_after_compromise",
    "compromised_key_signature_alone_establishes_historical_acceptance",
    "compromised_checkpoint_self_attests_recovery",
    "restored_REVOKED_to_VERIFY_ONLY_state_accepted",
}
COMPROMISE_ROLES = {
    "claimant", "requester", "issuer_signing", "deployment_trust_root", "checkpoint"
}
LIFECYCLE_MUTATIONS = {
    "compromised_issuer_key_becomes_VERIFY_ONLY",
    "compromised_requester_key_becomes_VERIFY_ONLY",
    "compromised_claimant_key_becomes_VERIFY_ONLY",
    "compromised_deployment_root_becomes_VERIFY_ONLY",
    "compromised_checkpoint_key_becomes_VERIFY_ONLY",
    "REVOKED_to_VERIFY_ONLY_allowed",
    "REVOKED_to_ACTIVE_allowed_after_compromise",
    "compromised_key_signature_alone_establishes_historical_acceptance",
    "compromised_checkpoint_self_attests_recovery",
    "restored_REVOKED_to_VERIFY_ONLY_state_accepted",
}
REQUESTER = "ROOT_PROOF_REQUESTER"
CLAIMANT = "ROOT_PROOF_CLAIMANT"
ISSUER_SIGNING = "ROOT_PROOF_ISSUER_SIGNING"
FRESHNESS = "ACCOUNT_GENESIS_FRESHNESS_PROPOSER"
CATALOG = "CATALOG_AUTHORITY"
STORAGE = "STORAGE_SECURITY_KEY"
REQUIRED_DISTINCT_PAIRS = {
    frozenset((REQUESTER, CLAIMANT)),
    frozenset((REQUESTER, ISSUER_SIGNING)),
    frozenset((REQUESTER, FRESHNESS)),
    frozenset((REQUESTER, CATALOG)),
    frozenset((REQUESTER, STORAGE)),
    frozenset((CLAIMANT, ISSUER_SIGNING)),
    frozenset((ISSUER_SIGNING, FRESHNESS)),
    frozenset((ISSUER_SIGNING, CATALOG)),
}
ALIAS_MUTATION_PAIRS = {
    "requester_key_reused_as_issuer_signing_key": frozenset((REQUESTER, ISSUER_SIGNING)),
    "claimant_key_reused_as_requester_key": frozenset((CLAIMANT, REQUESTER)),
    "requester_key_reused_as_freshness_proposer_key": frozenset((REQUESTER, FRESHNESS)),
    "requester_key_reused_as_Catalog_authority_key": frozenset((REQUESTER, CATALOG)),
    "requester_key_reused_as_storage_security_key": frozenset((REQUESTER, STORAGE)),
    "issuer_signing_key_reused_as_freshness_proposer_key": frozenset(
        (ISSUER_SIGNING, FRESHNESS)
    ),
    "issuer_signing_key_reused_as_Catalog_authority_key": frozenset(
        (ISSUER_SIGNING, CATALOG)
    ),
    "claimant_key_reused_as_issuer_signing_key": frozenset((CLAIMANT, ISSUER_SIGNING)),
}


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return (
        f"# M0.5 — wybór production substrate dla Root-Proof Issuer\n\n"
        f"> Deterministyczna projekcja pliku `{STEM}.json`. JSON jest źródłem prawdy.\n\n"
        "## Wynik\n\n"
        "**ROOT_PROOF_ISSUER_PRODUCTION_SUBSTRATE_SELECTION_BLOCKED**\n\n"
        "Profil repo wspiera wdrożenie lokalne/jednohostowe, SQLite i ogólny OS keyring, "
        "lecz nie wspiera wielohostowego rejestru, niezależnego checkpointu ani Ed25519 "
        "HSM/KMS. Dlatego nie wolno arbitralnie wybrać infrastruktury operatorskiej. "
        "Jedynym wybranym substrate jest odrębny lokalny SQLite dla prób CHA; pozostałe "
        "dziesięć zależności ma `DECISION_REQUIRED`.\n\n"
        f"## Projekcja maszynowa\n\n```json\n{body}```\n"
    )


def validate_lifecycle_and_compromise(value: dict) -> None:
    lifecycle = value["credential_lifecycle"]
    assert set(lifecycle["states"]) == {"ACTIVE", "VERIFY_ONLY", "REVOKED"}
    assert lifecycle["planned_retirement_transition"] == "ACTIVE -> VERIFY_ONLY"
    assert set(lifecycle["compromise_transitions"]) == {
        "ACTIVE -> REVOKED", "VERIFY_ONLY -> REVOKED"
    }
    assert lifecycle["compromise_may_result_in_VERIFY_ONLY"] is False
    assert lifecycle["REVOKED_is_terminal"] is True
    assert lifecycle["REVOKED_to_VERIFY_ONLY_allowed"] is False
    assert lifecycle["REVOKED_to_ACTIVE_allowed"] is False

    history = lifecycle["history"]
    assert history["preserve_states"] == ["ACTIVE", "VERIFY_ONLY", "REVOKED"]
    assert history["preserve_compromised_generation_identity"] is True
    assert history["preserve_REVOKED_transition"] is True
    assert history["delete_compromised_generation"] is False
    assert history["reinterpret_REVOKED_as_VERIFY_ONLY"] is False
    assert history["retaining_REVOKED_equals_trusting_its_signature"] is False

    proof = lifecycle["historical_proof_for_REVOKED"]
    assert proof["independent_trusted_evidence_required"] is True
    assert len(proof["requirements"]) >= 5
    assert proof["compromised_key_signature_alone_sufficient"] is False
    assert proof["compromised_key_self_corroboration_allowed"] is False
    assert proof["wall_clock_establishes_direction"] is False
    assert proof["unavailable_evidence_result"] == "FAIL_CLOSED / UNAVAILABLE"
    assert proof["acceptance_changes_lifecycle_state"] is False

    checkpoint = lifecycle["checkpoint_interaction"]
    assert checkpoint["checkpoint_makes_compromised_key_trustworthy_again"] is False
    assert checkpoint["compromised_checkpoint_self_attests_recovery"] is False
    assert checkpoint["checkpoint_compromise_recovery_requires"] == (
        "independently trusted checkpoint/root lineage"
    )

    roles = lifecycle["roles"]
    assert set(roles) == COMPROMISE_ROLES
    assert roles == value["key_compromise"]
    for action in roles.values():
        assert action["compromise_transition"] == "ACTIVE|VERIFY_ONLY -> REVOKED"
        assert action["resulting_state"] == "REVOKED"
        assert action["compromised_key_signature_alone_sufficient"] is False
        assert action["retain_generation_and_revocation_in_history"] is True

    restore = value["restore"]
    assert restore["REVOKED_to_VERIFY_ONLY_restore_allowed"] is False
    assert restore["REVOKED_to_ACTIVE_restore_allowed"] is False
    assert restore["restored_VERIFY_ONLY_after_later_REVOKED"] == (
        "ROLLBACK_OR_TAMPER / FAIL_CLOSED"
    )


def credential_distinct_pairs(value: dict) -> set[frozenset[str]]:
    pairs = value["credential_role_non_aliasing"]["required_distinct_pairs"]
    assert all(isinstance(pair, list) and len(pair) == 2 for pair in pairs)
    normalized = {frozenset(pair) for pair in pairs}
    assert all(len(pair) == 2 for pair in normalized)
    assert len(normalized) == len(pairs)
    return normalized


def validate_credential_role_non_aliasing(value: dict) -> None:
    model = value["credential_role_non_aliasing"]
    assert set(model["roles"]) == {
        REQUESTER, CLAIMANT, ISSUER_SIGNING, FRESHNESS, CATALOG, STORAGE
    }
    assert credential_distinct_pairs(value) == REQUIRED_DISTINCT_PAIRS
    assert model["requester_role"]["role_id"] == REQUESTER
    assert model["requester_role"]["scope"] == "dedicated pre-account credential"
    assert model["issuer_signing_role"]["role_id"] == ISSUER_SIGNING
    assert model["pair_semantics"] == "unordered; absence of a pair never implies equality"
    assert set(model["distinct_dimensions"]) == {
        "semantic role", "credential identity",
        "cryptographic key material or non-exportable key handle", "custody role",
        "lifecycle namespace",
    }
    assert model["logical_ACLs_on_same_key_material_satisfy_separation"] is False
    assert model["future_exception_protocol_exists"] is False
    assert len(model["forbidden_alias_forms"]) == 5
    assert set(model["implementation_readiness_identity_check"]) == {
        "role credential ID", "provider namespace", "key version or key handle identity"
    }
    assert model["issuer_may_authenticate_requester_with_proof_signing_credential"] is False
    assert model["CHA_requester_obtains_issuer_signing_authority"] is False
    assert set(value["credential_role_separation"]) == {
        f"{pair[0]} != {pair[1]}" for pair in model["required_distinct_pairs"]
    }


def validate(value: dict) -> None:
    rows = {row["readiness_dependency"]: row for row in value["result_matrix"]}
    assert set(rows) == REQUIRED
    assert value["principal_result"] == (
        "ROOT_PROOF_ISSUER_PRODUCTION_SUBSTRATE_SELECTION_BLOCKED"
    )
    assert {row["selection_status"] for row in rows.values()} <= {
        "SELECTED", "DECISION_REQUIRED", "NOT_FEASIBLE"
    }
    assert rows["durable_local_attempt_storage"]["selection_status"] == "SELECTED"
    assert all(
        rows[name]["selection_status"] == "DECISION_REQUIRED"
        for name in REQUIRED - {"durable_local_attempt_storage"}
    )
    required_fields = {
        "selected_production_mechanism", "production_anchor", "authority_owner",
        "persistence_owner", "trust_owner", "credentials", "transaction_CAS_primitive",
        "crash_guarantee", "rollback_guarantee", "environment_separation",
        "implementation_interface", "remaining_prerequisite", "scope",
        "authentication_model", "restore_model", "TEST_PRODUCTION_isolation",
        "rejected_alternatives",
    }
    for row in rows.values():
        assert required_fields <= row.keys()
        assert row["selected_production_mechanism"]

    separation = value["authority_separation"]
    assert separation["rule"] == "PHYSICAL COLOCATION != AUTHORITY REUSE"
    assert separation["CHA_direct_registry_write"] is False
    assert separation["issuer_direct_account_commit"] is False
    checkpoint = value["checkpoint"]
    assert checkpoint["storage"] == "outside DB/issuer-key/CHA rollback domains"
    assert checkpoint["monotonic_dimension"] == "authority sequence/counter, never wall clock"
    assert value["coordinated_rollback"]["detected"] is False
    assert "no anti-rollback guarantee" in rows["durable_local_attempt_storage"][
        "rollback_guarantee"
    ]
    signing = rows["root_proof_signing_key_custody"]
    assert "plaintext config forbidden" in signing["credentials"]
    validate_credential_role_non_aliasing(value)
    assert value["isolation"]["TEST_may_authorize_PRODUCTION"] is False
    assert value["isolation"]["config_string_sufficient"] is False
    cas = rows["global_serialization_CAS"]
    assert "SERIALIZABLE" in cas["transaction_CAS_primitive"]
    assert "affected_rows=1" in cas["transaction_CAS_primitive"]
    assert "last-write-wins" in cas["rejected_alternatives"]
    assert value["reconciliation"]["NOT_FOUND"] == "OUTCOME_UNKNOWN"
    assert value["trust_root"]["forbidden"] == [
        "TOFU", "candidate-carried root", "CHA self-installation",
        "account-scoped bootstrap", "TEST root authorizing PRODUCTION",
    ]
    assert value["registry_schema_boundary"]["primary_authority_key"].endswith(
        "never caller selected"
    )
    assert value["migrations"]["unknown_schema"] == (
        "fail closed; never implicit upgrade"
    )
    assert value["split_brain"]["timestamp_arbitration"] is False
    least = value["least_privilege"]
    assert "never bind registry" in least["CHA"]
    assert "never commit account" in least["Issuer"]
    edges = value["topology"]["edges"]
    issuer_to_attempt = next(
        edge for edge in edges
        if edge["from"] == "Issuer" and edge["to"] == "CHA Attempt Store"
    )
    assert issuer_to_attempt["write"] is False
    outage = {row["condition"]: row for row in value["outage_matrix"]}
    assert len(outage) == 15
    for row in outage.values():
        assert not any(row[action] for action in ("ISSUE", "RECOVER", "REPLACE", "PREPARED"))
    assert value["restore"]["ambiguous_policy"] == {
        "ISSUE": False, "RECOVER": False, "REPLACE": False, "PREPARED": False
    }
    assert value["status"]["implementation_allowed"] is False
    if any(row["selection_status"] == "DECISION_REQUIRED" for row in rows.values()):
        assert value["status"]["production_substrate_selected"] is False
        assert value["status"]["implementation_allowed"] is False
    validate_lifecycle_and_compromise(value)
    assert set(value["redteam_mutations"]) == MUTATIONS


def mutate(value: dict, name: str) -> dict:
    result = deepcopy(value)
    rows = {row["readiness_dependency"]: row for row in result["result_matrix"]}
    outage = {row["condition"]: row for row in result["outage_matrix"]}
    if name == "CHA_directly_mutates_entitlement_registry":
        result["authority_separation"]["CHA_direct_registry_write"] = True
    elif name == "issuer_directly_commits_account":
        result["authority_separation"]["issuer_direct_account_commit"] = True
    elif name == "checkpoint_same_rollback_domain_without_independent_anchor":
        result["checkpoint"]["storage"] = "issuer registry database"
    elif name == "ordinary_DB_durability_claimed_as_anti_rollback":
        rows["durable_local_attempt_storage"]["rollback_guarantee"] = "DB durability"
    elif name == "root_proof_signing_key_plaintext_config":
        rows["root_proof_signing_key_custody"]["credentials"] = "plaintext config"
    elif name in ALIAS_MUTATION_PAIRS:
        forbidden_pair = ALIAS_MUTATION_PAIRS[name]
        pairs = result["credential_role_non_aliasing"]["required_distinct_pairs"]
        removed = next(pair for pair in pairs if frozenset(pair) == forbidden_pair)
        pairs.remove(removed)
        result["credential_role_separation"].remove(f"{removed[0]} != {removed[1]}")
    elif name == "TEST_namespace_reused_for_PRODUCTION":
        result["isolation"]["TEST_may_authorize_PRODUCTION"] = True
    elif name == "process_local_CAS_selected_for_multi_host_registry":
        rows["global_serialization_CAS"]["transaction_CAS_primitive"] = "process mutex"
    elif name == "last_write_wins_selected_as_BIND":
        rows["global_serialization_CAS"]["rejected_alternatives"].remove("last-write-wins")
    elif name == "checkpoint_unavailable_but_issuance_allowed":
        outage["checkpoint unavailable"]["ISSUE"] = True
    elif name == "registry_unavailable_but_signing_only_fallback_allowed":
        outage["registry DB unavailable"]["ISSUE"] = True
    elif name == "NOT_FOUND_used_as_authoritative_UNBOUND":
        result["reconciliation"]["NOT_FOUND"] = "AUTHORITATIVELY_UNBOUND"
    elif name == "restored_BOUND_to_UNBOUND_state_accepted":
        result["restore"]["ambiguous_policy"]["ISSUE"] = True
    elif name == "restored_REVOKED_to_ACTIVE_state_accepted":
        result["restore"]["ambiguous_policy"]["PREPARED"] = True
    elif name == "CHA_attempt_store_shares_authority_write_role_with_issuer_registry":
        next(edge for edge in result["topology"]["edges"] if edge["from"] == "Issuer" and edge["to"] == "CHA Attempt Store")["write"] = True
    elif name == "unknown_schema_auto_migrated":
        result["migrations"]["unknown_schema"] = "auto upgrade"
    elif name == "caller_selects_entitlement_record":
        result["registry_schema_boundary"]["primary_authority_key"] = "caller selected"
    elif name == "local_timestamp_used_to_resolve_split_brain":
        result["split_brain"]["timestamp_arbitration"] = True
    elif name == "implementation_allowed_with_DECISION_REQUIRED_dependency":
        result["status"]["implementation_allowed"] = True
    elif name.startswith("compromised_") and name.endswith("_becomes_VERIFY_ONLY"):
        role = {
            "compromised_issuer_key_becomes_VERIFY_ONLY": "issuer_signing",
            "compromised_requester_key_becomes_VERIFY_ONLY": "requester",
            "compromised_claimant_key_becomes_VERIFY_ONLY": "claimant",
            "compromised_deployment_root_becomes_VERIFY_ONLY": "deployment_trust_root",
            "compromised_checkpoint_key_becomes_VERIFY_ONLY": "checkpoint",
        }[name]
        result["credential_lifecycle"]["roles"][role]["resulting_state"] = "VERIFY_ONLY"
    elif name == "REVOKED_to_VERIFY_ONLY_allowed":
        result["credential_lifecycle"]["REVOKED_to_VERIFY_ONLY_allowed"] = True
    elif name == "REVOKED_to_ACTIVE_allowed_after_compromise":
        result["credential_lifecycle"]["REVOKED_to_ACTIVE_allowed"] = True
    elif name == "compromised_key_signature_alone_establishes_historical_acceptance":
        result["credential_lifecycle"]["historical_proof_for_REVOKED"][
            "compromised_key_signature_alone_sufficient"
        ] = True
    elif name == "compromised_checkpoint_self_attests_recovery":
        result["credential_lifecycle"]["checkpoint_interaction"][
            "compromised_checkpoint_self_attests_recovery"
        ] = True
    elif name == "restored_REVOKED_to_VERIFY_ONLY_state_accepted":
        result["restore"]["REVOKED_to_VERIFY_ONLY_restore_allowed"] = True
    else:  # pragma: no cover
        raise AssertionError(name)
    return result


def test_contract_and_deterministic_projection() -> None:
    value = load()
    validate(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


@pytest.mark.parametrize("mutation", sorted(MUTATIONS))
def test_redteam_mutation_is_rejected(mutation: str) -> None:
    with pytest.raises(AssertionError):
        validate(mutate(load(), mutation))


@pytest.mark.parametrize("mutation", sorted(LIFECYCLE_MUTATIONS))
def test_lifecycle_mutation_reaches_compromise_validator(mutation: str) -> None:
    with pytest.raises(AssertionError):
        validate_lifecycle_and_compromise(mutate(load(), mutation))


def test_required_credential_role_pairs_are_explicitly_distinct() -> None:
    validate_credential_role_non_aliasing(load())


@pytest.mark.parametrize("mutation", sorted(ALIAS_MUTATION_PAIRS))
def test_credential_alias_mutation_reaches_exact_validator(mutation: str) -> None:
    with pytest.raises(AssertionError):
        validate_credential_role_non_aliasing(mutate(load(), mutation))


@pytest.mark.parametrize("mutation", sorted(ALIAS_MUTATION_PAIRS))
def test_credential_alias_mutation_changes_exactly_its_named_pair(mutation: str) -> None:
    original = credential_distinct_pairs(load())
    changed = credential_distinct_pairs(mutate(load(), mutation))
    assert original - changed == {ALIAS_MUTATION_PAIRS[mutation]}
    assert changed - original == set()


def test_requester_issuer_mutation_does_not_touch_claimant_issuer_pair() -> None:
    changed = credential_distinct_pairs(
        mutate(load(), "requester_key_reused_as_issuer_signing_key")
    )
    assert frozenset((REQUESTER, ISSUER_SIGNING)) not in changed
    assert frozenset((CLAIMANT, ISSUER_SIGNING)) in changed


@pytest.mark.parametrize(
    ("transition", "accepted"),
    [
        ("ACTIVE -> VERIFY_ONLY", True),
        ("ACTIVE -> REVOKED", True),
        ("VERIFY_ONLY -> REVOKED", True),
        ("ACTIVE -> VERIFY_ONLY after compromise", False),
        ("REVOKED -> VERIFY_ONLY", False),
        ("REVOKED -> ACTIVE", False),
    ],
)
def test_frozen_lifecycle_transition_matrix(transition: str, accepted: bool) -> None:
    lifecycle = load()["credential_lifecycle"]
    allowed = {
        lifecycle["planned_retirement_transition"], *lifecycle["compromise_transitions"]
    }
    assert (transition in allowed) is accepted


def test_revoked_historical_proof_requires_independent_evidence() -> None:
    proof = load()["credential_lifecycle"]["historical_proof_for_REVOKED"]
    assert proof["compromised_key_signature_alone_sufficient"] is False
    assert proof["independent_trusted_evidence_required"] is True
    assert proof["acceptance_changes_lifecycle_state"] is False


def test_retired_snapshot_after_revocation_is_fail_closed() -> None:
    restore = load()["restore"]
    assert restore["REVOKED_to_VERIFY_ONLY_restore_allowed"] is False
    assert restore["restored_VERIFY_ONLY_after_later_REVOKED"] == (
        "ROLLBACK_OR_TAMPER / FAIL_CLOSED"
    )
