"""Wykonywalny kontrakt readiness produkcyjnego niezależnego root-proof issuer."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
STEM = "m05_account_genesis_root_proof_issuer_implementation_readiness_contract"
MACHINE = DOCS / f"{STEM}.json"
MARKDOWN = DOCS / f"{STEM}.md"
CHECKPOINT = "history_checkpoint_or_anti_rollback"
ISOLATION = "TEST_PRODUCTION_separation"
REQUIRED = {
    "deployment_trust_root_provider",
    "entitlement_registry_backend",
    "claimant_identity_registry",
    "requester_credential_registry",
    "root_proof_signing_key_custody",
    "global_serialization_CAS",
    "retained_authenticated_history",
    CHECKPOINT,
    "reconciliation_evidence_source",
    "durable_local_attempt_storage",
    ISOLATION,
}
EXPECTED_DISCOVERY = {
    "issuer_registry_backend": "NOT_AVAILABLE",
    "entitlement_registry_backend": "NOT_AVAILABLE",
    "claimant_identity_registry": "NOT_AVAILABLE",
    "requester_credential_registry": "NOT_AVAILABLE",
    "root_proof_signing_key_custody": "NOT_AVAILABLE",
    "deployment_trust_root_provider": "NOT_AVAILABLE",
    "global_serialization_CAS": "NOT_AVAILABLE",
    "retained_authenticated_history": "NOT_AVAILABLE",
    CHECKPOINT: "NOT_FOUND",
    "reconciliation_evidence_source": "NOT_AVAILABLE",
    "durable_local_attempt_storage": "NOT_AVAILABLE",
    "TEST_PRODUCTION_separation": "CONDITIONAL",
    "backup_restore_security_semantics": "NOT_AVAILABLE",
    "schema_migration_support": "CONDITIONAL",
}
FAILURES = {
    "registry_unavailable",
    "registry_history_unreadable",
    "trust_root_missing",
    "unknown_issuer_key",
    "claimant_registry_unavailable",
    "requester_verifier_unavailable",
    "signing_custody_unavailable",
    "CAS_unavailable",
    "history_checkpoint_unavailable",
    "stale_restored_registry",
    "reconciliation_provider_unavailable",
    "reconciliation_evidence_unverifiable",
    "key_lifecycle_store_unavailable",
    "TEST_credential_presented_to_PRODUCTION",
    "schema_version_unknown",
    "TEST_PRODUCTION_isolation_unavailable",
}
MUTATIONS = {
    "process_local_registry_marked_production_ready",
    "caller_selected_entitlement_backend",
    "catalog_key_reused_as_issuer_key",
    "freshness_key_reused_as_root_proof_signing_key",
    "CHA_self_installs_issuer_trust_root",
    "account_scoped_identity_bootstraps_first_account",
    "TEST_trust_root_authorizes_PRODUCTION",
    "entitlement_bind_without_global_CAS",
    "last_write_wins_bound_registry",
    "BOUND_history_not_retained",
    "revoked_key_history_discarded",
    "stale_restore_BOUND_to_UNBOUND_accepted",
    "stale_restore_REVOKED_to_ACTIVE_accepted",
    "reconciliation_from_NOT_FOUND",
    "caller_boolean_used_as_UNBOUND",
    "unsigned_untrusted_reconciliation_record",
    "process_memory_used_as_authoritative_history",
    "claimant_key_missing_but_issuance_allowed",
    "requester_verifier_missing_but_issuance_allowed",
    "signing_custody_missing_but_issuance_allowed",
    "unknown_schema_migrated_implicitly",
    "production_implementation_allowed_with_missing_provider",
    "root_proof_semantics_reopened_without_exploit",
    "WorkspaceAuthority_unblocked_by_readiness_only",
    "M08_unblocked_by_readiness_only",
    "implementation_allowed_with_checkpoint_missing",
    "production_dependencies_true_with_checkpoint_missing",
    "retained_history_available_without_required_rollback_protection",
    "checkpoint_not_in_required_provider_gate",
    "provider_classification_available_but_mechanism_unselected",
    "implementation_allowed_with_one_required_provider_unselected",
    "implementation_allowed_with_TEST_PRODUCTION_separation_CONDITIONAL",
    "production_dependencies_true_without_environment_isolation",
    "environment_isolation_mechanism_unselected",
    "TEST_credential_allowed_to_PRODUCTION",
    "trust_root_namespace_shared_TEST_PRODUCTION",
    "entitlement_namespace_shared_TEST_PRODUCTION",
    "claimant_namespace_shared_TEST_PRODUCTION",
    "requester_namespace_shared_TEST_PRODUCTION",
    "root_proof_signing_namespace_shared_TEST_PRODUCTION",
    "history_checkpoint_namespace_shared_TEST_PRODUCTION",
    "all_ready_fixture_forgets_to_ready_environment_isolation",
    "isolation_TEST_HELPER_marked_production_ready",
    "isolation_PROCESS_LOCAL_marked_production_ready",
    "isolation_CONFIG_PLACEHOLDER_marked_production_ready",
    "isolation_DOCUMENTATION_ONLY_marked_production_ready",
    "isolation_INTERFACE_ONLY_marked_production_ready",
    "isolation_owner_test_fixture_accepted",
    "isolation_owner_current_process_accepted",
    "isolation_owner_caller_selected_accepted",
    "isolation_owner_account_scoped_accepted",
    "isolation_status_not_production_ready_but_gate_passes",
}


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    return (
        "# M0.5 Independent Root-Proof Issuer — production substrate / "
        "implementation readiness\n\nTen plik jest deterministyczną, kompletną "
        f"projekcją `{STEM}.json`. JSON jest źródłem prawdy.\n\n"
        f"```json\n{body}```\n"
    )


def provider_satisfied(value: dict, name: str) -> bool:
    row = value["discovery_results"][name]
    if name == ISOLATION:
        return isolation_provider_satisfied(value)
    properties = row["mechanism_properties"]
    return (
        row["classification"] == "AVAILABLE"
        and row["production_mechanism_selected"] is True
        and properties["authority_backend_kind"]
        not in {"NONE_SELECTED", "PROCESS_LOCAL", "TEST_HELPER", "CONFIG_PLACEHOLDER"}
        and properties["durability_scope"] == "DURABLE"
        and properties["serialization_scope"] == "MULTI_HOST"
        and properties["multi_host_safe"] is True
    )


ISOLATION_FLAGS = {
    "environment_namespace_enforced",
    "trust_domain_namespace_enforced",
    "trust_root_namespace_separated",
    "entitlement_namespace_separated",
    "claimant_namespace_separated",
    "requester_namespace_separated",
    "issuer_signing_key_namespace_separated",
    "history_checkpoint_namespace_separated",
    "TEST_credentials_rejected_by_PRODUCTION",
}


def isolation_qualification_without_status(value: dict) -> bool:
    isolation = value["environment_isolation"]
    row = value["discovery_results"][ISOLATION]
    mechanism = isolation["mechanism_requirements"]
    owners = isolation["credential_namespace_owner_requirements"]
    return (
        row["classification"] == "AVAILABLE"
        and row["production_mechanism_selected"] is True
        and isolation["mechanism_kind"] in mechanism["production_valid_kinds"]
        and isolation["mechanism_kind"] not in mechanism["production_invalid_kinds"]
        and isolation["credential_namespace_owner_kind"] in owners["production_valid_owner_kinds"]
        and isolation["credential_namespace_owner_kind"]
        not in owners["production_invalid_owner_kinds"]
        and isolation["credential_namespace_owner"] != "NOT_SELECTED"
        and all(isolation[field] is True for field in ISOLATION_FLAGS)
        and isolation["TEST_may_authorize_PRODUCTION"] is False
    )


def isolation_provider_satisfied(value: dict) -> bool:
    isolation = value["environment_isolation"]
    return (
        isolation_qualification_without_status(value) and isolation["status"] == "PRODUCTION_READY"
    )


def environment_isolation_satisfied(value: dict) -> bool:
    return isolation_provider_satisfied(value)


def validate_environment_isolation(value: dict) -> None:
    isolation = value["environment_isolation"]
    assert isolation["mandatory_implementation_dependency"] is True
    assert isolation["plain_runtime_environment_string_sufficient"] is False
    mechanism = isolation["mechanism_requirements"]
    assert set(mechanism["production_invalid_kinds"]) >= {
        "NONE_SELECTED",
        "TEST_HELPER",
        "PROCESS_LOCAL",
        "CONFIG_PLACEHOLDER",
        "DOCUMENTATION_ONLY",
        "INTERFACE_ONLY",
    }
    assert mechanism["database_serialization_or_durability_semantics_required"] is False
    owners = isolation["credential_namespace_owner_requirements"]
    assert set(owners["production_invalid_owner_kinds"]) >= {
        "NOT_SELECTED",
        "TEST_FIXTURE",
        "CURRENT_PROCESS",
        "CALLER_SELECTED",
        "ACCOUNT_GENESIS_CANDIDATE",
        "ACCOUNT_SCOPED",
        "TEST_ONLY_AUTHORITY",
    }
    assert owners["owner_must_resolve_to_pre_account_boundary"] is True
    semantics = isolation["status_semantics"]
    assert semantics == {
        "derived_from_readiness_state": True,
        "ready_value": "PRODUCTION_READY",
        "not_ready_value": "CONDITIONAL_NOT_PRODUCTION_READY",
        "satisfied_iff_status_is_ready": True,
    }
    expected_status = (
        "PRODUCTION_READY"
        if isolation_qualification_without_status(value)
        else "CONDITIONAL_NOT_PRODUCTION_READY"
    )
    assert isolation["status"] == expected_status
    assert set(isolation["provider_linkage"]) == {
        "deployment_trust_root_provider",
        "entitlement_registry_backend",
        "claimant_identity_registry",
        "requester_credential_registry",
        "root_proof_signing_key_custody",
        "retained_authenticated_history",
        CHECKPOINT,
    }
    for provider, field in isolation["provider_linkage"].items():
        if value["discovery_results"][provider]["classification"] == "AVAILABLE":
            assert isolation[field] is True


def validate_production_gate(value: dict) -> None:
    gate = value["implementation_gate"]
    assert set(value["required_provider_gate"]) == REQUIRED
    assert gate["required_dependency_predicate"] == {
        "classification_must_equal": "AVAILABLE",
        "production_mechanism_selected_must_equal": True,
        "unresolved_CONDITIONAL_satisfies_gate": False,
        "NOT_AVAILABLE_satisfies_gate": False,
        "NOT_FOUND_satisfies_gate": False,
    }
    assert (
        gate[
            "production_dependencies_available_true_iff_all_required_dependencies_satisfy_predicate"
        ]
        is True
    )
    assert gate["implementation_allowed_requires_production_dependencies_available"] is True
    assert gate["production_dependencies_require_environment_isolation_satisfied"] is True
    assert gate["implementation_allowed_requires_environment_isolation_satisfied"] is True
    assert gate["non_production_implementation_mode"] == "NONE_FROZEN"
    all_ready = all(provider_satisfied(value, name) for name in REQUIRED)
    isolation_ready = environment_isolation_satisfied(value)
    status = value["status"]
    if status["production_dependencies_available"]:
        assert all_ready and isolation_ready
    if status["implementation_allowed"]:
        assert status["production_dependencies_available"] and all_ready and isolation_ready

    role = value["issuer_registry_backend_role"]
    assert role["role"] == "UMBRELLA_CAPABILITY_LABEL"
    assert role["authoritative_record_backend"] == "entitlement_registry_backend"
    assert role["independently_required_gate_dependency"] is False
    assert "issuer_registry_backend" not in REQUIRED

    history = value["retained_authenticated_history"]
    assert history["anti_rollback_requirement_satisfied_by"] == CHECKPOINT
    assert history["ordinary_durability_satisfies_anti_rollback"] is False
    assert history["production_available_requires_anti_rollback_capability_AVAILABLE"] is True
    if (
        value["discovery_results"]["retained_authenticated_history"]["classification"]
        == "AVAILABLE"
    ):
        assert provider_satisfied(value, CHECKPOINT)


def validate_registry_semantics(value: dict) -> None:
    registry = value["globally_serialized_entitlement_registry"]
    assert registry["atomic_conditional_bind_required"] is True
    assert registry["last_write_wins_allowed"] is False
    assert registry["multi_host_serialization_required"] is True
    assert {"process-local mutex authority", "last-write-wins overwrite"} <= set(
        registry["forbidden"]
    )


def validate_history_semantics(value: dict) -> None:
    history = value["retained_authenticated_history"]
    assert {"BOUND decision", "reconciliation evidence"} <= set(history["records"])
    assert history["key_lifecycle_history_required"] is True
    assert history["revocation_supersession_lineage_required"] is True
    assert history["historical_verification_after_rotation_or_revocation_required"] is True
    assert set(history["forbidden_authority_sources"]) >= {
        "process memory",
        "caller data",
        "unauthenticated cache / projection",
    }
    assert history["process_memory_authoritative"] is False


def validate_reconciliation_provider(value: dict) -> None:
    evidence = value["reconciliation_evidence_provider"]
    assert evidence["NOT_FOUND_proves_UNBOUND"] is False
    assert evidence["caller_assertion_proves_UNBOUND"] is False
    assert evidence["unauthenticated_record_accepted"] is False


def validate_restore_semantics(value: dict) -> None:
    restore = value["backup_restore"]
    assert restore["BOUND_to_UNBOUND_restore_allowed"] is False
    assert restore["REVOKED_to_ACTIVE_restore_allowed"] is False
    assert restore["status"] == "FAIL_CLOSED_WITHOUT_TRUSTED_CHECKPOINT"


def validate_status_preservation(value: dict) -> None:
    preserved = value["status_preservation"]
    assert preserved["RootProofIssuer_implementation_allowed"] is False
    assert preserved["FreshnessAuthority_implementation_allowed"] is False
    assert preserved["CryptoHunterAccountAuthority_implementation_allowed"] is False
    assert preserved["WorkspaceAuthority"] == "NOT_AVAILABLE"
    assert preserved["FullFillAuthority"] == "NOT_AVAILABLE"
    assert preserved["production_M05"] == "BLOCKED / NOT_AVAILABLE"
    assert preserved["M08"] == "BLOCKED / NOT_AVAILABLE"
    production = value["production_M05_gate"]
    assert production["readiness_freeze_makes_production_M05_available"] is False
    assert production["issuer_implementation_allowed_alone_is_sufficient"] is False
    assert set(production["additional_separately_frozen_prerequisites"]) == {
        "FreshnessAuthority",
        "CryptoHunterAccountAuthority",
    }
    assert production["current_production_M05_available"] is False


def validate_invariants(value: dict) -> None:
    validate_environment_isolation(value)
    validate_production_gate(value)
    validate_registry_semantics(value)
    validate_history_semantics(value)
    validate_reconciliation_provider(value)
    validate_restore_semantics(value)
    validate_status_preservation(value)
    assert value["bootstrap_entitlement_provisioning"]["identity"] == (
        "authority-generated unpredictable entitlement_id; caller cannot select it or backend"
    )
    assert (
        value["bootstrap_entitlement_provisioning"]["AccountGenesis_may_create_entitlement"]
        is False
    )
    assert set(value["no_authority_reuse"]) == {
        "Catalog key as issuer/signing key",
        "Freshness proposer key as root-proof signing key",
        "claimant key as requester/signing key",
        "storage key as protocol credential",
        "CHA-installed issuer trust root",
    }
    assert set(value["trust_root_bootstrap"]["forbidden"]) >= {
        "TOFU",
        "Catalog authority reuse",
        "FreshnessAuthority reuse",
        "CHA self-authorization",
        "account-scoped DeviceInstallation/Operator/Workspace root",
        "TEST root authorizes PRODUCTION",
    }
    assert value["environment_isolation"]["TEST_may_authorize_PRODUCTION"] is False
    assert value["schema_migrations"]["implicit_migration_allowed"] is False
    assert value["frozen_semantics_change_rule"] == (
        "DO_NOT_REOPEN_WITHOUT_CONCRETE_EXPLOIT_OR_CONTRADICTION"
    )
    assert value["mutation_policy"]["mutation_semantic_fidelity_required"] is True
    assert value["mutation_policy"]["JSON_mutation_names_equal_Python_names"] is True

    rows = {row["condition"]: row for row in value["fail_closed_matrix"]}
    assert set(rows) == FAILURES
    for row in rows.values():
        assert not any(
            (
                row["issuance_allowed"],
                row["historical_recovery_allowed"],
                row["replacement_allowed"],
                row["PREPARED_allowed"],
            )
        )


def validate_current_discovery_snapshot(value: dict) -> None:
    assert {
        name: row["classification"] for name, row in value["discovery_results"].items()
    } == EXPECTED_DISCOVERY
    assert all(
        not row["production_mechanism_selected"] for row in value["discovery_results"].values()
    )
    assert value["principal_readiness_result"] == (
        "ROOT_PROOF_ISSUER_IMPLEMENTATION_READINESS_CAN_BE_FROZEN"
    )
    assert value["provenance"] == {
        "classification": "UNKNOWN",
        "finding_scope": "CURRENT_TREE_ONLY",
        "formal_project_advancement": "WITHHELD",
    }
    assert value["status"] == {
        "semantic_contract_frozen": True,
        "semantic_result": "ACCOUNT_GENESIS_ROOT_PROOF_ISSUER_CONTRACT_CAN_BE_FROZEN",
        "implementation_readiness_frozen": True,
        "production_dependencies_available": False,
        "implementation_allowed": False,
        "production_M05_available": False,
    }


def validate(value: dict, *, current_snapshot: bool = True) -> None:
    validate_invariants(value)
    if current_snapshot:
        validate_current_discovery_snapshot(value)
        assert set(value["redteam_mutations"]) == MUTATIONS


def make_all_ready(value: dict) -> dict:
    result = deepcopy(value)
    for name in result["required_provider_gate"]:
        row = result["discovery_results"][name]
        row["classification"] = "AVAILABLE"
        row["production_mechanism_selected"] = True
        row["mechanism_properties"] = {
            "authority_backend_kind": "PRODUCTION_AUTHORITY",
            "serialization_scope": "MULTI_HOST",
            "durability_scope": "DURABLE",
            "multi_host_safe": True,
        }
    isolation = result["environment_isolation"]
    for field in ISOLATION_FLAGS:
        isolation[field] = True
    isolation["mechanism_kind"] = "REVIEWED_PRODUCTION_ISOLATION_BOUNDARY"
    isolation["credential_namespace_owner"] = "SYNTHETIC_PRODUCT_SECURITY_AUTHORITY"
    isolation["credential_namespace_owner_kind"] = "REVIEWED_PRODUCTION_ISOLATION_AUTHORITY"
    isolation["TEST_may_authorize_PRODUCTION"] = False
    isolation["status"] = "PRODUCTION_READY"
    return result


def make_regular_providers_ready_without_isolation(value: dict) -> dict:
    result = make_all_ready(value)
    row = result["discovery_results"][ISOLATION]
    row["classification"] = "CONDITIONAL"
    row["production_mechanism_selected"] = False
    isolation = result["environment_isolation"]
    for field in ISOLATION_FLAGS - {"TEST_credentials_rejected_by_PRODUCTION"}:
        isolation[field] = False
    isolation["credential_namespace_owner"] = "NOT_SELECTED"
    isolation["credential_namespace_owner_kind"] = "NOT_SELECTED"
    isolation["mechanism_kind"] = "NONE_SELECTED"
    isolation["status"] = "CONDITIONAL_NOT_PRODUCTION_READY"
    return result


def set_failure_allowed(value: dict, condition: str) -> None:
    row = next(row for row in value["fail_closed_matrix"] if row["condition"] == condition)
    row["issuance_allowed"] = True


def test_contract_and_deterministic_projection() -> None:
    value = load()
    validate(value)
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)


def test_future_gate_rejects_missing_checkpoint() -> None:
    future = make_all_ready(load())
    checkpoint = future["discovery_results"][CHECKPOINT]
    checkpoint["classification"] = "NOT_FOUND"
    checkpoint["production_mechanism_selected"] = False
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


def test_all_ready_synthetic_gate_is_coherent() -> None:
    future = make_all_ready(load())
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    validate_invariants(future)
    assert all(provider_satisfied(future, name) for name in REQUIRED)
    assert environment_isolation_satisfied(future)


def test_all_regular_providers_ready_without_isolation_is_rejected() -> None:
    future = make_regular_providers_ready_without_isolation(load())
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


ISOLATION_BREAKS = {
    "classification": ("discovery", "classification", "CONDITIONAL"),
    "mechanism_selection": ("discovery", "production_mechanism_selected", False),
    "environment_namespace": ("isolation", "environment_namespace_enforced", False),
    "trust_domain_namespace": ("isolation", "trust_domain_namespace_enforced", False),
    "trust_root_namespace": ("isolation", "trust_root_namespace_separated", False),
    "entitlement_namespace": ("isolation", "entitlement_namespace_separated", False),
    "claimant_namespace": ("isolation", "claimant_namespace_separated", False),
    "requester_namespace": ("isolation", "requester_namespace_separated", False),
    "issuer_signing_key_namespace": ("isolation", "issuer_signing_key_namespace_separated", False),
    "history_checkpoint_namespace": ("isolation", "history_checkpoint_namespace_separated", False),
    "TEST_credential_rejection": ("isolation", "TEST_credentials_rejected_by_PRODUCTION", False),
}


@pytest.mark.parametrize("case", sorted(ISOLATION_BREAKS))
def test_each_partial_environment_isolation_failure_is_rejected(case: str) -> None:
    future = make_all_ready(load())
    target, field, replacement = ISOLATION_BREAKS[case]
    if target == "discovery":
        future["discovery_results"][ISOLATION][field] = replacement
    else:
        future["environment_isolation"][field] = replacement
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


@pytest.mark.parametrize(
    "field",
    [
        "trust_root_namespace_separated",
        "entitlement_namespace_separated",
        "requester_namespace_separated",
        "claimant_namespace_separated",
        "issuer_signing_key_namespace_separated",
        "history_checkpoint_namespace_separated",
    ],
)
def test_ready_provider_cannot_alias_TEST_PRODUCTION_authority(field: str) -> None:
    future = make_all_ready(load())
    future["environment_isolation"][field] = False
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


@pytest.mark.parametrize(
    "mechanism_kind",
    [
        "NONE_SELECTED",
        "TEST_HELPER",
        "PROCESS_LOCAL",
        "CONFIG_PLACEHOLDER",
        "DOCUMENTATION_ONLY",
        "INTERFACE_ONLY",
    ],
)
def test_nonproduction_isolation_mechanism_kind_is_rejected(
    mechanism_kind: str,
) -> None:
    future = make_all_ready(load())
    future["environment_isolation"]["mechanism_kind"] = mechanism_kind
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


@pytest.mark.parametrize(
    "owner_kind",
    [
        "NOT_SELECTED",
        "TEST_FIXTURE",
        "CURRENT_PROCESS",
        "CALLER_SELECTED",
        "ACCOUNT_SCOPED",
        "TEST_ONLY_AUTHORITY",
    ],
)
def test_nonproduction_isolation_owner_is_rejected(owner_kind: str) -> None:
    future = make_all_ready(load())
    future["environment_isolation"]["credential_namespace_owner_kind"] = owner_kind
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


def test_namespace_flags_cannot_launder_TEST_HELPER_isolation() -> None:
    future = make_all_ready(load())
    assert all(future["environment_isolation"][field] for field in ISOLATION_FLAGS)
    future["environment_isolation"]["mechanism_kind"] = "TEST_HELPER"
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_each_missing_or_unselected_provider_rejects_implementation(missing: str) -> None:
    future = make_all_ready(load())
    row = future["discovery_results"][missing]
    row["classification"] = "NOT_FOUND" if missing == CHECKPOINT else "NOT_AVAILABLE"
    row["production_mechanism_selected"] = False
    future["status"]["production_dependencies_available"] = True
    future["status"]["implementation_allowed"] = True
    with pytest.raises(AssertionError):
        validate_invariants(future)


def mutate(value: dict, name: str) -> dict:
    result = deepcopy(value)
    if name == "process_local_registry_marked_production_ready":
        result = make_all_ready(result)
        row = result["discovery_results"]["entitlement_registry_backend"]
        row.update(classification="AVAILABLE", production_mechanism_selected=True)
        row["mechanism_properties"].update(
            authority_backend_kind="PROCESS_LOCAL",
            serialization_scope="PROCESS",
            durability_scope="MEMORY",
            multi_host_safe=False,
        )
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name == "caller_selected_entitlement_backend":
        result["bootstrap_entitlement_provisioning"]["identity"] = "caller selects backend"
    elif name == "catalog_key_reused_as_issuer_key":
        result["no_authority_reuse"].remove("Catalog key as issuer/signing key")
    elif name == "freshness_key_reused_as_root_proof_signing_key":
        result["no_authority_reuse"].remove("Freshness proposer key as root-proof signing key")
    elif name == "CHA_self_installs_issuer_trust_root":
        result["trust_root_bootstrap"]["forbidden"].remove("CHA self-authorization")
    elif name == "account_scoped_identity_bootstraps_first_account":
        result["trust_root_bootstrap"]["forbidden"].remove(
            "account-scoped DeviceInstallation/Operator/Workspace root"
        )
    elif name == "TEST_trust_root_authorizes_PRODUCTION":
        result["environment_isolation"]["TEST_may_authorize_PRODUCTION"] = True
    elif name == "entitlement_bind_without_global_CAS":
        result["globally_serialized_entitlement_registry"]["atomic_conditional_bind_required"] = (
            False
        )
    elif name == "last_write_wins_bound_registry":
        result["globally_serialized_entitlement_registry"]["last_write_wins_allowed"] = True
    elif name == "BOUND_history_not_retained":
        result["retained_authenticated_history"]["records"].remove("BOUND decision")
    elif name == "revoked_key_history_discarded":
        result["retained_authenticated_history"]["key_lifecycle_history_required"] = False
    elif name == "stale_restore_BOUND_to_UNBOUND_accepted":
        result["backup_restore"]["BOUND_to_UNBOUND_restore_allowed"] = True
    elif name == "stale_restore_REVOKED_to_ACTIVE_accepted":
        result["backup_restore"]["REVOKED_to_ACTIVE_restore_allowed"] = True
    elif name == "reconciliation_from_NOT_FOUND":
        result["reconciliation_evidence_provider"]["NOT_FOUND_proves_UNBOUND"] = True
    elif name == "caller_boolean_used_as_UNBOUND":
        result["reconciliation_evidence_provider"]["caller_assertion_proves_UNBOUND"] = True
    elif name == "unsigned_untrusted_reconciliation_record":
        result["reconciliation_evidence_provider"]["unauthenticated_record_accepted"] = True
    elif name == "process_memory_used_as_authoritative_history":
        result["retained_authenticated_history"]["process_memory_authoritative"] = True
    elif name == "claimant_key_missing_but_issuance_allowed":
        set_failure_allowed(result, "claimant_registry_unavailable")
    elif name == "requester_verifier_missing_but_issuance_allowed":
        set_failure_allowed(result, "requester_verifier_unavailable")
    elif name == "signing_custody_missing_but_issuance_allowed":
        set_failure_allowed(result, "signing_custody_unavailable")
    elif name == "unknown_schema_migrated_implicitly":
        result["schema_migrations"]["implicit_migration_allowed"] = True
    elif name in {
        "production_implementation_allowed_with_missing_provider",
        "implementation_allowed_with_one_required_provider_unselected",
    }:
        result = make_all_ready(result)
        result["discovery_results"]["claimant_identity_registry"][
            "production_mechanism_selected"
        ] = False
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name == "root_proof_semantics_reopened_without_exploit":
        result["frozen_semantics_change_rule"] = "REOPEN"
    elif name == "WorkspaceAuthority_unblocked_by_readiness_only":
        result["status_preservation"]["WorkspaceAuthority"] = "AVAILABLE"
    elif name == "M08_unblocked_by_readiness_only":
        result["status_preservation"]["M08"] = "AVAILABLE"
    elif name in {
        "implementation_allowed_with_checkpoint_missing",
        "production_dependencies_true_with_checkpoint_missing",
    }:
        result = make_all_ready(result)
        row = result["discovery_results"][CHECKPOINT]
        row.update(classification="NOT_FOUND", production_mechanism_selected=False)
        result["status"]["production_dependencies_available"] = True
        result["status"]["implementation_allowed"] = (
            name == "implementation_allowed_with_checkpoint_missing"
        )
    elif name == "retained_history_available_without_required_rollback_protection":
        result["discovery_results"]["retained_authenticated_history"]["classification"] = (
            "AVAILABLE"
        )
    elif name == "checkpoint_not_in_required_provider_gate":
        result["required_provider_gate"].remove(CHECKPOINT)
    elif name == "provider_classification_available_but_mechanism_unselected":
        result = make_all_ready(result)
        result["discovery_results"]["requester_credential_registry"][
            "production_mechanism_selected"
        ] = False
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name in {
        "implementation_allowed_with_TEST_PRODUCTION_separation_CONDITIONAL",
        "production_dependencies_true_without_environment_isolation",
        "all_ready_fixture_forgets_to_ready_environment_isolation",
    }:
        result = make_regular_providers_ready_without_isolation(result)
        result["status"]["production_dependencies_available"] = True
        result["status"]["implementation_allowed"] = name != (
            "production_dependencies_true_without_environment_isolation"
        )
    elif name == "environment_isolation_mechanism_unselected":
        result = make_all_ready(result)
        result["discovery_results"][ISOLATION]["production_mechanism_selected"] = False
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name == "TEST_credential_allowed_to_PRODUCTION":
        result = make_all_ready(result)
        result["environment_isolation"]["TEST_credentials_rejected_by_PRODUCTION"] = False
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name in {
        "trust_root_namespace_shared_TEST_PRODUCTION",
        "entitlement_namespace_shared_TEST_PRODUCTION",
        "claimant_namespace_shared_TEST_PRODUCTION",
        "requester_namespace_shared_TEST_PRODUCTION",
        "root_proof_signing_namespace_shared_TEST_PRODUCTION",
        "history_checkpoint_namespace_shared_TEST_PRODUCTION",
    }:
        result = make_all_ready(result)
        field_by_mutation = {
            "trust_root_namespace_shared_TEST_PRODUCTION": "trust_root_namespace_separated",
            "entitlement_namespace_shared_TEST_PRODUCTION": "entitlement_namespace_separated",
            "claimant_namespace_shared_TEST_PRODUCTION": "claimant_namespace_separated",
            "requester_namespace_shared_TEST_PRODUCTION": "requester_namespace_separated",
            "root_proof_signing_namespace_shared_TEST_PRODUCTION": (
                "issuer_signing_key_namespace_separated"
            ),
            "history_checkpoint_namespace_shared_TEST_PRODUCTION": (
                "history_checkpoint_namespace_separated"
            ),
        }
        result["environment_isolation"][field_by_mutation[name]] = False
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name in {
        "isolation_TEST_HELPER_marked_production_ready",
        "isolation_PROCESS_LOCAL_marked_production_ready",
        "isolation_CONFIG_PLACEHOLDER_marked_production_ready",
        "isolation_DOCUMENTATION_ONLY_marked_production_ready",
        "isolation_INTERFACE_ONLY_marked_production_ready",
    }:
        result = make_all_ready(result)
        kind_by_mutation = {
            "isolation_TEST_HELPER_marked_production_ready": "TEST_HELPER",
            "isolation_PROCESS_LOCAL_marked_production_ready": "PROCESS_LOCAL",
            "isolation_CONFIG_PLACEHOLDER_marked_production_ready": "CONFIG_PLACEHOLDER",
            "isolation_DOCUMENTATION_ONLY_marked_production_ready": "DOCUMENTATION_ONLY",
            "isolation_INTERFACE_ONLY_marked_production_ready": "INTERFACE_ONLY",
        }
        result["environment_isolation"]["mechanism_kind"] = kind_by_mutation[name]
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name in {
        "isolation_owner_test_fixture_accepted",
        "isolation_owner_current_process_accepted",
        "isolation_owner_caller_selected_accepted",
        "isolation_owner_account_scoped_accepted",
    }:
        result = make_all_ready(result)
        owner_by_mutation = {
            "isolation_owner_test_fixture_accepted": "TEST_FIXTURE",
            "isolation_owner_current_process_accepted": "CURRENT_PROCESS",
            "isolation_owner_caller_selected_accepted": "CALLER_SELECTED",
            "isolation_owner_account_scoped_accepted": "ACCOUNT_SCOPED",
        }
        result["environment_isolation"]["credential_namespace_owner_kind"] = owner_by_mutation[name]
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    elif name == "isolation_status_not_production_ready_but_gate_passes":
        result = make_all_ready(result)
        result["environment_isolation"]["status"] = "CONDITIONAL_NOT_PRODUCTION_READY"
        result["status"].update(production_dependencies_available=True, implementation_allowed=True)
    else:  # pragma: no cover
        raise AssertionError(name)
    return result


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_redteam_mutations_change_named_invariant_and_are_rejected(name: str) -> None:
    original = load()
    changed = mutate(original, name)
    assert changed != original
    with pytest.raises(AssertionError):
        validate_invariants(changed)
