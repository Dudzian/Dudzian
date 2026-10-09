"""Subordinate public-credential freeze, historical reconciliation and scope."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from copy import deepcopy
from dataclasses import fields
from pathlib import Path

import pytest

from bot_core import (
    postgresql_preaccount_credentials as credentials,
    postgresql_root_proof_issuance_authority as composition,
    root_proof_issuer_substrate as substrate,
)
from bot_core.licensing import cha_root_proof_attempt_reservation as boundary

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_root_proof_preaccount_credentials_contract.json"
FREEZE_PATH = DOCS / "stage9_root_proof_preaccount_credentials_freeze.json"
CURRENT_PATH = ROOT / "deployment/stage9_current_status.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())
STATUS = {
    "local_cha_attempt_reservation_boundary": "IMPLEMENTED",
    "root_proof_issuance_authorization_boundary": "IMPLEMENTED",
    "requester_claimant_production_provider": "IMPLEMENTED",
    "requester_credential_registry": "IMPLEMENTED",
    "claimant_identity_registry": "IMPLEMENTED",
    "root_proof_issuance_provider_composition": "IMPLEMENTED",
    "production_preaccount_credentials": "NOT_PROVISIONED",
    "root_proof_issuance_authorization_live_availability": "BLOCKED_UNTIL_PROVISIONING",
    "semantic_root_proof_issuer_runtime": "NOT_IMPLEMENTED / BLOCKED",
    "signed_immutable_attempt": "NOT_STARTED",
    "root_proof": "NOT_ISSUED",
    "root_proof_admission": "NOT_STARTED",
    "account_id": "CANDIDATE_RESERVED_NOT_GENUINE",
    "account_genesis": "INITIAL_BINDING_ONLY",
    "prepared": "NOT_STARTED",
    "production_provisioning_ready": False,
    "windows_production_ready": "NOT_READY",
    "legal_production_enrollment": "NOT_PERFORMED",
    "stage_10_production_lifecycle_live": "NOT_STARTED",
    "provisioning_membership": "BLOCKED",
    "protected_freshness": "NOT_STARTED",
    "secret_resource": "NOT_STARTED",
    "windows_0_14": "10/15 DONE",
}


def _assert_status(contract, current):
    assert contract["current_status_authority"] == {
        "canonical_artifact": CURRENT_PATH.relative_to(ROOT).as_posix(),
        "authority": "CURRENT_STATUS",
        "child_is_global_status_authority": False,
        "conflict_policy": "FAIL",
        "parity_fields": list(STATUS),
    }
    assert current["schema"] == "CryptoHunter.Stage9CurrentStatusV1"
    assert type(current["version"]) is int and current["version"] == 1
    assert current["authority"] == "CURRENT_STATUS"
    assert set(contract["current_status"]) == set(STATUS)
    for name, expected in STATUS.items():
        assert type(contract["current_status"][name]) is type(expected), name
        assert type(current[name]) is type(expected), name
        assert contract["current_status"][name] == expected, name
        # This historical child remains byte-identical. The signed-attempt
        # child now advances only the implementation status of this boundary.
        current_expected = "IMPLEMENTED" if name == "signed_immutable_attempt" else expected
        assert current[name] == current_expected, name


def test_child_freeze_and_all_historical_inputs_keep_exact_bytes():
    freeze = json.loads(FREEZE_PATH.read_bytes())
    assert freeze["status"] == CONTRACT["status"] == "FROZEN"
    assert freeze["production_provisioning_ready"] is False
    assert freeze["implementation_architecturally_authorized"] is True
    assert freeze["artifacts"] == [
        {
            "canonical_artifact": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "canonical_schema_version": CONTRACT["schema_version"],
            "sha256": hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest(),
        }
    ]
    protected = {row["canonical_artifact"]: row["sha256"] for row in freeze["protected_artifacts"]}
    assert len(protected) == len(freeze["protected_artifacts"])
    for path, digest in protected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    for row in CONTRACT["frozen_inputs"]:
        assert row["historical_bytes_unchanged"] is True
        assert row["sha256"] == protected[row["canonical_artifact"]]
    assert freeze["upstream_contract_sha256"] == CONTRACT["frozen_inputs"][0]["sha256"]
    assert CONTRACT["historical_reconciliation"]["historical_statuses_rewritten"] is False
    previous = json.loads(
        (DOCS / "stage9_root_proof_attempt_reservation_contract.json").read_bytes()
    )
    assert previous["authorization"]["production_adapter_allowlist_empty"] is True
    assert previous["authorization"]["requester_claimant_production_adapter_availability"] == (
        "NOT_IMPLEMENTED"
    )


def test_postgresql_selection_reconciles_frozen_ports_without_authority_reuse():
    historical = json.loads(
        (
            DOCS
            / "m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json"
        ).read_bytes()
    )
    rows = {row["readiness_dependency"]: row for row in historical["result_matrix"]}
    registry = CONTRACT["postgresql"]
    assert registry["engine"] == "PostgreSQL"
    assert registry["schema_version"] == credentials.SCHEMA_VERSION
    assert registry["storage_family"] == credentials.STORAGE_FAMILY
    assert registry["requester_schema_identity"] == credentials.REQUESTER_SCHEMA_IDENTITY
    assert registry["claimant_schema_identity"] == credentials.CLAIMANT_SCHEMA_IDENTITY
    for kind, constants, dependency in (
        ("requester", "REQUESTER", "requester_credential_registry"),
        ("claimant", "CLAIMANT", "claimant_identity_registry"),
    ):
        row = registry[kind]
        assert (
            rows[dependency]["production_local_mechanism"]
            == "separate PostgreSQL schema/API/DB role"
        )
        assert rows[dependency]["implementation_interface"] == row["port"]
        for field, suffix in (
            ("schema", "SCHEMA"),
            ("schema_owner", "OWNER_ROLE"),
            ("runtime_role", "RUNTIME_ROLE"),
            ("admin_role", "ADMIN_ROLE"),
        ):
            assert row[field] == getattr(credentials, f"{constants}_{suffix}")
    roles = [
        registry[kind][name]
        for kind in ("requester", "claimant")
        for name in ("schema_owner", "runtime_role", "admin_role")
    ]
    assert len(set(roles)) == 6
    assert registry["requester"]["schema"] != registry["claimant"]["schema"]
    assert registry["PUBLIC_privileges"] == "FORBIDDEN"
    assert registry["production_internal_seam_enforces_exact_schema_and_role_triples"] is True
    assert registry["process_local_mutex_is_authority"] is False
    assert registry["unknown_schema_or_version"] == "FAIL_CLOSED_WITHOUT_RUNTIME_MIGRATION"
    assert CONTRACT["production_composition"]["new_entitlement_registry"] is False
    for name in (
        "DDL",
        "credential_provisioning",
        "credential_rotation",
        "lifecycle_write",
        "table_mutation",
        "admin_or_owner_membership",
    ):
        assert registry["runtime"][name] is False


def test_exact_dtos_and_public_lookup_are_verifier_authority_only():
    for kind, record, provider, port in (
        (
            "requester",
            boundary._RequesterCredentialV1,
            credentials.PostgreSQLRequesterCredentialRegistryProvider,
            substrate.RequesterCredentialRegistry,
        ),
        (
            "claimant",
            boundary._ClaimantIdentityV1,
            credentials.PostgreSQLClaimantIdentityRegistryProvider,
            substrate.ClaimantIdentityRegistry,
        ),
    ):
        assert CONTRACT[kind]["active_DTO"]["type"] == record.__name__
        assert CONTRACT[kind]["active_DTO"]["fields"] == [field.name for field in fields(record)]
        assert list(inspect.signature(provider.public_key).parameters) == [
            "self",
            "credential_identity",
        ]
        assert list(inspect.signature(port.public_key).parameters) == [
            "self",
            "credential_identity",
        ]
        for name in (
            "provision_credential",
            "rotate_credential",
            "transition_lifecycle",
            "sign",
            "sign_request",
            "sign_root_proof",
        ):
            assert not hasattr(provider, name)
        assert not hasattr(provider, "active_credential_identity")
        assert not hasattr(provider, "lifecycle_generation")
    material = CONTRACT["canonical_public_key"]
    assert material["algorithm"] == "Ed25519" and material["length_bytes"] == 32
    assert material["every_lookup_recomputes_fingerprint"] is True
    assert material["stored_fingerprint_is_authority"] is False
    assert material["private_key_storage_or_signing"] is False
    assert material["material_identity_function"] == (
        "bot_core.root_proof_issuer_substrate.public_key_material_identity"
    )
    assert [field.name for field in fields(substrate.CredentialRoleIdentity)] == (
        CONTRACT["credential_role_evidence"]["fields"]
    )
    assert CONTRACT["requester"]["principal"] == credentials.REQUESTER_PRINCIPAL
    assert CONTRACT["requester"]["credential_role"] == credentials.REQUESTER_CREDENTIAL_ROLE


def test_runtime_ddl_scope_and_claimant_alias_checks_do_not_overclaim_authority():
    registry = CONTRACT["postgresql"]
    assert registry["runtime"]["DDL"] is False
    assert registry["runtime"]["DDL_scope"] == (
        "Persistent database schemas and credential-authority objects"
    )
    assert registry["runtime"]["temporary_workspace"] == {
        "database_PUBLIC_TEMP": "UNCHANGED / MAY_BE_ALLOWED",
        "grants_credential_authority": False,
        "authority_lookup": (
            "Pinned pg_catalog search_path and fully qualified authority tables "
            "prevent temporary-workspace shadowing"
        ),
    }
    assert registry["PUBLIC_privileges_scope"] == (
        "Reviewed credential-authority schemas, tables and functions; "
        "database TEMP grants are outside this scope"
    )
    assert CONTRACT["claimant"]["exact_principal_alias_rejected"] == [
        "INITIAL_BINDING.account_id",
        "INITIAL_BINDING.logical_operation_id",
    ]
    assert CONTRACT["claimant"]["temporal_preaccount_principal_history_established"] is False


def test_key_versions_lifecycle_and_namespace_are_distinct():
    life = CONTRACT["lifecycle_and_history"]
    assert life["states"] == [state.value for state in credentials.CredentialLifecycle]
    assert life["key_version_is_lifecycle_generation"] is False
    assert life["REVOKED_is_terminal"] is True
    assert life["new_operation_requires"] == "ACTIVE"
    for field in (
        "historical_signature_alone_is_trust",
        "destructive_overwrite",
        "runtime_or_normal_admin_lifecycle_rollback",
    ):
        assert life[field] is False
    retained = {field.name for field in fields(credentials.CredentialGeneration)}
    assert {
        "key_version",
        "lifecycle_generation",
        "registry_revision",
        "public_key",
        "key_material_identity",
        "environment",
        "trust_domain",
        "principal_id",
    } <= retained
    assert list(
        inspect.signature(
            credentials.PostgreSQLClaimantIdentityRegistryProvider.historical_claimant
        ).parameters
    ) == ["self", "claimant_id", "generation"]
    assert list(
        inspect.signature(
            credentials.PostgreSQLRequesterCredentialRegistryProvider.historical_generation
        ).parameters
    ) == ["self", "credential_id", "generation"]
    for provider in (
        credentials.PostgreSQLRequesterCredentialProvisioningAdminProvider,
        credentials.PostgreSQLClaimantIdentityProvisioningAdminProvider,
    ):
        parameters = inspect.signature(provider.transition_lifecycle).parameters
        assert list(parameters) == [
            "self",
            "principal_id",
            "lifecycle",
            "expected_revision",
            "credential_id",
        ]
        assert parameters["credential_id"].default is None
    assert (
        "preserves current replacement ACTIVE pointer" in life["historical_credential_transition"]
    )
    namespace = CONTRACT["environment_and_profile"]
    assert namespace["semantic_environment"] == "PRODUCTION"
    assert namespace["provider_security_profile"] == "PRODUCTION_LOCAL"
    assert namespace["namespace_dimensions"] == ["environment", "trust_domain"]
    assert namespace["TEST_authorizes_PRODUCTION"] is False


def test_genuine_aggregate_allowlist_and_config_have_no_public_stage9_input():
    aggregate = composition.PostgreSQLRootProofIssuanceAuthority
    assert CONTRACT["production_composition"]["exact_type"] == aggregate.__name__
    assert boundary._TRUSTED_PROVIDER_TYPES == (aggregate,)
    assert CONTRACT["production_composition"]["deployment_config_names"] == list(
        composition._CONFIG_ENVIRONMENT
    )
    assert list(
        inspect.signature(boundary.resolve_root_proof_issuance_authorization).parameters
    ) == ["binding"]
    assert (
        list(inspect.signature(composition._configured_root_proof_issuance_authority).parameters)
        == []
    )
    with pytest.raises(TypeError):
        type("UnreviewedAggregateSubclass", (aggregate,), {})
    assert CONTRACT["production_composition"]["entitlement_BIND"] is False
    assert CONTRACT["production_composition"]["evidence_schema_changed"] is True
    assert CONTRACT["production_composition"]["stop_state"] == "RESERVED_AWAITING_SIGNATURES"
    aliases = CONTRACT["credential_role_evidence"]
    assert (
        aliases["requester_claimant_material_alias"]
        == "REJECT_AT_PROVISIONING_AND_LIVE_COMPOSITION"
    )
    assert aliases["labels_replace_raw_material_comparison"] is False
    assert aliases["global_future_role_aliases_solved_here"] is False


def test_retained_evidence_is_exact_operation_scoped_with_global_qualification_separate():
    evidence = CONTRACT["production_composition"]["authorization_evidence"]
    assert evidence["schema_version"] == "RootProofOperationScopedAuthorizationEvidenceV1"
    assert evidence["scope"] == "EXACT_OPERATION_SCOPED"
    assert evidence["closed_fields"] == [
        "schema_version",
        "scope",
        "context",
        "resolution",
        "entitlement",
        "requester",
        "claimant",
        "requester_credential",
        "claimant_credential",
        "providers",
    ]
    assert evidence["resolved_credential_fields"] == [
        "identity",
        "public_key_material_identity",
    ]
    assert evidence["providers"] == ["entitlement", "requester", "claimant"]
    assert evidence["provider_fields"] == ["identity", "capabilities"]
    assert evidence["full_registry_credential_population_in_hash"] is False
    assert (
        evidence[
            "unrelated_principal_mutations_do_not_alter_another_operations_authorization_identity"
        ]
        is True
    )
    assert evidence["unrelated_mutation_precondition"] == (
        "Global provider/substrate qualification remains valid"
    )
    assert evidence["unrelated_claimant_mutations"] == [
        "provision additional principal",
        "rotate another principal",
        "transition another principal to VERIFY_ONLY",
        "revoke another principal's retained historical credential",
    ]
    assert evidence["global_qualification_outside_hash"] == [
        "Full requester and claimant current/historically relevant credential_identities() populations",
        "Provider structure and canonical credential semantic-role/namespace evidence",
        "Global forbidden cross-role public-key material aliases",
        "Exact live PostgreSQL schema/version/owner/runtime/admin role and ACL evidence",
        "Durability and environment/trust-domain/security-profile qualification",
    ]
    assert evidence["global_qualification_failure"] == (
        "FAIL_CLOSED even when operation-scoped evidence is unchanged"
    )


def test_schema_owner_cannot_authenticate_but_retains_trusted_mutation_authority():
    authentication = CONTRACT["postgresql"]["role_authentication"]
    assert authentication["schema_owner_LOGIN"] is False
    assert authentication["runtime_LOGIN"] is True
    assert authentication["admin_LOGIN"] is True
    assert authentication["direct_schema_owner_login_required"] is False
    assert authentication["runtime_admin_owner_membership"] == "FORBIDDEN"
    assert "Supported with NOLOGIN" in authentication["SECURITY_DEFINER_owner_execution"]
    assert "SET LOCAL ROLE owner" in authentication["SECURITY_DEFINER_owner_execution"]
    assert "can bypass the admin API" in authentication["owner_trust_assumption"]
    assert "NOLOGIN removes direct authentication only" in authentication["owner_trust_assumption"]


def test_canonical_status_separates_implementation_from_provisioning():
    _assert_status(CONTRACT, json.loads(CURRENT_PATH.read_bytes()))
    assert CONTRACT["live_qualification"] == {
        "real_PostgreSQL_tests_are_production_credential_provisioning": False,
        "installed_production_credential_provisioning": "NOT_PERFORMED",
        "production_credentials": "NOT_PROVISIONED",
        "authorization_live_availability": "BLOCKED_UNTIL_PROVISIONING",
        "physical_hardware_qualification": "NOT_RUN",
        "legal_production_enrollment": "NOT_PERFORMED",
    }


@pytest.mark.parametrize("field", STATUS)
@pytest.mark.parametrize("mutation", ["missing", "different"])
def test_current_status_rejects_missing_or_conflicting_parity(field, mutation):
    current = json.loads(CURRENT_PATH.read_bytes())
    if mutation == "missing":
        current.pop(field)
    else:
        current[field] = "CONTRADICTORY_STATUS"
    with pytest.raises((AssertionError, KeyError)):
        _assert_status(CONTRACT, current)


@pytest.mark.parametrize(
    "field,value",
    [
        ("production_preaccount_credentials", "PROVISIONED"),
        ("root_proof_issuance_authorization_live_availability", "AVAILABLE"),
        ("semantic_root_proof_issuer_runtime", "IMPLEMENTED"),
        ("signed_immutable_attempt", "SIGNED_IMMUTABLE_DURABLE_NOT_SENT"),
        ("root_proof", "ISSUED"),
        ("prepared", "PREPARED"),
        ("account_genesis", "COMMITTED"),
        ("production_provisioning_ready", True),
        ("production_provisioning_ready", 0),
        ("windows_0_14", "11/15 DONE"),
    ],
)
def test_even_matching_sources_cannot_promote_unachieved_production_state(field, value):
    contract = deepcopy(CONTRACT)
    current = json.loads(CURRENT_PATH.read_bytes())
    contract["current_status"][field] = current[field] = value
    with pytest.raises(AssertionError):
        _assert_status(contract, current)


@pytest.mark.parametrize("module", [credentials, composition, boundary])
def test_public_credential_boundary_has_no_downstream_signing_or_issuer_effect(module):
    tree = ast.parse(inspect.getsource(module))
    forbidden_modules = {
        "requests",
        "httpx",
        "socket",
        "urllib",
        "root_proof_signing_custody",
        "local_signing_custody",
        "protected_freshness",
        "account_genesis",
        "stage10",
    }
    forbidden_symbols = {
        "AttemptIdentity",
        "RootProofIssuer",
        "VerifiedRootProof",
        "RootProofAdmissionEvidence",
        "LocalRootProofSigningProvider",
        "BindRequest",
        "ProtectedFreshnessAuthority",
    }
    forbidden_calls = {
        "sign",
        "sign_request",
        "sign_root_proof",
        "finalize_attempt",
        "compare_and_swap_bind",
        "issue_root_proof",
        "send",
        "commit_account",
        "prepare",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(part in (node.module or "") for part in forbidden_modules)
            assert not {name.name for name in node.names} & forbidden_symbols
        elif isinstance(node, ast.Import):
            assert not any(part in alias.name for alias in node.names for part in forbidden_modules)
        elif isinstance(node, ast.Name):
            assert node.id not in forbidden_symbols
        elif isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            assert name not in forbidden_calls
        elif isinstance(node, ast.FunctionDef):
            assert node.name not in forbidden_calls
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in {
                "SIGNED_IMMUTABLE_DURABLE_NOT_SENT",
                "PREPARED",
                "ACCOUNT_GENESIS_COMMITTED",
                "EXTERNAL_FRESHNESS_CAS",
                "rpf_",
            }
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "public_key_material_identity"
        for node in ast.walk(tree)
    )
