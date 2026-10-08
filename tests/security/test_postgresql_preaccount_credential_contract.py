"""Offline negative paths for the verifier-side pre-account credential boundary."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import pytest

import bot_core.postgresql_preaccount_credentials as credentials
from bot_core.postgresql_entitlement_registry import PostgreSQLConnectionConfig
from bot_core.root_proof_issuer_substrate import (
    CredentialSemanticRole,
    public_key_material_identity,
)


def _config(kind: str = "requester", **changes):
    return credentials.PostgreSQLCredentialRegistryProvisioning(
        **{
            "schema": f"cryptohunter_{kind}",
            "schema_owner_role": f"{kind}_owner",
            "runtime_role": f"{kind}_runtime",
            "admin_role": f"{kind}_admin",
            "environment": "PRODUCTION",
            "trust_domain": "deployment-security",
            **changes,
        }
    )


def _generation(**changes):
    return credentials.CredentialGeneration(
        **{
            "credential_id": "retained-credential-v1",
            "principal_id": credentials.REQUESTER_PRINCIPAL,
            "semantic_role": CredentialSemanticRole.ROOT_PROOF_REQUESTER,
            "credential_role": credentials.REQUESTER_CREDENTIAL_ROLE,
            "key_id": "requester-key-v1",
            "key_version": 1,
            "public_key": b"A" * 32,
            "key_material_identity": public_key_material_identity(b"A" * 32),
            "environment": "PRODUCTION",
            "trust_domain": "deployment-security",
            "lifecycle_generation": 7,
            "lifecycle": credentials.CredentialLifecycle.VERIFY_ONLY,
            "registry_revision": 7,
            **changes,
        }
    )


@pytest.mark.parametrize("name", ["schema", "schema_owner_role", "runtime_role", "admin_role"])
@pytest.mark.parametrize("value", ["bad;DROP ROLE", "MixedCase", "", "a" * 64, 7])
def test_unreviewed_identifiers_fail_before_database_access(name, value) -> None:
    with pytest.raises(ValueError):
        _config(**{name: value})


@pytest.mark.parametrize(
    "environment", ["TEST", "PRODUCTION_LOCAL", "PRODUCTION_SERVER_READY", "", None]
)
def test_security_profile_cannot_replace_protocol_environment(environment) -> None:
    with pytest.raises(ValueError):
        _config(environment=environment)


def test_admin_runtime_and_schema_ownership_are_independent() -> None:
    with pytest.raises(ValueError):
        _config(admin_role="requester_runtime")
    with pytest.raises(ValueError):
        credentials.PostgreSQLPreaccountRegistryProvisioning(
            _config(), _config("claimant", schema="cryptohunter_requester")
        )
    with pytest.raises(ValueError):
        credentials.PostgreSQLPreaccountRegistryProvisioning(
            _config(), _config("claimant", admin_role="requester_admin")
        )
    with pytest.raises(ValueError):
        credentials.PostgreSQLPreaccountRegistryProvisioning(
            _config(), _config("claimant", trust_domain="other-domain")
        )


@pytest.mark.parametrize("key", [b"", b"A" * 31, b"A" * 33, bytearray(b"A" * 32), "A" * 32, None])
def test_only_exact_32_raw_ed25519_bytes_are_authority_material(key) -> None:
    with pytest.raises(ValueError):
        _generation(public_key=key)


def test_stored_fingerprint_label_cannot_replace_exact_key_recomputation() -> None:
    with pytest.raises(credentials.CredentialResolutionError):
        _generation(public_key=b"B" * 32)
    with pytest.raises(credentials.CredentialResolutionError):
        _generation(key_material_identity="sha256:" + "f" * 64)


@pytest.mark.parametrize("field", ["key_version", "lifecycle_generation", "registry_revision"])
@pytest.mark.parametrize(
    "value", [True, 0, -1, 1.0, float("nan"), float("inf"), 9_007_199_254_740_992]
)
def test_boolean_nonfinite_or_noninteroperable_counters_fail_closed(field, value) -> None:
    with pytest.raises(ValueError):
        _generation(**{field: value})


def test_key_version_is_independent_of_retained_lifecycle_generation() -> None:
    historical = _generation()
    assert historical.key_version == 1
    assert historical.lifecycle_generation == historical.registry_revision == 7
    assert (
        replace(historical, lifecycle=credentials.CredentialLifecycle.REVOKED).public_key
        == b"A" * 32
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("environment", "TEST"),
        ("trust_domain", ""),
        ("lifecycle", "ACTIVE"),
        ("semantic_role", "ROOT_PROOF_REQUESTER"),
        ("principal_id", None),
    ],
)
def test_malformed_namespace_and_enum_records_fail_closed(field, value) -> None:
    with pytest.raises((ValueError, TypeError)):
        _generation(**{field: value})


def test_preaccount_substrate_stops_before_private_signing_and_issuance() -> None:
    source = Path(credentials.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    forbidden = {
        "sign",
        "sign_root_proof",
        "sign_claimant",
        "sign_requester",
        "finalize_attempt",
        "AttemptIdentity",
        "send",
        "compare_and_swap_bind",
        "prepare",
        "commit_account_genesis",
    }
    assert names.isdisjoint(forbidden)
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "bot_core.local_root_proof_signing" not in imports
    assert "cryptography.hazmat.primitives.asymmetric.ed25519" not in imports
    assert not hasattr(
        credentials.PostgreSQLRequesterCredentialRegistryProvider, "provision_credential"
    )
    assert not hasattr(
        credentials.PostgreSQLClaimantIdentityRegistryProvider, "transition_lifecycle"
    )
    assert not hasattr(
        credentials.PostgreSQLRequesterCredentialProvisioningAdminProvider, "identity"
    )
    assert "user=" not in repr(PostgreSQLConnectionConfig("user=private-admin password=secret"))
