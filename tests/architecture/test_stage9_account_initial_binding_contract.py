"""Executable frozen INITIAL_BINDING architecture and production scope guards."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from bot_core.licensing import cha_account_reservation as capability
from deployment import windows_production_cha_account_reservation as installed

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_account_initial_binding_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())


def test_exact_freeze_parent_and_upstream_bytes():
    freeze = json.loads((DOCS / "stage9_account_initial_binding_freeze.json").read_bytes())
    assert freeze["status"] == CONTRACT["status"] == "FROZEN"
    assert freeze["production_provisioning_ready"] is False
    assert (
        freeze["artifacts"][0]["sha256"] == hashlib.sha256(CONTRACT_PATH.read_bytes()).hexdigest()
    )
    for name, field, expected in (
        (
            "stage9_external_provisioning_architecture_contract.json",
            "parent_contract_sha256",
            "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d",
        ),
        (
            "stage9_cha_logical_operation_contract.json",
            "upstream_contract_sha256",
            "f93964523631033a65822fa834e55326e813755376553d21ebb16eb4a047286c",
        ),
    ):
        assert hashlib.sha256((DOCS / name).read_bytes()).hexdigest() == expected
        assert freeze[field] == CONTRACT[field] == expected


def test_closed_state_and_internal_request_schema():
    assert CONTRACT["owner"] == installed.OWNER
    assert set(CONTRACT["retained_fields"]) == installed._FIELDS
    assert CONTRACT["request"]["fields"] == [
        "schema_version",
        "purpose",
        "pdsa_trust_domain",
        "logical_operation_id",
        "provisioning_operation_id",
        "cha_state_sha256",
    ]
    assert CONTRACT["recovery_model"] == "A_OPERATION_ID_BEFORE_OR_WITH_RESERVATION"
    assert CONTRACT["registry_key"] == ["pdsa_trust_domain", "logical_operation_id"]
    assert CONTRACT["reservation_id"] == "NONE"
    assert CONTRACT["state_machine"] == ["NO_ACCOUNT_RESERVATION", installed.STATUS]
    assert (
        CONTRACT["capability"]["name"] == capability.VerifiedAccountGenesisInitialBinding.__name__
    )
    assert CONTRACT["disk_rollback_protection"] == "NOT PROVIDED BY THIS LAYER"
    assert CONTRACT["account_id"]["caller_selectable"] is False
    assert CONTRACT["request"]["digest_authority"] is False
    for function in (
        installed.establish_installed_account_initial_binding,
        installed.load_installed_account_initial_binding,
    ):
        assert list(inspect.signature(function).parameters) == ["upstream"]


@pytest.mark.parametrize("module", [capability, installed])
def test_production_import_and_emission_scope(module):
    tree = ast.parse(inspect.getsource(module))
    forbidden_names = {
        "Stage9AccountGenesisAuthority",
        "ProvisioningRepository",
        "Stage9ProvisioningService",
        "ProvisioningMembershipBinding",
        "VerifiedRootProof",
        "RootProofIssuer",
        "ProtectedFreshnessAuthority",
        "SecretExternalResourcePort",
        "_uuid7",
    }
    forbidden_modules = (
        "root_proof",
        "cha_attempt_store",
        "stage10",
        "stage_10",
        "provisioning_repository",
        "provisioning_service",
        "external_provisioning",
        "account_genesis",
        "protected_freshness",
        "secret_external_resource",
        "production_enrollment_issuer",
        "production_pdsa_signing",
        "external_handoff",
    )
    forbidden_values = {
        "ACCOUNT_COMMITTED",
        "PREPARED",
        "EXTERNAL_FRESHNESS_CAS",
        "LOCAL_FINAL_COMMIT",
        "devinst_",
        "ACCOUNT_GENESIS_COMMITTED",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not any(part in (node.module or "").lower() for part in forbidden_modules)
            assert not ({alias.name for alias in node.names} & forbidden_names)
        if isinstance(node, ast.Import):
            assert not any(
                part in alias.name.lower() for alias in node.names for part in forbidden_modules
            )
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            assert node.value not in forbidden_values
    if module is installed:
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "bot_core.uuid7"
            for node in ast.walk(tree)
        )
        assert not any(
            isinstance(node, ast.FunctionDef) and node.name == "mint_uuid7"
            for node in ast.walk(tree)
        )


def test_current_status_never_promotes_readiness():
    status = json.loads((ROOT / "deployment/stage9_current_status.json").read_bytes())
    for key, expected in CONTRACT["current_status"].items():
        assert status[key] == expected
    assert status["provisioning_membership"] == "BLOCKED"
    assert status["legal_production_enrollment"] == "NOT_PERFORMED"
    assert status["windows_production_ready"] == "NOT_READY"
    assert status["protected_freshness"] == status["secret_resource"] == "NOT_STARTED"
    assert status["stage_10_production_lifecycle_live"] == "NOT_STARTED"


def test_ci_preserves_serial_security_durability_and_coverage():
    import yaml

    workflow = yaml.safe_load((ROOT / ".github/workflows/quality-security.yml").read_text())
    steps = workflow["jobs"]["quality-ratchet"]["steps"]
    serial = next(step for step in steps if step.get("name", "").endswith("(serial)"))
    assert "test_cha_account_reservation.py" in serial["run"]
    assert "test_cha_account_reservation_durability.py" in serial["run"]
    assert "--cov-append" in serial["run"]
    assert "xdist" not in serial["run"] and "-n " not in serial["run"]
    parallel = next(
        step for step in steps if step.get("name", "").startswith("Property and critical")
    )
    assert "test_cha_account_reservation.py" not in parallel["run"]
    assert "--cov=deployment.windows_production_cha_account_reservation" in serial["run"]
    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "test_cha_*.py" in ci
