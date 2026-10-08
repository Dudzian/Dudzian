"""Executable frozen INITIAL_BINDING architecture and production scope guards."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from bot_core.licensing import cha_account_reservation as capability
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from deployment import windows_production_cha_account_reservation as installed

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT_PATH = DOCS / "stage9_account_initial_binding_contract.json"
CONTRACT = json.loads(CONTRACT_PATH.read_bytes())
HISTORICAL_PATH = DOCS / "m05_account_genesis_operation_identity_request_binding_contract.json"
HISTORICAL = json.loads(HISTORICAL_PATH.read_bytes())
ISSUER_PATH = DOCS / "m05_account_genesis_independent_root_proof_issuer_contract.json"
ISSUER = json.loads(ISSUER_PATH.read_bytes())
SEMANTIC_MAPPING = {
    "request_schema_version": ["schema_version", "request_domain"],
    "environment/trust_domain": ["environment", "pdsa_trust_domain"],
    "requested_genesis_action": ["intended_action"],
    "product/account_scope": ["product_scope", "account_id"],
    "candidate account_id": ["account_id"],
    "reservation relationship": [
        "reservation_relation",
        "pdsa_trust_domain",
        "logical_operation_id",
        "account_id",
    ],
    "root-proof reference or handoff expectation": ["root_proof_handoff"],
}


def _validate_semantic_parity(contract):
    historical = HISTORICAL["canonical_request"]
    assert (
        historical["frozen_semantic_requirement_may_disappear_from_fingerprint_contract"] is False
    )
    required_slots = set(historical["required_semantic_slots"])
    request = contract["request"]
    mapping = request["semantic_mapping"]
    assert set(mapping) == required_slots == set(SEMANTIC_MAPPING)
    assert mapping == SEMANTIC_MAPPING
    fields = set(request["fields"])
    schema = request["json_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == fields
    assert set(request["fingerprint"]["fields"]) == fields
    for slot in required_slots:
        assert mapping[slot]
        assert set(mapping[slot]) <= fields
    properties = schema["properties"]
    assert properties["schema_version"]["const"] == request["schema_version"]
    assert properties["environment"]["const"] == "PRODUCTION"
    assert HISTORICAL["fingerprint_contract"]["environment_in_preimage"] is True
    assert HISTORICAL["fingerprint_contract"]["explicit_unique_domain_separation"] == "REQUIRED"
    assert (
        properties["request_domain"]["const"]
        == request["request_domain"]
        == HISTORICAL["fingerprint_contract"]["candidate_domain_literal"]
    )
    vector = ISSUER["root_proof_issuance_attempt_identity"]["conformance_vector"]["payload"]
    assert (
        properties["intended_action"]["const"]
        == request["intended_action"]
        == ISSUER["root_proof_object"]["intended_action"]
    )
    for field in ("product_scope", "reservation_relation"):
        assert properties[field]["const"] == request[field] == vector[field]
    assert properties["account_id"]["pattern"].startswith("^acct_")
    assert properties["logical_operation_id"]["pattern"].startswith("^ago_")
    assert properties["provisioning_operation_id"]["pattern"].startswith("^prvop_")
    handoff = properties["root_proof_handoff"]
    assert handoff["additionalProperties"] is False
    assert (
        set(handoff["required"])
        == set(handoff["properties"])
        == {
            "timing",
            "initial_binding_authenticity_model",
        }
    )
    assert (
        handoff["properties"]["timing"]["const"]
        == request["root_proof_handoff"]["timing"]
        == HISTORICAL["result"]["root_proof_timing_status"]
    )
    assert (
        handoff["properties"]["initial_binding_authenticity_model"]["const"]
        == request["root_proof_handoff"]["initial_binding_authenticity_model"]
        == ISSUER["initial_binding_authenticity"]["model"]
    )
    assert not fields & {"root_proof_id", "reservation_id", "assigned_at_utc", "timestamps"}


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
        "request_domain",
        "purpose",
        "environment",
        "pdsa_trust_domain",
        "intended_action",
        "product_scope",
        "logical_operation_id",
        "provisioning_operation_id",
        "account_id",
        "reservation_relation",
        "root_proof_handoff",
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


def test_frozen_historical_request_semantics_are_preserved():
    _validate_semantic_parity(CONTRACT)
    reconciliation = CONTRACT["historical_request_reconciliation"]
    assert reconciliation["canonical_artifact"] == HISTORICAL_PATH.relative_to(ROOT).as_posix()
    assert (
        reconciliation["sha256"]
        == hashlib.sha256(HISTORICAL_PATH.read_bytes()).hexdigest()
        == "63c8abf6847e78bea901770047fec816c252fa3ed1fbe1bc959791f1e4a0ba3d"
    )
    assert (
        reconciliation["historical_exact_schema_status"]
        == HISTORICAL["canonical_request"]["status"]
        == "SEMANTIC_REQUIREMENTS_PARTIALLY_FROZEN / EXACT_SCHEMA_DESIGN_BLOCKED"
    )
    assert (
        reconciliation["semantic_required_slots_already_frozen"]
        == HISTORICAL["canonical_request"]["required_semantic_slots"]
    )
    assert (
        reconciliation["frozen_semantic_requirement_may_disappear_from_fingerprint_contract"]
        is False
    )
    assert reconciliation["stage9_exact_schema_status"] == "FROZEN"
    handoff = reconciliation["root_proof_handoff"]
    assert handoff["canonical_artifact"] == ISSUER_PATH.relative_to(ROOT).as_posix()
    assert (
        handoff["sha256"]
        == hashlib.sha256(ISSUER_PATH.read_bytes()).hexdigest()
        == "36cd0d26f918769313c6f572541f40877b1f0ca9808a5fc69a9dac3a6bf6d30d"
    )
    assert handoff["expectation_only"] is True
    assert handoff["root_proof_present"] is False
    assert handoff["root_proof_issuance_or_admission_implemented"] is False
    assert (
        handoff["authorized_arrival_mutates_canonical_request"]
        == HISTORICAL["canonical_request"]["root_proof_handoff_progression"][
            "authorized_proof_arrival_satisfying_previously_bound_handoff_is_request_mutation"
        ]
        is False
    )


@pytest.mark.parametrize("slot", SEMANTIC_MAPPING)
def test_semantic_parity_rejects_missing_historical_slot(slot):
    changed = deepcopy(CONTRACT)
    del changed["request"]["semantic_mapping"][slot]
    with pytest.raises(AssertionError):
        _validate_semantic_parity(changed)


@pytest.mark.parametrize(
    "field", sorted({field for fields in SEMANTIC_MAPPING.values() for field in fields})
)
def test_semantic_parity_rejects_slot_disappearing_from_request_and_fingerprint(field):
    changed = deepcopy(CONTRACT)
    request = changed["request"]
    request["fields"].remove(field)
    request["fingerprint"]["fields"].remove(field)
    request["json_schema"]["required"].remove(field)
    del request["json_schema"]["properties"][field]
    with pytest.raises(AssertionError):
        _validate_semantic_parity(changed)


def test_runtime_request_matches_frozen_semantic_schema(monkeypatch):
    source_binding = canonical_json_bytes({"environment": "PRODUCTION"})
    lppi_state = canonical_json_bytes({"binding_raw_hex": source_binding.hex()})
    upstream = {
        "pdsa_trust_domain": "test-only-architecture-scope",
        "logical_operation_id": "ago_018f3e70-7b5b-7c21-8b9a-0123456789ab",
        "provisioning_operation_id": "prvop_018f3e70-7b5d-7c21-8b9a-0123456789ab",
        "lppi_operation_state_raw_hex": lppi_state.hex(),
    }
    upstream_raw = canonical_json_bytes(upstream)
    monkeypatch.setattr(installed, "_upstream_state", lambda raw: upstream)
    account_id = "acct_018f3e70-7b5c-7c21-8b9a-0123456789ab"
    request_raw = installed._canonical_request(upstream_raw, account_id)
    request = parse_canonical(request_raw)
    Draft202012Validator.check_schema(CONTRACT["request"]["json_schema"])
    Draft202012Validator(CONTRACT["request"]["json_schema"]).validate(request)
    assert set(request) == set(CONTRACT["request"]["fields"])
    assert request["account_id"] == account_id
    assert request["cha_state_sha256"] == hashlib.sha256(upstream_raw).hexdigest()
    assert request_raw == canonical_json_bytes(request)


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
