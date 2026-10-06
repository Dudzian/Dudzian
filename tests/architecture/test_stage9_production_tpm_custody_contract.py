"""Subordinate freeze guards preserve the existing top-level architecture hash."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jsonschema
import pytest

from bot_core.licensing import production_tpm_custody as module
from bot_core.licensing.tpm_attestation import CERTIFY_QUALIFYING_DOMAIN, POP_DOMAIN

DOCS = Path(__file__).parents[2] / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = DOCS / "stage9_production_tpm_custody_contract.json"
FREEZE = DOCS / "stage9_production_tpm_custody_freeze.json"
PARENT_SHA = "6538a75d8cf3a6e3cdab8f4a9022277a21d98e1a61923aa8d8eae9b80ae1d67d"


def test_production_custody_subordinate_freeze_and_unchanged_parent():
    contract, freeze = json.loads(CONTRACT.read_bytes()), json.loads(FREEZE.read_bytes())
    assert contract["status"] == freeze["status"] == "FROZEN"
    assert contract["implementation_authorized"] is True
    assert contract["production_provisioning_ready"] is False
    assert freeze["artifacts"][0]["sha256"] == hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
    assert freeze["parent_sha256"] == contract["parent_contract"]["sha256"] == PARENT_SHA
    assert (
        hashlib.sha256(
            (DOCS / "stage9_external_provisioning_architecture_contract.json").read_bytes()
        ).hexdigest()
        == PARENT_SHA
    )
    assert contract["state"] == {
        "stage_9": "IN_PROGRESS",
        "production_provisioning_ready": False,
        "windows_production_ready": "NOT_READY",
        "stage_10_production_lifecycle_live": "NOT_STARTED",
        "stage_10_prerequisite": "BLOCKED_UNTIL_LEGAL_ENROLLMENT",
        "legal_production_enrollment": "NOT_PERFORMED",
    }


def test_production_custody_exact_machine_checkable_wire_fields_and_domains():
    contract = json.loads(CONTRACT.read_bytes())
    endorsement_schema = contract["endorsement"]["wire_schema"]
    custody_schema = contract["custody"]["wire_schema"]
    for schema in (endorsement_schema, custody_schema):
        jsonschema.Draft202012Validator.check_schema(schema)
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"extra": "field"}, schema)
    assert (
        set(endorsement_schema["properties"]["payload"]["properties"]) == module.ENDORSEMENT_FIELDS
    )
    assert set(custody_schema["properties"]) == module.CUSTODY_FIELDS
    assert contract["endorsement"]["signature_domain_hex"] == module.ENDORSEMENT_DOMAIN.hex()
    assert (
        contract["production_exchange"]["creation_domain_hex"]
        == module.EXCHANGE_CREATION_DOMAIN.hex()
    )
    assert contract["production_exchange"]["pop_domain_hex"] == module.EXCHANGE_POP_DOMAIN.hex()
    assert contract["custody"]["creation_domain_hex"] == module.CUSTODY_CREATION_DOMAIN.hex()
    assert module.EXCHANGE_CREATION_DOMAIN != CERTIFY_QUALIFYING_DOMAIN
    assert module.EXCHANGE_POP_DOMAIN != POP_DOMAIN
    assert module.CUSTODY_CREATION_DOMAIN not in {
        CERTIFY_QUALIFYING_DOMAIN,
        module.EXCHANGE_CREATION_DOMAIN,
    }
    assert contract["creation_attestation"]["type"] == 0x801A
    assert contract["creation_attestation"]["signature"]["algorithm"] == 0x18
    assert contract["production_projection"]["evidence_profile"] == module.PROJECTION_PROFILE
    assert (
        contract["production_projection"]["source"]["profile"] == module.PROJECTION_SOURCE_PROFILE
    )
    assert "no digest cycle" in contract["custody"]["evidence_reference_semantics"]
    assert "not an independent local OEM chain" in contract["endorsement"]["hardware_trust_model"]
    assert contract["tpm_public"]["attributes"]["accepted_exact_values"] == {
        "pre_enrollment": [0x40072, 0x40472],
        "ak": [0x50072, 0x50472],
        "k_psa": [0x400B2, 0x404B2],
        "ek": [0x300B2, 0x304B2],
    }
