"""Executable review for M0.12 trust-primitive reuse by future M0.5 design."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_trust_primitives_reuse_discovery.json"
MARKDOWN = DOCS / "m05_account_genesis_trust_primitives_reuse_discovery.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 account-genesis trust-primitives reuse discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_trust_primitives_reuse_discovery.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == "M05_ACCOUNT_GENESIS_TRUST_PRIMITIVES_REUSE_DISCOVERY"
    assert value["iteration"] == "DISCOVERY / RECONCILIATION ONLY"
    assert value["reviewed_head_supplied"] == (
        "2a88c726cacebb718748b4872177e9ca639df8af"
    )
    assert value["repository_head_examined"] != value["reviewed_head_supplied"]
    provenance = value["provenance"]
    assert provenance["classification"] in {
        "EXACT_COMMIT",
        "PATH_CONTENT_EQUIVALENT",
        "HIGH_CONFIDENCE_LINEAGE_EQUIVALENT",
        "LIKELY_EQUIVALENT",
        "UNKNOWN",
        "MISMATCH",
    }
    assert provenance["reviewed_head_available_locally"] is False
    assert provenance["classification"] == "UNKNOWN"

    inventory = {item["class"]: item for item in value["m012_authentication_inventory"]}
    receipt = inventory["CatalogAdmissionReceiptAuthority"]
    assert receipt["algorithm"] == "HMAC-SHA-256"
    assert receipt["domain_separation"]["receipt_purpose"] == (
        "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_RECEIPT_V1"
    )
    assert receipt["domain_separation"]["authority_domain"] == (
        "cryptohunter.catalog-admission-receipt.production.v1"
    )
    assert inventory["SourceProducerMembershipAuthority"]["algorithm"].endswith(
        "NOT HMAC authentication"
    )
    assert inventory["CatalogRuntimeAcceptanceAuthority"]["key_owner"] == (
        "CatalogAdmissionReceiptAuthority only"
    )

    keyring = value["m012_keyring_inventory"]
    assert keyring["catalog_service_name"] == "dudzian.catalog-admission-receipt"
    assert keyring["scope"].startswith("catalog-specific")
    assert keyring["keyring_anti_rollback"].startswith("NOT_FOUND")

    freshness = value["m012_freshness_inventory"]
    assert freshness["actually_external"].startswith("External to mutable Catalog SQLite")
    assert freshness["atomicity_with_local_journal"].startswith("NOT_ATOMIC")
    assert "outside" in freshness["coordinated_rollback"]
    assert value["threat_boundary"]["authentication_is_freshness"] is False
    assert value["threat_boundary"]["machine_wide_rollback"].startswith("NOT_PROTECTED")

    isolation = value["production_test_isolation"]
    assert isolation["TEST_authentication_material_accepted_in_PRODUCTION"] is False
    assert "production.v1 versus" in isolation["authority_domain"]
    assert value["domain_separation"]["catalog_hmac_as_account_genesis_hmac"] == (
        "MUST_FAIL"
    )
    assert value["domain_separation"]["m012_record_domain_reuse"] == "FORBIDDEN"
    assert value["key_reuse_policy"]["CROSS_AUTHORITY_KEY_REUSE_POLICY"] == (
        "NOT_FROZEN"
    )
    assert value["key_rotation"]["KEY_ROTATION_SEMANTICS"].endswith(
        "account-genesis semantics NOT_FOUND"
    )

    scope = value["freshness_anchor_scope"]
    assert scope["multiple_independent_journals"] == "NOT_SUPPORTED_BY_CURRENT_INTERFACE"
    assert scope["account_lineage_direct_coverage"] is False
    cross = value["cross_authority_anchor_analysis"]
    assert cross["single_shared_anchor_sufficient"] is False
    assert cross["current_anchor_detection"].startswith("NO:")
    assert value["CAS_and_generation_reuse"]["account_genesis_result"] == (
        "REUSABLE_PATTERN_ONLY; does not directly provide "
        "operation/reservation/account_id CAS"
    )
    assert value["restart_order_parity"]["result"] == "PARTIAL_MATCH"

    matrix = {
        row["requirement"]: row["classification"]
        for row in value["account_genesis_requirement_matrix"]
    }
    assert matrix == {
        "authenticated operation state": "REUSABLE_WITH_NEW_DOMAIN",
        "authenticated reservation state": "REUSABLE_WITH_NEW_DOMAIN",
        "authenticated predecessor": "REUSABLE_WITH_NEW_DOMAIN",
        "expected generation / CAS": "REUSABLE_PATTERN_ONLY",
        "valid-prefix rollback rejection": "REUSABLE_PATTERN_ONLY",
        "freshness verification before resolver publication": "REUSABLE_PATTERN_ONLY",
        "TEST/PRODUCTION isolation": "REUSABLE_PATTERN_ONLY",
    }
    auth = value["authentication_reuse_result"]
    assert auth["result"] == "ACCOUNT_GENESIS_AUTH_PRIMITIVE_PATTERN_REUSE_ONLY"
    assert auth["production_authentication_primitive_found"] is True
    assert auth["direct_reuse_available"] is False
    assert auth["new_cryptographic_domain_required"] is True
    fresh = value["freshness_reuse_result"]
    assert fresh["result"] == "ACCOUNT_GENESIS_FRESHNESS_PRIMITIVE_PATTERN_REUSE_ONLY"
    assert fresh["production_freshness_primitive_found"] is True
    assert fresh["independent_account_lineage_supported"] is False

    boundary = value["authority_boundary"]
    assert boundary["CatalogRuntimeAcceptanceAuthority"] == "MUST_REMAIN_SCOPED"
    assert boundary["CatalogAdmissionReceiptAuthority"] == "MUST_REMAIN_SCOPED"
    assert boundary["SourceProducerMembershipAuthority"] == "MUST_REMAIN_SCOPED"
    assert boundary["CryptoHunterAccountAuthority"] == "NOT_AVAILABLE"
    assert boundary["primitive_availability_does_not_create_account_authority"] is True
    assert value["root_of_trust_impact"]["solved"] is False
    assert value["root_of_trust_impact"]["root_of_trust"] == "DESIGN_BLOCKED"
    assert value["reservation_owner_impact"]["solved"] is False
    assert value["reservation_owner_impact"]["m012_key_owner_is_reservation_owner"] is False
    assert set(value["mandatory_redteam"].values()) == {"FAIL"}
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert value["preserved_status"]["M0.12"] == "ACCEPTED / AVAILABLE"
    assert value["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"
    assert value["preserved_status"]["C25"] == "BLOCKED"
    assert value["preserved_status"]["S9D"] == "OPEN"


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        (("domain_separation", "catalog_hmac_as_account_genesis_hmac"), "ACCEPT"),
        (("production_test_isolation", "TEST_authentication_material_accepted_in_PRODUCTION"), True),
        (("threat_boundary", "authentication_is_freshness"), True),
        (("threat_boundary", "machine_wide_rollback"), "PROTECTED"),
        (("cross_authority_anchor_analysis", "single_shared_anchor_sufficient"), True),
        (("authority_boundary", "CatalogRuntimeAcceptanceAuthority"), "ACCOUNT_AUTHORITY"),
        (("reservation_owner_impact", "m012_key_owner_is_reservation_owner"), True),
        (("freshness_reuse_result", "independent_account_lineage_supported"), True),
    ],
)
def test_mandatory_redteam_mutations_fail(path: tuple[str, str], unsafe_value: object) -> None:
    mutated = deepcopy(load())
    section, field = path
    mutated[section][field] = unsafe_value
    with pytest.raises(AssertionError):
        validate(mutated)


def test_current_production_evidence_literals_remain_present() -> None:
    receipt_source = (
        ROOT / "bot_core/instruments/catalog_admission_receipt.py"
    ).read_text(encoding="utf-8")
    test_source = (
        ROOT / "bot_core/instruments/testing_catalog_admission_receipt.py"
    ).read_text(encoding="utf-8")
    runtime_source = (
        ROOT / "bot_core/instruments/catalog_runtime_acceptance.py"
    ).read_text(encoding="utf-8")
    membership_source = (
        ROOT / "bot_core/instruments/source_producer_membership.py"
    ).read_text(encoding="utf-8")
    assert "CRYPTOHUNTER_M0_12_CATALOG_ADMISSION_RECEIPT_V1" in receipt_source
    assert 'service_name="dudzian.catalog-admission-receipt"' in receipt_source
    assert '_ANCHOR_SLOT = "authority-anchor-v1"' in receipt_source
    assert "cryptohunter.catalog-admission-receipt.test.v1" in test_source
    assert "exact production Catalog admission receipt authority required" in runtime_source
    assert "cryptohunter.source_producer_membership.production.v1" in membership_source
