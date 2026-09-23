"""Executable contract for the M0.5 AccountGenesis security substrate design."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_account_genesis_security_substrate_contract.json"
MARKDOWN = DOCS / "m05_account_genesis_security_substrate_contract.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        "# M0.5 AccountGenesis security substrate contract\n\n"
        "This file is a deterministic complete projection of "
        "`m05_account_genesis_security_substrate_contract.json`. "
        "JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def validate(value: dict) -> None:
    assert value["artifact"] == "M05_ACCOUNT_GENESIS_SECURITY_SUBSTRATE_CONTRACT"
    assert value["iteration"] == "CONTRACT / DESIGN FREEZE ONLY"
    assert value["reviewed_head_supplied"] == ("bb58eb57071226049a92eac14a90e4dbb8eff415")
    assert value["provenance"]["source"] == "GIT"
    assert value["provenance"]["reviewed_head_available_locally"] is False

    frozen = value["frozen_inputs"]
    assert frozen["m012_authorities"] == "UNCHANGED_AND_SCOPED"
    for field in (
        "catalog_receipt_keys",
        "catalog_lifecycle_root",
        "catalog_anchor_slot",
        "catalog_authority_domains",
        "catalog_authority_as_account_authority",
    ):
        assert frozen[field] == "FORBIDDEN"
    assert frozen["valid_MAC_is_freshness"] is False
    assert frozen["PREPARED_is_genuine_account"] is False

    storage = value["generic_secret_storage_reuse"]
    assert storage["implementation"].endswith("KeyringSecretStorage")
    assert storage["classification"] == "REUSABLE_WITH_WRAPPER"
    assert storage["wrapper_fixed_scopes"] == [
        "service_name",
        "storage_key_prefix",
        "index_path",
    ]
    assert storage["service_name_isolation"] == "REQUIRED"
    assert storage["index_path_isolation"] == "REQUIRED"
    assert storage["service_name_isolation_alone_sufficient"] is False
    assert storage["shared_KeyringSecretStorage_index_between_authority_families"] == ("FORBIDDEN")
    assert storage["generic_default_index_path"] == "var/security/secret_index.json"
    assert storage["account_genesis_index_path"] == (
        "var/security/account_genesis_secret_index.json"
    )
    assert "not trust authority" in storage["index_behavior"]["role"]
    assert storage["index_behavior"]["foreign_service_inventory_mutation"] == ("FORBIDDEN")
    assert storage["cross_authority_rotation_attack"]["classification"] == ("UNSAFE / FORBIDDEN")
    assert value["authentication_algorithm"]["selected"] == (
        "HMAC-SHA-256 with opaque AccountGenesis custody"
    )

    domains = value["cryptographic_domains"]
    assert domains["exact_domain_literals"] == "FROZEN"
    assert domains["production_authority_domain"] != domains["test_authority_domain"]
    purposes = domains["purposes"]
    assert len(purposes) == 6
    assert len(set(purposes.values())) == len(purposes)
    assert purposes["operation_records"] != purposes["reservation_records"]
    assert purposes["genesis_finalization_records"] != purposes["abort_release_disposition"]

    coverage = value["record_authentication_coverage"]
    assert set(coverage["required_where_semantically_applicable"]) == {
        "authority_domain",
        "record_purpose/type",
        "schema/version",
        "identity",
        "generation/sequence",
        "predecessor",
        "canonical_semantic_body",
        "environment/trust_domain",
        "key_id",
    }
    assert coverage["authentication_is_authorization"] is False
    assert coverage["authentication_is_freshness"] is False

    namespace = value["key_namespace"]
    assert (
        namespace["production_custody_service_name"]
        != (namespace["catalog_service_name_forbidden"])
    )
    assert namespace["catalog_handles_or_raw_keys_reused"] is False
    assert namespace["production_index_scope"] == "ACCOUNT_GENESIS_ONLY"
    assert namespace["shared_index_with_catalog"] == "FORBIDDEN"
    assert (
        namespace["foreign_service_inventory_enumeration_mutation_removal_or_rotation"]
        == "FORBIDDEN"
    )
    lifecycle = value["key_lifecycle"]
    assert lifecycle["states"] == ["ACTIVE", "VERIFY_ONLY", "REVOKED"]
    assert lifecycle["lifecycle_state_authenticated"] is True
    assert lifecycle["lifecycle_root"].startswith("YES:")
    assert lifecycle["destructive_retirement"].startswith("FORBIDDEN")
    assert value["historical_verification"]["normal_rotation"].startswith("REQUIRED")
    assert value["historical_verification"]["terminal_facts_may_become_unverifiable"] is False
    assert value["key_rotation"]["same_operation"] is False
    assert value["key_rotation"]["storage_master_key_rotation_transactional"] is False
    assert value["key_rotation"]["partial_storage_rotation"] == "FAIL_CLOSED"

    freshness = value["freshness_lineage"]
    assert freshness["model"] == "AccountGenesis-specific atomic multi-lineage anchor document"
    assert freshness["physical_tables"] == "NOT_FROZEN"
    assert len(freshness["semantic_heads"]) == 5
    assert "COMMITTED -> PREPARED MUST_FAIL_CLOSED" in freshness["closure"]
    assert "ABORTED -> PREPARED MUST_FAIL_CLOSED" in freshness["closure"]
    anchor = value["anchor_models"]
    assert anchor["selected"] == "C. atomic multi-lineage anchor document"
    assert anchor["catalog_anchor_reused"] is False
    assert anchor["anchor_authentication"].startswith("HMAC-SHA-256")
    assert value["anchor_atomicity"]["local_and_external_atomic"] is False
    assert value["anchor_atomicity"]["priority"] == "safety > availability"
    mismatch = value["anchor_mismatch_semantics"]
    assert mismatch["local_greater_than_anchor"].endswith("FAIL_CLOSED")
    assert mismatch["local_less_than_anchor"].endswith("FAIL_CLOSED")
    assert mismatch["automatic_repair"].startswith("FORBIDDEN")

    threat = value["rollback_threat_boundary"]
    assert threat["local_DB_rollback_while_newer_keyring_survives"] == "DETECTED"
    assert threat["coordinated_DB_and_keyring_rollback"] == "OUT_OF_SCOPE / NOT_DETECTED"
    assert threat["machine_wide_rollback"] == "NOT_PROTECTED"
    assert threat["valid_MAC_with_stale_anchor"] == "MUST_FAIL_CLOSED"

    assert value["migration"]["model"] == "AUTHENTICATED_MIGRATION_REQUIRED"
    assert value["migration"]["protocol"] == "NOT_FROZEN"
    assert value["backup_restore"]["DB_same_machine"].endswith("fails closed")
    assert value["backup_restore"]["DB_new_machine"].startswith("Fails closed")
    assert value["CAS_compatibility"]["ambiguous_generation_transition"] == "FORBIDDEN"
    assert value["restart_order"] == [
        "load substrate",
        "verify authenticated key lifecycle",
        "verify authenticated journal closure",
        "verify freshness anchor",
        "reconstruct reservation/operation state",
        "reconcile genesis commit",
        "publish resolver",
    ]

    separation = value["production_test_separation"]
    assert separation["TEST_key_verifies_PRODUCTION_record"] is False
    assert separation["PRODUCTION_key_verifies_TEST_record"] is False
    assert separation["rotation_inventory/index_state"] == "DISTINCT"
    assert separation["shared_mutable_rotation_index"] == "FORBIDDEN"
    assert separation["TEST_mutates_PRODUCTION_rotation_inventory"] is False
    gate = value["exact_type_gating"]
    assert gate["decision"] == "FROZEN"
    assert gate["subclasses_accepted"] is False
    assert gate["test_double_laundering"] == "MUST_FAIL"

    boundary = value["authority_boundary"]
    assert boundary["domain_authority_owner"] == "DESIGN_BLOCKED"
    assert boundary["owners_are_semantically_identical"] is False
    assert boundary["M0.12_security_substrate_owner_becomes_AccountAuthority"] is False
    assert boundary["valid_MAC_is_authorized_account_creation"] is False
    assert value["result"]["primary_result"] == (
        "ACCOUNT_GENESIS_SECURITY_SUBSTRATE_CONTRACT_FROZEN"
    )
    assert value["result"]["root_of_trust_solved"] is False
    assert value["result"]["reservation_owner_solved"] is False
    assert all(allowed is False for allowed in value["implementation_allowed"].values())
    assert value["preserved_status"]["production M0.5"] == "NOT_AVAILABLE"
    assert set(value["mandatory_redteam"].values()) == {"FAIL"}


def test_markdown_is_deterministic_complete_projection() -> None:
    value = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(value)
    validate(value)


@pytest.mark.parametrize(
    ("path", "unsafe_value"),
    [
        (("key_namespace", "catalog_handles_or_raw_keys_reused"), True),
        (("anchor_models", "catalog_anchor_reused"), True),
        (
            ("cryptographic_domains", "purposes", "reservation_records"),
            "CRYPTOHUNTER_M0_5_ACCOUNT_GENESIS_OPERATION_RECORD_V1",
        ),
        (("production_test_separation", "TEST_key_verifies_PRODUCTION_record"), True),
        (("rollback_threat_boundary", "valid_MAC_with_stale_anchor"), "ACCEPT"),
        (("freshness_lineage", "closure", 1), "COMMITTED -> PREPARED ACCEPT"),
        (("freshness_lineage", "closure", 2), "ABORTED -> PREPARED ACCEPT"),
        (("anchor_mismatch_semantics", "automatic_repair"), "AUTOMATIC"),
        (("key_rotation", "same_operation"), True),
        (("authority_boundary", "M0.12_security_substrate_owner_becomes_AccountAuthority"), True),
        (("exact_type_gating", "subclasses_accepted"), True),
        (("key_namespace", "shared_index_with_catalog"), "ALLOWED"),
        (
            ("generic_secret_storage_reuse", "service_name_isolation_alone_sufficient"),
            True,
        ),
        (
            ("production_test_separation", "shared_mutable_rotation_index"),
            "ALLOWED",
        ),
        (
            (
                "key_namespace",
                "foreign_service_inventory_enumeration_mutation_removal_or_rotation",
            ),
            "ALLOWED",
        ),
    ],
)
def test_mandatory_redteam_mutations_fail(
    path: tuple[str | int, ...], unsafe_value: object
) -> None:
    mutated = deepcopy(load())
    target = mutated
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = unsafe_value
    with pytest.raises(AssertionError):
        validate(mutated)


def test_cross_artifact_status_parity() -> None:
    names = [
        "m05_account_genesis_trust_primitives_reuse_discovery.json",
        "m05_cryptohunter_account_genesis_reservation_state_model.json",
        "m05_cryptohunter_account_genesis_uniqueness_idempotency_design.json",
        "m05_cryptohunter_account_genesis_authority_model.json",
    ]
    artifacts = [json.loads((DOCS / name).read_text(encoding="utf-8")) for name in names]
    contract = load()
    assert all(
        artifact["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
        for artifact in artifacts
    )
    assert contract["implementation_allowed"]["CryptoHunterAccountAuthority"] is False
    assert contract["preserved_status"]["CryptoHunterAccountAuthority"] == "NOT_AVAILABLE"
    assert contract["preserved_status"]["root_of_trust"] == "DESIGN_BLOCKED"


def test_generic_keyring_index_and_rotation_source_evidence() -> None:
    source = (ROOT / "bot_core/security/keyring_storage.py").read_text(encoding="utf-8")
    assert 'DEFAULT_INDEX_PATH = Path("var/security/secret_index.json")' in source
    rotate_start = source.index("    def rotate_master_key(self) -> None:")
    next_section = source.index("    # Obsługa klucza głównego i indeksu", rotate_start)
    rotate_source = source[rotate_start:next_section]
    assert "index = self._load_index()" in rotate_source
    assert 'stored_keys = list(index.get("keys", {}).keys())' in rotate_source
    assert "for storage_key in stored_keys:" in rotate_source
    assert "self._keyring.get_password(self._service_name, storage_key)" in rotate_source
    assert "self._unregister_key(storage_key)" in rotate_source
    assert rotate_source.index("self._store_master_key(new_master, hwid_digest)") < (
        rotate_source.index("for storage_key in stored_keys:")
    )
