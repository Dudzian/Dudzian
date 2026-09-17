"""Executable discovery and red-team checks for CryptoHunterAccount authority."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path

import pytest

from bot_core.persistence.record_registry import PERSISTENCE_RECORD_REGISTRY
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreSnapshot

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_authority_discovery.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_authority_discovery.md"
VOCABULARY = DOCS / "canonical_domain_vocabulary.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(machine: dict[str, object]) -> str:
    return (
        "# M0.5 CryptoHunterAccount authority discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_authority_discovery.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(machine, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(machine: dict[str, object]) -> None:
    entity = machine["canonical_entity"]
    identity = machine["canonical_id_contract"]
    assert entity["status"] == "FOUND"
    assert (identity["field"], identity["prefix"], identity["grammar"]) == (
        "account_id",
        "acct_",
        "acct_<uuidv7>",
    )
    assert identity["syntax_grants_authority"] is False
    authority = machine["production_authority"]
    assert authority["status"] == "NOT_FOUND"
    assert authority["production_admission_authority"] == "NOT_FOUND"
    assert authority["production_genuine_account_resolver"] == "NOT_FOUND"
    assert authority["production_genuine_record_writer"]["status"] == "NOT_FOUND"
    schema = machine["accepted_record_schema"]
    assert schema["status"] == "NOT_FROZEN"
    assert schema["projection_found"]["is_authority"] is False
    assert machine["trusted_actor"]["status"] == "NOT_FOUND"
    assert machine["lifecycle"]["status"] == "NOT_FROZEN"
    assert machine["current_resolver"]["status"] == "NOT_FOUND"
    assert machine["historical_resolver"]["status"] == "NOT_FOUND"
    parent = machine["workspace_parent_binding"]
    assert parent["caller_parent_id_is_proof"] is False
    assert parent["PersistentEntityIdentityProjection_is_proof"] is False
    assert parent["raw_account_mapping_is_proof"] is False
    assert parent["binding_immutability"] == "UNRESOLVED"
    durability = machine["durability"]
    carrier = durability["current_record_carrier"]
    registry = PERSISTENCE_RECORD_REGISTRY["CryptoHunterAccount current record"]
    assert carrier["status"] == "FOUND"
    assert carrier["durability_class"] == registry["durability_class"]
    assert carrier["representation_category"] == registry["representation_category"]
    assert carrier["representation_schema"] == registry["projection_schema_if_any"]
    assert carrier["adds_new_domain_facts"] is registry["adds_new_domain_facts"] is False
    assert carrier["restorable_authority"] is registry["restorable_authority"] is False
    assert carrier["grants_authority"] is False
    assert durability["generic_M011_persistence_tables"]["status"] == "FOUND"
    assert durability["CryptoHunterAccount_specific_authority_tables"]["status"] == "NOT_FOUND"
    assert durability["account_authority_history"]["status"] == "NOT_FOUND"
    laundering = machine["authority_laundering_denials"]
    assert all(value is False for value in laundering.values())
    restore = machine["restore_semantics"]
    assert restore["StateStoreSnapshot_carries_authority"] is False
    assert restore["restore_can_recreate_genuine_account_authority_by_itself"] is False
    assert (
        machine["duplicate_store_policy"][
            "silently_create_second_current_account_persistence_source"
        ]
        is False
    )
    assert machine["authentication"]["classification"] == "NONE"
    assert machine["authentication"]["public_sha_only_is_authenticity"] is False
    assert machine["rollback"]["rollback_protection"] == "NOT_FROZEN"
    assert machine["restart"]["status"] == "NOT_FROZEN"
    assert machine["self_mint"]["result"] == "NO"
    assert machine["self_mint"]["caller_selected_account_id_grants_accepted_identity"] is False
    assert machine["self_mint"]["current_result_basis"] == "NO_CURRENT_AUTHORITY_PATH"
    assert machine["self_mint"]["future_self_mint_resistance"] == "UNPROVEN"
    completeness = machine["contract_completeness"]
    assert completeness["primary_result"] == "INSUFFICIENT_SEMANTICS"
    assert completeness["status"] == "INCOMPLETE"
    assert machine["implementation_allowed"] is False
    assert machine["preserved_status"]["production_M0.5"] == "NOT_AVAILABLE"
    assert (
        machine["impact_on_workspace"]["account_available_implies_workspace_implementation_allowed"]
        is False
    )


def test_markdown_is_deterministic_complete_projection() -> None:
    machine = load(MACHINE)
    assert MARKDOWN.read_text(encoding="utf-8") == render(machine)


def test_discovery_matches_canonical_entity_declaration() -> None:
    machine, vocabulary = load(MACHINE), load(VOCABULARY)
    entity = next(
        item
        for item in vocabulary["entity_kinds"]
        if item["canonical_name"] == "CryptoHunterAccount"
    )
    discovered = machine["canonical_entity"]
    identity = machine["canonical_id_contract"]
    assert entity["id_field"] == identity["field"]
    assert identity["prefix"] == entity["id_prefix"] + "_"
    assert entity["parent"] == discovered["parent"] == "none"
    assert entity["persistence"] is discovered["persistence"] is True
    validate(machine)


def test_semantic_production_search_distinguishes_nearby_non_authorities() -> None:
    machine = load(MACHINE)
    search = machine["production_authority"]["semantic_search"]
    assert len(search["concepts"]) >= 7
    assert {"bot_core", "tests/architecture", "docs/audits"} <= set(search["roots"])
    source = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "bot_core").rglob("*.py")
    )
    state_store_source = inspect.getsource(SQLiteStateStore)
    snapshot_source = inspect.getsource(StateStoreSnapshot)
    registry_source = (ROOT / "bot_core/persistence/record_registry.py").read_text(encoding="utf-8")
    assert "class CryptoHunterAccountAuthority" not in source
    assert "CryptoHunterAccount current record" in PERSISTENCE_RECORD_REGISTRY
    assert "PersistentEntityIdentityProjection" in registry_source
    assert "state_store_current_records" in state_store_source
    assert "carries no authority" in snapshot_source
    assert machine["durability"]["current_record_carrier"]["status"] == "FOUND"
    assert machine["production_authority"]["production_admission_authority"] == "NOT_FOUND"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["canonical_id_contract"].update(syntax_grants_authority=True),
        lambda x: x["accepted_record_schema"]["projection_found"].update(is_authority=True),
        lambda x: x["workspace_parent_binding"].update(raw_account_mapping_is_proof=True),
        lambda x: x["self_mint"].update(caller_selected_account_id_grants_accepted_identity=True),
        lambda x: x["workspace_parent_binding"].update(caller_parent_id_is_proof=True),
        lambda x: x["authentication"].update(public_sha_only_is_authenticity=True),
        lambda x: x["trusted_actor"].update(status="IGNORED"),
        lambda x: x["accepted_record_schema"].update(status="FOUND"),
        lambda x: (
            x["rollback"].update(rollback_protection="NOT_FROZEN"),
            x.update(implementation_allowed=True),
        ),
        lambda x: (x["restart"].update(status="NOT_FROZEN"), x.update(implementation_allowed=True)),
        lambda x: x["durability"]["current_record_carrier"].update(status="NOT_FOUND"),
        lambda x: x["duplicate_store_policy"].update(
            silently_create_second_current_account_persistence_source=True
        ),
    ],
    ids=[
        "valid-id-syntax-is-not-authority",
        "identity-projection-is-not-authority",
        "raw-mapping-is-not-authority",
        "caller-selected-id-is-not-accepted",
        "workspace-parent-string-is-not-proof",
        "public-sha-is-not-authenticity",
        "trusted-actor-gap-cannot-be-ignored",
        "missing-schema-cannot-be-complete",
        "missing-rollback-forbids-implementation",
        "missing-restart-forbids-implementation",
        "registry-carrier-cannot-be-denied",
        "silent-second-current-store-forbidden",
    ],
)
def test_red_team_mutations_fail(mutation) -> None:
    altered = deepcopy(load(MACHINE))
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)
