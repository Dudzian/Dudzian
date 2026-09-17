"""Contract/design-freeze and red-team checks for account authority."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path

import pytest

from bot_core.persistence import state_store
from bot_core.persistence.record_registry import PERSISTENCE_RECORD_REGISTRY
from bot_core.persistence.records import _derive_record_key
from bot_core.persistence.state_store import (
    StateStoreError,
    StateStoreMetadata,
    _validate_record_store_scope,
)
from tests.persistence.test_state_store_records import _account, _metadata

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_cryptohunter_account_authority_contract_design.json"
MARKDOWN = DOCS / "m05_cryptohunter_account_authority_contract_design.md"


def load() -> dict:
    return json.loads(MACHINE.read_text(encoding="utf-8"))


def render(value: dict) -> str:
    return (
        "# M0.5 CryptoHunterAccount authority contract design\n\n"
        "This file is a deterministic complete projection of "
        "`m05_cryptohunter_account_authority_contract_design.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(d: dict) -> None:
    entity = d["preserved_entity_contract"]
    assert (entity["entity"], entity["id_field"], entity["id_grammar"]) == (
        "CryptoHunterAccount", "account_id", "acct_<canonical lowercase UUIDv7>"
    )
    assert entity["parent"] == "none" and entity["persistent"] is True
    assert entity["secret_policy"] == "must not contain secrets"
    assert entity["syntax_grants_authority"] is False
    assert d["root_of_trust"]["selected_model"] == "DESIGN_BLOCKED"
    assert d["root_of_trust"]["self_bootstrap"] == "FORBIDDEN"
    assert d["root_of_trust"]["caller_supplied_account_id_can_become_genuine"] is False
    assert d["accepted_record"]["status"] == "DESIGN_BLOCKED"
    assert d["accepted_record"]["PersistentEntityIdentityProjection_is_accepted_fact"] is False
    assert any(
        item.startswith("genuine authority-bound account_id")
        for item in d["accepted_record"]["required_semantics_not_schema"]
    )
    minting = d["id_minting"]
    assert minting["account_id_owner"] == "UNRESOLVED_PENDING_ROOT_OF_TRUST_RECONCILIATION"
    assert minting["algorithm"] == "NOT_FROZEN"
    assert minting["account_id_syntax"] == "FROZEN: acct_<canonical lowercase UUIDv7>"
    assert minting["mint_reservation_protocol"] == "NOT_FROZEN"
    assert minting["entropy_clock_owner"] == "NOT_FROZEN"
    assert minting["caller_selected_genuine_account_id"] == "FORBIDDEN"
    assert minting["candidate_or_reserved_id_is_authority"] is False
    assert minting["status"] == "DESIGN_BLOCKED"
    assert minting["caller_may_choose"] == []
    assert set(d["id_minting"]["caller_fields_forbidden"]) == {
        "accepted_account_record_id", "generation", "authority metadata"
    }
    assert "account_id" not in d["admission"]["request_must_not_contain"]
    assert d["admission"]["caller_selected_account_id"] == "FORBIDDEN"
    assert d["admission"]["authority_bound_account_id_input_presence"] == "NOT_FROZEN"
    assert d["account_id_ownership_reopened_by"] == (
        "m05_cryptohunter_account_root_of_trust_reconciliation.json"
    )
    assert d["admission"]["circular_proof_allowed"] is False
    assert d["lifecycle"]["model"] == "IMMUTABLE_GENESIS_ONLY"
    assert d["lifecycle"]["states"] == []
    assert d["history_model"]["model"].startswith("immutable authenticated append-only")
    assert all(value is False for value in d["resolver_model"]["fallbacks"].values())
    carrier = d["m011_carrier_integration"]
    assert carrier["choice"] == "REUSE_AS_PROJECTION_CARRIER"
    assert carrier["adds_new_domain_facts"] is False
    assert carrier["restorable_authority"] is False
    assert carrier["reverse_minting_forbidden"] is True
    assert "contractually designated target" in carrier["mismatch_on_restart"]
    topology = d["m011_projection_topology"]
    assert topology["status"] == topology["selected_topology"] == "DESIGN_BLOCKED"
    identity = d["immutable_StateStore_identity"]
    assert identity["status"] == "FOUND"
    assert identity["fields"] == [
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
    ]
    assert identity["environment_is_field"] is False
    dimensions = d["StateStore_metadata_dimensions"]
    assert dimensions["environment"]["StateStore_metadata_dimension"] is True
    assert dimensions["environment"]["included_in_state_fingerprint"] is True
    assert dimensions["environment"]["included_in_transaction_protocol_bindings"] is True
    assert dimensions["environment"]["proven_part_of_immutable_StateStore_identity"] is False
    assert dimensions["record_key"]["uses_environment"] is False
    partitioning = d["projection_target_partitioning"]
    assert partitioning["status"] == "DESIGN_BLOCKED"
    assert partitioning["proven_cardinality"].startswith(
        "one CryptoHunterAccount may have multiple device-scoped/local StateStores"
    )
    assert partitioning["environment_partition"] == "NOT_PROVEN"
    assert partitioning["exactly_three_identities_for_one_device_across_PAPER_TESTNET_LIVE"] is False
    assert topology["models"]["F_DESIGN_BLOCKED"] == "SELECTED"
    assert topology["authority_resolver_publication_gate"][
        "requires_every_device_projection_current"
    ] is False
    assert topology["projection_replica_health"]["process_start_order_may_decide"] is False
    assert topology["projection_replica_health"][
        "stale_remote_projection_silently_ignored"
    ] is False
    assert topology["projection_repair_permission"]["M011_to_authority_repair"] == "FORBIDDEN"
    environment_mapping = topology["authority_domain_to_StateStore_environment_mapping"]
    assert environment_mapping["status"] == "DESIGN_BLOCKED"
    assert environment_mapping["PRODUCTION_automatically_equals_LIVE"] is False
    assert environment_mapping["TEST_automatically_equals_PAPER_and_TESTNET"] is False
    assert d["authentication"]["public_SHA_role"].endswith("never authenticity")
    assert d["authentication"]["secrets_in_account_record"] is False
    assert d["freshness"]["SQLite_generation_alone_sufficient"] is False
    assert d["rollback"]["valid_prefix_A_B_after_A_B_C"].startswith("MUST_FAIL_CLOSED")
    assert d["restore"]["M011_restore_can_mint_authority"] is False
    assert d["restore"]["authority_unrecoverable"] == "FAIL_CLOSED"
    assert d["concurrency"]["last_writer_wins"] is False
    assert "same account_id" in d["idempotency"]["exact_retry"]
    isolation = d["environment_isolation"]
    assert isolation["key_material"].startswith("distinct")
    assert isolation["test_to_production_laundering"] == "DENY"
    assert d["workspace_parent_binding"]["immutable"] is True
    assert d["workspace_parent_binding"]["account_authority_unblocks_WorkspaceAuthority"] is False
    assert d["self_mint_resistance"]["result"] == "DENY"
    assert d["self_mint_resistance"]["carrier_only_result"] == "NOT_A_GENUINE_ACCOUNT"
    assert d["self_mint_resistance"]["restore_carrier_only_result"] == "FAIL_CLOSED"
    assert d["design_result"] == "DESIGN_BLOCKED"
    assert d["implementation_allowed"] is False
    assert d["preserved_status"]["WorkspaceAuthority_implementation_allowed"] is False
    assert d["preserved_status"]["FullFillAuthority"] == "NOT_AVAILABLE"
    assert d["preserved_status"]["M0.8"] == "NOT_AVAILABLE"


def test_markdown_is_deterministic_complete_projection() -> None:
    data = load()
    assert MARKDOWN.read_text(encoding="utf-8") == render(data)
    validate(data)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda x: x["preserved_entity_contract"].update(syntax_grants_authority=True),
        lambda x: x["m011_carrier_integration"].update(reverse_minting_forbidden=False),
        lambda x: x["restore"].update(M011_restore_can_mint_authority=True),
        lambda x: x["authentication"].update(public_SHA_role="authenticity"),
        lambda x: x["root_of_trust"].update(self_bootstrap="ALLOWED"),
        lambda x: x["environment_isolation"].update(key_material="shared"),
        lambda x: x["rollback"].update(valid_prefix_A_B_after_A_B_C="ACCEPT"),
        lambda x: x["m011_carrier_integration"].update(choice="SECOND_CURRENT_STORE_WITHOUT_RECONCILIATION"),
        lambda x: x["m011_carrier_integration"].update(mismatch_on_restart="M0.11 WINS"),
        lambda x: x["concurrency"].update(last_writer_wins=True),
        lambda x: x["idempotency"].update(exact_retry="mint second account"),
        lambda x: x["workspace_parent_binding"].update(account_authority_unblocks_WorkspaceAuthority=True),
        lambda x: x["projection_target_partitioning"].update(
            proven_cardinality="one global M0.11 store per account"
        ),
        lambda x: x["m011_projection_topology"].update(selected_topology="ALL_KNOWN_DEVICE_STORES"),
        lambda x: x["m011_projection_topology"]["authority_resolver_publication_gate"].update(
            requires_every_device_projection_current=True
        ),
        lambda x: x["m011_projection_topology"]["projection_replica_health"].update(
            stale_remote_projection_silently_ignored=True
        ),
        lambda x: x["m011_projection_topology"][
            "authority_domain_to_StateStore_environment_mapping"
        ].update(PRODUCTION_automatically_equals_LIVE=True),
        lambda x: x["m011_projection_topology"][
            "authority_domain_to_StateStore_environment_mapping"
        ].update(TEST_automatically_equals_PAPER_and_TESTNET=True),
        lambda x: x["immutable_StateStore_identity"]["fields"].append("environment"),
        lambda x: x["projection_target_partitioning"].update(
            exactly_three_identities_for_one_device_across_PAPER_TESTNET_LIVE=True
        ),
        lambda x: x["id_minting"].update(
            account_id_owner="future genuine CryptoHunterAccountAuthority only"
        ),
        lambda x: x["id_minting"].update(status="PARTIALLY_FROZEN_WITH_OWNER"),
        lambda x: x["admission"]["request_must_not_contain"].append("account_id"),
    ],
    ids=[
        "caller-id-syntax-becomes-authority", "carrier-becomes-authority",
        "restored-carrier-mints-authority", "public-sha-becomes-authenticity",
        "self-bootstrap-root", "production-test-keys-shared", "valid-prefix-rollback",
        "second-current-store-no-reconciliation", "carrier-wins-mismatch",
        "last-writer-wins", "duplicate-mints-second-account", "workspace-auto-unblocked",
        "one-global-store-per-account", "all-stores-without-offline-policy",
        "every-device-projection-gates-resolver", "stale-remote-silently-ignored",
        "production-automatically-live", "test-automatically-paper-testnet",
        "environment-promoted-to-identity", "three-environments-three-store-identities",
        "account-authority-only-owner", "partially-frozen-owner", "permanent-account-id-ban",
    ],
)
def test_mandatory_negative_mutations_fail(mutation) -> None:
    altered = deepcopy(load())
    mutation(altered)
    with pytest.raises(AssertionError):
        validate(altered)


def test_required_sections_and_crash_points_are_closed() -> None:
    d = load()
    required = {
        "repository_head_examined", "provenance", "root_of_trust", "accepted_record",
        "id_minting", "admission", "lifecycle", "resolver_model", "history_model",
        "durability_model", "m011_carrier_integration", "m011_projection_topology",
        "immutable_StateStore_identity", "StateStore_metadata_dimensions",
        "projection_target_partitioning",
        "authentication", "freshness",
        "rollback", "restart", "crash_atomicity", "concurrency", "idempotency",
        "environment_isolation", "workspace_parent_binding", "self_mint_resistance",
        "missing_semantics", "design_result", "implementation_allowed", "next_stage",
        "preserved_status", "account_id_ownership_reopened_by", "targeted_reopen_scope",
    }
    assert required <= d.keys()
    assert set(d["crash_atomicity"]["points"]) == {
        "before_prepare", "after_prepare", "after_authority_record", "after_authority_head",
        "before_freshness_anchor", "after_freshness_anchor",
        "before_M011_projection_update", "after_M011_projection_update",
    }
    assert len(d["negative_requirements"]) == 20


def test_projection_scope_matches_production_state_store_contract() -> None:
    """The topology blocker is derived from production scope, not JSON invention."""

    production_fields = {field.name for field in fields(StateStoreMetadata)}
    assert {"account_id", "device_installation_id", "environment"} <= production_fields
    assert tuple(state_store._IDENTITY_FIELDS) == (
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
    )
    assert "environment" not in state_store._IDENTITY_FIELDS
    assert _metadata().environment in {"PAPER", "TESTNET", "LIVE"}

    metadata_payload = _metadata().to_mapping()
    metadata_key = _derive_record_key(
        "StateStoreMetadata",
        PERSISTENCE_RECORD_REGISTRY["StateStoreMetadata"],
        metadata_payload,
    )
    assert metadata_key == (
        f"state-store:{metadata_payload['account_id']}:"
        f"{metadata_payload['device_installation_id']}:"
        f"{metadata_payload['state_store_identity_fingerprint_sha256']}:"
        f"{metadata_payload['protected_freshness_generation']}"
    )
    assert metadata_payload["environment"] not in metadata_key

    matching = _account()
    _validate_record_store_scope(matching, _metadata())
    foreign = _account("acct_01890f4c-7b9a-7cc1-8a2b-123456789abd")
    with pytest.raises(StateStoreError, match="outside StateStore scope"):
        _validate_record_store_scope(foreign, _metadata())
