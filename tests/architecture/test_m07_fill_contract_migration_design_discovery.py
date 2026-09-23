"""Executable contract checks for the M0.7 Full Fill v2 design discovery."""

from __future__ import annotations

from copy import deepcopy
import inspect
import json
from pathlib import Path
from typing import Callable

import pytest

from bot_core.execution.m07_fill_validation import validate_structural_fill


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m07_fill_contract_migration_design_discovery.json"
MARKDOWN = DOCS / "m07_fill_contract_migration_design_discovery.md"
M07 = DOCS / "commands_events_order_lifecycle_and_idempotency.json"
M05 = DOCS / "exchange_accounts_and_instruments.json"
RECONCILIATION = DOCS / "m07_m05_fill_instrument_binding_reconciliation.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(machine: dict[str, object]) -> str:
    return (
        "# M0.7 Full Fill contract migration design discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m07_fill_contract_migration_design_discovery.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(machine, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(machine: dict[str, object]) -> None:
    m07 = load(M07)["fill_contract"]
    m05 = load(M05)
    reconciliation = load(RECONCILIATION)
    current = machine["current_contract"]
    schema = machine["new_fill_schema"]
    versioning = machine["fingerprint_versioning"]

    assert machine["accepted_reconciliation"] == "8eec34f92a32b87dd196122e0a1d4a18f46248f9"
    assert reconciliation["reconciliation_result"] == machine["reconciliation_result"]
    assert current["fields"] == m07["fact_fields"]
    assert current["field_count"] == len(m07["fact_fields"]) == 19
    assert current["fingerprint_input_count"] == len(m07["fingerprint"]["input_fields"]) == 18

    assert set(machine["candidate_designs"]) == {
        "ADD_ACCEPTED_SOURCE_CATALOG_SNAPSHOT_ID_TO_FILL",
        "AUTHORITY_BOUND_PROVENANCE_REFERENCE",
        "REUSE_EXISTING_CANONICAL_REFERENCE",
    }
    assert machine["selected_design"] == "ADD_ACCEPTED_SOURCE_CATALOG_SNAPSHOT_ID_TO_FILL"
    assert schema["field_count"] == len(schema["fields"]) == 20
    assert schema["fields"][:9] == m07["fact_fields"][:9]
    assert schema["fields"][9] == "accepted_source_catalog_snapshot_id"
    assert schema["fields"][10:] == m07["fact_fields"][9:]
    source_contract = m05["accepted_source_catalog_snapshot_contract"]
    expected_prefix = source_contract["id_prefix"]
    assert source_contract["id_field"] == "accepted_source_catalog_snapshot_id"
    assert expected_prefix == "ascat"
    snapshot_schema = schema["field_schemas"]["accepted_source_catalog_snapshot_id"]
    assert snapshot_schema["type"] == "prefixed_non_empty_string"
    assert snapshot_schema["prefix"] == expected_prefix
    assert snapshot_schema["semantic_type"] == "AcceptedSourceCatalogSnapshotId"
    assert snapshot_schema["nullable"] is False
    assert snapshot_schema["immutable"] is True
    assert snapshot_schema["syntax_grants_trust"] is False
    assert snapshot_schema["authority_resolution"].startswith("CatalogRuntimeAcceptanceAuthority")
    new_field = schema["new_field"]
    assert new_field["name"] == "accepted_source_catalog_snapshot_id"
    assert new_field["position"] == 10
    assert new_field["type"] == "prefixed_non_empty_string"
    assert new_field["semantic_type"] == "AcceptedSourceCatalogSnapshotId"
    assert new_field["prefix"] == expected_prefix
    assert new_field["immutable"] is True
    assert new_field["nullable"] is False
    assert new_field["fingerprint_input"] is True
    assert new_field["syntax_grants_trust"] is False
    grammar = new_field["structural_grammar"]
    assert grammar["reject"] == ["", "foo", "snapshot_1", "wcat_1", "ascat", "ASCat_1"]
    assert grammar["may_accept_syntactically"] == ["ascat_1", "ascat_fake"]

    def structural_identifier(value: object) -> bool:
        return (
            type(value) is str
            and value.startswith(expected_prefix + "_")
            and bool(value.removeprefix(expected_prefix + "_"))
        )

    assert not any(structural_identifier(value) for value in grammar["reject"])
    assert all(structural_identifier(value) for value in grammar["may_accept_syntactically"])
    assert schema["fill_schema_version_field"] is None

    assert versioning["old"]["domain"] == "LEGACY_UNDOMAINED_CANONICAL_JSON_SHA256_V1"
    assert versioning["new"]["domain"] == "cryptohunter.m0.7.full_fill.v2"
    assert versioning["new"]["input_fields"] == schema["fields"][:-1]
    assert versioning["new"]["excluded_fields"] == ["fill_fingerprint_sha256"]
    assert versioning["old"]["domain"] != versioning["new"]["domain"]
    assert versioning["cross_version_verification"].startswith("FORBIDDEN")

    legacy = machine["legacy_policy"]
    assert legacy["classification"] == "LEGACY_STRUCTURAL_ONLY"
    assert legacy["historically_readable"] is True
    assert legacy["new_admission"] == "REJECT_FOR_NEW_ADMISSION"
    assert legacy["v2_without_provenance"] == "DENY"

    snapshot = machine["snapshot_binding_semantics"]
    history = m05["instrument_contract"]["trusted_history_contract"]
    assert snapshot["binding_location"] == "FILL"
    assert snapshot["selector"] == [
        "instrument_id",
        "instrument_metadata_version",
        "accepted_source_catalog_snapshot_id",
    ]
    assert snapshot["caller_claim_is_proof"] is False
    assert snapshot["identifier_syntax_grants_trust"] is False
    assert snapshot["authority_validation"].startswith("CatalogRuntimeAcceptanceAuthority resolves")
    assert snapshot["fallbacks"] == {
        "latest_snapshot": False,
        "current_instrument": False,
        "symbol": False,
    }
    assert history["snapshot_binding_field"] == "accepted_source_catalog_snapshot_id"

    catalog = machine["catalog_runtime_authority_dependency"]
    assert catalog["status"] == "AVAILABLE"
    assert catalog["authority"] == "CatalogRuntimeAcceptanceAuthority"
    assert "sole" in catalog["single_authority_invariant"]
    assert catalog["identifier_syntax_is_proof"] is False
    projection = machine["workspace_projection_dependency"]
    assert projection["status"] == "NOT_AVAILABLE"
    assert projection["caller_supplied_projection_map_is_proof"] is False
    assert projection["required_future_interface"]["prove"] == [
        "exact workspace_catalog_projection",
        "projection.workspace_id",
        "projection.accepted_source_catalog_snapshot_id",
        "member instrument_id + instrument_metadata_version",
        "member source_exchange_id + market_type + venue_symbol + source_metadata_version_id",
    ]

    fee = machine["fee_asset_reference_migration"]
    assert fee["old_rule"].endswith("Fill.exchange_id")
    structural_fee = fee["structural_rule"]
    semantic_fee = fee["semantic_rule"]
    assert structural_fee["classification"] == "RAW_STRUCTURAL_FILL_V2_SHAPE_ONLY"
    assert structural_fee["namespace_comparison"] == "NONE"
    assert structural_fee["forbidden_comparisons"] == [
        "fee_asset_reference.asset_namespace == Fill.exchange_id",
        "fee_asset_reference.asset_namespace == historical Instrument.source_exchange_id",
    ]
    assert semantic_fee["classification"] == "TRUSTED_ECONOMIC_FILL_V2_SEMANTIC_RULE"
    assert semantic_fee["owner"] == "future M0.7 FullFillAuthority"
    assert semantic_fee["implementation_allowed_now"] is False
    assert semantic_fee["rule"] == (
        "For CHARGE, fee_asset_reference.asset_namespace == resolved historical "
        "Instrument.source_exchange_id and the exact fee asset resolves in that source "
        "namespace under M0.5."
    )
    assert fee["third_asset_fee"].startswith("LEGAL")
    paper = machine["paper_semantics"]
    assert paper["source_exchange_id"] == paper["fee_asset_namespace"] == "binance"
    assert paper["exchange_id"] == "paper_simulated_venue"
    assert paper["source_execution_equality_required"] is False
    assert paper["source_namespace_rewrite"] is False
    assert machine["live_semantics"]["note"].startswith("observed equality is incidental")

    raw = machine["trust_boundary"]["raw_structural_fill_v2"]
    assert raw["owner"] == "pure structural validator"
    assert raw["input_mode"] == "RAW_FILL_ONLY"
    assert raw["production_symbol"] == (
        "bot_core.execution.m07_fill_validation.validate_structural_fill"
    )
    assert list(inspect.signature(validate_structural_fill).parameters) == ["fill"]
    assert raw["authority_context_consumed"] is False
    assert "AcceptedSourceCatalogSnapshotId structural prefix grammar" in raw["may_prove"]
    assert "snapshot existence" in raw["must_not_prove"]
    assert "snapshot acceptance" in raw["must_not_prove"]
    assert "snapshot authority" in raw["must_not_prove"]
    assert not set(raw["may_prove"]) & set(raw["must_not_prove"])
    assert "fee source namespace correctness" in raw["must_not_prove"]
    assert "fee source namespace correctness" not in raw["may_prove"]
    trusted = machine["trust_boundary"]["trusted_economic_fill_v2"]
    assert trusted["future_owner"] == "M0.7 FullFillAuthority"
    assert (
        "fee_asset_reference.asset_namespace == resolved historical Instrument.source_exchange_id"
    ) in trusted["must_prove"]
    assert machine["trust_boundary"]["complete_accepted_fill_history"]["consumes"] == [
        "accepted trusted economic Fill v2 facts only"
    ]

    dedupe = machine["external_trade_dedupe_migration"]
    assert dedupe["canonical_key"] == m07["external_dedupe_scope"]
    assert dedupe["snapshot_in_key"] is False
    assert dedupe["same_trade_duplicate_effect_possible"] is False
    assert "Before any economic mutation" in dedupe["cross_version_rule"]
    assert machine["fill_id_migration"]["policy"] == "STABLE"
    assert machine["fill_id_migration"]["snapshot_changes_fill_id"] is False

    order = machine["order_authority_impact"]
    assert machine["order_snapshot_relationship"]["mode"] == "FILL_PINNED"
    assert order["snapshot_pin_required"] is False
    assert order["journal_migration_required"] is False
    assert order["disposition"] == "NO_ORDER_JOURNAL_MIGRATION"
    assert machine["m08_impact"]["raw_fill_resolution_by_m08"] is False
    assert machine["m08_impact"]["future_dependency"] == (
        "M0.8 consumes genuine M0.7 FullFillAuthority fact"
    )

    attacks = {row["attack"]: row["result"] for row in machine["attack_matrix"]}
    assert attacks["genuine snapshot lacks exact Instrument metadata version"] == "DENY"
    assert attacks["projection workspace A, Fill/Order workspace B"] == "DENY"
    assert attacks["correct instrument_id, wrong metadata_version"] == "DENY"
    assert attacks["genuine snapshot but member source tuple mismatch"] == "DENY"
    assert attacks["later producer membership revocation rewrites accepted historical Fill"] == (
        "DENY_RETROACTIVE_REWRITE"
    )

    status = machine["preserved_status"]
    scope = machine["candidate_implementation_scope"]
    assert machine["candidate_implementation_allowed"] is True
    assert scope["permission_scope"] == "STRUCTURAL_V2_CONTRACT_AND_VALIDATOR_ONLY"
    assert scope["semantic_admission_implementation_allowed"] is False
    assert scope["semantic_fee_namespace_validation_implementation_allowed"] is False
    assert scope["historical_instrument_resolution_in_structural_validator"] is False
    assert "FullFillAuthority admission" in scope["forbidden"]
    assert "historical Instrument resolution" in scope["forbidden"]
    assert status["M0.7_FullFillAuthority"] == "NOT_AVAILABLE"
    assert status["WorkspaceCatalogProjection_authority"] == "NOT_AVAILABLE"
    assert status["production_M0.5"] == status["M0.8"] == "NOT_AVAILABLE"


def test_design_matches_current_contracts_and_markdown_is_exact() -> None:
    machine = load(MACHINE)
    validate(machine)
    assert MARKDOWN.read_text(encoding="utf-8") == render(machine)


Mutation = Callable[[dict[str, object]], None]


def _set(path: tuple[str, ...], value: object) -> Mutation:
    def mutate(document: dict[str, object]) -> None:
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        (
            "snapshot ID loses prefix grammar",
            _set(
                ("new_fill_schema", "field_schemas", "accepted_source_catalog_snapshot_id", "type"),
                "non_empty_string",
            ),
        ),
        (
            "snapshot ID uses WCP prefix",
            _set(
                (
                    "new_fill_schema",
                    "field_schemas",
                    "accepted_source_catalog_snapshot_id",
                    "prefix",
                ),
                "wcat",
            ),
        ),
        (
            "snapshot syntax grants trust",
            _set(
                (
                    "new_fill_schema",
                    "field_schemas",
                    "accepted_source_catalog_snapshot_id",
                    "syntax_grants_trust",
                ),
                True,
            ),
        ),
        (
            "raw layer claims snapshot acceptance",
            _set(
                ("trust_boundary", "raw_structural_fill_v2", "may_prove"),
                [
                    "AcceptedSourceCatalogSnapshotId structural prefix grammar",
                    "snapshot acceptance",
                ],
            ),
        ),
        (
            "execution/source equality",
            _set(("paper_semantics", "source_execution_equality_required"), True),
        ),
        (
            "structural fee uses execution namespace",
            _set(
                ("fee_asset_reference_migration", "structural_rule", "namespace_comparison"),
                "fee_asset_reference.asset_namespace == Fill.exchange_id",
            ),
        ),
        (
            "structural fee uses historical source namespace",
            _set(
                ("fee_asset_reference_migration", "structural_rule", "namespace_comparison"),
                "fee_asset_reference.asset_namespace == historical Instrument.source_exchange_id",
            ),
        ),
        (
            "raw layer claims source namespace",
            _set(
                ("trust_boundary", "raw_structural_fill_v2", "may_prove"),
                ["fee source namespace correctness"],
            ),
        ),
        (
            "semantic layer omits source namespace",
            _set(
                ("trust_boundary", "trusted_economic_fill_v2", "must_prove"),
                ["external trade dedupe", "Fill/Order binding"],
            ),
        ),
        (
            "candidate includes semantic admission",
            _set(
                ("candidate_implementation_scope", "semantic_admission_implementation_allowed"),
                True,
            ),
        ),
        (
            "candidate includes historical Instrument resolution",
            _set(
                (
                    "candidate_implementation_scope",
                    "historical_instrument_resolution_in_structural_validator",
                ),
                True,
            ),
        ),
        (
            "caller snapshot is proof",
            _set(("snapshot_binding_semantics", "caller_claim_is_proof"), True),
        ),
        (
            "latest snapshot fallback",
            _set(
                ("snapshot_binding_semantics", "fallbacks"),
                {"latest_snapshot": True, "current_instrument": False, "symbol": False},
            ),
        ),
        (
            "current Instrument fallback",
            _set(
                ("snapshot_binding_semantics", "fallbacks"),
                {"latest_snapshot": False, "current_instrument": True, "symbol": False},
            ),
        ),
        (
            "competing Catalog authority",
            _set(
                ("catalog_runtime_authority_dependency", "single_authority_invariant"),
                "Another authority is permitted",
            ),
        ),
        ("M0.8 owns admission", _set(("m08_impact", "raw_fill_resolution_by_m08"), True)),
        (
            "schema changes under old domain",
            _set(
                ("fingerprint_versioning", "new", "domain"),
                "LEGACY_UNDOMAINED_CANONICAL_JSON_SHA256_V1",
            ),
        ),
        ("legacy admitted as v2", _set(("legacy_policy", "v2_without_provenance"), "ACCEPT")),
        (
            "cross-version duplicate effect",
            _set(("external_trade_dedupe_migration", "same_trade_duplicate_effect_possible"), True),
        ),
        ("PAPER source rewrite", _set(("paper_semantics", "source_namespace_rewrite"), True)),
    ],
)
def test_required_negative_mutations_fail(name: str, mutate: Mutation) -> None:
    machine = deepcopy(load(MACHINE))
    mutate(machine)
    with pytest.raises(AssertionError, match=".*"):
        validate(machine)
