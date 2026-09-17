"""Executable red-team checks for the M0.7 -> M0.5 binding decision."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable, get_type_hints
import inspect

import pytest

from bot_core.instruments.catalog_admission_receipt import CatalogAdmissionReceiptAuthority
from bot_core.instruments.catalog_runtime_acceptance import CatalogRuntimeAcceptanceAuthority
from bot_core.instruments.source_producer_membership import SourceProducerMembershipAuthority

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m07_m05_fill_instrument_binding_reconciliation.json"
MARKDOWN = DOCS / "m07_m05_fill_instrument_binding_reconciliation.md"
M07 = DOCS / "commands_events_order_lifecycle_and_idempotency.json"
M05 = DOCS / "exchange_accounts_and_instruments.json"
M06 = DOCS / "strategy_market_data_and_execution_routing.json"
VOCABULARY = DOCS / "canonical_domain_vocabulary.json"
ORACLE = ROOT / "bot_core/instruments/catalog_projection_oracle.py"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(machine: dict[str, object]) -> str:
    return (
        "# M0.7/M0.5 Fill–Instrument binding reconciliation\n\n"
        "This file is a deterministic complete projection of "
        "`m07_m05_fill_instrument_binding_reconciliation.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(machine, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(machine: dict[str, object]) -> None:
    m07, m05, m06 = load(M07), load(M05), load(M06)
    fill = m07["fill_contract"]
    instrument = m05["instrument_contract"]
    history = instrument["trusted_history_contract"]
    fields = set(instrument["record_fields"])

    assert machine["current_m07_fill_binding"]["schema_field_count"] == len(fill["fact_fields"]) == 19
    assert machine["current_m07_fill_binding"]["stale_required_matches"] == fill["instrument_binding"]["required_matches"]
    assert machine["current_m07_fill_binding"]["valid_direct_instrument_matches"] == ["workspace_id"]
    assert not {"exchange_id", "environment"} & fields
    assert machine["current_m07_fill_binding"]["forbidden_direct_instrument_matches"] == ["exchange_id", "environment"]
    assert instrument["identity_dimensions"] == ["source_exchange_id", "market_type", "venue_symbol"]
    assert "execution environment is not an Instrument field" in instrument["rules"]
    assert "source_exchange_id is never rewritten for PAPER execution" in instrument["rules"]
    assert machine["current_m05_instrument_identity"]["exchange_id_equals_source_exchange_id_assumption"] == "FORBIDDEN"

    required_resolution = "historical resolution requires exact instrument_id + metadata_version + accepted_source_catalog_snapshot_id; current fallback forbidden"
    assert required_resolution in instrument["rules"]
    assert history["snapshot_binding_field"] == "accepted_source_catalog_snapshot_id"
    assert machine["current_m05_historical_resolution"]["canonical_resolver_input"] == [
        "instrument_id", "metadata_version", "accepted_source_catalog_snapshot_id"
    ]
    assert machine["current_m05_historical_resolution"]["current_fallback"] == "FORBIDDEN"
    assert machine["current_m05_historical_resolution"]["latest_snapshot_fallback"] == "FORBIDDEN"

    candidate = machine["composite_resolver_candidate"]
    assert candidate["production_authority"] == "NOT_FOUND_TEST_ONLY"
    assert candidate["caller_may_supply_snapshot_as_trusted_proof"] is False
    assert candidate["status"] == "INSUFFICIENT"
    proof = machine["composite_uniqueness_proof"]
    assert proof["status"] == "FAILED_FOR_CANONICAL_RESOLUTION"
    assert proof["accepted_source_catalog_snapshot_id_role"] == "BOTH"
    assert machine["fill_schema_migration_required"]["value"] is True
    assert machine["reconciliation_result"] == "M07_FILL_CONTRACT_MIGRATION_REQUIRED"

    projection = m05["workspace_catalog_projection_contract"]
    assert projection["runtime_writer"] == "NOT_IMPLEMENTED"
    assert machine["workspace_catalog_projection_dependency"]["runtime_authority"] == "NOT_AVAILABLE"
    assert machine["workspace_catalog_projection_dependency"]["raw_mapping_is_proof"] is False
    ascat = m05["accepted_source_catalog_snapshot_contract"]
    dependency = machine["accepted_source_catalog_dependency"]
    assert ascat["runtime_writer"] == dependency["contract_declared_runtime_writer"] == "NOT_IMPLEMENTED"
    assert dependency["contract_metadata_interpretation"] == "HISTORICAL_ONLY_NOT_CURRENT_PRODUCTION_AVAILABILITY_PROOF"
    assert dependency["current_production_catalog_runtime_authority"] == "AVAILABLE"
    assert dependency["authority"] == "ACCEPTED_AVAILABLE"
    assert dependency["authority_symbol"] == "bot_core.instruments.catalog_runtime_acceptance.CatalogRuntimeAcceptanceAuthority"
    assert dependency["public_ingestion_boundary"] == "fetch_catalog_once"
    assert dependency["need_to_implement_new_catalog_runtime_authority"] is False
    assert dependency["caller_snapshot_id_is_proof"] is False
    assert dependency["latest_snapshot_fallback"] is False
    invariant = machine["catalog_authority_invariant"]
    assert invariant == {
        "accepted_source_catalog_runtime_owner": "CatalogRuntimeAcceptanceAuthority",
        "competing_catalog_runtime_authority_forbidden": True,
        "catalog_admission_receipt_authority_required": True,
        "source_producer_membership_authority_required": True,
    }
    assert m06["canonical_source_catalog_projection_chain"]["runtime_acceptance"] == "NOT_IMPLEMENTED"

    assets = m05["asset_reference_contract"]["rules"]
    assert "asset_namespace exact-binds source_exchange_id" in assets
    assert "PAPER execution never rewrites asset_namespace to paper_simulated_venue" in assets
    assert "asset_namespace must equal Fill.exchange_id" in fill["fee_semantics"]["CHARGE"]
    assert machine["asset_reference_drift_impact"]["classification"] == "BLOCKING"

    kinds = {row["canonical_name"]: row for row in load(VOCABULARY)["entity_kinds"]}
    assert kinds["Instrument"]["parent"] == "Workspace"
    assert kinds["Fill"]["parent"] == "Order"
    assert machine["entity_kinds_drift_impact"]["classification"] == "NONBLOCKING"

    preserved = machine["preserved_status"]
    assert preserved == {
        "M0.7_OrderAuthority_kernel": "ACCEPTED_AVAILABLE",
        "M0.7_structural_Full_Fill_foundation": "CLOSED",
        "production_M0.7_FullFillAuthority": "NOT_FOUND",
        "M0.7_semantic_SUBMIT_ORDER": "BLOCKED_UPSTREAM",
        "M0.8": "NOT_AVAILABLE",
        "production_M0.5": "NOT_AVAILABLE",
        "M0.12_1.46.0": "ACCEPTED",
        "CatalogRuntimeAcceptanceAuthority": "ACCEPTED_AVAILABLE",
        "CatalogAdmissionReceiptAuthority": "ACCEPTED_AVAILABLE",
        "AcceptedSourceProducerMembership": "ACCEPTED_AVAILABLE",
        "C25": "BLOCKED",
        "S9D": "OPEN",
    }


def test_markdown_is_deterministic_complete_projection() -> None:
    machine = load(MACHINE)
    assert MARKDOWN.read_text(encoding="utf-8") == render(machine)


def test_reconciliation_is_derived_from_current_contracts() -> None:
    validate(load(MACHINE))


def test_current_production_catalog_authority_boundary_and_dependencies_exist() -> None:
    assert inspect.isclass(CatalogRuntimeAcceptanceAuthority)
    assert callable(CatalogRuntimeAcceptanceAuthority.fetch_catalog_once)
    assert list(inspect.signature(CatalogRuntimeAcceptanceAuthority.fetch_catalog_once).parameters) == [
        "self", "release_binding"
    ]
    hints = get_type_hints(CatalogRuntimeAcceptanceAuthority.__init__)
    assert hints["membership_authority"] is SourceProducerMembershipAuthority
    assert hints["catalog_admission_receipt_authority"] is CatalogAdmissionReceiptAuthority


def test_test_only_nominal_history_is_not_production_authority() -> None:
    production = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in (ROOT / "bot_core").rglob("*.py")
    )
    tests = (ROOT / "tests/architecture/test_cryptohunter_commands_events_order_lifecycle_and_idempotency.py").read_text(encoding="utf-8")
    assert "class M05PrevalidatedInstrumentHistory" not in production
    assert "class M05PrevalidatedInstrumentHistory" in tests


def test_oracle_requires_exact_snapshot_and_has_no_latest_or_current_fallback() -> None:
    source = ORACLE.read_text(encoding="utf-8")
    assert 'record["accepted_source_catalog_snapshot_id"]' in source
    assert 'record.get("metadata_version") == binding["instrument_metadata_version"]' in source
    assert 'record.get("accepted_source_catalog_snapshot_id")' in source
    assert "latest" not in source.lower()


Mutation = Callable[[dict[str, object]], None]


@pytest.mark.parametrize("mutation", [
    lambda d: d["current_m07_fill_binding"].update(valid_direct_instrument_matches=["workspace_id", "exchange_id"]),
    lambda d: d["current_m07_fill_binding"].update(valid_direct_instrument_matches=["workspace_id", "environment"]),
    lambda d: d["current_m05_instrument_identity"].update(exchange_id_equals_source_exchange_id_assumption="ALLOWED"),
    lambda d: d["current_m05_historical_resolution"].update(current_fallback="ALLOWED"),
    lambda d: d["current_m05_historical_resolution"].update(latest_snapshot_fallback="ALLOWED"),
    lambda d: d["accepted_source_catalog_dependency"].update(caller_snapshot_id_is_proof=True),
    lambda d: d["accepted_source_catalog_dependency"].update(authority="NOT_AVAILABLE"),
    lambda d: d["accepted_source_catalog_dependency"].update(need_to_implement_new_catalog_runtime_authority=True),
    lambda d: d["catalog_authority_invariant"].update(competing_catalog_runtime_authority_forbidden=False),
    lambda d: d["catalog_authority_invariant"].update(catalog_admission_receipt_authority_required=False),
    lambda d: d["workspace_catalog_projection_dependency"].update(raw_mapping_is_proof=True),
    lambda d: d["composite_resolver_candidate"].update(production_authority="AVAILABLE"),
    lambda d: d["composite_uniqueness_proof"].update(status="PROVEN"),
    lambda d: d["fill_schema_migration_required"].update(value=False),
    lambda d: d.update(reconciliation_result="COMPOSITE_RESOLVER_SUFFICIENT"),
])
def test_forbidden_security_claim_mutations_fail(mutation: Mutation) -> None:
    changed = deepcopy(load(MACHINE))
    mutation(changed)
    with pytest.raises(AssertionError):
        validate(changed)
