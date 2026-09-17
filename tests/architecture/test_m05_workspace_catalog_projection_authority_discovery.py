"""Executable contract and red-team checks for WCP authority discovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import runpy
from typing import Callable

import pytest

from bot_core.instruments import catalog_projection_oracle as oracle

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m05_workspace_catalog_projection_authority_discovery.json"
MARKDOWN = DOCS / "m05_workspace_catalog_projection_authority_discovery.md"
M05 = DOCS / "exchange_accounts_and_instruments.json"
M06 = DOCS / "strategy_market_data_and_execution_routing.json"


def load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def render(machine: dict[str, object]) -> str:
    return (
        "# M0.5 WorkspaceCatalogProjection authority discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m05_workspace_catalog_projection_authority_discovery.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(machine, indent=2, ensure_ascii=False)}\n```\n"
    )


def validate(machine: dict[str, object]) -> None:
    m05, m06 = load(M05), load(M06)
    contract = m05["workspace_catalog_projection_contract"]
    schema = machine["canonical_schema"]
    assert schema["status"] == "FOUND"
    assert schema["closed_fields"] == contract["fields"]
    assert set(schema["closed_fields"]) == oracle.PROJECTION_FIELDS
    assert schema["member_closed_fields"] == contract["member_schema"]["fields"]
    assert set(schema["member_closed_fields"]) == oracle.PROJECTION_MEMBER_FIELDS
    identity = machine["identity_and_fingerprint"]
    fingerprint = contract["content_fingerprint_definition"]
    assert identity["domain_separator"] == fingerprint["domain_separator"]
    assert identity["input_fields"] == fingerprint["input_fields"] == list(oracle.PROJECTION_FINGERPRINT_FIELDS)
    assert identity["canonical_json"]["unicode_normalization"].startswith("NONE")
    assert identity["ordering"]["instrument_ids"].startswith("order retained")
    assert machine["production_authority"]["status"] == "NOT_FOUND"
    assert machine["production_authority"]["raw_mapping_is_authority"] is False
    assert machine["production_authority"]["test_fixture_or_nominal_wrapper_is_authority"] is False
    assert machine["workspace_dependency"]["WorkspaceAuthority"] == "NOT_FOUND"
    assert machine["workspace_dependency"]["syntactic_workspace_id_is_proof"] is False
    assert machine["ascat_dependency"]["status"] == "AVAILABLE"
    assert machine["ascat_dependency"]["caller_snapshot_id_is_proof"] is False
    assert machine["ascat_dependency"]["second_catalog_authority_allowed"] is False
    assert machine["instrument_dependency"]["status"] == "NOT_FOUND"
    instrument = machine["instrument_dependency"]
    assert instrument["production_authority"] == "NOT_FOUND"
    assert instrument["production_resolver_authority"] == "NOT_FOUND"
    assert instrument["instrument_resolution_mode"] == (
        "EXACT_CURRENT_OR_HISTORICAL_BY_METADATA_VERSION"
    )
    resolution = machine["member_closure"]["instrument_resolution"]
    assert resolution["exact_version_required"] is True
    assert resolution["current_record_permitted_on_exact_version_match"] is True
    assert resolution["historical_record_permitted_on_exact_version_match"] is True
    assert resolution["current_fallback_for_missing_requested_version"] is False
    assert resolution["latest_version_fallback"] is False
    assert machine["member_closure"]["latest_or_current_fallback"] is False
    assert machine["member_closure"]["symbol_only_lookup"] is False
    assert machine["member_closure"]["caller_tuple_is_proof"] is False
    assert machine["workspace_isolation"]["required_result"] == "DENY"
    assert machine["snapshot_closure"]["snapshot_A_member_B"] is False
    assert machine["history_model"]["contract_status"] == "NO_WCP_HISTORY_MODEL_FROZEN"
    assert machine["durability"]["production_WCP_store"] == "NOT_FOUND"
    assert machine["rollback_protection"]["sqlite_integrity_sufficient"] is False
    assert machine["tamper_protection"]["public_sha_authenticity"] is False
    assert machine["self_mint_result"]["result"] == "NO"
    assert machine["self_mint_result"]["current_result_basis"] == "NO_PRODUCTION_AUTHORITY_EXISTS"
    assert machine["self_mint_result"]["future_kernel_self_mint_resistance"] == "UNPROVEN_UNTIL_DESIGN"
    assert machine["structural_projection_validation"] == "AVAILABLE"
    structural = schema["structural_projection_validation"]
    assert structural == {
        "status": "AVAILABLE",
        "closed_schema_validation": "AVAILABLE",
        "fingerprint_calculation": "AVAILABLE",
        "member_structural_closure_oracle": "AVAILABLE",
        "is_authority_kernel": False,
    }
    assert machine["kernel_buildability"] == "BLOCKED_BY_MULTIPLE"
    assert machine["authority_kernel_implementation_allowed"] is False
    assert machine["structural_validator_implementation_allowed"] is True
    assert machine["durable_WCP_authority_kernel_implementation_allowed"] is False
    prerequisites = machine["authority_kernel_prerequisites"]
    assert prerequisites == {
        "workspace_authority_contract": "NOT_AVAILABLE",
        "instrument_authority_contract": "NOT_AVAILABLE_EXACT_CURRENT_AND_HISTORICAL_RESOLVER",
        "persistent_state_model": "NOT_FROZEN",
        "history_or_current_state_semantics": "NOT_FROZEN",
        "restart_validation_contract": "NOT_FROZEN",
        "rollback_policy": "NOT_FROZEN",
        "authenticated_state_policy": "NOT_FROZEN",
        "external_freshness_policy": "NOT_FROZEN",
        "selection_policy_owner": "NOT_FOUND",
    }
    blocking_values = {"NOT_FROZEN", "NOT_FOUND", "NOT_AVAILABLE", "NOT_DEFINED"}
    if any(value in blocking_values for value in prerequisites.values()):
        assert machine["authority_kernel_implementation_allowed"] is False
        assert machine["durable_WCP_authority_kernel_implementation_allowed"] is False
    assert machine["history_model"]["current_only_store_authorized"] is False
    assert machine["rollback_protection"]["current_contract_has_no_defined_answer"] is True
    assert machine["rollback_protection"]["authority_implementation_permission"] is False
    assert machine["rollback_protection"]["mutable_current_only_store_without_freshness_semantics_allowed"] is False
    assert machine["semantic_admission_status"] == "NOT_AVAILABLE"
    assert m06["source_catalog_projection_schemas"]["WorkspaceCatalogProjection"]["canonical_contract"].endswith("#/workspace_catalog_projection_contract")
    assert contract["runtime_writer"] == "NOT_IMPLEMENTED"


def test_markdown_is_deterministic_complete_projection() -> None:
    machine = load(MACHINE)
    assert MARKDOWN.read_text(encoding="utf-8") == render(machine)


def test_discovery_is_derived_from_current_contracts() -> None:
    validate(load(MACHINE))


def test_only_structural_oracle_exists_not_production_authority() -> None:
    production = "\n".join(path.read_text(encoding="utf-8", errors="ignore") for path in (ROOT / "bot_core").rglob("*.py"))
    assert "class WorkspaceCatalogProjectionAuthority" not in production
    assert "class WorkspaceAuthority" not in production
    assert "class InstrumentAuthority" not in production
    assert callable(oracle.validate_workspace_catalog_projection)
    catalog_authority = (ROOT / "bot_core/instruments/catalog_runtime_acceptance.py").read_text(encoding="utf-8")
    assert "class CatalogRuntimeAcceptanceAuthority:" in catalog_authority
    assert "def fetch_catalog_once(" in catalog_authority


def test_exact_current_or_historical_instrument_resolution_without_fallback() -> None:
    m06_tests = runpy.run_path(str(ROOT / "tests/architecture/test_m06_workspace_catalog_source_chain.py"))
    graph = m06_tests["canonical_graph"]
    validate_global = m06_tests["_global_graph"]
    history = m06_tests["_history"]
    fingerprint = m06_tests["_projection_fingerprint"]

    # Exact requested version may be the authoritative current record.
    source, projection, current, _, _ = graph()
    assert oracle.validate_workspace_catalog_projection(
        projection, source, {current["instrument_id"]: current}
    )
    assert validate_global(source, projection, current, {})

    # An older exact requested version may be resolved from authoritative history.
    source, projection, current, _, _ = graph()
    projection["member_bindings"][0]["instrument_metadata_version"] = 6
    fingerprint(projection)
    exact_history = {"instr_1": history(current, (6,))}
    assert validate_global(source, projection, current, exact_history)

    # Current v7 cannot replace requested v6, nor can historical v5 act as latest fallback.
    assert not validate_global(source, projection, current, {})
    assert not validate_global(source, projection, current, {"instr_1": history(current, (5,))})


@pytest.mark.parametrize(
    "mutation",
    [
        {"workspace_id": "ws_wrong"},
        {"accepted_source_catalog_snapshot_id": "ascat_wrong"},
        {"source_exchange_id": "kraken"},
        {"market_type": "MARGIN", "instrument_type": "MARGIN_PAIR"},
        {"venue_symbol": "ETHUSDT"},
    ],
)
def test_exact_current_version_still_denies_wrong_scope_source_or_snapshot(
    mutation: dict[str, object],
) -> None:
    graph = runpy.run_path(
        str(ROOT / "tests/architecture/test_m06_workspace_catalog_source_chain.py")
    )["canonical_graph"]
    source, projection, current, _, _ = graph()
    current.update(mutation)
    assert current["metadata_version"] == projection["member_bindings"][0][
        "instrument_metadata_version"
    ]
    assert not oracle.validate_workspace_catalog_projection(
        projection, source, {current["instrument_id"]: current}
    )


Mutation = Callable[[dict[str, object]], None]


@pytest.mark.parametrize("mutation", [
    lambda d: d["ascat_dependency"].update(caller_snapshot_id_is_proof=True),
    lambda d: d["production_authority"].update(raw_mapping_is_authority=True),
    lambda d: d["workspace_isolation"].update(required_result="ALLOW"),
    lambda d: d["snapshot_closure"].update(snapshot_A_member_B=True),
    lambda d: d["member_closure"].update(caller_tuple_is_proof=True),
    lambda d: d["member_closure"].update(latest_or_current_fallback=True),
    lambda d: d["member_closure"].update(symbol_only_lookup=True),
    lambda d: d["production_authority"].update(test_fixture_or_nominal_wrapper_is_authority=True),
    lambda d: d["tamper_protection"].update(public_sha_authenticity=True),
    lambda d: d["workspace_dependency"].update(syntactic_workspace_id_is_proof=True),
    lambda d: d["rollback_protection"].update(sqlite_integrity_sufficient=True),
    lambda d: d.update(kernel_buildability="BUILDABLE_INDEPENDENTLY_WITH_SEMANTIC_ADMISSION_DISABLED"),
    lambda d: d.update(authority_kernel_implementation_allowed=True),
    lambda d: d.update(durable_WCP_authority_kernel_implementation_allowed=True),
    lambda d: d["rollback_protection"].update(authority_implementation_permission=True),
    lambda d: d["rollback_protection"].update(mutable_current_only_store_without_freshness_semantics_allowed=True),
    lambda d: d["history_model"].update(current_only_store_authorized=True),
    lambda d: d["self_mint_result"].update(future_kernel_self_mint_resistance="PROVEN"),
    lambda d: d["instrument_dependency"].update(instrument_resolution_mode="HISTORICAL_ONLY"),
    lambda d: d["member_closure"]["instrument_resolution"].update(current_record_permitted_on_exact_version_match=False),
    lambda d: d["member_closure"]["instrument_resolution"].update(current_fallback_for_missing_requested_version=True),
    lambda d: d["member_closure"]["instrument_resolution"].update(latest_version_fallback=True),
    lambda d: d["member_closure"]["instrument_resolution"].update(exact_version_required=False),
])
def test_mandatory_red_team_mutations_fail(mutation: Mutation) -> None:
    candidate = deepcopy(load(MACHINE))
    mutation(candidate)
    with pytest.raises(AssertionError):
        validate(candidate)


def test_frozen_statuses_and_no_implementation_scope() -> None:
    machine = load(MACHINE)
    assert machine["preserved_status"] == {
        "M0.7_OrderAuthority_kernel": "ACCEPTED_AVAILABLE",
        "M0.7_structural_Full_Fill_v2": "ACCEPTED_AVAILABLE_FOR_STRUCTURAL_VALIDATION",
        "M0.7_FullFillAuthority": "NOT_AVAILABLE",
        "CatalogRuntimeAcceptanceAuthority": "ACCEPTED_AVAILABLE",
        "CatalogAdmissionReceiptAuthority": "ACCEPTED_AVAILABLE",
        "AcceptedSourceProducerMembership": "ACCEPTED_AVAILABLE",
        "production_M0.5": "NOT_AVAILABLE",
        "M0.8": "NOT_AVAILABLE",
        "C25": "BLOCKED",
        "S9D": "OPEN",
        "WorkspaceCatalogProjectionAuthority": "NOT_AVAILABLE",
        "WorkspaceAuthority": "NOT_FOUND",
        "Instrument_authority": "NOT_FOUND",
    }
    assert machine["impact_on_m07_full_fill"]["implemented_now"] is False
