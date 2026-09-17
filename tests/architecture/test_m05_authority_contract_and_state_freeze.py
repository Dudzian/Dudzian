"""Executable red-team checks for the M0.5 authority contract/design freeze."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Callable

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
NAMES = (
    "m05_workspace_authority_contract_freeze",
    "m05_instrument_authority_contract_freeze",
    "m05_wcp_authority_state_model_design",
)


def load(name: str) -> dict[str, object]:
    return json.loads((DOCS / f"{name}.json").read_text(encoding="utf-8"))


def render(name: str, title: str, value: dict[str, object]) -> str:
    body = json.dumps(value, indent=2, ensure_ascii=False)
    return (
        f"# {title}\n\nThis file is a deterministic complete projection of `{name}.json`. "
        f"JSON is the source of truth.\n\n```json\n{body}\n```\n"
    )


def validate(workspace: dict[str, object], instrument: dict[str, object], wcp: dict[str, object]) -> None:
    assert workspace["result"] == "INSUFFICIENT_SEMANTICS"
    assert workspace["contract_completeness"] == "INCOMPLETE"
    assert workspace["upstream_dependency_status"] == "BLOCKED_UPSTREAM"
    assert workspace["upstream_dependency"] == "CryptoHunterAccount authority/resolver"
    assert workspace["upstream_authority_availability"] == "NOT_FOUND / NOT_AVAILABLE"
    assert workspace["upstream_only_resolution_sufficient_for_implementation"] is False
    assert workspace["implementation_allowed"] is False
    assert workspace["frozen_denials"] == {
        "syntactic_workspace_id_is_authority": False,
        "caller_parent_id_is_proof": False,
        "PersistentEntityIdentityProjection_is_authority": False,
    }
    blocker_classes = workspace["blocker_classes"]
    assert blocker_classes["intrinsic_contract_semantics"] == [
        "accepted Workspace record schema", "admission command/event", "trusted admission actor",
        "lifecycle", "current/historical resolver semantics", "durable history model",
        "restart semantics", "rollback semantics", "authenticated persistence",
    ]
    assert blocker_classes["upstream_authorities"] == ["CryptoHunterAccount authority/resolver"]
    assert workspace["implementation_invariant"]["independent_of_upstream_status"] is True
    assert workspace["next_stage"]["dependency_rule"].startswith("Completing A does not")
    matrix = workspace["contract_matrix"]
    assert matrix["workspace_id_minting_owner"].endswith("caller forbidden")
    assert matrix["current_resolver"] == matrix["historical_resolver"] == "NOT_FOUND"
    assert matrix["authenticated_persistence"] == "NOT_FROZEN"
    assert intrinsic_gating_holds(workspace)

    assert instrument["result"] == "INSUFFICIENT_SEMANTICS"
    assert instrument["implementation_allowed"] is False
    assert instrument["canonical_identity"]["key"] == [
        "workspace_id", "source_exchange_id", "market_type", "venue_symbol"
    ]
    assert instrument["canonical_identity"]["execution_exchange_derivation"] is False
    assert instrument["canonical_identity"]["instrument_id_minted_by"] == "InstrumentAuthority; caller and WCP forbidden"
    assert instrument["version_contract"]["metadata_version"] == "positive non-boolean integer"
    assert instrument["version_contract"]["rollback"].endswith("DENY")
    resolver = instrument["exact_resolver"]
    assert resolver["fallbacks"] == {
        "latest": False, "nearest": False, "current_for_missing_old": False, "symbol_lookup": False
    }
    assert instrument["ordering"].startswith("genuine ASCAT -> Instrument version admission")

    assert wcp["result"] == "DESIGN_BLOCKED"
    assert wcp["implementation_allowed"] is False
    assert wcp["selection_policy"]["owner"] == "NOT_FOUND"
    assert wcp["id_contract"]["minted_by"] == "WCP authority only; caller forbidden"
    assert wcp["id_contract"]["cross_workspace_reuse"] == "DENY"
    assert wcp["selection_policy"]["omission_is_deletion"] is False
    assert wcp["options"]["D"].startswith("SELECTED: DESIGN_BLOCKED")
    assert wcp["history_semantics"]["current_only_authorized"] is False
    assert wcp["authentication"]["content_fingerprint"].endswith("never authenticity")
    assert wcp["authentication"]["public_SHA_authenticity"] is False
    assert wcp["rollback_freshness"]["available_frozen_mechanism"] == "NOT_FOUND"
    assert wcp["rollback_freshness"]["protection_result"] == "NOT_PROVEN"
    assert wcp["environment_isolation"]["test_to_production_laundering"] == "DENY"
    assert wcp["concurrency"]["conflict"].startswith("DENY/CONFLICT")
    assert wcp["concurrency"]["same_members_different_order"].endswith("order is fingerprint-significant")
    assert all(value is False for value in (
        workspace["provenance"]["formal_authority_status_minted"],
        instrument["provenance"]["formal_authority_status_minted"],
        wcp["provenance"]["formal_authority_status_minted"],
    ))
    assert len({tuple(value["authority_boundary"]["owns"]) for value in (workspace, instrument, wcp)}) == 3
    expected = {
        "M0.12_1.46.0": "ACCEPTED",
        "CatalogRuntimeAcceptanceAuthority": "ACCEPTED_AVAILABLE",
        "M0.7_structural_Full_Fill_v2": "ACCEPTED_AVAILABLE_FOR_STRUCTURAL_VALIDATION",
        "M0.7_FullFillAuthority": "NOT_AVAILABLE",
        "WorkspaceCatalogProjectionAuthority": "NOT_AVAILABLE",
        "production_M0.5": "NOT_AVAILABLE", "M0.8": "NOT_AVAILABLE", "C25": "BLOCKED", "S9D": "OPEN",
    }
    assert workspace["preserved_status"] == instrument["preserved_status"] == wcp["preserved_status"] == expected


def test_markdown_files_are_deterministic_complete_projections() -> None:
    titles = (
        "M0.5 Workspace authority contract freeze",
        "M0.5 Instrument authority contract freeze",
        "M0.5 WCP authority state model design",
    )
    for name, title in zip(NAMES, titles, strict=True):
        value = load(name)
        assert (DOCS / f"{name}.md").read_text(encoding="utf-8") == render(name, title, value)


def test_contract_and_design_freeze() -> None:
    validate(*(load(name) for name in NAMES))


INTRINSIC_FIELDS = (
    "accepted_record_schema", "admission_command_event", "trusted_admission_actor", "lifecycle",
    "current_resolver", "historical_resolver", "durable_history_model", "restart_semantics",
    "rollback_semantics", "authenticated_persistence",
)


def intrinsic_gating_holds(workspace: dict[str, object]) -> bool:
    matrix = workspace["contract_matrix"]
    unresolved = any(matrix[field] in {"NOT_FROZEN", "NOT_FOUND"} for field in INTRINSIC_FIELDS)
    return not unresolved or workspace["implementation_allowed"] is False


@pytest.mark.parametrize("unresolved_field", [
    "accepted_record_schema", "lifecycle", "restart_semantics", "rollback_semantics",
    "authenticated_persistence",
])
def test_upstream_account_availability_does_not_bypass_intrinsic_gating(
    unresolved_field: str,
) -> None:
    workspace = deepcopy(load(NAMES[0]))
    workspace["upstream_dependency_status"] = "AVAILABLE"
    workspace["upstream_authority_availability"] = "AVAILABLE"
    assert workspace["contract_matrix"][unresolved_field] == "NOT_FROZEN"
    assert intrinsic_gating_holds(workspace)
    assert workspace["implementation_allowed"] is False


Mutation = Callable[[dict[str, object], dict[str, object], dict[str, object]], None]


@pytest.mark.parametrize("mutation", [
    lambda ws, _i, _w: ws.update(result="BLOCKED_UPSTREAM"),
    lambda ws, _i, _w: ws.update(upstream_only_resolution_sufficient_for_implementation=True),
    lambda ws, _i, _w: ws.update(implementation_allowed=True, upstream_dependency_status="AVAILABLE", upstream_authority_availability="AVAILABLE"),
    lambda ws, _i, _w: ws.update(contract_completeness="COMPLETE"),
    lambda ws, _i, _w: ws["frozen_denials"].update(syntactic_workspace_id_is_authority=True),
    lambda ws, _i, _w: ws["frozen_denials"].update(PersistentEntityIdentityProjection_is_authority=True),
    lambda _ws, i, _w: i["version_contract"].update(metadata_version="caller selected"),
    lambda _ws, i, _w: i["exact_resolver"]["fallbacks"].update(current_for_missing_old=True),
    lambda _ws, i, _w: i["exact_resolver"]["fallbacks"].update(latest=True),
    lambda _ws, i, _w: i["canonical_identity"].update(instrument_id_minted_by="WCP"),
    lambda _ws, _i, w: w["authentication"].update(public_SHA_authenticity=True),
    lambda _ws, _i, w: w["history_semantics"].update(current_only_authorized=True),
    lambda _ws, _i, w: w["rollback_freshness"].update(protection_result="ACCEPT_STALE_PREFIX"),
    lambda _ws, _i, w: w["authentication"].update(content_fingerprint="SQL rewrite is trusted"),
    lambda _ws, _i, w: w["environment_isolation"].update(test_to_production_laundering="ALLOW"),
    lambda _ws, _i, w: w["id_contract"].update(minted_by="caller"),
    lambda _ws, _i, w: w["id_contract"].update(cross_workspace_reuse="ALLOW"),
    lambda _ws, _i, w: w["concurrency"].update(conflict="last-writer-wins"),
])
def test_required_negative_mutations_fail(mutation: Mutation) -> None:
    values = [deepcopy(load(name)) for name in NAMES]
    mutation(*values)
    with pytest.raises(AssertionError):
        validate(*values)
