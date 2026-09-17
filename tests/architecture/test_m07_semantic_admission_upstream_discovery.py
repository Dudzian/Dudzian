"""Executable contract for the M0.7 upstream-authority discovery result."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
from typing import Callable, Mapping

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m07_semantic_admission_upstream_discovery.json"
MARKDOWN = DOCS / "m07_semantic_admission_upstream_discovery.md"
ORDER_CONTRACT = DOCS / "commands_events_order_lifecycle_and_idempotency.json"


def _load() -> tuple[dict[str, object], str]:
    return json.loads(MACHINE.read_text(encoding="utf-8")), MARKDOWN.read_text(encoding="utf-8")


def render(machine: Mapping[str, object]) -> str:
    body = json.dumps(machine, indent=2, ensure_ascii=False)
    return (
        "# M0.7 semantic admission upstream discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m07_semantic_admission_upstream_discovery.json`. JSON is the source of truth.\n\n"
        f"```json\n{body}\n```\n"
    )


def _resolve(document: object, pointer: str) -> object:
    value = document
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        assert isinstance(value, (dict, list))
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def _rows(machine: Mapping[str, object]) -> dict[str, dict[str, object]]:
    table = machine["non_authoritative_examined_head_inventory"]["decision_table"]
    assert isinstance(table, list)
    return {row["dependency"]: row for row in table}


def _derive_next(machine: Mapping[str, object]) -> str:
    rows = machine["non_authoritative_examined_head_inventory"]["decision_table"]
    assert isinstance(rows, list)
    eligible = []
    for row in rows:
        if row["production_authority_found"]:
            continue
        buildability = row["buildability"]
        if all(
            buildability[key]
            for key in (
                "contract_complete",
                "no_missing_genuine_parent",
                "no_withheld_required_parent",
                "no_invented_provenance_required",
            )
        ):
            eligible.append(row["dependency"])
    return eligible[0] if eligible else "NONE_ALL_REMAINING_DEPENDENCIES_BLOCKED"


def _formal_discovery_available(machine: Mapping[str, object]) -> bool:
    provenance = machine["repository_provenance"]
    return (
        provenance["result"] == "VERIFIED_ACCEPTED_BASE"
        and provenance["accepted_base_object_available"] is True
        and provenance["accepted_base_ancestry_verified"] is True
        and provenance["exact_discovery_run_against_accepted_base"] is True
    )


def _validate(machine: Mapping[str, object]) -> None:
    contract = json.loads(ORDER_CONTRACT.read_text(encoding="utf-8"))
    frozen = contract["environment_execution_boundary"]["side_effect_gate_order"]
    assert machine["canonical_side_effect_gate_order"] == frozen
    assert machine["secondary_exact_gate_order"] == "NONE"
    assert "gate_order" not in machine

    inventory = machine["non_authoritative_examined_head_inventory"]
    rows = _rows(machine)
    coverage = {
        frozen[0]: {"M0.4 ProductCapabilities"},
        frozen[1]: {"ExchangeAccount", "Instrument"},
        frozen[2]: {"StrategyInstance", "ExecutionRoute", "RouteReadiness"},
        frozen[3]: {"M0.9 ExecutionLease"},
        frozen[4]: {"M0.7 atomic admission"},
        frozen[5]: {"ExecutionAdapter provenance"},
    }
    for parent, dependencies in coverage.items():
        assert dependencies <= set(rows)
        assert all(rows[name]["frozen_parent_gate"] == parent for name in dependencies)

    assert set(rows) == {
        "M0.4 ProductCapabilities", "Workspace prerequisite", "ExchangeAccount",
        "Instrument", "StrategyInstance", "ExecutionRoute", "RouteReadiness",
        "ExecutionAdapter provenance", "authority_context_id resolver",
        "M0.9 ExecutionLease", "M0.7 atomic admission",
    }
    for dependency in ("Workspace prerequisite", "authority_context_id resolver"):
        item = next(
            entry for entry in inventory["dependency_resolution_inventory"]
            if entry["dependency"] == dependency
        )
        assert item["ordering"] == "UNSPECIFIED"
        assert item["ordered_by_contract"] is False
        assert "ordering_json_pointer" not in item

    lease = rows["M0.9 ExecutionLease"]
    assert lease["plan_admission_requirement"] is False
    assert lease["external_side_effect_requirement"] is True
    boundary = inventory["execution_lease_boundary"]
    assert boundary["future_execution_lease_not_a_M0.7_blocker"] is True
    assert boundary["plan_admission_dependency"] is False
    assert boundary["external_side_effect_dependency"] is True

    derived = _derive_next(machine)
    assert inventory["next_authority_derivation"]["derived_result"] == derived
    assert inventory["next_authority_derivation"]["eligible_missing_authorities"] == [derived]
    assert inventory["candidate_next_buildable_authority"] == derived
    assert inventory["candidate_classification"] == "CANDIDATE_FROM_UNVERIFIED_EXAMINED_HEAD"
    assert inventory["authoritative_for_accepted_base"] is False
    assert inventory["may_advance_implementation"] is False
    assert inventory["implementation_permission"] is False
    assert inventory["accepted_base_equivalence"] == "NOT_PROVEN"

    available = _formal_discovery_available(machine)
    assert available is False
    assert machine["formal_accepted_base_discovery"] == "WITHHELD_PENDING_EXACT_ACCEPTED_BASE"
    assert machine["implementation_advancement_permitted"] is False
    assert machine["next_buildable_authority"] == "WITHHELD_PENDING_EXACT_ACCEPTED_BASE_DISCOVERY"
    assert machine["next_buildable_authority"] != derived
    assert machine["next_blocker"] == "DISCOVERY_BLOCKED_ACCEPTED_BASE_COMMIT_NOT_AVAILABLE"
    provenance = machine["repository_provenance"]
    assert provenance["publication_disposition"] == "DISCOVERY_WITHHELD_PENDING_EXACT_ACCEPTED_BASE"
    assert provenance["accepted_base_object_available"] is False
    assert provenance["accepted_base_ancestry_verified"] is False
    assert provenance["exact_discovery_run_against_accepted_base"] is False
    assert machine["order_kernel_status"] == {
        "production_order_authority_kernel": "ACCEPTED_AVAILABLE",
        "production_order_history": "ACCEPTED_AVAILABLE",
        "historical_order_resolver": "ACCEPTED_AVAILABLE",
        "semantic_submit_order": "BLOCKED_UPSTREAM",
        "M0.7": "NOT_AVAILABLE",
        "M0.8_order_side_dependency": "BLOCKED",
    }


def test_complete_markdown_is_exact_deterministic_projection() -> None:
    machine, markdown = _load()
    assert markdown == render(machine)


def test_machine_semantics_are_complete_and_consistent() -> None:
    machine, _ = _load()
    _validate(machine)


def test_repository_provenance_command_is_executable_and_matches_artifact() -> None:
    machine, _ = _load()
    provenance = machine["repository_provenance"]
    completed = subprocess.run(
        provenance["command"].split(), cwd=ROOT, check=False, capture_output=True, text=True
    )
    assert completed.returncode == provenance["exit_code"] == 128
    assert provenance["result"] == "BLOCKED_ACCEPTED_BASE_COMMIT_NOT_AVAILABLE"
    assert machine["formal_accepted_base_discovery"] == "WITHHELD_PENDING_EXACT_ACCEPTED_BASE"
    assert machine["next_buildable_authority"] == "WITHHELD_PENDING_EXACT_ACCEPTED_BASE_DISCOVERY"
    assert machine["next_blocker"] == "DISCOVERY_BLOCKED_ACCEPTED_BASE_COMMIT_NOT_AVAILABLE"
    assert machine["implementation_advancement_permitted"] is False


def test_every_reported_contract_pointer_resolves() -> None:
    machine, _ = _load()
    documents = {
        path.name: json.loads(path.read_text(encoding="utf-8"))
        for path in DOCS.glob("*.json")
    }

    def visit(value: object) -> None:
        if isinstance(value, dict):
            if set(value) == {"contract", "json_pointers"}:
                for pointer in value["json_pointers"]:
                    _resolve(documents[value["contract"]], pointer)
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(machine)


Mutation = Callable[[dict[str, object]], None]


def _remove_row(name: str) -> Mutation:
    def mutation(machine: dict[str, object]) -> None:
        inventory = machine["non_authoritative_examined_head_inventory"]
        inventory["decision_table"] = [
            row for row in inventory["decision_table"] if row["dependency"] != name
        ]
    return mutation


def _invent_order(name: str) -> Mutation:
    def mutation(machine: dict[str, object]) -> None:
        inventory = machine["non_authoritative_examined_head_inventory"]
        item = next(
            entry for entry in inventory["dependency_resolution_inventory"]
            if entry["dependency"] == name
        )
        item["ordered_by_contract"] = True
    return mutation


@pytest.mark.parametrize(
    "mutation",
    [
        _remove_row("M0.4 ProductCapabilities"),
        _remove_row("StrategyInstance"),
        _remove_row("M0.7 atomic admission"),
        _remove_row("RouteReadiness"),
        lambda d: _rows(d)["M0.9 ExecutionLease"].update(plan_admission_requirement=True),
        _invent_order("Workspace prerequisite"),
        _invent_order("authority_context_id resolver"),
        lambda d: d["canonical_side_effect_gate_order"].reverse(),
        lambda d: d.update(next_blocker="WRONG"),
        lambda d: d.update(next_buildable_authority="M0.4 ProductCapabilities"),
        lambda d: d["repository_provenance"].update(publication_disposition="DISCOVERY_AVAILABLE"),
        lambda d: d["non_authoritative_examined_head_inventory"].update(implementation_permission=True),
        lambda d: d.update(implementation_advancement_permitted=True),
        lambda d: d["non_authoritative_examined_head_inventory"]["decision_table"].pop(),
        lambda d: d["order_kernel_status"].update(semantic_submit_order="AVAILABLE"),
    ],
)
def test_required_negative_mutations_fail_semantic_validation(mutation: Mutation) -> None:
    machine, _ = _load()
    changed = deepcopy(machine)
    mutation(changed)
    with pytest.raises((AssertionError, KeyError)):
        _validate(changed)
