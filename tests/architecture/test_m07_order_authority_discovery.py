"""Deterministic assertions for the M0.7 production-authority discovery."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Mapping

import pytest


ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE_PATH = DOCS / "m07_order_authority_discovery.json"
MARKDOWN_PATH = DOCS / "m07_order_authority_discovery.md"


def _artifacts() -> tuple[dict[str, object], str]:
    machine = json.loads(MACHINE_PATH.read_text(encoding="utf-8"))
    markdown = MARKDOWN_PATH.read_text(encoding="utf-8")
    return machine, markdown


def render_m07_order_authority_discovery(machine: Mapping[str, object]) -> str:
    """Render the complete deterministic Markdown projection of the machine result."""
    schema = machine["canonical_order_schema_detail"]
    lifecycle = machine["canonical_order_lifecycle_detail"]
    fill_binding = machine["fill_order_binding_contract"]
    external = machine["frozen_external_status"]
    assert isinstance(schema, dict)
    assert isinstance(lifecycle, dict)
    assert isinstance(fill_binding, dict)
    assert isinstance(external, dict)

    result_rows = (
        ("Canonical Order schema", machine["canonical_order_schema"]),
        ("Canonical lifecycle", machine["canonical_order_lifecycle"]),
        ("Production OrderAuthority", machine["production_order_authority"]),
        ("Production Order store", machine["production_order_store"]),
        ("Exact historical Order resolver", machine["historical_order_resolver"]),
        ("Production ExchangeAccount authority", machine["production_account_authority"]),
        ("Self-mint possible", machine["self_mint_possible"]),
        ("M0.7 status", machine["M0.7_status"]),
        ("M0.8 semantic Order dependency", machine["M0.8_semantic_order_dependency"]),
    )
    external_rows = (
        ("M0.12 1.46.0", external["M0.12_1.46.0"]),
        ("Workspace contract", external["Workspace_contract"]),
        ("Workspace blocker", external["Workspace_blocker"]),
        ("production M0.5", external["production_M0.5"]),
        ("C25", external["C25"]),
        ("M0.8 Full Fill structural foundation", external["M0.8_Full_Fill_structural_foundation"]),
        ("S9D", external["S9D"]),
    )

    result_table = "\n".join(f"| {label} | **{value}** |" for label, value in result_rows)
    external_table = "\n".join(f"| {label} | **{value}** |" for label, value in external_rows)
    submit_fields = ", ".join(f"`{field}`" for field in schema["submit_order_fields"])
    event_fields = ", ".join(f"`{field}`" for field in schema["event_envelope_fields"])
    states = ", ".join(f"`{state}`" for state in lifecycle["states"])
    terminal = ", ".join(f"`{state}`" for state in lifecycle["terminal_states"])
    transitions = "\n".join(f"* `{transition}`" for transition in lifecycle["transitions"])
    binding_fields = ", ".join(f"`{field}`" for field in fill_binding["exact_fields"])
    self_mint_paths = "\n".join(f"* {path}" for path in machine["self_mint_exploit_paths"])
    machine_json = json.dumps(machine, ensure_ascii=False, indent=2)

    return f"""# M0.7 canonical Order authority discovery

This file is generated deterministically from
`m07_order_authority_discovery.json`. It examines repository HEAD
`{machine["repository_head_examined"]}`.

## Result

| Question | Result |
| --- | --- |
{result_table}

The frozen admission contract exists and is closed. No `OrderAuthority` is
implemented by this discovery. `self_mint_possible = {machine["self_mint_possible"]}`
means exactly: {machine["self_mint_scope"]}.

## Canonical schema and identity

Classification: {schema["classification"]}.

Exact `SUBMIT_ORDER` fields: {submit_fields}.

Exact immutable event-envelope fields: {event_fields}.

Command fingerprint: {schema["fingerprints"]["command"]}.

Event fingerprint: {schema["fingerprints"]["event"]}.

Production validator: **{schema["fingerprints"]["production_validator"]}**.

## Lifecycle

States: {states}.

Initial state: **{lifecycle["initial_state"]}**.

Terminal states: {terminal}.

### Exact transition graph

{transitions}

* Reopen legality: {lifecycle["reopen_legality"]}.
* Cancel: {lifecycle["cancel_semantics"]}.
* Reject: {lifecycle["reject_semantics"]}.
* Partial Fill: {lifecycle["partial_fill_semantics"]}.
* Replace: {lifecycle["replace_semantics"]}.

## Red-team self-mint paths

{self_mint_paths}

These paths create non-authoritative runtime/persistence Order-like facts; they
do not create a genuine Core-accepted Order authority fact.

## Dependencies and Fill binding

Order → Instrument: {machine["order_instrument_dependency"]}.

Order → ExchangeAccount: {machine["order_account_dependency"]}.

Exact Fill → Order equality fields: {binding_fields}.

M0.7 structural/lifecycle foundation:
**{machine["M0.7_structural_lifecycle_foundation"]}**.

M0.7 semantic admission: **{machine["M0.7_semantic_admission"]}**.

Next blocker: {machine["next_blocker"]}.

## Preserved external status

| Scope | Status |
| --- | --- |
{external_table}

## Complete machine projection

The block below is the complete canonical machine result. It prevents omitted
detail from becoming an implicit second source of truth.

```json
{machine_json}
```
"""


def _assert_markdown_parity(machine: Mapping[str, object], markdown: str) -> None:
    assert markdown == render_m07_order_authority_discovery(machine)


def test_machine_result_records_exact_head_and_required_disposition() -> None:
    machine, _ = _artifacts()
    assert machine["repository_head_examined"] == "942217531fc0199f6503bcbb5ede316358fabdb2"
    assert machine["canonical_order_schema"] == "FOUND"
    assert machine["canonical_order_lifecycle"] == "FOUND"
    assert machine["production_order_authority"] == "NOT_FOUND"
    assert machine["production_order_store"] == "NOT_FOUND"
    assert machine["historical_order_resolver"] == "NOT_FOUND"
    assert machine["production_account_authority"] == "NOT_FOUND"
    assert machine["self_mint_possible"] == "YES"
    assert machine["M0.7_status"] == "M0.7_ORDER_AUTHORITY_NOT_IMPLEMENTED"
    assert machine["M0.8_semantic_order_dependency"] == "BLOCKED"


def test_contract_fields_and_lifecycle_are_exactly_copied_from_frozen_oracle() -> None:
    machine, _ = _artifacts()
    contract = json.loads(
        (DOCS / "commands_events_order_lifecycle_and_idempotency.json").read_text(encoding="utf-8")
    )
    detail = machine["canonical_order_schema_detail"]
    lifecycle = machine["canonical_order_lifecycle_detail"]
    assert (
        detail["submit_order_fields"]
        == contract["command_registry"]["SUBMIT_ORDER"]["request_fields"]
    )
    assert detail["event_envelope_fields"] == contract["event_contract"]["fields"]
    assert lifecycle["states"] == contract["order_lifecycle"]["states"]
    assert lifecycle["terminal_states"] == contract["order_lifecycle"]["terminal_states"]


def test_markdown_machine_parity_for_decisions_and_preserved_status() -> None:
    machine, markdown = _artifacts()
    _assert_markdown_parity(machine, markdown)


def test_parity_rejects_wrong_order_authority_value() -> None:
    machine, markdown = _artifacts()
    machine["production_order_authority"] = "FOUND"
    with pytest.raises(AssertionError):
        _assert_markdown_parity(machine, markdown)


def test_parity_rejects_swapped_store_and_resolver_values() -> None:
    machine, _ = _artifacts()
    machine["production_order_store"] = "STORE_SENTINEL"
    machine["historical_order_resolver"] = "RESOLVER_SENTINEL"
    correctly_bound = deepcopy(machine)
    swapped = deepcopy(machine)
    swapped["production_order_store"], swapped["historical_order_resolver"] = (
        swapped["historical_order_resolver"],
        swapped["production_order_store"],
    )
    markdown = render_m07_order_authority_discovery(correctly_bound)
    with pytest.raises(AssertionError):
        _assert_markdown_parity(swapped, markdown)


def test_parity_rejects_missing_required_section() -> None:
    machine, _ = _artifacts()
    expected = render_m07_order_authority_discovery(machine)
    without_status_section = expected.replace("## Preserved external status\n", "", 1)
    with pytest.raises(AssertionError):
        _assert_markdown_parity(machine, without_status_section)


def test_parity_rejects_preserved_external_status_drift() -> None:
    machine, markdown = _artifacts()
    machine["frozen_external_status"]["M0.12_1.46.0"] = "PROPOSED"
    with pytest.raises(AssertionError):
        _assert_markdown_parity(machine, markdown)


def test_discovery_does_not_claim_a_fake_authority() -> None:
    machine, markdown = _artifacts()
    assert machine["authority_review"]["production_admission_api"] == "NOT_FOUND"
    assert machine["M0.7_structural_lifecycle_foundation"] == "AVAILABLE_CONTRACT_AND_TEST_ORACLE"
    assert machine["M0.7_semantic_admission"].startswith("BLOCKED_ON_PRODUCTION_AUTHORITY")
    assert "No `OrderAuthority` is\nimplemented by this discovery." in markdown
