"""Executable parity and red-team checks for hybrid M0.8 Fill discovery."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Callable
import unicodedata

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE = DOCS / "m08_full_fill_authority_discovery.json"
MARKDOWN = DOCS / "m08_full_fill_authority_discovery.md"
CONTRACT = DOCS / "commands_events_order_lifecycle_and_idempotency.json"
STATUS = DOCS / "m07_order_authority_kernel_status.json"
LEDGER = DOCS / "ledger_portfolio_capital_and_pnl.json"
INSTRUMENTS = DOCS / "exchange_accounts_and_instruments.json"


def load() -> tuple[dict[str, object], str]:
    return json.loads(MACHINE.read_text(encoding="utf-8")), MARKDOWN.read_text(encoding="utf-8")


def render(machine: dict[str, object]) -> str:
    return (
        "# M0.8 Full Fill authority discovery\n\n"
        "This file is a deterministic complete projection of "
        "`m08_full_fill_authority_discovery.json`. JSON is the source of truth.\n\n"
        f"```json\n{json.dumps(machine, indent=2, ensure_ascii=False)}\n```\n"
    )


def pointer(document: object, path: str) -> object:
    value = document
    for raw in path.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        assert isinstance(value, (dict, list))
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def digest(value: object) -> str:
    def normalize(item: object) -> object:
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        if isinstance(item, dict):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, list):
            return [normalize(child) for child in item]
        return item

    payload = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def validate(machine: dict[str, object]) -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    fill = contract["fill_contract"]
    assert machine["provenance_mode"] == "HYBRID_HEURISTIC"
    assert machine["equivalence_class"] == "MISMATCH"
    assert machine["formal_acceptance_allowed"] is False
    assert machine["candidate_implementation_allowed"] is False
    assert machine["m07_equivalence"]["semantic_differences"]
    assert machine["m07_equivalence"]["required_semantics"] == {
        "production_order_authority_kernel": status["production_order_authority_kernel"],
        "production_order_history": status["production_order_history"],
        "historical_order_resolver": status["historical_order_resolver"],
        "semantic_submit_order": status["semantic_submit_order"],
        "M0.7": status["M0.7"],
    }
    schema = machine["canonical_fill_schema"]
    assert schema["contract"].endswith(CONTRACT.name)
    assert schema["json_pointer"] == "/fill_contract"
    assert schema["exact_fields"] == fill["fact_fields"]
    assert schema["nullable_fields"] == fill["nullable_fields"]
    assert schema["fingerprint"]["domain"] == fill["fingerprint"]["input_fields"]
    assert machine["structural_fill_foundation"]["status"] == "ACCEPTED_CLOSED"
    assert machine["production_full_fill_authority"]["status"] == "NOT_FOUND"
    assert machine["production_full_fill_authority"]["architectural_owner"] == "M0.7"
    assert machine["production_fill_store"] == "NOT_FOUND"
    assert machine["historical_fill_resolver"] == "NOT_FOUND"
    assert machine["fill_order_binding"]["exact_fields"] == [
        "order_id", "environment", "workspace_id", "portfolio_id",
        "exchange_account_id", "exchange_id", "instrument_id", "execution_route_id",
    ]
    assert machine["fill_instrument_binding_status"] == "CONTRACT_INCONSISTENT"
    assert machine["fill_instrument_binding"]["M0.7_fill_migration_required"] == "UNRESOLVED"
    assert machine["fill_instrument_binding"]["historical_instrument_authority"] == "NOT_AVAILABLE"
    assert machine["execution_producer_authenticity"]["catalog_source_producer_membership_is_execution_proof"] is False
    assert machine["execution_producer_authenticity"]["status"] == "NOT_FOUND"
    assert machine["venue_trade_provenance"] == {
        "classification": "NOT_FOUND", "caller_supplied_venue_trade_id_is_proof": False
    }
    assert machine["fifo_accounting_authority"]["status"] == "NOT_FOUND"
    ownership = machine["fill_progression_ownership"]
    assert ownership["full_fill_admission_history_authority_owner"] == "M0.7"
    assert ownership["fill_identity_dedupe_owner"] == "M0.7"
    assert ownership["external_venue_trade_dedupe_owner"] == "M0.7"
    assert ownership["accepted_fill_ids_by_order_id_owner"] == "M0.7"
    assert ownership["cumulative_order_execution_progression_owner"] == "M0.7"
    assert ownership["partial_full_overfill_lifecycle_enforcement_owner"] == "M0.7"
    assert ownership["M0.8_role"].startswith("DOWNSTREAM_ACCOUNTING_CONSUMER_ONLY")
    assert ownership["M0.8_fill_authority_owner"] is False
    assert ownership["M0.8_independent_external_venue_trade_admission"] is False
    invariant = machine["single_fill_authority_invariant"]
    assert invariant["canonical_accepted_fill_authority_owner"] == "M0.7"
    assert invariant["canonical_accepted_fill_history_owner"] == "M0.7"
    assert invariant["m08_is_downstream_consumer_only"] is True
    assert invariant["competing_fill_authority_forbidden"] is True
    assert invariant["m08_may_mint_or_admit_fill"] is False
    assert invariant["m08_may_deduplicate_external_venue_trade_independently"] is False
    assert machine["authority_boundary_red_team"]["classification"] == (
        "IMPOSSIBLE_BY_AUTHORITY_BOUNDARY"
    )
    assert machine["self_mint_fill_authority_fact"]["result"] == "NO"
    assert machine["fill_authority_kernel_buildability"]["classification"] == (
        "BLOCKED_BY_CROSS_CONTRACT_FILL_INSTRUMENT_BINDING"
    )
    assert machine["next_buildable_stage"] == "M07_M05_FILL_INSTRUMENT_BINDING_RECONCILIATION"
    assert machine["cross_contract_consistency"]["overall_status"] == "DRIFT_FOUND"
    assert machine["M0.8_historical_target_contract"] == "CLOSED_AT_ORIGINAL_BASELINE"
    assert machine["M0.8_current_upstream_compatibility"] == "CONTRACT_INCONSISTENT"
    preserved = machine["preserved_status"]
    assert preserved["M0.7_OrderAuthority_kernel"] == "ACCEPTED_AVAILABLE"
    assert preserved["M0.7_semantic_SUBMIT_ORDER"] == "BLOCKED_UPSTREAM"
    assert preserved["M0.8_structural_foundation"] == "CLOSED"
    assert machine["M0.8_status"] == "NOT_AVAILABLE"


def test_markdown_is_deterministic_complete_projection() -> None:
    machine, markdown = load()
    assert markdown == render(machine)


def test_discovery_matches_current_contract_and_preserved_status() -> None:
    machine, _ = load()
    validate(machine)


def test_authority_relevant_equivalence_paths_exist_and_runtime_invariants_are_present() -> None:
    machine, _ = load()
    for relative in machine["m07_equivalence"]["authority_relevant_paths_compared"]:
        assert (ROOT / relative).is_file(), relative
    authority = (ROOT / "bot_core/orders/authority.py").read_text(encoding="utf-8")
    custody = (ROOT / "bot_core/orders/custody.py").read_text(encoding="utf-8")
    tests = (ROOT / "tests/architecture/test_m07_order_authority.py").read_text(encoding="utf-8")
    for literal in ("SEMANTIC_SUBMIT_ORDER_AVAILABLE: Final = False", "if type(authority) is not OrderAuthority"):
        assert literal in authority
    for literal in ("HMAC-SHA-256", "KeyringOrderAuthoritySecretCustody", "PRODUCTION_ORDER_AUTHENTICITY_PURPOSE", "TEST_ORDER_AUTHENTICITY_PURPOSE"):
        assert literal in custody
    for literal in ("test_coherent_public_sha_sql_mint_without_mac_is_denied", "test_coherent_public_rewrite_without_new_mac_is_denied"):
        assert literal in tests


def test_every_m08_dependency_attestation_is_recomputed_and_every_drift_reported() -> None:
    machine, _ = load()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    reported = {
        (row["contract"], row["json_pointer"]): row
        for row in machine["cross_contract_consistency"]["dependencies"]
    }
    expected_drift = set()
    assert len(reported) == len(ledger["cross_contract_dependencies"])
    for dependency in ledger["cross_contract_dependencies"]:
        key = dependency["contract"], dependency["json_pointer"]
        document = json.loads((DOCS / dependency["contract"]).read_text(encoding="utf-8"))
        current = digest(pointer(document, dependency["json_pointer"]))
        status = "MATCH" if current == dependency["content_fingerprint_sha256"] else "DRIFT"
        assert reported[key] == {
            "contract": dependency["contract"],
            "json_pointer": dependency["json_pointer"],
            "declared_fingerprint": dependency["content_fingerprint_sha256"],
            "current_recomputed_fingerprint": current,
            "status": status,
        }
        if status == "DRIFT":
            expected_drift.add(key)
    actual_drift = {
        (row["contract"], row["json_pointer"])
        for row in machine["cross_contract_consistency"]["drifted_pointers"]
    }
    assert actual_drift == expected_drift
    if expected_drift:
        assert machine["m07_equivalence"]["semantic_differences"]
        assert machine["candidate_implementation_allowed"] is False


def test_current_fill_to_m05_field_incompatibility_blocks_kernel() -> None:
    machine, _ = load()
    instruments = json.loads(INSTRUMENTS.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    record_fields = set(instruments["instrument_contract"]["record_fields"])
    required = contract["fill_contract"]["instrument_binding"]["required_matches"]
    assert "source_exchange_id" in record_fields
    assert not {"exchange_id", "environment"} & record_fields
    assert {"exchange_id", "environment"} <= set(required)
    assert machine["fill_instrument_binding_status"] != "CONSISTENT"
    assert machine["candidate_implementation_allowed"] is False
    assert machine["fill_instrument_binding"]["option_a_status"].startswith("UNPROVEN")


def test_canonical_m07_to_m08_fill_ownership_is_single_authority() -> None:
    machine, _ = load()
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    order = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert ledger["source_registry"]["fill"]["authority"] == (
        "M0.7 composite trusted accepted Full Fill"
    )
    complete_history = order["fill_contract"]["trust_boundaries"][
        "complete_accepted_history"
    ]
    assert "accepted_fill_ids_by_order_id" in complete_history
    assert "accepted M0.7 lifecycle history" in complete_history
    assert machine["single_fill_authority_invariant"] == {
        "canonical_accepted_fill_authority_owner": "M0.7",
        "canonical_accepted_fill_history_owner": "M0.7",
        "m08_is_downstream_consumer_only": True,
        "competing_fill_authority_forbidden": True,
        "m08_may_mint_or_admit_fill": False,
        "m08_may_deduplicate_external_venue_trade_independently": False,
        "journal_separation": (
            "M0.7 Fill journal/history != M0.8 LedgerEntry/accounting journal; a downstream "
            "accepted-source reference/evidence never becomes Fill authority"
        ),
    }
    assert machine["authority_boundary_red_team"]["required_result"].startswith(
        "M0.8 rejects F1"
    )


Mutation = Callable[[dict[str, object]], None]


@pytest.mark.parametrize("mutation", [
    lambda d: d["production_full_fill_authority"].update(status="FOUND"),
    lambda d: d["fill_order_binding"].update(exact_fields=[]),
    lambda d: d["fill_instrument_binding"].update(historical_instrument_authority="AVAILABLE"),
    lambda d: d["execution_producer_authenticity"].update(catalog_source_producer_membership_is_execution_proof=True),
    lambda d: d["venue_trade_provenance"].update(caller_supplied_venue_trade_id_is_proof=True),
    lambda d: d["fifo_accounting_authority"].update(status="FOUND"),
    lambda d: d["preserved_status"].update({"M0.7_OrderAuthority_kernel": "NOT_AVAILABLE"}),
    lambda d: d["preserved_status"].update({"M0.7_semantic_SUBMIT_ORDER": "AVAILABLE"}),
    lambda d: d.update(formal_acceptance_allowed=True),
    lambda d: d.update(provenance_mode="EXACT_COMMIT"),
    lambda d: d.update(candidate_implementation_allowed=True),
    lambda d: d.update(fill_instrument_binding_status="CONSISTENT"),
    lambda d: d["fill_authority_kernel_buildability"].update(classification="BUILDABLE_INDEPENDENTLY"),
    lambda d: d["fill_progression_ownership"].update(
        {"M0.8_role": "owns genuine Fill admission/history"}
    ),
    lambda d: d["production_full_fill_authority"].update(architectural_owner="M0.8"),
    lambda d: d["fill_progression_ownership"].update(
        {"M0.8_independent_external_venue_trade_admission": True}
    ),
    lambda d: d["single_fill_authority_invariant"].update(canonical_accepted_fill_history_owner=False),
])
def test_security_claim_mutations_fail(mutation: Mutation) -> None:
    machine, _ = load()
    changed = deepcopy(machine)
    mutation(changed)
    with pytest.raises(AssertionError):
        validate(changed)
