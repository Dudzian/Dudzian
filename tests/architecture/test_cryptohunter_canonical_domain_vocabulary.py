from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "docs/architecture/cryptohunter_product_architecture/canonical_domain_vocabulary.json"
DATA = json.loads(DOC.read_text(encoding="utf-8"))
UUIDV7_REGEX = r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"

REQUIRED_ENTITY_FIELDS = {
    "canonical_name", "id_field", "id_prefix", "purpose", "parent", "lifecycle_scope",
    "persistence", "saas_sync_candidate", "secret_policy", "relationships", "legacy_names",
}
REQUIRED_AXES = {
    "product_capability", "live_access_grant", "active_trading_environment", "runtime_state",
    "kill_switch_state", "operator_interface_authentication", "exchange_account_connection_state",
    "execution_authorization",
}
REQUIRED_ENTITIES = {
    "CryptoHunterAccount", "DeviceInstallation", "OperatorIdentity", "LiveAccessGrant", "Workspace",
    "Portfolio", "ExchangeAccount", "CredentialProfile", "TradingUniverse", "Instrument",
    "StrategyDefinition", "StrategyInstance", "MarketDataRoute", "ExecutionRoute", "RiskPolicy",
    "RiskBudget", "ExecutionLease", "Signal", "Decision", "OrderIntent", "Order", "Fill",
    "LedgerEntry", "RuntimeSession", "AuditEvent",
}


def _entities() -> dict[str, dict[str, object]]:
    return {e["canonical_name"]: e for e in DATA["entity_kinds"]}


def _relationships() -> dict[tuple[str, str], str]:
    return {(r["from"], r["to"]): r["cardinality"] for r in DATA["relationships"]}


def test_schema_and_baseline() -> None:
    assert DATA["schema_version"] == "cryptohunter.canonical_domain_vocabulary.v1"
    assert re.fullmatch(r"[0-9a-f]{40}", DATA["baseline_commit"])


def test_environment_sets() -> None:
    envs = {item["name"] for item in DATA["public_trading_environments"]}
    assert envs == {"paper", "testnet", "live"}
    assert "demo" not in envs
    assert "sandbox" not in envs
    assert {"smoke", "fixture"}.issubset(set(DATA["internal_test_modes"]))


def test_required_axes_entities_and_unique_ids() -> None:
    axes = {axis["name"] for axis in DATA["state_axes"]}
    assert REQUIRED_AXES <= axes
    entities = DATA["entity_kinds"]
    names = [e["canonical_name"] for e in entities]
    id_fields = [e["id_field"] for e in entities]
    prefixes = [e["id_prefix"] for e in entities]
    assert REQUIRED_ENTITIES == set(names)
    assert len(entities) == 25
    assert len(names) == len(set(names))
    assert len(id_fields) == len(set(id_fields))
    assert len(prefixes) == len(set(prefixes))
    assert "lgrant" in prefixes
    assert all(re.fullmatch(r"[a-z][a-z0-9]*", p) for p in prefixes)
    assert DATA["identifier_policy"]["persistent_id_format"] == "<prefix>_<uuidv7>"
    assert DATA["identifier_policy"]["regex"] == UUIDV7_REGEX


def test_each_entity_has_required_fields_and_clean_legacy_names() -> None:
    for entity in DATA["entity_kinds"]:
        assert REQUIRED_ENTITY_FIELDS <= entity.keys()
        assert isinstance(entity["legacy_names"], list)
        assert all(isinstance(name, str) and name.strip() for name in entity["legacy_names"])


def test_corrected_ownership_and_cardinality() -> None:
    entities = _entities()
    rels = _relationships()
    assert entities["ExchangeAccount"]["parent"] == "Portfolio"
    assert rels[("Portfolio", "ExchangeAccount")] == "one_to_many"
    assert "many_to_many_aggregation" not in rels.values()
    assert entities["Instrument"]["parent"] == "Workspace"
    assert rels[("TradingUniverse", "Instrument")] == "many_to_many_membership"
    assert entities["Instrument"]["identity_dimensions"] == [
        "exchange_id", "environment", "market_type", "venue_symbol"
    ]
    assert entities["StrategyInstance"]["parent"] == "Workspace"
    assert rels[("StrategyDefinition", "StrategyInstance")] == "one_to_many_reference"
    assert entities["Decision"]["parent"] == "Workspace"
    assert rels[("Workspace", "Decision")] == "one_to_many"
    assert rels[("Signal", "Decision")] == "one_to_many_possible_source"
    assert rels[("OperatorIdentity", "Decision")] == "one_to_many_possible_source"
    assert rels[("RuntimeSession", "Decision")] == "one_to_many_possible_source"
    assert entities["LedgerEntry"]["parent"] == "Portfolio"


def test_live_access_grant_and_ledger_model() -> None:
    entities = _entities()
    rels = _relationships()
    grant = entities["LiveAccessGrant"]
    assert grant["id_field"] == "live_access_grant_id"
    assert grant["id_prefix"] == "lgrant"
    assert grant["parent"] == "DeviceInstallation"
    assert grant["persistence"] is True
    assert grant["saas_sync_candidate"] is True
    assert rels[("DeviceInstallation", "LiveAccessGrant")] == "one_to_many_history"
    assert rels[("OperatorIdentity", "LiveAccessGrant")] == "authorization_reference"

    ledger = entities["LedgerEntry"]
    source_types = set(ledger["allowed_source_event_types"])
    assert {"fill", "fee", "funding", "deposit", "withdrawal", "internal_transfer", "reconciliation_correction"} <= source_types
    assert {"exchange_account_id", "strategy_instance_id", "order_id", "fill_id"} <= set(ledger["optional_references"])
    assert ("LedgerEntry", "AuditEvent") not in rels
    assert rels[("Portfolio", "LedgerEntry")] == "one_to_many"
    assert rels[("Fill", "LedgerEntry")] == "one_to_many_possible_source"


def test_decision_sources_and_manual_lifecycle() -> None:
    entities = _entities()
    rels = _relationships()
    decision = entities["Decision"]
    assert set(decision["decision_source_types"]) == {
        "strategy_signal", "operator_action", "risk_system", "recovery_policy", "reconciliation_import"
    }
    assert set(decision["optional_source_references"]) == {
        "signal_id", "operator_id", "runtime_session_id", "source_event_id"
    }
    assert rels[("OperatorIdentity", "Decision")] == "one_to_many_possible_source"
    assert rels[("Decision", "OrderIntent")] == "one_to_many"
    blob = "\n".join(DATA["invariants"]).lower()
    assert "exactly one explicit source/provenance" in blob
    assert "strategy_signal requires signal_id" in blob
    assert "operator_action requires operator_id" in blob
    assert "synthetic signal" in blob


def test_audit_event_can_exist_without_runtime_session() -> None:
    entities = _entities()
    rels = _relationships()
    audit = entities["AuditEvent"]
    assert audit["parent"] == "DeviceInstallation"
    assert rels[("DeviceInstallation", "AuditEvent")] == "one_to_many"
    assert rels[("RuntimeSession", "AuditEvent")] == "optional_runtime_reference"
    assert ("RuntimeSession", "AuditEvent", "one_to_many") not in {
        (r["from"], r["to"], r["cardinality"]) for r in DATA["relationships"]
    }
    assert set(audit["optional_references"]) == {
        "runtime_session_id", "operator_id", "workspace_id", "exchange_account_id",
        "order_id", "ledger_entry_id"
    }
    assert set(audit["audit_event_categories"]) == {
        "authentication", "authorization", "configuration", "credential_management",
        "device_management", "licensing", "live_activation", "runtime", "trading",
        "risk", "recovery", "update", "security"
    }
    blob = "\n".join(DATA["invariants"]).lower()
    assert "auditevent may exist without runtimesession" in blob
    assert "before core starts" in blob
    assert "auditevent is immutable and append-only" in blob
    assert "must not contain secrets or biometric data" in blob


def test_full_signal_to_ledger_lifecycle() -> None:
    rels = _relationships()
    expected = {
        ("StrategyInstance", "Signal"): "one_to_many",
        ("Signal", "Decision"): "one_to_many_possible_source",
        ("Decision", "OrderIntent"): "one_to_many",
        ("OrderIntent", "Order"): "one_to_many",
        ("Order", "Fill"): "one_to_many",
        ("Fill", "LedgerEntry"): "one_to_many_possible_source",
    }
    for edge, cardinality in expected.items():
        assert rels[edge] == cardinality


def test_parent_and_local_relationship_references_are_valid() -> None:
    entities = _entities()
    rels = _relationships()
    names = set(entities)
    for entity in entities.values():
        parent = entity["parent"]
        assert parent == "none" or parent in names
        if parent != "none":
            assert (parent, entity["canonical_name"]) in rels
        for local in entity["relationships"]:
            assert len(local) == 3
            assert tuple(local) in {(a, b, c) for (a, b), c in rels.items()} or local[-1] == "derived"


def test_required_legacy_rules_and_invariants() -> None:
    terms = {c["term"] for c in DATA["legacy_conflicts"]} | {a["term"] for a in DATA["deprecated_environment_aliases"]}
    assert {"demo", "sandbox", "production/prod", "primary"} <= terms
    blob = "\n".join(DATA["invariants"]).lower()
    for phrase in (
        "live", "kill switch", "gui", "core", "executionlease", "at most one executionlease",
        "order cannot be created without a source orderintent", "provenance",
        "exactly one explicit source/provenance", "auditevent may exist without runtimesession",
    ):
        assert phrase in blob


def test_no_duplicate_top_level_relationship_pairs() -> None:
    pairs = [(r["from"], r["to"]) for r in DATA["relationships"]]
    assert len(pairs) == len(set(pairs))


def test_evidence_paths_are_relative_existing_paths() -> None:
    paths = set(DATA["evidence_paths"])
    for conflict in DATA["legacy_conflicts"]:
        paths.update(conflict["evidence_paths"])
    for raw in paths:
        p = Path(raw)
        assert not p.is_absolute(), raw
        assert ".." not in p.parts, raw
        assert (ROOT / p).exists(), raw


def test_no_values_that_look_like_real_secrets() -> None:
    text = json.dumps(DATA, sort_keys=True)
    forbidden = [
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"xox[baprs]-[A-Za-z0-9-]{20,}",
        r"sk-[A-Za-z0-9]{20,}",
    ]
    for pattern in forbidden:
        assert re.search(pattern, text) is None
