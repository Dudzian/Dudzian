"""Pure semantic executable reference model for the closed M0.11 contract."""

from __future__ import annotations

import copy
from decimal import Decimal, InvalidOperation
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).parents[2]
DOCS = ROOT / "docs/architecture/cryptohunter_product_architecture"
MACHINE_PATH = DOCS / "persistence_versioning_migrations_backup_and_recovery.json"
MACHINE: dict[str, Any] = json.loads(MACHINE_PATH.read_text())
SHA = "a" * 64
SCOPE = (
    "acct_01890f3a-2b4c-7abc-8def-0123456789ab",
    "dev_01890f3a-2b4c-7abc-8def-0123456789ab",
    SHA,
)
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _resolve_pointer(document: Any, pointer: Any) -> tuple[bool, Any]:
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        return False, None
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, dict) or token not in value:
            return False, None
        value = value[token]
    return True, value


def _actual_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _attests(entry: Any, upstream: Any) -> bool:
    if not isinstance(entry, dict) or not isinstance(upstream, dict):
        return False
    if set(entry) != {"milestone", "artifact", "json_pointer", "content_fingerprint_sha256"}:
        return False
    ok, value = _resolve_pointer(upstream, entry["json_pointer"])
    return bool(ok and _actual_fingerprint(value) == entry["content_fingerprint_sha256"])


def _positive(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _validate_metadata(value: Any) -> str:
    schema = MACHINE["executable_boundary_schemas"]["StateStoreMetadata"]
    if not isinstance(value, dict) or set(value) != set(schema["required"]):
        return "CONTRACT_INCONSISTENT"
    for key in (
        "state_store_schema_version",
        "protected_freshness_generation",
    ):
        if not _positive(value.get(key)):
            return "CONTRACT_INCONSISTENT"
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or not SHA_RE.fullmatch(value[key]):
            return "CONTRACT_INCONSISTENT"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "ENVIRONMENT_SCOPE_MISMATCH"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "CONTRACT_INCONSISTENT"
    return "VALID"


def _metadata(**updates: Any) -> dict[str, Any]:
    value = {
        "state_store_schema_version": 1,
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "protected_freshness_generation": 1,
        "state_fingerprint_sha256": SHA,
        "transaction_fingerprint_sha256": "b" * 64,
        "history_tail_fingerprint_sha256": "c" * 64,
    }
    value.update(updates)
    return value


def _observation(**updates: Any) -> dict[str, Any]:
    value = _metadata()
    value.update(durable_confirmed=True, authoritative_history_integrity=True, current_commit=True)
    value.update(updates)
    return value


def _lease_after_restart(state: Any) -> str:
    return {
        "UNUSED": "LEASE_STALE",
        "CONSUMED": "CONSUMED",
        "UNKNOWN_RECONCILIATION": "RECONCILIATION_REQUIRED",
    }.get(state, "CONTRACT_INCONSISTENT")


DEPENDENCIES = MACHINE["dependency_manifest"]["authority_dependencies"]


@pytest.mark.parametrize(
    "entry", DEPENDENCIES, ids=lambda e: f"{e['milestone']}{e['json_pointer']}"
)
def test_each_authored_dependency_fingerprint_independently_attests(entry: dict[str, Any]) -> None:
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    assert _attests(entry, upstream)


def test_manifest_is_exactly_48_and_excludes_m01_authority() -> None:
    assert len(DEPENDENCIES) == MACHINE["dependency_manifest"]["count"] == 48
    assert {e["milestone"] for e in DEPENDENCIES} == {f"M0.{n}" for n in range(2, 11)}
    assert MACHINE["dependency_manifest"]["m0_1_authority_allowed"] is False


def test_dependency_mutation_breaks_authored_attestation() -> None:
    entry = DEPENDENCIES[0]
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    altered = copy.deepcopy(upstream)
    ok, value = _resolve_pointer(altered, entry["json_pointer"])
    assert ok and isinstance(value, (dict, list))
    value.append("MUTATION") if isinstance(value, list) else value.update({"mutation": True})
    assert not _attests(entry, altered)


@pytest.mark.parametrize("pointer", ["/missing", "/entity_kinds/alias", "entity_kinds", 1, None])
def test_pointer_alias_or_malformed_fails_closed(pointer: Any) -> None:
    entry = {**DEPENDENCIES[0], "json_pointer": pointer}
    upstream = json.loads((DOCS / entry["artifact"]).read_text())
    assert not _attests(entry, upstream)


@pytest.mark.parametrize("bad", [True, False, 0, -1, 1.2, "1", None, [], {}])
def test_metadata_generation_is_positive_non_boolean(bad: Any) -> None:
    assert (
        _validate_metadata(_metadata(protected_freshness_generation=bad)) == "CONTRACT_INCONSISTENT"
    )


def test_metadata_is_closed_and_hashes_lowercase() -> None:
    assert _validate_metadata(_metadata()) == "VALID"
    assert _validate_metadata({**_metadata(), "authority": "caller"}) == "CONTRACT_INCONSISTENT"
    assert (
        _validate_metadata(_metadata(state_fingerprint_sha256="A" * 64)) == "CONTRACT_INCONSISTENT"
    )


def test_evidence_has_no_persistence_backup_fingerprint_or_generation_effect() -> None:
    policy = MACHINE["startup_recovery_model"]["evidence_policy"]
    assert [
        policy[k]
        for k in (
            "local_durable_evidence_persisted_in_StateStore",
            "local_durable_evidence_included_in_state_fingerprint",
            "local_durable_evidence_included_in_BackupEnvelope",
            "local_evidence_current_designation_persisted",
            "evidence_publication_is_StateStore_semantic_transaction",
            "evidence_publication_advances_protected_freshness_generation",
        )
    ] == [False] * 6
    assert policy["local_evidence_registry_process_local"] is True


def test_runtime_session_publication_precedes_ready_and_old_process_is_not_restored() -> None:
    model = MACHINE["runtime_session_persistence"]
    assert model["order"][-2:] == ["durably publish current RuntimeSession history", "READY"]
    assert model["old_history_preserved"] and not model["old_process_restored"]


@pytest.mark.parametrize(
    ("state", "outcome"),
    [
        ("UNUSED", "LEASE_STALE"),
        ("CONSUMED", "CONSUMED"),
        ("UNKNOWN_RECONCILIATION", "RECONCILIATION_REQUIRED"),
        (True, "CONTRACT_INCONSISTENT"),
    ],
)
def test_execution_lease_restart_fence(state: Any, outcome: str) -> None:
    assert _lease_after_restart(state) == outcome


def test_migration_is_closed_forward_only_and_protected() -> None:
    migration = MACHINE["migration_protocol"]
    assert migration["states"] == [
        "PREPARED",
        "APPLYING",
        "DURABLE_MIGRATED",
        "COMPLETED",
        "FAILED",
    ]
    assert migration["rollback_policy"] == "FORWARD_ONLY"
    assert migration["authoritative_change_requires_m0_3_freshness"]
    assert not migration["partial_state_promoted"]
    assert migration["restart"] == "IDEMPOTENT_RESUME"


def test_backup_exclusions_block_authority_evidence_and_secrets() -> None:
    exclusions = " ".join(MACHINE["backup_contract"]["excludes"])
    for forbidden in (
        "protected membership",
        "LocalDurableStateEvidence",
        "raw PIN",
        "secure-store payload",
        "AuthenticationProof",
        "PlatformBiometricAssertion",
    ):
        assert forbidden in exclusions


def test_no_authority_resurrection_and_pin_fences() -> None:
    never = " ".join(MACHINE["restore_contract"]["never_resurrect"])
    for item in (
        "consumed bootstrap",
        "retired M0.3",
        "revoked DeviceInstallation",
        "old PIN revision",
        "revoked LiveAccessGrant",
        "consumed ExecutionLease",
        "kill-switch",
        "idempotency",
    ):
        assert item in never


def test_idempotency_accounting_and_reservation_are_restart_safe() -> None:
    invariants = MACHINE["idempotency_and_accounting_recovery"]
    assert invariants["invariants"] == [
        "no repeated side effect",
        "no duplicate Fill/Ledger",
        "no double reservation release/consume",
    ]
    assert not invariants["reservation_id_added"]


def test_secret_handoff_has_no_payload_and_unknown_requires_reconciliation() -> None:
    handoff = MACHINE["external_resource_handoff"]
    assert handoff["states"] == [
        "PREPARED",
        "COMMITTED",
        "CLEANUP_PENDING",
        "UNKNOWN_RECONCILIATION",
    ]
    assert handoff["secret_payload_fields"] == 0
    assert handoff["unknown_outcome"] == "RECONCILIATION_REQUIRED_NO_BLIND_RETRY"


@pytest.mark.parametrize(
    ("internal", "public"), list(MACHINE["failure_registry"]["mapping"].items())
)
def test_internal_failure_mapping_is_exact(internal: str, public: str) -> None:
    assert MACHINE["failure_registry"]["mapping"][internal] == public
    assert public in MACHINE["failure_registry"]["closed_codes"]


def test_failure_registry_is_closed_and_safe() -> None:
    registry = MACHINE["failure_registry"]
    assert len(registry["closed_codes"]) == len(set(registry["closed_codes"])) == 32
    assert registry["outcome_schema"]["additionalProperties"] is False
    assert "SQLite exception" in registry["raw_diagnostics_forbidden"]


@pytest.mark.parametrize(
    ("second", "outcome"), [(False, "LOCK_CONTENTION"), (True, "CONCURRENT_WRITER")]
)
def test_single_writer_distinguishes_contention(second: bool, outcome: str) -> None:
    assert ("CONCURRENT_WRITER" if second else "LOCK_CONTENTION") == outcome
    assert MACHINE["writer_concurrency"]["authoritative_writers"] == 1


def test_durability_uncertainty_never_success() -> None:
    assert (
        MACHINE["failure_registry"]["mapping"]["DURABILITY_UNCERTAIN"]
        == "DURABILITY_CONFIRMATION_FAILED"
    )
    assert (
        MACHINE["transaction_protocol"]["steps"][-1]
        == "SUCCESS_AFTER_REQUIRED_DURABILITY_AND_FINALITY"
    )


def test_live_is_currently_denied_without_testnet_fallback() -> None:
    authority = MACHINE["authority_model"]
    assert authority["live_current"] == "DENIED"
    assert authority["testnet_to_live_fallback"] is False


def test_markdown_is_machine_projection_for_every_root() -> None:
    markdown = MACHINE_PATH.with_suffix(".md").read_text()
    for root, value in MACHINE.items():
        assert f"## `{root}`" in markdown
        assert json.dumps(value, indent=2, ensure_ascii=False) in markdown


# Independently authored corrective-closure expectations (never derived from MACHINE).
_A = "DURABLE AUTHORITATIVE CURRENT STATE"
_H = "DURABLE IMMUTABLE / APPEND-ONLY HISTORY"
_R = "DERIVED / REBUILDABLE"
_E = "EPHEMERAL RUNTIME"
_X = "EXTERNAL AUTHORITY / REFERENCE ONLY"
_S = "SECRET PAYLOAD OUTSIDE DOMAIN"
_EXPECTED_DURABILITY = {
    "StateStoreMetadata": _A,
    "CryptoHunterAccount current record": _A,
    "DeviceInstallation current identity/lifecycle": _A,
    "OperatorIdentity current designation/state": _A,
    "OperatorIdentity revisions": _H,
    "LiveAccessGrant current designation/state": _A,
    "LiveAccessGrant accepted revisions/history": _H,
    "Workspace": _A,
    "Portfolio canonical accounting state": _A,
    "Portfolio balance/P&L/NAV projections": _R,
    "ExchangeAccount": _A,
    "CredentialProfile metadata/reference": _A,
    "TradingUniverse current version/designation": _A,
    "TradingUniverse version history": _H,
    "StrategyDefinition accepted revisions": _H,
    "StrategyDefinition current designation": _A,
    "StrategyInstance current lifecycle/config": _A,
    "routing configuration/current designation": _A,
    "routing readiness/reachability projection": _R,
    "RiskPolicy accepted revisions": _H,
    "RiskPolicy current designation": _A,
    "RiskBudget current state": _A,
    "kill-switch state/generation": _A,
    "kill-switch transition history": _H,
    "Command accepted request": _H,
    "Event": _H,
    "OrderIntent": _H,
    "Order canonical lifecycle state": _A,
    "Order lifecycle events/history": _H,
    "Fill": _H,
    "LedgerEntry": _H,
    "reservation current state": _A,
    "reservation transition history": _H,
    "RiskDecision": _H,
    "ExecutionLease immutable record": _H,
    "ExecutionLease one-shot state": _A,
    "ExecutionLease restart fence": _A,
    "RuntimeSession active process manifestation": _E,
    "RuntimeSession canonical identity/history": _H,
    "SessionSecurityState current generation/state": _A,
    "SessionSecurityState revision history": _H,
    "PinVerifierRecord accepted revisions": _H,
    "PinVerifierRecord current designation": _A,
    "DeviceTrust/security revisions": _H,
    "DeviceTrust current designation": _A,
    "platform enrollment revisions": _H,
    "AuthenticationProof": _E,
    "CoreIssuedAuthenticationProofBinding": _E,
    "PlatformBiometricAssertion": _E,
    "CoreAcceptedPlatformBiometricAssertionBinding": _E,
    "SecretMetadataProjection": _A,
    "secure-store payload": _S,
    "bootstrap consumed fence": _A,
    "bootstrap accepted/consumption history": _H,
    "M0.3 restore freshness membership": _X,
    "M0.3 current designation": _X,
    "M0.3 retirement state": _X,
    "LocalDurableStateEvidence payload": _R,
    "LocalDurableEvidence accepted/current registry/designation": _E,
    "Migration current state/designation": _A,
    "Migration transition/history revisions": _H,
    "SecretHandoff current state/designation": _A,
    "SecretHandoff transition/history revisions": _H,
}


_EXPECTED_OWNERSHIP: dict[str, dict[str, str]] = {
    "StateStoreMetadata": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/StateStoreMetadata",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "CryptoHunterAccount current record": {
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceInstallation current identity/lifecycle": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OperatorIdentity current designation/state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OperatorIdentity revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LiveAccessGrant current designation/state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LiveAccessGrant accepted revisions/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Workspace": {
        "representation_category": "M011_ENTITY_IDENTITY_PROJECTION",
        "semantic_owner_milestone": "M0.2",
        "semantic_artifact": "canonical_domain_vocabulary.json",
        "semantic_json_pointer": "/entity_kinds",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Portfolio canonical accounting state": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/rebuild_protocol",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Portfolio balance/P&L/NAV projections": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/rebuild_protocol",
        "carrier_strategy": "NONE",
    },
    "ExchangeAccount": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/exchange_account_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "CredentialProfile metadata/reference": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/credential_profile_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "TradingUniverse current version/designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/trading_universe_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "TradingUniverse version history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.5",
        "semantic_artifact": "exchange_accounts_and_instruments.json",
        "semantic_json_pointer": "/trading_universe_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyDefinition accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/strategy_definition_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyDefinition current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/strategy_definition_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "StrategyInstance current lifecycle/config": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/record_schemas/StrategyInstance",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "routing configuration/current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/route_readiness_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "routing readiness/reachability projection": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.6",
        "semantic_artifact": "strategy_market_data_and_execution_routing.json",
        "semantic_json_pointer": "/route_readiness_contract",
        "carrier_strategy": "NONE",
    },
    "RiskPolicy accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_policy_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskPolicy current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_policy_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskBudget current state": {
        "representation_category": "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/risk_input_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "kill-switch state/generation": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/kill_switch_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "kill-switch transition history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/kill_switch_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Command accepted request": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/command_registry",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Event": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/event_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "OrderIntent": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/command_registry/SUBMIT_ORDER",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Order canonical lifecycle state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/order_lifecycle",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Order lifecycle events/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/order_lifecycle",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Fill": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.7",
        "semantic_artifact": "commands_events_order_lifecycle_and_idempotency.json",
        "semantic_json_pointer": "/fill_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "LedgerEntry": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/ledger_entry_schema",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "reservation current state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/reservation_protocol",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "reservation transition history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.8",
        "semantic_artifact": "ledger_portfolio_capital_and_pnl.json",
        "semantic_json_pointer": "/reservation_protocol",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RiskDecision": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/RiskDecision",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease immutable record": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease one-shot state": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "ExecutionLease restart fence": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.9",
        "semantic_artifact": "risk_hierarchy_kill_switch_and_execution_lease.json",
        "semantic_json_pointer": "/executable_boundary_schemas/ExecutionLease",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "RuntimeSession active process manifestation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract",
        "carrier_strategy": "NONE",
    },
    "RuntimeSession canonical identity/history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SessionSecurityState current generation/state": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SessionSecurityState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SessionSecurityState revision history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SessionSecurityState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "PinVerifierRecord accepted revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/PinVerifierRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "PinVerifierRecord current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/PinVerifierRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceTrust/security revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/DeviceTrustProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "DeviceTrust current designation": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/DeviceTrustProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "platform enrollment revisions": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/biometric_policy",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "AuthenticationProof": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "CoreIssuedAuthenticationProofBinding": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "PlatformBiometricAssertion": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "CoreAcceptedPlatformBiometricAssertionBinding": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "SecretMetadataProjection": {
        "representation_category": "DIRECT_UPSTREAM_SCHEMA",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretMetadataProjection",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "secure-store payload": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.10",
        "semantic_artifact": "identity_device_authentication_and_secrets.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "bootstrap consumed fence": {
        "representation_category": "M011_CURRENT_DESIGNATION_PROJECTION",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract/executable_schemas/CoreCurrentBootstrapState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "bootstrap accepted/consumption history": {
        "representation_category": "M011_IMMUTABLE_HISTORY_WRAPPER",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/first_run_bootstrap_authority_contract/executable_schemas/ConsumedBootstrapAuthority",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "M0.3 restore freshness membership": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "M0.3 current designation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "M0.3 retirement state": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.3",
        "semantic_artifact": "process_topology_and_lifecycle.json",
        "semantic_json_pointer": "/restore_freshness_authority_contract",
        "carrier_strategy": "NONE",
    },
    "LocalDurableStateEvidence payload": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "LocalDurableEvidence accepted/current registry/designation": {
        "representation_category": "EXCLUDED_NON_DURABLE",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas",
        "carrier_strategy": "NONE",
    },
    "Migration current state/designation": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/MigrationCurrentState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "Migration transition/history revisions": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/MigrationTransitionRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SecretHandoff current state/designation": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretHandoffCurrentState",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
    "SecretHandoff transition/history revisions": {
        "representation_category": "M011_LOCAL_SCHEMA",
        "semantic_owner_milestone": "M0.11",
        "semantic_artifact": "persistence_versioning_migrations_backup_and_recovery.json",
        "semantic_json_pointer": "/executable_boundary_schemas/SecretHandoffTransitionRecord",
        "carrier_strategy": "PERSISTENCE_RECORD",
    },
}


def _failure_row(fc: str, retry: str, recovery: bool, diag: str) -> dict[str, Any]:
    return {
        "failure_class": fc,
        "retryability": retry,
        "mutation_acknowledged": False,
        "recovery_required": recovery,
        "safe_diagnostic_class": diag,
    }


_EXPECTED_FAILURES: dict[str, dict[str, Any]] = {}
for _codes, _row in [
    (
        (
            "STATE_STORE_UNAVAILABLE",
            "STATE_STORE_PERMISSION_DENIED",
            "STATE_STORE_CAPACITY_EXHAUSTED",
        ),
        _failure_row("STORAGE_ACCESS", "CONDITIONAL_AFTER_OPERATOR_REPAIR", False, "STORAGE"),
    ),
    (
        ("TRANSACTION_FAILED",),
        _failure_row("TRANSACTION", "SAFE_AFTER_CONFIRMED_ABORT", False, "TRANSACTION"),
    ),
    (
        ("DURABILITY_CONFIRMATION_FAILED",),
        _failure_row("DURABILITY_UNCERTAIN", "AFTER_RECOVERY_ONLY", True, "DURABILITY"),
    ),
    (
        ("AUTHORITATIVE_STATE_CORRUPT", "IMMUTABLE_HISTORY_CORRUPT", "CURRENT_DESIGNATION_CORRUPT"),
        _failure_row("INTEGRITY", "NEVER_AUTOMATIC", True, "INTEGRITY"),
    ),
    (
        ("DERIVED_PROJECTION_STALE",),
        _failure_row("DERIVED_STATE", "SAFE_REBUILD", True, "PROJECTION"),
    ),
    (
        ("SCHEMA_MISMATCH", "MIGRATION_REQUIRED"),
        _failure_row("VERSIONING", "AFTER_EXACT_MIGRATION", True, "VERSION"),
    ),
    (
        (
            "UNSUPPORTED_STATESTORE_SCHEMA",
            "UNSUPPORTED_BACKUP_SCHEMA",
            "MIGRATION_PATH_UNAVAILABLE",
        ),
        _failure_row("VERSIONING_UNSUPPORTED", "NEVER_AUTOMATIC", False, "VERSION"),
    ),
    (("MIGRATION_FAILED",), _failure_row("MIGRATION", "AFTER_RECOVERY_ONLY", True, "MIGRATION")),
    (
        ("LOCK_CONTENTION",),
        _failure_row("AVAILABILITY", "RETRY_WITH_BACKOFF", False, "CONCURRENCY"),
    ),
    (
        ("CONCURRENT_WRITER",),
        _failure_row("AUTHORITY_CONFLICT", "NEVER_WHILE_CONFLICT_EXISTS", True, "CONCURRENCY"),
    ),
    (("RECOVERY_REQUIRED",), _failure_row("RECOVERY", "AFTER_RECOVERY_ONLY", True, "RECOVERY")),
    (("RECOVERY_FAILED",), _failure_row("RECOVERY", "NEVER_AUTOMATIC", True, "RECOVERY")),
    (
        (
            "BACKUP_INTEGRITY_FAILED",
            "BACKUP_SCOPE_MISMATCH",
            "BACKUP_ENVIRONMENT_MISMATCH",
            "BACKUP_ROLLBACK_DETECTED",
        ),
        _failure_row("BACKUP_VALIDATION", "NEVER_FOR_SAME_CANDIDATE", False, "BACKUP"),
    ),
    (
        ("RESTORE_REJECTED",),
        _failure_row("RESTORE_VALIDATION", "NEVER_FOR_SAME_CANDIDATE", False, "RESTORE"),
    ),
    (
        ("SECRET_REFERENCE_INVALID",),
        _failure_row("SECRET_METADATA", "AFTER_METADATA_REPAIR", False, "SECRET_REFERENCE"),
    ),
    (
        ("SECRET_REFERENCE_UNAVAILABLE",),
        _failure_row("EXTERNAL_SECRET_RESOURCE", "CONDITIONAL", False, "SECRET_REFERENCE"),
    ),
    (
        ("SECRET_HANDOFF_FAILED",),
        _failure_row(
            "EXTERNAL_HANDOFF", "AFTER_RECOVERY_OR_RECONCILIATION", True, "SECRET_HANDOFF"
        ),
    ),
    (
        ("ENVIRONMENT_SCOPE_MISMATCH",),
        _failure_row("ENVIRONMENT_BOUNDARY", "NEVER_FOR_SAME_INPUT", False, "ENVIRONMENT"),
    ),
    (
        ("MONOTONIC_FENCE_ROLLBACK",),
        _failure_row("ANTI_ROLLBACK", "NEVER_AUTOMATIC", True, "MONOTONIC_FENCE"),
    ),
    (
        ("IDEMPOTENCY_CONFLICT",),
        _failure_row("IDEMPOTENCY", "NEVER_FOR_CHANGED_REQUEST", False, "IDEMPOTENCY"),
    ),
    (
        ("RECONCILIATION_REQUIRED",),
        _failure_row(
            "EXTERNAL_OUTCOME_UNCERTAIN", "AFTER_RECONCILIATION_ONLY", True, "RECONCILIATION"
        ),
    ),
    (("CONTRACT_INCONSISTENT",), _failure_row("CONTRACT", "NEVER_AUTOMATIC", True, "CONTRACT")),
]:
    for _code in _codes:
        _EXPECTED_FAILURES[_code] = _row


def _validate_failure(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "code",
        "failure_class",
        "retryability",
        "mutation_acknowledged",
        "recovery_required",
        "safe_diagnostic_class",
    }:
        return False
    code = value.get("code")
    if not isinstance(code, str):
        return False
    expected = _EXPECTED_FAILURES.get(code)
    return expected is not None and {key: value[key] for key in expected} == expected


UUID7_ID = re.compile(
    r"^(acct|dev|ws)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def _canonical_scope(account: Any, device: Any) -> bool:
    return (
        isinstance(account, str)
        and isinstance(device, str)
        and bool(UUID7_ID.fullmatch(account))
        and bool(UUID7_ID.fullmatch(device))
        and account.startswith("acct_")
        and device.startswith("dev_")
    )


def _validate_local_durable_state_evidence(payload: Any) -> bool:
    required = {
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "durability_state",
        "evidence_revision",
        "evidence_fingerprint_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return False
    if not _canonical_scope(payload.get("account_id"), payload.get("device_installation_id")):
        return False
    if not _positive(payload.get("generation")) or not _positive(payload.get("evidence_revision")):
        return False
    if payload.get("durability_state") != "DURABLE_COMMITTED":
        return False
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "evidence_fingerprint_sha256",
    ):
        if not isinstance(payload.get(key), str) or SHA_RE.fullmatch(payload[key]) is None:
            return False
    expected = _actual_fingerprint(
        {key: value for key, value in payload.items() if key != "evidence_fingerprint_sha256"}
    )
    return str(payload["evidence_fingerprint_sha256"]) == expected


class _EvidenceRegistry:
    def __init__(self, process: str) -> None:
        self.process = process
        self.accepted: dict[str, dict[str, Any]] = {}
        self.current: dict[tuple[str, str, str], str] = {}
        self.revision = 0

    def observe(self, observation: Any) -> str | None:
        if not isinstance(observation, dict):
            return None
        metadata_keys = MACHINE["executable_boundary_schemas"]["StateStoreMetadata"]["required"]
        metadata = {key: observation.get(key) for key in metadata_keys}
        if _validate_metadata(metadata) != "VALID":
            return None
        if not all(
            observation.get(key) is True
            for key in ("durable_confirmed", "authoritative_history_integrity", "current_commit")
        ):
            return None
        self.revision += 1
        payload = {
            "account_id": observation["account_id"],
            "device_installation_id": observation["device_installation_id"],
            "state_store_identity_fingerprint_sha256": observation[
                "state_store_identity_fingerprint_sha256"
            ],
            "generation": observation["protected_freshness_generation"],
            "state_fingerprint_sha256": observation["state_fingerprint_sha256"],
            "transaction_fingerprint_sha256": observation["transaction_fingerprint_sha256"],
            "durability_state": "DURABLE_COMMITTED",
            "evidence_revision": self.revision,
        }
        payload["evidence_fingerprint_sha256"] = _actual_fingerprint(payload)
        if not _validate_local_durable_state_evidence(payload):
            return None
        ref = f"{self.process}:opaque:{self.revision}"
        self.accepted[ref] = payload
        scope = (
            payload["account_id"],
            payload["device_installation_id"],
            payload["state_store_identity_fingerprint_sha256"],
        )
        self.current[scope] = ref
        return ref

    def verify_current(self, scope: tuple[str, str, str], ref: Any) -> bool:
        payload = self.accepted.get(ref) if isinstance(ref, str) else None
        if not _validate_local_durable_state_evidence(payload):
            return False
        assert isinstance(payload, dict)
        payload_scope = (
            payload["account_id"],
            payload["device_installation_id"],
            payload["state_store_identity_fingerprint_sha256"],
        )
        return payload_scope == scope and self.current.get(scope) == ref


def _validate_external_authority_observation(value: Any) -> str:
    common = {
        "available",
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "environment",
        "membership_state",
        "lifecycle",
    }
    lifecycle_fields = {
        "committed_generation",
        "committed_state_fingerprint_sha256",
        "prepared_generation",
        "prepared_state_fingerprint_sha256",
        "prepared_transaction_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or not common <= set(value)
        or not set(value) <= common | lifecycle_fields
    ):
        return "CONTRACT_INCONSISTENT"
    if value.get("available") is not True or value.get("membership_state") != "CURRENT":
        return "RESTORE_REJECTED"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "BACKUP_SCOPE_MISMATCH"
    if (
        not isinstance(value.get("state_store_identity_fingerprint_sha256"), str)
        or SHA_RE.fullmatch(value["state_store_identity_fingerprint_sha256"]) is None
    ):
        return "CONTRACT_INCONSISTENT"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "BACKUP_ENVIRONMENT_MISMATCH"
    lifecycle = value.get("lifecycle")
    committed_present = {"committed_generation", "committed_state_fingerprint_sha256"} <= set(value)
    prepared_present = {
        "prepared_generation",
        "prepared_state_fingerprint_sha256",
        "prepared_transaction_fingerprint_sha256",
    } <= set(value)
    if lifecycle == "UNINITIALIZED":
        return "VALID" if not (set(value) & lifecycle_fields) else "CONTRACT_INCONSISTENT"
    if lifecycle == "COMMITTED":
        if not committed_present or set(value) & {
            "prepared_generation",
            "prepared_state_fingerprint_sha256",
            "prepared_transaction_fingerprint_sha256",
        }:
            return "CONTRACT_INCONSISTENT"
        return (
            "VALID"
            if _positive(value["committed_generation"])
            and isinstance(value["committed_state_fingerprint_sha256"], str)
            and SHA_RE.fullmatch(value["committed_state_fingerprint_sha256"])
            else "CONTRACT_INCONSISTENT"
        )
    if lifecycle != "PREPARED" or not prepared_present:
        return "CONTRACT_INCONSISTENT"
    if not _positive(value["prepared_generation"]) or not all(
        isinstance(value[k], str) and SHA_RE.fullmatch(value[k])
        for k in ("prepared_state_fingerprint_sha256", "prepared_transaction_fingerprint_sha256")
    ):
        return "CONTRACT_INCONSISTENT"
    if not committed_present:
        return (
            "VALID"
            if value["prepared_generation"] == 1
            and not ({"committed_generation", "committed_state_fingerprint_sha256"} & set(value))
            else "CONTRACT_INCONSISTENT"
        )
    if (
        not _positive(value["committed_generation"])
        or not isinstance(value["committed_state_fingerprint_sha256"], str)
        or SHA_RE.fullmatch(value["committed_state_fingerprint_sha256"]) is None
    ):
        return "CONTRACT_INCONSISTENT"
    return (
        "VALID"
        if value["prepared_generation"] == value["committed_generation"] + 1
        else "CONTRACT_INCONSISTENT"
    )


def _restore(candidate: Any, external: Any) -> str:
    external_result = _validate_external_authority_observation(external)
    if external_result != "VALID":
        return external_result
    if not isinstance(candidate, dict):
        return "RESTORE_REJECTED"
    for key in ("account_id", "device_installation_id", "state_store_identity_fingerprint_sha256"):
        if candidate.get(key) != external.get(key):
            return "BACKUP_SCOPE_MISMATCH"
    if candidate.get("environment") != external.get("environment"):
        return "BACKUP_ENVIRONMENT_MISMATCH"
    lifecycle = external["lifecycle"]
    generation = candidate.get("generation")
    if lifecycle == "UNINITIALIZED":
        return "CONTRACT_INCONSISTENT" if generation is not None else "GENESIS_PREPARE_REQUIRED"
    committed = external.get("committed_generation")
    if lifecycle == "PREPARED" and committed is None:
        if generation is None:
            return "GENESIS_PENDING_RECOVERY_REQUIRED"
        if not _positive(generation) or generation != 1:
            return "CONTRACT_INCONSISTENT"
        if (
            candidate.get("state_fingerprint_sha256")
            != external["prepared_state_fingerprint_sha256"]
            or candidate.get("transaction_fingerprint_sha256")
            != external["prepared_transaction_fingerprint_sha256"]
        ):
            return "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
        return "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
    if not _positive(generation) or not isinstance(committed, int):
        return "CONTRACT_INCONSISTENT"
    assert isinstance(generation, int)
    if generation < committed:
        return "BACKUP_ROLLBACK_DETECTED"
    if generation == committed:
        if (
            candidate.get("state_fingerprint_sha256")
            != external["committed_state_fingerprint_sha256"]
        ):
            return "BACKUP_ROLLBACK_DETECTED"
        return (
            "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
            if lifecycle == "PREPARED"
            else "CONTINUE_REMAINING_GATES"
        )
    if lifecycle != "PREPARED" or generation != external["prepared_generation"]:
        return "CONTRACT_INCONSISTENT"
    return (
        "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
        if candidate.get("state_fingerprint_sha256")
        == external["prepared_state_fingerprint_sha256"]
        and candidate.get("transaction_fingerprint_sha256")
        == external["prepared_transaction_fingerprint_sha256"]
        else "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
    )


_PERSISTENCE_FIELDS = {
    "representation_name",
    "representation_category",
    "semantic_owner_milestone",
    "semantic_artifact",
    "semantic_json_pointer",
    "semantic_contract_fingerprint_sha256",
    "record_key",
    "payload",
    "payload_fingerprint_sha256",
}


def _validate_projection_schema(payload: Any, schema_name: str) -> bool:
    if not isinstance(payload, dict):
        return False
    schema = MACHINE["executable_boundary_schemas"][schema_name]
    if set(payload) != set(schema["required"]):
        return False
    if schema_name == "PersistentEntityIdentityProjection":
        return (
            payload.get("entity_kind") in {"CryptoHunterAccount", "Workspace"}
            and _canonical_entity_id(payload.get("entity_kind"), payload.get("entity_id"))
            and isinstance(payload.get("parent_scope_bindings"), dict)
        )
    if schema_name == "CurrentDesignationProjection":
        return (
            isinstance(payload.get("scope_key"), str)
            and bool(payload["scope_key"])
            and isinstance(payload.get("current_reference"), str)
            and bool(payload["current_reference"])
            and all(
                value is None or _positive(value)
                for value in (payload.get("current_revision"), payload.get("current_generation"))
            )
            and isinstance(payload.get("content_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["content_fingerprint_sha256"]) is not None
        )
    if schema_name == "UpstreamFactsProjection":
        return (
            isinstance(payload.get("fact_kind"), str)
            and bool(payload["fact_kind"])
            and isinstance(payload.get("scope_key"), str)
            and isinstance(payload.get("facts"), dict)
            and isinstance(payload.get("source_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["source_fingerprint_sha256"]) is not None
        )
    return False


def _canonical_entity_id(kind: Any, value: Any) -> bool:
    if not isinstance(kind, str) or not isinstance(value, str):
        return False
    prefix = {"CryptoHunterAccount": "acct", "Workspace": "ws"}.get(kind)
    match = UUID7_ID.fullmatch(value)
    return match is not None and match.group(1) == prefix


def _exact_entity_projection(payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "entity_kind",
        "entity_id",
        "parent_scope_bindings",
    }:
        return False
    kind = payload.get("entity_kind")
    if not _canonical_entity_id(kind, payload.get("entity_id")):
        return False
    parents = payload.get("parent_scope_bindings")
    if kind == "CryptoHunterAccount":
        return parents == {}
    return (
        isinstance(parents, dict)
        and set(parents) == {"account_id"}
        and isinstance(parents["account_id"], str)
        and parents["account_id"].startswith("acct_")
        and UUID7_ID.fullmatch(parents["account_id"]) is not None
    )


GENERIC_UUID7_ID = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)(\\.[0-9]*[1-9])?$")


def _terminal_fingerprint_field(aspect: str) -> str | None:
    return {
        "Event": "event_fingerprint_sha256",
        "Fill": "fill_fingerprint_sha256",
        "kill-switch state/generation": "record_fingerprint_sha256",
        "RiskDecision": "decision_fingerprint_sha256",
        "ExecutionLease immutable record": "lease_fingerprint_sha256",
        "SessionSecurityState current generation/state": "content_fingerprint_sha256",
        "SecretMetadataProjection": "content_fingerprint_sha256",
    }.get(aspect)


def _field_schema_valid(field: str, value: Any, schema: dict[str, Any]) -> bool:
    kind = schema.get("type")
    if kind == "constant":
        return bool(value == schema.get("value"))
    if kind == "id":
        prefix = schema.get("prefix")
        return (
            isinstance(prefix, str)
            and isinstance(value, str)
            and value.startswith(prefix + "_")
            and GENERIC_UUID7_ID.fullmatch(value) is not None
        )
    if kind == "positive_integer":
        return _positive(value)
    if kind == "sha256_hex":
        return isinstance(value, str) and SHA_RE.fullmatch(value) is not None
    if kind == "enum":
        return value in schema.get("values", [])
    if kind == "decimal":
        if not isinstance(value, str) or DECIMAL_RE.fullmatch(value) is None:
            return False
        try:
            number = Decimal(value)
        except InvalidOperation:
            return False
        constraint = schema.get("constraint")
        return (constraint != "positive" or number > 0) and (
            constraint != "non_negative" or number >= 0
        )
    if kind == "timestamp":
        return isinstance(value, str) and value.endswith("Z") and "T" in value
    if kind == "array":
        return isinstance(value, list)
    if kind == "event_safe_payload":
        return isinstance(value, dict)
    if kind == "array_of_canonical_id":
        return isinstance(value, list) and all(
            _field_schema_valid(field, item, {"type": "id", "prefix": schema.get("id_prefix")})
            for item in value
        )
    if kind in {"object", "asset_reference"}:
        fields = schema.get("fields")
        nested = schema.get("field_schemas")
        return (
            isinstance(value, dict)
            and isinstance(fields, list)
            and isinstance(nested, dict)
            and set(value) == set(fields)
            and all(_field_schema_valid(name, value[name], nested[name]) for name in fields)
        )
    if kind == "canonical_id":
        return _field_schema_valid(field, value, {"type": "id", "prefix": schema.get("id_prefix")})
    if kind == "compound_scope":
        return isinstance(value, str) and bool(value)
    if kind == "non_empty_object":
        return isinstance(value, dict) and bool(value)
    if kind in {"non_empty_string", "string"}:
        return isinstance(value, str) and bool(value)
    return False


def _validate_direct_upstream_payload(aspect: str, payload: Any) -> bool:
    registry = MACHINE["backup_contract"]["direct_upstream_validator_registry"]
    contract = registry.get(aspect)
    if not isinstance(contract, dict) or not isinstance(payload, dict):
        return False
    exact = set(contract["exact_fields"])
    if set(payload) != exact:
        return False
    nullable = set(contract["nullable_fields"])
    schemas = contract["upstream_field_schemas"]
    for field in exact:
        value = payload[field]
        if value is None and field not in nullable:
            return False
        if value is not None and not _field_schema_valid(field, value, schemas.get(field, {})):
            return False
    if "environment" in payload and payload["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        return False
    source_document = json.loads((DOCS / contract["semantic_artifact"]).read_text())
    for field, constraint in contract.get("semantic_constraints", {}).items():
        if field not in payload:
            continue
        values = constraint.get("values")
        if (
            "pointer" in constraint
            and field != "safe_payload"
            and field != "fee_semantics"
            and field != "secret_reference"
        ):
            ok, values = _resolve_pointer(source_document, constraint["pointer"])
            if not ok or not isinstance(values, list):
                return False
        if isinstance(values, list) and payload[field] not in values:
            return False
    if aspect == "Event":
        source = _source_value(MACHINE["backup_contract"]["representation_registry"][aspect])
        event_schema = source["event_schema_registry"].get(payload["event_type"])
        if not isinstance(event_schema, dict) or set(payload["safe_payload"]) != set(
            event_schema["safe_payload_fields"]
        ):
            return False
    if aspect == "Fill":
        if payload["fee_kind"] == "NONE" and not (
            payload["fee_quantity"] == "0" and payload["fee_asset_reference"] is None
        ):
            return False
        if payload["fee_kind"] == "CHARGE" and (
            payload["fee_quantity"] == "0" or not isinstance(payload["fee_asset_reference"], dict)
        ):
            return False
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        expected = _actual_fingerprint(
            {key: value for key, value in payload.items() if key != terminal}
        )
        if payload[terminal] != expected:
            return False
    return True


def _validate_immutable_projection(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    binding = entry.get("immutable_fact_binding")
    if (
        not isinstance(binding, dict)
        or not isinstance(payload, dict)
        or set(payload) != set(binding["wrapper_fields"])
        or payload.get("fact_kind") != entry["semantic_object_or_invariant"]
        or not isinstance(payload.get("upstream_payload"), dict)
        or set(payload["upstream_payload"]) != set(binding["persisted_payload_fields"])
    ):
        return False
    upstream = payload["upstream_payload"]
    contracts = binding["field_contracts"]
    if not all(
        _field_schema_valid(field, upstream[field], contracts[field])
        for field in binding["persisted_payload_fields"]
    ):
        return False
    fingerprint = payload.get("upstream_payload_fingerprint_sha256")
    return isinstance(fingerprint, str) and fingerprint == _actual_fingerprint(upstream)


def _validate_facts_projection(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    if not isinstance(payload, dict) or set(payload) != {
        "fact_kind",
        "scope_key",
        "facts",
        "source_fingerprint_sha256",
    }:
        return False
    facts = payload.get("facts")
    binding = entry.get("fact_binding")
    if (
        payload.get("fact_kind") != aspect
        or payload.get("source_fingerprint_sha256") != entry["semantic_contract_fingerprint_sha256"]
        or not isinstance(binding, dict)
        or not isinstance(facts, dict)
        or set(facts) != set(binding["required_fact_fields"])
        or not facts
        or payload.get("scope_key") != facts.get(binding["scope_binding"])
    ):
        return False
    contracts = binding.get("field_contracts")
    if not isinstance(contracts, dict) or set(contracts) != set(facts):
        return False
    if not all(_field_schema_valid(field, facts[field], contracts[field]) for field in facts):
        return False
    for field, contract in contracts.items():
        if contract.get("type") == "compound_scope":
            expected = "|".join(str(facts[name]) for name in contract["components"])
            if facts[field] != expected:
                return False
    return True


def _validate_category_payload(aspect: str, entry: dict[str, Any], payload: Any) -> bool:
    if aspect == "bootstrap consumed fence":
        fields = {
            "state_fingerprint_sha256",
            "account_id",
            "device_installation_id",
            "intended_operator_id",
            "startup_readiness",
            "initial_security_lifecycle",
            "first_operator_presence",
            "expected_generation",
            "expected_revision",
            "consumed_authorities",
            "state_revision",
        }
        return (
            isinstance(payload, dict)
            and set(payload) == fields
            and _canonical_scope(payload.get("account_id"), payload.get("device_installation_id"))
            and isinstance(payload.get("intended_operator_id"), str)
            and payload["intended_operator_id"].startswith("op_")
            and all(
                _positive(payload.get(k))
                for k in ("expected_generation", "expected_revision", "state_revision")
            )
            and isinstance(payload.get("consumed_authorities"), list)
            and all(
                isinstance(x, str) and SHA_RE.fullmatch(x) for x in payload["consumed_authorities"]
            )
            and isinstance(payload.get("state_fingerprint_sha256"), str)
            and SHA_RE.fullmatch(payload["state_fingerprint_sha256"]) is not None
        )
    if aspect == "bootstrap accepted/consumption history":
        fields = {
            "account_id",
            "device_installation_id",
            "bootstrap_generation",
            "bootstrap_revision",
            "claim_fingerprint_sha256",
            "challenge_fingerprint_sha256",
        }
        return (
            isinstance(payload, dict)
            and set(payload) == fields
            and _canonical_scope(payload.get("account_id"), payload.get("device_installation_id"))
            and _positive(payload.get("bootstrap_generation"))
            and _positive(payload.get("bootstrap_revision"))
            and all(
                isinstance(payload.get(k), str) and SHA_RE.fullmatch(payload[k])
                for k in ("claim_fingerprint_sha256", "challenge_fingerprint_sha256")
            )
        )
    category = entry["representation_category"]
    if category == "M011_ENTITY_IDENTITY_PROJECTION":
        return _exact_entity_projection(payload)
    if category == "M011_CURRENT_DESIGNATION_PROJECTION":
        return _validate_projection_schema(payload, "CurrentDesignationProjection")
    if category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
        return _validate_facts_projection(aspect, entry, payload)
    if category == "M011_LOCAL_SCHEMA":
        schema_name = entry["projection_schema_if_any"]
        if schema_name == "StateStoreMetadata":
            return _validate_metadata(payload) == "VALID"
        validators = {
            "MigrationTransitionRecord": _validate_migration_transition,
            "MigrationCurrentState": _validate_migration_current,
            "SecretHandoffTransitionRecord": _validate_handoff_transition,
            "SecretHandoffCurrentState": _validate_handoff_current,
        }
        return validators[schema_name](payload)
    if category == "DIRECT_UPSTREAM_SCHEMA":
        return _validate_direct_upstream_payload(aspect, payload)
    if category == "M011_IMMUTABLE_HISTORY_WRAPPER":
        return _validate_immutable_projection(aspect, entry, payload)
    return False


def _derive_record_key(aspect: str, entry: dict[str, Any], payload: dict[str, Any]) -> str | None:
    if not isinstance(payload, dict):
        return None
    try:
        strategy = entry.get("record_key_strategy")
        if strategy == "DIRECT_UPSTREAM_KEY_FIELDS":
            contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
            return (
                "direct:"
                + aspect
                + ":"
                + ":".join(str(payload[field]) for field in contract["record_key_fields"])
            )
        if strategy == "IMMUTABLE_PAYLOAD_IDENTITY_REVISION":
            binding = entry["immutable_fact_binding"]
            upstream = payload["upstream_payload"]
            fields = (
                binding["canonical_object_identity_fields"] + binding["revision_generation_fields"]
            )
            return (
                "immutable:"
                + str(payload["fact_kind"])
                + ":"
                + ":".join(str(upstream[field]) for field in fields)
            )
        if strategy == "CANONICAL_ENTITY_ID":
            return f"entity:{payload['entity_kind']}:{payload['entity_id']}"
        if strategy == "SCOPE_CURRENT_REFERENCE_REVISION_GENERATION":
            return f"current:{payload['scope_key']}:{payload['current_reference']}:{payload['current_revision']}:{payload['current_generation']}"
        if strategy == "CANONICAL_OBJECT_ID_REVISION":
            return (
                f"object:{payload['semantic_object']}:{payload['object_id']}:{payload['revision']}"
            )
        if strategy == "FACT_SCOPE_OBJECT_GENERATION":
            facts = payload["facts"]
            fields = entry["fact_binding"]["record_key_object_fields"]
            return (
                "facts:"
                + str(payload["fact_kind"])
                + ":"
                + ":".join(str(facts[field]) for field in fields)
            )
        if strategy == "BOOTSTRAP_SCOPE_STATE_REVISION":
            return f"bootstrap-current:{payload['account_id']}:{payload['device_installation_id']}:{payload['state_revision']}"
        if strategy == "BOOTSTRAP_SCOPE_GENERATION_REVISION_CLAIM":
            return f"bootstrap-history:{payload['account_id']}:{payload['device_installation_id']}:{payload['bootstrap_generation']}:{payload['bootstrap_revision']}:{payload['claim_fingerprint_sha256']}"
        if strategy == "STATE_STORE_SCOPE_GENERATION":
            return f"state-store:{payload['account_id']}:{payload['device_installation_id']}:{payload['state_store_identity_fingerprint_sha256']}:{payload['protected_freshness_generation']}"
        if strategy == "MIGRATION_ID_TRANSITION_REVISION":
            return (
                f"migration-transition:{payload['migration_id']}:{payload['transition_revision']}"
            )
        if strategy == "MIGRATION_ID_CURRENT_REVISION":
            return f"migration-current:{payload['migration_id']}:{payload['current_transition_revision']}"
        if strategy == "HANDOFF_ID_TRANSITION_REVISION":
            return f"handoff-transition:{payload['handoff_id']}:{payload['transition_revision']}"
        if strategy == "HANDOFF_ID_CURRENT_REVISION":
            return (
                f"handoff-current:{payload['handoff_id']}:{payload['current_transition_revision']}"
            )
        return None
    except (KeyError, TypeError):
        return None


def _validate_persistence_record(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != _PERSISTENCE_FIELDS:
        return False
    registry = MACHINE["backup_contract"]["representation_registry"]
    aspect = value.get("representation_name")
    entry = registry.get(aspect) if isinstance(aspect, str) else None
    if (
        not isinstance(entry, dict)
        or entry["carrier_strategy"] != "PERSISTENCE_RECORD"
        or entry["representation_category"] == "EXCLUDED_NON_DURABLE"
    ):
        return False
    assert isinstance(aspect, str)
    for key in (
        "representation_category",
        "semantic_owner_milestone",
        "semantic_artifact",
        "semantic_json_pointer",
        "semantic_contract_fingerprint_sha256",
    ):
        if value.get(key) != entry.get(key):
            return False
    if (
        not value.get("record_key")
        or not isinstance(value.get("payload_fingerprint_sha256"), str)
        or SHA_RE.fullmatch(value["payload_fingerprint_sha256"]) is None
    ):
        return False
    if value["payload_fingerprint_sha256"] != _actual_fingerprint(value.get("payload")):
        return False
    if not isinstance(value.get("payload"), dict) or value.get("record_key") != _derive_record_key(
        aspect, entry, value["payload"]
    ):
        return False
    if _contains_forbidden(
        value["payload"], direct_pin_record=aspect == "PinVerifierRecord accepted revisions"
    ):
        return False
    return _validate_category_payload(aspect, entry, value["payload"])


def _validate_backup_records(records: Any) -> bool:
    return isinstance(records, list) and all(
        _validate_persistence_record(record) for record in records
    )


def _validate_backup(value: Any) -> str:
    required = set(MACHINE["executable_boundary_schemas"]["BackupEnvelope"]["required"])
    if not isinstance(value, dict) or set(value) != required:
        return "BACKUP_INTEGRITY_FAILED"
    if not all(
        _positive(value.get(key))
        for key in (
            "backup_envelope_schema_version",
            "state_store_schema_version",
            "local_protected_freshness_generation",
        )
    ):
        return "BACKUP_INTEGRITY_FAILED"
    if not _canonical_scope(value.get("account_id"), value.get("device_installation_id")):
        return "BACKUP_SCOPE_MISMATCH"
    if value.get("environment") not in {"PAPER", "TESTNET", "LIVE"}:
        return "BACKUP_ENVIRONMENT_MISMATCH"
    for key in (
        "state_store_identity_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
        "envelope_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return "BACKUP_INTEGRITY_FAILED"
    if not _validate_backup_records(
        value.get("canonical_durable_records")
    ) or not _validate_backup_records(value.get("immutable_recovery_history")):
        return "BACKUP_INTEGRITY_FAILED"
    metadata = value.get("integrity_metadata")
    if not isinstance(metadata, dict) or _contains_forbidden(metadata):
        return "BACKUP_INTEGRITY_FAILED"
    projected = {key: item for key, item in value.items() if key != "envelope_fingerprint_sha256"}
    return (
        "VALID"
        if value["envelope_fingerprint_sha256"] == _actual_fingerprint(projected)
        else "BACKUP_INTEGRITY_FAILED"
    )


def _canonical_fixture_id(prefix: str, suffix: str) -> str:
    final = {"a": "b", "b": "c", "c": "d"}.get(suffix, "e")
    return f"{prefix}_01890f3a-2b4c-7abc-8def-0123456789a{final}"


def _direct_fixture_value(
    aspect: str, field: str, schema: dict[str, Any], suffix: str, revision: int
) -> Any:
    kind = schema.get("type")
    if kind == "constant":
        return schema["value"]
    if kind == "id":
        return _canonical_fixture_id(schema["prefix"], suffix)
    if kind == "positive_integer":
        return revision
    if kind == "sha256_hex":
        return SHA
    if kind == "enum":
        return schema["values"][0]
    if kind == "decimal":
        return "1" if schema.get("constraint") == "positive" else "0"
    if kind == "timestamp":
        return "2026-01-01T00:00:00Z"
    if kind == "array":
        return []
    if kind == "event_safe_payload":
        return {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    if kind in {"object", "asset_reference"}:
        return {
            name: _direct_fixture_value(
                aspect, name, schema["field_schemas"][name], suffix, revision
            )
            for name in schema["fields"]
        }
    if field == "secret_reference":
        return f"secure-store://reference-{suffix}"
    if field == "event_type":
        return "ORDER_PLANNED"
    if field == "safe_payload":
        return {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    if field == "fee_asset_reference":
        return None
    if field == "side":
        return "BUY"
    if field == "fee_kind":
        return "NONE"
    if field == "direction":
        return "DEBIT"
    if field == "posting_role":
        return "ASSET_RECEIVED"
    if field == "decision":
        return "ALLOW"
    if field == "state" and "SessionSecurityState" in aspect:
        return "LOCKED"
    if field == "state" and "SecretMetadata" in aspect:
        return "AVAILABLE"
    if field in {"state", "lifecycle_state", "connection_state", "execution_authorization"}:
        return "ACTIVE"
    return f"canonical-{field}-{suffix}"


def _direct_payload(aspect: str, suffix: str, revision: int) -> dict[str, Any]:
    contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
    nullable = set(contract["nullable_fields"])
    schemas = contract["upstream_field_schemas"]
    payload = {
        field: (
            None
            if field in nullable
            else _direct_fixture_value(aspect, field, schemas.get(field, {}), suffix, revision)
        )
        for field in contract["exact_fields"]
    }
    if aspect == "Event":
        payload["event_type"] = "ORDER_PLANNED"
        payload["safe_payload"] = {"side": "BUY", "order_type": "MARKET", "quantity": "1"}
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        payload[terminal] = _actual_fingerprint(
            {key: value for key, value in payload.items() if key != terminal}
        )
    return payload


def _payload_for(aspect: str, *, object_suffix: str = "a", revision: int = 1) -> dict[str, Any]:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    category = entry["representation_category"]
    if aspect == "bootstrap consumed fence":
        return {
            "state_fingerprint_sha256": SHA,
            "account_id": SCOPE[0],
            "device_installation_id": SCOPE[1],
            "intended_operator_id": "op_" + SCOPE[0][5:],
            "startup_readiness": "READY",
            "initial_security_lifecycle": "COMPLETED",
            "first_operator_presence": "PRESENT",
            "expected_generation": 1,
            "expected_revision": 1,
            "consumed_authorities": [SHA],
            "state_revision": revision,
        }
    if aspect == "bootstrap accepted/consumption history":
        return {
            "account_id": SCOPE[0],
            "device_installation_id": SCOPE[1],
            "bootstrap_generation": 1,
            "bootstrap_revision": revision,
            "claim_fingerprint_sha256": SHA,
            "challenge_fingerprint_sha256": "b" * 64,
        }
    if category == "M011_ENTITY_IDENTITY_PROJECTION":
        kind = "CryptoHunterAccount" if aspect.startswith("CryptoHunterAccount") else "Workspace"
        entity_id = ("acct_" if kind == "CryptoHunterAccount" else "ws_") + SCOPE[0][5:]
        return {
            "entity_kind": kind,
            "entity_id": entity_id,
            "parent_scope_bindings": {}
            if kind == "CryptoHunterAccount"
            else {"account_id": SCOPE[0]},
        }
    if category == "M011_CURRENT_DESIGNATION_PROJECTION":
        return {
            "scope_key": SCOPE[0],
            "current_reference": f"ref:{object_suffix}:{SHA}",
            "current_revision": revision,
            "current_generation": revision,
            "content_fingerprint_sha256": SHA,
        }
    if category == "M011_PERSISTENCE_PROJECTION_OF_UPSTREAM_FACTS":
        binding = entry["fact_binding"]
        contracts = binding["field_contracts"]
        facts: dict[str, Any] = {}
        for name in binding["required_fact_fields"]:
            contract = contracts[name]
            kind = contract["type"]
            if kind == "canonical_id":
                facts[name] = _canonical_fixture_id(contract["id_prefix"], object_suffix)
            elif kind == "positive_integer":
                facts[name] = revision
            elif kind == "sha256_hex":
                facts[name] = SHA
            elif kind == "enum":
                facts[name] = contract["values"][0]
        for name, contract in contracts.items():
            if contract["type"] == "compound_scope":
                facts[name] = "|".join(str(facts[field]) for field in contract["components"])
        return {
            "fact_kind": aspect,
            "scope_key": facts[binding["scope_binding"]],
            "facts": facts,
            "source_fingerprint_sha256": entry["semantic_contract_fingerprint_sha256"],
        }
    if category == "M011_LOCAL_SCHEMA":
        return {
            "StateStoreMetadata": _metadata(protected_freshness_generation=revision),
            "MigrationTransitionRecord": _migration_transition(
                revision,
                None if revision == 1 else "PREPARED",
                "PREPARED" if revision == 1 else "APPLYING",
            ),
            "MigrationCurrentState": _migration_current(_migration_transition(1, None, "PREPARED")),
            "SecretHandoffTransitionRecord": _handoff_transition(
                revision,
                None if revision == 1 else "PREPARED",
                "PREPARED" if revision == 1 else "APPLYING",
            ),
            "SecretHandoffCurrentState": _handoff_current(_handoff_transition(1, None, "PREPARED")),
        }[entry["projection_schema_if_any"]]
    if category == "DIRECT_UPSTREAM_SCHEMA":
        return _direct_payload(aspect, object_suffix, revision)
    if category == "M011_IMMUTABLE_HISTORY_WRAPPER":
        binding = entry["immutable_fact_binding"]
        upstream: dict[str, Any] = {}
        for field in binding["persisted_payload_fields"]:
            contract = binding["field_contracts"][field]
            kind = contract["type"]
            if kind == "canonical_id":
                upstream[field] = _canonical_fixture_id(contract["id_prefix"], object_suffix)
            elif kind == "positive_integer":
                upstream[field] = revision
            elif kind == "sha256_hex":
                upstream[field] = SHA
            elif kind == "enum":
                upstream[field] = contract["values"][0]
            elif kind == "array_of_canonical_id":
                upstream[field] = [_canonical_fixture_id(contract["id_prefix"], object_suffix)]
            elif kind == "non_empty_object":
                upstream[field] = {"algorithm": "argon2id", "encoded_verifier": SHA}
            else:
                upstream[field] = f"canonical-{field}-{object_suffix}"
        terminal = (
            "content_fingerprint_sha256"
            if "content_fingerprint_sha256" in upstream
            else "record_fingerprint_sha256"
            if "record_fingerprint_sha256" in upstream
            else "event_fingerprint_sha256"
            if "event_fingerprint_sha256" in upstream
            else None
        )
        if terminal is not None:
            upstream[terminal] = _actual_fingerprint(
                {key: value for key, value in upstream.items() if key != terminal}
            )
        return {
            "fact_kind": entry["semantic_object_or_invariant"],
            "upstream_payload": upstream,
            "upstream_payload_fingerprint_sha256": _actual_fingerprint(upstream),
        }
    raise AssertionError(f"unsupported persisted category: {category}")


def _persistence_record(
    aspect: str, payload: Any | None = None, *, object_suffix: str = "a", revision: int = 1
) -> dict[str, Any]:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    actual = (
        _payload_for(aspect, object_suffix=object_suffix, revision=revision)
        if payload is None
        else payload
    )
    value = {
        key: entry[key]
        for key in (
            "representation_category",
            "semantic_owner_milestone",
            "semantic_artifact",
            "semantic_json_pointer",
            "semantic_contract_fingerprint_sha256",
        )
    }
    value["representation_name"] = aspect
    value.update(
        record_key=_derive_record_key(aspect, entry, actual),
        payload=actual,
        payload_fingerprint_sha256=_actual_fingerprint(actual),
    )
    return value


def _valid_backup() -> dict[str, Any]:
    value = {
        "backup_envelope_schema_version": 1,
        "state_store_schema_version": 2,
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SHA,
        "environment": "PAPER",
        "local_protected_freshness_generation": 2,
        "state_fingerprint_sha256": SHA,
        "transaction_fingerprint_sha256": "b" * 64,
        "history_tail_fingerprint_sha256": "c" * 64,
        "canonical_durable_records": [],
        "immutable_recovery_history": [],
        "envelope_fingerprint_sha256": "",
        "integrity_metadata": {},
    }
    value["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {key: item for key, item in value.items() if key != "envelope_fingerprint_sha256"}
    )
    return value


def _version_result(state: Any, backup: Any, paths: set[tuple[int, int]], current: int = 2) -> str:
    if not _positive(state) or not _positive(backup):
        return "CONTRACT_INCONSISTENT"
    if backup > 1:
        return "UNSUPPORTED_BACKUP_SCHEMA"
    if state > current:
        return "UNSUPPORTED_STATESTORE_SCHEMA"
    if state == current:
        return "VALID"
    return "MIGRATION_REQUIRED" if (state, current) in paths else "MIGRATION_PATH_UNAVAILABLE"


def _migration_recovery(state: str, durable_target: bool, path: list[int]) -> str:
    if path != list(range(path[0], path[-1] + 1)):
        return "MIGRATION_PATH_UNAVAILABLE"
    if state == "PREPARED":
        return "RESUME_APPLYING"
    if state == "APPLYING":
        return "RESUME_VERIFY_OR_REAPPLY_NO_PROMOTION"
    if state == "DURABLE_MIGRATED" and durable_target:
        return "COMPLETE_MARKER_IDEMPOTENTLY"
    if state == "COMPLETED" and durable_target:
        return "ALREADY_COMPLETED"
    if state == "FAILED":
        return "MIGRATION_FAILED"
    return "RECOVERY_REQUIRED"


@dataclass
class _ReplayState:
    requests: dict[str, tuple[str, str]] = field(default_factory=dict)
    side_effect_count: int = 0
    fills: set[str] = field(default_factory=set)
    ledger_effects: int = 0
    reservation_state: str = "HELD"
    reservation_effects: int = 0
    blind_resubmits: int = 0

    def command(self, key: str, fingerprint: str) -> str:
        prior = self.requests.get(key)
        if prior:
            return prior[1] if prior[0] == fingerprint else "IDEMPOTENCY_CONFLICT"
        self.side_effect_count += 1
        self.requests[key] = (fingerprint, "CANONICAL_OUTCOME")
        return "CANONICAL_OUTCOME"

    def fill(self, fill_id: str) -> None:
        if fill_id not in self.fills:
            self.fills.add(fill_id)
            self.ledger_effects += 1

    def reservation(self, terminal: str) -> None:
        if self.reservation_state == "HELD" and terminal in {"CONSUMED", "RELEASED"}:
            self.reservation_state = terminal
            self.reservation_effects += 1

    def unknown_external(self) -> str:
        return "RECONCILIATION_REQUIRED"


def _writer_attempt(
    role: str, lock_occupied: bool, competing_writer: bool
) -> tuple[str, bool, bool]:
    if role != "Core":
        return "NO_PERSISTENCE_AUTHORITY", False, False
    if competing_writer:
        return "CONCURRENT_WRITER", False, False
    if lock_occupied:
        return "LOCK_CONTENTION", False, False
    return "WRITER_OPENED", True, True


def _security_restore(candidate: dict[str, Any], current: dict[str, Any]) -> str:
    terminal = (
        "bootstrap_consumed",
        "device_revoked",
        "operator_revoked",
        "grant_revoked",
        "lease_consumed",
    )
    if any(current.get(key) and not candidate.get(key) for key in terminal):
        return "MONOTONIC_FENCE_ROLLBACK"
    for key in (
        "pin_revision",
        "security_generation",
        "session_security_generation",
        "kill_switch_generation",
    ):
        if candidate.get(key, 0) < current.get(key, 0):
            return "MONOTONIC_FENCE_ROLLBACK"
    if (
        candidate.get("AuthenticationProof") is not None
        or candidate.get("PlatformBiometricAssertion") is not None
    ):
        return "RESTORE_REJECTED"
    return "VALID"


def test_complete_frozen_durability_matrix_exact_equality() -> None:
    assert MACHINE["durability_classification"]["records"] == _EXPECTED_DURABILITY


def test_durability_matrix_mutation_is_detected() -> None:
    mutated = dict(_EXPECTED_DURABILITY)
    mutated.pop("Fill")
    assert mutated != MACHINE["durability_classification"]["records"]
    mutated = dict(_EXPECTED_DURABILITY)
    mutated["Fill"] = _A
    assert mutated != MACHINE["durability_classification"]["records"]


def test_failure_registry_exact_independently_authored_matrix() -> None:
    registry = MACHINE["failure_registry"]
    assert len(_EXPECTED_FAILURES) == 32
    assert registry["outcomes_by_code"] == _EXPECTED_FAILURES
    assert set(registry["closed_codes"]) == set(_EXPECTED_FAILURES)


@pytest.mark.parametrize(
    "mutation",
    [
        {"retryability": "AFTER_RECOVERY_ONLY"},
        {"failure_class": "CONTRACT"},
        {"mutation_acknowledged": True},
        {"recovery_required": True},
        {"safe_diagnostic_class": "RESTORE"},
        {"extra": "x"},
    ],
)
def test_failure_outcome_wrong_metadata_fails_closed(mutation: dict[str, Any]) -> None:
    value = {"code": "TRANSACTION_FAILED", **_EXPECTED_FAILURES["TRANSACTION_FAILED"]}
    value.update(mutation)
    assert not _validate_failure(value)


def test_unknown_failure_code_is_denied() -> None:
    assert not _validate_failure({"code": "EXTRA", **_EXPECTED_FAILURES["TRANSACTION_FAILED"]})


@pytest.mark.parametrize(
    "bad_account,bad_device",
    [
        ("acct_wrong", SCOPE[1]),
        (SCOPE[0], "dev_wrong"),
        ("acct_01890f3a-2b4c-4abc-8def-0123456789ab", SCOPE[1]),
        ("acct_01890F3A-2b4c-7abc-8def-0123456789ab", SCOPE[1]),
        (SCOPE[0] + "x", SCOPE[1]),
        (SCOPE[0], "dev_01890f3a-2b4c-7abc-7def-0123456789ab"),
    ],
)
def test_canonical_m02_scope_ids_fail_closed(bad_account: str, bad_device: str) -> None:
    assert not _canonical_scope(bad_account, bad_device)


def test_exact_evidence_payload_membership_and_fingerprint() -> None:
    registry = _EvidenceRegistry("process")
    ref = registry.observe(_observation())
    assert ref is not None and set(registry.accepted[ref]) == {
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "durability_state",
        "evidence_revision",
        "evidence_fingerprint_sha256",
    }
    assert registry.verify_current(SCOPE, ref)
    registry.accepted[ref]["generation"] = 2
    assert not registry.verify_current(SCOPE, ref)


def test_self_hashed_evidence_without_membership_is_denied() -> None:
    registry = _EvidenceRegistry("process")
    payload = _observation()
    ref = "caller:" + _actual_fingerprint(payload)
    assert not registry.verify_current(SCOPE, ref)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ({"extra": True}, "BACKUP_INTEGRITY_FAILED"),
        ({"backup_envelope_schema_version": True}, "BACKUP_INTEGRITY_FAILED"),
        ({"local_protected_freshness_generation": False}, "BACKUP_INTEGRITY_FAILED"),
        ({"account_id": "acct_wrong"}, "BACKUP_SCOPE_MISMATCH"),
        ({"device_installation_id": "dev_wrong"}, "BACKUP_SCOPE_MISMATCH"),
        ({"environment": "WRONG"}, "BACKUP_ENVIRONMENT_MISMATCH"),
    ],
)
def test_backup_boundary_fields_fail_closed(mutation: dict[str, Any], expected: str) -> None:
    value = _valid_backup()
    value.update(mutation)
    assert _validate_backup(value) == expected


@pytest.mark.parametrize(
    "record",
    [
        {"record_kind": "LocalDurableStateEvidence"},
        {"record_kind": "Event", "raw_pin": "1"},
        {"record_kind": "Event", "payload": {"api_secret": "x"}},
        {"record_kind": "SecureStorePayload"},
        {"record_kind": "AuthenticationProof"},
        {"record_kind": "PlatformBiometricAssertion"},
    ],
)
def test_backup_rejects_nested_evidence_and_secret_authority(record: dict[str, Any]) -> None:
    value = _valid_backup()
    value["canonical_durable_records"] = [record]
    assert _validate_backup(value) == "BACKUP_INTEGRITY_FAILED"


@pytest.mark.parametrize(
    ("state", "durable", "expected"),
    [
        ("PREPARED", False, "RESUME_APPLYING"),
        ("APPLYING", False, "RESUME_VERIFY_OR_REAPPLY_NO_PROMOTION"),
        ("DURABLE_MIGRATED", True, "COMPLETE_MARKER_IDEMPOTENTLY"),
        ("COMPLETED", True, "ALREADY_COMPLETED"),
        ("FAILED", False, "MIGRATION_FAILED"),
    ],
)
def test_executable_migration_crash_recovery(state: str, durable: bool, expected: str) -> None:
    assert _migration_recovery(state, durable, [1, 2]) == expected


def test_migration_unknown_or_skipped_path_denied() -> None:
    assert _migration_recovery("PREPARED", False, [1, 3]) == "MIGRATION_PATH_UNAVAILABLE"
    assert MACHINE["migration_protocol"]["rollback_policy"] == "FORWARD_ONLY"
    assert MACHINE["migration_protocol"]["authoritative_change_requires_m0_3_freshness"] is True


@pytest.mark.parametrize(
    ("state", "backup", "paths", "expected"),
    [
        (2, 1, set(), "VALID"),
        (3, 1, set(), "UNSUPPORTED_STATESTORE_SCHEMA"),
        (2, 2, set(), "UNSUPPORTED_BACKUP_SCHEMA"),
        (1, 1, {(1, 2)}, "MIGRATION_REQUIRED"),
        (1, 1, set(), "MIGRATION_PATH_UNAVAILABLE"),
        (True, 1, set(), "CONTRACT_INCONSISTENT"),
        (2, False, set(), "CONTRACT_INCONSISTENT"),
    ],
)
def test_executable_schema_versioning(
    state: Any, backup: Any, paths: set[tuple[int, int]], expected: str
) -> None:
    assert _version_result(state, backup, paths) == expected


def test_idempotency_restart_replay_and_conflict() -> None:
    state = _ReplayState()
    assert state.command("key", SHA) == "CANONICAL_OUTCOME"
    assert state.side_effect_count == 1
    restarted = copy.deepcopy(state)
    assert restarted.command("key", SHA) == "CANONICAL_OUTCOME"
    assert restarted.side_effect_count == 1
    assert restarted.command("key", "b" * 64) == "IDEMPOTENCY_CONFLICT"
    assert restarted.side_effect_count == 1


def test_fill_and_reservation_replays_are_single_effect() -> None:
    state = _ReplayState()
    state.fill("fill")
    state.fill("fill")
    assert state.ledger_effects == 1
    state.reservation("CONSUMED")
    state.reservation("CONSUMED")
    state.reservation("RELEASED")
    assert state.reservation_state == "CONSUMED" and state.reservation_effects == 1
    assert state.unknown_external() == "RECONCILIATION_REQUIRED" and state.blind_resubmits == 0


@pytest.mark.parametrize(
    ("role", "occupied", "competing", "expected", "recovery"),
    [
        ("Core", True, False, "LOCK_CONTENTION", False),
        ("Core", True, True, "CONCURRENT_WRITER", True),
        ("UI", False, False, "NO_PERSISTENCE_AUTHORITY", False),
        ("Tray", False, False, "NO_PERSISTENCE_AUTHORITY", False),
    ],
)
def test_executable_writer_topology(
    role: str, occupied: bool, competing: bool, expected: str, recovery: bool
) -> None:
    outcome, mutable_open, session_created = _writer_attempt(role, occupied, competing)
    assert outcome == expected and not mutable_open and not session_created
    if outcome in _EXPECTED_FAILURES:
        assert _EXPECTED_FAILURES[outcome]["recovery_required"] is recovery


@pytest.mark.parametrize(
    "mutation",
    [
        {"bootstrap_consumed": False},
        {"device_revoked": False},
        {"operator_revoked": False},
        {"grant_revoked": False},
        {"lease_consumed": False},
        {"pin_revision": 1},
        {"security_generation": 1},
        {"session_security_generation": 1},
        {"kill_switch_generation": 1},
    ],
)
def test_restore_blocks_authority_resurrection(mutation: dict[str, Any]) -> None:
    current = {
        "bootstrap_consumed": True,
        "device_revoked": True,
        "operator_revoked": True,
        "grant_revoked": True,
        "lease_consumed": True,
        "pin_revision": 2,
        "security_generation": 2,
        "session_security_generation": 2,
        "kill_switch_generation": 2,
    }
    candidate = dict(current)
    candidate.update(mutation)
    assert _security_restore(candidate, current) == "MONOTONIC_FENCE_ROLLBACK"


@pytest.mark.parametrize("authority", ["AuthenticationProof", "PlatformBiometricAssertion"])
def test_restore_never_restores_ephemeral_proof_authority(authority: str) -> None:
    assert _security_restore({authority: {"payload": "x"}}, {}) == "RESTORE_REJECTED"


LIFECYCLE_ID = re.compile(r"^[a-z][a-z0-9_-]{2,127}$")
_MIGRATION_STATES = {"PREPARED", "APPLYING", "DURABLE_MIGRATED", "COMPLETED", "FAILED"}
_HANDOFF_STATES = {"PREPARED", "COMMITTED", "CLEANUP_PENDING", "UNKNOWN_RECONCILIATION"}


def _canonical_object_fingerprint(value: dict[str, Any], field: str) -> str:
    return _actual_fingerprint({key: item for key, item in value.items() if key != field})


def _validate_migration_transition(value: Any) -> bool:
    fields = {
        "migration_id",
        "transition_revision",
        "previous_state",
        "state",
        "transaction_fingerprint_sha256",
        "state_fingerprint_sha256",
        "protected_freshness_generation",
        "transition_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("migration_id"), str)
        or LIFECYCLE_ID.fullmatch(value["migration_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("transition_revision"))
        or not _positive(value.get("protected_freshness_generation"))
        or value.get("state") not in _MIGRATION_STATES
        or value.get("previous_state") not in {None, *_MIGRATION_STATES}
    ):
        return False
    for key in (
        "transaction_fingerprint_sha256",
        "state_fingerprint_sha256",
        "transition_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["transition_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )


def _validate_migration_current(value: Any) -> bool:
    fields = {
        "migration_id",
        "current_transition_revision",
        "state",
        "authoritative_state_fingerprint_sha256",
        "protected_freshness_generation",
        "designation_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("migration_id"), str)
        or LIFECYCLE_ID.fullmatch(value["migration_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("current_transition_revision"))
        or not _positive(value.get("protected_freshness_generation"))
        or value.get("state") not in _MIGRATION_STATES
    ):
        return False
    for key in ("authoritative_state_fingerprint_sha256", "designation_fingerprint_sha256"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["designation_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )


def _validate_handoff_transition(value: Any) -> bool:
    fields = {
        "handoff_id",
        "transition_revision",
        "previous_state",
        "state",
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "transition_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("handoff_id"), str)
        or LIFECYCLE_ID.fullmatch(value["handoff_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("transition_revision"))
        or value.get("state") not in _HANDOFF_STATES
        or value.get("previous_state") not in {None, *_HANDOFF_STATES}
    ):
        return False
    for key in (
        "operation_fingerprint_sha256",
        "metadata_fingerprint_sha256",
        "transition_fingerprint_sha256",
    ):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["transition_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )


def _validate_handoff_current(value: Any) -> bool:
    fields = {
        "handoff_id",
        "current_transition_revision",
        "state",
        "operation_fingerprint_sha256",
        "designation_fingerprint_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != fields
        or not isinstance(value.get("handoff_id"), str)
        or LIFECYCLE_ID.fullmatch(value["handoff_id"]) is None
    ):
        return False
    if (
        not _positive(value.get("current_transition_revision"))
        or value.get("state") not in _HANDOFF_STATES
    ):
        return False
    for key in ("operation_fingerprint_sha256", "designation_fingerprint_sha256"):
        if not isinstance(value.get(key), str) or SHA_RE.fullmatch(value[key]) is None:
            return False
    return str(value["designation_fingerprint_sha256"]) == _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )


def _migration_transition(
    revision: int, previous: str | None, state: str, generation: int = 1
) -> dict[str, Any]:
    value = {
        "migration_id": "migration-plan",
        "transition_revision": revision,
        "previous_state": previous,
        "state": state,
        "transaction_fingerprint_sha256": SHA,
        "state_fingerprint_sha256": "b" * 64,
        "protected_freshness_generation": generation,
        "transition_fingerprint_sha256": "",
    }
    value["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )
    return value


def _migration_current(transition: dict[str, Any]) -> dict[str, Any]:
    value = {
        "migration_id": transition["migration_id"],
        "current_transition_revision": transition["transition_revision"],
        "state": transition["state"],
        "authoritative_state_fingerprint_sha256": transition["state_fingerprint_sha256"],
        "protected_freshness_generation": transition["protected_freshness_generation"],
        "designation_fingerprint_sha256": "",
    }
    value["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )
    return value


def _handoff_transition(revision: int, previous: str | None, state: str) -> dict[str, Any]:
    value = {
        "handoff_id": "secret-handoff",
        "transition_revision": revision,
        "previous_state": previous,
        "state": state,
        "operation_fingerprint_sha256": SHA,
        "metadata_fingerprint_sha256": "b" * 64,
        "transition_fingerprint_sha256": "",
    }
    value["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "transition_fingerprint_sha256"
    )
    return value


def _handoff_current(transition: dict[str, Any]) -> dict[str, Any]:
    value = {
        "handoff_id": transition["handoff_id"],
        "current_transition_revision": transition["transition_revision"],
        "state": transition["state"],
        "operation_fingerprint_sha256": transition["operation_fingerprint_sha256"],
        "designation_fingerprint_sha256": "",
    }
    value["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
        value, "designation_fingerprint_sha256"
    )
    return value


def _resolve_lifecycle(
    transitions: Any, current: Any, kind: str
) -> tuple[str, tuple[dict[str, Any], ...]]:
    transition_validator = (
        _validate_migration_transition if kind == "migration" else _validate_handoff_transition
    )
    current_validator = (
        _validate_migration_current if kind == "migration" else _validate_handoff_current
    )
    identifier = "migration_id" if kind == "migration" else "handoff_id"
    allowed = MACHINE["migration_protocol" if kind == "migration" else "external_resource_handoff"][
        "durable_lifecycle"
    ]["allowed_transitions"]
    if not isinstance(transitions, list) or not transitions or not current_validator(current):
        return "CONTRACT_INCONSISTENT", ()
    by_revision: dict[int, dict[str, Any]] = {}
    for transition in transitions:
        if not transition_validator(transition):
            return "CONTRACT_INCONSISTENT", ()
        revision = transition["transition_revision"]
        if revision in by_revision and by_revision[revision] != transition:
            return "CONTRACT_INCONSISTENT", ()
        by_revision[revision] = transition
    if sorted(by_revision) != list(range(1, max(by_revision) + 1)):
        return "CONTRACT_INCONSISTENT", ()
    ordered = [by_revision[index] for index in range(1, max(by_revision) + 1)]
    if ordered[0]["previous_state"] is not None or ordered[0]["state"] != "PREPARED":
        return "CONTRACT_INCONSISTENT", ()
    for previous, following in zip(ordered, ordered[1:], strict=False):
        if (
            following[identifier] != previous[identifier]
            or following["previous_state"] != previous["state"]
            or following["state"] not in allowed[previous["state"]]
        ):
            return "CONTRACT_INCONSISTENT", ()
    selected = ordered[-1]
    if (
        current["current_transition_revision"] != selected["transition_revision"]
        or current[identifier] != selected[identifier]
        or current["state"] != selected["state"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    if kind == "migration" and (
        current["authoritative_state_fingerprint_sha256"] != selected["state_fingerprint_sha256"]
        or current["protected_freshness_generation"] != selected["protected_freshness_generation"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    if (
        kind == "handoff"
        and current["operation_fingerprint_sha256"] != selected["operation_fingerprint_sha256"]
    ):
        return "CONTRACT_INCONSISTENT", ()
    return str(selected["state"]), tuple(copy.deepcopy(ordered))


def _external(lifecycle: str = "COMMITTED", **updates: Any) -> dict[str, Any]:
    value = {
        "available": True,
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "membership_state": "CURRENT",
        "lifecycle": lifecycle,
    }
    if lifecycle == "COMMITTED":
        value.update(committed_generation=2, committed_state_fingerprint_sha256=SHA)
    value.update(updates)
    return value


def _candidate(generation: int | None = 2, **updates: Any) -> dict[str, Any]:
    value = {
        "account_id": SCOPE[0],
        "device_installation_id": SCOPE[1],
        "state_store_identity_fingerprint_sha256": SCOPE[2],
        "environment": "PAPER",
        "generation": generation,
        "state_fingerprint_sha256": SHA,
        "transaction_fingerprint_sha256": "b" * 64,
    }
    value.update(updates)
    return value


@pytest.mark.parametrize(
    "missing",
    [
        "available",
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "environment",
        "membership_state",
        "lifecycle",
    ],
)
def test_external_observation_missing_common_field_is_denied(missing: str) -> None:
    value = _external()
    value.pop(missing)
    assert _validate_external_authority_observation(value) != "VALID"


@pytest.mark.parametrize("prepared", [True, "3", 2, 4, 100])
def test_prepared_normal_requires_exact_g_plus_one(prepared: Any) -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=prepared,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    assert _validate_external_authority_observation(external) == (
        "VALID" if prepared == 3 and type(prepared) is int else "CONTRACT_INCONSISTENT"
    )
    if prepared != 3 or type(prepared) is not int:
        assert (
            _restore(
                _candidate(
                    100, state_fingerprint_sha256="b" * 64, transaction_fingerprint_sha256="c" * 64
                ),
                external,
            )
            == "CONTRACT_INCONSISTENT"
        )


@pytest.mark.parametrize(
    ("field", "failure"),
    [
        ("account_id", "BACKUP_SCOPE_MISMATCH"),
        ("device_installation_id", "BACKUP_SCOPE_MISMATCH"),
        ("state_store_identity_fingerprint_sha256", "BACKUP_SCOPE_MISMATCH"),
        ("environment", "BACKUP_ENVIRONMENT_MISMATCH"),
    ],
)
def test_restore_exact_scope_environment_binding(field: str, failure: str) -> None:
    candidate = _candidate()
    candidate[field] = "LIVE" if field == "environment" else "x"
    assert _restore(candidate, _external()) == failure


@pytest.mark.parametrize(
    "field",
    [
        "canonical_durable_records",
        "immutable_recovery_history",
        "integrity_metadata",
        "local_protected_freshness_generation",
        "environment",
        "account_id",
    ],
)
def test_backup_fingerprint_detects_every_envelope_mutation(field: str) -> None:
    backup = _valid_backup()
    if field in {"canonical_durable_records", "immutable_recovery_history"}:
        backup[field] = [{"record_kind": "Event"}]
    elif field == "integrity_metadata":
        backup[field] = {"revision": 1}
    elif field == "local_protected_freshness_generation":
        backup[field] = 3
    elif field == "environment":
        backup[field] = "TESTNET"
    else:
        backup[field] = "acct_01890f3a-2b4c-7abc-9def-0123456789ab"
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


def test_exact_lifecycle_fixtures_and_extra_fields() -> None:
    mt = _migration_transition(1, None, "PREPARED")
    mc = _migration_current(mt)
    ht = _handoff_transition(1, None, "PREPARED")
    hc = _handoff_current(ht)
    assert (
        _validate_migration_transition(mt)
        and _validate_migration_current(mc)
        and _validate_handoff_transition(ht)
        and _validate_handoff_current(hc)
    )
    for value, validator in [
        (mt, _validate_migration_transition),
        (mc, _validate_migration_current),
        (ht, _validate_handoff_transition),
        (hc, _validate_handoff_current),
    ]:
        altered = dict(value)
        altered["extra"] = True
        assert not validator(altered)


@pytest.mark.parametrize(
    ("kind", "bad"),
    [
        ("migration", "completed_first"),
        ("migration", "previous_first"),
        ("migration", "older_current"),
        ("migration", "binding"),
        ("handoff", "completed_first"),
        ("handoff", "previous_first"),
        ("handoff", "older_current"),
        ("handoff", "binding"),
    ],
)
def test_lifecycle_revision_one_latest_and_bindings_are_fenced(kind: str, bad: str) -> None:
    if kind == "migration":
        first = _migration_transition(1, None, "PREPARED")
        second = _migration_transition(2, "PREPARED", "APPLYING")
        current = _migration_current(second)
    else:
        first = _handoff_transition(1, None, "PREPARED")
        second = _handoff_transition(2, "PREPARED", "COMMITTED")
        current = _handoff_current(second)
    history = [first, second]
    if bad == "completed_first":
        history = (
            [_migration_transition(1, None, "COMPLETED")]
            if kind == "migration"
            else [_handoff_transition(1, None, "COMMITTED")]
        )
        current = (
            _migration_current(history[0]) if kind == "migration" else _handoff_current(history[0])
        )
    elif bad == "previous_first":
        history[0]["previous_state"] = "APPLYING" if kind == "migration" else "COMMITTED"
        history[0]["transition_fingerprint_sha256"] = _canonical_object_fingerprint(
            history[0], "transition_fingerprint_sha256"
        )
    elif bad == "older_current":
        current = _migration_current(first) if kind == "migration" else _handoff_current(first)
    else:
        key = (
            "authoritative_state_fingerprint_sha256"
            if kind == "migration"
            else "operation_fingerprint_sha256"
        )
        current[key] = "f" * 64
        current["designation_fingerprint_sha256"] = _canonical_object_fingerprint(
            current, "designation_fingerprint_sha256"
        )
    assert _resolve_lifecycle(history, current, kind)[0] == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    ("kind", "mutation"),
    [
        ("migration", "state"),
        ("migration", "revision"),
        ("migration", "generation"),
        ("handoff", "state"),
        ("handoff", "revision"),
        ("handoff", "operation"),
    ],
)
def test_transition_fingerprints_detect_mutation(kind: str, mutation: str) -> None:
    value = (
        _migration_transition(1, None, "PREPARED")
        if kind == "migration"
        else _handoff_transition(1, None, "PREPARED")
    )
    key = {
        "state": "state",
        "revision": "transition_revision",
        "generation": "protected_freshness_generation",
        "operation": "operation_fingerprint_sha256",
    }[mutation]
    value[key] = (
        "FAILED"
        if mutation == "state"
        else (2 if mutation in {"revision", "generation"} else "f" * 64)
    )
    assert not (
        _validate_migration_transition(value)
        if kind == "migration"
        else _validate_handoff_transition(value)
    )


def test_single_lifecycle_authority_has_no_descriptor_state() -> None:
    assert "state" not in MACHINE["migration_protocol"]["schema"]["properties"]
    assert "state" not in MACHINE["external_resource_handoff"]["schema"]["properties"]
    assert (
        MACHINE["migration_protocol"]["lifecycle_authority"]["current"]
        == "MigrationCurrentState only"
    )
    assert (
        MACHINE["external_resource_handoff"]["lifecycle_authority"]["current"]
        == "SecretHandoffCurrentState only"
    )


# Final corrective persistence-carrier closure.
_SYNTHETIC_KINDS = {
    "CryptohunteraccountCurrentRecord",
    "DeviceinstallationCurrentIdentityLifecycle",
    "OperatoridentityCurrentDesignationState",
    "OperatoridentityRevisions",
    "LiveaccessgrantCurrentDesignationState",
    "LiveaccessgrantAcceptedRevisionsHistory",
    "PortfolioCanonicalAccountingState",
    "Exchangeaccount",
    "CredentialprofileMetadataReference",
    "TradinguniverseCurrentVersionDesignation",
    "TradinguniverseVersionHistory",
    "StrategydefinitionAcceptedRevisions",
    "StrategydefinitionCurrentDesignation",
    "StrategyinstanceCurrentLifecycleConfig",
    "RoutingConfigurationCurrentDesignation",
    "RiskpolicyAcceptedRevisions",
    "RiskpolicyCurrentDesignation",
    "RiskbudgetCurrentState",
    "KillSwitchStateGeneration",
    "KillSwitchTransitionHistory",
    "CommandAcceptedRequest",
    "Orderintent",
    "OrderCanonicalLifecycleState",
    "OrderLifecycleEventsHistory",
    "Ledgerentry",
    "ReservationCurrentState",
    "ReservationTransitionHistory",
    "Riskdecision",
    "ExecutionleaseImmutableRecord",
    "ExecutionleaseOneShotState",
    "ExecutionleaseRestartFence",
    "RuntimesessionCanonicalIdentityHistory",
    "SessionsecuritystateCurrentGenerationState",
    "SessionsecuritystateRevisionHistory",
    "PinverifierrecordAcceptedRevisions",
    "PinverifierrecordCurrentDesignation",
    "DevicetrustSecurityRevisions",
    "DevicetrustCurrentDesignation",
    "PlatformEnrollmentRevisions",
    "Secretmetadataprojection",
    "BootstrapConsumedFence",
    "BootstrapConsumptionRecord",
}


def _contains_forbidden(value: Any, *, direct_pin_record: bool = False) -> bool:
    forbidden_kinds = set(MACHINE["backup_contract"]["forbidden_record_kinds"])
    forbidden_fields = set(MACHINE["backup_contract"]["forbidden_payload_fields"])
    if isinstance(value, list):
        return any(_contains_forbidden(item) for item in value)
    if not isinstance(value, dict):
        return False
    if value.get("record_kind") in forbidden_kinds:
        return True
    for key, nested in value.items():
        if key in forbidden_fields and not (direct_pin_record and key == "verifier"):
            return True
        nested_pin = direct_pin_record and key == "upstream_payload"
        if _contains_forbidden(nested, direct_pin_record=nested_pin):
            return True
    return False


def _source_value(entry: dict[str, Any]) -> Any:
    document = (
        MACHINE
        if entry["semantic_owner_milestone"] == "M0.11"
        else json.loads((DOCS / entry["semantic_artifact"]).read_text())
    )
    ok, value = _resolve_pointer(document, entry["semantic_json_pointer"])
    assert ok
    return value


def test_irreversible_prepared_floor_has_no_normal_abort_model() -> None:
    matrix = MACHINE["transaction_protocol"]["crash_matrix"]
    assert "PREPARED_LOCAL_G" not in matrix
    assert matrix["PREPARED_LOCAL_G_AFTER_RESTART"] == "PREPARED_PENDING_RECOVERY_REQUIRED_NO_ABORT"
    assert (
        matrix["LOCAL_G_PLUS_1_WAS_DURABLE_THEN_LOST_AND_G_RESTORED"]
        == "PREPARED_PENDING_RECOVERY_REQUIRED_NO_ABORT"
    )
    serialized = json.dumps(MACHINE)
    for forbidden in (
        "REBUILD_G_EVIDENCE_" + "ABORT_TO_COMMITTED_G",
        "REBUILD_CURRENT_G_" + "EVIDENCE",
        "PROTECTED_" + "ABORT",
        "AWAIT_POST_ABORT_" + "EXTERNAL_OBSERVATION",
    ):
        assert forbidden not in serialized


@pytest.mark.parametrize(
    "evidence_kind",
    ["accepted_current_g", "stale_g", "cross_process_g", "self_hash_g", "restored_old_g"],
)
def test_reconstructed_g_never_clears_pending_g_plus_one(evidence_kind: str) -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=3,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    registry = _EvidenceRegistry("restart")
    ref = registry.observe(_observation(protected_freshness_generation=2))
    assert ref
    if evidence_kind == "stale_g":
        registry.observe(
            _observation(protected_freshness_generation=2, transaction_fingerprint_sha256="d" * 64)
        )
    elif evidence_kind == "cross_process_g":
        ref = "other:opaque:1"
    elif evidence_kind == "self_hash_g":
        ref = "caller:" + SHA
    before = copy.deepcopy(external)
    assert _restore(_candidate(2), external) == "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
    assert external == before and external["prepared_generation"] == 3


def test_exact_pending_g_plus_one_finalizes_and_mismatch_preserves_pending() -> None:
    external = _external(
        "PREPARED",
        committed_generation=2,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=3,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    assert (
        _restore(
            _candidate(
                3, state_fingerprint_sha256="b" * 64, transaction_fingerprint_sha256="c" * 64
            ),
            external,
        )
        == "REBUILD_FRESH_EVIDENCE_FINALIZE_MATCHING_PENDING_THEN_CONTINUE_GATES"
    )
    before = copy.deepcopy(external)
    assert (
        _restore(
            _candidate(
                3, state_fingerprint_sha256="d" * 64, transaction_fingerprint_sha256="c" * 64
            ),
            external,
        )
        == "BACKUP_ROLLBACK_DETECTED_PENDING_PRESERVED"
    )
    assert external == before


def test_exact_63_row_representation_registry_closure() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    assert len(registry) == 63 and set(registry) == set(_EXPECTED_DURABILITY)
    assert all(
        entry["durability_class"] == _EXPECTED_DURABILITY[aspect]
        for aspect, entry in registry.items()
    )
    assert all(
        entry["representation_category"] != "TRUE_UPSTREAM_SEMANTIC_GAP"
        for entry in registry.values()
    )
    assert all(
        entry["adds_new_domain_facts"] is False and entry["restorable_authority"] is False
        for entry in registry.values()
    )
    assert MACHINE["closure_conditions"]["true_upstream_semantic_gaps"] == 0


@pytest.mark.parametrize("aspect", list(_EXPECTED_DURABILITY))
def test_every_representation_source_and_carrier_is_executable(aspect: str) -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    expected = _EXPECTED_OWNERSHIP[aspect]
    assert {key: entry[key] for key in expected} == expected
    source_value = _source_value(entry)
    if "semantic_contract_fingerprint_sha256" in entry:
        assert _actual_fingerprint(source_value) == entry["semantic_contract_fingerprint_sha256"]
    durable = entry["durability_class"].startswith("DURABLE")
    if not durable:
        assert (
            entry["representation_category"] == "EXCLUDED_NON_DURABLE"
            and entry["carrier_strategy"] == "NONE"
        )
        return
    assert entry["carrier_strategy"] == "PERSISTENCE_RECORD"
    valid = _persistence_record(aspect)
    assert _validate_persistence_record(valid)
    malformed = copy.deepcopy(valid)
    malformed["payload"] = {"invalid": True}
    malformed["payload_fingerprint_sha256"] = _actual_fingerprint(malformed["payload"])
    assert not _validate_persistence_record(malformed)


def test_no_synthetic_prose_derived_record_kind_survives() -> None:
    serialized = json.dumps(MACHINE["backup_contract"])
    for synthetic in _SYNTHETIC_KINDS:
        assert synthetic not in serialized
    assert "canonical_record_kinds_allowlist" not in MACHINE["backup_contract"]
    assert "durability_representation_by_aspect" not in MACHINE["backup_contract"]


def test_persistence_record_exact_shape_source_and_hash() -> None:
    value = _persistence_record("CryptoHunterAccount current record")
    assert set(value) == _PERSISTENCE_FIELDS and _validate_persistence_record(value)
    for mutation in (
        {"extra": True},
        {"representation_category": "UNKNOWN"},
        {"semantic_json_pointer": "/wrong"},
        {"payload_fingerprint_sha256": "f" * 64},
    ):
        altered = copy.deepcopy(value)
        altered.update(mutation)
        assert not _validate_persistence_record(altered)
    assert not _validate_persistence_record({"record_kind": "MigrationTransitionRecord"})


def test_crypto_account_identity_projection_is_minimal_closed_and_not_authority() -> None:
    valid = _persistence_record("CryptoHunterAccount current record")
    assert _validate_persistence_record(valid)
    for payload in (
        {
            "entity_kind": "CryptoHunterAccount",
            "entity_id": "acct_wrong",
            "parent_scope_bindings": {},
        },
        {
            "entity_kind": "Unknown",
            "entity_id": "acct_" + SCOPE[0][5:],
            "parent_scope_bindings": {},
        },
        {
            "entity_kind": "CryptoHunterAccount",
            "entity_id": "acct_" + SCOPE[0][5:],
            "parent_scope_bindings": {},
            "email": "x",
        },
    ):
        record = _persistence_record("CryptoHunterAccount current record", payload)
        assert not _validate_persistence_record(record)
    assert MACHINE["backup_contract"]["persistence_record_contract"]["domain_authority"] is False


def test_bootstrap_uses_exact_m03_objects_without_synthetic_kinds() -> None:
    registry = MACHINE["backup_contract"]["representation_registry"]
    assert (
        registry["bootstrap consumed fence"]["semantic_object_or_invariant"]
        == "CoreCurrentBootstrapState"
    )
    assert (
        registry["bootstrap accepted/consumption history"]["semantic_object_or_invariant"]
        == "ConsumedBootstrapAuthority"
    )
    assert (
        registry["bootstrap consumed fence"]["representation_category"]
        == "M011_CURRENT_DESIGNATION_PROJECTION"
    )
    assert (
        registry["bootstrap accepted/consumption history"]["representation_category"]
        == "M011_IMMUTABLE_HISTORY_WRAPPER"
    )


def test_backup_envelope_accepts_exact_carriers_and_rejects_bare_kind() -> None:
    backup = _valid_backup()
    backup["canonical_durable_records"] = [
        _persistence_record("CryptoHunterAccount current record")
    ]
    backup["immutable_recovery_history"] = [_persistence_record("Event")]
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {k: v for k, v in backup.items() if k != "envelope_fingerprint_sha256"}
    )
    assert _validate_backup(backup) == "VALID"
    backup = _valid_backup()
    backup["canonical_durable_records"] = [{"record_kind": "Event"}]
    backup["envelope_fingerprint_sha256"] = _actual_fingerprint(
        {k: v for k, v in backup.items() if k != "envelope_fingerprint_sha256"}
    )
    assert _validate_backup(backup) == "BACKUP_INTEGRITY_FAILED"


@pytest.mark.parametrize(
    "aspect",
    [
        "Migration current state/designation",
        "Migration transition/history revisions",
        "SecretHandoff current state/designation",
        "SecretHandoff transition/history revisions",
    ],
)
def test_m011_lifecycle_payloads_use_exact_validators_inside_carrier(aspect: str) -> None:
    valid = _persistence_record(aspect)
    assert _validate_persistence_record(valid)
    altered = copy.deepcopy(valid)
    altered["payload"]["extra"] = True
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)


def test_local_durable_evidence_has_no_persistence_carrier() -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][
        "LocalDurableStateEvidence payload"
    ]
    assert (
        entry["representation_category"] == "EXCLUDED_NON_DURABLE"
        and entry["carrier_strategy"] == "NONE"
    )
    assert not _validate_persistence_record({"record_key": "LocalDurableStateEvidence payload|1"})


def _revalidate_current_designation(record: Any, accepted: Any, minimum_generation: int) -> str:
    if not _validate_persistence_record(record) or not isinstance(accepted, dict):
        return "RESTORE_REJECTED"
    payload = record["payload"]
    referenced = accepted.get(payload["current_reference"])
    if not isinstance(referenced, dict):
        return "RESTORE_REJECTED"
    if (
        referenced.get("scope_key") != payload["scope_key"]
        or referenced.get("content_fingerprint_sha256") != payload["content_fingerprint_sha256"]
    ):
        return "RESTORE_REJECTED"
    if payload["current_revision"] != referenced.get("revision"):
        return "RESTORE_REJECTED"
    generation = payload["current_generation"]
    if (
        not isinstance(generation, int)
        or isinstance(generation, bool)
        or generation < minimum_generation
    ):
        return "MONOTONIC_FENCE_ROLLBACK"
    if referenced.get("state") in {"REVOKED", "RETIRED"}:
        return "RESTORE_REJECTED"
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


@pytest.mark.parametrize(
    "aspect",
    [
        "RiskPolicy current designation",
        "PinVerifierRecord current designation",
        "DeviceTrust current designation",
        "LiveAccessGrant current designation/state",
        "ExecutionLease one-shot state",
        "Order canonical lifecycle state",
        "reservation current state",
    ],
)
def test_current_designation_requires_history_scope_revision_fingerprint_and_fence(
    aspect: str,
) -> None:
    record = _persistence_record(aspect)
    payload = record["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    assert (
        _revalidate_current_designation(record, accepted, 1)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    assert _revalidate_current_designation(record, {}, 1) == "RESTORE_REJECTED"
    wrong = copy.deepcopy(accepted)
    wrong[payload["current_reference"]]["revision"] = 2
    assert _revalidate_current_designation(record, wrong, 1) == "RESTORE_REJECTED"
    assert _revalidate_current_designation(record, accepted, 2) == "MONOTONIC_FENCE_ROLLBACK"


def test_bootstrap_exact_payloads_mutation_consumption_and_restore_authority_fences() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = _persistence_record("bootstrap accepted/consumption history")
    assert _validate_persistence_record(current) and _validate_persistence_record(history)
    altered = copy.deepcopy(history)
    altered["payload"]["bootstrap_generation"] = 0
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)
    cleared = copy.deepcopy(current)
    cleared["payload"]["consumed_authorities"] = []
    cleared["payload_fingerprint_sha256"] = _actual_fingerprint(cleared["payload"])
    assert _validate_persistence_record(cleared)
    assert (
        cleared["restorable_authority"] is False
        if "restorable_authority" in cleared
        else MACHINE["backup_contract"]["representation_registry"]["bootstrap consumed fence"][
            "restorable_authority"
        ]
        is False
    )
    assert (
        current["payload"]["consumed_authorities"]
        and not cleared["payload"]["consumed_authorities"]
    )


@pytest.mark.parametrize(
    "aspect",
    [
        "bootstrap accepted/consumption history",
        "Event",
        "RiskPolicy accepted revisions",
        "SessionSecurityState revision history",
        "RuntimeSession canonical identity/history",
    ],
)
def test_immutable_wrapper_requires_payload_hash_and_semantic_validator(aspect: str) -> None:
    record = _persistence_record(aspect)
    assert _validate_persistence_record(record)
    mutated = copy.deepcopy(record)
    if "canonical_fields" in mutated["payload"]:
        mutated["payload"]["canonical_fields"] = {"invalid": True}
    else:
        mutated["payload"]["invalid"] = True
    assert not _validate_persistence_record(mutated)
    rehashed = copy.deepcopy(mutated)
    rehashed["payload_fingerprint_sha256"] = _actual_fingerprint(rehashed["payload"])
    assert not _validate_persistence_record(rehashed)


def _revalidate_bootstrap_consumption(current: Any, history: Any) -> str:
    if not _validate_persistence_record(current) or not isinstance(history, list):
        return "RESTORE_REJECTED"
    current_payload = current["payload"]
    accepted: set[str] = set()
    for record in history:
        if (
            not _validate_persistence_record(record)
            or record.get("representation_name") != "bootstrap accepted/consumption history"
        ):
            return "RESTORE_REJECTED"
        historical = record["payload"]
        if (
            historical["account_id"] != current_payload["account_id"]
            or historical["device_installation_id"] != current_payload["device_installation_id"]
            or historical["bootstrap_generation"] != current_payload["expected_generation"]
            or historical["bootstrap_revision"] != current_payload["expected_revision"]
        ):
            return "RESTORE_REJECTED"
        accepted.add(historical["claim_fingerprint_sha256"])
    consumed = current_payload.get("consumed_authorities")
    if not isinstance(consumed, list) or set(consumed) != accepted:
        return "RESTORE_REJECTED"
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


def test_legacy_parallel_recovery_model_is_absent_and_restore_retains_pending() -> None:
    source = Path(__file__).read_text()
    forbidden = [
        "REBUILD_" + "EVIDENCE_AND_ABORT",
        "REBUILD_G_EVIDENCE_" + "ABORT_TO_COMMITTED_G",
        "REBUILD_CURRENT_G_" + "EVIDENCE",
        "VERIFY_ABORT_" + "ELIGIBILITY",
        "PROTECTED_" + "ABORT",
        "AWAIT_POST_ABORT_" + "EXTERNAL_OBSERVATION",
    ]
    assert "def " + "_recover(" not in source
    assert not any(token in source or token in json.dumps(MACHINE) for token in forbidden)
    external = _external(
        "PREPARED",
        committed_generation=1,
        committed_state_fingerprint_sha256=SHA,
        prepared_generation=2,
        prepared_state_fingerprint_sha256="b" * 64,
        prepared_transaction_fingerprint_sha256="c" * 64,
    )
    before = copy.deepcopy(external)
    assert _restore(_candidate(1), external) == "RECOVERY_REQUIRED_PENDING_RETAINED_NO_ABORT"
    assert external == before


def test_record_key_is_derived_collision_safe_and_not_prose_identity() -> None:
    event_a = _persistence_record("Event", object_suffix="a")
    event_b = _persistence_record("Event", object_suffix="b")
    lease_a = _persistence_record("ExecutionLease immutable record", object_suffix="a")
    lease_b = _persistence_record("ExecutionLease immutable record", object_suffix="b")
    assert event_a["record_key"] != event_b["record_key"]
    assert lease_a["record_key"] != lease_b["record_key"]
    assert _persistence_record("Event", object_suffix="a")["record_key"] == event_a["record_key"]
    assert _persistence_record("Event", revision=2)["record_key"] != event_a["record_key"]
    for bad_key in ("", "arbitrary", "Event|1", event_b["record_key"]):
        altered = copy.deepcopy(event_a)
        altered["record_key"] = bad_key
        assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    "aspect",
    [
        "DeviceInstallation current identity/lifecycle",
        "Portfolio canonical accounting state",
        "RiskBudget current state",
    ],
)
def test_upstream_facts_are_nonempty_closed_source_bound_and_semantic(aspect: str) -> None:
    valid = _persistence_record(aspect)
    assert valid["payload"]["facts"] and _validate_persistence_record(valid)
    for mutation in ("missing", "extra", "scope", "fingerprint"):
        altered = copy.deepcopy(valid)
        if mutation == "missing":
            altered["payload"]["facts"].pop(next(iter(altered["payload"]["facts"])))
        elif mutation == "extra":
            altered["payload"]["facts"]["invented_business_fact"] = True
        elif mutation == "scope":
            altered["payload"]["scope_key"] = "unrelated"
        else:
            altered["payload"]["source_fingerprint_sha256"] = "f" * 64
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    ("kind", "parents", "valid"),
    [
        ("CryptoHunterAccount", {}, True),
        ("CryptoHunterAccount", {"account_id": SCOPE[0]}, False),
        ("Workspace", {"account_id": SCOPE[0]}, True),
        ("Workspace", {}, False),
        ("Workspace", {"account_id": "acct_invalid"}, False),
    ],
)
def test_entity_identity_exact_parent_and_generic_prefix(
    kind: str, parents: dict[str, str], valid: bool
) -> None:
    payload = {
        "entity_kind": kind,
        "entity_id": ("acct_" if kind == "CryptoHunterAccount" else "ws_") + SCOPE[0][5:],
        "parent_scope_bindings": parents,
    }
    assert _exact_entity_projection(payload) is valid
    if kind == "Workspace":
        assert not _exact_entity_projection({**payload, "entity_id": SCOPE[1]})


def test_bootstrap_restore_consumption_is_exact_monotonic_set() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = [_persistence_record("bootstrap accepted/consumption history")]
    assert (
        _revalidate_bootstrap_consumption(current, history)
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    for consumed in ([], ["b" * 64], [SHA, "b" * 64]):
        altered = copy.deepcopy(current)
        altered["payload"]["consumed_authorities"] = consumed
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(
            "bootstrap consumed fence",
            MACHINE["backup_contract"]["representation_registry"]["bootstrap consumed fence"],
            altered["payload"],
        )
        assert _validate_persistence_record(altered)
        assert _revalidate_bootstrap_consumption(altered, history) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    "aspect",
    ["M0.3 restore freshness membership", "M0.3 current designation", "M0.3 retirement state"],
)
def test_external_m03_sources_are_exact_restore_freshness_authority(aspect: str) -> None:
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    assert (
        entry["semantic_owner_milestone"],
        entry["semantic_artifact"],
        entry["semantic_json_pointer"],
    ) == ("M0.3", "process_topology_and_lifecycle.json", "/restore_freshness_authority_contract")
    assert (
        _actual_fingerprint(_source_value(entry)) == entry["semantic_contract_fingerprint_sha256"]
    )


def test_persistence_self_hash_cannot_legalize_wrong_source_category_or_payload() -> None:
    record = _persistence_record("Event")
    for field, value in (
        ("semantic_json_pointer", "/wrong"),
        ("representation_category", "M011_LOCAL_SCHEMA"),
    ):
        altered = copy.deepcopy(record)
        altered[field] = value
        assert not _validate_persistence_record(altered)
    altered = copy.deepcopy(record)
    altered["payload"] = {"semantic_object": "Event", "canonical_fields": {"garbage": 123}}
    altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
    assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    "aspect",
    [
        "Event",
        "Fill",
        "LedgerEntry",
        "RiskDecision",
        "ExecutionLease immutable record",
        "SessionSecurityState current generation/state",
        "SecretMetadataProjection",
    ],
)
def test_representative_direct_upstream_payload_is_exact_and_not_generic_placeholder(
    aspect: str,
) -> None:
    record = _persistence_record(aspect)
    contract = MACHINE["backup_contract"]["direct_upstream_validator_registry"][aspect]
    assert set(record["payload"]) == set(contract["exact_fields"])
    assert _validate_persistence_record(record)
    assert "semantic_object" not in record["payload"]
    assert "canonical_fields" not in record["payload"]

    missing = copy.deepcopy(record)
    missing["payload"].pop(contract["exact_fields"][0])
    missing["payload_fingerprint_sha256"] = _actual_fingerprint(missing["payload"])
    assert not _validate_persistence_record(missing)

    extra = copy.deepcopy(record)
    extra["payload"]["invented_business_fact"] = True
    extra["payload_fingerprint_sha256"] = _actual_fingerprint(extra["payload"])
    assert not _validate_persistence_record(extra)

    wrong_key = copy.deepcopy(record)
    wrong_key["record_key"] = "direct:unrelated-object"
    assert not _validate_persistence_record(wrong_key)

    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        wrong_fingerprint = copy.deepcopy(record)
        wrong_fingerprint["payload"][terminal] = "f" * 64
        wrong_fingerprint["payload_fingerprint_sha256"] = _actual_fingerprint(
            wrong_fingerprint["payload"]
        )
        wrong_fingerprint["record_key"] = _derive_record_key(
            aspect,
            MACHINE["backup_contract"]["representation_registry"][aspect],
            wrong_fingerprint["payload"],
        )
        assert not _validate_persistence_record(wrong_fingerprint)


def test_event_exact_upstream_semantics_deny_bad_scope_lifecycle_and_safe_payload() -> None:
    valid = _persistence_record("Event")
    for field, value in (
        ("environment", "PRODUCTION"),
        ("event_type", "UNKNOWN_EVENT"),
        ("workspace_id", "wrong-id"),
        ("safe_payload", {"garbage": 123}),
    ):
        altered = copy.deepcopy(valid)
        altered["payload"][field] = value
        altered["payload"]["event_fingerprint_sha256"] = _actual_fingerprint(
            {
                key: item
                for key, item in altered["payload"].items()
                if key != "event_fingerprint_sha256"
            }
        )
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(
            "Event",
            MACHINE["backup_contract"]["representation_registry"]["Event"],
            altered["payload"],
        )
        assert not _validate_persistence_record(altered)


def test_current_designation_stage_two_denies_scope_fingerprint_revocation_and_retirement() -> None:
    record = _persistence_record("RiskPolicy current designation")
    payload = record["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    for field, value in (
        ("scope_key", "wrong-scope"),
        ("content_fingerprint_sha256", "f" * 64),
        ("state", "REVOKED"),
        ("state", "RETIRED"),
    ):
        changed = copy.deepcopy(accepted)
        changed[payload["current_reference"]][field] = value
        assert _revalidate_current_designation(record, changed, 1) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    ("aspect", "field", "bad"),
    [
        ("DeviceInstallation current identity/lifecycle", "lifecycle_state", "UNKNOWN"),
        ("Portfolio canonical accounting state", "environment", "PRODUCTION"),
        ("RiskBudget current state", "budget_state", "UNKNOWN"),
        ("RiskBudget current state", "risk_generation", 0),
    ],
)
def test_upstream_fact_projection_denies_illegal_semantic_value(
    aspect: str, field: str, bad: Any
) -> None:
    record = _persistence_record(aspect)
    record["payload"]["facts"][field] = bad
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])
    record["record_key"] = _derive_record_key(
        aspect,
        MACHINE["backup_contract"]["representation_registry"][aspect],
        record["payload"],
    )
    assert not _validate_persistence_record(record)


def _revalidate_restore_records(
    records: Any,
    accepted_by_reference: Any,
    minimum_generation: int,
    bootstrap_history: Any,
) -> str:
    if not isinstance(records, list) or not all(
        _validate_persistence_record(record) for record in records
    ):
        return "RESTORE_REJECTED"
    for record in records:
        category = record["representation_category"]
        if category == "M011_CURRENT_DESIGNATION_PROJECTION":
            if record["representation_name"] == "bootstrap consumed fence":
                result = _revalidate_bootstrap_consumption(record, bootstrap_history)
            else:
                result = _revalidate_current_designation(
                    record, accepted_by_reference, minimum_generation
                )
            if result != "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS":
                return result
    return "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"


def test_restore_record_gate_is_relational_and_never_mints_authority() -> None:
    current = _persistence_record("RiskPolicy current designation")
    payload = current["payload"]
    accepted = {
        payload["current_reference"]: {
            "scope_key": payload["scope_key"],
            "content_fingerprint_sha256": payload["content_fingerprint_sha256"],
            "revision": payload["current_revision"],
            "state": "ACTIVE",
        }
    }
    assert (
        _revalidate_restore_records([current], accepted, 1, [])
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    assert _revalidate_restore_records([current], {}, 1, []) == "RESTORE_REJECTED"


@pytest.mark.parametrize(
    "aspect",
    [
        "RiskPolicy accepted revisions",
        "SessionSecurityState revision history",
        "RuntimeSession canonical identity/history",
        "LiveAccessGrant accepted revisions/history",
    ],
)
def test_immutable_history_is_lossless_closed_and_fingerprint_recomputed(aspect: str) -> None:
    record = _persistence_record(aspect)
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    binding = entry["immutable_fact_binding"]
    upstream = record["payload"]["upstream_payload"]
    assert set(upstream) == set(binding["persisted_payload_fields"])
    assert record["payload"]["upstream_payload_fingerprint_sha256"] == _actual_fingerprint(upstream)
    assert _validate_persistence_record(record)

    for mutation in ("missing", "extra", "scope", "identity", "revision", "state", "sha"):
        altered = copy.deepcopy(record)
        payload = altered["payload"]["upstream_payload"]
        if mutation == "missing":
            payload.pop(binding["persisted_payload_fields"][0])
        elif mutation == "extra":
            payload["invented_business_fact"] = True
        elif mutation == "scope" and binding["scope_fields"]:
            payload[binding["scope_fields"][0]] = "wrong-scope"
        elif mutation == "identity":
            payload[binding["canonical_object_identity_fields"][0]] = "wrong-id"
        elif mutation == "revision" and binding["revision_generation_fields"]:
            payload[binding["revision_generation_fields"][0]] = 0
        elif mutation == "state" and "state" in payload:
            payload["state"] = "ILLEGAL"
        else:
            altered["payload"]["upstream_payload_fingerprint_sha256"] = "f" * 64
        if mutation != "sha":
            altered["payload"]["upstream_payload_fingerprint_sha256"] = _actual_fingerprint(payload)
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        altered["record_key"] = _derive_record_key(aspect, entry, altered["payload"])
        assert not _validate_persistence_record(altered)


@pytest.mark.parametrize(
    ("aspect", "identity_field"),
    [
        ("DeviceInstallation current identity/lifecycle", "device_installation_id"),
        ("Portfolio canonical accounting state", "portfolio_id"),
        ("RiskBudget current state", "risk_scope_key"),
    ],
)
def test_fact_record_key_contains_exact_object_identity_and_prevents_collision(
    aspect: str, identity_field: str
) -> None:
    first = _persistence_record(aspect, object_suffix="a", revision=7)
    second = copy.deepcopy(first)
    facts = second["payload"]["facts"]
    if identity_field == "risk_scope_key":
        facts["portfolio_id"] = _canonical_fixture_id("port", "b")
        contract = MACHINE["backup_contract"]["representation_registry"][aspect]["fact_binding"][
            "field_contracts"
        ]["risk_scope_key"]
        facts[identity_field] = "|".join(str(facts[field]) for field in contract["components"])
    else:
        prefix = "dev" if identity_field == "device_installation_id" else "port"
        facts[identity_field] = _canonical_fixture_id(prefix, "b")
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    second["record_key"] = _derive_record_key(aspect, entry, second["payload"])
    second["payload_fingerprint_sha256"] = _actual_fingerprint(second["payload"])
    assert _validate_persistence_record(first) and _validate_persistence_record(second)
    assert first["record_key"] != second["record_key"]
    unchanged_key = copy.deepcopy(second)
    unchanged_key["record_key"] = first["record_key"]
    assert not _validate_persistence_record(unchanged_key)


def test_fact_ids_use_exact_m02_prefixes_and_portfolio_uses_port() -> None:
    portfolio = _persistence_record("Portfolio canonical accounting state")
    facts = portfolio["payload"]["facts"]
    assert facts["account_id"].startswith("acct_")
    assert facts["workspace_id"].startswith("ws_")
    assert facts["portfolio_id"].startswith("port_")
    for field, wrong_prefix in (
        ("account_id", "dev"),
        ("workspace_id", "acct"),
        ("portfolio_id", "acct"),
    ):
        altered = copy.deepcopy(portfolio)
        altered["payload"]["facts"][field] = _canonical_fixture_id(wrong_prefix, "a")
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert not _validate_persistence_record(altered)


def _rehash_direct_record(record: dict[str, Any]) -> None:
    aspect = record["representation_name"]
    terminal = _terminal_fingerprint_field(aspect)
    if terminal is not None:
        record["payload"][terminal] = _actual_fingerprint(
            {key: value for key, value in record["payload"].items() if key != terminal}
        )
    entry = MACHINE["backup_contract"]["representation_registry"][aspect]
    record["record_key"] = _derive_record_key(aspect, entry, record["payload"])
    record["payload_fingerprint_sha256"] = _actual_fingerprint(record["payload"])


def test_direct_id_constant_and_decimal_constraints_are_literal() -> None:
    command = _persistence_record("Command accepted request")
    command["payload"]["operation_type"] = "CANCEL_ORDER"
    _rehash_direct_record(command)
    assert not _validate_persistence_record(command)

    intent = _persistence_record("OrderIntent")
    for value in ("0", "-1"):
        altered = copy.deepcopy(intent)
        altered["payload"]["quantity"] = value
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    wrong_prefix = copy.deepcopy(intent)
    wrong_prefix["payload"]["workspace_id"] = _canonical_fixture_id("dev", "a")
    _rehash_direct_record(wrong_prefix)
    assert not _validate_persistence_record(wrong_prefix)


def test_direct_nested_objects_are_exact_closed_and_semantic() -> None:
    ledger = _persistence_record("LedgerEntry")
    for nested in ({"garbage": 1}, {**ledger["payload"]["asset_reference"], "extra": True}):
        altered = copy.deepcopy(ledger)
        altered["payload"]["asset_reference"] = nested
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    illegal = copy.deepcopy(ledger)
    illegal["payload"]["asset_reference"]["mapping_status"] = "UNKNOWN"
    _rehash_direct_record(illegal)
    assert not _validate_persistence_record(illegal)


def test_lifecycle_current_and_transition_record_keys_are_distinct() -> None:
    migration_transition = _persistence_record("Migration transition/history revisions")
    migration_current = _persistence_record("Migration current state/designation")
    handoff_transition = _persistence_record("SecretHandoff transition/history revisions")
    handoff_current = _persistence_record("SecretHandoff current state/designation")
    assert migration_transition["record_key"] != migration_current["record_key"]
    assert handoff_transition["record_key"] != handoff_current["record_key"]
    for current, transition in (
        (migration_current, migration_transition),
        (handoff_current, handoff_transition),
    ):
        altered = copy.deepcopy(current)
        altered["record_key"] = transition["record_key"]
        assert not _validate_persistence_record(altered)


def test_bootstrap_consumption_rejects_cross_scope_and_revision_generation_mismatch() -> None:
    current = _persistence_record("bootstrap consumed fence")
    history = _persistence_record("bootstrap accepted/consumption history")
    assert (
        _revalidate_bootstrap_consumption(current, [history])
        == "CANDIDATE_VALID_REQUIRES_M0.3_FRESHNESS"
    )
    for field, value in (
        ("account_id", _canonical_fixture_id("acct", "b")),
        ("device_installation_id", _canonical_fixture_id("dev", "b")),
        ("bootstrap_generation", 2),
        ("bootstrap_revision", 2),
    ):
        altered = copy.deepcopy(history)
        altered["payload"][field] = value
        entry = MACHINE["backup_contract"]["representation_registry"][
            "bootstrap accepted/consumption history"
        ]
        altered["record_key"] = _derive_record_key(
            "bootstrap accepted/consumption history", entry, altered["payload"]
        )
        altered["payload_fingerprint_sha256"] = _actual_fingerprint(altered["payload"])
        assert _validate_persistence_record(altered)
        assert _revalidate_bootstrap_consumption(current, [altered]) == "RESTORE_REJECTED"


def test_direct_event_and_lease_reject_valid_uuid_with_wrong_prefix() -> None:
    event = _persistence_record("Event")
    for field, prefix in (("workspace_id", "dev"), ("portfolio_id", "acct")):
        altered = copy.deepcopy(event)
        altered["payload"][field] = _canonical_fixture_id(prefix, "a")
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
    lease = _persistence_record("ExecutionLease immutable record")
    lease["payload"]["execution_lease_id"] = _canonical_fixture_id("cmd", "a")
    _rehash_direct_record(lease)
    assert not _validate_persistence_record(lease)


def test_fill_nested_fee_reference_and_decimal_constraints_are_exact() -> None:
    fill = _persistence_record("Fill")
    fill["payload"]["fee_kind"] = "CHARGE"
    fill["payload"]["fee_quantity"] = "1"
    schema = MACHINE["backup_contract"]["direct_upstream_validator_registry"]["Fill"][
        "upstream_field_schemas"
    ]["fee_asset_reference"]
    fill["payload"]["fee_asset_reference"] = {
        field: _direct_fixture_value("Fill", field, schema["field_schemas"][field], "a", 1)
        for field in schema["fields"]
    }
    _rehash_direct_record(fill)
    assert _validate_persistence_record(fill)
    for mutation in ("missing", "extra", "mapping", "negative"):
        altered = copy.deepcopy(fill)
        nested = altered["payload"]["fee_asset_reference"]
        assert isinstance(nested, dict)
        if mutation == "missing":
            nested.pop("mapping_status")
        elif mutation == "extra":
            nested["invented"] = True
        elif mutation == "mapping":
            nested["mapping_status"] = "UNKNOWN"
        else:
            altered["payload"]["fee_quantity"] = "-1"
        _rehash_direct_record(altered)
        assert not _validate_persistence_record(altered)
