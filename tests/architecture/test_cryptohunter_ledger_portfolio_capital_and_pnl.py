"""Executable pure M0.8 closure model; never production runtime or persistence."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from decimal import localcontext
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, cast

import pytest

ROOT = Path(__file__).parents[2]
ARCH = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT: dict[str, Any] = json.loads((ARCH / "ledger_portfolio_capital_and_pnl.json").read_text())
CANONICAL = {
    name: json.loads((ARCH / name).read_text())
    for name in {
        "canonical_domain_vocabulary.json",
        "exchange_accounts_and_instruments.json",
        "strategy_market_data_and_execution_routing.json",
        "commands_events_order_lifecycle_and_idempotency.json",
    }
}
ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
HEX_RE = re.compile(r"^[0-9a-f]{64}$")
DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")
TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z$")
RULE = "ACCOUNTING_SPOT_FIFO_V1"


def freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(freeze(item) for item in value)
    return value


def canonical(value: Any) -> bytes:
    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: thaw(child) for key, child in item.items()}
        if isinstance(item, tuple):
            return [thaw(child) for child in item]
        if isinstance(item, str):
            return unicodedata.normalize("NFC", item)
        return item

    return json.dumps(
        thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def pointer(document: Any, path: str) -> Any:
    if path == "":
        return document
    if not path.startswith("/"):
        raise KeyError(path)
    node = document
    for token in path[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if type(node) is not dict or token not in node:
            raise KeyError(path)
        node = node[token]
    return node


EXPECTED_PROTOCOLS = freeze(
    {
        "schema_version": "1.0.0",
        "m0_element": "M0.8",
        "status": "closed",
        "authority": "target architecture contract; not production runtime or persistence",
        "source_of_truth": [
            "immutable canonical LedgerEntry journal",
            "immutable trusted upstream economic facts",
            "ACCOUNTING_SPOT_FIFO_V1 deterministic policy",
        ],
        "ledger_entry_schema": {
            "exact_fields": [
                "ledger_entry_id",
                "workspace_id",
                "portfolio_id",
                "environment",
                "exchange_account_id",
                "strategy_instance_id",
                "asset_reference",
                "account_role",
                "direction",
                "quantity",
                "source_type",
                "accounting_source_identity",
                "accounting_source_fingerprint_sha256",
                "accounting_rule_version",
                "posting_index",
                "posting_role",
                "batch_fingerprint_sha256",
                "effective_at_utc",
                "append_sequence",
                "order_id",
                "fill_id",
                "audit_event_id",
                "correction_reason",
            ],
            "nullable_fields": [
                "exchange_account_id",
                "strategy_instance_id",
                "order_id",
                "fill_id",
                "audit_event_id",
                "correction_reason",
            ],
            "identity": {"kind": "LedgerEntry", "prefix": "led", "format": "M0.2 uuidv7"},
            "quantity": "strictly-positive canonical M0.5 decimal string; zero and signed "
            "values forbidden",
            "direction_registry": ["DEBIT", "CREDIT"],
            "immutability": "append-only; UPDATE and DELETE forbidden",
            "posting_role_registry": [
                "ASSET_RECEIVED",
                "ASSET_PAID",
                "TRADE_COUNTERPART",
                "FEE_CLASSIFIED",
                "FEE_PAID",
                "CAPITAL_CLASSIFIED",
                "RESERVE_AVAILABLE",
                "RESERVE_HELD",
                "TRANSFER_SOURCE",
                "TRANSFER_DESTINATION",
                "TRANSFER_CLEARING_SOURCE",
                "TRANSFER_CLEARING_DESTINATION",
                "PNL_CLASSIFIED",
                "PNL_COUNTERPART",
                "RECONCILIATION_OWNED",
                "RECONCILIATION_COUNTERPART",
            ],
            "source_reference_constraints": {
                "fill": "fill_id and order_id required; audit_event_id and correction_reason null",
                "realized_pnl": "forbidden for independent append; "
                "Fill-derived PNL classification "
                "entries retain source_type fill",
                "non_fill": "audit_event_id required; fill_id and order_id null",
                "reconciliation_correction": "audit_event_id and "
                "non-empty "
                "correction_reason "
                "required",
            },
        },
        "account_roles": {
            "OWNED_AVAILABLE": "owned asset included in NAV",
            "OWNED_RESERVED": "owned asset included in NAV",
            "TRADE_CLEARING": "balancing counterpart; excluded from owned balances and NAV",
            "FEE_EXPENSE": "classification only; excluded from owned balances and NAV",
            "REALIZED_PNL_CLASSIFICATION": "classification only; excluded from owned balances and NAV",
            "EXTERNAL_CAPITAL": "capital classification only; excluded from owned balances and NAV",
            "TRANSFER_CLEARING": "balancing counterpart; excluded from owned balances and NAV",
            "RECONCILIATION_CLEARING": "balancing counterpart; excluded from owned balances and NAV",
        },
        "batch_protocol": {
            "atomicity": "validate identity, schema, transitions and per-asset balance before "
            "allocating IDs or appending; append entire batch or none",
            "accounting_source_identity": "fill uses canonical fill_id; non-fill sources require an "
            "immutable exact-economic AuditEvent whose audit_event_id "
            "is authority",
            "accounting_source_fingerprint": "sha256 of canonical exact economic source projection",
            "accounting_rule_version": "ACCOUNTING_SPOT_FIFO_V1",
            "posting_key": "(source_type, accounting_source_identity, accounting_rule_version, "
            "posting_index, posting_role)",
            "batch_fingerprint": "sha256 of ordered canonical posting projections excluding "
            "ledger_entry_id and append_sequence",
            "balance_invariant": "for every exact asset_reference in one batch, sum(DEBIT quantity) "
            "== sum(CREDIT quantity)",
            "idempotency": "same identity+fingerprint+rule returns REPLAY_SUCCESS and original "
            "result with zero append; changed fingerprint returns "
            "ACCOUNTING_IDENTITY_CONFLICT with zero mutation",
            "derived_batch_replay": "same source identity, fingerprint and rule with a different "
            "derived batch fingerprint is CONTRACT_INCONSISTENT with zero "
            "append",
            "fill_replay_integrity": "replay verifies accepted source fingerprint and every stored "
            "journal entry/batch fingerprint against accepted original "
            "batch; mismatch CONTRACT_INCONSISTENT, zero append",
        },
        "source_registry": {
            "fill": {
                "authority": "M0.7 composite trusted accepted Full Fill",
                "identity": "fill_id",
                "roles": [
                    "OWNED_AVAILABLE",
                    "TRADE_CLEARING",
                    "FEE_EXPENSE",
                    "REALIZED_PNL_CLASSIFICATION",
                ],
                "effects": "spot inventory, FIFO cost basis, gross realized P&L and exact fee "
                "asset",
                "failure": "fail closed",
            },
            "fee": {
                "authority": "registry-reserved M0.2 source type; standalone fee authority is "
                "unavailable in current M0.8",
                "identity": "none",
                "roles": [],
                "effects": "none; M0.7 Full Fill is the sole trade-fee authority",
                "failure": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
            "funding": {
                "authority": "not supplied by an execution-capable current-edition "
                "instrument; unsupported",
                "identity": "audit_event_id",
                "roles": [],
                "effects": "none",
                "failure": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
            "interest": {
                "authority": "no closed borrow/interest economics upstream; unsupported",
                "identity": "audit_event_id",
                "roles": [],
                "effects": "none",
                "failure": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
            "deposit": {
                "authority": "exact-economic AuditEvent with external/internal provenance",
                "identity": "audit_event_id",
                "roles": ["OWNED_AVAILABLE", "EXTERNAL_CAPITAL"],
                "effects": "external contribution only with explicit Portfolio-boundary provenance",
                "failure": "fail closed",
            },
            "withdrawal": {
                "authority": "exact-economic AuditEvent with external/internal provenance",
                "identity": "audit_event_id",
                "roles": ["OWNED_AVAILABLE", "EXTERNAL_CAPITAL"],
                "effects": "external withdrawal only with explicit Portfolio-boundary provenance",
                "failure": "fail closed",
            },
            "internal_transfer": {
                "authority": "exact-economic AuditEvent binding both legs and provenance",
                "identity": "audit_event_id",
                "roles": ["OWNED_AVAILABLE", "TRANSFER_CLEARING"],
                "effects": "no P&L; distinct venue references need explicit "
                "equivalence or fail closed",
                "failure": "fail closed",
            },
            "capital_reservation": {
                "authority": "accepted exact-economic M0.8 AuditEvent-derived accounting fact plus "
                "nominal sealed accepted M0.7 SUBMIT_ORDER command context; exact command/order/"
                "scope/account binding; reservation asset/quantity remain M0.8 economics; not M0.9 "
                "risk authority",
                "identity": "audit_event_id",
                "roles": ["OWNED_AVAILABLE", "OWNED_RESERVED"],
                "effects": "available to reserved only",
                "failure": "fail closed",
            },
            "capital_release": {
                "authority": "accepted exact-economic M0.8 AuditEvent-derived accounting fact plus "
                "nominal sealed accepted M0.7 terminal lifecycle/event context; exact order/scope/"
                "account and terminal-state mapping; legal predecessor and contiguous version "
                "proven upstream",
                "identity": "audit_event_id",
                "roles": ["OWNED_AVAILABLE", "OWNED_RESERVED"],
                "effects": "reserved to available only",
                "failure": "fail closed",
            },
            "realized_pnl": {
                "authority": "registry-reserved M0.2 source type; independent source "
                "is unsupported in M0.8",
                "identity": "none",
                "roles": [],
                "effects": "none; Fill-derived REALIZED_PNL_CLASSIFICATION entries use "
                "source_type fill in the same atomic Fill batch",
                "failure": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
            "reconciliation_correction": {
                "authority": "accepted exact-economic AuditEvent plus exact referenced accepted target batch "
                "and target-aware inverse validation",
                "identity": "audit_event_id",
                "roles": "inherited exactly from referenced accepted target batch; no standalone "
                "role authority",
                "effects": "exact direction-inverse compensation only; target-aware batch "
                "validation mandatory; no history mutation",
                "failure": "fail closed",
            },
        },
        "balance_model": {
            "key": [
                "workspace_id",
                "portfolio_id",
                "environment",
                "exchange_account_id",
                "asset_reference",
            ],
            "owned_total": "OWNED_AVAILABLE plus OWNED_RESERVED net projection",
            "available": "OWNED_AVAILABLE net projection",
            "reserved": "OWNED_RESERVED net projection",
            "invariant": "owned_total = available + reserved for current SPOT scope",
            "excluded_roles": [
                "TRADE_CLEARING",
                "FEE_EXPENSE",
                "REALIZED_PNL_CLASSIFICATION",
                "EXTERNAL_CAPITAL",
                "TRANSFER_CLEARING",
                "RECONCILIATION_CLEARING",
            ],
            "non_negative": "every current-SPOT OWNED_AVAILABLE and OWNED_RESERVED projection is "
            "non-negative after every accepted batch",
        },
        "reservation_protocol": {
            "binding": "nominal M07PrevalidatedAcceptedCommandContext proving exact canonical "
            "accepted SUBMIT_ORDER request, command fingerprint/membership and "
            "command/order/scope/account/instrument/route identity; reservation asset/quantity are "
            "separate M0.8 economics; command_id is idempotency identity",
            "effect": "available to reserved in exact scope; no equity/P&L",
            "fill_consumption": "matching Fill.order_id consumes exact reserved spend "
            "atomically before available; partial Fill leaves remainder",
            "terminal_release": ["REJECTED", "CANCELLED", "EXPIRED", "FILLED", "REPLACED"],
            "replay": "same command economics replays across a new audit_event_id; changed "
            "economics conflicts",
            "boundary": "accounting state only; no M0.9 approval",
            "terminal_evidence": "nominal M07PrevalidatedAcceptedTerminalOrderEventContext proving "
            "exact accepted M0.7 event envelope/fingerprint/version/history, event-to-terminal "
            "lifecycle target and exact order/scope/account/instrument/route",
        },
        "asset_identity": {
            "key": "exact M0.5 AssetReference object including venue_asset_code, "
            "canonical_display_code, asset_namespace and mapping_status",
            "accepted_mapping_status": ["EXACT", "EXPLICIT_ALIAS"],
            "normalization": "none",
            "cross_venue_unit_aggregation": "forbidden without explicit canonical equivalence "
            "authority, which M0.5 does not provide",
            "portfolio_value_aggregation": "allowed only through explicit trusted valuation paths",
        },
        "instrument_coverage": {
            "current_edition_execution_capable": ["SPOT_PAIR"],
            "accounting_supported": ["SPOT_PAIR"],
            "fail_closed": ["MARGIN_PAIR", "PERPETUAL_CONTRACT", "DELIVERY_FUTURE", "OPTION"],
            "reason": "M0.6 current-edition readiness/activation allows SPOT/SPOT_PAIR only; "
            "derivatives remain structurally representable but lack complete "
            "collateral, liability, funding, expiry/exercise/settlement accounting "
            "economics",
        },
        "cost_basis_policy": {
            "id": "ACCOUNTING_SPOT_FIFO_V1",
            "scope": "analytical SPOT inventory projection, not tax policy",
            "ordering": ["append_sequence", "posting_index"],
            "lots": "projection-only, keyed by accepted Fill identity and deterministic posting "
            "index",
            "arithmetic": "Fraction exact",
            "sell": "consume oldest gross acquired lots; base-asset fee reduces received "
            "inventory before lot creation; no negative inventory, short, leverage or "
            "flip",
            "late_fact": "append at a new sequence; never rewrite history",
            "same_effective_time": "append_sequence then posting_index",
            "lot_fields": [
                "quantity",
                "unit_cost_basis",
                "basis_valuation_unit",
                "source_identity",
                "append_sequence",
                "posting_index",
            ],
            "cross_unit": "explicit trusted historical conversion from basis_valuation_unit to Fill "
            "quote/P&L unit; otherwise preserve exact valuation failure",
        },
        "pnl_model": {
            "gross_realized": "FIFO disposal proceeds minus exact consumed cost in valuation asset",
            "fee_effect": "separate exact fee quantity; valued only by trusted fee-asset valuation and "
            "subtracted once in net reporting",
            "funding_interest": "unsupported in current execution-capable scope",
            "net_realized": "gross realized minus once-valued fees plus supported funding/interest "
            "effects",
            "unrealized": "projection-only mark value minus remaining exact FIFO cost",
            "capital_flows": "excluded from trading P&L",
            "mark_updates": "never change ledger or realized P&L",
            "executable_projection": "scope-aware rebuild returns gross_realized, once-valued fee_effect, "
            "net_realized and unrealized; incomplete fee/mark conversion returns "
            "MISSING_VALUATION not zero",
            "fee_valuation_time_policy": "trusted valuation context as_of_utc supplied explicitly to the "
            "reporting projection; never current implicit price or zero "
            "fallback",
        },
        "valuation_protocol": {
            "required_fields": [
                "subject_reference",
                "valuation_unit",
                "rate",
                "source_id",
                "observed_at_utc",
                "effective_at_utc",
                "as_of_utc",
                "stale_after_utc",
                "source_fingerprint_sha256",
            ],
            "rate": "positive canonical decimal converted to exact Fraction",
            "path": "explicit ordered acyclic source-scoped edges; deterministic lexicographic "
            "fingerprint tie-break only among policy-authorized paths",
            "missing": "MISSING_VALUATION; never zero",
            "stale": "STALE_VALUATION; never trusted or zero",
            "unsupported_path": "UNSUPPORTED_VALUATION_PATH",
            "completeness": ["COMPLETE", "PARTIALLY_UNVALUED"],
            "trusted_context": "nominal PrevalidatedValuationContext only; raw mapping rejected",
            "freshness": "observed_at_utc <= effective_at_utc <= as_of_utc < stale_after_utc; "
            "stale is derived, never caller-supplied",
            "valuation_unit": "exact AssetReference; no display-code or stablecoin equivalence",
            "source_authority": "CoreAcceptedValuationFactProjection maps source_id to exact "
            "accepted fingerprint/scope/semantics; IDs or self-hashes alone "
            "grant no authority",
            "path_precedence": "enumerate authorized acyclic paths; discard stale paths; choose "
            "fresh winner by ordered edge-fingerprint tuple; otherwise STALE "
            "before UNSUPPORTED cycle before MISSING",
            "reporting_as_of": "one canonical reporting_as_of_utc per context/projection; every "
            "edge as_of_utc must equal it; freshness evaluated against it",
        },
        "equity_nav": {
            "canonical_equation": "sum of each owned available+reserved asset quantity converted by a "
            "complete trusted valuation path, minus explicitly supported "
            "liabilities; classification roles and analytical spot positions are "
            "excluded",
            "spot_rule": "marked owned inventory is counted once; neither position value nor "
            "realized/unrealized classifications are added",
            "capital_metrics": [
                "gross_contributions",
                "gross_withdrawals",
                "net_external_capital",
                "equity_nav",
                "gross_realized_trading_pnl",
                "fee_effect",
                "net_realized_pnl",
                "unrealized_pnl",
            ],
            "historical_capital_valuation": "requires trusted valuation at effective time; today's mark "
            "is not substituted",
            "scope": "one exact workspace_id, portfolio_id and environment; aggregates accounts only "
            "inside that scope; cross-environment total is NON_AUTHORITATIVE and not exposed by "
            "canonical nav",
        },
        "reconciliation_protocol": {
            "snapshot": "observed external fact, never accounting authority",
            "key": [
                "workspace_id",
                "portfolio_id",
                "environment",
                "exchange_account_id",
                "asset_reference",
                "as_of_utc",
            ],
            "outcomes": [
                "MATCH",
                "DRIFT",
                "MISSING_INTERNAL_FACT",
                "MISSING_EXTERNAL_FACT",
                "UNMAPPED_ASSET",
                "UNSUPPORTED",
            ],
            "drift": "RECONCILIATION_DRIFT; zero mutation",
            "correction": "exact reversal of existing accepted source/batch only; inverse "
            "postings derived; generic INCREASE/DECREASE forbidden",
            "correction_targets": [
                "fill",
                "deposit",
                "withdrawal",
                "internal_transfer",
                "capital_reservation",
                "capital_release",
            ],
            "target_idempotency": "exact target tuple may be effectively reversed once; same "
            "correction identity replays, a different correction identity conflicts without mutation",
            "target_aware_validation": "mandatory precommit correction validation before journal/"
            "maps mutation; correction scope exact-matches every target entry; same order/"
            "cardinality/workspace/portfolio/environment/account/asset/role/posting_role/quantity "
            "and opposite direction",
            "observed_fact_fields": [
                "workspace_id",
                "portfolio_id",
                "environment",
                "exchange_account_id",
                "asset_reference",
                "observed_quantity",
                "as_of_utc",
                "source_id",
                "source_fingerprint_sha256",
            ],
            "source_authority": "CoreAcceptedObservedBalanceFactProjection maps source_id "
            "to exact accepted fingerprint/scope/semantics; IDs or "
            "self-hashes alone grant no authority",
            "outcome_rules": {
                "MATCH": "internal history exists and exact projected quantity equals observed",
                "DRIFT": "internal history exists and quantities differ; zero mutation",
                "MISSING_INTERNAL_FACT": "no internal journal history for exact scope/asset",
                "MISSING_EXTERNAL_FACT": "no accepted external observation supplied",
                "UNMAPPED_ASSET": "observed asset mapping is AMBIGUOUS or UNKNOWN",
                "UNSUPPORTED": "authorized observation source declares "
                "unsupported balance semantics",
            },
        },
        "ordering_policy": {
            "journal": "strict append_sequence assigned only after atomic validation",
            "projection": ["append_sequence", "posting_index"],
            "effective_time": "immutable economic metadata, not append order",
            "late_and_backfill": "append new entries; no history rewrite",
            "ties": "append_sequence then posting_index; never container iteration",
        },
        "environment_policy": {
            "core": ["PAPER", "TESTNET", "LIVE"],
            "isolation": "execution-authoritative keys never cross environments",
            "paper": "simulator must emit canonical Full Fill; legacy paper ledger is not "
            "authority",
            "testnet": "trusted venue facts use the same core",
            "live": "first-class accounting target using the same core, while current-edition "
            "execution remains policy-disabled",
            "cross_environment_reporting": "explicit NON_AUTHORITATIVE only and forbidden for "
            "M0.9 execution/risk",
        },
        "m09_outputs": [
            "owned_balances",
            "available_capital",
            "reserved_capital",
            "spot_inventory_exposure",
            "equity_nav",
            "gross_and_net_realized_pnl",
            "unrealized_pnl",
            "valuation_completeness",
            "reconciliation_status",
        ],
        "failure_taxonomy": [
            "MALFORMED_ACCOUNTING_FACT",
            "ACCOUNTING_IDENTITY_CONFLICT",
            "UNBALANCED_ACCOUNTING_BATCH",
            "INVALID_ACCOUNTING_TRANSITION",
            "INSUFFICIENT_AVAILABLE_CAPITAL",
            "RESERVATION_CONFLICT",
            "MISSING_VALUATION",
            "STALE_VALUATION",
            "UNSUPPORTED_VALUATION_PATH",
            "RECONCILIATION_DRIFT",
            "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            "TRUSTED_CONTEXT_FAILURE",
            "CONTRACT_INCONSISTENT",
            "REPLAY_SUCCESS",
        ],
        "contract_inconsistent_scope": "only mutable machine-contract drift, schema corruption or "
        "dependency-attestation corruption",
        "cross_contract_dependencies": [
            {
                "contract": "canonical_domain_vocabulary.json",
                "json_pointer": "/public_trading_environments",
                "content_fingerprint_sha256": "b0114e386bf72439199ec8155d65a57dc7e00cabe79dba805f53221ba7713103",
            },
            {
                "contract": "canonical_domain_vocabulary.json",
                "json_pointer": "/entity_kinds",
                "content_fingerprint_sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1",
            },
            {
                "contract": "canonical_domain_vocabulary.json",
                "json_pointer": "/relationships",
                "content_fingerprint_sha256": "03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9",
            },
            {
                "contract": "canonical_domain_vocabulary.json",
                "json_pointer": "/identifier_policy",
                "content_fingerprint_sha256": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
            },
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/asset_reference_contract",
                "content_fingerprint_sha256": "837e1452a60de230d0ca091a7e2c05308ff41d961499f7800d350d7c9b3682ae",
            },
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/decimal_policy",
                "content_fingerprint_sha256": "e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48",
            },
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/instrument_contract/record_fields",
                "content_fingerprint_sha256": "4c10117c28505bc7b546126690f3f3ef6cbc15c47815a47e016bd8c646b71409",
            },
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/instrument_contract/trusted_history_contract",
                "content_fingerprint_sha256": "87b09322419aac6fcc36f12405931e1cf88cf0630ced72e68b341d5434e10e8f",
            },
            {
                "contract": "exchange_accounts_and_instruments.json",
                "json_pointer": "/instrument_type_registry",
                "content_fingerprint_sha256": "e99ba1e3af1a3b5771add15b7d0ab0659c0b24a1bd855c04c0d5c27f2d2831f8",
            },
            {
                "contract": "strategy_market_data_and_execution_routing.json",
                "json_pointer": "/current_edition_execution_pair_policy",
                "content_fingerprint_sha256": "7dd12317c8dab7bc2d19e751f39800db7895f737a42335df1e22927d465ec0b9",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/fill_contract",
                "content_fingerprint_sha256": "1846393d14f684fc2462eb12b5d92f9c18cfa8a0163b2a2f41fe88c913324d1b",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/command_registry/SUBMIT_ORDER",
                "content_fingerprint_sha256": "864bc27bf55228d08b6592f2042a3f9a1f447eae661382bde0b380602369d748",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/event_contract",
                "content_fingerprint_sha256": "c63d514a4546161798de2f7f90441821cd71653fba433c3577c6d7b630e4b6b2",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/order_lifecycle",
                "content_fingerprint_sha256": "48a514419aaa0863078e69ffc50c3acd4b37a06d873d257275fb68874cc840dd",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/idempotency_contract",
                "content_fingerprint_sha256": "43ba37d976eb5948970d289cc71cd06da82cb90803f9e5f61cc84109f00e9a62",
            },
            {
                "contract": "commands_events_order_lifecycle_and_idempotency.json",
                "json_pointer": "/closed_request_policy",
                "content_fingerprint_sha256": "f3a523b01cbce2bafabeaad261777e3a97fc4f60db751deee0e77cb912f93e7c",
            },
        ],
        "forbidden": [
            "float",
            "mutable balance source of truth",
            "production persistence",
            "new Posting/Batch/Balance/Position/PnL durable entity",
            "symbol parsing",
            "global asset aliases",
            "implicit stablecoin parity",
            "LIVE enablement",
            "risk approval",
            "ExecutionLease",
            "venue snapshot overwrite",
        ],
        "closure_conditions": "all executable M0.8 reference invariants and direct M0.2/M0.5/M0.6/M0.7 dependency "
        "attestations pass",
        "accounting_economic_fact_schema_registry": {
            "deposit": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "exchange_account_id",
                    "asset_reference",
                    "quantity",
                    "capital_flow_kind",
                    "basis_valuation_unit",
                    "unit_cost_basis",
                ],
                "constraints": "EXTERNAL_CONTRIBUTION only; "
                "positive exact basis per unit in "
                "explicit valuation unit; creates "
                "FIFO inventory without trading "
                "P&L",
            },
            "withdrawal": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "exchange_account_id",
                    "asset_reference",
                    "quantity",
                    "capital_flow_kind",
                    "basis_valuation_unit",
                    "unit_cost_basis",
                ],
                "constraints": "EXTERNAL_WITHDRAWAL only; "
                "consumes FIFO inventory/basis "
                "without trading P&L; cannot "
                "exceed available",
            },
            "internal_transfer": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "source_exchange_account_id",
                    "source_asset_reference",
                    "source_quantity",
                    "destination_environment",
                    "destination_exchange_account_id",
                    "destination_asset_reference",
                    "destination_quantity",
                ],
                "constraints": "one Portfolio; source "
                "environment equals "
                "destination_environment; "
                "exact equal "
                "AssetReference and "
                "quantity; four postings; "
                "no P&L",
            },
            "capital_reservation": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "exchange_account_id",
                    "asset_reference",
                    "quantity",
                    "order_id",
                    "command_id",
                ],
                "constraints": "accepted M0.7 SUBMIT_ORDER binds command/order/scope/account/"
                "instrument/route and idempotency_key == command_id; separately accepted M0.8 fact "
                "supplies exact reservation asset_reference/quantity; M0.7 does not supply or "
                "approve reservation economics; not M0.9 risk approval",
            },
            "capital_release": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "exchange_account_id",
                    "order_id",
                    "terminal_state",
                ],
                "constraints": "accepted exact M0.7 terminal event context and event "
                "fingerprint/version/history bind order and full scope including account; releases "
                "exact remainder once",
            },
            "reconciliation_correction": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "reason",
                    "target_source_type",
                    "target_accounting_source_identity",
                    "target_source_fingerprint_sha256",
                    "target_batch_fingerprint_sha256",
                ],
                "constraints": "exact accepted "
                "target "
                "source/fingerprint/batch "
                "required; "
                "derives inverse "
                "postings; "
                "generic quantity "
                "adjustment "
                "forbidden",
            },
            "fee": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                    "exchange_account_id",
                    "asset_reference",
                    "quantity",
                    "reason",
                ],
                "constraints": "UNSUPPORTED_ACCOUNTING_SEMANTICS; no "
                "canonical independent-fee authority "
                "exists; M0.7 Full Fill is sole "
                "trade-fee authority",
            },
            "funding": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                ],
                "constraints": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
            "interest": {
                "exact_fields": [
                    "audit_event_id",
                    "source_type",
                    "workspace_id",
                    "portfolio_id",
                    "environment",
                    "effective_at_utc",
                    "provenance",
                    "source_fingerprint_sha256",
                ],
                "constraints": "UNSUPPORTED_ACCOUNTING_SEMANTICS",
            },
        },
        "m07_fill_boundary": {
            "context": "nominal M07PrevalidatedAcceptedFillContext returned by composite "
            "final-M0.7 resolution; raw mappings and subsets rejected",
            "exact_full_fill_fields": [
                "fill_id",
                "order_id",
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "instrument_metadata_version",
                "execution_route_id",
                "venue_trade_id",
                "side",
                "executed_quantity",
                "execution_price",
                "executed_at_utc",
                "fee_kind",
                "fee_quantity",
                "fee_asset_reference",
                "fill_fingerprint_sha256",
            ],
            "fingerprint": "canonical M0.7 fill_fingerprint_sha256 over "
            "/fill_contract/fingerprint/input_fields; M0.8 accounting source "
            "fingerprint equals it",
            "composition": [
                "exact Full Fill validation",
                "canonical Fill fingerprint",
                "resolved historical Instrument",
                "Core-owned accepted Fill history projection",
            ],
            "preserved_scope": [
                "order_id",
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "instrument_metadata_version",
                "execution_route_id",
                "venue_trade_id",
            ],
        },
        "source_posting_matrix": {
            "fill": {
                "allowed": [
                    "OWNED_AVAILABLE:ASSET_RECEIVED:DEBIT",
                    "OWNED_AVAILABLE:ASSET_PAID:CREDIT",
                    "OWNED_RESERVED:ASSET_PAID:CREDIT",
                    "TRADE_CLEARING:TRADE_COUNTERPART:DEBIT_OR_CREDIT",
                    "FEE_EXPENSE:FEE_CLASSIFIED:DEBIT",
                    "OWNED_AVAILABLE:FEE_PAID:CREDIT",
                    "REALIZED_PNL_CLASSIFICATION:PNL_CLASSIFIED:DEBIT_OR_CREDIT",
                    "TRADE_CLEARING:PNL_COUNTERPART:DEBIT_OR_CREDIT",
                    "OWNED_RESERVED:FEE_PAID:CREDIT",
                ]
            },
            "deposit": {
                "allowed": [
                    "OWNED_AVAILABLE:ASSET_RECEIVED:DEBIT",
                    "EXTERNAL_CAPITAL:CAPITAL_CLASSIFIED:CREDIT",
                ]
            },
            "withdrawal": {
                "allowed": [
                    "EXTERNAL_CAPITAL:CAPITAL_CLASSIFIED:DEBIT",
                    "OWNED_AVAILABLE:ASSET_PAID:CREDIT",
                ]
            },
            "internal_transfer": {
                "allowed": [
                    "TRANSFER_CLEARING:TRANSFER_CLEARING_SOURCE:DEBIT",
                    "OWNED_AVAILABLE:TRANSFER_SOURCE:CREDIT",
                    "OWNED_AVAILABLE:TRANSFER_DESTINATION:DEBIT",
                    "TRANSFER_CLEARING:TRANSFER_CLEARING_DESTINATION:CREDIT",
                ]
            },
            "capital_reservation": {
                "allowed": [
                    "OWNED_RESERVED:RESERVE_HELD:DEBIT",
                    "OWNED_AVAILABLE:RESERVE_AVAILABLE:CREDIT",
                ]
            },
            "capital_release": {
                "allowed": [
                    "OWNED_AVAILABLE:RESERVE_AVAILABLE:DEBIT",
                    "OWNED_RESERVED:RESERVE_HELD:CREDIT",
                ]
            },
            "reconciliation_correction": {
                "allowed": "exact direction-inverted target batch tuples only"
            },
            "fee": "FORBIDDEN",
            "funding": "FORBIDDEN",
            "interest": "FORBIDDEN",
            "realized_pnl": "FORBIDDEN",
        },
        "rebuild_protocol": {
            "authority": [
                "immutable LedgerEntry journal",
                "CoreAcceptedAccountingSourceHistory records exact-bound by source "
                "identity/fingerprint/rule/batch",
                "one shared trusted valuation context when required",
            ],
            "mutable_cache": "forbidden as authority; cache removal/change cannot affect "
            "projection",
            "expected_batch": "rederive source fingerprint, rule, contiguous postings and complete "
            "batch from canonical immutable source context plus preceding reconstructed accounting "
            "state; compare exact entries and recomputed batch fingerprint",
            "inventory_effects": {
                "deposit": "adds FIFO lot with exact accepted effective-time unit basis",
                "withdrawal": "consumes FIFO basis; no trading P&L",
                "internal_transfer": "moves exact FIFO lot slices/basis "
                "account-to-account; Portfolio basis "
                "unchanged",
                "fill_fee": "consumes exact fee-asset inventory basis; missing "
                "inventory fails closed",
                "reconciliation_reversal": "removes exact accepted target source effect through "
                "direction-inverse compensation only",
            },
        },
        "non_fill_authority": {
            "context": "nominal CoreAcceptedAccountingFactContext",
            "membership": "Core-owned accepted projection maps audit_event_id to exact "
            "source_type, fingerprint and scope",
            "self_hash": "integrity only; raw/self-hashed facts rejected",
        },
        "m07_authority_boundary": {
            "model": "direct field interpretation with exact dependency attestation",
            "submit_order_consumed_fields": [
                "command_id",
                "operation_type",
                "authority_context_id",
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "strategy_instance_id",
                "source_type",
                "instrument_id",
                "execution_route_id",
                "correlation_id",
                "causation_id",
                "idempotency_key",
                "order_intent_id",
                "order_id",
                "side",
                "order_type",
                "quantity",
                "limit_price",
                "time_in_force",
                "expire_at_utc",
            ],
            "reservation_economics": "asset_reference and reservation quantity belong only to M0.8 "
            "AccountingEconomicFact; M0.7 does not provide or approve them; M0.8 reservation is "
            "accounting state, not M0.9 risk authority",
            "terminal_event_mapping": {
                "ORDER_REJECTED": "REJECTED",
                "ORDER_FILLED": "FILLED",
                "ORDER_CANCEL_CONFIRMED": "CANCELLED",
                "ORDER_REPLACE_CONFIRMED": "REPLACED",
                "ORDER_EXPIRED": "EXPIRED",
            },
            "accepted_history": "reservation retains exact accepted AccountingEconomicFact plus "
            "sealed nominal accepted SUBMIT_ORDER context; release retains exact fact plus sealed "
            "M0.7 lifecycle-ingestion proof binding event identity/fingerprint/order/version/type/"
            "terminal target/legal predecessor/previous contiguous version; rebuild rejects missing "
            "evidence",
            "terminal_event_attestation": "exact M0.7 envelope schemas including nullable "
            "command_id/causation_id and non-bool positive aggregate_version; exact terminal "
            "safe_payload field/value schemas; canonical event fingerprint; sealed accepted "
            "lifecycle proof with legal predecessor and contiguous previous version",
        },
    }
)

SEMANTIC_ROOTS = tuple(key for key in EXPECTED_PROTOCOLS if key != "cross_contract_dependencies")


def validate_contract(contract: Mapping[str, Any], upstream: Mapping[str, Any] = CANONICAL) -> str:
    try:
        if set(contract) != set(CONTRACT) or contract["status"] != "closed":
            return "CONTRACT_INCONSISTENT"
        for root in SEMANTIC_ROOTS:
            if freeze(contract[root]) != EXPECTED_PROTOCOLS[root]:
                return "CONTRACT_INCONSISTENT"
        dependencies = contract["cross_contract_dependencies"]
        if freeze(dependencies) != EXPECTED_PROTOCOLS["cross_contract_dependencies"]:
            return "CONTRACT_INCONSISTENT"
        for dependency in dependencies:
            target = pointer(upstream[dependency["contract"]], dependency["json_pointer"])
            if digest(target) != dependency["content_fingerprint_sha256"]:
                return "CONTRACT_INCONSISTENT"
        # Stable semantic extraction from collection roots, never array-index authority.
        entities = {
            item["canonical_name"]: item
            for item in upstream["canonical_domain_vocabulary.json"]["entity_kinds"]
        }
        if (
            entities["Portfolio"]["id_prefix"] != "port"
            or entities["LedgerEntry"]["id_prefix"] != "led"
        ):
            return "CONTRACT_INCONSISTENT"
        if set(entities["LedgerEntry"]["allowed_source_event_types"]) != set(
            EXPECTED_PROTOCOLS["source_registry"]
        ):
            return "CONTRACT_INCONSISTENT"
        relationships = {
            (item["from"], item["to"], item["cardinality"])
            for item in upstream["canonical_domain_vocabulary.json"]["relationships"]
        }
        if ("Portfolio", "LedgerEntry", "one_to_many") not in relationships:
            return "CONTRACT_INCONSISTENT"
    except (KeyError, TypeError, ValueError):
        return "CONTRACT_INCONSISTENT"
    return "VALID"


@dataclass(frozen=True)
class AssetReference:
    venue_asset_code: str
    canonical_display_code: str
    asset_namespace: str
    mapping_status: str

    @classmethod
    def trusted(cls, value: Mapping[str, Any]) -> AssetReference:
        fields = {"venue_asset_code", "canonical_display_code", "asset_namespace", "mapping_status"}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if any(type(value[name]) is not str or not value[name] for name in fields):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if value["mapping_status"] not in {"EXACT", "EXPLICIT_ALIAS"}:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        return cls(**cast(dict[str, str], value))

    def key(self) -> tuple[str, str, str, str]:
        return (
            self.venue_asset_code,
            self.canonical_display_code,
            self.asset_namespace,
            self.mapping_status,
        )

    def object(self) -> dict[str, str]:
        return asdict(self)


def decimal(value: Any, *, positive: bool = True) -> Fraction:
    if type(value) is not str or not DECIMAL_RE.fullmatch(value) or (positive and value == "0"):
        raise ValueError("MALFORMED_ACCOUNTING_FACT")
    return Fraction(value)


def timestamp(value: Any) -> datetime:
    if type(value) is not str or not TIME_RE.fullmatch(value):
        raise ValueError("MALFORMED_ACCOUNTING_FACT")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("MALFORMED_ACCOUNTING_FACT") from exc
    if parsed.tzinfo != UTC:
        raise ValueError("MALFORMED_ACCOUNTING_FACT")
    return parsed


def valid_id(value: Any, prefix: str) -> bool:
    return (
        type(value) is str and value.startswith(prefix + "_") and ID_RE.fullmatch(value) is not None
    )


@dataclass(frozen=True)
class Scope:
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str

    def validate(self) -> None:
        if (
            not valid_id(self.workspace_id, "ws")
            or not valid_id(self.portfolio_id, "port")
            or not valid_id(self.exchange_account_id, "xacc")
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if self.environment not in {"PAPER", "TESTNET", "LIVE"}:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")


@dataclass(frozen=True)
class PostingProjection:
    scope: Scope
    asset_reference: AssetReference
    account_role: str
    direction: str
    quantity: str
    posting_role: str

    def validate(self) -> None:
        self.scope.validate()
        if self.account_role not in EXPECTED_PROTOCOLS["account_roles"]:
            raise ValueError("MALFORMED_ACCOUNTING_FACT")
        if self.direction not in {"DEBIT", "CREDIT"}:
            raise ValueError("MALFORMED_ACCOUNTING_FACT")
        decimal(self.quantity)
        if (
            self.posting_role
            not in EXPECTED_PROTOCOLS["ledger_entry_schema"]["posting_role_registry"]
        ):
            raise ValueError("MALFORMED_ACCOUNTING_FACT")

    def canonical_projection(self, index: int) -> dict[str, Any]:
        return {
            "workspace_id": self.scope.workspace_id,
            "portfolio_id": self.scope.portfolio_id,
            "environment": self.scope.environment,
            "exchange_account_id": self.scope.exchange_account_id,
            "asset_reference": self.asset_reference.object(),
            "account_role": self.account_role,
            "direction": self.direction,
            "quantity": self.quantity,
            "posting_index": index,
            "posting_role": self.posting_role,
        }


@dataclass(frozen=True)
class LedgerEntry:
    ledger_entry_id: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str | None
    strategy_instance_id: str | None
    asset_reference: dict[str, str]
    account_role: str
    direction: str
    quantity: str
    source_type: str
    accounting_source_identity: str
    accounting_source_fingerprint_sha256: str
    accounting_rule_version: str
    posting_index: int
    posting_role: str
    batch_fingerprint_sha256: str
    effective_at_utc: str
    append_sequence: int
    order_id: str | None
    fill_id: str | None
    audit_event_id: str | None
    correction_reason: str | None


def validate_ledger_entry(value: Mapping[str, Any], previous_sequence: int) -> str:
    schema = EXPECTED_PROTOCOLS["ledger_entry_schema"]
    if type(value) is not dict or set(value) != set(schema["exact_fields"]):
        return "MALFORMED_ACCOUNTING_FACT"
    try:
        if (
            not valid_id(value["ledger_entry_id"], "led")
            or not valid_id(value["workspace_id"], "ws")
            or not valid_id(value["portfolio_id"], "port")
        ):
            raise ValueError
        if value["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
            raise ValueError
        if value["exchange_account_id"] is not None and not valid_id(
            value["exchange_account_id"], "xacc"
        ):
            raise ValueError
        if value["strategy_instance_id"] is not None and not valid_id(
            value["strategy_instance_id"], "sinst"
        ):
            raise ValueError
        AssetReference.trusted(value["asset_reference"])
        if value["account_role"] not in EXPECTED_PROTOCOLS["account_roles"] or value[
            "direction"
        ] not in {"DEBIT", "CREDIT"}:
            raise ValueError
        decimal(value["quantity"])
        if value["source_type"] not in EXPECTED_PROTOCOLS["source_registry"]:
            raise ValueError
        if (
            not HEX_RE.fullmatch(value["accounting_source_fingerprint_sha256"] or "")
            or value["accounting_rule_version"] != RULE
        ):
            raise ValueError
        if type(value["posting_index"]) is not int or value["posting_index"] < 0:
            raise ValueError
        if value["posting_role"] not in schema["posting_role_registry"] or not HEX_RE.fullmatch(
            value["batch_fingerprint_sha256"] or ""
        ):
            raise ValueError
        timestamp(value["effective_at_utc"])
        if (
            type(value["append_sequence"]) is not int
            or value["append_sequence"] != previous_sequence + 1
        ):
            raise ValueError
        for name, prefix in (("order_id", "ord"), ("fill_id", "fill"), ("audit_event_id", "evt")):
            if value[name] is not None and not valid_id(value[name], prefix):
                raise ValueError
        if value["correction_reason"] is not None and (
            type(value["correction_reason"]) is not str or not value["correction_reason"]
        ):
            raise ValueError
        if value["source_type"] == "fill":
            if (
                value["fill_id"] is None
                or value["accounting_source_identity"] != value["fill_id"]
                or value["order_id"] is None
                or value["audit_event_id"] is not None
                or value["correction_reason"] is not None
            ):
                raise ValueError
        elif value["source_type"] == "realized_pnl":
            raise ValueError
        elif (
            value["audit_event_id"] is None
            or value["accounting_source_identity"] != value["audit_event_id"]
            or value["fill_id"] is not None
            or value["order_id"] is not None
        ):
            raise ValueError
        if (
            value["source_type"] == "reconciliation_correction"
            and value["correction_reason"] is None
        ):
            raise ValueError
        forbidden_sources = {"fee", "funding", "interest", "realized_pnl"}
        if value["source_type"] in forbidden_sources:
            raise ValueError
        matrix = EXPECTED_PROTOCOLS["source_posting_matrix"][value["source_type"]]["allowed"]
        if value["source_type"] != "reconciliation_correction":
            candidate = f"{value['account_role']}:{value['posting_role']}:{value['direction']}"
            flexible = f"{value['account_role']}:{value['posting_role']}:DEBIT_OR_CREDIT"
            if candidate not in matrix and flexible not in matrix:
                raise ValueError
        elif (
            value["posting_role"]
            not in EXPECTED_PROTOCOLS["ledger_entry_schema"]["posting_role_registry"]
        ):
            raise ValueError
    except (TypeError, ValueError):
        return "MALFORMED_ACCOUNTING_FACT"
    return "VALID"


def validate_correction_batch(
    correction: AccountingEconomicFact,
    target_entries: tuple[LedgerEntry, ...],
    correction_entries: tuple[LedgerEntry, ...],
) -> str:
    try:
        payload = correction.payload
        if payload["source_type"] != "reconciliation_correction" or (
            payload["target_source_type"],
            payload["target_accounting_source_identity"],
            payload["target_source_fingerprint_sha256"],
            payload["target_batch_fingerprint_sha256"],
        ) != (
            target_entries[0].source_type,
            target_entries[0].accounting_source_identity,
            target_entries[0].accounting_source_fingerprint_sha256,
            target_entries[0].batch_fingerprint_sha256,
        ):
            raise ValueError
        if len(target_entries) != len(correction_entries) or not target_entries:
            raise ValueError
        scope = (payload["workspace_id"], payload["portfolio_id"], payload["environment"])
        for index, (target, inverse) in enumerate(
            zip(target_entries, correction_entries, strict=True)
        ):
            if (
                (target.workspace_id, target.portfolio_id, target.environment) != scope
                or (inverse.workspace_id, inverse.portfolio_id, inverse.environment) != scope
                or inverse.source_type != "reconciliation_correction"
                or inverse.posting_index != index
                or inverse.exchange_account_id != target.exchange_account_id
                or inverse.asset_reference != target.asset_reference
                or inverse.account_role != target.account_role
                or inverse.posting_role != target.posting_role
                or inverse.quantity != target.quantity
                or inverse.direction != ("CREDIT" if target.direction == "DEBIT" else "DEBIT")
            ):
                raise ValueError
    except (IndexError, KeyError, TypeError, ValueError):
        return "MALFORMED_ACCOUNTING_FACT"
    return "VALID"


@dataclass(frozen=True)
class CoreAcceptedFillProjection:
    order_id: str
    accepted_fill_ids: tuple[str, ...]


@dataclass(frozen=True)
class M07PrevalidatedAcceptedFillContext:
    """Nominal upstream attestation returned only by composite final-M0.7 resolution."""

    fill: Mapping[str, Any]
    historical_instrument: Mapping[str, Any]
    accepted_history: CoreAcceptedFillProjection
    source_fingerprint: str


def canonical_m07_fill_fingerprint(fill: Mapping[str, Any]) -> str:
    contract = CANONICAL["commands_events_order_lifecycle_and_idempotency.json"]["fill_contract"]
    projected = {name: fill[name] for name in contract["fingerprint"]["input_fields"]}
    return digest(projected)


def attest_m07_accepted_fill(
    fill: dict[str, Any],
    instrument: dict[str, Any],
    accepted_history: CoreAcceptedFillProjection,
) -> M07PrevalidatedAcceptedFillContext:
    """Reference fixture for an already successful M0.7 composite trusted resolver."""

    contract = CANONICAL["commands_events_order_lifecycle_and_idempotency.json"]["fill_contract"]
    instrument_fields = {
        "instrument_id",
        "metadata_version",
        "instrument_type",
        "workspace_id",
        "environment",
        "exchange_id",
        "base_asset_reference",
        "quote_asset_reference",
    }
    if set(fill) != set(contract["fact_fields"]) or set(instrument) != instrument_fields:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if fill["fill_fingerprint_sha256"] != canonical_m07_fill_fingerprint(fill):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if not isinstance(accepted_history, CoreAcceptedFillProjection):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if (
        accepted_history.order_id != fill["order_id"]
        or fill["fill_id"] not in accepted_history.accepted_fill_ids
        or len(accepted_history.accepted_fill_ids) != len(set(accepted_history.accepted_fill_ids))
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    Scope(
        fill["workspace_id"], fill["portfolio_id"], fill["environment"], fill["exchange_account_id"]
    ).validate()
    for name, prefix in (
        ("fill_id", "fill"),
        ("order_id", "ord"),
        ("instrument_id", "instr"),
        ("execution_route_id", "xroute"),
    ):
        if not valid_id(fill[name], prefix):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if any(
        type(fill[name]) is not str or not fill[name] for name in ("exchange_id", "venue_trade_id")
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if (
        fill["instrument_id"] != instrument["instrument_id"]
        or fill["instrument_metadata_version"] != instrument["metadata_version"]
        or fill["workspace_id"] != instrument["workspace_id"]
        or fill["environment"] != instrument["environment"]
        or fill["exchange_id"] != instrument["exchange_id"]
        or instrument["instrument_type"] != "SPOT_PAIR"
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if fill["side"] not in {"BUY", "SELL"}:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    decimal(fill["executed_quantity"])
    decimal(fill["execution_price"])
    timestamp(fill["executed_at_utc"])
    AssetReference.trusted(instrument["base_asset_reference"])
    AssetReference.trusted(instrument["quote_asset_reference"])
    if fill["fee_kind"] == "NONE":
        if fill["fee_quantity"] != "0" or fill["fee_asset_reference"] is not None:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif fill["fee_kind"] == "CHARGE":
        decimal(fill["fee_quantity"])
        fee_asset = AssetReference.trusted(fill["fee_asset_reference"])
        if fee_asset.asset_namespace != fill["exchange_id"]:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    else:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    return M07PrevalidatedAcceptedFillContext(
        freeze(fill), freeze(instrument), accepted_history, fill["fill_fingerprint_sha256"]
    )


@dataclass(frozen=True)
class AccountingEconomicFact:
    payload: Mapping[str, Any]
    source_fingerprint: str
    upstream_context: (
        M07PrevalidatedAcceptedCommandContext
        | M07PrevalidatedAcceptedTerminalOrderEventContext
        | None
    ) = None


@dataclass(frozen=True)
class CoreAcceptedAccountingFactProjection:
    accepted: Mapping[str, tuple[str, str, str, str, str]]


@dataclass(frozen=True)
class CoreAcceptedCommandProjection:
    accepted: Mapping[str, str]


@dataclass(frozen=True)
class M07PrevalidatedAcceptedCommandContext:
    request: Mapping[str, Any]
    command_fingerprint: str
    authority: CoreAcceptedCommandProjection
    acceptance_seal: object


@dataclass(frozen=True)
class CoreAcceptedOrderEventProjection:
    accepted: Mapping[str, AcceptedTerminalLifecycleProof]


@dataclass(frozen=True)
class AcceptedTerminalLifecycleProof:
    event_fingerprint: str
    order_id: str
    aggregate_version: int
    event_type: str
    terminal_state: str
    predecessor_state: str
    previous_aggregate_version: int
    lifecycle_validation_seal: object


@dataclass(frozen=True)
class M07PrevalidatedAcceptedTerminalOrderEventContext:
    event: Mapping[str, Any]
    terminal_state: str
    authority: CoreAcceptedOrderEventProjection
    acceptance_seal: object


_M07_CORE_ACCEPTANCE_SEAL = object()
_M07_LIFECYCLE_VALIDATION_SEAL = object()


M07_SUBMIT_ORDER_FIELDS = frozenset(
    {
        "command_id",
        "operation_type",
        "authority_context_id",
        "environment",
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "strategy_instance_id",
        "source_type",
        "instrument_id",
        "execution_route_id",
        "correlation_id",
        "causation_id",
        "idempotency_key",
        "order_intent_id",
        "order_id",
        "side",
        "order_type",
        "quantity",
        "limit_price",
        "time_in_force",
        "expire_at_utc",
    }
)
M07_EVENT_FIELDS = frozenset(
    {
        "audit_event_id",
        "event_type",
        "order_id",
        "aggregate_version",
        "correlation_id",
        "causation_id",
        "command_id",
        "environment",
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "exchange_id",
        "instrument_id",
        "execution_route_id",
        "occurred_at_utc",
        "safe_payload",
        "event_fingerprint_sha256",
    }
)
M07_TERMINAL_EVENT_TARGETS = {
    "ORDER_REJECTED": "REJECTED",
    "ORDER_FILLED": "FILLED",
    "ORDER_CANCEL_CONFIRMED": "CANCELLED",
    "ORDER_REPLACE_CONFIRMED": "REPLACED",
    "ORDER_EXPIRED": "EXPIRED",
}


def m07_command_fingerprint(request: Mapping[str, Any]) -> str:
    return digest({key: value for key, value in request.items() if key != "correlation_id"})


def attest_m07_accepted_submit_order(
    request: dict[str, Any], authority: CoreAcceptedCommandProjection
) -> M07PrevalidatedAcceptedCommandContext:
    if set(request) != M07_SUBMIT_ORDER_FIELDS or request["operation_type"] != "SUBMIT_ORDER":
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    for field, prefix in {
        "command_id": "cmd",
        "authority_context_id": "authctx",
        "workspace_id": "ws",
        "portfolio_id": "port",
        "exchange_account_id": "xacc",
        "instrument_id": "instr",
        "execution_route_id": "xroute",
        "correlation_id": "corr",
        "idempotency_key": "cmd",
        "order_intent_id": "oint",
        "order_id": "ord",
    }.items():
        if not valid_id(request[field], prefix):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if request["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if request["idempotency_key"] != request["command_id"]:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    decimal(request["quantity"])
    if (
        request["side"] not in {"BUY", "SELL"}
        or request["order_type"] not in {"MARKET", "LIMIT"}
        or request["time_in_force"] not in {"GTC", "IOC", "FOK", "GTD"}
        or request["source_type"] not in {"STRATEGY_INSTANCE", "OPERATOR", "SYSTEM_RECONCILIATION"}
        or (request["order_type"] == "MARKET") != (request["limit_price"] is None)
        or (request["time_in_force"] == "GTD") != (request["expire_at_utc"] is not None)
        or (request["source_type"] == "STRATEGY_INSTANCE")
        != (request["strategy_instance_id"] is not None)
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if request["limit_price"] is not None:
        decimal(request["limit_price"])
    if request["expire_at_utc"] is not None:
        timestamp(request["expire_at_utc"])
    for field, prefix in (("strategy_instance_id", "sinst"), ("causation_id", "cause")):
        if request[field] is not None and not valid_id(request[field], prefix):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    fingerprint = m07_command_fingerprint(request)
    if authority.accepted.get(request["command_id"]) != fingerprint:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    return M07PrevalidatedAcceptedCommandContext(
        freeze(request), fingerprint, authority, _M07_CORE_ACCEPTANCE_SEAL
    )


def m07_event_fingerprint(event: Mapping[str, Any]) -> str:
    return digest({key: value for key, value in event.items() if key != "event_fingerprint_sha256"})


def accepted_terminal_lifecycle_proof(
    event: Mapping[str, Any], predecessor_state: str
) -> AcceptedTerminalLifecycleProof:
    """Test fixture for the opaque result of prior canonical M0.7 lifecycle ingestion."""
    return AcceptedTerminalLifecycleProof(
        event["event_fingerprint_sha256"],
        event["order_id"],
        event["aggregate_version"],
        event["event_type"],
        M07_TERMINAL_EVENT_TARGETS[event["event_type"]],
        predecessor_state,
        event["aggregate_version"] - 1,
        _M07_LIFECYCLE_VALIDATION_SEAL,
    )


def attest_m07_accepted_terminal_event(
    event: dict[str, Any], authority: CoreAcceptedOrderEventProjection
) -> M07PrevalidatedAcceptedTerminalOrderEventContext:
    if set(event) != M07_EVENT_FIELDS or event["event_type"] not in M07_TERMINAL_EVENT_TARGETS:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    expected_payload_fields = {
        "ORDER_REJECTED": {"reason_code"},
        "ORDER_FILLED": {"fill_id", "venue_trade_id", "cumulative_executed_quantity"},
        "ORDER_CANCEL_CONFIRMED": {"venue_order_id"},
        "ORDER_REPLACE_CONFIRMED": {"replacement_order_id", "venue_order_id"},
        "ORDER_EXPIRED": {"venue_order_id"},
    }
    if (
        not isinstance(event["safe_payload"], Mapping)
        or set(event["safe_payload"]) != (expected_payload_fields[event["event_type"]])
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    payload = event["safe_payload"]
    event_type = event["event_type"]
    if event_type == "ORDER_REJECTED":
        if type(payload["reason_code"]) is not str or not payload["reason_code"]:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif event_type == "ORDER_FILLED":
        if (
            not valid_id(payload["fill_id"], "fill")
            or type(payload["venue_trade_id"]) is not str
            or not payload["venue_trade_id"]
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if (
            type(payload["cumulative_executed_quantity"]) is not str
            or not DECIMAL_RE.fullmatch(payload["cumulative_executed_quantity"])
            or payload["cumulative_executed_quantity"] == "0"
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif event_type == "ORDER_REPLACE_CONFIRMED":
        if not valid_id(payload["replacement_order_id"], "ord"):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if type(payload["venue_order_id"]) is not str or not payload["venue_order_id"]:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif type(payload["venue_order_id"]) is not str or not payload["venue_order_id"]:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if event["event_fingerprint_sha256"] != m07_event_fingerprint(event):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    Scope(
        event["workspace_id"],
        event["portfolio_id"],
        event["environment"],
        event["exchange_account_id"],
    ).validate()
    if type(event["aggregate_version"]) is not int or event["aggregate_version"] <= 0:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    for field, prefix in {
        "audit_event_id": "evt",
        "order_id": "ord",
        "correlation_id": "corr",
        "instrument_id": "instr",
        "execution_route_id": "xroute",
    }.items():
        if not valid_id(event[field], prefix):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    for field, prefix in (("causation_id", "cause"), ("command_id", "cmd")):
        if event[field] is not None and not valid_id(event[field], prefix):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if type(event["exchange_id"]) is not str or not event["exchange_id"]:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    proof = authority.accepted.get(event["audit_event_id"])
    legal_predecessors = {
        "ORDER_REJECTED": {"SUBMISSION_PENDING", "RECONCILIATION_REQUIRED"},
        "ORDER_FILLED": {
            "SUBMISSION_PENDING",
            "ACKNOWLEDGED",
            "PARTIALLY_FILLED",
            "CANCEL_PENDING",
            "REPLACE_PENDING",
            "RECONCILIATION_REQUIRED",
        },
        "ORDER_CANCEL_CONFIRMED": {"CANCEL_PENDING", "RECONCILIATION_REQUIRED"},
        "ORDER_REPLACE_CONFIRMED": {"REPLACE_PENDING", "RECONCILIATION_REQUIRED"},
        "ORDER_EXPIRED": {
            "SUBMISSION_PENDING",
            "ACKNOWLEDGED",
            "PARTIALLY_FILLED",
            "CANCEL_PENDING",
            "REPLACE_PENDING",
            "RECONCILIATION_REQUIRED",
        },
    }
    if (
        not isinstance(proof, AcceptedTerminalLifecycleProof)
        or proof.lifecycle_validation_seal is not _M07_LIFECYCLE_VALIDATION_SEAL
        or proof.event_fingerprint != event["event_fingerprint_sha256"]
        or proof.order_id != event["order_id"]
        or proof.aggregate_version != event["aggregate_version"]
        or proof.event_type != event_type
        or proof.terminal_state != M07_TERMINAL_EVENT_TARGETS[event_type]
        or proof.predecessor_state not in legal_predecessors[event_type]
        or proof.previous_aggregate_version + 1 != event["aggregate_version"]
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    timestamp(event["occurred_at_utc"])
    return M07PrevalidatedAcceptedTerminalOrderEventContext(
        freeze(event),
        M07_TERMINAL_EVENT_TARGETS[event["event_type"]],
        authority,
        _M07_CORE_ACCEPTANCE_SEAL,
    )


def preexisting_accounting_projection(
    accepted_records: tuple[Mapping[str, Any], ...],
) -> CoreAcceptedAccountingFactProjection:
    return CoreAcceptedAccountingFactProjection(
        freeze(
            {
                payload["audit_event_id"]: (
                    payload["source_type"],
                    payload["source_fingerprint_sha256"],
                    payload["workspace_id"],
                    payload["portfolio_id"],
                    payload["environment"],
                )
                for payload in accepted_records
            }
        )
    )


def economic_fact(
    payload: dict[str, Any],
    authority: CoreAcceptedAccountingFactProjection | None = None,
    upstream_context: (
        M07PrevalidatedAcceptedCommandContext
        | M07PrevalidatedAcceptedTerminalOrderEventContext
        | None
    ) = None,
) -> AccountingEconomicFact:
    source = payload.get("source_type")
    registry = EXPECTED_PROTOCOLS["accounting_economic_fact_schema_registry"]
    unsupported = {"funding", "interest", "fee", "realized_pnl"}
    if (
        source not in registry
        or source in unsupported
        or set(payload) != set(registry[source]["exact_fields"])
    ):
        raise ValueError(
            "UNSUPPORTED_ACCOUNTING_SEMANTICS"
            if source in unsupported
            else "TRUSTED_CONTEXT_FAILURE"
        )
    if not isinstance(authority, CoreAcceptedAccountingFactProjection):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    expected_membership = (
        source,
        payload["source_fingerprint_sha256"],
        payload["workspace_id"],
        payload["portfolio_id"],
        payload["environment"],
    )
    if authority.accepted.get(payload["audit_event_id"]) != expected_membership:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if source not in {"capital_release", "reconciliation_correction"}:
        Scope(
            payload["workspace_id"],
            payload["portfolio_id"],
            payload["environment"],
            cast(
                str, payload.get("exchange_account_id", payload.get("source_exchange_account_id"))
            ),
        ).validate()
    elif payload["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if (
        not valid_id(payload["audit_event_id"], "evt")
        or type(payload["provenance"]) is not str
        or not payload["provenance"]
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    timestamp(payload["effective_at_utc"])
    supplied = payload["source_fingerprint_sha256"]
    projected = {key: value for key, value in payload.items() if key != "source_fingerprint_sha256"}
    if supplied != digest(projected):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    if source in {"deposit", "withdrawal"}:
        expected = "EXTERNAL_CONTRIBUTION" if source == "deposit" else "EXTERNAL_WITHDRAWAL"
        if payload["capital_flow_kind"] != expected:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        AssetReference.trusted(payload["asset_reference"])
        AssetReference.trusted(payload["basis_valuation_unit"])
        decimal(payload["quantity"])
        decimal(payload["unit_cost_basis"])
    elif source == "capital_reservation":
        if not valid_id(payload["order_id"], "ord") or not valid_id(payload["command_id"], "cmd"):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        AssetReference.trusted(payload["asset_reference"])
        decimal(payload["quantity"])
        if not isinstance(upstream_context, M07PrevalidatedAcceptedCommandContext) or (
            upstream_context.acceptance_seal is not _M07_CORE_ACCEPTANCE_SEAL
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        request = upstream_context.request
        if (
            request["command_id"] != payload["command_id"]
            or request["order_id"] != payload["order_id"]
            or request["workspace_id"] != payload["workspace_id"]
            or request["portfolio_id"] != payload["portfolio_id"]
            or request["environment"] != payload["environment"]
            or request["exchange_account_id"] != payload["exchange_account_id"]
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif source == "capital_release":
        if not valid_id(payload["order_id"], "ord") or payload["terminal_state"] not in {
            "REJECTED",
            "FILLED",
            "CANCELLED",
            "EXPIRED",
            "REPLACED",
        }:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if not isinstance(upstream_context, M07PrevalidatedAcceptedTerminalOrderEventContext) or (
            upstream_context.acceptance_seal is not _M07_CORE_ACCEPTANCE_SEAL
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        event = upstream_context.event
        if (
            event["audit_event_id"] != payload["audit_event_id"]
            or event["order_id"] != payload["order_id"]
            or upstream_context.terminal_state != payload["terminal_state"]
            or event["workspace_id"] != payload["workspace_id"]
            or event["portfolio_id"] != payload["portfolio_id"]
            or event["environment"] != payload["environment"]
            or event["exchange_account_id"] != payload["exchange_account_id"]
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif source == "reconciliation_correction":
        if type(payload["reason"]) is not str or not payload["reason"]:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if payload["target_source_type"] not in EXPECTED_PROTOCOLS["source_registry"]:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if not HEX_RE.fullmatch(
            payload["target_source_fingerprint_sha256"] or ""
        ) or not HEX_RE.fullmatch(payload["target_batch_fingerprint_sha256"] or ""):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
    elif source == "internal_transfer":
        destination = Scope(
            payload["workspace_id"],
            payload["portfolio_id"],
            payload["destination_environment"],
            payload["destination_exchange_account_id"],
        )
        destination.validate()
        source_asset = AssetReference.trusted(payload["source_asset_reference"])
        destination_asset = AssetReference.trusted(payload["destination_asset_reference"])
        source_quantity = decimal(payload["source_quantity"])
        destination_quantity = decimal(payload["destination_quantity"])
        if (
            payload["environment"] != payload["destination_environment"]
            or source_asset != destination_asset
            or source_quantity != destination_quantity
        ):
            raise ValueError("UNSUPPORTED_ACCOUNTING_SEMANTICS")
    return AccountingEconomicFact(freeze(payload), supplied, upstream_context)


@dataclass
class ReservationState:
    order_id: str
    command_id: str
    scope: Scope
    asset: AssetReference
    original: Fraction
    remaining: Fraction
    economics_fingerprint: str


@dataclass(frozen=True)
class AcceptedSourceRecord:
    source_type: str
    identity: str
    fingerprint: str
    batch_fingerprint: str
    first_sequence: int
    context: Any


@dataclass(frozen=True)
class AcceptedFillAccountingContext:
    fill_context: M07PrevalidatedAcceptedFillContext
    historical_valuation: PrevalidatedValuationContext | None


@dataclass(frozen=True)
class InventoryLot:
    quantity: Fraction
    unit_cost_basis: Fraction
    basis_valuation_unit: AssetReference
    source_identity: str
    append_sequence: int
    posting_index: int


@dataclass
class Engine:
    entries: list[LedgerEntry] = field(default_factory=list)
    accepted: dict[tuple[str, str, str], tuple[str, str, tuple[LedgerEntry, ...]]] = field(
        default_factory=dict
    )
    accepted_source_history: dict[tuple[str, str], AcceptedSourceRecord] = field(
        default_factory=dict
    )
    reservations: dict[str, ReservationState] = field(default_factory=dict)
    reservation_commands: dict[str, tuple[str, str]] = field(default_factory=dict)
    correction_candidate_hook: (
        Callable[[tuple[LedgerEntry, ...]], tuple[LedgerEntry, ...]] | None
    ) = None

    @classmethod
    def rebuild(
        cls,
        entries: list[LedgerEntry],
        source_history: Mapping[tuple[str, str], AcceptedSourceRecord],
        *,
        _validate: bool = True,
    ) -> Engine:
        if _validate:
            canonical = cls()
            for key, record in sorted(
                source_history.items(), key=lambda item: item[1].first_sequence
            ):
                if key != (record.source_type, record.identity):
                    raise ValueError("CONTRACT_INCONSISTENT")
                outcome = (
                    canonical.account_trusted_fill(
                        cast(AcceptedFillAccountingContext, record.context).fill_context,
                        cast(AcceptedFillAccountingContext, record.context).historical_valuation,
                    )
                    if record.source_type == "fill"
                    else canonical.account_fact(record.context)
                )
                if outcome != "ACCEPTED":
                    raise ValueError("CONTRACT_INCONSISTENT")
                derived = canonical.accepted[(record.source_type, record.identity, RULE)][2]
                presented = tuple(
                    entry
                    for entry in entries
                    if entry.source_type == record.source_type
                    and entry.accounting_source_identity == record.identity
                )
                if (
                    record.fingerprint
                    != canonical.accepted[(record.source_type, record.identity, RULE)][0]
                    or record.batch_fingerprint
                    != canonical.accepted[(record.source_type, record.identity, RULE)][1]
                    or record.first_sequence != derived[0].append_sequence
                    or presented != derived
                ):
                    raise ValueError("CONTRACT_INCONSISTENT")
            if len(entries) != len(canonical.entries):
                raise ValueError("CONTRACT_INCONSISTENT")
            return canonical
        rebuilt = cls(entries=list(entries), accepted_source_history=dict(source_history))
        for record in source_history.values():
            journal = tuple(
                entry
                for entry in entries
                if entry.source_type == record.source_type
                and entry.accounting_source_identity == record.identity
            )
            rebuilt.accepted[(record.source_type, record.identity, RULE)] = (
                record.fingerprint,
                record.batch_fingerprint,
                journal,
            )
        reversed_targets = {
            (
                record.context.payload["target_source_type"],
                record.context.payload["target_accounting_source_identity"],
            )
            for record in source_history.values()
            if record.source_type == "reconciliation_correction"
        }
        for record in sorted(source_history.values(), key=lambda item: item.first_sequence):
            if (record.source_type, record.identity) in reversed_targets:
                continue
            if record.source_type == "capital_reservation":
                fact_ = cast(AccountingEconomicFact, record.context).payload
                scope_ = Scope(
                    fact_["workspace_id"],
                    fact_["portfolio_id"],
                    fact_["environment"],
                    fact_["exchange_account_id"],
                )
                economics = digest(
                    {
                        key: value
                        for key, value in fact_.items()
                        if key not in {"audit_event_id", "source_fingerprint_sha256"}
                    }
                )
                state = ReservationState(
                    fact_["order_id"],
                    fact_["command_id"],
                    scope_,
                    AssetReference.trusted(fact_["asset_reference"]),
                    decimal(fact_["quantity"]),
                    decimal(fact_["quantity"]),
                    economics,
                )
                rebuilt.reservations[state.order_id] = state
                rebuilt.reservation_commands[state.command_id] = (state.order_id, economics)
            elif record.source_type == "fill":
                context_ = cast(AcceptedFillAccountingContext, record.context).fill_context
                fill_reservation = rebuilt.reservations.get(context_.fill["order_id"])
                if fill_reservation is not None:
                    instrument_ = context_.historical_instrument
                    base_, quote_ = (
                        AssetReference.trusted(instrument_["base_asset_reference"]),
                        AssetReference.trusted(instrument_["quote_asset_reference"]),
                    )
                    spend_asset_ = quote_ if context_.fill["side"] == "BUY" else base_
                    spend_ = (
                        decimal(context_.fill["executed_quantity"])
                        * decimal(context_.fill["execution_price"])
                        if context_.fill["side"] == "BUY"
                        else decimal(context_.fill["executed_quantity"])
                    )
                    if (
                        context_.fill["fee_kind"] == "CHARGE"
                        and AssetReference.trusted(context_.fill["fee_asset_reference"])
                        == spend_asset_
                    ):
                        spend_ += decimal(context_.fill["fee_quantity"])
                    fill_reservation.remaining -= spend_
            elif record.source_type == "capital_release":
                order_ = cast(AccountingEconomicFact, record.context).payload["order_id"]
                if order_ in rebuilt.reservations:
                    rebuilt.reservations[order_].remaining = Fraction()
        return rebuilt

    @property
    def fills(self) -> list[tuple[int, M07PrevalidatedAcceptedFillContext]]:
        return sorted(
            [
                (
                    record.first_sequence,
                    cast(AcceptedFillAccountingContext, record.context).fill_context,
                )
                for record in self.accepted_source_history.values()
                if record.source_type == "fill"
            ],
            key=lambda pair: pair[0],
        )

    def _source_record_integrity(self, source_type: str, identity: str) -> bool:
        record = self.accepted_source_history.get((source_type, identity))
        accepted = self.accepted.get((source_type, identity, RULE))
        if (
            record is None
            or accepted is None
            or accepted[0] != record.fingerprint
            or accepted[1] != record.batch_fingerprint
        ):
            return False
        journal = [
            entry
            for entry in self.entries
            if entry.source_type == source_type and entry.accounting_source_identity == identity
        ]
        if tuple(journal) != accepted[2] or not journal:
            return False
        return all(
            entry.batch_fingerprint_sha256 == record.batch_fingerprint
            and validate_ledger_entry(asdict(entry), entry.append_sequence - 1) == "VALID"
            for entry in journal
        )

    def balances(
        self, role: str = "OWNED_AVAILABLE"
    ) -> dict[tuple[str, str, str, str, tuple[str, str, str, str]], Fraction]:
        result: dict[tuple[str, str, str, str, tuple[str, str, str, str]], Fraction] = {}
        for item in self.entries:
            if item.account_role != role:
                continue
            asset_key = AssetReference.trusted(item.asset_reference).key()
            key = (
                item.workspace_id,
                item.portfolio_id,
                item.environment,
                cast(str, item.exchange_account_id),
                asset_key,
            )
            result[key] = result.get(key, Fraction()) + (
                1 if item.direction == "DEBIT" else -1
            ) * decimal(item.quantity)
        return result

    def _available(self, scope: Scope, asset: AssetReference) -> Fraction:
        return self.balances().get((*asdict(scope).values(), asset.key()), Fraction())

    def _reserved(self, scope: Scope, asset: AssetReference) -> Fraction:
        return self.balances("OWNED_RESERVED").get(
            (*asdict(scope).values(), asset.key()), Fraction()
        )

    def _commit(
        self,
        source_type: str,
        identity: str,
        fingerprint: str,
        effective: str,
        postings: list[PostingProjection],
        *,
        order_id: str | None = None,
        fill_id: str | None = None,
        audit_event_id: str | None = None,
        correction_reason: str | None = None,
        precommit_validator: Callable[[tuple[LedgerEntry, ...]], str] | None = None,
    ) -> str:
        key = (source_type, identity, RULE)
        if key in self.accepted and self.accepted[key][0] != fingerprint:
            return "ACCOUNTING_IDENTITY_CONFLICT"
        try:
            if not HEX_RE.fullmatch(fingerprint):
                raise ValueError
            for posting in postings:
                posting.validate()
            keys = [
                (source_type, identity, RULE, index, posting.posting_role)
                for index, posting in enumerate(postings)
            ]
            if len(keys) != len(set(keys)):
                return "CONTRACT_INCONSISTENT"
            batch = digest(
                [posting.canonical_projection(index) for index, posting in enumerate(postings)]
            )
            if key in self.accepted:
                return (
                    "REPLAY_SUCCESS" if self.accepted[key][1] == batch else "CONTRACT_INCONSISTENT"
                )
            totals: dict[tuple[str, str, str, str], list[Fraction]] = {}
            for posting in postings:
                sides = totals.setdefault(posting.asset_reference.key(), [Fraction(), Fraction()])
                if posting.direction == "DEBIT":
                    sides[0] += decimal(posting.quantity)
                elif posting.direction == "CREDIT":
                    sides[1] += decimal(posting.quantity)
                else:
                    raise ValueError
            if any(debit != credit for debit, credit in totals.values()):
                return "UNBALANCED_ACCOUNTING_BATCH"
            created: list[LedgerEntry] = []
            for index, posting in enumerate(postings):
                sequence = len(self.entries) + index + 1
                uid = f"{sequence:012x}"[-12:]
                entry = LedgerEntry(
                    f"led_018f0f3e-7b5a-7abc-8def-{uid}",
                    posting.scope.workspace_id,
                    posting.scope.portfolio_id,
                    posting.scope.environment,
                    posting.scope.exchange_account_id,
                    None,
                    posting.asset_reference.object(),
                    posting.account_role,
                    posting.direction,
                    posting.quantity,
                    source_type,
                    identity,
                    fingerprint,
                    RULE,
                    index,
                    posting.posting_role,
                    batch,
                    effective,
                    sequence,
                    order_id,
                    fill_id,
                    audit_event_id,
                    correction_reason,
                )
                if validate_ledger_entry(asdict(entry), sequence - 1) != "VALID":
                    return "CONTRACT_INCONSISTENT"
                created.append(entry)
            if source_type == "reconciliation_correction" and self.correction_candidate_hook:
                created = list(self.correction_candidate_hook(tuple(created)))
            if precommit_validator is not None and precommit_validator(tuple(created)) != "VALID":
                return "MALFORMED_ACCOUNTING_FACT"
            projected = Engine(entries=[*self.entries, *created])
            if any(value < 0 for value in projected.balances().values()) or any(
                value < 0 for value in projected.balances("OWNED_RESERVED").values()
            ):
                return "INSUFFICIENT_AVAILABLE_CAPITAL"
        except (TypeError, ValueError):
            return "MALFORMED_ACCOUNTING_FACT"
        self.entries.extend(created)
        self.accepted[key] = (fingerprint, batch, tuple(created))
        return "ACCEPTED"

    def account_fact(self, fact: AccountingEconomicFact) -> str:
        if not isinstance(fact, AccountingEconomicFact):
            return "TRUSTED_CONTEXT_FAILURE"
        p, source = fact.payload, cast(str, fact.payload["source_type"])
        if source == "capital_reservation" and (
            not isinstance(fact.upstream_context, M07PrevalidatedAcceptedCommandContext)
            or fact.upstream_context.acceptance_seal is not _M07_CORE_ACCEPTANCE_SEAL
        ):
            return "TRUSTED_CONTEXT_FAILURE"
        if source == "capital_release" and (
            not isinstance(fact.upstream_context, M07PrevalidatedAcceptedTerminalOrderEventContext)
            or fact.upstream_context.acceptance_seal is not _M07_CORE_ACCEPTANCE_SEAL
        ):
            return "TRUSTED_CONTEXT_FAILURE"
        if source in {"funding", "interest", "fee", "realized_pnl"}:
            return "UNSUPPORTED_ACCOUNTING_SEMANTICS"
        effective, identity = cast(str, p["effective_at_utc"]), cast(str, p["audit_event_id"])
        scope: Scope
        asset: AssetReference
        q: str
        if source == "capital_reservation":
            scope = Scope(
                p["workspace_id"], p["portfolio_id"], p["environment"], p["exchange_account_id"]
            )
            asset, q = AssetReference.trusted(p["asset_reference"]), cast(str, p["quantity"])
            command_id, order_id = cast(str, p["command_id"]), cast(str, p["order_id"])
            economics = digest(
                {
                    key: value
                    for key, value in p.items()
                    if key not in {"audit_event_id", "source_fingerprint_sha256"}
                }
            )
            previous = self.reservation_commands.get(command_id)
            if previous:
                return (
                    "REPLAY_SUCCESS"
                    if previous == (order_id, economics)
                    else "ACCOUNTING_IDENTITY_CONFLICT"
                )
            if order_id in self.reservations:
                return "RESERVATION_CONFLICT"
            postings = [
                PostingProjection(scope, asset, "OWNED_RESERVED", "DEBIT", q, "RESERVE_HELD"),
                PostingProjection(
                    scope, asset, "OWNED_AVAILABLE", "CREDIT", q, "RESERVE_AVAILABLE"
                ),
            ]
        elif source == "capital_release":
            order_id = cast(str, p["order_id"])
            state = self.reservations.get(order_id)
            if state is not None and state.scope != Scope(
                p["workspace_id"], p["portfolio_id"], p["environment"], p["exchange_account_id"]
            ):
                return "RESERVATION_CONFLICT"
            if state is None or state.remaining == 0:
                return (
                    "REPLAY_SUCCESS"
                    if (source, identity, RULE) in self.accepted
                    else "RESERVATION_CONFLICT"
                )
            scope, asset, q = state.scope, state.asset, exact_decimal(state.remaining)
            postings = [
                PostingProjection(scope, asset, "OWNED_AVAILABLE", "DEBIT", q, "RESERVE_AVAILABLE"),
                PostingProjection(scope, asset, "OWNED_RESERVED", "CREDIT", q, "RESERVE_HELD"),
            ]
        elif source == "internal_transfer":
            scope = Scope(
                p["workspace_id"],
                p["portfolio_id"],
                p["environment"],
                p["source_exchange_account_id"],
            )
            dst = Scope(
                p["workspace_id"],
                p["portfolio_id"],
                p["destination_environment"],
                p["destination_exchange_account_id"],
            )
            asset, destination_asset, q = (
                AssetReference.trusted(p["source_asset_reference"]),
                AssetReference.trusted(p["destination_asset_reference"]),
                cast(str, p["source_quantity"]),
            )
            if (
                scope.environment != dst.environment
                or asset != destination_asset
                or q != p["destination_quantity"]
            ):
                return "UNSUPPORTED_ACCOUNTING_SEMANTICS"
            postings = [
                PostingProjection(
                    scope, asset, "TRANSFER_CLEARING", "DEBIT", q, "TRANSFER_CLEARING_SOURCE"
                ),
                PostingProjection(scope, asset, "OWNED_AVAILABLE", "CREDIT", q, "TRANSFER_SOURCE"),
                PostingProjection(
                    dst, asset, "OWNED_AVAILABLE", "DEBIT", q, "TRANSFER_DESTINATION"
                ),
                PostingProjection(
                    dst, asset, "TRANSFER_CLEARING", "CREDIT", q, "TRANSFER_CLEARING_DESTINATION"
                ),
            ]
        elif source == "reconciliation_correction":
            target_key = (
                cast(str, p["target_source_type"]),
                cast(str, p["target_accounting_source_identity"]),
            )
            target = self.accepted_source_history.get(target_key)
            supported_targets = {
                "fill",
                "deposit",
                "withdrawal",
                "internal_transfer",
                "capital_reservation",
                "capital_release",
            }
            already_reversed = any(
                record.source_type == "reconciliation_correction"
                and record.identity != identity
                and (
                    record.context.payload["target_source_type"],
                    record.context.payload["target_accounting_source_identity"],
                    record.context.payload["target_source_fingerprint_sha256"],
                    record.context.payload["target_batch_fingerprint_sha256"],
                )
                == (
                    p["target_source_type"],
                    p["target_accounting_source_identity"],
                    p["target_source_fingerprint_sha256"],
                    p["target_batch_fingerprint_sha256"],
                )
                for record in self.accepted_source_history.values()
            )
            if already_reversed:
                return "ACCOUNTING_IDENTITY_CONFLICT"
            if (
                target is None
                or target.source_type not in supported_targets
                or target.fingerprint != p["target_source_fingerprint_sha256"]
                or target.batch_fingerprint != p["target_batch_fingerprint_sha256"]
                or not self._source_record_integrity(*target_key)
            ):
                return "TRUSTED_CONTEXT_FAILURE"
            target_entries = self.accepted[(target.source_type, target.identity, RULE)][2]
            correction_scope = (p["workspace_id"], p["portfolio_id"], p["environment"])
            if any(
                (entry.workspace_id, entry.portfolio_id, entry.environment) != correction_scope
                for entry in target_entries
            ):
                return "TRUSTED_CONTEXT_FAILURE"
            postings = [
                PostingProjection(
                    Scope(
                        entry.workspace_id,
                        entry.portfolio_id,
                        entry.environment,
                        cast(str, entry.exchange_account_id),
                    ),
                    AssetReference.trusted(entry.asset_reference),
                    entry.account_role,
                    "CREDIT" if entry.direction == "DEBIT" else "DEBIT",
                    entry.quantity,
                    entry.posting_role,
                )
                for entry in target_entries
            ]
        else:
            scope = Scope(
                p["workspace_id"], p["portfolio_id"], p["environment"], p["exchange_account_id"]
            )
            asset, q = AssetReference.trusted(p["asset_reference"]), cast(str, p["quantity"])
            if source == "deposit":
                postings = [
                    PostingProjection(
                        scope, asset, "OWNED_AVAILABLE", "DEBIT", q, "ASSET_RECEIVED"
                    ),
                    PostingProjection(
                        scope, asset, "EXTERNAL_CAPITAL", "CREDIT", q, "CAPITAL_CLASSIFIED"
                    ),
                ]
            elif source == "withdrawal":
                postings = [
                    PostingProjection(
                        scope, asset, "EXTERNAL_CAPITAL", "DEBIT", q, "CAPITAL_CLASSIFIED"
                    ),
                    PostingProjection(scope, asset, "OWNED_AVAILABLE", "CREDIT", q, "ASSET_PAID"),
                ]
            else:
                return "UNSUPPORTED_ACCOUNTING_SEMANTICS"
        precommit_validator = (
            (lambda candidates: validate_correction_batch(fact, target_entries, candidates))
            if source == "reconciliation_correction"
            else None
        )
        outcome = self._commit(
            source,
            identity,
            fact.source_fingerprint,
            effective,
            postings,
            audit_event_id=identity,
            correction_reason=cast(str, p["reason"])
            if source == "reconciliation_correction"
            else None,
            precommit_validator=precommit_validator,
        )
        if outcome == "ACCEPTED":
            created = self.accepted[(source, identity, RULE)][2]
            self.accepted_source_history[(source, identity)] = AcceptedSourceRecord(
                source,
                identity,
                fact.source_fingerprint,
                created[0].batch_fingerprint_sha256,
                created[0].append_sequence,
                fact,
            )
            if source == "capital_reservation":
                state = ReservationState(
                    cast(str, p["order_id"]),
                    cast(str, p["command_id"]),
                    scope,
                    asset,
                    decimal(q),
                    decimal(q),
                    economics,
                )
                self.reservations[state.order_id] = state
                self.reservation_commands[state.command_id] = (state.order_id, economics)
            elif source == "capital_release":
                self.reservations[cast(str, p["order_id"])].remaining = Fraction()
            elif source == "reconciliation_correction":
                rebuilt = Engine.rebuild(
                    self.entries, self.accepted_source_history, _validate=False
                )
                self.reservations = rebuilt.reservations
                self.reservation_commands = rebuilt.reservation_commands
        return outcome

    def _inventory(
        self,
    ) -> dict[tuple[Scope, AssetReference], list[InventoryLot]]:
        inventory: dict[tuple[Scope, AssetReference], list[InventoryLot]] = {}
        reversed_targets = {
            (
                record.context.payload["target_source_type"],
                record.context.payload["target_accounting_source_identity"],
            )
            for record in self.accepted_source_history.values()
            if record.source_type == "reconciliation_correction"
        }

        def consume(key: tuple[Scope, AssetReference], amount: Fraction) -> list[InventoryLot]:
            current = inventory.setdefault(key, [])
            if sum((lot.quantity for lot in current), Fraction()) < amount:
                raise ValueError("INSUFFICIENT_AVAILABLE_CAPITAL")
            moved: list[InventoryLot] = []
            rebuilt: list[InventoryLot] = []
            remaining = amount
            for index, lot in enumerate(current):
                used = min(lot.quantity, remaining)
                if used:
                    moved.append(replace(lot, quantity=used))
                remaining -= used
                if lot.quantity > used:
                    rebuilt.append(replace(lot, quantity=lot.quantity - used))
                if remaining == 0:
                    rebuilt.extend(current[index + 1 :])
                    break
            inventory[key] = rebuilt
            return moved

        for record in sorted(
            self.accepted_source_history.values(), key=lambda item: item.first_sequence
        ):
            if (record.source_type, record.identity) in reversed_targets or record.source_type in {
                "capital_reservation",
                "capital_release",
                "reconciliation_correction",
            }:
                continue
            if record.source_type == "fill":
                context = cast(AcceptedFillAccountingContext, record.context).fill_context
                fact_, instrument = context.fill, context.historical_instrument
                account = Scope(
                    fact_["workspace_id"],
                    fact_["portfolio_id"],
                    fact_["environment"],
                    fact_["exchange_account_id"],
                )
                base, quote = (
                    AssetReference.trusted(instrument["base_asset_reference"]),
                    AssetReference.trusted(instrument["quote_asset_reference"]),
                )
                quantity_ = decimal(fact_["executed_quantity"])
                if fact_["side"] == "BUY":
                    consume((account, quote), quantity_ * decimal(fact_["execution_price"]))
                    net = quantity_ - (
                        decimal(fact_["fee_quantity"])
                        if fact_["fee_kind"] == "CHARGE"
                        and AssetReference.trusted(fact_["fee_asset_reference"]) == base
                        else Fraction()
                    )
                    if net:
                        inventory.setdefault((account, base), []).append(
                            InventoryLot(
                                net,
                                decimal(fact_["execution_price"]),
                                quote,
                                fact_["fill_id"],
                                record.first_sequence,
                                0,
                            )
                        )
                else:
                    consume((account, base), quantity_)
                    inventory.setdefault((account, quote), []).append(
                        InventoryLot(
                            quantity_ * decimal(fact_["execution_price"]),
                            Fraction(1),
                            quote,
                            fact_["fill_id"],
                            record.first_sequence,
                            2,
                        )
                    )
                if fact_["fee_kind"] == "CHARGE":
                    fee_asset = AssetReference.trusted(fact_["fee_asset_reference"])
                    if not (fact_["side"] == "BUY" and fee_asset == base):
                        consume((account, fee_asset), decimal(fact_["fee_quantity"]))
            else:
                fact_ = cast(AccountingEconomicFact, record.context).payload
                if record.source_type in {"deposit", "withdrawal"}:
                    account = Scope(
                        fact_["workspace_id"],
                        fact_["portfolio_id"],
                        fact_["environment"],
                        fact_["exchange_account_id"],
                    )
                    asset_ = AssetReference.trusted(fact_["asset_reference"])
                    key = (account, asset_)
                    if record.source_type == "deposit":
                        inventory.setdefault(key, []).append(
                            InventoryLot(
                                decimal(fact_["quantity"]),
                                decimal(fact_["unit_cost_basis"]),
                                AssetReference.trusted(fact_["basis_valuation_unit"]),
                                record.identity,
                                record.first_sequence,
                                0,
                            )
                        )
                    else:
                        consume(key, decimal(fact_["quantity"]))
                elif record.source_type == "internal_transfer":
                    source_scope = Scope(
                        fact_["workspace_id"],
                        fact_["portfolio_id"],
                        fact_["environment"],
                        fact_["source_exchange_account_id"],
                    )
                    destination_scope = Scope(
                        fact_["workspace_id"],
                        fact_["portfolio_id"],
                        fact_["destination_environment"],
                        fact_["destination_exchange_account_id"],
                    )
                    asset_ = AssetReference.trusted(fact_["source_asset_reference"])
                    moved = consume((source_scope, asset_), decimal(fact_["source_quantity"]))
                    inventory.setdefault((destination_scope, asset_), []).extend(moved)
        return inventory

    def lots(self, scope: Scope, base: AssetReference) -> list[InventoryLot]:
        return self._inventory().get((scope, base), [])

    def account_trusted_fill(
        self,
        context: M07PrevalidatedAcceptedFillContext | Mapping[str, Any],
        historical_valuation: PrevalidatedValuationContext | None = None,
    ) -> str:
        if not isinstance(context, M07PrevalidatedAcceptedFillContext):
            return "TRUSTED_CONTEXT_FAILURE"
        f = context.fill
        i = context.historical_instrument
        replay_key = ("fill", cast(str, f["fill_id"]), RULE)
        if replay_key in self.accepted:
            if self.accepted[replay_key][0] != context.source_fingerprint:
                return "ACCOUNTING_IDENTITY_CONFLICT"
            return (
                "REPLAY_SUCCESS"
                if self._source_record_integrity("fill", cast(str, f["fill_id"]))
                else "CONTRACT_INCONSISTENT"
            )
        scope = Scope(
            f["workspace_id"], f["portfolio_id"], f["environment"], f["exchange_account_id"]
        )
        base, quote = (
            AssetReference.trusted(i["base_asset_reference"]),
            AssetReference.trusted(i["quote_asset_reference"]),
        )
        q = decimal(f["executed_quantity"])
        fill_fee_asset = (
            AssetReference.trusted(f["fee_asset_reference"]) if f["fee_kind"] == "CHARGE" else None
        )
        if f["side"] == "BUY" and fill_fee_asset == base and decimal(f["fee_quantity"]) > q:
            return "INSUFFICIENT_AVAILABLE_CAPITAL"
        notional = q * decimal(f["execution_price"])
        qtext = exact_decimal(notional)
        if not DECIMAL_RE.fullmatch(qtext):
            return "MALFORMED_ACCOUNTING_FACT"
        spend_asset = quote if f["side"] == "BUY" else base
        spend_quantity = notional if f["side"] == "BUY" else q
        if fill_fee_asset == spend_asset:
            spend_quantity += decimal(f["fee_quantity"])
        reservation = self.reservations.get(cast(str, f["order_id"]))
        use_reserved = reservation is not None
        if reservation is not None and (
            reservation.scope != scope
            or reservation.asset != spend_asset
            or reservation.remaining < spend_quantity
        ):
            return "RESERVATION_CONFLICT"
        spend_role = "OWNED_RESERVED" if use_reserved else "OWNED_AVAILABLE"
        if f["side"] == "BUY":
            postings = [
                PostingProjection(
                    scope,
                    base,
                    "OWNED_AVAILABLE",
                    "DEBIT",
                    f["executed_quantity"],
                    "ASSET_RECEIVED",
                ),
                PostingProjection(
                    scope,
                    base,
                    "TRADE_CLEARING",
                    "CREDIT",
                    f["executed_quantity"],
                    "TRADE_COUNTERPART",
                ),
                PostingProjection(
                    scope, quote, "TRADE_CLEARING", "DEBIT", qtext, "TRADE_COUNTERPART"
                ),
                PostingProjection(scope, quote, spend_role, "CREDIT", qtext, "ASSET_PAID"),
            ]
        else:
            if sum((x.quantity for x in self.lots(scope, base)), Fraction()) < q:
                return "INSUFFICIENT_AVAILABLE_CAPITAL"
            postings = [
                PostingProjection(
                    scope,
                    base,
                    "TRADE_CLEARING",
                    "DEBIT",
                    f["executed_quantity"],
                    "TRADE_COUNTERPART",
                ),
                PostingProjection(
                    scope, base, spend_role, "CREDIT", f["executed_quantity"], "ASSET_PAID"
                ),
                PostingProjection(
                    scope, quote, "OWNED_AVAILABLE", "DEBIT", qtext, "ASSET_RECEIVED"
                ),
                PostingProjection(
                    scope, quote, "TRADE_CLEARING", "CREDIT", qtext, "TRADE_COUNTERPART"
                ),
            ]
            lots = self.lots(scope, base)
            remaining = q
            cost = Fraction()
            for lot in lots:
                used = min(lot.quantity, remaining)
                if lot.basis_valuation_unit != quote:
                    if historical_valuation is None:
                        return "MISSING_VALUATION"
                    status, rate = resolve_rate(
                        historical_valuation, lot.basis_valuation_unit, quote
                    )
                    if status != "COMPLETE":
                        return status
                    cost += used * lot.unit_cost_basis * cast(Fraction, rate)
                else:
                    cost += used * lot.unit_cost_basis
                remaining -= used
                if not remaining:
                    break
            pnl = notional - cost
            if pnl:
                ptext = exact_decimal(abs(pnl))
                directions = ("DEBIT", "CREDIT") if pnl > 0 else ("CREDIT", "DEBIT")
                postings.extend(
                    [
                        PostingProjection(
                            scope,
                            quote,
                            "REALIZED_PNL_CLASSIFICATION",
                            directions[0],
                            ptext,
                            "PNL_CLASSIFIED",
                        ),
                        PostingProjection(
                            scope, quote, "TRADE_CLEARING", directions[1], ptext, "PNL_COUNTERPART"
                        ),
                    ]
                )
        if f["fee_kind"] == "CHARGE":
            fee_asset = AssetReference.trusted(f["fee_asset_reference"])
            postings.extend(
                [
                    PostingProjection(
                        scope,
                        fee_asset,
                        "FEE_EXPENSE",
                        "DEBIT",
                        f["fee_quantity"],
                        "FEE_CLASSIFIED",
                    ),
                    PostingProjection(
                        scope,
                        fee_asset,
                        spend_role
                        if fee_asset == spend_asset and use_reserved
                        else "OWNED_AVAILABLE",
                        "CREDIT",
                        f["fee_quantity"],
                        "FEE_PAID",
                    ),
                ]
            )
        outcome = self._commit(
            "fill",
            f["fill_id"],
            context.source_fingerprint,
            f["executed_at_utc"],
            postings,
            order_id=f["order_id"],
            fill_id=f["fill_id"],
        )
        if outcome == "ACCEPTED":
            created = self.accepted[("fill", f["fill_id"], RULE)][2]
            self.accepted_source_history[("fill", f["fill_id"])] = AcceptedSourceRecord(
                "fill",
                f["fill_id"],
                context.source_fingerprint,
                created[0].batch_fingerprint_sha256,
                created[0].append_sequence,
                AcceptedFillAccountingContext(context, historical_valuation),
            )
            if reservation is not None:
                reservation.remaining -= spend_quantity
        return outcome


def exact_decimal(value: Fraction) -> str:
    if value < 0:
        raise ValueError("MALFORMED_ACCOUNTING_FACT")
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        raise ValueError("MALFORMED_ACCOUNTING_FACT")
    scale = max(twos, fives)
    numerator = value.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
    if scale == 0:
        return str(numerator)
    raw = str(numerator).zfill(scale + 1)
    result = raw[:-scale] + "." + raw[-scale:].rstrip("0")
    return result.rstrip(".")


@dataclass(frozen=True)
class ValuationEdge:
    subject_reference: AssetReference
    valuation_unit: AssetReference
    rate: str
    source_id: str
    observed_at_utc: str
    effective_at_utc: str
    as_of_utc: str
    stale_after_utc: str
    source_fingerprint_sha256: str


@dataclass(frozen=True)
class CoreAcceptedValuationSourceRegistry:
    """Core-owned accepted identity -> exact fingerprint projection."""

    accepted_fingerprints: Mapping[str, str]


@dataclass(frozen=True)
class PrevalidatedValuationContext:
    edges: tuple[ValuationEdge, ...]
    reporting_as_of_utc: str
    authority: CoreAcceptedValuationSourceRegistry


def preexisting_valuation_projection(
    raw_edges: list[dict[str, Any]],
) -> CoreAcceptedValuationSourceRegistry:
    return CoreAcceptedValuationSourceRegistry(
        freeze({edge["source_id"]: edge["source_fingerprint_sha256"] for edge in raw_edges})
    )


def validate_valuation_edges(
    raw_edges: list[dict[str, Any]],
    authority: CoreAcceptedValuationSourceRegistry,
    reporting_as_of_utc: str,
) -> PrevalidatedValuationContext:
    if not isinstance(authority, CoreAcceptedValuationSourceRegistry):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    reporting = timestamp(reporting_as_of_utc)
    fields = set(EXPECTED_PROTOCOLS["valuation_protocol"]["required_fields"])
    edges = []
    for raw in raw_edges:
        if type(raw) is not dict or set(raw) != fields:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        projected = {k: v for k, v in raw.items() if k != "source_fingerprint_sha256"}
        if (
            raw["source_fingerprint_sha256"] != digest(projected)
            or authority.accepted_fingerprints.get(raw["source_id"])
            != raw["source_fingerprint_sha256"]
            or raw["as_of_utc"] != reporting_as_of_utc
        ):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        subject, unit = (
            AssetReference.trusted(raw["subject_reference"]),
            AssetReference.trusted(raw["valuation_unit"]),
        )
        decimal(raw["rate"])
        observed, effective = timestamp(raw["observed_at_utc"]), timestamp(raw["effective_at_utc"])
        if not observed <= effective <= reporting:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        timestamp(raw["stale_after_utc"])
        edges.append(
            ValuationEdge(
                subject,
                unit,
                raw["rate"],
                raw["source_id"],
                raw["observed_at_utc"],
                raw["effective_at_utc"],
                raw["as_of_utc"],
                raw["stale_after_utc"],
                raw["source_fingerprint_sha256"],
            )
        )
    return PrevalidatedValuationContext(tuple(edges), reporting_as_of_utc, authority)


def accepted_valuation_fixture(
    raw_edges: list[dict[str, Any]], reporting_as_of_utc: str = "2026-01-01T00:00:02Z"
) -> PrevalidatedValuationContext:
    """Test-only two-boundary fixture: model prior Core acceptance, then validate presentation."""
    return validate_valuation_edges(
        raw_edges, preexisting_valuation_projection(raw_edges), reporting_as_of_utc
    )


def resolve_rate(
    context: PrevalidatedValuationContext | Mapping[str, Any],
    subject: AssetReference,
    unit: AssetReference,
) -> tuple[str, Fraction | None]:
    if not isinstance(context, PrevalidatedValuationContext):
        return "TRUSTED_CONTEXT_FAILURE", None
    if subject == unit:
        return "COMPLETE", Fraction(1)
    graph: dict[AssetReference, list[ValuationEdge]] = {}
    for edge in context.edges:
        graph.setdefault(edge.subject_reference, []).append(edge)
    winners: list[tuple[tuple[str, ...], Fraction]] = []
    stale_target = False
    cycle_seen = False
    queue: list[
        tuple[AssetReference, Fraction, frozenset[AssetReference], tuple[str, ...], bool]
    ] = [(subject, Fraction(1), frozenset({subject}), (), False)]
    while queue:
        current, rate, seen, fingerprints, path_stale = queue.pop(0)
        for edge in sorted(graph.get(current, []), key=lambda item: item.source_fingerprint_sha256):
            next_stale = path_stale or timestamp(context.reporting_as_of_utc) >= timestamp(
                edge.stale_after_utc
            )
            if edge.valuation_unit in seen:
                cycle_seen = True
                continue
            combined = rate * decimal(edge.rate)
            path = (*fingerprints, edge.source_fingerprint_sha256)
            if edge.valuation_unit == unit:
                if next_stale:
                    stale_target = True
                else:
                    winners.append((path, combined))
                continue
            queue.append(
                (edge.valuation_unit, combined, seen | {edge.valuation_unit}, path, next_stale)
            )
    if winners:
        return "COMPLETE", min(winners, key=lambda item: item[0])[1]
    if stale_target:
        return "STALE_VALUATION", None
    if cycle_seen:
        return "UNSUPPORTED_VALUATION_PATH", None
    return "MISSING_VALUATION", None


@dataclass(frozen=True)
class PortfolioAccountingScope:
    workspace_id: str
    portfolio_id: str
    environment: str

    def validate(self) -> None:
        if not valid_id(self.workspace_id, "ws") or not valid_id(self.portfolio_id, "port"):
            raise ValueError("TRUSTED_CONTEXT_FAILURE")
        if self.environment not in {"PAPER", "TESTNET", "LIVE"}:
            raise ValueError("TRUSTED_CONTEXT_FAILURE")


def portfolio_scope(scope: Scope) -> PortfolioAccountingScope:
    return PortfolioAccountingScope(scope.workspace_id, scope.portfolio_id, scope.environment)


def nav(
    engine: Engine,
    context: PrevalidatedValuationContext,
    unit: AssetReference,
    scope: PortfolioAccountingScope,
) -> tuple[str, Fraction | None]:
    scope.validate()
    total = Fraction()
    combined = engine.balances()
    reserved = engine.balances("OWNED_RESERVED")
    for key, amount in reserved.items():
        combined[key] = combined.get(key, Fraction()) + amount
    scoped = {
        key: amount
        for key, amount in combined.items()
        if key[:3] == (scope.workspace_id, scope.portfolio_id, scope.environment)
    }
    for key, amount in scoped.items():
        if amount == 0:
            continue
        subject = AssetReference(*key[-1])
        status, rate = resolve_rate(context, subject, unit)
        if status != "COMPLETE":
            return "PARTIALLY_UNVALUED" if status == "MISSING_VALUATION" else status, None
        total += amount * cast(Fraction, rate)
    return "COMPLETE", total


@dataclass(frozen=True)
class RealizedPnlProjection:
    status: str
    gross_realized: Fraction
    fee_effect: Fraction | None
    net_realized: Fraction | None


def _fill_in_scope(
    context: M07PrevalidatedAcceptedFillContext, scope: PortfolioAccountingScope
) -> bool:
    fact = context.fill
    return (fact["workspace_id"], fact["portfolio_id"], fact["environment"]) == (
        scope.workspace_id,
        scope.portfolio_id,
        scope.environment,
    )


def project_realized_pnl(
    engine: Engine,
    scope: PortfolioAccountingScope,
    context: PrevalidatedValuationContext,
    unit: AssetReference,
) -> RealizedPnlProjection:
    scope.validate()
    gross = Fraction()
    for entry in engine.entries:
        if (entry.workspace_id, entry.portfolio_id, entry.environment) != (
            scope.workspace_id,
            scope.portfolio_id,
            scope.environment,
        ) or entry.account_role != "REALIZED_PNL_CLASSIFICATION":
            continue
        status, rate = resolve_rate(context, AssetReference.trusted(entry.asset_reference), unit)
        if status != "COMPLETE":
            return RealizedPnlProjection(status, gross, None, None)
        gross += (
            (1 if entry.direction == "DEBIT" else -1)
            * decimal(entry.quantity)
            * cast(Fraction, rate)
        )
    fees = Fraction()
    for _, accepted in engine.fills:
        if not _fill_in_scope(accepted, scope) or accepted.fill["fee_kind"] != "CHARGE":
            continue
        fee_asset = AssetReference.trusted(accepted.fill["fee_asset_reference"])
        status, rate = resolve_rate(context, fee_asset, unit)
        if status != "COMPLETE":
            return RealizedPnlProjection(
                "MISSING_VALUATION" if status == "MISSING_VALUATION" else status, gross, None, None
            )
        fees += decimal(accepted.fill["fee_quantity"]) * cast(Fraction, rate)
    return RealizedPnlProjection("COMPLETE", gross, fees, gross - fees)


def project_unrealized_pnl(
    engine: Engine,
    scope: PortfolioAccountingScope,
    context: PrevalidatedValuationContext,
    unit: AssetReference,
) -> tuple[str, Fraction | None]:
    scope.validate()
    subjects: set[tuple[Scope, AssetReference, AssetReference]] = set()
    for _, accepted in engine.fills:
        if _fill_in_scope(accepted, scope):
            fact, instrument = accepted.fill, accepted.historical_instrument
            subjects.add(
                (
                    Scope(
                        fact["workspace_id"],
                        fact["portfolio_id"],
                        fact["environment"],
                        fact["exchange_account_id"],
                    ),
                    AssetReference.trusted(instrument["base_asset_reference"]),
                    AssetReference.trusted(instrument["quote_asset_reference"]),
                )
            )
    total = Fraction()
    failures: set[str] = set()
    ordered = sorted(
        subjects,
        key=lambda item: (
            item[0].workspace_id,
            item[0].portfolio_id,
            item[0].environment,
            item[0].exchange_account_id,
            item[1].key(),
        ),
    )
    for account_scope, base, quote in ordered:
        base_status, mark = resolve_rate(context, base, unit)
        quote_status, quote_rate = resolve_rate(context, quote, unit)
        if base_status != "COMPLETE":
            failures.add(base_status)
            continue
        if quote_status != "COMPLETE":
            failures.add(quote_status)
            continue
        for lot in engine.lots(account_scope, base):
            basis_status, basis_rate = resolve_rate(context, lot.basis_valuation_unit, unit)
            if basis_status != "COMPLETE":
                failures.add(basis_status)
                continue
            total += lot.quantity * cast(Fraction, mark) - (
                lot.quantity * lot.unit_cost_basis * cast(Fraction, basis_rate)
            )
    if failures:
        for failure in ("STALE_VALUATION", "UNSUPPORTED_VALUATION_PATH", "MISSING_VALUATION"):
            if failure in failures:
                return failure, None
    return "COMPLETE", total


@dataclass(frozen=True)
class ObservedBalanceFact:
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    asset_reference: AssetReference | None
    observed_quantity: str
    as_of_utc: str
    source_id: str
    source_fingerprint_sha256: str


@dataclass(frozen=True)
class CoreAcceptedObservedBalanceSourceRegistry:
    accepted: Mapping[str, tuple[str, str]]


@dataclass(frozen=True)
class PrevalidatedObservedBalanceContext:
    fact: ObservedBalanceFact
    mapping_authoritative: bool
    semantics: str
    authority: CoreAcceptedObservedBalanceSourceRegistry


def preexisting_observed_projection(
    raw: dict[str, Any], semantics: str = "BALANCE"
) -> CoreAcceptedObservedBalanceSourceRegistry:
    return CoreAcceptedObservedBalanceSourceRegistry(
        freeze({raw["source_id"]: (raw["source_fingerprint_sha256"], semantics)})
    )


def validate_observed_context(
    raw: dict[str, Any], authority: CoreAcceptedObservedBalanceSourceRegistry
) -> PrevalidatedObservedBalanceContext:
    if not isinstance(authority, CoreAcceptedObservedBalanceSourceRegistry):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    fields = set(EXPECTED_PROTOCOLS["reconciliation_protocol"]["observed_fact_fields"])
    if set(raw) != fields:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    projected = {k: v for k, v in raw.items() if k != "source_fingerprint_sha256"}
    accepted = authority.accepted.get(raw["source_id"])
    if (
        raw["source_fingerprint_sha256"] != digest(projected)
        or accepted is None
        or accepted[0] != raw["source_fingerprint_sha256"]
    ):
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    scope = Scope(
        raw["workspace_id"], raw["portfolio_id"], raw["environment"], raw["exchange_account_id"]
    )
    scope.validate()
    raw_asset = raw["asset_reference"]
    if type(raw_asset) is not dict or set(raw_asset) != {
        "venue_asset_code",
        "canonical_display_code",
        "asset_namespace",
        "mapping_status",
    }:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    mapping_authoritative = raw_asset["mapping_status"] in {"EXACT", "EXPLICIT_ALIAS"}
    asset = AssetReference.trusted(raw_asset) if mapping_authoritative else None
    if raw_asset["mapping_status"] not in {"EXACT", "EXPLICIT_ALIAS", "UNKNOWN", "AMBIGUOUS"}:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    decimal(raw["observed_quantity"], positive=False)
    timestamp(raw["as_of_utc"])
    return PrevalidatedObservedBalanceContext(
        ObservedBalanceFact(
            scope.workspace_id,
            scope.portfolio_id,
            scope.environment,
            scope.exchange_account_id,
            asset,
            raw["observed_quantity"],
            raw["as_of_utc"],
            raw["source_id"],
            raw["source_fingerprint_sha256"],
        ),
        mapping_authoritative,
        accepted[1],
        authority,
    )


def accepted_observed_fixture(raw: dict[str, Any]) -> PrevalidatedObservedBalanceContext:
    """Test-only fixture for a separately preexisting Core observation projection."""
    semantics = "UNSUPPORTED" if raw["source_id"] == ident("snap", 63) else "BALANCE"
    return validate_observed_context(raw, preexisting_observed_projection(raw, semantics))


def reconcile(
    engine: Engine,
    context: PrevalidatedObservedBalanceContext | Mapping[str, Any] | None,
) -> str:
    if context is None:
        return "MISSING_EXTERNAL_FACT"
    if not isinstance(context, PrevalidatedObservedBalanceContext):
        return "TRUSTED_CONTEXT_FAILURE"
    if context.semantics == "UNSUPPORTED":
        return "UNSUPPORTED"
    if not context.mapping_authoritative or context.fact.asset_reference is None:
        return "UNMAPPED_ASSET"
    fact = context.fact
    scope = Scope(fact.workspace_id, fact.portfolio_id, fact.environment, fact.exchange_account_id)
    asset = cast(AssetReference, fact.asset_reference)
    history = any(
        entry.workspace_id == scope.workspace_id
        and entry.portfolio_id == scope.portfolio_id
        and entry.environment == scope.environment
        and entry.exchange_account_id == scope.exchange_account_id
        and AssetReference.trusted(entry.asset_reference) == asset
        and entry.account_role in {"OWNED_AVAILABLE", "OWNED_RESERVED"}
        for entry in engine.entries
    )
    if not history:
        return "MISSING_INTERNAL_FACT"
    internal = engine._available(scope, asset) + engine._reserved(scope, asset)
    return "MATCH" if internal == decimal(fact.observed_quantity, positive=False) else "DRIFT"


UUID = "018f0f3e-7b5a-7abc-8def-1234567890ab"


def ident(prefix: str, suffix: int = 0) -> str:
    return f"{prefix}_018f0f3e-7b5a-7abc-8def-{suffix:012x}"


def asset(
    code: str, namespace: str = "paper_simulated_venue", status: str = "EXACT"
) -> dict[str, str]:
    return {
        "venue_asset_code": code,
        "canonical_display_code": code,
        "asset_namespace": namespace,
        "mapping_status": status,
    }


USD = AssetReference.trusted(asset("USD"))
BTC = AssetReference.trusted(asset("BTC"))
BNB = AssetReference.trusted(asset("BNB"))
PAPER = Scope(ident("ws"), ident("port"), "PAPER", ident("xacc"))
T = "2026-01-01T00:00:00Z"
INSTRUMENT = {
    "instrument_id": ident("instr"),
    "metadata_version": 1,
    "instrument_type": "SPOT_PAIR",
    "workspace_id": PAPER.workspace_id,
    "environment": "PAPER",
    "exchange_id": "paper_simulated_venue",
    "base_asset_reference": BTC.object(),
    "quote_asset_reference": USD.object(),
}


def full_fill_fact(
    side: str,
    quantity: str,
    price: str,
    *,
    number: int,
    fee_kind: str = "NONE",
    fee_quantity: str = "0",
    fee_asset: AssetReference | None = None,
    scope: Scope = PAPER,
    effective: str = T,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "fill_id": ident("fill", number),
        "order_id": ident("ord", number),
        "environment": scope.environment,
        "workspace_id": scope.workspace_id,
        "portfolio_id": scope.portfolio_id,
        "exchange_account_id": scope.exchange_account_id,
        "exchange_id": "paper_simulated_venue",
        "instrument_id": INSTRUMENT["instrument_id"],
        "instrument_metadata_version": 1,
        "execution_route_id": ident("xroute", number),
        "venue_trade_id": f"paper-trade-{number}",
        "side": side,
        "executed_quantity": quantity,
        "execution_price": price,
        "executed_at_utc": effective,
        "fee_kind": fee_kind,
        "fee_quantity": fee_quantity,
        "fee_asset_reference": fee_asset.object() if fee_asset else None,
        "fill_fingerprint_sha256": "",
    }
    raw["fill_fingerprint_sha256"] = canonical_m07_fill_fingerprint(raw)
    return raw


def fill(
    side: str,
    quantity: str,
    price: str,
    *,
    number: int,
    fee_kind: str = "NONE",
    fee_quantity: str = "0",
    fee_asset: AssetReference | None = None,
    scope: Scope = PAPER,
    effective: str = T,
) -> M07PrevalidatedAcceptedFillContext:
    raw = full_fill_fact(
        side,
        quantity,
        price,
        number=number,
        fee_kind=fee_kind,
        fee_quantity=fee_quantity,
        fee_asset=fee_asset,
        scope=scope,
        effective=effective,
    )
    instrument = copy.deepcopy(INSTRUMENT)
    instrument["workspace_id"] = scope.workspace_id
    instrument["environment"] = scope.environment
    return attest_m07_accepted_fill(
        raw, instrument, CoreAcceptedFillProjection(raw["order_id"], (raw["fill_id"],))
    )


def fact(
    source: str, scope: Scope, asset_: AssetReference, quantity_: str, number: int, **extra: Any
) -> AccountingEconomicFact:
    upstream = extra.pop("_upstream_context", None)
    if source in {"deposit", "withdrawal"}:
        extra.setdefault("basis_valuation_unit", USD.object())
        extra.setdefault("unit_cost_basis", "1")
    raw = {
        "audit_event_id": ident("evt", number),
        "source_type": source,
        "workspace_id": scope.workspace_id,
        "portfolio_id": scope.portfolio_id,
        "environment": scope.environment,
        "effective_at_utc": T,
        "provenance": "trusted_test_provenance",
        "exchange_account_id": scope.exchange_account_id,
        "asset_reference": asset_.object(),
        "quantity": quantity_,
        **extra,
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    authority = preexisting_accounting_projection((freeze(raw),))
    return economic_fact(raw, authority, upstream)


def deposit(
    engine: Engine, scope: Scope, asset_: AssetReference, quantity_: str, number: int
) -> str:
    return engine.account_fact(
        fact("deposit", scope, asset_, quantity_, number, capital_flow_kind="EXTERNAL_CONTRIBUTION")
    )


def lot_values(lots: list[InventoryLot]) -> list[tuple[Fraction, Fraction, AssetReference, str]]:
    return [
        (lot.quantity, lot.unit_cost_basis, lot.basis_valuation_unit, lot.source_identity)
        for lot in lots
    ]


def valuation_edge(
    subject: AssetReference,
    unit: AssetReference,
    rate: str,
    *,
    stale: bool = False,
    number: int = 0,
) -> dict[str, Any]:
    raw = {
        "subject_reference": subject.object(),
        "valuation_unit": unit.object(),
        "rate": rate,
        "source_id": ident("valsrc", number),
        "observed_at_utc": "2026-01-01T00:00:00Z",
        "effective_at_utc": "2026-01-01T00:00:01Z",
        "as_of_utc": "2026-01-01T00:00:02Z",
        "stale_after_utc": "2026-01-01T00:00:02Z" if stale else "2026-01-01T00:01:00Z",
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    return raw


def test_contract_and_real_upstream_dependencies_are_valid() -> None:
    assert validate_contract(CONTRACT) == "VALID"


@pytest.mark.parametrize(
    "root,path,value",
    [
        ("ledger_entry_schema", ("direction_registry",), ["DEBIT", "CREDIT", "BANANA"]),
        ("account_roles", ("FEE_EXPENSE",), "owned asset included in NAV"),
        ("batch_protocol", ("atomicity",), "best effort"),
        ("source_registry", ("fill", "authority"), "raw Fill"),
        ("balance_model", ("key",), ["portfolio_id"]),
        ("reservation_protocol", ("release",), "unbounded"),
        ("asset_identity", ("cross_venue_unit_aggregation",), "by display code"),
        ("instrument_coverage", ("accounting_supported",), ["SPOT_PAIR", "OPTION"]),
        ("cost_basis_policy", ("ordering",), ["effective_at_utc"]),
        ("pnl_model", ("fee_effect",), "subtract twice"),
        ("valuation_protocol", ("missing",), "zero"),
        ("equity_nav", ("canonical_equation",), "owned plus position"),
        ("reconciliation_protocol", ("snapshot",), "overwrites ledger"),
        ("ordering_policy", ("late_and_backfill",), "rewrite"),
        ("environment_policy", ("isolation",), "fallback"),
        ("environment_policy", ("live",), "enabled"),
        ("m09_outputs", (), ["risk_ok"]),
        ("failure_taxonomy", (), ["CONTRACT_INCONSISTENT"]),
        ("forbidden", (), []),
        ("accounting_economic_fact_schema_registry", ("deposit", "constraints"), "anything"),
    ],
)
def test_every_semantic_protocol_mutation_fails(
    root: str, path: tuple[str, ...], value: Any
) -> None:
    changed = copy.deepcopy(CONTRACT)
    target = changed[root]
    if not path:
        changed[root] = value
    else:
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
    assert validate_contract(changed) == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "mutation", ["remove", "pointer", "fingerprint", "nonexistent", "upstream"]
)
def test_dependency_mutations_fail(mutation: str) -> None:
    changed = copy.deepcopy(CONTRACT)
    upstream = copy.deepcopy(CANONICAL)
    if mutation == "remove":
        changed["cross_contract_dependencies"].pop()
    elif mutation == "pointer":
        changed["cross_contract_dependencies"][0]["json_pointer"] = "/identifier_policy"
    elif mutation == "fingerprint":
        changed["cross_contract_dependencies"][0]["content_fingerprint_sha256"] = "0" * 64
    elif mutation == "nonexistent":
        changed["cross_contract_dependencies"][0]["json_pointer"] = "/not_here"
    else:
        upstream["commands_events_order_lifecycle_and_idempotency.json"]["fill_contract"][
            "float_forbidden"
        ] = False
    assert validate_contract(changed, upstream) == "CONTRACT_INCONSISTENT"


def valid_entry() -> dict[str, Any]:
    return asdict(
        LedgerEntry(
            ident("led"),
            PAPER.workspace_id,
            PAPER.portfolio_id,
            "PAPER",
            PAPER.exchange_account_id,
            None,
            USD.object(),
            "OWNED_AVAILABLE",
            "DEBIT",
            "1",
            "deposit",
            ident("evt"),
            "1" * 64,
            RULE,
            0,
            "ASSET_RECEIVED",
            "2" * 64,
            T,
            1,
            None,
            None,
            ident("evt"),
            None,
        )
    )


def test_exact_full_ledger_entry_is_valid() -> None:
    assert validate_ledger_entry(valid_entry(), 0) == "VALID"


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing",
        "id",
        "direction",
        "role",
        "source",
        "mapping",
        "zero",
        "fingerprint",
        "rule",
        "index",
        "posting_role",
        "batch",
        "timestamp",
        "sequence",
        "reference",
    ],
)
def test_malformed_ledger_entry_fails(change: str) -> None:
    item = valid_entry()
    if change == "extra":
        item["extra"] = 1
    elif change == "missing":
        item.pop("quantity")
    elif change == "id":
        item["ledger_entry_id"] = "led_bad"
    elif change == "direction":
        item["direction"] = "banana"
    elif change == "role":
        item["account_role"] = "UNKNOWN"
    elif change == "source":
        item["source_type"] = "whatever"
    elif change == "mapping":
        item["asset_reference"]["mapping_status"] = "UNKNOWN"
    elif change == "zero":
        item["quantity"] = "0"
    elif change == "fingerprint":
        item["accounting_source_fingerprint_sha256"] = "bad"
    elif change == "rule":
        item["accounting_rule_version"] = "V2"
    elif change == "index":
        item["posting_index"] = -1
    elif change == "posting_role":
        item["posting_role"] = "OPEN"
    elif change == "batch":
        item["batch_fingerprint_sha256"] = "bad"
    elif change == "timestamp":
        item["effective_at_utc"] = "today"
    elif change == "sequence":
        item["append_sequence"] = 2
    else:
        item["audit_event_id"] = None
    assert validate_ledger_entry(item, 0) == "MALFORMED_ACCOUNTING_FACT"


def test_asset_reference_exact_object_and_key_order() -> None:
    reordered = {
        "mapping_status": "EXACT",
        "asset_namespace": "venue",
        "canonical_display_code": "BTC",
        "venue_asset_code": "XBT",
    }
    assert AssetReference.trusted(reordered).key() == ("XBT", "BTC", "venue", "EXACT")


@pytest.mark.parametrize(
    "raw",
    [
        asset("BTC", status="UNKNOWN"),
        asset("BTC", status="AMBIGUOUS"),
        {"venue_asset_code": "BTC"},
        ("BTC", "BTC", "venue", "EXACT"),
    ],
)
def test_untrusted_asset_reference_fails(raw: Any) -> None:
    with pytest.raises(ValueError):
        AssetReference.trusted(raw)


def test_same_display_code_different_namespace_is_not_same_asset() -> None:
    assert AssetReference.trusted(asset("BTC", "venue_a")) != AssetReference.trusted(
        asset("BTC", "venue_b")
    )


def test_raw_fill_is_rejected() -> None:
    assert Engine().account_trusted_fill({"fill_id": ident("fill")}) == "TRUSTED_CONTEXT_FAILURE"


def test_unbalanced_projection_rejected_before_append() -> None:
    engine = Engine()
    posting = PostingProjection(PAPER, USD, "OWNED_AVAILABLE", "DEBIT", "1", "ASSET_RECEIVED")
    assert (
        engine._commit("deposit", ident("evt"), "1" * 64, T, [posting], audit_event_id=ident("evt"))
        == "UNBALANCED_ACCOUNTING_BATCH"
    )
    assert not engine.entries


def test_batch_fingerprint_is_derived_used_and_deterministic() -> None:
    a = Engine()
    b = Engine()
    fa = fact("deposit", PAPER, USD, "10", 1, capital_flow_kind="EXTERNAL_CONTRIBUTION")
    assert a.account_fact(fa) == b.account_fact(fa) == "ACCEPTED"
    assert a.entries[0].batch_fingerprint_sha256 == b.entries[0].batch_fingerprint_sha256
    assert a.account_fact(fa) == "REPLAY_SUCCESS" and len(a.entries) == 2
    key = ("deposit", ident("evt", 1), RULE)
    fingerprint, batch, entries = a.accepted[key]
    a.accepted[key] = (fingerprint, "0" * 64, entries)
    assert (
        a.account_fact(fa) == "CONTRACT_INCONSISTENT"
        and a.entries[0].batch_fingerprint_sha256 == batch
    )


def test_source_fingerprint_tamper_is_rejected() -> None:
    raw = dict(
        fact("deposit", PAPER, USD, "1", 3, capital_flow_kind="EXTERNAL_CONTRIBUTION").payload
    )
    raw["quantity"] = "2"
    with pytest.raises(ValueError):
        economic_fact(raw)


def funded(amount: str = "10000") -> Engine:
    engine = Engine()
    assert deposit(engine, PAPER, USD, amount, 10) == "ACCEPTED"
    return engine


def test_buy_no_fee_and_fill_replay_conflict() -> None:
    engine = funded()
    context = fill("BUY", "1", "100", number=1)
    assert engine.account_trusted_fill(context) == "ACCEPTED"
    count = len(engine.entries)
    lots = engine.lots(PAPER, BTC)
    assert (
        engine.account_trusted_fill(context) == "REPLAY_SUCCESS"
        and len(engine.entries) == count
        and engine.lots(PAPER, BTC) == lots
    )
    altered = fill("BUY", "2", "100", number=1)
    assert (
        engine.account_trusted_fill(altered) == "ACCOUNTING_IDENTITY_CONFLICT"
        and len(engine.entries) == count
    )
    assert not any(item.account_role == "FEE_EXPENSE" for item in engine.entries)


def test_buy_quote_fee_and_no_double_count() -> None:
    engine = funded()
    assert (
        engine.account_trusted_fill(
            fill("BUY", "1", "100", number=2, fee_kind="CHARGE", fee_quantity="1", fee_asset=USD)
        )
        == "ACCEPTED"
    )
    assert engine._available(PAPER, USD) == 9899 and engine._available(PAPER, BTC) == 1
    assert len([x for x in engine.entries if x.account_role == "FEE_EXPENSE"]) == 1


def test_buy_base_fee_changes_lot_and_inventory_exactly() -> None:
    engine = funded()
    assert (
        engine.account_trusted_fill(
            fill("BUY", "1", "100", number=3, fee_kind="CHARGE", fee_quantity="0.01", fee_asset=BTC)
        )
        == "ACCEPTED"
    )
    assert engine._available(PAPER, BTC) == Fraction(99, 100)
    assert lot_values(engine.lots(PAPER, BTC)) == [
        (Fraction(99, 100), Fraction(100), USD, ident("fill", 3))
    ]


def test_third_asset_fee_requires_sufficient_balance() -> None:
    engine = funded()
    context = fill(
        "BUY", "1", "100", number=4, fee_kind="CHARGE", fee_quantity="0.01", fee_asset=BNB
    )
    before = copy.deepcopy(engine.entries)
    assert engine.account_trusted_fill(context) == "INSUFFICIENT_AVAILABLE_CAPITAL"
    assert engine.entries == before
    assert deposit(engine, PAPER, BNB, "1", 11) == "ACCEPTED"
    assert engine.account_trusted_fill(context) == "ACCEPTED" and engine._available(
        PAPER, BNB
    ) == Fraction(99, 100)


def test_sell_no_fee_integrates_fifo_and_balanced_pnl() -> None:
    engine = funded()
    assert engine.account_trusted_fill(fill("BUY", "1", "100", number=20)) == "ACCEPTED"
    assert engine.account_trusted_fill(fill("BUY", "2", "120", number=21)) == "ACCEPTED"
    sell = fill("SELL", "1.5", "150", number=22)
    assert engine.account_trusted_fill(sell) == "ACCEPTED"
    assert lot_values(engine.lots(PAPER, BTC)) == [
        (Fraction(3, 2), Fraction(120), USD, ident("fill", 21))
    ]
    pnl = [
        x
        for x in engine.entries
        if x.fill_id == ident("fill", 22) and x.account_role == "REALIZED_PNL_CLASSIFICATION"
    ]
    assert len(pnl) == 1 and decimal(pnl[0].quantity) == 65 and pnl[0].source_type == "fill"
    assert (
        engine.account_trusted_fill(fill("SELL", "1.5", "130", number=23)) == "ACCEPTED"
        and engine.lots(PAPER, BTC) == []
    )


def test_sell_quote_base_and_third_fee() -> None:
    for number, fee_asset in enumerate((USD, BTC, BNB), 30):
        engine = funded()
        assert deposit(engine, PAPER, BNB, "1", number + 100) == "ACCEPTED"
        assert (
            engine.account_trusted_fill(fill("BUY", "2", "100", number=number + 10)) == "ACCEPTED"
        )
        before = engine._available(PAPER, fee_asset)
        result = engine.account_trusted_fill(
            fill(
                "SELL",
                "1",
                "120",
                number=number,
                fee_kind="CHARGE",
                fee_quantity="0.01",
                fee_asset=fee_asset,
            )
        )
        assert result == "ACCEPTED" and engine._available(PAPER, fee_asset) == before + (
            {USD: Fraction(120), BTC: Fraction(-1), BNB: Fraction()}[fee_asset]
        ) - Fraction(1, 100)


def test_sell_insufficient_inventory_has_zero_effect() -> None:
    engine = funded()
    before = copy.deepcopy(engine.entries)
    assert (
        engine.account_trusted_fill(fill("SELL", "1", "100", number=40))
        == "INSUFFICIENT_AVAILABLE_CAPITAL"
        and engine.entries == before
        and engine.lots(PAPER, BTC) == []
    )


def test_no_ordinary_operation_can_make_owned_negative() -> None:
    engine = Engine()
    assert deposit(engine, PAPER, USD, "10", 50) == "ACCEPTED"
    withdrawal = fact("withdrawal", PAPER, USD, "11", 51, capital_flow_kind="EXTERNAL_WITHDRAWAL")
    before = copy.deepcopy(engine.entries)
    assert (
        engine.account_fact(withdrawal) == "INSUFFICIENT_AVAILABLE_CAPITAL"
        and engine.entries == before
    )
    assert all(value >= 0 for value in engine.balances().values())


def reservation(
    scope: Scope,
    quantity_: str,
    number: int,
    order_suffix: int,
    command_suffix: int | None = None,
    *,
    audit_suffix: int | None = None,
) -> AccountingEconomicFact:
    command_id = ident("cmd", command_suffix or number)
    order_id = ident("ord", order_suffix)
    request = {
        "command_id": command_id,
        "operation_type": "SUBMIT_ORDER",
        "authority_context_id": ident("authctx", number),
        "environment": scope.environment,
        "workspace_id": scope.workspace_id,
        "portfolio_id": scope.portfolio_id,
        "exchange_account_id": scope.exchange_account_id,
        "strategy_instance_id": None,
        "source_type": "OPERATOR",
        "instrument_id": ident("instr", number),
        "execution_route_id": ident("xroute", number),
        "correlation_id": ident("corr", number),
        "causation_id": None,
        "idempotency_key": command_id,
        "order_intent_id": ident("oint", number),
        "order_id": order_id,
        "side": "BUY",
        "order_type": "MARKET",
        "quantity": "1",
        "limit_price": None,
        "time_in_force": "GTC",
        "expire_at_utc": None,
    }
    fingerprint = m07_command_fingerprint(request)
    upstream = attest_m07_accepted_submit_order(
        request, CoreAcceptedCommandProjection(freeze({command_id: fingerprint}))
    )
    return fact(
        "capital_reservation",
        scope,
        USD,
        quantity_,
        audit_suffix or number,
        order_id=order_id,
        command_id=command_id,
        _upstream_context=upstream,
    )


def terminal_release(
    order_suffix: int, number: int, state: str = "CANCELLED"
) -> AccountingEconomicFact:
    event_type = {value: key for key, value in M07_TERMINAL_EVENT_TARGETS.items()}[state]
    safe_payload = (
        {"reason_code": "VENUE_REJECTED"}
        if event_type == "ORDER_REJECTED"
        else {
            "fill_id": ident("fill", number),
            "venue_trade_id": f"trade-{number}",
            "cumulative_executed_quantity": "1",
        }
        if event_type == "ORDER_FILLED"
        else {"replacement_order_id": ident("ord", number + 1000), "venue_order_id": f"v-{number}"}
        if event_type == "ORDER_REPLACE_CONFIRMED"
        else {"venue_order_id": f"v-{number}"}
    )
    event = {
        "audit_event_id": ident("evt", number),
        "event_type": event_type,
        "order_id": ident("ord", order_suffix),
        "aggregate_version": 2,
        "correlation_id": ident("corr", number),
        "causation_id": ident("cause", number),
        "command_id": ident("cmd", number),
        "environment": PAPER.environment,
        "workspace_id": PAPER.workspace_id,
        "portfolio_id": PAPER.portfolio_id,
        "exchange_account_id": PAPER.exchange_account_id,
        "exchange_id": "paper_simulated_venue",
        "instrument_id": ident("instr", number),
        "execution_route_id": ident("xroute", number),
        "occurred_at_utc": T,
        "safe_payload": safe_payload,
        "event_fingerprint_sha256": "",
    }
    event["event_fingerprint_sha256"] = m07_event_fingerprint(event)
    predecessor = {
        "ORDER_REJECTED": "SUBMISSION_PENDING",
        "ORDER_FILLED": "PARTIALLY_FILLED",
        "ORDER_CANCEL_CONFIRMED": "CANCEL_PENDING",
        "ORDER_REPLACE_CONFIRMED": "REPLACE_PENDING",
        "ORDER_EXPIRED": "ACKNOWLEDGED",
    }[event_type]
    evidence = attest_m07_accepted_terminal_event(
        event,
        CoreAcceptedOrderEventProjection(
            freeze({event["audit_event_id"]: accepted_terminal_lifecycle_proof(event, predecessor)})
        ),
    )
    raw = {
        "audit_event_id": ident("evt", number),
        "source_type": "capital_release",
        "workspace_id": PAPER.workspace_id,
        "portfolio_id": PAPER.portfolio_id,
        "environment": PAPER.environment,
        "exchange_account_id": PAPER.exchange_account_id,
        "effective_at_utc": T,
        "provenance": "accepted terminal M0.7 order event",
        "order_id": ident("ord", order_suffix),
        "terminal_state": state,
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    return economic_fact(raw, preexisting_accounting_projection((freeze(raw),)), evidence)


def test_reservation_exact_and_terminal_release_replay() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "100", 60)
    reserve = reservation(PAPER, "100", 61, 61)
    assert engine.account_fact(reserve) == "ACCEPTED"
    assert engine.account_fact(reserve) == "REPLAY_SUCCESS"
    assert engine._available(PAPER, USD) == 0 and engine._reserved(PAPER, USD) == 100
    release = terminal_release(61, 62)
    assert engine.account_fact(release) == "ACCEPTED"
    assert engine.account_fact(release) == "REPLAY_SUCCESS"
    assert engine._available(PAPER, USD) == 100 and engine._reserved(PAPER, USD) == 0


def test_over_reserve_and_wrong_terminal_order_fail() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "10", 70)
    before = len(engine.entries)
    assert engine.account_fact(reservation(PAPER, "11", 71, 71)) == "INSUFFICIENT_AVAILABLE_CAPITAL"
    assert len(engine.entries) == before
    assert engine.account_fact(reservation(PAPER, "5", 72, 72)) == "ACCEPTED"
    assert engine.account_fact(terminal_release(999, 73)) == "RESERVATION_CONFLICT"


def transfer_fact(
    source: Scope,
    destination: Scope,
    source_asset: AssetReference,
    destination_asset: AssetReference,
    quantity_: str,
    number: int,
) -> AccountingEconomicFact:
    raw = {
        "audit_event_id": ident("evt", number),
        "source_type": "internal_transfer",
        "workspace_id": source.workspace_id,
        "portfolio_id": source.portfolio_id,
        "environment": source.environment,
        "effective_at_utc": T,
        "provenance": "transfer_provenance",
        "source_exchange_account_id": source.exchange_account_id,
        "source_asset_reference": source_asset.object(),
        "source_quantity": quantity_,
        "destination_environment": destination.environment,
        "destination_exchange_account_id": destination.exchange_account_id,
        "destination_asset_reference": destination_asset.object(),
        "destination_quantity": quantity_,
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    return economic_fact(raw, preexisting_accounting_projection((freeze(raw),)))


def test_internal_transfer_has_four_legs_constant_portfolio_owned_and_no_pnl() -> None:
    engine = Engine()
    destination = Scope(PAPER.workspace_id, PAPER.portfolio_id, "PAPER", ident("xacc", 2))
    deposit(engine, PAPER, USD, "10", 80)
    before = sum(engine.balances().values())
    assert engine.account_fact(transfer_fact(PAPER, destination, USD, USD, "4", 81)) == "ACCEPTED"
    entries = [x for x in engine.entries if x.audit_event_id == ident("evt", 81)]
    assert (
        len(entries) == 4
        and sum(engine.balances().values()) == before
        and not any(x.account_role == "REALIZED_PNL_CLASSIFICATION" for x in entries)
    )
    assert engine._available(PAPER, USD) == 6 and engine._available(destination, USD) == 4


def test_cross_namespace_and_cross_environment_transfer_fail_closed() -> None:
    engine = Engine()
    destination = Scope(PAPER.workspace_id, PAPER.portfolio_id, "PAPER", ident("xacc", 2))
    deposit(engine, PAPER, USD, "10", 90)
    other = AssetReference.trusted(asset("USD", "other_venue"))
    before = copy.deepcopy(engine.entries)
    with pytest.raises(ValueError, match="UNSUPPORTED_ACCOUNTING_SEMANTICS"):
        transfer_fact(PAPER, destination, USD, other, "1", 91)
    assert engine.entries == before
    testnet = Scope(
        PAPER.workspace_id, PAPER.portfolio_id, "TESTNET", destination.exchange_account_id
    )
    with pytest.raises(ValueError, match="UNSUPPORTED_ACCOUNTING_SEMANTICS"):
        transfer_fact(PAPER, testnet, USD, USD, "1", 92)
    assert engine.entries == before


def test_contribution_and_withdrawal_are_not_pnl() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "100", 100)
    withdraw = fact("withdrawal", PAPER, USD, "20", 101, capital_flow_kind="EXTERNAL_WITHDRAWAL")
    assert engine.account_fact(withdraw) == "ACCEPTED" and engine._available(PAPER, USD) == 80
    assert not any(x.account_role == "REALIZED_PNL_CLASSIFICATION" for x in engine.entries)


def test_trusted_direct_and_deterministic_multihop_valuation() -> None:
    ETH = AssetReference.trusted(asset("ETH"))
    context = accepted_valuation_fixture(
        [valuation_edge(BTC, ETH, "10", number=1), valuation_edge(ETH, USD, "20", number=2)]
    )
    assert resolve_rate(context, BTC, USD) == ("COMPLETE", 200)


def test_raw_missing_stale_cycle_and_different_venue_valuation() -> None:
    assert resolve_rate({}, BTC, USD) == ("TRUSTED_CONTEXT_FAILURE", None)
    assert resolve_rate(accepted_valuation_fixture([]), BTC, USD) == ("MISSING_VALUATION", None)
    assert resolve_rate(
        accepted_valuation_fixture([valuation_edge(BTC, USD, "100", stale=True)]), BTC, USD
    ) == ("STALE_VALUATION", None)
    ETH = AssetReference.trusted(asset("ETH"))
    cycle = accepted_valuation_fixture(
        [valuation_edge(BTC, ETH, "2", number=3), valuation_edge(ETH, BTC, "0.5", number=4)]
    )
    assert resolve_rate(cycle, BTC, USD) == ("UNSUPPORTED_VALUATION_PATH", None)
    other_btc = AssetReference.trusted(asset("BTC", "other_venue"))
    context = accepted_valuation_fixture([valuation_edge(BTC, USD, "100", number=5)])
    assert resolve_rate(context, other_btc, USD) == ("MISSING_VALUATION", None)


@pytest.mark.parametrize("tamper", ["negative", "fingerprint", "fields"])
def test_invalid_valuation_edge_fails(tamper: str) -> None:
    raw = valuation_edge(BTC, USD, "100")
    if tamper == "negative":
        raw["rate"] = "-1"
    elif tamper == "fingerprint":
        raw["source_fingerprint_sha256"] = "0" * 64
    else:
        raw["extra"] = True
    with pytest.raises(ValueError):
        accepted_valuation_fixture([raw])


def test_mark_changes_nav_and_unrealized_without_ledger_or_realized_mutation() -> None:
    engine = funded()
    engine.account_trusted_fill(fill("BUY", "1", "100", number=110))
    snapshot = copy.deepcopy(engine.entries)
    first = accepted_valuation_fixture([valuation_edge(BTC, USD, "120", number=10)])
    second = accepted_valuation_fixture([valuation_edge(BTC, USD, "150", number=11)])
    assert nav(engine, first, USD, portfolio_scope(PAPER)) == ("COMPLETE", 10020) and nav(
        engine, second, USD, portfolio_scope(PAPER)
    ) == (
        "COMPLETE",
        10050,
    )
    assert (
        engine.entries == snapshot
        and len([x for x in engine.entries if x.account_role == "REALIZED_PNL_CLASSIFICATION"]) == 0
    )


def test_missing_mark_is_partially_unvalued_not_zero() -> None:
    engine = funded()
    engine.account_trusted_fill(fill("BUY", "1", "100", number=120))
    assert nav(engine, accepted_valuation_fixture([]), USD, portfolio_scope(PAPER)) == (
        "PARTIALLY_UNVALUED",
        None,
    )


def snapshot_raw(
    scope: Scope, asset_: AssetReference, quantity_: str, number: int
) -> dict[str, Any]:
    raw = {
        "workspace_id": scope.workspace_id,
        "portfolio_id": scope.portfolio_id,
        "environment": scope.environment,
        "exchange_account_id": scope.exchange_account_id,
        "asset_reference": asset_.object(),
        "observed_quantity": quantity_,
        "as_of_utc": T,
        "source_id": ident("snap", number),
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    return raw


def test_observed_snapshot_match_drift_and_zero_mutation() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "10", 130)
    before = copy.deepcopy(engine.entries)
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "10", 1))) == "MATCH"
    )
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "12", 2))) == "DRIFT"
        and engine.entries == before
    )
    assert reconcile(engine, snapshot_raw(PAPER, USD, "12", 2)) == "TRUSTED_CONTEXT_FAILURE"


def reversal_fact(
    engine: Engine, target_source: str, target_identity: str, number: int
) -> AccountingEconomicFact:
    target = engine.accepted_source_history[(target_source, target_identity)]
    raw = {
        "audit_event_id": ident("evt", number),
        "source_type": "reconciliation_correction",
        "workspace_id": PAPER.workspace_id,
        "portfolio_id": PAPER.portfolio_id,
        "environment": PAPER.environment,
        "effective_at_utc": T,
        "provenance": "Core accepted correction review",
        "reason": "reverse exact erroneous batch",
        "target_source_type": target.source_type,
        "target_accounting_source_identity": target.identity,
        "target_source_fingerprint_sha256": target.fingerprint,
        "target_batch_fingerprint_sha256": target.batch_fingerprint,
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    return economic_fact(raw, preexisting_accounting_projection((freeze(raw),)))


def test_trusted_reconciliation_exact_reversal_is_append_only() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "10", 140)
    before = copy.deepcopy(engine.entries)
    correction = reversal_fact(engine, "deposit", ident("evt", 140), 141)
    assert engine.account_fact(correction) == "ACCEPTED"
    assert engine.entries[: len(before)] == before and engine._available(PAPER, USD) == 0
    count = len(engine.entries)
    assert engine.account_fact(correction) == "REPLAY_SUCCESS" and len(engine.entries) == count


def test_paper_testnet_live_account_environment_isolation() -> None:
    engine = Engine()
    testnet = Scope(PAPER.workspace_id, PAPER.portfolio_id, "TESTNET", ident("xacc", 2))
    live = Scope(PAPER.workspace_id, PAPER.portfolio_id, "LIVE", ident("xacc", 3))
    assert deposit(engine, PAPER, USD, "1", 150) == "ACCEPTED"
    assert deposit(engine, testnet, USD, "2", 151) == "ACCEPTED"
    assert deposit(engine, live, USD, "3", 152) == "ACCEPTED"
    assert (
        engine._available(PAPER, USD) == 1
        and engine._available(testnet, USD) == 2
        and engine._available(live, USD) == 3
    )
    keys = engine.balances()
    assert len(keys) == 3 and {key[2] for key in keys} == {"PAPER", "TESTNET", "LIVE"}


def test_fill_account_a_never_mutates_account_b() -> None:
    engine = funded()
    account_b = Scope(PAPER.workspace_id, PAPER.portfolio_id, "PAPER", ident("xacc", 2))
    deposit(engine, account_b, USD, "500", 160)
    before = engine._available(account_b, USD)
    assert engine.account_trusted_fill(fill("BUY", "1", "100", number=161)) == "ACCEPTED"
    assert engine._available(account_b, USD) == before and engine._available(PAPER, BTC) == 1


def test_decimal_context_independence_and_no_float() -> None:
    with localcontext() as context:
        context.prec = 6
        engine = funded()
        assert (
            engine.account_trusted_fill(fill("BUY", "0.333333333333333333", "3", number=170))
            == "ACCEPTED"
        )
        assert engine._available(PAPER, BTC) == Fraction(333333333333333333, 10**18)
    assert all(type(entry.quantity) is str for entry in engine.entries)


def test_same_time_and_late_fill_use_append_sequence_without_rewrite() -> None:
    engine = funded()
    a = fill("BUY", "1", "100", number=180)
    b = fill("BUY", "1", "110", number=181)
    assert engine.account_trusted_fill(a) == "ACCEPTED"
    snapshot = copy.deepcopy(engine.entries)
    assert engine.account_trusted_fill(b) == "ACCEPTED"
    assert engine.entries[: len(snapshot)] == snapshot and engine.fills[0][0] < engine.fills[1][0]


@pytest.mark.parametrize(
    "root,value",
    [
        ("schema_version", "0"),
        ("m0_element", "M0.9"),
        ("status", "under audit"),
        ("authority", "UI"),
        ("source_of_truth", ["mutable venue balance"]),
        ("contract_inconsistent_scope", "all failures"),
        ("closure_conditions", "best effort"),
    ],
)
def test_top_level_contract_semantic_mutations_fail(root: str, value: Any) -> None:
    changed = copy.deepcopy(CONTRACT)
    changed[root] = value
    assert validate_contract(changed) == "CONTRACT_INCONSISTENT"


def test_complete_canonical_m07_fill_is_accepted_and_fingerprint_is_source() -> None:
    context = fill("BUY", "1", "100", number=300)
    assert set(context.fill) == set(
        CANONICAL["commands_events_order_lifecycle_and_idempotency.json"]["fill_contract"][
            "fact_fields"
        ]
    )
    assert context.source_fingerprint == context.fill["fill_fingerprint_sha256"]
    assert context.fill["venue_trade_id"] == "paper-trade-300"


@pytest.mark.parametrize("missing", ["venue_trade_id", "execution_route_id", "exchange_id"])
def test_incomplete_m07_full_fill_is_rejected(missing: str) -> None:
    raw = full_fill_fact("BUY", "1", "100", number=301)
    raw.pop(missing)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_fill(
            raw,
            copy.deepcopy(INSTRUMENT),
            CoreAcceptedFillProjection(raw["order_id"], (raw["fill_id"],)),
        )


@pytest.mark.parametrize("tamper", ["fingerprint", "fee", "price", "time"])
def test_m07_fill_economics_tamper_under_old_fingerprint_is_rejected(tamper: str) -> None:
    raw = full_fill_fact("BUY", "1", "100", number=302)
    if tamper == "fingerprint":
        raw["fill_fingerprint_sha256"] = "0" * 64
    elif tamper == "fee":
        raw["fee_kind"], raw["fee_quantity"], raw["fee_asset_reference"] = (
            "CHARGE",
            "1",
            USD.object(),
        )
    elif tamper == "price":
        raw["execution_price"] = "101"
    else:
        raw["executed_at_utc"] = "2026-01-01T00:00:01Z"
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_fill(
            raw,
            copy.deepcopy(INSTRUMENT),
            CoreAcceptedFillProjection(raw["order_id"], (raw["fill_id"],)),
        )


def test_old_raw_subset_fill_factory_shape_is_rejected() -> None:
    subset = {
        key: value
        for key, value in full_fill_fact("BUY", "1", "100", number=303).items()
        if key
        not in {"exchange_id", "execution_route_id", "venue_trade_id", "fill_fingerprint_sha256"}
    }
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_fill(
            subset,
            copy.deepcopy(INSTRUMENT),
            CoreAcceptedFillProjection(subset["order_id"], (subset["fill_id"],)),
        )


@pytest.mark.parametrize(
    "source,extra",
    [
        ("deposit", {"capital_flow_kind": "INTERNAL_ARRIVAL"}),
        ("deposit", {"capital_flow_kind": "anything"}),
        ("withdrawal", {"capital_flow_kind": "INTERNAL_DEPARTURE"}),
    ],
)
def test_source_specific_non_fill_constraints_reject(source: str, extra: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        fact(source, PAPER, USD, "1", 310, **extra)


def test_empty_non_fill_provenance_is_rejected() -> None:
    raw = dict(
        fact("deposit", PAPER, USD, "1", 311, capital_flow_kind="EXTERNAL_CONTRIBUTION").payload
    )
    raw["provenance"] = ""
    raw["source_fingerprint_sha256"] = digest(
        {key: value for key, value in raw.items() if key != "source_fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(raw)


def test_standalone_fee_is_reserved_unsupported_and_cannot_duplicate_fill_fee() -> None:
    engine = funded()
    context = fill(
        "BUY", "1", "100", number=312, fee_kind="CHARGE", fee_quantity="1", fee_asset=USD
    )
    assert engine.account_trusted_fill(context) == "ACCEPTED"
    before = copy.deepcopy(engine.entries)
    with pytest.raises(ValueError, match="UNSUPPORTED_ACCOUNTING_SEMANTICS"):
        fact("fee", PAPER, USD, "1", 313, reason="duplicate Fill fee")
    assert engine.entries == before


def test_three_lot_fifo_partial_sell_preserves_all_trailing_lots() -> None:
    engine = funded()
    for number, quantity_, price in ((320, "1", "100"), (321, "2", "120"), (322, "3", "130")):
        assert (
            engine.account_trusted_fill(fill("BUY", quantity_, price, number=number)) == "ACCEPTED"
        )
    sell = fill("SELL", "1.5", "150", number=323)
    assert engine.account_trusted_fill(sell) == "ACCEPTED"
    expected = [
        (Fraction(3, 2), Fraction(120), USD, ident("fill", 321)),
        (Fraction(3), Fraction(130), USD, ident("fill", 322)),
    ]
    assert lot_values(engine.lots(PAPER, BTC)) == expected
    snapshot = copy.deepcopy(expected)
    assert engine.account_trusted_fill(sell) == "REPLAY_SUCCESS"
    assert lot_values(engine.lots(PAPER, BTC)) == snapshot


def test_fifo_exact_boundary_preserves_later_lots_then_full_close() -> None:
    engine = funded()
    for number, quantity_, price in ((330, "1", "100"), (331, "2", "120"), (332, "3", "130")):
        engine.account_trusted_fill(fill("BUY", quantity_, price, number=number))
    engine.account_trusted_fill(fill("SELL", "3", "150", number=333))
    assert lot_values(engine.lots(PAPER, BTC)) == [
        (Fraction(3), Fraction(130), USD, ident("fill", 332))
    ]
    engine.account_trusted_fill(fill("SELL", "3", "140", number=334))
    assert engine.lots(PAPER, BTC) == []


def test_buy_base_fee_greater_than_execution_is_zero_append() -> None:
    engine = funded()
    before = copy.deepcopy(engine.entries)
    context = fill(
        "BUY", "1", "100", number=335, fee_kind="CHARGE", fee_quantity="1.1", fee_asset=BTC
    )
    assert engine.account_trusted_fill(context) == "INSUFFICIENT_AVAILABLE_CAPITAL"
    assert engine.entries == before and engine.lots(PAPER, BTC) == []


def test_nav_is_portfolio_environment_scoped_and_never_authoritative_600() -> None:
    engine = Engine()
    testnet = Scope(PAPER.workspace_id, PAPER.portfolio_id, "TESTNET", ident("xacc", 401))
    live = Scope(PAPER.workspace_id, PAPER.portfolio_id, "LIVE", ident("xacc", 402))
    other = Scope(ident("ws", 2), ident("port", 2), "PAPER", ident("xacc", 403))
    deposit(engine, PAPER, USD, "100", 400)
    deposit(engine, testnet, USD, "200", 401)
    deposit(engine, live, USD, "300", 402)
    deposit(engine, other, USD, "700", 403)
    context = accepted_valuation_fixture([])
    values = {
        nav(engine, context, USD, portfolio_scope(scope))[1] for scope in (PAPER, testnet, live)
    }
    assert values == {Fraction(100), Fraction(200), Fraction(300)}
    assert Fraction(600) not in values
    assert nav(engine, context, USD, portfolio_scope(other)) == ("COMPLETE", 700)


def test_executable_realized_fee_net_and_unrealized_pnl() -> None:
    engine = funded()
    engine.account_trusted_fill(fill("BUY", "1", "100", number=410))
    journal = copy.deepcopy(engine.entries)
    mark120 = accepted_valuation_fixture([valuation_edge(BTC, USD, "120", number=20)])
    mark150 = accepted_valuation_fixture([valuation_edge(BTC, USD, "150", number=21)])
    assert project_unrealized_pnl(engine, portfolio_scope(PAPER), mark120, USD) == ("COMPLETE", 20)
    assert project_unrealized_pnl(engine, portfolio_scope(PAPER), mark150, USD) == ("COMPLETE", 50)
    assert engine.entries == journal
    engine.account_trusted_fill(
        fill("SELL", "0.5", "150", number=411, fee_kind="CHARGE", fee_quantity="1", fee_asset=USD)
    )
    realized = project_realized_pnl(engine, portfolio_scope(PAPER), mark150, USD)
    assert realized == RealizedPnlProjection("COMPLETE", Fraction(25), Fraction(1), Fraction(24))
    assert project_unrealized_pnl(engine, portfolio_scope(PAPER), mark150, USD) == (
        "COMPLETE",
        Fraction(25),
    )


def test_third_asset_fee_without_valuation_makes_net_pnl_incomplete() -> None:
    engine = funded()
    deposit(engine, PAPER, BNB, "1", 420)
    engine.account_trusted_fill(
        fill("BUY", "1", "100", number=421, fee_kind="CHARGE", fee_quantity="0.01", fee_asset=BNB)
    )
    result = project_realized_pnl(
        engine, portfolio_scope(PAPER), accepted_valuation_fixture([]), USD
    )
    assert (
        result.status == "MISSING_VALUATION"
        and result.fee_effect is None
        and result.net_realized is None
    )


def test_capital_flows_do_not_change_trading_pnl_projection() -> None:
    engine = Engine()
    context = accepted_valuation_fixture([])
    before = project_realized_pnl(engine, portfolio_scope(PAPER), context, USD)
    deposit(engine, PAPER, USD, "100", 430)
    engine.account_fact(
        fact("withdrawal", PAPER, USD, "20", 431, capital_flow_kind="EXTERNAL_WITHDRAWAL")
    )
    assert project_realized_pnl(engine, portfolio_scope(PAPER), context, USD) == before


def test_accounting_source_identity_must_equal_durable_reference() -> None:
    deposit_entry = valid_entry()
    assert validate_ledger_entry(deposit_entry, 0) == "VALID"
    deposit_entry["accounting_source_identity"] = ident("evt", 99)
    assert validate_ledger_entry(deposit_entry, 0) == "MALFORMED_ACCOUNTING_FACT"
    engine = funded()
    engine.account_trusted_fill(fill("BUY", "1", "100", number=440))
    fill_entry = asdict(
        next(entry for entry in engine.entries if entry.fill_id == ident("fill", 440))
    )
    assert validate_ledger_entry(fill_entry, fill_entry["append_sequence"] - 1) == "VALID"
    fill_entry["accounting_source_identity"] = ident("fill", 999)
    assert (
        validate_ledger_entry(fill_entry, fill_entry["append_sequence"] - 1)
        == "MALFORMED_ACCOUNTING_FACT"
    )


def test_self_hashed_arbitrary_valuation_source_has_no_authority() -> None:
    accepted = valuation_edge(BTC, USD, "100", number=10)
    authority = preexisting_valuation_projection([accepted])
    tampered = copy.deepcopy(accepted)
    tampered["rate"] = "101"
    tampered["source_fingerprint_sha256"] = digest(
        {key: value for key, value in tampered.items() if key != "source_fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_valuation_edges([tampered], authority, tampered["as_of_utc"])
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_valuation_edges([accepted], cast(Any, {}), accepted["as_of_utc"])


def test_valuation_path_fresh_alternative_beats_stale_and_cycle() -> None:
    eth = AssetReference.trusted(asset("ETH"))
    stale_and_fresh = accepted_valuation_fixture(
        [
            valuation_edge(BTC, USD, "90", stale=True, number=30),
            valuation_edge(BTC, USD, "100", number=31),
        ]
    )
    assert resolve_rate(stale_and_fresh, BTC, USD) == ("COMPLETE", 100)
    cycle_and_valid = accepted_valuation_fixture(
        [
            valuation_edge(BTC, eth, "2", number=32),
            valuation_edge(eth, BTC, "0.5", number=33),
            valuation_edge(BTC, USD, "110", number=34),
        ]
    )
    assert resolve_rate(cycle_and_valid, BTC, USD) == ("COMPLETE", 110)


def test_two_fresh_valuation_paths_have_fingerprint_deterministic_winner() -> None:
    first = valuation_edge(BTC, USD, "100", number=35)
    second = valuation_edge(BTC, USD, "101", number=36)
    context = accepted_valuation_fixture([second, first])
    expected = Fraction(
        first["rate"]
        if first["source_fingerprint_sha256"] < second["source_fingerprint_sha256"]
        else second["rate"]
    )
    assert resolve_rate(context, BTC, USD) == ("COMPLETE", expected)


def test_reconciliation_declared_outcomes_are_all_executable() -> None:
    engine = Engine()
    assert reconcile(engine, None) == "MISSING_EXTERNAL_FACT"
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "0", 50)))
        == "MISSING_INTERNAL_FACT"
    )
    deposit(engine, PAPER, USD, "1", 450)
    engine.account_fact(
        fact("withdrawal", PAPER, USD, "1", 451, capital_flow_kind="EXTERNAL_WITHDRAWAL")
    )
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "0", 51))) == "MATCH"
    )
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "2", 52))) == "DRIFT"
    )
    unmapped = snapshot_raw(PAPER, USD, "0", 53)
    unmapped["asset_reference"]["mapping_status"] = "UNKNOWN"
    unmapped["source_fingerprint_sha256"] = digest(
        {key: value for key, value in unmapped.items() if key != "source_fingerprint_sha256"}
    )
    assert reconcile(engine, accepted_observed_fixture(unmapped)) == "UNMAPPED_ASSET"
    assert (
        reconcile(engine, accepted_observed_fixture(snapshot_raw(PAPER, USD, "0", 63)))
        == "UNSUPPORTED"
    )


def test_order_bound_reservation_partial_fills_and_terminal_release() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "100", 500)
    reserve = reservation(PAPER, "100", 501, 510, command_suffix=511)
    assert engine.account_fact(reserve) == "ACCEPTED"
    first = fill("BUY", "0.4", "100", number=510)
    assert engine.account_trusted_fill(first) == "ACCEPTED"
    assert engine._available(PAPER, USD) == 0 and engine._reserved(PAPER, USD) == 60
    second_raw = full_fill_fact("BUY", "0.3", "100", number=512)
    second_raw["order_id"] = ident("ord", 510)
    second_raw["fill_fingerprint_sha256"] = canonical_m07_fill_fingerprint(second_raw)
    second = attest_m07_accepted_fill(
        second_raw,
        copy.deepcopy(INSTRUMENT),
        CoreAcceptedFillProjection(
            second_raw["order_id"], (first.fill["fill_id"], second_raw["fill_id"])
        ),
    )
    assert engine.account_trusted_fill(second) == "ACCEPTED"
    assert engine._reserved(PAPER, USD) == 30
    release = terminal_release(510, 513, "CANCELLED")
    assert engine.account_fact(release) == "ACCEPTED"
    assert engine._available(PAPER, USD) == 30 and engine._reserved(PAPER, USD) == 0
    count = len(engine.entries)
    assert engine.account_fact(release) == "REPLAY_SUCCESS" and len(engine.entries) == count


def test_other_order_cannot_consume_reserved_capital() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "100", 520)
    engine.account_fact(reservation(PAPER, "100", 521, 522))
    before = copy.deepcopy(engine.entries)
    assert (
        engine.account_trusted_fill(fill("BUY", "0.4", "100", number=523))
        == "INSUFFICIENT_AVAILABLE_CAPITAL"
    )
    assert engine.entries == before and engine._reserved(PAPER, USD) == 100


def test_command_identity_dedupes_reservation_across_new_audit_event() -> None:
    engine = Engine()
    deposit(engine, PAPER, USD, "100", 530)
    first = reservation(PAPER, "50", 531, 532, command_suffix=533, audit_suffix=534)
    duplicate = reservation(PAPER, "50", 531, 532, command_suffix=533, audit_suffix=535)
    assert engine.account_fact(first) == "ACCEPTED"
    count = len(engine.entries)
    assert engine.account_fact(duplicate) == "REPLAY_SUCCESS" and len(engine.entries) == count
    changed = reservation(PAPER, "60", 531, 532, command_suffix=533, audit_suffix=536)
    assert engine.account_fact(changed) == "ACCOUNTING_IDENTITY_CONFLICT"


def test_inventory_deposit_basis_sell_withdraw_transfer_and_third_fee() -> None:
    engine = funded()
    account_b = Scope(PAPER.workspace_id, PAPER.portfolio_id, "PAPER", ident("xacc", 540))
    btc_deposit = fact(
        "deposit",
        PAPER,
        BTC,
        "2",
        541,
        capital_flow_kind="EXTERNAL_CONTRIBUTION",
        basis_valuation_unit=USD.object(),
        unit_cost_basis="90",
    )
    assert engine.account_fact(btc_deposit) == "ACCEPTED"
    assert engine.account_trusted_fill(fill("SELL", "1", "120", number=542)) == "ACCEPTED"
    pnl = [
        entry
        for entry in engine.entries
        if entry.fill_id == ident("fill", 542)
        and entry.account_role == "REALIZED_PNL_CLASSIFICATION"
    ]
    assert len(pnl) == 1 and decimal(pnl[0].quantity) == 30
    withdrawal = fact(
        "withdrawal",
        PAPER,
        BTC,
        "0.25",
        543,
        capital_flow_kind="EXTERNAL_WITHDRAWAL",
        basis_valuation_unit=USD.object(),
        unit_cost_basis="90",
    )
    assert engine.account_fact(withdrawal) == "ACCEPTED"
    assert engine.account_fact(transfer_fact(PAPER, account_b, BTC, BTC, "0.5", 544)) == "ACCEPTED"
    assert lot_values(engine.lots(PAPER, BTC)) == [
        (Fraction(1, 4), Fraction(90), USD, ident("evt", 541))
    ]
    assert lot_values(engine.lots(account_b, BTC)) == [
        (Fraction(1, 2), Fraction(90), USD, ident("evt", 541))
    ]
    total_basis = sum(
        amount * basis
        for scope_ in (PAPER, account_b)
        for amount, basis in (
            (lot.quantity, lot.unit_cost_basis) for lot in engine.lots(scope_, BTC)
        )
    )
    assert total_basis == Fraction(135, 2)
    bnb = fact(
        "deposit",
        PAPER,
        BNB,
        "1",
        545,
        capital_flow_kind="EXTERNAL_CONTRIBUTION",
        basis_valuation_unit=USD.object(),
        unit_cost_basis="10",
    )
    engine.account_fact(bnb)
    engine.account_trusted_fill(
        fill("BUY", "0.1", "100", number=546, fee_kind="CHARGE", fee_quantity="0.25", fee_asset=BNB)
    )
    assert lot_values(engine.lots(PAPER, BNB)) == [
        (Fraction(3, 4), Fraction(10), USD, ident("evt", 545))
    ]


def test_full_inventory_removal_and_base_fee_equal_quantity_leave_no_lot() -> None:
    engine = funded()
    assert (
        engine.account_trusted_fill(
            fill("BUY", "1", "100", number=550, fee_kind="CHARGE", fee_quantity="1", fee_asset=BTC)
        )
        == "ACCEPTED"
    )
    assert engine.lots(PAPER, BTC) == [] and engine._available(PAPER, BTC) == 0


def test_rebuild_from_journal_and_accepted_history_without_fill_cache() -> None:
    engine = funded()
    deposit(engine, PAPER, BNB, "1", 560)
    engine.account_trusted_fill(
        fill("BUY", "1", "100", number=561, fee_kind="CHARGE", fee_quantity="0.1", fee_asset=BNB)
    )
    engine.account_trusted_fill(fill("SELL", "0.5", "130", number=562))
    context = accepted_valuation_fixture(
        [valuation_edge(BTC, USD, "140", number=40), valuation_edge(BNB, USD, "10", number=41)]
    )
    before = (
        engine.balances(),
        engine.lots(PAPER, BTC),
        project_realized_pnl(engine, portfolio_scope(PAPER), context, USD),
        project_unrealized_pnl(engine, portfolio_scope(PAPER), context, USD),
    )
    assert "fills" not in engine.__dict__
    rebuilt = Engine.rebuild(copy.deepcopy(engine.entries), dict(engine.accepted_source_history))
    after = (
        rebuilt.balances(),
        rebuilt.lots(PAPER, BTC),
        project_realized_pnl(rebuilt, portfolio_scope(PAPER), context, USD),
        project_unrealized_pnl(rebuilt, portfolio_scope(PAPER), context, USD),
    )
    assert after == before


def test_unknown_drift_never_adjusts_and_missing_deposit_uses_deposit() -> None:
    engine = Engine()
    observation = accepted_observed_fixture(snapshot_raw(PAPER, USD, "10", 55))
    before = copy.deepcopy(engine.entries)
    assert reconcile(engine, observation) == "MISSING_INTERNAL_FACT" and engine.entries == before
    deposit(engine, PAPER, USD, "10", 570)
    assert reconcile(engine, observation) == "MATCH"
    assert not any(entry.source_type == "reconciliation_correction" for entry in engine.entries)


def test_correction_without_exact_target_rejected_and_exact_reversal_preserves_inventory() -> None:
    engine = Engine()
    engine.account_fact(
        fact(
            "deposit",
            PAPER,
            BTC,
            "1",
            580,
            capital_flow_kind="EXTERNAL_CONTRIBUTION",
            basis_valuation_unit=USD.object(),
            unit_cost_basis="100",
        )
    )
    before = copy.deepcopy(engine.entries)
    raw = {
        "audit_event_id": ident("evt", 581),
        "source_type": "reconciliation_correction",
        "workspace_id": PAPER.workspace_id,
        "portfolio_id": PAPER.portfolio_id,
        "environment": PAPER.environment,
        "effective_at_utc": T,
        "provenance": "review",
        "reason": "bad target",
        "target_source_type": "deposit",
        "target_accounting_source_identity": ident("evt", 999),
        "target_source_fingerprint_sha256": "1" * 64,
        "target_batch_fingerprint_sha256": "2" * 64,
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    malformed = economic_fact(raw, preexisting_accounting_projection((freeze(raw),)))
    assert engine.account_fact(malformed) == "TRUSTED_CONTEXT_FAILURE" and engine.entries == before
    correction = reversal_fact(engine, "deposit", ident("evt", 580), 582)
    assert engine.account_fact(correction) == "ACCEPTED"
    assert engine._available(PAPER, BTC) == 0 and engine.lots(PAPER, BTC) == []


@pytest.mark.parametrize("source", ["fee", "funding", "interest", "realized_pnl"])
def test_unsupported_durable_source_entries_are_malformed(source: str) -> None:
    item = valid_entry()
    item["source_type"] = source
    assert validate_ledger_entry(item, 0) == "MALFORMED_ACCOUNTING_FACT"


@pytest.mark.parametrize(
    "source,role,posting_role,direction",
    [
        ("deposit", "TRANSFER_CLEARING", "TRANSFER_CLEARING_SOURCE", "DEBIT"),
        ("withdrawal", "OWNED_RESERVED", "RESERVE_HELD", "CREDIT"),
        ("internal_transfer", "EXTERNAL_CAPITAL", "CAPITAL_CLASSIFIED", "CREDIT"),
        ("capital_reservation", "FEE_EXPENSE", "FEE_CLASSIFIED", "DEBIT"),
    ],
)
def test_cross_source_posting_substitution_is_malformed(
    source: str, role: str, posting_role: str, direction: str
) -> None:
    item = valid_entry()
    item.update(
        {
            "source_type": source,
            "account_role": role,
            "posting_role": posting_role,
            "direction": direction,
        }
    )
    assert validate_ledger_entry(item, 0) == "MALFORMED_ACCOUNTING_FACT"


def test_fill_replay_verifies_stored_batch_and_journal_integrity() -> None:
    engine = funded()
    context = fill("BUY", "1", "100", number=590)
    assert engine.account_trusted_fill(context) == "ACCEPTED"
    count = len(engine.entries)
    record = engine.accepted_source_history[("fill", ident("fill", 590))]
    engine.accepted_source_history[("fill", ident("fill", 590))] = AcceptedSourceRecord(
        record.source_type,
        record.identity,
        record.fingerprint,
        "0" * 64,
        record.first_sequence,
        record.context,
    )
    snapshot = copy.deepcopy(engine.entries)
    assert engine.account_trusted_fill(context) == "CONTRACT_INCONSISTENT"
    assert engine.entries == snapshot and len(engine.entries) == count


def test_raw_self_hashed_non_fill_and_snapshot_mutations_lack_authority() -> None:
    accepted = dict(
        fact("deposit", PAPER, USD, "1", 600, capital_flow_kind="EXTERNAL_CONTRIBUTION").payload
    )
    authority = preexisting_accounting_projection((freeze(accepted),))
    accepted["quantity"] = "2"
    accepted["source_fingerprint_sha256"] = digest(
        {key: value for key, value in accepted.items() if key != "source_fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(accepted, authority)
    snapshot = snapshot_raw(PAPER, USD, "1", 56)
    snapshot_authority = preexisting_observed_projection(snapshot)
    snapshot["observed_quantity"] = "2"
    snapshot["source_fingerprint_sha256"] = digest(
        {key: value for key, value in snapshot.items() if key != "source_fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_observed_context(snapshot, snapshot_authority)


def test_shared_reporting_asof_and_valuation_failure_preservation() -> None:
    edge = valuation_edge(BTC, USD, "120", number=57)
    authority = preexisting_valuation_projection([edge])
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_valuation_edges([edge], authority, "2026-01-01T00:00:03Z")
    engine = funded()
    engine.account_trusted_fill(fill("BUY", "1", "100", number=610))
    stale = accepted_valuation_fixture([valuation_edge(BTC, USD, "120", stale=True, number=58)])
    assert (
        project_unrealized_pnl(engine, portfolio_scope(PAPER), stale, USD)[0] == "STALE_VALUATION"
    )
    eth = AssetReference.trusted(asset("ETH"))
    cycle = accepted_valuation_fixture(
        [valuation_edge(BTC, eth, "2", number=59), valuation_edge(eth, BTC, "0.5", number=60)]
    )
    assert (
        project_unrealized_pnl(engine, portfolio_scope(PAPER), cycle, USD)[0]
        == "UNSUPPORTED_VALUATION_PATH"
    )
    zero = Engine()
    deposit(zero, PAPER, BNB, "1", 611)
    zero.account_fact(
        fact("withdrawal", PAPER, BNB, "1", 612, capital_flow_kind="EXTERNAL_WITHDRAWAL")
    )
    assert nav(zero, accepted_valuation_fixture([]), USD, portfolio_scope(PAPER)) == ("COMPLETE", 0)


def test_fill_replay_detects_changed_stored_posting_semantics() -> None:
    engine = funded()
    context = fill("BUY", "1", "100", number=620)
    assert engine.account_trusted_fill(context) == "ACCEPTED"
    index = next(i for i, entry in enumerate(engine.entries) if entry.fill_id == ident("fill", 620))
    changed = replace(engine.entries[index], posting_role="FEE_PAID")
    engine.entries[index] = changed
    key = ("fill", ident("fill", 620), RULE)
    fingerprint, batch, accepted_entries = engine.accepted[key]
    changed_tuple = tuple(
        changed if entry.ledger_entry_id == changed.ledger_entry_id else entry
        for entry in accepted_entries
    )
    engine.accepted[key] = (fingerprint, batch, changed_tuple)
    snapshot = copy.deepcopy(engine.entries)
    assert engine.account_trusted_fill(context) == "CONTRACT_INCONSISTENT"
    assert engine.entries == snapshot


def test_invalid_order_command_and_terminal_state_rejected_at_source_boundary() -> None:
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        fact(
            "capital_reservation",
            PAPER,
            USD,
            "1",
            630,
            order_id="ord_bad",
            command_id=ident("cmd", 630),
        )
    raw = {
        "audit_event_id": ident("evt", 631),
        "source_type": "capital_release",
        "workspace_id": PAPER.workspace_id,
        "portfolio_id": PAPER.portfolio_id,
        "environment": PAPER.environment,
        "exchange_account_id": PAPER.exchange_account_id,
        "effective_at_utc": T,
        "provenance": "terminal",
        "order_id": ident("ord", 631),
        "terminal_state": "OPEN",
    }
    raw["source_fingerprint_sha256"] = digest(raw)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(raw, preexisting_accounting_projection((freeze(raw),)))


def test_reservation_requires_nominal_exact_submit_order_authority() -> None:
    accepted = reservation(PAPER, "1", 640, 640, command_suffix=641)
    raw = dict(accepted.payload)
    original = cast(M07PrevalidatedAcceptedCommandContext, accepted.upstream_context)
    wrong_request = dict(original.request)
    wrong_request["order_id"] = ident("ord", 999)
    wrong_fingerprint = m07_command_fingerprint(wrong_request)
    wrong = attest_m07_accepted_submit_order(
        wrong_request,
        CoreAcceptedCommandProjection(freeze({raw["command_id"]: wrong_fingerprint})),
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(raw, preexisting_accounting_projection((freeze(raw),)), wrong)


def test_cross_unit_fifo_requires_and_uses_explicit_historical_conversion() -> None:
    eur = AssetReference.trusted(asset("EUR"))
    engine = funded()
    assert (
        engine.account_fact(
            fact(
                "deposit",
                PAPER,
                BTC,
                "1",
                650,
                capital_flow_kind="EXTERNAL_CONTRIBUTION",
                basis_valuation_unit=eur.object(),
                unit_cost_basis="90",
            )
        )
        == "ACCEPTED"
    )
    sell = fill("SELL", "1", "120", number=651)
    before = copy.deepcopy(engine.entries)
    assert engine.account_trusted_fill(sell) == "MISSING_VALUATION"
    assert engine.entries == before
    conversion = accepted_valuation_fixture([valuation_edge(eur, USD, "2", number=66)])
    assert engine.account_trusted_fill(sell, conversion) == "ACCEPTED"
    pnl = next(
        entry
        for entry in engine.entries
        if entry.fill_id == ident("fill", 651)
        and entry.account_role == "REALIZED_PNL_CLASSIFICATION"
    )
    assert pnl.direction == "CREDIT" and decimal(pnl.quantity) == 60
    rebuilt = Engine.rebuild(copy.deepcopy(engine.entries), dict(engine.accepted_source_history))
    assert rebuilt.lots(PAPER, BTC) == [] and rebuilt.balances() == engine.balances()


def test_coordinated_journal_and_source_record_tamper_is_detected_by_rederivation() -> None:
    engine = funded()
    target_entries = engine.entries[:2]
    attacker_projections = [
        PostingProjection(
            Scope(
                entry.workspace_id,
                entry.portfolio_id,
                entry.environment,
                cast(str, entry.exchange_account_id),
            ),
            AssetReference.trusted(entry.asset_reference),
            entry.account_role,
            entry.direction,
            "999",
            entry.posting_role,
        )
        for entry in target_entries
    ]
    attacker_batch = digest(
        [posting.canonical_projection(index) for index, posting in enumerate(attacker_projections)]
    )
    entries = [
        replace(entry, quantity="999", batch_fingerprint_sha256=attacker_batch)
        for entry in target_entries
    ] + engine.entries[2:]
    record = engine.accepted_source_history[("deposit", ident("evt", 10))]
    history = dict(engine.accepted_source_history)
    history[("deposit", ident("evt", 10))] = replace(record, batch_fingerprint=attacker_batch)
    with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
        Engine.rebuild(entries, history)


def test_fill_correction_is_single_use_and_restores_consumed_reservation() -> None:
    engine = funded("100")
    assert engine.account_fact(reservation(PAPER, "100", 660, 661)) == "ACCEPTED"
    accepted_fill = fill("BUY", "0.4", "100", number=661)
    assert engine.account_trusted_fill(accepted_fill) == "ACCEPTED"
    assert engine.reservations[ident("ord", 661)].remaining == 60
    first = reversal_fact(engine, "fill", ident("fill", 661), 662)
    assert engine.account_fact(first) == "ACCEPTED"
    assert engine.reservations[ident("ord", 661)].remaining == 100
    second = reversal_fact(engine, "fill", ident("fill", 661), 663)
    before = copy.deepcopy(engine.entries)
    assert engine.account_fact(second) == "ACCOUNTING_IDENTITY_CONFLICT"
    assert engine.entries == before


def test_real_m07_submit_order_schema_membership_and_tamper_boundaries() -> None:
    reservation_fact = reservation(PAPER, "7", 700, 701, command_suffix=702)
    context = cast(M07PrevalidatedAcceptedCommandContext, reservation_fact.upstream_context)
    assert set(context.request) == M07_SUBMIT_ORDER_FIELDS
    assert set(EXPECTED_PROTOCOLS["m07_authority_boundary"]["submit_order_consumed_fields"]) == (
        M07_SUBMIT_ORDER_FIELDS
    )
    assert context.request["operation_type"] == "SUBMIT_ORDER"
    assert (
        "reservation_asset" not in context.request and "reservation_quantity" not in context.request
    )
    assert context.request["quantity"] == "1" and reservation_fact.payload["quantity"] == "7"
    assert Engine(entries=[]).account_fact(reservation_fact) == "INSUFFICIENT_AVAILABLE_CAPITAL"

    cancel = dict(context.request)
    cancel["operation_type"] = "CANCEL_ORDER"
    cancel_fingerprint = m07_command_fingerprint(cancel)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_submit_order(
            cancel,
            CoreAcceptedCommandProjection(freeze({cancel["command_id"]: cancel_fingerprint})),
        )
    changed = dict(context.request)
    changed["quantity"] = "2"
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_submit_order(changed, context.authority)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_submit_order(
            dict(context.request), CoreAcceptedCommandProjection(freeze({}))
        )
    wrong_idempotency = dict(context.request)
    wrong_idempotency["idempotency_key"] = ident("cmd", 999)
    wrong_idempotency_fingerprint = m07_command_fingerprint(wrong_idempotency)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_submit_order(
            wrong_idempotency,
            CoreAcceptedCommandProjection(
                freeze({wrong_idempotency["command_id"]: wrong_idempotency_fingerprint})
            ),
        )


def test_fabricated_m07_command_wrapper_and_missing_source_evidence_are_rejected() -> None:
    accepted = reservation(PAPER, "1", 710, 711)
    genuine = cast(M07PrevalidatedAcceptedCommandContext, accepted.upstream_context)
    fabricated = M07PrevalidatedAcceptedCommandContext(
        genuine.request, genuine.command_fingerprint, genuine.authority, object()
    )
    raw = dict(accepted.payload)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(raw, preexisting_accounting_projection((freeze(raw),)), fabricated)
    stripped = replace(accepted, upstream_context=None)
    assert Engine().account_fact(stripped) == "TRUSTED_CONTEXT_FAILURE"


def test_terminal_event_authority_exact_type_scope_fingerprint_and_history() -> None:
    accepted = terminal_release(720, 721)
    context = cast(M07PrevalidatedAcceptedTerminalOrderEventContext, accepted.upstream_context)
    assert context.event["event_type"] == "ORDER_CANCEL_CONFIRMED"
    assert context.terminal_state == "CANCELLED"
    assert context.event["exchange_account_id"] == accepted.payload["exchange_account_id"]

    for field, value in (
        ("exchange_account_id", ident("xacc", 999)),
        ("environment", "TESTNET"),
    ):
        event = dict(context.event)
        event[field] = value
        event["event_fingerprint_sha256"] = m07_event_fingerprint(event)
        altered = attest_m07_accepted_terminal_event(
            event,
            CoreAcceptedOrderEventProjection(
                freeze(
                    {
                        event["audit_event_id"]: accepted_terminal_lifecycle_proof(
                            event, "CANCEL_PENDING"
                        )
                    }
                )
            ),
        )
        with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
            economic_fact(
                dict(accepted.payload),
                preexisting_accounting_projection((accepted.payload,)),
                altered,
            )
    nonterminal = dict(context.event)
    nonterminal["event_type"] = "ORDER_ACKNOWLEDGED"
    nonterminal["event_fingerprint_sha256"] = m07_event_fingerprint(nonterminal)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_terminal_event(
            nonterminal,
            CoreAcceptedOrderEventProjection(freeze({})),
        )
    tampered = dict(context.event)
    tampered["occurred_at_utc"] = "2026-01-01T00:00:01Z"
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_terminal_event(tampered, context.authority)
    fabricated = replace(context, acceptance_seal=object())
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(
            dict(accepted.payload),
            preexisting_accounting_projection((accepted.payload,)),
            fabricated,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("command_id", "cmd_bad"),
        ("causation_id", "cause_bad"),
        ("exchange_id", ""),
        ("aggregate_version", True),
    ],
)
def test_terminal_event_envelope_values_are_exact(field: str, value: Any) -> None:
    context = cast(
        M07PrevalidatedAcceptedTerminalOrderEventContext,
        terminal_release(770, 771).upstream_context,
    )
    event = dict(context.event)
    event[field] = value
    event["event_fingerprint_sha256"] = m07_event_fingerprint(event)
    proof = accepted_terminal_lifecycle_proof(event, "CANCEL_PENDING")
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_terminal_event(
            event, CoreAcceptedOrderEventProjection(freeze({event["audit_event_id"]: proof}))
        )


@pytest.mark.parametrize(
    "state,event_type,payload",
    [
        (
            "FILLED",
            "ORDER_FILLED",
            {
                "fill_id": "fill_bad",
                "venue_trade_id": "trade",
                "cumulative_executed_quantity": "1",
            },
        ),
        (
            "FILLED",
            "ORDER_FILLED",
            {
                "fill_id": ident("fill", 780),
                "venue_trade_id": "trade",
                "cumulative_executed_quantity": "1.00",
            },
        ),
        (
            "REPLACED",
            "ORDER_REPLACE_CONFIRMED",
            {"replacement_order_id": "ord_bad", "venue_order_id": "venue"},
        ),
    ],
)
def test_terminal_safe_payload_values_are_exact(
    state: str, event_type: str, payload: dict[str, Any]
) -> None:
    context = cast(
        M07PrevalidatedAcceptedTerminalOrderEventContext,
        terminal_release(780, 781, state).upstream_context,
    )
    event = dict(context.event)
    event["event_type"] = event_type
    event["safe_payload"] = payload
    event["event_fingerprint_sha256"] = m07_event_fingerprint(event)
    proof = accepted_terminal_lifecycle_proof(
        event, "PARTIALLY_FILLED" if state == "FILLED" else "REPLACE_PENDING"
    )
    with pytest.raises((ValueError,), match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_terminal_event(
            event, CoreAcceptedOrderEventProjection(freeze({event["audit_event_id"]: proof}))
        )


@pytest.mark.parametrize(
    "proof_change",
    [
        {"predecessor_state": "ACKNOWLEDGED"},
        {"previous_aggregate_version": 0},
        {"aggregate_version": 1},
    ],
)
def test_terminal_lifecycle_proof_rejects_illegal_predecessor_gap_and_stale(
    proof_change: dict[str, Any],
) -> None:
    context = cast(
        M07PrevalidatedAcceptedTerminalOrderEventContext,
        terminal_release(790, 791).upstream_context,
    )
    event = dict(context.event)
    proof = accepted_terminal_lifecycle_proof(event, "CANCEL_PENDING")
    broken = replace(proof, **proof_change)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        attest_m07_accepted_terminal_event(
            event, CoreAcceptedOrderEventProjection(freeze({event["audit_event_id"]: broken}))
        )


def test_reservation_release_history_preserves_upstream_context_and_rebuild_requires_it() -> None:
    engine = funded("10")
    reserve = reservation(PAPER, "10", 730, 731)
    assert engine.account_fact(reserve) == "ACCEPTED"
    release = terminal_release(731, 732)
    assert engine.account_fact(release) == "ACCEPTED"
    for source, identity, context_type in (
        (
            "capital_reservation",
            reserve.payload["audit_event_id"],
            M07PrevalidatedAcceptedCommandContext,
        ),
        (
            "capital_release",
            release.payload["audit_event_id"],
            M07PrevalidatedAcceptedTerminalOrderEventContext,
        ),
    ):
        record = engine.accepted_source_history[(source, identity)]
        assert isinstance(record.context.upstream_context, context_type)
        broken = dict(engine.accepted_source_history)
        broken[(source, identity)] = replace(
            record, context=replace(record.context, upstream_context=None)
        )
        with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
            Engine.rebuild(copy.deepcopy(engine.entries), broken)


def test_raw_nonfill_valuation_and_snapshot_cannot_self_enroll() -> None:
    raw = dict(
        fact("deposit", PAPER, USD, "1", 740, capital_flow_kind="EXTERNAL_CONTRIBUTION").payload
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(raw, CoreAcceptedAccountingFactProjection(freeze({})))
    other = dict(raw)
    other["audit_event_id"] = ident("evt", 741)
    other["source_fingerprint_sha256"] = digest(
        {key: value for key, value in other.items() if key != "source_fingerprint_sha256"}
    )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        economic_fact(other, preexisting_accounting_projection((freeze(raw),)))
    edge = valuation_edge(BTC, USD, "2", number=74)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_valuation_edges(
            [edge], CoreAcceptedValuationSourceRegistry(freeze({})), edge["as_of_utc"]
        )
    snapshot = snapshot_raw(PAPER, USD, "1", 74)
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        validate_observed_context(snapshot, CoreAcceptedObservedBalanceSourceRegistry(freeze({})))


@pytest.mark.parametrize("root", ["command_registry", "event_contract", "order_lifecycle"])
def test_direct_m07_authority_dependency_mutation_fails(root: str) -> None:
    upstream = copy.deepcopy(CANONICAL)
    document = upstream["commands_events_order_lifecycle_and_idempotency.json"]
    if root == "command_registry":
        document[root]["SUBMIT_ORDER"]["accepted_effect"] = "execute"
    elif root == "event_contract":
        document[root]["event_types"].append("ARBITRARY")
    else:
        document[root]["terminal_states"].remove("CANCELLED")
    assert validate_contract(CONTRACT, upstream) == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "mutation",
    ["fingerprint_exclusion", "identity", "nfc", "numeric_encoding"],
)
def test_direct_m07_idempotency_and_closed_request_dependency_mutations_fail(
    mutation: str,
) -> None:
    upstream = copy.deepcopy(CANONICAL)
    document = upstream["commands_events_order_lifecycle_and_idempotency.json"]
    if mutation == "fingerprint_exclusion":
        document["idempotency_contract"]["fingerprint_excluded_fields"] = [
            "correlation_id",
            "causation_id",
        ]
    elif mutation == "identity":
        document["idempotency_contract"]["identity"] = "command_id only"
    elif mutation == "nfc":
        document["closed_request_policy"]["canonical_json"]["unicode_normalization"] = "NFD"
    else:
        document["closed_request_policy"]["numeric_encoding"] = "JSON number"
    assert validate_contract(CONTRACT, upstream) == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize("source", ["capital_reservation", "capital_release"])
def test_composite_source_authority_cannot_be_weakened_to_audit_event_only(source: str) -> None:
    changed = copy.deepcopy(CONTRACT)
    changed["source_registry"][source]["authority"] = "exact-economic AuditEvent"
    assert validate_contract(changed) == "CONTRACT_INCONSISTENT"


@pytest.mark.parametrize(
    "field,value",
    [
        ("workspace_id", ident("ws", 999)),
        ("portfolio_id", ident("port", 999)),
        ("environment", "TESTNET"),
    ],
)
def test_correction_scope_must_exact_match_target(field: str, value: str) -> None:
    engine = funded("10")
    correction = reversal_fact(engine, "deposit", ident("evt", 10), 750)
    raw = dict(correction.payload)
    raw[field] = value
    raw["source_fingerprint_sha256"] = digest(
        {key: item for key, item in raw.items() if key != "source_fingerprint_sha256"}
    )
    changed = economic_fact(raw, preexisting_accounting_projection((freeze(raw),)))
    before = copy.deepcopy(engine.entries)
    assert engine.account_fact(changed) == "TRUSTED_CONTEXT_FAILURE"
    assert engine.entries == before


@pytest.mark.parametrize("mutation", ["role", "posting_role", "quantity", "account"])
def test_target_aware_correction_batch_rejects_semantic_mutation(mutation: str) -> None:
    engine = funded("10")
    correction = reversal_fact(engine, "deposit", ident("evt", 10), 760)
    assert engine.account_fact(correction) == "ACCEPTED"
    target = engine.accepted[("deposit", ident("evt", 10), RULE)][2]
    inverse = list(engine.accepted[("reconciliation_correction", ident("evt", 760), RULE)][2])
    changes: dict[str, Any] = {
        "role": {"account_role": "FEE_EXPENSE"},
        "posting_role": {"posting_role": "FEE_CLASSIFIED"},
        "quantity": {"quantity": "9"},
        "account": {"exchange_account_id": ident("xacc", 999)},
    }[mutation]
    inverse[0] = replace(inverse[0], **changes)
    assert (
        validate_correction_batch(correction, target, tuple(inverse)) == "MALFORMED_ACCOUNTING_FACT"
    )
    assert (
        validate_correction_batch(
            correction,
            target,
            engine.accepted[("reconciliation_correction", ident("evt", 760), RULE)][2],
        )
        == "VALID"
    )


def test_invalid_correction_candidate_fails_before_any_engine_mutation() -> None:
    engine = funded("10")
    correction = reversal_fact(engine, "deposit", ident("evt", 10), 800)
    engine.correction_candidate_hook = lambda candidates: (
        replace(candidates[0], quantity="9"),
        *candidates[1:],
    )
    entries_before = copy.deepcopy(engine.entries)
    accepted_before = copy.deepcopy(engine.accepted)
    history_before = dict(engine.accepted_source_history)
    reservations_before = copy.deepcopy(engine.reservations)
    assert engine.account_fact(correction) == "MALFORMED_ACCOUNTING_FACT"
    assert engine.entries == entries_before
    assert engine.accepted == accepted_before
    assert engine.accepted_source_history == history_before
    assert engine.reservations == reservations_before


@pytest.mark.parametrize(
    "root,path,value",
    [
        ("reservation_protocol", ("binding",), "arbitrary string"),
        ("reservation_protocol", ("fill_consumption",), "available only"),
        ("rebuild_protocol", ("mutable_cache",), "authoritative"),
        ("rebuild_protocol", ("inventory_effects", "internal_transfer"), "quantity only"),
        ("reconciliation_protocol", ("correction",), "generic adjustment"),
        ("source_posting_matrix", ("fee",), {"allowed": ["FEE_EXPENSE"]}),
        ("batch_protocol", ("fill_replay_integrity",), "fingerprint only"),
        ("non_fill_authority", ("self_hash",), "authority"),
        ("valuation_protocol", ("reporting_as_of",), "per-edge now"),
    ],
)
def test_new_closure_protocol_mutations_are_contract_inconsistent(
    root: str, path: tuple[str, ...], value: Any
) -> None:
    changed = copy.deepcopy(CONTRACT)
    target = changed[root]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    assert validate_contract(changed) == "CONTRACT_INCONSISTENT"
