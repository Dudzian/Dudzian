"""Executable pure M0.7 reference model; never production execution runtime."""

import copy
import hashlib
import json
import re
import unicodedata
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import pytest

ROOT = Path(__file__).parents[2]
PATH = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT: dict[str, Any] = json.loads(
    (PATH / "commands_events_order_lifecycle_and_idempotency.json").read_text(encoding="utf-8")
)
CANONICAL = {
    name: json.loads((PATH / name).read_text(encoding="utf-8"))
    for name in (
        "canonical_domain_vocabulary.json",
        "environment_and_product_capabilities.json",
        "exchange_accounts_and_instruments.json",
        "strategy_market_data_and_execution_routing.json",
    )
}
UUID = "018f0f3e-7b5a-7abc-8def-1234567890ab"
ID_RE = re.compile(
    r"^(?P<prefix>[a-z]+)_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)(\.[0-9]*[1-9])?$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d*[1-9])?Z$")


def deep_freeze(value: Any) -> Any:
    if type(value) is dict:
        return MappingProxyType({key: deep_freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(deep_freeze(item) for item in value)
    if type(value) is set:
        return frozenset(deep_freeze(item) for item in value)
    return value


def field(kind: str, **values: Any) -> dict[str, Any]:
    return {"type": kind, **values}


BASE_FIELDS = {
    "command_id": field("id", id_kind="Command", prefix="cmd"),
    "operation_type": field("constant"),
    "authority_context_id": field("id", id_kind="AuthorityContext", prefix="authctx"),
    "environment": field("enum", values=["PAPER", "TESTNET", "LIVE"]),
    "workspace_id": field("id", id_kind="Workspace", prefix="ws"),
    "portfolio_id": field("id", id_kind="Portfolio", prefix="port"),
    "exchange_account_id": field("id", id_kind="ExchangeAccount", prefix="xacc"),
    "strategy_instance_id": field("id", id_kind="StrategyInstance", prefix="sinst"),
    "source_type": field("enum", values=["STRATEGY_INSTANCE", "OPERATOR", "SYSTEM_RECONCILIATION"]),
    "instrument_id": field("id", id_kind="Instrument", prefix="instr"),
    "execution_route_id": field("id", id_kind="ExecutionRoute", prefix="xroute"),
    "correlation_id": field("id", id_kind="Correlation", prefix="corr"),
    "causation_id": field("id", id_kind="Causation", prefix="cause"),
    "idempotency_key": field("id", id_kind="Command", prefix="cmd"),
}
COMMAND_EXTRAS = {
    "SUBMIT_ORDER": {
        "order_intent_id": field("id", id_kind="OrderIntent", prefix="oint"),
        "order_id": field("id", id_kind="Order", prefix="ord"),
        "side": field("enum", values=["BUY", "SELL"]),
        "order_type": field("enum", values=["MARKET", "LIMIT"]),
        "quantity": field("decimal", constraint="positive"),
        "limit_price": field("decimal", constraint="positive"),
        "time_in_force": field("enum", values=["GTC", "IOC", "FOK", "GTD"]),
        "expire_at_utc": field("timestamp"),
    },
    "CANCEL_ORDER": {
        "order_id": field("id", id_kind="Order", prefix="ord"),
        "expected_order_version": field("positive_integer"),
        "reason_code": field(
            "enum",
            values=["OPERATOR_REQUEST", "STRATEGY_REQUEST", "SYSTEM_SAFETY", "RECONCILIATION"],
        ),
    },
    "REPLACE_ORDER": {
        "order_id": field("id", id_kind="Order", prefix="ord"),
        "replacement_order_id": field("id", id_kind="Order", prefix="ord"),
        "expected_order_version": field("positive_integer"),
        "quantity": field("decimal", constraint="positive"),
        "limit_price": field("decimal", constraint="positive"),
        "time_in_force": field("enum", values=["GTC", "IOC", "FOK", "GTD"]),
        "expire_at_utc": field("timestamp"),
    },
}
NULLABLE = {
    "SUBMIT_ORDER": ["strategy_instance_id", "causation_id", "limit_price", "expire_at_utc"],
    "CANCEL_ORDER": ["strategy_instance_id", "causation_id"],
    "REPLACE_ORDER": ["strategy_instance_id", "causation_id", "limit_price", "expire_at_utc"],
}
CONSTRAINTS = {
    "SUBMIT_ORDER": [
        "order_id differs from command_id",
        "MARKET requires limit_price null",
        "LIMIT requires limit_price non-null",
        "GTD requires expire_at_utc non-null",
        "non-GTD requires expire_at_utc null",
        "STRATEGY_INSTANCE source requires strategy_instance_id non-null",
        "other sources require strategy_instance_id null",
    ],
    "CANCEL_ORDER": ["expected_order_version > 0 and bool forbidden"],
    "REPLACE_ORDER": [
        "replacement_order_id differs from order_id",
        "expected_order_version > 0 and bool forbidden",
        "LIMIT semantics require limit_price non-null",
        "GTD requires expire_at_utc non-null",
        "non-GTD requires expire_at_utc null",
    ],
}
ACCEPTED_EFFECTS = {
    "SUBMIT_ORDER": {
        "creates": "one Order from one OrderIntent",
        "accepted_effect": "persist plan only; not acknowledgement or execution",
    },
    "CANCEL_ORDER": {
        "targets": "existing nonterminal Order",
        "accepted_effect": "plan cancellation only",
    },
    "REPLACE_ORDER": {
        "targets": "existing nonterminal Order and preallocates distinct replacement Order identity",
        "accepted_effect": "plan venue replace; confirmation terminates original as REPLACED",
    },
}
_expected_commands: dict[str, Any] = {}
for _operation in ("SUBMIT_ORDER", "CANCEL_ORDER", "REPLACE_ORDER"):
    _fields = {**BASE_FIELDS, **COMMAND_EXTRAS[_operation]}
    _fields["operation_type"] = field("constant", value=_operation)
    _expected_commands[_operation] = {
        "authority": "CoreHost command handler",
        "request_fields": list(_fields),
        "nullable_fields": NULLABLE[_operation],
        "field_schemas": _fields,
        "decimal_fields": [name for name, spec in _fields.items() if spec["type"] == "decimal"],
        "timestamp_fields": [name for name, spec in _fields.items() if spec["type"] == "timestamp"],
        "expected_version_fields": [
            name for name, spec in _fields.items() if spec["type"] == "positive_integer"
        ],
        "constraints": CONSTRAINTS[_operation],
        **ACCEPTED_EFFECTS[_operation],
    }
EXPECTED_COMMAND_REGISTRY = deep_freeze(_expected_commands)
EXPECTED_STATES = frozenset(
    {
        "PLANNED",
        "SUBMISSION_PENDING",
        "ACKNOWLEDGED",
        "PARTIALLY_FILLED",
        "CANCEL_PENDING",
        "REPLACE_PENDING",
        "RECONCILIATION_REQUIRED",
        "REJECTED",
        "FILLED",
        "CANCELLED",
        "EXPIRED",
        "REPLACED",
    }
)
EXPECTED_FAILURES = frozenset(
    {
        "MALFORMED_REQUEST",
        "UNAUTHORIZED_OPERATION",
        "INVALID_LIFECYCLE_TRANSITION",
        "IDEMPOTENCY_CONFLICT",
        "REPLAY_SUCCESS",
        "TRUSTED_CONTEXT_FAILURE",
        "CONTRACT_INCONSISTENT",
        "EXTERNAL_OUTCOME_UNKNOWN",
        "STALE_EVENT",
        "ORDER_SCOPE_MISMATCH",
        "EVENT_IDENTITY_CONFLICT",
        "EVENT_VERSION_GAP",
        "MALFORMED_EVENT",
        "EVENT_FINGERPRINT_MISMATCH",
        "FILL_IDENTITY_CONFLICT",
        "MALFORMED_FILL",
        "FILL_PROGRESSION_CONFLICT",
    }
)
EXPECTED_INSTRUMENT_RECORD_FIELDS = (
    "instrument_id",
    "workspace_id",
    "exchange_id",
    "environment",
    "market_type",
    "instrument_type",
    "venue_symbol",
    "display_symbol",
    "base_asset_reference",
    "quote_asset_reference",
    "settlement_asset_reference",
    "trading_status",
    "price_tick",
    "quantity_step",
    "min_quantity",
    "max_quantity",
    "min_notional",
    "max_notional",
    "contract_size",
    "contract_value_currency",
    "derivative_settlement_type",
    "expiry_at_utc",
    "strike_price",
    "option_side",
    "catalog_snapshot_id",
    "metadata_version",
    "observed_at_utc",
    "effective_at_utc",
    "stale_after_utc",
    "source_adapter_family_id",
)
EXPECTED_FILL_CONTRACT = deep_freeze(
    {
        "identity": "canonical fill_id prefix fill from M0.2; parent is canonical order_id",
        "immutable_full_economic_fact": True,
        "fact_fields": [
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
        "nullable_fields": ["fee_asset_reference"],
        "field_schemas": {
            "fill_id": {"type": "id", "prefix": "fill"},
            "order_id": {"type": "id", "prefix": "ord"},
            "environment": {"type": "enum", "values": ["PAPER", "TESTNET", "LIVE"]},
            "workspace_id": {"type": "id", "prefix": "ws"},
            "portfolio_id": {"type": "id", "prefix": "port"},
            "exchange_account_id": {"type": "id", "prefix": "xacc"},
            "exchange_id": {"type": "non_empty_string"},
            "instrument_id": {"type": "id", "prefix": "instr"},
            "instrument_metadata_version": {"type": "positive_integer"},
            "execution_route_id": {"type": "id", "prefix": "xroute"},
            "venue_trade_id": {"type": "non_empty_string"},
            "side": {"type": "enum", "values": ["BUY", "SELL"]},
            "executed_quantity": {"type": "decimal", "constraint": "positive"},
            "execution_price": {"type": "decimal", "constraint": "positive"},
            "executed_at_utc": {"type": "timestamp"},
            "fee_kind": {"type": "enum", "values": ["NONE", "CHARGE"]},
            "fee_quantity": {"type": "decimal", "constraint": "non_negative"},
            "fee_asset_reference": {
                "type": "asset_reference",
                "fields": [
                    "venue_asset_code",
                    "canonical_display_code",
                    "asset_namespace",
                    "mapping_status",
                ],
                "field_schemas": {
                    "venue_asset_code": {"type": "non_empty_string"},
                    "canonical_display_code": {"type": "non_empty_string"},
                    "asset_namespace": {"type": "non_empty_string"},
                    "mapping_status": {"type": "enum", "values": ["EXACT", "EXPLICIT_ALIAS"]},
                },
                "rules": [
                    "exact M0.5 asset-reference value object",
                    "AMBIGUOUS and UNKNOWN forbidden",
                    "no default to base, quote, or settlement asset",
                ],
            },
            "fill_fingerprint_sha256": {"type": "sha256_hex"},
        },
        "instrument_binding": {
            "identity": "instrument_id",
            "metadata_binding": "instrument_metadata_version resolves exactly one full "
            "trusted historical Instrument record",
            "required_matches": ["workspace_id", "exchange_id", "environment"],
            "asset_semantics": "resolved bound record supplies base, quote, and "
            "settlement asset references; no symbol parsing or "
            "current-version fallback",
            "trusted_context": "nominal M05PrevalidatedInstrumentHistory produced only "
            "after successful canonical M0.5 trusted-history "
            "validation; raw/unvalidated mappings and caller "
            "booleans are rejected",
            "resolution_failures": "missing/duplicate requested version, malformed "
            "history, or scope mismatch => "
            "TRUSTED_CONTEXT_FAILURE; no current-version "
            "fallback",
            "m0_7_responsibility": "map key, requested instrument/version, ordered "
            "unique versions, identity continuity, Fill scope "
            "match, settlement namespace defense; M0.7 does not "
            "reimplement or weaken M0.5 structural validation",
        },
        "fee_semantics": {
            "fee_kind_values": ["NONE", "CHARGE"],
            "NONE": "fee_quantity must be exactly 0 and fee_asset_reference must be null",
            "CHARGE": "fee_quantity must be positive and fee_asset_reference must be "
            "non-null, exact, mapping_status EXACT or EXPLICIT_ALIAS, and "
            "asset_namespace must equal Fill.exchange_id",
            "third_asset_fee": "legal and preserved without base/quote default",
            "maker_rebate": "unsupported fail-closed; negative decimals forbidden; a future "
            "explicit economic fact kind is required",
        },
        "fingerprint": {
            "field": "fill_fingerprint_sha256",
            "algorithm": "SHA-256",
            "input_fields": [
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
            ],
            "excluded_fields": ["fill_fingerprint_sha256"],
            "canonicalization": [
                "exact fact field order is schema attestation; JSON object keys sorted for hashing",
                "UTF-8",
                "NFC",
                "canonical JSON separators comma and colon",
                "M0.5 canonical decimals",
                "canonical UTC RFC3339 Z timestamps",
            ],
        },
        "external_dedupe_scope": [
            "environment",
            "exchange_account_id",
            "exchange_id",
            "venue_trade_id",
        ],
        "dedupe": {
            "same_fill_id_same_fingerprint": "REPLAY_SUCCESS; zero new economic fact",
            "same_fill_id_different_fingerprint": "FILL_IDENTITY_CONFLICT; reconciliation; no "
            "mutation",
            "same_external_key_same_economics": "REPLAY_SUCCESS; if fill_id differs retain first "
            "fact and create zero duplicate economic effect",
            "same_external_key_different_economics": "FILL_IDENTITY_CONFLICT; reconciliation; no "
            "mutation",
        },
        "trusted_resolution": {
            "context_field": "fills_by_id",
            "shape": "map fill_id -> exact immutable full Fill fact",
            "authority": "composite Core-owned trusted boundary: exact Fill validation "
            "plus nominal prevalidated M0.5 instrument history binding; "
            "never caller or event safe_payload authority",
            "rules": [
                "map key equals fact.fill_id",
                "exact schema and fingerprint required",
                "one deterministic fact per fill_id",
                "missing fill is TRUSTED_CONTEXT_FAILURE, never zero execution or no-fee",
                "conflicting duplicate is FILL_IDENTITY_CONFLICT",
                "persistence implementation deferred to M0.11",
                "composite resolver returns Fill only with its exact resolved "
                "historical Instrument record",
                "raw instrument_history_by_id mapping is rejected even when shaped correctly",
            ],
        },
        "event_referential_integrity": {
            "events": ["ORDER_PARTIALLY_FILLED", "ORDER_FILLED"],
            "rules": [
                "safe_payload fill_id resolves through trusted fills_by_id",
                "Fill order_id equals event order_id",
                "Fill environment workspace portfolio exchange_account "
                "exchange instrument and route equal event envelope",
                "Fill venue_trade_id equals safe_payload venue_trade_id",
                "Fill fingerprint validates before lifecycle effect",
                "cumulative_executed_quantity is recomputed from "
                "accepted unique Fill history and must equal payload; "
                "caller value is not authority",
                "all historical Fill scope fields match event envelope",
                "cumulative uses canonical durable and scoped external "
                "venue-trade dedupe semantics",
                "complete Core-owned accepted_fill_ids_by_order_id "
                "sequence is mandatory; no caller list and no "
                "current-fill fallback",
                "current canonical Fill occurs in the exact Order "
                "sequence; an exact duplicate external trade outside it "
                "is REPLAY_SUCCESS",
            ],
            "missing_or_mismatch": "TRUSTED_CONTEXT_FAILURE; reconciliation; no lifecycle mutation",
            "accepted_history_context": {
                "field": "accepted_fill_ids_by_order_id",
                "shape": "nominal "
                "CoreAcceptedFillProjection: "
                "order_id -> one ordered "
                "canonical accepted fill_id "
                "sequence",
                "authority": "Core-owned projection "
                "derived from accepted "
                "M0.7 event/order "
                "history; raw "
                "mapping/list, caller "
                "request and "
                "safe_payload are "
                "rejected",
                "duplicate_external_trade": "only "
                "first "
                "canonical "
                "economic "
                "effect "
                "remains "
                "in "
                "sequence; "
                "later "
                "exact "
                "duplicate "
                "resolves "
                "to "
                "REPLAY_SUCCESS "
                "and "
                "creates "
                "no "
                "lifecycle "
                "or "
                "cumulative "
                "effect",
            },
        },
        "decimal_fields": ["executed_quantity", "execution_price", "fee_quantity"],
        "float_forbidden": True,
        "cumulative_executed_quantity": "sum of unique accepted full-economic Fill facts in deterministic "
        "accepted event sequence",
        "remaining_quantity": "order quantity minus cumulative executed quantity",
        "average_execution_price": "retain exact notional numerator and cumulative executed denominator; "
        "emit canonical decimal only when quotient has a finite decimal "
        "expansion; otherwise null for presentation and never apply implicit "
        "Decimal-context rounding",
        "invariants": [
            "fill quantity > 0",
            "execution price > 0",
            "executed_at_utc is immutable economic time and distinct from event occurred_at_utc",
            "cumulative never regresses",
            "cumulative <= order quantity",
            "remaining >= 0",
            "PARTIALLY_FILLED iff 0 < cumulative < quantity",
            "FILLED iff cumulative == quantity",
            "overfill rejected and reconciliation required",
        ],
        "canonical_decimal_output": "M0.5 regex; fractional trailing zeros removed and zero exactly 0",
        "trust_boundaries": {
            "raw_structural_fill": "validate_fill proves exact fields, types, canonical "
            "encoding, fee scope and full-economic fingerprint "
            "only; not trusted economic authority",
            "trusted_economic_fill": "composite resolver requires structural validity "
            "plus resolution against a nominal prevalidated "
            "M0.5 instrument history context",
            "complete_accepted_history": "Core-owned accepted_fill_ids_by_order_id "
            "projection derived only from accepted M0.7 "
            "lifecycle history; caller request, "
            "safe_payload, and raw list/dict are never "
            "authority",
        },
    }
)
EXPECTED_FILL_FACT_SCHEMA = deep_freeze(
    {
        "fields": [
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
        "nullable_fields": ["fee_asset_reference"],
        "field_schemas": {
            "fill_id": {"type": "id", "prefix": "fill"},
            "order_id": {"type": "id", "prefix": "ord"},
            "environment": {"type": "enum", "values": ["PAPER", "TESTNET", "LIVE"]},
            "workspace_id": {"type": "id", "prefix": "ws"},
            "portfolio_id": {"type": "id", "prefix": "port"},
            "exchange_account_id": {"type": "id", "prefix": "xacc"},
            "exchange_id": {"type": "non_empty_string"},
            "instrument_id": {"type": "id", "prefix": "instr"},
            "instrument_metadata_version": {"type": "positive_integer"},
            "execution_route_id": {"type": "id", "prefix": "xroute"},
            "venue_trade_id": {"type": "non_empty_string"},
            "side": {"type": "enum", "values": ["BUY", "SELL"]},
            "executed_quantity": {"type": "decimal", "constraint": "positive"},
            "execution_price": {"type": "decimal", "constraint": "positive"},
            "executed_at_utc": {"type": "timestamp"},
            "fee_kind": {"type": "enum", "values": ["NONE", "CHARGE"]},
            "fee_quantity": {"type": "decimal", "constraint": "non_negative"},
            "fee_asset_reference": {
                "type": "asset_reference",
                "fields": [
                    "venue_asset_code",
                    "canonical_display_code",
                    "asset_namespace",
                    "mapping_status",
                ],
                "field_schemas": {
                    "venue_asset_code": {"type": "non_empty_string"},
                    "canonical_display_code": {"type": "non_empty_string"},
                    "asset_namespace": {"type": "non_empty_string"},
                    "mapping_status": {"type": "enum", "values": ["EXACT", "EXPLICIT_ALIAS"]},
                },
                "rules": [
                    "exact M0.5 asset-reference value object",
                    "AMBIGUOUS and UNKNOWN forbidden",
                    "no default to base, quote, or settlement asset",
                ],
            },
            "fill_fingerprint_sha256": {"type": "sha256_hex"},
        },
    }
)

EXPECTED_EVENT_SCHEMAS = deep_freeze(
    {
        "ORDER_PLANNED": {
            "safe_payload_fields": ["side", "order_type", "quantity"],
            "field_schemas": {
                "side": {"type": "string"},
                "order_type": {"type": "string"},
                "quantity": {"type": "decimal"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_DISPATCHED": {
            "safe_payload_fields": ["client_order_id"],
            "field_schemas": {"client_order_id": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_ACKNOWLEDGED": {
            "safe_payload_fields": ["venue_order_id"],
            "field_schemas": {"venue_order_id": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_REJECTED": {
            "safe_payload_fields": ["reason_code"],
            "field_schemas": {"reason_code": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_PARTIALLY_FILLED": {
            "safe_payload_fields": ["fill_id", "venue_trade_id", "cumulative_executed_quantity"],
            "field_schemas": {
                "fill_id": {"type": "id", "prefix": "fill"},
                "venue_trade_id": {"type": "string"},
                "cumulative_executed_quantity": {"type": "decimal"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_FILLED": {
            "safe_payload_fields": ["fill_id", "venue_trade_id", "cumulative_executed_quantity"],
            "field_schemas": {
                "fill_id": {"type": "id", "prefix": "fill"},
                "venue_trade_id": {"type": "string"},
                "cumulative_executed_quantity": {"type": "decimal"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_CANCEL_REQUESTED": {
            "safe_payload_fields": ["reason_code"],
            "field_schemas": {"reason_code": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_CANCEL_CONFIRMED": {
            "safe_payload_fields": ["venue_order_id"],
            "field_schemas": {"venue_order_id": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_CANCEL_REJECTED": {
            "safe_payload_fields": ["reason_code"],
            "field_schemas": {"reason_code": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_REPLACE_REQUESTED": {
            "safe_payload_fields": ["replacement_order_id"],
            "field_schemas": {"replacement_order_id": {"type": "id", "prefix": "ord"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_REPLACE_CONFIRMED": {
            "safe_payload_fields": ["replacement_order_id", "venue_order_id"],
            "field_schemas": {
                "replacement_order_id": {"type": "id", "prefix": "ord"},
                "venue_order_id": {"type": "string"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_REPLACE_REJECTED": {
            "safe_payload_fields": ["reason_code"],
            "field_schemas": {"reason_code": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_EXPIRED": {
            "safe_payload_fields": ["venue_order_id"],
            "field_schemas": {"venue_order_id": {"type": "string"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_EXTERNAL_OUTCOME_UNKNOWN": {
            "safe_payload_fields": ["operation_type", "client_order_id"],
            "field_schemas": {
                "operation_type": {"type": "string"},
                "client_order_id": {"type": "string"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "ORDER_RECONCILIATION_OBSERVED": {
            "safe_payload_fields": ["trusted_fact_kind", "venue_order_id"],
            "field_schemas": {
                "trusted_fact_kind": {
                    "type": "enum",
                    "values": [
                        "ACKNOWLEDGED",
                        "REJECTED",
                        "PARTIAL_FILL",
                        "FULL_FILL",
                        "CANCEL_CONFIRMED",
                        "REPLACE_CONFIRMED",
                        "EXPIRED",
                    ],
                },
                "venue_order_id": {"type": "string"},
            },
            "nullable_fields": ["venue_order_id"],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "COMMAND_ACCEPTED": {
            "safe_payload_fields": ["command_id", "operation_type"],
            "field_schemas": {
                "command_id": {"type": "id", "prefix": "cmd"},
                "operation_type": {"type": "string"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "COMMAND_REJECTED": {
            "safe_payload_fields": ["command_id", "denial_code"],
            "field_schemas": {
                "command_id": {"type": "id", "prefix": "cmd"},
                "denial_code": {"type": "string"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "COMMAND_REPLAYED": {
            "safe_payload_fields": ["command_id"],
            "field_schemas": {"command_id": {"type": "id", "prefix": "cmd"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "IDEMPOTENCY_CONFLICT": {
            "safe_payload_fields": ["command_id"],
            "field_schemas": {"command_id": {"type": "id", "prefix": "cmd"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "EVENT_REPLAY_IGNORED": {
            "safe_payload_fields": ["replayed_audit_event_id"],
            "field_schemas": {"replayed_audit_event_id": {"type": "id", "prefix": "evt"}},
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
        "EVENT_REJECTED": {
            "safe_payload_fields": ["rejected_audit_event_id", "reason_code"],
            "field_schemas": {
                "rejected_audit_event_id": {"type": "id", "prefix": "evt"},
                "reason_code": {"type": "string"},
            },
            "nullable_fields": [],
            "required_scope": [
                "environment",
                "workspace_id",
                "portfolio_id",
                "exchange_account_id",
                "exchange_id",
                "instrument_id",
                "execution_route_id",
            ],
        },
    }
)
EXPECTED_EVENT_TYPES = tuple(EXPECTED_EVENT_SCHEMAS)
EXPECTED_ORDER_LIFECYCLE = deep_freeze(
    {
        "states": [
            "PLANNED",
            "SUBMISSION_PENDING",
            "ACKNOWLEDGED",
            "PARTIALLY_FILLED",
            "CANCEL_PENDING",
            "REPLACE_PENDING",
            "RECONCILIATION_REQUIRED",
            "REJECTED",
            "FILLED",
            "CANCELLED",
            "EXPIRED",
            "REPLACED",
        ],
        "initial_state": "PLANNED",
        "terminal_states": ["REJECTED", "FILLED", "CANCELLED", "EXPIRED", "REPLACED"],
        "transitions": [
            {
                "event": "ORDER_PLANNED",
                "owner": "COMMAND_HANDLER",
                "sources": ["NONE"],
                "target": "PLANNED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_DISPATCHED",
                "owner": "DISPATCHER",
                "sources": ["PLANNED"],
                "target": "SUBMISSION_PENDING",
                "replay_safe": True,
            },
            {
                "event": "ORDER_ACKNOWLEDGED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["SUBMISSION_PENDING", "RECONCILIATION_REQUIRED"],
                "target": "ACKNOWLEDGED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_REJECTED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["SUBMISSION_PENDING", "RECONCILIATION_REQUIRED"],
                "target": "REJECTED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_PARTIALLY_FILLED",
                "owner": "FILL_INGESTOR",
                "sources": [
                    "SUBMISSION_PENDING",
                    "ACKNOWLEDGED",
                    "PARTIALLY_FILLED",
                    "CANCEL_PENDING",
                    "REPLACE_PENDING",
                    "RECONCILIATION_REQUIRED",
                ],
                "target": "PARTIALLY_FILLED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_FILLED",
                "owner": "FILL_INGESTOR",
                "sources": [
                    "SUBMISSION_PENDING",
                    "ACKNOWLEDGED",
                    "PARTIALLY_FILLED",
                    "CANCEL_PENDING",
                    "REPLACE_PENDING",
                    "RECONCILIATION_REQUIRED",
                ],
                "target": "FILLED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_CANCEL_REQUESTED",
                "owner": "COMMAND_HANDLER",
                "sources": ["ACKNOWLEDGED", "PARTIALLY_FILLED"],
                "target": "CANCEL_PENDING",
                "replay_safe": True,
            },
            {
                "event": "ORDER_CANCEL_CONFIRMED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["CANCEL_PENDING", "RECONCILIATION_REQUIRED"],
                "target": "CANCELLED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_CANCEL_REJECTED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["CANCEL_PENDING"],
                "replay_safe": True,
                "target_rule": "RESTORE_PRE_REQUEST_STATE",
            },
            {
                "event": "ORDER_REPLACE_REQUESTED",
                "owner": "COMMAND_HANDLER",
                "sources": ["ACKNOWLEDGED", "PARTIALLY_FILLED"],
                "target": "REPLACE_PENDING",
                "replay_safe": True,
            },
            {
                "event": "ORDER_REPLACE_CONFIRMED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["REPLACE_PENDING", "RECONCILIATION_REQUIRED"],
                "target": "REPLACED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_REPLACE_REJECTED",
                "owner": "EXECUTION_ADAPTER",
                "sources": ["REPLACE_PENDING"],
                "replay_safe": True,
                "target_rule": "RESTORE_PRE_REQUEST_STATE",
            },
            {
                "event": "ORDER_EXPIRED",
                "owner": "EXECUTION_ADAPTER",
                "sources": [
                    "SUBMISSION_PENDING",
                    "ACKNOWLEDGED",
                    "PARTIALLY_FILLED",
                    "CANCEL_PENDING",
                    "REPLACE_PENDING",
                    "RECONCILIATION_REQUIRED",
                ],
                "target": "EXPIRED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_EXTERNAL_OUTCOME_UNKNOWN",
                "owner": "DISPATCHER",
                "sources": ["SUBMISSION_PENDING", "CANCEL_PENDING", "REPLACE_PENDING"],
                "target": "RECONCILIATION_REQUIRED",
                "replay_safe": True,
            },
            {
                "event": "ORDER_RECONCILIATION_OBSERVED",
                "owner": "RECONCILER",
                "sources": ["RECONCILIATION_REQUIRED"],
                "replay_safe": True,
                "target_rule": "RESOLVE_TRUSTED_RECONCILIATION_FACT",
            },
        ],
        "restore_rule": "RESTORE_PRE_REQUEST_STATE is exact stored ACKNOWLEDGED or PARTIALLY_FILLED state/version "
        "from request transition; never caller supplied",
        "derived_reconciliation_rule": "target is deterministically derived from trusted "
        "acknowledgement/fill/cancel/replace/expiry fact and full aggregate history",
        "terminal_rule": "terminal states have no outgoing transitions; late identical facts are replay, other "
        "facts require reconciliation/audit and never regress terminal state",
        "dynamic_target_rules": {
            "RESTORE_PRE_REQUEST_STATE": {
                "allowed_stored_states": ["ACKNOWLEDGED", "PARTIALLY_FILLED"],
                "missing_or_other": "INVALID_LIFECYCLE_TRANSITION",
                "caller_target_forbidden": True,
            },
            "RESOLVE_TRUSTED_RECONCILIATION_FACT": {
                "trusted_fact_targets": {
                    "ACKNOWLEDGED": "ACKNOWLEDGED",
                    "REJECTED": "REJECTED",
                    "PARTIAL_FILL": "PARTIALLY_FILLED",
                    "FULL_FILL": "FILLED",
                    "CANCEL_CONFIRMED": "CANCELLED",
                    "REPLACE_CONFIRMED": "REPLACED",
                    "EXPIRED": "EXPIRED",
                },
                "unknown_or_untrusted": "TRUSTED_CONTEXT_FAILURE",
            },
        },
    }
)
EXPECTED_CROSS_CONTRACT_DEPENDENCIES = deep_freeze(
    [
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
            "json_pointer": "/identifier_policy",
            "content_fingerprint_sha256": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20",
        },
        {
            "contract": "canonical_domain_vocabulary.json",
            "json_pointer": "/external_identifier_fields",
            "content_fingerprint_sha256": "4259419170eeb4677098999b288048213329f9b9a16be58ab67a24838d95b4be",
        },
        {
            "contract": "environment_and_product_capabilities.json",
            "json_pointer": "/execution_environments",
            "content_fingerprint_sha256": "61f5dd195aa69c646dc296c2d8c0f2ce97683979e3ef6ddc35b0adaa269a809b",
        },
        {
            "contract": "environment_and_product_capabilities.json",
            "json_pointer": "/current_product_edition",
            "content_fingerprint_sha256": "5fa73f356d9b315161031b4fb41661b8294ed14ce42bfae4565e87b34120adc8",
        },
        {
            "contract": "environment_and_product_capabilities.json",
            "json_pointer": "/capability_id_registry/current_schema_allowed_capability_ids",
            "content_fingerprint_sha256": "da387f22de48e101f293b5dd304eedf52e0cb5dbcf8142b0f0c98b972acd55d2",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/environment_registry",
            "content_fingerprint_sha256": "c2182111523b676163dda381a902ed4ef238a9e1508484e1da1ce790e83cf0d9",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/exchange_account_contract/execution_authorizations",
            "content_fingerprint_sha256": "06cd91f979f0f90a6ae247565950152397305215299cc3450a54acee307d120b",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/decimal_policy",
            "content_fingerprint_sha256": "e526f728e0075a3a27380a87a213eb5e5f074d90cc4b63a99b48485957070b48",
        },
        {
            "contract": "strategy_market_data_and_execution_routing.json",
            "json_pointer": "/entity_registry",
            "content_fingerprint_sha256": "8bde48ea01fbc636040ce6c3764ac4e331e03fbb042defa449b85d20b54a5b43",
        },
        {
            "contract": "strategy_market_data_and_execution_routing.json",
            "json_pointer": "/execution_route_contract",
            "content_fingerprint_sha256": "a855a114e6b19b8218f9975d64fb04b9dac86e28a208e9e7d3a4976773de0a12",
        },
        {
            "contract": "strategy_market_data_and_execution_routing.json",
            "json_pointer": "/route_readiness_contract",
            "content_fingerprint_sha256": "92dda02de1e327d248bc9e987701b33279534f48b57ac2a97104c8380d7c4c33",
        },
        {
            "contract": "strategy_market_data_and_execution_routing.json",
            "json_pointer": "/live_execution_authority_policy",
            "content_fingerprint_sha256": "eb2034a4691a18f3037b2a5f93d1fb41fcb38ef67c6c43c5dfff891dd35fe1de",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/asset_reference_contract",
            "content_fingerprint_sha256": "837e1452a60de230d0ca091a7e2c05308ff41d961499f7800d350d7c9b3682ae",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/instrument_contract/trusted_history_contract",
            "content_fingerprint_sha256": "87b09322419aac6fcc36f12405931e1cf88cf0630ced72e68b341d5434e10e8f",
        },
        {
            "contract": "exchange_accounts_and_instruments.json",
            "json_pointer": "/instrument_contract/record_fields",
            "content_fingerprint_sha256": "4c10117c28505bc7b546126690f3f3ef6cbc15c47815a47e016bd8c646b71409",
        },
    ]
)
EXPECTED_EVENT_ENVELOPE = deep_freeze(
    {
        "fields": [
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
        ],
        "nullable_fields": ["causation_id", "command_id"],
        "field_schemas": {
            "audit_event_id": {"type": "id", "prefix": "evt"},
            "event_type": {"type": "enum", "registry": "event_types"},
            "order_id": {"type": "id", "prefix": "ord"},
            "aggregate_version": {"type": "positive_integer"},
            "correlation_id": {"type": "id", "prefix": "corr"},
            "causation_id": {"type": "id", "prefix": "cause"},
            "command_id": {"type": "id", "prefix": "cmd"},
            "environment": {"type": "enum", "values": ["PAPER", "TESTNET", "LIVE"]},
            "workspace_id": {"type": "id", "prefix": "ws"},
            "portfolio_id": {"type": "id", "prefix": "port"},
            "exchange_account_id": {"type": "id", "prefix": "xacc"},
            "exchange_id": {"type": "non_empty_string"},
            "instrument_id": {"type": "id", "prefix": "instr"},
            "execution_route_id": {"type": "id", "prefix": "xroute"},
            "occurred_at_utc": {"type": "timestamp"},
            "safe_payload": {"type": "event_safe_payload"},
            "event_fingerprint_sha256": {"type": "sha256_hex"},
        },
        "fingerprint_excluded_fields": ["event_fingerprint_sha256"],
        "scope_fields": [
            "environment",
            "workspace_id",
            "portfolio_id",
            "exchange_account_id",
            "exchange_id",
            "instrument_id",
            "execution_route_id",
        ],
    }
)
# This fingerprint prevents the independently frozen expectation above from being regenerated
# as authority after a coordinated mutable schema edit.
EXPECTED_EVENT_SCHEMA_FINGERPRINT = (
    "1a00bea752efca1cc06740eb144b1d6c72162c47bead09a72e5406fe8e1bfa85"
)
EXPECTED_DYNAMIC_RULES = deep_freeze(
    {
        "RESTORE_PRE_REQUEST_STATE": {
            "allowed_stored_states": ["ACKNOWLEDGED", "PARTIALLY_FILLED"],
            "missing_or_other": "INVALID_LIFECYCLE_TRANSITION",
            "caller_target_forbidden": True,
        },
        "RESOLVE_TRUSTED_RECONCILIATION_FACT": {
            "trusted_fact_targets": {
                "ACKNOWLEDGED": "ACKNOWLEDGED",
                "REJECTED": "REJECTED",
                "PARTIAL_FILL": "PARTIALLY_FILLED",
                "FULL_FILL": "FILLED",
                "CANCEL_CONFIRMED": "CANCELLED",
                "REPLACE_CONFIRMED": "REPLACED",
                "EXPIRED": "EXPIRED",
            },
            "unknown_or_untrusted": "TRUSTED_CONTEXT_FAILURE",
        },
    }
)


def canonical_json(value: Any) -> bytes:
    normalized = unicodedata.normalize(
        "NFC", json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    )
    return normalized.encode("utf-8")


def canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonical_command_fingerprint(request: dict[str, Any]) -> str:
    excluded = frozenset({"correlation_id"})
    return canonical_fingerprint(
        {key: value for key, value in request.items() if key not in excluded}
    )


def canonical_event_fingerprint(event: dict[str, Any]) -> str:
    return canonical_fingerprint(
        {key: value for key, value in event.items() if key != "event_fingerprint_sha256"}
    )


def resolve_pointer(document: Any, pointer: str) -> Any:
    value = document
    for encoded in pointer[1:].split("/"):
        key = encoded.replace("~1", "/").replace("~0", "~")
        if type(value) is not dict or key not in value:
            raise ValueError("CONTRACT_INCONSISTENT")
        value = value[key]
    return value


def validate_contract(contract: dict[str, Any]) -> None:
    if contract["status"] != "closed":
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(contract["command_registry"]) != EXPECTED_COMMAND_REGISTRY:
        raise ValueError("CONTRACT_INCONSISTENT")
    lifecycle = contract["order_lifecycle"]
    if deep_freeze(lifecycle) != EXPECTED_ORDER_LIFECYCLE:
        raise ValueError("CONTRACT_INCONSISTENT")
    if frozenset(contract["failure_taxonomy"]) != EXPECTED_FAILURES:
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(contract["fill_contract"]) != EXPECTED_FILL_CONTRACT:
        raise ValueError("CONTRACT_INCONSISTENT")
    if (
        deep_freeze(
            {
                "fields": contract["fill_contract"]["fact_fields"],
                "nullable_fields": contract["fill_contract"]["nullable_fields"],
                "field_schemas": contract["fill_contract"]["field_schemas"],
            }
        )
        != EXPECTED_FILL_FACT_SCHEMA
    ):
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(contract["event_contract"]["event_schema_registry"]) != EXPECTED_EVENT_SCHEMAS:
        raise ValueError("CONTRACT_INCONSISTENT")
    if (
        canonical_fingerprint(contract["event_contract"]["event_schema_registry"])
        != EXPECTED_EVENT_SCHEMA_FINGERPRINT
    ):
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(lifecycle["dynamic_target_rules"]) != EXPECTED_DYNAMIC_RULES:
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(contract["event_contract"]["envelope_schema"]) != EXPECTED_EVENT_ENVELOPE:
        raise ValueError("CONTRACT_INCONSISTENT")
    if tuple(contract["event_contract"]["event_types"]) != EXPECTED_EVENT_TYPES:
        raise ValueError("CONTRACT_INCONSISTENT")
    if tuple(contract["event_contract"]["fields"]) != tuple(EXPECTED_EVENT_ENVELOPE["fields"]):
        raise ValueError("CONTRACT_INCONSISTENT")
    if deep_freeze(contract["cross_contract_dependencies"]) != EXPECTED_CROSS_CONTRACT_DEPENDENCIES:
        raise ValueError("CONTRACT_INCONSISTENT")
    if contract["idempotency_contract"]["fingerprint_excluded_fields"] != ["correlation_id"]:
        raise ValueError("CONTRACT_INCONSISTENT")
    for transition in lifecycle["transitions"]:
        if "target" in transition and transition["target"] not in EXPECTED_STATES:
            raise ValueError("CONTRACT_INCONSISTENT")
    terminal = frozenset(lifecycle["terminal_states"])
    if terminal & {source for item in lifecycle["transitions"] for source in item["sources"]}:
        raise ValueError("CONTRACT_INCONSISTENT")
    for dependency in EXPECTED_CROSS_CONTRACT_DEPENDENCIES:
        source = CANONICAL.get(dependency["contract"])
        if source is None:
            raise ValueError("CONTRACT_INCONSISTENT")
        actual = resolve_pointer(source, dependency["json_pointer"])
        if canonical_fingerprint(actual) != dependency["content_fingerprint_sha256"]:
            raise ValueError("CONTRACT_INCONSISTENT")


def valid_id(value: Any, prefix: str) -> bool:
    match = ID_RE.fullmatch(value) if type(value) is str else None
    return bool(match and match["prefix"] == prefix)


def valid_decimal(value: Any, positive: bool = True) -> bool:
    if type(value) is not str or not DECIMAL_RE.fullmatch(value):
        return False
    try:
        parsed = Decimal(value)
    except InvalidOperation:
        return False
    return parsed.is_finite() and (parsed > 0 if positive else parsed >= 0)


def valid_timestamp(value: Any) -> bool:
    if type(value) is not str or not TIMESTAMP_RE.fullmatch(value):
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo == UTC
    except ValueError:
        return False


def canonical_decimal_string(value: Fraction) -> str | None:
    if value < 0:
        raise ValueError("negative decimal result")
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    if denominator != 1:
        return None
    scale = max(twos, fives)
    scaled = value.numerator * (2 ** (scale - twos)) * (5 ** (scale - fives))
    if scale == 0:
        return str(scaled)
    digits = str(scaled).zfill(scale + 1)
    return f"{digits[:-scale]}.{digits[-scale:]}".rstrip("0").rstrip(".")


def validate_command(operation: str, request: dict[str, Any]) -> str:
    if operation not in EXPECTED_COMMAND_REGISTRY or type(request) is not dict:
        return "MALFORMED_REQUEST"
    schema = EXPECTED_COMMAND_REGISTRY[operation]
    if set(request) != set(schema["request_fields"]):
        return "MALFORMED_REQUEST"
    nullable = frozenset(schema["nullable_fields"])
    if any(request[name] is None for name in request if name not in nullable):
        return "MALFORMED_REQUEST"
    for name, spec in schema["field_schemas"].items():
        value = request[name]
        if value is None:
            continue
        kind = spec["type"]
        if kind == "id" and not valid_id(value, spec["prefix"]):
            return "MALFORMED_REQUEST"
        if kind == "constant" and value != spec["value"]:
            return "MALFORMED_REQUEST"
        if kind == "enum" and (type(value) is not str or value not in spec["values"]):
            return "MALFORMED_REQUEST"
        if kind == "decimal" and not valid_decimal(value):
            return "MALFORMED_REQUEST"
        if kind == "timestamp" and not valid_timestamp(value):
            return "MALFORMED_REQUEST"
        if kind == "positive_integer" and (type(value) is not int or value <= 0):
            return "MALFORMED_REQUEST"
    if request["command_id"] != request["idempotency_key"]:
        return "MALFORMED_REQUEST"
    if operation in {"SUBMIT_ORDER", "REPLACE_ORDER"}:
        if (request["time_in_force"] == "GTD") != (request["expire_at_utc"] is not None):
            return "MALFORMED_REQUEST"
        if operation == "SUBMIT_ORDER" and (request["order_type"] == "LIMIT") != (
            request["limit_price"] is not None
        ):
            return "MALFORMED_REQUEST"
    if (request["source_type"] == "STRATEGY_INSTANCE") != (
        request["strategy_instance_id"] is not None
    ):
        return "MALFORMED_REQUEST"
    if operation == "REPLACE_ORDER" and request["replacement_order_id"] == request["order_id"]:
        return "MALFORMED_REQUEST"
    return "VALID"


def idempotency_decision(stored: str | None, request: dict[str, Any]) -> str:
    current = canonical_command_fingerprint(request)
    if stored is None:
        return "RESERVE_BEFORE_SIDE_EFFECT"
    return "REPLAY_SUCCESS" if stored == current else "IDEMPOTENCY_CONFLICT"


def validate_event_payload(event_type: str, payload: dict[str, Any]) -> str:
    schema = EXPECTED_EVENT_SCHEMAS.get(event_type)
    if (
        schema is None
        or type(payload) is not dict
        or set(payload) != set(schema["safe_payload_fields"])
    ):
        return "MALFORMED_EVENT"
    if any(payload[name] is None for name in payload if name not in schema["nullable_fields"]):
        return "MALFORMED_EVENT"
    for name, spec in schema["field_schemas"].items():
        value = payload[name]
        if value is None:
            continue
        kind = spec["type"]
        if kind == "string" and type(value) is not str:
            return "MALFORMED_EVENT"
        if kind == "id" and not valid_id(value, spec["prefix"]):
            return "MALFORMED_EVENT"
        if kind == "decimal" and not valid_decimal(value, positive=False):
            return "MALFORMED_EVENT"
        if kind == "positive_integer" and (type(value) is not int or value <= 0):
            return "MALFORMED_EVENT"
        if kind == "enum" and value not in spec["values"]:
            return "MALFORMED_EVENT"
    return "VALID"


def validate_event(event: dict[str, Any], order_scope: dict[str, str] | None = None) -> str:
    schema = EXPECTED_EVENT_ENVELOPE
    if type(event) is not dict or set(event) != set(schema["fields"]):
        return "MALFORMED_EVENT"
    nullable = frozenset(schema["nullable_fields"])
    if any(event[name] is None for name in event if name not in nullable):
        return "MALFORMED_EVENT"
    for name, spec in schema["field_schemas"].items():
        value = event[name]
        if value is None or spec["type"] in {"event_safe_payload", "enum"}:
            continue
        if spec["type"] == "id" and not valid_id(value, spec["prefix"]):
            return "MALFORMED_EVENT"
        if spec["type"] == "positive_integer" and (type(value) is not int or value <= 0):
            return "MALFORMED_EVENT"
        if spec["type"] == "timestamp" and not valid_timestamp(value):
            return "MALFORMED_EVENT"
        if spec["type"] == "non_empty_string" and (type(value) is not str or not value):
            return "MALFORMED_EVENT"
        if spec["type"] == "sha256_hex" and (
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            return "MALFORMED_EVENT"
    if event["event_type"] not in EXPECTED_EVENT_SCHEMAS:
        return "MALFORMED_EVENT"
    if event["environment"] not in {"PAPER", "TESTNET", "LIVE"}:
        return "MALFORMED_EVENT"
    if validate_event_payload(event["event_type"], event["safe_payload"]) != "VALID":
        return "MALFORMED_EVENT"
    if order_scope is not None and any(
        event[name] != order_scope[name] for name in schema["scope_fields"]
    ):
        return "ORDER_SCOPE_MISMATCH"
    if event["event_fingerprint_sha256"] != canonical_event_fingerprint(event):
        return "EVENT_FINGERPRINT_MISMATCH"
    return "VALID"


def ingest_event(current_version: int, known: dict[str, str], event: dict[str, Any]) -> str:
    fingerprint = canonical_event_fingerprint(event)
    identity = event["audit_event_id"]
    if identity in known:
        return "REPLAY_SUCCESS" if known[identity] == fingerprint else "EVENT_IDENTITY_CONFLICT"
    if event["aggregate_version"] <= current_version:
        return "STALE_EVENT"
    return "ACCEPTED" if event["aggregate_version"] == current_version + 1 else "EVENT_VERSION_GAP"


def resolve_dynamic_target(
    rule: str, *, stored_state: str | None = None, trusted_fact: str | None = None
) -> str:
    rules = EXPECTED_DYNAMIC_RULES
    if rule == "RESTORE_PRE_REQUEST_STATE":
        if stored_state not in rules[rule]["allowed_stored_states"]:
            raise ValueError("INVALID_LIFECYCLE_TRANSITION")
        return cast(str, stored_state)
    targets = rules[rule]["trusted_fact_targets"]
    if trusted_fact not in targets:
        raise ValueError("TRUSTED_CONTEXT_FAILURE")
    return cast(str, targets[trusted_fact])


def canonical_fill_fingerprint(fill: dict[str, Any]) -> str:
    return canonical_fingerprint(
        {
            key: fill[key]
            for key in EXPECTED_FILL_FACT_SCHEMA["fields"]
            if key != "fill_fingerprint_sha256"
        }
    )


def valid_asset_reference(value: Any) -> bool:
    return (
        type(value) is dict
        and set(value)
        == {"venue_asset_code", "canonical_display_code", "asset_namespace", "mapping_status"}
        and all(
            type(value[name]) is str and value[name]
            for name in ("venue_asset_code", "canonical_display_code", "asset_namespace")
        )
        and value["mapping_status"] in {"EXACT", "EXPLICIT_ALIAS"}
    )


class M05PrevalidatedInstrumentHistory:
    """Nominal result of the upstream canonical M0.5 validation boundary."""

    def __init__(self, history_by_id: dict[str, list[dict[str, Any]]]) -> None:
        self.history_by_id = MappingProxyType(history_by_id)


class CoreAcceptedFillProjection:
    """Nominal Core-owned projection derived from accepted M0.7 lifecycle history."""

    def __init__(self, fill_ids_by_order_id: dict[str, tuple[str, ...]]) -> None:
        self.fill_ids_by_order_id = MappingProxyType(fill_ids_by_order_id)


def represent_successful_m05_validation(
    history_by_id: dict[str, list[dict[str, Any]]],
) -> M05PrevalidatedInstrumentHistory:
    """Reference boundary: input is supplied only after canonical M0.5 validation succeeds."""
    return M05PrevalidatedInstrumentHistory(history_by_id)


def project_accepted_fill_history(
    fill_ids_by_order_id: dict[str, tuple[str, ...]],
) -> CoreAcceptedFillProjection:
    """Reference boundary: Core derives this mapping from accepted aggregate history."""
    return CoreAcceptedFillProjection(fill_ids_by_order_id)


def resolve_fill_instrument_binding(
    fill: dict[str, Any], instrument_history: Any
) -> tuple[str, dict[str, Any] | None]:
    if type(instrument_history) is not M05PrevalidatedInstrumentHistory:
        return "TRUSTED_CONTEXT_FAILURE", None
    instrument_history_by_id = instrument_history.history_by_id
    if fill["instrument_id"] not in instrument_history_by_id:
        return "TRUSTED_CONTEXT_FAILURE", None
    history = instrument_history_by_id.get(fill["instrument_id"])
    if type(history) is not list or not history:
        return "TRUSTED_CONTEXT_FAILURE", None
    versions: list[int] = []
    identity: tuple[Any, ...] | None = None
    for record in history:
        if type(record) is not dict or set(record) != set(EXPECTED_INSTRUMENT_RECORD_FIELDS):
            return "TRUSTED_CONTEXT_FAILURE", None
        version = record.get("metadata_version")
        if type(version) is not int or version <= 0 or version in versions:
            return "TRUSTED_CONTEXT_FAILURE", None
        if versions and version <= versions[-1]:
            return "TRUSTED_CONTEXT_FAILURE", None
        versions.append(version)
        if record.get("instrument_id") != fill["instrument_id"]:
            return "TRUSTED_CONTEXT_FAILURE", None
        candidate_identity = tuple(
            record.get(name)
            for name in ("exchange_id", "environment", "market_type", "venue_symbol")
        )
        if identity is not None and candidate_identity != identity:
            return "TRUSTED_CONTEXT_FAILURE", None
        identity = candidate_identity
        for name in (
            "instrument_id",
            "workspace_id",
            "exchange_id",
            "environment",
            "market_type",
            "instrument_type",
            "venue_symbol",
            "display_symbol",
            "catalog_snapshot_id",
            "source_adapter_family_id",
        ):
            if type(record.get(name)) is not str or not record[name]:
                return "TRUSTED_CONTEXT_FAILURE", None
        if not valid_decimal(record.get("price_tick")) or not valid_decimal(
            record.get("quantity_step")
        ):
            return "TRUSTED_CONTEXT_FAILURE", None
        for name in ("observed_at_utc", "effective_at_utc", "stale_after_utc"):
            if not valid_timestamp(record.get(name)):
                return "TRUSTED_CONTEXT_FAILURE", None
        if not (
            datetime.fromisoformat(record["observed_at_utc"].replace("Z", "+00:00"))
            <= datetime.fromisoformat(record["effective_at_utc"].replace("Z", "+00:00"))
            < datetime.fromisoformat(record["stale_after_utc"].replace("Z", "+00:00"))
        ):
            return "TRUSTED_CONTEXT_FAILURE", None
        for name in ("base_asset_reference", "quote_asset_reference"):
            if (
                not valid_asset_reference(record.get(name))
                or record[name]["asset_namespace"] != record["exchange_id"]
            ):
                return "TRUSTED_CONTEXT_FAILURE", None
        settlement = record.get("settlement_asset_reference")
        if settlement is not None and (
            not valid_asset_reference(settlement)
            or settlement["asset_namespace"] != record["exchange_id"]
        ):
            return "TRUSTED_CONTEXT_FAILURE", None
    matches = [
        record
        for record in history
        if record["metadata_version"] == fill["instrument_metadata_version"]
    ]
    if len(matches) != 1:
        return "TRUSTED_CONTEXT_FAILURE", None
    resolved = matches[0]
    if any(resolved[name] != fill[name] for name in ("workspace_id", "exchange_id", "environment")):
        return "TRUSTED_CONTEXT_FAILURE", None
    return "VALID", resolved


def validate_fill(fill: Any) -> str:
    schema = EXPECTED_FILL_FACT_SCHEMA
    if type(fill) is not dict or set(fill) != set(schema["fields"]):
        return "MALFORMED_FILL"
    for name, spec in schema["field_schemas"].items():
        value = fill[name]
        if value is None:
            if name not in schema["nullable_fields"]:
                return "MALFORMED_FILL"
            continue
        kind = spec["type"]
        if kind == "id" and not valid_id(value, spec["prefix"]):
            return "MALFORMED_FILL"
        if kind == "enum" and value not in spec["values"]:
            return "MALFORMED_FILL"
        if kind == "non_empty_string" and (type(value) is not str or not value):
            return "MALFORMED_FILL"
        if kind == "positive_integer" and (type(value) is not int or value <= 0):
            return "MALFORMED_FILL"
        if kind == "decimal" and not valid_decimal(value, spec["constraint"] == "positive"):
            return "MALFORMED_FILL"
        if kind == "timestamp" and not valid_timestamp(value):
            return "MALFORMED_FILL"
        if kind == "asset_reference" and not valid_asset_reference(value):
            return "MALFORMED_FILL"
        if kind == "sha256_hex" and (
            type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
        ):
            return "MALFORMED_FILL"
    if (fill["fee_kind"] == "NONE") != (
        fill["fee_quantity"] == "0" and fill["fee_asset_reference"] is None
    ):
        return "MALFORMED_FILL"
    if fill["fee_kind"] == "CHARGE" and (
        fill["fee_quantity"] == "0"
        or fill["fee_asset_reference"] is None
        or fill["fee_asset_reference"]["asset_namespace"] != fill["exchange_id"]
    ):
        return "MALFORMED_FILL"
    if fill["fill_fingerprint_sha256"] != canonical_fill_fingerprint(fill):
        return "MALFORMED_FILL"
    return "VALID"


def accepted_unique_economic_fills(
    fills: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    by_fill: dict[str, str] = {}
    by_external: dict[tuple[str, str, str, str], str] = {}
    unique: list[dict[str, Any]] = []
    for fill in fills:
        if validate_fill(fill) != "VALID":
            return "MALFORMED_FILL", []
        fingerprint = fill["fill_fingerprint_sha256"]
        external = tuple(fill[name] for name in EXPECTED_FILL_CONTRACT["external_dedupe_scope"])
        if fill["fill_id"] in by_fill:
            if by_fill[fill["fill_id"]] != fingerprint:
                return "FILL_IDENTITY_CONFLICT", []
            continue
        economics = canonical_fingerprint(
            {
                key: fill[key]
                for key in EXPECTED_FILL_FACT_SCHEMA["fields"]
                if key not in {"fill_id", "fill_fingerprint_sha256"}
            }
        )
        if external in by_external:
            if by_external[external] != economics:
                return "FILL_IDENTITY_CONFLICT", []
            by_fill[fill["fill_id"]] = fingerprint
            continue
        by_fill[fill["fill_id"]] = fingerprint
        by_external[external] = economics
        unique.append(fill)
    return "VALID", unique


def apply_fills(quantity: str, fills: list[dict[str, Any]]) -> tuple[str, str, str | None, str]:
    order_quantity = Fraction(quantity)
    result, unique = accepted_unique_economic_fills(fills)
    if result != "VALID":
        return "0", canonical_decimal_string(order_quantity) or "0", None, result
    total = sum((Fraction(fill["executed_quantity"]) for fill in unique), Fraction(0))
    if total > order_quantity:
        return (
            "0",
            canonical_decimal_string(order_quantity) or "0",
            None,
            "FILL_PROGRESSION_CONFLICT",
        )
    notional = sum(
        (
            Fraction(fill["executed_quantity"]) * Fraction(fill["execution_price"])
            for fill in unique
        ),
        Fraction(0),
    )
    average = canonical_decimal_string(notional / total) if total else None
    state = "FILLED" if total == order_quantity else "PARTIALLY_FILLED" if total else "ACKNOWLEDGED"
    return (
        canonical_decimal_string(total) or "0",
        canonical_decimal_string(order_quantity - total) or "0",
        average,
        state,
    )


def resolve_trusted_fill(
    fill_id: str,
    fills_by_id: dict[str, dict[str, Any]],
    instrument_history: M05PrevalidatedInstrumentHistory,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    fill = fills_by_id.get(fill_id)
    if fill is None or fill.get("fill_id") != fill_id:
        return "TRUSTED_CONTEXT_FAILURE", None, None
    result = validate_fill(fill)
    if result != "VALID":
        return result, None, None
    result, instrument = resolve_fill_instrument_binding(fill, instrument_history)
    if result != "VALID" or instrument is None:
        return "TRUSTED_CONTEXT_FAILURE", None, None
    return "VALID", fill, instrument


def validate_fill_event_reference(
    event: dict[str, Any],
    fills_by_id: dict[str, dict[str, Any]],
    instrument_history: M05PrevalidatedInstrumentHistory,
    accepted_history: Any,
) -> str:
    if event.get("event_type") not in {"ORDER_PARTIALLY_FILLED", "ORDER_FILLED"}:
        return "VALID"
    if type(accepted_history) is not CoreAcceptedFillProjection:
        return "TRUSTED_CONTEXT_FAILURE"
    sequence = accepted_history.fill_ids_by_order_id.get(event["order_id"])
    if (
        type(sequence) is not tuple
        or not sequence
        or any(type(item) is not str for item in sequence)
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    current_id = event["safe_payload"]["fill_id"]
    history: list[dict[str, Any]] = []
    for fill_id in sequence:
        result, fill, _instrument = resolve_trusted_fill(fill_id, fills_by_id, instrument_history)
        if result != "VALID" or fill is None:
            return result
        history.append(fill)
    result, unique = accepted_unique_economic_fills(history)
    if result != "VALID":
        return result
    if len(unique) != len(history):
        return "TRUSTED_CONTEXT_FAILURE"
    scope = (
        "order_id",
        "environment",
        "workspace_id",
        "portfolio_id",
        "exchange_account_id",
        "exchange_id",
        "instrument_id",
        "execution_route_id",
    )
    if any(any(fill[name] != event[name] for name in scope) for fill in history):
        return "TRUSTED_CONTEXT_FAILURE"
    current_count = sequence.count(current_id)
    if current_count > 1:
        return "TRUSTED_CONTEXT_FAILURE"
    if current_count == 0:
        result, current, _instrument = resolve_trusted_fill(
            current_id, fills_by_id, instrument_history
        )
        if result != "VALID" or current is None:
            return result
        if any(current[name] != event[name] for name in scope):
            return "TRUSTED_CONTEXT_FAILURE"
        if current["venue_trade_id"] != event["safe_payload"]["venue_trade_id"]:
            return "TRUSTED_CONTEXT_FAILURE"
        cumulative = sum((Fraction(fill["executed_quantity"]) for fill in unique), Fraction(0))
        if (
            canonical_decimal_string(cumulative)
            != event["safe_payload"]["cumulative_executed_quantity"]
        ):
            return "TRUSTED_CONTEXT_FAILURE"
        candidate_result, candidate_unique = accepted_unique_economic_fills([*history, current])
        if candidate_result != "VALID":
            return candidate_result
        return (
            "REPLAY_SUCCESS" if len(candidate_unique) == len(unique) else "TRUSTED_CONTEXT_FAILURE"
        )
    current = fills_by_id[current_id]
    if current["venue_trade_id"] != event["safe_payload"]["venue_trade_id"]:
        return "TRUSTED_CONTEXT_FAILURE"
    cumulative = sum((Fraction(fill["executed_quantity"]) for fill in unique), Fraction(0))
    return (
        "VALID"
        if canonical_decimal_string(cumulative)
        == event["safe_payload"]["cumulative_executed_quantity"]
        else "TRUSTED_CONTEXT_FAILURE"
    )


def command(operation: str = "SUBMIT_ORDER") -> dict[str, Any]:
    values: dict[str, Any] = {
        "command_id": f"cmd_{UUID}",
        "operation_type": operation,
        "authority_context_id": f"authctx_{UUID}",
        "environment": "TESTNET",
        "workspace_id": f"ws_{UUID}",
        "portfolio_id": f"port_{UUID}",
        "exchange_account_id": f"xacc_{UUID}",
        "strategy_instance_id": f"sinst_{UUID}",
        "source_type": "STRATEGY_INSTANCE",
        "instrument_id": f"instr_{UUID}",
        "execution_route_id": f"xroute_{UUID}",
        "correlation_id": f"corr_{UUID}",
        "causation_id": f"cause_{UUID}",
        "idempotency_key": f"cmd_{UUID}",
        "order_intent_id": f"oint_{UUID}",
        "order_id": f"ord_{UUID}",
        "side": "BUY",
        "order_type": "LIMIT",
        "quantity": "1.25",
        "limit_price": "100.5",
        "time_in_force": "GTC",
        "expire_at_utc": None,
        "expected_order_version": 1,
        "reason_code": "OPERATOR_REQUEST",
        "replacement_order_id": f"ord_{UUID[:-1]}c",
    }
    return {name: values[name] for name in EXPECTED_COMMAND_REGISTRY[operation]["request_fields"]}


def fill_item(
    fill_id: str = f"fill_{UUID}",
    trade: str = "trade-1",
    qty: str = "0.25",
    price: str = "100",
    *,
    fee: str = "0.1",
    fee_asset: dict[str, str] | None = None,
    environment: str = "TESTNET",
    account: str = f"xacc_{UUID}",
    exchange: str = "venue-a",
) -> dict[str, Any]:
    if not fill_id.startswith("fill_"):
        variant = str((int(fill_id.rsplit("-", 1)[-1]) % 9) + 1)
        fill_id = f"fill_{UUID[:-1]}{variant}"
    if fee_asset is None and fee != "0":
        fee_asset = {
            "venue_asset_code": "BNB",
            "canonical_display_code": "BNB",
            "asset_namespace": exchange,
            "mapping_status": "EXACT",
        }
    values: dict[str, Any] = {
        "fill_id": fill_id,
        "order_id": f"ord_{UUID}",
        "environment": environment,
        "workspace_id": f"ws_{UUID}",
        "portfolio_id": f"port_{UUID}",
        "exchange_account_id": account,
        "exchange_id": exchange,
        "instrument_id": f"instr_{UUID}",
        "instrument_metadata_version": 3,
        "execution_route_id": f"xroute_{UUID}",
        "venue_trade_id": trade,
        "side": "BUY",
        "executed_quantity": qty,
        "execution_price": price,
        "executed_at_utc": "2026-08-09T00:00:00Z",
        "fee_kind": "NONE" if fee == "0" else "CHARGE",
        "fee_quantity": fee,
        "fee_asset_reference": None if fee == "0" else fee_asset,
        "fill_fingerprint_sha256": "",
    }
    values["fill_fingerprint_sha256"] = canonical_fill_fingerprint(values)
    return values


def instrument_record(fill: dict[str, Any], version: int = 3, **overrides: Any) -> dict[str, Any]:
    asset = lambda code: {
        "venue_asset_code": code,
        "canonical_display_code": code,
        "asset_namespace": fill["exchange_id"],
        "mapping_status": "EXACT",
    }
    record: dict[str, Any] = {
        "instrument_id": fill["instrument_id"],
        "workspace_id": fill["workspace_id"],
        "exchange_id": fill["exchange_id"],
        "environment": fill["environment"],
        "market_type": "SPOT",
        "instrument_type": "SPOT_PAIR",
        "venue_symbol": "BTCUSDT",
        "display_symbol": "BTC/USDT",
        "base_asset_reference": asset("BTC"),
        "quote_asset_reference": asset("USDT"),
        "settlement_asset_reference": None,
        "trading_status": "TRADING",
        "price_tick": "0.01",
        "quantity_step": "0.0001",
        "min_quantity": "0.0001",
        "max_quantity": "100",
        "min_notional": "5",
        "max_notional": None,
        "contract_size": None,
        "contract_value_currency": None,
        "derivative_settlement_type": None,
        "expiry_at_utc": None,
        "strike_price": None,
        "option_side": None,
        "catalog_snapshot_id": f"catalog-{version}",
        "metadata_version": version,
        "observed_at_utc": "2026-01-01T00:00:00Z",
        "effective_at_utc": "2026-01-01T00:00:00Z",
        "stale_after_utc": "2027-01-01T00:00:00Z",
        "source_adapter_family_id": "adapter-family",
    }
    record.update(overrides)
    return record


def test_contract_exact_protocol_and_cross_roots() -> None:
    validate_contract(CONTRACT)
    for dependency in CONTRACT["cross_contract_dependencies"]:
        actual = resolve_pointer(CANONICAL[dependency["contract"]], dependency["json_pointer"])
        assert canonical_fingerprint(actual) == dependency["content_fingerprint_sha256"]


def test_canonical_audit_event_identity_is_exact() -> None:
    entities = {
        item["canonical_name"]: item
        for item in CANONICAL["canonical_domain_vocabulary.json"]["entity_kinds"]
    }
    assert entities["AuditEvent"]["id_prefix"] == "evt"
    assert entities["AuditEvent"]["id_field"] == "audit_event_id"
    assert CONTRACT["identity_policy"]["durable_ids"]["AuditEvent"] == {
        "field": "audit_event_id",
        "prefix": "evt",
    }
    assert "event_id" not in CONTRACT["event_contract"]["fields"]


def test_real_closed_command_validation() -> None:
    request = command()
    assert validate_command("SUBMIT_ORDER", request) == "VALID"
    for name, value in (
        ("environment", None),
        ("command_id", "cmd-1"),
        ("order_id", f"fill_{UUID}"),
        ("quantity", 1.25),
        ("quantity", "Infinity"),
        ("side", "HOLD"),
    ):
        broken = {**request, name: value}
        if name == "command_id":
            broken["idempotency_key"] = value
        assert validate_command("SUBMIT_ORDER", broken) == "MALFORMED_REQUEST"
    assert validate_command("SUBMIT_ORDER", {**request, "extra": True}) == "MALFORMED_REQUEST"
    cancel = command("CANCEL_ORDER")
    assert validate_command("CANCEL_ORDER", cancel) == "VALID"
    assert (
        validate_command("CANCEL_ORDER", {**cancel, "expected_order_version": True})
        == "MALFORMED_REQUEST"
    )
    assert validate_command("SUBMIT_ORDER", {**request, "quantity": "1.00"}) == "MALFORMED_REQUEST"


def test_fingerprint_exact_exclusion_and_semantic_conflicts() -> None:
    request = command()
    stored = canonical_command_fingerprint(request)
    assert (
        idempotency_decision(stored, {**request, "correlation_id": f"corr_{UUID[:-1]}c"})
        == "REPLAY_SUCCESS"
    )
    assert idempotency_decision(stored, {**request, "quantity": "2"}) == "IDEMPOTENCY_CONFLICT"
    assert (
        idempotency_decision(stored, {**request, "causation_id": f"cause_{UUID[:-1]}c"})
        == "IDEMPOTENCY_CONFLICT"
    )
    assert canonical_command_fingerprint({**request, "arbitrary": "x"}) != stored


def test_event_conflict_ordering_and_exact_payload() -> None:
    payload = {"venue_order_id": "venue-1"}
    event: dict[str, Any] = {
        "audit_event_id": f"evt_{UUID}",
        "event_type": "ORDER_ACKNOWLEDGED",
        "order_id": f"ord_{UUID}",
        "aggregate_version": 4,
        "correlation_id": f"corr_{UUID}",
        "causation_id": f"cause_{UUID}",
        "command_id": f"cmd_{UUID}",
        "environment": "TESTNET",
        "workspace_id": f"ws_{UUID}",
        "portfolio_id": f"port_{UUID}",
        "exchange_account_id": f"xacc_{UUID}",
        "exchange_id": "venue-a",
        "instrument_id": f"instr_{UUID}",
        "execution_route_id": f"xroute_{UUID}",
        "occurred_at_utc": "2026-08-09T00:00:00Z",
        "safe_payload": payload,
        "event_fingerprint_sha256": "",
    }
    event["event_fingerprint_sha256"] = canonical_event_fingerprint(event)
    identity = cast(str, event["audit_event_id"])
    known = {identity: canonical_event_fingerprint(event)}
    assert validate_event(event) == "VALID"
    assert ingest_event(4, known, event) == "REPLAY_SUCCESS"
    assert ingest_event(4, {identity: "different"}, event) == "EVENT_IDENTITY_CONFLICT"
    assert ingest_event(4, {}, event) == "STALE_EVENT"
    assert ingest_event(1, {}, event) == "EVENT_VERSION_GAP"
    assert validate_event_payload("ORDER_ACKNOWLEDGED", payload) == "VALID"
    assert validate_event_payload("ORDER_ACKNOWLEDGED", {}) == "MALFORMED_EVENT"
    assert validate_event_payload("ORDER_ACKNOWLEDGED", {**payload, "raw": {}}) == "MALFORMED_EVENT"
    for changed in (
        {"order_id": f"ord_{UUID[:-1]}c"},
        {"exchange_account_id": f"xacc_{UUID[:-1]}c"},
        {"environment": "LIVE"},
    ):
        conflicting = {**event, **changed}
        conflicting["event_fingerprint_sha256"] = canonical_event_fingerprint(conflicting)
        assert ingest_event(4, known, conflicting) == "EVENT_IDENTITY_CONFLICT"
    scope = {name: cast(str, event[name]) for name in EXPECTED_EVENT_ENVELOPE["scope_fields"]}
    assert validate_event(event, scope) == "VALID"
    assert validate_event(event, {**scope, "environment": "LIVE"}) == "ORDER_SCOPE_MISMATCH"
    assert validate_event({**event, "extra": True}) == "MALFORMED_EVENT"
    malformed_payload = {**event, "safe_payload": {}}
    malformed_payload["event_fingerprint_sha256"] = canonical_event_fingerprint(malformed_payload)
    assert validate_event(malformed_payload) == "MALFORMED_EVENT"
    assert validate_event({**event, "event_fingerprint_sha256": "0" * 64}) == (
        "EVENT_FINGERPRINT_MISMATCH"
    )
    assert {"MALFORMED_EVENT", "EVENT_FINGERPRINT_MISMATCH"} <= set(CONTRACT["failure_taxonomy"])


def test_dynamic_targets_resolve_only_to_closed_states() -> None:
    assert (
        resolve_dynamic_target("RESTORE_PRE_REQUEST_STATE", stored_state="ACKNOWLEDGED")
        == "ACKNOWLEDGED"
    )
    assert (
        resolve_dynamic_target("RESTORE_PRE_REQUEST_STATE", stored_state="PARTIALLY_FILLED")
        == "PARTIALLY_FILLED"
    )
    with pytest.raises(ValueError, match="INVALID_LIFECYCLE_TRANSITION"):
        resolve_dynamic_target("RESTORE_PRE_REQUEST_STATE", stored_state="FILLED")
    targets = CONTRACT["order_lifecycle"]["dynamic_target_rules"][
        "RESOLVE_TRUSTED_RECONCILIATION_FACT"
    ]["trusted_fact_targets"]
    assert set(targets.values()) <= EXPECTED_STATES
    for fact, expected in targets.items():
        assert (
            resolve_dynamic_target("RESOLVE_TRUSTED_RECONCILIATION_FACT", trusted_fact=fact)
            == expected
        )
    with pytest.raises(ValueError, match="TRUSTED_CONTEXT_FAILURE"):
        resolve_dynamic_target("RESOLVE_TRUSTED_RECONCILIATION_FACT", trusted_fact="ARBITRARY")


def test_fill_dedupes_durable_and_exact_external_scope() -> None:
    first = fill_item("fill-1", "trade-1")
    same_external_other_id = fill_item("fill-2", "trade-1")
    assert apply_fills("1", [first, same_external_other_id])[:2] == ("0.25", "0.75")
    conflict = fill_item("fill-2", "trade-1", qty="0.5")
    assert apply_fills("1", [first, conflict])[3] == "FILL_IDENTITY_CONFLICT"
    other_account = fill_item("fill-3", "trade-1", account=f"xacc_{UUID[:-1]}c")
    other_environment = fill_item("fill-4", "trade-1", environment="LIVE")
    other_exchange = fill_item("fill-5", "trade-1", exchange="venue-b")
    assert apply_fills("1", [first, other_account, other_environment, other_exchange])[:2] == (
        "1",
        "0",
    )
    assert apply_fills("0.2", [first])[3] == "FILL_PROGRESSION_CONFLICT"
    assert apply_fills("1", [fill_item("fill-6", "trade-6", qty="0.250")])[3] == "MALFORMED_FILL"
    float_fill: dict[str, Any] = fill_item("fill-7", "trade-7")
    float_fill["executed_quantity"] = 0.25
    assert apply_fills("1", [float_fill])[3] == "MALFORMED_FILL"
    bad_identity = fill_item("fill-8", "trade-8")
    bad_identity["fill_id"] = "fill-bad"
    assert apply_fills("1", [bad_identity])[3] == "MALFORMED_FILL"
    conflicting_identity = {**first, "execution_price": "101"}
    assert apply_fills("1", [first, conflicting_identity])[3] == "MALFORMED_FILL"
    for field in ("venue_trade_id", "exchange_id"):
        non_string = {**first, field: 1}
        assert apply_fills("1", [non_string])[3] == "MALFORMED_FILL"


def test_fill_arithmetic_is_exact_and_decimal_context_independent() -> None:
    quantity = "0.1234567890123456789012345678"
    price = "12345678901234567890.12345678"
    long_fill = fill_item("fill-9", "trade-9", qty=quantity, price=price)
    expected = (quantity, "0", price, "FILLED")
    assert apply_fills(quantity, [long_fill]) == expected
    with localcontext() as context:
        context.prec = 6
        assert apply_fills(quantity, [long_fill]) == expected


def test_instrument_metadata_binding_resolves_exact_historical_version() -> None:
    fill = fill_item()
    older, exact = instrument_record(fill, 2), instrument_record(fill, 3)
    trusted = represent_successful_m05_validation({fill["instrument_id"]: [older, exact]})
    result, resolved = resolve_fill_instrument_binding(fill, trusted)
    assert result == "VALID" and resolved is exact
    assert (
        resolve_fill_instrument_binding(fill, {fill["instrument_id"]: [exact]})[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )
    for field, value in (
        ("workspace_id", f"ws_{UUID[:-1]}c"),
        ("exchange_id", "venue-b"),
        ("environment", "LIVE"),
    ):
        wrong = instrument_record(fill, 3, **{field: value})
        context = represent_successful_m05_validation({fill["instrument_id"]: [wrong]})
        assert resolve_fill_instrument_binding(fill, context)[0] == "TRUSTED_CONTEXT_FAILURE"
    newer = represent_successful_m05_validation(
        {fill["instrument_id"]: [instrument_record(fill, 4)]}
    )
    assert resolve_fill_instrument_binding(fill, newer)[0] == "TRUSTED_CONTEXT_FAILURE"
    duplicate = represent_successful_m05_validation(
        {fill["instrument_id"]: [instrument_record(fill, 3), instrument_record(fill, 3)]}
    )
    assert resolve_fill_instrument_binding(fill, duplicate)[0] == "TRUSTED_CONTEXT_FAILURE"
    settlement = {
        "venue_asset_code": "USDT",
        "canonical_display_code": "USDT",
        "asset_namespace": "venue-b",
        "mapping_status": "EXACT",
    }
    bad_settlement = instrument_record(fill, 3, settlement_asset_reference=settlement)
    context = represent_successful_m05_validation({fill["instrument_id"]: [bad_settlement]})
    assert resolve_fill_instrument_binding(fill, context)[0] == "TRUSTED_CONTEXT_FAILURE"


def test_composite_trusted_fill_requires_exact_instrument_version() -> None:
    fill = fill_item()
    fills = {fill["fill_id"]: fill}
    missing = represent_successful_m05_validation(
        {fill["instrument_id"]: [instrument_record(fill, 4)]}
    )
    assert resolve_trusted_fill(fill["fill_id"], fills, missing)[0] == "TRUSTED_CONTEXT_FAILURE"
    exact = represent_successful_m05_validation(
        {fill["instrument_id"]: [instrument_record(fill, 3)]}
    )
    result, resolved_fill, resolved_instrument = resolve_trusted_fill(fill["fill_id"], fills, exact)
    assert result == "VALID" and resolved_fill is fill and resolved_instrument is not None


def test_fill_and_asset_json_key_order_is_not_authority() -> None:
    fill = fill_item()
    reordered = dict(reversed(list(fill.items())))
    reordered["fee_asset_reference"] = dict(reversed(list(fill["fee_asset_reference"].items())))
    assert validate_fill(reordered) == "VALID"
    assert canonical_fill_fingerprint(reordered) == canonical_fill_fingerprint(fill)


def test_complete_fill_fee_semantics_and_third_asset() -> None:
    fill = fill_item()
    assert validate_fill(fill) == "VALID"
    assert fill["fee_asset_reference"]["venue_asset_code"] == "BNB"
    missing_asset = {**fill, "fee_asset_reference": None}
    missing_asset["fill_fingerprint_sha256"] = canonical_fill_fingerprint(missing_asset)
    assert validate_fill(missing_asset) == "MALFORMED_FILL"
    unknown = copy.deepcopy(fill)
    unknown["fee_asset_reference"]["mapping_status"] = "UNKNOWN"
    unknown["fill_fingerprint_sha256"] = canonical_fill_fingerprint(unknown)
    assert validate_fill(unknown) == "MALFORMED_FILL"
    assert validate_fill(fill_item(fee="0")) == "VALID"
    wrong_namespace = copy.deepcopy(fill)
    wrong_namespace["fee_asset_reference"]["asset_namespace"] = "venue-b"
    wrong_namespace["fill_fingerprint_sha256"] = canonical_fill_fingerprint(wrong_namespace)
    assert validate_fill(wrong_namespace) == "MALFORMED_FILL"
    for decimal_field in ("executed_quantity", "execution_price", "fee_quantity"):
        malformed = copy.deepcopy(fill)
        malformed[decimal_field] = 0.1
        assert validate_fill(malformed) == "MALFORMED_FILL"
    noncanonical = copy.deepcopy(fill)
    noncanonical["fee_quantity"] = "0.10"
    assert validate_fill(noncanonical) == "MALFORMED_FILL"
    invalid_time = copy.deepcopy(fill)
    invalid_time["executed_at_utc"] = "2026-08-09T00:00:00+00:00"
    assert validate_fill(invalid_time) == "MALFORMED_FILL"


def test_full_economic_dedupe_conflicts_include_fee_time_scope_and_route() -> None:
    original = fill_item()
    assert apply_fills("1", [original, copy.deepcopy(original)])[3] == "PARTIALLY_FILLED"
    for field, value in (
        ("fee_quantity", "0.2"),
        ("executed_at_utc", "2026-08-09T00:00:01Z"),
        ("execution_route_id", f"xroute_{UUID[:-1]}c"),
        ("instrument_id", f"instr_{UUID[:-1]}c"),
    ):
        changed = copy.deepcopy(original)
        changed[field] = value
        changed["fill_fingerprint_sha256"] = canonical_fill_fingerprint(changed)
        assert apply_fills("1", [original, changed])[3] == "FILL_IDENTITY_CONFLICT"
    changed_asset = copy.deepcopy(original)
    changed_asset["fee_asset_reference"]["venue_asset_code"] = "USDT"
    changed_asset["fill_fingerprint_sha256"] = canonical_fill_fingerprint(changed_asset)
    assert apply_fills("1", [original, changed_asset])[3] == "FILL_IDENTITY_CONFLICT"


def test_fill_event_requires_composite_trust_and_core_history() -> None:
    first = fill_item("fill-1", "trade-1")
    second = fill_item("fill-2", "trade-2", qty="0.3")
    third = fill_item("fill-3", "trade-3", qty="0.2")
    fills = {fill["fill_id"]: fill for fill in (first, second, third)}
    instruments = represent_successful_m05_validation(
        {first["instrument_id"]: [instrument_record(first, 3)]}
    )
    projection = project_accepted_fill_history(
        {first["order_id"]: (first["fill_id"], second["fill_id"], third["fill_id"])}
    )
    event: dict[str, Any] = {
        "audit_event_id": f"evt_{UUID}",
        "event_type": "ORDER_PARTIALLY_FILLED",
        "order_id": first["order_id"],
        "aggregate_version": 3,
        "correlation_id": f"corr_{UUID}",
        "causation_id": f"cause_{UUID}",
        "command_id": f"cmd_{UUID}",
        "environment": first["environment"],
        "workspace_id": first["workspace_id"],
        "portfolio_id": first["portfolio_id"],
        "exchange_account_id": first["exchange_account_id"],
        "exchange_id": first["exchange_id"],
        "instrument_id": first["instrument_id"],
        "execution_route_id": first["execution_route_id"],
        "occurred_at_utc": "2026-08-09T00:00:01Z",
        "safe_payload": {
            "fill_id": third["fill_id"],
            "venue_trade_id": third["venue_trade_id"],
            "cumulative_executed_quantity": "0.75",
        },
        "event_fingerprint_sha256": "",
    }
    event["event_fingerprint_sha256"] = canonical_event_fingerprint(event)
    assert validate_fill_event_reference(event, fills, instruments, projection) == "VALID"
    assert (
        validate_fill_event_reference(event, fills, instruments, [third["fill_id"]])
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert (
        validate_fill_event_reference(event, fills, instruments, project_accepted_fill_history({}))
        == "TRUSTED_CONTEXT_FAILURE"
    )
    malformed = CoreAcceptedFillProjection(cast(Any, {first["order_id"]: [third["fill_id"]]}))
    assert (
        validate_fill_event_reference(event, fills, instruments, malformed)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    other_order = project_accepted_fill_history({f"ord_{UUID[:-1]}c": (third["fill_id"],)})
    assert (
        validate_fill_event_reference(event, fills, instruments, other_order)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    event["safe_payload"]["cumulative_executed_quantity"] = "0.2"
    assert (
        validate_fill_event_reference(event, fills, instruments, projection)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    event["safe_payload"]["cumulative_executed_quantity"] = "0.75"
    missing_version = represent_successful_m05_validation(
        {first["instrument_id"]: [instrument_record(first, 4)]}
    )
    assert (
        validate_fill_event_reference(event, fills, missing_version, projection)
        == "TRUSTED_CONTEXT_FAILURE"
    )

    duplicate = fill_item("fill-4", first["venue_trade_id"])
    duplicate_fills = {first["fill_id"]: first, duplicate["fill_id"]: duplicate}
    canonical = project_accepted_fill_history({first["order_id"]: (first["fill_id"],)})
    event["safe_payload"] = {
        "fill_id": duplicate["fill_id"],
        "venue_trade_id": duplicate["venue_trade_id"],
        "cumulative_executed_quantity": "0.25",
    }
    assert (
        validate_fill_event_reference(event, duplicate_fills, instruments, canonical)
        == "REPLAY_SUCCESS"
    )
    duplicate_durable_projection = project_accepted_fill_history(
        {first["order_id"]: (first["fill_id"], first["fill_id"])}
    )
    event["safe_payload"] = {
        "fill_id": first["fill_id"],
        "venue_trade_id": first["venue_trade_id"],
        "cumulative_executed_quantity": "0.25",
    }
    assert (
        validate_fill_event_reference(
            event, duplicate_fills, instruments, duplicate_durable_projection
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )
    duplicate_external_projection = project_accepted_fill_history(
        {first["order_id"]: (first["fill_id"], duplicate["fill_id"])}
    )
    event["safe_payload"]["fill_id"] = duplicate["fill_id"]
    assert (
        validate_fill_event_reference(
            event, duplicate_fills, instruments, duplicate_external_projection
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )
    event["safe_payload"]["venue_trade_id"] = "wrong-trade"
    assert (
        validate_fill_event_reference(event, duplicate_fills, instruments, canonical)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    event["safe_payload"]["venue_trade_id"] = duplicate["venue_trade_id"]
    event["safe_payload"]["cumulative_executed_quantity"] = "0.5"
    assert (
        validate_fill_event_reference(event, duplicate_fills, instruments, canonical)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    event["safe_payload"] = {
        "fill_id": second["fill_id"],
        "venue_trade_id": second["venue_trade_id"],
        "cumulative_executed_quantity": "0.25",
    }
    assert (
        validate_fill_event_reference(event, fills, instruments, canonical)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    conflicting = fill_item("fill-5", first["venue_trade_id"], fee="0.2")
    conflict_fills = {first["fill_id"]: first, conflicting["fill_id"]: conflicting}
    event["safe_payload"] = {
        "fill_id": conflicting["fill_id"],
        "venue_trade_id": conflicting["venue_trade_id"],
        "cumulative_executed_quantity": "0.25",
    }
    assert (
        validate_fill_event_reference(event, conflict_fills, instruments, canonical)
        == "FILL_IDENTITY_CONFLICT"
    )


def test_fill_contract_mutation_fails_closed() -> None:
    for mutation in ("add", "remove", "nullable", "fee_rule"):
        weakened = copy.deepcopy(CONTRACT)
        if mutation == "add":
            weakened["fill_contract"]["fact_fields"].append("arbitrary")
        elif mutation == "remove":
            weakened["fill_contract"]["field_schemas"].pop("fee_asset_reference")
        elif mutation == "nullable":
            weakened["fill_contract"]["nullable_fields"].append("fee_quantity")
        else:
            weakened["fill_contract"]["fee_semantics"]["CHARGE"] = "fee asset optional"
        with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
            validate_contract(weakened)


@pytest.mark.parametrize(
    "mutation", ["add_field", "remove_field", "nullable", "enum", "type", "event_schema"]
)
def test_coordinated_mutable_schema_weakening_fails_closed(mutation: str) -> None:
    weakened = copy.deepcopy(CONTRACT)
    submit = weakened["command_registry"]["SUBMIT_ORDER"]
    if mutation == "add_field":
        submit["request_fields"].append("arbitrary")
        submit["field_schemas"]["arbitrary"] = {"type": "string"}
    elif mutation == "remove_field":
        submit["request_fields"].remove("instrument_id")
        submit["field_schemas"].pop("instrument_id")
    elif mutation == "nullable":
        submit["nullable_fields"].append("environment")
    elif mutation == "enum":
        submit["field_schemas"]["side"]["values"].append("HOLD")
    elif mutation == "type":
        submit["field_schemas"]["quantity"]["type"] = "string"
    else:
        weakened["event_contract"]["event_schema_registry"]["ORDER_ACKNOWLEDGED"][
            "safe_payload_fields"
        ].append("raw")
        weakened["event_contract"]["event_schema_registry"]["ORDER_ACKNOWLEDGED"]["field_schemas"][
            "raw"
        ] = {"type": "object"}
    with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
        validate_contract(weakened)


@pytest.mark.parametrize("mutation", ["owner", "sources", "target", "removed", "added", "terminal"])
def test_exact_lifecycle_mutations_fail_closed(mutation: str) -> None:
    weakened = copy.deepcopy(CONTRACT)
    transitions = weakened["order_lifecycle"]["transitions"]
    if mutation == "owner":
        transitions[0]["owner"] = "CALLER"
    elif mutation == "sources":
        transitions[1]["sources"].append("ACKNOWLEDGED")
    elif mutation == "target":
        transitions[1]["target"] = "FILLED"
    elif mutation == "removed":
        transitions.pop()
    elif mutation == "added":
        transitions.append(copy.deepcopy(transitions[0]))
    else:
        weakened["order_lifecycle"]["terminal_states"].remove("REPLACED")
    with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
        validate_contract(weakened)


@pytest.mark.parametrize("mutation", ["removed", "added", "pointer", "fingerprint", "source"])
def test_exact_cross_dependency_manifest_mutations_fail_closed(mutation: str) -> None:
    weakened = copy.deepcopy(CONTRACT)
    dependencies = weakened["cross_contract_dependencies"]
    if mutation == "removed":
        dependencies.pop()
    elif mutation == "added":
        dependencies.append(copy.deepcopy(dependencies[0]))
    elif mutation == "pointer":
        dependencies[0]["json_pointer"] = "/invariants"
    elif mutation == "fingerprint":
        dependencies[0]["content_fingerprint_sha256"] = "0" * 64
    else:
        dependencies[0]["contract"] = "environment_and_product_capabilities.json"
    with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
        validate_contract(weakened)


@pytest.mark.parametrize(
    "mutation", ["removed_type", "added_type", "reordered_type", "removed_field", "added_field"]
)
def test_exact_event_registry_and_envelope_fields_fail_closed(mutation: str) -> None:
    weakened = copy.deepcopy(CONTRACT)
    if mutation == "removed_type":
        weakened["event_contract"]["event_types"].pop()
    elif mutation == "added_type":
        weakened["event_contract"]["event_types"].append("ARBITRARY_EVENT")
    elif mutation == "reordered_type":
        weakened["event_contract"]["event_types"].reverse()
    elif mutation == "removed_field":
        weakened["event_contract"]["fields"].pop()
    else:
        weakened["event_contract"]["fields"].append("arbitrary")
    with pytest.raises(ValueError, match="CONTRACT_INCONSISTENT"):
        validate_contract(weakened)
