"""Independent M0.9 attestation and pure, non-production reference model."""

from __future__ import annotations
import copy, hashlib, json, re, unicodedata
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
import pytest

ROOT = Path(__file__).parents[2]
ARCH = ROOT / "docs/architecture/cryptohunter_product_architecture"
CONTRACT = ARCH / "risk_hierarchy_kill_switch_and_execution_lease.json"


def deep_freeze(v: Any) -> Any:
    if isinstance(v, dict):
        return MappingProxyType({k: deep_freeze(x) for k, x in v.items()})
    if isinstance(v, list):
        return tuple(deep_freeze(x) for x in v)
    return v


def normalize(v: Any) -> Any:
    if is_dataclass(v):
        return normalize(asdict(v))  # type: ignore[arg-type]
    if isinstance(v, Fraction):
        return f"{v.numerator}/{v.denominator}"
    if isinstance(v, datetime):
        return v.isoformat().replace("+00:00", "Z")
    if isinstance(v, (tuple, list)):
        return [normalize(x) for x in v]
    if isinstance(v, dict):
        return {k: normalize(x) for k, x in v.items()}
    return v


def canonical(v: Any) -> bytes:
    return json.dumps(
        normalize(v), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def fingerprint(v: Any) -> str:
    return hashlib.sha256(canonical(v)).hexdigest()


def thaw(v: Any) -> Any:
    if isinstance(v, Mapping):
        return {k: thaw(x) for k, x in v.items()}
    if isinstance(v, tuple):
        return [thaw(x) for x in v]
    return v


# Independently authored full expected contract; CONTRACT is never read to construct it.
EXPECTED_PROTOCOLS = deep_freeze(
    json.loads(r"""
{
  "schema_version": "1.0.0",
  "m0_element": "M0.9",
  "status": "closed",
  "authority": {
    "owner": "CoreHost",
    "attestation": "this JSON is untrusted until independently validated with all dependency fingerprints",
    "execution_equation": [
      "upstream capability and readiness",
      "accepted M0.7 command/order",
      "accepted Core M0.9 policy and ALLOW decision",
      "no applicable ACTIVE kill switch",
      "valid exact ExecutionLease",
      "exact M0.8 reservation",
      "M0.7 idempotency reservation and accepted plan"
    ],
    "non_authorities": [
      "UI",
      "Tray",
      "raw JSON",
      "self-hash",
      "venue capability",
      "credential",
      "risk_ok",
      "risk_passed",
      "approved"
    ]
  },
  "source_of_truth": {
    "machine_contract": "this file after independent validation",
    "reference_model": "tests only; pure and non-production",
    "runtime": "not implemented"
  },
  "scope_hierarchy": {
    "environment_required": true,
    "no_cross_environment_inheritance": true,
    "applicable_order": [
      "PRODUCT_SYSTEM",
      "WORKSPACE",
      "PORTFOLIO",
      "EXCHANGE_ACCOUNT",
      "STRATEGY_INSTANCE",
      "INSTRUMENT",
      "EXECUTION_ROUTE"
    ],
    "composition": "PRODUCT_SYSTEM is the root policy scope. WORKSPACE is its child. PORTFOLIO is a WORKSPACE child and EXCHANGE_ACCOUNT its child. STRATEGY_INSTANCE, INSTRUMENT and EXECUTION_ROUTE are WORKSPACE-owned intersecting qualifiers, applicable only when their exact IDs occur in the evaluated command; they are incomparable peers, not invented parents. Exact ancestry and qualifiers come from accepted upstream projections.",
    "applicability": "same concrete environment and exact ancestor/qualifier identity only",
    "ordering": "fixed applicable_order then scope_id then policy revision; never container iteration",
    "deny": "any applicable DENY dominates",
    "incomparable_conflict": "two applicable policies at the same exact scope identity and revision with unequal semantic fingerprints => POLICY_CONFLICT"
  },
  "risk_policy_contract": {
    "projection": "PrevalidatedRiskPolicyContext",
    "identity": [
      "risk_policy_id",
      "revision",
      "environment",
      "scope_type",
      "scope_id",
      "action",
      "limits",
      "semantic_fingerprint_sha256"
    ],
    "immutability": "identity, revision, limits and fingerprints fixed for one evaluation",
    "raw_policy_authority": false,
    "self_hash_authority": false,
    "upstream_identity": "M0.2 RiskPolicy: risk_policy_id, prefix rpol, UUIDv7",
    "action_registry": [
      "ALLOW",
      "DENY"
    ],
    "coverage": "at least one applicable current accepted policy with explicit ALLOW or DENY is required; zero applicable policies is RISK_CONTEXT_INCOMPLETE",
    "semantic_fingerprint_input": [
      "risk_policy_id",
      "revision",
      "environment",
      "scope_type",
      "scope_id",
      "action",
      "limits"
    ],
    "accepted_context_types": [
      "RiskPolicyRecord",
      "CoreAcceptedContentBinding",
      "PrevalidatedRiskPolicyContext"
    ],
    "enabled_limit_validation": "before composition: canonical rpol UUIDv7, positive non-bool revision, closed environment/scope/action, exact scope ID prefix, supported limit name, Fraction-only threshold and exact AssetReference unit; known unsupported or unknown names return UNSUPPORTED_RISK_SEMANTICS"
  },
  "risk_policy_authority_boundary": {
    "acceptance_owner": "CoreHost",
    "required": "a validated PrevalidatedRiskPolicyContext whose complete record/history content, semantic fingerprints, current designations, context fingerprint and CoreAcceptedContentBinding independently validate; a nominal class name alone is not authority",
    "authentication_deferred_to": "M0.10",
    "failure": "TRUSTED_CONTEXT_FAILURE",
    "nominal_membership": "pre-existing Core membership identifier is an independent reference-model input; hashes prove integrity and cannot self-enroll"
  },
  "effective_policy_resolution": {
    "steps": [
      "validate every accepted context and concrete environment",
      "select exact ancestors and exact command qualifiers",
      "reject duplicate exact scope/revision with different semantics",
      "apply any DENY",
      "for each required supported limit take minimum threshold for maximum limits and maximum threshold for minimum limits",
      "sort results by closed limit registry order and supplying scope order"
    ],
    "maximum": "minimum exact threshold",
    "minimum": "maximum exact threshold",
    "missing_required_input": "INCOMPLETE",
    "unsupported": "UNSUPPORTED_RISK_SEMANTICS",
    "no_implicit_default": true,
    "current_revision": "exactly one designated current revision per (risk_policy_id, environment, scope_type, scope_id); two designations conflict; accepted newer history makes designation of an older revision a rollback conflict; missing, duplicate or semantically conflicting designated record fails POLICY_CONFLICT",
    "deny": "any applicable current DENY produces RISK_DENIED before numeric composition",
    "coverage": "zero applicable current accepted policies is RISK_CONTEXT_INCOMPLETE; optional absence of a limit is authorized only by explicit ALLOW policy semantics",
    "effective_set_fence": "SHA-256 over closed-scope-ordered immutable bindings of every effective policy ID, current revision, action and recomputed semantic fingerprint",
    "ordering": "current applicable policies are ordered only by scope_hierarchy.applicable_order, then scope_id, then revision; input record/current container order is irrelevant"
  },
  "supported_spot_limit_registry": {
    "MAX_ORDER_QUANTITY": {
      "unit": "base AssetReference quantity",
      "formula": "abs(order.quantity)",
      "buy_sell": "same absolute quantity",
      "fees": "excluded; fees are distinct M0.8 assets",
      "boundary": "observed <= threshold passes",
      "price_required": false,
      "input_schema": "exact command economics plus accepted M0.8 per-AssetReference/per-instrument projection and accepted valuation/reservation contexts material to this limit",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    },
    "MAX_ORDER_NOTIONAL": {
      "unit": "quote AssetReference quantity",
      "formula": "abs(quantity) * conservative_risk_price",
      "buy_sell": "same notional",
      "fees": "fees added to required reservation but not order notional",
      "boundary": "observed <= threshold passes",
      "price_required": true,
      "input_schema": "exact command economics plus accepted M0.8 per-AssetReference/per-instrument projection and accepted valuation/reservation contexts material to this limit",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    },
    "MAX_POST_TRADE_POSITION_QUANTITY": {
      "unit": "base AssetReference quantity",
      "formula": "abs(current signed SPOT inventory + signed quantity), BUY positive and SELL negative",
      "buy_sell": "signed projection",
      "fees": "excluded",
      "boundary": "observed <= threshold passes",
      "price_required": false,
      "input_schema": "exact command economics plus accepted M0.8 per-AssetReference/per-instrument projection and accepted valuation/reservation contexts material to this limit",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    },
    "MAX_POST_TRADE_POSITION_NOTIONAL": {
      "unit": "valuation quote AssetReference quantity",
      "formula": "abs(current signed SPOT inventory + signed quantity) * conservative_risk_price",
      "buy_sell": "signed projection",
      "fees": "excluded",
      "boundary": "observed <= threshold passes",
      "price_required": true,
      "input_schema": "exact command economics plus accepted M0.8 per-AssetReference/per-instrument projection and accepted valuation/reservation contexts material to this limit",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    },
    "MAX_GROSS_EXPOSURE": {
      "unit": "valuation quote AssetReference quantity",
      "formula": "current trusted gross exposure - current instrument absolute exposure + projected instrument absolute exposure",
      "buy_sell": "signed position then absolute valuation",
      "fees": "excluded",
      "boundary": "observed <= threshold passes",
      "price_required": true,
      "input_schema": "accepted M0.8 total gross exposure and current/projected instrument exposure with identical reporting AssetReference, valuation basis and as_of_utc",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    },
    "MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION": {
      "unit": "exact single primary-spend AssetReference quantity",
      "formula": "accepted PRE available_capital[ReservationRequirementProjection.asset_reference] - pure derived ReservationRequirementProjection.required_quantity exactly once; ReservationState is POST authority and is never subtracted again",
      "buy_sell": "BUY reservation asset is exact historical Instrument quote AssetReference; SELL reservation asset is exact historical Instrument base AssetReference",
      "fees": "only a fee included in the same single M0.8 primary-spend reservation quantity is representable; a separate third-asset pre-trade reservation is UNSUPPORTED_RISK_SEMANTICS and cannot issue a lease",
      "boundary": "observed >= threshold passes",
      "price_required": "BUY notional requires LIMIT price or trusted MARKET valuation; SELL base reservation does not require price unless another enabled limit does",
      "input_schema": "exact command economics plus accepted M0.8 per-AssetReference/per-instrument projection and accepted valuation/reservation contexts material to this limit",
      "missing_input": "INCOMPLETE; accepted explicit zero is valid, absent fact is never zero",
      "unit_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status"
    }
  },
  "unsupported_limit_registry": [
    "LEVERAGE",
    "MARGIN",
    "DERIVATIVES_GREEKS",
    "LIQUIDATION",
    "BORROW",
    "FUNDING",
    "OPTION_EXERCISE",
    "TAX",
    "CROSS_MARGIN",
    "DAILY_LOSS_RESET",
    "NAV_CONCENTRATION"
  ],
  "exact_arithmetic_policy": {
    "representation": "canonical decimal strings converted to exact rational/Fraction",
    "float_forbidden": true,
    "rounding": "none during evaluation; upstream instrument quantization must already be valid",
    "canonical_hash": "SHA-256 of UTF-8 canonical JSON: sorted object keys, compact separators, arrays preserved"
  },
  "risk_input_contract": {
    "projection": "AccountingRiskProjection",
    "required_binding": [
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_ids",
      "as_of_utc",
      "reporting_asset_reference",
      "owned_balances",
      "available_capital",
      "reserved_capital",
      "inventory_exposure",
      "valuations",
      "reconciliation_outcomes",
      "projection_fingerprint_sha256",
      "accepted_membership_id"
    ],
    "available_outputs": [
      "owned_balances",
      "available_capital",
      "reserved_capital",
      "spot_inventory_exposure",
      "equity_nav",
      "gross_realized_pnl",
      "net_realized_pnl",
      "unrealized_pnl",
      "valuation_completeness",
      "reconciliation_status"
    ],
    "authority": "accepted immutable M0.8 projection; caller balances and mutable cache forbidden",
    "materiality": "only values and valuation paths required by enabled limits; irrelevant zero holdings do not gate",
    "accepted_context_types": [
      "AccountingRiskProjection",
      "CoreAcceptedContentBinding"
    ],
    "fingerprint_validation": "recompute projection fingerprint over all immutable economic content, then validate Core accepted-membership seal",
    "asset_identity": "exact M0.5 AssetReference object: venue_asset_code, canonical_display_code, asset_namespace, mapping_status",
    "gross_exposure": "derived by M0.9 from every accepted SPOT inventory/exposure fact and accepted valuation contexts sharing exact reporting AssetReference and as_of_utc; no caller gross scalar authority",
    "nominal_membership": "pre-existing Core accepted projection membership is required independently of content hashes"
  },
  "post_state_projection_contract": {
    "purity": "no LedgerEntry, reservation or projection mutation",
    "formula": "accepted M0.8 state plus conservative exact order economic effect plus exact fees/reservation",
    "limit_order_price": "canonical limit_price is conservative risk price for current LIMIT path",
    "market_order_price": "requires an explicit accepted M0.6/M0.8 trusted valuation context bound to instrument, environment, as_of, expiry and fingerprint; absent context is INCOMPLETE",
    "forbidden_prices": [
      "UI price",
      "random ticker",
      "adapter response",
      "implicit last trade",
      "zero",
      "stablecoin parity"
    ],
    "reservation_economics": "one accepted M0.8 capital_reservation fact/state only: BUY exact quote AssetReference quantity; SELL exact base AssetReference quantity; same-asset fee may be included in that quantity; separate third-asset reservation is unsupported/incomplete",
    "market_economics_gate": "MARKET always requires trusted valuation sufficient to establish dispatch reservation economics, even when enabled numeric limits only mention quantity",
    "zero_price_fallback": false
  },
  "valuation_reconciliation_gates": {
    "fail_closed": [
      "DRIFT",
      "MISSING_INTERNAL_FACT",
      "MISSING_EXTERNAL_FACT",
      "UNMAPPED_ASSET",
      "UNSUPPORTED",
      "MISSING_VALUATION",
      "STALE_VALUATION",
      "UNSUPPORTED_VALUATION_PATH"
    ],
    "result": "RISK_CONTEXT_INCOMPLETE; no lease",
    "material_inputs_only": true,
    "valuation_context_fields": [
      "subject_reference",
      "valuation_unit",
      "rate",
      "source_id",
      "observed_at_utc",
      "effective_at_utc",
      "as_of_utc",
      "stale_after_utc",
      "source_fingerprint_sha256",
      "accepted_authority_fingerprint_sha256",
      "context_fingerprint_sha256"
    ],
    "freshness": "derived as evaluated_at_utc <= stale_at_utc and matching exact scope; never a caller boolean"
  },
  "kill_switch_contract": {
    "states": [
      "INACTIVE",
      "ACTIVE"
    ],
    "record_fields": [
      "scope_type",
      "scope_id",
      "environment",
      "state",
      "source_revision",
      "effective_at_utc",
      "generation",
      "accepted_authority_fingerprint_sha256",
      "record_fingerprint_sha256"
    ],
    "authority": "accepted Core context; raw record is not authority",
    "transition": "every accepted state transition strictly increments the environment/scope authority generation; generation never decreases or reuses",
    "issuance_effect": "any applicable ACTIVE blocks issuance",
    "existing_lease_effect": "dispatch re-resolves current switches; ACTIVE or generation/fingerprint mismatch invalidates old lease before side effect",
    "accepted_context_types": [
      "KillSwitchRecord",
      "CoreAcceptedContentBinding",
      "PrevalidatedKillSwitchContext"
    ],
    "generation_validation": "generation is positive and strictly increases per exact scope/environment history; duplicate/reuse/rollback or unknown state fails TRUSTED_CONTEXT_FAILURE"
  },
  "kill_switch_hierarchy": {
    "applicability": "same hierarchy/qualifiers and exact environment as policy",
    "monotonic": "any applicable ancestor or qualifier ACTIVE dominates all descendant INACTIVE states",
    "unrelated": "different environment or non-applicable identity has no effect"
  },
  "generation_fencing_policy": {
    "fence": "deterministic fingerprint over ordered applicable kill-switch records and their monotonic generations",
    "lease_binding": "exact current kill-switch fence plus policy revision/fingerprint and accounting projection fingerprint",
    "stale_on": [
      "any accepted policy revision change",
      "any applicable kill-switch transition including ACTIVE then INACTIVE",
      "authority-relevant accounting projection change"
    ],
    "no_resurrection": true,
    "fence_order": "explicit scope_hierarchy.applicable_order then scope_id; never lexical or container order"
  },
  "risk_decision_contract": {
    "projection": "immutable non-durable RiskDecision",
    "decisions": [
      "ALLOW",
      "DENY",
      "INCOMPLETE"
    ],
    "required_fields": [
      "command_id",
      "command_request_fingerprint_sha256",
      "order_id",
      "scope_binding",
      "environment",
      "effective_policy_fingerprint_sha256",
      "pre_reservation_accounting_projection_fingerprint_sha256",
      "reservation_requirement_fingerprint_sha256",
      "evaluated_at_utc",
      "ordered_limit_results",
      "kill_switch_result",
      "kill_switch_fence_sha256",
      "decision",
      "decision_fingerprint_sha256"
    ],
    "limit_result_fields": [
      "limit_type",
      "effective_threshold",
      "observed_projected_value",
      "unit_asset_reference",
      "supplying_policy_scope",
      "result",
      "reason_code"
    ],
    "aggregation": "any FAIL => DENY; else any INCOMPLETE => INCOMPLETE; else ALLOW",
    "lease_rule": "only accepted ALLOW can contribute to issuance; it is never sufficient alone",
    "accepted_context": "PrevalidatedRiskDecisionContext binds an independently recomputed immutable ALLOW RiskDecision fingerprint to the exact evaluation inputs",
    "decision_fingerprint": "SHA-256 over every semantic RiskDecision field except decision_fingerprint_sha256; recomputed at issuance and when retrieving accepted historical decision at dispatch",
    "rederivation": "issuance independently rederives the full PRE decision. Dispatch retrieves the exact accepted historical RiskDecision, recomputes all its semantic fields/fingerprint, requires ALLOW and matching lease binding, then revalidates current execution authority, policy/switch fences, POST accounting, reservation and lifetime; it does not reinterpret the historical PRE decision."
  },
  "execution_lease_contract": {
    "entity": "ExecutionLease",
    "identity": {
      "field": "execution_lease_id",
      "prefix": "lease",
      "format": "M0.2 durable UUIDv7 identity",
      "upstream_pointer": "/entity_kinds canonical_name=ExecutionLease",
      "validation": "stable extraction of M0.2 entity_kinds entry canonical_name=ExecutionLease requires execution_lease_id/prefix lease/persistence true, then UUIDv7 regex"
    },
    "capability": "immutable, exact-bound, one-shot additional authorization; never generic",
    "required_fields": [
      "execution_lease_id",
      "command_id",
      "command_request_fingerprint_sha256",
      "order_id",
      "order_intent_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "instrument_metadata_version",
      "execution_route_id",
      "strategy_instance_id",
      "source_identity",
      "side",
      "order_type",
      "quantity",
      "limit_price",
      "time_in_force",
      "order_expire_at_utc",
      "effective_policy_bindings",
      "effective_policy_fingerprint_sha256",
      "kill_switch_bindings",
      "kill_switch_fence_sha256",
      "pre_reservation_accounting_projection_fingerprint_sha256",
      "post_reservation_accounting_projection_fingerprint_sha256",
      "risk_decision_fingerprint_sha256",
      "reservation_source_audit_event_id",
      "reservation_source_fingerprint_sha256",
      "reservation_state_fingerprint_sha256",
      "reservation_asset_reference",
      "reservation_original_quantity",
      "reservation_remaining_quantity",
      "issued_at_utc",
      "expires_at_utc",
      "lease_fingerprint_sha256"
    ],
    "fingerprint": "SHA-256 over every semantic lease field except lease_fingerprint_sha256; recomputed before dispatch",
    "partial_match_forbidden": true,
    "distinct_identity": "each issuance receives a distinct canonical execution_lease_id"
  },
  "lease_issuance_conditions": [
    "a validated PrevalidatedExecutionAuthorityContext whose complete M0.4-M0.7 content, fingerprints, accepted memberships, exact bindings and current mutable designations independently validate",
    "exact AccountingRiskProjection fingerprint and CoreAcceptedContentBinding validate",
    "effective accepted policy set has coverage, no DENY/conflict and an exact composite fence",
    "accepted kill-switch history validates and no applicable ACTIVE exists",
    "PrevalidatedRiskDecisionContext validates an immutable exact ALLOW decision and recomputed fingerprint",
    "exact accepted M0.8 capital_reservation AuditEvent-derived accounting context binds command/order/scope/account and sufficient per-AssetReference requirements",
    "issued_at_utc < expires_at_utc and duration <= 30 seconds"
  ],
  "lease_validation_conditions": {
    "validator": "validate_execution_lease_for_dispatch",
    "all_exact": [
      "canonical execution_lease_id and recomputed lease fingerprint",
      "all command, request, Order, OrderIntent, scope, economics and expiry fields",
      "current composite effective-policy bindings/fingerprint",
      "current composite kill-switch bindings/fence and no ACTIVE state",
      "current accepted accounting projection and RiskDecision fingerprints",
      "accepted M0.8 reservation source AuditEvent/fingerprint/accounting-state fingerprint and every per-AssetReference remaining requirement",
      "exact PrevalidatedExecutionAuthorityContext for M0.4/M0.5/M0.6/M0.7",
      "Core-owned M0.7 dispatch/idempotency state"
    ],
    "time_rule": "issued_at_utc <= now_utc <= expires_at_utc",
    "fail_closed_for": [
      "wrong command or fingerprint",
      "wrong order or intent",
      "wrong workspace or portfolio",
      "wrong environment",
      "wrong account or exchange",
      "wrong instrument or metadata",
      "wrong route or strategy/source",
      "expired",
      "stale policy, switch or accounting fence",
      "ACTIVE kill switch",
      "missing, foreign or insufficient reservation",
      "upstream denial",
      "consumed/replayed authority"
    ]
  },
  "lease_lifetime": {
    "maximum_seconds": 30,
    "timestamps": "canonical UTC with Z; expires_at_utc > issued_at_utc",
    "validity": "inclusive at expires_at_utc",
    "clock": "trusted reference-model input; production clock/security deferred"
  },
  "one_shot_idempotency_relation_to_m07": {
    "authority": "M0.7 command_id/fingerprint idempotency remains sole dispatch dedupe authority",
    "same_retry": "byte-equivalent recorded replay; zero new dispatch",
    "changed_request": "IDEMPOTENCY_CONFLICT",
    "consumption": "successful dispatch marks exact lease/order command authority consumed in the same dispatch plan; it cannot authorize another submission",
    "unknown": "reconcile; never resubmit",
    "core_state": "CoreDispatchAuthorityState maps exact command_id to request fingerprint, order_id, execution_lease_id, lease fingerprint and state UNUSED/CONSUMED/UNKNOWN_RECONCILIATION; replay is returned only after exact request/order/economics match, otherwise IDEMPOTENCY_CONFLICT",
    "atomic_authorization": "successful final validation atomically changes UNUSED to CONSUMED before returning DISPATCH_AUTHORIZED",
    "states": [
      "UNUSED",
      "CONSUMED",
      "UNKNOWN_RECONCILIATION"
    ],
    "replay_validation": "canonical lease ID, recomputed lease fingerprint and every immutable command/order/economic/scope binding are validated before consumed replay lookup; historical replay does not re-authorize against current policy",
    "unknown_reconciliation": "never dispatch or resubmit; return RECONCILIATION_REQUIRED; side-effect count unchanged"
  },
  "reservation_relation_to_m08": {
    "ordering": [
      "pure risk evaluation",
      "accepted ALLOW decision",
      "provisional lease materialization",
      "successful exact M0.8 order-bound reservation",
      "final lease fingerprint/dispatch validation",
      "M0.7 atomic idempotency reservation and accepted plan",
      "one external side effect"
    ],
    "dispatchable": "only after exact reservation is present and sufficient",
    "issuance_mutation": false,
    "risk_is_not_reservation": true,
    "identity": "exactly one M0.8 capital_reservation AuditEvent-derived fact and one ReservationState for one Order/command, one exact AssetReference and one quantity; no aggregate requirements and no durable reservation_id",
    "exact_binding": [
      "audit_event_id",
      "source_fingerprint_sha256",
      "command_id",
      "order_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "asset_reference",
      "original_quantity",
      "remaining_quantity",
      "state_fingerprint_sha256",
      "accepted_membership_id"
    ],
    "provisional_object": "a non-authoritative LeaseDraft may exist after ALLOW; only the final immutable ExecutionLease created after accepted exact reservation is dispatchable",
    "fee_boundary": "third-asset pre-trade fee reservation is not representable by current M0.8 single-asset reservation and therefore fails UNSUPPORTED_RISK_SEMANTICS; it is not required for closure of primary-spend current-SPOT limits",
    "phases": [
      "accepted PRE-reservation accounting projection",
      "pure non-authoritative ReservationRequirementProjection",
      "pure RiskDecision using PRE available minus requirement exactly once",
      "accepted ALLOW",
      "exact accepted M0.8 capital_reservation fact",
      "derived accepted ReservationState",
      "accepted POST-reservation projection proving one available-to-reserved transition",
      "final ExecutionLease binds distinct PRE decision and current POST dispatch fingerprints"
    ],
    "risk_formula": "MIN_AVAILABLE uses PRE available minus derived requirement; it never subtracts accepted ReservationState from POST available",
    "transition_proof": "POST available == PRE available - fact.quantity and POST reserved == PRE reserved + fact.quantity for exactly the same scope/account/environment/AssetReference; fact.quantity == requirement.required_quantity; ReservationState binds fact and remaining quantity",
    "exact_fact_fields": [
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
      "command_id"
    ],
    "reservation_state_fields": [
      "command_id",
      "order_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "asset_reference",
      "original_quantity",
      "remaining_quantity",
      "source_audit_event_id",
      "source_fingerprint_sha256",
      "state_fingerprint_sha256",
      "accepted_membership_id"
    ]
  },
  "environment_isolation": {
    "environments": [
      "PAPER",
      "TESTNET",
      "LIVE"
    ],
    "exact": "policy, switch, decision, accounting projection, reservation and lease bind one equal environment",
    "substitution": false,
    "fallback": false
  },
  "live_target_policy": {
    "current": "constructed only by current-edition upstream fixture carrying exact M0.4 LIVE_BLOCKED_BY_EDITION and M0.6 POLICY_BLOCKED; no boolean override",
    "future": "separate pre-existing hypothetical future ProductCapabilities, M0.5 instrument/account, READY M0.6 route and accepted M0.7 command projections with independent nominal memberships; no magic result value alone grants authority",
    "fixture": "tests may provide nominal prevalidated future capability only; real upstream unchanged"
  },
  "m010_boundary": {
    "deferred": [
      "operator identity proof",
      "PIN",
      "biometrics",
      "device trust",
      "secrets",
      "credential storage",
      "authorization mechanism for policy/switch changes"
    ],
    "m09_only": "names nominal prevalidated authority contexts; no fake admin/user ID"
  },
  "m011_boundary": {
    "must_persist_before_production": [
      "accepted risk policy revisions and membership",
      "kill-switch state and monotonic generations",
      "issued/consumed lease state required for one-shot correctness",
      "audit references"
    ],
    "reference_model": "in-memory only",
    "production_crash_safety_claimed": false
  },
  "failure_registry": [
    "RISK_DENIED",
    "RISK_CONTEXT_INCOMPLETE",
    "KILL_SWITCH_ACTIVE",
    "LEASE_NOT_ISSUED",
    "LEASE_EXPIRED",
    "LEASE_STALE",
    "LEASE_SCOPE_MISMATCH",
    "LEASE_ALREADY_CONSUMED",
    "POLICY_CONFLICT",
    "TRUSTED_CONTEXT_FAILURE",
    "UNSUPPORTED_RISK_SEMANTICS",
    "RESERVATION_INVALID",
    "UPSTREAM_AUTHORITY_DENIED",
    "IDEMPOTENCY_CONFLICT",
    "REPLAY_SUCCESS",
    "CONTRACT_INCONSISTENT"
  ],
  "cross_contract_dependencies": [
    {
      "contract": "canonical_domain_vocabulary.json",
      "json_pointer": "/public_trading_environments",
      "content_fingerprint_sha256": "b0114e386bf72439199ec8155d65a57dc7e00cabe79dba805f53221ba7713103"
    },
    {
      "contract": "canonical_domain_vocabulary.json",
      "json_pointer": "/entity_kinds",
      "content_fingerprint_sha256": "1913a18c7e7479d9c20850a690ee81b9f311a7d08cf2458374c91f6332d277f1"
    },
    {
      "contract": "canonical_domain_vocabulary.json",
      "json_pointer": "/relationships",
      "content_fingerprint_sha256": "03c64a8159cd7ae7f1103b029234c858d0cf86898d9dc0b652e1043667fc11a9"
    },
    {
      "contract": "canonical_domain_vocabulary.json",
      "json_pointer": "/identifier_policy",
      "content_fingerprint_sha256": "44726b4e51c53722ebbc95212d708521007c59cb54292ea7bc5b970320031d20"
    },
    {
      "contract": "process_topology_and_lifecycle.json",
      "json_pointer": "/authority_boundaries",
      "content_fingerprint_sha256": "6a491a4910109f2859f39c871a1111baf9f8cd052919ec419174ad557e136a0d"
    },
    {
      "contract": "environment_and_product_capabilities.json",
      "json_pointer": "/execution_environments",
      "content_fingerprint_sha256": "61f5dd195aa69c646dc296c2d8c0f2ce97683979e3ef6ddc35b0adaa269a809b"
    },
    {
      "contract": "environment_and_product_capabilities.json",
      "json_pointer": "/ProductCapabilities",
      "content_fingerprint_sha256": "fd8cd8b62827b96d52bd6870985f7e2438a0306cd267048b7745472157877605"
    },
    {
      "contract": "environment_and_product_capabilities.json",
      "json_pointer": "/capability_trust_policy",
      "content_fingerprint_sha256": "a8a66584a0fc48b8a0170a86fb03c92f65e592dbdcb317fd3f47d1fe5e8e8a35"
    },
    {
      "contract": "exchange_accounts_and_instruments.json",
      "json_pointer": "/exchange_account_contract",
      "content_fingerprint_sha256": "f1f7b18513a53d107b6effe169e793e74f5c70c8df6261a25ee96a3012c51ed4"
    },
    {
      "contract": "exchange_accounts_and_instruments.json",
      "json_pointer": "/instrument_contract",
      "content_fingerprint_sha256": "a3b72fe19c84c63ab7702a00f4ad7e8b7f2b4cb6f093daffce4d4433f883f98c"
    },
    {
      "contract": "exchange_accounts_and_instruments.json",
      "json_pointer": "/asset_reference_contract",
      "content_fingerprint_sha256": "837e1452a60de230d0ca091a7e2c05308ff41d961499f7800d350d7c9b3682ae"
    },
    {
      "contract": "strategy_market_data_and_execution_routing.json",
      "json_pointer": "/execution_route_contract",
      "content_fingerprint_sha256": "a855a114e6b19b8218f9975d64fb04b9dac86e28a208e9e7d3a4976773de0a12"
    },
    {
      "contract": "strategy_market_data_and_execution_routing.json",
      "json_pointer": "/route_readiness_contract",
      "content_fingerprint_sha256": "92dda02de1e327d248bc9e987701b33279534f48b57ac2a97104c8380d7c4c33"
    },
    {
      "contract": "strategy_market_data_and_execution_routing.json",
      "json_pointer": "/live_execution_authority_policy",
      "content_fingerprint_sha256": "eb2034a4691a18f3037b2a5f93d1fb41fcb38ef67c6c43c5dfff891dd35fe1de"
    },
    {
      "contract": "commands_events_order_lifecycle_and_idempotency.json",
      "json_pointer": "/command_registry/SUBMIT_ORDER",
      "content_fingerprint_sha256": "864bc27bf55228d08b6592f2042a3f9a1f447eae661382bde0b380602369d748"
    },
    {
      "contract": "commands_events_order_lifecycle_and_idempotency.json",
      "json_pointer": "/closed_request_policy",
      "content_fingerprint_sha256": "f3a523b01cbce2bafabeaad261777e3a97fc4f60db751deee0e77cb912f93e7c"
    },
    {
      "contract": "commands_events_order_lifecycle_and_idempotency.json",
      "json_pointer": "/idempotency_contract",
      "content_fingerprint_sha256": "43ba37d976eb5948970d289cc71cd06da82cb90803f9e5f61cc84109f00e9a62"
    },
    {
      "contract": "commands_events_order_lifecycle_and_idempotency.json",
      "json_pointer": "/environment_execution_boundary",
      "content_fingerprint_sha256": "423939e49b651d91a8a4b70d0ba0d64275886c5c3b23544ef2e822956eb153db"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/accounting_economic_fact_schema_registry/capital_reservation",
      "content_fingerprint_sha256": "534ac4bec8ab864378e7b5900afc7e1634e5cbbeed443571ccb160e8c9ea1932"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/reservation_protocol",
      "content_fingerprint_sha256": "b3579e387a02b9f291468f5cc89a9ed650e4024a538cbb7f6f0c0818ed0c4641"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/balance_model",
      "content_fingerprint_sha256": "082ab812c1cfb4cd4e0f21de698129b59a8a034910fe1149461d12b9fd7f2fff"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/rebuild_protocol",
      "content_fingerprint_sha256": "e6bade298e6debc1c2972824f3b8fe2a177e15690e24868cde49e052648314a3"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/environment_policy",
      "content_fingerprint_sha256": "4499550489e137353a08ecb4f06b43c0e62d5bcf588cd9cb07ea8dd3a28f02ba"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/m09_outputs",
      "content_fingerprint_sha256": "42a97f0a077226763cc77a75c7eab0ef2879031f8fd87010268d2ab35d8403f8"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/valuation_protocol",
      "content_fingerprint_sha256": "5328a34d1ad389dd19b84027a48cfb0476d2eba4614b684ffe8bd55e03469d4a"
    },
    {
      "contract": "ledger_portfolio_capital_and_pnl.json",
      "json_pointer": "/reconciliation_protocol",
      "content_fingerprint_sha256": "747ad52f8c2d2f6983d80e5bf9219802bfb382fb02aebda697a946e6ab911ce1"
    }
  ],
  "forbidden_scope": [
    "runtime implementation",
    "exchange adapter or side effect",
    "real LIVE execution",
    "M0.10 identity/secrets",
    "M0.11 persistence",
    "caller boolean authority",
    "cross-environment fallback",
    "float arithmetic"
  ],
  "contract_inconsistent_scope": "only machine-contract semantic drift, schema corruption, missing dependency root or dependency fingerprint mismatch; never ordinary risk denial",
  "closure_conditions": [
    "deterministic exact hierarchy and effective policy",
    "closed executable SPOT limits and fail-closed unsupported registry",
    "monotonic hierarchical kill switch fences old leases without resurrection",
    "policy/accounting revisions fence old leases",
    "exact short-lived durable M0.2 ExecutionLease identity and one-shot binding",
    "M0.7 idempotency and M0.8 reservation ordering preserved",
    "exact environment isolation and future-capable upstream-gated LIVE",
    "no boolean, UI, fake security or fake persistence authority",
    "every directly interpreted dependency root fingerprint-attested",
    "no caller acceptance/freshness/upstream/consumed boolean or scalar reservation shadow-authority",
    "PRE risk and POST dispatch accounting fingerprints are distinct and exact M0.8 transition-attested",
    "every authority-bearing executable dataclass and registry binding schema equals its machine declaration"
  ],
  "executable_boundary_schemas": {
    "CoreAcceptedContentBinding": [
      "membership_id",
      "content_fingerprint_sha256"
    ],
    "CoreCurrentProductDesignation": [
      "environment",
      "membership_id"
    ],
    "CoreCurrentRouteDesignation": [
      "environment",
      "execution_route_id",
      "membership_id"
    ],
    "AssetReference": [
      "venue_asset_code",
      "canonical_display_code",
      "asset_namespace",
      "mapping_status"
    ],
    "M07SubmitOrderRequest": [
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
      "expire_at_utc"
    ],
    "CoreAcceptedCommandProjection": [
      "entries",
      "membership_id"
    ],
    "M07PrevalidatedAcceptedCommandContext": [
      "request",
      "request_fingerprint_sha256",
      "membership_id",
      "context_fingerprint_sha256"
    ],
    "ProductCapabilitiesProjection": [
      "environment",
      "authorized_operations",
      "edition_policy",
      "semantic_fingerprint_sha256",
      "membership_id"
    ],
    "InstrumentProjection": [
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "metadata_version",
      "base_asset_reference",
      "quote_asset_reference",
      "semantic_fingerprint_sha256",
      "membership_id"
    ],
    "RouteProjection": [
      "environment",
      "execution_route_id",
      "strategy_instance_id",
      "source_type",
      "readiness",
      "semantic_fingerprint_sha256",
      "membership_id"
    ],
    "PrevalidatedExecutionAuthorityContext": [
      "product",
      "instrument",
      "route",
      "command",
      "context_fingerprint_sha256"
    ],
    "RiskPolicyRecord": [
      "risk_policy_id",
      "revision",
      "environment",
      "scope_type",
      "scope_id",
      "action",
      "limits",
      "semantic_fingerprint_sha256"
    ],
    "PrevalidatedRiskPolicyContext": [
      "records",
      "current",
      "membership_id",
      "context_fingerprint_sha256"
    ],
    "KillSwitchRecord": [
      "scope_type",
      "scope_id",
      "environment",
      "state",
      "source_revision",
      "effective_at_utc",
      "generation",
      "accepted_authority_fingerprint_sha256",
      "record_fingerprint_sha256"
    ],
    "PrevalidatedKillSwitchContext": [
      "history",
      "membership_id",
      "context_fingerprint_sha256"
    ],
    "ValuationContext": [
      "subject_reference",
      "valuation_unit",
      "rate",
      "source_id",
      "observed_at_utc",
      "effective_at_utc",
      "as_of_utc",
      "stale_after_utc",
      "source_fingerprint_sha256",
      "accepted_authority_fingerprint_sha256",
      "context_fingerprint_sha256"
    ],
    "InventoryExposure": [
      "instrument_id",
      "environment",
      "asset_reference",
      "quantity",
      "as_of_utc"
    ],
    "AccountingRiskProjection": [
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_ids",
      "as_of_utc",
      "reporting_asset_reference",
      "owned_balances",
      "available_capital",
      "reserved_capital",
      "inventory_exposure",
      "valuations",
      "reconciliation_outcomes",
      "projection_fingerprint_sha256",
      "accepted_membership_id"
    ],
    "ReservationRequirementProjection": [
      "command_id",
      "order_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "asset_reference",
      "required_quantity",
      "derivation_fingerprint_sha256"
    ],
    "CapitalReservationFact": [
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
      "command_id"
    ],
    "ReservationState": [
      "command_id",
      "order_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "asset_reference",
      "original_quantity",
      "remaining_quantity",
      "source_audit_event_id",
      "source_fingerprint_sha256",
      "state_fingerprint_sha256",
      "accepted_membership_id"
    ],
    "RiskDecision": [
      "command_id",
      "command_request_fingerprint_sha256",
      "order_id",
      "scope_binding",
      "environment",
      "effective_policy_fingerprint_sha256",
      "pre_reservation_accounting_projection_fingerprint_sha256",
      "reservation_requirement_fingerprint_sha256",
      "evaluated_at_utc",
      "ordered_limit_results",
      "kill_switch_result",
      "kill_switch_fence_sha256",
      "decision",
      "decision_fingerprint_sha256"
    ],
    "PrevalidatedRiskDecisionContext": [
      "decision",
      "membership_id"
    ],
    "CoreAcceptedDecisionBinding": [
      "decision_fingerprint_sha256",
      "decision"
    ],
    "ExecutionLease": [
      "execution_lease_id",
      "command_id",
      "command_request_fingerprint_sha256",
      "order_id",
      "order_intent_id",
      "workspace_id",
      "portfolio_id",
      "environment",
      "exchange_account_id",
      "exchange_id",
      "instrument_id",
      "instrument_metadata_version",
      "execution_route_id",
      "strategy_instance_id",
      "source_identity",
      "side",
      "order_type",
      "quantity",
      "limit_price",
      "time_in_force",
      "order_expire_at_utc",
      "effective_policy_bindings",
      "effective_policy_fingerprint_sha256",
      "kill_switch_bindings",
      "kill_switch_fence_sha256",
      "pre_reservation_accounting_projection_fingerprint_sha256",
      "post_reservation_accounting_projection_fingerprint_sha256",
      "risk_decision_fingerprint_sha256",
      "reservation_source_audit_event_id",
      "reservation_source_fingerprint_sha256",
      "reservation_state_fingerprint_sha256",
      "reservation_asset_reference",
      "reservation_original_quantity",
      "reservation_remaining_quantity",
      "issued_at_utc",
      "expires_at_utc",
      "lease_fingerprint_sha256"
    ],
    "CoreDispatchRecord": [
      "command_request_fingerprint_sha256",
      "order_id",
      "execution_lease_id",
      "lease_fingerprint_sha256",
      "state"
    ],
    "CoreIssuedExecutionLeaseProjection": [
      "execution_lease_id",
      "lease_fingerprint_sha256",
      "command_id",
      "order_id"
    ],
    "CoreDispatchAuthorityState": [
      "records",
      "side_effect_count"
    ],
    "EffectivePolicy": [
      "bindings",
      "limits",
      "fingerprint_sha256"
    ],
    "LimitResult": [
      "limit_type",
      "effective_threshold",
      "observed_projected_value",
      "unit_asset_reference",
      "supplying_policy_scope",
      "result",
      "reason_code"
    ]
  },
  "core_authority_registries": {
    "accepted_content": {
      "model": "CoreAcceptedContentBinding",
      "key": "membership_id",
      "value": "content_fingerprint_sha256",
      "semantics": "immutable exact-content acceptance; integrity hash alone is not acceptance"
    },
    "current_product": {
      "model": "CoreCurrentProductDesignation",
      "key": "environment",
      "value": "currently designated accepted membership_id"
    },
    "current_route": {
      "model": "CoreCurrentRouteDesignation",
      "key": [
        "environment",
        "execution_route_id"
      ],
      "value": "currently designated accepted membership_id"
    },
    "accepted_decisions": {
      "model": "CoreAcceptedDecisionBinding",
      "key": "decision_fingerprint_sha256",
      "value": "exact immutable RiskDecision"
    },
    "issued_leases": {
      "model": "CoreIssuedExecutionLeaseProjection",
      "key": "execution_lease_id",
      "value": "exact issued lease identity/fingerprint/command/order binding"
    },
    "dispatch_state": {
      "model": "CoreDispatchAuthorityState",
      "key": "command_id",
      "value": "CoreDispatchRecord and one-shot side_effect_count"
    }
  }
}
""")
)
EXPECTED_CROSS_CONTRACT_DEPENDENCIES = deep_freeze(
    thaw(EXPECTED_PROTOCOLS["cross_contract_dependencies"])
)


def pointer(doc: Any, p: str) -> Any:
    v = doc
    for t in p.split("/")[1:]:
        v = v[t.replace("~1", "/").replace("~0", "~")]
    return v


def validate_contract(doc: Mapping[str, Any]) -> str:
    try:
        expected = thaw(EXPECTED_PROTOCOLS)
        assert set(doc) == set(expected)
        for k in expected:
            assert doc[k] == expected[k]
        assert doc["cross_contract_dependencies"] == thaw(EXPECTED_CROSS_CONTRACT_DEPENDENCIES)
        for dep in thaw(EXPECTED_CROSS_CONTRACT_DEPENDENCIES):
            up = json.loads((ARCH / dep["contract"]).read_text())
            assert (
                fingerprint(pointer(up, dep["json_pointer"])) == dep["content_fingerprint_sha256"]
            )
    except (AssertionError, KeyError, TypeError, OSError, json.JSONDecodeError):
        return "CONTRACT_INCONSISTENT"
    return "VALID"


ENVIRONMENTS = ("PAPER", "TESTNET", "LIVE")
SCOPE_ORDER = (
    "PRODUCT_SYSTEM",
    "WORKSPACE",
    "PORTFOLIO",
    "EXCHANGE_ACCOUNT",
    "STRATEGY_INSTANCE",
    "INSTRUMENT",
    "EXECUTION_ROUTE",
)
NOW = datetime(2026, 1, 1, tzinfo=UTC)
ID_RE = re.compile(
    r"^[a-z][a-z0-9]*_[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)
PREEXISTING_MEMBERSHIPS = frozenset(
    {
        "m04-current",
        "m04-future-live",
        "m05-current",
        "m05-future-live",
        "m06-current",
        "m06-future-live",
        "m07-accepted",
        "policy-accepted",
        "account-pre",
        "account-post",
        "valuation-accepted",
        "switch-accepted",
        "reservation-fact-accepted",
        "reservation-state-accepted",
        "decision-accepted",
    }
)


@dataclass(frozen=True)
class CoreAcceptedContentBinding:
    membership_id: str
    content_fingerprint_sha256: str


@dataclass(frozen=True)
class CoreCurrentProductDesignation:
    environment: str
    membership_id: str


@dataclass(frozen=True)
class CoreCurrentRouteDesignation:
    environment: str
    execution_route_id: str
    membership_id: str


CORE_ACCEPTED_CONTENT: dict[str, CoreAcceptedContentBinding] = {}
CORE_ISSUED_LEASES: dict[str, Any] = {}
CORE_ACCEPTED_DECISIONS: dict[str, Any] = {}
CORE_CURRENT_PRODUCT: dict[str, CoreCurrentProductDesignation] = {}
CORE_CURRENT_ROUTE: dict[tuple[str, str], CoreCurrentRouteDesignation] = {}


def bind_core_content(membership_id: str, content_fingerprint: str) -> str:
    binding = CoreAcceptedContentBinding(membership_id, content_fingerprint)
    existing = CORE_ACCEPTED_CONTENT.setdefault(membership_id, binding)
    if existing != binding:
        raise ValueError("membership identity already binds different content")
    return membership_id


def core_accepts(membership_id: str, content_fingerprint: str) -> bool:
    return CORE_ACCEPTED_CONTENT.get(membership_id) == CoreAcceptedContentBinding(
        membership_id, content_fingerprint
    )


def member(value: str) -> bool:
    return value in PREEXISTING_MEMBERSHIPS


def valid_id(v: str, prefix: str) -> bool:
    return isinstance(v, str) and v.startswith(prefix + "_") and bool(ID_RE.fullmatch(v))


def utc_text(v: str) -> bool:
    if not isinstance(v, str):
        return False
    try:
        return v.endswith("Z") and datetime.fromisoformat(v[:-1] + "+00:00").tzinfo == UTC
    except ValueError:
        return False


def exact_fraction(text: str) -> Fraction | None:
    if not isinstance(text, str) or not re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", text):
        return None
    return Fraction(text)


@dataclass(frozen=True, order=True)
class AssetReference:
    venue_asset_code: str
    canonical_display_code: str
    asset_namespace: str
    mapping_status: str


def valid_asset(a: AssetReference) -> bool:
    return (
        isinstance(a, AssetReference)
        and all(
            isinstance(x, str) and x and unicodedata.normalize("NFC", x) == x
            for x in (a.venue_asset_code, a.canonical_display_code, a.asset_namespace)
        )
        and a.mapping_status in ("EXACT", "EXPLICIT_ALIAS")
    )


USD = AssetReference("USD", "USD", "BINANCE:SPOT", "EXACT")
BTC = AssetReference("BTC", "BTC", "BINANCE:SPOT", "EXACT")


@dataclass(frozen=True)
class M07SubmitOrderRequest:
    command_id: str
    operation_type: str
    authority_context_id: str
    environment: str
    workspace_id: str
    portfolio_id: str
    exchange_account_id: str
    strategy_instance_id: str | None
    source_type: str
    instrument_id: str
    execution_route_id: str
    correlation_id: str
    causation_id: str | None
    idempotency_key: str
    order_intent_id: str
    order_id: str
    side: str
    order_type: str
    quantity: str
    limit_price: str | None
    time_in_force: str
    expire_at_utc: str | None


def request() -> M07SubmitOrderRequest:
    return M07SubmitOrderRequest(
        "cmd_018f0000-0000-7000-8000-000000000001",
        "SUBMIT_ORDER",
        "authctx_018f0000-0000-7000-8000-000000000001",
        "TESTNET",
        "ws_018f0000-0000-7000-8000-000000000001",
        "port_018f0000-0000-7000-8000-000000000001",
        "xacc_018f0000-0000-7000-8000-000000000001",
        "sinst_018f0000-0000-7000-8000-000000000001",
        "STRATEGY_INSTANCE",
        "instr_018f0000-0000-7000-8000-000000000001",
        "xroute_018f0000-0000-7000-8000-000000000001",
        "corr_018f0000-0000-7000-8000-000000000001",
        None,
        "cmd_018f0000-0000-7000-8000-000000000001",
        "oint_018f0000-0000-7000-8000-000000000001",
        "ord_018f0000-0000-7000-8000-000000000001",
        "BUY",
        "LIMIT",
        "2",
        "10",
        "GTC",
        None,
    )


M07_FIELDS = tuple(thaw(EXPECTED_PROTOCOLS["executable_boundary_schemas"]["M07SubmitOrderRequest"]))


def validate_m07(r: M07SubmitOrderRequest) -> bool:
    if tuple(x.name for x in fields(r)) != M07_FIELDS:
        return False
    ids: tuple[tuple[str, str], ...] = (
        (r.command_id, "cmd"),
        (r.authority_context_id, "authctx"),
        (r.workspace_id, "ws"),
        (r.portfolio_id, "port"),
        (r.exchange_account_id, "xacc"),
        (r.instrument_id, "instr"),
        (r.execution_route_id, "xroute"),
        (r.correlation_id, "corr"),
        (r.idempotency_key, "cmd"),
        (r.order_intent_id, "oint"),
        (r.order_id, "ord"),
    )
    if r.strategy_instance_id is not None:
        ids += ((r.strategy_instance_id, "sinst"),)
    if r.causation_id is not None:
        ids += ((r.causation_id, "cause"),)
    return (
        all(valid_id(v, p) for v, p in ids)
        and r.operation_type == "SUBMIT_ORDER"
        and r.environment in ENVIRONMENTS
        and r.source_type in ("STRATEGY_INSTANCE", "OPERATOR", "SYSTEM_RECONCILIATION")
        and r.side in ("BUY", "SELL")
        and r.order_type in ("MARKET", "LIMIT")
        and r.time_in_force in ("GTC", "IOC", "FOK", "GTD")
        and r.idempotency_key == r.command_id
        and exact_fraction(r.quantity) not in (None, Fraction(0))
        and (
            (
                r.order_type == "LIMIT"
                and r.limit_price is not None
                and exact_fraction(r.limit_price) not in (None, Fraction(0))
            )
            or (r.order_type == "MARKET" and r.limit_price is None)
        )
        and (
            (r.time_in_force == "GTD" and r.expire_at_utc is not None and utc_text(r.expire_at_utc))
            or (r.time_in_force != "GTD" and r.expire_at_utc is None)
        )
        and (
            (r.source_type == "STRATEGY_INSTANCE" and r.strategy_instance_id is not None)
            or (r.source_type != "STRATEGY_INSTANCE" and r.strategy_instance_id is None)
        )
    )


def m07_fingerprint(r: M07SubmitOrderRequest) -> str:
    data = asdict(r)
    data.pop("correlation_id")
    return fingerprint(data)


@dataclass(frozen=True)
class CoreAcceptedCommandProjection:
    entries: tuple[tuple[str, str], ...]
    membership_id: str


@dataclass(frozen=True)
class M07PrevalidatedAcceptedCommandContext:
    request: M07SubmitOrderRequest
    request_fingerprint_sha256: str
    membership_id: str
    context_fingerprint_sha256: str


def attest_command(
    r: M07SubmitOrderRequest, projection: CoreAcceptedCommandProjection
) -> M07PrevalidatedAcceptedCommandContext | str:
    if not validate_m07(r) or not core_accepts(
        projection.membership_id, fingerprint(projection.entries)
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    fp = m07_fingerprint(r)
    if (r.command_id, fp) not in projection.entries:
        return "TRUSTED_CONTEXT_FAILURE"
    return M07PrevalidatedAcceptedCommandContext(
        r, fp, projection.membership_id, fingerprint((r, fp, projection.membership_id))
    )


def accepted_command_projection(r: M07SubmitOrderRequest) -> CoreAcceptedCommandProjection:
    entries = ((r.command_id, m07_fingerprint(r)),)
    membership = f"m07:{fingerprint(entries)[:16]}"
    bind_core_content(membership, fingerprint(entries))
    return CoreAcceptedCommandProjection(entries, membership)


@dataclass(frozen=True)
class ProductCapabilitiesProjection:
    environment: str
    authorized_operations: tuple[str, ...]
    edition_policy: str
    semantic_fingerprint_sha256: str
    membership_id: str


@dataclass(frozen=True)
class InstrumentProjection:
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    exchange_id: str
    instrument_id: str
    metadata_version: int
    base_asset_reference: AssetReference
    quote_asset_reference: AssetReference
    semantic_fingerprint_sha256: str
    membership_id: str


@dataclass(frozen=True)
class RouteProjection:
    environment: str
    execution_route_id: str
    strategy_instance_id: str | None
    source_type: str
    readiness: str
    semantic_fingerprint_sha256: str
    membership_id: str


@dataclass(frozen=True)
class PrevalidatedExecutionAuthorityContext:
    product: ProductCapabilitiesProjection
    instrument: InstrumentProjection
    route: RouteProjection
    command: M07PrevalidatedAcceptedCommandContext
    context_fingerprint_sha256: str


def _upstream_fixture(
    r: M07SubmitOrderRequest, mode: str
) -> tuple[
    ProductCapabilitiesProjection,
    InstrumentProjection,
    RouteProjection,
    CoreAcceptedCommandProjection,
]:
    # Fixture simulates projections that existed before M0.9 resolution; values are not derived by validator.
    live = r.environment == "LIVE"
    future_live = mode == "FUTURE_LIVE"
    allowed = not live or future_live
    operations = ("SUBMIT_ORDER",) if allowed else ()
    pv = (
        r.environment,
        operations,
        "FUTURE_SIGNED_LIVE_POLICY"
        if future_live
        else ("LIVE_BLOCKED_BY_EDITION" if live else "CURRENT_EDITION_ALLOWED"),
    )
    pfp = fingerprint(pv)
    pm = bind_core_content(f"m04:{pfp[:16]}", pfp)
    p = ProductCapabilitiesProjection(*pv, pfp, pm)
    iv = (
        r.workspace_id,
        r.portfolio_id,
        r.environment,
        r.exchange_account_id,
        "binance",
        r.instrument_id,
        7,
        BTC,
        USD,
    )
    ifp = fingerprint(iv)
    im = bind_core_content(f"m05:{ifp[:16]}", ifp)
    i = InstrumentProjection(*iv, ifp, im)
    rv = (
        r.environment,
        r.execution_route_id,
        r.strategy_instance_id,
        r.source_type,
        "READY" if allowed else "POLICY_BLOCKED",
    )
    rfp = fingerprint(rv)
    rm = bind_core_content(f"m06:{rfp[:16]}", rfp)
    route = RouteProjection(*rv, rfp, rm)
    CORE_CURRENT_PRODUCT[r.environment] = CoreCurrentProductDesignation(r.environment, pm)
    CORE_CURRENT_ROUTE[(r.environment, r.execution_route_id)] = CoreCurrentRouteDesignation(
        r.environment, r.execution_route_id, rm
    )
    return p, i, route, accepted_command_projection(r)


def validate_execution_authority_context(
    auth: PrevalidatedExecutionAuthorityContext,
    *,
    require_current_mutable_authority: bool,
) -> str:
    if not isinstance(auth, PrevalidatedExecutionAuthorityContext):
        return "TRUSTED_CONTEXT_FAILURE"
    r, p, i, route, command = (
        auth.command.request,
        auth.product,
        auth.instrument,
        auth.route,
        auth.command,
    )
    pv = (p.environment, p.authorized_operations, p.edition_policy)
    iv = (
        i.workspace_id,
        i.portfolio_id,
        i.environment,
        i.exchange_account_id,
        i.exchange_id,
        i.instrument_id,
        i.metadata_version,
        i.base_asset_reference,
        i.quote_asset_reference,
    )
    rv = (
        route.environment,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
        route.readiness,
    )
    request_fp = m07_fingerprint(r) if validate_m07(r) else ""
    command_entries_fp = fingerprint(((r.command_id, request_fp),))
    command_context_fp = fingerprint((r, request_fp, command.membership_id))
    composite_fp = fingerprint(
        (
            p.semantic_fingerprint_sha256,
            i.semantic_fingerprint_sha256,
            route.semantic_fingerprint_sha256,
            command.context_fingerprint_sha256,
        )
    )
    if not (
        request_fp == command.request_fingerprint_sha256
        and core_accepts(command.membership_id, command_entries_fp)
        and command.context_fingerprint_sha256 == command_context_fp
        and p.semantic_fingerprint_sha256 == fingerprint(pv)
        and i.semantic_fingerprint_sha256 == fingerprint(iv)
        and route.semantic_fingerprint_sha256 == fingerprint(rv)
        and core_accepts(p.membership_id, p.semantic_fingerprint_sha256)
        and core_accepts(i.membership_id, i.semantic_fingerprint_sha256)
        and core_accepts(route.membership_id, route.semantic_fingerprint_sha256)
        and valid_asset(i.base_asset_reference)
        and valid_asset(i.quote_asset_reference)
        and auth.context_fingerprint_sha256 == composite_fp
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    if (p.environment, i.environment, route.environment) != (r.environment,) * 3 or (
        i.workspace_id,
        i.portfolio_id,
        i.exchange_account_id,
        i.instrument_id,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
    ) != (
        r.workspace_id,
        r.portfolio_id,
        r.exchange_account_id,
        r.instrument_id,
        r.execution_route_id,
        r.strategy_instance_id,
        r.source_type,
    ):
        return "UPSTREAM_AUTHORITY_DENIED"
    if require_current_mutable_authority and (
        CORE_CURRENT_PRODUCT.get(r.environment)
        != CoreCurrentProductDesignation(r.environment, p.membership_id)
        or CORE_CURRENT_ROUTE.get((r.environment, r.execution_route_id))
        != CoreCurrentRouteDesignation(r.environment, r.execution_route_id, route.membership_id)
        or "SUBMIT_ORDER" not in p.authorized_operations
        or route.readiness != "READY"
    ):
        return "UPSTREAM_AUTHORITY_DENIED"
    return "OK"


def resolve_execution_authority(
    r: M07SubmitOrderRequest,
    p: ProductCapabilitiesProjection,
    i: InstrumentProjection,
    route: RouteProjection,
    commands: CoreAcceptedCommandProjection,
) -> PrevalidatedExecutionAuthorityContext | str:
    accepted = attest_command(r, commands)
    if isinstance(accepted, str):
        return accepted
    pv = (p.environment, p.authorized_operations, p.edition_policy)
    iv = (
        i.workspace_id,
        i.portfolio_id,
        i.environment,
        i.exchange_account_id,
        i.exchange_id,
        i.instrument_id,
        i.metadata_version,
        i.base_asset_reference,
        i.quote_asset_reference,
    )
    rv = (
        route.environment,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
        route.readiness,
    )
    if not (
        core_accepts(p.membership_id, p.semantic_fingerprint_sha256)
        and core_accepts(i.membership_id, i.semantic_fingerprint_sha256)
        and core_accepts(route.membership_id, route.semantic_fingerprint_sha256)
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    if (
        p.semantic_fingerprint_sha256 != fingerprint(pv)
        or i.semantic_fingerprint_sha256 != fingerprint(iv)
        or route.semantic_fingerprint_sha256 != fingerprint(rv)
        or not valid_asset(i.base_asset_reference)
        or not valid_asset(i.quote_asset_reference)
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    if r.environment == "LIVE":
        future_memberships = (
            p.authorized_operations,
            p.edition_policy,
            route.readiness,
        ) == (
            ("SUBMIT_ORDER",),
            "FUTURE_SIGNED_LIVE_POLICY",
            "READY",
        )
        current_denial = (
            p.authorized_operations,
            p.edition_policy,
            route.readiness,
        ) == (
            (),
            "LIVE_BLOCKED_BY_EDITION",
            "POLICY_BLOCKED",
        )
        if not future_memberships and not current_denial:
            return "TRUSTED_CONTEXT_FAILURE"
    if (p.environment, i.environment, route.environment) != (
        r.environment,
        r.environment,
        r.environment,
    ) or (
        i.workspace_id,
        i.portfolio_id,
        i.exchange_account_id,
        i.instrument_id,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
    ) != (
        r.workspace_id,
        r.portfolio_id,
        r.exchange_account_id,
        r.instrument_id,
        r.execution_route_id,
        r.strategy_instance_id,
        r.source_type,
    ):
        return "UPSTREAM_AUTHORITY_DENIED"
    if "SUBMIT_ORDER" not in p.authorized_operations or route.readiness != "READY":
        return "UPSTREAM_AUTHORITY_DENIED"
    result = PrevalidatedExecutionAuthorityContext(
        p,
        i,
        route,
        accepted,
        fingerprint(
            (
                p.semantic_fingerprint_sha256,
                i.semantic_fingerprint_sha256,
                route.semantic_fingerprint_sha256,
                accepted.context_fingerprint_sha256,
            )
        ),
    )
    status = validate_execution_authority_context(result, require_current_mutable_authority=True)
    return result if status == "OK" else status


@dataclass(frozen=True)
class RiskPolicyRecord:
    risk_policy_id: str
    revision: int
    environment: str
    scope_type: str
    scope_id: str
    action: str
    limits: tuple[tuple[str, Fraction, AssetReference], ...]
    semantic_fingerprint_sha256: str


@dataclass(frozen=True)
class PrevalidatedRiskPolicyContext:
    records: tuple[RiskPolicyRecord, ...]
    current: tuple[tuple[str, str, str, str, int], ...]
    membership_id: str
    context_fingerprint_sha256: str


@dataclass(frozen=True)
class EffectivePolicy:
    bindings: tuple[tuple[str, int, str, str, str, str], ...]
    limits: tuple[tuple[str, Fraction, AssetReference, str], ...]
    fingerprint_sha256: str


SUPPORTED = tuple(thaw(EXPECTED_PROTOCOLS["supported_spot_limit_registry"]))
UNSUPPORTED = tuple(thaw(EXPECTED_PROTOCOLS["unsupported_limit_registry"]))


def policy_record(
    limit: str = "MAX_ORDER_NOTIONAL",
    threshold: Fraction = Fraction(20),
    unit: AssetReference = USD,
    environment: str = "TESTNET",
    scope_type: str = "WORKSPACE",
    scope_id: str = "ws_018f0000-0000-7000-8000-000000000001",
    action: str = "ALLOW",
    revision: int = 1,
    policy_id: str = "rpol_018f0000-0000-7000-8000-000000000001",
) -> RiskPolicyRecord:
    vals = (
        policy_id,
        revision,
        environment,
        scope_type,
        scope_id,
        action,
        ((limit, threshold, unit),),
    )
    return RiskPolicyRecord(*vals, fingerprint(vals))


def policy_context(
    *records: RiskPolicyRecord,
    current: tuple[tuple[str, str, str, str, int], ...] | None = None,
) -> PrevalidatedRiskPolicyContext:
    if current is None:
        current = tuple(
            (x.risk_policy_id, x.environment, x.scope_type, x.scope_id, x.revision) for x in records
        )
    content = fingerprint((records, current))
    membership = bind_core_content(f"policy:{content[:16]}", content)
    return PrevalidatedRiskPolicyContext(
        records, current, membership, fingerprint((records, current, membership))
    )


def resolve_policy(
    ctx: PrevalidatedRiskPolicyContext, r: M07SubmitOrderRequest
) -> EffectivePolicy | str:
    if not core_accepts(
        ctx.membership_id, fingerprint((ctx.records, ctx.current))
    ) or ctx.context_fingerprint_sha256 != fingerprint(
        (ctx.records, ctx.current, ctx.membership_id)
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    designations: dict[tuple[str, str, str, str], int] = {}
    for policy_id, environment, scope_type, scope_id, revision in ctx.current:
        key = (policy_id, environment, scope_type, scope_id)
        if key in designations:
            return "POLICY_CONFLICT"
        designations[key] = revision
    selected: list[RiskPolicyRecord] = []
    history: dict[tuple[str, str, str, str], list[RiskPolicyRecord]] = {}
    scope_ids = {
        "PRODUCT_SYSTEM": "product",
        "WORKSPACE": r.workspace_id,
        "PORTFOLIO": r.portfolio_id,
        "EXCHANGE_ACCOUNT": r.exchange_account_id,
        "STRATEGY_INSTANCE": r.strategy_instance_id,
        "INSTRUMENT": r.instrument_id,
        "EXECUTION_ROUTE": r.execution_route_id,
    }
    for rec in ctx.records:
        vals = (
            rec.risk_policy_id,
            rec.revision,
            rec.environment,
            rec.scope_type,
            rec.scope_id,
            rec.action,
            rec.limits,
        )
        if rec.semantic_fingerprint_sha256 != fingerprint(vals):
            return "TRUSTED_CONTEXT_FAILURE"
        if (
            not valid_id(rec.risk_policy_id, "rpol")
            or isinstance(rec.revision, bool)
            or not isinstance(rec.revision, int)
            or rec.revision <= 0
            or rec.environment not in ENVIRONMENTS
            or rec.scope_type not in SCOPE_ORDER
            or rec.action not in ("ALLOW", "DENY")
        ):
            return "TRUSTED_CONTEXT_FAILURE"
        if rec.scope_type != "PRODUCT_SYSTEM":
            prefixes = {
                "WORKSPACE": "ws",
                "PORTFOLIO": "port",
                "EXCHANGE_ACCOUNT": "xacc",
                "STRATEGY_INSTANCE": "sinst",
                "INSTRUMENT": "instr",
                "EXECUTION_ROUTE": "xroute",
            }
            if not valid_id(rec.scope_id, prefixes[rec.scope_type]):
                return "TRUSTED_CONTEXT_FAILURE"
        elif rec.scope_id != "product":
            return "TRUSTED_CONTEXT_FAILURE"
        key = (rec.risk_policy_id, rec.environment, rec.scope_type, rec.scope_id)
        history.setdefault(key, []).append(rec)
        for name, value, unit in rec.limits:
            if name not in SUPPORTED or not isinstance(value, Fraction) or not valid_asset(unit):
                return "UNSUPPORTED_RISK_SEMANTICS"
    if set(designations) != set(history):
        return "POLICY_CONFLICT"
    for key, revision in designations.items():
        records = history.get(key, [])
        designated = [x for x in records if x.revision == revision]
        if len(designated) != 1 or (records and revision != max(x.revision for x in records)):
            return "POLICY_CONFLICT"
        rec = designated[0]
        if rec.environment == r.environment and scope_ids.get(rec.scope_type) == rec.scope_id:
            selected.append(rec)
    if not selected:
        return "RISK_CONTEXT_INCOMPLETE"
    collisions: dict[tuple[str, str, str, int], set[str]] = {}
    for rec in selected:
        collision_key = (rec.environment, rec.scope_type, rec.scope_id, rec.revision)
        collisions.setdefault(collision_key, set()).add(rec.semantic_fingerprint_sha256)
    if any(len(fingerprints) > 1 for fingerprints in collisions.values()):
        return "POLICY_CONFLICT"
    if any(x.action == "DENY" for x in selected):
        return "RISK_DENIED"
    selected.sort(key=lambda x: (SCOPE_ORDER.index(x.scope_type), x.scope_id, x.revision))
    bindings = tuple(
        (
            x.risk_policy_id,
            x.revision,
            x.scope_type,
            x.scope_id,
            x.action,
            x.semantic_fingerprint_sha256,
        )
        for x in selected
    )
    limits = []
    for name in SUPPORTED:
        for unit in sorted({u for x in selected for n, _, u in x.limits if n == name}):
            choices = [(v, x) for x in selected for n, v, u in x.limits if n == name and u == unit]
            chosen = (max if name.startswith("MIN_") else min)(choices, key=lambda z: z[0])
            limits.append((name, chosen[0], unit, f"{chosen[1].scope_type}:{chosen[1].scope_id}"))
    return EffectivePolicy(bindings, tuple(limits), fingerprint(bindings))


@dataclass(frozen=True)
class KillSwitchRecord:
    scope_type: str
    scope_id: str
    environment: str
    state: str
    source_revision: int
    effective_at_utc: str
    generation: int
    accepted_authority_fingerprint_sha256: str
    record_fingerprint_sha256: str


@dataclass(frozen=True)
class PrevalidatedKillSwitchContext:
    history: tuple[KillSwitchRecord, ...]
    membership_id: str
    context_fingerprint_sha256: str


def kill_record(
    state: str = "INACTIVE",
    generation: int = 1,
    source_revision: int = 1,
    effective: str = "2026-01-01T00:00:00Z",
    environment: str = "TESTNET",
    scope_type: str = "WORKSPACE",
    scope_id: str = "ws_018f0000-0000-7000-8000-000000000001",
) -> KillSwitchRecord:
    vals = (
        scope_type,
        scope_id,
        environment,
        state,
        source_revision,
        effective,
        generation,
        "accepted-switch-authority",
    )
    return KillSwitchRecord(*vals, fingerprint(vals))


def switch_context(*records: KillSwitchRecord) -> PrevalidatedKillSwitchContext:
    content = fingerprint(records)
    membership = bind_core_content(f"switch:{content[:16]}", content)
    return PrevalidatedKillSwitchContext(records, membership, fingerprint((records, membership)))


def resolve_switches(
    ctx: PrevalidatedKillSwitchContext, r: M07SubmitOrderRequest
) -> tuple[str, tuple[tuple[Any, ...], ...], str]:
    if not core_accepts(
        ctx.membership_id, fingerprint(ctx.history)
    ) or ctx.context_fingerprint_sha256 != fingerprint((ctx.history, ctx.membership_id)):
        return "TRUSTED_CONTEXT_FAILURE", (), ""
    grouped: dict[tuple[str, str, str], list[KillSwitchRecord]] = {}
    scope_ids = {
        "PRODUCT_SYSTEM": "product",
        "WORKSPACE": r.workspace_id,
        "PORTFOLIO": r.portfolio_id,
        "EXCHANGE_ACCOUNT": r.exchange_account_id,
        "STRATEGY_INSTANCE": r.strategy_instance_id,
        "INSTRUMENT": r.instrument_id,
        "EXECUTION_ROUTE": r.execution_route_id,
    }
    for x in ctx.history:
        vals = (
            x.scope_type,
            x.scope_id,
            x.environment,
            x.state,
            x.source_revision,
            x.effective_at_utc,
            x.generation,
            x.accepted_authority_fingerprint_sha256,
        )
        prefixes = {
            "WORKSPACE": "ws",
            "PORTFOLIO": "port",
            "EXCHANGE_ACCOUNT": "xacc",
            "STRATEGY_INSTANCE": "sinst",
            "INSTRUMENT": "instr",
            "EXECUTION_ROUTE": "xroute",
        }
        if (
            x.record_fingerprint_sha256 != fingerprint(vals)
            or x.scope_type not in SCOPE_ORDER
            or x.environment not in ENVIRONMENTS
            or x.state not in ("INACTIVE", "ACTIVE")
            or type(x.source_revision) is not int
            or x.source_revision <= 0
            or type(x.generation) is not int
            or x.generation <= 0
            or not utc_text(x.effective_at_utc)
            or (x.scope_type == "PRODUCT_SYSTEM" and x.scope_id != "product")
            or (
                x.scope_type != "PRODUCT_SYSTEM"
                and not valid_id(x.scope_id, prefixes[x.scope_type])
            )
        ):
            return "TRUSTED_CONTEXT_FAILURE", (), ""
        grouped.setdefault((x.scope_type, x.scope_id, x.environment), []).append(x)
    current = []
    for values in grouped.values():
        gs = [x.generation for x in values]
        if gs != sorted(gs) or len(gs) != len(set(gs)):
            return "TRUSTED_CONTEXT_FAILURE", (), ""
        x = values[-1]
        if x.environment == r.environment and scope_ids.get(x.scope_type) == x.scope_id:
            current.append(x)
    current.sort(key=lambda x: (SCOPE_ORDER.index(x.scope_type), x.scope_id))
    bindings = tuple(
        (
            x.scope_type,
            x.scope_id,
            x.state,
            x.source_revision,
            x.effective_at_utc,
            x.generation,
            x.accepted_authority_fingerprint_sha256,
            x.record_fingerprint_sha256,
        )
        for x in current
    )
    return (
        "KILL_SWITCH_ACTIVE" if any(x.state == "ACTIVE" for x in current) else "OK",
        bindings,
        fingerprint(bindings),
    )


@dataclass(frozen=True)
class ValuationContext:
    subject_reference: AssetReference
    valuation_unit: AssetReference
    rate: Fraction
    source_id: str
    observed_at_utc: str
    effective_at_utc: str
    as_of_utc: str
    stale_after_utc: str
    source_fingerprint_sha256: str
    accepted_authority_fingerprint_sha256: str
    context_fingerprint_sha256: str


@dataclass(frozen=True)
class InventoryExposure:
    instrument_id: str
    environment: str
    asset_reference: AssetReference
    quantity: Fraction
    as_of_utc: str


@dataclass(frozen=True)
class AccountingRiskProjection:
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_ids: tuple[str, ...]
    as_of_utc: str
    reporting_asset_reference: AssetReference
    owned_balances: tuple[tuple[AssetReference, Fraction], ...]
    available_capital: tuple[tuple[AssetReference, Fraction], ...]
    reserved_capital: tuple[tuple[AssetReference, Fraction], ...]
    inventory_exposure: tuple[InventoryExposure, ...]
    valuations: tuple[ValuationContext, ...]
    reconciliation_outcomes: tuple[str, ...]
    projection_fingerprint_sha256: str
    accepted_membership_id: str


def valuation(
    environment: str = "TESTNET", rate: Fraction = Fraction(10), unit: AssetReference = USD
) -> ValuationContext:
    source = (
        "price-source-1",
        environment,
        BTC,
        unit,
        rate,
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
        "2026-01-01T00:01:00Z",
    )
    sfp = fingerprint(source)
    accepted_content = fingerprint(source)
    authority = f"valuation:{accepted_content[:16]}"
    bind_core_content(authority, accepted_content)
    vals = (BTC, unit, rate, source[0], source[5], source[6], source[7], source[8], sfp, authority)
    return ValuationContext(*vals, fingerprint(vals))


def accounting(
    phase: str = "PRE",
    environment: str = "TESTNET",
    available: Fraction | None = None,
    reserved: Fraction | None = None,
    vals: tuple[ValuationContext, ...] | None = None,
    inventory: tuple[InventoryExposure, ...] | None = None,
    membership: str | None = None,
    reconciliation: tuple[str, ...] = ("MATCH",),
) -> AccountingRiskProjection:
    available = Fraction(100 if phase == "PRE" else 80) if available is None else available
    reserved = Fraction(0 if phase == "PRE" else 20) if reserved is None else reserved
    vals = (valuation(environment),) if vals is None else vals
    inventory = (
        (
            InventoryExposure(
                "instr_018f0000-0000-7000-8000-000000000001",
                environment,
                BTC,
                Fraction(1),
                "2026-01-01T00:00:00Z",
            ),
        )
        if inventory is None
        else inventory
    )
    base = (
        "ws_018f0000-0000-7000-8000-000000000001",
        "port_018f0000-0000-7000-8000-000000000001",
        environment,
        ("xacc_018f0000-0000-7000-8000-000000000001",),
        "2026-01-01T00:00:00Z",
        USD,
        ((USD, available + reserved), (BTC, Fraction(1))),
        ((USD, available),),
        ((USD, reserved),),
        inventory,
        vals,
        reconciliation,
    )
    projection_fp = fingerprint(base)
    if membership is None:
        membership = bind_core_content(
            f"account-{phase.lower()}:{projection_fp[:16]}", projection_fp
        )
    return AccountingRiskProjection(*base, projection_fp, membership)


def accounting_semantics(a: AccountingRiskProjection) -> tuple[Any, ...]:
    return tuple(
        asdict(a)[k]
        for k in asdict(a)
        if k not in ("projection_fingerprint_sha256", "accepted_membership_id")
    )


def validate_accounting(a: AccountingRiskProjection, r: M07SubmitOrderRequest) -> str:
    if not core_accepts(
        a.accepted_membership_id, a.projection_fingerprint_sha256
    ) or a.projection_fingerprint_sha256 != fingerprint(accounting_semantics(a)):
        return "TRUSTED_CONTEXT_FAILURE"
    if (a.workspace_id, a.portfolio_id, a.environment) != (
        r.workspace_id,
        r.portfolio_id,
        r.environment,
    ) or r.exchange_account_id not in a.exchange_account_ids:
        return "RISK_CONTEXT_INCOMPLETE"
    if not a.reconciliation_outcomes or any(
        outcome != "MATCH" for outcome in a.reconciliation_outcomes
    ):
        return "RISK_CONTEXT_INCOMPLETE"
    buckets = list(a.owned_balances) + list(a.available_capital) + list(a.reserved_capital)
    if any(not valid_asset(x) or not isinstance(q, Fraction) or q < 0 for x, q in buckets) or any(
        len([x for x, _ in group]) != len(set(x for x, _ in group))
        for group in (a.owned_balances, a.available_capital, a.reserved_capital)
    ):
        return "RISK_CONTEXT_INCOMPLETE"
    if any(
        p.environment != a.environment
        or p.as_of_utc != a.as_of_utc
        or not valid_asset(p.asset_reference)
        or not isinstance(p.quantity, Fraction)
        for p in a.inventory_exposure
    ):
        return "RISK_CONTEXT_INCOMPLETE"
    for v in a.valuations:
        vals = (
            v.subject_reference,
            v.valuation_unit,
            v.rate,
            v.source_id,
            v.observed_at_utc,
            v.effective_at_utc,
            v.as_of_utc,
            v.stale_after_utc,
            v.source_fingerprint_sha256,
            v.accepted_authority_fingerprint_sha256,
        )
        source_values = (
            v.source_id,
            a.environment,
            v.subject_reference,
            v.valuation_unit,
            v.rate,
            v.observed_at_utc,
            v.effective_at_utc,
            v.as_of_utc,
            v.stale_after_utc,
        )
        if (
            v.context_fingerprint_sha256 != fingerprint(vals)
            or v.source_fingerprint_sha256 != fingerprint(source_values)
            or not core_accepts(
                v.accepted_authority_fingerprint_sha256, v.source_fingerprint_sha256
            )
            or not valid_asset(v.subject_reference)
            or not valid_asset(v.valuation_unit)
            or not isinstance(v.rate, Fraction)
            or v.rate <= 0
            or not all(
                utc_text(t)
                for t in (v.observed_at_utc, v.effective_at_utc, v.as_of_utc, v.stale_after_utc)
            )
            or not (v.observed_at_utc <= v.effective_at_utc <= v.as_of_utc < v.stale_after_utc)
            or v.as_of_utc != a.as_of_utc
        ):
            return "RISK_CONTEXT_INCOMPLETE"
    return "OK"


def trusted_price(a: AccountingRiskProjection, i: InstrumentProjection) -> Fraction | None:
    found = [
        v
        for v in a.valuations
        if v.subject_reference == i.base_asset_reference
        and v.valuation_unit == i.quote_asset_reference
        and v.as_of_utc == a.as_of_utc
    ]
    return found[0].rate if len(found) == 1 else None


def derive_gross(a: AccountingRiskProjection) -> Fraction | None:
    total = Fraction(0)
    for p in a.inventory_exposure:
        found = [
            v
            for v in a.valuations
            if p.environment == a.environment
            and v.subject_reference == p.asset_reference
            and v.valuation_unit == a.reporting_asset_reference
            and v.as_of_utc == p.as_of_utc == a.as_of_utc
            and v.source_fingerprint_sha256
            == fingerprint(
                (
                    v.source_id,
                    a.environment,
                    v.subject_reference,
                    v.valuation_unit,
                    v.rate,
                    v.observed_at_utc,
                    v.effective_at_utc,
                    v.as_of_utc,
                    v.stale_after_utc,
                )
            )
        ]
        if len(found) != 1:
            return None
        total += abs(p.quantity) * found[0].rate
    return total


@dataclass(frozen=True)
class ReservationRequirementProjection:
    command_id: str
    order_id: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    asset_reference: AssetReference
    required_quantity: Fraction
    derivation_fingerprint_sha256: str


@dataclass(frozen=True)
class CapitalReservationFact:
    audit_event_id: str
    source_type: str
    workspace_id: str
    portfolio_id: str
    environment: str
    effective_at_utc: str
    provenance: str
    source_fingerprint_sha256: str
    exchange_account_id: str
    asset_reference: AssetReference
    quantity: Fraction
    order_id: str
    command_id: str


@dataclass(frozen=True)
class ReservationState:
    command_id: str
    order_id: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    asset_reference: AssetReference
    original_quantity: Fraction
    remaining_quantity: Fraction
    source_audit_event_id: str
    source_fingerprint_sha256: str
    state_fingerprint_sha256: str
    accepted_membership_id: str


def derive_requirement(
    auth: PrevalidatedExecutionAuthorityContext, pre: AccountingRiskProjection
) -> ReservationRequirementProjection | str:
    r = auth.command.request
    i = auth.instrument
    q = exact_fraction(r.quantity)
    if q is None:
        return "RISK_CONTEXT_INCOMPLETE"
    if r.side == "BUY":
        price = exact_fraction(r.limit_price) if r.limit_price else trusted_price(pre, i)
        if price is None:
            return "RISK_CONTEXT_INCOMPLETE"
        asset = i.quote_asset_reference
        required = q * price
    else:
        asset = i.base_asset_reference
        required = q
    vals = (
        r.command_id,
        r.order_id,
        r.workspace_id,
        r.portfolio_id,
        r.environment,
        r.exchange_account_id,
        asset,
        required,
    )
    return ReservationRequirementProjection(*vals, fingerprint(vals))


_fact_counter = 0


def reservation_fact(req: ReservationRequirementProjection) -> CapitalReservationFact:
    global _fact_counter
    _fact_counter += 1
    vals = (
        f"evt_018f0000-0000-7000-8000-{_fact_counter:012x}",
        "capital_reservation",
        req.workspace_id,
        req.portfolio_id,
        req.environment,
        "2026-01-01T00:00:01Z",
        "M0.8 accepted accounting economic fact",
        req.exchange_account_id,
        req.asset_reference,
        req.required_quantity,
        req.order_id,
        req.command_id,
    )
    source = fingerprint(vals)
    fact = CapitalReservationFact(*vals[:7], source, *vals[7:])
    bind_core_content(fact.audit_event_id, fact.source_fingerprint_sha256)
    return fact


def reservation_state(f: CapitalReservationFact) -> ReservationState:
    vals = (
        f.command_id,
        f.order_id,
        f.workspace_id,
        f.portfolio_id,
        f.environment,
        f.exchange_account_id,
        f.asset_reference,
        f.quantity,
        f.quantity,
        f.audit_event_id,
        f.source_fingerprint_sha256,
    )
    state_fp = fingerprint(vals)
    membership = bind_core_content(f"reservation-state:{state_fp[:16]}", state_fp)
    return ReservationState(*vals, state_fp, membership)


def validate_fact(f: CapitalReservationFact) -> str:
    vals = (
        f.audit_event_id,
        f.source_type,
        f.workspace_id,
        f.portfolio_id,
        f.environment,
        f.effective_at_utc,
        f.provenance,
        f.exchange_account_id,
        f.asset_reference,
        f.quantity,
        f.order_id,
        f.command_id,
    )
    return (
        "OK"
        if valid_id(f.audit_event_id, "evt")
        and f.source_type == "capital_reservation"
        and utc_text(f.effective_at_utc)
        and valid_asset(f.asset_reference)
        and isinstance(f.quantity, Fraction)
        and f.quantity > 0
        and f.source_fingerprint_sha256 == fingerprint(vals)
        and core_accepts(f.audit_event_id, f.source_fingerprint_sha256)
        else "TRUSTED_CONTEXT_FAILURE"
    )


def validate_transition(
    pre: AccountingRiskProjection,
    post: AccountingRiskProjection,
    req: ReservationRequirementProjection,
    f: CapitalReservationFact,
    s: ReservationState,
) -> str:
    if (
        validate_fact(f) != "OK"
        or not core_accepts(s.accepted_membership_id, s.state_fingerprint_sha256)
        or not core_accepts(pre.accepted_membership_id, pre.projection_fingerprint_sha256)
        or not core_accepts(post.accepted_membership_id, post.projection_fingerprint_sha256)
        or pre.accepted_membership_id == post.accepted_membership_id
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    if req.derivation_fingerprint_sha256 != fingerprint(
        tuple(asdict(req)[k] for k in asdict(req) if k != "derivation_fingerprint_sha256")
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    state_vals = (
        s.command_id,
        s.order_id,
        s.workspace_id,
        s.portfolio_id,
        s.environment,
        s.exchange_account_id,
        s.asset_reference,
        s.original_quantity,
        s.remaining_quantity,
        s.source_audit_event_id,
        s.source_fingerprint_sha256,
    )
    if s.state_fingerprint_sha256 != fingerprint(state_vals):
        return "TRUSTED_CONTEXT_FAILURE"
    identity = (
        req.command_id,
        req.order_id,
        req.workspace_id,
        req.portfolio_id,
        req.environment,
        req.exchange_account_id,
        req.asset_reference,
        req.required_quantity,
    )
    if (
        identity
        != (
            f.command_id,
            f.order_id,
            f.workspace_id,
            f.portfolio_id,
            f.environment,
            f.exchange_account_id,
            f.asset_reference,
            f.quantity,
        )
        or identity[:-1]
        != (
            s.command_id,
            s.order_id,
            s.workspace_id,
            s.portfolio_id,
            s.environment,
            s.exchange_account_id,
            s.asset_reference,
        )
        or (s.original_quantity, s.remaining_quantity) != (f.quantity, f.quantity)
    ):
        return "RESERVATION_INVALID"
    pre_a, pre_r = dict(pre.available_capital), dict(pre.reserved_capital)
    post_a, post_r = dict(post.available_capital), dict(post.reserved_capital)
    asset = f.asset_reference
    if (
        asset not in pre_a
        or asset not in pre_r
        or asset not in post_a
        or asset not in post_r
        or post_a[asset] != pre_a[asset] - f.quantity
        or post_r[asset] != pre_r[asset] + f.quantity
    ):
        return "RESERVATION_INVALID"
    if (
        pre.workspace_id,
        pre.portfolio_id,
        pre.environment,
        pre.exchange_account_ids,
        pre.as_of_utc,
    ) != (
        post.workspace_id,
        post.portfolio_id,
        post.environment,
        post.exchange_account_ids,
        post.as_of_utc,
    ):
        return "RESERVATION_INVALID"
    return "OK"


@dataclass(frozen=True)
class LimitResult:
    limit_type: str
    effective_threshold: Fraction
    observed_projected_value: Fraction | None
    unit_asset_reference: AssetReference
    supplying_policy_scope: str
    result: str
    reason_code: str


@dataclass(frozen=True)
class RiskDecision:
    command_id: str
    command_request_fingerprint_sha256: str
    order_id: str
    scope_binding: tuple[str, ...]
    environment: str
    effective_policy_fingerprint_sha256: str
    pre_reservation_accounting_projection_fingerprint_sha256: str
    reservation_requirement_fingerprint_sha256: str
    evaluated_at_utc: str
    ordered_limit_results: tuple[LimitResult, ...]
    kill_switch_result: str
    kill_switch_fence_sha256: str
    decision: str
    decision_fingerprint_sha256: str


@dataclass(frozen=True)
class PrevalidatedRiskDecisionContext:
    decision: RiskDecision
    membership_id: str


@dataclass(frozen=True)
class CoreAcceptedDecisionBinding:
    decision_fingerprint_sha256: str
    decision: RiskDecision


def risk_decision_fingerprint(decision: RiskDecision) -> str:
    return fingerprint(
        tuple(
            asdict(decision)[key]
            for key in asdict(decision)
            if key != "decision_fingerprint_sha256"
        )
    )


def derive_decision(
    auth: PrevalidatedExecutionAuthorityContext,
    pre: AccountingRiskProjection,
    effective: EffectivePolicy,
    switches: PrevalidatedKillSwitchContext,
    req: ReservationRequirementProjection,
) -> RiskDecision:
    r = auth.command.request
    i = auth.instrument
    sw, _, fence = resolve_switches(switches, r)
    price = (
        trusted_price(pre, i) if r.order_type == "MARKET" else exact_fraction(r.limit_price or "")
    )
    positions = [
        p
        for p in pre.inventory_exposure
        if p.instrument_id == r.instrument_id and p.asset_reference == i.base_asset_reference
    ]
    current = positions[0].quantity if len(positions) == 1 else None
    available = dict(pre.available_capital)
    results = []
    q = exact_fraction(r.quantity)
    assert q is not None
    for name, threshold, unit, supplier in effective.limits:
        observed = None
        if name == "MAX_ORDER_QUANTITY" and unit == i.base_asset_reference:
            observed = q
        elif name == "MAX_ORDER_NOTIONAL" and unit == i.quote_asset_reference and price is not None:
            observed = q * price
        elif (
            name == "MAX_POST_TRADE_POSITION_QUANTITY"
            and unit == i.base_asset_reference
            and current is not None
        ):
            observed = abs(current + (q if r.side == "BUY" else -q))
        elif (
            name == "MAX_POST_TRADE_POSITION_NOTIONAL"
            and unit == i.quote_asset_reference
            and current is not None
            and price is not None
        ):
            observed = abs(current + (q if r.side == "BUY" else -q)) * price
        elif (
            name == "MAX_GROSS_EXPOSURE"
            and unit == pre.reporting_asset_reference
            and current is not None
            and price is not None
        ):
            gross = derive_gross(pre)
            if gross is not None:
                observed = (
                    gross
                    - abs(current) * price
                    + abs(current + (q if r.side == "BUY" else -q)) * price
                )
        elif (
            name == "MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION"
            and unit == req.asset_reference
            and unit in available
        ):
            observed = available[unit] - req.required_quantity
        result = (
            "INCOMPLETE"
            if observed is None
            else (
                "FAIL"
                if (observed < threshold if name.startswith("MIN_") else observed > threshold)
                else "PASS"
            )
        )
        results.append(
            LimitResult(
                name,
                threshold,
                observed,
                unit,
                supplier,
                result,
                "MISSING_REQUIRED_INPUT"
                if result == "INCOMPLETE"
                else ("LIMIT_BREACH" if result == "FAIL" else "PASS"),
            )
        )
    if r.order_type == "MARKET" and price is None:
        results.append(
            LimitResult(
                "DISPATCH_RESERVATION_ECONOMICS",
                Fraction(0),
                None,
                i.quote_asset_reference,
                "SYSTEM",
                "INCOMPLETE",
                "MISSING_VALUATION",
            )
        )
    overall = (
        "DENY"
        if sw == "KILL_SWITCH_ACTIVE" or any(x.result == "FAIL" for x in results)
        else ("INCOMPLETE" if any(x.result == "INCOMPLETE" for x in results) else "ALLOW")
    )
    vals = (
        r.command_id,
        auth.command.request_fingerprint_sha256,
        r.order_id,
        (
            r.workspace_id,
            r.portfolio_id,
            r.exchange_account_id,
            r.instrument_id,
            r.execution_route_id,
        ),
        r.environment,
        effective.fingerprint_sha256,
        pre.projection_fingerprint_sha256,
        req.derivation_fingerprint_sha256,
        "2026-01-01T00:00:00Z",
        tuple(results),
        sw,
        fence,
        overall,
    )
    return RiskDecision(*vals, fingerprint(vals))


def decision_context(d: RiskDecision) -> PrevalidatedRiskDecisionContext:
    membership = bind_core_content(
        f"decision:{d.decision_fingerprint_sha256[:16]}", d.decision_fingerprint_sha256
    )
    CORE_ACCEPTED_DECISIONS[d.decision_fingerprint_sha256] = CoreAcceptedDecisionBinding(
        d.decision_fingerprint_sha256, d
    )
    return PrevalidatedRiskDecisionContext(d, membership)


def validate_decision(ctx: PrevalidatedRiskDecisionContext, expected: RiskDecision) -> str:
    return (
        "OK"
        if core_accepts(ctx.membership_id, ctx.decision.decision_fingerprint_sha256)
        and ctx.decision == expected
        and ctx.decision.decision_fingerprint_sha256 == risk_decision_fingerprint(ctx.decision)
        else "TRUSTED_CONTEXT_FAILURE"
    )


@dataclass(frozen=True)
class ExecutionLease:
    execution_lease_id: str
    command_id: str
    command_request_fingerprint_sha256: str
    order_id: str
    order_intent_id: str
    workspace_id: str
    portfolio_id: str
    environment: str
    exchange_account_id: str
    exchange_id: str
    instrument_id: str
    instrument_metadata_version: int
    execution_route_id: str
    strategy_instance_id: str | None
    source_identity: str
    side: str
    order_type: str
    quantity: Fraction
    limit_price: Fraction | None
    time_in_force: str
    order_expire_at_utc: str | None
    effective_policy_bindings: tuple[tuple[str, int, str, str, str, str], ...]
    effective_policy_fingerprint_sha256: str
    kill_switch_bindings: tuple[tuple[Any, ...], ...]
    kill_switch_fence_sha256: str
    pre_reservation_accounting_projection_fingerprint_sha256: str
    post_reservation_accounting_projection_fingerprint_sha256: str
    risk_decision_fingerprint_sha256: str
    reservation_source_audit_event_id: str
    reservation_source_fingerprint_sha256: str
    reservation_state_fingerprint_sha256: str
    reservation_asset_reference: AssetReference
    reservation_original_quantity: Fraction
    reservation_remaining_quantity: Fraction
    issued_at_utc: str
    expires_at_utc: str
    lease_fingerprint_sha256: str


@dataclass(frozen=True)
class CoreDispatchRecord:
    command_request_fingerprint_sha256: str
    order_id: str
    execution_lease_id: str
    lease_fingerprint_sha256: str
    state: str


@dataclass(frozen=True)
class CoreIssuedExecutionLeaseProjection:
    execution_lease_id: str
    lease_fingerprint_sha256: str
    command_id: str
    order_id: str


@dataclass
class CoreDispatchAuthorityState:
    records: dict[str, CoreDispatchRecord]
    side_effect_count: int = 0


_counter = 0


def new_lease_id() -> str:
    global _counter
    _counter += 1
    return f"lease_018f0000-0000-7000-8000-{_counter:012x}"


def lease_semantics(x: ExecutionLease) -> tuple[Any, ...]:
    return tuple(asdict(x)[k] for k in asdict(x) if k != "lease_fingerprint_sha256")


def issue_lease(
    auth: PrevalidatedExecutionAuthorityContext,
    pre: AccountingRiskProjection,
    post: AccountingRiskProjection,
    policy_ctx: PrevalidatedRiskPolicyContext,
    switches: PrevalidatedKillSwitchContext,
    req: ReservationRequirementProjection,
    fact: CapitalReservationFact,
    state: ReservationState,
    decision: PrevalidatedRiskDecisionContext,
) -> ExecutionLease | str:
    authority_status = validate_execution_authority_context(
        auth, require_current_mutable_authority=True
    )
    if authority_status != "OK":
        return authority_status
    r = auth.command.request
    effective = resolve_policy(policy_ctx, r)
    if isinstance(effective, str):
        return effective
    if validate_accounting(pre, r) != "OK" or validate_accounting(post, r) != "OK":
        return "RISK_CONTEXT_INCOMPLETE"
    expected_req = derive_requirement(auth, pre)
    if not isinstance(expected_req, ReservationRequirementProjection) or expected_req != req:
        return "RESERVATION_INVALID"
    if validate_transition(pre, post, req, fact, state) != "OK":
        return "RESERVATION_INVALID"
    expected = derive_decision(auth, pre, effective, switches, req)
    if validate_decision(decision, expected) != "OK":
        return "TRUSTED_CONTEXT_FAILURE"
    if expected.decision != "ALLOW":
        return "RISK_CONTEXT_INCOMPLETE" if expected.decision == "INCOMPLETE" else "RISK_DENIED"
    sw, bindings, fence = resolve_switches(switches, r)
    if sw != "OK":
        return sw
    q = exact_fraction(r.quantity)
    risk_price = (
        exact_fraction(r.limit_price) if r.limit_price else trusted_price(pre, auth.instrument)
    )
    command_limit_price = exact_fraction(r.limit_price) if r.limit_price else None
    assert q is not None
    vals = (
        new_lease_id(),
        r.command_id,
        auth.command.request_fingerprint_sha256,
        r.order_id,
        r.order_intent_id,
        r.workspace_id,
        r.portfolio_id,
        r.environment,
        r.exchange_account_id,
        auth.instrument.exchange_id,
        r.instrument_id,
        auth.instrument.metadata_version,
        r.execution_route_id,
        r.strategy_instance_id,
        r.source_type,
        r.side,
        r.order_type,
        q,
        command_limit_price,
        r.time_in_force,
        r.expire_at_utc,
        effective.bindings,
        effective.fingerprint_sha256,
        bindings,
        fence,
        pre.projection_fingerprint_sha256,
        post.projection_fingerprint_sha256,
        expected.decision_fingerprint_sha256,
        fact.audit_event_id,
        fact.source_fingerprint_sha256,
        state.state_fingerprint_sha256,
        state.asset_reference,
        state.original_quantity,
        state.remaining_quantity,
        "2026-01-01T00:00:02Z",
        "2026-01-01T00:00:32Z",
    )
    if r.side == "BUY" and risk_price is None:
        return "RISK_CONTEXT_INCOMPLETE"
    lease = ExecutionLease(*vals, fingerprint(vals))
    CORE_ISSUED_LEASES[lease.execution_lease_id] = CoreIssuedExecutionLeaseProjection(
        lease.execution_lease_id,
        lease.lease_fingerprint_sha256,
        lease.command_id,
        lease.order_id,
    )
    return lease


def validate_lease_identity(
    lease: ExecutionLease, auth: PrevalidatedExecutionAuthorityContext
) -> str:
    r = auth.command.request
    q = exact_fraction(r.quantity)
    price = exact_fraction(r.limit_price) if r.limit_price else None
    if not valid_id(
        lease.execution_lease_id, "lease"
    ) or lease.lease_fingerprint_sha256 != fingerprint(lease_semantics(lease)):
        return "TRUSTED_CONTEXT_FAILURE"
    expected = (
        r.command_id,
        auth.command.request_fingerprint_sha256,
        r.order_id,
        r.order_intent_id,
        r.workspace_id,
        r.portfolio_id,
        r.environment,
        r.exchange_account_id,
        auth.instrument.exchange_id,
        r.instrument_id,
        auth.instrument.metadata_version,
        r.execution_route_id,
        r.strategy_instance_id,
        r.source_type,
        r.side,
        r.order_type,
        q,
        price,
        r.time_in_force,
        r.expire_at_utc,
    )
    actual = (
        lease.command_id,
        lease.command_request_fingerprint_sha256,
        lease.order_id,
        lease.order_intent_id,
        lease.workspace_id,
        lease.portfolio_id,
        lease.environment,
        lease.exchange_account_id,
        lease.exchange_id,
        lease.instrument_id,
        lease.instrument_metadata_version,
        lease.execution_route_id,
        lease.strategy_instance_id,
        lease.source_identity,
        lease.side,
        lease.order_type,
        lease.quantity,
        lease.limit_price,
        lease.time_in_force,
        lease.order_expire_at_utc,
    )
    return "OK" if expected == actual else "LEASE_SCOPE_MISMATCH"


def validate_execution_lease_for_dispatch(
    lease: ExecutionLease,
    auth: PrevalidatedExecutionAuthorityContext,
    post: AccountingRiskProjection,
    policy_ctx: PrevalidatedRiskPolicyContext,
    switches: PrevalidatedKillSwitchContext,
    fact: CapitalReservationFact,
    reservation: ReservationState,
    dispatch: CoreDispatchAuthorityState,
    now: str = "2026-01-01T00:00:03Z",
) -> str:
    # Integrity and exact historical economics precede replay lookup.
    integrity = validate_lease_identity(lease, auth)
    if integrity != "OK":
        return integrity
    if CORE_ISSUED_LEASES.get(lease.execution_lease_id) != CoreIssuedExecutionLeaseProjection(
        lease.execution_lease_id,
        lease.lease_fingerprint_sha256,
        lease.command_id,
        lease.order_id,
    ):
        return "LEASE_NOT_ISSUED"
    prior = dispatch.records.get(lease.command_id)
    if prior is not None:
        if (
            prior.command_request_fingerprint_sha256,
            prior.order_id,
            prior.execution_lease_id,
            prior.lease_fingerprint_sha256,
        ) != (
            lease.command_request_fingerprint_sha256,
            lease.order_id,
            lease.execution_lease_id,
            lease.lease_fingerprint_sha256,
        ):
            return "IDEMPOTENCY_CONFLICT"
        if prior.state == "CONSUMED":
            return "REPLAY_SUCCESS"
        if prior.state == "UNKNOWN_RECONCILIATION":
            return "RECONCILIATION_REQUIRED"
        if prior.state != "UNUSED":
            return "TRUSTED_CONTEXT_FAILURE"
    if not all(utc_text(x) for x in (lease.issued_at_utc, lease.expires_at_utc, now)):
        return "LEASE_EXPIRED"
    issued = datetime.fromisoformat(lease.issued_at_utc[:-1] + "+00:00")
    expires = datetime.fromisoformat(lease.expires_at_utc[:-1] + "+00:00")
    current = datetime.fromisoformat(now[:-1] + "+00:00")
    if (
        not (issued <= current <= expires)
        or expires <= issued
        or expires - issued > timedelta(seconds=30)
    ):
        return "LEASE_EXPIRED"
    r = auth.command.request
    authority_status = validate_execution_authority_context(
        auth, require_current_mutable_authority=True
    )
    if authority_status != "OK":
        return authority_status
    if (
        validate_accounting(post, r) != "OK"
        or post.projection_fingerprint_sha256
        != lease.post_reservation_accounting_projection_fingerprint_sha256
    ):
        return "LEASE_STALE"
    state_values = (
        reservation.command_id,
        reservation.order_id,
        reservation.workspace_id,
        reservation.portfolio_id,
        reservation.environment,
        reservation.exchange_account_id,
        reservation.asset_reference,
        reservation.original_quantity,
        reservation.remaining_quantity,
        reservation.source_audit_event_id,
        reservation.source_fingerprint_sha256,
    )
    if (
        validate_fact(fact) != "OK"
        or not core_accepts(
            reservation.accepted_membership_id, reservation.state_fingerprint_sha256
        )
        or reservation.state_fingerprint_sha256 != fingerprint(state_values)
        or reservation.state_fingerprint_sha256 != lease.reservation_state_fingerprint_sha256
        or (
            fact.audit_event_id,
            fact.source_fingerprint_sha256,
            fact.asset_reference,
            fact.quantity,
            fact.command_id,
            fact.order_id,
        )
        != (
            lease.reservation_source_audit_event_id,
            lease.reservation_source_fingerprint_sha256,
            lease.reservation_asset_reference,
            lease.reservation_original_quantity,
            lease.command_id,
            lease.order_id,
        )
        or reservation.source_audit_event_id != fact.audit_event_id
        or reservation.source_fingerprint_sha256 != fact.source_fingerprint_sha256
        or dict(post.reserved_capital).get(reservation.asset_reference, Fraction(-1))
        < reservation.remaining_quantity
    ):
        return "RESERVATION_INVALID"
    accepted_binding = CORE_ACCEPTED_DECISIONS.get(lease.risk_decision_fingerprint_sha256)
    if not isinstance(accepted_binding, CoreAcceptedDecisionBinding):
        return "TRUSTED_CONTEXT_FAILURE"
    accepted_decision = accepted_binding.decision
    if (
        accepted_binding.decision_fingerprint_sha256 != lease.risk_decision_fingerprint_sha256
        or risk_decision_fingerprint(accepted_decision)
        != accepted_decision.decision_fingerprint_sha256
        or accepted_decision.decision_fingerprint_sha256 != lease.risk_decision_fingerprint_sha256
        or accepted_decision.decision != "ALLOW"
        or accepted_decision.command_id != lease.command_id
        or accepted_decision.command_request_fingerprint_sha256
        != lease.command_request_fingerprint_sha256
        or accepted_decision.pre_reservation_accounting_projection_fingerprint_sha256
        != lease.pre_reservation_accounting_projection_fingerprint_sha256
        or accepted_decision.effective_policy_fingerprint_sha256
        != lease.effective_policy_fingerprint_sha256
        or accepted_decision.kill_switch_fence_sha256 != lease.kill_switch_fence_sha256
    ):
        return "TRUSTED_CONTEXT_FAILURE"
    effective = resolve_policy(policy_ctx, r)
    sw, bindings, fence = resolve_switches(switches, r)
    if (
        isinstance(effective, str)
        or sw != "OK"
        or (effective.bindings, effective.fingerprint_sha256, bindings, fence)
        != (
            lease.effective_policy_bindings,
            lease.effective_policy_fingerprint_sha256,
            lease.kill_switch_bindings,
            lease.kill_switch_fence_sha256,
        )
    ):
        return "LEASE_STALE"
    dispatch.records[lease.command_id] = CoreDispatchRecord(
        lease.command_request_fingerprint_sha256,
        lease.order_id,
        lease.execution_lease_id,
        lease.lease_fingerprint_sha256,
        "CONSUMED",
    )
    dispatch.side_effect_count += 1
    return "DISPATCH_AUTHORIZED"


def _scenario(
    r: M07SubmitOrderRequest, policy: RiskPolicyRecord | None, mode: str
) -> tuple[Any, ...]:
    p, i, route, commands = _upstream_fixture(r, mode)
    auth = resolve_execution_authority(r, p, i, route, commands)
    assert isinstance(auth, PrevalidatedExecutionAuthorityContext)
    pre = accounting("PRE", r.environment)
    req = derive_requirement(auth, pre)
    assert isinstance(req, ReservationRequirementProjection)
    fact = reservation_fact(req)
    rs = reservation_state(fact)
    post = accounting("POST", r.environment)
    pol = policy_context(policy or policy_record(environment=r.environment))
    effective = resolve_policy(pol, r)
    assert isinstance(effective, EffectivePolicy)
    switches = switch_context(kill_record(environment=r.environment))
    decision = decision_context(derive_decision(auth, pre, effective, switches, req))
    lease = issue_lease(auth, pre, post, pol, switches, req, fact, rs, decision)
    assert isinstance(lease, ExecutionLease)
    return auth, pre, post, pol, switches, req, fact, rs, decision, lease


def scenario(
    r: M07SubmitOrderRequest | None = None, policy: RiskPolicyRecord | None = None
) -> tuple[Any, ...]:
    return _scenario(request() if r is None else r, policy, "CURRENT")


def future_live_scenario(r: M07SubmitOrderRequest) -> tuple[Any, ...]:
    return _scenario(r, None, "FUTURE_LIVE")


# Machine attestation and schema equality
def test_contract_and_dependencies_exact() -> None:
    assert validate_contract(json.loads(CONTRACT.read_text())) == "VALID"


def test_all_executable_schemas_equal_machine() -> None:
    schemas = thaw(EXPECTED_PROTOCOLS["executable_boundary_schemas"])
    local_dataclasses = {
        name: value
        for name, value in globals().items()
        if isinstance(value, type) and value.__module__ == __name__ and is_dataclass(value)
    }
    assert set(schemas) == set(local_dataclasses)
    for model_name, machine_fields in schemas.items():
        model = local_dataclasses[model_name]
        assert [x.name for x in fields(model)] == machine_fields
    d = thaw(EXPECTED_PROTOCOLS)
    assert list(d["risk_policy_contract"]["semantic_fingerprint_input"]) == [
        x.name for x in fields(RiskPolicyRecord) if x.name != "semantic_fingerprint_sha256"
    ]
    assert [x.name for x in fields(ExecutionLease)] == d["execution_lease_contract"][
        "required_fields"
    ]
    assert [x.name for x in fields(RiskDecision)] == d["risk_decision_contract"]["required_fields"]
    assert [x.name for x in fields(LimitResult)] == d["risk_decision_contract"][
        "limit_result_fields"
    ]


def test_dependency_pointer_attacker_hash_fails() -> None:
    d = json.loads(CONTRACT.read_text())
    dep = next(
        x for x in d["cross_contract_dependencies"] if x["json_pointer"] == "/closed_request_policy"
    )
    up = json.loads((ARCH / dep["contract"]).read_text())
    dep["json_pointer"] = "/idempotency_contract"
    dep["content_fingerprint_sha256"] = fingerprint(up["idempotency_contract"])
    assert validate_contract(d) == "CONTRACT_INCONSISTENT"


def test_exact_m08_fact_schema_and_audit_identity() -> None:
    up = json.loads((ARCH / "ledger_portfolio_capital_and_pnl.json").read_text())
    assert [x.name for x in fields(CapitalReservationFact)] == up[
        "accounting_economic_fact_schema_registry"
    ]["capital_reservation"]["exact_fields"]
    f = scenario()[6]
    assert valid_id(f.audit_event_id, "evt") and f.source_type == "capital_reservation"


# PRE -> requirement -> accepted fact/state -> POST
def test_pre_requirement_decision_post_transition_exact() -> None:
    auth, pre, post, pol, switches, req, fact, state, decision, _ = scenario(
        policy=policy_record("MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION", Fraction(75), USD)
    )
    assert dict(pre.available_capital)[USD] == 100 and dict(pre.reserved_capital)[USD] == 0
    assert req.required_quantity == 20
    assert decision.decision.ordered_limit_results[0].observed_projected_value == 80
    assert dict(post.available_capital)[USD] == 80 and dict(post.reserved_capital)[USD] == 20
    assert validate_transition(pre, post, req, fact, state) == "OK"


def test_no_double_subtraction_at_dispatch() -> None:
    auth, pre, post, pol, switches, req, fact, state, decision, lease = scenario(
        policy=policy_record("MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION", Fraction(75), USD)
    )
    assert dict(post.available_capital)[USD] == 80
    dispatch = CoreDispatchAuthorityState({})
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, state, dispatch
        )
        == "DISPATCH_AUTHORIZED"
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f: replace(f, quantity=Fraction(19)),
        lambda f: replace(f, asset_reference=BTC),
        lambda f: replace(f, exchange_account_id="xacc_018f0000-0000-7000-8000-000000000002"),
        lambda f: replace(f, environment="PAPER"),
    ],
)
def test_mismatched_transition_fails(mutation: Any) -> None:
    _, pre, post, _, _, req, fact, state, _, _ = scenario()
    assert validate_transition(pre, post, req, mutation(fact), state) != "OK"


def test_stale_pre_cannot_dispatch_as_post() -> None:
    auth, pre, _, pol, switches, _, fact, state, _, lease = scenario()
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, pre, pol, switches, fact, state, CoreDispatchAuthorityState({})
        )
        == "LEASE_STALE"
    )


# Exact M0.7 accepted membership and fingerprint
def test_raw_valid_command_cannot_self_enroll() -> None:
    r = request()
    empty = CoreAcceptedCommandProjection((), "m07-accepted")
    assert attest_command(r, empty) == "TRUSTED_CONTEXT_FAILURE"


def test_m07_fingerprint_excludes_only_correlation() -> None:
    r = request()
    changed = replace(r, correlation_id="corr_018f0000-0000-7000-8000-000000000002")
    assert m07_fingerprint(r) == m07_fingerprint(changed)
    assert m07_fingerprint(r) != m07_fingerprint(replace(r, quantity="3"))


@pytest.mark.parametrize(
    "change",
    [
        {"source_type": "BANANA", "strategy_instance_id": None},
        {"time_in_force": "BANANA"},
        {"environment": "BANANA"},
        {"command_id": "cmd-bad", "idempotency_key": "cmd-bad"},
        {"quantity": 2.0},
        {"quantity": True},
        {"quantity": "02"},
        {"expire_at_utc": "not-time", "time_in_force": "GTD"},
        {"order_type": "MARKET", "limit_price": "10"},
        {"time_in_force": "GTD", "expire_at_utc": None},
    ],
)
def test_full_closed_m07_validation_rejects(change: dict[str, Any]) -> None:
    assert not validate_m07(replace(request(), **change))


# Pre-existing upstream authority and LIVE
def test_caller_generated_m04_m05_m06_proofs_rejected() -> None:
    r = request()
    p, i, route, commands = _upstream_fixture(r, "CURRENT")
    bad_p = replace(p, membership_id="caller")
    bad_i = replace(i, membership_id="caller")
    bad_route = replace(route, membership_id="caller")
    assert resolve_execution_authority(r, bad_p, i, route, commands) == "TRUSTED_CONTEXT_FAILURE"
    assert resolve_execution_authority(r, p, bad_i, route, commands) == "TRUSTED_CONTEXT_FAILURE"
    assert resolve_execution_authority(r, p, i, bad_route, commands) == "TRUSTED_CONTEXT_FAILURE"


def test_current_live_denied_fake_magic_denied_future_projection_works() -> None:
    r = replace(request(), environment="LIVE")
    p, i, route, commands = _upstream_fixture(r, "CURRENT")
    assert resolve_execution_authority(r, p, i, route, commands) == "UPSTREAM_AUTHORITY_DENIED"
    fake = replace(
        p,
        authorized_operations=("SUBMIT_ORDER",),
        edition_policy="MAGIC",
        semantic_fingerprint_sha256=fingerprint((p.environment, ("SUBMIT_ORDER",), "MAGIC")),
    )
    assert resolve_execution_authority(r, fake, i, route, commands) == "TRUSTED_CONTEXT_FAILURE"
    assert isinstance(future_live_scenario(r)[-1], ExecutionLease)


def test_no_boolean_or_testnet_live_fallback() -> None:
    source = Path(__file__).read_text()
    forbidden_live = "live_" + "future" + ":bool"
    forbidden_enabled = "enabled" + ":bool"
    assert forbidden_live not in source and forbidden_enabled not in source
    r = replace(request(), environment="LIVE")
    p, i, route, commands = _upstream_fixture(request(), "CURRENT")
    _, _, _, live_commands = _upstream_fixture(r, "CURRENT")
    assert resolve_execution_authority(r, p, i, route, live_commands) == "TRUSTED_CONTEXT_FAILURE"


# Switch exact state
def test_switch_revision_effective_tamper_and_unknown_fail() -> None:
    r = request()
    base = kill_record()
    assert (
        resolve_switches(switch_context(replace(base, source_revision=2)), r)[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert (
        resolve_switches(switch_context(replace(base, effective_at_utc="bad")), r)[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert (
        resolve_switches(switch_context(kill_record(state="UNKNOWN")), r)[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )


def test_switch_generation_reuse_and_rollback_fail() -> None:
    r = request()
    assert (
        resolve_switches(
            switch_context(kill_record(generation=1), kill_record(state="ACTIVE", generation=1)), r
        )[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert (
        resolve_switches(
            switch_context(kill_record(generation=2), kill_record(state="ACTIVE", generation=1)), r
        )[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )


# Valuation/accounting/asset boundaries
def test_valuation_authority_tamper_fails() -> None:
    a = accounting()
    v = a.valuations[0]
    bad = replace(v, source_fingerprint_sha256="0" * 64)
    assert (
        validate_accounting(
            replace(
                a,
                valuations=(bad,),
                projection_fingerprint_sha256=fingerprint(
                    accounting_semantics(replace(a, valuations=(bad,)))
                ),
            ),
            request(),
        )
        != "OK"
    )


def test_testnet_accounting_rejects_live_valuation() -> None:
    live = valuation("LIVE")
    a = accounting(vals=(live,))
    assert derive_gross(a) is None
    assert validate_accounting(a, request()) == "RISK_CONTEXT_INCOMPLETE"


@pytest.mark.parametrize(
    "asset",
    [
        AssetReference("", "USD", "BINANCE:SPOT", "EXACT"),
        AssetReference("USD", "USD", "BINANCE:SPOT", "UNKNOWN"),
        AssetReference("USD", "USD", "BINANCE:SPOT", "AMBIGUOUS"),
    ],
)
def test_malformed_non_authoritative_asset_fails(asset: AssetReference) -> None:
    assert not valid_asset(asset)


def test_same_display_different_namespace_distinct() -> None:
    other = AssetReference("USD", "USD", "OTHER:SPOT", "EXACT")
    assert USD != other and len({USD, other}) == 2


def test_accounting_duplicate_negative_and_wrong_asof_fail() -> None:
    a = accounting()
    dup = replace(a, available_capital=((USD, Fraction(100)), (USD, Fraction(1))))
    dup = replace(dup, projection_fingerprint_sha256=fingerprint(accounting_semantics(dup)))
    assert validate_accounting(dup, request()) != "OK"
    neg = replace(a, available_capital=((USD, Fraction(-1)),))
    neg = replace(neg, projection_fingerprint_sha256=fingerprint(accounting_semantics(neg)))
    assert validate_accounting(neg, request()) != "OK"


# Replay integrity and complete dispatch states
@pytest.mark.parametrize(
    "field,value",
    [
        ("quantity", Fraction(3)),
        ("execution_route_id", "xroute_018f0000-0000-7000-8000-000000000002"),
        ("expires_at_utc", "2026-01-01T00:01:00Z"),
    ],
)
def test_mutated_consumed_lease_old_fingerprint_never_replays(field: str, value: Any) -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    dispatch = CoreDispatchAuthorityState({})
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, state, dispatch
        )
        == "DISPATCH_AUTHORIZED"
    )
    assert (
        validate_execution_lease_for_dispatch(
            replace(lease, **{field: value}), auth, post, pol, switches, fact, state, dispatch
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert dispatch.side_effect_count == 1


def test_exact_consumed_replay_without_current_reauthorization() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    dispatch = CoreDispatchAuthorityState({})
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, state, dispatch
        )
        == "DISPATCH_AUTHORIZED"
    )
    deny = policy_context(replace(policy_record(), action="DENY"))
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, deny, switches, fact, state, dispatch
        )
        == "REPLAY_SUCCESS"
    )
    assert dispatch.side_effect_count == 1


def test_unknown_reconciliation_never_resubmits() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    record = CoreDispatchRecord(
        lease.command_request_fingerprint_sha256,
        lease.order_id,
        lease.execution_lease_id,
        lease.lease_fingerprint_sha256,
        "UNKNOWN_RECONCILIATION",
    )
    dispatch = CoreDispatchAuthorityState({lease.command_id: record})
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, state, dispatch
        )
        == "RECONCILIATION_REQUIRED"
    )
    assert dispatch.side_effect_count == 0


def test_coordinated_self_hash_forgery_cannot_enroll() -> None:
    r = request()
    p, i, route, commands = _upstream_fixture(r, "CURRENT")
    assert (
        resolve_execution_authority(r, replace(p, membership_id="caller"), i, route, commands)
        == "TRUSTED_CONTEXT_FAILURE"
    )
    a = replace(accounting(), accepted_membership_id="caller")
    assert validate_accounting(a, r) == "TRUSTED_CONTEXT_FAILURE"
    ctx = policy_context(policy_record())
    assert (
        resolve_policy(
            replace(
                ctx,
                membership_id="caller",
                context_fingerprint_sha256=fingerprint((ctx.records, ctx.current, "caller")),
            ),
            r,
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )
    s = switch_context(kill_record())
    assert (
        resolve_switches(
            replace(
                s,
                membership_id="caller",
                context_fingerprint_sha256=fingerprint((s.history, "caller")),
            ),
            r,
        )[0]
        == "TRUSTED_CONTEXT_FAILURE"
    )


# Full seven-scope hierarchy and executable DENY semantics.
def scoped_policy(
    scope_type: str,
    scope_id: str,
    threshold: Fraction,
    serial: int,
    action: str = "ALLOW",
    limit: str = "MAX_ORDER_NOTIONAL",
) -> RiskPolicyRecord:
    return policy_record(
        limit,
        threshold,
        USD,
        scope_type=scope_type,
        scope_id=scope_id,
        action=action,
        policy_id=f"rpol_018f0000-0000-7000-8000-{serial:012x}",
    )


@pytest.mark.parametrize(
    "scope_type,scope_id",
    [
        ("PRODUCT_SYSTEM", "product"),
        ("WORKSPACE", request().workspace_id),
        ("PORTFOLIO", request().portfolio_id),
        ("EXCHANGE_ACCOUNT", request().exchange_account_id),
        ("STRATEGY_INSTANCE", request().strategy_instance_id),
        ("INSTRUMENT", request().instrument_id),
        ("EXECUTION_ROUTE", request().execution_route_id),
    ],
)
def test_every_policy_scope_applies(scope_type: str, scope_id: str | None) -> None:
    assert scope_id is not None
    p = scoped_policy(scope_type, scope_id, Fraction(17), SCOPE_ORDER.index(scope_type) + 10)
    result = resolve_policy(policy_context(p), request())
    assert isinstance(result, EffectivePolicy) and result.limits[0][1] == 17


def test_strictest_maximum_and_minimum_compose() -> None:
    system = scoped_policy("PRODUCT_SYSTEM", "product", Fraction(15), 30)
    workspace = scoped_policy("WORKSPACE", request().workspace_id, Fraction(20), 31)
    result = resolve_policy(policy_context(system, workspace), request())
    assert isinstance(result, EffectivePolicy) and result.limits[0][1] == 15
    minimum_a = scoped_policy(
        "PRODUCT_SYSTEM",
        "product",
        Fraction(5),
        32,
        limit="MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION",
    )
    minimum_b = scoped_policy(
        "WORKSPACE",
        request().workspace_id,
        Fraction(8),
        33,
        limit="MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION",
    )
    result = resolve_policy(policy_context(minimum_a, minimum_b), request())
    assert isinstance(result, EffectivePolicy) and result.limits[0][1] == 8


def test_unrelated_policy_and_environment_are_ignored() -> None:
    unrelated = scoped_policy(
        "INSTRUMENT", "instr_018f0000-0000-7000-8000-000000000099", Fraction(1), 34
    )
    paper = replace(
        scoped_policy("WORKSPACE", request().workspace_id, Fraction(1), 35), environment="PAPER"
    )
    vals = (
        paper.risk_policy_id,
        paper.revision,
        paper.environment,
        paper.scope_type,
        paper.scope_id,
        paper.action,
        paper.limits,
    )
    paper = replace(paper, semantic_fingerprint_sha256=fingerprint(vals))
    assert resolve_policy(policy_context(unrelated, paper), request()) == "RISK_CONTEXT_INCOMPLETE"


def test_applicable_deny_blocks_issuance_unrelated_deny_does_not() -> None:
    auth, pre, post, _, switches, req, fact, state, decision, _ = scenario()
    deny = scoped_policy("PRODUCT_SYSTEM", "product", Fraction(100), 36, "DENY")
    assert (
        issue_lease(auth, pre, post, policy_context(deny), switches, req, fact, state, decision)
        == "RISK_DENIED"
    )
    unrelated = scoped_policy(
        "INSTRUMENT", "instr_018f0000-0000-7000-8000-000000000099", Fraction(1), 37, "DENY"
    )
    allow = scoped_policy("WORKSPACE", request().workspace_id, Fraction(20), 38)
    effective = resolve_policy(policy_context(unrelated, allow), request())
    assert isinstance(effective, EffectivePolicy)


def test_current_revision_selects_latest_and_rollback_conflicts() -> None:
    old = policy_record(threshold=Fraction(1), revision=1)
    new = policy_record(threshold=Fraction(20), revision=2)
    key = (old.risk_policy_id, old.environment, old.scope_type, old.scope_id)
    result = resolve_policy(policy_context(old, new, current=((*key, 2),)), request())
    assert isinstance(result, EffectivePolicy) and result.limits[0][1] == 20
    assert (
        resolve_policy(policy_context(old, new, current=((*key, 1),)), request())
        == "POLICY_CONFLICT"
    )


def test_zero_policy_fails_closed() -> None:
    assert resolve_policy(policy_context(), request()) == "RISK_CONTEXT_INCOMPLETE"


@pytest.mark.parametrize(
    "scope_type,scope_id",
    [
        ("PRODUCT_SYSTEM", "product"),
        ("WORKSPACE", request().workspace_id),
        ("PORTFOLIO", request().portfolio_id),
        ("EXCHANGE_ACCOUNT", request().exchange_account_id),
        ("STRATEGY_INSTANCE", request().strategy_instance_id),
        ("INSTRUMENT", request().instrument_id),
        ("EXECUTION_ROUTE", request().execution_route_id),
    ],
)
def test_every_switch_scope_active_blocks(scope_type: str, scope_id: str | None) -> None:
    assert scope_id is not None
    ctx = switch_context(kill_record(state="ACTIVE", scope_type=scope_type, scope_id=scope_id))
    assert resolve_switches(ctx, request())[0] == "KILL_SWITCH_ACTIVE"


def test_child_inactive_cannot_override_system_active() -> None:
    ctx = switch_context(
        kill_record(state="ACTIVE", scope_type="PRODUCT_SYSTEM", scope_id="product"),
        kill_record(state="INACTIVE", scope_type="WORKSPACE", scope_id=request().workspace_id),
    )
    assert resolve_switches(ctx, request())[0] == "KILL_SWITCH_ACTIVE"


@pytest.mark.parametrize(
    "outcome",
    ["DRIFT", "MISSING_INTERNAL_FACT", "MISSING_EXTERNAL_FACT", "UNMAPPED_ASSET", "UNSUPPORTED"],
)
def test_reconciliation_failures_block_lease(outcome: str) -> None:
    auth, pre, post, pol, switches, req, fact, state, decision, _ = scenario()
    bad = accounting("PRE", reconciliation=(outcome,))
    assert (
        issue_lease(auth, bad, post, pol, switches, req, fact, state, decision)
        == "RISK_CONTEXT_INCOMPLETE"
    )


def test_manual_self_minted_lease_not_issued() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    clone = replace(
        lease,
        execution_lease_id="lease_018f0000-0000-7000-8000-000000999999",
        lease_fingerprint_sha256="",
    )
    clone = replace(clone, lease_fingerprint_sha256=fingerprint(lease_semantics(clone)))
    assert (
        validate_execution_lease_for_dispatch(
            clone, auth, post, pol, switches, fact, state, CoreDispatchAuthorityState({})
        )
        == "LEASE_NOT_ISSUED"
    )


def test_dispatch_time_window_and_duration_enforced() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    assert (
        validate_execution_lease_for_dispatch(
            lease,
            auth,
            post,
            pol,
            switches,
            fact,
            state,
            CoreDispatchAuthorityState({}),
            "2026-01-01T00:00:01Z",
        )
        == "LEASE_EXPIRED"
    )
    forged = replace(lease, expires_at_utc="2026-01-01T00:01:02Z", lease_fingerprint_sha256="")
    forged = replace(forged, lease_fingerprint_sha256=fingerprint(lease_semantics(forged)))
    CORE_ISSUED_LEASES[forged.execution_lease_id] = CoreIssuedExecutionLeaseProjection(
        forged.execution_lease_id,
        forged.lease_fingerprint_sha256,
        forged.command_id,
        forged.order_id,
    )
    assert (
        validate_execution_lease_for_dispatch(
            forged, auth, post, pol, switches, fact, state, CoreDispatchAuthorityState({})
        )
        == "LEASE_EXPIRED"
    )


def test_market_end_to_end_preserves_null_limit_price() -> None:
    market = replace(request(), order_type="MARKET", limit_price=None)
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario(market)
    assert lease.limit_price is None
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, state, CoreDispatchAuthorityState({})
        )
        == "DISPATCH_AUTHORIZED"
    )


def test_coordinated_policy_content_rewrite_old_membership_fails() -> None:
    ctx = policy_context(policy_record())
    changed = replace(ctx.records[0], limits=(("MAX_ORDER_NOTIONAL", Fraction(999), USD),))
    vals = (
        changed.risk_policy_id,
        changed.revision,
        changed.environment,
        changed.scope_type,
        changed.scope_id,
        changed.action,
        changed.limits,
    )
    changed = replace(changed, semantic_fingerprint_sha256=fingerprint(vals))
    forged = replace(
        ctx,
        records=(changed,),
        context_fingerprint_sha256=fingerprint(((changed,), ctx.current, ctx.membership_id)),
    )
    assert resolve_policy(forged, request()) == "TRUSTED_CONTEXT_FAILURE"


def test_coordinated_switch_content_rewrite_old_membership_fails() -> None:
    ctx = switch_context(kill_record(state="ACTIVE"))
    changed = kill_record(state="INACTIVE", generation=2)
    forged = replace(
        ctx,
        history=(changed,),
        context_fingerprint_sha256=fingerprint(((changed,), ctx.membership_id)),
    )
    assert resolve_switches(forged, request())[0] == "TRUSTED_CONTEXT_FAILURE"


def test_coordinated_upstream_rewrites_old_memberships_fail() -> None:
    r = request()
    p, i, route, commands = _upstream_fixture(r, "CURRENT")
    pvals = (p.environment, ("CANCEL_ORDER",), p.edition_policy)
    bad_p = replace(
        p, authorized_operations=("CANCEL_ORDER",), semantic_fingerprint_sha256=fingerprint(pvals)
    )
    assert resolve_execution_authority(r, bad_p, i, route, commands) == "TRUSTED_CONTEXT_FAILURE"
    ivals = (
        i.workspace_id,
        i.portfolio_id,
        i.environment,
        i.exchange_account_id,
        i.exchange_id,
        i.instrument_id,
        8,
        i.base_asset_reference,
        i.quote_asset_reference,
    )
    bad_i = replace(i, metadata_version=8, semantic_fingerprint_sha256=fingerprint(ivals))
    assert resolve_execution_authority(r, p, bad_i, route, commands) == "TRUSTED_CONTEXT_FAILURE"
    rvals = (
        route.environment,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
        "POLICY_BLOCKED",
    )
    bad_route = replace(
        route, readiness="POLICY_BLOCKED", semantic_fingerprint_sha256=fingerprint(rvals)
    )
    assert resolve_execution_authority(r, p, i, bad_route, commands) == "TRUSTED_CONTEXT_FAILURE"
    changed = replace(r, quantity="3")
    bad_commands = replace(commands, entries=((r.command_id, m07_fingerprint(changed)),))
    assert attest_command(changed, bad_commands) == "TRUSTED_CONTEXT_FAILURE"


def test_coordinated_valuation_rewrite_old_acceptance_fails() -> None:
    a = accounting()
    v = a.valuations[0]
    source = (
        v.source_id,
        a.environment,
        v.subject_reference,
        v.valuation_unit,
        Fraction(99),
        v.observed_at_utc,
        v.effective_at_utc,
        v.as_of_utc,
        v.stale_after_utc,
    )
    sfp = fingerprint(source)
    vals = (
        v.subject_reference,
        v.valuation_unit,
        Fraction(99),
        v.source_id,
        v.observed_at_utc,
        v.effective_at_utc,
        v.as_of_utc,
        v.stale_after_utc,
        sfp,
        v.accepted_authority_fingerprint_sha256,
    )
    forged = replace(
        v,
        rate=Fraction(99),
        source_fingerprint_sha256=sfp,
        context_fingerprint_sha256=fingerprint(vals),
    )
    bad = replace(a, valuations=(forged,))
    bad = replace(bad, projection_fingerprint_sha256=fingerprint(accounting_semantics(bad)))
    assert validate_accounting(bad, request()) == "TRUSTED_CONTEXT_FAILURE"


def test_current_readiness_denial_blocks_unused_lease() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    old = auth.route
    vals = (
        old.environment,
        old.execution_route_id,
        old.strategy_instance_id,
        old.source_type,
        "POLICY_BLOCKED",
    )
    blocked = replace(
        old, readiness="POLICY_BLOCKED", semantic_fingerprint_sha256=fingerprint(vals)
    )
    bad_auth = replace(auth, route=blocked)
    assert (
        validate_execution_lease_for_dispatch(
            lease, bad_auth, post, pol, switches, fact, state, CoreDispatchAuthorityState({})
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )


def test_policy_requires_current_designation_for_every_history_identity() -> None:
    workspace = policy_record()
    system = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000002",
        scope_type="PRODUCT_SYSTEM",
        scope_id="product",
        action="DENY",
    )
    current = (
        (
            workspace.risk_policy_id,
            workspace.environment,
            workspace.scope_type,
            workspace.scope_id,
            workspace.revision,
        ),
    )
    assert resolve_policy(policy_context(workspace, system, current=current), request()) == (
        "POLICY_CONFLICT"
    )


def test_malformed_switch_values_fail_closed_without_exception() -> None:
    for change in (
        {"source_revision": "1"},
        {"generation": "1"},
        {"effective_at_utc": 1},
        {"scope_id": None},
    ):
        record = replace(kill_record(), **change)
        assert resolve_switches(switch_context(record), request())[0] == ("TRUSTED_CONTEXT_FAILURE")


def test_nominal_forged_execution_authority_cannot_issue() -> None:
    auth, pre, post, policy, switches, req, fact, state, decision, _ = scenario()
    forged = replace(auth, context_fingerprint_sha256="0" * 64)
    before = dict(CORE_ISSUED_LEASES)
    assert issue_lease(forged, pre, post, policy, switches, req, fact, state, decision) == (
        "TRUSTED_CONTEXT_FAILURE"
    )
    assert CORE_ISSUED_LEASES == before


def test_dispatch_rejects_mutated_reservation_state_and_unrelated_fact() -> None:
    auth, _, post, pol, switches, _, fact, state, _, lease = scenario()
    mutated = replace(state, remaining_quantity=Fraction(19))
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, fact, mutated, CoreDispatchAuthorityState({})
        )
        == "RESERVATION_INVALID"
    )
    derived = derive_requirement(auth, accounting("PRE"))
    assert isinstance(derived, ReservationRequirementProjection)
    other_req = replace(
        derived,
        order_id="ord_018f0000-0000-7000-8000-000000000099",
    )
    other = reservation_fact(other_req)
    assert (
        validate_execution_lease_for_dispatch(
            lease, auth, post, pol, switches, other, state, CoreDispatchAuthorityState({})
        )
        == "RESERVATION_INVALID"
    )


def test_under_reserved_buy_chain_cannot_issue_or_enter_core_registry() -> None:
    auth, pre, _, policy, switches, req, _, _, decision, _ = scenario()
    req_values = (
        req.command_id,
        req.order_id,
        req.workspace_id,
        req.portfolio_id,
        req.environment,
        req.exchange_account_id,
        req.asset_reference,
        Fraction(1),
    )
    forged_req = replace(
        req,
        required_quantity=Fraction(1),
        derivation_fingerprint_sha256=fingerprint(req_values),
    )
    forged_fact = reservation_fact(forged_req)
    forged_state = reservation_state(forged_fact)
    forged_post = accounting("POST", available=Fraction(99), reserved=Fraction(1))
    assert validate_transition(pre, forged_post, forged_req, forged_fact, forged_state) == "OK"
    before = dict(CORE_ISSUED_LEASES)
    assert (
        issue_lease(
            auth,
            pre,
            forged_post,
            policy,
            switches,
            forged_req,
            forged_fact,
            forged_state,
            decision,
        )
        == "RESERVATION_INVALID"
    )
    assert CORE_ISSUED_LEASES == before


def test_accepted_ready_route_history_is_not_current_after_blocked_designation() -> None:
    auth, _, post, policy, switches, _, fact, state, _, lease = scenario()
    route = auth.route
    blocked_values = (
        route.environment,
        route.execution_route_id,
        route.strategy_instance_id,
        route.source_type,
        "POLICY_BLOCKED",
    )
    blocked_fp = fingerprint(blocked_values)
    blocked_membership = bind_core_content(f"m06:{blocked_fp[:16]}", blocked_fp)
    blocked = replace(
        route,
        readiness="POLICY_BLOCKED",
        semantic_fingerprint_sha256=blocked_fp,
        membership_id=blocked_membership,
    )
    assert core_accepts(route.membership_id, route.semantic_fingerprint_sha256)
    assert core_accepts(blocked.membership_id, blocked.semantic_fingerprint_sha256)
    key = (route.environment, route.execution_route_id)
    old_current = CORE_CURRENT_ROUTE[key]
    CORE_CURRENT_ROUTE[key] = CoreCurrentRouteDesignation(*key, blocked.membership_id)
    dispatch = CoreDispatchAuthorityState({})
    try:
        assert (
            validate_execution_lease_for_dispatch(
                lease, auth, post, policy, switches, fact, state, dispatch
            )
            == "UPSTREAM_AUTHORITY_DENIED"
        )
        assert dispatch.side_effect_count == 0
    finally:
        CORE_CURRENT_ROUTE[key] = old_current


def test_accepted_product_history_is_not_current_after_revocation() -> None:
    auth, _, post, policy, switches, _, fact, state, _, lease = scenario()
    product = auth.product
    disabled_values = (product.environment, (), "CURRENT_EDITION_DISABLED")
    disabled_fp = fingerprint(disabled_values)
    disabled_membership = bind_core_content(f"m04:{disabled_fp[:16]}", disabled_fp)
    disabled = ProductCapabilitiesProjection(
        product.environment, (), "CURRENT_EDITION_DISABLED", disabled_fp, disabled_membership
    )
    assert core_accepts(product.membership_id, product.semantic_fingerprint_sha256)
    assert core_accepts(disabled.membership_id, disabled.semantic_fingerprint_sha256)
    old_current = CORE_CURRENT_PRODUCT[product.environment]
    CORE_CURRENT_PRODUCT[product.environment] = CoreCurrentProductDesignation(
        product.environment, disabled.membership_id
    )
    dispatch = CoreDispatchAuthorityState({})
    try:
        assert (
            validate_execution_lease_for_dispatch(
                lease, auth, post, policy, switches, fact, state, dispatch
            )
            == "UPSTREAM_AUTHORITY_DENIED"
        )
        assert dispatch.side_effect_count == 0
    finally:
        CORE_CURRENT_PRODUCT[product.environment] = old_current


@pytest.mark.parametrize("target", ["route", "product", "command"])
def test_unused_dispatch_recomputes_upstream_content_fingerprints(target: str) -> None:
    auth, _, post, policy, switches, _, fact, state, _, lease = scenario()
    if target == "route":
        bad_auth = replace(auth, route=replace(auth.route, readiness="POLICY_BLOCKED"))
    elif target == "product":
        bad_auth = replace(auth, product=replace(auth.product, authorized_operations=()))
    else:
        bad_request = replace(
            auth.command.request,
            authority_context_id="authctx_018f0000-0000-7000-8000-000000000099",
        )
        bad_auth = replace(auth, command=replace(auth.command, request=bad_request))
    dispatch = CoreDispatchAuthorityState({})
    assert (
        validate_execution_lease_for_dispatch(
            lease, bad_auth, post, policy, switches, fact, state, dispatch
        )
        == "TRUSTED_CONTEXT_FAILURE"
    )
    assert dispatch.side_effect_count == 0


def test_every_accepted_allow_policy_identity_requires_current_designation() -> None:
    first = policy_record()
    second = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000099",
        scope_type="INSTRUMENT",
        scope_id=request().instrument_id,
    )
    current = ((first.risk_policy_id, first.environment, first.scope_type, first.scope_id, 1),)
    assert resolve_policy(policy_context(first, second, current=current), request()) == (
        "POLICY_CONFLICT"
    )


def test_dispatch_recomputes_exact_accepted_historical_decision() -> None:
    auth, _, post, policy, switches, _, fact, state, _, lease = scenario()
    key = lease.risk_decision_fingerprint_sha256
    original = CORE_ACCEPTED_DECISIONS[key]
    assert isinstance(original, CoreAcceptedDecisionBinding)
    tampered = replace(original.decision, evaluated_at_utc="2026-01-01T00:00:01Z")
    CORE_ACCEPTED_DECISIONS[key] = CoreAcceptedDecisionBinding(key, tampered)
    dispatch = CoreDispatchAuthorityState({})
    try:
        assert (
            validate_execution_lease_for_dispatch(
                lease, auth, post, policy, switches, fact, state, dispatch
            )
            == "TRUSTED_CONTEXT_FAILURE"
        )
        assert dispatch.side_effect_count == 0
    finally:
        CORE_ACCEPTED_DECISIONS[key] = original


@pytest.mark.parametrize("reverse", [False, True])
def test_same_scope_revision_distinct_policy_semantics_conflict_in_every_order(
    reverse: bool,
) -> None:
    first = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000101",
        threshold=Fraction(20),
    )
    second = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000102",
        threshold=Fraction(10),
    )
    ordered = (second, first) if reverse else (first, second)
    assert resolve_policy(policy_context(*ordered), request()) == "POLICY_CONFLICT"


def test_same_scope_different_revisions_are_container_order_independent() -> None:
    first = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000103",
        revision=1,
        threshold=Fraction(20),
    )
    second = policy_record(
        policy_id="rpol_018f0000-0000-7000-8000-000000000104",
        revision=2,
        threshold=Fraction(10),
    )
    forward = resolve_policy(policy_context(first, second), request())
    reverse = resolve_policy(policy_context(second, first), request())
    assert isinstance(forward, EffectivePolicy)
    assert isinstance(reverse, EffectivePolicy)
    assert forward.bindings == reverse.bindings
    assert forward.limits == reverse.limits
    assert forward.fingerprint_sha256 == reverse.fingerprint_sha256
