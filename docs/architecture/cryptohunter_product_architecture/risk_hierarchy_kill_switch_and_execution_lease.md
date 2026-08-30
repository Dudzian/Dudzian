# M0.9 — Risk hierarchy, kill switch and ExecutionLease

**Status: closed.** Ten dokument jest kontraktem architektury, nie implementacją runtime'u. JSON jest
niezaufaną atestacją do czasu niezależnej walidacji semantyki i fingerprintów upstream.

## Authority i hierarchia

Jedynym właścicielem mutable trading/risk authority jest CoreHost. UI i Tray są klientami. Raw policy,
jej własny SHA-256, `risk_ok`, `risk_passed`, `approved`, venue, endpoint i credential nie przyznają
authority. `PrevalidatedRiskPolicyContext` jest authority dopiero gdy komplet record/history content,
semantic fingerprints (z pól `risk_policy_id`, revision, environment, scope type/ID, action i `limits`),
current designations, context fingerprint oraz `CoreAcceptedContentBinding` niezależnie się walidują.
Sama nazwa nominalnego typu niczego nie przyznaje; sposób uwierzytelnienia zmiany pozostaje M0.10.

Każda policy i decyzja wiąże dokładnie jedno z `PAPER`, `TESTNET`, `LIVE`. Stała kolejność scope to:

1. `PRODUCT_SYSTEM`;
2. `WORKSPACE`;
3. `PORTFOLIO`;
4. `EXCHANGE_ACCOUNT`;
5. kwalifikatory `STRATEGY_INSTANCE`, `INSTRUMENT`, `EXECUTION_ROUTE`.

Workspace jest dzieckiem product/system, Portfolio dzieckiem Workspace, a ExchangeAccount dzieckiem
Portfolio. Trzy ostatnie scope są równorzędnymi, nieporównywalnymi kwalifikatorami należącymi do
Workspace; są applicable tylko dla exact ID komendy. Nie tworzymy fikcyjnego drzewa organizacyjnego.
Inne środowisko lub ID nie dziedziczy policy. Selekcja i sortowanie używają powyższej kolejności,
`scope_id` i revision, nigdy kolejności słownika.

Effective maximum to minimum wszystkich applicable thresholds; effective minimum to maksimum.
Applicable DENY dominuje. Dwie policy dla tego samego exact scope/revision o różnych semantic
fingerprints są `POLICY_CONFLICT`. Core accepted history wskazuje dokładnie jedną current revision dla
każdej policy identity/scope/environment; stare revisions nie uczestniczą w kompozycji. Zero applicable
current policies oraz brak wymaganego inputu daje `INCOMPLETE`, nigdy allow. Composite policy fence
haszuje cały uporządkowany effective set, a nie jedną policy.

## Current-SPOT limits i exact arithmetic

Wszystkie liczby są canonical decimal zamienionym na exact rational/Fraction; float jest zabroniony.
Równość z maksimum/minimum przechodzi. Zamknięty supported registry zawiera:

* `MAX_ORDER_QUANTITY = abs(quantity)`;
* `MAX_ORDER_NOTIONAL = abs(quantity) * conservative_risk_price`;
* `MAX_POST_TRADE_POSITION_QUANTITY = abs(current_signed_inventory + signed_quantity)`;
* `MAX_POST_TRADE_POSITION_NOTIONAL` — poprzedni wynik razy trusted risk price;
* `MAX_GROSS_EXPOSURE = current_gross - current_instrument_abs_exposure + projected_abs_exposure`;
* `MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION = accepted_PRE_available[exact spend AssetReference] -
  derived_ReservationRequirementProjection.required_quantity` — dokładnie raz; POST ReservationState
  nie jest ponownie odejmowany.

BUY zwiększa, SELL zmniejsza signed base inventory. BUY rezerwuje jeden exact quote AssetReference i
quantity, SELL jeden exact base AssetReference i quantity. Jest to dokładna granica pojedynczego
`capital_reservation` fact/ReservationState M0.8. Fee może być uwzględnione wyłącznie w tej samej
spend-asset quantity; separate third-asset pre-trade reservation jest obecnie
`UNSUPPORTED_RISK_SEMANTICS`, a nie fikcyjnym agregatem. LIMIT używa canonical `limit_price`. MARKET zawsze wymaga
accepted M0.6/M0.8 valuation context z exact instrument/environment/as-of/expiry/fingerprint. Bez niego
nie można ustalić dispatch reservation economics, nawet jeśli enabled limit sprawdza tylko quantity;
wynik jest `INCOMPLETE`. Nie istnieje price-zero fallback. UI price, losowy ticker, adapter response, implicit last, zero i stablecoin
parity są zabronione.

`LEVERAGE`, `MARGIN`, `DERIVATIVES_GREEKS`, `LIQUIDATION`, `BORROW`, `FUNDING`, `OPTION_EXERCISE`,
`TAX`, `CROSS_MARGIN`, `DAILY_LOSS_RESET` i `NAV_CONCENTRATION` są jawnie unsupported i fail closed.

Risk używa immutable `AccountingRiskProjection` z exact workspace, Portfolio, environment,
accounts, as-of i fingerprint. DRIFT, missing facts, unmapped asset, unsupported semantics oraz
missing/stale/unsupported wymaganej valuation dają `RISK_CONTEXT_INCOMPLETE`. Kontrolowane są tylko
materialne inputy enabled limitów; irrelevant zero holding nie blokuje. Projekcja post-state jest
czysta: nie zapisuje LedgerEntry, reservation ani żadnego M0.8 state. Asset buckets wszędzie używają
pełnego M0.5 `AssetReference` (`venue_asset_code`, display code, namespace, mapping status), nigdy
display-code aliasu. Gross exposure jest wyprowadzane z exact inventory i zgodnych valuation contexts
w jednym reporting AssetReference i as-of; nie jest caller scalar.

Accounting ma dwa różne, accepted M0.8 snapshots. `PRE` poprzedza reservation i jest jedynym inputem
RiskDecision. Pure `ReservationRequirementProjection` powstaje z accepted M0.7 request, historical M0.5
Instrument i trusted valuation. `MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION` odejmuje requirement dokładnie
raz od PRE available. Po ALLOW accepted M0.8 `capital_reservation` fact przenosi tę samą quantity z
OWNED_AVAILABLE do OWNED_RESERVED; odrębny `POST` snapshot musi dowieść obu zmian i jest dispatch
authority. Lease wiąże osobno PRE fingerprint decyzji i current POST fingerprint dispatchu. POST nie
jest ponownie pomniejszany o ReservationState.

## Kill switch i fences

Switch ma wyłącznie `INACTIVE`/`ACTIVE` oraz exact scope, environment, source revision, effective time,
accepted Core projection/seal, monotonic positive generation i recomputed record fingerprint. Unknown
state, generation reuse albo rollback fail closed. Applicable ACTIVE na ancestor lub
kwalifikatorze blokuje każde nowe execution-authoritative lease; child INACTIVE nie może go nadpisać.

Każda zaakceptowana zmiana stanu zwiększa generation. Lease przechowuje fingerprint uporządkowanego
zbioru applicable switch records/generations. Validator przed side effect ponownie rozwiązuje switch:
ACTIVE blokuje, a nierówność fence daje stale. ACTIVE→INACTIVE również ma nowszą generation, więc stare
lease nie zmartwychwstaje. Analogicznie każda policy revision/fingerprint i authority-relevant M0.8
projection fingerprint wymagają świeżej oceny i lease.

## RiskDecision, lease, reservation i idempotency

Immutable, non-durable `RiskDecision` wiąże command/order, scope/environment, policy, accounting input,
czas, ordered limit results, switch result i deterministic fingerprint. Agregacja: dowolny FAIL daje
`DENY`; w przeciwnym razie dowolny wymagany incomplete daje `INCOMPLETE`; tylko komplet daje `ALLOW`.
Issuance rederivuje pełną decyzję z PRE inputs. Dispatch nie interpretuje ponownie historycznego PRE:
odczytuje exact accepted `RiskDecision`, recomputuje fingerprint ze wszystkich jego pól, wymaga `ALLOW`
i bindingu z lease, a następnie osobno waliduje current execution authority, policy/switch fences, POST
accounting, reservation i lifetime.

Jedynym modelem accepted content jest Core-owned `CoreAcceptedContentBinding`:
`membership_id -> content_fingerprint_sha256`. Accepted history nie jest current mutable authority.
`CoreCurrentProductDesignation` wskazuje current accepted ProductCapabilities per environment, a
`CoreCurrentRouteDesignation` wskazuje current accepted route/readiness per environment i route ID.
Stary accepted READY pozostaje audytowalny, ale po designation nowszego BLOCKED nie może dispatchować.
Każdy `Prevalidated...` context jest authority tylko po niezależnej walidacji całego contentu,
fingerprintów, accepted membership, exact bindings oraz current designations; nazwa typu niczego nie nadaje.

M0.2 już definiuje durable `ExecutionLease` (`execution_lease_id`, prefix `lease`, UUIDv7), więc M0.9
używa tej identity zamiast tworzyć nową; każde issuance otrzymuje odrębny canonical UUIDv7. Lease wiąże exact command ID/request fingerprint, Order i
OrderIntent, Workspace/Portfolio/environment, ExchangeAccount/exchange, Instrument/metadata version,
ExecutionRoute, StrategyInstance/source, side/type/quantity/limit price/TIF/order expiry, RiskPolicy,
cały ordered effective-policy set/fence, cały ordered kill-switch set/fence, accounting/RiskDecision
fingerprints, accepted reservation authority, issuance/expiry i recomputed własny fingerprint.

Lifetime wynosi najwyżej 30 sekund, `expires_at > issued_at`, a valid interval jest inclusive. Lease nie
jest reusable. Jedyna kolejność to: pure risk evaluation → accepted ALLOW → provisional materialization
→ exact order-bound M0.8 reservation → final immutable ExecutionLease/fingerprint → M0.7 atomic idempotency
reservation and accepted plan → jeden side effect. Żadne z wcześniejszych działań samo nie dispatchuje.
Brak/obca/niewystarczająca reservation blokuje dispatch.

M0.8 nie definiuje durable Reservation ani `reservation_id`. Binding używa jednego accepted
`capital_reservation` AuditEvent-derived accounting fact: source audit-event identity/fingerprint,
command/Order/scope/account, jednego exact AssetReference, original i remaining quantity oraz current
reservation accounting-state fingerprint i Core membership seal. Przed reservation może istnieć tylko
non-authoritative `LeaseDraft`; nie jest ExecutionLease ani dispatch authority.

Immutable fact zachowuje cały M0.8 schema: canonical `evt_` AuditEvent, `source_type`, scope/environment,
effective time, provenance, source fingerprint, account, exact AssetReference, quantity, Order i command.
Rebuildable ReservationState jest osobnym projection wiążącym source fact oraz remaining quantity; nie
zastępuje immutable fact.

M0.7 pozostaje authority command idempotency. Core-owned `CoreDispatchAuthorityState`, a nie pole
`lease.consumed`, atomowo przełącza exact command/request/lease/Order z UNUSED na CONSUMED przed
zwróceniem dispatch authority. Ten sam command/fingerprint zwraca immutable replay bez
drugiego side effect; ten sam command ID z inną ekonomią jest conflict; unknown dispatch jest
reconciled, nigdy resubmitted. M0.9 nie tworzy równoległego dedupe systemu.

Accepted command authority pochodzi z istniejącego wcześniej `CoreAcceptedCommandProjection`, który
mapuje command ID na canonical M0.7 business fingerprint. Attestation waliduje dokładny closed
SUBMIT_ORDER schema i constraints, haszuje canonical normalized request bez jedynego excluded pola
`correlation_id`, a następnie wymaga membership. Podobnie M0.4 ProductCapabilities, M0.5 historical
Instrument/account i M0.6 route/readiness są pre-existing projections; command ani self-hash ich nie
tworzy.

## Environments, LIVE i granice

Policy, switch, decision, accounting, reservation i lease muszą mieć jedno identyczne środowisko. Nie
ma PAPER/TESTNET/LIVE substitution ani TESTNET→LIVE fallback. Current M0.4/M0.6 blokuje LIVE, więc
current executable LIVE lease nie powstaje. Po legalnym future enablement canonical ProductCapabilities
i readiness ten sam M0.9 core obsłuży exact LIVE bez redesignu.

M0.10 zdefiniuje identity proof, PIN, biometrics, device trust, secrets i mechanizm autoryzacji zmian.
M0.11 musi przed production readiness trwale przechować accepted policies, switch state/generations,
issued/consumed leases potrzebne dla one-shot correctness i audit references. Model M0.9 jest in-memory
i nie deklaruje crash safety.

## Canonical Source-Closure Projection

The canonical JSON remains authoritative. This section is its deterministic projection for the source-closure rules.

### `canonical_integrity_fingerprint_policy`

```json
{
  "algorithm": "SHA-256",
  "input_shape": "EXACT_CLOSED_JSON_OBJECT_OF_EXPLICIT_SEMANTIC_INPUT_FIELDS",
  "json_canonicalization": {
    "sort_keys": true,
    "separators": [
      ",",
      ":"
    ],
    "ensure_ascii": false,
    "allow_nan": false
  },
  "encoding": "UTF-8",
  "array_policy": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER; NEVER_SORT_UNLESS_OWNING_CONTRACT_REQUIRES",
  "object_key_policy": "KEY_ORDER_NON_SEMANTIC; SORT_KEYS_CANONICALIZES",
  "number_policy": "NO_FLOAT_COERCION; VALIDATE_OWNING_FIELD_CONTRACT_BEFORE_HASHING",
  "decimal_string_policy": "VALIDATE_CANONICAL_OWNING_FIELD_CONTRACT; NO_FINGERPRINT_LAYER_NORMALIZATION",
  "timestamp_policy": "VALIDATE_CANONICAL_OWNING_TIMESTAMP_CONTRACT; NO_FINGERPRINT_LAYER_NORMALIZATION",
  "unicode_policy": "HASH_EXACT_VALIDATED_STRINGS; NO_HIDDEN_NFC_OR_NFD_TRANSFORMATION",
  "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
  "validation": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
  "authority_boundary": "INTEGRITY_ONLY; NEVER_CREATES_ACCEPTED_MEMBERSHIP, CURRENT_AUTHORITY, LIVE_READINESS, OR SELF_ENROLLMENT"
}
```

### `scope_hierarchy`

```json
{
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
  "incomparable_conflict": "two applicable policies at the same exact scope identity and revision with unequal semantic fingerprints => POLICY_CONFLICT",
  "scope_id_policy": {
    "scope_type_field": "scope_type",
    "scope_id_field": "scope_id",
    "bindings": {
      "PRODUCT_SYSTEM": {
        "type": "exact_literal",
        "value": "product"
      },
      "WORKSPACE": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "ws"
      },
      "PORTFOLIO": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "port"
      },
      "EXCHANGE_ACCOUNT": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "xacc"
      },
      "STRATEGY_INSTANCE": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "sinst"
      },
      "INSTRUMENT": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "instr"
      },
      "EXECUTION_ROUTE": {
        "type": "canonical_uuid7_prefixed_id",
        "prefix": "xroute"
      }
    },
    "validation_stage": "STAGE_1_INTRINSIC",
    "contextual_exclusions": [
      "ANCESTRY",
      "MEMBERSHIP",
      "ACCEPTED_OR_CURRENT_DESIGNATION",
      "HIERARCHY_APPLICABILITY"
    ]
  }
}
```

### `kill_switch_contract`

```json
{
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
  "generation_validation": "generation is positive and strictly increases per exact scope/environment history; duplicate/reuse/rollback or unknown state fails TRUSTED_CONTEXT_FAILURE",
  "field_schemas": {
    "scope_type": {
      "type": "enum",
      "values_source_pointer": "/scope_hierarchy/applicable_order"
    },
    "scope_id": {
      "type": "canonical_scope_id",
      "scope_type_field": "scope_type",
      "policy_source_pointer": "/scope_hierarchy/scope_id_policy"
    },
    "environment": {
      "type": "enum",
      "values": [
        "PAPER",
        "TESTNET",
        "LIVE"
      ]
    },
    "state": {
      "type": "enum",
      "values_source_pointer": "/kill_switch_contract/states"
    },
    "source_revision": {
      "type": "positive_non_boolean_integer"
    },
    "effective_at_utc": {
      "type": "canonical_utc_timestamp"
    },
    "generation": {
      "type": "positive_non_boolean_integer"
    },
    "accepted_authority_fingerprint_sha256": {
      "type": "sha256_lowercase_hex"
    },
    "record_fingerprint_sha256": {
      "type": "terminal_fingerprint",
      "derivation_source_pointer": "/kill_switch_contract/terminal_fingerprint"
    }
  },
  "terminal_fingerprint": {
    "field": "record_fingerprint_sha256",
    "algorithm": "SHA-256",
    "input_fields": [
      "scope_type",
      "scope_id",
      "environment",
      "state",
      "source_revision",
      "effective_at_utc",
      "generation",
      "accepted_authority_fingerprint_sha256"
    ],
    "excluded_fields": [
      "record_fingerprint_sha256"
    ],
    "input_shape": "JSON_OBJECT",
    "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
    "canonicalization": {
      "sort_keys": true,
      "separators": [
        ",",
        ":"
      ],
      "ensure_ascii": false,
      "allow_nan": false
    },
    "encoding": "UTF-8",
    "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
    "unicode_normalization": "NONE",
    "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
    "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
    "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY"
  },
  "intrinsic_authority_fence": "VALID_FINGERPRINT_IS_NOT_ACCEPTED_AUTHORITY; VALID_RECORD_IS_NOT_CURRENT_KILL_SWITCH; RAW_RECORD_IS_NOT_POLICY_AUTHORITY"
}
```

### `risk_decision_contract`

```json
{
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
  "aggregation": "any FAIL => DENY; else any INCOMPLETE => INCOMPLETE; else ALLOW",
  "lease_rule": "only accepted ALLOW can contribute to issuance; it is never sufficient alone",
  "accepted_context": "PrevalidatedRiskDecisionContext binds an independently recomputed immutable ALLOW RiskDecision fingerprint to the exact evaluation inputs",
  "decision_fingerprint": "SHA-256 over every semantic RiskDecision field except decision_fingerprint_sha256; recomputed at issuance and when retrieving accepted historical decision at dispatch",
  "rederivation": "issuance independently rederives the full PRE decision. Dispatch retrieves the exact accepted historical RiskDecision, recomputes all its semantic fields/fingerprint, requires ALLOW and matching lease binding, then revalidates current execution authority, policy/switch fences, POST accounting, reservation and lifetime; it does not reinterpret the historical PRE decision.",
  "terminal_fingerprint": {
    "field": "decision_fingerprint_sha256",
    "algorithm": "SHA-256",
    "input_fields": [
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
      "decision"
    ],
    "excluded_fields": [
      "decision_fingerprint_sha256"
    ],
    "input_shape": "JSON_OBJECT",
    "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
    "canonicalization": {
      "sort_keys": true,
      "separators": [
        ",",
        ":"
      ],
      "ensure_ascii": false,
      "allow_nan": false
    },
    "encoding": "UTF-8",
    "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
    "unicode_normalization": "NONE",
    "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
    "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
    "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY"
  },
  "intrinsic_authority_fence": "VALID_FINGERPRINT_DOES_NOT_SELF_ENROLL; ONLY_ACCEPTED_CORE_BINDING_CAN_PARTICIPATE_IN_LEASE_ISSUANCE",
  "limit_result_type_registry": [
    "MAX_ORDER_QUANTITY",
    "MAX_ORDER_NOTIONAL",
    "MAX_POST_TRADE_POSITION_QUANTITY",
    "MAX_POST_TRADE_POSITION_NOTIONAL",
    "MAX_GROSS_EXPOSURE",
    "MIN_AVAILABLE_CAPITAL_AFTER_RESERVATION",
    "DISPATCH_RESERVATION_ECONOMICS"
  ],
  "limit_result_result_registry": [
    "PASS",
    "FAIL",
    "INCOMPLETE"
  ],
  "limit_result_reason_codes": [
    "PASS",
    "LIMIT_BREACH",
    "MISSING_REQUIRED_INPUT",
    "MISSING_VALUATION"
  ],
  "limit_result_schema": {
    "exact_fields": [
      "limit_type",
      "effective_threshold",
      "observed_projected_value",
      "unit_asset_reference",
      "supplying_policy_scope",
      "result",
      "reason_code"
    ],
    "field_schemas": {
      "limit_type": {
        "type": "enum",
        "values_source_pointer": "/risk_decision_contract/limit_result_type_registry"
      },
      "effective_threshold": {
        "type": "canonical_exact_fraction_string",
        "nullable": false,
        "float_forbidden": true,
        "normalization": "REDUCED_NUMERATOR_SLASH_POSITIVE_DENOMINATOR"
      },
      "observed_projected_value": {
        "type": "nullable_canonical_exact_fraction_string",
        "nullable": true,
        "float_forbidden": true,
        "normalization": "REDUCED_NUMERATOR_SLASH_POSITIVE_DENOMINATOR"
      },
      "unit_asset_reference": {
        "type": "exact_upstream_object",
        "source_artifact": "exchange_accounts_and_instruments.json",
        "source_pointer": "/asset_reference_contract",
        "allowed_mapping_statuses": [
          "EXACT",
          "EXPLICIT_ALIAS"
        ],
        "mapping_status_registry_source_pointer": "/asset_reference_contract/mapping_statuses"
      },
      "supplying_policy_scope": {
        "type": "conditional_supplying_policy_scope",
        "ordinary_format": "scope_type + COLON + canonical scope_id",
        "scope_type_registry_pointer": "/scope_hierarchy/applicable_order",
        "scope_id_policy_pointer": "/scope_hierarchy/scope_id_policy",
        "synthetic_limit_type": "DISPATCH_RESERVATION_ECONOMICS",
        "synthetic_exact_value": "SYSTEM",
        "authority_boundary": "LABEL_ONLY_NOT_ACCEPTED_POLICY_AUTHORITY"
      },
      "result": {
        "type": "enum",
        "values_source_pointer": "/risk_decision_contract/limit_result_result_registry"
      },
      "reason_code": {
        "type": "enum",
        "values_source_pointer": "/risk_decision_contract/limit_result_reason_codes"
      }
    },
    "intrinsic_constraints": {
      "observed_value_matrix": {
        "PASS": "REQUIRED_NON_NULL",
        "FAIL": "REQUIRED_NON_NULL",
        "INCOMPLETE": "REQUIRED_NULL_BECAUSE_CURRENT_EMITTER_ONLY_USES_INCOMPLETE_WHEN_REQUIRED_TRUSTED_OBSERVATION_UNAVAILABLE"
      },
      "synthetic_dispatch_reservation_economics": {
        "effective_threshold": "0/1",
        "observed_projected_value": null,
        "supplying_policy_scope": "SYSTEM",
        "result": "INCOMPLETE",
        "reason_code": "MISSING_VALUATION"
      },
      "policy_reason_matrix": {
        "PASS": [
          "PASS"
        ],
        "FAIL": [
          "LIMIT_BREACH"
        ],
        "INCOMPLETE": [
          "MISSING_REQUIRED_INPUT"
        ]
      }
    }
  },
  "ordered_limit_results_array_schema": {
    "type": "array_of_exact_LimitResult",
    "item_schema_pointer": "/risk_decision_contract/limit_result_schema",
    "min_items": 0,
    "ordering": {
      "semantic": true,
      "validator_behavior": "REJECT_NON_CANONICAL_ORDER_NEVER_SORT",
      "policy_results_key": [
        "limit_type_registry_order",
        "unit_asset_reference_lexicographic_field_tuple"
      ],
      "synthetic_position": "DISPATCH_RESERVATION_ECONOMICS_LAST"
    },
    "duplicates": {
      "allowed": false,
      "uniqueness_key": [
        "limit_type",
        "unit_asset_reference",
        "supplying_policy_scope"
      ]
    }
  },
  "kill_switch_result_registry": [
    "OK",
    "KILL_SWITCH_ACTIVE",
    "TRUSTED_CONTEXT_FAILURE"
  ],
  "kill_switch_result_authority_boundary": "IMMUTABLE_EVALUATION_FACT_ONLY; OK_DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_KILL_SWITCH_AUTHORITY"
}
```

### `execution_lease_contract`

```json
{
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
  "distinct_identity": "each issuance receives a distinct canonical execution_lease_id",
  "terminal_fingerprint": {
    "field": "lease_fingerprint_sha256",
    "algorithm": "SHA-256",
    "input_fields": [
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
      "expires_at_utc"
    ],
    "excluded_fields": [
      "lease_fingerprint_sha256"
    ],
    "input_shape": "JSON_OBJECT",
    "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
    "canonicalization": {
      "sort_keys": true,
      "separators": [
        ",",
        ":"
      ],
      "ensure_ascii": false,
      "allow_nan": false
    },
    "encoding": "UTF-8",
    "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
    "unicode_normalization": "NONE",
    "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
    "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
    "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY"
  },
  "intrinsic_authority_fence": "VALID_FINGERPRINT_DOES_NOT_ESTABLISH_DISPATCH_AUTHORITY",
  "binding_array_schemas": {
    "effective_policy_bindings": {
      "type": "array_of_exact_tuple",
      "min_items": 1,
      "tuple_length": 6,
      "item_schema": [
        {
          "index": 0,
          "name": "risk_policy_id",
          "type": "canonical_uuid7_prefixed_id",
          "prefix": "rpol"
        },
        {
          "index": 1,
          "name": "revision",
          "type": "positive_non_boolean_integer"
        },
        {
          "index": 2,
          "name": "scope_type",
          "type": "enum",
          "values_source_pointer": "/scope_hierarchy/applicable_order"
        },
        {
          "index": 3,
          "name": "scope_id",
          "type": "canonical_scope_id",
          "scope_type_index": 2,
          "policy_source_pointer": "/scope_hierarchy/scope_id_policy"
        },
        {
          "index": 4,
          "name": "action",
          "type": "enum",
          "values": [
            "ALLOW",
            "DENY"
          ]
        },
        {
          "index": 5,
          "name": "semantic_fingerprint_sha256",
          "type": "sha256_lowercase_hex"
        }
      ],
      "ordering": {
        "semantic": true,
        "key": [
          "scope_hierarchy.applicable_order(scope_type)",
          "scope_id",
          "revision"
        ],
        "validator_behavior": "REJECT_NON_CANONICAL_ORDER_NEVER_SORT"
      },
      "duplicates": {
        "allowed": false,
        "uniqueness_key_indexes": [
          2,
          3,
          1
        ],
        "validator_behavior": "REJECT_NEVER_DEDUPLICATE"
      }
    },
    "kill_switch_bindings": {
      "type": "array_of_exact_tuple",
      "min_items": 0,
      "tuple_length": 8,
      "item_schema": [
        {
          "index": 0,
          "name": "scope_type",
          "schema_reference": "/kill_switch_contract/field_schemas/scope_type"
        },
        {
          "index": 1,
          "name": "scope_id",
          "schema_reference": "/kill_switch_contract/field_schemas/scope_id",
          "scope_type_index": 0
        },
        {
          "index": 2,
          "name": "state",
          "schema_reference": "/kill_switch_contract/field_schemas/state"
        },
        {
          "index": 3,
          "name": "source_revision",
          "schema_reference": "/kill_switch_contract/field_schemas/source_revision"
        },
        {
          "index": 4,
          "name": "effective_at_utc",
          "schema_reference": "/kill_switch_contract/field_schemas/effective_at_utc"
        },
        {
          "index": 5,
          "name": "generation",
          "schema_reference": "/kill_switch_contract/field_schemas/generation"
        },
        {
          "index": 6,
          "name": "accepted_authority_fingerprint_sha256",
          "schema_reference": "/kill_switch_contract/field_schemas/accepted_authority_fingerprint_sha256"
        },
        {
          "index": 7,
          "name": "record_fingerprint_sha256",
          "type": "sha256_lowercase_hex"
        }
      ],
      "ordering": {
        "semantic": true,
        "key": [
          "scope_hierarchy.applicable_order(scope_type)",
          "scope_id"
        ],
        "validator_behavior": "REJECT_NON_CANONICAL_ORDER_NEVER_SORT"
      },
      "duplicates": {
        "allowed": false,
        "uniqueness_key_indexes": [
          0,
          1
        ],
        "validator_behavior": "REJECT_NEVER_DEDUPLICATE"
      }
    }
  },
  "binding_authority_boundary": "BINDING_POSSESSION_AND_VALID_EMBEDDED_FINGERPRINTS_DO_NOT_ESTABLISH_MEMBERSHIP_OR_ACCEPTED_CURRENT_AUTHORITY; DISPATCH_MUST_RE_RESOLVE_CURRENT_ACCEPTED_AUTHORITY"
}
```

### `executable_boundary_schemas`

```json
{
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
  "RiskDecision": {
    "exact_fields": [
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
    "terminal_fingerprint": {
      "field": "decision_fingerprint_sha256",
      "algorithm": "SHA-256",
      "input_fields": [
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
        "decision"
      ],
      "excluded_fields": [
        "decision_fingerprint_sha256"
      ],
      "input_shape": "JSON_OBJECT",
      "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
      "canonicalization": {
        "sort_keys": true,
        "separators": [
          ",",
          ":"
        ],
        "ensure_ascii": false,
        "allow_nan": false
      },
      "encoding": "UTF-8",
      "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
      "unicode_normalization": "NONE",
      "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
      "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
      "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY"
    },
    "field_schemas": {
      "ordered_limit_results": {
        "type": "schema_reference",
        "source_pointer": "/risk_decision_contract/ordered_limit_results_array_schema",
        "validate_before_terminal_fingerprint": true
      },
      "kill_switch_result": {
        "type": "enum",
        "values_source_pointer": "/risk_decision_contract/kill_switch_result_registry",
        "validate_before_terminal_fingerprint": true
      }
    }
  },
  "PrevalidatedRiskDecisionContext": [
    "decision",
    "membership_id"
  ],
  "CoreAcceptedDecisionBinding": [
    "decision_fingerprint_sha256",
    "decision"
  ],
  "ExecutionLease": {
    "exact_fields": [
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
    "terminal_fingerprint": {
      "field": "lease_fingerprint_sha256",
      "algorithm": "SHA-256",
      "input_fields": [
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
        "expires_at_utc"
      ],
      "excluded_fields": [
        "lease_fingerprint_sha256"
      ],
      "input_shape": "JSON_OBJECT",
      "canonical_policy_pointer": "/canonical_integrity_fingerprint_policy",
      "canonicalization": {
        "sort_keys": true,
        "separators": [
          ",",
          ":"
        ],
        "ensure_ascii": false,
        "allow_nan": false
      },
      "encoding": "UTF-8",
      "array_order": "PRESERVE_VALIDATED_SEMANTIC_SOURCE_ORDER",
      "unicode_normalization": "NONE",
      "digest_format": "64_LOWERCASE_HEXADECIMAL_CHARACTERS",
      "validator": "RECOMPUTE_AND_COMPARE_EXACT_EQUALITY",
      "authority_boundary": "INTEGRITY_ONLY; DOES_NOT_ESTABLISH_ACCEPTED_OR_CURRENT_AUTHORITY"
    },
    "field_schemas": {
      "effective_policy_bindings": {
        "type": "schema_reference",
        "source_pointer": "/execution_lease_contract/binding_array_schemas/effective_policy_bindings",
        "validate_before_terminal_fingerprint": true
      },
      "kill_switch_bindings": {
        "type": "schema_reference",
        "source_pointer": "/execution_lease_contract/binding_array_schemas/kill_switch_bindings",
        "validate_before_terminal_fingerprint": true
      }
    }
  },
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
}
```
