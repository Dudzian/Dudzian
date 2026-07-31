# M0.6 — Strategy, Market Data and Execution Routing

**Status: under audit.** Autorytatywnym źródłem prawdy jest `strategy_market_data_and_execution_routing.json`; ten dokument jest jego opisowym odpowiednikiem. M0.6 jest kontraktem architektury z czystymi validatorami referencyjnymi w testach. Nie jest runtime'em i nie dodaje storage, IPC, QML, sieci, adapterów, URL-i endpointów, sekretów, credential payloads ani wykonywania lub lifecycle'u zleceń. M0.7 nie został rozpoczęty.

## Identity i encje

M0.6 przejmuje canonical lowercase UUIDv7 identity z M0.2: `sdef`, `sinst`, `mdr` i `xroute`. RouteReadiness jest nie-durable Core-owned projection mapowaną po exact `mdr` albo `xroute` route ID. Exact prefix i kanoniczny UUIDv7 route są częścią identity; display name oraz symbol nie są identity, a każdy foreign ID rozwiązuje się do jednej konkretnej encji.

`StrategyDefinition` obejmuje exact: `strategy_definition_id`, `workspace_id`, `strategy_type_id`, `definition_version`, `configuration`, `canonical_content_hash`, `hash_domain_separator`, `lifecycle_state`. Lifecycle to `DRAFT → ACTIVE → RETIRED`, bez wyjścia z `RETIRED`.

`StrategyInstance` obejmuje exact: `strategy_instance_id`, `workspace_id`, `portfolio_id`, exact ID i wersję definicji, account, universe, osobne route IDs oraz lifecycle. Powstaje jako `DRAFT`; osobne pierwsze bindy prowadzą po związaniu obu tras do `BOUND`; aktywacja jest dozwolona z `BOUND` lub `INACTIVE`, `ACTIVE → INACTIVE`, a `DRAFT`, `BOUND` i `INACTIVE` mogą przejść do `RETIRED`. Non-null binding jest immutable i rebind wymaga nowej instancji.

## Immutable StrategyDefinition

Current record jest w `strategy_definitions_by_id`. Wersje historyczne są w `previous_strategy_definitions_by_version_key` pod exact kluczem `<strategy_definition_id>@<definition_version>`, bez luk i duplikatów, dokładnie dla `1..current-1`. Jedyny resolver `resolve_strategy_definition_exact_version` nie wykonuje implicit upgrade. Nowa instancja wymaga current `ACTIVE`; istniejący pin do wcześniejszej `ACTIVE` wersji pozostaje poprawny mimo powstania nowszej wersji. Exact `RETIRED` nie może służyć nowej aktywacji.

Hash to SHA-256 jawnego separatora `cryptohunter.strategy-definition.v1\0` i kanonicznej konfiguracji UTF-8. Klucze są deterministycznie sortowane, whitespace nie ma znaczenia, float/NaN/Infinity są zabronione, bool nie jest integerem, a nested object jest zamknięty. Trusted validation zawsze przelicza hash.

## Routes, endpoint classes i readiness

MarketDataRoute i ExecutionRoute są oddzielnymi encjami. MarketDataRoute deklaruje exact workspace, exchange, environment, market type, adapter family, endpoint class, `PUBLIC`/`PRIVATE` scope, instruments, kanały (trades, order book, ticker, candles i właściwe kanały prywatne), snapshot/stream semantics, sekwencję, deterministic freshness, reconnect i route lifecycle. Public data nigdy nie nadaje execution authority.

ExecutionRoute deklaruje exact account/exchange/environment/market type, adapter family, endpoint class, instrument types, status, capability ceiling i dependencies. Autorytatywne `authorization_dependencies_by_environment` jest jedynym registry używanym przez bind, readiness, activation, testy i ten dokument; wymagane jest exact set equality. Bind sprawdza deklarację, natomiast chwilową operacyjność wszystkich dependencies dopiero readiness/activation.

Endpoint classes są abstrakcyjne i nie zawierają URL-i: `PAPER_PUBLIC_DATA`, `PAPER_SIMULATION`, `TESTNET_PUBLIC_DATA`, `TESTNET_PRIVATE_DATA`, `LIVE_PUBLIC_DATA`, `LIVE_PRIVATE_DATA`. Każda ma exact environment, access scope i allowed route kinds. Nie ma implicit selection ani cross-environment fallback; PAPER nie przechodzi do TESTNET/LIVE, TESTNET do LIVE, endpoint publiczny nie służy private execution, a prywatny kanał nie może znaleźć się na public route.

Bieżący readiness nie jest persisted route field. Jedynym źródłem jest Core-owned `route_readiness_by_id`; rekord zawiera exact `route_id`, `route_kind`, `readiness_state`, `observed_at`, `metadata_version`, `sequence_state`. Stany to `UNKNOWN`, `NOT_READY`, `READY`, `STALE`. Freshness i timestamp w przyszłości są oceniane wyłącznie wobec Core-owned `validation_time_utc`, nigdy zegara systemowego. Market readiness nie implikuje execution readiness, kind musi odpowiadać istniejącej trasie i orphan readiness jest nieważny.

## PAPER, TESTNET, LIVE

PAPER oznacza wyłącznie lokalną symulację, bez private exchange execution i bez obowiązkowego CredentialProfile; `active_credential_profile_id` może być null, a ProductCapabilities nadal muszą spełniać M0.4. TESTNET wymaga aktywnego exact account, capabilities, VALID AccountCapabilitySnapshot, aktywnego CredentialProfile o purpose `ORDER_ENTRY` i permission `PLACE_ORDERS`, z exact account/exchange/environment binding i bez LIVE fallback. W bieżącej edycji LIVE execution jest zawsze `LIVE_EXECUTION_FORBIDDEN`; widoczność może być opisana, lecz nie daje execution authority.

## Trusted context i integralność

Context jest zamknięty i zawiera wszystkie mapy wskazane w JSON oraz `validation_time_utc`. Każda mapa musi być rzeczywistym `dict`, mieć exact closed records i map-key binding. Jawne `{}` jest dozwolone według polityki mapy; `None`, `False`, lista, string ani integer nie są mapą. Malformed unrelated record unieważnia cały context.

Kolejność jest stała: exact request; exact top-level context; wszystkie rekordy; global references; operation lookup; lifecycle/routing/authority; immutable planned outcome; dispatcher denial membership. Persisted dangling graph daje `TRUSTED_CONTEXT_INVALID`; dopiero brak zasobu wskazanego wyłącznie przez poprawny request daje operation-specific denial.

Globalna walidacja rozwiązuje exact definition version, Workspace, Portfolio, ExchangeAccount, universe należący do exact account i obie trasy z exact workspace/account/exchange/environment/market type. Każdy instrument i catalog istnieje, universe source catalog set jest dokładnie zbiorem katalogów jego instrumentów, Instrument należy do Catalog, Catalog back-referencuje Instrument, scope zgadza się z account, a `Instrument.source_adapter_family_id == Catalog.adapter_family_id`.

Dla bound graph obowiązuje `ExecutionRoute.adapter_family_id == Instrument.source_adapter_family_id == Catalog.adapter_family_id`; persisted mismatch jest `TRUSTED_CONTEXT_INVALID`. Dla nieprzypiętego kandydata BIND mismatch jest `ROUTE_ADAPTER_MISMATCH`. Snapshot ma exact ID/account/exchange/environment/market type, canonical status `VALID`, `STALE` albo `REJECTED`, source hash zgodny z attestation oraz invariants czasowe wobec validation clock. Tylko `VALID` może przejść TESTNET operability; `STALE` i `REJECTED` dają `CAPABILITY_SNAPSHOT_BLOCKED`.

`credential_profiles_by_id` zawiera wyłącznie aktywnie wybrane profile, najwyżej jeden na account, z exact account/exchange/environment i lifecycle `ACTIVE`. TESTNET wymaga `ORDER_ENTRY`/`PLACE_ORDERS`; PAPER pozwala na null i pustą mapę.

## Operacje, denials i audit

Zamknięte operacje to `CREATE_STRATEGY_DEFINITION`, `ACTIVATE_STRATEGY_DEFINITION`, `RETIRE_STRATEGY_DEFINITION`, `CREATE_STRATEGY_INSTANCE`, `BIND_MARKET_DATA_ROUTE`, `BIND_EXECUTION_ROUTE`, `VALIDATE_ROUTE_READINESS`, `ACTIVATE_STRATEGY_INSTANCE`, `DEACTIVATE_STRATEGY_INSTANCE`, `RETIRE_STRATEGY_INSTANCE`. Authority to zawsze CoreHost, default success jest false, wynik jest immutable planem bez storage mutation. Readiness ma wyłącznie intent `VALIDATE_ONLY`, aktywacja wyłącznie `ACTIVATE`. `SUBMIT_ORDER`, `CREATE_ORDER`, `CANCEL_ORDER`, `REPLACE_ORDER`, `EXECUTE_TRADE` są jawnie zabronione.

Autorytatywne `validator_denial_registry`, `operation_validator_call_graph` i `allowed_denials_by_operation` eliminują kopiowane listy na krawędziach. Union deniali osiągalnych validatorów musi być dokładnie allowed minus `CONTRACT_INCONSISTENT`. Każda runtime operation/denial pair ma wykonywalny przypadek; poza request/context failure zaczyna on z poprawnym trusted context. `CONTRACT_INCONSISTENT` jest wyłącznie uszkodzeniem machine contract, nigdy zwykłym błędem domenowym.

Każda operacja ma exact success i denial audit event. Payload jest zamknięty, zawiera operation, outcome, denial i bezpieczne ID, bez sekretów, credential material lub URL. `UNKNOWN_OPERATION` oraz `CONTRACT_INCONSISTENT` mają odrębne eventy.

## Wykonywalny model walidacji referencyjnej

Kontrakt maszynowy definiuje dla każdej operacji zamknięte `request_fields`, `request_types`, `constants`, `nullable_fields` i `nested_schemas`; nie istnieje ogólny ani domyślny payload. Wspólny `validate_request` sprawdza rzeczywisty dict, exact fields, durable ID prefixes, enumy, constant authority/intent, rozróżnienie integer/bool i zamknięte obiekty konfiguracji.

`record_schemas` zawiera wykonywalne schema dla StrategyDefinition, StrategyInstance, obu routes, RouteReadiness oraz projekcji Account, Universe, Instrument, Catalog, CapabilitySnapshot, CredentialProfile, ProductCapabilities i Portfolio. Każde schema deklaruje exact fields/types, identity, enumy, nullability, unikalność/pustość tablic, nested schemas i entity references. `validate_record` jest wspólnym interpreterem tych definicji, a `validate_context` wykonuje kolejno exact top level, wszystkie schemas i map-key bindings, definition lineage/hashes, global references oraz derived invariants. `active_strategy_instances` jest unique `sinst` array równą dokładnie zbiorowi persisted rekordów `ACTIVE`, a nie mapą ani falsey placeholderem.

Referencyjny `dispatcher(operation, request, context)` nie przyjmuje oczekiwanego deniala. Wykonuje callable wskazane przez `operation_validator_call_graph`: `validate_request`, następnie pełny `validate_context`, następnie mały validator operacji. Dopiero potem zwraca immutable planned decision i sprawdza denial membership; rozbieżność machine contract jako jedyna prowadzi do `CONTRACT_INCONSISTENT`. Testowy builder reachability mutuje request albo graph contextu, lecz oczekiwany kod jest używany wyłącznie do zbudowania wejścia i asercji, nigdy jako argument dispatchera lub validatora.

## Semantic closure of lifecycle and route authority

`UPDATE_STRATEGY_INSTANCE_STATE` is not an operation. `BOUND` is derived only after both
immutable first binds; activation, deactivation, and retirement are exclusively planned by
their dedicated operations. Instance creation requires an active current definition and
active universe, exact Workspace/Portfolio/Account ownership, and two null route IDs.

Definition lineage preserves Workspace, strategy type, and hash domain separator across all
versions. Trusted-context validation covers every route, including unbound candidates, while
operation-specific candidate mismatches require an independently valid candidate graph.
Readiness and activation share bound-route operability checks. TESTNET activation requires
`PLACE_ORDERS` in the snapshot, execution-route ceiling, and selected credential. Future
snapshot observations invalidate trusted context; `STALE` or `REJECTED` snapshots are activation
denials. Every successful operation is checked against derived planned-outcome invariants.

## Trusted graph and readiness closure

Both route binds accept only a `DRAFT` StrategyInstance. `BOUND`, `ACTIVE`, and `INACTIVE`
produce `STRATEGY_INSTANCE_BINDING_MISMATCH`; `RETIRED` produces
`RETIRED_RESOURCE_FORBIDDEN`. Creating any StrategyDefinition version requires an existing
Workspace (`WORKSPACE_NOT_FOUND` otherwise), and every persisted version uses the contract's
exact hash domain separator.

Every Instrument–Catalog relationship has exact exchange, environment, market type, adapter,
catalog identity, and reverse membership. Every Account requires the ProductCapabilities
projection for its exact environment. The selected-credential map contains only profiles in
`ACTIVE` lifecycle selected by their Account; a structurally absent TESTNET selection remains
an activation-time `ACCOUNT_READINESS_BLOCKED` denial.

The authoritative array registries close market-data channels, instrument types, route
capabilities, and authorization-dependency names. Market route data scope exactly matches
endpoint access scope, and public scope rejects private channels. Market readiness age uses
the bound route's `freshness_policy.max_age_seconds` with an inclusive boundary. Execution
readiness uses the machine `execution_readiness_max_age_seconds` policy and requires
`NOT_APPLICABLE` sequence state.

Planned definition creation includes its canonical content hash, exact version, ownership,
strategy type, and authoritative separator. Bind outcomes prove DRAFT source state, no rebind,
and DRAFT/BOUND derivation. Trusted graph omissions are rejected during context validation;
only malformed machine registries or call graphs can become `CONTRACT_INCONSISTENT`.

## Authorization and reverse-integrity closure

`validate_authorization_operability` is the single current-authority check shared by readiness
and activation. PAPER depends only on ProductCapabilities and never on snapshot status/age/
permissions or credentials. TESTNET additionally requires an ACTIVE Account, a VALID and fresh
snapshot containing `PLACE_ORDERS`, enabled ProductCapabilities permitting the exact operation,
an ACTIVE selected `ORDER_ENTRY` credential with `PLACE_ORDERS`, and an execution ceiling with
`PLACE_ORDERS`. LIVE execution readiness and activation are always forbidden. Snapshot age uses
`account_capability_snapshot_policy.max_age_seconds_by_environment` with an inclusive boundary.

Trusted validation checks both directions of every Instrument–Catalog edge, every snapshot
against its existing Account, and every Portfolio against an existing Workspace, including
unrelated records. UNKNOWN_OPERATION and CONTRACT_INCONSISTENT use their dedicated audit events.
Every registered operation has an exact planned-outcome shape; no unknown plan is accepted.
RouteReadiness is a non-durable Core-owned projection keyed by an exact `mdr` or `xroute` ID and
has no independent identity prefix. Broken lineage is a trusted-context failure, so there is no
runtime `STRATEGY_DEFINITION_LINEAGE_INVALID` denial.

## Readiness lifecycle and complete-plan closure

Readiness runs `validate_readiness_lifecycle` before authorization: DRAFT is a binding mismatch,
BOUND/ACTIVE/INACTIVE continue, and RETIRED is forbidden. PAPER checks ProductCapabilities
before—and instead of—Account lifecycle, snapshot, credential, or TESTNET ceiling requirements.
A persisted instance may never reference a DRAFT definition; RETIRED pinning remains historical,
but reactivation requires an exact ACTIVE definition.

Definition lifecycle conflicts now use `STRATEGY_DEFINITION_STATE_CONFLICT`; only RETIRED uses
`RETIRED_RESOURCE_FORBIDDEN`. CREATE decisions contain the exact complete record fields,
including explicit DRAFT lifecycle and copied configuration. Decisions recursively freeze
objects as mapping proxies and arrays as tuples, so nested planned data is immutable without
mutating either request or trusted context.

## Terminal lifecycle and canonical projection enums

`validate_strategy_instance_lifecycle_preflight` is shared by activation and deactivation. For
activation it runs before authorization: only BOUND and INACTIVE continue; DRAFT and ACTIVE are
binding conflicts, while RETIRED is terminal. For deactivation the same preflight succeeds only
from ACTIVE, returns the same terminal denial for RETIRED, and a binding conflict otherwise.
Readiness retains its independent DRAFT/RETIRED preflight.

Projection arrays resolve rather than duplicate closed registries: snapshot capabilities and
credential permissions use M0.5 `credential_profile_contract.permission_registry`, while
ProductCapabilities allowed values use M0.4
`capability_id_registry.current_schema_allowed_capability_ids`. Unknown array members invalidate
the trusted context. TESTNET still requires `PLACE_ORDERS` in snapshot, credential, and route
ceiling; PAPER checks only its M0.4 `PAPER_LOCAL_SIMULATION` capability and LIVE execution stays
forbidden.

## Final consistency closure

The shared lifecycle preflight accepts only activation and deactivation operation names; any
other call-graph placement is a machine-contract fault with the dedicated contract audit event.
Snapshot `VALID`, `STALE`, and `REJECTED` states are structurally valid. Orphan/scope/hash/future-time
failures invalidate context, while `STALE` or `REJECTED` snapshots remain trusted records that deny
TESTNET operability.

M0.4 supplies one global closed capability-ID registry plus separate environment booleans, not a
per-environment ID allowlist. Therefore extra canonical IDs are structurally harmless and grant
no authority: M0.6 still requires the exact environment capability. All declared external JSON
Pointers are executable contract inputs and a broken pointer is a machine-contract fault. Direct
call-graph execution and dispatcher decisions have exact parity for every runtime denial pair.

## Canonical M0.5 scalar projections

Scalar projection enums are resolved directly from M0.5 rather than copied locally. Account
lifecycle uses `/exchange_account_contract/lifecycle_states`, universe lifecycle uses
`/trading_universe_contract/lifecycle_states`, snapshot status uses
`/account_capability_snapshot_contract/statuses`, and credential purpose uses
`/credential_profile_contract/credential_purposes`. Canonical inactive states remain structurally
trusted but fail operation-specific readiness: inactive TESTNET accounts and non-order-entry
credentials return `ACCOUNT_READINESS_BLOCKED`, inactive universes return
`TRADING_UNIVERSE_INVALID`, and `STALE` or `REJECTED` snapshots return
`CAPABILITY_SNAPSHOT_BLOCKED`. Broken references are machine-contract faults.


## Canonical M0.5 Instrument and Catalog projections

`InstrumentProjection.trading_status` resolves `/instrument_contract/trading_statuses`; `TRADING`
is operational, while `HALTED`, `SUSPENDED`, `DELISTED`, and `UNKNOWN` are structurally trusted
but deny activation with `INSTRUMENT_SCOPE_MISMATCH`. `InstrumentCatalogProjection.status`
resolves `/instrument_catalog_snapshot_contract/statuses` and carries `observed_at_utc`, `effective_at_utc`, and `stale_after_utc` with an exact ordered timestamp graph. A catalog
is operational only when its status is `VALID` and `validation_time_utc < stale_after_utc`; canonical
`PARTIAL`, `STALE`, `REJECTED`, or an expired `VALID` catalog denies activation with
`TRADING_UNIVERSE_INVALID`. Structural schema validation and operation-specific operability are
distinct. All declared external JSON Pointers fail closed as machine-contract faults.
