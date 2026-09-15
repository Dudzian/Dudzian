# M0.6 — Strategy, Market Data and Execution Routing

**Status: closed.** Autorytatywnym źródłem prawdy jest `strategy_market_data_and_execution_routing.json`; ten dokument jest jego opisowym odpowiednikiem. M0.6 jest kontraktem architektury z czystymi validatorami referencyjnymi w testach. Nie jest runtime'em i nie dodaje storage, IPC, QML, sieci, adapterów, URL-i endpointów, sekretów, credential payloads ani wykonywania lub lifecycle'u zleceń. M0.7 nie został rozpoczęty.

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

PAPER oznacza wyłącznie lokalną symulację, bez private exchange execution i bez obowiązkowego CredentialProfile; `active_credential_profile_id` może być null, a ProductCapabilities nadal muszą spełniać M0.4. TESTNET wymaga aktywnego exact account, observed_permission_set, VALID AccountCapabilitySnapshot, aktywnego CredentialProfile o purpose `ORDER_ENTRY` i permission `PLACE_ORDERS`, z exact account/exchange/environment binding i bez LIVE fallback. LIVE jest first-class środowiskiem docelowej architektury, lecz jego execution jest w bieżącej edycji wyłączone przez kanoniczną politykę M0.4 ProductCapabilities. Kanoniczny M0.5 aktualnie nie ma ENABLED venue obsługującego LIVE, ale nie jest to permanentny invariant bezpieczeństwa M0.6. Venue capability jest konieczne dla przyszłego LIVE, lecz samo nigdy nie nadaje execution authority ani nie omija account, credential/capability lub route readiness.

## Trusted context i integralność

Context jest zamknięty i zawiera wszystkie mapy wskazane w JSON oraz `validation_time_utc`. Każda mapa musi być rzeczywistym `dict`, mieć exact closed records i map-key binding. Jawne `{}` jest dozwolone według polityki mapy; `None`, `False`, lista, string ani integer nie są mapą. Malformed unrelated record unieważnia cały context.

Kolejność jest stała: exact request; exact top-level context; wszystkie rekordy; global references; operation lookup; lifecycle/routing/authority; immutable planned outcome; dispatcher denial membership. Persisted dangling graph daje `TRUSTED_CONTEXT_INVALID`; dopiero brak zasobu wskazanego wyłącznie przez poprawny request daje operation-specific denial.

Globalna walidacja rozwiązuje exact definition version, Workspace, Portfolio, ExchangeAccount, universe należący do exact account i obie trasy z exact workspace/account/exchange/environment/market type. Każdy instrument i catalog istnieje, universe source catalog set jest dokładnie zbiorem katalogów jego instrumentów, Instrument należy do Catalog, Catalog back-referencuje Instrument, scope zgadza się z account, a `Instrument.source_adapter_family_id == Catalog.adapter_family_id`.

Dla bound graph obowiązuje `ExecutionRoute.adapter_family_id == Instrument.source_adapter_family_id == Catalog.adapter_family_id`; persisted mismatch jest `TRUSTED_CONTEXT_INVALID`. Globalna walidacja zamkniętego Exchange Registry odrzuca także nieprzypiętego kandydata z adapter mismatch jako `TRUSTED_CONTEXT_INVALID`. Snapshot ma exact ID/account/exchange/environment/market type, canonical status `VALID`, `STALE` albo `REJECTED`, source hash zgodny z attestation oraz invariants czasowe wobec validation clock. Tylko `VALID` może przejść TESTNET operability; `STALE` i `REJECTED` dają `CAPABILITY_SNAPSHOT_BLOCKED`.

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
observed_permission_set, and authorization-dependency names. Market route data scope exactly matches
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
`PLACE_ORDERS`. Current-edition LIVE execution readiness and activation are disabled by the canonical ProductCapabilities authority policy. Snapshot age uses
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

Projection arrays resolve rather than duplicate closed registries: snapshot observed_permission_set and
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

## Instrument authority, freshness and capability types

`InstrumentProjection` preserves the authority-relevant M0.5 fields exactly:
`instrument_id`, `workspace_id`, `catalog_snapshot_id`, `exchange_id`, `environment`,
`market_type`, `instrument_type`, `source_adapter_family_id`, `trading_status`,
`metadata_version`, `observed_at_utc`, `effective_at_utc`, and `stale_after_utc`. The canonical
catalog field is directly `catalog_snapshot_id`; M0.6 does not retain a local identity alias or perform any rename. Trading statuses
continue to resolve from `/instrument_contract/trading_statuses`.

An Instrument in an instance Universe has exact ownership:
`Instrument.workspace_id == StrategyInstance.workspace_id == ExchangeAccount.workspace_id ==
MarketDataRoute.workspace_id == ExecutionRoute.workspace_id`. A dangling Workspace or persisted
cross-Workspace Instrument is `TRUSTED_CONTEXT_INVALID`. Instrument metadata is structurally
valid only when `metadata_version` is a positive non-bool integer,
`observed_at_utc <= effective_at_utc < stale_after_utc`, and neither observation nor effective
time is later than `validation_time_utc`.

Structural validity is separate from operability. A canonical `HALTED`, `SUSPENDED`, `DELISTED`,
or `UNKNOWN` Instrument remains structurally trusted but returns `INSTRUMENT_SCOPE_MISMATCH`.
`TRADING` is operational only while `validation_time_utc < stale_after_utc`; equality is stale
and returns the same denial. Thus an expired but well-formed Instrument is not mislabeled as a
malformed trusted context.

`AccountCapabilitySnapshotProjection.supported_instrument_types` is an exact, unique array; M0.5 permits the canonical array to be empty whose values resolve from canonical M0.5 `/instrument_type_registry`; they must also be
supported by the applicable exchange entry and allowed for the exact market type. The machine
contract explicitly maps local projection `observed_permission_set` to canonical M0.5
`observed_permission_set`, so that rename is not implicit.

For TESTNET, readiness and activation require
`set(Universe Instrument.instrument_type) <=
set(AccountCapabilitySnapshot.supported_instrument_types)`. A structurally valid snapshot that
omits a required type returns exactly `CAPABILITY_SNAPSHOT_BLOCKED`; it is neither
`TRUSTED_CONTEXT_INVALID` nor `CONTRACT_INCONSISTENT`. PAPER remains governed only by
ProductCapabilities. LIVE is non-runtime and non-reachable under the current edition's ProductCapabilities policy.

The reusable `validate_strategy_execution_operability` callable is present in both
`VALIDATE_ROUTE_READINESS` and `ACTIVATE_STRATEGY_INSTANCE` call graphs after lifecycle and
authorization checks. It performs Universe lifecycle, Catalog status/freshness, Instrument
status/freshness and Workspace ownership, plus the TESTNET snapshot instrument-type subset check.
Consequently readiness cannot succeed for an instance that the same current Instrument or
Catalog authority would immediately prevent from activating.

## Canonical projection identities and closed Exchange Registry

M0.6 uses canonical M0.5 identities without local aliases: InstrumentCatalogProjection uses
`catalog_snapshot_id`, AccountCapabilitySnapshotProjection uses
`account_capability_snapshot_id` and `observed_permission_set`, and TradingUniverseProjection
uses `source_catalog_snapshot_ids`. The executable cross-contract projection audit records every
canonical identity, parent, authority field, registry, structural invariant, operability rule,
and intentional omission.

Every exchange-bound Account, Instrument, Catalog, Snapshot, CredentialProfile, MarketDataRoute,
and ExecutionRoute resolves the closed M0.5 `/exchange_registry_contract/entries`. The entry must
exist and be `ENABLED`; environment, market type, instrument type, and adapter family must match.
The positive PAPER and TESTNET fixtures resolve `paper_simulated_venue` and
`generic_testnet_venue` directly from M0.5. M0.5 currently has no enabled LIVE entry, so the
current canonical data cannot resolve a LIVE exchange-bound graph. That data state is not an
execution authority and is not a permanent M0.6 invariant: a future coordinated contract version
may add an enabled LIVE-capable venue and update its canonical fingerprints. LIVE execution would
still remain disabled until ProductCapabilities and all later authority gates explicitly allow it.

M0.5 permits an empty canonical `supported_instrument_types` list. M0.6 therefore accepts `[]`
structurally, but TESTNET readiness and activation for a non-empty Universe return exactly
`CAPABILITY_SNAPSHOT_BLOCKED`. Snapshot authority carries canonical `version`,
`observed_at_utc`, `effective_at_utc`, `stale_after_utc`, `adapter_family_id`, `adapter_version`,
and `content_hash`. Its temporal graph and future timestamps are structural; TESTNET operability
additionally requires `VALID`, `validation_time_utc < stale_after_utc`, and the stricter local
max-age policy.

## Executable M0.5 projection attestation and account authority

M0.6 uses one trust boundary for every M0.5 projection: **Model A — full canonical projection**.
A descriptive “validated upstream” statement, boolean, arbitrary attestation string, or unbound
local digest grants no authority. Exact schemas, canonical registries, map-key binding and
operation-specific operability are executable. Hash-versioned Snapshot, Catalog and Universe
records additionally recalculate the M0.5 SHA-256 digest and traverse trusted predecessor maps
with visited-set cycle protection.

AccountCapabilitySnapshot includes canonical `previous_snapshot_id`. Its hash uses the exact
M0.5 `/account_capability_snapshot_contract/content_hash_definition`: the canonical input object
contains precisely the declared fields, object keys are lexicographically sorted, unordered
permission and instrument-type arrays are sorted, the M0.5 domain separator plus newline prefixes
the canonical UTF-8 JSON, and the result is lowercase hexadecimal SHA-256. Version 1 requires a
null predecessor; later versions require the exact previous version with identical Account,
Exchange, Environment and Market scope. Missing predecessors, gaps, self-reference and cycles
invalidate trusted context.

InstrumentCatalogSnapshot includes canonical `adapter_version`, `content_hash`, and
`previous_snapshot_id`, hashes exactly through
`/instrument_catalog_snapshot_contract/content_hash_definition`, and traverses
`previous_catalogs_by_id`. TradingUniverse includes canonical version, lifecycle timestamps,
`previous_version_id`, `content_hash`, and `creation_reason`, hashes through
`/trading_universe_contract/content_hash_definition`, canonically sorts instrument/catalog IDs,
and traverses `previous_universes_by_id` with exact account scope and version continuity.
Mutating any hash input without recomputing its canonical hash is `TRUSTED_CONTEXT_INVALID`.

ExchangeAccount authority now includes canonical `connection_state`, `execution_authorization`,
and `external_account_identity_state`. TESTNET readiness and activation apply
`/current_edition_account_operability_policy`: lifecycle must be operational, connection state
must be one of the canonical operational connection states, and execution authorization must be
`ORDER_ENTRY_ALLOWED`. A structurally canonical but non-operational value returns
`ACCOUNT_READINESS_BLOCKED`.

A `VERIFIED` scalar alone is insufficient. The exact trusted external identity projection is
resolved by Account ID, exact-binds Exchange/Environment/Market, validates the trusted identity
tuple, and enforces tuple uniqueness across the trusted context. Missing, mismatched,
unavailable, or colliding identity blocks account readiness; malformed projection structure is
`TRUSTED_CONTEXT_INVALID`.

CredentialProfile uses canonical M0.5 names `environment_scope`, `credential_purpose`, and
`permission_snapshot`, plus created/retired timestamps and `rotated_from_credential_profile_id`.
The selected ACTIVE profile exact-binds Account/Exchange/Environment and its predecessor chain is
resolved through `previous_credential_profiles_by_id` with retired predecessor scope and cycle
checks. For TESTNET order entry, effective permissions are the intersection of CredentialProfile
`permission_snapshot`, AccountCapabilitySnapshot `observed_permission_set`, and trusted external
identity `observed_permission_set`. Every source must contain both `READ_ACCOUNT` and
`PLACE_ORDERS`; `WITHDRAW` in any source blocks readiness and activation with
`ACCOUNT_READINESS_BLOCKED`.

LIVE metadata remains descriptive vocabulary only: `LIVE_PUBLIC_DATA` and `LIVE_PRIVATE_DATA`
do not attest venue support or grant execution authority. Canonical M0.5 must contain LIVE in its
environment registry and no ENABLED Exchange Registry entry supporting LIVE. This required
current-edition cross-contract invariant keeps LIVE non-runtime and non-reachable; coordinated
canonical or machine-attestation drift fails closed as `CONTRACT_INCONSISTENT`. M0.7 remains
unstarted.


## Historical projections and exception-safe machine pointers

All canonical references, including `/exchange_registry_contract/entries`, pass through one
exception-safe JSON Pointer resolver. It validates the source-contract allowlist, absolute pointer
syntax and escapes, every segment, result type, the exact Exchange Registry entry shape, closed
values, and unique `exchange_id`. Machine-pointer corruption is converted by the dispatcher into
immutable `CONTRACT_INCONSISTENT` with `STRATEGY_ROUTING_CONTRACT_INCONSISTENT`; malformed
persisted context remains `TRUSTED_CONTEXT_INVALID`.

Model A validation is global, not merely lineage-reachable. Every record in current and previous
Snapshot, Catalog, Universe, and Credential maps passes its exact schema, map-key, canonical
registries, scope, timestamps, membership, and account/exchange/adapter bindings. Hash-versioned
records additionally pass canonical hash recalculation, exact predecessor continuity, scope, and
visited-set cycle protection. Credential history must be fully reached from selected ACTIVE
profiles, contain only RETIRED predecessors, and satisfy creation/retirement chronology. Every
TrustedExternalIdentity record is validated globally for its map-key Account binding, exact
Exchange/Environment/Market scope, canonical state and permissions, non-future verification, and
trusted tuple uniqueness; non-operational but structurally canonical states remain operation-level
`ACCOUNT_READINESS_BLOCKED`.

M0.5 permits an empty canonical `supported_instrument_types` set. It is structurally valid, while
a non-empty TESTNET Universe with a missing type is denied as `CAPABILITY_SNAPSHOT_BLOCKED`.
The LIVE execution authority policy records that LIVE is supported by the target architecture but
is not executable or runtime-reachable in the current edition. Authority comes from the canonical
ProductCapabilities policy; venue support and endpoint vocabulary alone grant no authority, and
cross-environment fallback remains disabled.

## Canonical Instrument history and contract metadata closure

`InstrumentProjection` is a full Model A projection whose exact fields are resolved from M0.5
`/instrument_contract/record_fields`, including venue/display symbols, asset references, trading
constraints, derivative metadata, catalog identity, metadata version, timestamps, and adapter
lineage. The closed trusted context carries `instrument_history_by_id` as a map from an Instrument
ID to an ordered, non-empty array of full Instrument records. Every record exact-binds the map key,
Catalog, reverse Catalog membership, Exchange scope and adapter. Versions are positive, unique and
strictly increasing, the identity tuple `(exchange_id, environment, market_type, venue_symbol)` is
immutable, and a current record must have a metadata version greater than its historical maximum.

Current Catalog membership resolves only a current Instrument with the exact current
`catalog_snapshot_id`. Previous Catalog membership resolves only a full record from
`instrument_history_by_id` with the exact historical Catalog ID; a current Instrument cannot stand
in for it. Historical Universe membership likewise resolves Instrument records against its exact
`source_catalog_snapshot_ids` and their validated current or predecessor Catalog versions.

The M0.5 Exchange Registry entries and each of the three canonical hash-definition objects are
bound by SHA-256 fingerprints of canonical JSON stored in the M0.6 machine contract. Validation
covers every field and rejects malformed metadata, extra/missing hash-definition keys, changed
algorithms, domain separators, inputs, canonicalization, encoding or digest format as
`CONTRACT_INCONSISTENT` with `STRATEGY_ROUTING_CONTRACT_INCONSISTENT`. Dispatcher and direct parity
use the shared exception-safe `execute_validator_call_graph` executor.

Snapshot presence follows the canonical Exchange Registry discovery policy, not an environment
shortcut. `STATIC_BUILD_TIME` permits the PAPER simulated Account to omit its Snapshot; a supplied
optional Snapshot remains fully validated. `ADAPTER_SNAPSHOT_REQUIRED` requires TESTNET to bind a
full canonical Snapshot. CredentialProfile is also full Model A metadata and carries a validated
opaque `secure-store://` locator, nullable non-empty public-key identifier, and literal
`saas_sync_candidate = false`, without secret material. TrustedExternalIdentity additionally
exact-binds `adapter_version_source` to the Exchange adapter family plus a non-empty version.

LIVE remains non-runtime and non-reachable in the current edition, and neither readiness nor
activation can acquire LIVE execution authority while canonical ProductCapabilities disable it.
Uncoordinated M0.5 registry edits still fail as generic canonical dependency fingerprint drift;
the presence of a LIVE-capable venue is not itself a contract fault. M0.6 is closed and M0.7
remains unstarted.

Canonical Instrument structural validation resolves the complete M0.5 AssetReference and decimal
contracts. Asset references are exact closed objects (`venue_asset_code`,
`canonical_display_code`, `asset_namespace`, `mapping_status`), accept trusted `EXACT` or
`EXPLICIT_ALIAS` mappings, and fail closed for `AMBIGUOUS`, `UNKNOWN`, malformed, missing-field,
extra-field, or wrong-namespace values. The same validator covers base, quote, and every non-null
settlement reference. Spot and margin settlement is optional but, when present, its venue code
equals the quote venue code.

The schema resolves the canonical environment, market-type and instrument-type registries,
allowed pairs, trading statuses, record fields, AssetReference contract, and decimal policy. It
therefore represents `SPOT/SPOT_PAIR`, `MARGIN/MARGIN_PAIR`,
`PERPETUAL/PERPETUAL_CONTRACT`, `DELIVERY_FUTURES/DELIVERY_FUTURE`, and `OPTIONS/OPTION`.
That structural coverage grants no execution authority: narrower edition limits remain solely in
operation-specific operability. Credential environment scope resolves the same full registry,
including `PAPER`.

These bindings are executable rather than documentary. The single
`resolve_instrument_projection_contract` helper resolves every declared
`InstrumentProjection.canonical_projection_refs` pointer, including
`/derivative_consistency_rules`, through the exception-safe canonical pointer resolver. It closes
registry contents, allowed-pair and derivative mapping keys, AssetReference and decimal-policy
object keys, and requires the local exact field set to equal canonical
`/instrument_contract/record_fields`. Pointer corruption is a machine-contract fault and both
dispatcher and direct execution return `CONTRACT_INCONSISTENT` with
`STRATEGY_ROUTING_CONTRACT_INCONSISTENT`; malformed persisted Instrument data remains
`TRUSTED_CONTEXT_INVALID`.

`canonical_projection_binding_manifest` attests every projection binding with its exact source
contract, JSON Pointer, result type, canonical JSON fingerprint, and closed consumer list. A second
`canonical_enum_binding_manifest` covers the dynamically collected complete set of external
canonical scalar, array, and pointer enum consumers across
Account, Instrument, Catalog, Snapshot, Credential, TrustedExternalIdentity and both route schemas.
Validation cross-checks the Instrument environment, market type, instrument type and trading status
schema bindings against their projection bindings. Consequently a same-type pointer swap, removed
schema binding, changed canonical content, or changed fingerprint is
`CONTRACT_INCONSISTENT`, while a persisted noncanonical enum remains
`TRUSTED_CONTEXT_INVALID`.

Each enum attestation also binds the schema name, field name, exact field type, binding kind,
non-nullability, single exact-fields membership, and—where applicable—the entire array policy.
Scalar bindings forbid local enum fallbacks. Array bindings require `array[enum]`, the exact
`item_registry_ref`, declared `unique` and `empty_allowed` booleans, and no local `item_registry`.
The manifest set must equal the consumer set collected from every
`enum_canonical_pointer_ref`, `enum_registry_ref`, and `array_policy.item_registry_ref`; no fixed
entry count is used as completeness evidence. Same-type swaps among lifecycle/status registries or
between M0.4 product capabilities and M0.5 permissions fail as machine-contract corruption through
both dispatcher and direct execution.

One decimal helper first applies the exact M0.5 regex and then constructs `Decimal`. It requires
positive price tick and quantity step, positive derivative contract size and option strike when
required, non-negative non-null minima and maxima, and ordered min/max pairs. All four limit fields
are nullable: null/null means no limits, value/null means a lower limit only, null/value is invalid,
and value/value requires minimum no greater than maximum. The string `"0"` is a real zero limit;
only null means absence. Consequently non-strings, scientific notation, signs, negative zero, leading zeroes,
trailing fractional zeroes, whitespace, underscores, empty strings, NaN and Infinity fail closed.

Complete PAPER trusted-graph fixtures—not isolated Instrument records—prove structural coverage of
all five pairs across Account, Catalog, Instrument, Universe, MarketDataRoute, ExecutionRoute,
ProductCapabilities, hashes, and reverse references. Account, Catalog, Snapshot, identity, and both
route schemas resolve canonical market types, while ExecutionRoute supported instrument types
resolve canonical `/instrument_type_registry`.

Structural coverage remains separate from authority. The closed
`current_edition_execution_pair_policy` grants M0.6 readiness and activation only to
`SPOT/SPOT_PAIR`. `VALIDATE_ROUTE_READINESS` and `ACTIVATE_STRATEGY_INSTANCE` return the ordinary
`INSTRUMENT_SCOPE_MISMATCH` denial for each other structurally valid pair. Bind continues to check
the declared graph rather than current operability and therefore does not apply this edition gate.
The exception-safe `resolve_current_edition_execution_pair_policy` requires the exact six-field
policy object, exact SPOT-only pair, exact two consumers, fixed denial, and both separation flags.
Mutating the policy to add MARGIN or any other authority cannot grant or remove access: dispatcher
and direct execution return `CONTRACT_INCONSISTENT` with the dedicated contract audit event. Both
market-data and execution-route first binds remain outside this gate and produce their declared
`DRAFT` resulting state for structurally valid unsupported pairs.

A global identity index spans current and historical Instrument maps. The exact tuple
`(exchange_id, environment, market_type, venue_symbol)` maps to one `instrument_id`, and each ID
maps immutably to one tuple while allowing multiple metadata versions. Current and historical
Universe resolution also requires `Instrument.workspace_id == ExchangeAccount.workspace_id`, so
an exact predecessor Catalog cannot authorize a foreign-Workspace Instrument.

TrustedExternalIdentity uses one helper in global integrity and authorization operability. Its
exact tuple is `(exchange_id, environment, market_type, venue_account_identifier,
subaccount_identifier)`—never `account_type`—and collision scope contains only ACTIVE
ExchangeAccounts with matching `VERIFIED` identity. Thus two active accounts collide even when
their account types differ, while a disabled duplicate does not block the active account.
End-to-end regressions exercise both cases through the full trusted context. Historical Universe
validation likewise rejects a historical Instrument bound through an exact previous Catalog when
that Instrument belongs to a second, otherwise existing Workspace.
Persisted Instrument history validates record shape and scalar types before ordering or identity
comparisons. Dispatcher and direct parity execute identical planned-outcome validation through the
same executor.

### Immutable external canonical enum consumer specification

M0.6 does not treat schema discovery as the source of required external enum consumers. The executable reference validator owns an independent immutable consumer specification; validation requires exact equality between that specification, the dynamically observed schema bindings, and `canonical_enum_binding_manifest`. The dynamic collector therefore detects missing and additional bindings but cannot reduce the required set.

Every M0.4/M0.5-bound consumer has an immutable per-consumer assignment covering schema, field, field type, binding kind, source contract, JSON Pointer, registry reference, nullability, exact-field membership, and (for arrays) the complete array policy. Every such field must have exactly one external binding. Local scalar enum fallbacks and local array item registries are forbidden, including for all route, Account, Instrument, Catalog, Snapshot, Credential, TrustedExternalIdentity, and ProductCapabilities environment fields.

Canonical registry fingerprints are independently pinned once per registry identity. Validation requires the actual canonical content, the immutable fingerprint, and every manifest attestation to agree. Coordinated edits to schema and manifest, coordinated canonical-content and manifest-fingerprint drift, registry substitution, dual binding, binding removal, or array-policy weakening are machine-contract faults and produce `CONTRACT_INCONSISTENT`. With unchanged machine metadata, noncanonical persisted enum values remain `TRUSTED_CONTEXT_INVALID`.

### Deep-immutable canonical dependency roots

The reference validator deep-freezes independent expectations recursively: dictionaries become read-only mappings, lists become tuples, and nested consumer assignments and array policies cannot be modified. The complete canonical dependency root inventory independently pins M0.4 capability IDs, execution environments and both environment-capability key sets; M0.5 environment, market, instrument, pairing, Instrument schema, AssetReference, decimal and derivative contracts; Exchange Registry entries; all three lineage hash definitions; account operability policy; and secure-store locator grammar.

`ProductCapabilitiesProjection.environment` is a derived canonical projection of ordered `environment_id` values from M0.4 `/execution_environments`, not a direct M0.5 binding. Its resolver validates the closed M0.4 environment record shape and requires equality among those IDs, both M0.4 environment-capability key sets, and the M0.5 environment registry. Divergence is a machine-contract fault.

Every authority-bearing dependency requires the actual canonical JSON fingerprint, the independent deep-immutable fingerprint, and the M0.6 root-manifest attestation to match. Exchange Registry and the three hash definitions retain their specialized M0.6 attestations as additional checks. Account operability and secure-store validation consume only their validated resolvers. Coordinated canonical-source and M0.6 fingerprint changes therefore remain `CONTRACT_INCONSISTENT`; malformed persisted records under unchanged metadata remain `TRUSTED_CONTEXT_INVALID` or their ordinary readiness denial.

### Executable canonical dependency consumers

A canonical dependency root, identified by dependency ID, exclusively defines source-contract and JSON-Pointer authority. Every authority-bearing consumer is bound to exactly one dependency ID by the deep-immutable expected consumer-binding registry; validation requires exact equality among that registry, `canonical_dependency_consumer_bindings`, and each root manifest's closed consumer list. Consumers resolve through `resolve_canonical_dependency(dependency_id)`, which ignores mutable consumer-supplied pointer choices and returns a recursively frozen result.

Exchange Registry and hash-definition specialized references are attestations only. They must repeat the root dependency ID, source, pointer, result type, and fingerprint exactly; `canonical_exchange_entries()` and each named hash resolver obtain content from the dependency resolver rather than from those references. Specialized references therefore cannot redirect a consumer. Additional, uninventoried canonical-source properties are harmless while unused, but no alternate pointer can acquire authority by updating a specialized fingerprint or by synchronizing and rehashing persisted data.

### Exact ProductCapabilities projection and closed local authority

The global M0.4 capability-ID registry is vocabulary only; granted authority is the exact current-edition `capability_set`. The dependency inventory independently fingerprints `/current_product_edition`, `/ProductCapabilities/current_edition_capability_policy`, `/current_edition_signed_payload_policy`, and the global capability vocabulary. The executable policy resolver requires the three edition identities to equal `CRYPTOHUNTER_TESTNET_EDITION`, requires exact capability-set equality, follows the closed environment-capabilities reference, and closes feature flags, source, fail-closed policy, LIVE authority, and hash requirements.

`ProductCapabilitiesProjection` is an exact derived canonical projection. PAPER, TESTNET, and LIVE records bind all four fields—environment, execution flag, ordered allowed operations, and exact M0.4 edition identity. Any persisted mismatch is `TRUSTED_CONTEXT_INVALID`; M0.4 policy or attestation drift is `CONTRACT_INCONSISTENT`.

M0.6 local authority policies are independently deep-frozen for endpoint classes, authorization dependencies, Snapshot freshness, execution-readiness freshness, and current-edition execution pairs. Authority validators resolve these policies through the closed local-policy resolver rather than reading mutable contract values directly. Coordinated policy and machine-manifest weakening remains a machine-contract fault, while stale persisted Snapshot and readiness observations retain their ordinary domain denials.

### Independently immutable executable operation protocol

The executable operation protocol is independently and recursively frozen: it closes the complete operation set, exact request schemas and CoreHost authority, intents, exact ordered validator graphs, validator denials, per-operation allowed denials, denial registry, success/denial and special events, and forbidden operations. Mutable request schemas do not define their own authority.

`validate_executable_operation_protocol()` runs at the start of the common executor, before graph lookup, the first validator, or terminal success. Successful validation executes the immutable expected graph rather than the mutable contract list. Terminal success cannot precede mandatory lifecycle, authorization, strategy-operability, or readiness gates, and no validator may follow it. Coordinated graph, denial, request-schema, or event metadata drift remains `CONTRACT_INCONSISTENT`; a genuinely unknown operation remains `UNKNOWN_OPERATION`.

The preflight also validates untrusted executable metadata: the `REQUESTS` cache must exactly equal every immutable expected request schema, and every graph validator name must belong to the trusted read-only `EXPECTED_VALIDATOR_IMPLEMENTATIONS` dispatch table. Contract-fault and unknown-operation audit events are emitted from the immutable expected special-event mapping, so corruption of mutable special-event metadata cannot alter fail-closed evidence.


### Executable trust boundary

M0.6 treats the Python module source loaded from the verified repository or package, its module namespace and function objects, the Python interpreter and standard library, process memory and closure cells, both public entrypoint implementations, and the validator, resolver, and helper implementations as its trusted computing base. The deployment assumption is that this trusted code remains unchanged after verification.

The M0.6 machine contract, the canonical M0.4 and M0.5 JSON contracts, requests, persisted trusted-validation context, machine-readable schemas and manifests, the mutable request-schema cache, and external canonical pointers and metadata are untrusted data. `EXPECTED_EXECUTABLE_OPERATION_PROTOCOL` remains an independently authored, deeply frozen specification. Protocol preflight exactly compares the untrusted operation registry, request schemas, ordered graphs, denial registries, allowed denials, denial-code registry, event mappings, special events, and forbidden operations against that specification before executing a graph. `EXPECTED_VALIDATOR_IMPLEMENTATIONS` is a read-only trusted-code dispatch table, not an anti-tamper mechanism.

Arbitrary in-process Python execution, monkeypatching functions, replacing `__code__`, changing closure cells, rebinding dispatcher or executor globals, memory corruption, and interpreter compromise are explicitly outside this machine-data threat model. Any such capability is already a compromise of the trusted computing base and cannot be independently detected by code executing in that same compromised process. In-process self-attestation is therefore not a security boundary.

After protocol validation, the executor uses the exact graph from the immutable expected protocol, invokes the trusted validator mapped to each name, validates denial membership and planned outcomes, and maps machine-data corruption to `CONTRACT_INCONSISTENT`. `dispatcher()` and `run_direct_call_graph()` are functionally equivalent public paths to this executor; their parity is a functional requirement rather than protection against process-code replacement.

Runtime code-integrity enforcement, if required, must be implemented by an external bootstrap or deployment trust mechanism that validates signed/hash-pinned artifacts before importing the application module. M0.6 does not implement package signing, a bootloader, runtime anti-tamper, or an external verifier. M0.6 is closed, M0.7 is not started, and LIVE execution remains policy-disabled and unreachable in the current edition.

### Fail-closed decision boundary

Every decision event—including ordinary success, ordinary denial, contract corruption, and unknown-operation denial—is selected exclusively from the independently deep-frozen `EXPECTED_EXECUTABLE_OPERATION_PROTOCOL`. The mutable success, denial, and special-event structures in the machine contract are untrusted attestations validated by protocol preflight only. Neither the contract-fault path nor the unknown-operation path reads machine event metadata while constructing its response, so structurally malformed or missing event registries produce the immutable `STRATEGY_ROUTING_CONTRACT_INCONSISTENT` event rather than an exception.

The outer `operation` value is untrusted and is type-checked before any mapping lookup or membership operation. A non-string value is denied as `UNKNOWN_OPERATION`, emits `STRATEGY_ROUTING_UNKNOWN_OPERATION_DENIED`, and is normalized to JSON `null` in the deeply frozen decision payload; an unknown string is retained as supplied. A malformed `request.operation` remains request data for a known outer operation and therefore receives the ordinary `REQUEST_SCHEMA_INVALID` denial. Dispatcher/direct-call parity applies to all these paths as a functional requirement within the existing trusted computing base.

## M0.6 source/execution projection migration

`InstrumentProjection` now exact-projects `source_exchange_id` and contains neither execution `exchange_id` nor `environment`. Source validation covers the source exchange, accepted Catalog, and metadata lineage. Execution validation separately covers ExchangeAccount, ExecutionRoute, their exact execution environment/backend, and authorization. Source adapter equality with execution backend is not required. The same-Workspace invariant remains mandatory; a foreign Instrument returns `TRUSTED_CONTEXT_INVALID`. TradingUniverse remains an ExchangeAccount child and references, but never creates or mutates, Workspace Instruments.

Canonical TradingUniverse source references resolve `WorkspaceCatalogProjection → AcceptedSourceCatalogSnapshot`; the environment-bound `InstrumentCatalogProjection` is legacy migration-blocked and cannot mint source membership. PAPER routes additionally require explicit source-product permission, rather than workspace and market type alone.
