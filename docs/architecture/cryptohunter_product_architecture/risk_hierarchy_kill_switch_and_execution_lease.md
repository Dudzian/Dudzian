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
