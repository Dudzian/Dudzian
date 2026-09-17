# M0.7 — Commands, Events, Order Lifecycle and Idempotency

**Status: closed.** Źródłem prawdy jest plik machine-readable JSON; ten dokument objaśnia kontrakt. M0.7 jest wyłącznie kontraktem architektonicznym i nie implementuje adaptera, sieci, storage, ledgeru, risk engine, `ExecutionLease`, autoryzacji ani realnego tradingu.

## Wynik audytu

M0.2 ustanowiło trwałe `OrderIntent` (`oint`), `Order` (`ord`), `Fill` (`fill`) i `AuditEvent` (`audit_event_id`, `evt`), dlatego M0.7 używa tych identity zamiast tworzyć konkurencyjne prefiksy. `command_id` jest odrębną identity żądania/idempotency: polecenie i agregat Order mają różne lifecycle. Venue order/trade ID oraz deterministic client order ID są zewnętrznymi korelacjami, nie durable product identity.

Historyczny runtime miesza proste statusy `filled`, wartości `float`, synchroniczny submit oraz adapterowe order ID. Są one dowodem stanu zastanego, ale nie authority architektury docelowej. M0.6 celowo wykluczało submit/cancel/replace, więc ich zamkniętym właścicielem jest M0.7.

## Zamknięty command model

Registry zawiera dokładnie:

1. `SUBMIT_ORDER` — waliduje i utrwala plan jednego Order z jednego OrderIntent;
2. `CANCEL_ORDER` — planuje anulowanie istniejącego nieterminalnego Order;
3. `REPLACE_ORDER` — planuje zmianę i z góry przydziela odrębny `replacement_order_id`.

Każde polecenie ma zamknięty, operacyjnie swoisty schema: command/idempotency identity, authority context, environment, workspace, portfolio, exchange account, opcjonalne źródło StrategyInstance, instrument, ExecutionRoute, correlation/causation oraz pola właściwe operacji. Nie ma `payload: object`. Nieznane/brakujące pola, float, NaN/Infinity, niekanoniczny decimal/timestamp lub niedozwolony null oznaczają `MALFORMED_REQUEST`.

Immutable outcome to `ACCEPTED`, `REJECTED` albo `REPLAYED`. Acceptance oznacza wyłącznie autoryzowany, utrwalony planned effect. Nie oznacza dispatch, exchange acknowledgement, fill ani wykonania transakcji.

## Idempotency i retry

`command_id` jest canonical idempotency identity, a `idempotency_key` musi być mu równy. Scope to `(workspace_id, environment, exchange_account_id, command_id)`. Fingerprint SHA-256 obejmuje operation type i całe kanoniczne zamknięte żądanie poza dokładnie jednym polem `correlation_id`; `causation_id`, `command_id` i `idempotency_key` pozostają semantyczne, z jednoznacznym JSON/UTF-8/NFC, canonical decimal string i UTC.

Rezerwacja identity, fingerprint i immutable outcome następuje atomowo **przed** external side effect. Ten sam scope + fingerprint zwraca byte-equivalent zapisany outcome jako replay bez nowego eventu, transition, dispatch ani side effect. Ten sam scope + inny fingerprint to `IDEMPOTENCY_CONFLICT`, bez mutacji. Timeout lub nieznany wynik nigdy nie wywołuje blind resubmit: Order przechodzi do `RECONCILIATION_REQUIRED`, a reconciler szuka trusted venue facts po stabilnym deterministic `client_order_id`.

Identyczny event ID/fingerprint, acknowledgement lub fill jest ignorowany jako udany replay. Konflikt `audit_event_id`/fingerprint daje zwykły fail-closed wynik `EVENT_IDENTITY_CONFLICT`, nie `CONTRACT_INCONSISTENT`. Fill deduplikuje durable `fill_id` oraz scoped `(environment, exchange_account_id, exchange_id, venue_trade_id)`; konflikt wymaga reconciliation.

## Order lifecycle

Stan początkowy to `PLANNED`. Zamknięte stany:

- aktywne: `PLANNED`, `SUBMISSION_PENDING`, `ACKNOWLEDGED`, `PARTIALLY_FILLED`, `CANCEL_PENDING`, `REPLACE_PENDING`, `RECONCILIATION_REQUIRED`;
- terminalne: `REJECTED`, `FILLED`, `CANCELLED`, `EXPIRED`, `REPLACED`.

Plan tworzy command handler; dispatcher tworzy submission pending; tylko adapter/trusted venue fact potwierdza acknowledgement/rejection/cancel/replace/expiry; fill ingestor prowadzi partial/filled; reconciler rozstrzyga unknown outcome. Cancel/replace rejection wykonawczo rozwiązuje `RESTORE_PRE_REQUEST_STATE` wyłącznie do zapisanego `ACKNOWLEDGED` albo `PARTIALLY_FILLED`, nigdy do stanu podanego przez caller. Reconciliation rozwiązuje zamknięte trusted fact kinds do legalnych stanów `ACKNOWLEDGED`, `REJECTED`, `PARTIALLY_FILLED`, `FILLED`, `CANCELLED`, `REPLACED` albo `EXPIRED`; brak lub obcy fact failuje closed. Replace confirmation terminalizuje pierwotny Order jako `REPLACED`; replacement ma inną Order identity. Każdy transition w JSON ma exact owner, source set, event i target. Terminal state nie ma transition wstecz.

## Fill invariants

Wszystkie ilości i ceny są canonical decimal strings zgodne dokładnie z M0.5 (zero jako `0`, bez fractional trailing zeros) i są przeliczane przez exact integer/Fraction arithmetic niezależne od globalnego Decimal context; float jest zakazany. Cumulative executed quantity jest sumą unikalnych fills i nie może maleć ani przekroczyć order quantity. Remaining to quantity minus cumulative. Średnia cena zachowuje dokładny notional numerator i cumulative denominator; canonical decimal jest emitowany tylko dla skończonego rozwinięcia, a nieskończone nie otrzymuje arbitralnego M0.7 rounding i pozostaje null do późniejszej prezentacji. `PARTIALLY_FILLED` zachodzi dokładnie dla `0 < cumulative < quantity`, a `FILLED` dokładnie dla równości. Overfill jest odrzucany i wymaga reconciliation.

## Immutable events i ordering

Event jest immutable fact z exact envelope i exact type, canonical `audit_event_id`/`evt` identity, Order aggregate identity/version, correlation/causation/command identity, pełnym environment/account/exchange/instrument/route scope, UTC time, fingerprint oraz safe payload związanym z exact `event_schema_registry` (exact fields, types, nullability i required scope) bez sekretów, credentiali, endpointów i raw venue response.

Fingerprint eventu obejmuje cały semantyczny immutable envelope poza samym `event_fingerprint_sha256`; zgodny payload przy różnicy order/account/environment lub dowolnego innego fact field nie jest replayem. Scope jest porównywany z istniejącym Order i mismatch daje `ORDER_SCOPE_MISMATCH`.

Per-Order `aggregate_version` zaczyna się od 1 i rośnie dokładnie o jeden dla nowego zaakceptowanego faktu. Identyczny duplicate nie podnosi wersji. Inny event o wersji starej to `STALE_EVENT`; luka to `EVENT_VERSION_GAP` i reconciliation. Scope mismatch, niemożliwy transition i terminal regression failują closed.

## PAPER / TESTNET / LIVE

Jeden core command/order/event model obowiązuje wszystkie trzy środowiska. PAPER może użyć deterministycznego lokalnego adaptera, lecz zachowuje idempotency i lifecycle. TESTNET wykonuje private external effects dopiero po upstream authority/readiness. LIVE jest first-class target, ale CURRENT EDITION pozostaje policy-disabled przez M0.4/M0.6; M0.7 go ani nie odblokowuje, ani nie uznaje za zakazany na zawsze. Przyszła skoordynowana policy może go uruchomić bez zmiany lifecycle.

Venue, endpoint, credential, poprawne polecenie ani jego acceptance nie nadają authority. Nie istnieje TESTNET→LIVE fallback ani implicit account/route. Side effect wymaga kolejno authority M0.4, exact M0.5 account/exchange/instrument, M0.6 strategy/route/readiness, przyszłego M0.9 `ExecutionLease`, a następnie atomowej rezerwacji M0.7. `risk_ok=true` nie zastępuje przyszłego lease.

## Fail-closed i trust boundary

Rozłączne wyniki obejmują `MALFORMED_EVENT` dla wadliwego envelope/payload, `EVENT_FINGERPRINT_MISMATCH` dla well-shaped lecz niezgodnego full-fact fingerprintu, malformed command, unauthorized, invalid transition, idempotency conflict, replay success, trusted-context failure, external unknown/reconciliation oraz machine inconsistency. `CONTRACT_INCONSISTENT` jest wyłącznie drift/corruption machine contract, nigdy zwykłym domain denial.

JSON jest untrusted attestation. Reference tests posiadają deep-frozen immutable expected command, complete order-lifecycle, event-envelope/event-schema i cross-dependency protocols, exact command field types/ID kinds/enums/nullability/constraints, event schema fingerprint, exact registries/transitions/mappings i niezależne fingerprinty roots M0.2–M0.6. Skoordynowane osłabienie mutable JSON failuje closed. To nie jest fałszywy in-process anti-tamper mechanizm; produkcyjny trust i persistence należą do późniejszych milestone'ów.

## Closure

Audit, registry, exact state machine, pre-side-effect idempotency, monotonic event/fill rules, environment boundary, canonical dependency bindings i executable validators są zamknięte. Brak blockerów M0.7. M0.8–M0.11 pozostają świadomie poza zakresem, a przyszły M0.9 `ExecutionLease` jest zaprojektowaną dependency, nie defektem closure.

## Corrective closure: canonical immutable economic Fill fact

M0.7 zamyka teraz Fill jako pełny immutable economic fact, a nie jedynie parę quantity/price. Exact schema obejmuje `fill_id`, parent `order_id`, pełny workspace/portfolio/environment/exchange-account/exchange scope, `instrument_id` wraz z dodatnią `instrument_metadata_version`, `execution_route_id`, `venue_trade_id`, `side`, exact `executed_quantity`, exact `execution_price`, odrębny ekonomiczny `executed_at_utc`, `fee_kind`, `fee_quantity`, opcjonalny według zamkniętej reguły `fee_asset_reference` oraz `fill_fingerprint_sha256`. `side` jest zapisany bezpośrednio, więc accounting nigdy nie zgaduje BUY/SELL z aktualnego mutable Order.

`instrument_metadata_version` wiąże Fill z dokładnie jedną pełną historyczną wersją Instrument w Core-owned trusted `instrument_history_by_id`. Ta wersja musi zgadzać workspace, exchange i environment oraz dostarcza historyczne base/quote/settlement references. Nie ma current-version fallback, kopiowania całego instrumentu ani parsowania symbolu. Fee używa dokładnego M0.5 asset-reference value object (`venue_asset_code`, `canonical_display_code`, `asset_namespace`, `mapping_status`), nie nieistniejącego durable Asset ID. Tylko `EXACT` i `EXPLICIT_ALIAS` są accounting-authoritative; `AMBIGUOUS` i `UNKNOWN` failują closed.

Fee ma zamknięte dwie semantyki. `NONE` wymaga dokładnie `fee_quantity == "0"` i `fee_asset_reference == null`. `CHARGE` wymaga dodatniej canonical quantity i nie-null trusted asset reference. Third-asset fee jest legalne i zachowane bez defaultu do quote/base. Maker rebate jest w tym kontrakcie jawnie unsupported i fail-closed: M0.5 zabrania wartości ujemnych, a przyszłe wsparcie wymaga nowego jawnego economic fact kind, nie signed fee hack.

Fingerprint SHA-256 obejmuje każde pole Fill poza samym fingerprintem: identity i Order parent, cały scope, instrument i metadata version, route, venue trade, side, quantity, price, execution time oraz pełne fee semantics/reference. Canonicalization używa UTF-8, NFC, sorted JSON keys, canonical separators, M0.5 decimals i canonical UTC RFC3339 `Z`. Ten sam `fill_id` i fingerprint jest replayem bez nowego fact; różnica dowolnej ekonomii daje `FILL_IDENTITY_CONFLICT`, nie `CONTRACT_INCONSISTENT`. Scoped venue key `(environment, exchange_account_id, exchange_id, venue_trade_id)` deduplikuje dokładnie tę samą ekonomię także przy drugim `fill_id`, ale konfliktuje przy dowolnej różnicy.

Core-owned trusted context `fills_by_id` mapuje key równy `fill.fill_id` do dokładnie jednego exact-schema fact z poprawnym fingerprintem. Brak nie oznacza zero ani no-fee. Persistence pozostaje M0.11. `ORDER_PARTIALLY_FILLED` i `ORDER_FILLED` zachowują bezpieczny payload referencyjny; przed lifecycle effect validator rozwiązuje Fill, porównuje Order oraz cały envelope scope i venue trade, waliduje fingerprint, a cumulative quantity oblicza z unikalnej accepted Fill history zamiast ufać caller-provided cumulative. `executed_at_utc` jest czasem transakcji; event `occurred_at_utc` pozostaje czasem zdarzenia i nie zastępuje czasu ekonomicznego.

Cross-contract attestation wiąże dodatkowo wyłącznie używane M0.5 roots: `/asset_reference_contract` i `/instrument_contract/trusted_history_contract`. Machine JSON nadal nie jest samodzielną authority: reference tests mają niezależnie zapisane, deep-frozen expected Fill contract/schema i dependency fingerprints oraz mutation tests. Po tej korekcie M0.7 pozostaje `closed`, a ujawniony blocker pełnego immutable Fill + fee dla M0.8 jest usunięty.

Executable binding rozwiązuje `instrument_metadata_version` wyłącznie z nominalnego kontekstu, który wcześniej przeszedł pełną canonical walidację M0.5 `instrument_history_by_id`. Po tej upstream granicy M0.7 sprawdza exact record field set, key/record identity, rosnące unikalne wersje, identity continuity i exact Fill scope; nie przedstawia tych lokalnych kontroli jako zamiennika pełnego M0.5 validatora. Missing, duplicate lub newer/current-only context daje `TRUSTED_CONTEXT_FAILURE`; nie ma fallbacku ani symbol parsing. `/instrument_contract/record_fields` pozostaje exact-bound obok trusted-history contract.

Filled-event cumulative validation now requires an explicit complete accepted Fill-ID sequence; the current Fill must occur in that sequence and every fact must resolve with the full event scope. Both event validation and fill arithmetic share one durable/external dedupe helper, so an identical second identity for one venue trade counts once while conflicting economics yields `FILL_IDENTITY_CONFLICT`. Quantitative overfill is separately `FILL_PROGRESSION_CONFLICT`. JSON object key order is never data authority because exact field sets are validated and fingerprints retain sorted-key canonicalization. `nullable_fields` plus conditional `NONE`/`CHARGE` rules are the sole fee-asset nullability authority; the contradictory nested `nullable: false` marker was removed.

Composite trust boundary rozdziela trzy poziomy. Raw Full Fill przechodzi wyłącznie structural schema, fee-scope i fingerprint validation. Trusted economic Fill powstaje dopiero przez jeden composite resolver, który dodatkowo wymaga exact historical Instrument z nominalnego `M05PrevalidatedInstrumentHistory`; ten kontekst może pochodzić wyłącznie z udanego canonical M0.5 trusted-history validation boundary, a raw mapping ani caller boolean nie nadają authority. M0.7 nie utrzymuje konkurencyjnej, częściowej implementacji całego M0.5 record validatora: lokalnie dowodzi jedynie requested identity/version, ordered uniqueness, identity continuity, Fill scope oraz defensywnie wymaga exchange namespace dla settlement reference. Fee `CHARGE` analogicznie wymaga `fee_asset_reference.asset_namespace == Fill.exchange_id`; third-asset code nadal jest legalny.

Complete accepted history jest nominalnym Core-owned `accepted_fill_ids_by_order_id`, wyprowadzonym wyłącznie z accepted M0.7 aggregate history, a nie listą caller/safe-payload. Event resolver pobiera całą canonical sequence dla exact `event.order_id`, composite-resolve'uje każdy Fill i liczy cumulative wspólnym dedupe helperem. Jeżeli kolejny `fill_id` oznacza już zaakceptowany identyczny scoped external trade, nie trafia do canonical sequence i jego event zwraca `REPLAY_SUCCESS`, bez nowego lifecycle ani economic effect. Missing, raw, malformed, wrong-order lub skrócony caller context failuje `TRUSTED_CONTEXT_FAILURE`.

Canonical Core projection sama musi być wolna od durable i scoped-external duplicate effects: po composite resolution wspólny dedupe helper musi zachować dokładnie tyle facts, ile zawiera sequence, inaczej kontekst jest `TRUSTED_CONTEXT_FAILURE`. `REPLAY_SUCCESS` dotyczy wyłącznie incoming candidate spoza canonical sequence; przed replay validator wiąże jego pełny scope i payload venue trade oraz wymaga, aby payload cumulative było dokładnie istniejącym canonical cumulative. Nowy unique candidate spoza projection nigdy nie jest automatycznie dopisywany.

## Wersjonowanie strukturalnego Full Fill

Historyczny kontrakt v1 pozostaje niezmiennie dostępny pod `/fill_contract`. Kanoniczny rejestr wersji znajduje się pod `/fill_contract_versions`: v1 wskazuje historyczny kontrakt, natomiast v2 opisuje 20-polowy, wyłącznie strukturalny kandydat implementacyjny. Dispatcher wybiera wersję tylko przez dokładny zamknięty zbiór pól; każdy inny kształt jest `MALFORMED_FILL`.

Fingerprint v2 obejmuje 19 pól wejściowych i używa domeny `cryptohunter.m0.7.full_fill.v2` oraz preimage `UTF-8(domain || 0x00 || NFC(canonical JSON projection))`. Pole ASCAT ma wyłącznie znaczenie składniowe i nie nadaje zaufania. Walidacja fee v2 jest shape-only, z `namespace_comparison = NONE`. `FullFillAuthority` i semantic Fill admission pozostają `NOT_AVAILABLE`; surowy strukturalny v2 nie jest źródłem dla M0.8.
