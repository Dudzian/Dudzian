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
