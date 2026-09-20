# M0.5 — executable semantic contract rejestru entitlementów

Status: **FROZEN**  
Target: `ENTITLEMENT_REGISTRY_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN`  
Zakres: kontrakt portu, bez implementacji storage i bez wyboru sterownika PostgreSQL.

## 1. Granice authority

`EntitlementRegistryProvider` jest portem runtime dla semantic issuer. Może czytać
stan i kompletną zachowaną historię oraz próbować pierwszego bindu. Nie może
provisionować, revoke'ować ani supersede'ować entitlementu. Te operacje należą
do niezależnej deployment/provisioning security authority i osobnego
`EntitlementProvisioningAdminProvider`. Rozdział portów jest zarazem wymaganiem
oddzielnych, least-privilege credentials przyszłego adaptera.

Registry jest authority dla mapowania handle, lifecycle, binding tuple,
monotonicznej rewizji i historii. AccountGenesis/CHA jest authority dla
`issuance_attempt_id`. Semantic issuer przygotowuje `root_proof_id` przed CAS.
Signing provider jest jedynym właścicielem operacji podpisu, a authenticated
issuer history przechowuje byte-identical signed RootProof. Registry zachowuje
`root_proof_id`, signing credential/version i identity podpisanego requestu,
ale nie udaje właściciela signature ani signed-proof bytes.

## 2. Subject, handle i authoritative identity

Publiczne read API przyjmuje wyłącznie zamknięty `RegistrySubject`:
`(lookup_handle, environment, trust_domain)`. Handle jest caller-supplied lookup
coordinate, nie entitlement ID i nie dowód autoryzacji. Provisioning instaluje
immutable mapping o kardynalności jeden handle do jednej lineage jednego
`bootstrap_entitlement_id`; ponowne mapowanie handle na inny ID jest zabronione.

Reverse cardinality jest również zamknięta. Authority identity key to exact
`(environment, trust_domain, product_scope, bootstrap_entitlement_id)` i w
registry może wskazywać dokładnie jeden `RegistrySubject` oraz jedną physical/
semantic history lineage. Reviewed alias model nie istnieje. Provisioning tego
samego key pod drugim handle kończy się typed `CONFLICT` przed utworzeniem
current/history. Zatem jeden authoritative entitlement nie może mieć dwóch
niezależnych UNBOUND→BOUND winnerów. Ten globalny w obrębie identity key
single-use invariant obejmuje wszystkie generation: superseding ACTIVE/UNBOUND
nie pozwala na drugi bind, jeżeli retained lineage zawiera wcześniejszy BOUND.

Semantic database subject stanowi dokładny `RegistrySubject` oraz jego
authority-owned lineage. `EntitlementIdentity` zawiera authoritative
`bootstrap_entitlement_id`, positive `entitlement_generation`, environment,
trust domain, product scope i stałą akcję `ACCOUNT_GENESIS_BOOTSTRAP`.
Environment/trust domain są zarówno lookup namespace, jak i częścią identity;
niezgodność fail-closed. TEST nie może odnaleźć ani autoryzować PRODUCTION.

Istniejące frozen formaty pozostają obowiązujące: `ent_`, `ago_`, `acct_`,
`rpa_` i `rpf_` plus canonical lowercase UUIDv7; SHA-256 to 64 lowercase hex;
generation, key version i revision są exact dodatnimi `int` (`bool` odpada).
Pozostałe opaque identity/reference są exact niepustymi `str`; kontrakt nie
zgaduje dla nich nowego UUID ani prefiksu.

## 3. Dwa niezależne monotoniczne wymiary

`entitlement_generation` wersjonuje provisioning lineage. Zmienia ją wyłącznie
admin supersession, dokładnie `n -> n+1`. `UNBOUND -> BOUND` **nie zmienia**
generation.

`authoritative_state_revision` jest **per-RegistrySubject** registry-owned
mutation fence. Genesis danego subjectu ma revision 1, a każda następna zmiana
tego samego subjectu ma revision poprzednika + 1. Mutacje innego subjectu nie
zmieniają current revision tego subjectu. Nie istnieje w tym kontrakcie drugi,
registry-globalny sequence number. Provisioning, bind, lifecycle transition i
instalacja superseding generation zwiększają subject revision. Caller nie
przekazuje successor revision. Wall clock, insertion order i proces nie są
authority. Generation i revision nie są zamienne ani porównywalne semantycznie.

## 4. Lifecycle i binding

Jedyny closed lifecycle enum to `ACTIVE`, `REVOKED`, `SUPERSEDED`. Legalne
przejścia admin to `ACTIVE -> REVOKED` oraz `ACTIVE -> SUPERSEDED`; terminalne
stany nie wracają do ACTIVE. Supersession może objąć UNBOUND lub BOUND. Tworzy
ACTIVE/UNBOUND następną generation, a poprzednia generation staje się
SUPERSEDED i zostaje w historii. Tylko ACTIVE może wykonać pierwszy bind.
Historyczny BOUND pozostaje czytelny nawet po revoke/supersession, lecz nie
zezwala to na nowy bind.

`UnboundBinding` jest closed, bez payloadu. `BoundBinding` jest exact tuple:

* logical operation, account i canonical genesis request fingerprint;
* entitlement generation i authority-owned issuance attempt;
* requester principal, requester key ID/version;
* provisioning principal, claimant key ID/version;
* signed request payload digest oraz immutable canonical-bytes reference;
* issuer-prepared root proof ID i issuer signing credential ID/version.

Nie jest dict. Entitlement ID pozostaje w otaczającym `EntitlementIdentity`, a
environment/trust domain w subject/identity; nie dubluje się ich w tuple.
Signature i signed RootProof bytes należą do signing/history authorities.
Provisioning principal oraz claimant key ID/version są registry-owned anchor z
`EntitlementProvenance`; BOUND musi być im dokładnie równy. Requester identity
pozostaje osobną domeną `RequesterCredentialRegistry` i nie jest przypisywana
do provisioning provenance.

## 5. Authoritative record i CAS

`AuthoritativeEntitlementState` oddziela exact subject, immutable entitlement
identity/generation, immutable provisioning provenance, lifecycle, closed
binding, current authority revision i predecessor revision. Dataclasses są
frozen/slots, walidują głęboko exact runtime types i odrzucają subclass/dict.
Dataclass equality jest canonical semantic equality.

`BindPredecessor` jest explicit CAS fence:
`(bootstrap_entitlement_id, entitlement_generation,
authoritative_state_revision, lifecycle, binding_identity_digest_sha256,
provisioning_principal_id, claimant_key_id, claimant_key_version)`.
Nie jest całym arbitralnym obiektem.

`AdminPredecessor` niesie exact subject, pełną current entitlement identity,
provenance, lifecycle, subject revision i binding digest. Dzięki temu
supersession może wykazać zachowanie entitlement ID, environment, trust domain,
product scope i intended action, zamiast zgadywać je z niepełnego CAS tokenu.

`BindRequest` zawiera subject, predecessor i exact attempted `BoundBinding`.
Nie ma pola successor revision. Entitlement ID występuje tylko w odczytanym
predecessorze, więc caller nie może podstawić go jako successor. Issuer musi
przed CAS przygotować authority-owned opaque `root_proof_id` i wybrać exact
signing credential/version. CAS zapisuje tę deterministyczną decision identity;
po commit signing layer podpisuje exact committed RootProof, a authenticated
history utrwala signed bytes. Jeśli signing/history nie można ukończyć,
registry BOUND nadal jest jedynym winnerem i recovery musi dokończyć ten sam
proof, nigdy utworzyć drugi.

Binding fence używa jawnego preimage:
`ASCII("ENTITLEMENT_REGISTRY_BINDING_IDENTITY_V1") || 0x00 || UTF-8(canonical_json)`.
`canonical_json` jest istniejącym repo kernel: posortowane klucze, separatory
`,`/`:`, UTF-8, `ensure_ascii=false`, `allow_nan=false`. Payload zawiera każde
pole closed bindingu, a enum jest kodowany przez jego frozen string value.
Interpreter-dependent `repr` jest zabronione. Przed kodowaniem wykonywany jest
exact/deep validated snapshot.

## 6. Wynik, replay i lost response

`BindResult` nigdy nie jest bool. Closed outcomes to:

* `NEW_BIND_COMMITTED`;
* `EXACT_REPLAY`;
* `CONFLICT_BOUND_TO_DIFFERENT_TUPLE`;
* `STALE_PREDECESSOR`;
* `NOT_FOUND`;
* `INACTIVE_REVOKED` / `INACTIVE_SUPERSEDED`;
* `CORRUPT` / `UNAVAILABLE`;
* `RETRYABLE_SERIALIZATION_FAILURE`.

Database serialization retry jest transport/concurrency outcome, a nie stale
semantic predecessor. Exact replay wymaga equality wszystkich pól
`BoundBinding` oraz otaczającego entitlement ID/generation/scope. Inny attempt,
proof ID, key version, request identity, operation/account albo fingerprint to
conflict, nie replay.

Po commit i utracie odpowiedzi identyczny request zwraca `EXACT_REPLAY` oraz
dokładnie ten sam authoritative BOUND record (ten sam attempt i root proof ID,
bez nowej revision). Issuer odzyskuje byte-identical proof z authenticated
history po zapisanej identity. Registry nie mintuje kolejnego winnera i nie
obiecuje być właścicielem podpisanych bajtów.

Exact replay jest historyczną decyzją, a nie testem current row. Shared
`resolve_bind_request(validated_retained_history, request)` odnajduje pierwsze
przejście UNBOUND→BOUND i uznaje replay wyłącznie, gdy jednocześnie:

* `request.subject == historical_predecessor.subject`;
* `request.expected == predecessor_for(historical_predecessor)`;
* `request.attempted_binding == historical_committed_successor.binding`.

Zgodność tuple przy innym entitlement ID, revision, lifecycle, binding digest
lub claimant anchor predecessor oznacza `STALE_PREDECESSOR`, nigdy replay.
Inny tuple po istnieniu winnera oznacza
`CONFLICT_BOUND_TO_DIFFERENT_TUPLE`, także po revoke lub supersession.
`BindResult.authoritative_state` dla `EXACT_REPLAY` jest dokładnie historycznym
ACTIVE/BOUND recordem z bind revision, nie current REVOKED/SUPERSEDED/nowej
generation head. Późniejsze admin mutations nie zmieniają lost-response
promise i nie mintują revision podczas retry.

Publiczny `BindResult` ma zamkniętą macierz payloadu: wyłącznie
`NEW_BIND_COMMITTED` i `EXACT_REPLAY` wymagają exact BOUND state. Każdy failure
(`CONFLICT`, `STALE`, `NOT_FOUND`, inactive, `CORRUPT`, `UNAVAILABLE` albo
serialization failure) wymaga `authoritative_state=None`; failure nie jest
alternatywnym kanałem ujawniania arbitralnego current state.

## 7. Reads, NOT_FOUND i historia

`authoritative_state(subject)` zwraca `RegistryReadResult`. `FOUND` wymaga exact
state. `NOT_FOUND`, `CORRUPT` i `UNAVAILABLE` nie zawierają stanu. NOT_FOUND nie
jest UNBOUND i nigdy nie stanowi reconciliation evidence.

Nie istnieje dwuznaczne `historical_state(subject, generation)`.
`state_at_revision(subject, authoritative_state_revision)` adresuje exact
historyczny record po authority revision. Generation odczytuje się z recordu;
nie jest ona indeksem mutation history. `retained_history(subject)` zwraca
strictly revision-ordered, complete subject mutation lineage zakończoną exact
current subject revision oraz `retained_from_authoritative_state_revision`.
BOUND starej generation musi pozostać queryable zgodnie z retention policy.

Historia jest pełną state-machine lineage od canonical genesis: generation 1,
ACTIVE/UNBOUND, subject revision 1, predecessor `None`. Każdy następny record
wskazuje dokładnie poprzedni record przez `predecessor_revision`; przy modelu
per-subject ma również revision poprzednika + 1. Walidator odrzuca brakujące
ogniwa, zmianę subject/entitlement/environment/trust/product/action, zmianę
provenance wewnątrz generation, nielegalny lifecycle, BOUND→UNBOUND, zmianę
BOUND winnera i nielegalny skok generation. Następna generation wymaga
SUPERSEDED predecessor i zaczyna ACTIVE/UNBOUND.
Record różniący się wyłącznie revision/predecessor jest semantic no-op i jest
odrzucany: revision może reprezentować wyłącznie rzeczywistą authority mutation.

`HistoricalStateResult` jest self-bound do requested `RegistrySubject`. Dla
FOUND zwracany state musi mieć ten subject i dokładnie requested revision.
`current_authoritative_state_revision` w nim i w retained history zawsze znaczy
current **subject** head revision.

Metadata rewizji są outcome-specific. FOUND wymaga exact positive current head
i retained-from revision. NOT_FOUND, CORRUPT i UNAVAILABLE wymagają obu pól
`None`; provider nie fabrykuje wartości 1 ani requested revision. Dla całkiem
nieznanego subjectu dowolna requested positive revision zwraca typed NOT_FOUND.
Dla istniejącego subjectu complete revisions 1..N muszą istnieć: brak K w tym
zakresie jest CORRUPT, a nie NOT_FOUND. `RetainedHistoryResult(NOT_FOUND)` ma
puste states i oba revision metadata `None`.

## 8. Reconciliation i anti-staleness

Sam stan UNBOUND nie tworzy `RootProofIssuanceReconciliationEvidenceV1`.
Reconciliation source musi uwierzytelnić registry/history authority i związać
obserwację z exact `RegistrySubject`, canonical `product_scope`, entitlement
ID/generation, old issuance attempt, operation,
account, request fingerprint, initial binding identity, exact current authority
revision oraz retained-history fence. Następnie musi wykazać z kompletnej
retained lineage brak jakiegokolwiek późniejszego BOUND dla starego attemptu.

`RootProofIssuanceReconciliationEvidenceV1.authoritative_state_revision`
zapisuje current per-subject head revision entitlementu w chwili obserwacji.
Predicate porównuje ją wyłącznie z
`RetainedHistoryResult.current_authoritative_state_revision` tego samego exact
subjectu. `AuthoritativelyUnboundQuery.subject` musi być równy
`RetainedHistoryResult.subject`; inny handle, environment albo trust domain
zawsze daje wynik false, nawet przy takim samym opaque entitlement ID. Query
niesie również exact non-empty `product_scope`, który musi być równy
`current.identity.product_scope`. Positive predicate wykazuje więc pełny key:
`(subject.environment, subject.trust_domain, product_scope,
bootstrap_entitlement_id)` oraz generation. Syntaktycznie poprawny, lecz inny
produkt daje false, nie construction error.
`CURRENT_AUTHORITY_REVISION_AND_RETAINED_HISTORY_FENCE` oznacza:
history kończy się na tej current subject revision, obejmuje kompletny
predecessor chain od canonical genesis, a exact generation była UNBOUND w badanej
rewizji i nie ma późniejszego BOUND winnera. Stale snapshot, brak historii,
NOT_FOUND, timeout, wall clock lub nieuwierzytelniona projekcja oznaczają
OUTCOME_UNKNOWN/fail-closed, nigdy AUTHORITATIVELY_UNBOUND.

Frozen RootProof signed payload już używa pola `product_scope`; nie powstaje
żaden alias produktu. `RootProofIssuanceReconciliationEvidenceV1.required_fields`
oraz replacement `evidence_exact_binding_fields` zawierają tę samą canonical
wartość `product_scope`, dzięki czemu składnik authority identity nie ginie na
drodze issuance → registry → reconciliation evidence.

PRODUCTION_LOCAL CHA używa dokładnie tej samej wartości `product_scope` w
`AttemptAuthorization`, idempotency preimage, reservation JSON,
`AttemptIdentity` payload/digest, `AuthoritativeUnboundEvidence` i replacement
relation. Stable predecessor binding porównuje environment, trust domain,
product, operation/account/request, entitlement ID/generation i initial binding.
Zmiana obowiązkowego durable shape jest fail-closed: CHA store metadata schema
została podniesiona z 3 do 4, a `AuthoritativeUnboundEvidence.schema_version`
z 1 do 2. Nie istnieje silent migration; baza z wcześniejszą wersją jest
odrzucana jako unsupported. Jest to reviewed CURRENT_TREE_ONLY pre-runtime
schema cut, a nie deklaracja kompatybilności danych produkcyjnych.
Provider implementation identity pozostaje `cha-attempt-sqlite-v1`, ponieważ
identyfikuje rodzinę implementacji, a nie wire/storage schema; kompatybilność
persistencji jest autorytatywnie rozstrzygana przez exact `store_metadata`
schema identity/version i dlatego nie może zostać ukryta przez tę nazwę.

## 9. Provisioning i supersession

Admin `ProvisionEntitlementRequest` instaluje przed pierwszym kontem exact
subject, authority-minted entitlement ID/generation 1, ACTIVE/UNBOUND,
provisioning principal/claimant anchor, product/environment/trust scope oraz
authenticated immutable creation reference/digest/authority identity. Adapter
nie może wybrać innego genesis: predecessor `None` i subject revision 1 są
obowiązkowe.

Admin `SupersedeEntitlementRequest` wymaga exact predecessor, tego samego
subject i authoritative entitlement ID, następnej generation oraz nowej
authenticated provenance. Może supersede'ować historyczne UNBOUND lub BOUND.
Operacja atomowo unieważnia starą generation i instaluje nową ACTIVE/UNBOUND;
obie zmiany mają kolejne registry-controlled subject revisions. Entitlement ID,
environment, trust domain, product scope i intended action są immutable w całej
lineage. Nowa generation może otrzymać nowy provisioning principal i claimant
key/version wyłącznie przez tę uwierzytelnioną admin supersession; provenance
jest immutable wewnątrz jednej generation. Runtime port nie ujawnia
żadnej operacji provisioning, supersession, revoke ani migration.

Admin `RevokeEntitlementRequest` zawiera exact `AdminPredecessor`. Nie dodaje
nowego arbitralnego audit payloadu: uwierzytelnienie wywołującej deployment
security authority jest obowiązkiem oddzielnego admin portu/credentials, a
exact predecessor jest zamkniętym inputem decyzji. `revoke_entitlement`
zwraca typed `AdminResult` (`COMMITTED`, `CONFLICT`, `NOT_FOUND`, `CORRUPT`,
`UNAVAILABLE`, `RETRYABLE_SERIALIZATION_FAILURE`). Shared `revoked_state_for`
jest jedynym builderem ACTIVE→REVOKED: zachowuje subject, identity/generation,
provenance oraz binding, ustawia revision=current+1 i predecessor=current.
Dlatego ACTIVE/UNBOUND pozostaje UNBOUND, a ACTIVE/BOUND zachowuje dokładnego
winnera. REVOKED jest terminalny: drugi revoke nie tworzy revision,
supersession i bind są zabronione, a powrót do ACTIVE jest nielegalny.
Runtime `EntitlementRegistryProvider` nie posiada revoke API.

`AdminResult` również ma zamkniętą macierz: `COMMITTED` wymaga dokładnego nowo
committed authoritative state. `CONFLICT`, `NOT_FOUND`, `CORRUPT`, `UNAVAILABLE`
i `RETRYABLE_SERIALIZATION_FAILURE` wymagają `state=None`. Powtórzony revoke lub
supersession po wygranym CAS jest typed `CONFLICT` bez nowej revision i bez
current-state payloadu; pełny admin lost-response protocol nie jest tu tworzony.

Każda publiczna security-sensitive wartość przechodzi shared exact/deep
snapshot: exact class, bezpieczny odczyt wszystkich slots, rekonstrukcja przez
normalny konstruktor oraz rekonstrukcja nested records. Obiekty fabricated
przez `object.__new__`, brakujące slots i malformed nested records kończą się
kontrolowanym `ContractValidationError`, nigdy surowym `AttributeError`.

## 10. Interpretacje zabronione i status implementacyjny

Zabronione są: authoritative `object`, dict lub bool CAS; NOT_FOUND jako
UNBOUND; caller-selected entitlement ID/revision; conflation generation i
revision; inny tuple jako replay; drugi proof ID na retry; bind REVOKED albo
SUPERSEDED; stale snapshot lub brak retained history jako UNBOUND evidence;
wall-clock ordering; admin method na runtime porcie.

`POSTGRESQL_DRIVER_SELECTION = DEFERRED_TO_PROVIDER_IMPLEMENTATION`  
`ROOT_PROOF_ISSUER_PRODUCTION_LOCAL_ENTITLEMENT_REGISTRY_IMPLEMENTED = false`  
`ROOT_PROOF_ISSUER_IMPLEMENTED = false`  
`PRODUCTION_LOCAL_RUNTIME_AVAILABLE = false`

Provenance: `classification = UNKNOWN`, `finding_scope = CURRENT_TREE_ONLY`,
`formal_project_advancement = WITHHELD`.
