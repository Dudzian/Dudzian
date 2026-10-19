# Stage 9 — S9-RPI-07: independent CCA i trwały rejestr decyzji

## Wynik i zakres

S9-RPI-07 przygotowuje qualification-only protokół decyzji niezależnej
Custody Completion Authority (CCA) oraz osobny demonstrator trwałego logu
**TEST-only**. Nie instaluje produkcyjnej CCA ani jej credentials. Nawet
poprawny podpis syntetycznej decyzji pozostawia `NOT_QUALIFIED`,
`production_emission=BLOCKED` i `PRODUCTION_NO_GO`. Trwałość zapisu testowego
nie jest dowodem wykonania mutation, wcześniejszej trwałości source event
ani autoryzacją publikacji completion.

```text
independent_custody_completion_authority = NOT_QUALIFIED
authenticated_custody_mutation_source = NOT_QUALIFIED
independent_custody_durability_confirmation = NOT_QUALIFIED
production_custody_completion_publication = BLOCKED
CONTRACT_ADDENDUM_INTEGRATION_PENDING
semantic_root_proof_issuer_runtime = NOT_IMPLEMENTED / BLOCKED
production_provisioning_ready = false
ROOT_PROOF_ISSUANCE_AVAILABLE = false
ATOMIC_ISSUER_READY = NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION = BLOCKED
public_root_proof_issuance = NO-GO
```

To opis zakresu zmiany, bez aktualizacji canonical statusów. Właściciel
zaakceptował [kierunek A+C](decisions/stage9_root_proof_transaction_authority_owner_approval_v1.md),
lecz formalna integracja wersjonowanego addendum i kwalifikacja deploymentu
pozostają niezakończone. Osobny finalny COMMIT decyzji issuera nie staje się
transakcją CCA, custody mutation, CHA, FreshnessAuthority lub AccountGenesis.
Retirement custody i decyzja CCA nie oznaczają issuer BOUND ani account closure.

## Audyt i zachowane granice

Podstawą audytu jest `main` po PR #3160, commit `97e798f`, oraz rzeczywiste
źródła: [S9-RPI-05](stage9_independent_custody_completion_authority.md),
[S9-RPI-06](stage9_authenticated_custody_mutation_source.md),
[`independent_custody_completion_qualification.py`](../../bot_core/independent_custody_completion_qualification.py),
[`authenticated_custody_mutation_source.py`](../../bot_core/authenticated_custody_mutation_source.py),
[`local_signing_lifecycle_evidence.py`](../../bot_core/local_signing_lifecycle_evidence.py),
[`production_root_lifecycle.py`](../../bot_core/production_root_lifecycle.py),
reviewed SQL V2 i testy istniejących odmów. Codebase Memory MCP i jego CLI
nie były dostępne; wnioski sprawdzono celowanym `rg` i odczytem kodu. Nie
deklarujemy kompletności nieobecnego indeksu ani kwalifikacji instalacji.
Przed wystawieniem PR bazę uaktualniono do `7cdc0f77` po #3161. Jedyną
zmianą upstream po bazie audytu były ograniczone pytest IDs istniejących
przypadków S9-RPI-06 na Windows; kontrakty i runtime nie zmieniły się.

| Mechanizm | Wykorzystywana właściwość | Nadal brakujący dowód |
| --- | --- | --- |
| S9-RPI-01, production PostgreSQL 16 lifecycle | Immutable request, exact predecessor, osobny COMMIT `PENDING_CLOSED`, `close_xid`, lock-synchronized resolution, direct-login/OID/peer i catalog pin | Wykonanie mutation i autentyczne niezależne completion |
| S9-RPI-02, Root/History signer IPC | Odrębne role i namespace, Linux peer/DAC checks, zamknięte `DIAGNOSTIC`/`QUALIFY` | Production private control, SIGN i installed CCA |
| S9-RPI-03/05, odczyt custody | Exact canonical descriptor/journal `COMPLETE`, metadata i recheck pod istniejącą blokadą | Autorstwo plików, uprzedni `fsync`, wykonanie i publication authorization |
| S9-RPI-06, source event | Zamknięty `CustodyMutationExecutionV1`, podpis oryginalnych payload bytes, jawna stabilna policy | Niezależne pochodzenie policy, installed executor i jego trwały execution log |
| S9-RPI-07, decyzja/log TEST | Qualification-only verification, original-byte retention, serializacja i recovery lokalnego demonstratora | Produkcyjne źródło faktów, CCA credential, trusted floor i publication |

Źródła frozen to [independent issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json),
[production substrate](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json),
[implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json)
i [physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json).
Readiness wciąż ma `implementation_allowed=false`; test helper, interfejs,
dokument lub pomyślne CI nie są `AVAILABLE` production capability.

`ProductionRootLifecycleBoundary.commit()` oraz
`publish_administrator_completion()` bezwarunkowo odmawiają, również po
przekazaniu nowej decyzji lub raportu. SQL V2
`complete_close(text,text,bytea)` pozostaje tombstone z `SQLSTATE 42501`;
nie należy do `ISSUER_FUNCTIONS` ani `ADMINISTRATOR_FUNCTIONS`, a instalator
odbiera `EXECUTE` od PUBLIC. Nie ma nowego grantu, principal, migracji,
write-capable SQL helpera, provisioningu private key, systemd unit ani
automatycznego re-pin. SIGN/BIND, enrollment, activation, Stage 10 i
production issuer runtime pozostają zablokowane. Historyczne
`LEGACY_UNVERIFIED_COMPLETION` nie są awansowane.

## Trust boundaries i trzy authorities

Root source, History source i CCA to trzy projektowane odrębne authorities.
Root/History source wykonują wyłącznie właściwą operację swojego custody
namespace i uwierzytelniają jej execution event osobnym source credential.
Nie używają do tego Root Proof lub History signing key. CCA weryfikuje
publiczne retained evidence i podejmuje późniejszą własną decyzję; nie
mutuje custody i nie wykonuje Root Proof SIGN ani History Attestation.
Source nie posiada completion key i nie może wystawiać decyzji CCA.

CCA wymaga własnych UID/GID, rozłącznych supplementary groups i efektywnych
ACL, chronionej konfiguracji/pinu instalacji, service endpoint, Ed25519
credential wyłącznie dla completion domain, provider namespace, trust epoch,
key version i lifecycle generation. Te zasoby nie mogą być współdzielone
z obydwoma sources, signerami, issuerem ani administratorem lifecycle.
Nazwy kont i zadeklarowane liczby UID/GID w syntetycznej policy nie dowodzą
fizycznej separacji. Production qualification musi sprawdzić kernel identity,
memberships, efektywny dostęp do plików/socketów/keyringu i alternatywne porty.

Zaufany installation pin i policy pochodzą z niezależnego chronionego
provisioningu; nie z envelope, caller-selected path, public key callera ani
badanej bazy. Podpis weryfikuje się kluczem z policy. Powtórny odczyt policy
wykrywa zmianę snapshotu podczas verification; nie jest trwałym lifecycle
fence. Deklaratywna zgodność pól oznacza zgodność z oczekiwaniami testu,
bez kwalifikacji ich pochodzenia.

Bootstrap administrator i root hosta pozostają osobnymi zaufanymi granicami.
SQL nie jest kryptograficznie niepodrabialnym źródłem wobec superusera.
Rozdział UID/GID chroni przed zwykłymi runtime principals; nie chroni przed
root, ekstrakcją software keys, przejętym source lub skoordynowanym rollbackiem
całego hosta. Poprawny podpis przejętego source nie przywraca prawdziwości
wykonania. Odrębna CCA nie zastępuje kwalifikacji źródła.

## Siedem przyszłych zależności przed pozytywną decyzją

Każda przesłanka wymaga własnego dowodu i właściciela. Kontrakty są projektem
interfejsów; brakującego backendu nie zastępuje `approved=true`, podpisane
oświadczenie callera ani status testowego logu.

| Przesłanka | Wymagany przyszły kontrakt i odmowa |
| --- | --- |
| 1. Independently installed source identity/trusted policy | Chroniony loader wybiera fixed installation/environment/trust pin, source role, endpoint, UID/GID, namespace, key/material/version i epoch. Brak, drift lub caller-selected anchor odmawia. |
| 2. Autentyczność retained source event | Exact lookup w przypiętym source logu odzyskuje oryginalny envelope dla operation/execution ID i jego log position. Sam podpis bez potwierdzonego retention/provenance nie wystarcza; inny oryginał lub brak jest nierozstrzygnięciem. |
| 3. Wykonanie i trwałość mutation | Kwalifikowany executor posiada cały transition, właściwy lock, before/after metadata, journal i wymagane synchronizacje. Odczyt `COMPLETE` lub source signature nie dowodzą poprzednich `fsync`. |
| 4. Własny qualified committed-close read CCA | Odrębny minimalny reader potwierdza fizyczną bazę/catalog, direct-login/OID/peer, exact request bytes/digest, role/identity/generation i cudzy potwierdzony COMMIT `PENDING_CLOSED`. Brak, timeout, własny xid, niepewność lub zmiana odmawiają. Production reader wymaga osobnego review, approved catalog migration i nowego zewnętrznego pinu; nie jest tu implementowany. |
| 5. Current i historical lifecycle eligibility | Osobno kwalifikowane źródła lifecycle określają eligibility source, custody i CCA credential dla retained event oraz chwili decyzji. Rotacja/revoke/generation drift podlegają fence utrzymanemu przez trwałe utrwalenie decyzji. Sam historical signature nie upoważnia do nowej decyzji. |
| 6. Source custody lease | Uwierzytelniony exact-operation lease utrzymuje oczekiwane custody generation/state i source evidence aż do trwałej decyzji CCA. Utrata lease lub przejście do późniejszego legalnego descendant odmawia. |
| 7. Własny log CCA i lock order | Własny pinned namespace, immutable retention, exact retry, trwały konflikt, sequence/previous-original digest, checkpoint/floor i kontrolowane recovery. Jedna serializacja decyzji nie może zwalniać lease przed trwałym zapisem ani odwracać lock order. |

Future reader CCA nie jest dotychczasowym administracyjnym
`resolve_close()` przeniesionym do nowego procesu. Frozen instalacja ma
zamknięty zbiór wcześniejszych principals i funkcji; nie dopisujemy pozornego
CCA read principal. PostgreSQL przechowuje request jako `jsonb`: oryginalne
retained bytes są `request::text` UTF8 (`PostgreSQL16/jsonb-text/UTF8`),
nie utraconym client wire formatem ani ponownie canonicalizowanym JSON.

Kod deklaruje osobne kontrakty `RetainedSourceEventReader`,
`IndependentCommittedCloseReader`, `SourceCustodyLease` oraz
`CredentialEligibilityLease`. Odczyt exact close i oba lease mają postać
context managerów, aby przyszły backend mógł utrzymać swoje gwarancje przez
decyzję. Nie mają produkcyjnej implementacji ani drogi do `QUALIFIED`.
Historyczna eligibility nadal wymaga osobnej kwalifikacji. Podwójny odczyt
policy snapshotu nie wykrywa ABA i nie zastępuje właściwego lock/CAS.

## Zamknięty protokół decyzji V1

[`independent_cca_decision.py`](../../bot_core/independent_cca_decision.py)
udostępnia `parse_independent_cca_decision`,
`parse_cca_verification_policy` i `verify_independent_cca_decision`.
`IndependentCCADecisionV1` przechowuje original canonical envelope bytes;
`ExplicitCCAVerificationPolicy` przechowuje osobny expected snapshot.
Verifier nie ma private key, signing, SQL connection, write portu ani
production policy loadera.

```text
domain bytes = b"CryptoHunter/IndependentCustodyCompletion/V1\0"
payload schema = cryptohunter.independent-custody-completion
envelope schema = cryptohunter.signed-independent-custody-completion
policy schema = cryptohunter.explicit-independent-cca-verification-policy
version = 1
```

Envelope ma dokładnie `schema`, `version`, `payload_b64`, `credential_id`,
`key_version`, `signature_b64`. Podpis Ed25519 obejmuje domain bytes razem
z końcowym NUL i **dokładne oryginalne decoded payload bytes**. Ponowna
canonicalizacja służy sprawdzeniu formatu, nigdy podmianie wiadomości
weryfikowanej kryptograficznie. Klucz pochodzi wyłącznie z independent policy;
envelope credential/version muszą zgadzać się z payload i policy.

Payload ma dokładnie:

```text
schema, version, domain, installation, completion_authority, database,
custody, close, close_observation, originals,
source_event_envelope_b64, source_event_envelope_sha256,
decision_id, decision_log_position,
source_authentication_result, execution_result, durability_result,
completion_result
```

`installation`, `database`, `custody`, `close` i `originals` muszą być exact
odpowiednikami danych osadzonych w oryginalnym source envelope. Zamknięte
podobiekty zachowują [schema S9-RPI-06](stage9_authenticated_custody_mutation_source.md):
installation/environment/profile/trust domain, installation digest,
pin ID/version/digest; fizyczny PostgreSQL system identifier/database OID,
instance, adresy/porty i qualified catalog; exact role, credential/material,
key version oraz before/after lifecycle generation/state. `originals`
zawiera oryginalne descriptor/journal/before/after metadata Base64 i ich
osobne SHA-256. `close` zachowuje exact operation ID, request bytes/digest,
format, `close_xid` oraz `PENDING_CLOSED`.

`completion_authority` ma zamknięte pola authority/service/host ID, UID/GID,
endpoint, provider namespace, credential identity/semantic role, public key
i material identity, key version, lifecycle generation/state i trust epoch.
Role to `CUSTODY_COMPLETION_ATTESTATION_V1`; parser wymaga `ACTIVE`.
`close_observation` ma `reader_role`, `reader_role_oid`, `observation_xid`,
`qualified_catalog_sha256`. Wymaga odrębnych reader role/OID/xid względem
source i zgodności z independently expected snapshotem. To format oczekiwań,
bez implementacji qualified produkcyjnego odczytu.

`decision_log_position` ma dokładnie `namespace`, `sequence`,
`previous_entry_sha256`, `entry_id`; namespace odpowiada CCA, entry ID jest
równy `decision_id`, a sequence i previous digest muszą odpowiadać policy.
Source event digest odnosi się do całego retained original envelope, również
jego podpisu. Pierwszy previous digest to 64 zera.

Zamknięte wartości czterech result fields to kolejno
`QUALIFIED_INSTALLED_SOURCE`, `EXACT_LIFECYCLE_TRANSITION_COMPLETED`,
`SOURCE_AND_CCA_RECORDS_SYNCED` i
`AUTHENTICATED_CUSTODY_COMPLETION_RETAINED`. W parserze są to **niezaufane
oświadczenia podpisanego formatu**, bez dowodu wystąpienia tych zdarzeń.
Raport po obu poprawnych podpisach nadal ma `source_authorization` i
`committed_close` równe `NOT_QUALIFIED`, execution/durability `NOT_ATTESTED`
oraz pełną listę blockers.

Policy ma dokładnie `schema`, `version`,
`source_verification_policy_b64`, `source_event_envelope_sha256`,
`completion_authority`, `close_observation`, `decision_id`,
`decision_log_position`, `principal_inventory`. Source policy osobno wiąże
installation/source/database/descriptor/close i pozycję source logu.
Inventory wymaga wszystkich pięciu rozłącznych principals: Root source,
History source, Root signer, History signer i CCA, wraz z supplementary GIDs.
Współdzielenie UID, GID/grup, namespace, credential lub key material odmawia.
Inventory pozostaje deklaracją do przyszłej instalacyjnej kwalifikacji.

Verifier weryfikuje source signature w jego osobnej domenie, następnie CCA
signature w completion domain, porównując dane z polityką. Loader nie
otrzymuje caller envelope/klucza/pinu/ścieżki; brak lub zmiana snapshotu,
revoke/rotation, inna operacja/sequence/previous digest i podmiana oryginałów
odmawiają. Weryfikacja nie konsumuje pozycji logu, więc niezmieniona policy
może ponownie sprawdzić te same bytes; durable replay protection należy do
właściciela przyszłego checkpointu/logu.

| Dane | Maksymalna liczba original bytes |
| --- | --- |
| CCA envelope / payload / policy | 14 MiB / 10 MiB / 2 MiB |
| Source envelope / source payload / source policy | 4 MiB / 3 MiB / 1 MiB |
| Descriptor / journal / metadata / close request | 256 KiB / 1 MiB / 64 KiB / 64 KiB |

Wszystkie listy pól są zamknięte, Base64 to canonical RFC 4648 z paddingiem,
SHA-256 to 64 małe znaki hex, Ed25519 public key ma 32 bytes, signature 64.
Duplicate keys, unknown/missing fields, niekanoniczny JSON/Base64, bool zamiast
int, floats/NaN/infinity, niepoprawne Unicode/UTF8, przekroczone limity
rozmiaru lub JSON depth i cross-role substitution odmawiają przed uznaniem
podpisu. Numeryczne limity i typy są odziedziczone z S9-RPI-05/06, w tym
canonical decimal strings dla PostgreSQL system identifier/xids.

## TEST-only durable decision log

[`test_only_cca_decision_log.py`](../../bot_core/test_only_cca_decision_log.py)
jest oddzielnym eksperymentem Linux. Jawne jednorazowe
`provision_test_only_cca_log` wymaga dwóch istniejących pustych katalogów
`0700`: logu i floor, rozłącznych także jako ancestor/descendant. Zwraca
`TestOnlyCCALogPin` z installation/namespace, dev/inode obu katalogów i
manifest digest. Runtime nie odtwarza brakującego manifestu, locka lub floor.
Pin jest fixture outputem; nie staje się production trust anchor.

`TestOnlyCCADecisionLog.decide(operation_id, immutable_binding,
create_decision=...)` utrwala syntetyczny original envelope. Binding obejmuje
cały canonical payload poza nowym `decision_id` i `decision_log_position`;
zawiera zatem oryginalny source envelope, descriptor/journal/close i wszystkie
identities/generations. Idempotency key to `(installation_id, operation_id)`.
Ten sam klucz i identyczny binding zwracają poprzednie retained bytes bez
callbacka i bez ponownego podpisania. Inny binding zapisuje immutable
`conflict.<sha256>` i odmawia; poprzednia decyzja pozostaje niezmieniona.
Konflikt przetrwa restart, a późniejszy exact retry nadal odzyskuje oryginał.

`exact_lookup` wymaga tego samego pełnego bindingu i nigdy nie wywołuje
callbacka. `NOT_FOUND` oznacza `OUTCOME_UNKNOWN`, bez uprawnienia do podpisania
zamiennika. Syntetyczne signing keys należą do testowych fixtures; sam log
nie posiada klucza ani qualified source/close admission. Sprawdza zamknięty
format decyzji i exact binding, bez samodzielnego nadawania autentyczności
podpisowi. Test callback nie jest produkcyjnym signing endpointem.

Jedna sequence obowiązuje w całym przypiętym namespace. Każdy intent i
decision position wskazują SHA-256 **oryginalnych bytes poprzedniego
retained envelope**, nie jego ponownej serializacji. Log przechowuje pełny
envelope, a nie sam digest. Limit demonstratora to 10 000 entries, 6 MiB
bindingu, 8 MiB retained envelope oraz 9 MiB rekordów intent/conflict;
to węższe limity niż sam maksymalny protokół CCA.

### Lock order i bezpieczeństwo filesystem

Rzeczywistą blokadą tego modułu jest exclusive Linux `flock` na istniejącym
przypiętym `lock`, utrzymany przez scan, exact retry/conflict, zapis, recovery
i końcowe rechecks. Floor aktualizuje wyłącznie właściciel tej samej blokady;
nie ma osobnego konkurencyjnego floor locka. Dwa procesy dostają jeden
oryginał albo trwały konflikt, bez dwóch podpisów tej samej operacji.

Rzeczywisty process harness TEST ma kolejność: exclusive source lease →
exclusive credential lifecycle lease → CCA log `flock` → trwała decyzja →
zwolnienie w odwrotnej kolejności. Proces rotacji bierze tylko credential
lease; nie istnieje odwrotna krawędź do source lub logu. Osobne testy
współbieżności logu używają niezależnych fixture leases, aby wykazać
kontencję samego CCA `flock`.

Kandydat przyszłej kompozycji zachowuje source exact-state lease oraz source/CCA
credential eligibility leases przed CCA log lockiem, następnie wykonuje własny
krótki qualified exact-close read/slot lock, kończy odczyt DB, utrwala decyzję
i zwalnia locks w odwrotnej kolejności. Nie wolno pozyskiwać source/lifecycle
lease dopiero po log locku ani zwolnić go przed trwałą decyzją. Implementacja
tego modułu nie zdobywa source/SQL/lifecycle leases: testowy właściciel callbacka
musi utrzymać własną serializację; produkcyjne backends i pełny globalny lock
order pozostają do osobnego review. To nie jest finalny issuer transaction
lock order A+C.

Odczyty i zapisy używają dirfd dla wszystkich komponentów, `O_NOFOLLOW`,
`O_EXCL`, rechecków directory/file dev/inode/mode/UID/GID oraz rozmiaru,
mtime/ctime/link count. Katalogi wymagają `0700`, retained immutable pliki
`0400`, lock/floor `0600`, regular file i dokładnie jednego hardlinka.
POSIX ACL, symlinki, hardlinki, writable non-sticky ancestors, podmiana
inode/katalogu lub manifestu odmawiają. Sticky `/tmp` jest dopuszczony jako
ancestor disposable tests. Oryginały publikowane są Linux
`renameat2(RENAME_NOREPLACE)`; nigdy przez zastąpienie istniejącej decyzji.
Tylko mutable floor pointer jest aktualizowany przez `os.replace` po
sprawdzeniu poprzedniego stanu. Ta ochrona nie izoluje złośliwych procesów
o tym samym UID ani root.

### Exact phases i recovery

Intent ma original immutable binding i ustaloną pozycję/decision ID.
Podpisujący fixture callback jest wywoływany **dopiero po** trwałym intent
i floor reservation. Brak późniejszego oryginału nie pozwala wywołać go
drugi raz. Nie ma wieloplikowego atomowego COMMIT dla intent, decision i
floor; protokół jawnie zachowuje niepewność.

| Crash cut / stan po nim | Rzeczywiste zapisy | Recovery |
| --- | --- | --- |
| `before_intent` | Brak nowego intent, bez callbacka | Zdrowy poprzedni log; exact lookup brakującej operacji daje `OUTCOME_UNKNOWN`. Pierwszy nowy `decide` może rozpocząć operację. |
| `after_intent_write` | `NN.intent.stage`, file `fsync`, bez publication | `INDETERMINATE`; staged intent nie jest automatycznie usuwany lub awansowany, bez podpisu. |
| `after_intent_publication` | `NN.intent` przez no-replace rename i directory `fsync`, floor bez reservation | `INDETERMINATE`; unreserved intent nie uprawnia do callbacka. |
| `after_floor_reservation` | `floor.stage` z pending intent SHA-256, file `fsync`, replace floor i directory `fsync` | Bez retained decision pozostaje `INDETERMINATE`; callback nie jest ponawiany. |
| `after_decision_write` | Callback raz; pełne `NN.decision.stage`, file `fsync` | Przy exact intent i pending floor: zweryfikuj original/binding, ponownie sync plik/katalog, no-replace publication, utrwal finalny floor; bez ponownego podpisu. |
| `after_decision_publication` | `NN.decision` po no-replace rename i directory `fsync`; floor nadal pending | Odczyt i recheck dokładnych original bytes, ponowny sync, finalny floor, exact return bez callbacka. |
| `after_durable_decision` | Finalny floor sequence/digest bez pending, file i directory `fsync`; odpowiedź jeszcze nie zwrócona | Restart/exact retry odzyskuje byte-identical envelope i podpis, bez ponownego callbacka. |

`NN` w tabeli to 20-cyfrowa monotoniczna sequence. Przy normalnym return
następuje ponowny pełny scan i recheck pinned paths/originals. Każdy udany
scan dodatkowo synchronizuje aktualny floor i jego katalog: widoczny floor
po wcześniejszym rename bez potwierdzonego directory `fsync` nie wystarcza
do odpowiedzi. Dalszy błąd tej synchronizacji zachowuje `INDETERMINATE`. Recovery
może dokończyć wyłącznie pełną retained decision zgodną z dokładnym intent
i matching pending floor. Nie naprawia niepełnego podpisu, nie rekonstruuje
envelope z hasha i nie tworzy nowej decyzji.

Błędy write/`fsync`/rename, częściowy `floor.stage`, unknown staged file,
brak decision poniżej floor, gap/replay sequence, inny previous digest,
rollback/ahead floor lub brak logu/floor odmawiają. Gdy wynik operacji I/O
jest nieznany, moduł zgłasza `TestOnlyCCALogIndeterminate`; naruszenia formatu
i pinów zgłaszają błąd logu, a immutable binding conflict osobny trwały
konflikt. Otwarty nierozstrzygnięty intent blokuje także następne appendy:
nie wolno ominąć go nową sequence. Brakujące pliki/floor nie są po cichu
tworzone na restart. Scan obejmuje cały namespace, więc nierozstrzygnięty
stan późniejszego wpisu może odmówić także lookup wcześniejszej decyzji;
demonstrator nie udaje częściowej dostępności zdrowego logu.

### Replay, floor i ograniczenie rollback

Scan od początku sprawdza contiguous sequence, brak powtórzeń operation ID,
previous-original digest, exact intent/decision binding oraz zgodność head
z floor. Osobny TEST floor wykrywa rollback lub usunięcie samego logu,
a retained log wykrywa niespójny/stary floor. Floor zapisuje również pending
reservation, więc niezakończona próba nie staje się pozwoleniem na re-sign.

**Hash chain sam nie chroni przed rollbackiem całego hosta.** Log, fixture
pin i floor znajdują się na tym samym hoście; skoordynowane przywrócenie ich
spójnego starego snapshotu może przejść lokalne sprawdzenia. Nie ma tu
produkcyjnego trusted floor, niezależnego rollback domain, ochrony przed
złośliwym same-UID writerem ani kwalifikacji fizycznej power-loss durability.
Nie wolno awansować tego demonstratora do production anti-rollback service.

## Lifecycle, retry i brak production acceptance

Nowa decyzja wymaga aktywnego, przypiętego completion credential oraz exact
source/custody generation. Powtórny exact lookup ma odzyskiwać oryginalną
decyzję wraz z jej historycznym credential/version; rotacja nie uzasadnia
ponownego podpisania lub przepisywania bytes. Production eligibility
historycznego retained podpisu, dopuszczenie jego użycia i bieżąca możliwość
podjęcia nowej decyzji są odrębnymi ocenami.

CCA production publication będzie wymagała niezależnej weryfikacji także
na samym SQL boundary. Kontrola wyłącznie w Pythonie pozwalałaby adminowi
ominąć ją bezpośrednim wywołaniem SQL. W tym etapie bezpieczny wynik każdej
próby publikacji pozostaje odmową tombstone; TEST report, pozytywny verify
lub durable entry nie jest capability enrollment/SIGN/BIND.

## HARDWARE_DEPENDENCY

**Następny etap testowy nie wymaga drugiego fizycznego komputera.** Linux CI
wystarcza do testowania parsera, podpisów syntetycznych, rzeczywistych procesów,
`flock`, zapisów/fsync/rename, crash cuts, restartu i exact recovery TEST.
Izolowany PostgreSQL 16 pozwala przetestować dotychczasowy SQL i zachowane
odmowy. To nie dowodzi zachowania dysku po fizycznej utracie zasilania.

Kwalifikacja rzeczywistej lokalnej instalacji wymaga hosta Linux z systemd,
trwale skonfigurowanymi, niezależnymi UID/GID i supplementary groups,
efektywnymi DAC/ACL, rzeczywistym native keyringiem, rozdzielnymi namespaces,
chronionymi config/pin i usługami o fixed identity. Musi obejmować niedostępność
kluczy i logów obcej roli, peer credentials, lifecycle/lease races, restarty
usług/hosta, recovery oraz kontrolowane power-loss/storage checks. Testowy
keyring, zadeklarowane UID/GID lub sam subprocess nie zastępują tych dowodów.
Tę lokalną kwalifikację można przeprowadzić na jednym właściwie przygotowanym
hoście; w tej zmianie jej nie wykonano.

Dopiero silniejszy model z odpornością na skoordynowany rollback lub pełne
przejęcie lokalnego hosta wymaga niezależnego security/rollback domain:
np. osobno administrowanego zdalnego urządzenia/serwera świadczącego
uwierzytelniony monotoniczny checkpoint albo kwalifikowanego zewnętrznego
secure urządzenia z nonrollbackable state. Sam drugi komputer, osobny folder,
hash chain lub software klucz na tym samym hoście nie tworzą tej gwarancji.
Wybór urządzenia/providera i jego protokół wymagają przyszłej decyzji; nie ma
podstawy do zakupu drugiego komputera dla S9-RPI-07.

## Warunki kolejnego etapu i walidacja

Przed produkcyjną instalacją pozostają: review protokołu i provenance,
formalna wersjonowana integracja A+C, jawny provisioning trzech authorities
bez współdzielenia credentials/namespaces, kwalifikowany independent close
reader i migracja catalogu/pinu, trwałe source/CCA lifecycle interlocks,
source lease i rzeczywisty decision log/floor, weryfikacja na SQL publication
boundary, private-control/trust activation i wszystkie pozostałe
CHA/FreshnessAuthority/CheckpointAuthority/AccountGenesis gates. Zakres i
parametry silniejszego rollback modelu wymagają osobnej decyzji. Dopiero
osobny przegląd deploymentu może wydać GO/NO-GO.

Weryfikacja obejmuje nowe negatywne przypadki protokołu i filesystem logu,
rzeczywiste procesy/concurrency/restart, celowane S9-RPI-05–07, właściwe
PostgreSQL 16 regresje, frozen-contract i production refusal tests oraz
exact CI inventory/changed-code coverage. Ruff, pełny skonfigurowany
`ruff-format --all-files`, mypy Linux/Win32, Semgrep i Betterleaks pozostają
wymaganymi kontrolami bez obniżania progów, zmian timeoutów i nowych
`skip`/`xfail`. Rzeczywiste polecenia, wyniki i niedostępne dowody należą do
opisu PR; wymagań w tym akapicie nie należy odczytywać jako wyników testów.
Pełne wielogodzinne CI pozostaje `FULL_CI_PENDING`, dopóki nie jest
potwierdzone. Merge ani zakończone CI nie nadają produkcyjnego GO.

## Wykonana walidacja — 2026-10-19

Walidacja lokalna używała Python 3.11.16 oraz jednorazowego kontenera
Python 3.11.17 z domyślnymi capabilities, bez trybu privileged. Wszystkie
klucze i operacje nowego harnessu były syntetyczne. Dwa niezależne trwałe
klastry PostgreSQL 16.15 miały `fsync`, `synchronous_commit` i
`full_page_writes` włączone; osobny jednorazowy klaster sprawdzał odmowę
przy wyłączonym `fsync`. Kontenery PostgreSQL i ich wolumeny usunięto po testach.

| Kontrola | Rzeczywisty wynik |
| --- | --- |
| S9-RPI-05/06/07, nowe production boundary tests i exact CI/platform inventory | **585 PASS**: 102 qualification claims, 175 source protocol, 38 source processes, 202 CCA protocol, 27 production boundary, 41 CI/platform inventory; bez skip/xfail |
| TEST durable log w izolowanym Linux | **53 PASS**; prawdziwe zapisy, `flock`, procesy, crash/restart, exact recovery, konflikt, replay/rollback, podmiany inode/katalogów, POSIX ACL i awarie I/O; bez skip/xfail |
| Dotychczasowe production PostgreSQL lifecycle | **136 PASS**, jeden istniejący Linux skip testu odmowy na systemie innym niż Linux; schema/ACL/qualification/identity/races i restart |
| Dotychczasowy isolated custody lifecycle PostgreSQL | **31 PASS** w 525.05 s, z genuine lineage, restartem, COMMIT-response-loss i wyścigami; nadal bez production CCA |
| Dotychczasowy root Linux signer process/DAC harness | **6 PASS**, rzeczywiste odrębne testowe UID/GID i brak supplementary groups; bez native-keyring qualification |
| Frozen contracts i istniejące odmowy | Osobny zestaw **597 PASS**; dodatkowy zestaw local evidence/admission i frozen persistence **474 PASS** |
| Regresje signer IPC/custody/deployment/lifecycle | **147 PASS** |
| Nowe moduły — branch coverage | CCA: **100% linii**, 99% łącznego pomiaru; TEST log: **93.51% linii**, 90% łącznego pomiaru |
| Changed-code coverage | **591/618 linii, 95.63%**, `diff-cover --compare-branch origin/main --fail-under=70`: PASS |
| Ruff i format | Pełny scope Ruff: PASS; po staging wszystkich deliverables dokładny `python -m pre_commit run ruff-format --all-files`: PASS, bez zmian formattera |
| Mypy | Skonfigurowane `python -m mypy` i `python -m mypy --platform win32`, bez filename overrides: PASS, po **342 source files** |
| Semgrep CE 1.179.0 | Blocking custom scope: **0 findings**, 1051 Python targets/5 rules; fixtures **5/5 PASS**; oba nowe moduły dodatkowo **0 findings** |
| Betterleaks 1.9.0 | Offline working-tree snapshot: **0 nowych leaks**, 3758 unikalnych tracked/new nonignored ścieżek, 58.17 MB; niezmieniona baseline, wyłączone tylko scratch `work/` |
| Po rebase na #3161 | Source protocol i oba CI/platform inventories: **216 PASS**; pełny `ruff-format --all-files` oraz changed-code coverage: PASS |

Pomiary coverage połączono z dwóch ukończonych baz branch coverage z
identycznymi względnymi ścieżkami; raport samego parsera nie zastąpił pomiaru
logu. Liczby poszczególnych regresji mogą obejmować powtórzenia — nie są
sumą unikalnych testów całego repozytorium. Jedno ostrzeżenie kontenera
dotyczyło ponownego importu NumPy, nie pominiętego testu. Istniejące ostrzeżenia
Ruff o trzech legacy `WPS433` i siedem notatek mypy nie zostały ukryte.

Natywne ACL sprawdzono w rzeczywistym filesystem kontenera: `/tmp` hostowego
workspace nie obsługiwał potrzebnego xattr, więc nie zastąpiono dowodu mockiem
ani skipem. Linux CI musi zachować rzeczywiste wsparcie POSIX ACL dla tego
integration testu. Nie wykonano fizycznego power-loss, installed CCA/source
systemd/native-keyring qualification, produkcyjnego independent close read
ani testu niezależnego rollback domain. Wyniki nie znoszą żadnego blockera.
