# Stage 9: Atomic Root Proof Commit Authority V1

Ten dokument opisuje administracyjną migrację oraz granicę wewnętrznego
testowania atomowej decyzji issuera. Nie stanowi zmiany zamrożonych kontraktów
M0.5 ani zgody na publiczną emisję. `ROOT_PROOF_ISSUANCE_AVAILABLE=False`,
`PRODUCTION_READY=False` i
`semantic_root_proof_issuer_runtime=NOT_IMPLEMENTED / BLOCKED` pozostają
obowiązkowe. `ATOMIC_ISSUER_READY=NOT_READY`.

## Źródła audytu i konflikt wymagający osobnego rozstrzygnięcia

Audyt obejmuje rzeczywiste SQL oraz kwalifikatory entitlement registry,
signing intent, retention, authenticated issuer history, checkpoint i
pre-account credentials, a także lokalne signing custody. Podstawą jest
[kwalifikowana granica atomowa](stage9_qualified_atomic_root_proof_boundary.md).
Codebase Memory nie jest dostępne w tym środowisku; ustalenia wynikają z
celowanych wyszukiwań i kontroli źródeł, a nie z domniemanej kompletności
indeksu.

Zamrożony
[independent issuer contract](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json)
wymaga `8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY`.
`issuer_bound_attempt_linkage` wymaga jednej faktycznej zwycięskiej próby,
exact signed request identity i odrzucenia innego attempt jako exact retry.
`lost_response_reconciliation` wymaga odzyskania oryginalnych bajtów; brak
rekordu nie stanowi autorytatywnego `UNBOUND`.

Jednocześnie zamrożony
[production substrate selection](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json)
zawiera literalne
`separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles=true`.
Wspólny fizyczny commit podległych persistence schemas issuera jest zgodny
z wymaganą atomową decyzją, lecz nie usuwa niejednoznaczności tego literalnego
wymogu odrębnych transakcji. Włączenie produkcyjne wymaga osobno przejrzanej
reconciliation zakresu tego pola. Nie wolno po cichu uznać kolokacji za zgodę
na połączenie odrębnych autorytetów ani zmienić historycznego freeze.

[Implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json)
utrzymuje niedostępne zależności produkcyjne i
`non_production_implementation_mode=NONE_FROZEN`. Nowy testowy mechanizm
persistence nie zmienia tego freeze i nie ustanawia dostępnego semantycznego
RootProofIssuer. Publiczny provider gate pozostaje zamknięty.

[Physical persistence / crash atomicity](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json)
dotyczy własnych faz i lokalnych write sets CHA oraz niezależnej decyzji
FreshnessAuthority. Decyzja issuera nie jest `AccountGenesis COMMITTED`,
`PREPARED`, consume reservation, freshness CAS ani publikacją konta.
Checkpoint pozostaje odrębną, późniejszą transakcją. Nie istnieje wspólny
commit issuer/CHA/checkpoint/freshness.

## Audyt istniejących alternatywnych zapisów

| API | Zachowanie przed migracją | Wymaganie nowej granicy |
| --- | --- | --- |
| `append_bind` | Entitlement runtime może sam wykonać BIND; SQL sprawdza lineage CAS i `ACTIVE/UNBOUND`, lecz nie zachowuje podpisanego proof ani issuer head. Guard nie sprawdza własnego `session_user`. | Stary runtime traci samodzielną możliwość BIND. Jedynym nowym wejściem do emisji jest atomic coordinator i helper należący do entitlement ownera. |
| `append_record` | History runtime może dopisać event w własnej transakcji; predecessor CAS i immutable record nie tworzą wspólnego BIND. | Standalone runtime nie może utworzyć autorytatywnego issuance event z pominięciem coordinatora. |
| `retain_head` | History runtime może zachować signed head w kolejnej transakcji; SQL wiąże bajty z rekordem, ale weryfikacja kryptograficzna należy do trusted runtime. | Atomic head i issuance record zachowuje ta sama transakcja coordinatora. Nie wystarcza dodatkowy grant do starej funkcji. |
| `reserve_intent` | Zachowuje immutable `RESERVED`, oryginalne payload bytes i identity; nie podpisuje i nie wykonuje BIND. | Pozostaje legalnym przygotowaniem. Sam RESERVED nigdy nie autoryzuje emisji. |
| `retain_prepared` | Zachowuje immutable `STAGED`, exact payload/signature/envelope `BYTEA`; `require_issued()` odmawia. | Pozostaje legalnym przygotowaniem. Sam STAGED i poprawny podpis nie ustanawiają decyzji. |

History, intent i retention mają direct-login guards: `session_user` pozostaje
loginem po `SET ROLE` oraz przy zagnieżdżonym `SECURITY DEFINER`. Nie wolno
udawać, że stara runtime rola jest callerem nested helpera. Nowy helper
sprawdza stałego issuer writera i własnego semantic schema ownera; wejście
coordinator ownera egzekwuje jego wąskie ACL. Writer nie ma bezpośredniego
`EXECUTE` na helperze, DML, DDL ani role membership.

Stare kwalifikatory wymagają dokładnego inventory, ACL, role OIDs i źródeł.
Nowa funkcja, odwołany stary grant, peer helper grant lub zmieniona definicja
oznaczają nową jawną wersję kontraktu. Stara kwalifikacja musi odrzucać
migrowany schemat; arbitralne wyjątki w V1 nie są migracją.

## Administracyjna aktywacja i zachowanie danych

Nowy schemat ma identity
`cryptohunter.atomic_root_proof_commit.postgresql`, `schema_version=1`
i migration ID `ATOMIC_ROOT_PROOF_AUTHORITY_V1_COMPONENT_INVENTORY_V2`.
Konfiguracja administracyjna przypina konkretne nazwy siedmiu schemas:
coordinator, entitlement, signing intent, retention, authenticated history,
requester credentials i claimant credentials.
Osobne role to coordinator owner `NOLOGIN`, issuer writer `LOGIN`
i test verifier/admin `LOGIN`; ownerzy component schemas zachowują swoją
semantyczną własność. W V2 ownerzy muszą być `NOLOGIN`. Rola admin nie jest
credentialem writera ani produkcyjnego issuera.

[SQL migration](../../bot_core/postgresql_atomic_root_proof_schema.py)
instaluje owner-owned `atomic_bind`, `atomic_append_record`,
`atomic_retain_head`, wąskie exact readers oraz canonical parser facades.
Writer wykonuje tylko `commit_root_proof(bytea)` i
`read_decision(bytea,bytea)`, `read_approved_identity(bytea)`, odczytowe
`qualify_catalog()` oraz czyta metadata. Otrzymuje `USAGE`, bez
bezpośredniego DML, DDL, helper `EXECUTE` i memberships. Oryginalne standalone
grants `append_bind`, `append_record` i `retain_head` zostają odwołane.
`reserve_intent` oraz `retain_prepared` zachowują dotychczasowy sens
przygotowania. Deferred constraints wiążą nowe BOUND i history records/heads
z decyzją tej samej transakcji.

Odrębne [preparation adapters V2](stage9_postgresql_v2_root_proof_preparation.md)
umożliwiają nowe genuine RESERVED i STAGED po migracji w jawnie
zakwalifikowanym, izolowanym wdrożeniu testowym. Dotychczasowe V1 kwalifikatory
nadal odrzucają schemas z dodatkowymi helpers i zmienionymi ACL; jest to celowy
fail-closed. `prepare_atomic_root_proof_commit()` zachowuje swoją ścieżkę
przed migracją; ten sam entrypoint akceptuje także exact zgodny zestaw nowych
portów V2 i ponownie weryfikuje ich genuine przygotowanie. Mieszany zestaw
portów odmawia. Preparation principal zachowuje własne wąskie SQL grants,
a jawna migration `ROOT_PROOF_RESERVED_STAGED_PREPARATION_ACL_V2` dodaje tylko
kwalifikację odczytową physical identity i exact enrolled ACTIVE signer fence.
Nowe preparation może być przekazane do tego samego qualified approval oraz
`commit_root_proof(bytea)`. Nie ustanawia produkcyjnego pipeline emisji.
Jawny `resolve_root_proof_issuance_authorization_v2(binding, authority)` tworzy
nową genuine autoryzację z read-only public capability V2; nie zastępuje
zablokowanego produkcyjnego selector ani nie przepisuje wcześniejszych signed
attempt bytes. Durable preparation metadata oraz administracyjna odczytowa
`requalify_root_proof_preparation_v2()` pozwalają odtworzyć pin po restarcie i
odzyskać exact original RESERVED/STAGED po unknown preparation result.

Odmowa nowej kwalifikacji nie oznacza, że każdy wcześniej utworzony obiekt
odmawia wszystkich operacji. Entitlement provider kwalifikuje się w konstruktorze;
cache'owany obiekt sprzed migracji może nadal czytać oraz wykonywać legalne admin
`provision/revoke/supersede`. Odebrany grant `append_bind` blokuje jego standalone
BIND. Legacy intent, retention, history i pre-account credentials ponawiają
kwalifikację operacji i odmawiają inventory V2. Nowe odrębne porty V2 obejmują
tylko kwalifikowane preparation i potrzebne publiczne/history reads, bez
standalone BIND, history writes, checkpoint lub ogólnej administracji
sześciu starych autorytetów. Wewnętrzny atomic coordinator zachowuje jedyny
finalny write set; publiczna produkcyjna kompozycja pozostaje zablokowana.

[Kwalifikacja V2](../../bot_core/postgresql_atomic_root_proof_qualification.py)
przypina role OIDs, physical identity, schema identity/version oraz dokładny
katalog siedmiu schemas: źródła i właściwości funkcji, ACL, relation/index
inventory, column definitions, constraints, triggers i auxiliary objects.
Pin powstaje administracyjnie po kwalifikacji oraz porównaniu reviewed
sources; writer nie wybiera nowego manifestu. Drift odmawia wejścia.

Oddzielne `approve_commit(jsonb)` jest wejściem kwalifikowanego testowego
verifiera, który sprawdza genuine INITIAL_BINDING/signed attempt,
Root Proof bytes i signed head poza finalną transakcją. Immutable approval
zachowuje exact package. Commit SQL przyjmuje wyłącznie attempt locator,
odczytuje to approval oraz weryfikuje exact RESERVED/STAGED, BOUND i history
linkage. Writer nie może sam zatwierdzić caller-owned evidence.

Migracji nie uruchamia start aplikacji ani nieznane `schema_version`. Wykonuje
ją administrator w odizolowanej instancji testowej przez jawne narzędzie
provisioning/migration. Produkcyjny DSN, credential ani globalny `test_mode`
nie mogą stać się takim upoważnieniem.

Przed aktywacją należy kwalifikować starsze schemas i przygotować staging
copy. Porównanie obejmuje istniejące IDs, entitlement generation/revision,
scope, dokładne canonical/signature/envelope bytes, history chain, credential
references oraz dotychczasową interpretację digestów i statusów. Po zmianie
kwalifikacja obejmuje nowy manifest, katalog, ACL, constraints, helper sources
i powiązanie z jednym kwalifikowanym DB16. Administracyjny completion record
i checkpoint po aktywacji należą do przyszłej produkcyjnej procedury,
a nie do normalnego runtime.

Migracja V1 bezwzględnie odmawia, jeżeli entitlement history zawiera
wcześniejszy `BOUND` albo authenticated history zawiera records lub
attestations. Nie zmienia tych danych ani nie tworzy fikcyjnego marker.
Sama zmiana katalogu nie ustanawia atomowości wcześniejszej decyzji.
Obsługa wcześniej wyemitowanych danych wymaga osobno przejrzanej migracji,
która uwierzytelni i zachowa pełne oryginalne proof/history linkage.
Ten wariant zachowuje istniejące UNBOUND, credentials i RESERVED/STAGED
w odizolowanej bazie testowej. Niezgodność inventory, statusów lub
oryginalnych bajtów kończy migrację odmową. Nie wykonuje się destrukcyjnej
migracji produkcyjnej.

Świadome zmiany ACL to `REVOKE TEMPORARY ON DATABASE ... FROM PUBLIC`
oraz `REVOKE UPDATE ON TABLE pg_catalog.pg_settings FROM PUBLIC`
w tej odizolowanej bazie testowej. Domyślny PUBLIC TEMP umożliwiałby writerowi
DDL tymczasowych obiektów, a PUBLIC UPDATE na systemowym widoku pozostawiałby
bezpośredni grant DML poza zamkniętym API. Odwołania dotyczą wszystkich
użytkowników tej bazy, więc migracji nie wolno uruchamiać jako niejawnej zmiany
współdzielonej bazy produkcyjnej. Catalog manifest i kwalifikacja przypinają
wynikowy DB ACL oraz ACL systemowego widoku; writer i test verifier nie mogą
odzyskać CREATE, TEMPORARY ani tego bezpośredniego UPDATE.

Kontrola retained bytes i reviewed sources odbywa się pod administracyjnymi
table locks w jednej transakcji migracji. Rzeczywisty direct-login nowej roli
writera jest kwalifikowany po jej zatwierdzeniu, ponieważ nowe role są
widoczne innemu połączeniu dopiero po COMMIT. Jeśli ten ostatni probe odmawia,
instalacja może pozostać trwała, lecz nie jest dostępna; błąd nie oznacza
cofnięcia wcześniejszego administracyjnego COMMIT. Potrzebna jest jawna
administracyjna diagnoza i kwalifikacja, bez runtime auto-repair.

## Fizyczna granica i wymagany write set

Jedno połączenie należy do jednego direct-login issuer writera. Jedna
fizyczna transakcja `SERIALIZABLE` na PostgreSQL 16 obejmuje:

1. Exact operation/attempt/entitlement identity oraz authority-owned
   serialization key i blokady wymaganych rekordów.
   Requester i claimant należą do własnych ownerów: ich fixed
   `atomic_lock_credential(jsonb)` helpers utrzymują istniejący wspólny
   admin alias advisory lock oraz current credential row locks do końca
   tej samej transakcji. Porównują credential ID, principal, semantic role,
   credential role, key ID/version/material, scope, lifecycle generation
   i registry revision, wymagając bieżącego `ACTIVE`.
2. Aktualną entitlement generation/revision i `ACTIVE/UNBOUND`, pod blokadą
   lineage kompatybilną z revoke/supersede.
3. Exact immutable RESERVED/STAGED linkage do oryginalnych payload,
   signature i envelope `BYTEA`, wraz z request identity, scope, root ID,
   signer credential i key version.
4. Expected predecessor history CAS i dokładny, wcześniej podpisany head.
5. Entitlement CAS BIND, registry history, authenticated issuer record,
   signed head i immutable decision marker wskazujący zwycięski attempt.
6. Jedną decyzję `COMMIT` albo pełny `ROLLBACK` wszystkich nowych zapisów.

Immutable FK do chronionych, unikalnych rekordów w tej samej bazie zachowuje
oryginalne bajty równoważnie ich kopii. Sam digest, niechroniona referencja
lub link do innej fizycznej bazy nie spełnia wymogu. Stage nie zmienia stanu
ani bajtów przez `UPDATE`; marker odróżnia zachowanego kandydata od decyzji.
Constraints muszą gwarantować unikalność entitlement generation, root ID,
operation oraz attempt w zamrożonym scope.

Obecny V1 signing intent ma globalne `UNIQUE(operation_key)`,
`UNIQUE(entitlement_key)` i `UNIQUE(account_key)`. Retention ma odpowiadające
unique constraints w scope. Immutable RESERVED/STAGED może więc przypiąć
entitlement zanim dojdzie do BIND. Dwie różne próby nie mogą równocześnie
zostać legalnymi STAGED kandydatami tego samego entitlement w tej authority.
Race dwóch rezerwacji rozstrzyga się już w przygotowaniu; race dwóch exact
wywołań commitu tego samego attempt dowodzi jednej nowej decyzji i replay.
Żaden z tych testów nie zastępuje race dwóch różnych legalnych finalnych
attempts. Ten wymagany scenariusz pozostaje niespełniony przy zachowanej
kardynalności V1.

Zestaw integration zawiera także dwóch niezależnie, genuine przygotowanych
kandydatów różnych entitlement i operations, ze wspólnym history predecessor.
Race ich finalnych commitów ma sprawdzić jednego history CAS winnera i pełny
rollback BIND przegranego entitlement. Odrębny scenariusz najpierw zachowuje
prawidłowy history record/head pierwszego kandydata, następnie wysyła drugi
genuine kandydat z faktycznie nieaktualnym predecessor; oczekiwana odmowa
pozostawia jego entitlement UNBOUND i bez marker. Te scenariusze sprawdzają
serializację wspólnego history stream i rollback, lecz nie zastępują finalnego
race dwóch attempts tego samego entitlement. Ich wyniki wymagają wykonania
testów PostgreSQL; opis scenariuszy nie deklaruje wyniku bieżącego rerun.

Zamrożony M0.5 wymaga `two_operations_one_entitlement: one winner; loser
REJECT` i pozwala na durably superseded replacement po autorytatywnym
UNBOUND. Nie zakazuje wszelkich różnych attempts. Obsługa dwóch legalnych
finalnych kandydatów wymaga osobno przejrzanej, jawnej wersji przygotowania
oraz replacement protocol; nie wolno usunąć starych unique constraints jako
nieudokumentowanego skrótu. Obecny wariant zachowuje bezpieczeństwo kosztem
liveness opisanej w V1 retention.

`system_identifier` i database OID są koniecznymi pinami lineage, lecz
sklonowane `PGDATA` mogą zachować oba. Równe DSN i snapshoty nie dowodzą jednej
transakcji. Dowodem fizycznej atomowości jest wykonanie rzeczywistego write
set na jednym backendzie i trwały odczyt po race/crash/restart, przy
kwalifikowanych katalogach oraz jednej odizolowanej instancji.

## Lifecycle fencing i granica produkcyjna

Wcześniejszy lokalny snapshot `ACTIVE` i lokalny custody lock nie są fence
finalnego DB commitu. Śmierć klienta może zwolnić lock przed ukończeniem
COMMIT przez backend. Prywatny klucz pozostaje poza SQL; finalna transakcja
nie trwa podczas signing ani operacji custody.

Produkcja wymaga DB-authoritative fences issuer signer, history signer oraz
trust activation, z backend locks utrzymanymi aż do COMMIT/ROLLBACK. Ponadto
requester i claimant muszą uczestniczyć w istniejącym alias advisory lock
oraz blokadach current credential rows. Admin revoke/rotation/activation
stosuje kompatybilny lock order. `VERIFY_ONLY`, `REVOKED`, nieaktualna
generation i brak kwalifikowanego fence odmawiają nowego BIND.

Administracyjne `PENDING_CLOSED` musi być trwale zatwierdzone i rozstrzygnięte
przed zmianą custody. Dopiero zweryfikowane completion tej samej operacji
może opublikować successor. Crash pozostawia fence zamknięty. `REVOKED`
nie przechodzi do `ACTIVE` ani `VERIFY_ONLY`. Testowy DB fence nie dowodzi,
że wszystkie produkcyjne ścieżki custody i trust activation go respektują.
Brak pełnego uczestnictwa pozostaje jawnym blockerem publicznego commitu.

Wewnętrzny wariant testowy blokuje trzy własne DB fences signerów i trust
activation oraz genuine requester/claimant `current_credentials` w ich
istniejącym admin alias lock order. Helpers nie zapisują credential records;
semantyczną własność i dotychczasowy lifecycle zachowują pre-account ownerzy.
Ich porównanie pełnego exact snapshotu obejmuje także public key material
fingerprint. Wcześniejsze ACTIVE przygotowanie nie zastępuje finalnej
kontroli bieżącego DB stanu. Pełna produkcyjna integracja local custody
oraz trust activation nadal wymaga odrębnej kwalifikacji.

## Recovery i ograniczenia dowodów

Wyjątek przed wysłaniem COMMIT wymaga rollback. Rozłączenie podczas COMMIT
lub utrata odpowiedzi po nim oznacza `INDETERMINATE`. Recovery nie mintuje
Root Proof ID, nie podpisuje ponownie, nie kompensuje przez UNBIND i nie
dopisuje drugiego history event.

Exact retry porównuje ten sam immutable approved package, odczytuje
ten sam marker i odzyskuje oryginalne bajty. Pełne proof/registry/history
linkage ustanawia fizyczny write set z chronionymi constraints przy COMMIT.
Marker read nie zastępuje późniejszej kryptograficznej weryfikacji całej
history i checkpoint ani publicznego recovery. Zmieniony attempt, entitlement,
generation, operation, account, request lub root ID odmawia. Brak marker
nie dowodzi, że wcześniejszy COMMIT nie nastąpił. Rozstrzygnięcie trwającego
wyniku korzysta z tego samego authority-owned serialization key;
`NOT_FOUND` pozostaje nieautorytatywnym wynikiem odczytu.

Immutable approval ogranicza liveness po stale history CAS. W V1
`approved_commits` ma jeden package per attempt, obejmujący expected predecessor,
history record i signed head; UPDATE/DELETE odmawia, a `approve_commit()`
odrzuca inny package dla tego samego attempt. Jeżeli inna decyzja wyprzedzi
tego kandydata w history stream, powtórzenie identycznego approval pozostaje
stale. Obecny API nie zachowuje nowego history successor dla tego samego proof
i attempt. Entitlement przegranego pozostaje UNBOUND, a RESERVED/STAGED
zachowują oryginalne bajty; bezpieczeństwo nie ustanawia tu postępu emisji.
Odblokowanie wymaga jawnie wersjonowanego reproposal protocol dla history
candidate, z zachowaniem audytu poprzedniego approval i tych samych proof,
attempt oraz request identities. Nie wolno usuwać approval, mintować
zastępczego Root Proof ani ponownie podpisywać Root Proof jako skrótu recovery.

Odzyskanie decyzji DB jest odrębne od publicznego authenticated recovery
i CHA admission. History/checkpoint authentication oraz reguły historycznej
kwalifikacji klucza nadal obowiązują. `VERIFY_ONLY` może weryfikować exact
historyczny winner, ale nie autoryzuje nowego BIND. Podpis `REVOKED` bez
niezależnie zaufanej evidence sprzed compromise nie tworzy zaufania.

Issuance event zachowuje exact istniejącą projekcję
`_expected_issuance_history_payload`, z winning attempt identity digest
i registry predecessor CAS. Oryginalne proof bytes zachowuje immutable
decision marker. Nie dodajemy pól do history payload, które zmieniłyby jego
interpretację albo wynik pełnego porównania publicznego authenticated readera.
Publiczne recovery pozostaje zablokowane do późniejszej kwalifikacji
checkpoint i pełnej kompozycji.

## Pozostałe invariants przed produkcyjną kompozycją

- Pełna izolacja produkcyjnego issuer service, credentials i namespace od
  CHA, requester, claimant oraz fixture testowego.
- Uczestnictwo wszystkich custody/trust/admin paths w DB-authoritative
  lifecycle fencing i recovery `PENDING_CLOSED`.
- Osobno przejrzane rozstrzygnięcie literalnej separacji transakcji
  w zamrożonym substrate contract.
- Administracyjna staging migration, canonical-byte comparison,
  authenticated completion oraz checkpoint activation istniejących danych.
- Legalne przygotowanie dwóch różnych kandydatów tego samego entitlement
  i rzeczywisty race ich finalnych commitów, po jawnej migracji obecnej
  kardynalności RESERVED/STAGED oraz kwalifikacji replacement protocol.
- Produkcyjna kwalifikacja izolowanych preparation adapters inventory V2
  oraz jawny history candidate reproposal po stale predecessor,
  zachowujący immutable approval history i oryginalne Root Proof bytes/identities.
- Późniejszy checkpoint acceptance z własnym lifecycle fence oraz
  authenticated byte-identical recovery i CHA admission.
- Weryfikacja rollback-safe restore; ordinary durability nie dowodzi
  niezależnej ochrony przed skoordynowanym full-host rollback.

Wyniki rzeczywistych testów oraz status `ATOMIC_COMMIT_TESTED` muszą wynikać
z wykonanych scenariuszy PostgreSQL 16 i są raportowane w PR. Ten dokument
nie zastępuje ich wyniku ani nie promuje Stage 9 do READY.
