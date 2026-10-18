# Stage 9 — S9-RPI-01: production RootProofIssuer lifecycle isolation

## Zakres i decyzja

S9-RPI-01 przygotowuje granicę izolacji oraz DB-authoritative lifecycle
fencing dla przyszłego `PRODUCTION_LOCAL` issuera. Nie ustanawia dostępnego
Semantic RootProofIssuer, nie wykonuje produkcyjnego BIND ani enrollmentu
i nie zmienia zamrożonych kontraktów M0.5. Rekomendacja dla publicznej emisji
pozostaje **NO-GO**.

```text
semantic_root_proof_issuer_runtime = NOT_IMPLEMENTED / BLOCKED
production_provisioning_ready = false
ROOT_PROOF_ISSUANCE_AVAILABLE = false
ATOMIC_ISSUER_READY = NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION = BLOCKED
```

Implementacja kodu, wykonanie kwalifikacji na izolowanym PostgreSQL 16 oraz
autoryzacja rzeczywistego wdrożenia są trzema odrębnymi wynikami. Testy nie
provisionują produkcyjnego principal, nie tworzą produkcyjnych kluczy i nie
stanowią zgody na zmianę powyższych statusów. Stage 10 pozostaje nietknięty.

## Podstawa audytu

Punktem wyjścia jest `origin/main` po PR #3148,
`f50b0186`; gałąź uwzględnia także późniejsze poprawki main z #3149
(`b0f04ee6`). Sprawdzono bezpośrednio dziesięć wymaganych źródeł:

- `bot_core/postgresql_atomic_root_proof_commit.py`;
- `bot_core/postgresql_atomic_root_proof_schema.py`;
- `bot_core/postgresql_atomic_root_proof_qualification.py`;
- `bot_core/postgresql_atomic_root_proof_migration.py`;
- `bot_core/postgresql_custody_lifecycle.py`;
- `bot_core/postgresql_custody_lifecycle_schema.py`;
- `bot_core/local_signing_custody.py`;
- `bot_core/postgresql_preaccount_credentials.py`;
- `bot_core/root_proof_issuer_runtime.py`;
- `bot_core/root_proof_issuer_orchestration.py`.

Odczytano także [Atomic Commit Authority V1](stage9_atomic_root_proof_commit_authority_v1.md),
[PostgreSQL V2 preparation](stage9_postgresql_v2_root_proof_preparation.md)
i [qualified atomic boundary](stage9_qualified_atomic_root_proof_boundary.md).
Ostatni dokument opisuje wcześniejszy wybór architektoniczny po PR #3110;
jego historyczne stwierdzenie o braku zainstalowanego coordinatora nie opisuje
późniejszej, wykonywalnej implementacji testowej V1/V2. Źródła pozostają
rozstrzygające przy porównaniu kolejnych etapów.

Zamrożone wejścia to:

- [independent issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json):
  `bind_authorization_order`, `credential_rotation_recovery`,
  `issuer_bound_attempt_linkage`, `lost_response_reconciliation`;
- [production substrate selection](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json):
  `production_local_substrate`, `authority_separation`, `profile_isolation`,
  `least_privilege`, `migrations`;
- [implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json):
  provider gate, rzeczywista izolacja środowisk i warunki dostępności;
- [physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json),
  [admission binding](cryptohunter_product_architecture/m05_account_genesis_root_proof_admission_binding_contract.json)
  oraz [authority topology](cryptohunter_product_architecture/m05_account_genesis_authority_topology_resolution_after_binding_freeze.json):
  odrębna własność AccountGenesis, lokalnych transakcji CHA i FreshnessAuthority;
- [signed immutable attempt freeze](cryptohunter_product_architecture/stage9_root_proof_signed_immutable_attempt_freeze.json)
  oraz [pre-account credentials freeze](cryptohunter_product_architecture/stage9_root_proof_preaccount_credentials_freeze.json):
  produkcyjne statusy pozostają zablokowane.

Codebase Memory MCP nie jest callable w tej sesji. Audyt używa celowanych
wyszukiwań i rzeczywistych źródeł; nie deklaruje kompletności nieobecnego
indeksu ani kwalifikacji wdrożenia.

## Gwarancje istniejące przed S9-RPI-01

| Obszar | Wykonywalna gwarancja | Granica dowodu |
| --- | --- | --- |
| Atomic commit V1 | Jeden backend i jedna finalna transakcja `SERIALIZABLE`; exact RESERVED/STAGED, entitlement CAS, BIND, registry history, issuer event/head i decision marker; protected original `BYTEA` i deferred closure constraints | `PostgreSQLAtomicRootProofTestDeployment`, `ISOLATED_TEST_ONLY` i oddzielny testowy verifier approval; brak produkcyjnej autoryzacji |
| SQL identities | Direct login `session_user = current_user` przy wejściu, OID, rozłączne role, ownerzy `NOLOGIN`, brak memberships i DML/DDL writera; owner-owned helpers z fixed `search_path` | Zaufany provisioning izolowanej instancji; nazwa roli i DSN same nie dowodzą odrębnego procesu lub izolacji OS |
| Catalog qualification | Dokładne źródła i właściwości funkcji, ACL, relacje, constraints, triggers, indexes, auxiliary objects, durable settings i physical coordinates | Zaufany manifest po jawnej migracji; nie jest dowodem, że produkcyjny deployment i wszystkie ścieżki administracji uczestniczą w fence |
| Final lifecycle checks | Root signer, history signer i trust activation mają trzy różne fence rows; commit blokuje je w sortowanym porządku i wymaga exact role, `ACTIVE` oraz generation | Fences należą do testowego coordinatora; nie wolno awansować ich do produkcji przez zmianę flagi lub nazwy |
| Requester/claimant | Wspólny admin alias advisory transaction lock, następnie exact current credential row locks i porównanie identity, public material, generation/revision oraz `ACTIVE` | Poprawność wymaga zachowania tego samego namespace locka i kolejności także po stronie administracji |
| Custody Lifecycle V2/V3 | Potwierdzony DB commit `PENDING_CLOSED` poprzedza mutation custody; exact durable operation i uwierzytelnione completion; jedynie przejścia do `VERIFY_ONLY` lub `REVOKED`; restart nie otwiera fence | Qualification i enrollment mają granicę `ISOLATED_TEST_ONLY`; brak produkcyjnej aktywacji/rotacji successorów i trust activation |
| Local signing custody | Oddzielne root/history credential, namespace i material; Ed25519 i snapshot `ACTIVE` pod rzeczywistą blokadą custody; root/history mutacje wymagają DB-first admission | Sama blokada plikowa zwalnia się po śmierci klienta i nie może zastąpić blokad backendu DB do końca COMMIT |
| Recovery | Exact decision odczytuje oryginalne payload/signature/envelope; nie podpisuje ponownie; brak marker i utrata odpowiedzi pozostają `INDETERMINATE` | Odczyt decision nie zastępuje kryptograficznej history/checkpoint authentication ani publicznego produkcyjnego recovery |
| History Successor Reproposal V1 | Wersjonowane immutable candidates, supersession i nowy history head dla tych samych oryginalnych Root Proof bytes oraz attempt; jeden finalny atomic write set | Osobny kwalifikowany inventory `ISOLATED_TEST_ONLY`; brak produkcyjnego ingress i lifecycle qualification |
| Runtime/orchestration | Genuine InitialBinding i immutable signed attempt; ponowne publiczne weryfikacje; execution-blocked bez sign/BIND/history/checkpoint/CHA finalize/replacement/transport | Raport i powtórzone odczyty nie tworzą globalnego snapshotu ani capability emisji |

V2 preparation pozostawia oddzielne legalne `RESERVED` i `STAGED` commity.
Nie wolno łączyć istniejących portów przez zewnętrzne `with transaction()`:
ich własne połączenia zachowują własne transakcje. Finalny coordinator już
istnieje; S9-RPI-01 nie implementuje go ponownie.

## Nowa, niezależna granica produkcyjnego lifecycle

`bot_core/postgresql_production_root_lifecycle_schema.py` definiuje odrębny
production-only SQL authority o identity
`cryptohunter.production_root_lifecycle.postgresql`. Nie adaptuje
`PostgreSQLAtomicRootProofTestDeployment` i nie zmienia znaczenia testowych
V1/V2 capabilities. Jawny, odrębny bootstrap principal instaluje trzy publiczne
bindingi root signer, history signer i trust activation, każdy początkowo
`PENDING_CLOSED`. Żaden publiczny entrypoint tej zmiany nie ustawia `ACTIVE`.
W PostgreSQL są wyłącznie publiczne identities/material fingerprints,
generations oraz exact operation/completion receipts; brak signing keys.

Każda rola ma authority-owned slot wskazujący bieżącą credential identity.
`assert_active_fences(jsonb)` wymaga kompletnego zestawu trzech exact bindings
w writable `SERIALIZABLE`, blokuje wszystkie slots w stałym porządku semantic
role, następnie ich credential rows. Porównuje authority, identity, material,
key version i generation oraz wymaga `ACTIVE` bez pending operation.
Locks pozostają w transakcji caller backendu; admin bierze slot przed fence.
Ten krok musi poprzedzać wspólny requester/claimant alias advisory lock
i current row locks istniejącego coordinatora.

`begin_close` trwale wiąże immutable operation z exact predecessor i zamyka
fence. `read_operation` wymaga `READ COMMITTED`, blokuje slot i dopiero potem
odczytuje operation, aby rozstrzygać nadal trwający backend wcześniejszego
COMMIT. `complete_close` przyjmuje jedynie dokładny receipt tej operacji od
oddzielnego direct-login administratora; dopisuje completion i zwiększa
generation przejściem do `VERIFY_ONLY` lub `REVOKED`. Nie otwiera starego fence.
Zapisany `close_xid` uniemożliwia completion lub pozytywne resolution w tej
samej transakcji, która dopiero tworzy close. Koordynator administracyjny
potwierdza własny COMMIT przed zwróceniem `PENDING_CLOSED`; nie wywołuje custody.

`enroll_closed_successor` wymaga zakończonego retirement poprzednika, nowej
identity i public material oraz rosnących key version/generation. Successor
pozostaje `PENDING_CLOSED`. To bezpieczny punkt przygotowania rotacji,
a nie gotowy protokół aktywacji produkcyjnego signera lub trust bundle.
SQL receipt dowodzi deklaracji uwierzytelnionego administratora; sam SQL
nie odczytuje custody. Produkcyjny verifier rzeczywistego custody completion
i włączenie faktycznego custody do koordynacji muszą zostać osobno podłączone
i zakwalifikowane przed aktywacją.

`read_historical_fence` może odczytać zachowaną obserwację dla credential
`ACTIVE` lub `VERIFY_ONLY`; nie autoryzuje nowej operacji. `REVOKED` i brak
historycznej obserwacji nie tworzą pozytywnego dowodu historycznej trust.
Ta obserwacja lifecycle nie zastępuje original proof/history/checkpoint.

## Instalacyjna granica procesu

`bot_core/production_root_lifecycle.py` otwiera granicę wyłącznie z fixed
`/etc/cryptohunter/root-proof-issuer/installation.json`. Nie ma parametru
requestera wybierającego installation path, profile, principal, DB lub signer
socket. Loader otwiera każdy element ścieżki przez deskryptor z `O_NOFOLLOW`,
wymaga root ownership oraz braku group/world write i odrzuca zmianę pliku
w czasie odczytu. Duplicate JSON fields i inne profile odmawiają.

Wymagane top-level pola to dokładnie `schema_version=1`,
`profile=PRODUCTION_LOCAL`, `provisioning`, `identity`, `catalog_sha256`,
`role_oids`, `principals` i `signer_sockets`. Każdy wpis principal zawiera
`unix_user`, `uid`, `gid`, `postgresql_role` i `postgresql_oid`.
`identity.system_identifier` jest dokładnym dziesiętnym stringiem dla
PostgreSQL uint64. Plik jest pojedynczym regular file, ma jeden hardlink
i rozmiar najwyżej 64 KiB. Te dane instaluje administrator; caller operacji
nie może dostarczyć ich jako capability lub override.

Installation przypina siedem różnych systemowych kont/UID/GID i PostgreSQL
role/OID: issuer, administrator, CHA, requester, claimant oraz oba signery.
Sprawdza rzeczywiste `pwd`/group membership i rzeczywiste UID/GID procesu,
DB direct-login roles/OID/attributes oraz memberships. CHA, requester,
claimant i signery nie otrzymują dostępu do lifecycle schema. Osobne signer
Unix sockets mają chronioną ancestry; `SO_PEERCRED` weryfikuje rzeczywisty
odrębny PID/UID/GID, a nie etykietę providera. Socket należy do UID signera
i GID issuera z mode `0660`; klucz ani wykonanie operacji signing nie są
wnioskiem wynikającym z samej obserwacji peer identity.

Entrypoint `deployment/production_root_proof_issuer.py --check` i przykładowy
unit `deployment/systemd/cryptohunter-root-proof-issuer-check.service`
przygotowują osobny, ograniczony Linux proces kwalifikacji. Obserwuje on
instalacyjne tożsamości i lokalne połączenie PostgreSQL oraz oba signer
processes. Połączenie wybiera fixed lokalny Unix endpoint z instalacji,
explicit login i parametry połączenia, bez caller DSN. Nie implementuje
protokołu podpisu ani obsługi publicznych żądań.
Każda operacja zainstalowanej granicy wymaga PostgreSQL 16 `system_user`
równego dokładnie `peer:<installed unix_user>`. Sam lokalny socket, requested
DB user albo `pg_hba.conf` z `trust` nie przechodzą tej kontroli.
Odrębny kwalifikator katalogu SQL nie stanowi dowodu OS-to-DB authentication.
Rzeczywiste wdrożenie nadal wymaga przeglądu całego `pg_hba.conf`, `pg_ident.conf`
i wszystkich endpointów, aby wykluczyć alternatywne logowanie do tych samych
ról. Nie wykonano tego przeglądu na fizycznym wdrożeniu; pozostaje blokadą.
Przykładowy unit i brak rzeczywistej instalacji nie są dowodem wdrożenia.
Publiczna metoda `commit()` tej granicy zawsze odmawia produkcyjnego BIND.

## Niezależna kwalifikacja SQL

`bot_core/postgresql_production_root_lifecycle_qualification.py` porównuje
live authority z `PostgreSQLProductionRootLifecyclePin` pochodzącym
z chronionej instalacji. Pin nie zawiera bootstrap/admin credential.
`pin_production_root_lifecycle_catalog` działa wyłącznie dla oddzielnego
direct-login bootstrap superusera, wobec wcześniej wybranej physical identity
i reviewed SQL. Runtime nie wybiera nowego catalog hash i nie naprawia drift.

Kwalifikacja sprawdza PostgreSQL 16 primary, `fsync`, `synchronous_commit`,
`full_page_writes`, physical database/service coordinates, dokładny direct
login i role OID, minimalne attributes, owner `NOLOGIN` oraz brak memberships.
Wymaga exact źródeł/signatures, ownerów, language, `SECURITY DEFINER`,
`search_path` i ACL znanego katalogu funkcji. Weryfikuje definition fields
relations/columns/constraints/indexes/triggers/rules/policies i typów, role
settings oraz globalne paths uprawnień, w tym PUBLIC i column ACL.
Maintenance statistics nie stanowią catalog definition i nie służą jako pin.

Issuer i administrator nie mogą mieć bezpośredniego table/column DML,
database/schema DDL/TEMP, ownership poza kontrolowanym ownerem ani dostępnych
zewnętrznych `SECURITY DEFINER` ścieżek. Nowa kwalifikacja nie opiera się na
`SET ROLE`, rozszerzonym grancie ani klasie nazwanej production. Stara
kwalifikacja testowa i jej źródła pozostają odrębne; współdzielony reader
physical coordinates jest wyłącznie odczytem faktów PostgreSQL.

## Nowy punkt integracji istniejącego finalnego commitu

`bot_core/postgresql_atomic_root_proof_transaction.py` obserwuje rzeczywistego
właściciela rozpoczętej finalnej transakcji: obiekt połączenia, backend PID,
PostgreSQL transaction ID, database OID/name oraz direct-login role/OID.
Wymaga writable `SERIALIZABLE` i `session_user = current_user` dla oczekiwanego
loginu. Nie przyjmuje callbacku, nie rozpoczyna transakcji i nie wykonuje
COMMIT lub retry.

Istniejący `PostgreSQLAtomicRootProofCommit.commit()` zachowuje tę obserwację
po swojej niezależnej kwalifikacji oraz sprawdza continuity bezpośrednio
przed jedynym finalnym COMMIT. Podmienione połączenie, `SET ROLE` lub pośredni
COMMIT odmawiają. Token nie jest capability emisji: nie zastępuje kwalifikacji
physical instance, catalog, lifecycle ani immutable attempt/entitlement.
Udostępnia wąski punkt dla przyszłego production adaptera, bez przyjmowania
testowego deploymentu jako produkcyjnego upoważnienia.

Continuity nie dowodzi, że caller zachował wszystkie row locks po dowolnym
`ROLLBACK TO SAVEPOINT`. Finalny coordinator musi posiadać całe SQL swojej
transakcji; token nie upoważnia do przyjęcia obcej transakcji z dowolnymi
savepoints jako zakwalifikowanego commitu.

## Nierozstrzygnięty konflikt kontraktów transakcyjnych

Independent issuer wymaga literalnego kroku
`8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY`.
Wspólny issuer write set jest wykonywany przez obecny coordinator w jednej
transakcji fizycznej.

Jednocześnie `authority_separation` zamrożonego production substrate selection
zawiera literalne
`separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles=true`.
Wymóg odrębnych transakcji nie został ograniczony w tym polu wyłącznie do
CHA/Freshness/checkpoint. Nie można jednostronnie uznać, że pozwala on na
wspólną transakcję podległych issuer persistence schemas. Rozdzielenie
semantycznych ownerów i ACL nie usuwa literalnej niejednoznaczności.

Publiczny produkcyjny commit pozostaje zablokowany do osobno przejrzanej
reconciliation zakresu tego konkretnego pola. S9-RPI-01 zachowuje freeze;
nie obchodzi konfliktu osobnymi BIND/proof/history commitami, `SET ROLE`,
grantami do starych writerów ani ukrytą zmianą kontraktu.

Nie ma wspólnego issuer/CHA/Freshness commit. RootProofIssuer nie wykonuje
AccountGenesis `PREPARED` lub `COMMITTED`, nie zmienia CHA AttemptStore
i nie zastępuje Freshness CAS. Checkpoint jest późniejszą, odrębną transakcją.

## Lock order oraz wynik nieznany

Istniejąca finalna ścieżka przy nowej decyzji zachowuje kolejność:

1. Root/history/trust fence rows `FOR UPDATE` w sortowanym porządku identity.
2. Wspólny requester/claimant alias advisory transaction lock.
3. Requester/claimant current credential rows w sortowanym porządku schema;
   admin `apply(jsonb)` bierze ten sam alias lock przed current row lock.
4. Entitlement lineage/CAS, exact immutable RESERVED/STAGED i history CAS,
   a następnie cały chroniony decision write set.

Backend PostgreSQL utrzymuje blokady do końca finalnego COMMIT/ROLLBACK,
również po śmierci procesu klienta. Administracja nie trzyma transakcji DB
podczas signing lub mutation custody: najpierw oddzielny, potwierdzony
`PENDING_CLOSED`, później operacja custody i dopiero exact completion.
Crash lub nieznany wynik zachowuje zamknięcie. `REVOKED` nie wraca do `ACTIVE`
ani `VERIFY_ONLY`.

Dokładny replay istniejącej decyzji w SQL poprzedza sprawdzanie `ACTIVE`
fences nowej emisji. Odczyt oryginalnego już BOUND wyniku może więc pozostać
możliwy po planned rotation do `VERIFY_ONLY`; nie jest nowym BIND ani nowym
podpisem. Pełny historyczny acceptance nadal wymaga zachowanej evidence
`ACTIVE` w chwili oryginalnej decyzji. Dla `REVOKED` sam podpis tego klucza
nie stanowi niezależnej evidence historycznej.

Utrata połączenia, `NOT_FOUND` i nieznany wynik COMMIT nie uprawniają do
sukcesu, automatycznego write retry, replacement attempt, nowego `rpf_`
ani ponownego podpisania. Recovery używa exact immutable attempt/operation/
entitlement locator oraz oryginalnych zachowanych bajtów.

## Ograniczenia liveness istniejącego przygotowania

V1 signing intent ma `UNIQUE(operation_key)`, `UNIQUE(entitlement_key)`
i `UNIQUE(account_key)`; retention ma odpowiadające constraints w scope.
Nie istnieją więc dwie różne legalne STAGED próby tego samego entitlement
gotowe do równoczesnego finalnego commitu. Race dwóch przygotowań rozstrzyga
się przed BIND, a race dwóch exact commitów tego samego attempt dowodzi
jednej decyzji i replay. Żaden nie dowodzi scenariusza dwóch różnych legalnych
finalnych attemptów dla jednego entitlement.

Zamrożony scenariusz `two_operations_one_entitlement: one winner; loser REJECT`
i durably superseded replacement po authoritative UNBOUND wymagają osobno
przejrzanej wersji preparation/replacement. S9-RPI-01 nie usuwa unique
constraints w celu uzyskania pozytywnego testu. Dwa różne entitlementy ze
wspólnym history predecessor nadal pozwalają kwalifikować history CAS oraz
pełny rollback BIND przegranego.

Legacy V1 `approved_commits` ma jeden exact package per attempt, wraz
z history predecessor i signed head. Tę historyczną blokadę liveness rozwiązuje
już [History Successor Reproposal V1](stage9_history_successor_reproposal_v1.md)
i `PostgreSQLHistorySuccessorReproposalV1`, z nowym wersjonowanym inventory
zachowującym oryginalne proof bytes i attempt. S9-RPI-01 nie implementuje tego
mechanizmu ponownie. Produkcyjną blokadą pozostaje jego granica
`ISOLATED_TEST_ONLY`, a nie brak testowej implementacji reproposal.

## Walidacja

Kwalifikacja lokalna: PostgreSQL **16.15**, odizolowane bazy i role,
Python 3.12.14. Zestaw pięciu nowych modułów testowych: **117 passed**,
bez pominięć, 20,32 s. Łączne pokrycie gałęzi pięciu nowych modułów kodu:
**78%**. Celowane regresje credentials, kontraktów atomic commit,
administracji custody i import isolation: **602 passed**, 73,93 s.
Regresje transportu/pliku, Windows offline oraz importów z #3149:
**34 passed, 1 skipped**, 3,15 s; test zmiany właściciela wymaga `chown` niedostępnego dla tego konta.

| Wymagany scenariusz | Wykonany dowód i granica |
| --- | --- |
| Rotacja przed commitem | Rzeczywisty PostgreSQL odrzuca starą identity/generation po zamknięciu i enrollment zamkniętego następcy; bez produkcyjnego custody rotation |
| Revocation podczas commitu | Admin czeka na backend issuera trzymający slot/fence locks; następny issuer widzi zamknięcie |
| Dwa procesy / entitlement | Dwa procesy rozstrzygają jeden lifecycle slot. Nie jest to dowód dwóch różnych legalnych finalnych attemptów dla jednego entitlement; blokadę kardynalności opisano wyżej |
| Crash przed/po COMMIT | Śmierć rzeczywistego procesu przed/po close COMMIT zachowuje odpowiednio poprzedni stan albo trwały `PENDING_CLOSED` |
| Utrata odpowiedzi | Przerwanie odczytu rzeczywistego socketu po wysłaniu zapisu: wynik nieznany, zero automatycznego retry, pozytywne exact resolution |
| Restart `PENDING_CLOSED` | Restart klastra PostgreSQL i świeży proces; fence nie otwiera się i receipt pozostaje immutable |
| Podmiana signera | SQL odrzuca podmienione authority/identity/material/generation; testy OS boundary nie stanowią kwalifikacji fizycznych signer services |
| TEST_ONLY jako produkcja | Odrzucane profile/typy pins i override konfiguracji; testowy `trust` zwraca rzeczywisty `system_user=NULL` i nie przechodzi zainstalowanej granicy peer |
| Nieautoryzowany principal | Rzeczywiste role, funkcje, owner, memberships, DML/column grants i alternatywne SECURITY DEFINER paths odmawiają kwalifikacji |
| Niekompletne fences | Brakujący, zduplikowany, podmieniony i rozszerzony zestaw exact bindings odmawia |

Historyczne obserwacje `VERIFY_ONLY` pozostają odczytywalne bez prawa do nowej
operacji. Ponowne użycie tej samej transakcji dla close/completion i close/
positive resolution jest negatywnie sprawdzone przez rzeczywisty `xid8`.
Pełny wielogodzinny genuine INITIAL_BINDING/atomic/reproposal suite nie został
tu ukończony: szerszy przebieg przerwano przy setup legacy lineage, po 570
udanych testach. Wszystkie wcześniejsze testy i pełne zakresy CI pozostają
włączone. Nowe wyniki nie kwalifikują produkcyjnego BIND ani fizycznego custody.

`python -m pre_commit run ruff-format --all-files` przeszedł bez dalszych
zmian; osobny format check obejmuje CLI deployment. Ruff, skonfigurowany mypy
dla Linux i Windows (**335 plików**), offline staged Betterleaks 1.9.0
oraz pięć reguł Semgrep 1.179.0 przeszły; Semgrep: zero findings, testy reguł
5/5. `git diff --check` przeszedł. Mypy używał `--no-incremental
--cache-dir=/dev/null`, omijając błąd serializacji cache narzędzia bez
wyłączenia kontroli typów. Poprawki formatowania i Windows POSIX adapters
z #3149 są zachowane z main; końcowy diff S9-RPI-01 nie zmienia istniejącego
transportu enrollment ani jego testu Windows.

Izolowany harness może wstrzyknąć `ACTIVE` uprzywilejowanym połączeniem
wyłącznie w celu wykonania race/retirement tests mechanizmu blokad. Nie jest
to produkcyjne API aktywacji, dowód legalnego enrollmentu lub procedura do
przeniesienia na wdrożenie. Normalny installer tworzy zamknięte slots.

## Pozostałe blokady i rekomendacja

Publiczna emisja wymaga udokumentowanego, zakwalifikowanego wdrożenia
odrębnego issuera i signerów, instalacyjnej konfiguracji poza kontrolą
requestera, rzeczywistych rozłącznych OS/DB principal oraz zamknięcia
wszystkich alternatywnych ścieżek zapisu. Wymaga też pełnego uczestnictwa
root/history rotation, trust activation i checkpoint acceptance w tym samym
DB-authoritative lifecycle protokole oraz rozstrzygnięcia wymienionego
konfliktu transakcyjnego. Brak któregoś dowodu oznacza `BLOCKED`.

Koordynaty `system_identifier` i database OID wykrywają niezgodność lineage,
ale clone/restore `PGDATA` może zachować oba. Nie zastępują kwalifikacji
rzeczywistej fizycznej instancji, tożsamości serwisu i pojedynczego backendu
wykonującego finalny write set. `PRODUCTION_LOCAL` nie chroni przed
skoordynowanym rollback całego hosta, uprzywilejowanym odczytem lokalnych
software keys ani pełnym przejęciem hosta.

**NO-GO dla publicznej emisji Root Proof.** Kod izolacji oraz wyniki
izolowanej kwalifikacji nie zmieniają dostępności semantycznego runtime,
`production_provisioning_ready` ani uprawnień do produkcyjnej operacji.
