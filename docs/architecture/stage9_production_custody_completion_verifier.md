# Stage 9 — S9-RPI-03: production custody completion verifier

## Wynik i granica implementacji

S9-RPI-03 usuwa produkcyjną ścieżkę uznającą oświadczenie administratora
PostgreSQL za wystarczający dowód zakończenia zmiany custody. Przygotowuje
odczytową weryfikację dokładnego, trwałego journal lokalnego custody oraz
zamknięty katalog production lifecycle V2. Nie provisionuje niezależnego
autorytetu completion i nie publikuje produkcyjnego completion.

```text
semantic_root_proof_issuer_runtime = NOT_IMPLEMENTED / BLOCKED
production_provisioning_ready = false
ROOT_PROOF_ISSUANCE_AVAILABLE = false
ATOMIC_ISSUER_READY = NOT_READY
PRODUCTION_LIFECYCLE_ISOLATION = BLOCKED
production_custody_completion_publication = BLOCKED
public_root_proof_issuance = NO-GO
```

Odczytowe sprawdzenie journal, kryptograficzne uwierzytelnienie completion
i kwalifikacja fizycznej instalacji są odrębnymi wynikami. Wykonywalny
verifier lokalnego journal nie ustanawia dwóch pozostałych gwarancji.
Produkcja nie otrzymuje `SIGN`, `commit()`, generowania lub aktywacji kluczy,
BIND, enrollmentu, przejścia do `ACTIVE` ani Stage 10. Zamrożone kontrakty
M0.5 i wcześniejsze statusy pozostają bez zmian.

## Audyt i ponowne użycie istniejących mechanizmów

Podstawą są scalone S9-RPI-01 i S9-RPI-02, opisane w
[lifecycle isolation](stage9_production_root_proof_lifecycle_isolation.md)
oraz [independent signers](stage9_independent_production_signers.md),
i bezpośrednio sprawdzone źródła:

- `production_root_lifecycle.py`;
- `postgresql_production_root_lifecycle_schema.py` oraz
  `postgresql_production_root_lifecycle_qualification.py`;
- `production_signer_custody.py` i `production_signer_ipc.py`;
- `local_signing_lifecycle_evidence.py` i `local_signing_custody.py`;
- `postgresql_custody_lifecycle.py`.

Codebase Memory MCP nie był callable w tej sesji. Celowane wyszukiwania
i odczyty rzeczywistych źródeł nie są deklaracją kompletności nieobecnego
indeksu ani kwalifikacją wdrożenia.

| Istniejący mechanizm | Bezpieczne ponowne użycie | Ograniczenie |
| --- | --- | --- |
| Production lifecycle | Immutable `begin_close`, exact predecessor, slot/fence locking, osobny potwierdzony COMMIT `PENDING_CLOSED`, `close_xid`, direct-login/OID i peer authentication | Receipt zgodny strukturalnie z publicznymi faktami nie dowodzi mutation custody |
| Kwalifikacja katalogu | Dokładne źródła funkcji, ACL, ownership, `SECURITY DEFINER`, stały `search_path`, catalog/physical pins i globalne ścieżki uprawnień | Kontrola drift nie tworzy niezależnego autorytetu ani ochrony przed zaufanym bootstrap superuserem |
| Durable journal | Kanoniczne oryginalne bytes, exact operation descriptor, predecessor/after metadata, trzy dozwolone przejścia i blokada custody | Journal i lokalne członkostwo obiektu nie są kryptograficznym dowodem niezależnym od hosta lub administratora storage |
| Testowy custody coordinator | Regresje DB-first close, trwałości, replay i crash consistency | Pozostaje `ISOLATED_TEST_ONLY`; testowy MAC, enrollment i capabilities nie upoważniają produkcji |
| Signer IPC | Minimalne `DIAGNOSTIC`/`QUALIFY`, przypięte ścieżki, live peer identity i role separation | Diagnostyczny DTO, metadata hash i seed/public-key match nie dowodzą niezależnego private control |

Historyczny opis `complete_close` w S9-RPI-01 dotyczy V1. Opisane tam
zaufanie do receipt administratora zostaje zastąpione poniższą odmową V2.
Nie zmienia to znaczenia osobnego testowego custody coordinatora.

Zamrożone źródła prawdy sprawdzono bez zmiany ich zawartości:

- [Independent issuer contract](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.json):
  `bind_authorization_order`, `credential_rotation_recovery` oraz
  `lost_response_reconciliation`; historyczny odczyt nie jest nowym BIND,
  a brak odpowiedzi nie zezwala na replacement ani ponowne podpisanie.
- [Production substrate selection](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_production_substrate_selection_contract.json):
  odrębne authority, role, credential, namespace i lifecycle, izolacja profili,
  jawne migracje, `NOT_FOUND=OUTCOME_UNKNOWN` i model lokalnego software
  custody bez odporności na wspólnego host administratora.
- [Implementation readiness](cryptohunter_product_architecture/m05_account_genesis_root_proof_issuer_implementation_readiness_contract.json)
  oraz [physical persistence](cryptohunter_product_architecture/m05_account_genesis_physical_persistence_crash_atomicity_contract.json):
  produkcyjna dostępność wymaga rzeczywistej kwalifikacji; issuer, CHA
  i FreshnessAuthority zachowują własne granice odpowiedzialności.

## Zamknięta granica SQL V2

`complete_close(text,text,bytea)` pozostaje rozpoznawalnym, odmawiającym
tombstone. Jego wykonanie nie zapisuje completion i nie zmienia fence.
Administrator i issuer nie otrzymują `EXECUTE`; odebrane są także domyślne
uprawnienia PUBLIC. Nawet uprzywilejowane bezpośrednie wywołanie tej dokładnej
funkcji odmawia. Kwalifikator wymaga dokładnej odmowy oraz ACL V2; dodanie
grantu, zmiana funkcji lub inny catalog drift unieważniają qualification.

`publish_administrator_completion()` w Pythonie zawsze odmawia z
`INDEPENDENT_CUSTODY_COMPLETION_AUTHORITY_REQUIRED`. Nie przyjmuje poprawnego
hasha, strukturalnie poprawnego receipt ani obiektu lokalnego verifiera jako
capability publikacji. Dzięki odmowie także w SQL obejście Pythona nie
przywraca starej ścieżki. Nie ma zastępczego entrypointu publikacji ani
rozszerzenia diagnostycznego IPC o mutation lub completion DTO.

Nowy `begin_close` nadal zamyka fence w osobnym commicie i nie uruchamia
mutation custody. Odczyt `read_operation` zachowuje slot lock oraz
`READ COMMITTED` do rozstrzygania wcześniejszego backendu. Sam brak rekordu
pozostaje wynikiem nieznanym. Brak niezależnej publikacji oznacza, że operacja
po poprawnym close pozostaje `PENDING_CLOSED`; odczyt custody jej nie otwiera.

Przechowane completion z V1 zachowują oryginalne `receipt_bytes` i digest.
Są klasyfikowane `LEGACY_UNVERIFIED_COMPLETION`, bez awansu do niezależnie
uwierzytelnionego completion. Wcześniejszy zapis nie jest dowodem udanej
operacji custody. Istniejący port administracyjnego enrollmentu zamkniętego
successora nadal może korzystać z historycznego stanu `VERIFY_ONLY`/`REVOKED`;
tworzy wyłącznie `PENDING_CLOSED` i nie weryfikuje wstecznie custody ani nie
aktywuje klucza. Klasyfikacja legacy nie naprawia historycznej decyzji i nie
przepisuje evidence.

Model rozróżnia lifecycle administratora PostgreSQL, który nie ma ownership,
DML/DDL ani członkostwa ownera, od bootstrap superusera. Pierwszy nie może
obejść odmowy przez zachowane API. Superuser może zmienić funkcje, katalog,
role i dane PostgreSQL; sama SQL authority nie zapewnia mu kryptograficznej
niepodrabialności. External catalog pin pozwala odrzucić niezgodne środowisko,
ale nie jest niezależnym podpisem completion ani ochroną przed wspólnym
administratorem hosta i weryfikatora. Nie deklarujemy takiej gwarancji.

## Jawna migracja i installation pin

Katalog V2 wymaga odrębnej kontrolowanej migracji z dokładnie przypiętego V1.
Wejściem jest wcześniejszy zewnętrzny installation/catalog pin i zweryfikowana
physical database identity. Przed transakcją migracji osobna zatwierdzona
operacja musi trwale ustawić `ALLOW_CONNECTIONS=false` dla tej bazy i opróżnić
pozostałe sesje; połączenie bootstrap migratora musi już istnieć. Migracja
odmawia przy otwartej bramie połączeń, zmianie bramy w tej samej transakcji
lub innej sesji. Zapobiega to kontynuowaniu wcześniej rozpoczętego ciała
funkcji V1 po wymianie katalogu. Helper nie odcina ani nie przywraca połączeń
i nie kończy cudzych sesji. Migracja jest operacją bootstrap administratora;
runtime, requester, issuer i signery nie dobierają nowego pina i nie
wykonują automatycznego upgrade lub re-pin.

Zmiana katalogu usuwa stare granty completion, instaluje odmowę V2 i zachowuje
oryginalne operation/receipt bytes oraz zamknięcia. Wcześniejsze zakończenia
otrzymują jawną klasyfikację unverified. Drift V1, niezgodne źródła, ACL,
ownership, katalog lub physical identity wymagają odmowy migracji, zamiast
przyjęcia bieżącej zawartości jako nowej prawdy.

Przy porównaniu z zewnętrznym pinem V1 dopuszczona jest wyłącznie powyższa
zmiana `datallowconn` bieżącej bazy z `true` na `false`. Pozostałe fakty
katalogowe muszą zgadzać się dokładnie. Po COMMIT migracji osobna operacja
bootstrap przywraca bramę połączeń; usługi pozostają zatrzymane. Dopiero
wtedy osobny bootstrap etap wylicza i przegląda nowy katalog; zewnętrzna
chroniona instalacja musi jawnie otrzymać zatwierdzony pin V2. Do tej chwili
stary pin odmawia qualification. Samo wykonanie helpera migracji nie jest
autoryzacją rzeczywistej instalacji ani pełnym wdrożeniem procedury migracji
określonej w zamrożonym kontrakcie, w tym jego authenticated completion
record i checkpoint po aktywacji. Ta zmiana nie aktywuje produkcji.

## Odczytowa weryfikacja custody evidence

Nowy odczytowy verifier używa istniejącego formatu journal V2 i metadanych.
Jest diagnostycznym portem Linux; nie adaptuje requestów produkcyjnego SQL
do izolowanego koordynatora V2 i nie poświadcza wcześniejszego `fsync`.
Wymaga dokładnych kanonicznych bytes immutable operation oraz journal,
wiąże operation ID, profile/trust domain, semantic role, credential identity,
namespace, public key/material, key version, predecessor generation i target.
Legalne są jedynie `ACTIVE → VERIFY_ONLY`, `ACTIVE → REVOKED` oraz
`VERIFY_ONLY → REVOKED`, ze wzrostem generation o jeden.

Pod blokadą custody odczytuje zapis `COMPLETE` i rzeczywiste aktualne metadata;
aktualny rekord musi być dokładnym `after_metadata` tego journal. Późniejszy
legalny descendant, wymieniona generation, obca rola lub podobne publiczne
fakty nie wystarczają. Sprawdzenia plików i ponowne odczyty wiążą chronione
ścieżki, identity plików i katalogów oraz oryginalne bytes; symlink, path swap,
uszkodzenie i zmiana podczas obserwacji odmawiają.

`INTENT` zawsze odmawia pozytywnego wyniku, również gdy current metadata
wyglądają już jak stan końcowy. Verifier nie wykonuje przejścia, nie odtwarza
mutation, nie kończy journal, nie tworzy brakującego magazynu i nie uruchamia
recovery. Oryginalne bytes pozostają evidence; digest jest ich identyfikatorem,
nie substytutem. Wynik lokalnego odczytu nie staje się produkcyjnym SQL
completion, tokenem close ani świadectwem niezależnej authority.

Nie dodano portu requestera, cross-role custody ani odczytu sekretów przez
IPC. Istniejące zainstalowane role, prywatne storage i `DIAGNOSTIC`/`QUALIFY`
zachowują ograniczenia S9-RPI-02. Lokalne testy verifiera sprawdzają rzeczywisty
journal i metadata. Używają jawnego testowego authorization seam i mockowanego
keyring; nie kwalifikują niezależnej authority, native keyring ani separacji dwóch
produkcyjnych procesów na fizycznej instalacji.

## Crash, replay i recovery

| Zdarzenie | Wynik i dozwolony odczyt |
| --- | --- |
| Crash przed potwierdzeniem COMMIT `begin_close` | `INDETERMINATE`; read-only resolution musi rozstrzygnąć exact operation po zakończeniu wcześniejszego backendu; brak odpowiedzi nie dowodzi rollbacku |
| Potwierdzony `PENDING_CLOSED`, brak mutation custody | Fence pozostaje zamknięty, publication `BLOCKED`; odczyt nie uruchamia mutation |
| Crash podczas zapisu journal | Brak, uszkodzony lub `INTENT` journal nie jest completion; `INDETERMINATE`/`BLOCKED`, bez automatycznego retry mutation |
| Trwałe `COMPLETE`, brak DB completion | Można odczytowo zweryfikować exact journal/current metadata; brak niezależnej authority nadal blokuje publikację i fence |
| Utrata odpowiedzi po historycznym DB completion COMMIT V1 | Exact odczyt zachowuje oryginalne receipt i klasyfikację legacy unverified; nie tworzy nowego completion |
| Restart PostgreSQL lub verifiera | Ponowna qualification i odczyt trwałych exact bytes; procesowy wynik/cache nie zastępuje trwałej evidence; fence nie jest otwierany |
| Równoczesna operacja tego samego credential | Slot/fence i custody locking serializują właściwe odczyty; inny operation ID/descriptor lub current generation odmawia |
| Exact replay completion | Lokalny odczyt ponownie sprawdza te same bytes i aktualne metadata; produkcyjna publication nadal odmawia |
| Ten sam operation ID, inna treść | Immutable operation conflict; brak last-write-wins i zastępowania oryginalnych bytes |
| Timeout, `NOT_FOUND`, utrata połączenia | Wynik nieznany; bez wnioskowania rollbacku, nowej mutation lub otwarcia fence |

S9-RPI-03 nie implementuje recovery produkcyjnego completion. Pozytywne
testy istniejącego testowego protokołu completion pozostają osobnym wynikiem
izolowanego środowiska; nie dowodzą nowej produkcyjnej ścieżki publikacji.

## Wymagane dalsze prace i walidacja

Przed umożliwieniem produkcyjnego completion wymagany jest osobno
provisionowany, niezależnie uwierzytelniony autorytet lub osobno zatwierdzona
zmiana kontraktu określająca taką topologię. Musi mieć własny chroniony
credential i trust anchor niedostępny administratorowi lifecycle PostgreSQL,
ściśle weryfikować trwałe custody oraz committed close i uwierzytelniać
oryginalne bytes exact operation/evidence/result. Weryfikacja musi obowiązywać
na granicy publication odpornej na bezpośrednie wywołanie SQL. Hash dostarczony
przez administratora, dostępny mu MAC i niezweryfikowany signer DTO tego
warunku nie spełniają. Brak takiej authority jest jawną blokadą tej zmiany.

Pozostają także blokady S9-RPI-01/02: rzeczywiste private-control attestation,
cross-role native-keyring isolation, kwalifikacja namespaces i wszystkich
ścieżek logowania PostgreSQL, aktywacja/sukcesja i trust activation oraz
produkcyjny semantic issuer. Nierozstrzygnięty konflikt literalnego
`separate_semantic_owners_schemas_roles_credentials_namespaces_transactions_and_lifecycles=true`
z krokiem `8_ATOMICALLY_BIND_AND_PERSIST_IDENTICAL_SIGNED_PROOF_AND_HISTORY`
wymaga osobnej zatwierdzonej reconciliation; ten PR nie zmienia freeze.

Celowane testy obejmują odmowę fałszywych receipt i hash-only evidence,
bezpośredni SQL, nieautoryzowane principals, ACL/catalog drift, exact
migrację i legacy bytes oraz lokalny journal, generation, role, material,
replay i wyścigi. Regresje S9-RPI-01/02 i istniejące isolated custody
qualification zachowują dotychczasowe znaczenie. Dokładne wykonane polecenia,
wyniki i niewykonane sprawdzenia z powodami należą do opisu PR; niewykonanych
testów nie uznajemy za PASS.

Obowiązują istniejące Ruff, osobny `ruff-format --all-files`, skonfigurowany
mypy Linux/Windows i właściwe skany bezpieczeństwa. Nie zmieniono workflow,
timeoutów, coverage ani wcześniejszych bramek. Testy historycznych stanów
używają jawnego setupu przez superusera wyłącznie w jednorazowej bazie:
nie nazywają tego setupu dowodem completion. PR otrzymuje
`FULL_CI_PENDING`, gdy pełne CI nie zakończyło się; nie czekamy na wielogodzinny
atomic suite ani nie wykonujemy automatycznego merge. Root Proof pozostaje
**NO-GO**.
