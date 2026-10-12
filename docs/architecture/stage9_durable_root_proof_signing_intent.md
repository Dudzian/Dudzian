# Stage 9: trwały zamiar podpisania Root Proof

## Baza i źródła audytu

Audyt rozpoczęto na `main` po PR #3109, przy
`1ee55b2408d354ebfcefe7c8addefcc1a7b10563`. Codebase Memory MCP i CLI
nie były dostępne w środowisku; zakres ustalono
przez wyszukiwanie i bezpośrednią inspekcję źródeł. Wnioski bezpieczeństwa
opierają się na kodzie, nie na nieobecnym indeksie.

Przejrzane rzeczywiste granice:

- [Retention](../../bot_core/postgresql_root_proof_retention.py) i
  [jego schema](../../bot_core/postgresql_root_proof_retention_schema.py)
  przechowują podpisane bajty `STAGED`, bez rezerwacji przed podpisem.
- [Signed CHA attempt](../../bot_core/licensing/cha_root_proof_signed_attempt.py)
  udostępnia genuine źródło, a
  [issuer runtime](../../bot_core/root_proof_issuer_runtime.py) ponownie
  weryfikuje żądanie względem żywego authority.
- [PostgreSQL issuance authority](../../bot_core/postgresql_root_proof_issuance_authority.py)
  agreguje autorytety entitlement, requester i claimant; nie jest wspólnym
  właścicielem zapisu proof i history.
- [Entitlement registry](../../bot_core/postgresql_entitlement_registry.py)
  wykonuje BIND w swojej transakcji `SERIALIZABLE`.
- [Authenticated issuer history](../../bot_core/postgresql_authenticated_issuer_history.py)
  i [checkpoint authority](../../bot_core/postgresql_checkpoint_authority.py)
  wykonują własne transakcje, odrębne od BIND.
- [Publiczna orkiestracja](../../bot_core/root_proof_issuer_orchestration.py)
  pozostaje odczytowa i blokuje mutacje issuera.

Zamrożony
[kontrakt niezależnego Root Proof Issuer](cryptohunter_product_architecture/m05_account_genesis_independent_root_proof_issuer_contract.md)
nadal określa tożsamość, domenę podpisu, exact retry i wymóg atomowego BIND +
signed proof + history. Jego źródłem prawdy pozostaje istniejący JSON;
ten dokument nie zmienia zamrożonego kontraktu ani historycznych statusów.

## Zakres i granica zaufania

`postgresql_root_proof_signing_intent.py` i
`postgresql_root_proof_signing_intent_schema.py` wprowadzają osobny magazyn v1
dla zamiaru podpisania Root Proof. Stan `RESERVED` utrwala jeden `root_proof_id`
oraz dokładne, niepodpisane bajty payloadu przed możliwością podpisania. Wynik
ma `authorizes_operations=False`; `require_issued()` zawsze odmawia. Rezerwacja
nie stanowi podpisanego proof, zatwierdzonego wydania ani uprawnienia do Account
Genesis.

Schemat ma odrębnego właściciela `NOLOGIN` i odrębny principal
`issuer_writer` z `LOGIN`. Właściciel funkcji `SECURITY DEFINER` nie jest rolą
runtime. Writer otrzymuje wyłącznie przejrzane wejścia SQL, bez bezpośredniego
DML, a niezaufany runtime nie otrzymuje dostępu do schematu ani jego funkcji.
Obie role wymagają kwalifikacji rzeczywistych uprawnień i braku niebezpiecznych
członkostw; nie wystarcza nazwa roli lub deklaracja w konfiguracji.

Kod weryfikujący żądanie działa po stronie zaufanego issuera. Kontrola
kanonicznego kształtu i spójności przez SQL nie dowodzi zweryfikowania
`INITIAL_BINDING`, podpisanego attempt ani tożsamości custody. Posiadacz danych
logowania writera może wywołać funkcję SQL bez przejścia przez Python; dlatego
izolacja writera od niezaufanego runtime jest warunkiem wdrożenia. Ten schemat
nie naprawia uprawnień istniejącego magazynu `STAGED` opisanego w
[dokumentacji retention](stage9_durable_root_proof_retention.md).

## Przyjęcie żądania przez wewnętrzny port issuera

Wewnętrzne `reserve(binding, signed_attempt)` wymaga rzeczywistego capability
`INITIAL_BINDING` i podpisanego, niezmiennego CHA attempt. Caller nie dostarcza
osobnego payloadu, `root_proof_id`, providera ani DTO, które mogłoby zastąpić te
capability. Konstruktor przypina dokładny `RootProofRetentionScope`,
`LocalRootProofSigningProvider`, konfigurację magazynu i autorytet PostgreSQL.
Zakres obejmuje `PRODUCTION_LOCAL`, `PRODUCTION`, trust domain, product oraz
issuer.

Backend ponownie odczytuje źródło przez `_read_issuer_preflight_source()` oraz
sprawdza żądanie przez `_verify_request()` wobec przypiętego, żywego autorytetu
PostgreSQL. Weryfikacja obejmuje pochodzenie `INITIAL_BINDING`, authorization,
request fingerprint, trwały attempt, oryginalne kanoniczne bajty requestu,
requester i claimant wraz z ich rolami, kluczami i wersjami. Tożsamości i zakresy
muszą się zgadzać z authority; samo poprawne podpisanie dowolnego requestu nie
wystarcza.

Istniejący preflight zachowuje także kontrolę provenance requester/claimant:
service, credential i lifecycle namespace oraz key handle muszą odpowiadać
zainstalowanemu custody. Ten wewnętrzny port nadal wymaga dostępu do lokalnych
źródeł custody CHA. Publiczny port provenance dla w pełni odizolowanego issuera
wymaga osobnego przeglądu; sam podpis zweryfikowany przez publiczny klucz nie
zastępuje tych kontroli.

Odczyt oraz exact retry również wymagają żywego, zgodnego stanu `UNBOUND` i
aktywnych credentials requester/claimant, a zachowany signer snapshot musi
odpowiadać bieżącemu `ACTIVE` snapshotowi przypiętego providera. Zmiana stanu
lub credentials zamyka tę ścieżkę. Port nie ustanawia historycznej eligibility
rezerwacji po BIND, rotacji lub revocation.

Nowa rezerwacja wymaga dokładnego snapshotu przypiętego signera w stanie
`ACTIVE`. Zachowany snapshot wiąże provider namespace, semantic role,
credential identity, key version, lifecycle generation i publiczny materiał
klucza. Żądanie podpisania odnosi się do zamrożonej domeny:

```text
ASCII("CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1") || 0x00 || UTF-8 JCS(signed_payload)
```

Payload powstaje z autorytatywnych danych i zachowuje zamknięty zestaw pól Root
Proof, w tym action, account, operation, entitlement/generation, request
fingerprint, issuance attempt, issuer/key/version oraz zarezerwowany Root Proof
ID. Magazyn utrwala oryginalne kanoniczne bajty `BYTEA`; digest potwierdza ich
integralność i powiązanie, ale nie zastępuje bajtów. Rezerwacja nie wykonuje
podpisu i nie przechowuje udawanego podpisanego envelope.

## Rezerwacja, unikalność i exact retry

Issuer tworzy `rpf_` z UUIDv7 dopiero wewnątrz transakcji PostgreSQL, po
kwalifikacji schematu, uzyskaniu transakcyjnej blokady advisory dla attempt,
pełnej ponownej weryfikacji źródeł i autorytatywnym odczycie potwierdzającym
brak rezerwacji. Odczyt dla exact retry poprzedza tworzenie identyfikatora.
Blokada i constraints należą do PostgreSQL; działają również dla niezależnych
procesów i po restarcie klienta.

Do wyboru blokady backend odczytuje z rzeczywistego signed capability wyłącznie
zachowany selector attempt. Jest to niezweryfikowany locator do serializacji,
bez uprawnienia do odczytu proof, alokacji ID ani zapisu. Dopiero pod blokadą
wykonuje pełną weryfikację żywych źródeł i wymaga zgodności zweryfikowanego
attempt ID z selectorem. Odczyt i rezerwacja używają tej samej blokady. Dzięki
temu konkurencyjne ponowienia tego samego attempt nie uruchamiają równolegle
istniejących, nieblokujących kontroli liveness LPPI/custody. Nie pomija się
żadnej weryfikacji podpisu, provenance lub lifecycle.

Attempt, Root Proof ID, account, logical operation oraz para
entitlement/generation mają globalne ograniczenia unikalności w magazynie.
Konflikt na dowolnej z tych granic zamyka żądanie;
nie powstaje kolejny identyfikator jako mechanizm obejścia konfliktu.

Exact retry odczytuje istniejący rekord i wymaga zgodności całej niezmiennej
tożsamości: scope, `INITIAL_BINDING` reference/digest, authorization, attempt,
request bytes/reference/digest, obu podpisujących tożsamości i wersji,
snapshotu signera, domeny podpisu, payloadu oraz całego kanonicznego envelope
rezerwacji. Nie ogranicza porównania do digestu ani `root_proof_id`. Zwraca te
same oryginalne bajty oraz ten sam identyfikator; nie losuje ponownie, nie
podpisuje i nie poprawia rekordu. Niespójny stan, podmieniona tożsamość lub
niekwalifikowany autorytet oznaczają odmowę.

SQL odczytuje attempt globalnie. Dla znalezionego rekordu porównuje klucze
fencingu, scope oraz stan z niepodpisanym kanonicznym envelope i payloadem.
Wykryta niespójność powoduje `XX001` i odmowę `CORRUPT` backendu. Uprzywilejowana
zmiana samego klucza wyszukiwania może ukryć rekord; brak dopasowania nie
dowodzi braku wcześniejszej rezerwacji ani emisji. Kontrola znalezionego
rekordu ustala jego spójność, nie dowodzi kryptograficznego pochodzenia żądania.

Przed zwróceniem zachowanego intent exact retry wykonuje końcową weryfikację
żywego binding i signed attempt. Odczyt historycznych bajtów nie pozwala
pominąć zmiany autorytetu podczas operacji.

## Nieznany wynik zapisu i granice recovery

Utrata odpowiedzi w trakcie zapisu lub po commicie ma wynik `INDETERMINATE`.
Klient nie zamienia jej w brak rekordu ani automatycznie ponawia podpisu.
Wymagany jest jawny, nowy, kwalifikowany odczyt autorytatywnego PostgreSQL,
powiązany z tym samym oryginalnym binding i signed attempt. Niedostępna lub
niekwalifikowana baza nie ustanawia autorytatywnego braku.

Port rozróżnia `CONFLICT`, `CORRUPT`, `UNAVAILABLE` i `INDETERMINATE`.
Jawne, rozpoznane przyczyny walidacyjne `_verify_request()`, które dowodzą
niezgodności tożsamości, podpisanego requestu, stanu entitlement lub aktywnych
signerów, oznaczają `CONFLICT`. Tak samo klasyfikowana jest jawna niezgodność
scope w polach environment, trust domain lub product. Uszkodzone dowody
wykryte przez `_exact_inputs()` poza tą kontrolą zakresu pozostają `CORRUPT`.
Pozostały
`RootProofIssuerRuntimeError` nie dowodzi przyczyny odmowy i zachowuje wynik
`INDETERMINATE`; backend nie zgaduje, czy błąd oznacza niezgodne dane, czy
niedostępny autorytet.
Żaden wynik odmowy nie nadaje prawa do podpisu lub emisji.

Odczyt istniejącego `RESERVED` odzyskuje identyfikator i dokładny materiał do
przyszłego podpisania. Nie odzyskuje podpisanego Root Proof i nie zezwala na
zmianę attempt, utworzenie zastępczego ID ani rozpoczęcie drugiej emisji. Sam
autorytatywny brak rezerwacji także nie dowodzi `AUTHORITATIVELY_UNBOUND` całego
issuera: stan BIND, history, retention i lifecycle wymagają własnych
autorytetów.

Trwała rezerwacja identyfikatora nie gwarantuje dokładnie jednego fizycznego
wywołania podpisywania. Awaria pomiędzy wykonaniem podpisu a trwałym
zachowaniem sygnatury może pozostawić nieznany wynik fizycznego wywołania
signera. Nie wolno na tej podstawie automatycznie ponawiać podpisu. Ta
implementacja nie wywołuje signera; rezerwacja i exact retry zachowują ID oraz
bajty, ale nie dowodzą liczby fizycznych wywołań przyszłego podpisywania.

## Kwalifikacja i dostępność produkcyjna

Instalator i runtime wymagają rzeczywistego PostgreSQL 16, włączonych `fsync`,
`synchronous_commit` i `full_page_writes`, dokładnych ról i OID, ACL,
metadata v1, definicji funkcji, tabel, constraints i ograniczeń niezmienności.
Osobny testowy klaster z `fsync=off` sprawdza odmowę bez osłabienia trwałości
głównego klastra. Raporty błędów nie ujawniają DSN ani sekretów.

Przywrócenie wcześniejszego, poprawnego stanu całej bazy przez uprzywilejowanego
administratora nie jest wykrywane przez ten magazyn. Rezerwacja nie zapewnia
niezależnego checkpointu, granicy revocation ani globalnego, atomowego snapshotu
lifecycle. `ACTIVE` jest warunkiem nowej rezerwacji, ale nie stanowi
linearizacji uprawnienia w późniejszym momencie podpisu lub emisji.

Brakuje kwalifikowanej kompozycji wdrożenia izolującej `issuer_writer` oraz
pełnego właściciela transakcji BIND + dokładny signed proof + authenticated
history. Dlatego `ROOT_PROOF_ISSUANCE_AVAILABLE=False`. Wniosek
`CODE_READY_FOR_REVIEW` opisuje możliwość przeglądu i wyniki bramek dla tego
komponentu; nie jest deklaracją produkcyjnej dostępności issuera.

Publiczna orkiestracja pozostaje bez zmian i nadal blokuje emisję. Ten
komponent nie wykonuje `sign`, zapisu `STAGED`, BIND, append history,
checkpoint, finalize, transportu ani Account Genesis COMMITTED. Nie zmienia
zamrożonych kontraktów, `deployment/stage9_current_status.json` ani bramek
Stage 10.

Następny krok wymaga przejrzanej kompozycji wdrożenia i atomowego BIND +
identyczny podpisany proof + history w jednym autorytecie transakcyjnym,
z linearizacją lifecycle signera i requester/claimant. Oddzielne commity,
advisory lock dla samego intent lub kompensacje nie spełniają tego wymagania.

## Bramki jakości

Oba moduły obejmuje istniejący Ruff i skonfigurowany mypy dla Linux oraz
Windows. Jawny krok formatowania obejmuje moduły oraz testy
`tests/security/test_postgresql_root_proof_signing_intent.py` i
`tests/security/test_postgresql_root_proof_signing_intent_schema.py`. Test schematu
działa seryjnie z istniejącymi testami historii i retention na PostgreSQL 16,
dopisując pokrycie gałęzi `_schema.py` do `coverage.xml` w `quality-ratchet`.
Genuine runtime działa w istniejącym seryjnym `semantic-issuer-preflight`,
którego `semantic-preflight-coverage.xml` obejmuje nowy moduł runtime i zachowany
zakres `cha_root_proof_signed_attempt.py`, w tym wewnętrzny selector blokady.
Obie istniejące bramki changed-code pozostają 70%; wszystkie wcześniejsze testy,
scope pokrycia, timeouty 120/180 minut i manualne bramki sprzętowe są zachowane.
W tej sesji pełne oba jobs nie zostały uruchomione. Podział nowych testów opiera się
na wcześniejszych udanych ratchet 80–108 minut i oszacowaniu istniejącej semantic
lineage 85–100 minut z komentarza workflow; dodany runtime jest szacowany na
60–70 minut, co daje około 145–170 minut w budżecie 180 minut. Lokalnie weryfikowane
są odpowiednie zakresy PostgreSQL oraz dwa raporty coverage odpowiadające jobs.
