# Stage 9 — niezależny audyt reviewed source closure

Stan bazowy: `main` `a05c2a33f7e6c5078ba7112a6beaf3ac4b5170bf`,
2026-11-04. Materiał do przeglądu, **bez zatwierdzenia nowych bajtów**.
Pełna kwalifikacja pozostaje **BLOCKED**. Analiza nie jest dowodem działania
na fizycznym Linux, próbą VM ani awansem Stage 9.

## Metoda i granice dowodu

Odczytano `AGENTS.md`, moduły kwalifikatora, AccountGenesis, verifiera i jego
zależności, instalator v2, bootstrap runtime, eksporter oraz odpowiednie testy
odmów. Codebase Memory nie było dostępne w tej sesji; zależności sprawdzono
w rzeczywistych importach i wywołaniach źródeł. Historia lokalnego Git nie jest
płytka. Dla każdego wpisu z bazowego `REVIEWED_CODE_DIGESTS` odczytano surowe
bajty przez `git show REV:ścieżka`, policzono SHA-256 i przejrzano wszystkie
rewizje zmieniające tę ścieżkę osiągalne z bazowego HEAD. SHA-256 treści jest
odrębny od identyfikatora Git blob (SHA-1 obiektu Git).

[Maszynowy materiał audytu](stage9_reviewed_source_audit.json) zawiera pełne
hashe, rozmiary, identyfikatory blobów, wersje historyczne i kompletny unified
diff AccountGenesis. Hash bazowych plików roboczych porównano z treścią Git;
wszystkie sześć odpowiada swoim blobom bazowego HEAD. Nie jest to dowód
pochodzenia dowolnej kopii o tej samej nazwie ani autoryzacja nowego wydania.

## Inwentarz sześciu pinów

Każda ścieżka w tabeli jest względna wobec `bot_core/`. W kolumnie rewizji
podano commit wprowadzający dokładnie pasujące historyczne bajty; obecność
commitu w historii sama w sobie nie jest zatwierdzeniem. Powiązanie z reviewed
wynika z literalnego digestu zamrożonego w kwalifikatorze.

| Plik | Oczekiwany SHA-256 | SHA-256 bazowego Git blob | Rewizja odpowiadająca pinowi | Różnica bajtów względem pinu |
| --- | --- | --- | --- | --- |
| `linux_service_ipc.py` | `099e6a3a08717b48e7a01606b9908f56f0eec40aced260d554d7dfc4562b451e` | `099e6a3a08717b48e7a01606b9908f56f0eec40aced260d554d7dfc4562b451e` | `f660d236a476a1f2b5b067c1c48a8f0ef2ac0cbc` | brak, 14746 B |
| `linux_service_postgresql.py` | `d09d359535e18408d8cee282cb757c4eca8a4e8c422bfce8fa804860ad59f0d1` | `d09d359535e18408d8cee282cb757c4eca8a4e8c422bfce8fa804860ad59f0d1` | `f660d236a476a1f2b5b067c1c48a8f0ef2ac0cbc` | brak, 10829 B |
| `production_signer_ipc.py` | `be755e67cb1ea4466c074e4c2c0dbe8e6528552f4a70fd7769a5e876c878f552` | `be755e67cb1ea4466c074e4c2c0dbe8e6528552f4a70fd7769a5e876c878f552` | `24f7c861117791d3fba0d3817d8af1e87326754b` | brak, 21091 B |
| `freshness_verifier_service.py` | `df23a3c2fcd1e04877192fe2fe376909a29aa5183a516485fbfdad5a72f9f057` | `df23a3c2fcd1e04877192fe2fe376909a29aa5183a516485fbfdad5a72f9f057` | `f660d236a476a1f2b5b067c1c48a8f0ef2ac0cbc` | brak, 3636 B |
| `freshness_semantic_verifier.py` | `64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1` | `64dd105d54a2eabd9be9164846fe642ca0a274b30ca5d0a83735a4436d9f39e1` | `f660d236a476a1f2b5b067c1c48a8f0ef2ac0cbc` | brak, 33906 B |
| `account_genesis_freshness_authority.py` | `cd508809f4b4f0663610f592b77695784d74eb4c7e2fba50c42d7a4dbf252f7b` | `81cd0df375fe1a48ba943c7645376b4d97350288c0ebe6523207ff90ed22572b` | `3157385cdffc4b06ab27ad141048db3ebb9bb8c4` | 14146 → 14324 B, formatowanie i guard AF_UNIX |

Pięć aktualnych pinów wpisano w
`f660d236a476a1f2b5b067c1c48a8f0ef2ac0cbc` (2026-11-01).
Bajty `production_signer_ipc.py` pochodzą z wcześniejszej rewizji
`24f7c861117791d3fba0d3817d8af1e87326754b`, ale aktualny pin do pełnego
kwalifikatora dodano dopiero w `f660d236…`.

## Odtworzenie driftu AccountGenesis

1. `3157385cdffc4b06ab27ad141048db3ebb9bb8c4`, 2026-09-22:
   źródło ma 14146 bajtów, Git blob
   `0504f11f3388645ff96f404bfef33d6427530265` i SHA-256 `cd508809…`.
2. `45c705e13462a0d2260be886fcc75e54af5680ab`, 2026-09-22:
   kwalifikator zamraża ten właśnie SHA-256. To punkt odniesienia reviewed.
3. `2f746e23ea2e0e0818c24ca4e5bc75317eff9217`, 2026-09-23:
   formatowanie zmienia źródło do 14198 bajtów i SHA-256
   `80f7e42be9ee64e3d21db38e10415e639109f8d3af3dbf5ac446ea4a9d1f5c7a`.
   `ast.dump(ast.parse(old)) == ast.dump(ast.parse(new))` jest prawdą.
   Pin pozostaje poprzedni, więc już ta rewizja powoduje odmowę.
4. `b0ed5d589a275ac7b161f72f5e1374d52e10b0e7`, 2026-09-24:
   dodano poniższe dwa wiersze; plik ma 14324 bajty, Git blob
   `98e2f74a1519a7c0fa53f650aaa5ba176c62c120` i SHA-256 `81cd0df3…`.

```python
if not hasattr(socket, "AF_UNIX"):
    return FreshnessAuthorityResult(FreshnessAuthorityOutcome.UNAVAILABLE)
```

Po usunięciu wyłącznie tego węzła `If` z AST aktualnego pliku, AST jest równy
wersji pinowanej. Pełny textual diff ma 81 dodanych i 46 usuniętych wierszy;
większość wynika z formatowania, a dodatnia różnica rozmiaru to 178 bajtów.
To nie jest drift CRLF: aktualny blob Git ma inne bajty i inny AST niż pin.
Nie znaleziono w historii zmian badanego pliku dalszej zmiany implementacji
pomiędzy `b0ed5d58…` a bazowym HEAD.

Nowy guard odmawia na platformie bez AF_UNIX przed próbą utworzenia socketu.
Na Linux z AF_UNIX ścieżka wykonania pozostaje taka sama. Nie wprowadza PASS,
nowej roli ani obejścia weryfikacji. Ocena ma służyć decyzji uprawnionego
reviewera; nie zastępuje tej decyzji i nie zmienia reviewed SHA-256.

## Audyt semantyczny granic AccountGenesis

| Obszar | Ustalenie bezpośrednio ze źródła | Wpływ driftu |
| --- | --- | --- |
| Autoryzacja API | Dokładny typ konfiguracji, bezwzględne ścieżki, walidowany port i identyfikator bazy. `authenticate_and_advance` przyjmuje dokładne `bytes`, utrwala trzy wejścia i nie pozwala wstrzyknąć własnego preparation. | Bez zmian. |
| Uwierzytelnienie dokumentów | Adapter deleguje trzy obiekty do verifiera. `FreshnessSemanticVerifier` wymaga kanonicznych schematów, PRODUCTION, dokładnego następnika, związania dokumentu i receipt, retained credentials, Ed25519 i rozdzielenia materiału kluczy ról. Zależności są ustalone przy konstrukcji. | Bez zmian w dwóch rewizjach driftu. |
| Tożsamość IPC | Serwer `serve_local_unix_socket` przyjmuje mutację wyłącznie od UID `os_freshness_runtime` przez `SO_PEERCRED`; osobna gałąź roota jest odczytowa. Główny executable wymaga odrębnego UID verifiera i łączy się z bazą przed bind. | Guard dodaje odmowę przy braku AF_UNIX. |
| Tożsamość serwera po stronie klienta | Klient AccountGenesis łączy się z ustaloną ścieżką AF_UNIX, ale sam nie odczytuje `SO_PEERCRED` serwera ani pidfd. Opiera się na ochronie ścieżki/socketu oraz niezależnej kwalifikacji procesu i deploymentu. | Ograniczenie obecne także w pinowanych bajtach; brak nowej gwarancji. |
| Model PostgreSQL | Klient używa stałej roli `freshness_runtime`, transportu lokalnego i sprawdza `session_user == current_user == freshness_runtime`, oba adresy `NULL`, SSL false. Brak caller-selected roli i SQL. Verifier używa osobnej roli `freshness_crypto_verifier`. | Bez zmian. |
| SERIALIZABLE | Każdy CAS zaczyna `SET TRANSACTION ISOLATION LEVEL SERIALIZABLE`. SQL `compare_and_advance` niezależnie odmawia innego poziomu izolacji. Native SQLSTATE `40001` uruchamia najwyżej trzy próby na nowych połączeniach, z tym samym zamrożonym materiałem. | Bez zmian. |
| CAS | Funkcja SECURITY DEFINER wymaga runtime session_user i właściciela funkcji, blokuje preparation i lineage `FOR UPDATE`, porównuje dokładny predecessor generation/document/head digest, aktualny lifecycle i rozdział ról. W jednej transakcji utrwala dokument, decyzję, receipt, successor head i zużycie preparation. | Bez zmian w adapterze. Definicja funkcji i katalog są osobno kwalifikowane. |
| Idempotencja | Preparation toleruje tylko identyczny materiał przy tej samej tożsamości. Ponowny CAS zwraca `ALREADY_ACCEPTED_EXACT` wyłącznie przy zgodnym binding oraz zachowanych kanonicznych bajtach dokumentu i receipt; inne powtórzenie jest konfliktem. Adapter wymaga zgodności zwróconych receipt bytes. | Bez zmian. |
| Nieznany wynik | Stan REQUEST_STARTED ustawiany przed sendall. Błąd po możliwym wysłaniu oznacza PREPARATION_OUTCOME_UNKNOWN, a nie pewny brak mutacji. Dopuszczalny jest jeden dokładny retry; jego UNAVAILABLE nie usuwa wcześniejszej niepewności. Błąd transportu po wysłaniu CAS daje FRESHNESS_OUTCOME_UNKNOWN, a wyczerpanie prób `40001` również nie daje akceptacji. | Bez zmian. |
| Awaria i korupcja | Nieprawidłowe frame/binding/base64/schema dają CORRUPT. SQLSTATE poprzednika daje CAS_CONFLICT, auth `28000` INVALID_AUTHENTICATION, błędy materiału CORRUPT. Brak połączenia daje UNAVAILABLE. Tylko poprawne CAS_ACCEPTED/ALREADY_ACCEPTED_EXACT mogą być wynikiem pozytywnym. | Guard rozszerza kontrolowaną odmowę niedostępności. |
| Granice zaufania | Root hosta i PostgreSQL superuser pozostają poza gwarancją. Brak odporności na pełny rollback hosta i brak HSM. Lokalna ochrona plików, podpisany interpreter i rzeczywista kwalifikacja usług pozostają konieczne. | Nie powstaje pełny dowód produkcyjny. |

Nie wykonywano rzeczywistego CAS ani provisioningu klastra. Ocena katalogu SQL
jest odczytową analizą implementacji, nie wynikiem kwalifikacji działającej bazy.
Żaden mock nie jest podstawą stwierdzenia pełnej gotowości.

## Wymagane zakresy

Dla pięciu źródeł zgodnych z pinami istnieją bezpośrednie zależności
minimalnego executable: service importuje semantic verifier oraz dwa adaptery
Linux, semantic verifier importuje PostgreSQL authority i IPC, a
`linux_service_ipc` importuje `production_signer_ipc`. Moduł
`postgresql_freshness_authority.py` jest również potrzebny minimalnej usłudze,
ale nie występuje w badanym sześciowpisowym słowniku; jego dokładne bajty muszą
być objęte podpisanym manifestem oraz dopuszczonym materiałem minimalnego
profilu. Własne pliki entrypoint/bootstrap/runtime, inicjalizatory oraz
konfiguracje nie mogą znikać z podpisanego inwentarza.

`account_genesis_freshness_authority.py` jest klientem runtime: wykonuje krok
po przygotowaniu przez verifier. Nie jest importowany przez startowy graf
minimalnego verifiera. Nie należy go dopisywać do minimalnej usługi tylko po
to, by dopasować ją do starego pełnego kwalifikatora. Jest konieczny dla pełnej
kwalifikacji PRODUCTION_LOCAL. Samodzielne skopiowanie go przez administratora
nie tworzy autoryzacji i nie naprawia podpisanego payloadu.

Minimalny inwentarz bazowego instalatora obejmuje 15 ścieżek i scope
`cryptohunter.freshness.substrate.v2`. Podpis inwentarza, zgodność wszystkich
bajtów i zaufanie do reviewed źródeł są różnymi wymaganiami: poprawny podpis
nie zmienia starego pinu AccountGenesis. Pełny kwalifikator wymaga wszystkich
sześciu pinów oraz dalszych dowodów hosta i bazy. W bazowym kodzie minimalny
payload nie zawiera także samego pełnego kwalifikatora ani
`freshness_service_identity`; nie może sam z siebie stać się kompletną
instalacją produktu.

## Decyzja pozostawiona reviewerowi

Zachowano historyczny pin AccountGenesis. Potrzebna jest odrębna decyzja
uprawnionego reviewer, wskazująca dokładny digest, historyczne bajty, semantyczny
guard AF_UNIX oraz zakres wydania, jeżeli nowe bajty mają być dopuszczone do
pełnej kwalifikacji. Dopiero potem można przygotować zgodny, podpisany pełny
payload i oddzielny rzeczywisty dowód hosta. Niniejszy dokument i JSON nie są
taką decyzją ani podpisanym manifestem wydania.

`REVIEWED_SOURCE_DRIFT_ANALYZED=YES`

`FULL_SOURCE_QUALIFICATION=BLOCKED`

`STAGE9_PRODUCTION_READY=NO`
