# PR #3205 — końcowy przegląd kodu i bezpieczeństwa

## A. Stan PR i granice dowodów

Przegląd rozpoczęto od dokładnego HEAD
`8c836bdc698f5f57322113f4b21944b7abbf58fe`, base `aa625cd8`.
Sprawdzono wszystkie 56 plików pierwotnego diffu i pełne **95 wykonanych logów**
siedmiu zakończonych przebiegów: **105 jobów, 92 SUCCESS, 3 FAILURE, 10 SKIP**.
CI push i PR mają po 37 SUCCESS/3 SKIP; Quality 8 SUCCESS/1 SKIP; CodeQL 2 SUCCESS.
Style i Marketplace mają SUCCESS. Trzy odmowy należą do Platform deployment tracks.
Liczby 30/30 nie obejmują całego materializowanego inventory jobów.

| Przebieg badanego SHA | Odnośnik |
| --- | --- |
| CI push | https://github.com/Dudzian/Dudzian/actions/runs/40170343475 |
| CI PR | https://github.com/Dudzian/Dudzian/actions/runs/40170347913 |
| Quality | https://github.com/Dudzian/Dudzian/actions/runs/40170348054 |
| CodeQL | https://github.com/Dudzian/Dudzian/actions/runs/40170347827 |
| Platform | https://github.com/Dudzian/Dudzian/actions/runs/40170347817 |

Status SUCCESS nie wystarczył: dalszy przegląd ujawnił dwie rzeczywiste usterki.
Ten raport to zapis przeglądu źródeł i lokalnej walidacji poprawek. Dokładny nowy
SHA i wyniki dotyczących go kontroli są aktualizowane w opisie tego samego
[PR #3205](https://github.com/Dudzian/Dudzian/pull/3205), bez kosmetycznego commita
wyłącznie aktualizującego stan CI. Wyników `8c836bdc` nie przypisuje się nowemu SHA.

## B. Audyt zmian i architektury

Cały pierwotny diff podzielono na rozłączne zakresy:

| Zakres | Pliki | Najważniejsze sprawdzenia |
| --- | ---: | --- |
| PostgreSQL i sześć znaczników klas testowych | 13 | Transakcje, direct-login roles, guard IDLE, A+C, recovery, zachowane SQL/ACL |
| Qt/UI i testy | 21 | Własność engine/rootów, DeferredDelete, izolacja procesu, propagacja awarii, fonty, aktualne API |
| AnyIO/HTTP i trzy usunięcia komentarzy noqa | 6 | Admission/drain, lock ordering, reentry, shutdown, cache lifecycle |
| Zależności, generator i preflight | 5 | Jeden pin desktop, test-only HTTP stack, kanoniczny generator, rzeczywiste importy |
| Workflow i kontrakty | 5 | Selektory, localhost, zakresy coverage, progi i bramki bezpieczeństwa |
| Dokumentacja | 4 | Zgodność deklaracji z dowodami i granicą produkcji |
| DayTrading i regresje | 2 | Brak fikcyjnej zmienności, neutralny sygnał, skończone wyniki |

Codebase Memory nie był dostępny; wnioski oparto na diffie i źródłach, AST oraz
rzeczywistych logach. Aktualny inventory kwalifikatorów: 2513 śledzonych plików
Python, zero błędów parsowania, 46 call sites, w tym 25 nietestowych.
Sprawdzono również opakowujące metody; nie jest to dowód kompletności dynamicznych
wywołań Pythona.

Konstruktor HSR dodaje wyłącznie transakcję kwalifikacji. Admin approval nadal
działa w jednej SERIALIZABLE z kwalifikatorem i operacją, z SET LOCAL pg_catalog.
Guard odrzuca autocommit-IDLE; session_user/current_user muszą odpowiadać direct
principal. Nie zmieniono SQL function bodies, GRANT/REVOKE, finalnego COMMIT,
protocol owner, trust pins ani write set A+C. Recovery nie zamienia braku markera
w dowód niezrealizowanego COMMIT i nie dodaje automatycznego ponowienia write.

Workbench zachowuje dwa aktualne presety i siedem agregatów, rzeczywiste zwroty
QVariant, odrzucenie nieznanego presetu, demo/live isolation i wyczyszczenie danych
po disable. Nie przywraca usuniętych operacji schedulera. Własny proces, limit,
stdout/stderr i wymaganie exit 0 zachowują awarie natywne jako FAIL. Jawny flush
DeferredDelete dotyczy własnych odbiorców; processEvents nadal ma zakres aplikacji.

DayTrading zwraca zero przy braku dodatniej skończonej zmienności i maskuje
nieprawidłowe bary, zamiast podstawiać fikcyjne 1.0. Dodatkowy probe sprawdził
dziewięć przypadków plugin → ensemble → risk: brak otwarcia pozycji dla neutralnej
strategii, brak NaN w kierunku/rozmiarze; RuntimeWarning traktowano jako błąd.
Zero jednej strategii nie jest globalnym veto dla innych poprawnych strategii
ensemble ani nową klasyfikacją ryzyka. Nie zmieniono centralnych bramek zleceń.

## C. Potwierdzone usterki i minimalne naprawy

### F-ANYIO-1 — wyciek połączenia po TTL

**Root cause:** `_async_open` rozwiązywał klienta w wątku portalu; wygaśnięcie TTL
wywoływało `_close_client` i blokujące run_sync na tym samym wątku. Odmowę łapał
istniejący handler DEBUG, a usuniętego wpisu nie obejmował już cache shutdown.
Default TTL to 300 s. Błąd istniał także w base; GC starego HTTPX klienta nie
zamykał jego TCP. Rzeczywisty test loopback na `8c836bdc` zakończył się FAIL:
po rotacji pozostały dwa połączenia zamiast jednego.

**Minimal fix:** wpis przechowuje liczbę aktywnych żądań, retirement i pojedyncze
claim zamknięcia. TTL/shutdown wycofuje klienta; ostatni użytkownik zamyka go
asynchronicznie na właściwym portalu. Wybór, acquire i claim są pod tą samą
blokadą; wszystkie await/aclose poza nią. Finally oddaje lease także przy błędzie
i anulowaniu, shield kończy cleanup anulowanego AnyIO scope. Factory failure
pozostawia stary wpis dostępny do cleanupu. Nie dodano timerów, zadań w tle ani
zmiany publicznego API. Timeout 0 i mapowanie RequestError → URLError zachowane.

**Regression/result:** siedem nowych przypadków obejmuje realny portal, idle TTL,
dwóch aktywnych użytkowników, powtarzany shutdown, błąd żądania, anulowanie,
factory failure i prawdziwe połączenie HTTP/1.1 na loopback. Po rotacji pozostaje
jedno połączenie; po shutdown zero, bez czekania na GC lub peer-close.
Wraz z poprzednimi testami bridge/cache/timeout: **26 PASS, 9.48 s**.
Niezależny przegląd potwierdził ownership i brak podwójnego close.

### F-QML-1 — trzy samoodwołania runtimeService

**Root cause:** odroczone komponenty Strategies, RiskControls i AiDecisionsView
w MainWindow miały `runtimeService: runtimeService`, odczytując własną właściwość.
Zielone frontend jobs emitowały łącznie 22 binding-loop diagnostics, po 11/job.
Źródła: linie 3148/3331/3336 badanego pliku. Dotychczasowy bootstrap nie tworzył
wszystkich odroczonych komponentów, więc nie zapewniał tej regresji.

**Minimal fix:** trzy istniejące bindingi wskazują `root.runtimeService`.
Nie dodano API ani operacji live. AiDecisionsView odczytuje istniejący
aiGovernorSnapshot; nie uruchamia schedulera, treningu ani zleceń.

**Regression/result:** bootstrap tworzy wszystkie trzy rzeczywiste komponenty,
sprawdza tożsamość kontrolera, odłączenie przez QML null i ponowne podłączenie.
Komponenty są dziećmi contentItem i podlegają istniejącemu cleanupowi okna/engine.
Przed poprawką: **1 FAIL**, kontroler None i rzeczywista pętla bindingu.
Po poprawce bootstrap + kreator: **4 PASS, 4.67 s**, bez diagnostyk QML.
Wszystkie dotychczasowe asercje pozostają. Niezależny przegląd: brak blokera.

### O-1 — brak powodów skipów Windows QML

Wcześniej helper PowerShell nadal używał --disable-warnings i nie miał -ra.
Usunięto wyłącznie flagę ukrywającą podsumowanie i dodano -ra. Porównanie pełnego
skryptu po normalizacji tych dwóch flag jest identyczne: marker, timeout,
ProcDump, artefakty i propagacja exit code nie zmieniły się. Powody 14 historycznych
skipów tego kroku nie są zgadywane; następny run ujawnia je w konsoli.

### Końcowe bramki lokalne

Dokładny hook ruff-format --all-files i Ruff całego zakresu: PASS. Mypy z
niezmienioną konfiguracją: Linux i Win32 po 347 plików PASS. Betterleaks staged:
zero nowych wycieków. Testy HTTP/bridge 26 PASS, Qt 4 PASS, URL guards/DayTrading
24 PASS; niezależna walidacja guardów/metryk ma 20 PASS. Nie sumuje się tych
częściowo pokrywających się zestawów.

Dodatkowe mypy pojedynczego HTTP modułu, poza skonfigurowanym scope, ma cztery
identyczne diagnostyki przed/po (porównane przez shadow-file z `8c836bdc`): brak
HTTPX w venv narzędziowym, dotychczasowy typ Request.data, typ hdrs HTTPError
i Any z istniejącego wrappera. Nie zgłoszono nowej diagnostyki z naprawy.
Nie rozszerzano lub ograniczano bramki ani nie poprawiano tych adnotacji kosmetycznie.

## D. Warning audit i Semgrep

Poniższe liczby dotyczą 95 pełnych wykonanych logów `8c836bdc`, nie nowego SHA.

| Diagnostyka | Źródło/przyczyna | Wpływ, oczekiwanie i rozstrzygnięcie |
| --- | --- | --- |
| 66 Node DEP0005 | download-artifact v8.0.2 → @actions/artifact 6.3.1 → unzip-stream 0.3.4 / new Buffer | Upstream deprecation. Brak bezpiecznej wydanej naprawy; pozostaje jawna. Wszystkie 68 par digestów pobrań pasują. Bez forków i osłabienia digest-mismatch:error. |
| 30 CoverageWarning module-not-imported | HSR używa wspólnego scope, którego część mierzy Atomic | Wszystkie 30 mają rzeczywiste wiersze pomiaru w Atomic. Union 57 modułów i dane branch/arcs faktycznie łączone; próg 70% i zakres identyczne z base. Combined: 87%, 7/8 zmienionych linii, nie 100%. |
| 20 trust messages | Start disposable PostgreSQL | Oczekiwane dla syntetycznych direct-login/ACL fixtures. Host loopback nie izoluje innych kontenerów tej samej sieci. Granicą zaufania jest cały jednorazowy job. Nie jest to produkcyjne uwierzytelnienie. |
| Odmowy SQL i konflikty | Negatywne role/ACL, wyścigi i SERIALIZABLE | Oczekiwane odmowy/recovery; sprawdzane brak approval/write po odmowie, pełny snapshot i wynik retry. Nie są zamieniane w PASS przez nowy wyjątek. |
| Recovery/WAL po SIGKILL | Kontrolowana awaria Atomic/HSR | Oczekiwane testy trwałości. Zachowane unknown outcome, brak częściowego write set i retry pierwotnej próby po recovery. Bez zmiany parametrów produkcyjnej trwałości. |
| 20 PyInstaller | 4 opcjonalne sklearn, 12 DBAPI, 4 NumPy array_api | hooks-contrib 2025.10 bezwarunkowo próbuje sklearn; PyInstaller 6.5.0 hardkoduje pysqlite2/MySQLdb/psycopg2. Runtime PostgreSQL używa psycopg3; wybór niedostępnego sklearn jest jawnie odrzucany. NumPy 1.26.4 experimental import pochodzi z DLL discovery. Wymagane importy Win32/OpenBLAS i packaged EXE smoke PASS. Bez zbędnych zależności. |
| 22 QML binding loops | Trzy odroczone runtimeService bindingi | Rzeczywisty błąd, naprawiony jako F-QML-1. Nie uznano go za nieszkodliwy tylko dlatego, że job był zielony. |
| EGL w 112 liniach tekstu | -ra raportuje optional UI collection skipy w zwykłych shardach | Brak failure kolekcji; markery kierują wymagane Qt do dedykowanych jobów. Nie wolno traktować samych trafień tekstowych jako 112 błędów EGL. |
| RuntimeWarning empty mean, ResourceWarning, PytestCollectionWarning, Exception ignored in | Brak nieoczekiwanych payloadów tych klas w wykonanych logach | To ograniczone stwierdzenie o odczytanych logach, nie dowód braku wszystkich usterek. F-ANYIO-1 wykryto niezależnie mimo braku takiego warningu. |

Szczegóły upstream oraz źródła Qt/izolacji są w poprzednich raportach tego PR.
Nie zmieniono konfiguracji, baseline ani suppressions skanerów.

### Wszystkie 17 community advisories

Lista check_id/path/line/message została dostarczona przez użytkownika z
semgrep-community.json; każdą lokalizację sprawdzono bezpośrednio w źródłach.
Oryginalnych bajtów ZIP nie pobrano: host File Service i registry semgrep.dev
odmawiają dostępu przez politykę środowiska. Nie przedstawia się listy użytkownika
jako samodzielnie pobranego JSON ani nie opiera klasyfikacji na samym SUCCESS.

| # | Lokalizacja badanego SHA | Rozstrzygnięcie i istotna granica |
| --- | --- | --- |
| 1 | bot_core/alerts/channels/_http.py:21 | Dynamic urllib: istniejące require_http_request przed transportem; file/ftp i URL credentials odrzucane. |
| 2 | bot_core/alerts/channels/signal.py:33 | Ten sam guard, endpoint konfiguracji operatora; treść alertu nie wybiera URL. Lokalny Signal jest zamierzonym endpointem. |
| 3 | bot_core/alerts/channels/sms.py:23 | Ten sam guard dla skonfigurowanego providera; payload/recipients nie zmieniają scheme transportu. |
| 4 | bot_core/freshness_semantic_verifier.py:696 | 0660 dotyczy Unix socketu, nie publicznego pliku. Owner/group ACL jest celowe; SO_PEERCRED sprawdza dokładny allowed_peer_uid przed ramką. 0644 nie jest poprawnym zamiennikiem. |
| 5 | bot_core/logging/app.py:92 | Walidacja endpointu w konstruktorze i require_http_request przy wysyłce; log record jest payloadem. |
| 6 | bot_core/persistence/physical_backup.py:264 | 0700 na świeżym mkdtemp to owner-only staging directory, nie nadmierna dostępność; 0644 odebrałoby traversal i dodało world-read. Nie jest to samodzielny dowód Windows ACL. |
| 7 | bot_core/runtime/metrics_alerts.py:382 | Guard konstruktora i requestu; docelowy webhook z konfiguracji, nie z alert context. |
| 8 | bot_core/update/differential.py:271 | urlopen tylko w gałęzi http/https po require_http_url. Oddzielna jawna ścieżka lokalnego importu jest intencjonalna; nie dowodzi to bezpieczeństwa dowolnego wskazanego przez operatora pliku. |
| 9 | scripts/benchmark_issuer_authenticated_history.py:725 | Stały interpreter/moduł, lista argv, shell=False domyślnie; rozmiar/samples są liczbami z CLI. Brak shell expansion, shlex.quote nie jest rozwiązaniem dla argv. |
| 10 | scripts/dr_failover_validation.py:81 | Skonfigurowany webhook i oba HTTP guards; file URI nie trafia do transportu. |
| 11 | scripts/dr_failover_validation.py:154 | HTTP-only helper chroni zapytania do skonfigurowanego Prometheus. |
| 12 | scripts/dr_synthetic_probes.py:63 | HTTP-only guard przed request/urlopen; endpoint pochodzi z operator CLI. |
| 13 | scripts/list_exchange_adapters.py:200 | Walidowany HTTP endpoint dashboardu; CSV stanowi body. |
| 14 | scripts/marketplace_cli.py:160 | Zdalny katalog przechodzi HTTP guard; lokalny/file import jest odrębnym, jawnym trybem operatora, nie fallbackiem urlopen. |
| 15 | scripts/prepare_stage41b_environment.py:98 | SHA1 odtwarza format Git object ID. Nie jest podpisem ani password hash; inwentarze/pliki/wheels mają osobne SHA256. Bez zmiany Git formatu lub trust pins i bez uruchamiania Stage 41B. |
| 16 | scripts/publish_licensing_drift_metrics.py:90 | Istniejący inline scheme http/https + hostname + zakaz credentials. Dane .prom nie wybierają URL. |
| 17 | scripts/run_paper_smoke_ci.py:1715 | Jawne shell=False i argv ze stałym interpreterem/skryptem. Operator może przekazać dodatkowe argumenty CLI, nie zdalny payload. Nie jest to powłoka ani ograniczona piaskownica dla niezaufanego operatora. |

Dla 12 sinków HTTP guardy nie są host/IP allowlistą ani ogólną ochroną SSRF;
zaufana konfiguracja endpointów pozostaje granicą. Standard urllib nie pozwala
HTTP redirect do file://. Jawnych lokalnych importów nie utożsamia się z
niekontrolowanym odczytem przez HTTP. Nie wykazano podatności wymagającej nowej
zmiany w tych 17 miejscach; nie nazywa się ich zbiorczo bezwarunkowo bezpiecznymi.
Istniejące adnotacje nosemgrep nie są podstawą tej oceny i nie zostały rozszerzone.

## E. Skipped tests

W obu przebiegach main: Ubuntu 22980 PASS / 121 SKIP / 1471 deselected;
Windows 21277 PASS / 1785 SKIP / 1491 deselected; macOS 21467 PASS / 1595 SKIP /
1491 deselected. Dodatkowy Linux sudo harness wykonuje 21 testów natywnych.
Różnicy SKIP nie interpretuje się bez markerów i dedykowanych shardów.

- Oczekiwane: inne systemy plików/Win32/POSIX, brak fizycznego TPM/verified
  production package, OEM modeli, osobnego root harness i opcjonalnego browsera;
  puste macierze przypadków oraz przeniesienie ciężkich PG do osobnych shardów.
- Linux QML: 16374 PASS/7 SKIP; sześć pustych macierzy i Playwright.
  Windows/macOS Workbench oraz wymagany case QCore-parent/brak Playwright
  faktycznie wykonały się. Fonty Windows nie są już lokalną symulacją jako jedyny dowód.
- Niekwalifikowane: przypadki brakujących mypy/grpc_tools w danym jobie,
  TicketDialog wymagający flagi PYTEST_REQUIRE_QML nieustawionej na macOS/Windows,
  sześć root-process testów pominiętych w Atomic oraz brak browser mirror.
  Wykonanie innego sharda nie jest dowodem wykonania tych konkretnych node IDs.
- Powody 14 skipów kroku Windows QML nie były ujawnione w jego konsoli.
  Nie przypisuje się im powodów przez analogię do macOS; O-1 przywraca widoczność.

Nie dodano SKIP/XFAIL ani nie usunięto asercji w celu przejścia CI. Brak dowodu
produkcyjnej kwalifikacji pozostaje jawny również dla zielonego marketing joba,
który zwraca raport SKIPPED przy braku current evidence/mirror.

## F. Security assessment

Zmiany PostgreSQL są GO jako kod w zbadanym zakresie. HSR 106 PASS, Atomic
4040 PASS/7 SKIP i combined 87% dotyczą dokładnie `8c836bdc`. Wszystkie dziewięć
rzeczywistych testów transakcyjnych wykonało się w obu remaining-security jobach;
odmowa guardu nie jest pominiętą asercją mimo osobnego scope raportu coverage.

Porty 14 serwisów są localhost; **trust dopuszcza podszycie się pod LOGIN przez
inne kontenery tej samej sieci i lokalne procesy runnera**. Zaufany jest cały
jednorazowy GitHub-hosted job z syntetycznymi danymi. Nie wolno używać tego modelu
na shared/self-hosted runnerze z niezaufanymi peerami ani w produkcji. Natywny
Linux peer-auth proof i wymagania produkcyjne są oddzielne, nie zmienione.

Pozostają Stage 9 IN_PROGRESS / NO-GO, private_control_status BLOCKED,
production_provisioning_ready false, ROOT_PROOF_ISSUANCE_AVAILABLE false,
produkcyjne SIGN/BIND zablokowane i Stage 10 NO-GO. Nie zmieniono uprawnień,
frozen contracts, trust pins, CodeQL, Semgrep ani Quality Ratchet. Nie wykonywano
provisioningu, enrollmentu, aktywacji signerów, fizycznego Stage 41B, Daybreak
ani live trading.

## G. Pozostałe blokady

Trzy czerwone joby aktualnego Platform run mają nadal MISSING_CURRENT_EVIDENCE /
PRODUCTION_READY=NOT_READY: Linux/macOS deployment i Windows release gate.
Ich pełne logi odczytano; nie są błędami do naprawienia przez fałszywy evidence
lub zmianę statusu. Brak fizycznego Maca nie znosi wymagania.

Nowy commit wymaga walidacji dotkniętych ścieżek: main/exchange na trzech OS,
frontend bootstrap, Windows QML diagnostics, format/lint/type/security scans.
Niezmienionych wielogodzinnych testów PostgreSQL nie ponawiano lokalnie.
Automatycznych przebiegów nowego SHA nie przedstawia się jako PASS przed odczytem.

## H. Rekomendacja merge i production

`8c836bdc` miał zielone bramki kodu, ale F-ANYIO-1 i F-QML-1 uzasadniały naprawę
przed rekomendacją merge. Obie minimalne poprawki mają odtwarzalne regresje
i niezależny source review; nie są kosmetycznym czyszczeniem logów.

**GO dla merge jako zmiany kodu wymaga PASS odpowiednich kontroli nowego SHA**
i braku nowych istotnych diagnostyk. Do tego momentu rekomendacja to NO-GO.
Aktualny wynik tej oceny jest podany w opisie PR, bez przypisywania starszych
wyników nowemu commitowi. Nie wykonano merge ani nie utworzono nowej gałęzi/PR.
Niezależnie od code GO: **Stage 9 i Stage 10 production pozostają NO-GO**.
