# CryptoHunter Stage 9 — remediacja audytu fc15b982

Remediacja powstała na osobnej gałęzi `codex/stage9-security-remediation`,
bezpośrednio z `fc15b98298c5430286dd2206ec7eeb8f20e2497b`.
**S3 pozostaje BLOCKED; PRODUCTION pozostaje NO_GO.** Przygotowano lokalny szkic PR do
przeglądu; publikacja nie została jeszcze wykonana. Nie wykonano merge, instalacji, startu usług ani próby VM.

| Ustalenie | Status poprawki i lokalnych testów | Status dowodu na właściwym hoście/VM |
| --- | --- | --- |
| A — HIGH, wybór niezweryfikowanego importu | PASS w opisanym zakresie kodu/testów | BLOCKED |
| B — MEDIUM, konfiguracja/provider OpenSSL | PASS w opisanym zakresie kodu/testów | BLOCKED dla pochodzenia systemowego OpenSSL i jego bibliotek |
| C — HIGH, ochrona checkoutu i kompletność | PASS narzędzia/procedury | BLOCKED dla zbudowania i sprawdzenia chronionej instalacji |
| D — HIGH, pochodzenie zewnętrznego runtime | PASS testów kolektora; samo ustalenie BLOCKED | BLOCKED dla wszystkich wymaganych dowodów pochodzenia |

PASS lokalnych testów nie zamyka blokad hosta ani nie jest deklaracją gotowości.
Pełny wykaz zmienionych linii znajduje się w
[evidence/stage9-security-remediation/changed-lines.json](evidence/stage9-security-remediation/changed-lines.json).

## A — dokładny import harnessu

**Przyczyna.** Poprzednie `_native_module()` sprawdzało sąsiadujące `*.py`,
a następnie oddawało wybór kodu `importlib.import_module`. Python mógł wybrać
katalog pakietu, natywne rozszerzenie albo istniejący obiekt w `sys.modules`.
Kontrola wyłącznie plików `.py` pomijała dodatkowe katalogi i cache. Wcześniejszy
helper środowiska także wykonywał się przed kompletnym uwierzytelnieniem drzewa.

**Pliki i linie.** `deployment/stage9_vm_trial.py`: 10–96 (kontrakt),
166–180 (stabilny odczyt), 241–254 (helpery), 410–719 (inwentarz, graf,
cache i loader), 912–913 (kontrola przed pierwszym helperem).
`tests/deployment/test_stage9_harness_import_security.py`: 1–366;
`test_stage9_harness_runtime_guard.py`: 1–156. W istniejącym
`test_stage9_vm_trial.py` dodano jawny model uwierzytelnienia dla testów samej
orkiestracji; rzeczywiste kontrole plików mają osobne testy adversarialne.

**Poprawka.** Stały zewnętrzny inwentarz
`/etc/cryptohunter/stage9-harness-inventory.json` wiąże źródła z pełnym commitem
i absolutną ścieżką drzewa. Dozwolonych jest dokładnie 18 plików oraz trzy
katalogi: `.`, `deployment`, `bot_core`. Każdy obiekt, także pusty dodatkowy
katalog, powoduje odmowę, jeśli nie należy do kontraktu. Sprawdzane są UID root,
przodkowie, typy, tryby `0555/0444`, brak linków i integralność SHA-256 oraz
stabilność odczytu. Inwentarz sam pochodzi z niezależnie zweryfikowanej instalacji
opisanej w C; hash z transportu sam nie ustanawia zaufania.

Graf obejmuje wszystkie węzły AST, w tym importy opóźnione i `TYPE_CHECKING`,
z jawną listą 14 modułów projektowych i czterech helperów wykonywanych jako skrypty. Nieznane zależności i niezatwierdzone
mechanizmy dynamicznego importu są odrzucane. Projektowe namespace mają puste
`__path__`; resolver nie wybiera dla nich pliku, pakietu ani rozszerzenia.
Loader kompiluje dokładnie sprawdzone bajty `.py`, bez odczytu bytecode.

Przed helperami wymagany jest świeży proces `-I -B`, bez wstępnie załadowanych
modułów projektowych lub zewnętrznych. Obce meta hooks, path hooks i cache
są odrzucane bez wywoływania ich callbacków. Stan importów jest budowany
ponownie z kanonicznych mechanizmów CPython. Po ponownej weryfikacji
zainstalowanego runtime jego podpisany inwentarz kontroluje również pochodzenie,
hash, rozmiar i tryb każdego nowego importu zewnętrznego, w tym rozszerzeń.
Pliki odpowiadające już załadowanemu stdlib również są sprawdzane.

**Testy.** 42 testy nowej granicy harnessu i 13 testów importów runtime:
podstawienie pakietu, dwóch nazw `.so`, zależności projektowej i zewnętrznej,
cache, dodatkowego katalogu, zmiana hasha/trybu/rewizji/root, duplikaty,
niebezpieczne hooki i cache modułów, właściciel i prawa przodków,
linki, bytecode i błędne rekordy manifestu. Dodatnie testy wykonują wyłącznie
zweryfikowane rzeczywiste źródło `stage9_vm_native.py` lub kopię zaufanego
stdlib `textwrap`. Podstawione obiekty pozostają danymi. Osobny rzeczywisty
proces potwierdza wybór loadera źródłowego. Rzeczywiste CLI na niechronionym
checkout odmawia: exit **2**, `INPUT_NOT_PROTECTED`, wszystkie fazy
`NOT_EXECUTED`; zachowano odpowiedź i kod wyjścia.

**Ryzyka.** Interpreter i początkowy stdlib są częścią niezależnego bootstrapu
systemu. Kontrola plików nie dowodzi poprawności wcześniej skompromitowanego
procesu, kernela ani administratora root. Przed natywnym workerem rodzic nadal
weryfikuje zainstalowany runtime. Nie wykonano kompletnego workera na docelowej
VM; ta próba pozostaje BLOCKED. `INSPECT` także wymaga chronionego drzewa;
niechroniony checkout nie służy już jako instalacja wykonywalna.

## B — zamknięta konfiguracja bootstrapu OpenSSL

**Przyczyna.** Ochrona `/usr/bin/openssl` nie zamykała domyślnego odczytu
konfiguracji, `.include`, engine ani ładowania providerów.

**Pliki i linie.** `deployment/freshness_runtime_bootstrap.py`: 9, 16–71,
85–157. `tests/deployment/test_freshness_runtime_bootstrap_security.py`: 1–217.

**Poprawka.** Bootstrap ma dokładny kontrakt bajtów konfiguracji i jego SHA-256,
jedyny provider `default` oraz jawny filtr `provider=default`. Dane, podpis,
klucz publiczny i konfiguracja trafiają do zapieczętowanych memfd. Ścieżki
konfiguracji, include, providerów i engine wskazują sprawdzony memfd; jest on
zwykłym plikiem, więc nie może być katalogiem zawierającym moduły.
Środowisko potomka jest jawne. Nie dodano importu `cryptography` do bootstrapu.
Istniejące `verify_ed25519(raw, signature, key)` nadal poprzedza `_json(raw)`
manifestu runtime.

**Testy.** 11 nowych testów: rzeczywiste OpenSSL i opublikowany wcześniej
publiczny wektor RFC8032 TEST 2; zły podpis, wiadomość i klucz; zmieniony
kontrakt, `.include`, nieznany provider; kernel odmawia modyfikacji i truncate
każdego memfd; import bootstrapu przy zakazanym `cryptography`; odmowa
interpretacji manifestu po błędzie weryfikacji. `strace` rzeczywistego procesu
potwierdza brak otwarcia podstawionych konfiguracji, include i providera.
Przeszło też 19 starszych testów bootstrapu, bez fixture generujących klucze.

**Ryzyka.** Loader, libcrypto i pozostałe biblioteki systemowe nadal wymagają
rozliczenia z D. Testy modelują ochronę właściciela samego systemowego OpenSSL
w tym executorze; wykonanie OpenSSL, seal i ślad syscall są rzeczywiste.
Brak właściwej wersji OpenSSL 3 lub seal powoduje odmowę, a nie fallback.

## C — osobne chronione drzewo

**Przyczyna.** Checkout edytowalny przez użytkownika i TAR transportowy nie
ustanawiają instalacji chronionej. Lista wybranych plików bez porównania całego
zbioru nie wykrywa dodatkowego kodu, katalogu ani obiektu specjalnego.

**Pliki i linie.** Nowy `scripts/prepare_stage9_protected_tree.py`: 1–464,
zwłaszcza 135–244, 246–359 i 361–423. Nowy
`tests/deployment/test_stage9_protected_tree.py`: 1–283. Procedura:
[stage9_protected_execution_tree.md](../runbooks/stage9_protected_execution_tree.md).

**Poprawka.** Nieuprzywilejowany eksport odczytuje jawnie wybrane bloby
z dokładnego pełnego commita, weryfikuje identyfikatory commita/drzew/blobów
i nie czyta odpowiadających im bajtów checkoutu. Kandydat pozostaje transportem.
Osobna operacja root wymaga niezależnego pinu inwentarza, sprawdza cały zbiór
i kopiuje zweryfikowane bajty do nowego drzewa pod chronionymi przodkami.
Zewnętrzny inwentarz jest publikowany dopiero po kontroli nowego drzewa.
Istniejące cele nie są nadpisywane. Nie ma wejścia TAR ani automatycznej
naprawy. Osobne `verify` sprawdza prawa inwentarza przed jego interpretacją.

**Testy.** 29 testów PASS: dokładne obiekty Git, podmiana commita/drzewa/bloba,
niezależność od zmian checkoutu, kompletność katalogów, dodatkowe `.so`/pakiety,
linki/FIFO, brak pliku, zmiana treści, schema/pin i odmowa fazy root.
Narzędzie nie zmienia praw oryginalnego repozytorium.

**Ryzyka i status.** Faktyczne stworzenie i kontrola instalacji root na
właściwym hoście/VM: **BLOCKED**. Procedura podaje konkretne 18 ścieżek,
komendy, kody wyjścia, porównanie zbioru i sposób zachowania dowodów.
Nie wykonano symulacji sukcesu instalacji ani kwalifikacji fizycznej.

## D — 1352 zewnętrzne wpisy runtime

**Przyczyna.** Zgodność lokalnego hasha lub wpisu dpkg nie ustanawia
uwierzytelnionego pochodzenia pakietu, wygenerowanego bytecode ani konfiguracji
loadera. Dane audytu wskazują 573 `.pyc` i 19 innych nierozstrzygniętych plików.

**Pliki i linie.** Nowy `scripts/stage9_runtime_provenance.py`: 1–238,
zwłaszcza 75–115, 122–217 i 219–238;
`tests/deployment/test_stage9_runtime_provenance.py`: 1–176. Procedura:
[stage9_runtime_provenance.md](../runbooks/stage9_runtime_provenance.md).

**Poprawka proceduralna.** Odczytowy kolektor przyjmuje digest wyniku
wcześniejszej niezależnej weryfikacji podpisu; sprawdza go przed JSON.
Raportuje każdy obiekt, dokładne wersje i architektury dpkg oraz surowe
obserwacje właściciela pakietowego. Czyta `.pyc` wyłącznie jako bajty.
CLI wymaga liczebności 1352/573; niespójność daje FAIL/exit **1**, zgodna
obserwacja wymagająca pochodzenia daje BLOCKED/exit **2**. Nie zwraca PASS.

Procedura wymaga łańcucha niezależna kotwica Ubuntu → InRelease → Packages →
dokładny `.deb` → obiekt payloadu. Obejmuje pełne wersje z epoch/revision,
573 pojedyncze dowody bytecode, dokładne 19 ścieżek z audytu, generowane pliki,
symlinki, loader, libc, OpenSSL i natywne closure `DT_NEEDED`/`dlopen`/NSS.
Nie uruchamia `ldd`, bytecode ani niezaufanego ELF, niczego nie instaluje,
nie kompiluje i nie zmienia manifestu.

**Testy.** 16 testów PASS: digest przed JSON, schema/ścieżki/duplikaty,
race odczytu, pliki/linki/specjalne obiekty, `.pyc` jako dane, błędy dpkg,
jawne statusy i rzeczywiste kody wyjścia 1/2 w modelu kolektora.

**Braki i status.** **BLOCKED**: nie dostarczono podpisanego manifestu z tymi
1352 wpisami, dokładnej listy 19 plików, artefaktów `.deb` i metadanych
archiwum ani dostępu do badanego hosta. Nie odtworzono tych liczebności na
executorze i nie przedstawia się jego pakietów jako pakietów badanego hosta.
Nie aktualizowano manifestu i nie przebudowywano runtime.

## Wykonana walidacja i zachowane dowody

[evidence/stage9-security-remediation](evidence/stage9-security-remediation)
zawiera logi, JUnit XML, metadane argv/czasów i oryginalne exity, hashe źródeł
oraz rzeczywisty ślad `strace`. Środowisko to nieuprzywilejowany executor
Cloud z CPython 3.11; nie jest właściwą VM Ubuntu 24.04/CPython 3.12.

| Kontrola | Wynik |
| --- | --- |
| Regresje i adversarialne, 433 przypadki | PASS, exit 0, bez SKIP |
| Wybrane starsze testy bootstrapu, 19 przypadków | PASS, exit 0; 61 innych jawnie niewybranych |
| Skonfigurowany mypy Linux, 395 źródeł | PASS, exit 0 |
| Skonfigurowany mypy Win32, 395 źródeł | PASS, exit 0; analiza statyczna, bez wykonania Windows |
| Repozytoryjny Ruff baseline i oba zmienione moduły deployment | PASS, exit 0 |
| Dokładny `python -m pre_commit run ruff-format --all-files` | PASS, exit 0, po dodaniu nowych plików do indeksu |
| Osobny format zmienionych modułów deployment | PASS, exit 0 |
| Betterleaks 1.9.0, offline staged scan z istniejącym baseline | PASS, exit 0 |
| Fizyczna instalacja, VM, systemd/PostgreSQL i pochodzenie systemu | BLOCKED, NOT_EXECUTED |

Wszystkie nowe testy dodano do istniejących zakresów CI, dwa narzędzia do mypy
i coverage, a `strace` do dwóch jobów Linux. Progi i ręczne bramki zachowano.
Codebase Memory nie było dostępne; wnioski zweryfikowano bezpośrednio w źródłach.

Nie uruchomiono pełnego zestawu testów generujących klucze/podpisy, testu
celowo wykonującego podstawiony bytecode ani natywnego buildera. Istniejący
workflow `Freshness Python runtime` automatycznie budowałby runtime na PR.
Dlatego commit zawiera `[skip ci]`, a przygotowany PR ma pozostać **draft z niewykonanym CI**.
Nie wyłącza to konfiguracji ani wymaganych bramek i nie zastępuje ich wynikiem
PASS. Uruchomienie workflow wymagających zabronionych operacji wymaga osobnej
decyzji zgodnej z zakresem zadania. Nie generowano kluczy, podpisów ani approval.
