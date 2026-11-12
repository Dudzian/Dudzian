# PR #3265 — poprawki po review

Punkt odniesienia review: `5e0646a533d071b3f4a1c62d8d9836ece9081c95`.
Poprawki pozostają na `codex/stage9-security-remediation` w istniejącym PR
#3265. **S3: BLOCKED. PRODUCTION: NO_GO.** Nie wykonuje się merge ani
kwalifikacji Stage 9. Poprzednie dowody i artefakty wydania pozostają bez zmian.

## 1. HIGH — jedna rewizja runtime, aplikacji i narzędzi

Manifest runtime `release_revision`, chroniony binding `source_revision`,
podpisany manifest aplikacji v2 i jego obiekt `runtime`, oczekiwana rewizja
instalatora, inwentarz harnessu i zewnętrzne dopuszczenie VM muszą wskazywać
to samo pełne `R`. Kontrole dokładnych bajtów helperów instalatora obowiązują
niezależnie od równości tych pól. Weryfikacja podpisu nadal poprzedza JSON.

Runtime `fc15b98298c5430286dd2206ec7eeb8f20e2497b` nie jest zgodnym podłożem
nowego wydania z poprawionym bootstrapem. Obecny kontrakt wymaga nowego wydania
runtime dla finalnego zatwierdzonego `R`, utworzonego w osobno autoryzowanym
procesie budowy. Nie dowodzi to potrzeby zmiany wersji pakietów. Nie ma jednak
zatwierdzonej operacji przepięcia starego podpisanego materiału na nowy SHA.
Nie wolno przepisać manifestu, bindingu lub hashy, by wymusić PASS.

W `freshness_application._runtime_binding` wydzielono dotychczasową bramkę
równości do testowania bez generowania podpisów. Jej wywołanie nadal następuje
po podpisie i kontroli inwentarza. Instalator `_verified_runtime` dodatkowo
sprawdza bezpośrednio `runtime.revision == config.expected_revision`.
25 testów w pamięci odrzuca niedozwolone kombinacje rewizji i bindingów;
nie tworzą one podpisanych manifestów ani syntetycznych zatwierdzeń.

**Status: PASS kontroli kodu/testów; migracja BLOCKED / NOT_EXECUTED.**
[Procedura migracji](../runbooks/stage9_revision_migration.md) wskazuje dokładne
obowiązki operatora. Osobna zgoda musi objąć nie tylko nowy build, lecz także
jego sondę `_PROBE`, która generuje efemeryczny klucz i podpis, późniejsze
materiały podpisu i odrębne dopuszczenie nowej VM. Niczego tego nie wykonano.

## 2. MEDIUM — jednoznaczny INSPECT

INSPECT jest odczytową diagnostyką przed instalacją produktu. Wymaga jednak
osobnego uwierzytelnionego drzewa 18 źródeł oraz chronionego zewnętrznego
inwentarza; nie wykonuje dowolnego checkoutu. Nie wymaga zainstalowanego runtime,
aplikacji, ich wejść ani dopuszczenia VM. Nie wymaga root do samej diagnostyki.
Wymagana izolacja interpretera to `-I -s -B`.

Poprawiono główny runbook, procedurę chronionego drzewa i pomoc CLI. Usunięto
sprzeczną instrukcję kopiowania całych katalogów `bot_core`/`deployment`, które
nie odpowiadały dokładnemu inwentarzowi. Niedozwolone argumenty INSPECT są
odrzucane przed załadowaniem helpera. Brak chronionego drzewa oznacza odmowę,
a nie diagnostykę wykonaną niezweryfikowanym kodem.

7 testów obejmuje rzeczywisty helper środowiska przy jawnym modelu granicy
uwierzytelnienia, brak wywołań produktu/VM, pomoc CLI i rzeczywisty izolowany
proces odmawiający wykonania niechronionego checkoutu. **Status: PASS** w tym
zakresie; próba na docelowym hoście pozostaje NOT_EXECUTED.

## 3. HIGH — bezpieczna walidacja GitHub Actions

Istniejący `platform-deployment.yml` ma osobny dispatch `validation_scope:
stage9-safe`, dla Python 3.11 i 3.12, z wymaganym pełnym `expected_commit`.
Job odrzuca inną rewizję i flagi Stage 10/physical. Uruchamia wyłącznie jawnie
wybrane testy bez fixture generujących materiały, bez konftest i automatycznych
pluginów pytest. Wszystkie dotychczasowe joby, ich kroki, zależności i bramki
pozostają zachowane w zwykłym PR i pełnym dispatch; nie są wykonywane w tym
oddzielnym, jawnym zakresie.

Runner zachowuje format `pre_commit ruff-format --all-files`, Ruff, mypy
Linux/Win32, skan Betterleaks zmienionych ścieżek, regresje/adversarialne i próg
**70% diff coverage** liczony od pierwotnego audytowanego `fc15…`. Nie zawęża
coverage do samych korzystnych plików; obejmuje wszystkie zmienione źródła
produkcyjne. SKIP, pusty JUnit i brak narzędzia nie stają się PASS. Każdy krok
zachowuje argv, czas UTC, rzeczywisty exit lub jawny brak wykonania, stdout/
stderr, JUnit i coverage. Kontrola końcowa wymaga niezmienionego checkoutu.

Commit zachowuje `[skip ci]` dla automatycznych, niedozwolonych w tym zadaniu
workflow. Ręczny `stage9-safe` nie jest blokowany tym znacznikiem. Nie zastępuje
pełnych wymaganych bramek. Generatory kluczy/podpisów, syntetyczne dopuszczenia,
builder, celowe wykonanie podstawionego bytecode i próby hosta/VM mają jawny
status **NOT_EXECUTED**. Ich pominięcie nie jest pozytywną kwalifikacją.

**W chwili zapisu raportu GitHub CI: NOT_EXECUTED.** Po publikacji zamrożonego
HEAD uruchamiany jest wyłącznie powyższy dispatch; jego aktualny wynik i adres
runu są podane w opisie istniejącego PR. Pierwsza próba dispatch dla wcześniejszej rewizji została odrzucona przez
walidator GitHub (HTTP 422: niedozwolone `runner.temp` w job.env); nie utworzyła
runu. Ścieżka jest teraz składana w kroku na przypisanym runnerze i ma własną
regresję. Pierwszy rzeczywisty run
[43319211410](https://github.com/Dudzian/Dudzian/actions/runs/43319211410)
zachował na obu Pythonach FAIL jednego fixture importu projektowego (483/484
przypadki PASS), przy pozostałych bramkach exit 0. Fixture zakładał uprawnienia
chronionego stdlib w development toolcache `setup-python`. Teraz izoluje
wyłącznie wybór źródła projektu, z zaufanym stdlib załadowanym w bootstrapie.
Oddzielny test pełnego chronionego grafu i odmowy metadanych pozostają bez
osłabienia; produkcyjny loader nadal odrzuca niechronione pliki. Włączono też
retencję jawnie wskazanego ukrytego pliku `.coverage` w artefakcie CI.
Autorytatywny HEAD wykonania znajduje
się w `validation.json.context.commit` obu artefaktów macierzy. Raport nie
przedstawia wyników lokalnych jako wyników GitHub Actions.

## 4. Dodatkowa regresja granicy zaufania

Potwierdzono rzeczywisty błąd uruchomieniowy poprzedniego loadera: pierwszy import
`freshness_runtime → dataclasses → copy` próbował opcjonalnego
`org.python.core`. Brak modułu powodował `TrialError`, którego stdlib nie
przechwytuje. Teraz brak spec powoduje `ModuleNotFoundError` i kończy wyszukiwanie
bez przekazania do obcego findera. Istniejący niezatwierdzony plik lub namespace
nadal powoduje twardą odmowę przed wykonaniem. Nie jest to zgoda na fallback.

Przed ustanowieniem runtime loader dopuszcza wyłącznie niezależnie chroniony
host stdlib i kompiluje sprawdzone bajty źródeł, zamiast korzystać z bytecode.
Ścieżka stdlib pochodzi wyłącznie z wartości zaufanego interpretera, bez
konfigurowalnego importu `sysconfigdata` i bez rekurencji findera. Dla pakietów zachowano lokalizację
spec, `__file__`, `__cached__` i `__path__`. Kontrola import state sprawdza
ścisłe typy kontenerów i modułów przed iteracją/odczytem atrybutów; wyjątki
aliasów `typing.io/re` wiążą się z tożsamością rzeczywistego `typing`.
Odczyt pliku porównuje metadane ścieżki z otwartym deskryptorem przed czytaniem
oraz ponownie po odczycie. Nowe testy odrzucają podmianę deskryptora i obce
kontenery bez wywołania callbacków.

Test dodatni uruchamia dokładną kolejność bootstrapu, wszystkie 14 przejrzanych
modułów projektowych oraz rzeczywiste zależności development cryptography/
psycopg na obu wersjach Python. Modeluje wyłącznie wcześniejsze uwierzytelnienie
zewnętrznego manifestu oraz metadane root; bajty, hashe, deskryptory i ładowanie
są rzeczywiste. Nie jest dowodem podpisanego runtime ani VM. To sprawdzenie
dynamicznego wykonania uzupełnia AST i dokładny inwentarz.

OpenSSL dodatkowo odmawia hardlinka, niewykonywalnego binarium, niewłaściwego
typu przodka lub końcowego obiektu; testy awarii memfd/write/seek/seal sprawdzają
zamknięcie wszystkich deskryptorów i brak wykonania procesu. Kontrakt
konfiguracji i jedynego providera pozostaje zamknięty. Eksport Git wyłącza
lazy fetch i wszystkie transporty, aby brak obiektu partial clone nie uruchomił
repozytoryjnego remote helpera. Dane testów Git nie korzystają z globalnego
configu i mają wyłączone podpisywanie commitów.

**Status: PASS lokalnych regresji.** Pochodzenie loadera/libc/OpenSSL i innych
zależności, 1352 wpisy/573 `.pyc`/19 plików, uprzywilejowane zmiany root/kernel,
rzeczywista instalacja i VM pozostają BLOCKED. Loader nie udaje ogólnego
sandboxu dla już skompromitowanego procesu ani kompletnego API zasobów każdego
możliwego przyszłego pakietu.

## Pliki, linie i faktycznie wykonane testy

Dokładny wykaz hunków i SHA-256 zmienionych źródeł znajduje się w
[evidence/stage9-pr3265-review/changed-lines.json](evidence/stage9-pr3265-review/changed-lines.json).
Obejmuje kod, testy, CLI, workflow i procedury względem commita poddanego review.

- Rewizje: `freshness_application.py`, `production_local_installer.py`,
  `test_stage9_revision_contract.py`, `stage9_revision_migration.md`.
- INSPECT: `stage9_vm_trial.py`, `STAGE9_LINUX_VM_HARNESS.md`,
  `stage9_protected_execution_tree.md`, `test_stage9_inspect_contract.py`.
- Dynamiczny loader i odmowy: `stage9_vm_trial.py`,
  `test_stage9_harness_import_security.py`, `test_stage9_loader_execution.py`.
- OpenSSL: `freshness_runtime_bootstrap.py`,
  `test_freshness_runtime_bootstrap_security.py`.
- Git i inwentarz: `prepare_stage9_protected_tree.py`,
  `test_stage9_protected_tree.py`, `test_prepare_stage9_vm_bundle.py`.
- CI: `platform-deployment.yml`, `run_stage9_safe_validation.py`,
  `test_stage9_safe_ci.py`, `pyproject.toml`, `quality-security-toolchain.md`.

Przed zamrożeniem commita wykonano **484 dozwolone testy na Python 3.11.17**
i **484 na Python 3.12.14**, każdy zestaw exit **0**, bez SKIP. Są to wyniki
lokalnego drzewa development. Logi, JUnit i metadane argv zachowano
lokalnie w katalogu roboczym; nie są eksportowane z tego środowiska do repozytorium. Nie zastępują wykonania dokładnego finalnego HEAD
w CI. Po zamrożeniu uruchamiany jest ten sam runner z pełnymi statycznymi
bramkami, a wyniki są zachowane w lokalnym katalogu roboczym i artefaktach
GitHub Actions. Nie zmieniano wcześniejszych dowodów Stage 9.

Gotowość do ponownego review: kod, procedury i dozwolona ścieżka CI przygotowane;
ostateczna ocena zależy od faktycznego wyniku macierzy dla HEAD wskazanego w PR.
Gotowość do merge/Stage 9/produkcji nie jest deklarowana.
