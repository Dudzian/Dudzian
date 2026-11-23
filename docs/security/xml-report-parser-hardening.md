# Odmowa DTD w raportach XML

Powiązanie: [issue #3280](https://github.com/Dudzian/Dudzian/issues/3280).
Baza: `b143b570a8159426020ddaac8c31d74059667c70`.
Commit poprawki: `b2e7a99df028d26ae22340f81f70ea5ff8a1a37f`.

## Potwierdzony problem i poprawka

`scripts/enforce_coverage.py` oraz `deployment/core_test_plan.py` odmawiały
deklaracji XML na podstawie wyszukiwania bajtowego przed rozpoznaniem kodowania.
Regresje potwierdziły, że nie realizowało to deklarowanej odmowy wszystkich DTD.
Status poprawki w tej gałęzi: `CONFIRMED_FIXED`; stan w bazowym main: `MISSING`.

Obie ścieżki używają teraz osobnego `XMLParser` z `TreeBuilder`, którego callback
`doctype` zawsze zgłasza dotychczasowy błąd domenowy. Odmowa następuje po
rozpoznaniu kodowania przez parser, przed odczytaniem deklaracji encji i ich
ekspansją. Obsługiwane zwykłe dokumenty UTF-8 i UTF-16 pozostają poprawne.
Nieznane lub nieobsługiwane deklaracje kodowania kończą się kontrolowaną odmową.

Odczyt jest ograniczony do 16 MiB + jednego bajtu kontrolnego; ocena rozmiaru
dotyczy faktycznie odczytanych danych. Wcześniejszy oddzielny `stat` nie stanowi
już podstawy decyzji. Nie dodano zależności ani wyjątków skanera; zachowane
wąskie, istniejące adnotacje Semgrep dotyczą parsera z bezwarunkową odmową DTD.

## Dowody i wpływ

Na niezmienionych dwóch plikach produkcyjnych z bazowego SHA nowe regresje
zakończyły się wynikiem **42 failed, 59 passed**, exit 1. Po poprawce ten sam
zestaw zakończył się wynikiem **101 passed**, exit 0. Wyniki pochodzą z bieżącej
sesji Linux/CPython 3.12.14, nie z poprzedniego środowiska.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  tests/test_enforce_coverage_script.py tests/deployment/test_core_test_plan.py -q
python -m ruff check scripts/enforce_coverage.py deployment/core_test_plan.py \
  tests/test_enforce_coverage_script.py tests/deployment/test_core_test_plan.py
python -m ruff format --check scripts/enforce_coverage.py \
  tests/test_enforce_coverage_script.py tests/deployment/test_core_test_plan.py
git diff --check
```

Trzy ostatnie kontrole zakończyły się kodem 0. Ponowne uruchomienie 101 testów
z branch coverage na commicie poprawki zakończyło się kodem 0; `coverage xml`
oraz `diff-cover --compare-branch=origin/main --fail-under=70` również zwróciły
0. Pokrycie zmienionego kodu wyniosło **100% (17 linii)**. Nie zmieniono progu.
Odrębna regresja sąsiednich ścieżek zakończyła się wynikiem **393 passed**, exit 0:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest --noconftest \
  tests/scripts/test_s9_fv_01_junit.py \
  tests/deployment/test_stage9_observer_kvm_domain_v5d1.py \
  tests/scripts/test_platform_core_evidence_workflow.py -q
```

Dokładny hook `python -m pre_commit run ruff-format --all-files` zakończył się
kodem 0, bez zmian plików, z odrębnym cache narzędzi w bieżącym środowisku.
Skonfigurowany mypy Linux/Win32 i wymagane CI muszą mieć odrębny aktualny dowód;
lokalna regresja nie zastępuje ich wyniku.

Przypadki obejmują DTD bez encji, encje wewnętrzne i zewnętrzne, odwołania
wewnętrzne w atrybutach, UTF-8/BOM i UTF-16/BOM/LE/BE, błędne deklaracje kodowania,
niekompletne XML oraz rozmiar danych różny od wcześniejszych metadanych pliku.
Test planu core wykonuje rzeczywistą ścieżkę oceny raportu i sprawdza brak
markera po odmowie. Nie uruchamia wymaganych pakietów testowych jako części
próby fizycznej: wstrzyknięty runner zapisuje jedynie syntetyczny raport.

Zmiana dotyczy wyłącznie parserów raportów i ich granicy odczytu. Nie modyfikuje
progów pokrycia, planu wymaganych testów, podpisów, risk gate ani kontroli handlu.
Zachowane są statusy:

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```

## Granice audytu

Parser domeny V5d1 już używa `defusedxml` z jawną odmową DTD, encji i zasobów
zewnętrznych oraz limitami rozmiaru/liczby/głębokości elementów; nie wymaga tej
samej poprawki. Bramka S9-FV-01 ogranicza odczyt i dekoduje UTF-8 przed istniejącą
odmową DTD. Nie jest to deklaracja pełnego audytu wszystkich konsumentów XML
w narzędziach, testach, archiwum ani zależnościach. Kwalifikacja natywna
Windows/Linux i gotowość produkcji pozostają poza zakresem tej poprawki.


## Uzupełnienie granicy JUnit

Ten sam callback odmowy DTD obejmuje teraz dwa sąsiednie czytniki JUnit:
`check_s9_fv_01_junit.validate_report` i `run_stage9_safe_validation.junit_result`.
Odmowa nie zależy od wyszukiwania tekstu przed rozpoznaniem kodowania. Pomocniczy
czytnik bezpiecznego zakresu również ogranicza rzeczywisty odczyt do 16 MiB + 1
bajtu i zwraca FAIL przy błędzie odczytu, niepoprawnym dokumencie lub kodowaniu.
Zachowano dotychczasowe przyjmowanie zwykłych raportów UTF-8 oraz obsługiwanych
raportów UTF-16; nie wprowadzono globalnego zakazu poprawnego UTF-16. Brak raportu
w pomocniczym czytniku nadal oznacza NOT_EXECUTED. Inwentarze przypadków,
semantyka wyników, progi i status produkcyjny pozostają bez zmian.
