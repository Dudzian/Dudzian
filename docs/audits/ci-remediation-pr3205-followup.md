# PR #3205 — kontynuacja remediacji regresji CI

## 1. Stan i zakres

Kontynuacja tej samej gałęzi `codex/ci-remediation-warning-audit` i tego samego
[PR #3205](https://github.com/Dudzian/Dudzian/pull/3205). Stan wyjściowy:
`8581f5fca84e81f474b77c59394d38010171e86e`. Dokładny nowy SHA i przypisane mu
przebiegi są podawane w aktualizowanym opisie PR; wyniki starego commita nie
stają się dowodem dla nowego przez samą zmianę kodu.

Poprzednia walidacja nie obejmowała rzeczywistego pełnego HSR, kompletnego
kontraktu argumentów General CI ani Workbench w kontekście poprzednich testów
Qt. To były luki walidacji. Poniższe poprawki dotyczą osobnych przyczyn;
naprawienie samego historycznego API Workbench nie wystarczało.

## 2. Problemy niezwiązane z kontraktem Workbench

| Problem | Potwierdzona przyczyna | Poprawka |
| --- | --- | --- |
| Ubuntu i trzy shardy PostgreSQL zatrzymane podczas kolekcji | Moduł Workbench ładował QtWidgets/EGL przed selekcją markerów; usunięcie globalnego warunku Playwright odsłoniło niepoprawną zależność kolekcji od bibliotek grafiki | Kolekcja modułu nie importuje Qt; wymagany przypadek natywny importuje je dopiero w swoim procesie. Helper odróżnia brak opcjonalnej zależności od rzeczywistego błędu importu/ABI |
| Windows/macOS: kontrakt pytest odrzuca `-ra` | Stare asercje prefiksów nie dopuszczały diagnostyki; kolejne, wcześniej nieosiągnięte asercje oczekiwały też wildcard portu i starych prefiksów licensing/native-peer | Pełny argv z rzeczywistego Bash jest porównywany po usunięciu wyłącznie jednego dozwolonego początkowego `-ra`; wymagane localhost i wszystkie selektory pozostają ścisłe |
| History Successor Reproposal: 76 setup ERROR | Konstruktor HSR otwierał autocommit bez transakcji; łańcuch HSR → custody → Atomic trafiał na prawidłowy guard IDLE | Jedna jawna transakcja konstruktora; guard pozostaje. Administracyjne admission już używało BEGIN SERIALIZABLE i nie wymagało zmiany |

HSR odtworzono na PostgreSQL 16.15: przed poprawką rzeczywiste połączenie miało
IDLE, po poprawce INTRANS. Sanitizujący wyjątek konstruktora ukrywał przyczynę
przez `from None`, stąd jednakowe zewnętrzne błędy fixture. Nie zmieniono tej
granicy ani nie zastąpiono odmowy PASS. 76 błędów jest kaskadą jednej przyczyny.

Wcześniejsze stwierdzenie, że wszystkie ścieżki HSR mają transakcję, było błędne.
Nowy audyt AST 2510 śledzonych plików Python znalazł 46 wywołań, z czego 25 poza
tests/archive. Wszystkie 25 oraz opakowujące metody sprawdzono w źródłach i
niezależnym przeglądzie. Po poprawce w tych zbadanych ścieżkach nie znaleziono
kolejnego autocommit-IDLE. To nie jest dowód kompletności dowolnych dynamicznych
wywołań Pythona. Codebase Memory nie był dostępny.

## 3. Workbench: crash, aktualne API i warningi

### SIGSEGV jest odrębny od libEGL i starych asercji

CI kończyło się kodem 139 podczas `QQmlApplicationEngine.load`, zanim test
dochodził do historycznych pól. Lokalnie odtworzono kod 139 w tym samym układzie
poprzedzających testów, podczas globalnego flush `DeferredDelete`. Proces miał
poprawny QApplication; nie przypisujemy awarii CI wyłącznie QCoreApplication.
Osobno potwierdzono, że zastany QCoreApplication stanowi kolejny niepoprawny
kontekst dla tego testu GUI.

Natywne scenariusze Workbench wykonują się w świeżym procesie tego samego
interpretera, z własnym QApplication i engine o określonej własności/lifetime.
Nie dziedziczą obiektów ani kolejki odroczonych usunięć wcześniejszych testów.
Rodzic zachowuje stdout/stderr i odmawia PASS dla sygnału, niezerowego kodu,
assertion failure lub timeoutu. Wymagany roundtrip nie może stać się SKIP.
Opcjonalny browser mirror ma osobny, jawny protokół pominięcia.

Szerszy zestaw ujawnił ten sam problem we wspólnym teardownie bez Workbench,
w późniejszych testach TicketDialog. Zwykły teardown silnika wykonywał globalny
`sendPostedEvents(None, DeferredDelete)`, obejmując też sceny pozostawione przez
celowo nieagresywny teardown hostowanych widoków. Teraz jawnie usuwa tylko własne
rooty przed cache/GC, a następnie własny engine. `None` jest odrzucane przed
wywołaniem Qt. Rzeczywisty test obiektów dowodzi zniszczenia własnego roota i
engine oraz zachowania obcego oczekującego QObject do jego własnego cleanupu.
Nie zmieniono jawnych globalnych API ani warunków conftest. `processEvents()`
pozostaje globalne: callback lub zagnieżdżona pętla nadal może uczynić obce
usunięcia wykonywalnymi. Nie jest to obietnica pełnej izolacji eventów.

Timeout procesu wynosi 60 s, z marginesem 75 s rodzica na start i cleanup;
nie ma retry. Przekroczenie limitu zabija i odbiera bezpośredni proces potomny,
a następnie kończy test błędem. Regresja używa rzeczywistego SIGSEGV na POSIX;
wariant Windows wyłącza dialogi CRT/WER wyłącznie w swoim procesie testowym.

Nie uzyskano natywnego backtrace C++ identyfikującego konkretny uszkodzony obiekt.
Dowód obejmuje odtworzony kontekst awarii, korektę lifetime przez izolację oraz
regresje propagacji awarii. Nie przedstawiamy tej granicy dowodowej jako pełnej
diagnozy wewnętrznego błędu Qt.

### Aktualny kontrakt

Commit `dc730e78` wprowadzał instrumentDetails/controlState i mutatory schedulera.
Merge `d16d7b8f` usunął tę implementację oraz odpowiadające jej asercje C++,
pozostawiając nieaktualny test Python. Obecny monolityczny widok używa siedmiu
agregatów: schedulerEntries, exchangeConnections, aiConfiguration,
portfolioSummary, riskSnapshot, runtimeStatus i licenseStatus.

Test sprawdza aktualne oba presety, ponowne przełączanie, rzeczywiste zwracane
QVariant, odrzucenie pustego/nieznanego presetu bez mutacji, ochronę demo przed
refreshFromLive oraz wyczyszczenie danych/ryzyka/licencji po disable i jego
idempotencję. Nie przywraca nieużywanych metod ani sterowania rzeczywistym
schedulerem. Brak Playwright nie pomija wymaganego scenariusza Qt.

### Dodatkowe źródła ostrzeżeń

Przechodzący scenariusz ujawnił istniejące wcześniej warningi: pętlę slidera
AutoModePanel, undefined-to-bool i odczyt count z odłączonego ListView. Nie są
filtrowane. Zmiany obejmują jawne odczyty istniejących statusów kontrolera
katalogu, ochronę odłączonego widoku, bool dla opcjonalnych błędów licencji oraz
aktualizację preferencji slidera wyłącznie po interakcji `onMoved`. Nie powstaje
fikcyjny status licencji ani nowe prawo do operacji tradingowych.

## 4. Zmienione pliki i powody

- `bot_core/postgresql_history_successor_reproposal.py`: transakcja konstruktora.
- Dwa testy PostgreSQL qualification/HSR: rzeczywiste INTRANS, direct role,
  pg_catalog, zachowana odmowa IDLE i brak SQL admission po odmowie admina.
- `tests/scripts/test_general_ci_platform_contract.py`: pełny kontrakt argv,
  przypadki pozytywne/negatywne i zachowanie podziału testów.
- `tests/ui/_qt.py` i `tests/test_qt_dependency_requirements.py`: opcjonalność
  zależności tylko poza wymaganą kwalifikacją QML; rzeczywista kolekcja procesu.
- `tests/ui/test_strategy_workbench_demo.py` i nowy kontrakt procesu:
  aktualne API, natywny lifetime, obowiązkowe wykonanie i propagacja awarii.
- `tests/_qt_utils.py` i regresje helpera: własność jawnego flushu DeferredDelete,
  rzeczywiste zniszczenie własnych obiektów, zachowanie obcego oczekującego obiektu
  oraz odmowa globalnego odbiorcy `None`.
- Workbench QML/ViewModel i AutoModePanel: potwierdzone źródła warningów,
  kontrola statusów i interakcji, bez dodawania operacji schedulera.
- `.github/workflows/ci.yml`: tylko `-ra` w trzech pozostałych poleceniach
  Release Quality, aby nowe logi pokazywały powody historycznych skipów.
  Porównanie sparsowanego YAML/argv potwierdza identyczne selektory i kroki.

## 5. Regresje lokalne

| Zakres | Wynik |
| --- | --- |
| HSR przed poprawką: rzeczywisty łańcuch kwalifikacji | 1 FAIL, 1 PASS — IDLE odtworzone |
| Kwalifikacja Atomic/owner/guard po poprawce | 247 PASS, bez skipów i warningów |
| Pełny genuine HSR fresh/legacy/evidence/architecture | 106 PASS, 0 ERROR/FAIL/SKIP, 9819.72 s |
| Pełny kontrakt General CI | 51 PASS, 22.87 s |
| Kontrakt: shell pozytywne/negatywne | 6 pozytywnych, 17 negatywnych; 64 identyczne syntetyczne node IDs w rozłącznych shardach z/bez -ra |
| Zależności Qt: kolekcja i required-QML | 23 PASS; brak EGL nie importuje Qt w zwykłej kolekcji, wymagany scenariusz daje JUnit failure, bez SKIP |
| Workbench w poprzednio crashującym kontekście | 5 PASS, 1 SKIP Chromium; wcześniej native exit 139 |
| TicketDialog po czterech poprzedzających testach | 6 PASS, rzeczywisty exit 0; wcześniej native exit 139 |
| Pełne `tests/ui` QML i dwa nowe natywne przypadki helpera/procesu | 56 PASS, 1 SKIP Chromium, 102 deselected, 37.09 s, rzeczywisty exit 0 po podsumowaniu |
| Końcowy Workbench / kontrakt procesu / zależności Qt / helper, po obu poprawkach przeglądu | 47 PASS, 1 SKIP Chromium, 15.43 s, rzeczywisty exit 0 |
| Bootstrap aplikacji i kreator trybu po zmianie helpera | 4 PASS, 5.44 s, rzeczywisty exit 0, stderr 0 B |
| Zachowane AnyIO/network/cache/trading/catalog i zależności Qt | 111 PASS, 25.31 s; RuntimeWarning traktowany jako błąd |
| Mypy skonfigurowany Linux / --platform win32 | Oba PASS, po 347 plików |
| Ruff pełnego zakresu / dokładny hook ruff-format --all-files | PASS; końcowe wyniki przed pushem są odnotowane w opisie PR |

Powyższe zestawy częściowo się pokrywają i nie są sumą rozłącznego pokrycia.
Wymagany Workbench zachowuje stderr i ma zero warning/critical/fatal Qt,
włącznie z cleanupem; siedem wcześniejszych diagnostyk usunięto u źródła.
Lokalny Python 3.12 / Qt 6.7.2 nie zastępuje
wyników Windows/macOS ani runnera CI. Zatrzymane podczas kolekcji próby pełnego
repozytoryjnego `-m qml` z brakującymi zależnościami lokalnego venv nie są
zaliczone. Lokalne uzupełnianie deklarowanych zależności ograniczono do
izolowanego venv; nie zmieniono z tego powodu manifestów ani markerów. Wynik
ostatniej szerokiej próby jest raportowany w opisie PR, bez blokowania pushu
i bez zastępowania pełnego wymaganego QML CI.

## 6. Pełne wyniki CI dla bazowego 8581

Odczytano wszystkie 103 joby i pełne logi wszystkich 84 wykonanych: 66 SUCCESS,
18 FAILURE, 19 SKIPPED. Wszystkie przebiegi: attempt 1, dokładny SHA 8581.

| Run | Wynik | Istotne ograniczenie |
| --- | --- | --- |
| [CI push 40113181935](https://github.com/Dudzian/Dudzian/actions/runs/40113181935) | 26 SUCCESS / 7 FAIL / 7 SKIP | Ubuntu/PG nie wykonały testów po błędzie kolekcji; Windows/macOS zatrzymały się na kontrakcie |
| [CI PR 40113328209](https://github.com/Dudzian/Dudzian/actions/runs/40113328209) | 26 SUCCESS / 7 FAIL / 7 SKIP | Linux QML SIGSEGV; Windows/macOS QML nie uruchomiono przez zależności jobów |
| [Quality 40113328302](https://github.com/Dudzian/Dudzian/actions/runs/40113328302) | 6 SUCCESS / 1 FAIL / 2 SKIP | HSR 76 setup ERROR; Atomic 4040 PASS / 7 SKIP; wspólny Atomic/HSR coverage pominięty |
| [CodeQL 40113328292](https://github.com/Dudzian/Dudzian/actions/runs/40113328292) | Python i JavaScript SUCCESS | Wynik wyłącznie starego SHA |
| [Platform 40113328254](https://github.com/Dudzian/Dudzian/actions/runs/40113328254) | 6 SUCCESS / 3 FAIL / 3 SKIP | Oczekiwane odmowy readiness i niewykonane Stage 10 |

4040 PASS Atomic jest rzeczywistym wynikiem bazowego SHA; źródła Atomic/custody
nie zmieniają się w tej kontynuacji. Siedem skipów dotyczy negatywnego przypadku
innej platformy i sześciu testów wymagających osobnego root harness Linux.
Nie wymuszano tych granic ani nie przedstawiano ich jako zakwalifikowanych.

## 7. Warning audit i pominięcia

W 84 logach bazowego SHA nie ma payloadów PytestCollectionWarning, tradingowego
Mean-of-empty-slice, ignorowanych wyjątków portalu ani nieskutecznego SET LOCAL.
To nie dowodzi czystości niewykonanych części głównych testów.

Pozostają 58 wiadomości Node DEP0005 i 20 build-time warningów opcjonalnych
hooków PyInstaller. Mniejsza liczba Node w pojedynczym CI (26 zamiast 29) wynika
z niewykonanych jobów, a nie naprawy. Wszystkie 60 par digestów artefaktów pasuje.
Źródła opcjonalnych warningów: PyInstaller 6.5.0 hardkoduje opcjonalne DBAPI;
hooks-contrib 2025.10 próbuje zebrać nieobowiązkowy sklearn; DLL discovery
importuje eksperymentalne NumPy array_api. To odrębne od błędu tradingowego.
Nie zmieniono pinów pakowania ani bramki wymaganych importów i build-smoke.

Opcjonalna nieobecność przeglądarki nie może ukryć warningu Qt zebranego w tym
samym scenariuszu: diagnostyka ma pierwszeństwo przed SKIP. Osobna regresja
potwierdza tę granicę. Kontrolowany Windows abort wymaga również znacznika
osiągnięcia samej operacji awarii, aby błąd Pythona w przygotowaniu procesu nie
mógł udawać prawidłowo wstrzykniętej awarii natywnej.

Oczekiwane WAL Database/DROP po kontrolowanym SIGKILL pozostają jawne. Testy
sprawdzają recovery, pełny write set i dokładny retry; lokalny cleanup potwierdza
brak dodatkowych baz i schematów. Marketing parity SUCCESS nadal oznacza raport
SKIPPED przy braku evidence/mirror, nie dowód parity PASS.

## 8. Pozostałe blokady

- Nowy HEAD musi wykonać pełne zwykłe shardy, Linux/Windows/macOS QML, HSR,
  Atomic/HSR combined coverage, Quality Ratchet, CodeQL i dependency audit.
- Opcjonalna przeglądarka nie jest dostępna lokalnie. Nie stanowi to SKIP
  obowiązkowego Qt; nie deklarujemy browser mirror jako PASS.
- Oryginalne bajty historycznych pip-audit/JUnit pozostają niedostępne przez
  odmowę hosta File Service. Lokalne reprodukcje i metadane nie są ich odczytem.
- Upstream Node/optional packaging hook diagnostics pozostają widoczne,
  sklasyfikowane i nie są rozwiązywane przez rozszerzanie runtime zależności.
- Dokładny obiekt C++ powodujący stary crash nie został wskazany native stackiem;
  korektę lifetime weryfikuje kontekstowy test i propagacja native failures.

## 9. Bezpieczeństwo

Zachowano guard Atomic, direct-login roles, zakres SET LOCAL, SERIALIZABLE i A+C.
Bez zmian frozen contracts, trust pins, produkcyjnych grantów, CodeQL i progów
Quality Ratchet. Stage 9 IN_PROGRESS / NO-GO; private control BLOCKED,
production provisioning false, Root Proof issuance unavailable, produkcyjne
SIGN/BIND BLOCKED, Stage 10 NO-GO. Nie wykonywano hardware Stage 41B,
provisioningu, enrollmentu, aktywacji ani Daybreak/live trading.

## 10. Gotowość PR

**NO-GO dla merge**, dopóki bieżący commit nie ma wyników wymaganych kontroli
i przeglądu warningów/pominięć. PR pozostaje draft. Nie tworzono nowego PR
i nie wykonano merge. Wyniki oraz linki nowego SHA są aktualizowane w opisie PR.
