# Stage 9 V5d5 — walidacja

Baza main: `c6dbd5acaa87edd5c6b823bfb013764e5c5be6a8`. Gałąź:
`stage9/v5d5-shared-wheel-members`. Walidacja lokalna: 18 listopada 2026,
nieuprzywilejowany Cloud Linux, CPython 3.12.14. Nie jest to kwalifikacja
docelowego Ubuntu 24.04/CPython 3.12.3 ani uruchomienie kontrolera.

[Architektura i przegląd źródeł](stage9_v5d5_shared_wheel_members.md) oraz
[pełne 62 kolizje](evidence/stage9_v5d5_shared_wheel_inspection.json)
oddzielają zmierzone bajty od niezależnej aprobaty, której nie udzielono.

| Zakres | Wynik lokalny |
| --- | --- |
| V5d5 | 95 passed: 21 dependencies/policy, 14 materializacja, 60 niezależne próby negatywne i receipt. |
| Pełne V5d4 po ostatniej poprawce writera | 379 passed, 22 skipped. Razem z V5d5: 474 passed, 22 skipped. |
| V5d3 | 255 passed we wspólnym przebiegu V5d5/V5d4/V5d3; ten przebieg: 728 passed, 22 skipped. V5d5/V5d4 ponowiono po ostatnim wzmocnieniu kontroli tożsamości katalogu. |
| Wcześniejsze Observer V1–V5d1 | 2517 passed, 32 skipped. |
| V5d2 i wcześniejszy inventory CI/history | Czysty pełny przebieg: 67 passed, w tym wszystkie 63 przypadki V5d2. |
| Końcowy inventory CI/history i agregacja pokrycia | 13 passed; obejmuje 11 końcowych przypadków CI i dwa historyczne. Rzeczywiste SQLite coverage: ścieżki Windows oraz odmowy brakujących, pustych, dodatkowych, niepełnych, bezgałęziowych danych i nieudanej macierzy. |
| Coverage sześciu narzędzi | 89% według coverage.py przy włączonym pomiarze gałęzi; shared 98%, environment 94%. |
| Pokrycie zmienionego kodu | 91%: 254 mierzone wiersze, 22 niepokryte. Niezmieniony próg 70% PASS. |
| Ruff 0.14.11 | Pełny dotychczasowy zakres PASS. |
| Format | Dokładny `python -m pre_commit run ruff-format --all-files` po dodaniu nowych plików do indeksu Git: PASS, bez zmian formattera. |
| Configured mypy 1.10.0 | Linux i Win32: 425 plików, zero błędów; bez argumentów nazw plików ograniczających zakres. |
| Pre-commit | Pełne `python -m pre_commit run --all-files`: Ruff, format, configured mypy i staged Betterleaks PASS. |
| Semgrep CE 1.179.0 — blokujące reguły | Pełny dotychczasowy zakres: zero ustaleń. Pierwotny timeout `bot_core/app.py` domknięto odrębnym skanem z limitem 60 s. Końcowy skan czterech zmienionych skryptów: zero ustaleń i luk analizy. Fixtures 5/5 PASS. |
| Semgrep — dotychczasowy advisory | 25 ustaleń: 20 WARNING, 5 ERROR; 18 timeoutów oraz jedna luka parsowania C, wszystkie poza zmienionymi skryptami. Exit 0 jest wynikiem istniejącego trybu advisory, nie dowodem czystego pełnego skanu. |
| Betterleaks 1.9.0 | Pełny working tree z dotychczasową bazą, bez walidacji u providerów i z redakcją treści: PASS. Usunięto wyłącznie nieśledzone cache `__pycache__`, które powielały istniejące testowe materiały kryptograficzne. |
| pip-audit 2.10.1 | Kanoniczny `deploy/packaging/requirements-desktop.lock`: 104 zależności, zero podatności, bez wyjątków i automatycznych poprawek. |

22 pominięcia V5d4 na Linux dotyczą rzeczywistych uchwytów/reparse points
Windows oraz odmów hosta Windows/macOS. Zestaw wcześniejszych etapów ma
31 pominięć wymagających preprovisioned `gpgv` i jedno istniejące pominięcie
natywnej odmowy na innym hoście. Nie zmieniono warunków pomijania ani limitów.
Dwa wstępne, przerwane przebiegi inventory V5d2 napotkały timeouty przy
równoległym obciążeniu CPU; pełny końcowy przebieg wykonano osobno, zachowując
oryginalne 20 s na wywołanie fixture.

GitHub Actions ma osobną macierz Ubuntu/Windows/macOS. Każda platforma
wykonuje odpowiednie kontrole statyczne i odmowy; build i launcher pozostają
Linux-only. Osobna bramka wymaga sukcesu całej macierzy i dokładnie trzech
pełnych danych branch coverage, łączy je i wymaga 70%. Dotychczasowe joby,
progi i ręczne bramki sprzętowe pozostają bez zmian. Rzeczywisty stan Actions
dla końcowego HEAD jest podany w draft PR; lokalne modele nie są wynikiem
uruchomienia na Windows lub macOS.

[Dowody walidacji](evidence/stage9_v5d5_validation/checks.json) zawierają
końcowe exit codes, dokładne argv, JUnit, coverage, raporty skanerów oraz
hashe zmienionych źródeł. Surowe strumienie i raporty zachowano w bezstratnym
gzip; `RAW_SHA256SUMS` dotyczy bajtów po dekompresji, `SHA256SUMS` plików
zapisanych w repozytorium. Te hashe nie są niezależnym approval ani podpisem.

Inspekcja rzeczywistych wheels objęła dokładnie Essentials i Addons 6.10.3:
3623 oryginalne pliki, 3621 kontrolowanych hashów/rozmiarów RECORD, dokładnie
62 kolizje i 124 deklaracje właścicieli. Wszystkie dane są zgodne z V5d4;
odwrócenie kolejności wheels daje identyczny wykaz, brak approval odmawia.
Nie instalowano ani nie wykonywano zawartości tych archiwów. Pozostałych
67 wheels nie pobrano ponownie i nie uznano za sprawdzone.

Testy materializacji używają wyłącznie jawnych TEST ONLY oczekiwań i
syntetycznych bajtów; granice hosta/ABI są zastąpione, rzeczywiste odczyty,
weryfikacja RECORD, planowanie i zapis pozostają aktywne. Testy launchera
zastępują import natywny rejestratorem. Dotychczasowe kryptograficzne fixtures
regresji nie są podpisami operatora ani aprobatą badanych pakietów.
Nie używano sudo, VM, instalacji do `/opt`/`/etc` ani nie wykonywano merge.

Pozostają niezależne oczekiwania dla 62 członków i osobne referencje review
trzech źródeł, aprobaty dwóch pominięć bytecode, pełnego wheelhouse,
źródeł/native/ABI i bazowego runtime. Chroniony launch, immutable view,
kwalifikacja hosta i odrębna autoryzacja eksperymentu nadal są wymagane.
Codebase Memory nie było dostępne; ustalenia zweryfikowano w źródłach i testach.

```text
STAGE_9=IN_PROGRESS
OBSERVER_3A=BLOCKED
STAGE_3B=NOT_AUTHORIZED
PRODUCTION=NO-GO
```
