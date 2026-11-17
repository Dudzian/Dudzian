# PR #3282 — poprawki historii CI i klasyfikacji AST

Przejrzana baza: `07e1d2a3f5e514b3148b41eef413ba9f49c1a34c`.
Zmiany pozostają na istniejącej gałęzi `codex/stage9-v5d4-observer-env`.

## P1: pełna historia tylko w jobie V5d4

Run `44073734720`, job `134006237555` wykonywał checkout z `fetch-depth: 1`.
Fixture V5d3 potrzebuje historycznego commita
`17e03a9cb42a84302401415f96ac90e756926915`; brak obiektu powodował 117 błędów.

Checkout `stage9-v5d4-static-environment` ma teraz `fetch-depth: 0`.
Przed pierwszym pytest uruchamiane jest:

```bash
git cat-file -e '17e03a9cb42a84302401415f96ac90e756926915^{commit}'
```

Bash kroku CI zachowuje odmowę przy niezerowym kodzie. Nie zmieniono
historycznego SHA, fixture V5d3 ani checkoutów innych jobów; zachowano wszystkie
wcześniejsze testy.
Nowy test używa rzeczywistego Git: płytki lokalny klon odmawia przed markerem
testów, a pełna historia udostępnia wymagany obiekt. Syntetyczne repozytorium
ma własny autentyczny commit; osobny test przypina niezmieniony SHA właściwego
kontraktu workflow. Porównanie YAML potwierdza brak zmian pozostałych jobów,
w tym fizycznych bramek Linux/Windows.

## P2: pochodzenie funkcji zamiast samej nazwy compile

Generator i niezależna kopia weryfikatora V5d3 rozpoznają `compile()` oraz
`builtins.compile()` i ich rzeczywiste aliasy. Wyjątek dotyczy wyłącznie
`re.compile` o ustalonym pochodzeniu z importu, także przez alias modułu,
import funkcji lub przypisanie. Nie jest to wykluczenie wszystkich `.compile`.
Niezwiązane `re.compile` pozostaje nierozstrzygnięte: sam napis `re` nie
dowodzi importu modułu biblioteki standardowej.

Przesłonięcia parametrów, ponowne przypisania, zasięgi funkcji/klas,
opcjonalne gałęzie i nieznana refleksja odbierają dowód pochodzenia regex.
Późniejsze jawne przepięcie globalnego aliasu nie ukrywa `compile` w ciele
odroczonej funkcji. `exec`, `eval`, `__import__`, importlib, runpy, ctypes
i wcześniejsze mechanizmy pozostają widoczne. Alias `execute = exec` nie
powoduje fałszywej klasyfikacji odrębnego `collector.execute(...)`.

Test porównuje AST obu kopii reguł. Analizator V5d4 ma analogiczne kontrole
pochodzenia i przesłonięć; raport rzeczywistych źródeł kontrolera nie zawiera
fałszywego `re.compile` ani `collector.execute`, zachowując odroczone importy
libvirt i zstandard. Uzupełniono analizę adnotacji wszystkich rodzajów
argumentów oraz parametrów typów funkcji/klas Pythona 3.12. Testy tych
konstrukcji analizują wyłącznie AST; nie wykonują badanego kodu.

Wynik pozostaje statycznym wskazaniem do review, a nie dowodem pełnego
dynamicznego domknięcia ani tożsamości już działającego procesu.

## Walidacja

Końcowe lokalne regresje obejmują V5d2 i kontrolę historii (65 testów),
V5d3 (255 testów) oraz pełne V5d4 (228 testów). Polecenia używają zaufanego
runtime narzędziowego, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` i odpowiedniego
`--confcutdir`. Testy nie uruchamiają badanego CPython ani VM.

Sprawdzane są pełny dotychczasowy zakres Ruff, dokładny hook
`python -m pre_commit run ruff-format --all-files`, wszystkie hooki
`python -m pre_commit run --all-files`, configured mypy Linux/Win32 bez
argumentów nazw plików, niezmienione blokujące reguły Semgrep oraz pełny
working tree Betterleaks z dotychczasową bazą. Dotychczasowe progi i
wymagania ABI, podpisów, zależności oraz izolacji pozostają bez zmian.
Wszystkie te kontrole zakończyły się kodem 0; łącznie przeszło 548 testów.

Nowy run GitHub Actions musi potwierdzić sukces joba V5d4 dla opublikowanego
SHA. Bieżący opis PR zawiera identyfikatory przebiegów i stan pozostałych
workflow. Lokalnego wyniku nie należy utożsamiać z wynikiem Actions ani
gotowością fizycznego środowiska. Nie wykonywać merge automatycznie.

```text
Stage 9 = IN_PROGRESS
Observer 3A = BLOCKED
Stage 3B = NOT AUTHORIZED
Production = NO-GO
```
