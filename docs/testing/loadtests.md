# Testy obciążeniowe warstwy integracji giełd

Nowy moduł *exchange stress* pozwala szybko zasymulować ruch do mockowanych
endpointów giełdowych z użyciem `AsyncIOTaskQueue`. Narzędzie składa się z:

- pliku konfiguracyjnego scenariusza `config/loadtests/exchange_stress.yml`,
- skryptu CLI `scripts/loadtests/exchange_stress.py`,
- testów regresyjnych/benchmarków w `tests/load/test_exchange_stress.py`.

## Uruchomienie skryptu

```bash
python scripts/loadtests/exchange_stress.py \
    --config config/loadtests/exchange_stress.yml \
    --seed 1337
```

Domyślnie wyniki zapisywane są do `logs/loadtests/` w pliku
`exchange_stress-<timestamp>.json`. Ścieżkę można nadpisać parametrem
`--output`. Na standardowe wyjście trafi równocześnie podsumowanie JSON, dzięki
czemu łatwo zasilić np. narzędzia CI.

## Struktura konfiguracji

Sekcja `defaults` definiuje parametry kolejki (maksymalna współbieżność, limit
burst) oraz charakterystykę opóźnień i błędów. Każdy element tablicy `exchanges`
może nadpisać dowolne z tych pól. Minimalny przykład:

```yaml
defaults:
  max_concurrency: 8
  burst: 16
  request_count: 60
  base_latency_ms: 35.0
  jitter_ms: 8.0
  error_rate: 0.02
  throttle_rate: 0.05

exchanges:
  - name: binance_spot
    request_count: 90
    throttle_rate: 0.07
  - name: kraken_spot
    max_concurrency: 4
    burst: 8
    error_rate: 0.03
```

## Metryki i raportowanie

Każdy scenariusz zbiera liczby żądań, błędów oraz zdarzeń throttlingu.
Dodatkowo liczone są podstawowe statystyki latencji (`min`, `max`, `avg`,
`p95`). Raport JSON może być bezpośrednio archiwizowany w pipeline’ach lub
analizowany off-line.

## Testy i benchmarki

W katalogu `tests/load/test_exchange_stress.py` znajdują się testy korzystające
z `pytest` oraz `pytest-asyncio`. Pokrywają one:

- walidację parsera konfiguracji,
- deterministyczne scenariusze sukcesów i throttlingu,
- zapis wyników do formatu JSON.

Uruchomienie samych testów obciążeniowych:

```bash
pytest tests/load/test_exchange_stress.py
```

## Integracja z CI

Automatyczne uruchomienia obsługuje workflow
`.github/workflows/exchange-stress.yml`. Dostępne są dwa wyzwalacze:

- `workflow_dispatch` do ręcznego odpalenia,
- harmonogram `cron: "0 3 * * 1"`, czyli każdy poniedziałek o 03:00 UTC.

Workflow składa się z pojedynczego joba, który:

1. uruchamia skrypt z konfiguracją `config/loadtests/exchange_stress.yml`,
   zapisując surowe wyniki do `logs/loadtests/exchange_stress.json`,
2. waliduje próg SLA w `tests/load/test_exchange_stress.py` – krok
   `Validate exchange stress SLA` kończy cały job statusem failed, jeżeli
   zostaną przekroczone założone limity,
3. publikuje tabelę metryk w `exchange_stress_summary.md` oraz dopisuje ją do
   podsumowania `GITHUB_STEP_SUMMARY` (aby nie przepadła nawet w razie faila).

Publikacja metryk z joba `performance-benchmarks` wymaga dostępnego Pushgateway.
Adres instancji należy wstrzyknąć jako sekret lub zmienną
`PERFORMANCE_METRICS_PUSHGATEWAY`, a endpoint healthcheck (`/-/healthy`) musi być
osiągalny z runnera GitHub Actions przed startem testów.

### Tryb fallback dla publikacji metryk

Wykonania z forków (PR z zewnętrznego repozytorium) lub bez skonfigurowanego
Pushgateway wchodzą w **tryb fallback**. Zamiast przerywać job, workflow
wykonuje testy, zapisuje raporty JSON jako artefakty i dodaje ostrzeżenie do
`GITHUB_STEP_SUMMARY`, że metryki nie zostały wysłane do Pushgateway. Podsumowanie
zawsze wskazuje wybrany tryb (`strict` lub `fallback`) oraz konsekwencje dla
metryk, więc od razu widać, czy dashboardy Prometheus/Grafana dostaną dane z
danego uruchomienia.

W środowiskach wewnętrznych włącz **twardą walidację** (tryb `strict`) przez
zapewnienie dostępnego endpointu (sekret/zmienna `PERFORMANCE_METRICS_PUSHGATEWAY`
lub domyślny `CI_PUSHGATEWAY_DEFAULT`). Workflow sam wykryje poprawną
konfigurację i zablokuje job przy braku dostępu (`/-/healthy`) lub błędzie
publikacji – dzięki temu dashboards Prometheusa/Grafany zawsze dostaną dane.

Aby podejrzeć metryki z uruchomienia, otwórz konkretne wykonanie w Actions,
kliknij job „Exchange stress load test”, a w zakładce **Summary** przewiń do
sekcji `GITHUB_STEP_SUMMARY` – wyświetli się tam tabela z
`exchange_stress_summary.md`. Surowy raport JSON jest archiwizowany jako
artefakt `exchange-stress-report`; w widoku runu w Actions w sekcji
**Artifacts** znajdziesz do pobrania plik `logs/loadtests/exchange_stress.json`.

## SLA dla dashboardów i backtestów

Nowe testy wydajnościowe pokrywają dwa obszary:

- **render SLA/Risk w `RuntimeOverview.qml`** – p90 czasu renderu kart SLA pod
  sztucznym obciążeniem feedu nie może przekraczać **220 ms**, a średni czas
  renderu kart ryzyka przy gęstej osi czasu musi być niższy niż **180 ms**.
- **render paneli SLA/Risk na próbkach telemetrii** – p90 renderu na
  odtworzeniu historycznych alertów i timeline’u ryzyka nie może przekraczać
  **230 ms** (SLA) ani spaść poniżej budżetu regresji **10%** względem
  poprzednich runów.
- **backtest throughput** – minimalna przepustowość backtestów w CI to:
  - 2 pary @ 1m: **≥ 1.5 par/s**,
  - 4 pary @ 5m: **≥ 1.0 par/s**,
  - 6 par @ 1h: **≥ 0.6 par/s**.
  Dodatkowo benchmark per-strategy nie może spaść poniżej 10% względem SLA dla
  datasetów `mean_reversion`, `volatility_target` i `cross_exchange_arbitrage`
  (min. throughput 600–750 barów/s, p95 ≤ 5–6 ms).

### Progi alertowe i reakcja

- **Wzrost p90 renderu UI** – alert `PerformanceUIRenderP90Regression` włącza
  się, gdy `ci_performance_p90_ms{source="ui_render"}` rośnie o >25% względem
  średniej z 7 dni i jednocześnie przekracza 220 ms. Procedura:
  1. Otwórz artefakt `performance-benchmark-history` (Parquet/SQLite) z danego
     runu i sprawdź, które scenariusze `scenario` oraz `git_commit` są
     oznaczone najwyższymi wartościami.
  2. W Grafanie na panelu „UI render p90 (rolling)” zweryfikuj, czy trend jest
     chwilowy, czy utrzymuje się w oknach 24h/7d.
  3. Jeśli regresja pokrywa się z ostatnim SHA, wykonaj bisekcję zmian w QML
     (dashboard `RuntimeOverview.qml`) lub cofnij regresyjną zmianę animacji.

- **Spadek throughputu backtestów** – alert
  `PerformanceBacktestThroughputRegression` uruchamia się przy spadku
  `ci_performance_pairs_per_second` o >10% względem średniej 7 dni.
  Procedura:
  1. Sprawdź w artefakcie `performance-benchmark-history` rekordy z danym SHA
     i porównaj metryki `pairs_per_second`/`bars_per_second` do SLA.
  2. W panelu Grafany „Backtest throughput (rolling)” upewnij się, że problem
     nie wynika z pojedynczego datasetu (etykiety `pair_count`/`timeframe`).
  3. Jeśli degradacja dotyczy wszystkich scenariuszy, zdiagnozuj zmiany w
     silniku backtestów (`bot_core.backtest`) i przywróć poprzednią wersję lub
     ogranicz funkcje zwiększające koszt CPU.

Wyniki benchmarków backtestów są automatycznie zapisywane do
`reports/ci/performance_backtests/` jako JSON, dzięki czemu mogą być
zaciągane przez pipeline’y i Grafanę. Analogicznie, pomiary renderingu
paneli SLA i Risk w QML zapisujemy w `reports/ci/performance_ui_render/`
z polami `avg_ms`, `p90_ms` i `sla_ms` – ułatwia to wizualizację trendów i
alertowanie na odchylenia od progów SLA.
Benchmarki per-strategy uruchamiane przez `scripts/benchmark_backtesting.py`
zapisują wyniki do `reports/ci/benchmark_backtests/` wraz z p95 i budżetem
regresji, który musi być <10% względem wartości SLA.

## Raportowanie w CI i Grafanie

W CI zapisuj artefakty z katalogu `reports/ci/performance_backtests/` oraz
strumień STDOUT z testów wydajnościowych. Raporty JSON zawierają czasy
całkowite, przepustowość oraz liczbę transakcji na parę, co pozwala
automatycznie porównać je z progami SLA. Każdy plik zawiera również stemple
`timestamp_utc` oraz `git_commit`, co ułatwia korelację wyników z konkretną
wersją kodu. W Grafanie podpinaj pliki raportów jako źródło danych (np.
lokacje artefaktów z ostatnich buildów) i twórz alerty na podstawie pól
`pairs_per_second` oraz `sla_pairs_per_second`. Te same metadane commit/timestamp
znajdziesz w raportach UI renderingu w `reports/ci/performance_ui_render/`,
dzięki czemu można śledzić trend p90/avg względem rewizji i buildów.

## Publikacja metryk do zewnętrznych dashboardów

Job `performance-benchmarks` w workflow `.github/workflows/ci.yml` po kroku
`Summarize performance metrics` publikuje wartości `pairs_per_second`, `p90_ms`
oraz `avg_ms` (wraz z `git_commit` i `timestamp_utc`) z raportów
`reports/ci/performance_backtests/*.json` i `reports/ci/performance_ui_render/*.json`
do skonfigurowanego Pushgateway. Adres podaj w sekrecie/zmiennej
`PERFORMANCE_METRICS_PUSHGATEWAY` (repo lub organizacja); opcjonalnie można
udostępnić domyślny endpoint jako zmienną organizacyjną
`CI_PUSHGATEWAY_DEFAULT`, którą workflow pobierze automatycznie zarówno przy
healthchecku, jak i publikacji. Brak konfiguracji lub niedostępność `/-/healthy`
powoduje **fail joba** jeszcze przed uruchomieniem testów, a samo
publikowanie przerwie się z błędem również wtedy, gdy URL został usunięty na
etapie publikacji – dzięki temu nie pominiemy metryk i łatwo zdiagnozujesz
problem (jasny komunikat trafi do `GITHUB_STEP_SUMMARY`). W Pushgateway metryki
trafiają do jobu `performance-benchmarks` z etykietami źródła
(`backtest`/`ui_render`) i scenariusza (plus `pair_count`/`timeframe` tam,
gdzie są dostępne), co ułatwia wizualizację. Etykiety są bezpiecznie
escapowane, a błędne raporty JSON są pomijane, więc publikacja nie zatrzyma
całego pipeline'u.

Historia wyników z CI jest utrzymywana w artefakcie
`performance-benchmark-history` (Parquet + SQLite) generowanym w jobie
`performance-benchmarks`. Skrypt `scripts/persist_performance_benchmarks.py`
łączy bieżące raporty JSON z cachem (`reports/ci/performance_history/`),
odcina dane do okna kroczącego (domyślnie 120 rekordów na scenariusz) i zapisuje
je z metadanymi SHA i znacznikami czasu. Dzięki temu w dashboardach można
wykresami rolling window porównywać trend do konkretnego commita.

### Wizualizacja i alerty

W Grafanie dodaj datasource Prometheus wskazujący na instancję scrapującą
Pushgateway. Najważniejsze metryki to:

- `ci_performance_pairs_per_second` (etykiety `pair_count`, `timeframe`),
- `ci_performance_p90_ms` (etykieta `scenario`),
- `ci_performance_avg_ms` (etykieta `scenario`).

Do alertowania wykorzystaj progowe porównania z wartościami SLA (np.
`ci_performance_p90_ms > 220`) i oznaczaj serie labelami `git_commit` oraz
`timestamp_utc`, aby łatwo zidentyfikować regresje wydajności w konkretnych
buildach.

### Diagnostyka braku metryk

- Upewnij się, że sekret/zmienna `PERFORMANCE_METRICS_PUSHGATEWAY` jest ustawiona
  w repozytorium lub środowisku, a URL kończy się hostem dostępnym z runnera.
- Przed uruchomieniem testów sprawdź zdrowie instancji (np. z retry i limitem
  czasu):

  ```bash
  curl -sSf --retry 3 --retry-delay 2 --retry-connrefused --max-time 10 \  
    "$PERFORMANCE_METRICS_PUSHGATEWAY/-/healthy"
  ```
  polecenie powinno zwrócić status 200.
- Jeśli check w jobie `performance-benchmarks` zgłosi brak konfiguracji lub
  niedostępność, zajrzyj do `GITHUB_STEP_SUMMARY` po szczegóły i uzupełnij
  zmienną, aby kolejne runy mogły wypchnąć metryki.
