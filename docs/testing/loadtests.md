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
- **backtest throughput** – minimalna przepustowość backtestów w CI to:
  - 2 pary @ 1m: **≥ 1.5 par/s**,
  - 4 pary @ 5m: **≥ 1.0 par/s**,
  - 6 par @ 1h: **≥ 0.6 par/s**.

Wyniki benchmarków backtestów są automatycznie zapisywane do
`reports/ci/performance_backtests/` jako JSON, dzięki czemu mogą być
zaciągane przez pipeline’y i Grafanę. Analogicznie, pomiary renderingu
paneli SLA i Risk w QML zapisujemy w `reports/ci/performance_ui_render/`
z polami `avg_ms`, `p90_ms` i `sla_ms` – ułatwia to wizualizację trendów i
alertowanie na odchylenia od progów SLA.

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
`PERFORMANCE_METRICS_PUSHGATEWAY`; brak konfiguracji lub brak plików z
metrykami powoduje pominięcie wysyłki, ale nie psuje joba. W Pushgateway
metryki trafiają do jobu `performance-benchmarks` z etykietami źródła
(`backtest`/`ui_render`) i scenariusza (plus `pair_count`/`timeframe` tam, gdzie
są dostępne), co ułatwia wizualizację. Etykiety są bezpiecznie escapowane, a
błędne raporty JSON są pomijane, więc publikacja nie zatrzyma całego pipeline'u.

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
