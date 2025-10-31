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

Skrypt nie ma osobnego joba CI – można go dodać do istniejących pipeline’ów,
wywołując CLI i archiwizując plik z `logs/loadtests/`. Dzięki parametrowi
`--seed` raporty mogą być deterministyczne, co ułatwia porównania między
buildami.
