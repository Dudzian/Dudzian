# Kalibracja progów autotradera

Skrypt `scripts/calibrate_autotrade_thresholds.py` pozwala operatorom szybko
oszacować nowe progi sygnałów (`signal_after_adjustment`,
`signal_after_clamp`) oraz poziomów ryzyka (`regime_summary.risk_score`) na
podstawie istniejących danych operacyjnych. Narzędzie łączy zdarzenia z
`TradingDecisionJournal` z eksportem historii autotradera i oblicza
percentyle, które można wykorzystać do aktualizacji konfiguracji
`auto_trader` (np. `risk_score` w `map_regime_to_signal`).

## Wymagane dane wejściowe

1. Ścieżka do pliku (lub katalogu) z dziennikiem decyzji w formacie JSONL.
   Można przekazać wiele plików/katalogów – zostaną scalone.
2. Eksport autotradera wygenerowany przez `export_risk_evaluations()` lub
   zrzut statusów z `push_autotrade_status` (plik JSON zawierający listę
   wpisów z polami `symbol` i `summary.risk_score`).

## Przykładowe uruchomienie

```bash
python scripts/calibrate_autotrade_thresholds.py \
  --journal logs/decision_journal/decisions-20240101.jsonl \
  --autotrade-export audit/autotrade/risk_history.json \
  --percentiles 0.5,0.9,0.95 \
  --suggestion-percentile 0.95 \
  --since 2024-01-01T00:00:00Z \
  --until 2024-01-31T23:59:59Z \
  --output-json reports/autotrade_thresholds.json \
  --output-csv reports/autotrade_thresholds.csv \
  --plot-dir reports/autotrade_thresholds_plots
```

Polecenie:

- zaczytuje wszystkie zdarzenia z dziennika oraz eksportu autotradera,
- oblicza wskazane percentyle w rozbiciu na pary `giełda/strategia`,
- zapisuje pełny raport JSON (z surowymi wartościami, statystykami i
  sugerowanymi progami),
- tworzy tabelę CSV gotową do importu w arkuszu kalkulacyjnym,
- zbiera statystyki blokad ryzyka (`risk_freeze` / `auto_risk_freeze`) wraz z
  rozkładem długości blokad i powodów,
- pozwala ograniczyć analizę do konkretnego zakresu czasowego dzięki `--since`
  i `--until`,
- dodaje globalne podsumowanie obejmujące wszystkie kombinacje giełda/strategia
  (również w pliku CSV jako wiersze `__all__/__all__`),
- opcjonalnie generuje histogramy (wymaga `matplotlib`).

Wynik na stdout zawiera podsumowanie liczby przetworzonych zdarzeń. Raport
JSON zawiera pole `suggested_threshold`, które można porównać z aktualną
wartością w konfiguracji (`current_threshold`). Każda grupa otrzymuje też
sekcję `freeze_summary` z rozbiciem na blokady automatyczne i manualne,
powody blokad oraz histogramy długości (`risk_freeze_duration`).

Pole `global_summary` w raporcie JSON zawiera zagregowane metryki i statystyki
blokad dla całego zbioru danych. Dzięki temu można szybko ocenić, jak nowe
progi wpłyną na wszystkie strategie łącznie, zanim rozpocznie się szczegółowa
analiza poszczególnych par.

## Wskazówki operacyjne

- Percentyl sugerowany (`--suggestion-percentile`) domyślnie wynosi 0,95 i
  bazuje na wartości bezwzględnej sygnału. Dla `risk_score` sugerowana wartość
  odnosi się do aktualnych obserwacji i warto sprawdzić, czy nie przekracza
  limitów z `config/risk_thresholds.yaml`.
- Jeśli w dzienniku pojawiają się symbole bez dopasowania do eksportu
  autotradera, w raporcie mogą pojawić się grupy `unknown/unknown`. Warto
  wtedy uzupełnić źródłowe dane lub zaktualizować mapowanie symbolu na giełdę
  i strategię.
- Histogramy pomagają szybko zweryfikować, czy rozkład wartości nie jest
  wielomodalny i czy percentyl nie wycina istotnych obserwacji.
