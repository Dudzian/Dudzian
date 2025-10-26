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
   Parser działa strumieniowo – akceptuje zarówno pliki JSONL, jak i duże pliki
   JSON, w których dane znajdują się w tablicy lub wewnątrz obiektu z polem
   `entries`. Inne struktury (np. zagnieżdżone tablice w wielu polach) nie są
   obsługiwane.
3. *(Opcjonalnie)* Aktualne progi sygnałów – można je przekazać jako plik
   JSON/YAML albo listę par `metric=value` w parametrze
   `--current-threshold`. Parametr można wskazać wielokrotnie (np. plik z
   domyślnymi progami + szybka korekta w CLI), co pozwala porównać nowe
   propozycje z bieżącą konfiguracją `signal_after_adjustment` i
   `signal_after_clamp`.
4. *(Opcjonalnie)* Konfiguracja progów ryzyka – plik YAML/JSON kompatybilny z
   `load_risk_thresholds`. Przekaż go przez `--risk-thresholds`, aby w raporcie
   pojawiła się aktualna wartość `risk_score`. Flagi można powtórzyć dla kilku
   plików, a ostatnia ścieżka nadpisze poprzednie wartości (podobnie jak
   warstwy `--current-threshold`).

## Przykładowe uruchomienie

```bash
python scripts/calibrate_autotrade_thresholds.py \
  --journal logs/decision_journal/decisions-20240101.jsonl \
  --autotrade-export audit/autotrade/risk_history.json \
  --percentiles 0.5,0.9,0.95 \
  --suggestion-percentile 0.95 \
  --since 2024-01-01T00:00:00Z \
  --until 2024-01-31T23:59:59Z \
  --current-threshold signal_after_adjustment=0.8,signal_after_clamp=0.75 \
  --risk-thresholds config/risk_thresholds.yaml \
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
- opcjonalnie dołącza próbkę surowych blokad (`--limit-freeze-events 20`)
  ograniczoną wskazaną wartością (dla kompatybilności nadal działa
  kombinacja `--raw-freeze-events sample --raw-freeze-events-limit 20`),
- pozwala ograniczyć analizę do konkretnego zakresu czasowego dzięki `--since`
  i `--until`,
- dodaje globalne podsumowanie obejmujące wszystkie kombinacje giełda/strategia
  (również w pliku CSV jako wiersze `__all__/__all__`),
- opcjonalnie generuje histogramy (wymaga `matplotlib`).

Jeżeli przekażesz aktualne progi w pliku (np. `config/current_thresholds.yaml`),
użyj `--current-threshold config/current_thresholds.yaml`. Skrypt automatycznie
wyszuka wartości `signal_after_adjustment`, `signal_after_clamp` oraz
`risk_score` wewnątrz struktury JSON/YAML oraz umieści je w polu
`current_threshold` w raporcie i pliku CSV. Możesz też połączyć plik z
dodatkowym nadpisaniem progu w CLI:

```bash
--current-threshold config/current_thresholds.yaml \
--current-threshold signal_after_clamp=0.78
```

Analogicznie `--risk-thresholds` pozwala zaczytać alternatywny plik z
konfiguracją progów ryzyka (`load_risk_thresholds(config_path=...)`). Wskazane
ścieżki są przetwarzane w kolejności podania – jeżeli ostatni plik zawiera
zmodyfikowany `map_regime_to_signal.risk_score`, to właśnie ta wartość pojawi
się w kolumnie `current_threshold` dla metryki `risk_score`.

Jeżeli potrzebujesz przeanalizować konkretne przykłady blokad ryzyka, możesz
włączyć próbkę `raw_freeze_events`:

```bash
--limit-freeze-events 20
```

Raport dołączy maksymalnie 20 pierwszych blokad dla każdej kombinacji
giełda/strategia (pozostałe zostaną zagregowane w podsumowaniu). Pozostawienie
domyślnych wartości pomija sekcję `raw_freeze_events`, co skraca czas
generowania raportu i zmniejsza jego rozmiar. Starsze flagi
`--raw-freeze-events sample` oraz `--raw-freeze-events-limit` pozostają
obsługiwane, ale zalecamy korzystanie z `--limit-freeze-events`.

Źródło musi wskazywać istniejący plik, który zawiera słownik lub listę
słowników – w przeciwnym razie skrypt zakończy się z komunikatem o błędzie,
aby uniknąć cichego pominięcia progów. Parser potrafi wyłuskać wartości
`signal_after_adjustment` i `signal_after_clamp` zarówno z prostych pól
(`"signal_after_clamp": 0.78`), jak i zagnieżdżonych struktur (np.
`{"metric": "signal_after_adjustment", "current_threshold": 0.8}` lub
`{"signal_after_clamp": {"threshold": 0.75}}`).

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
