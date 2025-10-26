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
  --max-freeze-events 100 \
  --max-raw-values 500 \
  --max-global-samples 50000 \
  --output-json reports/autotrade_thresholds.json \
  --output-csv reports/autotrade_thresholds.csv \
  --plot-dir reports/autotrade_thresholds_plots
```

Polecenie:

- zaczytuje wszystkie zdarzenia z dziennika oraz eksportu autotradera,
- oblicza wskazane percentyle w rozbiciu na pary `giełda/strategia`
  (kolumny percentylowe w raporcie JSON/CSV są oznaczane jako `pXX` dla
  wartości całkowitych lub `pXX_Y` dla ułamków, np. `p97_5` dla 97,5
  percentyla),
- zapisuje pełny raport JSON (z surowymi wartościami w `groups[*].raw_values`,
  statystykami i sugerowanymi progami; sekcja `global_summary` przechowuje tylko
  dane zagregowane),
- tworzy tabelę CSV gotową do importu w arkuszu kalkulacyjnym,
- zbiera statystyki blokad ryzyka (`risk_freeze` / `auto_risk_freeze`) wraz z
  rozkładem długości blokad i powodów,
- w pliku CSV dodaje dodatkowy wiersz `metric=__freeze_summary__` dla każdej
  kombinacji giełda/strategia; kolumny `freeze_total`, `freeze_auto`,
  `freeze_manual`, `freeze_omitted`, `freeze_status_counts`,
  `freeze_reason_counts` oraz `freeze_truncated` przedstawiają odpowiednio
  liczbę blokad, rozbicie auto/manual, liczbę pominiętych wpisów, a także
  szczegóły statusów i powodów (format JSON). Wiersz `__all__/__all__`
  zawiera analogiczne dane zagregowane,
- pozwala ograniczyć liczbę szczegółowych wpisów blokad dzięki
  `--max-freeze-events` (0 = pominięcie szczegółowej listy, dodatnia wartość =
  prefiks o zadanej długości),
- pozwala ograniczyć analizę do konkretnego zakresu czasowego dzięki `--since`
  i `--until`,
- dodaje globalne podsumowanie obejmujące wszystkie kombinacje giełda/strategia
  (również w pliku CSV jako wiersze `__all__/__all__`),
- opcjonalnie generuje histogramy (wymaga `matplotlib`).
- umożliwia ograniczenie liczby przechowywanych surowych próbek metryk poprzez
  `--max-raw-values` (0 = brak próbek w raporcie; dodatnie wartości =
  deterministyczne próbkowanie rezerwuarowe o zadanym rozmiarze dla każdej pary
  giełda/strategia, co zapewnia reprezentatywną próbkę do histogramów zamiast
  sztywnego prefiksu).
- ogranicza liczbę próbek wykorzystywanych do percentyli w sekcji
  `global_summary` za pomocą `--max-global-samples` (domyślnie 50 000). Limit
  dotyczy każdej metryki agregowanej globalnie; ustawienie wartości ujemnej
  wyłącza limit, a `0` powoduje, że percentyle i sugerowane progi w podsumowaniu
  bazują wyłącznie na statystykach agregowanych (mean, stddev) i nie posiadają
  próbek do histogramów. Przy aktywnym limicie percentyle są szacowane na
  podstawie deterministycznego próbkowania rezerwuarowego, co należy uwzględnić
  przy interpretacji wyników.

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

Wartości `risk_score` wczytane z plików wskazanych przez `--current-threshold`
trafiają do metadanych `sources.current_signal_thresholds.file_risk_score` i są
traktowane jako aktualny próg w raporcie. Jeśli podasz `risk_score` bezpośrednio
w CLI (np. `--current-threshold risk_score=0.72`), wartość pojawi się w
`sources.risk_score_override` oraz w mapie
`sources.current_signal_thresholds.inline`. Brak nadpisania w CLI oznacza, że
raport traktuje `risk_score` z pliku wyłącznie jako dokumentację istniejącej
konfiguracji.

Analogicznie `--risk-thresholds` pozwala zaczytać alternatywny plik z
konfiguracją progów ryzyka (`load_risk_thresholds(config_path=...)`). Wskazane
ścieżki są przetwarzane w kolejności podania – jeżeli ostatni plik zawiera
zmodyfikowany `map_regime_to_signal.risk_score`, to właśnie ta wartość pojawi
się w kolumnie `current_threshold` dla metryki `risk_score`.

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
blokad dla całego zbioru danych (bez duplikowania surowych wartości). Dzięki
temu można szybko ocenić, jak nowe progi wpłyną na wszystkie strategie łącznie,
z zachowaniem lekkiego rozmiaru raportu. Metryki w `global_summary` zawierają
teraz dodatkowe pola `sample_truncated`, `retained_samples` oraz
`omitted_samples`, które sygnalizują, czy limit `--max-global-samples` został
osiągnięty oraz ile obserwacji znalazło się poza próbką wykorzystywaną do
percentyli/sugestii progów. Metadane raportu (`sources`) przechowują użyty limit
(`max_global_samples`) oraz liczbę metryk i próbek pominiętych przez filtr
(`global_samples_truncated_metrics`, `global_samples_omitted_total`).

Jeżeli raport generowany jest na potrzeby krótkiego podsumowania, ustaw
`--max-freeze-events` na niewielką liczbę (np. 25), aby nie kopiować tysięcy
zdarzeń `raw_freeze_events` do wyniku. Pole `raw_freeze_events_truncated`
informuje, czy lista została ucięta względem łącznej liczby blokad widocznej w
`freeze_summary`, a `raw_freeze_events_omitted` wskazuje dokładną liczbę
pominiętych wpisów. Dodatkowo `freeze_summary.omitted` (zarówno w grupach, jak i
w sekcji `global_summary`) prezentuje łączną liczbę blokad pominiętych w danym
wierszu. Metadane raportu przechowują wykorzystany limit w polu
`sources.max_freeze_events`, a liczba skróconych grup dostępna jest w
`sources.raw_freeze_events_truncated_groups`.

Analogicznie limit `--max-raw-values` kontroluje liczbę próbek
przechowywanych w `groups[*].raw_values`. Po osiągnięciu limitu skrypt
wykorzystuje deterministyczne próbkowanie rezerwuarowe – oznacza to, że
przechowywana próbka jest losowo dobierana (z ustalonym ziarnem), a nie tylko
pierwszymi N obserwacjami. Raport oznaczy takie grupy flagą
`raw_values_truncated`, wskaże liczbę pominiętych próbek w mapie
`raw_values_omitted` (w rozbiciu na metryki), a także zapisze wykorzystany
limit i łączną liczbę pominiętych wartości w metadanych `sources`
(`max_raw_values`, `raw_values_truncated_groups`, `raw_values_omitted_total`).
Histogramy bazujące na `raw_values` należy więc interpretować jako wynik
losowego, ale deterministycznie powtarzalnego próbkowania całej serii.

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
