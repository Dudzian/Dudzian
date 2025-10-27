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
  --limit-freeze-events 25 \
  --raw-freeze-events-sample-limit 15 \
  --freeze-events-limit 50 \
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
- zapisuje pełny raport JSON (z surowymi wartościami, statystykami i
  sugerowanymi progami),
- tworzy tabelę CSV gotową do importu w arkuszu kalkulacyjnym,
- w pliku CSV dołącza kolumny `sample_truncated`, `retained_samples` oraz
  `omitted_samples`, dzięki którym łatwo rozpoznać, czy percentyle bazują na
  ograniczonej próbce (wartości puste oznaczają, że metryka nie korzysta z
  próbkowania),
- zbiera statystyki blokad ryzyka (`risk_freeze` / `auto_risk_freeze`) wraz z
  rozkładem długości blokad i powodów,
- pozwala kontrolować długość list blokad w sekcjach `freeze_events`
  niezależnie od próbkowania surowych zdarzeń dzięki `--freeze-events-limit`
  (np. `--freeze-events-limit 50`) oraz w razie potrzeby całkowicie je
  pominąć (`--omit-freeze-events`),
- opcjonalnie dołącza próbkę surowych blokad (`--limit-freeze-events 25`) i
  pozwala oddzielnie kontrolować jej rozmiar (`--raw-freeze-events-sample-limit 15`);
  ustawienie limitu na 0 wyłącza próbkę bez ingerencji w agregaty, a tryb
  zgodności z `--raw-freeze-events sample --raw-freeze-events-limit N`
  pozostaje dostępny,
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

Jeżeli potrzebujesz przeanalizować konkretne przykłady blokad ryzyka, włącz
próbkę `raw_freeze_events`, łącząc limit z rozmiarem próbki:

```bash
--limit-freeze-events 25 \
--raw-freeze-events-sample-limit 15
```

Raport dołączy maksymalnie 15 pierwszych blokad dla każdej kombinacji
giełda/strategia (pozostałe zostaną zagregowane w podsumowaniu). Pole
`--limit-freeze-events` aktywuje sekcję próbek, natomiast
`--raw-freeze-events-sample-limit` pozwala zmienić jej długość niezależnie od
pozostałych ograniczeń; ustawienie 0 wyłącza próbkę mimo zadanego limitu.
Pozostawienie domyślnych wartości pomija sekcję `raw_freeze_events`, co skraca
czas generowania raportu i zmniejsza jego rozmiar. Starsze flagi
`--raw-freeze-events sample` oraz `--raw-freeze-events-limit` pozostają
obsługiwane w trybie zgodności, jednak preferowanym sposobem sterowania próbką
jest duet `--limit-freeze-events` + `--raw-freeze-events-sample-limit`. Jeśli
potrzebujesz czasowo wyłączyć próbkę bez zmiany domyślnych limitów, połącz
`--raw-freeze-events-sample-limit 0` z `--limit-freeze-events N` albo użyj
`--omit-raw-freeze-events`.

Nowa flaga `--raw-freeze-events-sample-limit` pozwala wymusić niezależny limit
próbki – nawet gdy `--limit-freeze-events` pozostaje nieustawione. Dzięki temu
można skrócić sekcję `raw_freeze_events` bez ingerencji w liczbę zdarzeń w
`freeze_events`. Wartość `0` zachowuje wyłącznie agregaty, co przydaje się przy
generowaniu zwięzłych raportów do szybkiej inspekcji. Niezależnie od wybranego
limitu, pole `freeze_summary` zachowuje kompletne zliczenia blokad oraz ich
powodów – próbkowanie wpływa jedynie na listę przykładów i metadane w sekcji
`sources.raw_freeze_events`.

Jeżeli lista surowych blokad zaczyna dominować w raporcie, można skrócić ją bez
zmiany agregatów, korzystając z `--max-raw-freeze-events`. Parametr ten
przycina liczbę zdarzeń zapisywanych w sekcjach `raw_freeze_events`, a resztę
zlicza do `overflow_summary`, które nadal odzwierciedla pełny rozkład typów i
powodów blokad. W połączeniu z `--omit-raw-freeze-events` można całkowicie
wyłączyć sekcję z próbkami, zachowując jedynie zagregowane podsumowania.

Nowy przełącznik `--raw-freeze-events-sample-limit` uzupełnia istniejące opcje –
steruje liczbą zdarzeń, które trafią do próbek `raw_freeze_events`, również w
przypadku raportów generowanych z `--freeze-events-limit`. Wartość dodatnia
przycina próbkę do wskazanego rozmiaru, `0` wyłącza sekcję próbek, a `None`
(wartość domyślna) pozostawia limit wyznaczony przez `--limit-freeze-events` lub
domyślne 25 zdarzeń. Przełącznik `--freeze-events-limit` działa równolegle –
kontroluje długość list w sekcjach `freeze_events`, które prezentują
znormalizowane blokady opracowane przez raport (nie surowe wpisy). Przykładowe
`--freeze-events-limit 25` skraca każdą listę do 25 rekordów, ale nie wpływa na
to, czy próbka surowych zdarzeń zostanie wygenerowana (`--limit-freeze-events`)
ani na decyzję o jej pominięciu (`--omit-raw-freeze-events`). Chcąc całkowicie
zrezygnować z list `freeze_events`, użyj `--omit-freeze-events` – agregaty
pozostaną kompletne.

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
z zachowaniem lekkiego rozmiaru raportu.

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

## Interpretacja sekcji `sources`

Sekcja `sources` na końcu raportu zbiera metadane o wejściowych plikach oraz
zastosowanych limitach. Pozwala to odtworzyć konfigurację bez zaglądania w
parametry CLI, co bywa pomocne podczas wymiany raportów między zespołami.

Niezależnie od tego, jak ustawisz limity (`--limit-freeze-events`,
`--raw-freeze-events-sample-limit`, `--max-raw-freeze-events`,
`--freeze-events-limit`), sekcja `freeze_summary` w grupach i w
`global_summary` zawsze bazuje na pełnym zbiorze zdarzeń. Metadane w
`sources.raw_freeze_events` pomagają odtworzyć, jak raport dobrał próbkę –
`mode` informuje, czy zapisano listę (i w jaki sposób), `limit` i
`display_limit` pokazują rzeczywiste ograniczenia, `requested_limit`
odzwierciedla wartości z CLI (w tym `--raw-freeze-events-sample-limit`), a
`overflow_summary` zlicza odcięte wpisy. Dzięki temu można zweryfikować, że
agregaty pozostały kompletne, nawet jeśli próbka została mocno przycięta.

- `current_signal_thresholds.files` / `inline` – źródła aktualnych progów
  przekazanych przez `--current-threshold`. Wartości inline odpowiadają parom
  `metric=value`, a ścieżki wskazują pliki JSON/YAML.
- `risk_threshold_files` oraz `risk_thresholds.files` / `inline` – analogiczne
  metadane dla `--risk-thresholds`, dzięki którym można zweryfikować, czy
  wartości `risk_score` pochodzą z pliku konfiguracyjnego, czy z parametru CLI.
- `raw_freeze_events.mode` – informuje, czy surowe blokady są próbkowane
  niezależnym samplerem (`sample`), wykorzystywane bezpośrednio z limitu
  blokad (`limit`) czy pominięte (`omit`). To samo pole pojawia się w sekcjach
  `raw_freeze_events` w grupach i podsumowaniu globalnym, dzięki czemu łatwo
  powiązać listę zdarzeń z zastosowanym ograniczeniem.
  - Gdy raport zawiera próbkę (`sample` lub `limit`), pole `limit` opisuje
    faktyczną liczbę zdarzeń zapisanych w grupie/globalnie po zastosowaniu
    ograniczeń. `requested_limit` wskazuje żądanie operatora (np. z
    `--limit-freeze-events`, `--raw-freeze-events-sample-limit` albo limitu
    przekazanego do `_generate_report`), a
    `display_limit` pojawia się, gdy lista została dodatkowo przycięta przez
    `--max-raw-freeze-events`. Ewentualne odcięte rekordy są zliczane w
    `overflow_summary`, która zachowuje pełny rozkład statusów, typów i powodów.
    Dodatkowe pole `display_overflow_summary` (jeżeli występuje) obejmuje wyłącznie
    elementy ucięte przez ograniczenie wyświetlania, co pozwala rozróżnić je od
    rekordów odrzuconych wcześniej przez limit próbkowania.
  - Dla trybu `omit` dostępne są powody (`reason`): `explicit_omit` oznacza
    użycie `--omit-raw-freeze-events`, `limit_zero` odpowiada `--max-raw-freeze-events 0`,
    a `sampling_disabled`/`no_samples` sygnalizują brak aktywnej próbki mimo
    ograniczenia `--limit-freeze-events`.
- `freeze_events.mode` – wskazuje, czy sekcje `freeze_events` zawierają wszystkie
  zdarzenia (`all`), czy zostały przycięte (`limit`). W drugim przypadku pole
  `overflow_summary` prezentuje agregaty dla odrzuconych rekordów, co ułatwia
  ocenę skali przycięcia.

Przykładowy fragment metadanych dla raportu z aktywną próbką blokad może
wyglądać następująco:

```json
"sources": {
  "raw_freeze_events": {
    "mode": "sample",
    "display_limit": 3,
    "requested_limit": 10,
    "limit": 3,
    "overflow_summary": {
      "total": 7,
      "reasons": [
        {"reason": "risk_score_threshold", "count": 5},
        {"reason": "manual_override", "count": 2}
      ]
    }
  },
  "freeze_events": {
    "mode": "limit",
    "limit": 50
  }
}
```

Interpretacja:

- Operator poprosił o próbkę maksymalnie 10 zdarzeń (`requested_limit`), ale po
  przycięciu listy przez `--max-raw-freeze-events 3` w raporcie znalazły się
  tylko trzy pierwsze wpisy (`limit`).
- Pozostałe wpisy nie znikają – `overflow_summary.total` zlicza wszystkie 7
  odrzuconych blokad, a ich powody można odczytać z listy `reasons`.
- Sekcja `freeze_events` została ograniczona do 50 rekordów (często jest to
  domyślna wartość narzucona przez `--freeze-events-limit`), ale ponieważ
  liczba blokad była mniejsza, w raporcie znajdziemy komplet wpisów.

Jeżeli raport zostanie uruchomiony z `--omit-raw-freeze-events`, metadane
zmienią się na `{"mode": "omit", "reason": "explicit_omit"}` – brak listy
zdarzeń jest wtedy świadomą decyzją operatora, a pełne agregaty nadal znajdują
się w `freeze_summary` i `global_summary`.

### Kompletność agregatów i interpretacja `sources.raw_freeze_events`

Limity próbkowania (`--limit-freeze-events`, `--raw-freeze-events-sample-limit`,
`--max-raw-freeze-events`, `--freeze-events-limit`) wpływają wyłącznie na to, ile
rekordów trafi na listy w sekcjach `raw_freeze_events` i `freeze_events`.
Wszystkie zdarzenia – także te ucięte na etapie prezentacji – nadal zasilają
agregaty w `freeze_summary` oraz `global_summary`. Dzięki temu można bez obaw
przycinać próbki w celu uproszczenia raportu, nie tracąc informacji o łącznej
liczbie blokad, ich typach ani powodach.

Sekcja `sources.raw_freeze_events` opisuje zastosowane ograniczenia. Pole
`mode` rozróżnia próbkę (`sample`), limit wymuszony przez `--limit-freeze-events`
(`limit`) oraz świadome pominięcie (`omit`). `requested_limit` odzwierciedla
wartość przekazaną z CLI (np. `--limit-freeze-events 25` lub
`--raw-freeze-events-sample-limit 15`), `limit` pokazuje faktyczną liczbę wpisów
po uwzględnieniu wszystkich ograniczeń, a `display_limit` wskazuje dodatkowe
przycięcie wynikające z `--max-raw-freeze-events`. Kiedy `mode` przyjmuje
wartość `omit`, pole `reason` pozwala rozpoznać, czy lista została pominięta z
inicjatywy operatora (`--omit-raw-freeze-events`), czy na przykład ustawiono
limit 0. Analiza tych metadanych pomaga odtworzyć konfigurację raportu oraz
potwierdzić, że agregaty pozostały kompletne mimo zmian w próbkach.

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
