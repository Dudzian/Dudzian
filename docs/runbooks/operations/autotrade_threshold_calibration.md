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
   wpisów z polami `symbol` i `summary.risk_score`). Narzędzie obsługuje
   zarówno pojedynczy dokument JSON (lista lub obiekt z polem `entries`), jak
   i eksporty strumieniowe w formacie JSONL/NDJSON (po jednym obiekcie na
   linię). Pojedyncze pliki NDJSON bez rozszerzenia, zawierające pojedynczy
   wpis w pierwszej linii, są również rozpoznawane heurystycznie. Jeżeli
   przekazujesz pojedynczy obiekt JSON, musi on zawierać pole `entries`
   będące tablicą wpisów; plik opisujący pojedynczy rekord bez `entries`
   zostanie odrzucony z komunikatem o błędnym formacie.
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
  --max-group-samples 50000 \
  --max-global-samples 50000 \
  --output-json reports/autotrade_thresholds.json \
  --output-csv reports/autotrade_thresholds.csv \
  --output-threshold-config reports/autotrade_thresholds.yaml \
  --plot-dir reports/autotrade_thresholds_plots
```

Polecenie:

- zaczytuje wszystkie zdarzenia z dziennika oraz eksportu autotradera,
- oblicza wskazane percentyle w rozbiciu na pary `giełda/strategia`
  (kolumny percentylowe w raporcie JSON/CSV są oznaczane jako `pXX` dla
  wartości całkowitych lub `pXX_Y` dla ułamków, np. `p97_5` dla 97,5
  percentyla). Metryki absolutne (`signal_after_adjustment`,
  `signal_after_clamp`) otrzymują dodatkową sekcję `absolute_percentiles`
  (oraz kolumny CSV `abs_pXX` / `abs_pXX_Y`), w której percentyle liczone są na
  podstawie wartości bezwzględnych lub – przy wyłączonym próbkowaniu – ich
  aproksymacji folded-normal,
- zapisuje pełny raport JSON (z surowymi wartościami w `groups[*].raw_values`,
  statystykami i sugerowanymi progami; sekcja `global_summary` przechowuje tylko
  dane zagregowane),
- tworzy tabelę CSV gotową do importu w arkuszu kalkulacyjnym,
- generuje opcjonalny plik konfiguracyjny (`--output-threshold-config`) w
  formacie zgodnym z `load_risk_thresholds`, który zawiera sugerowane progi dla
  sekcji `auto_trader` – zarówno globalne (`signal_thresholds`,
  `map_regime_to_signal`), jak i w rozbiciu na pary `giełda/strategia`
  (`strategy_signal_thresholds`),
- w pliku CSV dołącza kolumny `sample_truncated`, `retained_samples`,
  `omitted_samples`, `raw_values_truncated`, `raw_values_omitted`, a także
  liczniki `clamp_regular` i `clamp_absolute`, dzięki którym łatwo rozpoznać,
  czy percentyle bazują na ograniczonej próbce, czy lista surowych wartości
  została ucięta oraz czy wartości były przycinane do domeny metryki (puste
  pola oznaczają brak próbkowania/klamrowania). Kolumna `approximation_mode`
  sygnalizuje, że metryka korzysta z przybliżenia (np. `approximate_from_moments`
  przy ustawieniu `--max-*-samples 0`). Dodatkowe kolumny `abs_pXX`/`abs_pXX_Y`
  odwzorowują percentyle amplitudy dla metryk absolutnych i są przydatne przy
  analizie progów ustawianych w oparciu o moduł sygnału,
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
- pozwala kontrolować liczbę próbek gromadzonych przez agregatory metryk w
  każdej parze giełda/strategia dzięki `--max-group-samples` (wartość 0 lub
  ujemna wyłącza limit). Parametr jest niezależny od `--max-raw-values`, więc
  można np. ustawić `--max-raw-values 0`, aby wyłączyć próbki do histogramów,
  zachowując pełne percentyle i sugestie progów,
- pozwala ograniczyć analizę do konkretnego zakresu czasowego dzięki `--since`
  i `--until`,
- dodaje globalne podsumowanie obejmujące wszystkie kombinacje giełda/strategia
  (również w pliku CSV jako wiersze `__all__/__all__`). Wiersze te zawierają
  również skumulowane liczniki `raw_values_omitted` oraz flagę
  `raw_values_truncated`, dzięki czemu można szybko sprawdzić, czy na poziomie
  całego zestawu wystąpiły ucięcia próbek,
- opcjonalnie generuje histogramy (wymaga `matplotlib`).
- umożliwia ograniczenie liczby przechowywanych surowych próbek metryk poprzez
  `--max-raw-values` (0 = brak próbek w raporcie; dodatnie wartości =
  deterministyczne próbkowanie rezerwuarowe o zadanym rozmiarze dla każdej pary
  giełda/strategia, co zapewnia reprezentatywną próbkę do histogramów zamiast
  sztywnego prefiksu).
- ogranicza liczbę próbek wykorzystywanych do percentyli w sekcji
  `global_summary` za pomocą `--max-global-samples` (domyślnie 50 000). Limit
  dotyczy każdej metryki agregowanej globalnie; ustawienie wartości ujemnej
  wyłącza limit, natomiast `0` powoduje, że percentyle i sugerowane progi w
  podsumowaniu są szacowane przy założeniu rozkładu normalnego na podstawie
  sumy i wariancji (bez gromadzenia próbek do histogramów). Aproksymowane
  wartości są automatycznie przycinane do zakresu zaobserwowanych danych oraz
  znanych domen metryk (np. `risk_score` ∈ ⟨0,1⟩, `signal_after_*` ∈ ⟨−1,1⟩,
  `risk_freeze_duration` ≥ 0), więc raport nie pokaże wyników spoza realnego
  przedziału. Przy aktywnym limicie dodatnim percentyle są wyznaczane z
  deterministycznego próbkowania rezerwuarowego, co należy uwzględnić przy
  interpretacji wyników. Pole `approximation_mode` w raporcie JSON oraz
  analogiczna kolumna w CSV sygnalizują, że metryka korzysta z takiej
  aproksymacji; w metadanych `sources` znajdziesz liczbę metryk, których to
  dotyczy (`approximation_metrics` dla grup oraz `approximation_global_metrics`
  dla podsumowania globalnego) oraz szczegółowe listy z identyfikatorem giełdy,
  strategii i nazwy metryki (`approximation_metrics_list`) albo nazwą metryki
  globalnej (`approximation_global_metrics_list`). Dzięki temu podczas audytu
  można szybko ustalić, które sekcje raportu bazują na przybliżeniach.

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

## Eksport sugerowanych progów do pliku konfiguracyjnego

Parametr `--output-threshold-config` zapisuje dodatkowy plik w formacie JSON lub
YAML (w zależności od rozszerzenia), kompatybilny z `load_risk_thresholds`. W
sekcji `auto_trader` umieszczane są:

- `signal_thresholds` – sugerowane wartości globalne dla
  `signal_after_adjustment` i `signal_after_clamp` (percentyl wskazany w
  `--suggestion-percentile`),
- `map_regime_to_signal.risk_score` – nowy próg ryzyka wynikający z
  `global_summary`,
- `strategy_signal_thresholds` – zagnieżdżona mapa `primary_exchange ->
  strategy -> metryka`, która zawiera sugerowane wartości progów dla każdej
  kombinacji giełda/strategia obecnej w raporcie.

### Integracja z runtime

Eksportowane progi są konsumowane bezpośrednio przez runtime:

- `bot_core.ai.config_loader.load_risk_thresholds` waliduje i normalizuje
  sekcje `signal_thresholds` oraz `strategy_signal_thresholds`, dzięki czemu
  wartości z raportu trafiają do konfiguracji `auto_trader` niezależnie od
  formatu (JSON/YAML) czy wariantów wielkości liter.
- `AutoTradeEngine` korzysta z tych sekcji przy starcie (oraz po wywołaniu
  `apply_signal_threshold_overrides`). Wartość `signal_after_clamp` zastępuje
  klasyczny `activation_threshold`, a `signal_after_adjustment` wyzerowuje
  sygnał jeszcze przed porównaniem z progiem clamp – pozwala to całkowicie
  wyciszyć zbyt słabe sygnały bez modyfikowania wag strategii.
- Aktywne progi są publikowane w telemetrii (`metadata.signal_thresholds` w
  zdarzeniach `SIGNAL` i `AUTOTRADE_STATUS`) oraz w migawce
  `AutoTradeEngine.signal_threshold_snapshot()`, co ułatwia audyt i korelację z
  raportem kalibracyjnym.

Kombinacje, dla których nie udało się ustalić giełdy lub strategii (np. wpisy
oznaczone jako `unknown`), są pomijane – nie pojawią się więc w wygenerowanym
pliku progów.

Plik można wprost skopiować do `config/risk_thresholds.yaml` (lub wskazać jako
warstwę override) – nie wymaga dodatkowej normalizacji kluczy. Jeżeli w danej
grupie brakuje danych dla którejś metryki, nie zostanie ona uwzględniona w
eksporcie. W przypadku gdy raport nie zawiera żadnych sugerowanych progów,
narzędzie zgłosi ostrzeżenie i nie utworzy pliku.

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
percentyli/sugestii progów. Metryki w sekcjach `metrics` zawierają dodatkowe
pole `approximation_mode`, które przyjmuje wartość `null`, gdy percentyle
bazują na rzeczywistej próbce, albo nazwę metody aproksymacji (np.
`approximate_from_moments`, gdy próbkowanie jest wyłączone). Metadane raportu
(`sources`) przechowują użyte limity próbkowania – globalny
(`max_global_samples`) oraz grupowy (`max_group_samples`) – a także liczbę
metryk i próbek pominiętych przez oba mechanizmy
(`global_samples_truncated_metrics`, `global_samples_omitted_total`,
`group_samples_truncated_metrics`, `group_samples_omitted_total`). Dodatkowo
liczniki `approximation_global_metrics` i `approximation_metrics` wskazują,
ile metryk (odpowiednio w podsumowaniu globalnym oraz w grupach) korzystało z
aproksymacji. Warto pamiętać, że dla `--max-global-samples 0` percentyle są
przybliżane na bazie średniej i wariancji (założenie rozkładu normalnego),
dlatego przy interpretacji wyników należy zweryfikować, czy rozkład nie
odbiega znacząco od gaussowskiego. Metryki absolutne (`signal_after_*`)
korzystają w takim trybie z rozkładu modułu (folded normal), a uzyskane
percentyle i sugerowane progi są dodatkowo przycinane do obserwowanego
zakresu oraz domeny metryki.

Agregatory strumieniowe śledzą również najmniejszą napotkaną wartość
bezwzględną danej metryki. Dzięki temu progi dla `signal_after_*` nie są
sztucznie podbijane do `min(|min|, |max|)` – nawet przy dwukierunkowych
rozkładach raport potrafi zasugerować wartości bliższe faktycznemu minimum
modułu (np. 0.02), jeżeli takie obserwacje wystąpiły w danych wejściowych.

Ponieważ agregatory pilnują domen metryk, każda sekcja `metrics` zawiera teraz
pole `clamped_values` z licznikami przycięć w trakcie obliczeń. Wartość
`regular` odnosi się do klasycznych ograniczeń (np. `risk_score`), a `absolute`
do metryk przetwarzanych na wartości bezwzględne (`signal_after_*`). Dodatkowe
pole `clamped_values` w metadanych `sources` gromadzi sumaryczne liczniki oraz
liczbę metryk, których dotyczyło przycinanie. Jeśli którykolwiek z liczników
jest dodatni, warto sprawdzić, czy wartości wejściowe nie wychodzą poza
oczekiwany zakres albo czy zdefiniowana domena metryki nie jest zbyt wąska.

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
pierwszymi N obserwacjami. Ziarnowanie pochodzi ze stabilnego skrótu BLAKE2s,
dzięki czemu wynik nie zależy od `PYTHONHASHSEED` środowiska i pozostaje
powtarzalny między uruchomieniami. Raport oznaczy takie grupy flagą
`raw_values_truncated`, wskaże liczbę pominiętych próbek w mapie
`raw_values_omitted` (w rozbiciu na metryki), a także zapisze wykorzystany
limit i łączną liczbę pominiętych wartości w metadanych `sources`
(`max_raw_values`, `raw_values_truncated_groups`, `raw_values_omitted_total`).
Histogramy bazujące na `raw_values` należy więc interpretować jako wynik
losowego, ale deterministycznie powtarzalnego próbkowania całej serii. Limit
agregatorów (`sources.max_group_samples`) działa niezależnie, więc ustawienie
`--max-raw-values 0` wyłącza jedynie próbki wykorzystywane do histogramów –
percentyle i sugerowane progi pozostają dostępne, o ile limit grupowy jest
dodatni lub wyłączony.

Wewnętrzne agregatory przetwarzają obecnie każdą metrykę tylko raz –
posortowana próbka wykorzystywana jest równocześnie do raportowania
percentyli oraz do wyliczania sugerowanego progu, co eliminuje wcześniejsze
podwójne sortowania przy dużych zbiorach danych. Dzięki temu raportowanie
dużych datasetów nie generuje zbędnych kosztów CPU, o ile limity próbkowania
nie wymuszają odrzucania większości obserwacji.

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
