# Backfill danych OHLCV

Skrypt `python scripts/backfill.py` automatyzuje pobieranie oraz odświeżanie danych OHLCV
z publicznych API giełd obsługiwanych przez platformę. Mechanizm korzysta z
`PublicAPIDataSource`, lokalnej pamięci podręcznej Parquet/SQLite oraz
harmonogramu `OHLCVRefreshScheduler`, dzięki czemu po pierwszym backfillu
możliwe jest cykliczne dogrywanie świeżych danych.

Domyślne częstotliwości odświeżania zależą od interwału (np. `1d` co 24 h,
`1h` co 15 min, `15m` co 5 min), a harmonogram dodaje losowy jitter, aby nie
odpytywać wszystkich giełd w tym samym momencie (domyślnie ±15 min dla `1d`,
±2 min dla `1h`, ±45 s dla `15m`). W configu `core_multi_exchange` utrzymujemy
pełną historię dla `1d` (10 lat), skróconą dla `1h` (5–3 lata) oraz lekkie
sanity-checki `15m` (ok. 6 miesięcy). W razie potrzeby można je nadpisać poprzez
sekcję `environments.*.adapter_settings.ohlcv_refresh_overrides` w
`config/core.yaml`, podając mapowanie `interwał -> sekundy` dla konkretnego
środowiska. Analogicznie jitter można skonfigurować przez
`environments.*.adapter_settings.ohlcv_refresh_jitter` (wartość w sekundach,
np. `1h: 180`), aby dopasować rozkład zapytań do ograniczeń API.

## Obsługiwane giełdy

Aktualna konfiguracja `core_multi_exchange` obejmuje następujące adaptery
publicznych API:

- `binance_spot` – `BinanceSpotAdapter`
- `binance_futures` – `BinanceFuturesAdapter`
- `kraken_spot` – `KrakenSpotAdapter`
- `kraken_futures` – `KrakenFuturesAdapter`
- `zonda_spot` – `ZondaSpotAdapter`

Skrypt automatycznie dobiera właściwy adapter na podstawie pola `exchange`
ze wskazanego środowiska w `config/core.yaml`.

## Wymagania sieciowe

Część giełd (w szczególności Binance i Kraken) wymaga skonfigurowania listy
zaufanych adresów IP (IP allowlist) dla kluczy API wykorzystywanych nawet do
publicznych zapytań REST. Upewnij się, że każdy wpis `environments.*.ip_allowlist`
w `config/core.yaml` obejmuje adres hosta, na którym uruchamiany jest backfill,
w przeciwnym razie żądania mogą być odrzucane.

## Uruchomienie

Przykładowe uruchomienie pełnego backfillu dla środowiska `binance_paper`:

```bash
python scripts/backfill.py --environment binance_paper --run-once
```

Dla pracy ciągłej (backfill + inkrementalne odświeżanie) pomiń flagę `--run-once`
i pozostaw proces działający w tle.

Jeśli chcesz jedynie zweryfikować zakres danych przed startem, użyj flagi
`--plan-only`. Skrypt wypisze wówczas plan synchronizacji (interwały, liczba
symboli, daty początkowe, częstotliwość odświeżania) i zakończy działanie bez
wysyłania żądań do API ani sięgania po sekrety.

## Monitoring luk danych i alerty

Skrypt potrafi monitorować manifest SQLite i wysyłać alerty o długotrwałych
lukach w danych OHLCV. Aby aktywować mechanizm, uruchom go z flagą
`--enable-alerts`. W środowiskach headless (Linux bez środowiska graficznego)
należy dodatkowo przekazać `--headless-passphrase` (oraz opcjonalnie
`--headless-secrets-path`), aby `create_default_secret_storage` mogło otworzyć
zaszyfrowany magazyn sekretów.

Polityka eskalacji jest konfigurowalna poprzez sekcję
`environments.*.adapter_settings.ohlcv_gap_alerts` w `config/core.yaml`.
Przykład:

```yaml
ohlcv_gap_alerts:
  warning_gap_minutes:
    1d: 1800   # ostrzeżenie po ~30 godzinach braku świec dziennych
    1h: 90     # ostrzeżenie po 90 minutach ciszy na interwale godzinowym
    15m: 20    # ostrzeżenie po 20 minutach dla sanity-checków
  incident_threshold_count: 5   # liczba ostrzeżeń w oknie, po której otwieramy incydent
  incident_window_minutes: 10   # szerokość okna przesuwnego na eskalację (Telegram + e-mail)
  sms_escalation_minutes: 15    # czas trwania incydentu po którym uruchamiamy SMS
  warning_throttle_minutes: 5   # minimalny odstęp pomiędzy ostrzeżeniami dla tego samego symbolu
```

Domyślne progi bazują na dwukrotności długości interwału i są bezpieczne dla
środowiska demo/paper. Kanały alertowe (Telegram/e-mail/SMS) konfigurowane są
tak jak dla runtime – wymagają obecności sekretów w natywnym keychainie lub
zaszyfrowanym magazynie.

Każdy przebieg backfillu zapisuje ponadto wpisy audytowe luk do pliku
`<data_cache_path>/audit/<environment>_ohlcv_gaps.jsonl`, gdzie utrwalane są
ostatni znany znacznik czasu, liczba świec oraz status (`ok`, `warning`,
`warning_suppressed`, `incident`, `sms_escalated`). Plik jest w formacie JSONL
i można go trzymać w retencji ≥24 miesięcy na potrzeby audytu operacyjnego.

### Raportowanie luk z pliku audytu

Do szybkiej inspekcji bieżącego stanu luk służy skrypt
`python scripts/gap_audit_report.py`, który wczytuje plik JSONL i agreguje wpisy po
symbolu/interwale. Podstawowe użycie:

```bash
python scripts/gap_audit_report.py \
  data/cache/audit/binance_paper_ohlcv_gaps.jsonl \
  --environment binance_paper \
  --since-hours 24
```

Wynik zawiera tabelę z ostatnim statusem, wielkością luki (minuty), liczbą
wierszy w cache oraz liczbą ostrzeżeń/incydentów/SMS w zadanym oknie
czasowym (domyślnie 24 h). Parametr `--window-hours` pozwala zmienić szerokość
tego okna do własnych potrzeb operacyjnych.

### Raport zdrowia manifestu SQLite

W sytuacjach, gdy potrzebna jest szybka inspekcja aktualnego stanu cache bez
sięgania do logów audytowych, można użyć skryptu
`python scripts/manifest_gap_report.py`. Narzędzie odczytuje manifest SQLite,
porównuje ostatnie stemple czasowe z bieżącą godziną i stosuje progi ostrzeżeń
zdefiniowane w `ohlcv_gap_alerts.warning_gap_minutes` (a w razie braku –
domyślnie dwukrotność interwału). Raport można otrzymać jako tabelę tekstową
lub JSON, np.:

```bash
python scripts/manifest_gap_report.py \
  --environment binance_paper \
  --as-of 2024-05-20T12:00:00Z
```

Wynik pokazuje każdy symbol/interwał wraz z liczbą wierszy, ostatnim
timestampem, długością luki oraz statusem (`ok`, `warning`, `missing_metadata`,
`invalid_metadata`). Dzięki temu łatwo wychwycić instrumenty pominięte w
backfillu lub manifesty z uszkodzonymi danymi, zanim pipeline trafi na
środowisko paper/live.

Jeżeli backfill uruchomiono z flagą `--enable-alerts`, raport manifestu jest
generowany automatycznie po zakończeniu synchronizacji. Wszelkie wykryte
nieprawidłowości trafiają na kanały Telegram/e-mail/SMS zgodnie z polityką
alertów: braki metadanych i uszkodzone wpisy eskalują się jako krytyczne
powiadomienia, a długotrwałe luki (`warning`) wysyłane są jako ostrzeżenia z
pełnym kontekstem (ostatni timestamp, liczba świec, długość luki). Dzięki temu
operacje otrzymują komplet informacji o stanie cache natychmiast po
backfillu – jeszcze zanim harmonogram odświeżania rozpocznie kolejne cykle.

W razie potrzeby można również poprosić skrypt o wydruk raportu manifestu do
STDOUT, ustawiając flagę `--manifest-report-format` na `table` (tabelaryczny
podgląd) lub `json` (struktura przyjazna automatyzacji). Opcja działa tylko w
trybie wykonania backfillu (nie łączy się z `--plan-only`) i zawiera zarówno
poszczególne wpisy manifestu, jak i zagregowane podsumowanie statusów.
