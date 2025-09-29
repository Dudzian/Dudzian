# Backfill danych OHLCV

Skrypt `scripts/backfill.py` automatyzuje pobieranie oraz odświeżanie danych OHLCV
z publicznych API giełd obsługiwanych przez platformę. Mechanizm korzysta z
`PublicAPIDataSource`, lokalnej pamięci podręcznej Parquet/SQLite oraz
harmonogramu `OHLCVRefreshScheduler`, dzięki czemu po pierwszym backfillu
możliwe jest cykliczne dogrywanie świeżych danych.

Domyślne częstotliwości odświeżania zależą od interwału (np. `1d` co 24 h,
`1h` co 15 min, `15m` co 5 min). W razie potrzeby można je nadpisać poprzez
sekcję `environments.*.adapter_settings.ohlcv_refresh_overrides` w
`config/core.yaml`, podając mapowanie `interwał -> sekundy` dla konkretnego
środowiska.

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
```

Domyślne progi bazują na dwukrotności długości interwału i są bezpieczne dla
środowiska demo/paper. Kanały alertowe (Telegram/e-mail/SMS) konfigurowane są
tak jak dla runtime – wymagają obecności sekretów w natywnym keychainie lub
zaszyfrowanym magazynie.

Każdy przebieg backfillu zapisuje ponadto wpisy audytowe luk do pliku
`<data_cache_path>/audit/<environment>_ohlcv_gaps.jsonl`, gdzie utrwalane są
ostatni znany znacznik czasu, liczba świec oraz status (`ok`, `warning`,
`incident`, `sms_escalated`). Plik jest w formacie JSONL i można go trzymać w
retencji ≥24 miesięcy na potrzeby audytu operacyjnego.
