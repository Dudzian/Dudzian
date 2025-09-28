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
