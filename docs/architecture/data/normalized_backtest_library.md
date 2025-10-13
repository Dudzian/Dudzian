# Biblioteka znormalizowanych danych backtestowych

## Cel
Biblioteka w katalogu `data/backtests/normalized` dostarcza spójne, ręcznie zweryfikowane
zestawy OHLCV/latency dla strategii Mean Reversion, Volatility Targeting oraz
Cross-Exchange Arbitrage. Dane są zsynchronizowane do interwału 1m w strefie czasowej
UTC, co pozwala na replikowalne testy regresyjne w pipeline’ach demo→paper.

## Struktura
- `manifest.yaml` – definicja schematów, interwałów oraz wymagań jakościowych.
- `mean_reversion.csv` – próbka BTC-USD do kalibracji z-score i limitów wolumenu.
- `volatility_target.csv` – próbka ETH-USD z polami realizowanej/targetowanej zmienności.
- `cross_exchange_arbitrage.csv` – dane spreadu DEMEX↔GLOBEX dla detekcji okazji arbitrażowych.

## Walidacja
Skrypt `scripts/validate_backtest_datasets.py` wykonuje komplet kontroli:
- zgodność nagłówków i typów z manifestem,
- monotoniczne znaczniki czasu z krokiem równym interwałowi,
- dodatnie ceny, nieujemne wolumeny, przedziały dla pól ograniczonych,
- poprawność par bid/ask i spójność świec OHLC.

Wyniki raportowane są w formacie JSON (issues + metadane), a testy `tests/test_backtest_dataset_library.py`
blokują regresje schematu.

## Integracja
Biblioteka jest ładowana poprzez `bot_core.data.BacktestDatasetLibrary`, a operacyjna procedura
walidacji jakości danych (zadanie 6.4) wykorzystuje `DataQualityValidator`. W przypadku rozbudowy
danych należy dopisać nowy wpis w `manifest.yaml`, uzupełnić sekcję `checks` oraz dodać test regresyjny.
