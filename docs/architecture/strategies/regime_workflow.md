# RegimeSwitchWorkflow – propagacja metadanych strategii

`RegimeSwitchWorkflow` od wersji rozszerzonej o obowiązkowe metadane strategii
(
licencje, klasy ryzyka, wymagane dane, capability oraz tagi
autorskich pluginów) przekazuje komplet informacji o aktywowanych strategiach
w każdej decyzji reżimowej. Dzięki temu warstwa autotradera i raportowania może
budować podpisy decyzji spójne z katalogiem strategii.

## Zawartość decyzji reżimu

Każdy obiekt `RegimeSwitchDecision` zawiera obecnie:

- `strategy_metadata` – słownik nazw strategii i ich metadanych (licencje,
  capability, klasy ryzyka, wymagane dane, tagi),
- `license_tiers`, `risk_classes`, `required_data`, `capabilities`, `tags` –
  znormalizowane listy unikalnych wartości scalonych z aktywnych strategii.

Decyzja przekazywana do `AutoTradeEngine` trafia następnie do payloadów sygnałów
oraz statusów (`regime_update`, `entry_long`, `entry_short`). Operatorzy mogą
wykorzystać te dane do filtrowania raportów, walidacji licencyjnej i
monitorowania pokrycia danych.

## Aktualizacja pluginów strategii

Wbudowane pluginy (`TrendFollowingStrategy`, `DayTradingStrategy`,
`MeanReversionStrategy`, `ArbitrageStrategy`) zostały uzupełnione o
metadane kompatybilne z katalogiem strategii:

| Strategia             | License tier   | Klasy ryzyka                   | Wymagane dane                         | Capability         |
|-----------------------|----------------|--------------------------------|---------------------------------------|--------------------|
| trend_following       | `standard`     | `directional`, `momentum`      | `ohlcv`, `technical_indicators`       | `trend_d1`         |
| day_trading          | `standard`     | `intraday`, `momentum`         | `ohlcv`, `technical_indicators`       | `day_trading`      |
| mean_reversion       | `professional` | `statistical`, `mean_reversion`| `ohlcv`, `spread_history`             | `mean_reversion`   |
| arbitrage            | `enterprise`   | `arbitrage`, `liquidity`       | `order_book`, `latency_monitoring`    | `cross_exchange`   |

Metadane są wykorzystywane zarówno przez workflow, jak i fallbackowy tryb
autotradera, który agreguje wymagania licencyjne dla zestawu wag wynikowych.

## Integracja z autotraderem

`AutoTradeEngine` umieszcza metadane w:

- payloadzie sygnałów (`EventType.SIGNAL`) w sekcji `metadata`,
- statusach wejścia (`entry_long`, `entry_short`),
- aktualizacjach reżimu (`regime_update`).

Dzięki temu UI oraz mosty konfiguracyjne otrzymują takie same informacje,
jak scheduler i katalog strategii. Wszystkie listy w metadanych są
normalizowane i deduplikowane, aby ułatwić raportowanie.

## Testy regresyjne

Testy `tests/test_regime_switch_workflow.py` oraz
`tests/test_auto_trade_engine_native.py` weryfikują obecność i poprawność nowych
metadanych, chroniąc pipeline decyzyjny przed regresjami.
