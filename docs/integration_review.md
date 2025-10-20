# Exchange Integration Review

> **Legacy notice:** Historical sources once kept in `KryptoLowca/bot` now live under
> `archive/legacy_bot/` and are excluded from the supported runtime. All integration
> work must target the main `KryptoLowca` package. The legacy module
> The legacy `KryptoLowca.managers` package has been removed; new code must rely on
> the top-level `KryptoLowca.*` modules together with
> `bot_core.execution.live_router.LiveExecutionRouter` and
> `bot_core.exchanges.ccxt_adapter.CCXTSpotAdapter` (or higher-level helpers in
> `KryptoLowca.exchange_adapter`).

## Implemented Scope
- Unified adapter abstraction defined in `KryptoLowca/exchanges/interfaces.py`, with REST and WebSocket helpers, rate limiting, and order handling primitives.
- Binance Testnet and Kraken Demo adapters leveraging shared base classes, with dependency injection points for HTTP and WebSocket clients.
- Encrypted API key manager integrated with `ConfigManager` for credential storage, rotation, and compliance controls.
- Live execution router (`bot_core.execution.live_router.LiveExecutionRouter`) obsługujący trasowanie między adapterami oraz telemetryjne
  metryki tras wraz z fallbackiem dla profili ryzyka.
- ZondaAdapter dostępny w `KryptoLowca/exchanges/zonda.py` wraz ze smoke testem tickera i scenariuszem cyklu życia zlecenia w `KryptoLowca/tests/integration/test_demo_sandboxes.py`. Streaming WebSocket pozostaje funkcją opcjonalną, domyślnie wyłączoną w produkcji, a `MarketDataPoller` w `KryptoLowca/exchanges/polling.py` zapewnia REST-owe odpytywanie dla finalnego GUI.
- Nowy Trading GUI korzysta z `MarketDataPoller` poprzez mostek REST, aktualizując ticker w interfejsie po kliknięciu „Start”, prezentując komunikaty o stanie połączenia (łączenie, aktywny ticker, błędy REST) i utrzymując kompatybilność z trybem sandbox/live. Domyślny symbol oraz interwał odpytywania można nadpisać zmiennymi środowiskowymi `TRADING_GUI_DEFAULT_SYMBOL` oraz `TRADING_GUI_MARKET_INTERVAL`.

## Missing Scope / Follow-up Items
- Finalne GUI korzysta wyłącznie z odpytywania REST; ewentualne włączenie strumieni WebSocket wymaga osobnej walidacji wydajnościowej i monitoringu reconnectów w stagingu.

## Test Coverage Notes
- Contract tests exist for adapter protocols i rotację kluczy, a zestaw integracyjny pokrywa cykl zlecenia Binance/Kraken oraz smoke ticker Zondy; potrzebne są jednak dalsze scenariusze integracyjne.
- Scenariusz cyklu życia zlecenia Zondy wymaga ustawienia flagi środowiskowej `ZONDA_ENABLE_ORDER_TESTS` oraz dostępnych poświadczeń API.

