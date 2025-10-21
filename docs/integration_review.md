# Exchange Integration Review

> **Stan na 2025-10:** Warstwa `archive/legacy_bot` została usunięta – wspierane
> uruchomienia korzystają wyłącznie z modułów `KryptoLowca.*` oraz `bot_core.*`.
> Pakiet `KryptoLowca.managers` pozostaje wygaszony; do usług środowiska należy
> używać top-levelowych modułów (`KryptoLowca.exchange_manager`,
> `KryptoLowca.report_manager`, `KryptoLowca.risk_manager`, itd.) wraz z
> komponentami runtime z `bot_core` (np.
> `bot_core.execution.live_router.LiveExecutionRouter`,
> `bot_core.exchanges.ccxt_adapter.CCXTSpotAdapter`).  Skrypt
> `python scripts/find_duplicates.py --json` nadal pomaga identyfikować
> zdublowane implementacje i powinien być uruchamiany po większych refaktorach.

## Implemented Scope
- Unified adapter abstraction defined in `KryptoLowca/exchanges/interfaces.py`, with REST and WebSocket helpers, rate limiting, and order handling primitives.
- Binance Testnet and Kraken Demo adapters leveraging shared base classes, with dependency injection points for HTTP and WebSocket clients.
- Encrypted API key manager integrated with `ConfigManager` for credential storage, rotation, and compliance controls.
- Live execution router (`bot_core.execution.live_router.LiveExecutionRouter`) obsługujący trasowanie między adapterami oraz telemetryjne
  metryki tras wraz z fallbackiem dla profili ryzyka.
- ZondaAdapter dostępny w `KryptoLowca/exchanges/zonda.py` wraz ze smoke testem tickera i scenariuszem cyklu życia zlecenia w `KryptoLowca/tests/integration/test_demo_sandboxes.py`. Streaming WebSocket pozostaje funkcją opcjonalną, domyślnie wyłączoną w produkcji, a `MarketDataPoller` w `KryptoLowca/exchanges/polling.py` zapewnia REST-owe odpytywanie dla finalnego GUI i progresywny backoff błędów, aby ograniczyć obciążenie API podczas awarii.
- Nowy Trading GUI korzysta z `MarketDataPoller` poprzez mostek REST, aktualizując ticker w interfejsie po kliknięciu „Start”, prezentując komunikaty o stanie połączenia (łączenie, aktywny ticker, błędy REST) i utrzymując kompatybilność z trybem sandbox/live. Domyślny symbol oraz interwał odpytywania można nadpisać zmiennymi środowiskowymi `TRADING_GUI_DEFAULT_SYMBOL` oraz `TRADING_GUI_MARKET_INTERVAL`.
- Encrypted API key manager oparty na `EncryptedFileSecretStorage` i adapterze `PresetConfigService`, zapewniający rotację i kontrolę zgodności.
- Multi-exchange account manager supporting round-robin order dispatch and monitoring across adapters.

## Missing Scope / Follow-up Items
- Finalne GUI korzysta wyłącznie z odpytywania REST; ewentualne włączenie strumieni WebSocket wymaga osobnej walidacji wydajnościowej i monitoringu reconnectów w stagingu.

## Test Coverage Notes
- Contract tests exist for adapter protocols i rotację kluczy, a zestaw integracyjny pokrywa cykl zlecenia Binance/Kraken oraz smoke ticker Zondy; potrzebne są jednak dalsze scenariusze integracyjne.
- Scenariusz cyklu życia zlecenia Zondy wymaga ustawienia flagi środowiskowej `ZONDA_ENABLE_ORDER_TESTS` oraz dostępnych poświadczeń API.

