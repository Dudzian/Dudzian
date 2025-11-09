# Exchange Integration Review

> **Legacy notice:** Historyczne źródła, które dawniej znajdowały się w
> `KryptoLowca/bot`, zostały całkowicie usunięte z repozytorium. Wszelkie prace
> integracyjne muszą koncentrować się na aktywnie wspieranych pakietach
> `bot_core.*` i cienkich shimach `KryptoLowca.*`. Pakiet
> `KryptoLowca.managers` pozostaje usunięty; nowy kod powinien korzystać z
> `bot_core.execution.live_router.LiveExecutionRouter` oraz
> `bot_core.exchanges.ccxt_adapter.CCXTSpotAdapter` (lub wyższych helperów w
> `KryptoLowca.exchange_adapter`).
>
> **2025-02 sanity sweep:** Warstwa zgodności re-eksportuje kanoniczne
> implementacje z `bot_core`.  Moduły takie jak `bot_core.ai.manager`
> delegują bezpośrednio do nowego runtime.  Użyj
> `python scripts/find_duplicates.py --json` to verify that no redundant
> implementations reappear when touching this area.

## Implemented Scope
- Unified adapter abstraction defined in `KryptoLowca/exchanges/interfaces.py`, with REST and long-poll helpers, rate limiting, and order handling primitives.
- Binance Testnet and Kraken Demo adapters leveraging shared base classes, with dependency injection points for HTTP and long-poll clients.
- Encrypted API key manager integrated with `ConfigManager` for credential storage, rotation, and compliance controls.
- Live execution router (`bot_core.execution.live_router.LiveExecutionRouter`) obsługujący trasowanie między adapterami oraz telemetryjne
  metryki tras wraz z fallbackiem dla profili ryzyka.
- ZondaAdapter dostępny w `KryptoLowca/exchanges/zonda.py` wraz ze smoke testem tickera i scenariuszem cyklu życia zlecenia w `KryptoLowca/tests/integration/test_demo_sandboxes.py`. Streaming long-poll pozostaje funkcją opcjonalną, domyślnie wyłączoną w produkcji, a `MarketDataPoller` w `KryptoLowca/exchanges/polling.py` zapewnia REST-owe odpytywanie dla finalnego GUI i progresywny backoff błędów, aby ograniczyć obciążenie API podczas awarii.
- Nowy Trading GUI korzysta z `MarketDataPoller` poprzez mostek REST, aktualizując ticker w interfejsie po kliknięciu „Start”, prezentując komunikaty o stanie połączenia (łączenie, aktywny ticker, błędy REST) i utrzymując kompatybilność z trybem sandbox/live. Domyślny symbol oraz interwał odpytywania można nadpisać zmiennymi środowiskowymi `TRADING_GUI_DEFAULT_SYMBOL` oraz `TRADING_GUI_MARKET_INTERVAL`.
- Encrypted API key manager oparty na `EncryptedFileSecretStorage` i adapterze `PresetConfigService`, zapewniający rotację i kontrolę zgodności.
- Multi-exchange account manager supporting round-robin order dispatch and monitoring across adapters.
- NowaGielda Spot long-poll streaming wspiera konfigurację przez sekcję `[stream]` w profilu adaptera: kluczowe opcje to `base_url`, `public_path`/`private_path`, filtry `public_symbols`/`private_symbols`, limity reconnectów (`reconnect_attempts`, `reconnect_backoff`, `reconnect_backoff_cap`) oraz rozmiar bufora (`buffer_size`). Fallback REST z `LocalLongPollStream` zapewnia utrzymanie ostatnich paczek podczas ponownego łączenia.
- `NowaGieldaStreamClient` udostępnia wskaźnik `closed` oraz może działać jako menedżer kontekstu (`with`), co gwarantuje domknięcie strumienia i zwolnienie bufora przy wyjątkach lub wczesnym zakończeniu pracy. Dodatkowo eksponuje właściwości `channels`, `remote_channels`, `scope`, `last_cursor`, `buffer_size`, `history_size`, `pending_size`, `max_reconnects`, `reconnect_attempt` oraz liczniki `reconnects_total`, `total_batches`, `total_events`, `heartbeats_received`. Wbudowana metoda `replay_history()` pozwala manualnie odtworzyć zbuforowane paczki (z opcjonalnym pominięciem heartbeatów) oraz wymusić ponowną emisję nawet podczas oczekiwania na świeże dane po reconnectach, `force_reconnect()` umożliwia natychmiastowe restartowanie streamu z opcjonalnym nadpisaniem kursora i decyzją o ponownej emisji historii, a `reset_counters()` zeruje lokalne liczniki diagnostyczne po wykonaniu pomiarów.

## Missing Scope / Follow-up Items
- Finalne GUI korzysta wyłącznie z odpytywania REST; ewentualne włączenie strumieni long-poll wymaga osobnej walidacji wydajnościowej i monitoringu reconnectów w stagingu.

## Test Coverage Notes
- Contract tests exist for adapter protocols i rotację kluczy, a zestaw integracyjny pokrywa cykl zlecenia Binance/Kraken oraz smoke ticker Zondy; potrzebne są jednak dalsze scenariusze integracyjne.
- Scenariusz cyklu życia zlecenia Zondy wymaga ustawienia flagi środowiskowej `ZONDA_ENABLE_ORDER_TESTS` oraz dostępnych poświadczeń API.

