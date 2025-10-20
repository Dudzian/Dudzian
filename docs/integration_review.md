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

## Missing Scope / Follow-up Items
- No adapter implementation for the Zonda exchange within `KryptoLowca/exchanges/`.
- Lacks integration test suite (`pytest.mark.integration`) against real demo sandbox endpoints under `KryptoLowca/tests/integration/`.
- WebSocket factories default to stubs; real streaming implementations and accompanying tests are absent.

## Test Coverage Notes
- Contract tests exist for adapter protocols and API key rotation; however, sandbox-level integration tests are not present.

