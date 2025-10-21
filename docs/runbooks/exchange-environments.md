# Exchange environment switching runbook

This runbook describes how to move a portfolio between paper, testnet and
live exchanges.  The configuration is declarative – the same parameters
are consumed both by the runtime CLI and integration smoke tests.  The
reference YAML lives in `config/environments/exchange_modes.yaml`.

## Paper simulators

1. **Spot paper** – baseline behaviour for strategies that do not rely on
   leverage.  Enable it by setting:

   ```yaml
   exchange_manager:
     mode: paper
     paper_variant: spot
   ```

2. **Margin paper** – mirrors cross/isolated margin by tracking leverage,
   funding and liquidations.  Configure leverage limits and maintenance
   margin in the `simulator` section:

   ```yaml
   exchange_manager:
     mode: paper
     paper_variant: margin
     simulator:
       leverage_limit: 5.0
       maintenance_margin_ratio: 0.12
       funding_rate: 0.00005
   ```

3. **Futures paper** – extends the margin simulator with futures specific
   funding.  Use the `paper_variant: futures` flag and adjust the
   parameters when matching production risk settings.

Paper simulators emit the same account snapshot fields as native
adapters.  Funding payments and leverage changes are logged to the
telemetry bus so that dashboards can compare paper vs. live runs.

## Testnet (exchange sandbox)

* Set `mode: margin` with `testnet: true`.  The manager instantiates CCXT
  backends by default; for exchanges with native adapters the registry
  declares whether the testnet is available.  If a given exchange does
  not support it, the manager raises a configuration error during
  startup.
* Always provide dedicated API credentials.  Secrets can be passed via
  environment variables referenced in the YAML file.
* Watchdog policies should be more relaxed than in production to account
  for rate limiting on sandboxes.

## Live trading

* Set `mode: margin`/`mode: futures` without `testnet`.  The manager uses
  the native adapter registry to instantiate the correct implementation.
* Configure watchdogs and circuit breakers.  The sample YAML sets a
  three-attempt retry and a one minute recovery window.
* Provide exchange specific settings (e.g. Binance margin type) under the
  `native_adapter.settings` key.
* Telemetry: margin decisions are logged with the
  `order_close_for_reversal` and `margin_event` event types.

## Verification flow

1. Update the YAML profile with the desired environment.
2. Run smoke tests: `pytest tests/smoke -k exchange`.
3. For live/testnet make sure the account snapshot reports positive
   equity and available margin before enabling strategies.

This workflow ensures that multi-strategy runtimes experience identical
behaviour across paper and live deployments, enabling confident rollouts
of leverage-sensitive strategies.
