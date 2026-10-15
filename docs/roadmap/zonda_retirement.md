# Roadmap — Zonda Retirement (superseded case-specific plan)

**Status: SUPERSEDED AS A DELETION REQUIREMENT; RETIREMENT IMPLEMENTATION STILL PLANNED.**

The earlier instruction to delete all Zonda adapter source code before beta is no longer the product requirement. The mandatory replacement is [Exchange Lifecycle & Retirement Management](exchange_lifecycle_retirement_management.md), tracked in canonical **M1.8**. This page is retained to preserve the Zonda-specific decision history and a stable documentation link.

**Current product decision:** Zonda is permanently `retired`, not supported. Its existing code may be preserved as an inert, uninstantiable historical implementation if negative evidence proves that it cannot be reached by active runtime, dynamic factories, CLI, UI, jobs, network fetching, signed active marketplace packages, routing or any PAPER/TESTNET/LIVE path. This is *not* an exception allowing dormant code to load unsafe removed dependencies or to be activated by changing local configuration.

- Enforce Core-owned, release-controlled immutable retirement even if older configs or aliases request `active`; unknown exchange IDs and unqualified capabilities fail closed.
- Block creating/connecting active Zonda accounts, handling secrets for trading, order placement/cancellation, market-data networking, retries/failover, simulated-exchange activation and automation.
- Preserve durable ExchangeAccount/credential references, historical orders/fills/journals, audit signatures and frozen architecture snapshots through reviewed versioned migration; make historical data read-only.
- Exclude Zonda from supported venue selection, active marketplace catalogs and shipped operational presets. Where signed artifacts change, regenerate them only using authorized signing tools; historical signatures remain untouched.
- Replace the old deletion-only acceptance tests with startup/packaging/restore/alias/factory/routing denial tests, including negative PAPER/TESTNET/LIVE tests and proof that unrelated approved venues remain operational.

**Exit status:** NOT IMPLEMENTED until the generic policy and Zonda migration pass the [M1.8 acceptance suite](exchange_lifecycle_retirement_management.md). Roadmap edits alone do not block Zonda in every runtime path.
