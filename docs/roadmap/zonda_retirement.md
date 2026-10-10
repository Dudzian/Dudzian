# Roadmap — Complete Zonda Retirement

**Decision: PERMANENT PRODUCT EXCLUSION.**
**Implementation status: NOT YET EXECUTED — mandatory removal before beta.**
This decision is an explicit product-scope instruction, not a claim of independently verified exchange operational history. Zonda must **not** remain selectable, instantiable, packaged, enabled, recommended or described as supported by CryptoHunter. No new Zonda adapter work, trading capability or resurrection path is authorized.

## Removal scope

1. **Executable and discovery:** delete `bot_core/exchanges/zonda/` and imports/exports; remove native/CCXT registrations, built-in and dynamic factories, loader fallbacks, API-origin/mode discovery, CLI/onboarding choices, market data and execution routes, health/network/error mappings and metrics for Zonda. Unknown/retired venue IDs must fail closed. Do not leave a generic `primary` or legacy alias that silently redirects Zonda orders to another venue.
2. **Configuration and artifacts:** remove `config/exchanges/zonda.yaml`, Zonda account/environment/adapter entries and instrument-symbol route entries from active `config/core.yaml`, active marketplace presets/packages/catalog entries and their associated signatures, executable build/packaging resources, UI exchange selections, sample live configs, backfill utilities, scripts and active support reports. Rebuild signed catalogs/manifest artifacts using **authorized signing tooling**; never edit a signed payload without rebuilding/validating its signature and never fabricate a signer.
3. **Tests and documentation:** replace obsolete Zonda-only tests with retirement and negative-selection tests; adjust shared fixtures, multi-exchange tests, documentation and advertised exchange counts. Preserve historical documents/reports as **historical-only** with clear non-support status when deleting them would damage audit provenance.
4. **Existing persisted identities:** preserve past records, order/fill journals and financial audit history. Perform a reviewable versioned migration/retirement for Zonda `ExchangeAccount`, `CredentialProfile`, routes, instruments and external identifiers: disable creation/activation/trading and secret use; allow only controlled read-only historical display/cleanup. Never corrupt durable IDs or reinterpret old Zonda orders as another exchange.
5. **Frozen architecture contract:** do not rewrite historical frozen M0.5 snapshots. If canonical registries need a new current version to prevent active Zonda admission, do so through the approved contract/migration change process, keeping immutable historical fingerprints and their verification intact.

## Acceptance

- Source/build/packaged-app scan: no reachable Zonda adapter, network endpoint, market-data fetch, order/cancel command, selectable UI option, signed active marketplace product or registry entry permitting activation.
- Explicit `RETIRED_EXCHANGE`/unsupported behavior for old Zonda IDs, including restart/replay/restore and attempted cross-exchange fallback.
- Existing historical audit/account records remain verifiable, retained and read-only; migrated active accounts cannot place/cancel orders.
- All affected adapter/runtime/test suites and security scans pass; beta support matrix explicitly excludes Zonda.
- Final independent review confirms no live or test paths can instantiate Zonda in a release build.

**Implementation rule:** complete this as focused reviewed deletion/migration PRs. Roadmap acceptance alone is not code removal and must never be reported as completed decommissioning.
