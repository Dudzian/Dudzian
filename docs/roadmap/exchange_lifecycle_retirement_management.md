# Roadmap — Exchange Lifecycle & Retirement Management (ELR-1–ELR-7)

**Status: PLANNED / NOT IMPLEMENTED — mandatory first-beta M1.8, M3.5, M4.2/M4.3 and PB acceptance gate.**
**Product decision (2026-10-15):** replace the previous requirement for physical Zonda adapter deletion with one reusable, Core-authoritative mechanism that blocks inactive/unsupported/retired exchanges. Apply it to **Zonda**, **BitMEX** and every future closure/delisting. Do not claim the mechanism exists merely because a config flag is set; no trading, networking, release or signed-catalog modification is authorized by this planning document.

## Product intent and non-goals

- An exchange that is retired cannot be used as an exchange by CryptoHunter in any execution environment or through any discovery, routing, onboarding, background-job, strategy, AI, dynamic plugin, UI or signed active marketplace entry.
- **Adapter source code does not have to be deleted.** Safe inert source and historical fixtures may remain to minimize churn and preserve compatibility. Shipping an active factory, loading an unavailable CCXT class during startup, making upstream API requests, or showing an operational connector is forbidden for retired venues. If isolation is not provably safe/maintainable, remove that specific active integration or dependency; mandatory blanket source deletion is not the objective.
- Preserve original venue identifiers, immutable historic orders/fills, audit evidence and signed frozen contracts; enable read-only local history and offline deterministic replay only.
- This mechanism is not a substitute for capability qualification, account isolation, risk controls, LIVE authorization, native Windows release qualification or maintaining supported API integrations.

## ELR-1 — Status model, identity and authoritative policy

Establish a versioned Core-owned **ExchangeLifecyclePolicy** bound to canonical venue IDs and all known spot/margin/futures aliases, account/subaccount IDs and historical external IDs. Separate operational status from support qualification and per-market/per-environment capabilities.

| Lifecycle status | Meaning and policy |
| --- | --- |
| `active` | Eligible for operations only when independently qualified/authorized for the exact product, account and environment; not an automatic LIVE grant. |
| `disabled` | Temporary operator/security suspension: no new account activation, network trading, order/cancel, new routes or automated venue tasks. Controlled non-trading remediation needs a separate audited authority. |
| `deprecated` | Phase-out state: no new onboarding or trading capability; legacy clients receive an explicit blocked reason while migration proceeds. |
| `retired` | Permanently excluded from active product: no adapter instantiation or network connection, public/private feed, order/cancel, fallback, job, AI strategy, LIVE/TESTNET/PAPER connector activation or active Marketplace exposure. Local historical reads/offline replay remain permitted. |

`UNSUPPORTED`, `PARTIAL`, `QUALIFIED` and similar labels belong to the **independent support/capability matrix**, not to the lifecycle enum. Unknown/unqualified or ambiguous venue identity must fail closed.

Existing loader/model/runtime supports only `active`, `disabled` and `deprecated`, with partial LIVE-route enforcement. Extend and version it; accept old `disabled` config without weakening its meaning. Put permanent `retired` policy in an immutable, release-controlled current registry/contract; runtime/local YAML and dynamic plugins may tighten, never downgrade or revive it. A future reopening is a new independently reviewed product decision and contract version, **not** a local status toggle. Preserve frozen M0 fingerprints and records.

## ELR-2 — Core enforcement at every capability boundary

Apply the lifecycle decision before **factory import/adapter instantiation**, secret retrieval, account activation, public/private REST or WebSocket/long-poll, market-data collection, new/cancel/replace orders, reconciliation that would contact the venue, backfill, scheduled jobs and AI/strategy activation. Fail closed at the ultimate execution boundary as defense in depth; a UI or config filter alone is inadequate. Avoid crashes at startup from removed third-party CCXT attributes by never importing a retired adapter into active bootstrap or tooling.

Audit and guard all native/CCXT/plugin factories, `ExchangeManager`, bootstrap, live/paper/testnet services, command handlers, stream gateway, `primary` legacy alias, per-venue override, route fallback, restart/recovery/replay/restore and dynamic registry mutation. Never silently redirect a retired venue's order or account data to another venue. Historical/offline simulation reading locally stored data is not an online adapter.

Emergency operational suspension must take precedence over lower-trust configuration and immediately block new operations across existing sessions, including after status refresh, reconnect and process restart. No default-to-active when lifecycle metadata is missing for an unknown venue.

## ELR-3 — Accounts, credentials, orders and safe migration

Provide a reviewable, idempotent, versioned migration for existing persisted `ExchangeAccount`, `CredentialProfile`, instruments, routes, jobs and strategy bindings. Retire existing Zonda and BitMEX accounts without deleting/renumbering audit records or conflating venue IDs. Disable credentials for trading, revoke local activation privileges and explain to operators that any external secret/key cleanup must be handled separately. Do not log or leak secrets.

If suspension/retirement occurs with unresolved balances or orders, freeze automated execution, preserve exact state and provide an audited operator runbook for externally managing the position; do **not** assume a blocked exchange can safely receive cancel/reduce-only requests. Any future exceptional on-venue remediation requires an explicit narrow security review and separate authorization, never a generic fallback.

## ELR-4 — Product, discovery and release surfaces

- Expose authoritative lifecycle status/reason through typed Core read models, including read-only archived account history.
- Hide retired venues from supported *connectable* UI selectors, account-creation, credential-setup, environment options and strategy onboarding; disabled/deprecated venues show clear blocked reasons without operational actions.
- Withdraw active signed marketplace presets/packages/catalog entries that promote retired trading; preserve historical artifacts as labelled archives and re-sign changed active catalogs/manifests with authorized signing tools. Never hand-edit signatures.
- Exclude retired venues from required-active adapter counts, live gateway startup, HyperCare live snapshots and CI readiness requirements. Show historical rows with `retired` instead of fabricated healthy metrics. Preserve relevant old evidence without counting it as current qualification.
- Validate build/install packages, dynamic imports and plugin discovery so inert source can remain without introducing runtime or dependency failures.

## ELR-5 — Operator retirement playbook for future venues

1. Record independently verified incident/closure evidence, effective date, affected market/account environments and a signed product decision. A network hiccup alone is not proof of permanent closure.
2. Immediately apply a stricter Core-wide emergency `disabled` block if needed; stop new orders/auto-retries and isolate pending account state safely.
3. After approval, ship versioned `retired` state, migrate persisted identities and credentials, remove active catalog/GUI exposure and exclude the venue from active metrics/CI expectations.
4. Notify affected operators, document any off-platform balance/position steps and preserve audit evidence. Any restoration of product support needs a new reviewed release and qualification, never a local toggle.

## ELR-6 — Initial case inventory and interoperability

- **Zonda:** previous Zonda-specific deletion roadmap is superseded, not an implementation PASS. Its `zonda_spot: disabled` LIVE guard does not prove PAPER/TESTNET/factory/UI/Marketplace denial. Preserve its historical data and keep adapter code inert.
- **BitMEX:** quarantine both `bitmex_spot` and `bitmex_futures`, all aliases/accounts and PAPER/TESTNET/LIVE environments. Audit `bot_core/exchanges/bitmex/`, `bot_core/runtime/bootstrap.py`, `bot_core/exchanges/manager.py`, `config/core.yaml`, `config/exchanges/bitmex.yaml`, `scripts/check_required_adapter_factories.py`, signed Marketplace packages/presets and exchange report tooling. [PR #3124](https://github.com/Dudzian/Dudzian/pull/3124) addresses only scheduled monitoring, not runtime retirement.
- **Future venues:** no venue-specific mandatory delete/migrate roadmap section required; create per-venue decision/evidence entries under this general policy.

## ELR-7 — Negative evidence, tests and first-beta exit criteria

1. Unit/contract tests: all statuses and transitions; canonical ID/alias normalization; unknown ID, malformed/missing policy, stale rollback, lower-trust reactivation and unsupported capability rejected. Signed immutable retirement registry and historical fingerprints independently verified.
2. Integration/E2E: Zonda and BitMEX never instantiate/connect/submit/cancel/fetch/reconcile online in PAPER, TESTNET or LIVE, including dynamic plugins, retry/failover, `primary`, restart, replay and restore. Unrelated approved venues remain functional. No accidental network calls, including on startup.
3. Persistence: old Zonda/BitMEX accounts and immutable order/fill/ledger records survive migration and backup/restore unchanged, with only authorized read-only history/offline replay.
4. Product: operational GUI, command API and installer cannot activate a retired venue; active Marketplace artifacts cannot promote it; signatures and release manifests validate; historical reports correctly say `retired`.
5. CI: tests, architecture checks, package smoke, negative active-surface scans and independent beta security review PASS without skip/xfail, fake telemetry or disabled unrelated quality gates.
6. Documentation and support matrix name Zonda and BitMEX `RETIRED`, show the correct dates/reasons once verified, and provide a future-closure runbook.

**Exit status remains PLANNED** until reviewed implementation PRs provide the evidence above. This is a generalized **pre-first-beta blocker** replacing physical deletion; it does not mark M1/M3/M4/PB or Windows Stage 9/10 complete.
