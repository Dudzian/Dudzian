# Roadmap — Multi-Exchange & Multi-Account Production Readiness

**Status: PLANNED — mandatory before first beta.**
**Owners:** canonical M1.3/M1.4/M1.5/M1.6, M3.4/M3.5, M4 UI, PB beta gate.
**Scope:** working PAPER and real qualified TESTNET across simultaneously active independent exchange accounts; LIVE remains disabled until the separate M5 canary gates pass. A native adapter class, CCXT mapping, YAML preset or preview exchange selector does not prove a supported production exchange.

## MEX-1 — Adapters and support matrix (audit before implementation)

Inventory the source registry, dynamic/native factory registrations, CCXT paths, market types, instrument/symbol mapping, REST/WebSocket and private order/account APIs, API-key scopes, sandbox availability, reconciliation methods, rate limits, failure classes and test coverage. Classify every real venue plus each spot/margin/futures variant as **QUALIFIED**, **PARTIAL**, **CONTRACT_ONLY**, **UNSUPPORTED**, or **RETIRED** separately for PAPER, TESTNET and LIVE. Report explicit evidence and gaps; never market unverified support as production-ready. TESTNET must be confirmed per exact venue and product; where no genuine test environment exists, expose an unavailable/blocked state and never substitute real LIVE with test-sized orders.

**Zonda is RETIRED with zero target support.** See [Zonda retirement](zonda_retirement.md). Exclude it from active venue discovery, UI options, beta support matrix, and any future activation path.

## MEX-2 — Canonical account-scoped runtime

- Replace legacy `frontend.py` single `{"primary": adapter}` wiring with Core-owned composition of **multiple** explicitly selected `ExchangeAccount` identities.
- Bind `exchange_account_id`, `exchange_id`, `environment`, `market_type`, `portfolio_id`, `credential_profile_id`, adapter and execution/market-data routes deterministically; do not derive durable identity from display name or implicit `primary`.
- Support multiple portfolios, several accounts and distinct subaccounts **on the same venue** without ambiguous matching; keep data/public-feed and private/order routing independent.
- Preserve PAPER/TESTNET/LIVE isolation; no environment fallback or cross-account credential sharing; LIVE remains gated, and onboarding new adapter factories cannot expand frozen product capabilities.
- Prevent simultaneous strategies from double-reserving the same portfolio risk/capital. Enforce account limits, aggregate portfolio exposure, reservations and global kill switch atomically across parallel workers.

## MEX-3 — Trade lifecycle, safety and resilience

- Account-bound client-order-id and venue-order-id must survive retry, reconnect, process restart and reconciliation, including partial fills; ensure idempotence against uncertain timeout/network delivery.
- Review `LiveExecutionRouter.cancel()` missing-binding broadcast to all adapters: remove unsafe cross-venue cancellation behavior; resolve exact account-bound intent or fail closed and reconcile. The change must not allow ambiguous duplicate exchange order IDs to select the wrong account.
- Keep one intent's *venue selection/failover* distinct from *concurrent independent orders on separate venues*. Cross-venue fallback is opt-in only after unequivocal non-execution on the prior venue and fresh risk/authority checks; never split/send a single order to all venues without an explicitly approved split plan.
- Validate per-exchange rate limits, weighted requests, reconnect, circuit breaker, auth failure, clock drift, symbol precision/min notional, fee/currency conversion, margin/leverage and settlement differences.
- Recovery after disconnect, restart and crash must reconcile independent balances, orders and fills; stale or contradictory account state blocks only affected execution scopes where policy allows, with aggregate risk still protected.

## MEX-4 — Real qualification

- Independent deterministic unit/negative/property/fault-injection tests for the identity graph, cross-account isolation, permissions, route selection, multiple concurrent strategies, capital reservations, lifecycle and recovery.
- E2E with at least **two truly independent** exchange accounts/venues operating concurrently on real supported TESTNETs; extend to the approved beta-target set. Prioritize Binance, Kraken, Bybit and OKX **for evaluation**, not as an automatic promise of sandbox coverage.
- Verify live-capability claims only after separate M5 gate. No real-money calls or secret provisioning in CI.
- Each beta-supported venue must have an evidence pack covering protocol, private account identity, permitted API scopes, placement/cancellation/fill, retry/reconciliation, failure behavior and a Windows packaged-app smoke.
- A failed account/venue must not corrupt unrelated account state or silently migrate its order to another venue.

## MEX-5 — Product / UI handoff

Expose runtime-authoritative read models and typed Core commands for add account, validate public/private connection, inspect observed permissions, bind/rotate/retire credential profile, disconnect, inspect balances/open orders, assign exact strategy routes, view per-account diagnostics and aggregated portfolio status. The UI cannot read secrets back or submit orders outside canonical Core gates. See [Production UI overhaul](production_ui_overhaul.md).

## Exit evidence (required before first beta)

1. Support matrix approved; Zonda fully removed from active product paths.
2. Multiple account bootstrap reaches real Core, not a `primary` facade or preview-only registry.
3. Two independent TESTNET accounts/venues execute/reconcile concurrently on the canonical path; all proposed beta venues individually qualified.
4. Negative coverage of identity collisions, cross-environment routing, key leakage, cancellation, retry after uncertain execution, partial fills and global risk contention.
5. UI accounts page works without CLI/YAML editing and secrets never enter QML read models/logs.
6. Risk/capability and LIVE gates remain fail-closed; PB beta audit finds zero HIGH/BLOCKER issues.

**Not a M5 LIVE activation.** Qualification can start with PAPER, but no claim of TESTNET or LIVE readiness until matching external tests exist.
