# CryptoHunter Product Delivery Roadmap

Status: **canonical planning contract**

This document defines the delivery path from the current Product Architecture Contract work to a production-ready CryptoHunter v1. It is the planning source of truth for milestone sequencing, scope boundaries and exit criteria. Individual implementation plans, PRs and audits should reference the milestone and sub-step they advance.

## North-star outcome

CryptoHunter v1 is complete when it is a stable installable Windows desktop product with a durable background Core, PAPER and TESTNET end-to-end trading, guarded LIVE execution, multi-exchange/account routing, deterministic risk and accounting, secure identity/secrets, recovery, observability, updater/rollback and release-grade operational safety.

## Governing rules

1. **Contract before implementation.** M0 remains authoritative for target semantics. Runtime work may not silently redefine an M0 contract.
2. **Fail closed.** Missing, stale, ambiguous or unverifiable authority never opens trading, credentials, execution or release capabilities.
3. **PAPER before TESTNET; TESTNET before LIVE canary; canary before scale.** No milestone may bypass this order.
4. **One product model across environments.** PAPER, TESTNET and LIVE share the same domain model and core execution/accounting/risk semantics; only capabilities and adapters differ.
5. **Core owns authority.** UI, tray, raw caller input and preview/read models do not establish execution, security or lifecycle authority.
6. **Recovery is a feature, not a cleanup task.** Persistence, migrations, restart/rebuild, reconciliation and rollback must be proven before the corresponding environment is promoted.
7. **Existing code is classified, not protected.** Each reused subsystem is explicitly marked KEEP, ADAPT, REWRITE or DELETE based on compatibility with the canonical architecture, not sunk cost.
8. **Preview is not runtime evidence.** `preview_*`, fixtures and static read models can inform contracts/tests but cannot satisfy runtime acceptance criteria.
9. **Milestone exits require evidence.** A milestone closes only when its acceptance tests and audit are complete; feature presence alone is insufficient.
10. **Every substantial PR declares its roadmap target.** PR descriptions should state the milestone/sub-step and whether they change contract, runtime, migration, tests or release surfaces.

---

# M0 — Product Architecture Contract

**Purpose:** define the complete target product before reconstruction.

Current state at roadmap creation: **M0.1–M0.10 closed; M0.11–M0.14 remaining.**

## Remaining work

- **M0.11 — Persistence, versioning, migrations, backup and recovery**
- **M0.12 — Audit, observability, alerts and updater**
- **M0.13 — Final architecture contract and validators**
- **M0.14 — Closing audit and delivery handoff**

## M0 exit criteria

- All M0.1–M0.14 contracts are closed and mutually consistent.
- Machine-readable validators cover cross-contract identities/fingerprints and fail-closed invariants.
- LIVE remains blocked by policy; no architecture work silently activates runtime capabilities.
- A migration inventory maps existing implementation to KEEP / ADAPT / REWRITE / DELETE.
- M1 implementation backlog is generated from the final M0 contract.

---

# M1 — Core Runtime Reconstruction

**Purpose:** turn the M0 architecture into the canonical executable Core while preserving reusable foundations.

## M1.1 — Runtime topology and lifecycle

Implement and validate:

- CoreHost lifecycle and single-instance ownership
- DesktopShell ↔ Core reconnect semantics
- TrayAgent/background survival
- startup/shutdown intents
- autostart and first-run lifecycle
- IPC discovery, version compatibility and reconnect
- crash/restart behavior

## M1.2 — Canonical domain and persistence

Implement:

- durable IDs and canonical entities
- event/command persistence
- schema/version management
- migrations
- backup/restore
- deterministic rebuild
- corruption detection and fail-closed startup

## M1.3 — Accounts, exchanges and instruments

Implement:

- Exchange Registry
- multiple ExchangeAccounts
- CredentialProfiles and secret references
- instrument catalog and symbol aliases
- TradingUniverse
- environment-aware capability resolution

## M1.4 — Strategy, market-data and execution routing

Implement:

- StrategyDefinition / StrategyInstance runtime
- MarketDataRoute
- ExecutionRoute
- deterministic routing validation
- adapter lifecycle and health
- route failure handling

## M1.5 — Orders, fills and accounting

Implement:

- command/event pipeline
- order lifecycle
- idempotency and duplicate handling
- canonical Fill ingestion
- immutable ledger journal
- reservations/releases
- portfolio reconstruction
- FIFO P&L and valuation
- reconciliation primitives

## M1.6 — Risk and security runtime

Implement:

- risk hierarchy
- exact pre-trade checks
- kill switch and fencing
- ExecutionLease issuance/consumption
- identity/device trust
- PIN/biometric proof flow
- secret-reference access
- LiveAccessGrant semantics without enabling LIVE

## M1.7 — Migration audit

Classify old code and finish migration:

- KEEP — compatible and production-worthy
- ADAPT — reusable after contract alignment
- REWRITE — behavior retained but implementation replaced
- DELETE — legacy, duplicate or preview-only runtime candidate

### M1 exit criteria

- Core can start, persist state, restart and rebuild deterministically.
- Core boundaries match M0 contracts.
- Domain/account/routing/order/ledger/risk/security foundations execute in code, not preview models.
- LIVE remains closed.
- Legacy paths no longer create competing authority.

---

# M2 — PAPER End-to-End Product

**Purpose:** make PAPER the first fully real CryptoHunter environment.

## Required end-to-end path

Market data → Strategy → Decision → Risk → Reservation → Order → Paper execution → Fill → Ledger → Portfolio/P&L → Audit → UI projection.

## M2 workstreams

- persistent PAPER engine
- realistic fees/slippage/partial fills
- start/stop/resume strategies
- multi-account PAPER routing
- deterministic restart recovery
- ledger and portfolio rebuild
- reconciliation against paper execution state
- risk limits and kill switch
- operator controls
- audit and observability
- failure injection and long-running soak

### M2 exit criteria

- No fixture or preview component is required for the production PAPER path.
- Restart/crash does not lose or duplicate economic facts.
- Orders/fills/ledger/portfolio remain consistent after replay.
- Risk and kill switch are enforced by Core.
- PAPER passes defined soak and recovery tests.
- Desktop can operate PAPER end to end.

---

# M3 — TESTNET and Real Exchange Integration

**Purpose:** validate the system against real exchange protocols without risking production capital.

## M3.1 — Public market data

- real REST/WebSocket connectivity
- reconnect/backoff
- stale-data detection
- instrument metadata refresh
- clock/timestamp validation

## M3.2 — Private account connectivity

- credential validation
- balances
- open orders
- fills/trades
- account state reconciliation
- rate-limit handling

## M3.3 — Real execution on TESTNET

- submit/cancel/replace
- partial fills
- reject/error mapping
- idempotency across retries and reconnects
- exchange-order identity mapping
- recovery after process/network failure

## M3.4 — Multi-exchange / multi-account

- more than one real adapter path
- account-scoped routing
- instrument aliases
- venue-specific capability restrictions
- deterministic failover rules where allowed

### M3 exit criteria

- TESTNET trades execute through the canonical Core path.
- Reconnect/retry cannot create duplicate economic execution.
- Exchange state reconciles with local state after restart.
- At least the production-target exchange set required for v1 is validated.
- LIVE remains explicitly blocked.

---

# M4 — Desktop Productization

**Purpose:** turn the working Core into a coherent operator product.

## Product surfaces

- production DesktopShell
- tray/background lifecycle
- autostart
- Core connection/reconnect UX
- first-run onboarding
- account and credential setup
- exchange/instrument management
- strategy management
- orders/fills/history
- portfolio, P&L and capital
- risk configuration and kill switch
- alerts and audit views
- update/rollback UI
- environment visibility and explicit LIVE distinction

## Packaging

- canonical Windows entrypoint
- deterministic PyInstaller build
- QML/assets/dependency inventory
- artifact verification
- smoke tests
- version metadata
- installer/release packaging as selected by final architecture

### M4 exit criteria

- A non-developer can install and operate PAPER/TESTNET without CLI intervention.
- Closing the GUI behaves according to Core/Tray lifecycle contract.
- UI cannot manufacture privileged state or bypass Core gates.
- Build is repeatable and smoke-tested.

---

# M5 — LIVE Canary

**Purpose:** introduce production-capital execution under deliberately narrow authority.

## Promotion sequence

1. production read-only connectivity
2. account/credential validation
3. reconciliation stability
4. LIVE capability readiness
5. LiveAccessGrant
6. explicit operator authorization
7. tiny canary scope
8. observed canary soak
9. controlled scope expansion

## Initial LIVE constraints

- one approved account
- deliberately small capital/exposure
- narrow instrument universe
- strict per-order/per-position limits
- kill switch ready before activation
- complete audit trail
- immediate fail-closed on stale authority, stale market data, reconciliation failure or capability loss

### M5 exit criteria

- LIVE canary executes through the same order/risk/ledger path proven in PAPER/TESTNET.
- No alternate LIVE shortcut exists.
- Kill switch and fencing are independently verified.
- Recovery/reconciliation is proven with real production account state.
- Expansion requires explicit policy rather than implicit configuration.

---

# M6 — Production Hardening and v1 Release

**Purpose:** prove CryptoHunter is resilient enough to be treated as a production financial application.

## Hardening matrix

- process crash during order lifecycle
- abrupt machine restart
- corrupted persistence
- migration rollback/failure
- backup/restore
- network partitions
- DNS/HTTP/WebSocket failures
- exchange outage/degradation
- reconnect storms
- stale or missing market data
- duplicate/out-of-order exchange events
- partial fills and delayed fills
- reconciliation divergence
- credential revocation/rotation
- clock anomalies
- updater failure and rollback
- long-running soak
- resource leak monitoring
- security review
- release artifact integrity/signing as applicable

## Release readiness

- versioned release pipeline
- reproducible Windows artifacts
- automated smoke and upgrade tests
- rollback path
- release notes and migration notes
- operational runbook
- known-risk register

### M6 exit criteria — CryptoHunter v1

CryptoHunter v1 is considered complete only when:

- PAPER, TESTNET and guarded LIVE are supported through the canonical architecture.
- Multi-exchange/account scope required for v1 is production-validated.
- Persistence, recovery and reconciliation are demonstrated under failure.
- Security/risk gates cannot be bypassed by UI or caller input.
- Desktop packaging, update and rollback are release-grade.
- Soak/hardening acceptance criteria pass.
- Final production audit records remaining limitations explicitly.

---

# Progress model

Progress is reported at two levels:

1. **Milestone progress** — completed sub-steps / audited acceptance criteria inside the active milestone.
2. **Product progress** — weighted delivery progress toward M6/v1, not a naive count of documents or PRs.

A milestone is never marked complete merely because code exists. It closes only after its acceptance evidence and closing audit are merged.

# Immediate next action

Continue **M0.11**. Do not begin M1 runtime reconstruction until M0.14 closes the architecture and produces the migration/backlog handoff.
