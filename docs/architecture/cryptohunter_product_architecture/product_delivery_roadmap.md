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

## Windows Stage 9 → Stage 10 preflight — S9-S10-PRE-01

**Status: PLANNED.** This is a mandatory operational handoff **after Stage 9 closure and before Stage 10 live reboot / 24-hour qualification**. Preparing the infrastructure may begin earlier, but preparation must not activate Stage 10 or bypass legal production enrollment. This prerequisite does not add or close a numbered Windows Stage 0–14 acceptance item and does not change the M0–M6 product milestone taxonomy.

- **Dedicated physical Windows runner:** provision and harden a persistent Windows x64 GitHub Actions self-hosted runner with exact labels `self-hosted`, `Windows`, `X64`, `cryptohunter-stage10`. Establish dedicated host ownership, minimal GitHub/OS permissions, runner startup after reboot, host continuity and isolation from general PR/untrusted workloads.
- **Qualified MSI transfer:** define a reproducible, operator-approved procedure that retrieves `windows-stage9-qualified-msi` from the **same canonical workflow run** as the Stage 10 qualification, together with `installer-manifest.json`, `windows-clean-install-receipt.json` and `windows-clean-install-evidence.json`. Check the artifact digest, exact MSI/payload hashes, source revision, CI run ID, Production Trust identity and revision-bound evidence; reject stale, mixed-run, substituted or locally rebuilt installers.
- **Target-host deployment:** document and implement a reviewed, explicitly authorized elevated installation/handoff on that runner. Verify that the installed `CryptoHunterBackend.exe`, installed product version, Windows SCM service, runtime readiness and PostgreSQL/ACL boundaries match the qualified manifest and current-run provenance. Never silently replace an enrolled installation or treat a GitHub-hosted clean-install as proof that the self-hosted target has the same installed MSI.
- **Non-destructive rehearsal:** before enabling the destructive workflow, verify runner registration/labels and post-reboot startup configuration, artifact availability and hash lineage, administrative approval path, logging/evidence collection and read-only host preflight. Do not perform installation, reboot, TPM/NV mutations, enrollment or 24-hour soak as part of roadmap preparation alone.
- **Explicit Stage 10 gate:** actual lifecycle execution still requires completed legal production enrollment, satisfied readiness prerequisites and a separate operator-approved `workflow_dispatch` with `run_stage10=true`. The existing Stage 10 prepare job downloads clean-install evidence but does **not** install the MSI onto the self-hosted runner; that delivery gap must be resolved and reviewed before any reboot qualification.

**Exit criteria:** approved runner and MSI handoff runbook; dedicated runner identity and persistence setup verified without reboot; repeatable hash/provenance verification and fail-closed mismatches demonstrated; a reviewed plan for installing the exact current-run qualified MSI on the self-hosted machine; and independent sign-off that the live restart and 24-hour soak remain disabled until their separate authorization. This gate is **PREPARED**, not Stage 10 `PASS`.

## Numbered capabilities after Stage 10 — Stage 11 and Stage 12

**Status: PLANNED; numbering formalized, implementation not started.** These are product capability stages ordered after the Windows Stage 10 live lifecycle qualification. They do **not** renumber or extend frozen Windows Stage 0–14 acceptance items, or replace the canonical M0–M6 product milestone taxonomy. Neither stage may claim readiness solely from design documents, existing preview code or the presence of a workflow.

1. **Stage 11 — Trading Intelligence Hardening (TIH-1–TIH-7).** After Stage 10 closure and platform stabilization, harden regime classification, market microstructure and reversal intelligence, execution/TCA, anti-overfitting validation, live edge-decay monitoring, capital allocation and L2/trades replay. Detailed scope and acceptance criteria: [Stage 11 roadmap](../../roadmap/trading_intelligence_hardening.md).
2. **Stage 12 — Autonomous Strategy Discovery & Promotion (ASD-1–ASD-6).** Begin only after Stage 11's full accepted closure and stable risk/execution/persistence capabilities. Implement declarative strategy candidate generation, research/validation, guarded Shadow → Paper → Canary Live promotion and Champion/Challenger continuous discovery. Detailed scope and acceptance criteria: [Stage 12 roadmap](../../roadmap/autonomous_strategy_discovery_and_promotion.md).
3. **Following Stage 12 — Profitability & Edge Optimization (PEO-1–PEO-10).** Remains a named post-ASD capability block; it is **not** assigned a numbered stage by this change. Detailed scope and acceptance criteria: [PEO roadmap](../../roadmap/profitability_edge_optimization.md).

**Required order:** Stage 9 closure → S9-S10-PRE-01 → Stage 10 live qualification and closure → **Stage 11 / TIH** → **Stage 12 / ASD** → **PEO**.

No implementation, deployments, release gates, explicit Stage 10 opt-in policy, or Stage 9/10 readiness status are changed by assigning these planning numbers.

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

**Purpose:** turn the working Core into a coherent operator product with a production UI that is fully interactive before the first beta.

## M4.1 — UI/UX Production Implementation

The first-beta UI is an implemented product surface, not a static preview or mockup. Design decisions are validated through working vertical slices connected to the real Core.

Required principles:

- define and implement the production information architecture before broad visual polish
- build the production DesktopShell and navigation as working runtime surfaces
- use a complete Command Center vertical slice as the first integration proof
- connect visible state to real Core/read-model contracts rather than fixture-only preview data
- every visible interactive control must either perform its real action or be explicitly disabled with the reason exposed to the operator
- implement loading, empty, degraded, offline/reconnecting, permission-denied and error states where applicable
- preserve Core ownership of privileged state; UI actions request capabilities but never manufacture authority
- support the target Windows scaling/resolution matrix and keyboard/focus accessibility expected for beta
- evaluate UX by operating the running product, not by screenshot approval alone

## Product surfaces

The production UI must cover, at minimum:

- Command Center / operational overview
- trading, orders, fills and execution state
- portfolio, P&L and capital
- strategies and trading intelligence
- risk configuration, exposure, blocks and kill switch
- market/exchange/account connectivity and health
- backtesting/simulation workflows required by the beta scope
- alerts, audit and diagnostics
- update/rollback status
- system/settings and environment visibility
- explicit PAPER / TESTNET / LIVE distinction

Existing preview/QML surfaces may be reused only when their runtime behavior, ownership and UX fit the canonical architecture. Static mock behavior does not count as implementation evidence.

## Packaging

- canonical Windows entrypoint
- deterministic PyInstaller build
- QML/assets/dependency inventory
- artifact verification
- smoke tests
- version metadata
- installer/release packaging as selected by final architecture

### M4 exit criteria

- A non-developer can install and operate the beta-scope PAPER/TESTNET product without CLI intervention.
- All beta-critical UI workflows are clickable and functional end to end against real runtime contracts.
- No beta-critical control is a silent placeholder or mock-only action.
- Required loading/error/degraded/offline states are implemented and understandable without reading raw logs.
- Closing the GUI behaves according to Core/Tray lifecycle contract.
- UI cannot manufacture privileged state or bypass Core gates.
- Build is repeatable and smoke-tested.
- Remaining visual polish items may be deferred only when they do not impair correctness, discoverability or operation.

---

# PB — First Beta Readiness Gate

**Purpose:** freeze the first-beta feature surface, audit the entire repository after the production UI is integrated, remediate defects and establish a reproducible Beta Baseline before any public/limited beta build is cut.

This gate is mandatory between M4 and M5. It is not a cosmetic review and must not be replaced by green CI alone.

## PB.1 — Beta scope / feature freeze

Before repository-wide audit begins:

- production beta-scope UI from M4 is integrated
- no new features enter the beta branch except changes required to fix an accepted audit finding or unblock beta qualification
- architecture and product contracts required by the beta scope are frozen
- new opportunistic redesign is prohibited unless an identified defect proves the current design unsafe, incorrect or unmaintainable at beta risk level

“Could be cleaner” or “could be rewritten more elegantly” is not, by itself, an audit finding.

## PB.2 — Full repository inventory and module audit

Audit the complete repository, module by module. The audit must cover executable code, tests, configuration, CI, packaging, deployment and relevant canonical documentation.

The audit explicitly searches for:

- logic defects and invalid assumptions
- edge cases and boundary-condition failures
- race conditions, ordering and concurrency hazards
- retry, recovery, lifecycle and cleanup defects
- persistence/rebuild/migration inconsistencies
- stale, duplicate or competing authority paths
- dead or obsolete production-reachable code
- architecture drift and contract/implementation mismatch
- unsafe defaults and fail-open behavior
- error-handling paths that hide or corrupt state
- performance/resource defects that can affect beta operation

## PB.3 — Cross-module and architecture-drift audit

After module-level review, perform a separate repository-wide integration pass focused on boundaries that can appear correct in isolation.

Verify, at minimum:

- producer/consumer contract compatibility
- identity, ownership and lifecycle continuity across modules
- environment semantics across PAPER / TESTNET / LIVE
- retry/idempotency semantics across process/network boundaries
- persistence ↔ recovery ↔ reconciliation consistency
- UI ↔ Core authority boundaries
- packaging/runtime/version compatibility
- consistency between canonical architecture, machine-readable contracts and implementation

## PB.4 — Security and trust-boundary audit

Perform a dedicated security pass covering the beta-reachable surface, including:

- trust boundaries and capability provenance
- signing/key identity and verification
- TPM / provisioning / licensing paths in scope
- secret handling and credential references
- privilege boundaries
- input validation and canonicalization
- fail-closed semantics
- persistence and recovery security
- release/update artifact integrity

Hosted/cloud checks do not substitute for native Windows or physical-hardware qualification where the contract requires those environments.

## PB.5 — Test and quality audit

Do not treat existing green tests as proof that the right behavior is tested.

Audit for:

- missing negative tests
- false-positive or implementation-mirroring tests
- missing integration and cross-module tests
- missing property/fuzz tests where they provide material value
- failure-injection and recovery coverage
- Windows-specific qualification gaps
- coverage of beta-critical operator workflows and UI states
- CI gates that can pass while required behavior remains unverified

## PB.6 — CI, build, dependency and release-surface audit

Review:

- GitHub Actions and release workflows
- dependency declarations and lock authority
- reproducibility
- build/package contents
- installer/update/rollback paths
- artifact provenance and verification
- platform parity
- configuration/environment leakage
- security and quality gates

## PB.7 — Finding ledger and remediation policy

Every accepted finding is recorded in one audit ledger with:

- unique ID
- affected module/boundary
- evidence
- severity
- user/security/financial impact
- remediation decision
- validating tests/qualification
- PR/commit
- final disposition

Severity classes:

- **BLOCKER** — beta cannot start
- **HIGH** — beta cannot start
- **MEDIUM** — fix before beta unless explicitly accepted with written rationale, bounded impact and follow-up owner
- **LOW** — does not block beta by default; retain in backlog unless cheap/risk-reducing to fix during an already-open remediation

Remediation is done in controlled, reviewable PRs. A single repository-wide mega-PR is prohibited.

## PB.8 — Independent re-audit

After remediation, perform one independent repository-wide re-audit from the current Beta Candidate state rather than merely checking the previous finding list.

Normal policy is a maximum of **two complete repository-wide audit rounds**:

1. initial full audit → remediation
2. independent re-audit → final remediation/acceptance

A third full audit is justified only if the independent re-audit discovers a new class of BLOCKER/HIGH systemic defect that invalidates the previous coverage model.

## PB.9 — Final native qualification and Beta Baseline

Before cutting the first beta:

- run all required native Windows qualification
- run required TPM/hardware qualification that cannot be established in cloud
- complete beta-scope E2E, installer/update/rollback and recovery qualification
- record all explicitly accepted MEDIUM and LOW residual risks
- create the immutable **Beta Baseline SHA/tag**

### First Beta exit criteria

The first beta may be cut only when:

- the complete repository has been covered by the defined audit scope
- cross-module and architecture-drift audits are complete
- **BLOCKER = 0**
- **HIGH = 0**
- every remaining MEDIUM has an explicit written acceptance and follow-up disposition
- CI, quality and security gates are green for the Beta Candidate
- required Windows-native and physical TPM/hardware qualification is PASS
- no unresolved canonical architecture conflict remains in the beta scope
- all beta-critical UI flows are functional, connected to real runtime contracts and free of silent mock/placeholder actions
- the independent re-audit is complete
- Beta Baseline SHA/tag and residual-risk register are recorded

Once these criteria are met, beta begins. Further pre-beta redesign or polish is not allowed merely because a newer model, tool or alternative implementation appears more attractive; subsequent changes are driven by beta evidence, regressions, accepted findings or explicitly re-opened scope.

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
