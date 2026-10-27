# CryptoHunter Product Delivery Roadmap

Status: **canonical planning contract**

This document defines the delivery path from the current Product Architecture Contract work to a production-ready CryptoHunter v1. It is the planning source of truth for milestone sequencing, scope boundaries and exit criteria. Individual implementation plans, PRs and audits should reference the milestone and sub-step they advance.

## North-star outcome

CryptoHunter v1 is complete when its approved release scope has a stable installable Windows desktop product (primary platform) and a separately qualified Linux release (supported platform), with a durable background Core, PAPER and TESTNET end-to-end trading, guarded LIVE execution, multi-exchange/account routing, deterministic risk and accounting, secure identity/secrets, recovery, observability, updater/rollback and release-grade operational safety. Platform readiness is proven separately; a Windows-only initial release must be labelled Windows-only until Linux qualification passes. macOS is not in the v1 release scope.

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

## Platform release scope — PLAT-01 (decision: 2026-10-27)

**Product scope decision:** CryptoHunter v1 targets **Windows as PRIMARY** and **Linux as SUPPORTED**, subject to separate native release qualification. **macOS is DEFERRED**: it is not a promised beta/v1 production platform, no physical Mac purchase or physical macOS qualification is planned, and no macOS production-ready claim may be made from hosted CI.

| Platform | v1 delivery intent | Required release evidence |
| --- | --- | --- |
| Windows | PRIMARY; existing Windows development machine and native Windows qualification | Windows-specific Stage 9, legal enrollment, Stage 10 where applicable, platform release gates, MSI/service/TPM/security/lifecycle and v1 product acceptance. |
| Linux | SUPPORTED; native Dell Latitude 5490 Ubuntu test host | Independent Linux Stage 9 evidence and observer qualification where required, systemd/permissions/PostgreSQL/installer/lifecycle tests, Linux-specific gates and actual packaged product acceptance. Windows PASS does not qualify Linux. |
| macOS | DEFERRED; explicitly outside beta/v1 supported release scope | No physical Mac requirement for the Windows/Linux release **after** the machine-readable release-scope policy is formally aligned. Future macOS support needs its own approved platform project and native qualification. |

**Current contract mismatch and required follow-up:** `deployment/platform_readiness.json` still declares `MACOS=SUPPORTED` and contains `MACOS_PRODUCTION_READY` acceptance gates. This roadmap decision **does not modify, silently waive, mark NOT_APPLICABLE, or remove** those gates, CI jobs, or any recorded status. Before applying this v1 scope to release decisions, submit a separate reviewed PR aligning the machine-readable platform scope, validators, platform-specific acceptance and release workflows with Windows/Linux v1 and macOS deferred. Preserve the existing safeguards until that change is validated. Any conflict between this planning decision and the currently executable gate remains an **open release-scope blocker**, not implicit permission to ignore a failure.

**CI and evidence policy:** Keep macOS hosted build/unit/compatibility checks where useful and feasible, with their results explicitly classified as compatibility evidence, **not physical qualification**. Never suppress a real cross-platform regression affecting shared code, security or Windows/Linux runtime. A future reviewed policy may classify macOS-only native launchd/packaging acceptance as deferred for the Windows/Linux release; it must not weaken required Windows/Linux checks, change existing quality/security thresholds, mask mandatory workflow failures or declare unsupported macOS features implemented. Windows and Linux must each pass their actual target-platform gates before being advertised as supported; a staged Windows-first release cannot claim Linux READY until Linux has its own accepted evidence.

**Hardware and trust boundary:** The operator already has a primary Windows computer and a dedicated Dell Linux test laptop. Do not make purchasing a Mac a Stage 9, first-beta or v1 prerequisite. This does not eliminate the independent trust roots, reviewer/custody separation, protected physical Windows execution environment or separately authorized physical testing that Stage 9/10 require. Two available computers alone do not prove independence or attestation.

**No implicit promotion:** This decision does not change `Stage 9=IN_PROGRESS`, legal enrollment, Root Proof issuance, Windows 0–14 readiness, Stage 10 `NOT_STARTED`, LIVE permissions or any platform `PRODUCTION_READY` result. All applicable gates remain fail-closed.

## Windows Stage 9 → Stage 10 preflight — S9-S10-PRE-01

**Status: PLANNED.** This is a mandatory operational handoff **after Stage 9 closure and before Stage 10 live reboot / 24-hour qualification**. Preparing the infrastructure may begin earlier, but preparation must not activate Stage 10 or bypass legal production enrollment. This prerequisite does not add or close a numbered Windows Stage 0–14 acceptance item and does not change the M0–M6 product milestone taxonomy.

- **Dedicated physical Windows host:** PRE-01B.2A implements the manual installation design with no Actions runner or CI execution channel, separate transfer/elevated operator identities and independently protected staging, policy, host inventory and executor checkout. Physical provisioning and approval are pending. A future Stage 10 runner/lifecycle route remains a separate design and authorization decision; labels alone do not provide isolation.
- **Qualified MSI transfer:** define a reproducible, operator-approved procedure that retrieves `windows-stage9-qualified-msi` from the **same canonical workflow run** as the Stage 10 qualification, together with `installer-manifest.json`, `windows-clean-install-receipt.json` and `windows-clean-install-evidence.json`. Check the artifact digest, exact MSI/payload hashes, source revision, CI run ID, Production Trust identity and revision-bound evidence; reject stale, mixed-run, substituted or locally rebuilt installers.
- **Target-host deployment:** implement a reviewed, separately authorized elevated handoff of the exact qualified MSI. Qualify the clean base payload, MSI identity/version, SCM and PostgreSQL/ACL boundaries independently of enrolled runtime provenance/readiness. Never silently replace an enrolled installation or treat a GitHub-hosted clean-install as proof that the physical target has the same installed MSI.
- **Non-destructive rehearsal:** qualify the native host/operator, protected code/policy/staging, artifact hash lineage, approval/consent and logging boundaries before a separate installation decision. Hermetic tests cannot approve those physical controls. Do not perform installation, reboot, TPM/NV mutations, enrollment or 24-hour soak as part of roadmap preparation alone.
- **Explicit Stage 10 gate:** actual lifecycle execution still requires completed legal production enrollment, satisfied readiness prerequisites and a separate operator-approved `workflow_dispatch` with `run_stage10=true`. The existing Stage 10 prepare job downloads clean-install evidence but does **not** install the MSI onto the self-hosted runner; that delivery gap must be resolved and reviewed before any reboot qualification.

**Exit criteria:** approved manual-host MSI handoff runbook and independently verified host/operator protection; repeatable hash/provenance verification and fail-closed mismatches demonstrated on native Windows; a separately authorized installation and passive qualification of the exact qualified MSI; and independent sign-off that live restart and 24-hour soak remain disabled until their separate authorization. Stage 10 additionally requires an approved lifecycle execution route preserving its same-run provenance. Code/test preparation alone does not close this gate or establish Stage 10 `PASS`.

[S9-S10-PRE-01A read-only preflight and PRE-01B handoff runbook](../../windows_stage10_readonly_preflight.md) defines the preparation reports, remaining target-installation gap and separately approved next task. It does not close Stage 9 or authorize Stage 10.

### S9-S10-PRE-01B.2A secure manual executor and passive qualification

[The operator handoff contract](../../windows_stage9_operator_handoff.md) defines
qualified-artifact custody, completed-run versus active-run producer-job proof,
strict host/run/artifact-bound approval data, independent native operator
authentication, read-only host/runner-tree inspection and hermetic
installation/post-install verification preparation. Preparation plans do not
execute MSI and keep `installation_execution_authorized=false`. Fixtures do not
establish a live target, authenticated GitHub metadata or operator approval.

The current User-owned repository cannot satisfy the organization runner-group
model. The implemented design requires a dedicated manual host with no
Actions runner or CI execution credentials; an Organization-controlled,
manual-only workflow and restricted runner group is the eventual Actions
lifecycle design. Physical host/operator approval and verified enforcement
remain pending; this change modifies no ownership, settings or registration.

**Code implementation:** the separate executor reuses PRE-01A/PRE-01B.1
handoff, native token, host-inspection and ACL authorities. It requires an
independent protected host policy, a distinct execution authorization valid at
most ten minutes, explicit native-console confirmation of the exact plan digest
and fresh host/operator/artifact/run/attempt checks. Native file handles pin the
protected inputs/materialized MSI through the direct system `msiexec.exe`
boundary. Ctrl-C records interruption and retains handles while waiting for
the single existing installer to finish; it never kills or retries it.
Installer outcomes, protected full logs and immutable journal entries survive
for diagnosis without automatic repair, uninstall or reboot. Preparation scope
never grants execution rights. CI adds hermetic tests and quality scope only,
with no native executor invocation or route to an installation target.

**Hermetic tests:** substituted native/process boundaries establish refusals
and outcome classification only; their exact validation results belong in the
PR. They do not qualify Windows tokens/ACLs/handle-sharing or approve an actual
target. No MSI is executed by this task and native Windows qualification has
not been performed.

**Separate passive phases:** clean-base qualification checks the full installed
payload, MSI identity/version, SCM, PostgreSQL static configuration and ACLs,
with backend/verifier demand-start/stopped. Both phases require the separate
`postgresql_effective_sql` gate; its approved passive native service-token SQL
collection adapters are implemented (PRE-01B.2C/2D), but genuine service-local
committed-enrollment admission, qualified native IPC and active-HBA evidence
remain unavailable. Overall base and runtime qualification stay `BLOCKED`
despite independently successful static base checks. Base qualification
requires no legal enrollment or backend readiness. Enrolled-runtime qualification
first requires accepted legal enrollment, then native runtime provenance/SCM/PID,
SQL separation, mTLS and readiness authorities. Legal enrollment also remains
unavailable. Neither phase starts services or changes configuration/roles/ACLs.

**Future decisions:** independently approve and qualify the physical host,
operator, protected checkout/policy/staging and native race protections; then
make a separate decision about actual installation with fresh exact-plan
authorization. Actual installation, legal enrollment, any future runner route,
reboot and 24-hour lifecycle are outside this implementation. The handoff
contract records remaining live evidence gaps. Stage 9 remains
`IN_PROGRESS`, legal enrollment `NOT_PERFORMED`, `run_stage10=false`, and Stage
10 `BLOCKED` / `NOT_RUN`; reboot and 24-hour lifecycle stay outside this work.

## Stage 9 PDSA activation transport — PDSA-TR-01 (offline first; online ready)

**Status: DECISION ACCEPTED; IMPLEMENTATION / DEPLOYMENT NOT COMPLETE.** Initial
legal production enrollment must support an **offline file exchange without an
always-on enrollment server**. The eventual **online activation API** is a
second delivery adapter around the same frozen PDSA challenge, TPM evidence,
production authorization-package bytes, trust roots, quorum-signing authority,
expiry, replay protections and retained issuer state. Neither transport mints
subjects, selects signing keys, rewrites signed packages or establishes authority.

- **OFFLINE_FILE (initial requirement):** the Windows LPPI exports a bounded,
  canonical request/evidence bundle without private-key material. An independently
  protected PDSA operator workstation verifies it, obtains the approved 2-of-3
  quorum response and durably retains exact issuance/retry identity. A signed,
  canonical return package is transferred back out-of-band and verified by the
  Windows installed LPPI. Offline means *no Internet requirement*, not that a
  user-controlled installation may sign its own authorization. A separately
  protected signing authority and qualified production materials remain mandatory.
- **ONLINE_API (later, after server deployment):** an authenticated bounded
  network transport submits the same canonical request to that same independent
  PDSA authority and returns the identical class of signed package. TLS/API
  authentication is a transport control, never a substitute for PDSA signatures
  or hardware-bound acceptance. No live endpoint or online availability is
  asserted by this planning decision.
- **License/entitlement distinction:** legal device enrollment establishes
  cryptographic identity; separate signed Basic/Medium/Premium entitlements,
  renewal, revocation and any controlled grace period define feature access.
  Temporary licensing-server unavailability must not by itself trigger unsafe
  liquidation or unbounded live trading. Trading-exchange connectivity and
  order-risk recovery remain independent fail-safe boundaries. Precise grace,
  expiration and open-position policies require their own contract and tests.
- **Acceptance:** native Windows request/export/import path, offline operator
  workflow, strict canonical envelope/size/schema validation, PDSA quorum and
  exact-binding verification, replay/lost-response/restart/cross-device negatives,
  and explicit separation of TEST_ONLY and genuine installed production authority.
  The later online adapter must reuse the same verified domain workflow.

This choice does **not** authorize creation of substitute production PDSA keys,
local signing on an enrolling device, automatic production enrollment, enabling
Stage 10, or claiming physical TPM/PDSA qualification. Current Stage 9 readiness
and its canonical status file remain unchanged.

## Numbered capabilities after Stage 10 — Stage 11 and Stage 12

**Status: PLANNED; numbering formalized, implementation not started.** These are product capability stages ordered after the Windows Stage 10 live lifecycle qualification. They do **not** renumber or extend frozen Windows Stage 0–14 acceptance items, or replace the canonical M0–M6 product milestone taxonomy. Neither stage may claim readiness solely from design documents, existing preview code or the presence of a workflow.

1. **Stage 11 — Trading Intelligence Hardening (TIH-1–TIH-7).** After Stage 10 closure and platform stabilization, harden regime classification, market microstructure and reversal intelligence, execution/TCA, anti-overfitting validation, live edge-decay monitoring, capital allocation and L2/trades replay. Detailed scope and acceptance criteria: [Stage 11 roadmap](../../roadmap/trading_intelligence_hardening.md).
2. **Stage 12 — Autonomous Strategy Discovery & Promotion (ASD-1–ASD-6).** Begin only after Stage 11's full accepted closure and stable risk/execution/persistence capabilities. Implement declarative strategy candidate generation, research/validation, guarded Shadow → Paper → Canary Live promotion and Champion/Challenger continuous discovery. Detailed scope and acceptance criteria: [Stage 12 roadmap](../../roadmap/autonomous_strategy_discovery_and_promotion.md).
3. **Following Stage 12 — Profitability & Edge Optimization (PEO-1–PEO-10).** Remains a named post-ASD capability block; it is **not** assigned a numbered stage by this change. Detailed scope and acceptance criteria: [PEO roadmap](../../roadmap/profitability_edge_optimization.md).

**Required order:** Stage 9 closure → S9-S10-PRE-01 → Stage 10 live qualification and closure → **Stage 11 / TIH** → **Stage 12 / ASD** → **PEO**.

No implementation, deployments, release gates, explicit Stage 10 opt-in policy, or Stage 9/10 readiness status are changed by assigning these planning numbers.

## Cross-cutting trading-mode / strategy coverage — TSP-1–TSP-8

**Status: PLANNED.** Before any trading mode or strategy is advertised as fully supported, audit and qualify the exact **SPOT / MARGIN / PERPETUAL / DELIVERY_FUTURES / OPTIONS × LONG / SHORT / FLAT × strategy family × exchange account × environment** combination. A strategy signal or exchange adapter alone does not establish executable capability. The current M0.6 edition authorizes strategy activation only for SPOT/SPOT_PAIR; derivative expansion requires separately versioned contract authorization, real qualification and risk/ledger/recovery gates. SPOT inventory sale must never silently become a short.

The mandatory cross-cutting backlog is [Trading Modes, Strategy Parity & Autonomous AI Readiness](../../roadmap/trading_modes_strategy_parity_and_autonomous_ai.md): **TSP-1** capability matrix; **TSP-2** typed position/order intent; **TSP-3** derivatives/accounting parity; **TSP-4** multi-leg execution; **TSP-5** full family qualification; **TSP-6** Core-bound UI/telemetry; **TSP-7** Stage 12 capability-aware autonomous AI; **TSP-8** gates and E2E evidence. M1–M6 implement the approved subset; Stage 12 must consume the same capability registry, never bypass it. This adds no implicit first-beta, LIVE or Stage 9/10 authorization.


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
- exact per-account identity (including multiple subaccounts of the same venue), environment binding and Core-owned capability/credential authority
- **Core-owned exchange lifecycle:** support versioned `active`/`disabled`/`deprecated`/`retired` states and enforce a non-bypassable product retirement policy across accounts, credentials, venue capabilities and all execution environments. Zonda and BitMEX are initial `retired` cases; preserve immutable historical financial records, not active trading authority.

The binding audit, legacy disposition and acceptance gates are specified in [multi-exchange beta readiness](../../roadmap/multi_exchange_multi_account_beta_readiness.md) and [Exchange Lifecycle & Retirement Management](../../roadmap/exchange_lifecycle_retirement_management.md). The prior [Zonda-only plan](../../roadmap/zonda_retirement.md) is superseded by the generic mechanism.

## M1.4 — Strategy, market-data and execution routing

Implement:

- StrategyDefinition / StrategyInstance runtime
- MarketDataRoute
- ExecutionRoute
- deterministic routing validation
- adapter lifecycle and health
- route failure handling
- multiple simultaneously active account-scoped adapters, never only `{"primary": adapter}` as effective product composition
- independent concurrently executing strategies with atomic aggregate portfolio risk/capital reservations; no implicit cross-account or cross-environment order fallback
- exact account-bound cancellation/retry/reconciliation after ambiguous network outcomes, partial fills and restart

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

## M1.8 — Exchange Lifecycle & Retirement Management (mandatory)

**Decision: replace mandatory Zonda source-code deletion with a reusable Core-enforced exchange lifecycle and retirement gate.** Zonda and BitMEX are initial `retired` venues; subsequent delistings, closures, regulatory suspensions and unsupported integrations use the same mechanism. Preserve inactive adapter sources for historical compatibility if safe, but never permit retired venues to be discoverable as supported, initialized as executable, network-connected, selected in UI, used in PAPER/TESTNET/LIVE execution, or reached through marketplace, background jobs, fallback, dynamic registrations or restored legacy configuration. A retired venue has no trading/cancel/data-fetch authority; only locally retained historical data and offline deterministic replay remain readable.

**Implementation contract:** introduce an authoritative versioned, release-controlled retirement policy; normalize venue IDs, variants and aliases; extend current `exchange_lifecycle` beyond its present LIVE-route-only guard; force `retired` even when user YAML requests `active`; fail closed on unknown/ambiguous IDs, missing qualified capability or stale signed policy. Local/operator overrides may only restrict access, not revive a permanently retired venue. Versioned migrations must retire stored accounts, routes and secret profiles without changing durable IDs, immutable historical journals or frozen M0 contracts. Remove retired venue presets from *active* signed marketplace catalogs via authorized resigning; retain archival evidence.

**Status: PLANNED / NOT IMPLEMENTED.** Current `disabled/deprecated` LIVE guards and [PR #3124](https://github.com/Dudzian/Dudzian/pull/3124) exchange-report handling do not satisfy this full product gate. Implement in focused, independently reviewed code/migration/test PRs; no LIVE permission or release status changes follow from roadmap text. Detailed work and exit evidence: [Exchange Lifecycle & Retirement Management](../../roadmap/exchange_lifecycle_retirement_management.md).

### M1 exit criteria

- Core can start, persist state, restart and rebuild deterministically.
- Core boundaries match M0 contracts.
- Domain/account/routing/order/ledger/risk/security foundations execute in code, not preview models.
- LIVE remains closed.
- Legacy paths no longer create competing authority.
- Exchange lifecycle and retirement checks are enforced across all active paths; Zonda and BitMEX cannot trade, cancel, connect, register or be selected, while immutable historical audit records remain readable and exact.
- Multi-account Core composition and risk/route identity are validated without `primary` legacy aliasing.

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
- simultaneous independent strategies and separate accounts on different venues or distinct subaccounts, with exact order/cancel/fill binding
- atomic cross-venue aggregate portfolio exposure, risk caps and capital reservations; no double-spending of risk under concurrency
- fail-closed uncertain-execution reconciliation before retry/cross-venue fallback; no broadcast cancellation on missing order binding

## M3.5 — Multi-Exchange & Multi-Account Production Readiness (beta blocker)

**Status: PLANNED.** Audit all available real exchange adapters and variants, record an evidence-backed per-venue/market/environment support matrix, then qualify the **approved first-beta venue set** on genuine TESTNET where supported. Prioritize evaluation of Binance, Kraken, Bybit and OKX; do not advertise unsupported spot/futures sandboxes or claim all registered adapter classes are operational. Zonda and BitMEX are permanently **RETIRED**, not candidates; a retained adapter source does not qualify as active support.

Prove at least two independently authenticated venue/account integrations can trade concurrently through canonical Core; verify routing, private identity, least-privilege API keys, secrets, order lifecycle, crash/reconnect recovery, throttling, partial fills, idempotency, global portfolio risk and exact-account cancel without unintended venue crossover. Integration must reach the Windows packaged application's real UI. Review failure scenarios per venue rather than trusting mocks only.

Complete work contract: [Multi-Exchange & Multi-Account Production Readiness](../../roadmap/multi_exchange_multi_account_beta_readiness.md). This is a mandatory **M3 and PB gate before the first beta**; it does not authorize M5 LIVE.

### M3 exit criteria

- TESTNET trades execute through the canonical Core path.
- Reconnect/retry cannot create duplicate economic execution.
- Exchange state reconciles with local state after restart.
- At least the production-target exchange set required for v1 is validated.
- Two distinct qualified TESTNET venue/account paths execute and reconcile independent orders concurrently with no risk/capital leakage.
- Venue support status is evidence-backed; Zonda and BitMEX are blocked by the accepted Core retirement gate in PAPER/TESTNET/LIVE and absent from active account, adapter, UI and marketplace surfaces.
- All beta-supported venues are configurable from the real UI without editing YAML or invoking CLI.
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

## M4.2 — Production UI Overhaul & Core Integration (first-beta BLOCKER)

**Status: PLANNED.** This is a dedicated rebuild/adaptation workstream, not an optional styling pass. Inventory all existing PySide6/QML controls and legacy UI alternatives; classify KEEP/ADAPT/REWRITE/DELETE. Replace misleading preview-only screens and dead controls with coherent, real Core-connected vertical slices before first beta. Implement an operator-ready Command Center, orders/fills, strategies, portfolio/PnL, risk/kill switch, market scanner, alerts/audit/diagnostics, updater and system/settings, with working loading/offline/reconnection/error/permission states. No static mock can count as a functional beta screen.

**Dedicated Settings → Exchanges & Accounts:** create/manage multiple venue accounts and subaccounts, choose PAPER/TESTNET/LIVE only where permitted, choose spot/margin/futures when supported, securely write API key/secret/passphrase into OS-backed Core custody, verify identity/permissions/connection, rotate/revoke, configure portfolio/account routes and risk caps, and show actual balances, open orders and per-venue health. UI read models contain only masked state/opaque secret references, never material. Unsupported or retired venues (including Zonda and BitMEX), unavailable sandboxes and LIVE without authority cannot be selected or activated. A SAVE/CONNECT/TEST button must perform the corresponding real Core operation or be explicitly disabled with an explanation.

**Translation-ready foundation (first-beta preparation, not a multilingual launch):** While rebuilding the canonical PySide6/QML DesktopShell in M4.2, use Qt-compatible translatable UI strings/message IDs and keep Core-owned error/status codes separate from presentation text. Avoid introducing new hard-coded user-facing strings that would force a second UI rewrite. Full language catalogs, a runtime language switcher and multi-locale completeness testing are explicitly **post-first-beta**, not M4.2/M4.3/PB release blockers. The independent [post-beta I18N roadmap](../../roadmap/post_beta_internationalization_localization.md) defines their scope and tests.

Detailed work and test contract: [Production UI Overhaul & Core Integration](../../roadmap/production_ui_overhaul.md).

## M4.3 — UI native end-to-end qualification

Run the Windows packaged PySide6/QML product through actual first-launch → account onboarding → independent PAPER/TESTNET trading → account/portfolio state → risk/kill switch → restart/reconnect → audit and diagnostics. Confirm keyboard/DPI, consistent navigation, no secrets in logs, no live bypass and no silent placeholders. These checks must be supported by integration and native UI automation, not screenshots alone.

## Packaging

- canonical Windows entrypoint and MSI/release qualification for the primary platform
- separately qualified Linux entrypoint, package, native systemd/permission/lifecycle integration and release evidence before any Linux-supported build is advertised
- deterministic PyInstaller build where applicable to the selected platform
- QML/assets/dependency inventory
- artifact verification
- smoke tests
- version metadata
- installer/release packaging as selected by final architecture
- no required macOS package, launchd or physical Mac qualification for v1 **once** PLAT-01 machine-readable scope migration has been reviewed and merged; no macOS production support claim

### M4 exit criteria

- A non-developer can install and operate the beta-scope PAPER/TESTNET product without CLI intervention on every platform advertised as supported in that specific beta; Windows is primary, and Linux requires a separate native E2E and packaging sign-off before it is included.
- All beta-critical UI workflows are clickable and functional end to end against real runtime contracts.
- Settings → Exchanges & Accounts stores credentials securely, verifies accounts and supports multiple qualified TESTNET accounts without CLI/config editing.
- The dedicated UI overhaul M4.2 and native qualification M4.3 have complete evidence; Zonda and BitMEX are neither shown as connectable nor reachable through UI actions.
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
- MEX multi-exchange/account E2E and M4.2/M4.3 production UI gates have PASS evidence; the general M1.8 retirement policy and Zonda/BitMEX persisted-identity migrations have PASS evidence
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

- run all required native Windows qualification for the primary beta target
- run Linux-native installation/service/UI qualification before the beta is advertised as supporting Linux; hosted Linux unit tests alone are insufficient
- run required TPM/hardware and independent-trust qualification that cannot be established in cloud
- enforce the reviewed PLAT-01 machine-readable release-scope migration before deferring any macOS-specific acceptance gate; hosted macOS compatibility tests never qualify physical macOS
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
- required Windows-native and physical TPM/hardware qualification is PASS; Linux-native qualification is also PASS before Linux is included in the supported beta scope
- no unresolved canonical architecture conflict remains in the beta scope
- all beta-critical UI flows are functional, connected to real runtime contracts and free of silent mock/placeholder actions
- supported venues/credential profiles are manageable without CLI, two qualified independent TESTNET accounts operate simultaneously, and Zonda/BitMEX (plus any newly retired venue) have no reachable trading/configuration/UI/marketplace path under the versioned lifecycle gate
- cross-venue cancellation, retry, order identity and aggregate portfolio risk are proven safe by negative E2E tests
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
- Desktop packaging, update and rollback are release-grade on each platform explicitly supported by the release (Windows primary; Linux independently qualified when included). macOS remains deferred with no production support claim.
- Soak/hardening acceptance criteria pass.
- Final production audit records remaining limitations explicitly.

---

# POST-BETA-I18N — Internationalization & Localization (I18N-1–I18N-6)

**Status: PLANNED — after the first beta begins.** This is an independently scheduled product-experience workstream, not an additional numbered Windows Stage, not Stage 11/12, and not a prerequisite for the first beta, M5 LIVE Canary or M6 unless a future release explicitly opts into shipping a new locale.

**Pre-beta architectural seam:** M4.2 makes the production PySide6/QML DesktopShell translation-ready as the real Core-bound UI is built. Translation extraction and separation of presentation text from Core-owned status/error codes should avoid a costly later rewrite. Full catalogs, user-facing language switching and multi-language qualification are deferred until after the first beta.

**Post-beta user experience:** A visible language control in the application header and/or Settings displays a recognizable icon/flag, native language name and locale code. A click opens an accessible language picker; selecting a locale immediately retranslates the running app without restart. The chosen locale survives app restart. A flag is only a visual aid, not a one-to-one language identity.

**Planned delivery:**

- **I18N-1 — UI text inventory and translation infrastructure.** Inventory real QML/PySide6 surfaces, dialogs, menus, tray, toasts, notices, dynamic status/error messages and confirmation flows. Introduce Qt-compatible translation catalogs and stable message identifiers without crossing the Core authority boundary.
- **I18N-2 — Initial language catalogs.** Plan Polish (`pl-PL`) and English (`en`/reviewed regional variant) as the first candidate locales; approve exact shipped locales and translation owners at implementation time. Additional locales use versioned catalogs within the same application, not separate executables.
- **I18N-3 — Live switching and accessible picker.** Show flag/icon **plus** language name/code, retranslate existing and newly opened QML/PySide6/tray surfaces without restarting Core, losing unsaved input or cancelling asynchronous operations. Cover plural forms, parameter substitution and locale fallback.
- **I18N-4 — Preference persistence and safe fallback.** Store the selected locale in normal per-user UI settings; restore it after restart and recover predictably from missing/incomplete/invalid catalogs without exposing secrets or modifying trading state.
- **I18N-5 — Locale-aware presentation.** Localize display-only dates, times, number grouping, decimal separators and help text; keep monetary amounts, instrument symbols, API identifiers, canonical serialization, signed bytes, audit/security identifiers and execution semantics invariant. Warnings for PAPER/TESTNET/LIVE, orders, risk and kill switch must remain unambiguous.
- **I18N-6 — Native end-to-end localization qualification.** Test immediate switching during PAPER/TESTNET activity, changing languages repeatedly, reopen/restart persistence, missing translation fallback, long strings/DPI/accessibility, error dialogs, confirmation/warning parity and translation coverage for each shipped locale. No mock-only selector counts as implemented.

**Acceptance:** A real, fully interactive header/Settings selector switches all supported user-facing surfaces at runtime and retains preference after restart; all shipped catalogs meet reviewed coverage/terminology rules; untranslated keys have safe, observable fallback; UI text changes cannot affect Core/risk/credentials/order state or safety confirmations. Document evidence from packaged native Windows UI tests. Any new language remains unshipped until its review/tests pass.

Detailed backlog, boundaries and test contract: [Post-beta Internationalization & Localization](../../roadmap/post_beta_internationalization_localization.md). This section creates no new first-beta gate and does not change existing milestone status.

---

# Progress model

Progress is reported at two levels:

1. **Milestone progress** — completed sub-steps / audited acceptance criteria inside the active milestone.
2. **Product progress** — weighted delivery progress toward M6/v1, not a naive count of documents or PRs.

A milestone is never marked complete merely because code exists. It closes only after its acceptance evidence and closing audit are merged.

# Immediate next action

Continue **M0.11**. Do not begin M1 runtime reconstruction until M0.14 closes the architecture and produces the migration/backlog handoff.
