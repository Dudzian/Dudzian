# Roadmap — Trading Modes, Strategy Parity & Autonomous AI Readiness

**Status: PLANNED — evidence-driven capability completion. No runtime implementation or LIVE authorization is implied by this roadmap.**
**Workstream IDs:** TSP-1 through TSP-8.
**Scope:** every advertised trading direction, instrument type, and registered strategy family, from strategy decision through account-specific order intent, risk, execution, positions, accounting, recovery, UI and future autonomous strategy selection.

## Definitions and non-negotiable scope

- **Market/instrument type** is not a trading strategy: SPOT, MARGIN, PERPETUAL, DELIVERY_FUTURES and OPTIONS are distinct exchange/account/instrument capabilities.
- **Direction/position intent** is not an order side: LONG/SHORT/FLAT, OPEN/INCREASE/REDUCE/CLOSE/REVERSE, and BUY/SELL must never be conflated. Selling inventory on SPOT does not establish a naked short.
- **Strategy family** is a decision process (DCA, grid, momentum, arbitrage, options income, etc.) that may be eligible for only a subset of instruments and directions.
- **Autonomous AI** may discover, select, tune, pause, retire and submit approved strategy intents only within independently validated account/product/risk/execution permissions; it may not grant itself a capability, promote itself to LIVE, bypass a kill switch or activate an unqualified instrument.
- "Full support" means **all combinations actually offered by the product and available on the exact venue/account are qualified**. It does not mean enabling derivatives, borrow, options, or leverage on exchanges that do not support them. Unsupported combinations remain visibly disabled, with reason and provenance.

This workstream extends the existing [canonical delivery roadmap](../architecture/cryptohunter_product_architecture/product_delivery_roadmap.md), [multi-exchange readiness](multi_exchange_multi_account_beta_readiness.md), [Stage 11 TIH](trading_intelligence_hardening.md) and [Stage 12 ASD](autonomous_strategy_discovery_and_promotion.md). Do not renumber frozen Windows Stages or overwrite closed M0 contracts without a separately versioned, reviewed migration and validator update.

## Verified repository evidence and present qualification status

This is a **source-code/doc inventory**, not a complete executed test run or production certification.

| Area | Evidence | Current assessment / required disposition |
| --- | --- | --- |
| SPOT buy and inventory sell / LONG exposure | `bot_core/exchanges/*/spot.py`; `bot_core/execution/paper.py`; M0.6 | Source implementation exists; canonical SPOT is the **only** currently allowed M0.6 execution pair. End-to-end beta wiring remains to be proven. |
| SHORT / MARGIN | `bot_core/strategies/scalping.py`, `day_trading.py`, `mean_reversion.py`; `bot_core/exchanges/*/margin.py` | SHORT signals and adapters exist; canonical M0.6 activation does **not** currently authorize MARGIN. Do not claim live or testnet coverage. |
| PERPETUAL and DELIVERY_FUTURES long/short | `bot_core/exchanges/*/futures.py`; `bot_core/strategies/futures_spread.py`; M0.5/M0.6 | Instrument structures and some adapters/signals exist; distinguish perpetual vs dated, linear vs inverse, mark/index/settlement and funding/expiry. Canonical execution activation is not yet authorized. |
| OPTIONS | `bot_core/strategies/options.py`; `bot_core/strategies/catalog.py`; M0.5/M0.6 | Options-income signal exists; options chain/Greeks and contract metadata must not be confused with a tested multi-leg options order/exercise/assignment lifecycle. Not currently authorized for activation. |
| SPOT grid / DCA / long-only inventory logic | `bot_core/strategies/grid.py`, `dca.py` | Implementations exist; do not assume that a `sell` signal can short or that a strategy is bound to canonical Core routing. Enforce spot inventory constraints. |
| Directional momentum / mean reversion / scalping / intraday | `bot_core/strategies/daily_trend.py`, `mean_reversion.py`, `scalping.py`, `day_trading.py` | Strategy implementations exist; direction-specific qualification, exit semantics and realistic cost-aware runtime integration need proof. |
| Market making / adaptive market making / volatility target | `bot_core/strategies/market_making.py`, `adaptive_market_making.py`, `volatility_target.py` | Strategy implementations exist; market-specific inventory, quote cancellation, quote skew, adverse-selection and per-position exposure gates require explicit qualification. |
| Multi-leg arbitrage / hedge | `bot_core/strategies/cross_exchange_arbitrage.py`, `statistical_arbitrage.py`, `triangular_arbitrage.py`, `cross_exchange_hedge.py`, `futures_spread.py` | Multi-leg `StrategySignal.legs` exist, but coordinated end-to-end execution, compensating actions, fills/reconciliation and failed-leg risk are not demonstrated. |
| Autonomous AI integration | `bot_core/auto_trader/ai_governor.py`; `docs/roadmap/autonomous_strategy_discovery_and_promotion.md` | Current governor contains heuristic mode selection; Stage 12 full autonomous discovery/promotion is PLANNED, not a validated live autonomous agent. |
| Execution semantics / PAPER short accounting | `bot_core/strategies/base.py`; `bot_core/execution/bridge.py`; `bot_core/execution/paper.py` | Legacy `side: str` overloads direction and action; bridge defaults unspecified actions to BUY; paper short sale proceeds are included in available quote balance before margin check. Audit/fix before parity claims. |

**Canonical blocker:** M0.6 `current_edition_execution_pair_policy` permits readiness and activation for `SPOT/SPOT_PAIR` only. The other four structurally valid type pairs are rejected with `INSTRUMENT_SCOPE_MISMATCH`; merely editing a config, adapter or UI must never bypass the edition gate.

**Catalog baseline:** `build_default_catalog()` in `bot_core/strategies/catalog.py` registers fifteen families: `daily_trend_momentum`, `mean_reversion`, `grid_trading`, `dollar_cost_averaging`, `volatility_target`, `cross_exchange_arbitrage`, `scalping`, `market_making`, `adaptive_market_making`, `options_income`, `statistical_arbitrage`, `day_trading`, `futures_spread`, `cross_exchange_hedge`, and `triangular_arbitrage`. Inventory any separately shipped plugin, preset, execution engine or strategy after this baseline; count alone is not qualification.

## TSP-1 — Complete capability and strategy matrix (audit first)

Produce an **authoritative machine-readable, versioned** matrix covering every registered strategy/preset/plugin and every advertised combination of:

1. Environment: PAPER, genuine supported TESTNET or guarded LIVE (no implicit promotion/fallback).
2. Venue, exact ExchangeAccount/subaccount, observed permission scope, region and product eligibility.
3. Instrument: SPOT, MARGIN, PERPETUAL, DELIVERY_FUTURES or OPTIONS; exact venue instrument ID and metadata; linear/inverse, expiry/settlement/contract size where relevant.
4. Strategy family, eligible LONG/SHORT/FLAT direction, long-only/short-only/both, hedge/net position mode and required datasets.
5. Actions: OPEN, INCREASE, REDUCE, CLOSE, REVERSE, CANCEL, RECONCILE; supported order types, time-in-force and reduce-only/post-only if applicable.
6. Funding, borrow availability/rates/recall, collateral, margin, leverage, liquidation, delivery, exercise/assignment, option Greeks, fees and funding-specific market data.
7. Status per cell: **QUALIFIED**, **PARTIAL**, **CONTRACT_ONLY**, **UNSUPPORTED**, **RETIRED**, including test receipts, unsupported reason, version and last verified timestamp.

Reconcile the matrix against canonical M0.4/M0.5/M0.6 ProductCapabilities/Exchange Registry. Retain Zonda's explicit RETIRED status; never enable it from legacy adapter discovery. No UI or AI may present a `CONTRACT_ONLY` or `PARTIAL` combination as available for trading.

**Acceptance:** repository-generated matrix covers every factory/registered family and every offered instrument type; CI fails on undeclared exposure, stale unqualified claims and missing provenance.

## TSP-2 — Typed position/intent semantics and shared strategy protocol

- Introduce versioned, typed `TradeIntent`/`PositionIntent` semantics separate from venue `BUY/SELL`; explicitly require strategy instance, exact account, instrument, environment, side, effect (open/increase/reduce/close/reverse), quantity, reduce-only and net/hedge position side where applicable.
- Prohibit silent `BUY` defaults, ambiguous legacy `sell` conversions, cross-environment IDs and inferred short activation; migrate old signals through audited adapters with fail-closed behavior.
- SPOT cannot sell more than actually available inventory; MARGIN/derivatives SHORT requires product support, permissions, borrow/collateral or contract capacity, position-mode compatibility, liquidation risk and independent limits.
- Define deterministic reversal (close-confirm-reopen or reviewed atomic exchange mechanism) and partial-fill/partial-close transitions; use exact decimals/contract sizes and instrument metadata, not symbol-string parsing or floating monetary authority.
- Surface supported capability metadata to both operator controls and a future machine-readable AI action catalog, without granting execution authority.

**Acceptance:** negative tests prove `SELL_SPOT` never opens a naked short, missing effect/account/instrument fails closed, a close cannot add exposure, and unknown or unsupported venue flags cannot create executable intent.

## TSP-3 — Canonical execution, derivatives and accounting parity

- Implement and qualify compatible PAPER -> supported real TESTNET -> **separately approved** LIVE canary vertical slices for each advertised market type; any expansion beyond M0.6 SPOT-only requires a reviewed versioned M0 authority migration before activation.
- Cover MARGIN borrow/interest/recall/repayment, isolated/cross collateral; PERPETUAL funding and mark-price liquidation; DELIVERY_FUTURES expiry/roll/settlement; OPTIONS premiums, multipliers, Greeks, exercise/assignment/expiration and bounded uncovered exposure only if explicitly offered.
- Implement correct position lifecycle, PnL/fees, funding, collateral lock, available equity, margin calls, liquidation, partial fills, cancellation, reconciliation and restart recovery.
- Correct PAPER short collateral accounting before using short strategy backtests for promotions; short-sale proceeds must not silently become unrestricted free collateral. Test zero-equity and stress/insolvency cases against independent ledger oracles.
- Explicitly disable unsupported contract/instrument combinations, not simulate them as production exchange capabilities.

**Acceptance:** balance-sheet invariants and matching exchange-independent execution outcomes under deterministic scenarios; independently reviewed adverse-case ledger fixtures, crash/recovery, integration and certified supported-venue TESTNET receipts.

## TSP-4 — Multi-leg execution and capital reservation

- Carry `StrategySignal.legs` to account- and instrument-bound intents with order/fill identity per leg and one auditable parent plan. Treat multi-leg exchange execution as **non-atomic** unless venue explicitly guarantees atomicity.
- Implement preflight and atomic **local capital/risk reservations** for all legs, bounded leg sequencing, liquidity checks, slippage/funding stress and max unhedged exposure/time.
- Handle partial fills, failed second leg, stale quotes, duplicate acknowledgements, disconnect, retry, compensating unwind and strategy-scoped kill switch without retrying economically uncertain executions.
- Qualify cross-exchange, triangular, statistical, futures basis, hedge and options multi-leg plans on real matching environments before offering them to autonomous AI.

**Acceptance:** fault-injection proves no uncontrolled naked leg, duplicate economic execution, wrong-account fill or bypassed aggregate risk limit.

## TSP-5 — Strategy family completion

For **each** registered/shipped family: document supported `market_type × direction × environment`, input quality/freshness, sizing, position effects, exits, constraints and costs. Qualify realistic data -> signal -> intent -> pre-trade risk -> exchange/simulator -> partial/full fill -> ledger -> UI -> reconnect/recovery. At minimum:

- DCA, grid, trend, mean-reversion, scalping, day trading: long-only SPOT inventory behavior and separate, explicit derivatives/margin direction modes.
- Market making/adaptive MM: quote lifecycle, inventory bounds, adverse-selection controls, exchange post-only restrictions and kill-switch quote cleanup.
- Arbitrage/stat-arb/hedging/futures spread: multi-leg and cross-account risk, fees, funding, latency and failed-leg unwind.
- Options income: concrete instrument, bounded defined-risk/covered strategies, complete option contract mechanics, and an explicit block where the venue/instrument capabilities are absent.
- Volatility target/risk overlays: portfolio-wide notional, gross/net/delta exposure caps and no authority to widen the underlying strategy's eligible instruments.

**Acceptance:** automated suite lists every discovered strategy family with a non-empty capability disposition; cannot mark one QUALIFIED based only on import, a preset, a simulated signal or a UI screen.

## TSP-6 — Operator product and observability

- In the real Core-bound desktop UI, show eligible venue/account/product, strategy/direction, isolated/cross leverage mode, position side, funding/borrow/collateral exposure and explicit unsupported reasons.
- Expose separate LONG/SHORT/FLAT positions, order/fill/fee/funding/PnL lifecycle, risk limits, flatten/cancel/kill switch and post-restart state; no demo selectors presented as production capability.
- Include per-strategy, per-direction and per-instrument cost/latency/slippage/edge/turnover and liquidation-risk telemetry.

**Acceptance:** native Windows UI E2E proves strategy-selection and direction gating on actual Core state; no operator or AI override of unavailable capability.

## TSP-7 — Full autonomous AI use of every **qualified** strategy and mode (Stage 12 dependency)

Extend Stage 12 ASD rather than creating a parallel AI trading/execution stack:

1. **Capability-aware policy context:** AI reads a fresh, signed/trusted or Core-authoritative capability snapshot of eligible instrument/account/environment/direction/strategy pairs, observed fees, liquidity, borrow/funding, risk budget, data quality and deployment status. Stale or ambiguous snapshot -> ABSTAIN.
2. **Common declarative strategy interface:** versioned `StrategySpec`, input requirements and market eligibility, output typed `TradeIntent` or auditable multi-leg plan, expected exposure and reason codes; all fifteen baseline strategy families and future plugins have adapters or explicit unsupported dispositions.
3. **Discovery and evaluation:** AI can compare directional LONG/SHORT and market-neutral options; train/tune/select/compose candidates, test cross-regime performance, walk-forward, strict OOS, realistic execution/carry/borrow/option costs, leverage/liquidation tails, capacity and correlation, with multiple-testing defenses.
4. **Runtime orchestration:** AI can autonomously decide **ABSTAIN / SELECT / START / PAUSE / REBALANCE / EXIT / RETIRE** among already qualified strategies and accounts, and adapt position sizing only inside pre-authorized budgets. Deterministic gates authorize every order and lifecycle transition independently.
5. **Safety boundary:** AI output never creates exchange permission, M0 capability, unlocks LIVE, changes approved leverage ceiling, self-approves promotion, disables risk, transfers/withdraws funds, or executes generated arbitrary Python. New strategies go through Shadow -> Paper -> evidence-backed Canary LIVE gates with independent approval.
6. **Explainability and reproducibility:** audit model/policy version, source capability snapshot, candidate hash, chosen/abstained strategy, eligible alternatives, rejected intents, cost assumptions, market regime, approval/gating results, fills and PnL; support rollback to a validated prior strategy set.

**Acceptance:** deterministic simulated decision tests cover each eligible strategy/direction/type pair and ABSTAIN; adversarial tests show unsupported SHORT, stale derivatives catalog, missing option risk data, partial multi-leg failure and AI-created candidates cannot reach an order without Core risk/capability authority. Human-facing autonomous mode clearly discloses approved scope and kill switch.

## TSP-8 — Qualification gates, milestones and handoff

- **M0 architecture:** inventory authority deltas; create a separately reviewed versioned contract/migration proposal for non-SPOT execution. Do not silently edit closed M0.6 policies.
- **M1 Core:** TSP-1/2 authoritative registry, typed semantics, strategy discovery, account/instrument identity and risk reservations.
- **M2 PAPER:** TSP-3/4/5 deterministic simulation/ledger invariants and every supported strategy's complete end-to-end PAPER vertical slice.
- **M3 TESTNET:** venue/product-specific evidence, no fictitious sandbox substitutes, reconciliation, failures and independently approved product support.
- **M4 first-beta UI:** TSP-6 features for the **approved beta subset only**; non-qualified types remain locked, without expanding beta freeze by implication.
- **M5 LIVE canary and M6:** separately authorized strategy/instrument/account combinations with scale controls; complete production hardening for explicitly shipped scope.
- **Stage 11 TIH:** realistic microstructure, TCA, tail-risk and market-regime validation, including short and carry costs.
- **Stage 12 ASD:** TSP-7 autonomy may use *every qualified* combination; unsupported combinations remain blocked. New strategy generation must follow ASD's existing independent promotion gates.

### Release/closure checklist

- [ ] Machine-readable complete family × venue/account × instrument × direction × environment matrix produced from authoritative code/registry and approved.
- [ ] All currently marketed capabilities have a QUALIFIED evidence pack or are downgraded/removed from marketing and disabled in runtime/UI/AI.
- [ ] No ambiguous BUY/SELL vs OPEN/CLOSE semantics, no implicit spot short, no unauthorized reversal.
- [ ] Zero-capital paper short, stale mark/index, funding spikes, borrow recall, liquidations, expiry/assignment, partial-fill/failed-leg and restart/duplicate tests pass.
- [ ] Core, PAPER, genuine supported TESTNET, risk/ledger/reconciliation and packaged desktop UI receipts cover each proposed launch combination.
- [ ] Future ASD can consume the versioned capability matrix and every qualified family, and reliably abstains whenever authority, provenance, market data or expected net edge are insufficient.
- [ ] LIVE promotion remains gated by M5, independent approval, kill switch, approved leverage/exposure limits, and canary evidence.

**Never conflate PLANNED documentation with an implemented trading or AI capability.**
