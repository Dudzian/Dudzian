# FULL MODULAR BACKEND AUDIT — Premium SaaS Trading Platform Capability Map vs Existing Backend

Data audytu: 2026-06-23. Scope: discovery-only, bez implementacji, refactoru, runtime enablement, wiringu, testnet/live tradingu, QML/UI zmian, packagingu i zmian dependency declarations.

## 1. Executive summary

Audyt porównuje istniejący backend repo z klasą premium SaaS trading platform typu Cryptohopper / 3Commas. Repo zawiera szeroką warstwę backendową: strategie, backtesting, paper execution, adaptery giełd, execution, risk, portfolio, AI/ML, marketplace/presety, observability, runtime/gRPC i security. Jednocześnie wiele elementów jest **niepodłączonych do aplikacji**, **blokowanych bramkami bezpieczeństwa**, albo ma status contract/static/preview. BLOK H/read-only market data preview jest zgodny z repozytoryjnymi markerami bezpieczeństwa, a następny logiczny blok to BLOK I — testnet/sandbox adapter contract-first.

Najważniejszy werdykt: twierdzenie, że backend zawiera większość modułów klasy premium SaaS poza copy/social trading, jest **częściowo prawdziwe na poziomie kodu modułów**, ale **nieprawdziwe na poziomie pełnego produktu SaaS**. Pokrycie funkcjonalne backendu jest wysokie, natomiast wiring aplikacyjny, testnet readiness, operator readiness, SaaS multi-tenant i billing readiness są ograniczone lub brakujące.

Ruff Python changes: N/A — no Python files changed.

## 2. Premium SaaS benchmark scope

Benchmark obejmuje 48 kategorii wskazanych w zleceniu: bot core, strategie, designer, wskaźniki, sygnały/webhook, backtesting, paper/testnet/live gates, market/exchange adapters, orders/lifecycle/positions/portfolio/risk, DCA/grid/arbitrage/market making/trailing/SL/TP, terminal, AI/training, templates/marketplace/signals marketplace, copy/social, accounts/secrets/permissions/audit/observability/recovery/kill switch/orchestration/scheduler/UI/desktop/SaaS/billing/security/deploy/docs/tests.

## 3. Repo discovery method

Uruchomiono wymagane komendy discovery i preflight:

- `git status --short`
- `find . -maxdepth 3 -type f | sort | sed -n '1,240p'`
- `rg -n "class .*Strategy|Strategy|Backtest|backtest|Signal|TradingView|webhook|DCA|Grid|Arbitrage|MarketMaking|market making|Trailing|TakeProfit|StopLoss|Portfolio|Position|Order|Fill|Lifecycle|Risk|Governor|KillSwitch|kill switch|MarketData|Exchange|Adapter|Sandbox|Testnet|Paper|Simulator|AI|Scoring|Model|Train|Template|Preset|Marketplace|Copy|Social|Audit|Telemetry|Observability|Metrics|Health|Rollback|Scheduler|Runtime|Loop|Secret|Credential|API key|account|balance|permission|tenant|subscription|billing|PyInstaller|EXE|QML|Bridge" .`
- `rg -n "binance|bybit|okx|coinbase|kraken|kucoin|bitget|gate|htx|gemini|bitstamp|ccxt|websocket|aiohttp|httpx|requests|grpc|protobuf" .`
- `rg -n "TODO|FIXME|placeholder|mock|fixture|stub|not implemented|contract-only|dry-run|preview-only|live disabled|testnet|sandbox" .`
- `python scripts/dev/ensure_ui_runtime_deps.py --install`
- `python scripts/dev/ensure_pyside6.py --install` — wykonane, bo skrypt istnieje.
- `scripts/dev/ensure_linux_qt_native_deps.sh` — wykonane, bo Linux + apt-get + skrypt istnieje.
- `python scripts/dev/ensure_ui_runtime_deps.py`
- `python scripts/dev/ensure_pyside6.py`
- Import proof: `yaml`, `cryptography`, `grpc`, `google.protobuf`, `numpy`, `pandas`.

Dodatkowo wykonano lokalny, statyczny odczyt nazw plików i symboli przez Python AST. Nie uruchomiono live/testnet tradingu, giełd, fetchy account/balance/order/fill, runtime loopów ani schedulerów.

## 4. High-level backend map

- `bot_core/strategies/`: realne strategie i katalog/presety, m.in. DCA, Grid, market making, adaptive market making, arbitrage, scalping, trend, mean reversion.
- `bot_core/trading/`: engine, wskaźniki, autotrade, aliasy strategii i pluginy sygnałowe.
- `bot_core/backtest/`: engine backtestów, providers, matching simulation, metrics, walk-forward, raporty.
- `bot_core/exchanges/`: adaptery CCXT/native dla wielu giełd, network guard, rate limiter, health/circuit breaker, stream gateway.
- `bot_core/execution/`: paper execution, live router, mode policy, execution service, bridge decyzja→order.
- `bot_core/runtime/`: bootstrap, controller, preview/read-only contracts, paper event spine, portfolio reducer, market/risk/metrics services, schedulers, state manager.
- `bot_core/risk/`: risk engine, profiles, guardrails, simulation, portfolio stress, risk state/events.
- `bot_core/portfolio/`: governor, scoring, scheduler, decision logs, hypercare, payout metadata.
- `bot_core/ai/` i `bot_core/strategies/ml/`: AI manager, training, validation, monitoring, feature engineering, model adapters.
- `bot_core/marketplace/`, `bot_core/config_marketplace/`, `bot_core/strategies/marketplace/`: presety, podpisy, workflow publikacji.
- `bot_core/security/`, `bot_core/cloud/`, `bot_core/compliance/`: licencja, sekrety/security, cloud service, compliance reports.
- `proto/trading.proto`, `bot_core/generated/`: gRPC kontrakty usług MarketData/Order/Risk/Metrics/Runtime/Health/Marketplace.

## 5. Capability matrix

| Premium SaaS capability | Repo status | Evidence path(s) | Main module/class/function | Current maturity | Wired to app? | Safe for testnet? | Needs risk governor? | Needs observability/soak? | Needs live gate? | Missing pieces | Recommended block |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Trading bot core | exists_but_blocked | `bot_core/trading/engine.py`, `bot_core/trading/auto_trade.py`, `bot_core/runtime/controller.py` | `AutoTradeEngine`, `TechnicalIndicators` | implemented backend, high-risk runtime | częściowo | nie bez BLOKU I/J | tak | tak | tak | controlled runtime wiring | J/L |
| Strategy engine | implemented_not_wired | `bot_core/strategies/base.py`, `bot_core/runtime/strategy_builder.py` | `BaseStrategy`, `StrategyEngine`, `instantiate_strategies` | realny | częściowo | po contract gate | tak | tak | tak | safe handoff do runtime | I/J |
| Strategy designer / rules builder | partial | `bot_core/runtime/strategy_builder.py`, `bot_core/strategies/catalog.py`, `bot_core/strategies/presets/installer.py` | `StrategyFactory`, preset validation | backend/catalog | częściowo | tak dla static/preset | tak | tak | tak | pełny visual rules builder | osobny block |
| Technical indicators | implemented | `bot_core/trading/engine.py`, `bot_core/strategies/_volatility.py` | `TechnicalIndicators`, `realized_volatility` | realny | częściowo | tak offline | tak | tak | tak | szersza biblioteka TA | J |
| Signal ingestion | partial | `bot_core/runtime/data_pipeline.py`, `bot_core/runtime/multi_strategy_scheduler.py` | `InMemoryStrategySignalSink`, `StrategySignalSink` | realny/offline | częściowo | po adapter gate | tak | tak | tak | external signal auth/limits | I/J |
| TradingView/webhook style signals | missing | rg nie wykazał realnego webhook receivera | brak | missing | nie | nie | tak | tak | tak | webhook API, signing, replay protection | osobny block |
| Backtesting | implemented | `bot_core/backtest/engine.py`, `bot_core/backtest/walk_forward.py`, `bot_core/backtest/simulation.py` | `BacktestEngine`, `WalkForwardBacktester`, `MatchingEngine` | realny | nie app-core | tak offline | nie/częściowo | tak | nie | UI/operator integration | K |
| Paper trading / simulator | implemented_not_wired | `bot_core/execution/paper.py`, `bot_core/runtime/paper_event_spine.py`, `bot_core/runtime/paper_portfolio_reducer.py` | `PaperTradingExecutionService`, `PaperEventSpine` | realny + preview | częściowo preview | tak offline | tak | tak | tak dla przejścia | app/runtime wiring | I/J |
| Testnet/sandbox trading | contract_only | `config/exchanges/*.yaml`, `bot_core/exchanges/manager.py`, `tests/scripts/test_sandbox_testnet_readiness.py` | `_enable_sandbox_mode`, environment configs | config/test evidence | nie | dopiero BLOK I | tak | tak | tak | sandbox adapter contract, no live order path | I |
| Live trading readiness gates | exists_but_blocked | `bot_core/execution/live_router.py`, `bot_core/runtime/canary_gate.py`, `bot_core/execution/mode_policy.py` | `LiveExecutionRouter`, `evaluate_runtime_canary_gate` | high-risk gated | nie dla live | nie | tak | tak | tak | formal live transition + signoffs | L |
| Market data adapters | ready_for_contract_gate | `bot_core/runtime/read_only_market_data.py`, `bot_core/runtime/market_data_service.py`, `bot_core/exchanges/ccxt_adapter.py` | `ReadOnlyMarketDataProvider`, `MarketDataServiceServicer` | read-only safe path + adapters | częściowo preview | tak read-only | nie | tak | tak | testnet market data contract | I |
| Exchange adapters | high_risk_requires_gate | `bot_core/exchanges/*`, `bot_core/exchanges/factory.py`, `config/exchanges/*.yaml` | `build_exchange_adapter`, native adapters | szerokie realne adaptery | częściowo | nie bez BLOKU I | tak | tak | tak | sandbox-only policy, credentials gates | I/J/L |
| Order management | exists_but_blocked | `bot_core/exchanges/base.py`, `bot_core/execution/bridge.py`, `proto/trading.proto` | `OrderRequest`, `decision_to_order_request`, `SubmitOrder` | realny kontrakt | częściowo | nie bez gate | tak | tak | tak | order gate, idempotency, audit | J/L |
| Order lifecycle | implemented_not_wired | `bot_core/runtime/paper_event_spine.py`, `bot_core/exchanges/core.py`, `tests/execution/test_exchange_lifecycle_live.py` | `PaperOrderStatus`, `OrderStatus` | realny/paper | częściowo | offline only | tak | tak | tak | testnet fill reconciliation | I/J |
| Position management | partial | `bot_core/exchanges/core.py`, `bot_core/risk/state.py`, `bot_core/runtime/paper_portfolio_reducer.py` | `PositionDTO`, `PositionState`, `PaperPosition` | realny/paper | częściowo | offline/paper | tak | tak | tak | exchange reconciliation | I/J |
| Portfolio management | implemented_not_wired | `bot_core/portfolio/governor.py`, `bot_core/runtime/portfolio_runtime.py` | `PortfolioGovernor`, `build_portfolio_runtime` | realny | częściowo | po gate | tak | tak | tak | operator workflows | J/K |
| Risk management | implemented | `bot_core/risk/engine.py`, `bot_core/risk/guardrails.py`, `bot_core/runtime/risk_service.py` | `ThresholdRiskEngine`, `evaluate_backtest_guardrails`, `RiskService` | realny | częściowo | potrzebny jako gate | tak | tak | tak | central kill/approval policy | J |
| DCA bot | implemented_not_wired | `bot_core/strategies/dca.py`, `tests/strategies/test_strategy_plugins.py` | `DollarCostAveragingStrategy` | realny | nie pełne app | po gate | tak | tak | tak | runtime/order gating | J |
| Grid bot | implemented_not_wired | `bot_core/strategies/grid.py`, tests strategy catalog | `GridTradingStrategy` | realny | nie pełne app | po gate | tak | tak | tak | runtime/order gating | J |
| Arbitrage bot | implemented_not_wired | `bot_core/strategies/cross_exchange_arbitrage.py`, `bot_core/strategies/triangular_arbitrage.py`, `bot_core/strategies/statistical_arbitrage.py` | arbitrage strategies | realny backend | nie pełne app | nie bez multi-exchange gate | tak | tak | tak | latency, balances, fees, execution safety | J/L |
| Market making bot | implemented_not_wired | `bot_core/strategies/market_making.py`, `bot_core/strategies/adaptive_market_making.py` | `MarketMakingStrategy`, `AdaptiveMarketMakingStrategy` | realny | nie pełne app | nie bez gate | tak | tak | tak | inventory/risk/liquidity controls | J/K/L |
| Trailing stop / trailing take profit | partial | `tests/test_trading_controller.py`, `bot_core/trading/exit_reasons.py` | controller test evidence, exit reasons | appears in controller tests | nie audytowano runtime due do-not-run | nie | tak | tak | tak | explicit backend module/gate | J |
| Stop loss / take profit / multi take profit | partial | `bot_core/trading/engine.py`, `tests/test_trading_controller.py` | `EngineConfig`, controller tests | częściowo | częściowo | nie bez gate | tak | tak | tak | formal lifecycle coverage | J |
| Smart/manual trade terminal | fixture_only | `ui/pyside_app/qml/views/PaperTerminal.qml`, `tests/ui_pyside/test_source_smoke.py` | QML preview/mock fields | preview/mock UI, poza backend | nie | nie dotyczy | tak | tak | tak | backend terminal service | osobny block |
| AI/scoring/strategy intelligence | implemented_not_wired | `bot_core/ai/*`, `bot_core/decision/*`, `bot_core/portfolio/scoring.py` | `AIManager`, decision orchestrators, scoring | realny | częściowo | offline/paper only | tak | tak | tak | production inference gates | J/K |
| Model training / evaluation / overfit controls | partial | `bot_core/ai/training.py`, `bot_core/ai/validation.py`, `bot_core/monitoring/*`, `bot_core/strategies/ml/*` | training/validation/adapters | realny/partial | nie pełne app | offline | tak | tak | tak | dataset governance, drift SOP | K |
| Templates / presets | implemented | `bot_core/marketplace/presets.py`, `bot_core/strategies/presets/installer.py`, `bot_core/strategies/marketplace/catalog.yaml` | preset models/installers | realny | częściowo | tak static | nie | tak | tak | UI/operator lifecycle | K |
| Marketplace-like internal templates/catalog | implemented | `bot_core/marketplace/service.py`, `bot_core/config_marketplace/workflow.py`, `tests/marketplace/*` | marketplace service/workflow | realny internal | częściowo | tak static | nie | tak | tak | external marketplace governance | K |
| Signals marketplace readiness | docs_only | `bot_core/marketplace/*` only preset-oriented | brak signal marketplace | docs/adjacent only | nie | nie | tak | tak | tak | provider auth, billing, quality scores | osobny block |
| Copy trading | partial | `bot_core/portfolio/scheduler.py` | `CopyTradingFollowerConfig`, `CopyTradeInstruction` | model/scheduler fragment | nie | nie | tak | tak | tak | full copy trading product | out_of_scope/osobny |
| Social trading | out_of_scope | brak realnych backend modułów social trading | brak | missing/out-of-scope | nie | nie | tak | tak | tak | social graph/feed/reputation | out_of_scope |
| User/account management | partial | `bot_core/cloud/*`, `bot_core/security/*`, `bot_core/api/server.py` | cloud/security service | partial/licensing | częściowo | nie | tak | tak | tak | SaaS users/roles/orgs | osobny block |
| API key / secrets management | implemented_not_wired | `bot_core/security/*`, `bot_core/runtime/preset_service.py`, `tests/ui/test_local_security_store.py` | security store/preset flattening | realny | częściowo | po secret gate | tak | tak | tak | exchange key UX + rotation SOP | I/J |
| Permissions / private endpoint gates | implemented_not_wired | `bot_core/exchanges/network_guard.py`, `bot_core/execution/mode_policy.py`, `bot_core/runtime/preview_modes.py` | `ExchangeNetworkGuard`, `ExecutionModePolicy` | realny | częściowo | po gate | tak | tak | tak | central policy enforcement | I/J |
| Audit log / event log | implemented | `bot_core/runtime/paper_audit_journal.py`, `bot_core/risk/events.py`, `bot_core/portfolio/decision_log.py` | audit/decision logs | realny | częściowo | tak offline | tak | tak | tak | immutable production audit trail | K/L |
| Observability / metrics / health | implemented_not_wired | `bot_core/observability/*`, `bot_core/runtime/metrics_service.py`, `bot_core/exchanges/health.py` | metrics/health/SLO | realny | częściowo | po soak | tak | tak | tak | full soak dashboards/alerts | K |
| Rollback / recovery | partial | `bot_core/runtime/state_manager.py`, `bot_core/resilience/*`, `scripts/run_recovery_matrix.py` | checkpoint/resilience | partial | częściowo | offline | tak | tak | tak | exchange/order rollback procedures | K/L |
| Kill switch | partial | `bot_core/execution/live_router.py`, `bot_core/runtime/canary_gate.py`, UI tests mention local kill switch | circuit/canary | partial | częściowo | po gate | tak | tak | tak | explicit global kill switch service | J/K |
| Runtime orchestration | implemented_not_wired | `bot_core/runtime/bootstrap.py`, `bot_core/runtime/pipeline.py`, `bot_core/runtime/controller.py` | bootstrap/pipeline/controller | realny, high-risk | częściowo | nie bez BLOKU I/J | tak | tak | tak | safe activation path | I/J/K |
| Scheduler / loop management | implemented_not_wired | `bot_core/runtime/scheduler.py`, `bot_core/runtime/multi_strategy_scheduler.py`, `bot_core/ai/scheduler.py` | scheduler classes | realny | częściowo | nie uruchamiać bez gate | tak | tak | tak | lifecycle limits + soak | K |
| UI bridge / frontend binding | partial | `bot_core/runtime/ui_bridge.py`, `proto/trading.proto`, `bot_core/generated/*` | gRPC/UI bridge | partial | częściowo preview | nie dla orders | tak | tak | tak | controlled runtime binding | osobny UI block |
| Desktop app readiness | partial | `ui/`, `deploy/packaging/`, `build_preview_exe_windows_DEBUG_v2.bat` | desktop packaging artifacts | poza audytem backend | częściowo | nie dotyczy | tak | tak | tak | release gates, no changes here | later |
| SaaS/multi-tenant readiness | missing | `bot_core/cloud/*` adjacent | brak tenant model | missing/partial cloud | nie | nie | tak | tak | tak | orgs, tenants, isolation | osobny block |
| Billing/subscription readiness | missing | rg billing/subscription minimal/none | brak | missing | nie | nie | nie | tak | tak | billing, plans, entitlements | osobny block |
| Security / compliance / regulatory disclaimers | partial | `bot_core/security/*`, `bot_core/compliance/*`, `audit/security/*` | reports/security modules | partial | częściowo | po gate | tak | tak | tak | legal disclaimers, regional rules | L/compliance |
| Deployment / packaging / EXE readiness | partial | `deploy/packaging/*`, `.github/workflows/*`, `.bat` | bundle builders | partial | nie dotyczy | nie dotyczy | nie | tak | tak | production release process | later |
| Documentation / operator runbooks | partial | `reports/*`, `docs/*`, `proto/README.md` | runbooks/reports | partial | nie | tak docs | nie | tak | tak | operator runbooks per mode | K/L |
| Test coverage / smoke / CI readiness | implemented_not_wired | `tests/`, `.github/workflows/*`, `reports/ci_command_map.md` | many unit/static/integration tests | szerokie, uneven | częściowo | offline tests ok | tak | tak | tak | no live/testnet E2E by default | K |

## 6. Module-by-module evidence

- Strategies: `bot_core/strategies/base.py` definiuje `MarketSnapshot`, `StrategySignal`, `StrategyEngine`, `BaseStrategy`; konkretne strategie obejmują `DollarCostAveragingStrategy`, `GridTradingStrategy`, `MarketMakingStrategy`, `AdaptiveMarketMakingStrategy`, `CrossExchangeArbitrageStrategy`, `TriangularArbitrageStrategy`, `StatisticalArbitrageStrategy`, `ScalpingStrategy`, `DailyTrendMomentumStrategy`, `MeanReversionStrategy`, `VolatilityTargetStrategy`.
- Trading core: `bot_core/trading/engine.py` zawiera `TechnicalIndicators`, `Trade`, `EngineConfig`; `bot_core/trading/auto_trade.py` zawiera `AutoTradeEngine` i konfigurację auto-risk freeze.
- Backtest: `bot_core/backtest/engine.py`, `walk_forward.py`, `simulation.py`, `metrics.py`, `reporting.py` tworzą kompletny offline stack.
- Execution: `bot_core/execution/paper.py` jest realnym paper execution; `live_router.py` i `execution_service.py` istnieją, ale wymagają live/risk gates.
- Exchanges: `bot_core/exchanges/` zawiera bazowe DTO/adaptery, CCXT, native spot/futures/margin, rate limiting, network guard, health/circuit breaker i streaming.
- Risk: `bot_core/risk/engine.py`, `guardrails.py`, `simulation.py`, `portfolio_stress.py`, `state.py`; runtime ma `risk_service.py`.
- Portfolio: `bot_core/portfolio/governor.py`, `asset_governor.py`, `strategy_governor.py`, `scheduler.py`, `decision_log.py`.
- AI/ML: `bot_core/ai/manager.py`, `training.py`, `validation.py`, `monitoring.py`, `feature_engineering.py`, plus `bot_core/strategies/ml/*`.
- Observability: `bot_core/observability/metrics.py`, `server.py`, `slo.py`, `risk.py`, `dashboard_sync.py`; runtime metrics/alerts services.
- Preview safety: `bot_core/runtime/read_only_market_data.py` i `paper_preview_*` moduły wskazują na static-local/read-only/refusal-matrix podejście.

## 7. Existing modules that match premium SaaS capabilities

1. Multi-strategy engine and concrete strategy catalog.
2. Backtesting/walk-forward/matching engine.
3. Paper execution and paper event lifecycle.
4. Exchange adapter layer with CCXT/native adapters.
5. Risk engine/guardrails/profile/state.
6. Portfolio governor/scoring/scheduler.
7. AI/ML intelligence/training/validation modules.
8. Marketplace/presets/signature workflows.
9. Observability/metrics/health/SLO modules.
10. gRPC contracts for MarketData/Order/Risk/Metrics/Runtime/Health/Marketplace.

## 8. Existing modules not yet wired into the app

Top candidates: DCA, Grid, arbitrage, market making, adaptive market making, strategy builder, backtest UI/operator flow, portfolio governor runtime, AI/ML training/inference production path, exchange private order paths, metrics service, scheduler loops, live router, copy-trading fragments in portfolio scheduler.

## 9. Modules ready for BLOK I / testnet-sandbox contract

- Read-only market data provider/service as a safe input contract.
- Exchange environment configs with `testnet`/`sandbox` metadata.
- Exchange manager sandbox-mode mechanics.
- Paper execution and paper event spine as comparison oracle, not live route.
- Network guard/rate limiter as enforcement primitives.
- gRPC proto contracts for order/risk/market data as contract surfaces.

## 10. Modules that must wait for BLOK J / risk governor

Order management, order lifecycle, live/testnet execution bridge, DCA/Grid/arbitrage/market making runtime activation, position management, portfolio rebalancing, auto trade engine, kill-switch/circuit breaker centralization.

## 11. Modules that must wait for BLOK K / observability-soak

Runtime orchestration, schedulers/loops, metrics service, risk service streaming, portfolio hypercare, AI monitoring, exchange health, alerting, state rollback/recovery, audit retention.

## 12. Modules that must wait for BLOK L / live transition

Live execution router, private exchange endpoints, account/balance/order/fill APIs, live credentials, production AI inference affecting orders, live market making/arbitrage, compliance/legal signoff, operator runbooks.

## 13. Missing modules

- TradingView/webhook receiver with signing/replay protection.
- Full smart/manual trade terminal backend.
- Signals marketplace product.
- SaaS multi-tenant identity/org isolation.
- Billing/subscriptions/entitlements.
- Full social trading product.
- Explicit global kill switch service as one owner.
- Testnet order/fill reconciliation proof.
- Operator runbooks per mode.
- Commercial compliance/regulatory workflow.

## 14. Out-of-scope modules: copy trading and social trading

Social trading is out-of-scope and effectively missing. Copy trading is not fully missing because `bot_core/portfolio/scheduler.py` contains `CopyTradingFollowerConfig` and `CopyTradeInstruction`, but this is not a complete copy-trading product; classify as partial/out-of-scope unless a separate product block is opened.

## 15. Safety blockers

- Private exchange I/O and live/testnet order calls must remain blocked until BLOK I/J/L gates.
- Existing live router and native exchange adapters are high-risk without mode policy, credentials gate, risk governor, audit and observability soak.
- Runtime loops/schedulers must not be enabled merely because classes exist.
- `.env` exists in repo root; audit did not copy secret values.
- QML/PaperTerminal evidence is preview/mock-only for terminal features, not backend trading capability.

## 16. Test coverage map

- Strategies: `tests/strategies/*`, `tests/test_daily_trend_strategy.py`, `tests/test_scalping_strategy.py`, `tests/test_options_income_strategy.py`.
- Backtest: `tests/test_backtest_metrics.py`, `tests/test_backtest_matching_engine.py`.
- Execution: `tests/execution/*`, `tests/test_paper_execution.py`, `tests/test_live_router.py`, `tests/test_live_execution_dry_run.py`.
- Exchanges: `tests/integration/exchanges/*`, `tests/test_exchange_manager_native.py`, per-exchange tests.
- Risk/portfolio: `tests/test_risk_engine.py`, `tests/risk/*`, `tests/test_portfolio_governor.py`, `tests/test_portfolio_io.py`.
- Runtime/observability: `tests/runtime/*`, `tests/test_runtime_scheduler.py`, `tests/test_metrics_service_tls.py`, `tests/test_observability_bundle_script.py`.
- UI bridge/static preview: `tests/ui_pyside/test_source_smoke.py`, `tests/ui_backend/*`; not modified or run.
- Gaps: no safe live/testnet E2E run in this audit; no webhook tests found; no billing/tenant/social tests found.

## 17. Risk assessment

Risk is medium-high for accidentally treating existing modules as production-ready. The repo is module-rich, but the critical product risk is activation/wiring rather than absence of code. Highest-risk areas: live router, private exchange adapters, account/balance/order/fill calls, runtime loops, market making/arbitrage, secrets handling, and AI-driven order generation.

## 18. Recommended next blocks

1. BLOK I — testnet/sandbox adapter contract: sandbox-only adapter boundary, no real live route, no balances/orders unless explicitly stubbed/contracted.
2. BLOK J — risk governor: central pre-order gate, global kill switch, idempotency/audit, position/order constraints.
3. BLOK K — observability soak: metrics, health, SLO, alerting, state snapshots, recovery drills.
4. BLOK L — live transition: live credentials, legal/compliance, runbooks, operator approvals, limited rollout.
5. Separate commercial SaaS blocks: tenants/users/billing/signals marketplace/social/copy trading.

## 19. Final verdict: how close backend is to premium SaaS product class

Audit estimates, intentionally coarse:

- Backend capability coverage vs premium SaaS: **68%** — many core trading/backtest/risk/exchange/AI/portfolio modules exist; SaaS/billing/webhook/social/operator gaps remain.
- App wiring coverage: **32%** — preview/UI/static/gRPC evidence exists, but many modules are not safely wired into app runtime.
- Testnet readiness: **38%** — configs and sandbox mechanics exist, but BLOK I contract and safe execution boundary are not complete.
- Risk readiness: **55%** — strong risk modules exist, but central pre-order governor/kill-switch/live gate integration remains.
- Observability readiness: **52%** — metrics/health/SLO modules exist, but soak and production dashboards/gates are not complete.
- Live readiness: **18%** — live-capable code exists, but must remain blocked pending gates/compliance/soak.
- UI/operator readiness: **35%** — UI preview/static tests are broad, but operator backend activation/runbooks remain partial.
- Commercial SaaS readiness: **22%** — no full tenants/billing/social/signals marketplace product.

Final: repo is closer to a **module-rich trading backend platform** than to a fully wired premium SaaS product. It has enough inventory to plan BLOK I/J/K without adding features now.

## 20. Appendix: raw file evidence list

Key files/directories: `bot_core/strategies/`, `bot_core/trading/`, `bot_core/backtest/`, `bot_core/exchanges/`, `bot_core/execution/`, `bot_core/runtime/`, `bot_core/risk/`, `bot_core/portfolio/`, `bot_core/ai/`, `bot_core/decision/`, `bot_core/marketplace/`, `bot_core/config_marketplace/`, `bot_core/security/`, `bot_core/cloud/`, `bot_core/compliance/`, `proto/trading.proto`, `bot_core/generated/`, `config/exchanges/*.yaml`, `tests/`, `reports/`, `docs/`.

Potential secret file observed: `.env` exists; values were not read into/report copied.
