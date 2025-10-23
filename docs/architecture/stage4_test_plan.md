# Plan testów regresyjnych Etapu 4

## Cel
Zapewnienie, że rozszerzona biblioteka strategii oraz scheduler przechodzą pipeline demo → paper → live bez regresji funkcjonalnych i operacyjnych.

## Matryca testowa
| Obszar | Typ testu | Narzędzie | Zakres |
| --- | --- | --- | --- |
| Mean Reversion | unit/backtest | `pytest tests/test_mean_reversion_strategy.py`, `run_trading_stub_server.py` | sygnały wej./wyj., filtry zmienności |
| Volatility Target | unit/regresja | `pytest tests/test_volatility_target_strategy.py`, `telemetry_risk_profiles.py` | kalibracja alokacji, telemetria |
| Cross-Exchange Arbitrage | unit/integracja | `pytest tests/test_cross_exchange_arbitrage_strategy.py`, `run_trading_stub_server.py` | spread entry/exit, opóźnienia |
| Scheduler | async/integracja | `pytest tests/test_multi_strategy_scheduler.py`, `run_paper_smoke_ci.py` | telemetria, decision log |
| Ryzyko | regresja | `pytest tests/test_risk_engine.py::test_combined_strategy_orders_respect_max_position_pct`, `pytest tests/test_risk_engine.py::test_force_liquidation_due_to_drawdown_allows_only_reducing_orders` | limity pozycji multi-strategy, wymuszona likwidacja |
| Dane backtestowe | walidacja/regresja | `python scripts/validate_backtest_datasets.py`, `pytest tests/test_backtest_dataset_library.py tests/test_core_config_instrument_buckets.py` | spójność, braki, outliery, mapowanie profili |
| Telemetria i dashboardy | integracja | `python scripts/telemetry_risk_profiles.py audit`, `python scripts/watch_metrics_stream.py --dry-run`, `python scripts/watch_metrics_stream.py --headers-report-only --header x-trace=smoke` | metryki strategii, latencja scheduler-a, widgety OTEL, audyt nagłówków gRPC |
| Alerty | scenariusze operacyjne | `python scripts/run_metrics_service.py --simulate-alerts`, `pytest tests/test_alert_thresholds.py` | progi PnL/ryzyko/opóźnienia, eskalacje |
| CI/coverage | pipeline/regresja | `scripts/run_ci_pipeline.sh`, `pytest --cov=bot_core.strategies --cov=bot_core.runtime.multi_strategy_scheduler --cov-fail-under=85` | włączenie testów, progi coverage |
| Smoke demo CLI | smoke/integracja | `python scripts/smoke_demo_strategies.py --cycles 3` | validacja multi-strategy na danych demo |
| Bezpieczeństwo | audyt/manual | `python -m scripts.verify_decision_log audit/decision_logs/runtime.jsonl --schema builtin:decision_log_v2`, `python scripts/audit_security_baseline.py --scheduler-required-scope runtime.schedule.write` | HMAC, RBAC, mTLS, schemat decision log |
| Operacje | smoke/manual | `python scripts/run_multi_strategy_scheduler.py --demo-smoke`, `docs/runbooks/paper_trading.md` checklist | CLI smoke, playbook L1/L2 |
| Wydajność | obciążenie | `python scripts/load_test_scheduler.py`, `pytest tests/test_scheduler_load_test.py` | latencja, jitter, budżety zasobów |

## Procedura demo → paper
1. **Demo**: uruchom `run_trading_stub_server.py` z datasetem `tests/assets/scheduler_demo.yaml`; sprawdź telemetrię i decision log (`verify_decision_log.py`).
2. **Paper**: `run_paper_smoke_ci.py --environment binance_paper` – upewnij się, że scheduler rejestruje sygnały bez zleceń live.
3. **Live (kontrolowane)**: symulacja sucha – ładowanie konfigu `core_multi_pipeline`, walidacja RBAC (`security/token_audit.py`).

## Walidacja danych i procedury QA
1. Generuj znormalizowane zestawy OHLCV i spreadów (`python scripts/data/build_backtest_sets.py --strategies mean_reversion volatility_target cross_exchange_arbitrage`) – opcjonalne rozszerzenie poza próbkami repozytoryjnymi.
2. Uruchom `python scripts/validate_backtest_datasets.py --manifest data/backtests/normalized/manifest.yaml` – raport braków i odchyleń trafia do `audit/data_quality/*.json`.
3. Skoreluj wyniki z profilami ryzyka (`python scripts/telemetry_risk_profiles.py link-datasets --profile balanced`), aktualizując koszyki instrumentów.
4. Dodaj regresję do pipeline’u CI (`scripts/run_ci_pipeline.sh data-quality`) i monitoruj raport `data/reports/backtest_quality.md`; test `pytest tests/test_backtest_dataset_library.py` pełni rolę blokera schematu.

## Obserwowalność i alerty
1. Rozszerz `telemetry_risk_profiles.py render --section scheduler_metrics --section strategy_metrics` i potwierdź obecność metryk `avg_abs_zscore`, `avg_realized_volatility`, `allocation_error_pct`, `realized_vs_target_vol_pct`, `secondary_delay_ms`, `spread_capture_bps` oraz `decision_log_requirements`.
2. Zaktualizuj dashboard Grafany (`deploy/grafana/provisioning/dashboards/kryptolowca_overview.json`) i OTEL (`deploy/otel/strategies_dashboard.yaml`) – weryfikacja poprzez `python scripts/watch_metrics_stream.py --dashboard-check`.
3. Skonfiguruj alerty w `deploy/prometheus/rules/multi_strategy_rules.yml` z progami dla PnL, odchyleń ryzyka i opóźnień scheduler-a; zautomatyzuj eskalację `ops/oncall_rotation.yaml`.
4. Upewnij się, że logowanie decyzji zapisuje pola `schedule`, `strategy`, `confidence`, `latency_ms`, `telemetry_namespace`, podpisywane HMAC i agregowane w `audit/decision_log/`.
5. Zweryfikuj raport nagłówków gRPC: `python scripts/watch_metrics_stream.py --headers-report-only --header x-trace=demo --header auth-token=sekret` – wynik powinien maskować wartości wrażliwe oraz wskazywać źródła CLI/ENV/konfiguracji i listę kluczy usuniętych.

## Automatyzacja CI i smoke CLI
1. Włącz nowe testy i backtesty do `scripts/run_ci_pipeline.sh` (cele: `unit`, `integration`, `backtest`, `smoke_cli`, `coverage`).
2. Uruchamiaj `python scripts/smoke_demo_strategies.py --cycles 3` jako bramkę przed deploymentem paper; wynik w `logs/smoke_cli/*.log`.
3. Gating coverage: `pytest --cov=bot_core.strategies --cov=bot_core.runtime.multi_strategy_scheduler --cov-report xml --cov-fail-under=85` i walidacja progu (`python pytest_cov_stub.py validate --min 85`).

## Testy bezpieczeństwa i compliance
1. `python scripts/rbac_audit.py --config config/core.yaml` – walidacja przypisań ról dla scheduler-a i strategii.
2. `python -m scripts.verify_decision_log audit/decision_logs/runtime.jsonl --schema builtin:decision_log_v2` – sprawdzenie nowych pól; w razie potrzeby użyj `--list-schema-aliases`, aby potwierdzić dostępność aliasów schematu (`docs/schemas/decision_log_v2.json`).
3. Mini-audyt HMAC: `python scripts/key_rotation_check.py --context stage4` oraz aktualizacja `docs/architecture/iteration_gate_checklists.md`.

## Testy obciążeniowe i budżety zasobów
1. `python scripts/load_test_scheduler.py --iterations 180 --schedules 3 --output logs/load_tests/scheduler_profile.json` – generuje raport średniej latencji, jitteru i statusu budżetów.
2. `pytest tests/test_resource_monitor.py` oraz `python scripts/audit_security_baseline.py --print --scheduler-required-scope runtime.schedule.write` – walidacja budżetów CPU/RAM/I/O i audytu RBAC/mTLS.
3. Procedura rollbacku: `python scripts/run_multi_strategy_scheduler.py --disable-strategy <id>` + checklistę `docs/runbooks/rollback_multi_strategy.md`.

## Kryteria wyjścia
- 100 % pokrycia testów jednostkowych dla nowych strategii oraz spełniony próg coverage ≥ 85 % dla modułów strategii/scheduler-a.
- Telemetria scheduler-a raportuje `latency_ms < 250` oraz brak dryfu > `max_drift_seconds`; dashboardy Prometheus/OTEL i alerty przechodzą smoke testy.
- Decision log podpisany HMAC dla wszystkich sygnałów, zawiera pola multi-strategy, audyt RBAC/mTLS zakończony sukcesem.
- Risk engine blokuje przekroczenia `max_position_pct` przy kumulacji ekspozycji; smoke CLI oraz testy obciążeniowe zakończone bez regresji budżetów zasobów.

