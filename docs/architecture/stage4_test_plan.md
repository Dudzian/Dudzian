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
| Ryzyko | regresja | `pytest tests/test_risk_engine.py::test_combined_strategy_orders_respect_max_position_pct` | limity pozycji multi-strategy |

## Procedura demo → paper
1. **Demo**: uruchom `run_trading_stub_server.py` z datasetem `tests/assets/scheduler_demo.yaml`; sprawdź telemetrię i decision log (`verify_decision_log.py`).
2. **Paper**: `run_paper_smoke_ci.py --environment binance_paper` – upewnij się, że scheduler rejestruje sygnały bez zleceń live.
3. **Live (kontrolowane)**: symulacja sucha – ładowanie konfigu `core_multi_pipeline`, walidacja RBAC (`security/token_audit.py`).

## Kryteria wyjścia
- 100 % pokrycia testów jednostkowych dla nowych strategii.
- Telemetria scheduler-a raportuje `latency_ms < 250` oraz brak dryfu > `max_drift_seconds`.
- Decision log podpisany HMAC dla wszystkich sygnałów.
- Risk engine blokuje przekroczenia `max_position_pct` przy kumulacji ekspozycji.

