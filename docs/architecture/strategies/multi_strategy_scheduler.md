# Harmonogram wielostrate-giczny

## Przegląd
- Komponent runtime (`bot_core.runtime.multi_strategy_scheduler.MultiStrategyScheduler`) zarządza wieloma strategiami bazującymi na kontrakcie `StrategyEngine`.
- Każda strategia rejestrowana jest z parametrami `cadence_seconds`, `max_drift_seconds`, `warmup_bars`, `risk_profile`, `max_signals` oraz `interval` (dla feedu danych).
- Telemetria (`TelemetryEmitter`) raportuje liczbę sygnałów i opóźnienie (`latency_ms`).
- Decyzje każdej strategii zapisywane są w `TradingDecisionJournal` wraz z metadanymi (schedule, strategia, confidence, HMAC).

## Architektura
1. **Źródło danych (`OHLCVStrategyFeed`)** – pobiera świece z cache (Parquet/SQLite) i mapuje je na `MarketSnapshot`.
2. **Scheduler** – asynchronicznie uruchamia `_execute_schedule`, ograniczając liczbę sygnałów (`max_signals`) i dbając o dryf czasowy.
3. **Sink (`InMemoryStrategySignalSink`)** – buforuje sygnały do audytu/regresji; w środowisku produkcyjnym można podmienić na sink wykonawczy.
4. **Builder (`build_multi_strategy_runtime`)** – scala bootstrap, scheduler, feed i strategie na podstawie `CoreConfig`.
5. **CLI (`scripts/run_multi_strategy_scheduler.py`)** – uruchamia scheduler w trybie `--run-once` (audit) lub `--run-forever` (paper/live), korzystając z telemetry stdout.

## Bezpieczeństwo
- RBAC: token `CORE_SCHEDULER_TOKEN` wymagany w `config/core.yaml` (sekcja `runtime.multi_strategy_schedulers`).
- Kanały komunikacji: wyłącznie gRPC/HTTP2 lub IPC – brak WebSocketów.
- Decision log podpisywany HMAC; walidacja narzędziem `verify_decision_log.py`.

## Testy
- Jednostkowe: `tests/test_multi_strategy_scheduler.py` (w tym telemetria i decision journal).
- Integracyjne: `PYTHONPATH=. pytest tests/test_runtime_pipeline.py` – weryfikacja buildera.
- Smoke CI: `PYTHONPATH=. python scripts/run_multi_strategy_scheduler.py --environment binance_paper --run-once`.

