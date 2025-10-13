# Harmonogram wielostrate-giczny

## Przegląd
- Komponent runtime (`bot_core.runtime.multi_strategy_scheduler.MultiStrategyScheduler`) zarządza wieloma strategiami bazującymi na kontrakcie `StrategyEngine`.
- Każda strategia rejestrowana jest z parametrami `cadence_seconds`, `max_drift_seconds`, `warmup_bars`, `risk_profile`, `max_signals` oraz `interval` (dla feedu danych).
- Telemetria (`TelemetryEmitter`) raportuje liczbę sygnałów, opóźnienie (`latency_ms`), średnią pewność (`avg_confidence`) oraz metryki specyficzne: `avg_abs_zscore`, `avg_realized_volatility`, `allocation_error_pct`, `realized_vs_target_vol_pct`, `secondary_delay_ms`, `spread_capture_bps`.
- Decyzje każdej strategii zapisywane są w `TradingDecisionJournal` z polami `schedule`, `strategy`, `confidence`, `latency_ms`, `telemetry_namespace` oraz podpisami HMAC.

## Architektura
1. **Źródło danych (`OHLCVStrategyFeed`)** – pobiera świece z cache (Parquet/SQLite) i mapuje je na `MarketSnapshot`.
2. **Scheduler** – asynchronicznie uruchamia `_execute_schedule`, ograniczając liczbę sygnałów (`max_signals`) i dbając o dryf czasowy.
3. **Sink (`InMemoryStrategySignalSink`)** – buforuje sygnały do audytu/regresji; w środowisku produkcyjnym można podmienić na sink wykonawczy.
4. **Builder (`build_multi_strategy_runtime`)** – scala bootstrap, scheduler, feed i strategie na podstawie `CoreConfig`.
5. **CLI (`scripts/run_multi_strategy_scheduler.py`, `scripts/smoke_demo_strategies.py`)** – pierwsze narzędzie uruchamia scheduler w trybie `--run-once` (audit) lub `--run-forever` (paper/live), drugie rejestruje cykle demo na znormalizowanych datasetach backtestowych.

## Bezpieczeństwo
- RBAC: token `CORE_SCHEDULER_TOKEN` wymagany w `config/core.yaml` (sekcja `runtime.multi_strategy_schedulers`).
- Kanały komunikacji: wyłącznie gRPC/HTTP2 lub IPC – brak WebSocketów.
- Decision log podpisywany HMAC; walidacja narzędziem `verify_decision_log.py`.

## Testy
- Jednostkowe: `tests/test_multi_strategy_scheduler.py` (w tym telemetria i decision journal).
- Integracyjne: `PYTHONPATH=. pytest tests/test_runtime_pipeline.py` – weryfikacja buildera.
- Smoke CI: `python scripts/smoke_demo_strategies.py --cycles 3`.

