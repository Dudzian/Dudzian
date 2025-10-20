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
4. **Builder (`build_multi_strategy_runtime`)** – scala bootstrap, scheduler, feed i strategie na podstawie `CoreConfig`. Dostępne są gotowe warianty `build_demo_multi_strategy_runtime`, `build_paper_multi_strategy_runtime` oraz `build_live_multi_strategy_runtime`, które automatycznie wybierają odpowiednie środowisko na podstawie konfiguracji (preferencje: aliasy nazwy, typ środowiska i tryb offline).
5. **CLI (`scripts/run_multi_strategy_scheduler.py`, `scripts/smoke_demo_strategies.py`)** – pierwsze narzędzie uruchamia scheduler w trybie `--run-once` (audit) lub `--run-forever` (paper/live), drugie rejestruje cykle demo na znormalizowanych datasetach backtestowych. Przykład uruchomienia paper: `python scripts/run_multi_strategy_scheduler.py --environment binance_paper --scheduler core_multi`.

## Katalog strategii

- Strategia to instancja `StrategyEngine` budowana dynamicznie przez `bot_core.strategies.catalog.StrategyCatalog`.
- Rejestr domyślny (`DEFAULT_STRATEGY_CATALOG`) obejmuje m.in. silniki: `daily_trend_momentum`, `mean_reversion`, `grid_trading`, `volatility_target`, `cross_exchange_arbitrage`.
- Definicje w konfiguracji (`core.yaml`) mogą korzystać z ujednoliconego formatu:

```yaml
strategies:
  core_daily_trend:
    engine: daily_trend_momentum
    risk_profile: balanced
    tags: [core, trend]
    parameters:
      fast_ma: 25
      slow_ma: 100
      breakout_lookback: 55
      momentum_window: 20
      atr_window: 14
      atr_multiplier: 2.0
      min_trend_strength: 0.005
      min_momentum: 0.001
```

- Sekcje specjalizowane (`mean_reversion_strategies`, `volatility_target_strategies`, `cross_exchange_arbitrage_strategies`, `grid_strategies`) również trafiają do katalogu i mogą być referencjonowane przez harmonogramy.

## Definiowanie harmonogramów

- Harmonogramy (`multi_strategy_schedulers`) wskazują strategie z katalogu oraz parametry czasowe. Nowe pola:
  - `signal_limits`: ograniczenia liczby sygnałów per strategia/profil (opcjonalnie, konfigurowane przez API).
  - `capital_policy`: nazwa (`equal_weight`, `risk_parity`, `volatility_target`, `signal_strength`, `fixed_weight`) lub słownik z polami `name`, `weights`, `label`, `rebalance_seconds`.
  - `allocation_rebalance_seconds`: wymusza minimalny odstęp pomiędzy obliczeniami alokacji (sekundy).
  - `portfolio_governor`: integracja z PortfolioGovernorem (jak dotychczas), rozszerzona o dynamiczne wagi.
- `MultiStrategyScheduler` obsługuje polityki alokacji kapitału (`CapitalAllocationPolicy`). Domyślnie stosowana jest wariacja risk-parity (`RiskParityAllocation`), można ją nadpisać przy konstrukcji schedulera lub w konfiguracji YAML.
- Scheduler loguje każde odświeżenie wag (`Capital allocator ... weights: ...`) oraz decyzje PortfolioGovernora, w tym nowe limity sygnałów.
- API runtime udostępnia `MultiStrategyScheduler.set_capital_policy(...)` oraz `replace_capital_policy(...)`,
  dzięki czemu można dynamicznie podmienić politykę kapitału (np. po zmianie konfiguracji YAML lub interwencji operatora).
  Aktualny stan wag można odczytać metodą `allocation_snapshot()`.
- W przypadku polityki `RiskProfileBudgetAllocation` można odpytać `profile_allocation_snapshot()` i flagę `floor_adjustment_applied`,
  a scheduler publikuje również metrykę `allocator_profile_weight` dla każdego harmonogramu oraz dodatkowy log z udziałami profili.

Obsługiwane polityki kapitału:

- `equal_weight` – równy udział każdej strategii.
- `risk_parity` – odwrotność obserwowanej zmienności (`avg_realized_volatility` / `realized_volatility`).
- `volatility_target` – preferuje strategie trafiające w docelową zmienność (`realized_vs_target_vol_pct`).
- `signal_strength` – zwiększa udział strategii generujących częściej wysokiej jakości sygnały (`signals`, `avg_confidence`).
- `smoothed` / `smoothed_allocation` – wygładza wagi zwracane przez inną politykę (EMA), ograniczając skoki i drobne korekty (`alpha`, `min_delta`).
- `drawdown_guard` / `drawdown_adaptive` – degraduje strategie znajdujące się pod presją obsunięcia (`max_drawdown_pct`, `drawdown_pressure`) i raportuje diagnostykę ostatnich kar wagowych (`allocation_diagnostics`).
- `fixed_weight` – manualne wagi (strategia/harmonogram/profil) zdefiniowane w konfiguracji YAML.
- `risk_profile` – budżetuje kapitał pomiędzy profile ryzyka (np. `balanced`, `aggressive`) i deleguje rozdział wewnątrz profilu do kolejnej polityki (domyślnie `risk_parity`, opcjonalnie np. `signal_strength`).
- `profile_floor` w konfiguracji `risk_profile` wymusza minimalny udział profilu po normalizacji (wartość jest automatycznie przycinana do maksymalnie `1 / liczba_profili`, aby uniknąć nieosiągalnych limitów).

Przykład konfiguracji budżetu profili ryzyka i limitów:

```yaml
multi_strategy_schedulers:
  core_multi:
    telemetry_namespace: runtime.core
    capital_policy:
      name: risk_profile
      profiles:
        balanced: 0.6
        aggressive: 0.4
      within_profile: signal_strength
      profile_floor: 0.1
    allocation_rebalance_seconds: 90
    signal_limits:
      trend_engine:
        balanced: 3
      grid_engine:
        aggressive: 1
    schedules:
      trend_schedule:
        strategy: trend_engine
        cadence_seconds: 300
        max_drift_seconds: 45
        warmup_bars: 100
        risk_profile: balanced
        max_signals: 4
      grid_schedule:
        strategy: grid_engine
        cadence_seconds: 180
        max_drift_seconds: 45
        warmup_bars: 50
        risk_profile: aggressive
        max_signals: 2
```

Konfiguracja adaptacyjna względem drawdownu:

```yaml
multi_strategy_schedulers:
  hedged_multi:
    capital_policy:
      name: drawdown_guard
      warning_pct: 8.0
      panic_pct: 15.0
      pressure_weight: 0.6
      min_weight: 0.05
    allocation_rebalance_seconds: 60
    schedules:
      trend:
        strategy: trend_engine
        cadence_seconds: 300
        max_signals: 4
        risk_profile: balanced
      arb:
        strategy: cross_exchange_arbitrage
        cadence_seconds: 90
        max_signals: 2
        risk_profile: aggressive
```

Przykład konfiguracji wygładzenia wag wokół polityki sygnałowej:

```yaml
    capital_policy:
      name: smoothed
      alpha: 0.35
      min_delta: 0.05
      base:
        name: signal_strength
```


Polityki `fixed_weight` nadal wspierają dotychczasowy format wag:

```yaml
    capital_policy:
      name: fixed_weight
      label: core_manual
      rebalance_seconds: 120
      weights:
        trend_schedule: 0.5
        mean_reversion:
          balanced: 0.3
        grid_engine:
          aggressive: 0.2
    allocation_rebalance_seconds: 90
    signal_limits:
      trend_engine:
        balanced: 3
      grid_engine:
        aggressive: 1
    schedules:
      trend_schedule:
        strategy: trend_engine
        cadence_seconds: 300
        max_drift_seconds: 45
        warmup_bars: 100
        risk_profile: balanced
        max_signals: 4
```

## Dodawanie nowych strategii krok po kroku

1. Dodaj definicję strategii do `config/core.yaml` w sekcji `strategies` (lub odpowiadającej sekcji specjalizowanej).
2. Upewnij się, że licencja/zdolność (`capability`) jest aktywna – scheduler wymusi to przy ładowaniu (`guard.require_strategy`).
3. W `multi_strategy_schedulers` wskaż nazwę strategii oraz parametry harmonogramu.
4. Uruchom bootstrap CLI:

```bash
python scripts/run_multi_strategy_scheduler.py --environment binance_paper --scheduler core_multi --run-once
```

5. W przypadku środowisk demo/paper/live można skorzystać z helperów `build_demo_multi_strategy_runtime`, `build_paper_multi_strategy_runtime`, `build_live_multi_strategy_runtime` w kodzie aplikacji lub testach.

## Bezpieczeństwo
- RBAC: token `CORE_SCHEDULER_TOKEN` wymagany w `config/core.yaml` (sekcja `runtime.multi_strategy_schedulers`).
- Kanały komunikacji: wyłącznie gRPC/HTTP2 lub IPC – brak WebSocketów.
- Decision log podpisywany HMAC; walidacja narzędziem `verify_decision_log.py`.

## Testy
- Jednostkowe: `tests/test_multi_strategy_scheduler.py` (w tym telemetria i decision journal).
- Integracyjne: `PYTHONPATH=. pytest tests/test_runtime_pipeline.py` – weryfikacja buildera.
- Smoke CI: `python scripts/smoke_demo_strategies.py --cycles 3`.

