# Harmonogram wielostrate-giczny

## Przegląd
- Komponent runtime (`bot_core.runtime.multi_strategy_scheduler.MultiStrategyScheduler`) zarządza wieloma strategiami bazującymi na kontrakcie `StrategyEngine`.
- Każda strategia rejestrowana jest z parametrami `cadence_seconds`, `max_drift_seconds`, `warmup_bars`, `risk_profile`, `max_signals` oraz `interval` (dla feedu danych).
- Telemetria (`TelemetryEmitter`) raportuje liczbę sygnałów, opóźnienie (`latency_ms`), średnią pewność (`avg_confidence`) oraz metryki specyficzne: `avg_abs_zscore`, `avg_realized_volatility`, `allocation_error_pct`, `realized_vs_target_vol_pct`, `secondary_delay_ms`, `spread_capture_bps`, `allocator_raw_weight`, `allocator_smoothed_weight`, `allocator_profile_weight`.
- Decyzje każdej strategii zapisywane są w `TradingDecisionJournal` z polami `schedule`, `strategy`, `confidence`, `latency_ms`, `telemetry_namespace` oraz podpisami HMAC.

## Architektura
1. **Źródło danych (`OHLCVStrategyFeed`)** – pobiera świece z cache (Parquet/SQLite) i mapuje je na `MarketSnapshot`.
2. **Scheduler** – asynchronicznie uruchamia `_execute_schedule`, ograniczając liczbę sygnałów (`max_signals`) i dbając o dryf czasowy.
3. **Sink (`InMemoryStrategySignalSink`)** – buforuje sygnały do audytu/regresji; w środowisku produkcyjnym można podmienić na sink wykonawczy.
4. **Builder (`build_multi_strategy_runtime`)** – scala bootstrap, scheduler, feed i strategie na podstawie `CoreConfig`. Dostępne są gotowe warianty `build_demo_multi_strategy_runtime`, `build_paper_multi_strategy_runtime` oraz `build_live_multi_strategy_runtime`, które automatycznie wybierają odpowiednie środowisko na podstawie konfiguracji (preferencje: aliasy nazwy, typ środowiska i tryb offline).
5. **CLI (`scripts/run_multi_strategy_scheduler.py`, `scripts/smoke_demo_strategies.py`)** – pierwsze narzędzie uruchamia scheduler w trybie `--run-once` (audit) lub `--run-forever` (paper/live), drugie rejestruje cykle demo na znormalizowanych datasetach backtestowych. Przykład uruchomienia paper: `python scripts/run_multi_strategy_scheduler.py --environment binance_paper --scheduler core_multi`.
   - Flagi inspekcji kapitału: `--show-capital-state` wypisuje ostatnie udziały (surowe, wygładzone, profilowe, tagowe), a `--show-capital-diagnostics` prezentuje diagnostykę polityki (np. wkłady alokatorów, flagi drawdownu).
   - Flagi zarządzania kapitałem: `--rebalance-capital` wymusza natychmiastowe przeliczenie wag (z pominięciem cool-downu) i kończy działanie, chyba że użyto `--run-after-management`.
   - Flagi eksportu: `--export-capital-state PATH` oraz `--export-capital-diagnostics PATH` zapisują migawkę/diagnostykę do plików JSON (tworząc katalogi pośrednie), co ułatwia automatyczne audyty i integrację z zewnętrznymi narzędziami.
   - Flagi zmiany polityki: `--set-capital-policy PATH` ładuje nową politykę z pliku JSON/YAML (opcjonalnie `--skip-policy-rebalance` aby odroczyć przeliczenie i `--apply-policy-interval` aby przyjąć rekomendowany interwał), natomiast `--set-allocation-interval SECONDS` nadpisuje interwał przeliczeń niezależnie od polityki.
   - Flagi limitów sygnałów: `--set-signal-limit STRATEGIA:PROFIL=LIMIT` aktualizuje nadpisanie limitu, `--clear-signal-limit STRATEGIA:PROFIL` usuwa override, `--list-signal-limits` drukuje aktywne limity, a `--export-signal-limits PATH` zapisuje je do JSON. Opcjonalne flagi `--signal-limit-reason`, `--signal-limit-until` i `--signal-limit-duration` pozwalają zapisać powód oraz czas obowiązywania limitu – przy listowaniu pojawiają się w nawiasie (`powód=…`, `do=…`, `pozostało=…`).

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
  - `signal_limits`: ograniczenia liczby sygnałów per strategia/profil (opcjonalnie, konfigurowane przez API) – wpis może być prostą liczbą lub słownikiem z polami `limit`, `reason`, `until`, `duration_seconds`.
  - `signal_limits`: ograniczenia liczby sygnałów per strategia/profil (opcjonalnie, konfigurowane przez API).
  - `capital_policy`: nazwa (`equal_weight`, `risk_parity`, `volatility_target`, `signal_strength`, `fixed_weight`) lub słownik z polami `name`, `weights`, `label`, `rebalance_seconds`.
  - `allocation_rebalance_seconds`: wymusza minimalny odstęp pomiędzy obliczeniami alokacji (sekundy).
  - `portfolio_governor`: integracja z PortfolioGovernorem (jak dotychczas), rozszerzona o dynamiczne wagi.
- `MultiStrategyScheduler` obsługuje polityki alokacji kapitału (`CapitalAllocationPolicy`). Domyślnie stosowana jest wariacja risk-parity (`RiskParityAllocation`), można ją nadpisać przy konstrukcji schedulera lub w konfiguracji YAML. Implementacje dostępne są w module `bot_core.runtime.capital_policies` i mogą być niezależnie wykorzystywane w innych komponentach.
- Scheduler loguje każde odświeżenie wag (`Capital allocator ... weights: ...`) oraz decyzje PortfolioGovernora, w tym nowe limity sygnałów.
- API runtime udostępnia `MultiStrategyScheduler.set_capital_policy(...)`, `replace_capital_policy(...)` oraz `rebalance_capital(...)`,
  dzięki czemu można dynamicznie podmienić politykę kapitału, wymusić przeliczenie wag (z pominięciem cool-downu) oraz
  wyzwolić natychmiastowe logowanie udziałów.
  Aktualny stan wag można odczytać metodą `allocation_snapshot()` lub pełną migawkę (surowe/wygładzone/profilowe) przez
  `capital_allocation_state()`.
- `capital_policy_diagnostics()` udostępnia ostatnie metadane polityki kapitału (np. kary `DrawdownAdaptiveAllocation`, flagę
  `profile_floor_adjustment` oraz surowe/wygładzone wagi z wrappera `SmoothedCapitalAllocationPolicy`).
- `configure_signal_limit(...)` pozwala dynamicznie ustawić limit sygnałów dla pary strategia/profil (z opcjonalnym powodem, czasem trwania lub konkretną datą wygaśnięcia), a `signal_limit_snapshot()` zwraca aktualną migawkę override’ów wraz z metadanymi (`limit`, `reason`, `expires_at`, `remaining_seconds`, `active`) wykorzystywaną przez CLI oraz eksport JSON. Wygasające override’y są automatycznie logowane i natychmiast odświeżają aktywny limit harmonogramu, dzięki czemu liczba sygnałów wraca do wartości bazowej bez czekania na kolejne uruchomienie pętli.
- `describe_schedules()` zwraca aktualną konfigurację i stan wszystkich harmonogramów (strategie, profile ryzyka, limity sygnałów,
  wagi alokatora, aktywne zawieszenia i ostatnie uruchomienia) – wykorzystywane przez CLI do inspekcji konfiguracji.
- `set_allocation_rebalance_seconds(...)` pozwala dynamicznie nadpisać interwał przeliczeń alokacji – zarówno z kodu, jak i z CLI (`--apply-policy-interval`, `--set-allocation-interval`).
- W przypadku polityki `RiskProfileBudgetAllocation` można odpytać `profile_allocation_snapshot()` i flagę `floor_adjustment_applied`,
  a scheduler publikuje również metrykę `allocator_profile_weight` dla każdego harmonogramu oraz dodatkowy log z udziałami profili.
- Scheduler pozwala na dynamiczne wstrzymanie harmonogramów i całych grup tagów:
  - `suspend_schedule(name, reason=..., duration_seconds=.../until=...)` natychmiast blokuje wykonywanie pojedynczego harmonogramu,
    raportując metryki `suspended`, `suspension_remaining_seconds`, `suspension_tag_indicator` oraz zerując liczbę sygnałów.
  - `suspend_tag(tag, reason=...)` obejmuje wszystkie strategie oznaczone wskazanym tagiem (w tym tag podstawowy z katalogu),
    dzięki czemu można szybko ograniczyć ekspozycję całej klasy strategii.
  - `resume_schedule(...)` i `resume_tag(...)` przywracają normalne działanie, a metoda `suspension_snapshot()` zwraca aktualne
    powody, terminy wygaśnięcia i źródła zawieszeń (harmonogram vs tag).
  - Scheduler automatycznie czyści i loguje wygasłe zawieszenia harmonogramów oraz tagów, dzięki czemu log audytowy
    natychmiast odnotowuje odblokowanie, a migawka `suspension_snapshot()` zawiera wyłącznie aktywne wpisy gotowe do eksportu.
  - Dziennik i logi informują o aktywacji/wyłączeniu blokad, a PortfolioGovernor nadal otrzymuje telemetrię zerową, więc może
    reagować na zmianę ekspozycji (np. wykonując rebalance).
  - W plikach YAML można zdefiniować początkowe blokady (`initial_suspensions`) dla harmonogramów lub tagów – pipeline zastosuje je
    zaraz po zarejestrowaniu strategii, dzięki czemu środowisko startuje od razu z wymaganymi ograniczeniami.
  - Analogicznie można zadeklarować startowe nadpisania limitów sygnałów (`initial_signal_limits`) dla par strategia/profil ryzyka;
    wartości zostaną zapisane w schedulerze przed pierwszym uruchomieniem i pojawią się w migawce `signal_limit_snapshot()` oraz
    eksporcie CLI.
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
- `metric_weighted` – buduje wagi na podstawie zadanych metryk telemetrycznych (wagi mogą być dodatnie/ujemne, dostępne są clampy,
  wartości domyślne oraz fallback do innej polityki).
- `smoothed` / `smoothed_allocation` – wygładza wagi zwracane przez inną politykę (EMA), ograniczając skoki i drobne korekty (`alpha`, `min_delta`).
- `drawdown_guard` / `drawdown_adaptive` – degraduje strategie znajdujące się pod presją obsunięcia (`max_drawdown_pct`, `drawdown_pressure`) i raportuje diagnostykę ostatnich kar wagowych (`allocation_diagnostics`).
- `fixed_weight` – manualne wagi (strategia/harmonogram/profil) zdefiniowane w konfiguracji YAML.
- `risk_profile` – budżetuje kapitał pomiędzy profile ryzyka (np. `balanced`, `aggressive`) i deleguje rozdział wewnątrz profilu do kolejnej polityki (domyślnie `risk_parity`, opcjonalnie np. `signal_strength`).
- `blended` / `composite` – miesza kilka polityk kapitału z wagami udziałów, raportując ich wkład (`allocation_diagnostics` zawiera `mix_weight` oraz udziały strategii per komponent).
- `tag_quota` / `tag_budget` – przydziela udział kapitału według tagów strategii, wspiera fallbacki dla brakujących tagów oraz raportuje liczbę strategii w grupie (`tag_members`).
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
        balanced:
          limit: 3
          reason: risk_cap
          until: 2024-01-05T12:00:00+00:00
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

Przykładowe początkowe zawieszenia w konfiguracji runtime:

```yaml
multi_strategy_schedulers:
  core_multi:
    initial_suspensions:
      - schedule: grid_schedule
        reason: maintenance
        duration_seconds: 3600
      - tag: experimental
        until: 2030-02-01T00:00:00+00:00
        reason: compliance_hold
```

Startowe nadpisania limitów sygnałów zadeklarowane w YAML (klucze odpowiadają wartościom pola `strategy` z sekcji `schedules`):

```yaml
multi_strategy_schedulers:
  core_multi:
    initial_signal_limits:
      trend_engine:
        balanced:
          limit: 4
          reason: bootstrap_window
          until: 2024-02-01T12:00:00+00:00
      mean_reversion:
        balanced:
          limit: 2
          duration_seconds: 1800
```

Podczas bootstrapu pipeline wywoła `configure_signal_limit(...)` dla każdej pary strategia/profil, więc jeszcze przed startem
`signal_limit_snapshot()` oraz `describe_schedules()` pokażą aktywne override’y razem z powodem, czasem wygaśnięcia lub pozostałym
czasem trwania. Nowe wartości są również dostępne dla flag CLI `--list-schedules`, `--export-schedules` oraz `--export-signal-limits`.

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

Przykład konfiguracji polityki telemetrycznej z fallbackiem:

```yaml
    capital_policy:
      name: metric_weighted
      label: telemetry_mix
      default_score: 0.1
      metrics:
        avg_confidence:
          weight: 1.5
          min: 0.0
          max: 1.0
        last_latency_ms:
          weight: -0.02
          min: 0.0
          default: 150.0
      fallback: signal_strength
```

Przykład konfiguracji mieszanej polityki wagowej:

```yaml
    capital_policy:
      name: blended
      components:
        - policy: signal_strength
          weight: 2.0
          label: signals
        - policy:
            name: drawdown_guard
            warning_pct: 6.0
            panic_pct: 15.0
            pressure_weight: 0.5
          weight: 1.0
          label: drawdown
      normalize_components: true
      fallback: equal_weight
```

Przykład konfiguracji budżetu tagów strategii z fallbackiem dla niezmapowanych silników:

```yaml
    capital_policy:
      name: tag_quota
      label: core_tags
      tags:
        trend: 2.0
        mean: 1.0
      default_weight: 1.0
      within_tag: equal_weight
      fallback: signal_strength
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
        balanced:
          limit: 3
          reason: risk_cap
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

### Zarządzanie zawieszeniami przez CLI

Skrypt `run_multi_strategy_scheduler.py` pozwala na szybkie zawieszanie/wznawianie harmonogramów oraz tagów przed startem pętli runtime. Przykładowe użycie:

```bash
# zawieszenie harmonogramu i tagu na 30 minut z powodem "maintenance"
python scripts/run_multi_strategy_scheduler.py \
  --environment binance_paper \
  --suspend-schedule trend_alpha \
  --suspend-tag futures \
  --suspension-reason maintenance \
  --suspension-duration 30m \
  --list-suspensions

# wznowienie wcześniej zablokowanych elementów i uruchomienie schedulera
python scripts/run_multi_strategy_scheduler.py \
  --environment binance_paper \
  --resume-schedule trend_alpha \
  --resume-tag futures \
  --run-after-management --run-once
```

Flagi `--list-suspensions`, `--suspend-schedule`, `--suspend-tag`, `--resume-schedule`, `--resume-tag` można podawać wielokrotnie. Parametry `--suspension-reason`, `--suspension-until` (ISO 8601) oraz `--suspension-duration` (np. `15m`, `2h`, `3600`) stosowane są do wszystkich operacji zawieszenia wykonanych w pojedynczym wywołaniu.
`--export-suspensions PATH` zapisuje aktualną migawkę zawieszeń (harmonogramy i tagi wraz z powodami oraz czasem wygaśnięcia) do pliku JSON, co ułatwia audyt stanu przed uruchomieniem schedulera.

### Inspekcja harmonogramów i alokacji

- `--list-schedules` wypisuje aktualną konfigurację zarejestrowanych harmonogramów (strategie, profile ryzyka, limity sygnałów, tagi, ostatnie uruchomienie oraz ewentualne aktywne zawieszenia).
- `--export-schedules PATH` zapisuje powyższy opis do pliku JSON (przydatne w audycie konfiguracji lub w CI).
- `--show-capital-state` oraz `--show-capital-diagnostics` prezentują bieżące wagi i diagnostykę polityki kapitału bez uruchamiania pętli runtime.
- `--export-capital-state PATH` oraz `--export-capital-diagnostics PATH` zapisują migawki alokacji/diagnostyki do pliku JSON.

## Bezpieczeństwo
- RBAC: token `CORE_SCHEDULER_TOKEN` wymagany w `config/core.yaml` (sekcja `runtime.multi_strategy_schedulers`).
- Kanały komunikacji: wyłącznie gRPC/HTTP2 lub IPC – brak WebSocketów.
- Decision log podpisywany HMAC; walidacja narzędziem `verify_decision_log.py`.

## Testy
- Jednostkowe: `tests/test_multi_strategy_scheduler.py` (w tym telemetria i decision journal).
- Integracyjne: `PYTHONPATH=. pytest tests/test_runtime_pipeline.py` – weryfikacja buildera.
- Smoke CI: `python scripts/smoke_demo_strategies.py --cycles 3`.

