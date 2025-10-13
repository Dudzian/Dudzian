# Tracker iteracyjny Etapu 4

Aby uniknąć jednorazowego „skoku” statusu całego etapu, wprowadzamy jawny tracker stanu z rozbiciem na poszczególne zadania z backlogu. Status `[x]` oznacza pełną akceptację po przejściu wszystkich bramek (code review, testy regresyjne, audyt operacyjny), a `[ ]` – zadania nadal wymagające pracy, walidacji lub zatwierdzenia.

## Stan iteracji 4M
- **Zakres ukończony:** 1.1 Specyfikacja Etapu 4, 1.2 Profile ryzyka + koszyki instrumentów, 1.3 Checklisty wejścia/wyjścia iteracji, 1.4 Plan testów regresyjnych, 2.1 Dokumentacja mean reversion, 2.2 Implementacja strategii, 2.3 Konfiguracja i presety, 2.4 Testy regresyjne mean reversion, 3.1 Dokumentacja volatility targeting, 3.2 Implementacja `VolatilityTargetSettings/Strategy`, 3.3 Konfiguracja presetów volatility targeting, 3.4 Testy jednostkowe/regresyjne volatility targeting, 4.1 Dokumentacja strategii arbitrażowej, 4.2 Silnik arbitrażowy, 4.3 Konfiguracja arbitrażu w modelach i `core.yaml`, 4.4 Testy integracyjne wejścia/wyjścia spreadu, 5.1 Projekt harmonogramu multi-strategy, 5.2 Implementacja scheduler-a z telemetryką i journalingiem, 5.3 Integracja scheduler-a z pipeline’em oraz CLI `run_multi_strategy_scheduler.py`, 5.4 Testy funkcjonalne schedulera i pipeline’u, 6.1 Rozszerzony harness silnika ryzyka (scenariusze likwidacji/daily reset), 6.2 Aktualizacja runbooka paper tradingu o audyty HMAC i obsługę multi-strategy, 6.3 Biblioteka danych backtestowych (manifest + próbki), 6.4 Procedury walidacji jakości danych zautomatyzowane w `DataQualityValidator` + CLI, 7.1 Rozszerzenie `telemetry_risk_profiles.py` o metryki strategii/scheduler-a, 7.2 Aktualizacja dashboardu Grafany o widgety scheduler-a i skuteczność sygnałów, 7.3 Prometheus Alertmanager z progami PnL/ryzyko/latencja, 7.4 Integracja decision logu multi-strategy z centralnym audytem, 8.1 Integracja testów strategii/scheduler-a z pipeline’em CI i pomiarem coverage, 8.2 Wspólne fixtures/stuby danych dla testów multi-exchange/volatility, 8.3 Smoke test CLI `smoke_demo_strategies.py`, 8.4 Gating jakości na poziomie coverage modułów strategii, 9.1 Audyt RBAC/mTLS uwzględniający scheduler multi-strategy oraz raport bezpieczeństwa, 9.2 Rozszerzenie schematu decision logu o pola multi-exchange/schedule wraz z testami regresyjnymi, 9.3 Mini-audyt compliance (RBAC, podpisy HMAC, rotacje kluczy) z raportem w `audit/security/`, 10.1 Playbook wsparcia L1/L2 dla incydentów strategii i scheduler-a, 10.2 Warsztat/szkolenie operatorów nt. konfiguracji scheduler-a i presetów profili ryzyka, 11.1 Test obciążeniowy scheduler-a (`load_test_scheduler.py`) z raportem jitteru i metryk zasobów, 11.2 Monitorowanie budżetów zasobów (nowe limity `runtime.resource_limits` + moduł `resource_monitor`), 11.3 Procedury awaryjnego wyłączenia/rollbacku multi-strategy (checklista operatora + runbook).
- **Aktualizacja utrzymaniowa:**
  - Iteracja 4K rozszerzyła obsługę strażników `NONE`/`NULL`/`DEFAULT` na wszystkie argumenty tekstowe i listowe `watch_metrics_stream`, doprecyzowując jednocześnie detekcję materiału TLS tak, aby ustawienie sentinelowe nie wymuszało TLS.
  - Iteracja 4L dodaje możliwość wstrzykiwania własnych nagłówków gRPC do `watch_metrics_stream` (flaga `--header` oraz zmienna środowiskowa `BOT_CORE_WATCH_METRICS_HEADERS`) wraz z walidacją kluczy, sanitacją metadanych w decision logu i testami regresyjnymi.
  - Iteracja 4M rozszerza konfigurację `core.yaml` o blok `grpc_metadata` dla `metrics_service`, co umożliwia automatyczne wstrzykiwanie nagłówków gRPC z konfiguracji, walidowane przez loader i validator, z zachowaniem strażnika środowiskowego `BOT_CORE_WATCH_METRICS_HEADERS=NONE` jako wyłącznika.
  - Iteracja 4N przenosi poufne wartości nagłówków gRPC do warstwy środowiskowej/plików (pola `value_env`/`value_file` w `grpc_metadata`), zapisując źródła w konfiguracji runtime i decision logu `watch_metrics_stream`.
  - Iteracja 4O naprawia normalizację `grpc_metadata` w loaderze konfiguracji, aby wariant słownikowy (np. `authorization: {value_env: ...}`) poprawnie pobierał wartości z ENV/plików i zachowywał informacje o źródle.
  - Iteracja 4P deduplikuje nagłówki gRPC przy scalaniu konfiguracji `core.yaml` z parametrami CLI, zapewniając że wartości operatora nadpisują preset oraz że kolejność metadanych odzwierciedla ostatnie wystąpienia kluczy.
  - Iteracja 4Q konsoliduje źródła nagłówków gRPC (core.yaml, zmienne środowiskowe, CLI) i propaguje je do decision logu oraz metadanych telemetrycznych, aby audytorzy widzieli finalne nadpisania.
- **Czynności w toku:** brak – backlog Etapu 4 zamknięty.
- **Blokery:** brak.

**Postęp Etapu 4:** 40/40 (100 %)
**Pasek:** `[####################]`
**Raport zamknięcia:** patrz dokument [Raport zamknięcia Etapu 4](stage4_final_report.md); aktualizacja zakresu opisane w [Specyfikacji Etapu 4](stage4_spec.md).

```json
{"total_tasks":40,"completed":40,"percent":100}
```

## Backlog zadań (stan iteracji 4F)
1. Wymagania zarządcze
   - [x] 1.1 Opracować szczegółową specyfikację Etapu 4 (zakres, zależności, definicje „done”).
   - [x] 1.2 Zaktualizować mapę profili ryzyka i koszyków instrumentów w `config/core.yaml`.
   - [x] 1.3 Rozszerzyć checklisty wejścia/wyjścia iteracji o kryteria paper smoke i audyty.
   - [x] 1.4 Zdefiniować plan testów regresyjnych obejmujący nowe strategie i scheduler.
2. Strategia Mean Reversion
   - [x] 2.1 Udokumentować sygnały i parametry.
   - [x] 2.2 Zaimplementować `MeanReversionSettings` oraz `MeanReversionStrategy`.
   - [x] 2.3 Dodać konfigurację strategii do modeli i `core.yaml`.
   - [x] 2.4 Przygotować testy jednostkowe/backtest dla strategii.
3. Strategia Volatility Targeting
   - [x] 3.1 Opisać założenia strategii kontroli zmienności.
   - [x] 3.2 Zaimplementować `VolatilityTargetSettings/Strategy`.
   - [x] 3.3 Rozszerzyć konfigurację o parametry strategii oraz presety w `core.yaml`.
   - [x] 3.4 Dodać testy jednostkowe/integracyjne dla strategii.
4. Strategia Cross-Exchange Arbitrage
   - [x] 4.1 Zdefiniować dokumentację strategii arbitrażowej.
   - [x] 4.2 Zaimplementować silnik arbitrażowy.
   - [x] 4.3 Dodać konfigurację strategii do modeli i `core.yaml`.
   - [x] 4.4 Przygotować testy integracyjne z różnicami cen.
5. Scheduler i orchestracja
   - [x] 5.1 Zaprojektować harmonogram wielostrate-giczny.
   - [x] 5.2 Zaimplementować moduł scheduler-a z telemetryką.
   - [x] 5.3 Zintegrować scheduler z pipeline’em i CLI.
   - [x] 5.4 Dodać testy funkcjonalne scheduler-a i pipeline’u.
6. Dane, ryzyko i dokumentacja
   - [x] 6.1 Rozszerzyć harness testowy silnika ryzyka o nowe scenariusze.
   - [x] 6.2 Zaktualizować runbook paper/dokumentację operacyjną.
   - [x] 6.3 Uzupełnić bibliotekę danych backtestowych o znormalizowane zestawy dla nowych strategii.
   - [x] 6.4 Przygotować procedury walidacji jakości danych (spójność, braki, outliery) dla profili ryzyka.
7. Observability i alerty
   - [x] 7.1 Rozszerzyć `telemetry_risk_profiles.py` o metryki specyficzne dla nowych strategii.
   - [x] 7.2 Zaktualizować dashboardy Prometheus/OTEL o widgety scheduler-a, latencję i skuteczność sygnałów.
   - [x] 7.3 Skonfigurować alerty (thresholdy, eskalacje) dla degradacji PnL, odchyleń ryzyka i opóźnień scheduler-a.
   - [x] 7.4 Zintegrować logowanie decyzji strategii z centralnym decision logiem oraz raportami audytowymi.
8. Automatyzacja testów i CI
   - [x] 8.1 Włączyć nowe testy strategii/scheduler-a do pipeline’u CI (pytest + backtest).
   - [x] 8.2 Przygotować fixtures/stuby danych dla testów multi-exchange i kontroli zmienności.
   - [x] 8.3 Zbudować smoke test CLI uruchamiający zestaw strategii w trybie demo.
   - [x] 8.4 Dodać raport pokrycia (coverage) i gating jakości dla modułów strategii.
9. Bezpieczeństwo i compliance
   - [x] 9.1 Zweryfikować polityki RBAC/mTLS dla nowych usług scheduler-a i strategii.
   - [x] 9.2 Zaktualizować schemat decision log JSONL o pola specyficzne dla wielu strategii i giełd.
   - [x] 9.3 Przeprowadzić mini-audyt zgodności (checklista RBAC, podpisy HMAC, rotacje kluczy) przed domknięciem etapu.
10. Operacje i wsparcie
   - [x] 10.1 Opracować playbook wsparcia L1/L2 dla incydentów strategii (alert fatigue, degradacja sygnału, awaria adaptera).
   - [x] 10.2 Przygotować szkolenie/warsztat dla operatorów nt. konfiguracji scheduler-a i nowych presetów profili ryzyka.
11. Wydajność i stabilność
   - [x] 11.1 Przeprowadzić testy obciążeniowe scheduler-a i strategii (latencja, konkurencja, jitter) z raportem wyników.
   - [x] 11.2 Monitorować budżety zasobów (CPU, RAM, I/O) podczas równoległego działania strategii i aktualizować limity operacyjne.
   - [x] 11.3 Przygotować procedury awaryjnego wyłączenia/rollbacku strategii i scheduler-a wraz z checklistą operatora.
