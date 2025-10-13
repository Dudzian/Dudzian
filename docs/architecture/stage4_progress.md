# Tracker iteracyjny Etapu 4

Aby uniknąć jednorazowego „skoku” statusu całego etapu, wprowadzamy jawny tracker stanu z rozbiciem na poszczególne zadania z backlogu. Status `[x]` oznacza pełną akceptację po przejściu wszystkich bramek (code review, testy regresyjne, audyt operacyjny), a `[ ]` – zadania nadal wymagające pracy, walidacji lub zatwierdzenia.

## Stan iteracji 4F
- **Zakres ukończony:** 1.1 Specyfikacja Etapu 4, 1.2 Profile ryzyka + koszyki instrumentów, 1.3 Checklisty wejścia/wyjścia iteracji, 1.4 Plan testów regresyjnych, 2.1 Dokumentacja mean reversion, 2.2 Implementacja strategii, 2.3 Konfiguracja i presety, 2.4 Testy regresyjne mean reversion, 3.1 Dokumentacja volatility targeting, 3.2 Implementacja `VolatilityTargetSettings/Strategy`, 3.3 Konfiguracja presetów volatility targeting, 3.4 Testy jednostkowe/regresyjne volatility targeting, 4.1 Dokumentacja strategii arbitrażowej, 4.2 Silnik arbitrażowy, 4.3 Konfiguracja arbitrażu w modelach i `core.yaml`, 4.4 Testy integracyjne wejścia/wyjścia spreadu, 5.1 Projekt harmonogramu multi-strategy, 5.2 Implementacja scheduler-a z telemetryką i journalingiem, 5.3 Integracja scheduler-a z pipeline’em oraz CLI `run_multi_strategy_scheduler.py`, 5.4 Testy funkcjonalne schedulera i pipeline’u, 6.1 Rozszerzony harness silnika ryzyka (scenariusze likwidacji/daily reset), 6.2 Aktualizacja runbooka paper tradingu o audyty HMAC i obsługę multi-strategy.
- **Czynności w toku:** plan rozszerzenia Etapu 4 o zadania 7.x–11.x (observability, CI, bezpieczeństwo, operacje, wydajność) – trwa inwentaryzacja braków i harmonogramowanie prac.
- **Blokery:** oczekiwane dane referencyjne do walidacji jakości danych backtestowych (koordynacja z zespołem danych), analiza kosztów budżetu zasobów dla równoległych strategii.

**Postęp Etapu 4:** 22/40 (55 %)
**Pasek:** `[###########---------]`
**Raport zamknięcia:** patrz dokument [Raport zamknięcia Etapu 4](stage4_final_report.md); aktualizacja zakresu opisane w [Specyfikacji Etapu 4](stage4_spec.md).

```json
{"total_tasks":40,"completed":22,"percent":55}
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
   - [ ] 6.3 Uzupełnić bibliotekę danych backtestowych o znormalizowane zestawy dla nowych strategii.
   - [ ] 6.4 Przygotować procedury walidacji jakości danych (spójność, braki, outliery) dla profili ryzyka.
7. Observability i alerty
   - [ ] 7.1 Rozszerzyć `telemetry_risk_profiles.py` o metryki specyficzne dla nowych strategii.
   - [ ] 7.2 Zaktualizować dashboardy Prometheus/OTEL o widgety scheduler-a, latencję i skuteczność sygnałów.
   - [ ] 7.3 Skonfigurować alerty (thresholdy, eskalacje) dla degradacji PnL, odchyleń ryzyka i opóźnień scheduler-a.
   - [ ] 7.4 Zintegrować logowanie decyzji strategii z centralnym decision logiem oraz raportami audytowymi.
8. Automatyzacja testów i CI
   - [ ] 8.1 Włączyć nowe testy strategii/scheduler-a do pipeline’u CI (pytest + backtest).
   - [ ] 8.2 Przygotować fixtures/stuby danych dla testów multi-exchange i kontroli zmienności.
   - [ ] 8.3 Zbudować smoke test CLI uruchamiający zestaw strategii w trybie demo.
   - [ ] 8.4 Dodać raport pokrycia (coverage) i gating jakości dla modułów strategii.
9. Bezpieczeństwo i compliance
   - [ ] 9.1 Zweryfikować polityki RBAC/mTLS dla nowych usług scheduler-a i strategii.
   - [ ] 9.2 Zaktualizować schemat decision log JSONL o pola specyficzne dla wielu strategii i giełd.
   - [ ] 9.3 Przeprowadzić mini-audyt zgodności (checklista RBAC, podpisy HMAC, rotacje kluczy) przed domknięciem etapu.
10. Operacje i wsparcie
   - [ ] 10.1 Opracować playbook wsparcia L1/L2 dla incydentów strategii (alert fatigue, degradacja sygnału, awaria adaptera).
   - [ ] 10.2 Przygotować szkolenie/warsztat dla operatorów nt. konfiguracji scheduler-a i nowych presetów profili ryzyka.
11. Wydajność i stabilność
   - [ ] 11.1 Przeprowadzić testy obciążeniowe scheduler-a i strategii (latencja, konkurencja, jitter) z raportem wyników.
   - [ ] 11.2 Monitorować budżety zasobów (CPU, RAM, I/O) podczas równoległego działania strategii i aktualizować limity operacyjne.
   - [ ] 11.3 Przygotować procedury awaryjnego wyłączenia/rollbacku strategii i scheduler-a wraz z checklistą operatora.
