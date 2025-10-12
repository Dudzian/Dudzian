# Tracker iteracyjny Etapu 4

Aby uniknąć jednorazowego „skoku” statusu całego etapu, wprowadzamy jawny tracker stanu z rozbiciem na poszczególne zadania z backlogu. Status `[x]` oznacza pełną akceptację po przejściu wszystkich bramek (code review, testy regresyjne, audyt operacyjny), a `[ ]` – zadania nadal wymagające pracy, walidacji lub zatwierdzenia.

## Stan iteracji 4E
- **Zakres ukończony:** 1.1 Specyfikacja Etapu 4, 1.2 Profile ryzyka + koszyki instrumentów, 1.3 Checklisty wejścia/wyjścia iteracji, 1.4 Plan testów regresyjnych, 2.1 Dokumentacja mean reversion, 2.2 Implementacja strategii, 2.3 Konfiguracja i presety, 2.4 Testy regresyjne mean reversion, 3.1 Dokumentacja volatility targeting, 3.2 Implementacja `VolatilityTargetSettings/Strategy`, 3.3 Konfiguracja presetów volatility targeting, 3.4 Testy jednostkowe/regresyjne volatility targeting, 4.1 Dokumentacja strategii arbitrażowej, 4.2 Silnik arbitrażowy, 4.3 Konfiguracja arbitrażu w modelach i `core.yaml`, 4.4 Testy integracyjne wejścia/wyjścia spreadu, 5.1 Projekt harmonogramu multi-strategy, 5.2 Implementacja scheduler-a z telemetryką i journalingiem, 5.3 Integracja scheduler-a z pipeline’em oraz CLI `run_multi_strategy_scheduler.py`, 5.4 Testy funkcjonalne schedulera i pipeline’u, 6.1 Rozszerzony harness silnika ryzyka (scenariusze likwidacji/daily reset), 6.2 Aktualizacja runbooka paper tradingu o audyty HMAC i obsługę multi-strategy.
- **Czynności w toku:** brak – etap zamknięty po zatwierdzeniu audytów papier tradingu i rozszerzeń harnessu.
- **Blokery:** brak (klucze HMAC zarejestrowane w keychainie operatorskim, procedura opisane w runbooku).

**Postęp Etapu 4:** 22/22 (100 %)
**Pasek:** `[####################]`
**Raport zamknięcia:** patrz dokument [Raport zamknięcia Etapu 4](stage4_final_report.md) z podsumowaniem testów, compliance i rekomendacji.

```json
{"total_tasks":22,"completed":22,"percent":100}
```

## Backlog zadań (stan iteracji 4E)
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
