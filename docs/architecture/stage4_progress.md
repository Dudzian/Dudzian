# Tracker iteracyjny Etapu 4

> Tracker służy do komunikacji stanu iteracji 4 (rozszerzona biblioteka strategii + scheduler). Wszystkie wartości są zsynchronizowane z backlogiem 40 zadań i aktualizowane po przejściu pełnego łańcucha weryfikacji (code review → testy regresyjne → audyty operacyjne).

## Metryki postępu (iteracja 4AA)

| Metryka | Wartość |
| --- | --- |
| Zadania ukończone | **40 / 40** |
| Procent | **100 %** |
| Pasek | `####################` |
| Stan czynności | brak elementów w toku, brak blokerów |

**Raport zamknięcia:** patrz [Raport zamknięcia Etapu 4](stage4_final_report.md). Rozszerzony zakres opisano w [Specyfikacji Etapu 4](stage4_spec.md).

```json
{"total_tasks":40,"completed":40,"percent":100}
```

## Zakres ukończony

### Biblioteka strategii i ryzyko
- 1.1–1.4: Specyfikacja, profile ryzyka, checklisty gate’ów oraz plan testów regresyjnych.
- 2.1–2.4: Strategia mean reversion (dokumentacja, implementacja, konfiguracja, testy/regresje).
- 3.1–3.4: Strategia volatility targeting z presetami, testami i dokumentacją.
- 4.1–4.4: Strategia cross-exchange arbitrage (silnik, konfiguracja, testy różnic cen).
- 6.1–6.4: Rozszerzone harnessy ryzyka, runbooki paper/live oraz biblioteka znormalizowanych danych + walidacja jakości.

### Scheduler, automatyzacja i operacje
- 5.1–5.4: Harmonogram multi-strategy (projekt, implementacja, integracja z pipeline’em i testy funkcjonalne).
- 7.1–7.4: Telemetria i alerty (metryki scheduler-a, dashboardy Prometheus/Grafana, alerty PnL/ryzyko/opóźnienia, decision log).
- 8.1–8.4: Integracja regresji do CI, wspólne fixtures, smoke test CLI w trybie demo oraz coverage gating.
- 9.1–9.3: RBAC/mTLS dla nowych usług, schema decision logu, mini-audyt compliance z raportem bezpieczeństwa.
- 10.1–10.2: Playbook wsparcia L1/L2 + szkolenie operatorów.
- 11.1–11.3: Testy obciążeniowe scheduler-a, monitoring budżetów zasobów i procedury awaryjne (rollback, freeze).

## Historia iteracji utrzymaniowych

| Iteracja | Zakres |
| --- | --- |
| 4K | Strażnicy `NONE/NULL/DEFAULT` dla wszystkich argumentów `watch_metrics_stream`, poprawiona detekcja TLS. |
| 4L | Flaga `--header` i zmienna `BOT_CORE_WATCH_METRICS_HEADERS` z audytem kluczy w decision logu. |
| 4M | Konfiguracja `metrics_service.grpc_metadata` w `core.yaml` (loader + walidator). |
| 4N | Pochodzenie nagłówków z ENV/plików (`value_env`/`value_file`) w runtime i decision logu. |
| 4O | Poprawki normalizacji wariantu słownikowego `grpc_metadata`. |
| 4P | Deduplikacja nagłówków przy nadpisaniach CLI. |
| 4Q | Raportowanie źródeł nagłówków w decision logu. |
| 4R | Referencje `@env:`/`@file:` w CLI z obsługą `@@`. |
| 4S | Wsparcie separatorów linii/średników w zmiennej `BOT_CORE_WATCH_METRICS_HEADERS`. |
| 4T | Ignorowanie komentarzy `#` w zmiennej nagłówków. |
| 4U | Dekodowanie `@env64`/`@file64` i obsługa nagłówków binarnych `*-bin`. |
| 4V | Pola `value*_base64` w `core.yaml` dla nagłówków base64. |
| 4W | Flaga `--headers-file` i audyt źródeł plikowych. |
| 4X | Zmienna `BOT_CORE_WATCH_METRICS_HEADERS_FILE` z listą plików. |
| 4Y | `grpc_metadata_files` w `core.yaml` + normalizacja ścieżek. |
| 4Z | Obsługa katalogów nagłówków (`grpc_metadata_directories`, `--headers-dir`). |
| 4AA | Raport audytowy `--headers-report` / `--headers-report-only` z maskowaniem wartości. |

## Backlog zadań (zamknięty)

| Kategoria | ID | Opis | Status |
| --- | --- | --- | --- |
| 1. Wymagania zarządcze | 1.1 | Opracować szczegółową specyfikację Etapu 4 (zakres, zależności, definicje „done”). | ✅ |
| 1. Wymagania zarządcze | 1.2 | Zaktualizować mapę profili ryzyka i koszyków instrumentów w `config/core.yaml`. | ✅ |
| 1. Wymagania zarządcze | 1.3 | Rozszerzyć checklisty wejścia/wyjścia iteracji o kryteria paper smoke i audyty. | ✅ |
| 1. Wymagania zarządcze | 1.4 | Zdefiniować plan testów regresyjnych obejmujący nowe strategie i scheduler. | ✅ |
| 2. Mean Reversion | 2.1 | Udokumentować sygnały i parametry. | ✅ |
| 2. Mean Reversion | 2.2 | Zaimplementować `MeanReversionSettings` oraz `MeanReversionStrategy`. | ✅ |
| 2. Mean Reversion | 2.3 | Dodać konfigurację strategii do modeli i `core.yaml`. | ✅ |
| 2. Mean Reversion | 2.4 | Przygotować testy jednostkowe/backtest dla strategii. | ✅ |
| 3. Volatility Targeting | 3.1 | Opisać założenia strategii kontroli zmienności. | ✅ |
| 3. Volatility Targeting | 3.2 | Zaimplementować `VolatilityTargetSettings/Strategy`. | ✅ |
| 3. Volatility Targeting | 3.3 | Rozszerzyć konfigurację o parametry strategii oraz presety w `core.yaml`. | ✅ |
| 3. Volatility Targeting | 3.4 | Dodać testy jednostkowe/integracyjne dla strategii. | ✅ |
| 4. Cross-Exchange Arbitrage | 4.1 | Zdefiniować dokumentację strategii arbitrażowej. | ✅ |
| 4. Cross-Exchange Arbitrage | 4.2 | Zaimplementować silnik arbitrażowy. | ✅ |
| 4. Cross-Exchange Arbitrage | 4.3 | Dodać konfigurację strategii do modeli i `core.yaml`. | ✅ |
| 4. Cross-Exchange Arbitrage | 4.4 | Przygotować testy integracyjne z różnicami cen. | ✅ |
| 5. Scheduler i orchestracja | 5.1 | Zaprojektować harmonogram wielostrate-giczny. | ✅ |
| 5. Scheduler i orchestracja | 5.2 | Zaimplementować moduł scheduler-a z telemetryką. | ✅ |
| 5. Scheduler i orchestracja | 5.3 | Zintegrować scheduler z pipeline’em i CLI. | ✅ |
| 5. Scheduler i orchestracja | 5.4 | Dodać testy funkcjonalne scheduler-a i pipeline’u. | ✅ |
| 6. Dane, ryzyko i dokumentacja | 6.1 | Rozszerzyć harness testowy silnika ryzyka o nowe scenariusze. | ✅ |
| 6. Dane, ryzyko i dokumentacja | 6.2 | Zaktualizować runbook paper/dokumentację operacyjną. | ✅ |
| 6. Dane, ryzyko i dokumentacja | 6.3 | Uzupełnić bibliotekę danych backtestowych o znormalizowane zestawy dla nowych strategii. | ✅ |
| 6. Dane, ryzyko i dokumentacja | 6.4 | Przygotować procedury walidacji jakości danych (spójność, braki, outliery) dla profili ryzyka. | ✅ |
| 7. Observability i alerty | 7.1 | Rozszerzyć `telemetry_risk_profiles.py` o metryki specyficzne dla nowych strategii. | ✅ |
| 7. Observability i alerty | 7.2 | Zaktualizować dashboardy Prometheus/OTEL o widgety scheduler-a, latencję i skuteczność sygnałów. | ✅ |
| 7. Observability i alerty | 7.3 | Skonfigurować alerty (thresholdy, eskalacje) dla degradacji PnL, odchyleń ryzyka i opóźnień scheduler-a. | ✅ |
| 7. Observability i alerty | 7.4 | Zintegrować logowanie decyzji strategii z centralnym decision logiem oraz raportami audytowymi. | ✅ |
| 8. Automatyzacja testów i CI | 8.1 | Włączyć nowe testy strategii/scheduler-a do pipeline’u CI (pytest + backtest). | ✅ |
| 8. Automatyzacja testów i CI | 8.2 | Przygotować fixtures/stuby danych dla testów multi-exchange i kontroli zmienności. | ✅ |
| 8. Automatyzacja testów i CI | 8.3 | Zbudować smoke test CLI uruchamiający zestaw strategii w trybie demo. | ✅ |
| 8. Automatyzacja testów i CI | 8.4 | Dodać raport pokrycia (coverage) i gating jakości dla modułów strategii. | ✅ |
| 9. Bezpieczeństwo i compliance | 9.1 | Zweryfikować polityki RBAC/mTLS dla nowych usług scheduler-a i strategii. | ✅ |
| 9. Bezpieczeństwo i compliance | 9.2 | Zaktualizować schemat decision log JSONL o pola specyficzne dla wielu strategii i giełd. | ✅ |
| 9. Bezpieczeństwo i compliance | 9.3 | Przeprowadzić mini-audyt zgodności (checklista RBAC, podpisy HMAC, rotacje kluczy). | ✅ |
| 10. Operacje i wsparcie | 10.1 | Opracować playbook wsparcia L1/L2 dla incydentów strategii. | ✅ |
| 10. Operacje i wsparcie | 10.2 | Przygotować szkolenie/warsztat dla operatorów nt. konfiguracji scheduler-a i presetów profili ryzyka. | ✅ |
| 11. Wydajność i stabilność | 11.1 | Przeprowadzić testy obciążeniowe scheduler-a i strategii (latencja, konkurencja, jitter) z raportem wyników. | ✅ |
| 11. Wydajność i stabilność | 11.2 | Monitorować budżety zasobów (CPU, RAM, I/O) podczas równoległego działania strategii i aktualizować limity operacyjne. | ✅ |
| 11. Wydajność i stabilność | 11.3 | Przygotować procedury awaryjnego wyłączenia/rollbacku strategii i scheduler-a wraz z checklistą operatora. | ✅ |
