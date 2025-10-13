# Procedura awaryjnego wyłączenia / rollbacku strategii multi-strategy

## Przesłanki wykonania rollbacku
- Nieprawidłowe sygnały po stronie strategii (np. rozbieżność PnL > 3x dzienny budżet).
- Awarie adapterów giełdowych powodujące brak filli lub gwałtowny wzrost opóźnień.
- Naruszenie budżetów zasobów (`runtime.resource_limits`) utrzymujące się > 10 minut.

## Checklista krok po kroku
1. **Potwierdź decyzję**
   - L2 / lider techniczny zatwierdza rollback w decision logu (`verify_decision_log.py summary --append`).
   - Zapisz aktualny raport `logs/load_tests/scheduler_profile.json` (jeśli dostępny).
2. **Zabezpiecz kontekst**
   - `python scripts/audit_security_baseline.py --print --scheduler-required-scope runtime.schedule.write` – upewnij się, że RBAC/mTLS pozostają poprawne.
   - `python scripts/load_test_scheduler.py --iterations 10 --schedules 3 --output logs/load_tests/pre_rollback.json` – metryki przed rollbackiem.
3. **Wyłącz strategię**
   - `python scripts/run_multi_strategy_scheduler.py --disable-strategy <strategy_name>` – wyłącza zadany schedule.
   - Zweryfikuj w logu decyzji, że `status=disabled` oraz `risk_budget_bucket` zmienił się na `manual`.
4. **Przywróć parametry profilu ryzyka**
   - Zmniejsz alokację w `config/core.yaml` (np. `risk_profiles.balanced.strategy_allocations.cross_exchange: 0.0`).
   - Commit + wpis do decision logu (sekcja `config_change`).
5. **Smoke test po rollbacku**
   - `python scripts/smoke_demo_strategies.py --cycles 1 --strategy daily_trend` – potwierdź, że pozostałe strategie działają.
   - `python scripts/load_test_scheduler.py --iterations 10 --schedules 2 --output logs/load_tests/post_rollback.json` – porównaj metryki.
6. **Aktualizacja runbooków**
   - Dodaj notatkę w `docs/runbooks/operations/strategy_incident_playbook.md` (sekcja Post-Mortem).
   - Wprowadź poprawki w `config/core.yaml` oraz `docs/training/scheduler_workshop.md` (jeśli zmieniły się rekomendacje).

## Powrót do konfiguracji bazowej
1. Po przywróceniu stabilności ponownie uruchom `python scripts/load_test_scheduler.py` i potwierdź `resource_status == "ok"`.
2. Przywróć poprzednie alokacje strategii z repozytorium Git (`git show <commit> config/core.yaml`).
3. Uruchom pełny smoke pipeline paper: `python scripts/run_paper_smoke_ci.py --environment binance_paper`.
4. Zaktualizuj decision log (`status=restored`) oraz zamknij incydent w `logs/incidents/`.

## Referencje
- `config/core.yaml` – limity zasobów i definicje scheduler-a.
- `audit/security/security_baseline.json` – audyt RBAC/mTLS po rollbacku.
- `docs/runbooks/operations/strategy_incident_playbook.md` – ścieżka eskalacji i analiza po incydencie.
