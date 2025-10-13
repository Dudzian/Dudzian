# Warsztat operatorów – konfiguracja scheduler-a i presetów profili ryzyka

## Agenda (2h)
1. **Przegląd architektury** (15 min)
   - Moduły runtime: scheduler, strategie, silnik ryzyka, adaptery.
   - Ścieżka demo → paper → live i punkty kontrolne RBAC/HMAC.
2. **Konfiguracja `config/core.yaml`** (35 min)
   - Sekcja `runtime.multi_strategy_schedulers`: harmonogramy, `rbac_tokens`, `health_check_interval`.
   - Nowa sekcja `runtime.resource_limits`: limity CPU/RAM/I/O, próg ostrzeżeń.
   - Profile ryzyka (`risk_profiles`) – mapowanie `strategy_allocations`, `instrument_buckets`.
3. **Narzędzia operacyjne** (35 min)
   - `python scripts/audit_security_baseline.py --scheduler-required-scope runtime.schedule.write` – audyt RBAC/mTLS.
   - `python scripts/load_test_scheduler.py --iterations 30 --schedules 3` – szybki benchmark latencji i budżetów.
   - `python scripts/smoke_demo_strategies.py --cycles 3` – test dymny pipeline’u demo.
4. **Ćwiczenie praktyczne** (25 min)
   - Zadanie: dostroić preset `balanced` pod wyższy udział arbitrażu (20%).
   - Krok 1: Edycja `config/core.yaml` (`risk_profiles.balanced.strategy_allocations`).
   - Krok 2: `pytest tests/test_config_loader.py::test_load_core_config_resource_limits` – potwierdzenie poprawności.
   - Krok 3: `python scripts/load_test_scheduler.py --schedules 3 --signals 5 --output logs/workshop/balanced.json` – ocena latencji.
   - Krok 4: Dokumentacja zmian w decision logu (`verify_decision_log.py summary --append`).
5. **Q&A i zadania domowe** (10 min)
   - Przygotować własny preset profilu ryzyka (np. `balanced_high_vol`) i zarejestrować w runbooku.
   - Przećwiczyć rollback wg `docs/runbooks/rollback_multi_strategy.md`.

## Materiały
- `docs/architecture/stage4_spec.md`, `docs/architecture/stage4_test_plan.md`.
- `docs/runbooks/operations/strategy_incident_playbook.md` – procedury awaryjne.
- Logi przykładowe: `logs/load_tests/scheduler_profile.sample.json`, `audit/security/security_baseline.json`.

## Notatki dla prowadzącego
- Sprawdź dostępność kluczy RBAC (`CORE_SCHEDULER_TOKEN`) i datasetów w `data/backtests/normalized/`.
- Przygotuj konto uczestnika z uprawnieniami `runtime.schedule.write` (token `scheduler-local`).
- Monitoruj budżety zasobów podczas ćwiczenia: `python scripts/audit_security_baseline.py --print`.
