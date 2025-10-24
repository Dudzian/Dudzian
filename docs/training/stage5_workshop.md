# Warsztat Stage5 – Compliance & Cost Control

## Cel szkolenia
Przekazać zespołom OEM wiedzę i procedury konieczne do utrzymania modułów TCO, DecisionOrchestrator oraz Observability+ w Etapie 5 – od analizy kosztów, przez audyt compliance, po działania naprawcze i rotację kluczy.

## Agenda (3,5 h)
1. **Przegląd architektury Stage5** (30 min)
   - Strumień TCO (`bot_core/tco`, raporty CSV/PDF) i integracja z `run_oem_acceptance.py`.
   - DecisionOrchestrator w runtime – przepływy decyzji, integracja z ThresholdRiskEngine, pola decision logu (`tco_kpi`, `decision_path`).
   - Observability+: dashboard Stage5, alerty SLO, paczka obserwowalności.
2. **Analiza danych i KPI** (35 min)
   - Interpretacja metryk `avg_cost_per_trade`, `slippage_bps`, `decision_latency_ms`, `compliance_audit_score`.
   - Praca na prototypowym raporcie `run_tco_analysis.py` i wynikach stress testów (`run_risk_simulation_lab.py --include-tco`).
3. **Narzędzia compliance i rotacji** (40 min)
   - `audit_stage4_compliance.py --profile stage5` – rozszerzenia raportu, interpretacja `issues`/`warnings`.
   - `rotate_keys.py` – planowanie rotacji, tryb `--dry-run` vs `--execute`, wpisy decision logu.
   - `export_observability_bundle.py` i `verify_signature.py` – dystrybucja dashboardów/alertów.
4. **Procedury operacyjne L1/L2** (40 min)
   - Omówienie `docs/runbooks/STAGE5_SUPPORT_PLAYBOOK.md` i checklisty `STAGE5_COMPLIANCE_CHECKLIST.md`.
   - Eskalacja incydentów, integracja z runbookiem `strategy_incident_playbook.md`.
5. **Ćwiczenie praktyczne** (45 min)
   - Symulacja wzrostu kosztów: uczestnicy generują raport TCO, uruchamiają smoke test decision engine, aktualizują decision log i plan rotacji.
   - Weryfikacja dashboardu Stage5 i alertów, przygotowanie komunikatu dla OEM.
6. **Rotacja kluczy i plan ciągłości działania** (20 min)
   - Harmonogram przypomnień, archiwizacja kluczy offline, checklisty audytu.
7. **Q&A + zadania po warsztacie** (20 min)
   - Przygotowanie harmonogramu rotacji i audytów TCO.
   - Zgłoszenie wyników w decision logu (kategoria `stage5_training`).

## Materiały i przygotowanie
- Dokumenty: `docs/architecture/stage5_spec.md`, `docs/architecture/stage5_discovery.md`, `docs/runbooks/STAGE5_SUPPORT_PLAYBOOK.md`, `docs/runbooks/STAGE5_COMPLIANCE_CHECKLIST.md`.
- Narzędzia CLI: `python scripts/run_tco_analysis.py`, `python scripts/run_decision_engine_smoke.py`, `python scripts/audit_stage4_compliance.py --profile stage5`, `python scripts/rotate_keys.py`, `python scripts/export_observability_bundle.py`, `python scripts/verify_signature.py`.
- Artefakty demonstracyjne:
  - Raport TCO (`var/audit/tco/sample_stage5.csv`), `tco_report.pdf`, sygnatury.
  - Logi decision engine (`logs/decision_engine/sample_incident.json`), `smoke.sig`.
  - Pakiet obserwowalności (`var/audit/observability/stage5_bundle.zip`).
- Wymagania wstępne: dostęp do repo OEM, konfiguracja Python 3.11+, uprawnienia do odczytu kluczy Stage5 i decision logu.

## Notatki dla prowadzącego
- Zweryfikuj aktualność kluczy HMAC i mTLS (`python scripts/rotate_keys.py --status --bundle core-oem`, opcjonalnie skrót `--status core-oem` lub `status core-oem`) oraz przygotuj próbkę rotacji z wpisem w decision logu. Raport `--status` pokazuje sekcję `summary` oraz listę `entries`, dzięki czemu od razu widać wpisy `due/overdue`.
- Przygotuj konto testowe z wypełnionym decision logiem (kategorie `stage5_incident`, `stage5_training`).
- Zapewnij dostęp do środowiska offline z najnowszym bundlem Stage4/Stage5 (raport `run_oem_acceptance.py`).
- Po warsztacie zaktualizuj `docs/training/stage5_workshop.md` o listę uczestników i wnioski.

## Rejestracja warsztatu
- Po zakończeniu sesji uruchom `python scripts/log_stage5_training.py` z listą uczestników, materiałów i artefaktów (np. nagranie,
  slajdy). Przykład:
  ```bash
  python scripts/log_stage5_training.py \
      --facilitator "Anna Kowalska" \
      --location "Sala 3A" \
      --training-date 2024-05-18 \
      --participant "Jan Nowak" --participant "Ewa Wiśniewska" \
      --material "Prezentacja PDF" --artifact var/audit/training/slides.pdf \
      --log-hmac-key-file secrets/stage5_training.key \
      --decision-log-path audit/decision_logs/runtime.jsonl \
      --decision-log-hmac-key-file secrets/decision_log_stage5.key
  ```
- Wygenerowany wpis trafia do `var/audit/training/stage5_training_log.jsonl` (podpis HMAC) oraz do decision logu z kategorią
  `stage5_training`. Artefakty kopiuj do `var/audit/training/<data>/` zgodnie z checklistą compliance.
