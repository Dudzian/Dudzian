# Playbook wsparcia L1/L2 – incydenty strategii multi-strategy

## Cel
Zapewnić zespołowi operacyjnemu powtarzalną procedurę reagowania na incydenty związane z harmonogramem multi-strategy (`bot_core`). Dokument obejmuje warstwy L1 (monitoring/triage) oraz L2 (diagnoza techniczna i działania naprawcze) z naciskiem na alert fatigue, degradację sygnałów oraz awarie adapterów giełdowych.

## Strefy odpowiedzialności
- **L1 (NOC / monitoring)** – obserwacja dashboardów Prometheus/Grafana, alarmów Alertmanagera i logów decision log (`audit/decision_log/`).
- **L2 (inżynier dyżurny)** – analiza scheduler-a, strategii i warstwy danych, eskalacja do właścicieli modułów, rollback strategii.

## Checklista L1
1. **Potwierdź alert**:
   - Sprawdź kanał alertowy (`ops/oncall_rotation.yaml`) i zidentyfikuj typ alertu: PnL, ryzyko, latencja, brak sygnałów.
   - Zaloguj zdarzenie w decision logu (`verify_decision_log.py summary --incident`).
2. **Wstępna diagnoza**:
 - Dashboard Grafana → panel *Multi-Strategy Scheduler* (`deploy/grafana/provisioning/dashboards/kryptolowca_overview.json`).
  - `python scripts/audit_stage4_compliance.py --mtls-bundle-name core-oem --allow-missing-env` – szybki audyt RBAC/mTLS/rotacji
    (status `fail` = natychmiastowa eskalacja).
  - `python scripts/load_test_scheduler.py --iterations 5 --schedules 3` (opcjonalne) – potwierdź lokalnie, czy scheduler generuje sygnały.
   - `python scripts/audit_security_baseline.py --print --scheduler-required-scope runtime.schedule.write` – upewnij się, że RBAC/mTLS nie zgłasza błędów.
3. **Decyzja**:
   - Jeśli alert jest falsy (np. krótkotrwały spike, brak przekroczeń budżetów) → oznacz w logu jako *acknowledged*.
   - Jeśli problem trwa > 5 minut lub budżety zasobów przekraczają `runtime.resource_limits` → eskaluj do L2.

## Checklista L2
1. **Zbierz artefakty**:
   - Decision log (`audit/decision_log/*.jsonl`) – potwierdź nowe pola (`schedule_run_id`, `primary_exchange`, `risk_budget_bucket`).
   - Raport bezpieczeństwa (`audit/security/security_baseline.json`) – zweryfikuj brak błędów RBAC/mTLS.
   - Load test: `python scripts/load_test_scheduler.py --iterations 60 --schedules 3 --output logs/load_tests/latest.json`.
   - Budżety zasobów: `pytest tests/test_resource_monitor.py` (szybka weryfikacja logiki), `python scripts/audit_security_baseline.py --print` (realne zużycie vs limity).
2. **Diagnoza przyczyny**:
   - **Alert fatigue**: sprawdź histogram `telemetry_risk_profiles.py render --section scheduler_alerts`; dopasuj progi w `deploy/prometheus/rules/multi_strategy_rules.yml`.
   - **Degradacja sygnału**: uruchom `python scripts/smoke_demo_strategies.py --cycles 2 --strategy mean_reversion` i porównaj `avg_abs_zscore`, `spread_capture_bps`.
   - **Awarie adapterów**: `python scripts/run_trading_stub_server.py --print-runtime-plan` – weryfikuj ścieżki danych, `logs/adapters/*.log`.
3. **Działania naprawcze**:
   - Dostosuj limity `runtime.resource_limits` (CPU/RAM/I/O) i przekaż aktualizację do repo (`config/core.yaml`).
   - Jeśli sygnały są niepoprawne, tymczasowo wyłącz strategię: `python scripts/run_multi_strategy_scheduler.py --disable-strategy <name>` i dodaj wpis w decision logu.
   - Przy awarii adaptera – przełącz scheduler na fallback feed (`config/core.yaml` → `instrument_universes`) i poinformuj zespół danych.
4. **Zakończenie**:
   - Uzupełnij checklistę incydentu (`logs/incidents/<date>-multi_strategy.md`).
   - Przekaż wnioski do właścicieli modułów i zaktualizuj `docs/runbooks/operations/strategy_incident_playbook.md` jeżeli procedura uległa zmianie.

## Eskalacja
- **SLA reakcji**: L1 < 5 minut, L2 < 15 minut.
- **Kanały**: Slack `#ops-trading`, telefon dyżurny zgodnie z `ops/oncall_rotation.yaml`.
- **Warunki eskalacji do lidera technicznego**:
  1. Brak sygnałów > 15 minut mimo aktywnych danych.
  2. Budżet CPU > 120% limitu przez > 10 minut.
  3. Rozbieżność PnL > 3x dzienny budżet ryzyka.

## Artefakty referencyjne
- `config/core.yaml` – sekcja `runtime.resource_limits`, `runtime.multi_strategy_schedulers` (RBAC).
- `audit/security/security_baseline.json` – raport audytu bezpieczeństwa.
- `docs/runbooks/rollback_multi_strategy.md` – procedura awaryjnego wyłączenia strategii.
- `docs/training/scheduler_workshop.md` – materiały szkoleniowe dla operatorów.
