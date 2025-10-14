# Warsztat Stage4 Operations – audyt i wsparcie L1/L2

## Cel szkolenia
Przekazać operatorom OEM wiedzę niezbędną do prowadzenia cyklicznych
audytów Stage4 (RBAC/mTLS/rotacje) oraz reagowania na incydenty multi-
strategy w oparciu o playbooki L1/L2 i narzędzia automatyzujące.

## Agenda (3h)
1. **Architektura Stage4** (20 min)
   - Przypomnienie modułów (strategie, scheduler, ryzyko, egzekucja,
     telemetria) i ścieżki demo→paper→live.
   - Wymogi OEM: mTLS, RBAC, podpisy HMAC, rotacje 90-dniowe.
2. **Narzędzia audytowe** (40 min)
   - `python scripts/audit_stage4_compliance.py --mtls-bundle-name core-oem`
     – interpretacja raportu (`status`, `issues`, `warnings`).
   - `python scripts/audit_security_baseline.py --scheduler-required-scope runtime.schedule.write` – integracja z audytem RBAC.
   - `python scripts/check_key_rotation.py --environment live_binance` –
     log rotacji kluczy API.
3. **Monitoring i obserwowalność** (35 min)
   - Dashboard Grafany „Stage4 – Multi-Strategy Operations” – analiza
     paneli i progów alertów.
   - Raport `watch_metrics_stream.py --headers-report` – weryfikacja
     nagłówków gRPC i źródeł tokenów.
4. **Procedury operacyjne L1/L2** (35 min)
   - Omówienie `docs/runbooks/STAGE4_SUPPORT_PLAYBOOK.md` oraz
     powiązanych runbooków (`STAGE4_ROLLBACK_PLAYBOOK`, `strategy_incident_playbook`).
   - Rola decision logu i podpisów HMAC.
5. **Ćwiczenie praktyczne** (35 min)
   - Symulacja alertu latencji: uczestnicy uruchamiają `audit_stage4_compliance`
     (wynik `warn` dla rotacji), load test scheduler-a oraz tworzą wpis
     incidentowy.
   - Zastosowanie rollbacku (`disable_multi_strategy.py`) i przywrócenie
     scheduler-a.
6. **Q&A + zadania po szkoleniu** (15 min)
   - Przygotowanie harmonogramu dry-runów OEM (`run_oem_acceptance.py`).
   - Opracowanie własnej checklisty eskalacji dla zespołu (format JSONL).

## Materiały i przygotowanie
- Dokumentacja: `docs/architecture/stage4_spec.md`, `docs/runbooks/STAGE4_SUPPORT_PLAYBOOK.md`,
  `docs/runbooks/STAGE4_ROLLBACK_PLAYBOOK.md`.
- Narzędzia CLI: `audit_stage4_compliance.py`, `audit_security_baseline.py`,
  `load_test_scheduler.py`, `watch_metrics_stream.py`.
- Artefakty demonstracyjne:
  - Raport `var/audit/acceptance/<TS>/stage4_support/compliance.json`.
  - Logi `logs/load_tests/sample_stage4.json`, `logs/metrics/sample_headers.json`.
- Wymagania wstępne: dostęp do repozytorium OEM, skonfigurowane
  środowisko Python 3.11+, uprawnienia do czytania sekretów Stage4.

## Notatki dla prowadzącego
- Przed warsztatem uzupełnij rejestr rotacji (`var/security/tls_rotation.json`)
  o aktualne wpisy bundla mTLS (`core-oem`).
- Zweryfikuj, że zmienne `METRICS_SERVICE_AUTH_TOKEN`, `CORE_SCHEDULER_TOKEN`
  oraz `LIVE_DECISION_LOG_HMAC` są obecne w środowisku szkoleniowym.
- Przygotuj przykładowy incident log (`audit/decision_logs/runtime.jsonl`)
  i wpisz numer zgłoszenia używany podczas ćwiczenia praktycznego.
