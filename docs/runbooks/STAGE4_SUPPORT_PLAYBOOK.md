# Playbook wsparcia L1/L2 – Stage4 Operations

## Cel
Zapewnić zespołom dyżurnym powtarzalny proces obsługi środowiska Stage4
od wstępnego monitoringu (L1) po analizę techniczną i działania
naprawcze (L2). Playbook integruje kontrole RBAC/mTLS, audyty rotacji
kluczy, pracę z decision logiem oraz szkolenia operatorskie wymagane w
fazie OEM.

## Zakres odpowiedzialności
- **L1 (NOC/monitoring)** – obserwacja alertów Prometheus/Alertmanager,
  dashboardu Grafany *Stage4 – Multi-Strategy Operations* oraz logów
  decision logu (`audit/decision_logs/*`).
- **L2 (Ops/Engineering)** – analiza scheduler-a, strategii, danych
  wejściowych oraz zabezpieczeń (RBAC/mTLS, rotacje), przygotowanie
  działań naprawczych i komunikacja z właścicielami modułów.

## Checklista L1
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Potwierdź alert i typ (PnL, ryzyko, latencja, brak sygnałów) w Alertmanagerze | L1 | Log Alertmanagera, wpis w kanale on-call | Alert oznaczony jako `acknowledged`, ID incidentu zapisane |
| 2. Uruchom `python scripts/audit_stage4_compliance.py --mtls-bundle-name core-oem --allow-missing-env` i sprawdź ostrzeżenia | L1 | Raport JSON na STDOUT, log w `var/audit/acceptance/<TS>/stage4_support/` | Brak statusu `fail`; ostrzeżenia przekazane do L2 |
| 3. Zweryfikuj dashboard Grafany (panele `avg_abs_zscore`, `allocation_error_pct`, `spread_capture_bps`, `secondary_delay_ms`) | L1 | Zrzut panelu, komentarz w decision logu | Panele w statusie zielonym/żółtym lub eskalacja do L2 |
| 4. Zaloguj incydent w decision logu `audit/decision_logs/runtime.jsonl` (`verify_decision_log.py summary --incident`) | L1 | Wpis decision logu podpisany HMAC | Wpis posiada ID zgłoszenia i znacznik czasu |
| 5. Eskaluj do L2, jeżeli problem trwa >5 min lub `audit_stage4_compliance` zwraca `fail` | L1 | Kanał Slack/telefon wg `ops/oncall_rotation.yaml` | Eskalacja potwierdzona przez L2 |

## Checklista L2
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Pobierz raport `var/audit/acceptance/<TS>/stage4_support/compliance.json` (jeżeli istnieje) i porównaj z bieżącym wynikiem `audit_stage4_compliance` | L2 | Raport audytu, log CLI | Brak statusu `fail`; wszystkie ostrzeżenia zdiagnozowane |
| 2. Uruchom `python scripts/load_test_scheduler.py --iterations 30 --schedules 3 --output logs/load_tests/stage4_incident.json` | L2 | Raport load testu, wyniki SLA | Latencja i jitter w dopuszczalnych granicach lub decyzja o wyłączeniu scheduler-a |
| 3. Wykonaj `python scripts/watch_metrics_stream.py --headers-report --duration 10m --output logs/metrics/stage4_headers.json` | L2 | Raport nagłówków, metryki scheduler-a | Nagłówki mTLS/RBAC poprawne, brak anomalii w `avg_abs_zscore` |
| 4. Zweryfikuj rotacje TLS (`python scripts/audit_stage4_compliance.py --mtls-bundle-name core-oem --rotation-warn-days 7 --check-paths`) i klucze RBAC (`scripts/audit_security_baseline.py --print`) | L2 | Raporty audytowe, logi CLI | Rotacje w oknie bezpieczeństwa, brak błędów RBAC |
| 5. W razie potrzeby wykonaj rollback: `python scripts/disable_multi_strategy.py --reason <ID> --ticket <INC>` oraz zastosuj runbook `docs/runbooks/STAGE4_ROLLBACK_PLAYBOOK.md` | L2 | `var/runtime/overrides/multi_strategy_disable.json`, wpis decision logu | Scheduler w stanie oczekiwanym (enabled/disabled), rollback potwierdzony |
| 6. Po incydencie zaktualizuj playbook (`docs/runbooks/STAGE4_SUPPORT_PLAYBOOK.md`) i przekaż lessons learned zespołowi strategii | L2 | Zaktualizowana dokumentacja, wpis w repozie change-log | Review dokonane przez właściciela produktu |

## Procedura eskalacji
1. **Warunki eskalacji do Incident Managera**: brak sygnałów > 15 minut,
   przekroczenie budżetu PnL > 3× dzienny limit lub wynik audytu Stage4
   = `fail`.
2. **Kanały**: Slack `#ops-trading`, telefon dyżurny (`ops/oncall_rotation.yaml`),
   awaryjnie wideokonferencja MS Teams/Zoom.
3. **Komunikacja zewnętrzna**: przy incydentach wpływających na klientów
   OEM przygotuj komunikat według wzoru w `docs/runbooks/operations/strategy_incident_playbook.md`.

## Artefakty / Akceptacja
- Raport `audit_stage4_compliance` zapisany w
  `var/audit/acceptance/<timestamp>/stage4_support/compliance.json`.
- Logi z `load_test_scheduler.py`, `watch_metrics_stream.py` oraz
  `audit_security_baseline.py` dołączone do wpisu incidentowego.
- Wpis decision logu (`audit/decision_logs/runtime.jsonl`) z kategorią
  `stage4_incident`, podpisany HMAC i zweryfikowany `verify_decision_log.py`.
- Uaktualniona checklista L1/L2 (ten dokument) z podpisami operatora i
  Incident Managera.
