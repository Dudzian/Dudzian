# Runbook: Stage4 Multi-Strategy Rollback

## Cel
Bezpieczne i audytowalne wyłączenie scheduler-a wielostrate-gicznego Stage4 w odpowiedzi na incydent, testy obciążeniowe lub planowaną konserwację, wraz z kontrolą alertów i przywróceniem ruchu.

## Lista kontrolna
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 0. Ogłoś okno serwisowe i zarejestruj zgłoszenie (JIRA/ServiceNow) | Incident Manager | Nr zgłoszenia, kanał ogłoszeń | Potwierdzone przez właściciela produktu |
| 1. Uruchom `scripts/disable_multi_strategy.py --reason "<powód>" --ticket <ID> --requested-by <zespół>` (opcjonalnie dodaj `--component multi_strategy`, aby jawnie wskazać scheduler Stage4) | L2 Ops | `var/runtime/overrides/multi_strategy_disable.json` | Plik istnieje, posiada poprawny JSON i uprawnienia 600 |
| 2. Wycisz alerty Stage4 (`deploy/ci/github_actions_stage4_multi_strategy.yml` → job `validate_alerts`, w trybie `silence`) | Observability | Log workflow, identyfikator silence w Alertmanager | Alerty Stage4 posiadają aktywne wyciszenie z czasem wygaśnięcia |
| 3. Potwierdź zatrzymanie scheduler-a (`scripts/smoke_demo_strategies.py --check-disabled`) | L2 Ops | Log CLI, status `disabled` | Status `disabled` zwrócony w CLI |
| 4. Monitoruj metryki awaryjne (`scripts/watch_metrics_stream.py --headers-report --duration 15m`) | NOC | Raport ze skryptu, zrzuty dashboardu Stage4 | Brak nowych alertów krytycznych w oknie obserwacji |
| 5. Po zakończeniu incydentu usuń plik override (`rm var/runtime/overrides/multi_strategy_disable.json`) i potwierdź restart scheduler-a | L2 Ops | Log usunięcia, wynik `scripts/smoke_demo_strategies.py --check-disabled` | Status `enabled`, brak ostrzeżeń |
| 6. Zaktualizuj decision log (`python scripts/verify_decision_log.py summary --append audit/decision_logs/stage4.jsonl --stage stage4 --status rollback_completed --hash-algorithm sha384 --artefact incident_id=<ID> --artefact-from-file summary_sha384=var/audit/stage4/rollback_summary.json --metadata duration_min=<MIN> --tag rollback`) | Incident Manager | `audit/decision_logs/stage4.jsonl`, `var/audit/stage4/rollback_summary.json` | Wpis podpisany HMAC, zawiera identyfikator zgłoszenia, skrót podsumowania i metadane czasu trwania |

## Artefakty końcowe
- `var/runtime/overrides/multi_strategy_disable.json` (kopie przed i po usunięciu).
- Raport workflow wyciszenia alertów oraz identyfikator silence w Alertmanager.
- Logi ze skryptów `smoke_demo_strategies.py` i `watch_metrics_stream.py` potwierdzające status scheduler-a.
- Zrzuty dashboardu Grafany „Stage4 – Multi-Strategy Operations”.
- Wpis w decision logu opisujący incydent/rollback wraz z podpisem HMAC.
