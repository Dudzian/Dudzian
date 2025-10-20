# Playbook wsparcia L1/L2 – Stage5 Compliance & Cost Control

## Cel
Zapewnić zespołom dyżurnym powtarzalny proces obsługi środowiska Stage5 – w szczególności monitorowanie kosztów transakcyjnych, decyzji orchestratora oraz rotacji kluczy – od wstępnej obserwacji (L1) po działania naprawcze i audyt (L2).

## Zakres odpowiedzialności
- **L1 (NOC/monitoring)** – monitorowanie alertów SLO (koszt, fill rate, decision latency), dashboardu Grafany *Stage5 – Compliance & Cost Control* oraz wyników automatycznych dry-runów OEM.
- **L2 (Ops/Engineering)** – analiza DecisionOrchestrator, modułu TCO, integracji z risk engine oraz procesów rotacji kluczy; przygotowanie działań naprawczych i komunikacja z właścicielami strategii.

## Checklista L1
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Potwierdź alert w Alertmanagerze (`stage5_tco_cost_spike`, `stage5_decision_latency`, `stage5_fill_rate_drop`). | L1 | Log Alertmanagera, wpis w kanale on-call | Alert oznaczony jako `acknowledged`; ID incydentu zapisane |
| 2. Sprawdź dashboard *Stage5 – Compliance & Cost Control* (panele `avg_cost_per_trade`, `slippage_bps`, `decision_latency_ms`, `slo_breach_count`). | L1 | Zrzut panelu, komentarz w decision logu | Wartości w zielonym/żółtym zakresie lub eskalacja do L2 |
| 3. Uruchom `python scripts/audit_stage4_compliance.py --profile stage5 --mtls-bundle-name core-oem` w trybie odczytu. | L1 | Raport JSON w `var/audit/acceptance/<TS>/stage5_support/` | Status `pass` lub ostrzeżenia przekazane do L2 |
| 4. Zweryfikuj ostatni wpis `run_oem_acceptance.py` – sekcja TCO i rotacji kluczy (`var/audit/acceptance/<TS>/meta.json`). | L1 | Plik meta, decyzja HMAC | Raport oznaczony jako aktualny (<7 dni) |
| 5. Jeśli alert utrzymuje się >5 min lub audyt zgłasza `fail`, eskaluj do L2. | L1 | Kanał eskalacji wg `ops/oncall_rotation.yaml` | Eskalacja potwierdzona przez L2 |

## Checklista L2
| Krok | Odpowiedzialny | Artefakty | Akceptacja |
| --- | --- | --- | --- |
| 1. Uruchom `python scripts/run_tco_analysis.py --output var/audit/tco/<TS>/incident.csv` i porównaj z progiem w `config/core.yaml`. | L2 | Raport CSV/PDF, podpis HMAC | Koszt poniżej progu lub decyzja o ograniczeniu strategii |
| 2. Wykonaj `python scripts/run_decision_engine_smoke.py --mode incident --include-tco` dla strategii objętej alertem. | L2 | Log CLI, podpis `smoke.sig` | Status `success`; w razie `fail` przygotuj plan rollbacku |
| 3. Zweryfikuj rotację kluczy: `python scripts/rotate_keys.py --status --bundle core-oem`. | L2 | Raport JSON, wpis decision logu | Brak przeterminowanych kluczy; w przeciwnym razie zaplanuj rotację |
| 4. Uaktualnij decision log (`verify_decision_log.py summary --category stage5_incident`) i potwierdź obecność pól TCO. | L2 | Raport w `var/audit/decisions/` | Wpis zawiera `tco_kpi`, `decision_path`, `rotation_event_id` |
| 5. Jeśli konieczne, uruchom `python scripts/disable_multi_strategy.py --component decision_orchestrator --reason <ID>` zgodnie z runbookiem rollbacku. | L2 | `var/runtime/overrides/decision_orchestrator_disable.json`, wpis logu | Orchestrator w oczekiwanym stanie, plan przywrócenia przygotowany |
| 6. Po incydencie zaktualizuj dokumentację (`docs/runbooks/STAGE5_SUPPORT_PLAYBOOK.md`, `docs/runbooks/STAGE5_COMPLIANCE_CHECKLIST.md`) i przekaż lessons learned zespołowi compliance. | L2 | Commit, wpis w change-logu | Review zatwierdzone przez właściciela produktu |
| 7. Jeżeli incydent wymagał warsztatu ad-hoc (lessons learned), zarejestruj go przy pomocy `python scripts/log_stage5_training.py` i załącz artefakty (nagranie, slajdy) do `var/audit/training/<data>/`. | L2 | `var/audit/training/stage5_training_log.jsonl`, wpis decision log `stage5_training` | Wpis podpisany HMAC, artefakty dostępne offline |

## Walidacja modeli Decision Engine
- Uruchom pipeline treningowy: `python -m bot_core.ai.pipeline training.csv models/decision autotrader target --features close volume --register --config config/decision_engine.yaml`. Upewnij się, że artefakt zawiera metadane (`trained_at`, `metrics.mae`, `metadata.feature_scalers`).
- Zarejestruj wygenerowany model w DecisionOrchestratorze (w logach pojawi się wpis „Registered model autotrader”). Sprawdź w `var/runtime/decision_orchestrator.json`, że `model_selection` odnotowuje nową wersję.
- W `config/core.yaml` ustaw `decision_engine.evaluation_history_limit` adekwatnie do potrzeb audytu (domyślnie 512 wpisów). Limit chroni pamięć runtime i determinuje, ile ostatnich ewaluacji będzie widocznych w DecisionAwareSignalSink i UI.
- Wyeksportuj bieżące podsumowanie jakości: `python scripts/export_decision_engine_summary.py --ledger <ledger_dir> --output var/audit/decision_engine/<TS>/summary.json --environment <env> --portfolio <portfolio> --include-history`. Artefakt dołącz do raportu hypercare Stage5 i zweryfikuj, że `total`, `acceptance_rate` oraz `latest_model` spełniają polityki.
- W UI/CLI Stage5 sprawdź sekcję `decision_engine_summary` (całkowita liczba ewaluacji, acceptance rate, ostatni model i snapshot progów). Dane pochodzą z `DecisionAwareSignalSink.evaluation_summary()` i pozwalają szybko ocenić jakość decyzji AI oraz najczęstsze powody odrzuceń.
- Przed startem AutoTradera potwierdź, że GUI widzi `decision_engine.accepted == True` w ostatnim `RiskDecision`. W trybie live każda decyzja z flagą `ai_degraded` musi zostać ręcznie zatwierdzona (menedżer modeli powinien zgłosić ostrzeżenie „AI backend degraded”).

## Procedura eskalacji
1. **Warunki eskalacji do Incident Managera:** `avg_cost_per_trade` > 2× progu, `decision_latency_ms` > 500 ms p95 przez >10 min, brak raportu rotacji kluczy > 30 dni, status `fail` w `audit_stage4_compliance --profile stage5`.
2. **Kanały:** Slack `#ops-stage5`, telefon dyżurny (`ops/oncall_rotation.yaml`), rezerwowy most konferencyjny.
3. **Komunikacja z OEM:** przygotuj komunikat według wzoru w `docs/runbooks/operations/strategy_incident_playbook.md`, uwzględniając wpływ kosztów i decyzji.

## Artefakty / Akceptacja
- Raporty TCO, decision engine smoke, rotacji kluczy i audytu compliance zapisane w `var/audit/acceptance/<timestamp>/stage5_support/`.
- Wpis decision logu `audit/decision_logs/runtime.jsonl` z kategorią `stage5_incident`, podpisany HMAC i zweryfikowany `verify_decision_log.py`.
- Aktualny eksport dashboardu i alertów (JSON + sygnatura) dołączony do incydentu.
- Lista obecności warsztatu Stage5 oraz aktualizacja dokumentacji po incydencie.
