# Stage 5 – Compliance & Observability+ Checklist

Każdy dry-run oraz release Etapu 5 musi spełnić poniższe warunki przed promocją środowiska na kolejny poziom.

## Sekcja: TCO & Raporty
- [ ] Raport TCO (CSV + PDF) wygenerowany przez `python scripts/run_tco_analysis.py` i podpisany `scripts/verify_signature.py`.
  - Artefakty/Akceptacja: `var/audit/tco/<data>/tco_report.csv`, `tco_report.pdf`, `tco_report.sig`.
- [ ] Zestawienie kosztów na profil ryzyka z aktualnym progiem w `config/core.yaml` (konserwatywny/zbalansowany/agresywny/manualny).
  - Artefakty/Akceptacja: `var/audit/tco/<data>/profile_costs.json` + wpis decision log.
- [ ] Raport Paper Labs zawiera sekcję TCO i jest archiwizowany wraz z poprzednimi wynikami.
  - Artefakty/Akceptacja: `var/audit/acceptance/<data>/paper_labs_tco.json`.

## Sekcja: DecisionOrchestrator
- [ ] Wynik `python scripts/run_decision_engine_smoke.py --mode paper` zakończył się sukcesem.
  - Artefakty/Akceptacja: `var/audit/decision_engine/smoke_<data>.json` + podpis `smoke_<data>.json.sig`.
- [ ] Decision log zawiera pola `tco_kpi`, `decision_path`, `rotation_event_id`; `scripts/verify_decision_log.py` raportuje status PASS.
  - Artefakty/Akceptacja: `var/audit/decisions/verification_<data>.json`.
- [ ] Stress test latency (`run_risk_simulation_lab.py --scenario latency_spike`) z flagą `--include-tco` przeszedł pomyślnie.
  - Artefakty/Akceptacja: `var/audit/tco/<data>/latency_spike_summary.json`.

## Sekcja: Observability & SLO
- [ ] Dashboard Grafany „Stage5 – Compliance & Cost Control” zaktualizowany (eksport JSON, sygnatura HMAC).
  - Artefakty/Akceptacja: `deploy/grafana/provisioning/dashboards/stage5_compliance_cost.json`, `stage5_compliance_cost.json.sig`.
- [ ] Reguły alertów Prometheus `deploy/prometheus/stage5_alerts.yaml` zweryfikowane przez `validate_prometheus_rules.py`.
  - Artefakty/Akceptacja: `var/audit/alerts/stage5_rules_report.json`.
- [ ] SLO monitor (`scripts/slo_monitor.py`) wygenerował raport z trendem i metadanymi audytu.
  - Artefakty/Akceptacja: `var/audit/slo/<data>/slo_report.json`.

## Sekcja: Rotacja kluczy i bezpieczeństwo
- [ ] `scripts/rotate_keys.py --dry-run` wykonane i zatwierdzone w decision logu; w przypadku rotacji produkcyjnej `--execute` z kopią zapasową offline.
  - Artefakty/Akceptacja: `var/audit/keys/rotation_plan_<data>.json`, wpis decision log HMAC.
- [ ] Audit Stage5 compliance (`python scripts/audit_stage4_compliance.py --profile stage5`) rozszerzony o moduły TCO i observability (SLO, rotacja kluczy).
  - Artefakty/Akceptacja: `var/audit/compliance/stage5_report.json`.
- [ ] Weryfikacja uprawnień plików kluczy (`chmod 600`) oraz potwierdzenie podpisów bundli (`scripts/verify_signature.py`).
  - Artefakty/Akceptacja: raport `var/audit/security/key_permissions_<data>.json`.

## Sekcja: Enablement operacyjny
- [ ] Przeprowadzony warsztat L1/L2 (Stage5 operations workshop) i podpisana lista obecności.
  - Artefakty/Akceptacja: `var/audit/training/stage5_training_log.jsonl` (wpis wygenerowany `scripts/log_stage5_training.py`, podpis HMAC), decision log `stage5_training`, aktualizacja `docs/training/stage5_workshop.md`.
- [ ] Zaktualizowany playbook incydentów TCO/DecisionOrchestrator (`docs/runbooks/STAGE5_SUPPORT_PLAYBOOK.md`).
  - Artefakty/Akceptacja: commit z numerem iteracji i wpisem w decision log.
- [ ] Checklisty demo→paper→live rozszerzone o sekcję TCO oraz SLO.
  - Artefakty/Akceptacja: `docs/runbooks/DEMO_PAPER_LIVE_CHECKLIST.md` – sekcja Stage5, podpis operatora.
