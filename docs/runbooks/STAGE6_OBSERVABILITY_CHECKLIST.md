# STAGE6 – Checklist: Observability & SLO Monitor

## Cel
Zweryfikować kompletność pakietu obserwowalności, statusy SLO oraz integrację z
PortfolioGovernorem podczas hypercare Stage6.

## Prerekwizyty
- Aktualne definicje SLO w `config/observability/slo.yml`.
- Dostęp do metryk z `var/metrics/` lub źródeł wskazanych w konfiguracji.
- Klucz HMAC do podpisu raportów w `secrets/hmac/observability.key`.

> **Uwaga:** Wszystkie skrypty Stage6 uruchamiamy poprzez `python <ścieżka_do_skryptu>` (alias `python3` w aktywnym venv). Bezpośrednie `./scripts/...` nie są wspierane, aby zachować spójność zależności i konfiguracji.

> **Skrót automatyczny:** Cały cykl (SLO + override + anotacje + paczka) można
> wykonać jednym poleceniem:
> ```bash
> python scripts/run_stage6_observability_cycle.py \
>     --definitions config/observability/slo.yml \
>     --metrics var/metrics/stage6_measurements.json \
>     --slo-json var/audit/observability/slo_report.json \
>     --slo-csv var/audit/observability/slo_report.csv \
>     --overrides-json var/audit/observability/alert_overrides.json \
>     --dashboard deploy/grafana/provisioning/dashboards/stage6_resilience_operations.json \
>     --annotations-output var/audit/observability/dashboard_annotations.json \
>     --bundle-output-dir var/observability --signing-key-path secrets/hmac/observability.key \
>     --tag stage6 --annotations-panel-id 1
> ```
> Parametry można dostosować do lokalnej struktury. Poniższe kroki opisują wersję
> manualną krok po kroku.

## Procedura
1. Uruchom monitor SLO i wygeneruj raport ze statusami Stage6 (wraz z
   kompozytami SLO2):
   ```bash
   python scripts/slo_monitor.py --definitions config/observability/slo.yml \
       --metrics var/metrics/stage6_measurements.json \
       --output var/audit/observability/slo_report.json \
       --output-csv var/audit/observability/slo_report.csv \
       --signature var/audit/observability/slo_report.sig \
       --signing-key-path secrets/hmac/observability.key
   ```
2. Na podstawie raportu SLO wygeneruj override'y alertów Stage6 wraz z podpisem HMAC:
   ```bash
   python scripts/generate_alert_overrides.py --slo-report var/audit/observability/slo_report.json \
       --output var/audit/observability/alert_overrides.json \
       --signature var/audit/observability/alert_overrides.sig \
       --signing-key-path secrets/hmac/observability.key \
       --requested-by NOC --tag stage6
   ```
3. Wygeneruj anotacje override'ów dla dashboardu Grafana oraz podpis HMAC:
   ```bash
   python scripts/sync_alert_overrides_dashboard.py \
       --overrides var/audit/observability/alert_overrides.json \
       --dashboard deploy/grafana/provisioning/dashboards/stage6_resilience_operations.json \
       --output var/audit/observability/dashboard_annotations.json \
       --signature var/audit/observability/dashboard_annotations.sig \
       --signing-key-path secrets/hmac/observability.key \
       --panel-id 1
   ```
4. Zbuduj paczkę obserwowalności z metadanymi override'ów i podpisem manifestu:
   ```bash
   python scripts/export_observability_bundle.py \
       --output-dir var/observability --bundle-name stage6-observability \
       --overrides var/audit/observability/alert_overrides.json \
       --metadata slo_report=var/audit/observability/slo_report.json \
       --metadata dashboard_annotations=var/audit/observability/dashboard_annotations.json \
       --hmac-key-file secrets/hmac/observability.key --hmac-key-id ops-stage6
   ```
5. Zwaliduj utworzoną paczkę obserwowalności wraz z podpisem manifestu:
   ```bash
   python scripts/verify_observability_bundle.py \
       var/observability/stage6-observability-YYYYMMDDThhmmssZ.zip \
       --hmac-key-file secrets/hmac/observability.key
   ```
   Uaktualnij ścieżkę do archiwum zgodnie z wygenerowaną nazwą paczki. Raport
   CLI powinien potwierdzić liczbę zweryfikowanych plików oraz brak błędów.
6. Zweryfikuj sekcję `summary.composites` oraz `composites.results` w raporcie,
   aby potwierdzić statusy SLO2 dla krytycznych domen (np. `core_stack`).
7. Przekaż statusy SLO do PortfolioGovernora i sprawdź ewentualne `slo_overrides`
   w wynikach `var/audit/portfolio/decision.json` (lub odpowiadającego raportu).
   Scheduler runtime odczyta raport z lokalizacji wskazanej w
   `runtime.multi_strategy_schedulers.*.portfolio_inputs`; upewnij się, że plik
   jest świeższy niż limit w konfiguracji.
8. Zweryfikuj, że dashboard Stage6 i alerty Prometheus są aktualne i zsynchronizuj
   bundel w repozytorium operacyjnym.
9. Zapisz wynik monitoringu w decision logu Stage6 wraz z referencją do raportów.
10. (Opcjonalnie) Połącz Observability z pozostałymi modułami Stage6 poprzez
    `python scripts/run_stage6_hypercare_cycle.py` – użyj checklisty Stage6 Hypercare, aby
    przygotować wspólną konfigurację i wygenerować podpisany raport zbiorczy.

## Artefakty/Akceptacja
- `var/observability/bundle.zip` wraz z podpisem `.sig` i metadanymi `alert_overrides`.
- Log JSON z `python scripts/verify_observability_bundle.py` potwierdzający weryfikację paczki.
- `var/audit/observability/slo_report.json` oraz podpis `.sig` (wraz z sekcją SLO2).
- `var/audit/observability/slo_report.csv` z wynikami SLO i kompozytów.
- `var/audit/observability/alert_overrides.json` oraz podpis `.sig`.
- `var/audit/observability/dashboard_annotations.json` wraz z podpisem `.sig`.
- Potwierdzenie zastosowania `slo_overrides` w logu decyzji portfelowych.
- Aktualny wpis w decision logu z odnośnikiem do raportów SLO.
