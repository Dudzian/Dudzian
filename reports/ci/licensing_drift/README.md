# Raport kompatybilności HWID/licencji

Artefakt `compatibility.json` generowany przez `scripts/generate_hwid_drift_report.py` opisuje tolerancję dryfu (MAC/CPU/dysk/TPM) zgodnie z dokumentacją bezpieczeństwa.

Raport jest publikowany w nightly CI (`deploy/ci/github_actions_security_nightly.yml`) jako artefakt `licensing-drift-<run_id>` i powinien być dołączany do przeglądów bezpieczeństwa lub zgłoszeń rebind/appeal.

Job `Licensing drift consolidation` w workflow `.github/workflows/licensing-drift.yml` pobiera artefakty testów,
uruchamia `scripts/aggregate_licensing_drift_reports.py` (również wtedy, gdy job testowy zakończył się niepowodzeniem)
i publikuje zagregowane pliki. W przypadku braku lub uszkodzonego `compatibility.json` generowane jest puste
podsumowanie z informacją o problemie z artefaktem, aby nie blokować publikacji metryk:

- `reports/ci/licensing_drift/licensing_drift_summary.json`
- `reports/ci/licensing_drift/licensing_drift_summary.csv`
- kopie do `reports/ci/licensing_drift/dashboard/` wraz z plikiem `licensing_drift.prom` pod metryki Prometheus

Podsumowanie JSON zawiera pole `diagnostics` z listą komunikatów o brakujących lub uszkodzonych artefaktach (np. brak
`compatibility.json` albo logu pytest), co pozwala szybko zidentyfikować problemy po stronie CI bez zaglądania do logów.

Pliki z katalogu `dashboard/` mogą być serwowane statycznie lub scrapowane przez textfile collector,
aby zasilić panele Grafany `deploy/grafana/provisioning/dashboards/licensing_drift.json`.
