# CloudOrchestrator – lista kontrolna DR

## Cel
Zapewnić, że Alertmanager/Prometheus oraz `_health`/`_lastError` CloudOrchestratora
są odtwarzalne w scenariuszu awarii control-plane lub przełączenia regionu.

## Kroki operacyjne
1. **Backup konfiguracji alertów**
   - Zrzut `deploy/prometheus/rules/cloud_worker_alerts.yml` oraz `alertmanager.yml`
     do `reports/ci/dr_alerts/` (wraz z adnotacjami owner/team/channel).
   - Archiwizuj sekrety receiverów (HyperCare/CloudAlertService) w sejfie kluczy
     i odnotuj datę ważności tokenów.
2. **Replikacja multi-region**
   - Przełącz scrape’y `bot_cloud_health_status`, `bot_cloud_worker_last_error`
     i `bot_cloud_last_error` na zapasowy Prometheus (sanity-check `/federate`).
   - Zaimportuj reguły Alertmanagera na węzeł DR i wyzwól próbny alert
     `CloudWorkerHealthDegraded` (status HTTP 2xx z receivera).
3. **Synthetic probes / rehydratacja**
   - Uruchom `scripts/dr_synthetic_probes.py` z flagami `--health-timeout 5`
     oraz `--latency-threshold-ms 5000`; wynik trafi do `reports/ci/dr_probes/`.
   - Zweryfikuj `healthOk=true`, `prometheusOk=true` oraz brak `_lastError`.
     Jeżeli proces był świeżo przełączony, oczekuj `rehydratedFromPrevious=true`
     oraz `effectiveLastError` zgodnego z poprzednim regionem. `failoverReady`
     musi być `true` dopiero, gdy Prometheus/Alertmanager są zdrowe
     (`prometheusOk=true`) **i** nie ma aktywnego `_lastError`; pominięcie
     `prometheusOk` traktuj jako blokadę failover readiness.
4. **Przywracanie po awarii**
   - Odtwórz konfigurację Prometheus/Alertmanagera z paczki DR i wykonaj restart
     usług monitoringu.
   - Zweryfikuj, że CloudOrchestrator publikuje `_health=running` i metryki są
     widoczne w `/metrics` zapasowego Prometheusa.
5. **Retrospektywa**
   - Zanotuj czas przełączenia, opóźnienia sond oraz wszelkie niezgodności w
     `reports/ci/dr_alerts/postmortem.md`.

## Walidacja zakończenia
- Wszystkie testowe alerty zostały dostarczone (HyperCare/CloudAlertService).
- Synthetic probes raportują `failoverReady=true` lub zarejestrowaną rehydratację
  poprzedniego `_lastError`.
- Dashboard DR w Grafanie pokazuje nowy region jako aktywny źródłowy scraper.
