# CloudAlertService – Checklist eskalacji HyperCare

## Cel
Zapewnić szybką reakcję na alerty HyperCare powiązane z metrykami
`bot_cloud_worker_status` oraz `bot_cloud_worker_last_error`, tak aby
zminimalizować czas niedostępności workerów orchestratora cloudowego.

## Sygnały i klasyfikacja
- **Degraded (`CloudWorkerHealthDegraded`)** – `bot_cloud_worker_status=0`
  utrzymuje się >5 minut; worker jest aktywny, ale `_health` sygnalizuje
  brak gotowości.
- **Critical (`CloudWorkerErrorCritical`)** – `bot_cloud_worker_last_error>0`
  utrzymuje się >10 minut; `_lastError` wskazuje konkretny kod błędu.

## Procedura L1 (monitoring/HyperCare)
1. Potwierdź alert w Alertmanagerze (kanał `HyperCare` lub
   `CloudAlertService`).
2. Odczytaj z adnotacji nazwę workera (`{{ $labels.worker }}`) i błąd
   (`{{ $labels.error }}` dla krytycznych zdarzeń).
3. Uruchom `HealthService.Check` dla `cloud_health` i zweryfikuj
   pola `_health`, `_lastError` oraz sekcję `workers` dla wskazanego
   workera.
4. Sprawdź logi orchestratora (`cloud_health`, `CloudOrchestrator`)
   pod kątem wpisów z `_health`/`_lastError` w czasie alertu.
5. W przypadku degradacji:
   - wymuś pojedyncze uruchomienie workera (np. retrain/marketplace),
     monitorując metryki `bot_cloud_worker_status`.
   - jeżeli status nie wraca do `1`, eskaluj do L2.
6. W przypadku błędu krytycznego:
   - zapisz kod błędu z `_lastError` i ostatnie zdarzenia z logów
     orchestratora;
   - wykonaj restart workera lub przełącz na tryb awaryjny, jeśli dostępny;
   - zgłoś do L2/L3 z załączonym kodem błędu i wycinkiem logów.

## Procedura L2 (ops/CloudAlertService)
1. Koreluj alert z ostatnimi wdrożeniami i zmianami konfiguracyjnymi
   (katalog `config/cloud/`, `deploy/prometheus/rules/cloud_worker_alerts.yml`).
2. Zweryfikuj częstotliwość błędów w `bot_cloud_worker_last_error` dla
   wszystkich workerów (`PromQL: max_over_time(... ) by (worker, error)`).
3. Ustal właściciela (`owner=cloud-alerts`) i wybierz właściwy kanał:
   - `HyperCare` dla degradacji,
   - `CloudAlertService` dla błędów krytycznych.
4. Jeśli błąd wynika z zależności zewnętrznych (np. API marketplace),
   skontaktuj się z odpowiednim SRE i dodaj komentarz do incydentu.

## Procedura L3 (engineering)
- Zbierz pełny snapshot `_health` i ostatnie logi orchestratora.
- Wdróż poprawkę lub obejście, aktualizując `bot_cloud_worker_status`
  oraz `bot_cloud_worker_last_error` zgodnie z wynikiem.
- Zaktualizuj runbook w przypadku nowych kodów błędów lub ścieżek
  naprawczych.

## Kryteria zamknięcia incydentu
- Metryki `bot_cloud_worker_status` wracają do `1` dla wszystkich
  workerów, a `bot_cloud_worker_last_error` wynosi `0` w oknie 15 min.
- Alert w Alertmanagerze jest zamknięty lub wyciszony z komentarzem.
- Notatka z przebiegu eskalacji dodana do dziennika incydentów.
