# Harmonogram retrainingu z chaos engineering

Dokument opisuje sposób konfigurowania oraz monitorowania modułu `core.runtime.retraining_scheduler.RetrainingScheduler` odpowiedzialnego za planowanie cykli ponownego treningu modeli AI.

## Podstawy działania

Scheduler utrzymuje ostatni oraz kolejny termin retrainingu na podstawie skonfigurowanego interwału. Przed wywołaniem właściwego pipeline’u treningowego może on zasymulować scenariusze chaosowe, które pomagają przetestować odporność systemu:

- **Brak danych (missing data)** – symuluje sytuację, w której wejściowe batch’e danych są niedostępne i przerywa cykl retrainingu.
- **Dryf danych (data drift)** – generuje zdarzenie z wynikiem `drift_score`, jednocześnie kontynuując trening, aby sprawdzić reakcję modułów walidacyjnych.
- **Opóźnienia (delay)** – wstrzymuje start retrainingu o losowy czas z zadanego przedziału, raportując zdarzenie dla guardrail’i.

## Konfiguracja

Parametry znajdują się w pliku [`config/runtime_retraining.yml`](../../config/runtime_retraining.yml). Sekcja `chaos` umożliwia włączenie konkretnych scenariuszy oraz ustawienie ich częstotliwości i intensywności.

```yaml
interval_minutes: 180
chaos:
  enabled: true
  missing_data_frequency: 0.05   # prawdopodobieństwo pominięcia cyklu z powodu braku danych
  missing_data_intensity: 2       # liczba brakujących batchy raportowana w zdarzeniu
  drift_frequency: 0.1            # prawdopodobieństwo wygenerowania zdarzenia dryfu
  drift_threshold: 0.35           # oczekiwany próg dryfu używany do raportowania
  delay_frequency: 0.1            # prawdopodobieństwo wstrzymania cyklu
  delay_min_seconds: 2.0
  delay_max_seconds: 8.0
```

Częstotliwości są interpretowane jako prawdopodobieństwo (0-1) na każdy cykl. Intensywność braku danych określa liczbę brakujących porcji raportowaną w logach.

## Emisja zdarzeń monitorujących

W przypadku wystąpienia scenariusza chaosowego scheduler publikuje zdarzenia z pakietu `core.monitoring.events` (`MissingDataDetected`, `DataDriftDetected`, `RetrainingDelayInjected`). Zdarzenia mogą być kierowane do guardrail’i, interfejsu użytkownika lub systemów alertowania za pomocą przekazanego `event_publisher`.

## Testy i walidacja

Jednostkowe scenariusze chaosowe znajdują się w [`tests/runtime/test_retraining_scheduler.py`](../../tests/runtime/test_retraining_scheduler.py). Testy pokrywają wszystkie trzy zdarzenia chaosowe oraz weryfikują, że scheduler poprawnie pomija lub kontynuuje cykl retrainingu.

W ramach procesów QA rekomendowane jest również uruchamianie scenariusza E2E retrainingu (planowanego w Sprint 2) z włączonymi scenariuszami chaosu, aby ocenić reakcję całego pipeline’u.
