# FAQ – najczęstsze pytania

## Czy mogę uruchomić bota na koncie live?
Nie, dopóki nie spełnisz warunków compliance: konfiguracja kluczy API, potwierdzenie KYC/AML i akceptacja ryzyka. W kodzie `StrategyConfig` wymusza blokadę trybu LIVE, jeśli którykolwiek warunek nie jest spełniony.

## Jak monitorować działanie bota?
Użyj zestawu Prometheus + Grafana + Loki (dostarczonego w `deploy/docker-compose.yml`). Dashboard `KryptoLowca – Overview` pokazuje liczbę zleceń, frakcję ryzyka i logi.

## Co zrobić w razie awarii giełdy?
- Alerty (Slack/email/webhook) poinformują o błędach API.
- Guardrails mogą przejść w tryb `reduce-only`, blokując nowe pozycje.
- Po przywróceniu połączenia sprawdź logi w Grafanie i `risk_audit_logs`.

## Jak często rotować klucze API?
Domyślnie co 30 dni. `KeyRotationManager.ensure_rotation()` automatycznie zapisuje metadane i może być uruchamiany przez cron/CI.

## Czy mogę dodać własne metryki?
Tak, użyj `KryptoLowca.telemetry.prometheus_exporter.metrics`. Metody `observe_risk`, `record_order`, `record_trade_close` oraz `set_open_positions` przykładowo pokazują, jak aktualizować liczniki.

## Jak wykonać testy obciążeniowe?
- Uruchom środowisko dockerowe.
- Przygotuj skrypt generujący sygnały (planowany `scripts/load_test_signals.py`).
- Obserwuj opóźnienia API oraz kolejkę logów w Grafanie.

## Gdzie zgłaszać błędy?
Utwórz issue w repozytorium lub skontaktuj się mailowo. Do zgłoszenia dołącz logi z Loki oraz export z Prometheusa.
