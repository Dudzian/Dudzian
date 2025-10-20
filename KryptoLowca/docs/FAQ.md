# FAQ – najczęstsze pytania

## Czy mogę uruchomić bota na koncie live?
Nie, dopóki nie spełnisz warunków compliance: konfiguracja kluczy API, potwierdzenie KYC/AML i akceptacja ryzyka. W nowej konfiguracji `bot_core.config.CoreConfig` oraz metadane runtime (`bot_core.runtime.metadata.load_runtime_entrypoint_metadata`) blokują tryb LIVE dla profili niespełniających wymogów.

## Jak monitorować działanie bota?
Użyj zestawu Prometheus + Grafana + Loki (dostarczonego w `deploy/docker-compose.yml`). Dashboard `KryptoLowca – Overview` pokazuje liczbę zleceń, frakcję ryzyka i logi.

## Co zrobić w razie awarii giełdy?
- Alerty (Slack/email/webhook) poinformują o błędach API.
- Guardrails mogą przejść w tryb `reduce-only`, blokując nowe pozycje.
- Po przywróceniu połączenia sprawdź logi w Grafanie i `risk_audit_logs`.

## Jak często rotować klucze API?
Domyślnie co 30 dni. `bot_core.security.RotationRegistry` wraz z `SecretManager`em zapisuje metadane rotacji; uruchom zadanie cron/CI wywołujące `RotationRegistry.mark_rotated()` po każdej zmianie klucza.

## Czy mogę dodać własne metryki?
Tak, użyj `KryptoLowca.telemetry.prometheus_exporter.metrics`. Metody `observe_risk`, `record_order`, `record_trade_close` oraz `set_open_positions` przykładowo pokazują, jak aktualizować liczniki.

## Jak wykonać testy obciążeniowe?
- Uruchom środowisko dockerowe.
- Przygotuj skrypt generujący sygnały (planowany `scripts/load_test_signals.py`).
- Obserwuj opóźnienia API oraz kolejkę logów w Grafanie.

## Gdzie zgłaszać błędy?
Utwórz issue w repozytorium lub skontaktuj się mailowo. Do zgłoszenia dołącz logi z Loki oraz export z Prometheusa.
