# Testy obciążeniowe adapterów giełdowych

## Instrumentacja limitów i retry
- Limiter żądań publikuje teraz metryki `exchange_rate_limiter_wait_seconds` oraz `exchange_rate_limiter_wait_total`, a każdy przypadek oczekiwania jest logowany z poziomu modułu `bot_core.exchanges.rate_limiter`.
- Watchdog w `bot_core.exchanges.health` raportuje próby ponowień z ostrzeżeniem i przekazuje metadane do monitora limitów.
- Moduł `bot_core.monitoring.exchange_limits` agreguje zdarzenia, wylicza serie przekroczeń i wysyła alerty (`AlertSeverity.WARNING` przy wielokrotnym oczekiwaniu na limiter, `AlertSeverity.ERROR` przy skrajnym retry).

## Metryki i alerty
- `exchange_rate_limit_monitor_events_total` – licznik wszystkich zdarzeń oczekiwania na limiter.
- `exchange_retry_monitor_events_total` – licznik zdarzeń retry watchdog-a.
- `exchange_rate_limit_alerts_total` / `exchange_retry_alerts_total` – liczba alertów wysłanych przez monitor.
- Alerty mają źródło `exchange.limit-monitor` i trafiają do standardowego dispatcher-a, dlatego są widoczne w logach offline oraz w UI, jeśli skonfigurowano kanały powiadomień.

## Skrypt `scripts/exchange_load_test.py`
- Uruchomienie: `python -m scripts.exchange_load_test BTC/USDT --exchange binance --mode spot --operation ticker --concurrency 8 --duration 120`.
- Dostępne operacje: `ticker`, `ohlcv`, `order_book`.
- Obsługuje tryby `paper`, `spot`, `margin`, `futures`, w tym `--testnet` dla środowisk sandboxowych.
- Można przekazać dane API (`--api-key`, `--secret`, `--passphrase`) lub korzystać z symulatora papierowego (`--paper-variant`).
- Na zakończenie wypisywany jest wolumen żądań, liczba błędów, średnie opóźnienie oraz percentyl 95.

## Zalecany przebieg testów
1. W środowisku papierowym uruchom test z małą równoległością, aby zweryfikować poprawność konfiguracji.
2. Stopniowo zwiększaj `--concurrency`, obserwując w logach komunikaty o oczekiwaniu na limiter i alerty monitora.
3. W razie konieczności modyfikuj reguły w `bot_core/exchanges/rate_limiter.py` lub konfigurację adapterów, aby utrzymać oczekiwane SLA.
4. Raport z testu (logi + metryki Prometheus) dołącz do katalogu `audit/`, co umożliwi przegląd w kolejnych sprintach.
